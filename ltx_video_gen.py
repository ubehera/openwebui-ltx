"""
title: LTX-2.3 Video Generation
author: ubehera
author_url: https://github.com/ubehera/openwebui-ltx
license: Apache-2.0
version: 0.3.0
description: Local LTX-2.3 video gen via ComfyUI. <=8s clips include synced audio. >8s clips chain LTXVExtendSampler segments (video-only — LTX's audio path doesn't extend). Up to 80s with quality warnings past 24s. Optional prompt enhancement via any OpenAI-compatible chat endpoint using LTX's official Creative Assistant system prompt.
required_open_webui_version: 0.5.0
"""

import asyncio
import json
import math
import os
import time
from typing import Awaitable, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field


# Defaults assume ComfyUI on the same host as Open WebUI. Point at your own host via valves.
COMFYUI_BASE_URL_DEFAULT = "http://127.0.0.1:8188"
# Prompt enhancement uses any OpenAI-compatible /chat/completions endpoint.
# Leave empty to disable enhancement entirely (or toggle per-call via enhance_prompt=False).
# Tested with vLLM, NIM, llama.cpp server, Ollama (--openai-compat).
ENHANCER_BASE_URL_DEFAULT = ""
ENHANCER_MODEL_DEFAULT = ""
ENHANCER_API_KEY_DEFAULT = ""

# LTX official T2V Creative Assistant system prompt
# Source: ComfyUI-LTXVideo/system_prompts/gemma_t2v_system_prompt.txt
LTX_T2V_SYSTEM_PROMPT = """\
You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.

#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone."""


def _build_workflow(
    prompt: str,
    width: int,
    height: int,
    length: int,
    fps: int,
    steps: int,
    seed: int,
    cfg: float,
    negative_prompt: str,
    ckpt: str,
    text_encoder: str,
) -> dict:
    return {
        "1": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": text_encoder, "ckpt_name": ckpt, "device": "default"},
        },
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["1", 0]}},
        "4": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["2", 0], "negative": ["3", 0], "frame_rate": float(fps)},
        },
        "5": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "6": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": ckpt}},
        "7": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": length, "batch_size": 1},
        },
        "9": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": length,
                "frame_rate": fps,
                "batch_size": 1,
                "audio_vae": ["6", 0],
            },
        },
        "10": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["7", 0], "audio_latent": ["9", 0]},
        },
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["10", 0],
            },
        },
        "13": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["5", 0], "positive": ["4", 0], "negative": ["4", 1], "cfg": cfg},
        },
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["14", 0],
                "guider": ["13", 0],
                "sampler": ["11", 0],
                "sigmas": ["12", 0],
                "latent_image": ["10", 0],
            },
        },
        "16": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}},
        "17": {
            "class_type": "LTXVTiledVAEDecode",
            "inputs": {
                "vae": ["5", 2],
                "latents": ["16", 0],
                "horizontal_tiles": 2,
                "vertical_tiles": 2,
                "overlap": 6,
                "last_frame_fix": True,
            },
        },
        "18": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": ["16", 1], "audio_vae": ["6", 0]},
        },
        "19": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["17", 0], "fps": float(fps), "audio": ["18", 0]},
        },
        "20": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["19", 0],
                "filename_prefix": "ltx2.3",
                "format": "auto",
                "codec": "auto",
            },
        },
    }


def _build_extend_workflow(
    prompt: str,
    seconds: float,
    width: int,
    height: int,
    fps: int,
    seed: int,
    steps: int,
    negative_prompt: str,
    ckpt: str,
    text_encoder: str,
    seg_max_seconds: float = 8.0,
    frame_overlap: int = 16,
) -> tuple[dict, int, int]:
    """Video-only chained workflow: LTXVBaseSampler + N x LTXVExtendSampler.
    Returns (workflow, actual_total_frames, n_extends)."""

    def to_frames(sec: float) -> int:
        n = max(8, int(round(sec * fps / 8) * 8))
        return n + 1   # frames must be 1+8k

    base_frames = to_frames(seg_max_seconds)
    target_total = max(base_frames, to_frames(seconds))
    remaining = target_total - base_frames
    extend_chunk = to_frames(seg_max_seconds) - 1   # ~192 frames per ext for 8s
    n_extends = math.ceil(remaining / extend_chunk) if remaining > 0 else 0
    new_frames_per_ext = (math.ceil(remaining / n_extends) // 8) * 8 if n_extends > 0 else 0
    actual_total = base_frames + n_extends * new_frames_per_ext

    wf: dict = {
        "text_enc": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": text_encoder, "ckpt_name": ckpt, "device": "default"},
        },
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["text_enc", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["text_enc", 0]}},
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "stg_preset": {"class_type": "STGAdvancedPresets", "inputs": {"preset": "13b Distilled"}},
        # STGGuiderAdvanced: required strings supply preset defaults; the linked preset overrides at runtime.
        "guider": {
            "class_type": "STGGuiderAdvanced",
            "inputs": {
                "model": ["ckpt", 0],
                "positive": ["pos", 0],
                "negative": ["neg", 0],
                "skip_steps_sigma_threshold": 0.997,
                "cfg_star_rescale": True,
                "sigmas": "1.0",
                "cfg_values": "1",
                "stg_scale_values": "0",
                "stg_rescale_values": "1",
                "stg_layers_indices": "25",
                "preset": ["stg_preset", 0],
            },
        },
        "sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "empty_for_sigmas": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": base_frames, "batch_size": 1},
        },
        "sched_base": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": ["empty_for_sigmas", 0],
            },
        },
        "noise_base": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "base_sampler": {
            "class_type": "LTXVBaseSampler",
            "inputs": {
                "model": ["ckpt", 0],
                "vae": ["ckpt", 2],
                "width": width,
                "height": height,
                "num_frames": base_frames,
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": ["sched_base", 0],
                "noise": ["noise_base", 0],
            },
        },
    }

    last_node = "base_sampler"
    for i in range(n_extends):
        chunk_frames = ((frame_overlap + new_frames_per_ext - 1) // 8) * 8 + 1
        empty_id, sched_id, noise_id, ext_id = f"e{i}", f"s{i}", f"n{i}", f"x{i}"
        wf[empty_id] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": chunk_frames, "batch_size": 1},
        }
        wf[sched_id] = {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps, "max_shift": 2.05, "base_shift": 0.95,
                "stretch": True, "terminal": 0.1, "latent": [empty_id, 0],
            },
        }
        wf[noise_id] = {"class_type": "RandomNoise", "inputs": {"noise_seed": seed + 1 + i}}
        wf[ext_id] = {
            "class_type": "LTXVExtendSampler",
            "inputs": {
                "model": ["ckpt", 0],
                "vae": ["ckpt", 2],
                "latents": [last_node, 0],
                "num_new_frames": new_frames_per_ext,
                "frame_overlap": frame_overlap,
                "guider": ["guider", 0],
                "sampler": ["sampler", 0],
                "sigmas": [sched_id, 0],
                "noise": [noise_id, 0],
                "strength": 0.5,
            },
        }
        last_node = ext_id

    wf["decode"] = {
        "class_type": "LTXVTiledVAEDecode",
        "inputs": {
            "vae": ["ckpt", 2], "latents": [last_node, 0],
            "horizontal_tiles": 2, "vertical_tiles": 2, "overlap": 6, "last_frame_fix": True,
        },
    }
    wf["create_video"] = {
        "class_type": "CreateVideo",
        "inputs": {"images": ["decode", 0], "fps": float(fps)},
    }
    wf["save"] = {
        "class_type": "SaveVideo",
        "inputs": {
            "video": ["create_video", 0],
            "filename_prefix": "ltx2.3_long",
            "format": "auto",
            "codec": "auto",
        },
    }
    return wf, actual_total, n_extends


class Tools:
    class Valves(BaseModel):
        comfyui_base_url: str = Field(
            default=COMFYUI_BASE_URL_DEFAULT,
            description="Base URL of the ComfyUI server hosting LTX-2.3.",
        )
        ckpt_name: str = Field(
            default="ltx-2.3-22b-distilled-fp8.safetensors",
            description="LTX checkpoint filename in ComfyUI's checkpoints folder.",
        )
        text_encoder: str = Field(
            default="gemma_3_12B_it_fp4_mixed.safetensors",
            description="Gemma 3 12B text encoder filename in text_encoders folder.",
        )
        max_wait_seconds: int = Field(
            default=900,
            description="Max seconds to wait for video generation before timing out.",
        )
        enhancer_base_url: str = Field(
            default=ENHANCER_BASE_URL_DEFAULT,
            description="OpenAI-compatible chat endpoint (e.g. http://host:8001/v1) for prompt enhancement. Empty = enhancement disabled.",
        )
        enhancer_model: str = Field(
            default=ENHANCER_MODEL_DEFAULT,
            description="Model id for prompt enhancement (must be served by enhancer_base_url).",
        )
        enhancer_api_key: str = Field(
            default=ENHANCER_API_KEY_DEFAULT,
            description="API key for the enhancer endpoint (leave empty for no-auth servers).",
        )
        enhancer_max_tokens: int = Field(
            default=512,
            description="Max tokens for the enhanced prompt output.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _enhance_prompt(self, raw_prompt: str) -> str:
        """Run the user prompt through the LTX Creative Assistant system prompt
        via any OpenAI-compatible chat endpoint. Returns the enhanced text, or the
        original prompt on any failure or when no endpoint is configured."""
        if not self.valves.enhancer_base_url or not self.valves.enhancer_model:
            return raw_prompt
        url = self.valves.enhancer_base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.valves.enhancer_api_key:
            headers["Authorization"] = f"Bearer {self.valves.enhancer_api_key}"
        body = {
            "model": self.valves.enhancer_model,
            "messages": [
                {"role": "system", "content": LTX_T2V_SYSTEM_PROMPT},
                {"role": "user", "content": raw_prompt},
            ],
            "temperature": 0.5,
            "max_tokens": self.valves.enhancer_max_tokens,
        }
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(url, json=body, headers=headers) as r:
                if r.status != 200:
                    text = await r.text()
                    raise RuntimeError(f"enhancer {r.status}: {text[:200]}")
                data = await r.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        content = content.strip()
        # Strip leading/trailing fences or quotes the model might add despite the system prompt.
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
            content = content.strip()
        return content or raw_prompt

    async def generate_video(
        self,
        prompt: str,
        seconds: float = 5.0,
        width: int = 960,
        height: int = 544,
        fps: int = 24,
        seed: int = 42,
        steps: int = 8,
        cfg: float = 1.0,
        negative_prompt: str = "",
        enhance_prompt: bool = True,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> str:
        """
        Generate a short video clip from a text prompt using LTX-2.3.

        Mode dispatches on duration:
          - <=8s: single-segment with synced audio (V+A workflow).
          - >8s: chained LTXVExtendSampler segments, video-only (LTX has no audio extender).

        :param prompt: Text describing the desired scene. Be descriptive about subject, action, lighting, camera.
        :param seconds: Target duration in seconds (1.0 - 80.0). Past 24s the model drifts noticeably.
        :param width: Frame width in pixels (multiple of 32, default 960).
        :param height: Frame height in pixels (multiple of 32, default 544).
        :param fps: Frames per second (default 24).
        :param seed: Random seed for reproducibility (default 42).
        :param steps: Diffusion steps. Distilled checkpoint runs cleanly at 6-12 steps (default 8).
        :param cfg: Classifier-free guidance scale. Distilled needs cfg=1.0; raise only with non-distilled ckpt.
        :param negative_prompt: Optional negative prompt (rarely needed for distilled model).
        :param enhance_prompt: If true and an enhancer endpoint is configured (see valves), run the prompt through it with LTX's official Creative Assistant system prompt for richer detail (~4-30s extra depending on model).
        :return: Markdown with an embedded video URL once generation completes.
        """
        seconds = max(1.0, min(80.0, float(seconds)))
        width = max(256, (int(width) // 32) * 32)
        height = max(256, (int(height) // 32) * 32)
        long_mode = seconds > 8.0

        async def emit(msg: str, done: bool = False):
            if __event_emitter__ is None:
                return
            await __event_emitter__(
                {"type": "status", "data": {"description": msg, "done": done}}
            )

        warning = ""
        if long_mode:
            warning_lines = [
                f"Long-clip mode ({seconds:.0f}s): LTX has no audio extender, output will be video-only."
            ]
            if seconds > 24:
                warning_lines.append(
                    f"At {seconds:.0f}s, expect identity/scene drift across segments — distilled LTX is trained on shorter clips."
                )
            warning = " ".join(warning_lines)
            await emit(warning)

        raw_prompt = prompt
        enhanced_prompt = None
        if enhance_prompt and self.valves.enhancer_base_url and self.valves.enhancer_model:
            await emit(f"Enhancing prompt with {self.valves.enhancer_model}...")
            t0 = time.time()
            try:
                enhanced_prompt = await self._enhance_prompt(raw_prompt)
                if enhanced_prompt and enhanced_prompt != raw_prompt:
                    prompt = enhanced_prompt
                    await emit(f"Enhanced in {int(time.time()-t0)}s, submitting LTX-2.3...")
                else:
                    enhanced_prompt = None
            except Exception as e:
                await emit(f"Enhancer failed ({e}); using raw prompt.")
                enhanced_prompt = None

        if long_mode:
            wf, actual_frames, n_extends = _build_extend_workflow(
                prompt=prompt,
                seconds=seconds,
                width=width,
                height=height,
                fps=fps,
                seed=seed,
                steps=steps,
                negative_prompt=negative_prompt,
                ckpt=self.valves.ckpt_name,
                text_encoder=self.valves.text_encoder,
            )
            length = actual_frames
            mode_label = f"chained ({n_extends + 1} segments, video-only)"
        else:
            length = max(9, int(round(seconds * fps / 8)) * 8 + 1)
            wf = _build_workflow(
                prompt=prompt,
                width=width,
                height=height,
                length=length,
                fps=fps,
                steps=steps,
                seed=seed,
                cfg=cfg,
                negative_prompt=negative_prompt,
                ckpt=self.valves.ckpt_name,
                text_encoder=self.valves.text_encoder,
            )
            mode_label = "AV (with audio)"

        base = self.valves.comfyui_base_url.rstrip("/")
        await emit(f"Submitting LTX-2.3 [{mode_label}] {width}x{height}, {length} frames @ {fps}fps, {steps} steps...")

        async with aiohttp.ClientSession() as sess:
            async with sess.post(f"{base}/prompt", json={"prompt": wf}) as r:
                if r.status != 200:
                    body = await r.text()
                    return f"ComfyUI rejected workflow ({r.status}): {body[:500]}"
                data = await r.json()
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                return f"No prompt_id in response: {data}"

            await emit(f"Queued as {prompt_id[:8]}, waiting for generation...")

            start = time.time()
            poll_interval = 2.0
            last_status = ""
            while True:
                if time.time() - start > self.valves.max_wait_seconds:
                    return f"Timed out after {self.valves.max_wait_seconds}s waiting for {prompt_id}"
                async with sess.get(f"{base}/history/{prompt_id}") as r:
                    hist = await r.json()
                if hist:
                    break
                async with sess.get(f"{base}/queue") as r:
                    q = await r.json()
                running = len(q.get("queue_running", []))
                pending = len(q.get("queue_pending", []))
                status = f"running={running} pending={pending} elapsed={int(time.time()-start)}s"
                if status != last_status:
                    await emit(f"LTX-2.3 generating: {status}")
                    last_status = status
                await asyncio.sleep(poll_interval)

            entry = hist[prompt_id]
            outputs = entry.get("outputs", {})
            video_files = []
            for nid, out in outputs.items():
                for k, v in out.items():
                    if not isinstance(v, list):
                        continue
                    for item in v:
                        if isinstance(item, dict) and item.get("filename"):
                            video_files.append(item)

            if not video_files:
                return f"No output videos found in history for {prompt_id}: {json.dumps(outputs)[:500]}"

            f0 = video_files[0]
            fname = f0["filename"]
            subfolder = f0.get("subfolder", "")
            ftype = f0.get("type", "output")
            url = f"{base}/view?filename={fname}&subfolder={subfolder}&type={ftype}"
            elapsed = int(time.time() - start)
            await emit(f"Video ready in {elapsed}s.", done=True)

            # Open WebUI HTMLToken matches /<video[^>]*>(.*?)<\/video>/ within a block-level html token
            # and pulls the URL from inner text. Marked needs <details> (a known block tag) to keep
            # the inner <video>...</video> as raw HTML rather than auto-linking the URL.
            enhanced_block = ""
            if enhanced_prompt and enhanced_prompt != raw_prompt:
                # Plain markdown — a nested <details> has blank lines inside which break
                # marked's HTML-block parsing and end up showing escaped tags.
                enhanced_block = f"**Enhanced prompt:** _{enhanced_prompt}_\n\n"
            warning_block = f"_{warning}_\n\n" if warning else ""
            duration = length / fps
            video_md = (
                f"<details open><summary>video</summary>\n"
                f"<video>{url}</video>\n"
                f"</details>\n\n"
                f"{warning_block}"
                f"{enhanced_block}"
                f"[Download MP4]({url}) - {width}x{height}, {length} frames @ {fps}fps "
                f"({duration:.1f}s), seed={seed}, mode={mode_label}, generated in {elapsed}s\n"
            )

            if __event_emitter__ is not None:
                await __event_emitter__(
                    {"type": "message", "data": {"content": video_md}}
                )

            return (
                f"Video generated successfully and rendered above. "
                f"File: {fname} ({width}x{height}, {length} frames, {elapsed}s). "
                f"Do not repeat the video URL or markdown — it's already shown."
            )
