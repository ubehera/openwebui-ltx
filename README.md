# openwebui-ltx

A single-file [Open WebUI](https://github.com/open-webui/open-webui) tool that wraps a local
[LTX-2.3](https://github.com/Lightricks/LTX-Video) video-generation pipeline running on ComfyUI,
and renders the resulting MP4 inline in the chat.

Two paths in one tool:

| Duration  | Workflow                                  | Audio |
| :-------- | :---------------------------------------- | :---: |
| `<= 8 s`  | single-segment AV (video + synced audio)  |  yes  |
| `> 8 s`   | `LTXVBaseSampler + N x LTXVExtendSampler` |  no   |

Hard cap is 80 s. Past 24 s the tool emits an extra warning about identity / scene drift.

Optional prompt enhancement runs the user prompt through any OpenAI-compatible chat
endpoint with LTX's official "Creative Assistant" system prompt before encoding.

## Why this exists

The example workflows shipped with `ComfyUI-LTXVideo/example_workflows/2.3/` route prompts
through a paid Lightricks cloud endpoint (`api.ltx.video`) by default. The shipped graphs
also stack RES4LYF samplers, multimodal guiders, dual-stage decode chains, and custom-node
deps that aren't installed out of the box.

This tool strips all that. It builds a minimal API-format workflow from scratch using only
nodes that ship with mainline ComfyUI plus `ComfyUI-LTXVideo`, points at local model files,
and exposes it as a typed function the chat model can call.

## Prerequisites

1. **ComfyUI** with the [`ComfyUI-LTXVideo`](https://github.com/Lightricks/ComfyUI-LTXVideo)
   custom nodes installed and reachable on an HTTP port. Tested against a recent ComfyUI
   build with the LTX-2.3 nodes.
2. **LTX-2.3 distilled fp8 checkpoint** at `ComfyUI/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors`.
   The non-distilled `dev` variant also works with `cfg > 1`, but defaults assume distilled.
3. **A Gemma 3 12B text encoder** at `ComfyUI/models/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors`.
   Pull from [`Comfy-Org/ltx-2`](https://huggingface.co/Comfy-Org/ltx-2) — the
   `gemma_3_12B_it_fp4_mixed.safetensors` file (~9.5 GB) is the lightest option that works.
   The `_fp8_scaled` (~13 GB) and full bf16 (~24 GB) variants in the same repo also work.
4. **Open WebUI 0.5+**.

If your LTX-2.3 weights live in `models/diffusion_models/` instead of `models/checkpoints/`
(default for some installs), symlink them: `LTXVBaseSampler` and `LTXVExtendSampler`'s
`CheckpointLoaderSimple` looks in `checkpoints/` only.

```sh
ln -s "$PWD/models/diffusion_models/ltx-2.3-22b-distilled-fp8.safetensors" \
      "$PWD/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
```

## Install

In Open WebUI: **Workspace -> Tools -> Import** and paste the contents of `ltx_video_gen.py`.
Or use the API:

```sh
curl -X POST http://127.0.0.1:8080/api/v1/tools/create \
  -H "Authorization: Bearer <admin_jwt>" \
  -H "Content-Type: application/json" \
  -d "$(jq -Rs '{id:"ltx_video_gen", name:"LTX-2.3 Video Generation", content:., meta:{description:"", manifest:{}}}' < ltx_video_gen.py)"
```

After install, configure valves under **Workspace -> Tools -> LTX-2.3 Video Generation -> Valves**:

| Valve                | Default                | What                                                            |
| :------------------- | :--------------------- | :-------------------------------------------------------------- |
| `comfyui_base_url`   | `http://127.0.0.1:8188` | ComfyUI HTTP host                                              |
| `ckpt_name`          | `ltx-2.3-22b-distilled-fp8.safetensors` | filename in `models/checkpoints/`             |
| `text_encoder`       | `gemma_3_12B_it_fp4_mixed.safetensors`  | filename in `models/text_encoders/`           |
| `max_wait_seconds`   | `900`                  | gen-time hard timeout                                           |
| `enhance_with`       | `current`              | `current` = chat-dropdown model, `fast` = small model, `off`, or any explicit model id |
| `fast_enhancer_model`| _(empty)_              | model id used when `enhance_with=fast`                          |
| `openwebui_base_url` | _(empty)_              | OWUI host (e.g. `http://127.0.0.1:8080`); enables routing through OWUI's `/openai/chat/completions` proxy so the enhancer inherits whatever connection the dropdown model uses |
| `openwebui_token`    | _(empty)_              | OWUI Bearer token for the proxy call. Required for `enhance_with=current`. |
| `enhancer_base_url`  | _(empty)_              | _Fallback._ Direct OpenAI-compat chat endpoint, used when `openwebui_token` is empty. |
| `enhancer_model`     | _(empty)_              | _Fallback._ Model id for the direct enhancer endpoint           |
| `enhancer_api_key`   | _(empty)_              | _Fallback._ Bearer token for the direct endpoint                |

Then enable the tool in any chat (input bar -> integrations menu -> Tools -> LTX-2.3 Video Generation toggle).

### Configuration recipes

**Use the model selected in the chat dropdown** (recommended; respects user choice, single source of routing truth):

1. Set `openwebui_base_url` to where Open WebUI runs (e.g. `http://127.0.0.1:8080`).
2. Set `openwebui_token` to an admin/service JWT.
3. Leave `enhance_with` at default `"current"`.

The chat model the user picks for the conversation is also what rewrites the prompt. Quality scales with the model — but so does latency (a 120B reasoning model adds 20-40s per call vs ~5s for a 30B).

**Use a fixed fast model** (recommended when video gen is the only thing you care about and the chat model is heavyweight):

1. Configure `openwebui_base_url` + `openwebui_token` as above.
2. Set `fast_enhancer_model` to a small model id known to OWUI (e.g. a 7B or 30B chat model).
3. Set `enhance_with` to `"fast"`.

**Disable enhancement entirely**:

Set `enhance_with` to `"off"`. Or pass `enhance_prompt=False` in the tool call.

**Direct endpoint (no Open WebUI proxy)**:

If you'd rather not pass an OWUI token to the tool, leave `openwebui_token` empty and use the legacy `enhancer_base_url` / `enhancer_model` / `enhancer_api_key` valves to point at any OpenAI-compatible chat endpoint (vLLM, NIM, llama.cpp, Ollama with `--openai-compat`).

## Usage

Ask the chat model to call the tool. Any tool-calling model works — the
function signature is exposed via the docstring schema:

```
generate_video(prompt: str, seconds: float = 5.0, width: int = 960, height: int = 544,
               fps: int = 24, seed: int = 42, steps: int = 8, cfg: float = 1.0,
               negative_prompt: str = "", enhance_prompt: bool = True,
               enhance_with: str | None = None)
```

Per-call `enhance_with` override lets the chat model (or the user) flip enhancers
mid-conversation: `enhance_with="fast"` for quick iteration, `enhance_with="current"`
when the dropdown model is the right tool, `enhance_with="off"` to skip.

Example chat input:

> generate a 4 second clip of a snow leopard slowly padding through fresh powder at golden hour

The model will issue a tool call, the tool will (optionally) enhance the prompt, submit
to ComfyUI, poll until done, fetch the `/view` URL, and inject the video markdown
back into the assistant message.

## Performance

Numbers from a single-GPU box with both LTX checkpoint and Gemma encoder warm in memory:

| Duration | Mode                | Gen time   | Audio |
| :------- | :------------------ | :--------- | :---: |
| 4 s      | AV single           | ~30-50 s   |  yes  |
| 8 s      | AV single           | ~50-70 s   |  yes  |
| 12 s     | 1 extension         | ~80-100 s  |  no   |
| 24 s     | 3 extensions        | ~3 min     |  no   |
| 80 s     | 9-10 extensions     | ~8-10 min  |  no   |

Cold start (first call after ComfyUI boot) adds ~3-4 min for model staging.

## Notes from building this — non-obvious findings

These cost real iteration time. Worth knowing if you're building anything similar.

### 1. Use the `13b Distilled` STG preset to satisfy `LTXVExtendSampler` with cfg=1

`LTXVBaseSampler` and `LTXVExtendSampler` both require an `STGGuiderAdvanced`
(not the plain `CFGGuider`). STG (spatio-temporal guidance) normally needs
`cfg > 1` to work, which is incompatible with the distilled checkpoint's required
`cfg = 1`.

The `STGAdvancedPresets` node has a `13b Distilled` preset whose values are
`cfg_values=[1], stg_scale_values=[0]` — STG is functionally a no-op, behaves
identically to `CFGGuider` with `cfg=1`, but the guider has the right type to
satisfy the extension samplers. **No need to switch to dev+LoRA, no need to
install RES4LYF, no need to wrap the model in `LTXVApplySTG`.**

`STGGuiderAdvanced` still requires its 5 string inputs (`sigmas`, `cfg_values`,
`stg_scale_values`, `stg_rescale_values`, `stg_layers_indices`) for ComfyUI
schema validation even when a preset is linked — supply the preset's defaults
inline so validation passes; the linked preset overrides at runtime.

### 2. LTX has no audio-latent extender

`LTXVExtendSampler.sample()` allocates new latents via `EmptyLTXVLatentVideo`
(5D `(B, C, F, H, W)` — pure video). The `LTXVConcatAVLatent +
LTXVAudioVAELoader` path used in the short-clip workflow doesn't carry through.
This tool dispatches on duration: `<= 8 s` keeps audio, `> 8 s` is video-only.

### 3. The LTX-2.3 example workflows have vestigial paid-API nodes

Every `example_workflows/2.3/*.json` file references `GemmaAPITextEncode`,
which posts the prompt to `https://api.ltx.video` and needs an API key from
`console.ltx.video`. **In `LTX-2.3_T2V_I2V_Single_Stage_Distilled_Full.json`
the cloud nodes have no output edges to downstream nodes** — they're vestigial.
The active conditioning path is fully local via `LTXAVTextEncoderLoader ->
CLIPTextEncode -> LTXVConditioning`. Don't be scared off the workflows by
the API-key requirement; just delete those nodes.

### 4. Open WebUI strips raw `<video src="...">` tags

Three independent layers each break the obvious approach:

- **Open WebUI's `Image.svelte` always renders `<img>`**, regardless of
  whether the URL ends in `.mp4`. So `![alt](video.mp4)` shows a broken-image
  icon, not a video player.
- **Open WebUI's `HTMLToken.svelte` does match `<video>` content** — but it
  pulls the URL from the tag's *inner text*, not the `src` attribute:
  ```html
  <video>https://host/path/to/video.mp4</video>
  ```
- **Marked treats `<video>` as inline HTML** because `<video>` is not in
  CommonMark's known block-tag list (Type 6). Inline HTML gets escaped, so
  the URL inside also gets auto-linked, and the whole thing renders as text
  instead of producing an `html` token.

The pattern that works is wrapping in a known block-level tag:

```markdown
<details open><summary>video</summary>
<video>https://host/path/to/video.mp4</video>
</details>
```

`<details>` is in CommonMark's block-tag list, marked emits a single `html`
token spanning the wrapper, `HTMLToken.svelte` then strips the wrapper and
renders only the `<video controls>` element. Side note: `<details>` blocks
with blank lines inside break this — keep all the inner content on
consecutive non-blank lines.

### 5. Tool return values aren't visible to the user

The tool's `return ...` value is fed back to the model as `tool_call_output`,
but the model is responsible for repeating any media markdown — and most
models don't, even when instructed. To make the video actually show up,
inject it via:

```python
await __event_emitter__({"type": "message", "data": {"content": video_markdown}})
```

Then `return` a short non-media confirmation string for the model to use as
context.

## Roadmap (maybe)

- Image-to-video (use `optional_cond_images` on the base/extend samplers)
- Looping video via `LTXVLoopingSampler` for seamless loops
- Width/height presets (square, vertical, etc.)
- Optional audio gen for long clips via a separate audio-only LTX call + ffmpeg mux
- Negative prompt enhancement (current enhancer rewrites positive only)

PRs welcome.

## License

[Apache-2.0](LICENSE)
