# Qwen3.8 VL for ComfyUI

[English](README.md) | [简体中文](README_zh-CN.md)

A ComfyUI custom-node pack for local Qwen3.8 VL inference through GGUF/`llama.cpp`, plus simple OpenAI-compatible multimodal API backends. The local and API backends share one chat node, so the same text, image, and video workflow can be switched between backends.

> This is an independent community project. Qwen3.8 model weights are not included.

## Quick start

From the ComfyUI installation directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/xzbdqian10nian/ComfyUI-Qwen3.8-VL.git
```

1. Put one matching GGUF main model and its `mmproj` projector in:

   ```text
   ComfyUI/models/LLM/Qwen3.8/
   ```

2. Restart ComfyUI (or refresh the browser after the server has loaded the node pack).
3. Add `Qwen3.8 VL Local Loader` and select the main model and projector.
4. Connect its `backend` output to `Vision LLM Chat`.
5. Enter a prompt and queue the workflow. For images, connect an `IMAGE` input; for videos, connect `VIDEO` or an `IMAGE` frame batch.

The loader uses ComfyUI's configured model directory, so the same node pack works with normal ComfyUI installs, portable builds, and cloud images. `QWEN38_MODEL_DIR` is an optional override for users who deliberately keep this model in a separate directory.

### Update

```bash
git -C ComfyUI-Qwen3.8-VL pull --ff-only
```

Restart ComfyUI after updating. If the plugin was installed under an older local folder name, run the command from that folder instead.

## Nodes

| Node | Purpose |
| --- | --- |
| `Qwen3.8 VL Local Loader` | Loads a local Qwen3.8 GGUF model and matching vision projector. |
| `OpenAI-Compatible API · Environment Variable` | Calls an OpenAI-compatible endpoint using an API-key environment variable. |
| `OpenAI-Compatible API · Direct Key` | Calls an OpenAI-compatible endpoint using a key entered in the node. |
| `Vision LLM Chat` | Sends text, images, image batches, video frames, or ComfyUI `VIDEO` to the selected backend. |
| `Backend Unload` | Explicitly releases a local model or closes an API backend. |

The node classes keep stable internal types for workflow compatibility; the visible names are short and backend-neutral.

## Local Qwen3.8 VL model

The local backend expects two compatible files:

```text
ComfyUI/models/LLM/Qwen3.8/
├── <Qwen3.8 main model>.gguf
└── <matching mmproj>.gguf
```

The main model contains the language-model weights. The `mmproj` file is the vision projector required for image and video understanding. Keep both files from the same model release and do not mix projectors between publishers.

For a 32 GB card, start with `Q4_K_M` or an equivalent UD Q4 variant. Q6/Q8 variants need more VRAM. The loader exposes context length, batch size, micro-batch size, GPU layers, and ComfyUI VRAM cleanup; these settings are intentionally kept in the loader/chat nodes rather than hidden in a platform-specific launcher.

### Model downloads

These are community GGUF distributions of Qwen3.8-27B. Download only the files you need.

| Source | Typical files | Difference |
| --- | --- | --- |
| [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [UD Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-Q4_K_M.gguf?download=true) + [mmproj-BF16](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/mmproj-BF16.gguf?download=true) | Recommended starting point; compact dynamic quantization with the original model alignment. |
| [Huihui AI Abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF) | [Q4_K](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf?download=true) or Q4_K_L + `mmproj-model-bf16.gguf` | A community low-refusal variant; larger Q4_K_L retains more high-precision tensors. |
| [Orcarouter Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF) | [Q4_K_M](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/Qwen3.8-27B-Uncensored-Q4_K_M.gguf?download=true) + [mmproj f16](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-GGUF/resolve/main/mmproj-Qwen3.8-27B-Uncensored-f16.gguf?download=true) | Another community low-refusal variant. Hugging Face access conditions may apply. |

Community variants can differ in alignment, refusal behavior, quality, and licensing. Review each model card before redistribution or public deployment.

## API backends

Both API nodes send a small OpenAI Chat Completions-compatible request. Configure:

- `base_url`: the provider's API root, for example `https://api.openai.com/v1`;
- `model`: the provider's model ID;
- `prompt`: the user instruction;
- `image`: an optional single image or `IMAGE` batch;
- `video_frames` or `video`: optional video input, depending on provider support.

### Environment-variable key

Set the variable before starting ComfyUI:

```bash
export OPENAI_API_KEY='your-api-key'
```

In `OpenAI-Compatible API · Environment Variable`, enter `OPENAI_API_KEY` as `api_key_env`, not the secret itself.

### Direct key

Use `OpenAI-Compatible API · Direct Key` when a temporary key is more convenient. Do not save a real key in a public workflow or commit it to Git.

### Parameters

- `max_tokens = 0`: omit `max_tokens` and use the provider default.
- `temperature = 0`: omit `temperature` and use the provider default.
- `seed`: pass a seed when the provider supports it.
- API progress is conservative when the provider does not expose a total output limit; it advances gradually and completes at the end of a normal response.

### Environment-key endpoint security

The environment-variable node never sends a server-side key to an arbitrary
workflow-supplied host. It accepts exact HTTPS host matches from the server-side
allow-list; the default is `api.openai.com`. To allow another provider, set the
following before starting ComfyUI:

```bash
export COMFYUI_API_ALLOWED_HOSTS='api.openai.com,api.example.com'
```

Entries are exact `host` or `host:port` values; wildcards are not accepted. The
Direct Key node is separate and can use another endpoint because it sends only
the key explicitly entered by the workflow user, never a server environment
variable. Use HTTPS for public endpoints.

## Images and video

- Connect one image directly to `image`.
- Connect an `IMAGE` batch to `image` to send multiple images in one request.
- The plugin does not silently resize images or truncate an `IMAGE` batch. Add any resize or frame-limit node upstream when the workflow needs one.
- Connect decoded video frames to `video_frames` for local inference or providers that accept image frames.
- Connect ComfyUI `VIDEO` to `video` when the backend supports video transport; the plugin can sample frames according to `max_video_frames`.

## Example workflows

The [`example_workflows/`](example_workflows/) directory contains five local-backend examples:

| File | Demonstrates |
| --- | --- |
| [`01_text_chat.json`](example_workflows/01_text_chat.json) | Text-only chat |
| [`02_single_image.json`](example_workflows/02_single_image.json) | One image |
| [`03_multiple_images.json`](example_workflows/03_multiple_images.json) | Multiple images as one `IMAGE` batch |
| [`04_video_frames.json`](example_workflows/04_video_frames.json) | `VIDEO` decoded to image frames |
| [`05_comfyui_video.json`](example_workflows/05_comfyui_video.json) | Native ComfyUI `VIDEO` input |

Each example includes a model/setup Markdown note and connects the chat response to ComfyUI's `Preview as Text` node. Replace the sample media with your own files, then select the model pair available in your model directory.

## Compatibility and dependencies

- Python 3.10+.
- CUDA-capable `llama-cpp-python` is required for local GGUF inference. Install a wheel matching the CUDA/runtime already used by your ComfyUI image.
- API mode can use the existing `openai` SDK; a standard-library HTTP fallback is included.
- The plugin does not replace PyTorch, CUDA, the NVIDIA driver, or the existing Python environment.
- `requirements.txt` is intentionally limited to the plugin's small required dependency set.

## ComfyUI Manager and Registry

The repository includes Comfy Registry metadata in `pyproject.toml`, including a semantically versioned package name, repository URL, publisher ID, and display name. Once the package is published to the [Comfy Registry](https://registry.comfy.org), it can be discovered and installed from the current ComfyUI Manager. Until then, direct Git installation remains available:

```bash
git clone https://github.com/xzbdqian10nian/ComfyUI-Qwen3.8-VL.git
```

## Troubleshooting

**The nodes do not appear**

1. Confirm the repository is under `ComfyUI/custom_nodes/`.
2. Check the ComfyUI console for the plugin import line.
3. Hard-refresh the browser.
4. Make sure no other copy of this plugin is enabled under a second folder.

**The local loader has no model choices**

Confirm that both `.gguf` files are inside `ComfyUI/models/LLM/Qwen3.8/`, are complete downloads, and that the projector filename contains `mmproj`.

**The local model fails to load**

Use a matching main-model/projector pair and verify that the installed `llama-cpp-python` wheel supports the current machine. Lower context length, batch size, or GPU layers if VRAM is insufficient.

**An API request fails**

Check the endpoint's `/v1` suffix, model ID, key source, provider vision support, and whether that provider accepts the requested image/video format.

## Acknowledgements

Thanks to the [Qwen Team](https://huggingface.co/Qwen) for [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and to [Unsloth](https://huggingface.co/unsloth), [huihui-ai](https://huggingface.co/huihui-ai), and [orcarouter](https://huggingface.co/orcarouter) for community GGUF distributions.

This project is independent and is not affiliated with or endorsed by the Qwen Team or the listed community publishers.
