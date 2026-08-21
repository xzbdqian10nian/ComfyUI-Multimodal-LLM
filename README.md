# ComfyUI Qwen3.8 Vision LLM

First-pass local ComfyUI integration for `Qwen3.8-27B` using a vision-capable
`llama-cpp-python` CUDA backend.

## Current baseline

- Model: `Qwen3.8-27B-UD-Q4_K_M.gguf`
- Vision projector: `mmproj-BF16.gguf`
- Persistent model directory: `/poddata/ComfyUI/models/LLM/Qwen3.8`
- Inputs: text, one/multiple images, sampled video frames
- Outputs: final response, separated reasoning, raw response, timing/token stats
- Thinking can be selected in the loader (`thinking` / `instruct`).
- Optional OpenAI-style `tools_json` is accepted for function-call experiments.

The Q4 model is only the compatibility baseline. It is not described as BF16,
FP16, FP8, or a "half precision" model. Higher-precision/offload profiles will
be added after the image flow is proven stable.

## Nodes

1. `Qwen3.8 Model Loader (GGUF)`
2. `Qwen3.8 Vision Chat`
3. `Qwen3.8 Unload Model`

For video, connect an `IMAGE` batch. Frames are sampled evenly and sent as
ordered images because llama.cpp does not yet expose native video input here.

## Runtime requirement

Use a recent JamePeng `llama-cpp-python` wheel containing
`Qwen35ChatHandler`; Qwen3.8 shares the Qwen3.5-family multimodal chat handler.
The server baseline uses the CUDA 12.8 / CPython 3.12 Linux wheel, which remains
compatible with the newer NVIDIA driver on the RTX 5090 host.

