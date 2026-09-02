"""Visible ComfyUI nodes for local and API vision LLM backends."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from typing import Any

import torch

from .backends import (
    BackendError,
    LocalQwen38Backend,
    LocalRuntimeSettings,
    OpenAICompatibleBackend,
)
from .media import (
    encode_video_data_url,
    extract_video_frames,
    image_batch_frames,
    pil_to_data_url,
    sample_image_batch,
    tensor_frame_to_pil,
)
from .nodes import _choices, _resolve_file
from .progress import ConsoleProgressBar, StatusTicker, make_progress, send_status, update_progress


API_AUTO_PROGRESS_CHUNKS_PER_PERCENT = 2000


def _api_progress_value(streamed_chunks: int, requested_max_tokens: int) -> tuple[int, int]:
    """Map an API stream count onto a progress value without fake precision."""
    count = max(0, int(streamed_chunks))
    requested = max(0, int(requested_max_tokens))
    if requested > 0:
        # Reserve the final step for a confirmed completed response.
        return min(count, max(0, requested - 1)), requested
    # With a provider-controlled output limit there is no real denominator.
    # Advance slowly enough for long-context responses and stop at 99% until
    # the server emits a normal completion.
    percent = min(count // API_AUTO_PROGRESS_CHUNKS_PER_PERCENT, 99)
    return percent, 100


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain(model_dump())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _to_plain(to_dict())
        except Exception:
            pass
    return value


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_completion(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    data = _to_plain(result)
    if not isinstance(data, dict):
        raise BackendError(f"The backend returned an unsupported result type: {type(result).__name__}.")
    choices = data.get("choices") or []
    if not choices:
        raise BackendError("The backend response did not contain any choices.")
    choice = choices[0] if isinstance(choices[0], dict) else _to_plain(choices[0])
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        # A few completion-style compatible endpoints return `text` instead.
        message = {"content": choice.get("text", "") if isinstance(choice, dict) else ""}
    message = dict(message)
    if "content" in message:
        message["content"] = _content_text(message["content"])
    return message, data.get("usage") or {}


def _merge_tool_call_delta(
    tool_calls: dict[int, dict[str, Any]], delta_calls: Any
) -> None:
    """Merge streamed OpenAI tool-call fragments by their index."""
    for position, raw_call in enumerate(_to_plain(delta_calls) or []):
        if not isinstance(raw_call, dict):
            continue
        index = int(raw_call.get("index", position))
        target = tool_calls.setdefault(
            index,
            {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if raw_call.get("id"):
            target["id"] = str(raw_call["id"])
        if raw_call.get("type"):
            target["type"] = str(raw_call["type"])
        function = raw_call.get("function") or {}
        if isinstance(function, dict):
            target_function = target.setdefault("function", {"name": "", "arguments": ""})
            if function.get("name"):
                target_function["name"] += str(function["name"])
            if function.get("arguments"):
                target_function["arguments"] += str(function["arguments"])


def _collect_stream(result: Any, progress_callback=None) -> Any:
    """Turn a local/API completion stream into a regular response dictionary."""
    plain = _to_plain(result)
    if isinstance(plain, dict):
        if callable(progress_callback):
            usage = plain.get("usage") or {}
            progress_callback(int(usage.get("completion_tokens") or 0))
        return plain

    if isinstance(result, (str, bytes, bytearray)) or not isinstance(result, Iterable):
        return result

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason = None
    usage: dict[str, Any] = {}
    chunk_count = 0

    for raw_chunk in result:
        chunk = _to_plain(raw_chunk)
        if not isinstance(chunk, dict):
            continue
        if isinstance(chunk.get("usage"), dict):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0] if isinstance(choices[0], dict) else _to_plain(choices[0])
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or choice.get("message") or {}
        if not isinstance(delta, dict):
            continue
        content = _content_text(delta.get("content"))
        if content:
            content_parts.append(content)
        reasoning = (
            delta.get("reasoning_content")
            or delta.get("reasoning")
            or delta.get("thinking_content")
            or ""
        )
        if reasoning:
            reasoning_parts.append(str(reasoning))
        if delta.get("tool_calls"):
            _merge_tool_call_delta(tool_calls, delta["tool_calls"])
        finish_reason = choice.get("finish_reason") or finish_reason
        chunk_count += 1
        if callable(progress_callback):
            progress_callback(chunk_count)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tool_calls:
        message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }


def _split_message(message: dict[str, Any]) -> tuple[str, str, str]:
    content = str(message.get("content") or "")
    reasoning = str(
        message.get("reasoning_content")
        or message.get("reasoning")
        or message.get("thinking_content")
        or ""
    )
    raw = content
    if not reasoning and "<think>" in content:
        before, _, tail = content.partition("<think>")
        thought, marker, after = tail.partition("</think>")
        if marker:
            reasoning = thought.strip()
            content = (before + after).strip()
    elif not reasoning and "</think>" in content:
        # llama.cpp/Qwen can prefill the opening <think> token in the assistant
        # template.  Generated text then contains only the closing marker.  In
        # that representation everything before </think> is still reasoning,
        # not part of the response consumed by downstream ComfyUI nodes.
        thought, _, after = content.partition("</think>")
        reasoning = thought.strip()
        content = after.strip()
    content = content.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return content, reasoning.strip(), raw


def _parse_tools(tools_json: str | None) -> list[dict[str, Any]] | None:
    if not tools_json or not tools_json.strip():
        return None
    try:
        parsed = json.loads(tools_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tools_json is not valid JSON: {exc}") from exc
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError("tools_json must be a tool object or an array of tool objects.")
    return parsed


def _build_content(
    prompt: str,
    image: torch.Tensor | None,
    video_frames: torch.Tensor | None,
    video: Any | None,
    max_video_frames: int,
    video_transport: str,
    api_backend: bool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    if image is not None:
        frames = image_batch_frames(image)
        for index, frame in enumerate(frames, 1):
            if len(frames) > 1:
                content.append({"type": "text", "text": f"Image {index}/{len(frames)}:"})
            pil = tensor_frame_to_pil(frame)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil)}})

    # A VIDEO object can be sent as a native data URL to APIs that implement
    # the OpenAI-compatible `video_url` convention.  Local llama.cpp handlers instead
    # receive decoded frames, because they consume image parts.
    use_native_video = video is not None and api_backend and video_transport in {"auto", "video_url"}
    if use_native_video:
        try:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": encode_video_data_url(video)},
                }
            )
        except Exception:
            if video_transport == "video_url":
                raise
            use_native_video = False

    if video_frames is not None:
        frames = sample_image_batch(video_frames, max_video_frames)
        for index, frame in enumerate(frames, 1):
            content.append({"type": "text", "text": f"Video frame {index}/{len(frames)}:"})
            pil = tensor_frame_to_pil(frame)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil, 85)}})
    elif video is not None and not use_native_video:
        frames = extract_video_frames(video, max_video_frames)
        for index, pil in enumerate(frames, 1):
            content.append({"type": "text", "text": f"Video frame {index}/{len(frames)}:"})
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil, 85)}})

    content.append({"type": "text", "text": prompt.strip()})
    return content


def _run_chat(
    backend: Any,
    prompt: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repeat_penalty: float,
    seed: int,
    max_video_frames: int,
    video_transport: str,
    thinking_mode: str,
    image: torch.Tensor | None,
    video_frames: torch.Tensor | None,
    video: Any | None,
    tools_json: str | None,
    progress_callback=None,
):
    is_api = getattr(backend, "backend_kind", "") == "openai_compatible"
    content = _build_content(
        prompt,
        image,
        video_frames,
        video,
        int(max_video_frames),
        video_transport,
        is_api,
    )
    # Some image/video chat templates (including Qwen3.8's MTMD template)
    # insert their own default system message when none is supplied.  Passing
    # an empty system message leaves it in second position after that default
    # and the template rejects the prompt with "System message must be at the
    # beginning".  Omit empty optional messages instead of serializing them.
    messages = []
    cleaned_system_prompt = system_prompt.strip()
    if cleaned_system_prompt:
        messages.append({"role": "system", "content": cleaned_system_prompt})
    messages.append({"role": "user", "content": content})
    tools = _parse_tools(tools_json)
    kwargs: dict[str, Any] = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "repeat_penalty": float(repeat_penalty),
        "seed": int(seed),
        "stream": callable(progress_callback),
    }
    if is_api:
        kwargs["thinking_mode"] = thinking_mode
    if tools:
        kwargs["tools"] = tools

    started = time.perf_counter()
    result = backend.complete(**kwargs)
    result = _collect_stream(result, progress_callback)
    elapsed = time.perf_counter() - started
    message, usage = _extract_completion(result)
    response, reasoning, raw = _split_message(message)
    tool_calls = message.get("tool_calls")
    if tool_calls and not response:
        response = json.dumps(tool_calls, ensure_ascii=False, indent=2)
    if tool_calls:
        raw = json.dumps(message, ensure_ascii=False, indent=2)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    speed = completion_tokens / elapsed if completion_tokens and elapsed > 0 else 0.0
    image_count = len(image_batch_frames(image)) if image is not None else 0
    video_frame_count = len(sample_image_batch(video_frames, max_video_frames)) if video_frames is not None else 0
    stats = (
        f"time={elapsed:.2f}s\n"
        f"prompt_tokens={usage.get('prompt_tokens', 'n/a')}\n"
        f"completion_tokens={usage.get('completion_tokens', 'n/a')}\n"
        f"speed={speed:.2f} tok/s\n"
        f"images={image_count}\nvideo_frames={video_frame_count}\n"
        f"backend={getattr(backend, 'backend_kind', type(backend).__name__)}"
    )
    return response, reasoning, raw, stats


class Qwen38VLLoader:
    """Load the configured local Qwen3.8 GGUF backend."""
    RETURN_TYPES = ("VISION_LLM_BACKEND", "STRING")
    RETURN_NAMES = ("backend", "backend_info")
    OUTPUT_TOOLTIPS = (
        "Loaded local Qwen3.8 VL backend for the Vision LLM Chat node.",
        "Resolved model, projector, GPU layers, and backend state.",
    )
    CATEGORY = "Vision LLM/Backends"
    DESCRIPTION = "Loads a local Qwen3.8 GGUF model and its vision projector with CUDA llama.cpp."
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": (
                    cls._model_choices(),
                    {
                        "default": cls._model_choices()[0],
                        "tooltip": "Main Qwen3.8 GGUF model file. Q4_K_M is a practical starting point for 32 GB VRAM.",
                    },
                ),
                "mmproj_file": (
                    cls._mmproj_choices(),
                    {
                        "default": cls._mmproj_choices()[0],
                        "tooltip": "Vision projector GGUF required to understand images and video frames.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 128,
                        "max": 8192,
                        "step": 128,
                        "tooltip": "Number of prompt tokens processed in one llama.cpp batch. Higher can improve prompt processing speed but uses more VRAM.",
                    },
                ),
                "micro_batch_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 2048,
                        "step": 64,
                        "tooltip": "Maximum tokens processed in one physical llama.cpp sub-batch. Reduce this first if prompt processing runs out of VRAM.",
                    },
                ),
                "gpu_layers": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 256,
                        "tooltip": "Number of transformer layers offloaded to the GPU. -1 offloads all layers; lower it for partial GPU offload.",
                    },
                ),
                "free_comfy_vram": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Unload ComfyUI diffusion models before loading this large local vision-language model.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @staticmethod
    def _model_choices():
        return _choices("model")

    @staticmethod
    def _mmproj_choices():
        return _choices("mmproj")

    def load_model(
        self,
        model_file: str,
        mmproj_file: str,
        batch_size: int,
        micro_batch_size: int,
        gpu_layers: int,
        free_comfy_vram: bool,
        unique_id: str | None = None,
    ):
        send_status("Preparing local model backend…", unique_id)
        model_path = _resolve_file(model_file, "model")
        mmproj_path = _resolve_file(mmproj_file, "mmproj")
        backend = LocalQwen38Backend(
            LocalRuntimeSettings(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_batch=int(batch_size),
                n_ubatch=int(micro_batch_size),
                n_gpu_layers=int(gpu_layers),
                free_comfy_vram=bool(free_comfy_vram),
            )
        )
        send_status(f"Local backend ready: {model_path.name}", unique_id)
        return backend, backend.info()


class _VisionAPINodeBase:
    """Shared implementation for the two deliberately simple API nodes."""

    auth_mode = "env"

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "base_url": (
                "STRING",
                {
                    "default": "https://api.openai.com/v1",
                    "tooltip": (
                        "OpenAI-compatible API base URL; do not include /chat/completions. "
                        "The environment-key node only permits the administrator's host allow-list."
                        if cls.auth_mode == "env"
                        else "OpenAI-compatible API base URL; do not include /chat/completions."
                    ),
                },
            ),
            "model": (
                "STRING",
                {"default": "", "tooltip": "Exact model ID exposed by the API provider."},
            ),
        }
        if cls.auth_mode == "env":
            required["api_key_env"] = (
                "STRING",
                {
                    "default": "OPENAI_API_KEY",
                    "tooltip": "Environment variable name containing the API key. Its request host must be administrator-allow-listed.",
                },
            )
        else:
            required["api_key"] = (
                "STRING",
                {
                    "default": "",
                    "password": True,
                    "tooltip": "API key used for this request. It is not read from an environment variable.",
                },
            )
        required.update(
            {
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "User instruction sent to the API, together with any attached image or video input.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Optional persona and system-level instruction; leave blank for the provider default.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 32768,
                        "step": 16,
                        "tooltip": "Maximum output token count. 0 omits this parameter and uses the provider default.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Sampling temperature. 0 means do not send this parameter; use the provider default.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 2**32 - 1,
                        "control_after_generate": True,
                        "tooltip": "Seed sent for sampling when the provider supports it; providers may ignore it.",
                    },
                ),
            }
        )
        return {
            "required": required,
            "optional": {
                "max_video_frames": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 64,
                        "step": 1,
                        "tooltip": "Maximum number of evenly sampled frames used when a VIDEO input is converted to frames. It does not limit an IMAGE batch.",
                    },
                ),
                "video_transport": (
                    ["frames", "video_url", "auto"],
                    {
                        "default": "frames",
                        "tooltip": "Choose sampled image frames, native video_url, or automatic native-video fallback. The provider must support the selected format.",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "One still image or an IMAGE batch; every item is sent in the same API request.",
                    },
                ),
                "video_frames": (
                    "IMAGE",
                    {
                        "tooltip": "Decoded video as an IMAGE batch; frames are sampled according to max_video_frames.",
                    },
                ),
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "ComfyUI VIDEO input; use frames for broad support or video_url only when the provider supports native video.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "usage", "stats")
    OUTPUT_TOOLTIPS = (
        "Final API response text.",
        "Usage JSON returned by the provider, when available.",
        "Request timing and backend information.",
    )
    FUNCTION = "request"
    CATEGORY = "Vision LLM/API"
    DESCRIPTION = "Sends one simple text, image, image-batch, video-frame, or VIDEO request to an OpenAI-compatible API."
    OUTPUT_NODE = True

    def request(
        self,
        base_url: str,
        model: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int = 1,
        api_key_env: str = "",
        api_key: str = "",
        image: torch.Tensor | None = None,
        max_video_frames: int = 8,
        video_transport: str = "frames",
        video_frames: torch.Tensor | None = None,
        video: Any | None = None,
        unique_id: str | None = None,
    ):
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            timeout=120.0,
            restrict_endpoint=self.auth_mode == "env",
        )
        requested_max_tokens = int(max_tokens)
        # Keep 0 intact in the request so the API applies its own default. The
        # helper then uses a deliberately slow approximate scale because the
        # provider has not exposed a real output denominator.
        _, progress_total = _api_progress_value(0, requested_max_tokens)
        progress = make_progress(progress_total, unique_id)
        log_progress = ConsoleProgressBar("API generation", progress_total)
        ticker = StatusTicker(unique_id)
        update_progress(progress, 0, progress_total)
        ticker.send("Preparing API request…", force=True)

        def report_token(count: int):
            visible, _ = _api_progress_value(count, requested_max_tokens)
            update_progress(progress, visible, progress_total)
            log_progress.update(visible, suffix=f"{count} streamed chunks")
            ticker.send(
                f"API generating… {count} streamed chunks",
                mirror_log=False,
            )

        started = time.perf_counter()
        try:
            response, _reasoning, _raw, stats = _run_chat(
                backend,
                prompt,
                system_prompt,
                requested_max_tokens,
                float(temperature),
                0.0,
                0,
                0.0,
                1.0,
                int(seed),
                max(1, int(max_video_frames)),
                video_transport,
                "backend_default",
                image,
                video_frames,
                video,
                None,
                report_token,
            )
            update_progress(progress, progress_total, progress_total)
            log_progress.finish("response complete")
            ticker.send("API request complete", force=True)
            usage = json.dumps(
                {
                    line.split("=", 1)[0]: line.split("=", 1)[1]
                    for line in stats.splitlines()
                    if "=" in line and line.split("=", 1)[0] in {"prompt_tokens", "completion_tokens"}
                },
                ensure_ascii=False,
            )
            elapsed = time.perf_counter() - started
            direct_stats = f"{stats}\napi_elapsed={elapsed:.2f}s"
            return {
                "ui": {"text": (response,)},
                "result": (response, usage, direct_stats),
            }
        finally:
            log_progress.close()
            backend.unload()


class VisionAPIEnv(_VisionAPINodeBase):
    auth_mode = "env"


class VisionAPIDirect(_VisionAPINodeBase):
    auth_mode = "direct"


class VisionChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("VISION_LLM_BACKEND", {"tooltip": "Connect the Qwen3.8 VL local loader or any OpenAI-compatible API backend."}),
                "prompt": (
                    "STRING",
                    {"default": "Please describe the input image or video in detail.", "multiline": True, "tooltip": "User instruction sent with the attached image/video content."},
                ),
                "system_prompt": (
                    "STRING",
                    {"default": "You are a professional visual-understanding assistant running in ComfyUI. Answer accurately and directly.", "multiline": True, "tooltip": "Persona and system-level behavior or response-format instruction."},
                ),
                "thinking_mode": (["backend_default", "thinking", "instruct"], {"default": "backend_default", "tooltip": "Use the backend default, force reasoning output, or force direct instruct mode. Local Qwen3.8 defaults to direct instruct mode."}),
                "context_length": (
                    "INT",
                    {
                        "default": 8192,
                        "min": 2048,
                        "max": 262144,
                        "step": 1024,
                        "tooltip": "Local llama.cpp context window for the prompt, media tokens, and response. API backends ignore it; changing it reloads a loaded local model.",
                    },
                ),
                "max_tokens": ("INT", {"default": 4096, "min": 16, "max": 32768, "step": 16, "tooltip": "Maximum number of new text tokens to generate. This is only a limit; generation still stops when the model finishes its answer."}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "Sampling randomness. Lower values are steadier; higher values are more varied. With a local model, 0 uses deterministic greedy sampling when supported."}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Nucleus-sampling probability threshold. Lower values keep only more likely tokens; 0.95 is a strong general default."}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200, "tooltip": "For local llama.cpp, restrict each token choice to the top K candidates. 0 disables top-K filtering; API backends ignore it."}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "For local llama.cpp, discard tokens whose probability is below this fraction of the best token's probability. API backends ignore it."}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01, "tooltip": "For local llama.cpp, penalize repeated text. 1.0 disables the penalty; API backends ignore it."}),
                "seed": (
                    "INT",
                    {
                        "default": 1,
                        "min": 0,
                        "max": 2**32 - 1,
                        "control_after_generate": True,
                        "tooltip": "Random seed used for reproducible sampling when all other inputs and backend settings match. API providers may ignore it.",
                    },
                ),
                "max_video_frames": ("INT", {"default": 8, "min": 1, "max": 64, "tooltip": "Maximum number of evenly sampled frames used when a VIDEO input is converted to frames. It does not limit an IMAGE batch."}),
                "video_transport": (["auto", "frames", "video_url"], {"default": "auto", "tooltip": "For API backends, choose automatic native-video when available, sampled image frames, or native video_url. Local models always use frames."}),
                "unload_after": ("BOOLEAN", {"default": False, "tooltip": "Release this backend after generation. Enable to recover VRAM, disable for faster repeated chats."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "One still image or an IMAGE batch of still images."}),
                "video_frames": ("IMAGE", {"tooltip": "Decoded video frames as a ComfyUI IMAGE batch; the batch is sampled according to max video frames."}),
                "video": ("VIDEO", {"tooltip": "ComfyUI VIDEO object; local models sample it into frames, while API transport follows video transport."}),
                "tools_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional OpenAI function/tool schema as one JSON object or an array."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "raw_response", "stats")
    OUTPUT_TOOLTIPS = (
        "Final assistant answer without hidden reasoning tags.",
        "Reasoning text when the backend exposes it.",
        "Raw assistant content or complete tool-call message.",
        "Timing, token, speed, media-count, and backend statistics.",
    )
    FUNCTION = "generate"
    CATEGORY = "Vision LLM/Chat"
    DESCRIPTION = "Sends text, images, or video to a local Qwen3.8 VL model or any OpenAI-compatible API backend."
    OUTPUT_NODE = True

    def generate(
        self,
        backend: Any,
        prompt: str,
        system_prompt: str,
        thinking_mode: str,
        context_length: int,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
        seed: int,
        max_video_frames: int,
        video_transport: str,
        unload_after: bool,
        image: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        video: Any | None = None,
        tools_json: str | None = None,
        unique_id: str | None = None,
    ):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise BackendError("Connect this plugin's local model loader or an API backend node.")
        total = max(1, int(max_tokens))
        progress = make_progress(total, unique_id)
        generation_log = ConsoleProgressBar("Local model generation", total)
        ticker = StatusTicker(unique_id)
        update_progress(progress, 0, total)
        ticker.send(f"Preparing image/video input… (max_tokens={total})", force=True)

        configure_chat = getattr(backend, "configure_chat", None)
        if callable(configure_chat):
            configure_chat(context_length, thinking_mode)

        ensure_loaded = getattr(backend, "ensure_loaded", None)
        if callable(ensure_loaded):
            load_log = ConsoleProgressBar("Local model loading", 100, interval=0.0)
            phase_text = {
                "free_vram": "Freeing ComfyUI VRAM…",
                "projector": "Loading vision projector…",
                "weights": "Loading model weights…",
                "ready": "Local model ready",
            }

            def report_load(phase: str, fraction: float):
                phase_name, separator, phase_detail = phase.partition(":")
                phase_label = phase_text.get(phase_name, phase_name)
                if separator and phase_name == "weights_wait":
                    phase_label = f"Loading model weights… ({phase_detail}s elapsed)"
                load_value = round(max(0.0, min(1.0, fraction)) * 100)
                if phase_name == "ready":
                    load_log.finish("ready")
                else:
                    load_log.update(
                        load_value,
                        suffix=phase_label,
                        force=True,
                    )
                ticker.send(
                    f"{phase_label} [{load_value}%]",
                    force=True,
                    mirror_log=False,
                )

            try:
                ensure_loaded(progress_callback=report_load)
            finally:
                load_log.close()

        def report_token(count: int):
            visible = min(max(1, int(count)), total - 1) if total > 1 else 1
            update_progress(progress, visible, total)
            percent = round(100 * visible / total)
            generation_log.update(visible, suffix=f"{count} streamed chunks")
            ticker.send(
                f"Generating… {count} streamed chunks (token budget ~{percent}%)",
                mirror_log=False,
            )

        try:
            response, reasoning, raw, stats = _run_chat(
                backend,
                prompt,
                system_prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                min_p,
                repeat_penalty,
                seed,
                max_video_frames,
                video_transport,
                thinking_mode,
                image,
                video_frames,
                video,
                tools_json,
                report_token,
            )
            update_progress(progress, total, total)
            generation_log.finish(stats.splitlines()[0])
            ticker.send(f"Generation complete · {stats.splitlines()[0]}", force=True)
            print(f"[ComfyUI-Qwen3.8-VL] Generation finished: {stats.splitlines()[0]}")
            return {
                "ui": {"text": (response,)},
                "result": (response, reasoning, raw, stats),
            }
        finally:
            generation_log.close()
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()


class VisionUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"backend": ("VISION_LLM_BACKEND", {"tooltip": "Backend instance to unload or close."})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_TOOLTIPS = ("Human-readable unload result.",)
    FUNCTION = "unload"
    CATEGORY = "Vision LLM/Backends"
    DESCRIPTION = "Explicitly unloads a local model or closes an API client."
    OUTPUT_NODE = True

    def unload(self, backend: Any):
        if callable(getattr(backend, "unload", None)):
            backend.unload()
        status = "Vision LLM backend unloaded"
        return {"ui": {"text": (status,)}, "result": (status,)}


NODE_CLASS_MAPPINGS = {
    "Qwen38VLLoader": Qwen38VLLoader,
    "VisionAPIEnv": VisionAPIEnv,
    "VisionAPIDirect": VisionAPIDirect,
    "VisionChat": VisionChat,
    "VisionUnload": VisionUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen38VLLoader": "Qwen3.8 VL Local Loader",
    "VisionAPIEnv": "OpenAI-Compatible API · Environment Variable",
    "VisionAPIDirect": "OpenAI-Compatible API · Direct Key",
    "VisionChat": "Vision LLM Chat",
    "VisionUnload": "Backend Unload",
}
