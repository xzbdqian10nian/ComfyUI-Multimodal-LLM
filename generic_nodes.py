"""Generic backend and chat nodes.

The original Qwen3.8 nodes remain in :mod:`nodes` for workflow compatibility.
These nodes use a common ``MLLM_BACKEND`` socket so a local model loader or an
OpenAI-compatible API can feed the same chat node.
"""

from __future__ import annotations

import json
import time
from typing import Any

import torch

from .backends import BackendError, OpenAICompatibleBackend
from .media import (
    encode_video_data_url,
    extract_video_frames,
    pil_to_data_url,
    sample_image_batch,
    tensor_frame_to_pil,
)
from .nodes import Qwen38ModelLoader


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
        raise BackendError(f"后端返回了无法解析的结果类型：{type(result).__name__}")
    choices = data.get("choices") or []
    if not choices:
        raise BackendError("后端响应中没有 choices。")
    choice = choices[0] if isinstance(choices[0], dict) else _to_plain(choices[0])
    message = choice.get("message") if isinstance(choice, dict) else None
    if not isinstance(message, dict):
        # A few completion-style compatible endpoints return `text` instead.
        message = {"content": choice.get("text", "") if isinstance(choice, dict) else ""}
    message = dict(message)
    if "content" in message:
        message["content"] = _content_text(message["content"])
    return message, data.get("usage") or {}


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
    content = content.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    return content, reasoning.strip(), raw


def _parse_tools(tools_json: str | None) -> list[dict[str, Any]] | None:
    if not tools_json or not tools_json.strip():
        return None
    try:
        parsed = json.loads(tools_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tools_json 不是有效 JSON：{exc}") from exc
    if isinstance(parsed, dict):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError("tools_json 必须是工具对象或工具对象数组。")
    return parsed


def _build_content(
    prompt: str,
    image: torch.Tensor | None,
    video_frames: torch.Tensor | None,
    video: Any | None,
    max_image_edge: int,
    max_image_frames: int,
    max_video_frames: int,
    video_transport: str,
    api_backend: bool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []

    if image is not None:
        frames = sample_image_batch(image, max_image_frames)
        for index, frame in enumerate(frames, 1):
            if len(frames) > 1:
                content.append({"type": "text", "text": f"Image {index}/{len(frames)}:"})
            pil = tensor_frame_to_pil(frame, max_image_edge)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil)}})

    # A VIDEO object can be sent as a native data URL to APIs that implement
    # the RH/OpenAI `video_url` convention.  Local llama.cpp handlers instead
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
            pil = tensor_frame_to_pil(frame, max_image_edge)
            content.append({"type": "image_url", "image_url": {"url": pil_to_data_url(pil, 85)}})
    elif video is not None and not use_native_video:
        frames = extract_video_frames(video, max_video_frames, max_image_edge)
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
    max_image_edge: int,
    max_image_frames: int,
    max_video_frames: int,
    video_transport: str,
    thinking_mode: str,
    image: torch.Tensor | None,
    video_frames: torch.Tensor | None,
    video: Any | None,
    tools_json: str | None,
):
    is_api = getattr(backend, "backend_kind", "") == "openai_compatible"
    content = _build_content(
        prompt,
        image,
        video_frames,
        video,
        int(max_image_edge),
        int(max_image_frames),
        int(max_video_frames),
        video_transport,
        is_api,
    )
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": content},
    ]
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
        "stream": False,
    }
    if is_api:
        kwargs["thinking_mode"] = thinking_mode
    if tools:
        kwargs["tools"] = tools

    started = time.perf_counter()
    result = backend.complete(**kwargs)
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
    image_count = len(sample_image_batch(image, max_image_frames)) if image is not None else 0
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


class MultimodalQwen38Loader(Qwen38ModelLoader):
    """Common-socket version of the existing Qwen3.8 loader."""

    RETURN_TYPES = ("MLLM_BACKEND", "STRING")
    RETURN_NAMES = ("backend", "backend_info")
    CATEGORY = "Multimodal LLM/Backends"


class MultimodalAPIBackend:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {"default": "https://api.openai.com/v1", "multiline": False},
                ),
                "model": ("STRING", {"default": "", "multiline": False}),
                "api_key": (
                    "STRING",
                    {"default": "", "multiline": False, "dynamicPrompts": False},
                ),
                "api_key_env": ("STRING", {"default": "OPENAI_API_KEY", "multiline": False}),
                "timeout_seconds": ("FLOAT", {"default": 120.0, "min": 1.0, "max": 3600.0, "step": 1.0}),
            },
            "optional": {
                "organization": ("STRING", {"default": "", "multiline": False}),
                "headers_json": ("STRING", {"default": "", "multiline": True}),
                "extra_body_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("MLLM_BACKEND", "STRING")
    RETURN_NAMES = ("backend", "backend_info")
    FUNCTION = "create_backend"
    CATEGORY = "Multimodal LLM/Backends"

    def create_backend(
        self,
        base_url: str,
        model: str,
        api_key: str,
        api_key_env: str,
        timeout_seconds: float,
        organization: str = "",
        headers_json: str = "",
        extra_body_json: str = "",
    ):
        backend = OpenAICompatibleBackend(
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
            model=model,
            timeout=timeout_seconds,
            organization=organization,
            headers_json=headers_json,
            extra_body_json=extra_body_json,
        )
        return backend, backend.info()


class MultimodalChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "backend": ("MLLM_BACKEND",),
                "prompt": (
                    "STRING",
                    {"default": "请详细描述输入的图片或视频。", "multiline": True},
                ),
                "system_prompt": (
                    "STRING",
                    {"default": "你是运行在 ComfyUI 内的专业视觉理解助手。请准确、直接地回答。", "multiline": True},
                ),
                "thinking_mode": (["backend_default", "thinking", "instruct"], {"default": "backend_default"}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 32768, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
                "max_image_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "max_image_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
                "max_video_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
                "video_transport": (["auto", "frames", "video_url"], {"default": "auto"}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_frames": ("IMAGE",),
                "video": ("VIDEO",),
                "tools_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "raw_response", "stats")
    FUNCTION = "generate"
    CATEGORY = "Multimodal LLM/Chat"
    OUTPUT_NODE = True

    def generate(
        self,
        backend: Any,
        prompt: str,
        system_prompt: str,
        thinking_mode: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repeat_penalty: float,
        seed: int,
        max_image_edge: int,
        max_image_frames: int,
        max_video_frames: int,
        video_transport: str,
        unload_after: bool,
        image: torch.Tensor | None = None,
        video_frames: torch.Tensor | None = None,
        video: Any | None = None,
        tools_json: str | None = None,
    ):
        if backend is None or not callable(getattr(backend, "complete", None)):
            raise BackendError("请连接本插件的本地模型加载器或 API Backend 节点。")
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
                max_image_edge,
                max_image_frames,
                max_video_frames,
                video_transport,
                thinking_mode,
                image,
                video_frames,
                video,
                tools_json,
            )
            print(f"[ComfyUI-Multimodal-LLM] Generation finished: {stats.splitlines()[0]}")
            return {
                "ui": {"text": (response,)},
                "result": (response, reasoning, raw, stats),
            }
        finally:
            if unload_after and callable(getattr(backend, "unload", None)):
                backend.unload()


class MultimodalUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"backend": ("MLLM_BACKEND",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Multimodal LLM/Backends"
    OUTPUT_NODE = True

    def unload(self, backend: Any):
        if callable(getattr(backend, "unload", None)):
            backend.unload()
        status = "Multimodal LLM backend unloaded"
        return {"ui": {"text": (status,)}, "result": (status,)}


NODE_CLASS_MAPPINGS = {
    "MultimodalQwen38Loader": MultimodalQwen38Loader,
    "MultimodalAPIBackend": MultimodalAPIBackend,
    "MultimodalChat": MultimodalChat,
    "MultimodalUnload": MultimodalUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultimodalQwen38Loader": "Multimodal Local Qwen3.8 Loader",
    "MultimodalAPIBackend": "Multimodal API Backend (RH/OpenAI Compatible)",
    "MultimodalChat": "Multimodal Chat",
    "MultimodalUnload": "Multimodal Backend Unload",
}
