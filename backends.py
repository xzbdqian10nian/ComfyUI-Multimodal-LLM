"""Backend adapters for the ComfyUI Multimodal LLM plugin.

The node layer deliberately talks to a very small interface (``complete`` and
``unload``).  This keeps local llama.cpp models and OpenAI-compatible APIs
interchangeable inside the same workflow.
"""

from __future__ import annotations

import gc
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def _free_comfy_vram() -> None:
    """Release ComfyUI models before loading a large local VLM."""
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:
        print(f"[ComfyUI-Multimodal-LLM] ComfyUI VRAM cleanup skipped: {exc}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BackendError(RuntimeError):
    """A user-facing backend configuration or request error."""


@dataclass(frozen=True)
class LocalRuntimeSettings:
    model_path: Path
    mmproj_path: Path
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    thinking: bool
    free_comfy_vram: bool


class LocalQwen38Backend:
    """Qwen3.8 GGUF + multimodal projector through CUDA llama.cpp."""

    backend_kind = "local_llama_cpp"

    def __init__(self, settings: LocalRuntimeSettings):
        self.settings = settings
        self.llm = None
        self.chat_handler = None
        self._lock = threading.RLock()
        self.load_seconds = 0.0

    def _make_handler(self):
        try:
            from llama_cpp.llama_chat_format import Qwen35ChatHandler
        except Exception as exc:
            raise BackendError(
                "当前 llama-cpp-python 不包含 Qwen35ChatHandler，无法加载视觉模型。"
            ) from exc

        return Qwen35ChatHandler(
            mmproj_path=str(self.settings.mmproj_path),
            enable_thinking=self.settings.thinking,
            preserve_thinking=False,
            add_vision_id=True,
            image_min_tokens=256,
            image_max_tokens=4096,
            batch_max_tokens=1024,
            use_gpu=self.settings.n_gpu_layers != 0,
            verbose=False,
        )

    def ensure_loaded(self) -> None:
        with self._lock:
            if self.llm is not None:
                return
            if self.settings.free_comfy_vram:
                _free_comfy_vram()

            try:
                from llama_cpp import Llama
            except Exception as exc:
                raise BackendError(
                    "未安装视觉版 llama-cpp-python；请使用当前镜像中已有的 CUDA wheel。"
                ) from exc

            started = time.perf_counter()
            self.chat_handler = self._make_handler()
            print(
                "[ComfyUI-Multimodal-LLM] Loading "
                f"{self.settings.model_path.name}, ctx={self.settings.n_ctx}, "
                f"gpu_layers={self.settings.n_gpu_layers}"
            )
            self.llm = Llama(
                model_path=str(self.settings.model_path),
                n_ctx=int(self.settings.n_ctx),
                n_batch=int(self.settings.n_batch),
                n_ubatch=int(self.settings.n_ubatch),
                n_gpu_layers=int(self.settings.n_gpu_layers),
                chat_handler=self.chat_handler,
                use_mmap=True,
                use_mlock=False,
                offload_kqv=True,
                no_perf=False,
                verbose=False,
            )
            self.load_seconds = time.perf_counter() - started
            print(f"[ComfyUI-Multimodal-LLM] Local model loaded in {self.load_seconds:.1f}s")

    def complete(self, **kwargs):
        with self._lock:
            self.ensure_loaded()
            return self.llm.create_chat_completion(**kwargs)

    def info(self) -> str:
        return (
            f"backend=local_llama_cpp\nmodel={self.settings.model_path.name}\n"
            f"mmproj={self.settings.mmproj_path.name}\nctx={self.settings.n_ctx}\n"
            f"gpu_layers={self.settings.n_gpu_layers}\n"
            f"thinking={self.settings.thinking}\nload_seconds={self.load_seconds:.1f}"
        )

    def unload(self) -> None:
        with self._lock:
            if self.llm is not None:
                try:
                    self.llm.close()
                except Exception:
                    pass
            self.llm = None
            if self.chat_handler is not None:
                try:
                    self.chat_handler.close()
                except Exception:
                    pass
            self.chat_handler = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[ComfyUI-Multimodal-LLM] Local model unloaded")

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


def _parse_json_object(value: str, field_name: str) -> dict[str, Any]:
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BackendError(f"{field_name} 不是有效 JSON：{exc}") from exc
    if not isinstance(parsed, dict):
        raise BackendError(f"{field_name} 必须是 JSON 对象。")
    return parsed


class OpenAICompatibleBackend:
    """Lazy OpenAI-compatible chat-completions client.

    This follows the same contract as the RunningHub RH LLM API node, while
    keeping the client creation lazy so merely loading ComfyUI never makes a
    network request.
    """

    backend_kind = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        api_key_env: str,
        model: str,
        timeout: float,
        organization: str = "",
        headers_json: str = "",
        extra_body_json: str = "",
    ):
        self.base_url = base_url.strip().rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")].rstrip("/")
        self.api_key = api_key.strip() or os.getenv(api_key_env.strip(), "")
        self.api_key_env = api_key_env.strip()
        self.model = model.strip()
        self.timeout = max(1.0, float(timeout))
        self.organization = organization.strip()
        self.headers = _parse_json_object(headers_json, "headers_json")
        self.extra_body = _parse_json_object(extra_body_json, "extra_body_json")
        self.client = None
        self._lock = threading.RLock()

        if not self.base_url:
            raise BackendError("API base URL 不能为空。")
        if not self.model:
            raise BackendError("API model 不能为空。")

    def _ensure_client(self):
        if self.client is not None:
            return self.client
        try:
            from openai import OpenAI
        except Exception as exc:
            raise BackendError(
                "当前环境没有 openai 包；RH API 节点使用的 OpenAI SDK 应已在镜像中提供。"
            ) from exc

        # Some local OpenAI-compatible servers do not require a key.  The SDK
        # still requires a non-empty value, so use a harmless placeholder.
        kwargs: dict[str, Any] = {
            "api_key": self.api_key or "EMPTY",
            "base_url": self.base_url,
            "timeout": self.timeout,
        }
        if self.organization:
            kwargs["organization"] = self.organization
        if self.headers:
            kwargs["default_headers"] = self.headers
        self.client = OpenAI(**kwargs)
        return self.client

    def complete(self, **kwargs):
        with self._lock:
            request: dict[str, Any] = {
                "model": self.model,
                "messages": kwargs["messages"],
                "max_tokens": int(kwargs.get("max_tokens", 1024)),
                "temperature": float(kwargs.get("temperature", 0.6)),
                "top_p": float(kwargs.get("top_p", 0.95)),
            }
            seed = kwargs.get("seed")
            if seed is not None and int(seed) >= 0:
                request["seed"] = int(seed)
            tools = kwargs.get("tools")
            if tools:
                request["tools"] = tools
            extra_body = dict(self.extra_body)
            thinking_mode = kwargs.get("thinking_mode", "backend_default")
            if thinking_mode in {"thinking", "instruct"}:
                # This is understood by common vLLM/SGLang/Qwen-compatible
                # servers.  It is sent as an extra body field only when the
                # user explicitly chooses a mode, so normal OpenAI requests
                # remain strictly standard.
                extra_body.setdefault("enable_thinking", thinking_mode == "thinking")

            if extra_body:
                request["extra_body"] = extra_body

            try:
                client = self._ensure_client()
            except BackendError as exc:
                if "没有 openai 包" not in str(exc):
                    raise
                return self._complete_with_urllib(request, extra_body)

            try:
                return client.chat.completions.create(**request)
            except Exception as exc:
                # Do not include the API key in the error text.
                raise BackendError(f"OpenAI-compatible API 请求失败：{exc}") from exc

    def _complete_with_urllib(self, request: dict[str, Any], extra_body: dict[str, Any]):
        """Small stdlib fallback for images where the OpenAI SDK is absent."""
        body = dict(request)
        body.pop("extra_body", None)
        body.update(extra_body)
        endpoint = self.base_url + "/chat/completions"
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise BackendError(f"OpenAI-compatible API 返回 HTTP {exc.code}: {detail}") from exc
        except Exception as exc:
            raise BackendError(f"OpenAI-compatible API 请求失败：{exc}") from exc

    def info(self) -> str:
        key_source = "direct" if self.api_key else (self.api_key_env or "none")
        return (
            f"backend=openai_compatible\nbase_url={self.base_url}\n"
            f"model={self.model}\ntimeout={self.timeout:.1f}s\nkey_source={key_source}"
        )

    def unload(self) -> None:
        with self._lock:
            if self.client is not None:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None
