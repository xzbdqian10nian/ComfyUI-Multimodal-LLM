from __future__ import annotations

import base64
import gc
import io
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
except Exception:  # pragma: no cover - only used by standalone tests
    folder_paths = None


PLUGIN_DIR = Path(__file__).resolve().parent
CATALOG_PATH = PLUGIN_DIR / "models.json"
DEFAULT_POD_MODEL_ROOT = Path("/poddata/ComfyUI/models/LLM/Qwen3.8")
DEFAULT_MODEL = "Qwen3.8-27B-UD-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-BF16.gguf"


def _load_catalog() -> dict[str, Any]:
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


CATALOG = _load_catalog()


def _model_root() -> Path:
    configured = os.environ.get("QWEN38_MODEL_DIR") or CATALOG.get("model_root")
    if configured:
        return Path(configured).expanduser()
    if DEFAULT_POD_MODEL_ROOT.exists() or Path("/poddata").exists():
        return DEFAULT_POD_MODEL_ROOT
    if folder_paths is not None:
        return Path(folder_paths.models_dir) / "LLM" / "Qwen3.8"
    return DEFAULT_POD_MODEL_ROOT


def _is_complete_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024 * 1024 and not Path(f"{path}.aria2").exists()


def _choices(kind: str) -> list[str]:
    root = _model_root()
    pattern = "*mmproj*.gguf" if kind == "mmproj" else "*.gguf"
    found: list[str] = []
    if root.exists():
        for path in root.glob(pattern):
            if _is_complete_file(path):
                if kind == "model" and "mmproj" in path.name.lower():
                    continue
                found.append(path.name)

    preferred = DEFAULT_MMPROJ if kind == "mmproj" else DEFAULT_MODEL
    configured = CATALOG.get("default_mmproj" if kind == "mmproj" else "default_model", preferred)
    ordered = sorted(set(found), key=lambda name: (name != configured, name.lower()))
    return ordered or [configured]


def _resolve_file(filename: str, kind: str) -> Path:
    if not filename or filename.startswith("("):
        raise FileNotFoundError(f"No {kind} file selected")
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = _model_root() / path
    if Path(f"{path}.aria2").exists():
        raise RuntimeError(f"{kind} is still downloading: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path.resolve()


def _free_comfy_vram() -> None:
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception as exc:
        print(f"[Qwen3.8 Vision] ComfyUI VRAM cleanup skipped: {exc}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tensor_frame_to_pil(frame: torch.Tensor, max_edge: int) -> Image.Image:
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3:
        raise ValueError(f"IMAGE must be HWC or BHWC, got shape {tuple(frame.shape)}")
    array = (frame.detach().float().clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.shape[-1] != 3:
        raise ValueError(f"IMAGE needs 3 RGB channels, got shape {array.shape}")
    image = Image.fromarray(array, mode="RGB")
    if max_edge > 0 and max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def _pil_to_data_url(image: Image.Image, quality: int = 90) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality), optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _sample_batch(batch: torch.Tensor | None, count: int) -> list[torch.Tensor]:
    if batch is None:
        return []
    if batch.ndim == 3:
        return [batch]
    if batch.ndim != 4:
        raise ValueError(f"Expected IMAGE batch in BHWC format, got {tuple(batch.shape)}")
    total = int(batch.shape[0])
    if total <= count:
        return [batch[i] for i in range(total)]
    indices = np.linspace(0, total - 1, count, dtype=int)
    return [batch[int(i)] for i in indices]


def _split_reasoning(message: dict[str, Any]) -> tuple[str, str, str]:
    content = str(message.get("content") or "")
    reasoning = str(message.get("reasoning_content") or "")
    raw = content
    if not reasoning and "<think>" in content:
        before, _, tail = content.partition("<think>")
        thought, marker, after = tail.partition("</think>")
        if marker:
            reasoning = thought.strip()
            content = (before + after).strip()
    content = content.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
    reasoning = reasoning.strip()
    return content, reasoning, raw


@dataclass(frozen=True)
class RuntimeSettings:
    model_path: Path
    mmproj_path: Path
    n_ctx: int
    n_batch: int
    n_ubatch: int
    n_gpu_layers: int
    thinking: bool
    free_comfy_vram: bool


class Qwen38Runtime:
    """Owns one llama.cpp model and its multimodal projector."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.llm = None
        self.chat_handler = None
        self._lock = threading.RLock()
        self.load_seconds = 0.0

    def _make_handler(self):
        try:
            from llama_cpp.llama_chat_format import Qwen35ChatHandler
        except Exception as exc:
            raise RuntimeError(
                "llama-cpp-python lacks Qwen35ChatHandler; install a recent vision-capable build"
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
                raise RuntimeError(
                    "llama-cpp-python is not installed. This plugin needs the CUDA vision wheel."
                ) from exc

            started = time.perf_counter()
            self.chat_handler = self._make_handler()
            print(
                "[Qwen3.8 Vision] Loading "
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
            print(f"[Qwen3.8 Vision] Model loaded in {self.load_seconds:.1f}s")

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
            print("[Qwen3.8 Vision] Model unloaded")

    def complete(self, **kwargs):
        with self._lock:
            self.ensure_loaded()
            return self.llm.create_chat_completion(**kwargs)

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass


class Qwen38ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_file": (_choices("model"), {"default": _choices("model")[0]}),
                "mmproj_file": (_choices("mmproj"), {"default": _choices("mmproj")[0]}),
                "thinking_mode": (["thinking", "instruct"], {"default": "thinking"}),
                "context_length": ("INT", {"default": 8192, "min": 2048, "max": 262144, "step": 1024}),
                "batch_size": ("INT", {"default": 1024, "min": 128, "max": 8192, "step": 128}),
                "micro_batch_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 256}),
                "free_comfy_vram": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("QWEN38_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load_model"
    CATEGORY = "Qwen3.8/Vision LLM"

    def load_model(
        self,
        model_file: str,
        mmproj_file: str,
        thinking_mode: str,
        context_length: int,
        batch_size: int,
        micro_batch_size: int,
        gpu_layers: int,
        free_comfy_vram: bool,
    ):
        model_path = _resolve_file(model_file, "model")
        mmproj_path = _resolve_file(mmproj_file, "mmproj")
        settings = RuntimeSettings(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_ctx=int(context_length),
            n_batch=int(batch_size),
            n_ubatch=int(micro_batch_size),
            n_gpu_layers=int(gpu_layers),
            thinking=thinking_mode == "thinking",
            free_comfy_vram=bool(free_comfy_vram),
        )
        runtime = Qwen38Runtime(settings)
        runtime.ensure_loaded()
        info = (
            f"model={model_path.name}\nmmproj={mmproj_path.name}\n"
            f"mode={thinking_mode}\nctx={context_length}\n"
            f"gpu_layers={gpu_layers}\nload_seconds={runtime.load_seconds:.1f}"
        )
        return runtime, info


class Qwen38VisionChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("QWEN38_MODEL",),
                "prompt": (
                    "STRING",
                    {"default": "请详细描述画面，并指出主体、动作、环境、构图与光线。", "multiline": True},
                ),
                "system_prompt": (
                    "STRING",
                    {"default": "你是运行在 ComfyUI 内的专业视觉理解助手。请准确、直接地回答。", "multiline": True},
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 32768, "step": 16}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 200}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.05, "min": 0.5, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 2**32 - 1}),
                "max_image_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "max_video_frames": ("INT", {"default": 8, "min": 1, "max": 64}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
                "tools_json": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "reasoning", "raw_response", "stats")
    FUNCTION = "generate"
    CATEGORY = "Qwen3.8/Vision LLM"
    OUTPUT_NODE = True

    def generate(
        self,
        model: Qwen38Runtime,
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
        max_video_frames: int,
        unload_after: bool,
        image: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        tools_json: str | None = None,
    ):
        content: list[dict[str, Any]] = []
        image_frames = _sample_batch(image, max(1, int(image.shape[0]) if image is not None and image.ndim == 4 else 1))
        video_frames = _sample_batch(video, int(max_video_frames))

        for index, frame in enumerate(image_frames, 1):
            if len(image_frames) > 1:
                content.append({"type": "text", "text": f"Picture {index}:"})
            pil = _tensor_frame_to_pil(frame, int(max_image_edge))
            content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(pil)}})

        for index, frame in enumerate(video_frames, 1):
            content.append({"type": "text", "text": f"Video frame {index}/{len(video_frames)}:"})
            pil = _tensor_frame_to_pil(frame, int(max_image_edge))
            content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(pil, 85)}})

        content.append({"type": "text", "text": prompt.strip()})
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": content},
        ]

        completion_kwargs: dict[str, Any] = {
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
        if tools_json and tools_json.strip():
            try:
                tools = json.loads(tools_json)
            except json.JSONDecodeError as exc:
                raise ValueError(f"tools_json is invalid JSON: {exc}") from exc
            completion_kwargs["tools"] = tools if isinstance(tools, list) else [tools]

        started = time.perf_counter()
        try:
            result = model.complete(**completion_kwargs)
            elapsed = time.perf_counter() - started
            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            response, reasoning, raw = _split_reasoning(message)
            tool_calls = message.get("tool_calls")
            if tool_calls and not response:
                response = json.dumps(tool_calls, ensure_ascii=False, indent=2)
            if tool_calls:
                raw = json.dumps(message, ensure_ascii=False, indent=2)

            usage = result.get("usage") or {}
            completion_tokens = int(usage.get("completion_tokens") or 0)
            speed = completion_tokens / elapsed if completion_tokens and elapsed > 0 else 0.0
            stats = (
                f"time={elapsed:.2f}s\n"
                f"prompt_tokens={usage.get('prompt_tokens', 'n/a')}\n"
                f"completion_tokens={usage.get('completion_tokens', 'n/a')}\n"
                f"speed={speed:.2f} tok/s\n"
                f"images={len(image_frames)}\nvideo_frames={len(video_frames)}"
            )
            print(f"[Qwen3.8 Vision] Generation finished: {elapsed:.2f}s, {speed:.2f} tok/s")
            # Keep the normal tuple outputs so the response can be wired into
            # any STRING node, and also expose the result in ComfyUI's output
            # panel.  OUTPUT_NODE nodes that return only a tuple execute
            # successfully but have no history/UI payload to display.
            return {
                "ui": {
                    "text": (response,),
                    "reasoning": (reasoning,),
                    "raw_response": (raw,),
                    "stats": (stats,),
                },
                "result": (response, reasoning, raw, stats),
            }
        finally:
            if unload_after:
                model.unload()


class Qwen38Unload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("QWEN38_MODEL",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "unload"
    CATEGORY = "Qwen3.8/Vision LLM"
    OUTPUT_NODE = True

    def unload(self, model: Qwen38Runtime):
        model.unload()
        status = "Qwen3.8 model unloaded"
        return {"ui": {"text": (status,)}, "result": (status,)}


NODE_CLASS_MAPPINGS = {
    "Qwen38ModelLoader": Qwen38ModelLoader,
    "Qwen38VisionChat": Qwen38VisionChat,
    "Qwen38Unload": Qwen38Unload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen38ModelLoader": "Qwen3.8 Model Loader (GGUF)",
    "Qwen38VisionChat": "Qwen3.8 Vision Chat",
    "Qwen38Unload": "Qwen3.8 Unload Model",
}
