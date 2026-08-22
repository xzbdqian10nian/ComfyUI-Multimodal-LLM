"""Model discovery helpers for the ComfyUI Multimodal LLM nodes.

The visible ComfyUI nodes live in :mod:`generic_nodes`. This module only
handles model catalogues and safe path resolution; it does not register a
second set of nodes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import folder_paths
except Exception:  # pragma: no cover - only used by standalone tests
    folder_paths = None


PLUGIN_DIR = Path(__file__).resolve().parent
CATALOG_PATH = PLUGIN_DIR / "models.json"
DEFAULT_MODEL_SUBDIR = Path("LLM/Qwen3.8")
DEFAULT_MODEL = "Qwen3.8-27B-UD-Q4_K_M.gguf"
DEFAULT_MMPROJ = "mmproj-BF16.gguf"


def _load_catalog() -> dict[str, Any]:
    try:
        return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


CATALOG = _load_catalog()


def _model_root() -> Path:
    configured = os.environ.get("QWEN38_MODEL_DIR")
    if configured:
        return Path(configured).expanduser()
    if folder_paths is not None:
        return Path(folder_paths.models_dir) / CATALOG.get("model_subdir", DEFAULT_MODEL_SUBDIR)
    # Standalone fallback: custom_nodes/<plugin> is two levels below the
    # ComfyUI root, so keep the same portable models/ layout without knowing
    # anything about a platform-specific storage mount.
    return PLUGIN_DIR.parents[1] / "models" / CATALOG.get("model_subdir", DEFAULT_MODEL_SUBDIR)


def _is_complete_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024 * 1024 and not Path(f"{path}.aria2").exists()


def _choices(kind: str) -> list[str]:
    """Return available GGUF filenames, with the catalogue default first."""
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
    """Resolve a selected model file and reject incomplete downloads."""
    if not filename or filename.startswith("("):
        raise FileNotFoundError(f"No {kind} file selected")
    path = Path(filename).expanduser()
    if not path.is_absolute():
        path = _model_root() / path
    if Path(f"{path}.aria2").exists():
        raise RuntimeError(f"{kind} is still downloading: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{kind} not found: {path}")
    # Keep ComfyUI's configured path instead of exposing the target of a
    # platform-specific storage symlink in errors and status messages.
    return path.absolute()


__all__ = ["_choices", "_resolve_file"]
