"""Small ComfyUI progress/status compatibility helpers.

The plugin is also importable outside ComfyUI for smoke tests, so none of the
helpers in this module require ComfyUI to be installed.  When ComfyUI is
present they use its normal progress websocket and text-event APIs.
"""

from __future__ import annotations

import time
from typing import Any


class _NullProgress:
    def update_absolute(self, value: int, total: int | None = None, preview: Any = None):
        return None

    def update(self, value: int):
        return None


def make_progress(total: int, node_id: str | int | None = None):
    """Create a native ComfyUI progress bar across old/new ComfyUI builds."""
    try:
        from comfy.utils import ProgressBar

        try:
            return ProgressBar(int(total), node_id=str(node_id) if node_id is not None else None)
        except TypeError:  # Older ComfyUI only accepts ``total``.
            return ProgressBar(int(total))
    except Exception:
        return _NullProgress()


def send_status(text: str, node_id: str | int | None = None) -> None:
    """Send a visible node status message and always mirror it to the log.

    ``PromptServer.send_progress_text`` normally succeeds, which used to make
    this function return before printing anything.  That made progress visible
    in the browser (on supported frontends) but invisible in ``comfyUI.log``.
    Keep both channels independent: a websocket/UI failure must not suppress
    the server-side progress log.
    """
    message = str(text)
    delivered_to_ui = False
    if node_id is not None:
        try:
            from server import PromptServer

            instance = getattr(PromptServer, "instance", None)
            sender = getattr(instance, "send_progress_text", None)
            if callable(sender):
                sender(message, str(node_id))
                delivered_to_ui = True
        except Exception:
            # Status reporting must never make model inference fail.
            pass
    node_suffix = f" node={node_id}" if node_id is not None else ""
    channel = "ui+log" if delivered_to_ui else "log"
    print(f"[ComfyUI-Multimodal-LLM][{channel}]{node_suffix} {message}", flush=True)


def update_progress(progress, value: int, total: int | None = None) -> None:
    """Update a progress object without making it part of the hot path."""
    try:
        progress.update_absolute(int(value), total=total)
    except TypeError:
        try:
            progress.update_absolute(int(value))
        except Exception:
            pass
    except Exception:
        pass


class StatusTicker:
    """Throttle text events while a streamed completion is arriving."""

    def __init__(self, node_id: str | int | None, interval: float = 0.5):
        self.node_id = node_id
        self.interval = float(interval)
        self.last = 0.0

    def send(self, text: str, force: bool = False) -> None:
        now = time.monotonic()
        if force or now - self.last >= self.interval:
            send_status(text, self.node_id)
            self.last = now
