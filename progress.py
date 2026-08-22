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


def send_status(
    text: str,
    node_id: str | int | None = None,
    *,
    mirror_log: bool = True,
) -> None:
    """Send a visible node status message and optionally mirror it to the log.

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
    if mirror_log:
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

    def send(self, text: str, force: bool = False, *, mirror_log: bool = True) -> None:
        now = time.monotonic()
        if force or now - self.last >= self.interval:
            send_status(text, self.node_id, mirror_log=mirror_log)
            self.last = now


class ConsoleProgressBar:
    """Render one compact progress line in ComfyUI's server log.

    ComfyUI's native ``ProgressBar`` drives the browser overlay but is silent
    in the terminal.  This companion uses carriage returns, so cloud live logs
    show one updating bar instead of one new line per streamed chunk.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        width: int = 24,
        interval: float = 0.5,
    ):
        self.label = str(label)
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.interval = max(0.0, float(interval))
        self.last_time = 0.0
        self.last_value = -1
        self.closed = False

    def update(
        self,
        value: int,
        *,
        suffix: str = "",
        force: bool = False,
        complete: bool = False,
    ) -> None:
        if self.closed:
            return
        value = max(0, min(int(value), self.total))
        now = time.monotonic()
        if not force and not complete:
            if value == self.last_value or now - self.last_time < self.interval:
                return
        ratio = value / self.total
        filled = min(self.width, round(ratio * self.width))
        bar = "█" * filled + "░" * (self.width - filled)
        detail = f" · {suffix}" if suffix else ""
        print(
            f"[ComfyUI-Multimodal-LLM] {self.label}: "
            f"{ratio * 100:5.1f}%|{bar}| {value}/{self.total}{detail}",
            end="\n" if complete else "\r",
            flush=True,
        )
        self.last_value = value
        self.last_time = now
        if complete:
            self.closed = True

    def finish(self, suffix: str = "complete") -> None:
        self.update(self.total, suffix=suffix, force=True, complete=True)

    def close(self, suffix: str = "stopped") -> None:
        """Terminate an unfinished carriage-return line after an exception."""
        if not self.closed and self.last_value >= 0:
            self.update(self.last_value, suffix=suffix, force=True, complete=True)
