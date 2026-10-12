"""Process-safe latest-value stream for Quest 3 operator-view RGB frames."""

from __future__ import annotations

from dataclasses import dataclass
import struct
import time
from typing import Any

import numpy as np


_VIEW_FRAME_HEADER = struct.Struct("!QdIII")


@dataclass(frozen=True)
class Quest3ViewFrame:
    """One RGB frame captured from the synchronized human-viewer camera."""

    sequence: int
    captured_timestamp_s: float
    rgb: np.ndarray

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("Quest 3 view-frame sequence must be positive.")
        if not np.isfinite(self.captured_timestamp_s):
            raise ValueError("Quest 3 view-frame timestamp must be finite.")
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3:
            raise ValueError(
                "Quest 3 view frame must have shape [H, W, 3], got "
                f"{self.rgb.shape}."
            )
        if self.rgb.dtype != np.uint8:
            raise TypeError(
                "Quest 3 view frame must use uint8 RGB, got "
                f"{self.rgb.dtype}."
            )


class Quest3SharedViewStream:
    """A process-safe single-slot RGB stream with consumer freshness state."""

    def __init__(
        self,
        buffer: Any,
        lock: Any,
        consumed_sequence: Any,
        consumed_timestamp_s: Any,
    ) -> None:
        if len(buffer) <= _VIEW_FRAME_HEADER.size:
            raise ValueError("Shared Quest 3 view-frame buffer is too small.")
        self._buffer = buffer
        self._lock = lock
        self._consumed_sequence = consumed_sequence
        self._consumed_timestamp_s = consumed_timestamp_s

    def publish(
        self,
        rgb: Any,
        *,
        captured_timestamp_s: float | None = None,
    ) -> Quest3ViewFrame:
        frame_rgb = normalize_view_rgb(rgb)
        timestamp_s = (
            time.time()
            if captured_timestamp_s is None
            else float(captured_timestamp_s)
        )
        if not np.isfinite(timestamp_s):
            raise ValueError("captured_timestamp_s must be finite.")
        height, width, channels = frame_rgb.shape
        payload = frame_rgb.tobytes(order="C")
        if len(payload) > len(self._buffer) - _VIEW_FRAME_HEADER.size:
            raise ValueError(
                "Quest 3 RGB frame exceeds the configured shared buffer: "
                f"shape={frame_rgb.shape}, bytes={len(payload)}, "
                f"capacity={len(self._buffer) - _VIEW_FRAME_HEADER.size}."
            )
        with self._lock:
            current_header = bytes(self._buffer[: _VIEW_FRAME_HEADER.size])
            current_sequence = _VIEW_FRAME_HEADER.unpack(current_header)[0]
            sequence = int(current_sequence) + 1
            start = _VIEW_FRAME_HEADER.size
            self._buffer[start : start + len(payload)] = payload
            self._buffer[:start] = _VIEW_FRAME_HEADER.pack(
                sequence,
                timestamp_s,
                height,
                width,
                channels,
            )
        return Quest3ViewFrame(sequence, timestamp_s, frame_rgb)

    def poll(self, *, after_sequence: int = 0) -> Quest3ViewFrame | None:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative.")
        with self._lock:
            header = bytes(self._buffer[: _VIEW_FRAME_HEADER.size])
            sequence, timestamp_s, height, width, channels = (
                _VIEW_FRAME_HEADER.unpack(header)
            )
            if sequence == 0 or sequence <= after_sequence:
                return None
            if height <= 0 or width <= 0 or channels != 3:
                raise RuntimeError(
                    "Corrupt Quest 3 view-frame shape: "
                    f"{height}x{width}x{channels}."
                )
            payload_length = int(height) * int(width) * int(channels)
            if payload_length > len(self._buffer) - _VIEW_FRAME_HEADER.size:
                raise RuntimeError("Corrupt Quest 3 view-frame payload length.")
            start = _VIEW_FRAME_HEADER.size
            # [H * W * 3] -> [H, W, 3]. Copy before releasing shared memory.
            rgb = np.frombuffer(
                bytes(self._buffer[start : start + payload_length]),
                dtype=np.uint8,
            ).reshape((height, width, channels))
        return Quest3ViewFrame(int(sequence), float(timestamp_s), rgb)

    def mark_consumed(self, sequence: int) -> None:
        if sequence <= 0:
            raise ValueError("Consumed Quest 3 view-frame sequence must be positive.")
        with self._lock:
            if sequence >= int(self._consumed_sequence.value):
                self._consumed_sequence.value = int(sequence)
                self._consumed_timestamp_s.value = time.monotonic()

    def is_consumed_fresh(self, *, max_age_s: float) -> bool:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be positive.")
        with self._lock:
            sequence = int(self._consumed_sequence.value)
            consumed_at_s = float(self._consumed_timestamp_s.value)
        return sequence > 0 and time.monotonic() - consumed_at_s <= max_age_s


def view_buffer_size(width: int, height: int) -> int:
    if width <= 0 or height <= 0:
        raise ValueError("Quest 3 view width and height must be positive.")
    return _VIEW_FRAME_HEADER.size + width * height * 3


def normalize_view_rgb(value: Any) -> np.ndarray:
    """Convert a ManiSkill RGB result to contiguous ``[H, W, 3]`` uint8."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                "Quest 3 operator view requires one environment, got RGB shape "
                f"{array.shape}."
            )
        # [1, H, W, 3] -> [H, W, 3].
        array = array[0]
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(
            "Quest 3 operator RGB must have shape [H, W, C>=3], got "
            f"{array.shape}."
        )
    # [H, W, C] -> [H, W, 3].
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("Quest 3 operator RGB contains NaN or Inf.")
        minimum = float(array.min(initial=0.0))
        maximum = float(array.max(initial=0.0))
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                "Floating-point Quest 3 operator RGB must lie in [0, 1], got "
                f"range [{minimum}, {maximum}]."
            )
        array = np.rint(array * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError(f"Unsupported Quest 3 operator RGB dtype {array.dtype}.")
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)
