from __future__ import annotations

from collections.abc import Sequence
import math
from pathlib import Path
from typing import Any, Literal

import numpy as np


VideoMode = Literal["grid", "individual", "both"]


def rgb_frame(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    frame = np.asarray(value)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"Expected an HWC RGB/RGBA frame, got {frame.shape}.")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if not np.isfinite(frame).all() or frame.min() < 0 or frame.max() > 1:
            raise ValueError(
                "Floating-point video frames must be finite and in [0, 1]."
            )
        frame = np.rint(frame * 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        raise TypeError(
            f"Video frames must be uint8 or float in [0, 1], got {frame.dtype}."
        )
    return np.ascontiguousarray(frame)


def compose_grid(frames: Sequence[Any], *, slots: int) -> np.ndarray:
    """Arrange up to ``slots`` equal-sized RGB frames in a compact grid."""

    if slots <= 0:
        raise ValueError("slots must be positive.")
    converted = [rgb_frame(frame) for frame in frames]
    if not converted:
        raise ValueError("At least one frame is required to determine grid size.")
    height, width, channels = converted[0].shape
    if any(frame.shape != (height, width, channels) for frame in converted):
        raise ValueError("All grid frames must have the same shape.")
    if len(converted) > slots:
        raise ValueError(f"Received {len(converted)} frames for {slots} grid slots.")

    columns = math.ceil(math.sqrt(slots))
    rows = math.ceil(slots / columns)
    canvas = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, frame in enumerate(converted):
        row, column = divmod(index, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = frame
    return canvas


class ParallelVideoRecorder:
    """Write per-episode videos and/or one grid video across rollout waves."""

    def __init__(
        self,
        path: Path,
        *,
        fps: int,
        mode: VideoMode,
        num_slots: int,
    ) -> None:
        if fps <= 0:
            raise ValueError("video fps must be positive.")
        if mode not in ("grid", "individual", "both"):
            raise ValueError(f"Unsupported video mode {mode!r}.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise ImportError(
                "Video recording requires imageio and imageio-ffmpeg."
            ) from exc

        self.base_path = path
        self.fps = fps
        self.mode = mode
        self.num_slots = num_slots
        self._imageio = imageio
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        self.grid_path = (
            path if mode == "grid" else path.with_name(f"{path.stem}_grid{path.suffix}")
        )
        self._grid_writer = (
            self._make_writer(self.grid_path) if mode in ("grid", "both") else None
        )
        self._individual_writers: list[Any] = []
        self._active_count = 0
        self.individual_paths: list[Path] = []

    def _make_writer(self, path: Path):
        return self._imageio.get_writer(
            path,
            fps=self.fps,
            codec="libx264",
            quality=8,
        )

    def begin_wave(self, episode_ids: Sequence[int]) -> None:
        if self._individual_writers:
            raise RuntimeError("The previous video wave has not been closed.")
        episode_rows = tuple(episode_ids)
        self._active_count = len(episode_rows)
        if not 1 <= self._active_count <= self.num_slots:
            raise ValueError(
                "A video wave must contain between one and num_slots episodes."
            )
        if self.mode not in ("individual", "both"):
            return
        for episode_id in episode_rows:
            path = self.base_path.with_name(
                f"{self.base_path.stem}_episode{int(episode_id):04d}{self.base_path.suffix}"
            )
            self.individual_paths.append(path)
            self._individual_writers.append(self._make_writer(path))

    def capture(self, frames: Sequence[Any], _step: int) -> None:
        converted = [rgb_frame(frame) for frame in frames]
        if len(converted) != self.num_slots:
            raise ValueError(
                f"Expected {self.num_slots} rendered frames, got {len(converted)}."
            )
        if self._grid_writer is not None:
            self._grid_writer.append_data(
                compose_grid(converted[: self._active_count], slots=self.num_slots)
            )
        if self._individual_writers:
            for writer, frame in zip(
                self._individual_writers,
                converted[: self._active_count],
                strict=True,
            ):
                writer.append_data(frame)

    def end_wave(self) -> None:
        for writer in self._individual_writers:
            writer.close()
        self._individual_writers.clear()
        self._active_count = 0

    @property
    def paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        if self.mode in ("grid", "both"):
            paths.append(self.grid_path)
        paths.extend(self.individual_paths)
        return tuple(paths)

    def close(self) -> None:
        self.end_wave()
        if self._grid_writer is not None:
            self._grid_writer.close()
            self._grid_writer = None
