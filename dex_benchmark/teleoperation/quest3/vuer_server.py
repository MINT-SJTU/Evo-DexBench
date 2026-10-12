from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
import time
from pathlib import Path
from threading import Thread
from typing import Any

from dex_benchmark.teleoperation.quest3.stream import (
    Quest3InputFrame,
    Quest3InputStream,
)
from dex_benchmark.teleoperation.quest3.view_stream import (
    Quest3SharedViewStream,
    Quest3ViewFrame,
)


@dataclass(frozen=True)
class Quest3VuerConfig:
    host: str = "0.0.0.0"
    port: int = 8012
    fps: int = 30
    hand_key: str = "quest3-hands"
    enable_left: bool = True
    enable_right: bool = True
    cert_path: str | Path | None = None
    key_path: str | Path | None = None
    view_enabled: bool = False
    view_fps: int = 30
    view_quality: int = 80
    view_display_scale: float = 0.6
    view_key: str = "quest3-operator-view"

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must not be empty.")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}.")
        if not 1 <= self.fps <= 60:
            raise ValueError(f"fps must be in [1, 60], got {self.fps}.")
        if not self.enable_left and not self.enable_right:
            raise ValueError("At least one Quest 3 hand stream must be enabled.")
        if (self.cert_path is None) != (self.key_path is None):
            raise ValueError("cert_path and key_path must be provided together.")
        if not 1 <= self.view_fps <= 60:
            raise ValueError(
                f"view_fps must be in [1, 60], got {self.view_fps}."
            )
        if not 1 <= self.view_quality <= 100:
            raise ValueError(
                f"view_quality must be in [1, 100], got {self.view_quality}."
            )
        if not math.isfinite(self.view_display_scale) or self.view_display_scale <= 0.0:
            raise ValueError(
                "view_display_scale must be a finite positive value, got "
                f"{self.view_display_scale}."
            )
        if not self.view_key:
            raise ValueError("view_key must not be empty.")


class Quest3VuerServer:
    """Vuer WebXR transport that publishes ``HAND_MOVE`` events to a stream."""

    def __init__(
        self,
        stream: Quest3InputStream,
        *,
        config: Quest3VuerConfig | None = None,
        view_stream: Quest3SharedViewStream | None = None,
        on_frame: Callable[[Quest3InputFrame], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.config = config or Quest3VuerConfig()
        try:
            from vuer import Vuer
            from vuer.schemas import Hands
            if self.config.view_enabled:
                from vuer.schemas import ImageBackground
            else:
                ImageBackground = None
        except ImportError as exc:
            raise ImportError(
                "Quest 3 input requires Vuer. Install it with "
                "`pip install -e '.[vr]'`."
            ) from exc

        self.stream = stream
        if self.config.view_enabled and view_stream is None:
            raise ValueError("view_stream is required when Quest view is enabled.")
        if not self.config.view_enabled and view_stream is not None:
            raise ValueError("view_stream requires Quest view to be enabled.")
        self.view_stream = view_stream
        self.on_frame = on_frame
        self.on_error = on_error
        self._hands_schema_type = Hands
        self._image_background_schema_type = ImageBackground
        self._last_payload_error_report_s = 0.0
        self._last_view_error_report_s = 0.0

        vuer_kwargs: dict[str, Any] = {
            "host": self.config.host,
            "port": self.config.port,
        }
        if self.config.cert_path is not None:
            cert_path = Path(self.config.cert_path).expanduser()
            assert self.config.key_path is not None
            key_path = Path(self.config.key_path).expanduser()
            if not cert_path.is_file():
                raise FileNotFoundError(f"TLS certificate does not exist: {cert_path}")
            if not key_path.is_file():
                raise FileNotFoundError(f"TLS private key does not exist: {key_path}")
            vuer_kwargs.update(cert=str(cert_path), key=str(key_path))

        self.app = Vuer(**vuer_kwargs)
        self._server_thread: Thread | None = None
        self.app.add_handler("HAND_MOVE")(self._handle_hand_move)
        self.app.spawn()(self._setup_session)

    async def _handle_hand_move(self, event: Any, _session: Any) -> None:
        # This is the earliest timestamp exposed by Vuer: HAND_MOVE handler
        # entry. It is not the lower-level WebSocket/kernel receive time.
        received_timestamp_s = time.time()
        if getattr(event, "key", None) != self.config.hand_key:
            return
        value = getattr(event, "value", None)
        if not isinstance(value, Mapping):
            self._report_error(
                TypeError(
                    "Quest 3 HAND_MOVE event.value must be a mapping, "
                    f"got {type(value).__name__}."
                )
            )
            return
        try:
            frame = self.stream.update(
                value,
                source_timestamp_s=getattr(event, "ts", None),
                received_timestamp_s=received_timestamp_s,
            )
        except (TypeError, ValueError) as exc:
            now_s = time.monotonic()
            if now_s - self._last_payload_error_report_s >= 1.0:
                self._last_payload_error_report_s = now_s
                print(
                    "[quest3] HAND_MOVE payload: "
                    f"left={_describe_payload(value.get('left'))}, "
                    f"right={_describe_payload(value.get('right'))}",
                    flush=True,
                )
            self._report_error(exc)
            return
        if self.on_frame is not None:
            self.on_frame(frame)

    async def _setup_session(self, session: Any) -> None:
        # The Vuer client creates this infinite grid before the server scene
        # arrives. It is only a visual aid and is not needed for hand tracking.
        session.remove("grid")
        session.upsert(
            self._hands_schema_type(
                stream=True,
                key=self.config.hand_key,
                fps=self.config.fps,
                disableLeft=not self.config.enable_left,
                disableRight=not self.config.enable_right,
            ),
            to="bgChildren",
        )
        view_task: asyncio.Task[None] | None = None
        if getattr(self.config, "view_enabled", False):
            view_task = asyncio.create_task(self._publish_view_frames(session))
        try:
            forever = getattr(session, "forever", None)
            if forever is not None:
                await forever()
                return
            while True:
                await asyncio.sleep(3600)
        finally:
            if view_task is not None:
                view_task.cancel()
                await asyncio.gather(view_task, return_exceptions=True)

    async def _publish_view_frames(self, session: Any) -> None:
        assert self.view_stream is not None
        assert self._image_background_schema_type is not None
        last_sequence = 0
        poll_interval_s = 0.5 / float(self.config.view_fps)
        while True:
            frame = self.view_stream.poll(after_sequence=last_sequence)
            if frame is None:
                await asyncio.sleep(poll_interval_s)
                continue
            last_sequence = frame.sequence
            try:
                background = await asyncio.to_thread(
                    self._build_view_background,
                    frame,
                )
                session.upsert(background, to="bgChildren")
                self.view_stream.mark_consumed(frame.sequence)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                now_s = time.monotonic()
                if now_s - self._last_view_error_report_s >= 1.0:
                    self._last_view_error_report_s = now_s
                    self._report_error(exc)

    def _build_view_background(self, frame: Quest3ViewFrame) -> Any:
        assert self._image_background_schema_type is not None
        height, width, _channels = frame.rgb.shape
        return self._image_background_schema_type(
            frame.rgb,
            aspect=float(width) / float(height),
            height=self.config.view_display_scale,
            distanceToCamera=1.0,
            format="jpeg",
            quality=self.config.view_quality,
            key=self.config.view_key,
            interpolate=True,
        )

    def run(self) -> None:
        """Start the Vuer server and block until it is interrupted."""

        self.app.start()

    def start_background(self) -> Thread:
        """Start Vuer on a daemon thread so ManiSkill can own the main thread."""

        if self._server_thread is not None and self._server_thread.is_alive():
            raise RuntimeError("Quest 3 Vuer server is already running.")
        self._server_thread = Thread(
            target=self.run,
            name="quest3-vuer-server",
            daemon=True,
        )
        self._server_thread.start()
        return self._server_thread

    def _report_error(self, error: Exception) -> None:
        if self.on_error is not None:
            self.on_error(error)
            return
        print(f"[quest3] Dropped invalid HAND_MOVE frame: {error}", flush=True)


def _describe_payload(value: Any) -> str:
    """Describe payload structure without logging hand-tracking values."""

    if isinstance(value, Mapping):
        key_types = sorted(f"{key}:{type(item).__name__}" for key, item in value.items())
        return f"{type(value).__name__}(keys={key_types[:8]})"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, (list, tuple)):
        item_types = sorted({type(item).__name__ for item in value[:32]})
        return f"{type(value).__name__}(len={len(value)}, item_types={item_types})"
    return type(value).__name__
