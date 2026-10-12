from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import queue
import signal
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

from dex_benchmark.teleoperation.quest3.stream import Quest3InputFrame, Quest3InputStream
from dex_benchmark.teleoperation.quest3.view_stream import (
    Quest3SharedViewStream,
    view_buffer_size,
)
from dex_benchmark.teleoperation.quest3.vuer_server import Quest3VuerConfig, Quest3VuerServer


_FRAME_LENGTH = struct.Struct("!Q")
_FRAME_BUFFER_SIZE = 1 << 20


def _set_parent_death_signal(parent_pid: int) -> None:
    """Terminate the Vuer child if its simulation parent exits unexpectedly.

    Linux does not automatically terminate a multiprocessing child when its
    parent dies.  ``PR_SET_PDEATHSIG`` closes that gap and prevents a crashed
    simulation from leaving a Vuer listener behind on the configured port.
    Other platforms keep the existing daemon-process behavior.
    """

    if os.name != "posix" or not hasattr(signal, "SIGTERM"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG
        if result != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)
    except (AttributeError, OSError):
        # The process is still daemonized; this is an optional hardening path.
        return


class Quest3SharedInputStream:
    """A process-safe latest-value stream.

    The shared buffer contains exactly one frame. Publishing replaces that
    frame, so a busy simulator cannot create a queue of old Quest samples.
    """

    def __init__(self, buffer: Any, lock: Any) -> None:
        if len(buffer) < _FRAME_LENGTH.size:
            raise ValueError("Shared Quest 3 frame buffer is too small.")
        self._buffer = buffer
        self._lock = lock

    def publish(self, frame: Quest3InputFrame) -> None:
        payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self._buffer) - _FRAME_LENGTH.size:
            raise ValueError("Quest 3 frame exceeds the shared-memory buffer size.")
        with self._lock:
            start = _FRAME_LENGTH.size
            self._buffer[start : start + len(payload)] = payload
            self._buffer[:start] = _FRAME_LENGTH.pack(len(payload))

    def poll(self) -> Quest3InputFrame | None:
        with self._lock:
            header = bytes(self._buffer[: _FRAME_LENGTH.size])
            payload_length = _FRAME_LENGTH.unpack(header)[0]
            if payload_length == 0:
                return None
            if payload_length > len(self._buffer) - _FRAME_LENGTH.size:
                raise RuntimeError("Corrupt Quest 3 shared frame length.")
            start = _FRAME_LENGTH.size
            return pickle.loads(bytes(self._buffer[start : start + payload_length]))

    def wait_for_frame(
        self,
        *,
        after_sequence: int = 0,
        timeout_s: float | None = None,
    ) -> Quest3InputFrame | None:
        if after_sequence < 0:
            raise ValueError(f"after_sequence must be non-negative, got {after_sequence}.")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            frame = self.poll()
            if frame is not None and frame.sequence > after_sequence:
                return frame
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                time.sleep(min(0.005, remaining))
            else:
                time.sleep(0.005)


def _run_vuer_process(
    config: Quest3VuerConfig,
    parent_pid: int,
    buffer: Any,
    lock: Any,
    view_buffer: Any | None,
    view_lock: Any | None,
    view_consumed_sequence: Any | None,
    view_consumed_timestamp_s: Any | None,
    ready: Any,
    errors: Any,
    tracking_loss_grace_s: float,
) -> None:
    """Child-process entry point kept at module scope for spawn pickling."""

    _set_parent_death_signal(parent_pid)
    try:
        local_stream = Quest3InputStream(tracking_loss_grace_s=tracking_loss_grace_s)
        shared_stream = Quest3SharedInputStream(buffer, lock)
        if config.view_enabled:
            assert view_buffer is not None
            assert view_lock is not None
            assert view_consumed_sequence is not None
            assert view_consumed_timestamp_s is not None
            view_stream = Quest3SharedViewStream(
                view_buffer,
                view_lock,
                view_consumed_sequence,
                view_consumed_timestamp_s,
            )
        else:
            view_stream = None
        server = Quest3VuerServer(
            local_stream,
            config=config,
            view_stream=view_stream,
            on_frame=shared_stream.publish,
        )
        ready.set()
        server.run()
    except BaseException as exc:
        try:
            errors.put((type(exc).__name__, str(exc)))
        finally:
            ready.set()


@dataclass
class Quest3MultiprocessVuerServer:
    """Run the Vuer/WebXR receiver outside the simulation process."""

    config: Quest3VuerConfig
    tracking_loss_grace_s: float = 0.0
    start_timeout_s: float = 10.0
    view_resolution: tuple[int, int] = (640, 360)

    def __post_init__(self) -> None:
        if self.tracking_loss_grace_s < 0:
            raise ValueError("tracking_loss_grace_s must be non-negative.")
        if self.start_timeout_s <= 0:
            raise ValueError("start_timeout_s must be positive.")
        if len(self.view_resolution) != 2:
            raise ValueError("view_resolution must contain width and height.")
        view_width, view_height = self.view_resolution
        if view_width <= 0 or view_height <= 0:
            raise ValueError("view_resolution width and height must be positive.")
        self._context = mp.get_context("spawn")
        self._buffer = self._context.Array("B", _FRAME_BUFFER_SIZE, lock=False)
        self._lock = self._context.Lock()
        if self.config.view_enabled:
            self._view_buffer = self._context.Array(
                "B",
                view_buffer_size(view_width, view_height),
                lock=False,
            )
            self._view_lock = self._context.Lock()
            self._view_consumed_sequence = self._context.Value(
                "Q", 0, lock=False
            )
            self._view_consumed_timestamp_s = self._context.Value(
                "d", 0.0, lock=False
            )
            self.view_stream: Quest3SharedViewStream | None = (
                Quest3SharedViewStream(
                    self._view_buffer,
                    self._view_lock,
                    self._view_consumed_sequence,
                    self._view_consumed_timestamp_s,
                )
            )
        else:
            self._view_buffer = None
            self._view_lock = None
            self._view_consumed_sequence = None
            self._view_consumed_timestamp_s = None
            self.view_stream = None
        self._ready = self._context.Event()
        self._errors = self._context.Queue()
        self._process: mp.Process | None = None
        self._stopped = False
        self.stream = Quest3SharedInputStream(self._buffer, self._lock)

    def start(self) -> mp.Process:
        if self._stopped:
            raise RuntimeError("Quest 3 Vuer process server has already been stopped.")
        if self._process is not None and self._process.is_alive():
            raise RuntimeError("Quest 3 Vuer process is already running.")
        self._process = self._context.Process(
            target=_run_vuer_process,
            name="quest3-vuer-process",
            args=(
                self.config,
                os.getpid(),
                self._buffer,
                self._lock,
                self._view_buffer,
                self._view_lock,
                self._view_consumed_sequence,
                self._view_consumed_timestamp_s,
                self._ready,
                self._errors,
                self.tracking_loss_grace_s,
            ),
            daemon=True,
        )
        self._process.start()
        if not self._ready.wait(timeout=self.start_timeout_s):
            self.stop()
            raise RuntimeError("Quest 3 Vuer process did not become ready in time.")
        # Vuer signals readiness before entering its blocking event loop. Probe
        # the TCP listener so a bind/TLS failure is reported to the parent.
        probe_host = self.config.host
        if probe_host in {"", "0.0.0.0"}:
            probe_host = "127.0.0.1"
        elif probe_host == "::":
            probe_host = "::1"
        deadline = time.monotonic() + self.start_timeout_s
        while self._process.is_alive() and time.monotonic() < deadline:
            try:
                with socket.create_connection(
                    (probe_host, self.config.port),
                    timeout=min(0.1, max(0.01, deadline - time.monotonic())),
                ):
                    break
            except OSError:
                time.sleep(0.01)
        else:
            self.stop()
            raise RuntimeError("Quest 3 Vuer process did not bind its listening port.")
        try:
            error_name, error_message = self._errors.get_nowait()
        except queue.Empty:
            error_name = error_message = ""
        if error_name:
            self.stop()
            raise RuntimeError(
                f"Quest 3 Vuer process failed during startup: {error_name}: {error_message}"
            )
        if not self._process.is_alive():
            self.stop()
            raise RuntimeError("Quest 3 Vuer process exited during startup.")
        return self._process

    def stop(self, *, timeout_s: float = 2.0) -> None:
        if self._stopped:
            return
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=timeout_s)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=timeout_s)
        self._errors.close()
        self._errors.join_thread()
        self._stopped = True
