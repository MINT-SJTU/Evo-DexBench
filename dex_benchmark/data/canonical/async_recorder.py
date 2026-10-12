"""Asynchronous canonical recorder with bounded backpressure."""

from __future__ import annotations

from dataclasses import replace
import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from dex_benchmark.data.canonical.metrics import RecorderMetrics
from dex_benchmark.data.canonical.schema import (
    CanonicalObservation,
    CanonicalTransition,
)
from dex_benchmark.data.canonical.spool import ChunkSpool


_STOP = object()


def _without_images(observation: CanonicalObservation) -> CanonicalObservation:
    return CanonicalObservation(
        observation_id=observation.observation_id,
        timestamp_s=observation.timestamp_s,
        state=observation.state,
        taxel=observation.taxel,
        images={},
        depth={},
        image_timestamp_s=None,
    )


def _without_images_transition(transition: CanonicalTransition) -> CanonicalTransition:
    return replace(
        transition,
        observation_t=_without_images(transition.observation_t),
        observation_tp1=_without_images(transition.observation_tp1),
    )


class AsyncCanonicalRecorder:
    """Queue immutable transitions and write HDF5 chunks on a worker thread."""

    def __init__(
        self,
        output_dir: str | Path,
        episode_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        chunk_length: int = 256,
        queue_capacity: int = 256,
        spool_dir: str | Path | None = None,
        output_path: str | Path | None = None,
        uploader: Any | None = None,
        drop_images_on_full: bool = True,
        start: bool = True,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive.")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episode_id = str(episode_id)
        self.output_path = (
            self.output_dir / f"{self.episode_id}.h5"
            if output_path is None
            else Path(output_path).expanduser().resolve()
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        spool_root = (
            self.output_dir / ".spool"
            if spool_dir is None
            else Path(spool_dir).expanduser().resolve()
        )
        self.spool = ChunkSpool(
            spool_root,
            self.episode_id,
            chunk_length=chunk_length,
            root_is_episode_dir=spool_dir is not None,
        )
        self.metadata = dict(metadata or {})
        self.metadata.setdefault("episode_id", self.episode_id)
        self.metadata.setdefault("control_hz", 30.0)
        self.metadata.setdefault("image_hz", 30.0)
        self.uploader = uploader
        self.drop_images_on_full = bool(drop_images_on_full)
        self.queue: queue.Queue[CanonicalTransition | object] = queue.Queue(
            maxsize=int(queue_capacity)
        )
        self.queue_capacity = int(queue_capacity)
        self.chunk_length = int(chunk_length)
        self.metrics = RecorderMetrics()
        self._known_observation_ids, self._known_quest_sequences = (
            self.spool.recover_references()
        )
        if self.spool.chunk_paths:
            self.metrics.recovery_count = 1
        self._pending: list[CanonicalTransition] = []
        self._worker_error: BaseException | None = None
        self._last_quest_sequence: int | None = None
        self._closed = False
        self._stop_requested = False
        self._thread: threading.Thread | None = None
        if start:
            self.start()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name=f"canonical-recorder-{self.episode_id}",
            daemon=True,
        )
        self._thread.start()

    @property
    def worker_error(self) -> BaseException | None:
        return self._worker_error

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    def submit(
        self,
        transition: CanonicalTransition,
        *,
        copy: bool = True,
    ) -> bool:
        """Enqueue without waiting; return False on numeric backpressure.

        Callers that have already built a complete CPU-owned transition may
        pass ``copy=False`` to avoid copying large RGB-D arrays a second time.
        The default remains defensive for external recorder users.
        """

        if self._closed or self._stop_requested:
            self.metrics.transition_drops += 1
            return False
        if self._worker_error is not None:
            self.metrics.transition_drops += 1
            return False
        copied = transition.cpu_copy() if copy else transition
        try:
            self.queue.put_nowait(copied)
        except queue.Full:
            self.metrics.queue_full_events += 1
            if not self.drop_images_on_full:
                self.metrics.transition_drops += 1
                return False
            self.metrics.image_drops += 1
            try:
                self.queue.put_nowait(_without_images_transition(copied))
            except queue.Full:
                self.metrics.transition_drops += 1
                return False
        self.metrics.transitions_enqueued += 1
        if copied.quest_frame is not None:
            self.metrics.quest_frames += 1
            sequence = int(copied.quest_frame.sequence)
            if self._last_quest_sequence is not None and sequence > self._last_quest_sequence + 1:
                self.metrics.quest_frame_drops += sequence - self._last_quest_sequence - 1
            self._last_quest_sequence = max(
                sequence,
                self._last_quest_sequence if self._last_quest_sequence is not None else sequence,
            )
        return True

    def flush(self, *, timeout_s: float | None = None) -> None:
        started = time.monotonic()
        while self.queue.unfinished_tasks:
            if self._worker_error is not None:
                raise RuntimeError("Canonical recorder worker failed.") from self._worker_error
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                raise TimeoutError("Timed out waiting for canonical recorder queue.")
            time.sleep(0.002)

    def close(self, *, status: str = "completed", timeout_s: float = 30.0) -> Path | None:
        if self._closed:
            return None if not self.output_path.exists() else self.output_path
        self._stop_requested = True
        try:
            self.flush(timeout_s=timeout_s)
            self.queue.put(_STOP, timeout=timeout_s)
            if self._thread is not None:
                self._thread.join(timeout=max(0.0, timeout_s))
            if self._thread is not None and self._thread.is_alive():
                raise TimeoutError("Timed out stopping canonical recorder worker.")
            if self._worker_error is not None:
                raise RuntimeError("Canonical recorder worker failed.") from self._worker_error
            performance_metrics = self.snapshot_metrics()
            output = self.spool.finalize(
                self.output_path,
                metadata=self.metadata,
                status=status,
            )
            # Performance counters are collection audit data, not part of the
            # canonical training metadata contract. Keep them in the attempt
            # spool manifest while exposing the same point-in-time snapshot via
            # ``snapshot_metrics()`` to callers.
            self.spool.mark_status(status, performance=performance_metrics)
            self._closed = True
            return output
        except BaseException:
            self.spool.mark_status("aborted", error="recorder_close_failed")
            self._closed = True
            raise

    def abort(self, reason: str, *, timeout_s: float = 5.0) -> None:
        """Stop the worker without assembling a dataset HDF5 file."""

        if self._closed:
            return
        self._stop_requested = True
        abort_error: str | None = None
        try:
            if self._thread is not None:
                self.flush(timeout_s=timeout_s)
                self.queue.put(_STOP, timeout=timeout_s)
                self._thread.join(timeout=max(0.0, timeout_s))
            if self._thread is not None and self._thread.is_alive():
                abort_error = "recorder_abort_timeout"
            elif self._worker_error is not None:
                abort_error = repr(self._worker_error)
        except BaseException as exc:
            abort_error = repr(exc)
        finally:
            fields = {"reason": str(reason)}
            if abort_error is not None:
                fields["error"] = abort_error
            self.spool.mark_status("aborted", **fields)
            self._closed = True

    def snapshot_metrics(self) -> dict[str, Any]:
        return self.metrics.snapshot(
            queue_depth=self.queue_depth,
            queue_capacity=self.queue_capacity,
        )

    def _run(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is _STOP:
                        if self._pending:
                            self._write_pending()
                        return
                    assert isinstance(item, CanonicalTransition)
                    self._pending.append(item)
                    if len(self._pending) >= self.chunk_length:
                        self._write_pending()
                finally:
                    self.queue.task_done()
        except BaseException as exc:  # surfaced synchronously by flush/close.
            self._worker_error = exc
            try:
                self.spool.mark_status("aborted", error=repr(exc))
            except Exception:
                pass

    def _write_pending(self) -> None:
        transitions = self._pending
        self._pending = []
        started = time.perf_counter()
        new_observations, new_quest, _checksum, byte_count = self.spool.append_chunk(
            transitions,
            known_observation_ids=self._known_observation_ids,
            known_quest_sequences=self._known_quest_sequences,
        )
        self._known_observation_ids.update(new_observations)
        self._known_quest_sequences.update(new_quest)
        elapsed = time.perf_counter() - started
        self.metrics.transitions_written += len(transitions)
        self.metrics.chunks_written += 1
        self.metrics.record_disk(elapsed, byte_count)
        if self.uploader is not None:
            upload_started = time.perf_counter()
            self.uploader.upload_spool(self.spool)
            self.metrics.record_upload(time.perf_counter() - upload_started)
