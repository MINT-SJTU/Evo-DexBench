"""Lifecycle wrapper for Quest3 canonical episode recording."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from dex_benchmark.data.canonical.async_recorder import AsyncCanonicalRecorder
from dex_benchmark.data.canonical.validate import validate_canonical_hdf5
from dex_benchmark.teleoperation.quest3.action_canonical import (
    canonical_action_from_control_step,
)
from dex_benchmark.teleoperation.quest3.canonical_step_sink import Quest3CanonicalStepSink
from dex_benchmark.teleoperation.quest3.control_step import Quest3ControlStep
from dex_benchmark.teleoperation.quest3.observation_adapter import Quest3ObservationAdapter
from dex_benchmark.teleoperation.quest3.transition_recorder import Quest3TransitionAssembler


class Quest3CanonicalSession:
    """Own one canonical recorder episode and expose Runner callbacks."""

    def __init__(
        self,
        output_dir: str | Path,
        episode_id: str,
        *,
        control_hz: float = 30.0,
        image_hz: float = 30.0,
        required_cameras: tuple[str, ...] = (),
        required_manipulators: tuple[str, ...] = (),
        metadata: Mapping[str, Any] | None = None,
        queue_capacity: int = 256,
        chunk_length: int = 256,
        strict_backpressure: bool = True,
        spool_dir: str | Path | None = None,
    ) -> None:
        recording_metadata = dict(metadata or {})
        recording_metadata.setdefault("control_hz", float(control_hz))
        recording_metadata.setdefault("image_hz", float(image_hz))
        recording_metadata.setdefault("required_cameras", list(required_cameras))
        recording_metadata.setdefault("required_manipulators", list(required_manipulators))
        recording_metadata.setdefault("strict_backpressure", bool(strict_backpressure))
        self.recorder = AsyncCanonicalRecorder(
            output_dir,
            episode_id,
            metadata=recording_metadata,
            queue_capacity=queue_capacity,
            chunk_length=chunk_length,
            drop_images_on_full=not strict_backpressure,
            spool_dir=spool_dir,
        )
        adapter = Quest3ObservationAdapter(
            control_hz=control_hz,
            image_hz=image_hz,
            required_cameras=required_cameras,
            required_manipulators=required_manipulators,
        )
        self.assembler = Quest3TransitionAssembler(
            self.recorder,
            episode_id=episode_id,
        )
        self.sink = Quest3CanonicalStepSink(
            self.assembler,
            adapter,
            action_factory=canonical_action_from_control_step,
        )
        self.required_cameras = tuple(str(role) for role in required_cameras)
        self.required_manipulators = tuple(str(role) for role in required_manipulators)
        self._closed = False
        self._last_control_tick_s: float | None = None

    @property
    def output_path(self) -> Path:
        return self.recorder.output_path

    @property
    def metrics(self) -> dict[str, Any]:
        """Return a point-in-time recorder performance snapshot."""

        return self.recorder.snapshot_metrics()

    def validate_output(self, *, verify_checksums: bool = False) -> dict[str, Any]:
        """Run structural and required-sensor validation on the finalized file."""

        if not self.output_path.exists():
            raise FileNotFoundError(self.output_path)
        return validate_canonical_hdf5(
            self.output_path,
            manifest_path=(
                self.recorder.spool.manifest_path if verify_checksums else None
            ),
            verify_checksums=verify_checksums,
            required_cameras=self.required_cameras,
            required_tactile_roles=self.required_manipulators,
        )

    def start_recording(
        self,
        observation: Any,
        *,
        timestamp_s: float | None = None,
    ) -> None:
        """Start observation numbering at the post-calibration boundary."""

        self.sink.reset(observation, timestamp_s=timestamp_s)

    def update_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Add reset/provenance metadata before the episode is finalized."""

        if self._closed:
            raise RuntimeError("Cannot update metadata after the session is closed.")
        self.recorder.metadata.update({str(key): value for key, value in metadata.items()})

    def on_reset(self, observation: Any, *, timestamp_s: float | None = None) -> None:
        """Compatibility alias for callers that explicitly define episode start."""

        self.start_recording(observation, timestamp_s=timestamp_s)

    def on_step(self, step: Quest3ControlStep) -> None:
        self.sink(step)
        # Count only transitions accepted by the canonical sink.  This keeps
        # recorder performance metrics aligned with the data actually queued.
        control_tick_s = step.timestamps.get("control_tick_s")
        if control_tick_s is None:
            return
        control_tick_s = float(control_tick_s)
        period_s = (
            None
            if self._last_control_tick_s is None
            else max(0.0, control_tick_s - self._last_control_tick_s)
        )
        received_s = step.timestamps.get("received_timestamp_s")
        frame_age_s = (
            None
            if received_s is None
            else max(0.0, control_tick_s - float(received_s))
        )
        step_end_s = step.timestamps.get("step_end_s")
        step_s = (
            None
            if step_end_s is None
            else max(0.0, float(step_end_s) - control_tick_s)
        )
        self.recorder.metrics.record_control(
            period_s=period_s,
            frame_age_s=frame_age_s,
            step_s=step_s,
        )
        self._last_control_tick_s = control_tick_s

    def close(self, *, status: str = "completed") -> Path | None:
        if self._closed:
            return self.output_path if self.output_path.exists() else None
        self._closed = True
        return self.recorder.close(status=status)

    def abort(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        self.recorder.abort(reason)
