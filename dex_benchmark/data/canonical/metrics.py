"""Low-overhead recorder and control-loop performance counters."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any


@dataclass
class RecorderMetrics:
    control_ticks: int = 0
    transitions_enqueued: int = 0
    transitions_written: int = 0
    image_drops: int = 0
    transition_drops: int = 0
    quest_frames: int = 0
    quest_frame_drops: int = 0
    queue_full_events: int = 0
    chunks_written: int = 0
    checksum_failures: int = 0
    recovery_count: int = 0
    upload_retries: int = 0
    bytes_written: int = 0
    control_period_s: list[float] = field(default_factory=list)
    frame_age_s: list[float] = field(default_factory=list)
    encode_s: list[float] = field(default_factory=list)
    step_s: list[float] = field(default_factory=list)
    sample_s: list[float] = field(default_factory=list)
    disk_s: list[float] = field(default_factory=list)
    upload_s: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_control(
        self,
        *,
        period_s: float | None = None,
        frame_age_s: float | None = None,
        encode_s: float | None = None,
        step_s: float | None = None,
        sample_s: float | None = None,
    ) -> None:
        with self._lock:
            self.control_ticks += 1
            for collection, value in (
                (self.control_period_s, period_s),
                (self.frame_age_s, frame_age_s),
                (self.encode_s, encode_s),
                (self.step_s, step_s),
                (self.sample_s, sample_s),
            ):
                if value is not None:
                    collection.append(float(value))

    def record_disk(self, elapsed_s: float, byte_count: int) -> None:
        with self._lock:
            self.disk_s.append(float(elapsed_s))
            self.bytes_written += int(byte_count)

    def record_upload(self, elapsed_s: float) -> None:
        with self._lock:
            self.upload_s.append(float(elapsed_s))

    def snapshot(self, *, queue_depth: int = 0, queue_capacity: int = 0) -> dict[str, Any]:
        with self._lock:
            result: dict[str, Any] = {
                "control_ticks": self.control_ticks,
                "transitions_enqueued": self.transitions_enqueued,
                "transitions_written": self.transitions_written,
                "image_drops": self.image_drops,
                "transition_drops": self.transition_drops,
                "quest_frames": self.quest_frames,
                "quest_frame_drops": self.quest_frame_drops,
                "queue_full_events": self.queue_full_events,
                "chunks_written": self.chunks_written,
                "checksum_failures": self.checksum_failures,
                "recovery_count": self.recovery_count,
                "upload_retries": self.upload_retries,
                "bytes_written": self.bytes_written,
                "queue_depth": int(queue_depth),
                "queue_capacity": int(queue_capacity),
            }
            for name, values in (
                ("control_period_s", self.control_period_s),
                ("frame_age_s", self.frame_age_s),
                ("encode_s", self.encode_s),
                ("step_s", self.step_s),
                ("sample_s", self.sample_s),
                ("disk_s", self.disk_s),
                ("upload_s", self.upload_s),
            ):
                if values:
                    result[name] = {
                        "count": len(values),
                        "mean": sum(values) / len(values),
                        "p50": _percentile(values, 0.50),
                        "p95": _percentile(values, 0.95),
                        "p99": _percentile(values, 0.99),
                    }
                else:
                    result[name] = {"count": 0}
            return result


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
    return float(ordered[index])


def monotonic_timestamp_s() -> float:
    return time.perf_counter_ns() / 1_000_000_000.0
