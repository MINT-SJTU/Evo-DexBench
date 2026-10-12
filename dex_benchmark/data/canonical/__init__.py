"""Canonical data collection primitives."""

from dex_benchmark.data.canonical.schema import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalAction,
    CanonicalObservation,
    CanonicalTransition,
    QuestFrameSnapshot,
    QuestHandSnapshot,
    strict_json,
    to_cpu_copy,
)
from dex_benchmark.data.canonical.sensor_adapter import SensorAdapter
from dex_benchmark.data.canonical.async_recorder import AsyncCanonicalRecorder
from dex_benchmark.data.canonical.metrics import RecorderMetrics
from dex_benchmark.data.canonical.spool import ChunkSpool, sha256_file
from dex_benchmark.data.canonical.uploader import ChunkUploader, FilesystemChunkUploader
from dex_benchmark.data.canonical.validate import validate_canonical_hdf5
from dex_benchmark.data.canonical.quality import (
    QualityConfig,
    inspect_episode,
    quality_check_dataset,
)
from dex_benchmark.data.canonical.replay import ReplayResult, replay_canonical_episode

__all__ = [
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalAction",
    "CanonicalObservation",
    "CanonicalTransition",
    "QuestFrameSnapshot",
    "QuestHandSnapshot",
    "SensorAdapter",
    "AsyncCanonicalRecorder",
    "ChunkSpool",
    "RecorderMetrics",
    "sha256_file",
    "ChunkUploader",
    "FilesystemChunkUploader",
    "validate_canonical_hdf5",
    "QualityConfig",
    "inspect_episode",
    "quality_check_dataset",
    "ReplayResult",
    "replay_canonical_episode",
    "strict_json",
    "to_cpu_copy",
]


def snapshot_from_quest_frame(*args, **kwargs):
    """Lazily import Quest types to keep ``dex_benchmark.data`` acyclic."""

    from dex_benchmark.data.canonical.quest import snapshot_from_quest_frame as _snapshot

    return _snapshot(*args, **kwargs)


__all__.append("snapshot_from_quest_frame")
