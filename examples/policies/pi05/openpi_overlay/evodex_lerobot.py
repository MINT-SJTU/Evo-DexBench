"""LeRobot v3 datasets and samplers for the Evo-DexBench OpenPI overlay.

The batched Arrow/video reader and the task-balanced samplers are the same
``dex_benchmark.data.lerobot_cpu`` implementation used by the other benchmark
policies; ``install_overlay.py`` copies it next to this module.
"""

from __future__ import annotations

import os
from typing import Any, Iterator, Sequence

import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

from openpi.training import evodex_lerobot_cpu
from openpi.training.evodex_lerobot_cpu import (
    BatchedLeRobotDataset,
    CpuVideoPipelineConfig,
)


SAMPLING_MODES = ("locality", "random", "exact")


def parse_repo_ids(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value.split(",") if isinstance(value, str) else value
    return tuple(item.strip() for item in values if item.strip())


def create_dataset(data_config: Any, action_horizon: int):
    repo_ids = parse_repo_ids(getattr(data_config, "repo_ids", ()))
    if not repo_ids:
        repo_ids = parse_repo_ids(data_config.repo_id)
    if not repo_ids:
        raise ValueError("At least one Evo-DexBench LeRobot repo id is required.")

    replay_repo_ids = parse_repo_ids(getattr(data_config, "replay_repo_ids", ()))
    replay_ratio = float(getattr(data_config, "replay_ratio", 0.0))
    all_repo_ids = (*repo_ids, *replay_repo_ids)
    if replay_repo_ids and not 0 < replay_ratio < 1:
        raise ValueError("replay_ratio must be in (0, 1) with replay datasets.")
    if not replay_repo_ids and replay_ratio != 0:
        raise ValueError("replay_ratio must be zero without replay datasets.")
    video_backend = os.environ.get("EVODEX_LEROBOT_VIDEO_BACKEND", "pyav")
    metadata = [lerobot_dataset.LeRobotDatasetMetadata(repo_id) for repo_id in all_repo_ids]
    fps_values = {int(item.fps) for item in metadata}
    if len(fps_values) != 1:
        raise ValueError(f"All combined datasets must use one fps, got {sorted(fps_values)}.")
    feature_sets = [set(item.features) for item in metadata]
    if any(features != feature_sets[0] for features in feature_sets[1:]):
        raise ValueError("All combined datasets must expose the same LeRobot features.")

    delta_timestamps = {
        key: [step / metadata[0].fps for step in range(action_horizon)]
        for key in data_config.action_sequence_keys
    }
    if len(all_repo_ids) == 1:
        dataset = lerobot_dataset.LeRobotDataset(
            all_repo_ids[0],
            video_backend=video_backend,
            delta_timestamps=delta_timestamps,
        )
    else:
        dataset = lerobot_dataset.MultiLeRobotDataset(
            list(all_repo_ids),
            video_backend=video_backend,
            delta_timestamps=delta_timestamps,
        )

    pipeline = CpuVideoPipelineConfig(
        video_reader_cache_size=int(os.environ.get("EVODEX_VIDEO_READER_CACHE_SIZE", "8")),
        decode_max_gap_frames=int(os.environ.get("EVODEX_DECODE_MAX_GAP_FRAMES", "30")),
        decode_max_span_frames=int(os.environ.get("EVODEX_DECODE_MAX_SPAN_FRAMES", "128")),
    )
    return BatchedLeRobotDataset(
        dataset,
        fps=metadata[0].fps,
        config=pipeline,
        primary_count=len(repo_ids),
        replay_ratio=replay_ratio,
    ), metadata[0]


class ExactMixBatchSampler(evodex_lerobot_cpu.WeightedLocalityBatchSampler):
    """Give every source its exact quota in every global batch.

    With four equally weighted Core tasks and batch 32, each batch holds eight
    frames from every task. Within a source, frames are still drawn from
    shuffled temporal blocks to keep video decoding local.
    """

    def __init__(self, dataset: Any, batch_size: int, **kwargs: Any) -> None:
        super().__init__(dataset, batch_size, **kwargs)
        if self.world_size != 1:
            raise ValueError("Exact batch mixing builds complete global batches in one process.")
        weights = tuple(float(value) for value in dataset.source_weights)
        counts = [weight * self.batch_size for weight in weights]
        rounded = [int(round(value)) for value in counts]
        if any(abs(value - whole) > 1e-6 for value, whole in zip(counts, rounded)) or min(rounded) <= 0:
            raise ValueError(
                f"Exact batch mixing needs integer per-source counts; batch {self.batch_size} "
                f"with source weights {weights} gives {counts}."
            )
        self.counts = tuple(rounded)
        self.batches_per_epoch = int(sum(dataset.source_sample_counts)) // self.batch_size

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        streams = []
        for blocks, count in zip(self.dataset.source_episode_blocks, self.counts, strict=True):
            chunks = tuple(
                block[offset : offset + self.temporal_block_size]
                for block in blocks
                for offset in range(0, len(block), self.temporal_block_size)
            )
            runs = self._source_runs(chunks, self.batches_per_epoch * count, rng)
            values = np.asarray([value for run in runs for value in run], dtype=np.int64)
            streams.append(values.reshape(self.batches_per_epoch, count))
        self.epoch += 1
        for batch_index in range(self.batches_per_epoch):
            batch = np.concatenate([stream[batch_index] for stream in streams])
            rng.shuffle(batch)
            yield batch.tolist()


def make_dataloader(
    dataset: Any,
    *,
    sampling_mode: str,
    temporal_block_size: int,
    seed: int,
    **kwargs: Any,
) -> torch.utils.data.DataLoader:
    """Build the single-process OpenPI JAX loader for the selected sampling mode."""

    if sampling_mode not in SAMPLING_MODES:
        raise ValueError(f"sampling_mode must be one of {SAMPLING_MODES}, got {sampling_mode!r}.")
    if sampling_mode != "exact" or not bool(kwargs.get("shuffle", False)):
        if sampling_mode == "exact":
            sampling_mode = "locality"
        return evodex_lerobot_cpu.make_policy_dataloader(
            dataset,
            sampling_mode=sampling_mode,
            temporal_block_size=temporal_block_size,
            seed=seed,
            **kwargs,
        )
    if not hasattr(dataset, "source_weights"):
        raise ValueError("Exact batch mixing requires at least two datasets.")
    kwargs.pop("shuffle")
    batch_sampler = ExactMixBatchSampler(
        dataset,
        int(kwargs.pop("batch_size")),
        temporal_block_size=int(temporal_block_size),
        seed=int(seed),
        drop_last=bool(kwargs.pop("drop_last", True)),
    )
    if int(kwargs.get("num_workers", 0)) == 0:
        kwargs.pop("prefetch_factor", None)
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)
