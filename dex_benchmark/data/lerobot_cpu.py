from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import inspect
import os
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader


def make_projected_lerobot_dataset(
    dataset_cls: type,
    *,
    parquet_columns: Sequence[str],
    **kwargs: Any,
) -> Any:
    """Construct LeRobot while projecting frame and episode Parquet caches."""

    import datasets
    import pyarrow.parquet as pq
    import lerobot.datasets.lerobot_dataset as lerobot_dataset_module

    original_frame_loader = dataset_cls.load_hf_dataset
    original_episode_loader = lerobot_dataset_module.load_episodes

    def load_frames(owner: Any):
        from lerobot.datasets.utils import (
            get_hf_features_from_features,
            hf_transform_to_torch,
        )

        ordered = tuple(dict.fromkeys(str(column) for column in parquet_columns))
        available = set(owner.features)
        missing = set(ordered) - available
        if missing:
            raise KeyError(f"LeRobot dataset is missing columns: {sorted(missing)}")
        paths = sorted((owner.root / "data").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"Dataset contains no Parquet files under {owner.root / 'data'}."
            )
        all_features = get_hf_features_from_features(owner.features)
        selected_features = datasets.Features(
            {column: all_features[column] for column in ordered}
        )
        result = datasets.Dataset.from_parquet(
            [str(path) for path in paths],
            features=selected_features,
            columns=list(ordered),
        )
        result.set_transform(hf_transform_to_torch)
        return result

    def load_episodes(root: Path):
        paths = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"Dataset contains no episode Parquet files under {root}."
            )
        available = pq.read_schema(paths[0]).names
        columns = [name for name in available if not name.startswith("stats/")]
        return datasets.Dataset.from_parquet(
            [str(path) for path in paths], columns=columns
        )

    dataset_cls.load_hf_dataset = load_frames
    lerobot_dataset_module.load_episodes = load_episodes
    try:
        result = dataset_cls(**kwargs)
    finally:
        dataset_cls.load_hf_dataset = original_frame_loader
        lerobot_dataset_module.load_episodes = original_episode_loader
    result.load_hf_dataset = MethodType(load_frames, result)
    return result


def configure_source_sampling(
    owner: Any,
    *,
    source_lengths: Sequence[int],
    primary_count: int,
    replay_ratio: float,
) -> None:
    """Attach task-balanced epoch sampling metadata to a dataset wrapper."""

    source_count = len(source_lengths)
    if not 1 <= primary_count <= source_count:
        raise ValueError("primary_count must select at least one source.")
    replay_count = source_count - primary_count
    if replay_count == 0:
        if replay_ratio != 0:
            raise ValueError("replay_ratio must be zero without replay sources.")
        weights = (1.0 / primary_count,) * primary_count
    else:
        if not 0 < replay_ratio < 1:
            raise ValueError("replay_ratio must be in (0, 1) with replay sources.")
        weights = (
            *((1.0 - replay_ratio) / primary_count,) * primary_count,
            *(replay_ratio / replay_count,) * replay_count,
        )
    epoch_size = max(
        int(np.ceil(source_lengths[index] / weights[index]))
        for index in range(primary_count)
    )
    counts = [max(1, round(epoch_size * weight)) for weight in weights]
    counts[-1] += epoch_size - sum(counts)
    owner.primary_count = int(primary_count)
    owner.replay_ratio = float(replay_ratio)
    owner.source_weights = tuple(float(value) for value in weights)
    owner.source_sample_counts = tuple(int(value) for value in counts)
    owner.samples_per_epoch = int(sum(counts))


@dataclass(frozen=True)
class CpuVideoPipelineConfig:
    video_reader_cache_size: int = 8
    decode_max_gap_frames: int = 30
    decode_max_span_frames: int = 128

    def __post_init__(self) -> None:
        if min(
            self.video_reader_cache_size,
            self.decode_max_gap_frames,
            self.decode_max_span_frames,
        ) <= 0:
            raise ValueError("CPU video pipeline limits must be positive.")


class PersistentVideoReaderCache:
    """Process-local LRU cache of torchvision PyAV readers."""

    def __init__(self, config: CpuVideoPipelineConfig) -> None:
        self.config = config
        self._pid: int | None = None
        self._readers: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_pid"] = None
        state["_readers"] = OrderedDict()
        return state

    def _ensure_process_local(self) -> None:
        pid = os.getpid()
        if self._pid != pid:
            self.close()
            self._pid = pid

    @staticmethod
    def _close_reader(reader: Any) -> None:
        container = getattr(reader, "container", None)
        if container is not None:
            container.close()

    def close(self) -> None:
        for reader in self._readers.values():
            self._close_reader(reader)
        self._readers.clear()

    def _get_reader(self, video_path: Path | str) -> Any:
        self._ensure_process_local()
        key = str(Path(video_path).expanduser().resolve())
        reader = self._readers.pop(key, None)
        if reader is None:
            import torchvision

            torchvision.set_video_backend("pyav")
            reader = torchvision.io.VideoReader(key, "video")
        self._readers[key] = reader
        while len(self._readers) > self.config.video_reader_cache_size:
            _, evicted = self._readers.popitem(last=False)
            self._close_reader(evicted)
        return reader

    @staticmethod
    def _segments(
        timestamps: Sequence[float],
        fps: int,
        max_gap_frames: int,
        max_span_frames: int,
    ) -> list[list[float]]:
        unique = sorted(set(float(value) for value in timestamps))
        if not unique:
            return []
        segments: list[list[float]] = [[unique[0]]]
        for timestamp in unique[1:]:
            current = segments[-1]
            gap_frames = round((timestamp - current[-1]) * fps)
            span_frames = round((timestamp - current[0]) * fps)
            if gap_frames > max_gap_frames or span_frames > max_span_frames:
                segments.append([timestamp])
            else:
                current.append(timestamp)
        return segments

    @staticmethod
    def _decode_segment(
        reader: Any,
        timestamps: Sequence[float],
        tolerance_s: float,
    ) -> dict[float, torch.Tensor]:
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        reader.seek(first_ts, keyframes_only=True)
        frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []
        for frame in reader:
            current_ts = float(frame["pts"])
            frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
        if not loaded_ts:
            raise RuntimeError(f"Video reader returned no frames near {first_ts:.6f}s.")

        requested = torch.tensor(timestamps, dtype=torch.float64)
        decoded_ts = torch.tensor(loaded_ts, dtype=torch.float64)
        distance = torch.cdist(requested[:, None], decoded_ts[:, None], p=1)
        minimum, closest = distance.min(dim=1)
        if not bool((minimum < tolerance_s).all()):
            raise RuntimeError(
                "Decoded video timestamps exceed tolerance: "
                f"requested={list(timestamps)}, nearest_error={minimum.tolist()}, "
                f"tolerance={tolerance_s}."
            )
        return {
            float(timestamp): frames[int(frame_index)].to(torch.float32).div(255)
            for timestamp, frame_index in zip(timestamps, closest, strict=True)
        }

    def decode(
        self,
        video_path: Path | str,
        timestamps: Sequence[float],
        *,
        fps: int,
        tolerance_s: float,
    ) -> torch.Tensor:
        reader = self._get_reader(video_path)
        decoded: dict[float, torch.Tensor] = {}
        for segment in self._segments(
            timestamps,
            fps,
            self.config.decode_max_gap_frames,
            self.config.decode_max_span_frames,
        ):
            decoded.update(self._decode_segment(reader, segment, tolerance_s))
        return torch.stack([decoded[float(timestamp)] for timestamp in timestamps])


class LeRobotBatchFetcher:
    """Batched equivalent of LeRobot 0.4's item query with grouped video reads."""

    _BASE_COLUMNS = frozenset(
        {
            "timestamp",
            "task_index",
            "episode_index",
            "frame_index",
            "index",
            # Some policies (notably ACT) consume the current proprioceptive
            # state without requesting a state delta horizon.  Keep it in the
            # projected Arrow table in both cases.
            "observation.state",
        }
    )

    def __init__(
        self,
        dataset: Any,
        *,
        fps: int,
        config: CpuVideoPipelineConfig,
    ) -> None:
        self.dataset = dataset
        self.fps = int(fps)
        self.readers = PersistentVideoReaderCache(config)
        self._row_dataset = self._select_required_columns()
        episodes = dataset.meta.episodes
        self._episode_starts = tuple(
            int(episodes[index]["dataset_from_index"])
            for index in range(len(episodes))
        )
        self._episode_ends = tuple(
            int(episodes[index]["dataset_to_index"])
            for index in range(len(episodes))
        )

    def _select_required_columns(self) -> Any:
        """Keep large unused Parquet features out of the batch formatter.

        LeRobot installs a Hugging Face transform that converts every selected
        row and column to Python objects and then individual tensors. RGB
        features are read from video files, while policy tensors only need the
        non-video delta fields and lightweight sample metadata. Selecting the
        columns before querying therefore avoids materializing unused depth or
        reward arrays without changing the policy-facing result.
        """

        hf_dataset = self.dataset.hf_dataset
        available_columns = tuple(str(name) for name in hf_dataset.column_names)
        available = set(available_columns)
        video_keys = set(self.dataset.meta.video_keys)
        if self.dataset.delta_indices is None:
            required = available - video_keys
        else:
            required = self._BASE_COLUMNS | {
                str(key)
                for key in self.dataset.delta_indices
                if key not in video_keys
            }
        missing = required - available
        if missing:
            raise KeyError(
                "LeRobot dataset is missing required batch columns: "
                f"{sorted(missing)}"
            )
        selected = [name for name in available_columns if name in required]
        return hf_dataset.select_columns(selected)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["readers"] = PersistentVideoReaderCache(self.readers.config)
        return state

    def _episode_index(self, frame_index: int) -> int:
        episode_index = bisect_right(self._episode_starts, frame_index) - 1
        if (
            episode_index < 0
            or frame_index >= self._episode_ends[episode_index]
        ):
            raise IndexError(f"Frame index {frame_index} is outside every episode.")
        return episode_index

    @staticmethod
    def _take_one(column: Any, position: int) -> Any:
        return column[position]

    @staticmethod
    def _take_many(column: Any, positions: Sequence[int]) -> Any:
        if isinstance(column, torch.Tensor):
            index = torch.as_tensor(positions, dtype=torch.long)
            return torch.index_select(column, 0, index)
        if isinstance(column, np.ndarray):
            return column[np.asarray(positions, dtype=np.int64)]
        values = [column[position] for position in positions]
        if values and all(isinstance(value, torch.Tensor) for value in values):
            return torch.stack(values)
        return values

    def _video_timestamps(
        self,
        rows: dict[str, Any],
        row_positions: dict[int, int],
        current_timestamp: float,
        query_indices: dict[str, list[int]] | None,
    ) -> dict[str, list[float]]:
        timestamps: dict[str, list[float]] = {}
        for video_key in self.dataset.meta.video_keys:
            if query_indices is None or video_key not in query_indices:
                timestamps[video_key] = [current_timestamp]
                continue
            values = self._take_many(
                rows["timestamp"],
                [row_positions[int(value)] for value in query_indices[video_key]],
            )
            if isinstance(values, torch.Tensor):
                timestamps[video_key] = [float(value) for value in values.tolist()]
            else:
                timestamps[video_key] = [
                    float(value.item() if hasattr(value, "item") else value)
                    for value in values
                ]
        return timestamps

    def fetch(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        self.dataset._ensure_hf_dataset_loaded()
        frame_indices = [int(index) for index in indices]
        episode_indices = [self._episode_index(index) for index in frame_indices]
        query_plans: list[tuple[dict[str, list[int]], dict[str, Any]] | None] = []
        requested_indices = list(frame_indices)
        for frame_index, episode_index in zip(
            frame_indices, episode_indices, strict=True
        ):
            if self.dataset.delta_indices is None:
                query_plans.append(None)
                continue
            query_indices, padding = self.dataset._get_query_indices(
                frame_index, episode_index
            )
            query_plans.append((query_indices, padding))
            for key_indices in query_indices.values():
                requested_indices.extend(int(value) for value in key_indices)

        unique_indices = sorted(set(requested_indices))
        rows = self._row_dataset[unique_indices]
        row_positions = {
            frame_index: position
            for position, frame_index in enumerate(unique_indices)
        }

        items: list[dict[str, Any]] = []
        video_requests: dict[str, list[tuple[int, str, list[float]]]] = {}

        for output_index, (frame_index, episode_index, query_plan) in enumerate(
            zip(frame_indices, episode_indices, query_plans, strict=True)
        ):
            base_position = row_positions[frame_index]
            item = {
                key: self._take_one(column, base_position)
                for key, column in rows.items()
            }
            query_indices = None
            if query_plan is not None:
                query_indices, padding = query_plan
                query_result = {
                    key: self._take_many(
                        rows[key],
                        [row_positions[int(value)] for value in key_indices],
                    )
                    for key, key_indices in query_indices.items()
                    if key not in self.dataset.meta.video_keys
                }
                item = {
                    **item,
                    **padding,
                    **query_result,
                }

            current_ts = float(item["timestamp"].item())
            timestamps_by_key = self._video_timestamps(
                rows,
                row_positions,
                current_ts,
                query_indices,
            )
            episode = self.dataset.meta.episodes[episode_index]
            for video_key, timestamps in timestamps_by_key.items():
                shifted = [
                    float(episode[f"videos/{video_key}/from_timestamp"]) + float(ts)
                    for ts in timestamps
                ]
                video_path = self.dataset.root / self.dataset.meta.get_video_file_path(
                    episode_index, video_key
                )
                video_requests.setdefault(str(video_path), []).append(
                    (output_index, video_key, shifted)
                )

            task_index = int(item["task_index"].item())
            item["task"] = self.dataset.meta.tasks.iloc[task_index].name
            items.append(item)

        for video_path, requests in video_requests.items():
            all_timestamps = [
                timestamp
                for _, _, request_timestamps in requests
                for timestamp in request_timestamps
            ]
            decoded = self.readers.decode(
                video_path,
                all_timestamps,
                fps=self.fps,
                tolerance_s=float(self.dataset.tolerance_s),
            )
            offset = 0
            for output_index, video_key, request_timestamps in requests:
                count = len(request_timestamps)
                value = decoded[offset : offset + count].squeeze(0)
                offset += count
                if self.dataset.image_transforms is not None:
                    value = self.dataset.image_transforms(value)
                items[output_index][video_key] = value
        return items


class BatchedLeRobotDataset(torch.utils.data.Dataset):
    """Add batch Arrow/video reads to LeRobot single or multi datasets.

    LeRobot's ``MultiLeRobotDataset`` deliberately keeps each source dataset
    independent.  A mixed batch is therefore partitioned by source, fetched
    with the same optimized reader used by the other EvoDex policies, and
    restored to its original order.  No policy-specific transforms live here.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        fps: int,
        config: CpuVideoPipelineConfig,
        primary_count: int | None = None,
        replay_ratio: float = 0.0,
    ) -> None:
        self.dataset = dataset
        self.fps = int(fps)
        self.config = config
        sources = getattr(dataset, "_datasets", None)
        self._sources = tuple(sources) if sources is not None else (dataset,)
        self._offsets = np.cumsum(
            np.asarray([0, *(len(source) for source in self._sources)], dtype=np.int64)
        )
        self._fetchers: dict[int, LeRobotBatchFetcher] = {}
        if len(self._sources) > 1:
            configure_source_sampling(
                self,
                source_lengths=[len(source) for source in self._sources],
                primary_count=primary_count or len(self._sources),
                replay_ratio=float(replay_ratio),
            )

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fetchers"] = {}
        return state

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[int(index)]

    def _source_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_index = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return source_index, int(index - self._offsets[source_index])

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            source_index, local_index = self._source_index(int(index))
            grouped.setdefault(source_index, []).append((output_index, local_index))

        output: list[dict[str, Any] | None] = [None] * len(indices)
        for source_index, requests in grouped.items():
            fetcher = self._fetchers.get(source_index)
            source = self._sources[source_index]
            if getattr(source, "video_backend", "pyav") != "pyav":
                for output_index, local_index in requests:
                    output[output_index] = source[local_index]
                continue
            if fetcher is None or fetcher.dataset is not source:
                fetcher = LeRobotBatchFetcher(
                    source,
                    fps=self.fps,
                    config=self.config,
                )
                self._fetchers[source_index] = fetcher
            items = fetcher.fetch([local_index for _, local_index in requests])
            for (output_index, _), item in zip(requests, items, strict=True):
                if len(self._sources) > 1:
                    item["dataset_index"] = torch.tensor(source_index)
                output[output_index] = item
        if any(item is None for item in output):
            raise RuntimeError("Failed to restore every item in a LeRobot batch.")
        return [item for item in output if item is not None]

    @property
    def episode_blocks(self) -> tuple[np.ndarray, ...]:
        """Logical sample indices grouped by source episode."""
        blocks: list[np.ndarray] = []
        for source_index, source in enumerate(self._sources):
            offset = int(self._offsets[source_index])
            episodes = source.meta.episodes
            for episode_index in range(len(episodes)):
                episode = episodes[episode_index]
                start = int(episode["dataset_from_index"]) + offset
                end = int(episode["dataset_to_index"]) + offset
                blocks.append(np.arange(start, end, dtype=np.int64))
        return tuple(blocks)

    @property
    def source_episode_blocks(self) -> tuple[tuple[np.ndarray, ...], ...]:
        result: list[list[np.ndarray]] = [[] for _ in self._sources]
        for source_index, source in enumerate(self._sources):
            offset = int(self._offsets[source_index])
            for episode in source.meta.episodes:
                start = int(episode["dataset_from_index"]) + offset
                end = int(episode["dataset_to_index"]) + offset
                result[source_index].append(np.arange(start, end, dtype=np.int64))
        return tuple(tuple(blocks) for blocks in result)


def configure_cpu_video_pipeline(
    owner: Any,
    *,
    video_reader_cache_size: int,
    decode_max_gap_frames: int,
    decode_max_span_frames: int,
) -> None:
    owner._cpu_video_config = CpuVideoPipelineConfig(
        video_reader_cache_size=int(video_reader_cache_size),
        decode_max_gap_frames=int(decode_max_gap_frames),
        decode_max_span_frames=int(decode_max_span_frames),
    )
    owner._batch_fetcher = None


def fetch_lerobot_batch(owner: Any, indices: Sequence[int]) -> list[dict[str, Any]]:
    global_indices = [int(owner.frame_indices[int(index)]) for index in indices]
    if owner.video_backend != "pyav":
        return [owner.dataset[index] for index in global_indices]
    fetcher = getattr(owner, "_batch_fetcher", None)
    if fetcher is None or fetcher.dataset is not owner.dataset:
        fetcher = LeRobotBatchFetcher(
            owner.dataset,
            fps=owner.fps,
            config=owner._cpu_video_config,
        )
        owner._batch_fetcher = fetcher
    return fetcher.fetch(global_indices)


class LocalityAwareBatchSampler(BatchSampler):
    """Shuffle episode-local temporal blocks while covering each sample once."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        temporal_block_size: int = 128,
        seed: int = 42,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size <= 0 or temporal_block_size <= 0:
            raise ValueError("batch_size and temporal_block_size must be positive.")
        if temporal_block_size < batch_size:
            raise ValueError("temporal_block_size must be at least batch_size.")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.temporal_block_size = int(temporal_block_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0
        self._blocks = self._episode_blocks()

    def _episode_blocks(self) -> list[np.ndarray]:
        provided = getattr(self.dataset, "episode_blocks", None)
        if provided is not None:
            blocks: list[np.ndarray] = []
            for episode in provided:
                blocks.extend(
                    episode[offset : offset + self.temporal_block_size]
                    for offset in range(0, len(episode), self.temporal_block_size)
                )
            return blocks
        frame_indices = np.asarray(self.dataset.frame_indices, dtype=np.int64)
        self.dataset.dataset._ensure_hf_dataset_loaded()
        episode_column = np.asarray(
            self.dataset.dataset.hf_dataset["episode_index"], dtype=np.int64
        )[frame_indices]
        blocks: list[np.ndarray] = []
        start = 0
        while start < len(frame_indices):
            end = start + 1
            while end < len(frame_indices) and episode_column[end] == episode_column[start]:
                end += 1
            logical = np.arange(start, end, dtype=np.int64)
            blocks.extend(
                logical[offset : offset + self.temporal_block_size]
                for offset in range(0, len(logical), self.temporal_block_size)
            )
            start = end
        return blocks

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(len(self._blocks))
        flattened: list[int] = []
        for block_index in order:
            block = self._blocks[int(block_index)].copy()
            rng.shuffle(block)
            flattened.extend(int(value) for value in block)
        batches: list[list[int]] = []
        for offset in range(0, len(flattened), self.batch_size):
            batch = flattened[offset : offset + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        # DDP ranks must execute the same number of updates.  Dropping at most
        # ``world_size - 1`` global batches is preferable to DistributedSampler's
        # padding, which silently repeats samples across ranks.
        usable = len(batches) - (len(batches) % self.world_size)
        batches = batches[:usable] if self.world_size > 1 else batches
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        size = len(self.dataset)
        if self.drop_last:
            batch_count = size // self.batch_size
            return batch_count // self.world_size
        batch_count = (size + self.batch_size - 1) // self.batch_size
        return batch_count // self.world_size if self.world_size > 1 else batch_count


class WeightedLocalityBatchSampler(BatchSampler):
    """Task-weighted sampling that retains episode-local decode locality."""

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        *,
        temporal_block_size: int = 128,
        seed: int = 42,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if batch_size <= 0 or temporal_block_size < batch_size:
            raise ValueError("temporal_block_size must be at least batch_size > 0.")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size).")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.temporal_block_size = int(temporal_block_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

    @staticmethod
    def _source_runs(
        blocks: Sequence[np.ndarray], count: int, rng: np.random.Generator
    ) -> list[list[int]]:
        result: list[list[int]] = []
        remaining = int(count)
        while remaining > 0:
            for block_index in rng.permutation(len(blocks)):
                block = np.asarray(blocks[int(block_index)], dtype=np.int64).copy()
                rng.shuffle(block)
                selected = block[:remaining]
                result.append([int(value) for value in selected])
                remaining -= len(selected)
                if remaining == 0:
                    break
        return result

    def _global_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        source_blocks = self.dataset.source_episode_blocks
        counts = self.dataset.source_sample_counts
        source_runs = [
            self._source_runs(
                tuple(
                    block[offset : offset + self.temporal_block_size]
                    for block in blocks
                    for offset in range(0, len(block), self.temporal_block_size)
                ),
                int(count),
                rng,
            )
            for blocks, count in zip(source_blocks, counts, strict=True)
        ]
        # Interleave sources according to the remaining requested mass. This
        # preserves long local runs without making the epoch task-ordered.
        runs: list[list[int]] = []
        for values in source_runs:
            runs.extend(values)
        rng.shuffle(runs)
        flattened = [value for run in runs for value in run]
        batches = [
            flattened[offset : offset + self.batch_size]
            for offset in range(0, len(flattened), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) != self.batch_size:
            batches.pop()
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._global_batches()
        self.epoch += 1
        usable = len(batches) - (len(batches) % self.world_size)
        batches = batches[:usable] if self.world_size > 1 else batches
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        batch_count = int(sum(self.dataset.source_sample_counts))
        if self.drop_last:
            batch_count //= self.batch_size
            return batch_count // self.world_size
        batch_count = (batch_count + self.batch_size - 1) // self.batch_size
        return batch_count // self.world_size if self.world_size > 1 else batch_count


class WeightedRandomBatchSampler(WeightedLocalityBatchSampler):
    """Keep the same source quotas while shuffling individual frames globally."""

    def _global_batches(self) -> list[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        flattened: list[int] = []
        for blocks, count in zip(
            self.dataset.source_episode_blocks,
            self.dataset.source_sample_counts,
            strict=True,
        ):
            # Sample from a whole source, independent of episode/block lengths.
            # Complete permutations cover primary sources before repeating them.
            indices = np.concatenate(blocks)
            for run in self._source_runs((indices,), int(count), rng):
                flattened.extend(run)
        rng.shuffle(flattened)
        batches = [
            flattened[offset : offset + self.batch_size]
            for offset in range(0, len(flattened), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) != self.batch_size:
            batches.pop()
        return batches


def make_policy_dataloader(
    dataset: Any,
    *,
    sampling_mode: str = "locality",
    temporal_block_size: int = 128,
    seed: int = 42,
    **kwargs: Any,
) -> DataLoader:
    if sampling_mode not in {"locality", "random"}:
        raise ValueError("sampling_mode must be 'locality' or 'random'.")
    if temporal_block_size <= 0:
        raise ValueError("temporal_block_size must be positive.")
    batch_size_value = kwargs.get("batch_size")
    if batch_size_value is not None and temporal_block_size < int(batch_size_value):
        raise ValueError("temporal_block_size must be at least batch_size.")
    if kwargs.get("prefetch_factor") is not None and int(kwargs["prefetch_factor"]) <= 0:
        raise ValueError("prefetch_factor must be positive.")
    num_workers = int(kwargs.get("num_workers", 0))
    if kwargs.get("persistent_workers", False) and num_workers == 0:
        raise ValueError("persistent_workers requires num_workers > 0.")
    if num_workers == 0:
        kwargs.pop("prefetch_factor", None)

    weighted = hasattr(dataset, "source_weights")
    if not bool(kwargs.get("shuffle", False)) or (sampling_mode == "random" and not weighted):
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(dataset, generator=generator, **kwargs)

    batch_size = int(kwargs.pop("batch_size"))
    shuffle = bool(kwargs.pop("shuffle"))
    if not shuffle:
        raise ValueError("Locality sampling is only valid for shuffled training data.")
    drop_last = bool(kwargs.pop("drop_last", False))
    if weighted:
        sampler_cls = (
            WeightedRandomBatchSampler if sampling_mode == "random"
            else WeightedLocalityBatchSampler
        )
        batch_sampler = sampler_cls(
            dataset,
            batch_size,
            temporal_block_size=int(temporal_block_size),
            seed=int(seed),
            drop_last=drop_last,
        )
    else:
        batch_sampler = LocalityAwareBatchSampler(
            dataset,
            batch_size,
            temporal_block_size=int(temporal_block_size),
            seed=int(seed),
            drop_last=drop_last,
        )
    return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)


def make_distributed_policy_dataloader(
    dataset: Any,
    *,
    distributed_sampler: Any,
    batch_size: int,
    temporal_block_size: int,
    seed: int,
    drop_last: bool,
    **kwargs: Any,
) -> DataLoader:
    """Build a locality-aware loader sharded by complete batches across ranks."""

    rank = int(distributed_sampler.rank)
    world_size = int(distributed_sampler.num_replicas)
    sampler_cls = (
        WeightedLocalityBatchSampler
        if hasattr(dataset, "source_weights")
        else LocalityAwareBatchSampler
    )
    batch_sampler = sampler_cls(
        dataset,
        int(batch_size),
        temporal_block_size=int(temporal_block_size),
        seed=int(seed),
        drop_last=bool(drop_last),
        rank=rank,
        world_size=world_size,
    )
    return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)


def install_workspace_dataloader(
    workspace_cls: type,
    *,
    sampling_mode: str,
    temporal_block_size: int,
    seed: int,
    prefetch_factor: int,
) -> None:
    """Replace only an official workspace module's DataLoader constructor."""

    if prefetch_factor <= 0:
        raise ValueError("prefetch_factor must be positive.")
    module = inspect.getmodule(workspace_cls)
    if module is None or not hasattr(module, "DataLoader"):
        raise RuntimeError(f"Cannot locate DataLoader for workspace {workspace_cls}.")
    original = module.DataLoader
    if getattr(original, "_evodex_cpu_pipeline", False):
        return

    def data_loader(dataset: Any, **kwargs: Any) -> DataLoader:
        if int(kwargs.get("num_workers", 0)) > 0:
            kwargs.setdefault("prefetch_factor", int(prefetch_factor))
        mode = sampling_mode if bool(kwargs.get("shuffle", False)) else "random"
        return make_policy_dataloader(
            dataset,
            sampling_mode=mode,
            temporal_block_size=temporal_block_size,
            seed=seed,
            **kwargs,
        )

    data_loader._evodex_cpu_pipeline = True  # type: ignore[attr-defined]
    module.DataLoader = data_loader
