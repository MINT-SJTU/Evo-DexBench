from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch


T = TypeVar("T")


def normalized_source_weights(
    source_count: int,
    *,
    primary_count: int,
    replay_ratio: float,
) -> tuple[float, ...]:
    """Return task-balanced probabilities for primary and replay datasets."""

    if source_count <= 0 or not 1 <= primary_count <= source_count:
        raise ValueError("primary_count must select at least one source.")
    replay_count = source_count - primary_count
    if not 0 <= replay_ratio < 1:
        raise ValueError("replay_ratio must be in [0, 1).")
    if replay_count == 0:
        if replay_ratio != 0:
            raise ValueError("replay_ratio must be zero without replay datasets.")
        return (1.0 / primary_count,) * primary_count
    if replay_ratio <= 0:
        raise ValueError("replay datasets require replay_ratio > 0.")
    primary_weight = (1.0 - replay_ratio) / primary_count
    replay_weight = replay_ratio / replay_count
    return (primary_weight,) * primary_count + (replay_weight,) * replay_count


def source_sample_counts(
    lengths: Sequence[int],
    weights: Sequence[float],
    *,
    primary_count: int,
) -> tuple[int, ...]:
    """Choose one epoch worth of samples while covering every primary source."""

    if len(lengths) != len(weights) or not lengths:
        raise ValueError("lengths and weights must be non-empty and equal length.")
    if any(length <= 0 for length in lengths):
        raise ValueError("Every source dataset must contain samples.")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Every source weight must be positive.")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError("Source weights must sum to one.")
    # The epoch is the shortest weighted schedule that sees each primary source
    # at least once in full. Replay sources are sampled to their requested share.
    epoch_size = max(
        math.ceil(int(lengths[index]) / float(weights[index]))
        for index in range(primary_count)
    )
    counts = [max(1, round(epoch_size * float(weight))) for weight in weights]
    counts[-1] += epoch_size - sum(counts)
    return tuple(int(value) for value in counts)


def validate_source_weights(
    weights: Sequence[float], *, source_count: int
) -> tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if len(values) != source_count:
        raise ValueError("source_weights must contain one value per dataset.")
    if any(value <= 0 for value in values):
        raise ValueError("Every source weight must be positive.")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise ValueError(f"source_weights must sum to one, got {total}.")
    return values


def _dataset_signature(dataset: Any) -> dict[str, Any]:
    path = Path(dataset.dataset_path)
    digest = hashlib.sha256()
    meta_root = path / "meta"
    if meta_root.is_dir():
        for metadata_path in sorted(item for item in meta_root.rglob("*") if item.is_file()):
            digest.update(metadata_path.relative_to(path).as_posix().encode())
            digest.update(metadata_path.read_bytes())
    # Do not hash every byte of potentially hundreds of GB of video. Relative
    # names and sizes make the signature independent of the mount point while
    # detecting additions, removals, and ordinary content changes.
    for directory in (path / "data", path / "videos"):
        if directory.is_dir():
            for item in sorted(value for value in directory.rglob("*") if value.is_file()):
                digest.update(item.relative_to(path).as_posix().encode())
                digest.update(str(item.stat().st_size).encode())
    return {
        "repo_id": str(dataset.repo_id),
        "frames": int(len(dataset)),
        "content_signature": digest.hexdigest(),
    }


def training_manifest_for_dataset(dataset: Any) -> dict[str, Any]:
    if hasattr(dataset, "training_manifest"):
        return dataset.training_manifest()
    payload = {
        "format_version": 1,
        "primary_count": 1,
        "replay_ratio": 0.0,
        "source_weights": [1.0],
        "source_sample_counts": [len(dataset)],
        "sources": [_dataset_signature(dataset)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}


def policy_contract_for_dataset(dataset: Any) -> dict[str, Any]:
    """Return stable shape/sensor fields that an init checkpoint must satisfy."""

    fields = (
        "embodiment", "roles", "camera_roles", "fps", "image_hz", "state_dim",
        "action_dim", "motion_action_dim", "model_action_dim", "horizon",
        "n_obs_steps", "chunk_size", "tactile_dim", "tactile_mode", "schema_hash",
    )
    contract: dict[str, Any] = {}
    for field in fields:
        if hasattr(dataset, field):
            value = getattr(dataset, field)
            if isinstance(value, tuple):
                value = list(value)
            contract[field] = value
    return contract


class MultiPolicyDataset(torch.utils.data.Dataset):
    """A policy-neutral view over compatible primary and replay datasets."""

    def __init__(
        self,
        datasets: Sequence[Any],
        *,
        primary_count: int,
        replay_ratio: float,
        compatibility_fields: Sequence[str],
        source_weights: Sequence[float] | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("At least one dataset is required.")
        self.datasets = tuple(datasets)
        self.primary_count = int(primary_count)
        self.replay_ratio = float(replay_ratio)
        self.source_weights = (
            validate_source_weights(source_weights, source_count=len(self.datasets))
            if source_weights is not None
            else normalized_source_weights(
                len(self.datasets),
                primary_count=self.primary_count,
                replay_ratio=self.replay_ratio,
            )
        )
        reference = self.datasets[0]
        for dataset in self.datasets[1:]:
            mismatches = [
                field
                for field in compatibility_fields
                if getattr(dataset, field) != getattr(reference, field)
            ]
            if mismatches:
                raise ValueError(
                    f"Combined policy datasets disagree on {sorted(mismatches)}."
                )
        self._compatibility_fields = tuple(compatibility_fields)
        self._offsets = np.cumsum(
            np.asarray([0, *(len(dataset) for dataset in self.datasets)], dtype=np.int64)
        )
        self.source_sample_counts = source_sample_counts(
            [len(dataset) for dataset in self.datasets],
            self.source_weights,
            primary_count=self.primary_count,
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.datasets[0], name)

    def __len__(self) -> int:
        if hasattr(self, "_index_map"):
            return len(self._index_map)
        return int(self._offsets[-1])

    @property
    def samples_per_epoch(self) -> int:
        return int(sum(self.source_sample_counts))

    def _source_index(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if hasattr(self, "_index_map"):
            source, local = self._index_map[index]
            return int(source), int(local)
        source = int(np.searchsorted(self._offsets, index, side="right") - 1)
        return source, int(index - self._offsets[source])

    def __getitem__(self, index: int) -> Any:
        source, local = self._source_index(int(index))
        return self.datasets[source][local]

    def __getitems__(self, indices: Sequence[int]) -> list[Any]:
        grouped: dict[int, list[tuple[int, int]]] = {}
        for output_index, index in enumerate(indices):
            source, local = self._source_index(int(index))
            grouped.setdefault(source, []).append((output_index, local))
        output: list[Any | None] = [None] * len(indices)
        for source, requests in grouped.items():
            getter = getattr(self.datasets[source], "__getitems__", None)
            local_indices = [local for _, local in requests]
            items = (
                getter(local_indices)
                if getter is not None
                else [self.datasets[source][index] for index in local_indices]
            )
            for (output_index, _), item in zip(requests, items, strict=True):
                output[output_index] = item
        if any(item is None for item in output):
            raise RuntimeError("Failed to restore every combined dataset sample.")
        return [item for item in output if item is not None]

    @property
    def source_episode_blocks(self) -> tuple[tuple[np.ndarray, ...], ...]:
        result: list[tuple[np.ndarray, ...]] = []
        for source_index, dataset in enumerate(self.datasets):
            offset = int(self._offsets[source_index])
            blocks = getattr(dataset, "episode_blocks", None)
            if blocks is None:
                frame_indices = np.asarray(dataset.frame_indices, dtype=np.int64)
                dataset.dataset._ensure_hf_dataset_loaded()
                episodes = np.asarray(
                    dataset.dataset.hf_dataset["episode_index"], dtype=np.int64
                )[frame_indices]
                boundaries = np.flatnonzero(np.diff(episodes)) + 1
                blocks = tuple(
                    np.arange(start, end, dtype=np.int64)
                    for start, end in zip(
                        np.r_[0, boundaries], np.r_[boundaries, len(episodes)], strict=True
                    )
                )
            result.append(
                tuple(np.asarray(block, dtype=np.int64) + offset for block in blocks)
            )
        return tuple(result)

    @property
    def episode_blocks(self) -> tuple[np.ndarray, ...]:
        return tuple(block for source in self.source_episode_blocks for block in source)

    def get_validation_dataset(self) -> "MultiPolicyDataset":
        datasets = [dataset.get_validation_dataset() for dataset in self.datasets]
        nonempty = [dataset for dataset in datasets if len(dataset) > 0]
        if not nonempty:
            return copy.copy(self)
        primary_count = min(self.primary_count, len(nonempty))
        replay_ratio = self.replay_ratio if len(nonempty) > primary_count else 0.0
        result = type(self)(
            nonempty,
            primary_count=primary_count,
            replay_ratio=replay_ratio,
            compatibility_fields=self._compatibility_fields,
            source_weights=(
                self.source_weights
                if len(nonempty) == len(self.datasets)
                else None
            ),
        )
        result._set_weighted_index_map()
        return result

    def _set_weighted_index_map(self) -> None:
        """Make sequential validation honor the configured task weights."""

        self._index_map = tuple(
            (source_index, int(local_index))
            for source_index in range(len(self.datasets))
            for local_index in self.sampled_source_indices(source_index)
        )

    def sampled_source_indices(self, source_index: int) -> np.ndarray:
        """Deterministic indices used for weighted normalization statistics."""

        length = len(self.datasets[source_index])
        count = self.source_sample_counts[source_index]
        # Cover the complete source distribution rather than taking a prefix.
        # Repetition is evenly spread when a source is upsampled, while
        # downsampling uses the midpoint of equally sized ranges.
        return np.floor(
            (np.arange(count, dtype=np.float64) + 0.5) * length / count
        ).astype(np.int64)

    def weighted_arrays(
        self, getter: Callable[[Any], Sequence[np.ndarray]]
    ) -> tuple[np.ndarray, ...]:
        per_source = [tuple(getter(dataset)) for dataset in self.datasets]
        width = len(per_source[0])
        if any(len(values) != width for values in per_source):
            raise ValueError("Every source must provide the same statistic fields.")
        result: list[np.ndarray] = []
        for field_index in range(width):
            selected = []
            for source_index, values in enumerate(per_source):
                array = np.asarray(values[field_index])
                selected.append(array[self.sampled_source_indices(source_index)])
            result.append(np.concatenate(selected, axis=0))
        return tuple(result)

    def training_manifest(self) -> dict[str, Any]:
        payload = {
            "format_version": 1,
            "primary_count": self.primary_count,
            "replay_ratio": self.replay_ratio,
            "source_weights": list(self.source_weights),
            "source_sample_counts": list(self.source_sample_counts),
            "sources": [_dataset_signature(dataset) for dataset in self.datasets],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return {**payload, "content_sha256": hashlib.sha256(encoded).hexdigest()}


def make_multi_policy_dataset(
    dataset_factory: Callable[..., T],
    *,
    dataset_paths: Sequence[str],
    repo_ids: Sequence[str],
    replay_dataset_paths: Sequence[str] = (),
    replay_repo_ids: Sequence[str] = (),
    replay_ratio: float = 0.0,
    compatibility_fields: Sequence[str],
    dataset_kwargs: dict[str, Any],
) -> T | MultiPolicyDataset:
    if not dataset_paths or len(dataset_paths) != len(repo_ids):
        raise ValueError("dataset_paths and repo_ids must be non-empty and equal length.")
    if len(replay_dataset_paths) != len(replay_repo_ids):
        raise ValueError("Replay dataset paths and repo ids must have equal length.")
    paths = [*dataset_paths, *replay_dataset_paths]
    ids = [*repo_ids, *replay_repo_ids]
    datasets = [
        dataset_factory(dataset_path=path, repo_id=repo_id, **dataset_kwargs)
        for path, repo_id in zip(paths, ids, strict=True)
    ]
    if len(datasets) == 1:
        if replay_ratio != 0:
            raise ValueError("replay_ratio must be zero for a single dataset.")
        return datasets[0]
    return MultiPolicyDataset(
        datasets,
        primary_count=len(dataset_paths),
        replay_ratio=float(replay_ratio) if replay_dataset_paths else 0.0,
        compatibility_fields=compatibility_fields,
    )
