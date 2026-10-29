from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dex_benchmark.data.lerobot_cpu import (
    configure_cpu_video_pipeline,
    fetch_lerobot_batch,
    make_projected_lerobot_dataset,
)
from dex_benchmark.data.multi_dataset import MultiPolicyDataset, source_sample_counts
from dex_benchmark.policy_api.policy_features import (
    camera_roles_for_embodiment,
    roles_for_embodiment,
    select_compact_policy_state,
    validate_policy_frequency,
)

try:
    from diffusion_policy.dataset.base_dataset import BaseImageDataset as _DPBaseDataset
except ImportError:
    # Dataset helpers such as select_policy_state remain importable in the
    # simulation-only environment, where the external training repository is
    # deliberately not installed.
    class _DPBaseDataset(torch.utils.data.Dataset):
        pass


def _roles_for_embodiment(embodiment: str) -> tuple[str, ...]:
    return roles_for_embodiment(embodiment)


def select_policy_state(value: Any, embodiment: str) -> np.ndarray:
    """Backward-compatible alias for the shared policy-state selector."""

    return select_compact_policy_state(value, embodiment)


class LeRobotV3DPDataset(_DPBaseDataset):
    """Official DP BaseImageDataset adapter over one LeRobot v3 dataset."""

    def __init__(
        self,
        dataset_path: str,
        repo_id: str,
        embodiment: str,
        camera_roles: list[str] | tuple[str, ...] | None = None,
        horizon: int = 30,
        n_obs_steps: int = 2,
        val_ratio: float = 0.1,
        seed: int = 42,
        split: str = "train",
        video_backend: str = "pyav",
        video_reader_cache_size: int = 8,
        decode_max_gap_frames: int = 30,
        decode_max_span_frames: int = 128,
    ) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:
            raise ImportError(
                "LeRobotV3DPDataset requires the pinned official Diffusion "
                "Policy checkout and LeRobot 0.4."
            ) from exc
        if horizon < n_obs_steps or n_obs_steps <= 0:
            raise ValueError("Require horizon >= n_obs_steps > 0.")
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must be in [0, 1).")
        if split not in ("train", "val"):
            raise ValueError("split must be 'train' or 'val'.")

        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.repo_id = repo_id
        self.embodiment = embodiment
        self.roles = _roles_for_embodiment(embodiment)
        self.camera_roles = tuple(
            camera_roles
            if camera_roles is not None
            else camera_roles_for_embodiment(embodiment)
        )
        if len(self.camera_roles) != 3 or len(set(self.camera_roles)) != 3:
            raise ValueError("DP requires exactly three distinct RGB camera roles.")
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.split = split
        self.video_backend = video_backend
        configure_cpu_video_pipeline(
            self,
            video_reader_cache_size=video_reader_cache_size,
            decode_max_gap_frames=decode_max_gap_frames,
            decode_max_span_frames=decode_max_span_frames,
        )

        info = json.loads((self.dataset_path / "meta" / "info.json").read_text())
        if info.get("codebase_version") != "v3.0":
            raise ValueError("DP training requires a LeRobot v3.0 dataset.")
        self.fps = int(info["fps"])
        validate_policy_frequency(self.fps, policy_name="DP")
        self.total_episodes = int(info["total_episodes"])
        self._validate_evodex_metadata()
        train_episodes, val_episodes = self._split_episodes()
        self.episode_indices = train_episodes if split == "train" else val_episodes
        if not self.episode_indices:
            raise ValueError(f"The {split} split has no episodes.")

        obs_offsets = [
            step / self.fps for step in range(-(self.n_obs_steps - 1), 1)
        ]
        action_offsets = [
            step / self.fps
            for step in range(
                -(self.n_obs_steps - 1),
                self.horizon - self.n_obs_steps + 1,
            )
        ]
        self.delta_timestamps = {
            "observation.state": obs_offsets,
            "action": action_offsets,
            **{
                f"observation.images.{role}": obs_offsets
                for role in self.camera_roles
            },
        }
        projected_columns = (
            "timestamp",
            "task_index",
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "action",
        )
        self.dataset = make_projected_lerobot_dataset(
            LeRobotDataset,
            parquet_columns=projected_columns,
            repo_id=repo_id,
            root=self.dataset_path,
            episodes=list(self.episode_indices),
            delta_timestamps=self.delta_timestamps,
            video_backend=video_backend,
        )
        self._set_frame_indices()

    @property
    def state_dim(self) -> int:
        return 31 * len(self.roles)

    @property
    def action_dim(self) -> int:
        return 30 * len(self.roles)

    def _split_episodes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        indices = np.arange(self.total_episodes, dtype=np.int64)
        if self.total_episodes < 2 or self.val_ratio == 0:
            return tuple(int(x) for x in indices), ()
        n_val = min(
            max(1, round(self.total_episodes * self.val_ratio)),
            self.total_episodes - 1,
        )
        rng = np.random.default_rng(self.seed)
        val = set(int(x) for x in rng.choice(indices, size=n_val, replace=False))
        train = tuple(int(x) for x in indices if int(x) not in val)
        return train, tuple(sorted(val))

    def _set_frame_indices(self) -> None:
        # LeRobot 0.4 accepts ``episodes=`` but its v3 parquet loader still
        # exposes all rows and reports the global frame count. Keep an explicit
        # global-index mapping so split isolation also applies to delta-window
        # sampling and normalization statistics.
        self.dataset._ensure_hf_dataset_loaded()
        episode_column = np.asarray(
            self.dataset.hf_dataset["episode_index"], dtype=np.int64
        )
        selected = np.asarray(self.episode_indices, dtype=np.int64)
        self.frame_indices = np.flatnonzero(np.isin(episode_column, selected))
        if self.frame_indices.size == 0:
            raise ValueError(f"The {self.split} split contains no frames.")

    def _validate_evodex_metadata(self) -> None:
        path = self.dataset_path / "meta" / "dex_benchmark.json"
        metadata = json.loads(path.read_text())
        self.schema_hash = str(metadata["feature_schema_sha256"])
        episodes = metadata.get("episodes", [])
        if len(episodes) != self.total_episodes:
            raise ValueError("Evo-DexBench episode metadata count mismatch.")
        expected_action_dim = 30 * len(self.roles)
        for episode in episodes:
            roles = tuple(episode.get("manipulator_roles", ()))
            if roles != self.roles:
                raise ValueError(
                    f"Dataset mixes role order {roles}; expected {self.roles}."
                )
            action_spec = episode.get("action_spec", {})
            if action_spec.get("action_format") != "benchmark_physical_semantic":
                raise ValueError("DP requires benchmark physical semantic actions.")
            available_camera_roles = set(episode.get("camera_roles", ()))
            missing_camera_roles = set(self.camera_roles) - available_camera_roles
            if missing_camera_roles:
                raise ValueError(
                    "Dataset is missing required camera roles: "
                    f"{sorted(missing_camera_roles)}."
                )
            if len(episode.get("action_feature_order", ())) != len(self.roles) * 2:
                raise ValueError("Unexpected action feature order.")
        feature = json.loads((self.dataset_path / "meta" / "info.json").read_text())[
            "features"
        ]
        if tuple(feature["action"]["shape"]) != (expected_action_dim,):
            raise ValueError("Dataset action dimension does not match embodiment.")

    def get_validation_dataset(self):
        if self.val_ratio == 0 or self.total_episodes < 2:
            return torch.utils.data.Subset(self, [])
        result = copy.copy(self)
        result.split = "val"
        train, val = self._split_episodes()
        result.episode_indices = val
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        projected_columns = (
            "timestamp",
            "task_index",
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "action",
        )
        result.dataset = make_projected_lerobot_dataset(
            LeRobotDataset,
            parquet_columns=projected_columns,
            repo_id=self.repo_id,
            root=self.dataset_path,
            episodes=list(val),
            delta_timestamps=self.delta_timestamps,
            video_backend=self.video_backend,
        )
        result._batch_fetcher = None
        result._set_frame_indices()
        return result

    def _all_train_fields(self) -> tuple[np.ndarray, np.ndarray]:
        self.dataset._ensure_hf_dataset_loaded()
        states = np.asarray(
            self.dataset.hf_dataset["observation.state"], dtype=np.float32
        )[self.frame_indices]
        actions = np.asarray(self.dataset.hf_dataset["action"], dtype=np.float32)[
            self.frame_indices
        ]
        return select_policy_state(states, self.embodiment), actions

    def get_normalizer(self, **_: Any):
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )
        from diffusion_policy.common.normalize_util import get_image_range_normalizer

        states, actions = self._all_train_fields()
        normalizer = LinearNormalizer()
        normalizer["state"] = SingleFieldLinearNormalizer.create_fit(states)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(actions)
        for role in self.camera_roles:
            normalizer[f"{role}_rgb"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._all_train_fields()[1])

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def _format_item(self, item: dict[str, Any]) -> dict[str, Any]:
        state = select_policy_state(item["observation.state"], self.embodiment)
        action = np.asarray(item["action"], dtype=np.float32)
        expected_action = (self.horizon, self.action_dim)
        if action.shape != expected_action:
            raise ValueError(f"Expected action {expected_action}, got {action.shape}.")
        obs = {"state": torch.from_numpy(state)}
        for role in self.camera_roles:
            image = item[f"observation.images.{role}"]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            image = image.to(dtype=torch.float32)
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"Unexpected {role} video shape {tuple(image.shape)}.")
            obs[f"{role}_rgb"] = image
        return {"obs": obs, "action": torch.from_numpy(action)}

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._format_item(self.dataset[int(self.frame_indices[index])])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self._format_item(item) for item in fetch_lerobot_batch(self, indices)]


class MultiLeRobotV3DPDataset(MultiPolicyDataset, _DPBaseDataset):
    """Concatenate compatible LeRobot datasets without copying their data."""

    def __init__(
        self,
        dataset_paths: list[str],
        repo_ids: list[str],
        replay_dataset_paths: list[str] | None = None,
        replay_repo_ids: list[str] | None = None,
        replay_ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if not dataset_paths or len(dataset_paths) != len(repo_ids):
            raise ValueError("dataset_paths and repo_ids must be non-empty and equal length.")
        replay_dataset_paths = list(replay_dataset_paths or ())
        replay_repo_ids = list(replay_repo_ids or ())
        if len(replay_dataset_paths) != len(replay_repo_ids):
            raise ValueError("Replay DP paths and repo ids must have equal length.")
        paths = [*dataset_paths, *replay_dataset_paths]
        ids = [*repo_ids, *replay_repo_ids]
        datasets = tuple(
            LeRobotV3DPDataset(dataset_path=path, repo_id=repo_id, **kwargs)
            for path, repo_id in zip(paths, ids, strict=True)
        )
        super().__init__(
            datasets,
            primary_count=len(dataset_paths),
            replay_ratio=float(replay_ratio),
            compatibility_fields=(
                "embodiment", "roles", "fps", "camera_roles", "state_dim",
                "action_dim", "horizon", "n_obs_steps",
            ),
        )

    @property
    def camera_roles(self) -> tuple[str, ...]:
        return self.datasets[0].camera_roles

    @property
    def state_dim(self) -> int:
        return self.datasets[0].state_dim

    @property
    def action_dim(self) -> int:
        return self.datasets[0].action_dim

    def get_validation_dataset(self):
        validation = tuple(dataset.get_validation_dataset() for dataset in self.datasets)
        if all(len(dataset) == 0 for dataset in validation):
            return torch.utils.data.Subset(self, [])
        if any(len(dataset) == 0 for dataset in validation):
            raise ValueError(
                "Every mixed source needs validation episodes; use val_ratio=0 "
                "to train without a validation split."
            )
        result = copy.copy(self)
        result.datasets = validation
        result._offsets = np.cumsum(np.asarray(
            [0, *(len(dataset) for dataset in result.datasets)], dtype=np.int64
        ))
        result.source_sample_counts = source_sample_counts(
            [len(dataset) for dataset in result.datasets],
            result.source_weights,
            primary_count=result.primary_count,
        )
        result._set_weighted_index_map()
        return result

    def _all_train_fields(self) -> tuple[np.ndarray, np.ndarray]:
        return self.weighted_arrays(lambda dataset: dataset._all_train_fields())

    def get_normalizer(self, **_: Any):
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )
        from diffusion_policy.common.normalize_util import get_image_range_normalizer

        states, actions = self._all_train_fields()
        normalizer = LinearNormalizer()
        normalizer["state"] = SingleFieldLinearNormalizer.create_fit(states)
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(actions)
        for role in self.camera_roles:
            normalizer[f"{role}_rgb"] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self._all_train_fields()[1])


def make_lerobot_v3_dp_dataset(
    dataset_path: str | None = None,
    repo_id: str | None = None,
    dataset_paths: list[str] | None = None,
    repo_ids: list[str] | None = None,
    replay_dataset_paths: list[str] | None = None,
    replay_repo_ids: list[str] | None = None,
    replay_ratio: float = 0.0,
    **kwargs: Any,
):
    """Hydra factory preserving the original single-dataset configuration."""

    has_plural = dataset_paths is not None or repo_ids is not None
    has_single = dataset_path is not None or repo_id is not None
    if has_plural and has_single:
        raise ValueError("Use dataset_path/repo_id or dataset_paths/repo_ids, not both.")
    if has_single and (dataset_path is None or repo_id is None):
        raise ValueError("Set both dataset_path and repo_id.")
    if has_single and (replay_dataset_paths or replay_repo_ids):
        dataset_paths, repo_ids = [dataset_path], [repo_id]
        has_plural = True
    if has_plural:
        return MultiLeRobotV3DPDataset(
            dataset_paths=list(dataset_paths or ()),
            repo_ids=list(repo_ids or ()),
            replay_dataset_paths=list(replay_dataset_paths or ()),
            replay_repo_ids=list(replay_repo_ids or ()),
            replay_ratio=float(replay_ratio),
            **kwargs,
        )
    if dataset_path is None or repo_id is None:
        raise ValueError("Set dataset_path/repo_id or dataset_paths/repo_ids.")
    if replay_ratio != 0:
        raise ValueError("replay_ratio requires replay_dataset_paths/replay_repo_ids.")
    return LeRobotV3DPDataset(
        dataset_path=dataset_path,
        repo_id=repo_id,
        **kwargs,
    )
