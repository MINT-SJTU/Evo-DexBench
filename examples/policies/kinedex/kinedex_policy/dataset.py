from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset

from dex_benchmark.data.lerobot_cpu import (
    configure_cpu_video_pipeline,
    fetch_lerobot_batch,
    make_projected_lerobot_dataset,
)
from dex_benchmark.data.multi_dataset import MultiPolicyDataset, training_manifest_for_dataset
from dex_benchmark.policy_api.policy_features import (
    camera_roles_for_embodiment,
    roles_for_embodiment,
    select_compact_policy_state,
    validate_policy_frequency,
)
from kinedex_policy.tactile import (
    aggregate_finger_forces,
    expected_tactile_layout,
    interleave_motion_force,
    pack_finger_taxels,
)


class LeRobotV3KineDexDataset(BaseImageDataset):
    """Official KineDex tensor view over an Evo-DexBench LeRobot v3 dataset."""

    def __init__(
        self,
        dataset_path: str,
        repo_id: str,
        embodiment: str,
        *,
        camera_roles: list[str] | tuple[str, ...] | None = None,
        tactile_mode: str = "full",
        horizon: int = 16,
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
            raise ImportError("KineDex training requires LeRobot 0.4.") from exc
        if horizon < n_obs_steps or n_obs_steps <= 0:
            raise ValueError("KineDex horizon must cover positive observation steps.")
        if not 0 <= val_ratio < 1 or split not in ("train", "val"):
            raise ValueError("Invalid KineDex split settings.")
        if tactile_mode not in ("full", "none"):
            raise ValueError("KineDex tactile_mode must be full or none.")
        self.dataset_path = Path(dataset_path).expanduser().resolve()
        self.repo_id = repo_id
        self.embodiment = embodiment
        self.tactile_mode = tactile_mode
        self.roles = roles_for_embodiment(embodiment)
        self.camera_roles = tuple(
            camera_roles
            if camera_roles is not None
            else camera_roles_for_embodiment(embodiment)
        )
        if len(self.camera_roles) != 3 or len(set(self.camera_roles)) != 3:
            raise ValueError("KineDex requires exactly three distinct camera roles.")
        self.role_count = len(self.roles)
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
            raise ValueError("KineDex requires a LeRobot v3.0 dataset.")
        self.fps = int(info["fps"])
        validate_policy_frequency(self.fps, policy_name="KineDex")
        self.total_episodes = int(info["total_episodes"])
        self._validate_metadata(info)
        train_episodes, val_episodes = self._split_episodes()
        self.episode_indices = train_episodes if split == "train" else val_episodes
        if not self.episode_indices:
            raise ValueError(f"The KineDex {split} split has no episodes.")

        observation_offsets = [
            step / self.fps for step in range(-(self.n_obs_steps - 1), 1)
        ]
        action_offsets = [
            step / self.fps
            for step in range(
                -(self.n_obs_steps - 1), self.horizon - self.n_obs_steps + 1
            )
        ]
        self.delta_timestamps = {
            "observation.state": observation_offsets,
            "action": action_offsets,
            **{
                f"observation.images.{role}": observation_offsets
                for role in self.camera_roles
            },
            **(
                {
                    f"observation.tactile.{role}": action_offsets
                    for role in self.roles
                }
                if self.tactile_mode == "full"
                else {}
            ),
        }
        projected_columns = (
            "timestamp",
            "task_index",
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "action",
            *(
                (f"observation.tactile.{role}" for role in self.roles)
                if self.tactile_mode == "full"
                else ()
            ),
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
        self._normalizer: Any | None = None
        self._force_bounds: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def state_dim(self) -> int:
        return 31 * self.role_count

    @property
    def motion_action_dim(self) -> int:
        return 30 * self.role_count

    @property
    def force_dim(self) -> int:
        return 5 * self.role_count if self.tactile_mode == "full" else 0

    @property
    def model_action_dim(self) -> int:
        return self.motion_action_dim + self.force_dim

    @property
    def finger_count(self) -> int:
        return 5 * self.role_count

    def _split_episodes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        values = np.arange(self.total_episodes, dtype=np.int64)
        if self.total_episodes < 2 or self.val_ratio == 0:
            return tuple(int(value) for value in values), ()
        count = min(
            max(1, round(self.total_episodes * self.val_ratio)),
            self.total_episodes - 1,
        )
        rng = np.random.default_rng(self.seed)
        val = set(int(value) for value in rng.choice(values, size=count, replace=False))
        return (
            tuple(int(value) for value in values if int(value) not in val),
            tuple(sorted(val)),
        )

    def _validate_metadata(self, info: dict[str, Any]) -> None:
        metadata = json.loads(
            (self.dataset_path / "meta" / "dex_benchmark.json").read_text()
        )
        self.schema_hash = str(metadata["feature_schema_sha256"])
        episodes = metadata.get("episodes", [])
        if len(episodes) != self.total_episodes:
            raise ValueError("KineDex episode metadata count mismatch.")
        image_hz_values: set[int] = set()
        action_specs = []
        for episode in episodes:
            if tuple(episode.get("manipulator_roles", ())) != self.roles:
                raise ValueError("KineDex dataset embodiment/role order mismatch.")
            available_camera_roles = set(episode.get("camera_roles", ()))
            if not set(self.camera_roles).issubset(available_camera_roles):
                raise ValueError("KineDex camera roles do not match embodiment.")
            if int(round(float(episode.get("control_hz", 0)))) != self.fps:
                raise ValueError("KineDex episode frequency mismatch.")
            image_hz_values.add(int(round(float(episode.get("image_hz", 0)))))
            if self.tactile_mode == "full":
                layouts = episode.get("tactile_feature_specs", {})
                for role in self.roles:
                    if layouts.get(role) != expected_tactile_layout():
                        raise ValueError(f"KineDex tactile layout mismatch for {role}.")
            action_spec = episode.get("action_spec", {})
            if action_spec.get("action_format") != "benchmark_physical_semantic":
                raise ValueError("KineDex requires physical semantic actions.")
            action_specs.append(action_spec)
        if len(image_hz_values) != 1:
            raise ValueError("KineDex cannot mix image frequencies.")
        self.image_hz = image_hz_values.pop()
        self.action_spec = action_specs[0]
        if any(spec != self.action_spec for spec in action_specs[1:]):
            raise ValueError("KineDex cannot mix action specifications.")
        features = info["features"]
        if tuple(features["observation.state"]["shape"]) != (111 * self.role_count,):
            raise ValueError("KineDex state dimension mismatch.")
        if tuple(features["action"]["shape"]) != (self.motion_action_dim,):
            raise ValueError("KineDex action dimension mismatch.")
        if self.tactile_mode == "full":
            for role in self.roles:
                if tuple(features[f"observation.tactile.{role}"]["shape"]) != (123,):
                    raise ValueError(f"KineDex tactile feature mismatch for {role}.")
        for role in self.camera_roles:
            if tuple(features[f"observation.images.{role}"]["shape"]) != (224, 224, 3):
                raise ValueError(f"KineDex RGB feature mismatch for {role}.")

    def _set_frame_indices(self) -> None:
        self.dataset._ensure_hf_dataset_loaded()
        episode_column = np.asarray(
            self.dataset.hf_dataset["episode_index"], dtype=np.int64
        )
        selected = np.asarray(self.episode_indices, dtype=np.int64)
        self.frame_indices = np.flatnonzero(np.isin(episode_column, selected))
        if self.frame_indices.size == 0:
            raise ValueError(f"KineDex {self.split} split contains no frames.")

    def get_validation_dataset(self) -> "LeRobotV3KineDexDataset":
        result = copy.copy(self)
        _, result.episode_indices = self._split_episodes()
        if not result.episode_indices:
            raise ValueError("KineDex validation requires at least two episodes.")
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        result.split = "val"
        projected_columns = (
            "timestamp",
            "task_index",
            "episode_index",
            "frame_index",
            "index",
            "observation.state",
            "action",
            *(
                (f"observation.tactile.{role}" for role in self.roles)
                if self.tactile_mode == "full"
                else ()
            ),
        )
        result.dataset = make_projected_lerobot_dataset(
            LeRobotDataset,
            parquet_columns=projected_columns,
            repo_id=self.repo_id,
            root=self.dataset_path,
            episodes=list(result.episode_indices),
            delta_timestamps=self.delta_timestamps,
            video_backend=self.video_backend,
        )
        result._batch_fetcher = None
        result._set_frame_indices()
        result._normalizer = self.get_normalizer()
        result._force_bounds = (
            self.force_bounds() if self.tactile_mode == "full" else None
        )
        return result

    def _all_train_motion_fields(self) -> tuple[np.ndarray, np.ndarray]:
        self.dataset._ensure_hf_dataset_loaded()
        rows = self.dataset.hf_dataset
        states = select_compact_policy_state(
            np.asarray(rows["observation.state"], dtype=np.float32)[self.frame_indices],
            self.embodiment,
        )
        motion = np.asarray(rows["action"], dtype=np.float32)[self.frame_indices]
        return states, motion

    def _all_train_tactile(self) -> np.ndarray:
        if self.tactile_mode != "full":
            raise RuntimeError("NoTouch datasets do not expose tactile training data.")
        self.dataset._ensure_hf_dataset_loaded()
        rows = self.dataset.hf_dataset
        raw_tactile = np.concatenate(
            [
                np.asarray(rows[f"observation.tactile.{role}"], dtype=np.float32)[
                    self.frame_indices
                ]
                for role in self.roles
            ],
            axis=-1,
        )
        return raw_tactile

    def _zero_tactile_normalizer(self, normalizer_type: Any) -> Any:
        shape = (self.finger_count, 29, 1)
        zeros = np.zeros(shape, dtype=np.float32)
        return normalizer_type.create_manual(
            scale=np.ones(shape, dtype=np.float32),
            offset=zeros,
            input_stats_dict={
                "min": zeros,
                "max": zeros,
                "mean": zeros,
                "std": zeros,
            },
        )

    def get_normalizer(self, **_: Any) -> Any:
        if self._normalizer is None:
            from diffusion_policy.common.normalize_util import get_image_range_normalizer
            from diffusion_policy.model.common.normalizer import (
                LinearNormalizer,
                SingleFieldLinearNormalizer,
            )

            states, motion = self._all_train_motion_fields()
            normalizer = LinearNormalizer()
            normalizer["state"] = SingleFieldLinearNormalizer.create_fit(states)
            if self.tactile_mode == "full":
                tactile = self._all_train_tactile()
                packed = pack_finger_taxels(tactile, self.role_count)
                force = aggregate_finger_forces(tactile, self.role_count)
                action = interleave_motion_force(motion, force, self.role_count)
                normalizer["finger_raw_pressure"] = (
                    SingleFieldLinearNormalizer.create_fit(packed)
                )
                self._force_bounds = (force.min(axis=0), force.max(axis=0))
            else:
                action = motion
                normalizer["finger_raw_pressure"] = self._zero_tactile_normalizer(
                    SingleFieldLinearNormalizer
                )
            normalizer["action"] = SingleFieldLinearNormalizer.create_fit(action)
            for role in self.camera_roles:
                normalizer[f"{role}_rgb"] = get_image_range_normalizer()
            self._normalizer = normalizer
        return self._normalizer

    def force_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.tactile_mode != "full":
            raise RuntimeError("NoTouch KineDex has no force bounds.")
        self.get_normalizer()
        assert self._force_bounds is not None
        return tuple(value.copy() for value in self._force_bounds)  # type: ignore[return-value]

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = {
            "policy_type": "kinedex",
            "embodiment": self.embodiment,
            "tactile_mode": self.tactile_mode,
            "roles": list(self.roles),
            "camera_roles": list(self.camera_roles),
            "observation_keys": [
                *(f"{role}_rgb" for role in self.camera_roles),
                "state",
                "finger_raw_pressure",
            ],
            "fps": self.fps,
            "image_hz": self.image_hz,
            "prediction_horizon": self.horizon,
            "observation_horizon": self.n_obs_steps,
            "available_action_steps": self.horizon - self.n_obs_steps + 1,
            "state_dim": self.state_dim,
            "motion_action_dim": self.motion_action_dim,
            "force_dim": self.force_dim,
            "model_action_dim": self.model_action_dim,
            "schema_hash": self.schema_hash,
            "action_spec": self.action_spec,
            "dataset_manifest": training_manifest_for_dataset(self),
        }
        if self.tactile_mode == "full":
            force_lower, force_upper = self.force_bounds()
            metadata.update(
                tactile_layout=expected_tactile_layout(),
                force_lower=force_lower.tolist(),
                force_upper=force_upper.tolist(),
            )
        return metadata

    def get_all_actions(self) -> torch.Tensor:
        _, motion = self._all_train_motion_fields()
        if self.tactile_mode == "none":
            return torch.from_numpy(motion)
        tactile = self._all_train_tactile()
        force = aggregate_finger_forces(tactile, self.role_count)
        return torch.from_numpy(interleave_motion_force(motion, force, self.role_count))

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def _format_item(self, item: dict[str, Any]) -> dict[str, Any]:
        motion = np.asarray(item["action"], dtype=np.float32)
        if self.tactile_mode == "full":
            raw_tactile = np.concatenate(
                [
                    np.asarray(item[f"observation.tactile.{role}"], dtype=np.float32)
                    for role in self.roles
                ],
                axis=-1,
            )
            force = aggregate_finger_forces(raw_tactile, self.role_count)
            action = interleave_motion_force(motion, force, self.role_count)
            packed_tactile = pack_finger_taxels(
                raw_tactile[: self.n_obs_steps], self.role_count
            )
        else:
            action = motion
            packed_tactile = np.zeros(
                (self.n_obs_steps, self.finger_count, 29, 1), dtype=np.float32
            )
        obs: dict[str, torch.Tensor] = {
            "state": torch.from_numpy(
                select_compact_policy_state(item["observation.state"], self.embodiment)
            ),
            "finger_raw_pressure": torch.from_numpy(packed_tactile),
        }
        for role in self.camera_roles:
            image = torch.as_tensor(item[f"observation.images.{role}"], dtype=torch.float32)
            if image.shape != (self.n_obs_steps, 3, 224, 224):
                raise ValueError(f"Unexpected KineDex {role} image shape {image.shape}.")
            obs[f"{role}_rgb"] = image
        if action.shape != (self.horizon, self.model_action_dim):
            raise ValueError(f"Unexpected KineDex action shape {action.shape}.")
        return {"obs": obs, "action": torch.from_numpy(action)}

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._format_item(self.dataset[int(self.frame_indices[index])])

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        return [self._format_item(item) for item in fetch_lerobot_batch(self, indices)]


class MultiLeRobotV3KineDexDataset(MultiPolicyDataset, BaseImageDataset):
    def get_normalizer(self, **_: Any) -> Any:
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        from diffusion_policy.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )

        states, motion = self.weighted_arrays(
            lambda dataset: dataset._all_train_motion_fields()
        )
        normalizer = LinearNormalizer()
        normalizer["state"] = SingleFieldLinearNormalizer.create_fit(states)
        if self.tactile_mode == "full":
            (tactile,) = self.weighted_arrays(
                lambda dataset: (dataset._all_train_tactile(),)
            )
            packed = pack_finger_taxels(tactile, self.role_count)
            force = aggregate_finger_forces(tactile, self.role_count)
            action = interleave_motion_force(motion, force, self.role_count)
            normalizer["finger_raw_pressure"] = (
                SingleFieldLinearNormalizer.create_fit(packed)
            )
            self._weighted_force_bounds = (force.min(axis=0), force.max(axis=0))
        else:
            action = motion
            normalizer["finger_raw_pressure"] = self.datasets[0]._zero_tactile_normalizer(
                SingleFieldLinearNormalizer
            )
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(action)
        for role in self.camera_roles:
            normalizer[f"{role}_rgb"] = get_image_range_normalizer()
        return normalizer

    def force_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.tactile_mode != "full":
            raise RuntimeError("NoTouch KineDex has no force bounds.")
        if not hasattr(self, "_weighted_force_bounds"):
            self.get_normalizer()
        return tuple(value.copy() for value in self._weighted_force_bounds)

    def get_all_actions(self) -> torch.Tensor:
        _, motion = self.weighted_arrays(
            lambda dataset: dataset._all_train_motion_fields()
        )
        if self.tactile_mode == "none":
            return torch.from_numpy(motion)
        (tactile,) = self.weighted_arrays(
            lambda dataset: (dataset._all_train_tactile(),)
        )
        force = aggregate_finger_forces(tactile, self.role_count)
        return torch.from_numpy(interleave_motion_force(motion, force, self.role_count))

    def checkpoint_metadata(self) -> dict[str, Any]:
        metadata = dict(self.datasets[0].checkpoint_metadata())
        metadata["dataset_manifest"] = self.training_manifest()
        if self.tactile_mode == "full":
            force_lower, force_upper = self.force_bounds()
            metadata.update(
                force_lower=force_lower.tolist(),
                force_upper=force_upper.tolist(),
            )
        return metadata


def make_lerobot_v3_kinedex_dataset(
    dataset_path: str | None = None,
    repo_id: str | None = None,
    dataset_paths: list[str] | None = None,
    repo_ids: list[str] | None = None,
    replay_dataset_paths: list[str] | None = None,
    replay_repo_ids: list[str] | None = None,
    replay_ratio: float = 0.0,
    dataset_weights: list[float] | None = None,
    **kwargs: Any,
) -> LeRobotV3KineDexDataset | MultiLeRobotV3KineDexDataset:
    if dataset_paths is None and repo_ids is None:
        if dataset_path is None or repo_id is None:
            raise ValueError("Set dataset_path/repo_id or dataset_paths/repo_ids.")
        return LeRobotV3KineDexDataset(dataset_path, repo_id, **kwargs)
    paths = list(dataset_paths or ())
    ids = list(repo_ids or ())
    replay_paths = list(replay_dataset_paths or ())
    replay_ids = list(replay_repo_ids or ())
    if not paths or len(paths) != len(ids):
        raise ValueError("KineDex dataset paths and repo ids must match.")
    if len(replay_paths) != len(replay_ids):
        raise ValueError("KineDex replay paths and repo ids must match.")
    datasets = [
        LeRobotV3KineDexDataset(path, source_id, **kwargs)
        for path, source_id in zip([*paths, *replay_paths], [*ids, *replay_ids], strict=True)
    ]
    if len(datasets) == 1:
        if replay_ratio != 0:
            raise ValueError("replay_ratio must be zero without replay datasets.")
        if dataset_weights not in (None, [1], [1.0]):
            raise ValueError("A single KineDex dataset must have weight 1.0.")
        return datasets[0]
    if dataset_weights is not None and replay_paths:
        raise ValueError(
            "dataset_weights and replay datasets are mutually exclusive; "
            "use explicit weights for one concentrated-sampling source list."
        )
    return MultiLeRobotV3KineDexDataset(
        datasets,
        primary_count=len(paths),
        replay_ratio=float(replay_ratio) if replay_paths else 0.0,
        compatibility_fields=(
            "embodiment", "roles", "camera_roles", "fps", "image_hz", "horizon",
            "n_obs_steps", "state_dim", "motion_action_dim", "model_action_dim",
            "tactile_mode", "schema_hash", "action_spec",
        ),
        source_weights=dataset_weights,
    )
