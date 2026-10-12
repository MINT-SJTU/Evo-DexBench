"""Convert completed canonical HDF5 episodes to LeRobot dataset v3.0."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from dex_benchmark.data.canonical.hdf5 import _require_h5py
from dex_benchmark.data.canonical.schema import jsonable
from dex_benchmark.data.canonical.spool import sha256_file
from dex_benchmark.data.canonical.validate import (
    manifest_checksum_for_output,
    validate_canonical_hdf5,
)


CONVERTER_VERSION = "3"
DEFAULT_FPS = 30
DEFAULT_MAX_HELD_FRAMES = 2
DEFAULT_MAX_HELD_RATIO = 0.05
DEFAULT_IMAGE_WRITER_THREADS = 4


def _read_json_array(dataset: Any) -> list[Any]:
    values = dataset[...]
    result: list[Any] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(json.loads(str(value)))
    return result


def _canonical_files(input_dir: str | Path) -> list[Path]:
    path = Path(input_dir).expanduser().resolve()
    if path.is_file():
        if ".spool" in path.parts or path.name.endswith(".tmp"):
            raise ValueError(f"Canonical converter accepts completed HDF5 only: {path}")
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*.h5")
        if ".spool" not in candidate.parts
        and not any(part.endswith(".tmp") for part in candidate.parts)
        and not candidate.name.endswith(".tmp")
    )


def _filter_files_by_quality_report(
    input_files: list[Path], quality_report_path: str | Path | None
) -> tuple[list[Path], str | None]:
    """Select only episodes marked valid by a quality report."""

    if quality_report_path is None:
        return input_files, None
    report_path = Path(quality_report_path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"Quality report does not exist: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Quality report is not valid JSON: {report_path}") from exc
    episodes = report.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Quality report must contain an 'episodes' list.")
    valid_paths: set[Path] = set()
    valid_ids: set[str] = set()
    for item in episodes:
        if not isinstance(item, Mapping) or not bool(item.get("valid", False)):
            continue
        source_path = item.get("source_path")
        if source_path:
            valid_paths.add(Path(str(source_path)).expanduser().resolve())
        if item.get("episode_id") is not None:
            valid_ids.add(str(item["episode_id"]))
    selected = [
        path
        for path in input_files
        if path in valid_paths or path.stem in valid_ids
    ]
    if not selected:
        raise ValueError("Quality report marked no input episodes as valid.")
    return selected, str(report_path)


def _dataset_keys(group: Any) -> list[str]:
    return sorted(str(key) for key in group.keys()) if group is not None else []


def _meta_value(handle: Any, name: str, default: Any = None) -> Any:
    path = f"meta/{name}.json"
    if path not in handle:
        return default
    value = handle[path][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _flatten_feature_group(group: Any, prefix: str = "") -> list[tuple[str, Any]]:
    features: list[tuple[str, Any]] = []
    if group is None:
        return features
    for key in sorted(group.keys()):
        value = group[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if hasattr(value, "keys"):
            features.extend(_flatten_feature_group(value, name))
        else:
            features.append((name, value))
    return features


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = json.dumps(jsonable(value), ensure_ascii=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _remove_empty_image_directories(output_dir: Path) -> None:
    """Remove LeRobot's empty temporary image directories after video encoding."""

    images_root = output_dir / "images"
    if not images_root.is_dir():
        return
    for path in sorted(images_root.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                # Keep any non-empty directory; only temporary empty directories
                # are safe to remove here.
                pass
    try:
        images_root.rmdir()
    except OSError:
        pass


def _feature_schema_hash(features: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps(jsonable(features), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _manifest_candidates(path: Path) -> list[Path]:
    """Find recorder manifests for the finalized HDF5 without guessing silently."""

    candidates = {
        path.parent / ".spool" / path.stem / "manifest.json",
        path.parent.parent / ".spool" / path.stem / "manifest.json",
        path.parent.parent.parent / ".spool" / path.stem / "manifest.json",
    }
    for spool_root in (
        path.parent / ".spool",
        path.parent.parent / ".spool",
        path.parent.parent.parent / ".spool",
    ):
        candidates.update(spool_root.glob(f"{path.stem}/attempt-*/manifest.json"))
    return sorted(candidate for candidate in candidates if candidate.is_file())


def _manifest_matches_output(manifest_path: Path, output_path: Path) -> bool:
    try:
        manifest_checksum_for_output(manifest_path, output_path)
    except ValueError:
        return False
    return True


def _resolve_manifest(path: Path, manifest_path: str | Path | None) -> Path:
    if manifest_path is not None:
        resolved = Path(manifest_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Checksum manifest does not exist: {resolved}")
        if not _manifest_matches_output(resolved, path):
            raise ValueError(
                f"Recorder manifest {resolved} does not reference HDF5 output {path}."
            )
        return resolved

    matches: list[Path] = []
    for candidate in _manifest_candidates(path):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(manifest, Mapping):
            continue
        output_path = manifest.get("output_path")
        if output_path is not None:
            try:
                recorded_path = Path(str(output_path)).expanduser()
                if not recorded_path.is_absolute():
                    recorded_path = candidate.parent / recorded_path
                if recorded_path.resolve() == path:
                    matches.append(candidate)
            except OSError:
                continue
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple checksum manifests match {path}: {matches}")
    raise FileNotFoundError(
        f"Cannot find a recorder manifest for {path}; pass --manifest explicitly."
    )


def _numeric_json(value: Any, *, field_name: str, dtype: Any = np.float32) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{field_name} must be numeric, got {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} contains NaN or infinite values.")
    result = np.asarray(array, dtype=dtype).copy()
    return result.reshape(1) if result.ndim == 0 else result


class CanonicalEpisodeReader:
    """Read one completed canonical episode at its native control frequency."""

    def __init__(
        self,
        path: str | Path,
        *,
        verify_checksums: bool = False,
        manifest_path: str | Path | None = None,
        max_held_frames: int = DEFAULT_MAX_HELD_FRAMES,
        max_held_ratio: float = DEFAULT_MAX_HELD_RATIO,
        allow_task_id_instruction_fallback: bool = False,
    ) -> None:
        if max_held_frames < 0:
            raise ValueError("max_held_frames must be non-negative.")
        if not 0.0 <= max_held_ratio <= 1.0:
            raise ValueError("max_held_ratio must be in [0, 1].")
        self.path = Path(path).expanduser().resolve()
        self.h5py = _require_h5py()
        self.handle = self.h5py.File(self.path, "r")
        self._closed = False
        try:
            resolved_manifest = (
                _resolve_manifest(self.path, manifest_path)
                if verify_checksums
                else None
            )
            validate_canonical_hdf5(
                self.path,
                manifest_path=resolved_manifest,
                verify_checksums=verify_checksums,
            )
            self._validate_completed_status()
            self._validate_terminal_flags()
            self.transition_ids = np.asarray(self.handle["transitions/transition_id"][:])
            self.observation_ids_t = np.asarray(
                self.handle["transitions/observation_id_t"][:]
            )
            self.observation_ids = np.asarray(self.handle["observations/observation_id"][:])
            self.observation_index = {
                int(value): index for index, value in enumerate(self.observation_ids)
            }
            self.control_hz = float(self.handle.attrs.get("control_hz", math.nan))
            if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
                raise ValueError(f"{self.path}: HDF5 control_hz must be finite and positive.")
            self.control_timestamps = np.asarray(
                self.handle["transitions/timestamps/control_tick_s"][:], dtype=np.float64
            )
            if not np.isfinite(self.control_timestamps).all():
                raise ValueError(
                    f"{self.path}: control_tick_s must be finite for every transition."
                )
            if len(self.control_timestamps) > 1 and np.any(
                np.diff(self.control_timestamps) < 0.0
            ):
                raise ValueError(f"{self.path}: control_tick_s must be monotonic.")
            self.held_actions = np.asarray(
                self.handle["transitions/held_action"][:], dtype=np.bool_
            )
            if len(self.held_actions) != len(self.transition_ids):
                raise ValueError(f"{self.path}: held_action length does not match transitions.")
            self.held_count = int(self.held_actions.sum())
            self.held_ratio = self.held_count / len(self.transition_ids)
            self.max_held_frames = int(max_held_frames)
            self.max_held_ratio = float(max_held_ratio)
            self.reward_values = _read_json_array(self.handle["transitions/reward_json"])
            self.terminated_values = _read_json_array(
                self.handle["transitions/terminated_json"]
            )
            self.truncated_values = _read_json_array(
                self.handle["transitions/truncated_json"]
            )
            self.state_features = _flatten_feature_group(
                self.handle.get("observations/state")
            )
            self.taxel_features = _flatten_feature_group(
                self.handle.get("observations/taxel")
            )
            self.action_features = _flatten_feature_group(
                self.handle.get("actions/semantic")
            )
            self.image_cameras = _dataset_keys(self.handle.get("observations/images"))
            self.depth_cameras = _dataset_keys(self.handle.get("observations/depth"))
            self.image_ids = {
                camera: np.asarray(
                    self.handle[f"observations/image_observation_id/{camera}"][:],
                    dtype=np.int64,
                )
                for camera in self.image_cameras
            }
            self._validate_feature_shapes()
            self._instruction = self._read_instruction(
                allow_task_id_instruction_fallback=allow_task_id_instruction_fallback
            )
        except BaseException:
            self.handle.close()
            self._closed = True
            raise

    def _validate_completed_status(self) -> None:
        status = _meta_value(self.handle, "episode_status")
        if status != "completed":
            raise ValueError(
                f"{self.path}: only completed canonical episodes can be converted; "
                f"got episode_status={status!r}."
            )

    def _validate_terminal_flags(self) -> None:
        length = len(self.handle["transitions/transition_id"])
        for name in ("terminated_json", "truncated_json"):
            values = _read_json_array(self.handle[f"transitions/{name}"])
            flags = [bool(np.asarray(value).any()) for value in values]
            if any(flags[:-1]):
                raise ValueError(f"{self.path}: {name} is true before the final transition.")
            if len(values) != length:
                raise ValueError(f"{self.path}: {name} length does not match transitions.")

    def _validate_feature_shapes(self) -> None:
        if not self.state_features or not self.action_features:
            raise ValueError(f"{self.path}: state and semantic action features are required.")
        transition_count = len(self.transition_ids)
        observation_count = len(self.observation_ids)

        def validate_group(group: Any, expected_length: int, prefix: str) -> None:
            for name, dataset in _flatten_feature_group(group, prefix):
                if dataset.ndim == 0 or dataset.shape[0] != expected_length:
                    raise ValueError(
                        f"{self.path}: {name!r} must have {expected_length} samples, "
                        f"got shape {dataset.shape}."
                    )
                values = np.asarray(dataset[...])
                if not np.isfinite(values).all():
                    raise ValueError(f"{self.path}: feature {name!r} contains non-finite values.")

        # Validate source datasets in bulk. This replaces an O(frames) sequence of
        # derived frame() calls, while preserving the same shape and finite-value checks.
        validate_group(self.handle.get("observations/state"), observation_count, "observations.state")
        validate_group(self.handle.get("observations/taxel"), observation_count, "observations.taxel")
        validate_group(self.handle.get("actions/semantic"), transition_count, "actions.semantic")

        for camera in self.image_cameras:
            dataset = self.handle[f"observations/images/{camera}"]
            if dataset.dtype != np.uint8 or dataset.ndim != 4 or dataset.shape[-1] != 3:
                raise ValueError(
                    f"{self.path}: observation.images.{camera} must be uint8 [N,H,W,3], "
                    f"got {dataset.dtype}/{dataset.shape}."
                )
            if len(self.image_ids[camera]) != dataset.shape[0]:
                raise ValueError(f"{self.path}: image IDs do not match {camera} samples.")
        for camera in self.depth_cameras:
            dataset = self.handle[f"observations/depth/{camera}"]
            if dataset.ndim != 3 or any(int(dim) == 0 for dim in dataset.shape):
                raise ValueError(
                    f"{self.path}: observation.depth.{camera} must be [N,H,W], "
                    f"got {dataset.shape}."
                )

        # Cache the only derived frame needed to construct the feature schema.
        self._first_frame = self.frame(0)

    def __len__(self) -> int:
        return len(self.transition_ids)

    @property
    def accepted_by_held_policy(self) -> bool:
        """Keep isolated held ticks; reject episodes with sustained input gaps."""

        return self.held_count <= self.max_held_frames or self.held_ratio <= self.max_held_ratio

    def close(self) -> None:
        if not self._closed:
            self.handle.close()
            self._closed = True

    def __enter__(self) -> "CanonicalEpisodeReader":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _observation_value(self, feature: Any, observation_id: int) -> np.ndarray:
        index = self.observation_index[int(observation_id)]
        return np.asarray(feature[index])

    def _latest_image(self, camera: str, observation_id: int) -> np.ndarray:
        image_ids = self.image_ids[camera]
        selected = int(np.searchsorted(image_ids, int(observation_id), side="right") - 1)
        dataset = self.handle[f"observations/images/{camera}"]
        if selected < 0:
            return np.zeros(dataset.shape[1:], dtype=np.uint8)
        return np.asarray(dataset[selected])

    def _latest_depth(self, camera: str, observation_id: int) -> np.ndarray:
        image_ids = self.image_ids[camera]
        selected = int(np.searchsorted(image_ids, int(observation_id), side="right") - 1)
        dataset = self.handle[f"observations/depth/{camera}"]
        if selected < 0:
            return np.zeros(dataset.shape[1:], dtype=dataset.dtype)
        return np.asarray(dataset[selected])

    def _transition_metadata(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reward = self.reward_values[index]
        terminated = self.terminated_values[index]
        truncated = self.truncated_values[index]
        return (
            _numeric_json(reward, field_name="reward"),
            np.asarray(terminated, dtype=np.bool_).reshape(-1),
            np.asarray(truncated, dtype=np.bool_).reshape(-1),
        )

    def frame(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        observation_id = int(self.observation_ids_t[index])
        state_parts = [
            self._observation_value(feature, observation_id).reshape(-1)
            for _, feature in self.state_features
        ]
        action_parts = [
            np.asarray(feature[index]).reshape(-1)
            for _, feature in self.action_features
        ]
        reward, terminated, truncated = self._transition_metadata(index)
        frame: dict[str, Any] = {
            "observation.state": np.concatenate(state_parts).astype(np.float32)
            if state_parts
            else np.empty(0, dtype=np.float32),
            "action": np.concatenate(action_parts).astype(np.float32)
            if action_parts
            else np.empty(0, dtype=np.float32),
            "next.reward": reward.astype(np.float32, copy=False),
            "next.terminated": terminated,
            "next.truncated": truncated,
        }
        for name, feature in self.taxel_features:
            frame[f"observation.tactile.{name}"] = self._observation_value(
                feature, observation_id
            ).astype(np.float32, copy=False)
        for camera in self.image_cameras:
            frame[f"observation.images.{camera}"] = self._latest_image(camera, observation_id)
        for camera in self.depth_cameras:
            frame[f"observation.depth.{camera}"] = self._latest_depth(
                camera, observation_id
            ).astype(np.float32, copy=False)
        # The canonical runner already makes a held tick safe: hand targets stay
        # at their previous target and arm translation is zero. Keep it as a normal
        # 30 Hz frame; held_action is intentionally not a model feature.
        return frame

    def _read_instruction(self, *, allow_task_id_instruction_fallback: bool) -> str:
        metadata = _meta_value(self.handle, "episode_metadata", {})
        if isinstance(metadata, Mapping):
            instruction = metadata.get("instruction")
            if isinstance(instruction, str) and instruction.strip():
                return instruction.strip()
        if allow_task_id_instruction_fallback:
            task_id = self.handle.attrs.get("task_id", "")
            if str(task_id).strip():
                return str(task_id)
        raise ValueError(
            f"{self.path}: missing episode instruction; provide "
            "canonical episode_metadata.instruction or explicitly enable task-id fallback."
        )

    def instruction(self) -> str:
        return self._instruction

    def feature_specs(self) -> dict[str, dict[str, Any]]:
        first = self._first_frame
        specs: dict[str, dict[str, Any]] = {}
        for name, value in first.items():
            array = np.asarray(value)
            if name.startswith("observation.images."):
                specs[name] = {
                    "dtype": "video",
                    "shape": tuple(int(dim) for dim in array.shape),
                    "names": ["height", "width", "channel"],
                }
            elif name.startswith("observation.depth."):
                specs[name] = {
                    "dtype": "float32",
                    "shape": tuple(int(dim) for dim in array.shape),
                    "names": ["height", "width"],
                }
            elif array.dtype == np.bool_:
                specs[name] = {
                    "dtype": "bool",
                    "shape": tuple(int(dim) for dim in array.shape),
                }
            else:
                specs[name] = {
                    "dtype": "float32",
                    "shape": tuple(int(dim) for dim in array.shape),
                }
        return specs

    def project_metadata(self) -> dict[str, Any]:
        episode_metadata = _meta_value(self.handle, "episode_metadata", {})
        action_spec = _meta_value(self.handle, "action_spec", {})
        sensors = _meta_value(self.handle, "sensors", {})
        depth_encoding = (
            sensors.get("depth_encoding", "unknown")
            if isinstance(sensors, Mapping)
            else "unknown"
        )
        return {
            "episode_id": str(self.handle.attrs.get("episode_id", self.path.stem)),
            "task_id": str(self.handle.attrs.get("task_id", "")),
            "instruction": self.instruction(),
            "frame_count": len(self),
            "embodiment_id": str(self.handle.attrs.get("embodiment_id", "")),
            "control_hz": self.control_hz,
            "image_hz": (
                float(self.handle.attrs["image_hz"])
                if "image_hz" in self.handle.attrs
                else None
            ),
            "layout_id": (
                episode_metadata.get("layout_id")
                if isinstance(episode_metadata, Mapping)
                else None
            ),
            "manipulator_roles": (
                episode_metadata.get("manipulator_roles", [])
                if isinstance(episode_metadata, Mapping)
                else []
            ),
            "state_feature_order": [name for name, _ in self.state_features],
            "action_feature_order": [name for name, _ in self.action_features],
            "tactile_feature_order": [name for name, _ in self.taxel_features],
            "tactile_feature_specs": (
                sensors.get("tactile_layouts", {})
                if isinstance(sensors, Mapping)
                else {}
            ),
            "camera_roles": self.image_cameras,
            "depth_encoding": depth_encoding,
            "depth_source_dtypes": {
                camera: str(self.handle[f"observations/depth/{camera}"].dtype)
                for camera in self.depth_cameras
            },
            "action_spec": action_spec,
            "source_schema_version": int(self.handle.attrs.get("schema_version", -1)),
            "held_action_count": self.held_count,
            "held_action_ratio": self.held_ratio,
        }


def _create_lerobot_dataset(
    *,
    repo_id: str,
    output_dir: Path,
    fps: int,
    features: Mapping[str, Mapping[str, Any]],
    image_writer_processes: int = 0,
    image_writer_threads: int = DEFAULT_IMAGE_WRITER_THREADS,
) -> Any:
    try:
        from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
        except ImportError as exc:  # pragma: no cover - optional training stack.
            raise ImportError(
                "HDF5 -> LeRobot v3.0 conversion requires a LeRobot v3.0 runtime."
            ) from exc

    if CODEBASE_VERSION != "v3.0":
        raise RuntimeError(
            "This converter targets LeRobot dataset codebase v3.0, "
            f"but the installed package reports {CODEBASE_VERSION!r}."
        )
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=int(fps),
        features=dict(features),
        root=output_dir,
        use_videos=True,
        image_writer_processes=int(image_writer_processes),
        image_writer_threads=int(image_writer_threads),
    )


def _validate_lerobot_output(
    output_dir: Path,
    *,
    repo_id: str,
    fps: int,
    features: Mapping[str, Mapping[str, Any]],
    episode_count: int,
    frame_count: int,
) -> None:
    info_path = output_dir / "meta" / "info.json"
    tasks_path = output_dir / "meta" / "tasks.parquet"
    episodes_path = output_dir / "meta" / "episodes"
    if not info_path.is_file() or not tasks_path.is_file() or not episodes_path.is_dir():
        raise ValueError(
            "LeRobot v3 output is missing info.json, tasks.parquet, or episodes metadata."
        )
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(repo_id=repo_id, root=output_dir)
    metadata = getattr(dataset, "meta", None)
    if metadata is None:
        raise ValueError("Unable to reopen generated LeRobot dataset metadata.")
    actual_fps = int(getattr(metadata, "fps", fps))
    if actual_fps != fps:
        raise ValueError(f"Generated LeRobot fps={actual_fps}, expected {fps}.")
    actual_episodes = getattr(metadata, "total_episodes", None)
    if actual_episodes is not None and int(actual_episodes) != episode_count:
        raise ValueError(
            f"Generated LeRobot episode count={actual_episodes}, expected {episode_count}."
        )
    actual_frames = getattr(metadata, "total_frames", None)
    if actual_frames is not None and int(actual_frames) != frame_count:
        raise ValueError(
            f"Generated LeRobot frame count={actual_frames}, expected {frame_count}."
        )
    metadata_features = getattr(metadata, "features", None)
    if isinstance(metadata_features, Mapping):
        for name in features:
            if name not in metadata_features:
                raise ValueError(f"Generated LeRobot dataset is missing feature {name!r}.")
            if (
                name.startswith("observation.images.")
                and metadata_features[name].get("dtype") != "video"
            ):
                raise ValueError(f"Generated RGB feature {name!r} is not stored as video.")

    video_keys = list(getattr(metadata, "video_keys", []))
    expected_video_keys = sorted(
        name for name in features if name.startswith("observation.images.")
    )
    if sorted(video_keys) != expected_video_keys:
        raise ValueError(
            f"Generated video keys {sorted(video_keys)} differ from RGB features "
            f"{expected_video_keys}."
        )
    video_path = getattr(metadata, "video_path", None)
    if not video_path:
        raise ValueError("Generated LeRobot metadata has no video_path.")
    for episode_index in range(episode_count):
        for video_key in video_keys:
            video_file = output_dir / metadata.get_video_file_path(episode_index, video_key)
            if not video_file.is_file() or video_file.stat().st_size == 0:
                raise ValueError(f"Missing or empty video file: {video_file}")

    # Video features are intentionally absent from the frame parquet files.
    try:
        import pyarrow.parquet as pq
        data_files = sorted((output_dir / "data").rglob("*.parquet"))
        for data_file in data_files:
            columns = pq.ParquetFile(data_file).schema.names
            embedded_images = [
                column for column in columns if column.startswith("observation.images.")
            ]
            if embedded_images:
                raise ValueError(
                    f"Video dataset contains embedded RGB columns in {data_file}: "
                    f"{embedded_images}"
                )
    except ImportError as exc:  # pragma: no cover - required by LeRobot runtime.
        raise RuntimeError("pyarrow is required to validate LeRobot parquet schema.") from exc

    # Decode one sample through the public dataset API so a successful metadata
    # reopen also proves that the generated video stream is readable.
    if episode_count and video_keys:
        sample = dataset[0]
        for video_key in video_keys:
            value = sample.get(video_key)
            if value is None or getattr(value, "ndim", 0) != 3:
                raise ValueError(
                    f"Unable to decode video feature {video_key!r} from generated dataset."
                )


def _load_conversion_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Conversion manifest must contain an object: {path}")
    return value


def _validate_existing_output(
    output: Path,
    *,
    repo_id: str,
    fps: int,
    conversion_manifest: Mapping[str, Any],
) -> None:
    """Validate an existing output before returning it for an idempotent call."""

    extension_path = output / "meta" / "dex_benchmark.json"
    if not extension_path.is_file():
        raise ValueError(f"Existing LeRobot output is missing extension metadata: {extension_path}")
    with extension_path.open("r", encoding="utf-8") as handle:
        extension = json.load(handle)
    if not isinstance(extension, Mapping):
        raise ValueError(f"Extension metadata must contain an object: {extension_path}")
    episodes = extension.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Existing LeRobot output has no accepted episode metadata.")
    if not all(isinstance(item, Mapping) for item in episodes):
        raise ValueError("Existing LeRobot episode metadata contains a non-object entry.")
    if extension.get("feature_schema_sha256") != conversion_manifest.get("feature_schema_sha256"):
        raise ValueError("Existing LeRobot extension metadata has a different feature schema.")
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as exc:  # pragma: no cover - optional training stack.
            raise ImportError(
                "Validating an existing LeRobot output requires a LeRobot v3.0 runtime."
            ) from exc
    dataset = LeRobotDataset(repo_id=repo_id, root=output)
    metadata = getattr(dataset, "meta", None)
    if metadata is None:
        raise ValueError("Unable to reopen existing LeRobot dataset metadata.")
    actual_fps = int(getattr(metadata, "fps", -1))
    if actual_fps != fps:
        raise ValueError(f"Existing LeRobot fps={actual_fps}, expected {fps}.")
    actual_episodes = getattr(metadata, "total_episodes", None)
    if actual_episodes is not None and int(actual_episodes) != len(episodes):
        raise ValueError(
            f"Existing LeRobot episode count={actual_episodes}, expected {len(episodes)}."
        )
    actual_frames = getattr(metadata, "total_frames", None)
    expected_frames = sum(int(item.get("frame_count", 0)) for item in episodes)
    if actual_frames is not None and int(actual_frames) != expected_frames:
        raise ValueError(
            f"Existing LeRobot frame count={actual_frames}, expected {expected_frames}."
        )


def convert_to_lerobot(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    repo_id: str | None = None,
    fps: float = DEFAULT_FPS,
    verify_checksums: bool = False,
    manifest_path: str | Path | None = None,
    quality_report_path: str | Path | None = None,
    max_held_frames: int = DEFAULT_MAX_HELD_FRAMES,
    max_held_ratio: float = DEFAULT_MAX_HELD_RATIO,
    allow_task_id_instruction_fallback: bool = False,
    image_writer_processes: int = 0,
    image_writer_threads: int = DEFAULT_IMAGE_WRITER_THREADS,
) -> Path:
    """Convert completed canonical episodes into a LeRobot v3.0 dataset."""

    if fps <= 0 or not float(fps).is_integer():
        raise ValueError("LeRobot v3.0 requires a positive integer fps with no resampling.")
    if manifest_path is not None and not verify_checksums:
        raise ValueError("--manifest requires --verify-checksums so it cannot be ignored.")
    if image_writer_processes < 0 or image_writer_threads < 0:
        raise ValueError("Image writer process/thread counts must be non-negative.")
    if image_writer_processes == 0 and image_writer_threads == 0:
        raise ValueError("At least one image writer process or thread is required.")
    target_fps = int(fps)
    input_files = _canonical_files(input_dir)
    if not input_files:
        raise FileNotFoundError(f"No canonical .h5 episodes found under {input_dir!r}.")
    input_files, quality_report = _filter_files_by_quality_report(
        input_files, quality_report_path
    )
    output = Path(output_dir).expanduser().resolve()
    dataset_id = repo_id or output.name
    conversion_manifest_path = output.with_name(output.name + ".conversion_manifest.json")
    source_records = [
        {"path": str(path), "sha256": sha256_file(path), "episode_id": path.stem}
        for path in input_files
    ]
    existing_manifest = _load_conversion_manifest(conversion_manifest_path)
    if output.exists():
        expected = {
            "format": "lerobot",
            "codebase_version": "v3.0",
            "converter_version": CONVERTER_VERSION,
            "repo_id": dataset_id,
            "fps": target_fps,
            "source_episodes": source_records,
            "max_held_frames": int(max_held_frames),
            "max_held_ratio": float(max_held_ratio),
            "verify_checksums": bool(verify_checksums),
            "quality_report": quality_report,
            "image_writer_processes": int(image_writer_processes),
            "image_writer_threads": int(image_writer_threads),
        }
        if existing_manifest is None or any(
            existing_manifest.get(key) != value for key, value in expected.items()
        ):
            raise FileExistsError(
                "LeRobot output already exists and does not match the requested "
                f"conversion: {output}"
            )
        if verify_checksums:
            for path in input_files:
                reader = CanonicalEpisodeReader(
                    path,
                    verify_checksums=True,
                    manifest_path=manifest_path,
                    max_held_frames=max_held_frames,
                    max_held_ratio=max_held_ratio,
                    allow_task_id_instruction_fallback=allow_task_id_instruction_fallback,
                )
                reader.close()
        _validate_existing_output(
            output,
            repo_id=dataset_id,
            fps=target_fps,
            conversion_manifest=existing_manifest,
        )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    if temporary_output.exists():
        raise FileExistsError(
            "An interrupted conversion temporary directory exists; remove it before "
            f"restarting: {temporary_output}"
        )

    readers: list[CanonicalEpisodeReader] = []
    skipped: list[dict[str, Any]] = []
    dataset: Any | None = None
    try:
        for path in input_files:
            reader = CanonicalEpisodeReader(
                path,
                verify_checksums=verify_checksums,
                manifest_path=manifest_path,
                max_held_frames=max_held_frames,
                max_held_ratio=max_held_ratio,
                allow_task_id_instruction_fallback=allow_task_id_instruction_fallback,
            )
            if reader.control_hz != float(target_fps):
                control_hz = reader.control_hz
                reader.close()
                raise ValueError(
                    f"{path}: HDF5 control_hz={control_hz} does not equal output fps={target_fps}; "
                    "resampling is disabled."
                )
            if not reader.accepted_by_held_policy:
                skipped.append(
                    {
                        "path": str(path),
                        "episode_id": path.stem,
                        "held_action_count": reader.held_count,
                        "held_action_ratio": reader.held_ratio,
                        "reason": "held_action_ratio_exceeded",
                    }
                )
                reader.close()
                continue
            readers.append(reader)
        if not readers:
            raise ValueError("No canonical episodes remain after held-action quality filtering.")

        features = readers[0].feature_specs()
        schema_hash = _feature_schema_hash(features)
        for reader in readers[1:]:
            if reader.feature_specs() != features:
                raise ValueError(
                    f"Canonical episode {reader.path} feature schema differs from the "
                    "first accepted episode."
                )

        dataset = _create_lerobot_dataset(
            repo_id=dataset_id,
            output_dir=temporary_output,
            fps=target_fps,
            features=features,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )
        accepted_metadata: list[dict[str, Any]] = []
        frame_count = 0
        for reader in readers:
            for index in range(len(reader)):
                frame = reader.frame(index)
                # LeRobot 0.4.x consumes task from the frame dictionary.
                frame["task"] = reader.instruction()
                dataset.add_frame(frame)
            dataset.save_episode(parallel_encoding=True)
            accepted_metadata.append(reader.project_metadata())
            frame_count += len(reader)

        # LeRobot 0.4.x buffers meta/episodes parquet rows. Close its writers
        # before reopening the dataset, otherwise the parquet footer is absent.
        finalize = getattr(dataset, "finalize", None)
        if not callable(finalize):
            raise RuntimeError(
                "The installed LeRobot runtime does not expose finalize(); "
                "cannot safely validate the generated dataset."
            )
        finalize()
        stop_image_writer = getattr(dataset, "stop_image_writer", None)
        if callable(stop_image_writer):
            stop_image_writer()
        _remove_empty_image_directories(temporary_output)

        extension_metadata = {
            "schema_version": 1,
            "converter_version": CONVERTER_VERSION,
            "format": "lerobot",
            "codebase_version": "v3.0",
            "fps": target_fps,
            "feature_schema_sha256": schema_hash,
            "episodes": accepted_metadata,
            "skipped_episodes": skipped,
            "held_action_policy": {
                "max_held_frames": int(max_held_frames),
                "max_held_ratio": float(max_held_ratio),
                "held_action_is_model_feature": False,
                "kept_held_ticks_as_normal_frames": True,
            },
        }
        _atomic_write_json(temporary_output / "meta" / "dex_benchmark.json", extension_metadata)
        _validate_lerobot_output(
            temporary_output,
            repo_id=dataset_id,
            fps=target_fps,
            features=features,
            episode_count=len(readers),
            frame_count=frame_count,
        )
    except BaseException:
        if dataset is not None:
            stop_image_writer = getattr(dataset, "stop_image_writer", None)
            if callable(stop_image_writer):
                stop_image_writer()
        if temporary_output.is_dir():
            shutil.rmtree(temporary_output)
        elif temporary_output.exists():
            temporary_output.unlink()
        raise
    finally:
        for reader in readers:
            reader.close()

    temporary_output.replace(output)
    manifest = {
        "format": "lerobot",
        "codebase_version": "v3.0",
        "converter_version": CONVERTER_VERSION,
        "repo_id": dataset_id,
        "fps": target_fps,
        "source_episodes": source_records,
        "skipped_episodes": skipped,
        "max_held_frames": int(max_held_frames),
        "max_held_ratio": float(max_held_ratio),
        "verify_checksums": bool(verify_checksums),
        "quality_report": quality_report,
        "image_writer_processes": int(image_writer_processes),
        "image_writer_threads": int(image_writer_threads),
        "feature_schema_sha256": schema_hash,
    }
    _atomic_write_json(conversion_manifest_path, manifest)
    return output
