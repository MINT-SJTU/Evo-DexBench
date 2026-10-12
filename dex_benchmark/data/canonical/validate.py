"""Validation checks for completed canonical HDF5 episodes."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.data.canonical.hdf5 import _require_h5py
from dex_benchmark.data.canonical.spool import sha256_file
from dex_benchmark.data.canonical.schema import CANONICAL_SCHEMA_VERSION


def _read_json_array(dataset: Any) -> list[Any]:
    values = dataset[...]
    result: list[Any] = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        result.append(json.loads(str(value)))
    return result


def _require_dataset(handle: Any, path: str) -> Any:
    if path not in handle:
        raise ValueError(f"Canonical HDF5 is missing required dataset {path!r}.")
    return handle[path]


def manifest_checksum_for_output(
    manifest_path: str | Path,
    output_path: str | Path,
) -> str:
    """Resolve an episode checksum from recorder or collection manifests."""

    resolved_manifest = Path(manifest_path).expanduser().resolve()
    resolved_output = Path(output_path).expanduser().resolve()
    try:
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid checksum manifest: {resolved_manifest}") from exc
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Checksum manifest must contain an object: {resolved_manifest}")

    records: list[Mapping[str, Any]] = [manifest]
    episodes = manifest.get("episodes", [])
    if isinstance(episodes, list):
        for episode in episodes:
            if not isinstance(episode, Mapping):
                continue
            attempts = episode.get("attempts", [])
            if isinstance(attempts, list):
                records.extend(
                    attempt for attempt in attempts if isinstance(attempt, Mapping)
                )

    for record in records:
        recorded_output = record.get("output_path")
        if recorded_output is None:
            continue
        recorded_path = Path(str(recorded_output)).expanduser()
        if not recorded_path.is_absolute():
            recorded_path = resolved_manifest.parent / recorded_path
        if recorded_path.resolve() != resolved_output:
            continue
        expected = record.get("output_sha256")
        validation = record.get("validation")
        if expected is None and isinstance(validation, Mapping):
            expected = validation.get("output_sha256")
        if expected is None:
            raise ValueError(
                f"Checksum manifest {resolved_manifest} references {resolved_output} "
                "but does not record its SHA256."
            )
        return str(expected)

    raise ValueError(
        f"Checksum manifest {resolved_manifest} does not reference HDF5 output "
        f"{resolved_output}."
    )


def validate_canonical_hdf5(
    path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    verify_checksums: bool = False,
    required_cameras: tuple[str, ...] = (),
    required_tactile_roles: tuple[str, ...] = (),
    check_quest: bool = True,
) -> dict[str, Any]:
    """Validate one episode and return a compact report.

    Raises ``ValueError`` for structural/data-integrity failures.  The report is
    intentionally JSON-compatible so it can be written by a CLI or CI job.
    """

    h5py = _require_h5py()
    episode_path = Path(path).expanduser().resolve()
    if not episode_path.exists():
        raise FileNotFoundError(episode_path)
    with h5py.File(episode_path, "r") as handle:
        version = int(handle.attrs.get("schema_version", -1))
        if version != CANONICAL_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported canonical schema version {version}; "
                f"expected {CANONICAL_SCHEMA_VERSION}."
            )
        transition_ids = np.asarray(_require_dataset(handle, "transitions/transition_id"))
        observation_ids_t = np.asarray(
            _require_dataset(handle, "transitions/observation_id_t")
        )
        action_ids = np.asarray(_require_dataset(handle, "transitions/action_id"))
        observation_ids_tp1 = np.asarray(
            _require_dataset(handle, "transitions/observation_id_tp1")
        )
        held_action = np.asarray(_require_dataset(handle, "transitions/held_action"))
        transition_count = len(transition_ids)
        quest_sequence = (
            np.asarray(_require_dataset(handle, "transitions/quest_sequence"))
            if check_quest
            else np.full(transition_count, -1, dtype=np.int64)
        )
        if transition_count == 0:
            raise ValueError("Canonical episode contains no transitions.")
        if "chunk_index" in handle.attrs:
            raise ValueError(
                "Canonical episode root must not contain the chunk-local "
                "chunk_index attribute."
            )
        expected_episode_attributes = {
            "first_transition_id": int(transition_ids[0]),
            "last_transition_id": int(transition_ids[-1]),
            "transition_count": int(transition_count),
        }
        for attribute_name, expected_value in expected_episode_attributes.items():
            if attribute_name not in handle.attrs:
                raise ValueError(
                    f"Canonical episode is missing root attribute "
                    f"{attribute_name!r}."
                )
            actual_value = int(handle.attrs[attribute_name])
            if actual_value != expected_value:
                raise ValueError(
                    f"Canonical episode root attribute {attribute_name!r} is "
                    f"{actual_value}, expected {expected_value}."
                )
        chunk_count = int(handle.attrs.get("chunk_count", 0))
        if chunk_count <= 0:
            raise ValueError("Canonical episode chunk_count must be positive.")
        transition_datasets = (
            ("transition_id", transition_ids),
            ("observation_id_t", observation_ids_t),
            ("action_id", action_ids),
            ("observation_id_tp1", observation_ids_tp1),
            ("held_action", held_action),
        )
        if check_quest and len(quest_sequence) != transition_count:
            raise ValueError(
                f"transitions/quest_sequence has length {len(quest_sequence)}, "
                f"expected {transition_count}."
            )
        for name, values in transition_datasets:
            if len(values) != transition_count:
                raise ValueError(
                    f"transitions/{name} has length {len(values)}, "
                    f"expected {transition_count}."
                )
        if not np.array_equal(transition_ids, np.arange(transition_count)):
            raise ValueError("transition_id must be contiguous starting at zero.")
        if not np.array_equal(action_ids, np.arange(transition_count)):
            raise ValueError("action_id must be contiguous starting at zero.")
        if np.any(observation_ids_t >= observation_ids_tp1):
            raise ValueError("Each transition must reference obs_t before obs_t+1.")
        if transition_count > 1 and not np.array_equal(
            observation_ids_t[1:], observation_ids_tp1[:-1]
        ):
            raise ValueError(
                "Adjacent transitions must share the previous obs_t+1 observation."
            )
        observation_ids = np.asarray(_require_dataset(handle, "observations/observation_id"))
        if len(np.unique(observation_ids)) != len(observation_ids):
            raise ValueError("observations/observation_id contains duplicates.")
        observation_id_set = {int(value) for value in observation_ids}
        if not set(observation_ids_t).issubset(observation_id_set):
            raise ValueError("Some observation_id_t values are not materialized.")
        if not set(observation_ids_tp1).issubset(observation_id_set):
            raise ValueError("Some observation_id_tp1 values are not materialized.")
        action_materialized = np.asarray(_require_dataset(handle, "actions/action_id"))
        if not np.array_equal(action_materialized, action_ids):
            raise ValueError("actions/action_id does not match transitions/action_id.")
        transition_times = np.asarray(
            _require_dataset(handle, "transitions/timestamps/control_tick_s")
        )
        if np.any(np.diff(transition_times[np.isfinite(transition_times)]) < 0):
            raise ValueError("control_tick_s must be monotonic.")
        reward_count = len(_read_json_array(_require_dataset(handle, "transitions/reward_json")))
        if reward_count != transition_count:
            raise ValueError("reward_json length does not match transition count.")
        report: dict[str, Any] = {
            "path": str(episode_path),
            "schema_version": version,
            "transition_count": transition_count,
            "chunk_count": chunk_count,
            "observation_count": len(observation_ids),
            "held_action_count": int(np.asarray(held_action, dtype=np.bool_).sum()),
            "quest_reference_count": (
                int(np.count_nonzero(quest_sequence >= 0)) if check_quest else None
            ),
            "image_cameras": sorted(
                str(name) for name in handle.get("observations/images", {}).keys()
            ),
            "tactile_roles": sorted(
                str(name) for name in handle.get("observations/taxel", {}).keys()
            ),
        }
        configured_image_stride: int | None = None
        configured_control_hz = handle.attrs.get("control_hz")
        configured_image_hz = handle.attrs.get("image_hz")
        if configured_control_hz is not None and configured_image_hz is not None:
            control_hz = float(configured_control_hz)
            image_hz = float(configured_image_hz)
            raw_stride = control_hz / image_hz
            if (
                control_hz <= 0.0
                or image_hz <= 0.0
                or image_hz > control_hz
                or not np.isclose(raw_stride, round(raw_stride), atol=1e-6)
            ):
                raise ValueError(
                    "Canonical episode control_hz/image_hz must define a "
                    "positive integral image stride."
                )
            configured_image_stride = int(round(raw_stride))
            report["configured_control_hz"] = control_hz
            report["configured_image_hz"] = image_hz
            report["configured_image_stride"] = configured_image_stride
            report["image_sampling"] = {}
        image_group = handle.get("observations/images")
        depth_group = handle.get("observations/depth")
        image_id_group = handle.get("observations/image_observation_id")
        if image_group is not None:
            if depth_group is None or image_id_group is None:
                raise ValueError(
                    "RGB observations require depth and image_observation_id groups."
                )
            for camera in image_group.keys():
                if camera not in depth_group or camera not in image_id_group:
                    raise ValueError(
                        f"Camera {camera!r} is missing depth or observation IDs."
                    )
                if len(image_group[camera]) != len(depth_group[camera]):
                    raise ValueError(f"RGB/depth sample count differs for {camera!r}.")
                camera_ids = np.asarray(image_id_group[camera][:], dtype=np.int64)
                if len(camera_ids) != len(image_group[camera]):
                    raise ValueError(
                        f"Image observation ID count differs for {camera!r}."
                    )
                if np.any(np.diff(camera_ids) < 0):
                    raise ValueError(f"Image observation IDs are not monotonic for {camera!r}.")
                if len(np.unique(camera_ids)) != len(camera_ids):
                    raise ValueError(
                        f"Image observation IDs contain duplicates for {camera!r}."
                    )
                if not set(map(int, camera_ids)).issubset(observation_id_set):
                    raise ValueError(
                        f"Image observation IDs reference missing observations for {camera!r}."
                    )
                if configured_image_stride is not None:
                    scheduled_ids = observation_ids[
                        observation_ids % configured_image_stride == 0
                    ]
                    missing_scheduled_ids = sorted(
                        set(map(int, scheduled_ids)) - set(map(int, camera_ids))
                    )
                    if missing_scheduled_ids:
                        raise ValueError(
                            f"Camera {camera!r} is missing configured image samples at "
                            f"observation IDs {missing_scheduled_ids[:10]}."
                        )
                    report["image_sampling"][str(camera)] = {
                        "sample_count": int(len(camera_ids)),
                        "effective_hz": (
                            float(configured_control_hz) * len(camera_ids)
                            / len(observation_ids)
                        ),
                    }
        available_cameras = set(report["image_cameras"])
        missing_cameras = sorted(set(map(str, required_cameras)) - available_cameras)
        if missing_cameras:
            raise ValueError(
                "Canonical episode is missing required image cameras: "
                + ", ".join(missing_cameras)
            )
        taxel_group = handle.get("observations/taxel")
        available_tactile = set(report["tactile_roles"])
        missing_tactile = sorted(
            set(map(str, required_tactile_roles)) - available_tactile
        )
        if missing_tactile:
            raise ValueError(
                "Canonical episode is missing required tactile roles: "
                + ", ".join(missing_tactile)
            )
        for role in required_tactile_roles:
            dataset = None if taxel_group is None else taxel_group.get(str(role))
            if dataset is None or len(dataset) != len(observation_ids):
                raise ValueError(
                    f"Tactile role {role!r} must have one sample per observation."
                )
        if check_quest:
            quest_sequences = np.asarray(
                quest_sequence[quest_sequence >= 0], dtype=np.int64
            )
            if len(quest_sequences) > 1 and np.any(np.diff(quest_sequences) < 0):
                raise ValueError("Quest frame sequences must be monotonic.")
            report["quest_sequence_gaps"] = int(
                np.maximum(np.diff(quest_sequences) - 1, 0).sum()
            ) if len(quest_sequences) > 1 else 0

    if verify_checksums:
        if manifest_path is None:
            raise ValueError("manifest_path is required when verify_checksums=True.")
        expected = manifest_checksum_for_output(manifest_path, episode_path)
        actual_output_sha256 = sha256_file(episode_path)
        if expected != actual_output_sha256:
            raise ValueError("Canonical episode output checksum does not match manifest.")
        with Path(manifest_path).open("r", encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)
        for record in manifest.get("chunks", []):
            chunk_path = Path(manifest_path).parent / str(record["path"])
            if not chunk_path.exists():
                # A finalized manifest may live next to the episode while its
                # spool is elsewhere; only verify chunks that are addressable.
                continue
            actual = sha256_file(chunk_path)
            if actual != str(record["sha256"]):
                raise ValueError(f"Canonical chunk checksum mismatch: {chunk_path}.")
        report["output_sha256"] = actual_output_sha256
    return report
