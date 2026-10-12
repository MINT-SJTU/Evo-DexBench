"""HDF5 chunk serialization and canonical episode assembly."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.data.canonical.schema import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalAction,
    CanonicalObservation,
    CanonicalTransition,
    QuestFrameSnapshot,
    jsonable,
    strict_json,
    validate_numeric,
)


def _require_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs.
        raise ImportError(
            "HDF5 recording requires h5py. Install the dataset extra with "
            "`pip install -e '.[data]'`."
        ) from exc
    return h5py


def _safe_key(value: str) -> str:
    key = str(value).strip()
    if not key or key in {".", ".."}:
        raise ValueError(f"HDF5 key must be non-empty, got {value!r}.")
    return key.replace("/", "__")


def _stack(values: list[Any], *, field_name: str) -> np.ndarray:
    if not values:
        raise ValueError(f"Cannot write empty HDF5 dataset {field_name!r}.")
    arrays = [np.asarray(value) for value in values]
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays[1:]):
        shapes = [array.shape for array in arrays]
        raise ValueError(f"{field_name} shapes are inconsistent: {shapes}.")
    try:
        result = np.stack(arrays, axis=0)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Unable to stack {field_name} values.") from exc
    if result.dtype.kind not in "biufc":
        raise TypeError(f"{field_name} must be numeric, got {result.dtype}.")
    return np.ascontiguousarray(result)


def _write_dataset(
    group: Any,
    name: str,
    values: Any,
    *,
    compression: str | None = None,
) -> Any:
    data = np.asarray(values)
    if data.ndim == 0:
        return group.create_dataset(name, data=data)
    kwargs: dict[str, Any] = {
        "shape": data.shape,
        "maxshape": (None,) + data.shape[1:],
        "dtype": data.dtype,
    }
    if compression is not None and data.size:
        kwargs.update(compression=compression, compression_opts=4, shuffle=True)
    dataset = group.create_dataset(name, **kwargs)
    dataset[...] = data
    return dataset


def _write_json_dataset(group: Any, name: str, values: list[Any]) -> Any:
    h5py = _require_h5py()
    dtype = h5py.string_dtype(encoding="utf-8")
    encoded = np.asarray([strict_json(value) for value in values], dtype=dtype)
    return group.create_dataset(
        name,
        data=encoded,
        shape=(len(encoded),),
        maxshape=(None,),
        dtype=dtype,
    )


def _feature_maps(
    observations: list[CanonicalObservation],
    field_name: str,
) -> dict[str, list[np.ndarray]]:
    keys: set[str] = set()
    for observation in observations:
        keys.update(getattr(observation, field_name))
    result: dict[str, list[np.ndarray]] = {}
    for key in sorted(keys):
        values: list[np.ndarray] = []
        for observation in observations:
            collection = getattr(observation, field_name)
            if key not in collection:
                raise ValueError(
                    f"{field_name}.{key} is missing from observation "
                    f"{observation.observation_id}."
                )
            values.append(collection[key])
        result[key] = values
    return result


def _write_observations(group: Any, observations: list[CanonicalObservation]) -> None:
    if not observations:
        return
    _write_dataset(
        group,
        "observation_id",
        np.asarray([item.observation_id for item in observations], dtype=np.int64),
    )
    _write_dataset(
        group,
        "time_s",
        np.asarray([item.timestamp_s for item in observations], dtype=np.float64),
    )
    state_group = group.require_group("state")
    for key, values in _feature_maps(observations, "state").items():
        _write_dataset(state_group, _safe_key(key), _stack(values, field_name=f"state.{key}"))
    taxel_group = group.require_group("taxel")
    for key, values in _feature_maps(observations, "taxel").items():
        _write_dataset(
            taxel_group,
            _safe_key(key),
            _stack(values, field_name=f"taxel.{key}"),
        )

    image_records: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    depth_records: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    for observation in observations:
        for camera, image in observation.images.items():
            image_records.setdefault(camera, []).append(
                (
                    observation.observation_id,
                    float(observation.image_timestamp_s or observation.timestamp_s),
                    image,
                )
            )
        for camera, depth in observation.depth.items():
            depth_records.setdefault(camera, []).append(
                (
                    observation.observation_id,
                    float(observation.image_timestamp_s or observation.timestamp_s),
                    depth,
                )
            )

    images_group = group.require_group("images")
    depth_group = group.require_group("depth")
    image_ids_group = group.require_group("image_observation_id")
    image_times_group = group.require_group("image_timestamp_s")
    for camera in sorted(image_records):
        records = image_records[camera]
        _write_dataset(
            images_group,
            _safe_key(camera),
            _stack([record[2] for record in records], field_name=f"images.{camera}"),
            compression="gzip",
        )
        _write_dataset(
            depth_group,
            _safe_key(camera),
            _stack([record[2] for record in depth_records[camera]], field_name=f"depth.{camera}"),
            compression="gzip",
        )
        _write_dataset(
            image_ids_group,
            _safe_key(camera),
            np.asarray([record[0] for record in records], dtype=np.int64),
        )
        _write_dataset(
            image_times_group,
            _safe_key(camera),
            np.asarray([record[1] for record in records], dtype=np.float64),
        )


def _flatten_native(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        flattened: dict[str, np.ndarray] = {}
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_native(item, child_prefix))
        return flattened
    validate_numeric(value, field_name=f"native.{prefix or 'value'}")
    return {prefix or "value": np.asarray(value)}


def _write_actions(group: Any, actions: list[CanonicalAction]) -> None:
    if not actions:
        return
    _write_dataset(
        group,
        "action_id",
        np.asarray([item.action_id for item in actions], dtype=np.int64),
    )
    _write_dataset(
        group,
        "time_s",
        np.asarray([item.timestamp_s for item in actions], dtype=np.float64),
    )
    _write_dataset(
        group,
        "action_schema_version",
        np.asarray([item.action_schema_version for item in actions], dtype=np.int32),
    )

    semantic_group = group.require_group("semantic")
    role_fields: dict[tuple[str, str], list[np.ndarray]] = {}
    for action in actions:
        for role, fields in action.semantic.items():
            for field_name, value in fields.items():
                role_fields.setdefault((str(role), str(field_name)), []).append(
                    np.asarray(value)
                )
    expected_fields = set(role_fields)
    for role, fields in ((str(role), set(action.semantic.get(role, {}))) for action in actions):
        actual = {(role, field_name) for field_name in fields}
        if actual != {key for key in expected_fields if key[0] == role}:
            raise ValueError(f"Semantic action fields changed for role {role!r}.")
    for (role, field_name), values in sorted(role_fields.items()):
        role_group = semantic_group.require_group(_safe_key(role))
        _write_dataset(
            role_group,
            _safe_key(field_name),
            _stack(values, field_name=f"semantic.{role}.{field_name}"),
        )

    native_values = [_flatten_native(action.native) for action in actions]
    native_keys = set(native_values[0])
    if any(set(values) != native_keys for values in native_values[1:]):
        raise ValueError("Native action mapping fields changed within a chunk.")
    native_group = group.require_group("native")
    for key in sorted(native_keys):
        _write_dataset(
            native_group,
            _safe_key(key),
            _stack([values[key] for values in native_values], field_name=f"native.{key}"),
        )


def _write_quest(group: Any, frames: list[QuestFrameSnapshot]) -> None:
    if not frames:
        return
    _write_dataset(
        group,
        "sequence",
        np.asarray([frame.sequence for frame in frames], dtype=np.int64),
    )
    source_times = np.asarray(
        [np.nan if frame.source_timestamp_s is None else frame.source_timestamp_s for frame in frames],
        dtype=np.float64,
    )
    _write_dataset(group, "source_timestamp_s", source_times)
    _write_dataset(
        group,
        "received_timestamp_s",
        np.asarray([frame.received_timestamp_s for frame in frames], dtype=np.float64),
    )
    for side in ("left", "right"):
        side_frames = [
            (frame.sequence, getattr(frame, side))
            for frame in frames
            if getattr(frame, side) is not None
        ]
        if not side_frames:
            continue
        side_group = group.require_group(side)
        _write_dataset(
            side_group,
            "sequence",
            np.asarray([sequence for sequence, _ in side_frames], dtype=np.int64),
        )
        hands = [hand for _, hand in side_frames]
        assert all(hand is not None for hand in hands)
        typed_hands = [hand for hand in hands if hand is not None]
        _write_dataset(
            side_group,
            "joint_matrices",
            _stack(
                [hand.joint_matrices for hand in typed_hands],
                field_name=f"quest.{side}.joint_matrices",
            ),
            compression="gzip",
        )
        optional_arrays = (
            ("mano_xyz", "joint_pos_mano"),
            ("retarget_qpos", "retarget_qpos"),
        )
        for output_name, source_name in optional_arrays:
            values = [getattr(hand, source_name) for hand in typed_hands]
            if any(value is None for value in values):
                continue
            _write_dataset(
                side_group,
                output_name,
                _stack(
                    [value for value in values if value is not None],
                    field_name=f"quest.{side}.{output_name}",
                ),
            )
        h5py = _require_h5py()
        side_group.create_dataset(
            "ref_value_json",
            data=np.asarray(
                [strict_json(hand.ref_value) for hand in typed_hands],
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )
        side_group.create_dataset(
            "gesture_json",
            data=np.asarray(
                [strict_json(hand.gesture) for hand in typed_hands],
                dtype=h5py.string_dtype(encoding="utf-8"),
            ),
        )
        side_group.attrs["retarget_joint_names"] = strict_json(
            list(typed_hands[0].retarget_joint_names)
        )


def write_chunk_hdf5(
    path: str | Path,
    transitions: list[CanonicalTransition],
    *,
    chunk_index: int,
    known_observation_ids: set[int] | None = None,
    known_quest_sequences: set[int] | None = None,
) -> tuple[set[int], set[int]]:
    """Write one immutable chunk and return newly materialized references."""

    h5py = _require_h5py()
    if not transitions:
        raise ValueError("Cannot write an empty canonical chunk.")
    known_observation_ids = set() if known_observation_ids is None else set(known_observation_ids)
    known_quest_sequences = set() if known_quest_sequences is None else set(known_quest_sequences)
    new_observation_ids: set[int] = set()
    observations: list[CanonicalObservation] = []
    for transition in transitions:
        for observation in (transition.observation_t, transition.observation_tp1):
            observation_id = observation.observation_id
            if observation_id in known_observation_ids or observation_id in new_observation_ids:
                continue
            new_observation_ids.add(observation_id)
            observations.append(observation)

    new_quest_sequences: set[int] = set()
    quest_frames: list[QuestFrameSnapshot] = []
    for transition in transitions:
        frame = transition.quest_frame
        if frame is None or frame.sequence in known_quest_sequences or frame.sequence in new_quest_sequences:
            continue
        new_quest_sequences.add(frame.sequence)
        quest_frames.append(frame)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(target, "w", libver="latest") as handle:
        handle.attrs["schema_version"] = CANONICAL_SCHEMA_VERSION
        handle.attrs["chunk_index"] = int(chunk_index)
        handle.attrs["first_transition_id"] = int(transitions[0].transition_id)
        handle.attrs["last_transition_id"] = int(transitions[-1].transition_id)
        transition_group = handle.require_group("transitions")
        _write_dataset(
            transition_group,
            "transition_id",
            np.asarray([item.transition_id for item in transitions], dtype=np.int64),
        )
        _write_dataset(
            transition_group,
            "observation_id_t",
            np.asarray([item.observation_t.observation_id for item in transitions], dtype=np.int64),
        )
        _write_dataset(
            transition_group,
            "action_id",
            np.asarray([item.action.action_id for item in transitions], dtype=np.int64),
        )
        _write_dataset(
            transition_group,
            "observation_id_tp1",
            np.asarray([item.observation_tp1.observation_id for item in transitions], dtype=np.int64),
        )
        _write_dataset(
            transition_group,
            "held_action",
            np.asarray([item.held_action for item in transitions], dtype=np.bool_),
        )
        _write_dataset(
            transition_group,
            "quest_sequence",
            np.asarray(
                [-1 if item.quest_frame is None else item.quest_frame.sequence for item in transitions],
                dtype=np.int64,
            ),
        )
        _write_json_dataset(transition_group, "reward_json", [item.reward for item in transitions])
        _write_json_dataset(
            transition_group,
            "terminated_json",
            [item.terminated for item in transitions],
        )
        _write_json_dataset(
            transition_group,
            "truncated_json",
            [item.truncated for item in transitions],
        )
        _write_json_dataset(transition_group, "info_json", [item.info for item in transitions])
        timestamps_group = transition_group.require_group("timestamps")
        timestamp_keys = sorted({key for item in transitions for key in item.timestamps})
        for key in timestamp_keys:
            _write_dataset(
                timestamps_group,
                _safe_key(key),
                np.asarray(
                    [
                        np.nan if item.timestamps.get(key) is None else item.timestamps[key]
                        for item in transitions
                    ],
                    dtype=np.float64,
                ),
            )

        _write_actions(handle.require_group("actions"), [item.action for item in transitions])
        _write_observations(handle.require_group("observations"), observations)
        diagnostics_group = handle.require_group("diagnostics")
        _write_quest(diagnostics_group.require_group("quest"), quest_frames)
        # Keep the namespace explicit even when no viewer frames are enabled;
        # viewer data can never be mistaken for training observations.
        diagnostics_group.require_group("viewer")
    return new_observation_ids, new_quest_sequences


def _create_appendable_dataset_like(
    destination: Any,
    source_dataset: Any,
    name: str,
    values: np.ndarray,
) -> Any:
    if values.ndim == 0:
        return destination.create_dataset(name, data=values)
    kwargs: dict[str, Any] = {
        "shape": values.shape,
        "maxshape": (None,) + values.shape[1:],
        "dtype": source_dataset.dtype,
    }
    if source_dataset.chunks is not None:
        kwargs["chunks"] = source_dataset.chunks
    if source_dataset.compression is not None:
        kwargs["compression"] = source_dataset.compression
        kwargs["compression_opts"] = source_dataset.compression_opts
    if source_dataset.shuffle:
        kwargs["shuffle"] = True
    if source_dataset.fletcher32:
        kwargs["fletcher32"] = True
    if source_dataset.scaleoffset is not None:
        kwargs["scaleoffset"] = source_dataset.scaleoffset
    dataset = destination.create_dataset(name, **kwargs)
    dataset[...] = values
    for key, value in source_dataset.attrs.items():
        dataset.attrs[key] = value
    return dataset


def _append_dataset(destination: Any, source: Any, name: str) -> None:
    source_dataset = source[name]
    values = source_dataset[...]
    if name not in destination:
        _create_appendable_dataset_like(
            destination,
            source_dataset,
            name,
            np.asarray(values),
        )
        return
    target = destination[name]
    if target.ndim == 0 or np.asarray(values).ndim == 0:
        raise ValueError(f"Cannot append scalar HDF5 dataset {name!r}.")
    if target.shape[1:] != np.asarray(values).shape[1:]:
        raise ValueError(
            f"HDF5 dataset {name!r} trailing shape changed from "
            f"{target.shape[1:]} to {np.asarray(values).shape[1:]}."
        )
    old_size = target.shape[0]
    target.resize((old_size + np.asarray(values).shape[0],) + target.shape[1:])
    target[old_size:] = values


def _append_group(
    destination: Any,
    source: Any,
    *,
    copy_attributes: bool = True,
) -> None:
    if copy_attributes:
        for key, value in source.attrs.items():
            if key not in destination.attrs:
                destination.attrs[key] = value
    for name, item in source.items():
        if hasattr(item, "items"):
            child = destination.require_group(name)
            _append_group(child, item)
        else:
            _append_dataset(destination, source, name)


def assemble_episode_hdf5(
    output_path: str | Path,
    chunk_paths: Iterable[str | Path],
    *,
    metadata: Mapping[str, Any],
    checksums: list[str] | None = None,
) -> Path:
    """Concatenate immutable chunks into one canonical episode file."""

    h5py = _require_h5py()
    chunks = [Path(path) for path in chunk_paths]
    if not chunks:
        raise ValueError("Cannot assemble an episode without chunks.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with h5py.File(temporary, "w", libver="latest") as handle:
        handle.attrs["schema_version"] = CANONICAL_SCHEMA_VERSION
        handle.attrs["dataset_kind"] = "hdf5_canonical"
        for key in ("task_id", "embodiment_id", "episode_id", "control_hz", "image_hz"):
            if key in metadata:
                value = metadata[key]
                handle.attrs[key] = value if isinstance(value, (str, int, float, bool)) else strict_json(value)
        meta_group = handle.require_group("meta")
        for key, value in metadata.items():
            if key in {"task_id", "embodiment_id", "episode_id", "control_hz", "image_hz"}:
                continue
            encoded = strict_json(value)
            meta_group.create_dataset(
                _safe_key(key) + ".json",
                data=np.asarray(encoded, dtype=h5py.string_dtype(encoding="utf-8")),
            )
        for chunk_path in chunks:
            with h5py.File(chunk_path, "r") as source:
                _append_group(handle, source, copy_attributes=False)
        transition_ids = np.asarray(
            handle["transitions/transition_id"],
            dtype=np.int64,
        )
        if transition_ids.size == 0:
            raise ValueError("Cannot assemble an episode without transitions.")
        handle.attrs["first_transition_id"] = int(transition_ids[0])
        handle.attrs["last_transition_id"] = int(transition_ids[-1])
        handle.attrs["transition_count"] = int(transition_ids.size)
        handle.attrs["chunk_count"] = int(len(chunks))
        integrity = handle.require_group("integrity")
        if checksums:
            checksum_dtype = h5py.string_dtype(encoding="ascii", length=64)
            integrity.create_dataset(
                "chunk_sha256",
                data=np.asarray(checksums, dtype=checksum_dtype),
            )
            integrity.create_dataset(
                "chunk_index",
                data=np.arange(len(checksums), dtype=np.int64),
            )
    with temporary.open("rb") as file_handle:
        os.fsync(file_handle.fileno())
    temporary.replace(target)
    return target
