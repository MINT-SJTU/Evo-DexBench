"""In-memory replay of canonical HDF5 episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, Mapping

import numpy as np

from dex_benchmark.data.canonical.hdf5 import _require_h5py
from dex_benchmark.data.canonical.schema import to_cpu_copy


@dataclass(frozen=True)
class ReplayResult:
    """Values produced by replay; no value is written back to disk.

    ``observations`` includes the reset observation, so it normally has one
    more item than ``actions``.
    """

    seed: int | None
    layout_id: str | None
    observations: tuple[Any, ...]
    actions: tuple[Any, ...]
    rewards: tuple[Any, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    infos: tuple[Any, ...]
    reset_info: Mapping[str, Any]


def _read_meta(handle: Any, name: str) -> Any:
    dataset_name = f"meta/{name}.json"
    if dataset_name not in handle:
        return None
    value = handle[dataset_name][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        return json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {dataset_name!r}.") from exc


def _insert_field(root: dict[str, Any], name: str, value: np.ndarray) -> None:
    parts = name.split(".")
    current = root
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Native action field path {name!r} is ambiguous.")
        current = child
    current[parts[-1]] = value


def _native_actions(handle: Any) -> list[Any]:
    native_group = handle.get("actions/native")
    if native_group is None:
        raise ValueError("Canonical episode does not contain actions/native.")

    def visit(group: Any, prefix: str = "") -> list[tuple[str, Any]]:
        result: list[tuple[str, Any]] = []
        for name, item in group.items():
            field_name = f"{prefix}.{name}" if prefix else str(name)
            if hasattr(item, "items"):
                result.extend(visit(item, field_name))
            else:
                result.append((field_name, item))
        return result

    datasets = sorted(visit(native_group), key=lambda item: item[0])
    if not datasets:
        raise ValueError("Canonical episode contains no native action fields.")
    lengths = {int(dataset.shape[0]) for _, dataset in datasets}
    if len(lengths) != 1:
        raise ValueError("Native action fields have different lengths.")
    action_count = lengths.pop()
    actions: list[Any] = []
    for index in range(action_count):
        fields: dict[str, Any] = {}
        for name, dataset in datasets:
            _insert_field(fields, name, np.array(dataset[index], copy=True))
        if len(fields) == 1 and "value" in fields:
            actions.append(fields["value"])
        else:
            actions.append(fields)
    return actions


def _reset_environment(
    env: Any,
    *,
    seed: int | None,
    layout_id: str | None,
    options: Mapping[str, Any] | None,
) -> tuple[Any, Mapping[str, Any]]:
    reset_options = None if options is None else dict(options)
    if layout_id is not None:
        if reset_options is None:
            reset_options = {"layout_id": layout_id}
        else:
            reset_options.setdefault("layout_id", layout_id)
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = seed
    if reset_options is not None:
        kwargs["options"] = reset_options
    result = env.reset(**kwargs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], to_cpu_copy(dict(result[1]))
    return result, {}


def _step_result(result: Any) -> tuple[Any, Any, bool, bool, Any]:
    if not isinstance(result, tuple):
        raise TypeError("Environment step must return a tuple.")
    if len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, reward, bool(terminated), bool(truncated), info
    if len(result) == 4:
        observation, reward, done, info = result
        return observation, reward, bool(done), False, info
    raise ValueError(f"Environment step returned {len(result)} values; expected 4 or 5.")


def replay_canonical_episode(
    env: Any,
    episode_path: str | Path,
    *,
    seed: int | None = None,
    layout_id: str | None = None,
    options: Mapping[str, Any] | None = None,
    on_frame: Callable[[Any, int], None] | None = None,
) -> ReplayResult:
    """Replay native actions in ``env`` and return all outputs in memory.

    The HDF5 file is opened read-only and is never used as a replay output.
    ``seed``, ``layout_id`` and ``options`` override values stored in metadata.
    """

    h5py = _require_h5py()
    path = Path(episode_path).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        stored_seed = _read_meta(handle, "seed")
        if stored_seed is None and "seed" in handle.attrs:
            stored_seed = handle.attrs["seed"]
        stored_layout = _read_meta(handle, "layout_id")
        if stored_layout is None and "layout_id" in handle.attrs:
            stored_layout = handle.attrs["layout_id"]
        replay_seed = seed if seed is not None else (
            None if stored_seed is None else int(stored_seed)
        )
        stored_layout_id = None if stored_layout in (None, "") else str(stored_layout)
        replay_layout_id = layout_id if layout_id is not None else stored_layout_id
        actions = _native_actions(handle)
        if (
            "actions/action_id" in handle
            and len(handle["actions/action_id"]) != len(actions)
        ):
            raise ValueError("Native action count does not match actions/action_id.")

    initial_observation, reset_info = _reset_environment(
        env,
        seed=replay_seed,
        layout_id=replay_layout_id,
        options=options,
    )
    if on_frame is not None:
        on_frame(env, 0)
    observations: list[Any] = [to_cpu_copy(initial_observation)]
    replay_actions: list[Any] = []
    rewards: list[Any] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    infos: list[Any] = []
    for step_index, action in enumerate(actions, start=1):
        observation, reward, done, was_truncated, info = _step_result(env.step(action))
        if on_frame is not None:
            on_frame(env, step_index)
        replay_actions.append(to_cpu_copy(action))
        observations.append(to_cpu_copy(observation))
        rewards.append(to_cpu_copy(reward))
        terminated.append(done)
        truncated.append(was_truncated)
        infos.append(to_cpu_copy(info))
        if done or was_truncated:
            break
    return ReplayResult(
        seed=replay_seed,
        layout_id=replay_layout_id,
        observations=tuple(observations),
        actions=tuple(replay_actions),
        rewards=tuple(rewards),
        terminated=tuple(terminated),
        truncated=tuple(truncated),
        infos=tuple(infos),
        reset_info=dict(reset_info),
    )
