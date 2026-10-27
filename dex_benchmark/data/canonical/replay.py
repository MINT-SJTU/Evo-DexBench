"""In-memory replay of canonical HDF5 episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence
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


def _read_recorded_infos(handle: Any) -> list[dict[str, Any]] | None:
    """Return per-transition ``info`` dictionaries recorded during collection."""

    if "transitions/info_json" not in handle:
        return None
    infos: list[dict[str, Any]] = []
    for raw in handle["transitions/info_json"]:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("transitions/info_json must contain JSON objects.")
        infos.append(value)
    return infos


def recorded_recovery_roles(info: Mapping[str, Any]) -> tuple[str, ...]:
    """Roles whose arm target the teleoperation runner re-anchored after this step.

    The Quest3 runner logs ``teleop_ik_recovery`` (stalled accumulated IK target)
    and ``teleop_joint_limit_recovery`` (near-joint-limit target) after it has
    reset the arm controller target to the current end-effector pose. Replaying
    the recorded actions without repeating these resets changes the trajectory.
    """

    roles: set[str] = set()
    ik_recovery = info.get("teleop_ik_recovery")
    if isinstance(ik_recovery, Mapping):
        roles.update(str(role) for role in ik_recovery.get("roles", ()))
    for recovery in info.get("teleop_joint_limit_recovery", ()) or ():
        if isinstance(recovery, Mapping) and "role" in recovery:
            roles.add(str(recovery["role"]))
    return tuple(sorted(roles))


def resolve_env_callable(env: Any, name: str) -> Callable[..., Any] | None:
    """Resolve a benchmark environment method through the Gym wrapper chain."""

    current_env = env
    while current_env is not None:
        candidate = getattr(current_env, name, None)
        if callable(candidate):
            return candidate
        current_env = getattr(current_env, "env", None)
    wrapper_getter = getattr(env, "get_wrapper_attr", None)
    if callable(wrapper_getter):
        try:
            candidate = wrapper_getter(name)
        except AttributeError:
            return None
        if callable(candidate):
            return candidate
    return None


def reset_arm_targets_to_current_pose(env: Any, roles: Iterable[str]) -> None:
    """Re-anchor each role's accumulated end-effector target at its current pose.

    Mirrors ``Quest3TeleopRunner._reset_ee_target_to_current_pose``.
    """

    from dex_benchmark.sim.tasks.runtime import ManipulatorRole

    roles = tuple(roles)
    if not roles:
        return
    agent_for_role = resolve_env_callable(env, "agent_for_role")
    if not callable(agent_for_role):
        raise RuntimeError("Replay needs env.agent_for_role() to re-apply recorded arm re-anchors.")
    for role in roles:
        arm_controller = agent_for_role(ManipulatorRole(role)).controller.controllers["arm"]
        if not bool(getattr(getattr(arm_controller, "config", None), "use_target", False)):
            continue
        current_pose = getattr(arm_controller, "ee_pose_at_base", None)
        if current_pose is None:
            raise RuntimeError(f"Arm controller for role {role!r} does not expose ee_pose_at_base.")
        arm_controller._target_pose = current_pose


def bind_native_action(env: Any, action: Any) -> Any:
    """Map a role-keyed native action ({"left": ..., "right": ...}) to env action keys.

    Canonical HDF5 files store bimanual native actions per manipulator role, while
    the ManiSkill environment expects its own per-agent keys.
    """

    if not isinstance(action, Mapping):
        return action
    get_spec = resolve_env_callable(env, "get_native_action_spec")
    if not callable(get_spec):
        return action
    manipulators = get_spec().get("manipulators", {})
    if not manipulators or not set(action).issubset(manipulators):
        return action
    return {manipulators[role]["agent_action_key"]: value for role, value in action.items()}


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
    apply_recorded_recovery: bool = True,
    counterfactual_resets: Mapping[int, Sequence[str]] | None = None,
    stop_on_done: bool = True,
    pass_layout_to_reset: bool = True,
    keep_outputs: bool = True,
    on_step: Callable[[Any, int, Mapping[str, Any] | None, Mapping[str, Any]], None] | None = None,
) -> ReplayResult:
    """Replay native actions in ``env`` and return all outputs in memory.

    The HDF5 file is opened read-only and is never used as a replay output.
    ``seed``, ``layout_id`` and ``options`` override values stored in metadata.

    ``apply_recorded_recovery`` repeats the arm re-anchors the teleoperation
    runner logged after each step (see ``recorded_recovery_roles``).
    ``counterfactual_resets`` maps a 1-based step to roles re-anchored *before*
    that step; such resets are not recorded events and must be reported as
    inferred by callers. ``stop_on_done=False`` replays every recorded action
    even after the environment reports termination (recordings may continue).
    ``pass_layout_to_reset=False`` resets with the seed only, for environments
    already created with the episode layout. ``on_step(env, step, recorded_info,
    info)`` runs after each step and its recorded re-anchors.
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
        recorded_infos = _read_recorded_infos(handle) if (apply_recorded_recovery or on_step is not None) else None
        if recorded_infos is not None and len(recorded_infos) != len(actions):
            raise ValueError("transitions/info_json count does not match the native actions.")
        if (
            "actions/action_id" in handle
            and len(handle["actions/action_id"]) != len(actions)
        ):
            raise ValueError("Native action count does not match actions/action_id.")

    initial_observation, reset_info = _reset_environment(
        env,
        seed=replay_seed,
        layout_id=replay_layout_id if pass_layout_to_reset else None,
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
    resets_before_step = {int(step): tuple(roles) for step, roles in (counterfactual_resets or {}).items()}
    for step_index, action in enumerate(actions, start=1):
        if step_index in resets_before_step:
            reset_arm_targets_to_current_pose(env, resets_before_step[step_index])
        observation, reward, done, was_truncated, info = _step_result(env.step(bind_native_action(env, action)))
        recorded_info = None if recorded_infos is None else recorded_infos[step_index - 1]
        if apply_recorded_recovery and recorded_info is not None:
            reset_arm_targets_to_current_pose(env, recorded_recovery_roles(recorded_info))
        if on_step is not None:
            on_step(env, step_index, recorded_info, info)
        if on_frame is not None:
            on_frame(env, step_index)
        terminated.append(done)
        truncated.append(was_truncated)
        if keep_outputs:
            replay_actions.append(to_cpu_copy(action))
            observations.append(to_cpu_copy(observation))
            rewards.append(to_cpu_copy(reward))
            infos.append(to_cpu_copy(info))
        if stop_on_done and (done or was_truncated):
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
