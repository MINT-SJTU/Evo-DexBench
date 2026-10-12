"""Benchmark semantic-action backend for Quest 3 teleoperation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from dex_benchmark.teleoperation.quest3.action_builder import Quest3ActionPair


def _resolve_encoder(env: Any) -> Callable[[Any], Any]:
    encoder = getattr(env, "encode_action", None)
    if callable(encoder):
        return encoder
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        encoder = get_wrapper_attr("encode_action")
        if callable(encoder):
            return encoder
    raise AttributeError(
        "BenchmarkSemanticBackend requires an environment encode_action(action) method."
    )


class BenchmarkSemanticBackend:
    """Encode physical semantic actions before executing them in the env."""

    def __init__(self, env: Any, *, default_role: str | None = None) -> None:
        self.env = env
        self.default_role = default_role
        self._encode_action = _resolve_encoder(env)
        self._inactive_role_actions = _inactive_role_actions(
            env,
            active_role=default_role,
        )

    def step(self, action: Any):
        if isinstance(action, Quest3ActionPair):
            action = action.semantic
        semantic_action = action
        # Transitional support for single-arm builders that return the
        # manipulator component mapping without its role wrapper.
        if (
            self.default_role is not None
            and isinstance(action, dict)
            and set(action).issubset(
                {"end_effector_delta_pose", "hand_joint_position"}
            )
        ):
            semantic_action = {self.default_role: action}
        if self._inactive_role_actions:
            if not isinstance(semantic_action, Mapping):
                raise TypeError(
                    "Single-role semantic execution requires a role mapping."
                )
            semantic_action = {
                **self._inactive_role_actions,
                **semantic_action,
            }
        native_action = self._encode_action(semantic_action)
        return self.env.step(native_action)


def _inactive_role_actions(
    env: Any,
    *,
    active_role: str | None,
) -> dict[str, dict[str, np.ndarray]]:
    """Build stable neutral actions for roles omitted by single-hand control."""

    if active_role is None:
        return {}
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if not callable(get_wrapper_attr):
        return {}
    try:
        get_action_spec = get_wrapper_attr("get_action_spec")
    except AttributeError:
        return {}
    if not callable(get_action_spec):
        return {}
    action_spec = get_action_spec()
    manipulator_specs = action_spec.get("manipulators", {})
    if not isinstance(manipulator_specs, Mapping):
        raise TypeError("get_action_spec()['manipulators'] must be a mapping.")
    return {
        str(role): _neutral_manipulator_action(manipulator_spec)
        for role, manipulator_spec in manipulator_specs.items()
        if str(role) != active_role
    }


def _neutral_manipulator_action(
    manipulator_spec: Any,
) -> dict[str, np.ndarray]:
    if not isinstance(manipulator_spec, Mapping):
        raise TypeError("Manipulator action spec must be a mapping.")
    hand_spec = manipulator_spec.get("hand_joint_position", {})
    if not isinstance(hand_spec, Mapping):
        raise TypeError("hand_joint_position action spec must be a mapping.")
    hand_joint_names = tuple(hand_spec.get("joint_names", ()))
    hand_lower = np.asarray(hand_spec.get("lower"), dtype=np.float32)
    hand_upper = np.asarray(hand_spec.get("upper"), dtype=np.float32)
    expected_shape = (len(hand_joint_names),)
    if (
        not hand_joint_names
        or hand_lower.shape != expected_shape
        or hand_upper.shape != expected_shape
    ):
        raise ValueError("Task action spec has invalid canonical hand bounds.")
    return {
        "end_effector_delta_pose": np.zeros(6, dtype=np.float32),
        "hand_joint_position": np.clip(
            np.zeros(expected_shape, dtype=np.float32),
            hand_lower,
            hand_upper,
        ),
    }
