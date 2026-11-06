from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.observation import as_finite_numpy, required_value
from dex_benchmark.policy_api.policy_features import (
    CANONICAL_STATE_DIM_PER_ROLE,
    EE_POSE_SLICE,
    SEMANTIC_ACTION_DIM_PER_ROLE,
)
from dex_benchmark.policy_api.session import SemanticAction
from pi05_policy.config import Pi05PolicyConfig


CONTROL_PROMPT = (
    "<control_mode> end effector delta in robot root frame; xyz euler "
    "rotation delta; absolute hand joint positions </control_mode>"
)
# Canonical per-arm state layout: EE pose occupies [0:7] and the 31 joint
# positions (7 arm + 24 hand) occupy [42:73].
JOINT_POSITION_SLICE = slice(42, 73)
# OpenPI image slots, filled from the embodiment's camera roles in order.
IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _vector(value: Any, length: int, *, name: str) -> np.ndarray:
    array = as_finite_numpy(value, name=name).astype(np.float32, copy=False)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}.")
    return array


def _rgb(value: Any, *, name: str) -> np.ndarray:
    array = as_finite_numpy(value, name=name)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {array.shape}.")
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8 RGB pixels, got {array.dtype}.")
    return np.ascontiguousarray(array)


class Pi05Adapter:
    """Convert benchmark observations and π0.5 actions at the wire boundary."""

    def __init__(self, config: Pi05PolicyConfig) -> None:
        self.config = config

    def request(self, model_input: ModelInput) -> dict[str, Any]:
        observation = model_input["observation"]
        if not isinstance(observation, Mapping):
            raise TypeError("model_input observation must be a mapping.")
        cameras = required_value(observation, "cameras")
        manipulators = required_value(observation, "manipulators")

        state_parts: list[np.ndarray] = []
        for role in self.config.roles:
            manipulator = required_value(manipulators, role)
            ee_pose = _vector(
                required_value(manipulator, "end_effector_pose"),
                7,
                name=f"{role}.end_effector_pose",
            )
            joint_position = _vector(
                required_value(manipulator, "joint_position"),
                31,
                name=f"{role}.joint_position",
            )
            # Send the policy-independent dataset layout. The server-side
            # EvoDexInputs transform selects EE pose and hand qpos, exactly as
            # during training, before normalization and 32D/64D padding.
            canonical = np.zeros(CANONICAL_STATE_DIM_PER_ROLE, dtype=np.float32)
            canonical[EE_POSE_SLICE] = ee_pose
            canonical[JOINT_POSITION_SLICE] = joint_position
            state_parts.append(canonical)

        images = {
            slot: _rgb(required_value(cameras, role, "rgb"), name=f"{role}.rgb")
            for slot, role in zip(IMAGE_SLOTS, self.config.camera_roles, strict=True)
        }
        instruction = str(model_input["instruction"]).strip()
        if not instruction:
            raise ValueError("Policy instruction must be non-empty.")
        return {
            "images": images,
            "state": np.concatenate(state_parts),
            "prompt": f"{CONTROL_PROMPT} {instruction}",
        }

    def action_chunk(self, response: Mapping[str, Any]) -> np.ndarray:
        if "actions" not in response:
            raise KeyError("OpenPI response is missing 'actions'.")
        actions = as_finite_numpy(response["actions"], name="actions").astype(
            np.float32, copy=False
        )
        expected = (self.config.action_horizon, self.config.model_action_dim)
        if actions.shape != expected:
            raise ValueError(f"OpenPI actions must have shape {expected}, got {actions.shape}.")
        return np.ascontiguousarray(actions)

    def semantic_action(self, action: np.ndarray) -> SemanticAction:
        vector = _vector(action, self.config.model_action_dim, name="action")
        result: dict[str, dict[str, np.ndarray]] = {}
        offset = 0
        for role in self.config.roles:
            # Physical actions are packed role by role: EE delta [6] and
            # absolute hand qpos [24]. The trailing model padding is not a
            # control and is discarded.
            result[role] = {
                "end_effector_delta_pose": vector[offset : offset + 6].copy(),
                "hand_joint_position": vector[offset + 6 : offset + 30].copy(),
            }
            offset += SEMANTIC_ACTION_DIM_PER_ROLE
        return result
