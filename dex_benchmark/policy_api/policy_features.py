from __future__ import annotations

from typing import Any, Literal

import numpy as np


EmbodimentMode = Literal["single", "dual"]
CANONICAL_STATE_DIM_PER_ROLE = 111
EE_POSE_SLICE = slice(0, 7)
HAND_QPOS_SLICE = slice(49, 73)
COMPACT_STATE_DIM_PER_ROLE = 31
SEMANTIC_ACTION_DIM_PER_ROLE = 30
POLICY_CONTROL_FREQUENCY_HZ = 30


def validate_policy_frequency(fps: int, *, policy_name: str) -> None:
    """Reject datasets that violate the benchmark policy timing contract."""

    if fps != POLICY_CONTROL_FREQUENCY_HZ:
        raise ValueError(
            f"{policy_name} training expects Evo-DexBench's "
            f"{POLICY_CONTROL_FREQUENCY_HZ} Hz contract, got {fps} Hz."
        )


def roles_for_embodiment(embodiment: EmbodimentMode | str) -> tuple[str, ...]:
    if embodiment == "single":
        return ("right",)
    if embodiment == "dual":
        return ("left", "right")
    raise ValueError("embodiment must be 'single' or 'dual'.")


def camera_roles_for_embodiment(
    embodiment: EmbodimentMode | str,
) -> tuple[str, ...]:
    if embodiment == "single":
        return ("context", "interaction", "wrist_right")
    if embodiment == "dual":
        return ("context", "wrist_left", "wrist_right")
    raise ValueError("embodiment must be 'single' or 'dual'.")


def select_compact_policy_state(value: Any, embodiment: str) -> np.ndarray:
    """Select EE pose and hand qpos from canonical 111D role blocks."""

    state = np.asarray(value, dtype=np.float32)
    roles = roles_for_embodiment(embodiment)
    expected = CANONICAL_STATE_DIM_PER_ROLE * len(roles)
    if state.shape[-1] != expected:
        raise ValueError(
            f"Canonical {embodiment} state must end in {expected}, got {state.shape}."
        )
    parts: list[np.ndarray] = []
    for role_index in range(len(roles)):
        offset = role_index * CANONICAL_STATE_DIM_PER_ROLE
        parts.extend(
            (
                state[..., offset + EE_POSE_SLICE.start : offset + EE_POSE_SLICE.stop],
                state[
                    ...,
                    offset + HAND_QPOS_SLICE.start : offset + HAND_QPOS_SLICE.stop,
                ],
            )
        )
    return np.ascontiguousarray(np.concatenate(parts, axis=-1), dtype=np.float32)


def decode_semantic_action(
    value: Any, embodiment: str
) -> dict[str, dict[str, np.ndarray]]:
    """Decode compact physical actions with fixed single/dual role ordering."""

    roles = roles_for_embodiment(embodiment)
    action = np.asarray(value, dtype=np.float32)
    expected = SEMANTIC_ACTION_DIM_PER_ROLE * len(roles)
    if action.shape != (expected,):
        raise ValueError(f"{embodiment} action must have shape ({expected},), got {action.shape}.")
    if not np.isfinite(action).all():
        raise ValueError("action must contain only finite values.")
    result: dict[str, dict[str, np.ndarray]] = {}
    for role_index, role in enumerate(roles):
        offset = role_index * SEMANTIC_ACTION_DIM_PER_ROLE
        result[role] = {
            "end_effector_delta_pose": action[offset : offset + 6].copy(),
            "hand_joint_position": action[offset + 6 : offset + 30].copy(),
        }
    return result
