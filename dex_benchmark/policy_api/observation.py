from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def required_value(mapping: Mapping[str, Any], *path: str) -> Any:
    """Read a required value from a nested policy observation."""

    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError("Missing policy observation field " + ".".join(path))
        value = value[key]
    return value


def as_finite_numpy(value: Any, *, name: str) -> np.ndarray:
    """Convert a tensor-like value to NumPy and reject non-finite values."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value)
    if array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def rgb_chw_float(value: Any, *, name: str) -> np.ndarray:
    """Convert one uint8 HWC runtime image to contiguous float32 CHW."""

    rgb = as_finite_numpy(value, name=name)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError(
            f"{name} must be uint8 [H,W,3], got {rgb.shape} {rgb.dtype}."
        )
    # [H, W, 3] -> [3, H, W]
    return np.ascontiguousarray(np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0)


def compact_runtime_state(
    manipulators: Mapping[str, Any], roles: Sequence[str]
) -> np.ndarray:
    """Build EE-pose + hand-qpos state from semantic runtime observations."""

    state_parts: list[np.ndarray] = []
    for role in roles:
        manipulator = required_value(manipulators, role)
        ee_pose = as_finite_numpy(
            required_value(manipulator, "end_effector_pose"),
            name=f"{role}.end_effector_pose",
        ).astype(np.float32, copy=False)
        qpos = as_finite_numpy(
            required_value(manipulator, "joint_position"),
            name=f"{role}.joint_position",
        ).astype(np.float32, copy=False)
        if ee_pose.shape != (7,):
            raise ValueError(f"{role}.end_effector_pose must be 7D.")
        if qpos.shape != (31,):
            raise ValueError(
                f"{role}.joint_position must contain 7 arm and 24 hand joints."
            )
        state_parts.extend((ee_pose, qpos[7:]))
    return np.ascontiguousarray(np.concatenate(state_parts), dtype=np.float32)


def runtime_tactile_force(
    manipulators: Mapping[str, Any], roles: Sequence[str], *, taxels_per_role: int = 123
) -> np.ndarray:
    """Concatenate role-ordered distributed normal-force taxels."""

    tactile_parts: list[np.ndarray] = []
    for role in roles:
        manipulator = required_value(manipulators, role)
        force = as_finite_numpy(
            required_value(manipulator, "taxel_normal_force"),
            name=f"{role}.taxel_normal_force",
        ).astype(np.float32, copy=False)
        if force.shape != (taxels_per_role,):
            raise ValueError(
                f"{role}.taxel_normal_force must have shape "
                f"({taxels_per_role},), got {force.shape}."
            )
        if np.any(force < 0):
            raise ValueError(f"{role}.taxel_normal_force must be non-negative.")
        tactile_parts.append(force)
    return np.ascontiguousarray(np.concatenate(tactile_parts), dtype=np.float32)
