from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from dex_benchmark.policy_api.control_spec import (
    ControllerComponentSpec,
    EEAbsPoseControlSpec,
    EEControlSpec,
    EEDeltaPoseControlSpec,
    JointDeltaPositionControlSpec,
    JointPositionControlSpec,
)
from dex_benchmark.utils.transform import TensorLike, as_backend_array


JointControlSpec = JointPositionControlSpec | JointDeltaPositionControlSpec
ActionBoundsPolicy = Literal["clip", "error"]


def _validate_bounds_policy(bounds_policy: ActionBoundsPolicy) -> None:
    if bounds_policy not in ("clip", "error"):
        raise ValueError(
            f"Unsupported action bounds policy {bounds_policy!r}; "
            "expected 'clip' or 'error'."
        )


def _raise_if_outside_box(
    value: TensorLike,
    low: TensorLike,
    high: TensorLike,
    *,
    value_name: str,
) -> None:
    """Reject non-finite values and values outside an inclusive physical box."""

    if isinstance(value, torch.Tensor):
        low_tensor = as_backend_array(low, like=value, dtype=torch.float32)
        high_tensor = as_backend_array(high, like=value, dtype=torch.float32)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{value_name} must contain only finite values.")
        outside = torch.logical_or(value < low_tensor, value > high_tensor)
        if bool(outside.any().item()):
            raise ValueError(
                f"{value_name} is outside physical bounds "
                f"[{low_tensor}, {high_tensor}]."
            )
        return

    value_array = np.asarray(value, dtype=np.float32)
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    if not np.isfinite(value_array).all():
        raise ValueError(f"{value_name} must contain only finite values.")
    if np.logical_or(value_array < low_array, value_array > high_array).any():
        raise ValueError(
            f"{value_name} is outside physical bounds "
            f"[{low_array}, {high_array}]."
        )


def _zeros_like_action(value: TensorLike, action_dim: int) -> TensorLike:
    """Create an action tensor matching the input backend and batch shape."""

    if isinstance(value, torch.Tensor):
        return torch.zeros(
            (*value.shape[:-1], action_dim),
            dtype=torch.float32,
            device=value.device,
        )
    value_array = np.asarray(value, dtype=np.float32)
    return np.zeros((*value_array.shape[:-1], action_dim), dtype=np.float32)


def inverse_scale_action(
    value: TensorLike,
    low: TensorLike,
    high: TensorLike,
    *,
    bounds_policy: ActionBoundsPolicy = "clip",
    value_name: str = "action",
) -> TensorLike:
    """Map physical values inside [low, high] back to normalized actions."""

    _validate_bounds_policy(bounds_policy)
    if bounds_policy == "error":
        _raise_if_outside_box(value, low, high, value_name=value_name)

    if isinstance(value, torch.Tensor):
        low_tensor = as_backend_array(low, like=value, dtype=torch.float32)
        high_tensor = as_backend_array(high, like=value, dtype=torch.float32)
        midpoint = 0.5 * (high_tensor + low_tensor)
        half_range = 0.5 * (high_tensor - low_tensor)
        if bool(torch.any(torch.abs(half_range) < 1e-8).item()):
            raise ValueError(f"Invalid action range: low={low_tensor}, high={high_tensor}")
        normalized = (value - midpoint) / half_range
        if bounds_policy == "clip":
            normalized = torch.clamp(normalized, -1.0, 1.0)
        return normalized

    value_array = np.asarray(value, dtype=np.float32)
    low_array = np.asarray(low, dtype=np.float32)
    high_array = np.asarray(high, dtype=np.float32)
    midpoint = 0.5 * (high_array + low_array)
    half_range = 0.5 * (high_array - low_array)
    if np.any(np.abs(half_range) < 1e-8):
        raise ValueError(f"Invalid action range: low={low_array}, high={high_array}")
    normalized = (value_array - midpoint) / half_range
    if bounds_policy == "clip":
        normalized = np.clip(normalized, -1.0, 1.0)
    return normalized


def _encode_ee_pose_vector(
    component_spec: EEControlSpec,
    *,
    pos_root_m: TensorLike,
    rot_root_rad: TensorLike | None = None,
    bounds_policy: ActionBoundsPolicy = "clip",
) -> TensorLike:
    """Encode a root-frame EE pose vector with ManiSkill scaling rules."""

    _validate_bounds_policy(bounds_policy)
    if isinstance(pos_root_m, torch.Tensor):
        pos_root_m = pos_root_m.to(dtype=torch.float32)
        rot_root_rad = (
            torch.zeros_like(pos_root_m)
            if rot_root_rad is None
            else as_backend_array(rot_root_rad, like=pos_root_m, dtype=torch.float32)
        )
    else:
        pos_root_m = np.asarray(pos_root_m, dtype=np.float32)
        rot_root_rad = (
            np.zeros_like(pos_root_m)
            if rot_root_rad is None
            else np.asarray(rot_root_rad, dtype=np.float32)
        )

    action = _zeros_like_action(pos_root_m, 6)
    if not component_spec.normalize_action:
        if bounds_policy == "error":
            if component_spec.pos_lower is None or component_spec.pos_upper is None:
                raise ValueError(
                    f"{component_spec.name} has no EE position action bounds."
                )
            if component_spec.rot_lower is None or component_spec.rot_upper is None:
                raise ValueError(
                    f"{component_spec.name} has no EE rotation action bounds."
                )
            _raise_if_outside_box(
                pos_root_m,
                component_spec.pos_lower,
                component_spec.pos_upper,
                value_name=f"{component_spec.name} translation",
            )
            _raise_if_outside_box(
                rot_root_rad,
                component_spec.rot_lower,
                component_spec.rot_upper,
                value_name=f"{component_spec.name} rotation",
            )
        action[..., :3] = pos_root_m
        action[..., 3:6] = rot_root_rad
        return action

    if component_spec.pos_lower is None or component_spec.pos_upper is None:
        raise ValueError(f"{component_spec.name} has no EE position action bounds.")
    action[..., :3] = inverse_scale_action(
        pos_root_m,
        component_spec.pos_lower,
        component_spec.pos_upper,
        bounds_policy=bounds_policy,
        value_name=f"{component_spec.name} translation",
    )

    if component_spec.rotation_scale_mode != "maniskill_pdee_pose":
        raise NotImplementedError(
            f"Unsupported rotation scale mode: {component_spec.rotation_scale_mode}"
        )
    if component_spec.rot_lower is None:
        raise ValueError(f"{component_spec.name} has no EE rotation lower bound.")

    # ManiSkill PDEEPoseController clips rotation by norm, then multiplies rot_lower.
    if isinstance(rot_root_rad, torch.Tensor):
        rot_lower = as_backend_array(
            component_spec.rot_lower,
            like=rot_root_rad,
            dtype=torch.float32,
        )
        if bool(torch.any(torch.abs(rot_lower) < 1e-8).item()):
            raise ValueError(f"Invalid rotation lower bound: {rot_lower}")
        rot_action = rot_root_rad / rot_lower
        rot_norm = torch.linalg.norm(rot_action, dim=-1, keepdim=True)
        if bounds_policy == "error" and bool((rot_norm > 1.0).any().item()):
            raise ValueError(
                f"{component_spec.name} rotation exceeds the ManiSkill "
                "PDEEPoseController physical rotation-norm bound."
            )
        rot_action = torch.where(
            rot_norm > 1.0,
            rot_action / rot_norm.clamp_min(1e-8),
            rot_action,
        )
        action[..., 3:6] = (
            torch.clamp(rot_action, -1.0, 1.0)
            if bounds_policy == "clip"
            else rot_action
        )
        return action

    rot_lower = np.asarray(component_spec.rot_lower, dtype=np.float32)
    if np.any(np.abs(rot_lower) < 1e-8):
        raise ValueError(f"Invalid rotation lower bound: {rot_lower}")
    rot_action = rot_root_rad / rot_lower
    rot_norm = np.linalg.norm(rot_action, axis=-1, keepdims=True)
    if bounds_policy == "error" and np.any(rot_norm > 1.0):
        raise ValueError(
            f"{component_spec.name} rotation exceeds the ManiSkill "
            "PDEEPoseController physical rotation-norm bound."
        )
    rot_action = np.where(rot_norm > 1.0, rot_action / np.maximum(rot_norm, 1e-8), rot_action)
    action[..., 3:6] = (
        np.clip(rot_action, -1.0, 1.0)
        if bounds_policy == "clip"
        else rot_action
    )
    return action


def encode_ee_delta_pose_action(
    component_spec: EEDeltaPoseControlSpec,
    *,
    pos_delta_root_m: TensorLike,
    rot_delta_root_rad: TensorLike | None = None,
    bounds_policy: ActionBoundsPolicy = "clip",
) -> TensorLike:
    """Encode root-frame EE delta pose into a ManiSkill action."""

    return _encode_ee_pose_vector(
        component_spec,
        pos_root_m=pos_delta_root_m,
        rot_root_rad=rot_delta_root_rad,
        bounds_policy=bounds_policy,
    )


def encode_ee_abs_pose_action(
    component_spec: EEAbsPoseControlSpec,
    *,
    target_pos_root_m: TensorLike,
    target_rot_root_rad: TensorLike | None = None,
    bounds_policy: ActionBoundsPolicy = "clip",
) -> TensorLike:
    """Encode root-frame absolute EE pose into a ManiSkill action."""

    return _encode_ee_pose_vector(
        component_spec,
        pos_root_m=target_pos_root_m,
        rot_root_rad=target_rot_root_rad,
        bounds_policy=bounds_policy,
    )


def encode_joint_position_action(
    component_spec: JointPositionControlSpec,
    *,
    target_qpos: TensorLike,
    bounds_policy: ActionBoundsPolicy = "clip",
) -> TensorLike:
    """Encode joint position targets into a ManiSkill action."""

    return encode_joint_action(
        component_spec,
        joint_action=target_qpos,
        bounds_policy=bounds_policy,
    )


def encode_joint_delta_position_action(
    component_spec: JointDeltaPositionControlSpec,
    *,
    delta_qpos: TensorLike,
    bounds_policy: ActionBoundsPolicy = "clip",
) -> TensorLike:
    """Encode joint delta-position commands into a ManiSkill action."""

    return encode_joint_action(
        component_spec,
        joint_action=delta_qpos,
        bounds_policy=bounds_policy,
    )


def encode_joint_action(
    component_spec: JointControlSpec,
    *,
    joint_action: TensorLike,
    bounds_policy: ActionBoundsPolicy = "clip",
) -> TensorLike:
    """Encode joint commands using the component's normalization policy."""

    if isinstance(joint_action, torch.Tensor):
        joint_action = joint_action.to(dtype=torch.float32)
    else:
        joint_action = np.asarray(joint_action, dtype=np.float32)

    if joint_action.shape[-1] != component_spec.action_dim:
        raise ValueError(
            f"{component_spec.name} expects {component_spec.action_dim} joint values, "
            f"got shape {joint_action.shape}."
        )
    _validate_bounds_policy(bounds_policy)
    if bounds_policy == "error":
        if component_spec.action_lower is None or component_spec.action_upper is None:
            raise ValueError(f"{component_spec.name} has no joint action bounds.")
        _raise_if_outside_box(
            joint_action,
            component_spec.action_lower,
            component_spec.action_upper,
            value_name=f"{component_spec.name} joint position",
        )
    if not component_spec.normalize_action:
        return joint_action
    if component_spec.action_lower is None or component_spec.action_upper is None:
        raise ValueError(f"{component_spec.name} has no joint action bounds.")
    return inverse_scale_action(
        joint_action,
        component_spec.action_lower,
        component_spec.action_upper,
        bounds_policy=bounds_policy,
        value_name=f"{component_spec.name} joint position",
    )


def make_flat_action(
    *,
    action_shape: tuple[int, ...],
    component_actions: Sequence[tuple[ControllerComponentSpec, TensorLike]],
) -> np.ndarray:
    """Pack encoded component actions into one flat ManiSkill action."""

    action = np.zeros(action_shape, dtype=np.float32)
    for component_spec, component_action in component_actions:
        if isinstance(component_action, torch.Tensor):
            component_action = component_action.detach().cpu().numpy()
        component_array = np.asarray(component_action, dtype=np.float32)
        if action.ndim == 2:
            action[:, component_spec.action_slice] = component_array
        else:
            action[component_spec.action_slice] = component_array
    return action


# Backward-compatible aliases for older scripts.
inverse_scale_box_action = inverse_scale_action
encode_delta_pose_action = encode_ee_delta_pose_action
