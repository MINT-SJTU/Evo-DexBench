from __future__ import annotations

from typing import Protocol, Sequence, TypeAlias

import numpy as np
import torch


TensorLike: TypeAlias = np.ndarray | torch.Tensor
NumericInput: TypeAlias = TensorLike | Sequence[float] | Sequence[Sequence[float]]


class RawPoseLike(Protocol):
    raw_pose: TensorLike


class PQPoseLike(Protocol):
    p: TensorLike
    q: TensorLike


PoseValueLike: TypeAlias = TensorLike | NumericInput | RawPoseLike | PQPoseLike


def _as_torch_dtype(dtype: torch.dtype | np.dtype | None, fallback: torch.dtype) -> torch.dtype:
    """Normalize optional dtype values for torch tensor construction."""

    if dtype is None:
        return fallback
    if isinstance(dtype, torch.dtype):
        return dtype
    return torch.from_numpy(np.empty((), dtype=dtype)).dtype


def is_torch_tensor(value: object) -> bool:
    """Return whether value behaves like a torch tensor."""

    return isinstance(value, torch.Tensor)


def to_numpy(value: NumericInput) -> np.ndarray:
    """Explicitly convert torch-like or array-like values to NumPy."""

    if isinstance(value, torch.Tensor):
        value = value.detach()
        return value.cpu().numpy()
    return np.asarray(value)


def as_backend_array(
    value: NumericInput,
    *,
    like: TensorLike | None = None,
    dtype: torch.dtype | np.dtype | None = None,
) -> TensorLike:
    """Convert value to the backend used by like, preserving torch devices."""

    if isinstance(like, torch.Tensor):
        torch_dtype = _as_torch_dtype(dtype, like.dtype)
        return torch.as_tensor(value, dtype=torch_dtype, device=like.device)
    if isinstance(value, torch.Tensor):
        return value.to(dtype=_as_torch_dtype(dtype, value.dtype)) if dtype is not None else value
    return np.asarray(value, dtype=dtype)


def pose_to_xyzquat(pose: PoseValueLike) -> TensorLike:
    """Convert a pose-like value to an xyz + quaternion vector."""

    if hasattr(pose, "raw_pose"):
        return pose.raw_pose
    if hasattr(pose, "p") and hasattr(pose, "q"):
        if isinstance(pose.p, torch.Tensor) or isinstance(pose.q, torch.Tensor):
            return torch.cat([pose.p, pose.q], dim=-1)
        return np.concatenate([np.asarray(pose.p), np.asarray(pose.q)], axis=-1)
    return as_backend_array(pose)


def flatten_last_dim(value: PoseValueLike, *, name: str, min_dim: int) -> TensorLike:
    """Validate the last dimension while preserving backend and batch dims."""

    array = value
    if hasattr(array, "raw_pose"):
        array = array.raw_pose
    if not isinstance(array, torch.Tensor):
        array = np.asarray(array)
    if len(array.shape) < 1 or array.shape[-1] < min_dim:
        raise ValueError(f"{name} must have last dimension >= {min_dim}, got shape {array.shape}")
    return array


def pose_value_to_pos_quat(
    pose_value: PoseValueLike,
    *,
    name: str,
    allow_sim_pose: bool = False,
) -> tuple[TensorLike, TensorLike]:
    """Return position and [w, x, y, z] quaternion from a pose-like value."""

    if pose_value is None:
        raise ValueError(f"{name} is missing.")

    if hasattr(pose_value, "raw_pose"):
        if not allow_sim_pose:
            raise TypeError(f"{name} is a simulator Pose object, not a model-facing obs value.")
        pose_value = pose_value.raw_pose
    elif hasattr(pose_value, "p") and hasattr(pose_value, "q"):
        if not allow_sim_pose:
            raise TypeError(f"{name} is a simulator Pose object, not a model-facing obs value.")
        position_world = flatten_last_dim(pose_value.p, name=f"{name}.p", min_dim=3)[..., :3]
        quat_world = flatten_last_dim(pose_value.q, name=f"{name}.q", min_dim=4)[..., :4]
        return position_world, quat_world

    pose_vec = flatten_last_dim(pose_value, name=name, min_dim=7)
    return pose_vec[..., :3], pose_vec[..., 3:7]


def quat_to_rot_matrix(quat: NumericInput) -> TensorLike:
    """Convert [w, x, y, z] quaternions to rotation matrices."""

    if isinstance(quat, torch.Tensor):
        quat = quat.to(dtype=torch.float32)
        if quat.shape[-1] != 4:
            raise ValueError(f"Expected quaternion last dimension 4, got shape {quat.shape}")
        w, x, y, z = torch.unbind(quat, dim=-1)
        norm_sq = (quat * quat).sum(dim=-1).clamp_min(1e-12)
        scale = 2.0 / norm_sq
        wx, wy, wz = scale * w * x, scale * w * y, scale * w * z
        xx, xy, xz = scale * x * x, scale * x * y, scale * x * z
        yy, yz, zz = scale * y * y, scale * y * z, scale * z * z
        row0 = torch.stack([1.0 - (yy + zz), xy - wz, xz + wy], dim=-1)
        row1 = torch.stack([xy + wz, 1.0 - (xx + zz), yz - wx], dim=-1)
        row2 = torch.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got shape {quat.shape}")
    w, x, y, z = np.moveaxis(quat, -1, 0)
    norm_sq = np.maximum(np.sum(quat * quat, axis=-1), 1e-12)
    scale = 2.0 / norm_sq
    wx, wy, wz = scale * w * x, scale * w * y, scale * w * z
    xx, xy, xz = scale * x * x, scale * x * y, scale * x * z
    yy, yz, zz = scale * y * y, scale * y * z, scale * z * z
    row0 = np.stack([1.0 - (yy + zz), xy - wz, xz + wy], axis=-1)
    row1 = np.stack([xy + wz, 1.0 - (xx + zz), yz - wx], axis=-1)
    row2 = np.stack([xz - wy, yz + wx, 1.0 - (xx + yy)], axis=-1)
    return np.stack([row0, row1, row2], axis=-2)


def pose_axes_world(quat_world: NumericInput) -> dict[str, list[float]]:
    """Return frame basis axes expressed in world coordinates for logging."""

    rot_world_frame = to_numpy(quat_to_rot_matrix(quat_world))
    if rot_world_frame.ndim == 3:
        rot_world_frame = rot_world_frame[0]
    return {
        "x_in_world": rot_world_frame[:, 0].tolist(),
        "y_in_world": rot_world_frame[:, 1].tolist(),
        "z_in_world": rot_world_frame[:, 2].tolist(),
    }


def world_pos_to_local_pos(
    world_pos: TensorLike,
    local_origin_world: TensorLike,
    local_quat_world: TensorLike,
) -> TensorLike:
    """Transform world positions into a local frame, preserving batch dims."""

    rot_world_local = quat_to_rot_matrix(local_quat_world)
    delta_world = world_pos - local_origin_world
    if isinstance(delta_world, torch.Tensor):
        return torch.matmul(rot_world_local.transpose(-1, -2), delta_world.unsqueeze(-1)).squeeze(-1)
    return np.matmul(np.swapaxes(rot_world_local, -1, -2), np.expand_dims(delta_world, -1)).squeeze(-1)


def step_towards(current: TensorLike, target: TensorLike, *, step_size: float) -> TensorLike:
    """Return a bounded delta from current toward target."""

    delta = target - current
    if isinstance(delta, torch.Tensor):
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(step_size / norm.clamp_min(1e-8), max=1.0)
        return torch.where(norm < 1e-8, torch.zeros_like(delta), delta * scale)

    delta = np.asarray(delta, dtype=np.float64)
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    scale = np.minimum(step_size / np.maximum(norm, 1e-8), 1.0)
    return np.where(norm < 1e-8, np.zeros_like(delta), delta * scale).astype(np.float32)


def _quat_conjugate(quat: TensorLike) -> TensorLike:
    """Return quaternion conjugates."""

    if isinstance(quat, torch.Tensor):
        result = quat.clone()
        result[..., 1:] = -result[..., 1:]
        return result
    result = np.array(quat, copy=True)
    result[..., 1:] = -result[..., 1:]
    return result


def _quat_multiply(lhs: TensorLike, rhs: TensorLike) -> TensorLike:
    """Multiply two [w, x, y, z] quaternions."""

    if isinstance(lhs, torch.Tensor) or isinstance(rhs, torch.Tensor):
        like = lhs if isinstance(lhs, torch.Tensor) else rhs
        lhs = as_backend_array(lhs, like=like)
        rhs = as_backend_array(rhs, like=like)
        w1, x1, y1, z1 = torch.unbind(lhs, dim=-1)
        w2, x2, y2, z2 = torch.unbind(rhs, dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    w1, x1, y1, z1 = np.moveaxis(np.asarray(lhs), -1, 0)
    w2, x2, y2, z2 = np.moveaxis(np.asarray(rhs), -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def bounded_quat_delta_in_local_frame(
    *,
    current_quat_world: TensorLike,
    target_quat_world: TensorLike,
    local_quat_world: TensorLike,
    max_step: float,
) -> TensorLike:
    """Return a bounded rotation-vector delta expressed in a local/root frame."""

    target_quat_world = as_backend_array(target_quat_world, like=current_quat_world)
    diff_quat = _quat_multiply(target_quat_world, _quat_conjugate(current_quat_world))

    if isinstance(diff_quat, torch.Tensor):
        vector = diff_quat[..., 1:]
        vector_norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
        axis_world = vector / vector_norm.clamp_min(1e-8)
        angle = 2.0 * torch.atan2(vector_norm.squeeze(-1), diff_quat[..., 0])
        angle = torch.where(angle > torch.pi, angle - 2.0 * torch.pi, angle)
        angle = torch.where(angle < -torch.pi, angle + 2.0 * torch.pi, angle)
        bounded_angle = torch.clamp(angle, -max_step, max_step)
        rotvec_world = axis_world * bounded_angle.unsqueeze(-1)
        rot_world_local = quat_to_rot_matrix(local_quat_world)
        rotvec_local = torch.matmul(
            rot_world_local.transpose(-1, -2),
            rotvec_world.unsqueeze(-1),
        ).squeeze(-1)
        return torch.where(vector_norm < 1e-8, torch.zeros_like(rotvec_local), rotvec_local)

    vector = diff_quat[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    axis_world = vector / np.maximum(vector_norm, 1e-8)
    angle = 2.0 * np.arctan2(vector_norm[..., 0], diff_quat[..., 0])
    angle = np.where(angle > np.pi, angle - 2.0 * np.pi, angle)
    angle = np.where(angle < -np.pi, angle + 2.0 * np.pi, angle)
    bounded_angle = np.clip(angle, -max_step, max_step)
    rotvec_world = axis_world * np.expand_dims(bounded_angle, -1)
    rot_world_local = quat_to_rot_matrix(local_quat_world)
    rotvec_local = np.matmul(
        np.swapaxes(rot_world_local, -1, -2),
        np.expand_dims(rotvec_world, -1),
    ).squeeze(-1)
    return np.where(vector_norm < 1e-8, np.zeros_like(rotvec_local), rotvec_local).astype(np.float32)
