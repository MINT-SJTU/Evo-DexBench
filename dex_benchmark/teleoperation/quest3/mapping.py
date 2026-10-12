from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dex_benchmark.teleoperation.retargeting.coordinates import (
    OPERATOR2MANO_LEFT,
    OPERATOR2MANO_RIGHT,
)
from dex_benchmark.teleoperation.quest3.stream import Quest3HandInput


# WebXR has an extra metacarpal joint on each non-thumb finger. Dropping those
# joints yields the 21-joint MANO/MediaPipe order used by our retargeting YAML.
WEBXR_TO_MANO_JOINT_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24],
    dtype=np.int64,
)

# The camera-derived operator frame places hand width on MANO Y, while the
# Shadow Hand URDF spreads fingers along X. For row-vector landmarks this is
# the transpose of a -90 degree rotation about palm Z: [x, y, z] -> [y, -x, z].
QUEST3_MANO_PALM_ALIGNMENT = np.asarray(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


# WebXR: +X right, +Y up, -Z forward. Panda root: +X forward, +Y left, +Z up.
DEFAULT_WEBXR_TO_ROBOT_ROOT = np.asarray(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def quest3_hand_to_mano_joint_positions(
    hand: Quest3HandInput,
    *,
    side: str,
) -> np.ndarray:
    """Convert 25 WebXR joint positions to canonical 21-joint MANO space."""

    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
    joint_matrices = np.asarray(hand.joint_matrices, dtype=np.float32)
    if joint_matrices.shape != (25, 4, 4):
        raise ValueError(
            f"Expected Quest 3 joint_matrices shape (25, 4, 4), got {joint_matrices.shape}."
        )

    # [25, 4, 4] -> [21, 3]
    joint_positions = joint_matrices[WEBXR_TO_MANO_JOINT_INDICES, :3, 3].copy()
    joint_positions -= joint_positions[0:1]
    wrist_frame = _estimate_wrist_frame(joint_positions)
    operator_to_mano = OPERATOR2MANO_LEFT if side == "left" else OPERATOR2MANO_RIGHT
    # [21, 3] @ [3, 3] @ [3, 3] @ [3, 3] -> [21, 3]
    return (
        joint_positions
        @ wrist_frame
        @ operator_to_mano
        @ QUEST3_MANO_PALM_ALIGNMENT
    ).astype(np.float32)


def _estimate_wrist_frame(joint_positions: np.ndarray) -> np.ndarray:
    if joint_positions.shape != (21, 3):
        raise ValueError(f"Expected joint positions shape (21, 3), got {joint_positions.shape}.")
    wrist_index_middle = joint_positions[[0, 5, 9], :]
    centered_points = wrist_index_middle - wrist_index_middle.mean(axis=0, keepdims=True)
    _u, singular_values, vh = np.linalg.svd(centered_points)
    if singular_values[0] < 1e-6 or singular_values[1] < 1e-6:
        raise ValueError("Quest 3 palm landmarks are degenerate; cannot estimate wrist frame.")

    normal = vh[2]
    wrist_to_middle = wrist_index_middle[0] - wrist_index_middle[2]
    x_axis = wrist_to_middle - np.dot(wrist_to_middle, normal) * normal
    x_norm = float(np.linalg.norm(x_axis))
    if x_norm < 1e-6:
        raise ValueError("Quest 3 wrist and middle-finger landmarks are coincident.")
    x_axis /= x_norm
    z_axis = np.cross(x_axis, normal)
    if np.dot(z_axis, centered_points[1] - centered_points[2]) < 0:
        normal *= -1.0
        z_axis *= -1.0
    return np.stack([x_axis, normal, z_axis], axis=1).astype(np.float32)


@dataclass(frozen=True)
class Quest3WristMotionConfig:
    """Relative Quest wrist pose mapping into one Panda root frame."""

    motion_mode: Literal["frame_delta", "legacy_target"] = "frame_delta"
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    max_position_delta_m: float = 0.03
    max_rotation_delta_rad: float = 0.15
    max_target_distance_m: float = 0.60
    max_tracking_position_jump_m: float = 0.45
    max_tracking_rotation_jump_rad: float = 1.0
    enable_rotation: bool = True
    # Hand retargeting is absolute joint-position control.  These two limits
    # keep one reacquired/outlier frame from replacing the previous hand target
    # in a single control tick.
    # Quest hand tracking is already temporally filtered by WebXR. Keep the
    # retargeted target responsive by default; callers can opt into additional
    # smoothing for noisy tracking.
    hand_target_smoothing_alpha: float = 1.0
    max_hand_joint_delta: float = 0.1

    def __post_init__(self) -> None:
        if self.motion_mode not in ("frame_delta", "legacy_target"):
            raise ValueError(f"Unsupported wrist motion mode: {self.motion_mode!r}.")
        if self.position_scale <= 0:
            raise ValueError("position_scale must be positive.")
        if self.rotation_scale < 0:
            raise ValueError("rotation_scale must be non-negative.")
        if self.max_position_delta_m <= 0 or self.max_rotation_delta_rad <= 0:
            raise ValueError("Per-step wrist motion limits must be positive.")
        if self.max_target_distance_m <= 0:
            raise ValueError("max_target_distance_m must be positive.")
        if self.max_tracking_position_jump_m <= 0:
            raise ValueError("max_tracking_position_jump_m must be positive.")
        if self.max_tracking_rotation_jump_rad <= 0:
            raise ValueError("max_tracking_rotation_jump_rad must be positive.")
        if not 0.0 < self.hand_target_smoothing_alpha <= 1.0:
            raise ValueError("hand_target_smoothing_alpha must be in (0, 1].")
        if self.max_hand_joint_delta <= 0:
            raise ValueError("max_hand_joint_delta must be positive.")


class Quest3WristPoseMapper:
    """Map a calibrated WebXR wrist pose to bounded EE delta commands."""

    def __init__(
        self,
        config: Quest3WristMotionConfig | None = None,
        *,
        webxr_to_robot_root: np.ndarray | None = None,
    ) -> None:
        self.config = config or Quest3WristMotionConfig()
        coordinate_rotation = (
            DEFAULT_WEBXR_TO_ROBOT_ROOT
            if webxr_to_robot_root is None
            else np.asarray(webxr_to_robot_root, dtype=np.float32)
        )
        if coordinate_rotation.shape != (3, 3):
            raise ValueError(
                "webxr_to_robot_root must have shape (3, 3), "
                f"got {coordinate_rotation.shape}."
            )
        if not np.allclose(
            coordinate_rotation.T @ coordinate_rotation,
            np.eye(3),
            atol=1e-5,
        ):
            raise ValueError("webxr_to_robot_root must be an orthonormal rotation matrix.")
        if not np.isclose(np.linalg.det(coordinate_rotation), 1.0, atol=1e-5):
            raise ValueError("webxr_to_robot_root must be a proper rotation matrix.")

        self.webxr_to_robot_root = coordinate_rotation.copy()
        self.neutral_wrist_matrix: np.ndarray | None = None
        self.previous_wrist_matrix: np.ndarray | None = None
        self.previous_target_position_m = np.zeros(3, dtype=np.float32)
        self.latest_target_position_m = np.zeros(3, dtype=np.float32)
        self.previous_target_rotation = np.eye(3, dtype=np.float32)
        self.latest_target_rotation = np.eye(3, dtype=np.float32)

    def reset(self) -> None:
        self.neutral_wrist_matrix = None
        self.previous_wrist_matrix = None
        self.previous_target_position_m.fill(0.0)
        self.latest_target_position_m.fill(0.0)
        self.previous_target_rotation[:] = np.eye(3, dtype=np.float32)
        self.latest_target_rotation[:] = np.eye(3, dtype=np.float32)

    def update(self, wrist_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        wrist_matrix = _validated_rigid_matrix(wrist_matrix)
        if self.neutral_wrist_matrix is None:
            self.neutral_wrist_matrix = wrist_matrix.copy()
            self.previous_wrist_matrix = wrist_matrix.copy()
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        assert self.previous_wrist_matrix is not None
        tracking_position_jump_m = float(
            np.linalg.norm(
                wrist_matrix[:3, 3] - self.previous_wrist_matrix[:3, 3]
            )
        )
        tracking_rotation_jump = (
            wrist_matrix[:3, :3]
            @ self.previous_wrist_matrix[:3, :3].T
        )
        _, tracking_rotation_jump_rad = _rotation_matrix_to_axis_angle(
            tracking_rotation_jump
        )
        self.previous_wrist_matrix = wrist_matrix.copy()
        if (
            tracking_position_jump_m > self.config.max_tracking_position_jump_m
            or tracking_rotation_jump_rad
            > self.config.max_tracking_rotation_jump_rad
        ):
            # WebXR may report one valid but discontinuous matrix immediately
            # before/after tracking loss or a headset recenter. Treat it as a
            # new local origin instead of converting the jump into arm motion.
            self.neutral_wrist_matrix = wrist_matrix.copy()
            self.previous_target_position_m.fill(0.0)
            self.latest_target_position_m.fill(0.0)
            self.previous_target_rotation[:] = np.eye(3, dtype=np.float32)
            self.latest_target_rotation[:] = np.eye(3, dtype=np.float32)
            return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        neutral = self.neutral_wrist_matrix
        coordinate_rotation = self.webxr_to_robot_root
        target_position_m = (
            coordinate_rotation @ (wrist_matrix[:3, 3] - neutral[:3, 3])
        ) * self.config.position_scale
        target_norm = float(np.linalg.norm(target_position_m))
        if target_norm > self.config.max_target_distance_m:
            target_position_m *= self.config.max_target_distance_m / target_norm
        self.latest_target_position_m[:] = target_position_m
        position_delta_m = target_position_m - self.previous_target_position_m
        position_delta_m = _clip_vector_norm(
            position_delta_m,
            self.config.max_position_delta_m,
        )
        if self.config.motion_mode == "legacy_target":
            self.previous_target_position_m += position_delta_m
        else:
            self.previous_target_position_m[:] = target_position_m

        if self.config.enable_rotation and self.config.rotation_scale > 0:
            relative_webxr_rotation = wrist_matrix[:3, :3] @ neutral[:3, :3].T
            target_rotation = (
                coordinate_rotation
                @ relative_webxr_rotation
                @ coordinate_rotation.T
            )
            target_axis, target_angle = _rotation_matrix_to_axis_angle(target_rotation)
            self.latest_target_rotation = _axis_angle_to_matrix(
                target_axis,
                target_angle * self.config.rotation_scale,
            )
        rotation_delta_euler = self.continue_rotation()
        return position_delta_m.astype(np.float32), rotation_delta_euler

    def rewind_commanded_target(
        self,
        *,
        position_error_root_m: np.ndarray,
        rotation_error_root: np.ndarray,
    ) -> None:
        """Requeue EE error after IK reset without changing WebXR calibration."""

        position_error = np.asarray(position_error_root_m, dtype=np.float32)
        rotation_error = np.asarray(rotation_error_root, dtype=np.float32)
        if position_error.shape != (3,) or not np.isfinite(position_error).all():
            raise ValueError("position_error_root_m must be a finite three-vector.")
        if rotation_error.shape != (3, 3) or not np.isfinite(rotation_error).all():
            raise ValueError("rotation_error_root must be a finite 3x3 matrix.")
        if not np.allclose(rotation_error.T @ rotation_error, np.eye(3), atol=2e-3):
            raise ValueError("rotation_error_root must be orthonormal.")
        if not np.isclose(np.linalg.det(rotation_error), 1.0, atol=2e-3):
            raise ValueError("rotation_error_root must be a proper rotation matrix.")

        # The controller target is about to move back to the current TCP. Move
        # the mapper's commanded pose back by the same amount, so subsequent
        # bounded deltas retry the outstanding absolute WebXR-relative target.
        self.previous_target_position_m -= position_error
        self.previous_target_rotation = (
            rotation_error.T @ self.latest_target_rotation
        ).astype(np.float32)

    def continue_rotation(self) -> np.ndarray:
        """Advance one bounded control step toward the latest wrist orientation."""
        if (
            self.neutral_wrist_matrix is None
            or not self.config.enable_rotation
            or self.config.rotation_scale <= 0
        ):
            return np.zeros(3, dtype=np.float32)

        step_rotation = (
            self.latest_target_rotation @ self.previous_target_rotation.T
        )
        step_axis, step_angle = _rotation_matrix_to_axis_angle(step_rotation)
        step_angle = min(step_angle, self.config.max_rotation_delta_rad)
        limited_step_rotation = _axis_angle_to_matrix(step_axis, step_angle)
        self.previous_target_rotation = (
            limited_step_rotation @ self.previous_target_rotation
        ).astype(np.float32)
        return _rotation_matrix_to_xyz_euler(limited_step_rotation).astype(
            np.float32
        )


def _validated_rigid_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected wrist matrix shape (4, 4), got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Wrist matrix contains NaN or infinite values.")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-4):
        raise ValueError("Wrist matrix has an invalid homogeneous bottom row.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("Wrist matrix rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError("Wrist matrix rotation must be proper.")
    return matrix


def _clip_vector_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= max_norm:
        return vector
    return vector * (max_norm / max(vector_norm, 1e-8))


def _rotation_matrix_to_axis_angle(rotation: np.ndarray) -> tuple[np.ndarray, float]:
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), 0.0
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-7:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, np.argmin(np.abs(eigenvalues - 1.0))])
        axis_norm = float(np.linalg.norm(axis))
    return (axis / max(axis_norm, 1e-8)).astype(np.float32), angle


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-8)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return rotation.astype(np.float32)


def _rotation_matrix_to_xyz_euler(rotation: np.ndarray) -> np.ndarray:
    # ManiSkill reconstructs this command as Rx(x) @ Ry(y) @ Rz(z).
    y_angle = float(np.arcsin(np.clip(rotation[0, 2], -1.0, 1.0)))
    cos_y = float(np.cos(y_angle))
    if abs(cos_y) > 1e-6:
        x_angle = float(np.arctan2(-rotation[1, 2], rotation[2, 2]))
        z_angle = float(np.arctan2(-rotation[0, 1], rotation[0, 0]))
    else:
        x_angle = float(np.arctan2(rotation[2, 1], rotation[1, 1]))
        z_angle = 0.0
    return np.asarray([x_angle, y_angle, z_angle], dtype=np.float32)
