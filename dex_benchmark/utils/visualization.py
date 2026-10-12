from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import numpy as np


ROBOT_BASE_POSITION_KEYS: dict[str, tuple[int, float]] = {
    "up": (0, 1.0),
    "down": (0, -1.0),
    "left": (1, 1.0),
    "right": (1, -1.0),
    "pageup": (2, 1.0),
    "pagedown": (2, -1.0),
}


@dataclass(frozen=True)
class HumanViewerCameraPose:
    """World-frame pose of SAPIEN's freely movable human-viewer camera."""

    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    rpy_rad: tuple[float, float, float]


def read_human_viewer_camera_pose(viewer: Any) -> HumanViewerCameraPose:
    """Read and validate the current human-viewer camera pose."""

    window = getattr(viewer, "window", None)
    if window is None:
        raise AttributeError("The human viewer does not expose a window.")
    get_camera_pose = getattr(window, "get_camera_pose", None)
    if not callable(get_camera_pose):
        raise AttributeError(
            "The human viewer window does not expose get_camera_pose()."
        )

    camera_pose = get_camera_pose()
    position = np.asarray(getattr(camera_pose, "p", None), dtype=np.float64)
    quaternion = np.asarray(getattr(camera_pose, "q", None), dtype=np.float64)
    if position.shape != (3,):
        raise ValueError(
            f"Human viewer camera position must have shape (3,), got {position.shape}."
        )
    if quaternion.shape != (4,):
        raise ValueError(
            "Human viewer camera quaternion must have shape (4,), got "
            f"{quaternion.shape}."
        )
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("Human viewer camera pose contains non-finite values.")

    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 1e-12:
        raise ValueError("Human viewer camera quaternion has zero length.")
    quaternion /= quaternion_norm
    quaternion_wxyz = (
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
    )
    w, x, y, z = quaternion_wxyz
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(
        max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    )
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return HumanViewerCameraPose(
        position_xyz=(
            float(position[0]),
            float(position[1]),
            float(position[2]),
        ),
        quaternion_wxyz=quaternion_wxyz,
        rpy_rad=(roll, pitch, yaw),
    )


def format_human_viewer_camera_pose(viewer: Any) -> str:
    camera_pose = read_human_viewer_camera_pose(viewer)
    position = camera_pose.position_xyz
    quaternion = camera_pose.quaternion_wxyz
    rpy = camera_pose.rpy_rad
    return (
        "[human-viewer] camera_world: "
        f"p=({position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}), "
        f"q_wxyz=({quaternion[0]:.6f}, {quaternion[1]:.6f}, "
        f"{quaternion[2]:.6f}, {quaternion[3]:.6f}), "
        f"rpy_rad=({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f})"
    )


@dataclass
class HumanViewerPoseHotkey:
    """Print a viewer pose once per key-down edge."""

    key_name: str = "p"
    print_hint: bool = True
    _was_key_down: bool = field(default=False, init=False)
    _hint_was_printed: bool = field(default=False, init=False)

    def poll(self, viewer: Any | None) -> bool:
        if viewer is None:
            self._was_key_down = False
            return False

        window = getattr(viewer, "window", None)
        if window is None:
            self._was_key_down = False
            return False
        key_down = getattr(window, "key_down", None)
        if not callable(key_down):
            raise AttributeError("The human viewer window does not expose key_down().")

        if self.print_hint and not self._hint_was_printed:
            print(
                f"[human-viewer] Press {self.key_name.upper()} to print the "
                "current camera world pose."
            )
            self._hint_was_printed = True

        is_key_down = bool(key_down(self.key_name))
        just_pressed = is_key_down and not self._was_key_down
        self._was_key_down = is_key_down
        if not just_pressed:
            return False

        print(format_human_viewer_camera_pose(viewer))
        return True


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    return np.asarray(value)


def _robot_base_pose_arrays(
    robot_name: str,
    robot: Any,
) -> tuple[Any, np.ndarray, np.ndarray]:
    root_pose = getattr(robot, "root_pose", None)
    if root_pose is None:
        raise AttributeError(f"Robot {robot_name!r} does not expose root_pose.")

    position = _to_numpy(getattr(root_pose, "p", None))
    quaternion = _to_numpy(getattr(root_pose, "q", None))
    if position.ndim not in (1, 2) or position.shape[-1:] != (3,):
        raise ValueError(
            f"Robot {robot_name!r} base position must have shape (3,) or "
            f"[B, 3], got {position.shape}."
        )
    if quaternion.ndim != position.ndim or quaternion.shape[-1:] != (4,):
        raise ValueError(
            f"Robot {robot_name!r} base quaternion must match the position "
            f"batch rank and have shape (4,) or [B, 4], got {quaternion.shape}."
        )
    if position.ndim == 2 and position.shape[0] != quaternion.shape[0]:
        raise ValueError(
            f"Robot {robot_name!r} base pose batch sizes differ: "
            f"position={position.shape}, quaternion={quaternion.shape}."
        )
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError(f"Robot {robot_name!r} base pose contains non-finite values.")
    return root_pose, position, quaternion


def _quaternion_to_rpy(
    quaternion_wxyz: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    quaternion_norm = float(np.linalg.norm(quaternion_wxyz))
    if quaternion_norm <= 1e-12:
        raise ValueError("Robot base quaternion has zero length.")
    normalized_quaternion = quaternion_wxyz / quaternion_norm
    w, x, y, z = normalized_quaternion
    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return normalized_quaternion, (roll, pitch, yaw)


def format_robot_base_world_poses(robots: Mapping[str, Any]) -> str:
    """Format the visible env-0 base world pose for one or more robots."""

    if not robots:
        raise ValueError("At least one robot is required to print a base pose.")

    pose_lines: list[str] = []
    for robot_name, robot in robots.items():
        _root_pose, position, quaternion = _robot_base_pose_arrays(robot_name, robot)
        visible_position = position if position.ndim == 1 else position[0]
        visible_quaternion = quaternion if quaternion.ndim == 1 else quaternion[0]
        normalized_quaternion, rpy = _quaternion_to_rpy(visible_quaternion)
        pose_lines.append(
            f"[human-viewer] robot_base_world[{robot_name}][env=0]: "
            f"p=({visible_position[0]:.6f}, {visible_position[1]:.6f}, "
            f"{visible_position[2]:.6f}), "
            f"q_wxyz=({normalized_quaternion[0]:.6f}, "
            f"{normalized_quaternion[1]:.6f}, "
            f"{normalized_quaternion[2]:.6f}, "
            f"{normalized_quaternion[3]:.6f}), "
            f"rpy_rad=({rpy[0]:.6f}, {rpy[1]:.6f}, {rpy[2]:.6f})"
        )
    return "\n".join(pose_lines)


def _translated_robot_base_poses(
    robots: Mapping[str, Any],
    translation_world_xyz: np.ndarray,
) -> list[tuple[str, Any, Any]]:
    translated_poses: list[tuple[str, Any, Any]] = []
    for robot_name, robot in robots.items():
        set_pose = getattr(robot, "set_pose", None)
        if not callable(set_pose):
            raise AttributeError(f"Robot {robot_name!r} does not expose set_pose().")
        root_pose, _position_numpy, _quaternion_numpy = _robot_base_pose_arrays(
            robot_name,
            robot,
        )
        position = root_pose.p
        clone_position = getattr(position, "clone", None)
        translated_position = (
            clone_position()
            if callable(clone_position)
            else np.asarray(position).copy()
        )
        # [B, 3] + [3] -> [B, 3], preserving each vectorized env's base offset.
        translated_position[..., 0] += float(translation_world_xyz[0])
        translated_position[..., 1] += float(translation_world_xyz[1])
        translated_position[..., 2] += float(translation_world_xyz[2])

        create_from_pq = getattr(type(root_pose), "create_from_pq", None)
        if not callable(create_from_pq):
            raise TypeError(
                f"Robot {robot_name!r} root pose type {type(root_pose).__name__} "
                "does not expose create_from_pq()."
            )
        translated_pose = create_from_pq(p=translated_position, q=root_pose.q)
        translated_poses.append((robot_name, robot, translated_pose))
    return translated_poses


@dataclass
class HumanViewerRobotBaseControl:
    """Move robot bases in world XYZ and print their poses from a human viewer."""

    translation_step_m: float = 0.01
    print_key_name: str = "0"
    print_hint: bool = True
    _was_print_key_down: bool = field(default=False, init=False)
    _hint_was_printed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.translation_step_m) or self.translation_step_m <= 0:
            raise ValueError("translation_step_m must be finite and greater than zero.")
        if not self.print_key_name:
            raise ValueError("print_key_name must not be empty.")

    def poll(self, viewer: Any | None, robots: Mapping[str, Any]) -> bool:
        """Poll held movement keys and the edge-triggered pose print key."""

        if viewer is None:
            self._was_print_key_down = False
            return False
        if not robots:
            raise ValueError("At least one robot is required for base control.")

        window = getattr(viewer, "window", None)
        if window is None:
            self._was_print_key_down = False
            return False
        key_down = getattr(window, "key_down", None)
        if not callable(key_down):
            raise AttributeError("The human viewer window does not expose key_down().")

        if self.print_hint and not self._hint_was_printed:
            print(
                "[human-viewer] Move robot base(s) in world XY with arrow keys "
                "and world Z with PageUp/PageDown "
                f"(step={self.translation_step_m:.3f} m); press "
                f"{self.print_key_name.upper()} to print base world pose(s)."
            )
            self._hint_was_printed = True

        translation_world_xyz = np.zeros(3, dtype=np.float64)
        for key_name, (axis_index, direction) in ROBOT_BASE_POSITION_KEYS.items():
            if bool(key_down(key_name)):
                translation_world_xyz[axis_index] += (
                    direction * self.translation_step_m
                )

        moved = bool(np.any(translation_world_xyz != 0.0))
        if moved:
            translated_poses = _translated_robot_base_poses(
                robots,
                translation_world_xyz,
            )
            for _robot_name, robot, translated_pose in translated_poses:
                robot.set_pose(translated_pose)

        is_print_key_down = bool(key_down(self.print_key_name))
        just_pressed_print = is_print_key_down and not self._was_print_key_down
        self._was_print_key_down = is_print_key_down
        if just_pressed_print:
            print(format_robot_base_world_poses(robots))

        return moved or just_pressed_print


def save_report(text: str, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    return output
