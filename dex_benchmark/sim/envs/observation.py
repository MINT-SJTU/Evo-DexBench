from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Mapping

import numpy as np


ObsDict = Dict[str, Any]
ActionArray = Any


def _concat_pose_parts(position: Any, quaternion: Any) -> Any:
    if hasattr(position, "detach") and hasattr(quaternion, "detach"):
        try:
            import torch

            return torch.cat([position, quaternion], dim=-1)
        except Exception:
            pass

    position_array = np.asarray(position)
    quaternion_array = np.asarray(quaternion)
    return np.concatenate([position_array, quaternion_array], axis=-1)


def _to_obs_value(value: Any) -> Any:
    """Recursively convert internal values to model-facing obs values."""

    if value is None:
        return None

    if isinstance(value, str):
        raise TypeError(
            "String values are not part of the numeric simulator observation. "
            "Pass language instructions through the model-input adapter instead."
        )

    if isinstance(value, (int, float, bool, np.generic)):
        return value

    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value

    raw_pose = getattr(value, "raw_pose", None)
    if raw_pose is not None:
        return _to_obs_value(raw_pose)

    position = getattr(value, "p", None)
    quaternion = getattr(value, "q", None)
    if position is not None and quaternion is not None:
        return _to_obs_value(_concat_pose_parts(position, quaternion))

    if isinstance(value, dict):
        return {key: _to_obs_value(item) for key, item in value.items()}

    if isinstance(value, tuple):
        converted = tuple(_to_obs_value(item) for item in value)
        if all(isinstance(item, (int, float, bool, np.generic)) for item in converted):
            return np.asarray(converted)
        return converted

    if isinstance(value, list):
        converted = [_to_obs_value(item) for item in value]
        if all(isinstance(item, (int, float, bool, np.generic)) for item in converted):
            return np.asarray(converted)
        return converted

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_obs_value(getattr(value, field.name))
            for field in fields(value)
        }

    raise TypeError(
        f"Unsupported obs value type {type(value).__name__}; "
        "convert simulator objects before exposing observations."
    )


@dataclass(frozen=True)
class CameraObservation:
    """RGB/depth observation for one semantically named camera."""

    rgb: Any | None = None
    depth: Any | None = None

    def to_dict(self) -> ObsDict:
        obs: ObsDict = {}
        if self.rgb is not None:
            obs["rgb"] = _to_obs_value(self.rgb)
        if self.depth is not None:
            obs["depth"] = _to_obs_value(self.depth)
        return obs


@dataclass(frozen=True)
class ManipulatorObservation:
    """Proprioception and tactile state for one manipulator role."""

    joint_position: Any | None = None
    joint_velocity: Any | None = None
    end_effector_pose: Any | None = None
    palm_pose: Any | None = None
    fingertip_poses: Any | None = None
    taxel_normal_force: Any | None = None

    def to_dict(self) -> ObsDict:
        required_values = {
            "joint_position": self.joint_position,
            "joint_velocity": self.joint_velocity,
            "end_effector_pose": self.end_effector_pose,
            "palm_pose": self.palm_pose,
            "fingertip_poses": self.fingertip_poses,
            "taxel_normal_force": self.taxel_normal_force,
        }
        missing_fields = [
            field_name
            for field_name, value in required_values.items()
            if value is None
        ]
        if missing_fields:
            raise ValueError(
                "Manipulator observation fields must not be None: "
                + ", ".join(missing_fields)
                + "."
            )
        return {
            field_name: _to_obs_value(value)
            for field_name, value in required_values.items()
        }


def _role_key(role: Any) -> str:
    value = getattr(role, "value", role)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Observation role keys must be non-empty strings, got {role!r}.")
    return value


@dataclass(frozen=True)
class Observation:
    """Numeric observation returned by the simulator environment."""

    cameras: Mapping[Any, CameraObservation]
    manipulators: Mapping[Any, ManipulatorObservation]
    pointcloud: Any | None = None

    def to_dict(self, use_pointcloud: bool = False) -> ObsDict:
        camera_obs = (
            {}
            if use_pointcloud
            else {
                _role_key(role): camera.to_dict()
                for role, camera in self.cameras.items()
            }
        )
        obs: ObsDict = {
            "cameras": camera_obs,
            "manipulators": {
                _role_key(role): manipulator.to_dict()
                for role, manipulator in self.manipulators.items()
            },
        }
        if use_pointcloud and self.pointcloud is not None:
            obs["pointcloud"] = _to_obs_value(self.pointcloud)
        return obs
