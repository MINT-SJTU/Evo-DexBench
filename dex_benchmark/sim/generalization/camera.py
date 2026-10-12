from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from dex_benchmark.sim.sensors.cameras import CameraViewSpec


@dataclass(frozen=True)
class CameraChangeLimits:
    side_angle_rad: float
    height_angle_rad: float
    distance_fraction: float
    target_shift_m: float
    fov_angle_rad: float


@dataclass(frozen=True)
class CameraViewValues:
    eye_local_xyz: np.ndarray
    target_local_xyz: np.ndarray
    fov_rad: np.ndarray


CAMERA_LIMITS_BY_UID = {
    "context_camera": CameraChangeLimits(
        side_angle_rad=math.radians(7.5),
        height_angle_rad=math.radians(5.0),
        distance_fraction=0.10,
        target_shift_m=0.05,
        fov_angle_rad=math.radians(5.0),
    ),
    "interaction_camera": CameraChangeLimits(
        side_angle_rad=math.radians(5.0),
        height_angle_rad=math.radians(4.0),
        distance_fraction=0.08,
        target_shift_m=0.03,
        fov_angle_rad=math.radians(5.0),
    ),
}


def _sample_array(samples: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        values = np.asarray(samples[name], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing camera generalization value {name!r}.") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(
            f"Camera value {name!r} must be a finite one-dimensional array."
        )
    if (np.abs(values) > 1.0 + 1e-12).any():
        raise ValueError(f"Camera value {name!r} must stay in [-1, 1].")
    return values


def camera_view_values(
    view: CameraViewSpec,
    samples: Mapping[str, np.ndarray],
) -> CameraViewValues:
    """Build valid per-environment eye, target, and FOV values."""

    limits = CAMERA_LIMITS_BY_UID[view.camera_uid]
    side_sample = _sample_array(samples, "camera_side")
    batch_size = side_sample.shape[0]
    named_samples = {
        name: _sample_array(samples, name)
        for name in (
            "camera_height",
            "camera_distance",
            "camera_target_x",
            "camera_target_y",
            "camera_target_z",
            "camera_fov",
        )
    }
    for name, values in named_samples.items():
        if values.shape != (batch_size,):
            raise ValueError(
                f"Camera value {name!r} has shape {values.shape}; expected "
                f"({batch_size},)."
            )

    nominal_eye = np.asarray(view.eye_local_xyz, dtype=np.float64)
    nominal_target = np.asarray(view.target_local_xyz, dtype=np.float64)
    nominal_offset = nominal_eye - nominal_target
    nominal_distance = float(np.linalg.norm(nominal_offset))
    if nominal_distance <= 1e-6:
        raise ValueError(f"Camera {view.camera_uid!r} eye is too close to its target.")

    nominal_side_angle = math.atan2(nominal_offset[1], nominal_offset[0])
    nominal_height_angle = math.asin(
        float(np.clip(nominal_offset[2] / nominal_distance, -1.0, 1.0))
    )
    side_angle = nominal_side_angle + side_sample * limits.side_angle_rad
    height_angle = (
        nominal_height_angle
        + named_samples["camera_height"] * limits.height_angle_rad
    )
    distance = nominal_distance * (
        1.0 + named_samples["camera_distance"] * limits.distance_fraction
    )

    target_shift = np.stack(
        [
            named_samples["camera_target_x"],
            named_samples["camera_target_y"],
            named_samples["camera_target_z"],
        ],
        axis=1,
    ) * limits.target_shift_m
    # [3] -> [B, 3]: move the nominal target in the interaction-site frame.
    target_local_xyz = nominal_target[None, :] + target_shift

    horizontal_distance = distance * np.cos(height_angle)
    eye_offset = np.stack(
        [
            horizontal_distance * np.cos(side_angle),
            horizontal_distance * np.sin(side_angle),
            distance * np.sin(height_angle),
        ],
        axis=1,
    )
    # [B, 3] + [B, 3] -> [B, 3].
    eye_local_xyz = target_local_xyz + eye_offset
    fov_rad = (
        view.fov_rad + named_samples["camera_fov"] * limits.fov_angle_rad
    )
    if not ((fov_rad > 0.0) & (fov_rad < math.pi)).all():
        raise ValueError("Randomized camera FOV must remain in (0, pi).")

    return CameraViewValues(
        eye_local_xyz=eye_local_xyz,
        target_local_xyz=target_local_xyz,
        fov_rad=fov_rad,
    )
