"""Extract and rate-limit training sensors from benchmark observations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dex_benchmark.data.canonical.schema import CanonicalObservation, to_cpu_copy


def _array(value: Any, *, field_name: str) -> np.ndarray:
    copied = to_cpu_copy(value)
    try:
        array = np.asarray(copied)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be array-like.") from exc
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{field_name} must be numeric, got {array.dtype}.")
    # A single-env ManiSkill observation commonly has a leading [1] dimension.
    # Remove only singleton batch dimensions; never squeeze a real image axis.
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return np.array(array, copy=True, order="C")


def _rgb_array(value: Any, *, field_name: str) -> np.ndarray:
    array = _array(value, field_name=field_name)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(
            f"{field_name} must have shape [H, W, 3/4], got {array.shape}."
        )
    if array.dtype.kind == "f" and array.size:
        maximum = float(np.nanmax(array))
        if maximum <= 1.0 + 1e-6:
            array = array * 255.0
    return np.clip(array, 0.0, 255.0).astype(np.uint8, copy=False)


def _depth_array(value: Any, *, field_name: str) -> np.ndarray:
    array = _array(value, field_name=field_name)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"{field_name} must have shape [H, W], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} contains NaN or infinite values.")
    return array.astype(np.float32 if array.dtype.kind == "f" else array.dtype, copy=True)


class SensorAdapter:
    """Build canonical observations at separate control and image rates.

    The adapter only reads model-facing observations.  It never calls the
    viewer or simulator rendering APIs, which keeps the control thread
    independent from human-viewer latency.
    """

    def __init__(self, *, control_hz: float = 30.0, image_hz: float = 30.0) -> None:
        if control_hz <= 0.0:
            raise ValueError("control_hz must be positive.")
        if image_hz <= 0.0 or image_hz > control_hz:
            raise ValueError("image_hz must be positive and no greater than control_hz.")
        image_stride = control_hz / image_hz
        if not abs(image_stride - round(image_stride)) <= 1e-6:
            raise ValueError("control_hz must be an integer multiple of image_hz.")
        self.control_hz = float(control_hz)
        self.image_hz = float(image_hz)
        self.image_stride = int(round(image_stride))

    def reset(self) -> None:
        """Reset adapter state while retaining the observation-ID sampling grid."""

    def image_due(self, observation_id: int, *, force: bool = False) -> bool:
        """Return whether the next observation should include RGB-D data."""

        if observation_id < 0:
            raise ValueError("observation_id must be non-negative.")
        return bool(force or observation_id % self.image_stride == 0)

    def capture(
        self,
        obs: Mapping[str, Any],
        *,
        observation_id: int,
        timestamp_s: float,
        force_image: bool = False,
    ) -> CanonicalObservation:
        if not isinstance(obs, Mapping):
            raise TypeError(f"obs must be a mapping, got {type(obs).__name__}.")
        if not np.isfinite(float(timestamp_s)):
            raise ValueError("observation timestamp must be finite.")

        state: dict[str, np.ndarray] = {}
        taxel: dict[str, np.ndarray] = {}
        manipulators = obs.get("manipulators", {})
        if not isinstance(manipulators, Mapping):
            raise TypeError("obs['manipulators'] must be a mapping.")
        for raw_role, raw_manipulator in manipulators.items():
            role = str(raw_role)
            if not isinstance(raw_manipulator, Mapping):
                raise TypeError(f"obs['manipulators'][{role!r}] must be a mapping.")
            for field_name in (
                "joint_position",
                "joint_velocity",
                "end_effector_pose",
                "palm_pose",
                "fingertip_poses",
            ):
                value = raw_manipulator.get(field_name)
                if value is not None:
                    state[f"{role}.{field_name}"] = _array(
                        value,
                        field_name=f"manipulators.{role}.{field_name}",
                    )
            force = raw_manipulator.get("taxel_normal_force")
            if force is not None:
                taxel[role] = _array(
                    force,
                    field_name=f"manipulators.{role}.taxel_normal_force",
                )

        images: dict[str, np.ndarray] = {}
        depths: dict[str, np.ndarray] = {}
        cameras = obs.get("cameras", {})
        if not isinstance(cameras, Mapping):
            raise TypeError("obs['cameras'] must be a mapping.")
        image_due = self.image_due(observation_id, force=force_image)
        if image_due:
            for raw_role, raw_camera in cameras.items():
                role = str(raw_role)
                if not isinstance(raw_camera, Mapping):
                    raise TypeError(f"obs['cameras'][{role!r}] must be a mapping.")
                rgb = raw_camera.get("rgb")
                depth = raw_camera.get("depth")
                # A formal RGB-D sample is atomic.  If a camera has only one
                # modality, leave it out and let schema validation report it.
                if rgb is None and depth is None:
                    continue
                if rgb is None or depth is None:
                    raise ValueError(
                        f"Camera {role!r} must provide both rgb and depth when sampled."
                    )
                images[role] = _rgb_array(rgb, field_name=f"cameras.{role}.rgb")
                depths[role] = _depth_array(
                    depth,
                    field_name=f"cameras.{role}.depth",
                )

        # Every field above has already been converted to an independent,
        # CPU-owned NumPy array by _array/_rgb_array/_depth_array. Calling
        # CanonicalObservation.cpu_copy() here would duplicate every RGB-D
        # frame before it even reaches the asynchronous recorder.
        return CanonicalObservation(
            observation_id=int(observation_id),
            timestamp_s=float(timestamp_s),
            state=state,
            taxel=taxel,
            images=images,
            depth=depths,
            image_timestamp_s=(float(timestamp_s) if images else None),
        )
