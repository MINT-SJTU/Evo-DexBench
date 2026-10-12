"""Quest3-facing adapter from benchmark observations to canonical snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dex_benchmark.data.canonical.schema import CanonicalObservation
from dex_benchmark.data.canonical.sensor_adapter import SensorAdapter


class Quest3ObservationAdapter:
    """Normalize main-environment observations for the future recorder.

    The adapter is deliberately side-effect free with respect to the simulator:
    it only converts an already returned observation.  It accepts both the
    benchmark ``Observation`` dataclass and its model-facing dictionary form.
    """

    def __init__(
        self,
        *,
        control_hz: float = 30.0,
        image_hz: float = 30.0,
        required_cameras: tuple[str, ...] = (),
        required_manipulators: tuple[str, ...] = (),
    ) -> None:
        self.sensor_adapter = SensorAdapter(control_hz=control_hz, image_hz=image_hz)
        self.required_cameras = tuple(str(role) for role in required_cameras)
        self.required_manipulators = tuple(str(role) for role in required_manipulators)
        if len(set(self.required_cameras)) != len(self.required_cameras):
            raise ValueError("required_cameras must not contain duplicates.")
        if len(set(self.required_manipulators)) != len(self.required_manipulators):
            raise ValueError("required_manipulators must not contain duplicates.")

    @property
    def control_hz(self) -> float:
        return self.sensor_adapter.control_hz

    @property
    def image_hz(self) -> float:
        return self.sensor_adapter.image_hz

    def reset(self) -> None:
        self.sensor_adapter.reset()

    def snapshot(
        self,
        obs: Any,
        *,
        observation_id: int,
        timestamp_s: float,
        force_image: bool = False,
    ) -> CanonicalObservation:
        normalized = self._to_mapping(obs)
        # Validate the full contract on every control sample.  This catches a
        # sensor disappearing mid-episode instead of producing a partial file.
        self.validate_contract(normalized)
        return self.sensor_adapter.capture(
            normalized,
            observation_id=observation_id,
            timestamp_s=timestamp_s,
            force_image=force_image,
        )

    def validate_contract(self, obs: Any) -> None:
        """Validate the complete sensor contract before an episode starts.

        Role presence alone is insufficient for canonical collection: every
        required camera must expose an atomic RGB-D pair and every required
        manipulator must expose taxel normal-force data.  This check is intended
        for reset/preflight, while ``SensorAdapter`` continues validating
        sampled frames during the episode.
        """

        normalized = self._to_mapping(obs)
        self._validate_roles(normalized)
        cameras = normalized["cameras"]
        for role in self.required_cameras:
            camera = cameras[role]
            if not isinstance(camera, Mapping):
                raise TypeError(f"obs['cameras'][{role!r}] must be a mapping.")
            if camera.get("rgb") is None or camera.get("depth") is None:
                raise ValueError(
                    f"Required camera {role!r} must provide both rgb and depth."
                )
        manipulators = normalized["manipulators"]
        for role in self.required_manipulators:
            manipulator = manipulators[role]
            if not isinstance(manipulator, Mapping):
                raise TypeError(f"obs['manipulators'][{role!r}] must be a mapping.")
            if manipulator.get("taxel_normal_force") is None:
                raise ValueError(
                    f"Required manipulator {role!r} must provide "
                    "taxel_normal_force."
                )

    @staticmethod
    def _to_mapping(obs: Any) -> Mapping[str, Any]:
        to_dict = getattr(obs, "to_dict", None)
        if callable(to_dict):
            obs = to_dict()
        if not isinstance(obs, Mapping):
            raise TypeError(
                "Quest3 observations must be a mapping or expose to_dict(), "
                f"got {type(obs).__name__}."
            )
        return obs

    def _validate_roles(self, obs: Mapping[str, Any]) -> None:
        cameras = obs.get("cameras", {})
        manipulators = obs.get("manipulators", {})
        if not isinstance(cameras, Mapping):
            raise TypeError("obs['cameras'] must be a mapping.")
        if not isinstance(manipulators, Mapping):
            raise TypeError("obs['manipulators'] must be a mapping.")
        missing_cameras = [role for role in self.required_cameras if role not in cameras]
        if missing_cameras:
            raise KeyError("Observation is missing required cameras: " + ", ".join(missing_cameras))
        missing_manipulators = [
            role for role in self.required_manipulators if role not in manipulators
        ]
        if missing_manipulators:
            raise KeyError(
                "Observation is missing required manipulators: "
                + ", ".join(missing_manipulators)
            )
