from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dex_benchmark.policy_api.policy_features import (
    camera_roles_for_embodiment,
    roles_for_embodiment,
)


EmbodimentMode = Literal["single", "dual"]
CameraProfile = Literal["standard", "interaction"]
TactileMode = Literal["full", "none"]


@dataclass(frozen=True)
class KineDexPolicyConfig:
    host: str = "127.0.0.1"
    port: int = 8400
    embodiment: EmbodimentMode = "single"
    camera_profile: CameraProfile = "standard"
    tactile_mode: TactileMode = "full"
    prediction_horizon: int = 16
    observation_horizon: int = 2
    available_action_steps: int = 15
    execute_steps: int = 8
    control_hz: int = 30
    image_hz: int = 30
    checkpoint_fps: int = 30
    checkpoint_image_hz: int = 30
    force_gain_rad_per_n: float = 0.005
    max_force_residual_rad: float = 0.05

    def __post_init__(self) -> None:
        if not self.host.strip() or not 1 <= self.port <= 65535:
            raise ValueError("KineDex host/port are invalid.")
        if self.embodiment not in ("single", "dual"):
            raise ValueError("embodiment must be single or dual.")
        if self.camera_profile not in ("standard", "interaction"):
            raise ValueError("camera_profile must be standard or interaction.")
        if self.camera_profile == "interaction" and self.embodiment != "dual":
            raise ValueError("The interaction camera profile requires dual embodiment.")
        if self.tactile_mode not in ("full", "none"):
            raise ValueError("tactile_mode must be full or none.")
        if self.prediction_horizon < self.observation_horizon:
            raise ValueError("prediction_horizon must cover observation_horizon.")
        if self.available_action_steps <= 0:
            raise ValueError("available_action_steps must be positive.")
        if not 1 <= self.execute_steps <= self.available_action_steps:
            raise ValueError("execute_steps exceeds available actions.")
        if self.force_gain_rad_per_n < 0 or self.max_force_residual_rad < 0:
            raise ValueError("Force-control gains must be non-negative.")
        if self.control_hz <= 0 or self.image_hz <= 0:
            raise ValueError("KineDex control/image frequencies must be positive.")
        if self.checkpoint_fps <= 0 or self.checkpoint_image_hz <= 0:
            raise ValueError("KineDex checkpoint frequencies must be positive.")
        if self.control_hz != self.checkpoint_fps:
            raise ValueError("control_hz must match checkpoint_fps to preserve action timing.")
        if self.control_hz % self.image_hz != 0:
            raise ValueError("control_hz must be divisible by image_hz.")
        if self.image_hz != self.checkpoint_image_hz:
            raise ValueError("image_hz must match checkpoint_image_hz to preserve RGB history timing.")

    @property
    def roles(self) -> tuple[str, ...]:
        return roles_for_embodiment(self.embodiment)

    @property
    def camera_roles(self) -> tuple[str, ...]:
        if self.camera_profile == "interaction":
            return ("interaction", "wrist_left", "wrist_right")
        return camera_roles_for_embodiment(self.embodiment)

    @property
    def role_count(self) -> int:
        return len(self.roles)

    @property
    def state_dim(self) -> int:
        return 31 * self.role_count

    @property
    def motion_action_dim(self) -> int:
        return 30 * self.role_count

    @property
    def force_dim(self) -> int:
        return 5 * self.role_count if self.tactile_mode == "full" else 0

    @property
    def model_action_dim(self) -> int:
        return self.motion_action_dim + self.force_dim

    @property
    def finger_count(self) -> int:
        return 5 * self.role_count

    @property
    def image_update_interval(self) -> int:
        return self.control_hz // self.image_hz
