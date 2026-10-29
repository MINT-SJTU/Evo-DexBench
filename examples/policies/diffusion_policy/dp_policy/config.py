from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dex_benchmark.policy_api.policy_features import (
    camera_roles_for_embodiment,
    roles_for_embodiment,
)


DPVariant = Literal["dp_c", "dp_t"]
EmbodimentMode = Literal["single", "dual"]


@dataclass(frozen=True)
class DPPolicyConfig:
    """Runtime contract shared by the DP-C and DP-T clients."""

    variant: DPVariant
    host: str = "127.0.0.1"
    port: int = 8100
    embodiment: EmbodimentMode = "single"
    prediction_horizon: int = 30
    observation_horizon: int = 2
    execute_steps: int = 8
    camera_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.variant not in ("dp_c", "dp_t"):
            raise ValueError("variant must be 'dp_c' or 'dp_t'.")
        if not self.host.strip():
            raise ValueError("host must be non-empty.")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535.")
        if self.embodiment not in ("single", "dual"):
            raise ValueError("embodiment must be 'single' or 'dual'.")
        default_cameras = camera_roles_for_embodiment(self.embodiment)
        cameras = tuple(self.camera_roles or default_cameras)
        supported = (default_cameras,)
        if self.embodiment == "dual":
            supported += (("interaction", "wrist_left", "wrist_right"),)
        if cameras not in supported:
            raise ValueError(f"Unsupported {self.embodiment} camera_roles: {cameras}.")
        object.__setattr__(self, "camera_roles", cameras)
        if self.observation_horizon <= 0:
            raise ValueError("observation_horizon must be positive.")
        if self.prediction_horizon < self.observation_horizon:
            raise ValueError(
                "prediction_horizon must be at least observation_horizon."
            )
        if not 1 <= self.execute_steps <= self.available_action_steps:
            raise ValueError(
                "execute_steps must be in [1, available_action_steps]."
            )

    @property
    def roles(self) -> tuple[str, ...]:
        return roles_for_embodiment(self.embodiment)

    @property
    def state_dim(self) -> int:
        return 31 * len(self.roles)

    @property
    def action_dim(self) -> int:
        return 30 * len(self.roles)

    @property
    def available_action_steps(self) -> int:
        return self.prediction_horizon - self.observation_horizon + 1
