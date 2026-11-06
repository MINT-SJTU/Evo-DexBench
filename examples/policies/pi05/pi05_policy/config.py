from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dex_benchmark.policy_api.policy_features import (
    SEMANTIC_ACTION_DIM_PER_ROLE,
    camera_roles_for_embodiment,
    roles_for_embodiment,
)


EmbodimentMode = Literal["single", "dual"]
# The 30D (single) or 60D (dual) physical action is zero-padded at the end to
# the 32D or 64D π0.5 action dimension.
MODEL_ACTION_DIM_PER_ROLE = 32


@dataclass(frozen=True)
class Pi05PolicyConfig:
    """Runtime contract between the benchmark and an OpenPI π0.5 server."""

    host: str = "127.0.0.1"
    port: int = 8000
    embodiment: EmbodimentMode = "single"
    action_horizon: int = 50
    execute_steps: int = 10

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must be non-empty.")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535.")
        if self.embodiment not in ("single", "dual"):
            raise ValueError("embodiment must be 'single' or 'dual'.")
        if self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive.")
        if not 1 <= self.execute_steps <= self.action_horizon:
            raise ValueError("execute_steps must be in [1, action_horizon].")

    @property
    def roles(self) -> tuple[str, ...]:
        return roles_for_embodiment(self.embodiment)

    @property
    def camera_roles(self) -> tuple[str, ...]:
        """Benchmark cameras in OpenPI's base, left-wrist, right-wrist slots."""

        return camera_roles_for_embodiment(self.embodiment)

    @property
    def model_action_dim(self) -> int:
        return MODEL_ACTION_DIM_PER_ROLE * len(self.roles)

    @property
    def semantic_action_dim(self) -> int:
        return SEMANTIC_ACTION_DIM_PER_ROLE * len(self.roles)
