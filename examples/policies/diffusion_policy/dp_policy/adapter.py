from __future__ import annotations

import numpy as np

from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.policy_features import decode_semantic_action
from dex_benchmark.policy_api.session import SemanticAction
from dp_policy.config import DPPolicyConfig
from dex_benchmark.policy_api.observation import (
    compact_runtime_state,
    required_value,
    rgb_chw_float,
)


class DPAdapter:
    """Map Evo-DexBench observations/actions to the official image DP schema."""

    def __init__(self, config: DPPolicyConfig) -> None:
        self.config = config

    def observation(self, model_input: ModelInput) -> dict[str, np.ndarray]:
        observation = model_input["observation"]
        cameras = required_value(observation, "cameras")
        manipulators = required_value(observation, "manipulators")
        result: dict[str, np.ndarray] = {}
        for camera_role in self.config.camera_roles:
            result[f"{camera_role}_rgb"] = rgb_chw_float(
                required_value(cameras, camera_role, "rgb"),
                name=f"{camera_role}.rgb",
            )
        state = compact_runtime_state(manipulators, self.config.roles)
        if state.shape != (self.config.state_dim,):
            raise AssertionError(f"Unexpected DP state shape {state.shape}.")
        result["state"] = state
        return result

    def semantic_action(self, action: np.ndarray) -> SemanticAction:
        return decode_semantic_action(action, self.config.embodiment)
