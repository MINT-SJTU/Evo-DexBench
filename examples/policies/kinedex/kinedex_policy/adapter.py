from __future__ import annotations

import numpy as np

from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.policy_features import decode_semantic_action
from dex_benchmark.policy_api.session import SemanticAction
from kinedex_policy.config import KineDexPolicyConfig
from kinedex_policy.tactile import pack_finger_taxels
from dex_benchmark.policy_api.observation import (
    compact_runtime_state,
    required_value,
    rgb_chw_float,
    runtime_tactile_force,
)


class KineDexAdapter:
    """Map benchmark RGB/state/taxels to the official KineDex schema."""

    def __init__(self, config: KineDexPolicyConfig) -> None:
        self.config = config

    def images(self, model_input: ModelInput) -> dict[str, np.ndarray]:
        observation = model_input["observation"]
        cameras = required_value(observation, "cameras")
        return {
            f"{role}_rgb": rgb_chw_float(
                required_value(cameras, role, "rgb"), name=f"{role}.rgb"
            )
            for role in self.config.camera_roles
        }

    def lowdim_observation(
        self, model_input: ModelInput
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        observation = model_input["observation"]
        manipulators = required_value(observation, "manipulators")
        state = compact_runtime_state(manipulators, self.config.roles)
        if self.config.tactile_mode == "full":
            tactile = runtime_tactile_force(manipulators, self.config.roles)
            packed_tactile = pack_finger_taxels(tactile, self.config.role_count)
        else:
            tactile = None
            packed_tactile = np.zeros(
                (self.config.finger_count, 29, 1), dtype=np.float32
            )
        result = {
            "state": state,
            "finger_raw_pressure": packed_tactile,
        }
        return result, tactile

    def observation(
        self, model_input: ModelInput
    ) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
        result, tactile = self.lowdim_observation(model_input)
        result.update(self.images(model_input))
        return result, tactile

    def semantic_action(self, motion: np.ndarray) -> SemanticAction:
        return decode_semantic_action(motion, self.config.embodiment)
