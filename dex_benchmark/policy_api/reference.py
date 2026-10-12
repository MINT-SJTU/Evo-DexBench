from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dex_benchmark.policy_api.contracts import (
    ActionSpec,
    ObservationSpec,
    PolicyCapabilities,
)
from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.session import SemanticAction


class ZeroPolicySession:
    """Dependency-free reference policy used to validate the public API."""

    capabilities = PolicyCapabilities(
        observation=ObservationSpec(state=True),
        native_batching=False,
        action_chunk_size=1,
    )

    def __init__(self) -> None:
        self._action: dict[str, dict[str, np.ndarray]] | None = None

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        action_spec = ActionSpec.from_metadata(metadata)
        action: dict[str, dict[str, np.ndarray]] = {}
        for role, raw_manipulator in action_spec.manipulators.items():
            delta_spec = raw_manipulator.get("end_effector_delta_pose")
            hand_spec = raw_manipulator.get("hand_joint_position")
            if not isinstance(delta_spec, Mapping) or not isinstance(
                hand_spec, Mapping
            ):
                raise ValueError(
                    f"Manipulator {role!r} does not expose the benchmark action fields."
                )
            hand_lower = np.asarray(hand_spec["lower"], dtype=np.float32)
            hand_upper = np.asarray(hand_spec["upper"], dtype=np.float32)
            action[str(role)] = {
                "end_effector_delta_pose": np.zeros(6, dtype=np.float32),
                "hand_joint_position": np.clip(
                    np.zeros_like(hand_lower), hand_lower, hand_upper
                ),
            }
        self._action = action

    def reset(self) -> None:
        return None

    def act(self, model_input: ModelInput) -> SemanticAction:
        del model_input
        if self._action is None:
            raise RuntimeError("bind_environment() must be called before act().")
        return {
            role: {name: value.copy() for name, value in fields.items()}
            for role, fields in self._action.items()
        }

    def capture_visual_observation_after_action(self) -> bool:
        return False

    def capture_lowdim_observation_after_action(self) -> bool:
        return True

    def requires_control_period_tactile(self) -> bool:
        return False

    def close(self) -> None:
        return None


def create_zero_policy(**kwargs: Any) -> ZeroPolicySession:
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"The zero policy accepts no options; got {unknown}.")
    return ZeroPolicySession()

