from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from dex_benchmark.policy_api.contracts import ActionSpec
from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.policy_features import POLICY_CONTROL_FREQUENCY_HZ
from dex_benchmark.policy_api.session import PolicyClient, SemanticAction
from pi05_policy.adapter import Pi05Adapter
from pi05_policy.client import OpenPIWebsocketClient
from pi05_policy.config import Pi05PolicyConfig


class Pi05PolicySession:
    """Receding-horizon π0.5 session: one server request per action chunk."""

    def __init__(
        self,
        config: Pi05PolicyConfig,
        *,
        client: PolicyClient | None = None,
    ) -> None:
        self.config = config
        self.adapter = Pi05Adapter(config)
        self.client = client or OpenPIWebsocketClient(config.host, config.port)
        self._actions: deque[np.ndarray] = deque()

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        roles = tuple(ActionSpec.from_metadata(metadata).manipulators)
        if set(roles) != set(self.config.roles):
            raise ValueError(
                f"The {self.config.embodiment} π0.5 client expects manipulators "
                f"{self.config.roles}, but the task provides {roles}."
            )
        frequency = int(metadata.get("control_frequency_hz", 0))
        if frequency and frequency != POLICY_CONTROL_FREQUENCY_HZ:
            raise ValueError(
                f"π0.5 checkpoints use {POLICY_CONTROL_FREQUENCY_HZ} Hz control, got {frequency} Hz."
            )

    def reset(self) -> None:
        self._actions.clear()

    def act(self, model_input: ModelInput) -> SemanticAction:
        if not self._actions:
            response = self.client.infer(self.adapter.request(model_input))
            chunk = self.adapter.action_chunk(response)
            self._actions.extend(chunk[: self.config.execute_steps])
        return self.adapter.semantic_action(self._actions.popleft())

    def capture_visual_observation_after_action(self) -> bool:
        """π0.5 consumes only the current frame when it requests a new chunk."""

        return not self._actions

    def capture_lowdim_observation_after_action(self) -> bool:
        return not self._actions

    def requires_control_period_tactile(self) -> bool:
        """π0.5 consumes RGB and proprioceptive state, not tactile input."""

        return False

    def close(self) -> None:
        self.client.close()
