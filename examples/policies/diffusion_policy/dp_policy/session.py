from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.session import PolicyClient, SemanticAction
from dp_policy.adapter import DPAdapter
from dp_policy.client import DPWebsocketClient
from dp_policy.config import DPPolicyConfig


class DPPolicySession:
    """Observation-history and receding-horizon state for DP-C or DP-T."""

    def __init__(
        self,
        config: DPPolicyConfig,
        *,
        client: PolicyClient | None = None,
    ) -> None:
        self.config = config
        self.adapter = DPAdapter(config)
        self.client = client or DPWebsocketClient(config.host, config.port)
        self._histories: list[deque[dict[str, np.ndarray]]] = []
        self._actions: list[deque[np.ndarray]] = []

    def reset(self) -> None:
        self._histories.clear()
        self._actions.clear()

    def _ensure_batch(self, batch_size: int) -> None:
        if not self._histories:
            self._histories = [
                deque(maxlen=self.config.observation_horizon) for _ in range(batch_size)
            ]
            self._actions = [deque() for _ in range(batch_size)]
        elif len(self._histories) != batch_size:
            raise ValueError(
                "DP batch size changed within an episode; call reset() first."
            )

    def _append_observations(self, inputs: Sequence[ModelInput]) -> None:
        for history, model_input in zip(self._histories, inputs, strict=True):
            current = self.adapter.observation(model_input)
            if not history:
                for _ in range(self.config.observation_horizon - 1):
                    history.append(current)
            history.append(current)

    def _current_observation_is_required(self) -> bool:
        """Return whether the controller requested the current fresh frame."""

        if not self._histories or not self._actions or not any(self._actions):
            return True
        remaining = tuple(len(buffer) for buffer in self._actions)
        if len(set(remaining)) != 1:
            raise RuntimeError("DP action buffers became misaligned.")
        return remaining[0] < self.config.observation_horizon

    def _request_batch(self) -> None:
        if any(self._actions):
            raise RuntimeError("DP action buffers must be refilled synchronously.")
        keys = tuple(self._histories[0][0])
        observations = {
            key: np.stack(
                [
                    np.stack([frame[key] for frame in history], axis=0)
                    for history in self._histories
                ],
                axis=0,
            )
            for key in keys
        }
        response = self.client.infer(
            {
                "variant": self.config.variant,
                "embodiment": self.config.embodiment,
                "observations": observations,
            }
        )
        metadata = response.get("metadata", {})
        if metadata:
            expected = {
                "variant": self.config.variant,
                "embodiment": self.config.embodiment,
                "prediction_horizon": self.config.prediction_horizon,
                "observation_horizon": self.config.observation_horizon,
                "state_dim": self.config.state_dim,
                "action_dim": self.config.action_dim,
            }
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected.items()
                if metadata.get(key) != value
            }
            if mismatches:
                raise ValueError(f"DP checkpoint metadata mismatch: {mismatches}")
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        expected_shape = (
            len(self._histories),
            self.config.available_action_steps,
            self.config.action_dim,
        )
        if actions.shape != expected_shape:
            raise ValueError(
                f"DP server actions must have shape {expected_shape}, got "
                f"{actions.shape}."
            )
        for buffer, chunk in zip(self._actions, actions, strict=True):
            buffer.extend(chunk[: self.config.execute_steps])

    def act_batch(self, model_inputs: Sequence[ModelInput]) -> Sequence[SemanticAction]:
        inputs = tuple(model_inputs)
        if not inputs:
            raise ValueError("DP act_batch requires at least one input.")
        self._ensure_batch(len(inputs))
        # Intermediate action-chunk steps intentionally carry stale shared
        # observations. Do not repeatedly normalize and copy their RGB arrays.
        if self._current_observation_is_required():
            self._append_observations(inputs)
        if not any(self._actions):
            self._request_batch()
        if not all(self._actions):
            raise RuntimeError("DP action buffers became misaligned.")
        return tuple(
            self.adapter.semantic_action(buffer.popleft()) for buffer in self._actions
        )

    def act(self, model_input: ModelInput) -> SemanticAction:
        return self.act_batch((model_input,))[0]

    def capture_observation_after_action(self) -> bool:
        """Keep only the consecutive frames needed by the next DP request."""

        if not self._actions:
            return True
        remaining = tuple(len(buffer) for buffer in self._actions)
        if len(set(remaining)) != 1:
            raise RuntimeError("DP action buffers became misaligned.")
        return remaining[0] < self.config.observation_horizon

    def capture_visual_observation_after_action(self) -> bool:
        return self.capture_observation_after_action()

    def capture_lowdim_observation_after_action(self) -> bool:
        return self.capture_observation_after_action()

    def requires_control_period_tactile(self) -> bool:
        """DP-C/DP-T checkpoints consume RGB and compact state, not tactile."""

        return False

    def close(self) -> None:
        self.client.close()
