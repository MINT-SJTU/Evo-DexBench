from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias

import numpy as np

from dex_benchmark.policy_api.model_input import ModelInput


SemanticAction: TypeAlias = Mapping[str, Mapping[str, np.ndarray]]


class PolicyClient(Protocol):
    """Transport boundary implemented by a remote or local model client."""

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return one inference response containing an action chunk."""

    def close(self) -> None:
        """Release transport resources."""


class PolicySession(Protocol):
    """Per-environment policy state with an explicit episode lifecycle."""

    def reset(self) -> None:
        """Discard model- and action-chunk state for a new episode."""

    def act(self, model_input: ModelInput) -> SemanticAction:
        """Return one benchmark semantic action."""

    def close(self) -> None:
        """Release resources owned by this session."""


class BatchedPolicySession(Protocol):
    """Optional capability for one model call over multiple environments."""

    def reset(self) -> None:
        """Discard all per-environment history and action chunks."""

    def act_batch(self, model_inputs: Sequence[ModelInput]) -> Sequence[SemanticAction]:
        """Return one semantic action per input using native batched inference."""

    def close(self) -> None:
        """Release resources owned by this session."""


class EnvironmentAwarePolicySession(Protocol):
    """Optional capability for validating a policy against environment metadata."""

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        """Validate the static embodiment and sensor contract before rollout."""
