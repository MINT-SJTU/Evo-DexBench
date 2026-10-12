"""Action execution backends for Quest 3 teleoperation.

Keeping action encoding here prevents the control runner from having to know
whether a task consumes a native ManiSkill action or a benchmark semantic
action.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias

import numpy as np
from dex_benchmark.teleoperation.quest3.action_builder import Quest3ActionPair


NativeAction: TypeAlias = np.ndarray | Mapping[str, Any]
SemanticAction: TypeAlias = Mapping[str, Any]


class Quest3ActionBackend(Protocol):
    """Minimal execution contract used by :class:`Quest3TeleopRunner`."""

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
        """Encode, execute, and return one environment transition."""
