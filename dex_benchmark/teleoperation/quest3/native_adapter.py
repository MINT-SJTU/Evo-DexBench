"""Native ManiSkill action backend for Quest 3 teleoperation."""

from __future__ import annotations

from typing import Any

from dex_benchmark.teleoperation.quest3.action_backend import Quest3ActionBackend
from dex_benchmark.teleoperation.quest3.action_builder import Quest3ActionPair


class LegacyNativeBackend:
    """Pass Quest-built native actions directly to ``env.step``.

    This is the compatibility path used by the 4a9b774 Quest3 controller.  It
    intentionally accepts both flat single-agent arrays and native multi-agent
    mappings; the environment remains the authority on their exact shape.
    """

    def __init__(self, env: Any) -> None:
        self.env = env

    def step(self, action: Any):
        if isinstance(action, Quest3ActionPair):
            action = action.native
        return self.env.step(action)
