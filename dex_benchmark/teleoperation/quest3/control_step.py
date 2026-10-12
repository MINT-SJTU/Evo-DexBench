"""Immutable control-step event emitted by the Quest3 runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from dex_benchmark.data.canonical.schema import to_cpu_copy
from dex_benchmark.teleoperation.quest3.stream import Quest3InputFrame


@dataclass(frozen=True)
class Quest3ControlStep:
    """One executed control tick, independent of action representation."""

    control_step: int
    observation: Any
    action: Any
    held_action: bool
    reward: Any
    terminated: Any
    truncated: Any
    info: Mapping[str, Any]
    quest_frame: Quest3InputFrame | None
    timestamps: Mapping[str, float | None]

    def cpu_copy(self) -> "Quest3ControlStep":
        """Return an independent CPU-owned event for asynchronous consumers."""

        action = self.action.cpu_copy() if hasattr(self.action, "cpu_copy") else to_cpu_copy(self.action)

        return Quest3ControlStep(
            control_step=int(self.control_step),
            observation=to_cpu_copy(self.observation),
            action=action,
            held_action=bool(self.held_action),
            reward=to_cpu_copy(self.reward),
            terminated=to_cpu_copy(self.terminated),
            truncated=to_cpu_copy(self.truncated),
            info=to_cpu_copy(dict(self.info)),
            quest_frame=self.quest_frame,
            timestamps={
                str(name): (None if value is None else float(value))
                for name, value in self.timestamps.items()
            },
        )
