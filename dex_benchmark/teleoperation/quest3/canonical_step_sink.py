"""Adapt Quest3 control-step events into canonical recorder transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import time
from typing import Any

from dex_benchmark.data.canonical.quest import snapshot_from_quest_frame
from dex_benchmark.data.canonical.schema import CanonicalAction, CanonicalObservation
from dex_benchmark.teleoperation.quest3.control_step import Quest3ControlStep
from dex_benchmark.teleoperation.quest3.observation_adapter import Quest3ObservationAdapter
from dex_benchmark.teleoperation.quest3.transition_recorder import Quest3TransitionAssembler


class Quest3CanonicalStepSink:
    """Convert control events into transitions without inferring action semantics."""

    def __init__(
        self,
        assembler: Quest3TransitionAssembler,
        observation_adapter: Quest3ObservationAdapter,
        *,
        action_factory: Callable[[Quest3ControlStep], CanonicalAction],
    ) -> None:
        if not callable(action_factory):
            raise TypeError("action_factory must be callable.")
        self.assembler = assembler
        self.observation_adapter = observation_adapter
        self.action_factory = action_factory

    def reset(
        self,
        observation: Any,
        *,
        observation_id: int = 0,
        timestamp_s: float | None = None,
    ) -> CanonicalObservation:
        """Register the initial observation for the next episode chain."""

        self.observation_adapter.validate_contract(observation)
        snapshot = self.observation_adapter.snapshot(
            observation,
            observation_id=observation_id,
            timestamp_s=time.time() if timestamp_s is None else timestamp_s,
            force_image=True,
        )
        self.assembler.reset(snapshot)
        return snapshot

    def __call__(self, step: Quest3ControlStep):
        """Build and submit one transition from a Runner control-step event."""

        observation = self.observation_adapter.snapshot(
            step.observation,
            observation_id=self._next_observation_id(step),
            timestamp_s=self._observation_timestamp(step),
        )
        action = self.action_factory(step)
        if not isinstance(action, CanonicalAction):
            raise TypeError(
                "action_factory must return CanonicalAction; action representation "
                "must be explicit."
            )
        quest_frame = (
            None
            if step.quest_frame is None
            else snapshot_from_quest_frame(step.quest_frame)
        )
        return self.assembler.append(
            observation,
            action=action,
            held_action=step.held_action,
            reward=step.reward,
            terminated=step.terminated,
            truncated=step.truncated,
            info=step.info,
            quest_frame=quest_frame,
            timestamps=step.timestamps,
        )

    def _next_observation_id(self, step: Quest3ControlStep) -> int:
        pending = self.assembler.pending_observation
        if pending is None:
            raise RuntimeError("Call reset() before consuming control-step events.")
        return pending.observation_id + 1

    @staticmethod
    def _observation_timestamp(step: Quest3ControlStep) -> float:
        for key in ("step_end_s", "control_tick_s"):
            value = step.timestamps.get(key)
            if value is not None:
                return float(value)
        return time.time()
