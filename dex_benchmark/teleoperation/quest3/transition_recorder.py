"""Canonical transition assembly for Quest3 collection.

This module intentionally does not infer semantic actions.  The caller must
provide a validated ``CanonicalAction`` containing whatever semantic/native
representations the selected action backend produced.
"""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any

from dex_benchmark.data.canonical.async_recorder import AsyncCanonicalRecorder
from dex_benchmark.data.canonical.schema import (
    CanonicalAction,
    CanonicalObservation,
    CanonicalTransition,
    QuestFrameSnapshot,
)


class CanonicalRecorderBackpressure(RuntimeError):
    """Raised when a complete transition cannot be queued."""


class Quest3TransitionAssembler:
    """Turn consecutive observation snapshots into recorder transitions."""

    def __init__(
        self,
        recorder: AsyncCanonicalRecorder,
        *,
        episode_id: str,
        initial_observation: CanonicalObservation | None = None,
    ) -> None:
        if not str(episode_id).strip():
            raise ValueError("episode_id must be non-empty.")
        self.recorder = recorder
        self.episode_id = str(episode_id)
        self._observation_t = initial_observation
        self._transition_id = 0

    @property
    def pending_observation(self) -> CanonicalObservation | None:
        return self._observation_t

    @property
    def transition_id(self) -> int:
        return self._transition_id

    def reset(self, observation: CanonicalObservation) -> None:
        """Start a new episode observation chain without submitting data."""

        self._observation_t = observation
        self._transition_id = 0

    def append(
        self,
        observation_tp1: CanonicalObservation,
        *,
        action: CanonicalAction,
        held_action: bool,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: Mapping[str, Any] | None = None,
        quest_frame: QuestFrameSnapshot | None = None,
        timestamps: Mapping[str, float | None] | None = None,
    ) -> CanonicalTransition:
        """Build and enqueue one transition, advancing the observation chain."""

        if self._observation_t is None:
            raise RuntimeError("A reset observation is required before append().")
        transition = CanonicalTransition(
            episode_id=self.episode_id,
            transition_id=self._transition_id,
            observation_t=self._observation_t,
            action=action,
            observation_tp1=observation_tp1,
            held_action=bool(held_action),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=dict(info or {}),
            quest_frame=quest_frame,
            timestamps={
                "control_tick_s": time.time(),
                **dict(timestamps or {}),
            },
        )
        if not self.recorder.submit(transition):
            raise CanonicalRecorderBackpressure(
                "Canonical recorder rejected a complete transition; abort episode."
            )
        self._observation_t = observation_tp1
        self._transition_id += 1
        return transition
