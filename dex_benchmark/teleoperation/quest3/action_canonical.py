"""Explicit conversion from Quest3 action pairs to canonical actions."""

from __future__ import annotations

from dex_benchmark.teleoperation.quest3.action_builder import Quest3ActionPair
from dex_benchmark.data.canonical.schema import CanonicalAction
from dex_benchmark.teleoperation.quest3.control_step import Quest3ControlStep


def canonical_action_from_pair(
    pair: Quest3ActionPair,
    *,
    action_id: int,
    timestamp_s: float,
) -> CanonicalAction:
    """Convert an already paired semantic/native action without inference."""

    if not isinstance(pair, Quest3ActionPair):
        raise TypeError("canonical_action_from_pair requires Quest3ActionPair.")
    return pair.to_canonical_action(action_id=action_id, timestamp_s=timestamp_s)


def canonical_action_from_control_step(step: Quest3ControlStep) -> CanonicalAction:
    """Build a canonical action from a pair carried by a control event."""

    if not isinstance(step.action, Quest3ActionPair):
        raise TypeError(
            "Quest3 control step does not contain Quest3ActionPair; refusing to "
            "label an opaque native action as semantic."
        )
    timestamp_s = step.timestamps.get("control_tick_s")
    if timestamp_s is None:
        timestamp_s = 0.0
    return canonical_action_from_pair(
        step.action,
        action_id=step.control_step,
        timestamp_s=float(timestamp_s),
    )
