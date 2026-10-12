"""Conversion of Quest3 runtime objects into canonical diagnostics snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dex_benchmark.data.canonical.schema import QuestFrameSnapshot, QuestHandSnapshot
from dex_benchmark.teleoperation.quest3.action_builder import (
    Quest3HandRetargetingDiagnostics,
)
from dex_benchmark.teleoperation.quest3.stream import Quest3InputFrame


def _gesture_payload(gesture: Any) -> dict[str, bool | float] | None:
    if gesture is None:
        return None
    return {
        "pinch": bool(gesture.pinch),
        "squeeze": bool(gesture.squeeze),
        "tap": bool(gesture.tap),
        "pinch_value": float(gesture.pinch_value),
        "squeeze_value": float(gesture.squeeze_value),
        "tap_value": float(gesture.tap_value),
    }


def snapshot_from_quest_frame(
    frame: Quest3InputFrame,
    *,
    diagnostics: Mapping[str, Quest3HandRetargetingDiagnostics | None] | None = None,
    retarget_joint_names: Mapping[str, tuple[str, ...]] | None = None,
) -> QuestFrameSnapshot:
    """Copy a Quest frame and the latest retargeting diagnostics."""

    diagnostics = diagnostics or {}
    retarget_joint_names = retarget_joint_names or {}
    hands: dict[str, QuestHandSnapshot | None] = {}
    for side, hand in (("left", frame.left), ("right", frame.right)):
        if hand is None:
            hands[side] = None
            continue
        diagnostic = diagnostics.get(side)
        hands[side] = QuestHandSnapshot(
            joint_matrices=hand.joint_matrices,
            joint_pos_mano=None if diagnostic is None else diagnostic.joint_pos_mano,
            ref_value=None if diagnostic is None else diagnostic.ref_value,
            retarget_qpos=None if diagnostic is None else diagnostic.retarget_qpos,
            retarget_joint_names=tuple(retarget_joint_names.get(side, ())),
            gesture=_gesture_payload(hand.gesture),
        ).cpu_copy()
    return QuestFrameSnapshot(
        sequence=int(frame.sequence),
        source_timestamp_s=frame.source_timestamp_s,
        received_timestamp_s=float(frame.received_timestamp_s),
        left=hands["left"],
        right=hands["right"],
    ).cpu_copy()

