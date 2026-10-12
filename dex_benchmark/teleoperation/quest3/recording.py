from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np

from dex_benchmark.teleoperation.quest3.action_builder import (
    Quest3HandRetargetingDiagnostics,
)
from dex_benchmark.teleoperation.quest3.stream import (
    Quest3GestureState,
    Quest3HandInput,
    Quest3InputFrame,
)


class Quest3JSONLRecorder:
    """Persist one complete retargeting sample per JSONL row."""

    SCHEMA_VERSION = 2

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path).expanduser().resolve()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.output_path.touch(exist_ok=False)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Quest 3 recording already exists: {self.output_path}"
            ) from exc

    def record(
        self,
        *,
        frame: Quest3InputFrame,
        side: str,
        hand: Quest3HandInput,
        diagnostics: Quest3HandRetargetingDiagnostics,
        retarget_joint_names: tuple[str, ...],
        action: np.ndarray | Mapping[str, np.ndarray],
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "sequence": int(frame.sequence),
            "source_timestamp_s": frame.source_timestamp_s,
            "received_timestamp_s": float(frame.received_timestamp_s),
            "side": side,
            "gesture": _gesture_payload(hand.gesture),
            "retarget_joint_names": list(retarget_joint_names),
            "joint_matrices": _float_list(
                hand.joint_matrices, expected_shape=(25, 4, 4)
            ),
            "joint_pos_mano": _float_list(
                diagnostics.joint_pos_mano, expected_shape=(21, 3)
            ),
            "retargeting_reference": _float_list(diagnostics.ref_value),
            "retarget_qpos": _float_list(diagnostics.retarget_qpos),
        }
        if isinstance(action, Mapping):
            expected_fields = {
                "end_effector_delta_pose",
                "hand_joint_position",
            }
            if set(action) != expected_fields:
                raise ValueError(
                    "Recorded semantic action must contain exactly "
                    "end_effector_delta_pose and hand_joint_position."
                )
            payload["semantic_action"] = {
                "end_effector_delta_pose": _float_list(
                    action["end_effector_delta_pose"], expected_shape=(6,)
                ),
                "hand_joint_position": _float_list(
                    action["hand_joint_position"]
                ),
            }
        else:
            payload["native_action"] = _float_list(action)
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self.output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(encoded)
            output_file.write("\n")


def _gesture_payload(
    gesture: Quest3GestureState | None,
) -> dict[str, bool | float] | None:
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


def _float_list(
    value: np.ndarray,
    *,
    expected_shape: tuple[int, ...] | None = None,
) -> list[Any]:
    array = np.asarray(value, dtype=np.float32)
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"Expected recorded array shape {expected_shape}, got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError("Recorded array contains NaN or infinite values.")
    return array.tolist()
