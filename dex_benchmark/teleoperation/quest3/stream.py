from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import Condition
from typing import Any

import numpy as np

from dex_benchmark.teleoperation.vr_interface import VRInterface


QUEST3_HAND_JOINT_NAMES = (
    "wrist",
    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",
    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",
    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",
    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",
    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
)
QUEST3_HAND_JOINT_COUNT = len(QUEST3_HAND_JOINT_NAMES)
WEBXR_MATRIX_ELEMENT_COUNT = 16
WEBXR_HAND_ELEMENT_COUNT = QUEST3_HAND_JOINT_COUNT * WEBXR_MATRIX_ELEMENT_COUNT


@dataclass(frozen=True)
class Quest3GestureState:
    """Gesture values reported by Vuer without additional interpretation."""

    pinch: bool
    squeeze: bool
    tap: bool
    pinch_value: float
    squeeze_value: float
    tap_value: float


@dataclass(frozen=True)
class Quest3HandInput:
    """One tracked hand in the unmodified WebXR origin reference space."""

    # [25 * 16] column-major WebXR values -> [25, 4, 4] row-major matrices.
    joint_matrices: np.ndarray
    gesture: Quest3GestureState | None

    @property
    def wrist_matrix(self) -> np.ndarray:
        return self.joint_matrices[0]


@dataclass(frozen=True)
class Quest3InputFrame:
    """Latest raw hand-tracking sample received from a Quest 3 WebXR client."""

    sequence: int
    source_timestamp_s: float | None
    received_timestamp_s: float
    left: Quest3HandInput | None
    right: Quest3HandInput | None
    # Local wall-clock time after HAND_MOVE parsing and latest-value publish.
    # Vuer supplies received_timestamp_s at handler entry, so their difference
    # measures handler parsing/publish work rather than network transport.
    published_timestamp_s: float | None = None


class Quest3InputStream(VRInterface[Quest3InputFrame]):
    """Thread-safe latest-value stream for Quest 3 hand tracking.

    This class deliberately performs no coordinate conversion, filtering, IK, or
    dexterous-hand retargeting. A consumer that runs slower than WebXR always gets
    the newest frame instead of accumulating input latency in a queue.
    """

    def __init__(self, *, tracking_loss_grace_s: float = 5.0) -> None:
        if tracking_loss_grace_s < 0:
            raise ValueError("tracking_loss_grace_s must be non-negative.")
        self._tracking_loss_grace_s = tracking_loss_grace_s
        self._condition = Condition()
        self._latest_frame: Quest3InputFrame | None = None
        self._next_sequence = 1
        self._last_hand: dict[str, Quest3HandInput | None] = {"left": None, "right": None}
        self._last_hand_received_s: dict[str, float | None] = {"left": None, "right": None}

    def update(
        self,
        value: Mapping[str, Any],
        *,
        source_timestamp_s: float | None = None,
        received_timestamp_s: float | None = None,
    ) -> Quest3InputFrame:
        """Parse and publish one Vuer ``HAND_MOVE`` event value."""

        if not isinstance(value, Mapping):
            raise TypeError(f"HAND_MOVE value must be a mapping, got {type(value).__name__}.")

        source_timestamp_s = _optional_finite_float(
            source_timestamp_s,
            field_name="source_timestamp_s",
        )
        if received_timestamp_s is None:
            received_timestamp_s = time.time()
        received_timestamp_s = _finite_float(
            received_timestamp_s,
            field_name="received_timestamp_s",
        )

        left = _parse_hand(value, side="left")
        right = _parse_hand(value, side="right")

        with self._condition:
            left = self._coalesce_hand("left", left, received_timestamp_s)
            right = self._coalesce_hand("right", right, received_timestamp_s)
            published_timestamp_s = time.time()
            frame = Quest3InputFrame(
                sequence=self._next_sequence,
                source_timestamp_s=source_timestamp_s,
                received_timestamp_s=received_timestamp_s,
                left=left,
                right=right,
                published_timestamp_s=published_timestamp_s,
            )
            self._next_sequence += 1
            self._latest_frame = frame
            self._condition.notify_all()
        return frame

    def _coalesce_hand(
        self,
        side: str,
        hand: Quest3HandInput | None,
        received_timestamp_s: float,
    ) -> Quest3HandInput | None:
        if hand is not None:
            self._last_hand[side] = hand
            self._last_hand_received_s[side] = received_timestamp_s
            return hand
        last_received_s = self._last_hand_received_s[side]
        if (
            last_received_s is not None
            and received_timestamp_s - last_received_s <= self._tracking_loss_grace_s
        ):
            return self._last_hand[side]
        self._last_hand[side] = None
        self._last_hand_received_s[side] = None
        return None

    def poll(self) -> Quest3InputFrame | None:
        """Return the newest frame without blocking."""

        with self._condition:
            return self._latest_frame

    def wait_for_frame(
        self,
        *,
        after_sequence: int = 0,
        timeout_s: float | None = None,
    ) -> Quest3InputFrame | None:
        """Wait until a frame newer than ``after_sequence`` is available."""

        if after_sequence < 0:
            raise ValueError(f"after_sequence must be non-negative, got {after_sequence}.")
        if timeout_s is not None and timeout_s < 0:
            raise ValueError(f"timeout_s must be non-negative, got {timeout_s}.")

        with self._condition:
            ready = self._condition.wait_for(
                lambda: (
                    self._latest_frame is not None
                    and self._latest_frame.sequence > after_sequence
                ),
                timeout=timeout_s,
            )
            return self._latest_frame if ready else None


def parse_webxr_hand_matrices(raw_matrices: Any) -> np.ndarray:
    """Decode Vuer's 25 column-major WebXR hand matrices.

    Vuer/msgpack versions may expose a JavaScript ``Float32Array`` as a flat
    sequence, a byte buffer, or a sequence of byte chunks. All forms are kept
    numerically unchanged and returned as read-only ``float32`` matrices.
    """

    flat_values: np.ndarray
    extension_data = _msgpack_extension_data(raw_matrices)
    if extension_data is not None:
        flat_values = _decode_float32_bytes(extension_data)
    elif isinstance(raw_matrices, np.ndarray):
        try:
            flat_values = np.asarray(raw_matrices, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("WebXR hand matrices must contain numeric float32 values.") from exc
    elif isinstance(raw_matrices, (bytes, bytearray, memoryview)):
        flat_values = _decode_float32_bytes(raw_matrices)
    elif isinstance(raw_matrices, Sequence) and not isinstance(raw_matrices, str):
        if raw_matrices and all(
            isinstance(chunk, (bytes, bytearray, memoryview)) for chunk in raw_matrices
        ):
            packed_values = b"".join(bytes(chunk) for chunk in raw_matrices)
            flat_values = _decode_float32_bytes(packed_values)
        else:
            try:
                flat_values = np.asarray(raw_matrices, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "WebXR hand matrices must contain numeric float32 values."
                ) from exc
    else:
        raise TypeError(
            "WebXR hand matrices must be a float sequence or byte buffer, "
            f"got {type(raw_matrices).__name__}."
        )

    if flat_values.size != WEBXR_HAND_ELEMENT_COUNT:
        raise ValueError(
            "WebXR hand input must contain "
            f"{WEBXR_HAND_ELEMENT_COUNT} float32 values "
            f"({QUEST3_HAND_JOINT_COUNT} matrices), got {flat_values.size}."
        )
    if not np.isfinite(flat_values).all():
        raise ValueError("WebXR hand matrices contain NaN or infinite values.")

    # [25, 16] column-major values -> [25, 4, 4] conventional row-major matrices.
    joint_matrices = (
        flat_values.reshape(QUEST3_HAND_JOINT_COUNT, 4, 4)
        .transpose(0, 2, 1)
        .copy()
    )
    joint_matrices.setflags(write=False)
    return joint_matrices


def _parse_hand(value: Mapping[str, Any], *, side: str) -> Quest3HandInput | None:
    raw_matrices = value.get(side)
    if _is_untracked_hand_payload(raw_matrices):
        return None
    return Quest3HandInput(
        joint_matrices=parse_webxr_hand_matrices(raw_matrices),
        gesture=_parse_gesture_state(value.get(f"{side}State"), side=side),
    )


def _is_untracked_hand_payload(raw_matrices: Any) -> bool:
    """Return whether Vuer/WebXR reported this side as temporarily untracked."""

    extension_data = _msgpack_extension_data(raw_matrices)
    if (
        type(raw_matrices).__name__ == "ExtType"
        and getattr(raw_matrices, "code", None) == 0
        and extension_data is not None
        and bytes(extension_data) == b"\x00"
    ):
        return True

    if raw_matrices is None:
        return True
    if isinstance(raw_matrices, np.ndarray):
        if raw_matrices.size == 0:
            return True
        if raw_matrices.dtype == object:
            return any(value is None for value in raw_matrices.reshape(-1))
        return False
    if isinstance(raw_matrices, Sequence) and not isinstance(
        raw_matrices,
        (str, bytes, bytearray, memoryview),
    ):
        if len(raw_matrices) == 0:
            return True
        if all(isinstance(chunk, (bytes, bytearray, memoryview)) for chunk in raw_matrices):
            return False
        return any(value is None for value in raw_matrices)
    return False


def _parse_gesture_state(raw_state: Any, *, side: str) -> Quest3GestureState | None:
    if raw_state is None:
        return None
    if not isinstance(raw_state, Mapping):
        raise TypeError(
            f"{side}State must be a mapping when present, got {type(raw_state).__name__}."
        )
    return Quest3GestureState(
        pinch=bool(raw_state.get("pinch", False)),
        squeeze=bool(raw_state.get("squeeze", False)),
        tap=bool(raw_state.get("tap", False)),
        pinch_value=_finite_float(
            raw_state.get("pinchValue", 0.0),
            field_name=f"{side}State.pinchValue",
        ),
        squeeze_value=_finite_float(
            raw_state.get("squeezeValue", 0.0),
            field_name=f"{side}State.squeezeValue",
        ),
        tap_value=_finite_float(
            raw_state.get("tapValue", 0.0),
            field_name=f"{side}State.tapValue",
        ),
    )


def _decode_float32_bytes(raw_bytes: bytes | bytearray | memoryview) -> np.ndarray:
    if len(raw_bytes) % np.dtype("<f4").itemsize != 0:
        raise ValueError(
            "WebXR hand byte buffer length must be divisible by 4, "
            f"got {len(raw_bytes)} bytes."
        )
    return np.frombuffer(raw_bytes, dtype="<f4")


def _msgpack_extension_data(value: Any) -> bytes | bytearray | memoryview | None:
    """Return the byte payload from msgpack.ExtType without importing msgpack."""

    if (
        type(value).__name__ == "ExtType"
        and isinstance(getattr(value, "code", None), int)
        and isinstance(
            getattr(value, "data", None),
            (bytes, bytearray, memoryview),
        )
    ):
        return value.data
    return None


def _finite_float(value: Any, *, field_name: str) -> float:
    try:
        parsed_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(parsed_value):
        raise ValueError(f"{field_name} must be finite, got {parsed_value}.")
    return parsed_value


def _optional_finite_float(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    return _finite_float(value, field_name=field_name)
