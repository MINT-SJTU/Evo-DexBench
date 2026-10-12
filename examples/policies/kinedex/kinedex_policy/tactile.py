from __future__ import annotations

from collections.abc import Sequence

import numpy as np


TAXELS_PER_ROLE = 123
FINGERS_PER_ROLE = 5
MAX_TAXELS_PER_FINGER = 29
FINGER_NAMES = ("th", "ff", "mf", "rf", "lf")
FINGER_TAXEL_COUNTS = (29, 21, 18, 18, 14)
PALM_TAXEL_COUNT = 23
TACTILE_LAYOUT_ID = "shadow_hand_distributed_tactile_123"
TACTILE_LAYOUT_VERSION = 3
TACTILE_LAYOUT_HASH = "ccb1c3b07869b96a00c3ad51cd14d5aa49ab37c6ed667821eab78a91e44032da"
TACTILE_FORCE_UNIT = "N"
TACTILE_SENSOR_MODEL = "control_period_impulse_average"


def _role_slices() -> tuple[slice, ...]:
    starts = np.cumsum((0, *FINGER_TAXEL_COUNTS[:-1]))
    return tuple(
        slice(int(start), int(start + count))
        for start, count in zip(starts, FINGER_TAXEL_COUNTS, strict=True)
    )


FINGER_TAXEL_SLICES = _role_slices()


def validate_raw_tactile(value: np.ndarray, role_count: int) -> np.ndarray:
    tactile = np.asarray(value, dtype=np.float32)
    expected = TAXELS_PER_ROLE * role_count
    if tactile.shape[-1] != expected:
        raise ValueError(f"Tactile input must end in {expected}, got {tactile.shape}.")
    if not np.isfinite(tactile).all() or np.any(tactile < 0):
        raise ValueError("Tactile forces must be finite and non-negative.")
    return tactile


def pack_finger_taxels(value: np.ndarray, role_count: int) -> np.ndarray:
    """Pack scalar normal forces as [..., roles*5, 29, 1]."""

    tactile = validate_raw_tactile(value, role_count)
    result = np.zeros(
        (*tactile.shape[:-1], role_count * FINGERS_PER_ROLE, MAX_TAXELS_PER_FINGER, 1),
        dtype=np.float32,
    )
    for role_index in range(role_count):
        offset = role_index * TAXELS_PER_ROLE
        for finger_index, taxel_slice in enumerate(FINGER_TAXEL_SLICES):
            selected = tactile[..., offset + taxel_slice.start : offset + taxel_slice.stop]
            result[..., role_index * FINGERS_PER_ROLE + finger_index, : selected.shape[-1], 0] = selected
    return np.ascontiguousarray(result)


def aggregate_finger_forces(value: np.ndarray, role_count: int) -> np.ndarray:
    """Sum role-ordered finger taxels and discard palm taxels."""

    tactile = validate_raw_tactile(value, role_count)
    parts: list[np.ndarray] = []
    for role_index in range(role_count):
        offset = role_index * TAXELS_PER_ROLE
        for taxel_slice in FINGER_TAXEL_SLICES:
            parts.append(
                tactile[..., offset + taxel_slice.start : offset + taxel_slice.stop].sum(
                    axis=-1
                )
            )
    return np.ascontiguousarray(np.stack(parts, axis=-1), dtype=np.float32)


def interleave_motion_force(
    motion: np.ndarray, finger_force: np.ndarray, role_count: int
) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float32)
    finger_force = np.asarray(finger_force, dtype=np.float32)
    if motion.shape[:-1] != finger_force.shape[:-1]:
        raise ValueError("Motion and force leading dimensions must match.")
    if motion.shape[-1] != 30 * role_count or finger_force.shape[-1] != 5 * role_count:
        raise ValueError("Motion or force dimension does not match role count.")
    role_actions = []
    for role_index in range(role_count):
        role_actions.extend(
            (
                motion[..., role_index * 30 : (role_index + 1) * 30],
                finger_force[..., role_index * 5 : (role_index + 1) * 5],
            )
        )
    return np.ascontiguousarray(np.concatenate(role_actions, axis=-1), dtype=np.float32)


def split_motion_force(
    action: np.ndarray, role_count: int
) -> tuple[np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 35 * role_count:
        raise ValueError(f"KineDex action must end in {35 * role_count}.")
    motion = []
    force = []
    for role_index in range(role_count):
        offset = role_index * 35
        motion.append(action[..., offset : offset + 30])
        force.append(action[..., offset + 30 : offset + 35])
    return (
        np.ascontiguousarray(np.concatenate(motion, axis=-1), dtype=np.float32),
        np.ascontiguousarray(np.concatenate(force, axis=-1), dtype=np.float32),
    )


def expected_tactile_layout() -> dict[str, object]:
    return {
        "taxel_count": TAXELS_PER_ROLE,
        "layout_id": TACTILE_LAYOUT_ID,
        "layout_version": TACTILE_LAYOUT_VERSION,
        "layout_hash": TACTILE_LAYOUT_HASH,
        "force_unit": TACTILE_FORCE_UNIT,
        "sensor_model": TACTILE_SENSOR_MODEL,
    }
