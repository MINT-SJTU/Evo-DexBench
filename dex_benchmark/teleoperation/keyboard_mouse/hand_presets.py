from __future__ import annotations

from collections.abc import Mapping
from typing import Literal


HandPreset = Mapping[str, float]
HandClosure = Literal["loose", "closed"]


WRIST_JOINT_NAMES: tuple[str, ...] = ("rh_WRJ2", "rh_WRJ1")
FINGER_JOINT_NAMES: tuple[str, ...] = (
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_FFJ2",
    "rh_FFJ1",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_MFJ2",
    "rh_MFJ1",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_RFJ2",
    "rh_RFJ1",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_LFJ2",
    "rh_LFJ1",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
)
ALL_HAND_JOINT_NAMES: tuple[str, ...] = (*WRIST_JOINT_NAMES, *FINGER_JOINT_NAMES)


# Limits are copied from panda_shadow_simple_tactile.urdf. Keeping them here makes
# preset edits fail early instead of silently sending invalid PD targets.
SHADOW_HAND_JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "rh_WRJ2": (-0.5235987755982988, 0.17453292519943295),
    "rh_WRJ1": (-0.6981317007977318, 0.4886921905584123),
    "rh_FFJ4": (-0.3490658503988659, 0.3490658503988659),
    "rh_FFJ3": (-0.2617993877991494, 1.5707963267948966),
    "rh_FFJ2": (0.0, 1.5707963267948966),
    "rh_FFJ1": (0.0, 1.5707963267948966),
    "rh_MFJ4": (-0.3490658503988659, 0.3490658503988659),
    "rh_MFJ3": (-0.2617993877991494, 1.5707963267948966),
    "rh_MFJ2": (0.0, 1.5707963267948966),
    "rh_MFJ1": (0.0, 1.5707963267948966),
    "rh_RFJ4": (-0.3490658503988659, 0.3490658503988659),
    "rh_RFJ3": (-0.2617993877991494, 1.5707963267948966),
    "rh_RFJ2": (0.0, 1.5707963267948966),
    "rh_RFJ1": (0.0, 1.5707963267948966),
    "rh_LFJ5": (0.0, 0.7853981633974483),
    "rh_LFJ4": (-0.3490658503988659, 0.3490658503988659),
    "rh_LFJ3": (-0.2617993877991494, 1.5707963267948966),
    "rh_LFJ2": (0.0, 1.5707963267948966),
    "rh_LFJ1": (0.0, 1.5707963267948966),
    "rh_THJ5": (-1.0471975511965976, 1.0471975511965976),
    "rh_THJ4": (0.0, 1.2217304763960306),
    "rh_THJ3": (-0.20943951023931953, 0.20943951023931953),
    "rh_THJ2": (-0.6981317007977318, 0.6981317007977318),
    "rh_THJ1": (-0.2617993877991494, 1.5707963267948966),
}


OPEN_PALM_PRESET: dict[str, float] = {joint_name: 0.0 for joint_name in ALL_HAND_JOINT_NAMES}


def _preset(overrides: Mapping[str, float]) -> dict[str, float]:
    preset = dict(OPEN_PALM_PRESET)
    for joint_name, target_angle_rad in overrides.items():
        if joint_name not in SHADOW_HAND_JOINT_LIMITS:
            raise KeyError(f"Unknown Shadow Hand joint in preset: {joint_name}")
        lower, upper = SHADOW_HAND_JOINT_LIMITS[joint_name]
        if not lower <= float(target_angle_rad) <= upper:
            raise ValueError(
                f"Preset value for {joint_name}={target_angle_rad} is outside "
                f"URDF limits [{lower}, {upper}]."
            )
        preset[joint_name] = float(target_angle_rad)
    return preset


def _finger_curl(
    *,
    prefix: str,
    mcp_flex_rad: float,
    pip_flex_rad: float,
    dip_flex_rad: float,
    spread_rad: float = 0.0,
) -> dict[str, float]:
    return {
        f"rh_{prefix}J4": spread_rad,
        f"rh_{prefix}J3": mcp_flex_rad,
        f"rh_{prefix}J2": pip_flex_rad,
        f"rh_{prefix}J1": dip_flex_rad,
    }


def _little_finger_curl(
    *,
    metacarpal_flex_rad: float,
    mcp_flex_rad: float,
    pip_flex_rad: float,
    dip_flex_rad: float,
    spread_rad: float = 0.0,
) -> dict[str, float]:
    return {
        "rh_LFJ5": metacarpal_flex_rad,
        "rh_LFJ4": spread_rad,
        "rh_LFJ3": mcp_flex_rad,
        "rh_LFJ2": pip_flex_rad,
        "rh_LFJ1": dip_flex_rad,
    }


SHADOW_HAND_GESTURES: dict[str, dict[HandClosure, dict[str, float]]] = {
    "open_palm": {
        "loose": _preset({}),
    },
    "power_grasp": {
        "loose": _preset(
            {
                "rh_WRJ1": -0.10,
                **_finger_curl(
                    prefix="FF",
                    mcp_flex_rad=0.72,
                    pip_flex_rad=0.75,
                    dip_flex_rad=0.30,
                ),
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=0.78,
                    pip_flex_rad=0.80,
                    dip_flex_rad=0.32,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=0.76,
                    pip_flex_rad=0.78,
                    dip_flex_rad=0.30,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.08,
                    mcp_flex_rad=0.68,
                    pip_flex_rad=0.78,
                    dip_flex_rad=0.29,
                ),
                "rh_THJ5": -0.40,
                "rh_THJ4": 0.70,
                "rh_THJ3": 0.06,
                "rh_THJ2": 0.36,
                "rh_THJ1": 0.32,
            }
        ),
        "closed": _preset(
            {
                "rh_WRJ1": -0.15,
                **_finger_curl(
                    prefix="FF",
                    mcp_flex_rad=1.26,
                    pip_flex_rad=1.30,
                    dip_flex_rad=0.62,
                ),
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=1.32,
                    pip_flex_rad=1.35,
                    dip_flex_rad=0.66,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=1.29,
                    pip_flex_rad=1.32,
                    dip_flex_rad=0.64,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.052,
                    mcp_flex_rad=1.16,
                    pip_flex_rad=1.36,
                    dip_flex_rad=0.55,
                ),
                "rh_THJ5": -0.13,
                "rh_THJ4": 1.02,
                "rh_THJ3": 0.08,
                "rh_THJ2": 0.58,
                "rh_THJ1": 0.62,
            }
        ),
    },
    "index_point": {
        "loose": _preset(
            {
                "rh_WRJ1": -0.08,
                "rh_FFJ3": 0.05,
                "rh_FFJ2": 0.03,
                "rh_FFJ1": 0.00,
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=1.36,
                    pip_flex_rad=1.29,
                    dip_flex_rad=0.70,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=1.26,
                    pip_flex_rad=1.28,
                    dip_flex_rad=0.65,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.12,
                    mcp_flex_rad=1.08,
                    pip_flex_rad=1.46,
                    dip_flex_rad=0.78,
                ),
                "rh_THJ5": -0.02,
                "rh_THJ4": 0.72,
                "rh_THJ3": 0.06,
                "rh_THJ2": 0.68,
                "rh_THJ1": 1.46,
            }
        ),
    },
    "precision_pinch": {
        "loose": _preset(
            {
                "rh_WRJ1": -0.10,
                "rh_FFJ4": 0.10,
                "rh_FFJ3": 0.78,
                "rh_FFJ2": 0.72,
                "rh_FFJ1": 0.30,
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=1.36,
                    pip_flex_rad=1.29,
                    dip_flex_rad=0.70,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=1.26,
                    pip_flex_rad=1.28,
                    dip_flex_rad=0.65,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.12,
                    mcp_flex_rad=1.08,
                    pip_flex_rad=1.46,
                    dip_flex_rad=0.78,
                ),
                "rh_THJ5": -0.92,
                "rh_THJ4": 1.08,
                "rh_THJ3": 0.10,
                "rh_THJ2": 0.60,
                "rh_THJ1": 0.68,
            }
        ),
        "closed": _preset(
            {
                "rh_WRJ1": -0.10,
                "rh_FFJ4": 0.10,
                "rh_FFJ3": 1.23,
                "rh_FFJ2": 0.72,
                "rh_FFJ1": 0.30,
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=1.36,
                    pip_flex_rad=1.29,
                    dip_flex_rad=0.70,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=1.26,
                    pip_flex_rad=1.28,
                    dip_flex_rad=0.65,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.12,
                    mcp_flex_rad=1.08,
                    pip_flex_rad=1.46,
                    dip_flex_rad=0.78,
                ),
                "rh_THJ5": -0.17,
                "rh_THJ4": 1.14,
                "rh_THJ3": 0.10,
                "rh_THJ2": 0.42,
                "rh_THJ1": 0.56,
            }
        ),
    },
    "lateral_clip": {
        "loose": _preset(
            {
                "rh_WRJ1": -0.08,
                **_finger_curl(
                    prefix="FF",
                    mcp_flex_rad=0.46,
                    pip_flex_rad=0.28,
                    dip_flex_rad=0.10,
                    spread_rad=0.08,
                ),
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=0.48,
                    pip_flex_rad=0.28,
                    dip_flex_rad=0.10,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=0.48,
                    pip_flex_rad=0.28,
                    dip_flex_rad=0.10,
                    spread_rad=-0.06,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.15,
                    mcp_flex_rad=0.42,
                    pip_flex_rad=0.24,
                    dip_flex_rad=0.08,
                    spread_rad=-0.08,
                ),
                "rh_THJ5": -0.95,
                "rh_THJ4": 0.95,
                "rh_THJ3": 0.08,
                "rh_THJ2": 0.48,
                "rh_THJ1": 0.42,
            }
        ),
        "closed": _preset(
            {
                "rh_WRJ1": -0.08,
                "rh_FFJ4": 0.05,
                "rh_FFJ3": 1.20,
                "rh_FFJ2": 0.38,
                "rh_FFJ1": 0.20,
                **_finger_curl(
                    prefix="MF",
                    mcp_flex_rad=0.90,
                    pip_flex_rad=0.92,
                    dip_flex_rad=0.44,
                ),
                **_finger_curl(
                    prefix="RF",
                    mcp_flex_rad=0.88,
                    pip_flex_rad=0.90,
                    dip_flex_rad=0.42,
                ),
                **_little_finger_curl(
                    metacarpal_flex_rad=0.08,
                    mcp_flex_rad=0.80,
                    pip_flex_rad=0.90,
                    dip_flex_rad=0.41,
                ),
                "rh_THJ5": -0.06,
                "rh_THJ4": 1.21,
                "rh_THJ3": 0.20,
                "rh_THJ2": 0.28,
                "rh_THJ1": 0.38,
            }
        ),
    },
}


CLOSABLE_HAND_GESTURES: frozenset[str] = frozenset(
    {"power_grasp", "precision_pinch", "lateral_clip"}
)


def get_shadow_hand_preset(
    gesture_name: str,
    *,
    closure: HandClosure = "loose",
) -> dict[str, float]:
    gesture = SHADOW_HAND_GESTURES.get(gesture_name)
    if gesture is None:
        available = ", ".join(sorted(SHADOW_HAND_GESTURES))
        raise KeyError(f"Unknown hand gesture '{gesture_name}'. Available gestures: {available}.")
    if closure == "closed" and gesture_name not in CLOSABLE_HAND_GESTURES:
        closure = "loose"
    if closure not in gesture:
        closure = "loose"
    return dict(gesture[closure])


SHADOW_HAND_PRESETS: dict[str, dict[str, float]] = {
    gesture_name: get_shadow_hand_preset(gesture_name, closure="loose")
    for gesture_name in SHADOW_HAND_GESTURES
}
