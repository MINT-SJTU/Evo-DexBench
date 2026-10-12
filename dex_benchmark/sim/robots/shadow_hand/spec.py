"""Joint and link names shared by the benchmark Panda + Shadow Hand robots."""

from __future__ import annotations


FINGERTIP_NAMES: tuple[str, ...] = ("th", "ff", "mf", "rf", "lf")
RIGHT_HAND_TIP_LINK_NAMES: tuple[str, ...] = tuple(
    f"rh_{finger_name}tip" for finger_name in FINGERTIP_NAMES
)
TOUCH_LINK_NAMES: tuple[str, ...] = tuple(
    f"{finger_name}_tip_touch_link" for finger_name in FINGERTIP_NAMES
)

hand_joints_name = [
    "rh_WRJ2", "rh_WRJ1",
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_LFJ5", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_LFJ4", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2", "rh_LFJ3", "rh_THJ3",
    "rh_FFJ1", "rh_MFJ1", "rh_RFJ1", "rh_LFJ2", "rh_THJ2",
    "rh_LFJ1", "rh_THJ1",
]

distal_link_names = [
    "rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal",
]
middle_link_names = [
    "rh_thmiddle", "rh_ffmiddle", "rh_mfmiddle", "rh_rfmiddle", "rh_lfmiddle",
]
proximal_link_names = [
    "rh_thproximal", "rh_ffproximal", "rh_mfproximal", "rh_rfproximal", "rh_lfproximal",
]
dmp_links_names = distal_link_names + middle_link_names + proximal_link_names


__all__ = [
    "FINGERTIP_NAMES",
    "RIGHT_HAND_TIP_LINK_NAMES",
    "TOUCH_LINK_NAMES",
    "dmp_links_names",
    "hand_joints_name",
]
