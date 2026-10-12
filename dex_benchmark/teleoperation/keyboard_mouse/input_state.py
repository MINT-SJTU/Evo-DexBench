from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


IdlePolicy = Literal["freeze_render", "hold_step"]
SceneViewerCameraUid = Literal["context_camera", "interaction_camera"]


def _zero_vec3() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


@dataclass(frozen=True)
class TeleopCommand:
    """Device-independent command consumed by teleoperation action builders."""

    pos_delta_root_m: np.ndarray = field(default_factory=_zero_vec3)
    rot_delta_root_rad: np.ndarray = field(default_factory=_zero_vec3)
    hand_preset: str | None = None
    hand_close_requested: bool = False
    switch_manipulator_requested: bool = False
    debug_toggle_requested: bool = False
    viewer_camera_uid_requested: SceneViewerCameraUid | None = None
    reset_requested: bool = False
    quit_requested: bool = False
    help_requested: bool = False

    def has_motion(self) -> bool:
        return bool(
            np.any(np.abs(self.pos_delta_root_m) > 1e-8)
            or np.any(np.abs(self.rot_delta_root_rad) > 1e-8)
        )

    def has_robot_action(self) -> bool:
        return (
            self.has_motion()
            or self.hand_preset is not None
            or self.hand_close_requested
        )
