from __future__ import annotations

from typing import Any

import numpy as np

from dex_benchmark.teleoperation.keyboard_mouse.input_state import (
    SceneViewerCameraUid,
    TeleopCommand,
)
from dex_benchmark.teleoperation.keyboard_mouse.keymap import KeyboardTeleopKeymap


class SapienKeyboardReader:
    """Read keyboard state from a SAPIEN viewer window."""

    def __init__(
        self,
        viewer: Any,
        *,
        keymap: KeyboardTeleopKeymap | None = None,
        translation_step_m: float = 0.01,
        rotation_step_rad: float = 0.03,
    ) -> None:
        self.viewer = viewer
        self.keymap = keymap or KeyboardTeleopKeymap()
        self.translation_step_m = float(translation_step_m)
        self.rotation_step_rad = float(rotation_step_rad)
        self._previous_key_down: dict[str, bool] = {}

    def poll(self) -> TeleopCommand:
        pos_delta_root_m = np.zeros(3, dtype=np.float32)
        rot_delta_root_rad = np.zeros(3, dtype=np.float32)

        for key_name, (axis_index, direction) in self.keymap.position_keys.items():
            if self._key_down(key_name):
                pos_delta_root_m[axis_index] += np.float32(direction * self.translation_step_m)

        for key_name, (axis_index, direction) in self.keymap.rotation_keys.items():
            if self._key_down(key_name):
                rot_delta_root_rad[axis_index] += np.float32(direction * self.rotation_step_rad)

        hand_preset = None
        for key_name, preset_name in self.keymap.preset_keys.items():
            if self._key_press(key_name):
                hand_preset = preset_name
                break

        viewer_camera_uid_requested: SceneViewerCameraUid | None = None
        for key_name, camera_uid in self.keymap.viewer_camera_keys.items():
            if self._key_press(key_name):
                viewer_camera_uid_requested = camera_uid
                break

        return TeleopCommand(
            pos_delta_root_m=pos_delta_root_m,
            rot_delta_root_rad=rot_delta_root_rad,
            hand_preset=hand_preset,
            hand_close_requested=any(
                self._key_press(key_name) for key_name in self.keymap.close_keys
            ),
            switch_manipulator_requested=any(
                self._key_just_pressed(key_name)
                for key_name in self.keymap.switch_manipulator_keys
            ),
            debug_toggle_requested=any(
                self._key_just_pressed(key_name) for key_name in self.keymap.debug_toggle_keys
            ),
            viewer_camera_uid_requested=viewer_camera_uid_requested,
            reset_requested=any(self._key_press(key_name) for key_name in self.keymap.reset_keys),
            quit_requested=any(self._key_press(key_name) for key_name in self.keymap.quit_keys),
            help_requested=any(self._key_press(key_name) for key_name in self.keymap.help_keys),
        )

    def _key_down(self, key_name: str) -> bool:
        window = getattr(self.viewer, "window", None)
        if window is None or not hasattr(window, "key_down"):
            return False
        try:
            return bool(window.key_down(key_name))
        except Exception:
            return False

    def _key_press(self, key_name: str) -> bool:
        window = getattr(self.viewer, "window", None)
        if window is None:
            return False
        if hasattr(window, "key_press"):
            try:
                return bool(window.key_press(key_name))
            except Exception:
                return False
        return self._key_down(key_name)

    def _key_just_pressed(self, key_name: str) -> bool:
        is_down = self._key_down(key_name)
        was_down = self._previous_key_down.get(key_name, False)
        self._previous_key_down[key_name] = is_down
        return is_down and not was_down
