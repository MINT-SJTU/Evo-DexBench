from __future__ import annotations

from dataclasses import dataclass, field

from dex_benchmark.teleoperation.keyboard_mouse.input_state import (
    SceneViewerCameraUid,
)


AxisBinding = tuple[int, float]


@dataclass(frozen=True)
class KeyboardTeleopKeymap:
    """Default key bindings chosen to avoid ManiSkill viewer WASD camera controls."""

    position_keys: dict[str, AxisBinding] = field(
        default_factory=lambda: {
            "i": (0, 1.0),
            "k": (0, -1.0),
            "j": (1, 1.0),
            "l": (1, -1.0),
            "u": (2, 1.0),
            "o": (2, -1.0),
        }
    )
    rotation_keys: dict[str, AxisBinding] = field(
        default_factory=lambda: {
            "1": (0, 1.0),
            "2": (0, -1.0),
            "3": (1, 1.0),
            "4": (1, -1.0),
            "5": (2, 1.0),
            "6": (2, -1.0),
        }
    )
    preset_keys: dict[str, str] = field(
        default_factory=lambda: {
            "z": "open_palm",
            "x": "power_grasp",
            "c": "index_point",
            "v": "precision_pinch",
            "b": "lateral_clip",
        }
    )
    close_keys: tuple[str, ...] = ("m",)
    switch_manipulator_keys: tuple[str, ...] = ("n",)
    debug_toggle_keys: tuple[str, ...] = ("g",)
    viewer_camera_keys: dict[str, SceneViewerCameraUid] = field(
        default_factory=lambda: {
            "7": "context_camera",
            "8": "interaction_camera",
        }
    )
    reset_keys: tuple[str, ...] = ("r",)
    quit_keys: tuple[str, ...] = ("q", "escape")
    help_keys: tuple[str, ...] = ("h",)

    def help_text(self) -> str:
        return """Keyboard teleoperation controls:
  Move TCP: I/K root +/-X, J/L root +/-Y, U/O root +/-Z
  Rotate TCP: 1/2 roll +/-, 3/4 pitch +/-, 5/6 yaw +/-
  Hand gestures: Z open_palm, X power_grasp, C index_point, V precision_pinch, B lateral_clip
  Hand closure: M close/tighten current closable gesture
  Active arm: N switch left/right arm (bimanual tasks; starts on right)
  Debug: G toggle actual hand qpos change logging
  Camera preview: 7 context_camera, 8 interaction_camera, P print viewer pose
  Robot base: Arrow keys move in world XY, PageUp/PageDown move in world +/-Z, 0 print base world pose
  Session: R reset, Q/Escape quit, H help
"""
