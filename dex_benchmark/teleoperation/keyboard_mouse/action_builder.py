from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dex_benchmark.teleoperation.keyboard_mouse.hand_presets import (
    CLOSABLE_HAND_GESTURES,
    HandClosure,
    SHADOW_HAND_GESTURES,
    get_shadow_hand_preset,
)
from dex_benchmark.teleoperation.keyboard_mouse.input_state import TeleopCommand


@dataclass
class PandaShadowKeyboardSemanticActionBuilder:
    """Build physical semantic actions for one or two Panda + Shadow agents."""

    manipulator_roles: tuple[str, ...]
    hand_joint_names_by_role: dict[str, tuple[str, ...]]
    hand_target_qpos_rad_by_role: dict[str, np.ndarray]
    hand_target_lower_rad_by_role: dict[str, np.ndarray] | None = None
    hand_target_upper_rad_by_role: dict[str, np.ndarray] | None = None
    active_manipulator_role: str = "right"
    active_hand_preset_by_role: dict[str, str | None] | None = None
    active_hand_closure_by_role: dict[str, HandClosure] | None = None

    def __post_init__(self) -> None:
        if not self.manipulator_roles:
            raise ValueError("Keyboard action builder requires a manipulator.")
        if len(set(self.manipulator_roles)) != len(self.manipulator_roles):
            raise ValueError("Keyboard action builder roles must be unique.")
        if self.active_manipulator_role not in self.manipulator_roles:
            raise ValueError(
                f"Active role {self.active_manipulator_role!r} is not in "
                f"{self.manipulator_roles!r}."
            )
        expected_roles = set(self.manipulator_roles)
        if set(self.hand_joint_names_by_role) != expected_roles:
            raise ValueError(
                "hand_joint_names_by_role must contain exactly the configured "
                "manipulator roles."
            )
        if set(self.hand_target_qpos_rad_by_role) != expected_roles:
            raise ValueError(
                "hand_target_qpos_rad_by_role must contain exactly the configured "
                "manipulator roles."
            )
        if self.hand_target_lower_rad_by_role is None:
            self.hand_target_lower_rad_by_role = {
                role: np.full(
                    len(self.hand_joint_names_by_role[role]),
                    -np.inf,
                    dtype=np.float32,
                )
                for role in self.manipulator_roles
            }
        if self.hand_target_upper_rad_by_role is None:
            self.hand_target_upper_rad_by_role = {
                role: np.full(
                    len(self.hand_joint_names_by_role[role]),
                    np.inf,
                    dtype=np.float32,
                )
                for role in self.manipulator_roles
            }
        if set(self.hand_target_lower_rad_by_role) != expected_roles:
            raise ValueError(
                "hand_target_lower_rad_by_role must contain every manipulator role."
            )
        if set(self.hand_target_upper_rad_by_role) != expected_roles:
            raise ValueError(
                "hand_target_upper_rad_by_role must contain every manipulator role."
            )
        for role in self.manipulator_roles:
            joint_names = self.hand_joint_names_by_role[role]
            target_qpos = self.hand_target_qpos_rad_by_role[role]
            target_lower = self.hand_target_lower_rad_by_role[role]
            target_upper = self.hand_target_upper_rad_by_role[role]
            expected_shape = (len(joint_names),)
            if (
                not joint_names
                or target_qpos.shape != expected_shape
                or target_lower.shape != expected_shape
                or target_upper.shape != expected_shape
            ):
                raise ValueError(
                    f"Hand target and bounds for role {role!r} must have shape "
                    f"{expected_shape}."
                )
            if np.any(target_lower > target_upper):
                raise ValueError(
                    f"Hand target lower bounds exceed upper bounds for role {role!r}."
                )

        if self.active_hand_preset_by_role is None:
            self.active_hand_preset_by_role = {
                role: None for role in self.manipulator_roles
            }
        if self.active_hand_closure_by_role is None:
            self.active_hand_closure_by_role = {
                role: "loose" for role in self.manipulator_roles
            }
        if set(self.active_hand_preset_by_role) != expected_roles:
            raise ValueError(
                "active_hand_preset_by_role must contain every manipulator role."
            )
        if set(self.active_hand_closure_by_role) != expected_roles:
            raise ValueError(
                "active_hand_closure_by_role must contain every manipulator role."
            )

    @classmethod
    def from_env(cls, env: Any) -> "PandaShadowKeyboardSemanticActionBuilder":
        get_action_spec = env.get_wrapper_attr("get_action_spec")
        action_spec = get_action_spec()
        manipulator_specs = action_spec.get("manipulators", {})
        if not isinstance(manipulator_specs, dict) or not manipulator_specs:
            raise RuntimeError(
                "Keyboard teleoperation requires at least one manipulator in "
                "get_action_spec()."
            )

        manipulator_roles: list[str] = []
        hand_joint_names_by_role: dict[str, tuple[str, ...]] = {}
        hand_target_qpos_rad_by_role: dict[str, np.ndarray] = {}
        hand_target_lower_rad_by_role: dict[str, np.ndarray] = {}
        hand_target_upper_rad_by_role: dict[str, np.ndarray] = {}
        for raw_role, manipulator_spec in manipulator_specs.items():
            role = str(raw_role)
            hand_spec = manipulator_spec.get("hand_joint_position", {})
            hand_joint_names = tuple(
                str(name) for name in hand_spec.get("joint_names", ())
            )
            hand_shape = tuple(int(dim) for dim in hand_spec.get("shape", ()))
            if hand_shape != (len(hand_joint_names),) or not hand_joint_names:
                raise RuntimeError(
                    f"Keyboard teleoperation requires a non-empty canonical hand "
                    f"joint order for role {role!r}."
                )
            hand_lower = np.asarray(
                hand_spec.get("lower"),
                dtype=np.float32,
            )
            hand_upper = np.asarray(
                hand_spec.get("upper"),
                dtype=np.float32,
            )
            if (
                hand_lower.shape != hand_shape
                or hand_upper.shape != hand_shape
            ):
                raise RuntimeError(
                    f"Keyboard teleoperation requires canonical hand bounds for "
                    f"role {role!r}; expected {hand_shape}, got "
                    f"lower={hand_lower.shape}, upper={hand_upper.shape}."
                )
            manipulator_roles.append(role)
            hand_joint_names_by_role[role] = hand_joint_names
            hand_target_qpos_rad_by_role[role] = np.zeros(
                len(hand_joint_names),
                dtype=np.float32,
            )
            hand_target_lower_rad_by_role[role] = hand_lower
            hand_target_upper_rad_by_role[role] = hand_upper

        roles = tuple(manipulator_roles)
        active_role = "right" if "right" in roles else roles[0]
        return cls(
            manipulator_roles=roles,
            hand_joint_names_by_role=hand_joint_names_by_role,
            hand_target_qpos_rad_by_role=hand_target_qpos_rad_by_role,
            hand_target_lower_rad_by_role=hand_target_lower_rad_by_role,
            hand_target_upper_rad_by_role=hand_target_upper_rad_by_role,
            active_manipulator_role=active_role,
        )

    @property
    def manipulator_role(self) -> str:
        """Compatibility alias for the currently controlled role."""

        return self.active_manipulator_role

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return self.hand_joint_names_by_role[self.active_manipulator_role]

    @property
    def hand_target_qpos_rad(self) -> np.ndarray:
        return self.hand_target_qpos_rad_by_role[self.active_manipulator_role]

    @property
    def active_hand_preset(self) -> str | None:
        assert self.active_hand_preset_by_role is not None
        return self.active_hand_preset_by_role[self.active_manipulator_role]

    @property
    def active_hand_closure(self) -> HandClosure:
        assert self.active_hand_closure_by_role is not None
        return self.active_hand_closure_by_role[self.active_manipulator_role]

    def reset_targets(self) -> None:
        assert self.active_hand_preset_by_role is not None
        assert self.active_hand_closure_by_role is not None
        for role in self.manipulator_roles:
            self.hand_target_qpos_rad_by_role[role].fill(0.0)
            self.active_hand_preset_by_role[role] = None
            self.active_hand_closure_by_role[role] = "loose"
        self.active_manipulator_role = (
            "right" if "right" in self.manipulator_roles else self.manipulator_roles[0]
        )

    def switch_active_manipulator(self) -> str:
        """Cycle the active role; bimanual teleoperation starts on the right."""

        if len(self.manipulator_roles) == 1:
            return self.active_manipulator_role
        active_index = self.manipulator_roles.index(self.active_manipulator_role)
        next_index = (active_index + 1) % len(self.manipulator_roles)
        self.active_manipulator_role = self.manipulator_roles[next_index]
        return self.active_manipulator_role

    def build_semantic_action(
        self,
        command: TeleopCommand,
    ) -> dict[str, dict[str, np.ndarray]]:
        active_role = self.active_manipulator_role
        if command.hand_preset is not None:
            self.apply_hand_preset(
                command.hand_preset,
                role=active_role,
                closure="loose",
            )
        if command.hand_close_requested:
            self.close_active_hand_preset()

        active_ee_delta = np.concatenate(
            [
                np.asarray(command.pos_delta_root_m, dtype=np.float32),
                np.asarray(command.rot_delta_root_rad, dtype=np.float32),
            ]
        )
        semantic_action: dict[str, dict[str, np.ndarray]] = {}
        for role in self.manipulator_roles:
            semantic_action[role] = {
                "end_effector_delta_pose": (
                    active_ee_delta.copy()
                    if role == active_role
                    else np.zeros(6, dtype=np.float32)
                ),
                "hand_joint_position": self.hand_target_qpos_rad_by_role[
                    role
                ].copy(),
            }
        return semantic_action

    @staticmethod
    def _preset_for_role(
        preset_name: str,
        *,
        role: str,
        closure: HandClosure,
    ) -> dict[str, float]:
        right_hand_preset = get_shadow_hand_preset(
            preset_name,
            closure=closure,
        )
        if role != "left":
            return dict(right_hand_preset)
        return {
            (
                "lh_" + joint_name.removeprefix("rh_")
                if joint_name.startswith("rh_")
                else joint_name
            ): target_qpos
            for joint_name, target_qpos in right_hand_preset.items()
        }

    def apply_hand_preset(
        self,
        preset_name: str,
        *,
        role: str | None = None,
        closure: HandClosure = "loose",
    ) -> None:
        if preset_name not in SHADOW_HAND_GESTURES:
            available = ", ".join(sorted(SHADOW_HAND_GESTURES))
            raise KeyError(
                f"Unknown hand preset '{preset_name}'. Available presets: "
                f"{available}."
            )
        target_role = role or self.active_manipulator_role
        if target_role not in self.manipulator_roles:
            raise KeyError(f"Unknown manipulator role {target_role!r}.")
        preset = self._preset_for_role(
            preset_name,
            role=target_role,
            closure=closure,
        )
        joint_names = self.hand_joint_names_by_role[target_role]
        unknown_joint_names = sorted(set(preset) - set(joint_names))
        if unknown_joint_names:
            raise ValueError(
                f"Hand preset '{preset_name}' references unavailable joints for "
                f"role {target_role!r}: {unknown_joint_names}."
            )

        target_qpos = self.hand_target_qpos_rad_by_role[target_role]
        for joint_index, joint_name in enumerate(joint_names):
            if joint_name in preset:
                target_qpos[joint_index] = np.float32(preset[joint_name])
        assert self.active_hand_preset_by_role is not None
        assert self.active_hand_closure_by_role is not None
        self.active_hand_preset_by_role[target_role] = preset_name
        self.active_hand_closure_by_role[target_role] = (
            closure if preset_name in CLOSABLE_HAND_GESTURES else "loose"
        )

    def close_active_hand_preset(self) -> None:
        assert self.active_hand_preset_by_role is not None
        active_role = self.active_manipulator_role
        active_preset = self.active_hand_preset_by_role[active_role]
        if active_preset not in CLOSABLE_HAND_GESTURES:
            return
        self.apply_hand_preset(
            active_preset,
            role=active_role,
            closure="closed",
        )

    def sync_hand_targets_from_joint_qpos(
        self,
        hand_qpos_by_joint_name: dict[str, float],
        *,
        role: str | None = None,
    ) -> None:
        """Project current physical qpos into one valid canonical hand target."""

        target_role = role or self.active_manipulator_role
        if target_role not in self.manipulator_roles:
            raise KeyError(f"Unknown manipulator role {target_role!r}.")
        joint_names = self.hand_joint_names_by_role[target_role]
        target_qpos = self.hand_target_qpos_rad_by_role[target_role]
        assert self.hand_target_lower_rad_by_role is not None
        assert self.hand_target_upper_rad_by_role is not None
        target_lower = self.hand_target_lower_rad_by_role[target_role]
        target_upper = self.hand_target_upper_rad_by_role[target_role]
        for joint_index, joint_name in enumerate(joint_names):
            if joint_name in hand_qpos_by_joint_name:
                # A simulated joint can drift a few e-5 rad outside its action
                # target box. The synchronized control target must remain valid.
                target_qpos[joint_index] = np.clip(
                    np.float32(hand_qpos_by_joint_name[joint_name]),
                    target_lower[joint_index],
                    target_upper[joint_index],
                )


# Compatibility name for imports outside the keyboard teleoperation package.
PandaShadowKeyboardActionBuilder = PandaShadowKeyboardSemanticActionBuilder
