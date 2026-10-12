from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from kinedex_policy.tactile import aggregate_finger_forces


def _finger_index(joint_name: str) -> int | None:
    upper = joint_name.upper()
    for index, token in enumerate(("TH", "FF", "MF", "RF", "LF")):
        if f"_{token}J" in upper:
            return index
    return None


def _closing_sign(joint_name: str) -> float | None:
    upper = joint_name.upper()
    if _finger_index(upper) is None:
        return None
    if upper.endswith("J4") and any(token in upper for token in ("FFJ", "MFJ", "RFJ")):
        return None
    if "LFJ5" in upper:
        return None
    if "THJ5" in upper:
        return -1.0
    return 1.0


class FingerForceController:
    """Convert KineDex force targets into bounded Shadow Hand joint residuals."""

    def __init__(
        self,
        *,
        roles: Sequence[str],
        joint_names: Mapping[str, Sequence[str]],
        joint_lower: Mapping[str, np.ndarray],
        joint_upper: Mapping[str, np.ndarray],
        force_lower: np.ndarray,
        force_upper: np.ndarray,
        gain_rad_per_n: float,
        max_residual_rad: float,
    ) -> None:
        self.roles = tuple(roles)
        self.joint_names = {role: tuple(joint_names[role]) for role in self.roles}
        self.joint_lower = {role: np.asarray(joint_lower[role], dtype=np.float32) for role in self.roles}
        self.joint_upper = {role: np.asarray(joint_upper[role], dtype=np.float32) for role in self.roles}
        self.force_lower = np.asarray(force_lower, dtype=np.float32)
        self.force_upper = np.asarray(force_upper, dtype=np.float32)
        self.gain_rad_per_n = float(gain_rad_per_n)
        self.max_residual_rad = float(max_residual_rad)
        expected_force = 5 * len(self.roles)
        if self.force_lower.shape != (expected_force,) or self.force_upper.shape != (expected_force,):
            raise ValueError("Force bounds do not match KineDex embodiment.")
        for role in self.roles:
            if len(self.joint_names[role]) != 24:
                raise ValueError(f"{role} must provide 24 Shadow Hand joint names.")
            if self.joint_lower[role].shape != (24,) or self.joint_upper[role].shape != (24,):
                raise ValueError(f"{role} hand joint bounds must be 24D.")

    def apply(
        self, motion: np.ndarray, target_force: np.ndarray, raw_tactile: np.ndarray
    ) -> np.ndarray:
        motion = np.asarray(motion, dtype=np.float32).copy()
        target = np.asarray(target_force, dtype=np.float32)
        if motion.shape != (30 * len(self.roles),):
            raise ValueError("KineDex motion action has the wrong dimension.")
        if target.shape != self.force_lower.shape or not np.isfinite(target).all():
            raise ValueError("KineDex force target has the wrong dimension.")
        target = np.clip(target, self.force_lower, self.force_upper)
        current = aggregate_finger_forces(raw_tactile, len(self.roles))
        force_error = target - current
        role_actions = []
        for role_index, role in enumerate(self.roles):
            action = motion[role_index * 30 : (role_index + 1) * 30].copy()
            hand = action[6:]
            per_finger_joints: list[list[tuple[int, float]]] = [[] for _ in range(5)]
            for joint_index, name in enumerate(self.joint_names[role]):
                finger_index = _finger_index(name)
                closing_sign = _closing_sign(name)
                if finger_index is not None and closing_sign is not None:
                    per_finger_joints[finger_index].append((joint_index, closing_sign))
            for finger_index, joints in enumerate(per_finger_joints):
                if not joints:
                    continue
                error_index = role_index * 5 + finger_index
                total_residual = float(
                    np.clip(
                        self.gain_rad_per_n * force_error[error_index],
                        -self.max_residual_rad,
                        self.max_residual_rad,
                    )
                )
                for joint_index, closing_sign in joints:
                    hand[joint_index] += closing_sign * total_residual / len(joints)
            action[6:] = np.clip(
                hand, self.joint_lower[role], self.joint_upper[role]
            )
            role_actions.append(action)
        return np.ascontiguousarray(np.concatenate(role_actions), dtype=np.float32)
