"""Contact-aware semantic action filtering for Quest3 teleoperation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


ContactLevel = Literal["none", "light", "medium"]
DEFAULT_MAX_CONTACT_FORCE_N: float = 300.0


@dataclass(frozen=True)
class ContactActionFilterConfig:
    """Thresholds and speed scales for contact-aware control."""

    light_threshold_n: float = 5.0
    medium_threshold_n: float = 20.0
    max_force_n: float = DEFAULT_MAX_CONTACT_FORCE_N
    light_arm_normal_scale: float = 0.25
    light_finger_closure_scale: float = 0.25
    medium_finger_closure_scale: float = 0.10

    def __post_init__(self) -> None:
        numeric_values = (
            self.light_threshold_n,
            self.medium_threshold_n,
            self.max_force_n,
            self.light_arm_normal_scale,
            self.light_finger_closure_scale,
            self.medium_finger_closure_scale,
        )
        if not all(np.isfinite(float(value)) for value in numeric_values):
            raise ValueError("Contact filter settings must be finite numbers.")
        if not 0.0 < self.light_threshold_n < self.medium_threshold_n:
            raise ValueError(
                "Contact thresholds must satisfy 0 < light < medium."
            )
        if self.max_force_n <= self.medium_threshold_n:
            raise ValueError("max_force_n must be greater than medium_threshold_n.")
        for name in (
            "light_arm_normal_scale",
            "light_finger_closure_scale",
            "medium_finger_closure_scale",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class ContactActionFilterResult:
    semantic: dict[str, dict[str, np.ndarray]]
    diagnostics: dict[str, dict[str, Any]]


def _as_vector(value: Any, *, name: str, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise ValueError(f"{name} must be a vector, got {array.shape}.")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def _as_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [N, 3], got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array.copy()


def _quat_to_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    q = _as_vector(quaternion, name="root_rotation_world_wxyz", size=4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        raise ValueError("root_rotation_world_wxyz must be non-zero.")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _finger_index(joint_name: str) -> int | None:
    upper = str(joint_name).upper()
    for index, token in enumerate(("TH", "FF", "MF", "RF", "LF")):
        if f"_{token}J" in upper:
            return index
    return None


def _closing_sign(joint_name: str) -> float | None:
    upper = str(joint_name).upper()
    finger_index = _finger_index(upper)
    if finger_index is None:
        return None
    # FF/MF/RF J4 and LF J5 are spread/abduction joints. THJ4 is a thumb
    # flexion joint and must remain protected; THJ5 closes toward its negative
    # limit in this Shadow Hand model.
    if upper.endswith("J4") and any(token in upper for token in ("FFJ", "MFJ", "RFJ")):
        return None
    if "LFJ5" in upper:
        return None
    if "THJ5" in upper:
        return -1.0
    return 1.0


def _smoothstep(value: float) -> float:
    bounded = float(np.clip(value, 0.0, 1.0))
    return bounded * bounded * (3.0 - 2.0 * bounded)


def _contact_motion_scale(
    load_n: float,
    *,
    light_threshold_n: float,
    medium_threshold_n: float,
    light_scale: float,
    medium_scale: float,
) -> float:
    """Continuously blend free, light-contact, and medium-contact motion."""

    if load_n < light_threshold_n:
        return 1.0
    if load_n >= medium_threshold_n:
        return medium_scale
    midpoint_n = 0.5 * (light_threshold_n + medium_threshold_n)
    if load_n <= midpoint_n:
        blend = _smoothstep(
            (load_n - light_threshold_n) / (midpoint_n - light_threshold_n)
        )
        return (1.0 - blend) + blend * light_scale
    blend = _smoothstep(
        (load_n - midpoint_n) / (medium_threshold_n - midpoint_n)
    )
    return (1.0 - blend) * light_scale + blend * medium_scale


class ContactActionFilter:
    """Filter only command components that increase normal compression."""

    def __init__(self, config: ContactActionFilterConfig | None = None) -> None:
        self.config = config or ContactActionFilterConfig()

    def filter(
        self,
        semantic: Mapping[str, Mapping[str, Any]],
        *,
        observations: Mapping[str, Mapping[str, Any]],
        contact_states: Mapping[str, Mapping[str, Any]],
    ) -> ContactActionFilterResult:
        filtered: dict[str, dict[str, np.ndarray]] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for raw_role, raw_action in semantic.items():
            role = str(raw_role)
            if not isinstance(raw_action, Mapping):
                raise TypeError(f"semantic action for {role!r} must be a mapping.")
            state = contact_states.get(role)
            observation = observations.get(role)
            if state is None or observation is None:
                raise KeyError(f"Missing contact state or observation for role {role!r}.")
            filtered_action, role_diagnostics = self._filter_role(
                raw_action,
                observation=observation,
                state=state,
            )
            filtered[role] = filtered_action
            diagnostics[role] = role_diagnostics
        return ContactActionFilterResult(semantic=filtered, diagnostics=diagnostics)

    def _filter_role(
        self,
        action: Mapping[str, Any],
        *,
        observation: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        arm = _as_vector(
            action.get("end_effector_delta_pose"),
            name="end_effector_delta_pose",
            size=6,
        )
        hand = _as_vector(action.get("hand_joint_position"), name="hand_joint_position")
        all_qpos = _as_vector(observation.get("joint_position"), name="joint_position")
        hand_joint_indices = tuple(int(index) for index in state.get("hand_joint_indices", ()))
        qpos = (
            all_qpos[list(hand_joint_indices)]
            if hand_joint_indices
            else all_qpos
        )
        if qpos.shape != hand.shape:
            raise ValueError(
                "hand joint state and hand action shapes differ: "
                f"{qpos.shape} vs {hand.shape}."
            )
        parent_force = _as_matrix(state.get("parent_force_world"), name="parent_force_world")
        parent_normal_force = _as_matrix(
            state.get("parent_normal_force_world"),
            name="parent_normal_force_world",
        )
        if parent_normal_force.shape != parent_force.shape:
            raise ValueError(
                "parent normal and resultant force shapes differ: "
                f"{parent_normal_force.shape} vs {parent_force.shape}."
            )
        finger_force = _as_matrix(state.get("finger_force_world"), name="finger_force_world")
        taxel_force = _as_matrix(
            state.get("taxel_force_world", parent_force),
            name="taxel_force_world",
        )
        hand_force = _as_vector(state.get("hand_force_world"), name="hand_force_world", size=3)
        parent_norms = np.linalg.norm(parent_force, axis=1)
        parent_normal_norms = np.linalg.norm(parent_normal_force, axis=1)
        finger_norms = np.linalg.norm(finger_force, axis=1)
        taxel_norms = np.linalg.norm(taxel_force, axis=1)
        max_parent_force = float(parent_norms.max(initial=0.0))
        max_finger_force = float(finger_norms.max(initial=0.0))
        max_taxel_force = float(taxel_norms.max(initial=0.0))
        hand_resultant = float(np.linalg.norm(hand_force))
        load_for_level = max_parent_force
        level: ContactLevel
        if load_for_level < self.config.light_threshold_n:
            level = "none"
        elif load_for_level < self.config.medium_threshold_n:
            level = "light"
        else:
            level = "medium"
        hard_limit = max(
            max_parent_force,
            max_finger_force,
            max_taxel_force,
            hand_resultant,
        ) > self.config.max_force_n
        if hard_limit:
            level = "medium"

        filtered_arm = arm.copy()
        arm_scale = 1.0
        selected_parent_indices: list[int] = []
        reaction_direction_root = np.zeros(3, dtype=np.float32)
        if level != "none" and parent_normal_norms.size:
            root_rotation = _quat_to_matrix_wxyz(
                state.get("root_rotation_world_wxyz", (1.0, 0.0, 0.0, 0.0))
            )
            # [N, 3] @ [3, 3] -> [N, 3]. Rows are world vectors; multiplying
            # by R maps them into the robot root frame (v_root = R.T @ v_world).
            parent_normal_force_root = parent_normal_force @ root_rotation
            opposing = np.einsum(
                "ij,j->i",
                parent_normal_force_root,
                filtered_arm[:3],
            ) < 0.0
            active = parent_normal_norms >= self.config.light_threshold_n
            selected_parent_indices = np.flatnonzero(active & opposing).tolist()
            if selected_parent_indices:
                reaction_sum_root = parent_normal_force_root[
                    selected_parent_indices
                ].sum(axis=0)
                reaction_norm = float(np.linalg.norm(reaction_sum_root))
            else:
                reaction_norm = 0.0
            if reaction_norm > 1e-6:
                # The selected parent-link reactions oppose the current EE
                # motion. Their negative is the single inward/compression
                # direction used for action decomposition.
                inward = -reaction_sum_root / reaction_norm
                reaction_direction_root = (
                    reaction_sum_root / reaction_norm
                ).astype(np.float32)
                inward_command = max(float(np.dot(filtered_arm[:3], inward)), 0.0)
                arm_scale = _contact_motion_scale(
                    load_for_level,
                    light_threshold_n=self.config.light_threshold_n,
                    medium_threshold_n=self.config.medium_threshold_n,
                    light_scale=self.config.light_arm_normal_scale,
                    medium_scale=0.0,
                )
                filtered_arm[:3] -= inward * inward_command * (1.0 - arm_scale)

        filtered_hand = hand.copy()
        joint_names = tuple(str(name) for name in state.get("joint_names", ()))
        if joint_names and len(joint_names) != hand.size:
            raise ValueError("Contact state joint_names do not match hand action size.")
        finger_scale = (
            0.0
            if hard_limit
            else _contact_motion_scale(
                load_for_level,
                light_threshold_n=self.config.light_threshold_n,
                medium_threshold_n=self.config.medium_threshold_n,
                light_scale=self.config.light_finger_closure_scale,
                medium_scale=self.config.medium_finger_closure_scale,
            )
        )
        contacted_fingers = finger_norms >= self.config.light_threshold_n
        if level != "none":
            for joint_index, joint_name in enumerate(joint_names):
                finger_index = _finger_index(joint_name)
                closing_sign = _closing_sign(joint_name)
                if (
                    finger_index is None
                    or closing_sign is None
                    or finger_index >= contacted_fingers.size
                    or not contacted_fingers[finger_index]
                ):
                    continue
                target_delta = filtered_hand[joint_index] - qpos[joint_index]
                if closing_sign * target_delta > 0.0:
                    filtered_hand[joint_index] = qpos[joint_index] + (
                        finger_scale * target_delta
                    )

        return (
            {
                "end_effector_delta_pose": filtered_arm,
                "hand_joint_position": filtered_hand,
            },
            {
                "level": level,
                "max_parent_force_n": max_parent_force,
                "max_parent_normal_force_n": float(parent_normal_norms.max(initial=0.0)),
                "max_finger_force_n": max_finger_force,
                "max_taxel_force_n": max_taxel_force,
                "hand_resultant_force_n": hand_resultant,
                "parent_force_norms_n": parent_norms.tolist(),
                "finger_force_norms_n": finger_norms.tolist(),
                "contact_finger_indices": np.flatnonzero(contacted_fingers).tolist(),
                "selected_parent_indices": selected_parent_indices,
                "reaction_direction_root": reaction_direction_root.tolist(),
                "hard_limit_exceeded": hard_limit,
                "arm_normal_scale": arm_scale,
                "finger_closure_scale": (
                    1.0 if level == "none" else finger_scale
                ),
                "raw_arm_translation": arm[:3].tolist(),
                "filtered_arm_translation": filtered_arm[:3].tolist(),
            },
        )
