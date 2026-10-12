from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from dex_benchmark.policy_api.action_encoding import (
    encode_ee_delta_pose_action,
    encode_joint_position_action,
    make_flat_action,
)
from dex_benchmark.policy_api.control_spec import (
    EEDeltaPoseControlSpec,
    JointPositionControlSpec,
    component_spec_from_control_spec,
)
from dex_benchmark.teleoperation.retargeting.dex_retargeting import (
    DEFAULT_RETARGETING_URDF_ROOT,
    build_seq_retargeting,
    indices_by_joint_name,
)
from dex_benchmark.teleoperation.quest3.mapping import (
    Quest3WristMotionConfig,
    Quest3WristPoseMapper,
    quest3_hand_to_mano_joint_positions,
)
from dex_benchmark.teleoperation.quest3.stream import (
    Quest3HandInput,
    Quest3InputFrame,
)
from dex_benchmark.data.canonical.schema import CanonicalAction
from dex_benchmark.resources import teleoperation_config_path


DEFAULT_QUEST3_RETARGETING_CONFIGS = {
    "left": teleoperation_config_path("panda_shadow_hand_left_vector.yaml"),
    "right": teleoperation_config_path("panda_shadow_hand_right_vector.yaml"),
}


@dataclass(frozen=True)
class Quest3HandRetargetingDiagnostics:
    joint_pos_mano: np.ndarray
    ref_value: np.ndarray
    retarget_qpos: np.ndarray


@dataclass(frozen=True)
class Quest3ActionPair:
    """Physical semantic action and the native action derived from it."""

    semantic: Mapping[str, Mapping[str, np.ndarray]]
    native: np.ndarray | Mapping[str, np.ndarray]

    def cpu_copy(self) -> "Quest3ActionPair":
        return Quest3ActionPair(
            semantic={
                str(role): {
                    str(name): np.asarray(value, dtype=np.float32).copy()
                    for name, value in fields.items()
                }
                for role, fields in self.semantic.items()
            },
            native=_copy_action_value(self.native),
        )

    def to_canonical_action(self, *, action_id: int, timestamp_s: float) -> CanonicalAction:
        return CanonicalAction(
            action_id=action_id,
            timestamp_s=timestamp_s,
            semantic=self.semantic,
            native=self.native,
        )


def _copy_action_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_action_value(item) for key, item in value.items()}
    return np.asarray(value, dtype=np.float32).copy()


def _readonly_float32_copy(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    result.setflags(write=False)
    return result


@dataclass
class Quest3PandaShadowActionBuilder:
    """Build one Panda + Shadow Hand action from one tracked Quest hand."""

    side: str
    arm_spec: EEDeltaPoseControlSpec
    hand_3e1_spec: JointPositionControlSpec
    hand_1e1_spec: JointPositionControlSpec
    action_shape: tuple[int, ...]
    retargeting: Any
    retargeting_fixed_qpos: np.ndarray
    wrist_mapper: Quest3WristPoseMapper
    hand_3e1_retarget_indices: np.ndarray
    hand_1e1_retarget_indices: np.ndarray
    retargeting_joint_names: tuple[str, ...]
    canonical_hand_joint_names: tuple[str, ...]
    canonical_hand_lower: np.ndarray
    canonical_hand_upper: np.ndarray
    _filtered_hand_qpos: np.ndarray | None = field(default=None, init=False, repr=False)
    latest_diagnostics: Quest3HandRetargetingDiagnostics | None = field(
        default=None, init=False
    )

    @classmethod
    def from_action_spec(
        cls,
        action_spec: dict[str, Any],
        *,
        side: str,
        retargeting_config_path: str | Path | None = None,
        retargeting_urdf_root: str | Path | None = None,
        wrist_motion_config: Quest3WristMotionConfig | None = None,
        retargeting: Any | None = None,
    ) -> "Quest3PandaShadowActionBuilder":
        """Build a Quest adapter from the public semantic action schema.

        Older callers only have ``get_action_spec()`` and do not expose the
        low-level controller metadata.  Use conservative physical bounds in
        that compatibility path; environment-backed callers should prefer
        :meth:`from_env`, which uses the live controller bounds.
        """

        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        if action_spec.get("action_format") != "benchmark_physical_semantic":
            raise ValueError(
                "Quest teleoperation requires benchmark_physical_semantic actions."
            )
        manipulators = action_spec.get("manipulators")
        if not isinstance(manipulators, dict) or side not in manipulators:
            raise ValueError(f"Quest {side} hand is not available in action spec.")
        manipulator = manipulators[side]
        ee = manipulator.get("end_effector_delta_pose", {})
        hand = manipulator.get("hand_joint_position", {})
        joint_names = tuple(str(name) for name in hand.get("joint_names", ()))
        if tuple(ee.get("shape", ())) != (6,) or not joint_names:
            raise ValueError(
                f"Quest {side} hand requires a 6D EE action and canonical hand joints."
            )
        lower = np.asarray(hand.get("lower"), dtype=np.float32)
        upper = np.asarray(hand.get("upper"), dtype=np.float32)
        if lower.shape != (len(joint_names),) or upper.shape != lower.shape:
            raise ValueError(f"Quest {side} hand has invalid semantic joint bounds.")
        hand_components = {
            "arm": {
                "action_slice": (0, 6),
                "action_dim": 6,
                "controller_type": "PDEEPoseController",
                "normalize_action": False,
                "use_delta": True,
                "frame": "root_translation:root_aligned_body_rotation",
                "pos_lower": np.full(3, -0.1, dtype=np.float32),
                "pos_upper": np.full(3, 0.1, dtype=np.float32),
                "rot_lower": np.full(3, -0.1, dtype=np.float32),
                "rot_upper": np.full(3, 0.1, dtype=np.float32),
            },
            "hand_joint_3e1": {
                "action_slice": (6, 6 + len(joint_names)),
                "action_dim": len(joint_names),
                "controller_type": "PDJointPosController",
                "normalize_action": False,
                "use_delta": False,
                "joint_names": joint_names,
                "action_lower": lower,
                "action_upper": upper,
            },
            "hand_joint_1e1": {
                "action_slice": (6 + len(joint_names), 6 + len(joint_names)),
                "action_dim": 0,
                "controller_type": "PDJointPosController",
                "normalize_action": False,
                "use_delta": False,
                "joint_names": (),
                "action_lower": np.empty(0, dtype=np.float32),
                "action_upper": np.empty(0, dtype=np.float32),
            },
        }
        control_spec = {"components": hand_components}
        compatibility_wrist_config = (
            wrist_motion_config
            if wrist_motion_config is not None
            else Quest3WristMotionConfig()
        )
        return cls.from_control_spec(
            control_spec,
            action_shape=(6 + len(joint_names),),
            side=side,
            retargeting_config_path=retargeting_config_path,
            retargeting_urdf_root=retargeting_urdf_root,
            wrist_motion_config=compatibility_wrist_config,
            retargeting=retargeting,
            canonical_hand_joint_names=joint_names,
        )

    @classmethod
    def from_control_spec(
        cls,
        control_spec: dict[str, Any],
        *,
        action_shape: tuple[int, ...],
        side: str,
        retargeting_config_path: str | Path | None = None,
        retargeting_urdf_root: str | Path | None = None,
        wrist_motion_config: Quest3WristMotionConfig | None = None,
        retargeting: Any | None = None,
        canonical_hand_joint_names: tuple[str, ...] | None = None,
    ) -> "Quest3PandaShadowActionBuilder":
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        arm_spec = component_spec_from_control_spec(control_spec, "arm")
        hand_3e1_spec = component_spec_from_control_spec(control_spec, "hand_joint_3e1")
        hand_1e1_spec = component_spec_from_control_spec(control_spec, "hand_joint_1e1")
        if not isinstance(arm_spec, EEDeltaPoseControlSpec):
            raise TypeError(
                f"arm component is {arm_spec.control_kind}, expected EE_delta_pose."
            )
        if not isinstance(hand_3e1_spec, JointPositionControlSpec):
            raise TypeError(
                "hand_joint_3e1 component must use absolute joint-position control."
            )
        if not isinstance(hand_1e1_spec, JointPositionControlSpec):
            raise TypeError(
                "hand_joint_1e1 component must use absolute joint-position control."
            )

        if retargeting is None:
            retargeting = build_seq_retargeting(
                config_path=(
                    retargeting_config_path
                    or DEFAULT_QUEST3_RETARGETING_CONFIGS[side]
                ),
                urdf_root=retargeting_urdf_root or DEFAULT_RETARGETING_URDF_ROOT,
            )
        retargeting_joint_names = tuple(str(name) for name in retargeting.joint_names)
        expected_prefix = "lh_" if side == "left" else "rh_"
        wrong_side_joints = [
            name
            for name in hand_3e1_spec.joint_names + hand_1e1_spec.joint_names
            if not name.startswith(expected_prefix)
        ]
        if wrong_side_joints:
            raise ValueError(
                f"Quest {side} hand was assigned to incompatible robot joints: "
                f"{wrong_side_joints}."
            )
        if canonical_hand_joint_names is None:
            # Standalone callers predating get_action_spec() retain the
            # controller-component order. Environment-backed collection passes
            # the benchmark-declared canonical order explicitly below.
            canonical_hand_joint_names = tuple(
                str(name)
                for component in (hand_3e1_spec, hand_1e1_spec)
                for name in component.joint_names
            )
        canonical_hand_joint_names = tuple(canonical_hand_joint_names)
        if len(canonical_hand_joint_names) != len(set(canonical_hand_joint_names)):
            raise ValueError("Canonical hand joint names must be unique.")
        missing_canonical = sorted(
            set(canonical_hand_joint_names) - set(retargeting_joint_names)
        )
        if missing_canonical:
            raise ValueError(
                "Retargeting output is missing canonical hand joints: "
                + ", ".join(missing_canonical)
            )
        bounds_by_name: dict[str, tuple[float, float]] = {}
        for component in (hand_3e1_spec, hand_1e1_spec):
            if component.action_lower is None or component.action_upper is None:
                raise ValueError(
                    f"{component.name} must expose physical joint bounds."
                )
            for name, lower, upper in zip(
                component.joint_names, component.action_lower, component.action_upper
            ):
                if name in bounds_by_name:
                    raise ValueError(f"Duplicate hand joint bound for {name!r}.")
                bounds_by_name[str(name)] = (float(lower), float(upper))
        missing_bounds = sorted(set(canonical_hand_joint_names) - set(bounds_by_name))
        if missing_bounds:
            raise ValueError(
                "Controller bounds are missing canonical hand joints: "
                + ", ".join(missing_bounds)
            )

        optimizer = retargeting.optimizer
        fixed_indices = np.asarray(
            getattr(optimizer, "idx_pin2fixed", np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
        robot_q0 = np.asarray(
            getattr(getattr(optimizer, "robot", None), "q0", np.empty(0)),
            dtype=np.float32,
        )
        fixed_qpos = robot_q0[fixed_indices] if fixed_indices.size else np.empty(0)
        return cls(
            side=side,
            arm_spec=arm_spec,
            hand_3e1_spec=hand_3e1_spec,
            hand_1e1_spec=hand_1e1_spec,
            action_shape=action_shape,
            retargeting=retargeting,
            retargeting_fixed_qpos=fixed_qpos.astype(np.float32),
            wrist_mapper=Quest3WristPoseMapper(wrist_motion_config),
            hand_3e1_retarget_indices=indices_by_joint_name(
                source_joint_names=retargeting_joint_names,
                target_joint_names=hand_3e1_spec.joint_names,
            ),
            hand_1e1_retarget_indices=indices_by_joint_name(
                source_joint_names=retargeting_joint_names,
                target_joint_names=hand_1e1_spec.joint_names,
            ),
            retargeting_joint_names=retargeting_joint_names,
            canonical_hand_joint_names=canonical_hand_joint_names,
            canonical_hand_lower=np.asarray(
                [bounds_by_name[name][0] for name in canonical_hand_joint_names],
                dtype=np.float32,
            ),
            canonical_hand_upper=np.asarray(
                [bounds_by_name[name][1] for name in canonical_hand_joint_names],
                dtype=np.float32,
            ),
        )

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        side: str = "right",
        retargeting_config_path: str | Path | None = None,
        retargeting_urdf_root: str | Path | None = None,
        wrist_motion_config: Quest3WristMotionConfig | None = None,
    ) -> "Quest3PandaShadowActionBuilder":
        get_control_spec = env.get_wrapper_attr("get_control_spec")
        control_spec = get_control_spec()
        # New benchmark environments group the same legacy controller metadata
        # by manipulator role. Keep the old flat-action builder as the VR path.
        if isinstance(control_spec.get("manipulators"), dict):
            role_spec = control_spec["manipulators"].get(side)
            if role_spec is None:
                raise ValueError(
                    f"Quest {side} hand is not present in task embodiment; "
                    f"roles={sorted(control_spec['manipulators'])}."
                )
            control_spec = role_spec
        canonical_hand_joint_names: tuple[str, ...] | None = None
        get_action_spec = getattr(env, "get_wrapper_attr", None)
        if callable(get_action_spec):
            action_spec = get_action_spec("get_action_spec")()
            canonical_hand_joint_names = tuple(
                str(name)
                for name in action_spec["manipulators"][side][
                    "hand_joint_position"
                ]["joint_names"]
            )
        action_space = control_spec.get("action_space") or getattr(
            env, "action_space", None
        )
        action_shape = tuple(int(dim) for dim in getattr(action_space, "shape", ()))
        if not action_shape:
            raise ValueError("Quest teleoperation requires a flat action space.")
        return cls.from_control_spec(
            control_spec,
            action_shape=action_shape,
            side=side,
            retargeting_config_path=retargeting_config_path,
            retargeting_urdf_root=retargeting_urdf_root,
            wrist_motion_config=wrist_motion_config,
            canonical_hand_joint_names=canonical_hand_joint_names,
        )

    def reset(self) -> None:
        self.wrist_mapper.reset()
        self._filtered_hand_qpos = None
        self.latest_diagnostics = None
        reset = getattr(self.retargeting, "reset", None)
        if callable(reset):
            reset()
        self._reset_retargeting_filter()

    def reset_wrist(self) -> None:
        self.latest_diagnostics = None
        self.wrist_mapper.reset()
        self._reset_retargeting_filter()

    def recover_wrist_tracking_error(
        self,
        *,
        position_error_root_m: np.ndarray,
        rotation_error_root: np.ndarray,
    ) -> None:
        """Retry an IK-rejected wrist target without redefining its neutral pose."""

        self.wrist_mapper.rewind_commanded_target(
            position_error_root_m=position_error_root_m,
            rotation_error_root=rotation_error_root,
        )

    def seed_hand_target_from_robot_state(
        self,
        *,
        joint_names: tuple[str, ...],
        joint_qpos: np.ndarray,
    ) -> None:
        """Start hand target filtering from the simulator's current pose."""

        names = tuple(str(name) for name in joint_names)
        qpos = np.asarray(joint_qpos, dtype=np.float32)
        if qpos.shape != (len(names),):
            raise ValueError(
                "Robot hand target seed shape does not match joint names: "
                f"qpos={qpos.shape}, joint_names={len(names)}."
            )
        if not np.isfinite(qpos).all():
            raise ValueError("Robot hand target seed must contain only finite values.")
        if len(names) != len(set(names)):
            raise ValueError("Robot hand target seed joint names must be unique.")
        index_by_name = {name: index for index, name in enumerate(names)}
        missing = [
            name for name in self.retargeting_joint_names if name not in index_by_name
        ]
        if missing:
            raise ValueError(
                "Robot state is missing retargeting joints: " + ", ".join(missing)
            )
        self._filtered_hand_qpos = np.asarray(
            [qpos[index_by_name[name]] for name in self.retargeting_joint_names],
            dtype=np.float32,
        )

    def _reset_retargeting_filter(self) -> None:
        retargeting_filter = getattr(self.retargeting, "filter", None)
        reset = getattr(retargeting_filter, "reset", None)
        if callable(reset):
            reset()

    def build_action(self, hand: Quest3HandInput) -> dict[str, np.ndarray]:
        """Build the public semantic action for one manipulator.

        ``build_action_pair`` remains the execution path because it carries
        both semantic and native forms.  Returning semantic fields here keeps
        older integrations source-compatible and makes the side boundary
        explicit for bimanual callers.
        """

        return dict(self.build_action_pair(hand).semantic[self.side])

    def build_action_pair(self, hand: Quest3HandInput) -> Quest3ActionPair:
        """Build physical semantic and native actions from one Quest hand."""
        joint_pos_mano = quest3_hand_to_mano_joint_positions(hand, side=self.side)
        retarget_qpos, ref_value = self._retarget_hand_qpos(joint_pos_mano)
        retarget_qpos = self._filter_hand_target(retarget_qpos)
        position_delta_m, rotation_delta_rad = self.wrist_mapper.update(
            hand.wrist_matrix
        )

        arm_action = encode_ee_delta_pose_action(
            self.arm_spec,
            pos_delta_root_m=position_delta_m,
            rot_delta_root_rad=rotation_delta_rad,
        )
        bounded_retarget_qpos = retarget_qpos.copy()
        for component in (self.hand_3e1_spec, self.hand_1e1_spec):
            component_indices = np.asarray(
                [self.retargeting_joint_names.index(name) for name in component.joint_names],
                dtype=np.int64,
            )
            bounded_retarget_qpos[component_indices] = np.clip(
                retarget_qpos[component_indices],
                component.action_lower,
                component.action_upper,
            )
        hand_3e1_action = encode_joint_position_action(
            self.hand_3e1_spec,
            target_qpos=bounded_retarget_qpos[self.hand_3e1_retarget_indices],
        )
        hand_1e1_action = encode_joint_position_action(
            self.hand_1e1_spec,
            target_qpos=bounded_retarget_qpos[self.hand_1e1_retarget_indices],
        )
        native_action = make_flat_action(
            action_shape=self.action_shape,
            component_actions=[
                (self.arm_spec, arm_action),
                (self.hand_3e1_spec, hand_3e1_action),
                (self.hand_1e1_spec, hand_1e1_action),
            ],
        )
        self.latest_diagnostics = Quest3HandRetargetingDiagnostics(
            joint_pos_mano=_readonly_float32_copy(joint_pos_mano),
            ref_value=_readonly_float32_copy(ref_value),
            retarget_qpos=_readonly_float32_copy(retarget_qpos),
        )
        canonical_hand_joint_indices = np.asarray(
            [self.retargeting_joint_names.index(name) for name in self.canonical_hand_joint_names],
            dtype=np.int64,
        )
        semantic_hand = np.clip(
            retarget_qpos[canonical_hand_joint_indices],
            self.canonical_hand_lower,
            self.canonical_hand_upper,
        )
        semantic_action = {
            self.side: {
                "end_effector_delta_pose": _readonly_float32_copy(
                    np.concatenate((position_delta_m, rotation_delta_rad))
                ),
                "hand_joint_position": _readonly_float32_copy(
                    semantic_hand
                ),
            }
        }
        return Quest3ActionPair(semantic=semantic_action, native=native_action)

    def rebuild_action_pair_from_semantic(
        self,
        semantic: Mapping[str, Mapping[str, np.ndarray]],
    ) -> Quest3ActionPair:
        """Re-encode a filtered semantic action into the native action space."""

        if self.side not in semantic:
            raise KeyError(f"Filtered semantic action is missing role {self.side!r}.")
        fields = semantic[self.side]
        arm_value = np.asarray(fields["end_effector_delta_pose"], dtype=np.float32)
        hand_value = np.asarray(fields["hand_joint_position"], dtype=np.float32)
        if arm_value.shape != (6,):
            raise ValueError(f"Filtered arm action must have shape (6,), got {arm_value.shape}.")
        if hand_value.shape != (len(self.canonical_hand_joint_names),):
            raise ValueError(
                "Filtered hand action shape does not match canonical joints: "
                f"{hand_value.shape} vs {len(self.canonical_hand_joint_names)}."
            )
        canonical_index = {
            name: index for index, name in enumerate(self.canonical_hand_joint_names)
        }
        bounded_hand = np.clip(hand_value, self.canonical_hand_lower, self.canonical_hand_upper)
        arm_action = encode_ee_delta_pose_action(
            self.arm_spec,
            pos_delta_root_m=arm_value[:3],
            rot_delta_root_rad=arm_value[3:6],
        )
        component_actions: list[tuple[Any, np.ndarray]] = [(self.arm_spec, arm_action)]
        for component in (self.hand_3e1_spec, self.hand_1e1_spec):
            indices = np.asarray(
                [canonical_index[name] for name in component.joint_names],
                dtype=np.int64,
            )
            component_action = encode_joint_position_action(
                component,
                target_qpos=bounded_hand[indices],
            )
            component_actions.append((component, component_action))
        native_action = make_flat_action(
            action_shape=self.action_shape,
            component_actions=component_actions,
        )
        copied_semantic = {
            self.side: {
                "end_effector_delta_pose": arm_value.copy(),
                "hand_joint_position": bounded_hand.copy(),
            }
        }
        return Quest3ActionPair(semantic=copied_semantic, native=native_action)

    def _filter_hand_target(self, retarget_qpos: np.ndarray) -> np.ndarray:
        """Limit and smooth absolute hand targets across Quest frames.

        A tracking reacquisition can produce a valid but discontinuous hand
        pose. Filtering the retargeted qpos (rather than raw landmarks) keeps
        the protection in the robot's actual joint space and preserves the
        existing retargeting and controller limits.
        """

        target = np.asarray(retarget_qpos, dtype=np.float32)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise ValueError("Retargeted hand target must be a finite 1-D array.")
        if self._filtered_hand_qpos is None:
            self._filtered_hand_qpos = target.copy()
            return self._filtered_hand_qpos.copy()

        if self._filtered_hand_qpos.shape != target.shape:
            raise ValueError(
                "Retargeted hand target shape changed from "
                f"{self._filtered_hand_qpos.shape} to {target.shape}."
            )
        max_delta = float(self.wrist_mapper.config.max_hand_joint_delta)
        bounded_target = self._filtered_hand_qpos + np.clip(
            target - self._filtered_hand_qpos,
            -max_delta,
            max_delta,
        )
        alpha = float(self.wrist_mapper.config.hand_target_smoothing_alpha)
        self._filtered_hand_qpos += alpha * (
            bounded_target - self._filtered_hand_qpos
        )
        return self._filtered_hand_qpos.copy()

    def build_rotation_followup_action(
        self,
        action: np.ndarray | Mapping[str, np.ndarray],
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Hold hand targets while advancing toward the latest wrist rotation."""
        if isinstance(action, Mapping):
            followup_semantic = {
                str(name): np.asarray(value, dtype=np.float32).copy()
                for name, value in action.items()
            }
            rotation_delta_rad = self.wrist_mapper.continue_rotation()
            followup_semantic["end_effector_delta_pose"] = np.concatenate(
                (np.zeros(3, dtype=np.float32), rotation_delta_rad)
            ).astype(np.float32)
            return followup_semantic
        followup_action = np.asarray(action, dtype=np.float32).copy()
        rotation_delta_rad = self.wrist_mapper.continue_rotation()
        arm_action = encode_ee_delta_pose_action(
            self.arm_spec,
            pos_delta_root_m=np.zeros(3, dtype=np.float32),
            rot_delta_root_rad=rotation_delta_rad,
        )
        followup_action[self.arm_spec.action_slice] = arm_action
        return followup_action

    def build_rotation_followup_action_pair(
        self, pair: Quest3ActionPair
    ) -> Quest3ActionPair:
        """Advance wrist rotation while keeping semantic/native forms aligned."""

        rotation_delta_rad = self.wrist_mapper.continue_rotation()
        native = np.asarray(pair.native, dtype=np.float32).copy()
        native[self.arm_spec.action_slice] = encode_ee_delta_pose_action(
            self.arm_spec,
            pos_delta_root_m=np.zeros(3, dtype=np.float32),
            rot_delta_root_rad=rotation_delta_rad,
        )
        semantic = {
            str(role): {str(name): np.asarray(value, dtype=np.float32).copy() for name, value in fields.items()}
            for role, fields in pair.semantic.items()
        }
        semantic[self.side]["end_effector_delta_pose"] = _readonly_float32_copy(
            np.concatenate((np.zeros(3, dtype=np.float32), rotation_delta_rad))
        )
        return Quest3ActionPair(semantic=semantic, native=native)

    def _retarget_hand_qpos(
        self, joint_pos_mano: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if joint_pos_mano.shape != (21, 3):
            raise ValueError(
                f"Expected canonical hand positions shape (21, 3), got {joint_pos_mano.shape}."
            )
        optimizer = self.retargeting.optimizer
        human_indices = np.asarray(optimizer.target_link_human_indices, dtype=np.int64)
        retargeting_type = str(optimizer.retargeting_type).upper()
        if retargeting_type == "POSITION":
            ref_value = joint_pos_mano[human_indices, :]
        else:
            if human_indices.shape[0] != 2:
                raise ValueError(
                    "Vector retargeting human indices must have shape (2, N), "
                    f"got {human_indices.shape}."
                )
            # [N, 3] - [N, 3] -> [N, 3]
            ref_value = (
                joint_pos_mano[human_indices[1], :]
                - joint_pos_mano[human_indices[0], :]
            )

        retarget_qpos = np.asarray(
            self.retargeting.retarget(ref_value, fixed_qpos=self.retargeting_fixed_qpos),
            dtype=np.float32,
        )
        expected_shape = (len(tuple(self.retargeting.joint_names)),)
        if retarget_qpos.shape != expected_shape:
            raise ValueError(
                f"dex_retargeting returned qpos shape {retarget_qpos.shape}, "
                f"expected {expected_shape}."
            )
        return retarget_qpos, ref_value


@dataclass
class Quest3DualPandaShadowActionBuilder:
    """Maintain left/right Quest hand adapters as independent manipulators."""

    side_builders: dict[str, Quest3PandaShadowActionBuilder]
    agent_uids: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls,
        env: Any,
        *,
        left_retargeting_config_path: str | Path | None = None,
        right_retargeting_config_path: str | Path | None = None,
        retargeting_urdf_root: str | Path | None = None,
        wrist_motion_config: Quest3WristMotionConfig | None = None,
    ) -> "Quest3DualPandaShadowActionBuilder":
        get_spec = env.get_wrapper_attr("get_control_spec")
        raw_spec = get_spec()
        if isinstance(raw_spec.get("manipulators"), dict):
            multi_spec = {
                side: {
                    **raw_spec["manipulators"][side],
                    "control_spec": raw_spec["manipulators"][side],
                    "action_shape": tuple(
                        int(dim)
                        for dim in getattr(
                            raw_spec["manipulators"][side]
                            .get("action_space"),
                            "shape",
                            (),
                        )
                    ),
                }
                for side in ("left", "right")
            }
        else:
            # Compatibility with the original dual-task metadata provider.
            multi_spec = raw_spec
        if set(multi_spec) != {"left", "right"}:
            raise ValueError(
                "Dual Quest teleoperation requires left/right control specs, "
                f"got {sorted(multi_spec)}."
            )

        # Keep semantic hand values in the benchmark's canonical joint order.
        # The controller components are split into force groups and therefore
        # are not necessarily ordered like the public action schema.  The
        # single-arm builder already receives this order from get_action_spec;
        # dual-arm construction must do the same independently for each side.
        canonical_hand_joint_names_by_side: dict[str, tuple[str, ...] | None] = {
            "left": None,
            "right": None,
        }
        get_action_spec = getattr(env, "get_wrapper_attr", None)
        if callable(get_action_spec):
            action_spec = get_action_spec("get_action_spec")()
            for side in ("left", "right"):
                canonical_hand_joint_names_by_side[side] = tuple(
                    str(name)
                    for name in action_spec["manipulators"][side][
                        "hand_joint_position"
                    ]["joint_names"]
                )

        config_paths = {
            "left": left_retargeting_config_path,
            "right": right_retargeting_config_path,
        }
        side_builders: dict[str, Quest3PandaShadowActionBuilder] = {}
        agent_uids: dict[str, str] = {}
        for side in ("left", "right"):
            side_spec = multi_spec[side]
            if not side_spec.get("action_shape"):
                action_space = side_spec.get("action_space")
                side_spec["action_shape"] = tuple(
                    int(dim) for dim in getattr(action_space, "shape", ())
                )
            side_builders[side] = Quest3PandaShadowActionBuilder.from_control_spec(
                side_spec["control_spec"],
                action_shape=tuple(side_spec["action_shape"]),
                side=side,
                retargeting_config_path=config_paths[side],
                retargeting_urdf_root=retargeting_urdf_root,
                wrist_motion_config=wrist_motion_config,
                canonical_hand_joint_names=canonical_hand_joint_names_by_side[side],
            )
            agent_uids[side] = str(side_spec["agent_uid"])
        return cls(side_builders=side_builders, agent_uids=agent_uids)

    def reset(self) -> None:
        for side_builder in self.side_builders.values():
            side_builder.reset()

    def reset_wrist(self, side: str) -> None:
        self.side_builders[side].reset_wrist()

    def recover_wrist_tracking_error(
        self,
        side: str,
        *,
        position_error_root_m: np.ndarray,
        rotation_error_root: np.ndarray,
    ) -> None:
        self.side_builders[side].recover_wrist_tracking_error(
            position_error_root_m=position_error_root_m,
            rotation_error_root=rotation_error_root,
        )

    def seed_hand_targets_from_robot_state(
        self,
        *,
        joint_names_by_side: Mapping[str, tuple[str, ...]],
        joint_qpos_by_side: Mapping[str, np.ndarray],
    ) -> None:
        """Seed both hand filters from the matching simulator joint states."""

        expected_sides = {"left", "right"}
        if set(self.side_builders) != expected_sides:
            raise ValueError(
                "Dual hand builders must contain exactly left and right; "
                f"got {sorted(self.side_builders)}."
            )
        for field_name, values in (
            ("joint_names_by_side", joint_names_by_side),
            ("joint_qpos_by_side", joint_qpos_by_side),
        ):
            actual_sides = set(values)
            if actual_sides != expected_sides:
                raise ValueError(
                    f"{field_name} must contain exactly left and right; "
                    f"got {sorted(actual_sides)}."
                )

        # Validate both inputs before mutating either builder so calibration is
        # atomic across the two manipulators.
        validated_qpos: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            names = tuple(str(name) for name in joint_names_by_side[side])
            qpos = np.asarray(joint_qpos_by_side[side], dtype=np.float32)
            if qpos.shape != (len(names),):
                raise ValueError(
                    f"Robot {side} hand target seed shape does not match joint "
                    f"names: qpos={qpos.shape}, joint_names={len(names)}."
                )
            if not np.isfinite(qpos).all():
                raise ValueError(
                    f"Robot {side} hand target seed must contain only finite values."
                )
            if len(names) != len(set(names)):
                raise ValueError(
                    f"Robot {side} hand target seed joint names must be unique."
                )
            missing_names = [
                name
                for name in self.side_builders[side].retargeting_joint_names
                if name not in names
            ]
            if missing_names:
                raise ValueError(
                    f"Robot {side} state is missing retargeting joints: "
                    + ", ".join(missing_names)
                )
            validated_qpos[side] = qpos

        for side in ("left", "right"):
            self.side_builders[side].seed_hand_target_from_robot_state(
                joint_names=tuple(joint_names_by_side[side]),
                joint_qpos=validated_qpos[side],
            )

    def build_side_action(self, side: str, hand: Quest3HandInput) -> dict[str, np.ndarray]:
        """Build one arm action so the other arm can tolerate a brief loss."""

        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}.")
        return self.side_builders[side].build_action(hand)

    def build_action(self, frame: Quest3InputFrame) -> dict[str, dict[str, np.ndarray]]:
        if frame.left is None or frame.right is None:
            raise ValueError("Dual-arm action requires both Quest hands to be tracked.")
        return {
            "left": self.build_side_action("left", frame.left),
            "right": self.build_side_action("right", frame.right),
        }

    def build_action_pair(self, frame: Quest3InputFrame) -> Quest3ActionPair:
        if frame.left is None or frame.right is None:
            raise ValueError("Dual-arm action requires both Quest hands to be tracked.")
        pairs = {
            side: self.side_builders[side].build_action_pair(getattr(frame, side))
            for side in ("left", "right")
        }
        return Quest3ActionPair(
            semantic={
                side: pairs[side].semantic[side]
                for side in ("left", "right")
            },
            native={
                side: pairs[side].native
                for side in ("left", "right")
            },
        )

    def rebuild_action_pair_from_semantic(
        self,
        semantic: Mapping[str, Mapping[str, np.ndarray]],
    ) -> Quest3ActionPair:
        """Re-encode filtered semantic actions for both manipulators."""

        pairs: dict[str, Quest3ActionPair] = {}
        for side in ("left", "right"):
            pairs[side] = self.side_builders[side].rebuild_action_pair_from_semantic(
                {side: semantic[side]}
            )
        return Quest3ActionPair(
            semantic={side: pairs[side].semantic[side] for side in ("left", "right")},
            native={side: pairs[side].native for side in ("left", "right")},
        )

    def build_rotation_followup_action_pair(
        self, pair: Quest3ActionPair
    ) -> Quest3ActionPair:
        native: dict[str, np.ndarray] = {}
        semantic: dict[str, Mapping[str, np.ndarray]] = {}
        for side in ("left", "right"):
            side_pair = Quest3ActionPair(
                semantic={side: pair.semantic[side]},
                native=pair.native[side],  # type: ignore[index]
            )
            updated = self.side_builders[side].build_rotation_followup_action_pair(side_pair)
            semantic[side] = updated.semantic[side]
            native[side] = np.asarray(updated.native, dtype=np.float32)
        return Quest3ActionPair(semantic=semantic, native=native)
