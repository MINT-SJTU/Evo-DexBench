import copy
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import vectorize_pose
from transforms3d.euler import euler2quat

try:
    from mani_skill.agents.robots.panda.panda import Panda
except ImportError:
    from mani_skill.agents.robots.panda import Panda

from ..shadow_hand.spec import (
    FINGERTIP_NAMES,
    RIGHT_HAND_TIP_LINK_NAMES,
    TOUCH_LINK_NAMES,
    dmp_links_names,
    hand_joints_name,
)
from ..shadow_hand.taxel_layout import (
    DISTAL_TAXEL_MASK,
    FINGER_NAMES as TAXEL_FINGER_NAMES,
    LEFT_TAXEL_LAYOUT,
    LEFT_TAXEL_LINK_NAMES,
    RIGHT_TAXEL_LAYOUT,
    RIGHT_TAXEL_LINK_NAMES,
    TAXEL_FINGER_INDICES,
    TAXEL_FORCE_UNIT,
    TAXEL_LAYOUT_HASH,
    TAXEL_LAYOUT_ID,
    TAXEL_LAYOUT_VERSION,
    TAXEL_REGION_INDICES,
    TAXEL_SENSOR_MODEL,
    TaxelSpec,
)
from ..tactile import (
    CpuPairwiseContactQueryHandle,
    PairwiseContactQueryHandle,
    TactileContactDiagnostics,
    TactileContactState,
    TactileQueryCounters,
    TactileQueryManager,
    aggregate_distal_taxels_by_finger,
)
ROBOT_DIR = Path(__file__).resolve().parent
ASSET_ROOT = ROBOT_DIR.parent / "assets"

FINGER_JOINT_ARMATURE_KG_M2 = 1e-5
INDEX_FINGER_STIFFNESS = 1e3
INDEX_FINGER_FORCE_LIMIT_N_M = 1e1
EXTERNAL_COLLISION_MEMBERSHIP_BIT = 0
RIGHT_ROBOT_COLLISION_MEMBERSHIP_BIT = 1
LEFT_ROBOT_COLLISION_MEMBERSHIP_BIT = 3
RIGHT_ADAPTER_COLLISION_IGNORE_BIT = 24
LEFT_ADAPTER_COLLISION_IGNORE_BIT = 25
PANDA_SHADOW_ADAPTER_COLLISION_LINK_NAMES = ("panda_link7", "panda_hand")
# Joint-coordinate migration after rotating the link5 installation by -2.897 rad:
# qpos_new = qpos_legacy + PANDA_SHADOW_JOINT5_REZERO_OFFSET_RAD.
PANDA_SHADOW_JOINT5_REZERO_OFFSET_RAD: float = 2.897

# Default Panda arm posture shared by the right-hand robot variants. The
# remaining 24 DOFs belong to the Shadow wrist and hand and stay at zero.
PANDA_SHADOW_RIGHT_REST_ARM_QPOS: np.ndarray = np.asarray(
    [0.0, -0.282, 0.172, -2.726, 0.0, 2.216, 2.194],
    dtype=np.float32,
)
# The mirrored left hand has an additional pi flange rotation. Express the
# equivalent joint-7 compensation inside Franka's [-2.8973, 2.8973] limit.
PANDA_SHADOW_LEFT_REST_ARM_QPOS: np.ndarray = (
    PANDA_SHADOW_RIGHT_REST_ARM_QPOS.copy()
)
PANDA_SHADOW_LEFT_REST_ARM_QPOS[6] -= np.pi
PANDA_SHADOW_HAND_REST_QPOS: np.ndarray = np.zeros(24, dtype=np.float32)


def _replace_right_hand_prefix(
    names: Iterable[str],
    target_prefix: str,
) -> tuple[str, ...]:
    converted_names: list[str] = []
    for name in names:
        if not name.startswith("rh_"):
            raise ValueError(f"Expected a right-hand name, got {name!r}.")
        converted_names.append(f"{target_prefix}{name.removeprefix('rh_')}")
    return tuple(converted_names)


RIGHT_HAND_JOINT_NAMES = tuple(hand_joints_name)
LEFT_HAND_JOINT_NAMES = _replace_right_hand_prefix(hand_joints_name, "lh_")
RIGHT_FINGER_JOINT_NAMES = tuple(
    joint_name
    for joint_name in RIGHT_HAND_JOINT_NAMES
    if not joint_name.startswith("rh_WRJ")
)
LEFT_FINGER_JOINT_NAMES = _replace_right_hand_prefix(
    RIGHT_FINGER_JOINT_NAMES,
    "lh_",
)
RIGHT_INDEX_FINGER_JOINT_NAMES = tuple(
    joint_name
    for joint_name in RIGHT_HAND_JOINT_NAMES
    if joint_name.startswith("rh_FFJ")
)
LEFT_INDEX_FINGER_JOINT_NAMES = _replace_right_hand_prefix(
    RIGHT_INDEX_FINGER_JOINT_NAMES,
    "lh_",
)
RIGHT_HAND_DMP_LINK_NAMES = tuple(dmp_links_names)
LEFT_HAND_DMP_LINK_NAMES = _replace_right_hand_prefix(dmp_links_names, "lh_")
LEFT_HAND_TIP_LINK_NAMES = _replace_right_hand_prefix(
    RIGHT_HAND_TIP_LINK_NAMES,
    "lh_",
)


@register_agent()
class PandaShadowSimpleTactile(Panda):
    uid = "panda_shadow_simple_tactile"
    urdf_path = str(ASSET_ROOT / "panda_shadow" / "panda_shadow_simple_tactile.urdf")
    adapter_collision_ignore_bit = RIGHT_ADAPTER_COLLISION_IGNORE_BIT
    tcp_link_name = "panda_hand_tcp"
    fingertip_names = FINGERTIP_NAMES
    hand_joint_names = RIGHT_HAND_JOINT_NAMES
    finger_joint_names = RIGHT_FINGER_JOINT_NAMES
    index_finger_joint_names = RIGHT_INDEX_FINGER_JOINT_NAMES
    hand_tip_link_names = RIGHT_HAND_TIP_LINK_NAMES
    hand_palm_link_name = "rh_palm"
    hand_forearm_link_name = "rh_forearm"
    wrist_camera_uid = "wrist_right_camera"
    high_force_hand_joint_names = ("rh_THJ4", "rh_THJ5", "rh_WRJ1", "rh_WRJ2")
    wrist_camera_local_pose: sapien.Pose = sapien.Pose(
        p=[0.06, -0.10, 0.08],
        q=euler2quat(np.pi * 7.0 / 6.0, -np.pi / 2.0, 0.0),
    )

    @property
    def _sensor_configs(self) -> list[CameraConfig]:
        base_configs = list(super()._sensor_configs)
        wrist_mount_link_name = self.hand_forearm_link_name
        wrist_mount = self.robot.links_map.get(wrist_mount_link_name)
        if wrist_mount is None:
            known_links = ", ".join(sorted(self.robot.links_map)) or "<empty>"
            raise KeyError(
                f"Wrist-camera mount link {wrist_mount_link_name!r} was "
                f"not found on robot {self.uid!r}. Known links: {known_links}."
            )
        base_configs.append(
            CameraConfig(
                uid=self.wrist_camera_uid,
                pose=self.wrist_camera_local_pose,
                width=224,
                height=224,
                fov=np.pi / 3.0,
                near=0.01,
                far=2.0,
                mount=wrist_mount,
            )
        )
        return base_configs

    # 31 active DOFs in the robot:
    # 7 Panda arm joints + 2 wrist joints + 22 Shadow hand joints.
    # Define an explicit rest keyframe so demo_robot.py does not inherit
    # Panda's default 9-DOF keyframe.
    keyframes = dict(
        rest=Keyframe(
            qpos=np.concatenate(
                (
                    PANDA_SHADOW_RIGHT_REST_ARM_QPOS,
                    PANDA_SHADOW_HAND_REST_QPOS,
                )
            ),
            pose=sapien.Pose(),
        )
    )

    touch_link_names = TOUCH_LINK_NAMES

    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            tip_link: dict(material="tip", patch_radius=0.1, min_patch_radius=0.1)
            for tip_link in RIGHT_HAND_DMP_LINK_NAMES
            + ("rh_palm", "rh_lfmetacarpal")
        },
    )

    @property
    def _controller_configs(self):
        configs = super()._controller_configs

        for key in list(configs.keys()):
            if "gripper" in configs[key]:
                del configs[key]["gripper"]
            if "extra" in configs[key]:
                del configs[key]["extra"]

        active_joints_name = self.hand_joint_names
        stiffness = 1e3
        damping = 1e1
        joints_part_with_force_3e1 = self.high_force_hand_joint_names
        joints_part_with_force_1e1 = [
            joint_name
            for joint_name in active_joints_name
            if joint_name not in joints_part_with_force_3e1
        ]
        index_finger_joint_names = set(self.index_finger_joint_names)
        missing_index_finger_joints = index_finger_joint_names - set(
            joints_part_with_force_1e1
        )
        if missing_index_finger_joints:
            raise ValueError(
                "Index-finger joints must belong to the low-force controller "
                "group, missing: "
                + ", ".join(sorted(missing_index_finger_joints))
            )
        # Preserve the existing component and action ordering. Keep explicit
        # FFJ entries so index-finger tuning remains independently testable.
        low_force_stiffness = [
            INDEX_FINGER_STIFFNESS
            if joint_name in index_finger_joint_names
            else stiffness
            for joint_name in joints_part_with_force_1e1
        ]
        low_force_limits_n_m = [
            INDEX_FINGER_FORCE_LIMIT_N_M
            if joint_name in index_finger_joint_names
            else 1e1
            for joint_name in joints_part_with_force_1e1
        ]

        pd_joint_pos_3e1 = PDJointPosControllerConfig(
            joints_part_with_force_3e1,
            lower=None,
            upper=None,
            stiffness=stiffness,
            damping=damping,
            force_limit=3e1,
            normalize_action=False,
        )
        pd_joint_pos_1e1 = PDJointPosControllerConfig(
            joints_part_with_force_1e1,
            lower=None,
            upper=None,
            stiffness=low_force_stiffness,
            damping=damping,
            force_limit=low_force_limits_n_m,
            normalize_action=False,
        )

        pd_joint_delta_pos_3e1 = PDJointPosControllerConfig(
            joints_part_with_force_3e1,
            -0.1,
            0.1,
            stiffness=stiffness,
            damping=damping,
            force_limit=3e1,
            use_delta=True,
        )
        pd_joint_delta_pos_1e1 = PDJointPosControllerConfig(
            joints_part_with_force_1e1,
            -0.1,
            0.1,
            stiffness=low_force_stiffness,
            damping=damping,
            force_limit=low_force_limits_n_m,
            use_delta=True,
        )

        arm_joint_pos = configs["pd_joint_pos"]["arm"]
        arm_joint_delta_pos = configs["pd_joint_delta_pos"]["arm"]
        arm_ee_delta_pose = copy.deepcopy(configs["pd_ee_delta_pose"]["arm"])
        # Preserve uncompleted motion by accumulating deltas from the previous
        # controller target. Teleoperation runners may explicitly reset this
        # target to the live EE pose when their input policy requires it.
        arm_ee_delta_pose.use_target = True
        arm_ee_pose = configs["pd_ee_pose"]["arm"]

        hand_joint_pos = dict(
            hand_joint_3e1=pd_joint_pos_3e1,
            hand_joint_1e1=pd_joint_pos_1e1,
        )
        hand_joint_delta_pos = dict(
            hand_joint_3e1=pd_joint_delta_pos_3e1,
            hand_joint_1e1=pd_joint_delta_pos_1e1,
        )

        def combined_controller_config(
            arm_controller: Any,
            hand_controllers: dict[str, Any],
        ) -> dict[str, Any]:
            return dict(
                arm=copy.deepcopy(arm_controller),
                **copy.deepcopy(hand_controllers),
            )

        # Base joint modes keep matching arm/hand joint semantics.
        configs["pd_joint_pos"] = combined_controller_config(
            arm_joint_pos,
            hand_joint_pos,
        )
        configs["pd_joint_delta_pos"] = combined_controller_config(
            arm_joint_delta_pos,
            hand_joint_delta_pos,
        )

        # Explicit Bench modes avoid hidden overwrites between arm and hand control types.
        configs["arm_pd_joint_pos_hand_pd_joint_pos"] = combined_controller_config(
            arm_joint_pos,
            hand_joint_pos,
        )
        configs["arm_pd_joint_pos_hand_pd_joint_delta_pos"] = combined_controller_config(
            arm_joint_pos,
            hand_joint_delta_pos,
        )
        configs["arm_pd_joint_delta_pos_hand_pd_joint_pos"] = combined_controller_config(
            arm_joint_delta_pos,
            hand_joint_pos,
        )
        configs["arm_pd_joint_delta_pos_hand_pd_joint_delta_pos"] = combined_controller_config(
            arm_joint_delta_pos,
            hand_joint_delta_pos,
        )
        configs["arm_pd_ee_delta_pose_hand_pd_joint_pos"] = combined_controller_config(
            arm_ee_delta_pose,
            hand_joint_pos,
        )
        configs["arm_pd_ee_delta_pose_hand_pd_joint_delta_pos"] = combined_controller_config(
            arm_ee_delta_pose,
            hand_joint_delta_pos,
        )
        configs["arm_pd_ee_pose_hand_pd_joint_pos"] = combined_controller_config(
            arm_ee_pose,
            hand_joint_pos,
        )
        configs["arm_pd_ee_pose_hand_pd_joint_delta_pos"] = combined_controller_config(
            arm_ee_pose,
            hand_joint_delta_pos,
        )

        return configs

    @property
    def controller_joint_indices(self):
        return self.controller.active_joint_indices

    def get_arm_ee_target_position_error(self) -> torch.Tensor | None:
        """Return the current root-frame arm target error as ``[B, 3]``."""

        diagnostics = self.get_arm_ee_target_position_diagnostics()
        if diagnostics is None:
            return None
        return diagnostics["position_error_m"]

    def get_arm_ee_target_position_diagnostics(
        self,
    ) -> dict[str, torch.Tensor] | None:
        """Return target/current root-frame EE positions as ``[B, 3]``."""

        component_controllers = getattr(self.controller, "controllers", None)
        if not isinstance(component_controllers, dict):
            return None
        arm_controller = component_controllers.get("arm")
        if arm_controller is None:
            return None
        target_pose = getattr(arm_controller, "_target_pose", None)
        current_pose = getattr(arm_controller, "ee_pose_at_base", None)
        if target_pose is None or current_pose is None:
            return None
        # [B, 3] - [B, 3] -> [B, 3].
        position_error = target_pose.p - current_pose.p
        expected_shape = (self.scene.num_envs, 3)
        if (
            tuple(target_pose.p.shape) != expected_shape
            or tuple(current_pose.p.shape) != expected_shape
            or tuple(position_error.shape) != expected_shape
        ):
            raise ValueError(
                "Unexpected arm EE target/current position shape: "
                f"expected {expected_shape}, got target={tuple(target_pose.p.shape)}, "
                f"current={tuple(current_pose.p.shape)}, "
                f"error={tuple(position_error.shape)}."
            )
        return {
            "target_position_m": target_pose.p,
            "current_position_m": current_pose.p,
            "position_error_m": position_error,
        }

    # def set_action(self, action):
    #     action = action[..., self.controller_joint_indices]
    #     super(Panda, self).set_action(action)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tip_link_names = self.hand_tip_link_names
        self.palm_link_name = self.hand_palm_link_name
        self.forearm_link_name = self.hand_forearm_link_name
        self.gripper_joint_names = []
        super().__init__(*args, **kwargs)

    def _after_loading_articulation(self) -> None:
        """Add reflected inertia to active finger joints before GPU init."""

        super()._after_loading_articulation()
        missing_joint_names = sorted(
            set(self.finger_joint_names) - set(self.robot.active_joints_map)
        )
        if missing_joint_names:
            raise KeyError(
                "Finger joints required for armature were not loaded: "
                + ", ".join(missing_joint_names)
            )

        armature = np.asarray([FINGER_JOINT_ARMATURE_KG_M2], dtype=np.float32)
        for joint_name in self.finger_joint_names:
            joint = self.robot.active_joints_map[joint_name]
            for physx_joint in joint._objs:
                if physx_joint.type not in ("revolute", "revolute_unwrapped"):
                    raise TypeError(
                        f"Finger joint {joint_name!r} must be rotational, got "
                        f"{physx_joint.type!r}."
                    )
                if physx_joint.get_dof() != 1:
                    raise ValueError(
                        f"Finger joint {joint_name!r} must have one DOF, got "
                        f"{physx_joint.get_dof()}."
                    )
                physx_joint.set_armature(armature)

    def _after_init(self) -> None:
        super(Panda, self)._after_init()
        self.tip_links = sapien_utils.get_objs_by_names(self.robot.get_links(), self.tip_link_names)
        self.palm_link = sapien_utils.get_obj_by_name(self.robot.get_links(), self.palm_link_name)
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.tcp_link_name)
        self.touch_links = sapien_utils.get_objs_by_names(self.robot.get_links(), self.touch_link_names)
        self._configure_adapter_collision_filter()

    def _configure_adapter_collision_filter(self) -> None:
        """Disable only the overlapping Panda hand-adapter collision pair."""

        adapter_links = tuple(
            self.robot.links_map.get(link_name)
            for link_name in PANDA_SHADOW_ADAPTER_COLLISION_LINK_NAMES
        )
        if any(link is None for link in adapter_links):
            known_links = ", ".join(sorted(self.robot.links_map)) or "<empty>"
            raise RuntimeError(
                "Panda-Shadow adapter collision filtering requires links "
                f"{PANDA_SHADOW_ADAPTER_COLLISION_LINK_NAMES!r}; known links: "
                f"{known_links}."
            )
        for adapter_link in adapter_links:
            adapter_link.set_collision_group_bit(
                group=2,
                bit_idx=self.adapter_collision_ignore_bit,
                bit=1,
            )

    def get_tactile_forces(self, contact_objects) -> torch.Tensor:
        """Return fingertip forces against explicitly supplied task entities."""

        if not isinstance(contact_objects, (list, tuple)):
            contact_objects = [contact_objects]

        forces = []
        for link in self.touch_links:
            f_total = 0
            for obj in contact_objects:
                f_total = f_total + self.scene.get_pairwise_contact_forces(link, obj)
            forces.append(f_total)

        return torch.stack(forces, dim=1)

    def get_net_tactile_forces(self) -> torch.Tensor:
        """Return each fingertip's net force from all contacts as [B, F, 3]."""

        forces = self.robot.get_net_contact_forces(self.touch_link_names)
        expected_shape = (self.scene.num_envs, len(self.touch_link_names), 3)
        if tuple(forces.shape) != expected_shape:
            raise ValueError(
                "Unexpected net tactile force shape: "
                f"expected {expected_shape}, got {tuple(forces.shape)}."
            )
        return forces

    def get_proprioception(self):
        obs = super().get_proprioception()
        tip_poses = [vectorize_pose(link.pose, device=self.device) for link in self.tip_links]
        tip_poses_tensor = torch.stack(tip_poses, dim=-2) if tip_poses else torch.zeros(0, device=self.device)
        obs.update(
            {
                "palm_pose": vectorize_pose(self.palm_link.pose, device=self.device)
                if self.palm_link
                else torch.zeros(7, device=self.device),
                "tip_poses": tip_poses_tensor.reshape(-1, len(self.tip_links) * 7)
                if len(tip_poses) > 0
                else torch.zeros(0, device=self.device),
            }
        )
        return obs


@register_agent()
class PandaShadowSimpleTactileLeft(PandaShadowSimpleTactile):
    """Panda arm with a kinematically and geometrically mirrored left hand."""

    uid = "panda_shadow_simple_tactile_left"
    urdf_path = str(
        ASSET_ROOT / "panda_shadow" / "panda_shadow_simple_tactile_left.urdf"
    )
    hand_joint_names = LEFT_HAND_JOINT_NAMES
    finger_joint_names = LEFT_FINGER_JOINT_NAMES
    index_finger_joint_names = LEFT_INDEX_FINGER_JOINT_NAMES
    hand_tip_link_names = LEFT_HAND_TIP_LINK_NAMES
    hand_palm_link_name = "lh_palm"
    hand_forearm_link_name = "lh_forearm"
    wrist_camera_uid = "wrist_left_camera"
    high_force_hand_joint_names = ("lh_THJ4", "lh_THJ5", "lh_WRJ1", "lh_WRJ2")
    adapter_collision_ignore_bit = LEFT_ADAPTER_COLLISION_IGNORE_BIT
    # The left-hand URDF mirrors the right hand across the forearm-local X axis.
    # This pose mirrors the calibrated camera position, forward axis, and up axis.
    wrist_camera_local_pose: sapien.Pose = sapien.Pose(
        p=[-0.06, -0.10, 0.08],
        q=euler2quat(-np.pi / 6.0, -np.pi / 2.0, 0.0),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.concatenate(
                (
                    PANDA_SHADOW_LEFT_REST_ARM_QPOS,
                    PANDA_SHADOW_HAND_REST_QPOS,
                )
            ),
            pose=sapien.Pose(),
        )
    )

    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link={
            link_name: dict(
                material="tip",
                patch_radius=0.1,
                min_patch_radius=0.1,
            )
            for link_name in LEFT_HAND_DMP_LINK_NAMES
            + ("lh_palm", "lh_lfmetacarpal")
        },
    )


class _PandaShadowDistributedTactileMixin:
    """Runtime APIs shared by distributed-taxel Shadow Hand variants."""

    taxel_layout_id = TAXEL_LAYOUT_ID
    taxel_layout_version = TAXEL_LAYOUT_VERSION
    taxel_layout_hash = TAXEL_LAYOUT_HASH
    taxel_force_unit = TAXEL_FORCE_UNIT
    taxel_sensor_model = TAXEL_SENSOR_MODEL
    touch_link_names: tuple[str, ...] = ()
    taxel_specs: tuple[TaxelSpec, ...]
    taxel_link_names: tuple[str, ...]
    taxel_finger_indices_data: tuple[int, ...] = TAXEL_FINGER_INDICES
    taxel_region_indices_data: tuple[int, ...] = TAXEL_REGION_INDICES
    distal_taxel_mask_data: tuple[bool, ...] = DISTAL_TAXEL_MASK
    robot_collision_membership_bit: int
    other_robot_collision_membership_bits: tuple[int, ...]

    def reset(self, init_qpos: torch.Tensor | None = None) -> None:
        super().reset(init_qpos)
        tactile_query_manager = getattr(self, "tactile_query_manager", None)
        if tactile_query_manager is not None:
            tactile_query_manager.reset_control_step_accumulation()

    def _after_init(self) -> None:
        super()._after_init()
        expected_taxel_count = len(self.taxel_specs)
        if len(self.taxel_link_names) != expected_taxel_count:
            raise ValueError(
                "Logical taxel name/spec count mismatch: expected "
                f"{expected_taxel_count}, got {len(self.taxel_link_names)}."
            )
        self.taxel_parent_links = tuple(
            self.robot.links_map[spec.parent_link_name]
            for spec in self.taxel_specs
        )
        self._configure_robot_collision_filters()

        self.taxel_finger_indices = torch.as_tensor(
            self.taxel_finger_indices_data,
            device=self.device,
            dtype=torch.int64,
        )
        self.taxel_region_indices = torch.as_tensor(
            self.taxel_region_indices_data,
            device=self.device,
            dtype=torch.int64,
        )
        self.distal_taxel_mask = torch.as_tensor(
            self.distal_taxel_mask_data,
            device=self.device,
            dtype=torch.bool,
        )
        self.tactile_query_manager = TactileQueryManager(
            scene=self.scene,
            robot_links=self.robot.get_links(),
            taxel_parent_links=self.taxel_parent_links,
            taxel_specs=self.taxel_specs,
            # PhysX reports the impulse applied to the queried parent body.
            # External compression therefore points opposite its outward normal.
            force_sign=-1.0,
        )

    @staticmethod
    def _collision_membership_mask(bit_indices: Iterable[int]) -> int:
        bit_mask = 0
        for bit_index in bit_indices:
            if not 0 <= bit_index < 32:
                raise ValueError(
                    f"Collision membership bit must be in [0, 31], got "
                    f"{bit_index}."
                )
            bit_mask |= 1 << bit_index
        return bit_mask

    def _configure_robot_collision_filters(self) -> None:
        """Configure original robot shapes without extra tactile colliders."""

        category_bits = (
            EXTERNAL_COLLISION_MEMBERSHIP_BIT,
            self.robot_collision_membership_bit,
            *self.other_robot_collision_membership_bits,
        )
        if len(set(category_bits)) != len(category_bits):
            raise ValueError(
                "Robot collision membership bits must be unique, got "
                f"{category_bits}."
            )
        external_mask = self._collision_membership_mask(
            (EXTERNAL_COLLISION_MEMBERSHIP_BIT,)
        )
        other_robot_mask = self._collision_membership_mask(
            self.other_robot_collision_membership_bits
        )
        robot_membership = self._collision_membership_mask(
            (self.robot_collision_membership_bit,)
        )
        robot_collision_mask = (
            external_mask | robot_membership | other_robot_mask
        )

        for robot_link in self.robot.get_links():
            for body in robot_link._bodies:
                for collision_shape in body.get_collision_shapes():
                    collision_groups = list(
                        collision_shape.get_collision_groups()
                    )
                    if len(collision_groups) != 4:
                        raise ValueError(
                            f"Collision shape on link {robot_link.name!r} "
                            "must expose four collision groups, got "
                            f"{collision_groups}."
                        )
                    collision_groups[0] = robot_membership
                    collision_groups[1] = robot_collision_mask
                    collision_shape.set_collision_groups(collision_groups)

    @property
    def tactile_query_counters(self) -> TactileQueryCounters:
        return self.tactile_query_manager.counters

    def invalidate_tactile_cache(self) -> None:
        self.tactile_query_manager.invalidate()

    def begin_tactile_control_step(self) -> None:
        self.tactile_query_manager.begin_control_step_accumulation()

    def reset_tactile_control_step_accumulation(self) -> None:
        self.tactile_query_manager.reset_control_step_accumulation()

    def accumulate_tactile_sim_step(self) -> None:
        self.tactile_query_manager.accumulate_sim_step_impulses()

    def end_tactile_control_step(self, *, expected_sim_steps: int) -> None:
        self.tactile_query_manager.end_control_step_accumulation(
            expected_sim_steps=expected_sim_steps
        )

    def current_cpu_contacts(self) -> tuple[Any, ...]:
        return self.tactile_query_manager.current_cpu_contacts()

    def prepare_cpu_object_taxel_query(
        self,
        target_groups: tuple[tuple[str, tuple[Any, ...]], ...],
    ) -> None:
        self.tactile_query_manager.prepare_cpu_object_query(target_groups)

    def prepare_cpu_pairwise_contact_query(
        self,
        first_links: tuple[Any, ...],
        second_links: tuple[Any, ...],
    ) -> CpuPairwiseContactQueryHandle:
        return self.tactile_query_manager.prepare_cpu_pairwise_contact_query(
            first_links,
            second_links,
        )

    def prepare_pairwise_contact_query(
        self,
        first_links: tuple[Any, ...],
        second_links: tuple[Any, ...],
    ) -> PairwiseContactQueryHandle:
        return self.tactile_query_manager.prepare_pairwise_contact_query(
            first_links,
            second_links,
        )

    def get_cpu_pairwise_contact_forces(
        self,
        query_handle: CpuPairwiseContactQueryHandle,
    ) -> torch.Tensor:
        return self.tactile_query_manager.get_cpu_pairwise_contact_forces(
            query_handle
        )

    def get_pairwise_contact_forces(
        self,
        query_handle: PairwiseContactQueryHandle,
    ) -> torch.Tensor:
        return self.tactile_query_manager.get_pairwise_contact_forces(
            query_handle
        )

    def get_net_taxel_normal_forces(self) -> torch.Tensor:
        return self.tactile_query_manager.get_net_taxel_normal_forces()

    def get_tactile_contact_state(self) -> TactileContactState:
        """Return control-period parent-link and finger contact resultants."""

        return self.tactile_query_manager.get_contact_state()

    def get_tactile_contact_diagnostics(
        self,
    ) -> TactileContactDiagnostics | None:
        """Return completed control-period tactile mapping diagnostics."""

        return self.tactile_query_manager.get_control_contact_diagnostics()

    def get_tactile_cpu_control_step_profile(self) -> Mapping[str, float | int]:
        """Return timing/counts for the latest CPU tactile accumulation."""

        return self.tactile_query_manager.get_cpu_control_step_profile()

    def set_contact_diagnostics_enabled(self, enabled: bool) -> None:
        """Enable or disable detailed parent-to-taxel diagnostics."""

        self.tactile_query_manager.set_contact_diagnostics_enabled(enabled)

    def get_taxel_world_positions(self) -> torch.Tensor:
        return self.tactile_query_manager.get_taxel_world_positions()

    def get_object_taxel_world_forces(
        self,
        target_groups: tuple[tuple[str, tuple[Any, ...]], ...],
    ) -> dict[str, torch.Tensor]:
        result = self.tactile_query_manager.get_object_taxel_world_forces(
            target_groups
        )
        return dict(result)

    def get_object_taxel_normal_forces(
        self,
        target_groups: tuple[tuple[str, tuple[Any, ...]], ...],
    ) -> dict[str, torch.Tensor]:
        result = self.tactile_query_manager.get_object_taxel_normal_forces(
            target_groups
        )
        return dict(result)

    def aggregate_taxel_world_forces_by_parent(
        self,
        taxel_world_forces: torch.Tensor,
    ) -> torch.Tensor:
        return self.tactile_query_manager.aggregate_taxel_world_forces_by_parent(
            taxel_world_forces
        )

    def aggregate_distal_taxels_by_finger(
        self,
        taxel_normal_forces: torch.Tensor,
    ) -> torch.Tensor:
        return aggregate_distal_taxels_by_finger(
            taxel_normal_forces,
            finger_indices=self.taxel_finger_indices,
            distal_mask=self.distal_taxel_mask,
            finger_count=len(TAXEL_FINGER_NAMES),
        )


@register_agent()
class PandaShadowDistributedTactile123(
    _PandaShadowDistributedTactileMixin,
    PandaShadowSimpleTactile,
):
    """Right Panda + Shadow Hand with the balanced 123-taxel array."""

    uid = "panda_shadow_distributed_tactile_123"
    urdf_path = str(
        ASSET_ROOT
        / "panda_shadow"
        / "panda_shadow_distributed_tactile_123.urdf"
    )
    taxel_specs = RIGHT_TAXEL_LAYOUT
    taxel_link_names = RIGHT_TAXEL_LINK_NAMES
    robot_collision_membership_bit = RIGHT_ROBOT_COLLISION_MEMBERSHIP_BIT
    other_robot_collision_membership_bits = (
        LEFT_ROBOT_COLLISION_MEMBERSHIP_BIT,
    )
    urdf_config = copy.deepcopy(PandaShadowSimpleTactile.urdf_config)


@register_agent()
class PandaShadowDistributedTactile123Left(
    _PandaShadowDistributedTactileMixin,
    PandaShadowSimpleTactileLeft,
):
    """Mirrored left Panda + Shadow Hand with the balanced 123-taxel array."""

    uid = "panda_shadow_distributed_tactile_123_left"
    urdf_path = str(
        ASSET_ROOT
        / "panda_shadow"
        / "panda_shadow_distributed_tactile_123_left.urdf"
    )
    taxel_specs = LEFT_TAXEL_LAYOUT
    taxel_link_names = LEFT_TAXEL_LINK_NAMES
    robot_collision_membership_bit = LEFT_ROBOT_COLLISION_MEMBERSHIP_BIT
    other_robot_collision_membership_bits = (
        RIGHT_ROBOT_COLLISION_MEMBERSHIP_BIT,
    )
    urdf_config = copy.deepcopy(PandaShadowSimpleTactileLeft.urdf_config)
