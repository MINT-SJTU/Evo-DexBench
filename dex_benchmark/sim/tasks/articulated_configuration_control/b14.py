from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from dex_benchmark.assets import partnet_mobility_modified_model_urdf_path
from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.scenes.layouts import (
    ArticulatedAssetSpec,
    ArticulationBuildValues,
    PhysicalMaterialSpec,
    PoseSpec,
)
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


B14_SETTINGS = TaskRegistry.register(TaskRegistry.load("b14_v1"))

# PartNet-Mobility 100189 is a rectangular box with one complete rear-hinged
# lid. Its only active DoF is joint_0; unlike the shared workbench box and the
# B05 packing box, it does not split the lid into multiple independently moving
# panels.
B14_BOX_ENTITY_NAME = "b14_hinged_box"
B14_BOX_MODEL_ID = "100189"
B14_LID_JOINT_NAME = "joint_0"
B14_BASE_LINK_NAME = "link_1"
B14_LID_LINK_NAME = "link_0"
B14_LID_HANDLE_COLLISION_NAME = "b14_lid_front_handle"
B14_INTERACTION_SITE_ID = "workbench"

# The workbench interaction frame has +X pointing away from the robots. In the
# selected asset, the rear hinge is on root-local +X, so an identity orientation
# keeps the free lid edge closest to the two hands.
B14_BOX_SCALE = 0.22
B14_BOX_CLOSED_QPOS_RAD = -0.6928957130417486
B14_BOX_MAX_OPEN_ANGLE_DEG = 105.0
B14_BOX_MAX_QPOS_RAD = B14_BOX_CLOSED_QPOS_RAD + math.radians(
    B14_BOX_MAX_OPEN_ANGLE_DEG
)
# At the closed qpos, the unscaled base reaches z=-0.358302 m. This root height
# places the scaled box bottom 3 mm above the workbench before settling.
B14_BOX_CLOSED_MIN_ROOT_Z_M = -0.358302
B14_BOX_INITIAL_CLEARANCE_M = 0.003
B14_BOX_SITE_LOCAL_XYZ = (
    0.060,
    0.0,
    -B14_BOX_SCALE * B14_BOX_CLOSED_MIN_ROOT_Z_M + B14_BOX_INITIAL_CLEARANCE_M,
)
B14_BOX_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)

# SAPIEN assigns roughly 10.4 kg to model 100189 at scale 0.22. Scaling all
# link masses by 0.18 yields a roughly 1.9 kg box: light enough for Shadow Hands
# to manipulate, but light enough to move when the support hand is omitted.
B14_BOX_MASS_SCALE = 0.18
B14_BOX_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=0.85,
    dynamic_friction=0.70,
    restitution=0.01,
)
B14_HINGE_FRICTION = 0.08
B14_HINGE_DAMPING = 0.12

# Clear the central workbench and the lid sweep. The workbench itself and room
# decor remain, preserving the Studio visual context without task ambiguity.
B14_REMOVED_LAYOUT_ITEMS = frozenset(
    {
        "workbench_box",
        "workbench_pliers",
        "glasses",
        "stapler",
        "clutter_paper_placeholder",
        "clutter_large_marker",
        "clutter_adjustable_wrench",
        "clutter_power_drill",
        "clutter_phillips_screwdriver",
        "clutter_dice",
    }
)

B14_ROBOT_BASE_LOCAL_MIN_XYZ = (-0.90, -0.55, -1.00)
B14_ROBOT_BASE_LOCAL_MAX_XYZ = (-0.50, 0.55, 0.20)
B14_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.05, -0.95, -0.30)
B14_TCP_WORKSPACE_LOCAL_MAX_XYZ = (1.05, 0.95, 1.10)

B14_MIN_OPENING_SPEED_RAD_S = 0.02

B14_SPATIAL_PARAMETER_NAMES = (
    "box_x_offset_m",
    "box_y_offset_m",
    "box_yaw_offset_rad",
    "lid_initial_open_angle_deg",
)
B14_PHYSICAL_PARAMETER_NAMES = (
    "box_mass_log_scale",
    "box_friction_log_scale",
    "hinge_friction",
    "hinge_damping",
)
B14_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("box_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("box_y_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec(
                "box_yaw_offset_rad",
                -math.radians(8.0),
                math.radians(8.0),
            ),
            GeneralizationParameterSpec(
                "lid_initial_open_angle_deg",
                0.0,
                3.0,
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "box_mass_log_scale",
                math.log(0.80),
                math.log(1.20),
            ),
            GeneralizationParameterSpec(
                "box_friction_log_scale",
                math.log(0.80),
                math.log(1.20),
            ),
            GeneralizationParameterSpec("hinge_friction", 0.04, 0.14),
            GeneralizationParameterSpec("hinge_damping", 0.06, 0.20),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B14SuccessSpec:
    contact_force_n: float = 0.40
    opening_started_angle_deg: float = 10.0
    target_open_angle_deg: float = 90.0
    maximum_success_angle_deg: float = 105.0
    stable_joint_speed_rad_s: float = 0.25
    stable_dwell_s: float = 1.0
    box_displacement_m: float = 0.10
    box_tilt_deg: float = 7.0
    box_linear_speed_mps: float = 0.05
    box_angular_speed_rad_s: float = 0.50


@register_env(B14_SETTINGS.env_id, max_episode_steps=B14_SETTINGS.max_steps)
class B14Env(StudioTaskEnv):
    """Stabilize a free box with one hand and open its hinged lid with the other."""

    TASK_SETTINGS = B14_SETTINGS
    GENERALIZATION_SPEC = B14_GENERALIZATION_SPEC
    SUCCESS_SPEC = B14SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B14_INTERACTION_SITE_ID
    ROBOT_BASE_LOCAL_MIN_XYZ = B14_ROBOT_BASE_LOCAL_MIN_XYZ
    ROBOT_BASE_LOCAL_MAX_XYZ = B14_ROBOT_BASE_LOCAL_MAX_XYZ
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.286827, -0.527697, 1.116254),
            quaternion_world_wxyz=(
                0.678772,
                -0.203379,
                0.204404,
                0.675370,
            ),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-1.382406, 0.367292, 1.116254),
            quaternion_world_wxyz=(
                0.015449,
                -0.288309,
                0.004653,
                0.957401,
            ),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.577207, -1.211649, 1.116254),
            quaternion_world_wxyz=(
                0.686129,
                0.201128,
                0.206619,
                -0.667894,
            ),
        ),
    }

    def _task_layout_spec(self) -> Any:
        box_world_pose = self.interaction_site_local_pose_to_world(
            PoseSpec(
                position_xyz=B14_BOX_SITE_LOCAL_XYZ,
                quaternion_wxyz=B14_BOX_QUATERNION_WXYZ,
            )
        )
        retained_items = tuple(
            item
            for item in self.current_layout_spec.build_items
            if getattr(item, "name", None) not in B14_REMOVED_LAYOUT_ITEMS
        )
        box = ArticulatedAssetSpec(
            name=B14_BOX_ENTITY_NAME,
            asset_path=partnet_mobility_modified_model_urdf_path(B14_BOX_MODEL_ID),
            pose=box_world_pose,
            scale=B14_BOX_SCALE,
            fix_root_link=False,
            load_multiple_collisions_from_file=True,
            # Keep the default collision filter. ManiSkill implements
            # ``disable_self_collisions`` with a shared group-2 bit; the Studio
            # workbench uses that same bit, which would also suppress
            # box-to-workbench contacts and let this free articulation fall
            # through the support surface.
            disable_self_collisions=False,
            initial_qpos=(B14_BOX_CLOSED_QPOS_RAD,),
            joint_friction={B14_LID_JOINT_NAME: B14_HINGE_FRICTION},
            physical_material=B14_BOX_PHYSICAL_MATERIAL,
        )
        return replace(
            self.current_layout_spec,
            build_items=(*retained_items, box),
        )

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        materials = (B14_BOX_PHYSICAL_MATERIAL,) * self.num_envs
        mass_scales = (B14_BOX_MASS_SCALE,) * self.num_envs
        hinge_friction = (B14_HINGE_FRICTION,) * self.num_envs
        hinge_damping = (B14_HINGE_DAMPING,) * self.num_envs

        if self.generalization_axis is GeneralizationAxis.PHYSICAL:
            values = self._generalization_values(
                B14_PHYSICAL_PARAMETER_NAMES,
                cache_name="_b14_physical_values",
            )
            materials = tuple(
                scaled_physical_material(
                    B14_BOX_PHYSICAL_MATERIAL,
                    friction_scale=float(scale),
                    restitution=B14_BOX_PHYSICAL_MATERIAL.restitution,
                )
                for scale in positive_scale(values["box_friction_log_scale"])
            )
            mass_scales = tuple(
                B14_BOX_MASS_SCALE * float(scale)
                for scale in positive_scale(values["box_mass_log_scale"])
            )
            hinge_friction = tuple(float(value) for value in values["hinge_friction"])
            hinge_damping = tuple(float(value) for value in values["hinge_damping"])

        joint_limits = (
            (B14_BOX_CLOSED_QPOS_RAD, B14_BOX_MAX_QPOS_RAD),
        ) * self.num_envs
        return None, {
            B14_BOX_ENTITY_NAME: ArticulationBuildValues(
                physical_materials=materials,
                mass_scales=mass_scales,
                joint_limits={B14_LID_JOINT_NAME: joint_limits},
                joint_friction={B14_LID_JOINT_NAME: hinge_friction},
                joint_damping={B14_LID_JOINT_NAME: hinge_damping},
            )
        }

    def _bind_task_entities(self) -> None:
        self.box = self.layout_scene.require_entity(B14_BOX_ENTITY_NAME)
        try:
            self.lid_joint = self.box.active_joints_map[B14_LID_JOINT_NAME]
            self.base_link = self.box.links_map[B14_BASE_LINK_NAME]
            self.lid_link = self.box.links_map[B14_LID_LINK_NAME]
        except KeyError as exc:
            raise RuntimeError(
                "PartNet-Mobility box 100189 must expose link_1 as its body, "
                "link_0 as its lid, and joint_0 as its only active joint."
            ) from exc
        if self.lid_joint.child_link is not self.lid_link:
            raise RuntimeError(
                "Box joint_0 child link must be the configured lid link_0."
            )
        if self.lid_joint.active_index is None:
            raise RuntimeError("Box joint_0 has no active qpos index.")

        self.workbench = self.layout_scene.require_entity("workbench")
        manipulator_agents = self._manipulator_agents()
        if len(manipulator_agents) != 2:
            raise RuntimeError("B14 requires exactly two manipulator agents.")
        self._inter_arm_contact_query = self.left_agent.prepare_pairwise_contact_query(
            tuple(self.left_agent.robot.links),
            tuple(self.right_agent.robot.links),
        )

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._apply_box_spatial_generalization(env_idx, batch_size)
            self._set_initial_lid_angle(env_idx)
            self._hide_robots()
            self.layout_scene.settle(env_idx=env_idx)
            # Settling seats the free box on the workbench. Restore the exact
            # requested lid angle afterwards because gravity can move the lid.
            self._set_initial_lid_angle(env_idx)
            self._zero_box_root_velocity(env_idx)
            self._reset_robots(batch_size)
            self._initialize_tracking_state(env_idx)

    def _apply_box_spatial_generalization(
        self,
        env_idx: Any,
        batch_size: int,
    ) -> None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return
        values = self._generalization_values(
            B14_SPATIAL_PARAMETER_NAMES,
            cache_name="_b14_spatial_values",
        )
        indices = env_idx.detach().cpu().numpy()
        local_position = torch.zeros(
            (batch_size, 3),
            device=self.device,
            dtype=torch.float32,
        )
        local_position[:, 0] = torch.as_tensor(
            values["box_x_offset_m"][indices],
            device=self.device,
            dtype=torch.float32,
        )
        local_position[:, 1] = torch.as_tensor(
            values["box_y_offset_m"][indices],
            device=self.device,
            dtype=torch.float32,
        )
        yaw = torch.as_tensor(
            values["box_yaw_offset_rad"][indices],
            device=self.device,
            dtype=torch.float32,
        )
        yaw_quaternion = torch.stack(
            (
                torch.cos(yaw / 2.0),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(yaw / 2.0),
            ),
            dim=1,
        )
        # [B, 3] and [B, 4] define a site-local planar pose change.
        local_delta = Pose.create_from_pq(
            p=local_position,
            q=yaw_quaternion,
        )
        current_local_pose = self._site_pose(batch_size).inv() * self.box.pose[env_idx]
        self.box.set_pose(
            self._site_pose(batch_size) * current_local_pose * local_delta
        )

    def _set_initial_lid_angle(self, env_idx: Any) -> None:
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                B14_SPATIAL_PARAMETER_NAMES,
                cache_name="_b14_spatial_values",
            )
            indices = env_idx.detach().cpu().numpy()
            opening_angle_deg = torch.as_tensor(
                values["lid_initial_open_angle_deg"][indices],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            opening_angle_deg = torch.zeros(
                len(env_idx),
                device=self.device,
                dtype=torch.float32,
            )
        qpos = self.box.get_qpos()
        qpos[env_idx, self.lid_joint.active_index] = (
            B14_BOX_CLOSED_QPOS_RAD + torch.deg2rad(opening_angle_deg)
        )
        self.box.set_qpos(qpos)
        qvel = self.box.get_qvel()
        qvel[env_idx] = 0.0
        self.box.set_qvel(qvel)

    def _zero_box_root_velocity(self, env_idx: Any) -> None:
        linear_velocity = self.box.root_linear_velocity.clone()
        angular_velocity = self.box.root_angular_velocity.clone()
        linear_velocity[env_idx] = 0.0
        angular_velocity[env_idx] = 0.0
        self.box.set_root_linear_velocity(linear_velocity)
        self.box.set_root_angular_velocity(angular_velocity)

    def _initialize_tracking_state(self, env_idx: Any) -> None:
        if not hasattr(self, "_box_initial_position"):
            self._box_initial_position = self.box.pose.p.clone()
            self._box_initial_up = self._box_up_vector().clone()
            self._workbench_initial_position = self.workbench.pose.p.clone()
            self._box_was_stabilized = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
            self._lid_opening_started = torch.zeros_like(self._box_was_stabilized)
            self._lid_target_reached = torch.zeros_like(self._box_was_stabilized)
            self._box_or_lid_touched = torch.zeros_like(self._box_was_stabilized)
            self._stable_step_count = torch.zeros(
                self.num_envs,
                dtype=torch.int64,
                device=self.device,
            )
            self._left_stabilized_right_opened = torch.zeros_like(
                self._box_was_stabilized
            )
            self._right_stabilized_left_opened = torch.zeros_like(
                self._box_was_stabilized
            )

        self._box_initial_position[env_idx] = self.box.pose.p[env_idx]
        self._box_initial_up[env_idx] = self._box_up_vector()[env_idx]
        self._workbench_initial_position[env_idx] = self.workbench.pose.p[env_idx]
        for state in (
            self._box_was_stabilized,
            self._lid_opening_started,
            self._lid_target_reached,
            self._box_or_lid_touched,
            self._left_stabilized_right_opened,
            self._right_stabilized_left_opened,
        ):
            state[env_idx] = False
        self._stable_step_count[env_idx] = 0

    def _opening_angle_deg(self) -> Any:
        qpos = self.lid_joint.qpos.reshape(self.num_envs, -1)[:, 0]
        return torch.rad2deg(qpos - B14_BOX_CLOSED_QPOS_RAD).clamp(
            0.0,
            B14_BOX_MAX_OPEN_ANGLE_DEG,
        )

    def _box_up_vector(self) -> Any:
        rotation = self.box.pose.to_transformation_matrix()[:, :3, :3]
        local_up = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
            dtype=torch.float32,
        )
        local_up[:, 2] = 1.0
        # [B, 3, 3] x [B, 3] -> [B, 3].
        return torch.einsum("bij,bj->bi", rotation, local_up)

    def _box_displacement_m(self) -> Any:
        return torch.linalg.norm(
            self.box.pose.p - self._box_initial_position,
            dim=1,
        )

    def _box_tilt_deg(self) -> Any:
        cosine = (self._box_up_vector() * self._box_initial_up).sum(dim=1)
        return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))

    def _box_is_stable(self) -> Any:
        return (
            (self._box_displacement_m() <= self.SUCCESS_SPEC.box_displacement_m)
            & (self._box_tilt_deg() <= self.SUCCESS_SPEC.box_tilt_deg)
            & (
                torch.linalg.norm(self.box.root_linear_velocity, dim=1)
                <= self.SUCCESS_SPEC.box_linear_speed_mps
            )
            & (
                torch.linalg.norm(self.box.root_angular_velocity, dim=1)
                <= self.SUCCESS_SPEC.box_angular_speed_rad_s
            )
        )

    def _touch_force_by_agent(self, target_links: tuple[Any, ...]) -> Any:
        target_groups = (("target", target_links),)
        forces_by_agent: list[Any] = []
        for agent in self._manipulator_agents():
            taxel_world_forces = agent.get_object_taxel_world_forces(target_groups)[
                "target"
            ]
            parent_world_forces = agent.aggregate_taxel_world_forces_by_parent(
                taxel_world_forces
            )
            # [B, T, 3] -> [B, P, 3] -> [B]: strongest parent taxel group.
            force_norm = torch.linalg.norm(
                parent_world_forces,
                dim=-1,
            ).amax(dim=1)
            forces_by_agent.append(force_norm)
        if len(forces_by_agent) != 2:
            raise RuntimeError("B14 requires exactly two manipulator agents.")
        # Two [B] vectors -> [B, 2] in embodiment role order.
        return torch.stack(forces_by_agent, dim=1)

    def _base_contact_by_agent(self) -> Any:
        return (
            self._touch_force_by_agent((self.base_link,))
            >= self.SUCCESS_SPEC.contact_force_n
        )

    def _lid_contact_by_agent(self) -> Any:
        return (
            self._touch_force_by_agent((self.lid_link,))
            >= self.SUCCESS_SPEC.contact_force_n
        )

    def _cooperative_contact(self) -> tuple[Any, Any, Any]:
        base_contact = self._base_contact_by_agent()
        lid_contact = self._lid_contact_by_agent()
        left_base_right_lid = base_contact[:, 0] & lid_contact[:, 1]
        right_base_left_lid = base_contact[:, 1] & lid_contact[:, 0]
        cooperative = left_base_right_lid | right_base_left_lid
        return cooperative, left_base_right_lid, right_base_left_lid

    def evaluate(self) -> dict[str, Any]:
        opening_angle_deg = self._opening_angle_deg()
        joint_velocity = self.lid_joint.qvel.reshape(self.num_envs, -1)[:, 0]
        joint_speed = torch.abs(joint_velocity)
        base_contact = self._base_contact_by_agent()
        lid_contact = self._lid_contact_by_agent()
        left_base_right_lid = base_contact[:, 0] & lid_contact[:, 1]
        right_base_left_lid = base_contact[:, 1] & lid_contact[:, 0]
        cooperative_contact = left_base_right_lid | right_base_left_lid
        box_stable = self._box_is_stable()

        self._box_or_lid_touched |= base_contact.any(dim=1) | lid_contact.any(dim=1)
        any_task_contact = base_contact.any(dim=1) | lid_contact.any(dim=1)
        # This stage records that the manipulated box remained controlled. A
        # dedicated support-hand contact is useful but is not mandatory.
        self._box_was_stabilized |= any_task_contact & box_stable
        opening_motion = (
            lid_contact.any(dim=1)
            & box_stable
            & (joint_velocity >= B14_MIN_OPENING_SPEED_RAD_S)
        )
        self._lid_opening_started |= opening_motion & (
            opening_angle_deg >= self.SUCCESS_SPEC.opening_started_angle_deg
        )
        self._left_stabilized_right_opened |= left_base_right_lid & box_stable
        self._right_stabilized_left_opened |= right_base_left_lid & box_stable

        target_angle_current = (
            opening_angle_deg >= self.SUCCESS_SPEC.target_open_angle_deg
        ) & (opening_angle_deg <= self.SUCCESS_SPEC.maximum_success_angle_deg)
        self._lid_target_reached |= (
            self._lid_opening_started & target_angle_current & box_stable
        )
        stable_target_current = (
            self._lid_target_reached
            & target_angle_current
            & box_stable
            & (joint_speed <= self.SUCCESS_SPEC.stable_joint_speed_rad_s)
        )
        stable_steps = max(
            1,
            int(math.ceil(self.SUCCESS_SPEC.stable_dwell_s * self._control_freq)),
        )
        self._stable_step_count = torch.where(
            stable_target_current,
            self._stable_step_count + 1,
            torch.zeros_like(self._stable_step_count),
        )
        success = self._stable_step_count >= stable_steps

        return {
            "success": success,
            # Contact-quality accounting ends once the target angle has been
            # reached; the final stability dwell may legitimately happen
            # after both hands release the box.
            "box_stabilized": self._box_was_stabilized,
            "lid_opening_started": self._lid_opening_started,
            "lid_target_angle_stable": success,
            "box_stable": box_stable,
            "cooperative_contact": cooperative_contact,
            "left_stabilized_right_opened": (self._left_stabilized_right_opened),
            "right_stabilized_left_opened": (self._right_stabilized_left_opened),
            "lid_opening_angle_deg": opening_angle_deg,
            "box_displacement_m": self._box_displacement_m(),
            "box_tilt_deg": self._box_tilt_deg(),
        }

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            finite = torch.isfinite(self.box.pose.raw_pose).all(dim=1) & torch.isfinite(
                self.lid_link.pose.raw_pose
            ).all(dim=1)
            initial_angle_valid = self._opening_angle_deg() <= 5.0
            return finite & initial_angle_valid & self._box_is_stable()
        return super()._check_generalization_validity(axis)
