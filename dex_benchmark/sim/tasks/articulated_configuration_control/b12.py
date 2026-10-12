from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

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
    ActorBuildValues,
    ArticulatedAssetSpec,
    ArticulationBuildValues,
    JointPositionLatchSpec,
    PhysicalMaterialSpec,
    StaticBoxSpec,
)
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover
    sapien = None
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls):
            return cls

        return decorator


B12_SETTINGS = TaskRegistry.register(TaskRegistry.load("b12_v1"))
B12_PAPER_NAME = "clutter_paper_placeholder"
# Asset 103301 is a two-link stapler: joint_0 separates the top shell from
# the middle link, while joint_1 rotates that combined assembly about the base.
# Keep the former at its authored neutral position so a press moves both shells.
B12_LOCKED_JOINT_NAME = "joint_0"
B12_PRESS_JOINT_NAME = "joint_1"
B12_LOCKED_JOINT_QPOS_RAD = 0.0
B12_LOCKED_JOINT_LATCH = JointPositionLatchSpec(
    stiffness=1.0e6,
    damping=1.0e4,
    force_limit=100.0,
    relatch=False,
)
B12_PAPER_HALF_SIZE_XY = (0.1485, 0.105)
B12_PAPER_HALF_THICKNESS_M = 0.002
B12_PAPER_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=1.25,
    dynamic_friction=1.15,
    restitution=0.0,
)
B12_STAPLER_BASE_HEIGHT_M = 0.031724
B12_PAPER_SUPPORT_HALF_SIZE_XYZ = (0.045, 0.045, 0.018)
B12_PAPER_SUPPORT_LOCAL_X_OFFSET_M = 0.040
B12_STAPLER_LOCAL_XY = (-0.10, -0.230)
# Paper and its support move together; negative local X is left in the
# configured operator view for this task.
B12_PAPER_INITIAL_LOCAL_XY = (0.0, 0.130)
B12_PAPER_SUPPORT_CLEARANCE_M = 0.002
B12_PAPER_FRONT_OVERHANG_M = (
    B12_PAPER_HALF_SIZE_XY[0]
    + B12_PAPER_SUPPORT_LOCAL_X_OFFSET_M
    - B12_PAPER_SUPPORT_HALF_SIZE_XYZ[0]
)
B12_STAPLE_MARK_PRONG_OFFSET_X_M = 0.006
B12_STAPLE_MARK_PRONG_HALF_SIZE_XYZ = (0.00075, 0.0040, 0.0003)
B12_STAPLE_MARK_SURFACE_CLEARANCE_M = 0.0001
B12_STAPLE_MARK_LOCAL_Z_M = (
    B12_PAPER_HALF_THICKNESS_M
    + B12_STAPLE_MARK_PRONG_HALF_SIZE_XYZ[2]
    + B12_STAPLE_MARK_SURFACE_CLEARANCE_M
)
B12_STAPLE_THROAT_LOCAL_XYZ = (-0.005, 0.000425, -0.011283)
B12_STAPLE_THROAT_HALF_SIZE_XYZ = (0.060, 0.020, 0.004819)
B12_STAPLE_THROAT_SAMPLE_SCALE = 0.90
B12_PAPER_INTERACTION_DISPLACEMENT_M = 0.03
B12_STAPLER_REMOVAL_DISTANCE_M = 0.06
B12_ROBOT_BASE_LOCAL_MIN_XYZ = (-0.71, -0.35, -0.80)
B12_ROBOT_BASE_LOCAL_MAX_XYZ = (-0.44, 0.35, 0.20)
B12_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.00, -1.00, -0.40)
B12_TCP_WORKSPACE_LOCAL_MAX_XYZ = (1.00, 1.00, 1.20)
B12_FINAL_STABLE_STEPS = 5
B12_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("paper_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("paper_y_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("staple_x_offset_m", -0.012, 0.012),
            GeneralizationParameterSpec("staple_y_offset_m", -0.012, 0.012),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("paper_mass_log_scale", math.log(0.8), math.log(1.2)),
            GeneralizationParameterSpec("paper_friction_log_scale", math.log(0.8), math.log(1.25)),
            GeneralizationParameterSpec("stapler_friction", 0.05, 0.20),
            GeneralizationParameterSpec("stapler_damping", 0.08, 0.30),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B12SuccessSpec:
    max_paper_tilt_rad: float = math.radians(35.0)
    final_stable_steps: int = B12_FINAL_STABLE_STEPS
    # Asset 103301 joint_1 has limits [-0.176, 1.803]. In the installed
    # orientation, the positive limit is fully open and qpos≈0 is closed.
    open_angle_rad: float = 1.75
    pressed_angle_rad: float = 0.0
    pressed_angle_tolerance_rad: float = 0.10
    contact_force_n: float = 0.5


@register_env(B12_SETTINGS.env_id, max_episode_steps=B12_SETTINGS.max_steps)
class B12Env(StudioTaskEnv):
    """Staple an inserted sheet, then move the stapler away from the mark."""

    TASK_SETTINGS = B12_SETTINGS
    GENERALIZATION_SPEC = B12_GENERALIZATION_SPEC
    SUCCESS_SPEC = B12SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "workbench"
    ROBOT_BASE_LOCAL_MIN_XYZ = B12_ROBOT_BASE_LOCAL_MIN_XYZ
    ROBOT_BASE_LOCAL_MAX_XYZ = B12_ROBOT_BASE_LOCAL_MAX_XYZ
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.235014, -0.891390, 1.164175),
            quaternion_world_wxyz=(0.680739, -0.182474, 0.181092, 0.685933),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-1.017437, 0.409178, 1.164175),
            quaternion_world_wxyz=(0.009492, -0.257069, 0.002525, 0.966343),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.513800, -0.849797, 1.164175),
            quaternion_world_wxyz=(0.696722, 0.178153, 0.185344, -0.669692),
        ),
    }

    def _task_layout_spec(self) -> Any:
        build_items = tuple(
            replace(
                item,
                half_size_xyz=(
                    B12_PAPER_HALF_SIZE_XY[0],
                    B12_PAPER_HALF_SIZE_XY[1],
                    B12_PAPER_HALF_THICKNESS_M,
                ),
                physical_material=B12_PAPER_PHYSICAL_MATERIAL,
            )
            if isinstance(item, StaticBoxSpec) and item.name == B12_PAPER_NAME
            else replace(
                item,
                fix_root_link=True,
                joint_initial_qpos={
                    **item.joint_initial_qpos,
                    B12_LOCKED_JOINT_NAME: B12_LOCKED_JOINT_QPOS_RAD,
                    B12_PRESS_JOINT_NAME: B12SuccessSpec.open_angle_rad,
                },
                joint_position_latches={
                    **item.joint_position_latches,
                    B12_LOCKED_JOINT_NAME: B12_LOCKED_JOINT_LATCH,
                },
            )
            if isinstance(item, ArticulatedAssetSpec) and item.name == "stapler"
            else item
            for item in self.current_layout_spec.build_items
        )
        return replace(self.current_layout_spec, build_items=build_items)

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            (
                "paper_mass_log_scale",
                "paper_friction_log_scale",
                "stapler_friction",
                "stapler_damping",
            ),
            cache_name="_b12_physical_values",
        )
        paper_materials = tuple(
            scaled_physical_material(
                B12_PAPER_PHYSICAL_MATERIAL,
                friction_scale=float(scale),
                restitution=B12_PAPER_PHYSICAL_MATERIAL.restitution,
            )
            for scale in positive_scale(values["paper_friction_log_scale"])
        )
        return {B12_PAPER_NAME: ActorBuildValues(
            physical_materials=paper_materials,
            mass_scales=tuple(
                float(scale)
                for scale in positive_scale(values["paper_mass_log_scale"])
            ),
        )}, {
            "stapler": ArticulationBuildValues(
                joint_friction={B12_PRESS_JOINT_NAME: tuple(float(v) for v in values["stapler_friction"])},
                joint_damping={B12_PRESS_JOINT_NAME: tuple(float(v) for v in values["stapler_damping"])},
            )
        }

    def _bind_task_entities(self) -> None:
        self.paper = self.layout_scene.require_entity(B12_PAPER_NAME)
        self.stapler = self.layout_scene.require_entity("stapler")
        try:
            self.locked_joint = self.stapler.active_joints_map[B12_LOCKED_JOINT_NAME]
            self.press_joint = self.stapler.active_joints_map[B12_PRESS_JOINT_NAME]
        except KeyError as exc:
            raise RuntimeError(
                "Stapler asset 103301 must expose joint_0 and joint_1."
            ) from exc
        locked_joint_limits = torch.full(
            (self.num_envs, 2),
            B12_LOCKED_JOINT_QPOS_RAD,
            device=self.device,
            dtype=torch.float32,
        )
        self.locked_joint.set_limits(locked_joint_limits)
        self.locked_joint.set_drive_target(
            torch.full(
                (self.num_envs,),
                B12_LOCKED_JOINT_QPOS_RAD,
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.press_link = self.press_joint.child_link
        if self.press_link is None:
            raise RuntimeError("Stapler joint_1 has no child press link.")
        builder = self.scene.create_actor_builder()
        material = sapien.render.RenderMaterial(
            base_color=[0.015, 0.015, 0.020, 1.0],
            roughness=0.35,
        )
        # Two paper-scale staple impressions. They are visual-only and have no
        # collision shape, so feedback cannot alter the sheet dynamics.
        for offset_x in (-B12_STAPLE_MARK_PRONG_OFFSET_X_M, B12_STAPLE_MARK_PRONG_OFFSET_X_M):
            builder.add_box_visual(
                pose=sapien.Pose(p=[offset_x, 0.0, 0.0]),
                half_size=list(B12_STAPLE_MARK_PRONG_HALF_SIZE_XYZ),
                material=material,
            )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        self.staple_mark = builder.build_kinematic(name="b12_staple_mark")

        support_builder = self.scene.create_actor_builder()
        support_half_size = list(B12_PAPER_SUPPORT_HALF_SIZE_XYZ)
        support_builder.add_box_collision(half_size=support_half_size)
        support_builder.add_box_visual(
            half_size=support_half_size,
            material=sapien.render.RenderMaterial(
                base_color=[0.32, 0.36, 0.40, 1.0],
                roughness=0.88,
            ),
        )
        support_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        self.paper_support = support_builder.build_kinematic(
            name="b12_paper_support"
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
            site_pose = self._site_pose(batch_size)
            original_stapler_local = site_pose.inv() * self.stapler.pose[env_idx]
            identity_q = torch.zeros((batch_size, 4), device=self.device)
            identity_q[:, 0] = 1.0
            stapler_q = torch.zeros((batch_size, 4), device=self.device)
            stapler_q[:, 0] = math.sqrt(0.5)
            stapler_q[:, 3] = -math.sqrt(0.5)
            stapler_local = torch.tensor(
                [B12_STAPLER_LOCAL_XY[0], B12_STAPLER_LOCAL_XY[1], 0.0],
                device=self.device,
                dtype=torch.float32,
            ).repeat(batch_size, 1)
            stapler_local[:, 2] = original_stapler_local.p[:, 2]
            self.stapler.set_pose(
                site_pose
                * Pose.create_from_pq(
                    p=stapler_local,
                    q=stapler_q,
                )
            )
            qpos = self.stapler.get_qpos()
            qpos[env_idx, self.locked_joint.active_index] = B12_LOCKED_JOINT_QPOS_RAD
            qpos[env_idx, self.press_joint.active_index] = self.SUCCESS_SPEC.open_angle_rad
            self.stapler.set_qpos(qpos)
            self.stapler.set_qvel(torch.zeros_like(qpos))
            self.locked_joint.set_drive_target(
                torch.full(
                    (batch_size,),
                    B12_LOCKED_JOINT_QPOS_RAD,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            initial_local = torch.tensor(
                [B12_PAPER_INITIAL_LOCAL_XY[0], B12_PAPER_INITIAL_LOCAL_XY[1], 0.0],
                device=self.device,
                dtype=torch.float32,
            ).repeat(batch_size, 1)
            initial_local[:, 2] = 0.004
            if self.generalization_axis is GeneralizationAxis.SPATIAL:
                values = self._generalization_values(
                    ("paper_x_offset_m", "paper_y_offset_m", "staple_x_offset_m", "staple_y_offset_m"),
                    cache_name="_b12_spatial_values",
                )
                indices = env_idx.detach().cpu().numpy()
                initial_local[:, 0] += torch.as_tensor(values["paper_x_offset_m"][indices], device=self.device)
                initial_local[:, 1] += torch.as_tensor(values["paper_y_offset_m"][indices], device=self.device)
            support_local = initial_local.clone()
            support_local[:, 0] += B12_PAPER_SUPPORT_LOCAL_X_OFFSET_M
            support_local[:, 2] = B12_PAPER_SUPPORT_HALF_SIZE_XYZ[2]
            self.paper_support.set_pose(
                self._site_pose(batch_size)
                * Pose.create_from_pq(p=support_local, q=identity_q)
            )
            initial_local[:, 2] = (
                2.0 * B12_PAPER_SUPPORT_HALF_SIZE_XYZ[2]
                + B12_PAPER_HALF_THICKNESS_M
                + B12_PAPER_SUPPORT_CLEARANCE_M
            )
            self.paper.set_pose(self._site_pose(batch_size) * Pose.create_from_pq(p=initial_local, q=identity_q))
            self._zero_actor_velocity(self.paper, batch_size)
            self.staple_mark.set_pose(Pose.create_from_pq(p=torch.zeros((batch_size, 3), device=self.device) - 10.0))
            self._hide_robots()
            self.layout_scene.settle(env_idx=env_idx)
            # Restore the authored neutral/open state after settling so the
            # fixed stapler does not start already pressed.
            qpos = self.stapler.get_qpos()
            qpos[env_idx, self.locked_joint.active_index] = B12_LOCKED_JOINT_QPOS_RAD
            qpos[env_idx, self.press_joint.active_index] = self.SUCCESS_SPEC.open_angle_rad
            self.stapler.set_qpos(qpos)
            self.stapler.set_qvel(torch.zeros_like(qpos))
            self.locked_joint.set_drive_target(
                torch.full(
                    (batch_size,),
                    B12_LOCKED_JOINT_QPOS_RAD,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
            staple_target_in_stapler = torch.tensor(
                B12_STAPLE_THROAT_LOCAL_XYZ,
                device=self.device,
                dtype=torch.float32,
            ).repeat(self.num_envs, 1)
            if self.generalization_axis is GeneralizationAxis.SPATIAL:
                staple_target_in_stapler[env_idx, 0] += torch.as_tensor(
                    values["staple_x_offset_m"][indices], device=self.device
                )
                staple_target_in_stapler[env_idx, 1] += torch.as_tensor(
                    values["staple_y_offset_m"][indices], device=self.device
                )
            self._staple_target_in_stapler = Pose.create_from_pq(
                p=staple_target_in_stapler
            )
            self._reset_robots(batch_size)
            self._paper_was_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._paper_was_positioned = torch.zeros_like(self._paper_was_grasped)
            self._paper_ever_released = torch.zeros_like(self._paper_was_grasped)
            self._mark_created = torch.zeros_like(self._paper_was_grasped)
            self._staple_mark_local = torch.zeros(
                (self.num_envs, 3),
                dtype=torch.float32,
                device=self.device,
            )
            self._stapler_was_removed = torch.zeros_like(self._paper_was_grasped)
            self._completion_stable_step_count = torch.zeros(
                self.num_envs, dtype=torch.int64, device=self.device
            )
            self._paper_initial_position = self.paper.pose.p.clone()
            self._stapler_initial_pose = self.stapler.pose.raw_pose.clone()

    def _stapler_point_in_paper_frame(self) -> Any:
        staple_target_world = self.stapler.pose * self._staple_target_in_stapler
        return (self.paper.pose.inv() * staple_target_world).p

    def _staple_mark_world_pose(self) -> Pose:
        return self.paper.pose * Pose.create_from_pq(
            p=self._staple_mark_local
        )

    def _update_staple_mark_pose(self) -> None:
        mark_world_pose = self._staple_mark_world_pose()
        hidden = torch.full_like(mark_world_pose.p, -10.0)
        mark_world_position = torch.where(
            self._mark_created[:, None], mark_world_pose.p, hidden
        )
        self.staple_mark.set_pose(
            Pose.create_from_pq(p=mark_world_position, q=mark_world_pose.q)
        )

    def _paper_under_stapler(self) -> Any:
        stapler_to_paper = (
            self.paper.pose.inv() * self.stapler.pose
        ).to_transformation_matrix()
        center = self._staple_target_in_stapler.p
        half_size = torch.as_tensor(
            B12_STAPLE_THROAT_HALF_SIZE_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        sample_xy = torch.tensor(
            [
                [x, y, 0.0]
                for x in (-1.0, 0.0, 1.0)
                for y in (-1.0, 0.0, 1.0)
            ],
            device=self.device,
            dtype=torch.float32,
        )
        sample_offsets = (
            sample_xy
            * half_size
            * B12_STAPLE_THROAT_SAMPLE_SCALE
        )
        samples_in_stapler = center[:, None, :] + sample_offsets[None, :, :]
        samples_in_paper = torch.einsum(
            "bij,bnj->bni",
            stapler_to_paper[:, :3, :3],
            samples_in_stapler,
        ) + stapler_to_paper[:, None, :3, 3]
        half_size = torch.as_tensor(
            B12_PAPER_HALF_SIZE_XY,
            device=self.device,
            dtype=torch.float32,
        )
        within_sheet = (
            torch.abs(samples_in_paper[:, :, :2]) <= half_size
        ).all(dim=2)
        between_anvil_and_press = (
            torch.abs(samples_in_paper[:, :, 2])
            <= B12_PAPER_HALF_THICKNESS_M
            + B12_STAPLE_THROAT_HALF_SIZE_XYZ[2]
        )
        paper_in_stapler = (
            self.stapler.pose.inv() * self.paper.pose
        ).to_transformation_matrix()
        paper_normal_alignment = torch.abs(paper_in_stapler[:, 2, 2])
        orientation_valid = paper_normal_alignment >= math.cos(
            self.SUCCESS_SPEC.max_paper_tilt_rad
        )
        return (within_sheet & between_anvil_and_press).any(dim=1) & orientation_valid

    def _stapler_removed_after_mark(self) -> Any:
        stapling_point_world = self.stapler.pose * self._staple_target_in_stapler
        stapler_to_mark_distance = torch.linalg.norm(
            stapling_point_world.p - self._staple_mark_world_pose().p,
            dim=1,
        )
        return stapler_to_mark_distance >= B12_STAPLER_REMOVAL_DISTANCE_M

    def _paper_moved_from_start(self) -> Any:
        initial_position = getattr(self, "_paper_initial_position", None)
        if initial_position is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            torch.linalg.norm(self.paper.pose.p - initial_position, dim=1)
            >= B12_PAPER_INTERACTION_DISPLACEMENT_M
        )

    def _pressed(self) -> Any:
        return (
            self.press_joint.qpos.reshape(self.num_envs, -1)[:, 0]
            <= (
                self.SUCCESS_SPEC.pressed_angle_rad
                + self.SUCCESS_SPEC.pressed_angle_tolerance_rad
            )
        )

    def evaluate(self) -> dict[str, Any]:
        paper_force_n = self._max_touch_force((self.paper,))
        press_force_n = self._max_touch_force((self.press_link,))
        paper_contact = paper_force_n >= self.SUCCESS_SPEC.contact_force_n
        press_contact = press_force_n >= self.SUCCESS_SPEC.contact_force_n
        positioned = self._paper_under_stapler()
        pressed_by_joint = self._pressed()
        # The recorded real press reached the joint threshold while both taxel
        # attribution queries remained exactly 0 N, so use articulated state as
        # the physical press signal.
        pressed = pressed_by_joint
        mark_created_before = self._mark_created.clone()
        was_grasped_before = self._paper_was_grasped.clone()
        self._paper_was_grasped |= paper_contact | self._paper_moved_from_start()
        self._paper_ever_released |= self._paper_was_grasped & ~paper_contact
        ordered_positioned = positioned & was_grasped_before
        self._paper_was_positioned |= ordered_positioned
        valid_press = self._paper_was_positioned & positioned & pressed
        # The visual result appears on the first evaluation where the sheet is
        # currently in the physical throat and the press joint is closed.
        newly_marked = ~self._mark_created & valid_press
        self._mark_created |= newly_marked
        if newly_marked.any():
            mark_local = self._stapler_point_in_paper_frame()
            mark_half_size = torch.as_tensor(
                B12_PAPER_HALF_SIZE_XY,
                device=self.device,
                dtype=torch.float32,
            ) - 0.01
            mark_local[:, :2] = torch.maximum(
                torch.minimum(mark_local[:, :2], mark_half_size),
                -mark_half_size,
            )
            mark_local[:, 2] = B12_STAPLE_MARK_LOCAL_Z_M
            self._staple_mark_local = torch.where(
                newly_marked[:, None],
                mark_local,
                self._staple_mark_local,
            )
        if self._mark_created.any():
            # Attach the visual-only mark at the actual valid press location
            # so its distance from the stapling point follows the sheet.
            self._update_staple_mark_pose()
        stapler_removed_after_mark = (
            mark_created_before & self._stapler_removed_after_mark()
        )
        self._completion_stable_step_count = torch.where(
            stapler_removed_after_mark,
            self._completion_stable_step_count + 1,
            torch.zeros_like(self._completion_stable_step_count),
        )
        self._stapler_was_removed |= (
            self._completion_stable_step_count
            >= self.SUCCESS_SPEC.final_stable_steps
        )
        success = self._mark_created & self._stapler_was_removed
        return {
            "success": success,
            "paper_grasped": self._paper_was_grasped,
            "paper_under_stapler": self._paper_was_positioned,
            "stapler_pressed": valid_press,
            "staple_mark_created": self._mark_created,
            "stapler_moved_after_mark": success,
            "paper_extracted_with_staple_mark": success,
        }

    def _check_generalization_validity(self, axis: GeneralizationAxis | None) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            return torch.isfinite(self.paper.pose.raw_pose).all(dim=1) & self._actor_static(self.paper, 0.12, 1.0)
        return super()._check_generalization_validity(axis)
