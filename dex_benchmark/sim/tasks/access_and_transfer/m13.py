from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    InvalidGeneralizationSample,
    TaskGeneralizationSpec,
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.layout_types import LayoutId
from dex_benchmark.sim.scenes.layouts import (
    ActorBuildValues,
    ArticulationBuildValues,
    LayoutBuilder,
    LayoutSpec,
    YCBActorSpec,
)
from dex_benchmark.sim.scenes.layouts.studio import STUDIO_PHYSICAL_MATERIAL
from dex_benchmark.sim.tasks.access_and_transfer.m14 import M14Env
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - depends on the optional simulator stack.
    sapien = None
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls):
            return cls

        return decorator


M13_SETTINGS = TaskRegistry.register(TaskRegistry.load("m13_v1"))


@dataclass(frozen=True)
class M13SuccessSpec:
    cabinet_access_fraction: float = 0.55
    cabinet_closed_enter_fraction: float = 0.05
    cabinet_closed_exit_fraction: float = 0.08
    cabinet_closed_max_speed_mps: float = 0.01
    target_height_tolerance_m: float = 0.07
    ball_static_linear_speed_mps: float = 0.03
    ball_static_angular_speed_rad_s: float = 0.60
    grasp_contact_force_n: float = 0.40
    grasp_contact_count: int = 2
    support_contact_force_n: float = 0.20
    final_stable_steps: int = 5


M13_SUCCESS_SPEC = M13SuccessSpec()
M13_TARGET_DOOR_JOINT_NAME = "joint_2"

# The interaction frame is centered at the cabinet's authored bounding-box pose.
# The shared layout spawns the tennis ball on the upper surface at world Z=0.484 m.
# M13 moves it one shelf down to the surface at world Z=0.124 m; the corresponding
# ball-center references remain exactly 0.36 m apart in the cabinet-local frame.
M13_SOURCE_LOCAL_XYZ = (0.106652, -0.430306, -0.392738)
# Robot-base-frame XYZ offset applied on top of the original Studio placement.
# -X moves the ball backward and +Y moves it left. Adjust only this value when
# tuning M13 reachability.
M13_BALL_INITIAL_ROBOT_OFFSET_XYZ = (-0.10, 0.10, 0.0)
M13_ADJUSTED_SOURCE_LOCAL_XYZ: tuple[float, float, float] = (
    M13_SOURCE_LOCAL_XYZ[0] + M13_BALL_INITIAL_ROBOT_OFFSET_XYZ[0],
    M13_SOURCE_LOCAL_XYZ[1] + M13_BALL_INITIAL_ROBOT_OFFSET_XYZ[1],
    M13_SOURCE_LOCAL_XYZ[2] + M13_BALL_INITIAL_ROBOT_OFFSET_XYZ[2],
)
M13_TARGET_LOCAL_Z = -0.752738
# Link-4's target shelf spans approximately X=[-0.197, 0.205] and
# Y=[-0.570, 0.570] in the interaction-site frame. Joint-2 exposes the bay up
# to Y=0.107. These ball-center bounds inset the shelf/panel geometry by the
# tennis-ball radius plus edge clearance.
M13_TARGET_LOCAL_MIN_XY = (-0.15, -0.52)
M13_TARGET_LOCAL_MAX_XY = (0.16, 0.06)

M13_SPATIAL_PARAMETER_NAMES = (
    "ball_x_offset_m",
    "ball_y_offset_m",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
M13_PHYSICAL_PARAMETER_NAMES = (
    "ball_mass_log_scale",
    "ball_friction_log_scale",
    "ball_restitution",
    "door_friction",
    "door_damping",
)
M13_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("ball_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("ball_y_offset_m", -0.020, 0.020),
            GeneralizationParameterSpec("robot_forward_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec("robot_side_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec(
                "robot_yaw_offset_rad", -math.radians(2.0), math.radians(2.0)
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "ball_mass_log_scale", math.log(0.75), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "ball_friction_log_scale", math.log(0.70), math.log(1.30)
            ),
            GeneralizationParameterSpec("ball_restitution", 0.0, 0.08),
            GeneralizationParameterSpec("door_friction", 0.05, 0.20),
            GeneralizationParameterSpec("door_damping", 0.05, 0.20),
        ),
        requires_rebuild=True,
    ),
)

# Task-local base calibration from the accepted M13 Human Viewer pose. Keeping
# this out of the shared Studio site prevents unrelated cabinet tasks from
# inheriting an M13-specific reachability adjustment.
M13_ROBOT_BASE_LOCAL_XYZ = (-1.0697833, -0.4246822, -0.7920730)
# The evaluation profile expands the task's +/-2 cm spatial offsets to
# +/-3 cm.  Keep a small margin beyond the resulting minimum Y=-0.4546822 m;
# the previous -0.45 m bound rejected one valid fixed-protocol sample.
M13_ROBOT_BASE_LOCAL_MIN_XYZ = (-1.19, -0.46, -0.82)
M13_ROBOT_BASE_LOCAL_MAX_XYZ = (-0.68, -0.22, -0.02)
M13_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.20, -1.20, -1.00)
M13_TCP_WORKSPACE_LOCAL_MAX_XYZ = (1.20, 1.20, 1.30)
M13_FINAL_STABLE_STEPS = 5

# Human Viewer free-camera poses are intentionally listed per layout so each
# scene can be calibrated independently in world coordinates.
M13_HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
    LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
        position_world_xyz=(0.689927, -1.230477, 0.785354),
        quaternion_world_wxyz=(0.455383, 0.165533, 0.086765, -0.870458),
    ),
    # TODO: Replace with the Human Viewer pose recorded in Studio L2.
    LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
        position_world_xyz=(-0.381825, 0.944386, 0.785354),
        quaternion_world_wxyz=(0.873083, -0.085804, 0.166033, 0.450329),
    ),
    # TODO: Replace with the Human Viewer pose recorded in Studio L3.
    LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
        position_world_xyz=(1.931223, 0.952039, 0.785354),
        quaternion_world_wxyz=(0.867804, -0.087723, 0.165027, 0.460421),
    ),
}


@register_env(M13_SETTINGS.env_id, max_episode_steps=M13_SETTINGS.max_steps)
class M13Env(M14Env):
    """Open a sliding cabinet and move its tennis ball down one shelf.

    M14 supplies the shared keyframe-reset and CPU contact aggregation utilities.
    All task semantics, scene bindings, geometry, evaluation, generalization,
    and evaluation checks are overridden for M13.
    """

    TASK_SETTINGS = M13_SETTINGS
    SUCCESS_SPEC = M13_SUCCESS_SPEC
    GENERALIZATION_SPEC = M13_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "sliding_cabinet_a"
    HUMAN_VIEWER_CAMERA_POSES = M13_HUMAN_VIEWER_CAMERA_POSES

    def human_viewer_camera_uid(self) -> str | None:
        """Show the right wrist camera in the Human Viewer camera panel."""

        return "wrist_right_camera"

    def _task_layout_spec(self) -> LayoutSpec:
        """Apply M13's task-local tennis-ball spawn without changing Studio."""

        tennis_ball_placement = self.current_layout_spec.relative_placements[
            "clutter_tennis_ball"
        ]
        tennis_ball_pose = replace(
            tennis_ball_placement.bbox_center_pose,
            position_xyz=(
                tennis_ball_placement.bbox_center_pose.position_xyz[0]
                + M13_BALL_INITIAL_ROBOT_OFFSET_XYZ[0],
                tennis_ball_placement.bbox_center_pose.position_xyz[1]
                + M13_BALL_INITIAL_ROBOT_OFFSET_XYZ[1],
                tennis_ball_placement.bbox_center_pose.position_xyz[2]
                + M13_BALL_INITIAL_ROBOT_OFFSET_XYZ[2],
            ),
        )
        relative_placements = dict(self.current_layout_spec.relative_placements)
        relative_placements["clutter_tennis_ball"] = replace(
            tennis_ball_placement,
            bbox_center_pose=tennis_ball_pose,
        )
        support_surface = self.current_layout_spec.support_surfaces[
            tennis_ball_placement.surface_id
        ]
        tennis_ball_world_pose = support_surface.pose.transform_pose(
            tennis_ball_pose
        )
        build_items = [
            replace(build_item, pose=tennis_ball_world_pose)
            if isinstance(build_item, YCBActorSpec)
            and build_item.name == "clutter_tennis_ball"
            else build_item
            for build_item in self.current_layout_spec.build_items
        ]
        return replace(
            self.current_layout_spec,
            build_items=build_items,
            relative_placements=relative_placements,
        )

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        nominal_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                M13_ROBOT_BASE_LOCAL_XYZ,
                batch_size=1,
            ),
            q=self.interaction_site_quaternion_tensor(batch_size=1),
        )
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return self._validate_robot_base_pose(nominal_pose)

        values = self._generalization_values(
            M13_SPATIAL_PARAMETER_NAMES,
            cache_name="_m13_spatial_values",
        )
        if env_idx is None:
            value_indices = np.arange(len(values["robot_forward_offset_m"]))
        else:
            value_indices = env_idx.detach().cpu().numpy()
        forward_offset = values["robot_forward_offset_m"][value_indices]
        side_offset = values["robot_side_offset_m"][value_indices]
        yaw_offset = values["robot_yaw_offset_rad"][value_indices]
        local_position = np.stack(
            [forward_offset, side_offset, np.zeros_like(forward_offset)], axis=1
        )
        half_yaw = 0.5 * yaw_offset
        local_quaternion = np.stack(
            [
                np.cos(half_yaw),
                np.zeros_like(half_yaw),
                np.zeros_like(half_yaw),
                np.sin(half_yaw),
            ],
            axis=1,
        )
        # [1] * [B] -> [B]: perturb the base in the cabinet interaction frame.
        changed_pose = Pose.create(nominal_pose) * Pose.create_from_pq(
            p=local_position, q=local_quaternion
        )
        return self._validate_robot_base_pose(changed_pose)

    def _validate_robot_base_pose(self, base_pose: Any) -> Any:
        raw_pose = Pose.create(base_pose)
        batch_size = raw_pose.p.shape[0]
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )
        local_position = (site_pose.inv() * raw_pose).p
        lower = torch.as_tensor(
            M13_ROBOT_BASE_LOCAL_MIN_XYZ, device=self.device
        )
        upper = torch.as_tensor(
            M13_ROBOT_BASE_LOCAL_MAX_XYZ, device=self.device
        )
        invalid = ((local_position < lower) | (local_position > upper)).any(dim=1)
        if bool(invalid.any().item()):
            if self.generalization_axis is not None:
                assert self._generalization_seed is not None
                invalid_env_indices = (
                    invalid.nonzero(as_tuple=False)
                    .flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                raise InvalidGeneralizationSample(
                    task_id=self.TASK_SETTINGS.task_id,
                    layout_id=self.current_layout_id_value,
                    axis=self.generalization_axis.value,
                    seed=self._generalization_seed,
                    attempt=self._generalization_attempt,
                    invalid_env_indices=invalid_env_indices,
                )
            raise RuntimeError(
                "M13 robot base is outside cabinet-local limits: "
                f"position={local_position.detach().cpu().tolist()}."
            )
        return raw_pose

    def _load_scene(self, options: dict[str, Any]) -> None:
        actor_values, articulation_values = self._physical_build_values()
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self._task_layout_spec(),
            actor_values=actor_values,
            articulation_values=articulation_values,
        ).build()
        self.cabinet = self.layout_scene.require_entity("sliding_cabinet_a")
        self.tennis_ball = self.layout_scene.require_entity("clutter_tennis_ball")
        self.door_joint = self._require_target_door_joint()
        self.door_link = self.door_joint.child_link
        if self.door_link is None:
            raise RuntimeError(
                f"M13 door joint {self.door_joint.name!r} has no child link."
            )
        self._m13_tactile_target_groups = (
            ("ball", (self.tennis_ball,)),
            ("door", (self.door_link,)),
        )
        if not self.gpu_sim_enabled:
            self.agent.prepare_cpu_object_taxel_query(
                self._m13_tactile_target_groups
            )
        critical_robot_link_names = (
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_hand",
            "rh_forearm",
            "rh_palm",
        )
        self._critical_robot_links = tuple(
            link
            for link_name in critical_robot_link_names
            if (link := self.agent.robot.links_map.get(link_name)) is not None
        )
        if not self._critical_robot_links:
            raise RuntimeError("M13 resolved no critical robot links.")
        self._robot_cabinet_contact_query = (
            self.agent.prepare_pairwise_contact_query(
                self._critical_robot_links,
                tuple(self.cabinet.links),
            )
        )
        cabinet_body_link = self.cabinet.links_map.get("link_4")
        if cabinet_body_link is None:
            raise RuntimeError(
                "M13 cabinet must expose body link 'link_4' for shelf support."
            )
        self._ball_shelf_support_query = self.agent.prepare_pairwise_contact_query(
            (self.tennis_ball,),
            (cabinet_body_link,),
        )
    def _physical_build_values(
        self,
    ) -> tuple[
        dict[str, ActorBuildValues] | None,
        dict[str, ArticulationBuildValues] | None,
    ]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            M13_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m13_physical_values",
        )
        mass_scales = positive_scale(values["ball_mass_log_scale"])
        friction_scales = positive_scale(values["ball_friction_log_scale"])
        materials = tuple(
            scaled_physical_material(
                STUDIO_PHYSICAL_MATERIAL,
                friction_scale=float(friction_scale),
                restitution=float(restitution),
            )
            for friction_scale, restitution in zip(
                friction_scales, values["ball_restitution"], strict=True
            )
        )
        return (
            {
                "clutter_tennis_ball": ActorBuildValues(
                    physical_materials=materials,
                    mass_scales=tuple(float(scale) for scale in mass_scales),
                )
            },
            {
                "sliding_cabinet_a": ArticulationBuildValues(
                    joint_friction={
                        M13_TARGET_DOOR_JOINT_NAME: tuple(
                            float(value) for value in values["door_friction"]
                        )
                    },
                    joint_damping={
                        M13_TARGET_DOOR_JOINT_NAME: tuple(
                            float(value) for value in values["door_damping"]
                        )
                    },
                )
            },
        )

    def _require_target_door_joint(self) -> Any:
        try:
            joint = self.cabinet.active_joints_map[M13_TARGET_DOOR_JOINT_NAME]
        except KeyError as exc:
            known = ", ".join(sorted(self.cabinet.active_joints_map))
            raise RuntimeError(
                f"M13 cabinet has no joint {M13_TARGET_DOOR_JOINT_NAME!r}; "
                f"known active joints: {known}."
            ) from exc
        joint_type = (
            joint.type[0]
            if isinstance(joint.type, (tuple, list))
            else joint.type
        )
        if joint_type != "prismatic":
            raise RuntimeError(
                f"M13 target door must be prismatic, got {joint.type!r}."
            )
        return joint

    def _close_all_cabinet_doors(self, env_idx: Any) -> None:
        joint_limits = self.cabinet.get_qlimits()
        # [B, D, 2] -> [b, D]: initialize every sliding panel at its lower limit.
        closed_qpos = joint_limits[env_idx, :, 0]
        self.cabinet.set_qpos(closed_qpos)
        self.cabinet.set_qvel(torch.zeros_like(closed_qpos))

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            self._invalidate_contact_force_cache()
            batch_size = len(env_idx)
            self._close_all_cabinet_doors(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            if self.generalization_axis is GeneralizationAxis.SPATIAL:
                self._apply_spatial_generalization(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self.agent.robot.set_pose(self._robot_base_pose(env_idx))
            self._set_robot_to_rest_keyframe(batch_size)

            self._ensure_m13_episode_state()
            self._cabinet_was_accessible[env_idx] = False
            self._ball_was_grasped[env_idx] = False
            self._ball_was_released_on_target[env_idx] = False
            self._cabinet_closed_confirmed[env_idx] = False
            self._cabinet_closed_once[env_idx] = False
            self._cabinet_closed_stable_steps[env_idx] = 0
            self._final_stable_step_count[env_idx] = 0
            cabinet_qpos = self.cabinet.get_qpos()
            self._cabinet_initial_qpos[env_idx] = cabinet_qpos[env_idx]

    def _apply_spatial_generalization(self, env_idx: Any) -> None:
        values = self._generalization_values(
            M13_SPATIAL_PARAMETER_NAMES,
            cache_name="_m13_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                local_xyz=(0.0, 0.0, 0.0),
                batch_size=len(value_indices),
            ),
            q=self.interaction_site_quaternion_tensor(len(value_indices)),
        )
        local_ball_pose = site_pose.inv() * self.tennis_ball.pose[env_idx]
        offset = torch.as_tensor(
            np.stack(
                [
                    values["ball_x_offset_m"][value_indices],
                    values["ball_y_offset_m"][value_indices],
                    np.zeros(len(value_indices), dtype=np.float64),
                ],
                axis=1,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        changed_pose = Pose.create_from_pq(
            p=local_ball_pose.p + offset,
            q=local_ball_pose.q,
        )
        self.tennis_ball.set_pose(site_pose * changed_pose)
        zero_velocity = torch.zeros(
            (len(value_indices), 3), device=self.device, dtype=torch.float32
        )
        self.tennis_ball.set_linear_velocity(zero_velocity)
        self.tennis_ball.set_angular_velocity(zero_velocity)

    def _ensure_m13_episode_state(self) -> None:
        if hasattr(self, "_ball_was_grasped"):
            return
        self._ball_was_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._cabinet_was_accessible = torch.zeros_like(self._ball_was_grasped)
        self._cabinet_closed_confirmed = torch.zeros_like(self._ball_was_grasped)
        self._cabinet_closed_once = torch.zeros_like(self._ball_was_grasped)
        self._ball_was_released_on_target = torch.zeros_like(self._ball_was_grasped)
        self._cabinet_closed_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._final_stable_step_count = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._cabinet_initial_qpos = torch.zeros_like(self.cabinet.get_qpos())

    def _invalidate_contact_force_cache(self) -> None:
        self._cached_cpu_contacts: list[Any] | None = None
        self._cached_ball_tactile_force_norms: Any | None = None
        self._cached_ball_support_force: Any | None = None

    def _ball_tactile_force_norms(self) -> Any:
        if self._cached_ball_tactile_force_norms is None:
            self._populate_m13_tactile_force_cache()
        return self._cached_ball_tactile_force_norms

    def _populate_m13_tactile_force_cache(self) -> None:
        taxel_normal_forces = self.agent.get_object_taxel_normal_forces(
            self._m13_tactile_target_groups
        )
        ball_taxel_forces = taxel_normal_forces["ball"]
        # [B, N] -> [B, 5]: a stable grasp requires distinct anatomical
        # fingers, while multiple taxels on one fingertip count together.
        self._cached_ball_tactile_force_norms = (
            self.agent.aggregate_distal_taxels_by_finger(ball_taxel_forces)
        )

    def _cabinet_open_fraction(self) -> Any:
        qpos = self.door_joint.qpos.reshape(self.num_envs, -1)[:, 0]
        limits = self.door_joint.limits.reshape(self.num_envs, -1, 2)[:, 0]
        joint_range = (limits[:, 1] - limits[:, 0]).clamp_min(1e-6)
        return ((qpos - limits[:, 0]) / joint_range).clamp(0.0, 1.0)

    def _door_opening_velocity(self) -> Any:
        return self.door_joint.qvel.reshape(self.num_envs, -1)[:, 0]

    def _cabinet_closed(self) -> Any:
        open_fraction = self._cabinet_open_fraction()
        entered = open_fraction <= self.SUCCESS_SPEC.cabinet_closed_enter_fraction
        exited = open_fraction > self.SUCCESS_SPEC.cabinet_closed_exit_fraction
        self._cabinet_closed_confirmed = torch.where(
            entered,
            torch.ones_like(self._cabinet_closed_confirmed),
            torch.where(
                exited,
                torch.zeros_like(self._cabinet_closed_confirmed),
                self._cabinet_closed_confirmed,
            ),
        )
        return self._cabinet_closed_confirmed

    def _ball_has_shelf_support(self) -> Any:
        if self._cached_ball_support_force is None:
            pair_forces = self.agent.get_pairwise_contact_forces(
                self._ball_shelf_support_query
            )
            # [B, 1, 1, 3] -> [B].
            self._cached_ball_support_force = torch.linalg.norm(
                pair_forces[:, 0, 0, :],
                dim=-1,
            )
        return (
            self._cached_ball_support_force
            >= self.SUCCESS_SPEC.support_contact_force_n
        )

    def _ball_is_stably_grasped(self) -> Any:
        contacts = (
            self._ball_tactile_force_norms()
            >= self.SUCCESS_SPEC.grasp_contact_force_n
        )
        return contacts.sum(dim=1) >= self.SUCCESS_SPEC.grasp_contact_count

    def _ball_local_position(self) -> Any:
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                local_xyz=(0.0, 0.0, 0.0), batch_size=self.num_envs
            ),
            q=self.interaction_site_quaternion_tensor(self.num_envs),
        )
        return (site_pose.inv() * self.tennis_ball.pose).p

    def _ball_on_first_shelf(self) -> Any:
        local_position = self._ball_local_position()
        source = torch.as_tensor(
            M13_ADJUSTED_SOURCE_LOCAL_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        planar_distance = torch.linalg.norm(
            local_position[:, :2] - source[:2], dim=1
        )
        height_error = torch.abs(local_position[:, 2] - source[2])
        return (planar_distance <= 0.22) & (height_error <= 0.08)

    def _ball_at_second_shelf(self) -> Any:
        local_position = self._ball_local_position()
        lower_xy = torch.as_tensor(
            M13_TARGET_LOCAL_MIN_XY,
            device=self.device,
            dtype=torch.float32,
        )
        upper_xy = torch.as_tensor(
            M13_TARGET_LOCAL_MAX_XY,
            device=self.device,
            dtype=torch.float32,
        )
        within_shelf_xy = (
            (local_position[:, :2] >= lower_xy)
            & (local_position[:, :2] <= upper_xy)
        ).all(dim=1)
        height_error = torch.abs(local_position[:, 2] - M13_TARGET_LOCAL_Z)
        return within_shelf_xy & (
            height_error <= self.SUCCESS_SPEC.target_height_tolerance_m
        )

    def _ball_is_static(self) -> Any:
        linear_speed = torch.linalg.norm(self.tennis_ball.linear_velocity, dim=1)
        angular_speed = torch.linalg.norm(self.tennis_ball.angular_velocity, dim=1)
        return (
            linear_speed <= self.SUCCESS_SPEC.ball_static_linear_speed_mps
        ) & (
            angular_speed <= self.SUCCESS_SPEC.ball_static_angular_speed_rad_s
        )

    def _check_generalization_validity(
        self, axis: GeneralizationAxis | None
    ) -> Any:
        if axis is GeneralizationAxis.SPATIAL:
            return self._ball_on_first_shelf() & self._ball_is_static()
        if axis is GeneralizationAxis.PHYSICAL:
            return (self._cabinet_open_fraction() <= 0.02) & self._ball_is_static()
        return super(M14Env, self)._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_m13_episode_state()
        cabinet_accessible = (
            self._cabinet_open_fraction()
            >= self.SUCCESS_SPEC.cabinet_access_fraction
        )
        ball_grasped = self._ball_is_stably_grasped()
        ball_at_second_shelf = self._ball_at_second_shelf()
        ball_stably_placed = (
            ball_at_second_shelf
            & self._ball_has_shelf_support()
            & self._ball_is_static()
            & ~ball_grasped
        )

        self._cabinet_was_accessible |= cabinet_accessible
        ordered_grasp = ball_grasped & self._cabinet_was_accessible
        self._ball_was_grasped |= ordered_grasp
        placement_stable_now = (
            ball_stably_placed
            & self._cabinet_was_accessible
            & self._ball_was_grasped
        )
        self._final_stable_step_count = torch.where(
            placement_stable_now,
            self._final_stable_step_count + 1,
            torch.zeros_like(self._final_stable_step_count),
        )
        placement_confirmed = (
            self._final_stable_step_count >= self.SUCCESS_SPEC.final_stable_steps
        )
        self._ball_was_released_on_target |= placement_confirmed

        door_velocity = self._door_opening_velocity()
        cabinet_closed_now = self._cabinet_closed()
        cabinet_closed_and_static = (
            self._ball_was_released_on_target
            & cabinet_closed_now
            & (
                torch.abs(door_velocity)
                <= self.SUCCESS_SPEC.cabinet_closed_max_speed_mps
            )
        )
        self._cabinet_closed_stable_steps = torch.where(
            cabinet_closed_and_static,
            self._cabinet_closed_stable_steps + 1,
            torch.zeros_like(self._cabinet_closed_stable_steps),
        )
        cabinet_closed_stably = (
            self._cabinet_closed_stable_steps
            >= self.SUCCESS_SPEC.final_stable_steps
        )
        self._cabinet_closed_once |= cabinet_closed_stably
        success = cabinet_closed_stably & ball_stably_placed
        return {
            "success": success,
            "cabinet_accessible": self._cabinet_was_accessible,
            "tennis_ball_grasped": self._ball_was_grasped,
            "tennis_ball_stably_placed": self._ball_was_released_on_target,
            "cabinet_closed": success,
        }
