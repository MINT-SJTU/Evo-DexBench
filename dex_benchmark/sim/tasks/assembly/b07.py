from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
    positive_scale,
)
from dex_benchmark.sim.scenes.layouts import LayoutRegistry, PhysicalMaterialSpec, PoseSpec
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.assembly._assembly_common import (
    BimanualAssemblyTaskEnv,
    add_annular_prism,
    add_z_cylinder,
    local_z_axis_world,
    physical_material,
    render_material,
)
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


B07_SETTINGS = TaskRegistry.register(TaskRegistry.load("b07_v1"))
B07_CUP_NAME = "b07_tapered_cup"
B07_SLEEVE_NAME = "b07_cup_sleeve"
B07_INTERACTION_SITE_ID = "counter_top"
B07_LAYOUT_IDS = (
    LayoutId.KITCHEN_L1,
    LayoutId.KITCHEN_L2,
    LayoutId.KITCHEN_L3,
)
# Stable first-person viewpoint centered between the two hands. Camera local +X
# points toward the work area, with positive local pitch looking downward.
B07_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.16, 0.0, 0.55),
    quaternion_wxyz=(
        0.9305989566,
        0.0,
        0.3660406590,
        0.0,
    ),
)
B07_REMOVED_LAYOUT_ITEMS = frozenset(
    {
        "bottle",
        "bottle_blue",
        "bottle_yellow",
        "bottle_fridge",
        "pot",
        "pot_lid",
        "cup1",
        "cup2",
        "spoon",
        "plate",
        "sponge",
    }
)

# Scale every cup and sleeve dimension together so their fit relationship stays
# unchanged. The previous 80% model is reduced by another 10%, giving 72% of
# the original size overall.
B07_OBJECT_SCALE = 0.8 * 0.9

# Each tuple is (inner radius, outer radius, lower Z, upper Z). The cup is
# intentionally narrow at its bottom and wider at its open top.
B07_CUP_BANDS: tuple[tuple[float, float, float, float], ...] = (
    (
        0.028 * B07_OBJECT_SCALE,
        0.033 * B07_OBJECT_SCALE,
        0.004 * B07_OBJECT_SCALE,
        0.066 * B07_OBJECT_SCALE,
    ),
    (
        0.035 * B07_OBJECT_SCALE,
        0.040 * B07_OBJECT_SCALE,
        0.060 * B07_OBJECT_SCALE,
        0.112 * B07_OBJECT_SCALE,
    ),
    (
        0.043 * B07_OBJECT_SCALE,
        0.049 * B07_OBJECT_SCALE,
        0.106 * B07_OBJECT_SCALE,
        0.162 * B07_OBJECT_SCALE,
    ),
)
B07_CUP_BOTTOM_RADIUS_M = 0.033 * B07_OBJECT_SCALE
B07_CUP_BOTTOM_THICKNESS_M = 0.006 * B07_OBJECT_SCALE
B07_CUP_HEIGHT_M = B07_CUP_BANDS[-1][3]
B07_CUP_WALL_SEGMENTS = 16
B07_CUP_COLOR_RGBA = (0.015, 0.015, 0.018, 1.0)
B07_SLEEVE_INNER_RADIUS_M = 0.0415 * B07_OBJECT_SCALE
B07_SLEEVE_OUTER_RADIUS_M = 0.0490 * B07_OBJECT_SCALE
B07_SLEEVE_HALF_HEIGHT_M = 0.0275 * B07_OBJECT_SCALE
B07_SLEEVE_SEGMENTS = 18
B07_TARGET_SLEEVE_CENTER_Z_RANGE_M: tuple[float, float] = (
    0.062 * B07_OBJECT_SCALE,
    0.100 * B07_OBJECT_SCALE,
)
B07_BOTTOM_ENTRY_MAX_CENTER_Z_M = 0.040 * B07_OBJECT_SCALE
B07_INITIAL_CUP_LOCAL_XYZ = (-0.02, 0.14, 0.006)
B07_INITIAL_SLEEVE_LOCAL_XYZ = (
    -0.02,
    -0.15,
    B07_SLEEVE_HALF_HEIGHT_M + 0.006,
)
B07_CUP_MATERIAL = PhysicalMaterialSpec(1.20, 1.00, 0.01)
B07_SLEEVE_MATERIAL = PhysicalMaterialSpec(1.35, 1.15, 0.01)
B07_DENSITY_KG_M3 = 780.0

B07_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("assembly_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("assembly_y_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec(
                "assembly_yaw_offset_rad",
                -math.radians(12.0),
                math.radians(12.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "cup_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "sleeve_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "object_friction_log_scale", math.log(0.85), math.log(1.20)
            ),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B07SuccessSpec:
    grasp_force_n: float = 0.18
    alignment_lateral_tolerance_m: float = 0.018 * B07_OBJECT_SCALE
    fit_lateral_tolerance_m: float = 0.010 * B07_OBJECT_SCALE
    alignment_angle_tolerance_rad: float = math.radians(15.0)


B07_SUCCESS_SPEC = B07SuccessSpec()


def _b07_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B07_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            B07_INTERACTION_SITE_ID
        )
        robot_base_center_pose = (
            interaction_site.bimanual_robot_base_center_pose
            or interaction_site.robot_base_center_pose
        )
        camera_world_pose = robot_base_center_pose.transform_pose(
            B07_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
        )
        camera_quaternion = np.asarray(
            camera_world_pose.quaternion_wxyz,
            dtype=np.float64,
        )
        camera_quaternion /= np.linalg.norm(camera_quaternion)
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=tuple(
                float(value) for value in camera_quaternion
            ),
        )
    return camera_poses


@register_env(B07_SETTINGS.env_id, max_episode_steps=B07_SETTINGS.max_steps)
class B07Env(BimanualAssemblyTaskEnv):
    """Lift a tapered cup and slide a rigid sleeve onto it from below."""

    TASK_SETTINGS = B07_SETTINGS
    GENERALIZATION_SPEC = B07_GENERALIZATION_SPEC
    SUCCESS_SPEC = B07_SUCCESS_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B07_INTERACTION_SITE_ID
    REMOVED_LAYOUT_ITEMS = B07_REMOVED_LAYOUT_ITEMS
    HUMAN_VIEWER_CAMERA_POSES = _b07_human_viewer_camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Keep the Human Viewer on the operator's first-person free camera."""

        return None

    def _build_task_actors(self) -> None:
        self.cup = self._build_tapered_cup()
        self.sleeve = self._build_sleeve()
        friction_scales, cup_mass_scales, sleeve_mass_scales = (
            self._physical_randomization_values()
        )
        self._apply_actor_physics(
            self.cup,
            base_material=B07_CUP_MATERIAL,
            friction_scales=friction_scales,
            mass_scales=cup_mass_scales,
        )
        self._apply_actor_physics(
            self.sleeve,
            base_material=B07_SLEEVE_MATERIAL,
            friction_scales=friction_scales,
            mass_scales=sleeve_mass_scales,
        )

    def _build_tapered_cup(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B07_CUP_MATERIAL)
        body_visual = render_material(B07_CUP_COLOR_RGBA, roughness=0.32)
        add_z_cylinder(
            builder,
            center_xyz=(0.0, 0.0, 0.5 * B07_CUP_BOTTOM_THICKNESS_M),
            radius_m=B07_CUP_BOTTOM_RADIUS_M,
            half_height_m=0.5 * B07_CUP_BOTTOM_THICKNESS_M,
            collision_material=collision,
            visual_material=body_visual,
            density_kg_m3=B07_DENSITY_KG_M3,
            name="b07_cup_bottom",
        )
        for band_index, (inner_radius, outer_radius, lower_z, upper_z) in enumerate(
            B07_CUP_BANDS
        ):
            add_annular_prism(
                builder,
                center_z_m=0.5 * (lower_z + upper_z),
                inner_radius_m=inner_radius,
                outer_radius_m=outer_radius,
                half_height_m=0.5 * (upper_z - lower_z),
                segment_count=B07_CUP_WALL_SEGMENTS,
                collision_material=collision,
                visual_material=body_visual,
                density_kg_m3=B07_DENSITY_KG_M3,
                name_prefix=f"b07_cup_band_{band_index}",
            )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_dynamic(name=B07_CUP_NAME)

    def _build_sleeve(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B07_SLEEVE_MATERIAL)
        visual = render_material((0.52, 0.25, 0.10, 1.0), roughness=0.72)
        add_annular_prism(
            builder,
            center_z_m=0.0,
            inner_radius_m=B07_SLEEVE_INNER_RADIUS_M,
            outer_radius_m=B07_SLEEVE_OUTER_RADIUS_M,
            half_height_m=B07_SLEEVE_HALF_HEIGHT_M,
            segment_count=B07_SLEEVE_SEGMENTS,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B07_DENSITY_KG_M3,
            name_prefix="b07_sleeve",
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_dynamic(name=B07_SLEEVE_NAME)

    def _physical_randomization_values(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ones = np.ones(self.num_envs, dtype=np.float64)
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return ones, ones, ones
        values = self._generalization_values(
            (
                "cup_mass_log_scale",
                "sleeve_mass_log_scale",
                "object_friction_log_scale",
            ),
            cache_name="_b07_physical_values",
        )
        return (
            positive_scale(values["object_friction_log_scale"]),
            positive_scale(values["cup_mass_log_scale"]),
            positive_scale(values["sleeve_mass_log_scale"]),
        )

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        self.layout_scene.initialize_episode(
            env_idx,
            self._batched_episode_rng,
            self.device,
        )
        batch_size = len(env_idx)
        indices = env_idx.detach().cpu().numpy()
        offset_x = torch.zeros(batch_size, device=self.device)
        offset_y = torch.zeros(batch_size, device=self.device)
        yaw = torch.zeros(batch_size, device=self.device)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                (
                    "assembly_x_offset_m",
                    "assembly_y_offset_m",
                    "assembly_yaw_offset_rad",
                ),
                cache_name="_b07_spatial_values",
            )
            offset_x = torch.as_tensor(
                values["assembly_x_offset_m"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            offset_y = torch.as_tensor(
                values["assembly_y_offset_m"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            yaw = torch.as_tensor(
                values["assembly_yaw_offset_rad"][indices],
                device=self.device,
                dtype=torch.float32,
            )
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        yaw_quaternion = torch.stack(
            (
                torch.cos(0.5 * yaw),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(0.5 * yaw),
            ),
            dim=1,
        )
        site_pose = self._site_pose(batch_size)
        for actor, nominal_xyz in (
            (self.cup, B07_INITIAL_CUP_LOCAL_XYZ),
            (self.sleeve, B07_INITIAL_SLEEVE_LOCAL_XYZ),
        ):
            nominal_x = torch.full((batch_size,), nominal_xyz[0], device=self.device)
            nominal_y = torch.full((batch_size,), nominal_xyz[1], device=self.device)
            local_position = torch.stack(
                (
                    cos_yaw * nominal_x - sin_yaw * nominal_y + offset_x,
                    sin_yaw * nominal_x + cos_yaw * nominal_y + offset_y,
                    torch.full((batch_size,), nominal_xyz[2], device=self.device),
                ),
                dim=1,
            )
            actor.set_pose(
                site_pose
                * Pose.create_from_pq(p=local_position, q=yaw_quaternion)
            )
            self._zero_actor_velocity(actor, batch_size)
        self.layout_scene.settle(env_idx=env_idx)
        self._cup_ever_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._sleeve_ever_grasped = torch.zeros_like(self._cup_ever_grasped)
        self._opposite_hands_ever = torch.zeros_like(self._cup_ever_grasped)
        self._sleeve_entered_from_bottom = torch.zeros_like(
            self._cup_ever_grasped
        )
        self._reset_robots(batch_size)

    def _fit_geometry(self) -> tuple[Any, Any, Any]:
        sleeve_in_cup = self.cup.pose.inv() * self.sleeve.pose
        lateral_error = torch.linalg.norm(sleeve_in_cup.p[:, :2], dim=1)
        axis_cosine = torch.abs(
            torch.sum(
                local_z_axis_world(self.cup.pose)
                * local_z_axis_world(self.sleeve.pose),
                dim=1,
            )
        )
        return lateral_error, axis_cosine, sleeve_in_cup.p[:, 2]

    def evaluate(self) -> dict[str, Any]:
        cup_grasped, sleeve_grasped, opposite_hands = self._opposite_hand_grasp(
            self.cup,
            self.sleeve,
            minimum_force_n=self.SUCCESS_SPEC.grasp_force_n,
        )
        self._cup_ever_grasped |= cup_grasped
        self._sleeve_ever_grasped |= sleeve_grasped
        self._opposite_hands_ever |= opposite_hands
        lateral_error, axis_cosine, sleeve_center_z = self._fit_geometry()
        aligned_axes = axis_cosine >= math.cos(
            self.SUCCESS_SPEC.alignment_angle_tolerance_rad
        )
        sleeve_aligned_below = (
            aligned_axes
            & (lateral_error <= self.SUCCESS_SPEC.alignment_lateral_tolerance_m)
            & (sleeve_center_z >= -2.0 * B07_SLEEVE_HALF_HEIGHT_M)
            & (sleeve_center_z <= B07_BOTTOM_ENTRY_MAX_CENTER_Z_M)
        )
        self._sleeve_entered_from_bottom |= sleeve_aligned_below & opposite_hands
        fitted = (
            aligned_axes
            & (lateral_error <= self.SUCCESS_SPEC.fit_lateral_tolerance_m)
            & (sleeve_center_z >= B07_TARGET_SLEEVE_CENTER_Z_RANGE_M[0])
            & (sleeve_center_z <= B07_TARGET_SLEEVE_CENTER_Z_RANGE_M[1])
        )
        sleeve_fitted = self._latch_task_success(fitted)
        return {
            "success": sleeve_fitted,
            "one_assembly_object_grasped": self._cup_ever_grasped | self._sleeve_ever_grasped,
            "both_assembly_objects_grasped": self._cup_ever_grasped & self._sleeve_ever_grasped,
            "sleeve_aligned_below_cup": sleeve_aligned_below,
            "sleeve_fitted_on_cup": sleeve_fitted,
        }
