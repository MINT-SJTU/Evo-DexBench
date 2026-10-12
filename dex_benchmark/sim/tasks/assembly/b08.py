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
    transform_local_point,
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


B08_SETTINGS = TaskRegistry.register(TaskRegistry.load("b08_v1"))
B08_BOTTLE_NAME = "b08_hollow_bottle"
B08_FUNNEL_NAME = "b08_funnel"
B08_FUNNEL_STAND_NAME = "b08_funnel_stand"
B08_INTERACTION_SITE_ID = "counter_top"
B08_LAYOUT_IDS = (
    LayoutId.KITCHEN_L1,
    LayoutId.KITCHEN_L2,
    LayoutId.KITCHEN_L3,
)
# Stable first-person viewpoint centered between the two hands. Camera local +X
# points toward the work area, with positive local pitch looking downward.
B08_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.16, 0.0, 0.55),
    quaternion_wxyz=(
        0.9305989566,
        0.0,
        0.3660406590,
        0.0,
    ),
)
B08_REMOVED_LAYOUT_ITEMS = frozenset(
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

# The bottle collision is hollow all the way through its shoulder and narrow
# neck. Each tuple is (inner radius, outer radius, lower Z, upper Z).
B08_BOTTLE_MOUTH_SCALE = 1.2
B08_BOTTLE_MOUTH_INNER_RADIUS_M = 0.0135 * B08_BOTTLE_MOUTH_SCALE
# Preserve the original 5 mm neck-wall thickness while enlarging the opening.
B08_BOTTLE_MOUTH_OUTER_RADIUS_M = B08_BOTTLE_MOUTH_INNER_RADIUS_M + 0.005
B08_BOTTLE_RING_BANDS: tuple[tuple[float, float, float, float], ...] = (
    (0.0410, 0.0460, 0.006, 0.108),
    (0.0300, 0.0460, 0.104, 0.118),
    (0.0190, 0.0340, 0.114, 0.132),
    (
        B08_BOTTLE_MOUTH_INNER_RADIUS_M,
        B08_BOTTLE_MOUTH_OUTER_RADIUS_M,
        0.128,
        0.176,
    ),
)
B08_BOTTLE_BOTTOM_RADIUS_M = 0.046
B08_BOTTLE_BOTTOM_THICKNESS_M = 0.006
B08_BOTTLE_MOUTH_LOCAL_Z_M = 0.176
B08_BOTTLE_SEGMENTS = 18

B08_FUNNEL_SCALE = 0.9
B08_FUNNEL_SPOUT_RADIUS_M = 0.0095 * B08_FUNNEL_SCALE
B08_FUNNEL_SPOUT_TIP_LOCAL_Z_M = -0.070 * B08_FUNNEL_SCALE
B08_FUNNEL_SPOUT_HALF_LENGTH_M = 0.035 * B08_FUNNEL_SCALE
B08_FUNNEL_CYLINDER_DIAMETER_SCALE = 0.75
B08_FUNNEL_CYLINDER_WALL_THICKNESS_SCALE = 0.5
B08_FUNNEL_CYLINDER_OUTER_RADIUS_M = (
    0.063 * B08_FUNNEL_SCALE * B08_FUNNEL_CYLINDER_DIAMETER_SCALE
)
B08_FUNNEL_CYLINDER_WALL_THICKNESS_M = (
    (0.063 - 0.047)
    * B08_FUNNEL_SCALE
    * B08_FUNNEL_CYLINDER_WALL_THICKNESS_SCALE
)
B08_FUNNEL_CYLINDER_INNER_RADIUS_M = (
    B08_FUNNEL_CYLINDER_OUTER_RADIUS_M
    - B08_FUNNEL_CYLINDER_WALL_THICKNESS_M
)
# The upper funnel is one straight cylindrical wall rather than a stepped or
# sloped bowl. Its height remains unchanged while its diameter and wall
# thickness are reduced. Each tuple is (inner radius, outer radius, lower Z,
# upper Z).
B08_FUNNEL_BOWL_BANDS: tuple[tuple[float, float, float, float], ...] = (
    (
        B08_FUNNEL_CYLINDER_INNER_RADIUS_M,
        B08_FUNNEL_CYLINDER_OUTER_RADIUS_M,
        -0.002 * B08_FUNNEL_SCALE,
        0.076 * B08_FUNNEL_SCALE,
    ),
)
B08_FUNNEL_CONNECTOR_INNER_RADIUS_M = B08_FUNNEL_SPOUT_RADIUS_M
B08_FUNNEL_CONNECTOR_OUTER_RADIUS_M = B08_FUNNEL_BOWL_BANDS[0][1]
B08_FUNNEL_CONNECTOR_HALF_HEIGHT_M = 0.002 * B08_FUNNEL_SCALE
B08_FUNNEL_CONNECTOR_CENTER_Z_M = (
    B08_FUNNEL_BOWL_BANDS[0][2] + B08_FUNNEL_CONNECTOR_HALF_HEIGHT_M
)
B08_FUNNEL_SEGMENTS = 18
B08_FUNNEL_MIN_INSERTION_DEPTH_M = 0.010
B08_FUNNEL_TIP_TARGET_Z_RANGE_M: tuple[float, float] = (
    B08_BOTTLE_MOUTH_LOCAL_Z_M
    - (B08_BOTTLE_MOUTH_LOCAL_Z_M - 0.095) * B08_FUNNEL_SCALE,
    B08_BOTTLE_MOUTH_LOCAL_Z_M - B08_FUNNEL_MIN_INSERTION_DEPTH_M,
)
B08_INITIAL_BOTTLE_LOCAL_XYZ = (-0.02, -0.15, 0.005)
B08_INITIAL_FUNNEL_LOCAL_XYZ = (
    -0.02,
    0.14,
    -B08_FUNNEL_SPOUT_TIP_LOCAL_Z_M + 0.005,
)
B08_FUNNEL_STAND_INNER_RADIUS_M = 0.0125
B08_FUNNEL_STAND_OUTER_RADIUS_M = 0.027
B08_FUNNEL_STAND_HALF_HEIGHT_M = 0.018
B08_BOTTLE_MATERIAL = PhysicalMaterialSpec(1.15, 0.95, 0.01)
B08_FUNNEL_MATERIAL = PhysicalMaterialSpec(1.60, 1.35, 0.01)
B08_STAND_MATERIAL = PhysicalMaterialSpec(1.20, 1.00, 0.0)
B08_DENSITY_KG_M3 = 760.0

B08_GENERALIZATION_SPEC = TaskGeneralizationSpec(
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
                "bottle_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "funnel_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "object_friction_log_scale", math.log(0.85), math.log(1.20)
            ),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B08SuccessSpec:
    grasp_force_n: float = 0.18
    alignment_lateral_tolerance_m: float = 0.018
    insertion_lateral_tolerance_m: float = 0.010
    alignment_angle_tolerance_rad: float = math.radians(15.0)
    stable_relative_linear_speed_mps: float = 0.06
    stable_relative_angular_speed_rad_s: float = 1.0
    stable_steps: int = 3


B08_SUCCESS_SPEC = B08SuccessSpec()


def _b08_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B08_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            B08_INTERACTION_SITE_ID
        )
        robot_base_center_pose = (
            interaction_site.bimanual_robot_base_center_pose
            or interaction_site.robot_base_center_pose
        )
        camera_world_pose = robot_base_center_pose.transform_pose(
            B08_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
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


@register_env(B08_SETTINGS.env_id, max_episode_steps=B08_SETTINGS.max_steps)
class B08Env(BimanualAssemblyTaskEnv):
    """Hold a hollow narrow-neck bottle and insert a separate funnel."""

    TASK_SETTINGS = B08_SETTINGS
    GENERALIZATION_SPEC = B08_GENERALIZATION_SPEC
    SUCCESS_SPEC = B08_SUCCESS_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B08_INTERACTION_SITE_ID
    REMOVED_LAYOUT_ITEMS = B08_REMOVED_LAYOUT_ITEMS
    HUMAN_VIEWER_CAMERA_POSES = _b08_human_viewer_camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Keep the Human Viewer on the operator's first-person free camera."""

        return None

    def _build_task_actors(self) -> None:
        self.bottle = self._build_hollow_bottle()
        self.funnel = self._build_funnel()
        self.funnel_stand = self._build_funnel_stand()
        friction_scales, bottle_mass_scales, funnel_mass_scales = (
            self._physical_randomization_values()
        )
        self._apply_actor_physics(
            self.bottle,
            base_material=B08_BOTTLE_MATERIAL,
            friction_scales=friction_scales,
            mass_scales=bottle_mass_scales,
        )
        self._apply_actor_physics(
            self.funnel,
            base_material=B08_FUNNEL_MATERIAL,
            friction_scales=friction_scales,
            mass_scales=funnel_mass_scales,
        )

    def _build_hollow_bottle(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B08_BOTTLE_MATERIAL)
        visual = render_material((0.18, 0.55, 0.76, 0.88), roughness=0.24)
        add_z_cylinder(
            builder,
            center_xyz=(0.0, 0.0, 0.5 * B08_BOTTLE_BOTTOM_THICKNESS_M),
            radius_m=B08_BOTTLE_BOTTOM_RADIUS_M,
            half_height_m=0.5 * B08_BOTTLE_BOTTOM_THICKNESS_M,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B08_DENSITY_KG_M3,
            name="b08_bottle_bottom",
        )
        for band_index, (inner_radius, outer_radius, lower_z, upper_z) in enumerate(
            B08_BOTTLE_RING_BANDS
        ):
            add_annular_prism(
                builder,
                center_z_m=0.5 * (lower_z + upper_z),
                inner_radius_m=inner_radius,
                outer_radius_m=outer_radius,
                half_height_m=0.5 * (upper_z - lower_z),
                segment_count=B08_BOTTLE_SEGMENTS,
                collision_material=collision,
                visual_material=visual,
                density_kg_m3=B08_DENSITY_KG_M3,
                name_prefix=f"b08_bottle_band_{band_index}",
            )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_dynamic(name=B08_BOTTLE_NAME)

    def _build_funnel(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B08_FUNNEL_MATERIAL)
        visual = render_material((0.93, 0.47, 0.12, 1.0), roughness=0.35)
        add_z_cylinder(
            builder,
            center_xyz=(0.0, 0.0, -B08_FUNNEL_SPOUT_HALF_LENGTH_M),
            radius_m=B08_FUNNEL_SPOUT_RADIUS_M,
            half_height_m=B08_FUNNEL_SPOUT_HALF_LENGTH_M,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B08_DENSITY_KG_M3,
            name="b08_funnel_spout",
        )
        add_annular_prism(
            builder,
            center_z_m=B08_FUNNEL_CONNECTOR_CENTER_Z_M,
            inner_radius_m=B08_FUNNEL_CONNECTOR_INNER_RADIUS_M,
            outer_radius_m=B08_FUNNEL_CONNECTOR_OUTER_RADIUS_M,
            half_height_m=B08_FUNNEL_CONNECTOR_HALF_HEIGHT_M,
            segment_count=B08_FUNNEL_SEGMENTS,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B08_DENSITY_KG_M3,
            name_prefix="b08_funnel_connector",
        )
        for band_index, (inner_radius, outer_radius, lower_z, upper_z) in enumerate(
            B08_FUNNEL_BOWL_BANDS
        ):
            add_annular_prism(
                builder,
                center_z_m=0.5 * (lower_z + upper_z),
                inner_radius_m=inner_radius,
                outer_radius_m=outer_radius,
                half_height_m=0.5 * (upper_z - lower_z),
                segment_count=B08_FUNNEL_SEGMENTS,
                collision_material=collision,
                visual_material=visual,
                density_kg_m3=B08_DENSITY_KG_M3,
                name_prefix=f"b08_funnel_bowl_{band_index}",
            )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_dynamic(name=B08_FUNNEL_NAME)

    def _build_funnel_stand(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B08_STAND_MATERIAL)
        visual = render_material((0.28, 0.30, 0.32, 1.0), roughness=0.62)
        add_annular_prism(
            builder,
            center_z_m=B08_FUNNEL_STAND_HALF_HEIGHT_M + 0.002,
            inner_radius_m=B08_FUNNEL_STAND_INNER_RADIUS_M,
            outer_radius_m=B08_FUNNEL_STAND_OUTER_RADIUS_M,
            half_height_m=B08_FUNNEL_STAND_HALF_HEIGHT_M,
            segment_count=14,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B08_DENSITY_KG_M3,
            name_prefix="b08_funnel_stand",
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_kinematic(name=B08_FUNNEL_STAND_NAME)

    def _physical_randomization_values(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ones = np.ones(self.num_envs, dtype=np.float64)
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return ones, ones, ones
        values = self._generalization_values(
            (
                "bottle_mass_log_scale",
                "funnel_mass_log_scale",
                "object_friction_log_scale",
            ),
            cache_name="_b08_physical_values",
        )
        return (
            positive_scale(values["object_friction_log_scale"]),
            positive_scale(values["bottle_mass_log_scale"]),
            positive_scale(values["funnel_mass_log_scale"]),
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
                cache_name="_b08_spatial_values",
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

        def transformed_position(
            nominal_xyz: tuple[float, float, float],
        ) -> Any:
            nominal_x = torch.full((batch_size,), nominal_xyz[0], device=self.device)
            nominal_y = torch.full((batch_size,), nominal_xyz[1], device=self.device)
            return torch.stack(
                (
                    cos_yaw * nominal_x - sin_yaw * nominal_y + offset_x,
                    sin_yaw * nominal_x + cos_yaw * nominal_y + offset_y,
                    torch.full((batch_size,), nominal_xyz[2], device=self.device),
                ),
                dim=1,
            )

        bottle_position = transformed_position(B08_INITIAL_BOTTLE_LOCAL_XYZ)
        funnel_position = transformed_position(B08_INITIAL_FUNNEL_LOCAL_XYZ)
        stand_position = funnel_position.clone()
        stand_position[:, 2] = 0.0
        self.funnel_stand.set_pose(
            site_pose * Pose.create_from_pq(p=stand_position, q=yaw_quaternion)
        )
        for actor, position in (
            (self.bottle, bottle_position),
            (self.funnel, funnel_position),
        ):
            actor.set_pose(
                site_pose * Pose.create_from_pq(p=position, q=yaw_quaternion)
            )
            self._zero_actor_velocity(actor, batch_size)
        self.layout_scene.settle(env_idx=env_idx)
        self._bottle_ever_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._funnel_ever_grasped = torch.zeros_like(self._bottle_ever_grasped)
        self._aligned_ever = torch.zeros_like(self._bottle_ever_grasped)
        self._stable_insertion_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._reset_robots(batch_size)

    def _insertion_geometry(self) -> tuple[Any, Any, Any, Any]:
        funnel_in_bottle = self.bottle.pose.inv() * self.funnel.pose
        lateral_error = torch.linalg.norm(funnel_in_bottle.p[:, :2], dim=1)
        axis_cosine = torch.sum(
            local_z_axis_world(self.bottle.pose)
            * local_z_axis_world(self.funnel.pose),
            dim=1,
        )
        spout_tip_world = transform_local_point(
            self.funnel.pose,
            (0.0, 0.0, B08_FUNNEL_SPOUT_TIP_LOCAL_Z_M),
            device=self.device,
            batch_size=self.num_envs,
        )
        spout_tip_in_bottle = (
            self.bottle.pose.inv() * Pose.create_from_pq(p=spout_tip_world)
        ).p
        insertion_depth = B08_BOTTLE_MOUTH_LOCAL_Z_M - spout_tip_in_bottle[:, 2]
        return lateral_error, axis_cosine, spout_tip_in_bottle[:, 2], insertion_depth

    def evaluate(self) -> dict[str, Any]:
        bottle_grasped, funnel_grasped, opposite_hands = self._opposite_hand_grasp(
            self.bottle,
            self.funnel,
            minimum_force_n=self.SUCCESS_SPEC.grasp_force_n,
        )
        self._bottle_ever_grasped |= bottle_grasped
        self._funnel_ever_grasped |= funnel_grasped
        lateral_error, axis_cosine, spout_tip_z, insertion_depth = (
            self._insertion_geometry()
        )
        aligned_axes = axis_cosine >= math.cos(
            self.SUCCESS_SPEC.alignment_angle_tolerance_rad
        )
        funnel_aligned = (
            aligned_axes
            & (lateral_error <= self.SUCCESS_SPEC.alignment_lateral_tolerance_m)
            & (spout_tip_z >= B08_BOTTLE_MOUTH_LOCAL_Z_M - 0.030)
            & (spout_tip_z <= B08_BOTTLE_MOUTH_LOCAL_Z_M + 0.025)
        )
        self._aligned_ever |= funnel_aligned
        inserted = (
            aligned_axes
            & (lateral_error <= self.SUCCESS_SPEC.insertion_lateral_tolerance_m)
            & (insertion_depth >= B08_FUNNEL_MIN_INSERTION_DEPTH_M)
            & (spout_tip_z >= B08_FUNNEL_TIP_TARGET_Z_RANGE_M[0])
            & (spout_tip_z <= B08_FUNNEL_TIP_TARGET_Z_RANGE_M[1])
        )
        relative_linear_speed = torch.linalg.norm(
            self.funnel.linear_velocity - self.bottle.linear_velocity, dim=1
        )
        relative_angular_speed = torch.linalg.norm(
            self.funnel.angular_velocity - self.bottle.angular_velocity, dim=1
        )
        stable_insertion = (
            inserted
            & (
                relative_linear_speed
                <= self.SUCCESS_SPEC.stable_relative_linear_speed_mps
            )
            & (
                relative_angular_speed
                <= self.SUCCESS_SPEC.stable_relative_angular_speed_rad_s
            )
        )
        self._stable_insertion_steps = torch.where(
            stable_insertion,
            self._stable_insertion_steps + 1,
            torch.zeros_like(self._stable_insertion_steps),
        )
        funnel_inserted = (
            self._stable_insertion_steps >= self.SUCCESS_SPEC.stable_steps
        )
        funnel_inserted = self._latch_task_success(funnel_inserted)
        return {
            "success": funnel_inserted,
            "one_assembly_object_grasped": self._bottle_ever_grasped | self._funnel_ever_grasped,
            "both_assembly_objects_grasped": self._bottle_ever_grasped & self._funnel_ever_grasped,
            "funnel_aligned_with_bottle_mouth": funnel_aligned,
            "funnel_inserted_in_bottle": funnel_inserted,
        }
