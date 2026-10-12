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
from dex_benchmark.sim.scenes.layouts import PhysicalMaterialSpec
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.assembly._assembly_common import (
    BimanualAssemblyTaskEnv,
    add_annular_prism,
    add_box,
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


B06_SETTINGS = TaskRegistry.register(TaskRegistry.load("b06_v1"))
B06_PEN_NAME = "b06_uncapped_pen"
B06_CAP_NAME = "b06_pen_cap"
B06_PEN_HOLDER_NAME = "b06_pen_holder"
B06_INTERACTION_SITE_ID = "workbench"
B06_SUPPORT_ENTITY_NAME = "workbench"
B06_REMOVED_LAYOUT_ITEMS = frozenset(
    {
        "workbench_box",
        "clutter_large_marker",
        "clutter_adjustable_wrench",
        "clutter_power_drill",
        "clutter_phillips_screwdriver",
        "workbench_pliers",
        "glasses",
        "clutter_paper_placeholder",
        "stapler",
        "clutter_dice",
    }
)

# The pen and cap use actor-local +Z as their insertion axis. The pen model has
# no cap; its narrow post is the only geometry accepted by the hollow cap.
B06_PEN_BARREL_RADIUS_SCALE = 1.1
B06_PEN_BARREL_RADIUS_M = 0.0090 * B06_PEN_BARREL_RADIUS_SCALE
B06_PEN_HALF_LENGTH_M = 0.093
B06_PEN_CAP_END_LOCAL_Z_M = 0.089
B06_PEN_CAP_POST_RADIUS_M = 0.0068
B06_PEN_TAIL_RADIUS_M = 0.0100
B06_PEN_MAX_RADIUS_M = max(B06_PEN_BARREL_RADIUS_M, B06_PEN_TAIL_RADIUS_M)
B06_CAP_INNER_HALF_WIDTH_M = B06_PEN_CAP_POST_RADIUS_M + 0.0004
B06_CAP_OUTER_HALF_WIDTH_M = B06_PEN_MAX_RADIUS_M
B06_CAP_INNER_HALF_LENGTH_M = 0.0275
B06_CAP_OPEN_LOCAL_Z_M = -B06_CAP_INNER_HALF_LENGTH_M
B06_CAP_INNER_TOP_LOCAL_Z_M = B06_CAP_INNER_HALF_LENGTH_M
B06_CAP_WALL_HALF_THICKNESS_M = 0.5 * (
    B06_CAP_OUTER_HALF_WIDTH_M - B06_CAP_INNER_HALF_WIDTH_M
)
B06_CAP_WALL_CENTER_OFFSET_M = (
    B06_CAP_INNER_HALF_WIDTH_M + B06_CAP_WALL_HALF_THICKNESS_M
)
B06_CAP_WALL_BOXES: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float], str], ...
] = (
    (
        (-B06_CAP_WALL_CENTER_OFFSET_M, 0.0, 0.0),
        (
            B06_CAP_WALL_HALF_THICKNESS_M,
            B06_CAP_OUTER_HALF_WIDTH_M,
            B06_CAP_INNER_HALF_LENGTH_M,
        ),
        "negative_x_wall",
    ),
    (
        (B06_CAP_WALL_CENTER_OFFSET_M, 0.0, 0.0),
        (
            B06_CAP_WALL_HALF_THICKNESS_M,
            B06_CAP_OUTER_HALF_WIDTH_M,
            B06_CAP_INNER_HALF_LENGTH_M,
        ),
        "positive_x_wall",
    ),
    (
        (0.0, -B06_CAP_WALL_CENTER_OFFSET_M, 0.0),
        (
            B06_CAP_INNER_HALF_WIDTH_M,
            B06_CAP_WALL_HALF_THICKNESS_M,
            B06_CAP_INNER_HALF_LENGTH_M,
        ),
        "negative_y_wall",
    ),
    (
        (0.0, B06_CAP_WALL_CENTER_OFFSET_M, 0.0),
        (
            B06_CAP_INNER_HALF_WIDTH_M,
            B06_CAP_WALL_HALF_THICKNESS_M,
            B06_CAP_INNER_HALF_LENGTH_M,
        ),
        "positive_y_wall",
    ),
)
B06_CAP_CLOSED_END_HALF_THICKNESS_M = 0.002
B06_CAP_MIN_INSERTION_DEPTH_M = 0.015
B06_PEN_HOLDER_INNER_RADIUS_M = 0.013
B06_PEN_HOLDER_OUTER_RADIUS_M = 0.022
B06_PEN_HOLDER_BOTTOM_THICKNESS_M = 0.004
B06_PEN_HOLDER_WALL_TOP_Z_M = 0.060
B06_PEN_HOLDER_WALL_SEGMENTS = 18
B06_INITIAL_PEN_HOLDER_LOCAL_XYZ = (-0.06, 0.14, 0.0)
B06_INITIAL_PEN_LOCAL_XYZ = (
    B06_INITIAL_PEN_HOLDER_LOCAL_XYZ[0],
    B06_INITIAL_PEN_HOLDER_LOCAL_XYZ[1],
    B06_PEN_HOLDER_BOTTOM_THICKNESS_M + B06_PEN_HALF_LENGTH_M + 0.001,
)
B06_INITIAL_CAP_LOCAL_XYZ = (
    -0.06,
    -0.14,
    B06_CAP_INNER_HALF_LENGTH_M,
)
B06_CAP_INITIAL_YAW_OFFSET_RAD = -0.5 * math.pi
B06_PEN_MATERIAL = PhysicalMaterialSpec(1.60, 1.35, 0.01)
B06_CAP_MATERIAL = PhysicalMaterialSpec(2.60, 2.40, 0.01)
B06_PEN_HOLDER_MATERIAL = PhysicalMaterialSpec(1.40, 1.20, 0.0)
B06_DENSITY_KG_M3 = 720.0

B06_GENERALIZATION_SPEC = TaskGeneralizationSpec(
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
                "pen_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "cap_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "object_friction_log_scale", math.log(0.85), math.log(1.20)
            ),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B06SuccessSpec:
    grasp_force_n: float = 0.15
    alignment_lateral_tolerance_m: float = 0.015
    insertion_lateral_tolerance_m: float = 0.010
    alignment_angle_tolerance_rad: float = math.radians(18.0)


B06_SUCCESS_SPEC = B06SuccessSpec()


@register_env(B06_SETTINGS.env_id, max_episode_steps=B06_SETTINGS.max_steps)
class B06Env(BimanualAssemblyTaskEnv):
    """Lift an upright pen from its fixed holder and cap its exposed tip."""

    TASK_SETTINGS = B06_SETTINGS
    GENERALIZATION_SPEC = B06_GENERALIZATION_SPEC
    SUCCESS_SPEC = B06_SUCCESS_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B06_INTERACTION_SITE_ID
    SUPPORT_ENTITY_NAME = B06_SUPPORT_ENTITY_NAME
    REMOVED_LAYOUT_ITEMS = B06_REMOVED_LAYOUT_ITEMS
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.286827, -0.527697, 1.116254),
            quaternion_world_wxyz=(0.678772, -0.203379, 0.204404, 0.675370),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-1.382406, 0.367292, 1.116254),
            quaternion_world_wxyz=(0.015449, -0.288309, 0.004653, 0.957401),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.577207, -1.211649, 1.116254),
            quaternion_world_wxyz=(0.686129, 0.201128, 0.206619, -0.667894),
        ),
    }

    def _build_task_actors(self) -> None:
        self.pen_holder = self._build_pen_holder()
        self.pen = self._build_pen()
        self.cap = self._build_cap()
        friction_scales, pen_mass_scales, cap_mass_scales = (
            self._physical_randomization_values()
        )
        self._apply_actor_physics(
            self.pen,
            base_material=B06_PEN_MATERIAL,
            friction_scales=friction_scales,
            mass_scales=pen_mass_scales,
        )
        self._apply_actor_physics(
            self.cap,
            base_material=B06_CAP_MATERIAL,
            friction_scales=friction_scales,
            mass_scales=cap_mass_scales,
        )

    def _build_pen(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B06_PEN_MATERIAL)
        barrel_visual = render_material((0.08, 0.10, 0.12, 1.0), roughness=0.32)
        tip_visual = render_material((0.16, 0.20, 0.24, 1.0), roughness=0.28)
        for center_z, radius, half_height, visual, name in (
            (-0.008, B06_PEN_BARREL_RADIUS_M, 0.075, barrel_visual, "barrel"),
            (0.077, B06_PEN_CAP_POST_RADIUS_M, 0.012, tip_visual, "cap_post"),
            (-0.087, B06_PEN_TAIL_RADIUS_M, 0.006, tip_visual, "tail"),
        ):
            add_z_cylinder(
                builder,
                center_xyz=(0.0, 0.0, center_z),
                radius_m=radius,
                half_height_m=half_height,
                collision_material=collision,
                visual_material=visual,
                density_kg_m3=B06_DENSITY_KG_M3,
                name=f"b06_pen_{name}",
            )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_dynamic(name=B06_PEN_NAME)

    def _build_cap(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B06_CAP_MATERIAL)
        visual = render_material((0.10, 0.34, 0.72, 1.0), roughness=0.30)
        for center_xyz, half_size_xyz, name in B06_CAP_WALL_BOXES:
            add_box(
                builder,
                center_xyz=center_xyz,
                half_size_xyz=half_size_xyz,
                collision_material=collision,
                visual_material=visual,
                density_kg_m3=B06_DENSITY_KG_M3,
                name=f"b06_cap_{name}",
            )
        add_box(
            builder,
            center_xyz=(
                0.0,
                0.0,
                B06_CAP_INNER_HALF_LENGTH_M
                + B06_CAP_CLOSED_END_HALF_THICKNESS_M,
            ),
            half_size_xyz=(
                B06_CAP_OUTER_HALF_WIDTH_M,
                B06_CAP_OUTER_HALF_WIDTH_M,
                B06_CAP_CLOSED_END_HALF_THICKNESS_M,
            ),
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B06_DENSITY_KG_M3,
            name="b06_cap_closed_end",
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_dynamic(name=B06_CAP_NAME)

    def _build_pen_holder(self) -> Any:
        builder = self.scene.create_actor_builder()
        collision = physical_material(B06_PEN_HOLDER_MATERIAL)
        visual = render_material((0.20, 0.23, 0.27, 1.0), roughness=0.62)
        add_z_cylinder(
            builder,
            center_xyz=(0.0, 0.0, 0.5 * B06_PEN_HOLDER_BOTTOM_THICKNESS_M),
            radius_m=B06_PEN_HOLDER_OUTER_RADIUS_M,
            half_height_m=0.5 * B06_PEN_HOLDER_BOTTOM_THICKNESS_M,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B06_DENSITY_KG_M3,
            name="b06_pen_holder_bottom",
        )
        wall_lower_z = B06_PEN_HOLDER_BOTTOM_THICKNESS_M
        add_annular_prism(
            builder,
            center_z_m=0.5 * (wall_lower_z + B06_PEN_HOLDER_WALL_TOP_Z_M),
            inner_radius_m=B06_PEN_HOLDER_INNER_RADIUS_M,
            outer_radius_m=B06_PEN_HOLDER_OUTER_RADIUS_M,
            half_height_m=0.5 * (
                B06_PEN_HOLDER_WALL_TOP_Z_M - wall_lower_z
            ),
            segment_count=B06_PEN_HOLDER_WALL_SEGMENTS,
            collision_material=collision,
            visual_material=visual,
            density_kg_m3=B06_DENSITY_KG_M3,
            name_prefix="b06_pen_holder_wall",
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_kinematic(name=B06_PEN_HOLDER_NAME)

    def _physical_randomization_values(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ones = np.ones(self.num_envs, dtype=np.float64)
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return ones, ones, ones
        values = self._generalization_values(
            (
                "pen_mass_log_scale",
                "cap_mass_log_scale",
                "object_friction_log_scale",
            ),
            cache_name="_b06_physical_values",
        )
        return (
            positive_scale(values["object_friction_log_scale"]),
            positive_scale(values["pen_mass_log_scale"]),
            positive_scale(values["cap_mass_log_scale"]),
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
                cache_name="_b06_spatial_values",
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

        cap_clockwise_orientation = Pose.create_from_pq(
            q=torch.tensor(
                (
                    math.cos(0.5 * B06_CAP_INITIAL_YAW_OFFSET_RAD),
                    0.0,
                    0.0,
                    math.sin(0.5 * B06_CAP_INITIAL_YAW_OFFSET_RAD),
                ),
                device=self.device,
                dtype=torch.float32,
            ).repeat(batch_size, 1)
        )
        holder_position = transformed_position(B06_INITIAL_PEN_HOLDER_LOCAL_XYZ)
        self.pen_holder.set_pose(
            site_pose
            * Pose.create_from_pq(p=holder_position, q=yaw_quaternion)
        )

        pen_position = transformed_position(B06_INITIAL_PEN_LOCAL_XYZ)
        # [B, 7] * [B, 7] -> [B, 7]: actor-local +Z stays upright, so the pen
        # tip points upward while the spatial-randomization yaw rotates its XY.
        self.pen.set_pose(
            site_pose * Pose.create_from_pq(p=pen_position, q=yaw_quaternion)
        )
        self._zero_actor_velocity(self.pen, batch_size)

        cap_position = transformed_position(B06_INITIAL_CAP_LOCAL_XYZ)
        # [B, 7] * [B, 7] -> [B, 7]: actor-local -Z is the cap opening, so
        # keeping local +Z upright places the cap vertically with its opening down.
        cap_local_pose = Pose.create_from_pq(
            p=cap_position,
            q=yaw_quaternion,
        ) * cap_clockwise_orientation
        self.cap.set_pose(site_pose * cap_local_pose)
        self._zero_actor_velocity(self.cap, batch_size)

        self.layout_scene.settle(env_idx=env_idx)
        self._pen_ever_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._cap_ever_grasped = torch.zeros_like(self._pen_ever_grasped)
        self._opposite_hands_ever = torch.zeros_like(self._pen_ever_grasped)
        self._reset_robots(batch_size)

    def _assembly_geometry(self) -> tuple[Any, Any, Any, Any]:
        axis_cosine = torch.sum(
            local_z_axis_world(self.pen.pose) * local_z_axis_world(self.cap.pose),
            dim=1,
        )
        pen_tip_world = transform_local_point(
            self.pen.pose,
            (0.0, 0.0, B06_PEN_CAP_END_LOCAL_Z_M),
            device=self.device,
            batch_size=self.num_envs,
        )
        pen_tip_in_cap = (
            self.cap.pose.inv() * Pose.create_from_pq(p=pen_tip_world)
        ).p
        lateral_error = torch.linalg.norm(pen_tip_in_cap[:, :2], dim=1)
        insertion_depth = pen_tip_in_cap[:, 2] - B06_CAP_OPEN_LOCAL_Z_M
        return axis_cosine, lateral_error, insertion_depth, pen_tip_in_cap[:, 2]

    def evaluate(self) -> dict[str, Any]:
        pen_grasped, cap_grasped, opposite_hands = self._opposite_hand_grasp(
            self.pen,
            self.cap,
            minimum_force_n=self.SUCCESS_SPEC.grasp_force_n,
        )
        self._pen_ever_grasped |= pen_grasped
        self._cap_ever_grasped |= cap_grasped
        self._opposite_hands_ever |= opposite_hands
        axis_cosine, lateral_error, insertion_depth, tip_local_z = (
            self._assembly_geometry()
        )
        minimum_axis_cosine = math.cos(
            self.SUCCESS_SPEC.alignment_angle_tolerance_rad
        )
        cap_aligned = (
            (axis_cosine >= minimum_axis_cosine)
            & (lateral_error <= self.SUCCESS_SPEC.alignment_lateral_tolerance_m)
            & (tip_local_z >= B06_CAP_OPEN_LOCAL_Z_M - 0.012)
            & (tip_local_z <= B06_CAP_INNER_TOP_LOCAL_Z_M + 0.008)
        )
        cap_fully_covers_tip = (
            (axis_cosine >= minimum_axis_cosine)
            & (lateral_error <= self.SUCCESS_SPEC.insertion_lateral_tolerance_m)
            & (insertion_depth >= B06_CAP_MIN_INSERTION_DEPTH_M)
        )
        pen_capped = cap_fully_covers_tip
        pen_capped = self._latch_task_success(pen_capped)
        return {
            "success": pen_capped,
            "one_assembly_object_grasped": self._pen_ever_grasped | self._cap_ever_grasped,
            "both_assembly_objects_grasped": self._pen_ever_grasped & self._cap_ever_grasped,
            "cap_aligned_with_pen": cap_aligned,
            "pen_capped": pen_capped,
        }
