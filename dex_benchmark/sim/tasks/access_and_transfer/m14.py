from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.envs.base_env import DexVLABenchBaseEnv
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
    LayoutBuilder,
    LayoutBuildItemSpec,
    LayoutRegistry,
    LayoutSpec,
    PhysicalMaterialSpec,
    PoseSpec,
    YCBActorSpec,
)
from dex_benchmark.sim.layout_types import LayoutId
from dex_benchmark.sim.scenes.layouts.builder import physical_material_spec_to_sapien
from dex_benchmark.sim.scenes.layouts.studio import (
    PARTNET_BBOX_CENTER_XYZ,
    PARTNET_BBOX_EXTENT_XYZ,
    STUDIO_DESK_MODEL_ID,
    STUDIO_REMOTE_MODEL_ID,
    STUDIO_REMOTE_MODEL_SCALE,
)
from dex_benchmark.sim.robots.tactile import PairwiseContactQueryHandle
from dex_benchmark.sim.robots.virtual_taxel import (
    transform_world_points_to_local,
)
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


M14_SETTINGS = TaskRegistry.register(TaskRegistry.load("m14_v1"))


@dataclass(frozen=True)
class M14SuccessSpec:
    drawer_access_enter_fraction: float = 0.45
    drawer_access_exit_fraction: float = 0.40
    drawer_closed_enter_fraction: float = 0.05
    drawer_closed_exit_fraction: float = 0.08
    drawer_closed_max_speed_mps: float = 0.01
    drawer_closed_dwell_s: float = 0.15
    desktop_height_tolerance_m: float = 0.06
    banana_static_linear_speed_mps: float = 0.03
    banana_static_angular_speed_rad_s: float = 0.50
    grasp_contact_force_n: float = 0.40
    grasp_min_non_thumb_fingers: int = 1
    grasp_min_fingers_with_palm: int = 2
    grasp_dwell_s: float = 0.15
    drawer_extraction_clearance_m: float = 0.005
    support_contact_force_n: float = 0.20
    placement_dwell_s: float = 0.50
    release_monitor_s: float = 0.75
    release_unstable_linear_speed_mps: float = 0.10
    release_unstable_angular_speed_rad_s: float = 1.50
    release_unstable_dwell_s: float = 0.10


M14_SUCCESS_SPEC = M14SuccessSpec()
# PartNet 30666 semantic id 7 is the left pedestal's top drawer. Its child link
# owns both the drawer collision meshes and handle mesh original-180.obj.
M14_DRAWER_JOINT_NAME: str = "joint_7"
M14_BANANA_LATERAL_OFFSET_LIMIT_M: float = 0.015
M14_BANANA_DEPTH_OFFSET_LIMIT_M: float = 0.030
M14_BANANA_YAW_LIMIT_RAD: float = math.radians(15.0)
# Nominal task-local offsets expressed in the robot-base frame. Robot local +X
# points forward and local -Y points right.
M14_BANANA_BACKWARD_OFFSET_M: float = 0.100
M14_CHAIR_RIGHT_OFFSET_M: float = 0.200
M14_SPATIAL_PARAMETER_NAMES = (
    "banana_x_offset_m",
    "banana_y_offset_m",
    "banana_yaw_offset_rad",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
M14_PHYSICAL_PARAMETER_NAMES = (
    "banana_mass_log_scale",
    "banana_friction_log_scale",
    "banana_restitution",
    "drawer_friction",
    "drawer_damping",
)
M14_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "banana_x_offset_m",
                -M14_BANANA_LATERAL_OFFSET_LIMIT_M,
                M14_BANANA_LATERAL_OFFSET_LIMIT_M,
            ),
            GeneralizationParameterSpec(
                "banana_y_offset_m",
                -M14_BANANA_DEPTH_OFFSET_LIMIT_M,
                M14_BANANA_DEPTH_OFFSET_LIMIT_M,
            ),
            GeneralizationParameterSpec(
                "banana_yaw_offset_rad",
                -M14_BANANA_YAW_LIMIT_RAD,
                M14_BANANA_YAW_LIMIT_RAD,
            ),
            GeneralizationParameterSpec("robot_forward_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec("robot_side_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec(
                "robot_yaw_offset_rad",
                -math.radians(2.0),
                math.radians(2.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "banana_mass_log_scale", math.log(0.75), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "banana_friction_log_scale", math.log(0.70), math.log(1.30)
            ),
            GeneralizationParameterSpec("banana_restitution", 0.0, 0.05),
            GeneralizationParameterSpec("drawer_friction", 0.05, 0.20),
            GeneralizationParameterSpec("drawer_damping", 0.05, 0.20),
        ),
        requires_rebuild=True,
    ),
)

M14_REMOTE_SCALE_MULTIPLIER = 1.32
M14_REMOTE_MODEL_SCALE = STUDIO_REMOTE_MODEL_SCALE * M14_REMOTE_SCALE_MULTIPLIER
M14_REMOTE_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=1.55,
    dynamic_friction=1.45,
    restitution=0.0,
)
M14_BANANA_SCALE_MULTIPLIER = 1.15
M14_BANANA_DENSITY_MULTIPLIER = 0.75
M14_BANANA_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=1.70,
    dynamic_friction=1.60,
    restitution=0.0,
)
# ManiSkill YCB metadata bounds for 011_banana in its actor-local frame.
M14_BANANA_BBOX_MIN_XYZ: tuple[float, float, float] = (
    -0.035197413333955506,
    -0.09939188526640796,
    -0.01887479572529618,
)
M14_BANANA_BBOX_MAX_XYZ: tuple[float, float, float] = (
    0.035197413333955506,
    0.09939188526640796,
    0.01887479572529618,
)

# Studio's shared banana and remote slots are calibrated in the drawer directly
# below M14's target. Lift both task-local contents by the difference between
# the two drawer-bottom top surfaces. The offset is surface-local Z because the
# PartNet drawer-local +Y axis maps to the Studio desk's vertical direction.
M14_LOWER_DRAWER_FLOOR_TOP_Y_M: float = 0.068967
M14_UPPER_DRAWER_FLOOR_TOP_Y_M: float = 0.282636
M14_UPPER_DRAWER_HEIGHT_OFFSET_M: float = (
    M14_UPPER_DRAWER_FLOOR_TOP_Y_M - M14_LOWER_DRAWER_FLOOR_TOP_Y_M
)
M14_UPPER_DRAWER_APERTURE_UNDERSIDE_Y_M: float = 0.351783

# The desk interaction frame is centered on the calibrated PartNet bounding box.
# Its local X/Y dimensions are the full depth/width of the physical tabletop.
M14_DESK_TOP_HALF_EXTENT_XY: tuple[float, float] = (
    0.5 * PARTNET_BBOX_EXTENT_XYZ[STUDIO_DESK_MODEL_ID][0],
    0.5 * PARTNET_BBOX_EXTENT_XYZ[STUDIO_DESK_MODEL_ID][1],
)

# Bounds of desk model 30666 link_7 in the drawer-link frame. The upper Y bound
# remains below the fixed aperture underside so an "inside" banana is also able
# to pass through the opening when the drawer is pulled out.
M14_DRAWER_INTERIOR_MIN_XYZ: tuple[float, float, float] = (
    -0.75,
    0.270,
    -0.20,
)
M14_DRAWER_INTERIOR_MAX_XYZ: tuple[float, float, float] = (
    -0.41,
    0.350,
    0.44,
)

# Bounds of desk model 30666 link_7's handle collision mesh
# (textured_objs/original-180.obj), expressed in the drawer-link frame.
M14_DRAWER_HANDLE_MIN_XYZ: tuple[float, float, float] = (
    -0.601277,
    0.296586,
    0.425005,
)
M14_DRAWER_HANDLE_MAX_XYZ: tuple[float, float, float] = (
    -0.563669,
    0.334194,
    0.450581,
)
M14_DRAWER_HANDLE_BOUNDS_TOLERANCE_M = 1e-3
M14_DRAWER_HANDLE_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=1.25,
    dynamic_friction=1.15,
    restitution=0.0,
)
M14_DRAWER_TARGET_MASS_KG = 4.0
M14_DRAWER_NOMINAL_DAMPING = 0.01
M14_DRAWER_NOMINAL_FRICTION = 0.20


# Recorded in Studio L1 and stored in the desk_drawer interaction-site frame.
# Applying the same site-local pose preserves the viewer-to-desk relationship
# in L1/L2/L3 independently of later robot-base calibration changes.
M14_HUMAN_VIEWER_CAMERA_SITE_LOCAL_POSE = PoseSpec(
    position_xyz=(-0.701060, 0.958563, 0.120484),
    quaternion_wxyz=(
        0.8752804393147414,
        0.08472704252561478,
        0.166586083611742,
        -0.4460432238749489,
    ),
)
M14_LAYOUT_IDS = (
    LayoutId.STUDIO_L1,
    LayoutId.STUDIO_L2,
    LayoutId.STUDIO_L3,
)


def _m14_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in M14_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            "desk_drawer"
        )
        camera_world_pose = interaction_site.frame_pose.transform_pose(
            M14_HUMAN_VIEWER_CAMERA_SITE_LOCAL_POSE
        )
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=camera_world_pose.quaternion_wxyz,
        )
    return camera_poses


@register_env(M14_SETTINGS.env_id, max_episode_steps=M14_SETTINGS.max_steps)
class M14Env(DexVLABenchBaseEnv):
    """Open the desk's upper drawer and place its banana on the desk."""

    TASK_SETTINGS = M14_SETTINGS
    SUCCESS_SPEC = M14_SUCCESS_SPEC
    GENERALIZATION_SPEC = M14_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "desk_drawer"
    HUMAN_VIEWER_CAMERA_POSES = _m14_human_viewer_camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Show the right wrist camera in the Human Viewer camera panel."""

        return "wrist_right_camera"

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        nominal_pose = self.interaction_site_robot_base_sapien_pose()
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_pose

        values = self._generalization_values(
            M14_SPATIAL_PARAMETER_NAMES,
            cache_name="_m14_spatial_values",
        )
        if env_idx is None:
            value_indices = np.arange(len(values["robot_forward_offset_m"]))
        else:
            value_indices = env_idx.detach().cpu().numpy()
        forward_offset = values["robot_forward_offset_m"][value_indices]
        side_offset = values["robot_side_offset_m"][value_indices]
        yaw_offset = values["robot_yaw_offset_rad"][value_indices]
        local_position = np.stack(
            [forward_offset, side_offset, np.zeros_like(forward_offset)],
            axis=1,
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
        # [1] * [B] -> [B]: apply a planar base offset in the nominal base frame.
        return Pose.create(nominal_pose) * Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
        )

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _load_scene(self, options: dict[str, Any]) -> None:
        actor_values, articulation_values = self._physical_build_values()
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self._task_layout_spec(),
            actor_values=actor_values,
            articulation_values=articulation_values,
        ).build()
        self.desk = self.layout_scene.require_entity("desk")
        self.banana = self.layout_scene.require_entity("clutter_banana")
        self.drawer_joint = self._require_drawer_joint()
        self.drawer_link = self.drawer_joint.child_link
        if self.drawer_link is None:
            raise RuntimeError(
                f"Drawer joint {self.drawer_joint.name!r} has no child link."
            )
        self._configure_target_drawer_dynamics()
        self._configure_drawer_handle_surface_material()
        self._configure_contact_queries_and_step_constants()

    def _task_layout_spec(self) -> LayoutSpec:
        """Swap the contents and lift them into M14's upper drawer."""

        remote_placement = self.current_layout_spec.relative_placements[
            "remote"
        ]
        banana_placement = self.current_layout_spec.relative_placements[
            "clutter_banana"
        ]
        if remote_placement.surface_id != banana_placement.surface_id:
            raise RuntimeError(
                "M14 can swap banana and remote only when both use the same "
                "support surface."
            )
        remote_surface = self.current_layout_spec.support_surfaces[
            remote_placement.surface_id
        ]
        interaction_site = self.current_layout_spec.interaction_site(
            self.INTERACTION_SITE_ID
        )
        robot_rotation_world = (
            interaction_site.robot_base_center_pose.rotation_matrix.astype(
                np.float64
            )
        )
        # [3, 3] @ [3] -> [3]: robot-backward to world, then world to the
        # shared drawer support-surface frame.
        banana_backward_world = robot_rotation_world @ np.asarray(
            (-M14_BANANA_BACKWARD_OFFSET_M, 0.0, 0.0),
            dtype=np.float64,
        )
        banana_backward_surface = (
            remote_surface.pose.rotation_matrix.astype(np.float64).T
            @ banana_backward_world
        )
        if not math.isclose(
            float(banana_backward_surface[2]),
            0.0,
            abs_tol=1e-8,
        ):
            raise RuntimeError(
                "M14 banana backward offset must stay parallel to the drawer "
                "support surface."
            )
        # [3, 3] @ [3] -> [3]: robot-right is local -Y in world coordinates.
        chair_right_world = robot_rotation_world @ np.asarray(
            (0.0, -M14_CHAIR_RIGHT_OFFSET_M, 0.0),
            dtype=np.float64,
        )
        original_bbox_center_pose = remote_placement.bbox_center_pose
        remote_local_orientation = PoseSpec(
            position_xyz=(0.0, 0.0, 0.0),
            quaternion_wxyz=original_bbox_center_pose.quaternion_wxyz,
        )
        remote_bbox_half_extent = (
            0.5
            * M14_REMOTE_MODEL_SCALE
            * np.asarray(
                PARTNET_BBOX_EXTENT_XYZ[STUDIO_REMOTE_MODEL_ID],
                dtype=np.float64,
            )
        )
        # [3] @ [3] -> scalar: project the enlarged OBB onto shelf-local Z.
        bbox_center_height = float(
            np.abs(remote_local_orientation.rotation_matrix[2])
            @ remote_bbox_half_extent
        )
        remote_bbox_center_pose = replace(
            original_bbox_center_pose,
            position_xyz=(
                banana_placement.bbox_center_pose.position_xyz[0],
                banana_placement.bbox_center_pose.position_xyz[1],
                bbox_center_height
                + remote_placement.initial_clearance_z
                + M14_UPPER_DRAWER_HEIGHT_OFFSET_M,
            ),
        )
        world_bbox_center_pose = remote_surface.pose.transform_pose(
            remote_bbox_center_pose
        )
        bbox_center_offset_world = (
            world_bbox_center_pose.rotation_matrix
            @ (
                M14_REMOTE_MODEL_SCALE
                * np.asarray(
                    PARTNET_BBOX_CENTER_XYZ[STUDIO_REMOTE_MODEL_ID],
                    dtype=np.float64,
                )
            )
        )
        remote_root_position = (
            np.asarray(
                world_bbox_center_pose.position_xyz,
                dtype=np.float64,
            )
            - bbox_center_offset_world
        )
        remote_root_pose = PoseSpec(
            position_xyz=(
                float(remote_root_position[0]),
                float(remote_root_position[1]),
                float(remote_root_position[2]),
            ),
            quaternion_wxyz=world_bbox_center_pose.quaternion_wxyz,
        )

        banana_bbox_center_pose = replace(
            banana_placement.bbox_center_pose,
            position_xyz=(
                original_bbox_center_pose.position_xyz[0]
                + float(banana_backward_surface[0]),
                original_bbox_center_pose.position_xyz[1]
                + float(banana_backward_surface[1]),
                -M14_BANANA_BBOX_MIN_XYZ[2]
                * M14_BANANA_SCALE_MULTIPLIER
                + banana_placement.initial_clearance_z
                + M14_UPPER_DRAWER_HEIGHT_OFFSET_M,
            ),
        )
        banana_world_pose = remote_surface.pose.transform_pose(
            banana_bbox_center_pose
        )

        build_items: list[LayoutBuildItemSpec] = []
        for item in self.current_layout_spec.build_items:
            if isinstance(item, ArticulatedAssetSpec) and item.name == "remote":
                build_items.append(
                    replace(
                        item,
                        scale=M14_REMOTE_MODEL_SCALE,
                        pose=remote_root_pose,
                        physical_material=M14_REMOTE_PHYSICAL_MATERIAL,
                    )
                )
            elif (
                isinstance(item, ArticulatedAssetSpec)
                and item.name == "desk_chair"
            ):
                chair_root_position = np.asarray(
                    item.pose.position_xyz,
                    dtype=np.float64,
                ) + chair_right_world
                build_items.append(
                    replace(
                        item,
                        pose=replace(
                            item.pose,
                            position_xyz=tuple(
                                float(value) for value in chair_root_position
                            ),
                        ),
                    )
                )
            elif isinstance(item, YCBActorSpec) and item.name == "clutter_banana":
                build_items.append(
                    replace(
                        item,
                        pose=banana_world_pose,
                        scale_multiplier=(
                            item.scale_multiplier * M14_BANANA_SCALE_MULTIPLIER
                        ),
                        density_multiplier=(
                            item.density_multiplier
                            * M14_BANANA_DENSITY_MULTIPLIER
                        ),
                        episode_position_jitter_xy=(0.0, 0.0),
                        physical_material=M14_BANANA_PHYSICAL_MATERIAL,
                    )
                )
            else:
                build_items.append(item)
        relative_placements = dict(
            self.current_layout_spec.relative_placements
        )
        relative_placements["remote"] = replace(
            remote_placement,
            bbox_center_pose=remote_bbox_center_pose,
        )
        relative_placements["clutter_banana"] = replace(
            banana_placement,
            bbox_center_pose=banana_bbox_center_pose,
        )
        return replace(
            self.current_layout_spec,
            build_items=build_items,
            relative_placements=relative_placements,
        )

    @staticmethod
    def _rigid_component_for_raw_body(raw_body: Any) -> Any | None:
        candidates = [raw_body]
        candidates.extend(getattr(raw_body, "components", ()))
        raw_entity = getattr(raw_body, "entity", None)
        if raw_entity is not None:
            candidates.extend(getattr(raw_entity, "components", ()))
        return next(
            (
                component
                for component in candidates
                if hasattr(component, "get_collision_shapes")
                or hasattr(component, "collision_shapes")
            ),
            None,
        )

    @staticmethod
    def _collision_shape_local_bounds(
        collision_shape: Any,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        vertices_getter = getattr(collision_shape, "get_vertices", None)
        if not callable(vertices_getter):
            return None
        vertices = np.asarray(vertices_getter(), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return None
        scale_getter = getattr(collision_shape, "get_scale", None)
        scale = (
            np.asarray(scale_getter(), dtype=np.float64)
            if callable(scale_getter)
            else np.ones(3, dtype=np.float64)
        )
        local_pose = collision_shape.get_local_pose()
        local_pose_matrix = np.asarray(
            local_pose.to_transformation_matrix(), dtype=np.float64
        )
        # [N, 3] * [3] @ [3, 3].T -> [N, 3]: mesh to drawer-link frame.
        link_vertices = (
            (vertices * scale) @ local_pose_matrix[:3, :3].T
            + local_pose_matrix[:3, 3]
        )
        return link_vertices.min(axis=0), link_vertices.max(axis=0)

    def _configure_drawer_handle_surface_material(self) -> None:
        """Increase friction only on the target drawer's handle mesh."""

        material = physical_material_spec_to_sapien(
            M14_DRAWER_HANDLE_PHYSICAL_MATERIAL
        )
        expected_lower = np.asarray(
            M14_DRAWER_HANDLE_MIN_XYZ,
            dtype=np.float64,
        )
        expected_upper = np.asarray(
            M14_DRAWER_HANDLE_MAX_XYZ,
            dtype=np.float64,
        )
        tolerance = M14_DRAWER_HANDLE_BOUNDS_TOLERANCE_M
        raw_bodies = tuple(getattr(self.drawer_link, "_bodies", ()))
        if len(raw_bodies) != self.num_envs:
            raise RuntimeError(
                "M14 target drawer link must expose one raw body per "
                f"environment, got {len(raw_bodies)} for {self.num_envs}."
            )
        for env_index, raw_body in enumerate(raw_bodies):
            component = self._rigid_component_for_raw_body(raw_body)
            if component is None:
                raise RuntimeError(
                    "M14 could not resolve the target drawer rigid component "
                    f"for environment {env_index}."
                )
            shape_getter = getattr(component, "get_collision_shapes", None)
            collision_shapes = (
                tuple(shape_getter())
                if callable(shape_getter)
                else tuple(getattr(component, "collision_shapes", ()))
            )
            matched_shapes: list[Any] = []
            for collision_shape in collision_shapes:
                bounds = self._collision_shape_local_bounds(collision_shape)
                if bounds is None:
                    continue
                shape_lower, shape_upper = bounds
                if (
                    (shape_lower >= expected_lower - tolerance).all()
                    and (shape_upper <= expected_upper + tolerance).all()
                ):
                    matched_shapes.append(collision_shape)
            if not matched_shapes:
                raise RuntimeError(
                    "M14 could not identify the target drawer handle collision "
                    f"mesh in environment {env_index}."
                )
            for collision_shape in matched_shapes:
                collision_shape.set_physical_material(material)

    def _configure_target_drawer_dynamics(self) -> None:
        """Set M14-private mass and nominal joint dynamics for the drawer."""

        raw_bodies = tuple(getattr(self.drawer_link, "_bodies", ()))
        if len(raw_bodies) != self.num_envs:
            raise RuntimeError(
                "M14 target drawer link must expose one raw body per "
                f"environment, got {len(raw_bodies)} for {self.num_envs}."
            )
        for env_index, raw_body in enumerate(raw_bodies):
            original_mass = float(raw_body.mass)
            original_inertia = np.asarray(raw_body.inertia, dtype=np.float64)
            if (
                not np.isfinite(original_mass)
                or original_mass <= 0.0
                or original_inertia.shape != (3,)
                or not np.isfinite(original_inertia).all()
                or (original_inertia <= 0.0).any()
            ):
                raise RuntimeError(
                    "M14 target drawer has invalid mass or inertia in "
                    f"environment {env_index}: mass={original_mass}, "
                    f"inertia={original_inertia}."
                )
            mass_scale = M14_DRAWER_TARGET_MASS_KG / original_mass
            raw_body.mass = M14_DRAWER_TARGET_MASS_KG
            # [3] * scalar -> [3]: preserve the mass/inertia relationship.
            raw_body.inertia = original_inertia * mass_scale

        if self.generalization_axis is GeneralizationAxis.PHYSICAL:
            return
        raw_joints = tuple(getattr(self.drawer_joint, "_objs", ()))
        if len(raw_joints) != self.num_envs:
            raise RuntimeError(
                "M14 target drawer joint must expose one raw joint per "
                f"environment, got {len(raw_joints)} for {self.num_envs}."
            )
        for raw_joint in raw_joints:
            raw_joint.set_friction(M14_DRAWER_NOMINAL_FRICTION)
            raw_joint.set_drive_properties(
                float(raw_joint.stiffness),
                M14_DRAWER_NOMINAL_DAMPING,
                float(raw_joint.force_limit),
                raw_joint.drive_mode,
            )

    def _configure_contact_queries_and_step_constants(self) -> None:
        """Cache stable M14 bindings and device constants after scene load."""

        def duration_steps(duration_s: float) -> int:
            if duration_s <= 0.0:
                raise ValueError(
                    f"M14 dwell duration must be positive, got {duration_s}."
                )
            return max(1, math.ceil(duration_s * float(self._control_freq)))

        self._grasp_dwell_steps = duration_steps(
            self.SUCCESS_SPEC.grasp_dwell_s
        )
        self._placement_dwell_steps = duration_steps(
            self.SUCCESS_SPEC.placement_dwell_s
        )
        self._drawer_closed_dwell_steps = duration_steps(
            self.SUCCESS_SPEC.drawer_closed_dwell_s
        )
        self._release_monitor_steps = duration_steps(
            self.SUCCESS_SPEC.release_monitor_s
        )
        self._release_unstable_dwell_steps = duration_steps(
            self.SUCCESS_SPEC.release_unstable_dwell_s
        )
        self._m14_tactile_target_groups = (
            ("banana", (self.banana,)),
            ("drawer", (self.drawer_link,)),
        )
        self._desk_support_link = self.desk.links_map.get("link_9")
        if self._desk_support_link is None:
            raise RuntimeError(
                "M14 requires desk support link 'link_9'."
            )
        parent_group_index_by_name: dict[str, int] = {}
        taxel_parent_group_indices: list[int] = []
        for taxel_spec in self.agent.taxel_specs:
            parent_group_index = parent_group_index_by_name.setdefault(
                taxel_spec.parent_link_name,
                len(parent_group_index_by_name),
            )
            taxel_parent_group_indices.append(parent_group_index)
        self._taxel_parent_group_indices = torch.as_tensor(
            taxel_parent_group_indices,
            device=self.device,
            dtype=torch.int64,
        )
        self._taxel_parent_group_count = len(parent_group_index_by_name)
        self._taxel_finger_indices = self.agent.taxel_finger_indices
        self._finger_taxel_mask = self._taxel_finger_indices >= 0
        self._palm_taxel_mask = ~self._finger_taxel_mask
        if self.drawer_joint.active_index is None:
            raise RuntimeError(
                "M14 target drawer joint must expose an active qpos index."
            )
        drawer_active_indices = (
            self.drawer_joint.active_index.detach().cpu().reshape(-1)
        )
        if (
            drawer_active_indices.numel() == 0
            or not bool(
                (drawer_active_indices == drawer_active_indices[0]).all()
            )
        ):
            raise RuntimeError(
                "M14 target drawer must have one consistent active joint index."
            )
        self._drawer_active_joint_index = int(
            drawer_active_indices[0].item()
        )

        banana_bbox_min = torch.as_tensor(
            M14_BANANA_BBOX_MIN_XYZ,
            device=self.device,
            dtype=torch.float32,
        ) * M14_BANANA_SCALE_MULTIPLIER
        banana_bbox_max = torch.as_tensor(
            M14_BANANA_BBOX_MAX_XYZ,
            device=self.device,
            dtype=torch.float32,
        ) * M14_BANANA_SCALE_MULTIPLIER
        # [3] + [3] -> [3]: target-banana OBB center and half extents.
        self._banana_bbox_center_offset = 0.5 * (
            banana_bbox_min + banana_bbox_max
        )
        banana_bbox_half_extent = 0.5 * (
            banana_bbox_max - banana_bbox_min
        )
        corner_signs = torch.as_tensor(
            [
                (x_sign, y_sign, z_sign)
                for x_sign in (-1.0, 1.0)
                for y_sign in (-1.0, 1.0)
                for z_sign in (-1.0, 1.0)
            ],
            device=self.device,
            dtype=torch.float32,
        )
        # [8, 3] * [3] -> [8, 3]: banana bbox corners around its center.
        self._banana_bbox_corner_offsets = (
            corner_signs * banana_bbox_half_extent
        )
        self._drawer_interior_lower = torch.as_tensor(
            M14_DRAWER_INTERIOR_MIN_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        self._drawer_interior_upper = torch.as_tensor(
            M14_DRAWER_INTERIOR_MAX_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        self._desk_top_half_extent_xy = torch.as_tensor(
            M14_DESK_TOP_HALF_EXTENT_XY,
            device=self.device,
            dtype=torch.float32,
        )

        self._banana_support_query: PairwiseContactQueryHandle = (
            self.agent.prepare_pairwise_contact_query(
                (self.banana,),
                (self._desk_support_link,),
            )
        )
        if not self.gpu_sim_enabled:
            self.agent.prepare_cpu_object_taxel_query(
                self._m14_tactile_target_groups
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
            M14_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m14_physical_values",
        )
        mass_scales = positive_scale(values["banana_mass_log_scale"])
        friction_scales = positive_scale(values["banana_friction_log_scale"])
        materials = tuple(
            scaled_physical_material(
                M14_BANANA_PHYSICAL_MATERIAL,
                friction_scale=float(friction_scale),
                restitution=float(restitution),
            )
            for friction_scale, restitution in zip(
                friction_scales,
                values["banana_restitution"],
                strict=True,
            )
        )
        actor_values = {
            "clutter_banana": ActorBuildValues(
                physical_materials=materials,
                mass_scales=tuple(float(scale) for scale in mass_scales),
            )
        }
        articulation_values = {
            "desk": ArticulationBuildValues(
                joint_friction={
                    M14_DRAWER_JOINT_NAME: tuple(
                        float(value) for value in values["drawer_friction"]
                    )
                },
                joint_damping={
                    M14_DRAWER_JOINT_NAME: tuple(
                        float(value) for value in values["drawer_damping"]
                    )
                },
            ),
        }
        return actor_values, articulation_values

    def _require_drawer_joint(self) -> Any:
        joint_name = M14_DRAWER_JOINT_NAME
        try:
            joint = self.desk.active_joints_map[joint_name]
        except KeyError as exc:
            known_joint_names = ", ".join(sorted(self.desk.active_joints_map))
            raise RuntimeError(
                f"Desk in layout {self.current_layout_id_value!r} does not expose "
                f"M14 drawer joint {joint_name!r}. Known active joints: "
                f"{known_joint_names}."
            ) from exc
        joint_type = (
            joint.type[0]
            if isinstance(joint.type, (tuple, list))
            else joint.type
        )
        if joint_type != "prismatic":
            raise RuntimeError(
                f"M14 drawer joint {joint_name!r} must be prismatic, got "
                f"{joint.type!r}."
            )
        return joint

    def _set_robot_to_rest_keyframe(self, batch_size: int) -> None:
        keyframes = getattr(self.agent, "keyframes", None)
        rest_keyframe = keyframes.get("rest") if keyframes else None
        if rest_keyframe is None or getattr(rest_keyframe, "qpos", None) is None:
            raise RuntimeError(
                f"{self.TASK_SETTINGS.task_id.upper()} requires the configured "
                "robot to expose a 'rest' keyframe."
            )
        robot_qpos = torch.as_tensor(
            rest_keyframe.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        if robot_qpos.ndim == 1:
            robot_qpos = robot_qpos.unsqueeze(0).repeat(batch_size, 1)
        self.agent.robot.set_qpos(robot_qpos)
        self.agent.robot.set_qvel(torch.zeros_like(robot_qpos))
        self.agent.controller.reset()

    def _close_all_desk_drawers(self, env_idx: Any) -> None:
        desk_joint_limits = self.desk.get_qlimits()
        # [B, D, 2] -> [b, D]: initialize every desk drawer at its lower limit.
        closed_qpos = desk_joint_limits[env_idx, :, 0]
        self.desk.set_qpos(closed_qpos)
        self.desk.set_qvel(torch.zeros_like(closed_qpos))

    def _initialize_episode(
        self,
        env_idx: Any,
        options: dict[str, Any],
    ) -> None:
        with torch.device(self.device):
            self._invalidate_contact_force_cache()
            batch_size = len(env_idx)
            self._close_all_desk_drawers(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )

            if self.generalization_axis is GeneralizationAxis.SPATIAL:
                self._apply_spatial_generalization(env_idx)

            # Keep the robot out of the scene while dynamic drawer contents settle.
            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self.agent.robot.set_pose(self._robot_base_pose(env_idx))
            self._set_robot_to_rest_keyframe(batch_size)

            self._ensure_episode_state()
            self._drawer_access_confirmed[env_idx] = False
            self._drawer_was_accessible[env_idx] = False
            self._drawer_closed_confirmed[env_idx] = False
            self._drawer_closed_once[env_idx] = False
            self._drawer_closed_stable_steps[env_idx] = 0
            self._banana_grasp_candidate_steps[env_idx] = 0
            self._banana_was_grasped[env_idx] = False
            self._banana_was_extracted[env_idx] = False
            self._banana_was_released_on_desk[env_idx] = False
            self._release_settling_active[env_idx] = False
            self._release_settling_steps[env_idx] = 0
            self._release_unstable_steps[env_idx] = 0
            self._placement_stable_steps[env_idx] = 0
            self._banana_was_stably_placed[env_idx] = False

    def _apply_spatial_generalization(self, env_idx: Any) -> None:
        values = self._generalization_values(
            M14_SPATIAL_PARAMETER_NAMES,
            cache_name="_m14_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        banana_local_pose = (
            self.drawer_link.pose.inv() * self.banana.pose
        )[env_idx]
        # [B] x 3 -> [B, 3]: layout X/Y map to drawer-local X/Z. Local Y is
        # vertical and stays fixed so randomized bananas remain on the floor.
        position_offset = torch.as_tensor(
            np.stack(
                [
                    values["banana_x_offset_m"][value_indices],
                    np.zeros(len(value_indices), dtype=np.float64),
                    values["banana_y_offset_m"][value_indices],
                ],
                axis=1,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        yaw_offset = torch.as_tensor(
            values["banana_yaw_offset_rad"][value_indices],
            device=self.device,
            dtype=torch.float32,
        )
        half_yaw = 0.5 * yaw_offset
        yaw_quaternion = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
                torch.zeros_like(half_yaw),
            ],
            dim=1,
        )
        changed_local_pose = Pose.create_from_pq(
            p=banana_local_pose.p + position_offset,
            q=yaw_quaternion,
        ) * Pose.create_from_pq(q=banana_local_pose.q)
        self.banana.set_pose(self.drawer_link.pose[env_idx] * changed_local_pose)
        zero_velocity = torch.zeros(
            (len(value_indices), 3),
            device=self.device,
            dtype=torch.float32,
        )
        self.banana.set_linear_velocity(zero_velocity)
        self.banana.set_angular_velocity(zero_velocity)

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_banana_was_grasped"):
            return
        self._drawer_access_confirmed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._banana_was_grasped = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._banana_was_extracted = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._banana_was_stably_placed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._drawer_was_accessible = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._drawer_closed_confirmed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._drawer_closed_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._drawer_closed_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._banana_was_released_on_desk = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._release_settling_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._release_settling_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._banana_grasp_candidate_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._release_unstable_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._placement_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def _invalidate_contact_force_cache(self) -> None:
        """Invalidate values that are stable only within one control step."""

        self._cached_banana_finger_normal_forces: Any | None = None
        self._cached_drawer_finger_normal_forces: Any | None = None
        self._cached_banana_taxel_normal_forces: Any | None = None
        self._cached_drawer_taxel_normal_forces: Any | None = None
        self._cached_banana_center_pose: Any | None = None
        self._cached_banana_bbox_world_corners: Any | None = None
        self._cached_banana_bbox_drawer_corners: Any | None = None
        self._cached_banana_support_force: Any | None = None
        self._cached_drawer_open_fraction: Any | None = None
        self._cached_drawer_opening_velocity: Any | None = None
        self._cached_banana_inside_drawer: Any | None = None
        self._cached_banana_outside_drawer: Any | None = None
        self._cached_banana_static: Any | None = None
        self._cached_banana_on_desk_top: Any | None = None

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self._invalidate_contact_force_cache()

    def _banana_center_pose(self) -> Any:
        cached_pose = getattr(self, "_cached_banana_center_pose", None)
        if cached_pose is None:
            cached_pose = self.banana.pose * Pose.create_from_pq(
                p=self._banana_bbox_center_offset
            )
            self._cached_banana_center_pose = cached_pose
        return cached_pose

    def _drawer_open_fraction(self) -> Any:
        cached_fraction = getattr(self, "_cached_drawer_open_fraction", None)
        if cached_fraction is None:
            drawer_qpos = self.drawer_joint.qpos.reshape(
                self.num_envs,
                -1,
            )[:, 0]
            drawer_limits = self.drawer_joint.limits.reshape(
                self.num_envs,
                -1,
                2,
            )[:, 0]
            lower_limit = drawer_limits[:, 0]
            upper_limit = drawer_limits[:, 1]
            joint_range = (upper_limit - lower_limit).clamp_min(1e-6)
            cached_fraction = (
                (drawer_qpos - lower_limit) / joint_range
            ).clamp(0.0, 1.0)
            self._cached_drawer_open_fraction = cached_fraction
        return cached_fraction

    def _drawer_opening_velocity(self) -> Any:
        """Return drawer joint velocity in its positive/opening direction [B]."""

        cached_velocity = getattr(
            self,
            "_cached_drawer_opening_velocity",
            None,
        )
        if cached_velocity is None:
            cached_velocity = self.drawer_joint.qvel.reshape(
                self.num_envs,
                -1,
            )[:, 0]
            self._cached_drawer_opening_velocity = cached_velocity
        return cached_velocity

    def _aggregate_taxels_by_finger(self, taxel_forces: Any) -> Any:
        """Sum all distal/middle/proximal taxels into five finger forces."""

        expected_shape = (self.num_envs, len(self.agent.taxel_specs))
        if tuple(taxel_forces.shape) != expected_shape:
            raise ValueError(
                "Unexpected M14 taxel-force shape: "
                f"expected {expected_shape}, got {tuple(taxel_forces.shape)}."
            )
        finger_count = 5
        finger_forces = torch.zeros(
            (self.num_envs, finger_count),
            device=taxel_forces.device,
            dtype=taxel_forces.dtype,
        )
        safe_finger_indices = self._taxel_finger_indices.clamp_min(0)
        # [N] -> [B, N], then [B, N] -> [B, 5].
        finger_indices = safe_finger_indices.unsqueeze(0).expand(
            self.num_envs,
            -1,
        )
        finger_forces.scatter_add_(
            dim=1,
            index=finger_indices,
            src=torch.where(
                self._finger_taxel_mask.unsqueeze(0),
                taxel_forces,
                torch.zeros_like(taxel_forces),
            ),
        )
        return finger_forces

    def _palm_taxel_force(self, taxel_forces: Any) -> Any:
        """Return the object-attributed palm/subpalm normal force [B]."""

        return torch.where(
            self._palm_taxel_mask.unsqueeze(0),
            taxel_forces,
            torch.zeros_like(taxel_forces),
        ).sum(dim=1)

    def _banana_finger_normal_forces(self) -> Any:
        if getattr(self, "_cached_banana_finger_normal_forces", None) is None:
            self._populate_tactile_force_cache()
        finger_forces = self._cached_banana_finger_normal_forces
        assert finger_forces is not None
        return finger_forces

    def _drawer_finger_normal_forces(self) -> Any:
        if getattr(self, "_cached_drawer_finger_normal_forces", None) is None:
            self._populate_tactile_force_cache()
        finger_forces = self._cached_drawer_finger_normal_forces
        assert finger_forces is not None
        return finger_forces

    def _drawer_taxel_normal_forces(self) -> Any:
        if getattr(self, "_cached_drawer_taxel_normal_forces", None) is None:
            self._populate_tactile_force_cache()
        taxel_forces = self._cached_drawer_taxel_normal_forces
        assert taxel_forces is not None
        return taxel_forces

    def _banana_taxel_normal_forces(self) -> Any:
        if getattr(self, "_cached_banana_taxel_normal_forces", None) is None:
            self._populate_tactile_force_cache()
        taxel_forces = self._cached_banana_taxel_normal_forces
        assert taxel_forces is not None
        return taxel_forces

    def _populate_tactile_force_cache(self) -> None:
        """Cache object-attributed vectors and all-region finger scalars."""

        target_groups = getattr(
            self,
            "_m14_tactile_target_groups",
            (
                ("banana", (self.banana,)),
                ("drawer", (self.drawer_link,)),
            ),
        )
        object_taxel_normal_forces = self.agent.get_object_taxel_normal_forces(
            target_groups
        )
        self._cached_banana_taxel_normal_forces = (
            object_taxel_normal_forces["banana"]
        )
        self._cached_drawer_taxel_normal_forces = (
            object_taxel_normal_forces["drawer"]
        )
        self._cached_banana_finger_normal_forces = (
            self._aggregate_taxels_by_finger(
                object_taxel_normal_forces["banana"]
            )
        )
        self._cached_drawer_finger_normal_forces = (
            self._aggregate_taxels_by_finger(
                object_taxel_normal_forces["drawer"]
            )
        )

    def _max_parent_contact_resultant(
        self,
        taxel_world_forces: Any,
    ) -> Any:
        """Aggregate all taxels by rigid parent and return the max force [B]."""

        expected_shape = (
            self.num_envs,
            len(self.agent.taxel_specs),
            3,
        )
        if tuple(taxel_world_forces.shape) != expected_shape:
            raise ValueError(
                "Unexpected M14 taxel world-force shape: "
                f"expected {expected_shape}, got "
                f"{tuple(taxel_world_forces.shape)}."
            )
        parent_forces = torch.zeros(
            (
                self.num_envs,
                self._taxel_parent_group_count,
                3,
            ),
            device=taxel_world_forces.device,
            dtype=taxel_world_forces.dtype,
        )
        # [N] -> [B, N, 3], then [B, N, 3] -> [B, P, 3].
        parent_indices = self._taxel_parent_group_indices.view(
            1,
            -1,
            1,
        ).expand(self.num_envs, -1, 3)
        parent_forces.scatter_add_(
            dim=1,
            index=parent_indices,
            src=taxel_world_forces,
        )
        # [B, P, 3] -> [B, P] -> [B].
        return torch.linalg.norm(parent_forces, dim=-1).amax(dim=1)

    def _banana_is_stably_grasped(self) -> Any:
        contact_force = self.SUCCESS_SPEC.grasp_contact_force_n
        finger_contacts = self._banana_finger_normal_forces() >= contact_force
        thumb_contact = finger_contacts[:, 0]
        non_thumb_contact_count = finger_contacts[:, 1:].sum(dim=1)
        finger_contact_count = finger_contacts.sum(dim=1)
        palm_contact = (
            self._palm_taxel_force(self._banana_taxel_normal_forces())
            >= contact_force
        )
        finger_wrap = (
            thumb_contact
            & (
                non_thumb_contact_count
                >= self.SUCCESS_SPEC.grasp_min_non_thumb_fingers
            )
        )
        palm_wrap = (
            palm_contact
            & (
                finger_contact_count
                >= self.SUCCESS_SPEC.grasp_min_fingers_with_palm
            )
        )
        return finger_wrap | palm_wrap

    def _banana_bbox_world_corners(self) -> Any:
        """Return the live banana OBB corners in the world frame."""

        cached_corners = getattr(
            self,
            "_cached_banana_bbox_world_corners",
            None,
        )
        if cached_corners is None:
            banana_center_matrix = (
                self._banana_center_pose().to_transformation_matrix()
            )
            center_rotation = banana_center_matrix[:, :3, :3]
            center_position = banana_center_matrix[:, :3, 3]
            # [B, 3, 3] x [8, 3] -> [B, 8, 3].
            cached_corners = (
                torch.einsum(
                    "bij,kj->bki",
                    center_rotation,
                    self._banana_bbox_corner_offsets,
                )
                + center_position.unsqueeze(1)
            )
            self._cached_banana_bbox_world_corners = cached_corners
        return cached_corners

    def _banana_bbox_drawer_corners(self) -> Any:
        """Return the live banana bbox corners in the moving drawer frame."""

        cached_corners = getattr(
            self,
            "_cached_banana_bbox_drawer_corners",
            None,
        )
        if cached_corners is None:
            drawer_inverse_matrix = (
                self.drawer_link.pose.inv().to_transformation_matrix()
            )
            drawer_inverse_rotation = drawer_inverse_matrix[:, :3, :3]
            drawer_inverse_position = drawer_inverse_matrix[:, :3, 3]
            # [B, 3, 3] x [B, 8, 3] -> [B, 8, 3].
            cached_corners = (
                torch.einsum(
                    "bij,bkj->bki",
                    drawer_inverse_rotation,
                    self._banana_bbox_world_corners(),
                )
                + drawer_inverse_position.unsqueeze(1)
            )
            self._cached_banana_bbox_drawer_corners = cached_corners
        return cached_corners

    def _banana_inside_drawer(self) -> Any:
        cached_inside = getattr(self, "_cached_banana_inside_drawer", None)
        if cached_inside is None:
            banana_corners = self._banana_bbox_drawer_corners()
            cached_inside = (
                (banana_corners >= self._drawer_interior_lower)
                & (banana_corners <= self._drawer_interior_upper)
            ).all(dim=(1, 2))
            self._cached_banana_inside_drawer = cached_inside
        return cached_inside

    def _banana_outside_drawer(self) -> Any:
        """Return true only after the full banana bbox clears the drawer."""

        cached_outside = getattr(
            self,
            "_cached_banana_outside_drawer",
            None,
        )
        if cached_outside is None:
            banana_corners = self._banana_bbox_drawer_corners()
            banana_lower = banana_corners.amin(dim=1)
            banana_upper = banana_corners.amax(dim=1)
            clearance = self.SUCCESS_SPEC.drawer_extraction_clearance_m
            drawer_lower = self._drawer_interior_lower - clearance
            drawer_upper = self._drawer_interior_upper + clearance
            # [B, 3] -> [B]: two AABBs do not overlap if separated on any axis.
            cached_outside = (
                (banana_upper < drawer_lower)
                | (banana_lower > drawer_upper)
            ).any(dim=1)
            self._cached_banana_outside_drawer = cached_outside
        return cached_outside

    def _banana_is_static(self) -> Any:
        cached_static = getattr(self, "_cached_banana_static", None)
        if cached_static is None:
            linear_speed = torch.linalg.norm(
                self.banana.linear_velocity,
                dim=1,
            )
            angular_speed = torch.linalg.norm(
                self.banana.angular_velocity,
                dim=1,
            )
            cached_static = (
                linear_speed
                <= self.SUCCESS_SPEC.banana_static_linear_speed_mps
            ) & (
                angular_speed
                <= self.SUCCESS_SPEC.banana_static_angular_speed_rad_s
            )
            self._cached_banana_static = cached_static
        return cached_static

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis is GeneralizationAxis.SPATIAL:
            return self._banana_inside_drawer() & self._banana_is_static()
        if axis is GeneralizationAxis.PHYSICAL:
            drawer_near_closed = self._drawer_open_fraction() <= 0.02
            return drawer_near_closed & self._banana_is_static()
        return super()._check_generalization_validity(axis)

    def _banana_on_desk_top(self) -> Any:
        """Return true when the complete banana footprint is on the desktop."""

        cached_on_desk = getattr(
            self,
            "_cached_banana_on_desk_top",
            None,
        )
        if cached_on_desk is None:
            desk_position = self.interaction_site_local_point_tensor(
                local_xyz=(0.0, 0.0, 0.0),
                batch_size=self.num_envs,
            )
            desk_quaternion = self.interaction_site_quaternion_tensor(
                self.num_envs
            )
            desk_local_corners = transform_world_points_to_local(
                frame_positions=desk_position,
                frame_quaternions=desk_quaternion,
                world_points=self._banana_bbox_world_corners(),
            )
            # [B, 8, 2] and [2] -> [B]: the whole OBB projection must fit.
            footprint_inside = (
                torch.abs(desk_local_corners[:, :, :2])
                <= self._desk_top_half_extent_xy
            ).all(dim=(1, 2))
            height_error = torch.abs(
                self._banana_center_pose().p[:, 2] - desk_position[:, 2]
            )
            cached_on_desk = footprint_inside & (
                height_error
                <= self.SUCCESS_SPEC.desktop_height_tolerance_m
            )
            self._cached_banana_on_desk_top = cached_on_desk
        return cached_on_desk

    @staticmethod
    def _next_consecutive_steps(counter: Any, condition: Any) -> Any:
        """Increment counters while a condition holds and reset otherwise."""

        return torch.where(
            condition,
            counter + 1,
            torch.zeros_like(counter),
        )

    def _update_drawer_access_state(self) -> Any:
        """Apply enter/exit hysteresis to the provisional drawer-access stage."""

        drawer_open_fraction = self._drawer_open_fraction()
        entered = (
            drawer_open_fraction
            >= self.SUCCESS_SPEC.drawer_access_enter_fraction
        )
        exited = (
            drawer_open_fraction
            < self.SUCCESS_SPEC.drawer_access_exit_fraction
        )
        self._drawer_access_confirmed = torch.where(
            entered,
            torch.ones_like(self._drawer_access_confirmed),
            torch.where(
                exited,
                torch.zeros_like(self._drawer_access_confirmed),
                self._drawer_access_confirmed,
            ),
        )
        return self._drawer_access_confirmed

    def _update_drawer_closed_state(self) -> Any:
        """Apply independent hysteresis to the current drawer-closed state."""

        drawer_open_fraction = self._drawer_open_fraction()
        entered = (
            drawer_open_fraction
            <= self.SUCCESS_SPEC.drawer_closed_enter_fraction
        )
        exited = (
            drawer_open_fraction
            > self.SUCCESS_SPEC.drawer_closed_exit_fraction
        )
        self._drawer_closed_confirmed = torch.where(
            entered,
            torch.ones_like(self._drawer_closed_confirmed),
            torch.where(
                exited,
                torch.zeros_like(self._drawer_closed_confirmed),
                self._drawer_closed_confirmed,
            ),
        )
        return self._drawer_closed_confirmed

    def _banana_support_contact_force(self, query_mask: Any) -> Any:
        """Query banana-desk support only while placement logic needs it."""

        cached_force = getattr(
            self,
            "_cached_banana_support_force",
            None,
        )
        if cached_force is not None:
            return cached_force
        if not bool(query_mask.any().item()):
            cached_force = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            query_handle = getattr(self, "_banana_support_query", None)
            if query_handle is None:
                raise RuntimeError(
                    "M14 banana-support query was not prepared after scene load."
                )
            pair_forces = self.agent.get_pairwise_contact_forces(
                query_handle
            )
            # [B, R, 1, 3] -> [B, R] -> [B].
            cached_force = torch.linalg.norm(
                pair_forces[:, :, 0, :],
                dim=-1,
            ).amax(dim=1)
            cached_force = torch.where(
                query_mask,
                cached_force,
                torch.zeros_like(cached_force),
            )
        self._cached_banana_support_force = cached_force
        return cached_force

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        drawer_accessible = self._update_drawer_access_state()
        grasp_topology_satisfied = self._banana_is_stably_grasped()
        grasp_candidate = (
            drawer_accessible
            & grasp_topology_satisfied
            & ~self._banana_was_grasped
        )
        self._banana_grasp_candidate_steps = (
            self._next_consecutive_steps(
                self._banana_grasp_candidate_steps,
                grasp_candidate,
            )
        )
        grasp_confirmed = (
            self._banana_grasp_candidate_steps
            >= self._grasp_dwell_steps
        )
        self._banana_was_grasped |= grasp_confirmed

        banana_outside_now = self._banana_outside_drawer()
        self._banana_was_extracted |= (
            self._banana_was_grasped & banana_outside_now
        )
        banana_on_desk = self._banana_on_desk_top()
        banana_static = self._banana_is_static()
        # Final task success is an outcome metric: it depends only on the
        # banana's current placement, support, release, and stability. Ordered
        # process credit remains the responsibility of compute_stage_score().
        release_query_mask = (
            ~grasp_topology_satisfied
            & (
                banana_on_desk
                | self._release_settling_active
                | self._banana_was_stably_placed
            )
        )
        support_force = self._banana_support_contact_force(
            release_query_mask
        )
        banana_supported = (
            support_force >= self.SUCCESS_SPEC.support_contact_force_n
        )
        released_on_desk = (
            banana_on_desk
            & banana_supported
            & ~grasp_topology_satisfied
        )
        self._banana_was_released_on_desk |= released_on_desk
        self._release_settling_active |= (
            released_on_desk & ~self._banana_was_stably_placed
        )
        regrasped_during_settling = (
            self._release_settling_active & grasp_topology_satisfied
        )
        self._release_settling_active &= ~regrasped_during_settling
        self._release_settling_steps = self._next_consecutive_steps(
            self._release_settling_steps,
            self._release_settling_active,
        )
        placement_stable = (
            self._release_settling_active
            & released_on_desk
            & banana_static
        )
        self._placement_stable_steps = self._next_consecutive_steps(
            self._placement_stable_steps,
            placement_stable,
        )
        placement_confirmed = (
            self._placement_stable_steps
            >= self._placement_dwell_steps
        )
        self._banana_was_stably_placed |= placement_confirmed
        self._release_settling_active &= ~placement_confirmed

        self._drawer_was_accessible |= drawer_accessible

        drawer_velocity = self._drawer_opening_velocity()
        drawer_closed_now = self._update_drawer_closed_state()
        drawer_closed_and_static = (
            self._banana_was_stably_placed
            & drawer_closed_now
            & (
                torch.abs(drawer_velocity)
                <= self.SUCCESS_SPEC.drawer_closed_max_speed_mps
            )
        )
        self._drawer_closed_stable_steps = self._next_consecutive_steps(
            self._drawer_closed_stable_steps,
            drawer_closed_and_static,
        )
        drawer_closed_stably = (
            self._drawer_closed_stable_steps
            >= self._drawer_closed_dwell_steps
        )
        self._drawer_closed_once |= drawer_closed_stably

        banana_placement_valid_now = (
            released_on_desk & banana_static
        )
        success = drawer_closed_stably & banana_placement_valid_now

        return {
            "success": success,
            "drawer_accessible": self._drawer_was_accessible,
            "banana_grasped": self._banana_was_grasped,
            "banana_outside_drawer": self._banana_was_extracted,
            "banana_stably_placed": self._banana_was_stably_placed,
            "drawer_closed": success,
        }
