from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from dex_benchmark.assets import robocasa_asset_path
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
    ArticulationBuildValues,
    LayoutBuilder,
    LayoutRegistry,
    MJCFActorSpec,
    PoseSpec,
    ProceduralTableSpec,
    pose_spec_to_sapien_pose,
)
from dex_benchmark.sim.scenes.layouts.studio import (
    STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID,
    STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID,
    STUDIO_PHYSICAL_MATERIAL,
)
from dex_benchmark.sim.robots.tactile import PairwiseContactQueryHandle
from dex_benchmark.sim.tasks.attributes import LayoutId
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


M15_SETTINGS = TaskRegistry.register(TaskRegistry.load("m15_v1"))


@dataclass(frozen=True)
class M15SuccessSpec:
    door_open_angle_threshold_rad: float = math.radians(60.0)
    grasp_contact_force_n: float = 0.30
    grasp_min_contact_fingers: int = 2
    residual_hand_contact_force_n: float = 0.10
    max_linear_speed_mps: float = 0.03
    max_angular_speed_rad_s: float = 0.50
    second_level_height_tolerance_m: float = 0.025
    containment_tolerance_m: float = 0.005
    max_tilt_rad: float = math.radians(20.0)
    placement_dwell_s: float = 1.00


M15_SUCCESS_SPEC = M15SuccessSpec()
M15_CANDLE_NAME: str = "candle_10"
M15_CANDLE_ASSET_RELATIVE_PATH: str = (
    "objects/objaverse/candle/candle_10/model.xml"
)
M15_CANDLE_SCALE: float = 1.20
M15_CABINET_NAME = "dresser_cabinet_b"
M15_INTERACTION_SITE_ID = M15_CABINET_NAME
M15_UPPER_DOOR_JOINT_NAMES = ("joint_2", "joint_3")
M15_UPPER_DOOR_LINK_NAMES = ("link_2", "link_3")
M15_CABINET_BODY_LINK_NAME = "link_6"
M15_GRASP_FINGER_NAMES = ("th", "ff", "mf", "rf", "lf")
M15_GRASP_FINGER_LINK_PARTS = ("proximal", "middle", "distal")
M15_GRASP_FINGER_LINK_GROUPS = tuple(
    tuple(
        f"rh_{finger_name}{link_part}"
        for link_part in M15_GRASP_FINGER_LINK_PARTS
    )
    for finger_name in M15_GRASP_FINGER_NAMES
)
M15_GRASP_FINGER_LINK_NAMES = tuple(
    link_name
    for finger_link_names in M15_GRASP_FINGER_LINK_GROUPS
    for link_name in finger_link_names
)
# Collision bounds measured from candle_10's MJCF collision meshes after its
# declared 0.10 mesh scale and the task's 1.20 actor scale. Coordinates are
# relative to the actor root.
M15_CANDLE_COLLISION_BBOX_MIN_XYZ: tuple[float, float, float] = (
    -0.0276812520,
    -0.0281490660,
    -0.0598462212,
)
M15_CANDLE_COLLISION_BBOX_MAX_XYZ: tuple[float, float, float] = (
    0.0286032648,
    0.0282100800,
    0.0598941264,
)
# The 32 collision meshes are approximately circular around the actor-local Z
# axis. This radius is the maximum XY distance of all collision vertices from
# that axis, rounded up to the nearest 0.01 mm. A finite cylinder therefore
# remains conservative without introducing the yaw-dependent square corners of
# an oriented bounding box.
M15_CANDLE_COLLISION_RADIUS_M: float = 0.02972
M15_CANDLE_COLLISION_MIN_Z_M: float = M15_CANDLE_COLLISION_BBOX_MIN_XYZ[2]
M15_CANDLE_COLLISION_MAX_Z_M: float = M15_CANDLE_COLLISION_BBOX_MAX_XYZ[2]
# Model 47648's outer-counter front edge, measured from link_6 collision mesh
# original-98 after applying the PartNet bbox-center transform and 1.15 scale.
M15_OUTER_COUNTER_FRONT_EDGE_LOCAL_X = -0.22234445
M15_SOURCE_CLEARANCE_M = 0.004
M15_CANDLE_SOURCE_FRONT_MARGIN_M: float = 0.020
M15_CANDLE_SOURCE_LOCAL_XY: tuple[float, float] = (
    M15_OUTER_COUNTER_FRONT_EDGE_LOCAL_X
    - M15_CANDLE_COLLISION_BBOX_MIN_XYZ[0]
    + M15_CANDLE_SOURCE_FRONT_MARGIN_M,
    0.0,
)
M15_CANDLE_SOURCE_YAW_RAD: float = 0.0

# Task-private source riser. Its front edge stays aligned with the dresser's
# outer-counter edge. Adjust M15_SOURCE_RISER_HEIGHT_M to tune pickup height
# without changing the placement or success geometry inside the cabinet.
M15_SOURCE_RISER_NAME: str = "m15_candle_riser"
M15_SOURCE_RISER_HEIGHT_M: float = 0.15
M15_SOURCE_RISER_TABLETOP_HALF_SIZE_XYZ: tuple[float, float, float] = (
    0.12,
    0.15,
    0.01,
)
M15_SOURCE_RISER_LEG_HALF_SIZE_XYZ: tuple[float, float, float] = (
    0.012,
    0.012,
    0.5
    * (
        M15_SOURCE_RISER_HEIGHT_M
        - 2.0 * M15_SOURCE_RISER_TABLETOP_HALF_SIZE_XYZ[2]
    ),
)
M15_SOURCE_RISER_LEG_INSET_XY: tuple[float, float] = (0.020, 0.020)
M15_SOURCE_RISER_TOP_LOCAL_XY: tuple[float, float] = (
    M15_OUTER_COUNTER_FRONT_EDGE_LOCAL_X
    + M15_SOURCE_RISER_TABLETOP_HALF_SIZE_XYZ[0],
    0.0,
)
M15_SOURCE_SUPPORT_HEIGHT_TOLERANCE_M: float = 0.015

# Relative to the existing dresser interaction-site robot base. Local +X
# moves toward the cabinet and local +Z raises the base. Keep this as one
# explicit tuning knob for reachability experiments across all Studio layouts.
M15_ROBOT_BASE_OFFSET_LOCAL_XYZ: tuple[float, float, float] = (
    -0.10,
    0.0,
    0.10,
)

# Model 47648's physical inner faces around the complete upper-cabinet second
# level, expressed in the interaction-site frame after the URDF fixed-joint
# transform and 1.15 asset scale. X min is the shelf/door inner plane, X max is
# the back-panel inner face, Y min/max are the side-panel inner faces, and Z
# min/max are the lower-shelf top and upper-shelf underside. The center divider
# remains physical collision geometry but does not create a hidden target zone.
M15_SECOND_LEVEL_INTERIOR_MIN_XYZ = (-0.066556, -0.428441, 0.229369)
M15_SECOND_LEVEL_INTERIOR_MAX_XYZ = (0.219082, 0.428475, 0.553565)

M15_NOMINAL_DOOR_FRICTION = 0.10
M15_NOMINAL_DOOR_DAMPING = 0.15
# At the task's 1.20 actor scale, use a mass/inertia multiplier that yields
# roughly 106 g so the larger candle remains easy to manipulate without
# behaving like an unrealistically light shell.
M15_CANDLE_NOMINAL_MASS_SCALE: float = 6.20
M15_CANDLE_NOMINAL_FRICTION_SCALE: float = 2.00
M15_SPATIAL_PARAMETER_NAMES = (
    "candle_x_offset_m",
    "candle_y_offset_m",
    "candle_yaw_offset_rad",
    "left_door_initial_angle_rad",
    "right_door_initial_angle_rad",
)
M15_PHYSICAL_PARAMETER_NAMES = (
    "candle_mass_log_scale",
    "candle_friction_log_scale",
    "door_friction",
    "door_damping",
)
M15_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("candle_x_offset_m", -0.0015, 0.0015),
            GeneralizationParameterSpec("candle_y_offset_m", -0.015, 0.015),
            GeneralizationParameterSpec(
                "candle_yaw_offset_rad",
                -math.radians(2.0),
                math.radians(2.0),
            ),
            GeneralizationParameterSpec(
                "left_door_initial_angle_rad",
                0.0,
                math.radians(10.0),
            ),
            GeneralizationParameterSpec(
                "right_door_initial_angle_rad",
                0.0,
                math.radians(10.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "candle_mass_log_scale", math.log(0.80), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "candle_friction_log_scale", math.log(0.90), math.log(1.10)
            ),
            GeneralizationParameterSpec("door_friction", 0.06, 0.16),
            GeneralizationParameterSpec("door_damping", 0.08, 0.24),
        ),
        requires_rebuild=True,
    ),
)


# Recorded in studio L3 and stored relative to the single-arm robot base. The
# same base-local pose is transformed into every supported layout so the Human
# Viewer keeps an identical relationship to the operator and robot.
M15_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.1595543220, 0.3608824367, 0.5304620419),
    quaternion_wxyz=(
        -0.9641088509,
        -0.0289843201,
        -0.1184961642,
        0.2358234336,
    ),
)
M15_LAYOUT_IDS = (
    LayoutId.STUDIO_L1,
    LayoutId.STUDIO_L2,
    LayoutId.STUDIO_L3,
)


def _m15_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in M15_LAYOUT_IDS:
        robot_base_pose = LayoutRegistry.get(layout_id).interaction_site(
            M15_INTERACTION_SITE_ID
        ).robot_base_center_pose
        camera_world_pose = robot_base_pose.transform_pose(
            M15_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
        )
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=camera_world_pose.quaternion_wxyz,
        )
    return camera_poses


@register_env(M15_SETTINGS.env_id, max_episode_steps=M15_SETTINGS.max_steps)
class M15Env(DexVLABenchBaseEnv):
    """Open either upper glass door and place one candle on the second level."""

    TASK_SETTINGS = M15_SETTINGS
    SUCCESS_SPEC = M15_SUCCESS_SPEC
    GENERALIZATION_SPEC = M15_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = M15_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _m15_human_viewer_camera_poses()

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

    def _robot_base_pose_spec(self) -> PoseSpec:
        nominal_pose = self.interaction_site_robot_base_center_pose()
        offset_pose = PoseSpec(
            position_xyz=M15_ROBOT_BASE_OFFSET_LOCAL_XYZ,
        )
        return nominal_pose.transform_pose(offset_pose)

    def _robot_base_pose(self) -> Any:
        return pose_spec_to_sapien_pose(self._robot_base_pose_spec())

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        actor_values, articulation_values = self._physical_build_values()
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self._task_layout_spec(),
            actor_values=actor_values,
            articulation_values=articulation_values,
        ).build()
        self.cabinet = self.layout_scene.require_entity(M15_CABINET_NAME)
        self.candle = self.layout_scene.require_entity(M15_CANDLE_NAME)
        self.source_riser = self.layout_scene.require_entity(M15_SOURCE_RISER_NAME)
        self.upper_door_joints = self._require_upper_door_joints()
        self.upper_door_links = tuple(
            self._require_cabinet_link(link_name)
            for link_name in M15_UPPER_DOOR_LINK_NAMES
        )
        self.cabinet_body_link = self._require_cabinet_link(
            M15_CABINET_BODY_LINK_NAME
        )
        self._configure_geometry_queries_and_constants()

    def _task_layout_spec(self) -> Any:
        if M15_SOURCE_RISER_LEG_HALF_SIZE_XYZ[2] <= 0.0:
            raise ValueError(
                "M15_SOURCE_RISER_HEIGHT_M must exceed the riser tabletop "
                "thickness."
            )
        outer_counter_surface = self.current_layout_spec.support_surfaces[
            STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID
        ]
        riser_top_pose = outer_counter_surface.pose.transform_pose(
            PoseSpec(
                position_xyz=(
                    M15_SOURCE_RISER_TOP_LOCAL_XY[0],
                    M15_SOURCE_RISER_TOP_LOCAL_XY[1],
                    M15_SOURCE_RISER_HEIGHT_M,
                )
            )
        )
        source_riser = ProceduralTableSpec(
            name=M15_SOURCE_RISER_NAME,
            pose=riser_top_pose,
            tabletop_half_size_xyz=M15_SOURCE_RISER_TABLETOP_HALF_SIZE_XYZ,
            table_height=M15_SOURCE_RISER_HEIGHT_M,
            leg_half_size_xyz=M15_SOURCE_RISER_LEG_HALF_SIZE_XYZ,
            leg_inset_xy=M15_SOURCE_RISER_LEG_INSET_XY,
            tabletop_color_rgba=(0.60, 0.39, 0.22, 1.0),
            leg_color_rgba=(0.48, 0.29, 0.16, 1.0),
            body_type="kinematic",
        )
        candle_root_local_z = (
            M15_SOURCE_RISER_HEIGHT_M
            + M15_SOURCE_CLEARANCE_M
            - M15_CANDLE_COLLISION_BBOX_MIN_XYZ[2]
        )
        candle_pose = outer_counter_surface.pose.transform_pose(
            PoseSpec(
                position_xyz=(
                    M15_CANDLE_SOURCE_LOCAL_XY[0],
                    M15_CANDLE_SOURCE_LOCAL_XY[1],
                    candle_root_local_z,
                )
            )
        )
        candle = MJCFActorSpec(
            name=M15_CANDLE_NAME,
            asset_path=robocasa_asset_path(M15_CANDLE_ASSET_RELATIVE_PATH),
            pose=candle_pose,
            scale_xyz=(M15_CANDLE_SCALE,) * 3,
            body_type="dynamic",
            visual_groups=(1,),
        )
        return replace(
            self.current_layout_spec,
            build_items=(
                *self.current_layout_spec.build_items,
                candle,
                source_riser,
            ),
        )

    def _physical_build_values(
        self,
    ) -> tuple[
        dict[str, ActorBuildValues] | None,
        dict[str, ArticulationBuildValues],
    ]:
        if self.generalization_axis is GeneralizationAxis.PHYSICAL:
            values = self._generalization_values(
                M15_PHYSICAL_PARAMETER_NAMES,
                cache_name="_m15_physical_values",
            )
            mass_scales = (
                M15_CANDLE_NOMINAL_MASS_SCALE
                * positive_scale(values["candle_mass_log_scale"])
            )
            friction_scales = positive_scale(values["candle_friction_log_scale"])
            candle_materials = tuple(
                scaled_physical_material(
                    STUDIO_PHYSICAL_MATERIAL,
                    friction_scale=(
                        M15_CANDLE_NOMINAL_FRICTION_SCALE
                        * float(friction_scale)
                    ),
                    restitution=STUDIO_PHYSICAL_MATERIAL.restitution,
                )
                for friction_scale in friction_scales
            )
            actor_values: dict[str, ActorBuildValues] | None = {
                M15_CANDLE_NAME: ActorBuildValues(
                    physical_materials=candle_materials,
                    mass_scales=tuple(float(scale) for scale in mass_scales),
                )
            }
            door_friction = tuple(float(value) for value in values["door_friction"])
            door_damping = tuple(float(value) for value in values["door_damping"])
        else:
            nominal_candle_material = scaled_physical_material(
                STUDIO_PHYSICAL_MATERIAL,
                friction_scale=M15_CANDLE_NOMINAL_FRICTION_SCALE,
                restitution=STUDIO_PHYSICAL_MATERIAL.restitution,
            )
            actor_values = {
                M15_CANDLE_NAME: ActorBuildValues(
                    physical_materials=(nominal_candle_material,) * self.num_envs,
                    mass_scales=(M15_CANDLE_NOMINAL_MASS_SCALE,)
                    * self.num_envs,
                )
            }
            door_friction = (M15_NOMINAL_DOOR_FRICTION,) * self.num_envs
            door_damping = (M15_NOMINAL_DOOR_DAMPING,) * self.num_envs

        articulation_values = {
            M15_CABINET_NAME: ArticulationBuildValues(
                joint_friction={
                    joint_name: door_friction
                    for joint_name in M15_UPPER_DOOR_JOINT_NAMES
                },
                joint_damping={
                    joint_name: door_damping
                    for joint_name in M15_UPPER_DOOR_JOINT_NAMES
                },
            )
        }
        return actor_values, articulation_values

    def _require_upper_door_joints(self) -> tuple[Any, Any]:
        joints: list[Any] = []
        for joint_name in M15_UPPER_DOOR_JOINT_NAMES:
            joint = self.cabinet.active_joints_map.get(joint_name)
            if joint is None:
                known = ", ".join(sorted(self.cabinet.active_joints_map))
                raise RuntimeError(
                    f"M15 cabinet has no joint {joint_name!r}; known joints: {known}."
                )
            joint_type = (
                joint.type[0]
                if isinstance(joint.type, (tuple, list))
                else joint.type
            )
            if joint_type not in {"revolute", "revolute_unwrapped"}:
                raise RuntimeError(
                    f"M15 upper door {joint_name!r} must be rotational, "
                    f"got {joint.type!r}."
                )
            joints.append(joint)
        return joints[0], joints[1]

    def _require_cabinet_link(self, link_name: str) -> Any:
        link = self.cabinet.links_map.get(link_name)
        if link is None:
            known = ", ".join(sorted(self.cabinet.links_map))
            raise RuntimeError(
                f"M15 cabinet has no link {link_name!r}; known links: {known}."
            )
        return link

    @staticmethod
    def _consistent_active_joint_index(joint: Any) -> int:
        active_index = joint.active_index
        if active_index is None:
            raise RuntimeError(
                f"M15 joint {joint.name!r} has no active qpos index."
            )
        indices = active_index.detach().cpu().reshape(-1)
        if indices.numel() == 0 or not bool((indices == indices[0]).all()):
            raise RuntimeError(
                f"M15 joint {joint.name!r} has inconsistent active indices."
            )
        return int(indices[0].item())

    @staticmethod
    def _quaternion_rotate(quaternion: Any, vectors: Any) -> Any:
        """Rotate [B, N, 3] vectors with scalar-first [B, 4] quaternions."""

        quaternion_vector = quaternion[:, None, 1:]
        quaternion_scalar = quaternion[:, None, :1]
        # [B, 1, 3] x [B, N, 3] -> [B, N, 3].
        twice_cross = 2.0 * torch.cross(
            quaternion_vector.expand_as(vectors), vectors, dim=-1
        )
        return vectors + quaternion_scalar * twice_cross + torch.cross(
            quaternion_vector.expand_as(vectors), twice_cross, dim=-1
        )

    @staticmethod
    def _candle_cylinder_projection_bounds(
        axis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return axis-aligned offsets enclosing the finite candle cylinder."""

        if axis.ndim != 2 or axis.shape[1] != 3:
            raise ValueError(
                "M15 candle cylinder axis must have shape [B, 3], got "
                f"{tuple(axis.shape)}."
            )
        axis_norm = torch.linalg.norm(axis, dim=1, keepdim=True)
        if bool((axis_norm <= 1.0e-8).any()):
            raise ValueError("M15 candle cylinder axis must be non-zero.")
        normalized_axis = axis / axis_norm
        # [B, 3] -> [B, 3]: projection of the circular cross-section onto
        # each coordinate axis, including arbitrary candle tilt.
        radial_projection = torch.sqrt(
            torch.clamp(1.0 - normalized_axis.square(), min=0.0)
        )
        radial_extent = M15_CANDLE_COLLISION_RADIUS_M * radial_projection
        lower_cap_projection = (
            M15_CANDLE_COLLISION_MIN_Z_M * normalized_axis
        )
        upper_cap_projection = (
            M15_CANDLE_COLLISION_MAX_Z_M * normalized_axis
        )
        lower_offset = torch.minimum(
            lower_cap_projection,
            upper_cap_projection,
        ) - radial_extent
        upper_offset = torch.maximum(
            lower_cap_projection,
            upper_cap_projection,
        ) + radial_extent
        return lower_offset, upper_offset

    def _candle_up_world(self) -> torch.Tensor:
        local_up = torch.zeros(
            (self.num_envs, 1, 3), device=self.device, dtype=torch.float32
        )
        local_up[:, :, 2] = 1.0
        # [B, 1, 3] -> [B, 3]: candle-local +Z in world coordinates.
        return self._quaternion_rotate(self.candle.pose.q, local_up)[:, 0]

    def _candle_pose_site(self) -> tuple[torch.Tensor, torch.Tensor]:
        site_position = self.interaction_site_local_point_tensor(
            local_xyz=(0.0, 0.0, 0.0),
            batch_size=self.num_envs,
        )
        inverse_site_quaternion = self.interaction_site_quaternion_tensor(
            self.num_envs
        ).clone()
        inverse_site_quaternion[:, 1:] *= -1.0
        # [B, 3] -> [B, 1, 3] -> [B, 3]: express the candle root and local
        # Z axis in the interaction-site coordinate frame.
        candle_root_site = self._quaternion_rotate(
            inverse_site_quaternion,
            (self.candle.pose.p - site_position)[:, None, :],
        )[:, 0]
        candle_axis_site = self._quaternion_rotate(
            inverse_site_quaternion,
            self._candle_up_world()[:, None, :],
        )[:, 0]
        return candle_root_site, candle_axis_site

    def _candle_site_occupancy(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candle_root_site, candle_axis_site = self._candle_pose_site()
        lower_offset, upper_offset = self._candle_cylinder_projection_bounds(
            candle_axis_site
        )
        # [B, 3] + [B, 3] -> [B, 3]: conservative finite-cylinder bounds.
        return (
            candle_root_site,
            candle_root_site + lower_offset,
            candle_root_site + upper_offset,
        )

    def _configure_geometry_queries_and_constants(self) -> None:
        self._second_level_interior_lower = torch.as_tensor(
            M15_SECOND_LEVEL_INTERIOR_MIN_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        self._second_level_interior_upper = torch.as_tensor(
            M15_SECOND_LEVEL_INTERIOR_MAX_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        second_level_surface = self.current_layout_spec.support_surfaces[
            STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID
        ]
        site_pose = self.interaction_site_frame_pose()
        self._second_level_surface_local_z = float(
            second_level_surface.pose.position_xyz[2]
            - site_pose.position_xyz[2]
        )
        outer_counter_surface = self.current_layout_spec.support_surfaces[
            STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID
        ]
        self._source_riser_top_local_z = float(
            outer_counter_surface.pose.position_xyz[2]
            - site_pose.position_xyz[2]
            + M15_SOURCE_RISER_HEIGHT_M
        )

        missing_finger_links = [
            name
            for name in M15_GRASP_FINGER_LINK_NAMES
            if name not in self.agent.robot.links_map
        ]
        if missing_finger_links:
            raise RuntimeError(
                "M15 missing grasp links: " + ", ".join(missing_finger_links)
            )
        self._finger_links = tuple(
            self.agent.robot.links_map[name]
            for name in M15_GRASP_FINGER_LINK_NAMES
        )
        self._candle_grasp_query = self.agent.prepare_pairwise_contact_query(
            self._finger_links,
            (self.candle,),
        )

        self._upper_door_active_indices = tuple(
            self._consistent_active_joint_index(joint)
            for joint in self.upper_door_joints
        )
        self._placement_dwell_steps = self._duration_steps(
            self.SUCCESS_SPEC.placement_dwell_s
        )

    def _candle_containment_diagnostics(self) -> tuple[Any, Any, Any]:
        _candle_root, candle_minimum, candle_maximum = (
            self._candle_site_occupancy()
        )
        # Two [B, 3] face distances -> [B, 6] signed clearances.
        face_clearance = torch.cat(
            (
                candle_minimum - self._second_level_interior_lower,
                self._second_level_interior_upper - candle_maximum,
            ),
            dim=1,
        )
        return candle_minimum, candle_maximum, face_clearance

    def _duration_steps(self, duration_s: float) -> int:
        if duration_s <= 0.0:
            raise ValueError(f"M15 duration must be positive, got {duration_s}.")
        return max(1, math.ceil(duration_s * float(self._control_freq)))

    def _initialize_episode(
        self,
        env_idx: Any,
        options: dict[str, Any],
    ) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._initialize_cabinet_joints(env_idx)
            self._place_source_candle(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_agent_to_rest(
                self.agent,
                self._robot_base_pose(),
                batch_size,
            )

            self._ensure_episode_state()
            self._door_was_opened[env_idx] = False
            self._candle_was_grasped[env_idx] = False
            self._candle_was_inside[env_idx] = False
            self._candle_was_stably_placed[env_idx] = False
            self._placement_steps[env_idx] = 0

    def _initialize_cabinet_joints(self, env_idx: Any) -> None:
        joint_limits = self.cabinet.get_qlimits()
        # [B, D, 2] -> [b, D]: start all doors and drawers at lower limits.
        qpos = joint_limits[env_idx, :, 0].clone()
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M15_SPATIAL_PARAMETER_NAMES,
                cache_name="_m15_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            for joint_index, parameter_name in zip(
                self._upper_door_active_indices,
                (
                    "left_door_initial_angle_rad",
                    "right_door_initial_angle_rad",
                ),
                strict=True,
            ):
                sampled_angles = torch.as_tensor(
                    values[parameter_name][value_indices],
                    device=self.device,
                    dtype=torch.float32,
                )
                qpos[:, joint_index] = sampled_angles.clamp(
                    min=joint_limits[env_idx, joint_index, 0],
                    max=joint_limits[env_idx, joint_index, 1],
                )
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_qvel(torch.zeros_like(qpos))

    def _surface_pose(self, surface_id: str, batch_size: int) -> Any:
        surface = self.current_layout_spec.support_surfaces[surface_id]
        return Pose.create_from_pq(
            p=torch.as_tensor(
                surface.pose.position_xyz,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(batch_size, 1),
            q=torch.as_tensor(
                surface.pose.quaternion_wxyz,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(batch_size, 1),
        )

    def _place_source_candle(self, env_idx: Any) -> None:
        batch_size = len(env_idx)
        surface_pose = self._surface_pose(
            STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID,
            batch_size,
        )
        x = np.full(batch_size, M15_CANDLE_SOURCE_LOCAL_XY[0])
        y = np.full(batch_size, M15_CANDLE_SOURCE_LOCAL_XY[1])
        yaw = np.full(batch_size, M15_CANDLE_SOURCE_YAW_RAD)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M15_SPATIAL_PARAMETER_NAMES,
                cache_name="_m15_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            x += values["candle_x_offset_m"][value_indices]
            y += values["candle_y_offset_m"][value_indices]
            yaw += values["candle_yaw_offset_rad"][value_indices]

        yaw_tensor = torch.as_tensor(yaw, device=self.device, dtype=torch.float32)
        half_yaw = 0.5 * yaw_tensor
        yaw_q = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(yaw_tensor),
                torch.zeros_like(yaw_tensor),
                torch.sin(half_yaw),
            ),
            dim=1,
        )
        orientation = (surface_pose * Pose.create_from_pq(q=yaw_q)).q
        local_up = torch.zeros(
            (batch_size, 1, 3), device=self.device, dtype=torch.float32
        )
        local_up[:, :, 2] = 1.0
        candle_axis_world = self._quaternion_rotate(orientation, local_up)[:, 0]
        lower_offset, _upper_offset = self._candle_cylinder_projection_bounds(
            candle_axis_world
        )
        bottom_offset_z = lower_offset[:, 2]
        local_root = torch.stack(
            (
                torch.as_tensor(x, device=self.device, dtype=torch.float32),
                torch.as_tensor(y, device=self.device, dtype=torch.float32),
                (
                    M15_SOURCE_RISER_HEIGHT_M
                    + M15_SOURCE_CLEARANCE_M
                    - bottom_offset_z
                ),
            ),
            dim=1,
        )
        desired_root = (surface_pose * Pose.create_from_pq(p=local_root)).p
        self.candle.set_pose(
            Pose.create_from_pq(
                p=desired_root,
                q=orientation,
            )
        )
        zero_velocity = torch.zeros((batch_size, 3), device=self.device)
        self.candle.set_linear_velocity(zero_velocity)
        self.candle.set_angular_velocity(zero_velocity)

    def _reset_agent_to_rest(
        self,
        agent: Any,
        pose: Any,
        batch_size: int,
    ) -> None:
        rest = agent.keyframes.get("rest")
        if rest is None or rest.qpos is None:
            raise RuntimeError(f"M15 agent {agent.uid!r} requires rest qpos.")
        qpos = torch.as_tensor(rest.qpos, device=self.device, dtype=torch.float32)
        agent.reset(qpos.unsqueeze(0).repeat(batch_size, 1))
        agent.robot.set_pose(pose)

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_door_was_opened"):
            return
        self._door_was_opened = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._candle_was_grasped = torch.zeros_like(self._door_was_opened)
        self._candle_was_inside = torch.zeros_like(self._door_was_opened)
        self._candle_was_stably_placed = torch.zeros_like(self._door_was_opened)
        self._placement_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def _override_bool(self, name: str) -> Any | None:
        value = getattr(self, "_state_injection_measurements", {}).get(name)
        if value is None:
            return None
        return self._as_per_env_tensor(value, dtype=torch.bool)

    def _upper_door_opened(self) -> Any:
        override = self._override_bool("cabinet_door_opened")
        if override is not None:
            return override
        qpos = self.cabinet.get_qpos()
        # [B, D] -> [B, 2]: either upper glass door may satisfy the task.
        upper_door_angles = qpos[:, self._upper_door_active_indices]
        return (
            upper_door_angles > self.SUCCESS_SPEC.door_open_angle_threshold_rad
        ).any(dim=1)

    def _candle_finger_contact_forces(self) -> Any:
        pair_forces = self.agent.get_pairwise_contact_forces(
            self._candle_grasp_query
        )
        return self._finger_group_force_magnitudes(pair_forces)

    @staticmethod
    def _finger_group_force_magnitudes(pair_forces: Any) -> Any:
        """Aggregate every phalanx/candle contact into one force per finger."""

        expected_link_count = len(M15_GRASP_FINGER_LINK_NAMES)
        if (
            pair_forces.ndim != 4
            or pair_forces.shape[1] != expected_link_count
            or pair_forces.shape[-1] != 3
        ):
            raise ValueError(
                "M15 candle pair forces must have shape "
                f"[B, {expected_link_count}, T, 3], got "
                f"{tuple(pair_forces.shape)}."
            )

        finger_force_magnitudes: list[Any] = []
        link_start = 0
        for finger_link_names in M15_GRASP_FINGER_LINK_GROUPS:
            link_end = link_start + len(finger_link_names)
            # [B, 3 phalanges, T targets, 3] -> [B, 3] -> [B]: combine the
            # whole anatomical finger before applying the contact threshold.
            finger_force_world = pair_forces[:, link_start:link_end].sum(
                dim=(1, 2)
            )
            finger_force_magnitudes.append(
                torch.linalg.norm(finger_force_world, dim=-1)
            )
            link_start = link_end
        return torch.stack(finger_force_magnitudes, dim=1)

    def _candle_grasped(self) -> Any:
        override = self._override_bool("candle_grasped")
        if override is not None:
            return override
        finger_contact = (
            self._candle_finger_contact_forces()
            >= self.SUCCESS_SPEC.grasp_contact_force_n
        )
        return (
            finger_contact.sum(dim=1)
            >= self.SUCCESS_SPEC.grasp_min_contact_fingers
        )

    def _candle_released(self) -> Any:
        override = self._override_bool("candle_released")
        if override is not None:
            return override
        return (
            self._candle_finger_contact_forces().amax(dim=1)
            < self.SUCCESS_SPEC.residual_hand_contact_force_n
        )

    def _candle_inside_second_level(self) -> Any:
        override = self._override_bool("candle_inside_cabinet")
        if override is not None:
            return override
        _candle_minimum, _candle_maximum, face_clearance = (
            self._candle_containment_diagnostics()
        )
        return (
            face_clearance >= -self.SUCCESS_SPEC.containment_tolerance_m
        ).all(dim=1)

    def _candle_on_second_level_surface(self) -> Any:
        override = self._override_bool("candle_on_second_level_surface")
        if override is not None:
            return override
        _candle_root, candle_minimum, _candle_maximum = (
            self._candle_site_occupancy()
        )
        candle_bottom_z = candle_minimum[:, 2]
        return (
            torch.abs(candle_bottom_z - self._second_level_surface_local_z)
            <= self.SUCCESS_SPEC.second_level_height_tolerance_m
        )

    def _candle_static(self) -> Any:
        override = self._override_bool("candle_static")
        if override is not None:
            return override
        return (
            torch.linalg.norm(self.candle.linear_velocity, dim=1)
            <= self.SUCCESS_SPEC.max_linear_speed_mps
        ) & (
            torch.linalg.norm(self.candle.angular_velocity, dim=1)
            <= self.SUCCESS_SPEC.max_angular_speed_rad_s
        )

    def _candle_source_footprint_valid(self) -> Any:
        _candle_root, minimum, maximum = self._candle_site_occupancy()
        half_x, half_y, _ = M15_SOURCE_RISER_TABLETOP_HALF_SIZE_XYZ
        center_x, center_y = M15_SOURCE_RISER_TOP_LOCAL_XY
        return (
            (minimum[:, 0] >= center_x - half_x)
            & (maximum[:, 0] <= center_x + half_x)
            & (minimum[:, 1] >= center_y - half_y)
            & (maximum[:, 1] <= center_y + half_y)
        )

    def _candle_upright(self) -> Any:
        override = self._override_bool("candle_upright")
        if override is not None:
            return override
        candle_up_world = self._candle_up_world()
        return candle_up_world[:, 2] >= math.cos(self.SUCCESS_SPEC.max_tilt_rad)

    def _candle_on_source_riser(self) -> Any:
        _candle_root, candle_minimum, _candle_maximum = (
            self._candle_site_occupancy()
        )
        candle_bottom_z = candle_minimum[:, 2]
        return (
            torch.abs(candle_bottom_z - self._source_riser_top_local_z)
            <= M15_SOURCE_SUPPORT_HEIGHT_TOLERANCE_M
        )

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            finite_candle_pose = torch.isfinite(self.candle.pose.raw_pose).all(dim=1)
            doors_below_threshold = ~self._upper_door_opened()
            return (
                finite_candle_pose
                & doors_below_threshold
                & self._candle_static()
                & self._candle_on_source_riser()
                & self._candle_source_footprint_valid()
            )
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        door_opened = self._upper_door_opened()
        candle_grasped = self._candle_grasped()
        candle_inside = self._candle_inside_second_level()
        candle_released = self._candle_released()
        candle_static = self._candle_static()
        candle_on_surface = self._candle_on_second_level_surface()
        candle_upright = self._candle_upright()

        self._door_was_opened |= door_opened
        self._candle_was_grasped |= candle_grasped & self._door_was_opened
        self._candle_was_inside |= (
            candle_inside
            & self._door_was_opened
            & self._candle_was_grasped
        )
        placement_candidate = (
            candle_inside
            & candle_on_surface
            & candle_released
            & candle_static
            & candle_upright
            & self._door_was_opened
            & self._candle_was_grasped
            & self._candle_was_inside
        )
        self._placement_steps = torch.where(
            placement_candidate,
            self._placement_steps + 1,
            torch.zeros_like(self._placement_steps),
        )
        success = self._placement_steps >= self._placement_dwell_steps
        self._candle_was_stably_placed |= success

        return {
            "success": success,
            "cabinet_door_opened": self._door_was_opened,
            "candle_grasped": self._candle_was_grasped,
            "candle_inside_cabinet": self._candle_was_inside,
            "candle_stably_placed": success,
            "candle_grasped_now": candle_grasped,
            "candle_inside_now": candle_inside,
            "candle_on_second_level_surface": candle_on_surface,
            "candle_released": candle_released,
            "candle_static": candle_static,
            "candle_upright": candle_upright,
        }




    def _max_pair_force(self, query: PairwiseContactQueryHandle) -> Any:
        pair_forces = self.agent.get_pairwise_contact_forces(query)
        # [B, R, D, 3] -> [B]: maximum force over all prepared pairs.
        return torch.linalg.norm(pair_forces, dim=-1).amax(dim=(1, 2))
