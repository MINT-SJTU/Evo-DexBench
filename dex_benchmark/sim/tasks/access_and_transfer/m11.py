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
)
from dex_benchmark.sim.scenes.layouts import (
    ArticulatedAssetSpec,
    LayoutBuilder,
    LayoutBuildItemSpec,
    PoseSpec,
    pose_spec_to_sapien_pose,
)
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
    from mani_skill.utils.structs import Actor, Articulation
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    torch = None
    Actor = None
    Articulation = None
    Counter = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


M11_SETTINGS = TaskRegistry.register(TaskRegistry.load("m11_v1"))


@dataclass(frozen=True)
class M11SuccessSpec:
    door_open_angle_threshold_rad: float = math.radians(45.0)
    grasp_contact_force_n: float = 0.40
    grasp_min_handle_fingers: int = 1
    grasp_min_mug_fingers: int = 2
    jar_max_tilt_rad: float = math.radians(15.0)
    jar_stable_linear_speed_mps: float = 0.05
    jar_stable_angular_speed_rad_s: float = 0.60
    residual_hand_contact_force_n: float = 0.10
    final_dwell_s: float = 0.40


M11_SUCCESS_SPEC = M11SuccessSpec()

M11_INTERACTION_SITE_ID = "cabinet"
M11_CABINET_ENTITY_NAME = "cabinet"
M11_CABINET_CLUTTER_NAMES = frozenset(("cup1",))
M11_RELOCATED_CLUTTER_NAME = "cup2"
M11_RELOCATED_CLUTTER_LOCAL_Y_M = 0.25
M11_LEFT_DOOR_JOINT_NAME = "leftdoorhinge"
M11_RIGHT_DOOR_JOINT_NAME = "rightdoorhinge"
M11_MUG_ASSET_ID = "procedural_large_handle_mug"
# Scale both visual and collision geometry to 115% of the unscaled procedural
# mug dimensions.
M11_MUG_SCALE: float = 1.15
M11_MUG_STATIC_FRICTION: float = 1.0
M11_MUG_DYNAMIC_FRICTION: float = 0.9
# Coarse extrema used for spawn height and size assertions. Placement and door-
# plane checks use a tighter compound-geometry envelope generated below, rather
# than treating every corner of this whole-object AABB as occupied geometry.
M11_MUG_COLLISION_MIN_XYZ_M = (
    -0.042,
    -0.080,
    -0.045,
)
M11_MUG_COLLISION_MAX_XYZ_M = (
    0.080,
    0.042,
    0.045,
)
# The previous YCB mug left only a nominal 22 x 40 mm collision aperture. That
# admitted an ideal fingertip ray but not two Shadow fingers with joint and
# approach-angle clearance. This fully procedural mug keeps visual and collision
# geometry identical and enlarges the C-handle opening to roughly 56 x 76 mm.
M11_MUG_COLLISION_DENSITY_KG_M3 = 1000.0
M11_MUG_CONTACT_OFFSET_M = 0.0002
M11_MUG_VISUAL_COLOR_RGBA = (0.28, 0.55, 0.78, 1.0)
M11_MUG_BODY_CENTER_XY_M = (0.0, 0.0)
M11_MUG_BODY_WALL_SEGMENT_COUNT = 20
M11_MUG_BODY_WALL_CENTER_RADIUS_M = 0.0370
M11_MUG_BODY_WALL_RADIAL_HALF_THICKNESS_M = 0.0040
M11_MUG_BODY_WALL_TANGENT_HALF_LENGTH_M = 0.0061
M11_MUG_BODY_WALL_CENTER_Z_M = 0.0
M11_MUG_BODY_WALL_HALF_HEIGHT_M = 0.0420
M11_MUG_BOTTOM_RADIUS_M = 0.0380
M11_MUG_BOTTOM_CENTER_Z_M = -0.0420
M11_MUG_BOTTOM_HALF_HEIGHT_M = 0.0030
M11_MUG_HANDLE_AUTHORED_OUTWARD_XY = (
    math.sqrt(0.5),
    -math.sqrt(0.5),
)
M11_MUG_HANDLE_CONNECTOR_CENTER_RADIUS_M = 0.0690
M11_MUG_HANDLE_CONNECTOR_CENTER_Z_M = 0.0390
M11_MUG_HANDLE_CONNECTOR_HALF_LENGTH_M = 0.0270
M11_MUG_HANDLE_COLLISION_RADIUS_M = 0.0060
M11_MUG_HANDLE_OUTER_CENTER_RADIUS_M = 0.0960
M11_MUG_HANDLE_OUTER_HALF_LENGTH_M = 0.0390
M11_MUG_HANDLE_HOLE_RADIAL_CLEARANCE_M = (
    M11_MUG_HANDLE_OUTER_CENTER_RADIUS_M
    - M11_MUG_HANDLE_COLLISION_RADIUS_M
    - M11_MUG_BODY_WALL_CENTER_RADIUS_M
    - M11_MUG_BODY_WALL_RADIAL_HALF_THICKNESS_M
) * M11_MUG_SCALE
M11_MUG_HANDLE_HOLE_VERTICAL_CLEARANCE_M = 2.0 * (
    M11_MUG_HANDLE_CONNECTOR_CENTER_Z_M
    - M11_MUG_HANDLE_COLLISION_RADIUS_M
) * M11_MUG_SCALE
# Shared collision-free cabinet placement calibrated in kitchen L1/L2/L3. The
# task-private layout removes the unrelated cup1. Local X=0.03 keeps the mug
# safely behind the door plane, while local Y=-0.15 places it completely behind
# the right door from the robot's viewpoint. Cabinet-site local -Y points toward
# the robot's right.
M11_MUG_INITIAL_LOCAL_XY_M = (0.03, -0.15)
M11_MUG_SHELF_SURFACE_LOCAL_Z_M = 0.0150
M11_MUG_INITIAL_CLEARANCE_M = 0.0020
M11_MUG_ROOT_Z_M = (
    M11_MUG_SHELF_SURFACE_LOCAL_Z_M
    - M11_MUG_COLLISION_MIN_XYZ_M[2] * M11_MUG_SCALE
    + M11_MUG_INITIAL_CLEARANCE_M
)
# The authored handle points approximately along local (+X, -Y). The base yaw
# points it toward cabinet-site -X; the additional positive yaw rotates the mug
# counterclockwise around its local Z axis when viewed from above.
M11_MUG_HANDLE_OUTWARD_YAW_RAD = -3.0 * math.pi / 4.0
M11_MUG_COUNTERCLOCKWISE_YAW_OFFSET_RAD = math.radians(15.0)
M11_MUG_INITIAL_YAW_RAD = (
    M11_MUG_HANDLE_OUTWARD_YAW_RAD
    + M11_MUG_COUNTERCLOCKWISE_YAW_OFFSET_RAD
)
# The short cabinet-local X edge is resolved from each layout's real counter
# footprint. Keep only a broad lateral strip so the target stays beneath the
# cabinet instead of spilling onto an adjacent worktop.
M11_COUNTER_REGION_LATERAL_HALF_WIDTH_M = 0.48
M11_COUNTER_SURFACE_PROBE_LOCAL_XYZ_M = (-0.20, 0.0, 0.0)
M11_COUNTER_FIXTURE_MAX_PROBE_DISTANCE_M = 0.05
M11_COUNTER_SURFACE_HEIGHT_TOLERANCE_M = 0.03

# The cabinet sites in kitchen_l1/l2/l3 all face the robot along local -X.
# The opening plane is first estimated from the two closed door child-link origins.
# If an asset's link origins are not located on the door plane, fall back to a
# conservative plane outside the mug's initial collision bounds.
M11_OUTWARD_LOCAL_AXIS = 0
M11_OPENING_PLANE_FALLBACK_GAP_M = 0.15

# Explicit handle-link hooks. The current project source does not expose reliable
# RoboCasa cabinet handle link names, so None retains the documented door-panel
# contact proxy. Set both names only after asset-level handle calibration.
M11_LEFT_DOOR_HANDLE_LINK_NAME: str | None = None
M11_RIGHT_DOOR_HANDLE_LINK_NAME: str | None = None

M11_GRASP_FINGER_NAMES = ("th", "ff", "mf", "rf", "lf")
M11_GRASP_FINGER_LINK_PARTS = ("proximal", "middle", "distal")
M11_GRASP_FINGER_LINK_GROUPS = tuple(
    tuple(
        f"rh_{finger_name}{link_part}"
        for link_part in M11_GRASP_FINGER_LINK_PARTS
    )
    for finger_name in M11_GRASP_FINGER_NAMES
)
M11_GRASP_FINGER_LINK_NAMES = tuple(
    link_name
    for finger_link_names in M11_GRASP_FINGER_LINK_GROUPS
    for link_name in finger_link_names
)

M11_SPATIAL_PARAMETER_NAMES = (
    "jar_x_offset_m",
    "jar_y_offset_m",
    "jar_yaw_offset_rad",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
M11_PHYSICAL_PARAMETER_NAMES = (
    "jar_mass_log_scale",
    "jar_friction_log_scale",
    "door_joint_friction",
    "door_joint_damping",
)
M11_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("jar_x_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec("jar_y_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec(
                "jar_yaw_offset_rad",
                -math.radians(10.0),
                math.radians(10.0),
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
                "jar_mass_log_scale", math.log(0.80), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "jar_friction_log_scale", math.log(0.75), math.log(1.35)
            ),
            GeneralizationParameterSpec("door_joint_friction", 0.02, 0.16),
            GeneralizationParameterSpec("door_joint_damping", 0.03, 0.20),
        ),
        requires_rebuild=True,
    ),
)

M11_REENTRY_CLEARANCE_M = 0.02
M11_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.30, -1.00, -0.80)
M11_TCP_WORKSPACE_LOCAL_MAX_XYZ = (0.85, 1.00, 1.00)

# M11-specific robot-base calibration in the cabinet interaction-site frame.
# Every kitchen layout therefore preserves the accepted L1 relationship to the
# cabinet without changing the shared layout definitions.
M11_ROBOT_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(-1.119998, -0.07, -0.355),
)
M11_HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
    # Keep one explicit world pose per layout so each Human Viewer can be tuned
    # independently without changing the other kitchen environments.
    LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
        position_world_xyz=(
            5.375271, -0.732659, 1.657115
        ),
        quaternion_world_wxyz=(
            0.960928, 0.007567, 0.026434, -0.275431
        ),
    ),
    LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
        position_world_xyz=(
            0.758452, -1.012041, 1.652123
        ),
        quaternion_world_wxyz=(
            0.845205, -0.042372, 0.067755, 0.528433
        ),
    ),
    LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
        position_world_xyz=(0.874590, -2.181834, 1.642138),
        # Equivalent viewer RPY: (-0.000019, 0.279998, 2.293299) rad.
        quaternion_world_wxyz=(
            0.407516,
            -0.127181,
            0.057419,
            0.902474,
        ),
    ),
}


@register_env(M11_SETTINGS.env_id, max_episode_steps=M11_SETTINGS.max_steps)
class M11Env(DexVLABenchBaseEnv):
    """Open a cabinet and place its handled mug on the real counter below."""

    TASK_SETTINGS = M11_SETTINGS
    SUCCESS_SPEC = M11_SUCCESS_SPEC
    GENERALIZATION_SPEC = M11_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = M11_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = M11_HUMAN_VIEWER_CAMERA_POSES
    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        nominal_pose_spec = self.interaction_site_local_pose_to_world(
            M11_ROBOT_BASE_LOCAL_POSE
        )
        nominal_pose = pose_spec_to_sapien_pose(nominal_pose_spec)
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_pose

        values = self._generalization_values(
            M11_SPATIAL_PARAMETER_NAMES,
            cache_name="_m11_spatial_values",
        )
        if env_idx is None:
            value_indices = np.arange(len(values["robot_forward_offset_m"]))
        else:
            value_indices = torch.as_tensor(
                env_idx, device=self.device, dtype=torch.long
            ).reshape(-1).detach().cpu().numpy()

        forward = values["robot_forward_offset_m"][value_indices]
        side = values["robot_side_offset_m"][value_indices]
        yaw = values["robot_yaw_offset_rad"][value_indices]
        local_position = np.stack(
            [forward, side, np.zeros_like(forward)], axis=1
        )
        half_yaw = 0.5 * yaw
        local_quaternion = np.stack(
            [
                np.cos(half_yaw),
                np.zeros_like(half_yaw),
                np.zeros_like(half_yaw),
                np.sin(half_yaw),
            ],
            axis=1,
        )
        offset_pose = Pose.create_from_pq(
            p=local_position, q=local_quaternion
        )
        return Pose.create(nominal_pose) * offset_pose

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _task_layout_spec(self) -> Any:
        cabinet_frame = self.current_layout_spec.interaction_site(
            M11_INTERACTION_SITE_ID
        ).frame_pose
        cabinet_position = np.asarray(
            cabinet_frame.position_xyz, dtype=np.float64
        )
        cabinet_rotation = cabinet_frame.rotation_matrix.astype(np.float64)
        task_items: list[LayoutBuildItemSpec] = []
        relocated_clutter_count = 0
        for item in self.current_layout_spec.build_items:
            item_name = getattr(item, "name", None)
            if item_name in M11_CABINET_CLUTTER_NAMES:
                continue
            if item_name == M11_RELOCATED_CLUTTER_NAME:
                if not isinstance(item, ArticulatedAssetSpec):
                    raise TypeError(
                        f"M11 clutter {item_name!r} must be an articulated "
                        f"asset, got {type(item).__name__}."
                    )
                world_position = np.asarray(
                    item.pose.position_xyz, dtype=np.float64
                )
                # [3] -> [3]: preserve the authored depth and height while
                # moving the distractor into the left-door half of the cabinet.
                local_position = cabinet_rotation.T @ (
                    world_position - cabinet_position
                )
                local_position[1] = M11_RELOCATED_CLUTTER_LOCAL_Y_M
                relocated_world_position = cabinet_position + (
                    cabinet_rotation @ local_position
                )
                item = replace(
                    item,
                    pose=replace(
                        item.pose,
                        position_xyz=tuple(
                            float(value) for value in relocated_world_position
                        ),
                    ),
                )
                relocated_clutter_count += 1
            task_items.append(item)

        if relocated_clutter_count != 1:
            raise RuntimeError(
                "M11 expected exactly one task-private clutter item named "
                f"{M11_RELOCATED_CLUTTER_NAME!r}, found "
                f"{relocated_clutter_count}."
            )
        return replace(
            self.current_layout_spec,
            build_items=tuple(task_items),
        )

    def _require_cabinet_entity(self) -> Any:
        interaction_site = self.require_current_interaction_site()
        target_entity_name = interaction_site.target_entity_name
        if not target_entity_name:
            target_entity_name = M11_CABINET_ENTITY_NAME

        if interaction_site.target_entity_source == "external_fixture":
            external_builders = getattr(
                self.layout_scene,
                "external_scene_builders",
                None,
            )
            if not external_builders:
                raise RuntimeError(
                    "M11 requires an external RoboCasa scene builder when the "
                    "interaction site targets an external fixture."
                )

            target_normalized = target_entity_name.casefold()
            cabinet_articulations: list[Any] = []
            for builder in external_builders.values():
                scene_data = getattr(builder, "scene_data", None) or ()
                for scene_index, scene_entry in enumerate(scene_data):
                    fixtures = (
                        scene_entry.get("fixtures", {})
                        if isinstance(scene_entry, dict)
                        else {}
                    )
                    matches = [
                        fixture
                        for fixture_name, fixture in fixtures.items()
                        if str(fixture_name).casefold() == target_normalized
                    ]
                    if len(matches) != 1:
                        raise RuntimeError(
                            "M11 expected exactly one cabinet fixture named "
                            f"{target_entity_name!r} in RoboCasa scene "
                            f"{scene_index}, found {len(matches)}. Known "
                            f"fixtures: {sorted(map(str, fixtures))}."
                        )
                    fixture = matches[0]
                    articulation = getattr(fixture, "articulation", None)
                    if articulation is None and hasattr(
                        fixture,
                        "active_joints_map",
                    ):
                        articulation = fixture
                    if articulation is None:
                        raise RuntimeError(
                            f"RoboCasa cabinet fixture {target_entity_name!r} "
                            f"in scene {scene_index} is not articulated."
                        )
                    cabinet_articulations.append(articulation)

            if len(cabinet_articulations) != self.num_envs:
                raise RuntimeError(
                    "M11 cabinet articulation count does not match the "
                    f"environment count: articulations="
                    f"{len(cabinet_articulations)}, num_envs={self.num_envs}."
                )
            if self.num_envs == 1:
                return cabinet_articulations[0]
            if Articulation is None:
                raise ImportError(
                    "ManiSkill Articulation is required to merge batched "
                    "RoboCasa cabinet fixtures."
                )

            expected_joint_names = tuple(
                cabinet_articulations[0].active_joints_map
            )
            for scene_index, articulation in enumerate(
                cabinet_articulations[1:],
                start=1,
            ):
                joint_names = tuple(articulation.active_joints_map)
                if joint_names != expected_joint_names:
                    raise RuntimeError(
                        "M11 cabinet active joints differ across RoboCasa "
                        f"scenes: scene 0={expected_joint_names}, scene "
                        f"{scene_index}={joint_names}."
                    )
            return Articulation.merge(
                cabinet_articulations,
                name="m11_cabinet",
                merge_links=True,
            )

        return self.layout_scene.require_entity(target_entity_name)

    def _resolve_door_joint(self, joint_name: str) -> Any:
        try:
            return self.cabinet.active_joints_map[joint_name]
        except KeyError as exc:
            known_joint_names = ", ".join(sorted(self.cabinet.active_joints_map))
            raise RuntimeError(
                f"Cabinet in layout {self.current_layout_id_value!r} does not "
                f"expose expected door joint {joint_name!r}. Known joints: "
                f"{known_joint_names}."
            ) from exc

    @staticmethod
    def _rpy_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = (float(value) for value in rpy)
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        rx = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)),
            dtype=np.float64,
        )
        ry = np.asarray(
            ((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)),
            dtype=np.float64,
        )
        rz = np.asarray(
            ((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        return rz @ ry @ rx

    @staticmethod
    def _aabb_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                (x, y, z)
                for x in (lower[0], upper[0])
                for y in (lower[1], upper[1])
                for z in (lower[2], upper[2])
            ],
            dtype=np.float64,
        )

    @classmethod
    def _mug_collision_envelope_points_local(cls) -> np.ndarray:
        """Return a tight conservative envelope of the compound mug collision."""

        point_sets: list[np.ndarray] = []

        def append_oriented_box(
            center_xyz: tuple[float, float, float] | np.ndarray,
            half_size_xyz: tuple[float, float, float] | np.ndarray,
            *,
            yaw_rad: float = 0.0,
        ) -> None:
            half_size = np.asarray(half_size_xyz, dtype=np.float64)
            corners = cls._aabb_corners(-half_size, half_size)
            cosine = math.cos(yaw_rad)
            sine = math.sin(yaw_rad)
            rotation = np.asarray(
                (
                    (cosine, -sine, 0.0),
                    (sine, cosine, 0.0),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float64,
            )
            # [8, 3] @ [3, 3] -> [8, 3].
            point_sets.append(
                corners @ rotation.T
                + np.asarray(center_xyz, dtype=np.float64)[None, :]
            )

        body_center_xy = np.asarray(M11_MUG_BODY_CENTER_XY_M, dtype=np.float64)
        for segment_index in range(M11_MUG_BODY_WALL_SEGMENT_COUNT):
            yaw = (
                2.0
                * math.pi
                * segment_index
                / M11_MUG_BODY_WALL_SEGMENT_COUNT
            )
            radial_xy = np.asarray((math.cos(yaw), math.sin(yaw)))
            segment_center_xy = body_center_xy + (
                M11_MUG_BODY_WALL_CENTER_RADIUS_M * radial_xy
            )
            append_oriented_box(
                (
                    float(segment_center_xy[0]),
                    float(segment_center_xy[1]),
                    M11_MUG_BODY_WALL_CENTER_Z_M,
                ),
                (
                    M11_MUG_BODY_WALL_RADIAL_HALF_THICKNESS_M,
                    M11_MUG_BODY_WALL_TANGENT_HALF_LENGTH_M,
                    M11_MUG_BODY_WALL_HALF_HEIGHT_M,
                ),
                yaw_rad=yaw,
            )

        # A circumscribed regular polygon contains the complete bottom cylinder
        # while avoiding the four empty corners of a square cylinder AABB.
        ring_point_count = max(16, M11_MUG_BODY_WALL_SEGMENT_COUNT)
        ring_radius = M11_MUG_BOTTOM_RADIUS_M / math.cos(
            math.pi / ring_point_count
        )
        ring_angles = (
            2.0
            * math.pi
            * np.arange(ring_point_count, dtype=np.float64)
            / ring_point_count
        )
        ring_xy = np.stack(
            (np.cos(ring_angles), np.sin(ring_angles)), axis=1
        ) * ring_radius
        ring_xy += body_center_xy[None, :]
        for z_offset in (-M11_MUG_BOTTOM_HALF_HEIGHT_M, M11_MUG_BOTTOM_HALF_HEIGHT_M):
            # [N, 2] -> [N, 3].
            point_sets.append(
                np.concatenate(
                    (
                        ring_xy,
                        np.full(
                            (ring_point_count, 1),
                            M11_MUG_BOTTOM_CENTER_Z_M + z_offset,
                            dtype=np.float64,
                        ),
                    ),
                    axis=1,
                )
            )

        handle_outward_xy = np.asarray(
            M11_MUG_HANDLE_AUTHORED_OUTWARD_XY,
            dtype=np.float64,
        )
        handle_yaw = math.atan2(handle_outward_xy[1], handle_outward_xy[0])
        connector_center_xy = body_center_xy + (
            M11_MUG_HANDLE_CONNECTOR_CENTER_RADIUS_M * handle_outward_xy
        )
        connector_radius = M11_MUG_HANDLE_COLLISION_RADIUS_M
        for connector_z_sign in (-1.0, 1.0):
            # This oriented box tightly contains the horizontal capsule. Its
            # small radius adds at most millimetres of conservatism, instead of
            # filling the large empty corners around the complete mug.
            append_oriented_box(
                (
                    float(connector_center_xy[0]),
                    float(connector_center_xy[1]),
                    connector_z_sign * M11_MUG_HANDLE_CONNECTOR_CENTER_Z_M,
                ),
                (
                    M11_MUG_HANDLE_CONNECTOR_HALF_LENGTH_M + connector_radius,
                    connector_radius,
                    connector_radius,
                ),
                yaw_rad=handle_yaw,
            )

        outer_center_xy = body_center_xy + (
            M11_MUG_HANDLE_OUTER_CENTER_RADIUS_M * handle_outward_xy
        )
        append_oriented_box(
            (float(outer_center_xy[0]), float(outer_center_xy[1]), 0.0),
            (
                connector_radius,
                connector_radius,
                M11_MUG_HANDLE_OUTER_HALF_LENGTH_M + connector_radius,
            ),
        )

        # [N_i, 3] -> [sum(N_i), 3], expressed in the mug root frame.
        return np.concatenate(point_sets, axis=0) * M11_MUG_SCALE

    @staticmethod
    def _add_mug_compound_geometry(
        body_builder: Any,
        handle_builder: Any,
        collision_material: Any,
        visual_material: Any,
    ) -> None:
        """Add matching hollow visual/collision geometry with a large handle."""

        scale = M11_MUG_SCALE
        body_center_xy = np.asarray(M11_MUG_BODY_CENTER_XY_M, dtype=np.float64)
        for segment_index in range(M11_MUG_BODY_WALL_SEGMENT_COUNT):
            yaw = (
                2.0
                * math.pi
                * segment_index
                / M11_MUG_BODY_WALL_SEGMENT_COUNT
            )
            radial_xy = np.asarray((math.cos(yaw), math.sin(yaw)))
            segment_center_xy = body_center_xy + (
                M11_MUG_BODY_WALL_CENTER_RADIUS_M * radial_xy
            )
            segment_pose = sapien.Pose(
                p=(
                    float(segment_center_xy[0]) * scale,
                    float(segment_center_xy[1]) * scale,
                    M11_MUG_BODY_WALL_CENTER_Z_M * scale,
                ),
                q=(math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)),
            )
            segment_half_size = (
                M11_MUG_BODY_WALL_RADIAL_HALF_THICKNESS_M * scale,
                M11_MUG_BODY_WALL_TANGENT_HALF_LENGTH_M * scale,
                M11_MUG_BODY_WALL_HALF_HEIGHT_M * scale,
            )
            body_builder.add_box_visual(
                pose=segment_pose,
                half_size=segment_half_size,
                material=visual_material,
                name=f"m11_mug_wall_{segment_index}",
            )
            body_builder.add_box_collision(
                pose=segment_pose,
                half_size=segment_half_size,
                material=collision_material,
                density=M11_MUG_COLLISION_DENSITY_KG_M3,
            )

        # SAPIEN cylinder/capsule primitives use local X as their long axis.
        # A +90-degree Y rotation aligns the bottom cylinder with mug-local Z.
        axis_x_to_z_quaternion = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
        bottom_pose = sapien.Pose(
            p=(
                M11_MUG_BODY_CENTER_XY_M[0] * scale,
                M11_MUG_BODY_CENTER_XY_M[1] * scale,
                M11_MUG_BOTTOM_CENTER_Z_M * scale,
            ),
            q=axis_x_to_z_quaternion,
        )
        bottom_radius = M11_MUG_BOTTOM_RADIUS_M * scale
        bottom_half_height = M11_MUG_BOTTOM_HALF_HEIGHT_M * scale
        body_builder.add_cylinder_visual(
            pose=bottom_pose,
            radius=bottom_radius,
            half_length=bottom_half_height,
            material=visual_material,
            name="m11_mug_bottom",
        )
        body_builder.add_cylinder_collision(
            pose=bottom_pose,
            radius=bottom_radius,
            half_length=bottom_half_height,
            material=collision_material,
            density=M11_MUG_COLLISION_DENSITY_KG_M3,
        )

        handle_outward_xy = np.asarray(
            M11_MUG_HANDLE_AUTHORED_OUTWARD_XY,
            dtype=np.float64,
        )
        handle_yaw = math.atan2(handle_outward_xy[1], handle_outward_xy[0])
        connector_center_xy = body_center_xy + (
            M11_MUG_HANDLE_CONNECTOR_CENTER_RADIUS_M * handle_outward_xy
        )
        connector_quaternion = (
            math.cos(0.5 * handle_yaw),
            0.0,
            0.0,
            math.sin(0.5 * handle_yaw),
        )
        for connector_z_sign in (-1.0, 1.0):
            connector_pose = sapien.Pose(
                p=(
                    float(connector_center_xy[0]) * scale,
                    float(connector_center_xy[1]) * scale,
                    connector_z_sign
                    * M11_MUG_HANDLE_CONNECTOR_CENTER_Z_M
                    * scale,
                ),
                q=connector_quaternion,
            )
            connector_radius = M11_MUG_HANDLE_COLLISION_RADIUS_M * scale
            connector_half_length = (
                M11_MUG_HANDLE_CONNECTOR_HALF_LENGTH_M * scale
            )
            connector_name = "upper" if connector_z_sign > 0.0 else "lower"
            handle_builder.add_capsule_visual(
                pose=connector_pose,
                radius=connector_radius,
                half_length=connector_half_length,
                material=visual_material,
                name=f"m11_mug_handle_{connector_name}",
            )
            handle_builder.add_capsule_collision(
                pose=connector_pose,
                radius=connector_radius,
                half_length=connector_half_length,
                material=collision_material,
                density=M11_MUG_COLLISION_DENSITY_KG_M3,
            )

        outer_center_xy = body_center_xy + (
            M11_MUG_HANDLE_OUTER_CENTER_RADIUS_M * handle_outward_xy
        )
        outer_handle_pose = sapien.Pose(
            p=(
                float(outer_center_xy[0]) * scale,
                float(outer_center_xy[1]) * scale,
                0.0,
            ),
            q=axis_x_to_z_quaternion,
        )
        handle_radius = M11_MUG_HANDLE_COLLISION_RADIUS_M * scale
        outer_half_length = M11_MUG_HANDLE_OUTER_HALF_LENGTH_M * scale
        handle_builder.add_capsule_visual(
            pose=outer_handle_pose,
            radius=handle_radius,
            half_length=outer_half_length,
            material=visual_material,
            name="m11_mug_handle_outer",
        )
        handle_builder.add_capsule_collision(
            pose=outer_handle_pose,
            radius=handle_radius,
            half_length=outer_half_length,
            material=collision_material,
            density=M11_MUG_COLLISION_DENSITY_KG_M3,
        )

    def _load_jar(self) -> None:
        """Build the small procedural cup with an oversized usable handle."""

        if sapien is None:
            raise ImportError("M11 procedural mug loading requires SAPIEN.")

        builder = self.scene.create_articulation_builder()
        body_builder = builder.create_link_builder(parent=None)
        body_builder.set_name("m11_mug_body")
        handle_builder = builder.create_link_builder(parent=body_builder)
        handle_builder.set_name("m11_mug_handle")
        handle_builder.set_joint_name("m11_mug_handle_fixed_joint")
        handle_builder.set_joint_properties(
            type="fixed",
            limits=[],
            pose_in_parent=sapien.Pose(),
            pose_in_child=sapien.Pose(),
        )
        visual_material = sapien.render.RenderMaterial(
            base_color=M11_MUG_VISUAL_COLOR_RGBA,
            roughness=0.55,
            specular=0.25,
        )
        collision_material = sapien.physx.PhysxMaterial(
            static_friction=M11_MUG_STATIC_FRICTION,
            dynamic_friction=M11_MUG_DYNAMIC_FRICTION,
            restitution=0.0,
        )
        self._add_mug_compound_geometry(
            body_builder,
            handle_builder,
            collision_material,
            visual_material,
        )

        interaction_site = self.interaction_site_frame_pose()
        local_pose = sapien.Pose(
            p=(*M11_MUG_INITIAL_LOCAL_XY_M, M11_MUG_ROOT_Z_M),
            q=(
                math.cos(0.5 * M11_MUG_INITIAL_YAW_RAD),
                0.0,
                0.0,
                math.sin(0.5 * M11_MUG_INITIAL_YAW_RAD),
            ),
        )
        builder.initial_pose = sapien.Pose(
            p=interaction_site.position_xyz,
            q=interaction_site.quaternion_wxyz,
        ) * local_pose
        self.jar = builder.build(name="m11_mug")
        try:
            self._mug_body_link = self.jar.links_map["m11_mug_body"]
            self._mug_handle_link = self.jar.links_map["m11_mug_handle"]
        except KeyError as exc:
            raise RuntimeError(
                "M11 procedural mug articulation did not expose its body and "
                "handle links."
            ) from exc
        self._set_collision_contact_offset(
            tuple(self.jar.links),
            M11_MUG_CONTACT_OFFSET_M,
        )
        self._jar_collision_corners_local = (
            self._mug_collision_envelope_points_local()
        )

    def _set_collision_contact_offset(
        self,
        entities: tuple[Any, ...],
        contact_offset_m: float,
    ) -> None:
        """Reduce PhysX's speculative contact envelope for fine insertion."""

        if contact_offset_m <= 0.0:
            raise ValueError("Collision contact offset must be positive.")
        for entity in entities:
            raw_bodies = tuple(getattr(entity, "_bodies", ()))
            if len(raw_bodies) != self.num_envs:
                raise RuntimeError(
                    f"M11 collision entity {entity.name!r} exposes "
                    f"{len(raw_bodies)} raw bodies; expected {self.num_envs}."
                )
            for raw_body in raw_bodies:
                component = self._rigid_component_for_raw_body(raw_body)
                if component is None:
                    raise RuntimeError(
                        "Could not resolve an M11 rigid component while setting "
                        "contact offsets."
                    )
                shape_getter = getattr(component, "get_collision_shapes", None)
                shapes = (
                    tuple(shape_getter())
                    if callable(shape_getter)
                    else tuple(getattr(component, "collision_shapes", ()))
                )
                for shape in shapes:
                    shape.set_contact_offset(contact_offset_m)
                    shape.set_rest_offset(0.0)

    def _jar_contact_entities(self) -> tuple[Any, ...]:
        links = getattr(self.jar, "links", None)
        if links is not None:
            return tuple(links)
        return (self.jar,)

    def _jar_pose(self) -> Any:
        root_pose = getattr(self.jar, "root_pose", None)
        if root_pose is not None:
            return root_pose
        pose = getattr(self.jar, "pose", None)
        if pose is None:
            raise TypeError("M11 mug entity exposes neither root_pose nor pose.")
        return pose

    @staticmethod
    def _rigid_component_for_raw_body(raw_body: Any) -> Any | None:
        if hasattr(raw_body, "mass") and (
            hasattr(raw_body, "get_collision_shapes")
            or hasattr(raw_body, "collision_shapes")
        ):
            return raw_body
        candidates = list(getattr(raw_body, "components", ()))
        raw_entity = getattr(raw_body, "entity", None)
        if raw_entity is not None:
            candidates.extend(getattr(raw_entity, "components", ()))
        return next(
            (
                component
                for component in candidates
                if hasattr(component, "mass")
                and (
                    hasattr(component, "get_collision_shapes")
                    or hasattr(component, "collision_shapes")
                )
            ),
            None,
        )

    def _apply_jar_physical_randomization(self) -> None:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return
        values = self._generalization_values(
            M11_PHYSICAL_PARAMETER_NAMES, cache_name="_m11_physical_values"
        )
        mass_scales = positive_scale(values["jar_mass_log_scale"])
        friction_scales = positive_scale(values["jar_friction_log_scale"])

        for rigid_entity in self._jar_contact_entities():
            raw_bodies = tuple(getattr(rigid_entity, "_bodies", ()))
            if len(raw_bodies) != self.num_envs:
                raise RuntimeError(
                    f"M11 mug entity {rigid_entity.name!r} exposes "
                    f"{len(raw_bodies)} raw "
                    f"bodies; expected {self.num_envs}."
                )
            for env_index, raw_body in enumerate(raw_bodies):
                component = self._rigid_component_for_raw_body(raw_body)
                if component is None:
                    raise RuntimeError(
                        "Could not resolve rigid component for M11 mug entity "
                        f"{rigid_entity.name!r} env {env_index}."
                    )
                nominal_mass = float(component.mass)
                if nominal_mass > 0.0:
                    target_mass = nominal_mass * float(mass_scales[env_index])
                    setter = getattr(component, "set_mass", None)
                    if callable(setter):
                        setter(target_mass)
                    else:
                        component.mass = target_mass

                friction_scale = float(friction_scales[env_index])
                try:
                    material = sapien.physx.PhysxMaterial(
                        static_friction=(
                            M11_MUG_STATIC_FRICTION * friction_scale
                        ),
                        dynamic_friction=(
                            M11_MUG_DYNAMIC_FRICTION * friction_scale
                        ),
                        restitution=0.0,
                    )
                except TypeError:
                    material = sapien.physx.PhysxMaterial(
                        M11_MUG_STATIC_FRICTION * friction_scale,
                        M11_MUG_DYNAMIC_FRICTION * friction_scale,
                        0.0,
                    )
                shape_getter = getattr(component, "get_collision_shapes", None)
                shapes = (
                    tuple(shape_getter())
                    if callable(shape_getter)
                    else tuple(getattr(component, "collision_shapes", ()))
                )
                for shape in shapes:
                    setter = getattr(shape, "set_physical_material", None)
                    if callable(setter):
                        setter(material)
                    else:
                        shape.physical_material = material

    def _apply_door_joint_physical_randomization(self) -> None:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return
        values = self._generalization_values(
            M11_PHYSICAL_PARAMETER_NAMES, cache_name="_m11_physical_values"
        )
        for joint in (self.left_door_joint, self.right_door_joint):
            joint_objects = tuple(getattr(joint, "_objs", ()))
            if len(joint_objects) != self.num_envs:
                raise RuntimeError(
                    f"M11 door joint {joint.name!r} exposes {len(joint_objects)} "
                    f"underlying joints; expected {self.num_envs}."
                )
            for env_index, joint_object in enumerate(joint_objects):
                joint_object.set_friction(
                    float(values["door_joint_friction"][env_index])
                )
                joint_object.set_drive_properties(
                    stiffness=0.0,
                    damping=float(values["door_joint_damping"][env_index]),
                    force_limit=1e10,
                )

    def _place_jar(self, env_idx: Any) -> None:
        env_idx = torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long
        ).reshape(-1)
        batch_size = len(env_idx)
        local_position = torch.zeros(
            (batch_size, 3), device=self.device, dtype=torch.float32
        )
        local_position[:, :2] = torch.as_tensor(
            M11_MUG_INITIAL_LOCAL_XY_M,
            device=self.device,
            dtype=torch.float32,
        )
        local_position[:, 2] = M11_MUG_ROOT_Z_M
        local_yaw = torch.full(
            (batch_size,),
            M11_MUG_INITIAL_YAW_RAD,
            device=self.device,
            dtype=torch.float32,
        )
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M11_SPATIAL_PARAMETER_NAMES, cache_name="_m11_spatial_values"
            )
            value_indices = env_idx.detach().cpu().numpy()
            local_position[:, 0] += torch.as_tensor(
                values["jar_x_offset_m"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )
            local_position[:, 1] += torch.as_tensor(
                values["jar_y_offset_m"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )
            local_yaw += torch.as_tensor(
                values["jar_yaw_offset_rad"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )

        half_yaw = 0.5 * local_yaw
        local_quaternion = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ],
            dim=1,
        )
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                local_xyz=(0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )
        jar_pose = site_pose * Pose.create_from_pq(
            p=local_position, q=local_quaternion
        )
        self.jar.set_pose(jar_pose)

    def _reset_cabinet_doors(self, env_idx: Any) -> None:
        env_idx = torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long
        ).reshape(-1)
        qpos = self._cabinet_episode_reset_qpos[env_idx].clone()
        qpos[:, self._left_door_qpos_index] = 0.0
        qpos[:, self._right_door_qpos_index] = 0.0
        self.cabinet.set_qpos(qpos)
        self.cabinet.set_qvel(torch.zeros_like(qpos))

    def _reset_jar_joints(self, env_idx: Any) -> None:
        """Reset joints only if a future replacement is articulated."""

        if getattr(self.jar, "max_dof", 0) == 0:
            return
        # [B, J] -> [b, J]: preserve non-reset parallel environments and match
        # the number of rows selected by ManiSkill's GPU reset mask.
        qpos = self.jar.get_qpos()[env_idx].clone()
        qvel = self.jar.get_qvel()[env_idx].clone()
        qpos.zero_()
        qvel.zero_()
        self.jar.set_qpos(qpos)
        self.jar.set_qvel(qvel)

    def _zero_jar_root_velocity(self, batch_size: int) -> None:
        zero_velocity = torch.zeros(
            (batch_size, 3),
            device=self.device,
            dtype=torch.float32,
        )
        linear_setter = getattr(self.jar, "set_root_linear_velocity", None)
        if not callable(linear_setter):
            linear_setter = getattr(self.jar, "set_linear_velocity", None)
        angular_setter = getattr(self.jar, "set_root_angular_velocity", None)
        if not callable(angular_setter):
            angular_setter = getattr(self.jar, "set_angular_velocity", None)
        if not callable(linear_setter) or not callable(angular_setter):
            raise TypeError("M11 mug entity exposes no velocity setters.")
        linear_setter(zero_velocity)
        angular_setter(zero_velocity)

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self._task_layout_spec(),
        ).build()

        self.cabinet = self._require_cabinet_entity()
        self.left_door_joint = self._resolve_door_joint(M11_LEFT_DOOR_JOINT_NAME)
        self.right_door_joint = self._resolve_door_joint(M11_RIGHT_DOOR_JOINT_NAME)
        active_joint_names = tuple(self.cabinet.active_joints_map)
        self._left_door_qpos_index = active_joint_names.index(
            self.left_door_joint.name
        )
        self._right_door_qpos_index = active_joint_names.index(
            self.right_door_joint.name
        )
        self._load_jar()
        self.counter_surface = self._require_counter_surface()

        # Material/joint randomization is applied to the actual runtime entities.
        self._apply_jar_physical_randomization()
        self._apply_door_joint_physical_randomization()

        jar_contact_entities = self._jar_contact_entities()
        mug_handle_contact_entities = (self._mug_handle_link,)
        self._jar_tactile_target_groups = (("jar", jar_contact_entities),)
        self._mug_handle_tactile_target_groups = (
            ("mug_handle", mug_handle_contact_entities),
        )

        handle_names = (
            M11_LEFT_DOOR_HANDLE_LINK_NAME,
            M11_RIGHT_DOOR_HANDLE_LINK_NAME,
        )
        if all(name is not None for name in handle_names):
            missing = [
                name for name in handle_names if name not in self.cabinet.links_map
            ]
            if missing:
                raise RuntimeError(
                    f"Configured M11 cabinet handle links are missing: {missing}."
                )
            self._door_contact_links = tuple(
                self.cabinet.links_map[name] for name in handle_names
            )
        else:
            # Current source does not provide calibrated handle link geometry.
            # Keep the previous door-child-link proxy rather than guessing names.
            self._door_contact_links = (
                self.left_door_joint.child_link,
                self.right_door_joint.child_link,
            )

        self._door_tactile_target_groups = (
            ("doors", self._door_contact_links),
        )
        if not self.gpu_sim_enabled:
            for agent in self._manipulator_agents():
                agent.prepare_cpu_object_taxel_query(
                    self._jar_tactile_target_groups
                )
                agent.prepare_cpu_object_taxel_query(
                    self._mug_handle_tactile_target_groups
                )
                agent.prepare_cpu_object_taxel_query(
                    self._door_tactile_target_groups
                )

        # Stable grasp requires one hooked handle finger and two fingers on the
        # whole mug. Each anatomical finger aggregates proximal/middle/distal
        # contact, and the handle finger also counts toward the whole-mug total.
        self._jar_grasp_queries = []
        self._mug_handle_grasp_queries = []
        self._jar_force_queries = []
        self._cabinet_force_queries = []
        for agent in self._manipulator_agents():
            missing = [
                name
                for name in M11_GRASP_FINGER_LINK_NAMES
                if name not in agent.robot.links_map
            ]
            if missing:
                raise RuntimeError(
                    f"M11 missing semantic grasp links for {agent.uid!r}: {missing}."
                )
            finger_links = tuple(
                agent.robot.links_map[name]
                for name in M11_GRASP_FINGER_LINK_NAMES
            )
            self._set_collision_contact_offset(
                finger_links,
                M11_MUG_CONTACT_OFFSET_M,
            )
            self._jar_grasp_queries.append(
                agent.prepare_pairwise_contact_query(
                    finger_links, jar_contact_entities
                )
            )
            self._mug_handle_grasp_queries.append(
                agent.prepare_pairwise_contact_query(
                    finger_links, mug_handle_contact_entities
                )
            )
            self._jar_force_queries.append(
                agent.prepare_pairwise_contact_query(
                    tuple(agent.robot.links), jar_contact_entities
                )
            )
            self._cabinet_force_queries.append(
                agent.prepare_pairwise_contact_query(
                    tuple(agent.robot.links), tuple(self.cabinet.links)
                )
            )
        # [N, 3] tight mug-local compound-collision envelope points.
        self._jar_collision_link_corners = [
            (
                self.jar,
                torch.as_tensor(
                    self._jar_collision_corners_local,
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
        ]

        self._final_dwell_steps_required = max(
            1, math.ceil(self.SUCCESS_SPEC.final_dwell_s * self._control_freq)
        )

    def _manipulator_agents(self) -> tuple[Any, ...]:
        return (self.agent,)

    @staticmethod
    def _counter_top_fixture_local_corners(fixture: Any) -> np.ndarray:
        """Return fixture-local corners of the real counter-top visuals."""

        top_corner_sets: list[np.ndarray] = []
        actor_builder = getattr(fixture, "actor_builder", None)
        for record in getattr(actor_builder, "visual_records", ()):
            record_name = str(getattr(record, "name", ""))
            fixture_name = str(getattr(fixture, "name", ""))
            short_name = record_name.removeprefix(f"{fixture_name}_")
            if not short_name.startswith("top"):
                continue
            half_size = np.asarray(record.scale, dtype=np.float64)
            center = np.asarray(record.pose.p, dtype=np.float64)
            if half_size.shape != (3,) or center.shape != (3,):
                continue
            signs = np.asarray(
                ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
                dtype=np.float64,
            )
            corners = np.repeat(center[None, :], 4, axis=0)
            corners[:, :2] += signs * half_size[None, :2]
            top_corner_sets.append(corners)
        if top_corner_sets:
            return np.concatenate(top_corner_sets, axis=0)

        fixture_size = np.asarray(fixture.size, dtype=np.float64)
        if fixture_size.shape != (3,) or np.any(fixture_size <= 0.0):
            raise ValueError(
                "M11 counter fixture size must contain three positive values, "
                f"got {fixture_size}."
            )
        half_size = 0.5 * fixture_size
        signs = np.asarray(
            ((-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)),
            dtype=np.float64,
        )
        corners = np.zeros((4, 3), dtype=np.float64)
        corners[:, :2] = signs * half_size[None, :2]
        corners[:, 2] = half_size[2]
        return corners

    @classmethod
    def _counter_top_site_xy_bounds(
        cls,
        fixture: Any,
        *,
        fixture_rotation: np.ndarray,
        site_position: np.ndarray,
        site_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a real counter-top footprint into cabinet-site XY bounds."""

        fixture_corners = cls._counter_top_fixture_local_corners(fixture)
        # [N, 3] -> [N, 3] -> [N, 2].
        world_corners = (
            fixture_rotation @ fixture_corners.T
        ).T + np.asarray(fixture.pos, dtype=np.float64)
        site_corners = (
            site_rotation.T @ (world_corners - site_position).T
        ).T
        return site_corners[:, :2].min(axis=0), site_corners[:, :2].max(axis=0)

    def _require_counter_surface(self) -> Any:
        """Resolve the real RoboCasa counter directly below the cabinet site."""

        if Counter is None or Actor is None:
            raise ImportError(
                "M11 real-counter resolution requires ManiSkill RoboCasa structs."
            )
        external_builders = getattr(
            self.layout_scene,
            "external_scene_builders",
            None,
        )
        if not external_builders:
            raise RuntimeError(
                "M11 requires an external RoboCasa scene builder to resolve "
                "the cabinet counter surface."
            )

        probe_world = self.interaction_site_local_point_to_world(
            M11_COUNTER_SURFACE_PROBE_LOCAL_XYZ_M
        ).astype(np.float64)
        site_pose = self.interaction_site_frame_pose()
        site_position = np.asarray(site_pose.position_xyz, dtype=np.float64)
        site_rotation = site_pose.rotation_matrix.astype(np.float64)
        counter_actors: list[Any] = []
        counter_surface_local_z: list[float] = []
        counter_surface_min_xy: list[np.ndarray] = []
        counter_surface_max_xy: list[np.ndarray] = []

        for builder in external_builders.values():
            scene_data = getattr(builder, "scene_data", None) or ()
            for scene_index, scene_entry in enumerate(scene_data):
                fixtures = (
                    scene_entry.get("fixtures", {})
                    if isinstance(scene_entry, dict)
                    else {}
                )
                candidates: list[tuple[float, str, Any, np.ndarray]] = []
                for fixture_name, fixture in fixtures.items():
                    if not isinstance(fixture, Counter):
                        continue
                    fixture_position = np.asarray(
                        fixture.pos, dtype=np.float64
                    )
                    fixture_rotation = self._rpy_rotation_matrix(
                        np.asarray(fixture.euler, dtype=np.float64)
                    )
                    probe_in_fixture = fixture_rotation.T @ (
                        probe_world - fixture_position
                    )
                    fixture_half_xy = (
                        np.asarray(fixture.size[:2], dtype=np.float64) / 2.0
                    )
                    overflow_xy = np.maximum(
                        np.abs(probe_in_fixture[:2]) - fixture_half_xy,
                        0.0,
                    )
                    candidates.append(
                        (
                            float(np.linalg.norm(overflow_xy)),
                            str(fixture_name),
                            fixture,
                            fixture_rotation,
                        )
                    )
                if not candidates:
                    raise RuntimeError(
                        f"M11 found no RoboCasa counter in scene {scene_index}."
                    )
                candidates.sort(key=lambda item: (item[0], item[1]))
                best_distance, best_name, best_fixture, best_rotation = candidates[0]
                if (
                    best_distance
                    > M11_COUNTER_FIXTURE_MAX_PROBE_DISTANCE_M
                ):
                    raise RuntimeError(
                        "M11 cabinet-local counter probe is not on a real "
                        f"counter in scene {scene_index}: nearest={best_name!r}, "
                        f"distance={best_distance:.4f} m."
                    )
                if (
                    len(candidates) > 1
                    and math.isclose(
                        candidates[1][0],
                        best_distance,
                        abs_tol=1.0e-6,
                    )
                ):
                    raise RuntimeError(
                        "M11 counter resolution is ambiguous in scene "
                        f"{scene_index}: {best_name!r} and "
                        f"{candidates[1][1]!r}."
                    )
                counter_actor = getattr(best_fixture, "actor", None)
                if counter_actor is None:
                    raise RuntimeError(
                        f"M11 counter fixture {best_name!r} has no built actor."
                    )
                counter_actors.append(counter_actor)

                surface_min_xy, surface_max_xy = (
                    self._counter_top_site_xy_bounds(
                        best_fixture,
                        fixture_rotation=best_rotation,
                        site_position=site_position,
                        site_rotation=site_rotation,
                    )
                )
                counter_surface_min_xy.append(surface_min_xy)
                counter_surface_max_xy.append(surface_max_xy)

                # Fixture local +Z top center -> world -> cabinet-site local Z.
                counter_top_world = np.asarray(
                    best_fixture.pos, dtype=np.float64
                ) + best_rotation @ np.asarray(
                    (0.0, 0.0, float(best_fixture.size[2]) / 2.0),
                    dtype=np.float64,
                )
                counter_top_site = site_rotation.T @ (
                    counter_top_world - site_position
                )
                counter_surface_local_z.append(float(counter_top_site[2]))

        if len(counter_actors) != self.num_envs:
            raise RuntimeError(
                "M11 counter actor count does not match the environment count: "
                f"actors={len(counter_actors)}, num_envs={self.num_envs}."
            )
        self._counter_surface_local_z = torch.as_tensor(
            counter_surface_local_z,
            device=self.device,
            dtype=torch.float32,
        )
        # One batched M11 environment uses one layout. Intersect any small
        # per-scene fixture variation so a single visible marker exactly matches
        # the valid region in every parallel scene.
        shared_surface_min_xy = np.stack(counter_surface_min_xy).max(axis=0)
        shared_surface_max_xy = np.stack(counter_surface_max_xy).min(axis=0)
        region_min_xy = shared_surface_min_xy.copy()
        region_max_xy = shared_surface_max_xy.copy()
        region_min_xy[1] = max(
            region_min_xy[1], -M11_COUNTER_REGION_LATERAL_HALF_WIDTH_M
        )
        region_max_xy[1] = min(
            region_max_xy[1], M11_COUNTER_REGION_LATERAL_HALF_WIDTH_M
        )
        if np.any(region_min_xy >= region_max_xy):
            raise RuntimeError(
                "M11 resolved an empty shared counter region: "
                f"minimum={region_min_xy}, maximum={region_max_xy}."
            )
        self._counter_region_local_min_xy = torch.as_tensor(
            np.repeat(region_min_xy[None, :], self.num_envs, axis=0),
            device=self.device,
            dtype=torch.float32,
        )
        self._counter_region_local_max_xy = torch.as_tensor(
            np.repeat(region_max_xy[None, :], self.num_envs, axis=0),
            device=self.device,
            dtype=torch.float32,
        )
        if self.num_envs == 1:
            return counter_actors[0]
        return Actor.merge(counter_actors, name="m11_counter_surface")

    def _reset_agent_to_rest(
        self,
        agent: Any,
        robot_pose: Any,
        batch_size: int,
    ) -> None:
        keyframes = getattr(agent, "keyframes", None)
        rest_keyframe = keyframes.get("rest") if keyframes else None
        if rest_keyframe is None or getattr(rest_keyframe, "qpos", None) is None:
            agent.robot.set_pose(robot_pose)
            return

        rest_qpos = torch.as_tensor(
            rest_keyframe.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        if rest_qpos.ndim == 1:
            rest_qpos = rest_qpos.unsqueeze(0).repeat(batch_size, 1)
        agent.reset(rest_qpos)
        agent.robot.set_pose(robot_pose)

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_jar_initial_world_position"):
            return
        self._jar_initial_world_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._jar_grasped_after_door_open_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._jar_taken_out_after_door_open_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._jar_taken_out_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._jar_placed_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._jar_interacted_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._door_opened_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._final_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._cabinet_opening_plane_x_local = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._initial_cabinet_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._initial_cabinet_quaternion = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._initial_cabinet_qpos = torch.zeros_like(self.cabinet.get_qpos())
        self._cabinet_episode_reset_qpos = torch.zeros_like(
            self.cabinet.get_qpos()
        )

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            env_idx = torch.as_tensor(
                env_idx, device=self.device, dtype=torch.long
            ).reshape(-1)
            batch_size = len(env_idx)
            self._ensure_episode_state()

            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )

            # Capture the layout-authored/reset articulation state before applying
            # the M11 door override. This avoids querying GPU qpos buffers during
            # _load_scene while still restoring non-target cabinet joints reliably.
            self._cabinet_episode_reset_qpos[env_idx] = (
                self.cabinet.get_qpos()[env_idx]
            )

            hidden_robot_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
            for agent in self._manipulator_agents():
                self._reset_agent_to_rest(agent, hidden_robot_pose, batch_size)

            self._reset_cabinet_doors(env_idx)
            self._reset_jar_joints(env_idx)
            self._place_jar(env_idx)
            self._zero_jar_root_velocity(batch_size)
            self.layout_scene.settle(env_idx=env_idx)

            # Settle can move passive doors and the free-root jar. Commit the task
            # initial state a second time before recording geometry baselines.
            self._reset_cabinet_doors(env_idx)
            self._reset_jar_joints(env_idx)
            self._place_jar(env_idx)
            self._zero_jar_root_velocity(batch_size)
            self._jar_initial_world_position[env_idx] = self._jar_pose().p[env_idx]
            self._initial_cabinet_position[env_idx] = self.cabinet.root_pose.p[env_idx]
            self._initial_cabinet_quaternion[env_idx] = self.cabinet.root_pose.q[env_idx]
            self._initial_cabinet_qpos[env_idx] = self.cabinet.get_qpos()[env_idx]
            self._calibrate_opening_geometry(env_idx)

            self._reset_agent_to_rest(
                self.agent, self._robot_base_pose(env_idx), batch_size
            )

            self._door_opened_once[env_idx] = False
            self._jar_grasped_after_door_open_once[env_idx] = False
            self._jar_taken_out_after_door_open_once[env_idx] = False
            self._jar_taken_out_once[env_idx] = False
            self._jar_placed_once[env_idx] = False
            self._jar_interacted_once[env_idx] = False
            self._final_stable_steps[env_idx] = 0

    def _door_angles_rad(self) -> Any:
        left_angle = self.left_door_joint.qpos.reshape(self.num_envs, -1)[:, 0]
        right_angle = self.right_door_joint.qpos.reshape(self.num_envs, -1)[:, 0]
        return left_angle, right_angle

    def _door_opened(self) -> Any:
        left_angle, right_angle = self._door_angles_rad()
        threshold = self.SUCCESS_SPEC.door_open_angle_threshold_rad
        return (torch.abs(left_angle) > threshold) | (
            torch.abs(right_angle) > threshold
        )

    @staticmethod
    def _quaternion_rotate(quaternion_wxyz: Any, vectors: Any) -> Any:
        quaternion = quaternion_wxyz / torch.linalg.norm(
            quaternion_wxyz, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        q_vector = quaternion[:, None, 1:]
        q_scalar = quaternion[:, None, :1]
        twice_cross = 2.0 * torch.cross(
            q_vector.expand_as(vectors), vectors, dim=-1
        )
        return vectors + q_scalar * twice_cross + torch.cross(
            q_vector.expand_as(vectors), twice_cross, dim=-1
        )

    def _site_world_pose(self, batch_size: int | None = None) -> Any:
        if batch_size is None:
            batch_size = self.num_envs
        return Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

    def _world_points_to_site_local(self, world_points: Any) -> Any:
        site_pose = self._site_world_pose(self.num_envs)
        inverse_q = site_pose.q.clone()
        inverse_q[:, 1:] *= -1.0
        delta = world_points - site_pose.p[:, None, :]
        return self._quaternion_rotate(inverse_q, delta)

    def _jar_collision_corners_world(self) -> Any:
        world_corner_sets: list[Any] = []
        for link, local_corners in self._jar_collision_link_corners:
            corners = local_corners.unsqueeze(0).repeat(self.num_envs, 1, 1)
            rotated = self._quaternion_rotate(link.pose.q, corners)
            world_corner_sets.append(rotated + link.pose.p[:, None, :])
        return torch.cat(world_corner_sets, dim=1)

    def _jar_collision_corners_site_local(self) -> Any:
        return self._world_points_to_site_local(
            self._jar_collision_corners_world()
        )

    def _jar_root_site_local(self) -> Any:
        site_pose = self._site_world_pose(self.num_envs)
        inverse_q = site_pose.q.clone()
        inverse_q[:, 1:] *= -1.0
        delta = self._jar_pose().p - site_pose.p
        return self._quaternion_rotate(
            inverse_q, delta[:, None, :]
        )[:, 0, :]

    def _calibrate_opening_geometry(self, env_idx: Any) -> None:
        """Estimate the closed cabinet front plane in cabinet-site coordinates."""

        env_idx = torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long
        ).reshape(-1)
        site_pose = self._site_world_pose(self.num_envs)
        inverse_q = site_pose.q.clone()
        inverse_q[:, 1:] *= -1.0
        door_world = torch.stack(
            [
                self.left_door_joint.child_link.pose.p,
                self.right_door_joint.child_link.pose.p,
            ],
            dim=1,
        )
        door_local = self._quaternion_rotate(
            inverse_q, door_world - site_pose.p[:, None, :]
        )
        candidate_plane_x = door_local[:, :, M11_OUTWARD_LOCAL_AXIS].mean(dim=1)

        jar_corners = self._jar_collision_corners_site_local()
        initial_outward_face_x = jar_corners[:, :, M11_OUTWARD_LOCAL_AXIS].amin(dim=1)
        fallback_plane_x = (
            initial_outward_face_x - M11_OPENING_PLANE_FALLBACK_GAP_M
        )
        # In this benchmark the cabinet interior is +X and extraction is -X.
        # A valid closed-door origin should therefore lie outside the complete
        # initial jar bounds. Reject implausible link origins instead of trusting
        # an asset-specific rigid-body origin that may sit at the hinge center.
        candidate_is_usable = (
            candidate_plane_x < initial_outward_face_x - 0.005
        ) & (
            candidate_plane_x > initial_outward_face_x - 0.50
        )
        opening_plane_x = torch.where(
            candidate_is_usable, candidate_plane_x, fallback_plane_x
        )
        self._cabinet_opening_plane_x_local[env_idx] = opening_plane_x[env_idx]

    def _jar_has_crossed_opening_plane(self) -> Any:
        corners = self._jar_collision_corners_site_local()
        # Cabinet-site local -X points outward. Crossing is a historical
        # extraction milestone, so one collision-envelope point past the plane
        # is sufficient and no extra clearance is imposed.
        outwardmost_corner_x = corners[:, :, M11_OUTWARD_LOCAL_AXIS].amin(dim=1)
        return outwardmost_corner_x < self._cabinet_opening_plane_x_local

    def _jar_fully_reentered_cabinet(self) -> Any:
        corners = self._jar_collision_corners_site_local()
        # Re-entry is deliberately the inverse *complete* transition: every
        # envelope point must be back inside. This prevents the newly relaxed
        # one-point crossing milestone from immediately triggering D2.
        outwardmost_corner_x = corners[:, :, M11_OUTWARD_LOCAL_AXIS].amin(dim=1)
        return outwardmost_corner_x > (
            self._cabinet_opening_plane_x_local + M11_REENTRY_CLEARANCE_M
        )

    def _object_tactile_force_norms_by_agent(
        self,
        *,
        group_name: str,
        target_groups: tuple[tuple[str, tuple[Any, ...]], ...],
    ) -> Any:
        per_agent: list[Any] = []
        for agent in self._manipulator_agents():
            taxel_world_forces = agent.get_object_taxel_world_forces(
                target_groups
            )[group_name]
            parent_world_forces = agent.aggregate_taxel_world_forces_by_parent(
                taxel_world_forces
            )
            per_agent.append(
                torch.linalg.norm(parent_world_forces, dim=-1).reshape(
                    self.num_envs, -1
                )
            )
        if not per_agent:
            return torch.zeros(
                (self.num_envs, 0, 0),
                dtype=torch.float32,
                device=self.device,
            )
        return torch.stack(per_agent, dim=1)

    def _jar_contact_by_agent(self) -> Any:
        force_norms = self._object_tactile_force_norms_by_agent(
            group_name="jar", target_groups=self._jar_tactile_target_groups
        )
        if force_norms.shape[2] == 0:
            return torch.zeros(
                (self.num_envs, len(self._manipulator_agents())),
                dtype=torch.bool,
                device=self.device,
            )
        return (
            force_norms.amax(dim=2) >= self.SUCCESS_SPEC.grasp_contact_force_n
        )

    def _door_contact_by_agent(self) -> Any:
        force_norms = self._object_tactile_force_norms_by_agent(
            group_name="doors", target_groups=self._door_tactile_target_groups
        )
        if force_norms.shape[2] == 0:
            return torch.zeros(
                (self.num_envs, len(self._manipulator_agents())),
                dtype=torch.bool,
                device=self.device,
            )
        return (
            force_norms.amax(dim=2) >= self.SUCCESS_SPEC.grasp_contact_force_n
        )

    @staticmethod
    def _finger_group_force_magnitudes(pair_forces: Any) -> Any:
        """Aggregate all phalanx/object pairs into one force per finger."""

        expected_link_count = len(M11_GRASP_FINGER_LINK_NAMES)
        if (
            pair_forces.ndim != 4
            or pair_forces.shape[1] != expected_link_count
            or pair_forces.shape[-1] != 3
        ):
            raise ValueError(
                "M11 mug pair forces must have shape "
                f"[B, {expected_link_count}, T, 3], got "
                f"{tuple(pair_forces.shape)}."
            )

        finger_force_magnitudes: list[Any] = []
        link_start = 0
        for finger_link_names in M11_GRASP_FINGER_LINK_GROUPS:
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

    def _jar_grasp_by_agent(self) -> Any:
        grasps: list[Any] = []
        for agent, mug_query, handle_query in zip(
            self._manipulator_agents(),
            self._jar_grasp_queries,
            self._mug_handle_grasp_queries,
            strict=True,
        ):
            mug_pair_forces = agent.get_pairwise_contact_forces(mug_query)
            handle_pair_forces = agent.get_pairwise_contact_forces(handle_query)
            # [B, 15, L, 3] -> [B, 5]: body and handle links both count, and
            # proximal/middle/distal contacts belong to the same finger.
            per_finger_mug_force = self._finger_group_force_magnitudes(
                mug_pair_forces
            )
            # [B, 15, 1, 3] -> [B, 5]: whole-finger contact with the handle.
            per_finger_handle_force = self._finger_group_force_magnitudes(
                handle_pair_forces
            )
            mug_contact = (
                per_finger_mug_force
                >= self.SUCCESS_SPEC.grasp_contact_force_n
            )
            handle_contact = (
                per_finger_handle_force
                >= self.SUCCESS_SPEC.grasp_contact_force_n
            )
            grasps.append(
                (
                    handle_contact.sum(dim=1)
                    >= self.SUCCESS_SPEC.grasp_min_handle_fingers
                )
                & (
                    mug_contact.sum(dim=1)
                    >= self.SUCCESS_SPEC.grasp_min_mug_fingers
                )
            )
        return torch.stack(grasps, dim=1)

    def _jar_grasped(self) -> Any:
        return self._jar_grasp_by_agent().any(dim=1)

    def _jar_released(self) -> Any:
        return (
            self._jar_hand_contact_force()
            <= self.SUCCESS_SPEC.residual_hand_contact_force_n
        )

    def _jar_hand_contact_force(self) -> Any:
        force_norms = self._object_tactile_force_norms_by_agent(
            group_name="jar", target_groups=self._jar_tactile_target_groups
        )
        if force_norms.shape[2] == 0:
            return torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        return force_norms.amax(dim=(1, 2))

    def _jar_displacement(self) -> Any:
        return torch.linalg.norm(
            self._jar_pose().p - self._jar_initial_world_position, dim=1
        )

    def _jar_linear_velocity(self) -> Any:
        for name in (
            "get_root_linear_velocity",
            "root_linear_velocity",
            "linear_velocity",
        ):
            value = getattr(self.jar, name, None)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor):
                return value.reshape(self.num_envs, -1)
        raise TypeError("M11 mug entity exposes no linear velocity.")

    def _jar_angular_velocity(self) -> Any:
        for name in (
            "get_root_angular_velocity",
            "root_angular_velocity",
            "angular_velocity",
        ):
            value = getattr(self.jar, name, None)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor):
                return value.reshape(self.num_envs, -1)
        raise TypeError("M11 mug entity exposes no angular velocity.")

    def _jar_linear_speed(self) -> Any:
        return torch.linalg.norm(self._jar_linear_velocity(), dim=1)

    def _jar_angular_speed(self) -> Any:
        return torch.linalg.norm(self._jar_angular_velocity(), dim=1)

    def _jar_kinematically_stable(self) -> Any:
        return (
            self._jar_linear_speed()
            <= self.SUCCESS_SPEC.jar_stable_linear_speed_mps
        ) & (
            self._jar_angular_speed()
            <= self.SUCCESS_SPEC.jar_stable_angular_speed_rad_s
        )

    def _jar_upright(self) -> Any:
        jar_quaternion = self._jar_pose().q
        local_up = torch.zeros(
            (self.num_envs, 1, 3),
            dtype=jar_quaternion.dtype,
            device=jar_quaternion.device,
        )
        local_up[:, :, 2] = 1.0
        # [B, 1, 3] -> [B, 3]: rotate mug-local +Z into world coordinates.
        world_up = self._quaternion_rotate(jar_quaternion, local_up)[:, 0, :]
        tilt_rad = torch.acos(world_up[:, 2].clamp(-1.0, 1.0))
        return tilt_rad <= self.SUCCESS_SPEC.jar_max_tilt_rad

    def _jar_height_valid(self) -> Any:
        local_z = self._jar_root_site_local()[:, 2]
        return (local_z >= -0.05) & (local_z <= 0.60)

    def _jar_fully_on_nearby_counter(self) -> Any:
        override = getattr(self, "_state_injection_measurements", {}).get(
            "jar_on_nearby_counter"
        )
        if override is not None:
            return self._as_per_env_tensor(override, dtype=torch.bool)
        local_corners = self._jar_collision_corners_site_local()
        minimum_xy = self._counter_region_local_min_xy[:, None, :]
        maximum_xy = self._counter_region_local_max_xy[:, None, :]
        # [B, N, 2] within two [B, 1, 2] bounds -> [B].
        inside_xy = (
            (local_corners[:, :, :2] >= minimum_xy)
            & (local_corners[:, :, :2] <= maximum_xy)
        ).all(dim=(1, 2))
        jar_bottom_z = local_corners[:, :, 2].amin(dim=1)
        on_surface_height = (
            torch.abs(jar_bottom_z - self._counter_surface_local_z)
            <= M11_COUNTER_SURFACE_HEIGHT_TOLERANCE_M
        )
        return inside_xy & on_surface_height

    def _doors_are_valid_initial_state(self) -> Any:
        left, right = self._door_angles_rad()
        left_speed = self.left_door_joint.qvel.reshape(self.num_envs, -1)[:, 0]
        right_speed = self.right_door_joint.qvel.reshape(self.num_envs, -1)[:, 0]
        return (torch.abs(left) <= 0.02) & (torch.abs(right) <= 0.02) & (
            torch.abs(left_speed) <= 0.02
        ) & (torch.abs(right_speed) <= 0.02)

    def _check_generalization_validity(
        self, axis: GeneralizationAxis | None
    ) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            finite_pose = torch.isfinite(self._jar_pose().raw_pose).all(dim=1)
            initial_inside = ~self._jar_has_crossed_opening_plane()
            return (
                finite_pose
                & initial_inside
                & self._doors_are_valid_initial_state()
                & self._jar_kinematically_stable()
                & self._jar_upright()
                & self._jar_height_valid()
            )
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()

        door_opened = self._door_opened()
        jar_contact_by_agent = self._jar_contact_by_agent()
        jar_grasp_by_agent = self._jar_grasp_by_agent()
        jar_grasped = jar_grasp_by_agent.any(dim=1)
        jar_crossed_opening_plane = self._jar_has_crossed_opening_plane()
        jar_kinematically_stable = self._jar_kinematically_stable()
        jar_upright = self._jar_upright()
        jar_released = self._jar_released()
        jar_on_nearby_counter = self._jar_fully_on_nearby_counter()

        self._door_opened_once |= door_opened
        self._jar_interacted_once |= jar_contact_by_agent.any(dim=1)
        self._jar_grasped_after_door_open_once |= (
            jar_grasped & self._door_opened_once
        )
        valid_take_out_now = (
            jar_crossed_opening_plane & self._door_opened_once
        )
        self._jar_taken_out_after_door_open_once |= valid_take_out_now
        self._jar_taken_out_once |= valid_take_out_now

        final_candidate = (
            jar_on_nearby_counter
            & jar_kinematically_stable
            & jar_upright
            & jar_released
            & self._door_opened_once
            & self._jar_grasped_after_door_open_once
            & self._jar_taken_out_after_door_open_once
        )
        self._final_stable_steps = torch.where(
            final_candidate,
            self._final_stable_steps + 1,
            torch.zeros_like(self._final_stable_steps),
        )
        success = (
            self._final_stable_steps >= self._final_dwell_steps_required
        )
        self._jar_placed_once |= success

        return {
            "success": success,
            "open_cabinet_door": self._door_opened_once,
            "grasp_jar": self._jar_grasped_after_door_open_once,
            "take_jar_out": self._jar_taken_out_after_door_open_once,
            "place_jar_on_counter": success,
            "jar_crossed_opening_plane": jar_crossed_opening_plane,
            "jar_on_nearby_counter": jar_on_nearby_counter,
            "jar_upright": jar_upright,
            "jar_released": jar_released,
            "jar_grasped_by_right": jar_grasp_by_agent[:, 0],
        }






    def _tcp_within_cabinet_workspace(self, agent: Any) -> Any:
        site_pose = self._site_world_pose(self.num_envs)
        inverse_q = site_pose.q.clone()
        inverse_q[:, 1:] *= -1.0
        tcp_delta = agent.tcp.pose.p - site_pose.p
        tcp_local = self._quaternion_rotate(
            inverse_q, tcp_delta[:, None, :]
        )[:, 0, :]
        lower = torch.as_tensor(
            M11_TCP_WORKSPACE_LOCAL_MIN_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        upper = torch.as_tensor(
            M11_TCP_WORKSPACE_LOCAL_MAX_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        return ~((tcp_local < lower) | (tcp_local > upper)).any(dim=1)
