from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.assets import (
    partnet_mobility_modified_model_urdf_path,
    ycb_model_obj_path,
)
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
    LayoutRegistry,
    PhysicalMaterialSpec,
    PoseSpec,
    StaticBoxSpec,
    StaticMeshSpec,
)
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs import Articulation
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    torch = None
    Articulation = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


M12_SETTINGS = TaskRegistry.register(TaskRegistry.load("m12_v1"))


@dataclass(frozen=True)
class M12SuccessSpec:
    """Thresholds for the four M12 completion stages."""

    door_access_open_fraction: float = 0.45
    door_closed_enter_fraction: float = 0.05
    door_closed_exit_fraction: float = 0.08
    door_closed_max_speed_rad_s: float = 0.08

    grasp_contact_force_n: float = 0.30
    grasp_min_fingers: int = 2
    grasp_dwell_s: float = 0.12

    support_contact_force_n: float = 0.20
    bowl_stable_linear_speed_mps: float = 0.05
    bowl_stable_angular_speed_rad_s: float = 0.60
    bowl_max_tilt_rad: float = math.radians(35.0)
    residual_hand_contact_force_n: float = 0.10
    placement_dwell_s: float = 0.35
    final_dwell_s: float = 0.30


M12_SUCCESS_SPEC = M12SuccessSpec()

M12_INTERACTION_SITE_ID = "microwave"
M12_BOWL_ENTITY_NAME = "bowl_microwave"
M12_BOWL_RISER_ENTITY_NAME = "m12_bowl_riser"
M12_BOWL_MODEL_NAME = "024_bowl"
M12_L3_MICROWAVE_MODEL_ID = "7167"
M12_L3_MICROWAVE_COLLISION_OVERRIDE_PATH = (
    partnet_mobility_modified_model_urdf_path(M12_L3_MICROWAVE_MODEL_ID)
)
M12_LAYOUT_IDS = (
    LayoutId.KITCHEN_L1,
    LayoutId.KITCHEN_L2,
    LayoutId.KITCHEN_L3,
)
M12_HUMAN_VIEWER_CAMERA_WORLD_POSES: dict[
    LayoutId, HumanViewerCameraPoseSpec
] = {
    LayoutId.KITCHEN_L1: HumanViewerCameraPoseSpec(
        position_world_xyz=(1.248409, -2.404044, 1.843952),
        quaternion_world_wxyz=(0.357797, 0.209721, 0.082808, -0.906169),
    ),
    # Calibration hook: replace these L2 values with the accepted viewer pose.
    LayoutId.KITCHEN_L2: HumanViewerCameraPoseSpec(
        position_world_xyz=(1.396496, -3.259096, 1.705486),
        quaternion_world_wxyz=(0.369766, 0.174898, 0.071088, -0.909742),
    ),
    # Calibration hook: replace these L3 values with the accepted viewer pose.
    LayoutId.KITCHEN_L3: HumanViewerCameraPoseSpec(
        position_world_xyz=(1.863352, -1.526196, 1.664283),
        quaternion_world_wxyz=(0.440309, -0.201136, 0.101904, 0.869073),
    ),
}

# Task-private YCB bowl spawn positions. These are deliberately kept outside
# the shared Kitchen layout because M12 uses a different bowl placement in each
# layout. L1/L2 expose 3 cm of the rim beyond the riser on the robot side; L3
# does the same beyond the counter edge. The z values leave a 5 mm settling gap
# above the support based on the collision mesh's 28 mm vertical half extent.
M12_BOWL_INITIAL_POSITION_XYZ: dict[LayoutId, tuple[float, float, float]] = {
    LayoutId.KITCHEN_L1: (0.588, -2.50, 1.213),
    LayoutId.KITCHEN_L2: (0.588, -3.50, 1.213),
    LayoutId.KITCHEN_L3: (1.45, -0.598, 0.953),
}
M12_BOWL_INITIAL_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
M12_L3_COUNTER_ROBOT_SIDE_EDGE_Y_M = -0.65
M12_L3_BOWL_ROBOT_SIDE_OVERHANG_M = 0.03

# L1/L2 share a RoboCasa counter with a top surface at world z=0.92 m.
# A task-private solid riser moves the bowl toward the robot-facing counter
# edge and raises it to a comfortable grasp height. A single collision box is
# deliberately used instead of a multi-part shelf to avoid seams that can snag
# Shadow fingertips or destabilize the bowl during reset settling.
M12_BOWL_RISER_HALF_SIZE_XYZ_M = (0.16, 0.19, 0.13)
M12_BOWL_RISER_POSITION_XYZ: dict[LayoutId, tuple[float, float, float]] = {
    LayoutId.KITCHEN_L1: (0.48, -2.50, 1.05),
    LayoutId.KITCHEN_L2: (0.48, -3.50, 1.05),
}
M12_BOWL_RISER_ROBOT_SIDE_OVERHANG_M = 0.03
M12_BOWL_RISER_COUNTER_BOUNDS_XY: dict[
    LayoutId, tuple[tuple[float, float], tuple[float, float]]
] = {
    LayoutId.KITCHEN_L1: ((0.00, -2.70), (0.65, -0.70)),
    LayoutId.KITCHEN_L2: ((0.00, -3.70), (0.65, -0.70)),
}
M12_BOWL_RISER_COLOR_RGBA = (0.58, 0.46, 0.32, 1.0)
M12_SETTLE_SIM_STEPS = 30
M12_BOWL_RISER_MATERIAL = PhysicalMaterialSpec(
    static_friction=0.70,
    dynamic_friction=0.55,
    restitution=0.0,
)

# Shift the robot base 0.25 m to its right in the interaction-site frame.
# The task/site frame follows the conventional +X-forward, +Y-left, +Z-up
# convention, so robot-right is the local -Y direction. Spatial
# generalization offsets are applied on top of this fixed task-level shift.
M12_ROBOT_BASE_RIGHT_OFFSET_M = 0.25
M12_KITCHEN_L3_EXTRA_CLEARANCE_M = 0.30

# Conservative cylinder dimensions for the YCB 024 bowl collision mesh. The
# measured bounds are approximately 0.1623 x 0.1623 x 0.0553 m. A cylinder is
# used for containment because the bowl is rotationally symmetric: yaw must not
# create fictitious box corners, while roll and pitch still enlarge the exact
# projected extent of the finite cylinder.
M12_BOWL_COLLISION_HALF_EXTENT_XYZ_M = (0.082, 0.082, 0.028)
M12_BOWL_COLLISION_RADIUS_M = M12_BOWL_COLLISION_HALF_EXTENT_XYZ_M[0]
M12_BOWL_COLLISION_HALF_HEIGHT_M = M12_BOWL_COLLISION_HALF_EXTENT_XYZ_M[2]

# Actual visible inner surfaces of PartNet 7167 after applying its Kitchen L3
# scale and fixed-link transform.
M12_L3_PHYSICAL_CAVITY_LOCAL_BOUNDS_M = (
    (-0.2189345, -0.1522860, -0.1715340),
    (0.2464035, 0.3299735, 0.1668935),
)

# The microwave interaction sites are authored at the appliance root. For
# insertion geometry we construct a task frame whose +X points from the robot
# into the microwave, +Y is horizontal lateral, and +Z is world-up. The bounds
# below are the measured inner surfaces of the actual L1/L2 RoboCasa collision
# boxes and the L3 PartNet visual cavity, respectively. They are intentionally
# asymmetric because the microwave control panel occupies one lateral side.
M12_CAVITY_LOCAL_BOUNDS_M: dict[
    LayoutId, tuple[tuple[float, float, float], tuple[float, float, float]]
] = {
    LayoutId.KITCHEN_L1: (
        (-0.1913, -0.1446, -0.1493),
        (0.1890, 0.3363, 0.1886),
    ),
    LayoutId.KITCHEN_L2: (
        (-0.1913, -0.1446, -0.1493),
        (0.1890, 0.3363, 0.1886),
    ),
    LayoutId.KITCHEN_L3: M12_L3_PHYSICAL_CAVITY_LOCAL_BOUNDS_M,
}
# Contact with the floor and walls is physically valid. Expand the measured
# surfaces by a small tolerance instead of shrinking the cavity, which would
# make the required support contact contradict the containment predicate.
M12_CAVITY_CONTACT_TOLERANCE_M = 0.003

# Physical (already scaled) bounds in [inward, lateral, up] metres.
# Build these primitives directly, so an alternate/cached asset cannot restore
# the convex hull across the opening. No front wall is present.
M12_L3_SHELL_BOX_BOUNDS: tuple[
    tuple[tuple[float, float, float], tuple[float, float, float]], ...
] = (
    ((-.2578535, -.3976455, -.222298), (-.2189345, -.228432, .2176575)),
    ((.2464035, -.3976455, -.222298), (.2836305, .3807375, .2176575)),
    ((-.2189345, -.3976455, -.222298), (.2836305, -.152286, .2176575)),
    ((-.2189345, .3299735, -.222298), (.2836305, .3807375, .2176575)),
    ((-.2189345, -.3976455, -.222298), (.2836305, .3807375, -.171534)),
    ((-.2189345, -.152286, .1668935), (.2836305, .3299735, .2176575)),
)


class M12OpenCavityLayoutBuilder(LayoutBuilder):
    """Construct the L3 shell before PhysX cooks any asset collision meshes."""

    def _parse_articulation_builder(
        self, *, loader: Any, asset_path: Path, item_name: str,
    ) -> Any:
        builder = super()._parse_articulation_builder(
            loader=loader, asset_path=asset_path, item_name=item_name,
        )
        if item_name != M12_INTERACTION_SITE_ID:
            return builder
        if not math.isclose(float(loader.scale), 0.5):
            raise ValueError("M12 L3 cavity geometry requires microwave scale=0.5.")
        body_builders = [link for link in builder.link_builders if link.name == "link_3"]
        if len(body_builders) != 1:
            raise ValueError("M12 L3 requires exactly one microwave body link_3.")
        body = body_builders[0]
        # Clear ALL imported body shapes, including unknown extra meshes. The
        # fixed appliance does not need collision shapes for decorative feet.
        body.collision_records.clear()
        for lower_values, upper_values in M12_L3_SHELL_BOX_BOUNDS:
            lower = np.asarray(lower_values, dtype=np.float64)
            upper = np.asarray(upper_values, dtype=np.float64)
            center = (lower + upper) * 0.5
            half_size = (upper - lower) * 0.5
            # [3] -> [3]: cavity to scaled link_3 frame is [-y, z, -x].
            # URDF parsing already applied scale; do not multiply by it again.
            body.add_box_collision(
                pose=sapien.Pose(p=(-center[1], center[2], -center[0])),
                half_size=half_size[[1, 2, 0]],
            )
        return builder

M12_GRASP_FINGER_LINK_NAMES = (
    "rh_thdistal",
    "rh_ffdistal",
    "rh_mfdistal",
    "rh_rfdistal",
    "rh_lfdistal",
)
M12_CRITICAL_ROBOT_LINK_NAMES = (
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "rh_forearm",
    "rh_palm",
    *M12_GRASP_FINGER_LINK_NAMES,
)

M12_SPATIAL_PARAMETER_NAMES = (
    "bowl_x_offset_m",
    "bowl_y_offset_m",
    "bowl_yaw_offset_rad",
    "door_initial_open_fraction",
    "microwave_forward_offset_m",
    "microwave_side_offset_m",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
M12_PHYSICAL_PARAMETER_NAMES = (
    "bowl_mass_log_scale",
    "bowl_friction_log_scale",
    "support_friction_log_scale",
    "door_joint_friction",
    "door_joint_damping",
)
M12_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            # The bowl radius is only 82 mm and its nominal rim overhang is
            # 30 mm. Keep its center at least 22 mm behind the support edge at
            # either randomization extreme so reset does not create a tipping
            # configuration in L1/L2 (+X edge) or L3 (-Y edge).
            GeneralizationParameterSpec("bowl_x_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec("bowl_y_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec(
                "bowl_yaw_offset_rad",
                -math.radians(15.0),
                math.radians(15.0),
            ),
            GeneralizationParameterSpec("door_initial_open_fraction", 0.0, 0.10),
            GeneralizationParameterSpec("microwave_forward_offset_m", -0.01, 0.01),
            GeneralizationParameterSpec("microwave_side_offset_m", -0.01, 0.01),
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
                "bowl_mass_log_scale", math.log(0.75), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "bowl_friction_log_scale", math.log(0.75), math.log(1.30)
            ),
            GeneralizationParameterSpec(
                "support_friction_log_scale", math.log(0.80), math.log(1.25)
            ),
            GeneralizationParameterSpec("door_joint_friction", 0.02, 0.16),
            GeneralizationParameterSpec("door_joint_damping", 0.03, 0.20),
        ),
        requires_rebuild=True,
    ),
)



def _m12_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in M12_LAYOUT_IDS:
        world_pose = M12_HUMAN_VIEWER_CAMERA_WORLD_POSES[layout_id]
        camera_quaternion = np.asarray(
            world_pose.quaternion_world_wxyz,
            dtype=np.float64,
        )
        camera_quaternion /= np.linalg.norm(camera_quaternion)
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=world_pose.position_world_xyz,
            quaternion_world_wxyz=tuple(float(value) for value in camera_quaternion),
        )
    return camera_poses


@register_env(M12_SETTINGS.env_id, max_episode_steps=M12_SETTINGS.max_steps)
class M12Env(DexVLABenchBaseEnv):
    """Open a microwave, place a YCB bowl inside, and close the door.

    The microwave is never identified by a hard-coded fixture name here. M12
    reads the selected layout's ``microwave`` interaction-site metadata first:
    L1/L2 resolve the named RoboCasa external fixture, while L3 resolves the
    layout-owned PartNet articulation. The rest of the task works on the common
    articulation interface.
    """

    TASK_SETTINGS = M12_SETTINGS
    SUCCESS_SPEC = M12_SUCCESS_SPEC
    GENERALIZATION_SPEC = M12_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = M12_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _m12_human_viewer_camera_poses()

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ---------------------------------------------------------------------
    # Layout / scene binding
    # ---------------------------------------------------------------------

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        nominal_pose = self.interaction_site_robot_base_sapien_pose()

        # Keep the shared Kitchen L3 microwave site unchanged. M12 alone needs
        # the 0.30 m additional clearance authored by its source branch.
        extra_clearance_m = (
            M12_KITCHEN_L3_EXTRA_CLEARANCE_M
            if self.current_layout_id == LayoutId.KITCHEN_L3
            else 0.0
        )
        # In the interaction-site frame +X points toward the fixture and +Y
        # points left, hence extra clearance and robot-right are local -X/-Y.
        task_base_offset_pose = Pose.create_from_pq(
            p=np.asarray(
                [
                    [
                        -extra_clearance_m,
                        -M12_ROBOT_BASE_RIGHT_OFFSET_M,
                        0.0,
                    ]
                ],
                dtype=np.float32,
            ),
            q=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        )
        shifted_nominal_pose = Pose.create(nominal_pose) * task_base_offset_pose

        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return shifted_nominal_pose

        values = self._generalization_values(
            M12_SPATIAL_PARAMETER_NAMES,
            cache_name="_m12_spatial_values",
        )
        if env_idx is None:
            value_indices = np.arange(len(values["robot_forward_offset_m"]))
        elif isinstance(env_idx, torch.Tensor):
            value_indices = env_idx.detach().cpu().numpy()
        else:
            value_indices = np.asarray(env_idx, dtype=np.int64).reshape(-1)

        forward = values["robot_forward_offset_m"][value_indices]
        side = values["robot_side_offset_m"][value_indices]
        yaw = values["robot_yaw_offset_rad"][value_indices]
        local_position = np.stack(
            [forward, side, np.zeros_like(forward)],
            axis=1,
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
        return shifted_nominal_pose * Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
        )

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _bowl_spec(self) -> StaticMeshSpec:
        layout_id = self.current_layout_id
        if layout_id not in M12_BOWL_INITIAL_POSITION_XYZ:
            raise RuntimeError(
                f"M12 has no bowl spawn for layout {self.current_layout_id_value!r}."
            )
        return StaticMeshSpec(
            name=M12_BOWL_ENTITY_NAME,
            mesh_path=str(ycb_model_obj_path(M12_BOWL_MODEL_NAME)),
            collision_mesh_path=str(
                ycb_model_obj_path(M12_BOWL_MODEL_NAME, "collision.ply")
            ),
            pose=PoseSpec(
                position_xyz=M12_BOWL_INITIAL_POSITION_XYZ[layout_id],
                quaternion_wxyz=M12_BOWL_INITIAL_QUATERNION_WXYZ,
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        )

    def _bowl_riser_spec(self) -> StaticBoxSpec | None:
        layout_id = self.current_layout_id
        riser_position = M12_BOWL_RISER_POSITION_XYZ.get(layout_id)
        if riser_position is None:
            return None

        counter_bounds = M12_BOWL_RISER_COUNTER_BOUNDS_XY.get(layout_id)
        if counter_bounds is None:
            raise RuntimeError(
                f"M12 has no counter bounds for riser layout {layout_id.value!r}."
            )
        counter_lower_xy = np.asarray(counter_bounds[0], dtype=np.float64)
        counter_upper_xy = np.asarray(counter_bounds[1], dtype=np.float64)
        riser_center_xy = np.asarray(riser_position[:2], dtype=np.float64)
        riser_half_size_xy = np.asarray(
            M12_BOWL_RISER_HALF_SIZE_XYZ_M[:2],
            dtype=np.float64,
        )
        if not (
            np.all(riser_center_xy - riser_half_size_xy >= counter_lower_xy)
            and np.all(riser_center_xy + riser_half_size_xy <= counter_upper_xy)
        ):
            raise RuntimeError(
                f"M12 riser footprint for {layout_id.value!r} exceeds the "
                f"calibrated counter bounds {counter_bounds}."
            )

        bowl_position = np.asarray(
            M12_BOWL_INITIAL_POSITION_XYZ[layout_id],
            dtype=np.float64,
        )
        bowl_half_extent = np.asarray(
            M12_BOWL_COLLISION_HALF_EXTENT_XYZ_M,
            dtype=np.float64,
        )
        # The robot stands on the world +X side in L1/L2. Keep the bowl fully
        # supported laterally and on its far edge, while deliberately exposing
        # the near rim over the riser's +X edge for easier fingertip access.
        bowl_fits_laterally = (
            abs(bowl_position[1] - riser_center_xy[1]) + bowl_half_extent[1]
            <= riser_half_size_xy[1]
        )
        bowl_far_edge_supported = (
            bowl_position[0] - bowl_half_extent[0]
            >= riser_center_xy[0] - riser_half_size_xy[0]
        )
        robot_side_overhang_m = (
            bowl_position[0]
            + bowl_half_extent[0]
            - riser_center_xy[0]
            - riser_half_size_xy[0]
        )
        if not bowl_fits_laterally or not bowl_far_edge_supported:
            raise RuntimeError(
                f"M12 bowl for {layout_id.value!r} is not safely supported by "
                "the task-private riser."
            )
        if not math.isclose(
            robot_side_overhang_m,
            M12_BOWL_RISER_ROBOT_SIDE_OVERHANG_M,
            abs_tol=1.0e-6,
        ):
            raise RuntimeError(
                f"M12 bowl robot-side overhang for {layout_id.value!r} must be "
                f"{M12_BOWL_RISER_ROBOT_SIDE_OVERHANG_M:.3f} m, got "
                f"{robot_side_overhang_m:.3f} m."
            )
        riser_top_z = riser_position[2] + M12_BOWL_RISER_HALF_SIZE_XYZ_M[2]
        bowl_bottom_z = bowl_position[2] - bowl_half_extent[2]
        bowl_spawn_gap_m = bowl_bottom_z - riser_top_z
        if not 0.0 <= bowl_spawn_gap_m <= 0.02:
            raise RuntimeError(
                f"M12 bowl spawn gap above the {layout_id.value!r} riser must "
                f"be in [0.00, 0.02] m, got {bowl_spawn_gap_m:.4f} m."
            )

        return StaticBoxSpec(
            name=M12_BOWL_RISER_ENTITY_NAME,
            half_size_xyz=M12_BOWL_RISER_HALF_SIZE_XYZ_M,
            pose=PoseSpec(position_xyz=riser_position),
            color_rgba=M12_BOWL_RISER_COLOR_RGBA,
            body_type="kinematic",
            physical_material=M12_BOWL_RISER_MATERIAL,
        )

    def _with_l3_microwave_collision_override(
        self,
        build_items: tuple[Any, ...] | list[Any],
    ) -> tuple[Any, ...]:
        """Use the open-shell microwave collision model only for M12 L3."""

        if self.current_layout_id is not LayoutId.KITCHEN_L3:
            return tuple(build_items)

        overridden_items: list[Any] = []
        microwave_count = 0
        for item in build_items:
            if getattr(item, "name", None) != M12_INTERACTION_SITE_ID:
                overridden_items.append(item)
                continue
            if not isinstance(item, ArticulatedAssetSpec):
                raise TypeError(
                    "M12 Kitchen L3 microwave must be an ArticulatedAssetSpec, "
                    f"got {type(item).__name__}."
                )
            microwave_count += 1
            overridden_items.append(
                replace(
                    item,
                    asset_path=str(M12_L3_MICROWAVE_COLLISION_OVERRIDE_PATH),
                )
            )

        if microwave_count != 1:
            raise ValueError(
                "M12 Kitchen L3 requires exactly one layout microwave, found "
                f"{microwave_count}."
            )
        return tuple(overridden_items)

    def _task_layout_spec(self) -> Any:
        layout_spec = self.require_current_layout_spec()
        # Idempotently replace a task-private bowl if a caller rebuilds the
        # same selected layout. Any generic scene objects are intentionally
        # left untouched because they are shared content, not M12's target.
        private_item_names = {
            M12_BOWL_ENTITY_NAME,
            M12_BOWL_RISER_ENTITY_NAME,
        }
        collision_safe_items = self._with_l3_microwave_collision_override(
            layout_spec.build_items
        )
        retained_items = tuple(
            item
            for item in collision_safe_items
            if getattr(item, "name", None) not in private_item_names
        )
        riser_spec = self._bowl_riser_spec()
        task_private_items = (
            (self._bowl_spec(),)
            if riser_spec is None
            else (riser_spec, self._bowl_spec())
        )
        return replace(
            layout_spec,
            build_items=(*retained_items, *task_private_items),
            settle_sim_steps=max(
                layout_spec.settle_sim_steps,
                M12_SETTLE_SIM_STEPS,
            ),
        )

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        layout_builder_type = (
            M12OpenCavityLayoutBuilder
            if self.current_layout_id is LayoutId.KITCHEN_L3
            else LayoutBuilder
        )
        self.layout_scene = layout_builder_type(
            env=self,
            layout_spec=self._task_layout_spec(),
        ).build()
        self.bowl = self.layout_scene.require_entity(M12_BOWL_ENTITY_NAME)
        self.microwave = self._require_microwave_from_site()
        self.door_joint = self._resolve_microwave_door_joint()
        self.door_link = self.door_joint.child_link
        if self.door_link is None:
            raise RuntimeError(
                f"M12 microwave door joint {self.door_joint.name!r} has no child link."
            )
        if self.door_joint.active_index is None:
            raise RuntimeError(
                f"M12 microwave door joint {self.door_joint.name!r} is not active."
            )
        self._door_qpos_index = int(self.door_joint.active_index)

        self._configure_cavity_frame()
        self._configure_contact_queries()
        self._apply_physical_randomization()
        self._configure_step_constants()

    def _require_microwave_from_site(self) -> Any:
        """Resolve the layout-specific microwave using site name + source."""

        interaction_site = self.require_current_interaction_site()
        target_name = interaction_site.target_entity_name
        target_source = interaction_site.target_entity_source
        if not target_name or not target_source:
            raise RuntimeError(
                "M12 microwave site must define target_entity_name and "
                "target_entity_source."
            )

        if target_source == "layout_entity":
            return self.layout_scene.require_entity(target_name)

        if target_source != "external_fixture":
            raise RuntimeError(
                f"M12 does not support microwave target source {target_source!r}."
            )

        external_builders = getattr(
            self.layout_scene,
            "external_scene_builders",
            None,
        )
        if not external_builders:
            raise RuntimeError(
                "M12 requires a RoboCasa external scene builder for the current "
                "microwave site."
            )

        target_normalized = target_name.casefold()
        microwave_articulations: list[Any] = []
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
                        "M12 expected exactly one microwave fixture named "
                        f"{target_name!r} in RoboCasa scene {scene_index}, found "
                        f"{len(matches)}. Known fixtures: "
                        f"{sorted(map(str, fixtures))}."
                    )
                fixture = matches[0]
                articulation = getattr(fixture, "articulation", None)
                if articulation is None and hasattr(fixture, "active_joints_map"):
                    articulation = fixture
                if articulation is None:
                    raise RuntimeError(
                        f"RoboCasa microwave fixture {target_name!r} in scene "
                        f"{scene_index} is not articulated."
                    )
                microwave_articulations.append(articulation)

        if len(microwave_articulations) != self.num_envs:
            raise RuntimeError(
                "M12 microwave articulation count does not match num_envs: "
                f"articulations={len(microwave_articulations)}, "
                f"num_envs={self.num_envs}."
            )
        if self.num_envs == 1:
            return microwave_articulations[0]
        if Articulation is None:
            raise ImportError(
                "ManiSkill Articulation is required to merge batched RoboCasa "
                "microwave fixtures."
            )

        expected_joint_names = tuple(microwave_articulations[0].active_joints_map)
        for scene_index, articulation in enumerate(
            microwave_articulations[1:],
            start=1,
        ):
            joint_names = tuple(articulation.active_joints_map)
            if joint_names != expected_joint_names:
                raise RuntimeError(
                    "M12 microwave active joints differ across RoboCasa scenes: "
                    f"scene 0={expected_joint_names}, scene "
                    f"{scene_index}={joint_names}."
                )
        return Articulation.merge(
            microwave_articulations,
            name="m12_microwave",
            merge_links=True,
        )

    @staticmethod
    def _joint_type_name(joint: Any) -> str:
        joint_type = getattr(joint, "type", "")
        if isinstance(joint_type, (tuple, list)):
            joint_type = joint_type[0] if joint_type else ""
        return str(joint_type).casefold()

    @staticmethod
    def _joint_limit_span(joint: Any) -> float | None:
        """Return a representative finite joint-limit span, if available.

        SAPIEN loads a finite URDF ``revolute`` joint as
        ``revolute_unwrapped`` and a URDF ``continuous`` joint as
        ``revolute``.  PartNet microwave assets commonly contain both the
        finite door hinge and continuous control knobs, so the distinction is
        useful when link/joint names are only ``joint_N`` / ``link_N``.
        """

        limits = getattr(joint, "limits", None)
        if limits is None:
            return None
        try:
            if torch is not None and torch.is_tensor(limits):
                values = limits.detach().to(dtype=torch.float32).reshape(-1, 2)
                finite_rows = torch.isfinite(values).all(dim=1)
                if not bool(torch.any(finite_rows).item()):
                    return None
                spans = torch.abs(values[finite_rows, 1] - values[finite_rows, 0])
                if spans.numel() == 0:
                    return None
                return float(torch.median(spans).item())

            values = np.asarray(limits, dtype=np.float64).reshape(-1, 2)
            finite_rows = np.isfinite(values).all(axis=1)
            if not np.any(finite_rows):
                return None
            spans = np.abs(values[finite_rows, 1] - values[finite_rows, 0])
            if spans.size == 0:
                return None
            return float(np.median(spans))
        except (TypeError, ValueError, RuntimeError):
            return None

    def _resolve_microwave_door_joint(self) -> Any:
        """Resolve the microwave door while tolerating PartNet knob joints.

        RoboCasa microwaves often expose an identifiable door joint, whereas
        the Kitchen L3 PartNet microwave (model 7167) exposes generic names such
        as ``joint_0``/``link_0`` plus additional revolute knob joints.  The
        resolver therefore uses progressively weaker evidence:

        1. a unique revolute joint when there is only one;
        2. a unique semantic door/hinge joint or child-link name;
        3. a unique ``revolute_unwrapped`` joint (finite URDF revolute);
        4. the uniquely widest finite revolute range, if clearly separated.

        The last two rules distinguish a finite door hinge from continuous
        microwave knobs without hard-coding a PartNet joint index.
        """

        active_joints = tuple(self.microwave.active_joints)

        # Kitchen L3 uses the PartNet microwave referenced by the ``microwave``
        # site.  That asset exposes three generic revolute joints: joint_0 is
        # the door hinge (child link_0), while joint_1/joint_2 are the control
        # knobs.  Resolve this known layout-specific topology explicitly before
        # applying the generic heuristics used by RoboCasa and other assets.
        if self.current_layout_id == LayoutId.KITCHEN_L3:
            l3_door_candidates = tuple(
                joint
                for joint in active_joints
                if getattr(joint, "name", None) == "joint_0"
                and "revolute" in self._joint_type_name(joint)
            )
            if len(l3_door_candidates) == 1:
                return l3_door_candidates[0]
            raise RuntimeError(
                "M12 Kitchen L3 expected microwave door joint 'joint_0', but "
                f"found active joints: {[(getattr(j, 'name', None), self._joint_type_name(j), getattr(getattr(j, 'child_link', None), 'name', None)) for j in active_joints]}."
            )

        revolute_joints = tuple(
            joint
            for joint in active_joints
            if "revolute" in self._joint_type_name(joint)
        )
        if len(revolute_joints) == 1:
            return revolute_joints[0]

        def is_door_like(joint: Any) -> bool:
            child_name = getattr(getattr(joint, "child_link", None), "name", "")
            text = f"{getattr(joint, 'name', '')} {child_name}".casefold()
            return any(token in text for token in ("door", "hinge"))

        door_like = tuple(joint for joint in revolute_joints if is_door_like(joint))
        if len(door_like) == 1:
            return door_like[0]

        # SAPIEN's URDF loader maps finite URDF revolute joints to
        # ``revolute_unwrapped`` and continuous joints to ``revolute``.  This
        # directly handles Kitchen L3 model 7167, whose trace is:
        # joint_0=revolute_unwrapped, joint_1/2=revolute.
        bounded_revolute = tuple(
            joint
            for joint in revolute_joints
            if self._joint_type_name(joint) == "revolute_unwrapped"
        )
        if len(bounded_revolute) == 1:
            return bounded_revolute[0]

        # If an asset has several finite revolute joints, prefer a clearly wider
        # range.  Door hinges normally have a substantially larger useful range
        # than bounded buttons/knobs.  Require a margin so ambiguous assets still
        # fail loudly rather than silently selecting the wrong joint.
        finite_candidates: list[tuple[float, Any]] = []
        for joint in revolute_joints:
            span = self._joint_limit_span(joint)
            if span is not None and span > 1.0e-4:
                finite_candidates.append((span, joint))
        finite_candidates.sort(key=lambda item: item[0], reverse=True)
        if finite_candidates:
            if len(finite_candidates) == 1:
                return finite_candidates[0][1]
            largest_span, largest_joint = finite_candidates[0]
            second_span = finite_candidates[1][0]
            if largest_span >= max(1.25 * second_span, second_span + 0.20):
                return largest_joint

        joint_summary = [
            (
                getattr(joint, "name", "<unnamed>"),
                self._joint_type_name(joint),
                getattr(getattr(joint, "child_link", None), "name", None),
                self._joint_limit_span(joint),
            )
            for joint in active_joints
        ]
        raise RuntimeError(
            "M12 could not uniquely resolve the microwave door revolute joint. "
            "Joint tuples are (name, type, child_link, finite_limit_span_rad): "
            f"{joint_summary}."
        )

    # ---------------------------------------------------------------------
    # Geometry, contacts, and physical randomization
    # ---------------------------------------------------------------------

    def _configure_cavity_frame(self) -> None:
        site = self.require_current_interaction_site()
        site_origin = np.asarray(site.frame_pose.position_xyz, dtype=np.float32)
        robot_center = np.asarray(
            site.robot_base_center_pose.position_xyz,
            dtype=np.float32,
        )
        outward = robot_center - site_origin
        outward[2] = 0.0
        outward_norm = float(np.linalg.norm(outward))
        if outward_norm <= 1.0e-6:
            raise RuntimeError(
                "M12 microwave site and robot base must differ in horizontal position."
            )
        outward /= outward_norm
        inward = -outward
        up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
        lateral = np.cross(up, inward)
        lateral_norm = float(np.linalg.norm(lateral))
        if lateral_norm <= 1.0e-6:
            raise RuntimeError("M12 could not construct the microwave lateral axis.")
        lateral /= lateral_norm

        self._cavity_origin_world = torch.as_tensor(
            site_origin,
            device=self.device,
            dtype=torch.float32,
        )
        self._cavity_inward_world = torch.as_tensor(
            inward,
            device=self.device,
            dtype=torch.float32,
        )
        self._cavity_lateral_world = torch.as_tensor(
            lateral,
            device=self.device,
            dtype=torch.float32,
        )
        self._cavity_up_world = torch.as_tensor(
            up,
            device=self.device,
            dtype=torch.float32,
        )

        layout_id = self.current_layout_id
        if layout_id not in M12_CAVITY_LOCAL_BOUNDS_M:
            raise RuntimeError(
                f"M12 has no cavity bounds for {self.current_layout_id_value!r}."
            )
        lower, upper = M12_CAVITY_LOCAL_BOUNDS_M[layout_id]
        self._cavity_lower = (
            torch.as_tensor(
                lower,
                device=self.device,
                dtype=torch.float32,
            )
            - M12_CAVITY_CONTACT_TOLERANCE_M
        )
        self._cavity_upper = (
            torch.as_tensor(
                upper,
                device=self.device,
                dtype=torch.float32,
            )
            + M12_CAVITY_CONTACT_TOLERANCE_M
        )

    @staticmethod
    def _entity_links(entity: Any) -> tuple[Any, ...]:
        links = getattr(entity, "links", None)
        return tuple(links) if links is not None else (entity,)

    @staticmethod
    def _entity_pose(entity: Any) -> Any:
        root_pose = getattr(entity, "root_pose", None)
        return root_pose if root_pose is not None else entity.pose

    @staticmethod
    def _entity_linear_velocity(entity: Any) -> Any:
        for name in ("root_linear_velocity", "linear_velocity"):
            value = getattr(entity, name, None)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor):
                return value.reshape(-1, 3)
        for name in ("get_root_linear_velocity", "get_linear_velocity"):
            getter = getattr(entity, name, None)
            if callable(getter):
                return getter().reshape(-1, 3)
        raise TypeError(
            f"Entity {getattr(entity, 'name', '<unnamed>')!r} exposes no linear velocity."
        )

    @staticmethod
    def _entity_angular_velocity(entity: Any) -> Any:
        for name in ("root_angular_velocity", "angular_velocity"):
            value = getattr(entity, name, None)
            if callable(value):
                value = value()
            if isinstance(value, torch.Tensor):
                return value.reshape(-1, 3)
        for name in ("get_root_angular_velocity", "get_angular_velocity"):
            getter = getattr(entity, name, None)
            if callable(getter):
                return getter().reshape(-1, 3)
        raise TypeError(
            f"Entity {getattr(entity, 'name', '<unnamed>')!r} exposes no angular velocity."
        )

    def _configure_contact_queries(self) -> None:
        missing_fingers = [
            name
            for name in M12_GRASP_FINGER_LINK_NAMES
            if name not in self.agent.robot.links_map
        ]
        if missing_fingers:
            raise RuntimeError(
                "M12 requires Shadow distal finger links: " + ", ".join(missing_fingers)
            )
        self._finger_links = tuple(
            self.agent.robot.links_map[name] for name in M12_GRASP_FINGER_LINK_NAMES
        )

        # Prefer a separately named handle/edge link when the asset exposes one.
        # Otherwise use the actual door child link as the contact proxy.
        door_link_name = getattr(self.door_link, "name", "")
        handle_links = tuple(
            link
            for link in self.microwave.links
            if any(
                token in getattr(link, "name", "").casefold()
                for token in ("handle", "door_handle")
            )
        )
        self._door_contact_links = handle_links or (self.door_link,)

        excluded_support_names = {
            getattr(link, "name", "") for link in self._door_contact_links
        }
        excluded_support_names.add(door_link_name)
        self._microwave_support_links = tuple(
            link
            for link in self.microwave.links
            if getattr(link, "name", "") not in excluded_support_names
        )
        if not self._microwave_support_links:
            raise RuntimeError(
                "M12 resolved no non-door microwave links for cavity support contact."
            )
        self._bowl_grasp_query = self.agent.prepare_pairwise_contact_query(
            self._finger_links,
            self._entity_links(self.bowl),
        )
        self._bowl_support_query = self.agent.prepare_pairwise_contact_query(
            self._entity_links(self.bowl),
            self._microwave_support_links,
        )

        self._critical_robot_links = tuple(
            link
            for name in M12_CRITICAL_ROBOT_LINK_NAMES
            if (link := self.agent.robot.links_map.get(name)) is not None
        )
        if not self._critical_robot_links:
            raise RuntimeError("M12 resolved no critical robot links.")
        self._robot_microwave_query = self.agent.prepare_pairwise_contact_query(
            self._critical_robot_links,
            tuple(self.microwave.links),
        )
        self._robot_bowl_query = self.agent.prepare_pairwise_contact_query(
            self._critical_robot_links,
            self._entity_links(self.bowl),
        )

    def _configure_step_constants(self) -> None:
        self._grasp_dwell_steps_required = max(
            1,
            math.ceil(self.SUCCESS_SPEC.grasp_dwell_s * float(self._control_freq)),
        )
        self._placement_dwell_steps_required = max(
            1,
            math.ceil(self.SUCCESS_SPEC.placement_dwell_s * float(self._control_freq)),
        )
        self._final_dwell_steps_required = max(
            1,
            math.ceil(self.SUCCESS_SPEC.final_dwell_s * float(self._control_freq)),
        )

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

    def _raw_bodies_for_entity(self, entity: Any) -> tuple[Any, ...]:
        raw_bodies = getattr(entity, "_bodies", None)
        if raw_bodies is None:
            raw_bodies = getattr(entity, "_objs", None)
        return tuple(raw_bodies or ())

    def _set_entity_mass_scales(self, entity: Any, scales: np.ndarray) -> None:
        raw_bodies = self._raw_bodies_for_entity(entity)
        if len(raw_bodies) != self.num_envs:
            raise RuntimeError(
                f"M12 could not resolve one rigid body per environment for "
                f"{getattr(entity, 'name', '<unnamed>')!r}: "
                f"{len(raw_bodies)} vs {self.num_envs}."
            )
        for env_index, raw_body in enumerate(raw_bodies):
            component = self._rigid_component_for_raw_body(raw_body)
            if component is None:
                raise RuntimeError(
                    f"M12 could not resolve rigid component for env {env_index}."
                )
            nominal_mass = float(component.mass)
            if nominal_mass <= 0.0:
                continue
            target_mass = nominal_mass * float(scales[env_index])
            setter = getattr(component, "set_mass", None)
            if callable(setter):
                setter(target_mass)
            else:
                component.mass = target_mass

    def _set_entity_friction_scales(
        self,
        entity: Any,
        scales: np.ndarray,
        *,
        nominal_static: float,
        nominal_dynamic: float,
        require_shapes: bool = True,
    ) -> None:
        if sapien is None:
            raise ImportError("SAPIEN is required for M12 physical randomization.")
        raw_bodies = self._raw_bodies_for_entity(entity)
        if len(raw_bodies) != self.num_envs:
            raise RuntimeError(
                f"M12 friction randomization expected {self.num_envs} bodies for "
                f"{getattr(entity, 'name', '<unnamed>')!r}, got {len(raw_bodies)}."
            )
        for env_index, raw_body in enumerate(raw_bodies):
            component = self._rigid_component_for_raw_body(raw_body)
            if component is None:
                if require_shapes:
                    raise RuntimeError(
                        "M12 could not resolve a rigid component for friction "
                        f"randomization in env {env_index}."
                    )
                continue
            scale = float(scales[env_index])
            try:
                material = sapien.physx.PhysxMaterial(
                    static_friction=nominal_static * scale,
                    dynamic_friction=nominal_dynamic * scale,
                    restitution=0.0,
                )
            except TypeError:
                material = sapien.physx.PhysxMaterial(
                    nominal_static * scale,
                    nominal_dynamic * scale,
                    0.0,
                )
            shape_getter = getattr(component, "get_collision_shapes", None)
            shapes = (
                tuple(shape_getter())
                if callable(shape_getter)
                else tuple(getattr(component, "collision_shapes", ()))
            )
            if require_shapes and not shapes:
                raise RuntimeError(
                    f"M12 entity {getattr(entity, 'name', '<unnamed>')!r} has no "
                    "collision shapes for friction randomization."
                )
            for shape in shapes:
                setter = getattr(shape, "set_physical_material", None)
                if callable(setter):
                    setter(material)
                else:
                    shape.physical_material = material

    def _apply_physical_randomization(self) -> None:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return
        values = self._generalization_values(
            M12_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m12_physical_values",
        )
        bowl_mass_scales = positive_scale(values["bowl_mass_log_scale"])
        bowl_friction_scales = positive_scale(values["bowl_friction_log_scale"])
        support_friction_scales = positive_scale(values["support_friction_log_scale"])
        self._set_entity_mass_scales(self.bowl, bowl_mass_scales)
        self._set_entity_friction_scales(
            self.bowl,
            bowl_friction_scales,
            nominal_static=0.55,
            nominal_dynamic=0.45,
        )
        for support_link in self._microwave_support_links:
            self._set_entity_friction_scales(
                support_link,
                support_friction_scales,
                nominal_static=0.50,
                nominal_dynamic=0.40,
                require_shapes=False,
            )
        joint_objects = tuple(getattr(self.door_joint, "_objs", ()))
        if len(joint_objects) != self.num_envs:
            raise RuntimeError(
                f"M12 door joint exposes {len(joint_objects)} raw joints; expected "
                f"{self.num_envs}."
            )
        for env_index, joint_object in enumerate(joint_objects):
            joint_object.set_friction(float(values["door_joint_friction"][env_index]))
            joint_object.set_drive_properties(
                stiffness=0.0,
                damping=float(values["door_joint_damping"][env_index]),
                force_limit=1e10,
            )

    # ---------------------------------------------------------------------
    # Episode initialization / generalization
    # ---------------------------------------------------------------------

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_door_closed_qpos"):
            return
        self._door_closed_qpos = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._microwave_world_offset = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._door_opened_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._bowl_grasp_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._bowl_grasped_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._bowl_placement_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._bowl_placed_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._final_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._bowl_interacted_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._door_closed_confirmed = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._bowl_initial_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._microwave_initial_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._microwave_initial_quaternion = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self.device
        )
        self._microwave_initial_qpos = torch.zeros_like(self.microwave.get_qpos())

    def _door_limits(self) -> Any:
        return self.door_joint.limits.reshape(self.num_envs, -1, 2)[:, 0]

    def _door_open_target_and_direction(self) -> tuple[Any, Any, Any]:
        limits = self._door_limits()
        closed = self._door_closed_qpos
        distance_to_lower = torch.abs(closed - limits[:, 0])
        distance_to_upper = torch.abs(limits[:, 1] - closed)
        open_target = torch.where(
            distance_to_upper >= distance_to_lower,
            limits[:, 1],
            limits[:, 0],
        )
        signed_span = open_target - closed
        span = torch.abs(signed_span).clamp_min(1e-6)
        direction = torch.sign(signed_span)
        direction = torch.where(direction == 0.0, torch.ones_like(direction), direction)
        return open_target, direction, span

    def _apply_microwave_spatial_offset(self, env_idx: Any) -> None:
        env_idx = torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long
        ).reshape(-1)
        self._microwave_world_offset[env_idx] = 0.0
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return
        values = self._generalization_values(
            M12_SPATIAL_PARAMETER_NAMES,
            cache_name="_m12_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        forward = torch.as_tensor(
            values["microwave_forward_offset_m"][value_indices],
            device=self.device,
            dtype=torch.float32,
        )
        side = torch.as_tensor(
            values["microwave_side_offset_m"][value_indices],
            device=self.device,
            dtype=torch.float32,
        )
        offset = (
            forward[:, None] * self._cavity_inward_world[None, :]
            + side[:, None] * self._cavity_lateral_world[None, :]
        )
        current_pose = self.microwave.root_pose
        changed_position = current_pose.p[env_idx] + offset
        changed_pose = Pose.create_from_pq(
            p=changed_position,
            q=current_pose.q[env_idx],
        )
        self.microwave.set_pose(changed_pose)
        self._microwave_world_offset[env_idx] = offset

    def _place_bowl(self, env_idx: Any) -> None:
        env_idx = torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long
        ).reshape(-1)
        layout_id = self.current_layout_id
        if layout_id not in M12_BOWL_INITIAL_POSITION_XYZ:
            raise RuntimeError("M12 bowl layout is not configured.")
        batch_size = len(env_idx)
        base_position = (
            torch.as_tensor(
                M12_BOWL_INITIAL_POSITION_XYZ[layout_id],
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        yaw = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M12_SPATIAL_PARAMETER_NAMES,
                cache_name="_m12_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            base_position[:, 0] += torch.as_tensor(
                values["bowl_x_offset_m"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )
            base_position[:, 1] += torch.as_tensor(
                values["bowl_y_offset_m"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )
            yaw = torch.as_tensor(
                values["bowl_yaw_offset_rad"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )
        half_yaw = 0.5 * yaw
        quaternion = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ],
            dim=1,
        )
        self.bowl.set_pose(Pose.create_from_pq(p=base_position, q=quaternion))
        zero = torch.zeros((batch_size, 3), device=self.device, dtype=torch.float32)
        linear_setter = getattr(self.bowl, "set_linear_velocity", None)
        angular_setter = getattr(self.bowl, "set_angular_velocity", None)
        if not callable(linear_setter):
            linear_setter = getattr(self.bowl, "set_root_linear_velocity", None)
        if not callable(angular_setter):
            angular_setter = getattr(self.bowl, "set_root_angular_velocity", None)
        if not callable(linear_setter) or not callable(angular_setter):
            raise RuntimeError("M12 bowl does not expose velocity setters.")
        linear_setter(zero)
        angular_setter(zero)

    def _set_initial_door_state(self, env_idx: Any) -> None:
        env_idx = torch.as_tensor(
            env_idx, device=self.device, dtype=torch.long
        ).reshape(-1)
        qpos = self.microwave.get_qpos()[env_idx].clone()
        qvel = self.microwave.get_qvel()[env_idx].clone()
        self._door_closed_qpos[env_idx] = qpos[:, self._door_qpos_index]
        initial_fraction = torch.zeros(
            len(env_idx), device=self.device, dtype=torch.float32
        )
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M12_SPATIAL_PARAMETER_NAMES,
                cache_name="_m12_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            initial_fraction = torch.as_tensor(
                values["door_initial_open_fraction"][value_indices],
                device=self.device,
                dtype=torch.float32,
            )
        _, direction, span = self._door_open_target_and_direction()
        qpos[:, self._door_qpos_index] = (
            self._door_closed_qpos[env_idx]
            + direction[env_idx] * span[env_idx] * initial_fraction
        )
        qvel.zero_()
        self.microwave.set_qpos(qpos)
        self.microwave.set_qvel(qvel)

    def _reset_agent_to_rest(self, pose: Any, batch_size: int) -> None:
        rest = self.agent.keyframes.get("rest")
        if rest is None or rest.qpos is None:
            raise RuntimeError("M12 requires the robot rest keyframe.")
        qpos = torch.as_tensor(
            rest.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        self.agent.reset(qpos.unsqueeze(0).repeat(batch_size, 1))
        self.agent.robot.set_pose(pose)

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

            hidden_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
            self._reset_agent_to_rest(hidden_pose, batch_size)

            self._apply_microwave_spatial_offset(env_idx)
            self._set_initial_door_state(env_idx)
            self._place_bowl(env_idx)
            self.layout_scene.settle(env_idx=env_idx)

            # Passive settling may move the door slightly. Restore only the door
            # qpos/qvel, while leaving the bowl on the real support it settled on.
            qpos = self.microwave.get_qpos()[env_idx].clone()
            qvel = self.microwave.get_qvel()[env_idx].clone()
            _, direction, span = self._door_open_target_and_direction()
            initial_fraction = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32
            )
            if self.generalization_axis is GeneralizationAxis.SPATIAL:
                values = self._generalization_values(
                    M12_SPATIAL_PARAMETER_NAMES,
                    cache_name="_m12_spatial_values",
                )
                value_indices = env_idx.detach().cpu().numpy()
                initial_fraction = torch.as_tensor(
                    values["door_initial_open_fraction"][value_indices],
                    device=self.device,
                    dtype=torch.float32,
                )
            qpos[:, self._door_qpos_index] = (
                self._door_closed_qpos[env_idx]
                + direction[env_idx] * span[env_idx] * initial_fraction
            )
            qvel.zero_()
            self.microwave.set_qpos(qpos)
            self.microwave.set_qvel(qvel)

            self._bowl_initial_position[env_idx] = self._entity_pose(self.bowl).p[
                env_idx
            ]
            self._microwave_initial_position[env_idx] = self.microwave.root_pose.p[
                env_idx
            ]
            self._microwave_initial_quaternion[env_idx] = self.microwave.root_pose.q[
                env_idx
            ]
            self._microwave_initial_qpos[env_idx] = self.microwave.get_qpos()[env_idx]

            self._reset_agent_to_rest(self._robot_base_pose(env_idx), batch_size)

            self._door_opened_once[env_idx] = False
            self._bowl_grasp_steps[env_idx] = 0
            self._bowl_grasped_once[env_idx] = False
            self._bowl_placement_steps[env_idx] = 0
            self._bowl_placed_once[env_idx] = False
            self._final_steps[env_idx] = 0
            self._bowl_interacted_once[env_idx] = False
            self._door_closed_confirmed[env_idx] = True

    # ---------------------------------------------------------------------
    # Live task measurements
    # ---------------------------------------------------------------------

    @staticmethod
    def _quaternion_rotate(quaternion_wxyz: Any, vectors: Any) -> Any:
        quaternion = quaternion_wxyz / torch.linalg.norm(
            quaternion_wxyz,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        q_vector = quaternion[:, None, 1:]
        q_scalar = quaternion[:, None, :1]
        twice_cross = 2.0 * torch.cross(
            q_vector.expand_as(vectors),
            vectors,
            dim=-1,
        )
        return (
            vectors
            + q_scalar * twice_cross
            + torch.cross(
                q_vector.expand_as(vectors),
                twice_cross,
                dim=-1,
            )
        )

    def _door_open_fraction(self) -> Any:
        qpos = self.door_joint.qpos.reshape(self.num_envs, -1)[:, 0]
        _, direction, span = self._door_open_target_and_direction()
        return ((qpos - self._door_closed_qpos) * direction / span).clamp(0.0, 1.0)

    def _door_opening_velocity(self) -> Any:
        qvel = self.door_joint.qvel.reshape(self.num_envs, -1)[:, 0]
        _, direction, _ = self._door_open_target_and_direction()
        return qvel * direction

    def _door_accessible(self) -> Any:
        return self._door_open_fraction() >= self.SUCCESS_SPEC.door_access_open_fraction

    def _door_closed(self) -> Any:
        fraction = self._door_open_fraction()
        speed = torch.abs(self._door_opening_velocity())
        entered = (fraction <= self.SUCCESS_SPEC.door_closed_enter_fraction) & (
            speed <= self.SUCCESS_SPEC.door_closed_max_speed_rad_s
        )
        exited = fraction > self.SUCCESS_SPEC.door_closed_exit_fraction
        self._door_closed_confirmed = torch.where(
            entered,
            torch.ones_like(self._door_closed_confirmed),
            torch.where(
                exited,
                torch.zeros_like(self._door_closed_confirmed),
                self._door_closed_confirmed,
            ),
        )
        return self._door_closed_confirmed

    def _bowl_finger_force(self) -> Any:
        pair = self.agent.get_pairwise_contact_forces(self._bowl_grasp_query)
        # [B, 5, P, 3] -> [B, 5].
        return torch.linalg.norm(pair, dim=-1).amax(dim=2)

    def _bowl_grasped(self, finger_force: Any | None = None) -> Any:
        if finger_force is None:
            finger_force = self._bowl_finger_force()
        contact = finger_force >= self.SUCCESS_SPEC.grasp_contact_force_n
        # M12 is not a dexterity-specific benchmark; require two distinct fingers
        # rather than a precise thumb-opposition topology.
        return contact.sum(dim=1) >= self.SUCCESS_SPEC.grasp_min_fingers

    def _bowl_released(self, finger_force: Any | None = None) -> Any:
        if finger_force is None:
            finger_force = self._bowl_finger_force()
        return finger_force.amax(dim=1) <= (
            self.SUCCESS_SPEC.residual_hand_contact_force_n
        )

    def _bowl_support_force(self) -> Any:
        pair = self.agent.get_pairwise_contact_forces(self._bowl_support_query)
        return torch.linalg.norm(pair, dim=-1).amax(dim=(1, 2))

    def _bowl_supported(self, support_force: Any | None = None) -> Any:
        if support_force is None:
            support_force = self._bowl_support_force()
        return support_force >= self.SUCCESS_SPEC.support_contact_force_n

    def _bowl_motion_speeds(self) -> tuple[Any, Any]:
        linear_speed = torch.linalg.norm(
            self._entity_linear_velocity(self.bowl),
            dim=1,
        )
        angular_speed = torch.linalg.norm(
            self._entity_angular_velocity(self.bowl),
            dim=1,
        )
        return linear_speed, angular_speed

    def _bowl_stable(
        self,
        linear_speed: Any | None = None,
        angular_speed: Any | None = None,
    ) -> Any:
        if linear_speed is None or angular_speed is None:
            linear_speed, angular_speed = self._bowl_motion_speeds()
        return (linear_speed <= self.SUCCESS_SPEC.bowl_stable_linear_speed_mps) & (
            angular_speed <= self.SUCCESS_SPEC.bowl_stable_angular_speed_rad_s
        )

    def _bowl_world_up_axis(self) -> torch.Tensor:
        bowl_pose = self._entity_pose(self.bowl)
        local_up = torch.zeros(
            (self.num_envs, 1, 3),
            dtype=torch.float32,
            device=self.device,
        )
        local_up[:, :, 2] = 1.0
        # [B, 1, 3] -> [B, 3]: rotate the bowl's local +Z axis into world space.
        return self._quaternion_rotate(bowl_pose.q, local_up)[:, 0, :]

    def _bowl_tilt_angle(self) -> torch.Tensor:
        world_up = self._bowl_world_up_axis()
        return torch.acos(world_up[:, 2].clamp(-1.0, 1.0))

    def _bowl_upright(self, tilt_rad: Any | None = None) -> Any:
        if tilt_rad is None:
            tilt_rad = self._bowl_tilt_angle()
        return tilt_rad <= self.SUCCESS_SPEC.bowl_max_tilt_rad

    def _world_points_to_cavity_local(
        self,
        world_points: torch.Tensor,
    ) -> torch.Tensor:
        origin = self._cavity_origin_world.unsqueeze(0) + self._microwave_world_offset
        delta = world_points - origin[:, None, :]
        return self._world_vectors_to_cavity_local(delta)

    def _world_vectors_to_cavity_local(
        self,
        world_vectors: torch.Tensor,
    ) -> torch.Tensor:
        # [B, N, 3] -> [B, N, 3]: project world vectors onto cavity axes.
        inward = torch.sum(
            world_vectors * self._cavity_inward_world[None, None, :],
            dim=-1,
        )
        lateral = torch.sum(
            world_vectors * self._cavity_lateral_world[None, None, :],
            dim=-1,
        )
        vertical = torch.sum(
            world_vectors * self._cavity_up_world[None, None, :],
            dim=-1,
        )
        return torch.stack((inward, lateral, vertical), dim=-1)

    def _bowl_cavity_occupancy(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bowl_pose = self._entity_pose(self.bowl)
        center_local = self._world_points_to_cavity_local(
            bowl_pose.p[:, None, :]
        )[:, 0, :]
        bowl_axis_local = self._world_vectors_to_cavity_local(
            self._bowl_world_up_axis()[:, None, :]
        )[:, 0, :]
        axial_projection = torch.abs(bowl_axis_local).clamp(0.0, 1.0)
        radial_projection = torch.sqrt(
            torch.clamp(1.0 - axial_projection.square(), min=0.0)
        )
        # [B, 3]: exact support extent of a finite cylinder on each cavity axis.
        extent = (
            M12_BOWL_COLLISION_RADIUS_M * radial_projection
            + M12_BOWL_COLLISION_HALF_HEIGHT_M * axial_projection
        )
        return center_local, center_local - extent, center_local + extent

    def _bowl_fully_inside_cavity(
        self,
        occupied_lower: torch.Tensor | None = None,
        occupied_upper: torch.Tensor | None = None,
    ) -> torch.Tensor:
        override = getattr(self, "_state_injection_measurements", {}).get(
            "bowl_inside_microwave"
        )
        if override is not None:
            return self._as_per_env_tensor(override, dtype=torch.bool)
        if occupied_lower is None or occupied_upper is None:
            _, occupied_lower, occupied_upper = self._bowl_cavity_occupancy()
        return ((occupied_lower >= self._cavity_lower).all(dim=1)) & (
            (occupied_upper <= self._cavity_upper).all(dim=1)
        )

    def _bowl_root_cavity_local(self) -> torch.Tensor:
        point = self._entity_pose(self.bowl).p[:, None, :]
        return self._world_points_to_cavity_local(point)[:, 0, :]

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()

        door_open_fraction = self._door_open_fraction()
        door_accessible = (
            door_open_fraction >= self.SUCCESS_SPEC.door_access_open_fraction
        )
        bowl_finger_force = self._bowl_finger_force()
        bowl_grasped = self._bowl_grasped(bowl_finger_force)
        _, bowl_occupied_lower, bowl_occupied_upper = self._bowl_cavity_occupancy()
        bowl_inside = self._bowl_fully_inside_cavity(
            bowl_occupied_lower,
            bowl_occupied_upper,
        )
        bowl_support_force = self._bowl_support_force()
        bowl_supported = self._bowl_supported(bowl_support_force)
        bowl_linear_speed, bowl_angular_speed = self._bowl_motion_speeds()
        bowl_stable = self._bowl_stable(
            bowl_linear_speed,
            bowl_angular_speed,
        )
        bowl_upright = self._bowl_upright()
        bowl_released = self._bowl_released(bowl_finger_force)

        self._door_opened_once |= door_accessible
        self._bowl_interacted_once |= bowl_finger_force.amax(dim=1) > 0.05

        grasp_candidate = bowl_grasped & self._door_opened_once
        self._bowl_grasp_steps = torch.where(
            grasp_candidate,
            self._bowl_grasp_steps + 1,
            torch.zeros_like(self._bowl_grasp_steps),
        )
        grasp_confirmed = self._bowl_grasp_steps >= self._grasp_dwell_steps_required
        self._bowl_grasped_once |= grasp_confirmed

        placement_candidate = (
            self._door_opened_once
            & self._bowl_grasped_once
            & bowl_inside
            & bowl_supported
            & bowl_stable
            & bowl_upright
            & bowl_released
        )
        self._bowl_placement_steps = torch.where(
            placement_candidate,
            self._bowl_placement_steps + 1,
            torch.zeros_like(self._bowl_placement_steps),
        )
        placement_confirmed = (
            self._bowl_placement_steps >= self._placement_dwell_steps_required
        )
        self._bowl_placed_once |= placement_confirmed

        door_closed = self._door_closed()
        final_candidate = (
            self._bowl_placed_once
            & bowl_inside
            & bowl_supported
            & bowl_stable
            & bowl_upright
            & bowl_released
            & door_closed
        )
        self._final_steps = torch.where(
            final_candidate,
            self._final_steps + 1,
            torch.zeros_like(self._final_steps),
        )
        success = self._final_steps >= self._final_dwell_steps_required

        result = {
            "success": success,
            "open_microwave_door": self._door_opened_once,
            "grasp_bowl": self._bowl_grasped_once,
            "place_bowl_in_microwave": self._bowl_placed_once,
            "close_microwave_door": success,
            "door_open_fraction": self._door_open_fraction(),
            "door_closed": door_closed,
            "bowl_inside_microwave": bowl_inside,
            "bowl_supported": bowl_supported,
            "bowl_stable": bowl_stable,
            "bowl_upright": bowl_upright,
            "bowl_released": bowl_released,
        }
        return result

    # ---------------------------------------------------------------------
    # Generalization validity
    # ---------------------------------------------------------------------

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis not in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            return super()._check_generalization_validity(axis)
        bowl_pose = self._entity_pose(self.bowl)
        bowl_finite = torch.isfinite(bowl_pose.raw_pose).all(dim=1)
        microwave_finite = torch.isfinite(self.microwave.root_pose.raw_pose).all(dim=1)
        door_valid = (self._door_open_fraction() <= 0.15) & (
            torch.abs(self._door_opening_velocity()) <= 0.10
        )
        initial_bowl_outside = ~self._bowl_fully_inside_cavity()
        initial_bowl_reasonable = (
            torch.linalg.norm(self._entity_linear_velocity(self.bowl), dim=1) <= 0.25
        ) & (torch.linalg.norm(self._entity_angular_velocity(self.bowl), dim=1) <= 1.0)
        return (
            bowl_finite
            & microwave_finite
            & door_valid
            & initial_bowl_outside
            & initial_bowl_reasonable
            & self._bowl_upright()
        )

    # ---------------------------------------------------------------------
    # Violation monitoring
    # ---------------------------------------------------------------------





    @staticmethod
    def _quaternion_angular_distance(q0: Any, q1: Any) -> Any:
        q0 = q0 / torch.linalg.norm(q0, dim=1, keepdim=True).clamp_min(1e-8)
        q1 = q1 / torch.linalg.norm(q1, dim=1, keepdim=True).clamp_min(1e-8)
        dot = torch.abs(torch.sum(q0 * q1, dim=1)).clamp(0.0, 1.0)
        return 2.0 * torch.acos(dot)
