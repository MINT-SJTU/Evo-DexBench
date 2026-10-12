from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.assets import (
    evodex_asset_dir,
    partnet_mobility_model_urdf_path,
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
from dex_benchmark.sim.scenes.layouts import ArticulatedAssetSpec, LayoutBuilder, PoseSpec
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
    from transforms3d.euler import euler2quat
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    torch = None
    Pose = None
    euler2quat = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


B05_SETTINGS = TaskRegistry.register(TaskRegistry.load("b05_v1"))


@dataclass(frozen=True)
class B05SuccessSpec:
    containment_tolerance_m: float = 0.010
    lid_closed_fraction: float = 0.04
    object_static_linear_speed_mps: float = 0.035
    object_static_angular_speed_rad_s: float = 0.55
    object_contact_force_n: float = 0.40
    lid_jam_force_n: float = 25.0


B05_SUCCESS_SPEC = B05SuccessSpec()

B05_BOX_ENTITY_NAME = "box"
# Change only this value to switch between the B05-ready box assets below.
B05_BOX_MODEL_ID: str = "100664"
B05_SUPPORTED_BOX_MODEL_IDS: tuple[str, ...] = ("100664", "100194")
B05_THICK_BOX_SIZE_MULTIPLIER: float = 0.80
# Keep this false while bottle_blue disables its own self-collisions. ManiSkill
# assigns every self-collision-disabled articulation the same ignore-group bit;
# disabling both would also filter collisions between the bottle and box walls.
B05_BOX_DISABLE_SELF_COLLISIONS: bool = False
B05_BOX_SCALES_BY_MODEL_ID: dict[str, float] = {
    "100664": 0.45 * B05_THICK_BOX_SIZE_MULTIPLIER,
    "100194": 0.30,
}
B05_BOX_SITE_LOCAL_XYZ_BY_MODEL_ID: dict[
    str, tuple[float, float, float]
] = {
    # Thick, single hinged lid. Root height seats the repaired bottom on table.
    "100664": (
        0.15,
        0.0,
        0.222 * B05_THICK_BOX_SIZE_MULTIPLIER,
    ),
    # Original six-flap cardboard box.
    "100194": (
        0.15,
        0.0,
        1.09 - 0.924,
    ),
}
B05_LID_JOINT_NAMES_BY_MODEL_ID: dict[str, tuple[str, ...]] = {
    "100664": ("joint_0",),
    "100194": (
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
    ),
}
if B05_BOX_MODEL_ID not in B05_SUPPORTED_BOX_MODEL_IDS:
    raise ValueError(
        f"Unsupported B05_BOX_MODEL_ID={B05_BOX_MODEL_ID!r}; choose one of "
        f"{B05_SUPPORTED_BOX_MODEL_IDS}."
    )

B05_BOTTLE_BLUE_MODEL_ID = "3574"
B05_BOX_SCALE = B05_BOX_SCALES_BY_MODEL_ID[B05_BOX_MODEL_ID]
B05_CRACKER_BOX_DENSITY_KG_M3 = 1500.0
# Keep the box and all three target objects close to the dining-table operation
# centre. These are world-space yaws because ArticulatedAssetSpec receives a
# world pose: Kitchen L1 retains the original 90-degree counter-clockwise yaw;
# L2 turns the box clockwise by 90 degrees relative to L1, and L3 turns it by
# 180 degrees relative to L1.
B05_BOX_SITE_LOCAL_XYZ = B05_BOX_SITE_LOCAL_XYZ_BY_MODEL_ID[B05_BOX_MODEL_ID]
B05_BOX_YAW_RAD: float = math.radians(90.0)
B05_BOX_YAW_RAD_BY_LAYOUT_ID: dict[str, float] = {
    LayoutId.KITCHEN_L1.value: B05_BOX_YAW_RAD,
    LayoutId.KITCHEN_L2.value: B05_BOX_YAW_RAD - math.radians(90.0),
    LayoutId.KITCHEN_L3.value: B05_BOX_YAW_RAD + math.radians(180.0),
}
B05_BOX_QUATERNION_WXYZ: tuple[float, float, float, float] = (
    math.cos(B05_BOX_YAW_RAD / 2.0),
    0.0,
    0.0,
    math.sin(B05_BOX_YAW_RAD / 2.0),
)


def _box_quaternion_wxyz_for_layout(
    layout_id: str,
) -> tuple[float, float, float, float]:
    """Return the B05 box's world-space orientation for one Kitchen layout."""

    try:
        yaw = B05_BOX_YAW_RAD_BY_LAYOUT_ID[layout_id]
    except KeyError as exc:
        supported_layouts = ", ".join(sorted(B05_BOX_YAW_RAD_BY_LAYOUT_ID))
        raise ValueError(
            f"B05 has no box orientation for layout {layout_id!r}; expected one of "
            f"{supported_layouts}."
        ) from exc
    return (
        math.cos(0.5 * yaw),
        0.0,
        0.0,
        math.sin(0.5 * yaw),
    )
# The authored Kitchen bottle_blue is too tall for the packing box at its shared
# layout scale of 0.2. B05 uses the same asset at a task-local scale that leaves
# enough room for the lid to close.
B05_BOTTLE_BLUE_SCALE = 0.09
B05_BOTTLE_BLUE_LOCAL_XYZ = (
    -0.12,
    0.13,
    0.082,
)

# Contact materials used by B05. The hand material is applied to all Shadow
# Hand links, while the object material is applied to the three task objects.
B05_ROBOT_CONTACT_STATIC_FRICTION = 3.0
B05_ROBOT_CONTACT_DYNAMIC_FRICTION = 2.5
B05_OBJECT_CONTACT_STATIC_FRICTION = 1.20
B05_OBJECT_CONTACT_DYNAMIC_FRICTION = 1.00
B05_LID_JOINT_NAMES = B05_LID_JOINT_NAMES_BY_MODEL_ID[B05_BOX_MODEL_ID]
B05_OBJECT_ENTITY_NAMES = (
    "bottle_blue",
    "cracker_box",
    "mustard_bottle",
)
B05_IMPORTED_OBJECT_ENTITY_NAMES = (
    "cracker_box",
    "mustard_bottle",
)
B05_INTERACTION_SITE_ID = "dining_table"

# Visual-only packing reminder.  It lies on the near edge of the dining table,
# directly in front of the operator.  The actor has no collision geometry, so
# it cannot affect object motion, tactile queries, or task success.
B05_PACKING_GUIDE_TEXTURE_PATH = (
    evodex_asset_dir() / "task_assets" / "b05_packing_guide.png"
)
B05_PACKING_GUIDE_MESH_PATH = (
    evodex_asset_dir() / "task_assets" / "b05_instruction_quad.obj"
)
B05_PACKING_GUIDE_LOCAL_XYZ = (-0.35, 0.0, 0.003)
B05_PACKING_GUIDE_WIDTH_M = 0.20
B05_PACKING_GUIDE_HEIGHT_M = 0.20

# bottle_blue belongs to Kitchen L1/L2 and is added by B05 for Kitchen L3. The
# other two packing objects are task-local YCB actors.
#
# Canonical YCB names used by this repository:
#   cracker_box    -> 003_cracker_box
#   mustard_bottle -> 006_mustard_bottle
B05_IMPORTED_YCB_MODEL_IDS: dict[str, str] = {
    "cracker_box": "003_cracker_box",
    "mustard_bottle": "006_mustard_bottle",
}
B05_CRACKER_BOX_SCALE = 0.70
B05_IMPORTED_OBJECT_SCALES: dict[str, float] = {
    "cracker_box": B05_CRACKER_BOX_SCALE,
    "mustard_bottle": 1.0,
}

@dataclass(frozen=True)
class B05ObjectGeometrySpec:
    """Calibratable object-local bounding-box metadata.

    Imported YCB objects prefer bounds measured directly from their collision OBJ
    at runtime. The fallback center/half extents keep the task usable when mesh
    inspection is unavailable. PartNet bottle_blue uses calibrated bounds from
    model 3574 at the B05 task-local scale.
    """

    fallback_local_center_xyz: tuple[float, float, float]
    fallback_half_extents_m: tuple[float, float, float]
    ycb_model_id: str | None = None


B05_OBJECT_GEOMETRY_SPECS: dict[str, B05ObjectGeometrySpec] = {
    "bottle_blue": B05ObjectGeometrySpec(
        fallback_local_center_xyz=(0.0, 0.0, 0.0066),
        fallback_half_extents_m=(0.022, 0.022, 0.084),
    ),
    "cracker_box": B05ObjectGeometrySpec(
        fallback_local_center_xyz=(0.0, 0.0, 0.0),
        fallback_half_extents_m=(0.082, 0.036, 0.109),
        ycb_model_id=B05_IMPORTED_YCB_MODEL_IDS["cracker_box"],
    ),
    "mustard_bottle": B05ObjectGeometrySpec(
        fallback_local_center_xyz=(0.0, 0.0, 0.0),
        fallback_half_extents_m=(0.031, 0.049, 0.099),
        ycb_model_id=B05_IMPORTED_YCB_MODEL_IDS["mustard_bottle"],
    ),
}

# Backward-compatible view for code/tests that still inspect the previous
# half-extents constant. Runtime geometry uses B05_OBJECT_GEOMETRY_SPECS instead.
B05_OBJECT_HALF_EXTENTS_M: dict[str, tuple[float, float, float]] = {
    name: spec.fallback_half_extents_m
    for name, spec in B05_OBJECT_GEOMETRY_SPECS.items()
}

B05_IMPORTED_YCB_INITIAL_CLEARANCE_M = 0.005

# Interior bounds are calibrated strictly in box ROOT-LOCAL coordinates. B05
# creates the same PartNet box model with the same scale in every Kitchen layout,
# so all layouts share one calibration. No world-coordinate value enters the
# containment test. All eight object OBB corners must remain inside these limits
# after applying the configured outward tolerance.
B05_BOX_INTERIOR_BOUNDS_BY_MODEL_ID_M: dict[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
] = {
    # Bounds are box-root-local after applying the configured scale.
    "100664": (
        (
            -0.188 * B05_THICK_BOX_SIZE_MULTIPLIER,
            -0.260 * B05_THICK_BOX_SIZE_MULTIPLIER,
            -0.202 * B05_THICK_BOX_SIZE_MULTIPLIER,
        ),
        (
            0.096 * B05_THICK_BOX_SIZE_MULTIPLIER,
            0.234 * B05_THICK_BOX_SIZE_MULTIPLIER,
            0.081 * B05_THICK_BOX_SIZE_MULTIPLIER,
        ),
    ),
    "100194": ((-0.235, -0.185, 0.020), (0.235, 0.185, 0.235)),
}
B05_SELECTED_BOX_INTERIOR_BOUNDS_M = (
    B05_BOX_INTERIOR_BOUNDS_BY_MODEL_ID_M[B05_BOX_MODEL_ID]
)
B05_BOX_INTERIOR_BOUNDS_M: dict[
    str,
    tuple[tuple[float, float, float], tuple[float, float, float]],
] = {
    layout_id: B05_SELECTED_BOX_INTERIOR_BOUNDS_M
    for layout_id in ("kitchen_l1", "kitchen_l2", "kitchen_l3")
}

# Task-created YCB objects are placed in the dining-table interaction-site
# frame, not relative to the box. Therefore the same local offsets remain valid
# across Kitchen layouts even when the site's world pose changes.
#
# Local +Z is the table/support normal. The Z values below place each upright
# YCB root just above the site plane so layout_scene.settle() can seat it.
B05_IMPORTED_OBJECT_LOCAL_XYZ: dict[str, tuple[float, float, float]] = {
    "cracker_box": (
        -0.04,
        0.30,
        B05_IMPORTED_YCB_INITIAL_CLEARANCE_M,
    ),
    "mustard_bottle": (
        -0.10,
        -0.18,
        B05_IMPORTED_YCB_INITIAL_CLEARANCE_M,
    ),
}
B05_IMPORTED_OBJECT_RPY_RAD: dict[str, tuple[float, float, float]] = {
    "cracker_box": (0.0, 0.0, math.radians(90.0)),
    "mustard_bottle": (0.0, 0.0, math.radians(-18.0)),
}

B05_SPATIAL_PARAMETER_NAMES = (
    "box_x_offset_m",
    "box_y_offset_m",
    "box_yaw_offset_rad",
    "lid_initial_open_fraction",
    "cracker_box_x_offset_m",
    "cracker_box_y_offset_m",
    "cracker_box_yaw_offset_rad",
    "mustard_bottle_x_offset_m",
    "mustard_bottle_y_offset_m",
    "mustard_bottle_yaw_offset_rad",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
B05_PHYSICAL_PARAMETER_NAMES = (
    "object_mass_log_scale",
    "object_friction_scale",
    "box_interior_friction_scale",
    "lid_joint_friction",
    "lid_joint_damping",
)
B05_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("box_x_offset_m", -0.035, 0.035),
            GeneralizationParameterSpec("box_y_offset_m", -0.035, 0.035),
            GeneralizationParameterSpec(
                "box_yaw_offset_rad",
                -math.radians(8.0),
                math.radians(8.0),
            ),
            GeneralizationParameterSpec(
                "lid_initial_open_fraction", 0.68, 0.96
            ),
            GeneralizationParameterSpec(
                "cracker_box_x_offset_m",
                -0.035,
                0.035,
            ),
            GeneralizationParameterSpec(
                "cracker_box_y_offset_m",
                -0.035,
                0.035,
            ),
            GeneralizationParameterSpec(
                "cracker_box_yaw_offset_rad",
                -math.radians(18.0),
                math.radians(18.0),
            ),
            GeneralizationParameterSpec(
                "mustard_bottle_x_offset_m",
                -0.035,
                0.035,
            ),
            GeneralizationParameterSpec(
                "mustard_bottle_y_offset_m",
                -0.035,
                0.035,
            ),
            GeneralizationParameterSpec(
                "mustard_bottle_yaw_offset_rad",
                -math.radians(18.0),
                math.radians(18.0),
            ),
            GeneralizationParameterSpec("robot_forward_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("robot_side_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec(
                "robot_yaw_offset_rad",
                -math.radians(3.0),
                math.radians(3.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "object_mass_log_scale", math.log(0.75), math.log(1.30)
            ),
            GeneralizationParameterSpec("object_friction_scale", 0.70, 1.35),
            GeneralizationParameterSpec(
                "box_interior_friction_scale", 0.75, 1.35
            ),
            GeneralizationParameterSpec("lid_joint_friction", 0.02, 0.16),
            GeneralizationParameterSpec("lid_joint_damping", 0.03, 0.22),
        ),
        requires_rebuild=True,
    ),
)

B05_HIDDEN_Z_M = -20.0


B05_ALLOWED_LID_JOINT_TYPES = {
    "revolute",
    "revolute_unwrapped",
    "continuous",
}


@register_env(B05_SETTINGS.env_id, max_episode_steps=B05_SETTINGS.max_steps)
class B05Env(DexVLABenchBaseEnv):
    """Place the centres of three differently shaped objects inside the box."""

    TASK_SETTINGS = B05_SETTINGS
    SUCCESS_SPEC = B05_SUCCESS_SPEC
    GENERALIZATION_SPEC = B05_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B05_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        # Captured from the B05 Human Viewer on the default kitchen_l1 layout.
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(3.245553, -3.501561, 1.745055),
            quaternion_world_wxyz=(
                0.649845,
                -0.267689,
                0.263220,
                0.660878,
            ),
        ),
        # These layouts retain the same dining-table-relative view direction.
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.998439, -3.245553, 1.745055),
            quaternion_world_wxyz=(
                0.926821122,
                -0.003160060,
                0.375409354,
                0.007801509,
            ),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.854447, -3.498439, 1.745055),
            quaternion_world_wxyz=(
                0.660878,
                0.263220,
                0.267689,
                -0.649845,
            ),
        ),
    }

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_poses(self, env_idx: Any | None = None) -> list[Any]:
        nominal_poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(nominal_poses, list) or len(nominal_poses) != 2:
            raise RuntimeError(
                "B05 requires exactly two interaction-site robot base poses."
            )
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_poses

        values = self._generalization_values(
            B05_SPATIAL_PARAMETER_NAMES,
            cache_name="_b05_spatial_values",
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
        local_offset_pose = Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
            device=self.device,
        )
        return [
            Pose.create(nominal_pose) * local_offset_pose
            for nominal_pose in nominal_poses
        ]

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_poses())

    def _dual_agents(self) -> tuple[Any, Any]:
        return self.left_agent, self.right_agent

    @staticmethod
    def _joint_type_values(joint: Any) -> tuple[str, ...]:
        raw_types = joint.type
        if isinstance(raw_types, (tuple, list)):
            return tuple(str(value) for value in raw_types)
        return (str(raw_types),)

    def _require_lid_joints(self) -> tuple[Any, ...]:
        """Get all known box lid joints by their exact runtime names."""

        active_joints = self.box.active_joints_map
        known_joint_names = tuple(active_joints)

        missing_joint_names = [
            joint_name
            for joint_name in B05_LID_JOINT_NAMES
            if joint_name not in active_joints
        ]

        if missing_joint_names:
            raise RuntimeError(
                f"Box entity {B05_BOX_ENTITY_NAME!r} does not expose "
                f"the required lid joints {missing_joint_names}. "
                f"Known active joints: {known_joint_names}."
            )

        lid_joints = tuple(
            active_joints[joint_name]
            for joint_name in B05_LID_JOINT_NAMES
        )

        for joint in lid_joints:
            raw_joint_types = joint.type

            if isinstance(raw_joint_types, (tuple, list)):
                joint_types = tuple(
                    str(joint_type)
                    for joint_type in raw_joint_types
                )
            else:
                joint_types = (str(raw_joint_types),)

            invalid_joint_types = [
                joint_type
                for joint_type in joint_types
                if joint_type not in B05_ALLOWED_LID_JOINT_TYPES
            ]

            if invalid_joint_types:
                raise RuntimeError(
                    f"Box lid joint {joint.name!r} must be rotational, "
                    f"got type {joint.type!r}."
                )

        return lid_joints

    def _require_lid_child_link(self, joint: Any) -> Any:
        child_link = getattr(joint, "child_link", None)

        if child_link is None:
            raise RuntimeError(
                f"Box lid joint {joint.name!r} has no child link."
            )

        return child_link

    def _build_task_ycb_actor(
        self,
        *,
        name: str,
        model_id: str,
    ) -> Any:
        """Build one dynamic YCB actor owned only by B05.

        Visual geometry uses textured.obj and collision uses the repository's
        precomputed textured_vhacd.obj, matching the existing Kitchen YCB assets.
        The actor starts hidden and is placed from the interaction-site anchor
        during _initialize_episode().
        """

        if sapien is None:
            raise ImportError("SAPIEN is required to build B05 YCB objects.")

        visual_path = ycb_model_obj_path(model_id)
        collision_path = ycb_model_obj_path(
            model_id,
            "textured_vhacd.obj",
        )
        if not visual_path.is_file():
            raise FileNotFoundError(
                f"B05 YCB visual mesh was not found: {visual_path}"
            )
        if not collision_path.is_file():
            raise FileNotFoundError(
                f"B05 YCB collision mesh was not found: {collision_path}"
            )

        builder = self.scene.create_actor_builder()
        if name not in B05_IMPORTED_OBJECT_SCALES:
            raise KeyError(f"B05 has no imported-object scale for {name!r}.")
        object_scale = B05_IMPORTED_OBJECT_SCALES[name]
        scale = [object_scale, object_scale, object_scale]
        builder.add_visual_from_file(
            filename=str(visual_path),
            scale=scale,
        )
        builder.add_multiple_convex_collisions_from_file(
            filename=str(collision_path),
            scale=scale,
            density=(
                B05_CRACKER_BOX_DENSITY_KG_M3
                if name == "cracker_box"
                else 1000.0
            ),
        )
        builder.initial_pose = sapien.Pose(
            p=[0.0, 0.0, B05_HIDDEN_Z_M]
        )
        return builder.build(name=name)

    def _build_packing_guide(self) -> Any:
        """Build the visual-only tabletop card for the B05 packing task."""

        if sapien is None:
            raise ImportError("SAPIEN is required to build the B05 packing guide.")
        if not B05_PACKING_GUIDE_TEXTURE_PATH.is_file():
            raise FileNotFoundError(
                "B05 packing-guide texture was not found: "
                f"{B05_PACKING_GUIDE_TEXTURE_PATH}"
            )
        if not B05_PACKING_GUIDE_MESH_PATH.is_file():
            raise FileNotFoundError(
                "B05 packing-guide mesh was not found: "
                f"{B05_PACKING_GUIDE_MESH_PATH}"
            )

        texture = sapien.render.RenderTexture2D(
            str(B05_PACKING_GUIDE_TEXTURE_PATH)
        )
        material = sapien.render.RenderMaterial(
            base_color=[1.0, 1.0, 1.0, 1.0]
        )
        material.set_base_color_texture(texture)

        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=str(B05_PACKING_GUIDE_MESH_PATH),
            # Unit YZ quad; the reset pose rotates it onto the XY table plane.
            scale=[
                1.0,
                B05_PACKING_GUIDE_WIDTH_M,
                B05_PACKING_GUIDE_HEIGHT_M,
            ],
            material=material,
            name="b05_packing_guide_visual",
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, B05_HIDDEN_Z_M])
        return builder.build_kinematic(name="b05_packing_guide")

    def _update_packing_guide_pose(self) -> None:
        """Lay the instruction card on the operator side of every table."""

        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                local_xyz=(0.0, 0.0, 0.0),
                batch_size=self.num_envs,
            ),
            q=self.interaction_site_quaternion_tensor(self.num_envs),
        )
        local_position = torch.as_tensor(
            B05_PACKING_GUIDE_LOCAL_XYZ,
            device=self.device,
            dtype=torch.float32,
            # [3] -> [1, 3] -> [B, 3]: same guide anchor in every scene.
        ).unsqueeze(0).repeat(self.num_envs, 1)
        local_quaternion = torch.tensor(
            euler2quat(0.0, math.pi / 2.0, 0.0),
            device=self.device,
            dtype=torch.float32,
            # [4] -> [1, 4] -> [B, 4]: lay the YZ quad onto the XY table plane.
        ).unsqueeze(0).repeat(self.num_envs, 1)
        self.packing_guide.set_pose(
            site_pose * Pose.create_from_pq(
                p=local_position,
                q=local_quaternion,
            )
        )

    def _load_scene(self, options: dict[str, Any]) -> None:
        box_world_xyz = self.interaction_site_local_point_to_world(
            B05_BOX_SITE_LOCAL_XYZ
        )
        task_build_items: list[Any] = []
        bottle_blue_count = 0
        bottle_yellow_item: ArticulatedAssetSpec | None = None
        for build_item in self.current_layout_spec.build_items:
            if build_item.name == "bottle_yellow":
                if not isinstance(build_item, ArticulatedAssetSpec):
                    raise TypeError(
                        "B05 bottle_yellow must be an ArticulatedAssetSpec, got "
                        f"{type(build_item).__name__}."
                    )
                if bottle_yellow_item is not None:
                    raise ValueError(
                        "B05 layout contains more than one bottle_yellow."
                    )
                bottle_yellow_item = build_item
                continue
            if build_item.name == "bottle_blue":
                if not isinstance(build_item, ArticulatedAssetSpec):
                    raise TypeError(
                        "B05 bottle_blue must be an ArticulatedAssetSpec, got "
                        f"{type(build_item).__name__}."
                    )
                bottle_blue_count += 1
                build_item = replace(
                    build_item,
                    scale=B05_BOTTLE_BLUE_SCALE,
                )
            task_build_items.append(build_item)
        if bottle_yellow_item is None:
            raise ValueError(
                "B05 layout must contain exactly one bottle_yellow to replace."
            )
        if bottle_blue_count == 0:
            task_build_items.append(
                replace(
                    bottle_yellow_item,
                    name="bottle_blue",
                    asset_path=str(
                        partnet_mobility_model_urdf_path(
                            B05_BOTTLE_BLUE_MODEL_ID
                        )
                    ),
                    scale=B05_BOTTLE_BLUE_SCALE,
                )
            )
        elif bottle_blue_count != 1:
            raise ValueError(
                "B05 layout must contain at most one bottle_blue, found "
                f"{bottle_blue_count}."
            )
        task_layout_spec = replace(
            self.current_layout_spec,
            build_items=[
                *task_build_items,
                ArticulatedAssetSpec(
                    name=B05_BOX_ENTITY_NAME,
                    asset_path=str(
                        partnet_mobility_modified_model_urdf_path(
                            B05_BOX_MODEL_ID
                        )
                    ),
                    pose=PoseSpec(
                        position_xyz=tuple(
                            float(value) for value in box_world_xyz
                        ),
                        quaternion_wxyz=_box_quaternion_wxyz_for_layout(
                            self.current_layout_id_value
                        ),
                    ),
                    scale=B05_BOX_SCALE,
                    fix_root_link=False,
                    disable_self_collisions=B05_BOX_DISABLE_SELF_COLLISIONS,
                ),
            ],
        )
        multi_agent = self.agent
        try:
            self.agent = None
            self.layout_scene = LayoutBuilder(
                env=self,
                layout_spec=task_layout_spec,
            ).build()
        finally:
            self.agent = multi_agent

        self.box = self.layout_scene.require_entity(
            B05_BOX_ENTITY_NAME
        )

        self._nominal_box_root_position = self.box.root_pose.p.clone()
        self._nominal_box_root_quaternion = self.box.root_pose.q.clone()

        # B05 replaces the glass bottle with the blue bottle and places it near
        # the task box during reset.
        self.bottle_blue = self.layout_scene.require_entity(
            "bottle_blue"
        )

        # The other two packing targets exist only for this task.
        self.cracker_box = self._build_task_ycb_actor(
            name="cracker_box",
            model_id=B05_IMPORTED_YCB_MODEL_IDS["cracker_box"],
        )
        self.mustard_bottle = self._build_task_ycb_actor(
            name="mustard_bottle",
            model_id=B05_IMPORTED_YCB_MODEL_IDS["mustard_bottle"],
        )
        self.packing_guide = self._build_packing_guide()

        self.task_objects_by_name = {
            "bottle_blue": self.bottle_blue,
            "cracker_box": self.cracker_box,
            "mustard_bottle": self.mustard_bottle,
        }
        self.task_objects = tuple(
            self.task_objects_by_name[object_name]
            for object_name in B05_OBJECT_ENTITY_NAMES
        )
        (
            self._object_bbox_local_centres,
            self._object_bbox_half_extents,
            self._object_bbox_sources,
        ) = self._resolve_object_geometry_bounds()

        self.lid_joints = self._require_lid_joints()

        self.lid_links = tuple(
            self._require_lid_child_link(joint)
            for joint in self.lid_joints
        )
        self._task_tactile_target_groups = (
            ("lids", self.lid_links),
            *tuple(
                (
                    object_name,
                    self._entity_contact_links(task_object),
                )
                for object_name, task_object in zip(
                    B05_OBJECT_ENTITY_NAMES,
                    self.task_objects,
                    strict=True,
                )
            ),
        )
        if not self.gpu_sim_enabled:
            for agent in self._dual_agents():
                agent.prepare_cpu_object_taxel_query(
                    self._task_tactile_target_groups
                )

        active_joint_names = tuple(self.box.active_joints_map)

        self._lid_joint_qpos_indices = tuple(
            active_joint_names.index(joint.name)
            for joint in self.lid_joints
        )

        for joint_index, (joint, link, qpos_index) in enumerate(
            zip(
                self.lid_joints,
                self.lid_links,
                self._lid_joint_qpos_indices,
                strict=True,
            )
        ):
            print(
                f"[{joint_index}] "
                f"joint={joint.name!r}, "
                f"type={joint.type!r}, "
                f"child_link={link.name!r}, "
                f"qpos_index={qpos_index}, "
                f"limits={joint.limits}"
            )

        print("=====================================\n")

        self._inter_arm_contact_query = (
            self.left_agent.prepare_pairwise_contact_query(
                tuple(self.left_agent.robot.links),
                tuple(self.right_agent.robot.links),
            )
        )
        self._apply_b05_physical_properties()
    @staticmethod
    def _obj_vertex_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Return object-local (center, half-extents) from OBJ vertices.

        Reading the authored collision OBJ is more reliable than hand-entered
        dimensions and also captures mesh origins that are not at bbox center.
        """

        lower = np.full(3, np.inf, dtype=np.float64)
        upper = np.full(3, -np.inf, dtype=np.float64)
        vertex_count = 0
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if not line.startswith("v "):
                    continue
                fields = line.split()
                if len(fields) < 4:
                    continue
                vertex = np.asarray(
                    [float(fields[1]), float(fields[2]), float(fields[3])],
                    dtype=np.float64,
                )
                if not np.isfinite(vertex).all():
                    continue
                lower = np.minimum(lower, vertex)
                upper = np.maximum(upper, vertex)
                vertex_count += 1

        if vertex_count == 0 or not np.all(upper > lower):
            raise ValueError(
                f"OBJ {path} contains no usable 3D vertex bounds."
            )
        return 0.5 * (lower + upper), 0.5 * (upper - lower)

    def _resolve_object_geometry_bounds(self,) :
        """Resolve calibrated object-local OBBs, preferring collision meshes.

        cracker_box and mustard_bottle are built directly from YCB OBJ geometry,
        so their runtime collision mesh is an authoritative source. The PartNet
        yellow bottle spans URDF links and therefore deliberately retains its
        centralized fallback until a link-aware PartNet bounds extractor is added.
        """

        centres = {}
        half_extents = {}
        sources = {}

        for object_name in B05_OBJECT_ENTITY_NAMES:
            spec = B05_OBJECT_GEOMETRY_SPECS[object_name]
            centre = np.asarray(
                spec.fallback_local_center_xyz, dtype=np.float64
            )
            extent = np.asarray(
                spec.fallback_half_extents_m, dtype=np.float64
            )
            source = "fallback configuration"

            if spec.ycb_model_id is not None:
                collision_path = ycb_model_obj_path(
                    spec.ycb_model_id, "textured_vhacd.obj"
                )
                visual_path = ycb_model_obj_path(
                    spec.ycb_model_id, "textured.obj"
                )
                for candidate, label in (
                    (collision_path, "collision OBJ"),
                    (visual_path, "visual OBJ"),
                ):
                    if not candidate.is_file():
                        continue
                    try:
                        centre, extent = self._obj_vertex_bounds(candidate)
                    except (OSError, ValueError) as exc:
                        print(
                            f"WARNING: B05 could not read {label} bounds for "
                            f"{object_name!r} from {candidate}: {exc}"
                        )
                        continue
                    source = f"{label}: {candidate}"
                    break

            object_scale = B05_IMPORTED_OBJECT_SCALES.get(object_name, 1.0)
            centre = centre * object_scale
            extent = extent * object_scale

            if not np.isfinite(centre).all() or not np.isfinite(extent).all():
                raise RuntimeError(
                    f"B05 invalid OBB calibration for {object_name!r}: "
                    f"center={centre}, half_extents={extent}."
                )
            if np.any(extent <= 0.0):
                raise RuntimeError(
                    f"B05 non-positive OBB half extents for {object_name!r}: "
                    f"{extent}."
                )

            centres[object_name] = torch.as_tensor(
                centre, device=self.device, dtype=torch.float32
            )
            half_extents[object_name] = torch.as_tensor(
                extent, device=self.device, dtype=torch.float32
            )
            sources[object_name] = source
            print(
                f"B05 geometry {object_name}: source={source}; "
                f"center={centre.tolist()}; half_extents={extent.tolist()}"
            )

        return centres, half_extents, sources

    def _reset_agent_to_rest(
        self,
        agent: Any,
        robot_pose: Any,
        batch_size: int,
    ) -> None:
        keyframes = getattr(agent, "keyframes", None)
        rest_keyframe = keyframes.get("rest") if keyframes else None
        if rest_keyframe is None or getattr(rest_keyframe, "qpos", None) is None:
            raise RuntimeError(
                f"B05 agent {agent.uid!r} must expose a 'rest' keyframe."
            )
        robot_qpos = torch.as_tensor(
            rest_keyframe.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        if robot_qpos.ndim == 1:
            robot_qpos = robot_qpos.unsqueeze(0).repeat(batch_size, 1)
        expected_dof = agent.robot.max_dof
        if robot_qpos.shape != (batch_size, expected_dof):
            raise ValueError(
                f"B05 agent {agent.uid!r} rest qpos must have shape "
                f"({batch_size}, {expected_dof}), got {tuple(robot_qpos.shape)}."
            )
        agent.reset(robot_qpos)
        agent.robot.set_pose(robot_pose)

    @staticmethod
    def _entity_pose(entity: Any) -> Any:
        if hasattr(entity, "root_pose"):
            return entity.root_pose
        if hasattr(entity, "pose"):
            return entity.pose
        raise TypeError(
            f"Entity {type(entity).__name__} exposes neither root_pose nor pose."
        )

    @staticmethod
    def _set_entity_pose(entity: Any, pose: Any) -> None:
        setter = getattr(entity, "set_pose", None)
        if not callable(setter):
            raise TypeError(
                f"Entity {type(entity).__name__} does not expose set_pose()."
            )
        setter(pose)

    @staticmethod
    def _entity_linear_velocity(entity: Any) -> Any:
        for attribute_name in ("root_linear_velocity", "linear_velocity"):
            value = getattr(entity, attribute_name, None)
            if value is not None:
                return value
        raise TypeError(
            f"Entity {type(entity).__name__} exposes no linear velocity."
        )

    @staticmethod
    def _entity_angular_velocity(entity: Any) -> Any:
        for attribute_name in ("root_angular_velocity", "angular_velocity"):
            value = getattr(entity, attribute_name, None)
            if value is not None:
                return value
        raise TypeError(
            f"Entity {type(entity).__name__} exposes no angular velocity."
        )

    def _zero_entity_velocity(self, entity: Any, batch_size: int) -> None:
        zero = torch.zeros(
            (batch_size, 3), device=self.device, dtype=torch.float32
        )
        linear_setters = (
            "set_root_linear_velocity",
            "set_linear_velocity",
        )
        angular_setters = (
            "set_root_angular_velocity",
            "set_angular_velocity",
        )
        for setter_name in linear_setters:
            setter = getattr(entity, setter_name, None)
            if callable(setter):
                setter(zero)
                break
        for setter_name in angular_setters:
            setter = getattr(entity, setter_name, None)
            if callable(setter):
                setter(zero)
                break

    def _initial_lid_open_fraction(self, env_idx: Any) -> Any:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return torch.full(
                (len(env_idx),),
                0.88,
                device=self.device,
                dtype=torch.float32,
            )
        values = self._generalization_values(
            B05_SPATIAL_PARAMETER_NAMES,
            cache_name="_b05_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        return torch.as_tensor(
            values["lid_initial_open_fraction"][value_indices],
            device=self.device,
            dtype=torch.float32,
        )

    def _lid_endpoints(self) -> tuple[Any, Any, Any]:
        limits = torch.stack(
            [
                joint.limits.reshape(self.num_envs, -1, 2,)[:, 0, :]
                for joint in self.lid_joints
            ],
            dim=1,
        )
        first = limits[:, :, 0]
        second = limits[:, :, 1]
        # For the PartNet box, the negative/larger-magnitude endpoint is the
        # flat, folded-over (closed) pose.  The positive endpoint leaves each
        # lid hanging outward, so gravity preserves the open initial state.
        first_is_closed = torch.abs(first) >= torch.abs(second)
        closed = torch.where(first_is_closed, first, second)
        opened = torch.where(first_is_closed, second, first)
        joint_range = torch.abs(opened - closed).clamp_min(1e-6)
        return closed, opened, joint_range

    def _reset_lid_joints(self, env_idx: Any) -> None:
        closed, opened, _ = self._lid_endpoints()
        open_fraction = self._initial_lid_open_fraction(env_idx).reshape(-1, 1)
        target = closed[env_idx] + open_fraction * (
            opened[env_idx] - closed[env_idx]
        )
        qpos = self.box.get_qpos()[env_idx].clone()
        if target.shape != (
            len(env_idx),
            len(self.lid_joints),
        ):
            raise RuntimeError(
                "Invalid lid target shape: "
                f"target={tuple(target.shape)}, "
                f"expected="
                f"{(len(env_idx), len(self.lid_joints))}, "
                f"closed={tuple(closed.shape)}, "
                f"opened={tuple(opened.shape)}, "
                f"open_fraction={tuple(open_fraction.shape)}."
            )

        for lid_index, qpos_index in enumerate(
            self._lid_joint_qpos_indices
        ):
            qpos[:, qpos_index] = target[:, lid_index]

        self.box.set_qpos(qpos)
        self.box.set_qvel(torch.zeros_like(qpos))

    def _reset_box_pose(self, env_idx: Any) -> None:
        current_position = self.box.root_pose.p.clone()
        current_quaternion = self.box.root_pose.q.clone()
        nominal_position = self._nominal_box_root_position[env_idx]
        nominal_quaternion = self._nominal_box_root_quaternion[env_idx]

        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                B05_SPATIAL_PARAMETER_NAMES,
                cache_name="_b05_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            local_offset = np.stack(
                [
                    values["box_x_offset_m"][value_indices],
                    values["box_y_offset_m"][value_indices],
                    np.zeros(len(value_indices), dtype=np.float64),
                ],
                axis=1,
            )
            yaw = values["box_yaw_offset_rad"][value_indices]
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
            nominal_pose = Pose.create_from_pq(
                p=nominal_position,
                q=nominal_quaternion,
            )
            changed_pose = nominal_pose * Pose.create_from_pq(
                p=local_offset,
                q=local_quaternion,
            )
            current_position[env_idx] = changed_pose.p
            current_quaternion[env_idx] = changed_pose.q
        else:
            current_position[env_idx] = nominal_position
            current_quaternion[env_idx] = nominal_quaternion

        self.box.set_pose(
            Pose.create_from_pq(
                p=current_position,
                q=current_quaternion,
            )
        )
        self._zero_entity_velocity(self.box, len(env_idx))

    def _object_spatial_values(
        self,
        object_name: str,
        env_idx: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value_indices = env_idx.detach().cpu().numpy()
        count = len(value_indices)
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return (
                np.zeros(count, dtype=np.float64),
                np.zeros(count, dtype=np.float64),
                np.zeros(count, dtype=np.float64),
            )
        values = self._generalization_values(
            B05_SPATIAL_PARAMETER_NAMES,
            cache_name="_b05_spatial_values",
        )
        return (
            values[f"{object_name}_x_offset_m"][value_indices],
            values[f"{object_name}_y_offset_m"][value_indices],
            values[f"{object_name}_yaw_offset_rad"][value_indices],
        )

    def _place_imported_task_objects(self, env_idx: Any) -> None:
        """Place all B05 objects around the dining-table operation centre."""

        batch_size = len(env_idx)
        if batch_size <= 0:
            return

        # World pose of the dining-table interaction-site frame.
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                local_xyz=(0.0, 0.0, 0.0),
                batch_size=batch_size,
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

        for object_name in B05_IMPORTED_OBJECT_ENTITY_NAMES:
            task_object = self.task_objects_by_name[object_name]
            x_offset, y_offset, yaw_offset = (
                self._object_spatial_values(
                    object_name,
                    env_idx,
                )
            )

            nominal_xyz = B05_IMPORTED_OBJECT_LOCAL_XYZ[
                object_name
            ]
            local_position = np.stack(
                [
                    nominal_xyz[0] + x_offset,
                    nominal_xyz[1] + y_offset,
                    np.full(
                        batch_size,
                        nominal_xyz[2],
                        dtype=np.float64,
                    ),
                ],
                axis=1,
            )

            roll, pitch, nominal_yaw = (
                B05_IMPORTED_OBJECT_RPY_RAD[object_name]
            )
            local_quaternions = np.stack(
                [
                    euler2quat(
                        roll,
                        pitch,
                        nominal_yaw + float(yaw_value),
                    )
                    for yaw_value in yaw_offset
                ],
                axis=0,
            )

            local_pose = Pose.create_from_pq(
                p=local_position,
                q=local_quaternions,
                device=self.device,
            )
            self._set_entity_pose(
                task_object,
                site_pose * local_pose,
            )
            self._zero_entity_velocity(
                task_object,
                batch_size,
            )

        # The blue bottle is part of the task layout, so reset it here after
        # LayoutBuilder has restored the authored pose for the episode.
        bottle_pose = Pose.create_from_pq(
            p=np.tile(
                np.asarray(B05_BOTTLE_BLUE_LOCAL_XYZ, dtype=np.float64),
                (batch_size, 1),
            ),
            q=np.tile(
                np.asarray(B05_BOX_QUATERNION_WXYZ, dtype=np.float64),
                (batch_size, 1),
            ),
            device=self.device,
        )
        self._set_entity_pose(self.bottle_blue, site_pose * bottle_pose)
        self._zero_entity_velocity(self.bottle_blue, batch_size)

    def _initialize_episode(
        self,
        env_idx: Any,
        options: dict[str, Any],
    ) -> None:
        with torch.device(self.device):
            self._invalidate_contact_force_cache()
            batch_size = len(env_idx)
            multi_agent = self.agent
            try:
                self.agent = None
                self.layout_scene.initialize_episode(
                    env_idx=env_idx,
                    batched_episode_rng=self._batched_episode_rng,
                    device=self.device,
                )
            finally:
                self.agent = multi_agent

            # Keep both robots away while the dynamic box and loose objects settle.
            for agent in self._dual_agents():
                agent.robot.set_pose(
                    sapien.Pose(p=[0.0, 0.0, B05_HIDDEN_Z_M])
                )
            self._reset_box_pose(env_idx)
            self._reset_lid_joints(env_idx)
            self._place_imported_task_objects(env_idx)
            self._update_packing_guide_pose()
            multi_agent = self.agent
            try:
                self.agent = None
                self.layout_scene.settle(env_idx=env_idx)
            finally:
                self.agent = multi_agent

            base_poses = self._robot_base_poses(env_idx)
            for agent, base_pose in zip(
                self._dual_agents(),
                base_poses,
                strict=True,
            ):
                self._reset_agent_to_rest(agent, base_pose, batch_size)

            self._ensure_episode_state()
            self._object_was_touched[env_idx] = False
            self._object_was_inside[env_idx] = False
            self._hand_participated[env_idx] = False
            self._packed_reference_valid[env_idx] = False
            self._packed_reference_local_positions[env_idx] = 0.0
            self._initial_box_position[env_idx] = self.box.root_pose.p[env_idx]

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_object_was_touched"):
            return
        object_count = len(self.task_objects)
        self._object_was_touched = torch.zeros(
            (self.num_envs, object_count),
            dtype=torch.bool,
            device=self.device,
        )
        self._object_was_inside = torch.zeros(
            (self.num_envs, object_count),
            dtype=torch.bool,
            device=self.device,
        )
        self._hand_participated = torch.zeros(
            (self.num_envs, 2),
            dtype=torch.bool,
            device=self.device,
        )
        self._packed_reference_valid = torch.zeros(
            (self.num_envs, object_count),
            dtype=torch.bool,
            device=self.device,
        )
        self._packed_reference_local_positions = torch.zeros(
            (self.num_envs, object_count, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._initial_box_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )

    def _invalidate_contact_force_cache(self) -> None:
        self._cached_cpu_contacts: list[Any] | None = None
        self._cached_lid_tactile_force_norms: Any | None = None
        self._cached_object_tactile_force_norms: Any | None = None
        self._cached_hand_task_contact_flags: Any | None = None

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self._invalidate_contact_force_cache()

    def _lid_open_fractions(self) -> Any:
        """Return the opening fraction of every lid joint as [B, L]."""

        lid_qpos = torch.stack(
            [
                joint.qpos.reshape(
                    self.num_envs,
                    -1,
                )[:, 0]
                for joint in self.lid_joints
            ],
            dim=1,
        )
        # lid_qpos: [B, L]

        closed, opened, joint_range = self._lid_endpoints()

        opening_direction = torch.sign(opened - closed)
        opening_direction = torch.where(
            torch.abs(opening_direction) < 1e-6,
            torch.ones_like(opening_direction),
            opening_direction,
        )

        displacement_toward_open = (
            lid_qpos - closed
        ) * opening_direction

        return (
            displacement_toward_open / joint_range
        ).clamp(0.0, 1.0)

    @staticmethod
    def _quaternion_to_rotation_matrix(quaternion_wxyz: Any) -> Any:
        quaternion = quaternion_wxyz / torch.linalg.norm(
            quaternion_wxyz, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(dim=-1)
        return torch.stack(
            [
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        ).reshape(-1, 3, 3)

    def _object_bbox_corners_world(
        self,
        object_name: str,
        task_object: Any,
    ) -> Any:
        half_extent = self._object_bbox_half_extents[object_name]
        local_center = self._object_bbox_local_centres[object_name]
        signs = torch.as_tensor(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        local_corners = local_center + signs * half_extent
        pose = self._entity_pose(task_object)
        rotation = self._quaternion_to_rotation_matrix(pose.q)
        return (
            torch.einsum("bij,kj->bki", rotation, local_corners)
            + pose.p[:, None, :]
        )

    def _object_bbox_corners_in_box(
        self,
        object_name: str,
        task_object: Any,
    ) -> Any:
        world_corners = self._object_bbox_corners_world(
            object_name,
            task_object,
        )
        box_pose = self.box.root_pose
        inverse_rotation = self._quaternion_to_rotation_matrix(
            box_pose.q
        ).transpose(1, 2)
        return torch.einsum(
            "bij,bkj->bki",
            inverse_rotation,
            world_corners - box_pose.p[:, None, :],
        )

    def _interior_bounds(self) -> tuple[Any, Any]:
        layout_id = self.current_layout_id_value
        try:
            lower_xyz, upper_xyz = B05_BOX_INTERIOR_BOUNDS_M[layout_id]
        except KeyError as exc:
            raise RuntimeError(
                f"B05 has no calibrated box interior bounds for layout "
                f"{layout_id!r}. Known layouts: "
                f"{sorted(B05_BOX_INTERIOR_BOUNDS_M)}."
            ) from exc
        tolerance = self.SUCCESS_SPEC.containment_tolerance_m
        lower = torch.as_tensor(
            lower_xyz, device=self.device, dtype=torch.float32
        ) - tolerance
        upper = torch.as_tensor(
            upper_xyz, device=self.device, dtype=torch.float32
        ) + tolerance
        return lower, upper

    def _object_inside_flags(self) -> Any:
        lower, upper = self._interior_bounds()
        flags = []
        for object_name, task_object in self.task_objects_by_name.items():
            corners = self._object_bbox_corners_in_box(
                object_name,
                task_object,
            )
            flags.append(
                ((corners >= lower) & (corners <= upper)).all(dim=-1).all(dim=-1)
            )
        return torch.stack(flags, dim=1)

    def _object_local_centres(self) -> Any:
        inverse_box_pose = self.box.root_pose.inv()
        return torch.stack(
            [
                (inverse_box_pose * self._entity_pose(task_object)).p
                for task_object in self.task_objects
            ],
            dim=1,
        )

    def _object_static_flags(self) -> Any:
        flags = []
        for task_object in self.task_objects:
            linear_speed = torch.linalg.norm(
                self._entity_linear_velocity(task_object), dim=1
            )
            angular_speed = torch.linalg.norm(
                self._entity_angular_velocity(task_object), dim=1
            )
            flags.append(
                (
                    linear_speed
                    <= self.SUCCESS_SPEC.object_static_linear_speed_mps
                )
                & (
                    angular_speed
                    <= self.SUCCESS_SPEC.object_static_angular_speed_rad_s
                )
            )
        return torch.stack(flags, dim=1)

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            all_objects_outside = ~self._object_inside_flags().any(dim=1)
            all_objects_static = self._object_static_flags().all(dim=1)
            lid_open = self._lid_open_fractions().amin(dim=1) >= 0.60
            box_static = torch.linalg.norm(
                self.box.root_linear_velocity, dim=1
            ) <= self.SUCCESS_SPEC.object_static_linear_speed_mps
            return all_objects_outside & all_objects_static & lid_open & box_static
        return super()._check_generalization_validity(axis)

    def _lid_tactile_force_norms(self) -> Any:
        if self._cached_lid_tactile_force_norms is None:
            self._populate_tactile_force_cache()
        assert self._cached_lid_tactile_force_norms is not None
        return self._cached_lid_tactile_force_norms

    def _object_tactile_force_norms(self) -> Any:
        if self._cached_object_tactile_force_norms is None:
            self._populate_tactile_force_cache()
        assert self._cached_object_tactile_force_norms is not None
        return self._cached_object_tactile_force_norms

    def _hand_task_contact_flags(self) -> Any:
        if self._cached_hand_task_contact_flags is None:
            self._populate_tactile_force_cache()
        assert self._cached_hand_task_contact_flags is not None
        return self._cached_hand_task_contact_flags

    def _populate_tactile_force_cache(self) -> None:
        per_hand_lid_norms: list[Any] = []
        per_hand_object_norms: list[Any] = []
        hand_contact_flags: list[Any] = []
        for agent in self._dual_agents():
            group_forces = agent.get_object_taxel_world_forces(
                self._task_tactile_target_groups
            )
            lid_parent_forces = agent.aggregate_taxel_world_forces_by_parent(
                group_forces["lids"]
            )
            # [B, P, 3] -> [B, P], where P is the tactile parent-link count.
            lid_norms = torch.linalg.norm(lid_parent_forces, dim=-1)
            object_norms = torch.stack(
                tuple(
                    torch.linalg.norm(
                        agent.aggregate_taxel_world_forces_by_parent(
                            group_forces[object_name]
                        ),
                        dim=-1,
                    )
                    for object_name in B05_OBJECT_ENTITY_NAMES
                ),
                dim=1,
            )
            per_hand_lid_norms.append(lid_norms)
            per_hand_object_norms.append(object_norms)
            hand_contact_flags.append(
                (
                    lid_norms
                    >= self.SUCCESS_SPEC.object_contact_force_n
                ).any(dim=1)
                | (
                    object_norms
                    >= self.SUCCESS_SPEC.object_contact_force_n
                ).any(dim=(1, 2))
            )

        # [B, P] + [B, P] -> [B, 2P]; [B, O, P] likewise concatenates hands.
        self._cached_lid_tactile_force_norms = torch.cat(
            per_hand_lid_norms,
            dim=1,
        )
        self._cached_object_tactile_force_norms = torch.cat(
            per_hand_object_norms,
            dim=2,
        )
        # [B, 2]: one task-contact participation flag per physical hand.
        self._cached_hand_task_contact_flags = torch.stack(
            hand_contact_flags,
            dim=1,
        )

    def _cpu_contacts(self) -> list[Any]:
        if self._cached_cpu_contacts is None:
            self._cached_cpu_contacts = list(self.scene.px.get_contacts())
        return self._cached_cpu_contacts

    @staticmethod
    def _contact_impulse(contact: Any) -> np.ndarray:
        impulse = np.zeros(3, dtype=np.float32)
        for contact_point in contact.points:
            impulse += np.asarray(contact_point.impulse, dtype=np.float32)
        return impulse

    @staticmethod
    def _raw_contact_entities(target: Any) -> dict[Any, int]:
        result: dict[Any, int] = {}
        bodies = getattr(target, "_bodies", None)
        if bodies is not None:
            for env_index, body in enumerate(bodies):
                result[body.entity] = env_index
            return result
        objects = getattr(target, "_objs", None)
        if objects is not None:
            for env_index, raw_object in enumerate(objects):
                result[raw_object] = env_index
            return result
        links = getattr(target, "links", None)
        if links is not None:
            for link in links:
                result.update(B05Env._raw_contact_entities(link))
        return result

    def _object_has_tactile_contact_flags(self) -> Any:
        return (
            self._object_tactile_force_norms()
            >= self.SUCCESS_SPEC.object_contact_force_n
        ).any(dim=2)

    def _entity_contact_links(self, entity: Any) -> tuple[Any, ...]:
        links = getattr(entity, "links", None)
        if links is not None:
            return tuple(links)
        return (entity,)

    def _lid_object_contact_force(self) -> Any:
        max_force = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        for task_object in self.task_objects:
            for object_link in self._entity_contact_links(task_object):
                if self.gpu_sim_enabled:
                    force = self.scene.get_pairwise_contact_forces(
                        self.lid_links,
                        object_link,
                    )
                    max_force = torch.maximum(
                        max_force,
                        torch.linalg.norm(force, dim=1),
                    )
                else:
                    pair_force = self._cpu_max_pairwise_contact_force(
                        self.lid_links,
                        (object_link,),
                    )
                    max_force = torch.maximum(max_force, pair_force)
        return max_force

    def _diagnostic_pairwise_contact_force(
        self,
        first_links: tuple[Any, ...],
        second_links: tuple[Any, ...],
    ) -> Any:
        """Return maximum pairwise force for diagnostics, never success gating."""

        if not self.gpu_sim_enabled:
            return self._cpu_max_pairwise_contact_force(
                first_links,
                second_links,
            )

        max_force = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        for first_link in first_links:
            for second_link in second_links:
                pair_force = self.scene.get_pairwise_contact_forces(
                    first_link,
                    second_link,
                )
                max_force = torch.maximum(
                    max_force,
                    torch.linalg.norm(pair_force, dim=1),
                )
        return max_force

    def get_task_success_diagnostics(self) -> dict[str, Any]:
        """Return live B05 success gates and non-gating debug measurements."""

        object_names = tuple(self.task_objects_by_name)
        inside = self._object_inside_flags()
        lower, upper = self._interior_bounds()
        local_centres = self._object_local_centres()
        # [B, O, 3]: signed centre-point clearance to the nearest accepted
        # boundary on each box-local axis. Negative values are outside.
        boundary_margin = torch.minimum(
            local_centres - lower[None, None, :],
            upper[None, None, :] - local_centres,
        )
        linear_speed = torch.stack(
            [
                torch.linalg.norm(
                    self._entity_linear_velocity(task_object),
                    dim=1,
                )
                for task_object in self.task_objects
            ],
            dim=1,
        )
        angular_speed = torch.stack(
            [
                torch.linalg.norm(
                    self._entity_angular_velocity(task_object),
                    dim=1,
                )
                for task_object in self.task_objects
            ],
            dim=1,
        )

        lid_open_fraction = self._lid_open_fractions()
        lid_qpos = torch.stack(
            [
                joint.qpos.reshape(self.num_envs, -1)[:, 0]
                for joint in self.lid_joints
            ],
            dim=1,
        )
        lid_qvel = torch.stack(
            [
                joint.qvel.reshape(self.num_envs, -1)[:, 0]
                for joint in self.lid_joints
            ],
            dim=1,
        )

        lid_link_ids = {id(link) for link in self.lid_links}
        box_body_links = tuple(
            link for link in self.box.links if id(link) not in lid_link_ids
        )
        object_box_force = torch.stack(
            [
                self._diagnostic_pairwise_contact_force(
                    self._entity_contact_links(task_object),
                    box_body_links,
                )
                for task_object in self.task_objects
            ],
            dim=1,
        )

        pair_names: list[str] = []
        pair_forces: list[Any] = []
        for first_index, first_name in enumerate(object_names):
            for second_index in range(first_index + 1, len(object_names)):
                second_name = object_names[second_index]
                pair_names.append(f"{first_name}<->{second_name}")
                pair_forces.append(
                    self._diagnostic_pairwise_contact_force(
                        self._entity_contact_links(
                            self.task_objects_by_name[first_name]
                        ),
                        self._entity_contact_links(
                            self.task_objects_by_name[second_name]
                        ),
                    )
                )
        object_pair_force = torch.stack(pair_forces, dim=1)

        all_objects_inside = inside.all(dim=1)
        lid_closed = (
            lid_open_fraction.amax(dim=1)
            <= self.SUCCESS_SPEC.lid_closed_fraction
        )
        return {
            "schema": "b05_packing_v1",
            "task_id": "b05_v1",
            "object_names": object_names,
            "inside": inside,
            "boundary_margin_m": boundary_margin,
            "local_centres_m": local_centres,
            "linear_speed_mps": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "lid_names": tuple(joint.name for joint in self.lid_joints),
            "lid_open_fraction": lid_open_fraction,
            "lid_qpos_rad": lid_qpos,
            "lid_qvel_rad_s": lid_qvel,
            "object_box_force_n": object_box_force,
            "object_pair_names": tuple(pair_names),
            "object_pair_force_n": object_pair_force,
            "lid_object_force_n": self._lid_object_contact_force(),
            "all_objects_inside": all_objects_inside,
            "lid_closed": lid_closed,
            "success": all_objects_inside & lid_closed,
            "left_hand_participated": self._hand_participated[:, 0],
            "right_hand_participated": self._hand_participated[:, 1],
            "thresholds": {
                "containment_tolerance_m": (
                    self.SUCCESS_SPEC.containment_tolerance_m
                ),
                "lid_closed_fraction": self.SUCCESS_SPEC.lid_closed_fraction,
            },
        }

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        self._hand_participated |= self._hand_task_contact_flags()
        inside_flags = self._object_inside_flags()
        static_flags = self._object_static_flags()
        packed_count = inside_flags.sum(dim=1)

        first_object_packed = packed_count >= 1
        second_object_packed = packed_count >= 2
        all_objects_packed = packed_count == len(self.task_objects)
        lid_closed = (
            self._lid_open_fractions().amax(dim=1)
            <= self.SUCCESS_SPEC.lid_closed_fraction
        )
        goal_satisfied_now = all_objects_packed & lid_closed
        success = self._latch_task_success(goal_satisfied_now)

        object_touched = self._object_has_tactile_contact_flags()
        self._object_was_touched |= object_touched
        self._object_was_inside |= inside_flags

        local_centres = self._object_local_centres()
        new_reference = inside_flags & static_flags & ~self._packed_reference_valid
        self._packed_reference_local_positions = torch.where(
            new_reference[:, :, None],
            local_centres,
            self._packed_reference_local_positions,
        )
        self._packed_reference_valid |= new_reference
        return {
            "success": success,
            "first_object_packed": first_object_packed,
            "second_object_packed": second_object_packed,
            "all_objects_packed": all_objects_packed,
            "left_hand_participated": self._hand_participated[:, 0],
            "right_hand_participated": self._hand_participated[:, 1],
            "box_lid_fully_closed": success,
            "goal_satisfied_now": goal_satisfied_now,
        }


    def _table_reference_position(self) -> Any:
        return self.interaction_site_local_point_tensor(
            local_xyz=(0.0, 0.0, 0.0),
            batch_size=self.num_envs,
        )



    def _critical_robot_links(self) -> tuple[Any, ...]:
        critical_names = (
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_link7",
            "panda_hand",
            "rh_forearm",
            "rh_palm",
            "lh_forearm",
            "lh_palm",
        )
        return tuple(
            link
            for agent in self._dual_agents()
            for name in critical_names
            if (link := agent.robot.links_map.get(name)) is not None
        )


    def _cpu_max_pairwise_contact_force(
        self,
        first_links: tuple[Any, ...],
        second_links: tuple[Any, ...],
    ) -> Any:
        if not first_links or not second_links:
            return torch.zeros(
                self.num_envs,
                dtype=torch.float32,
                device=self.device,
            )

        first_entities: dict[Any, tuple[int, int]] = {}
        for link_index, link in enumerate(first_links):
            for entity, env_index in self._raw_contact_entities(link).items():
                first_entities[entity] = (env_index, link_index)
        second_entities: dict[Any, tuple[int, int]] = {}
        for link_index, link in enumerate(second_links):
            for entity, env_index in self._raw_contact_entities(link).items():
                second_entities[entity] = (env_index, link_index)

        pair_impulses = np.zeros(
            (
                self.num_envs,
                len(first_links),
                len(second_links),
                3,
            ),
            dtype=np.float32,
        )
        for contact in self._cpu_contacts():
            body0_entity = contact.bodies[0].entity
            body1_entity = contact.bodies[1].entity
            first_index = first_entities.get(body0_entity)
            second_index = second_entities.get(body1_entity)
            direction = 1.0
            if first_index is None or second_index is None:
                first_index = first_entities.get(body1_entity)
                second_index = second_entities.get(body0_entity)
                direction = -1.0
            if first_index is None or second_index is None:
                continue
            first_env_index, first_link_index = first_index
            second_env_index, second_link_index = second_index
            if first_env_index != second_env_index:
                continue
            pair_impulses[
                first_env_index,
                first_link_index,
                second_link_index,
            ] += direction * self._contact_impulse(contact)

        pair_forces = torch.as_tensor(
            pair_impulses,
            device=self.device,
            dtype=torch.float32,
        ) / float(self.scene.px.timestep)
        return torch.linalg.norm(pair_forces, dim=-1).amax(dim=(1, 2))



    def _apply_b05_physical_properties(self) -> None:
        """Apply nominal contact materials before optional randomization."""

        self._apply_b05_contact_friction()
        self._apply_box_physical_randomization()

    def _apply_box_physical_randomization(self) -> None:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return

        values = self._generalization_values(
            B05_PHYSICAL_PARAMETER_NAMES,
            cache_name="_b05_physical_values",
        )

        object_mass_scales = positive_scale(values["object_mass_log_scale"])
        object_friction_scales = values["object_friction_scale"]
        box_interior_friction_scales = values[
            "box_interior_friction_scale"
        ]
        for task_object in self.task_objects:
            object_friction_applied = False
            for rigid_entity in self._entity_contact_links(task_object):
                self._scale_entity_mass(rigid_entity, object_mass_scales)
                object_friction_applied |= self._scale_entity_friction(
                    rigid_entity,
                    object_friction_scales,
                    require_collision_shapes=False,
                )
            if not object_friction_applied:
                raise RuntimeError(
                    "B05 task object exposes no collision shapes for friction "
                    f"randomization: {getattr(task_object, 'name', task_object)!r}."
                )

        lid_link_names = {link.name for link in self.lid_links}
        box_body_links = tuple(
            link for link in self.box.links if link.name not in lid_link_names
        )
        if not box_body_links:
            raise RuntimeError(
                "B05 could not resolve a non-lid box link for interior friction."
            )
        box_friction_applied = False
        for box_body_link in box_body_links:
            box_friction_applied |= self._scale_entity_friction(
                box_body_link,
                box_interior_friction_scales,
                require_collision_shapes=False,
            )
        if not box_friction_applied:
            raise RuntimeError(
                "B05 box exposes no non-lid collision shapes for interior "
                "friction randomization."
            )

        friction_values = values["lid_joint_friction"]
        damping_values = values["lid_joint_damping"]

        for lid_joint in self.lid_joints:
            joint_objects = tuple(
                getattr(lid_joint, "_objs", ())
            )

            if len(joint_objects) != self.num_envs:
                raise RuntimeError(
                    f"Box lid joint {lid_joint.name!r} exposes "
                    f"{len(joint_objects)} underlying joint objects, "
                    f"expected {self.num_envs}."
                )

            for env_index, joint_object in enumerate(joint_objects):
                friction = float(
                    friction_values[env_index]
                )
                damping = float(
                    damping_values[env_index]
                )

                joint_object.set_friction(
                    friction=friction,
                )
                joint_object.set_drive_properties(
                    stiffness=0.0,
                    damping=damping,
                    force_limit=1e10,
                )

    def _apply_b05_contact_friction(self) -> None:
        """Increase friction on B05 objects and the robot hand collision links."""

        for task_object in self.task_objects:
            for rigid_entity in self._entity_contact_links(task_object):
                self._set_entity_contact_material(
                    rigid_entity,
                    static_friction=B05_OBJECT_CONTACT_STATIC_FRICTION,
                    dynamic_friction=B05_OBJECT_CONTACT_DYNAMIC_FRICTION,
                )

        # The box is also a grasp target during packing; apply the same object
        # contact material to its body and lid links.
        for box_link in self.box.links:
            self._set_entity_contact_material(
                box_link,
                static_friction=B05_OBJECT_CONTACT_STATIC_FRICTION,
                dynamic_friction=B05_OBJECT_CONTACT_DYNAMIC_FRICTION,
            )

        for agent in self._dual_agents():
            hand_links = tuple(
                link
                for link in agent.robot.links
                if str(getattr(link, "name", "")).startswith(("rh_", "lh_"))
            )
            for hand_link in hand_links:
                self._set_entity_contact_material(
                    hand_link,
                    static_friction=B05_ROBOT_CONTACT_STATIC_FRICTION,
                    dynamic_friction=B05_ROBOT_CONTACT_DYNAMIC_FRICTION,
                )

    def _set_entity_contact_material(
        self,
        entity: Any,
        *,
        static_friction: float,
        dynamic_friction: float,
    ) -> None:
        if sapien is None:
            raise ImportError("SAPIEN is required for B05 contact materials.")

        components = self._underlying_rigid_components(entity)
        if len(components) != self.num_envs:
            raise RuntimeError(
                "B05 contact material could not resolve one rigid body per "
                "environment for "
                f"{getattr(entity, 'name', type(entity).__name__)!r}; "
                f"resolved {len(components)} for {self.num_envs}."
            )

        try:
            material = sapien.physx.PhysxMaterial(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=0.02,
            )
        except TypeError:
            material = sapien.physx.PhysxMaterial(
                static_friction,
                dynamic_friction,
                0.02,
            )

        for component in components:
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

    @staticmethod
    def _underlying_rigid_components(entity: Any) -> tuple[Any, ...]:
        raw_objects = getattr(entity, "_objs", None)
        if raw_objects is None:
            raw_objects = getattr(entity, "_bodies", None)
        if raw_objects is None:
            return ()

        components: list[Any] = []
        for raw_object in raw_objects:
            component_candidates = list(getattr(raw_object, "components", ()))
            if hasattr(raw_object, "entity"):
                component_candidates.extend(
                    getattr(raw_object.entity, "components", ())
                )
            rigid_component = next(
                (
                    component
                    for component in component_candidates
                    if hasattr(component, "mass")
                    and (
                        hasattr(component, "get_collision_shapes")
                        or hasattr(component, "collision_shapes")
                    )
                ),
                None,
            )
            if rigid_component is not None:
                components.append(rigid_component)
        return tuple(components)

    def _scale_entity_mass(self, entity: Any, scales: Any) -> None:
        components = self._underlying_rigid_components(entity)
        if len(components) != self.num_envs:
            raise RuntimeError(
                f"Physical mass randomization could not resolve one rigid body "
                f"per environment for {getattr(entity, 'name', type(entity).__name__)!r}; "
                f"resolved {len(components)} for {self.num_envs}."
            )
        for env_index, component in enumerate(components):
            target_mass = float(component.mass) * float(scales[env_index])
            setter = getattr(component, "set_mass", None)
            if callable(setter):
                setter(target_mass)
            else:
                component.mass = target_mass

    def _scale_entity_friction(
        self,
        entity: Any,
        scales: Any,
        *,
        require_collision_shapes: bool = True,
    ) -> bool:
        if sapien is None:
            raise ImportError("SAPIEN is required for B05 physical randomization.")
        components = self._underlying_rigid_components(entity)
        if len(components) != self.num_envs:
            raise RuntimeError(
                f"Physical friction randomization could not resolve one rigid "
                f"body per environment for "
                f"{getattr(entity, 'name', type(entity).__name__)!r}; resolved "
                f"{len(components)} for {self.num_envs}."
            )

        material_applied = False
        for env_index, component in enumerate(components):
            scale = float(scales[env_index])
            try:
                material = sapien.physx.PhysxMaterial(
                    static_friction=B05_OBJECT_CONTACT_STATIC_FRICTION * scale,
                    dynamic_friction=B05_OBJECT_CONTACT_DYNAMIC_FRICTION * scale,
                    restitution=0.02,
                )
            except TypeError:
                material = sapien.physx.PhysxMaterial(
                    B05_OBJECT_CONTACT_STATIC_FRICTION * scale,
                    B05_OBJECT_CONTACT_DYNAMIC_FRICTION * scale,
                    0.02,
                )
            shape_getter = getattr(component, "get_collision_shapes", None)
            shapes = (
                tuple(shape_getter())
                if callable(shape_getter)
                else tuple(getattr(component, "collision_shapes", ()))
            )
            if not shapes:
                if require_collision_shapes:
                    raise RuntimeError(
                        f"Rigid body for "
                        f"{getattr(entity, 'name', type(entity).__name__)!r} "
                        "exposes no collision shapes for friction randomization."
                    )
                continue
            for shape in shapes:
                setter = getattr(shape, "set_physical_material", None)
                if callable(setter):
                    setter(material)
                else:
                    shape.physical_material = material
                material_applied = True
        return material_applied
