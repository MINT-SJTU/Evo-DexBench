from __future__ import annotations

from dataclasses import replace
import math

from dex_benchmark.assets import (
    evodex_asset_dir,
    partnet_mobility_model_urdf_path,
    partnet_mobility_model_textured_obj_path,
    partnet_mobility_modified_model_urdf_path,
    robocasa_texture_path,
    ycb_model_obj_path,
)
from dex_benchmark.sim.layout_types import LayoutId, SceneClass
from dex_benchmark.sim.scenes.layouts.common import (
    pose,
    static_box,
    table,
    textured_plane,
)
from dex_benchmark.sim.scenes.layouts.spec import (
    ArticulatedAssetSpec,
    ExternalSceneSpec,
    InteractionSiteSpec,
    LayoutBuildItemSpec,
    LayoutLightingSpec,
    LayoutSpec,
    PointLightSpec,
    PoseSpec,
    StaticBoxSpec,
    StaticMeshSpec,
    SpotLightSpec,
    TexturedPlaneSpec,
)


# 3 kitchen layouts building config。
_BUILD_CONFIG_ID: dict[LayoutId, int] = {
    LayoutId.KITCHEN_L1: 74,

    LayoutId.KITCHEN_L2: 36,

    LayoutId.KITCHEN_L3: 100,
}


# External scene walls. The tuple is (left wall, right wall, back wall and door).
_EXTERNAL_SCENE_WALLS: dict[LayoutId, tuple[bool, bool, bool]] = {
    LayoutId.KITCHEN_L1: (False, False, True ),

    LayoutId.KITCHEN_L2: (False, True , True ),

    LayoutId.KITCHEN_L3: (False, False, True ),
}


def _dishwasher_closed_joint_qpos(scale: float) -> dict[str, float]:
    """Return the fully retracted shelves and closed door for PartNet 12065."""

    return {
        # Prismatic joint positions scale with the complete dishwasher asset.
        "joint_0": scale * -0.308,
        "joint_1": scale * -0.136,
        # Revolute joint positions are invariant to uniform asset scaling.
        "joint_2": math.radians(-37.0),
    }


# Room size (x, y, z) in meters.
_ROOM_SIZE: dict[LayoutId, tuple[float, float, float]] = {
    LayoutId.KITCHEN_L1: (6.4, 5.0, 3.0),

    LayoutId.KITCHEN_L2: (5.2, 5.9, 3.0),

    LayoutId.KITCHEN_L3: (6.08, 6.0, 3.0),
}

# M02 wall-switch calibration. The common clearance moves L1/L2 farther back
# along the robot's facing direction. L3 uses the operator-accepted Human Viewer
# pose, including its small lateral offset and lower base height.
_KITCHEN_WALL_SWITCH_ROBOT_CLEARANCE_M = 0.92
_KITCHEN_L3_WALL_SWITCH_ROBOT_BASE_POSITION_XYZ = (
    2.479984,
    -0.919984,
    0.85,
)
_KITCHEN_WALL_SWITCH_ROBOT_BASE_YAW_RAD = 1.5708

# L1's accepted wall switch uses 0.5 joint friction and disables gravity on the
# moving rocker. Apply that same physical setup to L2/L3 for consistent actuation.
_KITCHEN_WALL_SWITCH_JOINT_FRICTION = 0.5

# M08 calibration contract. The accepted Kitchen L2 base pose is expressed in
# the coffee-machine frame so layouts with different world axes preserve the
# same robot-to-machine translation and orientation.
_COFFEE_MACHINE_ROBOT_BASE_LOCAL_XYZ = (-1.14, -0.07, -0.216)


# Invisible downward spot lights for local work areas.
_WORK_AREA_SPOT_CONFIG: dict[LayoutId, dict[str, tuple[tuple[float, float], float]]] = {
    LayoutId.KITCHEN_L1: {
        "counter": ((2.40, -0.55), 1.00),
        "fridge_cabinet": ((5.65, -2.25), 0.85),
    },

    LayoutId.KITCHEN_L2: {
        "counter": ((2.20, -0.75), 0.80),
        "sink_cabinet": ((0.45, -2.40), 0.90),
        "microwave_dishwasher": ((0.75, -4.00), 0.80),
    },

    LayoutId.KITCHEN_L3: {
        "counter_appliances": ((4.00, -0.55), 0.95),
        "dining_table": ((1.80, -4.00), 0.90),
        "sink_cabinet": ((0.55, -1.75), 0.90),
        "fridge": ((5.20, -1.80), 0.85),
    },
}


# Names of objects that need to be deleted. Objects that align to it will be deleted.
# RoboCasa only
_REMOVE_FIXTURE_TUPLE: dict[LayoutId, tuple[str, ...]] = {
    LayoutId.KITCHEN_L1: ("fridge_right_group",
                          "toaster_right_group",
                          "coffee_machine_right_group",
                          "light_switch_room",
                          "light_switch_2_room",
                          "utensil_holder_main_group",
                          "knife_block_main_group",
                          "utensil_rack_room",
                          "plant_island_group",
                          "paper_towel_main_group",
                          "stool_1_island_group",
                          "stool_2_island_group",
                          "stool_3_island_group",
                          "window_group_main_group",),

    LayoutId.KITCHEN_L2: ("fridge_main_group",
                          "light_switch_room",
                          "light_switch_2_room",
                          "paper_towel_left_group",
                          "toaster_main_group",
                          "coffee_machine_left_group",
                          "knife_block_left_group",
                          "utensil_holder_main_group",
                          "plant_left_group",),

    LayoutId.KITCHEN_L3: ("fridge_right_group",
                          "dishwasher_left_group",
                          "coffee_machine_main_group",
                          "plant_left_group",
                          "paper_towel_left_group",
                          "toaster_main_group",
                          "utensil_holder_main_group",
                          "utensil_rack_room",
                          "light_switch_room",
                          "light_switch_2_room",),
}


# Extra objects
# RoboCasa only
_EXTRA_PARTNET_OBJECTS: dict[LayoutId, tuple[ArticulatedAssetSpec, ...]] = {
    LayoutId.KITCHEN_L1: (
        ArticulatedAssetSpec(
            name="switch",
            asset_path=str(partnet_mobility_model_urdf_path("100906")),
            pose=PoseSpec(
                position_xyz=(
                    2.4,
                    0,
                    1.2,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.1,
            fix_root_link=True,
            disable_self_collisions=True,
            joint_friction={
                "joint_0": _KITCHEN_WALL_SWITCH_JOINT_FRICTION,
            },
            link_disable_gravity=("link_0",),
        ),
        ArticulatedAssetSpec(
            name="dishwasher",
            asset_path=str(partnet_mobility_model_urdf_path("12065")),
            pose=PoseSpec(
                position_xyz=(
                    0.58,
                    -4.00,
                    0.63,
                ),
                quaternion_wxyz=(0.0, 0.0, 0.0, 1.0),
            ),
            fix_root_link=True,
            disable_self_collisions=True,
            joint_initial_qpos=_dishwasher_closed_joint_qpos(scale=1.0),
        ),
        ArticulatedAssetSpec(
            name="coffee_machine",
            asset_path=str(partnet_mobility_model_urdf_path("103041")),
            pose=PoseSpec(
                position_xyz=(
                    1.50,
                    -0.40,
                    1.25,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.4,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="bottle_blue",
            asset_path=str(partnet_mobility_model_urdf_path("3574")),
            pose=PoseSpec(
                position_xyz=(
                    3.70,
                    -3.0,
                    1.09,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=0.2,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="bottle_yellow",
            asset_path=str(partnet_mobility_model_urdf_path("4216")),
            pose=PoseSpec(
                position_xyz=(
                    3.20,
                    -3.20,
                    1.09,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=0.12,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="refrigerator",
            asset_path=str(partnet_mobility_modified_model_urdf_path("10143")),
            pose=PoseSpec(
                position_xyz=(
                    5.80,
                    -3.30,
                    1.06,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=1.2,
            fix_root_link=True,
            disable_self_collisions=True,
            initial_qpos=(0.0, 0.0),
        ),
        ArticulatedAssetSpec(
            name="trashcan",
            asset_path=str(partnet_mobility_model_urdf_path("12445")),
            pose=PoseSpec(
                position_xyz=(
                    6.00,
                    -4.30,
                    0.435,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=0.5,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="toaster",
            asset_path=str(partnet_mobility_model_urdf_path("103477")),
            pose=PoseSpec(
                position_xyz=(
                    2.70,
                    -0.30,
                    1.02,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.15,
            fix_root_link=True,
            disable_self_collisions=True,
            link_disable_gravity=("link_1",)
        ),
        StaticMeshSpec(
            name="pot",
            mesh_path=str(partnet_mobility_model_textured_obj_path("100038", "original-8.obj")),
            pose=PoseSpec(
                position_xyz=(
                    3.535,
                    -0.357,
                    0.994,
                ),
                quaternion_wxyz=(0.7071, 0.7071, 0, 0.0),
            ),
            scale_xyz=(0.15, 0.15, 0.15),
            collision_type="convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="pot_lid",
            mesh_path=str(partnet_mobility_modified_model_urdf_path("100038", "pot_lid.obj")),
            pose=PoseSpec(
                position_xyz=(
                    3.535,
                    -0.357,
                    1.2,
                ),
                quaternion_wxyz=(0.7071, 0.7071, 0, 0.0),
            ),
            scale_xyz=(0.15, 0.15, 0.15),
            collision_type="nonconvex",
            body_type="static",
        ),
        ArticulatedAssetSpec(
            name="cup1",
            asset_path=str(partnet_mobility_model_urdf_path("8736")),
            pose=PoseSpec(
                position_xyz=(
                    6.20,
                    -1.0,
                    1.6,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=False,
        ),
        ArticulatedAssetSpec(
            name="cup2",
            asset_path=str(partnet_mobility_model_urdf_path("3558")),
            pose=PoseSpec(
                position_xyz=(
                    6.1,
                    -1.55,
                    1.6,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=False,
        ),
        StaticMeshSpec(
            name="spoon",
            mesh_path=str(ycb_model_obj_path("031_spoon")),
            collision_mesh_path=str(
                ycb_model_obj_path("031_spoon", "textured_vhacd.obj")
            ),
            pose=PoseSpec(
                position_xyz=(
                    4.00,
                    -3.00,
                    0.94,
                ),
                quaternion_wxyz=(0.0, 0.0, 1.0, 0.0),
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="plate",
            mesh_path=str(ycb_model_obj_path("029_plate")),
            collision_mesh_path=str(
                ycb_model_obj_path("029_plate", "textured_vhacd.obj")
            ),
            pose=PoseSpec(
                position_xyz=(
                    3.45,
                    -2.20,
                    0.94,
                ),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="sponge",
            mesh_path=str(ycb_model_obj_path("026_sponge")),
            collision_mesh_path=str(
                ycb_model_obj_path("026_sponge", "textured_vhacd.obj")
            ),
            pose=PoseSpec(
                position_xyz=(
                    3.30,
                    -2.40,
                    0.94,
                ),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        )
    ),

    LayoutId.KITCHEN_L2: (
        ArticulatedAssetSpec(
            name="switch",
            asset_path=str(partnet_mobility_model_urdf_path("100906")),
            pose=PoseSpec(
                position_xyz=(
                    3.3,
                    0,
                    1.2,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.1,
            fix_root_link=True,
            disable_self_collisions=True,
            joint_friction={
                "joint_0": _KITCHEN_WALL_SWITCH_JOINT_FRICTION,
            },
            link_disable_gravity=("link_0",),
        ),
        ArticulatedAssetSpec(
            name="dishwasher",
            asset_path=str(partnet_mobility_model_urdf_path("12065")),
            pose=PoseSpec(
                position_xyz=(
                    0.58,
                    -5.00,
                    0.63,
                ),
                quaternion_wxyz=(0.0, 0.0, 0.0, 1.0),
            ),
            fix_root_link=True,
            disable_self_collisions=True,
            joint_initial_qpos=_dishwasher_closed_joint_qpos(scale=1.0),
        ),
        ArticulatedAssetSpec(
            name="coffee_machine",
            asset_path=str(partnet_mobility_model_urdf_path("103041")),
            pose=PoseSpec(
                position_xyz=(
                    0.215,
                    -3,
                    1.25,
                ),
                quaternion_wxyz=(0, 0, 0, 1),
            ),
            scale=0.4,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="bottle_blue",
            asset_path=str(partnet_mobility_model_urdf_path("3574")),
            pose=PoseSpec(
                position_xyz=(
                    3.00,
                    -3.20,
                    1.09,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=0.2,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="bottle_yellow",
            asset_path=str(partnet_mobility_model_urdf_path("4216")),
            pose=PoseSpec(
                position_xyz=(
                    2.5,
                    -3.25,
                    1.09,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=0.12,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="refrigerator",
            asset_path=str(partnet_mobility_model_urdf_path("10489")),
            pose=PoseSpec(
                position_xyz=(
                    4.2,
                    -0.380,
                    1.06,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=1.2,
            fix_root_link=True,
            disable_self_collisions=True,
            initial_qpos=(0.0, 0.0),
        ),
        ArticulatedAssetSpec(
            name="trashcan",
            asset_path=str(partnet_mobility_model_urdf_path("12445")),
            pose=PoseSpec(
                position_xyz=(
                    3.5,
                    -4.70,
                    0.435,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.5,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="toaster",
            asset_path=str(partnet_mobility_model_urdf_path("103477")),
            pose=PoseSpec(
                position_xyz=(
                    1.4,
                    -0.201,
                    1.02,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.15,
            fix_root_link=True,
            disable_self_collisions=True,
            link_disable_gravity=("link_1",)
        ),
        ArticulatedAssetSpec(
            name="pot",
            asset_path=str(partnet_mobility_model_urdf_path("100021")),
            pose=PoseSpec(
                position_xyz=(
                    1.96,
                    -0.38,
                    1.03,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="cup1",
            asset_path=str(partnet_mobility_model_urdf_path("4500")),
            pose=PoseSpec(
                position_xyz=(
                    1.0,
                    -0.25,
                    1.58,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=False,
        ),
        ArticulatedAssetSpec(
            name="cup2",
            asset_path=str(partnet_mobility_model_urdf_path("3614")),
            pose=PoseSpec(
                position_xyz=(
                    0.9,
                    -0.3,
                    1.55,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=False,
        ),
        StaticMeshSpec(
            name="spoon",
            mesh_path=str(ycb_model_obj_path("031_spoon")),
            pose=PoseSpec(
                position_xyz=(
                    2.5,
                    -2.70,
                    0.94,
                ),
                quaternion_wxyz=(0.0, 0.0, 1.0, 0.0),
            ),
            collision_mesh_path=str(
                ycb_model_obj_path("031_spoon", "textured_vhacd.obj")
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="plate",
            mesh_path=str(ycb_model_obj_path("029_plate")),
            pose=PoseSpec(
                position_xyz=(
                    0.3,
                    -2.40,
                    0.94,
                ),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_mesh_path=str(
                ycb_model_obj_path("029_plate", "textured_vhacd.obj")
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="sponge",
            mesh_path=str(ycb_model_obj_path("026_sponge")),
            pose=PoseSpec(
                position_xyz=(
                    0.5,
                    -2.40,
                    0.94,
                ),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_mesh_path=str(
                ycb_model_obj_path("026_sponge", "textured_vhacd.obj")
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
    ),

    LayoutId.KITCHEN_L3: (
        ArticulatedAssetSpec(
            name="switch",
            asset_path=str(partnet_mobility_model_urdf_path("100906")),
            pose=PoseSpec(
                position_xyz=(
                    2.45,
                    0,
                    1.2,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.1,
            fix_root_link=True,
            disable_self_collisions=True,
            joint_friction={
                "joint_0": _KITCHEN_WALL_SWITCH_JOINT_FRICTION,
            },
            link_disable_gravity=("link_0",),
        ),
        ArticulatedAssetSpec(
            name="dishwasher",
            asset_path=str(partnet_mobility_model_urdf_path("12065")),
            pose=PoseSpec(
                position_xyz=(
                    0.35,
                    -1.2,
                    0.45,
                ),
                quaternion_wxyz=(0, 0, 0, 1),
            ),
            scale=0.75,
            fix_root_link=True,
            disable_self_collisions=True,
            joint_initial_qpos=_dishwasher_closed_joint_qpos(scale=0.75),
        ),
        ArticulatedAssetSpec(
            name="coffee_machine",
            asset_path=str(partnet_mobility_model_urdf_path("103041")),
            pose=PoseSpec(
                position_xyz=(
                    0.330003,
                    -3.186997,
                    1.25,
                ),
                # The viewer reference has about -178.1 degrees of yaw plus
                # negligible roll/pitch. Use an exact planar 180-degree yaw.
                quaternion_wxyz=(0.0, 0.0, 0.0, 1.0),
            ),
            scale=0.4,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="bottle",
            asset_path=str(partnet_mobility_model_urdf_path("3678")),
            pose=PoseSpec(
                position_xyz=(
                    0.277,
                    -3.7,
                    1.045,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="bottle_yellow",
            asset_path=str(partnet_mobility_model_urdf_path("4216")),
            pose=PoseSpec(
                position_xyz=(
                    1.85,
                    -4.0,
                    1.09,
                ),
                quaternion_wxyz=(1.0, 0, 0, 0),
            ),
            scale=0.12,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="microwave",
            asset_path=str(partnet_mobility_model_urdf_path("7167")),
            pose=PoseSpec(
                position_xyz=(
                    0.9,
                    -0.35,
                    1.16,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.5,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="pot",
            asset_path=str(partnet_mobility_model_urdf_path("100021")),
            pose=PoseSpec(
                position_xyz=(
                    3.96,
                    -0.4,
                    1.06,
                ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=0.16,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="refrigerator",
            asset_path=str(partnet_mobility_model_urdf_path("12050")),
            pose=PoseSpec(
                position_xyz=(
                    5.6,
                    -1.8,
                    1.2,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=1.3,
            fix_root_link=True,
            disable_self_collisions=True,
            initial_qpos=(0.0, 0.0),
        ),
        ArticulatedAssetSpec(
            name="bottle_fridge",
            asset_path=str(partnet_mobility_model_urdf_path("4200")),
            pose=PoseSpec(
                position_xyz=(
                    5.3,
                    -1.8,
                    1.78,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="trashcan",
            asset_path=str(partnet_mobility_model_urdf_path("12445")),
            pose=PoseSpec(
                position_xyz=(
                    5.6,
                    -2.6,
                    0.435,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.5,
            fix_root_link=True,
            disable_self_collisions=True,
        ),
        ArticulatedAssetSpec(
            name="toaster",
            asset_path=str(partnet_mobility_model_urdf_path("103477")),
            pose=PoseSpec(
                position_xyz=(
                    3.6,
                    -0.201,
                    1.04,
                ),
                quaternion_wxyz=(0.701, 0, 0, 0.701),
            ),
            scale=0.22,
            fix_root_link=True,
            disable_self_collisions=True,
            link_disable_gravity=("link_1",)
        ),
        ArticulatedAssetSpec(
            name="cup1",
            asset_path=str(partnet_mobility_model_urdf_path("5601")),
            pose=PoseSpec(
                position_xyz=(
                    0.3,
                    -1.55,
                    1.58,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=False,
        ),
        ArticulatedAssetSpec(
            name="cup2",
            asset_path=str(partnet_mobility_model_urdf_path("3614")),
            pose=PoseSpec(
                position_xyz=(
                    0.3,
                    -1.4,
                    1.55,
                ),
                quaternion_wxyz=(1, 0, 0, 0),
            ),
            scale=0.15,
            fix_root_link=False,
            disable_self_collisions=False,
        ),
        StaticMeshSpec(
            name="spoon",
            mesh_path=str(ycb_model_obj_path("031_spoon")),
            pose=PoseSpec(
                position_xyz=(
                    2.5,
                    -4.20,
                    0.94,
                ),
                quaternion_wxyz=(0.0, 0.0, 1.0, 0.0),
            ),
            collision_mesh_path=str(
                ycb_model_obj_path("031_spoon", "textured_vhacd.obj")
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="plate",
            mesh_path=str(ycb_model_obj_path("029_plate")),
            pose=PoseSpec(
                position_xyz=(
                    0.35,
                    -2.00,
                    0.94,
                ),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_mesh_path=str(
                ycb_model_obj_path("029_plate", "textured_vhacd.obj")
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
        StaticMeshSpec(
            name="sponge",
            mesh_path=str(ycb_model_obj_path("026_sponge")),
            pose=PoseSpec(
                position_xyz=(
                    0.4,
                    -1.70,
                    0.94,
                ),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            collision_mesh_path=str(
                ycb_model_obj_path("026_sponge", "textured_vhacd.obj")
            ),
            scale_xyz=(1.0, 1.0, 1.0),
            collision_type="multiple_convex",
            body_type="dynamic",
        ),
    ),
}


# Repeat UV for wall, ceiling. The tuple is (wall, ceiling).
# RoboCasa only
_REPEAT_UV: dict[LayoutId, tuple[tuple[float, float], tuple[float, float]]] = {
    LayoutId.KITCHEN_L1: ((0.75, 0.75), (0.8, 0.8)),

    LayoutId.KITCHEN_L2: ((1.6, 1.6), (0.8, 0.8)),

    LayoutId.KITCHEN_L3: ((0.375, 0.375), (0.8, 0.8)),
}


# Texture for wall, ceiling. The tuple is (wall, ceiling).
# RoboCasa only
_TEXTURE: dict[LayoutId, tuple[str, str]] = {
    LayoutId.KITCHEN_L1: ("tiles/white_square_tiles.png",
                          "prev/white-plaster.png"),

    LayoutId.KITCHEN_L2: ("bricks/red_bricks.png",
                          "prev/white-plaster.png"),

    LayoutId.KITCHEN_L3: ("tiles/black_marble_tiles.png",
                          "prev/white-plaster.png"),
}


# If True, build invisible, non-collidable semantic location markers.
_BUILD_SEMANTIC_MARKER = False


def _external_scene(layout_id: LayoutId) -> ExternalSceneSpec | None:
    return ExternalSceneSpec(
        name=f"robocasa_{layout_id.value}",
        backend="robocasa",  # type: ignore[arg-type]
        build_config_idx=_BUILD_CONFIG_ID[layout_id],
        init_robot_base_pos=None,
        remove_fixture_names=_REMOVE_FIXTURE_TUPLE[layout_id],
    )


def _semantic_marker(name: str, marker_pose: PoseSpec) -> StaticBoxSpec:
    return StaticBoxSpec(
        name=name,
        half_size_xyz=(0.01, 0.01, 0.005),
        pose=marker_pose,
        color_rgba=(0.0, 0.0, 0.0, 0.0),
        add_collision=False,
        body_type="kinematic",
    )


def _kitchen_room_shell_items(
    layout_id: LayoutId,
    left: bool, right: bool, back: bool
) -> list[TexturedPlaneSpec | ArticulatedAssetSpec]:
    room_min_x = 0
    room_max_x = _ROOM_SIZE[layout_id][0]
    room_min_y = - _ROOM_SIZE[layout_id][1]
    room_max_y = 0
    room_min_z = 0
    room_max_z = _ROOM_SIZE[layout_id][2]
    wall_center_z = _ROOM_SIZE[layout_id][2] / 2.0

    door_scale = 1.14
    door_center_x = -1.35 + _ROOM_SIZE[layout_id][0] / 2.0
    door_width = 0.6582 * door_scale
    door_height = 1.7551 * door_scale
    door_min_x = door_center_x - door_width / 2.0
    door_max_x = door_center_x + door_width / 2.0

    segments = []

    segments.append(textured_plane(
            "ceiling_texture",
            robocasa_texture_path(_TEXTURE[layout_id][1]),
            (_ROOM_SIZE[layout_id][0] / 2.0, _ROOM_SIZE[layout_id][1] / 2.0),
            PoseSpec(
                position_xyz=(
                    (room_min_x + room_max_x) / 2.0,
                    (room_min_y + room_max_y) / 2.0,
                    room_max_z
                )
            ),
            _REPEAT_UV[layout_id][1],
        ))

    if left:
        segments.append(textured_plane(
            name="wall_left_full",
            texture_path=robocasa_texture_path(_TEXTURE[layout_id][0]),
            half_size_xy=(_ROOM_SIZE[layout_id][1] / 2.0, _ROOM_SIZE[layout_id][2] / 2.0),
            pose=PoseSpec(
                position_xyz=(
                    _ROOM_SIZE[layout_id][0] / 2.0,
                    - _ROOM_SIZE[layout_id][1] / 2.0,
                    wall_center_z
                ),
                quaternion_wxyz=(0.5, 0.5, -0.5, -0.5),
            ),
            texture_repeat_uv=_REPEAT_UV[layout_id][0],
            collision_half_size_xyz=(
                _ROOM_SIZE[layout_id][1] / 2.0,
                _ROOM_SIZE[layout_id][2] / 2.0,
                0.04,
            ),
        ))

    if right:
        segments.append(textured_plane(
            name="wall_right_full",
            texture_path=robocasa_texture_path(_TEXTURE[layout_id][0]),
            half_size_xy=(_ROOM_SIZE[layout_id][1] / 2.0, _ROOM_SIZE[layout_id][2] / 2.0),
            pose=PoseSpec(
                position_xyz=(
                    room_max_x,
                    - _ROOM_SIZE[layout_id][1] / 2.0,
                    wall_center_z
                ),
                quaternion_wxyz=(-0.5, 0.5, -0.5, 0.5),
            ),
            texture_repeat_uv=_REPEAT_UV[layout_id][0],
            collision_half_size_xyz=(
                _ROOM_SIZE[layout_id][1] / 2.0,
                _ROOM_SIZE[layout_id][2] / 2.0,
                0.04,
            ),
        ))

    if back:
        segments.append(textured_plane(
            name="wall_back_door_left",
            texture_path=robocasa_texture_path(_TEXTURE[layout_id][0]),
            half_size_xy=((door_min_x - room_min_x) / 2.0, _ROOM_SIZE[layout_id][2] / 2.0),
            pose=PoseSpec(
                position_xyz=(
                    (room_min_x + door_min_x) / 2.0,
                    room_min_y,
                    wall_center_z,
                ),
                quaternion_wxyz=(0, 0, 0.7071, -0.7071),
            ),
            texture_repeat_uv=_REPEAT_UV[layout_id][0],
            collision_half_size_xyz=(
                (door_min_x - room_min_x) / 2.0,
                _ROOM_SIZE[layout_id][2] / 2.0,
                0.04,
            ),
        ))
        segments.append(textured_plane(
            name="wall_back_door_right",
            texture_path=robocasa_texture_path(_TEXTURE[layout_id][0]),
            half_size_xy=((room_max_x - door_max_x) / 2.0, _ROOM_SIZE[layout_id][2] / 2.0),
            pose=PoseSpec(
                position_xyz=(
                    (room_max_x + door_max_x) / 2.0,
                    room_min_y,
                    wall_center_z,
                ),
                quaternion_wxyz=(0, 0, 0.7071, -0.7071),
            ),
            texture_repeat_uv=_REPEAT_UV[layout_id][0],
            collision_half_size_xyz=(
                (room_max_x - door_max_x) / 2.0,
                _ROOM_SIZE[layout_id][2] / 2.0,
                0.04,
            ),
        ))
        segments.append(textured_plane(
            name="wall_back_door_header",
            texture_path=robocasa_texture_path(_TEXTURE[layout_id][0]),
            half_size_xy=(door_width / 2.0, (_ROOM_SIZE[layout_id][2] - door_height) / 2.0),
            pose=PoseSpec(
                position_xyz=(
                    door_center_x,
                    room_min_y,
                    (door_height + _ROOM_SIZE[layout_id][2]) / 2.0,
                ),
                quaternion_wxyz=(0, 0, 0.7071, -0.7071),
            ),
            texture_repeat_uv=_REPEAT_UV[layout_id][0],
            collision_half_size_xyz=(
                door_width / 2.0,
                (_ROOM_SIZE[layout_id][2] - door_height) / 2.0,
                0.04,
            ),
        ))
        segments.append(ArticulatedAssetSpec(
            name="door_panel",
            asset_path=partnet_mobility_model_urdf_path("8867"),
            pose=PoseSpec(
                position_xyz=(
                        door_center_x + 0.065,
                        room_min_y - 0.15,
                        door_height / 2.0 + 0.0675,
                    ),
                quaternion_wxyz=(0.7071, 0, 0, 0.7071),
            ),
            scale=door_scale,
            fix_root_link=True,
            load_multiple_collisions_from_file=True,
            disable_self_collisions=True,
            initial_qpos=(-0.8168140899333464, -0.7853981633974483),
        ))

    return segments


def _scale_light_color(
    color_rgb: tuple[float, float, float],
    intensity_scale: float,
) -> tuple[float, float, float]:
    red, green, blue = color_rgb
    return (
        red * intensity_scale,
        green * intensity_scale,
        blue * intensity_scale,
    )


def _kitchen_ceiling_light_item(layout_id: LayoutId) -> StaticMeshSpec:
    root_position_xyz = (
        _ROOM_SIZE[layout_id][0] / 2.0,
        - _ROOM_SIZE[layout_id][1] / 2.0,
        _ROOM_SIZE[layout_id][2] - 0.2 - 1.1726436614990234 * 0.65,
    )
    return StaticMeshSpec(
        name="ceiling_light",
        mesh_path=evodex_asset_dir() / "polyhaven/modern_ceiling_lamp_01/modern_ceiling_lamp_01_1k.glb",
        pose=PoseSpec(
            position_xyz=root_position_xyz,
            quaternion_wxyz=(0.7071, 0.7071, 0.0, 0.0),
        ),
        scale_xyz=(0.65, 0.65, 0.65),
        collision_type="none",
        body_type="static",
    )


def _kitchen_lighting_spec(layout_id: LayoutId) -> LayoutLightingSpec:
    emission_position_xyz = (
        _ROOM_SIZE[layout_id][0] / 2.0,
        - _ROOM_SIZE[layout_id][1] / 2.0,
        _ROOM_SIZE[layout_id][2] - 0.2 - 0.65,
    )

    ceiling_light_color_rgb = (4.0, 4.0, 3.9)
    ceiling_light_point_intensity_scale = 0.40
    ceiling_light_spot_inner_fov_rad = math.radians(130.0)
    ceiling_light_spot_outer_fov_rad = math.radians(150.0)

    point_light = PointLightSpec(
        position_xyz=emission_position_xyz,
        color_rgb=_scale_light_color(
            ceiling_light_color_rgb,
            ceiling_light_point_intensity_scale,
        ),
        shadow=False,
        shadow_near=0.1,
        shadow_far=8.0,
        shadow_map_size=2048,
    )

    spot_lights = [SpotLightSpec(
        position_xyz=emission_position_xyz,
        direction_xyz=(0.0, 0.0, -1.0),
        inner_fov_rad=ceiling_light_spot_inner_fov_rad,
        outer_fov_rad=ceiling_light_spot_outer_fov_rad,
        color_rgb=_scale_light_color(
            ceiling_light_color_rgb,
            ceiling_light_point_intensity_scale,
        ),
        shadow=True,
        shadow_near=0.4,
        shadow_far=6.0,
        shadow_map_size=4096,
    )]

    source_z = (
            _ROOM_SIZE[layout_id][2]
            - 0.01
        )
    spot_lights += [SpotLightSpec(
                position_xyz=(center_xy[0], center_xy[1], source_z),
                direction_xyz=(0.0, 0.0, -1.0),
                inner_fov_rad=math.radians(52.0),
                outer_fov_rad=math.radians(72.0),
                color_rgb=_scale_light_color(
                    (1.60, 1.65, 1.65),
                    intensity_scale,
                ),
                shadow=False,
                shadow_near=0.20,
                shadow_far=4.0,
                shadow_map_size=1024,
            )
            for center_xy, intensity_scale in _WORK_AREA_SPOT_CONFIG[
                layout_id
            ].values()]

    spot_lights = tuple(spot_lights)

    return LayoutLightingSpec(
        ambient_rgb=(0.28, 0.28, 0.27),
        point_lights=(point_light,),
        spot_lights=spot_lights,
    )


def _kitchen_layout(
    layout_id: LayoutId,
    site_configs: dict[str, InteractionSiteSpec],
) -> LayoutSpec:
    external_scene = _external_scene(layout_id)
    add_default_ground = True

    build_items: list[LayoutBuildItemSpec] = []

    if external_scene is not None:
        add_default_ground = False
        build_items.append(external_scene)

        left, right, back = _EXTERNAL_SCENE_WALLS.get(layout_id, (True, True, True))
        build_items += _kitchen_room_shell_items(
            layout_id,
            left=left,
            right=right,
            back=back,
        )

        build_items.append(_kitchen_ceiling_light_item(layout_id))

        for obj_cfg in _EXTRA_PARTNET_OBJECTS.get(layout_id, ()):
            build_items.append(obj_cfg)

    if _BUILD_SEMANTIC_MARKER:
        for area_name, cfg in site_configs.items():
            build_items.append(
                _semantic_marker(
                    area_name,
                    cfg.frame_pose,
                )
            )

    lighting = _kitchen_lighting_spec(layout_id)

    return LayoutSpec(
        layout_id=layout_id,
        scene_class=SceneClass.KITCHEN,
        description=f"Kitchen layout {layout_id.value[-1]}.",
        build_items=build_items,
        interaction_sites=site_configs,
        lighting=lighting,
        add_default_ground=add_default_ground,
    )


def get_site_from_poses(
        site_id: str,
        work_area_pose: PoseSpec,
        robot_base_pose: PoseSpec,
        camera_rig_id: str | None = None,
        notes: str = ""
) -> InteractionSiteSpec:
    if camera_rig_id is None:
        camera_rig_id = f"kitchen_{site_id}"
    return InteractionSiteSpec(
        site_id=site_id,
        frame_pose=work_area_pose,
        robot_base_center_pose=robot_base_pose,
        camera_rig_id=camera_rig_id,
        notes=notes
    )


def get_site_from_poses_with_fixture(
        site_id: str,
        work_area_pose: PoseSpec,
        robot_base_pose: PoseSpec,
        fixture_name: str | None = None,
        fixture_source: str = "layout_entity",
        camera_rig_id: str | None = None,
        notes: str = ""
) -> InteractionSiteSpec:
    if camera_rig_id is None:
        camera_rig_id = f"kitchen_{site_id}"
    if fixture_name is None:
        fixture_name = site_id
    return InteractionSiteSpec(
        site_id=site_id,
        frame_pose=work_area_pose,
        robot_base_center_pose=robot_base_pose,
        camera_rig_id=camera_rig_id,
        target_entity_name=fixture_name,
        target_entity_source=fixture_source,
        notes=notes
    )


def get_site_from_fixture(
        site_id: str,
        layout_id: LayoutId,
        fixture_name: str | None = None,
        fixture_source: str = "layout_entity",
        clearance_m: float = 1.0,
        robot_base_z: float = 0.924,
        camera_rig_id: str | None = None,
        notes: str = ""
) -> InteractionSiteSpec:
    if camera_rig_id is None:
        camera_rig_id = f"kitchen_{site_id}"
    if fixture_name is None:
        fixture_name = site_id

    matching_items = []

    if fixture_source == "external_fixture":
        raise NotImplementedError("External fixture support is not implemented yet, please use get_site_from_poses_with_fixture instead.")

    elif fixture_source == "layout_entity":
        matching_items = [
            item
            for item in _EXTRA_PARTNET_OBJECTS[layout_id]
            if isinstance(item, (ArticulatedAssetSpec, StaticMeshSpec)) and item.name == fixture_name
        ]

    else:
        raise ValueError(f"Unknown fixture_source {fixture_source!r}.")

    if len(matching_items) > 1:
        raise ValueError(
            f"Expected exactly one articulated kitchen item {fixture_name!r}, got "
            f"{len(matching_items)}."
        )
    if len(matching_items) == 0:
        raise ValueError(
            f"Expected exactly one articulated kitchen item {fixture_name!r}, got "
            f"{len(matching_items)}."
        )

    item = matching_items[0]

    robot_base_xyz = item.pose.transform_point((-clearance_m, 0.0, 0.0))
    robot_base_pose = PoseSpec(
        position_xyz=(
            float(robot_base_xyz[0]),
            float(robot_base_xyz[1]),
            robot_base_z,
        ),
        quaternion_wxyz=item.pose.quaternion_wxyz,
    )

    return InteractionSiteSpec(
        site_id=site_id,
        frame_pose=item.pose,
        robot_base_center_pose=robot_base_pose,
        camera_rig_id=camera_rig_id,
        target_entity_name=fixture_name,
        target_entity_source=fixture_source,
        notes=notes
    )


def _coffee_machine_site(layout_id: LayoutId) -> InteractionSiteSpec:
    """Build the M08 site from one accepted machine-relative base pose."""

    matching_items = tuple(
        item
        for item in _EXTRA_PARTNET_OBJECTS[layout_id]
        if isinstance(item, ArticulatedAssetSpec)
        and item.name == "coffee_machine"
    )
    if len(matching_items) != 1:
        raise ValueError(
            f"Expected one coffee machine in {layout_id.value}, got "
            f"{len(matching_items)}."
        )
    coffee_machine = matching_items[0]
    robot_base_xyz = coffee_machine.pose.transform_point(
        _COFFEE_MACHINE_ROBOT_BASE_LOCAL_XYZ
    )
    return InteractionSiteSpec(
        site_id="coffee_machine",
        frame_pose=coffee_machine.pose,
        robot_base_center_pose=PoseSpec(
            position_xyz=tuple(float(value) for value in robot_base_xyz),
            quaternion_wxyz=coffee_machine.pose.quaternion_wxyz,
        ),
        camera_rig_id="kitchen_coffee_machine",
        target_entity_name="coffee_machine",
        target_entity_source="layout_entity",
        notes=(
            "M08 base keeps the accepted Kitchen L2 relative pose in every "
            "Kitchen layout."
        ),
    )


KITCHEN_LAYOUTS: dict[LayoutId, LayoutSpec] = {
    LayoutId.KITCHEN_L1: _kitchen_layout(
        LayoutId.KITCHEN_L1,
        site_configs={
            "counter_top": get_site_from_poses(
                site_id="counter_top",
                work_area_pose=pose(
                    2.4,
                    -0.40,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    2.4,
                    -1.10,
                    0.924,
                    1.57079632679,
                ),
                camera_rig_id="kitchen_counter",
                notes="Shared work surface for non-device-specific countertop tasks."
            ),

            "dining_table": get_site_from_poses(
                site_id="dining_table",
                work_area_pose=pose(
                    3.3,
                    -3.0,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    3.3,
                    -3.7,
                    0.924,
                    1.57079632679,
                ),
                notes="Shared work surface for dining-table manipulation tasks."
            ),

            "sink": get_site_from_poses_with_fixture(
                site_id="sink",
                work_area_pose=pose(
                    3.3,
                    -2.30,
                    0.924,
                    -1.57079632679,
                ),
                robot_base_pose=pose(
                    3.3,
                    -1.60,
                    0.924,
                    -1.57079632679,
                ),
                fixture_name="sink_island_group",
                fixture_source="external_fixture",
            ),

            "refrigerator": get_site_from_poses_with_fixture(
                site_id="refrigerator",
                work_area_pose=pose(
                    5.95,
                    -3.30,
                    1.06,
                    0.0,
                ),
                robot_base_pose=pose(
                    4.95,
                    -3.30,
                    1.06,
                    0.0,
                ),
                fixture_name="refrigerator",
                fixture_source="layout_entity",
            ),

            "cabinet": get_site_from_poses_with_fixture(
                site_id="cabinet",
                work_area_pose=pose(
                    6.20,
                    -1.20,
                    1.405,
                    0.0,
                ),
                robot_base_pose=pose(
                    5.0,
                    -1.20,
                    1.0,
                    0.0,
                ),
                fixture_name="cab_1_right_group",
                fixture_source="external_fixture",
            ),

            "wall_switch": get_site_from_fixture(
                site_id="wall_switch",
                layout_id=LayoutId.KITCHEN_L1,
                fixture_name="switch",
                clearance_m=_KITCHEN_WALL_SWITCH_ROBOT_CLEARANCE_M,
            ),

            "coffee_machine": _coffee_machine_site(LayoutId.KITCHEN_L1),

            "toaster": get_site_from_fixture(
                site_id="toaster",
                layout_id=LayoutId.KITCHEN_L1,
                clearance_m=0.85,
            ),

            "dishwasher": get_site_from_fixture(
                site_id="dishwasher",
                layout_id=LayoutId.KITCHEN_L1,
                clearance_m=1.0,
            ),

            "microwave": get_site_from_poses_with_fixture(
                site_id="microwave",
                work_area_pose=pose(
                    0.374704,
                    -3.099999,
                    1.608125,
                    1.5708
                ),
                robot_base_pose=pose(
                    1.4,
                    -3.1,
                    1.0,
                    3.14159265359
                ),
                fixture_name="microwave_left_group",
                fixture_source="external_fixture",
            ),

            "stove": get_site_from_poses(
                site_id="stove",
                work_area_pose=pose(
                    3.836,
                    -0.363,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    3.836,
                    -1.063,
                    0.924,
                    1.57079632679,
                ),
            ),

        },
    ),

    LayoutId.KITCHEN_L2: _kitchen_layout(
        LayoutId.KITCHEN_L2,
        site_configs={
            "counter_top": get_site_from_poses(
                site_id="counter_top",
                work_area_pose=pose(
                    3.15,
                    -0.30,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    3.15,
                    -1.0,
                    0.924,
                    1.57079632679,
                ),
                camera_rig_id="kitchen_counter",
                notes="Shared work surface for non-device-specific countertop tasks."
            ),

            "dining_table": get_site_from_poses(
                site_id="dining_table",
                work_area_pose=pose(
                    2.5,
                    -3.3,
                    0.924,
                    0.0,
                ),
                robot_base_pose=pose(
                    1.8,
                    -3.3,
                    0.924,
                    0.0,
                ),
                notes="Shared work surface for dining-table manipulation tasks."
            ),

            "sink": get_site_from_poses_with_fixture(
                site_id="sink",
                work_area_pose=pose(
                    0.3,
                    -2.20,
                    0.924,
                    3.1415926,
                ),
                robot_base_pose=pose(
                    1.0,
                    -2.20,
                    0.924,
                    3.1415926,
                ),
                fixture_name="sink_left_group",
                fixture_source="external_fixture",
            ),

            "refrigerator": get_site_from_poses_with_fixture(
                site_id="refrigerator",
                work_area_pose=pose(
                    4.2,
                    -0.38,
                    1.0,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    4.2,
                    -1.5,
                    1.0,
                    1.57079632679,
                ),
                fixture_name="refrigerator",
                fixture_source="layout_entity"
            ),

            "cabinet": get_site_from_poses_with_fixture(
                site_id="cabinet",
                work_area_pose=pose(
                    1.2,
                    -0.2,
                    1.405,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    1.2,
                    -1.4,
                    1.0,
                    1.57079632679,
                ),
                fixture_name="cab_main_main_group",
                fixture_source="external_fixture",
            ),

            "wall_switch": get_site_from_fixture(
                site_id="wall_switch",
                layout_id=LayoutId.KITCHEN_L2,
                fixture_name="switch",
                clearance_m=_KITCHEN_WALL_SWITCH_ROBOT_CLEARANCE_M,
            ),

            "coffee_machine": _coffee_machine_site(LayoutId.KITCHEN_L2),

            "toaster": get_site_from_fixture(
                site_id="toaster",
                layout_id=LayoutId.KITCHEN_L2,
                clearance_m=0.85,
            ),

            "dishwasher": get_site_from_fixture(
                site_id="dishwasher",
                layout_id=LayoutId.KITCHEN_L2,
                clearance_m=1.0,
            ),

            "microwave": get_site_from_poses_with_fixture(
                site_id="microwave",
                work_area_pose=pose(
                    0.3763,
                    -4.099999,
                    1.6076,
                    1.5708
                ),
                robot_base_pose=pose(
                    1.4,
                    -4.1,
                    1.0,
                    3.14159265359
                ),
                fixture_name="microwave_left_group",
                fixture_source="external_fixture",
            ),

            "stove": get_site_from_poses(
                site_id="stove",
                work_area_pose=pose(
                    2.445,
                    -0.385,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    2.445,
                    -1.085,
                    0.924,
                    1.57079632679,
                ),
            ),

        },
    ),

    LayoutId.KITCHEN_L3: _kitchen_layout(
        LayoutId.KITCHEN_L3,
        site_configs={
            "counter_top": get_site_from_poses(
                site_id="counter_top",
                work_area_pose=pose(
                    2.2,
                    -0.4,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    2.2,
                    -1.1,
                    0.924,
                    1.57079632679,
                ),
                camera_rig_id="kitchen_counter",
                notes="Shared work surface for non-device-specific countertop tasks."
            ),

            "dining_table": get_site_from_poses(
                site_id="dining_table",
                work_area_pose=pose(
                    1.80,
                    -4.00,
                    0.924,
                    -1.57079632679,
                ),
                robot_base_pose=pose(
                    1.80,
                    -3.30,
                    0.924,
                    -1.57079632679,
                ),
                notes="Shared work surface for dining-table manipulation tasks."
            ),

            "sink": get_site_from_poses_with_fixture(
                site_id="sink",
                work_area_pose=pose(
                    0.50,
                    -2.00,
                    0.924,
                    3.1415926,
                ),
                robot_base_pose=pose(
                    1.20,
                    -2.00,
                    0.924,
                    3.1415926,
                ),
                fixture_name="sink_left_group",
                fixture_source="external_fixture",
            ),

            "refrigerator": get_site_from_poses_with_fixture(
                site_id="refrigerator",
                work_area_pose=pose(
                    5.19,
                    -1.80,
                    1.0,
                    0.0,
                ),
                robot_base_pose=pose(
                    4.5,
                    -1.80,
                    1.0,
                    0.0,
                ),
                fixture_name="refrigerator",
                fixture_source="layout_entity"
            ),

            "cabinet": get_site_from_poses_with_fixture(
                site_id="cabinet",
                work_area_pose=pose(
                    0.2,
                    -1.7,
                    1.405,
                    3.1415926,
                ),
                robot_base_pose=pose(
                    1.4,
                    -1.7,
                    1.0,
                    3.1415926,
                ),
                fixture_name="cab_1_left_group",
                fixture_source="external_fixture"
            ),

            "wall_switch": get_site_from_poses_with_fixture(
                site_id="wall_switch",
                work_area_pose=pose(
                    2.45,
                    0.0,
                    1.2,
                    _KITCHEN_WALL_SWITCH_ROBOT_BASE_YAW_RAD,
                ),
                robot_base_pose=PoseSpec(
                    position_xyz=(
                        _KITCHEN_L3_WALL_SWITCH_ROBOT_BASE_POSITION_XYZ
                    ),
                    quaternion_wxyz=(
                        0.707105,
                        0.0,
                        0.0,
                        0.707108,
                    ),
                ),
                fixture_name="switch",
                fixture_source="layout_entity",
            ),

            "coffee_machine": _coffee_machine_site(LayoutId.KITCHEN_L3),

            "toaster": get_site_from_fixture(
                site_id="toaster",
                layout_id=LayoutId.KITCHEN_L3,
                clearance_m=0.85,
            ),

            "dishwasher": replace(
                get_site_from_fixture(
                    site_id="dishwasher",
                    layout_id=LayoutId.KITCHEN_L3,
                    clearance_m=1.0,
                ),
                robot_base_center_pose=PoseSpec(
                    position_xyz=(1.61, -1.21, 0.604),
                    quaternion_wxyz=(0.0, 0.0, 0.0, 1.0),
                ),
            ),

            "microwave": get_site_from_fixture(
                site_id="microwave",
                layout_id=LayoutId.KITCHEN_L3,
                clearance_m=0.9,
            ),

            "stove": get_site_from_poses(
                site_id="stove",
                work_area_pose=pose(
                    4.418,
                    -0.393,
                    0.924,
                    1.57079632679,
                ),
                robot_base_pose=pose(
                    4.418,
                    -1.093,
                    0.924,
                    1.57079632679,
                ),
            ),

        },
    ),
}
