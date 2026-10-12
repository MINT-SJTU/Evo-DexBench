from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from dex_benchmark.assets import (
    evodex_asset_dir,
    partnet_mobility_modified_model_urdf_path,
    partnet_mobility_model_urdf_path,
    robocasa_asset_path,
    robocasa_texture_path,
)
from dex_benchmark.sim.layout_types import LayoutId, SceneClass
from dex_benchmark.sim.scenes.layouts.common import (
    pose,
    static_box,
    textured_plane,
)
from dex_benchmark.sim.scenes.layouts.spec import (
    ArticulatedAssetSpec,
    DirectionalLightSpec,
    InteractionSiteSpec,
    JointPositionLatchSpec,
    LayoutBuildItemSpec,
    LayoutLightingSpec,
    LayoutSpec,
    LayoutSupportSurfaceSpec,
    MJCFActorSpec,
    PhysicalMaterialSpec,
    PointLightSpec,
    PoseSpec,
    RectangularLightApertureSpec,
    RelativePlacementSpec,
    SpotLightSpec,
    StaticBoxSpec,
    StaticMeshSpec,
    TexturedPlaneSpec,
    YCBActorSpec,
    quat_multiply_wxyz,
)


STUDIO_STORAGE_CABINET_MODEL_ID = "48381"
STUDIO_DOOR_MODEL_ID = "8867"
STUDIO_WINDOW_MODEL_ID = "103323"
STUDIO_DESK_MODEL_ID = "30666"
STUDIO_WORKBENCH_MODEL_ID = "25144"
STUDIO_DESK_CHAIR_MODEL_ID = "36912"
STUDIO_PRINTER_TABLE_MODEL_ID = "27189"
STUDIO_PRINTER_MODEL_ID = "103981"
STUDIO_SLIDING_CABINET_A_MODEL_ID = "45642"
STUDIO_OPEN_SHELF_MODEL_ID = "47443"
STUDIO_DRESSER_CABINET_B_MODEL_ID = "47648"
STUDIO_DESK_LAMP_MODEL_ID = "16378"
STUDIO_WORKBENCH_BOX_MODEL_ID = "100129"
STUDIO_LAPTOP_MODEL_ID = "9992"
STUDIO_DISPLAY_MODEL_ID = "3392"
STUDIO_MOUSE_MODEL_ID = "101408"
STUDIO_WALL_CLOCK_MODEL_ID = "6963"
STUDIO_WALL_SWITCH_MODEL_ID = "100908"
STUDIO_TRASH_CAN_MODEL_ID = "101377"
STUDIO_WALL_FAN_MODEL_ID = "102100"
STUDIO_WORKBENCH_PLIERS_MODEL_ID = "100150"
STUDIO_GLASSES_MODEL_ID = "101860"
STUDIO_REMOTE_MODEL_ID = "100828"
STUDIO_STAPLER_MODEL_ID = "103301"
STUDIO_USB_MODEL_ID = "100078"
STUDIO_PHONE_MODEL_ID = "103927"
STUDIO_MUSTARD_BOTTLE_YCB_MODEL_ID = "006_mustard_bottle"
STUDIO_LARGE_MARKER_YCB_MODEL_ID = "040_large_marker"
STUDIO_MUG_YCB_MODEL_ID = "025_mug"
STUDIO_POWER_DRILL_YCB_MODEL_ID = "035_power_drill"
STUDIO_ADJUSTABLE_WRENCH_YCB_MODEL_ID = "042_adjustable_wrench"
STUDIO_RUBIKS_CUBE_YCB_MODEL_ID = "077_rubiks_cube"
STUDIO_MASTER_CHEF_CAN_YCB_MODEL_ID = "002_master_chef_can"
STUDIO_CRACKER_BOX_YCB_MODEL_ID = "003_cracker_box"
STUDIO_TOMATO_SOUP_CAN_YCB_MODEL_ID = "005_tomato_soup_can"
STUDIO_PUDDING_BOX_YCB_MODEL_ID = "008_pudding_box"
STUDIO_POTTED_MEAT_CAN_YCB_MODEL_ID = "010_potted_meat_can"
STUDIO_BANANA_YCB_MODEL_ID = "011_banana"
STUDIO_APPLE_YCB_MODEL_ID = "013_apple"
STUDIO_BLEACH_CLEANSER_YCB_MODEL_ID = "021_bleach_cleanser"
STUDIO_PHILLIPS_SCREWDRIVER_YCB_MODEL_ID = "043_phillips_screwdriver"
STUDIO_MEDIUM_CLAMP_YCB_MODEL_ID = "050_medium_clamp"
STUDIO_MINI_SOCCER_BALL_YCB_MODEL_ID = "053_mini_soccer_ball"
STUDIO_TENNIS_BALL_YCB_MODEL_ID = "056_tennis_ball"
STUDIO_CHAIN_YCB_MODEL_ID = "059_chain"
STUDIO_FOAM_BRICK_YCB_MODEL_ID = "061_foam_brick"
STUDIO_CUP_YCB_MODEL_ID = "065-a_cups"
STUDIO_TOY_AIRPLANE_YCB_MODEL_ID = "072-a_toy_airplane"
STUDIO_LEGO_DUPLO_YCB_MODEL_ID = "073-c_lego_duplo"
STUDIO_DICE_YCB_MODEL_ID = "062_dice"
STUDIO_GROUND_Z = 0.0
STUDIO_TABLETOP_Z = 0.75
STUDIO_FLOOR_HALF_HEIGHT = 0.025
STUDIO_STORAGE_CABINET_ROOT_Z = 0.803224
STUDIO_DESK_CHAIR_LOCAL_POSITION_XYZ = (-0.80, 0.0, -0.51)
STUDIO_DESK_TOP_SINGLE_BASE_LOCAL_XYZ = (-0.68, -0.02, 0.0038)
STUDIO_DESK_TOP_BIMANUAL_CENTER_LOCAL_XYZ = (-0.68, 0.0, 0.0038)
# Calibrated from the M14 L1 world pose p=(1.120543, 0.179606, 0.569999).
# L2/L3 reuse this desk-local transform so the robot-to-desk relation is fixed.
STUDIO_DESK_DRAWER_SINGLE_BASE_LOCAL_XYZ: tuple[float, float, float] = (
    -1.279455,
    0.619607,
    -0.296201,
)
STUDIO_DESK_DRAWER_BIMANUAL_CENTER_LOCAL_XYZ: tuple[float, float, float] = (
    -1.279455,
    0.0,
    -0.296201,
)
STUDIO_DESK_DRAWER_BASE_LOCAL_YAW_RAD = 0.03
STUDIO_WORKBENCH_SINGLE_BASE_LOCAL_XYZ: tuple[float, float, float] = (
    -0.6944315,
    -0.0180767,
    -0.0864240,
)
STUDIO_WORKBENCH_BIMANUAL_CENTER_LOCAL_XYZ: tuple[float, float, float] = (
    -0.6944315,
    0.0,
    -0.0864240,
)
STUDIO_SLIDING_CABINET_SINGLE_BASE_LOCAL_XYZ: tuple[float, float, float] = (
    -1.1597780,
    -0.4237106,
    -0.7320730,
)
STUDIO_SLIDING_CABINET_BIMANUAL_CENTER_LOCAL_XYZ: tuple[
    float, float, float
] = (
    -1.1597780,
    0.0,
    -0.7320730,
)
# Calibrated from the L3 human-viewer base pose
# p=(-1.147733, -0.587709, 0.779999), then reused in every Studio layout.
STUDIO_DRESSER_CABINET_SINGLE_BASE_LOCAL_XYZ: tuple[float, float, float] = (
    -0.9109247275,
    -0.0720318069,
    -0.2110710000,
)
STUDIO_OPEN_SHELF_BIMANUAL_MIDPOINT_FROM_CABINETS_LOCAL_XYZ: tuple[
    float, float, float
] = (
    -0.9959863823,
    -0.0189558900,
    -0.2312350000,
)
STUDIO_OPEN_SHELF_BIMANUAL_BASE_OFFSETS_LOCAL_XYZS: tuple[
    tuple[float, float, float],
    tuple[float, float, float],
] = (
    (-0.0032388350, 0.2999825120, 0.0),
    (0.0032388350, -0.2999825120, 0.0),
)
STUDIO_ROOM_CENTER_X = 0.25
STUDIO_ROOM_CENTER_Y = 0.0
STUDIO_ROOM_HALF_SIZE_X = 2.75
STUDIO_ROOM_HALF_SIZE_Y = 2.25
STUDIO_ROOM_HEIGHT = 2.80
STUDIO_CHANDELIER_COLOR_RGB = (4.2, 3.6, 2.76)
STUDIO_CHANDELIER_POINT_LIGHT_FRACTION = 0.35
STUDIO_CHANDELIER_SPOT_LIGHT_FRACTION = 0.65
STUDIO_CHANDELIER_SPOT_INNER_FOV_RAD = math.radians(130.0)
STUDIO_CHANDELIER_SPOT_OUTER_FOV_RAD = math.radians(150.0)
STUDIO_CEILING_LIGHT_Z = STUDIO_ROOM_HEIGHT - 0.22
STUDIO_CHANDELIER_MOUNT_Z = STUDIO_ROOM_HEIGHT - 0.02
STUDIO_WALL_COLLISION_THICKNESS = 0.04
STUDIO_SMALL_OBJECT_INITIAL_CLEARANCE_Z = 0.005
STUDIO_SETTLE_SIM_STEPS = 30
STUDIO_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
)
STUDIO_DEFAULT_JOINT_FRICTION = 0.1

STUDIO_DESK_TOP_SURFACE_ID = "desk.top"
STUDIO_DESK_SECOND_SHELF_SURFACE_ID = "desk.second_shelf"
STUDIO_WORKBENCH_TOP_SURFACE_ID = "workbench.top"
STUDIO_STORAGE_TOP_SURFACE_ID = "storage_cabinet.top"
STUDIO_OPEN_SHELF_TOP_SURFACE_ID = "open_shelf.top"
STUDIO_OPEN_SHELF_THIRD_LEVEL_SURFACE_ID = "open_shelf.third_level"
STUDIO_OPEN_SHELF_SECOND_LEVEL_SURFACE_ID = "open_shelf.second_level"
STUDIO_OPEN_SHELF_FIRST_LEVEL_SURFACE_ID = "open_shelf.first_level"
STUDIO_DRESSER_TOP_SURFACE_ID = "dresser_cabinet_b.top"
STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID = "dresser_cabinet_b.outer_counter"
STUDIO_DRESSER_LOWER_CABINET_UPPER_SURFACE_ID = (
    "dresser_cabinet_b.lower_cabinet_upper"
)
STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID = (
    "dresser_cabinet_b.upper_cabinet_lower"
)
STUDIO_DRESSER_UPPER_CABINET_UPPER_SURFACE_ID = (
    "dresser_cabinet_b.upper_cabinet_upper"
)
STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_SURFACE_ID = (
    "sliding_cabinet_a.lower_cabinet_upper"
)
STUDIO_SLIDING_CABINET_LOWER_CABINET_LOWER_SURFACE_ID = (
    "sliding_cabinet_a.lower_cabinet_lower"
)
STUDIO_PRINTER_TABLE_BOTTOM_SHELF_SURFACE_ID = "printer_table.bottom_shelf"

STUDIO_DESK_SECOND_SHELF_Z = 0.554
STUDIO_OPEN_SHELF_THIRD_LEVEL_Z = 1.354
STUDIO_OPEN_SHELF_SECOND_LEVEL_Z = 1.044
STUDIO_OPEN_SHELF_FIRST_LEVEL_Z = 0.714
STUDIO_DRESSER_OUTER_COUNTER_Z = 0.864
STUDIO_DRESSER_LOWER_CABINET_UPPER_Z = 0.434
STUDIO_DRESSER_UPPER_CABINET_LOWER_Z = 1.234
STUDIO_DRESSER_UPPER_CABINET_UPPER_Z = 1.574
STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_Z = 0.484
STUDIO_SLIDING_CABINET_LOWER_CABINET_LOWER_Z = 0.124
STUDIO_PRINTER_TABLE_BOTTOM_SHELF_Z = 0.154

WALL_BACK_QUATERNION_WXYZ = (-0.707, 0.707, 0.0, 0.0)
WALL_FRONT_QUATERNION_WXYZ = (0.0, 0.0, 0.707, -0.707)
WALL_LEFT_QUATERNION_WXYZ = (0.5, 0.5, -0.5, -0.5)
WALL_RIGHT_QUATERNION_WXYZ = (-0.5, 0.5, -0.5, 0.5)

PARTNET_BBOX_MIN_Z: dict[str, float] = {
    STUDIO_DOOR_MODEL_ID: -0.9315,
    STUDIO_WINDOW_MODEL_ID: -0.7205,
    STUDIO_DESK_MODEL_ID: -0.4707,
    STUDIO_WORKBENCH_MODEL_ID: -0.5190,
    STUDIO_DESK_CHAIR_MODEL_ID: -0.8001,
    STUDIO_PRINTER_TABLE_MODEL_ID: -0.6964,
    STUDIO_PRINTER_MODEL_ID: -0.4624,
    STUDIO_SLIDING_CABINET_A_MODEL_ID: -0.8671,
    STUDIO_OPEN_SHELF_MODEL_ID: -0.9103,
    STUDIO_DRESSER_CABINET_B_MODEL_ID: -0.8428,
    STUDIO_STORAGE_CABINET_MODEL_ID: -0.803224,
    STUDIO_DESK_LAMP_MODEL_ID: -0.776024,
    STUDIO_WORKBENCH_BOX_MODEL_ID: -0.375444,
    STUDIO_LAPTOP_MODEL_ID: -0.215293,
    STUDIO_DISPLAY_MODEL_ID: -0.517509,
    # Mouse joint_3 rotates mesh coordinates into the articulation root frame.
    STUDIO_MOUSE_MODEL_ID: -0.164461,
    STUDIO_WALL_CLOCK_MODEL_ID: -0.061566,
    STUDIO_WALL_SWITCH_MODEL_ID: -0.069956,
    STUDIO_TRASH_CAN_MODEL_ID: -0.568043,
    STUDIO_WALL_FAN_MODEL_ID: -0.234318,
    STUDIO_WORKBENCH_PLIERS_MODEL_ID: -0.095317,
    STUDIO_GLASSES_MODEL_ID: -0.262959,
    STUDIO_REMOTE_MODEL_ID: -0.923825,
    STUDIO_STAPLER_MODEL_ID: -0.339308,
    STUDIO_USB_MODEL_ID: -0.161533,
    # Model 103927 has a fixed base-to-phone joint with
    # rpy=(pi / 2, 0, -pi / 2); cache its bounds in the articulation-root frame.
    STUDIO_PHONE_MODEL_ID: -0.917632,
}

PARTNET_BBOX_CENTER_XYZ: dict[str, tuple[float, float, float]] = {
    STUDIO_DOOR_MODEL_ID: (0.1444, 0.0548, -0.0540),
    STUDIO_WINDOW_MODEL_ID: (-0.0061, 0.0321, -0.0045),
    STUDIO_DESK_MODEL_ID: (-0.0487, 0.0075, -0.0376),
    STUDIO_WORKBENCH_MODEL_ID: (-0.0212, 0.0096, -0.1581),
    STUDIO_DESK_CHAIR_MODEL_ID: (-0.0162, 0.0276, 0.0597),
    STUDIO_PRINTER_TABLE_MODEL_ID: (-0.0649, -0.0211, -0.1475),
    STUDIO_PRINTER_MODEL_ID: (-0.1275, -0.0075, 0.0260),
    STUDIO_SLIDING_CABINET_A_MODEL_ID: (-0.0127, 0.0107, -0.0653),
    STUDIO_OPEN_SHELF_MODEL_ID: (-0.0255, 0.0105, -0.0189),
    STUDIO_DRESSER_CABINET_B_MODEL_ID: (-0.0313, -0.0010, 0.0190),
    STUDIO_STORAGE_CABINET_MODEL_ID: (-0.028135, -0.0161755, -0.0279555),
    STUDIO_DESK_LAMP_MODEL_ID: (0.0669275, 0.0068545, -0.1003105),
    STUDIO_WORKBENCH_BOX_MODEL_ID: (0.027861, -0.006557, 0.139414),
    STUDIO_LAPTOP_MODEL_ID: (-0.09843, -0.0054765, 0.223419),
    STUDIO_DISPLAY_MODEL_ID: (-0.053867, 0.004145, 0.0497275),
    STUDIO_MOUSE_MODEL_ID: (0.0552555, 0.0021035, 0.0885075),
    STUDIO_WALL_CLOCK_MODEL_ID: (0.006297, -0.019233, 0.015845),
    STUDIO_WALL_SWITCH_MODEL_ID: (0.000337, 0.027134, 0.026533),
    STUDIO_TRASH_CAN_MODEL_ID: (0.0020115, -0.0014255, -0.0212195),
    STUDIO_WALL_FAN_MODEL_ID: (-0.024277, -0.001078, -0.024076),
    STUDIO_WORKBENCH_PLIERS_MODEL_ID: (-0.1268625, -0.2998705, -0.0012375),
    STUDIO_GLASSES_MODEL_ID: (0.3104415, 0.0161085, -0.042399),
    STUDIO_REMOTE_MODEL_ID: (-0.020499, -0.0113645, 0.0079765),
    STUDIO_STAPLER_MODEL_ID: (-0.0357635, 0.005308, 0.098331),
    STUDIO_USB_MODEL_ID: (0.1296135, -0.1802445, 0.005861),
    STUDIO_PHONE_MODEL_ID: (-0.0004555, -0.008414, -0.008702),
}

PARTNET_BBOX_EXTENT_XYZ: dict[str, tuple[float, float, float]] = {
    STUDIO_DOOR_MODEL_ID: (0.1440, 0.6582, 1.7551),
    STUDIO_WINDOW_MODEL_ID: (0.1371, 1.4424, 1.4321),
    STUDIO_DESK_MODEL_ID: (0.8225, 1.6304, 0.8662),
    STUDIO_WORKBENCH_MODEL_ID: (0.8091, 1.6182, 0.7218),
    STUDIO_DESK_CHAIR_MODEL_ID: (1.2127, 1.2440, 1.7196),
    STUDIO_PRINTER_TABLE_MODEL_ID: (1.2895, 1.1982, 1.0977),
    STUDIO_PRINTER_MODEL_ID: (1.5824, 1.2889, 0.9769),
    STUDIO_SLIDING_CABINET_A_MODEL_ID: (0.3916, 1.1079, 1.6035),
    STUDIO_OPEN_SHELF_MODEL_ID: (0.3713, 0.9431, 1.7826),
    STUDIO_DRESSER_CABINET_B_MODEL_ID: (0.3923, 0.9127, 1.7236),
    STUDIO_STORAGE_CABINET_MODEL_ID: (0.599326, 1.292115, 1.550537),
    STUDIO_DESK_LAMP_MODEL_ID: (1.167979, 0.820419, 1.351427),
    STUDIO_WORKBENCH_BOX_MODEL_ID: (1.356872, 0.874162, 1.029716),
    STUDIO_LAPTOP_MODEL_ID: (1.21733, 1.285505, 0.877424),
    STUDIO_DISPLAY_MODEL_ID: (0.381336, 1.620676, 1.134473),
    STUDIO_MOUSE_MODEL_ID: (1.906169, 0.950039, 0.505937),
    STUDIO_WALL_CLOCK_MODEL_ID: (1.960156, 1.960156, 0.154822),
    STUDIO_WALL_SWITCH_MODEL_ID: (1.219122, 1.536552, 0.192978),
    STUDIO_TRASH_CAN_MODEL_ID: (1.061109, 1.706769, 1.093647),
    STUDIO_WALL_FAN_MODEL_ID: (1.966636, 1.966636, 0.420484),
    STUDIO_WORKBENCH_PLIERS_MODEL_ID: (1.760113, 1.154407, 0.188159),
    STUDIO_GLASSES_MODEL_ID: (0.950903, 1.213749, 0.441120),
    STUDIO_REMOTE_MODEL_ID: (0.221858, 0.709945, 1.863604),
    STUDIO_STAPLER_MODEL_ID: (1.891176, 0.510714, 0.875278),
    STUDIO_USB_MODEL_ID: (1.744237, 1.181251, 0.334788),
    STUDIO_PHONE_MODEL_ID: (0.107395, 1.018002, 1.817860),
}

STUDIO_DOOR_MODEL_SCALE = 1.25
STUDIO_DOOR_CLOSED_YAW = 1.5707963267948966
STUDIO_DOOR_CLOSED_QPOS = (-0.8168140899333464, -0.7853981633974483)
STUDIO_WINDOW_MODEL_SCALE = 1.0
STUDIO_DESK_MODEL_SCALE = 1.0
STUDIO_WORKBENCH_MODEL_SCALE = 1.0
STUDIO_DESK_CHAIR_MODEL_SCALE = 0.65
STUDIO_PRINTER_TABLE_MODEL_SCALE = 0.70
STUDIO_PRINTER_MODEL_SCALE = 0.50
STUDIO_SLIDING_CABINET_A_MODEL_SCALE = 1.15
STUDIO_OPEN_SHELF_MODEL_SCALE = 1.0
STUDIO_DRESSER_CABINET_B_MODEL_SCALE = 1.15
STUDIO_STORAGE_CABINET_MODEL_SCALE = 1.0
STUDIO_DESK_LAMP_MODEL_SCALE = 0.30
STUDIO_WORKBENCH_BOX_MODEL_SCALE = 0.15
STUDIO_LAPTOP_MODEL_SCALE = 0.28
STUDIO_LAPTOP_INITIAL_CLEARANCE_Z = 0.001
STUDIO_DISPLAY_MODEL_SCALE = 0.35
STUDIO_MOUSE_MODEL_SCALE = 0.06
STUDIO_WALL_CLOCK_MODEL_SCALE = 0.35
STUDIO_WALL_SWITCH_MODEL_SCALE = 0.15
STUDIO_TRASH_CAN_MODEL_SCALE = 0.40
STUDIO_WALL_FAN_MODEL_SCALE = 0.285
STUDIO_WORKBENCH_PLIERS_MODEL_SCALE = 0.10
STUDIO_GLASSES_MODEL_SCALE = 0.12
STUDIO_REMOTE_MODEL_SCALE = 0.08
STUDIO_STAPLER_MODEL_SCALE = 0.08
STUDIO_USB_MODEL_SCALE = 0.03
STUDIO_PHONE_MODEL_SCALE = 0.12
STUDIO_PHONE_INITIAL_CLEARANCE_Z = 0.002
STUDIO_PHONE_FLAT_LOCAL_QUATERNION_WXYZ = (
    math.sqrt(0.5),
    0.0,
    math.sqrt(0.5),
    0.0,
)
STUDIO_POTTED_PLANT_ROBOCASA_ASSET = (
    "fixtures/accessories/plants/senecio/model.xml"
)
STUDIO_L2_POTTED_PLANT_ROBOCASA_ASSET = (
    "fixtures/accessories/plants/aloe/model.xml"
)
STUDIO_L3_POTTED_PLANT_ROBOCASA_ASSET = (
    "fixtures/accessories/plants/spikey_in_white_pot/model.xml"
)
STUDIO_POTTED_PLANT_SCALE_XYZ = (0.9, 0.9, 1.4)
# All three potted-plant MJCF assets are centered on local Z with exterior
# bounds [-0.5, 0.5]. Their scaled lower face reaches ground here.
STUDIO_CENTERED_POTTED_PLANT_GROUNDED_ROOT_Z = (
    0.5 * STUDIO_POTTED_PLANT_SCALE_XYZ[2]
)
STUDIO_CHANDELIER_ASSET_RELATIVE_PATH = (
    "polyhaven/Chandelier_01/Chandelier_01_1k.glb"
)
STUDIO_L2_CHANDELIER_ASSET_RELATIVE_PATH = (
    "polyhaven/Chandelier_02/Chandelier_02_1k.glb"
)
STUDIO_L3_CEILING_LIGHT_ASSET_RELATIVE_PATH = (
    "polyhaven/modern_ceiling_lamp_01/modern_ceiling_lamp_01_1k.glb"
)
# After the +90-degree X rotation, local +Y maps to world +Z. Offset each
# positive-Y model so its upper endpoint stays at the ceiling mount height.
STUDIO_L2_CHANDELIER_ROOT_Z = STUDIO_CHANDELIER_MOUNT_Z - 0.9077612161636353
STUDIO_L3_CEILING_LIGHT_ROOT_Z = (
    STUDIO_CHANDELIER_MOUNT_Z - 1.1726436614990234
)
STUDIO_CHANDELIER_MODEL_TO_WORLD_QUATERNION_WXYZ = (
    math.sqrt(0.5),
    math.sqrt(0.5),
    0.0,
    0.0,
)
STUDIO_COAT_RACK_ASSET_RELATIVE_PATH = (
    "kenney/furniture_kit/coatRackStanding.glb"
)
STUDIO_COAT_RACK_SCALE_XYZ = (2.2, 2.2, 2.2)
STUDIO_COAT_RACK_MODEL_TO_WORLD_QUATERNION_WXYZ = (
    math.sqrt(0.5),
    math.sqrt(0.5),
    0.0,
    0.0,
)
STUDIO_UPRIGHT_BOOK_MODEL_TO_SURFACE_QUATERNION_WXYZ = (
    0.5,
    0.5,
    0.5,
    0.5,
)
# Rz(0.05) @ Ry(-pi / 2): map model-space thickness (+X) onto shelf +Z.
STUDIO_FLAT_BOOK_MODEL_TO_SURFACE_QUATERNION_WXYZ = (
    0.7068858218260865,
    0.01767582816329799,
    -0.7068858218260865,
    0.01767582816329799,
)
STUDIO_BOOK_ASSET_RELATIVE_PATHS: dict[str, str] = {
    "book_calculus": (
        "polyhaven/decorative_book_set_01/"
        "book_hardcover_01_cover02_1k.glb"
    ),
    "book_chrysanthemums": (
        "polyhaven/decorative_book_set_01/"
        "book_hardcover_01_cover06_1k.glb"
    ),
    "book_travel_flat": (
        "polyhaven/decorative_book_set_01/"
        "book_hardcover_01_cover35_1k.glb"
    ),
    "book_space": (
        "polyhaven/decorative_book_set_01/"
        "book_hardcover_01_cover65_1k.glb"
    ),
}
# The converted GLBs are centered at their model-space bounding-box centers.
STUDIO_BOOK_BBOX_HALF_EXTENT_XYZ: dict[
    str, tuple[float, float, float]
] = {
    "book_calculus": (0.0209294595, 0.1136102229, 0.0767762959),
    "book_chrysanthemums": (0.0208408516, 0.1131292358, 0.0764512569),
    "book_travel_flat": (0.0200143103, 0.1132446378, 0.0765292421),
    "book_space": (0.0206057001, 0.1133840755, 0.0766234696),
}
STUDIO_BOOK_SCALE_XYZ: dict[str, tuple[float, float, float]] = {
    "book_calculus": (0.85, 1.10, 1.05),
    "book_chrysanthemums": (0.75, 1.04, 0.98),
    "book_travel_flat": (0.70, 1.05, 1.08),
    "book_space": (0.82, 1.08, 1.04),
}
STUDIO_BOOK_LOCAL_XY: dict[str, tuple[float, float]] = {
    "book_calculus": (0.0, 0.32),
    "book_chrysanthemums": (0.0, 0.20),
    "book_space": (0.0, 0.07),
    "book_travel_flat": (0.0, -0.20),
}
STUDIO_RUG_TEXTURE_RELATIVE_PATH = (
    "polyhaven/fabric_pattern_07/fabric_pattern_07_col_1_1k.jpg"
)
STUDIO_L2_RUG_TEXTURE_RELATIVE_PATH = (
    "polyhaven/dirty_carpet/dirty_carpet_diff_1k.jpg"
)
STUDIO_L3_RUG_TEXTURE_RELATIVE_PATH = (
    "polyhaven/floral_jacquard/floral_jacquard_diff_1k.jpg"
)
STUDIO_DESK_LAMP_INITIAL_QPOS = (0.004, 0.0)
STUDIO_WORKBENCH_BOX_INITIAL_QPOS = (0.0, 0.0)
STUDIO_LAPTOP_INITIAL_QPOS = (0.0,)
STUDIO_DISPLAY_HEIGHT_JOINT_QPOS = 0.12
STUDIO_MOUSE_INITIAL_QPOS = (0.0, 0.0, 0.0)
STUDIO_WALL_CLOCK_INITIAL_QPOS = (0.0, 0.0, 0.0)
STUDIO_WALL_SWITCH_INITIAL_QPOS = (0.0, 0.0)
STUDIO_TRASH_CAN_INITIAL_QPOS = (0.0, 0.0)
STUDIO_WALL_FAN_INITIAL_QPOS = (0.0,)
STUDIO_STORAGE_CABINET_INITIAL_QPOS = (0.0, 0.0)
STUDIO_WORKBENCH_PLIERS_INITIAL_QPOS = (0.0,)
STUDIO_GLASSES_INITIAL_QPOS = (0.0, 0.0)
STUDIO_REMOTE_INITIAL_QPOS = (0.0, 0.0, 0.0, 0.0)
STUDIO_STAPLER_INITIAL_QPOS = (0.0, 0.0)
STUDIO_USB_INITIAL_QPOS = (0.0,)
STUDIO_PHONE_INITIAL_QPOS = (0.0,) * 7
STUDIO_LAPTOP_SCREEN_JOINT_FRICTION = 0.1
STUDIO_LAPTOP_SCREEN_JOINT_LATCH = JointPositionLatchSpec(
    stiffness=500.0,
    damping=10.0,
    force_limit=1.5,
)
STUDIO_DESK_LAMP_HEAD_JOINT_FRICTION = 1.5
STUDIO_FURNITURE_ROBOT_CLEARANCE = 0.35
STUDIO_OPEN_SHELF_USABLE_TOP_OFFSET_Z = -0.055
YCB_BBOX_MIN_Z: dict[str, float] = {
    STUDIO_MASTER_CHEF_CAN_YCB_MODEL_ID: -0.07089759050000001,
    STUDIO_CRACKER_BOX_YCB_MODEL_ID: -0.108566575,
    STUDIO_TOMATO_SOUP_CAN_YCB_MODEL_ID: -0.051258729,
    STUDIO_MUSTARD_BOTTLE_YCB_MODEL_ID: -0.09776508246011102,
    STUDIO_PUDDING_BOX_YCB_MODEL_ID: -0.02062462,
    STUDIO_POTTED_MEAT_CAN_YCB_MODEL_ID: -0.041813795,
    STUDIO_BANANA_YCB_MODEL_ID: -0.01887479572529618,
    STUDIO_APPLE_YCB_MODEL_ID: -0.036133501,
    STUDIO_BLEACH_CLEANSER_YCB_MODEL_ID: -0.1286904144606002,
    STUDIO_LARGE_MARKER_YCB_MODEL_ID: -0.009808582,
    STUDIO_MUG_YCB_MODEL_ID: -0.0407278125,
    STUDIO_POWER_DRILL_YCB_MODEL_ID: -0.02980699,
    STUDIO_ADJUSTABLE_WRENCH_YCB_MODEL_ID: -0.00769642877,
    STUDIO_PHILLIPS_SCREWDRIVER_YCB_MODEL_ID: -0.017866396141640116,
    STUDIO_MEDIUM_CLAMP_YCB_MODEL_ID: -0.013659795,
    STUDIO_MINI_SOCCER_BALL_YCB_MODEL_ID: -0.06222656100881105,
    STUDIO_TENNIS_BALL_YCB_MODEL_ID: -0.033261895,
    STUDIO_CHAIN_YCB_MODEL_ID: -0.013744232872271842,
    STUDIO_FOAM_BRICK_YCB_MODEL_ID: -0.026036734721481644,
    STUDIO_CUP_YCB_MODEL_ID: -0.03153898,
    STUDIO_TOY_AIRPLANE_YCB_MODEL_ID: -0.09064775810542858,
    STUDIO_LEGO_DUPLO_YCB_MODEL_ID: -0.012190477780702612,
    STUDIO_RUBIKS_CUBE_YCB_MODEL_ID: -0.029140974,
    STUDIO_DICE_YCB_MODEL_ID: -0.008914011,
}
YCB_MODEL_SCALE: dict[str, float] = {
    STUDIO_MASTER_CHEF_CAN_YCB_MODEL_ID: 0.7,
    STUDIO_CRACKER_BOX_YCB_MODEL_ID: 1.0,
    STUDIO_TOMATO_SOUP_CAN_YCB_MODEL_ID: 1.0,
    STUDIO_MUSTARD_BOTTLE_YCB_MODEL_ID: 1.0,
    STUDIO_PUDDING_BOX_YCB_MODEL_ID: 0.8,
    STUDIO_POTTED_MEAT_CAN_YCB_MODEL_ID: 1.0,
    STUDIO_BANANA_YCB_MODEL_ID: 1.0,
    STUDIO_APPLE_YCB_MODEL_ID: 1.0,
    STUDIO_BLEACH_CLEANSER_YCB_MODEL_ID: 1.0,
    STUDIO_LARGE_MARKER_YCB_MODEL_ID: 1.0,
    STUDIO_MUG_YCB_MODEL_ID: 1.0,
    STUDIO_POWER_DRILL_YCB_MODEL_ID: 1.0,
    STUDIO_ADJUSTABLE_WRENCH_YCB_MODEL_ID: 1.0,
    STUDIO_PHILLIPS_SCREWDRIVER_YCB_MODEL_ID: 1.0,
    STUDIO_MEDIUM_CLAMP_YCB_MODEL_ID: 1.0,
    STUDIO_MINI_SOCCER_BALL_YCB_MODEL_ID: 0.5,
    STUDIO_TENNIS_BALL_YCB_MODEL_ID: 1.0,
    STUDIO_CHAIN_YCB_MODEL_ID: 1.0,
    STUDIO_FOAM_BRICK_YCB_MODEL_ID: 1.0,
    STUDIO_CUP_YCB_MODEL_ID: 1.0,
    STUDIO_TOY_AIRPLANE_YCB_MODEL_ID: 1.0,
    STUDIO_LEGO_DUPLO_YCB_MODEL_ID: 1.0,
    STUDIO_RUBIKS_CUBE_YCB_MODEL_ID: 1.0,
    STUDIO_DICE_YCB_MODEL_ID: 1.0,
}
STUDIO_DESK_CHAIR_QPOS_JOINT_NAMES: tuple[str, ...] = (
    "joint_0",
    "joint_3",
    "joint_6",
    "joint_9",
    "joint_12",
    "joint_18",
    "joint_1",
    "joint_2",
    "joint_4",
    "joint_5",
    "joint_7",
    "joint_8",
    "joint_10",
    "joint_11",
    "joint_13",
    "joint_14",
    "joint_15",
    "joint_17",
)
STUDIO_DESK_CHAIR_SEAT_YAW_JOINT_NAME = "joint_15"
STUDIO_DESK_CHAIR_SEAT_YAW_QPOS_INDEX = STUDIO_DESK_CHAIR_QPOS_JOINT_NAMES.index(
    STUDIO_DESK_CHAIR_SEAT_YAW_JOINT_NAME
)
STUDIO_L1_DESK_CHAIR_INITIAL_QPOS = tuple(
    3.213 if joint_index == STUDIO_DESK_CHAIR_SEAT_YAW_QPOS_INDEX else 0.0
    for joint_index in range(len(STUDIO_DESK_CHAIR_QPOS_JOINT_NAMES))
)


@dataclass(frozen=True)
class StudioRoomTextureSpec:
    floor_texture: str
    wall_texture: str
    ceiling_texture: str
    floor_repeat_uv: tuple[float, float] = (1.0, 1.0)
    wall_repeat_uv: tuple[float, float] = (1.0, 1.0)
    ceiling_repeat_uv: tuple[float, float] = (0.8, 0.8)


@dataclass(frozen=True)
class StudioLargeAssetPlacementSpec:
    printer_table_pose: PoseSpec
    sliding_cabinet_a_pose: PoseSpec
    open_shelf_pose: PoseSpec
    dresser_cabinet_b_pose: PoseSpec
    printer_pose: PoseSpec | None = None
    ceiling_light_xy: tuple[float, float] = (STUDIO_ROOM_CENTER_X, STUDIO_ROOM_CENTER_Y)
    ceiling_light_root_z: float = STUDIO_CHANDELIER_MOUNT_Z
    # The point light sits just below the fixture's lowest visual geometry so
    # the shadow map cannot let the fixture occlude its own light source.
    ceiling_light_emission_z: float = 1.93
    desk_chair_initial_qpos: tuple[float, ...] | None = None
    ceiling_light_mesh_relative_path: str | None = None
    ceiling_light_quaternion_wxyz: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )
    ceiling_light_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)


def _normalized_direction(
    source_xyz: tuple[float, float, float],
    target_xyz: tuple[float, float, float],
) -> tuple[float, float, float]:
    direction = np.asarray(target_xyz, dtype=np.float64) - np.asarray(
        source_xyz,
        dtype=np.float64,
    )
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-9:
        raise ValueError(
            f"Light source {source_xyz} and target {target_xyz} must be distinct."
        )
    return tuple(float(component) for component in direction / direction_norm)


def _studio_lighting_spec(
    placement_spec: StudioLargeAssetPlacementSpec,
) -> LayoutLightingSpec:
    room_max_x = STUDIO_ROOM_CENTER_X + STUDIO_ROOM_HALF_SIZE_X
    window_center_y = 0.80
    window_bottom_z = 1.05
    window_height = (
        PARTNET_BBOX_EXTENT_XYZ[STUDIO_WINDOW_MODEL_ID][2]
        * STUDIO_WINDOW_MODEL_SCALE
    )
    window_half_depth = (
        0.5
        * PARTNET_BBOX_EXTENT_XYZ[STUDIO_WINDOW_MODEL_ID][0]
        * STUDIO_WINDOW_MODEL_SCALE
    )
    # Raster shadow maps treat the PartNet window's half-transparent panes as
    # opaque. Put the light 5 cm inside the pane's inner face so sunlight is not
    # blocked by the glass while the window model remains visible.
    window_light_position_xyz = (
        room_max_x - window_half_depth - 0.05,
        window_center_y,
        window_bottom_z + window_height / 2.0,
    )
    # Parallel sunlight crosses the entire rectangular window and reaches the
    # floor near the room center. The source pose is just inside the glass so
    # transparent panes cannot incorrectly block the raster shadow map.
    sunlight_direction_xyz = _normalized_direction(
        source_xyz=window_light_position_xyz,
        target_xyz=(STUDIO_ROOM_CENTER_X, STUDIO_ROOM_CENTER_Y, 0.15),
    )
    chandelier_position_xyz = (
        placement_spec.ceiling_light_xy[0],
        placement_spec.ceiling_light_xy[1],
        placement_spec.ceiling_light_emission_z,
    )
    chandelier_red, chandelier_green, chandelier_blue = (
        STUDIO_CHANDELIER_COLOR_RGB
    )
    return LayoutLightingSpec(
        ambient_rgb=(0.20, 0.19, 0.17),
        point_lights=(
            PointLightSpec(
                position_xyz=chandelier_position_xyz,
                color_rgb=(
                    chandelier_red * STUDIO_CHANDELIER_POINT_LIGHT_FRACTION,
                    chandelier_green * STUDIO_CHANDELIER_POINT_LIGHT_FRACTION,
                    chandelier_blue * STUDIO_CHANDELIER_POINT_LIGHT_FRACTION,
                ),
                # SAPIEN's point-light cube shadow has no receiver bias or PCF.
                # Keep only the omnidirectional fill contribution shadow-free.
                shadow=False,
                shadow_near=0.1,
                shadow_far=8.0,
                shadow_map_size=2048,
            ),
        ),
        spot_lights=(
            SpotLightSpec(
                position_xyz=chandelier_position_xyz,
                direction_xyz=(0.0, 0.0, -1.0),
                inner_fov_rad=STUDIO_CHANDELIER_SPOT_INNER_FOV_RAD,
                outer_fov_rad=STUDIO_CHANDELIER_SPOT_OUTER_FOV_RAD,
                color_rgb=(
                    chandelier_red * STUDIO_CHANDELIER_SPOT_LIGHT_FRACTION,
                    chandelier_green * STUDIO_CHANDELIER_SPOT_LIGHT_FRACTION,
                    chandelier_blue * STUDIO_CHANDELIER_SPOT_LIGHT_FRACTION,
                ),
                shadow=True,
                shadow_near=0.4,
                shadow_far=4.5,
                shadow_map_size=4096,
            ),
        ),
        directional_lights=(
            DirectionalLightSpec(
                position_xyz=window_light_position_xyz,
                direction_xyz=sunlight_direction_xyz,
                color_rgb=(1.02, 0.82, 0.43),
                aperture=RectangularLightApertureSpec(
                    center_xyz=(
                        room_max_x,
                        window_center_y,
                        window_bottom_z + window_height / 2.0,
                    ),
                    axis_u_xyz=(0.0, 1.0, 0.0),
                    axis_v_xyz=(0.0, 0.0, 1.0),
                    half_size_uv=(
                        0.5
                        * PARTNET_BBOX_EXTENT_XYZ[STUDIO_WINDOW_MODEL_ID][1]
                        * STUDIO_WINDOW_MODEL_SCALE,
                        window_height / 2.0,
                    ),
                    edge_softness_m=0.01,
                ),
                shadow=True,
                shadow_scale=6.0,
                shadow_near=0.05,
                shadow_far=8.0,
                shadow_map_size=2048,
            ),
        ),
    )


@dataclass(frozen=True)
class StudioRoomDecorPlacementSpec:
    rug_texture_relative_path: str
    rug_pose: PoseSpec
    potted_plant_robocasa_asset: str
    potted_plant_root_pose: PoseSpec
    potted_plant_scale_xyz: tuple[float, float, float]
    coat_rack_asset_relative_path: str
    coat_rack_root_pose: PoseSpec
    coat_rack_scale_xyz: tuple[float, float, float]
    wall_clock_bbox_center_pose: PoseSpec
    wall_switch_bbox_center_pose: PoseSpec
    trash_can_bbox_center_pose: PoseSpec
    wall_fan_bbox_center_pose: PoseSpec


@dataclass(frozen=True)
class StudioRelativeContentPlacementOverride:
    local_xy: tuple[float, float]
    bbox_center_z: float | None = None
    yaw: float | None = None
    surface_id: str | None = None


def _grounded_coat_rack_pose(x: float, y: float, yaw: float) -> PoseSpec:
    return PoseSpec(
        position_xyz=(x, y, STUDIO_GROUND_Z),
        quaternion_wxyz=quat_multiply_wxyz(
            pose(0.0, 0.0, 0.0, yaw).quaternion_wxyz,
            STUDIO_COAT_RACK_MODEL_TO_WORLD_QUATERNION_WXYZ,
        ),
    )


def _desk_top_robot_base_pose(
    desk_work_area_pose: PoseSpec,
    local_position_xyz: tuple[float, float, float],
) -> PoseSpec:
    """Transform a calibrated desk-top base pose into the world frame."""

    robot_base_position = desk_work_area_pose.transform_point(local_position_xyz)
    return PoseSpec(
        position_xyz=tuple(float(value) for value in robot_base_position),
        quaternion_wxyz=desk_work_area_pose.quaternion_wxyz,
    )


def _desk_drawer_robot_base_pose(
    desk_work_area_pose: PoseSpec,
    local_position_xyz: tuple[float, float, float],
) -> PoseSpec:
    """Transform the calibrated drawer-task base pose into the world frame."""

    robot_base_position = desk_work_area_pose.transform_point(local_position_xyz)
    local_quaternion = pose(
        0.0,
        0.0,
        0.0,
        STUDIO_DESK_DRAWER_BASE_LOCAL_YAW_RAD,
    ).quaternion_wxyz
    return PoseSpec(
        position_xyz=tuple(float(value) for value in robot_base_position),
        quaternion_wxyz=quat_multiply_wxyz(
            desk_work_area_pose.quaternion_wxyz,
            local_quaternion,
        ),
    )


def _workbench_robot_base_pose(
    workbench_work_area_pose: PoseSpec,
    local_position_xyz: tuple[float, float, float],
) -> PoseSpec:
    """Transform the calibrated workbench base pose into the world frame."""

    robot_base_position = workbench_work_area_pose.transform_point(
        local_position_xyz
    )
    return PoseSpec(
        position_xyz=tuple(float(value) for value in robot_base_position),
        quaternion_wxyz=workbench_work_area_pose.quaternion_wxyz,
    )


def _sliding_cabinet_robot_base_pose(
    cabinet_pose: PoseSpec,
    local_position_xyz: tuple[float, float, float],
) -> PoseSpec:
    """Transform the calibrated sliding-cabinet base into the world frame."""

    robot_base_position = cabinet_pose.transform_point(local_position_xyz)
    return PoseSpec(
        position_xyz=tuple(float(value) for value in robot_base_position),
        quaternion_wxyz=cabinet_pose.quaternion_wxyz,
    )


def _dresser_cabinet_robot_base_pose(
    cabinet_pose: PoseSpec,
    local_position_xyz: tuple[float, float, float],
) -> PoseSpec:
    """Transform the calibrated dresser-cabinet base into the world frame."""

    robot_base_position = cabinet_pose.transform_point(local_position_xyz)
    return PoseSpec(
        position_xyz=tuple(float(value) for value in robot_base_position),
        quaternion_wxyz=cabinet_pose.quaternion_wxyz,
    )


def _furniture_front_robot_base_pose(
    furniture_pose: PoseSpec,
    model_id: str,
    scale: float,
    clearance: float = STUDIO_FURNITURE_ROBOT_CLEARANCE,
) -> PoseSpec:
    furniture_half_depth = 0.5 * PARTNET_BBOX_EXTENT_XYZ[model_id][0] * scale
    local_base_position = furniture_pose.transform_point(
        (-(furniture_half_depth + clearance), 0.0, 0.0)
    )
    return PoseSpec(
        position_xyz=(
            float(local_base_position[0]),
            float(local_base_position[1]),
            STUDIO_GROUND_Z,
        ),
        quaternion_wxyz=furniture_pose.quaternion_wxyz,
    )


def _partnet_top_z(model_id: str, scale: float) -> float:
    return STUDIO_GROUND_Z + PARTNET_BBOX_EXTENT_XYZ[model_id][2] * scale


STUDIO_DESK_WORK_SURFACE_Z = _partnet_top_z(
    STUDIO_DESK_MODEL_ID,
    STUDIO_DESK_MODEL_SCALE,
)
STUDIO_WORKBENCH_WORK_SURFACE_Z = _partnet_top_z(
    STUDIO_WORKBENCH_MODEL_ID,
    STUDIO_WORKBENCH_MODEL_SCALE,
)


def _frame_support_surface(
    surface_id: str,
    parent_entity_name: str,
    frame_pose: PoseSpec,
) -> LayoutSupportSurfaceSpec:
    return LayoutSupportSurfaceSpec(
        surface_id=surface_id,
        parent_entity_name=parent_entity_name,
        pose=frame_pose,
        local_height_z=frame_pose.position_xyz[2],
        initial_clearance_z=STUDIO_SMALL_OBJECT_INITIAL_CLEARANCE_Z,
    )


def _support_surface(
    surface_id: str,
    parent_entity_name: str,
    furniture_pose: PoseSpec,
    local_height_z: float,
    initial_clearance_z: float,
) -> LayoutSupportSurfaceSpec:
    return LayoutSupportSurfaceSpec(
        surface_id=surface_id,
        parent_entity_name=parent_entity_name,
        pose=PoseSpec(
            position_xyz=(
                furniture_pose.position_xyz[0],
                furniture_pose.position_xyz[1],
                STUDIO_GROUND_Z + local_height_z,
            ),
            quaternion_wxyz=furniture_pose.quaternion_wxyz,
        ),
        local_height_z=local_height_z,
        initial_clearance_z=initial_clearance_z,
    )


def _studio_support_surfaces(
    desk_surface_pose: PoseSpec,
    workbench_surface_pose: PoseSpec,
    storage_pose: PoseSpec,
    placement_spec: StudioLargeAssetPlacementSpec,
    include_storage_cabinet: bool,
) -> dict[str, LayoutSupportSurfaceSpec]:
    surface_definitions = (
        (
            STUDIO_DESK_SECOND_SHELF_SURFACE_ID,
            "desk",
            desk_surface_pose,
            STUDIO_DESK_SECOND_SHELF_Z,
        ),
        (
            STUDIO_STORAGE_TOP_SURFACE_ID,
            "storage_cabinet",
            storage_pose,
            _partnet_top_z(
                STUDIO_STORAGE_CABINET_MODEL_ID,
                STUDIO_STORAGE_CABINET_MODEL_SCALE,
            ),
        ),
        (
            STUDIO_OPEN_SHELF_TOP_SURFACE_ID,
            "open_shelf",
            placement_spec.open_shelf_pose,
            _partnet_top_z(
                STUDIO_OPEN_SHELF_MODEL_ID,
                STUDIO_OPEN_SHELF_MODEL_SCALE,
            )
            + STUDIO_OPEN_SHELF_USABLE_TOP_OFFSET_Z,
        ),
        (
            STUDIO_OPEN_SHELF_THIRD_LEVEL_SURFACE_ID,
            "open_shelf",
            placement_spec.open_shelf_pose,
            STUDIO_OPEN_SHELF_THIRD_LEVEL_Z,
        ),
        (
            STUDIO_OPEN_SHELF_SECOND_LEVEL_SURFACE_ID,
            "open_shelf",
            placement_spec.open_shelf_pose,
            STUDIO_OPEN_SHELF_SECOND_LEVEL_Z,
        ),
        (
            STUDIO_OPEN_SHELF_FIRST_LEVEL_SURFACE_ID,
            "open_shelf",
            placement_spec.open_shelf_pose,
            STUDIO_OPEN_SHELF_FIRST_LEVEL_Z,
        ),
        (
            STUDIO_DRESSER_TOP_SURFACE_ID,
            "dresser_cabinet_b",
            placement_spec.dresser_cabinet_b_pose,
            _partnet_top_z(
                STUDIO_DRESSER_CABINET_B_MODEL_ID,
                STUDIO_DRESSER_CABINET_B_MODEL_SCALE,
            ),
        ),
        (
            STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID,
            "dresser_cabinet_b",
            placement_spec.dresser_cabinet_b_pose,
            STUDIO_DRESSER_OUTER_COUNTER_Z,
        ),
        (
            STUDIO_DRESSER_LOWER_CABINET_UPPER_SURFACE_ID,
            "dresser_cabinet_b",
            placement_spec.dresser_cabinet_b_pose,
            STUDIO_DRESSER_LOWER_CABINET_UPPER_Z,
        ),
        (
            STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID,
            "dresser_cabinet_b",
            placement_spec.dresser_cabinet_b_pose,
            STUDIO_DRESSER_UPPER_CABINET_LOWER_Z,
        ),
        (
            STUDIO_DRESSER_UPPER_CABINET_UPPER_SURFACE_ID,
            "dresser_cabinet_b",
            placement_spec.dresser_cabinet_b_pose,
            STUDIO_DRESSER_UPPER_CABINET_UPPER_Z,
        ),
        (
            STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_SURFACE_ID,
            "sliding_cabinet_a",
            placement_spec.sliding_cabinet_a_pose,
            STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_Z,
        ),
        (
            STUDIO_SLIDING_CABINET_LOWER_CABINET_LOWER_SURFACE_ID,
            "sliding_cabinet_a",
            placement_spec.sliding_cabinet_a_pose,
            STUDIO_SLIDING_CABINET_LOWER_CABINET_LOWER_Z,
        ),
        (
            STUDIO_PRINTER_TABLE_BOTTOM_SHELF_SURFACE_ID,
            "printer_table",
            placement_spec.printer_table_pose,
            STUDIO_PRINTER_TABLE_BOTTOM_SHELF_Z,
        ),
    )
    automatically_aligned_surface_ids = {
        STUDIO_STORAGE_TOP_SURFACE_ID,
        STUDIO_OPEN_SHELF_TOP_SURFACE_ID,
        STUDIO_DRESSER_TOP_SURFACE_ID,
    }
    support_surfaces = {
        surface_id: _support_surface(
            surface_id=surface_id,
            parent_entity_name=parent_entity_name,
            furniture_pose=furniture_pose,
            local_height_z=local_height_z,
            initial_clearance_z=(
                0.0
                if surface_id in automatically_aligned_surface_ids
                else STUDIO_SMALL_OBJECT_INITIAL_CLEARANCE_Z
            ),
        )
        for surface_id, parent_entity_name, furniture_pose, local_height_z in (
            surface_definitions
        )
    }
    support_surfaces[STUDIO_DESK_TOP_SURFACE_ID] = _frame_support_surface(
        surface_id=STUDIO_DESK_TOP_SURFACE_ID,
        parent_entity_name="desk",
        frame_pose=desk_surface_pose,
    )
    support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID] = _frame_support_surface(
        surface_id=STUDIO_WORKBENCH_TOP_SURFACE_ID,
        parent_entity_name="workbench",
        frame_pose=workbench_surface_pose,
    )
    if not include_storage_cabinet:
        support_surfaces.pop(STUDIO_STORAGE_TOP_SURFACE_ID)
    return support_surfaces


def _dynamic_box_on_surface(
    name: str,
    surface: LayoutSupportSurfaceSpec,
    local_xy: tuple[float, float],
    half_size_xyz: tuple[float, float, float],
    color_rgba: tuple[float, float, float, float],
    episode_position_jitter_xy: tuple[float, float],
    episode_spawn_probability: float = 1.0,
    yaw: float = 0.0,
    initial_clearance_z: float | None = None,
) -> tuple[StaticBoxSpec, RelativePlacementSpec]:
    if initial_clearance_z is None:
        initial_clearance_z = surface.initial_clearance_z
    bbox_center_pose = pose(
        local_xy[0],
        local_xy[1],
        half_size_xyz[2] + initial_clearance_z,
        yaw,
    )
    world_bbox_center_pose = surface.pose.transform_pose(bbox_center_pose)
    build_item = StaticBoxSpec(
        name=name,
        half_size_xyz=half_size_xyz,
        pose=world_bbox_center_pose,
        color_rgba=color_rgba,
        body_type="dynamic",
        episode_position_jitter_xy=episode_position_jitter_xy,
        episode_spawn_probability=episode_spawn_probability,
        physical_material=STUDIO_PHYSICAL_MATERIAL,
    )
    return build_item, RelativePlacementSpec(
        item_name=name,
        surface_id=surface.surface_id,
        bbox_center_pose=bbox_center_pose,
        initial_clearance_z=initial_clearance_z,
    )


def _dynamic_ycb_on_surface(
    name: str,
    model_id: str,
    surface: LayoutSupportSurfaceSpec,
    local_xy: tuple[float, float],
    episode_position_jitter_xy: tuple[float, float],
    episode_spawn_probability: float = 1.0,
    yaw: float = 0.0,
    initial_clearance_z: float | None = None,
    scale_multiplier: float = 1.0,
) -> tuple[YCBActorSpec, RelativePlacementSpec]:
    if scale_multiplier <= 0.0:
        raise ValueError(
            f"YCB scale_multiplier for layout item {name!r} must be positive, "
            f"got {scale_multiplier}."
        )
    if initial_clearance_z is None:
        initial_clearance_z = surface.initial_clearance_z
    try:
        model_bbox_min_z = YCB_BBOX_MIN_Z[model_id]
        model_scale = YCB_MODEL_SCALE[model_id]
    except KeyError as exc:
        raise ValueError(
            f"Missing cached YCB bbox metadata for model {model_id!r}."
        ) from exc

    bbox_center_pose = pose(
        local_xy[0],
        local_xy[1],
        -model_bbox_min_z * model_scale * scale_multiplier + initial_clearance_z,
        yaw,
    )
    world_bbox_center_pose = surface.pose.transform_pose(bbox_center_pose)
    build_item = YCBActorSpec(
        name=name,
        model_id=model_id,
        pose=world_bbox_center_pose,
        scale_multiplier=scale_multiplier,
        body_type="dynamic",
        episode_position_jitter_xy=episode_position_jitter_xy,
        episode_spawn_probability=episode_spawn_probability,
        physical_material=STUDIO_PHYSICAL_MATERIAL,
    )
    return build_item, RelativePlacementSpec(
        item_name=name,
        surface_id=surface.surface_id,
        bbox_center_pose=bbox_center_pose,
        initial_clearance_z=initial_clearance_z,
    )


def _polyhaven_book_on_surface(
    name: str,
    surface: LayoutSupportSurfaceSpec,
    local_xy: tuple[float, float],
    model_to_surface_quaternion_wxyz: tuple[float, float, float, float],
) -> tuple[StaticMeshSpec, RelativePlacementSpec]:
    try:
        asset_relative_path = STUDIO_BOOK_ASSET_RELATIVE_PATHS[name]
        scale_xyz_tuple = STUDIO_BOOK_SCALE_XYZ[name]
        model_bbox_half_extent_xyz = np.asarray(
            STUDIO_BOOK_BBOX_HALF_EXTENT_XYZ[name],
            dtype=float,
        )
    except KeyError as exc:
        raise ValueError(f"Unknown Studio book asset {name!r}.") from exc

    scale_xyz = np.asarray(scale_xyz_tuple, dtype=float)
    model_to_surface_pose = PoseSpec(
        position_xyz=(0.0, 0.0, 0.0),
        quaternion_wxyz=model_to_surface_quaternion_wxyz,
    )

    # [3] -> [3]: scale the centered model-space bbox before shelf rotation.
    scaled_bbox_half_extent_xyz = model_bbox_half_extent_xyz * scale_xyz
    # [3, 3] @ [3] -> scalar: project the oriented bbox onto shelf-local Z.
    bbox_center_height = float(
        np.abs(model_to_surface_pose.rotation_matrix[2])
        @ scaled_bbox_half_extent_xyz
    )
    bbox_center_pose = PoseSpec(
        position_xyz=(
            local_xy[0],
            local_xy[1],
            bbox_center_height + surface.initial_clearance_z,
        ),
        quaternion_wxyz=model_to_surface_quaternion_wxyz,
    )
    world_root_pose = surface.pose.transform_pose(bbox_center_pose)

    build_item = StaticMeshSpec(
        name=name,
        mesh_path=evodex_asset_dir() / asset_relative_path,
        pose=world_root_pose,
        scale_xyz=scale_xyz_tuple,
        collision_type="nonconvex",
        body_type="kinematic",
        physical_material=STUDIO_PHYSICAL_MATERIAL,
    )
    return build_item, RelativePlacementSpec(
        item_name=build_item.name,
        surface_id=surface.surface_id,
        bbox_center_pose=bbox_center_pose,
        initial_clearance_z=surface.initial_clearance_z,
    )


def _partnet_root_pose_from_bbox_center(
    model_id: str,
    desired_center_pose: PoseSpec,
    scale: float,
    support_z: float | None = STUDIO_GROUND_Z,
) -> PoseSpec:
    bbox_center_xyz = PARTNET_BBOX_CENTER_XYZ[model_id]
    # [3, 3] @ [3] -> [3]: rotate the scaled model bbox center into layout space.
    bbox_center_offset_world = desired_center_pose.rotation_matrix @ (
        scale * np.asarray(bbox_center_xyz, dtype=float)
    )
    root_position_xyz = [
        desired_center_pose.position_xyz[axis_index] - float(bbox_center_offset_world[axis_index])
        for axis_index in range(3)
    ]
    if support_z is not None:
        root_position_xyz[2] = support_z - PARTNET_BBOX_MIN_Z[model_id] * scale
    return PoseSpec(
        position_xyz=(
            root_position_xyz[0],
            root_position_xyz[1],
            root_position_xyz[2],
        ),
        quaternion_wxyz=desired_center_pose.quaternion_wxyz,
    )


def _partnet_asset_on_surface(
    name: str,
    model_id: str,
    surface: LayoutSupportSurfaceSpec,
    local_xy: tuple[float, float],
    scale: float,
    initial_qpos: tuple[float, ...] | None,
    fix_root_link: bool,
    yaw: float = 0.0,
    local_quaternion_wxyz: tuple[float, float, float, float] | None = None,
    initial_clearance_z: float | None = None,
    disable_self_collisions: bool = False,
    joint_friction: dict[str, float] | None = None,
    joint_initial_qpos: dict[str, float] | None = None,
    joint_position_latches: dict[str, JointPositionLatchSpec] | None = None,
    link_disable_gravity: tuple[str, ...] = (),
    urdf_path_override: Path | None = None,
) -> tuple[ArticulatedAssetSpec, RelativePlacementSpec]:
    if scale <= 0.0:
        raise ValueError(
            f"PartNet scale for layout item {name!r} must be positive, got {scale}."
        )
    if local_quaternion_wxyz is not None and yaw != 0.0:
        raise ValueError(
            f"PartNet layout item {name!r} cannot specify both yaw and "
            "local_quaternion_wxyz."
        )
    if local_quaternion_wxyz is not None:
        quaternion_norm = math.sqrt(
            sum(component * component for component in local_quaternion_wxyz)
        )
        if not math.isclose(quaternion_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"PartNet local quaternion for layout item {name!r} must be "
                f"normalized, got norm {quaternion_norm}."
            )
    if initial_clearance_z is None:
        initial_clearance_z = surface.initial_clearance_z
    local_orientation = (
        PoseSpec(
            position_xyz=(0.0, 0.0, 0.0),
            quaternion_wxyz=local_quaternion_wxyz,
        )
        if local_quaternion_wxyz is not None
        else pose(0.0, 0.0, 0.0, yaw)
    )
    scaled_bbox_half_extent_xyz = (
        0.5 * scale * np.asarray(PARTNET_BBOX_EXTENT_XYZ[model_id], dtype=float)
    )
    # [3] @ [3] -> scalar: project the oriented bbox half extents onto surface Z.
    bbox_center_height = float(
        np.abs(local_orientation.rotation_matrix[2]) @ scaled_bbox_half_extent_xyz
    )
    bbox_center_pose = PoseSpec(
        position_xyz=(
            local_xy[0],
            local_xy[1],
            bbox_center_height + initial_clearance_z,
        ),
        quaternion_wxyz=local_orientation.quaternion_wxyz,
    )
    world_bbox_center_pose = surface.pose.transform_pose(bbox_center_pose)
    build_item = ArticulatedAssetSpec(
        name=name,
        asset_path=(
            urdf_path_override
            if urdf_path_override is not None
            else partnet_mobility_model_urdf_path(model_id)
        ),
        pose=_partnet_root_pose_from_bbox_center(
            model_id=model_id,
            desired_center_pose=world_bbox_center_pose,
            scale=scale,
            support_z=None,
        ),
        scale=scale,
        fix_root_link=fix_root_link,
        load_multiple_collisions_from_file=True,
        disable_self_collisions=disable_self_collisions,
        initial_qpos=initial_qpos,
        joint_initial_qpos=(
            dict(joint_initial_qpos)
            if joint_initial_qpos is not None
            else {}
        ),
        episode_position_jitter_xy=(0.0, 0.0),
        default_joint_friction=STUDIO_DEFAULT_JOINT_FRICTION,
        joint_friction=(
            dict(joint_friction) if joint_friction is not None else {}
        ),
        joint_position_latches=(
            dict(joint_position_latches)
            if joint_position_latches is not None
            else {}
        ),
        link_disable_gravity=link_disable_gravity,
        physical_material=STUDIO_PHYSICAL_MATERIAL,
    )
    return build_item, RelativePlacementSpec(
        item_name=name,
        surface_id=surface.surface_id,
        bbox_center_pose=bbox_center_pose,
        initial_clearance_z=initial_clearance_z,
    )


def _fixed_partnet_asset(
    name: str,
    model_id: str,
    pose: PoseSpec,
    scale: float,
    initial_qpos: tuple[float, ...] | None = None,
    episode_position_jitter_xy: tuple[float, float] | None = None,
    urdf_path_override: str | Path | None = None,
) -> ArticulatedAssetSpec:
    return ArticulatedAssetSpec(
        name=name,
        asset_path=(
            urdf_path_override
            if urdf_path_override is not None
            else partnet_mobility_model_urdf_path(model_id)
        ),
        pose=pose,
        scale=scale,
        fix_root_link=True,
        load_multiple_collisions_from_file=True,
        disable_self_collisions=True,
        initial_qpos=initial_qpos,
        episode_position_jitter_xy=episode_position_jitter_xy,
        default_joint_friction=STUDIO_DEFAULT_JOINT_FRICTION,
        physical_material=STUDIO_PHYSICAL_MATERIAL,
    )


def _textured_wall_segment(
    name: str,
    texture_path: str,
    center_xyz: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    half_width_m: float,
    half_height_m: float,
    texture_repeat_uv: tuple[float, float],
) -> TexturedPlaneSpec:
    return textured_plane(
        name=name,
        texture_path=robocasa_texture_path(texture_path),
        half_size_xy=(half_width_m, half_height_m),
        pose=PoseSpec(
            position_xyz=center_xyz,
            quaternion_wxyz=quaternion_wxyz,
        ),
        texture_repeat_uv=texture_repeat_uv,
        collision_half_size_xyz=(
            half_width_m,
            half_height_m,
            STUDIO_WALL_COLLISION_THICKNESS,
        ),
    )


def _studio_room_shell_items(
    texture_spec: StudioRoomTextureSpec,
) -> list[TexturedPlaneSpec | ArticulatedAssetSpec]:
    room_min_x = STUDIO_ROOM_CENTER_X - STUDIO_ROOM_HALF_SIZE_X
    room_max_x = STUDIO_ROOM_CENTER_X + STUDIO_ROOM_HALF_SIZE_X
    room_min_y = STUDIO_ROOM_CENTER_Y - STUDIO_ROOM_HALF_SIZE_Y
    room_max_y = STUDIO_ROOM_CENTER_Y + STUDIO_ROOM_HALF_SIZE_Y
    wall_center_z = STUDIO_ROOM_HEIGHT / 2.0

    door_center_x = -1.35
    door_width = (
        PARTNET_BBOX_EXTENT_XYZ[STUDIO_DOOR_MODEL_ID][1] * STUDIO_DOOR_MODEL_SCALE
    )
    door_height = (
        PARTNET_BBOX_EXTENT_XYZ[STUDIO_DOOR_MODEL_ID][2] * STUDIO_DOOR_MODEL_SCALE
    )
    door_min_x = door_center_x - door_width / 2.0
    door_max_x = door_center_x + door_width / 2.0

    window_center_y = 0.80
    window_width = (
        PARTNET_BBOX_EXTENT_XYZ[STUDIO_WINDOW_MODEL_ID][1] * STUDIO_WINDOW_MODEL_SCALE
    )
    window_bottom_z = 1.05
    window_height = (
        PARTNET_BBOX_EXTENT_XYZ[STUDIO_WINDOW_MODEL_ID][2] * STUDIO_WINDOW_MODEL_SCALE
    )
    window_top_z = window_bottom_z + window_height
    window_min_y = window_center_y - window_width / 2.0
    window_max_y = window_center_y + window_width / 2.0

    return [
        textured_plane(
            "floor_texture",
            robocasa_texture_path(texture_spec.floor_texture),
            (STUDIO_ROOM_HALF_SIZE_X, STUDIO_ROOM_HALF_SIZE_Y),
            PoseSpec(
                position_xyz=(STUDIO_ROOM_CENTER_X, STUDIO_ROOM_CENTER_Y, 0.002),
            ),
            texture_spec.floor_repeat_uv,
            visual_pose=PoseSpec(
                position_xyz=(0.0, 0.0, 0.0),
                quaternion_wxyz=(0.0, 0.0, 1.0, 0.0),
            ),
        ),
        textured_plane(
            "ceiling_texture",
            robocasa_texture_path(texture_spec.ceiling_texture),
            (STUDIO_ROOM_HALF_SIZE_X, STUDIO_ROOM_HALF_SIZE_Y),
            PoseSpec(
                position_xyz=(
                    STUDIO_ROOM_CENTER_X,
                    STUDIO_ROOM_CENTER_Y,
                    STUDIO_ROOM_HEIGHT,
                )
            ),
            texture_spec.ceiling_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_back_full",
            texture_spec.wall_texture,
            (STUDIO_ROOM_CENTER_X, room_max_y, wall_center_z),
            WALL_BACK_QUATERNION_WXYZ,
            STUDIO_ROOM_HALF_SIZE_X,
            STUDIO_ROOM_HEIGHT / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_left_full",
            texture_spec.wall_texture,
            (room_min_x, STUDIO_ROOM_CENTER_Y, wall_center_z),
            WALL_LEFT_QUATERNION_WXYZ,
            STUDIO_ROOM_HALF_SIZE_Y,
            STUDIO_ROOM_HEIGHT / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_front_door_left",
            texture_spec.wall_texture,
            (
                (room_min_x + door_min_x) / 2.0,
                room_min_y,
                wall_center_z,
            ),
            WALL_FRONT_QUATERNION_WXYZ,
            (door_min_x - room_min_x) / 2.0,
            STUDIO_ROOM_HEIGHT / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_front_door_right",
            texture_spec.wall_texture,
            (
                (door_max_x + room_max_x) / 2.0,
                room_min_y,
                wall_center_z,
            ),
            WALL_FRONT_QUATERNION_WXYZ,
            (room_max_x - door_max_x) / 2.0,
            STUDIO_ROOM_HEIGHT / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_front_door_header",
            texture_spec.wall_texture,
            (
                door_center_x,
                room_min_y,
                (door_height + STUDIO_ROOM_HEIGHT) / 2.0,
            ),
            WALL_FRONT_QUATERNION_WXYZ,
            door_width / 2.0,
            (STUDIO_ROOM_HEIGHT - door_height) / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_right_window_front",
            texture_spec.wall_texture,
            (
                room_max_x,
                (room_min_y + window_min_y) / 2.0,
                wall_center_z,
            ),
            WALL_RIGHT_QUATERNION_WXYZ,
            (window_min_y - room_min_y) / 2.0,
            STUDIO_ROOM_HEIGHT / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_right_window_back",
            texture_spec.wall_texture,
            (
                room_max_x,
                (window_max_y + room_max_y) / 2.0,
                wall_center_z,
            ),
            WALL_RIGHT_QUATERNION_WXYZ,
            (room_max_y - window_max_y) / 2.0,
            STUDIO_ROOM_HEIGHT / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_right_window_sill",
            texture_spec.wall_texture,
            (
                room_max_x,
                window_center_y,
                window_bottom_z / 2.0,
            ),
            WALL_RIGHT_QUATERNION_WXYZ,
            window_width / 2.0,
            window_bottom_z / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _textured_wall_segment(
            "wall_right_window_header",
            texture_spec.wall_texture,
            (
                room_max_x,
                window_center_y,
                (window_top_z + STUDIO_ROOM_HEIGHT) / 2.0,
            ),
            WALL_RIGHT_QUATERNION_WXYZ,
            window_width / 2.0,
            (STUDIO_ROOM_HEIGHT - window_top_z) / 2.0,
            texture_spec.wall_repeat_uv,
        ),
        _fixed_partnet_asset(
            name="door_panel",
            model_id=STUDIO_DOOR_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_DOOR_MODEL_ID,
                desired_center_pose=PoseSpec(
                    position_xyz=(
                        door_center_x,
                        room_min_y,
                        door_height / 2.0,
                    ),
                    quaternion_wxyz=pose(
                        0.0,
                        0.0,
                        0.0,
                        STUDIO_DOOR_CLOSED_YAW,
                    ).quaternion_wxyz,
                ),
                scale=STUDIO_DOOR_MODEL_SCALE,
            ),
            scale=STUDIO_DOOR_MODEL_SCALE,
            initial_qpos=STUDIO_DOOR_CLOSED_QPOS,
        ),
        _fixed_partnet_asset(
            name="window",
            model_id=STUDIO_WINDOW_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_WINDOW_MODEL_ID,
                desired_center_pose=PoseSpec(
                    position_xyz=(
                        room_max_x,
                        window_center_y,
                        (window_bottom_z + window_top_z) / 2.0,
                    ),
                ),
                scale=STUDIO_WINDOW_MODEL_SCALE,
                support_z=None,
            ),
            scale=STUDIO_WINDOW_MODEL_SCALE,
        ),
    ]


def _studio_large_asset_items(
    desk_pose: PoseSpec,
    placement_spec: StudioLargeAssetPlacementSpec,
) -> list[LayoutBuildItemSpec]:
    chair_position = desk_pose.transform_point(
        STUDIO_DESK_CHAIR_LOCAL_POSITION_XYZ
    )
    printer_table_top_z = (
        PARTNET_BBOX_EXTENT_XYZ[STUDIO_PRINTER_TABLE_MODEL_ID][2]
        * STUDIO_PRINTER_TABLE_MODEL_SCALE
    )
    printer_table_xy = placement_spec.printer_table_pose.position_xyz[:2]
    printer_pose = placement_spec.printer_pose or PoseSpec(
        position_xyz=(
            printer_table_xy[0],
            printer_table_xy[1],
            printer_table_top_z + 0.17,
        ),
        quaternion_wxyz=placement_spec.printer_table_pose.quaternion_wxyz,
    )
    if placement_spec.ceiling_light_mesh_relative_path is not None:
        ceiling_light_item: LayoutBuildItemSpec = StaticMeshSpec(
            name="ceiling_light",
            mesh_path=(
                evodex_asset_dir()
                / placement_spec.ceiling_light_mesh_relative_path
            ),
            pose=PoseSpec(
                position_xyz=(
                    placement_spec.ceiling_light_xy[0],
                    placement_spec.ceiling_light_xy[1],
                    placement_spec.ceiling_light_root_z,
                ),
                quaternion_wxyz=placement_spec.ceiling_light_quaternion_wxyz,
            ),
            scale_xyz=placement_spec.ceiling_light_scale_xyz,
            collision_type="none",
            body_type="static",
        )
    else:
        ceiling_light_item = StaticBoxSpec(
            name="ceiling_light_placeholder",
            half_size_xyz=(0.16, 0.16, 0.06),
            pose=PoseSpec(
                position_xyz=(
                    placement_spec.ceiling_light_xy[0],
                    placement_spec.ceiling_light_xy[1],
                    STUDIO_CEILING_LIGHT_Z,
                ),
            ),
            color_rgba=(0.95, 0.86, 0.58, 1.0),
            add_collision=False,
        )
    return [
        _fixed_partnet_asset(
            name="desk_chair",
            model_id=STUDIO_DESK_CHAIR_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_DESK_CHAIR_MODEL_ID,
                desired_center_pose=PoseSpec(
                    position_xyz=(
                        float(chair_position[0]),
                        float(chair_position[1]),
                        float(chair_position[2]),
                    ),
                    quaternion_wxyz=desk_pose.quaternion_wxyz,
                ),
                scale=STUDIO_DESK_CHAIR_MODEL_SCALE,
            ),
            scale=STUDIO_DESK_CHAIR_MODEL_SCALE,
            initial_qpos=placement_spec.desk_chair_initial_qpos,
        ),
        ceiling_light_item,
        _fixed_partnet_asset(
            name="printer_table",
            model_id=STUDIO_PRINTER_TABLE_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_PRINTER_TABLE_MODEL_ID,
                desired_center_pose=placement_spec.printer_table_pose,
                scale=STUDIO_PRINTER_TABLE_MODEL_SCALE,
            ),
            scale=STUDIO_PRINTER_TABLE_MODEL_SCALE,
        ),
        _fixed_partnet_asset(
            name="printer",
            model_id=STUDIO_PRINTER_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_PRINTER_MODEL_ID,
                desired_center_pose=printer_pose,
                scale=STUDIO_PRINTER_MODEL_SCALE,
                support_z=printer_table_top_z,
            ),
            scale=STUDIO_PRINTER_MODEL_SCALE,
        ),
        _fixed_partnet_asset(
            name="sliding_cabinet_a",
            model_id=STUDIO_SLIDING_CABINET_A_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_SLIDING_CABINET_A_MODEL_ID,
                desired_center_pose=placement_spec.sliding_cabinet_a_pose,
                scale=STUDIO_SLIDING_CABINET_A_MODEL_SCALE,
            ),
            scale=STUDIO_SLIDING_CABINET_A_MODEL_SCALE,
        ),
        _fixed_partnet_asset(
            name="open_shelf",
            model_id=STUDIO_OPEN_SHELF_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_OPEN_SHELF_MODEL_ID,
                desired_center_pose=placement_spec.open_shelf_pose,
                scale=STUDIO_OPEN_SHELF_MODEL_SCALE,
            ),
            scale=STUDIO_OPEN_SHELF_MODEL_SCALE,
        ),
        _fixed_partnet_asset(
            name="dresser_cabinet_b",
            model_id=STUDIO_DRESSER_CABINET_B_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_DRESSER_CABINET_B_MODEL_ID,
                desired_center_pose=placement_spec.dresser_cabinet_b_pose,
                scale=STUDIO_DRESSER_CABINET_B_MODEL_SCALE,
            ),
            scale=STUDIO_DRESSER_CABINET_B_MODEL_SCALE,
        ),
    ]


def _studio_relative_content_items(
    support_surfaces: dict[str, LayoutSupportSurfaceSpec],
    placement_overrides: dict[
        str, StudioRelativeContentPlacementOverride
    ] | None = None,
) -> tuple[list[LayoutBuildItemSpec], dict[str, RelativePlacementSpec]]:
    item_placement_pairs: list[
        tuple[LayoutBuildItemSpec, RelativePlacementSpec]
    ] = [
        _partnet_asset_on_surface(
            name="desk_lamp",
            model_id=STUDIO_DESK_LAMP_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_TOP_SURFACE_ID],
            local_xy=(0.12, 0.54),
            scale=STUDIO_DESK_LAMP_MODEL_SCALE,
            initial_qpos=STUDIO_DESK_LAMP_INITIAL_QPOS,
            fix_root_link=True,
            initial_clearance_z=0.0,
            disable_self_collisions=True,
            joint_friction={
                "joint_1": STUDIO_DESK_LAMP_HEAD_JOINT_FRICTION,
            },
        ),
        _partnet_asset_on_surface(
            name="laptop",
            model_id=STUDIO_LAPTOP_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_TOP_SURFACE_ID],
            local_xy=(0.01, -0.09),
            scale=STUDIO_LAPTOP_MODEL_SCALE,
            initial_qpos=STUDIO_LAPTOP_INITIAL_QPOS,
            fix_root_link=False,
            urdf_path_override=partnet_mobility_modified_model_urdf_path(
                STUDIO_LAPTOP_MODEL_ID
            ),
            initial_clearance_z=STUDIO_LAPTOP_INITIAL_CLEARANCE_Z,
            joint_friction={
                "joint_1": STUDIO_LAPTOP_SCREEN_JOINT_FRICTION,
            },
            joint_position_latches={
                "joint_1": STUDIO_LAPTOP_SCREEN_JOINT_LATCH,
            },
        ),
        _partnet_asset_on_surface(
            name="display",
            model_id=STUDIO_DISPLAY_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_TOP_SURFACE_ID],
            local_xy=(0.28, -0.09),
            scale=STUDIO_DISPLAY_MODEL_SCALE,
            initial_qpos=None,
            fix_root_link=False,
            joint_initial_qpos={
                "joint_2": STUDIO_DISPLAY_HEIGHT_JOINT_QPOS,
            },
            link_disable_gravity=("link_2", "link_0"),
        ),
        _partnet_asset_on_surface(
            name="mouse",
            model_id=STUDIO_MOUSE_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_TOP_SURFACE_ID],
            local_xy=(-0.02034, -0.490602),
            scale=STUDIO_MOUSE_MODEL_SCALE,
            initial_qpos=STUDIO_MOUSE_INITIAL_QPOS,
            fix_root_link=False,
            yaw=3.03,
        ),
        _partnet_asset_on_surface(
            name="phone",
            model_id=STUDIO_PHONE_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_TOP_SURFACE_ID],
            local_xy=(0.12, 0.30),
            scale=STUDIO_PHONE_MODEL_SCALE,
            initial_qpos=STUDIO_PHONE_INITIAL_QPOS,
            fix_root_link=False,
            local_quaternion_wxyz=STUDIO_PHONE_FLAT_LOCAL_QUATERNION_WXYZ,
            initial_clearance_z=STUDIO_PHONE_INITIAL_CLEARANCE_Z,
        ),
        _partnet_asset_on_surface(
            name="usb",
            model_id=STUDIO_USB_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_TOP_SURFACE_ID],
            local_xy=(-0.28, 0.16),
            scale=STUDIO_USB_MODEL_SCALE,
            initial_qpos=STUDIO_USB_INITIAL_QPOS,
            fix_root_link=False,
            yaw=0.12,
        ),
        _partnet_asset_on_surface(
            name="workbench_box",
            model_id=STUDIO_WORKBENCH_BOX_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(0.12, 0.0),
            scale=STUDIO_WORKBENCH_BOX_MODEL_SCALE,
            initial_qpos=STUDIO_WORKBENCH_BOX_INITIAL_QPOS,
            fix_root_link=False,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_large_marker",
            model_id=STUDIO_LARGE_MARKER_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(-0.20, 0.22),
            episode_position_jitter_xy=(0.008, 0.008),
            yaw=0.20,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_adjustable_wrench",
            model_id=STUDIO_ADJUSTABLE_WRENCH_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(-0.06, 0.423077),
            episode_position_jitter_xy=(0.012, 0.012),
            yaw=-0.18,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_power_drill",
            model_id=STUDIO_POWER_DRILL_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(-0.18, -0.32),
            episode_position_jitter_xy=(0.010, 0.010),
            yaw=0.18,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_phillips_screwdriver",
            model_id=STUDIO_PHILLIPS_SCREWDRIVER_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(0.20, -0.55),
            episode_position_jitter_xy=(0.008, 0.008),
            yaw=0.10,
        ),
        _partnet_asset_on_surface(
            name="workbench_pliers",
            model_id=STUDIO_WORKBENCH_PLIERS_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(0.20, 0.35),
            scale=STUDIO_WORKBENCH_PLIERS_MODEL_SCALE,
            initial_qpos=STUDIO_WORKBENCH_PLIERS_INITIAL_QPOS,
            fix_root_link=False,
            yaw=-0.12,
        ),
        _partnet_asset_on_surface(
            name="glasses",
            model_id=STUDIO_GLASSES_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(0.22, 0.18),
            scale=STUDIO_GLASSES_MODEL_SCALE,
            initial_qpos=STUDIO_GLASSES_INITIAL_QPOS,
            fix_root_link=False,
        ),
        _dynamic_box_on_surface(
            name="clutter_paper_placeholder",
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(0.24, -0.18),
            half_size_xyz=(0.105, 0.075, 0.002),
            color_rgba=(0.92, 0.91, 0.86, 1.0),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.08,
        ),
        _partnet_asset_on_surface(
            name="stapler",
            model_id=STUDIO_STAPLER_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(-0.28, -0.05),
            scale=STUDIO_STAPLER_MODEL_SCALE,
            initial_qpos=STUDIO_STAPLER_INITIAL_QPOS,
            fix_root_link=False,
            yaw=0.05,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_dice",
            model_id=STUDIO_DICE_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_WORKBENCH_TOP_SURFACE_ID],
            local_xy=(0.25, -0.36),
            episode_position_jitter_xy=(0.008, 0.008),
            yaw=-0.08,
        ),
        *(
            _polyhaven_book_on_surface(
                name=book_name,
                surface=support_surfaces[
                    STUDIO_OPEN_SHELF_THIRD_LEVEL_SURFACE_ID
                ],
                local_xy=STUDIO_BOOK_LOCAL_XY[book_name],
                model_to_surface_quaternion_wxyz=(
                    STUDIO_FLAT_BOOK_MODEL_TO_SURFACE_QUATERNION_WXYZ
                    if book_name == "book_travel_flat"
                    else STUDIO_UPRIGHT_BOOK_MODEL_TO_SURFACE_QUATERNION_WXYZ
                ),
            )
            for book_name in (
                "book_calculus",
                "book_chrysanthemums",
                "book_space",
                "book_travel_flat",
            )
        ),
        _dynamic_ycb_on_surface(
            name="clutter_toy_airplane",
            model_id=STUDIO_TOY_AIRPLANE_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_OPEN_SHELF_SECOND_LEVEL_SURFACE_ID],
            local_xy=(0.0, -0.16),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.10,
            scale_multiplier=0.6,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_rubiks_cube",
            model_id=STUDIO_RUBIKS_CUBE_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_OPEN_SHELF_SECOND_LEVEL_SURFACE_ID],
            local_xy=(0.02, -0.32),
            episode_position_jitter_xy=(0.008, 0.008),
            yaw=0.08,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_mug",
            model_id=STUDIO_MUG_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_OPEN_SHELF_FIRST_LEVEL_SURFACE_ID],
            local_xy=(0.0, 0.25),
            episode_position_jitter_xy=(0.008, 0.008),
            yaw=-0.10,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_lego_duplo",
            model_id=STUDIO_LEGO_DUPLO_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_OPEN_SHELF_FIRST_LEVEL_SURFACE_ID],
            local_xy=(0.0, 0.0),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.12,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_apple",
            model_id=STUDIO_APPLE_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_OPEN_SHELF_FIRST_LEVEL_SURFACE_ID],
            local_xy=(0.0, -0.22),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_box_on_surface(
            name="clutter_card_placeholder",
            surface=support_surfaces[STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID],
            local_xy=(0.0, 0.32),
            half_size_xyz=(0.045, 0.030, 0.003),
            color_rgba=(0.93, 0.88, 0.68, 1.0),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.16,
        ),
        _partnet_asset_on_surface(
            name="remote",
            model_id=STUDIO_REMOTE_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_SECOND_SHELF_SURFACE_ID],
            local_xy=(-0.10, 0.596162),
            scale=STUDIO_REMOTE_MODEL_SCALE,
            initial_qpos=STUDIO_REMOTE_INITIAL_QPOS,
            fix_root_link=False,
            local_quaternion_wxyz=(
                math.sqrt(0.5),
                0.0,
                math.sqrt(0.5),
                0.0,
            ),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_cup",
            model_id=STUDIO_CUP_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_DRESSER_OUTER_COUNTER_SURFACE_ID],
            local_xy=(0.0, 0.0),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_master_chef_can",
            model_id=STUDIO_MASTER_CHEF_CAN_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_UPPER_SURFACE_ID
            ],
            local_xy=(0.0, 0.25),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_cracker_box",
            model_id=STUDIO_CRACKER_BOX_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_UPPER_SURFACE_ID
            ],
            local_xy=(0.0, 0.0),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.05,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_tomato_soup_can",
            model_id=STUDIO_TOMATO_SOUP_CAN_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_UPPER_SURFACE_ID
            ],
            local_xy=(0.0, -0.25),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_mustard_bottle",
            model_id=STUDIO_MUSTARD_BOTTLE_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID
            ],
            local_xy=(0.0, 0.33),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_pudding_box",
            model_id=STUDIO_PUDDING_BOX_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID
            ],
            local_xy=(0.0, 0.11),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=-0.04,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_potted_meat_can",
            model_id=STUDIO_POTTED_MEAT_CAN_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID
            ],
            local_xy=(0.0, -0.11),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_bleach_cleanser",
            model_id=STUDIO_BLEACH_CLEANSER_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_DRESSER_UPPER_CABINET_LOWER_SURFACE_ID
            ],
            local_xy=(0.0, -0.33),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_mini_soccer_ball",
            model_id=STUDIO_MINI_SOCCER_BALL_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_SURFACE_ID
            ],
            local_xy=(-0.08, -0.10),
            episode_position_jitter_xy=(0.006, 0.006),
            scale_multiplier=1.5,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_tennis_ball",
            model_id=STUDIO_TENNIS_BALL_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_SURFACE_ID
            ],
            local_xy=(0.08, -0.12),
            episode_position_jitter_xy=(0.006, 0.006),
        ),
        _dynamic_ycb_on_surface(
            name="clutter_chain",
            model_id=STUDIO_CHAIN_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_SLIDING_CABINET_LOWER_CABINET_UPPER_SURFACE_ID
            ],
            local_xy=(0.0, -0.26),
            episode_position_jitter_xy=(0.004, 0.004),
            yaw=0.04,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_medium_clamp",
            model_id=STUDIO_MEDIUM_CLAMP_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_SLIDING_CABINET_LOWER_CABINET_LOWER_SURFACE_ID
            ],
            local_xy=(0.0, -0.12),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=-0.08,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_foam_brick",
            model_id=STUDIO_FOAM_BRICK_YCB_MODEL_ID,
            surface=support_surfaces[
                STUDIO_SLIDING_CABINET_LOWER_CABINET_LOWER_SURFACE_ID
            ],
            local_xy=(0.0, -0.28),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.10,
        ),
        _dynamic_ycb_on_surface(
            name="clutter_banana",
            model_id=STUDIO_BANANA_YCB_MODEL_ID,
            surface=support_surfaces[STUDIO_DESK_SECOND_SHELF_SURFACE_ID],
            local_xy=(0.086741, 0.596162),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=0.10,
            initial_clearance_z=0.02193120427470382,
        ),
        _dynamic_box_on_surface(
            name="clutter_printer_card_placeholder",
            surface=support_surfaces[STUDIO_PRINTER_TABLE_BOTTOM_SHELF_SURFACE_ID],
            local_xy=(0.0, 0.0),
            half_size_xyz=(0.210, 0.115, 0.002),
            color_rgba=(0.94, 0.94, 0.90, 1.0),
            episode_position_jitter_xy=(0.006, 0.006),
            yaw=-0.10,
        ),
    ]
    if placement_overrides:
        item_placement_pairs_by_name = {
            placement.item_name: (build_item, placement)
            for build_item, placement in item_placement_pairs
        }
        unknown_item_names = (
            placement_overrides.keys() - item_placement_pairs_by_name.keys()
        )
        if unknown_item_names:
            raise ValueError(
                "Unknown Studio relative content placement overrides: "
                + ", ".join(sorted(unknown_item_names))
            )

        overridden_item_placement_pairs: list[
            tuple[LayoutBuildItemSpec, RelativePlacementSpec]
        ] = []
        for build_item, placement in item_placement_pairs:
            placement_override = placement_overrides.get(placement.item_name)
            if placement_override is None:
                overridden_item_placement_pairs.append((build_item, placement))
                continue

            current_bbox_center_pose = placement.bbox_center_pose
            local_quaternion_wxyz = (
                pose(0.0, 0.0, 0.0, placement_override.yaw).quaternion_wxyz
                if placement_override.yaw is not None
                else current_bbox_center_pose.quaternion_wxyz
            )
            overridden_bbox_center_pose = PoseSpec(
                position_xyz=(
                    placement_override.local_xy[0],
                    placement_override.local_xy[1],
                    (
                        placement_override.bbox_center_z
                        if placement_override.bbox_center_z is not None
                        else current_bbox_center_pose.position_xyz[2]
                    ),
                ),
                quaternion_wxyz=local_quaternion_wxyz,
            )
            overridden_surface_id = (
                placement_override.surface_id
                if placement_override.surface_id is not None
                else placement.surface_id
            )
            if overridden_surface_id not in support_surfaces:
                raise ValueError(
                    "Unknown Studio support surface in relative content "
                    f"placement override for {placement.item_name!r}: "
                    f"{overridden_surface_id!r}."
                )
            support_surface = support_surfaces[overridden_surface_id]
            world_bbox_center_pose = support_surface.pose.transform_pose(
                overridden_bbox_center_pose
            )
            if isinstance(build_item, ArticulatedAssetSpec):
                model_id = Path(build_item.asset_path).expanduser().parent.name
                overridden_build_item: LayoutBuildItemSpec = replace(
                    build_item,
                    pose=_partnet_root_pose_from_bbox_center(
                        model_id=model_id,
                        desired_center_pose=world_bbox_center_pose,
                        scale=build_item.scale,
                        support_z=None,
                    ),
                )
            elif isinstance(build_item, (StaticBoxSpec, YCBActorSpec)):
                overridden_build_item = replace(
                    build_item,
                    pose=world_bbox_center_pose,
                )
            else:
                raise TypeError(
                    "Studio relative content placement override does not support "
                    f"layout item type {type(build_item).__name__!r} for "
                    f"{build_item.name!r}."
                )
            overridden_item_placement_pairs.append(
                (
                    overridden_build_item,
                    replace(
                        placement,
                        surface_id=overridden_surface_id,
                        bbox_center_pose=overridden_bbox_center_pose,
                    ),
                )
            )
        item_placement_pairs = overridden_item_placement_pairs

    relative_placements = {
        placement.item_name: placement for _, placement in item_placement_pairs
    }
    if len(relative_placements) != len(item_placement_pairs):
        raise ValueError("Studio relative content item names must be unique.")
    return (
        [build_item for build_item, _ in item_placement_pairs],
        relative_placements,
    )


def _studio_room_decor_items(
    placement_spec: StudioRoomDecorPlacementSpec,
) -> list[LayoutBuildItemSpec]:
    return [
        textured_plane(
            name="rug_texture",
            texture_path=(
                evodex_asset_dir() / placement_spec.rug_texture_relative_path
            ),
            half_size_xy=(0.65, 0.45),
            pose=placement_spec.rug_pose,
            texture_repeat_uv=(3.25, 2.25),
            visual_pose=PoseSpec(
                position_xyz=(0.0, 0.0, 0.0),
                quaternion_wxyz=(0.0, 0.0, 1.0, 0.0),
            ),
        ),
        MJCFActorSpec(
            name="potted_plant",
            asset_path=robocasa_asset_path(
                placement_spec.potted_plant_robocasa_asset
            ),
            pose=placement_spec.potted_plant_root_pose,
            scale_xyz=placement_spec.potted_plant_scale_xyz,
            visual_groups=(1,),
            body_type="static",
        ),
        StaticMeshSpec(
            name="coat_rack",
            mesh_path=(
                evodex_asset_dir()
                / placement_spec.coat_rack_asset_relative_path
            ),
            pose=placement_spec.coat_rack_root_pose,
            scale_xyz=placement_spec.coat_rack_scale_xyz,
            collision_type="nonconvex",
            body_type="static",
            physical_material=STUDIO_PHYSICAL_MATERIAL,
        ),
        _fixed_partnet_asset(
            name="wall_clock",
            model_id=STUDIO_WALL_CLOCK_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_WALL_CLOCK_MODEL_ID,
                desired_center_pose=placement_spec.wall_clock_bbox_center_pose,
                scale=STUDIO_WALL_CLOCK_MODEL_SCALE,
                support_z=None,
            ),
            scale=STUDIO_WALL_CLOCK_MODEL_SCALE,
            initial_qpos=STUDIO_WALL_CLOCK_INITIAL_QPOS,
            episode_position_jitter_xy=(0.0, 0.0),
        ),
        _fixed_partnet_asset(
            name="wall_switch",
            model_id=STUDIO_WALL_SWITCH_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_WALL_SWITCH_MODEL_ID,
                desired_center_pose=placement_spec.wall_switch_bbox_center_pose,
                scale=STUDIO_WALL_SWITCH_MODEL_SCALE,
                support_z=None,
            ),
            scale=STUDIO_WALL_SWITCH_MODEL_SCALE,
            initial_qpos=STUDIO_WALL_SWITCH_INITIAL_QPOS,
            episode_position_jitter_xy=(0.0, 0.0),
        ),
        _fixed_partnet_asset(
            name="trash_can",
            model_id=STUDIO_TRASH_CAN_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_TRASH_CAN_MODEL_ID,
                desired_center_pose=placement_spec.trash_can_bbox_center_pose,
                scale=STUDIO_TRASH_CAN_MODEL_SCALE,
            ),
            scale=STUDIO_TRASH_CAN_MODEL_SCALE,
            initial_qpos=STUDIO_TRASH_CAN_INITIAL_QPOS,
            episode_position_jitter_xy=(0.0, 0.0),
        ),
        _fixed_partnet_asset(
            name="wall_fan",
            model_id=STUDIO_WALL_FAN_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_WALL_FAN_MODEL_ID,
                desired_center_pose=placement_spec.wall_fan_bbox_center_pose,
                scale=STUDIO_WALL_FAN_MODEL_SCALE,
                support_z=None,
            ),
            scale=STUDIO_WALL_FAN_MODEL_SCALE,
            initial_qpos=STUDIO_WALL_FAN_INITIAL_QPOS,
            episode_position_jitter_xy=(0.0, 0.0),
        ),
    ]


def _studio_layout(
    layout_id: LayoutId,
    room_texture_spec: StudioRoomTextureSpec,
    large_asset_placement_spec: StudioLargeAssetPlacementSpec,
    room_decor_placement_spec: StudioRoomDecorPlacementSpec,
    desk_pose: PoseSpec,
    desk_work_area_pose: PoseSpec,
    desk_robot_base_pose: PoseSpec,
    workbench_pose: PoseSpec,
    workbench_work_area_pose: PoseSpec,
    storage_pose: PoseSpec,
    relative_content_placement_overrides: dict[
        str, StudioRelativeContentPlacementOverride
    ] | None = None,
    articulated_storage_pose: PoseSpec | None = None,
    include_storage_cabinet: bool = True,
    interaction_site_robot_base_overrides: dict[str, PoseSpec] | None = None,
) -> LayoutSpec:
    support_surfaces = _studio_support_surfaces(
        desk_surface_pose=desk_work_area_pose,
        workbench_surface_pose=workbench_work_area_pose,
        storage_pose=storage_pose,
        placement_spec=large_asset_placement_spec,
        include_storage_cabinet=include_storage_cabinet,
    )
    relative_content_items, relative_placements = _studio_relative_content_items(
        support_surfaces=support_surfaces,
        placement_overrides=relative_content_placement_overrides,
    )
    site_robot_base_poses = {
        "desk_top": desk_robot_base_pose,
        "desk_drawer": _desk_drawer_robot_base_pose(
            desk_work_area_pose,
            STUDIO_DESK_DRAWER_SINGLE_BASE_LOCAL_XYZ,
        ),
        "workbench": _workbench_robot_base_pose(
            workbench_work_area_pose,
            STUDIO_WORKBENCH_SINGLE_BASE_LOCAL_XYZ,
        ),
        "printer": _furniture_front_robot_base_pose(
            furniture_pose=large_asset_placement_spec.printer_table_pose,
            model_id=STUDIO_PRINTER_TABLE_MODEL_ID,
            scale=STUDIO_PRINTER_TABLE_MODEL_SCALE,
        ),
        "sliding_cabinet_a": _sliding_cabinet_robot_base_pose(
            large_asset_placement_spec.sliding_cabinet_a_pose,
            STUDIO_SLIDING_CABINET_SINGLE_BASE_LOCAL_XYZ,
        ),
        "dresser_cabinet_b": _dresser_cabinet_robot_base_pose(
            large_asset_placement_spec.dresser_cabinet_b_pose,
            STUDIO_DRESSER_CABINET_SINGLE_BASE_LOCAL_XYZ,
        ),
    }
    known_site_ids = {
        "desk_top",
        "desk_drawer",
        "workbench",
        "printer",
        "sliding_cabinet_a",
        "dresser_cabinet_b",
    }
    unknown_override_ids = (
        set(interaction_site_robot_base_overrides or {}) - known_site_ids
    )
    if unknown_override_ids:
        raise ValueError(
            "Unknown Studio interaction-site robot-base overrides: "
            + ", ".join(sorted(unknown_override_ids))
        )
    if interaction_site_robot_base_overrides is not None:
        site_robot_base_poses.update(interaction_site_robot_base_overrides)
    missing_site_robot_bases = known_site_ids - set(site_robot_base_poses)
    if missing_site_robot_bases:
        raise ValueError(
            "Missing Studio interaction-site robot-base poses: "
            + ", ".join(sorted(missing_site_robot_bases))
        )

    printer_site_pose = (
        large_asset_placement_spec.printer_pose
        or large_asset_placement_spec.printer_table_pose
    )
    interaction_sites = {
        "desk_top": InteractionSiteSpec(
            site_id="desk_top",
            frame_pose=desk_work_area_pose,
            robot_base_center_pose=site_robot_base_poses["desk_top"],
            camera_rig_id="studio_desk",
            bimanual_robot_base_center_pose=_desk_top_robot_base_pose(
                desk_work_area_pose,
                STUDIO_DESK_TOP_BIMANUAL_CENTER_LOCAL_XYZ,
            ),
            target_entity_name="desk",
            target_entity_source="layout_entity",
        ),
        "desk_drawer": InteractionSiteSpec(
            site_id="desk_drawer",
            frame_pose=desk_work_area_pose,
            robot_base_center_pose=site_robot_base_poses["desk_drawer"],
            camera_rig_id="studio_desk",
            bimanual_robot_base_center_pose=_desk_drawer_robot_base_pose(
                desk_work_area_pose,
                STUDIO_DESK_DRAWER_BIMANUAL_CENTER_LOCAL_XYZ,
            ),
            target_entity_name="desk",
            target_entity_source="layout_entity",
        ),
        "workbench": InteractionSiteSpec(
            site_id="workbench",
            frame_pose=workbench_work_area_pose,
            robot_base_center_pose=site_robot_base_poses["workbench"],
            camera_rig_id="studio_workbench",
            bimanual_robot_base_center_pose=_workbench_robot_base_pose(
                workbench_work_area_pose,
                STUDIO_WORKBENCH_BIMANUAL_CENTER_LOCAL_XYZ,
            ),
            target_entity_name="workbench",
            target_entity_source="layout_entity",
        ),
        "printer": InteractionSiteSpec(
            site_id="printer",
            frame_pose=printer_site_pose,
            robot_base_center_pose=site_robot_base_poses["printer"],
            camera_rig_id="studio_printer",
            target_entity_name="printer",
            target_entity_source="layout_entity",
        ),
        "sliding_cabinet_a": InteractionSiteSpec(
            site_id="sliding_cabinet_a",
            frame_pose=large_asset_placement_spec.sliding_cabinet_a_pose,
            robot_base_center_pose=site_robot_base_poses["sliding_cabinet_a"],
            camera_rig_id="studio_cabinet",
            bimanual_robot_base_center_pose=_sliding_cabinet_robot_base_pose(
                large_asset_placement_spec.sliding_cabinet_a_pose,
                STUDIO_SLIDING_CABINET_BIMANUAL_CENTER_LOCAL_XYZ,
            ),
            target_entity_name="sliding_cabinet_a",
            target_entity_source="layout_entity",
        ),
        "dresser_cabinet_b": InteractionSiteSpec(
            site_id="dresser_cabinet_b",
            frame_pose=large_asset_placement_spec.dresser_cabinet_b_pose,
            robot_base_center_pose=site_robot_base_poses["dresser_cabinet_b"],
            camera_rig_id="studio_dresser_cabinet_b",
            target_entity_name="dresser_cabinet_b",
            target_entity_source="layout_entity",
        ),
    }

    build_items = [
        replace(
            static_box(
                "studio_floor",
                (
                    STUDIO_ROOM_HALF_SIZE_X,
                    STUDIO_ROOM_HALF_SIZE_Y,
                    STUDIO_FLOOR_HALF_HEIGHT,
                ),
                PoseSpec(
                    position_xyz=(
                        0.25,
                        0.0,
                        STUDIO_GROUND_Z - STUDIO_FLOOR_HALF_HEIGHT,
                    ),
                ),
                (0.30, 0.31, 0.30, 1.0),
            ),
            physical_material=STUDIO_PHYSICAL_MATERIAL,
        ),
        *_studio_room_shell_items(room_texture_spec),
        _fixed_partnet_asset(
            name="desk",
            model_id=STUDIO_DESK_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_DESK_MODEL_ID,
                desired_center_pose=desk_pose,
                scale=STUDIO_DESK_MODEL_SCALE,
            ),
            scale=STUDIO_DESK_MODEL_SCALE,
        ),
        _fixed_partnet_asset(
            name="workbench",
            model_id=STUDIO_WORKBENCH_MODEL_ID,
            pose=_partnet_root_pose_from_bbox_center(
                model_id=STUDIO_WORKBENCH_MODEL_ID,
                desired_center_pose=workbench_pose,
                scale=STUDIO_WORKBENCH_MODEL_SCALE,
            ),
            scale=STUDIO_WORKBENCH_MODEL_SCALE,
            # Preserve the historical fix for the official missing new-24 mesh.
            urdf_path_override=partnet_mobility_modified_model_urdf_path(
                STUDIO_WORKBENCH_MODEL_ID,
            ),
        ),
    ]
    build_items.extend(
        _studio_large_asset_items(
            desk_pose,
            large_asset_placement_spec,
        )
    )

    if include_storage_cabinet:
        if articulated_storage_pose is None:
            build_items.append(
                replace(
                    static_box(
                        "storage_cabinet",
                        (0.28, 0.35, 0.50),
                        PoseSpec(
                            position_xyz=(
                                storage_pose.position_xyz[0],
                                storage_pose.position_xyz[1],
                                storage_pose.position_xyz[2] + 0.50,
                            ),
                            quaternion_wxyz=storage_pose.quaternion_wxyz,
                        ),
                        (0.42, 0.45, 0.49, 1.0),
                    ),
                    physical_material=STUDIO_PHYSICAL_MATERIAL,
                )
            )
        else:
            build_items.append(
                ArticulatedAssetSpec(
                    name="storage_cabinet",
                    asset_path=partnet_mobility_model_urdf_path(
                        STUDIO_STORAGE_CABINET_MODEL_ID
                    ),
                    pose=articulated_storage_pose,
                    scale=STUDIO_STORAGE_CABINET_MODEL_SCALE,
                    fix_root_link=True,
                    disable_self_collisions=True,
                    initial_qpos=STUDIO_STORAGE_CABINET_INITIAL_QPOS,
                    default_joint_friction=STUDIO_DEFAULT_JOINT_FRICTION,
                    physical_material=STUDIO_PHYSICAL_MATERIAL,
                )
            )

    build_items.extend(relative_content_items)
    build_items.extend(_studio_room_decor_items(room_decor_placement_spec))

    return LayoutSpec(
        layout_id=layout_id,
        scene_class=SceneClass.STUDIO,
        description=(
            f"Studio layout {layout_id.value[-1]} with desk, workbench, "
            "and storage zones."
        ),
        build_items=build_items,
        interaction_sites=interaction_sites,
        support_surfaces=support_surfaces,
        relative_placements=relative_placements,
        lighting=_studio_lighting_spec(large_asset_placement_spec),
        add_default_ground=False,
        settle_sim_steps=STUDIO_SETTLE_SIM_STEPS,
    )


STUDIO_TUNED_RELATIVE_CONTENT_PLACEMENT_OVERRIDES: dict[
    str, StudioRelativeContentPlacementOverride
] = {
    "phone": StudioRelativeContentPlacementOverride(
        local_xy=(-0.010000, 0.290000),
    ),
    "usb": StudioRelativeContentPlacementOverride(
        local_xy=(-0.110000, -0.579999),
        bbox_center_z=0.006696,
        yaw=0.120000,
    ),
    "clutter_power_drill": StudioRelativeContentPlacementOverride(
        local_xy=(0.224439, -0.550392),
        bbox_center_z=0.029807,
        yaw=0.180000,
    ),
    "clutter_phillips_screwdriver": StudioRelativeContentPlacementOverride(
        local_xy=(-0.126276, 0.697268),
        bbox_center_z=0.017866,
        yaw=-3.003185,
    ),
    "workbench_pliers": StudioRelativeContentPlacementOverride(
        local_xy=(-0.204340, 0.497776),
        bbox_center_z=0.009408,
        yaw=2.790000,
    ),
    "workbench_box": StudioRelativeContentPlacementOverride(
        local_xy=(-0.201943, -0.556645),
        bbox_center_z=0.077229,
        yaw=0.0,
    ),
    "clutter_large_marker": StudioRelativeContentPlacementOverride(
        local_xy=(-0.009743, -0.536872),
        bbox_center_z=0.009809,
        yaw=0.200000,
    ),
    "glasses": StudioRelativeContentPlacementOverride(
        local_xy=(0.094463, 0.406812),
        bbox_center_z=0.026467,
        yaw=-2.323185,
    ),
    "clutter_paper_placeholder": StudioRelativeContentPlacementOverride(
        local_xy=(-0.207070, -0.342312),
        bbox_center_z=0.002000,
        yaw=0.080000,
    ),
    "stapler": StudioRelativeContentPlacementOverride(
        local_xy=(-0.231667, -0.215348),
        bbox_center_z=0.035011,
        yaw=-3.040000,
    ),
    "clutter_dice": StudioRelativeContentPlacementOverride(
        local_xy=(-0.260237, 0.426432),
        bbox_center_z=0.008914,
        yaw=-0.080000,
    ),
    "clutter_toy_airplane": StudioRelativeContentPlacementOverride(
        local_xy=(-0.000540, -0.110003),
        bbox_center_z=0.064389,
        yaw=0.100000,
    ),
    "clutter_rubiks_cube": StudioRelativeContentPlacementOverride(
        local_xy=(0.019676, -0.290002),
        bbox_center_z=0.039141,
        yaw=0.080000,
    ),
    "clutter_mug": StudioRelativeContentPlacementOverride(
        local_xy=(-0.050000, 0.320000),
        yaw=-0.100000,
    ),
    "clutter_pudding_box": StudioRelativeContentPlacementOverride(
        local_xy=(0.040000, 0.080000),
        yaw=-0.070000,
        surface_id=STUDIO_OPEN_SHELF_FIRST_LEVEL_SURFACE_ID,
    ),
    "clutter_lego_duplo": StudioRelativeContentPlacementOverride(
        local_xy=(-0.030000, -0.100000),
        yaw=0.120000,
    ),
    "clutter_apple": StudioRelativeContentPlacementOverride(
        local_xy=(0.050000, -0.350000),
        yaw=0.0,
    ),
    "clutter_card_placeholder": StudioRelativeContentPlacementOverride(
        local_xy=(-0.069996, 0.319244),
        bbox_center_z=0.013000,
        yaw=0.160000,
    ),
    "clutter_cup": StudioRelativeContentPlacementOverride(
        local_xy=(0.107834, 0.201176),
        bbox_center_z=0.041539,
        yaw=0.0,
    ),
    "clutter_mini_soccer_ball": StudioRelativeContentPlacementOverride(
        local_xy=(-0.040002, -0.100432),
        bbox_center_z=0.056670,
        yaw=0.0,
    ),
    "clutter_tennis_ball": StudioRelativeContentPlacementOverride(
        local_xy=(0.106652, -0.430306),
        bbox_center_z=0.043262,
        yaw=0.0,
    ),
    "clutter_chain": StudioRelativeContentPlacementOverride(
        local_xy=(0.029998, -0.260324),
        bbox_center_z=0.023744,
        yaw=0.040000,
    ),
    "clutter_medium_clamp": StudioRelativeContentPlacementOverride(
        local_xy=(-0.000648, -0.179996),
        bbox_center_z=0.023660,
        yaw=-0.080000,
    ),
    "clutter_foam_brick": StudioRelativeContentPlacementOverride(
        local_xy=(0.059349, -0.340644),
        bbox_center_z=0.036037,
        yaw=0.100000,
    ),
    "clutter_banana": StudioRelativeContentPlacementOverride(
        local_xy=(0.056741, 0.596162),
        bbox_center_z=0.040806,
        yaw=0.100000,
    ),
    "clutter_printer_card_placeholder": StudioRelativeContentPlacementOverride(
        local_xy=(0.040000, -0.280000),
        bbox_center_z=0.013000,
        yaw=-0.100000,
    ),
}

STUDIO_L2_RELATIVE_CONTENT_PLACEMENT_OVERRIDES: dict[
    str, StudioRelativeContentPlacementOverride
] = {
    **STUDIO_TUNED_RELATIVE_CONTENT_PLACEMENT_OVERRIDES,
    "clutter_toy_airplane": StudioRelativeContentPlacementOverride(
        local_xy=(-0.000540, 0.110003),
        bbox_center_z=0.064389,
        yaw=0.100000,
    ),
    "clutter_rubiks_cube": StudioRelativeContentPlacementOverride(
        local_xy=(0.019676, 0.290002),
        bbox_center_z=0.039141,
        yaw=0.080000,
    ),
    "clutter_card_placeholder": StudioRelativeContentPlacementOverride(
        local_xy=(-0.070104, -0.310756),
        bbox_center_z=0.013000,
        yaw=0.160000,
    ),
}


STUDIO_LAYOUTS: dict[LayoutId, LayoutSpec] = {
    LayoutId.STUDIO_L1: _studio_layout(
        LayoutId.STUDIO_L1,
        room_texture_spec=StudioRoomTextureSpec(
            floor_texture="wood/light_wood_planks.png",
            wall_texture="bricks/white_bricks.png",
            ceiling_texture="prev/white-plaster.png",
        ),
        large_asset_placement_spec=StudioLargeAssetPlacementSpec(
            printer_table_pose=pose(-0.500001, 1.760000, 0.62, 0.0),
            printer_pose=pose(-0.432349, 1.690009, 0.79, 1.573186),
            sliding_cabinet_a_pose=pose(
                0.566753,
                -1.977567,
                0.922070,
                -1.560000,
            ),
            open_shelf_pose=pose(2.184725, 1.914114, 0.891400, 1.560000),
            dresser_cabinet_b_pose=pose(
                1.136757,
                1.895144,
                0.991070,
                1.560000,
            ),
            desk_chair_initial_qpos=STUDIO_L1_DESK_CHAIR_INITIAL_QPOS,
            ceiling_light_mesh_relative_path=(
                STUDIO_CHANDELIER_ASSET_RELATIVE_PATH
            ),
            ceiling_light_quaternion_wxyz=(
                STUDIO_CHANDELIER_MODEL_TO_WORLD_QUATERNION_WXYZ
            ),
        ),
        room_decor_placement_spec=StudioRoomDecorPlacementSpec(
            rug_texture_relative_path=STUDIO_RUG_TEXTURE_RELATIVE_PATH,
            rug_pose=pose(0.35, 0.88, 0.0025, 0.05),
            potted_plant_robocasa_asset=STUDIO_POTTED_PLANT_ROBOCASA_ASSET,
            potted_plant_root_pose=pose(
                -2.18,
                -0.92,
                STUDIO_CENTERED_POTTED_PLANT_GROUNDED_ROOT_Z,
            ),
            potted_plant_scale_xyz=STUDIO_POTTED_PLANT_SCALE_XYZ,
            coat_rack_asset_relative_path=STUDIO_COAT_RACK_ASSET_RELATIVE_PATH,
            coat_rack_root_pose=PoseSpec(
                position_xyz=(-2.18, 1.78, STUDIO_GROUND_Z),
                quaternion_wxyz=STUDIO_COAT_RACK_MODEL_TO_WORLD_QUATERNION_WXYZ,
            ),
            coat_rack_scale_xyz=STUDIO_COAT_RACK_SCALE_XYZ,
            wall_clock_bbox_center_pose=PoseSpec(
                position_xyz=(2.955446, -1.308860, 2.11),
                quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
            wall_switch_bbox_center_pose=PoseSpec(
                position_xyz=(-0.617991, -2.222096, 1.5),
                quaternion_wxyz=(0.700285, 0.0, 0.0, -0.713864),
            ),
            trash_can_bbox_center_pose=pose(2.65, 0.72, 0.32),
            wall_fan_bbox_center_pose=PoseSpec(
                position_xyz=(-2.366619, -0.456944, 1.85),
                quaternion_wxyz=(0.999792, 0.0, 0.0, -0.020397),
            ),
        ),
        desk_pose=pose(2.399998, -0.440001, STUDIO_TABLETOP_Z, 0.0),
        desk_work_area_pose=pose(
            2.399998,
            -0.440001,
            STUDIO_DESK_WORK_SURFACE_Z,
            0.0,
        ),
        desk_robot_base_pose=_desk_top_robot_base_pose(
            pose(
                2.399998,
                -0.440001,
                STUDIO_DESK_WORK_SURFACE_Z,
                0.0,
            ),
            STUDIO_DESK_TOP_SINGLE_BASE_LOCAL_XYZ,
        ),
        workbench_pose=pose(-0.182189, -0.229589, STUDIO_TABLETOP_Z, 1.559635),
        workbench_work_area_pose=pose(
            -0.182189,
            -0.229589,
            STUDIO_WORKBENCH_WORK_SURFACE_Z,
            1.559635,
        ),
        storage_pose=pose(-2.171199, 0.687321, STUDIO_GROUND_Z, -3.133186),
        relative_content_placement_overrides=(
            STUDIO_TUNED_RELATIVE_CONTENT_PLACEMENT_OVERRIDES
        ),
        articulated_storage_pose=_partnet_root_pose_from_bbox_center(
            model_id=STUDIO_STORAGE_CABINET_MODEL_ID,
            desired_center_pose=pose(-2.171199, 0.687321, 0.775269, -3.133186),
            scale=STUDIO_STORAGE_CABINET_MODEL_SCALE,
        ),
        interaction_site_robot_base_overrides={
            "printer": pose(
                -0.331327,
                0.870001,
                0.760000,
                1.560000,
            ),
        },
    ),
    LayoutId.STUDIO_L2: _studio_layout(
        LayoutId.STUDIO_L2,
        room_texture_spec=StudioRoomTextureSpec(
            floor_texture="wood/varnished_wood_planks.png",
            wall_texture="prev/plywood.png",
            ceiling_texture="wood/wood_planks_wall.png",
        ),
        large_asset_placement_spec=StudioLargeAssetPlacementSpec(
            printer_table_pose=pose(2.479996, -1.679999, 0.384230, 1.570000),
            printer_pose=pose(2.412485, -1.638721, 1.012590, -0.020000),
            sliding_cabinet_a_pose=pose(
                -0.249998,
                1.689998,
                0.922070,
                1.569999,
            ),
            open_shelf_pose=pose(2.489122, 0.531296, 0.891400, -0.010000),
            dresser_cabinet_b_pose=pose(
                2.427990,
                -0.571407,
                0.991070,
                -0.023185,
            ),
            desk_chair_initial_qpos=STUDIO_L1_DESK_CHAIR_INITIAL_QPOS,
            ceiling_light_xy=(0.05, 0.10),
            ceiling_light_root_z=STUDIO_L2_CHANDELIER_ROOT_Z,
            ceiling_light_mesh_relative_path=(
                STUDIO_L2_CHANDELIER_ASSET_RELATIVE_PATH
            ),
            ceiling_light_emission_z=1.88,
            ceiling_light_quaternion_wxyz=(
                STUDIO_CHANDELIER_MODEL_TO_WORLD_QUATERNION_WXYZ
            ),
        ),
        room_decor_placement_spec=StudioRoomDecorPlacementSpec(
            rug_texture_relative_path=STUDIO_L2_RUG_TEXTURE_RELATIVE_PATH,
            rug_pose=pose(-0.320437, 0.009999, 0.002500, 1.570000),
            potted_plant_robocasa_asset=(
                STUDIO_L2_POTTED_PLANT_ROBOCASA_ASSET
            ),
            potted_plant_root_pose=pose(
                -2.100435,
                -1.800000,
                STUDIO_CENTERED_POTTED_PLANT_GROUNDED_ROOT_Z,
                1.570000,
            ),
            potted_plant_scale_xyz=STUDIO_POTTED_PLANT_SCALE_XYZ,
            coat_rack_asset_relative_path=STUDIO_COAT_RACK_ASSET_RELATIVE_PATH,
            coat_rack_root_pose=_grounded_coat_rack_pose(
                -2.100435,
                -1.040000,
                1.570000,
            ),
            coat_rack_scale_xyz=STUDIO_COAT_RACK_SCALE_XYZ,
            wall_clock_bbox_center_pose=PoseSpec(
                position_xyz=(-2.438697, 0.843541, 2.110000),
                quaternion_wxyz=(-0.019202, 0.0, 0.0, -0.999816),
            ),
            wall_switch_bbox_center_pose=PoseSpec(
                position_xyz=(-0.617991, -2.222096, 1.5),
                quaternion_wxyz=(0.700285, 0.0, 0.0, -0.713864),
            ),
            trash_can_bbox_center_pose=pose(
                2.650000,
                1.609999,
                0.218729,
            ),
            wall_fan_bbox_center_pose=pose(
                1.700461,
                -2.156898,
                1.850000,
                1.549203,
            ),
        ),
        desk_pose=pose(1.510000, 1.639998, 0.433100, 1.570000),
        desk_work_area_pose=pose(
            1.510000,
            1.639998,
            STUDIO_DESK_WORK_SURFACE_Z,
            1.570000,
        ),
        desk_robot_base_pose=_desk_top_robot_base_pose(
            pose(
                1.510000,
                1.639998,
                STUDIO_DESK_WORK_SURFACE_Z,
                1.570000,
            ),
            STUDIO_DESK_TOP_SINGLE_BASE_LOCAL_XYZ,
        ),
        workbench_pose=pose(-1.677553, 0.480012, 0.469170, 3.103186),
        workbench_work_area_pose=pose(
            -1.677553,
            0.480012,
            STUDIO_WORKBENCH_WORK_SURFACE_Z,
            3.103186,
        ),
        storage_pose=pose(0.309998, -1.829998, STUDIO_GROUND_Z, -1.570000),
        relative_content_placement_overrides=(
            STUDIO_L2_RELATIVE_CONTENT_PLACEMENT_OVERRIDES
        ),
        articulated_storage_pose=_partnet_root_pose_from_bbox_center(
            model_id=STUDIO_STORAGE_CABINET_MODEL_ID,
            desired_center_pose=pose(
                0.309998,
                -1.829998,
                0.775269,
                -1.570000,
            ),
            scale=STUDIO_STORAGE_CABINET_MODEL_SCALE,
        ),
        interaction_site_robot_base_overrides={
            "printer": pose(
                1.859358,
                -1.131325,
                0.969999,
                1.570000,
            ),
        },
    ),
    LayoutId.STUDIO_L3: _studio_layout(
        LayoutId.STUDIO_L3,
        room_texture_spec=StudioRoomTextureSpec(
            floor_texture="tiles/diamond_tiles_bak.png",
            wall_texture="marble/marble_5.png",
            ceiling_texture="bricks/stone_bricks.png",
        ),
        large_asset_placement_spec=StudioLargeAssetPlacementSpec(
            printer_table_pose=pose(0.720001, 1.590000, 0.384230, -1.570000),
            printer_pose=pose(0.737066, 1.472476, 1.012590, 1.563185),
            sliding_cabinet_a_pose=pose(
                2.045728,
                1.700507,
                0.922070,
                1.593185,
            ),
            open_shelf_pose=pose(
                -2.048779,
                0.409554,
                0.891400,
                -3.119999,
            ),
            dresser_cabinet_b_pose=pose(
                -2.058020,
                -0.667396,
                0.991070,
                -3.133186,
            ),
            desk_chair_initial_qpos=STUDIO_L1_DESK_CHAIR_INITIAL_QPOS,
            ceiling_light_xy=(0.40, -0.10),
            ceiling_light_root_z=STUDIO_L3_CEILING_LIGHT_ROOT_Z,
            ceiling_light_mesh_relative_path=(
                STUDIO_L3_CEILING_LIGHT_ASSET_RELATIVE_PATH
            ),
            ceiling_light_emission_z=1.78,
            ceiling_light_quaternion_wxyz=(
                STUDIO_CHANDELIER_MODEL_TO_WORLD_QUATERNION_WXYZ
            ),
        ),
        room_decor_placement_spec=StudioRoomDecorPlacementSpec(
            rug_texture_relative_path=STUDIO_L3_RUG_TEXTURE_RELATIVE_PATH,
            rug_pose=pose(-0.755264, -0.136085, 0.002500, 1.560000),
            potted_plant_robocasa_asset=(
                STUDIO_L3_POTTED_PLANT_ROBOCASA_ASSET
            ),
            potted_plant_root_pose=pose(
                -2.035263,
                1.613913,
                STUDIO_CENTERED_POTTED_PLANT_GROUNDED_ROOT_Z,
                1.560000,
            ),
            potted_plant_scale_xyz=STUDIO_POTTED_PLANT_SCALE_XYZ,
            coat_rack_asset_relative_path=STUDIO_COAT_RACK_ASSET_RELATIVE_PATH,
            coat_rack_root_pose=_grounded_coat_rack_pose(
                -0.115264,
                -1.626084,
                1.560000,
            ),
            coat_rack_scale_xyz=STUDIO_COAT_RACK_SCALE_XYZ,
            wall_clock_bbox_center_pose=pose(
                -0.740067,
                2.209062,
                2.110000,
                1.590000,
            ),
            wall_switch_bbox_center_pose=PoseSpec(
                position_xyz=(-0.617991, -2.222096, 1.5),
                quaternion_wxyz=(0.700285, 0.0, 0.0, -0.713864),
            ),
            trash_can_bbox_center_pose=pose(
                -2.099998,
                -1.700000,
                0.218729,
            ),
            wall_fan_bbox_center_pose=pose(
                -0.069537,
                -2.136897,
                1.950000,
                1.549203,
            ),
        ),
        desk_pose=pose(-0.719326, 1.533882, 0.433100, 1.560000),
        desk_work_area_pose=pose(
            -0.719326,
            1.533882,
            STUDIO_DESK_WORK_SURFACE_Z,
            1.560000,
        ),
        desk_robot_base_pose=_desk_top_robot_base_pose(
            pose(
                -0.719326,
                1.533882,
                STUDIO_DESK_WORK_SURFACE_Z,
                1.560000,
            ),
            STUDIO_DESK_TOP_SINGLE_BASE_LOCAL_XYZ,
        ),
        workbench_pose=pose(1.482148, -1.512948, 0.469170, -1.550000),
        workbench_work_area_pose=pose(
            1.482148,
            -1.512948,
            STUDIO_WORKBENCH_WORK_SURFACE_Z,
            -1.550000,
        ),
        storage_pose=pose(2.605875, 0.001691, STUDIO_GROUND_Z, 0.010001),
        relative_content_placement_overrides=(
            STUDIO_TUNED_RELATIVE_CONTENT_PLACEMENT_OVERRIDES
        ),
        articulated_storage_pose=_partnet_root_pose_from_bbox_center(
            model_id=STUDIO_STORAGE_CABINET_MODEL_ID,
            desired_center_pose=pose(
                2.605875,
                0.001691,
                0.775269,
                0.010001,
            ),
            scale=STUDIO_STORAGE_CABINET_MODEL_SCALE,
        ),
        include_storage_cabinet=False,
        interaction_site_robot_base_overrides={
            "printer": pose(
                0.389363,
                0.651326,
                0.869999,
                -1.570000,
            ),
        },
    ),
}
