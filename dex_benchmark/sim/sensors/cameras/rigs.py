from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


SceneCameraUid = Literal["context_camera", "interaction_camera"]


@dataclass(frozen=True)
class HumanViewerCameraPoseSpec:
    """Task/layout-specific free Human Viewer pose in world coordinates."""

    position_world_xyz: tuple[float, float, float]
    quaternion_world_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position_world_xyz) != 3:
            raise ValueError(
                "Human Viewer camera position must contain exactly 3 values."
            )
        if len(self.quaternion_world_wxyz) != 4:
            raise ValueError(
                "Human Viewer camera quaternion must contain exactly 4 values."
            )
        pose_values = self.position_world_xyz + self.quaternion_world_wxyz
        if not all(math.isfinite(value) for value in pose_values):
            raise ValueError(
                "Human Viewer camera position and quaternion must be finite."
            )
        quaternion_norm = math.sqrt(
            sum(value * value for value in self.quaternion_world_wxyz)
        )
        if not math.isclose(quaternion_norm, 1.0, abs_tol=1e-5):
            raise ValueError(
                "Human Viewer camera quaternion must have unit length, got "
                f"norm={quaternion_norm:.8f}."
            )


@dataclass(frozen=True)
class RenderCameraSpec:
    """Default human-viewer camera in the current interaction-site frame.

    The render camera is independent from the observation cameras. Its eye and
    target are expressed in the interaction-site local frame and transformed to
    world coordinates by the environment when the scene is configured.
    """

    eye_local_xyz: tuple[float, float, float]
    target_local_xyz: tuple[float, float, float]
    fov_rad: float = math.pi / 2.0
    width: int = 512
    height: int = 512
    near: float = 0.01
    far: float = 10.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Render camera resolution must be positive, got "
                f"{self.width}x{self.height}."
            )
        if not 0.0 < self.fov_rad < math.pi:
            raise ValueError(
                f"Render camera fov_rad must be in (0, pi), got {self.fov_rad}."
            )
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError(
                "Render camera clipping planes must satisfy 0 < near < far, got "
                f"near={self.near}, far={self.far}."
            )
        squared_view_distance = sum(
            (eye_value - target_value) ** 2
            for eye_value, target_value in zip(
                self.eye_local_xyz,
                self.target_local_xyz,
                strict=True,
            )
        )
        if squared_view_distance <= 1e-12:
            raise ValueError("Render camera eye and target must be distinct.")


@dataclass(frozen=True)
class CameraViewSpec:
    camera_uid: SceneCameraUid
    eye_local_xyz: tuple[float, float, float]
    target_local_xyz: tuple[float, float, float]
    fov_rad: float
    width: int = 224
    height: int = 224
    near: float = 0.01
    far: float = 10.0
    # Optional site-local up direction to preserve calibrated camera roll.
    up_local_xyz: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.up_local_xyz is not None:
            if (
                len(self.up_local_xyz) != 3
                or not all(math.isfinite(value) for value in self.up_local_xyz)
                or sum(value * value for value in self.up_local_xyz) <= 1e-12
            ):
                raise ValueError("Camera up direction must be a finite nonzero 3-vector.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"Camera {self.camera_uid!r} resolution must be positive, got "
                f"{self.width}x{self.height}."
            )
        if not 0.0 < self.fov_rad < math.pi:
            raise ValueError(
                f"Camera {self.camera_uid!r} fov_rad must be in (0, pi), got "
                f"{self.fov_rad}."
            )
        if self.near <= 0.0 or self.far <= self.near:
            raise ValueError(
                f"Camera {self.camera_uid!r} clipping planes must satisfy "
                f"0 < near < far, got near={self.near}, far={self.far}."
            )
        squared_view_distance = sum(
            (eye_value - target_value) ** 2
            for eye_value, target_value in zip(
                self.eye_local_xyz,
                self.target_local_xyz,
                strict=True,
            )
        )
        if squared_view_distance <= 1e-12:
            raise ValueError(
                f"Camera {self.camera_uid!r} eye and target must be distinct."
            )


@dataclass(frozen=True)
class CameraRigSpec:
    rig_id: str
    context_view: CameraViewSpec
    interaction_view: CameraViewSpec

    def __post_init__(self) -> None:
        if not self.rig_id.strip():
            raise ValueError("Camera rig_id must be non-empty.")
        if self.context_view.camera_uid != "context_camera":
            raise ValueError(
                f"Camera rig {self.rig_id!r} context_view must use "
                "camera_uid='context_camera'."
            )
        if self.interaction_view.camera_uid != "interaction_camera":
            raise ValueError(
                f"Camera rig {self.rig_id!r} interaction_view must use "
                "camera_uid='interaction_camera'."
            )


def _view(
    camera_uid: SceneCameraUid,
    eye_local_xyz: tuple[float, float, float],
    target_local_xyz: tuple[float, float, float],
    *,
    fov_rad: float = math.pi / 2.0,
    far: float = 10.0,
    up_local_xyz: tuple[float, float, float] | None = None,
) -> CameraViewSpec:
    return CameraViewSpec(
        camera_uid=camera_uid,
        eye_local_xyz=eye_local_xyz,
        target_local_xyz=target_local_xyz,
        fov_rad=fov_rad,
        far=far,
        up_local_xyz=up_local_xyz,
    )


def _rig(
    rig_id: str,
    context_eye: tuple[float, float, float],
    context_target: tuple[float, float, float],
    interaction_eye: tuple[float, float, float],
    interaction_target: tuple[float, float, float],
    *,
    context_fov_rad: float = math.pi / 2.0,
    interaction_fov_rad: float = math.pi / 2.0,
    context_up: tuple[float, float, float] | None = None,
    interaction_up: tuple[float, float, float] | None = None,
) -> CameraRigSpec:
    return CameraRigSpec(
        rig_id=rig_id,
        context_view=_view(
            "context_camera",
            context_eye,
            context_target,
            fov_rad=context_fov_rad,
            up_local_xyz=context_up,
        ),
        interaction_view=_view(
            "interaction_camera",
            interaction_eye,
            interaction_target,
            fov_rad=interaction_fov_rad,
            up_local_xyz=interaction_up,
        ),
    )


_CAMERA_RIGS = (
    _rig(
        "studio_desk",
        (-0.65, -0.80, 0.65),
        (-0.05, 0.40, -0.15),
        (0.50, 0.00, 0.40),
        (0.20, 0.00, 0.03),
    ),
    _rig(
        "studio_workbench",
        (-0.80, -0.85, 0.75),
        (0.00, 0.15, 0.05),
        (-0.65, 0.70, 0.60),
        (0.05, 0.05, 0.05),
    ),
    _rig(
        "studio_printer",
        (-0.90, -0.70, 0.75),
        (0.00, 0.00, 0.15),
        (-0.65, 0.60, 0.55),
        (0.00, 0.00, 0.12),
    ),
    _rig(
        "studio_cabinet",
        (-1.10, -0.70, 0.85),
        (0.00, 0.00, 0.25),
        (-0.90, 0.65, 0.60),
        (0.00, 0.00, 0.25),
    ),
    _rig(
        "studio_dresser_cabinet_b",
        (-1.45, -0.95, 0.95),
        (0.00, -0.10, 0.18),
        (-1.10, 0.75, 0.75),
        (0.00, -0.10, 0.30),
    ),
    _rig(
        "kitchen_counter",
        (0.50, 0.00, 0.40),
        (0.20, 0.00, 0.03),
        (-0.70, -0.65, 0.65),
        (0.00, 0.05, 0.05),
    ),
    _rig(
        "kitchen_dining_table",
        (-0.90, -0.85, 0.80),
        (0.00, 0.00, 0.05),
        (-0.75, 0.75, 0.60),
        (0.00, 0.00, 0.05),
    ),
    _rig(
        "kitchen_sink",
        (-0.90, -0.70, 0.80),
        (0.00, 0.00, 0.05),
        (-0.35, 0.60, 1.00),
        (0.00, 0.00, 0.05),
    ),
    _rig(
        "kitchen_cabinet",
        (-1.10, -0.70, 0.80),
        (0.00, 0.00, 0.25),
        (-0.90, 0.65, 0.60),
        (0.00, 0.00, 0.25),
    ),
    _rig(
        "kitchen_refrigerator",
        (-1.25, -0.70, 0.95),
        (0.00, 0.00, 0.30),
        (-0.95, 0.70, 0.75),
        (0.00, 0.00, 0.35),
    ),
    _rig(
        "kitchen_wall_switch",
        (-0.70, -0.45, 0.45),
        (0.00, 0.00, 0.05),
        (-0.55, 0.40, 0.35),
        (0.00, 0.00, 0.05),
        interaction_fov_rad=math.pi / 3.0,
    ),
    _rig(
        "kitchen_coffee_machine",
        (-0.80, -0.55, 0.65),
        (0.00, 0.00, 0.15),
        (-0.60, 0.50, 0.50),
        (0.00, 0.00, 0.15),
    ),
    _rig(
        "kitchen_toaster",
        (-0.80, -0.60, 0.70),
        (0.00, 0.00, 0.10),
        (-0.45, 0.50, 0.90),
        (0.00, 0.00, 0.10),
    ),
    _rig(
        "kitchen_dishwasher",
        (-1.30, -0.80, 0.90),
        (0.00, 0.00, 0.20),
        (-1.00, 0.75, 0.60),
        (0.00, 0.00, 0.20),
    ),
    _rig(
        "kitchen_microwave",
        (-1.00, -0.65, 0.70),
        (0.00, 0.00, 0.10),
        (-0.75, 0.60, 0.55),
        (0.00, 0.00, 0.10),
    ),
    _rig(
        "kitchen_oven",
        (-1.15, -0.70, 0.85),
        (0.00, 0.00, 0.20),
        (-0.90, 0.65, 0.60),
        (0.00, 0.00, 0.20),
    ),
    _rig(
        "kitchen_dispenser",
        (-0.75, -0.50, 0.60),
        (0.00, 0.00, 0.15),
        (-0.60, 0.45, 0.45),
        (0.00, 0.00, 0.15),
    ),
    _rig(
        "kitchen_stove",
        (-0.80, -0.60, 0.70),
        (0.00, 0.00, 0.10),
        (-0.45, 0.50, 0.90),
        (0.00, 0.00, 0.10),
    ),
)


# Keyed by (layout_id, site_id). Add only manually tuned exceptions here.
LAYOUT_RIG_OVERRIDES: dict[tuple[str, str], CameraRigSpec] = {
    # Kitchen L1. These values initially match the shared kitchen rigs above,
    # but each layout/site pair owns a separate context and interaction view so
    # the cameras can be tuned independently without affecting other layouts.
    ("kitchen_l1", "counter_top"): _rig(
        "kitchen_l1_counter_top",
        context_eye=(0.50, 0.00, 0.40),
        context_target=(0.20, 0.00, 0.03),
        interaction_eye=(-0.70, -0.65, 0.65),
        interaction_target=(0.00, 0.05, 0.05),
    ),
    ("kitchen_l1", "dining_table"): _rig(
        "kitchen_l1_dining_table",
        # Tuned from the M09 Human Viewer in world coordinates, then converted
        # into the dining-table site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.632435000, -0.511819000, 0.618428000),
        context_target=(-0.040180576, 0.800134663, 0.466729999),
        interaction_eye=(-0.194512000, -0.139752000, 0.507646000),
        interaction_target=(0.633258388, 0.567078888, 0.015014007),
    ),
    ("kitchen_l1", "sink"): _rig(
        "kitchen_l1_sink",
        # Tuned from the M06 Human Viewer in world coordinates, then converted
        # into the sink site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.099725000, -0.464678000, 0.675960000),
        context_target=(0.062209980, 0.249381685, 0.625701986),
        interaction_eye=(-0.388829000, -0.262979000, 0.650182000),
        interaction_target=(0.759429425, -0.088208159, 0.460556732),
    ),
    ("kitchen_l1", "refrigerator"): _rig(
        "kitchen_l1_refrigerator",
        context_eye=(-1.25, -0.70, 0.95),
        context_target=(0.00, 0.00, 0.30),
        interaction_eye=(-0.95, 0.70, 0.75),
        interaction_target=(0.00, 0.00, 0.35),
    ),
    ("kitchen_l1", "cabinet"): _rig(
        "kitchen_l1_cabinet",
        # Tuned from the M11 Human Viewer in world coordinates, then converted
        # into the cabinet site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.319880000, -0.847837000, 0.238658000),
        context_target=(-0.525616893, 0.290784787, 0.512859629),
        interaction_eye=(-0.756523000, -0.184958000, 0.328687000),
        interaction_target=(0.353324216, -0.032236442, 0.012593224),
    ),
    ("kitchen_l1", "wall_switch"): _rig(
        "kitchen_l1_wall_switch",
        # Tuned from the M02 Human Viewer in world coordinates, then converted
        # into the wall-switch site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.031796169, -0.606588432, 0.421783000),
        context_target=(-0.470082144, 0.088798889, 0.190602333),
        interaction_eye=(-0.495648816, -0.192722205, 0.302043000),
        interaction_target=(0.028709355, 0.268369571, 0.047167352),
        interaction_fov_rad=math.pi / 3.0,
    ),
    ("kitchen_l1", "coffee_machine"): _rig(
        "kitchen_l1_coffee_machine",
        # Tuned from the M08 Human Viewer in world coordinates, then converted
        # into the coffee-machine site frame. Target distances retain the
        # previous calibration while matching the recorded camera +X axes.
        context_eye=(-1.486449557, -0.676348491, 0.408443000),
        context_target=(-0.678074259, 0.043140446, 0.554764570),
        interaction_eye=(-1.015511176, -0.277060795, 0.485865000),
        interaction_target=(-0.168907346, -0.177304341, 0.562275063),
    ),
    ("kitchen_l1", "toaster"): _rig(
        "kitchen_l1_toaster",
        # Tuned from the M04 Human Viewer in world coordinates, then converted
        # into the toaster site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.111729172, -0.529969495, 0.411113000),
        context_target=(-0.151816774, 0.117432500, 0.550720674),
        interaction_eye=(-0.449104998, -0.293102239, 0.424982000),
        interaction_target=(0.344679331, 0.373296954, 0.289483709),
    ),
    ("kitchen_l1", "dishwasher"): _rig(
        "kitchen_l1_dishwasher",
        # Tuned from the M10 Human Viewer in world coordinates, then converted
        # into the dishwasher site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.895075000, -0.487646000, 0.880597000),
        context_target=(-0.456730687, 1.132027988, 0.813437206),
        interaction_eye=(-0.667255000, 0.107536000, 1.017003000),
        interaction_target=(0.502849433, 0.115133282, 0.422614321),
    ),
    ("kitchen_l1", "microwave"): _rig(
        "kitchen_l1_microwave",
        context_eye=(-1.00, -0.65, 0.70),
        context_target=(0.00, 0.00, 0.10),
        interaction_eye=(-0.75, 0.60, 0.55),
        interaction_target=(0.00, 0.00, 0.10),
    ),
    ("kitchen_l1", "stove"): _rig(
        "kitchen_l1_stove",
        context_eye=(-0.80, -0.60, 0.70),
        context_target=(0.00, 0.00, 0.10),
        interaction_eye=(-0.45, 0.50, 0.90),
        interaction_target=(0.00, 0.00, 0.10),
    ),
    # Kitchen L2.
    ("kitchen_l2", "counter_top"): _rig(
        "kitchen_l2_counter_top",
        context_eye=(0.50, 0.00, 0.40),
        context_target=(0.20, 0.00, 0.03),
        interaction_eye=(-0.70, -0.65, 0.65),
        interaction_target=(0.00, 0.05, 0.05),
    ),
    ("kitchen_l2", "dining_table"): _rig(
        "kitchen_l2_dining_table",
        # Tuned from the M09 Human Viewer in world coordinates, then converted
        # into the dining-table site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.647451000, -0.657289000, 0.864777000),
        context_target=(0.010018801, 0.608797084, 0.620319418),
        interaction_eye=(-0.185694000, -0.144819000, 0.547271000),
        interaction_target=(0.550855758, 0.639557480, 0.027898307),
    ),
    ("kitchen_l2", "sink"): _rig(
        "kitchen_l2_sink",
        # Tuned from the M06 Human Viewer in world coordinates, then converted
        # into the sink site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.934076980, -0.382134050, 0.664374000),
        context_target=(0.197254802, 0.352698518, 0.457953322),
        interaction_eye=(-0.439727983, -0.319038024, 0.573622000),
        interaction_target=(0.684345977, 0.023812684, 0.636160986),
    ),
    ("kitchen_l2", "refrigerator"): _rig(
        "kitchen_l2_refrigerator",
        context_eye=(-1.25, -0.70, 0.95),
        context_target=(0.00, 0.00, 0.30),
        interaction_eye=(-0.95, 0.70, 0.75),
        interaction_target=(0.00, 0.00, 0.35),
    ),
    ("kitchen_l2", "cabinet"): _rig(
        "kitchen_l2_cabinet",
        # Tuned from the M11 Human Viewer in world coordinates, then converted
        # into the cabinet site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.357155000, -0.893273000, 0.235419000),
        context_target=(-0.461515077, 0.140158966, 0.599214957),
        interaction_eye=(-0.979024000, -0.307641000, 0.418751000),
        interaction_target=(0.143738728, -0.000885233, 0.401293623),
    ),
    ("kitchen_l2", "wall_switch"): _rig(
        "kitchen_l2_wall_switch",
        # Tuned from the M02 Human Viewer in world coordinates, then converted
        # into the wall-switch site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.002517004, -0.533756473, 0.199058000),
        context_target=(-0.491065557, 0.222810421, 0.062837716),
        interaction_eye=(-0.293151565, -0.055312684, 0.191977000),
        interaction_target=(0.137197524, 0.375959666, -0.233837596),
        interaction_fov_rad=math.pi / 2.0,
    ),
    ("kitchen_l2", "coffee_machine"): _rig(
        "kitchen_l2_coffee_machine",
        # Tuned from the M08 Human Viewer in world coordinates, then converted
        # into the coffee-machine site frame. Target distances retain the
        # previous calibration while matching the recorded camera +X axes.
        context_eye=(-1.372081000, -0.651323000, 0.522541000),
        context_target=(-0.535129002, 0.048389966, 0.473417504),
        interaction_eye=(-0.677016000, -0.208980000, 0.433700000),
        interaction_target=(0.027359331, 0.181627917, 0.144250785),
    ),
    ("kitchen_l2", "toaster"): _rig(
        "kitchen_l2_toaster",
        # Tuned from the M04 Human Viewer in world coordinates, then converted
        # into the toaster site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.202147886, -0.687514252, 0.504414000),
        context_target=(-0.501075451, 0.220870395, 0.712705413),
        interaction_eye=(-0.422440016, -0.161172196, 0.475717000),
        interaction_target=(0.494549341, 0.252187561, 0.191458200),
    ),
    ("kitchen_l2", "dishwasher"): _rig(
        "kitchen_l2_dishwasher",
        # Tuned from the M10 Human Viewer in world coordinates, then converted
        # into the dishwasher site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.889169000, -0.590200000, 0.725301000),
        context_target=(-0.382779454, 0.968400741, 1.091816138),
        interaction_eye=(-0.654335000, 0.071432000, 0.940246000),
        interaction_target=(0.484168896, 0.039171962, 0.288120119),
    ),
    ("kitchen_l2", "microwave"): _rig(
        "kitchen_l2_microwave",
        context_eye=(-1.00, -0.65, 0.70),
        context_target=(0.00, 0.00, 0.10),
        interaction_eye=(-0.75, 0.60, 0.55),
        interaction_target=(0.00, 0.00, 0.10),
    ),
    ("kitchen_l2", "stove"): _rig(
        "kitchen_l2_stove",
        context_eye=(-0.80, -0.60, 0.70),
        context_target=(0.00, 0.00, 0.10),
        interaction_eye=(-0.45, 0.50, 0.90),
        interaction_target=(0.00, 0.00, 0.10),
    ),
    # Kitchen L3.
    ("kitchen_l3", "counter_top"): _rig(
        "kitchen_l3_counter_top",
        context_eye=(0.50, 0.00, 0.40),
        context_target=(0.20, 0.00, 0.03),
        interaction_eye=(-0.70, -0.65, 0.65),
        interaction_target=(0.00, 0.05, 0.05),
    ),
    ("kitchen_l3", "dining_table"): _rig(
        "kitchen_l3_dining_table",
        # Tuned from the M09 Human Viewer in world coordinates, then converted
        # into the dining-table site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.535892000, -0.489654000, 0.455747000),
        context_target=(-0.180871529, 0.907499061, 0.585838518),
        interaction_eye=(-0.194853000, -0.150343000, 0.467396000),
        interaction_target=(0.697023080, 0.537625027, 0.068952727),
    ),
    ("kitchen_l3", "sink"): _rig(
        "kitchen_l3_sink",
        # Tuned from the M06 Human Viewer in world coordinates, then converted
        # into the sink site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.767900952, -0.886637041, 1.027112000),
        context_target=(-0.023093244, 0.250796025, 0.908758727),
        interaction_eye=(-0.191451978, -0.410947010, 0.587485000),
        interaction_target=(0.665266083, 0.388956555, 0.481709531),
    ),
    ("kitchen_l3", "refrigerator"): _rig(
        "kitchen_l3_refrigerator",
        context_eye=(-1.25, -0.70, 0.95),
        context_target=(0.00, 0.00, 0.30),
        interaction_eye=(-0.95, 0.70, 0.75),
        interaction_target=(0.00, 0.00, 0.35),
    ),
    ("kitchen_l3", "cabinet"): _rig(
        "kitchen_l3_cabinet",
        # Tuned from the M11 Human Viewer in world coordinates, then converted
        # into the cabinet site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.316316953, -0.872491071, 0.297808000),
        context_target=(-0.560934044, 0.295472115, 0.558112889),
        interaction_eye=(-0.933677982, -0.328056050, 0.279428000),
        interaction_target=(0.208078024, -0.112369522, 0.349231657),
    ),
    ("kitchen_l3", "wall_switch"): _rig(
        "kitchen_l3_wall_switch",
        # Tuned from the M02 Human Viewer in world coordinates, then converted
        # into the wall-switch site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.894393594, -0.433965715, 0.218960000),
        context_target=(-0.512115634, 0.389149465, 0.049121952),
        interaction_eye=(-0.284794430, -0.117184954, 0.210865000),
        interaction_target=(0.103256442, 0.307980110, -0.259402147),
        interaction_fov_rad=math.pi / 2.0,
    ),
    ("kitchen_l3", "coffee_machine"): _rig(
        "kitchen_l3_coffee_machine",
        # Tuned from the M08 Human Viewer in world coordinates, then converted
        # into the coffee-machine site frame. Target distances retain the
        # previous calibration while matching the recorded camera +X axes.
        context_eye=(-1.262624000, -0.962816000, 0.272642000),
        context_target=(-0.582943544, -0.177647415, 0.610347154),
        interaction_eye=(-0.666563000, -0.212264000, 0.352405000),
        interaction_target=(0.050617176, 0.163195029, 0.074586512),
    ),
    ("kitchen_l3", "toaster"): _rig(
        "kitchen_l3_toaster",
        # Tuned from the M04 Human Viewer in world coordinates, then converted
        # into the toaster site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-0.925614382, -0.550201089, 0.687097000),
        context_target=(-0.237943709, 0.378645760, 0.423461868),
        interaction_eye=(-0.444165446, -0.167073085, 0.399960000),
        interaction_target=(0.403573187, 0.374528794, 0.061261677),
    ),
    ("kitchen_l3", "dishwasher"): _rig(
        "kitchen_l3_dishwasher",
        # Tuned from the M10 Human Viewer in world coordinates, then converted
        # into the dishwasher site frame. Target distances retain the previous
        # calibration while matching the recorded camera +X forward axes.
        context_eye=(-1.271132000, -0.508873000, 0.680455000),
        context_target=(-0.703489001, 1.068352969, 0.781150853),
        interaction_eye=(-0.815571000, 0.071237000, 0.769569000),
        interaction_target=(0.166198169, 0.037184511, -0.100758371),
    ),
    ("kitchen_l3", "microwave"): _rig(
        "kitchen_l3_microwave",
        context_eye=(-1.00, -0.65, 0.70),
        context_target=(0.00, 0.00, 0.10),
        interaction_eye=(-0.75, 0.60, 0.55),
        interaction_target=(0.00, 0.00, 0.10),
    ),
    ("kitchen_l3", "stove"): _rig(
        "kitchen_l3_stove",
        context_eye=(-0.80, -0.60, 0.70),
        context_target=(0.00, 0.00, 0.10),
        interaction_eye=(-0.45, 0.50, 0.90),
        interaction_target=(0.00, 0.00, 0.10),
    ),
    ("studio_l1", "desk_top"): _rig(
        "studio_l1_desk_top",
        # Tuned from the human viewer in world coordinates. Both eye and target
        # are stored in the desk_top interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(0.409253000, -0.504340000, 1.109614000),
        context_target=(-0.349803777, 0.335767119, 0.033485365),
        interaction_eye=(0.057585000, -0.132470000, 1.134185000),
        interaction_target=(0.245610922, -0.130485340, 0.696529527),
    ),
    ("studio_l1", "desk_drawer"): _rig(
        "studio_l1_desk_drawer",
        # Tuned from the human viewer in world coordinates. Both eye and target
        # are stored in the desk_drawer interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(-1.502638000, 0.148701000, 0.275720000),
        context_target=(-0.446227187, 1.157100372, -0.278470312),
        interaction_eye=(-0.981143000, 0.257704000, 0.158868000),
        interaction_target=(-0.535233126, 0.382145387, 0.046713082),
    ),
    ("studio_l1", "workbench"): _rig(
        "studio_l1_workbench",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated workbench interaction-site frame. Target distances retain
        # the previous calibration while matching the new view axes.
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("studio_l1", "printer"): _rig(
        "studio_l1_printer",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated printer interaction-site frame.
        context_eye=(-0.900000335, -0.700000290, 0.750000000),
        context_target=(-0.162528657, 0.184938096, 0.172921530),
        interaction_eye=(-0.801335377, -0.411811246, 0.317735000),
        interaction_target=(0.083723353, 0.000586807, 0.436051868),
    ),
    ("studio_l1", "sliding_cabinet_a"): _rig(
        "studio_l1_sliding_cabinet_a",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated sliding-cabinet interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(-1.246426297, -0.974116194, 0.212244000),
        context_target=(-0.182019679, -0.074438423, -0.130711325),
        interaction_eye=(-0.687768035, -0.633279675, -0.045984000),
        interaction_target=(0.371938034, -0.371913798, -0.450595984),
    ),
    ("studio_l1", "dresser_cabinet_b"): _rig(
        "studio_l1_dresser_cabinet_b",
        context_eye=(-1.45, -0.95, 0.95),
        context_target=(0.00, -0.10, 0.18),
        interaction_eye=(-1.10, 0.75, 0.75),
        interaction_target=(0.00, -0.10, 0.30),
    ),
    ("studio_l2", "desk_top"): _rig(
        "studio_l2_desk_top",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated desk_top interaction-site frame. Target distances retain
        # the previous calibration while matching the new view axes.
        context_eye=(0.238143021, -0.770966619, 1.108163000),
        context_target=(-0.381676094, 0.333801937, 0.194209630),
        interaction_eye=(-0.210442920, -0.027815591, 1.052608000),
        interaction_target=(0.120893248, -0.089111329, 0.715919449),
    ),
    ("studio_l2", "desk_drawer"): _rig(
        "studio_l2_desk_drawer",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated desk_drawer interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(-1.507000209, 0.226920012, 0.363291000),
        context_target=(-0.366104539, 1.019390824, -0.351094860),
        interaction_eye=(-0.934799677, 0.286770690, 0.185317000),
        interaction_target=(-0.529114504, 0.491364475, 0.042275290),
    ),
    ("studio_l2", "workbench"): _rig(
        "studio_l2_workbench",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated workbench interaction-site frame. Target distances retain
        # the previous calibration while matching the new view axes.
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("studio_l2", "printer"): _rig(
        "studio_l2_printer",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated printer interaction-site frame.
        context_eye=(-1.131648327, -0.609690392, 0.426323000),
        context_target=(-0.374361894, 0.284052763, -0.110091422),
        interaction_eye=(-0.676742685, -0.345055960, 0.155703000),
        interaction_target=(0.157316929, 0.169596691, 0.072774054),
    ),
    ("studio_l2", "sliding_cabinet_a"): _rig(
        "studio_l2_sliding_cabinet_a",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated sliding-cabinet interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(-1.702800324, -1.016209031, 0.240633000),
        context_target=(-0.503983977, -0.227529335, 0.211931416),
        interaction_eye=(-0.765686717, -0.621503712, 0.042574000),
        interaction_target=(0.267505414, -0.372061051, -0.432075790),
    ),
    ("studio_l2", "dresser_cabinet_b"): _rig(
        "studio_l2_dresser_cabinet_b",
        context_eye=(-1.45, -0.95, 0.95),
        context_target=(0.00, -0.10, 0.18),
        interaction_eye=(-1.10, 0.75, 0.75),
        interaction_target=(0.00, -0.10, 0.30),
    ),
    ("studio_l3", "desk_top"): _rig(
        "studio_l3_desk_top",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated desk_top interaction-site frame. Target distances retain
        # the previous calibration while matching the new view axes.
        context_eye=(0.185414373, -0.828313507, 0.943289000),
        context_target=(-0.373945484, 0.325662963, 0.051406265),
        interaction_eye=(-0.380973948, -0.058377441, 0.934025000),
        interaction_target=(-0.005618265, -0.093331486, 0.642844091),
    ),
    ("studio_l3", "desk_drawer"): _rig(
        "studio_l3_desk_drawer",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated desk_drawer interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(-1.425854362, 0.165733966, 0.367539000),
        context_target=(-0.376083294, 1.109264796, -0.301587463),
        interaction_eye=(-0.895171098, 0.166165310, 0.202473000),
        interaction_target=(-0.538833357, 0.432649462, 0.032444750),
    ),
    ("studio_l3", "workbench"): _rig(
        "studio_l3_workbench",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated workbench interaction-site frame. Target distances retain
        # the previous calibration while matching the new view axes.
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("studio_l3", "printer"): _rig(
        "studio_l3_printer",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated printer interaction-site frame.
        context_eye=(-1.031612023, -0.868301012, 0.223621000),
        context_target=(-0.313084048, 0.160228292, -0.069372209),
        interaction_eye=(-0.580950748, -0.614232555, 0.312529000),
        interaction_target=(-0.056312806, 0.186939248, 0.088299716),
    ),
    ("studio_l3", "sliding_cabinet_a"): _rig(
        "studio_l3_sliding_cabinet_a",
        # Tuned from the human viewer in world coordinates, then converted to
        # the rotated sliding-cabinet interaction-site frame. Target distances
        # retain the previous calibration while matching the new view axes.
        context_eye=(-1.296162288, -0.767348363, -0.141793000),
        context_target=(-0.137861212, 0.054436214, -0.349178248),
        interaction_eye=(-0.719589979, -0.619679969, -0.060043000),
        interaction_target=(0.328536425, -0.275765546, -0.431733540),
    ),
    ("studio_l3", "dresser_cabinet_b"): _rig(
        "studio_l3_dresser_cabinet_b",
        context_eye=(-1.45, -0.95, 0.95),
        context_target=(0.00, -0.10, 0.18),
        interaction_eye=(-1.10, 0.75, 0.75),
        interaction_target=(0.00, -0.10, 0.30),
    ),
}


# Tasks that share an interaction site keep a separate, explicit rig for every
# task/layout pair so calibrating either camera for one task cannot affect the
# other task. Uncalibrated task/layout pairs retain the corresponding
# layout-specific site calibration.
TASK_LAYOUT_RIG_OVERRIDES: dict[tuple[str, str, str], CameraRigSpec] = {
    ("b03_v1", "kitchen_l1", "counter_top"): _rig(
        "b03_kitchen_l1_counter_top",
        # Calibrated from the requested B03 L1 context-camera world pose and
        # converted to the counter-top interaction-site frame.
        context_eye=(-0.099089000, -0.779277000, 0.663834000),
        context_target=(-0.389431317, 0.385943239, 0.537885064),
        # Calibrated from the requested B03 L1 interaction-camera world pose
        # and converted to the counter-top interaction-site frame.
        interaction_eye=(-0.401320000, 0.052774000, 0.607097000),
        interaction_target=(0.393488554, 0.052774000, 0.136587316),
        context_up=(-0.025220283, 0.101216345, 0.994544714),
        interaction_up=(0.509411325, 0.000000000, 0.860523156),
    ),
    ("b03_v1", "kitchen_l2", "counter_top"): _rig(
        "b03_kitchen_l2_counter_top",
        # Calibrated from the requested B03 L2 camera world poses and
        # converted to the counter-top interaction-site frame.
        context_eye=(-0.419867000, -0.614952000, 0.807382000),
        context_target=(-0.516530737, 0.356244334, 0.096513655),
        interaction_eye=(-0.422818000, 0.055927000, 0.555480000),
        interaction_target=(0.383461888, 0.063989332, 0.104982785),
        context_up=(-0.058310900, 0.585847627, 0.808320727),
        interaction_up=(0.487719857, 0.004875780, 0.872986579),
    ),
    ("b03_v1", "kitchen_l3", "counter_top"): _rig(
        "b03_kitchen_l3_counter_top",
        # Calibrated from the requested B03 L3 camera world poses and
        # converted to the counter-top interaction-site frame.
        context_eye=(-0.409827000, -0.656310000, 0.721313000),
        context_target=(-0.453424739, 0.455191073, 0.251669943),
        interaction_eye=(-0.422450000, 0.038097000, 0.591397000),
        interaction_target=(0.374541270, 0.022155430, 0.124866420),
        context_up=(-0.015244298, 0.388660345, 0.921254985),
        interaction_up=(0.505002200, -0.010102021, 0.863058936),
    ),
    ("b07_v1", "kitchen_l1", "counter_top"): _rig(
        "b07_kitchen_l1_counter_top",
        # Reuse the calibrated B03 L1 context/interaction viewpoints.
        context_eye=(-0.099089000, -0.779277000, 0.663834000),
        context_target=(-0.389431317, 0.385943239, 0.537885064),
        interaction_eye=(-0.401320000, 0.052774000, 0.607097000),
        interaction_target=(0.393488554, 0.052774000, 0.136587316),
        context_up=(-0.025220283, 0.101216345, 0.994544714),
        interaction_up=(0.509411325, 0.000000000, 0.860523156),
    ),
    ("b07_v1", "kitchen_l2", "counter_top"): _rig(
        "b07_kitchen_l2_counter_top",
        # Reuse the calibrated B03 L2 context/interaction viewpoints.
        context_eye=(-0.419867000, -0.614952000, 0.807382000),
        context_target=(-0.516530737, 0.356244334, 0.096513655),
        interaction_eye=(-0.422818000, 0.055927000, 0.555480000),
        interaction_target=(0.383461888, 0.063989332, 0.104982785),
        context_up=(-0.058310900, 0.585847627, 0.808320727),
        interaction_up=(0.487719857, 0.004875780, 0.872986579),
    ),
    ("b07_v1", "kitchen_l3", "counter_top"): _rig(
        "b07_kitchen_l3_counter_top",
        # Reuse the calibrated B03 L3 context/interaction viewpoints.
        context_eye=(-0.409827000, -0.656310000, 0.721313000),
        context_target=(-0.453424739, 0.455191073, 0.251669943),
        interaction_eye=(-0.422450000, 0.038097000, 0.591397000),
        interaction_target=(0.374541270, 0.022155430, 0.124866420),
        context_up=(-0.015244298, 0.388660345, 0.921254985),
        interaction_up=(0.505002200, -0.010102021, 0.863058936),
    ),
    ("m12_v1", "kitchen_l1", "microwave"): _rig(
        "m12_kitchen_l1_microwave",
        # Tuned from the human viewer in world coordinates, then converted to
        # the microwave interaction-site frame.
        context_eye=(0.859357635, -0.915967157, 0.188611),
        context_target=(-0.028690728, -0.488425397, 0.019560277),
        interaction_eye=(0.256383937, -0.833957942, 0.209634),
        interaction_target=(0.156007285, 0.145662747, 0.035657464),
    ),
    ("m12_v1", "kitchen_l2", "microwave"): _rig(
        "m12_kitchen_l2_microwave",
        # Tuned from the human viewer in world coordinates, then converted to
        # the microwave interaction-site frame.
        context_eye=(0.794022309, -1.004732917, 0.089927),
        context_target=(-0.03645776, -0.473955303, -0.07912323),
        interaction_eye=(0.142727229, -0.754363524, 0.129777),
        interaction_target=(0.150463188, 0.225699122, -0.068761066),
    ),
    ("m12_v1", "kitchen_l3", "microwave"): _rig(
        "m12_kitchen_l3_microwave",
        # World viewer poses converted to the microwave interaction-site frame.
        # Explicit up directions preserve the small nonzero camera roll.
        context_eye=(-1.221588851, -0.760943035, 0.276596),
        context_target=(-0.614623309, 0.033416012, 0.251598024),
        interaction_eye=(-0.796207385, -0.255309171, 0.480707),
        interaction_target=(0.113783981, -0.211290537, 0.068384391),
        context_up=(0.01550813, 0.019611058, 0.999687415),
        interaction_up=(0.411868905, 0.019509144, 0.911037828),
    ),
    ("b04_v1", "studio_l1", "workbench"): _rig(
        "b04_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b04_v1", "studio_l2", "workbench"): _rig(
        "b04_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b04_v1", "studio_l3", "workbench"): _rig(
        "b04_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("b06_v1", "studio_l1", "workbench"): _rig(
        "b06_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b06_v1", "studio_l2", "workbench"): _rig(
        "b06_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b06_v1", "studio_l3", "workbench"): _rig(
        "b06_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("b13_v1", "studio_l1", "workbench"): _rig(
        "b13_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b13_v1", "studio_l2", "workbench"): _rig(
        "b13_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b13_v1", "studio_l3", "workbench"): _rig(
        "b13_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("m05_v1", "studio_l1", "desk_top"): _rig(
        "m05_studio_l1_desk_top",
        # Calibrated from the M05 Human Viewer in world coordinates, then
        # converted to the Studio L1 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.620607000, -0.759925000, 0.738277000),
        context_target=(0.201087818, 0.564659206, 0.636816125),
        interaction_eye=(-0.257993000, -0.214262000, 0.555193000),
        interaction_target=(0.199074354, -0.135542735, 0.446598754),
    ),
    ("m05_v1", "studio_l2", "desk_top"): _rig(
        "m05_studio_l2_desk_top",
        # Calibrated from the M05 Human Viewer in world coordinates, then
        # converted to the rotated Studio L2 desk-top interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.809414777, -0.517977723, 0.631911000),
        context_target=(0.299765165, 0.572897091, 0.491511373),
        interaction_eye=(-0.303350227, -0.230896639, 0.567018000),
        interaction_target=(0.164415638, -0.232263822, 0.477055618),
    ),
    ("m05_v1", "studio_l3", "desk_top"): _rig(
        "m05_studio_l3_desk_top",
        # Calibrated from the M05 Human Viewer in world coordinates, then
        # converted to the rotated Studio L3 desk-top interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.730582651, -0.550561542, 0.535565000),
        context_target=(0.206341941, 0.698288051, 0.586032376),
        interaction_eye=(-0.269137426, -0.176568930, 0.596329000),
        interaction_target=(0.195959867, -0.175570606, 0.493452829),
    ),
    ("m07_v1", "studio_l1", "desk_top"): _rig(
        "m07_studio_l1_desk_top",
        # Calibrated from the M07 Human Viewer in world coordinates, then
        # converted to the Studio L1 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.767731000, -0.633188000, 0.523859000),
        context_target=(0.267463562, 0.534473993, 0.594127696),
        interaction_eye=(-0.284782000, -0.327856000, 0.755103000),
        interaction_target=(0.036654792, -0.332529690, 0.403596891),
    ),
    ("m07_v1", "studio_l2", "desk_top"): _rig(
        "m07_studio_l2_desk_top",
        # Calibrated from the M07 Human Viewer in world coordinates, then
        # converted to the rotated Studio L2 desk-top interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.770026694, -0.489183348, 0.618266000),
        context_target=(0.229144598, 0.664330716, 0.285006201),
        interaction_eye=(-0.189614039, -0.283830085, 0.655169000),
        interaction_target=(0.038852566, -0.251174116, 0.238471745),
    ),
    ("m07_v1", "studio_l3", "desk_top"): _rig(
        "m07_studio_l3_desk_top",
        # Calibrated from the M07 Human Viewer in world coordinates, then
        # converted to the rotated Studio L3 desk-top interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.838993676, -0.527339607, 0.726523000),
        context_target=(0.227817940, 0.575822687, 0.435068230),
        interaction_eye=(-0.115262796, -0.127723834, 0.537616000),
        interaction_target=(0.068987438, -0.115333188, 0.098527946),
    ),

    ("m01_v1", "studio_l1", "desk_top"): _rig(
        "m01_studio_l1_desk_top",
        # Calibrated from the M01 Human Viewer in world coordinates, then
        # converted to the Studio L1 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.449488000, -0.724429000, 0.469782000),
        context_target=(-0.623848431, 0.788231883, 0.818286656),
        interaction_eye=(-0.352334000, 0.314346000, 0.629069000),
        interaction_target=(0.091765476, 0.390833999, 0.474714730),
    ),
    ("m01_v1", "studio_l2", "desk_top"): _rig(
        "m01_studio_l2_desk_top",
        # Calibrated from the M01 Human Viewer in world coordinates, then
        # converted to the Studio L2 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.544927878, -0.668450165, 0.479897000),
        context_target=(-0.561977077, 0.842993724, 0.873910756),
        interaction_eye=(-0.408447745, 0.445303892, 0.745536000),
        interaction_target=(0.033312424, 0.468329351, 0.568850422),
    ),
    ("m01_v1", "studio_l3", "desk_top"): _rig(
        "m01_studio_l3_desk_top",
        # Calibrated from the M01 Human Viewer in world coordinates, then
        # converted to the Studio L3 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.609415087, -0.412687362, 0.657025000),
        context_target=(-0.504935530, 1.106401579, 0.308524348),
        interaction_eye=(-0.475963920, 0.456663046, 1.018588000),
        interaction_target=(-0.050419458, 0.470337641, 0.804988124),
    ),
    ("m03_v1", "studio_l1", "desk_top"): _rig(
        "m03_studio_l1_desk_top",
        # Calibrated from the M03 Human Viewer in world coordinates, then
        # converted to the Studio L1 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.478350000, -0.740788000, 0.874253000),
        context_target=(-0.407817493, 0.816366623, 0.772790461),
        interaction_eye=(0.035075000, 0.269901000, 0.622661000),
        interaction_target=(0.139376705, 0.267869871, 0.157884679),
    ),
    ("m03_v1", "studio_l2", "desk_top"): _rig(
        "m03_studio_l2_desk_top",
        # Calibrated from the M03 Human Viewer in world coordinates, then
        # converted to the Studio L2 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.538118999, -0.741130768, 0.489849000),
        context_target=(-0.387548230, 0.779165288, 0.815469315),
        interaction_eye=(-0.171485488, 0.201462509, 0.678733000),
        interaction_target=(0.158223302, 0.218648264, 0.335372000),
    ),
    ("m03_v1", "studio_l3", "desk_top"): _rig(
        "m03_studio_l3_desk_top",
        # Calibrated from the M03 Human Viewer in world coordinates, then
        # converted to the Studio L3 desk-top interaction-site frame. Preserve
        # the previous eye-to-target distances while matching the new view axes.
        context_eye=(-0.560452991, -0.586290877, 0.720425000),
        context_target=(-0.508127091, 0.968173831, 0.575851969),
        interaction_eye=(-0.130598041, 0.371006665, 0.681599000),
        interaction_target=(0.101763630, 0.397176603, 0.266601190),
    ),
    ("m13_v1", "studio_l1", "sliding_cabinet_a"): _rig(
        "m13_studio_l1_sliding_cabinet_a",
        # Calibrated from the M13 Human Viewer in world coordinates, then
        # converted to the Studio L1 sliding-cabinet interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-1.053554342, -1.034905004, -0.026755000),
        context_target=(-0.662536096, 0.342315952, -0.128573505),
        interaction_eye=(-0.735027535, -0.165310189, -0.138016000),
        interaction_target=(0.379222190, -0.121657922, -0.471994933),
    ),
    ("m13_v1", "studio_l2", "sliding_cabinet_a"): _rig(
        "m13_studio_l2_sliding_cabinet_a",
        # Calibrated from the M13 Human Viewer in world coordinates, then
        # converted to the Studio L2 sliding-cabinet interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-1.382646780, -0.932725736, -0.252878000),
        context_target=(-0.619218887, 0.262297754, -0.031299868),
        interaction_eye=(-0.675703384, -0.223050832, -0.211079000),
        interaction_target=(0.392790483, -0.311608982, -0.664376253),
    ),
    ("m13_v1", "studio_l3", "sliding_cabinet_a"): _rig(
        "m13_studio_l3_sliding_cabinet_a",
        # Calibrated from the M13 Human Viewer in world coordinates, then
        # converted to the Studio L3 sliding-cabinet interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-1.271627830, -0.905898331, -0.134470000),
        context_target=(-0.671047942, 0.397655326, -0.127294591),
        interaction_eye=(-0.799560248, -0.157944982, -0.071518000),
        interaction_target=(0.307042161, -0.047966311, -0.415517552),
    ),
    ("m14_v1", "studio_l1", "desk_drawer"): _rig(
        "m14_studio_l1_desk_drawer",
        # Calibrated from the M14 Human Viewer in world coordinates, then
        # converted to the Studio L1 desk-drawer interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-1.379827000, 0.172367000, 0.384707000),
        context_target=(-0.712444674, 1.559430841, 0.118897417),
        interaction_eye=(-0.810445000, 0.686545000, 0.567998000),
        interaction_target=(-0.472193379, 0.670393823, 0.232998029),
    ),
    ("m14_v1", "studio_l2", "desk_drawer"): _rig(
        "m14_studio_l2_desk_drawer",
        # Calibrated from the M14 Human Viewer in world coordinates, then
        # converted to the Studio L2 desk-drawer interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-1.367454773, 0.285452155, 0.265341000),
        context_target=(-0.723945465, 1.675903025, -0.038866684),
        interaction_eye=(-0.689754302, 0.666899954, 0.298484000),
        interaction_target=(-0.459020483, 0.661817346, -0.118212630),
    ),
    ("m14_v1", "studio_l3", "desk_drawer"): _rig(
        "m14_studio_l3_desk_drawer",
        # Calibrated from the M14 Human Viewer in world coordinates, then
        # converted to the Studio L3 desk-drawer interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-1.638578051, 0.032935639, 0.269583000),
        context_target=(-0.507460720, 1.081095796, 0.518447926),
        interaction_eye=(-0.948339769, 0.669269609, 0.680261000),
        interaction_target=(-0.550450800, 0.648300969, 0.419215649),
    ),
    ("m15_v1", "studio_l1", "dresser_cabinet_b"): _rig(
        "m15_studio_l1_dresser_cabinet_b",
        # Calibrated from the M15 Human Viewer in world coordinates, then
        # converted to the Studio L1 dresser-cabinet interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.870610713, -0.743231520, 0.358195000),
        context_target=(-0.496149281, 1.050023600, 0.607021482),
        interaction_eye=(-0.642351922, 0.184741858, 0.642141000),
        interaction_target=(0.685729161, 0.171378047, 0.033025250),
    ),
    ("m15_v1", "studio_l2", "dresser_cabinet_b"): _rig(
        "m15_studio_l2_dresser_cabinet_b",
        # Calibrated from the M15 Human Viewer in world coordinates, then
        # converted to the Studio L2 dresser-cabinet interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.796794474, -0.764945614, 0.458443000),
        context_target=(-0.519493260, 1.062673667, 0.486935488),
        interaction_eye=(-0.507283731, 0.262020102, 0.764171000),
        interaction_target=(0.556409544, 0.219680120, -0.236710586),
    ),
    ("m15_v1", "studio_l3", "dresser_cabinet_b"): _rig(
        "m15_studio_l3_dresser_cabinet_b",
        # Calibrated from the M15 Human Viewer in world coordinates, then
        # converted to the Studio L3 dresser-cabinet interaction-site frame.
        # Preserve the previous eye-to-target distances while matching the new
        # view axes.
        context_eye=(-0.802240863, -0.638968491, 0.180257000),
        context_target=(-0.637966233, 1.177264526, 0.483920316),
        interaction_eye=(-0.702811389, 0.144463326, 0.652161000),
        interaction_target=(0.627522577, 0.191834226, 0.049693157),
    ),
    ("b11_v1", "studio_l1", "desk_top"): _rig(
        "b11_studio_l1_desk_top",
        context_eye=(0.409253000, -0.504340000, 1.109614000),
        context_target=(-0.349803777, 0.335767119, 0.033485365),
        interaction_eye=(0.057585000, -0.132470000, 1.134185000),
        interaction_target=(0.245610922, -0.130485340, 0.696529527),
    ),
    ("b11_v1", "studio_l2", "desk_top"): _rig(
        "b11_studio_l2_desk_top",
        context_eye=(0.238143021, -0.770966619, 1.108163000),
        context_target=(-0.381676094, 0.333801937, 0.194209630),
        interaction_eye=(-0.210442920, -0.027815591, 1.052608000),
        interaction_target=(0.120893248, -0.089111329, 0.715919449),
    ),
    ("b11_v1", "studio_l3", "desk_top"): _rig(
        "b11_studio_l3_desk_top",
        context_eye=(0.185414373, -0.828313507, 0.943289000),
        context_target=(-0.373945484, 0.325662963, 0.051406265),
        interaction_eye=(-0.380973948, -0.058377441, 0.934025000),
        interaction_target=(-0.005618265, -0.093331486, 0.642844091),
    ),
    ("m09_v1", "kitchen_l1", "dining_table"): _rig(
        "m09_kitchen_l1_dining_table",
        context_eye=(-0.632435000, -0.511819000, 0.618428000),
        context_target=(-0.040180576, 0.800134663, 0.466729999),
        interaction_eye=(-0.194512000, -0.139752000, 0.507646000),
        interaction_target=(0.633258388, 0.567078888, 0.015014007),
    ),
    ("m09_v1", "kitchen_l2", "dining_table"): _rig(
        "m09_kitchen_l2_dining_table",
        context_eye=(-0.647451000, -0.657289000, 0.864777000),
        context_target=(0.010018801, 0.608797084, 0.620319418),
        interaction_eye=(-0.185694000, -0.144819000, 0.547271000),
        interaction_target=(0.550855758, 0.639557480, 0.027898307),
    ),
    ("m09_v1", "kitchen_l3", "dining_table"): _rig(
        "m09_kitchen_l3_dining_table",
        context_eye=(-0.535892000, -0.489654000, 0.455747000),
        context_target=(-0.180871529, 0.907499061, 0.585838518),
        interaction_eye=(-0.194853000, -0.150343000, 0.467396000),
        interaction_target=(0.697023080, 0.537625027, 0.068952727),
    ),
    ("b01_v1", "kitchen_l1", "dining_table"): _rig(
        "b01_kitchen_l1_dining_table",
        # Calibrated from the requested B01 L1 world camera poses and converted
        # to the dining-table interaction-site frame. Preserve the existing
        # eye-to-target distances while matching the requested camera axes.
        context_eye=(0.719630000, -0.878007000, 0.790540000),
        context_target=(-0.112552862, 0.299085969, 0.660448391),
        interaction_eye=(-0.301943000, -0.014713000, 0.730838000),
        interaction_target=(0.572246361, -0.009296873, -0.083571987),
        context_up=(-0.052004827, 0.073305648, 0.995952700),
        interaction_up=(0.681625956, 0.004368073, 0.731687758),
    ),
    ("b01_v1", "kitchen_l2", "dining_table"): _rig(
        "b01_kitchen_l2_dining_table",
        # Calibrated from the requested B01 L2 world camera poses and converted
        # to the dining-table interaction-site frame. Preserve the existing
        # eye-to-target distances while matching the requested camera axes.
        context_eye=(0.447192000, -0.729542000, 0.812086000),
        context_target=(-0.398258742, 0.347598270, 0.343063717),
        interaction_eye=(-0.245594000, 0.091555000, 0.801918000),
        interaction_target=(0.482944433, 0.085188976, -0.145019816),
        context_up=(-0.200020398, 0.254941439, 0.946042654),
        interaction_up=(0.792531458, -0.006990350, 0.609790967),
    ),
    ("b01_v1", "kitchen_l3", "dining_table"): _rig(
        "b01_kitchen_l3_dining_table",
        # Calibrated from the requested B01 L3 world camera poses and converted
        # to the dining-table interaction-site frame. Preserve the existing
        # eye-to-target distances while matching the requested camera axes.
        context_eye=(0.644485000, -0.922494000, 0.814444000),
        context_target=(-0.283450081, 0.177983834, 0.663165990),
        interaction_eye=(-0.361963000, 0.082636000, 0.990343000),
        interaction_target=(0.581638079, 0.060198950, 0.257803873),
        context_up=(-0.067375900, 0.079900769, 0.994523180),
        interaction_up=(0.612945512, -0.014465239, 0.789992757),
    ),
    ("b05_v1", "kitchen_l1", "dining_table"): _rig(
        "b05_kitchen_l1_dining_table",
        # Calibrated from the requested L1 context-camera world pose. The
        # target distance remains the previous B05 L1 calibration.
        context_eye=(0.505535000, -1.025030000, 0.915145000),
        context_target=(-0.276014729, 0.191520639, 0.850456786),
        # Calibrated from the requested L1 interaction-camera world pose. The
        # target distance remains the previous B05 L1 calibration.
        interaction_eye=(-0.267141000, 0.039460000, 0.741270000),
        interaction_target=(0.582185480, 0.062256012, -0.098744805),
        context_up=(-0.024155559, 0.037602057, 0.999000798),
        interaction_up=(0.702817404, 0.018863693, 0.711120143),
    ),
    ("b05_v1", "kitchen_l2", "dining_table"): _rig(
        "b05_kitchen_l2_dining_table",
        # Calibrated from the requested L2 context-camera world pose. The
        # target distance remains the previous B05 L2 calibration.
        context_eye=(0.524017000, -1.313985000, 0.944146000),
        context_target=(-0.131832348, -0.026171956, 1.024141107),
        # Calibrated from the requested L2 interaction-camera world pose. The
        # target distance remains the previous B05 L2 calibration.
        interaction_eye=(-0.300153000, 0.038899000, 0.778645000),
        interaction_target=(0.602101126, 0.022508313, -0.004405982),
        context_up=(0.025080169, -0.049249406, 0.998471573),
        interaction_up=(0.655285188, -0.011902331, 0.755287797),
    ),
    ("b05_v1", "kitchen_l3", "dining_table"): _rig(
        "b05_kitchen_l3_dining_table",
        # Calibrated from the requested L3 context-camera world pose. The
        # target distance remains the previous B05 L3 calibration.
        context_eye=(0.453317000, -1.298045000, 1.005853000),
        context_target=(0.079805078, 0.100180969, 0.984569259),
        # Calibrated from the requested L3 interaction-camera world pose. The
        # target distance remains the previous B05 L3 calibration.
        interaction_eye=(-0.405181000, 0.045035000, 0.741382000),
        interaction_target=(0.631569853, 0.021013895, 0.148026588),
        context_up=(-0.003795182, 0.014206498, 0.999891880),
        interaction_up=(0.496489800, -0.011502510, 0.867966342),
    ),
    ("b09_v1", "studio_l1", "workbench"): _rig(
        "b09_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b09_v1", "studio_l2", "workbench"): _rig(
        "b09_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b09_v1", "studio_l3", "workbench"): _rig(
        "b09_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    # B10 retains independent rig objects while matching B01's calibrated
    # dining-table context and interaction cameras for each kitchen layout.
    ("b10_v1", "kitchen_l1", "dining_table"): _rig(
        "b10_kitchen_l1_dining_table",
        context_eye=(0.719630000, -0.878007000, 0.790540000),
        context_target=(-0.112552862, 0.299085969, 0.660448391),
        interaction_eye=(-0.301943000, -0.014713000, 0.730838000),
        interaction_target=(0.572246361, -0.009296873, -0.083571987),
        context_up=(-0.052004827, 0.073305648, 0.995952700),
        interaction_up=(0.681625956, 0.004368073, 0.731687758),
    ),
    ("b10_v1", "kitchen_l2", "dining_table"): _rig(
        "b10_kitchen_l2_dining_table",
        context_eye=(0.447192000, -0.729542000, 0.812086000),
        context_target=(-0.398258742, 0.347598270, 0.343063717),
        interaction_eye=(-0.245594000, 0.091555000, 0.801918000),
        interaction_target=(0.482944433, 0.085188976, -0.145019816),
        context_up=(-0.200020398, 0.254941439, 0.946042654),
        interaction_up=(0.792531458, -0.006990350, 0.609790967),
    ),
    ("b10_v1", "kitchen_l3", "dining_table"): _rig(
        "b10_kitchen_l3_dining_table",
        context_eye=(0.644485000, -0.922494000, 0.814444000),
        context_target=(-0.283450081, 0.177983834, 0.663165990),
        interaction_eye=(-0.361963000, 0.082636000, 0.990343000),
        interaction_target=(0.581638079, 0.060198950, 0.257803873),
        context_up=(-0.067375900, 0.079900769, 0.994523180),
        interaction_up=(0.612945512, -0.014465239, 0.789992757),
    ),
    ("b02_v1", "studio_l1", "workbench"): _rig(
        "b02_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b02_v1", "studio_l2", "workbench"): _rig(
        "b02_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b02_v1", "studio_l3", "workbench"): _rig(
        "b02_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("b12_v1", "studio_l1", "workbench"): _rig(
        "b12_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.412401550, 0.339560300, 0.673159000),
        interaction_target=(0.513884620, 0.051771720, 0.149544960),
        interaction_up=(0.453641917, -0.140942194, 0.879968357),
    ),
    ("b12_v1", "studio_l2", "workbench"): _rig(
        "b12_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        # Calibrated from the requested L2 world camera pose and converted to
        # the rotated workbench interaction-site frame. Preserve the existing
        # target distance while matching the requested camera axes.
        interaction_eye=(-0.334484170, 0.155597480, 0.729317000),
        interaction_target=(0.465429260, -0.188498400, 0.053494140),
        interaction_up=(0.563219770, -0.242277390, 0.789990600),
    ),
    ("b12_v1", "studio_l3", "workbench"): _rig(
        "b12_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        # Calibrated from the requested L3 world camera pose and converted to
        # the rotated workbench interaction-site frame. Preserve the existing
        # target distance while matching the requested camera axes.
        interaction_eye=(-0.482585036, 0.289413852, 0.762645000),
        interaction_target=(0.449464798, -0.133809388, 0.353786627),
        interaction_up=(0.337736532, -0.153357537, 0.928663286),
    ),
    ("b14_v1", "studio_l1", "workbench"): _rig(
        "b14_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b14_v1", "studio_l2", "workbench"): _rig(
        "b14_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b14_v1", "studio_l3", "workbench"): _rig(
        "b14_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("b15_v1", "studio_l1", "workbench"): _rig(
        "b15_studio_l1_workbench",
        context_eye=(1.011462020, -0.719106767, 1.145252000),
        context_target=(-0.215822484, 0.051119024, 0.970538010),
        interaction_eye=(-0.144155018, -0.018095055, 0.690360000),
        interaction_target=(0.362931309, -0.070447410, -0.286943785),
    ),
    ("b15_v1", "studio_l2", "workbench"): _rig(
        "b15_studio_l2_workbench",
        context_eye=(0.453687374, -0.492947200, 0.839571000),
        context_target=(-0.599530160, 0.287673800, 0.198194562),
        interaction_eye=(-0.304073207, -0.028492575, 0.835424000),
        interaction_target=(0.538170823, -0.065602680, 0.125319883),
    ),
    ("b15_v1", "studio_l3", "workbench"): _rig(
        "b15_studio_l3_workbench",
        context_eye=(0.453590236, -0.411104222, 1.095938000),
        context_target=(-0.616268741, 0.245623995, 0.351552492),
        interaction_eye=(-0.206275975, -0.046952679, 0.928390000),
        interaction_target=(0.416597978, -0.052524833, 0.018996255),
    ),
    ("m06_v1", "kitchen_l1", "sink"): _rig(
        "m06_kitchen_l1_sink",
        context_eye=(-1.099725000, -0.464678000, 0.675960000),
        context_target=(0.062209980, 0.249381685, 0.625701986),
        interaction_eye=(-0.388829000, -0.262979000, 0.650182000),
        interaction_target=(0.759429425, -0.088208159, 0.460556732),
    ),
    ("m06_v1", "kitchen_l2", "sink"): _rig(
        "m06_kitchen_l2_sink",
        context_eye=(-0.934076980, -0.382134050, 0.664374000),
        context_target=(0.197254802, 0.352698518, 0.457953322),
        interaction_eye=(-0.439727983, -0.319038024, 0.573622000),
        interaction_target=(0.684345977, 0.023812684, 0.636160986),
    ),
    ("m06_v1", "kitchen_l3", "sink"): _rig(
        "m06_kitchen_l3_sink",
        context_eye=(-0.767900952, -0.886637041, 1.027112000),
        context_target=(-0.023093244, 0.250796025, 0.908758727),
        interaction_eye=(-0.191451978, -0.410947010, 0.587485000),
        interaction_target=(0.665266083, 0.388956555, 0.481709531),
    ),
    ("b08_v1", "kitchen_l1", "counter_top"): _rig(
        "b08_kitchen_l1_counter_top",
        context_eye=(-0.099089000, -0.779277000, 0.663834000),
        context_target=(-0.389431317, 0.385943239, 0.537885064),
        interaction_eye=(-0.54, 0.00, 0.55),
        interaction_target=(0.00, 0.00, 0.10),
        context_up=(-0.025220283, 0.101216345, 0.994544714),
    ),
    ("b08_v1", "kitchen_l2", "counter_top"): _rig(
        "b08_kitchen_l2_counter_top",
        context_eye=(-0.419867000, -0.614952000, 0.807382000),
        context_target=(-0.516530737, 0.356244334, 0.096513655),
        interaction_eye=(-0.54, 0.00, 0.55),
        interaction_target=(0.00, 0.00, 0.10),
        context_up=(-0.058310900, 0.585847627, 0.808320727),
    ),
    ("b08_v1", "kitchen_l3", "counter_top"): _rig(
        "b08_kitchen_l3_counter_top",
        context_eye=(-0.409827000, -0.656310000, 0.721313000),
        context_target=(-0.453424739, 0.455191073, 0.251669943),
        interaction_eye=(-0.54, 0.00, 0.55),
        interaction_target=(0.00, 0.00, 0.10),
        context_up=(-0.015244298, 0.388660345, 0.921254985),
    ),
}


class CameraRigRegistry:
    _rigs: dict[str, CameraRigSpec] = {rig.rig_id: rig for rig in _CAMERA_RIGS}

    @classmethod
    def get(
        cls,
        rig_id: str,
        *,
        layout_id: str | None = None,
        site_id: str | None = None,
        task_id: str | None = None,
    ) -> CameraRigSpec:
        if (
            task_id is not None
            and layout_id is not None
            and site_id is not None
        ):
            task_override = TASK_LAYOUT_RIG_OVERRIDES.get(
                (task_id, layout_id, site_id)
            )
            if task_override is not None:
                return task_override
        if layout_id is not None and site_id is not None:
            override = LAYOUT_RIG_OVERRIDES.get((layout_id, site_id))
            if override is not None:
                return override
        try:
            return cls._rigs[rig_id]
        except KeyError as exc:
            known_rigs = ", ".join(sorted(cls._rigs)) or "<empty>"
            raise ValueError(
                f"Unknown camera rig {rig_id!r}. Known rigs: {known_rigs}."
            ) from exc

    @classmethod
    def ids(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._rigs))
