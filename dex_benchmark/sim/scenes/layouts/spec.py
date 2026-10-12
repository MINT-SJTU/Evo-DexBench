from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

from dex_benchmark.sim.layout_types import LayoutId, SceneClass


@dataclass(frozen=True)
class PoseSpec:
    position_xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def transform_point(self, local_xyz: tuple[float, float, float]) -> np.ndarray:
        # [3] -> [3]: rotate a local point into the layout/world frame.
        return self.rotation_matrix @ np.asarray(local_xyz, dtype=np.float32) + np.asarray(
            self.position_xyz, dtype=np.float32
        )

    def transform_pose(self, local_pose: "PoseSpec") -> "PoseSpec":
        return PoseSpec(
            position_xyz=tuple(
                float(value) for value in self.transform_point(local_pose.position_xyz)
            ),
            quaternion_wxyz=quat_multiply_wxyz(
                self.quaternion_wxyz,
                local_pose.quaternion_wxyz,
            ),
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        w, x, y, z = self.quaternion_wxyz
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )


def yaw_quaternion_wxyz(yaw_rad: float) -> tuple[float, float, float, float]:
    half_yaw = 0.5 * yaw_rad
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def quat_multiply_wxyz(
    left_wxyz: tuple[float, float, float, float],
    right_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left_w, left_x, left_y, left_z = left_wxyz
    right_w, right_x, right_y, right_z = right_wxyz
    return (
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    )


InteractionTargetSource: TypeAlias = Literal[
    "layout_entity",
    "external_fixture",
]


@dataclass(frozen=True)
class InteractionSiteSpec:
    """One layout-specific frame from which a task interacts with the scene.

    ``robot_base_center_pose`` is the single-manipulator pose and the fallback
    center for two manipulators. Sites may provide a distinct bimanual center
    when the single base is intentionally off the site's local centerline.
    """

    site_id: str
    frame_pose: PoseSpec
    robot_base_center_pose: PoseSpec
    camera_rig_id: str
    bimanual_base_spacing_m: float = 0.6
    bimanual_robot_base_center_pose: PoseSpec | None = None
    target_entity_name: str | None = None
    target_entity_source: InteractionTargetSource | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.site_id.strip():
            raise ValueError("Interaction site_id must be non-empty.")
        if not self.camera_rig_id.strip():
            raise ValueError(
                f"Interaction site {self.site_id!r} camera_rig_id must be "
                "non-empty."
            )
        if (
            not math.isfinite(self.bimanual_base_spacing_m)
            or self.bimanual_base_spacing_m <= 0.0
        ):
            raise ValueError(
                f"Interaction site {self.site_id!r} bimanual_base_spacing_m "
                f"must be finite and positive, got "
                f"{self.bimanual_base_spacing_m}."
            )
        has_target_name = self.target_entity_name is not None
        has_target_source = self.target_entity_source is not None
        if has_target_name != has_target_source:
            raise ValueError(
                f"Interaction site {self.site_id!r} must define target_entity_name "
                "and target_entity_source together."
            )
        if (
            self.target_entity_name is not None
            and not self.target_entity_name.strip()
        ):
            raise ValueError(
                f"Interaction site {self.site_id!r} target_entity_name must be "
                "non-empty when provided."
            )


@dataclass(frozen=True)
class PhysicalMaterialSpec:
    """Contact material shared by collision shapes owned by one layout item."""

    static_friction: float
    dynamic_friction: float
    restitution: float = 0.0

    def __post_init__(self) -> None:
        if self.static_friction < 0.0:
            raise ValueError(
                "Physical material static_friction must be non-negative, got "
                f"{self.static_friction}."
            )
        if self.dynamic_friction < 0.0:
            raise ValueError(
                "Physical material dynamic_friction must be non-negative, got "
                f"{self.dynamic_friction}."
            )
        if self.dynamic_friction > self.static_friction:
            raise ValueError(
                "Physical material dynamic_friction cannot exceed static_friction, "
                f"got dynamic={self.dynamic_friction}, static={self.static_friction}."
            )
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(
                "Physical material restitution must be in [0, 1], got "
                f"{self.restitution}."
            )


def _validate_light_color(
    color_rgb: tuple[float, float, float],
    *,
    light_name: str,
) -> None:
    color = np.asarray(color_rgb, dtype=np.float64)
    if color.shape != (3,) or not np.isfinite(color).all():
        raise ValueError(
            f"{light_name} color_rgb must contain three finite values, got "
            f"{color_rgb}."
        )
    if (color < 0.0).any():
        raise ValueError(
            f"{light_name} color_rgb must be non-negative, got {color_rgb}."
        )


@dataclass(frozen=True)
class PointLightSpec:
    """One layout-anchored point light used by the nominal renderer setup."""

    position_xyz: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    shadow: bool = True
    shadow_near: float = 0.1
    shadow_far: float = 10.0
    shadow_map_size: int = 2048

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xyz, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(
                "Point-light position_xyz must contain three finite values, got "
                f"{self.position_xyz}."
            )
        _validate_light_color(self.color_rgb, light_name="Point-light")
        if self.shadow_near <= 0.0 or self.shadow_far <= self.shadow_near:
            raise ValueError(
                "Point-light shadow planes must satisfy 0 < shadow_near < "
                f"shadow_far, got {self.shadow_near} and {self.shadow_far}."
            )
        if self.shadow_map_size <= 0:
            raise ValueError("Point-light shadow_map_size must be positive.")


@dataclass(frozen=True)
class SpotLightSpec:
    """One layout-anchored cone light, such as sunlight entering a window."""

    position_xyz: tuple[float, float, float]
    direction_xyz: tuple[float, float, float]
    inner_fov_rad: float
    outer_fov_rad: float
    color_rgb: tuple[float, float, float]
    shadow: bool = True
    shadow_near: float = 0.1
    shadow_far: float = 10.0
    shadow_map_size: int = 2048

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xyz, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(
                "Spot-light position_xyz must contain three finite values, got "
                f"{self.position_xyz}."
            )
        direction = np.asarray(self.direction_xyz, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError(
                "Spot-light direction_xyz must contain three finite values, got "
                f"{self.direction_xyz}."
            )
        direction_norm = float(np.linalg.norm(direction))
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Spot-light direction_xyz must be normalized, got norm "
                f"{direction_norm}."
            )
        _validate_light_color(self.color_rgb, light_name="Spot-light")
        if not 0.0 < self.inner_fov_rad <= self.outer_fov_rad < math.pi:
            raise ValueError(
                "Spot-light fields of view must satisfy 0 < inner_fov_rad <= "
                f"outer_fov_rad < pi, got {self.inner_fov_rad} and "
                f"{self.outer_fov_rad}."
            )
        if self.shadow_near <= 0.0 or self.shadow_far <= self.shadow_near:
            raise ValueError(
                "Spot-light shadow planes must satisfy 0 < shadow_near < "
                f"shadow_far, got {self.shadow_near} and {self.shadow_far}."
            )
        if self.shadow_map_size <= 0:
            raise ValueError("Spot-light shadow_map_size must be positive.")


@dataclass(frozen=True)
class RectangularLightApertureSpec:
    """A rectangular plane that admits parallel light into a scene."""

    center_xyz: tuple[float, float, float]
    axis_u_xyz: tuple[float, float, float]
    axis_v_xyz: tuple[float, float, float]
    half_size_uv: tuple[float, float]
    edge_softness_m: float = 0.01

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xyz, dtype=np.float64)
        axis_u = np.asarray(self.axis_u_xyz, dtype=np.float64)
        axis_v = np.asarray(self.axis_v_xyz, dtype=np.float64)
        half_size = np.asarray(self.half_size_uv, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError(
                "Light-aperture center_xyz must contain three finite values, got "
                f"{self.center_xyz}."
            )
        for axis_name, axis in (("axis_u_xyz", axis_u), ("axis_v_xyz", axis_v)):
            if axis.shape != (3,) or not np.isfinite(axis).all():
                raise ValueError(
                    f"Light-aperture {axis_name} must contain three finite values."
                )
            axis_norm = float(np.linalg.norm(axis))
            if not math.isclose(axis_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(
                    f"Light-aperture {axis_name} must be normalized, got norm "
                    f"{axis_norm}."
                )
        if not math.isclose(
            float(np.dot(axis_u, axis_v)),
            0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("Light-aperture axes must be orthogonal.")
        if (
            half_size.shape != (2,)
            or not np.isfinite(half_size).all()
            or (half_size <= 0.0).any()
        ):
            raise ValueError(
                "Light-aperture half_size_uv must contain two positive finite "
                f"values, got {self.half_size_uv}."
            )
        if not 0.0 < self.edge_softness_m < float(half_size.min()):
            raise ValueError(
                "Light-aperture edge_softness_m must be positive and smaller "
                f"than both half sizes, got {self.edge_softness_m}."
            )


@dataclass(frozen=True)
class DirectionalLightSpec:
    """One parallel light field, such as sunlight entering through a window."""

    direction_xyz: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    position_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    aperture: RectangularLightApertureSpec | None = None
    shadow: bool = True
    shadow_scale: float = 10.0
    shadow_near: float = -10.0
    shadow_far: float = 10.0
    shadow_map_size: int = 2048

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xyz, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError(
                "Directional-light position_xyz must contain three finite values, "
                f"got {self.position_xyz}."
            )
        direction = np.asarray(self.direction_xyz, dtype=np.float64)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError(
                "Directional-light direction_xyz must contain three finite values, "
                f"got {self.direction_xyz}."
            )
        direction_norm = float(np.linalg.norm(direction))
        if not math.isclose(direction_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "Directional-light direction_xyz must be normalized, got norm "
                f"{direction_norm}."
            )
        _validate_light_color(self.color_rgb, light_name="Directional-light")
        if self.aperture is not None and not self.shadow:
            raise ValueError(
                "Directional-light rectangular aperture requires shadow=True so "
                "interior occluders can contribute to the same light visibility."
            )
        if self.shadow_scale <= 0.0:
            raise ValueError("Directional-light shadow_scale must be positive.")
        if self.shadow_far <= self.shadow_near:
            raise ValueError(
                "Directional-light shadow planes must satisfy shadow_near < "
                f"shadow_far, got {self.shadow_near} and {self.shadow_far}."
            )
        if self.shadow_map_size <= 0:
            raise ValueError("Directional-light shadow_map_size must be positive.")


@dataclass(frozen=True)
class LayoutLightingSpec:
    """Fixed nominal lighting associated with one physical scene layout."""

    ambient_rgb: tuple[float, float, float]
    point_lights: tuple[PointLightSpec, ...] = ()
    spot_lights: tuple[SpotLightSpec, ...] = ()
    directional_lights: tuple[DirectionalLightSpec, ...] = ()

    def __post_init__(self) -> None:
        _validate_light_color(self.ambient_rgb, light_name="Ambient-light")
        if (
            not self.point_lights
            and not self.spot_lights
            and not self.directional_lights
        ):
            raise ValueError(
                "Layout lighting must contain at least one direct light source."
            )


@dataclass(frozen=True)
class StaticBoxSpec:
    name: str
    half_size_xyz: tuple[float, float, float]
    pose: PoseSpec
    color_rgba: tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    add_collision: bool = True
    body_type: Literal["static", "kinematic", "dynamic"] = "kinematic"
    episode_position_jitter_xy: tuple[float, float] | None = None
    episode_spawn_probability: float = 1.0
    physical_material: PhysicalMaterialSpec | None = None


@dataclass(frozen=True)
class ProceduralTableSpec:
    name: str
    pose: PoseSpec
    tabletop_half_size_xyz: tuple[float, float, float] = (1.0, 0.45, 0.025)
    table_height: float = 0.75
    leg_half_size_xyz: tuple[float, float, float] = (0.025, 0.025, 0.35)
    leg_inset_xy: tuple[float, float] = (0.08, 0.08)
    tabletop_color_rgba: tuple[float, float, float, float] = (0.62, 0.66, 0.63, 1.0)
    leg_color_rgba: tuple[float, float, float, float] = (0.28, 0.30, 0.29, 1.0)
    body_type: str = "kinematic"


@dataclass(frozen=True)
class CollisionBoxSpec:
    """One box in a compound collision shape, in the visual mesh frame."""

    half_size_xyz: tuple[float, float, float]
    pose: PoseSpec = field(
        default_factory=lambda: PoseSpec(position_xyz=(0.0, 0.0, 0.0))
    )

    def __post_init__(self) -> None:
        half_size = np.asarray(self.half_size_xyz, dtype=np.float64)
        if half_size.shape != (3,) or not np.isfinite(half_size).all():
            raise ValueError(
                "CollisionBoxSpec half_size_xyz must contain three finite values."
            )
        if (half_size <= 0.0).any():
            raise ValueError(
                "CollisionBoxSpec half_size_xyz must contain positive values, "
                f"got {self.half_size_xyz}."
            )


@dataclass(frozen=True)
class CollisionCylinderSpec:
    """One Z-axis cylinder in a compound collision shape's mesh frame."""

    radius_m: float
    half_length_m: float
    pose: PoseSpec = field(
        default_factory=lambda: PoseSpec(position_xyz=(0.0, 0.0, 0.0))
    )

    def __post_init__(self) -> None:
        dimensions = np.asarray(
            (self.radius_m, self.half_length_m), dtype=np.float64
        )
        if not np.isfinite(dimensions).all() or (dimensions <= 0.0).any():
            raise ValueError(
                "CollisionCylinderSpec radius_m and half_length_m must be "
                f"finite and positive, got {tuple(dimensions)}."
            )


@dataclass(frozen=True)
class StaticMeshSpec:
    name: str
    mesh_path: str | Path
    pose: PoseSpec
    collision_mesh_path: str | Path | None = None
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision_type: Literal[
        "none",
        "nonconvex",
        "convex",
        "multiple_convex",
        "compound_boxes",
        "compound_primitives",
    ] = "nonconvex"
    collision_boxes: tuple[CollisionBoxSpec, ...] = ()
    collision_cylinders: tuple[CollisionCylinderSpec, ...] = ()
    body_type: Literal["static", "kinematic", "dynamic"] = "static"
    episode_position_jitter_xy: tuple[float, float] | None = None
    episode_spawn_probability: float = 1.0
    physical_material: PhysicalMaterialSpec | None = None


@dataclass(frozen=True)
class MJCFActorSpec:
    name: str
    asset_path: str | Path
    pose: PoseSpec
    scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    visual_groups: tuple[int, ...] = (0, 2)
    body_type: Literal["static", "kinematic", "dynamic"] = "static"


@dataclass(frozen=True)
class YCBActorSpec:
    name: str
    model_id: str
    pose: PoseSpec
    scale_multiplier: float = 1.0
    scale_xyz_multiplier: tuple[float, float, float] = (1.0, 1.0, 1.0)
    density_multiplier: float = 1.0
    body_type: Literal["static", "kinematic", "dynamic"] = "dynamic"
    add_collision: bool = True
    add_visual: bool = True
    episode_position_jitter_xy: tuple[float, float] | None = None
    episode_spawn_probability: float = 1.0
    physical_material: PhysicalMaterialSpec | None = None


@dataclass(frozen=True)
class TexturedPlaneSpec:
    name: str
    texture_path: str | Path
    half_size_xy: tuple[float, float]
    pose: PoseSpec
    visual_pose: PoseSpec = field(
        default_factory=lambda: PoseSpec(position_xyz=(0.0, 0.0, 0.0))
    )
    texture_repeat_uv: tuple[float, float] = (1.0, 1.0)
    collision_half_size_xyz: tuple[float, float, float] | None = None
    body_type: Literal["static", "kinematic"] = "static"
    mipmap_levels: int = 1


@dataclass(frozen=True)
class JointPositionLatchSpec:
    """Finite-torque joint brake with an optional re-latching behavior."""

    stiffness: float
    damping: float
    force_limit: float
    relatch: bool = True
    """Follow externally reached positions after the brake yields."""

    def __post_init__(self) -> None:
        values = np.asarray(
            (self.stiffness, self.damping, self.force_limit),
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError(
                "Joint position latch parameters must be finite, got "
                f"stiffness={self.stiffness}, damping={self.damping}, "
                f"force_limit={self.force_limit}."
            )
        if self.stiffness <= 0.0:
            raise ValueError(
                "Joint position latch stiffness must be positive, got "
                f"{self.stiffness}."
            )
        if self.damping < 0.0:
            raise ValueError(
                "Joint position latch damping must be non-negative, got "
                f"{self.damping}."
            )
        if self.force_limit <= 0.0:
            raise ValueError(
                "Joint position latch force_limit must be positive, got "
                f"{self.force_limit}."
            )

    @property
    def yield_angle_rad(self) -> float:
        """Maximum elastic angular error before the latch begins to slip."""

        return self.force_limit / self.stiffness


@dataclass(frozen=True)
class ArticulatedAssetSpec:
    name: str
    asset_path: str | Path
    pose: PoseSpec
    asset_format: Literal["urdf", "mjcf"] = "urdf"
    scale: float = 1.0
    scale_xyz: tuple[float, float, float] | None = None
    fix_root_link: bool = True
    load_multiple_collisions_from_file: bool = True
    disable_self_collisions: bool = False
    initial_qpos: tuple[float, ...] | None = None
    episode_position_jitter_xy: tuple[float, float] | None = None
    joint_initial_qpos: dict[str, float] = field(default_factory=dict)
    default_joint_friction: float | None = None
    joint_friction: dict[str, float] = field(default_factory=dict)
    joint_position_latches: dict[str, JointPositionLatchSpec] = field(
        default_factory=dict
    )
    link_disable_gravity: tuple[str, ...] = ()
    physical_material: PhysicalMaterialSpec | None = None

    def __post_init__(self) -> None:
        if self.scale_xyz is not None:
            scale_xyz = np.asarray(self.scale_xyz, dtype=float)
            if (
                scale_xyz.shape != (3,)
                or not np.isfinite(scale_xyz).all()
                or (scale_xyz <= 0.0).any()
            ):
                raise ValueError(
                    "Articulated asset scale_xyz must contain three finite "
                    f"positive values, got {self.scale_xyz}."
                )
        if self.default_joint_friction is not None and self.default_joint_friction < 0.0:
            raise ValueError(
                "Articulated asset default_joint_friction must be non-negative, got "
                f"{self.default_joint_friction} for {self.name!r}."
            )
        negative_joint_friction = {
            joint_name: friction
            for joint_name, friction in self.joint_friction.items()
            if friction < 0.0
        }
        if negative_joint_friction:
            raise ValueError(
                f"Articulated asset {self.name!r} has negative joint friction: "
                f"{negative_joint_friction}."
            )


@dataclass(frozen=True)
class ExternalSceneSpec:
    name: str
    backend: Literal["robocasa", "replicacad", "ai2thor"]
    build_config_idx: int | None = None
    init_robot_base_pos: str | None = None
    remove_fixture_names: tuple[str, ...] = ()


LayoutBuildItemSpec: TypeAlias = (
    StaticBoxSpec
    | ProceduralTableSpec
    | StaticMeshSpec
    | MJCFActorSpec
    | YCBActorSpec
    | TexturedPlaneSpec
    | ArticulatedAssetSpec
    | ExternalSceneSpec
)


@dataclass(frozen=True)
class LayoutSupportSurfaceSpec:
    surface_id: str
    parent_entity_name: str
    pose: PoseSpec
    local_height_z: float
    initial_clearance_z: float = 0.0


@dataclass(frozen=True)
class RelativePlacementSpec:
    item_name: str
    surface_id: str
    bbox_center_pose: PoseSpec
    initial_clearance_z: float = 0.0


@dataclass(frozen=True)
class LayoutSpec:
    layout_id: LayoutId
    scene_class: SceneClass
    description: str
    build_items: list[LayoutBuildItemSpec] = field(default_factory=list)
    interaction_sites: dict[str, InteractionSiteSpec] = field(default_factory=dict)
    support_surfaces: dict[str, LayoutSupportSurfaceSpec] = field(default_factory=dict)
    relative_placements: dict[str, RelativePlacementSpec] = field(default_factory=dict)
    lighting: LayoutLightingSpec | None = None
    add_default_ground: bool = True
    settle_sim_steps: int = 0

    @property
    def static_boxes(self) -> list[StaticBoxSpec]:
        return [
            build_item
            for build_item in self.build_items
            if isinstance(build_item, StaticBoxSpec)
        ]

    def interaction_site(self, site_id: str) -> InteractionSiteSpec:
        try:
            site = self.interaction_sites[site_id]
        except KeyError as exc:
            known_sites = ", ".join(sorted(self.interaction_sites)) or "<empty>"
            raise ValueError(
                f"Layout {self.layout_id.value!r} does not define interaction "
                f"site {site_id!r}. Known sites: {known_sites}."
            ) from exc
        if site.site_id != site_id:
            raise ValueError(
                f"Layout {self.layout_id.value!r} stores interaction site key "
                f"{site_id!r} with mismatched site_id {site.site_id!r}."
            )
        return site
