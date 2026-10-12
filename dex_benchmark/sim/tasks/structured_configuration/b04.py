from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch
from dex_benchmark.assets import evodex_asset_dir
from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import AxisRandomizationSpec, GeneralizationAxis, GeneralizationParameterSpec, TaskGeneralizationSpec, positive_scale, scaled_physical_material
from dex_benchmark.sim.scenes.layouts import ActorBuildValues, CollisionBoxSpec, CollisionCylinderSpec, PhysicalMaterialSpec, PoseSpec, StaticMeshSpec
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import sapien
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover
    sapien = None
    Pose = None
    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls
        return decorator


B04_SETTINGS = TaskRegistry.register(TaskRegistry.load("b04_v1"))
B04_INTERACTION_SITE_ID = "workbench"
B04_ASSET_DIR = evodex_asset_dir() / "task_assets" / "b04"
B04_RING_MESH = B04_ASSET_DIR / "hanoi_ring.obj"
B04_STAND_MESH = B04_ASSET_DIR / "hanoi_stand.obj"
B04_STAND_NAME = "b04_hanoi_stand"
B04_RING_NAMES = ("b04_ring_bottom", "b04_ring_middle", "b04_ring_top")
B04_RING_COLORS_RGBA = ((0.88, 0.18, 0.12, 1.0), (0.12, 0.48, 0.92, 1.0), (0.98, 0.72, 0.10, 1.0))
B04_RING_OUTER_RADII_M = (0.075, 0.062, 0.050)
B04_RING_INNER_RADIUS_RATIO = 0.30
B04_RING_THICKNESS_M = 0.024
B04_RING_HALF_THICKNESS_M = 0.012
B04_COLLISION_SEGMENTS = 16
B04_BASE_RADIUS_M, B04_BASE_HEIGHT_M = 0.090, 0.018
B04_POST_RADIUS_M, B04_POST_HEIGHT_M = 0.012, 0.160
# The bimanual workbench robot base is on interaction-local -X.  Shift the
# complete Hanoi assembly and its reference card toward that operator position.
B04_OPERATOR_LOCAL_X_OFFSET_M = -0.20
B04_STAND_LOCAL_XY = (0.10, 0.07)
# The workbench robot faces interaction-local +X, so local -Y is the
# operator's right. These explicit offsets keep all rings grouped on that side
# while preserving clearance from both the stand base and one another.
B04_RING_INITIAL_RELATIVE_TO_STAND_LOCAL_XY = (
    (0.08, -0.24),
    (0.20, -0.12),
    (0.28, -0.28),
)
B04_RING_INITIAL_CENTER_Z_M = 0.014
B04_TARGET_CENTER_HEIGHTS_M = tuple(B04_BASE_HEIGHT_M + (i + 0.5) * B04_RING_THICKNESS_M for i in range(3))
# Relative to the post, put the reference sheet at the operator's left-front:
# local -X is toward the operator and local +Y is their left side.
B04_REFERENCE_RELATIVE_TO_STAND_LOCAL_XY = (-0.03, 0.24)
B04_REFERENCE_LOCAL_YAW_RAD = -math.pi / 2.0
B04_REFERENCE_PAPER_HALF_SIZE_XY_M = (0.115, 0.090)
B04_REFERENCE_PAPER_HALF_HEIGHT_M = 0.001
B04_XY_TOLERANCE_M, B04_HEIGHT_TOLERANCE_M = 0.028, 0.018
B04_UPRIGHT_TOLERANCE_RAD = math.radians(22.0)
B04_STABLE_DWELL_STEPS = 20
B04_MAX_OBJECT_SPEED_MPS, B04_MAX_CONTACT_FORCE_N = 1.5, 60.0
B04_WORKSPACE_LOCAL_MIN_XYZ = (-0.60, -0.62, -0.10)
B04_WORKSPACE_LOCAL_MAX_XYZ = (0.70, 0.62, 0.90)
B04_REMOVED_CLUTTER = frozenset({"workbench_box", "workbench_pliers", "stapler", "glasses", "mouse", "phone", "usb"})
B04_PHYSICAL_MATERIAL = PhysicalMaterialSpec(static_friction=1.05, dynamic_friction=0.90, restitution=0.0)


def _operator_shifted_local_xy(xy: tuple[float, float]) -> tuple[float, float]:
    """Move a task item toward the workbench operator in site-local space."""

    return (xy[0] + B04_OPERATOR_LOCAL_X_OFFSET_M, xy[1])


def _ring_initial_local_xy(
    relative_to_stand_xy: tuple[float, float],
) -> tuple[float, float]:
    """Return a ring's fixed site-local position from its post-relative pose."""

    stand_xy = _operator_shifted_local_xy(B04_STAND_LOCAL_XY)
    return (
        stand_xy[0] + relative_to_stand_xy[0],
        stand_xy[1] + relative_to_stand_xy[1],
    )


def _reference_local_xy() -> tuple[float, float]:
    """Return the reference sheet's fixed post-relative site-local position."""

    stand_xy = _operator_shifted_local_xy(B04_STAND_LOCAL_XY)
    return (
        stand_xy[0] + B04_REFERENCE_RELATIVE_TO_STAND_LOCAL_XY[0],
        stand_xy[1] + B04_REFERENCE_RELATIVE_TO_STAND_LOCAL_XY[1],
    )


def _ring_collision_boxes() -> tuple[CollisionBoxSpec, ...]:
    # Collision primitives live in the unit OBJ frame and are scaled together
    # with the visual mesh by LayoutBuilder.
    outer = 1.0
    inner = B04_RING_INNER_RADIUS_RATIO
    center, radial = 0.5 * (outer + inner), 0.5 * (outer - inner)
    tangent = outer * math.sin(math.pi / B04_COLLISION_SEGMENTS) * 1.04
    return tuple(
        CollisionBoxSpec(
            half_size_xyz=(radial, tangent, 0.5),
            pose=PoseSpec(
                position_xyz=(center * math.cos(a), center * math.sin(a), 0.0),
                quaternion_wxyz=(math.cos(0.5 * a), 0.0, 0.0, math.sin(0.5 * a)),
            ),
        )
        for a in (2.0 * math.pi * i / B04_COLLISION_SEGMENTS for i in range(B04_COLLISION_SEGMENTS))
    )


B04_RING_COLLISION_BOXES = tuple(
    _ring_collision_boxes() for _ in B04_RING_OUTER_RADII_M
)
B04_STAND_COLLISION_CYLINDERS = (
    CollisionCylinderSpec(B04_BASE_RADIUS_M, 0.5 * B04_BASE_HEIGHT_M, PoseSpec(position_xyz=(0, 0, 0.5 * B04_BASE_HEIGHT_M))),
    CollisionCylinderSpec(B04_POST_RADIUS_M, 0.5 * B04_POST_HEIGHT_M, PoseSpec(position_xyz=(0, 0, B04_BASE_HEIGHT_M + 0.5 * B04_POST_HEIGHT_M))),
)
B04_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(parameters=(
        GeneralizationParameterSpec("tower_x_offset_m", -0.025, 0.025),
        GeneralizationParameterSpec("tower_y_offset_m", -0.025, 0.025),
        GeneralizationParameterSpec("assembly_yaw_offset_rad", -math.radians(8), math.radians(8)),
    ), requires_rebuild=False),
    physical=AxisRandomizationSpec(parameters=(
        GeneralizationParameterSpec("ring_mass_log_scale", math.log(0.8), math.log(1.2)),
        GeneralizationParameterSpec("ring_friction_log_scale", math.log(0.85), math.log(1.2)),
    ), requires_rebuild=True),
)


@dataclass(frozen=True)
class B04SuccessSpec:
    stable_steps: int = B04_STABLE_DWELL_STEPS


@register_env(B04_SETTINGS.env_id, max_episode_steps=B04_SETTINGS.max_steps)
class B04Env(StudioTaskEnv):
    """Place three true-holed discs on a fixed post, largest to smallest."""

    TASK_SETTINGS = B04_SETTINGS
    GENERALIZATION_SPEC = B04_GENERALIZATION_SPEC
    SUCCESS_SPEC = B04SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B04_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(position_world_xyz=(-0.286827, -0.527697, 1.116254), quaternion_world_wxyz=(0.678772, -0.203379, 0.204404, 0.675370)),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(position_world_xyz=(-1.382406, 0.367292, 1.116254), quaternion_world_wxyz=(0.015449, -0.288309, 0.004653, 0.957401)),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(position_world_xyz=(1.577207, -1.211649, 1.116254), quaternion_world_wxyz=(0.686129, 0.201128, 0.206619, -0.667894)),
    }

    def _task_layout_spec(self) -> Any:
        retained = tuple(item for item in self.current_layout_spec.build_items if getattr(item, "name", None) not in B04_REMOVED_CLUTTER)
        site = self.interaction_site_local_pose_to_world(PoseSpec(position_xyz=(0, 0, 0)))
        stand = StaticMeshSpec(
            name=B04_STAND_NAME, mesh_path=B04_STAND_MESH,
            pose=site.transform_pose(
                PoseSpec(position_xyz=(*_operator_shifted_local_xy(B04_STAND_LOCAL_XY), 0.0))
            ),
            collision_type="compound_primitives", collision_cylinders=B04_STAND_COLLISION_CYLINDERS,
            body_type="kinematic", physical_material=B04_PHYSICAL_MATERIAL,
        )
        rings = tuple(
            StaticMeshSpec(
                name=name, mesh_path=B04_RING_MESH,
                pose=site.transform_pose(
                    PoseSpec(
                        position_xyz=(
                            *_ring_initial_local_xy(relative_xy),
                            B04_RING_INITIAL_CENTER_Z_M,
                        )
                    )
                ),
                scale_xyz=(radius, radius, B04_RING_THICKNESS_M),
                collision_type="compound_boxes", collision_boxes=boxes,
                body_type="dynamic", physical_material=B04_PHYSICAL_MATERIAL,
            )
            for name, radius, relative_xy, boxes in zip(
                B04_RING_NAMES,
                B04_RING_OUTER_RADII_M,
                B04_RING_INITIAL_RELATIVE_TO_STAND_LOCAL_XY,
                B04_RING_COLLISION_BOXES,
                strict=True,
            )
        )
        return replace(self.current_layout_spec, build_items=(*retained, stand, *rings))

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        self._configure_ring_visuals()
        self.reference_card = self._build_reference_card()

    def _bind_task_entities(self) -> None:
        self.stand = self.layout_scene.require_entity(B04_STAND_NAME)
        self.rings = tuple(self.layout_scene.require_entity(name) for name in B04_RING_NAMES)
        self.ring_bottom, self.ring_middle, self.ring_top = self.rings

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(("ring_mass_log_scale", "ring_friction_log_scale"), cache_name="_b04_physical_values")
        actor_values = ActorBuildValues(
            physical_materials=tuple(scaled_physical_material(B04_PHYSICAL_MATERIAL, friction_scale=float(v), restitution=0.0) for v in positive_scale(values["ring_friction_log_scale"])),
            mass_scales=tuple(float(v) for v in positive_scale(values["ring_mass_log_scale"])),
        )
        return {name: actor_values for name in B04_RING_NAMES}, None

    def _configure_ring_visuals(self) -> None:
        if sapien is None or not self.scene.can_render():
            return
        for color, actor in zip(B04_RING_COLORS_RGBA, self.rings, strict=True):
            for raw in tuple(getattr(actor, "_objs", ())):
                body = raw.find_component_by_type(sapien.render.RenderBodyComponent)
                if body is not None:
                    for shape in body.render_shapes:
                        for part in shape.parts:
                            part.material.set_base_color(list(color))
                            part.material.set_base_color_texture(None)
                            part.material.roughness = 0.78
                            part.material.metallic = 0.0

    def _build_reference_card(self) -> Any:
        if sapien is None:
            raise ImportError("SAPIEN is required to build the B04 reference card.")
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[*B04_REFERENCE_PAPER_HALF_SIZE_XY_M, B04_REFERENCE_PAPER_HALF_HEIGHT_M],
            material=sapien.render.RenderMaterial(base_color=[0.97, 0.96, 0.91, 1.0], roughness=0.95),
        )
        lower_y, ink_z = -0.045, 0.0025
        stand_ink = sapien.render.RenderMaterial(
            base_color=[0.18, 0.18, 0.20, 1.0], roughness=0.9
        )
        builder.add_box_visual(
            half_size=[0.085, 0.004, 0.0007],
            pose=sapien.Pose(p=[0.0, lower_y - 0.004, ink_z]),
            material=stand_ink,
        )
        builder.add_box_visual(
            half_size=[0.004, 0.050, 0.0007],
            pose=sapien.Pose(p=[0.0, lower_y + 0.046, ink_z + 0.0015]),
            material=stand_ink,
        )
        for radius, color in zip(B04_RING_OUTER_RADII_M, B04_RING_COLORS_RGBA, strict=True):
            builder.add_box_visual(
                half_size=[radius, B04_RING_HALF_THICKNESS_M, 0.0007],
                pose=sapien.Pose(p=[0.0, lower_y + B04_RING_HALF_THICKNESS_M, ink_z]),
                material=sapien.render.RenderMaterial(base_color=list(color), roughness=0.9),
            )
            lower_y += B04_RING_THICKNESS_M
        builder.initial_pose = sapien.Pose(p=[0, 0, -20])
        return builder.build_kinematic(name="b04_hanoi_reference_card")

    def _spatial_offsets(self, env_idx: Any) -> tuple[Any, Any, Any]:
        batch = len(env_idx)
        x = torch.zeros(batch, device=self.device)
        y = torch.zeros(batch, device=self.device)
        yaw = torch.zeros(batch, device=self.device)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(("tower_x_offset_m", "tower_y_offset_m", "assembly_yaw_offset_rad"), cache_name="_b04_spatial_values")
            ids = env_idx.detach().cpu().numpy()
            x = torch.as_tensor(values["tower_x_offset_m"][ids], device=self.device)
            y = torch.as_tensor(values["tower_y_offset_m"][ids], device=self.device)
            yaw = torch.as_tensor(values["assembly_yaw_offset_rad"][ids], device=self.device)
        return x.float(), y.float(), yaw.float()

    def _rotated_local_pose(self, xy: tuple[float, float], z: float, x: Any, y: Any, yaw: Any) -> Any:
        c, s = torch.cos(yaw), torch.sin(yaw)
        p = torch.stack((c * xy[0] - s * xy[1] + x, s * xy[0] + c * xy[1] + y, torch.full_like(x, z)), dim=1)
        q = torch.stack((torch.cos(0.5 * yaw), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(0.5 * yaw)), dim=1)
        return self._site_pose(len(x)) * Pose.create_from_pq(p=p, q=q)

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        batch = len(env_idx)
        self.layout_scene.initialize_episode(env_idx=env_idx, batched_episode_rng=self._batched_episode_rng, device=self.device)
        x, y, yaw = self._spatial_offsets(env_idx)
        c, s = torch.cos(yaw), torch.sin(yaw)
        stand_xy = _operator_shifted_local_xy(B04_STAND_LOCAL_XY)
        self._target_local_xy = torch.stack((c * stand_xy[0] - s * stand_xy[1] + x, s * stand_xy[0] + c * stand_xy[1] + y), dim=1)
        self.stand.set_pose(self._rotated_local_pose(stand_xy, 0.0, x, y, yaw))
        for ring, relative_xy in zip(
            self.rings, B04_RING_INITIAL_RELATIVE_TO_STAND_LOCAL_XY, strict=True
        ):
            ring.set_pose(
                self._rotated_local_pose(
                    _ring_initial_local_xy(relative_xy),
                    B04_RING_INITIAL_CENTER_Z_M,
                    x,
                    y,
                    yaw,
                )
            )
            self._zero_actor_velocity(ring, batch)
        reference_pose = self._rotated_local_pose(
            _reference_local_xy(), 0.002, x, y, yaw
        )
        reference_rotation = Pose.create_from_pq(
            p=torch.zeros((batch, 3), device=self.device),
            q=torch.tensor(
                (
                    math.cos(0.5 * B04_REFERENCE_LOCAL_YAW_RAD),
                    0.0,
                    0.0,
                    math.sin(0.5 * B04_REFERENCE_LOCAL_YAW_RAD),
                ),
                device=self.device,
                dtype=torch.float32,
            ).repeat(batch, 1),
        )
        self.reference_card.set_pose(reference_pose * reference_rotation)
        self._hide_robots()
        self.layout_scene.settle(env_idx=env_idx)
        self._reset_robots(batch)
        self._stable_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

    def _ring_diagnostics(self) -> tuple[Any, Any, Any]:
        site_inv = self._site_pose(self.num_envs).inv()
        local = tuple(site_inv * ring.pose for ring in self.rings)
        flags, stable = [], []
        for index, (ring, pose) in enumerate(zip(self.rings, local, strict=True)):
            xy_error = torch.linalg.norm(pose.p[:, :2] - self._target_local_xy, dim=1)
            up_z = 1.0 - 2.0 * (pose.q[:, 1].square() + pose.q[:, 2].square())
            tilt = torch.acos(torch.clamp(torch.abs(up_z), 0.0, 1.0))
            z_error = torch.abs(pose.p[:, 2] - B04_TARGET_CENTER_HEIGHTS_M[index])
            flags.append((xy_error <= B04_XY_TOLERANCE_M) & (z_error <= B04_HEIGHT_TOLERANCE_M) & (tilt <= B04_UPRIGHT_TOLERANCE_RAD))
            stable.append(self._actor_static(ring, 0.06, 0.8))
        return torch.stack(flags, dim=1), torch.stack(stable, dim=1), torch.stack([p.p for p in local], dim=1)

    def evaluate(self) -> dict[str, Any]:
        placed, stable, _ = self._ring_diagnostics()
        bottom = placed[:, 0]
        middle = bottom & placed[:, 1]
        top = middle & placed[:, 2]
        final_now = top & stable.all(dim=1)
        self._stable_steps = torch.where(final_now, self._stable_steps + 1, torch.zeros_like(self._stable_steps))
        success = self._latch_task_success(
            self._stable_steps >= self.SUCCESS_SPEC.stable_steps
        )
        return {
            "success": success,
            "bottom_ring_on_post": bottom, "middle_ring_on_post": middle,
            "tower_complete_and_stable": success,
            "ring_pose_ok": placed, "ring_stable": stable, "stable_steps": self._stable_steps,
        }

    def _check_generalization_validity(self, axis: Any) -> Any:
        del axis
        finite = torch.stack([ring.pose.raw_pose.isfinite().all(dim=1) for ring in self.rings], dim=1).all(dim=1)
        heights = torch.stack([self._site_local_positions(ring)[:, 2] for ring in self.rings], dim=1)
        return finite & (heights > -0.12).all(dim=1)
