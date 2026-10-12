from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from dex_benchmark.assets import evodex_asset_dir
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
    LayoutBuilder,
    LayoutRegistry,
    PhysicalMaterialSpec,
    PoseSpec,
    StaticMeshSpec,
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


B03_SETTINGS = TaskRegistry.register(TaskRegistry.load("b03_v1"))
B03_INTERACTION_SITE_ID = "counter_top"
B03_LAYOUT_IDS = (
    LayoutId.KITCHEN_L1,
    LayoutId.KITCHEN_L2,
    LayoutId.KITCHEN_L3,
)
# Stable first-person viewpoint centered between the two hands. Camera local +X
# points toward the work area, with positive local pitch looking downward.
B03_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.16, 0.0, 0.55),
    quaternion_wxyz=(
        0.9305989566,
        0.0,
        0.3660406590,
        0.0,
    ),
)
B03_ASSET_DIR = evodex_asset_dir() / "task_assets" / "b03"

B03_FOOD_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=0.95,
    dynamic_friction=0.82,
    restitution=0.0,
)


@dataclass(frozen=True)
class B03IngredientSpec:
    """One sandwich layer and its simulation geometry semantics."""

    name: str
    asset_file: str
    size_xyz_m: tuple[float, float, float]
    mesh_scale_xyz: tuple[float, float, float]
    nominal_mass_kg: float
    spawn_local_xy_m: tuple[float, float]
    spawn_yaw_rad: float
    footprint_kind: str
    visual_color_rgba: tuple[float, float, float, float]

    @property
    def half_height_m(self) -> float:
        return 0.5 * self.size_xyz_m[2]


B03_INGREDIENT_SPECS = (
    B03IngredientSpec(
        name="b03_bread_bottom",
        asset_file="bread_bottom.obj",
        size_xyz_m=(0.100, 0.100, 0.024),
        mesh_scale_xyz=(1.0, 1.0, 4.0 / 3.0),
        nominal_mass_kg=0.040,
        spawn_local_xy_m=(-0.18, 0.14),
        spawn_yaw_rad=math.radians(18.0),
        footprint_kind="box",
        visual_color_rgba=(0.82, 0.55, 0.25, 1.0),
    ),
    B03IngredientSpec(
        name="b03_cheese_slice",
        asset_file="cheese_slice.obj",
        size_xyz_m=(0.085, 0.085, 0.006),
        mesh_scale_xyz=(1.0, 1.0, 1.0),
        nominal_mass_kg=0.015,
        spawn_local_xy_m=(-0.17, -0.14),
        spawn_yaw_rad=math.radians(-28.0),
        footprint_kind="box",
        visual_color_rgba=(0.96, 0.72, 0.12, 1.0),
    ),
    B03IngredientSpec(
        name="b03_patty",
        asset_file="patty.obj",
        size_xyz_m=(0.075, 0.075, 0.018),
        mesh_scale_xyz=(1.0, 1.0, 9.0 / 7.0),
        nominal_mass_kg=0.045,
        spawn_local_xy_m=(0.02, 0.12),
        spawn_yaw_rad=math.radians(42.0),
        footprint_kind="circle",
        visual_color_rgba=(0.32, 0.12, 0.05, 1.0),
    ),
    B03IngredientSpec(
        name="b03_bread_top",
        asset_file="bread_top.obj",
        size_xyz_m=(0.080, 0.0784, 0.018),
        mesh_scale_xyz=(0.8, 0.8, 1.125),
        nominal_mass_kg=0.035,
        # Place it in the left-hand workspace, visually above the patty.
        spawn_local_xy_m=(0.15, 0.17),
        spawn_yaw_rad=math.radians(22.0),
        footprint_kind="box",
        visual_color_rgba=(0.82, 0.55, 0.25, 1.0),
    ),
)
B03_OBJECT_NAMES = tuple(spec.name for spec in B03_INGREDIENT_SPECS)
B03_ROLE_NAMES = ("bottom_bread", "cheese", "patty", "top_bread")

# The target center is in counter_top local coordinates. All target heights are
# derived from the authored OBJ thicknesses, so the layer order is explicit.
B03_NOMINAL_TARGET_LOCAL_XY = (0.01, 0.00)
B03_NOMINAL_TARGET_YAW_RAD = 0.0
B03_TARGET_CENTER_HEIGHTS_M: tuple[float, ...] = tuple(
    sum(item.size_xyz_m[2] for item in B03_INGREDIENT_SPECS[:index])
    + B03_INGREDIENT_SPECS[index].half_height_m
    for index in range(len(B03_INGREDIENT_SPECS))
)
B03_PRESTACKED_LAYER_COUNT = 2
B03_LOOSE_ROLE_NAMES = B03_ROLE_NAMES[B03_PRESTACKED_LAYER_COUNT:]

# The tabletop reference card shows a side cross-section of the completed
# sandwich. Its colored bands preserve every layer's actual width and thickness.
B03_REFERENCE_CENTER_OFFSET_LOCAL_XY = (-0.20, 0.0)
B03_REFERENCE_YAW_OFFSET_RAD = -0.5 * math.pi
B03_REFERENCE_PAPER_HALF_SIZE_XY_M = (0.065, 0.038)
B03_REFERENCE_PAPER_HALF_HEIGHT_M = 0.00025
B03_REFERENCE_INK_HALF_HEIGHT_M = 0.00008
B03_REFERENCE_INK_GAP_M = 0.00002
B03_PEDESTAL_HALF_SIZE_XYZ_M = (0.020, 0.020, 0.004)
B03_PEDESTAL_COLOR_RGBA = (0.52, 0.58, 0.60, 1.0)


@dataclass(frozen=True)
class B03SuccessSpec:
    target_xy_tolerance_m: float = 0.050
    layer_height_tolerance_m: float = 0.018
    maximum_tilt_rad: float = math.radians(25.0)
    minimum_overlap_ratio: float = 0.60
    support_contact_force_n: float = 0.02
    hand_contact_force_n: float = 0.25
    maximum_linear_speed_mps: float = 0.080
    maximum_angular_speed_rad_s: float = 1.20
    completion_dwell_s: float = 0.25


B03_SUCCESS_SPEC = B03SuccessSpec()
B03_MASK_SIZE = 96
B03_MASK_EXTENT_M = 0.18
B03_WORKSPACE_HALF_EXTENT_XY_M = (0.43, 0.43)


# Nearby movable/fixed kitchen items are not part of the sandwich task. If a
# layout exposes them, disturbing them counts as protected-object disturbance.

B03_SPATIAL_PARAMETER_NAMES = tuple(
    name
    for role in B03_LOOSE_ROLE_NAMES
    for name in (
        f"{role}_x_offset_m",
        f"{role}_y_offset_m",
        f"{role}_yaw_offset_rad",
    )
) + (
    "target_x_offset_m",
    "target_y_offset_m",
    "target_yaw_offset_rad",
)

B03_PHYSICAL_PARAMETER_NAMES = tuple(
    name
    for role in B03_LOOSE_ROLE_NAMES
    for name in (
        f"{role}_mass_log_scale",
        f"{role}_com_x_offset_m",
        f"{role}_com_y_offset_m",
    )
) + ("food_friction_log_scale",)

B03_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            *tuple(
                GeneralizationParameterSpec(f"{role}_{axis}_offset_m", -0.020, 0.020)
                for role in B03_LOOSE_ROLE_NAMES
                for axis in ("x", "y")
            ),
            *tuple(
                GeneralizationParameterSpec(
                    f"{role}_yaw_offset_rad",
                    -math.radians(15.0),
                    math.radians(15.0),
                )
                for role in B03_LOOSE_ROLE_NAMES
            ),
            GeneralizationParameterSpec("target_x_offset_m", -0.030, 0.030),
            GeneralizationParameterSpec("target_y_offset_m", -0.030, 0.030),
            GeneralizationParameterSpec(
                "target_yaw_offset_rad",
                -math.radians(15.0),
                math.radians(15.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            *tuple(
                GeneralizationParameterSpec(
                    f"{role}_mass_log_scale", math.log(0.80), math.log(1.20)
                )
                for role in B03_LOOSE_ROLE_NAMES
            ),
            GeneralizationParameterSpec(
                "food_friction_log_scale", math.log(0.80), math.log(1.20)
            ),
            *tuple(
                GeneralizationParameterSpec(f"{role}_{axis}_offset_m", -0.003, 0.003)
                for role in B03_LOOSE_ROLE_NAMES
                for axis in ("com_x", "com_y")
            ),
        ),
        requires_rebuild=True,
    ),
)


def _b03_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B03_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            B03_INTERACTION_SITE_ID
        )
        robot_base_center_pose = (
            interaction_site.bimanual_robot_base_center_pose
            or interaction_site.robot_base_center_pose
        )
        camera_world_pose = robot_base_center_pose.transform_pose(
            B03_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
        )
        camera_quaternion = np.asarray(
            camera_world_pose.quaternion_wxyz,
            dtype=np.float64,
        )
        camera_quaternion /= np.linalg.norm(camera_quaternion)
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=tuple(float(value) for value in camera_quaternion),
        )
    return camera_poses


@register_env(B03_SETTINGS.env_id, max_episode_steps=B03_SETTINGS.max_steps)
class B03Env(DexVLABenchBaseEnv):
    """Build a four-layer sandwich in the specified order on counter_top."""

    TASK_SETTINGS = B03_SETTINGS
    SUCCESS_SPEC = B03_SUCCESS_SPEC
    GENERALIZATION_SPEC = B03_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B03_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _b03_human_viewer_camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Keep the initial Human Viewer on the stable first-person camera."""

        return None

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        # counter_top already owns a calibrated bimanual center pose. BaseEnv
        # expands it into left/right bases using the site's bimanual spacing.
        super()._load_agent(options, self.interaction_site_robot_base_sapien_poses())

    def _dual_agents(self) -> tuple[Any, Any]:
        return self.left_agent, self.right_agent

    def _validate_assets(self) -> None:
        missing = [
            str(B03_ASSET_DIR / spec.asset_file)
            for spec in B03_INGREDIENT_SPECS
            if not (B03_ASSET_DIR / spec.asset_file).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "B03 sandwich OBJ assets are missing: " + ", ".join(missing)
            )

    def _site_world_pose_spec(
        self,
        local_xyz: tuple[float, float, float],
        yaw_rad: float = 0.0,
    ) -> PoseSpec:
        half_yaw = 0.5 * yaw_rad
        return self.interaction_site_local_pose_to_world(
            PoseSpec(
                position_xyz=local_xyz,
                quaternion_wxyz=(
                    math.cos(half_yaw),
                    0.0,
                    0.0,
                    math.sin(half_yaw),
                ),
            )
        )

    def _task_layout_spec(self) -> Any:
        self._validate_assets()
        task_items = tuple(
            StaticMeshSpec(
                name=spec.name,
                mesh_path=B03_ASSET_DIR / spec.asset_file,
                pose=self._site_world_pose_spec(
                    (
                        spec.spawn_local_xy_m[0],
                        spec.spawn_local_xy_m[1],
                        spec.half_height_m,
                    ),
                    spec.spawn_yaw_rad,
                ),
                scale_xyz=spec.mesh_scale_xyz,
                collision_type="convex",
                body_type=(
                    "kinematic"
                    if layer_index < B03_PRESTACKED_LAYER_COUNT
                    else "dynamic"
                ),
                physical_material=B03_FOOD_PHYSICAL_MATERIAL,
            )
            for layer_index, spec in enumerate(B03_INGREDIENT_SPECS)
        )
        return replace(
            self.current_layout_spec,
            build_items=(*self.current_layout_spec.build_items, *task_items),
        )

    def _physical_actor_values(self) -> dict[str, ActorBuildValues] | None:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None
        values = self._generalization_values(
            B03_PHYSICAL_PARAMETER_NAMES,
            cache_name="_b03_physical_values",
        )
        friction_scales = positive_scale(values["food_friction_log_scale"])
        materials = tuple(
            scaled_physical_material(
                B03_FOOD_PHYSICAL_MATERIAL,
                friction_scale=float(scale),
                restitution=0.0,
            )
            for scale in friction_scales
        )
        return {
            spec.name: ActorBuildValues(physical_materials=materials)
            for spec in B03_INGREDIENT_SPECS
        }

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        # Match the existing bimanual pose-structure task loading pattern: keep
        # robots out of LayoutBuilder while the task-private dynamic actors build.
        multi_agent = self.agent
        try:
            self.agent = None
            self.layout_scene = LayoutBuilder(
                env=self,
                layout_spec=self._task_layout_spec(),
                actor_values=self._physical_actor_values(),
            ).build()
        finally:
            self.agent = multi_agent

        self.ingredients = tuple(
            self.layout_scene.require_entity(name) for name in B03_OBJECT_NAMES
        )
        (
            self.bread_bottom,
            self.cheese_slice,
            self.patty,
            self.bread_top,
        ) = self.ingredients
        self._configure_food_visuals()
        self.reference_card = self._build_reference_card()
        self.ingredient_pedestals = self._build_ingredient_pedestals()

        self._configure_food_mass_and_com()
        self._configure_contact_queries()
        self._completion_dwell_steps = max(
            1,
            math.ceil(self.SUCCESS_SPEC.completion_dwell_s * self._control_freq),
        )
        self._build_overlap_grid()

    def _configure_food_visuals(self) -> None:
        """Apply stable semantic colors to all replicated ingredient visuals."""

        if sapien is None:
            raise ImportError("SAPIEN is required to color B03 ingredients.")
        if not self.scene.can_render():
            return
        for spec, actor in zip(B03_INGREDIENT_SPECS, self.ingredients, strict=True):
            colored_part_count = 0
            for raw_entity in tuple(getattr(actor, "_objs", ())):
                render_body = raw_entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                if render_body is None:
                    continue
                for render_shape in render_body.render_shapes:
                    for part in render_shape.parts:
                        part.material.set_base_color(list(spec.visual_color_rgba))
                        part.material.set_base_color_texture(None)
                        part.material.roughness = 0.82
                        part.material.metallic = 0.0
                        colored_part_count += 1
            if colored_part_count == 0:
                raise RuntimeError(
                    f"B03 ingredient {spec.name!r} has no render material to color."
                )

    def _build_reference_card(self) -> Any:
        """Build a flat, non-colliding side view of the finished sandwich."""

        if sapien is None:
            raise ImportError("SAPIEN is required to build the B03 reference card.")
        builder = self.scene.create_actor_builder()
        paper_material = sapien.render.RenderMaterial(
            base_color=[0.96, 0.95, 0.90, 1.0],
            roughness=0.95,
        )
        builder.add_box_visual(
            half_size=[
                B03_REFERENCE_PAPER_HALF_SIZE_XY_M[0],
                B03_REFERENCE_PAPER_HALF_SIZE_XY_M[1],
                B03_REFERENCE_PAPER_HALF_HEIGHT_M,
            ],
            material=paper_material,
        )

        total_thickness_m = sum(spec.size_xyz_m[2] for spec in B03_INGREDIENT_SPECS)
        layer_lower_y_m = -0.5 * total_thickness_m
        ink_center_z_m = (
            B03_REFERENCE_PAPER_HALF_HEIGHT_M
            + B03_REFERENCE_INK_GAP_M
            + B03_REFERENCE_INK_HALF_HEIGHT_M
        )
        for spec in B03_INGREDIENT_SPECS:
            layer_half_thickness_m = spec.half_height_m
            layer_center_y_m = layer_lower_y_m + layer_half_thickness_m
            layer_material = sapien.render.RenderMaterial(
                base_color=list(spec.visual_color_rgba),
                roughness=0.90,
            )
            builder.add_box_visual(
                half_size=[
                    0.5 * spec.size_xyz_m[0],
                    layer_half_thickness_m,
                    B03_REFERENCE_INK_HALF_HEIGHT_M,
                ],
                pose=sapien.Pose(p=[0.0, layer_center_y_m, ink_center_z_m]),
                material=layer_material,
            )
            layer_lower_y_m += spec.size_xyz_m[2]

        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_kinematic(name="b03_sandwich_reference_card")

    def _build_ingredient_pedestals(self) -> tuple[Any, ...]:
        """Build one low, narrow support under each initially loose layer."""

        if sapien is None:
            raise ImportError("SAPIEN is required to build B03 ingredient pedestals.")
        pedestal_material = sapien.render.RenderMaterial(
            base_color=list(B03_PEDESTAL_COLOR_RGBA),
            roughness=0.90,
        )
        pedestals: list[Any] = []
        for spec in B03_INGREDIENT_SPECS[B03_PRESTACKED_LAYER_COUNT:]:
            builder = self.scene.create_actor_builder()
            builder.add_box_collision(half_size=list(B03_PEDESTAL_HALF_SIZE_XYZ_M))
            builder.add_box_visual(
                half_size=list(B03_PEDESTAL_HALF_SIZE_XYZ_M),
                material=pedestal_material,
            )
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
            pedestals.append(
                builder.build_kinematic(name=f"{spec.name}_grasp_pedestal")
            )
        return tuple(pedestals)

    def _configure_food_mass_and_com(self) -> None:
        # StaticMeshSpec does not expose an absolute mass field. Assign the
        # authored task masses after build and preserve the builder-computed
        # inertia ratios. Physical-axis mass and COM variation is per env.
        if self.generalization_axis is GeneralizationAxis.PHYSICAL:
            values = self._generalization_values(
                B03_PHYSICAL_PARAMETER_NAMES,
                cache_name="_b03_physical_values",
            )
        else:
            values = None

        for role, spec, actor in zip(
            B03_LOOSE_ROLE_NAMES,
            B03_INGREDIENT_SPECS[B03_PRESTACKED_LAYER_COUNT:],
            self.ingredients[B03_PRESTACKED_LAYER_COUNT:],
            strict=True,
        ):
            if values is None:
                mass_scales = np.ones(self.num_envs, dtype=np.float64)
                com_x = np.zeros(self.num_envs, dtype=np.float64)
                com_y = np.zeros(self.num_envs, dtype=np.float64)
            else:
                mass_scales = positive_scale(values[f"{role}_mass_log_scale"])
                com_x = values[f"{role}_com_x_offset_m"]
                com_y = values[f"{role}_com_y_offset_m"]

            raw_bodies = tuple(getattr(actor, "_bodies", ()))
            if len(raw_bodies) != self.num_envs:
                raise RuntimeError(
                    f"B03 actor {spec.name!r} expected {self.num_envs} raw bodies, "
                    f"got {len(raw_bodies)}."
                )
            for env_index, raw_body in enumerate(raw_bodies):
                if raw_body is None:
                    continue
                current_mass = float(raw_body.mass)
                target_mass = spec.nominal_mass_kg * float(mass_scales[env_index])
                if current_mass > 0.0 and np.isfinite(current_mass):
                    raw_body.inertia = np.asarray(raw_body.inertia) * (
                        target_mass / current_mass
                    )
                raw_body.mass = target_mass

                dx = float(com_x[env_index])
                dy = float(com_y[env_index])
                if abs(dx) + abs(dy) <= 0.0:
                    continue
                # SAPIEN 3 exposes cmass_local_pose on rigid dynamic bodies. The
                # guard keeps the task usable on compatible builds that omit it.
                if hasattr(raw_body, "cmass_local_pose"):
                    old_pose = raw_body.cmass_local_pose
                    old_p = np.asarray(old_pose.p, dtype=np.float64)
                    raw_body.cmass_local_pose = sapien.Pose(
                        p=[old_p[0] + dx, old_p[1] + dy, old_p[2]],
                        q=old_pose.q,
                    )

    def _configure_contact_queries(self) -> None:
        self._target_groups = tuple(
            (name, (actor,))
            for name, actor in zip(B03_OBJECT_NAMES, self.ingredients, strict=True)
        )
        if not self.gpu_sim_enabled:
            for agent in self._dual_agents():
                agent.prepare_cpu_object_taxel_query(self._target_groups)

        # Adjacent food-food contacts verify that each upper layer is physically
        # supported rather than merely passing through the desired Z band.
        self._support_queries: list[PairwiseContactQueryHandle] = []
        for lower, upper in zip(
            self.ingredients[:-1], self.ingredients[1:], strict=True
        ):
            self._support_queries.append(
                self.left_agent.prepare_pairwise_contact_query((lower,), (upper,))
            )

        self._inter_arm_contact_query = self.left_agent.prepare_pairwise_contact_query(
            tuple(self.left_agent.robot.links),
            tuple(self.right_agent.robot.links),
        )

    def _build_overlap_grid(self) -> None:
        pixel_size = B03_MASK_EXTENT_M / B03_MASK_SIZE
        centers = (
            torch.arange(
                B03_MASK_SIZE,
                device=self.device,
                dtype=torch.float32,
            )
            + 0.5
        ) * pixel_size - 0.5 * B03_MASK_EXTENT_M
        grid_y, grid_x = torch.meshgrid(centers, centers, indexing="ij")
        self._mask_grid_xy = torch.stack((grid_x, grid_y), dim=-1)

    def _site_pose(self, batch_size: int) -> Any:
        return Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

    def _local_planar_pose(self, local_xy: Any, local_yaw: Any, z: float) -> Any:
        half_yaw = 0.5 * local_yaw
        quaternion = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=1,
        )
        position = torch.cat(
            (
                local_xy,
                torch.full((len(local_xy), 1), z, device=self.device),
            ),
            dim=1,
        )
        return self._site_pose(len(local_xy)) * Pose.create_from_pq(
            p=position, q=quaternion
        )

    def _spatial_values(self) -> dict[str, np.ndarray] | None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return None
        return self._generalization_values(
            B03_SPATIAL_PARAMETER_NAMES,
            cache_name="_b03_spatial_values",
        )

    def _target_geometry(self, env_idx: Any) -> tuple[Any, Any]:
        batch_size = len(env_idx)
        xy = np.repeat(
            np.asarray(B03_NOMINAL_TARGET_LOCAL_XY, dtype=np.float64)[None, :],
            batch_size,
            axis=0,
        )
        yaw = np.full(batch_size, B03_NOMINAL_TARGET_YAW_RAD, dtype=np.float64)
        values = self._spatial_values()
        if values is not None:
            indices = env_idx.detach().cpu().numpy()
            xy[:, 0] += values["target_x_offset_m"][indices]
            xy[:, 1] += values["target_y_offset_m"][indices]
            yaw += values["target_yaw_offset_rad"][indices]
        return (
            torch.as_tensor(xy, device=self.device, dtype=torch.float32),
            torch.as_tensor(yaw, device=self.device, dtype=torch.float32),
        )

    def _set_initial_food_poses(self, env_idx: Any) -> None:
        batch_size = len(env_idx)
        values = self._spatial_values()
        indices = env_idx.detach().cpu().numpy()
        ingredient_entries = zip(
            B03_ROLE_NAMES, B03_INGREDIENT_SPECS, self.ingredients, strict=True
        )
        for layer_index, (role, spec, actor) in enumerate(ingredient_entries):
            if layer_index < B03_PRESTACKED_LAYER_COUNT:
                # [B, 2] and [B]: the authored lower stack is the task's target.
                local_xy = self._target_local_xy[env_idx]
                local_yaw = self._target_local_yaw[env_idx]
                center_height_m = B03_TARGET_CENTER_HEIGHTS_M[layer_index] + 0.001
            else:
                xy = np.repeat(
                    np.asarray(spec.spawn_local_xy_m, dtype=np.float64)[None, :],
                    batch_size,
                    axis=0,
                )
                yaw = np.full(batch_size, spec.spawn_yaw_rad, dtype=np.float64)
                if values is not None:
                    xy[:, 0] += values[f"{role}_x_offset_m"][indices]
                    xy[:, 1] += values[f"{role}_y_offset_m"][indices]
                    yaw += values[f"{role}_yaw_offset_rad"][indices]
                # [B, 2] and [B]: independently randomized loose-layer poses.
                local_xy = torch.as_tensor(
                    xy,
                    device=self.device,
                    dtype=torch.float32,
                )
                local_yaw = torch.as_tensor(
                    yaw,
                    device=self.device,
                    dtype=torch.float32,
                )
                pedestal = self.ingredient_pedestals[
                    layer_index - B03_PRESTACKED_LAYER_COUNT
                ]
                # [B, 2] + [B] -> [B, 3]: center each 8 mm pedestal under its
                # randomized loose ingredient while leaving wide grasp overhangs.
                pedestal.set_pose(
                    self._local_planar_pose(
                        local_xy,
                        torch.zeros_like(local_yaw),
                        B03_PEDESTAL_HALF_SIZE_XYZ_M[2],
                    )
                )
                center_height_m = (
                    2.0 * B03_PEDESTAL_HALF_SIZE_XYZ_M[2]
                    + spec.half_height_m
                    + 0.001
                )
            actor.set_pose(
                self._local_planar_pose(
                    local_xy,
                    local_yaw,
                    center_height_m,
                )
            )
            zero = torch.zeros((batch_size, 3), device=self.device)
            actor.set_linear_velocity(zero)
            actor.set_angular_velocity(zero)

    def _set_reference_card_pose(self, env_idx: Any) -> None:
        target_yaw = self._target_local_yaw[env_idx]
        card_yaw = target_yaw + B03_REFERENCE_YAW_OFFSET_RAD
        offset = torch.as_tensor(
            B03_REFERENCE_CENTER_OFFSET_LOCAL_XY,
            device=self.device,
            dtype=torch.float32,
        )
        cosine, sine = torch.cos(target_yaw), torch.sin(target_yaw)
        # [2] + two [B] rotations -> [B, 2]: keep the card beside the target.
        rotated_offset = torch.stack(
            (
                cosine * offset[0] - sine * offset[1],
                sine * offset[0] + cosine * offset[1],
            ),
            dim=1,
        )
        card_local_xy = self._target_local_xy[env_idx] + rotated_offset
        self.reference_card.set_pose(
            self._local_planar_pose(
                card_local_xy,
                card_yaw,
                B03_REFERENCE_PAPER_HALF_HEIGHT_M,
            )
        )

    def _reset_agents_to_rest(self, batch_size: int) -> None:
        base_poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(base_poses, list) or len(base_poses) != 2:
            raise RuntimeError("B03 requires exactly two robot base poses.")
        for agent, base_pose in zip(self._dual_agents(), base_poses, strict=True):
            rest = agent.keyframes.get("rest")
            if rest is None or rest.qpos is None:
                raise RuntimeError(
                    f"B03 agent {agent.uid!r} requires the rest keyframe."
                )
            qpos = (
                torch.as_tensor(rest.qpos, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            agent.reset(qpos)
            agent.robot.set_pose(base_pose)

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_target_local_xy"):
            return
        self._target_local_xy = torch.zeros(
            (self.num_envs, 2), device=self.device, dtype=torch.float32
        )
        self._target_local_yaw = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self._food_touched = torch.zeros(
            (self.num_envs, len(self.ingredients)),
            dtype=torch.bool,
            device=self.device,
        )
        self._layer_ever_placed = torch.zeros_like(self._food_touched)
        self._completion_dwell_counter = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._ensure_episode_state()
            target_xy, target_yaw = self._target_geometry(env_idx)
            self._target_local_xy[env_idx] = target_xy
            self._target_local_yaw[env_idx] = target_yaw
            self._set_reference_card_pose(env_idx)
            self._set_initial_food_poses(env_idx)

            # Keep both robots out while the authored two-layer base and two
            # loose upper ingredients settle into physically valid poses.
            for agent in self._dual_agents():
                agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_agents_to_rest(batch_size)

            self._food_touched[env_idx] = False
            self._layer_ever_placed[env_idx] = False
            self._layer_ever_placed[env_idx, :B03_PRESTACKED_LAYER_COUNT] = True
            self._completion_dwell_counter[env_idx] = 0

    def _world_to_site(self, positions: Any) -> Any:
        return (
            self._site_pose(self.num_envs).inv() * Pose.create_from_pq(p=positions)
        ).p

    def _local_yaw_and_tilt(self, actor: Any) -> tuple[Any, Any]:
        site_q = self._site_pose(self.num_envs).q
        inverse_site_q = site_q.clone()
        inverse_site_q[:, 1:] *= -1.0
        relative_q = (
            Pose.create_from_pq(q=inverse_site_q) * Pose.create_from_pq(q=actor.pose.q)
        ).q
        w, x, y, z = relative_q.unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        up_cosine = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
        tilt = torch.acos(up_cosine)
        return yaw, tilt

    def _hand_contact_by_object(self) -> tuple[Any, Any]:
        contact = torch.zeros(
            (self.num_envs, len(self.ingredients)),
            dtype=torch.bool,
            device=self.device,
        )
        maximum = torch.zeros(self.num_envs, device=self.device)
        for agent in self._dual_agents():
            forces = agent.get_object_taxel_world_forces(self._target_groups)
            for index, name in enumerate(B03_OBJECT_NAMES):
                parent_forces = agent.aggregate_taxel_world_forces_by_parent(
                    forces[name]
                )
                norms = torch.linalg.norm(parent_forces, dim=-1)
                contact[:, index] |= (
                    norms >= self.SUCCESS_SPEC.hand_contact_force_n
                ).any(dim=1)
                maximum = torch.maximum(maximum, norms.amax(dim=1))
        return contact, maximum

    def _support_contact_flags(self) -> Any:
        flags: list[Any] = []
        for support_index, query in enumerate(self._support_queries):
            if support_index < B03_PRESTACKED_LAYER_COUNT - 1:
                # Both authored base layers are kinematic, so their fixed
                # assembly guarantees support without a dynamic contact force.
                flags.append(
                    torch.ones(
                        self.num_envs,
                        dtype=torch.bool,
                        device=self.device,
                    )
                )
                continue
            forces = self.left_agent.get_pairwise_contact_forces(query)
            # [B, 1, 1, 3] -> [B].
            flags.append(
                torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))
                >= self.SUCCESS_SPEC.support_contact_force_n
            )
        return torch.stack(flags, dim=1)

    def _footprint_mask(
        self,
        center_xy: Any,
        yaw: Any,
        spec: B03IngredientSpec,
    ) -> Any:
        # Shift the fixed grid to each environment's randomized sandwich target.
        world_grid = (
            self._mask_grid_xy[None, :, :, :] + self._target_local_xy[:, None, None, :]
        )
        delta = world_grid - center_xy[:, None, None, :]
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        local_x = (
            cosine[:, None, None] * delta[..., 0] + sine[:, None, None] * delta[..., 1]
        )
        local_y = (
            -sine[:, None, None] * delta[..., 0] + cosine[:, None, None] * delta[..., 1]
        )
        if spec.footprint_kind == "circle":
            radius = 0.5 * min(spec.size_xyz_m[0], spec.size_xyz_m[1])
            return local_x * local_x + local_y * local_y <= radius * radius
        half_x = 0.5 * spec.size_xyz_m[0]
        half_y = 0.5 * spec.size_xyz_m[1]
        return (torch.abs(local_x) <= half_x) & (torch.abs(local_y) <= half_y)

    def _adjacent_overlap_ratios(self, local_positions: Any, yaws: Any) -> Any:
        masks = [
            self._footprint_mask(local_positions[:, index, :2], yaws[:, index], spec)
            for index, spec in enumerate(B03_INGREDIENT_SPECS)
        ]
        ratios: list[Any] = []
        for lower, upper in zip(masks[:-1], masks[1:], strict=True):
            intersection = (lower & upper).sum(dim=(1, 2)).to(torch.float32)
            lower_area = lower.sum(dim=(1, 2)).to(torch.float32)
            upper_area = upper.sum(dim=(1, 2)).to(torch.float32)
            # Normalize by the smaller footprint. This measures whether most
            # of the comparable layer area overlaps without penalizing a round
            # patty merely because the top bread is slightly larger and square.
            comparable_area = torch.minimum(lower_area, upper_area).clamp_min(1)
            ratios.append(intersection / comparable_area)
        return torch.stack(ratios, dim=1)

    def _stack_diagnostics(self) -> dict[str, Any]:
        local_positions = torch.stack(
            [self._world_to_site(actor.pose.p) for actor in self.ingredients],
            dim=1,
        )
        yaw_tilt = [self._local_yaw_and_tilt(actor) for actor in self.ingredients]
        yaws = torch.stack([item[0] for item in yaw_tilt], dim=1)
        tilts = torch.stack([item[1] for item in yaw_tilt], dim=1)

        xy_error = torch.linalg.norm(
            local_positions[:, :, :2] - self._target_local_xy[:, None, :],
            dim=2,
        )
        target_heights = torch.as_tensor(
            B03_TARGET_CENTER_HEIGHTS_M,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        height_error = torch.abs(local_positions[:, :, 2] - target_heights)

        stable = torch.stack(
            [
                (
                    torch.linalg.norm(actor.linear_velocity, dim=1)
                    <= self.SUCCESS_SPEC.maximum_linear_speed_mps
                )
                & (
                    torch.linalg.norm(actor.angular_velocity, dim=1)
                    <= self.SUCCESS_SPEC.maximum_angular_speed_rad_s
                )
                for actor in self.ingredients
            ],
            dim=1,
        )
        hand_contact, maximum_hand_force = self._hand_contact_by_object()
        released = ~hand_contact
        support_contact = self._support_contact_flags()
        overlap_ratio = self._adjacent_overlap_ratios(local_positions, yaws)

        pose_ok = (
            (xy_error <= self.SUCCESS_SPEC.target_xy_tolerance_m)
            & (height_error <= self.SUCCESS_SPEC.layer_height_tolerance_m)
            & (tilts <= self.SUCCESS_SPEC.maximum_tilt_rad)
        )

        # Layer geometry excludes hand release/speed so a completed lower layer
        # can be touched briefly without immediately becoming a D2 violation.
        layer_geometry_ok = pose_ok.clone()
        for index in range(1, len(self.ingredients)):
            layer_geometry_ok[:, index] &= (
                overlap_ratio[:, index - 1] >= self.SUCCESS_SPEC.minimum_overlap_ratio
            ) & support_contact[:, index - 1]

        stage_current: list[Any] = []
        for index in range(len(self.ingredients)):
            current = (
                layer_geometry_ok[:, : index + 1].all(dim=1)
                & stable[:, : index + 1].all(dim=1)
                & released[:, : index + 1].all(dim=1)
            )
            stage_current.append(current)
        stage_current_tensor = torch.stack(stage_current, dim=1)

        return {
            "local_positions": local_positions,
            "yaws": yaws,
            "tilts": tilts,
            "xy_error": xy_error,
            "height_error": height_error,
            "stable": stable,
            "hand_contact": hand_contact,
            "released": released,
            "maximum_hand_force": maximum_hand_force,
            "support_contact": support_contact,
            "overlap_ratio": overlap_ratio,
            "pose_ok": pose_ok,
            "layer_geometry_ok": layer_geometry_ok,
            "stage_current": stage_current_tensor,
        }

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        diagnostics = self._stack_diagnostics()
        contact = diagnostics["hand_contact"]
        self._food_touched |= contact

        placed_now = (
            diagnostics["layer_geometry_ok"]
            & diagnostics["stable"]
            & diagnostics["released"]
        )
        self._layer_ever_placed |= placed_now
        movable_count = placed_now[:, B03_PRESTACKED_LAYER_COUNT:].sum(dim=1)

        complete_now = diagnostics["stage_current"][:, -1]
        self._completion_dwell_counter = torch.where(
            complete_now,
            self._completion_dwell_counter + 1,
            torch.zeros_like(self._completion_dwell_counter),
        )
        success = complete_now & (
            self._completion_dwell_counter >= self._completion_dwell_steps
        )
        success = self._latch_task_success(success)

        return {
            "success": success,
            "one_movable_ingredient_placed": movable_count >= 1,
            "both_movable_ingredients_placed": movable_count == 2,
            # The final YAML stage includes the requested released stable dwell.
            "place_top_bread_and_stabilize": success,
            "ingredient_xy_error_m": diagnostics["xy_error"],
            "ingredient_height_error_m": diagnostics["height_error"],
            "ingredient_stable": diagnostics["stable"],
            "ingredient_released": diagnostics["released"],
            "adjacent_overlap_ratio": diagnostics["overlap_ratio"],
            "adjacent_support_contact": diagnostics["support_contact"],
            "layer_geometry_ok": diagnostics["layer_geometry_ok"],
        }

    def _check_generalization_validity(self, axis: GeneralizationAxis | None) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            diagnostics = self._stack_diagnostics()
            finite = torch.stack(
                [
                    torch.isfinite(actor.pose.raw_pose).all(dim=1)
                    for actor in self.ingredients
                ],
                dim=1,
            ).all(dim=1)
            local_positions = diagnostics["local_positions"]
            inside_workspace = (
                (
                    torch.abs(local_positions[:, :, 0])
                    <= B03_WORKSPACE_HALF_EXTENT_XY_M[0]
                )
                & (
                    torch.abs(local_positions[:, :, 1])
                    <= B03_WORKSPACE_HALF_EXTENT_XY_M[1]
                )
                & (local_positions[:, :, 2] >= -0.02)
            ).all(dim=1)
            # Spawned ingredients must remain separated before manipulation.
            separated = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for first in range(len(self.ingredients)):
                for second in range(first + 1, len(self.ingredients)):
                    if second < B03_PRESTACKED_LAYER_COUNT:
                        continue
                    distance = torch.linalg.norm(
                        local_positions[:, first, :2] - local_positions[:, second, :2],
                        dim=1,
                    )
                    separated &= distance >= 0.070
            return finite & inside_workspace & separated
        return super()._check_generalization_validity(axis)
