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
    MJCFActorSpec,
    PhysicalMaterialSpec,
    PoseSpec,
    StaticBoxSpec,
    YCBActorSpec,
)
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


B01_SETTINGS = TaskRegistry.register(TaskRegistry.load("b01_v1"))
B01_INTERACTION_SITE_ID = "dining_table"
B01_CLOTH_ENTITY_NAME = "b01_white_tablecloth"
B01_GLOBAL_DINING_CLUTTER_NAMES = frozenset(
    ("plate", "spoon", "bottle_blue", "bottle_yellow")
)
B01_OBJECT_NAMES = ("b01_plate", "b01_fork", "b01_knife")
B01_MODEL_IDS = ("pedestal_plate", "030_fork", "032_knife")
B01_PEDESTAL_PLATE_ASSET_PATH = (
    evodex_asset_dir() / "task_assets" / "b01" / "pedestal_plate.xml"
)
B01_PEDESTAL_PLATE_UNDERSIDE_HEIGHT_M = 0.028
B01_PEDESTAL_PLATE_RADIUS_M = 0.131
B01_PEDESTAL_RADIUS_M = 0.050
B01_PEDESTAL_PLATE_RADIAL_FINGER_DEPTH_M = (
    B01_PEDESTAL_PLATE_RADIUS_M - B01_PEDESTAL_RADIUS_M
)
# The source YCB utensils use +X as their long axis. Keep length growth modest
# while scaling both transverse axes together so visual and collision geometry
# gain the same, slightly larger grasping cross-section. The YCB assets are
# single meshes, so this necessarily thickens the full cross-section rather
# than only a semantically isolated handle region. Compensate the 20% Z-scale
# increase in density so nominal utensil mass remains unchanged.
B01_UTENSIL_SCALE_XYZ = (1.12, 1.60, 1.92)
B01_UTENSIL_DENSITY_MULTIPLIER = 1.0 / 1.20
B01_UTENSIL_ROOT_HEIGHT_M = 0.021
B01_TABLEWARE_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=2.00,
    dynamic_friction=1.80,
    restitution=0.0,
)
# The recorded L1 Human Viewer pose is stored relative to the bimanual robot
# base center. Reapplying this local pose to each layout keeps the observer at
# the same distance and orientation relative to the robot in L1/L2/L3.
B01_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.451338, 0.019193, 0.551949),
    # Unit-normalized base-local form derived from the six-decimal L1 pose.
    quaternion_wxyz=(
        0.8927431372351271,
        0.0030780369214366277,
        0.4505141551795293,
        -0.006100212387143034,
    ),
)
B01_LAYOUT_IDS = (
    LayoutId.KITCHEN_L1,
    LayoutId.KITCHEN_L2,
    LayoutId.KITCHEN_L3,
)


def _b01_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B01_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            B01_INTERACTION_SITE_ID
        )
        robot_base_center_pose = (
            interaction_site.bimanual_robot_base_center_pose
            or interaction_site.robot_base_center_pose
        )
        camera_world_pose = robot_base_center_pose.transform_pose(
            B01_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
        )
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=camera_world_pose.quaternion_wxyz,
        )
    return camera_poses


@dataclass(frozen=True)
class B01PlaceSettingItemSpec:
    name: str
    model_id: str
    target_local_xy_m: tuple[float, float]
    target_yaw_rad: float
    spawn_local_xy_m: tuple[float, float]
    spawn_yaw_rad: float
    root_height_m: float
    position_tolerance_m: float
    yaw_tolerance_rad: float | None
    yaw_period_rad: float
    allow_upside_down: bool
    footprint_half_size_xy_m: tuple[float, float]
    scale_xyz_multiplier: tuple[float, float, float]


# Dining-table local -X points toward the robot/diner. The YCB fork and knife
# handles are at model-local +X, so a pi yaw points both handles toward the
# robot. Local +Y/-Y place the utensils on the visual left/right of the plate.
B01_PLACE_SETTING_SPEC = (
    B01PlaceSettingItemSpec(
        name="b01_plate",
        model_id="pedestal_plate",
        target_local_xy_m=(0.0, 0.0),
        target_yaw_rad=0.0,
        spawn_local_xy_m=(0.0, -0.18),
        spawn_yaw_rad=math.radians(18.0),
        root_height_m=0.005,
        position_tolerance_m=0.12,
        yaw_tolerance_rad=None,
        yaw_period_rad=2.0 * math.pi,
        allow_upside_down=False,
        footprint_half_size_xy_m=(0.132, 0.132),
        scale_xyz_multiplier=(1.0, 1.0, 1.0),
    ),
    B01PlaceSettingItemSpec(
        name="b01_fork",
        model_id="030_fork",
        target_local_xy_m=(0.0, 0.205),
        target_yaw_rad=math.pi,
        spawn_local_xy_m=(-0.11, 0.22),
        spawn_yaw_rad=math.radians(38.0),
        root_height_m=B01_UTENSIL_ROOT_HEIGHT_M,
        position_tolerance_m=0.12,
        yaw_tolerance_rad=math.radians(35.0),
        yaw_period_rad=2.0 * math.pi,
        allow_upside_down=True,
        footprint_half_size_xy_m=(0.112, 0.021),
        scale_xyz_multiplier=B01_UTENSIL_SCALE_XYZ,
    ),
    B01PlaceSettingItemSpec(
        name="b01_knife",
        model_id="032_knife",
        target_local_xy_m=(0.0, -0.205),
        target_yaw_rad=math.pi,
        spawn_local_xy_m=(0.13, 0.22),
        spawn_yaw_rad=math.radians(-35.0),
        root_height_m=B01_UTENSIL_ROOT_HEIGHT_M,
        position_tolerance_m=0.12,
        yaw_tolerance_rad=math.radians(35.0),
        yaw_period_rad=2.0 * math.pi,
        allow_upside_down=True,
        footprint_half_size_xy_m=(0.121, 0.016),
        scale_xyz_multiplier=B01_UTENSIL_SCALE_XYZ,
    ),
)


@dataclass(frozen=True)
class B01SuccessSpec:
    maximum_tilt_rad: float = math.radians(35.0)
    maximum_linear_speed_mps: float = 0.04
    maximum_angular_speed_rad_s: float = 0.60
    contact_force_n: float = 0.35
    final_stable_dwell_s: float = 1.0
    minimum_plate_utensil_clearance_m: float = 0.008
    minimum_utensil_center_distance_m: float = 0.10
    side_min_offset_m: float = 0.105
    side_max_offset_m: float = 0.315
    longitudinal_alignment_tolerance_m: float = 0.14


B01_SUCCESS_SPEC = B01SuccessSpec()
# Local +X/-X follows the table's short edge (front/back in the task view).
# Shorten that edge for Kitchen L3 while preserving one size across layouts.
B01_CLOTH_HALF_SIZE_XY_M = (0.30, 0.48)
B01_CLOTH_HALF_HEIGHT_M = 0.002
B01_CLOTH_CENTER_LOCAL_Z_M = B01_CLOTH_HALF_HEIGHT_M
# Kitchen L3's RoboCasa counter center is 0.075 m along dining-table local +X
# from the shared interaction-site origin. Apply one offset to both the cloth
# and target regions; L1/L2 are already visually aligned.
B01_TABLE_ALIGNMENT_LOCAL_X_OFFSET_M_BY_LAYOUT = {
    LayoutId.KITCHEN_L1: 0.0,
    LayoutId.KITCHEN_L2: 0.0,
    LayoutId.KITCHEN_L3: 0.075,
}
B01_REFERENCE_LOCAL_Z_M = 0.006

B01_SPATIAL_PARAMETER_NAMES = (
    "plate_spawn_x_offset_m",
    "plate_spawn_y_offset_m",
    "fork_spawn_x_offset_m",
    "fork_spawn_y_offset_m",
    "knife_spawn_x_offset_m",
    "knife_spawn_y_offset_m",
    "utensil_spawn_yaw_offset_rad",
)
B01_PHYSICAL_PARAMETER_NAMES = (
    "plate_mass_log_scale",
    "fork_mass_log_scale",
    "knife_mass_log_scale",
    "tableware_friction_log_scale",
)
B01_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            *(
                GeneralizationParameterSpec(name, -0.025, 0.025)
                for name in B01_SPATIAL_PARAMETER_NAMES[:-1]
            ),
            GeneralizationParameterSpec(
                "utensil_spawn_yaw_offset_rad",
                -math.radians(12.0),
                math.radians(12.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "plate_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "fork_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "knife_mass_log_scale", math.log(0.85), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "tableware_friction_log_scale",
                math.log(0.85),
                math.log(1.20),
            ),
        ),
        requires_rebuild=True,
    ),
)


@register_env(B01_SETTINGS.env_id, max_episode_steps=B01_SETTINGS.max_steps)
class B01Env(DexVLABenchBaseEnv):
    """Arrange a plate, fork, and knife on a task-private white tablecloth."""

    TASK_SETTINGS = B01_SETTINGS
    SUCCESS_SPEC = B01_SUCCESS_SPEC
    GENERALIZATION_SPEC = B01_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B01_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _b01_human_viewer_camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Keep B01's Human Viewer on its calibrated free camera."""

        return None

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(
            options,
            self.interaction_site_robot_base_sapien_poses(),
        )

    def _dual_agents(self) -> tuple[Any, Any]:
        return self.left_agent, self.right_agent

    def _site_world_pose_spec(
        self,
        local_xyz: tuple[float, float, float],
        yaw_rad: float = 0.0,
    ) -> PoseSpec:
        half_yaw = 0.5 * yaw_rad
        return self.interaction_site_local_pose_to_world(
            PoseSpec(
                position_xyz=local_xyz,
                quaternion_wxyz=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
            )
        )

    def _task_layout_spec(self) -> Any:
        layout_id = self.current_layout_spec.layout_id
        retained_items = tuple(
            item
            for item in self.current_layout_spec.build_items
            if getattr(item, "name", None) not in B01_GLOBAL_DINING_CLUTTER_NAMES
        )
        cloth = StaticBoxSpec(
            name=B01_CLOTH_ENTITY_NAME,
            half_size_xyz=(
                B01_CLOTH_HALF_SIZE_XY_M[0],
                B01_CLOTH_HALF_SIZE_XY_M[1],
                B01_CLOTH_HALF_HEIGHT_M,
            ),
            pose=self._site_world_pose_spec(
                (
                    B01_TABLE_ALIGNMENT_LOCAL_X_OFFSET_M_BY_LAYOUT[layout_id],
                    0.0,
                    B01_CLOTH_CENTER_LOCAL_Z_M,
                )
            ),
            color_rgba=(0.97, 0.97, 0.95, 1.0),
            add_collision=True,
            body_type="kinematic",
        )
        task_objects: list[MJCFActorSpec | YCBActorSpec] = []
        for item in B01_PLACE_SETTING_SPEC:
            pose = self._site_world_pose_spec(
                (
                    item.spawn_local_xy_m[0],
                    item.spawn_local_xy_m[1],
                    item.root_height_m,
                ),
                item.spawn_yaw_rad,
            )
            if item.name == "b01_plate":
                task_objects.append(
                    MJCFActorSpec(
                        name=item.name,
                        asset_path=B01_PEDESTAL_PLATE_ASSET_PATH,
                        pose=pose,
                        body_type="dynamic",
                    )
                )
            else:
                task_objects.append(
                    YCBActorSpec(
                        name=item.name,
                        model_id=item.model_id,
                        pose=pose,
                        scale_xyz_multiplier=item.scale_xyz_multiplier,
                        density_multiplier=B01_UTENSIL_DENSITY_MULTIPLIER,
                        body_type="dynamic",
                        physical_material=B01_TABLEWARE_PHYSICAL_MATERIAL,
                    )
                )
        return replace(
            self.current_layout_spec,
            build_items=(*retained_items, cloth, *task_objects),
        )

    def _physical_actor_values(
        self,
    ) -> dict[str, ActorBuildValues] | None:
        roles = ("plate", "fork", "knife")
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            nominal_materials = tuple(
                B01_TABLEWARE_PHYSICAL_MATERIAL for _ in range(self.num_envs)
            )
            return {
                object_name: ActorBuildValues(
                    physical_materials=nominal_materials,
                )
                for object_name in B01_OBJECT_NAMES
            }
        values = self._generalization_values(
            B01_PHYSICAL_PARAMETER_NAMES,
            cache_name="_b01_physical_values",
        )
        friction_scales = positive_scale(
            values["tableware_friction_log_scale"]
        )
        materials = tuple(
            scaled_physical_material(
                B01_TABLEWARE_PHYSICAL_MATERIAL,
                friction_scale=float(friction_scale),
                restitution=B01_TABLEWARE_PHYSICAL_MATERIAL.restitution,
            )
            for friction_scale in friction_scales
        )
        return {
            object_name: ActorBuildValues(
                physical_materials=materials,
                mass_scales=tuple(
                    float(scale)
                    for scale in positive_scale(
                        values[f"{role}_mass_log_scale"]
                    )
                ),
            )
            for role, object_name in zip(roles, B01_OBJECT_NAMES, strict=True)
        }

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
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
        self.tablecloth = self.layout_scene.require_entity(B01_CLOTH_ENTITY_NAME)
        self.objects = tuple(
            self.layout_scene.require_entity(name) for name in B01_OBJECT_NAMES
        )
        self.plate, self.fork, self.knife = self.objects
        self.reference_actors = self._build_reference_actors()
        self._configure_tactile_queries()
        self._completion_dwell_steps = max(
            1,
            math.ceil(self.SUCCESS_SPEC.final_stable_dwell_s * self._control_freq),
        )
        self._inter_arm_contact_query = (
            self.left_agent.prepare_pairwise_contact_query(
                tuple(self.left_agent.robot.links),
                tuple(self.right_agent.robot.links),
            )
        )

    def _build_reference_actors(self) -> tuple[Any, ...]:
        colors = (
            (0.55, 0.70, 0.86, 0.34),
            (0.42, 0.68, 0.42, 0.42),
            (0.78, 0.34, 0.34, 0.42),
        )
        references: list[Any] = []
        for item, color in zip(B01_PLACE_SETTING_SPEC, colors, strict=True):
            builder = self.scene.create_actor_builder()
            material = sapien.render.RenderMaterial(
                base_color=list(color), roughness=0.92
            )
            if item.name == "b01_plate":
                builder.add_cylinder_visual(
                    radius=item.footprint_half_size_xy_m[0],
                    half_length=0.0003,
                    # SAPIEN cylinders use local +X as their long axis. Rotate
                    # X onto Z so the plate target lies flat on the cloth.
                    pose=sapien.Pose(
                        q=[0.70710678, 0.0, 0.70710678, 0.0]
                    ),
                    material=material,
                )
            else:
                builder.add_box_visual(
                    half_size=[
                        item.footprint_half_size_xy_m[0],
                        item.footprint_half_size_xy_m[1],
                        0.0003,
                    ],
                    material=material,
                )
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
            references.append(
                builder.build_kinematic(name=f"{item.name}_reference")
            )
        return tuple(references)

    def _configure_tactile_queries(self) -> None:
        self._target_groups = tuple(
            (name, (actor,))
            for name, actor in zip(B01_OBJECT_NAMES, self.objects, strict=True)
        )
        if not self.gpu_sim_enabled:
            for agent in self._dual_agents():
                agent.prepare_cpu_object_taxel_query(self._target_groups)

    def _site_pose(self, batch_size: int) -> Any:
        return Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

    def _local_planar_pose(
        self,
        local_xy: Any,
        local_yaw: Any,
        local_z: float,
    ) -> Any:
        half_yaw = 0.5 * local_yaw
        # Four [B] entries -> [B, 4] WXYZ planar quaternions.
        quaternion = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=1,
        )
        # [B, 2] + [B, 1] -> [B, 3] task-local positions.
        position = torch.cat(
            (
                local_xy,
                torch.full((len(local_xy), 1), local_z, device=self.device),
            ),
            dim=1,
        )
        return self._site_pose(len(local_xy)) * Pose.create_from_pq(
            p=position, q=quaternion
        )

    def _target_local_xy_m(self) -> tuple[tuple[float, float], ...]:
        layout_id = self.current_layout_id
        if layout_id is None:
            raise RuntimeError("B01 target regions require a selected layout.")
        local_x_offset_m = B01_TABLE_ALIGNMENT_LOCAL_X_OFFSET_M_BY_LAYOUT[
            layout_id
        ]
        return tuple(
            (
                item.target_local_xy_m[0] + local_x_offset_m,
                item.target_local_xy_m[1],
            )
            for item in B01_PLACE_SETTING_SPEC
        )

    def _set_reference_poses(self, env_idx: Any) -> None:
        for item, target_local_xy_m, reference in zip(
            B01_PLACE_SETTING_SPEC,
            self._target_local_xy_m(),
            self.reference_actors,
            strict=True,
        ):
            batch_size = len(env_idx)
            xy = torch.as_tensor(
                target_local_xy_m,
                device=self.device,
                dtype=torch.float32,
            ).repeat(batch_size, 1)
            yaw = torch.full(
                (batch_size,), item.target_yaw_rad, device=self.device
            )
            reference.set_pose(
                self._local_planar_pose(xy, yaw, B01_REFERENCE_LOCAL_Z_M)
            )

    def _spatial_values(self) -> dict[str, np.ndarray] | None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return None
        return self._generalization_values(
            B01_SPATIAL_PARAMETER_NAMES,
            cache_name="_b01_spatial_values",
        )

    def _set_initial_object_poses(self, env_idx: Any) -> None:
        values = self._spatial_values()
        indices = env_idx.detach().cpu().numpy()
        batch_size = len(env_idx)
        roles = ("plate", "fork", "knife")
        for object_index, (role, item, actor) in enumerate(
            zip(roles, B01_PLACE_SETTING_SPEC, self.objects, strict=True)
        ):
            xy = np.repeat(
                np.asarray(item.spawn_local_xy_m, dtype=np.float64)[None, :],
                batch_size,
                axis=0,
            )
            yaw = np.full(batch_size, item.spawn_yaw_rad, dtype=np.float64)
            if values is not None:
                xy[:, 0] += values[f"{role}_spawn_x_offset_m"][indices]
                xy[:, 1] += values[f"{role}_spawn_y_offset_m"][indices]
                if object_index > 0:
                    yaw += values["utensil_spawn_yaw_offset_rad"][indices]
            actor.set_pose(
                self._local_planar_pose(
                    torch.as_tensor(xy, device=self.device, dtype=torch.float32),
                    torch.as_tensor(yaw, device=self.device, dtype=torch.float32),
                    item.root_height_m,
                )
            )
            zero = torch.zeros((batch_size, 3), device=self.device)
            actor.set_linear_velocity(zero)
            actor.set_angular_velocity(zero)

    def _reset_agents(self, batch_size: int) -> None:
        base_poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(base_poses, list) or len(base_poses) != 2:
            raise RuntimeError("B01 requires exactly two robot base poses.")
        for agent, base_pose in zip(self._dual_agents(), base_poses, strict=True):
            rest = agent.keyframes.get("rest")
            if rest is None or rest.qpos is None:
                raise RuntimeError(f"B01 agent {agent.uid!r} has no rest keyframe.")
            # [N] -> [B, N]: one rest vector per reset environment.
            qpos = torch.as_tensor(
                rest.qpos, device=self.device, dtype=torch.float32
            ).unsqueeze(0).repeat(batch_size, 1)
            agent.reset(qpos)
            agent.robot.set_pose(base_pose)

    def _ensure_state(self) -> None:
        if hasattr(self, "_object_touched"):
            return
        self._object_touched = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )
        self._object_ever_placed = torch.zeros_like(self._object_touched)
        self._completion_steps_task = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._tablecloth_initial_position = torch.zeros(
            (self.num_envs, 3), device=self.device
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
            self._ensure_state()
            self._set_reference_poses(env_idx)
            self._set_initial_object_poses(env_idx)
            for agent in self._dual_agents():
                agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_agents(batch_size)
            self._object_touched[env_idx] = False
            self._object_ever_placed[env_idx] = False
            self._completion_steps_task[env_idx] = 0
            self._tablecloth_initial_position[env_idx] = self.tablecloth.pose.p[
                env_idx
            ]

    @staticmethod
    def _periodic_error(value: Any, target: float, period: float) -> Any:
        return torch.abs(
            torch.remainder(value - target + 0.5 * period, period)
            - 0.5 * period
        )

    @staticmethod
    def _tilt_angle_from_up_cosine(
        up_cosine: Any,
        *,
        allow_upside_down: bool,
    ) -> Any:
        aligned_up_cosine = (
            torch.abs(up_cosine) if allow_upside_down else up_cosine
        )
        return torch.acos(aligned_up_cosine.clamp(-1.0, 1.0))

    def _object_local_pose_diagnostics(self) -> tuple[Any, Any, Any]:
        site_inverse = self._site_pose(self.num_envs).inv()
        local_poses = [site_inverse * actor.pose for actor in self.objects]
        # Three [B, 3] -> [B, 3, 3].
        positions = torch.stack([pose.p for pose in local_poses], dim=1)
        yaws: list[Any] = []
        tilts: list[Any] = []
        for pose, item in zip(
            local_poses,
            B01_PLACE_SETTING_SPEC,
            strict=True,
        ):
            w, x, y, z = pose.q.unbind(dim=1)
            yaws.append(
                torch.atan2(
                    2.0 * (w * z + x * y),
                    1.0 - 2.0 * (y * y + z * z),
                )
            )
            up_cosine = 1.0 - 2.0 * (x * x + y * y)
            # A fork or knife laid flat is semantically valid on either face.
            # Keep the plate directional so an upside-down plate still fails.
            tilts.append(
                self._tilt_angle_from_up_cosine(
                    up_cosine,
                    allow_upside_down=item.allow_upside_down,
                )
            )
        return positions, torch.stack(yaws, dim=1), torch.stack(tilts, dim=1)

    def _contact_by_object(self) -> tuple[Any, Any]:
        contact = torch.zeros(
            (self.num_envs, 3), dtype=torch.bool, device=self.device
        )
        maximum = torch.zeros(self.num_envs, device=self.device)
        for agent in self._dual_agents():
            forces = agent.get_object_taxel_world_forces(self._target_groups)
            for object_index, object_name in enumerate(B01_OBJECT_NAMES):
                parent_forces = agent.aggregate_taxel_world_forces_by_parent(
                    forces[object_name]
                )
                norms = torch.linalg.norm(parent_forces, dim=-1)
                contact[:, object_index] |= (
                    norms >= self.SUCCESS_SPEC.contact_force_n
                ).any(dim=1)
                maximum = torch.maximum(maximum, norms.amax(dim=1))
        return contact, maximum

    def _plate_utensil_clear(self, positions: Any, yaws: Any, index: int) -> Any:
        item = B01_PLACE_SETTING_SPEC[index]
        delta = positions[:, 0, :2] - positions[:, index, :2]
        cosine, sine = torch.cos(yaws[:, index]), torch.sin(yaws[:, index])
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        half_x, half_y = item.footprint_half_size_xy_m
        closest_x = local_x.clamp(-half_x, half_x)
        closest_y = local_y.clamp(-half_y, half_y)
        distance = torch.sqrt(
            (local_x - closest_x) ** 2 + (local_y - closest_y) ** 2
        )
        plate_radius = B01_PLACE_SETTING_SPEC[0].footprint_half_size_xy_m[0]
        return distance >= (
            plate_radius + self.SUCCESS_SPEC.minimum_plate_utensil_clearance_m
        )

    def _placement_diagnostics(self) -> dict[str, Any]:
        positions, yaws, tilts = self._object_local_pose_diagnostics()
        targets = torch.as_tensor(
            self._target_local_xy_m(),
            device=self.device,
            dtype=torch.float32,
        )
        position_error = torch.linalg.norm(
            positions[:, :, :2] - targets[None, :, :], dim=2
        )
        orientation_ok = torch.ones_like(position_error, dtype=torch.bool)
        for index, item in enumerate(B01_PLACE_SETTING_SPEC):
            if item.yaw_tolerance_rad is None:
                continue
            orientation_ok[:, index] = self._periodic_error(
                yaws[:, index], item.target_yaw_rad, item.yaw_period_rad
            ) <= item.yaw_tolerance_rad
        position_ok = position_error <= torch.as_tensor(
            [item.position_tolerance_m for item in B01_PLACE_SETTING_SPEC],
            device=self.device,
        )
        plate_xy = positions[:, 0, :2]
        fork_delta = positions[:, 1, :2] - plate_xy
        knife_delta = positions[:, 2, :2] - plate_xy
        side_relation_ok = (
            (fork_delta[:, 1] >= self.SUCCESS_SPEC.side_min_offset_m)
            & (fork_delta[:, 1] <= self.SUCCESS_SPEC.side_max_offset_m)
            & (knife_delta[:, 1] <= -self.SUCCESS_SPEC.side_min_offset_m)
            & (knife_delta[:, 1] >= -self.SUCCESS_SPEC.side_max_offset_m)
            & (
                torch.abs(fork_delta[:, 0])
                <= self.SUCCESS_SPEC.longitudinal_alignment_tolerance_m
            )
            & (
                torch.abs(knife_delta[:, 0])
                <= self.SUCCESS_SPEC.longitudinal_alignment_tolerance_m
            )
        )
        non_overlap = (
            self._plate_utensil_clear(positions, yaws, 1)
            & self._plate_utensil_clear(positions, yaws, 2)
            & (
                torch.linalg.norm(
                    positions[:, 1, :2] - positions[:, 2, :2], dim=1
                )
                >= self.SUCCESS_SPEC.minimum_utensil_center_distance_m
            )
        )
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
                for actor in self.objects
            ],
            dim=1,
        )
        correct = (
            position_ok
            & orientation_ok
            & (tilts <= self.SUCCESS_SPEC.maximum_tilt_rad)
            & stable
        )
        return {
            "positions": positions,
            "position_error": position_error,
            "orientation_ok": orientation_ok,
            "stable": stable,
            "correct": correct,
            "side_relation_ok": side_relation_ok,
            "non_overlap": non_overlap,
        }

    def evaluate(self) -> dict[str, Any]:
        self._ensure_state()
        contact, _ = self._contact_by_object()
        self._object_touched |= contact
        diagnostics = self._placement_diagnostics()
        placed_now = diagnostics["correct"] & ~contact
        self._object_ever_placed |= placed_now
        relation_now = (
            diagnostics["correct"].all(dim=1)
            & diagnostics["side_relation_ok"]
            & diagnostics["non_overlap"]
            & ~contact.any(dim=1)
        )
        self._completion_steps_task = torch.where(
            relation_now,
            self._completion_steps_task + 1,
            torch.zeros_like(self._completion_steps_task),
        )
        success = self._latch_task_success(
            self._completion_steps_task >= self._completion_dwell_steps
        )
        placed_count = placed_now.sum(dim=1)
        return {
            "success": success,
            "one_tableware_placed": placed_count >= 1,
            "two_tableware_placed": placed_count >= 2,
            "all_tableware_placed": placed_count == 3,
            "place_setting_stable": success,
            "placement_position_error_m": diagnostics["position_error"],
            "placement_orientation_ok": diagnostics["orientation_ok"],
            "place_setting_side_relation_ok": diagnostics["side_relation_ok"],
            "place_setting_non_overlap": diagnostics["non_overlap"],
        }

    def _check_generalization_validity(
        self, axis: GeneralizationAxis | None
    ) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            finite = torch.stack(
                [torch.isfinite(actor.pose.raw_pose).all(dim=1) for actor in self.objects],
                dim=1,
            ).all(dim=1)
            return finite
        return super()._check_generalization_validity(axis)
