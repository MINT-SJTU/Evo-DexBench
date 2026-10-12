from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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
    ArticulationBuildValues,
    LayoutBuilder,
    LayoutRegistry,
    PhysicalMaterialSpec,
    PoseSpec,
)
from dex_benchmark.sim.scenes.layouts.studio import STUDIO_PHYSICAL_MATERIAL
from dex_benchmark.sim.robots.tactile import PairwiseContactQueryHandle
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.actor import Actor
    from mani_skill.utils.structs.pose import Pose
    from transforms3d.euler import euler2quat
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    torch = None
    Actor = None
    Pose = None
    euler2quat = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls):
            return cls

        return decorator


B02_SETTINGS = TaskRegistry.register(TaskRegistry.load("b02_v1"))
B02_INTERACTION_SITE_ID = "workbench"
B02_LAYOUT_IDS = (
    LayoutId.STUDIO_L1,
    LayoutId.STUDIO_L2,
    LayoutId.STUDIO_L3,
)
# Recorded with the Human Viewer pose hotkey in Studio L1, then expressed in
# the workbench interaction frame so L2/L3 retain the same task-relative view.
B02_HUMAN_VIEWER_CAMERA_LOCAL_POSE = PoseSpec(
    position_xyz=(-0.582489295982795, 0.08825427846407519, 0.478372),
    quaternion_wxyz=(
        0.9872119661541303,
        -0.00088768273083005,
        0.15931577902614003,
        0.005498040992926445,
    ),
)


def _b02_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B02_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            B02_INTERACTION_SITE_ID
        )
        camera_world_pose = interaction_site.frame_pose.transform_pose(
            B02_HUMAN_VIEWER_CAMERA_LOCAL_POSE
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


@dataclass(frozen=True)
class B02SuccessSpec:
    xy_tolerance_m: float = 0.015
    yaw_tolerance_rad: float = math.radians(8.0)
    tilt_tolerance_rad: float = math.radians(8.0)
    height_tolerance_m: float = 0.015
    max_linear_speed_mps: float = 0.03
    max_angular_speed_rad_s: float = 0.50
    adjacency_center_tolerance_m: float = 0.020
    adjacency_angle_tolerance_rad: float = math.radians(12.0)
    adjacency_min_overlap_m: float = 0.020
    footprint_iou_quality_target: float = 0.90
    completion_dwell_s: float = 0.50
    grasp_contact_force_n: float = 0.30


@dataclass(frozen=True)
class B02TargetPieceSpec:
    """One role in the authored hammer pattern and its matching reference."""

    name: str
    asset_file: str
    footprint_size_xy_m: tuple[float, float]
    target_local_xy_m: tuple[float, float]
    target_yaw_rad: float
    yaw_symmetry_period_rad: float


B02_SUCCESS_SPEC = B02SuccessSpec()
B02_TARGET_PATTERN = (
    B02TargetPieceSpec(
        name="hammer_head",
        asset_file="bevel-lq-plate-2x4.glb",
        footprint_size_xy_m=(0.1272, 0.0632),
        target_local_xy_m=(-0.0550, 0.0),
        target_yaw_rad=math.pi / 2.0,
        yaw_symmetry_period_rad=math.pi,
    ),
    B02TargetPieceSpec(
        name="hammer_handle",
        asset_file="bevel-lq-plate-1x4.glb",
        footprint_size_xy_m=(0.1272, 0.0316),
        target_local_xy_m=(0.0406, 0.0),
        target_yaw_rad=math.pi,
        yaw_symmetry_period_rad=math.pi,
    ),
    B02TargetPieceSpec(
        name="hammer_claw",
        asset_file="bevel-lq-plate-corner.glb",
        footprint_size_xy_m=(0.0636, 0.0632),
        target_local_xy_m=(-0.0550, 0.0954),
        target_yaw_rad=3.0 * math.pi / 2.0,
        yaw_symmetry_period_rad=2.0 * math.pi,
    ),
)
B02_BLOCK_NAMES = tuple(piece.name for piece in B02_TARGET_PATTERN)
B02_ASSET_FILES = tuple(piece.asset_file for piece in B02_TARGET_PATTERN)
B02_REQUIRED_ASSET_FILES = (*B02_ASSET_FILES, "Textures/colormap.png")
B02_SCALE = 0.4
# Planar dimensions measured from the submitted GLBs after the 0.4 XY scale.
B02_FOOTPRINT_SIZE_XY = tuple(
    piece.footprint_size_xy_m for piece in B02_TARGET_PATTERN
)
B02_BLOCK_THICKNESS_MULTIPLIER = 1.50
B02_BLOCK_HEIGHT_M = 0.020 * B02_BLOCK_THICKNESS_MULTIPLIER
# The source meshes use +Y as height before ``visual_pose`` rotates them into
# workbench +Z, so only visual scale Y changes. Collision Z dimensions follow
# the same 1.5x thickness multiplier.
B02_BLOCK_VISUAL_SCALE_XYZ = (
    B02_SCALE,
    B02_SCALE * B02_BLOCK_THICKNESS_MULTIPLIER,
    B02_SCALE,
)
B02_BLOCK_BASE_HALF_HEIGHT_M = 0.006 * B02_BLOCK_THICKNESS_MULTIPLIER
B02_BLOCK_STUD_HALF_HEIGHT_M = 0.004 * B02_BLOCK_THICKNESS_MULTIPLIER
B02_BLOCK_STUD_CENTER_HEIGHT_M = (
    2.0 * B02_BLOCK_BASE_HALF_HEIGHT_M + B02_BLOCK_STUD_HALF_HEIGHT_M
)
B02_BLOCK_PHYSICAL_MATERIAL = PhysicalMaterialSpec(
    static_friction=1.60,
    dynamic_friction=1.40,
    restitution=0.0,
)
# Preserve approximately the original mass after increasing collision volume.
B02_BLOCK_DENSITY_KG_M3 = 650.0 / B02_BLOCK_THICKNESS_MULTIPLIER
# The complete hammer is rotated by 90 degrees from the workbench axes so that
# it appears upright in the task's interaction camera.
B02_NOMINAL_TARGET_LOCAL_XY = tuple(
    piece.target_local_xy_m for piece in B02_TARGET_PATTERN
)
B02_NOMINAL_TARGET_YAW = tuple(
    piece.target_yaw_rad for piece in B02_TARGET_PATTERN
)
B02_YAW_SYMMETRY_PERIOD = tuple(
    piece.yaw_symmetry_period_rad for piece in B02_TARGET_PATTERN
)
B02_REFERENCE_PAPER_CENTER_LOCAL_XY = (0.0088, 0.0318)
B02_REFERENCE_PAPER_HALF_SIZE_XY = (0.135, 0.125)
B02_REFERENCE_PAPER_HALF_HEIGHT_M = 0.00025
B02_REFERENCE_INK_HALF_HEIGHT_M = 0.00005
B02_REFERENCE_INK_CENTER_Z_M = 0.00060
B02_NOMINAL_INITIAL_LOCAL_XY = (
    (-0.16, -0.13),
    (0.15, -0.12),
    (0.16, 0.15),
)
B02_NOMINAL_INITIAL_YAW = (0.25, -0.55, 0.80)
B02_MASK_SIZE = 128
B02_MASK_EXTENT_M = 0.45
B02_SPATIAL_PARAMETER_NAMES = (
    "head_x_offset_m",
    "head_y_offset_m",
    "head_yaw_offset_rad",
    "handle_x_offset_m",
    "handle_y_offset_m",
    "handle_yaw_offset_rad",
    "claw_x_offset_m",
    "claw_y_offset_m",
    "claw_yaw_offset_rad",
    "target_x_offset_m",
    "target_y_offset_m",
    "target_yaw_offset_rad",
)
B02_PHYSICAL_PARAMETER_NAMES = (
    "head_mass_log_scale",
    "handle_mass_log_scale",
    "claw_mass_log_scale",
    "contact_friction_log_scale",
)
B02_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            *(
                GeneralizationParameterSpec(name, -0.05, 0.05)
                for name in (
                    "head_x_offset_m",
                    "head_y_offset_m",
                    "handle_x_offset_m",
                    "handle_y_offset_m",
                    "claw_x_offset_m",
                    "claw_y_offset_m",
                )
            ),
            *(
                GeneralizationParameterSpec(
                    name,
                    -math.radians(30.0),
                    math.radians(30.0),
                )
                for name in (
                    "head_yaw_offset_rad",
                    "handle_yaw_offset_rad",
                    "claw_yaw_offset_rad",
                )
            ),
            GeneralizationParameterSpec("target_x_offset_m", -0.05, 0.05),
            GeneralizationParameterSpec("target_y_offset_m", -0.05, 0.05),
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
            GeneralizationParameterSpec(
                "head_mass_log_scale", math.log(0.75), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "handle_mass_log_scale", math.log(0.75), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "claw_mass_log_scale", math.log(0.75), math.log(1.25)
            ),
            GeneralizationParameterSpec(
                "contact_friction_log_scale",
                math.log(0.70),
                math.log(1.30),
            ),
        ),
        requires_rebuild=True,
    ),
)


@register_env(B02_SETTINGS.env_id, max_episode_steps=B02_SETTINGS.max_steps)
class B02Env(DexVLABenchBaseEnv):
    """Arrange three Kenney Brick Kit pieces into a hammer silhouette."""

    TASK_SETTINGS = B02_SETTINGS
    SUCCESS_SPEC = B02_SUCCESS_SPEC
    GENERALIZATION_SPEC = B02_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B02_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _b02_human_viewer_camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Keep B02's Human Viewer on its calibrated free camera."""

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

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self.current_layout_spec,
            articulation_values=self._workbench_physical_values(),
        ).build()
        self.workbench = self.layout_scene.require_entity("workbench")
        self.blocks = self._build_blocks()
        self.reference_paper = self._build_reference_paper()
        self.reference_actors = self._build_reference_actors()
        self._configure_queries_and_constants()

    def _physical_values(self) -> dict[str, np.ndarray] | None:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None
        return self._generalization_values(
            B02_PHYSICAL_PARAMETER_NAMES,
            cache_name="_b02_physical_values",
        )

    def _workbench_physical_values(
        self,
    ) -> dict[str, ArticulationBuildValues] | None:
        values = self._physical_values()
        if values is None:
            return None
        friction = positive_scale(values["contact_friction_log_scale"])
        return {
            "workbench": ArticulationBuildValues(
                physical_materials=tuple(
                    scaled_physical_material(
                        STUDIO_PHYSICAL_MATERIAL,
                        friction_scale=float(scale),
                        restitution=STUDIO_PHYSICAL_MATERIAL.restitution,
                    )
                    for scale in friction
                )
            )
        }

    def _brick_material_and_density(
        self,
        env_index: int,
        block_index: int,
    ) -> tuple[Any, float]:
        values = self._physical_values()
        if values is None:
            friction_scale, mass_scale = 1.0, 1.0
        else:
            friction_scale = float(
                positive_scale(values["contact_friction_log_scale"])[env_index]
            )
            block_role = ("head", "handle", "claw")[block_index]
            mass_scale = float(
                positive_scale(values[f"{block_role}_mass_log_scale"])[env_index]
            )
        material_spec = scaled_physical_material(
            B02_BLOCK_PHYSICAL_MATERIAL,
            friction_scale=friction_scale,
            restitution=B02_BLOCK_PHYSICAL_MATERIAL.restitution,
        )
        material = sapien.physx.PhysxMaterial(
            material_spec.static_friction,
            material_spec.dynamic_friction,
            material_spec.restitution,
        )
        return material, B02_BLOCK_DENSITY_KG_M3 * mass_scale

    def _add_block_collision(
        self,
        builder: Any,
        block_index: int,
        material: Any,
        density: float,
    ) -> None:
        half_x, half_y = (
            0.5 * value for value in B02_FOOTPRINT_SIZE_XY[block_index]
        )
        base_half_z = B02_BLOCK_BASE_HALF_HEIGHT_M
        if block_index < 2:
            builder.add_box_collision(
                half_size=[half_x, half_y, base_half_z],
                pose=sapien.Pose(p=[0.0, 0.0, base_half_z]),
                material=material,
                density=density,
            )
        else:
            # The GLB is a 2x2 footprint with its lower-left quadrant removed.
            # These two non-overlapping boxes preserve that footprint exactly.
            builder.add_box_collision(
                half_size=[
                    half_x,
                    half_y / 2.0,
                    base_half_z,
                ],
                pose=sapien.Pose(
                    p=[
                        0.0,
                        half_y / 2.0,
                        base_half_z,
                    ]
                ),
                material=material,
                density=density,
            )
            builder.add_box_collision(
                half_size=[
                    half_x / 2.0,
                    half_y / 2.0,
                    base_half_z,
                ],
                pose=sapien.Pose(
                    p=[
                        half_x / 2.0,
                        -half_y / 2.0,
                        base_half_z,
                    ]
                ),
                material=material,
                density=density,
            )
        stud_positions = (
            (
                (-0.0477, -0.0158),
                (-0.0159, -0.0158),
                (0.0159, -0.0158),
                (0.0477, -0.0158),
                (-0.0477, 0.0158),
                (-0.0159, 0.0158),
                (0.0159, 0.0158),
                (0.0477, 0.0158),
            ),
            ((-0.0477, 0.0), (-0.0159, 0.0), (0.0159, 0.0), (0.0477, 0.0)),
            ((-0.0159, 0.0158), (0.0159, 0.0158), (0.0159, -0.0158)),
        )[block_index]
        for stud_x, stud_y in stud_positions:
            builder.add_cylinder_collision(
                radius=0.0075,
                half_length=B02_BLOCK_STUD_HALF_HEIGHT_M,
                pose=sapien.Pose(
                    p=[stud_x, stud_y, B02_BLOCK_STUD_CENTER_HEIGHT_M],
                    q=euler2quat(0.0, math.pi / 2.0, 0.0),
                ),
                material=material,
                density=density,
            )

    def _build_blocks(self) -> tuple[Any, ...]:
        asset_root = Path(evodex_asset_dir()) / "kenney" / "brick_kit"
        missing = [
            str(asset_root / name)
            for name in B02_REQUIRED_ASSET_FILES
            if not (asset_root / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "B02 Brick Kit assets are missing: " + ", ".join(missing)
            )
        blocks: list[Any] = []
        visual_pose = sapien.Pose(q=euler2quat(math.pi / 2.0, 0.0, 0.0))
        block_assets = zip(B02_BLOCK_NAMES, B02_ASSET_FILES, strict=True)
        for block_index, (block_name, filename) in enumerate(block_assets):
            raw_actors: list[Any] = []
            for env_index in range(self.num_envs):
                builder = self.scene.create_actor_builder()
                material, density = self._brick_material_and_density(
                    env_index,
                    block_index,
                )
                self._add_block_collision(builder, block_index, material, density)
                builder.add_visual_from_file(
                    filename=str(asset_root / filename),
                    pose=visual_pose,
                    scale=B02_BLOCK_VISUAL_SCALE_XYZ,
                )
                builder.set_scene_idxs([env_index])
                builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
                raw_actors.append(
                    builder.build(name=f"b02_{block_name}_{env_index}")
                )
            blocks.append(Actor.merge(raw_actors, name=f"b02_{block_name}"))
        return tuple(blocks)

    def _build_reference_paper(self) -> Any:
        material = sapien.render.RenderMaterial(
            base_color=[0.96, 0.95, 0.90, 1.0],
            roughness=0.95,
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=[
                B02_REFERENCE_PAPER_HALF_SIZE_XY[0],
                B02_REFERENCE_PAPER_HALF_SIZE_XY[1],
                B02_REFERENCE_PAPER_HALF_HEIGHT_M,
            ],
            material=material,
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        return builder.build_kinematic(name="b02_reference_paper")

    def _build_reference_actors(self) -> tuple[Any, ...]:
        fill_colors = (
            (0.50, 0.64, 0.70, 1.0),
            (0.57, 0.69, 0.64, 1.0),
            (0.62, 0.65, 0.73, 1.0),
        )
        references: list[Any] = []
        for block_index, block_name in enumerate(B02_BLOCK_NAMES):
            builder = self.scene.create_actor_builder()
            material = sapien.render.RenderMaterial(
                base_color=fill_colors[block_index],
                roughness=0.95,
            )
            half_x, half_y = (
                0.5 * value for value in B02_FOOTPRINT_SIZE_XY[block_index]
            )
            if block_index < 2:
                builder.add_box_visual(
                    half_size=[
                        half_x,
                        half_y,
                        B02_REFERENCE_INK_HALF_HEIGHT_M,
                    ],
                    material=material,
                )
            else:
                # Match the GLB's three occupied quadrants without overlapping
                # coplanar visuals, which can otherwise cause rendering artifacts.
                builder.add_box_visual(
                    half_size=[
                        half_x,
                        half_y / 2.0,
                        B02_REFERENCE_INK_HALF_HEIGHT_M,
                    ],
                    pose=sapien.Pose(
                        p=[
                            0.0,
                            half_y / 2.0,
                            0.0,
                        ]
                    ),
                    material=material,
                )
                builder.add_box_visual(
                    half_size=[
                        half_x / 2.0,
                        half_y / 2.0,
                        B02_REFERENCE_INK_HALF_HEIGHT_M,
                    ],
                    pose=sapien.Pose(
                        p=[
                            half_x / 2.0,
                            -half_y / 2.0,
                            0.0,
                        ]
                    ),
                    material=material,
                )
            builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
            references.append(
                builder.build_kinematic(name=f"b02_reference_{block_name}")
            )
        return tuple(references)

    def _configure_queries_and_constants(self) -> None:
        self._grasp_queries: list[list[PairwiseContactQueryHandle]] = []
        for role_index, agent in enumerate(self._dual_agents()):
            prefix = "lh" if role_index == 0 else "rh"
            finger_names = tuple(
                f"{prefix}_{finger}distal"
                for finger in ("th", "ff", "mf", "rf", "lf")
            )
            missing = [
                name for name in finger_names if name not in agent.robot.links_map
            ]
            if missing:
                raise RuntimeError(
                    f"B02 missing grasp links for agent {agent.uid!r}: "
                    + ", ".join(missing)
                )
            finger_links = tuple(agent.robot.links_map[name] for name in finger_names)
            self._grasp_queries.append(
                [
                    agent.prepare_pairwise_contact_query(finger_links, (block,))
                    for block in self.blocks
                ]
            )
        self._inter_arm_contact_query = (
            self.left_agent.prepare_pairwise_contact_query(
                tuple(self.left_agent.robot.links),
                tuple(self.right_agent.robot.links),
            )
        )
        self._completion_dwell_steps = max(
            1, math.ceil(self.SUCCESS_SPEC.completion_dwell_s * self._control_freq)
        )
        pixel_size = B02_MASK_EXTENT_M / B02_MASK_SIZE
        # [128] pixel indices -> [128] workbench-local pixel-center coordinates.
        centers = (
            torch.arange(
                B02_MASK_SIZE,
                device=self.device,
                dtype=torch.float32,
            )
            + 0.5
        ) * pixel_size - 0.5 * B02_MASK_EXTENT_M
        grid_y, grid_x = torch.meshgrid(centers, centers, indexing="ij")
        # [128, 128] + [128, 128] -> [128, 128, 2] local XY grid.
        self._mask_grid_xy = torch.stack([grid_x, grid_y], dim=-1)

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
            target_local_xy, target_local_yaw = self._target_local_geometry(
                env_idx
            )
            self._target_local_xy[env_idx] = target_local_xy
            self._target_local_yaw[env_idx] = target_local_yaw
            self._set_reference_poses(env_idx)
            self._set_initial_block_poses(env_idx)
            for agent in self._dual_agents():
                agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_agents_to_rest(batch_size)
            self._completion_steps_task[env_idx] = 0
            self._block_was_grasped[env_idx] = False
            self._block_was_correct[env_idx] = False

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_target_local_xy"):
            return
        self._target_local_xy = torch.zeros(
            (self.num_envs, 3, 2),
            device=self.device,
        )
        self._target_local_yaw = torch.zeros(
            (self.num_envs, 3),
            device=self.device,
        )
        self._completion_steps_task = torch.zeros(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._block_was_grasped = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.bool,
            device=self.device,
        )
        self._block_was_correct = torch.zeros_like(self._block_was_grasped)

    def _spatial_values(self) -> dict[str, np.ndarray] | None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return None
        return self._generalization_values(
            B02_SPATIAL_PARAMETER_NAMES,
            cache_name="_b02_spatial_values",
        )

    def _target_local_geometry(self, env_idx: Any) -> tuple[Any, Any]:
        batch_size = len(env_idx)
        # [3, 2] -> [B, 3, 2]: one target XY per role and environment.
        positions = (
            torch.as_tensor(
                B02_NOMINAL_TARGET_LOCAL_XY,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        # [3] -> [B, 3]: one target yaw per role and environment.
        yaws = (
            torch.as_tensor(
                B02_NOMINAL_TARGET_YAW,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        values = self._spatial_values()
        if values is None:
            return positions, yaws
        indices = env_idx.detach().cpu().numpy()
        target_offset = torch.as_tensor(
            np.stack(
                [
                    values["target_x_offset_m"][indices],
                    values["target_y_offset_m"][indices],
                ],
                axis=1,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        target_yaw = torch.as_tensor(
            values["target_yaw_offset_rad"][indices],
            device=self.device,
            dtype=torch.float32,
        )
        cosine, sine = torch.cos(target_yaw), torch.sin(target_yaw)
        # Four [B] entries -> [B, 2, 2] planar rotation matrices.
        rotation = torch.stack(
            [cosine, -sine, sine, cosine],
            dim=1,
        ).reshape(batch_size, 2, 2)
        # [B, 3, 2] @ [B, 2, 2].T -> [B, 3, 2]: whole-target planar transform.
        positions = torch.bmm(
            positions,
            rotation.transpose(1, 2),
        ) + target_offset[:, None, :]
        return positions, yaws + target_yaw[:, None]

    def _site_pose(self, batch_size: int) -> Any:
        return Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0),
                batch_size=batch_size,
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

    def _local_planar_pose(
        self,
        local_xy: Any,
        local_yaw: Any,
        z: float,
    ) -> Any:
        half = 0.5 * local_yaw
        # Four [B] entries -> [B, 4] WXYZ planar quaternions.
        quaternion = torch.stack(
            [
                torch.cos(half),
                torch.zeros_like(half),
                torch.zeros_like(half),
                torch.sin(half),
            ],
            dim=1,
        )
        # [B, 2] + [B, 1] -> [B, 3] local XYZ positions.
        position = torch.cat(
            [
                local_xy,
                torch.full((len(local_xy), 1), z, device=self.device),
            ],
            dim=1,
        )
        local_pose = Pose.create_from_pq(p=position, q=quaternion)
        return self._site_pose(len(local_xy)) * local_pose

    def _set_reference_poses(self, env_idx: Any) -> None:
        target_yaw = self._target_local_yaw[env_idx, 0]
        target_yaw_delta = target_yaw - B02_NOMINAL_TARGET_YAW[0]
        paper_from_head = torch.as_tensor(
            [
                B02_REFERENCE_PAPER_CENTER_LOCAL_XY[0]
                - B02_NOMINAL_TARGET_LOCAL_XY[0][0],
                B02_REFERENCE_PAPER_CENTER_LOCAL_XY[1]
                - B02_NOMINAL_TARGET_LOCAL_XY[0][1],
            ],
            device=self.device,
            dtype=torch.float32,
        )
        cosine = torch.cos(target_yaw_delta)
        sine = torch.sin(target_yaw_delta)
        # [2] + two [B] rotations -> [B, 2]: paper follows the complete
        # translated/rotated target while remaining centered around its outline.
        rotated_paper_offset = torch.stack(
            [
                cosine * paper_from_head[0] - sine * paper_from_head[1],
                sine * paper_from_head[0] + cosine * paper_from_head[1],
            ],
            dim=1,
        )
        paper_local_xy = (
            self._target_local_xy[env_idx, 0] + rotated_paper_offset
        )
        self.reference_paper.set_pose(
            self._local_planar_pose(
                paper_local_xy,
                target_yaw,
                B02_REFERENCE_PAPER_HALF_HEIGHT_M,
            )
        )
        for index, reference in enumerate(self.reference_actors):
            reference.set_pose(
                self._local_planar_pose(
                    self._target_local_xy[env_idx, index],
                    self._target_local_yaw[env_idx, index],
                    B02_REFERENCE_INK_CENTER_Z_M,
                )
            )

    def _set_initial_block_poses(self, env_idx: Any) -> None:
        batch_size = len(env_idx)
        values = self._spatial_values()
        indices = env_idx.detach().cpu().numpy()
        role_names = ("head", "handle", "claw")
        for index, block in enumerate(self.blocks):
            # [2] -> [B, 2]: nominal initial XY for this fixed role.
            xy = np.repeat(
                np.asarray(B02_NOMINAL_INITIAL_LOCAL_XY[index])[None, :],
                batch_size,
                axis=0,
            )
            yaw = np.full(batch_size, B02_NOMINAL_INITIAL_YAW[index])
            if values is not None:
                xy[:, 0] += values[f"{role_names[index]}_x_offset_m"][indices]
                xy[:, 1] += values[f"{role_names[index]}_y_offset_m"][indices]
                yaw += values[f"{role_names[index]}_yaw_offset_rad"][indices]
            block.set_pose(
                self._local_planar_pose(
                    torch.as_tensor(
                        xy,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    torch.as_tensor(
                        yaw,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    0.002,
                )
            )
            zero = torch.zeros((batch_size, 3), device=self.device)
            block.set_linear_velocity(zero)
            block.set_angular_velocity(zero)

    def _reset_agents_to_rest(self, batch_size: int) -> None:
        base_poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(base_poses, list) or len(base_poses) != 2:
            raise RuntimeError("B02 requires exactly two robot base poses.")
        for agent, base_pose in zip(self._dual_agents(), base_poses, strict=True):
            rest = agent.keyframes.get("rest")
            if rest is None or rest.qpos is None:
                raise RuntimeError(
                    f"B02 agent {agent.uid!r} requires the rest keyframe."
                )
            # [N] -> [B, N]: repeat this hand's keyframe for every environment.
            qpos = (
                torch.as_tensor(
                    rest.qpos,
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1)
            )
            agent.reset(qpos)
            agent.robot.set_pose(base_pose)

    def _world_to_site(self, positions: Any) -> Any:
        site = self._site_pose(self.num_envs)
        inverse = site.inv()
        return (inverse * Pose.create_from_pq(p=positions)).p

    def _block_local_yaw_and_tilt(self, block: Any) -> tuple[Any, Any]:
        site_q = self._site_pose(self.num_envs).q
        inverse_site_q = site_q.clone()
        inverse_site_q[:, 1:] *= -1.0
        relative_q = (
            Pose.create_from_pq(q=inverse_site_q)
            * Pose.create_from_pq(q=block.pose.q)
        ).q
        w, x, y, z = relative_q.unbind(dim=1)
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        local_z_cosine = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
        tilt = torch.acos(local_z_cosine)
        return yaw, tilt

    @staticmethod
    def _periodic_error(value: Any, target: Any, period: float) -> Any:
        wrapped_error = torch.remainder(
            value - target + 0.5 * period,
            period,
        ) - 0.5 * period
        return torch.abs(wrapped_error)

    def _block_grasped(self, index: int) -> Any:
        override = getattr(self, "_state_injection_measurements", {}).get(
            f"{B02_BLOCK_NAMES[index]}_grasped"
        )
        if override is not None:
            return self._as_per_env_tensor(override, dtype=torch.bool)
        grasped_by_agent: list[Any] = []
        for agent, queries in zip(
            self._dual_agents(), self._grasp_queries, strict=True
        ):
            pair = agent.get_pairwise_contact_forces(queries[index])
            force = torch.linalg.norm(pair, dim=-1).amax(dim=2)
            contact = force >= self.SUCCESS_SPEC.grasp_contact_force_n
            grasped_by_agent.append(
                contact[:, 0] & (contact[:, 1:].sum(dim=1) >= 2)
            )
        # Two [B] role vectors -> [B, 2] -> [B]: either hand may move a piece.
        return torch.stack(grasped_by_agent, dim=1).any(dim=1)

    def _geometry_diagnostics(self) -> dict[str, Any]:
        # Three [B, 3] tensors -> [B, 3, 3] role-ordered local positions.
        local_positions = torch.stack(
            [self._world_to_site(block.pose.p) for block in self.blocks],
            dim=1,
        )
        yaws_and_tilts = [
            self._block_local_yaw_and_tilt(block) for block in self.blocks
        ]
        yaws = torch.stack([item[0] for item in yaws_and_tilts], dim=1)
        tilts = torch.stack([item[1] for item in yaws_and_tilts], dim=1)
        xy_error = torch.linalg.norm(
            local_positions[:, :, :2] - self._target_local_xy,
            dim=2,
        )
        # Three [B] errors -> [B, 3] symmetry-aware yaw errors.
        yaw_error = torch.stack(
            [
                self._periodic_error(
                    yaws[:, index],
                    self._target_local_yaw[:, index],
                    B02_YAW_SYMMETRY_PERIOD[index],
                )
                for index in range(3)
            ],
            dim=1,
        )
        height_error = torch.abs(local_positions[:, :, 2] - 0.002)
        released = ~torch.stack(
            [self._block_grasped(index) for index in range(3)],
            dim=1,
        )
        correct = (
            (xy_error <= self.SUCCESS_SPEC.xy_tolerance_m)
            & (yaw_error <= self.SUCCESS_SPEC.yaw_tolerance_rad)
            & (tilts <= self.SUCCESS_SPEC.tilt_tolerance_rad)
            & (height_error <= self.SUCCESS_SPEC.height_tolerance_m)
            & released
        )
        return {
            "local_positions": local_positions,
            "local_yaws": yaws,
            "tilts": tilts,
            "xy_error": xy_error,
            "yaw_error": yaw_error,
            "height_error": height_error,
            "released": released,
            "correct": correct,
        }

    def _adjacency_hits(self, diagnostics: dict[str, Any]) -> Any:
        positions = diagnostics["local_positions"][:, :, :2]
        yaws = diagnostics["local_yaws"]
        expected_distances = (0.0956, 0.0954)
        pairs = ((0, 1), (0, 2))
        hits: list[Any] = []
        for pair_index, (first, second) in enumerate(pairs):
            center_distance = torch.linalg.norm(
                positions[:, first] - positions[:, second],
                dim=1,
            )
            distance_error = torch.abs(
                center_distance - expected_distances[pair_index]
            )
            distance_ok = (
                distance_error
                <= self.SUCCESS_SPEC.adjacency_center_tolerance_m
            )
            first_angle = self._periodic_error(
                yaws[:, first],
                self._target_local_yaw[:, first],
                B02_YAW_SYMMETRY_PERIOD[first],
            )
            second_angle = self._periodic_error(
                yaws[:, second],
                self._target_local_yaw[:, second],
                B02_YAW_SYMMETRY_PERIOD[second],
            )
            # Nominal shared edges overlap by 31.6 mm, above the 20 mm minimum.
            overlap_ok = torch.full_like(
                distance_ok,
                0.0316 >= self.SUCCESS_SPEC.adjacency_min_overlap_m,
            )
            hits.append(
                distance_ok
                & overlap_ok
                & (
                    first_angle
                    <= self.SUCCESS_SPEC.adjacency_angle_tolerance_rad
                )
                & (
                    second_angle
                    <= self.SUCCESS_SPEC.adjacency_angle_tolerance_rad
                )
            )
        return torch.stack(hits, dim=1)

    def _piece_mask(self, center_xy: Any, yaw: Any, block_index: int) -> Any:
        grid = self._mask_grid_xy[None, :, :, :] - center_xy[:, None, None, :]
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        local_x = (
            cosine[:, None, None] * grid[..., 0]
            + sine[:, None, None] * grid[..., 1]
        )
        local_y = (
            -sine[:, None, None] * grid[..., 0]
            + cosine[:, None, None] * grid[..., 1]
        )
        half_x, half_y = (
            0.5 * value for value in B02_FOOTPRINT_SIZE_XY[block_index]
        )
        if block_index < 2:
            return (torch.abs(local_x) <= half_x) & (torch.abs(local_y) <= half_y)
        # The corner GLB occupies the upper two quadrants and lower-right
        # quadrant of its 2x2 footprint.
        horizontal = (
            (torch.abs(local_x) <= half_x)
            & (local_y >= 0.0)
            & (local_y <= half_y)
        )
        vertical = (
            (local_x >= 0.0)
            & (local_x <= half_x)
            & (local_y >= -half_y)
            & (local_y <= 0.0)
        )
        return horizontal | vertical

    def _footprint_iou(self, diagnostics: dict[str, Any]) -> Any:
        predicted = torch.zeros(
            (self.num_envs, B02_MASK_SIZE, B02_MASK_SIZE),
            dtype=torch.bool,
            device=self.device,
        )
        target = torch.zeros_like(predicted)
        for index in range(3):
            predicted |= self._piece_mask(
                diagnostics["local_positions"][:, index, :2],
                diagnostics["local_yaws"][:, index],
                index,
            )
            target |= self._piece_mask(
                self._target_local_xy[:, index],
                self._target_local_yaw[:, index],
                index,
            )
        intersection = (predicted & target).sum(dim=(1, 2)).to(torch.float32)
        union = (predicted | target).sum(dim=(1, 2)).clamp_min(1).to(torch.float32)
        return intersection / union

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        diagnostics = self._geometry_diagnostics()
        correct = diagnostics["correct"]
        self._block_was_correct |= correct
        grasped = ~diagnostics["released"]
        self._block_was_grasped |= grasped
        adjacency = self._adjacency_hits(diagnostics)
        iou = self._footprint_iou(diagnostics)
        # Three [B] speed vectors -> [B, 3] role-ordered speed magnitudes.
        linear_speed = torch.stack(
            [
                torch.linalg.norm(block.linear_velocity, dim=1)
                for block in self.blocks
            ],
            dim=1,
        )
        angular_speed = torch.stack(
            [
                torch.linalg.norm(block.angular_velocity, dim=1)
                for block in self.blocks
            ],
            dim=1,
        )
        static = (
            (linear_speed <= self.SUCCESS_SPEC.max_linear_speed_mps)
            & (angular_speed <= self.SUCCESS_SPEC.max_angular_speed_rad_s)
        )
        complete_now = (
            correct.all(dim=1)
            & diagnostics["released"].all(dim=1)
            & static.all(dim=1)
            & adjacency.all(dim=1)
        )
        self._completion_steps_task = torch.where(
            complete_now,
            self._completion_steps_task + 1,
            torch.zeros_like(self._completion_steps_task),
        )
        success = complete_now & (
            self._completion_steps_task >= self._completion_dwell_steps
        )
        success = self._latch_task_success(success)
        correct_count = correct.sum(dim=1)
        adjacent_correct = adjacency & torch.stack(
            (correct[:, 0] & correct[:, 1], correct[:, 0] & correct[:, 2]), dim=1
        )
        adjacency_count = adjacency.sum(dim=1)
        self._latest_success_diagnostics = {
            "task_id": "b02_v1",
            "piece_names": B02_BLOCK_NAMES,
            "xy_error_m": diagnostics["xy_error"],
            "yaw_error_rad": diagnostics["yaw_error"],
            "tilt_error_rad": diagnostics["tilts"],
            "height_error_m": diagnostics["height_error"],
            "released": diagnostics["released"],
            "linear_speed_mps": linear_speed,
            "angular_speed_rad_s": angular_speed,
            "static": static,
            "correct": correct,
            "adjacency": adjacency,
            "footprint_iou": iou,
            "footprint_iou_quality_target": (
                self.SUCCESS_SPEC.footprint_iou_quality_target
            ),
            "complete_now": complete_now,
            "dwell_steps": self._completion_steps_task,
            "dwell_required_steps": self._completion_dwell_steps,
            "thresholds": {
                "xy_error_m": self.SUCCESS_SPEC.xy_tolerance_m,
                "yaw_error_rad": self.SUCCESS_SPEC.yaw_tolerance_rad,
                "tilt_error_rad": self.SUCCESS_SPEC.tilt_tolerance_rad,
                "height_error_m": self.SUCCESS_SPEC.height_tolerance_m,
                "linear_speed_mps": self.SUCCESS_SPEC.max_linear_speed_mps,
                "angular_speed_rad_s": self.SUCCESS_SPEC.max_angular_speed_rad_s,
            },
        }
        return {
            "success": success,
            "one_block_correct": correct_count >= 1,
            "two_blocks_adjacent": (correct_count >= 2) & adjacent_correct.any(dim=1),
            "three_blocks_correct": correct_count == 3,
            "hammer_stably_complete": success,
            "block_xy_error_m": diagnostics["xy_error"],
            "block_yaw_error_rad": diagnostics["yaw_error"],
            "block_tilt_error_rad": diagnostics["tilts"],
            "block_height_error_m": diagnostics["height_error"],
            "block_linear_speed_mps": linear_speed,
            "block_angular_speed_rad_s": angular_speed,
            "correct_block_count": correct_count,
            "adjacency_hit_count": adjacency_count,
            "adjacency_hit": adjacency,
            "footprint_iou": iou,
            "block_correct": correct,
            "block_released": diagnostics["released"],
            "block_static": static,
            "complete_now": complete_now,
            "completion_dwell_steps": self._completion_steps_task,
        }

    def get_task_success_diagnostics(self) -> dict[str, Any] | None:
        """Return the latest B02 success gates for low-rate operator logging."""

        diagnostics = getattr(self, "_latest_success_diagnostics", None)
        return None if diagnostics is None else dict(diagnostics)
