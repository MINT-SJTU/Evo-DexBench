from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.envs.base_env import DexVLABenchBaseEnv
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
)
from dex_benchmark.sim.layout_types import LayoutId
from dex_benchmark.sim.scenes.layouts import LayoutBuilder
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


M08_SETTINGS = TaskRegistry.register(TaskRegistry.load("m08_v1"))


@dataclass(frozen=True)
class M08SuccessSpec:
    near_target_tolerance_rad: float = math.radians(20.0)
    aligned_tolerance_rad: float = math.radians(6.0)
    aligned_max_joint_speed_rad_s: float = 0.20
    contact_force_n: float = 0.40
    max_machine_displacement_m: float = 0.025
    max_machine_tilt_rad: float = math.radians(12.0)
    max_machine_linear_speed_m_s: float = 0.05
    max_machine_angular_speed_rad_s: float = 0.60


@dataclass(frozen=True)
class M08InteractionSpec:
    min_rotation_speed_rad_s: float = 0.03


M08_SUCCESS_SPEC = M08SuccessSpec()
M08_INTERACTION_SPEC = M08InteractionSpec()

# Layout contract. The Kitchen layout should expose the existing/custom coffee
# machine articulation under this entity name. Change only this constant if the
# layout uses another stable entity name.
M08_COFFEE_MACHINE_ENTITY_NAME = "coffee_machine"
M08_INTERACTION_SITE_ID = "coffee_machine"

# If the custom CoffeeMachine asset has a known stable knob joint name, put it
# here (for example "joint_0"). None means: first prefer a rotational joint whose
# joint/child-link name contains "knob"; if none exists, accept the articulation
# only when it has exactly one rotational active joint.
M08_TARGET_KNOB_JOINT_NAME = "joint_0"

M08_ALLOWED_KNOB_JOINT_TYPES = {
    "revolute",
    "revolute_unwrapped",
    "continuous",
}

# Task-level detents. These deliberately do NOT depend on joint limits because
# custom/PartNet knob joints may be revolute_unwrapped with [-inf, inf] limits.
# Calibrate this one tuple to the CoffeeMachine indicator geometry if needed:
#   index 0 = off, index 1 = first gear, index 2 = second gear.
M08_DETENT_ANGLES_RAD = (
    0.0,
    math.radians(45.0),
    math.radians(90.0),
)
M08_TARGET_DETENT_INDEX = 2
M08_NOMINAL_INITIAL_DETENT_INDEX = 0
M08_SPATIAL_INITIAL_DETENT_INDICES = (0, 1)
M08_SPATIAL_TARGET_DETENT_INDICES = (1, 2)

# PartNet 103041 is loaded at scale 0.4. Its joint_0 knob rotates about local
# +Z and the visible dial face is centered near local Z=0.562 before scaling.
# These visual-only markers are collision-free and are positioned relative to
# the live knob link so machine pose randomization and joint motion stay synced.
M08_KNOB_VISUAL_CENTER_LINK_LOCAL_Z_M = 0.562 * 0.4
M08_TARGET_INDICATOR_RING_RADIUS_M = 0.038
M08_TARGET_INDICATOR_SURFACE_OFFSET_M = 0.006
M08_TARGET_INDICATOR_DETENT_RADIUS_M = 0.0035
M08_TARGET_INDICATOR_HIGHLIGHT_RADIUS_M = 0.006
M08_TARGET_INDICATOR_HIDDEN_Z_M = -10.0

# Three high-contrast, collision-free scale lines share one visual actor whose
# batched pose tracks the rotating knob link. At the initial knob pose they are
# top/right/bottom, so a 90-degree counterclockwise knob rotation carries the
# middle (90-degree) line from the right side to the top.
M08_KNOB_SCALE_LINE_ANGLES_RAD = (
    0.0,
    math.radians(90.0),
    math.radians(180.0),
)
M08_KNOB_SCALE_ZERO_DIRECTION_RAD = math.radians(90.0)
M08_KNOB_SCALE_LINE_CENTER_RADIUS_M = 0.0195
M08_KNOB_SCALE_LINE_HALF_LENGTH_M = 0.007
M08_KNOB_SCALE_LINE_HALF_WIDTH_M = 0.0022
M08_KNOB_SCALE_LINE_HALF_THICKNESS_M = 0.00025
M08_KNOB_SCALE_LINE_SURFACE_OFFSET_M = 0.001
# Calibrated from PartNet 103041's knob meshes at scale 0.4: the dark inner
# face ends near radius 0.02687 m, while the light outer ring spans roughly
# 0.02866--0.03102 m. Keep the ticks inside the former and the arc on the latter.
M08_KNOB_SCALE_ARC_RADIUS_M = 0.0298
M08_KNOB_SCALE_ARC_HALF_WIDTH_M = 0.0011
M08_KNOB_SCALE_ARC_SURFACE_OFFSET_M = 0.0015
M08_KNOB_SCALE_ARC_SAMPLE_COUNT = 65
M08_KNOB_SCALE_ARC_ANGLES_RAD = tuple(
    -0.5 * math.pi
    + sample_index * math.pi / (M08_KNOB_SCALE_ARC_SAMPLE_COUNT - 1)
    for sample_index in range(M08_KNOB_SCALE_ARC_SAMPLE_COUNT)
)

M08_SPATIAL_PARAMETER_NAMES = (
    "knob_initial_selector",
    "target_detent_selector",
    "machine_x_offset_m",
    "machine_y_offset_m",
    "machine_yaw_offset_rad",
)
M08_PHYSICAL_PARAMETER_NAMES = (
    "knob_joint_friction",
    "knob_joint_damping",
)
M08_KNOB_JOINT_FRICTION = 0.15
M08_KNOB_JOINT_DAMPING = 0.50
M08_KNOB_JOINT_FORCE_LIMIT = 1e10
M08_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("knob_initial_selector", 0.0, 1.0),
            GeneralizationParameterSpec("target_detent_selector", 0.0, 1.0),
            GeneralizationParameterSpec("machine_x_offset_m", -0.015, 0.015),
            GeneralizationParameterSpec("machine_y_offset_m", -0.015, 0.015),
            GeneralizationParameterSpec(
                "machine_yaw_offset_rad",
                -math.radians(3.0),
                math.radians(3.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("knob_joint_friction", 0.10, 0.20),
            GeneralizationParameterSpec("knob_joint_damping", 0.35, 0.65),
        ),
        requires_rebuild=True,
    ),
)



@register_env(M08_SETTINGS.env_id, max_episode_steps=M08_SETTINGS.max_steps)
class M08Env(DexVLABenchBaseEnv):
    """Use one hand to rotate the fixed coffee-machine knob to its target."""

    TASK_SETTINGS = M08_SETTINGS
    SUCCESS_SPEC = M08_SUCCESS_SPEC
    INTERACTION_SPEC = M08_INTERACTION_SPEC
    GENERALIZATION_SPEC = M08_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = M08_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.344805, -0.834866, 1.479210),
            quaternion_world_wxyz=(0.834029, -0.012572, 0.019020, 0.551249),
        ),
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(0.648710, -3.126952, 1.449574),
            quaternion_world_wxyz=(0.227258, -0.027797, 0.006490, 0.973416),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(0.805745, -3.377407, 1.486519),
            quaternion_world_wxyz=(0.284653, -0.036224, 0.010764, 0.957886),
        ),
    }

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
            self.interaction_site_robot_base_sapien_pose(),
        )

    @staticmethod
    def _joint_type_values(joint: Any) -> tuple[str, ...]:
        raw_types = joint.type
        if isinstance(raw_types, (tuple, list)):
            return tuple(str(value) for value in raw_types)
        return (str(raw_types),)

    @classmethod
    def _is_rotational_joint(cls, joint: Any) -> bool:
        return all(
            joint_type in M08_ALLOWED_KNOB_JOINT_TYPES
            for joint_type in cls._joint_type_values(joint)
        )

    def _select_target_knob_joint(self) -> Any:
        """Select and validate the CoffeeMachine knob joint."""

        active_joints = self.coffee_machine.active_joints_map
        known_joint_names = tuple(active_joints)

        if M08_TARGET_KNOB_JOINT_NAME is not None:
            try:
                joint = active_joints[M08_TARGET_KNOB_JOINT_NAME]
            except KeyError as exc:
                raise RuntimeError(
                    f"CoffeeMachine does not expose configured knob joint "
                    f"{M08_TARGET_KNOB_JOINT_NAME!r}. Known active joints: "
                    f"{known_joint_names}."
                ) from exc
            candidates = (joint,)
        else:
            semantic_candidates = tuple(
                joint
                for joint_name, joint in active_joints.items()
                if self._is_rotational_joint(joint)
                and (
                    "knob" in joint_name.lower()
                    or "knob"
                    in getattr(getattr(joint, "child_link", None), "name", "").lower()
                )
            )

            if len(semantic_candidates) == 1:
                candidates = semantic_candidates
            elif len(semantic_candidates) > 1:
                raise RuntimeError(
                    "CoffeeMachine exposes multiple knob-like rotational joints. "
                    "Set M08_TARGET_KNOB_JOINT_NAME to the desired one. "
                    f"Candidates: {[joint.name for joint in semantic_candidates]}."
                )
            else:
                rotational_candidates = tuple(
                    joint
                    for joint in active_joints.values()
                    if self._is_rotational_joint(joint)
                )
                if len(rotational_candidates) != 1:
                    raise RuntimeError(
                        "Could not uniquely identify the CoffeeMachine knob joint. "
                        "Set M08_TARGET_KNOB_JOINT_NAME explicitly. "
                        f"Known active joints: {known_joint_names}; rotational "
                        f"candidates: {[joint.name for joint in rotational_candidates]}."
                    )
                candidates = rotational_candidates

        target_joint = candidates[0]
        if not self._is_rotational_joint(target_joint):
            raise RuntimeError(
                f"CoffeeMachine knob joint {target_joint.name!r} must be rotational, "
                f"got type {target_joint.type!r}."
            )

        child_link = getattr(target_joint, "child_link", None)
        if child_link is None:
            raise RuntimeError(
                f"CoffeeMachine knob joint {target_joint.name!r} has no child link."
            )

        return target_joint

    def _build_target_detent_indicators(self) -> None:
        """Build collision-free detent dots and a per-environment target dot."""

        if sapien is None:
            raise ImportError(
                "SAPIEN is required to build the M08 target detent indicator."
            )

        detent_material = sapien.render.RenderMaterial(
            base_color=[0.25, 0.25, 0.25, 0.95],
            roughness=0.7,
        )
        target_material = sapien.render.RenderMaterial(
            base_color=[1.00, 0.15, 0.05, 1.00],
            emission=[3.0, 0.20, 0.05, 1.0],
            roughness=0.5,
        )
        hidden_pose = sapien.Pose(p=[0.0, 0.0, M08_TARGET_INDICATOR_HIDDEN_Z_M])

        detent_markers: list[Any] = []
        for detent_index in range(len(M08_DETENT_ANGLES_RAD)):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=M08_TARGET_INDICATOR_DETENT_RADIUS_M,
                material=detent_material,
            )
            builder.initial_pose = hidden_pose
            detent_markers.append(
                builder.build_kinematic(
                    name=f"m08_detent_marker_{detent_index}"
                )
            )
        self.detent_markers = tuple(detent_markers)

        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(
            radius=M08_TARGET_INDICATOR_HIGHLIGHT_RADIUS_M,
            material=target_material,
        )
        builder.initial_pose = hidden_pose
        self.target_detent_marker = builder.build_kinematic(
            name="m08_target_detent_marker"
        )

    @staticmethod
    def _knob_scale_line_local_pose(
        scale_angle_rad: float,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        """Return one radial scale line pose in the rotating knob frame."""

        line_angle_rad = M08_KNOB_SCALE_ZERO_DIRECTION_RAD - scale_angle_rad
        position_xyz = (
            M08_KNOB_SCALE_LINE_CENTER_RADIUS_M * math.cos(line_angle_rad),
            M08_KNOB_SCALE_LINE_CENTER_RADIUS_M * math.sin(line_angle_rad),
            M08_KNOB_VISUAL_CENTER_LINK_LOCAL_Z_M
            + M08_KNOB_SCALE_LINE_SURFACE_OFFSET_M,
        )
        half_angle_rad = 0.5 * line_angle_rad
        quaternion_wxyz = (
            math.cos(half_angle_rad),
            0.0,
            0.0,
            math.sin(half_angle_rad),
        )
        return position_xyz, quaternion_wxyz

    @staticmethod
    def _knob_scale_arc_mesh(
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a continuous right-half annulus mesh in the knob frame."""

        angles_rad = np.asarray(M08_KNOB_SCALE_ARC_ANGLES_RAD, dtype=np.float32)
        inner_radius_m = (
            M08_KNOB_SCALE_ARC_RADIUS_M - M08_KNOB_SCALE_ARC_HALF_WIDTH_M
        )
        outer_radius_m = (
            M08_KNOB_SCALE_ARC_RADIUS_M + M08_KNOB_SCALE_ARC_HALF_WIDTH_M
        )
        surface_z_m = (
            M08_KNOB_VISUAL_CENTER_LINK_LOCAL_Z_M
            + M08_KNOB_SCALE_ARC_SURFACE_OFFSET_M
        )
        cosine = np.cos(angles_rad)
        sine = np.sin(angles_rad)
        inner_vertices = np.stack(
            (
                inner_radius_m * cosine,
                inner_radius_m * sine,
                np.full_like(cosine, surface_z_m),
            ),
            axis=1,
        )
        outer_vertices = np.stack(
            (
                outer_radius_m * cosine,
                outer_radius_m * sine,
                np.full_like(cosine, surface_z_m),
            ),
            axis=1,
        )
        # [N, 2, 3] -> [2N, 3], interleaving inner and outer vertices.
        vertices = np.stack(
            (inner_vertices, outer_vertices),
            axis=1,
        ).reshape(-1, 3)

        segment_indices = np.arange(len(angles_rad) - 1, dtype=np.uint32)
        inner = 2 * segment_indices
        outer = inner + 1
        next_inner = inner + 2
        next_outer = inner + 3
        # [N-1, 2, 3] -> [2(N-1), 3] counterclockwise triangles.
        triangles = np.stack(
            (
                np.stack((inner, outer, next_outer), axis=1),
                np.stack((inner, next_outer, next_inner), axis=1),
            ),
            axis=1,
        ).reshape(-1, 3)
        normals = np.zeros_like(vertices)
        normals[:, 2] = 1.0
        uvs = np.zeros((len(vertices), 2), dtype=np.float32)
        return vertices, triangles, normals, uvs

    def _build_knob_scale_visual_actor(self) -> None:
        """Build one collision-free actor containing the ticks and scale arc."""

        if sapien is None:
            raise ImportError("SAPIEN is required to build M08 knob scale lines.")

        line_material = sapien.render.RenderMaterial(
            base_color=[0.0, 0.0, 0.0, 1.0],
            emission=[0.0, 0.0, 0.0, 0.0],
            roughness=0.8,
        )
        line_half_size = (
            M08_KNOB_SCALE_LINE_HALF_LENGTH_M,
            M08_KNOB_SCALE_LINE_HALF_WIDTH_M,
            M08_KNOB_SCALE_LINE_HALF_THICKNESS_M,
        )
        builder = self.scene.create_actor_builder()
        for scale_index, scale_angle_rad in enumerate(
            M08_KNOB_SCALE_LINE_ANGLES_RAD
        ):
            position_xyz, quaternion_wxyz = self._knob_scale_line_local_pose(
                scale_angle_rad
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=position_xyz, q=quaternion_wxyz),
                half_size=line_half_size,
                material=line_material,
                name=f"m08_knob_scale_line_{scale_index}",
            )
        arc_vertices, arc_triangles, arc_normals, arc_uvs = (
            self._knob_scale_arc_mesh()
        )
        arc_shape = sapien.render.RenderShapeTriangleMesh(
            vertices=arc_vertices,
            triangles=arc_triangles,
            normals=arc_normals,
            uvs=arc_uvs,
            material=line_material,
        )
        arc_shape.set_front_face("clockwise")
        arc_shape.name = "m08_knob_scale_arc"
        procedural_shapes = getattr(builder, "_procedural_shapes", None)
        if not isinstance(procedural_shapes, list):
            raise RuntimeError(
                "M08 requires ManiSkill ActorBuilder procedural render shapes "
                "to build the continuous knob scale arc."
            )
        procedural_shapes.append(arc_shape)
        builder.initial_pose = sapien.Pose(
            p=[0.0, 0.0, M08_TARGET_INDICATOR_HIDDEN_Z_M]
        )
        self.knob_scale_visual_actor = builder.build_kinematic(
            name="m08_knob_scale_visual"
        )

    def _detent_indicator_positions_in_knob_frame(self) -> Any:
        """Return [B, 3, 3] marker positions in the rotating knob frame."""

        detent_angles = torch.as_tensor(
            M08_DETENT_ANGLES_RAD,
            device=self.device,
            dtype=torch.float32,
        )
        # Counter-rotate by the live joint angle so the world-space panel dots
        # remain fixed while the knob link turns: [3] - [B, 1] -> [B, 3].
        relative_angles = detent_angles[None, :] - self._knob_angle()[:, None]
        x = M08_TARGET_INDICATOR_RING_RADIUS_M * torch.cos(relative_angles)
        y = M08_TARGET_INDICATOR_RING_RADIUS_M * torch.sin(relative_angles)
        z = torch.full_like(
            x,
            M08_KNOB_VISUAL_CENTER_LINK_LOCAL_Z_M
            + M08_TARGET_INDICATOR_SURFACE_OFFSET_M,
        )
        return torch.stack((x, y, z), dim=-1)

    def _update_target_detent_indicators(self) -> None:
        """Synchronize the visual target with the current per-env detent."""

        if hasattr(self, "knob_scale_visual_actor"):
            # All three local box visuals share this single batched pose update.
            self.knob_scale_visual_actor.set_pose(self.knob_link.pose)
        if not hasattr(self, "target_detent_marker"):
            return
        local_positions = self._detent_indicator_positions_in_knob_frame()
        for detent_index, marker in enumerate(self.detent_markers):
            marker.set_pose(
                self.knob_link.pose
                * Pose.create_from_pq(p=local_positions[:, detent_index, :])
            )

        row_indices = torch.arange(self.num_envs, device=self.device)
        target_local_position = local_positions[
            row_indices,
            self._target_detent_indices,
        ]
        self.target_detent_marker.set_pose(
            self.knob_link.pose * Pose.create_from_pq(p=target_local_position)
        )

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self.current_layout_spec,
        ).build()

        self.coffee_machine = self.layout_scene.require_entity(
            M08_COFFEE_MACHINE_ENTITY_NAME
        )
        self._nominal_machine_root_position = (
            self.coffee_machine.root_pose.p.clone()
        )
        self._nominal_machine_root_quaternion = (
            self.coffee_machine.root_pose.q.clone()
        )
        self.knob_joint = self._select_target_knob_joint()
        self.knob_link = self.knob_joint.child_link
        self._build_knob_scale_visual_actor()
        self._build_target_detent_indicators()

        active_joint_names = tuple(self.coffee_machine.active_joints_map)
        self._knob_joint_qpos_index = active_joint_names.index(
            self.knob_joint.name
        )

        self._knob_tactile_target_groups = (
            ("knob", (self.knob_link,)),
        )
        self._machine_tactile_target_groups = (
            ("machine", tuple(self.coffee_machine.links)),
        )
        if not self.gpu_sim_enabled:
            self.agent.prepare_cpu_object_taxel_query(
                self._knob_tactile_target_groups
            )
            self.agent.prepare_cpu_object_taxel_query(
                self._machine_tactile_target_groups
            )

        self._configure_knob_joint_physics()

    def _configure_knob_joint_physics(self) -> None:
        joint_objects = tuple(getattr(self.knob_joint, "_objs", ()))
        if len(joint_objects) != self.num_envs:
            raise RuntimeError(
                "M08 joint configuration expected one underlying knob joint "
                f"per environment, got {len(joint_objects)} for {self.num_envs}."
            )

        for joint_object in joint_objects:
            joint_object.set_friction(M08_KNOB_JOINT_FRICTION)
            joint_object.set_drive_properties(
                stiffness=0.0,
                damping=M08_KNOB_JOINT_DAMPING,
                force_limit=M08_KNOB_JOINT_FORCE_LIMIT,
            )

        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return

        values = self._generalization_values(
            M08_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m08_physical_values",
        )
        for env_index, joint_object in enumerate(joint_objects):
            joint_object.set_friction(
                float(values["knob_joint_friction"][env_index])
            )
            joint_object.set_drive_properties(
                stiffness=0.0,
                damping=float(values["knob_joint_damping"][env_index]),
                force_limit=M08_KNOB_JOINT_FORCE_LIMIT,
            )

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_target_angles_rad"):
            return

        self._target_angles_rad = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._initial_angles_rad = torch.zeros_like(self._target_angles_rad)
        self._target_detent_indices = torch.full(
            (self.num_envs,),
            M08_TARGET_DETENT_INDEX,
            dtype=torch.long,
            device=self.device,
        )
        self._knob_contacted_once = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._knob_rotation_started = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._knob_aligned_once = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._protected_machine_initial_position = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._protected_machine_initial_up = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def _selector_to_initial_detent_index(selector: np.ndarray) -> np.ndarray:
        selector = np.clip(np.asarray(selector, dtype=np.float64), 0.0, 1.0)
        count = len(M08_SPATIAL_INITIAL_DETENT_INDICES)
        local_index = np.minimum(
            (selector * count).astype(np.int64),
            count - 1,
        )
        choices = np.asarray(M08_SPATIAL_INITIAL_DETENT_INDICES, dtype=np.int64)
        return choices[local_index]

    def _initial_detent_indices(self, env_idx: Any) -> Any:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return torch.full(
                (len(env_idx),),
                M08_NOMINAL_INITIAL_DETENT_INDEX,
                dtype=torch.long,
                device=self.device,
            )

        values = self._generalization_values(
            M08_SPATIAL_PARAMETER_NAMES,
            cache_name="_m08_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        indices = self._selector_to_initial_detent_index(
            values["knob_initial_selector"][value_indices]
        )
        return torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.device,
        )

    def _target_detent_indices_for_reset(self, env_idx: Any) -> Any:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return torch.full(
                (len(env_idx),),
                M08_TARGET_DETENT_INDEX,
                dtype=torch.long,
                device=self.device,
            )

        values = self._generalization_values(
            M08_SPATIAL_PARAMETER_NAMES,
            cache_name="_m08_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        indices = self._selector_to_target_detent_index(
            values["target_detent_selector"][value_indices]
        )
        return torch.as_tensor(
            indices,
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _selector_to_target_detent_index(selector: np.ndarray) -> np.ndarray:
        selector = np.clip(np.asarray(selector, dtype=np.float64), 0.0, 1.0)
        count = len(M08_SPATIAL_TARGET_DETENT_INDICES)
        selected_indices = np.minimum(
            (selector * count).astype(np.int64),
            count - 1,
        )
        choices = np.asarray(M08_SPATIAL_TARGET_DETENT_INDICES, dtype=np.int64)
        return choices[selected_indices]

    def _reset_knob_angle(self, env_idx: Any) -> None:
        """Reset the coffee knob and assign this episode's target detent."""

        env_idx = torch.as_tensor(
            env_idx,
            dtype=torch.long,
            device=self.device,
        ).reshape(-1)
        self._ensure_episode_state()

        detents = torch.as_tensor(
            M08_DETENT_ANGLES_RAD,
            dtype=torch.float32,
            device=self.device,
        )
        target_indices = self._target_detent_indices_for_reset(env_idx)
        initial_indices = torch.minimum(
            self._initial_detent_indices(env_idx),
            target_indices - 1,
        )
        initial_angles = detents[initial_indices]
        target_angles = detents[target_indices]

        self._initial_angles_rad[env_idx] = initial_angles
        self._target_angles_rad[env_idx] = target_angles
        self._target_detent_indices[env_idx] = target_indices

        qpos = self.coffee_machine.get_qpos().clone()
        qvel = self.coffee_machine.get_qvel().clone()
        qpos[env_idx, self._knob_joint_qpos_index] = initial_angles
        qvel[env_idx, self._knob_joint_qpos_index] = 0.0
        self.coffee_machine.set_qpos(qpos)
        self.coffee_machine.set_qvel(qvel)
        self._update_target_detent_indicators()

    def sample_instruction(self) -> str:
        target_indices = getattr(self, "_target_detent_indices", None)
        if target_indices is None:
            return self.TASK_SETTINGS.instruction
        unique_indices = torch.unique(target_indices).detach().cpu().tolist()
        if unique_indices == [1]:
            return "Turn the knob on the fixed coffee machine to setting one."
        if unique_indices == [2]:
            return "Turn the knob on the fixed coffee machine to setting two."
        return (
            "Turn the knob on each fixed coffee machine to its specified target "
            "setting."
        )

    @staticmethod
    def _quaternion_up_axis(quaternion_wxyz: Any) -> Any:
        quaternion = quaternion_wxyz / torch.linalg.norm(
            quaternion_wxyz,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(dim=-1)
        return torch.stack(
            [
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ],
            dim=-1,
        )

    def _machine_tilt_angle(self) -> Any:
        current_up = self._quaternion_up_axis(self.coffee_machine.root_pose.q)
        reference_up = self._protected_machine_initial_up
        cosine = torch.sum(current_up * reference_up, dim=1).clamp(-1.0, 1.0)
        return torch.acos(cosine)

    def _machine_linear_velocity(self) -> Any:
        for name in ("root_linear_velocity", "linear_velocity"):
            value = getattr(self.coffee_machine, name, None)
            if value is not None:
                return value
        raise TypeError("CoffeeMachine articulation exposes no linear velocity.")

    def _machine_angular_velocity(self) -> Any:
        for name in ("root_angular_velocity", "angular_velocity"):
            value = getattr(self.coffee_machine, name, None)
            if value is not None:
                return value
        raise TypeError("CoffeeMachine articulation exposes no angular velocity.")

    def _machine_root_displacement(self) -> Any:
        return torch.linalg.norm(
            self.coffee_machine.root_pose.p
            - self._protected_machine_initial_position,
            dim=1,
        )

    def _machine_geometrically_stable(self) -> Any:
        linear_speed = torch.linalg.norm(
            self._machine_linear_velocity(),
            dim=1,
        )
        angular_speed = torch.linalg.norm(
            self._machine_angular_velocity(),
            dim=1,
        )
        return (
            self._machine_root_displacement()
            <= self.SUCCESS_SPEC.max_machine_displacement_m
        ) & (
            self._machine_tilt_angle()
            <= self.SUCCESS_SPEC.max_machine_tilt_rad
        ) & (
            linear_speed
            <= self.SUCCESS_SPEC.max_machine_linear_speed_m_s
        ) & (
            angular_speed
            <= self.SUCCESS_SPEC.max_machine_angular_speed_rad_s
        )

    def _coffee_machine_stable(self) -> Any:
        return self._machine_geometrically_stable()

    def _set_robot_to_rest_keyframe(self, batch_size: int) -> None:
        rest_keyframe = self.agent.keyframes.get("rest")
        if rest_keyframe is None or rest_keyframe.qpos is None:
            raise RuntimeError(
                f"Agent {self.agent.uid!r} must define a rest keyframe with qpos."
            )

        rest_qpos = torch.as_tensor(
            rest_keyframe.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        expected_dof = self.agent.robot.max_dof
        if rest_qpos.shape != (expected_dof,):
            raise ValueError(
                f"Agent {self.agent.uid!r} rest qpos must have shape "
                f"({expected_dof},), got {tuple(rest_qpos.shape)}."
            )

        batched_rest_qpos = rest_qpos.unsqueeze(0).repeat(batch_size, 1)
        self.agent.reset(batched_rest_qpos)
        self.agent.robot.set_pose(
            self.interaction_site_robot_base_sapien_pose()
        )

    def _apply_machine_spatial_generalization(self, env_idx: Any) -> None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return
        values = self._generalization_values(
            M08_SPATIAL_PARAMETER_NAMES,
            cache_name="_m08_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        local_position = np.stack(
            [
                values["machine_x_offset_m"][value_indices],
                values["machine_y_offset_m"][value_indices],
                np.zeros(len(value_indices), dtype=np.float64),
            ],
            axis=1,
        )
        yaw = values["machine_yaw_offset_rad"][value_indices]
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
            p=self._nominal_machine_root_position[env_idx],
            q=self._nominal_machine_root_quaternion[env_idx],
        )
        changed_pose = nominal_pose * Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
            device=self.device,
        )
        current_position = self.coffee_machine.root_pose.p.clone()
        current_quaternion = self.coffee_machine.root_pose.q.clone()
        current_position[env_idx] = changed_pose.p
        current_quaternion[env_idx] = changed_pose.q
        self.coffee_machine.set_pose(
            Pose.create_from_pq(
                p=current_position,
                q=current_quaternion,
            )
        )

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            self._invalidate_contact_force_cache()
            batch_size = len(env_idx)
            if batch_size <= 0:
                raise ValueError(
                    "Episode initialization requires at least one environment."
                )

            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )

            self._ensure_episode_state()
            self._apply_machine_spatial_generalization(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)

            self._reset_knob_angle(env_idx)
            self._set_robot_to_rest_keyframe(batch_size)

            self._knob_contacted_once[env_idx] = False
            self._knob_rotation_started[env_idx] = False
            self._knob_aligned_once[env_idx] = False
            self._protected_machine_initial_position[env_idx] = (
                self.coffee_machine.root_pose.p[env_idx]
            )
            self._protected_machine_initial_up[env_idx] = (
                self._quaternion_up_axis(self.coffee_machine.root_pose.q)[env_idx]
            )

    def _knob_angle(self) -> Any:
        return self.knob_joint.qpos.reshape(self.num_envs, -1)[:, 0]

    def _knob_angular_speed(self) -> Any:
        return self.knob_joint.qvel.reshape(self.num_envs, -1)[:, 0]

    def _signed_target_angle_error(self) -> Any:
        # Circular error is intentional: revolute_unwrapped knobs can accumulate
        # turns even though the visual indicator repeats every 2*pi.
        delta = self._knob_angle() - self._target_angles_rad
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    def _target_angle_error(self) -> Any:
        return torch.abs(self._signed_target_angle_error())

    def _knob_near_target(self) -> Any:
        return (
            self._target_angle_error()
            <= self.SUCCESS_SPEC.near_target_tolerance_rad
        )

    def _knob_aligned(self) -> Any:
        return (
            self._target_angle_error()
            <= self.SUCCESS_SPEC.aligned_tolerance_rad
        ) & (
            torch.abs(self._knob_angular_speed())
            <= self.SUCCESS_SPEC.aligned_max_joint_speed_rad_s
        )

    def _invalidate_contact_force_cache(self) -> None:
        self._cached_knob_contact: Any | None = None
        self._cached_max_machine_force_norm: Any | None = None

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self._invalidate_contact_force_cache()

    def _populate_tactile_force_cache(self) -> None:
        knob_forces = self.agent.get_object_taxel_world_forces(
            self._knob_tactile_target_groups
        )["knob"]
        machine_forces = self.agent.get_object_taxel_world_forces(
            self._machine_tactile_target_groups
        )["machine"]
        knob_norms = torch.linalg.norm(
            self.agent.aggregate_taxel_world_forces_by_parent(knob_forces),
            dim=-1,
        )
        machine_norms = torch.linalg.norm(
            self.agent.aggregate_taxel_world_forces_by_parent(machine_forces),
            dim=-1,
        )
        self._cached_knob_contact = (
            knob_norms >= self.SUCCESS_SPEC.contact_force_n
        ).any(dim=1)
        self._cached_max_machine_force_norm = machine_norms.amax(dim=1)

    def _max_machine_tactile_force(self) -> Any:
        if self._cached_max_machine_force_norm is None:
            self._populate_tactile_force_cache()
        return self._cached_max_machine_force_norm

    def _knob_has_tactile_contact(self) -> Any:
        if self._cached_knob_contact is None:
            self._populate_tactile_force_cache()
        return self._cached_knob_contact

    def _knob_rotating_toward_target(self) -> Any:
        desired_delta = self._target_angles_rad - self._initial_angles_rad
        desired_delta = torch.atan2(
            torch.sin(desired_delta),
            torch.cos(desired_delta),
        )
        desired_direction = torch.sign(desired_delta)
        return (
            self._knob_angular_speed() * desired_direction
            >= self.INTERACTION_SPEC.min_rotation_speed_rad_s
        )

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis is GeneralizationAxis.SPATIAL:
            return (
                self._target_angle_error()
                > self.SUCCESS_SPEC.near_target_tolerance_rad
            ) & self._machine_geometrically_stable()

        if axis is GeneralizationAxis.PHYSICAL:
            return (
                torch.abs(self._knob_angular_speed()) <= 0.05
            ) & self._machine_geometrically_stable()

        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()

        geometric_stable = self._machine_geometrically_stable()
        knob_contacted = self._knob_has_tactile_contact()
        coffee_machine_stable = self._coffee_machine_stable()

        knob_near_target = self._knob_near_target()
        knob_aligned = self._knob_aligned()
        knob_rotating_correctly = self._knob_rotating_toward_target()

        rotation_started_now = (
            geometric_stable
            & knob_contacted
            & knob_rotating_correctly
            & ~self._knob_aligned_once
        )
        self._knob_rotation_started |= rotation_started_now

        self._knob_contacted_once |= knob_contacted
        aligned_after_valid_rotation = knob_aligned & self._knob_rotation_started
        self._knob_aligned_once |= aligned_after_valid_rotation

        # Final state remains independent from the interaction history: a stable
        # machine with a slowly aligned knob is successful.
        success = coffee_machine_stable & knob_aligned
        self._update_target_detent_indicators()

        return {
            "success": success,
            "coffee_machine_stable": coffee_machine_stable,
            "coffee_knob_contacted": knob_contacted,
            "coffee_knob_rotated_correctly": self._knob_rotation_started,
            "coffee_knob_near_target": knob_near_target,
            "coffee_knob_aligned": knob_aligned,
            "coffee_target_detent_index": self._target_detent_indices,
        }
