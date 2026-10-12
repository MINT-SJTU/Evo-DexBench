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
from dex_benchmark.sim.scenes.layouts import ArticulatedAssetSpec, LayoutBuilder
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


M10_SETTINGS = TaskRegistry.register(TaskRegistry.load("m10_v1"))


@dataclass(frozen=True)
class M10SuccessSpec:
    near_target_tolerance_rad: float = math.radians(45.0)
    aligned_tolerance_rad: float = math.radians(9.0)
    aligned_max_joint_speed_rad_s: float = 1.00
    contact_force_n: float = 0.40


@dataclass(frozen=True)
class M10InteractionSpec:
    min_rotation_speed_rad_s: float = 0.03


M10_SUCCESS_SPEC = M10SuccessSpec()
M10_INTERACTION_SPEC = M10InteractionSpec()

# The existing kitchens currently use the shared counter interaction site for
# robot placement and camera routing. The dishwasher itself is discovered from the
# RoboCasa external scene and is never created or moved by this task.
M10_INTERACTION_SITE_ID = "dishwasher"

# Five task-level joint detents around RoboCasa's nominal off angle (0 rad).
# Positive joint qpos rotates the whole knob counterclockwise when viewed from
# the front. RoboCasa dishwasher knobs are normally unwrapped revolute joints,
# so circular angle error is used during evaluation.
M10_DETENT_ANGLES_RAD = (
    0.0,
    math.radians(45.0),
    math.radians(135.0),
    math.radians(-135.0),
    math.radians(-45.0),
)

M10_NOMINAL_INITIAL_DETENT_INDEX = 0
M10_NOMINAL_TARGET_DETENT_INDEX = 2

M10_SPATIAL_PARAMETER_NAMES = (
    "knob_initial_selector",
    "target_detent_selector",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
M10_PHYSICAL_PARAMETER_NAMES = (
    "knob_joint_friction",
    "knob_joint_damping",
)
M10_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("knob_initial_selector", 0.0, 1.0),
            GeneralizationParameterSpec("target_detent_selector", 0.0, 1.0),
            GeneralizationParameterSpec("robot_forward_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec("robot_side_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec(
                "robot_yaw_offset_rad",
                -math.radians(2.0),
                math.radians(2.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("knob_joint_friction", 0.05, 0.15),
            GeneralizationParameterSpec("knob_joint_damping", 0.12, 0.28),
        ),
        requires_rebuild=True,
    ),
)

# Temporary robot parking height used while the RoboCasa scene settles.
M10_HIDDEN_Z_M = -20.0

M10_KNOB_JOINT_FRICTION = 0.08
M10_KNOB_JOINT_DAMPING = 0.20
M10_KNOB_JOINT_FORCE_LIMIT = 1e10
M10_NON_TARGET_DISHWASHER_JOINT_NAMES = (
    "joint_0",
    "joint_1",
    "joint_2",
)

# Visual-only scale calibration in the rotating knob-link frame. The PartNet
# knob already has one long 0-degree line. These five short ticks add the
# remaining requested scale positions without adding collision geometry. The
# display angle is clockwise-positive, opposite the joint's rotation direction.
M10_KNOB_SCALE_TICK_ANGLES_RAD = (
    math.radians(-90.0),
    math.radians(-45.0),
    math.radians(45.0),
    math.radians(90.0),
    math.radians(135.0),
)
M10_KNOB_SCALE_TICK_CENTER_RADIUS_M = 0.0415
M10_KNOB_SCALE_TICK_HALF_LENGTH_M = 0.0065
M10_KNOB_SCALE_TICK_HALF_WIDTH_M = 0.0015
M10_KNOB_SCALE_TICK_HALF_THICKNESS_M = 0.00035
M10_KNOB_SCALE_TICK_SURFACE_OFFSET_M = 0.0025

M10_ALLOWED_KNOB_JOINT_TYPES = {
    "revolute",
    "revolute_unwrapped",
}


@register_env(M10_SETTINGS.env_id, max_episode_steps=M10_SETTINGS.max_steps)
class M10Env(DexVLABenchBaseEnv):
    """Rotate the leftmost knob on the existing RoboCasa dishwasher to a target."""

    TASK_SETTINGS = M10_SETTINGS
    SUCCESS_SPEC = M10_SUCCESS_SPEC
    INTERACTION_SPEC = M10_INTERACTION_SPEC
    GENERALIZATION_SPEC = M10_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = M10_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.309915, -4.459677, 1.343648),
            quaternion_world_wxyz=(0.262165, -0.205508, 0.057246, 0.941147),
        ),
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.233572, -5.434585, 1.384331),
            quaternion_world_wxyz=(0.258977, -0.252290, 0.070278, 0.929700),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(0.896054, -1.596948, 0.913277),
            quaternion_world_wxyz=(0.263986, -0.134474, 0.037271, 0.954379),
        ),
    }

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        nominal_pose = self.interaction_site_robot_base_sapien_pose()
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_pose

        values = self._generalization_values(
            M10_SPATIAL_PARAMETER_NAMES,
            cache_name="_m10_spatial_values",
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
        return Pose.create(nominal_pose) * Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
        )

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    @staticmethod
    def _joint_type_values(joint: Any) -> tuple[str, ...]:
        raw_types = joint.type
        if isinstance(raw_types, (tuple, list)):
            return tuple(str(value) for value in raw_types)
        return (str(raw_types),)

    @classmethod
    def _is_rotational_knob_joint(
        cls,
        joint_name: str,
        joint: Any,
    ) -> bool:
        child_link = getattr(joint, "child_link", None)
        child_link_name = getattr(child_link, "name", "")

        knob_semantics = (
            "knob" in joint_name.lower()
            or "knob" in child_link_name.lower()
        )
        if not knob_semantics:
            return False

        return all(
            joint_type in M10_ALLOWED_KNOB_JOINT_TYPES
            for joint_type in cls._joint_type_values(joint)
        )


    def _dishwasher_knob_joints(self) -> tuple[Any, ...]:
        """Return the four rotational knob joints on the selected dishwasher."""

        knob_joints = tuple(
            joint
            for joint_name, joint in self.dishwasher.active_joints_map.items()
            if self._is_rotational_knob_joint(joint_name, joint)
        )

        for joint in knob_joints:
            if getattr(joint, "child_link", None) is None:
                raise RuntimeError(
                    f"Dishwasher knob joint {joint.name!r} has no child link."
                )

        return knob_joints

    def _select_target_knob_joint(self) -> Any:
        """Select dishwasher joint_3 as the target knob joint."""

        target_joint_name = "joint_3"
        active_joints = self.dishwasher.active_joints_map

        try:
            target_joint = active_joints[target_joint_name]
        except KeyError as exc:
            known_joint_names = tuple(active_joints)

            raise RuntimeError(
                f"Dishwasher does not expose target knob joint "
                f"{target_joint_name!r}. Known active joints: "
                f"{known_joint_names}."
            ) from exc

        raw_joint_types = target_joint.type
        if isinstance(raw_joint_types, (tuple, list)):
            joint_types = tuple(
                str(joint_type)
                for joint_type in raw_joint_types
            )
        else:
            joint_types = (str(raw_joint_types),)

        allowed_joint_types = {
            "revolute",
            "revolute_unwrapped",
            "continuous",
        }

        invalid_joint_types = [
            joint_type
            for joint_type in joint_types
            if joint_type not in allowed_joint_types
        ]

        if invalid_joint_types:
            raise RuntimeError(
                f"Dishwasher target knob joint "
                f"{target_joint_name!r} must be rotational, "
                f"got type {target_joint.type!r}."
            )

        child_link = getattr(target_joint, "child_link", None)
        if child_link is None:
            raise RuntimeError(
                f"Dishwasher target knob joint "
                f"{target_joint_name!r} has no child link."
            )

        print(
            "Selected dishwasher knob:",
            f"joint={target_joint.name!r},",
            f"type={target_joint.type!r},",
            f"child_link={child_link.name!r},",
            f"limits={target_joint.limits}",
        )

        return target_joint


    def _knob_detent_angles(self) -> Any:
        """Return fixed task detent angles as [B, D].

        The selected dishwasher knob is revolute_unwrapped and therefore
        exposes infinite limits. Its task detents must be explicitly defined.
        """

        detent_angles = torch.as_tensor(
            M10_DETENT_ANGLES_RAD,
            device=self.device,
            dtype=torch.float32,
        )

        return detent_angles.unsqueeze(0).expand(
            self.num_envs,
            -1,
        )

    @staticmethod
    def _selector_to_detent_indices(
        selectors: Any,
        detent_count: int,
    ) -> np.ndarray:
        """Map generalization selectors in [0, 1] to detent indices."""

        selectors = np.asarray(
            selectors,
            dtype=np.float64,
        )
        selectors = np.clip(
            selectors,
            0.0,
            1.0,
        )

        indices = np.floor(
            selectors * detent_count
        ).astype(np.int64)

        return np.minimum(
            indices,
            detent_count - 1,
        )

    @staticmethod
    def _circular_angle_difference(angle: Any, reference: Any) -> Any:
        """Return signed angle-reference wrapped to [-pi, pi]."""

        delta = angle - reference
        return torch.atan2(torch.sin(delta), torch.cos(delta))

    @classmethod
    def _circular_angle_distance(cls, angle: Any, reference: Any) -> Any:
        """Return the shortest absolute angular distance on the circle."""

        return torch.abs(cls._circular_angle_difference(angle, reference))

    def _reset_knob_angles(self, env_idx: Any,) -> None:
        """Reset initial knob angles and assign target detents."""

        env_idx = torch.as_tensor(
            env_idx,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

        self._ensure_episode_state()

        detent_angles = self._knob_detent_angles()[
            env_idx
        ]
        # [b, 5]

        detent_count = detent_angles.shape[1]

        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M10_SPATIAL_PARAMETER_NAMES,
                cache_name="_m10_spatial_values",
            )

            value_indices = (
                env_idx.detach().cpu().numpy()
            )

            initial_indices = torch.as_tensor(
                self._selector_to_detent_indices(
                    values["knob_initial_selector"][
                        value_indices
                    ],
                    detent_count,
                ),
                device=self.device,
                dtype=torch.long,
            )

            target_indices = torch.as_tensor(
                self._selector_to_detent_indices(
                    values["target_detent_selector"][
                        value_indices
                    ],
                    detent_count,
                ),
                device=self.device,
                dtype=torch.long,
            )
        else:
            initial_indices = torch.full(
                (len(env_idx),),
                M10_NOMINAL_INITIAL_DETENT_INDEX,
                device=self.device,
                dtype=torch.long,
            )

            target_indices = torch.full(
                (len(env_idx),),
                M10_NOMINAL_TARGET_DETENT_INDEX,
                device=self.device,
                dtype=torch.long,
            )

        row_indices = torch.arange(
            len(env_idx),
            device=self.device,
        )

        initial_angles = detent_angles[
            row_indices,
            initial_indices,
        ]

        target_angles = detent_angles[
            row_indices,
            target_indices,
        ]

        # The selectors can map to the same or neighboring detents. All
        # comparisons use circular distance so values around +/-pi are handled
        # identically to every other pair of detents.
        too_close = (
            self._circular_angle_distance(
                target_angles,
                initial_angles,
            )
            <= self.SUCCESS_SPEC.near_target_tolerance_rad
        )

        if too_close.any():
            # Keep the sampled initial detent and move only the target to the
            # circularly farthest valid detent.
            detent_distances = self._circular_angle_distance(
                detent_angles,
                initial_angles[:, None],
            )
            farthest_indices = torch.argmax(
                detent_distances,
                dim=1,
            )

            target_indices = torch.where(
                too_close,
                farthest_indices,
                target_indices,
            )

            target_angles = detent_angles[
                row_indices,
                target_indices,
            ]

        remaining_error = self._circular_angle_distance(
            target_angles,
            initial_angles,
        )

        if (
            remaining_error
            <= self.SUCCESS_SPEC.near_target_tolerance_rad
        ).any():
            raise RuntimeError(
                "M10 dishwasher knob range is too small for the "
                "configured near-target tolerance. "
                f"joint_limits={self.knob_joint.limits}, "
                f"near_tolerance_rad="
                f"{self.SUCCESS_SPEC.near_target_tolerance_rad:.6f}."
            )

        # 保存本 episode 的目标。
        self._target_angles_rad[env_idx] = (
            target_angles
        )
        self._target_detent_indices[env_idx] = (
            target_indices
        )

        # 设置 joint_3 的初始角度。
        # [B, D] -> [b, D]: setters write only through ManiSkill's reset mask.
        qpos = self.dishwasher.get_qpos()[env_idx].clone()
        qvel = self.dishwasher.get_qvel()[env_idx].clone()

        qpos[
            :,
            self._knob_joint_qpos_index,
        ] = initial_angles

        # The selected knob is explicitly reset to its sampled detent and the
        # complete fixed dishwasher articulation is left motionless. This avoids
        # a residual qvel from settle being interpreted as valid manipulation.
        qvel.zero_()

        self.dishwasher.set_qpos(qpos)
        self.dishwasher.set_qvel(qvel)

        print(
            "[M10 ANGLE RESET]",
            "initial_deg=",
            torch.rad2deg(
                initial_angles
            ).detach().cpu().tolist(),
            "target_deg=",
            torch.rad2deg(
                target_angles
            ).detach().cpu().tolist(),
            "joint_limits=",
            self.knob_joint.limits.detach().cpu().tolist(),
        )

    def _require_knob_contact_link(self) -> Any:
        """Return the real link contacted when turning the selected knob."""

        direct_child = getattr(self.knob_joint, "child_link", None)
        if direct_child is None:
            raise RuntimeError(
                f"Selected dishwasher knob joint {self.knob_joint.name!r} has no "
                "child link."
            )

        if "knob" in direct_child.name.lower():
            return direct_child

        knob_links = [
            link
            for link_name, link in self.dishwasher.links_map.items()
            if "knob" in link_name.lower()
        ]
        if not knob_links:
            raise RuntimeError(
                "The selected dishwasher articulation exposes no knob-like links. "
                f"Known links: {list(self.dishwasher.links_map)}."
            )

        direct_position = direct_child.pose.p[0]
        distances = [
            torch.linalg.norm(link.pose.p[0] - direct_position).item()
            for link in knob_links
        ]
        nearest_index = int(np.argmin(distances))
        nearest_link = knob_links[nearest_index]

        print(
            f"Selected knob contact link {nearest_link.name!r} for joint "
            f"{self.knob_joint.name!r}; direct child was "
            f"{direct_child.name!r}."
        )
        return nearest_link

    def _build_knob_scale_visuals(self) -> None:
        """Build one collision-free actor containing all rotating short ticks."""

        if sapien is None:
            raise ImportError(
                "SAPIEN is required to build the M10 knob scale visuals."
            )

        scale_material = sapien.render.RenderMaterial(
            base_color=[0.02, 0.02, 0.02, 1.00],
            roughness=0.8,
        )
        hidden_pose = sapien.Pose(p=[0.0, 0.0, M10_HIDDEN_Z_M])

        visual_scale = self._dishwasher_visual_scale
        scale_tick_half_size = (
            visual_scale * M10_KNOB_SCALE_TICK_HALF_WIDTH_M,
            visual_scale * M10_KNOB_SCALE_TICK_HALF_LENGTH_M,
            visual_scale * M10_KNOB_SCALE_TICK_HALF_THICKNESS_M,
        )
        builder = self.scene.create_actor_builder()
        for scale_tick_index, scale_tick_angle_rad in enumerate(
            M10_KNOB_SCALE_TICK_ANGLES_RAD
        ):
            position_xyz, quaternion_wxyz = self._knob_scale_tick_local_pose(
                scale_tick_angle_rad
            )
            builder.add_box_visual(
                pose=sapien.Pose(p=position_xyz, q=quaternion_wxyz),
                half_size=scale_tick_half_size,
                material=scale_material,
                name=f"m10_knob_scale_tick_shape_{scale_tick_index}",
            )
        builder.initial_pose = hidden_pose
        self.knob_scale_visual_actor = builder.build_kinematic(
            name="m10_knob_scale_visual"
        )

    def _knob_scale_tick_local_pose(
        self,
        display_angle_rad: float,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
    ]:
        """Return one clockwise-positive tick pose in the knob-link frame."""

        visual_scale = self._dishwasher_visual_scale
        tick_radius_m = visual_scale * M10_KNOB_SCALE_TICK_CENTER_RADIUS_M
        surface_offset_m = visual_scale * M10_KNOB_SCALE_TICK_SURFACE_OFFSET_M
        # The knob face is the local XY plane, its 0-degree line points along
        # +Y, and clockwise-positive display angles rotate toward local +X.
        position_xyz = (
            tick_radius_m * math.sin(display_angle_rad),
            tick_radius_m * math.cos(display_angle_rad),
            surface_offset_m,
        )
        half_angle_rad = -0.5 * display_angle_rad
        quaternion_wxyz = (
            math.cos(half_angle_rad),
            0.0,
            0.0,
            math.sin(half_angle_rad),
        )
        return position_xyz, quaternion_wxyz

    def _update_knob_scale_visuals(self) -> None:
        """Move the complete tick actor rigidly with the live knob link."""

        self.knob_scale_visual_actor.set_pose(self.knob_link.pose)

    def _require_dishwasher_visual_scale(self) -> float:
        """Return the selected layout's uniform dishwasher asset scale."""

        dishwasher_specs = [
            build_item
            for build_item in self.require_current_layout_spec().build_items
            if isinstance(build_item, ArticulatedAssetSpec)
            and build_item.name == "dishwasher"
        ]
        if len(dishwasher_specs) != 1:
            raise RuntimeError(
                "M10 requires exactly one articulated dishwasher layout item, "
                f"got {len(dishwasher_specs)}."
            )
        visual_scale = float(dishwasher_specs[0].scale)
        if not math.isfinite(visual_scale) or visual_scale <= 0.0:
            raise ValueError(
                "M10 dishwasher scale must be finite and positive, got "
                f"{visual_scale}."
            )
        return visual_scale

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self.current_layout_spec,
        ).build()

        self.dishwasher = self.layout_scene.require_entity("dishwasher")
        self._dishwasher_visual_scale = self._require_dishwasher_visual_scale()
        self.knob_joint = self._select_target_knob_joint()
        self.knob_link = self.knob_joint.child_link

        self._knob_joint_qpos_index = tuple(
            self.dishwasher.active_joints_map
        ).index(self.knob_joint.name)
        self._non_target_dishwasher_joint_qpos_indices = (
            self._require_non_target_dishwasher_joint_qpos_indices()
        )

        print(
            "M10 selected target knob:",
            f"joint={self.knob_joint.name!r},",
            f"link={self.knob_link.name!r},",
        )

        self._knob_tactile_target_groups = (
            ("knob", (self.knob_link,)),
        )
        if not self.gpu_sim_enabled:
            self.agent.prepare_cpu_object_taxel_query(
                self._knob_tactile_target_groups
            )

        # D3 covers impacts against the complete fixed dishwasher, not only
        # forces registered on the target knob taxels.
        self._dishwasher_force_query = self.agent.prepare_pairwise_contact_query(
            tuple(self.agent.robot.links),
            tuple(self.dishwasher.links),
        )

        self._build_knob_scale_visuals()
        self._configure_knob_joint_physics()

    def _require_non_target_dishwasher_joint_qpos_indices(
        self,
    ) -> dict[str, int]:
        """Return validated qpos indices for both shelves and the door."""

        active_joints = self.dishwasher.active_joints_map
        missing_joint_names = sorted(
            set(M10_NON_TARGET_DISHWASHER_JOINT_NAMES) - set(active_joints)
        )
        if missing_joint_names:
            raise RuntimeError(
                "M10 dishwasher is missing protected non-target joints: "
                f"{missing_joint_names}. Known joints: {list(active_joints)}."
            )
        active_joint_names = tuple(active_joints)
        return {
            joint_name: active_joint_names.index(joint_name)
            for joint_name in M10_NON_TARGET_DISHWASHER_JOINT_NAMES
        }

    def _configure_knob_joint_physics(self) -> None:
        """Apply nominal knob dynamics, then optional physical randomization."""

        joint_objects = tuple(getattr(self.knob_joint, "_objs", ()))
        if len(joint_objects) != self.num_envs:
            raise RuntimeError(
                "M10 joint configuration expected one underlying selected "
                f"knob joint per environment, got {len(joint_objects)} for "
                f"{self.num_envs}."
            )

        for joint_object in joint_objects:
            joint_object.set_friction(M10_KNOB_JOINT_FRICTION)
            joint_object.set_drive_properties(
                stiffness=0.0,
                damping=M10_KNOB_JOINT_DAMPING,
                force_limit=M10_KNOB_JOINT_FORCE_LIMIT,
            )

        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return

        values = self._generalization_values(
            M10_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m10_physical_values",
        )
        for env_index, joint_object in enumerate(joint_objects):
            friction = float(values["knob_joint_friction"][env_index])
            damping = float(values["knob_joint_damping"][env_index])
            joint_object.set_friction(friction)
            joint_object.set_drive_properties(
                stiffness=0.0,
                damping=damping,
                force_limit=M10_KNOB_JOINT_FORCE_LIMIT,
            )

    def _set_robot_to_rest_keyframe(self, batch_size: int) -> None:
        keyframes = getattr(self.agent, "keyframes", None)
        rest_keyframe = keyframes.get("rest") if keyframes else None
        if rest_keyframe is None or getattr(rest_keyframe, "qpos", None) is None:
            raise RuntimeError(
                "M10 requires the configured robot to expose a 'rest' keyframe."
            )

        robot_qpos = torch.as_tensor(
            rest_keyframe.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        if robot_qpos.ndim == 1:
            robot_qpos = robot_qpos.unsqueeze(0).repeat(batch_size, 1)
        self.agent.robot.set_qpos(robot_qpos)
        self.agent.robot.set_qvel(torch.zeros_like(robot_qpos))
        self.agent.controller.reset()

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_target_angles_rad"):
            return
        self._target_angles_rad = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._target_detent_indices = torch.zeros(
            self.num_envs,
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
        self._protected_dishwasher_initial_position = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._protected_dishwasher_initial_qpos = torch.zeros_like(
            self.dishwasher.get_qpos()
        )

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        with torch.device(self.device):
            self._invalidate_contact_force_cache()
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )

            self._ensure_episode_state()

            self.agent.robot.set_pose(
                sapien.Pose(p=[0.0, 0.0, M10_HIDDEN_Z_M])
            )
            # The freshly loaded zero-qpos Shadow Hand self-intersects. Apply
            # the task rest qpos before advancing physics so GPU PhysX does not
            # carry large self-collision impulses into the first task step.
            self._set_robot_to_rest_keyframe(batch_size)
            self.layout_scene.settle(env_idx=env_idx)

            self._reset_knob_angles(env_idx)
            self._update_knob_scale_visuals()

            self.agent.robot.set_pose(self._robot_base_pose(env_idx))
            self._set_robot_to_rest_keyframe(batch_size)

            self._knob_contacted_once[env_idx] = False
            self._knob_rotation_started[env_idx] = False
            self._knob_aligned_once[env_idx] = False
            self._protected_dishwasher_initial_position[env_idx] = (
                self.dishwasher.root_pose.p[env_idx]
            )
            self._protected_dishwasher_initial_qpos[env_idx] = (
                self.dishwasher.get_qpos()[env_idx]
            )

    def _knob_angle(self) -> Any:
        return self.knob_joint.qpos.reshape(self.num_envs, -1)[:, 0]

    def _knob_angular_speed(self) -> Any:
        return self.knob_joint.qvel.reshape(self.num_envs, -1)[:, 0]

    def _signed_target_angle_error(self) -> Any:
        return self._circular_angle_difference(
            self._knob_angle(),
            self._target_angles_rad,
        )

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
        self._cached_knob_tactile_force_norms: Any | None = None

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self._invalidate_contact_force_cache()

    def _knob_tactile_force_norms(self) -> Any:
        cached = getattr(self, "_cached_knob_tactile_force_norms", None)
        if cached is not None:
            return cached

        taxel_world_forces = self.agent.get_object_taxel_world_forces(
            self._knob_tactile_target_groups
        )["knob"]
        # [B, N, 3] -> [B, P, 3] -> [B, P].
        parent_world_forces = self.agent.aggregate_taxel_world_forces_by_parent(
            taxel_world_forces
        )
        cached = torch.linalg.norm(parent_world_forces, dim=-1)
        self._cached_knob_tactile_force_norms = cached
        return cached

    def _knob_has_tactile_contact(self) -> Any:
        return (
            self._knob_tactile_force_norms()
            >= self.SUCCESS_SPEC.contact_force_n
        ).any(dim=1)

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis is GeneralizationAxis.SPATIAL:
            return (
                self._target_angle_error()
                > self.SUCCESS_SPEC.near_target_tolerance_rad
            )
        if axis is GeneralizationAxis.PHYSICAL:
            return torch.abs(self._knob_angular_speed()) <= 0.05
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        knob_contacted = self._knob_has_tactile_contact()
        knob_near_target = self._knob_near_target()
        knob_aligned = self._knob_aligned()
        knob_rotating = (
            torch.abs(self._knob_angular_speed())
            >= self.INTERACTION_SPEC.min_rotation_speed_rad_s
        )

        interaction_started = (
            knob_contacted
            & knob_rotating
            & ~self._knob_aligned_once
        )
        self._knob_rotation_started |= interaction_started

        self._knob_contacted_once |= knob_contacted
        self._knob_aligned_once |= knob_aligned

        success = knob_aligned & self._knob_rotation_started

        # The visual actor follows the rotating knob link, matching the native
        # long line instead of remaining fixed on the dishwasher panel.
        self._update_knob_scale_visuals()

        """knob_angle = self._knob_angle()
        knob_speed = self._knob_angular_speed()
        angle_error = self._target_angle_error()

        print(
            "[M10 SUCCESS DEBUG]",
            "angle_deg=", torch.rad2deg(knob_angle).detach().cpu().tolist(),
            "error_deg=", torch.rad2deg(angle_error).detach().cpu().tolist(),
            "speed=", knob_speed.detach().cpu().tolist(),
            "contacted=", knob_contacted.detach().cpu().tolist(),
            "rotation_started=",
            self._knob_rotation_started.detach().cpu().tolist(),
            "aligned=", knob_aligned.detach().cpu().tolist(),
            "success=", success.detach().cpu().tolist(),
        )"""

        return {
            "success": success,
            "dishwasher_knob_contacted": knob_contacted,
            "dishwasher_knob_near_target": knob_near_target,
            "dishwasher_knob_aligned": knob_aligned,
        }
