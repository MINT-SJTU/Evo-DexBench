from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.assets import (
    evodex_asset_dir,
)
from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.envs.base_env import DexVLABenchBaseEnv
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
)
from dex_benchmark.sim.layout_types import LayoutId
from dex_benchmark.sim.scenes.layouts import (
    ArticulatedAssetSpec,
    ArticulationBuildValues,
    LayoutBuilder,
    PoseSpec,
)
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


M09_SETTINGS = TaskRegistry.register(TaskRegistry.load("m09_v1"))
M09_INTERACTION_SITE_ID = "dining_table"
M09_PITCHER_ENTITY_NAME = "m09_pitcher"
M09_CAP_JOINT_NAME = "cap_joint"
M09_CAP_LINK_NAME = "cap"
M09_CAP_HANDLE_LINK_NAME = "cap_handle"
M09_GLOBAL_DINING_CLUTTER_NAMES = frozenset(("bottle_yellow",))
M09_FINGER_LINK_GROUPS = (
    ("rh_thproximal", "rh_thmiddle", "rh_thdistal"),
    ("rh_ffproximal", "rh_ffmiddle", "rh_ffdistal"),
    ("rh_mfproximal", "rh_mfmiddle", "rh_mfdistal"),
    ("rh_rfproximal", "rh_rfmiddle", "rh_rfdistal"),
    ("rh_lfproximal", "rh_lfmiddle", "rh_lfdistal"),
)
M09_FINGER_LINK_NAMES = tuple(
    link_name
    for finger_link_names in M09_FINGER_LINK_GROUPS
    for link_name in finger_link_names
)
M09_THUMB_FINGER_INDEX = 0
M09_INDEX_FINGER_INDEX = 1

# The pitcher body is YCB 019, uniformly enlarged by 5%. The task-specific cap
# is built from matching visual/collision primitives: a 145 mm round lid and a
# one-sided radial paddle sized for a Shadow Hand pinch. The paddle is both the
# grasp affordance and the only moving orientation cue.
M09_PITCHER_MODEL_ID = "019_pitcher_base"
M09_OBJECT_SCALE = 1.05
# Dining-table site frames share the robot-base yaw. Relative to the preceding
# accepted pose, move the pitcher another +30 mm in robot X and +80 mm in Y.
M09_PITCHER_ROOT_LOCAL_XYZ_M = (
    0.070,
    0.080,
    0.132 * M09_OBJECT_SCALE,
)
M09_CAP_JOINT_LOCAL_Z_M = 0.132 * M09_OBJECT_SCALE
M09_FIXTURE_BASE_CENTER_Z_M = -0.126 * M09_OBJECT_SCALE
M09_FIXTURE_BASE_RADIUS_M = 0.090 * M09_OBJECT_SCALE
M09_FIXTURE_BASE_HEIGHT_M = 0.012 * M09_OBJECT_SCALE
M09_CAP_RADIUS_M = 0.069 * M09_OBJECT_SCALE
M09_CAP_HEIGHT_M = 0.020 * M09_OBJECT_SCALE
M09_CAP_PADDLE_BOX_SIZE_M = tuple(
    value * M09_OBJECT_SCALE for value in (0.052, 0.028, 0.018)
)
M09_CAP_PADDLE_BOX_CENTER_M = tuple(
    value * M09_OBJECT_SCALE for value in (0.032, 0.0, 0.019)
)
M09_CAP_PADDLE_END_RADIUS_M = 0.014 * M09_OBJECT_SCALE
M09_CAP_PADDLE_END_HEIGHT_M = 0.018 * M09_OBJECT_SCALE
M09_CAP_PADDLE_INNER_CENTER_M = tuple(
    value * M09_OBJECT_SCALE for value in (0.006, 0.0, 0.019)
)
M09_CAP_PADDLE_OUTER_CENTER_M = tuple(
    value * M09_OBJECT_SCALE for value in (0.058, 0.0, 0.019)
)
M09_TARGET_NOTCH_CAP_OVERLAP_M = 0.004 * M09_OBJECT_SCALE
# The fixed target remains a pitcher-body visual so it does not rotate with the
# cap, but its inner V vertex overlaps the cap edge to avoid a floating marker.
M09_TARGET_NOTCH_RADIUS_M = (
    M09_CAP_RADIUS_M - M09_TARGET_NOTCH_CAP_OVERLAP_M
)
M09_TARGET_NOTCH_ARM_LENGTH_M = 0.014 * M09_OBJECT_SCALE
M09_TARGET_NOTCH_ARM_WIDTH_M = 0.003 * M09_OBJECT_SCALE
M09_TARGET_NOTCH_HEIGHT_M = 0.006 * M09_OBJECT_SCALE
M09_TARGET_NOTCH_HALF_ANGLE_RAD = math.radians(28.0)
M09_CAP_LOWER_LIMIT_RAD = math.radians(-8.0)
M09_CAP_UPPER_LIMIT_RAD = math.radians(95.0)
M09_TARGET_ANGLE_RAD = math.radians(75.0)
M09_CAP_JOINT_DAMPING = 0.10
M09_CAP_JOINT_FRICTION = 0.075
M09_ASSET_RELATIVE_DIR = Path("task_assets") / "m09"


@dataclass(frozen=True)
class M09SuccessSpec:
    contact_force_n: float = 0.35
    positive_rotation_min_angle_rad: float = math.radians(5.0)
    positive_rotation_min_speed_rad_s: float = 0.03
    near_target_min_angle_rad: float = math.radians(60.0)
    near_target_max_angle_rad: float = math.radians(90.0)
    aligned_tolerance_rad: float = math.radians(9.0)
    aligned_max_speed_rad_s: float = 0.5
    max_fixture_displacement_m: float = 0.01
    max_fixture_tilt_rad: float = math.radians(3.0)


M09_SUCCESS_SPEC = M09SuccessSpec()
M09_SPATIAL_PARAMETER_NAMES = (
    "pitcher_x_offset_m",
    "pitcher_y_offset_m",
    "pitcher_yaw_offset_rad",
)
M09_PHYSICAL_PARAMETER_NAMES = (
    "cap_joint_friction",
    "cap_joint_damping",
)
M09_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("pitcher_x_offset_m", -0.035, 0.035),
            GeneralizationParameterSpec("pitcher_y_offset_m", -0.035, 0.035),
            GeneralizationParameterSpec(
                "pitcher_yaw_offset_rad",
                -math.radians(10.0),
                math.radians(10.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("cap_joint_friction", 0.025, 0.14),
            GeneralizationParameterSpec("cap_joint_damping", 0.025, 0.18),
        ),
        requires_rebuild=True,
    ),
)



def m09_asset_paths() -> tuple[Path, Path, Path]:
    """Resolve the persistent M09 URDF, visual mesh, and collision mesh."""

    asset_root = evodex_asset_dir() / M09_ASSET_RELATIVE_DIR
    urdf_path = asset_root / "m09_pitcher.urdf"
    pitcher_visual_path = asset_root / "pitcher_body.obj"
    pitcher_collision_path = asset_root / "pitcher_body_collision.ply"
    required_paths = (
        urdf_path,
        pitcher_visual_path,
        pitcher_collision_path,
        asset_root / "material_0.mtl",
        asset_root / "material_0.png",
    )
    missing_paths = tuple(path for path in required_paths if not path.is_file())
    if missing_paths:
        raise FileNotFoundError(
            "M09 persistent asset bundle is incomplete; missing: "
            + ", ".join(str(path) for path in missing_paths)
        )
    return (
        urdf_path.resolve(),
        pitcher_visual_path.resolve(),
        pitcher_collision_path.resolve(),
    )


@register_env(M09_SETTINGS.env_id, max_episode_steps=M09_SETTINGS.max_steps)
class M09Env(DexVLABenchBaseEnv):
    """Rotate a top-handled cap on a fixed pitcher to the marked angle."""

    TASK_SETTINGS = M09_SETTINGS
    SUCCESS_SPEC = M09_SUCCESS_SPEC
    GENERALIZATION_SPEC = M09_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = M09_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.989318, -3.092527, 1.429823),
            quaternion_world_wxyz=(0.858509, -0.141532, 0.347869, 0.349168),
        ),
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.421648, -2.995369, 1.387938),
            quaternion_world_wxyz=(0.861520, 0.132520, 0.287750, -0.396763),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.108494, -3.951825, 1.412241),
            quaternion_world_wxyz=(0.242255, 0.335656, 0.089765, -0.905864),
        ),
    }

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self.interaction_site_robot_base_sapien_pose())

    def _task_layout_spec(self, urdf_path: Path) -> Any:
        world_pose = self.interaction_site_local_pose_to_world(
            PoseSpec(position_xyz=M09_PITCHER_ROOT_LOCAL_XYZ_M)
        )
        retained_items = [
            item
            for item in self.current_layout_spec.build_items
            if item.name not in M09_GLOBAL_DINING_CLUTTER_NAMES
        ]
        return replace(
            self.current_layout_spec,
            build_items=[
                *retained_items,
                ArticulatedAssetSpec(
                    name=M09_PITCHER_ENTITY_NAME,
                    asset_path=urdf_path,
                    pose=world_pose,
                    fix_root_link=True,
                    disable_self_collisions=True,
                    initial_qpos=(0.0,),
                ),
            ],
        )

    def _articulation_build_values(self) -> dict[str, ArticulationBuildValues]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return {}
        values = self._generalization_values(
            M09_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m09_physical_values",
        )
        return {
            M09_PITCHER_ENTITY_NAME: ArticulationBuildValues(
                joint_friction={
                    M09_CAP_JOINT_NAME: tuple(
                        float(value) for value in values["cap_joint_friction"]
                    )
                },
                joint_damping={
                    M09_CAP_JOINT_NAME: tuple(
                        float(value) for value in values["cap_joint_damping"]
                    )
                },
            )
        }

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        urdf_path, _, _ = m09_asset_paths()
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self._task_layout_spec(urdf_path),
            articulation_values=self._articulation_build_values(),
        ).build()
        self.pitcher = self.layout_scene.require_entity(M09_PITCHER_ENTITY_NAME)
        try:
            self.cap_joint = self.pitcher.active_joints_map[M09_CAP_JOINT_NAME]
            self.cap_link = self.pitcher.links_map[M09_CAP_LINK_NAME]
            self.cap_handle_link = self.pitcher.links_map[
                M09_CAP_HANDLE_LINK_NAME
            ]
        except KeyError as exc:
            raise RuntimeError(
                "M09 runtime articulation must expose cap_joint, cap, and "
                "cap_handle; "
                f"joints={tuple(self.pitcher.active_joints_map)}, "
                f"links={tuple(self.pitcher.links_map)}."
            ) from exc
        self._cap_qpos_index = tuple(self.pitcher.active_joints_map).index(
            M09_CAP_JOINT_NAME
        )
        self._nominal_root_position = self.pitcher.root_pose.p.clone()
        self._nominal_root_quaternion = self.pitcher.root_pose.q.clone()
        self._configure_contact_queries()

    def _configure_contact_queries(self) -> None:
        missing = [
            name
            for name in M09_FINGER_LINK_NAMES
            if name not in self.agent.robot.links_map
        ]
        if missing:
            raise RuntimeError(
                "M09 missing right-hand finger links: " + ", ".join(missing)
            )
        finger_links = tuple(
            self.agent.robot.links_map[name] for name in M09_FINGER_LINK_NAMES
        )
        self._cap_grasp_query = self.agent.prepare_pairwise_contact_query(
            finger_links, (self.cap_handle_link,)
        )
        self._pitcher_force_query = self.agent.prepare_pairwise_contact_query(
            tuple(self.agent.robot.links), tuple(self.pitcher.links)
        )
        self._pitcher_tactile_groups = (
            ("pitcher", tuple(self.pitcher.links)),
        )
        if not self.gpu_sim_enabled:
            self.agent.prepare_cpu_object_taxel_query(
                self._pitcher_tactile_groups
            )

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_cap_contacted_once"):
            return
        self._initial_cap_angle = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._cap_contacted_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._thumb_contacted_once = torch.zeros_like(
            self._cap_contacted_once
        )
        self._index_contacted_once = torch.zeros_like(
            self._cap_contacted_once
        )
        self._cap_grasped_once = torch.zeros_like(self._cap_contacted_once)
        self._positive_rotation_started = torch.zeros_like(
            self._cap_contacted_once
        )
        self._target_neighborhood_reached = torch.zeros_like(
            self._cap_contacted_once
        )
        self._cap_aligned_once = torch.zeros_like(self._cap_contacted_once)
        self._protected_root_position = torch.zeros(
            (self.num_envs, 3), dtype=torch.float32, device=self.device
        )
        self._protected_root_up = torch.zeros_like(self._protected_root_position)

    @staticmethod
    def _quaternion_up_axis(quaternion_wxyz: Any) -> Any:
        quaternion = quaternion_wxyz / torch.linalg.norm(
            quaternion_wxyz, dim=-1, keepdim=True
        ).clamp_min(1e-8)
        w, x, y, z = quaternion.unbind(dim=-1)
        # [B] components -> [B, 3] world-space local-Z axes.
        return torch.stack(
            (
                2.0 * (x * z + w * y),
                2.0 * (y * z - w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        )

    def _set_robot_to_rest(self, batch_size: int) -> None:
        keyframe = self.agent.keyframes.get("rest")
        if keyframe is None or keyframe.qpos is None:
            raise RuntimeError(f"Agent {self.agent.uid!r} has no rest keyframe.")
        qpos = torch.as_tensor(
            keyframe.qpos, device=self.device, dtype=torch.float32
        )
        if qpos.shape != (self.agent.robot.max_dof,):
            raise ValueError(
                "M09 rest qpos shape mismatch: expected "
                f"({self.agent.robot.max_dof},), got {tuple(qpos.shape)}."
            )
        # [D] -> [B, D]: reset all selected environments to the same keyframe.
        self.agent.reset(qpos.unsqueeze(0).repeat(batch_size, 1))
        self.agent.robot.set_pose(self.interaction_site_robot_base_sapien_pose())

    def _apply_spatial_generalization(self, env_idx: Any) -> None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return
        values = self._generalization_values(
            M09_SPATIAL_PARAMETER_NAMES,
            cache_name="_m09_spatial_values",
        )
        indices = env_idx.detach().cpu().numpy()
        x = values["pitcher_x_offset_m"][indices]
        y = values["pitcher_y_offset_m"][indices]
        yaw = values["pitcher_yaw_offset_rad"][indices]
        local_position = np.stack((x, y, np.zeros_like(x)), axis=1)
        half_yaw = 0.5 * yaw
        # Three [B] arrays -> [B, 4] WXYZ yaw quaternions.
        local_quaternion = np.stack(
            (
                np.cos(half_yaw),
                np.zeros_like(half_yaw),
                np.zeros_like(half_yaw),
                np.sin(half_yaw),
            ),
            axis=1,
        )
        nominal = Pose.create_from_pq(
            p=self._nominal_root_position[env_idx],
            q=self._nominal_root_quaternion[env_idx],
        )
        changed = nominal * Pose.create_from_pq(
            p=local_position, q=local_quaternion, device=self.device
        )
        position = self.pitcher.root_pose.p.clone()
        quaternion = self.pitcher.root_pose.q.clone()
        position[env_idx] = changed.p
        quaternion[env_idx] = changed.q
        self.pitcher.set_pose(Pose.create_from_pq(p=position, q=quaternion))

    def _reset_cap_joint(self, env_idx: Any) -> None:
        qpos = self.pitcher.get_qpos().clone()
        qvel = self.pitcher.get_qvel().clone()
        qpos[env_idx, self._cap_qpos_index] = 0.0
        qvel[env_idx, self._cap_qpos_index] = 0.0
        self.pitcher.set_qpos(qpos)
        self.pitcher.set_qvel(qvel)
        self._initial_cap_angle[env_idx] = 0.0

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            if batch_size <= 0:
                raise ValueError("M09 reset requires at least one environment.")
            self._invalidate_force_cache()
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._ensure_episode_state()
            self._apply_spatial_generalization(env_idx)
            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_cap_joint(env_idx)
            self._set_robot_to_rest(batch_size)
            self._cap_contacted_once[env_idx] = False
            self._thumb_contacted_once[env_idx] = False
            self._index_contacted_once[env_idx] = False
            self._cap_grasped_once[env_idx] = False
            self._positive_rotation_started[env_idx] = False
            self._target_neighborhood_reached[env_idx] = False
            self._cap_aligned_once[env_idx] = False
            self._protected_root_position[env_idx] = self.pitcher.root_pose.p[env_idx]
            self._protected_root_up[env_idx] = self._quaternion_up_axis(
                self.pitcher.root_pose.q
            )[env_idx]

    def _cap_angle(self) -> Any:
        return self.cap_joint.qpos.reshape(self.num_envs, -1)[:, 0]

    def _cap_speed(self) -> Any:
        return self.cap_joint.qvel.reshape(self.num_envs, -1)[:, 0]

    def _opening_angle(self) -> Any:
        return self._cap_angle() - self._initial_cap_angle

    def _target_error(self) -> Any:
        return torch.abs(self._opening_angle() - M09_TARGET_ANGLE_RAD)

    def _invalidate_force_cache(self) -> None:
        self._cached_cap_finger_contact: Any | None = None
        self._cached_cap_contact: Any | None = None
        self._cached_max_pitcher_force: Any | None = None

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self._invalidate_force_cache()

    @staticmethod
    def _finger_group_force_magnitudes(pair_forces: Any) -> Any:
        expected_link_count = len(M09_FINGER_LINK_NAMES)
        if (
            pair_forces.ndim != 4
            or pair_forces.shape[1] != expected_link_count
            or pair_forces.shape[-1] != 3
        ):
            raise ValueError(
                "M09 cap pair forces must have shape "
                f"[B, {expected_link_count}, H, 3], got "
                f"{tuple(pair_forces.shape)}."
            )

        finger_force_magnitudes: list[Any] = []
        link_start = 0
        for finger_link_names in M09_FINGER_LINK_GROUPS:
            link_end = link_start + len(finger_link_names)
            # [B, L, H, 3] -> [B, 3] -> [B]: sum every phalanx/handle
            # pair into one anatomical-finger resultant, then take its norm.
            finger_force_world = pair_forces[:, link_start:link_end].sum(
                dim=(1, 2)
            )
            finger_force_magnitudes.append(
                torch.linalg.norm(finger_force_world, dim=-1)
            )
            link_start = link_end
        return torch.stack(finger_force_magnitudes, dim=1)

    def _populate_force_cache(self) -> None:
        pair_forces = self.agent.get_pairwise_contact_forces(
            self._cap_grasp_query
        )
        # [B, 15, 1, 3] -> [B, 5]: whole-finger/cap resultant forces.
        finger_force = self._finger_group_force_magnitudes(pair_forces)
        finger_contact = finger_force >= self.SUCCESS_SPEC.contact_force_n
        self._cached_cap_finger_contact = finger_contact
        self._cached_cap_contact = finger_contact.any(dim=1)
        pitcher_forces = self.agent.get_object_taxel_world_forces(
            self._pitcher_tactile_groups
        )["pitcher"]
        # [B, N, 3] -> [B, P, 3] -> [B]: maximum parent resultant.
        parent_forces = self.agent.aggregate_taxel_world_forces_by_parent(
            pitcher_forces
        )
        tactile_force = torch.linalg.norm(
            parent_forces, dim=-1
        ).amax(dim=1)
        robot_pair_forces = self.agent.get_pairwise_contact_forces(
            self._pitcher_force_query
        )
        # [B, R, L, 3] -> [B]: include non-taxel arm/pitcher impacts in D3.
        pairwise_force = torch.linalg.norm(
            robot_pair_forces, dim=-1
        ).amax(dim=(1, 2))
        self._cached_max_pitcher_force = torch.maximum(
            tactile_force, pairwise_force
        )

    def _cap_contacted(self) -> Any:
        override = getattr(self, "_state_injection_measurements", {}).get(
            "cap_contacted"
        )
        if override is not None:
            return self._as_per_env_tensor(override, dtype=torch.bool)
        if self._cached_cap_contact is None:
            self._populate_force_cache()
        return self._cached_cap_contact

    def _cap_finger_contacted(self) -> Any:
        if self._cached_cap_finger_contact is None:
            self._populate_force_cache()
        return self._cached_cap_finger_contact

    def _max_pitcher_force(self) -> Any:
        if self._cached_max_pitcher_force is None:
            self._populate_force_cache()
        return self._cached_max_pitcher_force

    def _fixture_displacement(self) -> Any:
        return torch.linalg.norm(
            self.pitcher.root_pose.p - self._protected_root_position, dim=1
        )

    def _fixture_tilt(self) -> Any:
        current_up = self._quaternion_up_axis(self.pitcher.root_pose.q)
        cosine = torch.sum(current_up * self._protected_root_up, dim=1).clamp(
            -1.0, 1.0
        )
        return torch.acos(cosine)

    def _fixture_stable(self) -> Any:
        return (
            self._fixture_displacement()
            <= self.SUCCESS_SPEC.max_fixture_displacement_m
        ) & (self._fixture_tilt() <= self.SUCCESS_SPEC.max_fixture_tilt_rad)

    def _check_generalization_validity(
        self, axis: GeneralizationAxis | None
    ) -> Any:
        if axis is GeneralizationAxis.SPATIAL:
            return (
                self._opening_angle()
                < self.SUCCESS_SPEC.positive_rotation_min_angle_rad
            ) & self._fixture_stable()
        if axis is GeneralizationAxis.PHYSICAL:
            return (
                torch.abs(self._cap_speed()) <= 0.05
            ) & self._fixture_stable()
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        finger_contact = self._cap_finger_contacted()
        contact = finger_contact.any(dim=1)
        thumb_contact = finger_contact[:, M09_THUMB_FINGER_INDEX]
        index_contact = finger_contact[:, M09_INDEX_FINGER_INDEX]
        required_finger_contact = thumb_contact | index_contact
        opening_angle = self._opening_angle()
        cap_speed = self._cap_speed()
        fixture_stable = self._fixture_stable()

        self._cap_contacted_once |= contact
        self._thumb_contacted_once |= thumb_contact & fixture_stable
        self._index_contacted_once |= index_contact & fixture_stable
        self._cap_grasped_once |= (
            self._thumb_contacted_once & self._index_contacted_once
        )
        positive_now = (
            self._cap_grasped_once
            & (opening_angle >= self.SUCCESS_SPEC.positive_rotation_min_angle_rad)
            & (cap_speed >= self.SUCCESS_SPEC.positive_rotation_min_speed_rad_s)
        )
        self._positive_rotation_started |= positive_now
        near_target_now = (
            self._positive_rotation_started
            & (opening_angle >= self.SUCCESS_SPEC.near_target_min_angle_rad)
            & (opening_angle <= self.SUCCESS_SPEC.near_target_max_angle_rad)
        )
        self._target_neighborhood_reached |= near_target_now
        aligned_now = (
            self._target_neighborhood_reached
            & (self._target_error() <= self.SUCCESS_SPEC.aligned_tolerance_rad)
            & (torch.abs(cap_speed) <= self.SUCCESS_SPEC.aligned_max_speed_rad_s)
            & fixture_stable
        )
        success = aligned_now
        self._cap_aligned_once |= success
        return {
            "success": success,
            "cap_thumb_contacted": self._thumb_contacted_once,
            "cap_index_contacted": self._index_contacted_once,
            "cap_grasped": self._cap_grasped_once,
            "cap_positive_rotation": self._positive_rotation_started,
            "cap_target_neighborhood": self._target_neighborhood_reached,
            "cap_aligned_and_stable": success,
            "cap_opening_angle_rad": opening_angle,
            "cap_target_error_rad": self._target_error(),
            "pitcher_fixture_stable": fixture_stable,
        }
