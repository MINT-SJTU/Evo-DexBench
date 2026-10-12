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
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.scenes.layouts import ArticulationBuildValues, LayoutBuilder
from dex_benchmark.sim.scenes.layouts.studio import (
    STUDIO_DESK_TOP_SURFACE_ID,
    STUDIO_PHYSICAL_MATERIAL,
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
        def decorator(cls):
            return cls

        return decorator


M03_SETTINGS = TaskRegistry.register(TaskRegistry.load("m03_v1"))


@dataclass(frozen=True)
class M03SuccessSpec:
    home_trigger_fraction: float = 0.70
    home_release_fraction: float = 0.20
    home_contact_force_n: float = 0.12
    max_final_phone_speed_mps: float = 0.50
    max_workspace_radius_m: float = 0.65
    min_desk_local_height_m: float = -0.08
    max_desk_local_height_m: float = 0.18


M03_SUCCESS_SPEC = M03SuccessSpec()
M03_HOME_JOINT_NAME = "joint_4"
M03_HOME_LINK_NAME = "link_4"
M03_BODY_LINK_NAME = "link_7"
M03_NOMINAL_HOME_TRAVEL_M = 0.0005
M03_HOME_STIFFNESS_N_PER_M = 700.0
M03_HOME_DAMPING_N_S_PER_M = 8.0
M03_HOME_FORCE_LIMIT_N = 10.0
M03_LOCKED_LIMIT_M = 1.0e-5
M03_LOCKED_STIFFNESS_N_PER_M = 3000.0
M03_LOCKED_DAMPING_N_S_PER_M = 30.0
M03_LOCKED_FORCE_LIMIT_N = 20.0
M03_TASK_PHONE_LOCAL_XY: tuple[float, float] = (0.05, 0.08)
M03_TASK_DESK_ITEM_POSES: tuple[
    tuple[str, tuple[float, float], float], ...
] = (
    ("laptop", (0.03, -0.30), math.radians(-8.0)),
    ("mouse", (-0.02, -0.58), 3.03),
    ("usb", (-0.10, -0.72), 0.12),
)
# original-72.obj bounds measured in link_7 coordinates, then scaled by 0.12.
M03_SCREEN_LOCAL_CENTER_XYZ = (0.001051, 0.011454, 0.004981)
M03_SCREEN_HALF_SIZE_XYZ = (0.0540, 0.0800, 0.0002)
M03_SPATIAL_PARAMETER_NAMES = (
    "phone_x_offset_m",
    "phone_y_offset_m",
    "phone_yaw_offset_rad",
)
M03_PHYSICAL_PARAMETER_NAMES = (
    "phone_mass_log_scale",
    "phone_friction_log_scale",
    "home_travel_m",
    "home_stiffness_n_per_m",
    "home_damping_n_s_per_m",
    "home_force_limit_n",
)
M03_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("phone_x_offset_m", -0.03, 0.03),
            GeneralizationParameterSpec("phone_y_offset_m", -0.03, 0.03),
            GeneralizationParameterSpec(
                "phone_yaw_offset_rad",
                -math.radians(15.0),
                math.radians(15.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "phone_mass_log_scale", math.log(0.80), math.log(1.20)
            ),
            GeneralizationParameterSpec(
                "phone_friction_log_scale", math.log(0.80), math.log(1.20)
            ),
            GeneralizationParameterSpec("home_travel_m", 0.0015, 0.0022),
            GeneralizationParameterSpec(
                "home_stiffness_n_per_m", 600.0, 800.0
            ),
            GeneralizationParameterSpec("home_damping_n_s_per_m", 7.0, 9.0),
            GeneralizationParameterSpec("home_force_limit_n", 8.0, 12.0),
        ),
        requires_rebuild=True,
    ),
)

@register_env(M03_SETTINGS.env_id, max_episode_steps=M03_SETTINGS.max_steps)
class M03Env(DexVLABenchBaseEnv):
    """Press and release Home on a freely moving desk-top phone."""

    TASK_SETTINGS = M03_SETTINGS
    SUCCESS_SPEC = M03_SUCCESS_SPEC
    GENERALIZATION_SPEC = M03_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "desk_top"
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.138532, -0.180801, 1.137637),
            quaternion_world_wxyz=(0.938949, 0.041666, 0.122755, -0.318702),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.288287, 1.405112, 1.112868),
            quaternion_world_wxyz=(0.876005, -0.077530, 0.150373, 0.451656),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.940778, 1.326828, 1.065471),
            quaternion_world_wxyz=(0.912218, -0.039433, 0.090461, 0.397644),
        ),
    }

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_pose(self) -> Any:
        return self.interaction_site_robot_base_sapien_pose()

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self.current_layout_spec,
            articulation_values=self._articulation_build_values(),
        ).build()
        self.phone = self.layout_scene.require_entity("phone")
        self.laptop = self.layout_scene.require_entity("laptop")
        self.mouse = self.layout_scene.require_entity("mouse")
        self.usb = self.layout_scene.require_entity("usb")
        self.home_joint = self._require_phone_joint(M03_HOME_JOINT_NAME)
        self.home_link = self._require_phone_link(M03_HOME_LINK_NAME)
        self.phone_body_link = self._require_phone_link(M03_BODY_LINK_NAME)
        if self.home_joint.child_link is not self.home_link:
            raise RuntimeError(
                "M03 requires phone joint_4 to actuate the red Home link_4."
            )
        self._build_screen_overlay()
        self._configure_queries_and_constants()

    def _articulation_build_values(
        self,
    ) -> dict[str, ArticulationBuildValues]:
        if self.generalization_axis is GeneralizationAxis.PHYSICAL:
            values = self._generalization_values(
                M03_PHYSICAL_PARAMETER_NAMES,
                cache_name="_m03_physical_values",
            )
            mass_scales = positive_scale(values["phone_mass_log_scale"])
            friction_scales = positive_scale(values["phone_friction_log_scale"])
            materials = tuple(
                scaled_physical_material(
                    STUDIO_PHYSICAL_MATERIAL,
                    friction_scale=float(scale),
                    restitution=STUDIO_PHYSICAL_MATERIAL.restitution,
                )
                for scale in friction_scales
            )
            home_travel = values["home_travel_m"]
            home_stiffness = values["home_stiffness_n_per_m"]
            home_damping = values["home_damping_n_s_per_m"]
            home_force_limit = values["home_force_limit_n"]
        else:
            mass_scales = np.ones(self.num_envs)
            materials = None
            home_travel = np.full(self.num_envs, M03_NOMINAL_HOME_TRAVEL_M)
            home_stiffness = np.full(
                self.num_envs, M03_HOME_STIFFNESS_N_PER_M
            )
            home_damping = np.full(self.num_envs, M03_HOME_DAMPING_N_S_PER_M)
            home_force_limit = np.full(
                self.num_envs, M03_HOME_FORCE_LIMIT_N
            )

        locked_joint_names = tuple(
            f"joint_{index}" for index in (0, 1, 2, 3, 5, 6)
        )
        joint_limits = {
            joint_name: tuple(
                (-M03_LOCKED_LIMIT_M, M03_LOCKED_LIMIT_M)
                for _ in range(self.num_envs)
            )
            for joint_name in locked_joint_names
        }
        joint_limits[M03_HOME_JOINT_NAME] = tuple(
            (0.0, float(value)) for value in home_travel
        )
        joint_stiffness = {
            joint_name: (M03_LOCKED_STIFFNESS_N_PER_M,) * self.num_envs
            for joint_name in locked_joint_names
        }
        joint_stiffness[M03_HOME_JOINT_NAME] = tuple(
            float(value) for value in home_stiffness
        )
        joint_damping = {
            joint_name: (M03_LOCKED_DAMPING_N_S_PER_M,) * self.num_envs
            for joint_name in locked_joint_names
        }
        joint_damping[M03_HOME_JOINT_NAME] = tuple(
            float(value) for value in home_damping
        )
        joint_force_limit = {
            joint_name: (M03_LOCKED_FORCE_LIMIT_N,) * self.num_envs
            for joint_name in locked_joint_names
        }
        joint_force_limit[M03_HOME_JOINT_NAME] = tuple(
            float(value) for value in home_force_limit
        )
        return {
            "phone": ArticulationBuildValues(
                physical_materials=materials,
                mass_scales=tuple(float(value) for value in mass_scales),
                joint_limits=joint_limits,
                joint_stiffness=joint_stiffness,
                joint_damping=joint_damping,
                joint_force_limit=joint_force_limit,
            )
        }

    def _require_phone_joint(self, joint_name: str) -> Any:
        try:
            joint = self.phone.active_joints_map[joint_name]
        except KeyError as exc:
            known_names = ", ".join(sorted(self.phone.active_joints_map))
            raise RuntimeError(
                f"M03 requires phone joint {joint_name!r}; known: {known_names}."
            ) from exc
        if joint.active_index is None:
            raise RuntimeError(f"M03 phone joint {joint_name!r} is not active.")
        return joint

    def _require_phone_link(self, link_name: str) -> Any:
        try:
            return self.phone.links_map[link_name]
        except KeyError as exc:
            known_names = ", ".join(sorted(self.phone.links_map))
            raise RuntimeError(
                f"M03 requires phone link {link_name!r}; known: {known_names}."
            ) from exc

    def _build_screen_overlay(self) -> None:
        material = sapien.render.RenderMaterial(
            base_color=[0.025, 0.025, 0.025, 1.0],
            roughness=0.75,
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=M03_SCREEN_HALF_SIZE_XYZ,
            material=material,
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        self.screen_overlay = builder.build_kinematic(
            name="m03_non_home_screen_overlay"
        )

    def _configure_queries_and_constants(self) -> None:
        index_link_name = "rh_ffdistal"
        if index_link_name not in self.agent.robot.links_map:
            raise RuntimeError(f"M03 requires robot link {index_link_name!r}.")
        self._home_query: PairwiseContactQueryHandle = (
            self.agent.prepare_pairwise_contact_query(
                (self.agent.robot.links_map[index_link_name],),
                (self.home_link,),
            )
        )
        self._home_active_index = int(
            self.home_joint.active_index.reshape(-1)[0].item()
        )

    def _initialize_episode(
        self, env_idx: Any, options: dict[str, Any]
    ) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._position_desk_items_for_task(env_idx)
            self._position_phone_for_task(env_idx)
            qpos = self.phone.get_qpos()[env_idx]
            qpos.zero_()
            self.phone.set_qpos(qpos)
            self.phone.set_qvel(torch.zeros_like(qpos))
            self.home_joint.set_drive_target(
                torch.zeros(batch_size, device=self.device)
            )
            for joint_name, joint in self.phone.active_joints_map.items():
                if joint_name != M03_HOME_JOINT_NAME:
                    joint.set_drive_target(
                        torch.zeros(batch_size, device=self.device)
                    )

            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_agent_to_rest(
                self.agent, self._robot_base_pose(), batch_size
            )

            self._ensure_episode_state()
            self._home_shown[env_idx] = False
            self._home_contacted_once[env_idx] = False
            self._home_pressed_once[env_idx] = False
            self._home_released_after_press[env_idx] = False
            self._update_screen_overlay()

    def _position_desk_items_for_task(self, env_idx: Any) -> None:
        batch_size = len(env_idx)
        surface = self.current_layout_spec.support_surfaces[
            STUDIO_DESK_TOP_SURFACE_ID
        ]
        surface_pose = Pose.create_from_pq(
            p=torch.as_tensor(
                surface.pose.position_xyz,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1),
            q=torch.as_tensor(
                surface.pose.quaternion_wxyz,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1),
        )
        entities = {
            "laptop": self.laptop,
            "mouse": self.mouse,
            "usb": self.usb,
        }
        for item_name, desired_local_xy, desired_local_yaw in (
            M03_TASK_DESK_ITEM_POSES
        ):
            entity = entities[item_name]
            placement = self.current_layout_spec.relative_placements[item_name]
            nominal_local_bbox_pose = Pose.create_from_pq(
                p=torch.as_tensor(
                    placement.bbox_center_pose.position_xyz,
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1),
                q=torch.as_tensor(
                    placement.bbox_center_pose.quaternion_wxyz,
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1),
            )
            desired_local_bbox_pose = Pose.create_from_pq(
                p=torch.as_tensor(
                    [
                        desired_local_xy[0],
                        desired_local_xy[1],
                        placement.bbox_center_pose.position_xyz[2],
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1),
                q=torch.as_tensor(
                    [
                        math.cos(0.5 * desired_local_yaw),
                        0.0,
                        0.0,
                        math.sin(0.5 * desired_local_yaw),
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .repeat(batch_size, 1),
            )
            current_root_pose = entity.root_pose[env_idx]
            nominal_bbox_pose = surface_pose * nominal_local_bbox_pose
            # [B] bbox poses -> [B] root poses: preserve the model's fixed
            # bbox-to-articulation-root transform while changing XY and yaw.
            bbox_to_root_pose = nominal_bbox_pose.inv() * current_root_pose
            entity.set_pose(
                surface_pose * desired_local_bbox_pose * bbox_to_root_pose
            )
            zero_velocity = torch.zeros(
                (batch_size, 3),
                device=self.device,
                dtype=torch.float32,
            )
            entity.set_root_linear_velocity(zero_velocity)
            entity.set_root_angular_velocity(zero_velocity)

    def _position_phone_for_task(self, env_idx: Any) -> None:
        batch_size = len(env_idx)
        local_x = np.full(batch_size, M03_TASK_PHONE_LOCAL_XY[0])
        local_y = np.full(batch_size, M03_TASK_PHONE_LOCAL_XY[1])
        yaw = np.zeros(batch_size, dtype=np.float64)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M03_SPATIAL_PARAMETER_NAMES,
                cache_name="_m03_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            local_x += values["phone_x_offset_m"][value_indices]
            local_y += values["phone_y_offset_m"][value_indices]
            yaw = values["phone_yaw_offset_rad"][value_indices]
        phone_placement = self.current_layout_spec.relative_placements["phone"]
        authored_local_position = np.asarray(
            phone_placement.bbox_center_pose.position_xyz,
            dtype=np.float64,
        )
        local_delta = np.stack(
            [
                local_x - authored_local_position[0],
                local_y - authored_local_position[1],
                np.zeros(batch_size),
            ],
            axis=1,
        )
        site_rotation = self.interaction_site_frame_pose().rotation_matrix
        # [b, 3] @ [3, 3].T -> [b, 3]: desk-local relocation in world.
        world_delta = torch.as_tensor(
            local_delta @ site_rotation.T,
            device=self.device,
            dtype=torch.float32,
        )
        yaw_tensor = torch.as_tensor(
            yaw, device=self.device, dtype=torch.float32
        )
        half_yaw = 0.5 * yaw_tensor
        yaw_quaternion = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(yaw_tensor),
                torch.zeros_like(yaw_tensor),
                torch.sin(half_yaw),
            ],
            dim=1,
        )
        current_pose = self.phone.root_pose[env_idx]
        self.phone.set_pose(
            Pose.create_from_pq(p=current_pose.p + world_delta, q=yaw_quaternion)
            * Pose.create_from_pq(q=current_pose.q)
        )
        zero_velocity = torch.zeros(
            (batch_size, 3), device=self.device, dtype=torch.float32
        )
        self.phone.set_root_linear_velocity(zero_velocity)
        self.phone.set_root_angular_velocity(zero_velocity)

    def _reset_agent_to_rest(
        self, agent: Any, pose: Any, batch_size: int
    ) -> None:
        rest = agent.keyframes.get("rest")
        if rest is None or rest.qpos is None:
            raise RuntimeError(f"M03 agent {agent.uid!r} requires a rest keyframe.")
        qpos = torch.as_tensor(
            rest.qpos, device=self.device, dtype=torch.float32
        ).unsqueeze(0).repeat(batch_size, 1)
        agent.reset(qpos)
        agent.robot.set_pose(pose)

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_home_contacted_once"):
            return
        self._home_contacted_once = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._home_pressed_once = torch.zeros_like(self._home_contacted_once)
        self._home_released_after_press = torch.zeros_like(
            self._home_contacted_once
        )
        self._home_shown = torch.zeros_like(self._home_contacted_once)

    @staticmethod
    def _next_consecutive_steps(counter: Any, condition: Any) -> Any:
        return torch.where(condition, counter + 1, torch.zeros_like(counter))

    def _home_contact_force(self) -> Any:
        override = getattr(self, "_state_injection_measurements", {}).get(
            "home_force_n"
        )
        if override is not None:
            return self._as_per_env_tensor(override, dtype=torch.float32)
        pair_forces = self.agent.get_pairwise_contact_forces(
            self._home_query
        )
        # [B, 1, 1, 3] -> [B]: index-finger/Home resultant.
        return torch.linalg.norm(pair_forces, dim=-1).amax(dim=(1, 2))

    def _phone_site_local_position(self) -> Any:
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=self.num_envs
            ),
            q=self.interaction_site_quaternion_tensor(self.num_envs),
        )
        return (site_pose.inv() * self.phone.root_pose).p

    def _phone_safe_on_desk(self) -> Any:
        local_position = self._phone_site_local_position()
        planar_radius = torch.linalg.norm(local_position[:, :2], dim=1)
        speed = torch.linalg.norm(self.phone.root_linear_velocity, dim=1)
        return (
            (planar_radius <= self.SUCCESS_SPEC.max_workspace_radius_m)
            & (
                local_position[:, 2]
                >= self.SUCCESS_SPEC.min_desk_local_height_m
            )
            & (
                local_position[:, 2]
                <= self.SUCCESS_SPEC.max_desk_local_height_m
            )
            & (speed <= self.SUCCESS_SPEC.max_final_phone_speed_mps)
        )

    def _home_fraction(self) -> Any:
        qpos = self.phone.get_qpos()[:, self._home_active_index]
        limits = self.phone.get_qlimits()[:, self._home_active_index]
        return ((qpos - limits[:, 0]) / (limits[:, 1] - limits[:, 0])).clamp(
            0.0, 1.0
        )

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        home_fraction = self._home_fraction()
        home_contact = (
            self._home_contact_force() >= self.SUCCESS_SPEC.home_contact_force_n
        )
        self._home_contacted_once |= home_contact
        valid_press = (
            home_contact
            & (home_fraction >= self.SUCCESS_SPEC.home_trigger_fraction)
        )
        self._home_pressed_once |= valid_press
        self._home_shown |= valid_press
        valid_release = (
            self._home_pressed_once
            & ~home_contact
            & (home_fraction <= self.SUCCESS_SPEC.home_release_fraction)
        )
        self._home_released_after_press |= valid_release
        self._update_screen_overlay()
        phone_safe = self._phone_safe_on_desk()
        success = self._home_released_after_press & phone_safe
        return {
            "success": success,
            "home_button_contacted": self._home_contacted_once.clone(),
            "home_button_pressed": self._home_pressed_once.clone(),
            "home_button_released": success,
            "home_fraction": home_fraction,
            "phone_safe_on_desk": phone_safe,
        }

    def _update_screen_overlay(self) -> None:
        local_pose = Pose.create_from_pq(
            p=torch.as_tensor(
                M03_SCREEN_LOCAL_CENTER_XYZ,
                device=self.device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(self.num_envs, 1)
        )
        overlay_pose = self.phone_body_link.pose * local_pose
        overlay_position = overlay_pose.p.clone()
        overlay_position[self._home_shown, 2] -= 20.0
        self.screen_overlay.set_pose(
            Pose.create_from_pq(p=overlay_position, q=overlay_pose.q)
        )
