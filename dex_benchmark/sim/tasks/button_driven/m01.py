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
from dex_benchmark.sim.scenes.layouts import ArticulationBuildValues, LayoutBuilder
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


M01_SETTINGS = TaskRegistry.register(TaskRegistry.load("m01_v1"))


@dataclass(frozen=True)
class M01SuccessSpec:
    trigger_fraction: float = 0.70
    release_fraction: float = 0.20
    lit_dwell_s: float = 1.00
    contact_force_n: float = 0.15
    min_depression_per_step_m: float = 2.0e-5


M01_SUCCESS_SPEC = M01SuccessSpec()
M01_BUTTON_JOINT_NAME = "joint_0"
M01_BUTTON_LINK_NAME = "link_0"
M01_LAMP_HEAD_LINK_NAME = "link_1"
M01_NOMINAL_LOWER_LIMIT_M = 0.0012
M01_NOMINAL_TRAVEL_M = 0.0060
M01_NOMINAL_STIFFNESS_N_PER_M = 250.0
M01_NOMINAL_DAMPING_N_S_PER_M = 5.0
M01_NOMINAL_FORCE_LIMIT_N = 8.0
M01_NOMINAL_LAMP_LOCAL_X_OFFSET_M = -0.03
M01_BUTTON_COLLISION_REST_OFFSET_M = 0.0005
M01_SPATIAL_PARAMETER_NAMES = (
    "lamp_x_offset_m",
    "lamp_y_offset_m",
    "lamp_yaw_offset_rad",
)
M01_PHYSICAL_PARAMETER_NAMES = (
    "button_travel_m",
    "button_stiffness_n_per_m",
    "button_damping_n_s_per_m",
    "button_force_limit_n",
)
M01_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("lamp_x_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec("lamp_y_offset_m", -0.02, 0.02),
            GeneralizationParameterSpec(
                "lamp_yaw_offset_rad",
                -math.radians(8.0),
                math.radians(8.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("button_travel_m", 0.0054, 0.0066),
            GeneralizationParameterSpec(
                "button_stiffness_n_per_m", 210.0, 290.0
            ),
            GeneralizationParameterSpec("button_damping_n_s_per_m", 4.0, 6.0),
            GeneralizationParameterSpec("button_force_limit_n", 7.0, 9.0),
        ),
        requires_rebuild=True,
    ),
)



@register_env(M01_SETTINGS.env_id, max_episode_steps=M01_SETTINGS.max_steps)
class M01Env(DexVLABenchBaseEnv):
    """Press the desk-lamp button and keep the lamp illuminated."""

    TASK_SETTINGS = M01_SETTINGS
    SUCCESS_SPEC = M01_SUCCESS_SPEC
    GENERALIZATION_SPEC = M01_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "desk_top"
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.000726, 0.053888, 1.116294),
            quaternion_world_wxyz=(0.953939, 0.030836, 0.105358, -0.279192),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(0.950821, 1.233252, 1.147244),
            quaternion_world_wxyz=(0.877465, -0.073010, 0.141606, 0.452408),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-1.238490, 1.158192, 1.124003),
            quaternion_world_wxyz=(0.877126, -0.058865, 0.111412, 0.463435),
        ),
    }

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        # ManiSkill intentionally omits its render system when the benchmark is
        # run with ``render_backend="none"``.  Keep the physical lamp state in
        # that mode, but do not create point lights that require a renderer.
        self._lamp_render_feedback_enabled = (
            kwargs.get("render_backend", "gpu") != "none"
        )
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        del env_idx
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
        self.lamp = self.layout_scene.require_entity("desk_lamp")
        self.button_joint = self._require_active_joint(M01_BUTTON_JOINT_NAME)
        self.button_link = self._require_lamp_link(M01_BUTTON_LINK_NAME)
        self.lamp_head_link = self._require_lamp_link(M01_LAMP_HEAD_LINK_NAME)
        self._configure_button_collision_envelope()
        self._build_lamp_feedback()
        self._configure_queries_and_constants()

    def _articulation_build_values(
        self,
    ) -> dict[str, ArticulationBuildValues]:
        if self.generalization_axis is GeneralizationAxis.PHYSICAL:
            values = self._generalization_values(
                M01_PHYSICAL_PARAMETER_NAMES,
                cache_name="_m01_physical_values",
            )
            travel = values["button_travel_m"]
            stiffness = values["button_stiffness_n_per_m"]
            damping = values["button_damping_n_s_per_m"]
            force_limit = values["button_force_limit_n"]
        else:
            travel = np.full(self.num_envs, M01_NOMINAL_TRAVEL_M)
            stiffness = np.full(
                self.num_envs, M01_NOMINAL_STIFFNESS_N_PER_M
            )
            damping = np.full(self.num_envs, M01_NOMINAL_DAMPING_N_S_PER_M)
            force_limit = np.full(
                self.num_envs, M01_NOMINAL_FORCE_LIMIT_N
            )
        return {
            "desk_lamp": ArticulationBuildValues(
                joint_limits={
                    M01_BUTTON_JOINT_NAME: tuple(
                        (
                            M01_NOMINAL_LOWER_LIMIT_M,
                            M01_NOMINAL_LOWER_LIMIT_M + float(value),
                        )
                        for value in travel
                    )
                },
                joint_stiffness={
                    M01_BUTTON_JOINT_NAME: tuple(float(value) for value in stiffness)
                },
                joint_damping={
                    M01_BUTTON_JOINT_NAME: tuple(float(value) for value in damping)
                },
                joint_force_limit={
                    M01_BUTTON_JOINT_NAME: tuple(
                        float(value) for value in force_limit
                    )
                },
            )
        }

    def _require_active_joint(self, joint_name: str) -> Any:
        try:
            joint = self.lamp.active_joints_map[joint_name]
        except KeyError as exc:
            known_names = ", ".join(sorted(self.lamp.active_joints_map))
            raise RuntimeError(
                f"M01 requires lamp joint {joint_name!r}; known joints: "
                f"{known_names}."
            ) from exc
        if joint.active_index is None:
            raise RuntimeError(f"M01 joint {joint_name!r} is not active.")
        return joint

    def _require_lamp_link(self, link_name: str) -> Any:
        try:
            return self.lamp.links_map[link_name]
        except KeyError as exc:
            known_names = ", ".join(sorted(self.lamp.links_map))
            raise RuntimeError(
                f"M01 requires lamp link {link_name!r}; known links: {known_names}."
            ) from exc

    def _configure_button_collision_envelope(self) -> None:
        collision_shape_count = 0
        for raw_body in self.button_link._bodies:
            for collision_shape in raw_body.get_collision_shapes():
                contact_offset = float(collision_shape.get_contact_offset())
                if M01_BUTTON_COLLISION_REST_OFFSET_M >= contact_offset:
                    raise ValueError(
                        "M01 button collision rest offset must remain below "
                        f"the PhysX contact offset, got rest="
                        f"{M01_BUTTON_COLLISION_REST_OFFSET_M} and "
                        f"contact={contact_offset}."
                    )
                collision_shape.set_rest_offset(
                    M01_BUTTON_COLLISION_REST_OFFSET_M
                )
                collision_shape_count += 1
        if collision_shape_count == 0:
            raise RuntimeError(
                "M01 button link has no collision shapes to enlarge."
            )

    def _build_lamp_feedback(self) -> None:
        material = sapien.render.RenderMaterial(
            base_color=[1.0, 0.72, 0.28, 1.0],
            emission=[8.0, 4.5, 1.2, 1.0],
            roughness=0.6,
        )
        builder = self.scene.create_actor_builder()
        builder.add_sphere_visual(radius=0.025, material=material)
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        self.lamp_emissive_proxy = builder.build_kinematic(
            name="m01_lamp_emissive_proxy"
        )
        # Link GPU pose buffers do not exist during _load_scene(). This audited
        # desk-local point is inside the lamp shade; the emissive proxy follows
        # the actual head link after gpu_init().
        light_position = self.interaction_site_local_point_to_world(
            (0.12, 0.54, 0.35)
        )
        self._lamp_point_lights: list[Any] = []
        if not self._lamp_render_feedback_enabled:
            return
        for env_index in range(self.num_envs):
            light = self.scene.add_point_light(
                list(light_position),
                color=[0.0, 0.0, 0.0],
                shadow=False,
                scene_idxs=[env_index],
            )
            self._lamp_point_lights.append(light)

    def _configure_queries_and_constants(self) -> None:
        index_link = self.agent.robot.links_map.get("rh_ffdistal")
        if index_link is None:
            raise RuntimeError("M01 requires robot link 'rh_ffdistal'.")
        self._button_contact_query: PairwiseContactQueryHandle = (
            self.agent.prepare_pairwise_contact_query(
                (index_link,), (self.button_link,)
            )
        )
        self._button_active_index = int(
            self.button_joint.active_index.reshape(-1)[0].item()
        )
        self._lit_dwell_steps = max(
            1, math.ceil(self.SUCCESS_SPEC.lit_dwell_s * self._control_freq)
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
            self._apply_lamp_pose_offsets(env_idx)

            qpos = self.lamp.get_qpos()[env_idx]
            limits = self.lamp.get_qlimits()[env_idx]
            # [b, D, 2] -> [b]: reset the button to each environment's lower limit.
            button_lower = limits[:, self._button_active_index, 0]
            qpos[:, self._button_active_index] = button_lower
            self.lamp.set_qpos(qpos)
            self.lamp.set_qvel(torch.zeros_like(qpos))
            self.button_joint.set_drive_target(button_lower)

            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)
            self.agent.robot.set_pose(self._robot_base_pose())
            self._reset_agent_to_rest(batch_size)

            self._ensure_episode_state()
            self._lamp_on[env_idx] = False
            self._button_armed[env_idx] = True
            self._button_was_triggered[env_idx] = False
            self._lit_steps[env_idx] = 0
            self._update_lamp_feedback(force=True)

    def _reset_agent_to_rest(self, batch_size: int) -> None:
        keyframes = getattr(self.agent, "keyframes", None)
        rest = keyframes.get("rest") if keyframes else None
        if rest is None or getattr(rest, "qpos", None) is None:
            raise RuntimeError("M01 requires the robot 'rest' keyframe.")
        qpos = torch.as_tensor(
            rest.qpos, device=self.device, dtype=torch.float32
        )
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0).repeat(batch_size, 1)
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_qvel(torch.zeros_like(qpos))
        self.agent.controller.reset()

    def _apply_lamp_pose_offsets(self, env_idx: Any) -> None:
        batch_size = len(env_idx)
        local_offset = np.zeros((batch_size, 3), dtype=np.float64)
        local_offset[:, 0] = M01_NOMINAL_LAMP_LOCAL_X_OFFSET_M
        yaw_values = np.zeros(batch_size, dtype=np.float64)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                M01_SPATIAL_PARAMETER_NAMES,
                cache_name="_m01_spatial_values",
            )
            value_indices = env_idx.detach().cpu().numpy()
            local_offset[:, 0] += values["lamp_x_offset_m"][value_indices]
            local_offset[:, 1] += values["lamp_y_offset_m"][value_indices]
            yaw_values = values["lamp_yaw_offset_rad"][value_indices]

        site_rotation = self.interaction_site_frame_pose().rotation_matrix
        # [b, 3] @ [3, 3].T -> [b, 3]: site-local offsets in world coordinates.
        world_offset = torch.as_tensor(
            local_offset @ site_rotation.T,
            device=self.device,
            dtype=torch.float32,
        )
        yaw = torch.as_tensor(
            yaw_values,
            device=self.device,
            dtype=torch.float32,
        )
        half_yaw = 0.5 * yaw
        yaw_quaternion = torch.stack(
            [
                torch.cos(half_yaw),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(half_yaw),
            ],
            dim=1,
        )
        current_pose = self.lamp.pose[env_idx]
        self.lamp.set_pose(
            Pose.create_from_pq(p=current_pose.p + world_offset, q=yaw_quaternion)
            * Pose.create_from_pq(q=current_pose.q)
        )

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_lamp_on"):
            return
        self._lamp_on = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._button_armed = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._button_was_triggered = torch.zeros_like(self._lamp_on)
        self._lit_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._feedback_state = torch.zeros_like(self._lamp_on)

    @staticmethod
    def _next_consecutive_steps(counter: Any, condition: Any) -> Any:
        return torch.where(condition, counter + 1, torch.zeros_like(counter))

    def _button_fraction(self) -> Any:
        qpos = self.lamp.get_qpos()[:, self._button_active_index]
        limits = self.lamp.get_qlimits()[:, self._button_active_index]
        travel = (limits[:, 1] - limits[:, 0]).clamp_min(1.0e-6)
        return ((qpos - limits[:, 0]) / travel).clamp(0.0, 1.0)

    def _button_contact_force(self) -> Any:
        override = getattr(self, "_state_injection_measurements", {}).get(
            "button_contact_force_n"
        )
        if override is not None:
            return self._as_per_env_tensor(override, dtype=torch.float32)
        pair_forces = self.agent.get_pairwise_contact_forces(
            self._button_contact_query
        )
        # [B, 1, 1, 3] -> [B]: target index-finger/button resultant.
        return torch.linalg.norm(pair_forces, dim=-1).amax(dim=(1, 2))

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        fraction = self._button_fraction()
        button_contacted = (
            self._button_contact_force() >= self.SUCCESS_SPEC.contact_force_n
        )
        trigger_edge = (
            self._button_armed
            & (fraction >= self.SUCCESS_SPEC.trigger_fraction)
        )
        self._lamp_on ^= trigger_edge
        self._button_was_triggered |= trigger_edge
        self._button_armed &= ~trigger_edge
        released = fraction <= self.SUCCESS_SPEC.release_fraction
        self._button_armed |= released
        self._lit_steps = self._next_consecutive_steps(
            self._lit_steps, self._lamp_on
        )
        lamp_stably_lit = self._lamp_on & (
            self._lit_steps >= self._lit_dwell_steps
        )
        self._update_lamp_feedback()
        return {
            "success": lamp_stably_lit,
            "button_contacted": button_contacted,
            "button_triggered": self._button_was_triggered,
            "lamp_stably_lit": lamp_stably_lit,
            "lamp_on": self._lamp_on.clone(),
            "button_fraction": fraction,
        }

    def _update_lamp_feedback(self, *, force: bool = False) -> None:
        state_changed = force or not bool(
            (self._feedback_state == self._lamp_on).all()
        )
        head_position = self.lamp_head_link.pose.p
        proxy_position = head_position.clone()
        proxy_position[~self._lamp_on, 2] -= 20.0
        self.lamp_emissive_proxy.set_pose(
            Pose.create_from_pq(p=proxy_position)
        )
        light_positions = head_position.detach().cpu().numpy()
        if self.scene.parallel_in_single_scene:
            # [B, 3] + [B, 3] -> [B, 3]: logical scene coordinates to the
            # shared render scene coordinates used by ManiSkill point lights.
            light_positions = light_positions + self.scene.scene_offsets_np
        lamp_on_cpu = (
            self._lamp_on.detach().cpu().tolist() if state_changed else None
        )
        for env_index, light in enumerate(self._lamp_point_lights):
            light.pose = sapien.Pose(p=light_positions[env_index])
            if lamp_on_cpu is not None:
                light.color = (
                    [5.0, 2.8, 0.8]
                    if lamp_on_cpu[env_index]
                    else [0.0, 0.0, 0.0]
                )
        if state_changed:
            self._feedback_state.copy_(self._lamp_on)
