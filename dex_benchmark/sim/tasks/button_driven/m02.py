from __future__ import annotations

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
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - depends on the optional simulator stack.
    sapien = None
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls):
            return cls

        return decorator


SETTINGS = TaskRegistry.register(TaskRegistry.load("m02_v1"))


JOINT_NAME = "joint_0"

# Task-success and stage thresholds.
_CLOSED_SWITCH_JOINT_VALUE_RAD = 0.0
_OPEN_SWITCH_JOINT_THRESHOLD_RAD = 0.122
_HALF_OPEN_SWITCH_JOINT_THRESHOLD_RAD = 0.061
_DISTANCE_THRESHOLD_REACH_SWITCH_M = 0.20
_M02_SUCCESS_DWELL_S = 1.00

# Small task-level spatial generalization. The switch remains attached to the wall;
# only the robot standing pose changes, in the wall-switch interaction-site frame.
_M02_SPATIAL_PARAMETER_NAMES = (
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
_M02_PHYSICAL_PARAMETER_NAMES = (
    "switch_joint_friction",
    "switch_joint_damping",
)
_M02_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "robot_forward_offset_m",
                -0.015,
                0.015,
            ),
            GeneralizationParameterSpec(
                "robot_side_offset_m",
                -0.015,
                0.015,
            ),
            GeneralizationParameterSpec(
                "robot_yaw_offset_rad",
                -math.radians(1.5),
                math.radians(1.5),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            # Vary passive resistance around the accepted L1 nominal of 0.5.
            GeneralizationParameterSpec(
                "switch_joint_friction",
                0.35,
                0.65,
            ),
            GeneralizationParameterSpec(
                "switch_joint_damping",
                1.0e-7,
                1.0e-6,
            ),
        ),
        requires_rebuild=True,
    ),
)

# Reset/generalization validity thresholds. An episode must start well below the
# 0.061-rad half-press stage and with negligible switch velocity.
_INITIAL_SWITCH_MAX_JOINT_VALUE_RAD = 0.02
_INITIAL_SWITCH_MAX_SPEED_RAD_S = 0.01

_M02_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.05, -0.65, -0.70)
_M02_TCP_WORKSPACE_LOCAL_MAX_XYZ = (0.45, 0.65, 0.75)

# Temporary parking position used while the layout settles without robot contact.
_M02_HIDDEN_Z_M = -20.0

@register_env(
    SETTINGS.env_id,
    max_episode_steps=SETTINGS.max_steps,
)
class TaskEnv(DexVLABenchBaseEnv):
    """Move the wall switch to 0.122 rad and hold it for 1.00 second."""

    TASK_SETTINGS = SETTINGS
    GENERALIZATION_SPEC = _M02_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "wall_switch"
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.773035, -0.253763, 1.289633),
            quaternion_world_wxyz=(0.281008, -0.173017, 0.051581, 0.942571),
        ),
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(3.685742, -0.203838, 1.234555),
            quaternion_world_wxyz=(0.236749, -0.074275, 0.018161, 0.968557),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.844368, -0.238100, 1.229321),
            quaternion_world_wxyz=(0.285245, -0.057485, 0.017063, 0.956577),
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
        """Return the nominal or spatially randomized robot pose.

        The perturbation is composed in the wall-switch site's local frame. Kitchen
        layouts can therefore place/orient the switch differently without changing
        this task logic.
        """

        nominal_pose = self.interaction_site_robot_base_sapien_pose()
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_pose

        values = self._generalization_values(
            _M02_SPATIAL_PARAMETER_NAMES,
            cache_name="_m02_spatial_values",
        )
        if env_idx is None:
            value_indices = np.arange(
                len(values["robot_forward_offset_m"]),
                dtype=np.int64,
            )
        else:
            value_indices = torch.as_tensor(
                env_idx,
                device=self.device,
                dtype=torch.long,
            ).reshape(-1).detach().cpu().numpy()

        forward_offset = values["robot_forward_offset_m"][value_indices]
        side_offset = values["robot_side_offset_m"][value_indices]
        yaw_offset = values["robot_yaw_offset_rad"][value_indices]

        local_position = np.stack(
            [
                forward_offset,
                side_offset,
                np.zeros_like(forward_offset),
            ],
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

    def _get_fixture_from_layout(self) -> Any:
        interaction_site = self.require_current_interaction_site()
        assert interaction_site.target_entity_name is not None
        return self.layout_scene.require_entity(
            interaction_site.target_entity_name
        )

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self.current_layout_spec,
            articulation_values=self._articulation_build_values(),
        ).build()
        self.fixture = self._get_fixture_from_layout()
        self.target_joint = self._get_joint()
        self.target_link = self._require_target_link()

        active_joint_names = tuple(self.fixture.active_joints_map)
        self._target_joint_qpos_index = active_joint_names.index(
            self.target_joint.name
        )

        self._success_dwell_steps = max(
            1,
            math.ceil(_M02_SUCCESS_DWELL_S * float(self._control_freq)),
        )

    def _load_lighting(self, options: dict[str, Any]) -> None:
        """Load the layout lights and bind M02's controllable ceiling lamp."""

        super()._load_lighting(options)
        self._m02_ceiling_light_components = (
            self._capture_ceiling_light_components()
        )

    def _capture_ceiling_light_components(
        self,
    ) -> tuple[tuple[Any, Any], ...]:
        """Return the main point/spot pair for every Kitchen sub-scene."""

        lighting_spec = self.require_current_layout_spec().lighting
        if lighting_spec is None or not lighting_spec.point_lights:
            raise RuntimeError("M02 requires one Kitchen ceiling point light.")
        if not lighting_spec.spot_lights:
            raise RuntimeError("M02 requires one Kitchen ceiling spot light.")

        ceiling_spot_spec = lighting_spec.spot_lights[0]
        sub_scenes = tuple(self.scene.sub_scenes)
        if len(sub_scenes) != self.num_envs:
            raise RuntimeError(
                "M02 dynamic ceiling-light feedback requires one render "
                "sub-scene per environment."
            )

        component_pairs: list[tuple[Any, Any]] = []
        for env_index, sub_scene in enumerate(sub_scenes):
            point_components = []
            spot_components = []
            for entity in sub_scene.get_entities():
                if entity.name == "point_light":
                    point_component = entity.find_component_by_type(
                        sapien.render.RenderPointLightComponent
                    )
                    if point_component is not None:
                        point_components.append(point_component)
                elif entity.name == "spot_light":
                    spot_component = entity.find_component_by_type(
                        sapien.render.RenderSpotLightComponent
                    )
                    if spot_component is not None:
                        spot_components.append(spot_component)

            if len(point_components) != 1:
                raise RuntimeError(
                    "M02 expected exactly one Kitchen ceiling point light in "
                    f"environment {env_index}, got {len(point_components)}."
                )
            ceiling_spot_components = tuple(
                component
                for component in spot_components
                if math.isclose(
                    float(component.inner_fov),
                    ceiling_spot_spec.inner_fov_rad,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                )
                and math.isclose(
                    float(component.outer_fov),
                    ceiling_spot_spec.outer_fov_rad,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6,
                )
            )
            if len(ceiling_spot_components) != 1:
                raise RuntimeError(
                    "M02 expected exactly one broad Kitchen ceiling spot light "
                    f"in environment {env_index}, got "
                    f"{len(ceiling_spot_components)}."
                )
            component_pairs.append(
                (point_components[0], ceiling_spot_components[0])
            )
        return tuple(component_pairs)

    def _get_joint(self) -> Any:
        try:
            joint = self.fixture.active_joints_map[JOINT_NAME]
        except KeyError as exc:
            known_joint_names = (
                ", ".join(sorted(self.fixture.active_joints_map)) or "<empty>"
            )
            raise RuntimeError(
                f"Fixture in layout {self.current_layout_id_value!r} does not "
                f"provide the configured joint {JOINT_NAME!r}. Known active "
                f"joints: {known_joint_names}."
            ) from exc

        joint_dofs = joint.qpos.reshape(self.num_envs, -1).shape[1]
        if joint_dofs != 1:
            raise RuntimeError(
                f"M02 requires {JOINT_NAME!r} to be one-DoF, got {joint_dofs}."
            )
        return joint

    def _link_position(self) -> Any:
        return self.target_link.pose.p

    def _hand_is_near_switch(self) -> Any:
        """Return whether any fingertip or the palm is near the switch.

        The robot TCP is attached near the wrist/palm controller frame.  With a
        Shadow Hand, a fingertip can therefore operate the switch while the TCP
        remains more than the reach-stage threshold away.  Measure the physical
        hand links that can actually approach the switch instead.
        """

        hand_links = tuple(getattr(self.agent, "tip_links", ()))
        palm_link = getattr(self.agent, "palm_link", None)
        if palm_link is not None:
            hand_links += (palm_link,)
        if not hand_links:
            raise RuntimeError(
                "M02 requires the robot agent to expose fingertip links through "
                "tip_links and/or a palm_link for reach-stage evaluation."
            )

        switch_position = self._link_position().unsqueeze(1)
        hand_positions = torch.stack(
            [link.pose.p for link in hand_links],
            dim=1,
        )
        minimum_distance = torch.linalg.norm(
            hand_positions - switch_position,
            dim=-1,
        ).amin(dim=1)
        return minimum_distance < _DISTANCE_THRESHOLD_REACH_SWITCH_M

    def _require_target_link(self) -> Any:
        link = self.target_joint.child_link
        if link is None:
            raise RuntimeError(
                f"Joint {JOINT_NAME!r} has no child link."
            )
        return link

    def _articulation_build_values(
        self,
    ) -> dict[str, ArticulationBuildValues]:
        """Return per-env switch joint values for physical generalization."""

        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return {}

        values = self._generalization_values(
            _M02_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m02_physical_values",
        )
        interaction_site = self.require_current_interaction_site()
        fixture_name = interaction_site.target_entity_name
        if not fixture_name:
            raise RuntimeError(
                "M02 wall_switch interaction site must bind a layout entity."
            )
        return {
            fixture_name: ArticulationBuildValues(
                joint_friction={
                    JOINT_NAME: tuple(
                        float(value)
                        for value in values["switch_joint_friction"]
                    )
                },
                joint_damping={
                    JOINT_NAME: tuple(
                        float(value)
                        for value in values["switch_joint_damping"]
                    )
                },
            )
        }

    def _ensure_episode_state(self) -> None:
        if not hasattr(self, "_ceiling_light_is_off"):
            self._ceiling_light_is_off = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        if not hasattr(self, "_success_dwell_steps"):
            self._success_dwell_steps = max(
                1,
                math.ceil(_M02_SUCCESS_DWELL_S * float(self._control_freq)),
            )
        if not hasattr(self, "_success_dwell_counter"):
            self._success_dwell_counter = torch.zeros(
                self.num_envs,
                dtype=torch.int64,
                device=self.device,
            )

    def _set_ceiling_light_off(
        self,
        light_is_off: Any,
        *,
        env_idx: Any | None = None,
        force: bool = False,
    ) -> None:
        """Synchronize the logical ceiling lamp with each switch state."""

        self._ensure_episode_state()
        if env_idx is None:
            selected_indices = torch.arange(
                self.num_envs,
                device=self.device,
                dtype=torch.long,
            )
        else:
            selected_indices = torch.as_tensor(
                env_idx,
                device=self.device,
                dtype=torch.long,
            ).reshape(-1)
        desired_state = torch.as_tensor(
            light_is_off,
            device=self.device,
            dtype=torch.bool,
        ).reshape(-1)
        if desired_state.numel() == 1 and len(selected_indices) != 1:
            desired_state = desired_state.repeat(len(selected_indices))
        if desired_state.numel() != len(selected_indices):
            raise ValueError(
                "M02 ceiling-light state count must match selected environments; "
                f"got {desired_state.numel()} states for "
                f"{len(selected_indices)} environments."
            )

        previous_state = self._ceiling_light_is_off[selected_indices]
        changed = previous_state != desired_state
        update_mask = torch.ones_like(changed) if force else changed
        update_indices = selected_indices[update_mask]
        update_states = desired_state[update_mask]
        components = getattr(self, "_m02_ceiling_light_components", ())
        if components:
            update_indices_cpu = update_indices.detach().cpu().tolist()
            update_states_cpu = update_states.detach().cpu().tolist()
            for env_index, is_off in zip(
                update_indices_cpu,
                update_states_cpu,
                strict=True,
            ):
                for component in components[env_index]:
                    if is_off:
                        component.disable()
                    else:
                        component.enable()
        self._ceiling_light_is_off[selected_indices] = desired_state

    def _set_robot_to_rest_keyframe(self, batch_size: int) -> None:
        keyframes = getattr(self.agent, "keyframes", None)
        if not keyframes:
            return
        rest_kf = keyframes.get("rest")
        if rest_kf is None:
            return

        if getattr(rest_kf, "qpos", None) is not None:
            qpos = torch.as_tensor(
                rest_kf.qpos,
                device=self.device,
                dtype=torch.float32,
            )
            if qpos.ndim == 1:
                qpos = qpos.unsqueeze(0).repeat(batch_size, 1)
            self.agent.robot.set_qpos(qpos)

            qvel = getattr(rest_kf, "qvel", None)
            if qvel is not None:
                qvel_tensor = torch.as_tensor(
                    qvel,
                    device=self.device,
                    dtype=torch.float32,
                )
                if qvel_tensor.ndim == 1:
                    qvel_tensor = qvel_tensor.unsqueeze(0).repeat(
                        batch_size,
                        1,
                    )
                self.agent.robot.set_qvel(qvel_tensor)
            else:
                self.agent.robot.set_qvel(torch.zeros_like(qpos))

    def _reset_switch_joint(self, env_idx: Any) -> None:
        """Explicitly restore joint_0 to the closed state and zero velocity."""

        env_idx = torch.as_tensor(
            env_idx,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

        limits = self.target_joint.limits.reshape(
            self.num_envs,
            -1,
            2,
        )[env_idx, 0]
        lower_limit = limits[:, 0]
        upper_limit = limits[:, 1]
        closed_value = torch.full_like(
            lower_limit,
            _CLOSED_SWITCH_JOINT_VALUE_RAD,
        )
        finite_lower = torch.isfinite(lower_limit)
        finite_upper = torch.isfinite(upper_limit)
        invalid_closed_value = (
            (finite_lower & (closed_value < lower_limit - 1e-6))
            | (finite_upper & (closed_value > upper_limit + 1e-6))
        )
        if bool(invalid_closed_value.any().item()):
            raise RuntimeError(
                "M02 configured closed switch value 0 rad lies outside "
                f"{JOINT_NAME!r} limits in layout "
                f"{self.current_layout_id_value!r}."
            )

        fixture_qpos = self.fixture.get_qpos()[env_idx].clone()
        fixture_qpos[:, self._target_joint_qpos_index] = closed_value
        self.fixture.set_qpos(fixture_qpos)

        # A wall switch has no episode-relevant free motion outside joint_0;
        # clearing the articulation qvel also prevents residual settle impulses.
        self.fixture.set_qvel(torch.zeros_like(fixture_qpos))

    def _initialize_episode(
        self,
        env_idx: Any,
        options: dict[str, Any],
    ) -> None:
        with torch.device(self.device):
            env_idx_tensor = torch.as_tensor(
                env_idx,
                device=self.device,
                dtype=torch.long,
            ).reshape(-1)
            batch_size = len(env_idx_tensor)

            self.layout_scene.initialize_episode(
                env_idx=env_idx_tensor,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )

            # Explicitly close the switch before settling, so layout-authored or
            # previous-episode articulation state cannot influence the settle phase.
            self._reset_switch_joint(env_idx_tensor)

            self.agent.robot.set_pose(
                sapien.Pose(p=[0.0, 0.0, _M02_HIDDEN_Z_M])
            )
            self._set_robot_to_rest_keyframe(batch_size)
            self.layout_scene.settle(env_idx=env_idx_tensor)

            # Settling may leave a small residual qpos/qvel. Commit the closed state
            # again after settle; this is the state used as the episode baseline.
            self._reset_switch_joint(env_idx_tensor)

            self.agent.robot.set_pose(self._robot_base_pose(env_idx_tensor))
            self._set_robot_to_rest_keyframe(batch_size)

            self._ensure_episode_state()
            self._success_dwell_counter[env_idx_tensor] = 0
            # The switch starts closed while the ceiling lamp starts illuminated.
            self._set_ceiling_light_off(
                False,
                env_idx=env_idx_tensor,
                force=True,
            )

    def _get_joint_value(self) -> Any:
        # [B, D] -> [B]: select the configured one-DoF switch joint.
        return self.target_joint.qpos.reshape(self.num_envs, -1)[:, 0]

    def _get_joint_velocity(self) -> Any:
        # [B, D] -> [B]: select the configured one-DoF switch joint.
        return self.target_joint.qvel.reshape(self.num_envs, -1)[:, 0]

    def _switch_is_valid_initial_state(self) -> Any:
        return (
            self._get_joint_value()
            <= _INITIAL_SWITCH_MAX_JOINT_VALUE_RAD
        ) & (
            torch.abs(self._get_joint_velocity())
            <= _INITIAL_SWITCH_MAX_SPEED_RAD_S
        )

    def _tcp_within_wall_switch_workspace(self) -> Any:
        site_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0),
                batch_size=self.num_envs,
            ),
            q=self.interaction_site_quaternion_tensor(self.num_envs),
        )
        tcp_local_position = (site_pose.inv() * self.agent.tcp.pose).p
        workspace_lower = torch.as_tensor(
            _M02_TCP_WORKSPACE_LOCAL_MIN_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        workspace_upper = torch.as_tensor(
            _M02_TCP_WORKSPACE_LOCAL_MAX_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        return ~(
            (tcp_local_position < workspace_lower)
            | (tcp_local_position > workspace_upper)
        ).any(dim=1)

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis is GeneralizationAxis.SPATIAL:
            return (
                self._switch_is_valid_initial_state()
                & self._tcp_within_wall_switch_workspace()
            )
        if axis is GeneralizationAxis.PHYSICAL:
            return self._switch_is_valid_initial_state()
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()

        joint_value = self._get_joint_value()
        switch_at_target = joint_value >= _OPEN_SWITCH_JOINT_THRESHOLD_RAD
        self._success_dwell_counter = torch.where(
            switch_at_target,
            self._success_dwell_counter + 1,
            torch.zeros_like(self._success_dwell_counter),
        )
        success = switch_at_target & (
            self._success_dwell_counter >= self._success_dwell_steps
        )
        reached_switch = self._hand_is_near_switch()
        half_pressed_switch = (
            joint_value >= _HALF_OPEN_SWITCH_JOINT_THRESHOLD_RAD
        )
        # Reaching the pressed/open threshold immediately switches the main
        # ceiling lamp off; success additionally requires the 1.00-s dwell.
        self._set_ceiling_light_off(switch_at_target)

        return {
            "success": success,
            "reach_switch": reached_switch,
            "half_press_switch": half_pressed_switch,
            "turn_on_switch": success,
        }
