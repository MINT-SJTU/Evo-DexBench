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


SETTINGS = TaskRegistry.register(TaskRegistry.load("m04_v1"))


JOINT_NAME = "joint_1"

# Preserve the original task/stage semantics.
_PRESS_SLIDER_JOINT_THRESHOLD_M = 0.062
_DISTANCE_THRESHOLD_REACH_TOASTER_M = 0.40
_DISTANCE_THRESHOLD_REACH_SLIDER_M = 0.35
_INTERACTION_SITE_NAME = "toaster"

# This is a point in the moving slider-link frame, not a world-frame correction.
# Keeping the calibrated numbers while changing their frame makes the reach target
# follow toaster/link translation and rotation in every kitchen layout.
_SLIDER_INTERACTION_LOCAL_XYZ = (0.0, -0.10, 0.031)

# Small spatial randomization perturbs both the toaster root and robot standing
# pose in the toaster interaction-site frame. The ranges are deliberately small so
# the fixed-root appliance stays on its authored countertop while testing pose
# robustness without introducing layout-specific world coordinates.
_M04_SPATIAL_PARAMETER_NAMES = (
    "toaster_x_offset_m",
    "toaster_y_offset_m",
    "toaster_yaw_offset_rad",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
_M04_PHYSICAL_PARAMETER_NAMES = (
    "slider_joint_friction",
    "slider_joint_damping",
)
_M04_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "toaster_x_offset_m",
                -0.015,
                0.015,
            ),
            GeneralizationParameterSpec(
                "toaster_y_offset_m",
                -0.015,
                0.015,
            ),
            GeneralizationParameterSpec(
                "toaster_yaw_offset_rad",
                -math.radians(2.0),
                math.radians(2.0),
            ),
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
            GeneralizationParameterSpec(
                "slider_joint_friction",
                10.03,
                10.18,
            ),
            GeneralizationParameterSpec(
                "slider_joint_damping",
                10.02,
                10.15,
            ),
        ),
        requires_rebuild=True,
    ),
)

# Reset/generalization validity. The slider must start at its lower joint limit,
# have a substantial margin to the 0.062-m success threshold, and be nearly still.
_INITIAL_SLIDER_MAX_LOWER_LIMIT_ERROR_M = 1e-3
_INITIAL_SLIDER_MIN_SUCCESS_MARGIN_M = 0.020
_INITIAL_SLIDER_MAX_SPEED_MPS = 0.01

_M04_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.20, -0.85, -0.80)
_M04_TCP_WORKSPACE_LOCAL_MAX_XYZ = (0.70, 0.85, 0.95)

# Temporary robot parking position while the layout settles without robot contact.
_M04_HIDDEN_Z_M = -20.0

@register_env(
    SETTINGS.env_id,
    max_episode_steps=SETTINGS.max_steps,
)
class TaskEnv(DexVLABenchBaseEnv):
    """Press the layout-owned toaster slider until joint_1 reaches 0.062 m."""

    TASK_SETTINGS = SETTINGS
    GENERALIZATION_SPEC = _M04_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = _INTERACTION_SITE_NAME
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(3.158571, -0.748507, 1.479300),
            quaternion_world_wxyz=(0.422706, -0.252594, 0.123941, 0.861484),
        ),
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.793508, -0.663946, 1.288185),
            quaternion_world_wxyz=(0.387754, -0.146708, 0.062660, 0.907853),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(4.009356, -0.660080, 1.431299),
            quaternion_world_wxyz=(0.356812, -0.288487, 0.116867, 0.880797),
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
        """Return the nominal or toaster-local randomized robot base pose."""

        nominal_pose = self.interaction_site_robot_base_sapien_pose()
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_pose

        values = self._generalization_values(
            _M04_SPATIAL_PARAMETER_NAMES,
            cache_name="_m04_spatial_values",
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

        # [1] * [B] -> [B]: perturb in the interaction-site frame, so the same
        # generalization remains meaningful when layouts rotate/move the toaster.
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
        del options
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
        # LayoutBuilder does not provide an episode qpos reset for this toaster.
        # Preserve the build-time nominal state so non-target mechanisms cannot
        # leak state from one episode into the next. joint_1 is overridden below
        # with its actual lower limit because that is the task's unpressed state.
        self._nominal_fixture_qpos = self.fixture.get_qpos().clone()

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
                f"M04 requires {JOINT_NAME!r} to be one-DoF, got {joint_dofs}."
            )
        return joint

    def _slider_interaction_position(self) -> Any:
        """Return the calibrated slider interaction point in world coordinates.

        The offset is expressed in the moving joint_1 child-link frame. This is
        deliberately different from adding a constant world-space vector.
        """

        local_offset = torch.as_tensor(
            _SLIDER_INTERACTION_LOCAL_XYZ,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        interaction_pose = Pose.create(self.target_link.pose) * Pose.create_from_pq(
            p=local_offset
        )
        return interaction_pose.p

    def _hand_is_near_slider(self) -> Any:
        """Return whether any fingertip or the palm is near the slider.

        The TCP is a controller frame near the wrist/palm and can remain outside
        the stage threshold while a Shadow Hand fingertip operates the slider.
        Evaluate the physical hand links that can actually reach it instead.
        """

        hand_links = tuple(getattr(self.agent, "tip_links", ()))
        palm_link = getattr(self.agent, "palm_link", None)
        if palm_link is not None:
            hand_links += (palm_link,)
        if not hand_links:
            raise RuntimeError(
                "M04 requires the robot agent to expose fingertip links through "
                "tip_links and/or a palm_link for reach-stage evaluation."
            )

        slider_position = self._slider_interaction_position().unsqueeze(1)
        hand_positions = torch.stack(
            [link.pose.p for link in hand_links],
            dim=1,
        )
        minimum_distance = torch.linalg.norm(
            hand_positions - slider_position,
            dim=-1,
        ).amin(dim=1)
        return minimum_distance < _DISTANCE_THRESHOLD_REACH_SLIDER_M

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
        """Return per-env toaster slider values for physical generalization."""

        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return {}

        values = self._generalization_values(
            _M04_PHYSICAL_PARAMETER_NAMES,
            cache_name="_m04_physical_values",
        )
        interaction_site = self.require_current_interaction_site()
        fixture_name = interaction_site.target_entity_name
        if not fixture_name:
            raise RuntimeError(
                "M04 toaster interaction site must bind a layout entity."
            )
        return {
            fixture_name: ArticulationBuildValues(
                joint_friction={
                    JOINT_NAME: tuple(
                        float(value)
                        for value in values["slider_joint_friction"]
                    )
                },
                joint_damping={
                    JOINT_NAME: tuple(
                        float(value)
                        for value in values["slider_joint_damping"]
                    )
                },
            )
        }

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

        self.agent.controller.reset()

    def _reset_and_randomize_fixture_pose(self, env_idx: Any) -> None:
        """Restore the authored toaster pose, then apply site-local jitter."""

        env_idx = torch.as_tensor(
            env_idx,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)
        batch_size = len(env_idx)
        nominal_pose = Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0),
                batch_size=batch_size,
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            self.fixture.set_pose(nominal_pose)
            return

        values = self._generalization_values(
            _M04_SPATIAL_PARAMETER_NAMES,
            cache_name="_m04_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        x_offset = values["toaster_x_offset_m"][value_indices]
        y_offset = values["toaster_y_offset_m"][value_indices]
        yaw_offset = values["toaster_yaw_offset_rad"][value_indices]
        local_position = np.stack(
            [x_offset, y_offset, np.zeros_like(x_offset)],
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
        # The interaction-site frame is the toaster's authored root frame. Compose
        # the perturbation locally so layout rotations do not change its meaning.
        randomized_pose = nominal_pose * Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
        )
        self.fixture.set_pose(randomized_pose)

    def _reset_slider_joint(self, env_idx: Any) -> None:
        """Restore joint_1 to its unpressed lower limit and clear qvel."""

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
        if bool(
            (
                ~torch.isfinite(lower_limit)
                | ~torch.isfinite(upper_limit)
                | (lower_limit >= upper_limit)
            ).any().item()
        ):
            raise RuntimeError(
                f"M04 requires finite ordered limits for {JOINT_NAME!r}; got "
                f"{limits.detach().cpu().tolist()}."
            )

        # The benchmark success remains the absolute joint_1 >= 0.062 m rule.
        # Reject assets whose nominal unpressed limit would already be too close.
        if bool(
            (
                lower_limit
                > _PRESS_SLIDER_JOINT_THRESHOLD_M
                - _INITIAL_SLIDER_MIN_SUCCESS_MARGIN_M
            ).any().item()
        ):
            raise RuntimeError(
                f"M04 unpressed {JOINT_NAME!r} lower limit is too close to the "
                f"{_PRESS_SLIDER_JOINT_THRESHOLD_M:.3f} m success threshold: "
                f"lower={lower_limit.detach().cpu().tolist()}."
            )

        fixture_qpos = self._nominal_fixture_qpos[env_idx].clone()
        fixture_qpos[:, self._target_joint_qpos_index] = lower_limit
        self.fixture.set_qpos(fixture_qpos)

        # Clear all toaster active-joint velocities so another mechanism cannot
        # carry residual motion from settle/previous episodes into evaluation.
        self.fixture.set_qvel(torch.zeros_like(fixture_qpos))

    def _initialize_episode(
        self,
        env_idx: Any,
        options: dict[str, Any],
    ) -> None:
        del options
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
            # LayoutBuilder does not episode-reset this fixed-root toaster because
            # it has no authored jitter/initial_qpos. Restore its authored pose
            # explicitly, then apply task spatial jitter in the toaster/site frame.
            self._reset_and_randomize_fixture_pose(env_idx_tensor)

            # Commit the unpressed state before settle so previous-episode or
            # authored joint state cannot affect the scene's settle trajectory.
            self._reset_slider_joint(env_idx_tensor)

            self.agent.robot.set_pose(
                sapien.Pose(p=[0.0, 0.0, _M04_HIDDEN_Z_M])
            )
            self._set_robot_to_rest_keyframe(batch_size)
            self.layout_scene.settle(env_idx=env_idx_tensor)

            # Settle can leave residual slider displacement/velocity. Restore the
            # exact unpressed state again; this becomes the episode baseline.
            self._reset_slider_joint(env_idx_tensor)

            self.agent.robot.set_pose(self._robot_base_pose(env_idx_tensor))
            self._set_robot_to_rest_keyframe(batch_size)


    def _get_joint_value(self) -> Any:
        # [B, D] -> [B]: select the configured one-DoF toaster slider joint.
        return self.target_joint.qpos.reshape(self.num_envs, -1)[:, 0]

    def _get_joint_velocity(self) -> Any:
        # [B, D] -> [B]: select the configured one-DoF toaster slider joint.
        return self.target_joint.qvel.reshape(self.num_envs, -1)[:, 0]

    def _slider_is_valid_initial_state(self) -> Any:
        joint_value = self._get_joint_value()
        joint_speed = self._get_joint_velocity()
        limits = self.target_joint.limits.reshape(
            self.num_envs,
            -1,
            2,
        )[:, 0]
        lower_limit = limits[:, 0]
        at_unpressed_limit = (
            torch.abs(joint_value - lower_limit)
            <= _INITIAL_SLIDER_MAX_LOWER_LIMIT_ERROR_M
        )
        clearly_below_success = (
            joint_value
            <= _PRESS_SLIDER_JOINT_THRESHOLD_M
            - _INITIAL_SLIDER_MIN_SUCCESS_MARGIN_M
        )
        nearly_static = (
            torch.abs(joint_speed) <= _INITIAL_SLIDER_MAX_SPEED_MPS
        )
        return at_unpressed_limit & clearly_below_success & nearly_static

    def _tcp_within_toaster_workspace(self) -> Any:
        # Define D5 in the actual toaster root frame, not in world coordinates or
        # the nominal site frame. The workspace therefore follows both layout pose
        # changes and this task's small toaster spatial randomization.
        toaster_pose = Pose.create(self.fixture.root_pose)
        tcp_local_position = (toaster_pose.inv() * self.agent.tcp.pose).p
        workspace_lower = torch.as_tensor(
            _M04_TCP_WORKSPACE_LOCAL_MIN_XYZ,
            device=self.device,
            dtype=torch.float32,
        )
        workspace_upper = torch.as_tensor(
            _M04_TCP_WORKSPACE_LOCAL_MAX_XYZ,
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
                self._slider_is_valid_initial_state()
                & self._tcp_within_toaster_workspace()
            )
        if axis is GeneralizationAxis.PHYSICAL:
            return self._slider_is_valid_initial_state()
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        joint_value = self._get_joint_value()
        success = joint_value >= _PRESS_SLIDER_JOINT_THRESHOLD_M
        # Preserve the original reach-toaster definition: TCP to fixture root.
        fixture_pos = self.fixture.root_pose.p
        toaster_dist = torch.linalg.norm(
            fixture_pos - self.agent.tcp.pose.p,
            dim=1,
        )
        reached_toaster = (
            toaster_dist < _DISTANCE_THRESHOLD_REACH_TOASTER_M
        )

        # Measure the fingertips/palm against the calibrated moving-link point.
        reached_slider = self._hand_is_near_slider()

        return {
            "success": success,
            "reach_toaster": reached_toaster,
            "reach_slider": reached_slider,
            "press_slider": success,
        }
