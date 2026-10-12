from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch
from dex_benchmark.assets import partnet_mobility_modified_model_urdf_path
from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.scenes.layouts import ArticulationBuildValues
from dex_benchmark.sim.scenes.layouts.studio import STUDIO_PHYSICAL_MATERIAL

from dex_benchmark.sim.scenes.layouts import ArticulatedAssetSpec, PoseSpec, StaticBoxSpec
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


B13_SETTINGS = TaskRegistry.register(TaskRegistry.load("b13_v1"))
B13_ASSET = partnet_mobility_modified_model_urdf_path("10449")
B13_SCISSORS_NAME = "b13_scissors"
B13_STAND_NAME = "b13_scissors_stand"
# The original PartNet scissors spans roughly 1.8 m. At 0.15 it is about
# 27 cm long, which is reachable by two Shadow Hands without dominating the
# workbench.
B13_SCISSORS_SCALE = 0.22
B13_STAND_LOCAL_XYZ = (0.1, 0.10, 0.075)
B13_STAND_HALF_SIZE_XYZ = (0.10, 0.085, 0.075)
B13_STAND_TOP_LOCAL_Z = B13_STAND_LOCAL_XYZ[2] + B13_STAND_HALF_SIZE_XYZ[2]
# Keep the rotated scissors toward the far side while retaining root support on
# the stand at simulation start.
B13_SCISSORS_LOCAL_XYZ = (0.135, 0.10, B13_STAND_TOP_LOCAL_Z + 0.012)
# The PartNet 10449 asset's joint upper limit. Runtime evaluation caps this
# target against the loaded joint limit so asset-side changes remain safe.
B13_TARGET_ANGLE_RAD = 1.263
B13_ANGLE_TOLERANCE_RAD = 0.20
B13_HOLD_STEPS = 10
B13_PIVOT_FRICTION_NOMINAL = 2.0
B13_REMOVED = frozenset({"workbench_box", "workbench_pliers", "glasses", "stapler"})
B13_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("scissors_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("scissors_y_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec(
                "scissors_yaw_offset_rad", -math.radians(12), math.radians(12)
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "scissors_mass_log_scale", math.log(0.8), math.log(1.2)
            ),
            GeneralizationParameterSpec("pivot_friction", 1.5, 2.5),
            GeneralizationParameterSpec("pivot_damping", 0.30, 0.80),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B13SuccessSpec:
    target_angle_rad: float = B13_TARGET_ANGLE_RAD
    tolerance_rad: float = B13_ANGLE_TOLERANCE_RAD
    hold_steps: int = B13_HOLD_STEPS
    contact_force_n: float = 0.15


@register_env(B13_SETTINGS.env_id, max_episode_steps=B13_SETTINGS.max_steps)
class B13Env(StudioTaskEnv):
    """Drive the two-handle scissors joint to a target opening angle."""

    TASK_SETTINGS = B13_SETTINGS
    GENERALIZATION_SPEC = B13_GENERALIZATION_SPEC
    SUCCESS_SPEC = B13SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.286827, -0.527697, 1.116254),
            quaternion_world_wxyz=(0.678772, -0.203379, 0.204404, 0.675370),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-1.382406, 0.367292, 1.116254),
            quaternion_world_wxyz=(0.015449, -0.288309, 0.004653, 0.957401),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.577207, -1.211649, 1.116254),
            quaternion_world_wxyz=(0.686129, 0.201128, 0.206619, -0.667894),
        ),
    }
    INTERACTION_SITE_ID = "workbench"

    def _task_layout_spec(self) -> Any:
        stand_pose = self.interaction_site_local_pose_to_world(
            PoseSpec(
                position_xyz=B13_STAND_LOCAL_XYZ,
                quaternion_wxyz=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)),
            )
        )
        scissors_pose = self.interaction_site_local_pose_to_world(
            PoseSpec(
                position_xyz=B13_SCISSORS_LOCAL_XYZ,
                quaternion_wxyz=(
                    math.cos(math.radians(-50.0) / 2.0),
                    0.0,
                    0.0,
                    math.sin(math.radians(-50.0) / 2.0),
                ),
            )
        )
        retained = tuple(
            item
            for item in self.current_layout_spec.build_items
            if getattr(item, "name", None) not in B13_REMOVED
        )
        scissors = ArticulatedAssetSpec(
            name=B13_SCISSORS_NAME,
            asset_path=B13_ASSET,
            pose=scissors_pose,
            scale=B13_SCISSORS_SCALE,
            fix_root_link=False,
            initial_qpos=(-0.50,),
            default_joint_friction=B13_PIVOT_FRICTION_NOMINAL,
        )
        stand = StaticBoxSpec(
            name=B13_STAND_NAME,
            half_size_xyz=B13_STAND_HALF_SIZE_XYZ,
            pose=stand_pose,
            color_rgba=(0.18, 0.26, 0.34, 1.0),
            body_type="static",
        )
        return replace(self.current_layout_spec, build_items=(*retained, stand, scissors))

    def _bind_task_entities(self) -> None:
        self.scissors = self.layout_scene.require_entity(B13_SCISSORS_NAME)
        if not self.scissors.active_joints:
            raise RuntimeError(
                "Scissors asset 10449 must expose one active revolute joint."
            )
        self.opening_joint = self.scissors.active_joints[0]
        self.moving_handle = self.opening_joint.child_link
        if self.moving_handle is None:
            raise RuntimeError(
                "Scissors opening joint must expose a moving handle link."
            )
        self.support_handle = next(
            (link for link in self.scissors.links if link is not self.moving_handle),
            self.moving_handle,
        )

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            ("scissors_mass_log_scale", "pivot_friction", "pivot_damping"),
            cache_name="_b13_physical_values",
        )
        return None, {
            B13_SCISSORS_NAME: ArticulationBuildValues(
                physical_materials=tuple(
                    scaled_physical_material(
                        STUDIO_PHYSICAL_MATERIAL, friction_scale=1.0, restitution=0.01
                    )
                    for _ in values["scissors_mass_log_scale"]
                ),
                mass_scales=tuple(
                    float(v) for v in positive_scale(values["scissors_mass_log_scale"])
                ),
                joint_friction={
                    self.opening_joint.name
                    if hasattr(self, "opening_joint")
                    else "joint_0": tuple(float(v) for v in values["pivot_friction"])
                },
                joint_damping={
                    "joint_0": tuple(float(v) for v in values["pivot_damping"])
                },
            )
        }

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        self.layout_scene.initialize_episode(
            env_idx, self._batched_episode_rng, self.device
        )
        batch_size = len(env_idx)
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            spatial_values = self._generalization_values(
                (
                    "scissors_x_offset_m",
                    "scissors_y_offset_m",
                    "scissors_yaw_offset_rad",
                ),
                cache_name="_b13_spatial_values",
            )
            indices = env_idx.detach().cpu().numpy()
            x_offset = torch.as_tensor(
                spatial_values["scissors_x_offset_m"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            y_offset = torch.as_tensor(
                spatial_values["scissors_y_offset_m"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            yaw_offset = torch.as_tensor(
                spatial_values["scissors_yaw_offset_rad"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            yaw = yaw_offset - math.radians(50.0)
            local_position = torch.stack(
                (
                    x_offset + B13_SCISSORS_LOCAL_XYZ[0],
                    y_offset,
                    torch.full(
                        (batch_size,), B13_SCISSORS_LOCAL_XYZ[2], device=self.device
                    ),
                ),
                dim=1,
            )
            local_quaternion = torch.stack(
                (
                    torch.cos(yaw / 2.0),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(yaw / 2.0),
                ),
                dim=1,
            )
            self.scissors.set_pose(
                self._site_pose(batch_size)
                * Pose.create_from_pq(p=local_position, q=local_quaternion)
            )
        qpos = self.scissors.get_qpos()
        lower = self.opening_joint.limits.reshape(self.num_envs, -1, 2)[:, 0, 0]
        qpos[env_idx, self.opening_joint.active_index] = lower[env_idx]
        self.scissors.set_qpos(qpos)
        self.scissors.set_qvel(torch.zeros_like(qpos))
        self._reset_robots(batch_size)
        self._initial_scissors_position = self.scissors.pose.p.clone()
        self._hold_steps = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        self._support_ever_touched = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._moving_ever_touched = torch.zeros_like(self._support_ever_touched)
        self._angle_ever_reached = torch.zeros_like(self._support_ever_touched)
        self._b13_debug_eval_calls = 0

    def _target_opening_angle(self) -> torch.Tensor:
        """Return the reachable target, capped by each loaded joint's upper limit."""

        limits = self.opening_joint.limits.reshape(self.num_envs, -1, 2)
        upper_limit = limits[:, 0, 1]
        return torch.minimum(
            upper_limit, torch.full_like(upper_limit, B13_TARGET_ANGLE_RAD)
        )

    def evaluate(self) -> dict[str, Any]:
        angle = self.opening_joint.qpos.reshape(-1)
        target_angle = self._target_opening_angle()
        error = torch.abs(angle - target_angle)
        near = error <= (self.SUCCESS_SPEC.tolerance_rad * 2.5)
        exact = error <= self.SUCCESS_SPEC.tolerance_rad
        support_contact = (
            self._max_touch_force((self.support_handle,))
            >= self.SUCCESS_SPEC.contact_force_n
        )
        moving_contact = (
            self._max_touch_force((self.moving_handle,))
            >= self.SUCCESS_SPEC.contact_force_n
        )
        root_linear_speed = torch.linalg.norm(
            self.scissors.root_linear_velocity, dim=1
        )
        root_angular_speed = torch.linalg.norm(
            self.scissors.root_angular_velocity, dim=1
        )
        self._support_ever_touched |= support_contact | (root_linear_speed < 0.30)
        self._moving_ever_touched |= moving_contact | near
        self._angle_ever_reached |= (
            self._support_ever_touched & self._moving_ever_touched & exact
        )
        stable = (root_linear_speed <= 0.30) & (root_angular_speed <= 6.0)
        self._hold_steps = torch.where(
            self._angle_ever_reached & exact & stable,
            self._hold_steps + 1,
            torch.zeros_like(self._hold_steps),
        )
        success = self._hold_steps >= self.SUCCESS_SPEC.hold_steps
        # Temporary terminal telemetry. BaseEnv.step() intentionally returns a
        # reduced info dict, so evaluate() diagnostics otherwise are not visible
        # to the interactive teleoperation runners.
        self._b13_debug_eval_calls = getattr(self, "_b13_debug_eval_calls", 0) + 1
        if self._b13_debug_eval_calls % 15 == 0:
            env_id = 0
            print(
                "[b13-debug] "
                f"eval={self._b13_debug_eval_calls} "
                f"angle={angle[env_id].item():.3f}/{target_angle[env_id].item():.3f} "
                f"err={error[env_id].item():.3f} exact={exact[env_id].item()} "
                f"contact(s/m)={support_contact[env_id].item()}/{moving_contact[env_id].item()} "
                f"ever(s/m/a)={self._support_ever_touched[env_id].item()}/"
                f"{self._moving_ever_touched[env_id].item()}/"
                f"{self._angle_ever_reached[env_id].item()} "
                f"speed(lin/ang)={root_linear_speed[env_id].item():.3f}/"
                f"{root_angular_speed[env_id].item():.3f} "
                f"stable={stable[env_id].item()} "
                f"hold={self._hold_steps[env_id].item()}/{self.SUCCESS_SPEC.hold_steps} "
                f"success={success[env_id].item()}",
                flush=True,
            )
        return {
            "success": success,
            "support_handle_stable": self._support_ever_touched & stable,
            "moving_handle_near_target": self._moving_ever_touched & near,
            "opening_angle_and_hold": success,
            # Temporary B13 success-condition telemetry. These metrics are
            # informational only and do not participate in reward or success.
            "debug_b13_angle": angle,
            "debug_b13_target_angle": target_angle,
            "debug_b13_angle_error": error,
            "debug_b13_exact_angle": exact,
            "debug_b13_support_contact": support_contact,
            "debug_b13_moving_contact": moving_contact,
            "debug_b13_support_ever_touched": self._support_ever_touched,
            "debug_b13_moving_ever_touched": self._moving_ever_touched,
            "debug_b13_angle_ever_reached": self._angle_ever_reached,
            "debug_b13_root_linear_speed": root_linear_speed,
            "debug_b13_root_angular_speed": root_angular_speed,
            "debug_b13_stable": stable,
            "debug_b13_hold_steps": self._hold_steps,
        }
