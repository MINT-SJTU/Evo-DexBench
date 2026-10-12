from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.scenes.layouts import ArticulationBuildValues, LayoutRegistry, PoseSpec
from dex_benchmark.sim.scenes.layouts.studio import STUDIO_PHYSICAL_MATERIAL
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


B11_SETTINGS = TaskRegistry.register(TaskRegistry.load("b11_v1"))
B11_LAPTOP_ENTITY_NAME = "laptop"
B11_LID_JOINT_NAME = "joint_1"
B11_BASE_LINK_NAME = "link_0"

B11_ROBOT_BASE_LOCAL_MIN_XYZ = (-0.71, -0.35, -0.80)
B11_ROBOT_BASE_LOCAL_MAX_XYZ = (-0.44, 0.35, 0.20)
B11_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-0.95, -0.85, -0.25)
B11_TCP_WORKSPACE_LOCAL_MAX_XYZ = (0.95, 0.85, 1.15)


# Recorded in studio L1 and stored relative to the bimanual robot-base center.
# Reapplying this pose in L1/L2/L3 preserves the camera-to-robot relationship.
B11_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.3517001172, 0.0713730109, 0.4851410548),
    quaternion_wxyz=(
        0.9367698401,
        -0.0010769998,
        0.3499319403,
        0.0029569995,
    ),
)
B11_LAYOUT_IDS = (
    LayoutId.STUDIO_L1,
    LayoutId.STUDIO_L2,
    LayoutId.STUDIO_L3,
)


def _b11_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B11_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            "desk_top"
        )
        robot_base_center_pose = (
            interaction_site.bimanual_robot_base_center_pose
            or interaction_site.robot_base_center_pose
        )
        camera_world_pose = robot_base_center_pose.transform_pose(
            B11_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
        )
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=camera_world_pose.quaternion_wxyz,
        )
    return camera_poses


B11_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("laptop_x_offset_m", -0.020, 0.020),
            GeneralizationParameterSpec("laptop_y_offset_m", -0.020, 0.020),
            GeneralizationParameterSpec(
                "laptop_yaw_offset_rad", -math.radians(8.0), math.radians(8.0)
            ),
            GeneralizationParameterSpec(
                "lid_initial_open_angle_deg", 100.0, 120.0
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("laptop_mass_log_scale", math.log(0.85), math.log(1.15)),
            GeneralizationParameterSpec("base_friction_log_scale", math.log(0.75), math.log(1.25)),
            GeneralizationParameterSpec("hinge_friction", 0.05, 0.18),
            GeneralizationParameterSpec("hinge_damping", 0.04, 0.20),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B11SuccessSpec:
    initial_open_angle_deg: float = 110.0
    middle_open_angle_deg: float = 60.0
    middle_tolerance_deg: float = 8.0
    target_open_angle_deg: float = 25.0
    target_tolerance_deg: float = 5.0
    stable_joint_speed_rad_s: float = 1.00
    stable_dwell_s: float = 0.25
    contact_force_n: float = 0.40
    base_displacement_m: float = 0.040
    base_tilt_deg: float = 5.0


@register_env(B11_SETTINGS.env_id, max_episode_steps=B11_SETTINGS.max_steps)
class B11Env(StudioTaskEnv):
    """Use two interchangeable hands to stabilize and close a laptop."""

    TASK_SETTINGS = B11_SETTINGS
    GENERALIZATION_SPEC = B11_GENERALIZATION_SPEC
    SUCCESS_SPEC = B11SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "desk_top"
    HUMAN_VIEWER_CAMERA_POSES = _b11_human_viewer_camera_poses()
    ROBOT_BASE_LOCAL_MIN_XYZ = B11_ROBOT_BASE_LOCAL_MIN_XYZ
    ROBOT_BASE_LOCAL_MAX_XYZ = B11_ROBOT_BASE_LOCAL_MAX_XYZ

    def human_viewer_camera_uid(self) -> str | None:
        """Keep B11's Human Viewer on its calibrated free camera."""

        return None

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            (
                "laptop_mass_log_scale",
                "base_friction_log_scale",
                "hinge_friction",
                "hinge_damping",
            ),
            cache_name="_b11_physical_values",
        )
        materials = tuple(
            scaled_physical_material(
                STUDIO_PHYSICAL_MATERIAL,
                friction_scale=float(scale),
                restitution=0.01,
            )
            for scale in positive_scale(values["base_friction_log_scale"])
        )
        return None, {
            B11_LAPTOP_ENTITY_NAME: ArticulationBuildValues(
                physical_materials=materials,
                # The builder scales the complete laptop articulation. This is
                # the available asset-level proxy for screen mass variation.
                mass_scales=tuple(
                    float(scale)
                    for scale in positive_scale(values["laptop_mass_log_scale"])
                ),
                joint_friction={
                    B11_LID_JOINT_NAME: tuple(
                        float(value) for value in values["hinge_friction"]
                    )
                },
                joint_damping={
                    B11_LID_JOINT_NAME: tuple(
                        float(value) for value in values["hinge_damping"]
                    )
                },
            )
        }

    def _bind_task_entities(self) -> None:
        self.laptop = self.layout_scene.require_entity(B11_LAPTOP_ENTITY_NAME)
        try:
            self.lid_joint = self.laptop.active_joints_map[B11_LID_JOINT_NAME]
            self.base_link = self.laptop.links_map[B11_BASE_LINK_NAME]
        except KeyError as exc:
            raise RuntimeError(
                "Laptop asset 9992 must expose link_0 and joint_1."
            ) from exc
        self.lid_link = self.lid_joint.child_link
        if self.lid_link is None:
            raise RuntimeError("Laptop joint_1 has no child screen link.")
        self.desk = self.layout_scene.require_entity("desk")

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._apply_laptop_spatial_generalization(env_idx, batch_size)
            self._set_initial_lid_angle(env_idx)
            self.layout_scene.settle(env_idx=env_idx)
            # Restore the requested angle after gravity settling and update the
            # finite-torque latch to the same qpos.
            self._set_initial_lid_angle(env_idx)
            self.layout_scene.reset_joint_position_latches(env_idx)
            self._hide_robots()
            self._reset_robots(batch_size)
            if not hasattr(self, "_laptop_initial_position"):
                self._laptop_initial_position = self.laptop.pose.p.clone()
                self._base_initial_up = self._base_up_vector().clone()
                self._lid_edge_touched = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
                self._left_stabilized_right_closed = torch.zeros_like(
                    self._lid_edge_touched
                )
                self._right_stabilized_left_closed = torch.zeros_like(
                    self._lid_edge_touched
                )
                self._lid_mid_reached = torch.zeros_like(self._lid_edge_touched)
                self._lid_target_reached = torch.zeros_like(self._lid_edge_touched)
                self._lid_mid_valid_last = torch.zeros_like(self._lid_edge_touched)
                self._lid_target_valid_last = torch.zeros_like(self._lid_edge_touched)
                self._stable_step_count = torch.zeros(
                    self.num_envs, dtype=torch.int64, device=self.device
                )
            self._laptop_initial_position[env_idx] = self.laptop.pose.p[env_idx]
            self._base_initial_up[env_idx] = self._base_up_vector()[env_idx]
            self._lid_edge_touched[env_idx] = False
            self._left_stabilized_right_closed[env_idx] = False
            self._right_stabilized_left_closed[env_idx] = False
            self._lid_mid_reached[env_idx] = False
            self._lid_target_reached[env_idx] = False
            self._lid_mid_valid_last[env_idx] = False
            self._lid_target_valid_last[env_idx] = False
            self._stable_step_count[env_idx] = 0

    def _apply_laptop_spatial_generalization(
        self, env_idx: Any, batch_size: int
    ) -> None:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return
        values = self._generalization_values(
            (
                "laptop_x_offset_m",
                "laptop_y_offset_m",
                "laptop_yaw_offset_rad",
            ),
            cache_name="_b11_spatial_values",
        )
        indices = env_idx.detach().cpu().numpy()
        site_pose = self._site_pose(batch_size)
        local_pose = site_pose.inv() * self.laptop.pose[env_idx]
        offset = torch.zeros((batch_size, 3), device=self.device)
        offset[:, 0] = torch.as_tensor(
            values["laptop_x_offset_m"][indices], device=self.device
        )
        offset[:, 1] = torch.as_tensor(
            values["laptop_y_offset_m"][indices], device=self.device
        )
        yaw = torch.as_tensor(
            values["laptop_yaw_offset_rad"][indices], device=self.device
        )
        yaw_q = torch.stack(
            [
                torch.cos(yaw / 2),
                torch.zeros_like(yaw),
                torch.zeros_like(yaw),
                torch.sin(yaw / 2),
            ],
            dim=1,
        )
        changed_pose = (
            Pose.create_from_pq(p=local_pose.p + offset, q=local_pose.q)
            * Pose.create_from_pq(q=yaw_q)
        )
        self.laptop.set_pose(site_pose * changed_pose)

    def _set_initial_lid_angle(self, env_idx: Any) -> None:
        qpos = self.laptop.get_qpos()
        limits = self.lid_joint.limits.reshape(self.num_envs, -1, 2)[:, 0]
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                ("lid_initial_open_angle_deg",),
                cache_name="_b11_initial_angle_values",
            )
            indices = env_idx.detach().cpu().numpy()
            angle_deg = torch.as_tensor(
                values["lid_initial_open_angle_deg"][indices],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            angle_deg = torch.full(
                (len(env_idx),),
                self.SUCCESS_SPEC.initial_open_angle_deg,
                device=self.device,
            )
        if self.lid_joint.active_index is None:
            raise RuntimeError("Laptop joint_1 has no active qpos index.")
        joint_range = (limits[env_idx, 1] - limits[env_idx, 0]).clamp_min(1e-6)
        qpos[env_idx, self.lid_joint.active_index] = (
            limits[env_idx, 0]
            + angle_deg / 180.0 * joint_range
        )
        self.laptop.set_qpos(qpos)
        qvel = self.laptop.get_qvel()
        qvel[env_idx] = 0.0
        self.laptop.set_qvel(qvel)

    def _opening_angle_deg(self) -> Any:
        qpos = self.lid_joint.qpos.reshape(self.num_envs, -1)[:, 0]
        limits = self.lid_joint.limits.reshape(self.num_envs, -1, 2)[:, 0]
        fraction = (qpos - limits[:, 0]) / (limits[:, 1] - limits[:, 0]).clamp_min(1e-6)
        return fraction.clamp(0.0, 1.0) * 180.0

    def _base_up_vector(self) -> Any:
        rotation = self.base_link.pose.to_transformation_matrix()[:, :3, :3]
        local_up = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float32
        )
        local_up[:, 1] = 1.0
        return torch.einsum("bij,bj->bi", rotation, local_up)

    def _base_is_stable(self) -> Any:
        displacement = torch.linalg.norm(
            self.laptop.pose.p - self._laptop_initial_position, dim=1
        )
        cosine = (
            self._base_up_vector() * self._base_initial_up
        ).sum(dim=1).clamp(-1.0, 1.0)
        tilt_deg = torch.rad2deg(torch.acos(cosine))
        return (
            displacement <= self.SUCCESS_SPEC.base_displacement_m
        ) & (tilt_deg <= self.SUCCESS_SPEC.base_tilt_deg)

    def _lid_contact_by_agent(self) -> Any:
        per_agent_contact: list[Any] = []
        for agent in self._manipulator_agents():
            forces = agent.get_object_taxel_world_forces(
                (("lid", (self.lid_link,)),)
            )["lid"]
            force_norms = torch.linalg.norm(forces, dim=-1)
            per_agent_contact.append(
                force_norms.amax(dim=1) >= self.SUCCESS_SPEC.contact_force_n
            )
        if len(per_agent_contact) != 2:
            raise RuntimeError("B11 requires exactly two manipulator agents.")
        # Two [B] vectors -> [B, 2] in embodiment role order (left, right).
        return torch.stack(per_agent_contact, dim=1)

    def _base_contact_by_agent(self) -> Any:
        per_agent_contact: list[Any] = []
        target_groups = (("base", (self.base_link,)),)
        for agent in self._manipulator_agents():
            forces = agent.get_object_taxel_world_forces(target_groups)["base"]
            parent_forces = agent.aggregate_taxel_world_forces_by_parent(forces)
            # [B, T, 3] -> [B, P, 3] -> [B]: strongest base contact per hand.
            per_agent_contact.append(
                torch.linalg.norm(parent_forces, dim=-1).amax(dim=1)
                >= self.SUCCESS_SPEC.contact_force_n
            )
        if len(per_agent_contact) != 2:
            raise RuntimeError("B11 requires exactly two manipulator agents.")
        return torch.stack(per_agent_contact, dim=1)

    def _cooperative_contact(self) -> tuple[Any, Any, Any]:
        lid_by_agent = self._lid_contact_by_agent()
        base_by_agent = self._base_contact_by_agent()
        left_base_right_lid = base_by_agent[:, 0] & lid_by_agent[:, 1]
        right_base_left_lid = base_by_agent[:, 1] & lid_by_agent[:, 0]
        cooperative = left_base_right_lid | right_base_left_lid
        return cooperative, left_base_right_lid, right_base_left_lid

    def evaluate(self) -> dict[str, Any]:
        opening_angle = self._opening_angle_deg()
        (
            cooperative_contact,
            left_base_right_edge,
            right_base_left_edge,
        ) = self._cooperative_contact()
        base_stable = self._base_is_stable()
        joint_velocity = self.lid_joint.qvel.reshape(self.num_envs, -1)[:, 0]
        middle_current = (
            torch.abs(opening_angle - self.SUCCESS_SPEC.middle_open_angle_deg)
            <= self.SUCCESS_SPEC.middle_tolerance_deg
        )
        target_current = (
            torch.abs(opening_angle - self.SUCCESS_SPEC.target_open_angle_deg)
            <= self.SUCCESS_SPEC.target_tolerance_deg
        )
        # `_lid_edge_touched` and the public `lid_edge_contacted` stage name are
        # retained for recording compatibility. They now mean contact anywhere
        # on the lid; no hand-taxel position/remote-edge mask is applied.
        lid_touched_before = self._lid_edge_touched.clone()
        middle_reached_before = self._lid_mid_reached.clone()
        valid_cooperation = cooperative_contact & base_stable
        self._lid_edge_touched |= valid_cooperation
        self._left_stabilized_right_closed |= left_base_right_edge & base_stable
        self._right_stabilized_left_closed |= right_base_left_edge & base_stable
        entered_middle = middle_current & ~self._lid_mid_valid_last
        self._lid_mid_reached |= entered_middle & lid_touched_before
        entered_target = target_current & ~self._lid_target_valid_last
        self._lid_target_reached |= entered_target & middle_reached_before
        self._lid_mid_valid_last = middle_current
        self._lid_target_valid_last = target_current
        joint_speed = torch.abs(joint_velocity)
        stable_now = (
            self._lid_target_reached
            & target_current
            & base_stable
            & (joint_speed <= self.SUCCESS_SPEC.stable_joint_speed_rad_s)
        )
        stable_steps = max(
            1,
            int(math.ceil(self.SUCCESS_SPEC.stable_dwell_s * self._control_freq)),
        )
        self._stable_step_count = torch.where(
            stable_now,
            self._stable_step_count + 1,
            torch.zeros_like(self._stable_step_count),
        )
        success = self._stable_step_count >= stable_steps
        return {
            "success": success,
            "lid_edge_contacted": self._lid_edge_touched,
            "lid_mid_angle_reached": self._lid_mid_reached,
            "lid_target_angle_stable": success,
            "base_stable": base_stable,
            "left_stabilized_right_closed": self._left_stabilized_right_closed,
            "right_stabilized_left_closed": self._right_stabilized_left_closed,
        }

    def _check_generalization_validity(
        self, axis: GeneralizationAxis | None
    ) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            finite = (
                torch.isfinite(self.laptop.pose.raw_pose).all(dim=1)
                & torch.isfinite(self.lid_link.pose.raw_pose).all(dim=1)
            )
            valid_initial = self._opening_angle_deg() >= 90.0
            return finite & valid_initial & self._base_is_stable()
        return super()._check_generalization_validity(axis)
