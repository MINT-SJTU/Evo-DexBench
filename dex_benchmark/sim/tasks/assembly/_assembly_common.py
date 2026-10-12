from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Iterable

import numpy as np

from dex_benchmark.sim.envs.base_env import DexVLABenchBaseEnv
from dex_benchmark.sim.scenes.layouts import ExternalSceneSpec, LayoutBuilder, PhysicalMaterialSpec

try:
    import sapien
    import torch
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    torch = None
    Pose = None


AXIS_X_TO_Z_QUATERNION_WXYZ = (
    math.sqrt(0.5),
    0.0,
    -math.sqrt(0.5),
    0.0,
)


def require_simulator() -> None:
    if sapien is None or torch is None or Pose is None:
        raise ImportError("SAPIEN, ManiSkill, and PyTorch are required for assembly tasks.")


def physical_material(material_spec: PhysicalMaterialSpec) -> Any:
    require_simulator()
    return sapien.physx.PhysxMaterial(
        static_friction=material_spec.static_friction,
        dynamic_friction=material_spec.dynamic_friction,
        restitution=material_spec.restitution,
    )


def render_material(
    color_rgba: tuple[float, float, float, float],
    *,
    roughness: float = 0.45,
    specular: float = 0.25,
) -> Any:
    require_simulator()
    return sapien.render.RenderMaterial(
        base_color=list(color_rgba),
        roughness=roughness,
        specular=specular,
    )


def add_z_cylinder(
    builder: Any,
    *,
    center_xyz: tuple[float, float, float],
    radius_m: float,
    half_height_m: float,
    collision_material: Any,
    visual_material: Any,
    density_kg_m3: float,
    name: str,
) -> None:
    """Add a cylinder whose long axis is actor-local Z."""

    if radius_m <= 0.0 or half_height_m <= 0.0:
        raise ValueError("Cylinder radius and half-height must be positive.")
    local_pose = sapien.Pose(
        p=center_xyz,
        q=AXIS_X_TO_Z_QUATERNION_WXYZ,
    )
    builder.add_cylinder_collision(
        pose=local_pose,
        radius=radius_m,
        half_length=half_height_m,
        material=collision_material,
        density=density_kg_m3,
    )
    builder.add_cylinder_visual(
        pose=local_pose,
        radius=radius_m,
        half_length=half_height_m,
        material=visual_material,
        name=name,
    )


def add_box(
    builder: Any,
    *,
    center_xyz: tuple[float, float, float],
    half_size_xyz: tuple[float, float, float],
    collision_material: Any,
    visual_material: Any,
    density_kg_m3: float,
    name: str,
    yaw_rad: float = 0.0,
) -> None:
    if any(value <= 0.0 for value in half_size_xyz):
        raise ValueError(f"Box half sizes must be positive, got {half_size_xyz}.")
    local_pose = sapien.Pose(
        p=center_xyz,
        q=(math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)),
    )
    builder.add_box_collision(
        pose=local_pose,
        half_size=half_size_xyz,
        material=collision_material,
        density=density_kg_m3,
    )
    builder.add_box_visual(
        pose=local_pose,
        half_size=half_size_xyz,
        material=visual_material,
        name=name,
    )


def add_annular_prism(
    builder: Any,
    *,
    center_z_m: float,
    inner_radius_m: float,
    outer_radius_m: float,
    half_height_m: float,
    segment_count: int,
    collision_material: Any,
    visual_material: Any,
    density_kg_m3: float,
    name_prefix: str,
) -> None:
    """Approximate a hollow coaxial ring with overlapping tangent boxes."""

    if not 0.0 < inner_radius_m < outer_radius_m:
        raise ValueError(
            "Annular radii must satisfy 0 < inner < outer, got "
            f"{inner_radius_m}, {outer_radius_m}."
        )
    if segment_count < 8:
        raise ValueError("Annular collision requires at least eight segments.")
    middle_radius = 0.5 * (inner_radius_m + outer_radius_m)
    radial_half_size = 0.5 * (outer_radius_m - inner_radius_m)
    tangential_half_size = (
        middle_radius * math.tan(math.pi / segment_count) * 1.08
    )
    for segment_index in range(segment_count):
        theta = 2.0 * math.pi * segment_index / segment_count
        add_box(
            builder,
            center_xyz=(
                middle_radius * math.cos(theta),
                middle_radius * math.sin(theta),
                center_z_m,
            ),
            half_size_xyz=(
                radial_half_size,
                tangential_half_size,
                half_height_m,
            ),
            collision_material=collision_material,
            visual_material=visual_material,
            density_kg_m3=density_kg_m3,
            name=f"{name_prefix}_{segment_index}",
            yaw_rad=theta,
        )


def local_z_axis_world(pose: Any) -> Any:
    """Return actor local +Z expressed in world coordinates, shape [B, 3]."""

    quaternion = pose.q
    qw, qx, qy, qz = quaternion.unbind(dim=1)
    return torch.stack(
        (
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
        dim=1,
    )


def transform_local_point(
    pose: Any,
    local_xyz: tuple[float, float, float],
    *,
    device: Any,
    batch_size: int,
) -> Any:
    local_point = torch.as_tensor(
        local_xyz,
        device=device,
        dtype=torch.float32,
    ).repeat(batch_size, 1)
    return (pose * Pose.create_from_pq(p=local_point)).p


class BimanualAssemblyTaskEnv(DexVLABenchBaseEnv):
    """Shared runtime for two-object, bimanual insertion tasks."""

    REMOVED_LAYOUT_ITEMS: frozenset[str] = frozenset()
    SUPPORT_ENTITY_NAME: str | None = None

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self.interaction_site_robot_base_sapien_poses())

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        retained_items = tuple(
            item
            for item in self.require_current_layout_spec().build_items
            if getattr(item, "name", None) not in self.REMOVED_LAYOUT_ITEMS
        )
        task_layout = replace(
            self.require_current_layout_spec(),
            build_items=retained_items,
        )
        multi_agent = self.agent
        try:
            if any(
                isinstance(item, ExternalSceneSpec)
                for item in task_layout.build_items
            ):
                # RoboCasa scene builders inspect a single optional agent during
                # construction; the task restores its bimanual wrapper afterward.
                self.agent = None
            self.layout_scene = LayoutBuilder(
                env=self,
                layout_spec=task_layout,
            ).build()
        finally:
            self.agent = multi_agent
        self._build_task_actors()

    def _build_task_actors(self) -> None:
        raise NotImplementedError

    def _dual_agents(self) -> tuple[Any, Any]:
        return self.left_agent, self.right_agent

    def _site_pose(self, batch_size: int) -> Any:
        return Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

    def _reset_robots(self, batch_size: int) -> None:
        robot_base_poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(robot_base_poses, list):
            robot_base_poses = [robot_base_poses]
        agents = self._dual_agents()
        if len(robot_base_poses) != len(agents):
            raise RuntimeError(
                f"Expected {len(agents)} robot base poses, got "
                f"{len(robot_base_poses)}."
            )
        for agent, base_pose in zip(agents, robot_base_poses, strict=True):
            rest_keyframe = agent.keyframes.get("rest")
            if rest_keyframe is None or rest_keyframe.qpos is None:
                raise RuntimeError(f"Agent {agent.uid!r} has no rest keyframe.")
            rest_qpos = torch.as_tensor(
                rest_keyframe.qpos,
                device=self.device,
                dtype=torch.float32,
            )
            agent.reset(rest_qpos.unsqueeze(0).repeat(batch_size, 1))
            agent.robot.set_pose(base_pose)

    def _hide_robots(self) -> None:
        for agent in self._dual_agents():
            agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))

    def _zero_actor_velocity(self, actor: Any, batch_size: int) -> None:
        zero = torch.zeros((batch_size, 3), device=self.device, dtype=torch.float32)
        actor.set_linear_velocity(zero)
        actor.set_angular_velocity(zero)

    def _agent_touch_force(self, agent: Any, targets: Iterable[Any]) -> Any:
        target_links = tuple(
            link
            for target in targets
            for link in tuple(getattr(target, "links", None) or (target,))
        )
        target_groups = (("targets", target_links),)
        taxel_forces = agent.get_object_taxel_world_forces(target_groups)["targets"]
        # [B, N, 3] -> [B, P, 3] -> [B].
        parent_forces = agent.aggregate_taxel_world_forces_by_parent(taxel_forces)
        return torch.linalg.norm(parent_forces, dim=-1).amax(dim=1)

    def _max_touch_force(self, targets: Iterable[Any]) -> Any:
        maximum = torch.zeros(self.num_envs, device=self.device)
        for agent in self._dual_agents():
            maximum = torch.maximum(
                maximum,
                self._agent_touch_force(agent, targets),
            )
        return maximum

    def _pair_contact_force(self, first: Any, second: Any) -> Any:
        return torch.linalg.norm(
            self.scene.get_pairwise_contact_forces(first, second),
            dim=1,
        )

    @staticmethod
    def _actor_static(
        actor: Any,
        *,
        max_linear_speed_mps: float = 0.05,
        max_angular_speed_rad_s: float = 0.8,
    ) -> Any:
        return (
            torch.linalg.norm(actor.linear_velocity, dim=1)
            <= max_linear_speed_mps
        ) & (
            torch.linalg.norm(actor.angular_velocity, dim=1)
            <= max_angular_speed_rad_s
        )

    def _agent_grasp_contact(
        self,
        agent: Any,
        target: Any,
        minimum_force_n: float,
    ) -> Any:
        """Require contact on at least two hand links, rather than one touch."""

        links = tuple(getattr(target, "links", None) or (target,))
        forces = agent.get_object_taxel_world_forces((("grasp", links),))["grasp"]
        parents = agent.aggregate_taxel_world_forces_by_parent(forces)
        return (
            torch.linalg.norm(parents, dim=-1) >= minimum_force_n
        ).sum(dim=1) >= 2

    def _opposite_hand_grasp(
        self,
        first: Any,
        second: Any,
        *,
        minimum_force_n: float,
    ) -> tuple[Any, Any, Any]:
        held_now = torch.stack(
            [
                self._agent_grasp_contact(agent, target, minimum_force_n)
                for target in (first, second)
                for agent in self._dual_agents()
            ],
            dim=1,
        )
        if not hasattr(self, "_assembly_grasp_steps"):
            self._assembly_grasp_steps = torch.zeros_like(held_now, dtype=torch.int64)
        self._assembly_grasp_steps = torch.where(
            held_now,
            self._assembly_grasp_steps + 1,
            torch.zeros_like(self._assembly_grasp_steps),
        )
        sustained = self._assembly_grasp_steps >= max(
            1, math.ceil(0.1 * self._control_freq)
        )
        left_first, right_first, left_second, right_second = sustained.unbind(dim=1)
        first_grasped = left_first | right_first
        second_grasped = left_second | right_second
        opposite_hands = (left_first & right_second) | (right_first & left_second)
        return first_grasped, second_grasped, opposite_hands

    def _apply_actor_physics(
        self,
        actor: Any,
        *,
        base_material: PhysicalMaterialSpec,
        friction_scales: np.ndarray,
        mass_scales: np.ndarray,
    ) -> None:
        if friction_scales.shape != (self.num_envs,):
            raise ValueError("Friction scales must have shape [num_envs].")
        if mass_scales.shape != (self.num_envs,):
            raise ValueError("Mass scales must have shape [num_envs].")
        for env_index, body in enumerate(actor._bodies):
            mass_scale = float(mass_scales[env_index])
            body.mass = float(body.mass) * mass_scale
            body.inertia = np.asarray(body.inertia) * mass_scale
            friction_scale = float(friction_scales[env_index])
            material = sapien.physx.PhysxMaterial(
                static_friction=base_material.static_friction * friction_scale,
                dynamic_friction=base_material.dynamic_friction * friction_scale,
                restitution=base_material.restitution,
            )
            for shape in body.collision_shapes:
                shape.set_physical_material(material)

    def _robot_joint_and_workspace_safety(
        self,
        *,
        workspace_local_min_xyz: tuple[float, float, float],
        workspace_local_max_xyz: tuple[float, float, float],
        max_speed_rad_s: float = 6.0,
    ) -> Any:
        unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        site_inverse = self._site_pose(self.num_envs).inv()
        lower = torch.as_tensor(
            workspace_local_min_xyz,
            device=self.device,
            dtype=torch.float32,
        )
        upper = torch.as_tensor(
            workspace_local_max_xyz,
            device=self.device,
            dtype=torch.float32,
        )
        for agent in self._dual_agents():
            qpos = agent.robot.get_qpos()
            qvel = agent.robot.get_qvel()
            limits = agent.robot.get_qlimits()
            finite_limits = torch.isfinite(limits[:, :, 0]) & torch.isfinite(
                limits[:, :, 1]
            )
            outside_limits = (
                (
                    (qpos < limits[:, :, 0] - 1e-3)
                    | (qpos > limits[:, :, 1] + 1e-3)
                )
                & finite_limits
            ).any(dim=1)
            tcp_local = (site_inverse * agent.tcp.pose).p
            outside_workspace = ((tcp_local < lower) | (tcp_local > upper)).any(
                dim=1
            )
            unsafe |= (
                outside_limits
                | outside_workspace
                | (torch.abs(qvel).amax(dim=1) > max_speed_rad_s)
            )
        return unsafe
