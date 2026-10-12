from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable

import numpy as np

from dex_benchmark.sim.envs.base_env import DexVLABenchBaseEnv
from dex_benchmark.sim.scenes.layouts import LayoutBuilder
from dex_benchmark.sim.scenes.layouts.builder import pose_spec_to_sapien_pose

try:
    import sapien
    import torch
except Exception:  # pragma: no cover - optional simulator stack
    sapien = None
    torch = None


class StudioTaskEnv(DexVLABenchBaseEnv):
    """Small, shared runtime for Studio manipulation tasks.

    Subclasses bind their task entities after ``LayoutBuilder`` completes and
    implement task-specific initialization/evaluation. This class deliberately
    contains no task thresholds or semantic shortcuts.
    """

    ROBOT_BASE_LOCAL_MIN_XYZ: tuple[float, float, float] | None = None
    ROBOT_BASE_LOCAL_MAX_XYZ: tuple[float, float, float] | None = None

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_poses_for_task())

    def _robot_base_poses_for_task(self) -> Any:
        role_poses = self.interaction_site_robot_base_poses()
        adjusted_poses = [
            role_poses[role] for role in self.embodiment_spec.manipulator_roles
        ]
        if self.INTERACTION_SITE_ID not in {"desk_top", "workbench"}:
            m14_base_z = self.current_layout_spec.interaction_site(
                "desk_drawer"
            ).robot_base_center_pose.position_xyz[2]
            adjusted_poses = [
                replace(
                    pose,
                    position_xyz=(
                        pose.position_xyz[0],
                        pose.position_xyz[1],
                        m14_base_z,
                    ),
                )
                for pose in adjusted_poses
            ]
        self._validate_robot_base_pose_specs(tuple(adjusted_poses))
        sapien_poses = [pose_spec_to_sapien_pose(pose) for pose in adjusted_poses]
        return sapien_poses[0] if len(sapien_poses) == 1 else sapien_poses

    def _validate_robot_base_pose_specs(self, poses: tuple[Any, ...]) -> None:
        lower_values = self.ROBOT_BASE_LOCAL_MIN_XYZ
        upper_values = self.ROBOT_BASE_LOCAL_MAX_XYZ
        if lower_values is None and upper_values is None:
            return
        if lower_values is None or upper_values is None:
            raise RuntimeError(
                f"{type(self).__name__} must define both robot-base bounds."
            )
        site_frame = self.interaction_site_frame_pose()
        frame_position = np.asarray(site_frame.position_xyz, dtype=np.float64)
        frame_rotation = site_frame.rotation_matrix.astype(np.float64)
        lower = np.asarray(lower_values, dtype=np.float64)
        upper = np.asarray(upper_values, dtype=np.float64)
        for manipulator_index, pose in enumerate(poses):
            local_position = frame_rotation.T @ (
                np.asarray(pose.position_xyz, dtype=np.float64) - frame_position
            )
            if not ((local_position >= lower) & (local_position <= upper)).all():
                raise RuntimeError(
                    f"{type(self).__name__} robot {manipulator_index} base is "
                    "outside task-local limits: "
                    f"position={local_position.tolist()}, lower={lower.tolist()}, "
                    f"upper={upper.tolist()}."
                )

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        actor_values, articulation_values = self._physical_build_values()
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self._task_layout_spec(),
            actor_values=actor_values,
            articulation_values=articulation_values,
        ).build()
        self._bind_task_entities()

    def _task_layout_spec(self) -> Any:
        return self.current_layout_spec

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        return None, None

    def _bind_task_entities(self) -> None:
        raise NotImplementedError

    def _manipulator_agents(self) -> tuple[Any, ...]:
        return tuple(
            self.agent_for_role(binding.role)
            for binding in self.embodiment_spec.manipulators
        )

    def _hide_robots(self) -> None:
        for agent in self._manipulator_agents():
            agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))

    def _reset_robots(self, batch_size: int) -> None:
        base_poses = self._robot_base_poses_for_task()
        if len(self._manipulator_agents()) == 1:
            base_poses = [base_poses]
        if len(base_poses) != len(self._manipulator_agents()):
            raise RuntimeError(
                f"{type(self).__name__} expected {len(self._manipulator_agents())} "
                f"robot base poses, got {len(base_poses)}."
            )
        for agent, base_pose in zip(
            self._manipulator_agents(), base_poses, strict=True
        ):
            keyframe = agent.keyframes.get("rest")
            if keyframe is None or keyframe.qpos is None:
                raise RuntimeError(f"Agent {agent.uid!r} has no rest keyframe.")
            qpos = torch.as_tensor(
                keyframe.qpos, device=self.device, dtype=torch.float32
            ).unsqueeze(0).repeat(batch_size, 1)
            agent.reset(qpos)
            agent.robot.set_pose(base_pose)

    def _zero_actor_velocity(self, actor: Any, batch_size: int) -> None:
        zero = torch.zeros((batch_size, 3), device=self.device, dtype=torch.float32)
        actor.set_linear_velocity(zero)
        actor.set_angular_velocity(zero)

    def _site_pose(self, batch_size: int) -> Any:
        from mani_skill.utils.structs.pose import Pose

        return Pose.create_from_pq(
            p=self.interaction_site_local_point_tensor(
                (0.0, 0.0, 0.0), batch_size=batch_size
            ),
            q=self.interaction_site_quaternion_tensor(batch_size),
        )

    def _site_local_positions(self, entity: Any) -> Any:
        return (self._site_pose(self.num_envs).inv() * entity.pose).p

    @staticmethod
    def _entity_links(entity: Any) -> tuple[Any, ...]:
        links = getattr(entity, "links", None)
        return tuple(links) if links is not None else (entity,)

    def _max_touch_force(self, targets: Iterable[Any]) -> Any:
        """Return the maximum taxel-parent resultant force, in newtons."""

        target_links = tuple(
            link for target in targets for link in self._entity_links(target)
        )
        if not target_links:
            raise ValueError("At least one tactile target link is required.")
        target_groups = (("targets", target_links),)
        maximum = torch.zeros(self.num_envs, device=self.device)
        for agent in self._manipulator_agents():
            taxel_world_forces = agent.get_object_taxel_world_forces(
                target_groups
            )["targets"]
            # [B, N, 3] -> [B, P, 3] -> [B].
            parent_world_forces = agent.aggregate_taxel_world_forces_by_parent(
                taxel_world_forces
            )
            parent_force_norms = torch.linalg.norm(
                parent_world_forces,
                dim=-1,
            )
            maximum = torch.maximum(maximum, parent_force_norms.amax(dim=1))
        return maximum

    def _robot_joint_safety(self, max_speed_rad_s: float = 6.0) -> Any:
        unsafe = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self._manipulator_agents():
            qpos = agent.robot.get_qpos()
            qvel = agent.robot.get_qvel()
            limits = agent.robot.get_qlimits()
            lower, upper = limits[:, :, 0], limits[:, :, 1]
            finite = torch.isfinite(lower) & torch.isfinite(upper)
            outside = (((qpos < lower - 1e-3) | (qpos > upper + 1e-3)) & finite).any(
                dim=1
            )
            unsafe |= outside | (torch.abs(qvel).amax(dim=1) > max_speed_rad_s)
        return unsafe

    def _robot_joint_and_workspace_safety(
        self,
        *,
        workspace_local_min_xyz: tuple[float, float, float],
        workspace_local_max_xyz: tuple[float, float, float],
        max_speed_rad_s: float = 6.0,
    ) -> Any:
        """Check every manipulator's joints and task-frame TCP position."""

        unsafe = self._robot_joint_safety(max_speed_rad_s=max_speed_rad_s)
        site_inverse = self._site_pose(self.num_envs).inv()
        lower = torch.as_tensor(
            workspace_local_min_xyz, device=self.device, dtype=torch.float32
        )
        upper = torch.as_tensor(
            workspace_local_max_xyz, device=self.device, dtype=torch.float32
        )
        for agent in self._manipulator_agents():
            tcp_local_position = (site_inverse * agent.tcp.pose).p
            unsafe |= (
                (tcp_local_position < lower) | (tcp_local_position > upper)
            ).any(dim=1)
        return unsafe

    def _actor_static(
        self,
        actor: Any,
        linear_threshold: float = 0.04,
        angular_threshold: float = 0.6,
    ) -> Any:
        return (
            torch.linalg.norm(actor.linear_velocity, dim=1) <= linear_threshold
        ) & (torch.linalg.norm(actor.angular_velocity, dim=1) <= angular_threshold)
