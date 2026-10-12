from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np
import torch

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import AxisRandomizationSpec, TaskGeneralizationSpec
from dex_benchmark.sim.scenes.layouts import LayoutBuilder, LayoutRegistry, PoseSpec
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import sapien
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except ImportError:  # pragma: no cover
    sapien = Pose = None

    def register_env(*args: Any, **kwargs: Any) -> Any:
        return lambda cls: cls


B09_SETTINGS = TaskRegistry.register(TaskRegistry.load("b09_v1"))
B09_INTERACTION_SITE_ID = "workbench"
B09_LAYOUT_IDS = (LayoutId.STUDIO_L1, LayoutId.STUDIO_L2, LayoutId.STUDIO_L3)
B09_PLUG_NAME = "b09_two_pin_plug"
B09_SOCKET_NAME = "b09_mobile_socket"
B09_REMOVED_LAYOUT_ITEMS = frozenset(
    {
        "workbench_box",
        "clutter_large_marker",
        "clutter_adjustable_wrench",
        "clutter_power_drill",
        "clutter_phillips_screwdriver",
        "workbench_pliers",
        "glasses",
        "clutter_paper_placeholder",
        "stapler",
        "clutter_dice",
    }
)
B09_OBJECT_SCALE = 0.5625
B09_PLUG_BODY_HALF_SIZE_XYZ = (0.039375, 0.023625, 0.019125)
B09_PIN_X_M = 0.019125
B09_PIN_RADIUS_M = 0.0061875
# The rear 14 mm of each pin is embedded inside the square plug housing.
B09_PIN_LENGTH_M = 0.03375
B09_PIN_EMBEDDED_LENGTH_M = 0.007875
B09_INSERT_DEPTH_M = 0.01575
B09_ALIGNMENT_TOLERANCE_M = 0.012375
B09_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.16, 0.0, 0.55),
    quaternion_wxyz=(0.9305989566, 0.0, 0.3660406590, 0.0),
)


@dataclass(frozen=True)
class B09SuccessSpec:
    contact_force_n: float = 0.20
    alignment_tolerance_m: float = B09_ALIGNMENT_TOLERANCE_M
    insertion_depth_m: float = B09_INSERT_DEPTH_M
    maximum_speed_mps: float = 0.15


def plug_socket_alignment(
    pin_tips_in_socket: torch.Tensor,
    axis_cosine: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Check both physical pin tips against their socket channels."""

    if pin_tips_in_socket.ndim != 3 or pin_tips_in_socket.shape[1:] != (2, 3):
        raise ValueError("B09 requires both pin tips with shape [B, 2, 3].")
    tips = pin_tips_in_socket
    front_y = -0.010 * B09_OBJECT_SCALE
    back_y = 0.050 * B09_OBJECT_SCALE
    inner_x = 0.005 * B09_OBJECT_SCALE + B09_PIN_RADIUS_M
    outer_x = 0.078 * B09_OBJECT_SCALE - B09_PIN_RADIUS_M
    inner_z = 0.049 * B09_OBJECT_SCALE - B09_PIN_RADIUS_M

    channel_order = tips[:, :, 0].argsort(dim=1)
    channel_tips = torch.gather(
        tips, 1, channel_order[:, :, None].expand_as(tips)
    )
    channel_x = channel_tips[:, :, 0] * tips.new_tensor([-1.0, 1.0])
    lateral_ok = (
        (channel_x >= inner_x)
        & (channel_x <= outer_x)
        & (channel_tips[:, :, 2].abs() <= inner_z)
    ).all(dim=1)
    direction_ok = axis_cosine >= math.cos(math.radians(12.0))
    near_opening = (
        (tips[:, :, 1] >= front_y - 0.035)
        & (tips[:, :, 1] <= back_y)
    ).all(dim=1)
    aligned = lateral_ok & direction_ok & near_opening
    depth = tips[:, :, 1] - front_y
    entered = aligned & (depth >= 0.0).all(dim=1)
    inserted = aligned & (depth >= B09_INSERT_DEPTH_M).all(dim=1)
    expected_x = tips.new_tensor([-B09_PIN_X_M, B09_PIN_X_M])
    lateral_distance = torch.linalg.norm(
        torch.stack(
            (channel_tips[:, :, 0] - expected_x, channel_tips[:, :, 2]),
            dim=-1,
        ),
        dim=-1,
    ).amax(dim=1)
    return {
        "aligned": aligned,
        "entered": entered,
        "inserted": inserted,
        "lateral_distance": lateral_distance,
        "pin_depth_m": depth,
    }


def _camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B09_LAYOUT_IDS:
        site = LayoutRegistry.get(layout_id).interaction_site(B09_INTERACTION_SITE_ID)
        base_pose = site.bimanual_robot_base_center_pose or site.robot_base_center_pose
        world_pose = base_pose.transform_pose(B09_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE)
        quat = np.asarray(world_pose.quaternion_wxyz, dtype=np.float64)
        quat /= np.linalg.norm(quat)
        poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=world_pose.position_xyz,
            quaternion_world_wxyz=tuple(float(value) for value in quat),
        )
    return poses


@register_env(B09_SETTINGS.env_id, max_episode_steps=B09_SETTINGS.max_steps)
class B09Env(StudioTaskEnv):
    """Bimanual assembly of a large two-pin plug and a movable socket."""

    TASK_SETTINGS = B09_SETTINGS
    GENERALIZATION_SPEC = TaskGeneralizationSpec(
        spatial=AxisRandomizationSpec(parameters=(), requires_rebuild=False),
        physical=AxisRandomizationSpec(parameters=(), requires_rebuild=False),
    )
    SUCCESS_SPEC = B09SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B09_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _camera_poses()

    def human_viewer_camera_uid(self) -> str | None:
        """Use the fixed first-person rig instead of a wrist-mounted camera."""

        return None

    def _robot_base_poses_for_task(self) -> list[Any]:
        poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(poses, list) or len(poses) != 2:
            raise ValueError("B09 requires two workbench robot base poses.")
        return poses

    def _load_scene(self, options: dict[str, Any]) -> None:
        del options
        layout = replace(self.current_layout_spec, build_items=tuple(
            item for item in self.current_layout_spec.build_items
            if getattr(item, "name", None) not in B09_REMOVED_LAYOUT_ITEMS
        ))
        agents = self.agent
        try:
            self.agent = None
            self.layout_scene = LayoutBuilder(env=self, layout_spec=layout).build()
        finally:
            self.agent = agents
        self.plug = self._build_plug()
        self.socket = self._build_socket()
        self.layout_scene.actors[B09_PLUG_NAME] = self.plug
        self.layout_scene.actors[B09_SOCKET_NAME] = self.socket
        self._tactile_groups = (("plug", (self.plug,)), ("socket", (self.socket,)))
        if not self.gpu_sim_enabled:
            for agent in self._manipulator_agents():
                agent.prepare_cpu_object_taxel_query(self._tactile_groups)

    def _material(self) -> Any:
        return sapien.physx.PhysxMaterial(1.8, 1.5, 0.01)

    def _build_plug(self) -> Any:
        builder = self.scene.create_actor_builder()
        material = self._material()
        body_visual = sapien.render.RenderMaterial(
            base_color=(0.10, 0.19, 0.34, 1.0), roughness=0.38
        )
        pin_visual = sapien.render.RenderMaterial(
            base_color=(0.76, 0.62, 0.20, 1.0), metallic=0.75, roughness=0.26
        )
        body_half_size = B09_PLUG_BODY_HALF_SIZE_XYZ
        builder.add_box_collision(half_size=body_half_size, material=material, density=1100.0)
        builder.add_box_visual(half_size=body_half_size, material=body_visual)
        # SAPIEN cylinders are along local X; a +90 degree Z rotation puts
        # their length along the plug's forward local +Y direction.
        pin_rotation = sapien.Pose(q=(math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))
        for x_position in (-B09_PIN_X_M, B09_PIN_X_M):
            pin_center_y = (
                body_half_size[1]
                + B09_PIN_LENGTH_M / 2.0
                - B09_PIN_EMBEDDED_LENGTH_M
            )
            pose = sapien.Pose(p=(x_position, pin_center_y, 0.0)) * pin_rotation
            builder.add_cylinder_collision(pose=pose, radius=B09_PIN_RADIUS_M,
                                           half_length=B09_PIN_LENGTH_M / 2.0,
                                           material=material, density=1800.0)
            builder.add_cylinder_visual(pose=pose, radius=B09_PIN_RADIUS_M,
                                        half_length=B09_PIN_LENGTH_M / 2.0,
                                        material=pin_visual)
        builder.initial_pose = sapien.Pose(p=(0.0, 0.0, -20.0))
        return builder.build_dynamic(name=B09_PLUG_NAME)

    def _build_socket(self) -> Any:
        builder = self.scene.create_actor_builder()
        material = self._material()
        visual_material = sapien.render.RenderMaterial(
            base_color=(0.16, 0.16, 0.17, 1.0), roughness=0.48
        )
        # A wide, open-front box divided into two channels. The divider makes
        # the insertion geometry visibly and physically a two-pin socket.
        unscaled_parts = (((0.0, 0.060, 0.0), (0.090, 0.010, 0.060)),
                          ((0.0, 0.030, 0.055), (0.090, 0.040, 0.006)),
                          ((0.0, 0.030, -0.055), (0.090, 0.040, 0.006)),
                          ((0.084, 0.030, 0.0), (0.006, 0.040, 0.060)),
                          ((-0.084, 0.030, 0.0), (0.006, 0.040, 0.060)),
                          ((0.0, 0.030, 0.0), (0.005, 0.040, 0.060)))
        for unscaled_center, unscaled_half_size in unscaled_parts:
            center = tuple(value * B09_OBJECT_SCALE for value in unscaled_center)
            half_size = tuple(value * B09_OBJECT_SCALE for value in unscaled_half_size)
            pose = sapien.Pose(p=center)
            builder.add_box_collision(pose=pose, half_size=half_size, material=material, density=1200.0)
            builder.add_box_visual(pose=pose, half_size=half_size, material=visual_material)
        builder.initial_pose = sapien.Pose(p=(0.0, 0.0, -20.0))
        return builder.build_dynamic(name=B09_SOCKET_NAME)

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        self.layout_scene.initialize_episode(env_idx, self._batched_episode_rng, self.device)
        batch_size = len(env_idx)
        facing_quaternion = torch.tensor(
            (0.0, 0.0, 0.0, 1.0), device=self.device
        ).repeat(batch_size, 1)
        socket_pose = self._site_pose(batch_size) * Pose.create_from_pq(
            p=torch.tensor((0.03, -0.16, 0.055), device=self.device).repeat(batch_size, 1),
            q=facing_quaternion,
        )
        plug_pose = self._site_pose(batch_size) * Pose.create_from_pq(
            p=torch.tensor((0.03, 0.16, 0.055), device=self.device).repeat(batch_size, 1),
            q=facing_quaternion,
        )
        self.socket.set_pose(socket_pose)
        self.plug.set_pose(plug_pose)
        zero = torch.zeros((batch_size, 3), device=self.device)
        self.socket.set_linear_velocity(zero); self.socket.set_angular_velocity(zero)
        self.plug.set_linear_velocity(zero); self.plug.set_angular_velocity(zero)
        self._reset_robots(batch_size)
    def _held_by_hands(self, group_name: str) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for agent in self._manipulator_agents():
            forces = agent.get_object_taxel_world_forces(self._tactile_groups)[group_name]
            parent_forces = agent.aggregate_taxel_world_forces_by_parent(forces)
            links = torch.linalg.norm(parent_forces, dim=-1) >= self.SUCCESS_SPEC.contact_force_n
            values.append(links.sum(dim=1) >= 2)
        return torch.stack(values, dim=1)

    def _geometry(self) -> dict[str, torch.Tensor]:
        socket_rotation = self.socket.pose.to_transformation_matrix()[:, :3, :3]
        plug_rotation = self.plug.pose.to_transformation_matrix()[:, :3, :3]
        pin_end_y = (
            B09_PLUG_BODY_HALF_SIZE_XYZ[1]
            + B09_PIN_LENGTH_M
            - B09_PIN_EMBEDDED_LENGTH_M
        )
        tips_local = torch.tensor(
            [
                (-B09_PIN_X_M, pin_end_y, 0.0),
                (B09_PIN_X_M, pin_end_y, 0.0),
            ],
            device=self.device,
        )
        tips_world = self.plug.pose.p[:, None, :] + torch.einsum(
            "bij,nj->bni", plug_rotation, tips_local
        )
        local = torch.einsum(
            "bij,bnj->bni",
            socket_rotation.transpose(1, 2),
            tips_world - self.socket.pose.p[:, None, :],
        )
        axis_cosine = (
            plug_rotation[:, :, 1] * socket_rotation[:, :, 1]
        ).sum(dim=1)
        return plug_socket_alignment(local, axis_cosine)

    def evaluate(self) -> dict[str, Any]:
        plug_held = self._held_by_hands("plug")
        socket_held = self._held_by_hands("socket")
        cooperative = (plug_held[:, 0] & socket_held[:, 1]) | (plug_held[:, 1] & socket_held[:, 0])
        geometry = self._geometry()
        relative_linear_speed = torch.linalg.norm(
            self.plug.linear_velocity - self.socket.linear_velocity, dim=1
        )
        relative_angular_speed = torch.linalg.norm(
            self.plug.angular_velocity - self.socket.angular_velocity, dim=1
        )
        stable = (
            (relative_linear_speed <= self.SUCCESS_SPEC.maximum_speed_mps)
            & (relative_angular_speed <= 1.0)
        )
        success = self._latch_task_success(geometry["inserted"] & stable)
        return {
            "success": success,
            "plug_and_socket_held": cooperative,
            "plug_aligned_with_socket": geometry["aligned"],
            "both_pins_entered_socket": geometry["entered"],
            "plug_fully_inserted": success,
            "plug_lateral_error_m": geometry["lateral_distance"],
            "pin_depth_m": geometry["pin_depth_m"],
        }
