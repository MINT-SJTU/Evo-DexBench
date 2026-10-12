from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import math
from typing import Any

import numpy as np
import torch

from dex_benchmark.assets import partnet_mobility_modified_model_urdf_path
from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import AxisRandomizationSpec, TaskGeneralizationSpec
from dex_benchmark.sim.scenes.layouts import (
    ArticulatedAssetSpec,
    LayoutBuilder,
    LayoutRegistry,
    PoseSpec,
    StaticBoxSpec,
)
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


B10_SETTINGS = TaskRegistry.register(TaskRegistry.load("b10_v1"))
B10_INTERACTION_SITE_ID = "dining_table"
B10_LAYOUT_IDS = (LayoutId.KITCHEN_L1, LayoutId.KITCHEN_L2, LayoutId.KITCHEN_L3)
B10_SCISSORS_ASSET = partnet_mobility_modified_model_urdf_path("10449")
B10_SCISSORS_NAME = "b10_scissors"
B10_SHEATH_NAME = "b10_scissors_sheath"
B10_SCISSORS_STAND_NAME = "b10_scissors_grasp_stand"
B10_SCISSORS_SCALE = 0.24
# The current PartNet URDF loader only handles a scalar scale reliably for this
# asset. The larger uniform scale therefore supplies a thicker handle without
# passing an unsupported anisotropic scale through its mesh parser.
# The asset has a roughly 1.8 m original extent.  This produces a large,
# comfortable 30 cm tool. The 10449 collision geometry places the closed
# blade tip on root-local +Y. The view camera looks along site +X, so site +Y
# is the operator's screen-left.
B10_SCISSORS_LOCAL_XYZ = (0.10, 0.18, 0.085)
# The sheath's bottom collision wall reaches one half-height below its root.
# Keep it 11 mm above the tabletop at reset instead of interpenetrating it.
# Move the open sheath closer to the scissors tip while retaining clearance at
# reset, so the operator only needs a short insertion motion.
B10_SHEATH_LOCAL_XYZ = (0.10, -0.10, 0.040)
# Keep the platform directly under the scissors root/approximate centre of
# mass. Earlier blade-only support made the handle side torque the tool off.
B10_SCISSORS_STAND_LOCAL_XYZ = (0.10, 0.18, 0.030)
B10_SCISSORS_STAND_HALF_SIZE_XYZ = (0.110, 0.105, 0.030)
B10_SCISSORS_YAW_RAD = math.pi
B10_SHEATH_YAW_RAD = -math.pi / 2.0
B10_SCISSORS_QUATERNION_WXYZ = (
    math.cos(B10_SCISSORS_YAW_RAD / 2.0),
    0.0,
    0.0,
    math.sin(B10_SCISSORS_YAW_RAD / 2.0),
)
B10_SHEATH_QUATERNION_WXYZ = (
    math.cos(B10_SHEATH_YAW_RAD / 2.0),
    0.0,
    0.0,
    math.sin(B10_SHEATH_YAW_RAD / 2.0),
)
B10_SHEATH_HALF_LENGTH_M = 0.092
B10_SHEATH_HALF_WIDTH_M = 0.038
B10_SHEATH_HALF_HEIGHT_M = 0.029
B10_SHEATH_DENSITY_KG_M3 = 100.0
B10_SHEATH_TARGET_MASS_KG = 0.09
# The distal 80 mm is retained for blade-containment diagnostics only.
B10_PROTECTED_BLADE_LENGTH_M = 0.080
B10_SHEATH_WALL_M = 0.007
B10_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.16, 0.0, 0.55),
    quaternion_wxyz=(0.9305989566, 0.0, 0.3660406590, 0.0),
)


@dataclass(frozen=True)
class B10SuccessSpec:
    axis_tolerance_rad: float = math.radians(12.0)
    entrance_distance_m: float = 0.025
    geometry_tolerance_m: float = 0.002
    grasp_force_n: float = 0.20
    grasp_duration_s: float = 0.10
    stable_duration_s: float = 0.10
    maximum_relative_speed_mps: float = 0.08
    maximum_relative_angular_speed_rad_s: float = 1.0


def sheath_alignment(
    blade_points_in_sheath: torch.Tensor,
    blade_tips_in_sheath: torch.Tensor,
    blade_axis_in_sheath: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Require both real blade tips inside the channel, with correct direction."""

    spec = B10SuccessSpec()
    points, tips = blade_points_in_sheath, blade_tips_in_sheath
    if points.ndim != 3 or points.shape[-1] != 3 or tips.shape[1:] != (2, 3):
        raise ValueError("B10 expects blade points [B,N,3] and two tips [B,2,3].")
    lower = points.new_tensor([
        -B10_SHEATH_HALF_LENGTH_M,
        -B10_SHEATH_HALF_WIDTH_M + 2.0 * B10_SHEATH_WALL_M,
        -B10_SHEATH_HALF_HEIGHT_M + 2.0 * B10_SHEATH_WALL_M,
    ])
    upper = points.new_tensor([
        B10_SHEATH_HALF_LENGTH_M - 2.0 * B10_SHEATH_WALL_M,
        B10_SHEATH_HALF_WIDTH_M - 2.0 * B10_SHEATH_WALL_M,
        B10_SHEATH_HALF_HEIGHT_M,
    ])
    direction_ok = blade_axis_in_sheath[:, 0] >= math.cos(spec.axis_tolerance_rad)
    tip_cross_section_ok = (
        (tips[:, :, 1:] >= lower[1:] - spec.geometry_tolerance_m)
        & (tips[:, :, 1:] <= upper[1:] + spec.geometry_tolerance_m)
    ).all(dim=(1, 2))
    near_entrance = (
        (tips[:, :, 0] >= lower[0] - spec.entrance_distance_m)
        & (tips[:, :, 0] <= lower[0] + spec.entrance_distance_m)
    ).all(dim=1)
    protected_inside = (
        (points >= lower - spec.geometry_tolerance_m)
        & (points <= upper + spec.geometry_tolerance_m)
    ).all(dim=(1, 2))
    tips_inside = (
        (tips >= lower - spec.geometry_tolerance_m)
        & (tips <= upper + spec.geometry_tolerance_m)
    ).all(dim=(1, 2))
    inserted = direction_ok & tips_inside
    return {
        "aligned": direction_ok & tip_cross_section_ok & near_entrance,
        "inserted": inserted,
        "protected_blade_inside": protected_inside,
        "blade_tips_inside": tips_inside,
        "axis_cosine": blade_axis_in_sheath[:, 0],
        "tip_depth_m": tips[:, :, 0] - lower[0],
        "lateral_distance": torch.linalg.norm(tips[:, :, 1:], dim=-1).amax(dim=1),
    }


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4)
    if element is None:
        return result
    x, y, z = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    cx, cy, cz, sx, sy, sz = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    result[:3, :3] = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])
    result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ") * B10_SCISSORS_SCALE
    return result


@lru_cache(maxsize=1)
def _scissors_blade_calibration() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Calibrate both URDF blade meshes at the physically locked joint angle.

    Return diagnostic vertices, real tip and physical long axis in each
    blade link's scaled local frame.
    """

    import trimesh

    urdf = Path(B10_SCISSORS_ASSET)
    root = ET.parse(urdf).getroot()
    transforms = {"base": np.eye(4)}
    joints = list(root.findall("joint"))
    while joints:
        pending = []
        for joint in joints:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            if parent not in transforms:
                pending.append(joint)
                continue
            transform = _origin_matrix(joint.find("origin"))
            if joint.get("type") == "revolute":
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
                angle = float(joint.find("limit").get("lower"))
                ax, ay, az = axis
                cross = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]])
                rotation = np.eye(4)
                rotation[:3, :3] = np.eye(3) + math.sin(angle) * cross + (1 - math.cos(angle)) * (cross @ cross)
                transform = transform @ rotation
            transforms[child] = transforms[parent] @ transform
        if len(pending) == len(joints):
            raise ValueError("B10 scissors URDF has an unresolved joint tree.")
        joints = pending
    meshes = {}
    for link in root.findall("link"):
        for collision in link.findall("collision"):
            if collision.get("name") not in {"moving_blade", "support_blade"}:
                continue
            mesh_path = urdf.parent / collision.find("geometry/mesh").get("filename")
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            origin = _origin_matrix(collision.find("origin"))
            local = np.asarray(mesh.vertices) * B10_SCISSORS_SCALE
            local = local @ origin[:3, :3].T + origin[:3, 3]
            transform = transforms[link.get("name")]
            root_points = local @ transform[:3, :3].T + transform[:3, 3]
            meshes[link.get("name")] = (local, root_points, np.asarray(mesh.edges_unique), np.asarray(mesh.faces))
    if len(meshes) != 2:
        raise ValueError("B10 requires exactly two calibrated blade meshes.")
    cut_y = max(points[:, 1].max() for _, points, _, _ in meshes.values()) - B10_PROTECTED_BLADE_LENGTH_M
    result = {}
    for name, (local, root_points, edges, faces) in meshes.items():
        selected = local[root_points[:, 1] >= cut_y]
        a, b = edges[:, 0], edges[:, 1]
        crossing = (root_points[a, 1] - cut_y) * (root_points[b, 1] - cut_y) < 0
        a, b = a[crossing], b[crossing]
        fraction = (cut_y - root_points[a, 1]) / (root_points[b, 1] - root_points[a, 1])
        clipped = local[a] + fraction[:, None] * (local[b] - local[a])
        protected = np.concatenate((selected, clipped))
        tip = local[root_points[:, 1].argmax()]
        if not len(protected):
            raise ValueError(f"B10 blade {name!r} has no protected mesh segment.")
        # Surface-area moments avoid an axis bias from uneven OBJ tessellation.
        triangles = local[faces]  # [F,3,3]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2.0
        weights = areas / areas.sum()
        sums = triangles.sum(axis=1)
        mean = (weights[:, None] * sums / 3.0).sum(axis=0)
        second = (np.einsum("fki,fkj->fij", triangles, triangles) + np.einsum("fi,fj->fij", sums, sums)) / 12.0
        covariance = (weights[:, None, None] * second).sum(axis=0) - np.outer(mean, mean)
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        # Choose the heel-to-tip direction rather than the unsigned PCA axis.
        if (transforms[name][:3, :3] @ axis)[1] < 0.0:
            axis = -axis
        result[name] = (protected.astype(np.float32), tip.astype(np.float32), axis.astype(np.float32))
    return result


def scissors_blade_geometry() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return the diagnostic blade segment and real tip in each link frame."""
    return {name: (vertices, tip) for name, (vertices, tip, _) in _scissors_blade_calibration().items()}


def scissors_blade_axes() -> dict[str, np.ndarray]:
    """Return surface-calibrated, forward-facing blade axes in link frames."""
    return {name: axis for name, (_, _, axis) in _scissors_blade_calibration().items()}


def relative_tip_velocity(
    tips_world: torch.Tensor,
    blade_com_world: torch.Tensor,
    blade_linear_velocity: torch.Tensor,
    blade_angular_velocity: torch.Tensor,
    sheath_com_world: torch.Tensor,
    sheath_linear_velocity: torch.Tensor,
    sheath_angular_velocity: torch.Tensor,
) -> torch.Tensor:
    """Velocity at each tip relative to the moving sheath at that same point.

    Blade inputs are [B,2,3]; sheath inputs are [B,3]. All use world axes.
    SAPIEN rigid-body linear velocities are measured at the centre of mass.
    """
    blade_tip_velocity = blade_linear_velocity + torch.cross(
        blade_angular_velocity, tips_world - blade_com_world, dim=-1
    )
    # [B,3] -> [B,1,3], broadcast the reference field to both blade tips.
    sheath_tip_velocity = sheath_linear_velocity[:, None, :] + torch.cross(
        sheath_angular_velocity[:, None, :].expand_as(tips_world),
        tips_world - sheath_com_world[:, None, :], dim=-1,
    )
    return blade_tip_velocity - sheath_tip_velocity


def _camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    result: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B10_LAYOUT_IDS:
        site = LayoutRegistry.get(layout_id).interaction_site(B10_INTERACTION_SITE_ID)
        base = site.bimanual_robot_base_center_pose or site.robot_base_center_pose
        world = base.transform_pose(B10_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE)
        quaternion = np.asarray(world.quaternion_wxyz, dtype=np.float64)
        quaternion /= np.linalg.norm(quaternion)
        result[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=world.position_xyz,
            quaternion_world_wxyz=tuple(float(value) for value in quaternion),
        )
    return result


@register_env(B10_SETTINGS.env_id, max_episode_steps=B10_SETTINGS.max_steps)
class B10Env(StudioTaskEnv):
    """Close a large articulated scissors and fit its tip into a rigid sheath."""

    TASK_SETTINGS = B10_SETTINGS
    GENERALIZATION_SPEC = TaskGeneralizationSpec(
        spatial=AxisRandomizationSpec(parameters=(), requires_rebuild=False),
        physical=AxisRandomizationSpec(parameters=(), requires_rebuild=False),
    )
    SUCCESS_SPEC = B10SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = B10_INTERACTION_SITE_ID
    HUMAN_VIEWER_CAMERA_POSES = _camera_poses()

    def _robot_base_poses_for_task(self) -> list[Any]:
        poses = self.interaction_site_robot_base_sapien_poses()
        if not isinstance(poses, list) or len(poses) != 2:
            raise ValueError("B10 requires two dining-table robot base poses.")
        return poses

    def _load_scene(self, options: dict[str, Any]) -> None:
        """Build RoboCasa fixtures without exposing the dual-agent wrapper.

        RoboCasa's fixture builder accesses ``env.agent.robot`` directly.  The
        bimanual task instead owns a ``MultiAgent`` wrapper, so retain it only
        after the kitchen layout has been constructed.
        """

        del options
        agents = self.agent
        try:
            self.agent = None
            self.layout_scene = LayoutBuilder(
                env=self, layout_spec=self._task_layout_spec()
            ).build()
        finally:
            self.agent = agents
        self._bind_task_entities()

    def _task_layout_spec(self) -> Any:
        retained = tuple(
            item for item in self.current_layout_spec.build_items
            if getattr(item, "name", None) not in {"scissors", "bottle_yellow"}
        )
        scissors_pose = self.interaction_site_local_pose_to_world(PoseSpec(
            position_xyz=B10_SCISSORS_LOCAL_XYZ,
            quaternion_wxyz=B10_SCISSORS_QUATERNION_WXYZ,
        ))
        # A static placeholder gives the layout its visual/table registration;
        # the open-top collision sheath is built after the layout loads.
        sheath_pose = self.interaction_site_local_pose_to_world(PoseSpec(
            position_xyz=B10_SHEATH_LOCAL_XYZ,
            quaternion_wxyz=B10_SHEATH_QUATERNION_WXYZ,
        ))
        stand_pose = self.interaction_site_local_pose_to_world(PoseSpec(
            position_xyz=B10_SCISSORS_STAND_LOCAL_XYZ,
            quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        ))
        return replace(self.current_layout_spec, build_items=(*retained,
            ArticulatedAssetSpec(name=B10_SCISSORS_NAME, asset_path=B10_SCISSORS_ASSET,
                pose=scissors_pose, scale=B10_SCISSORS_SCALE, fix_root_link=False,
                initial_qpos=(0.0,), default_joint_friction=2.5),
            StaticBoxSpec(name="b10_sheath_visual_anchor", half_size_xyz=(0.001, 0.001, 0.001),
                pose=sheath_pose, color_rgba=(0.0, 0.0, 0.0, 0.0), body_type="static"),
            StaticBoxSpec(name=B10_SCISSORS_STAND_NAME,
                half_size_xyz=B10_SCISSORS_STAND_HALF_SIZE_XYZ, pose=stand_pose,
                color_rgba=(0.22, 0.34, 0.45, 1.0), body_type="static"),
        ))

    def _bind_task_entities(self) -> None:
        self.scissors = self.layout_scene.require_entity(B10_SCISSORS_NAME)
        if len(self.scissors.active_joints) != 1:
            raise RuntimeError("PartNet scissors 10449 must expose exactly one active joint.")
        self.opening_joint = self.scissors.active_joints[0]
        joint_limits = self.opening_joint.limits.reshape(self.num_envs, -1, 2)
        self._closed_joint_qpos = joint_limits[:, 0, 0].clone()
        locked_limits = torch.stack(
            (self._closed_joint_qpos, self._closed_joint_qpos), dim=1
        )
        # A zero-width limit is the physical lock. The drive target prevents
        # numerical drift when a gripper presses against a blade or handle.
        self.opening_joint.set_limits(locked_limits)
        self.opening_joint.set_drive_target(self._closed_joint_qpos)
        self.opening_joint.set_drive_properties(
            stiffness=1.0e4, damping=1.0e3, force_limit=1.0e4
        )
        self.moving_handle = self.opening_joint.child_link
        self.support_handle = next((link for link in self.scissors.links if link is not self.moving_handle), self.moving_handle)
        geometry = scissors_blade_geometry()
        self._blade_links = tuple(link for link in self.scissors.links if link.name in geometry)
        if len(self._blade_links) != 2:
            raise RuntimeError("B10 cannot find both physical blade links.")
        self._blade_points = tuple(
            torch.as_tensor(geometry[link.name][0], device=self.device)
            for link in self._blade_links
        )
        self._blade_tips = tuple(
            torch.as_tensor(geometry[link.name][1], device=self.device)
            for link in self._blade_links
        )
        axes = scissors_blade_axes()
        self._blade_axes = tuple(torch.as_tensor(axes[link.name], device=self.device) for link in self._blade_links)
        self.sheath = self._build_sheath()
        self._set_sheath_mass()
        self.layout_scene.actors[B10_SHEATH_NAME] = self.sheath
        # CPU taxel queries accept managed rigid bodies, not an Articulation
        # wrapper.  Register every scissors link as one logical target group.
        self._tactile_groups = (("scissors", tuple(self.scissors.links)),)
        if not self.gpu_sim_enabled:
            for agent in self._manipulator_agents():
                agent.prepare_cpu_object_taxel_query(self._tactile_groups)

    def _build_sheath(self) -> Any:
        """Build a high-friction, open-end plastic channel around a closed tip."""
        builder = self.scene.create_actor_builder()
        # Retain a grippy sheath while avoiding large impulse spikes when a
        # hand closes around the lighter, smaller scissors.
        material = sapien.physx.PhysxMaterial(0.85, 0.65, 0.01)
        visual_material = sapien.render.RenderMaterial(
            base_color=(0.88, 0.20, 0.12, 1.0), roughness=0.44
        )
        length = B10_SHEATH_HALF_LENGTH_M
        width = B10_SHEATH_HALF_WIDTH_M
        height = B10_SHEATH_HALF_HEIGHT_M
        wall = B10_SHEATH_WALL_M
        # local +X is the insertion direction; the open end is local -X.
        parts = (((0.0, 0.0, -height + wall), (length, width, wall)),
                 ((0.0, width - wall, 0.0), (length, wall, height)),
                 ((0.0, -width + wall, 0.0), (length, wall, height)),
                 ((length - wall, 0.0, 0.0), (wall, width, height)))
        for center, half_size in parts:
            pose = sapien.Pose(p=center)
            builder.add_box_collision(
                pose=pose,
                half_size=half_size,
                material=material,
                density=B10_SHEATH_DENSITY_KG_M3,
            )
            builder.add_box_visual(
                pose=pose, half_size=half_size, material=visual_material
            )
        builder.initial_pose = sapien.Pose(p=(0.0, 0.0, -20.0))
        return builder.build_dynamic(name=B10_SHEATH_NAME)

    def _set_sheath_mass(self) -> None:
        """Set each parallel-environment sheath body to the target mass."""

        raw_bodies = tuple(getattr(self.sheath, "_bodies", ()))
        if len(raw_bodies) != self.num_envs:
            raise RuntimeError(
                "B10 sheath mass calibration expected one rigid body per "
                f"environment, got {len(raw_bodies)} for {self.num_envs} environments."
            )
        for raw_body in raw_bodies:
            nominal_mass = float(raw_body.mass)
            nominal_inertia = np.asarray(raw_body.inertia, dtype=np.float64)
            if nominal_mass <= 0.0 or not np.isfinite(nominal_inertia).all():
                raise RuntimeError("B10 sheath has invalid nominal mass or inertia.")
            mass_scale = B10_SHEATH_TARGET_MASS_KG / nominal_mass
            raw_body.mass = B10_SHEATH_TARGET_MASS_KG
            # [3] * scalar -> [3]: keep inertia consistent with the new mass.
            raw_body.inertia = nominal_inertia * mass_scale

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        self.layout_scene.initialize_episode(env_idx, self._batched_episode_rng, self.device)
        batch_size = len(env_idx)
        sheath_pose = self._site_pose(batch_size) * Pose.create_from_pq(
            p=torch.tensor(B10_SHEATH_LOCAL_XYZ, device=self.device).repeat(batch_size, 1),
            q=torch.tensor(B10_SHEATH_QUATERNION_WXYZ, device=self.device).repeat(batch_size, 1),
        )
        self.sheath.set_pose(sheath_pose)
        zero = torch.zeros((batch_size, 3), device=self.device)
        self.sheath.set_linear_velocity(zero); self.sheath.set_angular_velocity(zero)
        qpos = self.scissors.get_qpos()
        # The articulated asset is physically locked closed in
        # _bind_task_entities; restore that angle on every partial reset.
        qpos[env_idx, self.opening_joint.active_index] = self._closed_joint_qpos[env_idx]
        self.scissors.set_qpos(qpos)
        self.scissors.set_qvel(torch.zeros_like(qpos))
        self.opening_joint.set_drive_target(self._closed_joint_qpos)
        self._reset_robots(batch_size)
        if not hasattr(self, "_grasp_steps"):
            self._grasp_steps = torch.zeros((self.num_envs, 2), dtype=torch.int64, device=self.device)
            self._scissors_ever_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._stable_sheath_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._grasp_steps[env_idx] = 0
        self._scissors_ever_grasped[env_idx] = False
        self._stable_sheath_steps[env_idx] = 0

    def _tip_geometry(self) -> dict[str, torch.Tensor]:
        sheath_rotation = self.sheath.pose.to_transformation_matrix()[:, :3, :3]
        inverse = sheath_rotation.transpose(1, 2)
        protected, tips, axes = [], [], []
        for link, vertices, tip, local_axis in zip(self._blade_links, self._blade_points, self._blade_tips, self._blade_axes, strict=True):
            rotation = link.pose.to_transformation_matrix()[:, :3, :3]
            world = link.pose.p[:, None, :] + torch.einsum("bij,nj->bni", rotation, vertices)
            protected.append(torch.einsum("bij,bnj->bni", inverse, world - self.sheath.pose.p[:, None, :]))
            tip_world = link.pose.p + torch.einsum("bij,j->bi", rotation, tip)
            tips.append(torch.einsum("bij,bj->bi", inverse, tip_world - self.sheath.pose.p))
            world_axis = torch.einsum("bij,j->bi", rotation, local_axis)
            axes.append(torch.einsum("bij,bj->bi", inverse, world_axis))
        # Use the closed scissors' physical centreline, not its root +Y axis.
        axis = torch.stack(axes, dim=1).mean(dim=1)
        axis = torch.nn.functional.normalize(axis, dim=-1)
        return sheath_alignment(torch.cat(protected, dim=1), torch.stack(tips, dim=1), axis)

    def _relative_tip_motion(self) -> tuple[torch.Tensor, torch.Tensor]:
        tips, centers, linear, angular = [], [], [], []
        for link, tip in zip(self._blade_links, self._blade_tips, strict=True):
            rotation = link.pose.to_transformation_matrix()[:, :3, :3]
            tips.append(link.pose.p + torch.einsum("bij,j->bi", rotation, tip))
            centers.append((link.pose * link.cmass_local_pose).p)
            linear.append(link.linear_velocity)
            angular.append(link.angular_velocity)
        angular_world = torch.stack(angular, dim=1)
        velocity = relative_tip_velocity(
            torch.stack(tips, dim=1), torch.stack(centers, dim=1),
            torch.stack(linear, dim=1), angular_world,
            (self.sheath.pose * self.sheath.cmass_local_pose).p,
            self.sheath.linear_velocity, self.sheath.angular_velocity,
        )
        # [B,2,3] -> [B]: both blade tips must satisfy the speed limit.
        speed = torch.linalg.norm(velocity, dim=-1).amax(dim=1)
        angular_speed = torch.linalg.norm(angular_world - self.sheath.angular_velocity[:, None, :], dim=-1).amax(dim=1)
        return speed, angular_speed

    def evaluate(self) -> dict[str, Any]:
        geometry = self._tip_geometry()
        contact = []
        for agent in self._manipulator_agents():
            forces = agent.get_object_taxel_world_forces(self._tactile_groups)["scissors"]
            parents = agent.aggregate_taxel_world_forces_by_parent(forces)
            contact.append((torch.linalg.norm(parents, dim=-1) >= self.SUCCESS_SPEC.grasp_force_n).sum(dim=1) >= 2)
        held = torch.stack(contact, dim=1)
        self._grasp_steps = torch.where(held, self._grasp_steps + 1, torch.zeros_like(self._grasp_steps))
        duration = max(1, math.ceil(self.SUCCESS_SPEC.grasp_duration_s * self._control_freq))
        self._scissors_ever_grasped |= (self._grasp_steps >= duration).any(dim=1)
        relative_speed, relative_angular_speed = self._relative_tip_motion()
        stable = (
            geometry["inserted"]
            & (relative_speed <= self.SUCCESS_SPEC.maximum_relative_speed_mps)
            & (relative_angular_speed <= self.SUCCESS_SPEC.maximum_relative_angular_speed_rad_s)
        )
        self._stable_sheath_steps = torch.where(stable, self._stable_sheath_steps + 1, torch.zeros_like(self._stable_sheath_steps))
        success = self._latch_task_success(
            self._stable_sheath_steps >= max(1, math.ceil(self.SUCCESS_SPEC.stable_duration_s * self._control_freq))
        )
        return {
            "success": success,
            "scissors_grasped": self._scissors_ever_grasped.clone(),
            "scissors_tip_aligned_with_sheath": geometry["aligned"],
            "scissors_fully_sheathed": success,
            "protected_blade_inside": geometry["protected_blade_inside"],
            "blade_tips_inside": geometry["blade_tips_inside"],
            "blade_axis_cosine": geometry["axis_cosine"],
            "relative_tip_speed_mps": relative_speed,
            "relative_angular_speed_rad_s": relative_angular_speed,
            "tip_depth_m": geometry["tip_depth_m"],
            "tip_lateral_error_m": geometry["lateral_distance"],
        }
