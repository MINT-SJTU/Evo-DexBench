from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.scenes.layouts import (
    ArticulatedAssetSpec,
    ArticulationBuildValues,
    JointPositionLatchSpec,
    LayoutRegistry,
    LayoutSpec,
    PoseSpec,
)
from dex_benchmark.sim.scenes.layouts.studio import (
    PARTNET_BBOX_CENTER_XYZ,
    PARTNET_BBOX_EXTENT_XYZ,
    STUDIO_GLASSES_MODEL_ID,
    STUDIO_PHYSICAL_MATERIAL,
)
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover
    torch = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls):
            return cls

        return decorator


B15_SETTINGS = TaskRegistry.register(TaskRegistry.load("b15_v1"))
B15_TEMPLE_JOINT_NAMES = ("joint_0", "joint_1")
B15_ROBOT_BASE_LOCAL_MIN_XYZ = (-0.71, -0.35, -0.80)
B15_ROBOT_BASE_LOCAL_MAX_XYZ = (-0.44, 0.35, 0.20)
B15_TCP_WORKSPACE_LOCAL_MIN_XYZ = (-1.00, -1.00, -0.40)
B15_TCP_WORKSPACE_LOCAL_MAX_XYZ = (1.00, 1.00, 1.20)
B15_TEMPLE_JOINT_FRICTION = 0.16
B15_GLASSES_SCALE: float = 0.15
B15_GLASSES_SITE_LOCAL_XY: tuple[float, float] = (0.0, 0.0)
B15_TEMPLE_JOINT_LATCH = JointPositionLatchSpec(
    stiffness=2.0,
    damping=0.12,
    force_limit=0.10,
)

# A head-height viewpoint on the bimanual centerline. The camera faces straight
# ahead and slightly downward, with a close view of the teleoperation workspace.
B15_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE = PoseSpec(
    position_xyz=(0.16, 0.0, 0.52),
    quaternion_wxyz=(
        0.9527610540,
        0.0,
        0.3037209511,
        0.0,
    ),
)
B15_LAYOUT_IDS = (
    LayoutId.STUDIO_L1,
    LayoutId.STUDIO_L2,
    LayoutId.STUDIO_L3,
)


def _b15_human_viewer_camera_poses() -> dict[str, HumanViewerCameraPoseSpec]:
    camera_poses: dict[str, HumanViewerCameraPoseSpec] = {}
    for layout_id in B15_LAYOUT_IDS:
        interaction_site = LayoutRegistry.get(layout_id).interaction_site(
            "workbench"
        )
        robot_base_center_pose = (
            interaction_site.bimanual_robot_base_center_pose
            or interaction_site.robot_base_center_pose
        )
        camera_world_pose = robot_base_center_pose.transform_pose(
            B15_HUMAN_VIEWER_CAMERA_BASE_LOCAL_POSE
        )
        camera_poses[layout_id.value] = HumanViewerCameraPoseSpec(
            position_world_xyz=camera_world_pose.position_xyz,
            quaternion_world_wxyz=camera_world_pose.quaternion_wxyz,
        )
    return camera_poses


B15_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("glasses_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("glasses_y_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("glasses_yaw_offset_rad", -math.radians(12.0), math.radians(12.0)),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("glasses_mass_log_scale", math.log(0.8), math.log(1.2)),
            GeneralizationParameterSpec("glasses_friction_log_scale", math.log(0.8), math.log(1.25)),
            GeneralizationParameterSpec("temple_friction", 0.03, 0.16),
            GeneralizationParameterSpec("temple_damping", 0.04, 0.18),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class B15SuccessSpec:
    # Each tuple is ordered as (joint_0/right temple, joint_1/left temple).
    # The two valid folding orders use equal normalized travel requirements
    # after applying qpos_tolerance_rad: the fully folded temple reaches
    # 89.4019% of its URDF range and the collision-limited second temple
    # reaches 45.1212%. This keeps left-first and right-first completion
    # equally strict despite their different joint limits.
    right_first_target_qpos_rad: tuple[float, float] = (
        0.96,
        0.60,
    )
    left_first_target_qpos_rad: tuple[float, float] = (
        0.5092783505154639,
        1.1397530864197531,
    )
    qpos_tolerance_rad: float = 0.05
    unfolded_hysteresis_rad: float = 0.08
    frame_stability_radius_m: float = 0.055
    contact_force_n: float = 0.4

    @property
    def completion_target_qpos_rad(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        return (
            self.right_first_target_qpos_rad,
            self.left_first_target_qpos_rad,
        )

    @property
    def per_temple_folded_qpos_rad(self) -> tuple[float, float]:
        return (
            min(
                self.right_first_target_qpos_rad[0],
                self.left_first_target_qpos_rad[0],
            ),
            min(
                self.right_first_target_qpos_rad[1],
                self.left_first_target_qpos_rad[1],
            ),
        )


@register_env(B15_SETTINGS.env_id, max_episode_steps=B15_SETTINGS.max_steps)
class B15Env(StudioTaskEnv):
    """Hold the eyeglass frame and fold both articulated temples inward."""

    TASK_SETTINGS = B15_SETTINGS
    GENERALIZATION_SPEC = B15_GENERALIZATION_SPEC
    SUCCESS_SPEC = B15SuccessSpec()
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "workbench"
    HUMAN_VIEWER_CAMERA_POSES = _b15_human_viewer_camera_poses()
    ROBOT_BASE_LOCAL_MIN_XYZ = B15_ROBOT_BASE_LOCAL_MIN_XYZ
    ROBOT_BASE_LOCAL_MAX_XYZ = B15_ROBOT_BASE_LOCAL_MAX_XYZ

    def human_viewer_camera_uid(self) -> str | None:
        """Keep B15's Human Viewer on its first-person free camera."""

        return None

    def _task_layout_spec(self) -> LayoutSpec:
        glasses_placement = self.current_layout_spec.relative_placements[
            "glasses"
        ]
        support_surface = self.current_layout_spec.support_surfaces[
            glasses_placement.surface_id
        ]
        interaction_site_pose = self.interaction_site_frame_pose()
        if not (
            np.allclose(
                support_surface.pose.position_xyz,
                interaction_site_pose.position_xyz,
                atol=1e-8,
            )
            and np.allclose(
                support_surface.pose.rotation_matrix,
                interaction_site_pose.rotation_matrix,
                atol=1e-8,
            )
        ):
            raise RuntimeError(
                "B15 glasses support surface must coincide with the workbench "
                "interaction-site frame."
            )

        glasses_local_orientation = PoseSpec(
            position_xyz=(0.0, 0.0, 0.0),
            quaternion_wxyz=glasses_placement.bbox_center_pose.quaternion_wxyz,
        )
        glasses_bbox_half_extent = (
            0.5
            * B15_GLASSES_SCALE
            * np.asarray(
                PARTNET_BBOX_EXTENT_XYZ[STUDIO_GLASSES_MODEL_ID],
                dtype=np.float64,
            )
        )
        # [3] @ [3] -> scalar: project the enlarged glasses OBB onto site Z.
        glasses_bbox_center_height = float(
            np.abs(glasses_local_orientation.rotation_matrix[2])
            @ glasses_bbox_half_extent
        )
        glasses_bbox_center_pose = PoseSpec(
            position_xyz=(
                B15_GLASSES_SITE_LOCAL_XY[0],
                B15_GLASSES_SITE_LOCAL_XY[1],
                glasses_bbox_center_height
                + glasses_placement.initial_clearance_z,
            ),
            quaternion_wxyz=(
                glasses_placement.bbox_center_pose.quaternion_wxyz
            ),
        )
        glasses_world_bbox_center_pose = support_surface.pose.transform_pose(
            glasses_bbox_center_pose
        )
        # [3, 3] @ [3] -> [3]: bbox-center offset in the world frame.
        glasses_bbox_center_offset_world = (
            glasses_world_bbox_center_pose.rotation_matrix
            @ (
                B15_GLASSES_SCALE
                * np.asarray(
                    PARTNET_BBOX_CENTER_XYZ[STUDIO_GLASSES_MODEL_ID],
                    dtype=np.float64,
                )
            )
        )
        glasses_root_position = (
            np.asarray(
                glasses_world_bbox_center_pose.position_xyz,
                dtype=np.float64,
            )
            - glasses_bbox_center_offset_world
        )
        glasses_root_pose = PoseSpec(
            position_xyz=(
                float(glasses_root_position[0]),
                float(glasses_root_position[1]),
                float(glasses_root_position[2]),
            ),
            quaternion_wxyz=glasses_world_bbox_center_pose.quaternion_wxyz,
        )

        build_items = tuple(
            replace(
                item,
                pose=glasses_root_pose,
                scale=B15_GLASSES_SCALE,
                joint_friction={
                    **item.joint_friction,
                    **{
                        name: B15_TEMPLE_JOINT_FRICTION
                        for name in B15_TEMPLE_JOINT_NAMES
                    },
                },
                joint_position_latches={
                    **item.joint_position_latches,
                    **{
                        name: B15_TEMPLE_JOINT_LATCH
                        for name in B15_TEMPLE_JOINT_NAMES
                    },
                },
            )
            if isinstance(item, ArticulatedAssetSpec) and item.name == "glasses"
            else item
            for item in self.current_layout_spec.build_items
        )
        relative_placements = dict(
            self.current_layout_spec.relative_placements
        )
        relative_placements["glasses"] = replace(
            glasses_placement,
            bbox_center_pose=glasses_bbox_center_pose,
        )
        return replace(
            self.current_layout_spec,
            build_items=build_items,
            relative_placements=relative_placements,
        )

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            (
                "glasses_mass_log_scale",
                "glasses_friction_log_scale",
                "temple_friction",
                "temple_damping",
            ),
            cache_name="_b15_physical_values",
        )
        return None, {
            "glasses": ArticulationBuildValues(
                physical_materials=tuple(
                    scaled_physical_material(
                        STUDIO_PHYSICAL_MATERIAL,
                        friction_scale=float(scale),
                        restitution=0.01,
                    )
                    for scale in positive_scale(
                        values["glasses_friction_log_scale"]
                    )
                ),
                mass_scales=tuple(
                    float(scale)
                    for scale in positive_scale(values["glasses_mass_log_scale"])
                ),
                joint_friction={name: tuple(float(v) for v in values["temple_friction"]) for name in B15_TEMPLE_JOINT_NAMES},
                joint_damping={name: tuple(float(v) for v in values["temple_damping"]) for name in B15_TEMPLE_JOINT_NAMES},
            )
        }

    def _bind_task_entities(self) -> None:
        self.glasses = self.layout_scene.require_entity("glasses")
        try:
            self.temple_joints = tuple(self.glasses.active_joints_map[name] for name in B15_TEMPLE_JOINT_NAMES)
        except KeyError as exc:
            raise RuntimeError("Glasses asset 101860 must expose joint_0 and joint_1.") from exc
        self.temple_links = tuple(joint.child_link for joint in self.temple_joints)
        if any(link is None for link in self.temple_links):
            raise RuntimeError("Each glasses temple joint must have a child link.")
        moving_links = set(self.temple_links)
        frame_links = tuple(link for link in self.glasses.links if link not in moving_links)
        if not frame_links:
            raise RuntimeError("Glasses articulation has no stationary frame link.")
        self.frame_links = frame_links

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        with torch.device(self.device):
            batch_size = len(env_idx)
            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            qpos = self.glasses.get_qpos()
            for joint in self.temple_joints:
                if joint.active_index is None:
                    raise RuntimeError(f"Temple joint {joint.name!r} has no active index.")
                limits = joint.limits.reshape(self.num_envs, -1, 2)
                qpos[env_idx, joint.active_index] = limits[env_idx, 0, 0]
            self.glasses.set_qpos(qpos)
            self.glasses.set_qvel(torch.zeros_like(qpos))
            if self.generalization_axis is GeneralizationAxis.SPATIAL:
                values = self._generalization_values(
                    ("glasses_x_offset_m", "glasses_y_offset_m", "glasses_yaw_offset_rad"),
                    cache_name="_b15_spatial_values",
                )
                indices = env_idx.detach().cpu().numpy()
                local_pose = self._site_pose(batch_size).inv() * self.glasses.pose[env_idx]
                offset = torch.zeros((batch_size, 3), device=self.device)
                offset[:, 0] = torch.as_tensor(values["glasses_x_offset_m"][indices], device=self.device)
                offset[:, 1] = torch.as_tensor(values["glasses_y_offset_m"][indices], device=self.device)
                yaw = torch.as_tensor(values["glasses_yaw_offset_rad"][indices], device=self.device)
                yaw_q = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=1)
                self.glasses.set_pose(self._site_pose(batch_size) * Pose.create_from_pq(p=local_pose.p + offset, q=local_pose.q) * Pose.create_from_pq(q=yaw_q))
            self._hide_robots()
            self.layout_scene.settle(env_idx=env_idx)
            self._reset_robots(batch_size)
            self._glasses_initial_position = self.glasses.pose.p.clone()
            self._temple_ever_folded = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
            self._frame_was_stabilized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._glasses_ever_touched = torch.zeros_like(
                self._frame_was_stabilized
            )

    def _folded_fractions(self) -> Any:
        fractions = []
        for joint in self.temple_joints:
            qpos = joint.qpos.reshape(self.num_envs, -1)[:, 0]
            limits = joint.limits.reshape(self.num_envs, -1, 2)[:, 0]
            fractions.append(((qpos - limits[:, 0]) / (limits[:, 1] - limits[:, 0]).clamp_min(1e-6)).clamp(0.0, 1.0))
        return torch.stack(fractions, dim=1)

    def _temple_qpos(self) -> Any:
        # Two scalar joint positions -> [B, 2], ordered as joint_0 then joint_1.
        return torch.stack(
            [
                joint.qpos.reshape(self.num_envs, -1)[:, 0]
                for joint in self.temple_joints
            ],
            dim=1,
        )

    def evaluate(self) -> dict[str, Any]:
        temple_qpos = self._temple_qpos()
        per_temple_thresholds = torch.as_tensor(
            self.SUCCESS_SPEC.per_temple_folded_qpos_rad,
            device=self.device,
            dtype=torch.float32,
        ) - self.SUCCESS_SPEC.qpos_tolerance_rad
        # [B, 2] >= [2] -> [B, 2]: each temple reached at least the smaller
        # collision-limited angle required by either valid folding order.
        folded = temple_qpos >= per_temple_thresholds
        completion_thresholds = torch.as_tensor(
            self.SUCCESS_SPEC.completion_target_qpos_rad,
            device=self.device,
            dtype=torch.float32,
        ) - self.SUCCESS_SPEC.qpos_tolerance_rad
        # [B, 1, 2] >= [1, 2, 2] -> [B, 2, 2] -> [B, 2]: match the final
        # joint-angle pair for either right-first or left-first folding.
        completion_by_order = (
            temple_qpos[:, None, :] >= completion_thresholds[None, :, :]
        ).all(dim=2)
        both_folded = completion_by_order.any(dim=1)
        frame_force_n = self._max_touch_force(self.frame_links)
        temple_force_n = self._max_touch_force(self.temple_links)
        frame_contact = frame_force_n >= self.SUCCESS_SPEC.contact_force_n
        temple_contact = temple_force_n >= self.SUCCESS_SPEC.contact_force_n
        frame_displacement_m = torch.linalg.norm(
            self.glasses.pose.p - self._glasses_initial_position, dim=1
        )
        frame_pose_stable = frame_displacement_m <= self.SUCCESS_SPEC.frame_stability_radius_m
        # Folding order is intentionally unconstrained. Task-object taxel
        # attribution can be zero even during real Quest manipulation, so
        # contact and frame-return history remain telemetry only and do not
        # block the articulated completion condition.
        self._glasses_ever_touched |= frame_contact | temple_contact
        self._frame_was_stabilized |= frame_contact & frame_pose_stable
        self._temple_ever_folded |= folded
        one_folded = self._temple_ever_folded.any(dim=1)
        success = both_folded
        return {
            "success": success,
            "frame_stabilized": self._frame_was_stabilized,
            "one_temple_folded": one_folded,
            "left_temple_folded": self._temple_ever_folded[:, 1],
            "right_temple_folded": self._temple_ever_folded[:, 0],
            "both_temples_folded": both_folded,
            "folded_returned_and_stable": success,
        }

    def _check_generalization_validity(self, axis: GeneralizationAxis | None) -> Any:
        if axis in (GeneralizationAxis.SPATIAL, GeneralizationAxis.PHYSICAL):
            return torch.isfinite(self.glasses.pose.raw_pose).all(dim=1) & (self._folded_fractions() <= 0.05).all(dim=1)
        return super()._check_generalization_validity(axis)
