from __future__ import annotations

from dataclasses import replace
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
from dex_benchmark.sim.scenes.layouts import ArticulationBuildValues, ArticulatedAssetSpec, LayoutSpec
from dex_benchmark.sim.scenes.layouts.studio import (
    PARTNET_BBOX_CENTER_XYZ,
    PARTNET_BBOX_MIN_Z,
    STUDIO_MOUSE_MODEL_ID,
    STUDIO_PHYSICAL_MATERIAL,
)

from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    import sapien
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - optional simulator stack.
    sapien = None
    Pose = None

try:
    from mani_skill.utils.registration import register_env
except Exception:  # pragma: no cover

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


M07_SETTINGS = TaskRegistry.register(TaskRegistry.load("m07_v1"))
M07_WHEEL_JOINT = "joint_2"
M07_MOUSE_SCALE_FACTOR = 1.20
M07_TARGET_ROLL_RAD = 1.5
M07_SCREEN_REVEAL_ROLL_RAD = 0.03
M07_REMOVED = frozenset({"phone", "usb", "clutter_large_marker"})
# The screen mesh in PartNet model 9992 is authored in link_1 coordinates.
# LayoutBuilder applies the model scale (0.28), so these values are already
# converted to the scaled link frame used by the runtime overlay.
M07_SCREEN_LOCAL_CENTER_XYZ = (-0.1799707, 0.1161545, -0.0433380)
M07_SCREEN_HALF_SIZE_XYZ = (0.1670347, 0.0996583, 0.0020)
M07_SCREEN_LOCAL_QUATERNION_WXYZ = (0.988238, -0.154934, 0.0, 0.0)
M07_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("mouse_x_offset_m", -0.03, 0.03),
            GeneralizationParameterSpec("mouse_y_offset_m", -0.03, 0.03),
            GeneralizationParameterSpec(
                "mouse_yaw_offset_rad", -math.radians(12), math.radians(12)
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "mouse_mass_log_scale", math.log(0.8), math.log(1.2)
            ),
            GeneralizationParameterSpec("wheel_friction", 0.02, 0.12),
            GeneralizationParameterSpec("wheel_damping", 0.03, 0.16),
        ),
        requires_rebuild=True,
    ),
)


@register_env(M07_SETTINGS.env_id, max_episode_steps=M07_SETTINGS.max_steps)
class M07Env(StudioTaskEnv):
    """Hold a mouse on the desk and rotate its wheel."""

    TASK_SETTINGS = M07_SETTINGS
    GENERALIZATION_SPEC = M07_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "desk_top"
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.850088, -0.979861, 1.298413),
            quaternion_world_wxyz=(0.913726, -0.081551, 0.325791, 0.228722),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(1.844860, 1.337104, 1.398183),
            quaternion_world_wxyz=(0.406754, -0.367007, 0.182867, 0.816343),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.306532, 1.007102, 1.451206),
            quaternion_world_wxyz=(0.457658, -0.337073, 0.192871, 0.799832),
        ),
    }

    def _scaled_mouse_spec(self, item: ArticulatedAssetSpec) -> ArticulatedAssetSpec:
        scale = item.scale * M07_MOUSE_SCALE_FACTOR
        center = PARTNET_BBOX_CENTER_XYZ[STUDIO_MOUSE_MODEL_ID]
        # Preserve the world position of the bbox bottom-center as size changes.
        bottom_center = (center[0], center[1], PARTNET_BBOX_MIN_Z[STUDIO_MOUSE_MODEL_ID])
        root_offset = tuple((item.scale - scale) * value for value in bottom_center)
        return replace(
            item,
            scale=scale,
            pose=replace(item.pose, position_xyz=item.pose.transform_point(root_offset)),
        )

    def _task_layout_spec(self) -> LayoutSpec:
        layout = self.current_layout_spec
        # Robot faces local +X: back into the desk is +X, left is +Y.
        # [3, 3] @ [3] -> [3]: task-local translation into world coordinates.
        offset_world = layout.support_surfaces["desk.top"].pose.rotation_matrix @ (
            -0.08, 0.16, -layout.relative_placements["mouse"].initial_clearance_z
        )
        mouse = next(item for item in layout.build_items if item.name == "mouse")
        assert isinstance(mouse, ArticulatedAssetSpec)
        mouse = self._scaled_mouse_spec(mouse)
        mouse = replace(
            mouse,
            fix_root_link=True,
            # Enlarged wheel overlaps its housing; retain hand/object collisions
            # while preventing internal mouse contacts from locking the wheel.
            disable_self_collisions=True,
            asset_path=partnet_mobility_modified_model_urdf_path("101408"),
            pose=replace(
                mouse.pose,
                position_xyz=tuple(
                    float(mouse.pose.position_xyz[index] + offset_world[index])
                    for index in range(3)
                ),
            ),
        )
        laptop = next(item for item in layout.build_items if item.name == "laptop")
        assert isinstance(laptop, ArticulatedAssetSpec)
        # [3, 3] @ [3] -> [3]: desk-local left translation into world space.
        laptop_offset_world = layout.support_surfaces["desk.top"].pose.rotation_matrix @ (
            0.0, 0.08, 0.0
        )
        laptop = replace(
            laptop,
            pose=replace(
                laptop.pose,
                position_xyz=tuple(
                    float(laptop.pose.position_xyz[index] + laptop_offset_world[index])
                    for index in range(3)
                ),
            ),
        )
        return replace(
            layout,
            build_items=tuple(
                mouse
                if isinstance(item, ArticulatedAssetSpec) and item.name == "mouse"
                else laptop
                if isinstance(item, ArticulatedAssetSpec) and item.name == "laptop"
                else item
                for item in self.current_layout_spec.build_items
                if getattr(item, "name", None) not in M07_REMOVED
            ),
        )

    def _load_scene(self, options: dict[str, Any]) -> None:
        super()._load_scene(options)
        self._build_screen_overlay()

    def _setup_viewer(self) -> None:
        from mani_skill.utils import sapien_utils

        from dex_benchmark.sim.tasks.rotational_manipulation.m07_viewer import MouseAlignmentPreview

        super()._setup_viewer()
        viewer = self._viewer
        assert viewer is not None
        camera = self.scene.sub_scenes[0].add_camera(
            name="m07_mouse_alignment", width=480, height=360,
            fovy=math.radians(50.0), near=0.01, far=5.0,
        )
        # [B, 3] -> [3]: show the first environment, like the human viewer.
        target = self.wheel_link.pose.p[0].detach().cpu().numpy()
        # [3, 3] @ [3] -> [3]: left/up viewpoint in the desk frame.
        offset = self.current_layout_spec.support_surfaces["desk.top"].pose.rotation_matrix @ (
            -0.08, 0.22, 0.12
        )
        camera.set_pose(sapien_utils.look_at(eye=target + offset, target=target).sp)
        preview = MouseAlignmentPreview(camera)
        preview.init(viewer)
        viewer.plugins.append(preview)

    def _bind_task_entities(self) -> None:
        self.mouse = self.layout_scene.require_entity("mouse")
        self.laptop = self.layout_scene.require_entity("laptop")
        try:
            self.wheel_joint = self.mouse.active_joints_map[M07_WHEEL_JOINT]
            self.lid_joint = self.laptop.active_joints_map["joint_1"]
        except KeyError as exc:
            raise RuntimeError(
                "M07 requires mouse joint_2 and laptop joint_1 in the Studio layout."
            ) from exc
        self.wheel_link = self.wheel_joint.child_link
        if self.wheel_link is None:
            raise RuntimeError("Mouse wheel joint must expose a child link.")
        self._color_mouse_wheel()
        self.screen_link = self.lid_joint.child_link
        if self.screen_link is None:
            raise RuntimeError("Laptop joint_1 must expose the screen link.")

    def _color_mouse_wheel(self) -> None:
        if sapien is None:
            raise ImportError("SAPIEN is required to color the mouse wheel.")
        for shapes in self.wheel_link.render_shapes:
            for shape in shapes:
                assert isinstance(shape, sapien.render.RenderShapeTriangleMesh)
                for part in shape.parts:
                    part.material.set_base_color([0.12, 0.12, 0.12, 1.0])
                    part.material.set_base_color_texture(None)
                    part.material.roughness = 0.65
                    part.material.metallic = 0.0

    def _build_screen_overlay(self) -> None:
        if sapien is None:
            raise ImportError("SAPIEN is required to build the M07 screen overlay.")
        material = sapien.render.RenderMaterial(
            base_color=[0.0, 0.0, 0.0, 1.0],
            roughness=0.85,
        )
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=M07_SCREEN_HALF_SIZE_XYZ,
            material=material,
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -20.0])
        self.screen_overlay = builder.build_kinematic(name="m07_black_screen_overlay")

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            ("mouse_mass_log_scale", "wheel_friction", "wheel_damping"),
            cache_name="_m07_physical_values",
        )
        return None, {
            "mouse": ArticulationBuildValues(
                physical_materials=tuple(
                    scaled_physical_material(
                        STUDIO_PHYSICAL_MATERIAL,
                        friction_scale=float(v),
                        restitution=0.01,
                    )
                    for v in positive_scale(values["wheel_friction"])
                ),
                mass_scales=tuple(
                    float(v) for v in positive_scale(values["mouse_mass_log_scale"])
                ),
                joint_friction={
                    M07_WHEEL_JOINT: tuple(float(v) for v in values["wheel_friction"])
                },
                joint_damping={
                    M07_WHEEL_JOINT: tuple(float(v) for v in values["wheel_damping"])
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
                ("mouse_x_offset_m", "mouse_y_offset_m", "mouse_yaw_offset_rad"),
                cache_name="_m07_spatial_values",
            )
            indices = env_idx.detach().cpu().numpy()
            site_pose = self._site_pose(batch_size)
            local_pose = site_pose.inv() * self.mouse.pose[env_idx]
            position_offset = torch.zeros(
                (batch_size, 3), device=self.device, dtype=torch.float32
            )
            position_offset[:, 0] = torch.as_tensor(
                spatial_values["mouse_x_offset_m"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            position_offset[:, 1] = torch.as_tensor(
                spatial_values["mouse_y_offset_m"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            yaw_offset = torch.as_tensor(
                spatial_values["mouse_yaw_offset_rad"][indices],
                device=self.device,
                dtype=torch.float32,
            )
            yaw_quaternion = torch.stack(
                (
                    torch.cos(yaw_offset / 2.0),
                    torch.zeros_like(yaw_offset),
                    torch.zeros_like(yaw_offset),
                    torch.sin(yaw_offset / 2.0),
                ),
                dim=1,
            )
            randomized_local_pose = Pose.create_from_pq(
                p=local_pose.p + position_offset,
                q=local_pose.q,
            ) * Pose.create_from_pq(q=yaw_quaternion)
            self.mouse.set_pose(site_pose * randomized_local_pose)
        # Keep the hand clear while gravity seats the mouse on the desk.
        # settle() requires a full reset because it advances every sub-scene.
        if batch_size == self.num_envs:
            self._hide_robots()
            self.layout_scene.settle(env_idx=env_idx)
        qpos = self.mouse.get_qpos()
        if not hasattr(self, "_initial_wheel_qpos"):
            self._initial_wheel_qpos = qpos[:, self.wheel_joint.active_index].clone()
            self._initial_mouse_pose = self.mouse.pose.p.clone()
            self._mouse_ever_grasped = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._wheel_ever_contact = torch.zeros_like(self._mouse_ever_grasped)
            self._wheel_target_reached = torch.zeros_like(self._mouse_ever_grasped)
            self._screen_revealed = torch.zeros_like(self._mouse_ever_grasped)
            self._last_wheel_delta = torch.zeros_like(self._initial_wheel_qpos)
        else:
            self._initial_wheel_qpos[env_idx] = qpos[
                env_idx, self.wheel_joint.active_index
            ]
            self._initial_mouse_pose[env_idx] = self.mouse.pose.p[env_idx]
        self.mouse.set_qvel(torch.zeros_like(qpos))
        self._mouse_ever_grasped[env_idx] = False
        self._wheel_ever_contact[env_idx] = False
        self._wheel_target_reached[env_idx] = False
        self._screen_revealed[env_idx] = False
        self._last_wheel_delta[env_idx] = 0.0
        self._reset_robots(batch_size)
        self._update_screen_overlay()

    def evaluate(self) -> dict[str, Any]:
        signed_delta = self.mouse.get_qpos()[:, self.wheel_joint.active_index].reshape(
            -1
        ) - self._initial_wheel_qpos.reshape(-1)
        wheel_delta = torch.abs(signed_delta)
        reached = wheel_delta >= M07_TARGET_ROLL_RAD
        wheel_force = self._max_touch_force((self.wheel_link,))
        body_force = self._max_touch_force((self.mouse,))
        wheel_contact = wheel_force >= 0.15
        grasped = body_force >= 0.15
        self._mouse_ever_grasped |= grasped | (wheel_delta > 0.03)
        self._wheel_ever_contact |= wheel_contact | (wheel_delta > 0.03)
        self._wheel_target_reached |= (
            self._mouse_ever_grasped & self._wheel_ever_contact & reached
        )
        self._screen_revealed |= wheel_delta >= M07_SCREEN_REVEAL_ROLL_RAD
        self._last_wheel_delta = wheel_delta
        self._update_screen_overlay()
        mouse_stable = (
            torch.linalg.norm(self.mouse.root_linear_velocity, dim=1) <= 0.04
        ) & (torch.linalg.norm(self.mouse.root_angular_velocity, dim=1) <= 0.6)
        success = self._wheel_target_reached & mouse_stable
        return {
            "success": success,
            "mouse_grasped": self._mouse_ever_grasped,
            "wheel_contact": self._wheel_ever_contact,
            "screen_revealed": self._screen_revealed,
            "wheel_target_reached": success,
        }

    def _update_screen_overlay(self) -> None:
        if Pose is None:
            raise ImportError(
                "ManiSkill Pose is required to update the M07 screen overlay."
            )
        batch_size = self.num_envs
        local_center = torch.as_tensor(
            M07_SCREEN_LOCAL_CENTER_XYZ,
            device=self.device,
            dtype=torch.float32,
        ).expand(batch_size, -1)
        local_quaternion = torch.as_tensor(
            M07_SCREEN_LOCAL_QUATERNION_WXYZ,
            device=self.device,
            dtype=torch.float32,
        ).expand(batch_size, -1)
        overlay_pose = self.screen_link.pose * Pose.create_from_pq(
            p=local_center,
            q=local_quaternion,
        )
        overlay_position = overlay_pose.p.clone()
        overlay_position[self._screen_revealed, 2] -= 20.0
        self.screen_overlay.set_pose(
            Pose.create_from_pq(p=overlay_position, q=overlay_pose.q)
        )
