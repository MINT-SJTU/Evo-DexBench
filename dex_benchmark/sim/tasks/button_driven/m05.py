from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

import torch
from mani_skill.utils.structs.pose import Pose
import sapien

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
    positive_scale,
    scaled_physical_material,
)
from dex_benchmark.sim.scenes.layouts import ArticulationBuildValues, JointPositionLatchSpec
from dex_benchmark.sim.scenes.layouts.studio import STUDIO_PHYSICAL_MATERIAL
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.sim.tasks.registry import TaskRegistry
from dex_benchmark.sim.tasks.studio_task_common import StudioTaskEnv

try:
    from mani_skill.utils.registration import register_env
except Exception:  # pragma: no cover

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls: type[Any]) -> type[Any]:
            return cls

        return decorator


M05_SETTINGS = TaskRegistry.register(TaskRegistry.load("m05_v1"))
M05_BUTTON_JOINT = "joint_0"
M05_BUTTON_THRESHOLD_M = 0.003
M05_BUTTON_COLLISION_REST_OFFSET_M = 0.001
# Centre of link_0's convex button mesh in the authored 3392 model, transformed
# by STUDIO_DISPLAY_MODEL_SCALE=0.35.  The mesh is offset by almost 30 cm from
# the link origin, so the link origin itself is not a meaningful proximity
# target.
M05_BUTTON_INTERACTION_LOCAL_XYZ = (
    0.262147375,
    -0.142756250,
    0.009677675,
)
# Fingertip-link origins sit slightly behind the physical Shadow Hand pads.
# Five centimetres captures an intentional approach to the small power button
# without crediting a hand that is merely somewhere in front of the display.
M05_BUTTON_PROXIMITY_THRESHOLD_M = 0.05
# M05-specific display setup.  joint_2 is the vertical slide on the stand;
# 0.32 m keeps the screen above the laptop while remaining inside its
# [-0.04, 0.568] m joint limits.
M05_DISPLAY_HEIGHT_JOINT_QPOS = 0.32
M05_DISPLAY_HEIGHT_LATCH = JointPositionLatchSpec(
    stiffness=5000.0,
    damping=100.0,
    force_limit=1000.0,
)
M05_REMOVED = frozenset({"phone", "usb", "clutter_large_marker", "mouse"})
M05_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec("display_x_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec("display_y_offset_m", -0.025, 0.025),
            GeneralizationParameterSpec(
                "display_yaw_offset_rad", -math.radians(8), math.radians(8)
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "display_mass_log_scale", math.log(0.85), math.log(1.15)
            ),
            GeneralizationParameterSpec("display_friction", 0.43, 0.56),
            GeneralizationParameterSpec("button_friction", 0.03, 0.16),
            GeneralizationParameterSpec("button_damping", 0.04, 0.18),
        ),
        requires_rebuild=True,
    ),
)


@dataclass(frozen=True)
class M05SuccessSpec:
    contact_force_n: float = 0.15
    trigger_fraction: float = 0.70
    stable_steps: int = 3


M05_SUCCESS_SPEC = M05SuccessSpec()


@register_env(M05_SETTINGS.env_id, max_episode_steps=M05_SETTINGS.max_steps)
class M05Env(StudioTaskEnv):
    """Press the display's articulated power button."""

    TASK_SETTINGS = M05_SETTINGS
    GENERALIZATION_SPEC = M05_GENERALIZATION_SPEC
    SUCCESS_SPEC = M05_SUCCESS_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "desk_top"
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.STUDIO_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.154275, -0.841772, 1.381571),
            quaternion_world_wxyz=(0.915644, -0.079333, 0.224073, 0.324183),
        ),
        LayoutId.STUDIO_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.038450, 1.459073, 1.342861),
            quaternion_world_wxyz=(0.339748, -0.163418, 0.060071, 0.924261),
        ),
        LayoutId.STUDIO_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(-0.308639, 1.303643, 1.360567),
            quaternion_world_wxyz=(0.486479, -0.133382, 0.075437, 0.860149),
        ),
    }

    def _task_layout_spec(self) -> Any:
        return replace(
            self.current_layout_spec,
            build_items=tuple(
                replace(
                    item,
                    fix_root_link=True,
                    joint_initial_qpos={"joint_2": M05_DISPLAY_HEIGHT_JOINT_QPOS},
                    joint_position_latches={"joint_2": M05_DISPLAY_HEIGHT_LATCH},
                )
                if getattr(item, "name", None) == "display"
                else item
                for item in self.current_layout_spec.build_items
                if getattr(item, "name", None) not in M05_REMOVED
            ),
        )

    def _bind_task_entities(self) -> None:
        self.display = self.layout_scene.require_entity("display")
        try:
            self.button_joint = self.display.active_joints_map[M05_BUTTON_JOINT]
        except KeyError as exc:
            raise RuntimeError(
                "Display asset 3392 must expose joint_0 power button."
            ) from exc
        self.button_link = self.button_joint.child_link or self.display
        self._configure_button_collision_envelope()
        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            # ``link_2`` already carries the loader's mesh transform; this
            # overlay uses the same local XY panel frame and leaves a bezel
            # margin around the imported screen mesh.
            # Keep the status panel inset from the bezel; this also avoids
            # protruding outside the display under camera/viewpoint changes.
            # Nearly fill the imported screen mesh while leaving a narrow
            # bezel margin so the feedback remains visibly attached.
            # Match the black active-display area; the white bezel remains
            # visible around the blue powered-on surface.
            half_size=[0.285, 0.185, 0.002],
            material=sapien.render.RenderMaterial(base_color=[0.05, 0.35, 0.95, 1.0]),
        )
        builder.initial_pose = sapien.Pose(p=[0.0, 0.0, -10.0])
        self.screen_on_visual = builder.build_kinematic(name="m05_screen_on_visual")

    def _configure_button_collision_envelope(self) -> None:
        """Increase M05 button contact margin without changing its mesh."""

        collision_shape_count = 0
        for raw_body in self.button_link._bodies:
            for collision_shape in raw_body.get_collision_shapes():
                contact_offset = float(collision_shape.get_contact_offset())
                if M05_BUTTON_COLLISION_REST_OFFSET_M >= contact_offset:
                    raise ValueError(
                        "M05 button collision rest offset must remain below the "
                        f"PhysX contact offset, got rest="
                        f"{M05_BUTTON_COLLISION_REST_OFFSET_M} and "
                        f"contact={contact_offset}."
                    )
                collision_shape.set_rest_offset(
                    M05_BUTTON_COLLISION_REST_OFFSET_M
                )
                collision_shape_count += 1
        if collision_shape_count == 0:
            raise RuntimeError("M05 button link has no collision shapes to configure.")

    def _physical_build_values(self) -> tuple[Any | None, Any | None]:
        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return None, None
        values = self._generalization_values(
            (
                "display_mass_log_scale",
                "display_friction",
                "button_friction",
                "button_damping",
            ),
            cache_name="_m05_physical_values",
        )
        return None, {
            "display": ArticulationBuildValues(
                physical_materials=tuple(
                    scaled_physical_material(
                        STUDIO_PHYSICAL_MATERIAL,
                        friction_scale=float(v),
                        restitution=0.01,
                    )
                    for v in positive_scale(values["display_friction"])
                ),
                mass_scales=tuple(
                    float(v) for v in positive_scale(values["display_mass_log_scale"])
                ),
                joint_friction={
                    M05_BUTTON_JOINT: tuple(float(v) for v in values["button_friction"])
                },
                joint_damping={
                    M05_BUTTON_JOINT: tuple(float(v) for v in values["button_damping"])
                },
            )
        }

    def _initialize_episode(self, env_idx: Any, options: dict[str, Any]) -> None:
        del options
        self.layout_scene.initialize_episode(
            env_idx, self._batched_episode_rng, self.device
        )
        batch_size = len(env_idx)
        qpos = self.display.get_qpos()
        limits = self.button_joint.limits.reshape(self.num_envs, -1, 2)
        qpos[env_idx, self.button_joint.active_index] = limits[env_idx, 0, 0]
        self.display.set_qpos(qpos)
        self.display.set_qvel(torch.zeros_like(qpos))
        if self.generalization_axis is GeneralizationAxis.SPATIAL:
            values = self._generalization_values(
                ("display_x_offset_m", "display_y_offset_m", "display_yaw_offset_rad"),
                cache_name="_m05_spatial_values",
            )
            indices = env_idx.detach().cpu().numpy()
            site_pose = self._site_pose(batch_size)
            local = site_pose.inv() * self.display.pose[env_idx]
            offset = torch.zeros((batch_size, 3), device=self.device)
            offset[:, 0] = torch.as_tensor(
                values["display_x_offset_m"][indices], device=self.device
            )
            offset[:, 1] = torch.as_tensor(
                values["display_y_offset_m"][indices], device=self.device
            )
            yaw = torch.as_tensor(
                values["display_yaw_offset_rad"][indices], device=self.device
            )
            yaw_q = torch.stack(
                (
                    torch.cos(yaw / 2),
                    torch.zeros_like(yaw),
                    torch.zeros_like(yaw),
                    torch.sin(yaw / 2),
                ),
                dim=1,
            )
            randomized_local_pose = Pose.create_from_pq(
                p=local.p + offset, q=local.q
            ) * Pose.create_from_pq(q=yaw_q)
            self.display.set_pose(site_pose * randomized_local_pose)
        self._reset_robots(batch_size)
        if not hasattr(self, "_stable_steps"):
            self._initial_display_pose = self.display.pose.p.clone()
            self._stable_steps = torch.zeros(
                self.num_envs, dtype=torch.int64, device=self.device
            )
            self._button_ever_touched = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._press_ever_reached = torch.zeros_like(self._button_ever_touched)
        else:
            self._initial_display_pose[env_idx] = self.display.pose.p[env_idx]
        self._stable_steps[env_idx] = 0
        self._button_ever_touched[env_idx] = False
        self._press_ever_reached[env_idx] = False
        self._m05_last_contact_report = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._m05_last_press_report = torch.zeros_like(self._m05_last_contact_report)
        self._update_screen_overlay()

    def evaluate(self) -> dict[str, Any]:
        button_near = self._finger_is_near_button()
        qpos = self.display.get_qpos()[:, self.button_joint.active_index].reshape(-1)
        limits = self.button_joint.limits.reshape(self.num_envs, -1, 2)[:, 0]
        depression = qpos - limits[:, 0]
        travel = (limits[:, 1] - limits[:, 0]).clamp_min(1e-5)
        contact_force = self._max_touch_force(
            (self.button_joint.child_link or self.display,)
        )
        # Only force measured on the button link counts as contact.  Button
        # displacement alone can be caused by display motion or solver drift
        # when the hand touches the screen/bezel, which must not satisfy the
        # task's contact prerequisite.
        contact = contact_force >= self.SUCCESS_SPEC.contact_force_n
        pressed = depression / travel >= self.SUCCESS_SPEC.trigger_fraction
        contact_report = contact & ~self._m05_last_contact_report
        press_report = pressed & ~self._m05_last_press_report
        if bool(contact_report.any().item()):
            print(
                "[m05] button contact detected "
                f"force_n={float(contact_force.max().item()):.3f} "
                f"depression_mm={float((depression.max() * 1000.0).item()):.2f} "
                f"press_fraction={float((depression / travel).max().item()):.3f}",
                flush=True,
            )
        if bool(press_report.any().item()):
            print(
                "[m05] button press threshold reached "
                f"force_n={float(contact_force.max().item()):.3f} "
                f"depression_mm={float((depression.max() * 1000.0).item()):.2f} "
                f"press_fraction={float((depression / travel).max().item()):.3f} "
                f"required_fraction={self.SUCCESS_SPEC.trigger_fraction:.3f}",
                flush=True,
            )
        self._m05_last_contact_report |= contact
        self._m05_last_press_report |= pressed
        self._button_ever_touched |= contact
        self._press_ever_reached |= self._button_ever_touched & pressed
        self._stable_steps = torch.where(
            self._press_ever_reached,
            self._stable_steps + 1,
            torch.zeros_like(self._stable_steps),
        )
        success = self._stable_steps >= self.SUCCESS_SPEC.stable_steps
        if bool(success.any().item()):
            print(
                "[m05] success condition met "
                f"stable_steps={int(self._stable_steps.max().item())} "
                f"required_steps={self.SUCCESS_SPEC.stable_steps}",
                flush=True,
            )
        self._update_screen_overlay()
        return {
            "success": success,
            "power_button_proximity": button_near | self._button_ever_touched,
            "power_button_contact": self._button_ever_touched,
            "display_turned_on": success,
            "button_pressed": self._press_ever_reached,
            "button_contact_force_n": contact_force,
            "button_depression_m": depression,
            "button_press_fraction": depression / travel,
            "button_stable_steps": self._stable_steps.clone(),
        }

    def _finger_is_near_button(self) -> Any:
        """Return whether any fingertip is within the button approach radius."""

        tip_links = tuple(getattr(self.agent, "tip_links", ()))
        if not tip_links:
            raise RuntimeError(
                "M05 requires the robot agent to expose fingertip links through "
                "tip_links for proximity-stage evaluation."
            )
        fingertip_positions = torch.stack(
            [link.pose.p for link in tip_links],
            dim=1,
        )
        button_position = self._button_interaction_position().unsqueeze(1)
        minimum_distance = torch.linalg.norm(
            fingertip_positions - button_position,
            dim=-1,
        ).amin(dim=1)
        return minimum_distance < M05_BUTTON_PROXIMITY_THRESHOLD_M

    def _button_interaction_position(self) -> Any:
        """Return the calibrated centre of the physical button in world space."""

        button_link = self.button_joint.child_link
        if button_link is None:
            raise RuntimeError(
                f"M05 button joint {M05_BUTTON_JOINT!r} has no child link."
            )
        local_position = torch.as_tensor(
            M05_BUTTON_INTERACTION_LOCAL_XYZ,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        interaction_pose = Pose.create(button_link.pose) * Pose.create_from_pq(
            p=local_position
        )
        return interaction_pose.p

    def _update_screen_overlay(self) -> None:
        screen_link = self.display.links_map.get("link_2")
        if screen_link is None:
            marker_position = self.display.pose.p.clone()
            marker_position[:, 1] -= 0.012
            marker_position[:, 2] += 0.08
            marker_quaternion = self.display.pose.q
        else:
            local_front = Pose.create_from_pq(
                # [B, 3]: offset the overlay from the scaled screen surface.
                p=torch.tensor(
                    [[0.0, 0.0, 0.012]],
                    device=self.device,
                    dtype=torch.float32,
                ).repeat(self.num_envs, 1)
            )
            marker_pose = screen_link.pose * local_front
            marker_position = marker_pose.p.clone()
            marker_quaternion = marker_pose.q
        marker_position[~self._press_ever_reached, 2] -= 20.0
        self.screen_on_visual.set_pose(
            Pose.create_from_pq(p=marker_position, q=marker_quaternion)
        )
