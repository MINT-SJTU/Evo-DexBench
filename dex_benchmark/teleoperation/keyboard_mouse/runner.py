from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from dex_benchmark.sim.envs.robot_state import parse_robot_frame_state
from dex_benchmark.sim.tasks.runtime import ManipulatorRole
from dex_benchmark.teleoperation.keyboard_mouse.action_builder import (
    PandaShadowKeyboardSemanticActionBuilder,
)
from dex_benchmark.teleoperation.keyboard_mouse.input_state import (
    IdlePolicy,
    SceneViewerCameraUid,
    TeleopCommand,
)
from dex_benchmark.teleoperation.keyboard_mouse.keymap import KeyboardTeleopKeymap
from dex_benchmark.teleoperation.keyboard_mouse.sapien_keyboard import SapienKeyboardReader
from dex_benchmark.utils.transform import to_numpy


@dataclass(frozen=True)
class KeyboardMouseTeleopConfig:
    translation_step_m: float = 0.01
    rotation_step_rad: float = 0.03
    control_hz: float = 20.0
    idle_policy: IdlePolicy = "freeze_render"
    exit_on_success: bool = True
    no_time_limit: bool = False
    reset_ee_target_each_step: bool = True
    print_state: bool = False
    print_tactile: bool = False
    print_every_steps: int = 20
    tactile_force_threshold_n: float = 0.05
    tactile_top_k: int = 8
    viewer_camera_uid: SceneViewerCameraUid | None = None


class KeyboardMouseTeleopRunner:
    """Real-time keyboard teleoperation loop for Panda + Shadow Hand tasks."""

    def __init__(
        self,
        env: Any,
        *,
        config: KeyboardMouseTeleopConfig | None = None,
        keymap: KeyboardTeleopKeymap | None = None,
    ) -> None:
        self.env = env
        self.config = config or KeyboardMouseTeleopConfig()
        self.keymap = keymap or KeyboardTeleopKeymap()
        if self.config.idle_policy not in ("freeze_render", "hold_step"):
            raise ValueError(f"Unsupported idle policy: {self.config.idle_policy}")
        if self.config.control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {self.config.control_hz}.")
        if self.config.print_every_steps <= 0:
            raise ValueError(
                "print_every_steps must be positive, got "
                f"{self.config.print_every_steps}."
            )
        if self.config.tactile_force_threshold_n < 0.0:
            raise ValueError(
                "tactile_force_threshold_n must be non-negative, got "
                f"{self.config.tactile_force_threshold_n}."
            )
        if self.config.tactile_top_k <= 0:
            raise ValueError(
                f"tactile_top_k must be positive, got {self.config.tactile_top_k}."
            )

        self.action_builder: PandaShadowKeyboardSemanticActionBuilder | None = None
        self.viewer: Any | None = None
        self.keyboard_reader: SapienKeyboardReader | None = None
        self.hand_joint_qpos_debug_enabled = False
        self.previous_debug_hand_qpos: dict[str, float] | None = None

    def run(self, *, seed: int | None = None) -> None:
        obs, _info = self.env.reset(seed=seed)
        self.action_builder = PandaShadowKeyboardSemanticActionBuilder.from_env(
            self.env
        )
        self.action_builder.reset_targets()
        self._sync_all_hand_targets_from_obs(obs)
        self.viewer = self._render()
        if self.config.viewer_camera_uid is not None:
            self._set_viewer_camera_from_sensor(self.config.viewer_camera_uid)
        self.keyboard_reader = SapienKeyboardReader(
            self.viewer,
            keymap=self.keymap,
            translation_step_m=self.config.translation_step_m,
            rotation_step_rad=self.config.rotation_step_rad,
        )

        print(self.keymap.help_text())
        print(
            "[teleop] Active manipulator: "
            f"{self.action_builder.active_manipulator_role}."
        )
        print("Click the SAPIEN viewer window before pressing keys.")

        step_idx = 0
        episode_done = False
        time_limit_reported = False
        control_dt = 1.0 / self.config.control_hz
        next_step_time = time.perf_counter()

        while True:
            now = time.perf_counter()
            if now < next_step_time:
                self._render_and_poll_debug()
                time.sleep(min(0.002, max(0.0, next_step_time - now)))
                continue
            next_step_time = now + control_dt

            assert self.keyboard_reader is not None
            command = self.keyboard_reader.poll()

            if command.quit_requested:
                break
            if command.help_requested:
                print(self.keymap.help_text())
            if command.switch_manipulator_requested:
                assert self.action_builder is not None
                active_role = self.action_builder.switch_active_manipulator()
                self._sync_hand_target_from_obs(
                    role=active_role,
                    obs=obs,
                )
                self.previous_debug_hand_qpos = (
                    self._read_hand_qpos_by_joint_name(
                        role=active_role,
                        obs=obs,
                    )
                    if self.hand_joint_qpos_debug_enabled
                    else None
                )
                print(f"[teleop] Active manipulator: {active_role}.")
            if command.debug_toggle_requested:
                self.hand_joint_qpos_debug_enabled = not self.hand_joint_qpos_debug_enabled
                self.previous_debug_hand_qpos = (
                    self._read_hand_qpos_by_joint_name(obs=obs)
                    if self.hand_joint_qpos_debug_enabled
                    else None
                )
                state_text = "enabled" if self.hand_joint_qpos_debug_enabled else "disabled"
                print(f"[hand-debug] actual hand qpos change logging {state_text}.")
            if command.viewer_camera_uid_requested is not None:
                self._set_viewer_camera_from_sensor(
                    command.viewer_camera_uid_requested
                )
            if command.reset_requested:
                obs, _info = self.env.reset(seed=seed)
                assert self.action_builder is not None
                self.action_builder.reset_targets()
                self._sync_all_hand_targets_from_obs(obs)
                self.previous_debug_hand_qpos = (
                    self._read_hand_qpos_by_joint_name(obs=obs)
                    if self.hand_joint_qpos_debug_enabled
                    else None
                )
                episode_done = False
                time_limit_reported = False
                step_idx = 0
                self._render_and_poll_debug(obs=obs)
                continue

            should_step = command.has_robot_action() or self.config.idle_policy == "hold_step"
            if episode_done or not should_step:
                self._render_and_poll_debug(obs=obs)
                continue

            action_command = command if command.has_robot_action() else TeleopCommand()
            assert self.action_builder is not None
            action = self._encode_command(action_command)
            if self.config.reset_ee_target_each_step:
                self._reset_ee_target_to_current_pose()
            obs, _reward, terminated, truncated, info = self.env.step(action)
            step_idx += 1

            should_print = step_idx % self.config.print_every_steps == 0
            if self.config.print_state and should_print:
                self._print_state(step_idx=step_idx, obs=obs, info=info)
            if self.config.print_tactile and should_print:
                self._print_tactile(step_idx=step_idx, obs=obs)

            success = _any_truthy(info.get("success", False))
            terminated_bool = _any_truthy(terminated)
            truncated_bool = _any_truthy(truncated)
            if success:
                print(f"[teleop] Task succeeded at control step {step_idx}.")
                if self.config.exit_on_success:
                    break
                episode_done = True
                print("Press R to reset or Q/Escape to quit.")
            elif terminated_bool or truncated_bool:
                if terminated_bool:
                    print(f"[teleop] Episode terminated at control step {step_idx}.")
                    break
                if self.config.no_time_limit:
                    if not time_limit_reported:
                        print(
                            "[teleop] Time limit reached; continuing because "
                            "--no-time-limit is enabled."
                        )
                        time_limit_reported = True
                else:
                    print(f"[teleop] Episode truncated at control step {step_idx}.")
                    break

            self._render_and_poll_debug(obs=obs)

    def _encode_command(self, command: TeleopCommand) -> Any:
        """Route keyboard commands through the benchmark semantic action API."""

        if self.action_builder is None:
            raise RuntimeError("Keyboard action builder is not initialized.")
        semantic_action = self.action_builder.build_semantic_action(command)
        encode_action = self.env.get_wrapper_attr("encode_action")
        return encode_action(semantic_action)

    def _render(self) -> Any:
        if hasattr(self.env, "render_human"):
            return self.env.render_human()
        return self.env.render()

    def _render_and_poll_debug(self, obs: Any | None = None) -> Any:
        render_result = self._render()
        self._sync_and_print_changed_hand_qpos(obs=obs)
        return render_result

    def _set_viewer_camera_from_sensor(
        self,
        camera_uid: SceneViewerCameraUid,
    ) -> None:
        if self.viewer is None:
            raise RuntimeError("The SAPIEN viewer must exist before selecting a camera.")

        base_env = getattr(self.env, "unwrapped", self.env)
        sensors = getattr(base_env, "_sensors", {}) or {}
        try:
            sensor = sensors[camera_uid]
        except KeyError as exc:
            known_camera_uids = ", ".join(sorted(sensors)) or "<none>"
            raise ValueError(
                f"Viewer camera {camera_uid!r} is not an environment sensor. "
                f"Known sensors: {known_camera_uids}."
            ) from exc

        camera = getattr(sensor, "camera", None)
        global_pose = getattr(camera, "global_pose", None)
        if global_pose is None:
            raise AttributeError(
                f"Sensor {camera_uid!r} does not expose a rendered global pose."
            )

        # The base environment selects a wrist sensor in the Human Viewer's
        # Camera dropdown by default. Switch back to the free camera before
        # applying the requested scene-camera pose; otherwise the focused wrist
        # camera overwrites this pose again on the next viewer frame.
        control_window = getattr(self.viewer, "control_window", None)
        if callable(control_window):
            control_window = control_window()
        if control_window is not None:
            camera_items = list(getattr(control_window, "camera_items", ()))
            if camera_items and camera_items[0] == "None":
                control_window.camera_index = 0

        self.viewer.set_camera_pose(global_pose[0].sp)

        # Keep subsequent mouse/keyboard camera movement anchored to the pose
        # just applied instead of the previous wrist/free-camera controller pose.
        sync_fps_camera_controller = getattr(
            control_window,
            "_sync_fps_camera_controller",
            None,
        )
        if callable(sync_fps_camera_controller):
            sync_fps_camera_controller()

        camera_configs = getattr(base_env, "_sensor_configs", {}) or {}
        camera_config = camera_configs.get(camera_uid)
        window = getattr(self.viewer, "window", None)
        if camera_config is not None and window is not None:
            fov_values = np.asarray(camera_config.fov, dtype=np.float64).reshape(-1)
            if fov_values.size == 0 or not np.isfinite(fov_values[0]):
                raise ValueError(
                    f"Sensor {camera_uid!r} has an invalid FOV: {camera_config.fov!r}."
                )
            window.set_camera_parameters(
                near=float(camera_config.near),
                far=float(camera_config.far),
                fovy=float(fov_values[0]),
            )
        print(f"[teleop] Viewer aligned to sensor camera {camera_uid!r}.")

    def _reset_ee_target_to_current_pose(self) -> None:
        if self.action_builder is None:
            return
        try:
            agent_for_role = self.env.get_wrapper_attr("agent_for_role")
        except Exception:
            return
        for role in self.action_builder.manipulator_roles:
            agent = agent_for_role(ManipulatorRole(role))
            controller = getattr(agent, "controller", None)
            component_controllers = getattr(controller, "controllers", None)
            if not isinstance(component_controllers, dict):
                continue
            arm_controller = component_controllers.get("arm")
            if arm_controller is None:
                continue
            controller_config = getattr(arm_controller, "config", None)
            if not bool(getattr(controller_config, "use_target", False)):
                continue
            current_ee_pose_at_base = getattr(
                arm_controller,
                "ee_pose_at_base",
                None,
            )
            if current_ee_pose_at_base is not None:
                arm_controller._target_pose = current_ee_pose_at_base

    def _read_hand_qpos_by_joint_name(
        self,
        *,
        role: str | None = None,
        obs: Any | None = None,
    ) -> dict[str, float]:
        if self.action_builder is None:
            return {}
        target_role = role or self.action_builder.active_manipulator_role
        qpos = self._read_robot_qpos_env0(target_role)
        if qpos is None and obs is not None:
            qpos = self._read_obs_qpos_env0(obs, target_role)
        if qpos is None:
            return {}
        active_joint_names = self._active_joint_names(target_role)
        if len(active_joint_names) < qpos.shape[0]:
            return {}

        hand_joint_names = set(
            self.action_builder.hand_joint_names_by_role[target_role]
        )
        values: dict[str, float] = {}
        for joint_index, joint_name in enumerate(active_joint_names[: qpos.shape[0]]):
            if joint_name in hand_joint_names:
                values[joint_name] = float(qpos[joint_index])
        return values

    def _read_robot_qpos_env0(self, role: str) -> np.ndarray | None:
        robot = self._robot(role)
        if robot is None:
            return None

        qpos_value = getattr(robot, "qpos", None)
        if qpos_value is None:
            get_qpos = getattr(robot, "get_qpos", None)
            if callable(get_qpos):
                try:
                    qpos_value = get_qpos()
                except Exception:
                    qpos_value = None
        if qpos_value is None:
            return None

        qpos = np.asarray(to_numpy(qpos_value), dtype=np.float32)
        if qpos.ndim == 2:
            qpos = qpos[0]
        if qpos.ndim != 1:
            return None
        return qpos

    def _read_obs_qpos_env0(
        self,
        obs: Any,
        role: str,
    ) -> np.ndarray | None:
        if not isinstance(obs, dict):
            return None
        manipulators = obs.get("manipulators")
        if not isinstance(manipulators, dict):
            return None
        manipulator = manipulators.get(role)
        if (
            not isinstance(manipulator, dict)
            or manipulator.get("joint_position") is None
        ):
            return None
        qpos = np.asarray(
            to_numpy(manipulator["joint_position"]),
            dtype=np.float32,
        )
        if qpos.ndim == 2:
            qpos = qpos[0]
        if qpos.ndim != 1:
            return None
        return qpos

    def _robot(self, role: str) -> Any | None:
        agent = self._agent(role)
        return None if agent is None else getattr(agent, "robot", None)

    def _agent(self, role: str) -> Any | None:
        try:
            agent_for_role = self.env.get_wrapper_attr("agent_for_role")
            return agent_for_role(ManipulatorRole(role))
        except Exception:
            return None

    def _active_joint_names(self, role: str) -> tuple[str, ...]:
        robot = self._robot(role)
        if robot is None:
            return ()

        active_joints_getter = getattr(robot, "get_active_joints", None)
        if callable(active_joints_getter):
            active_joints = active_joints_getter()
        else:
            active_joints = getattr(robot, "active_joints", ())
        return tuple(str(getattr(joint, "name", "")) for joint in active_joints)

    def _sync_all_hand_targets_from_obs(self, obs: Any) -> None:
        if self.action_builder is None:
            return
        for role in self.action_builder.manipulator_roles:
            self._sync_hand_target_from_obs(role=role, obs=obs)

    def _sync_hand_target_from_obs(self, *, role: str, obs: Any) -> None:
        if self.action_builder is None:
            return
        hand_qpos = self._read_hand_qpos_by_joint_name(
            role=role,
            obs=obs,
        )
        if hand_qpos:
            self.action_builder.sync_hand_targets_from_joint_qpos(
                hand_qpos,
                role=role,
            )

    def _sync_and_print_changed_hand_qpos(
        self,
        obs: Any | None = None,
        *,
        atol: float = 1e-6,
    ) -> None:
        if not self.hand_joint_qpos_debug_enabled:
            return
        assert self.action_builder is not None
        active_role = self.action_builder.active_manipulator_role
        current_hand_qpos = self._read_hand_qpos_by_joint_name(obs=obs)
        if not current_hand_qpos:
            return
        self.action_builder.sync_hand_targets_from_joint_qpos(
            current_hand_qpos,
            role=active_role,
        )
        if self.previous_debug_hand_qpos is None:
            self.previous_debug_hand_qpos = current_hand_qpos
            return

        for joint_name in sorted(current_hand_qpos):
            previous_angle_rad = self.previous_debug_hand_qpos.get(joint_name)
            if previous_angle_rad is None:
                continue
            current_angle_rad = current_hand_qpos[joint_name]
            if abs(current_angle_rad - previous_angle_rad) <= atol:
                continue
            print(
                "[hand-debug] "
                f"{active_role}/{joint_name}: {previous_angle_rad:.4f} rad -> "
                f"{current_angle_rad:.4f} rad"
            )
        self.previous_debug_hand_qpos = current_hand_qpos

    def _print_state(self, *, step_idx: int, obs: Any, info: dict[str, Any]) -> None:
        try:
            active_role = (
                self.action_builder.active_manipulator_role
                if self.action_builder is not None
                else "right"
            )
            robot_state = parse_robot_frame_state(obs, role=active_role)
            eef_pos_world = to_numpy(robot_state.eef.pos_world)
            palm_pos_world = to_numpy(robot_state.palm.pos_world)
            if eef_pos_world.ndim > 1:
                eef_pos_world = eef_pos_world[0]
            if palm_pos_world.ndim > 1:
                palm_pos_world = palm_pos_world[0]
            success = bool(np.asarray(info.get("success", False)).any())
            print(
                "[teleop]",
                {
                    "step": step_idx,
                    "active_manipulator": active_role,
                    "eef_pos_world": np.asarray(eef_pos_world, dtype=np.float32).tolist(),
                    "palm_pos_world": np.asarray(palm_pos_world, dtype=np.float32).tolist(),
                    "success": success,
                },
            )
        except Exception as exc:
            print(f"[teleop] state logging skipped: {exc}")

    def _print_tactile(self, *, step_idx: int, obs: Any) -> None:
        """Print an env-0 tactile summary for the active manipulator."""

        try:
            active_role = (
                self.action_builder.active_manipulator_role
                if self.action_builder is not None
                else "right"
            )
            normal_force_n = self._read_tactile_force_env0(
                obs=obs,
                role=active_role,
            )
            agent = self._agent(active_role)
            taxel_specs = tuple(getattr(agent, "taxel_specs", ()))
            if normal_force_n is None:
                raise ValueError(
                    f"observation has no {active_role!r} taxel_normal_force"
                )
            if normal_force_n.shape != (len(taxel_specs),):
                raise ValueError(
                    "taxel force/layout size mismatch: "
                    f"force={normal_force_n.shape}, specs={len(taxel_specs)}"
                )

            threshold_n = self.config.tactile_force_threshold_n
            active_indices = np.flatnonzero(normal_force_n >= threshold_n)
            sorted_indices = active_indices[
                np.argsort(normal_force_n[active_indices])[::-1]
            ]
            top_indices = sorted_indices[: self.config.tactile_top_k]

            group_force_n: dict[str, float] = {}
            top_taxels: list[str] = []
            for taxel_index in active_indices:
                spec = taxel_specs[int(taxel_index)]
                finger_name = getattr(spec, "finger_name", None)
                region_name = str(getattr(spec, "region_name", "unknown"))
                group_name = (
                    region_name
                    if finger_name is None
                    else f"{finger_name}/{region_name}"
                )
                group_force_n[group_name] = (
                    group_force_n.get(group_name, 0.0)
                    + float(normal_force_n[taxel_index])
                )

            max_force_n = (
                float(np.max(normal_force_n))
                if normal_force_n.size > 0
                else 0.0
            )
            for taxel_index in top_indices:
                spec = taxel_specs[int(taxel_index)]
                force_n = float(normal_force_n[taxel_index])
                area_m2 = float(getattr(spec, "area_m2", 0.0))
                if area_m2 <= 0.0:
                    raise ValueError(
                        f"taxel {getattr(spec, 'name', taxel_index)!r} "
                        f"has invalid area {area_m2}"
                    )
                pressure_kpa = force_n / area_m2 / 1000.0
                taxel_name = str(getattr(spec, "name", taxel_index))
                taxel_name = taxel_name.removeprefix("rh_taxel_")
                taxel_name = taxel_name.removeprefix("lh_taxel_")
                top_taxels.append(
                    f"{taxel_name}={force_n:.3f}N/{pressure_kpa:.2f}kPa"
                )

            group_text = ", ".join(
                f"{group_name}={force_n:.3f}N"
                for group_name, force_n in sorted(group_force_n.items())
            )
            top_text = ", ".join(top_taxels)
            print(
                "[tactile] "
                f"step={step_idx} role={active_role} "
                f"active={len(active_indices)}/{len(taxel_specs)} "
                f"total={float(np.sum(normal_force_n)):.3f}N "
                f"max={max_force_n:.3f}N "
                f"groups=[{group_text}] top=[{top_text}]"
            )
        except Exception as exc:
            print(f"[tactile] logging skipped: {exc}")

    @staticmethod
    def _read_tactile_force_env0(
        *,
        obs: Any,
        role: str,
    ) -> np.ndarray | None:
        if not isinstance(obs, dict):
            return None
        manipulators = obs.get("manipulators")
        if not isinstance(manipulators, dict):
            return None
        manipulator = manipulators.get(role)
        if not isinstance(manipulator, dict):
            return None
        force_value = manipulator.get("taxel_normal_force")
        if force_value is None:
            return None
        normal_force_n = np.asarray(
            to_numpy(force_value),
            dtype=np.float32,
        )
        if normal_force_n.ndim == 2:
            normal_force_n = normal_force_n[0]
        if normal_force_n.ndim != 1:
            return None
        return normal_force_n


def _any_truthy(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        return bool(np.asarray(value).any())
    except Exception:
        return bool(value)
