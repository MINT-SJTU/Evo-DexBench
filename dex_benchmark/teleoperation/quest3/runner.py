from __future__ import annotations

from collections.abc import Mapping
from collections import deque
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from dex_benchmark.sim.tasks.runtime import ManipulatorRole
from dex_benchmark.teleoperation.quest3.action_builder import (
    Quest3ActionPair,
    Quest3DualPandaShadowActionBuilder,
    Quest3PandaShadowActionBuilder,
)
from dex_benchmark.teleoperation.quest3.action_backend import Quest3ActionBackend
from dex_benchmark.teleoperation.quest3.benchmark_adapter import BenchmarkSemanticBackend
from dex_benchmark.teleoperation.quest3.control_step import Quest3ControlStep
from dex_benchmark.teleoperation.quest3.contact_action_filter import (
    DEFAULT_MAX_CONTACT_FORCE_N,
    ContactActionFilter,
    ContactActionFilterConfig,
)
from dex_benchmark.teleoperation.quest3.mapping import Quest3WristMotionConfig
from dex_benchmark.teleoperation.quest3.native_adapter import LegacyNativeBackend
from dex_benchmark.teleoperation.quest3.observation_adapter import Quest3ObservationAdapter
from dex_benchmark.teleoperation.quest3.recording import Quest3JSONLRecorder
from dex_benchmark.teleoperation.quest3.stream import (
    Quest3HandInput,
    Quest3InputFrame,
    Quest3InputStream,
)


TeleoperationMode = Literal["single", "dual"]

_IK_RECOVERY_ACTION_EPSILON = 1e-6
_IK_RECOVERY_QPOS_ATOL_RAD = 1e-7
_IK_RECOVERY_POSITION_ERROR_M = 1e-4
_IK_RECOVERY_ROTATION_ERROR_RAD = 1e-4
_IK_RECOVERY_QUATERNION_NORM_EPSILON = 1e-12


@dataclass(frozen=True)
class Quest3TeleopConfig:
    mode: TeleoperationMode = "single"
    hand_side: str = "right"
    control_hz: float = 30.0
    stale_frame_timeout_s: float = 5.0
    tracking_loss_reset_timeout_s: float = 5.0
    tracking_wait_timeout_s: float | None = None
    render: bool = True
    # 1 renders on every control tick, which is what teleoperation wants: at 30
    # presents per second on a 60Hz display the swapchain never fills, so the
    # SAPIEN present does not stall on vblank and costs about 1ms per tick
    # (measured, RTX 4090 + X11). Values above 1 trade operator-visible latency
    # (one skipped tick = one extra control period of stale pixels) for that 1ms,
    # so raise it only when a profile shows the present actually blocking.
    render_every_n: int = 1
    # Quest operator-view capture cadence in control ticks. When a desktop
    # viewer is enabled, it must align with viewer presents because that
    # viewer processes free-camera input.
    view_every_n: int = 1
    # Pay the first env.step/sensor initialization cost before accepting the
    # first tracked hand frame, so cold-start work is not perceived as input lag.
    startup_warmup_steps: int = 1
    # Pause after reset/render so the operator can prepare before control and
    # canonical recording begin.  Rendering and viewer command polling continue
    # during this interval.
    startup_preparation_delay_s: float = 0.0
    ik_stall_recovery_steps: int = 3
    joint_limit_reanchor_margin_rad: float = 0.2
    max_joint_limit_excess_rad: float = 0.1
    recoverable_joint_limit_excess_rad: float = 0.25
    joint_limit_hold_steps: int = 3
    max_joint_velocity_rad_s: float = 200.0
    warning_joint_velocity_rad_s: float = 180.0
    max_taxel_force_n: float = DEFAULT_MAX_CONTACT_FORCE_N
    contact_light_threshold_n: float = 5.0
    contact_medium_threshold_n: float = 20.0
    max_hand_force_n: float = DEFAULT_MAX_CONTACT_FORCE_N
    light_arm_normal_scale: float = 0.25
    light_finger_closure_scale: float = 0.25
    medium_finger_closure_scale: float = 0.10
    print_tactile: bool = False
    tactile_force_threshold_n: float = 1.0
    tactile_print_every_steps: int = 1
    exit_on_success: bool = True
    no_time_limit: bool = False
    require_squeeze: bool = False
    recenter_on_tap: bool = False
    retargeting_config_path: str | Path | None = None
    left_retargeting_config_path: str | Path | None = None
    right_retargeting_config_path: str | Path | None = None
    retargeting_urdf_root: str | Path | None = None
    record_path: str | Path | None = None
    wrist_motion: Quest3WristMotionConfig = Quest3WristMotionConfig()

    def __post_init__(self) -> None:
        if self.mode not in ("single", "dual"):
            raise ValueError(f"mode must be 'single' or 'dual', got {self.mode!r}.")
        if self.hand_side not in ("left", "right"):
            raise ValueError(
                f"hand_side must be 'left' or 'right', got {self.hand_side!r}."
            )
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive.")
        if self.stale_frame_timeout_s <= 0:
            raise ValueError("stale_frame_timeout_s must be positive.")
        if self.tracking_loss_reset_timeout_s < 0:
            raise ValueError("tracking_loss_reset_timeout_s must be non-negative.")
        if self.tracking_wait_timeout_s is not None and self.tracking_wait_timeout_s < 0:
            raise ValueError("tracking_wait_timeout_s must be non-negative when set.")
        if self.render_every_n < 1:
            raise ValueError(
                f"render_every_n must be at least 1, got {self.render_every_n}."
            )
        if self.view_every_n < 1:
            raise ValueError(
                f"view_every_n must be at least 1, got {self.view_every_n}."
            )
        if self.startup_warmup_steps < 0:
            raise ValueError("startup_warmup_steps must be non-negative.")
        if self.startup_preparation_delay_s < 0.0:
            raise ValueError("startup_preparation_delay_s must be non-negative.")
        if self.ik_stall_recovery_steps < 1:
            raise ValueError("ik_stall_recovery_steps must be at least 1.")
        if self.joint_limit_reanchor_margin_rad <= 0.0:
            raise ValueError("joint_limit_reanchor_margin_rad must be positive.")
        if self.max_joint_limit_excess_rad <= 0.0:
            raise ValueError("max_joint_limit_excess_rad must be positive.")
        if (
            self.recoverable_joint_limit_excess_rad
            <= self.max_joint_limit_excess_rad
        ):
            raise ValueError(
                "recoverable_joint_limit_excess_rad must be greater than "
                "max_joint_limit_excess_rad."
            )
        if self.joint_limit_hold_steps < 1:
            raise ValueError("joint_limit_hold_steps must be at least 1.")
        if self.max_joint_velocity_rad_s <= 0.0:
            raise ValueError("max_joint_velocity_rad_s must be positive.")
        if not 0.0 < self.warning_joint_velocity_rad_s <= self.max_joint_velocity_rad_s:
            raise ValueError(
                "warning_joint_velocity_rad_s must be positive and no greater "
                "than max_joint_velocity_rad_s."
            )
        if self.max_taxel_force_n <= 0.0:
            raise ValueError("max_taxel_force_n must be positive.")
        self._validate_contact_config()
        if self.tactile_force_threshold_n < 0.0:
            raise ValueError("tactile_force_threshold_n must be non-negative.")
        if self.tactile_print_every_steps <= 0:
            raise ValueError("tactile_print_every_steps must be positive.")

    def _validate_contact_config(self) -> None:
        ContactActionFilterConfig(
            light_threshold_n=self.contact_light_threshold_n,
            medium_threshold_n=self.contact_medium_threshold_n,
            max_force_n=self.max_hand_force_n,
            light_arm_normal_scale=self.light_arm_normal_scale,
            light_finger_closure_scale=self.light_finger_closure_scale,
            medium_finger_closure_scale=self.medium_finger_closure_scale,
        )


class Quest3TeleopRunner:
    """Run latest-value Quest hand tracking against a ManiSkill environment."""

    def __init__(
        self,
        env: Any,
        stream: Quest3InputStream,
        *,
        config: Quest3TeleopConfig | None = None,
        backend: Quest3ActionBackend | None = None,
        observation_adapter: Quest3ObservationAdapter | None = None,
        on_observation: Callable[[Any], None] | None = None,
        on_step: Callable[[Quest3ControlStep], None] | None = None,
        on_reset: Callable[[Any], None] | None = None,
        on_reset_metadata: Callable[[Mapping[str, Any]], None] | None = None,
        on_recording_start: Callable[..., None] | None = None,
        on_view_frame: Callable[..., Any] | None = None,
        view_ready: Callable[[], bool] | None = None,
        on_viewer_ready: Callable[[Any], None] | None = None,
        on_viewer_command: Callable[[str], None] | None = None,
        viewer_command_keys: Mapping[str, str] | None = None,
    ) -> None:
        self.env = env
        self.stream = stream
        self.config = config or Quest3TeleopConfig()
        self.observation_adapter = observation_adapter
        self.on_observation = on_observation
        self.on_step = on_step
        self.on_reset = on_reset
        self.on_reset_metadata = on_reset_metadata
        self.on_recording_start = on_recording_start
        if (on_view_frame is None) != (view_ready is None):
            raise ValueError(
                "on_view_frame and view_ready must be provided together."
            )
        if (
            on_view_frame is not None
            and self.config.render
            and self.config.view_every_n % self.config.render_every_n != 0
        ):
            raise ValueError(
                "view_every_n must be an integer multiple of render_every_n "
                "when desktop viewer rendering is enabled."
            )
        self.on_view_frame = on_view_frame
        self.view_ready = view_ready
        self.on_viewer_ready = on_viewer_ready
        self.on_viewer_command = on_viewer_command
        raw_viewer_command_keys = {
            "reset": "r",
            "quit": "q",
            "proceed": "e",
        }
        if viewer_command_keys is not None:
            raw_viewer_command_keys.update(viewer_command_keys)
        if set(raw_viewer_command_keys) != {"reset", "quit", "proceed"}:
            raise ValueError(
                "viewer_command_keys must contain exactly reset, quit, and proceed."
            )
        normalized_viewer_keys: dict[str, str] = {}
        for command, key in raw_viewer_command_keys.items():
            if not isinstance(key, str) or len(key) != 1:
                raise ValueError(
                    f"Viewer {command} key must be exactly one character."
                )
            normalized_viewer_keys[command] = key.lower()
        if len(set(normalized_viewer_keys.values())) != 3:
            raise ValueError("Viewer reset, quit, and proceed keys must differ.")
        self._viewer_command_keys = normalized_viewer_keys
        self._viewer_ready_notified = False
        self.stop_requested = False
        self.exit_reason: str | None = None
        self._stop_reason: str | None = None
        self.physics_violation: dict[str, Any] | None = None
        self._manipulator_state_specs: dict[str, dict[str, Any]] = {}
        self._observation_id = 0
        if backend is not None:
            self.backend = backend
        else:
            # Preserve the pre-backend Runner contract for direct library users
            # while making the executable entry point choose its backend
            # explicitly.  The Runner still never inspects action values.
            try:
                self.backend = BenchmarkSemanticBackend(
                    env, default_role=self.config.hand_side
                )
            except AttributeError:
                self.backend = LegacyNativeBackend(env)
        self.single_builder: Quest3PandaShadowActionBuilder | None = None
        self.dual_builder: Quest3DualPandaShadowActionBuilder | None = None
        self.recorder = (
            None
            if self.config.record_path is None
            else Quest3JSONLRecorder(self.config.record_path)
        )
        self._last_error_report_s = 0.0
        self._last_telemetry_s = 0.0
        self._last_wall_time_s: float | None = None
        self._last_input_frame_s: float | None = None
        self._input_rate_hz: float | None = None
        self._last_latency_received_timestamp_s: float | None = None
        self._latency_samples_ms: dict[str, deque[float]] = {
            name: deque(maxlen=300)
            for name in (
                "receive_dt",
                "handler_processing",
                "publish_to_control",
            )
        }
        self._telemetry_actions: dict[str, np.ndarray] = {}
        self._render_tick_index = 0
        self._view_capture_error: Exception | None = None
        self._view_blocked = self.on_view_frame is not None
        self._last_dual_actions: dict[str, Quest3ActionPair] = {}
        self._tracked_dual_sides: set[str] = set()
        self._tracking_lost_since_s: dict[str, float | None] = {
            "left": None,
            "right": None,
        }
        self._tracking_loss_reset_done: dict[str, bool] = {
            "left": False,
            "right": False,
        }
        self._tactile_contact_streak_steps: dict[str, int] = {}
        self._previous_eef_position_m: dict[str, np.ndarray] = {}
        self._latest_contact_diagnostics: dict[str, dict[str, Any]] = {}
        self._latest_observation: Any | None = None
        self._near_joint_limit_roles: set[str] = set()
        self._joint_limit_hold_steps_by_role: dict[str, int] = {}
        self._ik_stall_steps_by_role: dict[str, int] = {}
        self._contact_filter = ContactActionFilter(
            ContactActionFilterConfig(
                light_threshold_n=self.config.contact_light_threshold_n,
                medium_threshold_n=self.config.contact_medium_threshold_n,
                max_force_n=self.config.max_hand_force_n,
                light_arm_normal_scale=self.config.light_arm_normal_scale,
                light_finger_closure_scale=self.config.light_finger_closure_scale,
                medium_finger_closure_scale=self.config.medium_finger_closure_scale,
            )
        )

    def run(self, *, seed: int | None = None) -> None:
        self.physics_violation = None
        self.exit_reason = None
        self._last_dual_actions.clear()
        self._tracked_dual_sides.clear()
        self._tactile_contact_streak_steps.clear()
        self._previous_eef_position_m.clear()
        self._latest_contact_diagnostics.clear()
        self._latest_observation = None
        self._near_joint_limit_roles.clear()
        self._joint_limit_hold_steps_by_role.clear()
        self._ik_stall_steps_by_role.clear()
        self._view_capture_error = None
        self._view_blocked = self.on_view_frame is not None
        self._render_tick_index = 0
        self._viewer_ready_notified = False
        for side in ("left", "right"):
            self._tracking_lost_since_s[side] = None
            self._tracking_loss_reset_done[side] = False
        recording_started = self.on_recording_start is None
        startup_started_s = time.perf_counter()
        reset_result = self.env.reset(seed=seed)
        reset_elapsed_s = time.perf_counter() - startup_started_s
        reset_info: Mapping[str, Any] = {}
        if (
            isinstance(reset_result, tuple)
            and len(reset_result) == 2
            and isinstance(reset_result[1], Mapping)
        ):
            initial_obs = reset_result[0]
            reset_info = reset_result[1]
        else:
            initial_obs = reset_result
        if self.on_reset_metadata is not None:
            self.on_reset_metadata(reset_info)
        self._set_tactile_mapping_diagnostics(self.config.print_tactile)
        self._manipulator_state_specs = self._resolve_manipulator_state_specs()
        build_started_s = time.perf_counter()
        self._build_action_builder()
        build_elapsed_s = time.perf_counter() - build_started_s
        warmup_started_s = time.perf_counter()
        initial_obs = self._startup_warmup(initial_obs)
        self._latest_observation = initial_obs
        self._seed_hand_targets_from_observation(initial_obs)
        self._seed_eef_positions(initial_obs)
        warmup_elapsed_s = time.perf_counter() - warmup_started_s
        observation_started_s = time.perf_counter()
        self._publish_observation(initial_obs, force_image=True)
        observation_elapsed_s = time.perf_counter() - observation_started_s
        reset_hook_started_s = time.perf_counter()
        if self.on_reset is not None:
            self.on_reset(initial_obs)
        reset_hook_elapsed_s = time.perf_counter() - reset_hook_started_s
        render_started_s = time.perf_counter()
        self._render()
        self._wait_for_startup_preparation()
        render_elapsed_s = time.perf_counter() - render_started_s
        tracking_wait_started_s = time.monotonic()
        startup_elapsed_s = time.perf_counter() - startup_started_s
        print(
            "[quest3-teleop] startup_timing "
            f"reset={1000.0 * reset_elapsed_s:.1f}ms "
            f"build={1000.0 * build_elapsed_s:.1f}ms "
            f"warmup={1000.0 * warmup_elapsed_s:.1f}ms "
            f"observation={1000.0 * observation_elapsed_s:.1f}ms "
            f"reset_hook={1000.0 * reset_hook_elapsed_s:.1f}ms "
            f"render={1000.0 * render_elapsed_s:.1f}ms "
            f"total={1000.0 * startup_elapsed_s:.1f}ms",
            flush=True,
        )
        print(
            f"[quest3-teleop] mode={self.config.mode} ready; waiting for fresh hand frames.",
            flush=True,
        )
        print(
            "[quest3-teleop] latency receive_scope=vuer_handler_entry "
            "window=300 frames",
            flush=True,
        )
        if self.recorder is not None:
            print(
                f"[quest3-teleop] recording={self.recorder.output_path}",
                flush=True,
            )

        last_sequence = 0
        latest_action: Any = None
        step_index = 0
        time_limit_reported = False
        control_dt_s = 1.0 / self.config.control_hz
        next_step_s = time.perf_counter()
        perf_started_s = next_step_s
        perf_steps = 0
        perf_build_s = 0.0
        perf_step_s = 0.0
        perf_tactile_s = 0.0
        perf_tactile_mapping_s = 0.0
        perf_tactile_points = 0
        perf_tactile_max_points = 0
        perf_render_s = 0.0
        perf_render_calls = 0
        perf_render_max_s = 0.0

        while True:
            if self.stop_requested:
                self.exit_reason = self._stop_reason or "stop_requested"
                return
            now_s = time.perf_counter()
            if now_s < next_step_s:
                time.sleep(min(0.002, next_step_s - now_s))
                continue
            next_step_s += control_dt_s
            if next_step_s < now_s:
                next_step_s = now_s + control_dt_s

            frame = self.stream.poll()
            control_tick_wall_s: float | None = None
            built_new_action = False
            if frame is not None and frame.sequence != last_sequence:
                last_sequence = frame.sequence
                control_tick_wall_s = time.time()
                control_timestamp_s = control_tick_wall_s
                if self._frame_control_allowed(
                    frame,
                    now_s=control_timestamp_s,
                ) and self._view_control_allowed():
                    build_started_s = time.perf_counter()
                    try:
                        latest_action = self._build_action_pair_or_action(frame)
                        built_new_action = True
                    except ValueError as exc:
                        latest_action = None
                        self._reset_wrist_calibration()
                        self._report_invalid_frame(exc)
                    perf_build_s += time.perf_counter() - build_started_s
                else:
                    latest_action = None
                self._report_telemetry(
                    frame,
                    latest_action if built_new_action else None,
                    control_timestamp_s=control_timestamp_s,
                )
                if built_new_action and not recording_started:
                    assert self.on_recording_start is not None
                    self.on_recording_start(
                        initial_obs,
                        timestamp_s=control_tick_wall_s,
                    )
                    # The calibration frame is deliberately not executed. Reset
                    # the hand target anchor so its filter progress cannot make
                    # the first recorded command two increments away from the
                    # still-stationary robot pose.
                    self._seed_hand_targets_from_observation(initial_obs)
                    recording_started = True
                    latest_action = None
                    built_new_action = False
                    print(
                        "[quest3-teleop] calibration complete; canonical "
                        "recording starts with the next tracked frame.",
                        flush=True,
                    )
            elif latest_action is not None:
                control_tick_wall_s = time.time()
                if not self._view_control_allowed():
                    latest_action = None
                elif frame is None or not self._is_frame_fresh(
                    frame,
                    now_s=control_tick_wall_s,
                ):
                    # A latest-value stream keeps returning its last frame.
                    # Expire the action locally so a stalled WebXR connection
                    # cannot keep driving the robot.
                    latest_action = None
                    self._reset_wrist_calibration()

            if latest_action is None:
                if (
                    not recording_started
                    and self.config.tracking_wait_timeout_s is not None
                    and time.monotonic() - tracking_wait_started_s
                    >= self.config.tracking_wait_timeout_s
                ):
                    self.exit_reason = "tracking_timeout"
                    return
                render_started_s = time.perf_counter()
                self._render_if_due()
                render_elapsed_s = time.perf_counter() - render_started_s
                perf_render_s += render_elapsed_s
                perf_render_calls += 1
                perf_render_max_s = max(perf_render_max_s, render_elapsed_s)
                continue
            if control_tick_wall_s is None:
                control_tick_wall_s = time.time()

            # Translation deltas are consumed once. Rotation keeps taking bounded
            # control-tick steps toward the newest tracked wrist orientation.
            if built_new_action:
                step_action = latest_action
            else:
                step_action = _hold_hand_targets_without_arm_motion(latest_action)
                if (
                    isinstance(step_action, Quest3ActionPair)
                    and self.config.mode == "dual"
                    and self.dual_builder is not None
                ):
                    step_action = self._build_dual_followup_action_pair(step_action)
                elif isinstance(step_action, Quest3ActionPair) and self.single_builder is not None:
                    step_action = self.single_builder.build_rotation_followup_action_pair(step_action)
                elif not isinstance(step_action, dict) and self.single_builder is not None:
                    step_action = self.single_builder.build_rotation_followup_action(step_action)
            if self.stop_requested:
                self.exit_reason = self._stop_reason or "stop_requested"
                return
            step_action, joint_limit_guard_roles = self._guard_near_joint_limits(
                step_action
            )
            step_action, contact_control_diagnostics = self._filter_step_action(step_action)
            step_started_s = time.perf_counter()
            obs, reward, terminated, truncated, info = self.backend.step(step_action)
            recovered_ik_roles = self._recover_stalled_ee_targets(step_action)
            joint_limit_recoveries = self._recover_near_joint_limit_ee_targets(obs)
            tactile_profile_getter = self._resolve_env_callable(
                "get_tactile_cpu_control_step_profiles"
            )
            if callable(tactile_profile_getter):
                for tactile_profile in tactile_profile_getter():
                    perf_tactile_s += float(tactile_profile.get("total_s", 0.0))
                    perf_tactile_mapping_s += float(
                        tactile_profile.get("mapping_s", 0.0)
                    )
                    perf_tactile_points += int(
                        tactile_profile.get("tactile_point_count", 0)
                    )
                    perf_tactile_max_points = max(
                        perf_tactile_max_points,
                        int(
                            tactile_profile.get(
                                "max_tactile_points_per_sim_step",
                                0,
                            )
                        ),
                    )
            step_end_wall_s = control_tick_wall_s + (
                time.perf_counter() - step_started_s
            )
            if self.stop_requested:
                self.exit_reason = self._stop_reason or "stop_requested"
                return
            self._publish_observation(obs)
            self._latest_observation = obs
            step_info = dict(info)
            if recovered_ik_roles:
                step_info["teleop_ik_recovery"] = {
                    "roles": list(recovered_ik_roles),
                }
                print(
                    "[quest3-teleop] recovered stalled accumulated IK target "
                    f"for roles={','.join(recovered_ik_roles)}",
                    flush=True,
                )
            if joint_limit_recoveries:
                step_info["teleop_joint_limit_recovery"] = list(
                    joint_limit_recoveries
                )
                for recovery in joint_limit_recoveries:
                    print(
                        "[quest3-teleop] reanchored near joint limit "
                        f"role={recovery['role']} "
                        f"joint={recovery['joint_name']} "
                        f"side={recovery['limit_side']} "
                        f"distance={recovery['distance_rad']:.4f}rad "
                        f"velocity={recovery['velocity_rad_s']:.4f}rad/s",
                        flush=True,
                    )
            if joint_limit_guard_roles:
                step_info["teleop_joint_limit_guard"] = {
                    "roles": list(joint_limit_guard_roles),
                }
            if contact_control_diagnostics:
                step_info["teleop_contact_control"] = contact_control_diagnostics
            contact_diagnostics = self._contact_diagnostics(
                observation=obs,
                action=step_action,
                control_step=step_index,
                contact_control_diagnostics=contact_control_diagnostics,
            )
            if contact_diagnostics:
                step_info["teleop_contact_diagnostics"] = contact_diagnostics
            physics_warnings, physics_violations = self._physics_safety_status(obs)
            if physics_warnings:
                step_info["teleop_physics_warnings"] = physics_warnings
            if physics_violations:
                step_info["teleop_physics_violation"] = physics_violations
            if physics_violations:
                self.physics_violation = {
                    "control_step": int(step_index),
                    "violations": physics_violations,
                }
                self.exit_reason = "physics_violation"
                print(
                    "[quest3-teleop] physics safety violation: "
                    f"{self.physics_violation}",
                    flush=True,
                )
                self._print_violation_contact_diagnostics(control_step=step_index)
                return
            if self.on_step is not None:
                self.on_step(
                    Quest3ControlStep(
                        control_step=step_index,
                        observation=obs,
                        action=step_action,
                        held_action=not built_new_action,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=step_info,
                        quest_frame=frame,
                        timestamps={
                            "source_timestamp_s": (
                                None if frame is None else frame.source_timestamp_s
                            ),
                            "received_timestamp_s": (
                                None if frame is None else frame.received_timestamp_s
                            ),
                            "published_timestamp_s": (
                                None if frame is None else frame.published_timestamp_s
                            ),
                            "control_tick_s": control_tick_wall_s,
                            "step_end_s": step_end_wall_s,
                        },
                    ).cpu_copy()
                )
            perf_step_s += time.perf_counter() - step_started_s
            step_index += 1
            perf_steps += 1
            success = _any_truthy(info.get("success", False))
            if success:
                print(
                    f"[quest3-teleop] Task succeeded at control step {step_index}.",
                    flush=True,
                )
                if self.config.exit_on_success:
                    self.exit_reason = "success"
                    return
            if _any_truthy(terminated):
                print(
                    f"[quest3-teleop] Episode terminated at control step {step_index}.",
                    flush=True,
                )
                self.exit_reason = "environment_terminated"
                return
            if _any_truthy(truncated):
                if not self.config.no_time_limit:
                    print(
                        f"[quest3-teleop] Episode truncated at control step {step_index}.",
                        flush=True,
                    )
                    self.exit_reason = "time_limit"
                    return
                if not time_limit_reported:
                    print("[quest3-teleop] Time limit reached; continuing.", flush=True)
                    time_limit_reported = True
            render_started_s = time.perf_counter()
            self._render_if_due()
            render_elapsed_s = time.perf_counter() - render_started_s
            perf_render_s += render_elapsed_s
            perf_render_calls += 1
            perf_render_max_s = max(perf_render_max_s, render_elapsed_s)

            perf_now_s = time.perf_counter()
            perf_elapsed_s = perf_now_s - perf_started_s
            if perf_elapsed_s >= 2.0 and perf_steps:
                render_average_ms = (
                    1000.0 * perf_render_s / perf_render_calls
                    if perf_render_calls
                    else 0.0
                )
                print(
                    "[quest3-teleop] performance "
                    f"control_rate={perf_steps / perf_elapsed_s:.1f}Hz "
                    f"build={1000.0 * perf_build_s / perf_steps:.1f}ms "
                    f"step={1000.0 * perf_step_s / perf_steps:.1f}ms "
                    f"tactile={1000.0 * perf_tactile_s / perf_steps:.1f}ms "
                    f"tactile_map={1000.0 * perf_tactile_mapping_s / perf_steps:.1f}ms "
                    f"tactile_points={perf_tactile_points / perf_steps:.1f} "
                    f"tactile_points_max={perf_tactile_max_points} "
                    f"render={render_average_ms:.1f}ms "
                    f"render_max={1000.0 * perf_render_max_s:.1f}ms "
                    f"render_calls={perf_render_calls}",
                    flush=True,
                )
                for diagnostic_line in self._task_success_diagnostic_lines():
                    print(
                        f"[quest3-teleop] {diagnostic_line}",
                        flush=True,
                    )
                perf_started_s = perf_now_s
                perf_steps = 0
                perf_build_s = 0.0
                perf_step_s = 0.0
                perf_tactile_s = 0.0
                perf_tactile_mapping_s = 0.0
                perf_tactile_points = 0
                perf_tactile_max_points = 0
                perf_render_s = 0.0
                perf_render_calls = 0
                perf_render_max_s = 0.0

    def _startup_warmup(self, initial_obs: Any) -> Any:
        """Warm native simulation/sensors before the first tracked action."""

        if self.config.startup_warmup_steps <= 0:
            return initial_obs
        # Semantic backends cannot accept a generic zero native action. The
        # executable Quest path uses LegacyNativeBackend, so keep this optional
        # optimization scoped to that explicit compatibility path.
        if not isinstance(self.backend, LegacyNativeBackend):
            return initial_obs
        zero_action = _zero_action_from_space(getattr(self.env, "action_space", None))
        if zero_action is None:
            return initial_obs
        warmed_obs = initial_obs
        for _ in range(self.config.startup_warmup_steps):
            warmed_obs, _reward, _terminated, _truncated, _info = self.backend.step(
                zero_action
            )
        if (
            isinstance(warmed_obs, tuple)
            and len(warmed_obs) == 2
            and isinstance(warmed_obs[1], dict)
        ):
            warmed_obs = warmed_obs[0]
        print(
            f"[quest3-teleop] startup warmup steps={self.config.startup_warmup_steps}",
            flush=True,
        )
        return warmed_obs

    def request_stop(self, *, reason: str = "operator_abort") -> None:
        """Request an exit at the next recorder-safe control boundary."""

        self._stop_reason = str(reason)
        self.stop_requested = True

    def _resolve_manipulator_state_specs(self) -> dict[str, dict[str, Any]]:
        getter = getattr(self.env, "get_manipulator_state_spec", None)
        wrapper_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(wrapper_getter):
            try:
                getter = wrapper_getter("get_manipulator_state_spec")
            except AttributeError:
                pass
        if not callable(getter):
            return {}
        raw_specs = getter()
        if not isinstance(raw_specs, Mapping):
            raise TypeError("get_manipulator_state_spec() must return a mapping.")
        specs: dict[str, dict[str, Any]] = {}
        for raw_role, raw_spec in raw_specs.items():
            role = str(raw_role)
            if not isinstance(raw_spec, Mapping):
                raise TypeError(f"Manipulator state spec for {role!r} must be a mapping.")
            joint_names = tuple(str(name) for name in raw_spec.get("joint_names", ()))
            lower = self._cpu_vector(raw_spec.get("lower"), field_name=f"{role}.lower")
            upper = self._cpu_vector(raw_spec.get("upper"), field_name=f"{role}.upper")
            if not joint_names or lower.shape != (len(joint_names),) or upper.shape != lower.shape:
                raise ValueError(f"Manipulator state spec for {role!r} has invalid joint bounds.")
            specs[role] = {
                "joint_names": joint_names,
                "lower": lower,
                "upper": upper,
                "taxel_names": tuple(
                    str(name) for name in raw_spec.get("taxel_names", ())
                ),
                "taxel_finger_names": tuple(
                    None if name is None else str(name)
                    for name in raw_spec.get("taxel_finger_names", ())
                ),
                "taxel_region_names": tuple(
                    str(name)
                    for name in raw_spec.get("taxel_region_names", ())
                ),
            }
            taxel_metadata_lengths = {
                len(specs[role]["taxel_names"]),
                len(specs[role]["taxel_finger_names"]),
                len(specs[role]["taxel_region_names"]),
            }
            if len(taxel_metadata_lengths) != 1:
                raise ValueError(
                    f"Manipulator state spec for {role!r} has inconsistent "
                    "taxel metadata lengths."
                )
        return specs

    @staticmethod
    def _cpu_vector(value: Any, *, field_name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{field_name} must not be None.")
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        array = np.asarray(value, dtype=np.float64)
        while array.ndim > 1 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1:
            raise ValueError(f"{field_name} must be a vector, got shape {array.shape}.")
        return array

    @staticmethod
    def _observation_manipulators(
        observation: Any,
    ) -> Mapping[str, Any] | None:
        to_dict = getattr(observation, "to_dict", None)
        if callable(to_dict):
            observation = to_dict()
        if not isinstance(observation, Mapping):
            return None
        manipulators = observation.get("manipulators")
        return manipulators if isinstance(manipulators, Mapping) else None

    def _seed_eef_positions(self, observation: Any) -> None:
        manipulators = self._observation_manipulators(observation)
        if manipulators is None:
            return
        for role, manipulator in manipulators.items():
            if not isinstance(manipulator, Mapping):
                continue
            pose_value = manipulator.get("end_effector_pose")
            if pose_value is None:
                continue
            pose = self._cpu_vector(
                pose_value,
                field_name=f"{role}.end_effector_pose",
            )
            if pose.shape != (7,):
                raise ValueError(
                    f"{role}.end_effector_pose must have shape (7,), got "
                    f"{pose.shape}."
                )
            self._previous_eef_position_m[str(role)] = pose[:3].copy()

    def _arm_target_error_norms(self) -> dict[str, float]:
        getter = getattr(
            self.env,
            "get_manipulator_control_diagnostics",
            None,
        )
        wrapper_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(wrapper_getter):
            try:
                getter = wrapper_getter("get_manipulator_control_diagnostics")
            except AttributeError:
                pass
        if not callable(getter):
            return {}
        raw_diagnostics = getter()
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError(
                "get_manipulator_control_diagnostics() must return a mapping."
            )

        error_norms: dict[str, float] = {}
        for raw_role, raw_diagnostic in raw_diagnostics.items():
            role = str(raw_role)
            if not isinstance(raw_diagnostic, Mapping):
                raise TypeError(
                    f"Control diagnostics for {role!r} must be a mapping."
                )
            error_value = raw_diagnostic.get(
                "arm_ee_target_position_error_m"
            )
            if error_value is None:
                continue
            position_error = self._cpu_vector(
                error_value,
                field_name=f"{role}.arm_ee_target_position_error_m",
            )
            if position_error.shape != (3,):
                raise ValueError(
                    f"{role}.arm_ee_target_position_error_m must have shape "
                    f"(3,), got {position_error.shape}."
                )
            error_norms[role] = float(np.linalg.norm(position_error))
        return error_norms

    def _arm_target_position_diagnostics(
        self,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return controller target/current EE positions for contact logging."""

        getter = self._resolve_env_callable("get_manipulator_control_diagnostics")
        if not callable(getter):
            return {}
        raw_diagnostics = getter()
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError(
                "get_manipulator_control_diagnostics() must return a mapping."
            )

        result: dict[str, dict[str, np.ndarray]] = {}
        for raw_role, raw_diagnostic in raw_diagnostics.items():
            role = str(raw_role)
            if not isinstance(raw_diagnostic, Mapping):
                raise TypeError(
                    f"Control diagnostics for {role!r} must be a mapping."
                )
            role_values: dict[str, np.ndarray] = {}
            field_names = {
                "arm_ee_target_position_m": "target_position_m",
                "arm_ee_current_position_m": "current_position_m",
                "arm_ee_target_position_error_m": "position_error_m",
            }
            for field_name, output_name in field_names.items():
                value = raw_diagnostic.get(field_name)
                if value is None:
                    continue
                vector = self._cpu_vector(value, field_name=f"{role}.{field_name}")
                if vector.shape != (3,):
                    raise ValueError(
                        f"{role}.{field_name} must have shape (3,), got {vector.shape}."
                    )
                role_values[output_name] = vector
            result[role] = role_values
        return result

    def _resolve_env_callable(self, name: str) -> Callable[..., Any] | None:
        """Resolve a callable through the Gym wrapper chain."""

        current_env = self.env
        while current_env is not None:
            candidate = getattr(current_env, name, None)
            if callable(candidate):
                return candidate
            current_env = getattr(current_env, "env", None)
        wrapper_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(wrapper_getter):
            try:
                candidate = wrapper_getter(name)
            except AttributeError:
                return None
            if callable(candidate):
                return candidate
        return None

    def _task_success_diagnostic_lines(self) -> tuple[str, ...]:
        """Format an environment's latest task-success gates for the operator."""

        getter = self._resolve_env_callable("get_task_success_diagnostics")
        if not callable(getter):
            return ()
        raw_diagnostics = getter()
        if raw_diagnostics is None:
            return ()
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError(
                "get_task_success_diagnostics() must return a mapping or None."
            )
        if raw_diagnostics.get("schema") == "b05_packing_v1":
            return self._b05_success_diagnostic_lines(raw_diagnostics)
        if raw_diagnostics.get("schema") == "b06_drawing_v1":
            return self._b06_success_diagnostic_lines(raw_diagnostics)

        piece_names = tuple(
            str(name) for name in raw_diagnostics.get("piece_names", ())
        )
        if not piece_names:
            raise ValueError("Task success diagnostics must include piece_names.")
        piece_count = len(piece_names)

        def vector(name: str) -> np.ndarray:
            values = self._cpu_vector(raw_diagnostics.get(name), field_name=name)
            if values.shape != (piece_count,):
                raise ValueError(
                    f"{name} must have shape ({piece_count},), got {values.shape}."
                )
            return values

        def scalar(name: str) -> float:
            values = self._cpu_vector(raw_diagnostics.get(name), field_name=name)
            if values.shape != (1,):
                raise ValueError(f"{name} must contain one value, got {values.shape}.")
            return float(values[0])

        xy_error_m = vector("xy_error_m")
        yaw_error_rad = vector("yaw_error_rad")
        tilt_error_rad = vector("tilt_error_rad")
        height_error_m = vector("height_error_m")
        linear_speed_mps = vector("linear_speed_mps")
        angular_speed_rad_s = vector("angular_speed_rad_s")
        released = vector("released").astype(bool)
        static = vector("static").astype(bool)
        correct = vector("correct").astype(bool)
        adjacency = self._cpu_vector(
            raw_diagnostics.get("adjacency"),
            field_name="adjacency",
        ).astype(bool)
        thresholds = raw_diagnostics.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise TypeError("Task success diagnostics must include thresholds.")

        blockers: list[str] = []
        threshold_fields = (
            ("xy", xy_error_m, "xy_error_m"),
            ("yaw", yaw_error_rad, "yaw_error_rad"),
            ("tilt", tilt_error_rad, "tilt_error_rad"),
            ("height", height_error_m, "height_error_m"),
            ("linear_speed", linear_speed_mps, "linear_speed_mps"),
            ("angular_speed", angular_speed_rad_s, "angular_speed_rad_s"),
        )
        for label, values, threshold_name in threshold_fields:
            threshold = float(thresholds[threshold_name])
            blockers.extend(
                f"{piece_names[index]}:{label}"
                for index, value in enumerate(values)
                if value > threshold
            )
        blockers.extend(
            f"{piece_names[index]}:held"
            for index, is_released in enumerate(released)
            if not is_released
        )
        if not adjacency.all():
            blockers.append("adjacency")

        dwell_steps = int(round(scalar("dwell_steps")))
        dwell_required_steps = int(raw_diagnostics["dwell_required_steps"])
        complete_now = bool(scalar("complete_now"))
        footprint_iou = scalar("footprint_iou")
        footprint_iou_quality_target = float(
            raw_diagnostics["footprint_iou_quality_target"]
        )
        blocker_text = "none" if not blockers else ",".join(blockers)
        summary = (
            f"task_progress task={raw_diagnostics.get('task_id', 'unknown')} "
            f"correct={int(correct.sum())}/{piece_count} "
            f"adjacency={int(adjacency.sum())}/{len(adjacency)} "
            f"iou={footprint_iou:.3f} "
            f"iou_quality_target={footprint_iou_quality_target:.3f} "
            f"complete_now={int(complete_now)} "
            f"dwell={dwell_steps}/{dwell_required_steps} blockers={blocker_text}"
        )
        metrics = (
            "task_metrics "
            f"pieces={list(piece_names)} "
            f"xy_mm={np.round(1000.0 * xy_error_m, 1).tolist()} "
            f"yaw_deg={np.round(np.degrees(yaw_error_rad), 1).tolist()} "
            f"tilt_deg={np.round(np.degrees(tilt_error_rad), 1).tolist()} "
            f"height_mm={np.round(1000.0 * height_error_m, 1).tolist()} "
            f"linear_mps={np.round(linear_speed_mps, 3).tolist()} "
            f"angular_rad_s={np.round(angular_speed_rad_s, 3).tolist()} "
            f"released={released.astype(int).tolist()} "
            f"static={static.astype(int).tolist()}"
        )
        return summary, metrics

    def _b06_success_diagnostic_lines(
        self,
        diagnostics: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Format B06's live drawing gates without re-evaluating the task."""

        def scalar(name: str) -> float:
            values = self._cpu_vector(diagnostics.get(name), field_name=name)
            if values.shape != (1,):
                raise ValueError(f"{name} must contain one value, got {values.shape}.")
            return float(values[0])

        def flag(name: str) -> bool:
            return bool(round(scalar(name)))

        pen_held_now = flag("pen_held_now")
        pen_paper_contact = flag("pen_paper_contact")
        tip_in_contact_region = flag("pen_tip_in_contact_region")
        valid_draw_contact = flag("valid_draw_contact")
        mark_length_m = scalar("continuous_mark_length_m")
        minimum_length_m = scalar("minimum_mark_length_m")
        mark_steps = int(round(scalar("continuous_mark_steps")))
        minimum_steps = int(round(scalar("minimum_mark_steps")))
        success = flag("success")

        blockers: list[str] = []
        if not pen_held_now:
            blockers.append("pen:not_held")
        if not pen_paper_contact:
            blockers.append("pen:paper_no_contact")
        if not tip_in_contact_region:
            blockers.append("pen:tip_region")
        if mark_length_m < minimum_length_m:
            blockers.append("draw:length")
        if mark_steps < minimum_steps:
            blockers.append("draw:steps")
        blocker_text = "none" if not blockers else ",".join(blockers)
        progress = (
            f"task_progress task={diagnostics.get('task_id', 'b06_v1')} "
            f"held={int(pen_held_now)} paper_contact={int(pen_paper_contact)} "
            f"tip_region={int(tip_in_contact_region)} "
            f"draw_contact={int(valid_draw_contact)} "
            f"length_mm={1000.0 * mark_length_m:.1f}/"
            f"{1000.0 * minimum_length_m:.1f} "
            f"steps={mark_steps}/{minimum_steps} success={int(success)} "
            f"blockers={blocker_text}"
        )
        details = (
            "task_drawing "
            f"pen_paper_force_n={scalar('pen_paper_force_n'):.3f} "
            f"pen_hold_force_n={scalar('pen_hold_force_n'):.3f} "
            f"paper_stable_stage={int(flag('paper_stable_stage'))} "
            f"target_tip_contact_stage="
            f"{int(flag('target_tip_contact_stage'))}"
        )
        return progress, details

    def _b05_success_diagnostic_lines(
        self,
        diagnostics: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Format B05 gates plus non-gating geometry, motion, and force data."""

        object_names = tuple(
            str(name) for name in diagnostics.get("object_names", ())
        )
        if not object_names:
            raise ValueError("B05 success diagnostics must include object_names.")
        object_count = len(object_names)

        def vector(name: str, *, length: int) -> np.ndarray:
            values = self._cpu_vector(diagnostics.get(name), field_name=name)
            if values.shape != (length,):
                raise ValueError(
                    f"{name} must have shape ({length},), got {values.shape}."
                )
            return values

        def matrix(name: str, *, rows: int, columns: int) -> np.ndarray:
            value = diagnostics.get(name)
            if value is None:
                raise ValueError(f"{name} must not be None.")
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            values = np.asarray(value, dtype=np.float64)
            while values.ndim > 2 and values.shape[0] == 1:
                values = values[0]
            if values.shape != (rows, columns):
                raise ValueError(
                    f"{name} must have shape ({rows}, {columns}), got "
                    f"{values.shape}."
                )
            return values

        def scalar_bool(name: str) -> bool:
            return bool(round(float(vector(name, length=1)[0])))

        inside = vector("inside", length=object_count).astype(bool)
        boundary_margin_m = matrix(
            "boundary_margin_m",
            rows=object_count,
            columns=3,
        )
        local_centres_m = matrix(
            "local_centres_m",
            rows=object_count,
            columns=3,
        )
        linear_speed_mps = vector("linear_speed_mps", length=object_count)
        angular_speed_rad_s = vector("angular_speed_rad_s", length=object_count)
        object_box_force_n = vector("object_box_force_n", length=object_count)

        lid_names = tuple(str(name) for name in diagnostics.get("lid_names", ()))
        if not lid_names:
            raise ValueError("B05 success diagnostics must include lid_names.")
        lid_count = len(lid_names)
        lid_open_fraction = vector("lid_open_fraction", length=lid_count)
        lid_qpos_rad = vector("lid_qpos_rad", length=lid_count)
        lid_qvel_rad_s = vector("lid_qvel_rad_s", length=lid_count)

        pair_names = tuple(
            str(name) for name in diagnostics.get("object_pair_names", ())
        )
        object_pair_force_n = vector(
            "object_pair_force_n",
            length=len(pair_names),
        )
        lid_object_force_n = float(
            vector("lid_object_force_n", length=1)[0]
        )
        all_objects_inside = scalar_bool("all_objects_inside")
        lid_closed = scalar_bool("lid_closed")
        success = scalar_bool("success")
        left_participated = scalar_bool("left_hand_participated")
        right_participated = scalar_bool("right_hand_participated")

        thresholds = diagnostics.get("thresholds")
        if not isinstance(thresholds, Mapping):
            raise TypeError("B05 success diagnostics must include thresholds.")
        containment_tolerance_m = float(thresholds["containment_tolerance_m"])
        lid_closed_fraction_threshold = float(thresholds["lid_closed_fraction"])

        blockers = [
            f"{object_names[index]}:outside"
            for index, is_inside in enumerate(inside)
            if not is_inside
        ]
        if not lid_closed:
            blockers.append("lid:open")
        blocker_text = "none" if not blockers else ",".join(blockers)

        progress = (
            f"task_progress task={diagnostics.get('task_id', 'b05_v1')} "
            f"inside={int(inside.sum())}/{object_count} "
            f"all_inside={int(all_objects_inside)} "
            f"lid_closed={int(lid_closed)} success={int(success)} "
            f"blockers={blocker_text}"
        )
        geometry = (
            "task_geometry "
            f"objects={list(object_names)} inside={inside.astype(int).tolist()} "
            f"boundary_margin_xyz_mm="
            f"{np.round(1000.0 * boundary_margin_m, 1).tolist()} "
            f"centre_box_xyz_m={np.round(local_centres_m, 3).tolist()} "
            f"outside_tolerance_mm={1000.0 * containment_tolerance_m:.1f}"
        )
        motion = (
            "task_motion "
            f"linear_mps={np.round(linear_speed_mps, 3).tolist()} "
            f"angular_rad_s={np.round(angular_speed_rad_s, 3).tolist()} "
            f"hands_ever_participated="
            f"[{int(left_participated)}, {int(right_participated)}] "
            "(diagnostic_only)"
        )
        lid = (
            "task_lid "
            f"names={list(lid_names)} "
            f"open_fraction={np.round(lid_open_fraction, 4).tolist()} "
            f"closed_threshold={lid_closed_fraction_threshold:.4f} "
            f"qpos_rad={np.round(lid_qpos_rad, 4).tolist()} "
            f"qvel_rad_s={np.round(lid_qvel_rad_s, 4).tolist()}"
        )
        pair_force_by_name = dict(
            zip(
                pair_names,
                np.round(object_pair_force_n, 2).tolist(),
            )
        )
        forces = (
            "task_forces diagnostic_only "
            f"object_box_n={np.round(object_box_force_n, 2).tolist()} "
            f"object_pairs={pair_force_by_name} "
            f"lid_object_max_n={lid_object_force_n:.2f}"
        )
        return progress, geometry, motion, lid, forces

    def _tactile_mapping_diagnostics(self) -> dict[str, dict[str, float]]:
        getter = self._resolve_env_callable(
            "get_manipulator_tactile_diagnostics"
        )
        if not callable(getter):
            return {}
        raw_diagnostics = getter()
        if not isinstance(raw_diagnostics, Mapping):
            raise TypeError(
                "get_manipulator_tactile_diagnostics() must return a mapping."
            )
        diagnostics: dict[str, dict[str, float]] = {}
        for raw_role, raw_diagnostic in raw_diagnostics.items():
            role = str(raw_role)
            if not isinstance(raw_diagnostic, Mapping):
                raise TypeError(
                    f"Tactile diagnostics for {role!r} must be a mapping."
                )
            diagnostics[role] = {
                str(name): float(value)
                for name, value in raw_diagnostic.items()
                if isinstance(value, (int, float, np.integer, np.floating))
                and math.isfinite(float(value))
            }
        return diagnostics

    def _set_tactile_mapping_diagnostics(self, enabled: bool) -> None:
        setter = self._resolve_env_callable("set_tactile_diagnostics_enabled")
        if callable(setter):
            setter(bool(enabled))

    def _semantic_manipulator_actions(
        self,
        action: Any,
    ) -> Mapping[str, Any]:
        if isinstance(action, Quest3ActionPair):
            action = action.semantic
        if not isinstance(action, Mapping):
            return {}
        semantic_fields = {
            "end_effector_delta_pose",
            "hand_joint_position",
        }
        if set(action).issubset(semantic_fields):
            return {self.config.hand_side: action}
        if all(isinstance(value, Mapping) for value in action.values()):
            return action
        return {}

    @staticmethod
    def _taxel_identity(
        spec: Mapping[str, Any],
        *,
        taxel_index: int,
        taxel_count: int,
    ) -> tuple[str, str]:
        taxel_names = tuple(spec.get("taxel_names", ()))
        finger_names = tuple(spec.get("taxel_finger_names", ()))
        region_names = tuple(spec.get("taxel_region_names", ()))
        if not taxel_names:
            return f"taxel_{taxel_index}", "unknown"
        if not (
            len(taxel_names)
            == len(finger_names)
            == len(region_names)
            == taxel_count
        ):
            raise ValueError(
                "Taxel force count does not match manipulator taxel metadata: "
                f"force={taxel_count}, names={len(taxel_names)}, "
                f"fingers={len(finger_names)}, regions={len(region_names)}."
            )
        finger_name = finger_names[taxel_index]
        region_name = str(region_names[taxel_index])
        anatomical_region = (
            region_name
            if finger_name is None
            else f"{finger_name}/{region_name}"
        )
        return str(taxel_names[taxel_index]), anatomical_region

    @staticmethod
    def _format_tactile_mapping_diagnostics(
        diagnostic: Mapping[str, Any],
    ) -> str:
        required_fields = (
            "raw_total_force_n",
            "raw_covered_force_n",
            "mapped_force_n",
            "force_error",
            "raw_covered_torque_nm",
            "mapped_torque_nm",
            "torque_error_nm",
            "taxel_vector_force_n",
            "taxel_force_norm_sum_n",
            "coverage_ratio",
            "force_coverage_ratio",
            "covered_contact_count",
            "total_contact_count",
            "centroid_error_mean_m",
            "centroid_error_max_m",
        )
        if not all(field_name in diagnostic for field_name in required_fields):
            return ""
        contact_count_text = (
            f"{int(diagnostic['covered_contact_count'])}/"
            f"{int(diagnostic['total_contact_count'])}"
        )
        return (
            f" raw_total={diagnostic['raw_total_force_n']:.3f}N"
            f" raw={diagnostic['raw_covered_force_n']:.3f}N"
            f" mapped={diagnostic['mapped_force_n']:.3f}N"
            f" force_err={diagnostic['force_error']:.3e}"
            f" raw_tau={diagnostic['raw_covered_torque_nm']:.3f}Nm"
            f" mapped_tau={diagnostic['mapped_torque_nm']:.3f}Nm"
            f" tau_err={diagnostic['torque_error_nm']:.3e}Nm"
            f" taxel_vec={diagnostic['taxel_vector_force_n']:.3f}N"
            f" taxel_norm_sum={diagnostic['taxel_force_norm_sum_n']:.3f}N"
            f" coverage={diagnostic['coverage_ratio']:.3f}"
            f" force_coverage={diagnostic['force_coverage_ratio']:.3f}"
            f" contacts={contact_count_text}"
            f" centroid={diagnostic['centroid_error_mean_m']:.5f}m"
            f" centroid_max={diagnostic['centroid_error_max_m']:.5f}m"
        )

    def _contact_diagnostics(
        self,
        *,
        observation: Any,
        action: Any,
        control_step: int,
        contact_control_diagnostics: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.config.print_tactile:
            return []
        self._latest_contact_diagnostics.clear()
        manipulators = self._observation_manipulators(observation)
        if manipulators is None:
            return []

        semantic_actions = self._semantic_manipulator_actions(action)
        target_error_norms = self._arm_target_error_norms()
        target_position_diagnostics = self._arm_target_position_diagnostics()
        mapping_diagnostics = self._tactile_mapping_diagnostics()
        threshold_n = float(self.config.tactile_force_threshold_n)
        diagnostics: list[dict[str, Any]] = []
        for role, spec in self._manipulator_state_specs.items():
            manipulator = manipulators.get(role)
            if not isinstance(manipulator, Mapping):
                continue
            taxel_value = manipulator.get("taxel_normal_force")
            if taxel_value is None:
                continue
            taxel_force = self._cpu_vector(
                taxel_value,
                field_name=f"{role}.taxel_normal_force",
            )
            if not np.isfinite(taxel_force).all() or not taxel_force.size:
                continue

            taxel_index = int(np.argmax(taxel_force))
            peak_force_n = float(taxel_force[taxel_index])
            taxel_name, anatomical_region = self._taxel_identity(
                spec,
                taxel_index=taxel_index,
                taxel_count=taxel_force.size,
            )
            previous_streak = self._tactile_contact_streak_steps.get(role, 0)
            contact_active = peak_force_n > threshold_n
            contact_streak = previous_streak + 1 if contact_active else 0
            self._tactile_contact_streak_steps[role] = contact_streak

            eef_step_translation_norm_m: float | None = None
            pose_value = manipulator.get("end_effector_pose")
            if pose_value is not None:
                pose = self._cpu_vector(
                    pose_value,
                    field_name=f"{role}.end_effector_pose",
                )
                if pose.shape != (7,):
                    raise ValueError(
                        f"{role}.end_effector_pose must have shape (7,), got "
                        f"{pose.shape}."
                    )
                current_position_m = pose[:3]
                previous_position_m = self._previous_eef_position_m.get(role)
                if previous_position_m is not None:
                    eef_step_translation_norm_m = float(
                        np.linalg.norm(current_position_m - previous_position_m)
                    )
                self._previous_eef_position_m[role] = current_position_m.copy()

            commanded_translation_norm_m: float | None = None
            manipulator_action = semantic_actions.get(role)
            if isinstance(manipulator_action, Mapping):
                delta_value = manipulator_action.get(
                    "end_effector_delta_pose"
                )
                if delta_value is not None:
                    delta_pose = self._cpu_vector(
                        delta_value,
                        field_name=f"{role}.end_effector_delta_pose",
                    )
                    if delta_pose.shape != (6,):
                        raise ValueError(
                            f"{role}.end_effector_delta_pose must have shape "
                            f"(6,), got {delta_pose.shape}."
                        )
                    commanded_translation_norm_m = float(
                        np.linalg.norm(delta_pose[:3])
                    )

            role_control_diagnostics = (
                contact_control_diagnostics.get(role, {})
                if isinstance(contact_control_diagnostics, Mapping)
                else {}
            )
            raw_arm_translation = (
                self._cpu_vector(
                    role_control_diagnostics["raw_arm_translation"],
                    field_name=f"{role}.raw_arm_translation",
                ).tolist()
                if isinstance(role_control_diagnostics, Mapping)
                and role_control_diagnostics.get("raw_arm_translation") is not None
                else None
            )
            filtered_arm_translation = (
                self._cpu_vector(
                    role_control_diagnostics["filtered_arm_translation"],
                    field_name=f"{role}.filtered_arm_translation",
                ).tolist()
                if isinstance(role_control_diagnostics, Mapping)
                and role_control_diagnostics.get("filtered_arm_translation") is not None
                else None
            )
            position_diagnostics = target_position_diagnostics.get(role, {})
            target_position_m = position_diagnostics.get("target_position_m")
            current_position_m = position_diagnostics.get("current_position_m")
            target_error_vector_m = position_diagnostics.get("position_error_m")

            diagnostic = {
                "role": role,
                "contact_threshold_n": threshold_n,
                "contact_active": contact_active,
                "contact_streak_steps": contact_streak,
                "peak_force_n": peak_force_n,
                "taxel_index": taxel_index,
                "taxel_name": taxel_name,
                "anatomical_region": anatomical_region,
                "active_taxel_count": int(
                    np.count_nonzero(taxel_force > threshold_n)
                ),
                "taxel_scalar_sum_n": float(np.sum(taxel_force)),
                "commanded_translation_norm_m": commanded_translation_norm_m,
                "eef_step_translation_norm_m": eef_step_translation_norm_m,
                "arm_ee_target_error_norm_m": target_error_norms.get(role),
                "raw_arm_translation_m": raw_arm_translation,
                "filtered_arm_translation_m": filtered_arm_translation,
                "arm_ee_target_position_m": (
                    None if target_position_m is None else target_position_m.tolist()
                ),
                "arm_ee_current_position_m": (
                    None if current_position_m is None else current_position_m.tolist()
                ),
                "arm_ee_target_error_vector_m": (
                    None
                    if target_error_vector_m is None
                    else target_error_vector_m.tolist()
                ),
            }
            diagnostic.update(mapping_diagnostics.get(role, {}))
            self._latest_contact_diagnostics[role] = diagnostic.copy()
            if contact_active or previous_streak > 0:
                diagnostics.append(diagnostic)

            if contact_active and (
                control_step % self.config.tactile_print_every_steps == 0
            ):
                command_text = (
                    "n/a"
                    if commanded_translation_norm_m is None
                    else f"{commanded_translation_norm_m:.5f}m"
                )
                eef_step_text = (
                    "n/a"
                    if eef_step_translation_norm_m is None
                    else f"{eef_step_translation_norm_m:.5f}m"
                )
                target_error_norm_m = target_error_norms.get(role)
                target_error_text = (
                    "n/a"
                    if target_error_norm_m is None
                    else f"{target_error_norm_m:.5f}m"
                )
                raw_command_text = _format_vector_text(raw_arm_translation)
                filtered_command_text = _format_vector_text(filtered_arm_translation)
                target_position_text = _format_vector_text(
                    diagnostic["arm_ee_target_position_m"]
                )
                current_position_text = _format_vector_text(
                    diagnostic["arm_ee_current_position_m"]
                )
                target_error_vector_text = _format_vector_text(
                    diagnostic["arm_ee_target_error_vector_m"]
                )
                mapping_text = self._format_tactile_mapping_diagnostics(diagnostic)
                print(
                    "[quest3-contact] "
                    f"step={control_step} role={role} "
                    f"peak={peak_force_n:.3f}N taxel={taxel_name} "
                    f"region={anatomical_region} streak={contact_streak} "
                    f"active={diagnostic['active_taxel_count']} "
                    f"scalar_sum={diagnostic['taxel_scalar_sum_n']:.3f}N "
                    f"command={command_text} eef_step={eef_step_text} "
                    f"target_error={target_error_text} "
                    f"raw_cmd={raw_command_text} filtered_cmd={filtered_command_text} "
                    f"target_p={target_position_text} ee_p={current_position_text} "
                    f"error_vec={target_error_vector_text} {mapping_text}",
                    flush=True,
                )
            elif not contact_active and previous_streak > 0:
                print(
                    "[quest3-contact] "
                    f"step={control_step} role={role} contact_ended "
                    f"duration_steps={previous_streak}",
                    flush=True,
                )
        return diagnostics

    def _print_violation_contact_diagnostics(self, *, control_step: int) -> None:
        if not self._latest_contact_diagnostics:
            print(
                "[quest3-contact-violation] "
                f"step={control_step} unavailable=true "
                f"print_tactile={self.config.print_tactile}",
                flush=True,
            )
            return
        for role, diagnostic in self._latest_contact_diagnostics.items():
            payload = {
                "control_step": int(control_step),
                "role": role,
                **diagnostic,
            }
            print(
                "[quest3-contact-violation] "
                + json.dumps(payload, sort_keys=True, separators=(",", ":")),
                flush=True,
            )

    def _physics_safety_status(
        self,
        observation: Any,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not self._manipulator_state_specs:
            return [], []
        to_dict = getattr(observation, "to_dict", None)
        if callable(to_dict):
            observation = to_dict()
        if not isinstance(observation, Mapping):
            return [], [{"kind": "invalid_observation", "type": type(observation).__name__}]
        manipulators = observation.get("manipulators")
        if not isinstance(manipulators, Mapping):
            return [], [{"kind": "missing_manipulators"}]

        warnings: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        for role, spec in self._manipulator_state_specs.items():
            manipulator = manipulators.get(role)
            if not isinstance(manipulator, Mapping):
                violations.append({"kind": "missing_manipulator", "role": role})
                continue
            joint_names = spec["joint_names"]
            lower = spec["lower"]
            upper = spec["upper"]
            qpos_value = manipulator.get("joint_position")
            if qpos_value is None:
                violations.append(
                    {"kind": "missing_joint_position", "role": role}
                )
                continue
            try:
                qpos = self._cpu_vector(
                    qpos_value,
                    field_name=f"{role}.joint_position",
                )
            except (TypeError, ValueError) as exc:
                violations.append(
                    {
                        "kind": "invalid_joint_position",
                        "role": role,
                        "reason": str(exc),
                    }
                )
                continue
            if qpos.shape != lower.shape:
                violations.append(
                    {
                        "kind": "invalid_joint_position_shape",
                        "role": role,
                        "actual_shape": list(qpos.shape),
                        "expected_shape": list(lower.shape),
                    }
                )
                continue
            if not np.isfinite(qpos).all():
                violations.append({"kind": "non_finite_joint_position", "role": role})
            else:
                limit_excess = np.maximum(lower - qpos, qpos - upper)
                joint_index = int(np.argmax(limit_excess))
                excess_rad = float(limit_excess[joint_index])
                if excess_rad > self.config.max_joint_limit_excess_rad:
                    status = {
                        "role": role,
                        "joint_index": joint_index,
                        "joint_name": joint_names[joint_index],
                        "position_rad": float(qpos[joint_index]),
                        "lower_rad": float(lower[joint_index]),
                        "upper_rad": float(upper[joint_index]),
                        "excess_rad": excess_rad,
                    }
                    if excess_rad > self.config.recoverable_joint_limit_excess_rad:
                        violations.append({"kind": "joint_limit_excess", **status})
                    else:
                        warnings.append(
                            {"kind": "joint_limit_recovery_warning", **status}
                        )

            qvel_value = manipulator.get("joint_velocity")
            if qvel_value is None:
                violations.append(
                    {"kind": "missing_joint_velocity", "role": role}
                )
            else:
                qvel = self._cpu_vector(
                    qvel_value,
                    field_name=f"{role}.joint_velocity",
                )
                if qvel.shape != lower.shape:
                    violations.append(
                        {
                            "kind": "invalid_joint_velocity_shape",
                            "role": role,
                            "actual_shape": list(qvel.shape),
                            "expected_shape": list(lower.shape),
                        }
                    )
                elif not np.isfinite(qvel).all():
                    violations.append({"kind": "non_finite_joint_velocity", "role": role})
                else:
                    joint_index = int(np.argmax(np.abs(qvel)))
                    speed = float(abs(qvel[joint_index]))
                    velocity_status = {
                        "role": role,
                        "joint_index": joint_index,
                        "joint_name": joint_names[joint_index],
                        "velocity_rad_s": float(qvel[joint_index]),
                    }
                    if speed > self.config.max_joint_velocity_rad_s:
                        violations.append(
                            {"kind": "joint_velocity_limit", **velocity_status}
                        )
                    elif speed > self.config.warning_joint_velocity_rad_s:
                        warnings.append(
                            {"kind": "joint_velocity_warning", **velocity_status}
                        )

            taxel_value = manipulator.get("taxel_normal_force")
            if taxel_value is not None:
                taxel_force = self._cpu_vector(
                    taxel_value,
                    field_name=f"{role}.taxel_normal_force",
                )
                if not np.isfinite(taxel_force).all():
                    violations.append({"kind": "non_finite_taxel_force", "role": role})
                elif taxel_force.size:
                    taxel_index = int(np.argmax(taxel_force))
                    peak_force = float(taxel_force[taxel_index])
                    if peak_force > self.config.max_taxel_force_n:
                        taxel_name, anatomical_region = self._taxel_identity(
                            spec,
                            taxel_index=taxel_index,
                            taxel_count=taxel_force.size,
                        )
                        violations.append(
                            {
                                "kind": "taxel_force_limit",
                                "role": role,
                                "taxel_index": taxel_index,
                                "taxel_name": taxel_name,
                                "anatomical_region": anatomical_region,
                                "force_n": peak_force,
                            }
                        )
        contact_getter = self._resolve_env_callable(
            "get_manipulator_contact_control_state"
        )
        if callable(contact_getter):
            contact_states = contact_getter()
            if not isinstance(contact_states, Mapping):
                raise TypeError(
                    "get_manipulator_contact_control_state() must return a mapping."
                )
            for raw_role, raw_state in contact_states.items():
                role = str(raw_role)
                if not isinstance(raw_state, Mapping):
                    raise TypeError(f"Contact state for {role!r} must be a mapping.")
                parent_force = np.asarray(
                    raw_state.get("parent_force_world"), dtype=np.float64
                )
                while parent_force.ndim > 2 and parent_force.shape[0] == 1:
                    parent_force = parent_force[0]
                if parent_force.ndim != 2 or parent_force.shape[-1] != 3:
                    raise ValueError(
                        f"{role}.parent_force_world must have shape [N, 3], "
                        f"got {parent_force.shape}."
                    )
                parent_norms = np.linalg.norm(parent_force, axis=1)
                max_parent_force = float(parent_norms.max(initial=0.0))
                finger_force = np.asarray(
                    raw_state.get(
                        "finger_force_world", np.zeros((5, 3), dtype=np.float64)
                    ),
                    dtype=np.float64,
                )
                while finger_force.ndim > 2 and finger_force.shape[0] == 1:
                    finger_force = finger_force[0]
                if finger_force.ndim != 2 or finger_force.shape[-1] != 3:
                    raise ValueError(
                        f"{role}.finger_force_world must have shape [5, 3], "
                        f"got {finger_force.shape}."
                    )
                max_finger_force = float(
                    np.linalg.norm(finger_force, axis=1).max(initial=0.0)
                )
                taxel_force = np.asarray(
                    raw_state.get("taxel_force_world", parent_force),
                    dtype=np.float64,
                )
                while taxel_force.ndim > 2 and taxel_force.shape[0] == 1:
                    taxel_force = taxel_force[0]
                if taxel_force.ndim != 2 or taxel_force.shape[-1] != 3:
                    raise ValueError(
                        f"{role}.taxel_force_world must have shape [N, 3], "
                        f"got {taxel_force.shape}."
                    )
                max_taxel_force = float(
                    np.linalg.norm(taxel_force, axis=1).max(initial=0.0)
                )
                hand_force = self._cpu_vector(
                    raw_state.get("hand_force_world"),
                    field_name=f"{role}.hand_force_world",
                )
                hand_resultant = float(np.linalg.norm(hand_force))
                if max_parent_force > self.config.max_hand_force_n:
                    violations.append(
                        {
                            "kind": "parent_force_limit",
                            "role": role,
                            "force_n": max_parent_force,
                            "limit_n": self.config.max_hand_force_n,
                        }
                    )
                if max_finger_force > self.config.max_hand_force_n:
                    violations.append(
                        {
                            "kind": "finger_force_limit",
                            "role": role,
                            "force_n": max_finger_force,
                            "limit_n": self.config.max_hand_force_n,
                        }
                    )
                if max_taxel_force > self.config.max_hand_force_n:
                    violations.append(
                        {
                            "kind": "taxel_resultant_force_limit",
                            "role": role,
                            "force_n": max_taxel_force,
                            "limit_n": self.config.max_hand_force_n,
                        }
                    )
                if hand_resultant > self.config.max_hand_force_n:
                    violations.append(
                        {
                            "kind": "hand_resultant_force_limit",
                            "role": role,
                            "force_n": hand_resultant,
                            "limit_n": self.config.max_hand_force_n,
                        }
                    )
        return warnings, violations

    def _build_action_builder(self) -> None:
        if self.config.mode == "dual":
            self.dual_builder = Quest3DualPandaShadowActionBuilder.from_env(
                self.env,
                left_retargeting_config_path=self.config.left_retargeting_config_path,
                right_retargeting_config_path=self.config.right_retargeting_config_path,
                retargeting_urdf_root=self.config.retargeting_urdf_root,
                wrist_motion_config=self.config.wrist_motion,
            )
            self.dual_builder.reset()
            return

        self.single_builder = Quest3PandaShadowActionBuilder.from_env(
            self.env,
            side=self.config.hand_side,
            retargeting_config_path=self.config.retargeting_config_path,
            retargeting_urdf_root=self.config.retargeting_urdf_root,
            wrist_motion_config=self.config.wrist_motion,
        )
        self.single_builder.reset()

    def _seed_hand_targets_from_observation(self, observation: Any) -> None:
        """Anchor the first Quest hand command to the post-warmup robot state."""

        single_seeder = (
            None
            if self.single_builder is None
            else getattr(
                self.single_builder,
                "seed_hand_target_from_robot_state",
                None,
            )
        )
        dual_seeder = (
            None
            if self.dual_builder is None
            else getattr(
                self.dual_builder,
                "seed_hand_targets_from_robot_state",
                None,
            )
        )
        if not callable(single_seeder) and not callable(dual_seeder):
            return

        to_dict = getattr(observation, "to_dict", None)
        if callable(to_dict):
            observation = to_dict()
        if not isinstance(observation, Mapping):
            raise TypeError("Initial observation must be a mapping to seed hand targets.")
        manipulators = observation.get("manipulators")
        if not isinstance(manipulators, Mapping):
            raise ValueError("Initial observation is missing manipulator state.")

        roles = (
            ("left", "right")
            if callable(dual_seeder)
            else (self.config.hand_side,)
        )
        joint_names_by_role: dict[str, tuple[str, ...]] = {}
        joint_qpos_by_role: dict[str, np.ndarray] = {}
        for role in roles:
            state_spec = self._manipulator_state_specs.get(role)
            manipulator = manipulators.get(role)
            if state_spec is None or not isinstance(manipulator, Mapping):
                raise ValueError(
                    f"Initial observation is missing {role!r} robot state."
                )
            joint_names_by_role[role] = tuple(state_spec["joint_names"])
            joint_qpos_by_role[role] = self._cpu_vector(
                manipulator.get("joint_position"),
                field_name=f"{role}.joint_position",
            )

        if callable(dual_seeder):
            dual_seeder(
                joint_names_by_side=joint_names_by_role,
                joint_qpos_by_side=joint_qpos_by_role,
            )
            return

        assert callable(single_seeder)
        role = self.config.hand_side
        single_seeder(
            joint_names=joint_names_by_role[role],
            joint_qpos=joint_qpos_by_role[role],
        )

    def _publish_observation(self, obs: Any, *, force_image: bool = False) -> None:
        """Optionally normalize observations without coupling Runner to recording."""

        if self.observation_adapter is None or self.on_observation is None:
            return
        snapshot = self.observation_adapter.snapshot(
            obs,
            observation_id=self._observation_id,
            timestamp_s=time.time(),
            force_image=force_image,
        )
        self._observation_id += 1
        self.on_observation(snapshot)

    def _filter_step_action(self, action: Any) -> tuple[Any, dict[str, Any]]:
        """Filter a Quest action pair immediately before backend execution."""

        if not isinstance(action, Quest3ActionPair):
            return action, {}
        if self._latest_observation is None:
            raise RuntimeError("Contact filtering requires an initialized observation.")
        getter = self._resolve_env_callable("get_manipulator_contact_control_state")
        if not callable(getter):
            return action, {"enabled": False, "reason": "contact_state_unavailable"}
        observation = self._observation_manipulators(self._latest_observation)
        if observation is None:
            raise RuntimeError("Contact filtering requires manipulator observations.")
        contact_states = getter()
        if not isinstance(contact_states, Mapping):
            raise TypeError(
                "get_manipulator_contact_control_state() must return a mapping."
            )
        result = self._contact_filter.filter(
            action.semantic,
            observations=observation,
            contact_states=contact_states,
        )
        for role, state in contact_states.items():
            if role in result.diagnostics and isinstance(state, Mapping):
                result.diagnostics[role]["joint_names"] = list(
                    state.get("joint_names", ())
                )
        if self.config.mode == "dual":
            if self.dual_builder is None:
                raise RuntimeError("Dual contact filtering requires a dual builder.")
            filtered_action = self.dual_builder.rebuild_action_pair_from_semantic(
                result.semantic
            )
        else:
            if self.single_builder is None:
                raise RuntimeError("Single contact filtering requires a single builder.")
            filtered_action = self.single_builder.rebuild_action_pair_from_semantic(
                result.semantic
            )
        return filtered_action, result.diagnostics

    def _build_action(
        self,
        frame: Quest3InputFrame,
    ) -> dict[str, np.ndarray]:
        if self.config.mode != "single":
            raise RuntimeError("Dual mode must use independent action pairs.")
        hand = _hand_for_side(frame, self.config.hand_side)
        assert hand is not None
        assert self.single_builder is not None
        action = self.single_builder.build_action(hand)
        self._record_side(
            frame=frame,
            side=self.config.hand_side,
            hand=hand,
            builder=self.single_builder,
            action=action,
        )
        return action

    def _build_action_pair_or_action(self, frame: Quest3InputFrame) -> Any:
        if self.config.mode == "dual" and self.dual_builder is not None:
            side_pairs: dict[str, Quest3ActionPair] = {}
            self._tracked_dual_sides = set()
            for side, hand in (("left", frame.left), ("right", frame.right)):
                builder = self.dual_builder.side_builders[side]
                if hand is not None:
                    side_pair = builder.build_action_pair(hand)
                    self._last_dual_actions[side] = side_pair.cpu_copy()
                    self._tracked_dual_sides.add(side)
                    self._record_side(
                        frame=frame,
                        side=side,
                        hand=hand,
                        builder=builder,
                        action=side_pair.semantic[side],
                    )
                else:
                    previous = self._last_dual_actions.get(side)
                    if previous is None:
                        raise ValueError(
                            f"No initial action for untracked Quest {side} hand."
                        )
                    held = _hold_hand_targets_without_arm_motion(previous)
                    assert isinstance(held, Quest3ActionPair)
                    side_pair = held
                side_pairs[side] = side_pair
            return self._combine_dual_action_pairs(side_pairs)
        if self.config.mode == "single" and self.single_builder is not None:
            build_pair = getattr(self.single_builder, "build_action_pair", None)
            hand = _hand_for_side(frame, self.config.hand_side)
            if callable(build_pair) and hand is not None:
                pair = build_pair(hand)
                self._record_side(
                    frame=frame,
                    side=self.config.hand_side,
                    hand=hand,
                    builder=self.single_builder,
                    action=pair.semantic[self.config.hand_side],
                )
                return pair
        return self._build_action(frame)

    def _combine_dual_action_pairs(
        self,
        side_pairs: dict[str, Quest3ActionPair],
    ) -> Quest3ActionPair:
        assert self.dual_builder is not None
        if set(side_pairs) != {"left", "right"}:
            raise ValueError("Dual action pairs must contain left and right sides.")
        return Quest3ActionPair(
            semantic={
                side: side_pairs[side].semantic[side]
                for side in ("left", "right")
            },
            native={
                side: side_pairs[side].native
                for side in ("left", "right")
            },
        )

    def _build_dual_followup_action_pair(
        self,
        pair: Quest3ActionPair,
    ) -> Quest3ActionPair:
        """Advance only wrists still tracked in the latest dual-hand frame."""

        assert self.dual_builder is not None
        side_pairs: dict[str, Quest3ActionPair] = {}
        for side in ("left", "right"):
            side_pair = Quest3ActionPair(
                semantic={side: pair.semantic[side]},
                native=pair.native[side],  # type: ignore[index]
            )
            if side in self._tracked_dual_sides:
                side_pair = self.dual_builder.side_builders[
                    side
                ].build_rotation_followup_action_pair(side_pair)
            else:
                held = _hold_hand_targets_without_arm_motion(side_pair)
                assert isinstance(held, Quest3ActionPair)
                side_pair = held
            side_pairs[side] = side_pair
        return self._combine_dual_action_pairs(side_pairs)

    def _record_side(
        self,
        *,
        frame: Quest3InputFrame,
        side: str,
        hand: Quest3HandInput,
        builder: Quest3PandaShadowActionBuilder,
        action: np.ndarray | dict[str, np.ndarray],
    ) -> None:
        if self.recorder is None:
            return
        diagnostics = builder.latest_diagnostics
        if diagnostics is None:
            raise RuntimeError(
                f"Quest {side} action builder produced no recording diagnostics."
            )
        retarget_joint_names = tuple(
            str(name) for name in builder.retargeting.joint_names
        )
        self.recorder.record(
            frame=frame,
            side=side,
            hand=hand,
            diagnostics=diagnostics,
            retarget_joint_names=retarget_joint_names,
            action=action,
        )

    def _tracking_ready(
        self, frame: Quest3InputFrame, *, now_s: float | None = None
    ) -> bool:
        if now_s is None:
            now_s = time.time()
        if self.config.mode == "dual":
            tracked = []
            for side, hand in (("left", frame.left), ("right", frame.right)):
                tracked.append(
                    self._update_tracking_state(
                        side, is_tracked=hand is not None, now_s=now_s
                    )
                )
            return any(tracked)

        hand = _hand_for_side(frame, self.config.hand_side)
        return self._update_tracking_state(
            self.config.hand_side,
            is_tracked=hand is not None,
            now_s=now_s,
        )

    def _update_tracking_state(
        self, side: str, *, is_tracked: bool, now_s: float
    ) -> bool:
        if is_tracked:
            self._tracking_lost_since_s[side] = None
            self._tracking_loss_reset_done[side] = False
            return True

        lost_since_s = self._tracking_lost_since_s[side]
        if lost_since_s is None:
            lost_since_s = now_s
            self._tracking_lost_since_s[side] = lost_since_s

        timeout_s = self.config.tracking_loss_reset_timeout_s
        if (
            not self._tracking_loss_reset_done[side]
            and now_s - lost_since_s >= timeout_s
        ):
            self._reset_wrist_side(side)
            self._tracking_loss_reset_done[side] = True
            print(
                f"[quest3-teleop] {side} tracking lost for "
                f"{timeout_s:.1f}s; wrist calibration reset.",
                flush=True,
            )
        return False

    def _is_frame_fresh(
        self, frame: Quest3InputFrame, *, now_s: float | None = None
    ) -> bool:
        if now_s is None:
            now_s = time.time()
        self._last_wall_time_s = now_s
        return now_s <= (
            frame.received_timestamp_s + self.config.stale_frame_timeout_s
        )

    def _frame_control_allowed(
        self, frame: Quest3InputFrame, *, now_s: float | None = None
    ) -> bool:
        if now_s is None:
            now_s = time.time()
        fresh = self._is_frame_fresh(frame, now_s=now_s)
        tracking_ready = fresh and self._tracking_ready(frame, now_s=now_s)
        gesture_allowed = not self._gesture_recenter_requested(frame)
        deadman_active = (
            not self.config.require_squeeze or self._deadman_active(frame)
        )
        allowed = tracking_ready and gesture_allowed and deadman_active
        # Stale connections and explicit gesture gates reset immediately. Fresh
        # tracking loss is handled separately by the delayed tracking watchdog.
        if not fresh or (tracking_ready and (not gesture_allowed or not deadman_active)):
            self._reset_wrist_calibration()
        return allowed

    def _view_control_allowed(self) -> bool:
        if self.view_ready is None:
            return True
        ready = self._view_capture_error is None and bool(self.view_ready())
        if ready:
            if self._view_blocked:
                self._view_blocked = False
                print(
                    "[quest3-view] fresh operator view available; control enabled.",
                    flush=True,
                )
            return True
        if not self._view_blocked:
            self._view_blocked = True
            self._reset_wrist_calibration()
            print(
                "[quest3-view] operator view is unavailable or stale; "
                "holding control.",
                flush=True,
            )
        return False

    def _gesture_recenter_requested(self, frame: Quest3InputFrame) -> bool:
        if not self.config.recenter_on_tap:
            return False
        hands = self._active_hands(frame)
        return bool(hands) and all(
            hand.gesture is not None and hand.gesture.tap for hand in hands
        )

    def _deadman_active(self, frame: Quest3InputFrame) -> bool:
        if self.config.mode == "single":
            hand = _hand_for_side(frame, self.config.hand_side)
            return bool(
                hand is not None
                and hand.gesture is not None
                and hand.gesture.squeeze
            )

        return all(
            hand is not None
            and hand.gesture is not None
            and hand.gesture.squeeze
            for hand in (frame.left, frame.right)
        )

    def _active_hands(self, frame: Quest3InputFrame) -> list[Quest3HandInput]:
        if self.config.mode == "dual":
            return [hand for hand in (frame.left, frame.right) if hand is not None]
        hand = _hand_for_side(frame, self.config.hand_side)
        return [] if hand is None else [hand]

    def _reset_wrist_calibration(self) -> None:
        if self.dual_builder is not None:
            for side in ("left", "right"):
                self.dual_builder.reset_wrist(side)
        if self.single_builder is not None:
            self.single_builder.reset_wrist()
        self._reset_ee_target_to_current_pose()

    def _reset_wrist_side(self, side: str) -> None:
        if self.dual_builder is not None:
            self.dual_builder.reset_wrist(side)
        if self.single_builder is not None and side == self.config.hand_side:
            self.single_builder.reset_wrist()
        self._reset_ee_target_to_current_pose(side=side)

    def _arm_controller_for_role(self, role: str) -> Any | None:
        """Resolve the role-local accumulated-pose arm controller."""

        agent_for_role = self._resolve_env_callable("agent_for_role")
        if callable(agent_for_role):
            role_agent = agent_for_role(ManipulatorRole(role))
            role_controller = getattr(role_agent, "controller", None)
            component_controllers = getattr(
                role_controller,
                "controllers",
                None,
            )
            if isinstance(component_controllers, Mapping):
                return component_controllers.get("arm")
            return None

        # Compatibility path for non-benchmark environments that expose only
        # ManiSkill's physical-agent controller structure.
        base_env = getattr(self.env, "unwrapped", self.env)
        agent = getattr(base_env, "agent", None)
        controller = getattr(agent, "controller", None)
        if controller is None:
            return None
        if isinstance(controller, Mapping):
            if self.dual_builder is None:
                return None
            target_uid = self.dual_builder.agent_uids.get(role)
            if target_uid is None:
                return None
            agent_controller = controller.get(target_uid)
        else:
            agent_controller = controller
        component_controllers = getattr(
            agent_controller,
            "controllers",
            None,
        )
        if not isinstance(component_controllers, Mapping):
            return None
        return component_controllers.get("arm")

    def _recover_stalled_ee_targets(self, action: Any) -> tuple[str, ...]:
        """Re-anchor accumulated targets after ManiSkill CPU IK holds qpos."""

        semantic_actions = self._semantic_manipulator_actions(action)
        if not semantic_actions:
            return ()

        recovered_roles: list[str] = []
        for role, manipulator_action in semantic_actions.items():
            role_name = str(role)
            if not isinstance(manipulator_action, Mapping):
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue
            delta_pose_value = manipulator_action.get(
                "end_effector_delta_pose"
            )
            if delta_pose_value is None:
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue
            delta_pose = self._cpu_vector(
                delta_pose_value,
                field_name=f"{role}.end_effector_delta_pose",
            )
            if delta_pose.shape != (6,):
                raise ValueError(
                    f"{role}.end_effector_delta_pose must have shape (6,), "
                    f"got {delta_pose.shape}."
                )
            if (
                float(np.linalg.norm(delta_pose))
                <= _IK_RECOVERY_ACTION_EPSILON
            ):
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue

            arm_controller = self._arm_controller_for_role(role_name)
            if arm_controller is None:
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue
            controller_config = getattr(arm_controller, "config", None)
            if not bool(getattr(controller_config, "use_target", False)):
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue

            start_qpos_value = getattr(arm_controller, "_start_qpos", None)
            target_qpos_value = getattr(arm_controller, "_target_qpos", None)
            if start_qpos_value is None or target_qpos_value is None:
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue
            start_qpos = self._cpu_vector(
                start_qpos_value,
                field_name=f"{role}.arm_start_qpos",
            )
            target_qpos = self._cpu_vector(
                target_qpos_value,
                field_name=f"{role}.arm_target_qpos",
            )
            if start_qpos.shape != target_qpos.shape:
                raise ValueError(
                    f"{role} arm start/target qpos shapes differ: "
                    f"{start_qpos.shape} and {target_qpos.shape}."
                )
            if not np.allclose(
                target_qpos,
                start_qpos,
                atol=_IK_RECOVERY_QPOS_ATOL_RAD,
                rtol=0.0,
            ):
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue

            target_pose = getattr(arm_controller, "_target_pose", None)
            current_pose = getattr(arm_controller, "ee_pose_at_base", None)
            if target_pose is None or current_pose is None:
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue
            target_position_m = self._cpu_vector(
                target_pose.p,
                field_name=f"{role}.arm_target_position_m",
            )
            current_position_m = self._cpu_vector(
                current_pose.p,
                field_name=f"{role}.arm_current_position_m",
            )
            if (
                target_position_m.shape != (3,)
                or current_position_m.shape != (3,)
            ):
                raise ValueError(
                    f"{role} arm target/current positions must have shape (3,)."
                )
            position_error_m = float(
                np.linalg.norm(target_position_m - current_position_m)
            )
            target_quaternion = self._cpu_vector(
                target_pose.q,
                field_name=f"{role}.arm_target_quaternion",
            )
            current_quaternion = self._cpu_vector(
                current_pose.q,
                field_name=f"{role}.arm_current_quaternion",
            )
            if (
                target_quaternion.shape != (4,)
                or current_quaternion.shape != (4,)
            ):
                raise ValueError(
                    f"{role} arm target/current quaternions must have shape (4,)."
                )
            target_quaternion_norm = float(np.linalg.norm(target_quaternion))
            current_quaternion_norm = float(np.linalg.norm(current_quaternion))
            if (
                target_quaternion_norm <= _IK_RECOVERY_QUATERNION_NORM_EPSILON
                or current_quaternion_norm
                <= _IK_RECOVERY_QUATERNION_NORM_EPSILON
            ):
                raise ValueError(
                    f"{role} arm target/current quaternions must be non-zero."
                )
            # [4], [4] -> scalar shortest rotation error; q and -q are equal.
            normalized_target_quaternion = (
                target_quaternion / target_quaternion_norm
            )
            normalized_current_quaternion = (
                current_quaternion / current_quaternion_norm
            )
            quaternion_dot = float(
                np.clip(
                    abs(
                        np.dot(
                            normalized_target_quaternion,
                            normalized_current_quaternion,
                        )
                    ),
                    0.0,
                    1.0,
                )
            )
            rotation_error_rad = 2.0 * float(np.arccos(quaternion_dot))
            if (
                position_error_m <= _IK_RECOVERY_POSITION_ERROR_M
                and rotation_error_rad <= _IK_RECOVERY_ROTATION_ERROR_RAD
            ):
                self._ik_stall_steps_by_role.pop(role_name, None)
                continue

            stall_steps = self._ik_stall_steps_by_role.get(role_name, 0) + 1
            self._ik_stall_steps_by_role[role_name] = stall_steps
            if stall_steps < self.config.ik_stall_recovery_steps:
                continue
            self._ik_stall_steps_by_role.pop(role_name, None)
            target_rotation = _quaternion_wxyz_to_matrix(
                normalized_target_quaternion
            )
            current_rotation = _quaternion_wxyz_to_matrix(
                normalized_current_quaternion
            )
            self._recover_wrist_tracking_error(
                role_name,
                position_error_root_m=target_position_m - current_position_m,
                rotation_error_root=target_rotation @ current_rotation.T,
            )
            recovered_roles.append(role_name)

        return tuple(recovered_roles)

    def _recover_wrist_tracking_error(
        self,
        role: str,
        *,
        position_error_root_m: np.ndarray,
        rotation_error_root: np.ndarray,
    ) -> None:
        """Reset only the unreachable IK target and retain WebXR correspondence."""

        if self.config.mode == "dual" and self.dual_builder is not None:
            recover = getattr(
                self.dual_builder,
                "recover_wrist_tracking_error",
                None,
            )
            if callable(recover):
                recover(
                    role,
                    position_error_root_m=position_error_root_m,
                    rotation_error_root=rotation_error_root,
                )
            else:
                self.dual_builder.reset_wrist(role)
        elif self.single_builder is not None:
            recover = getattr(
                self.single_builder,
                "recover_wrist_tracking_error",
                None,
            )
            if callable(recover):
                recover(
                    position_error_root_m=position_error_root_m,
                    rotation_error_root=rotation_error_root,
                )
            else:
                self.single_builder.reset_wrist()
        self._reset_ee_target_to_current_pose(side=role)

    def _recover_near_joint_limit_ee_targets(
        self,
        observation: Any,
    ) -> tuple[dict[str, Any], ...]:
        """Re-anchor one role before accumulated wrist motion crosses an arm limit."""

        manipulators = self._observation_manipulators(observation)
        if manipulators is None:
            return ()

        recoveries: list[dict[str, Any]] = []
        margin_rad = self.config.joint_limit_reanchor_margin_rad
        release_margin_rad = margin_rad + 0.05
        for role, spec in self._manipulator_state_specs.items():
            manipulator = manipulators.get(role)
            if not isinstance(manipulator, Mapping):
                continue
            arm_controller = self._arm_controller_for_role(role)
            controller_config = getattr(arm_controller, "config", None)
            if arm_controller is None or not bool(
                getattr(controller_config, "use_target", False)
            ):
                continue
            arm_joint_names = tuple(
                str(name)
                for name in getattr(controller_config, "joint_names", ())
            )
            if not arm_joint_names:
                continue

            joint_names = tuple(spec["joint_names"])
            joint_index_by_name = {
                name: index for index, name in enumerate(joint_names)
            }
            if any(name not in joint_index_by_name for name in arm_joint_names):
                continue
            try:
                qpos = self._cpu_vector(
                    manipulator.get("joint_position"),
                    field_name=f"{role}.joint_position",
                )
                qvel = self._cpu_vector(
                    manipulator.get("joint_velocity"),
                    field_name=f"{role}.joint_velocity",
                )
            except (TypeError, ValueError):
                # The hard safety path reports malformed state with full context.
                continue
            lower = spec["lower"]
            upper = spec["upper"]
            if qpos.shape != lower.shape or qvel.shape != lower.shape:
                continue

            nearest: dict[str, Any] | None = None
            for joint_name in arm_joint_names:
                joint_index = joint_index_by_name[joint_name]
                position_rad = float(qpos[joint_index])
                velocity_rad_s = float(qvel[joint_index])
                lower_rad = float(lower[joint_index])
                upper_rad = float(upper[joint_index])
                if not all(
                    math.isfinite(value)
                    for value in (position_rad, velocity_rad_s, lower_rad, upper_rad)
                ):
                    continue
                lower_distance_rad = position_rad - lower_rad
                upper_distance_rad = upper_rad - position_rad
                if lower_distance_rad <= upper_distance_rad:
                    limit_side = "lower"
                    distance_rad = lower_distance_rad
                else:
                    limit_side = "upper"
                    distance_rad = upper_distance_rad
                if nearest is None or distance_rad < nearest["distance_rad"]:
                    nearest = {
                        "role": role,
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "limit_side": limit_side,
                        "position_rad": position_rad,
                        "distance_rad": distance_rad,
                        "velocity_rad_s": velocity_rad_s,
                    }

            if nearest is None:
                continue
            distance_rad = float(nearest["distance_rad"])
            was_near_limit = role in self._near_joint_limit_roles
            if distance_rad > release_margin_rad:
                self._near_joint_limit_roles.discard(role)
                self._joint_limit_hold_steps_by_role.pop(role, None)
                continue
            if distance_rad > margin_rad:
                continue
            self._near_joint_limit_roles.add(role)
            # Re-anchor once on entry, then latch until the arm clears the
            # hysteresis band. Re-resetting every frame while actuator inertia
            # still carries the joint toward its limit continually replaces
            # the WebXR neutral and makes palm tracking appear frozen.
            if was_near_limit:
                continue

            self._reset_wrist_side(role)
            self._joint_limit_hold_steps_by_role[role] = (
                self.config.joint_limit_hold_steps
            )
            recoveries.append(nearest)

        return tuple(recoveries)

    def _guard_near_joint_limits(self, action: Any) -> tuple[Any, tuple[str, ...]]:
        """Briefly hold only recovering arms while preserving hand targets."""

        guarded_roles = tuple(
            role
            for role in ("left", "right")
            if self._joint_limit_hold_steps_by_role.get(role, 0) > 0
        )
        if not guarded_roles:
            return action, ()

        for role in guarded_roles:
            remaining = self._joint_limit_hold_steps_by_role[role] - 1
            if remaining > 0:
                self._joint_limit_hold_steps_by_role[role] = remaining
            else:
                self._joint_limit_hold_steps_by_role.pop(role, None)

        if isinstance(action, Quest3ActionPair):
            semantic = {
                role: {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in fields.items()
                }
                for role, fields in action.semantic.items()
            }
            for role in guarded_roles:
                fields = semantic.get(role)
                if fields is not None and "end_effector_delta_pose" in fields:
                    fields["end_effector_delta_pose"][...] = 0.0
            if self.config.mode == "dual":
                if self.dual_builder is None:
                    raise RuntimeError("Dual joint-limit guarding requires a dual builder.")
                return (
                    self.dual_builder.rebuild_action_pair_from_semantic(semantic),
                    guarded_roles,
                )
            if self.single_builder is None:
                raise RuntimeError("Single joint-limit guarding requires a builder.")
            return (
                self.single_builder.rebuild_action_pair_from_semantic(semantic),
                guarded_roles,
            )

        # Legacy semantic/native callers do not expose a safe role-local native
        # rebuild. Holding all arm fields is conservative and retains every hand
        # target; the normal Quest path above remains independently role-local.
        return _hold_hand_targets_without_arm_motion(action), guarded_roles

    def _reset_ee_target_to_current_pose(self, *, side: str | None = None) -> None:
        if side is not None:
            roles = (side,)
        elif self.config.mode == "dual":
            roles = ("left", "right")
        else:
            roles = (self.config.hand_side,)

        for role in roles:
            arm_controller = self._arm_controller_for_role(role)
            if arm_controller is None:
                continue
            controller_config = getattr(arm_controller, "config", None)
            if not bool(getattr(controller_config, "use_target", False)):
                continue
            current_ee_pose_at_base = getattr(arm_controller, "ee_pose_at_base", None)
            if current_ee_pose_at_base is None:
                continue
            arm_controller._target_pose = current_ee_pose_at_base

    def _report_invalid_frame(self, error: ValueError) -> None:
        now_s = time.monotonic()
        if now_s - self._last_error_report_s >= 1.0:
            print(f"[quest3-teleop] Dropped invalid tracking frame: {error}", flush=True)
            self._last_error_report_s = now_s

    def _report_telemetry(
        self,
        frame: Quest3InputFrame,
        action: Any,
        *,
        control_timestamp_s: float | None = None,
    ) -> None:
        if control_timestamp_s is None:
            control_timestamp_s = time.time()
        self._record_latency_samples(frame, control_timestamp_s=control_timestamp_s)
        now_s = time.monotonic()
        if self._last_input_frame_s is not None:
            frame_period_s = now_s - self._last_input_frame_s
            if frame_period_s > 1e-6:
                instantaneous_rate_hz = 1.0 / frame_period_s
                self._input_rate_hz = (
                    instantaneous_rate_hz
                    if self._input_rate_hz is None
                    else 0.8 * self._input_rate_hz + 0.2 * instantaneous_rate_hz
                )
        self._last_input_frame_s = now_s
        if now_s - self._last_telemetry_s < 1.0:
            return
        self._last_telemetry_s = now_s
        frame_age_ms = max(
            0.0,
            1000.0
            * (control_timestamp_s - frame.received_timestamp_s),
        )
        input_rate = (
            "n/a"
            if self._input_rate_hz is None
            else f"{self._input_rate_hz:.1f}Hz"
        )

        hand_descriptions = []
        for side, hand in (("left", frame.left), ("right", frame.right)):
            if hand is None:
                hand_descriptions.append(f"{side}=untracked")
                continue
            wrist_position = hand.wrist_matrix[:3, 3]
            tip_positions = hand.joint_matrices[[4, 9, 14, 19, 24], :3, 3]
            tip_distances = np.linalg.norm(tip_positions - wrist_position, axis=1)
            hand_descriptions.append(
                f"{side}_wrist={np.round(wrist_position, 3).tolist()} "
                f"tips={np.round(tip_distances, 3).tolist()}"
            )
        print(
            f"[quest3-teleop] frame={frame.sequence} "
            f"input_rate={input_rate} frame_age={frame_age_ms:.1f}ms "
            f"source_ts_raw={_format_timestamp(frame.source_timestamp_s)} "
            f"received_ts={frame.received_timestamp_s:.6f} "
            f"published_ts={_format_timestamp(frame.published_timestamp_s)} "
            f"control_ts={control_timestamp_s:.6f}; "
            + "; ".join(hand_descriptions),
            flush=True,
        )
        print(
            "[quest3-teleop] latency "
            + " ".join(
                _format_latency_summary(name, self._latency_samples_ms[name])
                for name in (
                    "receive_dt",
                    "handler_processing",
                    "publish_to_control",
                )
            ),
            flush=True,
        )

        if action is None:
            print("[quest3-teleop] control_step=blocked: safety gate inactive", flush=True)
            return
        # Quest3ActionPair carries both canonical semantic and executable
        # native forms; telemetry is a numeric control diagnostic, so report
        # only the native form.
        if isinstance(action, Quest3ActionPair):
            action = action.native
        action_items = (
            action.items()
            if _is_multi_agent_action(action)
            else (("single", action),)
        )
        for uid, agent_action in action_items:
            if isinstance(agent_action, dict):
                # Compatibility with semantic actions supplied by older
                # adapters. New Quest builders emit flat native arrays.
                arm = np.asarray(
                    agent_action.get("end_effector_delta_pose", ()),
                    dtype=np.float32,
                ).reshape(-1)
                hand = np.asarray(
                    agent_action.get("hand_joint_position", ()),
                    dtype=np.float32,
                ).reshape(-1)
                action_array = np.concatenate((arm, hand))
            else:
                action_array = np.asarray(agent_action, dtype=np.float32).reshape(-1)
            previous_action = self._telemetry_actions.get(uid)
            hand_delta_norm = (
                0.0
                if previous_action is None
                else float(np.linalg.norm(action_array[6:] - previous_action[6:]))
            )
            hand_values = action_array[6:]
            hand_range = (
                [round(float(hand_values.min()), 3), round(float(hand_values.max()), 3)]
                if hand_values.size
                else []
            )
            print(
                f"[quest3-teleop] uid={uid} "
                f"arm={np.round(action_array[:6], 3).tolist()} "
                f"hand_range={hand_range} "
                f"hand_delta_norm={hand_delta_norm:.3f}",
                flush=True,
            )
            self._telemetry_actions[uid] = action_array.copy()

    def _record_latency_samples(
        self,
        frame: Quest3InputFrame,
        *,
        control_timestamp_s: float,
    ) -> None:
        received_timestamp_s = frame.received_timestamp_s
        published_timestamp_s = frame.published_timestamp_s

        self._append_latency_delta(
            "receive_dt",
            received_timestamp_s,
            self._last_latency_received_timestamp_s,
        )
        self._append_latency_delta(
            "handler_processing",
            published_timestamp_s,
            received_timestamp_s,
        )
        self._append_latency_delta(
            "publish_to_control",
            control_timestamp_s,
            published_timestamp_s,
        )
        self._last_latency_received_timestamp_s = received_timestamp_s

    def _append_latency_delta(
        self,
        name: str,
        current_timestamp_s: float | None,
        previous_timestamp_s: float | None,
        *,
        allow_negative: bool = False,
    ) -> None:
        if current_timestamp_s is None or previous_timestamp_s is None:
            return
        delta_ms = 1000.0 * (
            float(current_timestamp_s) - float(previous_timestamp_s)
        )
        if not np.isfinite(delta_ms) or (delta_ms < 0.0 and not allow_negative):
            return
        self._latency_samples_ms[name].append(delta_ms)

    def _render_if_due(self) -> Any:
        """Render on every ``render_every_n``-th control tick.

        Each control tick reaches exactly one call site, so a single counter
        decimates the viewer without skewing the loop's pacing.
        """

        render_due = self._render_tick_index % self.config.render_every_n == 0
        view_due = (
            self.on_view_frame is not None
            and self._render_tick_index % self.config.view_every_n == 0
        )
        self._render_tick_index += 1
        return self._render(publish_view=view_due) if render_due else None

    def _wait_for_startup_preparation(self) -> None:
        """Keep the viewer responsive during the post-reset operator pause."""

        delay_s = float(self.config.startup_preparation_delay_s)
        if delay_s <= 0.0:
            return
        started_s = time.perf_counter()
        deadline_s = started_s + delay_s
        print(
            "[quest3-teleop] preparation delay="
            f"{delay_s:.1f}s; control and recording are paused.",
            flush=True,
        )
        while not self.stop_requested:
            remaining_s = deadline_s - time.perf_counter()
            if remaining_s <= 0.0:
                break
            # Render/poll on the main thread so the window can receive focus
            # and operator commands while no Quest frame can drive the robot.
            self._render()
            time.sleep(min(0.01, remaining_s))

    def _render(self, *, publish_view: bool = True) -> Any:
        render_result = None
        if self.config.render:
            if hasattr(self.env, "render_human"):
                render_result = self.env.render_human()
            else:
                render_result = self.env.render()
            viewer = render_result
            if viewer is None:
                viewer = getattr(self.env, "_viewer", None)
            if viewer is not None:
                if not self._viewer_ready_notified:
                    self._viewer_ready_notified = True
                    if self.on_viewer_ready is not None:
                        self.on_viewer_ready(viewer)
                self._poll_viewer_commands(viewer)
        if not publish_view or self.on_view_frame is None:
            return render_result

        capture_name = (
            "capture_synced_human_viewer_rgb"
            if self.config.render
            else "capture_quest_view_rgb"
        )
        capture_view = getattr(self.env, capture_name, None)
        wrapper_getter = getattr(self.env, "get_wrapper_attr", None)
        if callable(wrapper_getter):
            try:
                capture_view = wrapper_getter(capture_name)
            except AttributeError:
                pass
        if not callable(capture_view):
            self._view_capture_error = AttributeError(
                f"The environment does not expose {capture_name}()."
            )
            return render_result
        try:
            self.on_view_frame(
                capture_view(),
                captured_timestamp_s=time.time(),
            )
            self._view_capture_error = None
        except Exception as exc:
            if (
                self._view_capture_error is None
                or type(self._view_capture_error) is not type(exc)
                or str(self._view_capture_error) != str(exc)
            ):
                print(f"[quest3-view] frame capture failed: {exc}", flush=True)
            self._view_capture_error = exc
        return render_result

    def _poll_viewer_commands(self, viewer: Any) -> None:
        """Dispatch one-shot keyboard commands from the focused viewer window."""

        if self.on_viewer_command is None:
            return
        window = getattr(viewer, "window", None)
        key_press = getattr(window, "key_press", None)
        if not callable(key_press):
            return
        for command in ("reset", "quit", "proceed"):
            try:
                pressed = bool(key_press(self._viewer_command_keys[command]))
            except Exception:
                # A viewer can disappear while an episode is being torn down.
                return
            if pressed:
                self.on_viewer_command(command)
                return


def _format_timestamp(timestamp_s: float | None) -> str:
    return "n/a" if timestamp_s is None else f"{float(timestamp_s):.6f}"


def _format_vector_text(value: Any) -> str:
    """Format a finite three-vector for compact control diagnostics."""

    if value is None:
        return "n/a"
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"Diagnostic vector must be finite with shape (3,), got {vector.shape}.")
    return "[" + ",".join(f"{float(component):.5f}" for component in vector) + "]"


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("quaternion must be a finite wxyz four-vector.")
    norm = float(np.linalg.norm(q))
    if norm <= _IK_RECOVERY_QUATERNION_NORM_EPSILON:
        raise ValueError("quaternion must be non-zero.")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _format_latency_summary(name: str, samples_ms: deque[float]) -> str:
    if not samples_ms:
        return f"{name}=n/a"
    values = np.asarray(samples_ms, dtype=np.float64)
    p50, p95, p99 = np.percentile(values, (50.0, 95.0, 99.0))
    return (
        f"{name}[n={values.size} p50={p50:.1f}ms p95={p95:.1f}ms "
        f"p99={p99:.1f}ms max={values.max():.1f}ms]"
    )



def _hold_hand_targets_without_arm_motion(
    action: Any,
) -> Any:
    if isinstance(action, Quest3ActionPair):
        held_semantic = _hold_hand_targets_without_arm_motion(action.semantic)
        held_native = _hold_hand_targets_without_arm_motion(action.native)
        return Quest3ActionPair(semantic=held_semantic, native=held_native)
    if _is_multi_agent_action(action):
        held_actions: dict[str, Any] = {}
        for uid, agent_action in action.items():
            if isinstance(agent_action, dict):
                held_action = {
                    key: np.asarray(value, dtype=np.float32).copy()
                    for key, value in agent_action.items()
                }
                if "end_effector_delta_pose" in held_action:
                    held_action["end_effector_delta_pose"][...] = 0.0
                held_actions[uid] = held_action
                continue
            held_action = np.asarray(agent_action, dtype=np.float32).copy()
            held_action[:6] = 0.0
            held_actions[uid] = held_action
        return held_actions
    if isinstance(action, dict):
        if len(action) == 1 and isinstance(next(iter(action.values())), dict):
            uid, semantic_action = next(iter(action.items()))
            return {
                uid: _hold_hand_targets_without_arm_motion(semantic_action)
            }
        held_action = {
            key: np.asarray(value, dtype=np.float32).copy()
            for key, value in action.items()
        }
        if "end_effector_delta_pose" in held_action:
            held_action["end_effector_delta_pose"][...] = 0.0
        return held_action
    held_action = np.asarray(action, dtype=np.float32).copy()
    held_action[:6] = 0.0
    return held_action


def _zero_action_from_space(space: Any) -> Any | None:
    """Construct a zero native action for Box/Dict action spaces."""

    subspaces = getattr(space, "spaces", None)
    if isinstance(subspaces, dict):
        return {
            str(key): _zero_action_from_space(subspace)
            for key, subspace in subspaces.items()
        }
    shape = getattr(space, "shape", None)
    if shape is None:
        return None
    return np.zeros(tuple(int(dim) for dim in shape), dtype=np.float32)


def _is_multi_agent_action(action: Any) -> bool:
    """Distinguish a MultiAgent mapping from one semantic action mapping."""

    if not isinstance(action, dict):
        return False
    return not {
        "end_effector_delta_pose",
        "hand_joint_position",
    }.intersection(action)


def _hand_for_side(
    frame: Quest3InputFrame,
    side: str,
) -> Quest3HandInput | None:
    return frame.left if side == "left" else frame.right


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
