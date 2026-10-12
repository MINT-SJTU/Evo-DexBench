from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
from typing import Any

# Keep dex-retargeting's Torch kernels from oversubscribing CPU PhysX.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch
from dex_benchmark.sim.tasks import make_task_env
from dex_benchmark.sim.tasks.attributes import LayoutId
from dex_benchmark.teleoperation.quest3 import (
    DEFAULT_MAX_CONTACT_FORCE_N,
    BenchmarkSemanticBackend,
    Quest3CanonicalSession,
    Quest3MultiprocessVuerServer,
    Quest3ObservationAdapter,
    Quest3TeleopConfig,
    Quest3TeleopRunner,
    Quest3VuerConfig,
    Quest3WristMotionConfig,
    LegacyNativeBackend,
)
from dex_benchmark.teleoperation.quest3.terminal_input import TerminalQuitMonitor
from dex_benchmark.teleoperation.quest3.viewer_window import (
    configure_human_viewer_window,
    task_viewer_window_size,
)


DEFAULT_CONTROL_MODE = "arm_pd_ee_delta_pose_hand_pd_joint_pos"
# Matches ManiSkill's viewer default. Shrinking this does not speed up the
# control loop, and neither does skipping presents: at 30 presents per second
# on a 60Hz display the swapchain never fills, so render_human() costs about
# 1ms at 1920x1080 (measured, RTX 4090 + X11). The control tick is dominated by
# env.step() instead; see the Quest 3 teleoperation README for that breakdown.
DEFAULT_VIEWER_RESOLUTION = (1920, 1080)
DEFAULT_QUEST_VIEW_RESOLUTION = (640, 360)
VUER_WEB_CLIENT_URL = "https://vuer.ai"


_STOP_REQUESTED = False
_ACTIVE_RUNNER: Quest3TeleopRunner | None = None


def configure_sapien_clipboard_fallback() -> None:
    """Keep the SAPIEN viewer usable when Linux has no clipboard backend.

    SAPIEN's articulation panel imports ``copy_to_clipboard`` directly and can
    invoke it while rendering.  Pyperclip raises in headless/minimal desktops
    without xclip, xsel, or wl-clipboard; copying joint positions is optional,
    whereas an exception there must not end a Quest session.
    """

    try:
        import pyperclip
        from sapien.utils.viewer import articulation_window
    except ModuleNotFoundError:
        return

    original_copy = articulation_window.copy_to_clipboard
    if getattr(original_copy, "_evodex_safe_clipboard", False):
        return
    warned = False

    def safe_copy_to_clipboard(content: str) -> None:
        nonlocal warned
        try:
            original_copy(content)
        except pyperclip.PyperclipException:
            if not warned:
                print(
                    "[quest3-teleop] No system clipboard backend; "
                    "viewer joint-position copy is disabled.",
                    flush=True,
                )
                warned = True

    safe_copy_to_clipboard._evodex_safe_clipboard = True  # type: ignore[attr-defined]
    articulation_window.copy_to_clipboard = safe_copy_to_clipboard


def _request_operator_abort() -> None:
    """Abort the active collection without finalizing it as a dataset."""

    global _STOP_REQUESTED
    if _STOP_REQUESTED:
        return
    _STOP_REQUESTED = True
    print(
        "[quest3-teleop] operator abort requested; current episode will be discarded.",
        flush=True,
    )
    if _ACTIVE_RUNNER is not None:
        _ACTIVE_RUNNER.request_stop(reason="operator_abort")


def _request_graceful_stop(signum: int, _frame: Any) -> None:
    """Set a stop flag so the control loop exits at a safe boundary."""

    _request_operator_abort()


def parse_viewer_resolution(value: str) -> tuple[int, int]:
    """Parse a ``WIDTHxHEIGHT`` viewer window resolution."""

    dimensions = value.lower().split("x")
    if len(dimensions) != 2:
        raise argparse.ArgumentTypeError(
            f"--viewer-resolution must look like 640x360, got {value!r}."
        )
    try:
        width, height = (int(dimension) for dimension in dimensions)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--viewer-resolution must contain two integers, got {value!r}."
        ) from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(
            f"--viewer-resolution must be positive, got {width}x{height}."
        )
    return width, height


def quest_browser_url(*, host: str, port: int, websocket_protocol: str) -> str:
    """Return the hosted Vuer WebXR client URL for one advertised server."""

    if websocket_protocol not in {"ws", "wss"}:
        raise ValueError(
            f"websocket_protocol must be 'ws' or 'wss', got {websocket_protocol!r}."
        )
    if not host.strip():
        raise ValueError("Quest advertised host must not be empty.")
    if not 1 <= port <= 65535:
        raise ValueError(f"Quest port must be in [1, 65535], got {port}.")
    return f"{VUER_WEB_CLIENT_URL}?ws={websocket_protocol}://{host}:{port}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate benchmark robots from Quest 3 hand tracking."
    )
    parser.add_argument("--mode", choices=("single", "dual"), default="single")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--task-id",
        default=None,
        help="DexBench task id (single mode defaults to m01_v1; dual mode to b01_v1).",
    )
    parser.add_argument(
        "--layout-id",
        choices=tuple(layout_id.value for layout_id in LayoutId),
        default=None,
        help="Scene layout id (default: use the task's first supported layout).",
    )
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--robot-uid", default=None)
    parser.add_argument("--control-mode", default=DEFAULT_CONTROL_MODE)
    parser.add_argument(
        "--action-backend",
        choices=("native", "semantic"),
        default="native",
        help="Execute Quest actions directly or encode benchmark semantic actions.",
    )
    parser.add_argument(
        "--obs-mode",
        default=None,
        help="Override the task observation mode (default: use the task YAML).",
    )
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--image-hz", type=float, default=30.0)
    parser.add_argument(
        "--canonical-queue-capacity",
        type=int,
        default=256,
        help="Bounded canonical recorder queue depth.",
    )
    parser.add_argument(
        "--canonical-chunk-length",
        type=int,
        default=256,
        help="Number of transitions per canonical HDF5 chunk.",
    )
    parser.add_argument("--sim-freq", type=int, default=120)
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--render-mode", choices=("auto", "human", "none"), default="auto")
    parser.add_argument("--render-backend", default="auto")
    parser.add_argument(
        "--shader",
        default="default",
        help="Shader used by RGBD sensors recorded in observations.",
    )
    parser.add_argument(
        "--viewer-shader",
        choices=("default",),
        default="default",
        help="Shader used only by the local viewer (must provide Segmentation).",
    )
    parser.add_argument(
        "--viewer-resolution",
        type=parse_viewer_resolution,
        default=DEFAULT_VIEWER_RESOLUTION,
        help=(
            "Local viewer window resolution as WIDTHxHEIGHT. This does not "
            "affect the control rate; see --render-every-n for that."
        ),
    )
    parser.add_argument(
        "--render-every-n",
        type=int,
        default=1,
        help=(
            "Render the local viewer once every N control ticks. Keep 1 for "
            "teleoperation: the present costs about 1ms at 30Hz (measured), so "
            "skipping ticks buys almost no control time while it halves the "
            "viewer refresh rate and adds up to one control period of visual "
            "latency for the operator watching the viewer."
        ),
    )
    parser.add_argument(
        "--quest-view",
        action="store_true",
        help=(
            "Show the current freely moved SAPIEN viewer camera in Quest. "
            "Without this flag the Quest page contains hand tracking only."
        ),
    )
    parser.add_argument(
        "--quest-view-only",
        action="store_true",
        help=(
            "Stream the fixed host render camera to Quest without creating or "
            "refreshing a desktop viewer. Requires --quest-view."
        ),
    )
    parser.add_argument(
        "--quest-view-fps",
        type=int,
        default=30,
        help="Quest operator-view capture and publish rate. Default: 30.",
    )
    parser.add_argument(
        "--quest-view-resolution",
        type=parse_viewer_resolution,
        default=DEFAULT_QUEST_VIEW_RESOLUTION,
        help="Quest operator-view RGB resolution as WIDTHxHEIGHT. Default: 640x360.",
    )
    parser.add_argument(
        "--quest-view-quality",
        type=int,
        default=80,
        help="Vuer JPEG quality in [1, 100]. Default: 80.",
    )
    parser.add_argument(
        "--quest-view-display-scale",
        type=float,
        default=0.6,
        help=(
            "Quest image-plane display scale; does not change capture resolution. "
            "Default: 0.6."
        ),
    )
    parser.add_argument(
        "--quest-view-stale-timeout",
        type=float,
        default=0.25,
        help=(
            "Hold robot control when Vuer has not consumed a new operator-view "
            "frame for this many seconds. Default: 0.25."
        ),
    )

    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--advertise-host",
        default=None,
        help="Stable LAN address printed for Quest Browser (does not change bind host).",
    )
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument(
        "--webxr-fps",
        type=int,
        default=30,
        help=(
            "Rate at which the Quest client streams hand frames (Vuer allows up "
            "to 60). Raising it above --control-hz does not speed up control, "
            "but it makes the frame each control tick consumes fresher: at 30 "
            "the newest frame is on average half a WebXR period (about 17ms) "
            "old, at 60 about 8ms. Costs proportionally more network traffic."
        ),
    )
    parser.add_argument("--cert", type=Path, default=None)
    parser.add_argument("--key", type=Path, default=None)
    parser.add_argument("--stale-timeout", type=float, default=5.0)
    parser.add_argument(
        "--tracking-reset-timeout",
        type=float,
        default=5.0,
        help=(
            "Reset wrist calibration after this many seconds of continuous "
            "tracking loss; use 0 for legacy immediate reset."
        ),
    )

    parser.add_argument(
        "--wrist-motion-mode",
        choices=("frame_delta", "legacy_target"),
        default="frame_delta",
        help="Use per-frame wrist deltas or the legacy bounded target follower.",
    )
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--max-position-delta", type=float, default=0.03)
    parser.add_argument("--max-rotation-delta", type=float, default=0.08)
    parser.add_argument("--max-target-distance", type=float, default=0.60)
    parser.add_argument(
        "--hand-target-smoothing-alpha",
        type=float,
        default=1.0,
        help=(
            "Smoothing factor for absolute hand joint targets after retargeting "
            "(default: 1.0, no extra smoothing)."
        ),
    )
    parser.add_argument(
        "--max-hand-joint-delta",
        type=float,
        default=0.1,
        help=(
            "Maximum per-frame retargeted hand joint change before smoothing "
            "(default: 0.1 rad)."
        ),
    )
    parser.add_argument(
        "--print-tactile",
        action="store_true",
        help=(
            "Print per-control-step contact diagnostics including peak Taxel, "
            "anatomical region, contact streak, EE motion, and arm target error."
        ),
    )
    parser.add_argument(
        "--tactile-force-threshold",
        type=float,
        default=1.0,
        help="Contact streak and printing threshold in N (default: 1.0).",
    )
    parser.add_argument(
        "--tactile-print-every-steps",
        type=int,
        default=1,
        help="Print active contact once every N control steps (default: 1).",
    )
    parser.add_argument(
        "--contact-light-threshold",
        type=float,
        default=5.0,
        help="Parent-link resultant force entering light contact, in N.",
    )
    parser.add_argument(
        "--contact-medium-threshold",
        type=float,
        default=20.0,
        help="Parent-link resultant force entering medium contact, in N.",
    )
    parser.add_argument(
        "--max-hand-force",
        type=float,
        default=DEFAULT_MAX_CONTACT_FORCE_N,
        help="Hard resultant-force limit for taxel, finger, parent, and hand, in N.",
    )
    parser.add_argument(
        "--light-arm-normal-scale",
        type=float,
        default=0.25,
        help="Fraction of inward arm motion retained during light contact.",
    )
    parser.add_argument(
        "--light-finger-closure-scale",
        type=float,
        default=0.25,
        help="Fraction of contacted-finger closure retained during light contact.",
    )
    parser.add_argument(
        "--medium-finger-closure-scale",
        type=float,
        default=0.10,
        help="Fraction of contacted-finger closure retained during medium contact.",
    )
    parser.add_argument("--disable-wrist-rotation", action="store_true")
    deadman_group = parser.add_mutually_exclusive_group()
    deadman_group.add_argument(
        "--require-squeeze",
        dest="require_squeeze",
        action="store_true",
        help="Require the controlled hand to hold the Quest squeeze gesture.",
    )
    deadman_group.add_argument(
        "--disable-squeeze-deadman",
        dest="require_squeeze",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(require_squeeze=False)
    parser.add_argument("--recenter-on-tap", action="store_true")

    parser.add_argument("--retargeting-config", type=Path, default=None)
    parser.add_argument("--left-retargeting-config", type=Path, default=None)
    parser.add_argument("--right-retargeting-config", type=Path, default=None)
    parser.add_argument("--retargeting-urdf-root", type=Path, default=None)
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help="Write raw tracking and retargeting samples to a new JSONL file.",
    )
    parser.add_argument(
        "--canonical-output",
        type=Path,
        default=None,
        help=(
            "Canonical HDF5 file or output directory; requires Quest3 action pairs. "
            "Manual stop finalizes a partial episode."
        ),
    )
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--no-exit-on-success", action="store_true")
    parser.add_argument("--no-time-limit", action="store_true")
    return parser.parse_args(argv)


def resolve_render_settings(args: argparse.Namespace) -> tuple[str | None, str, bool]:
    quest_view_only = bool(getattr(args, "quest_view_only", False))
    if quest_view_only:
        if not args.quest_view:
            raise ValueError("--quest-view-only requires --quest-view.")
        if args.render_mode == "human":
            raise ValueError(
                "--quest-view-only cannot be combined with --render-mode human; "
                "use --render-mode none or auto."
            )
        render_enabled = False
    elif args.render_mode == "human":
        render_enabled = True
    elif args.render_mode == "none":
        render_enabled = False
    else:
        render_enabled = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if not render_enabled:
            print(
                "[quest3-teleop] No display server; running without a viewer "
                "while keeping sensor rendering enabled."
            )

    render_mode = "human" if render_enabled else None
    if args.render_backend != "auto":
        render_backend = args.render_backend
    else:
        # A task can require RGB/depth observations even when no human viewer is
        # available. The render backend therefore cannot depend on DISPLAY.
        render_backend = "gpu"
    return render_mode, render_backend, render_enabled


def validate_control_timing(*, control_hz: float, sim_freq: int) -> None:
    """Validate the fixed-rate Quest control baseline before creating an env."""

    if control_hz <= 0.0:
        raise ValueError(f"control_hz must be positive, got {control_hz}.")
    if sim_freq <= 0:
        raise ValueError(f"sim_freq must be positive, got {sim_freq}.")
    substeps = sim_freq / float(control_hz)
    if substeps < 1.0 or not abs(substeps - round(substeps)) <= 1e-6:
        raise ValueError(
            "sim_freq must be an integer multiple of control_hz; "
            f"got sim_freq={sim_freq}, control_hz={control_hz}."
        )


def validate_sensor_timing(*, control_hz: float, image_hz: float) -> None:
    if image_hz <= 0.0:
        raise ValueError(f"image_hz must be positive, got {image_hz}.")
    ratio = control_hz / image_hz
    if ratio < 1.0 or not abs(ratio - round(ratio)) <= 1e-6:
        raise ValueError(
            "control_hz must be an integer multiple of image_hz; "
            f"got control_hz={control_hz}, image_hz={image_hz}."
        )


def validate_quest_view_settings(
    args: argparse.Namespace,
    *,
    render_enabled: bool,
) -> int:
    """Validate optional Quest-view timing and return its control-tick cadence."""

    if not args.quest_view:
        return 1
    if not render_enabled and not getattr(args, "quest_view_only", False):
        raise ValueError(
            "--quest-view requires a local human viewer; use --render-mode human."
        )
    if not 1 <= args.quest_view_fps <= 60:
        raise ValueError("--quest-view-fps must be in [1, 60].")
    ratio = args.control_hz / float(args.quest_view_fps)
    if ratio < 1.0 or not abs(ratio - round(ratio)) <= 1e-6:
        raise ValueError(
            "control_hz must be an integer multiple of quest_view_fps; "
            f"got control_hz={args.control_hz}, quest_view_fps={args.quest_view_fps}."
        )
    view_every_n = int(round(ratio))
    if render_enabled and view_every_n % args.render_every_n != 0:
        raise ValueError(
            "The Quest view cadence must align with local viewer presents: "
            "control_hz / quest_view_fps must be an integer multiple of "
            "render_every_n."
        )
    if not 1 <= args.quest_view_quality <= 100:
        raise ValueError("--quest-view-quality must be in [1, 100].")
    display_scale = float(getattr(args, "quest_view_display_scale", 0.6))
    if not math.isfinite(display_scale) or display_scale <= 0.0:
        raise ValueError("--quest-view-display-scale must be finite and positive.")
    if args.quest_view_stale_timeout <= 0.0:
        raise ValueError("--quest-view-stale-timeout must be positive.")
    return view_every_n


def resolve_canonical_target(
    target: Path,
    *,
    episode_id: str | None,
) -> tuple[Path, str]:
    """Resolve a canonical file or directory target without ambiguity."""

    target = target.expanduser()
    if target.suffix.lower() in {".h5", ".hdf5"}:
        return target.parent, episode_id or target.stem
    return target, episode_id or "episode"


def resolve_task_id(args: argparse.Namespace) -> str:
    """Resolve the benchmark task used by Quest teleoperation."""

    explicit_task_id = getattr(args, "task_id", None)
    if explicit_task_id:
        return str(explicit_task_id)
    if args.mode == "dual":
        return "b01_v1"
    return "m01_v1"


def configure_task_human_viewer_window(task_id: str) -> str | None:
    """Apply an optional task-specific Human Viewer outer window size."""

    window_size = task_viewer_window_size(task_id)
    if window_size is None:
        return None
    window_id = configure_human_viewer_window(window_size)
    if window_id is None:
        return None
    print(
        "[quest3-teleop] Human Viewer window configured "
        f"size={window_size[0]}x{window_size[1]}",
        flush=True,
    )
    return window_id


def make_environment(
    args: argparse.Namespace,
    *,
    render_mode: str | None,
    render_backend: str,
    quest_view_capture_every_n: int = 1,
    layout_id: str | None = None,
) -> Any:
    validate_control_timing(control_hz=args.control_hz, sim_freq=args.sim_freq)
    validate_sensor_timing(
        control_hz=args.control_hz,
        image_hz=getattr(args, "image_hz", 30.0),
    )
    viewer_shader = getattr(args, "viewer_shader", args.shader)
    viewer_width, viewer_height = getattr(
        args,
        "viewer_resolution",
        DEFAULT_VIEWER_RESOLUTION,
    )
    human_render_camera_configs: dict[str, Any] = {
        "shader_pack": viewer_shader,
    }
    if getattr(args, "quest_view", False):
        quest_view_width, quest_view_height = args.quest_view_resolution
        human_render_camera_configs.update(
            width=quest_view_width,
            height=quest_view_height,
        )
    env_kwargs: dict[str, Any] = {
        "control_mode": args.control_mode,
        "reward_mode": "none" if args.mode == "dual" else args.reward_mode,
        "render_mode": render_mode,
        "render_backend": render_backend,
        "enable_shadow": render_mode is not None,
        "sim_backend": getattr(args, "sim_backend", "cpu"),
        "sim_config": {
            "sim_freq": args.sim_freq,
            "control_freq": int(args.control_hz),
        },
        "sensor_configs": {"shader_pack": args.shader},
        "human_render_camera_configs": human_render_camera_configs,
        "quest_view_capture_enabled": bool(
            getattr(args, "quest_view_only", False)
        ),
        "quest_view_capture_every_n": quest_view_capture_every_n,
        "viewer_camera_configs": {
            "shader_pack": viewer_shader,
            "width": viewer_width,
            "height": viewer_height,
        },
    }
    if layout_id is not None:
        env_kwargs["layout_id"] = str(layout_id)
    generalization_profile = getattr(args, "generalization_profile", None)
    generalization_axis = getattr(args, "generalization_axis", None)
    generalization_seed = getattr(args, "generalization_seed", None)
    generalization_attempt = getattr(args, "generalization_attempt", None)
    if generalization_profile not in (None, "none"):
        env_kwargs["generalization_profile"] = generalization_profile
    if generalization_axis not in (None, "balanced", "none"):
        env_kwargs["generalization_axis"] = generalization_axis
        env_kwargs["generalization_seed"] = generalization_seed
        if generalization_attempt is not None:
            env_kwargs["generalization_attempt"] = generalization_attempt
    if args.obs_mode is not None:
        env_kwargs["obs_mode"] = args.obs_mode
    robot_uid = getattr(args, "robot_uid", None)
    if robot_uid is not None:
        if args.mode == "dual":
            raise ValueError(
                "--robot-uid is a single-robot override and cannot replace the "
                "fixed left/right embodiment used by dual mode."
            )
        env_kwargs["robot_uids"] = robot_uid
    explicit_env_id = getattr(args, "env_id", None)
    if explicit_env_id is not None:
        import gymnasium as gym

        return gym.make(explicit_env_id, **env_kwargs)
    return make_task_env(resolve_task_id(args), **env_kwargs)


def resolve_hand_tracking(args: argparse.Namespace) -> tuple[bool, bool]:
    if args.mode == "dual":
        return True, True
    return args.hand_side == "left", args.hand_side == "right"


def validate_collection_exit_policy(args: argparse.Namespace) -> None:
    """Require canonical episodes to finalize immediately on task success."""

    if args.canonical_output is not None and args.no_exit_on_success:
        raise ValueError(
            "--no-exit-on-success is incompatible with canonical collection; "
            "successful episodes must finalize immediately."
        )


def main(argv: list[str] | None = None) -> None:
    signal.signal(signal.SIGINT, _request_graceful_stop)
    signal.signal(signal.SIGTERM, _request_graceful_stop)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args = parse_args(argv)
    validate_collection_exit_policy(args)
    render_mode, render_backend, render_enabled = resolve_render_settings(args)
    if render_enabled:
        configure_sapien_clipboard_fallback()
    view_every_n = validate_quest_view_settings(
        args,
        render_enabled=render_enabled,
    )
    enable_left, enable_right = resolve_hand_tracking(args)

    # Keep WebXR ingestion in a separate process so simulation/render work in
    # this process cannot starve the Vuer event loop.
    server = Quest3MultiprocessVuerServer(
        config=Quest3VuerConfig(
            host=args.host,
            port=args.port,
            fps=args.webxr_fps,
            enable_left=enable_left,
            enable_right=enable_right,
            cert_path=args.cert,
            key_path=args.key,
            view_enabled=args.quest_view,
            view_fps=args.quest_view_fps,
            view_quality=args.quest_view_quality,
            view_display_scale=args.quest_view_display_scale,
        ),
        view_resolution=args.quest_view_resolution,
    )
    stream = server.stream
    server.start()
    protocol = "wss" if args.cert is not None else "ws"
    print(
        f"[quest3-teleop] Vuer listening on {protocol}://{args.host}:{args.port}",
        flush=True,
    )
    if args.advertise_host:
        print(
            f"[quest3-teleop] Quest Browser: "
            f"{quest_browser_url(host=args.advertise_host, port=args.port, websocket_protocol=protocol)}",
            flush=True,
        )
    if args.quest_view:
        view_camera_description = (
            "fixed host camera; desktop viewer disabled"
            if args.quest_view_only
            else "desktop viewer camera synchronized"
        )
        print(
            "[quest3-view] enabled "
            f"resolution={args.quest_view_resolution[0]}x{args.quest_view_resolution[1]} "
            f"fps={args.quest_view_fps} quality={args.quest_view_quality} "
            f"display_scale={args.quest_view_display_scale}; "
            f"{view_camera_description}; control waits for a fresh Vuer image stream.",
            flush=True,
        )

    env = None
    canonical_session = None
    terminal_quit_monitor = None
    try:
        env = make_environment(
            args,
            render_mode=render_mode,
            render_backend=render_backend,
            quest_view_capture_every_n=view_every_n,
            layout_id=args.layout_id,
        )
        # Single-hand mode retains the verified native path.  Bimanual mode
        # always crosses the public role-based semantic boundary so callers do
        # not construct or record ManiSkill MultiAgent dictionaries.
        if args.mode == "dual":
            action_backend = BenchmarkSemanticBackend(env)
        elif args.hand_side == "left":
            # The formal left-hand route uses the fixed dual-arm PlacePen
            # embodiment and fills the inactive right role semantically.
            action_backend = BenchmarkSemanticBackend(
                env,
                default_role=args.hand_side,
            )
        else:
            action_backend = (
                LegacyNativeBackend(env)
                if args.action_backend == "native"
                else BenchmarkSemanticBackend(env, default_role=args.hand_side)
            )
        teleop_config = Quest3TeleopConfig(
            mode=args.mode,
            hand_side=args.hand_side,
            control_hz=args.control_hz,
            stale_frame_timeout_s=args.stale_timeout,
            tracking_loss_reset_timeout_s=args.tracking_reset_timeout,
            render=render_enabled,
            render_every_n=args.render_every_n,
            max_taxel_force_n=args.max_hand_force,
            print_tactile=args.print_tactile,
            tactile_force_threshold_n=args.tactile_force_threshold,
            tactile_print_every_steps=args.tactile_print_every_steps,
            contact_light_threshold_n=args.contact_light_threshold,
            contact_medium_threshold_n=args.contact_medium_threshold,
            max_hand_force_n=args.max_hand_force,
            light_arm_normal_scale=args.light_arm_normal_scale,
            light_finger_closure_scale=args.light_finger_closure_scale,
            medium_finger_closure_scale=args.medium_finger_closure_scale,
            view_every_n=view_every_n,
            exit_on_success=not args.no_exit_on_success,
            no_time_limit=args.no_time_limit,
            require_squeeze=args.require_squeeze,
            recenter_on_tap=args.recenter_on_tap,
            retargeting_config_path=args.retargeting_config,
            left_retargeting_config_path=args.left_retargeting_config,
            right_retargeting_config_path=args.right_retargeting_config,
            retargeting_urdf_root=args.retargeting_urdf_root,
            record_path=args.record_path,
            wrist_motion=Quest3WristMotionConfig(
                motion_mode=args.wrist_motion_mode,
                position_scale=args.position_scale,
                rotation_scale=args.rotation_scale,
                max_position_delta_m=args.max_position_delta,
                max_rotation_delta_rad=args.max_rotation_delta,
                max_target_distance_m=args.max_target_distance,
                enable_rotation=not args.disable_wrist_rotation,
                hand_target_smoothing_alpha=args.hand_target_smoothing_alpha,
                max_hand_joint_delta=args.max_hand_joint_delta,
            ),
        )
        if args.canonical_output is not None:
            canonical_dir, episode_id = resolve_canonical_target(
                args.canonical_output,
                episode_id=args.episode_id,
            )
            if args.mode == "dual":
                required_cameras = (
                    "context",
                    "interaction",
                    "wrist_left",
                    "wrist_right",
                )
                required_manipulators = ("left", "right")
            else:
                required_cameras = (
                    "context",
                    "interaction",
                    f"wrist_{args.hand_side}",
                )
                required_manipulators = (args.hand_side,)
            get_episode_metadata = env.get_wrapper_attr("get_episode_metadata")
            episode_metadata = get_episode_metadata()
            recording_metadata = {
                "mode": args.mode,
                "task_id": resolve_task_id(args),
                "env_id": args.env_id or "default",
                "requested_robot_uid": args.robot_uid,
                "physics_safety": {
                    "ik_stall_recovery_steps": teleop_config.ik_stall_recovery_steps,
                    "max_joint_limit_excess_rad": (
                        teleop_config.max_joint_limit_excess_rad
                    ),
                    "recoverable_joint_limit_excess_rad": (
                        teleop_config.recoverable_joint_limit_excess_rad
                    ),
                    "joint_limit_hold_steps": teleop_config.joint_limit_hold_steps,
                    "max_joint_velocity_rad_s": (
                        teleop_config.max_joint_velocity_rad_s
                    ),
                    "warning_joint_velocity_rad_s": (
                        teleop_config.warning_joint_velocity_rad_s
                    ),
                    "max_taxel_force_n": teleop_config.max_taxel_force_n,
                    "contact_light_threshold_n": teleop_config.contact_light_threshold_n,
                    "contact_medium_threshold_n": teleop_config.contact_medium_threshold_n,
                    "max_hand_force_n": teleop_config.max_hand_force_n,
                    "light_arm_normal_scale": teleop_config.light_arm_normal_scale,
                    "light_finger_closure_scale": teleop_config.light_finger_closure_scale,
                    "medium_finger_closure_scale": teleop_config.medium_finger_closure_scale,
                },
                "contact_action_filter": {
                    "light_threshold_n": teleop_config.contact_light_threshold_n,
                    "medium_threshold_n": teleop_config.contact_medium_threshold_n,
                    "max_hand_force_n": teleop_config.max_hand_force_n,
                    "light_arm_normal_scale": teleop_config.light_arm_normal_scale,
                    "light_finger_closure_scale": teleop_config.light_finger_closure_scale,
                    "medium_finger_closure_scale": teleop_config.medium_finger_closure_scale,
                },
            }
            for key in (
                "embodiment_id",
                "robot_uids",
                "manipulator_roles",
                "control_mode",
                "tactile_layouts",
                "joint_state_specs",
            ):
                recording_metadata[key] = episode_metadata[key]
            canonical_session = Quest3CanonicalSession(
                canonical_dir,
                episode_id,
                control_hz=args.control_hz,
                image_hz=args.image_hz,
                required_cameras=required_cameras,
                required_manipulators=required_manipulators,
                metadata=recording_metadata,
                queue_capacity=args.canonical_queue_capacity,
                chunk_length=args.canonical_chunk_length,
            )
        view_stream = server.view_stream

        def _on_viewer_ready(_viewer: Any) -> None:
            configure_task_human_viewer_window(resolve_task_id(args))

        runner = Quest3TeleopRunner(
            env,
            stream,
            backend=action_backend,
            on_recording_start=(
                None if canonical_session is None else canonical_session.start_recording
            ),
            on_reset_metadata=(
                None
                if canonical_session is None
                else lambda reset_info: canonical_session.update_metadata(
                    {
                        "seed": args.seed,
                        "layout_id": reset_info.get(
                            "layout_id",
                            getattr(env, "current_layout_id_value", ""),
                        ),
                        "reset_info": dict(reset_info),
                    }
                )
            ),
            on_step=None if canonical_session is None else canonical_session.on_step,
            on_view_frame=(
                None if view_stream is None else view_stream.publish
            ),
            view_ready=(
                None
                if view_stream is None
                else lambda: view_stream.is_consumed_fresh(
                    max_age_s=args.quest_view_stale_timeout
                )
            ),
            on_viewer_ready=_on_viewer_ready if render_enabled else None,
            config=teleop_config,
        )
        global _ACTIVE_RUNNER
        _ACTIVE_RUNNER = runner
        terminal_quit_monitor = TerminalQuitMonitor(_request_operator_abort)
        if terminal_quit_monitor.start():
            print(
                "[quest3-teleop] Press q to discard the current episode and stop.",
                flush=True,
            )
        runner.run(seed=args.seed)
        if canonical_session is not None:
            if runner.exit_reason != "success":
                reason = runner.exit_reason or "runner_stopped_without_success"
                canonical_session.abort(reason)
                print(
                    "[quest3-teleop] canonical recorder discarded episode: "
                    f"reason={reason} spool={canonical_session.recorder.spool.root}",
                    flush=True,
                )
            else:
                output = canonical_session.close(status="completed")
                if output is None:
                    raise RuntimeError(
                        "Task succeeded without any canonical transitions."
                    )
                report = canonical_session.validate_output()
                print(
                    "[quest3-teleop] canonical recorder: "
                    f"output={output} report={report} "
                    f"metrics={canonical_session.metrics}",
                    flush=True,
                )
    except KeyboardInterrupt:
        if canonical_session is not None:
            canonical_session.abort("operator_abort")
            print(
                "[quest3-teleop] canonical recorder discarded interrupted episode: "
                f"spool={canonical_session.recorder.spool.root}",
                flush=True,
            )
        raise SystemExit(0)
    except BaseException as exc:
        if canonical_session is not None:
            canonical_session.abort(repr(exc))
        raise
    finally:
        _ACTIVE_RUNNER = None
        if terminal_quit_monitor is not None:
            terminal_quit_monitor.close()
        try:
            if env is not None:
                env.close()
        finally:
            server.stop()


if __name__ == "__main__":
    main()
