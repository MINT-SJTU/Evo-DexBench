"""Formal multi-episode Quest 3 canonical data collector.

The collector owns session/episode lifecycle only.  Quest retargeting,
semantic action construction, contact filtering, and physics safety remain in
``Quest3TeleopRunner`` and its existing adapters.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import threading
import time
from typing import Any, Mapping

from dex_benchmark.sim.generalization import GeneralizationProtocol
from dex_benchmark.sim.tasks import TaskRegistry
from dex_benchmark.data.canonical.validate import validate_canonical_hdf5
from dex_benchmark.commands.teleoperation.quest3 import (
    DEFAULT_QUEST_VIEW_RESOLUTION,
    DEFAULT_VIEWER_RESOLUTION,
    make_environment,
    validate_control_timing,
    validate_sensor_timing,
)
from dex_benchmark.teleoperation.quest3 import (
    DEFAULT_MAX_CONTACT_FORCE_N,
    BenchmarkSemanticBackend,
    Quest3CanonicalSession,
    Quest3MultiprocessVuerServer,
    Quest3TeleopConfig,
    Quest3TeleopRunner,
    Quest3VuerConfig,
    Quest3WristMotionConfig,
)
from dex_benchmark.teleoperation.quest3.terminal_input import TerminalQuitMonitor
from dex_benchmark.teleoperation.quest3.viewer_window import (
    DEFAULT_VIEWER_WINDOW_SIZE,
    configure_human_viewer_window,
    task_viewer_window_size,
)


MANIFEST_VERSION = 1
RESUME_CONFIG_VERSION = 1
FORMAL_RENDER_PROFILE = {
    "quest_view": False,
    "quest_view_only": False,
    "render_mode": "human",
    "render_backend": "gpu",
    "sim_backend": "cpu",
    "quest_view_fps": 30,
    "quest_view_resolution": "640x360",
    "quest_view_quality": 80,
    "quest_view_display_scale": 0.6,
}
_ACTIVE_COMMANDS: "_OperatorCommands | None" = None
_RETRYABLE_ATTEMPT_STATUSES: frozenset[str] = frozenset(
    {"operator_reset", "physics_violation", "time_limit"}
)
_TERMINAL_ATTEMPT_STATUSES: frozenset[str] = frozenset(
    {"operator_quit", "interrupted"}
)


def _attempt_status_from_exit_reason(exit_reason: str) -> str:
    """Map an expected runner stop to a collector attempt status."""

    passthrough_statuses = (
        _RETRYABLE_ATTEMPT_STATUSES | _TERMINAL_ATTEMPT_STATUSES
    )
    if exit_reason in passthrough_statuses:
        return exit_reason
    return f"error:{exit_reason}"


def _is_retryable_attempt_status(attempt_status: str) -> bool:
    """Return whether collection should reset and retry the logical episode."""

    return attempt_status in _RETRYABLE_ATTEMPT_STATUSES


def _print_physics_violation_reset_notice(
    attempt: Mapping[str, Any],
    *,
    episode_id: str,
    delay_s: float,
    proceed_key: str,
    quit_key: str,
) -> None:
    """Tell the operator that unsafe data was discarded before retrying."""

    attempt_id = str(attempt.get("attempt_id", "unknown"))
    violation = attempt.get("physics_violation")
    print(
        "[quest3-collector] Physics safety violation: "
        f"attempt {attempt_id} was discarded and will not be saved as "
        "successful data.",
        flush=True,
    )
    if isinstance(violation, Mapping):
        print(
            "[quest3-collector] Violation details: "
            + json.dumps(
                violation,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    print(
        f"[quest3-collector] Resetting logical episode {episode_id} in "
        f"{max(0.0, delay_s):.1f}s; press {proceed_key.upper()} to proceed "
        f"now or {quit_key.upper()} to quit.",
        flush=True,
    )


def _single_character(value: str) -> str:
    if len(value) != 1:
        raise argparse.ArgumentTypeError("command keys must be one character")
    return value


def _position(value: str) -> tuple[int, int]:
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("position must look like X,Y")
    try:
        x, y = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("position coordinates must be integers") from exc
    if x < 0 or y < 0:
        raise argparse.ArgumentTypeError("position coordinates must be non-negative")
    return x, y


def _resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like WIDTHxHEIGHT")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("resolution dimensions must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("resolution dimensions must be positive")
    return width, height


def _render_profile(args: argparse.Namespace) -> dict[str, Any]:
    """Return the render settings recorded in the collection manifest."""

    quest_view = bool(args.quest_view)
    return {
        "quest_view": quest_view,
        "quest_view_only": quest_view and args.render_mode == "none",
        "render_mode": args.render_mode,
        "render_backend": "gpu",
        "sim_backend": "cpu",
        "quest_view_fps": int(args.webxr_fps),
        "quest_view_resolution": (
            f"{DEFAULT_QUEST_VIEW_RESOLUTION[0]}x{DEFAULT_QUEST_VIEW_RESOLUTION[1]}"
            if quest_view
            else None
        ),
        "quest_view_quality": 80,
        "quest_view_display_scale": 0.6,
        "viewer_resolution": f"{args.viewer_resolution[0]}x{args.viewer_resolution[1]}",
        "viewer_window_size": (
            f"{args.viewer_window_size[0]}x{args.viewer_window_size[1]}"
        ),
        "viewer_window_position": (
            None
            if args.viewer_window_position is None
            else f"{args.viewer_window_position[0]},{args.viewer_window_position[1]}"
        ),
        "viewer_window_title": args.viewer_window_title,
        "startup_preparation_delay_s": float(args.startup_preparation_delay),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect multiple successful Quest 3 episodes into canonical HDF5."
    )
    parser.add_argument("--task-id", default="m01_v1")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/canonical"))
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--episode-prefix", default=None)
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--spool",
        action="store_true",
        help=(
            "Persist attempt chunk spools under <dataset>/.spool for recovery "
            "and diagnostics. Disabled by default; temporary chunks are "
            "removed after each attempt."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seed-mode", choices=("increment", "random", "fixed"), default="increment")
    parser.add_argument(
        "--generalization-profile",
        choices=("none", "collection"),
        default="none",
        help="Use the reduced-range collection generalization profile.",
    )
    parser.add_argument(
        "--generalization-axis",
        choices=("spatial", "physical", "balanced"),
        default=None,
        help="Collection axis, or balanced to assign one axis per logical episode.",
    )
    parser.add_argument("--generalization-seed", type=int, default=None)
    parser.add_argument("--operator-id", default="")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--record-tracking", action="store_true")
    parser.add_argument("--tracking-root", type=Path, default=None)
    parser.add_argument("--reset-key", type=_single_character, default="r")
    parser.add_argument("--quit-key", type=_single_character, default="q")
    parser.add_argument("--proceed-key", type=_single_character, default="e")
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--tracking-wait-timeout", type=float, default=120.0)
    parser.add_argument("--inter-episode-delay", type=float, default=10.0)
    parser.add_argument(
        "--startup-preparation-delay",
        type=float,
        default=1.0,
        help="Seconds after reset with viewer responsive but control/recording paused.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--canonical-queue-capacity", type=int, default=256)
    parser.add_argument("--canonical-chunk-length", type=int, default=256)
    parser.add_argument("--validate-after-episode", action="store_true", default=True)
    parser.add_argument("--no-validate-after-episode", dest="validate_after_episode", action="store_false")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--required-camera", action="append", default=[])
    parser.add_argument("--required-tactile-role", action="append", default=[])
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--image-hz", type=float, default=30.0)
    parser.add_argument("--sim-freq", type=int, default=120)
    parser.add_argument("--stale-timeout", type=float, default=5.0)
    parser.add_argument("--tracking-reset-timeout", type=float, default=5.0)
    parser.add_argument("--quest-view-stale-timeout", type=float, default=0.25)
    parser.add_argument(
        "--quest-view",
        dest="quest_view",
        action="store_true",
        default=False,
        help="Publish the host camera view to Quest.",
    )
    parser.add_argument(
        "--render-mode",
        choices=("none", "human"),
        default="human",
        help="Host viewer mode (default: human).",
    )
    parser.add_argument(
        "--viewer-resolution",
        type=_resolution,
        default=DEFAULT_VIEWER_RESOLUTION,
        help="Desktop viewer resolution as WIDTHxHEIGHT (default: 1920x1080).",
    )
    parser.add_argument(
        "--viewer-window-size",
        type=_resolution,
        default=None,
        help=(
            "Human Viewer outer window geometry. Defaults to 3072x1728 for "
            "b02_v1, b05_v1 through b10_v1, and b12_v1; defaults to "
            "2304x1296 for other tasks."
        ),
    )
    parser.add_argument(
        "--viewer-window-position",
        type=_position,
        default=None,
        help="Optional Human Viewer position as X,Y; default uses the window manager.",
    )
    parser.add_argument(
        "--viewer-window-title",
        default="SAPIEN",
        help="Case-insensitive title substring used by wmctrl to find Human Viewer.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--advertise-host",
        default=os.environ.get("QUEST3_LAN_IP", "10.42.0.1"),
    )
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--webxr-fps", type=int, default=30)
    parser.add_argument(
        "--cert",
        type=Path,
        default=(
            Path(os.environ["QUEST3_CERT"])
            if os.environ.get("QUEST3_CERT")
            else None
        ),
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=(
            Path(os.environ["QUEST3_KEY"])
            if os.environ.get("QUEST3_KEY")
            else None
        ),
    )
    parser.add_argument("--retargeting-config", type=Path, default=None)
    parser.add_argument("--left-retargeting-config", type=Path, default=None)
    parser.add_argument("--right-retargeting-config", type=Path, default=None)
    parser.add_argument("--retargeting-urdf-root", type=Path, default=None)
    parser.add_argument("--wrist-motion-mode", choices=("frame_delta", "legacy_target"), default="frame_delta")
    parser.add_argument("--position-scale", type=float, default=1.0)
    parser.add_argument("--rotation-scale", type=float, default=1.0)
    parser.add_argument("--max-position-delta", type=float, default=0.03)
    parser.add_argument("--max-rotation-delta", type=float, default=0.08)
    parser.add_argument("--max-target-distance", type=float, default=0.60)
    parser.add_argument("--hand-target-smoothing-alpha", type=float, default=1.0)
    parser.add_argument("--max-hand-joint-delta", type=float, default=0.1)
    parser.add_argument("--require-squeeze", action="store_true")
    parser.add_argument("--recenter-on-tap", action="store_true")
    args = parser.parse_args(argv)
    if args.viewer_window_size is None:
        args.viewer_window_size = (
            task_viewer_window_size(args.task_id) or DEFAULT_VIEWER_WINDOW_SIZE
        )
    return args


@dataclass(frozen=True)
class CollectorPreflight:
    task: Any
    mode: str
    required_cameras: tuple[str, ...]
    required_manipulators: tuple[str, ...]
    output_root: Path
    manifest_path: Path
    episode_prefix: str
    max_attempts: int


@dataclass(frozen=True)
class _AttemptSpool:
    """Own either a persistent diagnostic spool or temporary staging storage."""

    root: Path
    attempt_name: str
    persistent: bool
    temporary_directory: tempfile.TemporaryDirectory[str] | None = None

    def manifest_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {"spool_enabled": self.persistent}
        if self.persistent:
            fields["spool_path"] = str(self.root)
        return fields

    def cleanup(self) -> None:
        if self.temporary_directory is not None:
            self.temporary_directory.cleanup()


def _prepare_attempt_spool(
    *,
    output_root: Path,
    episode_id: str,
    attempt_index: int,
    persistent: bool,
) -> _AttemptSpool:
    """Allocate recorder staging without retaining it unless requested."""

    attempt_name = f"attempt-{attempt_index:04d}"
    if not persistent:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="dexbench-canonical-spool-"
        )
        return _AttemptSpool(
            root=Path(temporary_directory.name).resolve(),
            attempt_name=attempt_name,
            persistent=False,
            temporary_directory=temporary_directory,
        )

    spool_base = output_root.parent.parent / ".spool" / episode_id
    spool_dir = spool_base / attempt_name
    if spool_dir.exists():
        # A crash may have happened before the collection manifest was flushed.
        # Never reopen that directory as a recovery input for a new attempt.
        spool_dir = spool_base / f"{attempt_name}-fresh-{time.time_ns()}"
    return _AttemptSpool(
        root=spool_dir.resolve(),
        attempt_name=spool_dir.name,
        persistent=True,
    )


class _OperatorCommands:
    """Thread-safe command state shared by terminal monitor and collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proceed = threading.Event()
        self._stop = threading.Event()
        self.runner: Quest3TeleopRunner | None = None
        self.session_stop_reason: str | None = None

    def attach(self, runner: Quest3TeleopRunner) -> None:
        with self._lock:
            self.runner = runner

    def detach(self) -> None:
        with self._lock:
            self.runner = None

    def reset(self) -> None:
        with self._lock:
            runner = self.runner
        if runner is not None:
            runner.request_stop(reason="operator_reset")

    def quit(self) -> None:
        with self._lock:
            self.session_stop_reason = "quit_by_operator"
            runner = self.runner
        self._stop.set()
        if runner is not None:
            runner.request_stop(reason="operator_quit")

    def interrupt(self) -> None:
        with self._lock:
            self.session_stop_reason = "interrupted"
            runner = self.runner
        self._stop.set()
        if runner is not None:
            runner.request_stop(reason="interrupted")

    def proceed(self) -> None:
        self._proceed.set()

    def wait_proceed(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        result = False
        while True:
            if self._stop.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if self._proceed.wait(timeout=min(0.1, remaining)):
                result = True
                break
        self._proceed.clear()
        return result


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return value


def _resolve_manifest_path(
    args: argparse.Namespace,
    *,
    task_id: str,
    output_root: Path,
) -> Path:
    """Resolve a task-scoped manifest, with legacy single-task fallback."""

    if args.manifest_path is not None:
        return args.manifest_path.expanduser().resolve()

    task_manifest_path = output_root / "collection_manifest.json"
    if task_manifest_path.exists():
        return task_manifest_path

    # Before task-scoped manifests, the collector stored one manifest directly
    # under <dataset-root>/<dataset-id>. Reuse it only for the same task so a
    # dataset can now contain independent manifests for multiple tasks.
    legacy_manifest_path = output_root.parent.parent / "collection_manifest.json"
    if legacy_manifest_path.is_file():
        try:
            legacy_manifest = _load_json_mapping(legacy_manifest_path)
        except (OSError, ValueError):
            legacy_manifest = {}
        if (
            legacy_manifest.get("task_id") == task_id
            and legacy_manifest.get("split") == args.split
        ):
            return legacy_manifest_path
    return task_manifest_path


def _file_fingerprint(path: Path | None) -> dict[str, str | None] | None:
    """Return a stable path/content fingerprint for a config file."""

    if path is None:
        return None
    resolved = path.expanduser().resolve()
    digest: str | None = None
    if resolved.is_file():
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {"path": str(resolved), "sha256": digest}


def _validate_keys(args: argparse.Namespace) -> None:
    keys = {args.reset_key, args.quit_key, args.proceed_key}
    if len(keys) != 3:
        raise ValueError("reset, quit, and proceed keys must be distinct.")
    if args.episodes <= 0:
        raise ValueError("episodes must be positive.")
    max_attempts = args.max_attempts if args.max_attempts is not None else args.episodes * 3
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive.")
    if args.canonical_queue_capacity <= 0 or args.canonical_chunk_length <= 0:
        raise ValueError("canonical queue and chunk lengths must be positive.")
    if args.startup_timeout <= 0:
        raise ValueError("startup_timeout must be positive.")
    if not args.viewer_window_title.strip():
        raise ValueError("viewer_window_title must not be empty.")
    for name in (
        "tracking_wait_timeout",
        "inter_episode_delay",
        "startup_preparation_delay",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative.")
    if (args.cert is None) != (args.key is None):
        raise ValueError("--cert and --key must be provided together.")
    if args.cert is None or args.key is None:
        raise ValueError(
            "Formal Quest collection requires TLS: provide --cert/--key or "
            "export QUEST3_CERT and QUEST3_KEY."
        )
    if not args.cert.is_file():
        raise FileNotFoundError(f"Quest TLS certificate does not exist: {args.cert}")
    if not args.key.is_file():
        raise FileNotFoundError(f"Quest TLS private key does not exist: {args.key}")
    if args.episode_id is not None and args.episodes != 1:
        raise ValueError("--episode-id is only valid with --episodes 1.")


def preflight(args: argparse.Namespace) -> CollectorPreflight:
    _validate_keys(args)
    profile = _effective_generalization_profile(args)
    axis_mode = getattr(args, "generalization_axis", None)
    if profile == "collection" and axis_mode is None:
        axis_mode = "balanced"
        args.generalization_axis = axis_mode
    if profile == "collection" and axis_mode == "none":
        raise ValueError("collection generalization requires spatial, physical, or balanced axis.")
    if profile is None and axis_mode not in (None, "none"):
        raise ValueError(
            "generalization_axis requires --generalization-profile collection."
        )
    if getattr(args, "generalization_seed", None) is not None and axis_mode in (
        None,
        "none",
    ):
        raise ValueError("generalization_seed requires a generalization axis.")
    requested_seed = getattr(args, "generalization_seed", None)
    if requested_seed is not None:
        protocol = GeneralizationProtocol.load()
        if requested_seed not in protocol.seeds:
            raise ValueError(
                f"generalization_seed={requested_seed} is not in fixed protocol seeds "
                f"{protocol.seeds}."
            )
    if axis_mode not in (None, "none", "spatial", "physical", "balanced"):
        raise ValueError(
            f"Unsupported collection generalization axis {axis_mode!r}."
        )
    if Path(args.dataset_id).name != args.dataset_id or args.dataset_id in {".", ".."}:
        raise ValueError("dataset_id must be a single safe path component.")
    task = TaskRegistry.load(args.task_id)
    layouts = tuple(task.supported_layouts)
    if len(layouts) != 3:
        raise ValueError(
            f"Task {args.task_id!r} must expose exactly three supported layouts; "
            f"got {[layout.value for layout in layouts]!r}."
        )
    mode = "dual" if len(task.embodiment_spec.manipulator_roles) == 2 else "single"
    roles = tuple(role.value for role in task.embodiment_spec.manipulator_roles)
    if mode == "single" and args.hand_side not in roles:
        raise ValueError(
            f"Task {args.task_id!r} controls {roles!r}; --hand-side={args.hand_side!r} "
            "does not match the task embodiment."
        )
    cameras = tuple(role.value for role in task.embodiment_spec.camera_roles)
    required_cameras = tuple(dict.fromkeys((*cameras, *(str(item) for item in args.required_camera))))
    required_manipulators = tuple(dict.fromkeys((*roles, *(str(item) for item in args.required_tactile_role))))
    unknown_cameras = sorted(set(required_cameras) - set(cameras))
    unknown_roles = sorted(set(required_manipulators) - set(roles))
    if unknown_cameras:
        raise ValueError(f"Required cameras are not in the task embodiment: {unknown_cameras!r}.")
    if unknown_roles:
        raise ValueError(f"Required tactile roles are not in the task embodiment: {unknown_roles!r}.")
    validate_control_timing(control_hz=args.control_hz, sim_freq=args.sim_freq)
    validate_sensor_timing(control_hz=args.control_hz, image_hz=args.image_hz)
    if args.webxr_fps <= 0 or args.webxr_fps > 60:
        raise ValueError("webxr_fps must be in [1, 60].")
    output_root = (
        args.dataset_root.expanduser().resolve()
        / args.dataset_id
        / args.task_id
        / args.split
    )
    manifest_path = _resolve_manifest_path(
        args,
        task_id=args.task_id,
        output_root=output_root,
    )
    if manifest_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"Collection manifest already exists: {manifest_path}; use --resume.")
    if args.metadata_json is not None:
        metadata = _load_json_mapping(args.metadata_json.expanduser().resolve())
        protected = {
            "task_id",
            "dataset_id",
            "split",
            "episode_id",
            "attempt_id",
            "embodiment_id",
            "control_hz",
            "image_hz",
            "timing",
            "episode_budget",
            "episode_metadata",
            "action_spec",
            "sensors",
            "physics_safety",
        }
        if protected.intersection(metadata):
            raise ValueError("metadata-json cannot override core collection fields.")
    prefix = args.episode_prefix or args.task_id
    if not prefix.strip():
        raise ValueError("episode_prefix must not be empty.")
    if Path(prefix).name != prefix or prefix in {".", ".."}:
        raise ValueError("episode_prefix must be a single safe path component.")
    if args.episode_id is not None and (
        Path(args.episode_id).name != args.episode_id
        or args.episode_id in {".", ".."}
    ):
        raise ValueError("episode_id must be a single safe path component.")
    max_attempts = args.max_attempts if args.max_attempts is not None else args.episodes * 3
    if not args.resume and not args.overwrite:
        for index in range(args.episodes):
            episode_id = str(args.episode_id) if args.episode_id is not None else f"{prefix}_{index:05d}"
            output_path = output_root / f"{episode_id}.h5"
            if output_path.exists():
                raise FileExistsError(f"Canonical episode already exists: {output_path}")
    return CollectorPreflight(
        task=task,
        mode=mode,
        required_cameras=required_cameras,
        required_manipulators=required_manipulators,
        output_root=output_root,
        manifest_path=manifest_path,
        episode_prefix=prefix,
        max_attempts=max_attempts,
    )


def _episode_id(args: argparse.Namespace, preflight_result: CollectorPreflight, index: int) -> str:
    if args.episode_id is not None:
        return str(args.episode_id)
    return f"{preflight_result.episode_prefix}_{index:05d}"


def _attempt_seed(args: argparse.Namespace, logical_index: int, attempt_index: int) -> int:
    if args.seed_mode == "fixed":
        return int(args.seed)
    if args.seed_mode == "random":
        return random.Random(int(args.seed) + logical_index * 100003 + attempt_index).randrange(0, 2**31)
    return int(args.seed) + logical_index * 1000 + attempt_index


def _effective_generalization_profile(args: argparse.Namespace) -> str | None:
    requested = getattr(args, "generalization_profile", "none")
    axis = getattr(args, "generalization_axis", None)
    if requested == "none" and axis not in (None, "none"):
        # Preserve the historical collector meaning when callers supplied an
        # axis before the explicit profile option existed.
        return "evaluation"
    return None if requested == "none" else str(requested)


def _attempt_generalization_axis(
    args: argparse.Namespace,
    logical_index: int,
) -> str | None:
    axis = getattr(args, "generalization_axis", None)
    if axis is None and _effective_generalization_profile(args) == "collection":
        axis = "balanced"
    if axis in (None, "none"):
        return None
    if axis != "balanced":
        return str(axis)
    # Shuffle each pair independently so every two logical episodes contain
    # one spatial and one physical assignment while preserving a seed-defined
    # order.
    pair_index = logical_index // 2
    order = ["spatial", "physical"]
    random.Random(int(args.seed) + pair_index * 100003).shuffle(order)
    return order[logical_index % 2]


def _attempt_generalization_seed(
    args: argparse.Namespace,
    logical_index: int,
) -> int | None:
    axis = _attempt_generalization_axis(args, logical_index)
    if axis is None:
        return None
    protocol = GeneralizationProtocol.load()
    requested = getattr(args, "generalization_seed", None)
    if requested is not None:
        if requested not in protocol.seeds:
            raise ValueError(
                f"generalization_seed={requested} is not in fixed protocol seeds "
                f"{protocol.seeds}."
            )
        return int(requested)
    return protocol.seeds[logical_index % len(protocol.seeds)]


def _generalization_attempt_index(attempt_index: int) -> int:
    protocol = GeneralizationProtocol.load()
    if attempt_index >= protocol.max_resample_attempts:
        raise ValueError(
            "generalization attempts exhausted for one logical episode: "
            f"attempt_index={attempt_index} must be smaller than "
            f"{protocol.max_resample_attempts}."
        )
    return attempt_index


def _attempt_layout(args: argparse.Namespace, task: Any, logical_index: int, attempt_index: int) -> str:
    layouts = tuple(task.supported_layouts)
    rng = random.Random(int(args.seed) + logical_index * 100003 + attempt_index * 1009)
    return str(rng.choice(layouts).value)


def _resume_config(
    args: argparse.Namespace,
    preflight_result: CollectorPreflight,
) -> dict[str, Any]:
    """Build the data-semantic configuration that must stay fixed on resume."""

    task = preflight_result.task
    return {
        "version": RESUME_CONFIG_VERSION,
        "dataset_root": str(preflight_result.output_root.parent.parent.parent),
        "manifest_path": str(preflight_result.manifest_path),
        "episode_prefix": preflight_result.episode_prefix,
        "requested_episode_id": args.episode_id,
        "target_episodes": int(args.episodes),
        "seed": int(args.seed),
        "seed_mode": str(args.seed_mode),
        "generalization_axis": args.generalization_axis,
        "generalization_seed": args.generalization_seed,
        "generalization_profile": _effective_generalization_profile(args) or "none",
        "generalization_range_version": (
            "collection_v1"
            if _effective_generalization_profile(args) == "collection"
            else "evaluation_v1"
        ),
        "control_hz": float(args.control_hz),
        "image_hz": float(args.image_hz),
        "startup_preparation_delay_s": float(args.startup_preparation_delay),
        "sim_freq": int(args.sim_freq),
        "task_control_mode": str(task.control_mode.value),
        "task_obs_mode": str(task.obs_mode),
        "task_max_steps": int(task.max_steps),
        "supported_layouts": [str(layout.value) for layout in task.supported_layouts],
        "embodiment_id": str(task.embodiment_spec.embodiment_id),
        "hand_side": str(args.hand_side),
        "required_cameras": list(preflight_result.required_cameras),
        "required_manipulators": list(preflight_result.required_manipulators),
        "retargeting_config": _file_fingerprint(args.retargeting_config),
        "left_retargeting_config": _file_fingerprint(args.left_retargeting_config),
        "right_retargeting_config": _file_fingerprint(args.right_retargeting_config),
        "retargeting_urdf_root": (
            None
            if args.retargeting_urdf_root is None
            else str(args.retargeting_urdf_root.expanduser().resolve())
        ),
        "wrist_motion_mode": str(args.wrist_motion_mode),
        "position_scale": float(args.position_scale),
        "rotation_scale": float(args.rotation_scale),
        "max_position_delta": float(args.max_position_delta),
        "max_rotation_delta": float(args.max_rotation_delta),
        "max_target_distance": float(args.max_target_distance),
        "hand_target_smoothing_alpha": float(args.hand_target_smoothing_alpha),
        "max_hand_joint_delta": float(args.max_hand_joint_delta),
        "require_squeeze": bool(args.require_squeeze),
        "recenter_on_tap": bool(args.recenter_on_tap),
        "metadata_json": _file_fingerprint(args.metadata_json),
    }


def _env_metadata(env: Any) -> dict[str, Any]:
    getter = getattr(env, "get_episode_metadata", None)
    wrapper_getter = getattr(env, "get_wrapper_attr", None)
    if callable(wrapper_getter):
        try:
            getter = wrapper_getter("get_episode_metadata")
        except AttributeError:
            pass
    if not callable(getter):
        raise AttributeError("Task environment must expose get_episode_metadata().")
    metadata = getter()
    if not isinstance(metadata, Mapping):
        raise TypeError("get_episode_metadata() must return a mapping.")
    return dict(metadata)


def _manifest_template(args: argparse.Namespace, preflight_result: CollectorPreflight) -> dict[str, Any]:
    entries = [
        {"logical_index": index, "episode_id": _episode_id(args, preflight_result, index), "status": "pending", "attempts": []}
        for index in range(args.episodes)
    ]
    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": args.dataset_id,
        "task_id": args.task_id,
        "split": args.split,
        "session_id": args.session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "created_time_s": time.time(),
        "updated_time_s": time.time(),
        "target_episodes": args.episodes,
        "completed_episodes": 0,
        "config": {
            "seed": args.seed,
            "seed_mode": args.seed_mode,
            "control_hz": args.control_hz,
            "image_hz": args.image_hz,
            "sim_freq": args.sim_freq,
            "max_attempts": preflight_result.max_attempts,
            "embodiment_id": preflight_result.task.embodiment_spec.embodiment_id,
            "render_profile": _render_profile(args),
            "operator_id": args.operator_id,
            "notes": args.notes,
            "resume_config": _resume_config(args, preflight_result),
        },
        "episodes": entries,
    }


def _load_or_create_manifest(args: argparse.Namespace, preflight_result: CollectorPreflight) -> dict[str, Any]:
    path = preflight_result.manifest_path
    if args.resume and not path.exists():
        raise FileNotFoundError(f"Cannot resume without collection manifest: {path}")
    if not path.exists() or (args.overwrite and not args.resume):
        manifest = _manifest_template(args, preflight_result)
        _atomic_write_json(path, manifest)
        return manifest
    manifest = _load_json_mapping(path)
    for key, expected in (("dataset_id", args.dataset_id), ("task_id", args.task_id), ("split", args.split)):
        if manifest.get(key) != expected:
            raise ValueError(f"Existing manifest {key}={manifest.get(key)!r} does not match {expected!r}.")
    config = manifest.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("Existing manifest config must be an object.")
    for key, expected in (
        ("control_hz", args.control_hz),
        ("image_hz", args.image_hz),
        ("sim_freq", args.sim_freq),
    ):
        if key in config and float(config[key]) != float(expected):
            raise ValueError(
                f"Existing manifest config {key}={config[key]!r} does not match {expected!r}."
            )
    expected_resume_config = _resume_config(args, preflight_result)
    stored_resume_config = config.get("resume_config")
    if not isinstance(stored_resume_config, Mapping):
        raise ValueError(
            "Existing manifest has no strict resume configuration snapshot; "
            "start a new collection session instead of resuming it."
        )
    for key, expected in expected_resume_config.items():
        if key not in stored_resume_config:
            raise ValueError(
                f"Existing manifest resume_config is missing {key!r}; "
                "the session cannot be resumed safely."
            )
        if stored_resume_config[key] != expected:
            raise ValueError(
                f"Resume configuration changed for {key!r}: "
                f"manifest={stored_resume_config[key]!r}, current={expected!r}."
            )
    if int(manifest.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ValueError("Unsupported collection manifest version.")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != args.episodes:
        raise ValueError("Existing manifest episode count does not match --episodes.")
    if args.resume:
        for entry in episodes:
            if entry.get("status") != "completed":
                continue
            output_value = entry.get("attempts", [])[-1].get("output_path") if entry.get("attempts") else None
            if not output_value:
                raise ValueError(f"Completed manifest entry {entry.get('episode_id')!r} has no output path.")
            output_path = Path(str(output_value)).expanduser().resolve()
            if not output_path.is_file():
                raise FileNotFoundError(f"Completed canonical episode is missing: {output_path}")
            validate_canonical_hdf5(output_path, verify_checksums=args.verify_checksums)
    return manifest


def _save_manifest(manifest: dict[str, Any], path: Path) -> None:
    manifest["completed_episodes"] = sum(item.get("status") == "completed" for item in manifest["episodes"])
    manifest["updated_time_s"] = time.time()
    _atomic_write_json(path, manifest)


def _recording_metadata(
    *,
    env_metadata: Mapping[str, Any],
    task: Any,
    episode_id: str,
    layout_id: str,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    generalization = env_metadata.get("generalization")
    if not isinstance(generalization, Mapping):
        generalization = {
            "profile": _effective_generalization_profile(args),
            "axis": _attempt_generalization_axis(args, 0),
            "protocol_version": GeneralizationProtocol.load().version,
            "seed": getattr(args, "generalization_seed", None),
            "attempt": 0,
            "range_version": (
                "collection_v1"
                if _effective_generalization_profile(args) == "collection"
                else "evaluation_v1"
            ),
        }
    metadata: dict[str, Any] = {
        "task_id": task.task_id,
        "episode_id": episode_id,
        "embodiment_id": env_metadata["embodiment_id"],
        "control_hz": float(args.control_hz),
        "image_hz": float(args.image_hz),
        "seed": int(seed),
        "timing": {
            "clock_domain": "host_wall_time",
            "latency_metrics": [
                "received_to_published",
                "published_to_control",
            ],
            "quest_source_timestamp": "raw_diagnostic_only",
            "quest_source_timebase": "unknown",
        },
        "episode_budget": {
            "max_steps": int(task.max_steps),
            "max_episode_time_s": float(task.max_steps) / float(args.control_hz),
        },
        "episode_metadata": {
            "instruction": task.instruction,
            "layout_id": layout_id,
            "manipulator_roles": env_metadata["manipulator_roles"],
            "camera_roles": env_metadata["camera_roles"],
            "generalization": dict(generalization),
        },
        "action_spec": env_metadata.get("benchmark_action_spec", {}),
        "sensors": {
            "camera_roles": env_metadata.get("camera_roles", ()),
            "tactile_layouts": env_metadata.get("tactile_layouts", {}),
            "depth_encoding": "environment-defined",
        },
        "physics_safety": {
            "ik_stall_recovery_steps": 3,
            "max_joint_limit_excess_rad": 0.1,
            "recoverable_joint_limit_excess_rad": 0.25,
            "joint_limit_hold_steps": 3,
            "max_joint_velocity_rad_s": 200.0,
            "warning_joint_velocity_rad_s": 180.0,
            "max_taxel_force_n": DEFAULT_MAX_CONTACT_FORCE_N,
            "max_hand_force_n": DEFAULT_MAX_CONTACT_FORCE_N,
        },
    }
    if args.metadata_json is not None:
        metadata.update(_load_json_mapping(args.metadata_json.expanduser().resolve()))
    return metadata


def _build_teleop_args(args: argparse.Namespace, preflight_result: CollectorPreflight, layout_id: str) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "mode": preflight_result.mode,
            "hand_side": args.hand_side,
            "task_id": args.task_id,
            "env_id": None,
            "robot_uid": None,
            "control_mode": preflight_result.task.control_mode.value,
            "obs_mode": preflight_result.task.obs_mode,
            "reward_mode": "none" if preflight_result.mode == "dual" else "sparse",
            "shader": "default",
            "viewer_shader": "default",
            "viewer_resolution": args.viewer_resolution,
            "quest_view": args.quest_view,
            "quest_view_only": args.quest_view and args.render_mode == "none",
            "quest_view_fps": 30,
            "quest_view_resolution": DEFAULT_QUEST_VIEW_RESOLUTION,
            "quest_view_quality": 80,
            "quest_view_display_scale": 0.6,
            "render_mode": args.render_mode,
            "render_backend": "gpu",
            "sim_backend": "cpu",
            "render_every_n": 1,
            "quest_view_stale_timeout": args.quest_view_stale_timeout,
            "record_path": None,
        }
    )
    return argparse.Namespace(**values)


def _run_attempt(
    args: argparse.Namespace,
    preflight_result: CollectorPreflight,
    *,
    logical_index: int,
    episode_id: str,
    attempt_index: int,
    commands: _OperatorCommands,
    stream: Any,
    view_stream: Any,
) -> dict[str, Any]:
    layout_id = _attempt_layout(args, preflight_result.task, logical_index, attempt_index)
    seed = _attempt_seed(args, logical_index, attempt_index)
    teleop_args = _build_teleop_args(args, preflight_result, layout_id)
    generalization_profile = _effective_generalization_profile(args)
    generalization_axis = _attempt_generalization_axis(args, logical_index)
    generalization_seed = _attempt_generalization_seed(args, logical_index)
    teleop_args.generalization_profile = generalization_profile
    teleop_args.generalization_axis = generalization_axis
    teleop_args.generalization_seed = generalization_seed
    teleop_args.generalization_attempt = (
        _generalization_attempt_index(attempt_index)
        if generalization_axis is not None
        else 0
    )
    teleop_args.record_path = None
    if args.record_tracking:
        tracking_root = (args.tracking_root or (preflight_result.output_root.parent.parent / "tracking")).expanduser().resolve()
        tracking_root.mkdir(parents=True, exist_ok=True)
        teleop_args.record_path = tracking_root / f"{episode_id}.attempt-{attempt_index:04d}.jsonl"
    env = None
    session: Quest3CanonicalSession | None = None
    runner: Quest3TeleopRunner | None = None
    started = time.time()
    attempt_spool = _prepare_attempt_spool(
        output_root=preflight_result.output_root,
        episode_id=episode_id,
        attempt_index=attempt_index,
        persistent=bool(getattr(args, "spool", False)),
    )
    spool_dir = attempt_spool.root
    attempt_id = f"{episode_id}.{attempt_spool.attempt_name}"
    spool_manifest_fields = attempt_spool.manifest_fields()
    try:
        environment_started = time.monotonic()
        env = make_environment(
            teleop_args,
            render_mode="human" if args.render_mode == "human" else None,
            render_backend="gpu",
            quest_view_capture_every_n=1,
            layout_id=layout_id,
        )
        if time.monotonic() - environment_started > args.startup_timeout:
            raise TimeoutError(
                f"Environment startup exceeded {args.startup_timeout:.1f}s."
            )
        env_metadata = _env_metadata(env)
        metadata = _recording_metadata(
            env_metadata=env_metadata,
            task=preflight_result.task,
            episode_id=episode_id,
            layout_id=layout_id,
            args=args,
            seed=seed,
        )
        session = Quest3CanonicalSession(
            preflight_result.output_root,
            episode_id,
            control_hz=args.control_hz,
            image_hz=args.image_hz,
            required_cameras=preflight_result.required_cameras,
            required_manipulators=preflight_result.required_manipulators,
            metadata=metadata,
            queue_capacity=args.canonical_queue_capacity,
            chunk_length=args.canonical_chunk_length,
            spool_dir=spool_dir,
        )
        backend = BenchmarkSemanticBackend(
            env,
            default_role=None if preflight_result.mode == "dual" else args.hand_side,
        )
        config = Quest3TeleopConfig(
            mode=preflight_result.mode,
            hand_side=args.hand_side,
            control_hz=args.control_hz,
            stale_frame_timeout_s=args.stale_timeout,
            tracking_loss_reset_timeout_s=args.tracking_reset_timeout,
            tracking_wait_timeout_s=args.tracking_wait_timeout,
            render=args.render_mode == "human",
            startup_preparation_delay_s=args.startup_preparation_delay,
            view_every_n=1,
            exit_on_success=True,
            no_time_limit=False,
            require_squeeze=args.require_squeeze,
            recenter_on_tap=args.recenter_on_tap,
            retargeting_config_path=args.retargeting_config,
            left_retargeting_config_path=args.left_retargeting_config,
            right_retargeting_config_path=args.right_retargeting_config,
            retargeting_urdf_root=args.retargeting_urdf_root,
            record_path=teleop_args.record_path,
            wrist_motion=Quest3WristMotionConfig(
                motion_mode=args.wrist_motion_mode,
                position_scale=args.position_scale,
                rotation_scale=args.rotation_scale,
                max_position_delta_m=args.max_position_delta,
                max_rotation_delta_rad=args.max_rotation_delta,
                max_target_distance_m=args.max_target_distance,
                hand_target_smoothing_alpha=args.hand_target_smoothing_alpha,
                max_hand_joint_delta=args.max_hand_joint_delta,
            ),
        )

        def _on_viewer_ready(viewer: Any) -> None:
            del viewer
            window_id = configure_human_viewer_window(
                args.viewer_window_size,
                position=args.viewer_window_position,
                title_hint=args.viewer_window_title,
            )
            if window_id is not None:
                print(
                    "[quest3-teleop] Human Viewer window configured "
                    f"size={args.viewer_window_size[0]}x{args.viewer_window_size[1]}",
                    flush=True,
                )

        def _on_viewer_command(command: str) -> None:
            handlers = {
                "reset": commands.reset,
                "quit": commands.quit,
                "proceed": commands.proceed,
            }
            handlers[command]()

        runner = Quest3TeleopRunner(
            env,
            stream,
            backend=backend,
            on_recording_start=session.start_recording,
            on_reset_metadata=lambda reset_info: session.update_metadata(
                {
                    "seed": seed,
                    "layout_id": reset_info.get("layout_id", layout_id),
                    "reset_info": dict(reset_info),
                }
            ),
            on_step=session.on_step,
            on_view_frame=None if view_stream is None else view_stream.publish,
            view_ready=None if view_stream is None else lambda: view_stream.is_consumed_fresh(max_age_s=args.quest_view_stale_timeout),
            on_viewer_ready=_on_viewer_ready if args.render_mode == "human" else None,
            on_viewer_command=_on_viewer_command if args.render_mode == "human" else None,
            viewer_command_keys={
                "reset": args.reset_key,
                "quit": args.quit_key,
                "proceed": args.proceed_key,
            },
            config=config,
        )
        commands.attach(runner)
        runner.run(seed=seed)
        reason = runner.exit_reason or "runner_stopped_without_success"
        if reason == "success":
            output = session.close(status="completed")
            if output is None:
                raise RuntimeError("Task succeeded without canonical transitions.")
            try:
                report = (
                    session.validate_output(verify_checksums=args.verify_checksums)
                    if args.validate_after_episode
                    else {"path": str(output)}
                )
            except BaseException as exc:
                return {
                    "status": "validation_error",
                    "attempt_id": attempt_id,
                    **spool_manifest_fields,
                    "layout_id": layout_id,
                    "seed": seed,
                    "generalization_profile": generalization_profile,
                    "generalization_axis": generalization_axis,
                    "generalization_seed": generalization_seed,
                    "generalization_attempt": teleop_args.generalization_attempt,
                    "output_path": str(output),
                    "error": repr(exc),
                    "duration_s": time.time() - started,
                    "metrics": session.metrics,
                }
            return {
                "status": "completed",
                "attempt_id": attempt_id,
                **spool_manifest_fields,
                "layout_id": layout_id,
                "seed": seed,
                "generalization_profile": generalization_profile,
                "generalization_axis": generalization_axis,
                "generalization_seed": generalization_seed,
                "generalization_attempt": teleop_args.generalization_attempt,
                "output_path": str(output),
                "transition_count": report.get("transition_count"),
                "validation": report,
                "duration_s": time.time() - started,
                "metrics": session.metrics,
            }
        session.abort(reason)
        attempt_result: dict[str, Any] = {
            "status": _attempt_status_from_exit_reason(reason),
            "attempt_id": attempt_id,
            **spool_manifest_fields,
            "layout_id": layout_id,
            "seed": seed,
            "generalization_profile": generalization_profile,
            "generalization_axis": generalization_axis,
            "generalization_seed": generalization_seed,
            "generalization_attempt": teleop_args.generalization_attempt,
            "duration_s": time.time() - started,
            "metrics": session.metrics,
        }
        if (
            reason == "physics_violation"
            and runner.physics_violation is not None
        ):
            attempt_result["physics_violation"] = runner.physics_violation
        return attempt_result
    except BaseException as exc:
        if session is not None:
            session.abort(repr(exc))
        return {
            "status": f"error:{type(exc).__name__}",
            "attempt_id": attempt_id,
            **spool_manifest_fields,
            "layout_id": layout_id,
            "seed": seed,
            "generalization_profile": generalization_profile,
            "generalization_axis": generalization_axis,
            "generalization_seed": generalization_seed,
            "generalization_attempt": teleop_args.generalization_attempt,
            "error": repr(exc),
            "duration_s": time.time() - started,
        }
    finally:
        commands.detach()
        try:
            if env is not None:
                env.close()
        finally:
            attempt_spool.cleanup()


def collect(args: argparse.Namespace) -> dict[str, Any]:
    preflight_result = preflight(args)
    if args.dry_run:
        return _manifest_template(args, preflight_result)
    manifest = _load_or_create_manifest(args, preflight_result)
    if all(entry.get("status") == "completed" for entry in manifest["episodes"]):
        manifest["session_status"] = "completed"
        _save_manifest(manifest, preflight_result.manifest_path)
        return manifest
    commands = _OperatorCommands()
    global _ACTIVE_COMMANDS
    _ACTIVE_COMMANDS = commands
    monitor: TerminalQuitMonitor | None = None
    server: Quest3MultiprocessVuerServer | None = None
    try:
        server = Quest3MultiprocessVuerServer(
            config=Quest3VuerConfig(
                host=args.host,
                port=args.port,
                fps=args.webxr_fps,
                enable_left=preflight_result.mode == "dual" or args.hand_side == "left",
                enable_right=preflight_result.mode == "dual" or args.hand_side == "right",
                cert_path=args.cert,
                key_path=args.key,
                view_enabled=args.quest_view,
                view_fps=30,
                view_quality=80,
                view_display_scale=0.6,
            ),
            start_timeout_s=args.startup_timeout,
            view_resolution=DEFAULT_QUEST_VIEW_RESOLUTION,
        )
        server.start()
        advertise_host = args.advertise_host or "10.42.0.1"
        print(
            "[quest3-collector] Quest Browser: "
            f"https://{advertise_host}:{args.port}/?ws="
            f"wss://{advertise_host}:{args.port}",
            flush=True,
        )
        print(
            "[quest3-collector] TLS certificate check URL: "
            f"https://{advertise_host}:{args.port}",
            flush=True,
        )
        monitor = TerminalQuitMonitor(
            commands.quit,
            key=args.quit_key,
            on_reset=commands.reset,
            on_proceed=commands.proceed,
            reset_key=args.reset_key,
            proceed_key=args.proceed_key,
        )
        if monitor.start():
            print(
                f"[quest3-collector] commands: {args.reset_key}=reset "
                f"{args.proceed_key}=proceed {args.quit_key}=quit",
                flush=True,
            )
        entries = manifest["episodes"]
        attempts_this_run = 0
        for entry in entries:
            if entry.get("status") == "completed":
                continue
            logical_index = int(entry["logical_index"])
            episode_id = str(entry["episode_id"])
            while entry.get("status") != "completed":
                attempts = entry.setdefault("attempts", [])
                if attempts_this_run >= preflight_result.max_attempts:
                    raise RuntimeError(
                        "max_attempts exhausted for this collector process before "
                        "reaching target episodes; rerun with --resume to continue."
                    )
                attempt_index = len(attempts)
                attempts_this_run += 1
                attempt = _run_attempt(
                    args,
                    preflight_result,
                    logical_index=logical_index,
                    episode_id=episode_id,
                    attempt_index=attempt_index,
                    commands=commands,
                    stream=server.stream,
                    view_stream=server.view_stream,
                )
                attempts.append(attempt)
                entry["status"] = attempt["status"]
                entry["layout_id"] = attempt.get("layout_id")
                entry["last_attempt_id"] = attempt.get("attempt_id")
                _save_manifest(manifest, preflight_result.manifest_path)
                if attempt["status"] == "completed":
                    break
                if (
                    attempt["status"] in _TERMINAL_ATTEMPT_STATUSES
                    or commands.session_stop_reason is not None
                ):
                    manifest["session_status"] = (
                        commands.session_stop_reason or attempt["status"]
                    )
                    _save_manifest(manifest, preflight_result.manifest_path)
                    return manifest
                if not _is_retryable_attempt_status(str(attempt["status"])):
                    manifest["session_status"] = "error"
                    _save_manifest(manifest, preflight_result.manifest_path)
                    return manifest
                if attempt["status"] == "physics_violation":
                    _print_physics_violation_reset_notice(
                        attempt,
                        episode_id=episode_id,
                        delay_s=args.inter_episode_delay,
                        proceed_key=args.proceed_key,
                        quit_key=args.quit_key,
                    )
                commands.wait_proceed(args.inter_episode_delay)
                if commands.session_stop_reason is not None:
                    manifest["session_status"] = commands.session_stop_reason
                    _save_manifest(manifest, preflight_result.manifest_path)
                    return manifest
            if manifest["completed_episodes"] < args.episodes:
                commands.wait_proceed(args.inter_episode_delay)
                if commands.session_stop_reason is not None:
                    manifest["session_status"] = commands.session_stop_reason
                    _save_manifest(manifest, preflight_result.manifest_path)
                    return manifest
        manifest["session_status"] = "completed"
        _save_manifest(manifest, preflight_result.manifest_path)
        return manifest
    except BaseException as exc:
        manifest["session_status"] = "error"
        manifest["error"] = repr(exc)
        _save_manifest(manifest, preflight_result.manifest_path)
        raise
    finally:
        if monitor is not None:
            monitor.close()
        if server is not None:
            server.stop()
        _ACTIVE_COMMANDS = None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    def _signal_handler(_signum: int, _frame: Any) -> None:
        if _ACTIVE_COMMANDS is not None:
            _ACTIVE_COMMANDS.interrupt()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    manifest = collect(args)
    print(json.dumps(manifest, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
