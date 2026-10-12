"""Replay one canonical HDF5 episode and print a compact JSON review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.data.canonical.replay import replay_canonical_episode


def _stored_value(path: Path, name: str) -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional data dependency.
        raise ImportError("Replay requires h5py.") from exc
    with h5py.File(path, "r") as handle:
        value = handle.attrs.get(name)
        if value is None and f"meta/{name}.json" in handle:
            value = handle[f"meta/{name}.json"][()]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value


def _action_count(path: Path) -> int:
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle.get("actions/native")
        if group is None:
            return 0
        datasets: list[Any] = []

        def visit(node: Any) -> None:
            for item in node.values():
                if hasattr(item, "values"):
                    visit(item)
                else:
                    datasets.append(item)

        visit(group)
        return int(datasets[0].shape[0]) if datasets else 0


def _episode_metadata(path: Path) -> dict[str, Any]:
    value = _stored_value(path, "episode_metadata")
    return value if isinstance(value, dict) else {}


def _seed_from_collection_manifest(episode: Path) -> int | None:
    """Return the seed of the completed attempt that produced ``episode``."""

    manifest_path = episode.parent / "collection_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for record in manifest.get("episodes", ()):
        if record.get("episode_id") != episode.stem:
            continue
        completed = [
            attempt
            for attempt in record.get("attempts", ())
            if attempt.get("status") == "completed"
            and Path(str(attempt.get("output_path", ""))).name == episode.name
        ]
        if len(completed) != 1:
            return None
        value = completed[0].get("seed")
        return None if value is None else int(value)
    return None


def _rgb_frame(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    frame = np.asarray(value)
    if frame.ndim == 4:
        if frame.shape[0] != 1:
            raise ValueError(f"Replay video expects one environment, got {frame.shape[0]}.")
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] not in (3, 4):
        raise ValueError(f"render() returned unsupported shape {frame.shape}.")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.rint(np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


class _VideoRecorder:
    def __init__(self, path: Path, fps: int) -> None:
        if fps <= 0:
            raise ValueError("Video FPS must be positive.")
        try:
            import imageio.v2 as imageio
        except ImportError as exc:  # pragma: no cover - optional visualization dependency.
            raise ImportError("Video replay requires imageio and imageio-ffmpeg.") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)

    def capture(self, env: Any, _step: int) -> None:
        self._writer.append_data(_rgb_frame(env.render()))

    def close(self) -> None:
        self._writer.close()


def _flatten_numeric_group(group: Any, prefix: str = "") -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if group is None:
        return result
    for key in sorted(group.keys()):
        value = group[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if hasattr(value, "keys"):
            result.extend(_flatten_numeric_group(value, name))
        else:
            result.append((name, value))
    return result


def _canonical_observation_vectors(path: Path) -> list[Any]:
    """Read state/tactile only; RGB/depth are intentionally excluded."""

    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        state = _flatten_numeric_group(handle.get("observations/state"))
        tactile = _flatten_numeric_group(handle.get("observations/taxel"))
        features = state + tactile
        if not features:
            return []
        count = int(handle["observations/observation_id"].shape[0])
        return [
            np.concatenate([np.asarray(dataset[index]).reshape(-1) for _, dataset in features])
            for index in range(count)
        ]


def _replay_observation_vectors(
    observations: tuple[Any, ...], metadata: dict[str, Any], control_hz: float, image_hz: float
) -> list[Any]:
    import numpy as np
    from dex_benchmark.teleoperation.quest3.observation_adapter import Quest3ObservationAdapter

    required_cameras = tuple(str(value) for value in metadata.get("camera_roles", ()))
    required_manipulators = tuple(
        str(value) for value in metadata.get("manipulator_roles", ())
    )
    adapter = Quest3ObservationAdapter(
        control_hz=control_hz,
        image_hz=image_hz,
        required_cameras=required_cameras,
        required_manipulators=required_manipulators,
    )
    adapter.reset()
    vectors: list[Any] = []
    for index, observation in enumerate(observations):
        snapshot = adapter.snapshot(
            observation,
            observation_id=index,
            timestamp_s=float(index) / max(control_hz, 1.0),
            force_image=False,
        )
        values = [snapshot.state[key] for key in sorted(snapshot.state)] + [
            snapshot.taxel[key] for key in sorted(snapshot.taxel)
        ]
        vectors.append(np.concatenate([np.asarray(value).reshape(-1) for value in values]))
    return vectors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path, help="Canonical episode HDF5 file.")
    parser.add_argument("--task-id", default=None, help="Task id; defaults to HDF5 metadata.")
    parser.add_argument("--layout-id", default=None, help="Layout id; defaults to HDF5 metadata.")
    parser.add_argument("--seed", type=int, default=None, help="Replay seed; defaults to HDF5 metadata.")
    parser.add_argument("--control-hz", type=int, default=30)
    parser.add_argument("--sim-freq", type=int, default=120)
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--control-mode", default=None)
    parser.add_argument("--robot-uid", default=None)
    parser.add_argument("--sim-backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--render-backend", choices=("none", "cpu", "gpu"), default="gpu")
    parser.add_argument("--video", type=Path, default=None, help="Optional replay MP4 output.")
    parser.add_argument("--video-fps", type=int, default=None, help="Defaults to control_hz.")
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    from dex_benchmark.sim.tasks import make_task_env

    episode = args.episode.expanduser().resolve()
    task_id = args.task_id or _stored_value(episode, "task_id")
    if not task_id:
        parser.error("--task-id is required when the HDF5 has no task_id metadata.")
    metadata = _episode_metadata(episode)
    layout_id = args.layout_id or _stored_value(episode, "layout_id") or metadata.get("layout_id")
    seed_source = "argument" if args.seed is not None else "episode"
    seed = args.seed if args.seed is not None else _stored_value(episode, "seed")
    if seed is None:
        seed = _seed_from_collection_manifest(episode)
        seed_source = "collection_manifest" if seed is not None else "missing"
    if seed is not None:
        seed = int(seed)
    if seed is None:
        print(json.dumps({
            "seed": None,
            "status": "unreproducible",
            "replay_complete": False,
            "observation_match": None,
            "comparison_error": "reset_seed_missing",
        }, ensure_ascii=True, indent=2))
        return
    generalization = metadata.get("generalization")
    env_kwargs: dict[str, Any] = {
        "render_mode": "rgb_array" if args.video is not None else None,
        "render_backend": args.render_backend,
        "sim_backend": args.sim_backend,
        "sim_config": {"sim_freq": args.sim_freq, "control_freq": args.control_hz},
    }
    if layout_id:
        env_kwargs["layout_id"] = str(layout_id)
    if args.obs_mode:
        env_kwargs["obs_mode"] = args.obs_mode
    if args.control_mode:
        env_kwargs["control_mode"] = args.control_mode
    if args.robot_uid:
        env_kwargs["robot_uids"] = args.robot_uid
    if isinstance(generalization, dict):
        for key in ("profile", "axis", "seed", "attempt"):
            value = generalization.get(key)
            if value is not None:
                env_kwargs[f"generalization_{key}"] = value
    env = make_task_env(str(task_id), **env_kwargs)
    recorder = None
    try:
        if args.video is not None:
            recorder = _VideoRecorder(
                args.video.expanduser().resolve(),
                args.video_fps or args.control_hz,
            )
        result = replay_canonical_episode(
            env,
            episode,
            seed=seed,
            layout_id=layout_id,
            options=None,
            on_frame=recorder.capture if recorder is not None else None,
        )
    finally:
        if recorder is not None:
            recorder.close()
        close = getattr(env, "close", None)
        if callable(close):
            close()
    executed_actions = len(result.actions)
    expected_actions = _action_count(episode)
    early_terminated = any(
        value and index < expected_actions - 1
        for index, value in enumerate(result.terminated)
    )
    early_truncated = any(
        value and index < expected_actions - 1
        for index, value in enumerate(result.truncated)
    )
    complete = (
        executed_actions == expected_actions
        and not early_terminated
        and not early_truncated
    )
    observation_match: bool | None = None
    compared_observations = 0
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    comparison_error: str | None = None
    if complete and seed is not None:
        try:
            import numpy as np

            configured_control_hz = float(_stored_value(episode, "control_hz") or args.control_hz)
            configured_image_hz = float(_stored_value(episode, "image_hz") or configured_control_hz)
            expected_vectors = _canonical_observation_vectors(episode)
            replay_vectors = _replay_observation_vectors(
                result.observations,
                metadata,
                configured_control_hz,
                configured_image_hz,
            )
            compared_observations = min(len(expected_vectors), len(replay_vectors))
            if compared_observations != len(expected_vectors) or compared_observations != len(replay_vectors):
                comparison_error = "observation_count_mismatch"
            else:
                errors = []
                for expected, actual in zip(expected_vectors, replay_vectors):
                    if expected.shape != actual.shape:
                        comparison_error = "observation_shape_mismatch"
                        break
                    errors.append(np.abs(expected.astype(np.float64) - actual.astype(np.float64)))
                if comparison_error is None and errors:
                    all_errors = np.concatenate(errors)
                    max_abs_error = float(np.max(all_errors))
                    mean_abs_error = float(np.mean(all_errors))
                    observation_match = bool(max_abs_error <= args.atol)
        except Exception as exc:  # report unavailable comparison without hiding replay output
            comparison_error = f"comparison_unavailable:{type(exc).__name__}"
    elif seed is None:
        comparison_error = "reset_seed_missing"
    report = {
        "seed": result.seed,
        "seed_source": seed_source,
        "video": str(recorder.path) if recorder is not None else None,
        "status": (
            "failed" if not complete else
            "unreproducible" if seed is None else
            "match" if observation_match is True else
            "mismatch" if observation_match is False else
            "unavailable"
        ),
        "replay_complete": complete,
        "observation_match": observation_match,
        "compared_observations": compared_observations,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "comparison_error": comparison_error,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
