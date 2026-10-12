"""Dataset-level quality checks for canonical HDF5 episodes.

The checks in this module deliberately stay at the observation/action contract
boundary.  Quest tracking diagnostics, task success and physical safety limits
belong to other parts of the collection and evaluation pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dex_benchmark.data.canonical.hdf5 import _require_h5py
from dex_benchmark.data.canonical.validate import validate_canonical_hdf5


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the focused dataset quality checks."""

    expected_control_hz: float = 30.0
    expected_image_hz: float = 30.0
    frequency_tolerance_ratio: float = 0.20
    repeated_image_run: int = 3
    required_cameras: tuple[str, ...] = ()
    required_tactile_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.expected_control_hz <= 0.0 or self.expected_image_hz <= 0.0:
            raise ValueError("Expected control and image frequencies must be positive.")
        if not 0.0 <= self.frequency_tolerance_ratio < 1.0:
            raise ValueError("frequency_tolerance_ratio must be in [0, 1).")
        if self.repeated_image_run < 2:
            raise ValueError("repeated_image_run must be at least two.")


_INVALID_CODES = frozenset(
    {
        "structural_invalid",
        "observation_action_mismatch",
        "timestamp_invalid",
        "image_missing",
        "image_invalid",
        "image_frequency_mismatch",
        "depth_invalid",
        "tactile_missing",
        "tactile_invalid",
        "duplicate_episode_id",
        "file_unreadable",
    }
)


def _flatten_datasets(group: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if group is None:
        return
    for key in sorted(group.keys()):
        value = group[key]
        name = f"{prefix}.{key}" if prefix else str(key)
        if hasattr(value, "keys"):
            yield from _flatten_datasets(value, name)
        else:
            yield name, value


def _percentiles(values: np.ndarray) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return {
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _finding(code: str, severity: str, message: str, **fields: Any) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        **fields,
    }


def _action_jump_metrics(
    group: Any,
) -> tuple[dict[str, Any], int]:
    """Report action smoothness using local second differences.

    A sustained large motion is not a spike. A point is counted only when its
    second difference is locally dominant and the local change reverses
    direction on at least one side. This avoids an episode-wide magnitude
    baseline.
    """
    metrics: dict[str, Any] = {}
    spike_count = 0
    for name, dataset in _flatten_datasets(group):
        values = np.asarray(dataset[...], dtype=np.float64)
        if values.ndim == 0 or values.shape[0] < 3:
            continue
        flat_values = values.reshape(values.shape[0], -1)
        deltas = np.diff(flat_values, axis=0)
        second_deltas = np.diff(deltas, axis=0)
        delta_abs = np.abs(deltas)
        second_abs = np.abs(second_deltas)
        if not np.isfinite(delta_abs).all() or not np.isfinite(second_abs).all():
            continue
        local_spikes: list[dict[str, int]] = []
        for dimension in range(deltas.shape[1]):
            for index in range(1, second_abs.shape[0] - 1):
                neighbours = max(
                    second_abs[index - 1, dimension], second_abs[index + 1, dimension]
                )
                direction_before = deltas[index - 1, dimension] * deltas[index, dimension]
                direction_after = deltas[index, dimension] * deltas[index + 1, dimension]
                if (
                    second_abs[index, dimension] > 1.5 * max(neighbours, 1e-12)
                    and direction_before <= 0.0
                    and direction_after <= 0.0
                ):
                    local_spikes.append({"dimension": dimension, "index": index})
        spike_count += len(local_spikes)
        metrics[name] = {
            "dimension_count": int(deltas.shape[1]),
            "delta": [
                _percentiles(delta_abs[:, dimension])
                for dimension in range(deltas.shape[1])
            ],
            "delta_change": [
                _percentiles(second_abs[:, dimension])
                for dimension in range(second_abs.shape[1])
            ],
            "isolated_spike_count": len(local_spikes),
            "isolated_spike_indices": local_spikes,
        }
    return metrics, spike_count


def _image_metrics(dataset: Any, *, repeated_run: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    sample_count = int(dataset.shape[0]) if len(dataset.shape) else 0
    metrics: dict[str, Any] = {"sample_count": sample_count}
    if dataset.dtype != np.uint8 or len(dataset.shape) != 4 or dataset.shape[-1] != 3:
        findings.append(
            _finding(
                "image_invalid",
                "invalid",
                "RGB must be uint8 with shape [N,H,W,3].",
                dtype=str(dataset.dtype),
                shape=list(dataset.shape),
            )
        )
        return metrics, findings
    black_count = 0
    duplicate_count = 0
    longest_duplicate_run = 1 if sample_count else 0
    duplicate_run = 1 if sample_count else 0
    previous_hash: bytes | None = None
    for index in range(sample_count):
        image = np.asarray(dataset[index])
        image_hash = hashlib.sha1(np.ascontiguousarray(image).tobytes()).digest()
        if previous_hash is not None and image_hash == previous_hash:
            duplicate_count += 1
            duplicate_run += 1
            longest_duplicate_run = max(longest_duplicate_run, duplicate_run)
        else:
            duplicate_run = 1
        previous_hash = image_hash
        if float(np.mean(image == 0)) >= 0.995:
            black_count += 1
    metrics.update(
        {
            "black_frame_count": black_count,
            "black_frame_ratio": black_count / sample_count if sample_count else 0.0,
            "consecutive_duplicate_count": duplicate_count,
            "longest_duplicate_run": longest_duplicate_run,
        }
    )
    if longest_duplicate_run >= repeated_run:
        findings.append(
            _finding(
                "repeated_image",
                "review",
                "RGB contains a consecutive repeated-frame run.",
                longest_run=longest_duplicate_run,
            )
        )
    if black_count:
        findings.append(
            _finding(
                "black_image",
                "review",
                "RGB contains near-black frames.",
                count=black_count,
            )
        )
    return metrics, findings


def _depth_metrics(dataset: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Check depth shape and finite values without loading the full stream."""

    findings: list[dict[str, Any]] = []
    sample_count = int(dataset.shape[0]) if len(dataset.shape) else 0
    metrics: dict[str, Any] = {"sample_count": sample_count}
    valid_shape = len(dataset.shape) == 3 or (
        len(dataset.shape) == 4 and dataset.shape[-1] == 1
    )
    if dataset.dtype.kind not in "biuf" or not valid_shape:
        findings.append(
            _finding(
                "depth_invalid",
                "invalid",
                "Depth must be numeric with shape [N,H,W] or [N,H,W,1].",
                dtype=str(dataset.dtype),
                shape=list(dataset.shape),
            )
        )
        return metrics, findings
    invalid_count = 0
    for index in range(sample_count):
        frame = np.asarray(dataset[index])
        if not np.isfinite(frame).all():
            invalid_count += 1
    metrics["invalid_frame_count"] = invalid_count
    if invalid_count:
        findings.append(
            _finding(
                "depth_invalid",
                "invalid",
                "Depth contains NaN or infinite values.",
                count=invalid_count,
            )
        )
    return metrics, findings


def inspect_episode(path: str | Path, config: QualityConfig | None = None) -> dict[str, Any]:
    """Inspect one HDF5 episode and return a JSON-compatible report."""

    quality_config = config or QualityConfig()
    episode_path = Path(path).expanduser().resolve()
    findings: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    episode_id = episode_path.stem
    try:
        h5py = _require_h5py()
        with h5py.File(episode_path, "r") as handle:
            episode_id = str(handle.attrs.get("episode_id", episode_id))
            try:
                structural = validate_canonical_hdf5(
                    episode_path,
                    required_cameras=quality_config.required_cameras,
                    required_tactile_roles=quality_config.required_tactile_roles,
                    check_quest=False,
                )
                del structural
            except (ImportError, ModuleNotFoundError, RuntimeError):
                raise
            except Exception as exc:  # report structural/data errors and continue scanning
                findings.append(
                    _finding("structural_invalid", "invalid", str(exc))
                )

            transitions = handle.get("transitions")
            observations = handle.get("observations")
            actions = handle.get("actions")
            if transitions is None or observations is None or actions is None:
                findings.append(
                    _finding(
                        "observation_action_mismatch",
                "invalid",
                        "Canonical HDF5 is missing transitions, observations or actions.",
                    )
                )
            else:
                transition_ids = np.asarray(transitions["transition_id"][:], dtype=np.int64)
                observation_ids = np.asarray(observations["observation_id"][:], dtype=np.int64)
                action_ids = np.asarray(actions["action_id"][:], dtype=np.int64)
                transition_count = int(len(transition_ids))
                metrics.update(
                    {
                        "transition_count": transition_count,
                        "observation_count": int(len(observation_ids)),
                        "action_count": int(len(action_ids)),
                        "duration_s": (
                            float(np.asarray(observations["time_s"][:], dtype=np.float64)[-1]
                                  - np.asarray(observations["time_s"][:], dtype=np.float64)[0])
                            if len(observation_ids)
                            else 0.0
                        ),
                        "held_action_ratio": float(
                            np.mean(np.asarray(transitions["held_action"][:], dtype=np.bool_))
                        ) if transition_count else 0.0,
                    }
                )
                if len(observation_ids) != transition_count + 1 or len(action_ids) != transition_count:
                    findings.append(
                        _finding(
                            "observation_action_mismatch",
                "invalid",
                            "Expected observation_count=transition_count+1 and action_count=transition_count.",
                        )
                    )

                control_times = np.asarray(
                    transitions["timestamps/control_tick_s"][:], dtype=np.float64
                ) if "timestamps/control_tick_s" in transitions else np.empty(0, dtype=np.float64)
                if len(control_times) != transition_count or not np.isfinite(control_times).all():
                    findings.append(
                        _finding(
                            "timestamp_invalid",
                "invalid",
                            "control_tick_s is missing or contains non-finite values.",
                        )
                    )
                else:
                    dt = np.diff(control_times)
                    positive_dt = dt[np.isfinite(dt) & (dt > 0.0)]
                    actual_hz = (
                        float(1.0 / np.median(positive_dt))
                        if len(positive_dt)
                        else None
                    )
                    metrics["actual_wall_tick_hz"] = actual_hz
                    metrics["control_period_s"] = _percentiles(positive_dt)
                    if np.any(dt <= 0.0):
                        findings.append(
                            _finding("timestamp_invalid", "invalid", "Control timestamps are not strictly increasing.")
                        )

                transition_observation_ids = np.asarray(
                    transitions["observation_id_t"][:], dtype=np.int64
                )
                transition_observation_ids_tp1 = np.asarray(
                    transitions["observation_id_tp1"][:], dtype=np.int64
                )
                transition_action_ids = np.asarray(
                    transitions["action_id"][:], dtype=np.int64
                )
                if not np.array_equal(transition_action_ids, action_ids) or not set(
                    map(int, transition_observation_ids)
                ).issubset(set(map(int, observation_ids))) or not np.array_equal(
                    transition_observation_ids, observation_ids[:-1]
                ) or not np.array_equal(
                    transition_observation_ids_tp1, observation_ids[1:]
                ):
                    findings.append(
                        _finding(
                            "observation_action_mismatch",
                "invalid",
                            "Transition references do not match materialized observations/actions.",
                        )
                    )

                action_times = np.asarray(actions["time_s"][:], dtype=np.float64)
                if len(action_times) != transition_count or (
                    len(control_times) == len(action_times)
                    and (not np.isfinite(action_times).all()
                         or np.max(np.abs(action_times - control_times)) > 1e-4)
                ):
                    findings.append(
                        _finding(
                            "observation_action_mismatch",
                "invalid",
                            "Action timestamps do not match transition control timestamps.",
                        )
                    )

                action_metrics, action_jump_count = _action_jump_metrics(
                    handle.get("actions/semantic"),
                )
                metrics["action_jump_by_field"] = action_metrics
                metrics["action_jump_count"] = action_jump_count
                if action_jump_count:
                    findings.append(
                        _finding(
                            "action_jump",
                            "review",
                            "Semantic action contains isolated local smoothness spikes.",
                            count=action_jump_count,
                        )
                    )

            configured_image_hz = float(handle.attrs.get("image_hz", quality_config.expected_image_hz))
            configured_control_hz = float(handle.attrs.get("control_hz", quality_config.expected_control_hz))
            metrics["configured_image_hz"] = configured_image_hz
            metrics["configured_control_hz"] = configured_control_hz
            if not np.isclose(configured_image_hz, quality_config.expected_image_hz):
                findings.append(
                    _finding(
                        "image_frequency_mismatch",
                        "invalid",
                        "Configured image frequency differs from the expected frequency.",
                        configured_hz=configured_image_hz,
                        expected_hz=quality_config.expected_image_hz,
                    )
                )

            image_group = handle.get("observations/images")
            depth_group = handle.get("observations/depth")
            image_id_group = handle.get("observations/image_observation_id")
            image_time_group = handle.get("observations/image_timestamp_s")
            available_cameras = set(image_group.keys()) if image_group is not None else set()
            for camera in quality_config.required_cameras:
                if camera not in available_cameras:
                    findings.append(
                        _finding("image_missing", "invalid", f"Missing required camera {camera!r}.")
                    )
            if image_group is not None:
                observation_ids = np.asarray(
                    handle["observations/observation_id"][:], dtype=np.int64
                )
                stride = configured_control_hz / configured_image_hz
                expected_ids = set(
                    map(int, observation_ids[observation_ids % max(1, round(stride)) == 0])
                ) if stride >= 1.0 and np.isclose(stride, round(stride)) else set()
                for camera in image_group.keys():
                    dataset = image_group[camera]
                    camera_ids = np.asarray(image_id_group[camera][:], dtype=np.int64) if image_id_group is not None and camera in image_id_group else np.empty(0, dtype=np.int64)
                    image_metrics, image_findings = _image_metrics(
                        dataset,
                        repeated_run=quality_config.repeated_image_run,
                    )
                    metrics[f"image.{camera}"] = image_metrics
                    findings.extend(image_findings)
                    if expected_ids:
                        missing_ids = expected_ids - set(map(int, camera_ids))
                        image_metrics["expected_sample_count"] = len(expected_ids)
                        image_metrics["missing_sample_count"] = len(missing_ids)
                        image_metrics["missing_ratio"] = len(missing_ids) / len(expected_ids)
                    if expected_ids and not expected_ids.issubset(set(map(int, camera_ids))):
                        findings.append(
                            _finding(
                                "image_missing",
                "invalid",
                                f"Camera {camera!r} is missing scheduled observation images.",
                            )
                        )
                    if len(np.unique(camera_ids)) != len(camera_ids):
                        findings.append(
                            _finding(
                                "image_invalid",
                "invalid",
                                f"Camera {camera!r} contains duplicate observation IDs.",
                            )
                        )
                    if not set(map(int, camera_ids)).issubset(set(map(int, observation_ids))):
                        findings.append(
                            _finding(
                                "image_invalid",
                "invalid",
                                f"Camera {camera!r} references unknown observations.",
                            )
                        )
                    if image_time_group is None or camera not in image_time_group:
                        findings.append(
                        _finding("image_invalid", "invalid", f"Camera {camera!r} is missing image timestamps.")
                        )
                    elif len(image_time_group[camera]) != len(camera_ids):
                        findings.append(
                        _finding("image_invalid", "invalid", f"Camera {camera!r} image timestamp count differs from image count.")
                        )
                    else:
                        image_times = np.asarray(image_time_group[camera][:], dtype=np.float64)
                        observation_times = np.asarray(observations["time_s"][:], dtype=np.float64)
                        id_to_time = {
                            int(item_id): float(item_time)
                            for item_id, item_time in zip(observation_ids, observation_times)
                        }
                        if not np.isfinite(image_times).all():
                            findings.append(
                                _finding(
                                    "image_invalid",
                "invalid",
                                    f"Camera {camera!r} image timestamps contain non-finite values.",
                                )
                            )
                        elif any(
                            int(observation_id) not in id_to_time
                            or abs(float(image_time) - id_to_time[int(observation_id)]) > 1e-4
                            for observation_id, image_time in zip(camera_ids, image_times)
                        ):
                            findings.append(
                                _finding(
                                    "observation_action_mismatch",
                "invalid",
                                    f"Camera {camera!r} image timestamps do not match observation timestamps.",
                                )
                            )
                    if depth_group is None or camera not in depth_group:
                        findings.append(
                        _finding("depth_invalid", "invalid", f"Camera {camera!r} is missing depth data.")
                        )
                    else:
                        depth_metrics, depth_findings = _depth_metrics(depth_group[camera])
                        metrics[f"depth.{camera}"] = depth_metrics
                        findings.extend(depth_findings)
                        if len(depth_group[camera]) != len(dataset):
                            findings.append(
                                _finding(
                                    "depth_invalid",
                "invalid",
                                    f"Camera {camera!r} RGB/depth sample count differs.",
                                )
                            )
                if depth_group is not None:
                    for camera in set(depth_group.keys()) - set(image_group.keys()):
                        findings.append(
                            _finding(
                                "depth_invalid",
                "invalid",
                                f"Depth camera {camera!r} has no matching RGB stream.",
                            )
                        )
            else:
                findings.append(
                    _finding(
                        "image_missing",
                        "invalid",
                        "Canonical observations contain no RGB camera stream.",
                    )
                )

            tactile_group = handle.get("observations/taxel")
            available_tactile = set(tactile_group.keys()) if tactile_group is not None else set()
            for role in quality_config.required_tactile_roles:
                if role not in available_tactile:
                    findings.append(
                        _finding("tactile_missing", "invalid", f"Missing required tactile role {role!r}.")
                    )
            if tactile_group is not None:
                for role in tactile_group.keys():
                    values = np.asarray(tactile_group[role][:], dtype=np.float64)
                    role_metrics: dict[str, Any] = {
                        "sample_count": int(values.shape[0]) if values.ndim else 0,
                        "missing_ratio": (
                            max(0, len(observations["observation_id"]) - len(values))
                            / len(observations["observation_id"])
                            if len(observations["observation_id"])
                            else 0.0
                        ),
                        "all_zero_ratio": float(np.mean(np.all(values == 0, axis=tuple(range(1, values.ndim))))) if values.ndim > 1 else 0.0,
                    }
                    metrics[f"tactile.{role}"] = role_metrics
                    observation_count = len(observations["observation_id"])
                    if len(values) != observation_count:
                        findings.append(
                            _finding(
                                "tactile_missing",
                                "invalid",
                                f"Tactile role {role!r} sample count does not match observations.",
                                missing_ratio=role_metrics["missing_ratio"],
                                sample_count=len(values),
                                observation_count=observation_count,
                            )
                        )
                    if values.ndim == 0 or not np.isfinite(values).all():
                        findings.append(
                            _finding("tactile_invalid", "invalid", f"Tactile role {role!r} contains invalid values.")
                        )
                    elif role_metrics["all_zero_ratio"] >= 0.99:
                        findings.append(
                            _finding("tactile_all_zero", "review", f"Tactile role {role!r} is almost entirely zero.")
                        )
            elif quality_config.required_tactile_roles or not available_tactile:
                findings.append(
                    _finding(
                        "tactile_missing",
                        "invalid",
                        "Canonical observations contain no tactile stream.",
                    )
                )

    except (ImportError, ModuleNotFoundError, RuntimeError):
        raise
    except Exception as exc:
        findings.append(_finding("file_unreadable", "invalid", repr(exc)))

    invalid = any(item["code"] in _INVALID_CODES for item in findings)
    needs_review = any(item["severity"] == "review" for item in findings)
    compact_metrics: dict[str, Any] = {
        key: metrics[key]
        for key in (
            "transition_count",
            "observation_count",
            "action_count",
            "duration_s",
            "held_action_ratio",
            "configured_control_hz",
            "configured_image_hz",
            "actual_wall_tick_hz",
        )
        if key in metrics
    }
    compact_metrics["images"] = {
        str(name.removeprefix("image.")): {
            key: value
            for key, value in image_metrics.items()
            if key
            in {
                "sample_count",
                "black_frame_ratio",
                "consecutive_duplicate_count",
                "longest_duplicate_run",
                "missing_ratio",
            }
        }
        for name, image_metrics in metrics.items()
        if str(name).startswith("image.") and isinstance(image_metrics, dict)
    }
    compact_metrics["tactile"] = {
        str(name.removeprefix("tactile.")): {
            key: value
            for key, value in tactile_metrics.items()
            if key in {"sample_count", "missing_ratio", "all_zero_ratio"}
        }
        for name, tactile_metrics in metrics.items()
        if str(name).startswith("tactile.") and isinstance(tactile_metrics, dict)
    }
    compact_metrics["action_smoothness"] = {
        str(name): {
            "isolated_spike_count": int(value.get("isolated_spike_count", 0)),
            "dimension_count": int(value.get("dimension_count", 0)),
        }
        for name, value in metrics.get("action_jump_by_field", {}).items()
        if isinstance(value, dict)
    }
    return {
        "episode_id": episode_id,
        "source_path": str(episode_path),
        "valid": not invalid,
        "needs_review": needs_review,
        "reasons": list(dict.fromkeys(item["code"] for item in findings)),
        "metrics": compact_metrics,
    }


def _episode_files(input_path: Path, output_root: Path) -> list[Path]:
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(
            set(input_path.rglob("*.h5")) | set(input_path.rglob("*.hdf5"))
        )
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if ".spool" in resolved.parts or resolved.name.endswith(".tmp"):
            continue
        try:
            resolved.relative_to(output_root)
        except ValueError:
            result.append(resolved)
    return result


def quality_check_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    config: QualityConfig | None = None,
) -> Path:
    """Check all HDF5 episodes and write one JSON report."""

    source = Path(input_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    quality_config = config or QualityConfig()
    # Environment/dependency failures must abort before any report is written.
    _require_h5py()
    reports = [inspect_episode(path, quality_config) for path in _episode_files(source, output_root)]
    by_id: dict[str, list[dict[str, Any]]] = {}
    for report in reports:
        by_id.setdefault(str(report["episode_id"]), []).append(report)
    for episode_id, duplicates in by_id.items():
        if len(duplicates) < 2:
            continue
        for report in duplicates:
            report["valid"] = False
            report["needs_review"] = False
            report["reasons"].append("duplicate_episode_id")
    summary = {
        "valid": sum(bool(report["valid"]) for report in reports),
        "invalid": sum(not bool(report["valid"]) for report in reports),
        "needs_review": sum(bool(report["needs_review"]) for report in reports),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "quality_report.json"
    report_path.write_text(
        json.dumps(
            {
                "report_version": 2,
                "input_path": str(source),
                "expected_control_hz": quality_config.expected_control_hz,
                "expected_image_hz": quality_config.expected_image_hz,
                "summary": summary,
                "episodes": reports,
            },
            ensure_ascii=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path
