from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from dex_benchmark.evaluation.cpu_parallel import (
    CPUParallelEnv,
    FrameCallback,
    run_cpu_parallel_wave,
)
from dex_benchmark.evaluation.metrics import OnlineMetricAccumulator
from dex_benchmark.sim.generalization import GeneralizationAxis, GeneralizationProtocol
from dex_benchmark.resources import evaluation_protocol_path


CONDITIONS = ("nominal",) + tuple(axis.value for axis in GeneralizationAxis)
_EVALUATION_PROTOCOL_FIELDS = {
    "version",
    "episodes_per_condition",
    "layouts_per_task",
    "num_envs",
    "sim_freq",
    "control_freq",
    "action_bounds_policy",
    "observation_capture",
}


@dataclass(frozen=True)
class EvaluationProtocol:
    """Frozen execution settings for reportable benchmark evaluations."""

    version: str
    episodes_per_condition: int
    layouts_per_task: int
    num_envs: int
    sim_freq: int
    control_freq: int
    action_bounds_policy: str
    observation_capture: str

    @staticmethod
    def default_path() -> Path:
        return evaluation_protocol_path()

    @property
    def episodes_per_task(self) -> int:
        return self.episodes_per_condition * len(CONDITIONS)

    @classmethod
    def load(cls, path: Path | None = None) -> "EvaluationProtocol":
        protocol_path = path or cls.default_path()
        raw_data = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise TypeError(
                f"Evaluation protocol {protocol_path} must contain a mapping."
            )
        unknown_fields = sorted(set(raw_data) - _EVALUATION_PROTOCOL_FIELDS)
        if unknown_fields:
            raise ValueError(
                "Evaluation protocol contains unsupported fields: "
                + ", ".join(unknown_fields)
                + "."
            )
        missing_fields = sorted(_EVALUATION_PROTOCOL_FIELDS - set(raw_data))
        if missing_fields:
            raise ValueError(
                "Evaluation protocol is missing fields: "
                + ", ".join(missing_fields)
                + "."
            )
        version = raw_data["version"]
        if not isinstance(version, str) or not version:
            raise TypeError("Evaluation protocol version must be a non-empty string.")
        integer_fields = (
            "episodes_per_condition",
            "layouts_per_task",
            "num_envs",
            "sim_freq",
            "control_freq",
        )
        for field_name in integer_fields:
            value = raw_data[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if raw_data["layouts_per_task"] != 3:
            raise ValueError("The standard protocol requires exactly three layouts per task.")
        if raw_data["sim_freq"] % raw_data["control_freq"] != 0:
            raise ValueError("sim_freq must be divisible by control_freq.")
        if raw_data["action_bounds_policy"] not in {"error", "clip"}:
            raise ValueError("action_bounds_policy must be 'error' or 'clip'.")
        if raw_data["observation_capture"] not in {"policy", "all"}:
            raise ValueError("observation_capture must be 'policy' or 'all'.")
        return cls(
            version=version,
            episodes_per_condition=int(raw_data["episodes_per_condition"]),
            layouts_per_task=int(raw_data["layouts_per_task"]),
            num_envs=int(raw_data["num_envs"]),
            sim_freq=int(raw_data["sim_freq"]),
            control_freq=int(raw_data["control_freq"]),
            action_bounds_policy=str(raw_data["action_bounds_policy"]),
            observation_capture=str(raw_data["observation_capture"]),
        )


def _stable_seed(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**31)


def _balanced_counts(total: int, count: int, *, rotation: int = 0) -> list[int]:
    base, remainder = divmod(total, count)
    result = [base] * count
    for offset in range(remainder):
        result[(rotation + offset) % count] += 1
    return result


@dataclass(frozen=True)
class ProtocolEpisode:
    episode_id: int
    condition: str
    layout_id: str
    reset_seed: int
    generalization_seed: int | None
    generalization_sample_index: int | None
    generalization_attempt: int = 0

    def env_kwargs(self, base: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(base)
        result["layout_id"] = self.layout_id
        if self.condition == "nominal":
            for key in (
                "generalization_axis",
                "generalization_profile",
                "generalization_seed",
                "generalization_attempt",
                "generalization_sample_index",
            ):
                result.pop(key, None)
        else:
            result.update(
                generalization_profile="evaluation",
                generalization_axis=self.condition,
                generalization_seed=self.generalization_seed,
                generalization_attempt=self.generalization_attempt,
                generalization_sample_index=self.generalization_sample_index,
            )
        return result


def build_protocol_schedule(
    *,
    task_id: str,
    layout_ids: Sequence[str],
    episodes: int | None = None,
    protocol: GeneralizationProtocol | None = None,
    evaluation_protocol: EvaluationProtocol | None = None,
) -> tuple[ProtocolEpisode, ...]:
    evaluation_protocol = evaluation_protocol or EvaluationProtocol.load()
    episodes = (
        evaluation_protocol.episodes_per_task if episodes is None else episodes
    )
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    layouts = tuple(str(value) for value in layout_ids)
    if (
        len(layouts) != evaluation_protocol.layouts_per_task
        or len(set(layouts)) != evaluation_protocol.layouts_per_task
    ):
        raise ValueError(
            "Protocol evaluation requires exactly "
            f"{evaluation_protocol.layouts_per_task} distinct layouts."
        )
    protocol = protocol or GeneralizationProtocol.load()
    condition_counts = _balanced_counts(episodes, len(CONDITIONS))
    schedule: list[ProtocolEpisode] = []
    episode_id = 0
    for condition_index, (condition, condition_count) in enumerate(
        zip(CONDITIONS, condition_counts, strict=True)
    ):
        # For the default 20 rows this produces the agreed 7/7/6 rotation and
        # an aggregate 34/33/33 allocation over five conditions.
        layout_counts = _balanced_counts(
            condition_count,
            len(layouts),
            rotation=(-condition_index) % len(layouts),
        )
        assigned_layouts = [
            layout
            for layout, count in zip(layouts, layout_counts, strict=True)
            for _ in range(count)
        ]
        if condition == "nominal":
            generalized_rows: list[tuple[int | None, int | None]] = [
                (None, None) for _ in range(condition_count)
            ]
        else:
            seed_counts = _balanced_counts(condition_count, len(protocol.seeds))
            generalized_rows = []
            for generalization_seed, seed_count in zip(
                protocol.seeds, seed_counts, strict=True
            ):
                generalized_rows.extend(
                    (generalization_seed, sample_index)
                    for sample_index in range(seed_count)
                )
        for local_index, (layout_id, generalized) in enumerate(
            zip(assigned_layouts, generalized_rows, strict=True)
        ):
            generalization_seed, sample_index = generalized
            reset_seed = _stable_seed(
                protocol.version,
                task_id,
                condition,
                layout_id,
                generalization_seed,
                sample_index,
                local_index,
            )
            schedule.append(
                ProtocolEpisode(
                    episode_id=episode_id,
                    condition=condition,
                    layout_id=layout_id,
                    reset_seed=reset_seed,
                    generalization_seed=generalization_seed,
                    generalization_sample_index=sample_index,
                )
            )
            episode_id += 1
    return tuple(schedule)


def build_nominal_schedule(
    *,
    task_id: str,
    layout_ids: Sequence[str],
    episodes: int,
    seed_namespace: str = "execute_horizon_sweep_v1",
) -> tuple[ProtocolEpisode, ...]:
    """Build a reproducible, layout-balanced nominal-only schedule.

    Reset seeds deliberately exclude policy parameters such as execute horizon,
    so every sweep candidate sees exactly the same initial states.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    layouts = tuple(str(value) for value in layout_ids)
    if not layouts or len(set(layouts)) != len(layouts):
        raise ValueError("layout_ids must contain distinct layouts.")
    layout_counts = _balanced_counts(episodes, len(layouts))
    assigned_layouts = [
        layout
        for layout, count in zip(layouts, layout_counts, strict=True)
        for _ in range(count)
    ]
    return tuple(
        ProtocolEpisode(
            episode_id=episode_id,
            condition="nominal",
            layout_id=layout_id,
            reset_seed=_stable_seed(
                seed_namespace,
                task_id,
                "nominal",
                layout_id,
                episode_id,
            ),
            generalization_seed=None,
            generalization_sample_index=None,
        )
        for episode_id, layout_id in enumerate(assigned_layouts)
    )


def execution_order(schedule: Sequence[ProtocolEpisode]) -> tuple[ProtocolEpisode, ...]:
    """Group reset-only configurations while retaining stable episode IDs."""

    condition_order = {condition: index for index, condition in enumerate(CONDITIONS)}
    return tuple(
        sorted(
            schedule,
            key=lambda item: (
                item.layout_id,
                condition_order[item.condition],
                -1 if item.generalization_seed is None else item.generalization_seed,
                -1
                if item.generalization_sample_index is None
                else item.generalization_sample_index,
                item.episode_id,
            ),
        )
    )


class SingleEvaluationLog:
    """Append-only JSON-lines log containing configuration, episodes and summary."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("x", encoding="utf-8")

    @staticmethod
    def _default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}.")

    def write(self, event: str, **payload: Any) -> None:
        row = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        self._handle.write(json.dumps(row, default=self._default, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        accumulator = OnlineMetricAccumulator()
        for row in rows:
            accumulator.update(dict(row))
        result: dict[str, Any] = accumulator.result()
        result["episodes"] = len(rows)
        return result

    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_layout: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_cross: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        condition = str(row["condition"])
        layout_id = str(row["layout_id"])
        by_condition[condition].append(row)
        by_layout[layout_id].append(row)
        by_cross[f"{condition}/{layout_id}"].append(row)
    condition_summary = {
        key: aggregate(value) for key, value in sorted(by_condition.items())
    }
    return {
        "overall": aggregate(records),
        "by_condition": condition_summary,
        "by_layout": {key: aggregate(value) for key, value in sorted(by_layout.items())},
        "by_condition_layout": {
            key: aggregate(value) for key, value in sorted(by_cross.items())
        },
    }


@dataclass(frozen=True)
class ProtocolEvaluationResult:
    records: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


class CPUProtocolEvaluator:
    def __init__(
        self,
        env: CPUParallelEnv,
        policy: Any,
        *,
        schedule: Sequence[ProtocolEpisode],
        base_env_kwargs: Mapping[str, Any],
        max_steps: int,
        log: SingleEvaluationLog,
        allow_observation_elision: bool = True,
        max_resample_attempts: int | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.schedule = tuple(schedule)
        self.base_env_kwargs = dict(base_env_kwargs)
        self.max_steps = max_steps
        self.log = log
        self.allow_observation_elision = bool(allow_observation_elision)
        self.max_resample_attempts = (
            GeneralizationProtocol.load().max_resample_attempts
            if max_resample_attempts is None
            else int(max_resample_attempts)
        )
        if self.max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive.")

    def run(
        self,
        *,
        on_wave_start: Callable[[Sequence[int]], FrameCallback | None] | None = None,
        on_wave_end: Callable[[], None] | None = None,
    ) -> ProtocolEvaluationResult:
        ordered = execution_order(self.schedule)
        records: list[dict[str, Any]] = []
        for wave_index, start in enumerate(range(0, len(ordered), self.env.num_envs)):
            rows = list(ordered[start : start + self.env.num_envs])
            active_count = len(rows)
            rebuilds = [False] * self.env.num_envs
            while True:
                padded = list(rows)
                while len(padded) < self.env.num_envs:
                    padded.append(rows[-1])
                configurations = [row.env_kwargs(self.base_env_kwargs) for row in padded]
                statuses = self.env.prepare(
                    configurations,
                    [row.reset_seed for row in padded],
                )
                invalid_slots = []
                for slot, status in enumerate(statuses[:active_count]):
                    rebuilds[slot] |= bool(status.get("rebuilt", False))
                    if not bool(status.get("valid", False)):
                        invalid_slots.append(slot)
                        rejected = rows[slot]
                        self.log.write(
                            "rejected_sample",
                            episode=asdict(rejected),
                            message=status.get("message", "invalid generalization sample"),
                        )
                if not invalid_slots:
                    break
                for slot in invalid_slots:
                    if (
                        rows[slot].generalization_attempt + 1
                        >= self.max_resample_attempts
                    ):
                        raise RuntimeError(
                            "Generalization sample remained invalid after "
                            f"{self.max_resample_attempts} attempts: "
                            f"episode_id={rows[slot].episode_id}."
                        )
                    rows[slot] = replace(
                        rows[slot],
                        generalization_attempt=rows[slot].generalization_attempt + 1,
                    )
            callback = on_wave_start([row.episode_id for row in rows]) if on_wave_start else None
            started = time.perf_counter()
            try:
                result = run_cpu_parallel_wave(
                    self.env,
                    self.policy,
                    seeds=[row.reset_seed for row in padded],
                    active_count=active_count,
                    max_steps=self.max_steps,
                    on_frame=callback,
                    allow_observation_elision=self.allow_observation_elision,
                )
            finally:
                if on_wave_end is not None:
                    on_wave_end()
            elapsed = time.perf_counter() - started
            for slot, (spec, raw_record) in enumerate(zip(rows, result.records, strict=True)):
                record = {
                    **raw_record,
                    **asdict(spec),
                    "worker_slot": slot,
                    "wave_index": wave_index,
                    "environment_rebuilt": rebuilds[slot],
                    "wave_wall_time_s": elapsed,
                }
                records.append(record)
                running = summarize_records(records)["overall"]
                self.log.write(
                    "episode",
                    record=record,
                    completed=len(records),
                    total=len(self.schedule),
                    cumulative_success_rate=running["success_rate"],
                )
        records.sort(key=lambda row: int(row["episode_id"]))
        summary = summarize_records(records)
        self.log.write("summary", summary=summary)
        return ProtocolEvaluationResult(records=tuple(records), summary=summary)
