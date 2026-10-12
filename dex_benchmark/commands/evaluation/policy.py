from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dex_benchmark.evaluation.cpu_parallel import CPUParallelEnv
from dex_benchmark.evaluation.protocol_parallel import (
    CPUProtocolEvaluator,
    EvaluationProtocol,
    SingleEvaluationLog,
    build_protocol_schedule,
)
from dex_benchmark.sim.generalization import GeneralizationProtocol
from dex_benchmark.evaluation.video import ParallelVideoRecorder
from dex_benchmark.policy_api import PolicyRegistry
from dex_benchmark.policy_api.runner import create_centralized_policy_controller
from dex_benchmark.sim.tasks.bootstrap import load_builtin_task


def _json_object(value: str, *, option: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{option} must decode to a JSON object.")
    return parsed


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if not isinstance(value, dict):
        return value
    return {
        key: (
            "<redacted>"
            if any(
                marker in key.lower()
                for marker in ("token", "password", "secret", "api_key")
            )
            else _redact_secrets(item)
        )
        for key, item in value.items()
    }


def _default_log_path(task_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("log/evaluations") / f"{task_id}_{timestamp}_{uuid4().hex}.jsonl"


def _environment_overrides_preserve_standard_protocol(
    env_overrides: dict[str, Any],
    *,
    obs_mode: str,
) -> bool:
    """Accept only audited runtime optimizations that preserve observations."""

    if not env_overrides:
        return True
    return obs_mode == "state_dict" and env_overrides == {"render_backend": "none"}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed Evo-DexBench CPU parallel evaluation protocol."
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=EvaluationProtocol.default_path(),
        help="frozen evaluation protocol YAML",
    )
    registered_policies = ", ".join(PolicyRegistry.names())
    parser.add_argument(
        "--policy",
        required=True,
        help=(
            "installed policy name or module:callable factory"
            f" (registered: {registered_policies})"
        ),
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="diagnostic override for the protocol episode count",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="diagnostic override; formal evaluation uses the task YAML horizon",
    )
    parser.add_argument("--layout-ids", nargs=3, default=None)
    parser.add_argument("--control-mode")
    parser.add_argument("--obs-mode", default="rgb")
    parser.add_argument("--policy-kwargs", default="{}")
    parser.add_argument("--env-kwargs", default="{}")
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument(
        "--action-bounds-policy",
        choices=("error", "clip"),
        default=None,
        help=(
            "How to handle finite policy targets outside physical controller bounds. "
            "Formal evaluation defaults to clipping; use 'error' for strict debugging."
        ),
    )
    parser.add_argument(
        "--observation-capture",
        choices=("policy", "all"),
        default=None,
        help=(
            "Let chunk-aware policies skip unused intermediate camera frames "
            "or force the original every-step observation path."
        ),
    )
    parser.add_argument(
        "--log", type=Path,
        help="new result file; defaults to a unique path under log/evaluations",
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--video-mode", choices=("grid", "individual", "both"), default="both")
    args = parser.parse_args(argv)
    protocol_config_path = args.protocol_config.expanduser().resolve()
    evaluation_protocol = EvaluationProtocol.load(protocol_config_path)
    generalization_protocol = GeneralizationProtocol.load()
    num_envs = evaluation_protocol.num_envs if args.num_envs is None else args.num_envs
    episodes = (
        evaluation_protocol.episodes_per_task
        if args.episodes is None
        else args.episodes
    )
    action_bounds_policy = (
        evaluation_protocol.action_bounds_policy
        if args.action_bounds_policy is None
        else args.action_bounds_policy
    )
    observation_capture = (
        evaluation_protocol.observation_capture
        if args.observation_capture is None
        else args.observation_capture
    )
    if num_envs <= 0 or episodes <= 0:
        parser.error("--num-envs and --episodes must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive when provided.")
    if args.worker_timeout <= 0:
        parser.error("--worker-timeout must be positive.")

    settings = load_builtin_task(args.task_id)
    max_steps = settings.max_steps if args.max_steps is None else args.max_steps
    supported_layouts = tuple(layout.value for layout in settings.supported_layouts)
    layout_ids = tuple(args.layout_ids or supported_layouts)
    if len(layout_ids) != 3:
        parser.error(
            f"Task {args.task_id!r} must expose exactly three protocol layouts; "
            f"found {supported_layouts}."
        )
    unsupported = sorted(set(layout_ids) - set(supported_layouts))
    if unsupported:
        parser.error(f"Layouts are not supported by this task: {unsupported}.")

    schedule = build_protocol_schedule(
        task_id=args.task_id,
        layout_ids=layout_ids,
        episodes=episodes,
        protocol=generalization_protocol,
        evaluation_protocol=evaluation_protocol,
    )
    env_overrides = _json_object(args.env_kwargs, option="--env-kwargs")
    base_env_kwargs = dict(env_overrides)
    for reserved in (
        "layout_id",
        "generalization_axis",
        "generalization_profile",
        "generalization_seed",
        "generalization_attempt",
        "generalization_sample_index",
        "num_envs",
        "sim_backend",
        "sim_config",
    ):
        if reserved in base_env_kwargs:
            parser.error(f"--env-kwargs cannot override protocol field {reserved!r}.")
    base_env_kwargs["obs_mode"] = args.obs_mode
    base_env_kwargs["render_mode"] = "rgb_array" if args.video else None
    base_env_kwargs["sim_config"] = {
        "sim_freq": evaluation_protocol.sim_freq,
        "control_freq": evaluation_protocol.control_freq,
    }
    if args.control_mode is not None:
        base_env_kwargs["control_mode"] = args.control_mode

    policy_kwargs = _json_object(args.policy_kwargs, option="--policy-kwargs")
    first = schedule[0]
    initial_kwargs = first.env_kwargs(base_env_kwargs)
    log_path = (args.log or _default_log_path(args.task_id)).expanduser()
    try:
        log = SingleEvaluationLog(log_path)
    except FileExistsError:
        parser.error(f"Evaluation log already exists: {log_path}. Choose a new --log path.")
    env = policy = recorder = None
    try:
        log.write(
            "run",
            policy=args.policy,
            task_id=args.task_id,
            evaluation_protocol=evaluation_protocol.version,
            evaluation_protocol_path=str(protocol_config_path),
            standard_protocol=(
                protocol_config_path == EvaluationProtocol.default_path().resolve()
                and episodes == evaluation_protocol.episodes_per_task
                and num_envs == evaluation_protocol.num_envs
                and max_steps == settings.max_steps
                and layout_ids == supported_layouts
                and action_bounds_policy == evaluation_protocol.action_bounds_policy
                and observation_capture == evaluation_protocol.observation_capture
                and _environment_overrides_preserve_standard_protocol(
                    env_overrides,
                    obs_mode=args.obs_mode,
                )
            ),
            num_envs=num_envs,
            sim_freq=evaluation_protocol.sim_freq,
            control_freq=evaluation_protocol.control_freq,
            episodes=episodes,
            episodes_per_condition=evaluation_protocol.episodes_per_condition,
            max_steps=max_steps,
            layouts=layout_ids,
            generalization_protocol=generalization_protocol.version,
            generalization_seeds=list(generalization_protocol.seeds),
            schedule=[item.__dict__ for item in schedule],
            policy_kwargs=_redact_secrets(policy_kwargs),
            env_kwargs=_redact_secrets(base_env_kwargs),
            observation_capture=observation_capture,
            action_bounds_policy=action_bounds_policy,
        )
        env = CPUParallelEnv(
            task_id=args.task_id,
            env_kwargs=initial_kwargs,
            num_envs=num_envs,
            action_bounds_policy=action_bounds_policy,
            worker_timeout=args.worker_timeout,
        )
        policy = create_centralized_policy_controller(
            args.policy, policy_kwargs, num_envs=num_envs
        )
        if args.video:
            recorder = ParallelVideoRecorder(
                args.video.expanduser(),
                fps=args.video_fps,
                mode=args.video_mode,
                num_slots=num_envs,
            )
        evaluator = CPUProtocolEvaluator(
            env,
            policy,
            schedule=schedule,
            base_env_kwargs=base_env_kwargs,
            max_steps=max_steps,
            log=log,
            allow_observation_elision=observation_capture == "policy",
            max_resample_attempts=generalization_protocol.max_resample_attempts,
        )

        def start_video(episode_ids):
            if recorder is None:
                return None
            recorder.begin_wave(episode_ids)
            return recorder.capture

        result = evaluator.run(
            on_wave_start=start_video if recorder is not None else None,
            on_wave_end=recorder.end_wave if recorder is not None else None,
        )
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        print(f"Evaluation log: {log_path.resolve()}")
    except BaseException as exc:
        log.write("error", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if recorder is not None:
            recorder.close()
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
        log.close()


if __name__ == "__main__":
    main()
