from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dex_benchmark.policy_api.model_input import (
    ModelInput,
    build_model_input,
    instruction_from_env,
)
from dex_benchmark.policy_api.session import PolicySession


def _bind_environment(env: Any, sessions: Sequence[Any]) -> None:
    aware_sessions = [
        session
        for session in sessions
        if callable(getattr(session, "bind_environment", None))
    ]
    if not aware_sessions:
        return
    get_metadata = getattr(env, "get_episode_metadata", None)
    if not callable(get_metadata):
        get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
        if callable(get_wrapper_attr):
            try:
                get_metadata = get_wrapper_attr("get_episode_metadata")
            except AttributeError:
                get_metadata = None
    if not callable(get_metadata):
        raise RuntimeError(
            "This policy requires an environment exposing get_episode_metadata()."
        )
    metadata = get_metadata()
    if not isinstance(metadata, Mapping):
        raise TypeError("get_episode_metadata() must return a mapping.")
    for session in aware_sessions:
        session.bind_environment(metadata)


class SemanticPolicyRunner:
    """Encode model-independent semantic actions through the environment API."""

    def __init__(
        self,
        env: Any,
        session: PolicySession,
        *,
        action_bounds_policy: str = "error",
    ) -> None:
        if action_bounds_policy not in ("error", "clip"):
            raise ValueError("action_bounds_policy must be 'error' or 'clip'.")
        self.env = env
        self.session = session
        self.action_bounds_policy = action_bounds_policy
        _bind_environment(env, (session,))

    def reset(self) -> None:
        self.session.reset()

    def act(self, model_input: ModelInput) -> Any:
        semantic_action = self.session.act(model_input)
        encode_action = getattr(self.env, "encode_action", None)
        if not callable(encode_action):
            get_wrapper_attr = getattr(self.env, "get_wrapper_attr", None)
            if callable(get_wrapper_attr):
                encode_action = get_wrapper_attr("encode_action")
        if not callable(encode_action):
            raise RuntimeError(
                "A semantic policy requires an environment exposing encode_action()."
            )
        if self.action_bounds_policy == "error":
            return encode_action(semantic_action)
        return encode_action(semantic_action, bounds_policy="clip")

    def close(self) -> None:
        self.session.close()


def _environment_count(env: Any) -> int:
    get_wrapper_attr = getattr(env, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        try:
            value = get_wrapper_attr("num_envs")
        except AttributeError:
            value = None
    else:
        value = getattr(env, "num_envs", None)
    num_envs = 1 if value is None else int(value)
    if num_envs <= 0:
        raise ValueError(f"Environment num_envs must be positive, got {num_envs}.")
    return num_envs


def _slice_observation(value: Any, env_index: int, num_envs: int) -> Any:
    """Select one environment while preserving the nested observation schema."""

    if isinstance(value, Mapping):
        return {
            key: _slice_observation(item, env_index, num_envs)
            for key, item in value.items()
        }
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == num_envs:
        return value[env_index]
    return value


def _as_numpy(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    """Copy tensor-like values to host memory before NumPy conversion."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _stack_semantic_actions(
    actions: Sequence[Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, np.ndarray]]:
    if not actions:
        raise ValueError("At least one semantic action is required.")
    expected_roles = tuple(actions[0])
    result: dict[str, dict[str, np.ndarray]] = {}
    for role in expected_roles:
        expected_fields = tuple(actions[0][role])
        for env_index, action in enumerate(actions):
            if tuple(action) != expected_roles:
                raise ValueError(
                    "Batched policy sessions returned inconsistent manipulator "
                    f"roles at environment {env_index}."
                )
            if tuple(action[role]) != expected_fields:
                raise ValueError(
                    "Batched policy sessions returned inconsistent action fields "
                    f"for role {role!r} at environment {env_index}."
                )
        result[role] = {
            field: np.stack(
                [np.asarray(action[role][field]) for action in actions],
                axis=0,
            )
            for field in expected_fields
        }
    return result


class CentralizedBatchPolicyController:
    """Run policy state centrally for independent simulator processes.

    A CPU-vectorized evaluator cannot call ``env.encode_action`` in the main
    process because each SAPIEN environment belongs to its worker.  This
    controller therefore stops at the model-independent ``SemanticAction``
    boundary.  The worker-local environment wrapper performs the final native
    action encoding.
    """

    def __init__(
        self,
        sessions: Sequence[PolicySession],
        *,
        num_envs: int | None = None,
        native_batch: bool = False,
    ) -> None:
        if not sessions:
            raise ValueError("At least one policy session is required.")
        self.sessions = tuple(sessions)
        self.num_envs = len(self.sessions) if num_envs is None else int(num_envs)
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if native_batch:
            if len(self.sessions) != 1:
                raise ValueError("Native batching requires exactly one shared session.")
            if not callable(getattr(self.sessions[0], "act_batch", None)):
                raise TypeError(
                    "Native batch session must provide act_batch(model_inputs)."
                )
            self._native_batch_session: Any | None = self.sessions[0]
        else:
            if len(self.sessions) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} independent sessions, got "
                    f"{len(self.sessions)}."
                )
            self._native_batch_session = None

    @classmethod
    def from_session(
        cls,
        session: PolicySession,
        *,
        num_envs: int,
    ) -> "CentralizedBatchPolicyController":
        """Create a native-batch controller around one shared session."""

        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if not callable(getattr(session, "act_batch", None)):
            raise TypeError("session must provide act_batch(model_inputs).")
        return cls((session,), num_envs=num_envs, native_batch=True)

    def bind_environments(self, metadata: Sequence[Mapping[str, Any]]) -> None:
        """Bind one compatible environment contract to policy state."""

        metadata_rows = tuple(metadata)
        if len(metadata_rows) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} environment metadata rows, got "
                f"{len(metadata_rows)}."
            )
        reference = metadata_rows[0]
        for env_index, candidate in enumerate(metadata_rows[1:], start=1):
            _require_matching_environment_contract(
                reference,
                candidate,
                env_index=env_index,
            )

        if self._native_batch_session is not None:
            binder = getattr(self._native_batch_session, "bind_environment", None)
            if callable(binder):
                binder(reference)
            return
        for session, environment_metadata in zip(
            self.sessions, metadata_rows, strict=True
        ):
            binder = getattr(session, "bind_environment", None)
            if callable(binder):
                binder(environment_metadata)

    def reset(self) -> None:
        if self._native_batch_session is not None:
            self._native_batch_session.reset()
            return
        for session in self.sessions:
            session.reset()

    def act(
        self,
        observation: Mapping[str, Any],
        instructions: Sequence[str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Return one stacked semantic action for every environment slot."""

        instruction_rows = tuple(instructions)
        if len(instruction_rows) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} instructions, got "
                f"{len(instruction_rows)}."
            )
        inputs = tuple(
            build_model_input(
                _slice_observation(observation, env_index, self.num_envs),
                instruction_rows[env_index],
            )
            for env_index in range(self.num_envs)
        )
        if self._native_batch_session is not None:
            actions = tuple(self._native_batch_session.act_batch(inputs))
        else:
            actions = tuple(
                session.act(model_input)
                for session, model_input in zip(self.sessions, inputs, strict=True)
            )
        if len(actions) != self.num_envs:
            raise ValueError(
                f"Batched policy returned {len(actions)} actions for "
                f"{self.num_envs} environments."
            )
        return _stack_semantic_actions(actions)

    def capture_observation_after_action(self) -> bool:
        """Whether the observation produced by the next step must be fresh.

        Sessions without an explicit action-chunk contract retain the original
        behavior and capture every control step.
        """

        if self._native_batch_session is not None:
            required = getattr(
                self._native_batch_session,
                "capture_observation_after_action",
                None,
            )
            return True if not callable(required) else bool(required())
        decisions = []
        for session in self.sessions:
            required = getattr(session, "capture_observation_after_action", None)
            decisions.append(True if not callable(required) else bool(required()))
        return any(decisions)

    def capture_visual_observation_after_action(self) -> bool:
        """Whether the next environment step must render fresh cameras."""

        sessions = (
            (self._native_batch_session,)
            if self._native_batch_session is not None
            else self.sessions
        )
        decisions = []
        for session in sessions:
            required = getattr(session, "capture_visual_observation_after_action", None)
            if not callable(required):
                required = getattr(session, "capture_observation_after_action", None)
            decisions.append(True if not callable(required) else bool(required()))
        return any(decisions)

    def capture_lowdim_observation_after_action(self) -> bool:
        """Whether the next step must refresh state and policy tactile."""

        sessions = (
            (self._native_batch_session,)
            if self._native_batch_session is not None
            else self.sessions
        )
        decisions = []
        for session in sessions:
            required = getattr(session, "capture_lowdim_observation_after_action", None)
            if not callable(required):
                required = getattr(session, "capture_observation_after_action", None)
            decisions.append(True if not callable(required) else bool(required()))
        return any(decisions)

    def requires_control_period_tactile(self) -> bool:
        """Whether sessions consume the substep-averaged tactile observation."""

        sessions = (
            (self._native_batch_session,)
            if self._native_batch_session is not None
            else self.sessions
        )
        requirements = []
        for session in sessions:
            required = getattr(session, "requires_control_period_tactile", None)
            requirements.append(True if not callable(required) else bool(required()))
        return any(requirements)

    def close(self) -> None:
        if self._native_batch_session is not None:
            self._native_batch_session.close()
            return
        for session in self.sessions:
            session.close()


def create_centralized_policy_controller(
    policy_name: str,
    policy_kwargs: Mapping[str, Any],
    *,
    num_envs: int,
) -> CentralizedBatchPolicyController:
    """Create shared batch inference or one isolated session per CPU slot."""

    from dex_benchmark.policy_api.registry import PolicyRegistry

    first_session = PolicyRegistry.create(policy_name, **policy_kwargs)
    if callable(getattr(first_session, "act_batch", None)):
        return CentralizedBatchPolicyController.from_session(
            first_session,
            num_envs=num_envs,
        )

    sessions = [first_session]
    try:
        sessions.extend(
            PolicyRegistry.create(policy_name, **policy_kwargs)
            for _ in range(num_envs - 1)
        )
    except Exception:
        for session in sessions:
            session.close()
        raise
    return CentralizedBatchPolicyController(sessions)


_DYNAMIC_ENVIRONMENT_METADATA_KEYS = frozenset({"generalization"})


def _metadata_values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_metadata_values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _metadata_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if hasattr(left, "shape") or hasattr(right, "shape"):
        try:
            return bool(np.array_equal(_as_numpy(left), _as_numpy(right)))
        except (TypeError, ValueError):
            return False
    return bool(left == right)


def _require_matching_environment_contract(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    env_index: int,
) -> None:
    """Reject mixed embodiments or sensor/control contracts in one batch."""

    reference_contract = {
        key: value
        for key, value in reference.items()
        if key not in _DYNAMIC_ENVIRONMENT_METADATA_KEYS
    }
    candidate_contract = {
        key: value
        for key, value in candidate.items()
        if key not in _DYNAMIC_ENVIRONMENT_METADATA_KEYS
    }
    if not _metadata_values_equal(reference_contract, candidate_contract):
        raise ValueError(
            "CPU parallel environments must expose identical embodiment, "
            f"sensor, and control contracts; environment {env_index} differs."
        )


class BatchedSemanticPolicyRunner:
    """Run one independent policy session for each parallel ManiSkill env."""

    def __init__(
        self,
        env: Any,
        sessions: Sequence[PolicySession],
        *,
        action_bounds_policy: str = "error",
    ) -> None:
        self.env = env
        self.sessions = tuple(sessions)
        self.num_envs = _environment_count(env)
        if len(self.sessions) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} policy sessions, got {len(self.sessions)}."
            )
        if action_bounds_policy not in ("error", "clip"):
            raise ValueError("action_bounds_policy must be 'error' or 'clip'.")
        self.action_bounds_policy = action_bounds_policy
        _bind_environment(env, self.sessions)

    def reset(self) -> None:
        for session in self.sessions:
            session.reset()

    def act(self, model_input: ModelInput) -> Any:
        observation = model_input["observation"]
        instruction = model_input["instruction"]
        semantic_actions = [
            session.act(
                build_model_input(
                    _slice_observation(observation, env_index, self.num_envs),
                    instruction,
                )
            )
            for env_index, session in enumerate(self.sessions)
        ]
        batched_action = _stack_semantic_actions(semantic_actions)
        encode_action = getattr(self.env, "encode_action", None)
        if not callable(encode_action):
            get_wrapper_attr = getattr(self.env, "get_wrapper_attr", None)
            if callable(get_wrapper_attr):
                encode_action = get_wrapper_attr("encode_action")
        if not callable(encode_action):
            raise RuntimeError(
                "A semantic policy requires an environment exposing encode_action()."
            )
        if self.action_bounds_policy == "error":
            return encode_action(batched_action)
        return encode_action(batched_action, bounds_policy="clip")

    def close(self) -> None:
        for session in self.sessions:
            session.close()


class NativeBatchedSemanticPolicyRunner:
    """Use one batch-capable session for all parallel ManiSkill environments."""

    def __init__(
        self,
        env: Any,
        session: Any,
        *,
        action_bounds_policy: str = "error",
    ) -> None:
        self.env = env
        self.session = session
        self.num_envs = _environment_count(env)
        if not callable(getattr(session, "act_batch", None)):
            raise TypeError("session must provide act_batch(model_inputs).")
        if action_bounds_policy not in ("error", "clip"):
            raise ValueError("action_bounds_policy must be 'error' or 'clip'.")
        self.action_bounds_policy = action_bounds_policy
        _bind_environment(env, (session,))

    def reset(self) -> None:
        self.session.reset()

    def act(self, model_input: ModelInput) -> Any:
        observation = model_input["observation"]
        instruction = model_input["instruction"]
        inputs = [
            build_model_input(
                _slice_observation(observation, env_index, self.num_envs),
                instruction,
            )
            for env_index in range(self.num_envs)
        ]
        actions = tuple(self.session.act_batch(inputs))
        if len(actions) != self.num_envs:
            raise ValueError(
                "Batched policy returned "
                f"{len(actions)} actions for {self.num_envs} environments."
            )
        batched_action = _stack_semantic_actions(actions)
        encode_action = getattr(self.env, "encode_action", None)
        if not callable(encode_action):
            get_wrapper_attr = getattr(self.env, "get_wrapper_attr", None)
            if callable(get_wrapper_attr):
                encode_action = get_wrapper_attr("encode_action")
        if not callable(encode_action):
            raise RuntimeError(
                "A semantic policy requires an environment exposing encode_action()."
            )
        if self.action_bounds_policy == "error":
            return encode_action(batched_action)
        return encode_action(batched_action, bounds_policy="clip")

    def close(self) -> None:
        self.session.close()


@dataclass(frozen=True)
class RolloutResult:
    """Minimal outcome for a policy integration rollout."""

    steps: int
    terminated: bool
    truncated: bool
    final_info: dict[str, Any]
    terminated_by_env: tuple[bool, ...]
    truncated_by_env: tuple[bool, ...]


def rollout_episode(
    env: Any,
    policy: Any,
    *,
    max_steps: int,
    on_frame: Callable[[Any, int], None] | None = None,
) -> RolloutResult:
    """Run one bounded episode without coupling to official evaluation metrics.

    ``policy`` only needs ``act(ModelInput)`` and may optionally expose
    ``reset()``. ``on_frame`` receives the environment and completed step index,
    allowing callers to record video without adding media concerns to policies.
    """

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    observation, _ = env.reset()
    reset_policy = getattr(policy, "reset", None)
    if callable(reset_policy):
        reset_policy()
    instruction = instruction_from_env(env)
    num_envs = _environment_count(env)
    terminated_by_env = np.zeros(num_envs, dtype=np.bool_)
    truncated_by_env = np.zeros(num_envs, dtype=np.bool_)
    final_info: dict[str, Any] = {}
    steps = 0
    if on_frame is not None:
        on_frame(env, steps)
    while steps < max_steps:
        model_input = build_model_input(observation, instruction)
        action = policy.act(model_input)
        observation, _, terminated, truncated, final_info = env.step(action)
        step_terminated = _as_numpy(terminated, dtype=np.bool_).reshape(-1)
        step_truncated = _as_numpy(truncated, dtype=np.bool_).reshape(-1)
        if step_terminated.size == 1 and num_envs > 1:
            step_terminated = np.repeat(step_terminated, num_envs)
        if step_truncated.size == 1 and num_envs > 1:
            step_truncated = np.repeat(step_truncated, num_envs)
        if step_terminated.shape != (num_envs,):
            raise ValueError(
                "terminated must provide one value per environment; got "
                f"{step_terminated.shape} for num_envs={num_envs}."
            )
        if step_truncated.shape != (num_envs,):
            raise ValueError(
                "truncated must provide one value per environment; got "
                f"{step_truncated.shape} for num_envs={num_envs}."
            )
        terminated_by_env |= step_terminated
        truncated_by_env |= step_truncated
        steps += 1
        if on_frame is not None:
            on_frame(env, steps)
        if bool((terminated_by_env | truncated_by_env).all()):
            break
    return RolloutResult(
        steps=steps,
        terminated=bool(terminated_by_env.any()),
        truncated=bool(truncated_by_env.any()),
        final_info=dict(final_info),
        terminated_by_env=tuple(bool(value) for value in terminated_by_env),
        truncated_by_env=tuple(bool(value) for value in truncated_by_env),
    )
