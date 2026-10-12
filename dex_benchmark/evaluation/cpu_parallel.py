from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import inspect
import multiprocessing as mp
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv
from gymnasium.vector.utils import write_to_shared_memory

from dex_benchmark.evaluation.result import EvaluationResult
from dex_benchmark.evaluation.metrics import OnlineMetricAccumulator

if TYPE_CHECKING:
    from dex_benchmark.policy_api.runner import CentralizedBatchPolicyController


def _wrapper_attr(env: Any, name: str) -> Any:
    getter = getattr(env, "get_wrapper_attr", None)
    if callable(getter):
        return getter(name)
    return getattr(env, name)


def _semantic_action_space(metadata: Mapping[str, Any]) -> gym.spaces.Dict:
    action_spec = metadata.get("benchmark_action_spec")
    if not isinstance(action_spec, Mapping):
        raise ValueError("Environment metadata is missing benchmark_action_spec.")
    manipulators = action_spec.get("manipulators")
    if not isinstance(manipulators, Mapping) or not manipulators:
        raise ValueError("benchmark_action_spec must declare manipulators.")

    role_spaces: dict[str, gym.Space] = {}
    for role, raw_spec in manipulators.items():
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"Invalid action specification for role {role!r}.")
        ee_spec = raw_spec["end_effector_delta_pose"]
        hand_spec = raw_spec["hand_joint_position"]
        rotation_constraint = ee_spec["rotation_constraint"]
        if rotation_constraint["type"] == "scaled_l2_ball":
            rotation_scale = np.asarray(
                rotation_constraint["scale_rad"], dtype=np.float32
            )
            rotation_lower = -rotation_scale
            rotation_upper = rotation_scale
        elif rotation_constraint["type"] == "box":
            rotation_lower = np.asarray(rotation_constraint["lower"], dtype=np.float32)
            rotation_upper = np.asarray(rotation_constraint["upper"], dtype=np.float32)
        else:
            raise ValueError(
                "Unsupported end-effector rotation constraint "
                f"{rotation_constraint['type']!r}."
            )
        ee_lower = np.concatenate(
            (
                np.asarray(ee_spec["translation_lower"], dtype=np.float32),
                rotation_lower,
            )
        )
        ee_upper = np.concatenate(
            (
                np.asarray(ee_spec["translation_upper"], dtype=np.float32),
                rotation_upper,
            )
        )
        role_spaces[str(role)] = gym.spaces.Dict(
            {
                "end_effector_delta_pose": gym.spaces.Box(
                    low=ee_lower,
                    high=ee_upper,
                    dtype=np.float32,
                ),
                "hand_joint_position": gym.spaces.Box(
                    low=np.asarray(hand_spec["lower"], dtype=np.float32),
                    high=np.asarray(hand_spec["upper"], dtype=np.float32),
                    dtype=np.float32,
                ),
            }
        )
    return gym.spaces.Dict(role_spaces)


class CPUSemanticActionEnv(gym.Wrapper):
    """CPU Gym environment accepting benchmark semantic actions.

    The wrapper lives entirely inside one simulator process.  It also latches
    terminal state so a completed slot can safely wait for the other members
    of a synchronous evaluation wave without advancing its physics.
    """

    def __init__(self, env: gym.Env, *, action_bounds_policy: str) -> None:
        super().__init__(env)
        if action_bounds_policy not in ("error", "clip"):
            raise ValueError("action_bounds_policy must be 'error' or 'clip'.")
        self.action_bounds_policy = action_bounds_policy
        self._episode_metadata = dict(_wrapper_attr(env, "get_episode_metadata")())
        self.action_space = _semantic_action_space(self._episode_metadata)
        self._done = False
        self._last_observation: Any = None
        self._last_info: dict[str, Any] = {}
        self._reset_info: dict[str, Any] = {}
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._capture_visual_observation = True
        self._capture_lowdim_observation = True
        self._observation_fresh = True
        self._observation_update: str | None = "full"

    @property
    def observation_fresh(self) -> bool:
        return self._observation_fresh

    @property
    def observation_update(self) -> str | None:
        """Shared-memory subtree changed by the most recent step."""

        return self._observation_update

    def set_capture_observation(self, enabled: bool) -> None:
        """Compatibility switch for callers that treat observation atomically."""

        self.set_observation_capture(visual=enabled, lowdim=enabled)

    def set_observation_capture(self, *, visual: bool, lowdim: bool) -> None:
        self._capture_visual_observation = bool(visual)
        self._capture_lowdim_observation = bool(lowdim)

    def set_policy_tactile_observation_enabled(self, enabled: bool) -> None:
        setter = _wrapper_attr(self.env, "set_policy_tactile_observation_enabled")
        setter(bool(enabled))

    @property
    def episode_instruction(self) -> str:
        return str(_wrapper_attr(self.env, "current_instruction"))

    @property
    def episode_metadata(self) -> dict[str, Any]:
        return dict(_wrapper_attr(self.env, "get_episode_metadata")())

    def episode_record(self) -> dict[str, Any]:
        generalization = self.episode_metadata.get("generalization", {})
        return {
            "success": self._last_info.get("success", False),
            "episode_steps": self._steps,
            "stage_score": self._last_info.get("stage_score", 0.0),
            "layout_id": self._reset_info.get("layout_id", ""),
            "generalization_axis": self._reset_info.get(
                "generalization_axis", generalization.get("axis")
            ),
            "generalization_seed": self._reset_info.get(
                "generalization_seed", generalization.get("seed")
            ),
            "generalization_profile": self._reset_info.get(
                "generalization_profile", generalization.get("profile")
            ),
            "generalization_attempt": self._reset_info.get(
                "generalization_attempt", generalization.get("attempt", 0)
            ),
            "generalization_sample_index": self._reset_info.get(
                "generalization_sample_index", generalization.get("sample_index")
            ),
            "generalization_values": generalization.get("sampled_values", {}),
            "terminated": self._terminated,
            "truncated": self._truncated,
            "timeout": False,
        }

    def mark_timeout(self, timeout: bool) -> None:
        if timeout and not self._done:
            self._truncated = True
            self._done = True

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._done = False
        self._last_observation = observation
        self._last_info = {}
        self._reset_info = dict(info)
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._capture_visual_observation = True
        self._capture_lowdim_observation = True
        self._observation_fresh = True
        self._observation_update = "full"
        return observation, info

    def step(self, action: Mapping[str, Mapping[str, np.ndarray]]):
        if self._done:
            # The terminal observation is already present in shared memory.
            # Keep the fixed batch slot stable without copying the same three
            # camera frames on every step while other slots finish their wave.
            self._observation_fresh = False
            self._observation_update = None
            return (
                self._last_observation,
                0.0,
                self._terminated,
                self._truncated,
                dict(self._last_info),
            )
        encoder = _wrapper_attr(self.env, "encode_action")
        native_action = encoder(action, bounds_policy=self.action_bounds_policy)
        if self._capture_visual_observation:
            observation, reward, terminated, truncated, info = self.env.step(
                native_action
            )
            self._observation_fresh = True
            self._observation_update = "full"
        elif self._capture_lowdim_observation:
            from mani_skill.utils import common

            lowdim_step = _wrapper_attr(self.env, "step_without_visual_observation")
            lowdim, reward, terminated, truncated, info = lowdim_step(native_action)
            lowdim, reward, terminated, truncated, info = common.unbatch(
                common.to_numpy(lowdim),
                common.to_numpy(reward),
                common.to_numpy(terminated),
                common.to_numpy(truncated),
                common.to_numpy(info),
            )
            observation = dict(self._last_observation)
            observation["manipulators"] = lowdim["manipulators"]
            self._observation_fresh = True
            self._observation_update = "manipulators"
        else:
            from mani_skill.utils import common

            fast_step = _wrapper_attr(self.env, "step_without_observation")
            _, reward, terminated, truncated, info = fast_step(native_action)
            reward, terminated, truncated, info = common.unbatch(
                common.to_numpy(reward),
                common.to_numpy(terminated),
                common.to_numpy(truncated),
                common.to_numpy(info),
            )
            observation = self._last_observation
            self._observation_fresh = False
            self._observation_update = None
        self._steps += 1
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)
        self._done = self._terminated or self._truncated
        self._last_observation = observation
        self._last_info = dict(info)
        return observation, reward, terminated, truncated, info


def _build_cpu_semantic_env(
    task_id: str,
    env_kwargs: Mapping[str, Any],
    *,
    action_bounds_policy: str,
) -> CPUSemanticActionEnv:
    from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper

    from dex_benchmark.sim.tasks import make_task_env

    kwargs = dict(env_kwargs)
    kwargs["num_envs"] = 1
    kwargs["sim_backend"] = "cpu"
    return CPUSemanticActionEnv(
        CPUGymWrapper(make_task_env(task_id, **kwargs)),
        action_bounds_policy=action_bounds_policy,
    )


class ReconfigurableCPUSemanticEnv(gym.Env):
    """Worker-owned environment that can replace its concrete SAPIEN scene."""

    def __init__(
        self,
        *,
        task_id: str,
        env_kwargs: Mapping[str, Any],
        slot_index: int,
        action_bounds_policy: str,
    ) -> None:
        self.task_id = task_id
        self.slot_index = slot_index
        self.action_bounds_policy = action_bounds_policy
        self._env_kwargs = dict(env_kwargs)
        self._env = _build_cpu_semantic_env(
            task_id,
            self._env_kwargs,
            action_bounds_policy=action_bounds_policy,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.metadata = dict(getattr(self._env, "metadata", {}))
        self._prepared: tuple[int | None, Any, dict[str, Any]] | None = None

    @property
    def episode_instruction(self) -> str:
        return self._env.episode_instruction

    @property
    def episode_metadata(self) -> dict[str, Any]:
        return self._env.episode_metadata

    def episode_record(self) -> dict[str, Any]:
        return self._env.episode_record()

    def mark_timeout(self, timeouts: Sequence[bool]) -> None:
        self._env.mark_timeout(bool(timeouts[self.slot_index]))

    @property
    def observation_fresh(self) -> bool:
        return self._env.observation_fresh

    @property
    def observation_update(self) -> str | None:
        return self._env.observation_update

    def set_capture_observation(self, enabled: bool) -> None:
        self._env.set_capture_observation(enabled)

    def set_observation_capture(self, *, visual: bool, lowdim: bool) -> None:
        self._env.set_observation_capture(visual=visual, lowdim=lowdim)

    def set_policy_tactile_observation_enabled(self, enabled: bool) -> None:
        self._env.set_policy_tactile_observation_enabled(enabled)

    def set_tactile_control_accumulation_enabled(self, enabled: bool) -> None:
        setter = _wrapper_attr(self._env, "set_tactile_control_accumulation_enabled")
        setter(bool(enabled))

    def reconfigure(self, configurations: Sequence[Mapping[str, Any]]) -> bool:
        requested = dict(configurations[self.slot_index])
        if requested == self._env_kwargs:
            return False
        old_axis = self._env_kwargs.get("generalization_axis")
        new_axis = requested.get("generalization_axis")
        non_generalization_keys = (set(self._env_kwargs) | set(requested)) - {
            "generalization_axis",
            "generalization_profile",
            "generalization_seed",
            "generalization_attempt",
            "generalization_sample_index",
        }
        reset_only = (
            old_axis in (None, "spatial")
            and new_axis in (None, "spatial")
            and all(
                self._env_kwargs.get(key) == requested.get(key)
                for key in non_generalization_keys
            )
        )
        if reset_only:
            configure = _wrapper_attr(self._env.env, "configure_reset_generalization")
            configure(
                axis=new_axis,
                profile=requested.get("generalization_profile"),
                seed=requested.get("generalization_seed"),
                attempt=int(requested.get("generalization_attempt", 0)),
                sample_index=requested.get("generalization_sample_index"),
            )
            self._env_kwargs = requested
            return False

        replacement = _build_cpu_semantic_env(
            self.task_id,
            requested,
            action_bounds_policy=self.action_bounds_policy,
        )
        if replacement.observation_space != self.observation_space:
            replacement.close()
            raise ValueError("Reconfigured environment changed observation space.")
        if replacement.action_space != self.action_space:
            replacement.close()
            raise ValueError("Reconfigured environment changed action space.")
        previous = self._env
        self._env = replacement
        self._env_kwargs = requested
        previous.close()
        return True

    def prepare(
        self,
        configurations: Sequence[Mapping[str, Any]],
        seeds: Sequence[int | None],
    ) -> dict[str, Any]:
        """Reconfigure and validate reset without letting a rejected sample kill the worker."""

        seed = seeds[self.slot_index]
        requested = dict(configurations[self.slot_index])
        if (
            self._prepared is not None
            and self._prepared[0] == seed
            and requested == self._env_kwargs
        ):
            return {"valid": True, "rebuilt": False}
        from dex_benchmark.sim.generalization import InvalidGeneralizationSample

        self._prepared = None
        try:
            rebuilt = self.reconfigure(configurations)
            observation, info = self._env.reset(seed=seed)
        except InvalidGeneralizationSample as exc:
            return {
                "valid": False,
                "rebuilt": False,
                "message": str(exc),
                "invalid_env_indices": list(exc.invalid_env_indices),
            }
        self._prepared = (seed, observation, dict(info))
        return {"valid": True, "rebuilt": rebuilt}

    def reset(self, **kwargs: Any):
        seed = kwargs.get("seed")
        if self._prepared is not None and self._prepared[0] == seed:
            _, observation, info = self._prepared
            self._prepared = None
            return observation, info
        return self._env.reset(**kwargs)

    def step(self, action: Any):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()


@dataclass(frozen=True)
class CPUEnvFactory:
    task_id: str
    env_kwargs: Mapping[str, Any]
    action_bounds_policy: str = "error"
    parent_observation_space: gym.Space | None = None
    parent_action_space: gym.Space | None = None
    parent_metadata: Mapping[str, Any] | None = None
    parent_pid: int | None = None
    slot_index: int = 0

    def __call__(self) -> gym.Env:
        # AsyncVectorEnv constructs one dummy environment in the parent solely
        # to discover spaces. A forkserver probe supplies those spaces so the
        # parent gets a lightweight placeholder and never initializes SAPIEN,
        # Vulkan, or PhysX.
        if (
            self.parent_pid == os.getpid()
            and self.parent_observation_space is not None
            and self.parent_action_space is not None
        ):
            return _SpaceOnlyEnv(
                observation_space=self.parent_observation_space,
                action_space=self.parent_action_space,
                metadata=self.parent_metadata or {},
            )

        # Limit only tensor-library helper pools.  PhysX scene worker settings
        # stay untouched so CPU evaluation uses the collection-time physics.
        if self.parent_pid != os.getpid():
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            try:
                import torch

                torch.set_num_threads(1)
            except (ImportError, RuntimeError):
                pass

        return ReconfigurableCPUSemanticEnv(
            task_id=self.task_id,
            env_kwargs=self.env_kwargs,
            slot_index=self.slot_index,
            action_bounds_policy=self.action_bounds_policy,
        )


class _SpaceOnlyEnv(gym.Env):
    """Parent-process placeholder used only by AsyncVectorEnv.__init__."""

    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        metadata: Mapping[str, Any],
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.metadata = dict(metadata)


def _probe_environment_spaces(factory: CPUEnvFactory, pipe: Any) -> None:
    env: gym.Env | None = None
    try:
        env = factory()
        pipe.send(
            (
                True,
                env.observation_space,
                env.action_space,
                dict(getattr(env, "metadata", {})),
            )
        )
    except BaseException as exc:
        pipe.send((False, type(exc).__name__, str(exc), traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        pipe.close()


def _factory_with_parent_spaces(
    factory: CPUEnvFactory,
    *,
    context: str,
    timeout: float,
) -> CPUEnvFactory:
    """Discover Gym spaces in a disposable child, never in the main process."""

    if timeout <= 0:
        raise ValueError("startup timeout must be positive.")
    multiprocessing_context = mp.get_context(context)
    parent_pipe, child_pipe = multiprocessing_context.Pipe(duplex=False)
    process = multiprocessing_context.Process(
        target=_probe_environment_spaces,
        name="EvoDexCPUEnvSpaceProbe",
        args=(factory, child_pipe),
    )
    process.start()
    child_pipe.close()
    try:
        if not parent_pipe.poll(timeout):
            process.terminate()
            raise TimeoutError(
                f"CPU environment space probe timed out after {timeout:g} seconds."
            )
        result = parent_pipe.recv()
    finally:
        parent_pipe.close()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join()
    if not result[0]:
        _, error_type, message, worker_traceback = result
        raise RuntimeError(
            "CPU environment failed during worker-only startup probe: "
            f"{error_type}: {message}\n{worker_traceback}"
        )
    _, observation_space, action_space, metadata = result
    return CPUEnvFactory(
        task_id=factory.task_id,
        env_kwargs=dict(factory.env_kwargs),
        action_bounds_policy=factory.action_bounds_policy,
        parent_observation_space=observation_space,
        parent_action_space=action_space,
        parent_metadata=metadata,
        parent_pid=os.getpid(),
        slot_index=factory.slot_index,
    )


def no_autoreset_worker(
    index: int,
    env_fn: Callable[[], gym.Env],
    pipe: Any,
    parent_pipe: Any,
    shared_memory: Any,
    error_queue: Any,
    autoreset_mode: Any | None = None,
) -> None:
    """Worker preserving terminal state across Gymnasium vector API versions.

    Gymnasium 0.29 passes six arguments, while newer releases append an
    ``autoreset_mode`` argument.  This worker deliberately ignores that value:
    its defining behavior is to never reset in response to ``step``.
    """

    env: gym.Env | None = None
    try:
        env = env_fn()
        observation_space = env.observation_space
        parent_pipe.close()
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation, info = env.reset(**data)
                if shared_memory is not None:
                    write_to_shared_memory(
                        observation_space, index, observation, shared_memory
                    )
                    observation = None
                # AsyncVectorEnv infers an array dtype independently for every
                # info key from the first worker.  Protocol waves can mix
                # nominal and generalized episodes, so optional reset metadata
                # such as ``generalization_seed`` is legitimately an integer
                # in one slot and ``None`` in another.  Gymnasium 0.29 then
                # tries to assign None into an integer array and aborts reset.
                #
                # The parent never consumes vectorized reset metadata.  The
                # worker-owned CPUSemanticActionEnv already retains the complete
                # reset info and exposes it through episode_record(), where it
                # is logged without lossy vector coercion.  Return an empty
                # mapping across this redundant boundary to keep mixed protocol
                # waves type-safe while preserving all episode metadata.
                pipe.send(((observation, {}), True))
            elif command == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                observation_update = getattr(env, "observation_update", "full")
                if shared_memory is not None and observation_update == "full":
                    write_to_shared_memory(
                        observation_space, index, observation, shared_memory
                    )
                    observation = None
                elif shared_memory is not None and observation_update == "manipulators":
                    write_to_shared_memory(
                        observation_space["manipulators"],
                        index,
                        observation["manipulators"],
                        shared_memory["manipulators"],
                    )
                    observation = None
                pipe.send(((observation, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in {"reset", "step", "seed", "close"}:
                    raise ValueError(f"Use the vector API instead of calling {name!r}.")
                value = getattr(env, name)
                result = value(*args, **kwargs) if callable(value) else value
                pipe.send((result, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                if len(data) == 3:
                    _, requested_observation_space, requested_action_space = data
                else:
                    requested_observation_space, requested_action_space = data
                pipe.send(
                    (
                        (
                            requested_observation_space == observation_space,
                            requested_action_space == env.action_space,
                        ),
                        True,
                    )
                )
            else:
                raise RuntimeError(f"Unknown vector worker command {command!r}.")
    except (KeyboardInterrupt, Exception):
        error_type, error_message, _ = sys.exc_info()
        if autoreset_mode is None:
            error_queue.put((index, error_type, error_message))
        else:
            error_queue.put((index, error_type, error_message, traceback.format_exc()))
        try:
            pipe.send((None, False))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if env is not None:
            env.close()


class CPUParallelEnv:
    """Multiprocess vector of independent single-environment CPU scenes."""

    def __init__(
        self,
        *,
        task_id: str,
        env_kwargs: Mapping[str, Any],
        num_envs: int,
        action_bounds_policy: str = "error",
        context: str = "forkserver",
        startup_timeout: float = 600.0,
        worker_timeout: float | None = 300.0,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if worker_timeout is not None and worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive or None.")
        base_factory = CPUEnvFactory(
            task_id,
            dict(env_kwargs),
            action_bounds_policy,
        )
        has_disabled_autoreset = (
            "autoreset_mode" in inspect.signature(AsyncVectorEnv).parameters
        )
        base_factory = _factory_with_parent_spaces(
            base_factory,
            context=context,
            timeout=startup_timeout,
        )
        factories = tuple(
            CPUEnvFactory(
                task_id=base_factory.task_id,
                env_kwargs=dict(base_factory.env_kwargs),
                action_bounds_policy=base_factory.action_bounds_policy,
                parent_observation_space=base_factory.parent_observation_space,
                parent_action_space=base_factory.parent_action_space,
                parent_metadata=base_factory.parent_metadata,
                parent_pid=base_factory.parent_pid,
                slot_index=slot_index,
            )
            for slot_index in range(num_envs)
        )
        self.num_envs = num_envs
        self.worker_timeout = worker_timeout
        vector_kwargs: dict[str, Any] = {
            "shared_memory": True,
            "copy": False,
            "context": context,
            # Keep the custom worker on every Gymnasium version: besides
            # disabling autoreset it can update only the low-dimensional
            # shared-memory subtree when cameras are intentionally stale.
            "worker": no_autoreset_worker,
        }
        if has_disabled_autoreset:
            from gymnasium.vector import AutoresetMode

            vector_kwargs["autoreset_mode"] = AutoresetMode.DISABLED
        self._env = AsyncVectorEnv(factories, **vector_kwargs)

    def reset(
        self,
        *,
        seeds: Sequence[int | None],
        options: dict[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], dict[str, Any]]:
        seed_rows = list(seeds)
        if len(seed_rows) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} reset seeds, got {len(seed_rows)}."
            )
        self._env.reset_async(seed=seed_rows, options=options)
        return self._env.reset_wait(timeout=self.worker_timeout)

    def step(self, semantic_actions: Mapping[str, Any]):
        self._env.step_async(semantic_actions)
        return self._env.step_wait(timeout=self.worker_timeout)

    def reconfigure(
        self, configurations: Sequence[Mapping[str, Any]]
    ) -> tuple[bool, ...]:
        rows = tuple(dict(row) for row in configurations)
        if len(rows) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} environment configurations, got {len(rows)}."
            )
        self._env.call_async("reconfigure", rows)
        return tuple(
            bool(value) for value in self._env.call_wait(timeout=self.worker_timeout)
        )

    def prepare(
        self,
        configurations: Sequence[Mapping[str, Any]],
        seeds: Sequence[int | None],
    ) -> tuple[dict[str, Any], ...]:
        rows = tuple(dict(row) for row in configurations)
        seed_rows = tuple(seeds)
        if len(rows) != self.num_envs or len(seed_rows) != self.num_envs:
            raise ValueError(
                "prepare requires one configuration and seed per environment."
            )
        self._env.call_async("prepare", rows, seed_rows)
        return tuple(
            dict(value) for value in self._env.call_wait(timeout=self.worker_timeout)
        )

    def mark_timeouts(self, timeouts: Sequence[bool]) -> None:
        rows = tuple(bool(value) for value in timeouts)
        if len(rows) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} timeout flags, got {len(rows)}."
            )
        self._env.call_async("mark_timeout", rows)
        self._env.call_wait(timeout=self.worker_timeout)

    def set_capture_observation(self, enabled: bool) -> None:
        self._env.call_async("set_capture_observation", bool(enabled))
        self._env.call_wait(timeout=self.worker_timeout)

    def set_observation_capture(self, *, visual: bool, lowdim: bool) -> None:
        self._env.call_async(
            "set_observation_capture",
            visual=bool(visual),
            lowdim=bool(lowdim),
        )
        self._env.call_wait(timeout=self.worker_timeout)

    def set_tactile_control_accumulation_enabled(self, enabled: bool) -> None:
        self._env.call_async("set_tactile_control_accumulation_enabled", bool(enabled))
        self._env.call_wait(timeout=self.worker_timeout)

    def set_policy_tactile_observation_enabled(self, enabled: bool) -> None:
        self._env.call_async("set_policy_tactile_observation_enabled", bool(enabled))
        self._env.call_wait(timeout=self.worker_timeout)

    def _call(self, name: str) -> tuple[Any, ...]:
        self._env.call_async(name)
        return tuple(self._env.call_wait(timeout=self.worker_timeout))

    def instructions(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._call("episode_instruction"))

    def metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value) for value in self._call("episode_metadata"))

    def episode_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(value) for value in self._call("episode_record"))

    def render(self) -> tuple[np.ndarray, ...]:
        return tuple(np.asarray(frame) for frame in self._call("render"))

    def close(self) -> None:
        # Gymnasium escalates to process termination automatically if the
        # graceful close does not complete before the timeout.
        self._env.close(timeout=10)


@dataclass(frozen=True)
class CPUParallelWaveResult:
    records: tuple[dict[str, Any], ...]
    steps: int
    terminated_by_env: tuple[bool, ...]
    truncated_by_env: tuple[bool, ...]


FrameCallback = Callable[[Sequence[np.ndarray], int], None]


def run_cpu_parallel_wave(
    env: CPUParallelEnv,
    policy: CentralizedBatchPolicyController,
    *,
    seeds: Sequence[int | None],
    active_count: int,
    max_steps: int,
    on_frame: FrameCallback | None = None,
    allow_observation_elision: bool = True,
) -> CPUParallelWaveResult:
    """Run one synchronous wave while keeping every policy batch slot stable."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    if not 1 <= active_count <= env.num_envs:
        raise ValueError("active_count must be between one and num_envs.")
    requires_tactile = getattr(policy, "requires_control_period_tactile", None)
    tactile_required = (
        True if not callable(requires_tactile) else bool(requires_tactile())
    )
    tactile_setter = getattr(env, "set_tactile_control_accumulation_enabled", None)
    if callable(tactile_setter):
        tactile_setter(tactile_required)
    policy_tactile_setter = getattr(env, "set_policy_tactile_observation_enabled", None)
    if callable(policy_tactile_setter):
        policy_tactile_setter(tactile_required)
    observation, _ = env.reset(seeds=seeds)
    policy.bind_environments(env.metadata())
    policy.reset()
    instructions = env.instructions()
    terminated = np.zeros(env.num_envs, dtype=np.bool_)
    truncated = np.zeros(env.num_envs, dtype=np.bool_)
    steps = 0
    if on_frame is not None:
        on_frame(env.render(), steps)

    active_slice = slice(0, active_count)
    capture_visual = True
    capture_lowdim = True
    while steps < max_steps and not bool(
        np.logical_or(terminated, truncated)[active_slice].all()
    ):
        semantic_actions = policy.act(observation, instructions)
        visual_required = getattr(
            policy, "capture_visual_observation_after_action", None
        )
        lowdim_required = getattr(
            policy, "capture_lowdim_observation_after_action", None
        )
        legacy_required = getattr(policy, "capture_observation_after_action", None)
        if not allow_observation_elision:
            next_visual = next_lowdim = True
        else:
            next_visual = bool(
                visual_required()
                if callable(visual_required)
                else legacy_required() if callable(legacy_required) else True
            )
            next_lowdim = bool(
                lowdim_required()
                if callable(lowdim_required)
                else legacy_required() if callable(legacy_required) else True
            )
        if (next_visual, next_lowdim) != (capture_visual, capture_lowdim):
            capture_setter = getattr(env, "set_observation_capture", None)
            if callable(capture_setter):
                capture_setter(visual=next_visual, lowdim=next_lowdim)
            else:
                # Old/fake environments can only capture an atomic observation.
                env.set_capture_observation(next_visual or next_lowdim)
            capture_visual, capture_lowdim = next_visual, next_lowdim
        observation, _, step_terminated, step_truncated, _ = env.step(semantic_actions)
        terminated |= np.asarray(step_terminated, dtype=np.bool_)
        truncated |= np.asarray(step_truncated, dtype=np.bool_)
        steps += 1
        if on_frame is not None:
            on_frame(env.render(), steps)

    done = np.logical_or(terminated, truncated)
    if steps >= max_steps and not bool(done[active_slice].all()):
        timeout_flags = np.zeros(env.num_envs, dtype=np.bool_)
        timeout_flags[:active_count] = ~done[:active_count]
        env.mark_timeouts(timeout_flags)
    records = list(env.episode_records()[:active_count])
    for index, record in enumerate(records):
        record["timeout"] = bool(steps >= max_steps and not done[index])
    return CPUParallelWaveResult(
        records=tuple(records),
        steps=steps,
        terminated_by_env=tuple(bool(value) for value in terminated[:active_count]),
        truncated_by_env=tuple(bool(value) for value in truncated[:active_count]),
    )


class CPUParallelEvaluator:
    """Evaluate complete synchronous waves and aggregate official metrics."""

    def __init__(
        self,
        env: CPUParallelEnv,
        policy: CentralizedBatchPolicyController,
        *,
        episodes: int,
        max_steps: int,
        seed: int = 0,
    ) -> None:
        if episodes <= 0:
            raise ValueError("episodes must be positive.")
        self.env = env
        self.policy = policy
        self.episodes = episodes
        self.max_steps = max_steps
        self.seed = seed

    def run(
        self,
        *,
        on_wave_start: Callable[[Sequence[int]], FrameCallback | None] | None = None,
        on_wave_end: Callable[[], None] | None = None,
    ) -> EvaluationResult:
        accumulator = OnlineMetricAccumulator()
        last_record: dict[str, Any] = {}
        for start in range(0, self.episodes, self.env.num_envs):
            episode_ids = tuple(
                range(start, min(start + self.env.num_envs, self.episodes))
            )
            active_count = len(episode_ids)
            seeds: list[int | None] = [
                self.seed + episode_id for episode_id in episode_ids
            ]
            seeds.extend(
                self.seed + self.episodes + index
                for index in range(self.env.num_envs - active_count)
            )
            callback = on_wave_start(episode_ids) if on_wave_start else None
            try:
                result = run_cpu_parallel_wave(
                    self.env,
                    self.policy,
                    seeds=seeds,
                    active_count=active_count,
                    max_steps=self.max_steps,
                    on_frame=callback,
                )
            finally:
                if on_wave_end is not None:
                    on_wave_end()
            for record in result.records:
                accumulator.update(record)
                last_record = record
        return EvaluationResult(
            metrics=accumulator.result(),
            episodes=accumulator.episode_count,
            layout_id=str(last_record.get("layout_id", "")),
            generalization_axis=last_record.get("generalization_axis"),
            generalization_seed=last_record.get("generalization_seed"),
            generalization_profile=last_record.get("generalization_profile"),
        )
