from __future__ import annotations

from importlib import import_module
from typing import Any

from dex_benchmark.sim.tasks.bootstrap import load_builtin_task


def make_task_env(task_id: str, **env_kwargs: Any) -> Any:
    """Create a registered benchmark environment from its canonical task id.

    ``task_id`` is the public benchmark identifier. The ManiSkill/Gym
    environment id is derived internally from the registered task settings and
    must not be supplied by callers.
    """

    task_settings = load_builtin_task(task_id)
    robot_modules = {
        "panda_shadow_distributed_tactile_123": "dex_benchmark.sim.robots.panda_shadow",
        "panda_shadow_distributed_tactile_123_left": "dex_benchmark.sim.robots.panda_shadow",
    }
    for robot_uid in task_settings.robot_uids:
        try:
            robot_module = robot_modules[robot_uid]
        except KeyError as exc:
            raise RuntimeError(
                f"Task {task_id!r} references unsupported robot UID {robot_uid!r}."
            ) from exc
        import_module(robot_module)
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional simulator stack.
        raise ImportError(
            "gymnasium and ManiSkill are required to create benchmark tasks."
        ) from exc
    return gym.make(task_settings.env_id, **env_kwargs)
