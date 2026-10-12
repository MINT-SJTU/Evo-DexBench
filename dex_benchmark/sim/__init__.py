"""Simulation environments, tasks, scenes, robots, sensors, and rendering."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dex_benchmark.sim.tasks import (
        TaskCatalog,
        TaskRegistry,
        TaskSettings,
        make_task_env,
    )

__all__ = ["TaskCatalog", "TaskRegistry", "TaskSettings", "make_task_env"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from dex_benchmark.sim import tasks

        return getattr(tasks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
