"""Evo-DexBench public Python package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dex_benchmark.sim.tasks.catalog import TaskCatalog
    from dex_benchmark.sim.tasks.registry import TaskRegistry, TaskSettings

__all__ = ["TaskCatalog", "TaskRegistry", "TaskSettings"]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name == "TaskCatalog":
        from dex_benchmark.sim.tasks.catalog import TaskCatalog

        return TaskCatalog
    if name in {"TaskRegistry", "TaskSettings"}:
        from dex_benchmark.sim.tasks.registry import TaskRegistry, TaskSettings

        return {"TaskRegistry": TaskRegistry, "TaskSettings": TaskSettings}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
