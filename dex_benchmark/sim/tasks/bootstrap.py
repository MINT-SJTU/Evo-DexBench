from __future__ import annotations

from importlib import import_module
from typing import Iterable

from dex_benchmark.sim.tasks.builtins import BUILTIN_TASK_MODULES
from dex_benchmark.sim.tasks.registry import TaskRegistry, TaskSettings


class TaskLoadError(RuntimeError):
    """Raised when a built-in task cannot be imported and registered."""


def load_builtin_task(task_id: str) -> TaskSettings:
    try:
        module_name = BUILTIN_TASK_MODULES[task_id]
    except KeyError as exc:
        known_ids = ", ".join(BUILTIN_TASK_MODULES)
        raise ValueError(
            f"Task {task_id!r} is not a built-in task. Known task IDs: {known_ids}."
        ) from exc

    try:
        import_module(module_name)
    except Exception as exc:
        raise TaskLoadError(
            f"Failed to load task {task_id!r} from {module_name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    try:
        return TaskRegistry.get(task_id)
    except ValueError as exc:
        raise TaskLoadError(
            f"Task module {module_name!r} imported successfully but did not "
            f"register task {task_id!r}."
        ) from exc


def load_builtin_tasks(task_ids: Iterable[str] | None = None) -> list[TaskSettings]:
    selected_ids = tuple(BUILTIN_TASK_MODULES) if task_ids is None else tuple(task_ids)
    return [load_builtin_task(task_id) for task_id in selected_ids]
