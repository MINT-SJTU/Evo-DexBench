from __future__ import annotations

from pathlib import Path

from dex_benchmark.resources import task_config_dir
from dex_benchmark.sim.tasks.attributes import (
    AffordanceFamily,
    LayoutId,
    RobotSetup,
    SceneClass,
    TaskSplit,
)
from dex_benchmark.sim.tasks.registry import TaskRegistry, TaskSettings


class TaskCatalog:
    """Discover task metadata without importing simulator-backed task modules."""

    @staticmethod
    def _config_dir() -> Path:
        return task_config_dir()

    @classmethod
    def task_ids(cls) -> tuple[str, ...]:
        return tuple(
            f"{config_path.stem}_v1"
            for config_path in sorted(cls._config_dir().glob("*.yaml"))
        )

    @classmethod
    def get(cls, task_id: str) -> TaskSettings:
        if task_id not in cls.task_ids():
            known_ids = ", ".join(cls.task_ids()) or "<none>"
            raise ValueError(
                f"Task {task_id!r} is not present in the task catalog. "
                f"Known task IDs: {known_ids}."
            )
        return TaskRegistry.load(task_id)

    @classmethod
    def list_tasks(
        cls,
        affordance_family: AffordanceFamily | None = None,
        task_split: TaskSplit | None = None,
        robot_setup: RobotSetup | None = None,
        scene_class: SceneClass | None = None,
        layout_id: LayoutId | None = None,
    ) -> list[TaskSettings]:
        tasks = [cls.get(task_id) for task_id in cls.task_ids()]
        if affordance_family is not None:
            tasks = [
                task
                for task in tasks
                if task.taxonomy.affordance_family is affordance_family
            ]
        if task_split is not None:
            tasks = [
                task for task in tasks if task.taxonomy.task_split is task_split
            ]
        if robot_setup is not None:
            tasks = [
                task for task in tasks if task.taxonomy.robot_setup is robot_setup
            ]
        if scene_class is not None:
            tasks = [
                task for task in tasks if task.taxonomy.scene_class is scene_class
            ]
        if layout_id is not None:
            tasks = [task for task in tasks if layout_id in task.supported_layouts]
        return tasks
