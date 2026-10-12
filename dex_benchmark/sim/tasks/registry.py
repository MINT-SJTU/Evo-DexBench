from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

import yaml

from dex_benchmark.sim.envs.spaces import ControlMode
from dex_benchmark.sim.layout_types import default_layouts_for_scene_class
from dex_benchmark.sim.tasks.attributes import (
    AffordanceFamily,
    LayoutId,
    RobotSetup,
    SceneClass,
    StagePersistence,
    TaskEvaluationSpec,
    TaskSplit,
    TaskStageSpec,
    TaskTaxonomy,
)
from dex_benchmark.sim.tasks.runtime import (
    EmbodimentSpec,
    embodiment_spec_for_robot_setup,
)
from dex_benchmark.resources import task_config_dir


EnumT = TypeVar("EnumT", bound=Enum)

_TASK_FIELDS = {
    "task_id",
    "task_name",
    "instruction",
    "max_steps",
    "evaluation",
    "taxonomy",
}
_REQUIRED_TASK_FIELDS = _TASK_FIELDS
_EVALUATION_FIELDS = {"stages", "independent_stages"}
_TAXONOMY_FIELDS = {
    "affordance_family",
    "task_split",
    "robot_setup",
    "scene_class",
}
_STAGE_FIELDS = {"name", "score", "persistence"}


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"{field_name} must be a mapping, got {type(value).__name__}."
        )
    return value


def _reject_unknown_fields(
    data: dict[str, Any], allowed_fields: set[str], field_name: str
) -> None:
    unknown_fields = sorted(set(data) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            f"{field_name} contains unsupported fields: {', '.join(unknown_fields)}."
        )


@dataclass(frozen=True)
class TaskSettings:
    task_id: str
    task_name: str
    instruction: str
    max_steps: int
    taxonomy: TaskTaxonomy
    evaluation: TaskEvaluationSpec

    @property
    def task_key(self) -> str:
        return self.task_id.removesuffix("_v1")

    @property
    def env_id(self) -> str:
        camel_name = "".join(part.capitalize() for part in self.task_key.split("_"))
        return f"{camel_name}-v1"

    @property
    def description(self) -> str:
        return self.instruction

    @property
    def scene_class(self) -> SceneClass:
        return self.taxonomy.scene_class

    @property
    def embodiment_spec(self) -> EmbodimentSpec:
        return embodiment_spec_for_robot_setup(self.taxonomy.robot_setup)

    @property
    def robot_uids(self) -> tuple[str, ...]:
        return self.embodiment_spec.robot_uids

    @property
    def maniskill_robot_uids(self) -> str | tuple[str, ...]:
        robot_uids = self.embodiment_spec.robot_uids
        return robot_uids[0] if len(robot_uids) == 1 else robot_uids

    @property
    def control_mode(self) -> ControlMode:
        return self.embodiment_spec.control_mode

    @property
    def obs_mode(self) -> str:
        return self.embodiment_spec.obs_mode

    @property
    def required_sensors(self) -> tuple[str, ...]:
        return self.embodiment_spec.required_sensors

    @property
    def supported_layouts(self) -> tuple[LayoutId, ...]:
        return default_layouts_for_scene_class(self.scene_class)


class TaskRegistry:
    _registry: dict[str, TaskSettings] = {}

    @staticmethod
    def _config_dir() -> Path:
        return task_config_dir()

    @staticmethod
    def _parse_enum(
        value: Any,
        enum_type: type[EnumT],
        field_name: str,
    ) -> EnumT:
        if isinstance(value, enum_type):
            return value
        try:
            return enum_type(value)
        except (TypeError, ValueError) as exc:
            if isinstance(value, str):
                try:
                    return enum_type[value.upper()]
                except KeyError:
                    pass
            supported_values = ", ".join(member.value for member in enum_type)
            raise ValueError(
                f"Invalid {field_name} {value!r}. Supported values: {supported_values}."
            ) from exc

    @staticmethod
    def _parse_positive_int(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{field_name} must be a positive integer, got {type(value).__name__}."
            )
        if value <= 0:
            raise ValueError(f"{field_name} must be positive, got {value}.")
        return value

    @staticmethod
    def _parse_float(value: Any, field_name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{field_name} must be a number, got {type(value).__name__}."
            )
        return float(value)

    @classmethod
    def _parse_taxonomy(cls, value: Any) -> TaskTaxonomy:
        data = _require_mapping(value, "taxonomy")
        _reject_unknown_fields(data, _TAXONOMY_FIELDS, "taxonomy")
        missing_fields = sorted(_TAXONOMY_FIELDS - set(data))
        if missing_fields:
            raise ValueError(
                f"taxonomy is missing required fields: {', '.join(missing_fields)}."
            )
        return TaskTaxonomy(
            affordance_family=cls._parse_enum(
                data["affordance_family"], AffordanceFamily, "taxonomy.affordance_family"
            ),
            task_split=cls._parse_enum(
                data["task_split"], TaskSplit, "taxonomy.task_split"
            ),
            robot_setup=cls._parse_enum(
                data["robot_setup"], RobotSetup, "taxonomy.robot_setup"
            ),
            scene_class=cls._parse_enum(
                data["scene_class"], SceneClass, "taxonomy.scene_class"
            ),
        )

    @classmethod
    def _parse_evaluation(cls, value: Any) -> TaskEvaluationSpec:
        data = _require_mapping(value, "evaluation")
        _reject_unknown_fields(data, _EVALUATION_FIELDS, "evaluation")
        independent_stages = data.get("independent_stages", False)
        if not isinstance(independent_stages, bool):
            raise TypeError("evaluation.independent_stages must be a boolean.")
        raw_stages = data.get("stages", [])
        if not isinstance(raw_stages, list):
            raise TypeError(
                f"evaluation.stages must be a list, got {type(raw_stages).__name__}."
            )
        stages: list[TaskStageSpec] = []
        previous_score = -1.0
        known_names: set[str] = set()
        for stage_index, raw_stage in enumerate(raw_stages):
            field_name = f"evaluation.stages[{stage_index}]"
            stage_data = _require_mapping(raw_stage, field_name)
            _reject_unknown_fields(stage_data, _STAGE_FIELDS, field_name)
            stage_name = stage_data.get("name")
            if not isinstance(stage_name, str) or not stage_name:
                raise TypeError(f"{field_name}.name must be a non-empty string.")
            if stage_name in known_names:
                raise ValueError(f"Duplicate stage name {stage_name!r}.")
            stage_score = cls._parse_float(stage_data.get("score"), f"{field_name}.score")
            if not 0.0 <= stage_score <= 1.0:
                raise ValueError(
                    f"{field_name}.score must be in [0, 1], got {stage_score}."
                )
            if stage_score <= previous_score:
                raise ValueError("evaluation stage scores must be strictly increasing.")
            raw_persistence = stage_data.get(
                "persistence",
                StagePersistence.PERMANENT.value,
            )
            try:
                persistence = StagePersistence(raw_persistence)
            except (TypeError, ValueError) as exc:
                known_persistence = ", ".join(
                    value.value for value in StagePersistence
                )
                raise ValueError(
                    f"{field_name}.persistence must be one of "
                    f"{known_persistence}, got {raw_persistence!r}."
                ) from exc
            stages.append(
                TaskStageSpec(
                    name=stage_name,
                    score=stage_score,
                    persistence=persistence,
                )
            )
            known_names.add(stage_name)
            previous_score = stage_score
        if (
            stages
            and stages[-1].persistence is StagePersistence.UNTIL_NEXT
        ):
            raise ValueError(
                "The final evaluation stage cannot use persistence='until_next' "
                "because it has no following stage to commit it."
            )
        if independent_stages and (not stages or stages[-1].score != 1.0):
            raise ValueError(
                "Independent stage scoring requires a final score of 1.0."
            )
        return TaskEvaluationSpec(
            stages=tuple(stages),
            independent_stages=independent_stages,
        )

    @classmethod
    def load(cls, task_id: str) -> TaskSettings:
        config_name = task_id.removesuffix("_v1") + ".yaml"
        config_path = cls._config_dir() / config_name
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Task config for {task_id!r} was not found at {config_path}."
            )
        try:
            raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Failed to parse task config {config_path}: {exc}") from exc
        data = _require_mapping(raw_data, str(config_path))
        _reject_unknown_fields(data, _TASK_FIELDS, str(config_path))
        missing_fields = sorted(_REQUIRED_TASK_FIELDS - set(data))
        if missing_fields:
            raise ValueError(
                f"Task config {config_path} is missing required fields: "
                f"{', '.join(missing_fields)}."
            )
        configured_task_id = data["task_id"]
        if configured_task_id != task_id:
            raise ValueError(
                f"Task config {config_path} declares task_id {configured_task_id!r}, "
                f"expected {task_id!r}."
            )
        instruction = data["instruction"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise TypeError("instruction must be a non-empty string.")
        task_name = data["task_name"]
        if not isinstance(task_name, str) or not task_name.strip():
            raise TypeError("task_name must be a non-empty string.")
        return TaskSettings(
            task_id=task_id,
            task_name=task_name.strip(),
            instruction=instruction.strip(),
            max_steps=cls._parse_positive_int(data["max_steps"], "max_steps"),
            taxonomy=cls._parse_taxonomy(data["taxonomy"]),
            evaluation=cls._parse_evaluation(data["evaluation"]),
        )

    @classmethod
    def register(cls, task_settings: TaskSettings) -> TaskSettings:
        existing = cls._registry.get(task_settings.task_id)
        if existing is not None and existing != task_settings:
            raise ValueError(f"Task {task_settings.task_id!r} is already registered.")
        cls._registry[task_settings.task_id] = task_settings
        return task_settings

    @classmethod
    def get(cls, task_id: str) -> TaskSettings:
        try:
            return cls._registry[task_id]
        except KeyError as exc:
            raise ValueError(f"Task {task_id!r} is not loaded in the runtime registry") from exc

    @classmethod
    def list_loaded_tasks(
        cls,
        affordance_family: AffordanceFamily | None = None,
        task_split: TaskSplit | None = None,
        robot_setup: RobotSetup | None = None,
        scene_class: SceneClass | None = None,
        layout_id: LayoutId | None = None,
    ) -> list[TaskSettings]:
        tasks = list(cls._registry.values())
        if affordance_family is not None:
            tasks = [task for task in tasks if task.taxonomy.affordance_family == affordance_family]
        if task_split is not None:
            tasks = [task for task in tasks if task.taxonomy.task_split == task_split]
        if robot_setup is not None:
            tasks = [task for task in tasks if task.taxonomy.robot_setup == robot_setup]
        if scene_class is not None:
            tasks = [task for task in tasks if task.taxonomy.scene_class == scene_class]
        if layout_id is not None:
            tasks = [task for task in tasks if layout_id in task.supported_layouts]
        return tasks

    @classmethod
    def list_tasks(
        cls,
        affordance_family: AffordanceFamily | None = None,
        task_split: TaskSplit | None = None,
        robot_setup: RobotSetup | None = None,
        scene_class: SceneClass | None = None,
        layout_id: LayoutId | None = None,
    ) -> list[TaskSettings]:
        """List every configured task without importing task implementations."""

        from dex_benchmark.sim.tasks.catalog import TaskCatalog

        return TaskCatalog.list_tasks(
            affordance_family=affordance_family,
            task_split=task_split,
            robot_setup=robot_setup,
            scene_class=scene_class,
            layout_id=layout_id,
        )

    @classmethod
    def create_suite(cls, suite_name: str) -> list[str]:
        normalized_name = suite_name.lower()
        if normalized_name in {"all", "all_registered"}:
            return [task.task_id for task in cls.list_tasks()]
        if normalized_name in {split.value for split in TaskSplit}:
            split = TaskSplit(normalized_name)
            return [task.task_id for task in cls.list_tasks(task_split=split)]
        try:
            family = AffordanceFamily(normalized_name)
        except ValueError:
            return []
        return [
            task.task_id
            for task in cls.list_tasks(affordance_family=family)
        ]

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
