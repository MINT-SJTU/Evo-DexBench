from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dex_benchmark.sim.layout_types import LayoutId, SceneClass


class AffordanceFamily(str, Enum):
    BUTTON_DRIVEN = "button_driven"
    ROTATIONAL_MANIPULATION = "rotational_manipulation"
    ACCESS_AND_TRANSFER = "access_and_transfer"
    STRUCTURED_CONFIGURATION = "structured_configuration"
    ASSEMBLY = "assembly"
    ARTICULATED_CONFIGURATION_CONTROL = "articulated_configuration_control"


class TaskSplit(str, Enum):
    CORE = "core"
    GEN = "gen"


class RobotSetup(str, Enum):
    SINGLE_ARM_SINGLE_HAND = "single_arm_single_hand"
    DUAL_ARM_DUAL_HAND = "dual_arm_dual_hand"


class StagePersistence(str, Enum):
    """How a reached stage behaves before the following stage is reached."""

    PERMANENT = "permanent"
    UNTIL_NEXT = "until_next"


@dataclass(frozen=True)
class TaskTaxonomy:
    affordance_family: AffordanceFamily
    task_split: TaskSplit
    robot_setup: RobotSetup
    scene_class: SceneClass


@dataclass(frozen=True)
class TaskStageSpec:
    name: str
    score: float
    persistence: StagePersistence = StagePersistence.PERMANENT


@dataclass(frozen=True)
class TaskEvaluationSpec:
    stages: tuple[TaskStageSpec, ...] = ()
    independent_stages: bool = False

    @property
    def stage_milestones(self) -> list[str]:
        return [stage.name for stage in self.stages]

    @property
    def stage_scoring(self) -> dict[str, float]:
        return {stage.name: stage.score for stage in self.stages}
