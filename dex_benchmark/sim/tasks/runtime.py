from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from dex_benchmark.sim.envs.spaces import ControlMode
from dex_benchmark.sim.tasks.attributes import RobotSetup


class ManipulatorRole(str, Enum):
    """Semantic manipulator roles used by observations, actions, and datasets."""

    LEFT = "left"
    RIGHT = "right"


class CameraRole(str, Enum):
    """Semantic observation-camera roles exposed by the benchmark."""

    CONTEXT = "context"
    INTERACTION = "interaction"
    WRIST_LEFT = "wrist_left"
    WRIST_RIGHT = "wrist_right"


@dataclass(frozen=True)
class ManipulatorBinding:
    """Bind one semantic manipulator role to one ManiSkill agent UID."""

    role: ManipulatorRole
    agent_uid: str
    manipulator_key: str | None = None

    def __post_init__(self) -> None:
        if not self.agent_uid.strip():
            raise ValueError("Manipulator agent_uid must be non-empty.")
        if self.manipulator_key is not None and not self.manipulator_key.strip():
            raise ValueError("Manipulator manipulator_key must be non-empty.")


@dataclass(frozen=True)
class EmbodimentSpec:
    """Runtime contract for one fixed benchmark robot embodiment."""

    embodiment_id: str
    manipulators: tuple[ManipulatorBinding, ...]
    camera_roles: tuple[CameraRole, ...]
    control_mode: ControlMode
    obs_mode: str
    required_sensors: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.embodiment_id.strip():
            raise ValueError("embodiment_id must be non-empty.")
        if not self.manipulators:
            raise ValueError("An embodiment must contain at least one manipulator.")

        roles = tuple(binding.role for binding in self.manipulators)
        if len(set(roles)) != len(roles):
            raise ValueError(
                f"Embodiment {self.embodiment_id!r} contains duplicate roles."
            )
        bindings_by_uid: dict[str, list[ManipulatorBinding]] = {}
        for binding in self.manipulators:
            bindings_by_uid.setdefault(binding.agent_uid, []).append(binding)
        for agent_uid, bindings in bindings_by_uid.items():
            if len(bindings) == 1:
                continue
            manipulator_keys = tuple(
                binding.manipulator_key for binding in bindings
            )
            if (
                any(key is None for key in manipulator_keys)
                or len(set(manipulator_keys)) != len(manipulator_keys)
            ):
                raise ValueError(
                    f"Embodiment {self.embodiment_id!r} bindings for shared "
                    f"agent UID {agent_uid!r} require a unique manipulator_key."
                )
        if len(set(self.camera_roles)) != len(self.camera_roles):
            raise ValueError(
                f"Embodiment {self.embodiment_id!r} contains duplicate camera roles."
            )

    @property
    def robot_uids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(binding.agent_uid for binding in self.manipulators)
        )

    @property
    def manipulator_roles(self) -> tuple[ManipulatorRole, ...]:
        return tuple(binding.role for binding in self.manipulators)

    def binding_for_role(self, role: ManipulatorRole) -> ManipulatorBinding:
        for binding in self.manipulators:
            if binding.role is role:
                return binding
        raise KeyError(
            f"Embodiment {self.embodiment_id!r} has no {role.value!r} manipulator."
        )

    def physical_agent_index_for_role(self, role: ManipulatorRole) -> int:
        binding = self.binding_for_role(role)
        return self.robot_uids.index(binding.agent_uid)


PANDA_SHADOW_SINGLE_RIGHT = EmbodimentSpec(
    embodiment_id="panda_shadow_single_right",
    manipulators=(
        ManipulatorBinding(
            role=ManipulatorRole.RIGHT,
            agent_uid="panda_shadow_distributed_tactile_123",
        ),
    ),
    camera_roles=(
        CameraRole.CONTEXT,
        CameraRole.INTERACTION,
        CameraRole.WRIST_RIGHT,
    ),
    control_mode=ControlMode.ARM_PD_EE_DELTA_POSE_HAND_PD_JOINT_POS,
    obs_mode="rgb+depth",
    required_sensors=(
        "context_rgb",
        "interaction_rgb",
        "wrist_right_rgb",
        "right_taxel_normal_force",
    ),
)

PANDA_SHADOW_BIMANUAL = EmbodimentSpec(
    embodiment_id="panda_shadow_bimanual",
    # Keep left then right aligned with ManiSkill's MultiAgent construction
    # order and the existing mirrored dual-robot environment.
    manipulators=(
        ManipulatorBinding(
            role=ManipulatorRole.LEFT,
            agent_uid="panda_shadow_distributed_tactile_123_left",
        ),
        ManipulatorBinding(
            role=ManipulatorRole.RIGHT,
            agent_uid="panda_shadow_distributed_tactile_123",
        ),
    ),
    camera_roles=(
        CameraRole.CONTEXT,
        CameraRole.INTERACTION,
        CameraRole.WRIST_LEFT,
        CameraRole.WRIST_RIGHT,
    ),
    control_mode=ControlMode.ARM_PD_EE_DELTA_POSE_HAND_PD_JOINT_POS,
    obs_mode="rgb+depth",
    required_sensors=(
        "context_rgb",
        "interaction_rgb",
        "wrist_left_rgb",
        "wrist_right_rgb",
        "left_taxel_normal_force",
        "right_taxel_normal_force",
    ),
)

_EMBODIMENT_SPECS: dict[RobotSetup, EmbodimentSpec] = {
    RobotSetup.SINGLE_ARM_SINGLE_HAND: PANDA_SHADOW_SINGLE_RIGHT,
    RobotSetup.DUAL_ARM_DUAL_HAND: PANDA_SHADOW_BIMANUAL,
}

_EMBODIMENT_SPECS_BY_ID: dict[str, EmbodimentSpec] = {
    spec.embodiment_id: spec
    for spec in (
        PANDA_SHADOW_SINGLE_RIGHT,
        PANDA_SHADOW_BIMANUAL,
    )
}


def embodiment_spec_for_id(embodiment_id: str) -> EmbodimentSpec:
    try:
        return _EMBODIMENT_SPECS_BY_ID[embodiment_id]
    except KeyError as exc:
        known_ids = ", ".join(sorted(_EMBODIMENT_SPECS_BY_ID))
        raise ValueError(
            f"Unknown embodiment_id {embodiment_id!r}. Known IDs: {known_ids}."
        ) from exc


def embodiment_spec_for_robot_setup(robot_setup: RobotSetup) -> EmbodimentSpec:
    try:
        return _EMBODIMENT_SPECS[robot_setup]
    except KeyError as exc:
        raise ValueError(
            f"No embodiment spec is registered for robot setup {robot_setup.value!r}."
        ) from exc
