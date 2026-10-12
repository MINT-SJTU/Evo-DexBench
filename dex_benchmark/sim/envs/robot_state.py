from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dex_benchmark.utils.transform import PoseValueLike, TensorLike, pose_value_to_pos_quat


@dataclass(frozen=True)
class FramePose:
    """Pose of one named robot frame in world coordinates."""

    name: str
    pos_world: TensorLike
    quat_world: TensorLike


@dataclass(frozen=True)
class RobotFrameState:
    """Observable robot frames parsed from model-facing proprioception."""

    eef: FramePose
    palm: FramePose


def parse_frame_pose_from_obs(
    manipulator_obs: dict[str, PoseValueLike],
    *,
    obs_key: str,
    frame_name: str,
) -> FramePose:
    """Parse one named frame pose from proprioception observations."""

    pos_world, quat_world = pose_value_to_pos_quat(
        manipulator_obs.get(obs_key),
        name=f"obs.manipulators.{obs_key}",
    )
    return FramePose(name=frame_name, pos_world=pos_world, quat_world=quat_world)


def parse_robot_frame_state(
    obs: dict[str, Any],
    *,
    role: str = "right",
    eef_key: str = "end_effector_pose",
    palm_key: str = "palm_pose",
) -> RobotFrameState:
    """Parse the observable EEF and palm frames from model observations."""

    manipulators = obs.get("manipulators")
    if not isinstance(manipulators, dict):
        raise KeyError("obs['manipulators'] is required.")
    manipulator_obs = manipulators.get(role)
    if not isinstance(manipulator_obs, dict):
        raise KeyError(f"obs['manipulators'][{role!r}] is required.")

    return RobotFrameState(
        eef=parse_frame_pose_from_obs(
            manipulator_obs,
            obs_key=eef_key,
            frame_name="eef",
        ),
        palm=parse_frame_pose_from_obs(
            manipulator_obs,
            obs_key=palm_key,
            frame_name="palm",
        ),
    )
