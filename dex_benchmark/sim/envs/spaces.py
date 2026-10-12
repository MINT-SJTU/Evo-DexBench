from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class ControlMode(str, Enum):
    PD_JOINT_POS = "pd_joint_pos"
    PD_JOINT_VEL = "pd_joint_vel"
    PD_JOINT_DELTA_POS = "pd_joint_delta_pos"
    PD_EE_POSE = "pd_ee_pose"
    PD_EE_DELTA_POSE = "pd_ee_delta_pose"
    ARM_PD_JOINT_POS_HAND_PD_JOINT_POS = (
        "arm_pd_joint_pos_hand_pd_joint_pos"
    )
    ARM_PD_JOINT_POS_HAND_PD_JOINT_DELTA_POS = (
        "arm_pd_joint_pos_hand_pd_joint_delta_pos"
    )
    ARM_PD_JOINT_DELTA_POS_HAND_PD_JOINT_POS = (
        "arm_pd_joint_delta_pos_hand_pd_joint_pos"
    )
    ARM_PD_JOINT_DELTA_POS_HAND_PD_JOINT_DELTA_POS = (
        "arm_pd_joint_delta_pos_hand_pd_joint_delta_pos"
    )
    ARM_PD_EE_DELTA_POSE_HAND_PD_JOINT_POS = (
        "arm_pd_ee_delta_pose_hand_pd_joint_pos"
    )
    ARM_PD_EE_DELTA_POSE_HAND_PD_JOINT_DELTA_POS = (
        "arm_pd_ee_delta_pose_hand_pd_joint_delta_pos"
    )
    ARM_PD_EE_POSE_HAND_PD_JOINT_POS = (
        "arm_pd_ee_pose_hand_pd_joint_pos"
    )
    ARM_PD_EE_POSE_HAND_PD_JOINT_DELTA_POS = (
        "arm_pd_ee_pose_hand_pd_joint_delta_pos"
    )


@dataclass(frozen=True)
class ActionSpace:
    control_mode: ControlMode
    action_dim: int
    action_min: np.ndarray
    action_max: np.ndarray

    def normalize(self, action: np.ndarray) -> np.ndarray:
        return 2.0 * (action - self.action_min) / (self.action_max - self.action_min) - 1.0

    def denormalize(self, action: np.ndarray) -> np.ndarray:
        return (action + 1.0) * 0.5 * (self.action_max - self.action_min) + self.action_min
