from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ManipulatorAction:
    """Benchmark action for one arm-hand manipulator."""

    end_effector_delta_pose: Any
    hand_joint_position: Any

    @classmethod
    def from_value(cls, value: Any) -> ManipulatorAction:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "Manipulator action must be a mapping or ManipulatorAction, got "
                f"{type(value).__name__}."
            )
        expected_fields = {
            "end_effector_delta_pose",
            "hand_joint_position",
        }
        unknown_fields = sorted(set(value) - expected_fields)
        missing_fields = sorted(expected_fields - set(value))
        if unknown_fields or missing_fields:
            details: list[str] = []
            if missing_fields:
                details.append("missing " + ", ".join(missing_fields))
            if unknown_fields:
                details.append("unknown " + ", ".join(unknown_fields))
            raise ValueError(
                "Invalid manipulator action fields: " + "; ".join(details) + "."
            )
        return cls(
            end_effector_delta_pose=value["end_effector_delta_pose"],
            hand_joint_position=value["hand_joint_position"],
        )

    def validate(self, *, hand_joint_count: int, role: str) -> None:
        ee_shape = getattr(self.end_effector_delta_pose, "shape", None)
        hand_shape = getattr(self.hand_joint_position, "shape", None)
        if ee_shape is None or len(ee_shape) == 0 or ee_shape[-1] != 6:
            raise ValueError(
                f"{role} end_effector_delta_pose must have shape [..., 6], "
                f"got {ee_shape!r}."
            )
        if (
            hand_shape is None
            or len(hand_shape) == 0
            or hand_shape[-1] != hand_joint_count
        ):
            raise ValueError(
                f"{role} hand_joint_position must have shape "
                f"[..., {hand_joint_count}], got {hand_shape!r}."
            )
        if tuple(ee_shape[:-1]) != tuple(hand_shape[:-1]):
            raise ValueError(
                f"{role} arm and hand action batch dimensions must match, got "
                f"{tuple(ee_shape[:-1])} and {tuple(hand_shape[:-1])}."
            )
        ee_is_tensor = hasattr(self.end_effector_delta_pose, "detach")
        hand_is_tensor = hasattr(self.hand_joint_position, "detach")
        if ee_is_tensor != hand_is_tensor:
            raise TypeError(
                f"{role} arm and hand actions must both be NumPy-like values or "
                "both be torch tensors."
            )
        if ee_is_tensor:
            if not bool(torch.isfinite(self.end_effector_delta_pose).all().item()):
                raise ValueError(
                    f"{role} end_effector_delta_pose must contain only finite values."
                )
            if not bool(torch.isfinite(self.hand_joint_position).all().item()):
                raise ValueError(
                    f"{role} hand_joint_position must contain only finite values."
                )
            return

        try:
            ee_array = np.asarray(self.end_effector_delta_pose, dtype=np.float32)
            hand_array = np.asarray(self.hand_joint_position, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{role} arm and hand actions must be numeric arrays."
            ) from exc
        if not np.isfinite(ee_array).all():
            raise ValueError(
                f"{role} end_effector_delta_pose must contain only finite values."
            )
        if not np.isfinite(hand_array).all():
            raise ValueError(
                f"{role} hand_joint_position must contain only finite values."
            )
