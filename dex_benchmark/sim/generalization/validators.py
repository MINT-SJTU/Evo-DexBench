from __future__ import annotations

from typing import Any

import numpy as np
import torch


class InvalidGeneralizationSample(RuntimeError):
    def __init__(
        self,
        *,
        task_id: str,
        layout_id: str,
        axis: str,
        seed: int,
        attempt: int,
        invalid_env_indices: list[int],
    ) -> None:
        self.task_id = task_id
        self.layout_id = layout_id
        self.axis = axis
        self.seed = seed
        self.attempt = attempt
        self.invalid_env_indices = tuple(invalid_env_indices)
        super().__init__(
            "Invalid deterministic generalization sample: "
            f"task={task_id}, layout={layout_id}, axis={axis}, seed={seed}, "
            f"attempt={attempt}, invalid_env_indices={invalid_env_indices}."
        )


def as_validity_tensor(
    value: Any,
    *,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=torch.bool)
    else:
        tensor = torch.as_tensor(np.asarray(value), device=device, dtype=torch.bool)
    if tensor.ndim == 0:
        tensor = tensor.repeat(num_envs)
    tensor = tensor.reshape(-1)
    if tensor.shape != (num_envs,):
        raise ValueError(
            "Generalization validation must return one boolean per environment; "
            f"expected [{num_envs}], got {list(tensor.shape)}."
        )
    return tensor
