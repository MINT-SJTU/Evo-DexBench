from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _per_env_array(value: Any, *, num_envs: int, dtype: Any) -> np.ndarray:
    array = _to_numpy(value).astype(dtype, copy=False)
    if array.ndim == 0:
        return np.repeat(array.reshape(1), num_envs)
    array = array.reshape(-1)
    if array.shape != (num_envs,):
        raise ValueError(
            "Metric values must contain one value per environment; "
            f"expected [{num_envs}], got {list(array.shape)}."
        )
    return array


@dataclass
class OnlineMetricAccumulator:
    """Aggregate official metrics without retaining per-episode records."""

    episode_count: int = 0
    success_count: int = 0
    stage_score_sum: float = 0.0

    def update(self, record: dict[str, Any]) -> None:
        raw_success = _to_numpy(record.get("success", False))
        num_envs = int(raw_success.size) if raw_success.ndim else 1
        success = _per_env_array(
            record.get("success", False), num_envs=num_envs, dtype=np.bool_
        )
        self.episode_count += num_envs
        self.success_count += int(success.sum())
        self.stage_score_sum += float(
            _per_env_array(
                record.get("stage_score", 0.0),
                num_envs=num_envs,
                dtype=np.float64,
            ).sum()
        )
    def result(self) -> dict[str, float]:
        if self.episode_count == 0:
            return {
                "success_rate": 0.0,
                "stage_score": 0.0,
            }
        episode_denominator = float(self.episode_count)
        return {
            "success_rate": self.success_count / episode_denominator,
            "stage_score": self.stage_score_sum / episode_denominator,
        }
