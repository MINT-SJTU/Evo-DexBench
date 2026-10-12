"""Result types shared by the formal evaluation backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, float]
    episodes: int
    layout_id: str
    generalization_axis: str | None
    generalization_seed: int | None
    generalization_profile: str | None = None
