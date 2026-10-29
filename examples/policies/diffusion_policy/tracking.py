"""Tracking adapters shared by Diffusion Policy launchers."""

from __future__ import annotations

from collections.abc import Mapping
import time
from typing import Any


def _reported_metrics(data: Any, *, steps_per_epoch: int) -> Any:
    """Expose dataset-pass progress while preserving the official epoch index."""

    if not isinstance(data, Mapping) or "epoch" not in data:
        return data
    metrics = dict(data)
    epoch_index = int(metrics["epoch"])
    metrics["epoch_index"] = epoch_index
    global_step = int(metrics.get("global_step", epoch_index * steps_per_epoch))
    metrics["epoch"] = (global_step + 1) / steps_per_epoch
    return metrics


class _ReportedEpochRun:
    def __init__(self, run: Any, steps_per_epoch: int) -> None:
        self._run = run
        self._steps_per_epoch = steps_per_epoch
        self._last_log_time = time.perf_counter()
        self._last_step: int | None = None

    def log(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        metrics = _reported_metrics(data, steps_per_epoch=self._steps_per_epoch)
        if isinstance(metrics, Mapping) and "global_step" in metrics:
            now = time.perf_counter()
            current_step = int(metrics["global_step"])
            step_delta = (
                1
                if self._last_step is None
                else max(1, current_step - self._last_step)
            )
            metrics = dict(metrics)
            metrics["time_per_step"] = (
                now - self._last_log_time
            ) / step_delta
            self._last_log_time = now
            self._last_step = current_step
        return self._run.log(
            metrics,
            *args,
            **kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)


def install_continuous_epoch_reporting(
    wandb_module: Any, *, steps_per_epoch: int
) -> None:
    """Adapt W&B-compatible reporting without changing workspace scheduling.

    Official Diffusion Policy workspaces use a zero-based epoch counter for
    validation/checkpoint scheduling. Experiment dashboards conventionally use
    continuous dataset-pass progress, so only the tracking payload is
    translated. ``epoch_index`` preserves the official workspace value.
    """

    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive")
    if getattr(wandb_module, "_evodex_continuous_epoch_reporting", False):
        return
    original_init = wandb_module.init

    def init(*args: Any, **kwargs: Any) -> _ReportedEpochRun:
        return _ReportedEpochRun(
            original_init(*args, **kwargs), steps_per_epoch=steps_per_epoch
        )

    wandb_module.init = init
    wandb_module._evodex_continuous_epoch_reporting = True
