from __future__ import annotations

import inspect
import math
from typing import Any

import torch


def _cosine_with_floor(
    step: int,
    *,
    warmup_steps: int,
    training_steps: int,
    min_lr_ratio: float,
) -> float:
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(
        max(1, training_steps - warmup_steps)
    )
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def install_cosine_min_lr(
    workspace_cls: type,
    *,
    min_lr_ratio: float | None,
) -> None:
    """Optionally add a non-zero floor to the official cosine scheduler."""

    if min_lr_ratio is None:
        return
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1].")
    module = inspect.getmodule(workspace_cls)
    if module is None or not hasattr(module, "get_scheduler"):
        raise RuntimeError(f"Cannot locate scheduler for workspace {workspace_cls}.")
    original = module.get_scheduler
    if getattr(original, "_evodex_cosine_min_lr", False):
        return

    def get_scheduler(
        name: Any,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int | None = None,
        num_training_steps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        scheduler_name = getattr(name, "value", str(name)).lower()
        if scheduler_name != "cosine":
            return original(
                name,
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **kwargs,
            )
        last_epoch = int(kwargs.pop("last_epoch", -1))
        if kwargs:
            raise ValueError(
                "The minimum-LR cosine scheduler does not accept extra arguments: "
                f"{sorted(kwargs)}"
            )
        if num_warmup_steps is None or num_training_steps is None:
            raise ValueError("Cosine scheduling requires warmup and training steps.")
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: _cosine_with_floor(
                step,
                warmup_steps=int(num_warmup_steps),
                training_steps=int(num_training_steps),
                min_lr_ratio=min_lr_ratio,
            ),
            last_epoch=last_epoch,
        )

    get_scheduler._evodex_cosine_min_lr = True  # type: ignore[attr-defined]
    module.get_scheduler = get_scheduler
