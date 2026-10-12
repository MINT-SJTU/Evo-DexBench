"""Small training adapters kept local to the optional KineDex example."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time
from typing import Any


class _ReportedEpochRun:
    def __init__(self, run: Any, steps_per_epoch: int) -> None:
        self._run = run
        self._steps_per_epoch = steps_per_epoch
        self._last_log_time = time.perf_counter()
        self._last_step: int | None = None

    def log(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        metrics = data
        if isinstance(data, Mapping) and "epoch" in data:
            metrics = dict(data)
            epoch_index = int(metrics["epoch"])
            global_step = int(
                metrics.get("global_step", epoch_index * self._steps_per_epoch)
            )
            metrics["epoch_index"] = epoch_index
            metrics["epoch"] = (global_step + 1) / self._steps_per_epoch
            now = time.perf_counter()
            step_delta = (
                1
                if self._last_step is None
                else max(1, global_step - self._last_step)
            )
            metrics["time_per_step"] = (
                now - self._last_log_time
            ) / step_delta
            self._last_log_time = now
            self._last_step = global_step
        return self._run.log(metrics, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._run, name)


def install_continuous_epoch_reporting(
    wandb_module: Any, *, steps_per_epoch: int
) -> None:
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


def load_workspace_checkpoint(
    workspace: Any,
    checkpoint: str | Path,
    *,
    weights_only: bool,
) -> None:
    """Load a KineDex workspace checkpoint with explicit transfer semantics."""

    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(
        path.open("rb"),
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    if not weights_only:
        expected = OmegaConf.to_container(
            workspace.cfg.get("evodex", {}), resolve=True
        )
        saved = OmegaConf.to_container(
            payload["cfg"].get("evodex", {}), resolve=True
        )
        if expected != saved:
            raise ValueError(
                "Full resume requires the identical dataset and policy contract."
            )
        workspace.load_payload(payload)
        return

    source_cfg = payload["cfg"]
    source = hydra.utils.get_class(source_cfg._target_)(source_cfg)
    source.load_payload(payload)
    source_policy = (
        source.ema_model
        if bool(source_cfg.training.use_ema) and source.ema_model is not None
        else source.model
    )
    target_state = workspace.model.state_dict()
    transferable = {
        key: value
        for key, value in source_policy.state_dict().items()
        if key in target_state
        and target_state[key].shape == value.shape
        and "normalizer" not in key
    }
    missing = [
        key
        for key in target_state
        if "normalizer" not in key and key not in transferable
    ]
    if missing:
        raise ValueError(
            "Init checkpoint model contract mismatch; missing parameters: "
            f"{missing[:5]}"
        )
    workspace.model.load_state_dict(transferable, strict=False)
    if workspace.ema_model is not None:
        workspace.ema_model.load_state_dict(transferable, strict=False)
