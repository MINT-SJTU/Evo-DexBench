from __future__ import annotations

from pathlib import Path
from typing import Any


def load_workspace_checkpoint(
    workspace: Any,
    checkpoint: str | Path,
    *,
    weights_only: bool,
) -> None:
    """Load an official workspace checkpoint with explicit transfer semantics."""

    import dill
    import torch
    from omegaconf import OmegaConf

    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(
        path.open("rb"), pickle_module=dill, map_location="cpu", weights_only=False
    )
    if not weights_only:
        expected = OmegaConf.to_container(workspace.cfg.get("evodex", {}), resolve=True)
        saved = OmegaConf.to_container(payload["cfg"].get("evodex", {}), resolve=True)
        if expected != saved:
            raise ValueError(
                "Full resume requires the identical dataset manifest and policy contract. "
                "Use init_checkpoint when changing datasets."
            )
        workspace.load_payload(payload)
        return

    source_cfg = payload["cfg"]
    source_cls = __import__("hydra").utils.get_class(source_cfg._target_)
    source = source_cls(source_cfg)
    source.load_payload(payload)
    source_policy = (
        source.ema_model
        if bool(source_cfg.training.use_ema) and source.ema_model is not None
        else source.model
    )
    target_state = workspace.model.state_dict()
    source_state = source_policy.state_dict()
    # Normalizer state is dataset-specific and must be fitted by the new run.
    transferable = {
        key: value
        for key, value in source_state.items()
        if key in target_state
        and target_state[key].shape == value.shape
        and "normalizer" not in key
    }
    missing = [
        key for key in target_state if "normalizer" not in key and key not in transferable
    ]
    if missing:
        raise ValueError(
            "Init checkpoint model contract mismatch; missing parameters: "
            f"{missing[:5]}"
        )
    workspace.model.load_state_dict(transferable, strict=False)
    if workspace.ema_model is not None:
        workspace.ema_model.load_state_dict(transferable, strict=False)


def install_selected_checkpoint_epochs(
    workspace: Any, epochs: list[int] | tuple[int, ...]
) -> None:
    """Keep explicit completed epochs without modifying official workspaces."""

    selected = frozenset(int(epoch) for epoch in epochs)
    if not selected or min(selected) <= 0:
        raise ValueError("checkpoint epochs must be positive.")
    original = workspace.save_checkpoint

    def save_checkpoint(*args: Any, **kwargs: Any):
        # Official workspaces call this before incrementing their zero-based
        # epoch counter. Convert it to the user-facing completed epoch count.
        completed_epoch = int(workspace.epoch) + 1
        if completed_epoch not in selected:
            return None
        # Suppress top-k duplicates and write one deterministic, synchronous
        # checkpoint for the requested completed epoch.
        if kwargs.get("path") is not None:
            return None
        return original(
            tag=f"epoch_{completed_epoch:04d}",
            use_thread=False,
        )

    workspace.save_checkpoint = save_checkpoint


def save_final_checkpoint(workspace: Any) -> Path:
    """Synchronously make ``latest.ckpt`` represent the completed run.

    The official workspaces checkpoint before incrementing their zero-based
    epoch counter. Consequently, a run whose epoch count is an exact multiple
    of ``checkpoint_every`` otherwise leaves ``latest.ckpt`` one interval
    behind. Wait for any asynchronous periodic write before replacing it with
    the state returned by the final epoch.
    """

    saving_thread = getattr(workspace, "_saving_thread", None)
    if saving_thread is not None:
        saving_thread.join()

    checkpoint_path = workspace.save_checkpoint(tag="latest", use_thread=False)
    return Path(checkpoint_path)
