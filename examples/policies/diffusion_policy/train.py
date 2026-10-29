from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dp_policy.upstream import configure_upstream

configure_upstream()

import hydra
from omegaconf import DictConfig, OmegaConf

from tracking import install_continuous_epoch_reporting
from scheduler import install_cosine_min_lr
from checkpointing import (
    install_selected_checkpoint_epochs,
    load_workspace_checkpoint,
    save_final_checkpoint,
)
from dex_benchmark.data.multi_dataset import (
    policy_contract_for_dataset,
    training_manifest_for_dataset,
)
from dex_benchmark.data.lerobot_cpu import install_workspace_dataloader


@hydra.main(version_base=None, config_path="config", config_name="evodex_dp")
def main(cfg: DictConfig) -> None:
    """Launch an unmodified official Diffusion Policy training workspace."""

    import wandb

    tracking_dataset = hydra.utils.instantiate(cfg.task.dataset)
    sources = getattr(tracking_dataset, "datasets", (tracking_dataset,))
    if all(not source._split_episodes()[1] for source in sources):
        # No val_loss is produced without held-out episodes. Periodic and final
        # checkpoints remain available; only validation-ranked top-k is disabled.
        cfg.checkpoint.topk.k = 0
    samples_per_epoch = int(
        getattr(tracking_dataset, "samples_per_epoch", len(tracking_dataset))
    )
    steps_per_epoch = math.ceil(samples_per_epoch / int(cfg.dataloader.batch_size))
    OmegaConf.update(
        cfg,
        "evodex.dataset_manifest",
        training_manifest_for_dataset(tracking_dataset),
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "evodex.policy_contract",
        policy_contract_for_dataset(tracking_dataset),
        force_add=True,
    )
    del tracking_dataset
    install_continuous_epoch_reporting(wandb, steps_per_epoch=steps_per_epoch)

    workspace_cls = hydra.utils.get_class(cfg._target_)
    min_lr = cfg.training.get("min_lr")
    optimizer_lr = cfg.optimizer.get(
        "learning_rate", cfg.optimizer.get("lr")
    )
    install_cosine_min_lr(
        workspace_cls,
        min_lr_ratio=(
            None if min_lr is None else float(min_lr) / float(optimizer_lr)
        ),
    )
    install_workspace_dataloader(
        workspace_cls,
        sampling_mode=str(cfg.cpu_data_pipeline.sampling_mode),
        temporal_block_size=int(cfg.cpu_data_pipeline.temporal_block_size),
        seed=int(cfg.training.seed),
        prefetch_factor=int(cfg.cpu_data_pipeline.prefetch_factor),
    )
    workspace = workspace_cls(cfg)
    resume_checkpoint = cfg.training.get("resume_checkpoint")
    init_checkpoint = cfg.training.get("init_checkpoint")
    if resume_checkpoint and init_checkpoint:
        raise ValueError("resume_checkpoint and init_checkpoint are mutually exclusive.")
    if resume_checkpoint:
        load_workspace_checkpoint(workspace, resume_checkpoint, weights_only=False)
        cfg.training.resume = False
    elif init_checkpoint:
        load_workspace_checkpoint(workspace, init_checkpoint, weights_only=True)
        cfg.training.resume = False
    checkpoint_epochs = cfg.training.get("checkpoint_epochs")
    if checkpoint_epochs:
        # The official workspaces only invoke ``save_checkpoint`` when
        # ``save_last_ckpt`` is enabled and ``checkpoint_every`` divides the
        # current zero-based epoch.  Explicit completed-epoch selection must
        # therefore enable that call site and inspect every epoch; the wrapper
        # below suppresses all writes except the requested epochs.
        cfg.checkpoint.save_last_ckpt = True
        cfg.training.checkpoint_every = 1
        install_selected_checkpoint_epochs(workspace, list(checkpoint_epochs))
    workspace.run()
    if bool(cfg.checkpoint.save_last_ckpt) and not checkpoint_epochs:
        final_checkpoint = save_final_checkpoint(workspace)
        print(f"Saved final checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
