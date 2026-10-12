from __future__ import annotations

import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinedex_policy.upstream import configure_upstream

configure_upstream()

import hydra
from omegaconf import DictConfig, OmegaConf

from training_utils import (
    install_continuous_epoch_reporting,
    load_workspace_checkpoint,
)
from kinedex_policy.upstream import EXPECTED_COMMIT
from dex_benchmark.data.lerobot_cpu import install_workspace_dataloader
from dex_benchmark.data.multi_dataset import training_manifest_for_dataset


@hydra.main(version_base=None, config_path="config", config_name="evodex_kinedex")
def main(cfg: DictConfig) -> None:
    """Train the official KineDex policy through an Evo-DexBench data view."""

    import wandb

    tracking_dataset = hydra.utils.instantiate(cfg.task.dataset)
    metadata = tracking_dataset.checkpoint_metadata()
    metadata["dataset_manifest"] = training_manifest_for_dataset(tracking_dataset)
    metadata["upstream_commit"] = EXPECTED_COMMIT
    steps_per_epoch = math.ceil(
        int(getattr(tracking_dataset, "samples_per_epoch", len(tracking_dataset)))
        / int(cfg.dataloader.batch_size)
    )
    del tracking_dataset
    OmegaConf.update(cfg, "evodex", metadata, force_add=True)
    install_continuous_epoch_reporting(wandb, steps_per_epoch=steps_per_epoch)
    workspace_cls = hydra.utils.get_class(cfg._target_)
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
    workspace.run()

    # The upstream workspace writes checkpoints on background threads.  Wait for
    # the last periodic save, then publish a synchronous, atomically-renamed
    # checkpoint so a completed run always has a fully written final state.
    if not bool(cfg.checkpoint.save_final):
        return

    saving_thread = getattr(workspace, "_saving_thread", None)
    if saving_thread is not None:
        saving_thread.join()
    checkpoint_dir = Path(workspace.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "latest.ckpt"
    temporary_path = checkpoint_dir / "latest.ckpt.tmp"
    workspace.save_checkpoint(path=temporary_path, use_thread=False)
    temporary_path.replace(final_path)
    print(f"Final checkpoint saved atomically to {final_path}")


if __name__ == "__main__":
    main()
