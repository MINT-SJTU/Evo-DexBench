"""Official OpenPI JAX π0.5 training configs for Evo-DexBench.

``install_overlay.py`` copies this module into a pinned OpenPI checkout and
registers ``evodex_pi05_single_jax`` and ``evodex_pi05_dual_jax``. Dataset
identities, base weights and output directories are supplied through
environment variables, which ``train.py`` and ``serve.py`` set explicitly.
"""

from __future__ import annotations

import dataclasses
import os

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders

from openpi.policies import evodex_policy


CONTROL_PROMPT = (
    "<control_mode> end effector delta in robot root frame; xyz euler "
    "rotation delta; absolute hand joint positions </control_mode>"
)
ASSETS_BASE_DIR = os.environ.get("EVODEX_ASSETS_BASE_DIR", "./assets")
CHECKPOINT_BASE_DIR = os.environ.get("EVODEX_CHECKPOINT_BASE_DIR", "./checkpoints")
NUM_WORKERS = int(os.environ.get("EVODEX_NUM_WORKERS", "4"))
FSDP_DEVICES = int(os.environ.get("EVODEX_FSDP_DEVICES", "1"))
# The released π0.5 base has a 32D action projection. Dual-arm training starts
# from a 64D copy prepared once with ``extend_action_head.py``.
PI05_BASE_JAX = os.environ.get(
    "EVODEX_PI05_BASE_JAX", "gs://openpi-assets/checkpoints/pi05_base/params"
)
PI05_BASE_JAX64 = os.environ.get("EVODEX_PI05_BASE_JAX64", "")
# Dual-arm prompts carry 64 discretized state values and exceed π0.5's default
# 200-token budget. 384 tokens keep the complete state in every prompt.
DUAL_MAX_TOKEN_LEN = int(os.environ.get("EVODEX_DUAL_MAX_TOKEN_LEN", "384"))

CAMERAS = {
    "single": {
        "base_0_rgb": "observation.images.context",
        "left_wrist_0_rgb": "observation.images.interaction",
        "right_wrist_0_rgb": "observation.images.wrist_right",
    },
    "dual": {
        "base_0_rgb": "observation.images.context",
        "left_wrist_0_rgb": "observation.images.wrist_left",
        "right_wrist_0_rgb": "observation.images.wrist_right",
    },
}


def _repo_ids(name: str, fallback: str) -> tuple[str, ...]:
    value = os.environ.get(name, fallback)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclasses.dataclass(frozen=True)
class EvoDexDataConfig(_config.DataConfig):
    """OpenPI data config carrying the physical LeRobot collection members."""

    repo_ids: tuple[str, ...] = ()
    replay_repo_ids: tuple[str, ...] = ()
    replay_ratio: float = 0.0


@dataclasses.dataclass(frozen=True)
class LeRobotEvoDexDataConfig(_config.DataConfigFactory):
    """LeRobot v3.0 data with explicit Evo-DexBench semantic fields.

    ``repo_id`` is the logical dataset id used for normalization assets;
    ``repo_ids`` lists the physical LeRobot datasets that are mixed.
    """

    embodiment: str = "single"
    repo_ids: tuple[str, ...] = ()
    replay_repo_ids: tuple[str, ...] = ()
    replay_ratio: float = 0.0

    def create(self, assets_dirs, model_config):
        if self.embodiment not in CAMERAS:
            raise ValueError("embodiment must be single or dual")
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": CAMERAS[self.embodiment],
                        "state": "observation.state",
                        "actions": "action",
                        # LeRobot v3 exposes the instruction on every sample.
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                evodex_policy.EvoDexInputs(
                    model_config.action_dim,
                    CONTROL_PROMPT,
                    self.embodiment,
                )
            ],
            outputs=[evodex_policy.EvoDexOutputs(model_config.action_dim)],
        )
        base = dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory(
                default_prompt=CONTROL_PROMPT
            )(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=False,
        )
        return EvoDexDataConfig(
            **{field.name: getattr(base, field.name) for field in dataclasses.fields(base)},
            repo_ids=self.repo_ids or (str(base.repo_id),),
            replay_repo_ids=self.replay_repo_ids,
            replay_ratio=float(self.replay_ratio),
        )


def _data_config(embodiment: str) -> LeRobotEvoDexDataConfig:
    prefix = f"EVODEX_{embodiment.upper()}"
    repo_id = os.environ.get(
        f"{prefix}_DATASET_REPO_ID", f"evo-dexbench/REPLACE_WITH_{embodiment.upper()}_DATASET"
    )
    return LeRobotEvoDexDataConfig(
        repo_id=repo_id,
        repo_ids=_repo_ids(f"{prefix}_DATASET_REPO_IDS", repo_id),
        replay_repo_ids=_repo_ids(f"{prefix}_REPLAY_REPO_IDS", ""),
        replay_ratio=float(os.environ.get(f"{prefix}_REPLAY_RATIO", "0")),
        embodiment=embodiment,
        base_config=_config.DataConfig(prompt_from_task=False),
    )


def _model(action_dim: int, **kwargs) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=action_dim,
        action_horizon=50,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        **kwargs,
    )


def get_evodex_configs() -> list[_config.TrainConfig]:
    # ``train.py`` replaces the batch, update budget and schedule with the
    # values of the selected recipe; these are the shared training defaults.
    common = {
        "assets_base_dir": ASSETS_BASE_DIR,
        "checkpoint_base_dir": CHECKPOINT_BASE_DIR,
        "num_workers": NUM_WORKERS,
        "fsdp_devices": FSDP_DEVICES,
        "ema_decay": 0.99,
        "batch_size": 32,
        "num_train_steps": 30_000,
        "lr_schedule": _optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-6,
            decay_steps=30_000,
            decay_lr=5e-7,
        ),
        "optimizer": _optimizer.AdamW(clip_gradient_norm=1.0),
        "wandb_enabled": False,
    }
    return [
        _config.TrainConfig(
            name="evodex_pi05_single_jax",
            model=_model(32),
            data=_data_config("single"),
            weight_loader=_weight_loaders.CheckpointWeightLoader(PI05_BASE_JAX),
            **common,
        ),
        _config.TrainConfig(
            name="evodex_pi05_dual_jax",
            model=_model(64, max_token_len=DUAL_MAX_TOKEN_LEN),
            data=_data_config("dual"),
            # Serving does not read the initialization weights. Training
            # requires EVODEX_PI05_BASE_JAX64 and checks it before starting.
            weight_loader=_weight_loaders.CheckpointWeightLoader(
                PI05_BASE_JAX64 or "EVODEX_PI05_BASE_JAX64_IS_UNSET"
            ),
            **common,
        ),
    ]
