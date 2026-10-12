from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).with_name("config")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinedex_policy.upstream import EXPECTED_COMMIT, configure_upstream


def _compose(embodiment: str, tactile_mode: str) -> Any:
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(
            config_name="evodex_kinedex",
            overrides=[
                f"embodiment={embodiment if tactile_mode == 'full' else embodiment + '_no_touch'}"
            ],
        )


def _image_normalizer(normalizer: Any, camera_roles: tuple[str, ...]) -> None:
    from diffusion_policy.common.normalize_util import get_image_range_normalizer

    for role in camera_roles:
        normalizer[f"{role}_rgb"] = get_image_range_normalizer()


def _synthetic_batch(cfg: Any, *, batch_size: int) -> tuple[Any, dict[str, Any]]:
    import torch
    from diffusion_policy.model.common.normalizer import (
        LinearNormalizer,
        SingleFieldLinearNormalizer,
    )

    state_dim = int(cfg.embodiment.state_dim)
    action_dim = int(cfg.embodiment.model_action_dim)
    finger_count = int(cfg.embodiment.finger_count)
    camera_roles = tuple(cfg.embodiment.camera_roles)
    rng = np.random.default_rng(42)
    normalizer = LinearNormalizer()
    normalizer["state"] = SingleFieldLinearNormalizer.create_fit(
        rng.normal(size=(64, state_dim)).astype(np.float32)
    )
    if cfg.embodiment.tactile_mode == "full":
        tactile = rng.random(size=(64, finger_count, 29, 1), dtype=np.float32)
        batch_tactile = torch.rand(batch_size, 2, finger_count, 29, 1)
        normalizer["finger_raw_pressure"] = SingleFieldLinearNormalizer.create_fit(
            tactile
        )
    else:
        tactile = np.zeros((64, finger_count, 29, 1), dtype=np.float32)
        batch_tactile = torch.zeros(batch_size, 2, finger_count, 29, 1)
        normalizer["finger_raw_pressure"] = SingleFieldLinearNormalizer.create_manual(
            scale=np.ones((finger_count, 29, 1), dtype=np.float32),
            offset=np.zeros((finger_count, 29, 1), dtype=np.float32),
            input_stats_dict={
                "min": tactile.min(axis=0),
                "max": tactile.max(axis=0),
                "mean": tactile.mean(axis=0),
                "std": tactile.std(axis=0),
            },
        )
    normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
        rng.normal(size=(64, action_dim)).astype(np.float32)
    )
    _image_normalizer(normalizer, camera_roles)
    batch = {
        "obs": {
            "state": torch.randn(batch_size, 2, state_dim),
            "finger_raw_pressure": batch_tactile,
            **{
                f"{role}_rgb": torch.rand(batch_size, 2, 3, 224, 224)
                for role in camera_roles
            },
        },
        "action": torch.randn(batch_size, 16, action_dim),
    }
    return normalizer, batch


def _real_batch(
    cfg: Any,
    *,
    dataset_path: Path,
    repo_id: str,
    batch_size: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import hydra
    from torch.utils.data import DataLoader

    cfg.task.dataset.dataset_path = str(dataset_path)
    cfg.task.dataset.repo_id = repo_id
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    batch = next(iter(DataLoader(dataset, batch_size=batch_size, shuffle=False)))
    metadata = dataset.checkpoint_metadata()
    metadata["upstream_commit"] = EXPECTED_COMMIT
    return normalizer, batch, metadata


def _validate_predictor(policy: Any, metadata: dict[str, Any], batch: dict[str, Any]) -> None:
    import torch

    from kinedex_policy.server import KineDexInferenceServer

    server = object.__new__(KineDexInferenceServer)
    server.torch = torch
    server.device = next(policy.parameters()).device
    server.policy = policy.eval()
    server.metadata = metadata
    observations = {
        key: value.detach().cpu().numpy() for key, value in batch["obs"].items()
    }
    response = server._predict(
        {
            "policy_type": "kinedex",
            "embodiment": metadata["embodiment"],
            "tactile_mode": metadata.get("tactile_mode", "full"),
            "observations": observations,
        }
    )
    print(
        {
            "motion_shape": response["motions"].shape,
            "force_shape": (
                response["force_targets"].shape
                if "force_targets" in response
                else None
            ),
            "tactile_placeholder_zero": bool(
                np.count_nonzero(observations["finger_raw_pressure"]) == 0
            ),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one KineDex forward/backward and inference-boundary check."
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--embodiment", choices=("single", "dual"), default="single")
    parser.add_argument("--tactile-mode", choices=("full", "none"), default="full")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--repo-id", default="local/evodex")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--check-inference", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.embodiment == "single" and args.dataset_path is None:
        parser.error("single validation requires --dataset-path")

    configure_upstream(args.upstream_root)
    import hydra
    import torch

    from diffusion_policy.common.pytorch_util import dict_apply

    cfg = _compose(args.embodiment, args.tactile_mode)
    metadata: dict[str, Any] | None = None
    if args.dataset_path is None:
        normalizer, batch = _synthetic_batch(cfg, batch_size=args.batch_size)
    else:
        normalizer, batch, metadata = _real_batch(
            cfg,
            dataset_path=args.dataset_path.expanduser().resolve(),
            repo_id=args.repo_id,
            batch_size=args.batch_size,
        )
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(normalizer)
    policy.train().to(args.device)
    batch = dict_apply(batch, lambda value: value.to(args.device, non_blocking=True))
    loss = policy.compute_loss(batch)
    loss.backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
    print(
        {
            "embodiment": args.embodiment,
            "tactile_mode": args.tactile_mode,
            "action_dim": int(cfg.embodiment.model_action_dim),
            "loss": float(loss.detach().cpu()),
            "finite_gradients": bool(finite_gradients),
            "allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        }
    )
    if not torch.isfinite(loss) or not finite_gradients:
        raise RuntimeError("KineDex produced non-finite training values.")
    if args.check_inference:
        if metadata is None:
            parser.error("--check-inference requires a real dataset")
        _validate_predictor(policy, metadata, batch)


if __name__ == "__main__":
    main()
