from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    args = parser.parse_args()
    from dp_policy.upstream import configure_upstream

    configure_upstream(args.official_root)

    from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import (
        DiffusionTransformerHybridImagePolicy,
    )
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    assert DiffusionUnetHybridImagePolicy is not None
    assert DiffusionTransformerHybridImagePolicy is not None
    assert LeRobotDataset is not None
    print(
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "official_root": str(args.official_root.resolve()),
        }
    )


if __name__ == "__main__":
    main()
