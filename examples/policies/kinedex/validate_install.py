from __future__ import annotations

import argparse
from pathlib import Path
import platform
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kinedex_policy.upstream import configure_upstream


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--dataset-path")
    parser.add_argument("--repo-id")
    parser.add_argument("--embodiment", choices=("single", "dual"), default="single")
    parser.add_argument("--tactile-mode", choices=("full", "none"), default="full")
    args = parser.parse_args()
    root = configure_upstream(args.upstream_root)

    import av
    import diffusers
    import lerobot
    import numpy as np
    import torch
    import torchvision
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import get_safe_default_codec

    if np.__version__ != "1.26.4":
        raise RuntimeError(f"Expected NumPy 1.26.4, found {np.__version__}")
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"Expected PyTorch 2.5.1, found {torch.__version__}")
    if torchvision.__version__.split("+")[0] != "0.20.1":
        raise RuntimeError(
            f"Expected torchvision 0.20.1, found {torchvision.__version__}"
        )
    if lerobot.__version__ != "0.4.0":
        raise RuntimeError(f"Expected LeRobot 0.4.0, found {lerobot.__version__}")
    if get_safe_default_codec() != "pyav":
        raise RuntimeError("KineDex must use the validated PyAV video backend.")

    result: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": torchvision.__version__,
        "diffusers": diffusers.__version__,
        "lerobot": lerobot.__version__,
        "av": av.__version__,
        "lerobot_dataset": LeRobotDataset.__name__,
        "video_backend": "pyav",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "upstream_root": str(root),
    }
    if args.dataset_path:
        if not args.repo_id:
            parser.error("--repo-id is required with --dataset-path")
        from kinedex_policy.dataset import LeRobotV3KineDexDataset

        dataset = LeRobotV3KineDexDataset(
            args.dataset_path,
            args.repo_id,
            args.embodiment,
            tactile_mode=args.tactile_mode,
            val_ratio=0.1,
        )
        sample = dataset[0]
        result["dataset_samples"] = len(dataset)
        result["sample_shapes"] = {
            "obs": {key: tuple(value.shape) for key, value in sample["obs"].items()},
            "action": tuple(sample["action"].shape),
        }
        result["metadata"] = dataset.checkpoint_metadata()
    print(result)


if __name__ == "__main__":
    main()
