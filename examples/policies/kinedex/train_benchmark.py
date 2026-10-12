"""Friendly KineDex training entry point for benchmark sampling mixtures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train KineDex with explicit Evo-DexBench dataset sampling ratios."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        nargs=3,
        metavar=("PATH", "REPO_ID", "RATIO"),
        required=True,
        help="Repeat once per LeRobot v3 dataset.",
    )
    parser.add_argument(
        "--embodiment",
        default="single",
        choices=(
            "single", "single_no_touch", "dual", "dual_no_touch",
            "dual_interaction", "dual_interaction_no_touch",
        ),
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="Additional Hydra overrides after --.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    paths = [str(Path(item[0]).expanduser().resolve()) for item in args.dataset]
    repo_ids = [item[1] for item in args.dataset]
    weights = [float(item[2]) for item in args.dataset]
    if any(value <= 0 for value in weights):
        raise SystemExit("Every dataset RATIO must be positive.")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-7, abs_tol=1e-7):
        raise SystemExit(f"Dataset RATIO values must sum to 1.0, got {sum(weights)}.")
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    train_path = Path(__file__).with_name("train.py")
    sys.argv = [
        str(train_path),
        f"embodiment={args.embodiment}",
        f"task.dataset.dataset_paths={json.dumps(paths, separators=(',', ':'))}",
        f"task.dataset.repo_ids={json.dumps(repo_ids, separators=(',', ':'))}",
        f"task.dataset.dataset_weights={json.dumps(weights, separators=(',', ':'))}",
        *extra,
    ]
    runpy.run_path(str(train_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
