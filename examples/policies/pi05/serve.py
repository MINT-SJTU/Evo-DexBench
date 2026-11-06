"""Serve an Evo-DexBench π0.5 Orbax checkpoint with the official OpenPI server.

This is only a launcher: it checks the pinned checkout and the checkpoint
layout, sets the dataset identity used to locate normalization statistics,
and then executes OpenPI's unmodified ``scripts/serve_policy.py``. Run it with
the Python interpreter of the OpenPI environment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

HERE = Path(__file__).resolve().parent
PINNED_COMMIT = (HERE / "pi05_policy/UPSTREAM_COMMIT").read_text().split()[-1]
CONFIG_NAMES = {"single": "evodex_pi05_single_jax", "dual": "evodex_pi05_dual_jax"}
ACTION_DIMS = {"single": 32, "dual": 64}


def validate_openpi_checkout(openpi_root: Path) -> Path:
    root = openpi_root.expanduser().resolve()
    if not (root / "scripts/serve_policy.py").is_file():
        raise FileNotFoundError(f"OpenPI scripts/serve_policy.py is missing under {root}.")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"Expected OpenPI {PINNED_COMMIT}, got {commit}.")
    if not (root / "src/openpi/training/evodex_config.py").is_file():
        raise FileNotFoundError(
            f"The Evo-DexBench overlay is not installed in {root}; run install_overlay.py first."
        )
    return root


def resolve_dataset_id(checkpoint: Path, dataset_id: str | None) -> str:
    """Return the normalization asset name stored in the checkpoint."""

    available = sorted(path.parent.name for path in (checkpoint / "assets").glob("*/norm_stats.json"))
    if dataset_id is None:
        if len(available) != 1:
            raise ValueError(
                f"Pass --dataset-id; {checkpoint / 'assets'} holds normalization statistics for {available}."
            )
        return available[0]
    if dataset_id not in available:
        raise FileNotFoundError(f"No assets/{dataset_id}/norm_stats.json in {checkpoint}; found {available}.")
    return dataset_id


def validate_checkpoint(checkpoint: Path, embodiment: str, dataset_id: str | None) -> tuple[Path, str]:
    root = checkpoint.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {root}")
    if not (root / "params/_METADATA").is_file():
        raise FileNotFoundError(
            f"{root / 'params/_METADATA'} is missing. Pass the numbered checkpoint directory "
            "that contains params/ and assets/, not params/ itself or the run root."
        )
    dataset_id = resolve_dataset_id(root, dataset_id)
    stats = json.loads((root / "assets" / dataset_id / "norm_stats.json").read_text(encoding="utf-8"))
    norm = stats.get("norm_stats", stats)
    state_dim = len(norm.get("state", {}).get("mean", []))
    action_dim = len(norm.get("actions", {}).get("mean", []))
    expected = (31, 30) if embodiment == "single" else (62, 60)
    if (state_dim, action_dim) != expected:
        raise ValueError(
            f"Normalization statistics have state/action dims {(state_dim, action_dim)}, "
            f"but a {embodiment} checkpoint needs {expected}; check --embodiment."
        )
    return root, dataset_id


def server_command(python: Path, openpi_root: Path, checkpoint: Path, *, embodiment: str, port: int) -> list[str]:
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    return [
        # A venv interpreter is usually a symlink; resolving it would select
        # the base interpreter and lose the environment's packages.
        str(python.expanduser().absolute()),
        str(openpi_root / "scripts/serve_policy.py"),
        "--port",
        str(port),
        "policy:checkpoint",
        "--policy.config",
        CONFIG_NAMES[embodiment],
        "--policy.dir",
        str(checkpoint),
    ]


def server_environment(base: Mapping[str, str], *, embodiment: str, dataset_id: str, openpi_root: Path) -> dict[str, str]:
    prefix = f"EVODEX_{embodiment.upper()}"
    env = dict(base)
    env.update(
        {
            f"{prefix}_DATASET_REPO_ID": dataset_id,
            f"{prefix}_DATASET_REPO_IDS": dataset_id,
            f"{prefix}_REPLAY_REPO_IDS": "",
            f"{prefix}_REPLAY_RATIO": "0",
            "EVODEX_REJECT_PROMPT_TRUNCATION": "1",
        }
    )
    source = str(openpi_root / "src")
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint directory containing params/ and assets/.")
    parser.add_argument("--embodiment", choices=sorted(CONFIG_NAMES), required=True)
    parser.add_argument("--dataset-id", default=None,
                        help="Normalization asset name; inferred when the checkpoint holds one.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the launch command only.")
    args = parser.parse_args(argv)

    openpi_root = validate_openpi_checkout(args.openpi_root)
    checkpoint, dataset_id = validate_checkpoint(args.checkpoint, args.embodiment, args.dataset_id)
    command = server_command(args.python, openpi_root, checkpoint, embodiment=args.embodiment, port=args.port)
    env = server_environment(os.environ, embodiment=args.embodiment, dataset_id=dataset_id, openpi_root=openpi_root)
    if args.dry_run:
        print(json.dumps({"checkpoint": str(checkpoint), "dataset_id": dataset_id, "command": command}, indent=2))
        return
    os.chdir(openpi_root)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
