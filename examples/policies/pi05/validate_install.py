"""Check an OpenPI environment prepared for Evo-DexBench π0.5.

Run with the OpenPI environment's interpreter. It verifies the pinned commit,
the overlay, both training configs, the LeRobot v3 loader and the tokenizer
guard, without loading model weights or datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PINNED_COMMIT = (HERE / "pi05_policy/UPSTREAM_COMMIT").read_text().split()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--openpi-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.openpi_root.expanduser().resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise SystemExit(f"Expected OpenPI {PINNED_COMMIT}, got {commit}.")
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("EVODEX_PI05_BASE_JAX64", "/unused/for/validation")

    from importlib.metadata import version

    import jax
    import lerobot.datasets.lerobot_dataset  # noqa: F401  (LeRobot v3 layout)
    from openpi.models import tokenizer
    from openpi.training import config, data_loader, evodex_lerobot  # noqa: F401

    result = {"openpi_commit": commit, "jax": jax.__version__, "lerobot": version("lerobot"),
              "devices": [str(device) for device in jax.devices()]}
    for name, action_dim, token_len in (
        ("evodex_pi05_single_jax", 32, 200),
        ("evodex_pi05_dual_jax", 64, 384),
    ):
        cfg = config.get_config(name)
        model = cfg.model
        if (model.action_dim, model.action_horizon, model.max_token_len) != (action_dim, 50, token_len):
            raise SystemExit(f"{name} has an unexpected model contract: {model}.")
        result[name] = {"action_dim": model.action_dim, "action_horizon": model.action_horizon,
                        "max_token_len": model.max_token_len, "ema_decay": cfg.ema_decay}
    if "EVODEX_REJECT_PROMPT_TRUNCATION" not in Path(tokenizer.__file__).read_text():
        raise SystemExit("The tokenizer guard is missing; reinstall the overlay on a clean checkout.")
    print(json.dumps(result, indent=2))
    print("π0.5 OpenPI environment is ready.")


if __name__ == "__main__":
    main()
