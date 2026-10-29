"""Locate the unmodified official Diffusion Policy checkout for model processes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


EXPECTED_COMMIT = (Path(__file__).parent / "UPSTREAM_COMMIT").read_text().splitlines()[-1].strip()


def configure_upstream(value: str | Path | None = None) -> Path:
    """Validate the pinned source and expose it without installing it in simulation."""

    default = Path(__file__).resolve().parents[4] / ".third_party" / "DiffusionPolicy-official"
    path = Path(value or os.environ.get("DIFFUSION_POLICY_ROOT", default)).expanduser().resolve()
    if not (path / "diffusion_policy" / "policy" / "diffusion_unet_hybrid_image_policy.py").is_file():
        raise FileNotFoundError(
            f"Official Diffusion Policy checkout not found: {path}. "
            "Run tools/setup_diffusion_policy.sh or set DIFFUSION_POLICY_ROOT."
        )
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=path, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Diffusion Policy commit mismatch: {commit}; expected {EXPECTED_COMMIT}.")
    loaded = sys.modules.get("diffusion_policy")
    if loaded is not None:
        locations = tuple(getattr(loaded, "__path__", ()))
        if not locations or any(
            Path(location).resolve() != path / "diffusion_policy"
            for location in locations
        ):
            raise RuntimeError("A different diffusion_policy package is already imported.")
    # The source can be shared with the server without relying on PYTHONPATH.
    os.environ["DIFFUSION_POLICY_ROOT"] = str(path)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path
