from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
EXPECTED_COMMIT = (ROOT / "UPSTREAM_COMMIT").read_text().strip()
NOTOUCH_PATCH = ROOT / "notouch.patch"


def configure_upstream(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get("KINEDEX_ROOT")
    if raw is None:
        raise RuntimeError("Set KINEDEX_ROOT to the prepared official checkout.")
    path = Path(raw).expanduser().resolve()
    if not (path / "diffusion_policy" / "policy" / "diffusion_unet_image_policy.py").is_file():
        raise FileNotFoundError(f"Invalid KineDex checkout: {path}")
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=path, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"KineDex commit mismatch: {commit}.")
    patched = subprocess.run(
        ("git", "apply", "--reverse", "--check", str(NOTOUCH_PATCH)), cwd=path,
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if patched.returncode != 0:
        raise RuntimeError(
            "The Evo-DexBench KineDex patches are not fully applied. "
            "Run examples/policies/kinedex/prepare_upstream.py again."
        )
    loaded = sys.modules.get("diffusion_policy")
    if loaded is not None:
        loaded_path = Path(getattr(loaded, "__file__", "")).resolve()
        if path not in loaded_path.parents:
            raise RuntimeError("A different diffusion_policy package is already imported.")
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)
    return path
