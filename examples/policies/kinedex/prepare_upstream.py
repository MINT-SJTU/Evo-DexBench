from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "kinedex_policy"
EXPECTED_COMMIT = (PACKAGE_ROOT / "UPSTREAM_COMMIT").read_text().strip()
PATCH = PACKAGE_ROOT / "evodex.patch"
NOTOUCH_PATCH = PACKAGE_ROOT / "notouch.patch"
PERIODIC_CHECKPOINT_PATCH = PACKAGE_ROOT / "periodic_checkpoint.patch"


def _run(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=path,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def prepare(path: Path) -> None:
    path = path.expanduser().resolve()
    commit = _run(path, "git", "rev-parse", "HEAD").stdout.strip()
    if commit != EXPECTED_COMMIT:
        raise ValueError(f"KineDex commit mismatch: expected {EXPECTED_COMMIT}, got {commit}.")
    def apply_patch(patch: Path) -> None:
        check = _run(path, "git", "apply", "--check", str(patch), check=False)
        if check.returncode != 0:
            raise RuntimeError(
                f"KineDex patch {patch.name} does not apply cleanly:\n" + check.stderr
            )
        _run(path, "git", "apply", str(patch))
        print(f"Applied KineDex patch {patch.name} to {path}")

    periodic_reverse = _run(
        path,
        "git",
        "apply",
        "--reverse",
        "--check",
        str(PERIODIC_CHECKPOINT_PATCH),
        check=False,
    )
    if periodic_reverse.returncode == 0:
        print(f"Evo-DexBench KineDex patches already applied to {path}")
        return
    notouch_reverse = _run(
        path, "git", "apply", "--reverse", "--check", str(NOTOUCH_PATCH), check=False
    )
    if notouch_reverse.returncode == 0:
        apply_patch(PERIODIC_CHECKPOINT_PATCH)
        return
    base_reverse = _run(
        path, "git", "apply", "--reverse", "--check", str(PATCH), check=False
    )
    if base_reverse.returncode != 0:
        apply_patch(PATCH)
    apply_patch(NOTOUCH_PATCH)
    apply_patch(PERIODIC_CHECKPOINT_PATCH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("upstream_root", type=Path)
    prepare(parser.parse_args().upstream_root)


if __name__ == "__main__":
    main()
