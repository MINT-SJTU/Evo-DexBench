from __future__ import annotations

import argparse
import json
from pathlib import Path

from dex_benchmark.data.canonical import validate_canonical_hdf5


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate canonical HDF5 episodes.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--required-camera", action="append", default=[])
    parser.add_argument("--required-tactile-role", action="append", default=[])
    args = parser.parse_args(argv)
    report = validate_canonical_hdf5(
        args.path,
        manifest_path=args.manifest,
        verify_checksums=args.verify_checksums,
        required_cameras=tuple(args.required_camera),
        required_tactile_roles=tuple(args.required_tactile_role),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
