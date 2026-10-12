"""Run focused quality checks on canonical HDF5 episodes."""

from __future__ import annotations

import argparse
from pathlib import Path

from dex_benchmark.data.canonical.quality import QualityConfig, quality_check_dataset


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HDF5 file or dataset directory.")
    parser.add_argument("output", type=Path, help="Directory where quality_report.json is written.")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--image-hz", type=float, default=30.0)
    parser.add_argument("--required-camera", action="append", default=[])
    parser.add_argument("--required-tactile-role", action="append", default=[])
    args = parser.parse_args(argv)
    report_path = quality_check_dataset(
        args.input,
        args.output,
        config=QualityConfig(
            expected_control_hz=args.control_hz,
            expected_image_hz=args.image_hz,
            required_cameras=tuple(args.required_camera),
            required_tactile_roles=tuple(args.required_tactile_role),
        ),
    )
    print(report_path)


if __name__ == "__main__":
    main()
