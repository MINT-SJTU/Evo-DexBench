from __future__ import annotations

import argparse
from pathlib import Path

from dex_benchmark.data.converters.to_lerobot import (
    DEFAULT_IMAGE_WRITER_THREADS,
    convert_to_lerobot,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert validated HDF5 canonical episodes to LeRobot dataset v3.0."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Recorder or multi-episode collection manifest for checksum verification.",
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help="Only convert episodes marked valid=true in quality_report.json.",
    )
    parser.add_argument("--max-held-frames", type=int, default=2)
    parser.add_argument("--max-held-ratio", type=float, default=0.05)
    parser.add_argument(
        "--image-writer-processes",
        type=int,
        default=0,
        help="Number of LeRobot image-writer processes (0 uses threads).",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=DEFAULT_IMAGE_WRITER_THREADS,
        help="Number of asynchronous image-writer threads; default: 4.",
    )
    parser.add_argument(
        "--allow-task-id-instruction-fallback",
        action="store_true",
        help="Use task_id as instruction only when episode metadata has no instruction.",
    )
    args = parser.parse_args(argv)
    print(
        convert_to_lerobot(
            args.input,
            args.output,
            repo_id=args.repo_id,
            fps=args.fps,
            verify_checksums=args.verify_checksums,
            manifest_path=args.manifest,
            quality_report_path=args.quality_report,
            max_held_frames=args.max_held_frames,
            max_held_ratio=args.max_held_ratio,
            allow_task_id_instruction_fallback=args.allow_task_id_instruction_fallback,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
        )
    )


if __name__ == "__main__":
    main()
