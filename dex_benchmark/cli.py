from __future__ import annotations

import importlib
import json
import argparse
from collections.abc import Callable, Sequence
from typing import Any

_COMMAND_MODULES: dict[tuple[str, ...], str] = {
    ("run",): "dex_benchmark.commands.teleoperation.keyboard",
    ("teleop", "keyboard"): "dex_benchmark.commands.teleoperation.keyboard",
    ("teleop", "quest3"): "dex_benchmark.commands.teleoperation.quest3",
    ("teleop", "quest3-input"): "dex_benchmark.commands.teleoperation.quest3_input",
    ("data", "collect"): "dex_benchmark.commands.data.collect",
    ("data", "validate"): "dex_benchmark.commands.data.validate",
    ("data", "quality"): "dex_benchmark.commands.data.quality",
    ("data", "replay"): "dex_benchmark.commands.data.replay",
    ("data", "convert-lerobot"): "dex_benchmark.commands.data.convert_lerobot",
    ("evaluate",): "dex_benchmark.commands.evaluation.policy",
}

_GROUP_HELP: dict[str, str] = {
    "teleop": "Usage: dexbench teleop {keyboard,quest3,quest3-input} [options]",
    "data": "Usage: dexbench data {collect,validate,quality,replay,convert-lerobot} [options]",
}


def _print_help() -> None:
    print(
        """Evo-DexBench command line interface

Usage:
  dexbench tasks list
  dexbench tasks show TASK_ID
  dexbench run [keyboard teleoperation options]
  dexbench teleop {keyboard,quest3,quest3-input} [options]
  dexbench data {collect,validate,quality,replay,convert-lerobot} [options]
  dexbench evaluate --policy NAME --task-id TASK_ID [options]
  dexbench doctor

Defaults: m01_v1 for single-hand commands and b01_v1 for dual-hand commands.
Run `dexbench <command> --help` for command-specific options.
"""
    )


def _task_record(task_id: str) -> dict[str, Any]:
    from dex_benchmark.sim.tasks import TaskCatalog

    settings = TaskCatalog.get(task_id)
    return {
        "task_id": settings.task_id,
        "task_name": settings.task_name,
        "instruction": settings.instruction,
        "max_steps": settings.max_steps,
        "embodiment_id": settings.embodiment_spec.embodiment_id,
        "robot_setup": settings.taxonomy.robot_setup.value,
        "scene_class": settings.taxonomy.scene_class.value,
        "affordance_family": settings.taxonomy.affordance_family.value,
        "task_split": settings.taxonomy.task_split.value,
        "layouts": [layout.value for layout in settings.supported_layouts],
    }


def _run_tasks(argv: Sequence[str]) -> int:
    from dex_benchmark.sim.tasks import TaskCatalog

    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: dexbench tasks {list|show TASK_ID}")
        return 0
    if argv[0] == "list" and len(argv) == 1:
        for task_id in TaskCatalog.task_ids():
            record = _task_record(task_id)
            print(
                f"{task_id:<8} {record['task_name']:<28} "
                f"{record['robot_setup']:<23} {record['scene_class']:<8} "
                f"{record['instruction']}"
            )
        return 0
    if argv[0] == "show" and len(argv) == 2:
        print(json.dumps(_task_record(argv[1]), indent=2, ensure_ascii=False))
        return 0
    raise SystemExit("Usage: dexbench tasks {list|show TASK_ID}")


def _run_doctor(argv: Sequence[str]) -> int:
    if argv and argv != ["-h"] and argv != ["--help"]:
        raise SystemExit("Usage: dexbench doctor")
    from dex_benchmark.sim.tasks import TaskCatalog
    from dex_benchmark.sim.tasks.builtins import BUILTIN_TASK_MODULES
    import numpy
    import torch

    catalog_ids = set(TaskCatalog.task_ids())
    implementation_ids = set(BUILTIN_TASK_MODULES)
    missing_implementations = sorted(catalog_ids - implementation_ids)
    missing_configs = sorted(implementation_ids - catalog_ids)
    runtime_issues: list[str] = []
    try:
        torch.zeros(1).numpy()
        torch_numpy_compatible = True
    except (RuntimeError, TypeError) as exc:
        torch_numpy_compatible = False
        runtime_issues.append(
            "PyTorch cannot exchange arrays with NumPy; install the supported "
            f"NumPy/PyTorch pair ({exc})."
        )
    report = {
        "task_count": len(catalog_ids),
        "catalog_matches_implementations": not (
            missing_implementations or missing_configs
        ),
        "missing_implementations": missing_implementations,
        "missing_configs": missing_configs,
        "runtime": {
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "torch_numpy_compatible": torch_numpy_compatible,
        },
        "issues": runtime_issues,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["catalog_matches_implementations"] and not runtime_issues else 1


def _run_assets(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="dexbench assets")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--asset-dir")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--asset-dir", required=True)
    download_parser.add_argument("--bundle-repo")
    download_parser.add_argument("--bundle-revision")
    download_parser.add_argument("--skip-bundle", action="store_true")
    download_parser.add_argument("--skip-partnet", action="store_true")
    download_parser.add_argument("--force-partnet", action="store_true")
    download_parser.add_argument("--hf-endpoint")
    download_parser.add_argument("--hf-mirror-endpoint")
    download_parser.add_argument("--no-hf-mirror", action="store_true")
    download_parser.add_argument("--allow-token-to-mirror", action="store_true")
    args = parser.parse_args(list(argv))
    from dex_benchmark.assets import format_asset_report, verify_assets

    if args.command == "download":
        from dex_benchmark.asset_download import download_assets

        report = download_assets(
            args.asset_dir,
            bundle_repo=args.bundle_repo,
            bundle_revision=args.bundle_revision,
            include_bundle=not args.skip_bundle,
            include_partnet=not args.skip_partnet,
            force_partnet=args.force_partnet,
            hf_endpoint=args.hf_endpoint,
            hf_mirror_endpoint=args.hf_mirror_endpoint,
            allow_hf_mirror=not args.no_hf_mirror,
            allow_token_to_mirror=args.allow_token_to_mirror,
        )
        print(format_asset_report(report))
        return 0 if report["verification"]["ok"] else 1
    report = verify_assets(args.asset_dir)
    print(format_asset_report(report))
    return 0 if report["ok"] else 1


def _run_module(
    module_name: str,
    argv: Sequence[str],
    *,
    program_name: str,
) -> int:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Cannot start {module_name!r}: missing dependency {exc.name!r}. "
            "Install the matching optional dependency group."
        ) from exc
    command_main: Callable[[Sequence[str] | None], Any] = module.main
    import sys

    original_program_name = sys.argv[0]
    sys.argv[0] = program_name
    try:
        result = command_main(list(argv))
    finally:
        sys.argv[0] = original_program_name
    return int(result) if isinstance(result, int) else 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else None
    if arguments is None:
        import sys

        arguments = sys.argv[1:]
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if arguments[0] == "tasks":
        return _run_tasks(arguments[1:])
    if arguments[0] == "doctor":
        return _run_doctor(arguments[1:])
    if arguments[0] == "assets":
        return _run_assets(arguments[1:])
    if (
        arguments[0] in _GROUP_HELP
        and len(arguments) == 2
        and arguments[1] in {"-h", "--help"}
    ):
        print(_GROUP_HELP[arguments[0]])
        return 0

    for command_parts, module_name in sorted(
        _COMMAND_MODULES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if tuple(arguments[: len(command_parts)]) == command_parts:
            command_name = "dexbench " + " ".join(command_parts)
            return _run_module(
                module_name,
                arguments[len(command_parts) :],
                program_name=command_name,
            )

    raise SystemExit(f"Unknown command: {' '.join(arguments)}. Run `dexbench --help`.")


if __name__ == "__main__":
    raise SystemExit(main())
