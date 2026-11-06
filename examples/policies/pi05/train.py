"""Train π0.5 on Evo-DexBench LeRobot v3 datasets with official OpenPI JAX.

This script only prepares the dataset view, normalization statistics and run
configuration. Model construction, loss, optimizer, EMA, FSDP sharding and
Orbax checkpoints come unchanged from the pinned OpenPI ``scripts/train.py``.
Run it with the Python interpreter of the OpenPI environment; one process
drives all local GPUs through FSDP (do not wrap it in ``torchrun``).
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Sequence

HERE = Path(__file__).resolve().parent
PINNED_COMMIT = (HERE / "pi05_policy/UPSTREAM_COMMIT").read_text().split()[-1]
CONFIG_NAMES = {"single": "evodex_pi05_single_jax", "dual": "evodex_pi05_dual_jax"}
OFFICIAL_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"
CAMERAS = {
    "single": ("context", "interaction", "wrist_right"),
    "dual": ("context", "wrist_left", "wrist_right"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def check_openpi(openpi_root: Path) -> Path:
    root = openpi_root.expanduser().resolve()
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise ValueError(f"Expected OpenPI {PINNED_COMMIT}, got {commit}.")
    if not (root / "src/openpi/training/evodex_config.py").is_file():
        raise FileNotFoundError(
            f"The Evo-DexBench overlay is not installed in {root}; run install_overlay.py first."
        )
    return root


def inspect_dataset(path: Path, embodiment: str) -> dict[str, Any]:
    """Check that a directory is an Evo-DexBench LeRobot v3 dataset."""

    info_path = path / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"{path} is not a LeRobot dataset: {info_path} is missing.")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    roles = 1 if embodiment == "single" else 2
    features = info.get("features", {})
    problems = []
    if info.get("codebase_version") != "v3.0":
        problems.append(f"codebase_version {info.get('codebase_version')!r} (expected 'v3.0')")
    if int(info.get("fps", 0)) != 30:
        problems.append(f"fps {info.get('fps')} (expected 30)")
    if features.get("observation.state", {}).get("shape") != [111 * roles]:
        problems.append(f"observation.state shape {features.get('observation.state', {}).get('shape')}")
    if features.get("action", {}).get("shape") != [30 * roles]:
        problems.append(f"action shape {features.get('action', {}).get('shape')}")
    for camera in CAMERAS[embodiment]:
        if f"observation.images.{camera}" not in features:
            problems.append(f"missing camera observation.images.{camera}")
    if problems:
        raise ValueError(f"{path} does not match the {embodiment} π0.5 contract: " + "; ".join(problems))
    return {"path": str(path), "frames": int(info["total_frames"]), "episodes": int(info["total_episodes"])}


def samples_per_epoch(frames: Sequence[int], replay_frames: Sequence[int], replay_ratio: float) -> int:
    """Epoch size of the task-balanced sampler in ``lerobot_cpu.py``.

    Primary datasets are weighted equally, so one epoch draws as many samples
    as needed for every primary dataset to be seen once at its weight.
    """

    if len(frames) + len(replay_frames) == 1:
        return int(frames[0])
    primary_weight = (1.0 - replay_ratio) / len(frames)
    return max(int(math.ceil(count / primary_weight)) for count in frames)


def _view_name(index: int, path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", path.name).strip("-") or "dataset"
    return f"{index:02d}-{stem}"


def build_dataset_view(view: Path, paths: Sequence[Path]) -> list[str]:
    """Expose every dataset under one LeRobot home through symbolic links."""

    view.mkdir(parents=True, exist_ok=True)
    names = []
    for index, path in enumerate(paths):
        name = _view_name(index, path)
        link = view / name
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or link.resolve() != path:
                raise FileExistsError(f"{link} already points to a different dataset.")
        else:
            link.symlink_to(path, target_is_directory=True)
        names.append(name)
    return names


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--embodiment", choices=sorted(CONFIG_NAMES), required=True)
    parser.add_argument("--dataset", type=Path, action="append", required=True,
                        help="LeRobot v3 dataset directory; repeat for joint training.")
    parser.add_argument("--replay-dataset", type=Path, action="append", default=[],
                        help="Optional replay dataset; requires --replay-ratio.")
    parser.add_argument("--replay-ratio", type=float, default=0.0)
    parser.add_argument("--dataset-id", required=True,
                        help="Logical name of this data mixture; names the normalization assets.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-params", default=None,
                        help="Initial Orbax params. Default: the official 32D π0.5 base (single only).")
    parser.add_argument("--norm-stats", type=Path, default=None,
                        help="Reuse an existing norm_stats.json instead of computing it.")
    parser.add_argument("--gpus", type=int, default=None, help="FSDP devices (default: all local GPUs).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=25)
    parser.add_argument("--num-train-steps", type=int, default=None,
                        help="Override the update budget derived from --epochs.")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--peak-lr", type=float, default=5e-6)
    parser.add_argument("--min-lr", type=float, default=5e-7)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=("locality", "random", "exact"), default="locality",
                        help="Task-balanced sampling; 'exact' fixes every task's share in every batch.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run with the identical command.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Prepare normalization statistics and the run record, then exit.")
    parser.add_argument("--wandb-project", default=None, help="Optional Weights & Biases project.")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.dataset_id):
        parser.error("--dataset-id may contain only letters, digits, '.', '_' and '-'.")
    if args.replay_dataset and not 0 < args.replay_ratio < 1:
        parser.error("--replay-ratio must be in (0, 1) with --replay-dataset.")
    if not args.replay_dataset and args.replay_ratio:
        parser.error("--replay-ratio requires --replay-dataset.")
    if args.embodiment == "dual" and args.base_params is None:
        parser.error("Dual-arm training needs --base-params from extend_action_head.py.")
    if args.sampling == "exact" and (len(args.dataset) < 2 or args.replay_dataset):
        parser.error("--sampling exact needs at least two --dataset entries and no replay datasets.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    openpi_root = check_openpi(args.openpi_root)
    output = args.output_dir.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=True)
    config_name = CONFIG_NAMES[args.embodiment]

    primary = [inspect_dataset(path.expanduser().resolve(), args.embodiment) for path in args.dataset]
    replay = [inspect_dataset(path.expanduser().resolve(), args.embodiment) for path in args.replay_dataset]
    names = build_dataset_view(output / "dataset_view", [Path(item["path"]) for item in (*primary, *replay)])
    epoch_samples = samples_per_epoch(
        [item["frames"] for item in primary], [item["frames"] for item in replay], args.replay_ratio
    )
    steps_per_epoch = epoch_samples // args.batch_size  # the loader drops the last partial batch
    num_train_steps = args.num_train_steps or int(round(args.epochs * steps_per_epoch))
    if num_train_steps <= 0:
        raise ValueError("The update budget must be positive.")

    prefix = f"EVODEX_{args.embodiment.upper()}"
    base_params = args.base_params or OFFICIAL_BASE
    env = {
        # LeRobot reads HF_LEROBOT_HOME when it is imported.
        "HF_LEROBOT_HOME": str(output / "dataset_view"),
        f"{prefix}_DATASET_REPO_ID": args.dataset_id,
        f"{prefix}_DATASET_REPO_IDS": ",".join(names[: len(primary)]),
        f"{prefix}_REPLAY_REPO_IDS": ",".join(names[len(primary):]),
        f"{prefix}_REPLAY_RATIO": str(args.replay_ratio),
        "EVODEX_ASSETS_BASE_DIR": str(output / "assets"),
        "EVODEX_CHECKPOINT_BASE_DIR": str(output / "checkpoints"),
        "EVODEX_NUM_WORKERS": str(args.num_workers),
        "EVODEX_SAMPLING_MODE": args.sampling,
        "EVODEX_TEMPORAL_BLOCK_SIZE": "128",
        "EVODEX_PREFETCH_FACTOR": "2",
        "EVODEX_REJECT_PROMPT_TRUNCATION": "1",
        "EVODEX_PI05_BASE_JAX" if args.embodiment == "single" else "EVODEX_PI05_BASE_JAX64": base_params,
    }
    os.environ.update(env)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pythonpath = str(openpi_root / "src")
    os.environ["PYTHONPATH"] = pythonpath + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    sys.path.insert(0, pythonpath)

    norm_path = output / "assets" / config_name / args.dataset_id / "norm_stats.json"
    if args.norm_stats is not None:
        source = args.norm_stats.expanduser().resolve()
        if norm_path.exists() and _sha256(norm_path) != _sha256(source):
            raise FileExistsError(f"{norm_path} already holds different statistics.")
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, norm_path)
    elif not norm_path.exists():
        # Run the official script as __main__ so spawned loader workers can
        # import its transforms.
        subprocess.run(
            [sys.executable, str(openpi_root / "scripts/compute_norm_stats.py"), "--config-name", config_name],
            cwd=openpi_root, env=os.environ.copy(), check=True,
        )

    record = {
        "openpi_commit": PINNED_COMMIT,
        "config_name": config_name,
        "embodiment": args.embodiment,
        "dataset_id": args.dataset_id,
        "datasets": primary,
        "replay_datasets": replay,
        "replay_ratio": args.replay_ratio,
        "sampling": args.sampling,
        "samples_per_epoch": epoch_samples,
        "steps_per_epoch": steps_per_epoch,
        "num_train_steps": num_train_steps,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "peak_lr": args.peak_lr,
        "min_lr": args.min_lr,
        "ema_decay": args.ema_decay,
        "seed": args.seed,
        "base_params": base_params,
        "norm_stats_sha256": _sha256(norm_path),
    }
    record_path = output / "run_config.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous != record:
            changed = sorted(key for key in record if previous.get(key) != record[key])
            raise ValueError(f"{record_path} belongs to a different run (changed: {changed}).")
    else:
        _write_json(record_path, record)
    print(json.dumps({key: record[key] for key in ("dataset_id", "samples_per_epoch", "num_train_steps")}))
    if args.prepare_only:
        return
    train(args, openpi_root, output, config_name, num_train_steps, steps_per_epoch)


def train(
    args: argparse.Namespace,
    openpi_root: Path,
    output: Path,
    config_name: str,
    num_train_steps: int,
    steps_per_epoch: int,
) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from etils import epath
    from flax.training import common_utils
    from openpi.training import checkpoints, config, data_loader, optimizer, sharding

    logging.basicConfig(level=logging.INFO, force=True)
    gpus = args.gpus or jax.local_device_count()
    if jax.process_count() != 1 or jax.device_count() != gpus:
        raise ValueError(f"Expected one process controlling {gpus} devices, found {jax.device_count()}.")
    if args.batch_size % gpus:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by {gpus} devices.")
    cfg = dataclasses.replace(
        config.get_config(config_name),
        exp_name=args.dataset_id,
        batch_size=args.batch_size,
        num_train_steps=num_train_steps,
        fsdp_devices=gpus,
        seed=args.seed,
        ema_decay=args.ema_decay,
        save_interval=args.save_interval,
        keep_period=None,
        wandb_enabled=False,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=args.warmup_steps,
            peak_lr=args.peak_lr,
            decay_steps=num_train_steps,
            decay_lr=args.min_lr,
        ),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
    )
    jax.config.update("jax_compilation_cache_dir", str(output / "cache/jax"))
    spec = importlib.util.spec_from_file_location("official_openpi_train", openpi_root / "scripts/train.py")
    assert spec and spec.loader
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, name=args.dataset_id, dir=str(output),
                               config=json.loads((output / "run_config.json").read_text()),
                               resume="allow" if args.resume else None)

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        epath.Path(cfg.checkpoint_dir), keep_period=None, overwrite=False, resume=args.resume
    )
    metrics_path = output / "metrics.jsonl"
    try:
        loader = data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        batch = next(iterator)
        train_rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
        state, state_sharding = official.init_train_state(cfg, init_rng, mesh, resume=resuming)
        if resuming:
            # As in OpenPI, resume restores the train state, not a loader cursor.
            state = checkpoints.restore_state(manager, state, loader)
        jax.block_until_ready(state)
        start_step = int(state.step)
        step_fn = jax.jit(
            functools.partial(official.train_step, cfg),
            in_shardings=(replicated, state_sharding, data_sharding),
            out_shardings=(state_sharding, replicated),
            donate_argnums=(1,),
        )
        lr_fn = cfg.lr_schedule.create()
        infos = []
        last_log = time.perf_counter()
        for step in range(start_step, cfg.num_train_steps):
            with sharding.set_mesh(mesh):
                state, info = step_fn(train_rng, state, batch)
            infos.append(info)
            last = step == cfg.num_train_steps - 1
            if step == start_step or (step + 1) % args.log_interval == 0 or last:
                metrics = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
                values = {key: float(value) for key, value in metrics.items()}
                if not all(np.isfinite(value) for value in values.values()):
                    raise FloatingPointError(f"Non-finite training metrics at update {step + 1}: {values}")
                now = time.perf_counter()
                values.update(
                    learning_rate=float(lr_fn(step)),
                    epoch=(step + 1) / steps_per_epoch,
                    seconds_per_update=(now - last_log) / len(infos),
                )
                line = {"update": step + 1, **values}
                print(json.dumps(line), flush=True)
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(line) + "\n")
                if wandb_run is not None:
                    wandb_run.log(values, step=step + 1)
                infos = []
                last_log = now
            if not last:
                batch = next(iterator)
            # Checkpoint directories are named by the zero-based update index.
            if (step + 1) % cfg.save_interval == 0 or last:
                checkpoints.save_state(manager, state, loader, step)
        manager.wait_until_finished()
        final = Path(cfg.checkpoint_dir) / str(cfg.num_train_steps - 1)
        if not (final / "params/_METADATA").is_file():
            raise FileNotFoundError(f"The final checkpoint was not written: {final}")
        print(json.dumps({"state": "completed", "checkpoint": str(final)}), flush=True)
    finally:
        manager.close()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
