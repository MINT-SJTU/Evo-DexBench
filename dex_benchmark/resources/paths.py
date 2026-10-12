from __future__ import annotations

from importlib.resources import files
import os
from pathlib import Path


RESOURCE_ROOT_ENV_VAR = "EVODEX_RESOURCE_DIR"


def resource_root() -> Path:
    """Return the root containing packaged benchmark YAML resources."""

    configured_root = os.environ.get(RESOURCE_ROOT_ENV_VAR)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"{RESOURCE_ROOT_ENV_VAR} points to a missing directory: {root}."
            )
        return root
    return Path(str(files("dex_benchmark.resources"))).resolve()


def task_config_dir() -> Path:
    return resource_root() / "tasks"


def task_config_path(task_id: str) -> Path:
    normalized_id = task_id.removesuffix("_v1")
    return task_config_dir() / f"{normalized_id}.yaml"


def evaluation_protocol_path() -> Path:
    return resource_root() / "eval" / "protocol_v1.yaml"


def generalization_protocol_path() -> Path:
    return resource_root() / "generalization" / "protocol_v1.yaml"


def teleoperation_config_path(filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("Teleoperation config filename must not contain directories.")
    return resource_root() / "teleoperation" / filename

