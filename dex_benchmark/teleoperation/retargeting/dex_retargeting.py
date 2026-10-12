"""Minimal dex-retargeting integration shared by Quest 3 action builders."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from dex_benchmark.sim.robots import robot_asset_path


DEFAULT_RETARGETING_URDF_ROOT = Path(robot_asset_path())


def build_seq_retargeting(*, config_path: str | Path, urdf_root: str | Path) -> Any:
    try:
        from dex_retargeting.retargeting_config import RetargetingConfig
    except ImportError as exc:
        raise ImportError(
            "Quest 3 hand retargeting requires dex_retargeting. Install the "
            "benchmark VR dependency group first."
        ) from exc

    RetargetingConfig.set_default_urdf_dir(str(Path(urdf_root).expanduser().resolve()))
    config = RetargetingConfig.load_from_file(Path(config_path).expanduser().resolve())
    try:
        return config.build()
    except ValueError as exc:
        if "Unknown tag: capsule" not in str(exc):
            raise
        config.urdf_path = str(_kinematic_only_urdf(Path(config.urdf_path)))
        return config.build()


def indices_by_joint_name(
    *,
    source_joint_names: tuple[str, ...],
    target_joint_names: tuple[str, ...],
) -> np.ndarray:
    missing = [name for name in target_joint_names if name not in source_joint_names]
    if missing:
        raise ValueError(
            "Retargeting output does not contain required ManiSkill joints: "
            f"{missing}."
        )
    return np.asarray(
        [source_joint_names.index(name) for name in target_joint_names],
        dtype=np.int64,
    )


def _kinematic_only_urdf(source_path: Path) -> Path:
    """Create a temporary URDF retaining links and joints but no geometry."""

    if not source_path.exists():
        raise ValueError(f"URDF path does not exist: {source_path}")
    tree = ET.parse(source_path)
    root = tree.getroot()
    removable = {"collision", "visual", "gazebo", "material", "transmission"}
    for parent in root.iter():
        for child in list(parent):
            local_name = child.tag.rsplit("}", maxsplit=1)[-1]
            if local_name in removable:
                parent.remove(child)
    target_dir = Path(tempfile.mkdtemp(prefix="dexbench-retargeting-"))
    target_path = target_dir / source_path.name
    tree.write(target_path, encoding="utf-8", xml_declaration=True)
    return target_path
