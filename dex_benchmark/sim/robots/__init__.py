from __future__ import annotations

from importlib import resources

ROBOT_ASSET_PACKAGE = "dex_benchmark.sim.robots.assets"


def robot_asset_path(*parts: str) -> str:
    asset_path = resources.files(ROBOT_ASSET_PACKAGE).joinpath(*parts)
    return str(asset_path)


__all__ = [
    "ROBOT_ASSET_PACKAGE",
    "robot_asset_path",
]
