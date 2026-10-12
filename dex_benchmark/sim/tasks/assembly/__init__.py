"""Hand-effector benchmark task implementations."""

from typing import Any

__all__ = ["B09Env", "B10Env"]


def __getattr__(name: str) -> Any:
    # Keep catalog imports free of simulator initialization.
    if name == "B09Env":
        from dex_benchmark.sim.tasks.assembly.b09 import B09Env

        return B09Env
    if name == "B10Env":
        from dex_benchmark.sim.tasks.assembly.b10 import B10Env

        return B10Env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
