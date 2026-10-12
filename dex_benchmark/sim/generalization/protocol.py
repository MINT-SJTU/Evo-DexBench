from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import yaml

from dex_benchmark.sim.generalization.attributes import GeneralizationAxis
from dex_benchmark.resources import generalization_protocol_path


_PROTOCOL_FIELDS = {"version", "axes", "seeds", "max_resample_attempts"}


@dataclass(frozen=True)
class GeneralizationProtocol:
    version: str
    axes: tuple[GeneralizationAxis, ...]
    seeds: tuple[int, ...]
    max_resample_attempts: int

    @staticmethod
    def default_path() -> Path:
        return generalization_protocol_path()

    @classmethod
    def load(cls, path: Path | None = None) -> GeneralizationProtocol:
        override = os.environ.get("EVODEX_GENERALIZATION_PROTOCOL_PATH")
        protocol_path = path or (Path(override) if override else cls.default_path())
        raw_data = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise TypeError(f"Generalization protocol {protocol_path} must be a mapping.")
        unknown_fields = sorted(set(raw_data) - _PROTOCOL_FIELDS)
        if unknown_fields:
            raise ValueError(
                f"Generalization protocol contains unsupported fields: "
                f"{', '.join(unknown_fields)}."
            )
        missing_fields = sorted(_PROTOCOL_FIELDS - set(raw_data))
        if missing_fields:
            raise ValueError(
                f"Generalization protocol is missing fields: {', '.join(missing_fields)}."
            )
        version = raw_data["version"]
        if not isinstance(version, str) or not version:
            raise TypeError("Generalization protocol version must be a non-empty string.")
        raw_axes = raw_data["axes"]
        if not isinstance(raw_axes, list):
            raise TypeError("Generalization protocol axes must be a list.")
        axes = tuple(GeneralizationAxis(axis) for axis in raw_axes)
        if len(axes) != len(set(axes)):
            raise ValueError("Generalization protocol axes must be unique.")
        if set(axes) != set(GeneralizationAxis):
            raise ValueError("Generalization protocol must contain exactly the four fixed axes.")
        raw_seeds = raw_data["seeds"]
        if not isinstance(raw_seeds, list) or not raw_seeds:
            raise TypeError("Generalization protocol seeds must be a non-empty list.")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds):
            raise TypeError("Every generalization seed must be an integer.")
        seeds = tuple(raw_seeds)
        if len(seeds) != len(set(seeds)):
            raise ValueError("Generalization protocol seeds must be unique.")
        max_attempts = raw_data["max_resample_attempts"]
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts <= 0:
            raise ValueError("max_resample_attempts must be a positive integer.")
        return cls(
            version=version,
            axes=axes,
            seeds=seeds,
            max_resample_attempts=max_attempts,
        )
