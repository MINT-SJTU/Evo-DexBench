from __future__ import annotations

import math

import numpy as np

from dex_benchmark.sim.scenes.layouts.spec import PhysicalMaterialSpec


def positive_scale(log_value: np.ndarray) -> np.ndarray:
    """Map bounded log values to finite positive scales."""

    values = np.asarray(log_value, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Log-scale generalization values must be finite.")
    scales = np.exp(values)
    if not np.isfinite(scales).all() or (scales <= 0.0).any():
        raise ValueError("Generalization scales must be finite and positive.")
    return scales


def scaled_physical_material(
    nominal: PhysicalMaterialSpec,
    *,
    friction_scale: float,
    restitution: float,
) -> PhysicalMaterialSpec:
    """Scale friction while preserving the nominal dynamic/static ratio."""

    if not math.isfinite(friction_scale) or friction_scale <= 0.0:
        raise ValueError(
            f"friction_scale must be finite and positive, got {friction_scale}."
        )
    if not math.isfinite(restitution) or not 0.0 <= restitution <= 1.0:
        raise ValueError(
            f"restitution must be finite and in [0, 1], got {restitution}."
        )
    return PhysicalMaterialSpec(
        static_friction=nominal.static_friction * friction_scale,
        dynamic_friction=nominal.dynamic_friction * friction_scale,
        restitution=restitution,
    )
