from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import numpy as np

from dex_benchmark.sim.scenes.layouts import LayoutLightingSpec


def _values(samples: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    try:
        values = np.asarray(samples[name], dtype=np.float64)
    except KeyError as exc:
        raise KeyError(f"Missing lighting generalization value {name!r}.") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(
            f"Lighting value {name!r} must be a finite one-dimensional array."
        )
    return values


def _scaled_color(
    color_rgb: tuple[float, float, float],
    *,
    intensity_scale: float,
    color_shift: float,
) -> tuple[float, float, float]:
    # Positive coupled gains make positive values warmer and negative values cooler.
    color_gain = np.exp(np.asarray((color_shift, 0.0, -color_shift)))
    color = np.asarray(color_rgb, dtype=np.float64) * intensity_scale * color_gain
    if not np.isfinite(color).all() or (color < 0.0).any():
        raise ValueError("Randomized light colors must be finite and non-negative.")
    return tuple(float(value) for value in color)


def lighting_values(
    nominal: LayoutLightingSpec,
    samples: Mapping[str, np.ndarray],
) -> tuple[LayoutLightingSpec, ...]:
    """Apply three global color/intensity changes without changing light structure."""

    ambient_log_scale = _values(samples, "ambient_light_log_scale")
    direct_log_scale = _values(samples, "direct_light_log_scale")
    color_shift = _values(samples, "light_color_shift")
    batch_size = ambient_log_scale.shape[0]
    if direct_log_scale.shape != (batch_size,) or color_shift.shape != (batch_size,):
        raise ValueError("All lighting values must have the same batch size.")

    specs: list[LayoutLightingSpec] = []
    for env_index in range(batch_size):
        ambient_scale = float(np.exp(ambient_log_scale[env_index]))
        direct_scale = float(np.exp(direct_log_scale[env_index]))
        shift = float(color_shift[env_index])
        specs.append(
            replace(
                nominal,
                ambient_rgb=_scaled_color(
                    nominal.ambient_rgb,
                    intensity_scale=ambient_scale,
                    color_shift=shift,
                ),
                point_lights=tuple(
                    replace(
                        light,
                        color_rgb=_scaled_color(
                            light.color_rgb,
                            intensity_scale=direct_scale,
                            color_shift=shift,
                        ),
                    )
                    for light in nominal.point_lights
                ),
                spot_lights=tuple(
                    replace(
                        light,
                        color_rgb=_scaled_color(
                            light.color_rgb,
                            intensity_scale=direct_scale,
                            color_shift=shift,
                        ),
                    )
                    for light in nominal.spot_lights
                ),
                directional_lights=tuple(
                    replace(
                        light,
                        color_rgb=_scaled_color(
                            light.color_rgb,
                            intensity_scale=direct_scale,
                            color_shift=shift,
                        ),
                    )
                    for light in nominal.directional_lights
                ),
            )
        )
    return tuple(specs)
