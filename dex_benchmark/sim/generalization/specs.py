from __future__ import annotations

import math

from dex_benchmark.sim.generalization.attributes import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
)


CAMERA_PARAMETER_NAMES = (
    "camera_side",
    "camera_height",
    "camera_distance",
    "camera_target_x",
    "camera_target_y",
    "camera_target_z",
    "camera_fov",
)

CAMERA_RANDOMIZATION_SPEC = AxisRandomizationSpec(
    parameters=tuple(
        GeneralizationParameterSpec(name=name, low=-1.0, high=1.0)
        for name in CAMERA_PARAMETER_NAMES
    ),
    requires_rebuild=True,
)

LIGHTING_PARAMETER_NAMES = (
    "ambient_light_log_scale",
    "direct_light_log_scale",
    "light_color_shift",
)

LIGHTING_RANDOMIZATION_SPEC = AxisRandomizationSpec(
    parameters=(
        GeneralizationParameterSpec(
            name="ambient_light_log_scale",
            low=math.log(0.70),
            high=math.log(1.30),
        ),
        GeneralizationParameterSpec(
            name="direct_light_log_scale",
            low=math.log(0.75),
            high=math.log(1.25),
        ),
        GeneralizationParameterSpec(
            name="light_color_shift",
            low=-0.08,
            high=0.08,
        ),
    ),
    requires_rebuild=True,
)


def global_axis_spec(axis: GeneralizationAxis) -> AxisRandomizationSpec | None:
    if axis is GeneralizationAxis.CAMERA:
        return CAMERA_RANDOMIZATION_SPEC
    if axis is GeneralizationAxis.LIGHTING:
        return LIGHTING_RANDOMIZATION_SPEC
    return None
