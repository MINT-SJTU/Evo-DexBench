from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


EVALUATION_SPATIAL_SCALE = 1.5
COLLECTION_PHYSICAL_SCALE = 2.0 / 3.0


class GeneralizationAxis(str, Enum):
    SPATIAL = "spatial"
    CAMERA = "camera"
    LIGHTING = "lighting"
    PHYSICAL = "physical"


class GeneralizationProfile(str, Enum):
    """Range profile used for benchmark evaluation or data collection."""

    NONE = "none"
    EVALUATION = "evaluation"
    COLLECTION = "collection"


@dataclass(frozen=True)
class GeneralizationParameterSpec:
    """One bounded scalar used to build a valid generalization value."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Generalization parameter names must be non-empty.")
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError(
                f"Generalization parameter {self.name!r} bounds must be finite."
            )
        if self.low > self.high:
            raise ValueError(
                f"Generalization parameter {self.name!r} has low={self.low} "
                f"greater than high={self.high}."
            )


@dataclass(frozen=True)
class AxisRandomizationSpec:
    parameters: tuple[GeneralizationParameterSpec, ...] = ()
    requires_rebuild: bool = True

    def __post_init__(self) -> None:
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("Generalization parameter names must be unique within an axis.")


@dataclass(frozen=True)
class TaskGeneralizationSpec:
    """Task-owned axes.

    Camera and lighting use global specs in the environment base class. Their
    fields remain here for configuration compatibility, but task values are not
    used for those two axes.
    """

    spatial: AxisRandomizationSpec | None = None
    camera: AxisRandomizationSpec | None = None
    lighting: AxisRandomizationSpec | None = None
    physical: AxisRandomizationSpec | None = None

    def for_axis(self, axis: GeneralizationAxis) -> AxisRandomizationSpec | None:
        return getattr(self, axis.value)

    def for_profile(self, profile: GeneralizationProfile) -> "TaskGeneralizationSpec":
        """Return task ranges transformed for one benchmark profile.

        The task modules keep one nominal set of bounds. Evaluation expands
        position/orientation offsets, while collection keeps the nominal
        spatial range and narrows physical variation around its midpoint (or
        around zero for log-scale parameters).
        """

        if not isinstance(profile, GeneralizationProfile):
            profile = GeneralizationProfile(profile)
        if profile is GeneralizationProfile.NONE:
            return self
        if profile is GeneralizationProfile.EVALUATION:
            return TaskGeneralizationSpec(
                spatial=_transform_spatial_for_evaluation(self.spatial),
                camera=self.camera,
                lighting=self.lighting,
                physical=self.physical,
            )
        if profile is GeneralizationProfile.COLLECTION:
            return TaskGeneralizationSpec(
                spatial=self.spatial,
                camera=self.camera,
                lighting=self.lighting,
                physical=_transform_physical_for_collection(self.physical),
            )
        raise ValueError(f"Unsupported generalization profile: {profile!r}.")


def _transform_spatial_for_evaluation(
    spec: AxisRandomizationSpec | None,
) -> AxisRandomizationSpec | None:
    if spec is None:
        return None
    parameters = tuple(
        GeneralizationParameterSpec(
            parameter.name,
            parameter.low * EVALUATION_SPATIAL_SCALE,
            parameter.high * EVALUATION_SPATIAL_SCALE,
        )
        if parameter.name.endswith(("_offset_m", "_offset_rad"))
        else parameter
        for parameter in spec.parameters
    )
    return AxisRandomizationSpec(
        parameters=parameters,
        requires_rebuild=spec.requires_rebuild,
    )


def _transform_physical_for_collection(
    spec: AxisRandomizationSpec | None,
) -> AxisRandomizationSpec | None:
    if spec is None:
        return None
    parameters: list[GeneralizationParameterSpec] = []
    for parameter in spec.parameters:
        if parameter.name.endswith("_log_scale"):
            lower_scale = math.exp(parameter.low)
            upper_scale = math.exp(parameter.high)
            lower = math.log(1.0 + (lower_scale - 1.0) * COLLECTION_PHYSICAL_SCALE)
            upper = math.log(1.0 + (upper_scale - 1.0) * COLLECTION_PHYSICAL_SCALE)
            parameters.append(
                GeneralizationParameterSpec(parameter.name, lower, upper)
            )
            continue
        if (
            parameter.low == 0.0 and parameter.high >= 0.0
        ):
            center = 0.0
        else:
            center = (parameter.low + parameter.high) / 2.0
        parameters.append(
            GeneralizationParameterSpec(
                parameter.name,
                center + (parameter.low - center) * COLLECTION_PHYSICAL_SCALE,
                center + (parameter.high - center) * COLLECTION_PHYSICAL_SCALE,
            )
        )
    return AxisRandomizationSpec(
        parameters=tuple(parameters),
        requires_rebuild=spec.requires_rebuild,
    )
