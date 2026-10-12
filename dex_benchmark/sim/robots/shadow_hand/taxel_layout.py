from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Final, Literal, cast


HandSide = Literal["right", "left"]
PatchKind = Literal["box", "convex_prism"]

TAXEL_LAYOUT_ID: Final[str] = "shadow_hand_distributed_tactile_123"
TAXEL_LAYOUT_VERSION: Final[int] = 3
TAXEL_FORCE_UNIT: Final[str] = "N"
TAXEL_SENSOR_MODEL: Final[str] = "control_period_impulse_average"
SKIP_SURFACE_FIT_ENV_VAR: Final[str] = (
    "EVO_DEXBENCH_SKIP_TAXEL_SURFACE_FIT"
)

FINGER_NAMES: Final[tuple[str, ...]] = ("th", "ff", "mf", "rf", "lf")
REGION_NAMES: Final[tuple[str, ...]] = (
    "distal",
    "middle",
    "proximal",
    "palm",
    "subpalm",
)
FINGER_INDEX_BY_NAME: Final[dict[str, int]] = {
    name: index for index, name in enumerate(FINGER_NAMES)
}
REGION_INDEX_BY_NAME: Final[dict[str, int]] = {
    name: index for index, name in enumerate(REGION_NAMES)
}


@dataclass(frozen=True)
class TaxelSpec:
    """Static geometry and semantics for one independently queried taxel."""

    name: str
    parent_link_name: str
    local_position_m: tuple[float, float, float]
    local_normal: tuple[float, float, float]
    area_m2: float
    finger_index: int
    region_index: int
    local_index: int
    patch_kind: PatchKind
    patch_size_m: tuple[float, float]
    tangent_u: tuple[float, float, float]

    @property
    def finger_name(self) -> str | None:
        if self.finger_index < 0:
            return None
        return FINGER_NAMES[self.finger_index]

    @property
    def region_name(self) -> str:
        return REGION_NAMES[self.region_index]


def _normalized(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in vector))
    if length <= 0.0:
        raise ValueError("Taxel direction vectors must be non-zero.")
    return (
        vector[0] / length,
        vector[1] / length,
        vector[2] / length,
    )


def _make_taxel(
    *,
    finger_name: str | None,
    region_name: str,
    local_index: int,
    position: tuple[float, float, float],
    normal: tuple[float, float, float],
    patch_size: tuple[float, float],
    tangent_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
    patch_kind: PatchKind = "box",
) -> TaxelSpec:
    prefix = "rh"
    if finger_name is None:
        semantic_name = region_name
        parent_link_name = (
            f"{prefix}_palm"
            if region_name == "palm"
            else f"{prefix}_lfmetacarpal"
        )
        finger_index = -1
    else:
        semantic_name = f"{finger_name}_{region_name}"
        parent_link_name = f"{prefix}_{finger_name}{region_name}"
        finger_index = FINGER_INDEX_BY_NAME[finger_name]

    normal_unit = _normalized(normal)
    tangent_u_unit = _normalized(tangent_u)
    orthogonality = sum(
        normal_component * tangent_component
        for normal_component, tangent_component in zip(
            normal_unit,
            tangent_u_unit,
            strict=True,
        )
    )
    if abs(orthogonality) > 1e-6:
        raise ValueError(
            f"Taxel {semantic_name}_{local_index:02d} normal and tangent_u "
            f"must be orthogonal, got dot product {orthogonality}."
        )

    patch_width, patch_height = patch_size
    return TaxelSpec(
        name=f"{prefix}_taxel_{semantic_name}_{local_index:02d}",
        parent_link_name=parent_link_name,
        local_position_m=position,
        local_normal=normal_unit,
        area_m2=patch_width * patch_height,
        finger_index=finger_index,
        region_index=REGION_INDEX_BY_NAME[region_name],
        local_index=local_index,
        patch_kind=patch_kind,
        patch_size_m=patch_size,
        tangent_u=tangent_u_unit,
    )



def _standard_distal_taxels(finger_name: str) -> list[TaxelSpec]:
    """Create the 8-taxel distal layout used by the other three fingers."""

    taxels = [
        _make_taxel(
            finger_name=finger_name,
            region_name="distal",
            local_index=0,
            position=(0.0, -0.0072, 0.0100),
            normal=(0.0, -1.0, 0.0),
            patch_size=(0.0085, 0.0055),
            patch_kind="convex_prism",
        )
    ]

    for x_position in (-0.0023, 0.0023):
        taxels.append(
            _make_taxel(
                finger_name=finger_name,
                region_name="distal",
                local_index=len(taxels),
                position=(x_position, -0.0072, 0.0215),
                normal=(0.0, -1.0, 0.0),
                patch_size=(0.0043, 0.0080),
                patch_kind="convex_prism",
            )
        )

    taxels.append(
        _make_taxel(
            finger_name=finger_name,
            region_name="distal",
            local_index=len(taxels),
            position=(0.0, -0.0040, 0.0298),
            normal=(0.0, -0.55, 0.835),
            patch_size=(0.0065, 0.0040),
            patch_kind="convex_prism",
        )
    )

    for x_position in (-0.0024, 0.0024):
        taxels.append(
            _make_taxel(
                finger_name=finger_name,
                region_name="distal",
                local_index=len(taxels),
                position=(x_position, -0.0015, 0.0323),
                normal=(0.0, -0.20, 0.98),
                patch_size=(0.0045, 0.0035),
                patch_kind="convex_prism",
            )
        )

    for side_sign in (-1.0, 1.0):
        taxels.append(
            _make_taxel(
                finger_name=finger_name,
                region_name="distal",
                local_index=len(taxels),
                position=(side_sign * 0.0084, -0.0030, 0.0205),
                normal=(0.82 * side_sign, -0.573, 0.0),
                patch_size=(0.0085, 0.0045),
                tangent_u=(0.0, 0.0, 1.0),
                patch_kind="convex_prism",
            )
        )
    return taxels


def _middle_taxels(finger_name: str) -> list[TaxelSpec]:
    y_surface = -0.0095 if finger_name == "th" else -0.0050
    x_offset = 0.0040 if finger_name == "th" else 0.00325
    z_positions = (0.0120, 0.0270) if finger_name == "th" else (0.0060, 0.0215)
    return [
        _make_taxel(
            finger_name=finger_name,
            region_name="middle",
            local_index=index,
            position=(x_position, y_surface, z_position),
            normal=(0.0, -1.0, 0.0),
            patch_size=(0.0063, 0.0135),
        )
        for index, (z_position, x_position) in enumerate(
            (z_position, x_position)
            for z_position in z_positions
            for x_position in (-x_offset, x_offset)
        )
    ]


def _proximal_taxels(finger_name: str, count: int) -> list[TaxelSpec]:
    y_surface = -0.0115 if finger_name == "th" else -0.0050
    if count == 3:
        positions = (
            (-0.0038, y_surface, 0.0170),
            (0.0038, y_surface, 0.0170),
            (0.0, y_surface, 0.0320),
        )
        patch_size = (0.0068, 0.0120)
    else:
        positions = (
            (0.0, y_surface, 0.0170),
            (0.0, y_surface, 0.0320),
        )
        patch_size = (0.0105, 0.0120)
    return [
        _make_taxel(
            finger_name=finger_name,
            region_name="proximal",
            local_index=index,
            position=position,
            normal=(0.0, -1.0, 0.0),
            patch_size=patch_size,
        )
        for index, position in enumerate(positions)
    ]


def _palm_taxels() -> list[TaxelSpec]:
    x_positions = (-0.0140, -0.0020, 0.0100)
    z_positions = (0.0065, 0.0250, 0.0435, 0.0620, 0.0805)
    return [
        _make_taxel(
            finger_name=None,
            region_name="palm",
            local_index=index,
            position=(x_position, -0.0120, z_position),
            normal=(0.0, -1.0, 0.0),
            patch_size=(0.0115, 0.0180),
        )
        for index, (z_position, x_position) in enumerate(
            (z_position, x_position)
            for z_position in z_positions
            for x_position in x_positions
        )
    ]



def candidate_right_taxel_layout() -> tuple[TaxelSpec, ...]:
    """Return the canonical 123-taxel layout before surface fitting."""

    taxels: list[TaxelSpec] = []
    for finger_name in FINGER_NAMES:
        if finger_name == "th":
            taxels.extend(_balanced_thumb_distal_taxels())
            taxels.extend(_balanced_thumb_middle_taxels())
            taxels.extend(_balanced_thumb_proximal_taxels())
            continue
        if finger_name == "ff":
            taxels.extend(_balanced_index_distal_taxels())
        elif finger_name in ("mf", "rf"):
            taxels.extend(_enhanced_standard_distal_taxels(finger_name))
        else:
            taxels.extend(_standard_distal_taxels(finger_name))
        taxels.extend(_middle_taxels(finger_name))
        taxels.extend(
            _proximal_taxels(
                finger_name,
                count=3 if finger_name == "ff" else 2,
            )
        )
    taxels.extend(_balanced_palm_taxels())
    taxels.extend(_balanced_subpalm_taxels())
    return tuple(taxels)


def _balanced_thumb_distal_taxels() -> list[TaxelSpec]:
    """Create the 16-taxel thumb tip used by the balanced 123 layout."""

    x_scale = 1.15
    y_surface = -0.0087
    taxels: list[TaxelSpec] = [
        _make_taxel(
            finger_name="th",
            region_name="distal",
            local_index=0,
            position=(0.0, y_surface, 0.0115),
            normal=(0.0, -1.0, 0.0),
            patch_size=(0.0105 * x_scale, 0.0040),
            patch_kind="convex_prism",
        )
    ]

    # Three face rows retain fine pad localization without over-concentrating
    # taxels on the thumb tip.
    for z_position in (0.0160, 0.0220, 0.0280):
        for x_position in (-0.0030 * x_scale, 0.0030 * x_scale):
            taxels.append(
                _make_taxel(
                    finger_name="th",
                    region_name="distal",
                    local_index=len(taxels),
                    position=(x_position, y_surface, z_position),
                    normal=(0.0, -1.0, 0.0),
                    patch_size=(0.0056 * x_scale, 0.0044),
                    patch_kind="convex_prism",
                )
            )

    for x_position in (-0.0030 * x_scale, 0.0030 * x_scale):
        taxels.append(
            _make_taxel(
                finger_name="th",
                region_name="distal",
                local_index=len(taxels),
                position=(x_position, -0.0048, 0.0325),
                normal=(0.0, -0.55, 0.835),
                patch_size=(0.0048 * x_scale, 0.0038),
                tangent_u=(1.0, 0.0, 0.0),
                patch_kind="convex_prism",
            )
        )

    for side_sign in (-1.0, 1.0):
        taxels.append(
            _make_taxel(
                finger_name="th",
                region_name="distal",
                local_index=len(taxels),
                position=(side_sign * 0.0068, -0.0028, 0.0310),
                normal=(0.55 * side_sign, -0.35, 0.76),
                patch_size=(0.0046 * x_scale, 0.0038),
                tangent_u=(0.35, 0.55 * side_sign, 0.0),
                patch_kind="convex_prism",
            )
        )

    # One centered cap replaces the closely spaced terminal pair.
    taxels.append(
        _make_taxel(
            finger_name="th",
            region_name="distal",
            local_index=len(taxels),
            position=(0.0, -0.0015, 0.0350),
            normal=(0.0, -0.20, 0.98),
            patch_size=(0.0070, 0.0035),
            tangent_u=(1.0, 0.0, 0.0),
            patch_kind="convex_prism",
        )
    )

    for side_sign in (-1.0, 1.0):
        side_normal = _normalized((0.82 * side_sign, -0.573, 0.0))
        for z_position in (0.0160, 0.0245):
            taxels.append(
                _make_taxel(
                    finger_name="th",
                    region_name="distal",
                    local_index=len(taxels),
                    position=(side_sign * 0.0104, -0.0037, z_position),
                    normal=side_normal,
                    patch_size=(0.0065, 0.0043),
                    tangent_u=(0.0, 0.0, 1.0),
                    patch_kind="convex_prism",
                )
            )
    return taxels


def _balanced_index_distal_taxels() -> list[TaxelSpec]:
    """Create the medium-density 14-taxel index fingertip."""

    taxels: list[TaxelSpec] = [
        _make_taxel(
            finger_name="ff",
            region_name="distal",
            local_index=0,
            position=(0.0, -0.0072, 0.0095),
            normal=(0.0, -1.0, 0.0),
            patch_size=(0.0105, 0.0040),
            patch_kind="convex_prism",
        )
    ]
    for z_position in (0.0140, 0.0205, 0.0270):
        for x_position in (-0.0030, 0.0030):
            taxels.append(
                _make_taxel(
                    finger_name="ff",
                    region_name="distal",
                    local_index=len(taxels),
                    position=(x_position, -0.0072, z_position),
                    normal=(0.0, -1.0, 0.0),
                    patch_size=(0.0056, 0.0044),
                    patch_kind="convex_prism",
                )
            )

    taxels.append(
        _make_taxel(
            finger_name="ff",
            region_name="distal",
            local_index=len(taxels),
            position=(0.0, -0.0040, 0.0300),
            normal=(0.0, -0.55, 0.835),
            patch_size=(0.0068, 0.0038),
            tangent_u=(1.0, 0.0, 0.0),
            patch_kind="convex_prism",
        )
    )
    for side_sign in (-1.0, 1.0):
        taxels.append(
            _make_taxel(
                finger_name="ff",
                region_name="distal",
                local_index=len(taxels),
                position=(side_sign * 0.0059, -0.0023, 0.0288),
                normal=(0.55 * side_sign, -0.35, 0.76),
                patch_size=(0.0046, 0.0038),
                tangent_u=(0.35, 0.55 * side_sign, 0.0),
                patch_kind="convex_prism",
            )
        )
    for x_position in (-0.0025, 0.0025):
        taxels.append(
            _make_taxel(
                finger_name="ff",
                region_name="distal",
                local_index=len(taxels),
                position=(x_position, -0.0015, 0.0323),
                normal=(0.0, -0.20, 0.98),
                patch_size=(0.0048, 0.0035),
                tangent_u=(1.0, 0.0, 0.0),
                patch_kind="convex_prism",
            )
        )
    for side_sign in (-1.0, 1.0):
        taxels.append(
            _make_taxel(
                finger_name="ff",
                region_name="distal",
                local_index=len(taxels),
                position=(side_sign * 0.0084, -0.0030, 0.0210),
                normal=(0.82 * side_sign, -0.573, 0.0),
                patch_size=(0.0080, 0.0043),
                tangent_u=(0.0, 0.0, 1.0),
                patch_kind="convex_prism",
            )
        )
    return taxels


def _enhanced_standard_distal_taxels(
    finger_name: Literal["mf", "rf"],
) -> list[TaxelSpec]:
    """Create a 12-taxel distal layout with two face and side rows."""

    taxels: list[TaxelSpec] = [
        _make_taxel(
            finger_name=finger_name,
            region_name="distal",
            local_index=0,
            position=(0.0, -0.0072, 0.0095),
            normal=(0.0, -1.0, 0.0),
            patch_size=(0.0085, 0.0050),
            patch_kind="convex_prism",
        )
    ]
    for z_position in (0.0170, 0.0245):
        for x_position in (-0.0023, 0.0023):
            taxels.append(
                _make_taxel(
                    finger_name=finger_name,
                    region_name="distal",
                    local_index=len(taxels),
                    position=(x_position, -0.0072, z_position),
                    normal=(0.0, -1.0, 0.0),
                    patch_size=(0.0043, 0.0063),
                    patch_kind="convex_prism",
                )
            )

    taxels.append(
        _make_taxel(
            finger_name=finger_name,
            region_name="distal",
            local_index=len(taxels),
            position=(0.0, -0.0040, 0.0298),
            normal=(0.0, -0.55, 0.835),
            patch_size=(0.0065, 0.0040),
            patch_kind="convex_prism",
        )
    )
    for x_position in (-0.0024, 0.0024):
        taxels.append(
            _make_taxel(
                finger_name=finger_name,
                region_name="distal",
                local_index=len(taxels),
                position=(x_position, -0.0015, 0.0323),
                normal=(0.0, -0.20, 0.98),
                patch_size=(0.0045, 0.0035),
                patch_kind="convex_prism",
            )
        )
    for side_sign in (-1.0, 1.0):
        for z_position in (0.0160, 0.0250):
            taxels.append(
                _make_taxel(
                    finger_name=finger_name,
                    region_name="distal",
                    local_index=len(taxels),
                    position=(side_sign * 0.0084, -0.0030, z_position),
                    normal=(0.82 * side_sign, -0.573, 0.0),
                    patch_size=(0.0070, 0.0040),
                    tangent_u=(0.0, 0.0, 1.0),
                    patch_kind="convex_prism",
                )
            )
    return taxels


def _balanced_thumb_middle_taxels() -> list[TaxelSpec]:
    taxels = _middle_taxels("th")
    for side_sign in (-1.0, 1.0):
        for z_position in (0.0120, 0.0270):
            taxels.append(
                _make_taxel(
                    finger_name="th",
                    region_name="middle",
                    local_index=len(taxels),
                    position=(side_sign * 0.0093, -0.0030, z_position),
                    normal=(0.82 * side_sign, -0.573, 0.0),
                    patch_size=(0.0065, 0.0140),
                    tangent_u=(0.573, 0.82 * side_sign, 0.0),
                )
            )
    return taxels


def _balanced_thumb_proximal_taxels() -> list[TaxelSpec]:
    taxels = _proximal_taxels("th", count=3)
    centered_front_taxel = taxels[2]
    centered_front_patch_size = (0.0140, 0.0120)
    taxels[2] = replace(
        centered_front_taxel,
        patch_size_m=centered_front_patch_size,
        area_m2=math.prod(centered_front_patch_size),
    )
    for side_sign in (-1.0, 1.0):
        taxels.append(
            _make_taxel(
                finger_name="th",
                region_name="proximal",
                local_index=len(taxels),
                position=(side_sign * 0.0110, -0.0040, 0.0230),
                normal=(0.80 * side_sign, -0.60, 0.0),
                patch_size=(0.0080, 0.0240),
                tangent_u=(0.60, 0.80 * side_sign, 0.0),
            )
        )
    return taxels


def _balanced_palm_taxels() -> list[TaxelSpec]:
    """Return the central 3x5 grid plus two taxels at three palm corners."""

    taxels = _palm_taxels()
    thenar_yaw_rad = -0.48
    thenar_cosine = math.cos(thenar_yaw_rad)
    thenar_sine = math.sin(thenar_yaw_rad)
    thenar_normal = (thenar_sine, -thenar_cosine, 0.0)
    thenar_tangent = (thenar_cosine, thenar_sine, 0.0)
    thenar_collision_center = (0.0125, -0.0150)
    thenar_collision_half_depth_m = 0.0025
    thenar_surface_y = (
        thenar_collision_center[1]
        + thenar_collision_half_depth_m * thenar_normal[1]
    )
    # Move only across the palm towards the thumb-side edge. Keep both Z
    # coordinates unchanged so this does not move the pair towards the wrist.
    thenar_taxel_center_x = 0.0260
    corner_taxels = (
        (
            (-0.0320, -0.0120, 0.0000),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0160, 0.0170),
        ),
        (
            (-0.0320, -0.0120, 0.0200),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0160, 0.0170),
        ),
        (
            (thenar_taxel_center_x, thenar_surface_y, 0.0000),
            thenar_normal,
            thenar_tangent,
            (0.0180, 0.0170),
        ),
        (
            (thenar_taxel_center_x, thenar_surface_y, 0.0200),
            thenar_normal,
            thenar_tangent,
            (0.0180, 0.0170),
        ),
        (
            (0.0290, -0.0120, 0.0680),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0120, 0.0140),
        ),
        (
            (0.0290, -0.0120, 0.0840),
            (0.0, -1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0120, 0.0140),
        ),
    )
    for position, normal, tangent_u, patch_size in corner_taxels:
        taxels.append(
            _make_taxel(
                finger_name=None,
                region_name="palm",
                local_index=len(taxels),
                position=position,
                normal=normal,
                patch_size=patch_size,
                tangent_u=tangent_u,
            )
        )
    return taxels


def _balanced_subpalm_taxels() -> list[TaxelSpec]:
    """Return two taxels at the moving hypothenar palm corner."""

    rotation_y = 0.9599
    cosine = math.cos(rotation_y)
    sine = math.sin(rotation_y)
    center = (0.026986375, -0.0100, 0.0190)
    taxels: list[TaxelSpec] = []
    for local_z in (-0.0100, 0.0100):
        local_x = 0.0
        rotated_x = cosine * local_x + sine * local_z
        rotated_z = -sine * local_x + cosine * local_z
        taxels.append(
            _make_taxel(
                finger_name=None,
                region_name="subpalm",
                local_index=len(taxels),
                position=(
                    center[0] + rotated_x,
                    center[1],
                    center[2] + rotated_z,
                ),
                normal=(0.0, -1.0, 0.0),
                # The collision is 18 mm wide and 40 mm long. These sheets
                # cover its lower and upper halves with a 2 mm center gap.
                patch_size=(0.0160, 0.0180),
                tangent_u=(cosine, 0.0, -sine),
            )
        )
    return taxels


def _mirror_taxel(spec: TaxelSpec) -> TaxelSpec:
    position_x, position_y, position_z = spec.local_position_m
    normal_x, normal_y, normal_z = spec.local_normal
    tangent_x, tangent_y, tangent_z = spec.tangent_u
    return replace(
        spec,
        name=spec.name.replace("rh_", "lh_", 1),
        parent_link_name=spec.parent_link_name.replace("rh_", "lh_", 1),
        local_position_m=(-position_x, position_y, position_z),
        local_normal=(-normal_x, normal_y, normal_z),
        tangent_u=(-tangent_x, tangent_y, tangent_z),
    )


def validate_taxel_layout(
    layout: tuple[TaxelSpec, ...],
    *,
    expected_count: int = 123,
) -> None:
    if expected_count <= 0:
        raise ValueError(
            f"expected_count must be positive, got {expected_count}."
        )
    if len(layout) != expected_count:
        raise ValueError(
            f"Expected {expected_count} taxels, got {len(layout)}."
        )
    names = tuple(spec.name for spec in layout)
    if len(set(names)) != len(names):
        raise ValueError("Taxel names must be unique.")

    semantic_indices = tuple(
        (spec.finger_index, spec.region_index, spec.local_index) for spec in layout
    )
    if len(set(semantic_indices)) != len(semantic_indices):
        raise ValueError("Taxel semantic indices must be unique.")

    for spec in layout:
        normal_length = math.sqrt(sum(value * value for value in spec.local_normal))
        tangent_length = math.sqrt(sum(value * value for value in spec.tangent_u))
        normal_tangent_dot = sum(
            normal_value * tangent_value
            for normal_value, tangent_value in zip(
                spec.local_normal,
                spec.tangent_u,
                strict=True,
            )
        )
        if not math.isclose(normal_length, 1.0, abs_tol=1e-6):
            raise ValueError(f"Taxel {spec.name!r} has a non-unit normal.")
        if not math.isclose(tangent_length, 1.0, abs_tol=1e-6):
            raise ValueError(f"Taxel {spec.name!r} has a non-unit tangent.")
        if not math.isclose(normal_tangent_dot, 0.0, abs_tol=1e-6):
            raise ValueError(
                f"Taxel {spec.name!r} normal and tangent are not orthogonal."
            )
        if spec.area_m2 <= 0.0:
            raise ValueError(f"Taxel {spec.name!r} area must be positive.")
        if spec.finger_index not in (-1, 0, 1, 2, 3, 4):
            raise ValueError(f"Taxel {spec.name!r} has an invalid finger index.")
        if spec.region_index not in range(len(REGION_NAMES)):
            raise ValueError(f"Taxel {spec.name!r} has an invalid region index.")


TAXEL_SURFACE_FIT_PATH: Final[Path] = (
    Path(__file__).parents[1]
    / "assets"
    / "shadowhand"
    / "tactile"
    / "taxel_surface_fit_123.json"
)


def _float_tuple(
    value: object,
    *,
    field_name: str,
    expected_length: int,
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(
            f"Surface-fit field {field_name!r} must contain "
            f"{expected_length} values."
        )
    return tuple(float(component) for component in value)


def _apply_surface_fit(
    candidate_layout: tuple[TaxelSpec, ...],
    *,
    surface_fit_path: Path = TAXEL_SURFACE_FIT_PATH,
) -> tuple[TaxelSpec, ...]:
    payload = json.loads(surface_fit_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported tactile surface-fit schema version "
            f"{payload.get('schema_version')!r}."
        )
    raw_taxels = payload.get("taxels")
    if not isinstance(raw_taxels, list):
        raise ValueError("Surface-fit metadata must contain a taxels list.")
    fitted_by_name: dict[str, dict[str, object]] = {}
    for raw_taxel in raw_taxels:
        if not isinstance(raw_taxel, dict):
            raise ValueError("Each surface-fit taxel must be a mapping.")
        name = raw_taxel.get("name")
        if not isinstance(name, str) or name in fitted_by_name:
            raise ValueError(
                f"Surface-fit taxel names must be unique strings, got {name!r}."
            )
        fitted_by_name[name] = raw_taxel

    candidate_names = tuple(spec.name for spec in candidate_layout)
    if set(fitted_by_name) != set(candidate_names):
        missing_names = sorted(set(candidate_names) - set(fitted_by_name))
        extra_names = sorted(set(fitted_by_name) - set(candidate_names))
        raise ValueError(
            "Surface-fit taxels do not match the canonical candidate layout: "
            f"missing={missing_names}, extra={extra_names}."
        )

    fitted_layout: list[TaxelSpec] = []
    for spec in candidate_layout:
        raw_taxel = fitted_by_name[spec.name]
        local_position_m = cast(
            tuple[float, float, float],
            _float_tuple(
                raw_taxel.get("local_position_m"),
                field_name=f"{spec.name}.local_position_m",
                expected_length=3,
            ),
        )
        local_normal = cast(
            tuple[float, float, float],
            _float_tuple(
                raw_taxel.get("local_normal"),
                field_name=f"{spec.name}.local_normal",
                expected_length=3,
            ),
        )
        tangent_u = cast(
            tuple[float, float, float],
            _float_tuple(
                raw_taxel.get("tangent_u"),
                field_name=f"{spec.name}.tangent_u",
                expected_length=3,
            ),
        )
        patch_size_m = cast(
            tuple[float, float],
            _float_tuple(
                raw_taxel.get("patch_size_m"),
                field_name=f"{spec.name}.patch_size_m",
                expected_length=2,
            ),
        )
        fitted_layout.append(
            replace(
                spec,
                local_position_m=local_position_m,
                local_normal=local_normal,
                tangent_u=tangent_u,
                patch_size_m=patch_size_m,
                area_m2=patch_size_m[0] * patch_size_m[1],
            )
        )
    return tuple(fitted_layout)


_CANDIDATE_RIGHT_TAXEL_LAYOUT = candidate_right_taxel_layout()
RIGHT_TAXEL_LAYOUT: Final[tuple[TaxelSpec, ...]] = (
    _CANDIDATE_RIGHT_TAXEL_LAYOUT
    if os.environ.get(SKIP_SURFACE_FIT_ENV_VAR) == "1"
    else _apply_surface_fit(_CANDIDATE_RIGHT_TAXEL_LAYOUT)
)
LEFT_TAXEL_LAYOUT: Final[tuple[TaxelSpec, ...]] = tuple(
    _mirror_taxel(spec) for spec in RIGHT_TAXEL_LAYOUT
)
validate_taxel_layout(RIGHT_TAXEL_LAYOUT)
validate_taxel_layout(LEFT_TAXEL_LAYOUT)


def taxel_layout(side: HandSide) -> tuple[TaxelSpec, ...]:
    if side == "right":
        return RIGHT_TAXEL_LAYOUT
    if side == "left":
        return LEFT_TAXEL_LAYOUT
    raise ValueError(f"Unsupported hand side {side!r}.")


def _semantic_layout_payload(
    layout: tuple[TaxelSpec, ...],
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for spec in layout:
        item = asdict(spec)
        item["name"] = spec.name.removeprefix("rh_")
        item["parent_link_name"] = spec.parent_link_name.removeprefix("rh_")
        payload.append(item)
    return payload


TAXEL_LAYOUT_HASH: Final[str] = hashlib.sha256(
    json.dumps(
        {
            "layout_id": TAXEL_LAYOUT_ID,
            "layout_version": TAXEL_LAYOUT_VERSION,
            "taxels": _semantic_layout_payload(RIGHT_TAXEL_LAYOUT),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()

RIGHT_TAXEL_LINK_NAMES: Final[tuple[str, ...]] = tuple(
    spec.name for spec in RIGHT_TAXEL_LAYOUT
)
LEFT_TAXEL_LINK_NAMES: Final[tuple[str, ...]] = tuple(
    spec.name for spec in LEFT_TAXEL_LAYOUT
)
TAXEL_FINGER_INDICES: Final[tuple[int, ...]] = tuple(
    spec.finger_index for spec in RIGHT_TAXEL_LAYOUT
)
TAXEL_REGION_INDICES: Final[tuple[int, ...]] = tuple(
    spec.region_index for spec in RIGHT_TAXEL_LAYOUT
)
DISTAL_TAXEL_MASK: Final[tuple[bool, ...]] = tuple(
    spec.region_name == "distal" for spec in RIGHT_TAXEL_LAYOUT
)
