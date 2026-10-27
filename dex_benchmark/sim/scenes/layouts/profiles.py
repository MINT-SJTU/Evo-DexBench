"""Task-scoped scene physics, shared by collection, replay and evaluation."""

from dataclasses import replace

import numpy as np

from dex_benchmark.sim.scenes.layouts.spec import (
    ArticulatedAssetSpec,
    LayoutSpec,
    PoseSpec,
)


STUDIO_BUTTON_PRESS_PHYSICS_VERSION = "studio_button_press_v1"


def studio_button_press_layout(layout: LayoutSpec) -> LayoutSpec:
    """Preserve the M01/M03 collection physics without changing M07's mouse.

    The shared Studio mouse bounds were subsequently corrected for M07. M01
    and M03 retain their original placement and post-settle velocities as part
    of their task definition, including when evaluated without a replay recipe.
    Registry layouts are never mutated.
    """
    if layout.physics_version == STUDIO_BUTTON_PRESS_PHYSICS_VERSION:
        return layout

    mouse = next((item for item in layout.build_items if item.name == "mouse"), None)
    if not isinstance(mouse, ArticulatedAssetSpec):
        raise TypeError(
            "The Studio button-press profile requires an articulated mouse."
        )
    placement = layout.relative_placements["mouse"]
    surface = layout.support_surfaces[placement.surface_id]
    bbox_pose = placement.bbox_center_pose
    # These are collection-time model bounds, not a correction to the asset.
    bbox_center = np.asarray((-0.0021035, 0.0885075, -0.0552555), dtype=float)
    half_extent = (
        0.5 * mouse.scale * np.asarray((0.950039, 0.505937, 1.906169), dtype=float)
    )
    height = float(np.abs(bbox_pose.rotation_matrix[2]) @ half_extent)
    bbox_pose = replace(
        bbox_pose,
        position_xyz=(
            bbox_pose.position_xyz[0],
            bbox_pose.position_xyz[1],
            height + placement.initial_clearance_z,
        ),
    )
    world_bbox_pose = surface.pose.transform_pose(bbox_pose)
    offset = world_bbox_pose.rotation_matrix @ (mouse.scale * bbox_center)
    mouse_pose = PoseSpec(
        position_xyz=tuple(
            world_bbox_pose.position_xyz[i] - float(offset[i]) for i in range(3)
        ),
        quaternion_wxyz=world_bbox_pose.quaternion_wxyz,
    )
    return replace(
        layout,
        build_items=[
            replace(item, pose=mouse_pose) if item.name == "mouse" else item
            for item in layout.build_items
        ],
        relative_placements={
            **layout.relative_placements,
            "mouse": replace(placement, bbox_center_pose=bbox_pose),
        },
        settle_resets_velocity=False,
        physics_version=STUDIO_BUTTON_PRESS_PHYSICS_VERSION,
    )
