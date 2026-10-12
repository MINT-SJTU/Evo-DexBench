from __future__ import annotations

from pathlib import Path

from dex_benchmark.sim.scenes.layouts.spec import (
    PoseSpec,
    StaticBoxSpec,
    TexturedPlaneSpec,
    yaw_quaternion_wxyz,
)


def pose(x: float, y: float, z: float, yaw: float = 0.0) -> PoseSpec:
    return PoseSpec(
        position_xyz=(x, y, z),
        quaternion_wxyz=yaw_quaternion_wxyz(yaw),
    )


def static_box(
    name: str,
    half_size_xyz: tuple[float, float, float],
    pose: PoseSpec,
    color_rgba: tuple[float, float, float, float],
) -> StaticBoxSpec:
    return StaticBoxSpec(
        name=name,
        half_size_xyz=half_size_xyz,
        pose=pose,
        color_rgba=color_rgba,
    )


def textured_plane(
    name: str,
    texture_path: str | Path,
    half_size_xy: tuple[float, float],
    pose: PoseSpec,
    texture_repeat_uv: tuple[float, float],
    visual_pose: PoseSpec | None = None,
    collision_half_size_xyz: tuple[float, float, float] | None = None,
) -> TexturedPlaneSpec:
    return TexturedPlaneSpec(
        name=name,
        texture_path=texture_path,
        half_size_xy=half_size_xy,
        pose=pose,
        visual_pose=visual_pose or PoseSpec(position_xyz=(0.0, 0.0, 0.0)),
        texture_repeat_uv=texture_repeat_uv,
        collision_half_size_xyz=collision_half_size_xyz,
    )


def table(
    name: str,
    pose: PoseSpec,
    half_size_xy: tuple[float, float],
    color_rgba: tuple[float, float, float, float],
) -> StaticBoxSpec:
    table_center = pose.transform_point((0.0, 0.0, -0.025))
    return static_box(
        name=name,
        half_size_xyz=(half_size_xy[0], half_size_xy[1], 0.025),
        pose=PoseSpec(
            position_xyz=(
                float(table_center[0]),
                float(table_center[1]),
                float(table_center[2]),
            ),
            quaternion_wxyz=pose.quaternion_wxyz,
        ),
        color_rgba=color_rgba,
    )
