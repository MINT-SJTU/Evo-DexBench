from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dex_benchmark.sim.scenes.layouts.spec import LayoutLightingSpec


def load_layout_lighting(
    scene: Any,
    lighting_spec: LayoutLightingSpec,
    *,
    force_point_light_shadows: bool = False,
) -> None:
    """Instantiate one layout's declared lighting in a ManiSkill scene."""

    scene.set_ambient_light(list(lighting_spec.ambient_rgb))
    for point_light in lighting_spec.point_lights:
        scene.add_point_light(
            position=list(point_light.position_xyz),
            color=list(point_light.color_rgb),
            shadow=point_light.shadow or force_point_light_shadows,
            shadow_near=point_light.shadow_near,
            shadow_far=point_light.shadow_far,
            shadow_map_size=point_light.shadow_map_size,
        )
    for spot_light in lighting_spec.spot_lights:
        scene.add_spot_light(
            position=list(spot_light.position_xyz),
            direction=list(spot_light.direction_xyz),
            inner_fov=spot_light.inner_fov_rad,
            outer_fov=spot_light.outer_fov_rad,
            color=list(spot_light.color_rgb),
            shadow=spot_light.shadow,
            shadow_near=spot_light.shadow_near,
            shadow_far=spot_light.shadow_far,
            shadow_map_size=spot_light.shadow_map_size,
        )
    for directional_light in lighting_spec.directional_lights:
        scene.add_directional_light(
            direction=list(directional_light.direction_xyz),
            color=list(directional_light.color_rgb),
            position=list(directional_light.position_xyz),
            shadow=directional_light.shadow,
            shadow_scale=directional_light.shadow_scale,
            shadow_near=directional_light.shadow_near,
            shadow_far=directional_light.shadow_far,
            shadow_map_size=directional_light.shadow_map_size,
        )


def load_layout_lighting_batch(
    scene: Any,
    lighting_specs: Sequence[LayoutLightingSpec],
) -> None:
    """Load one lighting spec per ManiSkill sub-scene."""

    if not lighting_specs:
        raise ValueError("lighting_specs must contain at least one spec.")
    sub_scenes = tuple(getattr(scene, "sub_scenes", ()))
    if len(sub_scenes) != len(lighting_specs):
        raise ValueError(
            "Per-environment lighting count must match the ManiSkill sub-scenes; "
            f"got {len(lighting_specs)} specs and {len(sub_scenes)} sub-scenes."
        )
    if len(lighting_specs) > 1 and bool(
        getattr(scene, "parallel_in_single_scene", False)
    ):
        raise ValueError(
            "Per-environment lighting is not supported when all environments "
            "render in one shared scene."
        )

    for env_index, (sub_scene, lighting_spec) in enumerate(
        zip(sub_scenes, lighting_specs, strict=True)
    ):
        render_system = getattr(sub_scene, "render_system", None)
        if render_system is None:
            raise AttributeError(
                f"Sub-scene {env_index} does not expose a render_system for "
                "per-environment ambient lighting."
            )
        render_system.ambient_light = list(lighting_spec.ambient_rgb)
        scene_indices = [env_index]
        for point_light in lighting_spec.point_lights:
            scene.add_point_light(
                position=list(point_light.position_xyz),
                color=list(point_light.color_rgb),
                shadow=point_light.shadow,
                shadow_near=point_light.shadow_near,
                shadow_far=point_light.shadow_far,
                shadow_map_size=point_light.shadow_map_size,
                scene_idxs=scene_indices,
            )
        for spot_light in lighting_spec.spot_lights:
            scene.add_spot_light(
                position=list(spot_light.position_xyz),
                direction=list(spot_light.direction_xyz),
                inner_fov=spot_light.inner_fov_rad,
                outer_fov=spot_light.outer_fov_rad,
                color=list(spot_light.color_rgb),
                shadow=spot_light.shadow,
                shadow_near=spot_light.shadow_near,
                shadow_far=spot_light.shadow_far,
                shadow_map_size=spot_light.shadow_map_size,
                scene_idxs=scene_indices,
            )
        for directional_light in lighting_spec.directional_lights:
            scene.add_directional_light(
                direction=list(directional_light.direction_xyz),
                color=list(directional_light.color_rgb),
                position=list(directional_light.position_xyz),
                shadow=directional_light.shadow,
                shadow_scale=directional_light.shadow_scale,
                shadow_near=directional_light.shadow_near,
                shadow_far=directional_light.shadow_far,
                shadow_map_size=directional_light.shadow_map_size,
                scene_idxs=scene_indices,
            )
