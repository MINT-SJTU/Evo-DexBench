from __future__ import annotations

from dex_benchmark.sim.layout_types import (
    LayoutId,
    SceneClass,
    default_layouts_for_scene_class,
)
from dex_benchmark.sim.scenes.layouts.kitchen import KITCHEN_LAYOUTS
from dex_benchmark.sim.scenes.layouts.spec import LayoutSpec
from dex_benchmark.sim.scenes.layouts.studio import STUDIO_LAYOUTS


class LayoutRegistry:
    _registry: dict[LayoutId, LayoutSpec] = {
        **KITCHEN_LAYOUTS,
        **STUDIO_LAYOUTS,
    }

    @classmethod
    def get(cls, layout_id: LayoutId) -> LayoutSpec:
        try:
            return cls._registry[layout_id]
        except KeyError as exc:
            known_layouts = ", ".join(layout.value for layout in cls.list_layout_ids())
            raise ValueError(
                f"Unknown layout_id {layout_id!r}. Known layouts: {known_layouts}."
            ) from exc

    @classmethod
    def list_layout_ids(cls) -> list[LayoutId]:
        return list(cls._registry)

    @classmethod
    def list_layouts(cls, scene_class: SceneClass | None = None) -> list[LayoutSpec]:
        layouts = list(cls._registry.values())
        if scene_class is None:
            return layouts
        return [layout for layout in layouts if layout.scene_class == scene_class]

    @classmethod
    def default_layouts_for_scene_class(cls, scene_class: SceneClass) -> list[LayoutId]:
        layout_ids = default_layouts_for_scene_class(scene_class)
        missing_layouts = [
            layout_id for layout_id in layout_ids if layout_id not in cls._registry
        ]
        if missing_layouts:
            missing_values = ", ".join(layout_id.value for layout_id in missing_layouts)
            raise RuntimeError(
                f"Default layouts are not registered: {missing_values}."
            )
        return list(layout_ids)
