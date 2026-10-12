from __future__ import annotations

from enum import Enum


class SceneClass(str, Enum):
    """Top-level scene family used by both layouts and task taxonomy."""

    KITCHEN = "kitchen"
    STUDIO = "studio"


class LayoutId(str, Enum):
    """Canonical identifier for one evaluator-selectable scene layout."""

    KITCHEN_L1 = "kitchen_l1"
    KITCHEN_L2 = "kitchen_l2"
    KITCHEN_L3 = "kitchen_l3"
    STUDIO_L1 = "studio_l1"
    STUDIO_L2 = "studio_l2"
    STUDIO_L3 = "studio_l3"


_DEFAULT_LAYOUTS_BY_SCENE_CLASS: dict[SceneClass, tuple[LayoutId, ...]] = {
    SceneClass.KITCHEN: (
        LayoutId.KITCHEN_L1,
        LayoutId.KITCHEN_L2,
        LayoutId.KITCHEN_L3,
    ),
    SceneClass.STUDIO: (
        LayoutId.STUDIO_L1,
        LayoutId.STUDIO_L2,
        LayoutId.STUDIO_L3,
    ),
}


def default_layouts_for_scene_class(
    scene_class: SceneClass,
) -> tuple[LayoutId, ...]:
    """Return layout IDs without importing simulator-backed layout builders."""

    return _DEFAULT_LAYOUTS_BY_SCENE_CLASS[scene_class]


__all__ = ["LayoutId", "SceneClass", "default_layouts_for_scene_class"]
