"""Human-viewer-only mouse alignment preview."""

from typing import Any

from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin


class MouseAlignmentPreview(Plugin):
    def __init__(self, camera: Any) -> None:
        self.camera = camera
        self.picture = R.UIPicture().Size(360, 270)
        self.panel = (
            R.UIWindow()
            .Label("M07 Mouse Alignment")
            .Pos(10, 450)
            .Size(380, 310)
            .append(self.picture)
        )

    def get_ui_windows(self) -> list[Any]:
        self.camera.take_picture()
        self.picture.Picture(self.camera._internal_renderer, "Color")
        return [self.panel]
