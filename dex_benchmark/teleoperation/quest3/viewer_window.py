"""Linux desktop geometry helpers for the SAPIEN Human Viewer."""

from __future__ import annotations

from collections.abc import Sequence
import shutil
import subprocess
from typing import Final, Mapping
import warnings


DEFAULT_VIEWER_WINDOW_SIZE: Final[tuple[int, int]] = (2304, 1296)
TASK_VIEWER_WINDOW_SIZE_DEFAULTS: Final[Mapping[str, tuple[int, int]]] = {
    "b02_v1": (3072, 1728),
    "b05_v1": (3072, 1728),
    "b06_v1": (3072, 1728),
    "b07_v1": (3072, 1728),
    "b08_v1": (3072, 1728),
    "b09_v1": (3072, 1728),
    "b10_v1": (3072, 1728),
    "b12_v1": (3072, 1728),
}


def task_viewer_window_size(task_id: str) -> tuple[int, int] | None:
    """Return a task-specific Human Viewer window size, when configured."""

    return TASK_VIEWER_WINDOW_SIZE_DEFAULTS.get(task_id)


def configure_human_viewer_window(
    size: tuple[int, int],
    *,
    position: tuple[int, int] | None = None,
    title_hint: str = "SAPIEN",
) -> str | None:
    """Resize a newly-created SAPIEN window using ``wmctrl``.

    ``size`` describes the desktop window geometry, not the canonical RGB-D
    sensor resolution.  A missing ``position`` leaves the window-manager
    selected position unchanged.
    """

    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError("Human viewer window size must be positive.")
    if position is None:
        x, y = -1, -1
    else:
        x, y = (int(value) for value in position)
        if x < 0 or y < 0:
            raise ValueError("Human viewer window position must be non-negative.")
    if not title_hint.strip():
        raise ValueError("Human viewer window title hint must not be empty.")

    wmctrl = shutil.which("wmctrl")
    if wmctrl is None:
        warnings.warn(
            "wmctrl is not installed; continuing with the window-manager selected "
            "Human Viewer geometry. Install it with 'sudo apt install wmctrl' to "
            "apply --viewer-window-size and --viewer-window-position.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    window_id = _find_window_id(wmctrl, title_hint)
    _run_wmctrl(wmctrl, ("-i", "-r", window_id, "-b", "remove,maximized_vert,maximized_horz"))
    geometry = f"0,{x},{y},{width},{height}"
    _run_wmctrl(wmctrl, ("-i", "-r", window_id, "-e", geometry))
    return window_id


def _find_window_id(wmctrl: str, title_hint: str) -> str:
    result = subprocess.run(
        (wmctrl, "-l"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"wmctrl -l failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    hint = title_hint.casefold()
    for line in result.stdout.splitlines():
        fields = line.split(None, 3)
        if len(fields) == 4 and hint in fields[3].casefold():
            return fields[0]
    raise RuntimeError(
        f"Could not find a Human Viewer window whose title contains {title_hint!r}. "
        "The SAPIEN window must be rendered before geometry can be applied."
    )


def _run_wmctrl(wmctrl: str, arguments: Sequence[str]) -> None:
    result = subprocess.run(
        (wmctrl, *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"wmctrl {' '.join(arguments)} failed with exit code "
            f"{result.returncode}: {result.stderr.strip()}"
        )
