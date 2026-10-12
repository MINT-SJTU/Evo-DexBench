"""Non-blocking terminal controls for interactive Quest teleoperation."""

from __future__ import annotations

import os
import select
import sys
import termios
import threading
import tty
from collections.abc import Callable
from typing import Any, TextIO


class TerminalQuitMonitor:
    """Monitor non-blocking single-character operator commands.

    The original class only handled ``q``.  The callbacks remain optional so
    existing single-episode callers keep the same API while the formal
    collector can use reset/proceed commands on the same terminal thread.
    """

    def __init__(
        self,
        on_quit: Callable[[], None],
        *,
        key: str = "q",
        on_reset: Callable[[], None] | None = None,
        on_proceed: Callable[[], None] | None = None,
        reset_key: str = "r",
        proceed_key: str = "e",
        stream: TextIO | None = None,
    ) -> None:
        for name, value in (
            ("quit", key),
            ("reset", reset_key),
            ("proceed", proceed_key),
        ):
            if len(value) != 1:
                raise ValueError(f"Terminal {name} key must be one character.")
        keys = {key.lower(), reset_key.lower(), proceed_key.lower()}
        if len(keys) != 3:
            raise ValueError("Terminal quit, reset, and proceed keys must differ.")
        self.on_quit = on_quit
        self.key = key.lower().encode("utf-8")
        self.reset_key = reset_key.lower().encode("utf-8")
        self.proceed_key = proceed_key.lower().encode("utf-8")
        self.on_reset = on_reset
        self.on_proceed = on_proceed
        self.stream = sys.stdin if stream is None else stream
        self._fd: int | None = None
        self._terminal_attributes: list[Any] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Start monitoring; return False when stdin is not an interactive TTY."""

        if self._thread is not None:
            return True
        self._stop.clear()
        try:
            if not self.stream.isatty():
                return False
            file_descriptor = self.stream.fileno()
            terminal_attributes = termios.tcgetattr(file_descriptor)
            tty.setcbreak(file_descriptor)
        except (AttributeError, OSError, termios.error):
            return False
        self._fd = file_descriptor
        self._terminal_attributes = terminal_attributes
        self._thread = threading.Thread(
            target=self._run,
            name="quest3-terminal-quit",
            daemon=True,
        )
        self._thread.start()
        return True

    def close(self) -> None:
        """Stop monitoring and restore the terminal mode."""

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self._fd is not None and self._terminal_attributes is not None:
            try:
                termios.tcsetattr(
                    self._fd,
                    termios.TCSADRAIN,
                    self._terminal_attributes,
                )
            except (OSError, termios.error):
                pass
        self._fd = None
        self._terminal_attributes = None

    def _run(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], 0.1)
                if not readable:
                    continue
                value = os.read(self._fd, 1).lower()
            except OSError:
                return
            if value == self.key:
                self.on_quit()
                return
            if value == self.reset_key and self.on_reset is not None:
                self.on_reset()
                continue
            if value == self.proceed_key and self.on_proceed is not None:
                self.on_proceed()
