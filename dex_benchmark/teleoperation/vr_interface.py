from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


VRFrameT = TypeVar("VRFrameT")


class VRInterface(ABC, Generic[VRFrameT]):
    """Common latest-value interface for VR input backends."""

    @abstractmethod
    def poll(self) -> VRFrameT | None:
        """Return the latest input frame, or ``None`` before the first frame."""
