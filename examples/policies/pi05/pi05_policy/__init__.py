"""Simulator-side π0.5 client for an official OpenPI policy server."""

from __future__ import annotations

from typing import Any

from pi05_policy.config import Pi05PolicyConfig
from pi05_policy.session import Pi05PolicySession


def create_pi05_policy(**kwargs: Any) -> Pi05PolicySession:
    """Create a π0.5 session from ``--policy-kwargs``."""

    return Pi05PolicySession(Pi05PolicyConfig(**kwargs))


__all__ = ["Pi05PolicyConfig", "Pi05PolicySession", "create_pi05_policy"]
