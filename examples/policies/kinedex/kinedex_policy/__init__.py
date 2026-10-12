from __future__ import annotations

from typing import Any

from kinedex_policy.config import KineDexPolicyConfig
from kinedex_policy.session import KineDexPolicySession


def create_kinedex_policy(**kwargs: Any) -> KineDexPolicySession:
    return KineDexPolicySession(KineDexPolicyConfig(**kwargs))


__all__ = ["KineDexPolicyConfig", "create_kinedex_policy"]
