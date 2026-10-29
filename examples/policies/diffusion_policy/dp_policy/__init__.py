from __future__ import annotations

from typing import Any

from dp_policy.config import DPPolicyConfig
from dp_policy.session import DPPolicySession


def _create(variant: str, **kwargs: Any) -> DPPolicySession:
    return DPPolicySession(DPPolicyConfig(variant=variant, **kwargs))


def create_dp_c_policy(**kwargs: Any) -> DPPolicySession:
    return _create("dp_c", **kwargs)


def create_dp_t_policy(**kwargs: Any) -> DPPolicySession:
    return _create("dp_t", **kwargs)


__all__ = ["create_dp_c_policy", "create_dp_t_policy", "DPPolicyConfig"]
