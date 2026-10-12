from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from dex_benchmark.policy_api.reference import create_zero_policy
from dex_benchmark.policy_api.session import PolicySession


PolicyFactory = Callable[..., PolicySession]


class PolicyRegistry:
    """Lazy registry for built-in and externally installed policy plugins."""

    _factories: dict[str, PolicyFactory] = {"zero": create_zero_policy}
    _plugins_loaded = False

    @classmethod
    def _load_plugins(cls) -> None:
        if cls._plugins_loaded:
            return
        cls._plugins_loaded = True
        for entry_point in entry_points(group="dex_benchmark.policies"):
            normalized_name = entry_point.name.strip().lower()
            if normalized_name in cls._factories:
                raise ValueError(
                    f"Policy plugin {normalized_name!r} conflicts with an existing policy."
                )
            cls._factories[normalized_name] = entry_point.load()

    @staticmethod
    def _load_factory_spec(spec: str) -> PolicyFactory:
        module_name, separator, attribute_name = spec.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError(
                "Policy factory must be a registered name or 'module:callable'."
            )
        factory = getattr(import_module(module_name), attribute_name)
        if not callable(factory):
            raise TypeError(f"Policy factory {spec!r} is not callable.")
        return factory

    @classmethod
    def register(cls, name: str, factory: PolicyFactory) -> None:
        cls._load_plugins()
        normalized_name = name.strip().lower()
        if not normalized_name:
            raise ValueError("Policy name must be non-empty.")
        if normalized_name in cls._factories:
            raise ValueError(f"Policy {normalized_name!r} is already registered.")
        cls._factories[normalized_name] = factory

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> PolicySession:
        cls._load_plugins()
        normalized_name = name.strip().lower()
        factory = cls._factories.get(normalized_name)
        if factory is None and ":" in name:
            factory = cls._load_factory_spec(name)
        if factory is None:
            available = ", ".join(cls.names()) or "none"
            raise ValueError(
                f"Unknown policy {name!r}; registered policies: {available}."
            )
        return factory(**kwargs)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        cls._load_plugins()
        return tuple(sorted(cls._factories))
