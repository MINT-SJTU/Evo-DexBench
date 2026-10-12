from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from dex_benchmark.sim.generalization.attributes import (
    AxisRandomizationSpec,
    GeneralizationAxis,
)


def _derived_seed(
    *,
    protocol_version: str,
    scope_name: str,
    layout_id: str,
    axis: GeneralizationAxis,
    seed: int,
    attempt: int,
    env_index: int,
    parameter_name: str,
) -> int:
    payload = "\x1f".join(
        (
            protocol_version,
            scope_name,
            layout_id,
            axis.value,
            str(seed),
            str(attempt),
            str(env_index),
            parameter_name,
        )
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % (2**32)


@dataclass
class GeneralizationContext:
    """Deterministic, per-environment samples for one selected axis.

    ManiSkill recommends independently seeded RNG streams for each parallel
    environment. This context follows that contract for build-time sampling, where
    ManiSkill's reset-owned ``_batched_episode_rng`` is not yet available.
    Reset-time task code may still use ``_batched_episode_rng`` for unrelated episode
    randomness; benchmark generalization values must come from this context.
    """

    protocol_version: str
    task_id: str
    layout_id: str
    axis: GeneralizationAxis
    seed: int
    attempt: int
    num_envs: int
    axis_spec: AxisRandomizationSpec
    sample_indices: tuple[int, ...] | None = None
    scope_name: str | None = None
    debug: bool = False
    _values: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _consumed: set[str] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        if self.attempt < 0:
            raise ValueError(f"attempt must be non-negative, got {self.attempt}.")
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")
        if self.sample_indices is None:
            sample_indices = tuple(range(self.num_envs))
        else:
            sample_indices = tuple(self.sample_indices)
            if len(sample_indices) != self.num_envs:
                raise ValueError(
                    "sample_indices must contain one index per environment; "
                    f"expected {self.num_envs}, got {len(sample_indices)}."
                )
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in sample_indices
            ):
                raise ValueError("Generalization sample indices must be non-negative integers.")
        ordered_parameters = sorted(
            self.axis_spec.parameters, key=lambda parameter: parameter.name
        )
        scope_name = self.task_id if self.scope_name is None else self.scope_name
        values = {
            parameter.name: np.empty((self.num_envs,), dtype=np.float64)
            for parameter in ordered_parameters
        }
        for env_index, sample_index in enumerate(sample_indices):
            for parameter in ordered_parameters:
                rng = np.random.RandomState(
                    _derived_seed(
                        protocol_version=self.protocol_version,
                        scope_name=scope_name,
                        layout_id=self.layout_id,
                        axis=self.axis,
                        seed=self.seed,
                        attempt=self.attempt,
                        env_index=sample_index,
                        parameter_name=parameter.name,
                    )
                )
                values[parameter.name][env_index] = rng.uniform(
                    parameter.low, parameter.high
                )
        self._values = values
        if self.debug:
            print(
                f"[generalization-debug] axis={self.axis.value} seed={self.seed} "
                f"attempt={self.attempt} values={self._values}"
            )

    @property
    def requires_rebuild(self) -> bool:
        return self.axis_spec.requires_rebuild

    def value(self, name: str) -> np.ndarray:
        try:
            value = self._values[name]
        except KeyError as exc:
            known_names = ", ".join(sorted(self._values)) or "<none>"
            raise KeyError(
                f"Unknown generalization parameter {name!r}. Known parameters: {known_names}."
            ) from exc
        self._consumed.add(name)
        return value.copy()

    def assert_all_consumed(self) -> None:
        unconsumed = sorted(set(self._values) - self._consumed)
        if unconsumed:
            raise RuntimeError(
                "Task did not consume sampled generalization parameters: "
                + ", ".join(unconsumed)
            )

    def sampled_values_for_debug(self) -> dict[str, np.ndarray]:
        if not self.debug:
            raise RuntimeError(
                "Sampled values are only exposed when debug_generalization=True."
            )
        return {name: value.copy() for name, value in self._values.items()}

    def sampled_values(self) -> dict[str, np.ndarray]:
        """Return sampled values for reproducible evaluation records."""

        return {name: value.copy() for name, value in self._values.items()}
