from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


POLICY_PROTOCOL_VERSION = "1.0"


@dataclass(frozen=True)
class ObservationSpec:
    """Observation modalities and camera roles consumed by a policy."""

    camera_roles: tuple[str, ...] = ()
    rgb: bool = False
    depth: bool = False
    state: bool = True
    tactile: bool = False
    capture_visual_every_step: bool = False

    def __post_init__(self) -> None:
        if (self.rgb or self.depth) and not self.camera_roles:
            raise ValueError("Visual observations require at least one camera role.")


@dataclass(frozen=True)
class ActionSpec:
    """Versioned semantic action contract advertised by an environment."""

    schema_version: int
    embodiment_id: str
    manipulators: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "ActionSpec":
        raw_spec = metadata.get("benchmark_action_spec")
        if not isinstance(raw_spec, Mapping):
            raise ValueError("Environment metadata has no benchmark_action_spec.")
        manipulators = raw_spec.get("manipulators")
        if not isinstance(manipulators, Mapping) or not manipulators:
            raise ValueError("benchmark_action_spec.manipulators must be non-empty.")
        return cls(
            schema_version=int(raw_spec.get("action_schema_version", 0)),
            embodiment_id=str(raw_spec.get("embodiment_id", "")),
            manipulators=manipulators,
        )


@dataclass(frozen=True)
class PolicyCapabilities:
    """Static runtime requirements declared by a policy implementation."""

    observation: ObservationSpec = field(default_factory=ObservationSpec)
    native_batching: bool = False
    action_chunk_size: int = 1

    def __post_init__(self) -> None:
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive.")


@dataclass(frozen=True)
class PolicyHandshake:
    """Serializable policy/environment compatibility handshake."""

    protocol_version: str
    embodiment_id: str
    control_frequency_hz: int
    action_schema_version: int
    camera_roles: tuple[str, ...]
    tactile_layout_ids: tuple[str, ...]

    @classmethod
    def from_environment_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> "PolicyHandshake":
        action_spec = ActionSpec.from_metadata(metadata)
        tactile_layouts = metadata.get("tactile_layouts", {})
        if not isinstance(tactile_layouts, Mapping):
            raise TypeError("tactile_layouts must be a mapping.")
        return cls(
            protocol_version=POLICY_PROTOCOL_VERSION,
            embodiment_id=action_spec.embodiment_id,
            control_frequency_hz=int(metadata.get("control_frequency_hz", 0)),
            action_schema_version=action_spec.schema_version,
            camera_roles=tuple(str(role) for role in metadata.get("camera_roles", ())),
            tactile_layout_ids=tuple(
                str(layout["layout_id"])
                for layout in tactile_layouts.values()
                if isinstance(layout, Mapping) and "layout_id" in layout
            ),
        )

