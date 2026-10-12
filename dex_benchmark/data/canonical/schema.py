"""Canonical transition schema used by teleoperation data collection.

The schema deliberately contains only Python scalars, NumPy arrays and plain
mappings.  This makes a transition safe to hand to a background writer after
GPU-backed simulator observations have been copied to CPU-owned memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

import numpy as np


CANONICAL_SCHEMA_VERSION = 1


def to_cpu_copy(value: Any) -> Any:
    """Recursively copy arrays/tensors into independent CPU-owned values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "contiguous"):
            value = value.contiguous()
        if hasattr(value, "numpy"):
            return np.array(value.numpy(), copy=True)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True, order="C")
    if isinstance(value, Mapping):
        return {str(key): to_cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [to_cpu_copy(item) for item in value]
        return type(value)(converted)
    return value


def _finite_array(value: Any, *, field_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biufc":
        raise TypeError(f"{field_name} must be numeric, got dtype {array.dtype}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{field_name} contains NaN or infinite values.")
    return np.array(array, copy=True, order="C")


def validate_numeric(value: Any, *, field_name: str) -> None:
    """Validate a scalar/array or nested mapping of numeric action values."""

    if isinstance(value, Mapping):
        if not value:
            raise ValueError(f"{field_name} mapping must not be empty.")
        for key, item in value.items():
            validate_numeric(item, field_name=f"{field_name}.{key}")
        return
    _finite_array(value, field_name=field_name)


def _normalise_optional_time(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite when present.")
    return result


@dataclass(frozen=True)
class CanonicalObservation:
    """One observation snapshot in the canonical dataset.

    ``images`` and ``depth`` may be empty on control ticks where the configured
    image sampler is not due. They are indexed independently from the state
    stream by the observation ID and timestamp.
    """

    observation_id: int
    timestamp_s: float
    state: Mapping[str, np.ndarray]
    taxel: Mapping[str, np.ndarray]
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    depth: Mapping[str, np.ndarray] = field(default_factory=dict)
    image_timestamp_s: float | None = None

    def __post_init__(self) -> None:
        if self.observation_id < 0:
            raise ValueError("observation_id must be non-negative.")
        if not np.isfinite(float(self.timestamp_s)):
            raise ValueError("observation timestamp must be finite.")
        if set(self.images) != set(self.depth):
            missing_depth = sorted(set(self.images) - set(self.depth))
            missing_rgb = sorted(set(self.depth) - set(self.images))
            details = []
            if missing_depth:
                details.append("missing depth for " + ", ".join(missing_depth))
            if missing_rgb:
                details.append("missing RGB for " + ", ".join(missing_rgb))
            raise ValueError("Image/depth camera mismatch: " + "; ".join(details))

        for collection_name, collection in (
            ("state", self.state),
            ("taxel", self.taxel),
            ("images", self.images),
            ("depth", self.depth),
        ):
            for key, value in collection.items():
                if not str(key).strip() or "/" in str(key).strip("/"):
                    raise ValueError(
                        f"{collection_name} keys must be non-empty path components, "
                        f"got {key!r}."
                    )
                array = np.asarray(value)
                if array.dtype.kind not in "biufc":
                    raise TypeError(
                        f"{collection_name}.{key} must be numeric, got {array.dtype}."
                    )
                if collection_name not in ("images", "depth") and not np.isfinite(
                    array
                ).all():
                    raise ValueError(f"{collection_name}.{key} contains non-finite values.")

        if self.image_timestamp_s is not None and not np.isfinite(
            float(self.image_timestamp_s)
        ):
            raise ValueError("image_timestamp_s must be finite when present.")

    def cpu_copy(self) -> "CanonicalObservation":
        return CanonicalObservation(
            observation_id=int(self.observation_id),
            timestamp_s=float(self.timestamp_s),
            state={
                str(key): _finite_array(value, field_name=f"state.{key}")
                for key, value in self.state.items()
            },
            taxel={
                str(key): _finite_array(value, field_name=f"taxel.{key}")
                for key, value in self.taxel.items()
            },
            images={
                str(key): np.array(value, copy=True, order="C")
                for key, value in self.images.items()
            },
            depth={
                str(key): np.array(value, copy=True, order="C")
                for key, value in self.depth.items()
            },
            image_timestamp_s=(
                None
                if self.image_timestamp_s is None
                else float(self.image_timestamp_s)
            ),
        )

    @property
    def has_images(self) -> bool:
        return bool(self.images)


@dataclass(frozen=True)
class CanonicalAction:
    """Semantic and native action associated with one transition."""

    action_id: int
    timestamp_s: float
    semantic: Mapping[str, Mapping[str, np.ndarray]]
    native: Any
    action_schema_version: int = 1

    def __post_init__(self) -> None:
        if self.action_id < 0:
            raise ValueError("action_id must be non-negative.")
        if not np.isfinite(float(self.timestamp_s)):
            raise ValueError("action timestamp must be finite.")
        if self.action_schema_version <= 0:
            raise ValueError("action_schema_version must be positive.")
        validate_numeric(self.native, field_name="native action")
        if not self.semantic:
            raise ValueError("semantic action must contain at least one role.")
        for role, fields in self.semantic.items():
            if not fields:
                raise ValueError(f"semantic action role {role!r} is empty.")
            for field_name, value in fields.items():
                _finite_array(
                    value,
                    field_name=f"semantic.{role}.{field_name}",
                )

    def cpu_copy(self) -> "CanonicalAction":
        return CanonicalAction(
            action_id=int(self.action_id),
            timestamp_s=float(self.timestamp_s),
            semantic={
                str(role): {
                    str(name): _finite_array(
                        value,
                        field_name=f"semantic.{role}.{name}",
                    )
                    for name, value in fields.items()
                }
                for role, fields in self.semantic.items()
            },
            native=to_cpu_copy(self.native),
            action_schema_version=int(self.action_schema_version),
        )


@dataclass(frozen=True)
class QuestHandSnapshot:
    """Raw Quest and retargeting diagnostics for one hand."""

    joint_matrices: np.ndarray
    joint_pos_mano: np.ndarray | None = None
    ref_value: np.ndarray | None = None
    retarget_qpos: np.ndarray | None = None
    retarget_joint_names: tuple[str, ...] = ()
    gesture: Mapping[str, bool | float] | None = None

    def cpu_copy(self) -> "QuestHandSnapshot":
        return QuestHandSnapshot(
            joint_matrices=_finite_array(
                self.joint_matrices,
                field_name="quest.joint_matrices",
            ),
            joint_pos_mano=(
                None
                if self.joint_pos_mano is None
                else _finite_array(self.joint_pos_mano, field_name="quest.joint_pos_mano")
            ),
            ref_value=(
                None
                if self.ref_value is None
                else _finite_array(self.ref_value, field_name="quest.ref_value")
            ),
            retarget_qpos=(
                None
                if self.retarget_qpos is None
                else _finite_array(self.retarget_qpos, field_name="quest.retarget_qpos")
            ),
            retarget_joint_names=tuple(str(name) for name in self.retarget_joint_names),
            gesture=(None if self.gesture is None else dict(self.gesture)),
        )


@dataclass(frozen=True)
class QuestFrameSnapshot:
    """A deduplicated reference to the latest Quest input frame."""

    sequence: int
    source_timestamp_s: float | None
    received_timestamp_s: float
    left: QuestHandSnapshot | None = None
    right: QuestHandSnapshot | None = None

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("Quest sequence must be non-negative.")
        _normalise_optional_time(
            self.source_timestamp_s,
            field_name="quest.source_timestamp_s",
        )
        if not np.isfinite(float(self.received_timestamp_s)):
            raise ValueError("Quest received timestamp must be finite.")

    def cpu_copy(self) -> "QuestFrameSnapshot":
        return QuestFrameSnapshot(
            sequence=int(self.sequence),
            source_timestamp_s=_normalise_optional_time(
                self.source_timestamp_s,
                field_name="quest.source_timestamp_s",
            ),
            received_timestamp_s=float(self.received_timestamp_s),
            left=None if self.left is None else self.left.cpu_copy(),
            right=None if self.right is None else self.right.cpu_copy(),
        )


@dataclass(frozen=True)
class CanonicalTransition:
    """One complete ``obs_t -> action_t -> obs_t+1`` record."""

    episode_id: str
    transition_id: int
    observation_t: CanonicalObservation
    action: CanonicalAction
    observation_tp1: CanonicalObservation
    held_action: bool
    reward: Any
    terminated: Any
    truncated: Any
    info: Mapping[str, Any]
    quest_frame: QuestFrameSnapshot | None
    timestamps: Mapping[str, float | None]

    def __post_init__(self) -> None:
        if not str(self.episode_id).strip():
            raise ValueError("episode_id must be non-empty.")
        if self.transition_id < 0:
            raise ValueError("transition_id must be non-negative.")
        if self.observation_t.observation_id == self.observation_tp1.observation_id:
            raise ValueError("obs_t and obs_t+1 must have distinct observation IDs.")
        for name, value in self.timestamps.items():
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"timestamp {name!r} must be finite when present.")

    def cpu_copy(self) -> "CanonicalTransition":
        return CanonicalTransition(
            episode_id=str(self.episode_id),
            transition_id=int(self.transition_id),
            observation_t=self.observation_t.cpu_copy(),
            action=self.action.cpu_copy(),
            observation_tp1=self.observation_tp1.cpu_copy(),
            held_action=bool(self.held_action),
            reward=to_cpu_copy(self.reward),
            terminated=to_cpu_copy(self.terminated),
            truncated=to_cpu_copy(self.truncated),
            info=to_cpu_copy(dict(self.info)),
            quest_frame=(
                None if self.quest_frame is None else self.quest_frame.cpu_copy()
            ),
            timestamps={
                str(name): (None if value is None else float(value))
                for name, value in self.timestamps.items()
            },
        )


def jsonable(value: Any) -> Any:
    """Convert simulator values to strict JSON-compatible data."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return jsonable(to_cpu_copy(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def strict_json(value: Any) -> str:
    return json.dumps(jsonable(value), allow_nan=False, separators=(",", ":"))
