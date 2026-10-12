from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from dex_benchmark.utils.transform import TensorLike, to_numpy
from dex_benchmark.sim.tasks.runtime import ManipulatorRole


ControlKind = Literal[
    "EE_delta_pose",
    "EE_abs_pose",
    "joint_position",
    "joint_delta_position",
    "unknown",
]


@dataclass(frozen=True)
class ControllerComponentSpec:
    """Action metadata for one named controller component."""

    name: str
    action_slice: slice
    controller_type: str
    normalize_action: bool
    action_dim: int
    control_kind: ControlKind = "unknown"
    action_lower: np.ndarray | None = None
    action_upper: np.ndarray | None = None
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class EEControlSpec(ControllerComponentSpec):
    """Common metadata for EE pose controllers."""

    frame: str = ""
    pos_lower: np.ndarray | None = None
    pos_upper: np.ndarray | None = None
    rot_lower: np.ndarray | None = None
    rot_upper: np.ndarray | None = None
    ee_link: str | None = None
    root_link_name: str | None = None
    use_target: bool = False
    rotation_scale_mode: str = "maniskill_pdee_pose"


@dataclass(frozen=True)
class EEDeltaPoseControlSpec(EEControlSpec):
    """Metadata for local-frame EE delta-pose control."""

    control_kind: ControlKind = "EE_delta_pose"


@dataclass(frozen=True)
class EEAbsPoseControlSpec(EEControlSpec):
    """Metadata for root-frame absolute EE pose control."""

    control_kind: ControlKind = "EE_abs_pose"


@dataclass(frozen=True)
class JointPositionControlSpec(ControllerComponentSpec):
    """Metadata for joint position target control."""

    control_kind: ControlKind = "joint_position"


@dataclass(frozen=True)
class JointDeltaPositionControlSpec(ControllerComponentSpec):
    """Metadata for joint delta-position control."""

    control_kind: ControlKind = "joint_delta_position"


def _as_float_array_or_none(value: Any) -> np.ndarray | None:
    """Convert optional controller metadata to float arrays."""

    if value is None:
        return None
    return np.asarray(to_numpy(value), dtype=np.float32)


def three_vector(value: Any, *, default: float = 0.0) -> np.ndarray:
    """Convert scalar or vector bounds to a 3D float vector."""

    if value is None:
        return np.full(3, default, dtype=np.float32)
    array = np.asarray(to_numpy(value), dtype=np.float32)
    if array.ndim == 0:
        return np.full(3, float(array), dtype=np.float32)
    return np.broadcast_to(array, (3,)).astype(np.float32)


def _action_slice_from_metadata(component: dict[str, Any]) -> slice:
    """Parse ManiSkill flat action mapping metadata."""

    action_slice_value = component.get("action_slice")
    if action_slice_value is None:
        action_dim = int(component.get("action_dim") or 0)
        return slice(0, action_dim)
    if isinstance(action_slice_value, slice):
        return action_slice_value
    action_start, action_end = action_slice_value
    return slice(action_start, action_end)


def _joint_names_from_metadata(component: dict[str, Any]) -> tuple[str, ...]:
    """Parse controller joint names from metadata."""

    joint_names = component.get("joint_names") or ()
    return tuple(str(joint_name) for joint_name in joint_names)


def _common_spec_kwargs(component: dict[str, Any], component_name: str) -> dict[str, Any]:
    """Build fields shared by all controller component specs."""

    action_lower = _as_float_array_or_none(component.get("action_lower"))
    action_upper = _as_float_array_or_none(component.get("action_upper"))
    action_slice = _action_slice_from_metadata(component)
    action_dim = component.get("action_dim")
    if action_dim is None:
        action_dim = action_slice.stop - action_slice.start

    return {
        "name": component_name,
        "action_slice": action_slice,
        "controller_type": str(component.get("controller_type", "")),
        "normalize_action": bool(component.get("normalize_action", True)),
        "action_dim": int(action_dim),
        "action_lower": action_lower,
        "action_upper": action_upper,
        "joint_names": _joint_names_from_metadata(component),
    }


def _is_ee_controller(controller_type: str) -> bool:
    """Return whether a ManiSkill controller type controls an EE pose."""

    return controller_type in {
        "PDEEPoseController",
        "PostureBiasedPDEEPoseController",
    }


def _is_joint_controller(controller_type: str) -> bool:
    """Return whether a ManiSkill controller type controls joints."""

    return "Joint" in controller_type



def control_spec_from_controller(
    controller: Any,
    *,
    action_space: Any | None = None,
) -> dict[str, Any]:
    """Build serializable control metadata from a ManiSkill controller.

    This is used by custom multi-agent environments where the controller is a
    dictionary rather than a single combined controller.
    """

    if controller is None:
        return {"action_space": action_space, "components": {}}

    action_mapping = getattr(controller, "action_mapping", None) or {}
    component_controllers = getattr(controller, "controllers", None) or {}
    components: dict[str, dict[str, Any]] = {}
    for component_name, component_controller in component_controllers.items():
        config = getattr(component_controller, "config", None)
        original_space = getattr(component_controller, "_original_single_action_space", None)
        single_space = getattr(component_controller, "single_action_space", None)
        physical_space = original_space or single_space
        component_shape = getattr(single_space, "shape", ()) or ()
        component_spec: dict[str, Any] = {
            "action_slice": action_mapping.get(component_name),
            "action_dim": component_shape[0] if component_shape else None,
            "controller_type": type(component_controller).__name__,
            "action_lower": _as_float_array_or_none(getattr(physical_space, "low", None)),
            "action_upper": _as_float_array_or_none(getattr(physical_space, "high", None)),
        }
        if config is not None:
            component_spec.update(
                {
                    "frame": getattr(config, "frame", None),
                    "normalize_action": getattr(config, "normalize_action", None),
                    "use_delta": getattr(config, "use_delta", None),
                    "use_target": getattr(config, "use_target", None),
                    "joint_names": getattr(config, "joint_names", None),
                    "ee_link": getattr(config, "ee_link", None),
                    "root_link_name": getattr(config, "root_link_name", None),
                    "pos_lower": _as_float_array_or_none(getattr(config, "pos_lower", None)),
                    "pos_upper": _as_float_array_or_none(getattr(config, "pos_upper", None)),
                    "rot_lower": _as_float_array_or_none(getattr(config, "rot_lower", None)),
                    "rot_upper": _as_float_array_or_none(getattr(config, "rot_upper", None)),
                }
            )
        components[str(component_name)] = component_spec

    return {"action_space": action_space, "components": components}


def component_spec_from_control_spec(
    control_spec: dict[str, Any],
    component_name: str,
    *,
    role: ManipulatorRole | str = ManipulatorRole.RIGHT,
) -> ControllerComponentSpec:
    """Extract one typed controller component spec from DexBench metadata."""

    role_value = role.value if isinstance(role, ManipulatorRole) else str(role)
    manipulator_specs = control_spec.get("manipulators")
    if isinstance(manipulator_specs, dict):
        role_spec = manipulator_specs.get(role_value)
        if role_spec is None:
            raise RuntimeError(
                f"Control spec has no manipulator role {role_value!r}."
            )
        component = role_spec.get("components", {}).get(component_name)
    else:
        # Retain support for previously recorded flat control metadata.
        component = control_spec.get("components", {}).get(component_name)
    if component is None:
        raise RuntimeError(
            f"Control component {component_name!r} unavailable for "
            f"role {role_value!r}."
        )

    common_kwargs = _common_spec_kwargs(component, component_name)
    controller_type = common_kwargs["controller_type"]
    use_delta = bool(component.get("use_delta", False))

    if _is_ee_controller(controller_type):
        ee_kwargs = {
            **common_kwargs,
            "frame": str(component.get("frame", "")),
            "pos_lower": three_vector(component.get("pos_lower", -0.1)),
            "pos_upper": three_vector(component.get("pos_upper", 0.1)),
            "rot_lower": three_vector(component.get("rot_lower", -0.1)),
            "rot_upper": three_vector(component.get("rot_upper", 0.1)),
            "ee_link": component.get("ee_link"),
            "root_link_name": component.get("root_link_name"),
            "use_target": bool(component.get("use_target", False)),
        }
        if use_delta:
            return EEDeltaPoseControlSpec(**ee_kwargs)
        return EEAbsPoseControlSpec(**ee_kwargs)

    if _is_joint_controller(controller_type):
        if use_delta:
            return JointDeltaPositionControlSpec(**common_kwargs)
        return JointPositionControlSpec(**common_kwargs)

    return ControllerComponentSpec(**common_kwargs)


def component_spec_from_env(
    env: Any,
    component_name: str,
    *,
    role: ManipulatorRole | str = ManipulatorRole.RIGHT,
) -> ControllerComponentSpec:
    """Read one typed controller component spec from an environment."""

    get_control_spec = env.get_wrapper_attr("get_control_spec")
    return component_spec_from_control_spec(
        get_control_spec(),
        component_name,
        role=role,
    )


def ee_delta_pose_spec_from_env(
    env: Any,
    component_name: str,
    *,
    role: ManipulatorRole | str = ManipulatorRole.RIGHT,
) -> EEDeltaPoseControlSpec:
    """Read an EE delta-pose component spec from an environment."""

    component_spec = component_spec_from_env(env, component_name, role=role)
    if not isinstance(component_spec, EEDeltaPoseControlSpec):
        raise TypeError(
            f"Component '{component_name}' is {component_spec.control_kind}, "
            "expected EE_delta_pose."
        )
    return component_spec


def ee_abs_pose_spec_from_env(
    env: Any,
    component_name: str,
    *,
    role: ManipulatorRole | str = ManipulatorRole.RIGHT,
) -> EEAbsPoseControlSpec:
    """Read an EE absolute-pose component spec from an environment."""

    component_spec = component_spec_from_env(env, component_name, role=role)
    if not isinstance(component_spec, EEAbsPoseControlSpec):
        raise TypeError(
            f"Component '{component_name}' is {component_spec.control_kind}, "
            "expected EE_abs_pose."
        )
    return component_spec


def joint_position_spec_from_env(
    env: Any,
    component_name: str,
    *,
    role: ManipulatorRole | str = ManipulatorRole.RIGHT,
) -> JointPositionControlSpec:
    """Read a joint-position component spec from an environment."""

    component_spec = component_spec_from_env(env, component_name, role=role)
    if not isinstance(component_spec, JointPositionControlSpec):
        raise TypeError(
            f"Component '{component_name}' is {component_spec.control_kind}, "
            "expected joint_position."
        )
    return component_spec


def joint_delta_position_spec_from_env(
    env: Any,
    component_name: str,
) -> JointDeltaPositionControlSpec:
    """Read a joint delta-position component spec from an environment."""

    component_spec = component_spec_from_env(env, component_name)
    if not isinstance(component_spec, JointDeltaPositionControlSpec):
        raise TypeError(
            f"Component '{component_name}' is {component_spec.control_kind}, "
            "expected joint_delta_position."
        )
    return component_spec


# Backward-compatible aliases for older script imports.
ArmControlSpec = EEDeltaPoseControlSpec
ControlComponentSpec = ControllerComponentSpec
TensorAction = TensorLike


def inverse_scale_box_action(
    value: TensorLike,
    low: TensorLike,
    high: TensorLike,
) -> TensorLike:
    """Compatibility wrapper for the generic action inverse scaler."""

    from dex_benchmark.policy_api.action_encoding import inverse_scale_action

    return inverse_scale_action(value, low, high)


def encode_delta_pose_action(
    component_spec: EEDeltaPoseControlSpec,
    *,
    pos_delta_local_m: TensorLike,
    rot_delta_local_rad: TensorLike | None = None,
) -> TensorLike:
    """Compatibility wrapper for EE delta-pose action encoding."""

    from dex_benchmark.policy_api.action_encoding import encode_ee_delta_pose_action

    return encode_ee_delta_pose_action(
        component_spec,
        pos_delta_root_m=pos_delta_local_m,
        rot_delta_root_rad=rot_delta_local_rad,
    )
