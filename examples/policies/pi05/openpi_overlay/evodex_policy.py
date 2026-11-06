"""Evo-DexBench transforms for the pinned OpenPI checkout."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class EvoDexInputs(transforms.DataTransformFn):
    """Adapt the generic Evo-DexBench schema at the OpenPI boundary."""

    action_dim: int
    control_prompt: str
    embodiment: str = "single"
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )
    GENERIC_STATE_DIM_PER_ROLE: ClassVar[int] = 111
    EE_POSE_SLICE: ClassVar[slice] = slice(0, 7)
    HAND_QPOS_SLICE: ClassVar[slice] = slice(49, 73)

    def _policy_state(self, state: np.ndarray) -> np.ndarray:
        if self.embodiment not in ("single", "dual"):
            raise ValueError(f"Unsupported embodiment {self.embodiment!r}")
        role_count = 1 if self.embodiment == "single" else 2
        expected_model_dim = 32 * role_count
        if self.action_dim != expected_model_dim:
            raise ValueError(
                f"{self.embodiment} embodiment requires model dimension "
                f"{expected_model_dim}, got {self.action_dim}"
            )
        semantic_dim = 31 * role_count
        generic_dim = self.GENERIC_STATE_DIM_PER_ROLE * role_count
        if state.ndim != 1:
            raise ValueError(f"State must be rank 1, got {state.shape}")
        # Online requests are already padded by the benchmark adapter.
        if state.shape == (self.action_dim,):
            return state
        if state.shape != (generic_dim,):
            raise ValueError(
                f"Expected generic {self.embodiment} state [{generic_dim}] or "
                f"model-ready state [{self.action_dim}], got {state.shape}"
            )
        parts = []
        for role_index in range(role_count):
            offset = role_index * self.GENERIC_STATE_DIM_PER_ROLE
            ee_start = offset + self.EE_POSE_SLICE.start
            ee_stop = offset + self.EE_POSE_SLICE.stop
            hand_start = offset + self.HAND_QPOS_SLICE.start
            hand_stop = offset + self.HAND_QPOS_SLICE.stop
            parts.extend(
                (
                    state[ee_start:ee_stop],
                    state[hand_start:hand_stop],
                )
            )
        selected = np.concatenate(parts).astype(np.float32, copy=False)
        if selected.shape != (semantic_dim,):
            raise AssertionError(f"Unexpected selected state shape {selected.shape}")
        return selected

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        if set(images) != set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Expected cameras {self.EXPECTED_CAMERAS}, got {tuple(images)}"
            )
        state = self._policy_state(np.asarray(data["state"], dtype=np.float32))
        converted_images = {}
        for name in self.EXPECTED_CAMERAS:
            image = np.asarray(images[name])
            if image.ndim != 3:
                raise ValueError(f"Camera {name} must be rank 3, got {image.shape}")
            # LeRobot video frames are [C, H, W]; online frames are [H, W, C].
            if image.shape[0] == 3 and image.shape[-1] != 3:
                image = np.moveaxis(image, 0, -1)
            if image.shape[-1] != 3:
                raise ValueError(f"Camera {name} must be RGB, got {image.shape}")
            # LeRobot's PyAV decoder returns float video frames in [0, 1],
            # whereas online observations arrive as uint8. OpenPI expects the
            # latter at the policy boundary.
            if np.issubdtype(image.dtype, np.floating):
                if not np.all(np.isfinite(image)):
                    raise ValueError(f"Camera {name} contains non-finite values")
                image_min = float(image.min())
                image_max = float(image.max())
                if image_min < 0.0 or image_max > 1.0:
                    raise ValueError(
                        f"Floating-point camera {name} must be in [0, 1], "
                        f"got [{image_min}, {image_max}]"
                    )
                image = np.rint(image * 255.0).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError(
                    f"Camera {name} must be uint8 or floating point in [0, 1], "
                    f"got {image.dtype}"
                )
            converted_images[name] = image
        result = {
            "image": converted_images,
            "image_mask": {name: np.True_ for name in self.EXPECTED_CAMERAS},
            "state": state,
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            role_count = 1 if self.embodiment == "single" else 2
            expected_action_dim = 30 * role_count
            if actions.ndim != 2 or actions.shape[-1] != expected_action_dim:
                raise ValueError(
                    f"Expected generic action chunk [T, {expected_action_dim}], "
                    f"got {actions.shape}"
                )
            result["actions"] = actions
        if "prompt" in data:
            prompt = str(data["prompt"]).strip()
            result["prompt"] = (
                prompt
                if prompt.startswith("<control_mode>")
                else f"{self.control_prompt} {prompt}"
            )
        return result


@dataclasses.dataclass(frozen=True)
class EvoDexOutputs(transforms.DataTransformFn):
    """Return the complete padded action; the benchmark removes padding."""

    action_dim: int

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action chunk [T, {self.action_dim}], got {actions.shape}"
            )
        return {"actions": actions}
