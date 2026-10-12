from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from dex_benchmark.policy_api.model_input import ModelInput
from dex_benchmark.policy_api.session import PolicyClient, SemanticAction
from kinedex_policy.adapter import KineDexAdapter
from kinedex_policy.client import KineDexWebsocketClient
from kinedex_policy.config import KineDexPolicyConfig
from kinedex_policy.force_control import FingerForceController
from kinedex_policy.tactile import expected_tactile_layout


class KineDexPolicySession:
    """Batched KineDex history, chunk execution, and force-control state."""

    def __init__(
        self, config: KineDexPolicyConfig, *, client: PolicyClient | None = None
    ) -> None:
        self.config = config
        self.adapter = KineDexAdapter(config)
        self.client = client or KineDexWebsocketClient(config.host, config.port)
        self._environment_metadata: Mapping[str, Any] | None = None
        self._controller: FingerForceController | None = None
        self.reset()

    def reset(self) -> None:
        self._histories: list[deque[dict[str, np.ndarray]]] = []
        self._tactile: list[np.ndarray | None] = []
        self._motions: list[deque[np.ndarray]] = []
        self._forces: list[deque[np.ndarray]] = []
        self._image_cache: list[dict[str, np.ndarray]] = []
        self._tick = 0

    def bind_environment(self, metadata: Mapping[str, Any]) -> None:
        expected = {
            "manipulator_roles": self.config.roles,
            "control_frequency_hz": self.config.control_hz,
        }
        for key, value in expected.items():
            actual = metadata.get(key)
            if key.endswith("_roles"):
                actual = tuple(actual or ())
            if actual != value:
                raise ValueError(
                    f"KineDex environment {key}={actual!r}, expected {value!r}."
                )
        available_cameras = tuple(metadata.get("camera_roles") or ())
        missing_cameras = tuple(
            role for role in self.config.camera_roles if role not in available_cameras
        )
        if missing_cameras:
            raise ValueError(
                "KineDex environment is missing required camera roles "
                f"{missing_cameras!r}; available={available_cameras!r}."
            )
        if self.config.tactile_mode == "full":
            layouts = metadata.get("tactile_layouts")
            if not isinstance(layouts, Mapping):
                raise ValueError("KineDex requires tactile_layouts metadata.")
            for role in self.config.roles:
                if layouts.get(role) != expected_tactile_layout():
                    raise ValueError(f"KineDex tactile layout mismatch for {role!r}.")
        self._environment_metadata = metadata

    def _ensure_batch(self, size: int) -> None:
        if not self._histories:
            self._histories = [
                deque(maxlen=self.config.observation_horizon) for _ in range(size)
            ]
            self._tactile = [None for _ in range(size)]
            self._motions = [deque() for _ in range(size)]
            self._forces = (
                [deque() for _ in range(size)]
                if self.config.tactile_mode == "full"
                else []
            )
            self._image_cache = [{} for _ in range(size)]
        elif len(self._histories) != size:
            raise ValueError("KineDex batch size changed; call reset().")

    def _visual_observation_required(self) -> bool:
        if not self._image_cache or not all(self._image_cache):
            return True
        remaining = tuple(len(buffer) for buffer in self._motions)
        if len(set(remaining)) > 1:
            raise RuntimeError("KineDex action buffers became misaligned.")
        steps_to_request = remaining[0] if remaining else 0
        request_tick = self._tick + steps_to_request
        first_history_tick = max(0, request_tick - self.config.observation_horizon + 1)
        interval = self.config.image_update_interval
        required_image_ticks = {
            tick - tick % interval
            for tick in range(first_history_tick, request_tick + 1)
        }
        return self._tick in required_image_ticks

    def _append(self, inputs: Sequence[ModelInput]) -> None:
        refresh_images = self._visual_observation_required()
        for index, (history, model_input) in enumerate(
            zip(self._histories, inputs, strict=True)
        ):
            observation, tactile = self.adapter.lowdim_observation(model_input)
            image_keys = tuple(f"{role}_rgb" for role in self.config.camera_roles)
            if refresh_images:
                images = self.adapter.images(model_input)
                self._image_cache[index] = {
                    key: images[key].copy() for key in image_keys
                }
            for key in image_keys:
                observation[key] = self._image_cache[index][key].copy()
            if not history:
                for _ in range(self.config.observation_horizon - 1):
                    history.append(
                        {key: value.copy() for key, value in observation.items()}
                    )
            history.append(observation)
            self._tactile[index] = tactile
        self._tick += 1

    def _stack_observations(self) -> dict[str, np.ndarray]:
        keys = tuple(self._histories[0][0])
        return {
            key: np.stack(
                [
                    np.stack([frame[key] for frame in history])
                    for history in self._histories
                ]
            )
            for key in keys
        }

    def _validate_checkpoint(self, metadata: Mapping[str, Any]) -> None:
        checkpoint_tactile_mode = str(metadata.get("tactile_mode", "full"))
        expected = {
            "policy_type": "kinedex",
            "embodiment": self.config.embodiment,
            "tactile_mode": self.config.tactile_mode,
            "roles": list(self.config.roles),
            "camera_roles": list(self.config.camera_roles),
            "fps": self.config.checkpoint_fps,
            "image_hz": self.config.checkpoint_image_hz,
            "prediction_horizon": self.config.prediction_horizon,
            "observation_horizon": self.config.observation_horizon,
            "available_action_steps": self.config.available_action_steps,
            "state_dim": self.config.state_dim,
            "motion_action_dim": self.config.motion_action_dim,
            "force_dim": self.config.force_dim,
            "model_action_dim": self.config.model_action_dim,
        }
        actual_metadata = dict(metadata)
        actual_metadata["tactile_mode"] = checkpoint_tactile_mode
        if self.config.tactile_mode == "full":
            expected["tactile_layout"] = expected_tactile_layout()
        mismatches = {
            key: (actual_metadata.get(key), value)
            for key, value in expected.items()
            if actual_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"KineDex checkpoint metadata mismatch: {mismatches}")

    def _make_controller(
        self, checkpoint_metadata: Mapping[str, Any]
    ) -> FingerForceController:
        if self._environment_metadata is None:
            raise RuntimeError("KineDex session must bind environment metadata first.")
        state_specs = self._environment_metadata.get("joint_state_specs")
        if not isinstance(state_specs, Mapping):
            raise ValueError("Environment is missing joint_state_specs.")
        names: dict[str, tuple[str, ...]] = {}
        lower: dict[str, np.ndarray] = {}
        upper: dict[str, np.ndarray] = {}
        for role in self.config.roles:
            spec = state_specs.get(role)
            if not isinstance(spec, Mapping):
                raise ValueError(f"Environment is missing joint spec for {role}.")
            names[role] = tuple(str(item) for item in spec["joint_names"])[7:]
            lower[role] = np.asarray(spec["lower"], dtype=np.float32)[7:]
            upper[role] = np.asarray(spec["upper"], dtype=np.float32)[7:]
        return FingerForceController(
            roles=self.config.roles,
            joint_names=names,
            joint_lower=lower,
            joint_upper=upper,
            force_lower=np.asarray(
                checkpoint_metadata["force_lower"], dtype=np.float32
            ),
            force_upper=np.asarray(
                checkpoint_metadata["force_upper"], dtype=np.float32
            ),
            gain_rad_per_n=self.config.force_gain_rad_per_n,
            max_residual_rad=self.config.max_force_residual_rad,
        )

    def _request(self) -> None:
        if any(self._motions) or (
            self.config.tactile_mode == "full" and any(self._forces)
        ):
            raise RuntimeError("KineDex buffers must refill synchronously.")
        observations = self._stack_observations()
        response = self.client.infer(
            {
                "policy_type": "kinedex",
                "embodiment": self.config.embodiment,
                "tactile_mode": self.config.tactile_mode,
                "observations": observations,
            }
        )
        metadata = response.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("KineDex response is missing metadata.")
        self._validate_checkpoint(metadata)
        if self.config.tactile_mode == "full" and self._controller is None:
            self._controller = self._make_controller(metadata)
        motions = np.asarray(response.get("motions"), dtype=np.float32)
        expected_motion = (
            len(self._histories),
            self.config.available_action_steps,
            self.config.motion_action_dim,
        )
        if motions.shape != expected_motion or not np.isfinite(motions).all():
            raise ValueError(f"Invalid KineDex motion chunk {motions.shape}.")
        if self.config.tactile_mode == "full":
            forces = np.asarray(response.get("force_targets"), dtype=np.float32)
            expected_force = (
                len(self._histories),
                self.config.available_action_steps,
                self.config.force_dim,
            )
            if forces.shape != expected_force or not np.isfinite(forces).all():
                raise ValueError(f"Invalid KineDex force chunk {forces.shape}.")
            for motion_buffer, force_buffer, motion, force in zip(
                self._motions, self._forces, motions, forces, strict=True
            ):
                motion_buffer.extend(motion[: self.config.execute_steps])
                force_buffer.extend(force[: self.config.execute_steps])
        else:
            if "force_targets" in response:
                raise ValueError("NoTouch KineDex returned unexpected force targets.")
            for motion_buffer, motion in zip(self._motions, motions, strict=True):
                motion_buffer.extend(motion[: self.config.execute_steps])

    def act_batch(self, model_inputs: Sequence[ModelInput]) -> Sequence[SemanticAction]:
        inputs = tuple(model_inputs)
        if not inputs:
            raise ValueError("KineDex act_batch requires at least one input.")
        self._ensure_batch(len(inputs))
        self._append(inputs)
        if not any(self._motions):
            self._request()
        if not all(self._motions) or (
            self.config.tactile_mode == "full" and not all(self._forces)
        ):
            raise RuntimeError("KineDex action buffers became misaligned.")
        results = []
        if self.config.tactile_mode == "full":
            assert self._controller is not None
            for motion_buffer, force_buffer, tactile in zip(
                self._motions, self._forces, self._tactile, strict=True
            ):
                if tactile is None:
                    raise RuntimeError("Full KineDex is missing tactile feedback.")
                adjusted = self._controller.apply(
                    motion_buffer.popleft(), force_buffer.popleft(), tactile
                )
                results.append(self.adapter.semantic_action(adjusted))
        else:
            for motion_buffer in self._motions:
                results.append(self.adapter.semantic_action(motion_buffer.popleft()))
        return tuple(results)

    def act(self, model_input: ModelInput) -> SemanticAction:
        return self.act_batch((model_input,))[0]

    def capture_visual_observation_after_action(self) -> bool:
        """Capture cadence-aligned frames needed by the next 2-frame request."""

        return self._visual_observation_required()

    def capture_lowdim_observation_after_action(self) -> bool:
        """Motion history and tactile force feedback update every control step."""

        return True

    def requires_control_period_tactile(self) -> bool:
        """KineDex consumes the control-period tactile observation."""

        return self.config.tactile_mode == "full"

    def close(self) -> None:
        self.client.close()
