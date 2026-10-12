from __future__ import annotations

from collections.abc import Mapping
import copy
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import (
    CameraRigRegistry,
    CameraRigSpec,
    CameraViewSpec,
    HumanViewerCameraPoseSpec,
    RenderCameraSpec,
)
from dex_benchmark.sim.envs.action import ManipulatorAction
from dex_benchmark.policy_api.action_encoding import (
    ActionBoundsPolicy,
    encode_ee_delta_pose_action,
    encode_joint_position_action,
)
from dex_benchmark.policy_api.control_spec import (
    ControllerComponentSpec,
    EEDeltaPoseControlSpec,
    JointPositionControlSpec,
    component_spec_from_control_spec,
)
from dex_benchmark.sim.envs.observation import (
    CameraObservation,
    ManipulatorObservation,
    Observation,
)
from dex_benchmark.sim.scenes.layouts import (
    InteractionSiteSpec,
    LayoutRegistry,
    LayoutSpec,
    PoseSpec,
    RectangularLightApertureSpec,
    pose_spec_to_sapien_pose,
)
from dex_benchmark.sim.rendering import (
    load_layout_lighting,
    load_layout_lighting_batch,
    rectangular_aperture_camera_shader_config,
    rectangular_aperture_viewer_shader_config,
)
from dex_benchmark.sim.envs.spaces import ControlMode
from dex_benchmark.sim.generalization import (
    GeneralizationAxis,
    GeneralizationContext,
    GeneralizationProfile,
    GeneralizationProtocol,
    InvalidGeneralizationSample,
    TaskGeneralizationSpec,
    global_axis_spec,
)
from dex_benchmark.sim.generalization.camera import camera_view_values
from dex_benchmark.sim.generalization.lighting import lighting_values
from dex_benchmark.sim.generalization.specs import (
    CAMERA_PARAMETER_NAMES,
    LIGHTING_PARAMETER_NAMES,
)
from dex_benchmark.sim.generalization.validators import as_validity_tensor
from dex_benchmark.sim.tasks.attributes import LayoutId, StagePersistence
from dex_benchmark.sim.tasks.registry import TaskSettings
from dex_benchmark.sim.tasks.runtime import (
    CameraRole,
    EmbodimentSpec,
    ManipulatorBinding,
    ManipulatorRole,
)
from dex_benchmark.utils.visualization import (
    HumanViewerPoseHotkey,
    HumanViewerRobotBaseControl,
)

try:
    from mani_skill.envs.sapien_env import BaseEnv as ManiSkillBaseEnv
    from mani_skill.sensors.camera import Camera, CameraConfig
    from mani_skill.utils import sapien_utils
    from mani_skill.utils.structs.types import SimConfig
except Exception:  # pragma: no cover - depends on optional simulator stack.
    ManiSkillBaseEnv = object
    Camera = None
    CameraConfig = None
    SimConfig = None
    sapien_utils = None

try:
    import torch
except Exception:  # pragma: no cover - depends on optional simulator stack.
    torch = None


def _to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return value


def _space_bounds(space: Any) -> tuple[Any, Any]:
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    return _to_numpy(low), _to_numpy(high)


SCENE_CAMERA_UID_BY_ROLE = {
    CameraRole.CONTEXT: "context_camera",
    CameraRole.INTERACTION: "interaction_camera",
}
WRIST_CAMERA_UID_BY_ROLE = {
    CameraRole.WRIST_LEFT: "wrist_left_camera",
    CameraRole.WRIST_RIGHT: "wrist_right_camera",
}
STATE_ONLY_OBS_MODES = frozenset({"state", "state_dict", "none"})
VISUAL_REQUIRED_SENSORS = frozenset(
    {
        "context_rgb",
        "context_depth",
        "interaction_rgb",
        "interaction_depth",
        "wrist_left_rgb",
        "wrist_left_depth",
        "wrist_right_rgb",
        "wrist_right_depth",
        "pointcloud",
    }
)


class DexVLABenchBaseEnv(ManiSkillBaseEnv):
    """Base ManiSkill environment with VLA-friendly benchmark hooks."""

    # The largest benchmark rooms span 6.4 m. ManiSkill's 5 m default makes
    # neighboring GPU sub-scenes overlap, so a robot near a room boundary can
    # collide with fixtures from another vectorized environment. This matches
    # ManiSkill's RoboCasa kitchen default and still covers the studio layouts.
    PARALLEL_ENV_SPACING_M = 8.0
    TASK_SETTINGS: TaskSettings | None = None
    INTERACTION_SITE_ID: str | None = None
    GENERALIZATION_SPEC = TaskGeneralizationSpec()
    # Override per task when the Human Viewer should use a camera independent
    # from the observation context camera. Coordinates are site-local.
    HUMAN_RENDER_CAMERA_SPEC: RenderCameraSpec | None = None
    # Override per task to select a specific sensor in the Human Viewer
    # ControlWindow camera list. When unset, the first available wrist camera
    # is selected (right before left).
    HUMAN_VIEWER_CAMERA_UID: str | None = None
    # Exact world-frame render-camera poses keyed by layout ID. GUI camera
    # selection remains independent and continues to prefer a wrist sensor.
    HUMAN_VIEWER_CAMERA_POSES: Mapping[str, HumanViewerCameraPoseSpec] = {}

    @property
    def _default_sim_config(self) -> "SimConfig":
        if SimConfig is None:
            raise ImportError(
                "ManiSkill3 is required to configure DexVLABench simulation."
            )
        return SimConfig(spacing=self.PARALLEL_ENV_SPACING_M)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if self.TASK_SETTINGS is None:
            raise ValueError("Subclass must define TASK_SETTINGS")
        if ManiSkillBaseEnv is object:
            raise ImportError(
                "ManiSkill3 is required to instantiate DexVLABench environments. "
                "Install with `pip install -e .[maniskill]`."
            )
        requested_layout_id = kwargs.pop("layout_id", None)
        requested_axis = kwargs.pop("generalization_axis", None)
        requested_profile = kwargs.pop("generalization_profile", None)
        requested_generalization_seed = kwargs.pop("generalization_seed", None)
        requested_generalization_attempt = kwargs.pop("generalization_attempt", 0)
        requested_generalization_sample_index = kwargs.pop(
            "generalization_sample_index", None
        )
        self._debug_generalization = bool(kwargs.pop("debug_generalization", False))
        # Viewer-only diagnostics can use state_dict observations and rely on
        # the human render camera, without forcing RGB-D benchmark sensors.
        # The normal benchmark path remains strict by default.
        self._skip_observation_validation = bool(
            kwargs.pop("skip_observation_validation", False)
        )
        self._quest_view_capture_enabled = bool(
            kwargs.pop("quest_view_capture_enabled", False)
        )
        self._quest_view_capture_every_n = int(
            kwargs.pop("quest_view_capture_every_n", 1)
        )
        if self._quest_view_capture_every_n < 1:
            raise ValueError("quest_view_capture_every_n must be at least 1.")
        self._quest_view_capture_counter = 0
        self._latest_quest_view_rgb: np.ndarray | None = None
        self._human_viewer_pose_hotkey = HumanViewerPoseHotkey(key_name="p")
        self._human_viewer_robot_base_control = HumanViewerRobotBaseControl()
        self._human_viewer_camera_initialized = False
        self._current_instruction: str | None = None
        # CPU tactile policies require impulses accumulated over every physics
        # substep. Vision-only policies may disable that model-observation work;
        # task contact queries remain instantaneous and continue to run on
        # every control step.
        self._tactile_control_accumulation_enabled = True
        # The public benchmark observation always retains its tactile field so
        # Gymnasium spaces stay stable.  Vision-only policy evaluation can
        # replace that field with zeros without executing the virtual-taxel
        # projection.  This switch never affects task contact queries.
        self._policy_tactile_observation_enabled = True
        self._selected_layout_id: LayoutId | None = None
        self._selected_layout_spec: LayoutSpec | None = None
        layout_options = (
            {"layout_id": requested_layout_id}
            if requested_layout_id is not None
            else None
        )
        self._select_episode_layout(seed=None, options=layout_options)
        if self.INTERACTION_SITE_ID is not None:
            self.require_current_interaction_site()
        self._configure_rectangular_aperture_shaders(kwargs)
        self._generalization_protocol = GeneralizationProtocol.load()
        self._generalization_profile = self._parse_generalization_profile(
            requested_profile
        )
        self._generalization_axis = self._parse_generalization_axis(requested_axis)
        if (
            self._generalization_profile is GeneralizationProfile.NONE
            and self._generalization_axis is not None
        ):
            raise ValueError(
                "generalization_profile='none' cannot be combined with "
                "generalization_axis."
            )
        if (
            self._generalization_profile is GeneralizationProfile.COLLECTION
            and self._generalization_axis
            in (
                GeneralizationAxis.CAMERA,
                GeneralizationAxis.LIGHTING,
            )
        ):
            raise ValueError(
                "The collection generalization profile only supports spatial "
                "and physical axes."
            )
        self._generalization_seed = self._validate_generalization_seed(
            requested_generalization_seed
        )
        self._generalization_attempt = self._validate_generalization_attempt(
            requested_generalization_attempt
        )
        self._generalization_sample_index = self._validate_generalization_sample_index(
            requested_generalization_sample_index
        )
        self._generalization_cache_names: set[str] = set()
        self._generalization_context = self._make_generalization_context(
            num_envs=int(kwargs.get("num_envs", 1))
        )
        if kwargs.get("obs_mode") is None:
            kwargs["obs_mode"] = self.TASK_SETTINGS.obs_mode
        if kwargs.get("reward_mode") is None:
            kwargs["reward_mode"] = "sparse"
        control_mode = kwargs.get("control_mode")
        if control_mode is None:
            control_mode = self.TASK_SETTINGS.control_mode
        if isinstance(control_mode, ControlMode):
            control_mode = control_mode.value
        kwargs["control_mode"] = control_mode
        expected_robot_uids = self.TASK_SETTINGS.embodiment_spec.robot_uids
        maniskill_robot_uids: str | tuple[str, ...] = (
            expected_robot_uids[0]
            if len(expected_robot_uids) == 1
            else expected_robot_uids
        )
        requested_robot_uids = kwargs.get("robot_uids")
        if requested_robot_uids is None:
            kwargs["robot_uids"] = maniskill_robot_uids
        else:
            normalized_robot_uids = (
                (requested_robot_uids,)
                if isinstance(requested_robot_uids, str)
                else tuple(requested_robot_uids)
            )
            if normalized_robot_uids != expected_robot_uids:
                raise ValueError(
                    f"Task {self.TASK_SETTINGS.task_id!r} has fixed embodiment "
                    f"{self.TASK_SETTINGS.embodiment_spec.embodiment_id!r} and "
                    f"requires robot_uids={expected_robot_uids!r}, got "
                    f"{normalized_robot_uids!r}."
                )
            kwargs["robot_uids"] = maniskill_robot_uids
        super().__init__(*args, **kwargs)
        self._achieved_stages = torch.zeros(
            (self.num_envs, len(self.TASK_SETTINGS.evaluation.stages)),
            dtype=torch.bool,
            device=self.device,
        )
        self._validate_required_camera_configs()

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._validate_reset_layout_request(options)
        self._quest_view_capture_counter = 0
        self._latest_quest_view_rgb = None
        reset_indices = (options or {}).get("env_idx", slice(None))
        # ManiSkill calls evaluate() from inside super().reset(). CPU contact
        # wrappers are valid only for the PhysX state that produced them, so they
        # must be discarded before episode initialization changes that state.
        self._invalidate_tactile_caches()
        if hasattr(self, "_task_success"):
            self._task_success[reset_indices] = False
        if hasattr(self, "_assembly_grasp_steps"):
            self._assembly_grasp_steps[reset_indices] = 0
        obs, info = super().reset(seed=seed, options=options)
        self._invalidate_tactile_caches()
        if self._generalization_context is not None:
            self._generalization_context.assert_all_consumed()
        # Samples are generated inside valid parameter ranges. This hook is a
        # final error check, not a rejection-sampling loop.
        self._validate_generalization_state()
        # ManiSkill calls ``self.reset()`` once from ``BaseEnv.__init__`` before
        # DexVLABench-owned episode trackers are constructed. Task scene/state is
        # already available at that point, but these trackers are initialized only
        # after the parent constructor returns.
        if hasattr(self, "_achieved_stages"):
            self._achieved_stages[reset_indices] = False
        self._current_instruction = self.sample_instruction()
        observation = self._build_observation(base_obs=obs)
        if not self._skip_observation_validation:
            self._validate_required_observation_sensors(observation)
        profile = getattr(
            self, "_generalization_profile", GeneralizationProfile.EVALUATION
        )
        protocol = getattr(self, "_generalization_protocol", None)
        reset_info = {
            "reconfigure": bool(info.get("reconfigure", False)),
            "task_id": self.TASK_SETTINGS.task_id,
            "layout_id": self.current_layout_id_value,
            "generalization_axis": (
                self._generalization_axis.value
                if self._generalization_axis is not None
                else None
            ),
            "generalization_seed": self._generalization_seed,
            "generalization_profile": (
                profile.value if self._generalization_axis is not None else None
            ),
            "generalization_attempt": getattr(self, "_generalization_attempt", 0),
            "generalization_sample_index": getattr(
                self, "_generalization_sample_index", None
            ),
            "generalization_protocol_version": getattr(protocol, "version", None),
        }
        reset_info.update(self.get_episode_metadata())
        return self._extract_observation(observation), reset_info

    def render_human(self) -> Any:
        """Render the human viewer and handle project-wide diagnostic keys."""

        viewer = super().render_human()
        if not self._human_viewer_camera_initialized:
            self._human_viewer_camera_initialized = self._select_human_viewer_camera(
                viewer
            )
        self._human_viewer_pose_hotkey.poll(viewer)
        self._human_viewer_robot_base_control.poll(
            viewer,
            self._human_viewer_robot_bases(),
        )
        return viewer

    def human_viewer_camera_uid(self) -> str | None:
        """Return the sensor UID selected by default in the Human Viewer."""

        configured_uid = self.HUMAN_VIEWER_CAMERA_UID
        if configured_uid is not None:
            if not configured_uid.strip():
                raise ValueError("HUMAN_VIEWER_CAMERA_UID must not be empty.")
            return configured_uid

        sensor_uids = set(getattr(self, "_sensors", {}) or {})
        for camera_uid in ("wrist_right_camera", "wrist_left_camera"):
            if camera_uid in sensor_uids:
                return camera_uid
        return None

    def _setup_viewer(self) -> None:
        """Initialize the SAPIEN viewer and apply its task-default camera mode."""

        self._human_viewer_camera_initialized = False
        super()._setup_viewer()
        viewer = getattr(self, "_viewer", None)
        if viewer is None:
            return
        self._human_viewer_camera_initialized = self._select_human_viewer_camera(viewer)

    def _select_human_viewer_camera(self, viewer: Any) -> bool:
        """Select a configured sensor, or retain the initialized free camera."""

        control_window = getattr(viewer, "control_window", None)
        if callable(control_window):
            control_window = control_window()
        if control_window is None:
            return False

        camera_uid = self.human_viewer_camera_uid()
        if camera_uid is None:
            return True
        camera_items = list(getattr(control_window, "camera_items", ()))
        try:
            camera_index = camera_items.index(camera_uid)
        except ValueError as exc:
            # SAPIEN prefixes scene cameras in the viewer list (for example,
            # ``scene-0_wrist_right_camera``), while sensor UIDs stay unprefixed.
            camera_suffix = f"_{camera_uid}"
            camera_index = next(
                (
                    index
                    for index, camera_item in enumerate(camera_items)
                    if isinstance(camera_item, str)
                    and camera_item.endswith(camera_suffix)
                ),
                None,
            )
            if camera_index is not None:
                control_window.camera_index = camera_index
                return True
            if self.HUMAN_VIEWER_CAMERA_UID is not None:
                raise RuntimeError(
                    f"Configured Human Viewer camera {camera_uid!r} is not "
                    f"available. Known cameras: {camera_items!r}."
                ) from exc
            return False
        control_window.camera_index = camera_index
        return True

    def _human_viewer_robot_bases(self) -> dict[str, Any]:
        """Return named robot articulations without reaching through a wrapper."""

        agents_by_name = getattr(self.agent, "agents_dict", None)
        if isinstance(agents_by_name, Mapping):
            return {
                str(agent_name): agent.robot
                for agent_name, agent in agents_by_name.items()
            }
        agent_name = str(getattr(self.agent, "uid", "robot"))
        return {agent_name: self.agent.robot}

    def capture_synced_human_viewer_rgb(self) -> np.ndarray:
        """Render RGB from the current freely moved human-viewer camera pose."""

        viewer = getattr(self, "_viewer", None)
        if viewer is None:
            raise RuntimeError(
                "The human viewer must be rendered before capturing its camera view."
            )
        window = getattr(viewer, "window", None)
        get_camera_pose = getattr(window, "get_camera_pose", None)
        if not callable(get_camera_pose):
            raise AttributeError(
                "The human viewer window does not expose get_camera_pose()."
            )
        human_render_cameras = getattr(self, "_human_render_cameras", {})
        render_sensor = human_render_cameras.get("render_camera")
        if render_sensor is None:
            known_cameras = ", ".join(sorted(human_render_cameras)) or "<none>"
            raise KeyError(
                "Quest 3 operator view requires a human render camera named "
                f"'render_camera'. Known cameras: {known_cameras}."
            )
        if getattr(render_sensor, "entity", None) is not None:
            raise RuntimeError(
                "The synchronized human render camera must not be mounted to an entity."
            )
        render_camera = getattr(render_sensor, "camera", None)
        set_local_pose = getattr(render_camera, "set_local_pose", None)
        if not callable(set_local_pose):
            raise AttributeError(
                "The human render camera does not expose set_local_pose()."
            )

        # Both poses are in the world frame because render_camera is unmounted.
        set_local_pose(get_camera_pose())
        viewer_config = getattr(self, "_viewer_camera_config", None)
        viewer_fov = getattr(viewer_config, "fov", None)
        set_fovy = getattr(render_camera, "set_fovy", None)
        if viewer_fov is not None and callable(set_fovy):
            fov_values = np.asarray(viewer_fov, dtype=np.float64).reshape(-1)
            if fov_values.size != 1 or not np.isfinite(fov_values[0]):
                raise ValueError(
                    f"Human viewer FOV must be one finite value, got {viewer_fov!r}."
                )
            set_fovy(float(fov_values[0]))

        return self._capture_render_camera_rgb(render_sensor)

    def capture_quest_view_rgb(self) -> np.ndarray:
        """Render the fixed Quest operator camera without opening a desktop viewer."""

        cached_rgb = getattr(self, "_latest_quest_view_rgb", None)
        if cached_rgb is not None:
            return np.asarray(cached_rgb).copy()

        human_render_cameras = getattr(self, "_human_render_cameras", {})
        render_sensor = human_render_cameras.get("render_camera")
        if render_sensor is None:
            known_cameras = ", ".join(sorted(human_render_cameras)) or "<none>"
            raise KeyError(
                "Quest 3 operator view requires a human render camera named "
                f"'render_camera'. Known cameras: {known_cameras}."
            )
        return self._capture_render_camera_rgb(
            render_sensor,
            update_scene_each_capture=False,
        )

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        """Capture benchmark sensors and the optional Quest camera in one pass."""

        for hidden_object in self._hidden_objects:
            hidden_object.hide_visual()
        quest_render_sensor = None
        if self._quest_view_capture_enabled:
            quest_render_sensor = getattr(self, "_human_render_cameras", {}).get(
                "render_camera"
            )
        capture_quest_view = (
            quest_render_sensor is not None
            and self._quest_view_capture_counter % self._quest_view_capture_every_n == 0
        )
        self._quest_view_capture_counter += 1
        self.scene.update_render(
            update_sensors=True,
            update_human_render_cameras=capture_quest_view,
        )
        self.capture_sensor_data()
        if capture_quest_view:
            assert quest_render_sensor is not None
            quest_render_sensor.capture()

        sensor_obs: dict[str, Any] = {}
        for name, sensor in self.scene.sensors.items():
            if Camera is None or not isinstance(sensor, Camera):
                continue
            if self.obs_mode in ["state", "state_dict"]:
                sensor_obs[name] = sensor.get_obs(
                    position=False,
                    segmentation=False,
                    apply_texture_transforms=apply_texture_transforms,
                )
            else:
                sensor_obs[name] = sensor.get_obs(
                    rgb=self.obs_mode_struct.visual.rgb,
                    depth=self.obs_mode_struct.visual.depth,
                    position=self.obs_mode_struct.visual.position,
                    segmentation=self.obs_mode_struct.visual.segmentation,
                    normal=self.obs_mode_struct.visual.normal,
                    albedo=self.obs_mode_struct.visual.albedo,
                    apply_texture_transforms=apply_texture_transforms,
                )

        quest_rgb = None
        if capture_quest_view:
            assert quest_render_sensor is not None
            quest_observation = quest_render_sensor.get_obs(
                rgb=True,
                depth=False,
                segmentation=False,
                position=False,
            )
            if isinstance(quest_observation, Mapping):
                quest_rgb = quest_observation.get("rgb")

        render_device = getattr(getattr(self, "backend", None), "render_device", None)
        if render_device is not None and render_device.is_cuda():
            if torch is None:
                raise RuntimeError("CUDA rendering requires PyTorch.")
            torch.cuda.synchronize()
        if quest_rgb is not None:
            self._latest_quest_view_rgb = np.asarray(_to_numpy(quest_rgb)).copy()
        return sensor_obs

    def _capture_render_camera_rgb(
        self,
        render_sensor: Any,
        *,
        update_scene_each_capture: bool = True,
    ) -> np.ndarray:
        """Capture one human render camera without the generic multi-camera path.

        ``BaseEnv.step`` already updates the renderer while producing RGB-D
        observations.  Quest-only capture therefore uses
        ``update_scene_each_capture=False`` so only the first capture
        initializes the human camera; subsequent captures use the scene state
        produced by the preceding environment step. Desktop viewer capture
        keeps the per-capture update because its free-camera path is an
        independent rendering path.
        """

        capture = getattr(render_sensor, "capture", None)
        get_obs = getattr(render_sensor, "get_obs", None)
        if not callable(capture) or not callable(get_obs):
            raise AttributeError(
                "The Quest human render camera does not expose capture()/get_obs()."
            )
        for hidden_object in getattr(self, "_hidden_objects", ()):
            hidden_object.show_visual()
        try:
            initialized_sensor = getattr(
                self,
                "_quest_render_camera_initialized_sensor",
                None,
            )
            if update_scene_each_capture or initialized_sensor is not render_sensor:
                self.scene.update_render(
                    update_sensors=False,
                    update_human_render_cameras=True,
                )
                # Reconfiguration creates a new Camera object, so identity
                # also invalidates this one-time initialization automatically.
                self._quest_render_camera_initialized_sensor = render_sensor
            capture()
            observation = get_obs(
                rgb=True,
                depth=False,
                segmentation=False,
                position=False,
            )
            rgb = observation.get("rgb") if isinstance(observation, Mapping) else None
        finally:
            for hidden_object in getattr(
                self, "_hidden_objects", ()
            ):  # pragma: no branch
                hidden_object.hide_visual()
        if rgb is None:
            raise RuntimeError("The Quest human render camera returned no RGB.")
        return np.asarray(_to_numpy(rgb))

    def step(self, action: Any):
        base_obs, reward, terminated, truncated, info = super().step(action)
        # ManiSkill's BaseEnv.step() has already called evaluate() exactly once
        # and merged the result into info. Reuse that result for all metrics.
        step_info = self._evaluation_step_info(info)
        observation = self._build_observation(base_obs=base_obs)
        return (
            self._extract_observation(observation),
            reward,
            terminated,
            truncated,
            step_info,
        )

    def _evaluation_step_info(self, evaluation: Mapping[str, Any]) -> dict[str, Any]:
        """Update benchmark metrics from one already-evaluated physics step."""

        success = evaluation.get(
            "success",
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )
        stage_score = self.compute_stage_score(evaluation)
        elapsed_steps = self._as_per_env_tensor(
            evaluation.get("elapsed_steps", 0), dtype=torch.int64
        )
        return {
            "elapsed_steps": elapsed_steps,
            "success": success,
            "stage_score": stage_score,
        }

    def step_without_observation(self, action: Any):
        """Advance control, physics and metrics without rendering observations.

        This evaluation-only fast path mirrors ManiSkill's step ordering but
        deliberately omits ``get_obs`` and reward construction. It is used only
        between observations that an action-chunk policy will actually consume;
        task success and stage metrics are still evaluated on every control
        step.
        """

        self._step_action(action)
        self._elapsed_steps += 1
        evaluation = self.get_info()
        step_info = self._evaluation_step_info(evaluation)
        if "success" in evaluation:
            terminated = evaluation["success"].clone()
            if "fail" in evaluation:
                terminated = torch.logical_or(terminated, evaluation["fail"])
        elif "fail" in evaluation:
            terminated = evaluation["fail"].clone()
        else:
            terminated = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            None,
            reward,
            terminated,
            truncated,
            step_info,
        )

    def step_without_visual_observation(self, action: Any):
        """Advance one control step and return fresh non-visual observations.

        Policy sessions with proprioceptive or tactile histories need fresh
        low-dimensional values even on steps where no camera frame is used.
        This path preserves the regular physics/evaluation ordering while
        avoiding every renderer call.  The CPU vector wrapper merges the
        returned ``manipulators`` subtree with its cached camera observations.
        """

        self._step_action(action)
        self._elapsed_steps += 1
        evaluation = self.get_info()
        step_info = self._evaluation_step_info(evaluation)
        if "success" in evaluation:
            terminated = evaluation["success"].clone()
            if "fail" in evaluation:
                terminated = torch.logical_or(terminated, evaluation["fail"])
        elif "fail" in evaluation:
            terminated = evaluation["fail"].clone()
        else:
            terminated = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return (
            self._build_policy_lowdim_observation(),
            reward,
            terminated,
            truncated,
            step_info,
        )

    def _before_control_step(self) -> None:
        super()._before_control_step()
        for agent in self.physical_agents:
            if self._tactile_control_accumulation_enabled:
                accumulator = getattr(agent, "begin_tactile_control_step", None)
                if callable(accumulator):
                    accumulator()
            else:
                invalidator = getattr(agent, "invalidate_tactile_cache", None)
                if callable(invalidator):
                    invalidator()
        layout_scene = getattr(self, "layout_scene", None)
        if layout_scene is not None:
            layout_scene.update_joint_position_latches()

    def _after_simulation_step(self) -> None:
        super()._after_simulation_step()
        if not self._tactile_control_accumulation_enabled:
            return
        for agent in self.physical_agents:
            accumulator = getattr(agent, "accumulate_tactile_sim_step", None)
            if callable(accumulator):
                accumulator()

    def _after_control_step(self) -> None:
        if self._tactile_control_accumulation_enabled:
            for agent in self.physical_agents:
                finalizer = getattr(agent, "end_tactile_control_step", None)
                if callable(finalizer):
                    finalizer(expected_sim_steps=self._sim_steps_per_control)
        super()._after_control_step()

    def set_tactile_control_accumulation_enabled(self, enabled: bool) -> None:
        """Enable control-period tactile input without changing task contacts."""

        self._tactile_control_accumulation_enabled = bool(enabled)
        for agent in self.physical_agents:
            if not enabled:
                resetter = getattr(
                    agent, "reset_tactile_control_step_accumulation", None
                )
                if callable(resetter):
                    resetter()
                    continue
            invalidator = getattr(agent, "invalidate_tactile_cache", None)
            if callable(invalidator):
                invalidator()

    def set_policy_tactile_observation_enabled(self, enabled: bool) -> None:
        """Toggle policy-facing taxel projection without changing task physics."""

        self._policy_tactile_observation_enabled = bool(enabled)
        self._invalidate_tactile_caches()

    def _invalidate_tactile_caches(self) -> None:
        """Invalidate dynamic readings without rebuilding persistent queries."""

        if getattr(self, "agent", None) is None:
            # ManiSkill dispatches to this reset() once from BaseEnv.__init__
            # before _reconfigure() has loaded the robot.
            return
        for agent in self.physical_agents:
            invalidator = getattr(agent, "invalidate_tactile_cache", None)
            if callable(invalidator):
                invalidator()

    def get_tactile_cpu_control_step_profiles(
        self,
    ) -> tuple[Mapping[str, float | int], ...]:
        """Return the latest CPU tactile profile for each manipulator role."""

        profiles: list[Mapping[str, float | int]] = []
        for role in self.embodiment_spec.manipulator_roles:
            agent = self.agent_for_role(role)
            getter = getattr(agent, "get_tactile_cpu_control_step_profile", None)
            if callable(getter):
                profile = getter()
                if profile:
                    profiles.append(profile)
        return tuple(profiles)

    def sample_instruction(self) -> str:
        return self.TASK_SETTINGS.instruction

    @property
    def current_instruction(self) -> str:
        """Return the raw language instruction sampled for this episode."""

        if self._current_instruction is None:
            raise RuntimeError("current_instruction is only available after reset().")
        return self._current_instruction

    @property
    def generalization_axis(self) -> GeneralizationAxis | None:
        return self._generalization_axis

    @property
    def generalization_seed(self) -> int | None:
        return self._generalization_seed

    @property
    def generalization_profile(self) -> GeneralizationProfile:
        return self._generalization_profile

    @property
    def generalization_attempt(self) -> int:
        return self._generalization_attempt

    @property
    def generalization_protocol_version(self) -> str:
        return self._generalization_protocol.version

    def generalization_value(self, name: str) -> np.ndarray:
        """Return one per-environment value for the active axis."""

        if self._generalization_context is None:
            raise RuntimeError(
                "generalization_value() requires an active generalization_axis."
            )
        return self._generalization_context.value(name)

    def _parse_generalization_profile(self, value: Any) -> GeneralizationProfile:
        if value is None:
            return GeneralizationProfile.EVALUATION
        if isinstance(value, GeneralizationProfile):
            return value
        try:
            return GeneralizationProfile(value)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(profile.value for profile in GeneralizationProfile)
            raise ValueError(
                f"Unsupported generalization_profile {value!r}; expected one of {supported}."
            ) from exc

    def _parse_generalization_axis(self, value: Any) -> GeneralizationAxis | None:
        if value is None:
            return None
        axis = (
            value
            if isinstance(value, GeneralizationAxis)
            else GeneralizationAxis(value)
        )
        if axis not in self._generalization_protocol.axes:
            raise ValueError(
                f"Generalization axis {axis.value!r} is not enabled by protocol "
                f"{self._generalization_protocol.version!r}."
            )
        return axis

    def _validate_generalization_seed(self, value: Any) -> int | None:
        if self._generalization_axis is None:
            if value is not None:
                raise ValueError(
                    "generalization_seed cannot be set without generalization_axis."
                )
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "generalization_seed must be an integer from the global protocol."
            )
        if value not in self._generalization_protocol.seeds:
            raise ValueError(
                f"generalization_seed={value} is not in fixed protocol seeds "
                f"{self._generalization_protocol.seeds}."
            )
        return value

    def _validate_generalization_attempt(self, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("generalization_attempt must be a non-negative integer.")
        if value >= self._generalization_protocol.max_resample_attempts:
            raise ValueError(
                "generalization_attempt must be smaller than "
                f"max_resample_attempts={self._generalization_protocol.max_resample_attempts}."
            )
        return value

    @staticmethod
    def _validate_generalization_sample_index(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "generalization_sample_index must be a non-negative integer or None."
            )
        return value

    def _make_generalization_context(
        self, *, num_envs: int
    ) -> GeneralizationContext | None:
        if self._generalization_axis is None:
            return None
        axis_spec = global_axis_spec(self._generalization_axis)
        if axis_spec is None:
            profile = getattr(
                self, "_generalization_profile", GeneralizationProfile.EVALUATION
            )
            task_spec = self.GENERALIZATION_SPEC.for_profile(profile)
            axis_spec = task_spec.for_axis(self._generalization_axis)
        if axis_spec is None:
            raise NotImplementedError(
                f"Task {self.TASK_SETTINGS.task_id!r} does not implement the "
                f"{self._generalization_axis.value!r} generalization axis."
            )
        assert self._generalization_seed is not None
        return GeneralizationContext(
            protocol_version=self._generalization_protocol.version,
            task_id=self.TASK_SETTINGS.task_id,
            layout_id=self.current_layout_id_value,
            axis=self._generalization_axis,
            seed=self._generalization_seed,
            attempt=self._generalization_attempt,
            num_envs=num_envs,
            axis_spec=axis_spec,
            sample_indices=(
                (self._generalization_sample_index,)
                if self._generalization_sample_index is not None and num_envs == 1
                else None
            ),
            scope_name=self._generalization_scope_name(self._generalization_axis),
            debug=self._debug_generalization,
        )

    def _generalization_scope_name(self, axis: GeneralizationAxis) -> str:
        if axis is GeneralizationAxis.CAMERA:
            return self.current_camera_rig_spec().rig_id
        if axis is GeneralizationAxis.LIGHTING:
            return "layout_lighting"
        return self.TASK_SETTINGS.task_id

    def _generalization_values(
        self,
        names: tuple[str, ...],
        *,
        cache_name: str,
    ) -> dict[str, np.ndarray]:
        cached_values = getattr(self, cache_name, None)
        if cached_values is None:
            cached_values = {name: self.generalization_value(name) for name in names}
            setattr(self, cache_name, cached_values)
            cache_names = getattr(self, "_generalization_cache_names", None)
            if cache_names is None:
                cache_names = set()
                self._generalization_cache_names = cache_names
            cache_names.add(cache_name)
        return {
            name: np.asarray(value, dtype=np.float64).copy()
            for name, value in cached_values.items()
        }

    def configure_reset_generalization(
        self,
        *,
        axis: str | None,
        profile: str | None,
        seed: int | None,
        attempt: int = 0,
        sample_index: int | None = None,
    ) -> None:
        """Change a reset-time-only generalization without rebuilding the scene."""

        new_axis = self._parse_generalization_axis(axis)
        old_axis = self._generalization_axis
        allowed = {None, GeneralizationAxis.SPATIAL}
        if old_axis not in allowed or new_axis not in allowed:
            raise ValueError(
                "Only nominal/spatial generalization can be changed without rebuilding."
            )
        self._generalization_profile = self._parse_generalization_profile(profile)
        self._generalization_axis = new_axis
        self._generalization_seed = self._validate_generalization_seed(seed)
        self._generalization_attempt = self._validate_generalization_attempt(attempt)
        self._generalization_sample_index = self._validate_generalization_sample_index(
            sample_index
        )
        for cache_name in self._generalization_cache_names:
            if hasattr(self, cache_name):
                delattr(self, cache_name)
        self._generalization_cache_names.clear()
        self._generalization_context = self._make_generalization_context(
            num_envs=self.num_envs
        )

    def _validate_generalization_state(self) -> None:
        if self._generalization_context is None:
            return
        validity = as_validity_tensor(
            self._check_generalization_validity(self._generalization_axis),
            num_envs=self.num_envs,
            device=self.device,
        )
        if bool(validity.all().item()):
            return
        invalid_env_indices = (~validity).nonzero(as_tuple=False).flatten().tolist()
        assert self._generalization_seed is not None
        raise InvalidGeneralizationSample(
            task_id=self.TASK_SETTINGS.task_id,
            layout_id=self.current_layout_id_value,
            axis=self._generalization_axis.value,
            seed=self._generalization_seed,
            attempt=self._generalization_attempt,
            invalid_env_indices=invalid_env_indices,
        )

    def _as_per_env_tensor(self, value: Any, *, dtype: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.to(device=self.device, dtype=dtype)
        else:
            tensor = torch.as_tensor(value, device=self.device, dtype=dtype)
        if tensor.ndim == 0:
            tensor = tensor.repeat(self.num_envs)
        tensor = tensor.reshape(-1)
        if tensor.shape != (self.num_envs,):
            raise ValueError(
                "Expected one value per environment; "
                f"expected [{self.num_envs}], got {list(tensor.shape)}."
            )
        return tensor

    def _check_generalization_validity(self, axis: GeneralizationAxis | None) -> Any:
        """Return one validity flag per environment after reset.

        Tasks override this hook with the checks required by their own sampled
        targets and constraints. Nominal tasks and interface-only tasks are valid.
        """

        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def embodiment_spec(self) -> EmbodimentSpec:
        return self.TASK_SETTINGS.embodiment_spec

    def _binding_index(self, role: ManipulatorRole) -> int:
        for index, binding in enumerate(self.embodiment_spec.manipulators):
            if binding.role is role:
                return index
        raise KeyError(
            f"Embodiment {self.embodiment_spec.embodiment_id!r} has no "
            f"{role.value!r} manipulator."
        )

    def _multi_agent_key(self, role: ManipulatorRole) -> str:
        binding = self.embodiment_spec.binding_for_role(role)
        index = self.embodiment_spec.physical_agent_index_for_role(role)
        return f"{binding.agent_uid}-{index}"

    @property
    def physical_agents(self) -> tuple[Any, ...]:
        """Return loaded ManiSkill agents in deduplicated UID order."""

        agent = getattr(self, "agent", None)
        if agent is None:
            raise RuntimeError("The environment has no loaded robot agent.")
        expected_uids = self.embodiment_spec.robot_uids
        if len(expected_uids) == 1:
            return (agent,)
        agents = getattr(agent, "agents", None)
        if not isinstance(agents, (list, tuple)):
            raise RuntimeError(
                "Expected a ManiSkill MultiAgent for distinct robot UIDs."
            )
        if len(agents) != len(expected_uids):
            raise RuntimeError(
                "Loaded physical agent count does not match embodiment " "robot_uids."
            )
        return tuple(agents)

    def physical_agent_for_uid(self, agent_uid: str) -> Any:
        try:
            agent_index = self.embodiment_spec.robot_uids.index(agent_uid)
        except ValueError as exc:
            raise KeyError(
                f"Embodiment {self.embodiment_spec.embodiment_id!r} has no "
                f"physical agent UID {agent_uid!r}."
            ) from exc
        return self.physical_agents[agent_index]

    def agent_for_role(self, role: ManipulatorRole) -> Any:
        """Return the ManiSkill agent bound to a semantic manipulator role."""

        binding = self.embodiment_spec.binding_for_role(role)
        physical_agent = self.physical_agent_for_uid(binding.agent_uid)
        if binding.manipulator_key is None:
            return physical_agent
        view_factory = getattr(physical_agent, "manipulator_view", None)
        if not callable(view_factory):
            raise RuntimeError(
                f"Physical agent {binding.agent_uid!r} does not expose "
                f"manipulator view {binding.manipulator_key!r}."
            )
        try:
            return view_factory(binding.manipulator_key)
        except KeyError as exc:
            raise RuntimeError(
                f"Physical agent {binding.agent_uid!r} has no manipulator "
                f"view {binding.manipulator_key!r}."
            ) from exc

    @property
    def left_agent(self) -> Any:
        return self.agent_for_role(ManipulatorRole.LEFT)

    @property
    def right_agent(self) -> Any:
        return self.agent_for_role(ManipulatorRole.RIGHT)

    def _control_spec_for_agent(self, agent: Any) -> dict[str, Any]:
        controller = getattr(agent, "controller", None)
        if controller is None:
            return {"components": {}}

        action_mapping = getattr(controller, "action_mapping", None) or {}
        controllers = getattr(controller, "controllers", None) or {}
        components: dict[str, dict[str, Any]] = {}
        for component_name, component_controller in controllers.items():
            cfg = getattr(component_controller, "config", None)
            action_slice = action_mapping.get(component_name)
            original_space = getattr(
                component_controller,
                "_original_single_action_space",
                None,
            )
            single_space = getattr(component_controller, "single_action_space", None)
            action_shape = getattr(single_space, "shape", ()) or ()
            action_lower, action_upper = _space_bounds(original_space or single_space)
            component_spec: dict[str, Any] = {
                "action_slice": action_slice,
                "action_dim": action_shape[0] if action_shape else None,
                "controller_type": type(component_controller).__name__,
                "action_lower": action_lower,
                "action_upper": action_upper,
            }
            if cfg is not None:
                component_spec.update(
                    {
                        "frame": getattr(cfg, "frame", None),
                        "normalize_action": getattr(cfg, "normalize_action", None),
                        "use_delta": getattr(cfg, "use_delta", None),
                        "use_target": getattr(cfg, "use_target", None),
                        "joint_names": getattr(cfg, "joint_names", None),
                        "ee_link": getattr(cfg, "ee_link", None),
                        "root_link_name": getattr(cfg, "root_link_name", None),
                        "pos_lower": _to_numpy(getattr(cfg, "pos_lower", None)),
                        "pos_upper": _to_numpy(getattr(cfg, "pos_upper", None)),
                        "rot_lower": _to_numpy(getattr(cfg, "rot_lower", None)),
                        "rot_upper": _to_numpy(getattr(cfg, "rot_upper", None)),
                    }
                )
            components[component_name] = component_spec

        return {
            "action_space": getattr(agent, "single_action_space", None),
            "components": components,
        }

    def get_control_spec(self) -> dict[str, Any]:
        """Return controller metadata grouped by semantic manipulator role."""

        return {
            "embodiment_id": self.embodiment_spec.embodiment_id,
            "manipulators": {
                binding.role.value: {
                    "agent_uid": binding.agent_uid,
                    **self._control_spec_for_agent(self.agent_for_role(binding.role)),
                }
                for binding in self.embodiment_spec.manipulators
            },
        }

    def get_action_spec(self) -> dict[str, Any]:
        """Return the physical benchmark action schema for this embodiment."""

        manipulator_specs: dict[str, Any] = {}
        for binding in self.embodiment_spec.manipulators:
            role = binding.role
            agent = self.agent_for_role(role)
            joint_names = tuple(
                str(name) for name in getattr(agent, "hand_joint_names", ())
            )
            if not joint_names:
                raise RuntimeError(
                    f"Agent {binding.agent_uid!r} must declare hand_joint_names."
                )
            control_spec = self._control_spec_for_agent(agent)
            arm_component = self._typed_component_spec(control_spec, "arm")
            if not isinstance(arm_component, EEDeltaPoseControlSpec):
                raise RuntimeError(
                    f"Agent {binding.agent_uid!r} must use a 6D delta-pose arm "
                    "controller for the benchmark semantic action."
                )
            if arm_component.action_dim != 6:
                raise RuntimeError(
                    f"Agent {binding.agent_uid!r} arm controller must accept 6 "
                    f"values, got {arm_component.action_dim}."
                )
            expected_frame = "root_translation:root_aligned_body_rotation"
            if arm_component.frame != expected_frame:
                raise RuntimeError(
                    f"Agent {binding.agent_uid!r} arm controller frame must be "
                    f"{expected_frame!r}, got {arm_component.frame!r}."
                )
            if arm_component.pos_lower is None or arm_component.pos_upper is None:
                raise RuntimeError(
                    f"Agent {binding.agent_uid!r} arm controller must declare "
                    "physical translation bounds."
                )
            if arm_component.rot_lower is None or arm_component.rot_upper is None:
                raise RuntimeError(
                    f"Agent {binding.agent_uid!r} arm controller must declare "
                    "physical rotation bounds."
                )
            hand_lower, hand_upper = self._canonical_hand_joint_bounds(
                control_spec=control_spec,
                hand_joint_names=joint_names,
                role=role,
            )
            manipulator_specs[role.value] = {
                "agent_uid": binding.agent_uid,
                "representation": ("end_effector_delta_pose+hand_joint_position"),
                "end_effector_delta_pose": {
                    "shape": (6,),
                    "representation": "delta",
                    "translation_unit": "m",
                    "translation_frame": "robot_root",
                    "translation_lower": arm_component.pos_lower,
                    "translation_upper": arm_component.pos_upper,
                    "rotation_unit": "rad",
                    "rotation_frame": "root_aligned_body_rotation",
                    "rotation_convention": "xyz_euler_delta",
                    "controller_frame": arm_component.frame,
                    "rotation_constraint": (
                        {
                            "type": "scaled_l2_ball",
                            "scale_rad": np.abs(arm_component.rot_lower),
                            "normalized_l2_max": 1.0,
                        }
                        if arm_component.normalize_action
                        else {
                            "type": "box",
                            "lower": arm_component.rot_lower,
                            "upper": arm_component.rot_upper,
                        }
                    ),
                },
                "hand_joint_position": {
                    "shape": (len(joint_names),),
                    "representation": "absolute",
                    "unit": "rad",
                    "joint_names": joint_names,
                    "lower": hand_lower,
                    "upper": hand_upper,
                },
            }
        return {
            "action_format": "benchmark_physical_semantic",
            "action_schema_version": 1,
            "embodiment_id": self.embodiment_spec.embodiment_id,
            "manipulators": manipulator_specs,
        }

    def get_native_action_spec(self) -> dict[str, Any]:
        """Describe the ManiSkill action accepted directly by ``env.step``."""

        manipulator_specs: dict[str, Any] = {}
        for binding in self.embodiment_spec.manipulators:
            role = binding.role
            agent = self.agent_for_role(role)
            control_spec = self._control_spec_for_agent(agent)
            action_shape = tuple(
                int(dim)
                for dim in getattr(
                    control_spec.get("action_space"),
                    "shape",
                    (),
                )
            )
            components: dict[str, Any] = {}
            for component_name, component in control_spec["components"].items():
                start, end = self._action_slice_bounds(component["action_slice"])
                normalize_action = bool(component.get("normalize_action"))
                component_dim = end - start
                components[component_name] = {
                    "action_slice": (start, end),
                    "controller_type": component.get("controller_type"),
                    "normalize_action": normalize_action,
                    "native_lower": (
                        np.full(component_dim, -1.0, dtype=np.float32)
                        if normalize_action
                        else component.get("action_lower")
                    ),
                    "native_upper": (
                        np.full(component_dim, 1.0, dtype=np.float32)
                        if normalize_action
                        else component.get("action_upper")
                    ),
                    "physical_lower": component.get("action_lower"),
                    "physical_upper": component.get("action_upper"),
                    "joint_names": tuple(
                        str(name) for name in component.get("joint_names") or ()
                    ),
                }
            manipulator_specs[role.value] = {
                "agent_action_key": (
                    None
                    if len(self.embodiment_spec.robot_uids) == 1
                    else self._multi_agent_key(role)
                ),
                "shape": action_shape,
                "representation": "maniskill_flat_combined_controller",
                "components": components,
            }
        return {
            "container": (
                "array"
                if len(self.embodiment_spec.robot_uids) == 1
                else "agent_uid_dict"
            ),
            "manipulators": manipulator_specs,
        }

    @staticmethod
    def _action_slice_bounds(action_slice: Any) -> tuple[int, int]:
        if isinstance(action_slice, slice):
            if action_slice.start is None or action_slice.stop is None:
                raise ValueError("Controller action slices must have explicit bounds.")
            return int(action_slice.start), int(action_slice.stop)
        if isinstance(action_slice, (tuple, list)) and len(action_slice) == 2:
            return int(action_slice[0]), int(action_slice[1])
        raise TypeError(f"Unsupported controller action slice {action_slice!r}.")

    @staticmethod
    def _typed_component_spec(
        control_spec: dict[str, Any],
        component_name: str,
    ) -> ControllerComponentSpec:
        return component_spec_from_control_spec(control_spec, component_name)

    def _canonical_hand_joint_bounds(
        self,
        *,
        control_spec: dict[str, Any],
        hand_joint_names: tuple[str, ...],
        role: ManipulatorRole,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower_by_name: dict[str, float] = {}
        upper_by_name: dict[str, float] = {}
        for component_name in control_spec["components"]:
            if component_name == "arm":
                continue
            component = self._typed_component_spec(control_spec, component_name)
            if not isinstance(component, JointPositionControlSpec):
                raise RuntimeError(
                    f"Controller component {component_name!r} for {role.value} "
                    "must use absolute joint-position control."
                )
            if component.action_lower is None or component.action_upper is None:
                raise RuntimeError(
                    f"Controller component {component_name!r} for {role.value} "
                    "must declare physical joint bounds."
                )
            for joint_index, joint_name in enumerate(component.joint_names):
                if joint_name in lower_by_name:
                    raise RuntimeError(
                        f"{role.value} hand joint {joint_name!r} appears in more "
                        "than one controller component."
                    )
                lower_by_name[joint_name] = float(component.action_lower[joint_index])
                upper_by_name[joint_name] = float(component.action_upper[joint_index])

        expected_names = set(hand_joint_names)
        actual_names = set(lower_by_name)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise RuntimeError(
                f"{role.value} controller hand joints do not match the benchmark "
                f"canonical order; missing={missing}, extra={extra}."
            )
        return (
            np.asarray(
                [lower_by_name[name] for name in hand_joint_names],
                dtype=np.float32,
            ),
            np.asarray(
                [upper_by_name[name] for name in hand_joint_names],
                dtype=np.float32,
            ),
        )

    def _encode_manipulator_action(
        self,
        role: ManipulatorRole,
        action: ManipulatorAction,
        *,
        bounds_policy: ActionBoundsPolicy,
    ) -> Any:
        agent = self.agent_for_role(role)
        hand_joint_names = tuple(
            str(name) for name in getattr(agent, "hand_joint_names", ())
        )
        action.validate(
            hand_joint_count=len(hand_joint_names),
            role=role.value,
        )
        hand_joint_index = {
            joint_name: index for index, joint_name in enumerate(hand_joint_names)
        }

        control_spec = self._control_spec_for_agent(agent)
        components = control_spec["components"]
        ordered_parts: list[tuple[int, int, Any]] = []
        encoded_hand_joint_names: list[str] = []
        for component_name, component in components.items():
            start, end = self._action_slice_bounds(component["action_slice"])
            typed_component = self._typed_component_spec(
                control_spec,
                component_name,
            )
            if component_name == "arm":
                if not isinstance(typed_component, EEDeltaPoseControlSpec):
                    raise TypeError(
                        f"Controller component 'arm' for {role.value} must be "
                        "a 6D EE delta-pose controller."
                    )
                expected_frame = "root_translation:root_aligned_body_rotation"
                if (
                    typed_component.action_dim != 6
                    or typed_component.frame != expected_frame
                ):
                    raise ValueError(
                        f"Controller component 'arm' for {role.value} must use "
                        f"a 6D delta pose in frame {expected_frame!r}."
                    )
                # [..., 6] -> [..., 3] translation + [..., 3] XYZ Euler delta.
                part = encode_ee_delta_pose_action(
                    typed_component,
                    pos_delta_root_m=action.end_effector_delta_pose[..., :3],
                    rot_delta_root_rad=action.end_effector_delta_pose[..., 3:6],
                    bounds_policy=bounds_policy,
                )
            else:
                if not isinstance(typed_component, JointPositionControlSpec):
                    raise TypeError(
                        f"Controller component {component_name!r} for "
                        f"{role.value} must use absolute joint-position control."
                    )
                component_joint_names = tuple(
                    str(name) for name in component.get("joint_names") or ()
                )
                unknown_joint_names = sorted(
                    set(component_joint_names) - set(hand_joint_index)
                )
                if unknown_joint_names:
                    raise ValueError(
                        f"Controller component {component_name!r} for "
                        f"{role.value} contains non-hand joints: "
                        + ", ".join(unknown_joint_names)
                        + "."
                    )
                indices = [
                    hand_joint_index[joint_name] for joint_name in component_joint_names
                ]
                canonical_part = action.hand_joint_position[..., indices]
                part = encode_joint_position_action(
                    typed_component,
                    target_qpos=canonical_part,
                    bounds_policy=bounds_policy,
                )
                encoded_hand_joint_names.extend(component_joint_names)
            if part.shape[-1] != end - start:
                raise ValueError(
                    f"Controller component {component_name!r} expects "
                    f"{end - start} values, got {part.shape[-1]}."
                )
            ordered_parts.append((start, end, part))

        if len(encoded_hand_joint_names) != len(set(encoded_hand_joint_names)):
            raise ValueError(
                f"{role.value} hand joints are duplicated across controller "
                "components."
            )
        missing_hand_joint_names = sorted(
            set(hand_joint_names) - set(encoded_hand_joint_names)
        )
        if missing_hand_joint_names:
            raise ValueError(
                f"{role.value} hand joints are missing from controller components: "
                + ", ".join(missing_hand_joint_names)
                + "."
            )

        ordered_parts.sort(key=lambda item: item[0])
        expected_start = 0
        action_parts: list[Any] = []
        for start, end, part in ordered_parts:
            if start != expected_start:
                raise ValueError(
                    f"Controller action slices are not contiguous at {expected_start}."
                )
            expected_start = end
            action_parts.append(part)

        if hasattr(action_parts[0], "detach"):
            return torch.cat(action_parts, dim=-1)
        return np.concatenate(action_parts, axis=-1)

    def encode_action(
        self,
        action: Mapping[Any, Any],
        *,
        bounds_policy: ActionBoundsPolicy = "error",
    ) -> Any:
        """Encode a semantic benchmark action into ManiSkill's native action."""

        if not isinstance(action, Mapping):
            raise TypeError(
                f"Benchmark action must be a mapping, got {type(action).__name__}."
            )
        role_actions: dict[ManipulatorRole, ManipulatorAction] = {}
        for raw_role, raw_action in action.items():
            try:
                role = (
                    raw_role
                    if isinstance(raw_role, ManipulatorRole)
                    else ManipulatorRole(str(raw_role))
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unknown manipulator role {raw_role!r} in benchmark action."
                ) from exc
            if role in role_actions:
                raise ValueError(
                    f"Benchmark action contains duplicate role {role.value!r}."
                )
            role_actions[role] = ManipulatorAction.from_value(raw_action)

        expected_roles = set(self.embodiment_spec.manipulator_roles)
        actual_roles = set(role_actions)
        if actual_roles != expected_roles:
            missing_roles = sorted(role.value for role in expected_roles - actual_roles)
            extra_roles = sorted(role.value for role in actual_roles - expected_roles)
            details: list[str] = []
            if missing_roles:
                details.append("missing " + ", ".join(missing_roles))
            if extra_roles:
                details.append("unexpected " + ", ".join(extra_roles))
            raise ValueError(
                "Benchmark action roles do not match embodiment "
                f"{self.embodiment_spec.embodiment_id!r}: " + "; ".join(details) + "."
            )

        encoded_by_uid: dict[
            str,
            list[tuple[ManipulatorBinding, Any]],
        ] = {}
        for binding in self.embodiment_spec.manipulators:
            encoded_by_uid.setdefault(binding.agent_uid, []).append(
                (
                    binding,
                    self._encode_manipulator_action(
                        binding.role,
                        role_actions[binding.role],
                        bounds_policy=bounds_policy,
                    ),
                )
            )

        physical_actions: dict[str, Any] = {}
        for agent_uid in self.embodiment_spec.robot_uids:
            binding_actions = encoded_by_uid[agent_uid]
            if len(binding_actions) == 1:
                physical_actions[agent_uid] = binding_actions[0][1]
                continue
            actions_by_key: dict[str, Any] = {}
            for binding, encoded_action in binding_actions:
                manipulator_key = binding.manipulator_key
                if manipulator_key is None:
                    raise RuntimeError(
                        f"Shared physical agent {agent_uid!r} requires "
                        "manipulator keys for action composition."
                    )
                actions_by_key[manipulator_key] = encoded_action
            physical_agent = self.physical_agent_for_uid(agent_uid)
            composer = getattr(
                physical_agent,
                "compose_manipulator_actions",
                None,
            )
            if not callable(composer):
                raise RuntimeError(
                    f"Shared physical agent {agent_uid!r} does not implement "
                    "compose_manipulator_actions()."
                )
            physical_actions[agent_uid] = composer(actions_by_key)

        if len(physical_actions) == 1:
            return next(iter(physical_actions.values()))
        return {
            f"{agent_uid}-{agent_index}": physical_actions[agent_uid]
            for agent_index, agent_uid in enumerate(self.embodiment_spec.robot_uids)
        }

    def get_manipulator_state_spec(self) -> dict[str, dict[str, Any]]:
        """Return joint bounds and tactile layout metadata by role."""

        result: dict[str, dict[str, Any]] = {}
        for binding in self.embodiment_spec.manipulators:
            agent = self.agent_for_role(binding.role)
            if binding.manipulator_key is None:
                joints = tuple(agent.robot.get_active_joints())
                joint_names = tuple(str(joint.name) for joint in joints)
                raw_qlimits = agent.robot.get_qlimits()
            else:
                joint_names = tuple(
                    str(name) for name in getattr(agent, "joint_names", ())
                )
                qlimits_reader = getattr(agent, "get_qlimits", None)
                if not joint_names or not callable(qlimits_reader):
                    raise RuntimeError(
                        f"Manipulator view {binding.manipulator_key!r} for "
                        f"agent {binding.agent_uid!r} must expose joint_names "
                        "and get_qlimits()."
                    )
                raw_qlimits = qlimits_reader()
            qlimits = np.asarray(_to_numpy(raw_qlimits))
            if qlimits.ndim == 3:
                qlimits = qlimits[0]
            if qlimits.shape != (len(joint_names), 2):
                raise RuntimeError(
                    f"Robot {binding.agent_uid!r} qlimits shape {qlimits.shape} "
                    f"does not match {len(joint_names)} active joints."
                )
            taxel_specs = tuple(getattr(agent, "taxel_specs", ()))
            result[binding.role.value] = {
                "agent_uid": binding.agent_uid,
                "joint_names": joint_names,
                "lower": np.asarray(qlimits[:, 0], dtype=np.float32),
                "upper": np.asarray(qlimits[:, 1], dtype=np.float32),
                "taxel_names": tuple(
                    str(getattr(taxel_spec, "name")) for taxel_spec in taxel_specs
                ),
                "taxel_finger_names": tuple(
                    (
                        None
                        if getattr(taxel_spec, "finger_name", None) is None
                        else str(getattr(taxel_spec, "finger_name"))
                    )
                    for taxel_spec in taxel_specs
                ),
                "taxel_region_names": tuple(
                    str(getattr(taxel_spec, "region_name"))
                    for taxel_spec in taxel_specs
                ),
            }
        return result

    def get_manipulator_control_diagnostics(self) -> dict[str, dict[str, Any]]:
        """Return live controller positions/errors for teleoperation diagnostics."""

        diagnostics: dict[str, dict[str, Any]] = {}
        for binding in self.embodiment_spec.manipulators:
            agent = self.agent_for_role(binding.role)
            error_reader = getattr(
                agent,
                "get_arm_ee_target_position_error",
                None,
            )
            role_diagnostics: dict[str, Any] = {}
            if callable(error_reader):
                position_error = error_reader()
                if position_error is not None:
                    role_diagnostics["arm_ee_target_position_error_m"] = position_error
            position_reader = getattr(
                agent,
                "get_arm_ee_target_position_diagnostics",
                None,
            )
            if callable(position_reader):
                position_values = position_reader()
                if position_values is not None:
                    if not isinstance(position_values, Mapping):
                        raise TypeError(
                            "Arm EE target position diagnostics must be a mapping."
                        )
                    for field_name in (
                        "target_position_m",
                        "current_position_m",
                        "position_error_m",
                    ):
                        if field_name in position_values:
                            role_diagnostics[
                                (
                                    f"arm_ee_{field_name}"
                                    if field_name != "position_error_m"
                                    else "arm_ee_target_position_error_m"
                                )
                            ] = position_values[field_name]
            if role_diagnostics:
                diagnostics[binding.role.value] = role_diagnostics
        return diagnostics

    def get_manipulator_tactile_diagnostics(self) -> dict[str, dict[str, Any]]:
        """Return completed control-period tactile mapping diagnostics."""

        diagnostics: dict[str, dict[str, Any]] = {}
        for binding in self.embodiment_spec.manipulators:
            agent = self.agent_for_role(binding.role)
            reader = getattr(agent, "get_tactile_contact_diagnostics", None)
            if not callable(reader):
                continue
            diagnostic = reader()
            if diagnostic is None:
                # The first control period has not completed yet.  This is a
                # valid lifecycle state, not a zero or NaN measurement.
                continue
            if hasattr(diagnostic, "__dataclass_fields__"):
                values = {
                    field_name: getattr(diagnostic, field_name)
                    for field_name in diagnostic.__dataclass_fields__
                }
            elif isinstance(diagnostic, Mapping):
                values = dict(diagnostic)
            else:
                raise TypeError(
                    "get_tactile_contact_diagnostics() must return a dataclass "
                    "or mapping."
                )
            diagnostics[binding.role.value] = values
        return diagnostics

    def set_tactile_diagnostics_enabled(self, enabled: bool) -> None:
        """Toggle expensive tactile force/moment diagnostics for test runs."""

        for binding in self.embodiment_spec.manipulators:
            agent = self.agent_for_role(binding.role)
            setter = getattr(agent, "set_contact_diagnostics_enabled", None)
            if callable(setter):
                setter(enabled)

    def get_manipulator_contact_control_state(self) -> dict[str, dict[str, Any]]:
        """Return parent-link contact resultants for action-side filtering."""

        result: dict[str, dict[str, Any]] = {}
        for binding in self.embodiment_spec.manipulators:
            role = binding.role.value
            agent = self.agent_for_role(binding.role)
            reader = getattr(agent, "get_tactile_contact_state", None)
            if not callable(reader):
                raise RuntimeError(
                    f"Robot {type(agent).__name__} for role {role!r} does not "
                    "provide tactile contact control state."
                )
            state = reader()
            controller = getattr(agent, "controller", None)
            if isinstance(controller, Mapping):
                controller = controller.get("arm")
            else:
                controllers = getattr(controller, "controllers", None)
                if isinstance(controllers, Mapping):
                    controller = controllers.get("arm")
            root_link = getattr(controller, "root_link", None)
            root_pose = getattr(root_link, "pose", None)
            root_rotation = getattr(root_pose, "q", None)
            if root_rotation is None:
                robot_root = getattr(getattr(agent, "robot", None), "root", None)
                robot_root_pose = getattr(robot_root, "pose", None)
                root_rotation = getattr(robot_root_pose, "q", None)
            if root_rotation is None:
                raise RuntimeError(
                    f"Could not resolve the world rotation of the root link for "
                    f"manipulator role {role!r}."
                )
            if binding.manipulator_key is None:
                local_joint_names = tuple(
                    str(joint.name) for joint in agent.robot.get_active_joints()
                )
            else:
                local_joint_names = tuple(
                    str(name) for name in getattr(agent, "joint_names", ())
                )
            if not local_joint_names:
                raise RuntimeError(
                    f"Robot {type(agent).__name__} for role {role!r} does not "
                    "provide role-local joint names."
                )
            local_joint_index_by_name = {
                name: index for index, name in enumerate(local_joint_names)
            }
            hand_joint_names = tuple(
                str(name) for name in getattr(agent, "hand_joint_names", ())
            )
            missing_hand_joint_names = tuple(
                name
                for name in hand_joint_names
                if name not in local_joint_index_by_name
            )
            if missing_hand_joint_names:
                raise RuntimeError(
                    f"Robot {type(agent).__name__} for role {role!r} is "
                    "missing hand joints from its role-local state: "
                    + ", ".join(missing_hand_joint_names)
                )
            result[role] = {
                "parent_force_world": _to_numpy(state.parent_force_world),
                "parent_normal_force_world": _to_numpy(state.parent_normal_force_world),
                "finger_force_world": _to_numpy(state.finger_force_world),
                "taxel_force_world": _to_numpy(state.taxel_force_world),
                "hand_force_world": _to_numpy(state.hand_force_world),
                "parent_link_names": tuple(state.parent_link_names),
                "finger_names": tuple(state.finger_names),
                "parent_finger_indices": tuple(state.parent_finger_indices),
                "joint_names": hand_joint_names,
                "hand_joint_indices": tuple(
                    local_joint_index_by_name[name] for name in hand_joint_names
                ),
                "root_rotation_world_wxyz": _to_numpy(root_rotation),
            }
        return result

    def get_episode_metadata(self) -> dict[str, Any]:
        """Return stable embodiment metadata suitable for episode recording."""

        profile = getattr(
            self, "_generalization_profile", GeneralizationProfile.EVALUATION
        )
        axis = getattr(self, "_generalization_axis", None)
        protocol = getattr(self, "_generalization_protocol", None)
        tactile_layouts: dict[str, dict[str, Any]] = {}
        for manipulator_role in self.embodiment_spec.manipulator_roles:
            agent = self.agent_for_role(manipulator_role)
            taxel_link_names = tuple(getattr(agent, "taxel_link_names", ()))
            if not taxel_link_names:
                raise RuntimeError(
                    f"Robot {type(agent).__name__} for role "
                    f"{manipulator_role.value!r} does not expose a taxel layout."
                )
            tactile_layouts[manipulator_role.value] = {
                "taxel_count": len(taxel_link_names),
                "layout_id": getattr(agent, "taxel_layout_id"),
                "layout_version": getattr(agent, "taxel_layout_version"),
                "layout_hash": getattr(agent, "taxel_layout_hash"),
                "force_unit": getattr(agent, "taxel_force_unit"),
                "sensor_model": getattr(agent, "taxel_sensor_model"),
            }

        return {
            "embodiment_id": self.embodiment_spec.embodiment_id,
            "generalization": {
                "profile": (profile.value if axis is not None else None),
                "axis": (axis.value if axis is not None else None),
                "protocol_version": getattr(protocol, "version", None),
                "seed": getattr(self, "_generalization_seed", None),
                "attempt": getattr(self, "_generalization_attempt", 0),
                "sample_index": getattr(self, "_generalization_sample_index", None),
                "sampled_values": (
                    {
                        name: np.asarray(values).tolist()
                        for name, values in generalization_context.sampled_values().items()
                    }
                    if (
                        generalization_context := getattr(
                            self, "_generalization_context", None
                        )
                    )
                    is not None
                    else {}
                ),
                "range_version": (
                    "collection_v1"
                    if profile is GeneralizationProfile.COLLECTION
                    else "evaluation_v1"
                ),
            },
            "robot_uids": self.embodiment_spec.robot_uids,
            "manipulator_roles": tuple(
                role.value for role in self.embodiment_spec.manipulator_roles
            ),
            "camera_roles": tuple(
                role.value for role in self.embodiment_spec.camera_roles
            ),
            "control_mode": self.embodiment_spec.control_mode.value,
            "control_frequency_hz": int(self._control_freq),
            "tactile_layouts": tactile_layouts,
            "joint_state_specs": self.get_manipulator_state_spec(),
            "benchmark_action_spec": self.get_action_spec(),
            "native_action_spec": self.get_native_action_spec(),
        }

    def camera_sensor_uid(self, role: CameraRole) -> str:
        """Return the concrete ManiSkill sensor UID for a semantic camera role."""

        if role in SCENE_CAMERA_UID_BY_ROLE:
            return SCENE_CAMERA_UID_BY_ROLE[role]
        wrist_uid = WRIST_CAMERA_UID_BY_ROLE.get(role)
        if wrist_uid is None:
            raise KeyError(f"Unsupported camera role {role.value!r}.")
        manipulator_role = (
            ManipulatorRole.LEFT
            if role is CameraRole.WRIST_LEFT
            else ManipulatorRole.RIGHT
        )
        if len(self.embodiment_spec.robot_uids) == 1:
            self._binding_index(manipulator_role)
            return wrist_uid
        return f"{self._multi_agent_key(manipulator_role)}-{wrist_uid}"

    def _declared_camera_uids(self) -> set[str]:
        return {
            self.camera_sensor_uid(camera_role)
            for camera_role in self.embodiment_spec.camera_roles
        }

    def _configured_camera_uids(self) -> set[str]:
        sensor_configs = getattr(self, "_sensor_configs", {}) or {}
        wrist_suffixes = tuple(
            f"-{camera_uid}" for camera_uid in WRIST_CAMERA_UID_BY_ROLE.values()
        )
        return {
            str(sensor_uid)
            for sensor_uid in sensor_configs
            if sensor_uid in SCENE_CAMERA_UID_BY_ROLE.values()
            or sensor_uid in WRIST_CAMERA_UID_BY_ROLE.values()
            or str(sensor_uid).endswith(wrist_suffixes)
        }

    def _validate_required_camera_configs(self) -> None:
        # State-only rollouts deliberately do not construct camera sensors.
        # Camera declarations remain part of the canonical task contract, but
        # validating their instantiated UIDs here would force rendering even
        # when a policy has not requested visual observations.
        obs_mode = str(getattr(self, "obs_mode", self.TASK_SETTINGS.obs_mode)).lower()
        if obs_mode in STATE_ONLY_OBS_MODES:
            return
        declared_camera_uids = self._declared_camera_uids()
        actual_camera_uids = self._configured_camera_uids()
        missing_camera_uids = sorted(declared_camera_uids - actual_camera_uids)
        undeclared_camera_uids = sorted(actual_camera_uids - declared_camera_uids)
        if not missing_camera_uids and not undeclared_camera_uids:
            return

        details: list[str] = []
        if missing_camera_uids:
            details.append(
                "declared camera sensors missing from ManiSkill configs: "
                + ", ".join(missing_camera_uids)
            )
        if undeclared_camera_uids:
            details.append(
                "ManiSkill camera configs not declared in required_sensors: "
                + ", ".join(undeclared_camera_uids)
            )
        required_sensors = ", ".join(self.TASK_SETTINGS.required_sensors) or "<empty>"
        raise ValueError(
            f"Task {self.TASK_SETTINGS.task_id!r} camera config mismatch. "
            f"required_sensors=[{required_sensors}]. " + " ".join(details)
        )

    def _validate_required_observation_sensors(
        self,
        observation: Observation,
    ) -> None:
        sensor_values: dict[str, Any] = {}
        for camera_role, camera in observation.cameras.items():
            sensor_values[f"{camera_role.value}_rgb"] = camera.rgb
            sensor_values[f"{camera_role.value}_depth"] = camera.depth
        for manipulator_role, manipulator in observation.manipulators.items():
            sensor_values[f"{manipulator_role.value}_taxel_normal_force"] = (
                manipulator.taxel_normal_force
            )
        sensor_values["pointcloud"] = observation.pointcloud

        obs_mode = str(getattr(self, "obs_mode", self.TASK_SETTINGS.obs_mode)).lower()
        skip_visual_sensors = obs_mode in STATE_ONLY_OBS_MODES
        missing_sensors = [
            required_sensor
            for required_sensor in self.TASK_SETTINGS.required_sensors
            if not (skip_visual_sensors and required_sensor in VISUAL_REQUIRED_SENSORS)
            if sensor_values.get(required_sensor) is None
        ]
        if not missing_sensors:
            return
        raise ValueError(
            f"Task {self.TASK_SETTINGS.task_id!r} declared required_sensors that "
            "are missing from the current observation: "
            + ", ".join(sorted(missing_sensors))
            + ". Check the embodiment camera configs, obs_mode, and manipulator "
            "observation mapping."
        )

    def _base_obs_dict(self, base_obs: Any) -> Mapping[str, Any]:
        if base_obs is None:
            return {}
        if not isinstance(base_obs, Mapping):
            raise TypeError(
                "DexVLABench environments require a dict-like ManiSkill observation "
                "to build the model observation. Use obs_mode='rgbd' or another "
                f"dict observation mode instead of a flat observation type. Got "
                f"{type(base_obs).__name__}."
            )
        return base_obs

    @property
    def current_layout_id(self) -> LayoutId | None:
        return self._selected_layout_id

    @property
    def current_layout_id_value(self) -> str:
        return (
            self._selected_layout_id.value
            if self._selected_layout_id is not None
            else ""
        )

    @property
    def current_layout_spec(self) -> LayoutSpec | None:
        return self._selected_layout_spec

    def require_current_layout_spec(self) -> LayoutSpec:
        if self._selected_layout_spec is None:
            raise RuntimeError(
                f"Task {self.TASK_SETTINGS.task_id!r} requires a selected layout, "
                "but current_layout_spec is None."
            )
        return self._selected_layout_spec

    def require_current_interaction_site(self) -> InteractionSiteSpec:
        site_id = self.INTERACTION_SITE_ID
        if site_id is None:
            raise RuntimeError(
                f"Task {self.TASK_SETTINGS.task_id!r} does not declare "
                "INTERACTION_SITE_ID."
            )
        return self.require_current_layout_spec().interaction_site(site_id)

    def interaction_site_frame_pose(self) -> PoseSpec:
        return self.require_current_interaction_site().frame_pose

    def interaction_site_robot_base_center_pose(self) -> PoseSpec:
        return self.require_current_interaction_site().robot_base_center_pose

    def interaction_site_robot_base_poses(
        self,
    ) -> dict[ManipulatorRole, PoseSpec]:
        """Return role-keyed base poses using the site's embodiment center."""

        site = self.require_current_interaction_site()
        center_pose = site.robot_base_center_pose
        roles = self.embodiment_spec.manipulator_roles
        if len(roles) == 1:
            return {roles[0]: center_pose}
        if set(roles) != {ManipulatorRole.LEFT, ManipulatorRole.RIGHT}:
            raise RuntimeError(
                "Bimanual interaction-site placement requires exactly left and "
                f"right roles, got {[role.value for role in roles]}."
            )

        if site.bimanual_robot_base_center_pose is not None:
            center_pose = site.bimanual_robot_base_center_pose

        spacing_m = site.bimanual_base_spacing_m
        half_spacing_m = 0.5 * spacing_m
        role_offsets = {
            ManipulatorRole.LEFT: (0.0, half_spacing_m, 0.0),
            ManipulatorRole.RIGHT: (0.0, -half_spacing_m, 0.0),
        }
        return {
            role: PoseSpec(
                position_xyz=tuple(
                    float(value)
                    for value in center_pose.transform_point(role_offsets[role])
                ),
                quaternion_wxyz=center_pose.quaternion_wxyz,
            )
            for role in roles
        }

    def interaction_site_robot_base_sapien_poses(self) -> Any:
        """Return one pose or an ordered pose list accepted by ManiSkill."""

        role_poses = self.interaction_site_robot_base_poses()
        ordered_poses = [
            pose_spec_to_sapien_pose(role_poses[role])
            for role in self.embodiment_spec.manipulator_roles
        ]
        if len(ordered_poses) == 1:
            return ordered_poses[0]
        return ordered_poses

    def interaction_site_robot_base_sapien_pose(self) -> Any:
        """Return the single-manipulator base pose."""

        if len(self.embodiment_spec.manipulator_roles) != 1:
            raise RuntimeError(
                "interaction_site_robot_base_sapien_pose() is only valid for "
                "single-manipulator embodiments; use "
                "interaction_site_robot_base_sapien_poses() for bimanual tasks."
            )
        return self.interaction_site_robot_base_sapien_poses()

    def interaction_site_local_point_to_world(
        self,
        local_xyz: tuple[float, float, float],
    ) -> np.ndarray:
        return self.interaction_site_frame_pose().transform_point(local_xyz)

    def interaction_site_local_pose_to_world(self, local_pose: PoseSpec) -> PoseSpec:
        return self.interaction_site_frame_pose().transform_pose(local_pose)

    def interaction_site_local_point_tensor(
        self,
        local_xyz: tuple[float, float, float],
        batch_size: int,
    ) -> Any:
        if torch is None:
            raise ImportError("PyTorch is required for interaction-site tensors.")
        world_xyz = self.interaction_site_local_point_to_world(local_xyz)
        # [3] -> [B, 3]: replicate a site-local point in world coordinates.
        return (
            torch.as_tensor(
                world_xyz,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

    def interaction_site_quaternion_tensor(self, batch_size: int) -> Any:
        if torch is None:
            raise ImportError("PyTorch is required for interaction-site tensors.")
        quaternion_wxyz = self.interaction_site_frame_pose().quaternion_wxyz
        # [4] -> [B, 4]: replicate the site orientation in world coordinates.
        return (
            torch.as_tensor(
                quaternion_wxyz,
                device=self.device,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

    def current_camera_rig_spec(self) -> CameraRigSpec:
        site = self.require_current_interaction_site()
        task_settings = self.TASK_SETTINGS
        return CameraRigRegistry.get(
            site.camera_rig_id,
            layout_id=self.current_layout_id_value,
            site_id=site.site_id,
            task_id=None if task_settings is None else task_settings.task_id,
        )

    def camera_view_world_points(
        self,
        view_spec: CameraViewSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        site_frame_pose = self.interaction_site_frame_pose()
        eye_world_xyz = site_frame_pose.transform_point(view_spec.eye_local_xyz)
        target_world_xyz = site_frame_pose.transform_point(view_spec.target_local_xyz)
        return eye_world_xyz, target_world_xyz

    def _camera_config_from_view(self, view_spec: CameraViewSpec) -> CameraConfig:
        if CameraConfig is None or sapien_utils is None:
            raise ImportError("ManiSkill is required to build camera configs.")
        up_world_xyz = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        if view_spec.up_local_xyz is not None:
            # [3, 3] @ [3] -> [3]: rotate a direction without translation.
            up_world_xyz = self.interaction_site_frame_pose().rotation_matrix @ np.asarray(
                view_spec.up_local_xyz, dtype=np.float64
            )
        if (
            getattr(self, "_generalization_axis", None)
            is GeneralizationAxis.CAMERA
        ):
            samples = self._generalization_values(
                CAMERA_PARAMETER_NAMES,
                cache_name="_camera_generalization_values",
            )
            changed_view = camera_view_values(view_spec, samples)
            site_pose = self.interaction_site_frame_pose()
            rotation = site_pose.rotation_matrix.astype(np.float64)
            position = np.asarray(site_pose.position_xyz, dtype=np.float64)
            # [B, 3] @ [3, 3].T + [3] -> [B, 3].
            eye_world_xyz = changed_view.eye_local_xyz @ rotation.T + position
            target_world_xyz = changed_view.target_local_xyz @ rotation.T + position
            camera_pose = sapien_utils.look_at(
                eye=torch.as_tensor(eye_world_xyz, dtype=torch.float32),
                target=torch.as_tensor(target_world_xyz, dtype=torch.float32),
                up=up_world_xyz.tolist(),
            )
            fov_values = changed_view.fov_rad.tolist()
            fov: float | list[float] = (
                fov_values[0] if len(fov_values) == 1 else fov_values
            )
        else:
            eye_world_xyz, target_world_xyz = self.camera_view_world_points(view_spec)
            camera_pose = sapien_utils.look_at(
                eye=eye_world_xyz.tolist(),
                target=target_world_xyz.tolist(),
                up=up_world_xyz.tolist(),
            )
            fov = view_spec.fov_rad
        return CameraConfig(
            uid=view_spec.camera_uid,
            pose=camera_pose,
            width=view_spec.width,
            height=view_spec.height,
            fov=fov,
            near=view_spec.near,
            far=view_spec.far,
        )

    def _rectangular_light_aperture(
        self,
    ) -> RectangularLightApertureSpec | None:
        lighting_spec = self.require_current_layout_spec().lighting
        if lighting_spec is None:
            return None
        apertures = [
            light.aperture
            for light in lighting_spec.directional_lights
            if light.aperture is not None
        ]
        if len(apertures) > 1:
            raise RuntimeError(
                "The current raster shader supports exactly one rectangular "
                f"light aperture, but layout {self.current_layout_id_value!r} "
                f"declares {len(apertures)}."
            )
        if apertures and len(lighting_spec.directional_lights) != 1:
            raise RuntimeError(
                "A layout using rectangular window sunlight must declare exactly "
                "one directional light because the raster shader applies the "
                "aperture to the shadowed directional-light pass."
            )
        return apertures[0] if apertures else None

    def _configure_rectangular_aperture_shaders(
        self,
        environment_kwargs: dict[str, Any],
    ) -> None:
        """Apply the aperture shader to every camera in a Studio environment."""

        aperture = self._rectangular_light_aperture()
        if aperture is None:
            return
        legacy_shader_dir = environment_kwargs.get("shader_dir")
        if legacy_shader_dir is not None:
            if legacy_shader_dir not in {"minimal", "default"}:
                raise ValueError(
                    "shader_dir must be 'minimal' or 'default' for a layout with "
                    "rectangular window sunlight. Ray-tracing and external shader "
                    "packs do not contain the aperture implementation."
                )
            environment_kwargs.pop("shader_dir")

        sensor_shader_path = rectangular_aperture_camera_shader_config(
            aperture
        ).shader_pack
        viewer_shader_path = rectangular_aperture_viewer_shader_config(
            aperture
        ).shader_pack
        for config_key, shader_path in (
            ("sensor_configs", sensor_shader_path),
            ("human_render_camera_configs", sensor_shader_path),
            ("viewer_camera_configs", viewer_shader_path),
        ):
            existing_config = environment_kwargs.get(config_key)
            if existing_config is None:
                merged_config: dict[str, Any] = {}
            elif isinstance(existing_config, Mapping):
                merged_config = copy.deepcopy(dict(existing_config))
            else:
                raise TypeError(
                    f"{config_key} must be a mapping, got "
                    f"{type(existing_config).__name__}."
                )
            self._remove_builtin_shader_overrides(
                merged_config,
                config_key=config_key,
            )
            merged_config["shader_pack"] = shader_path
            environment_kwargs[config_key] = merged_config

    @staticmethod
    def _remove_builtin_shader_overrides(
        camera_configs: dict[str, Any],
        *,
        config_key: str,
    ) -> None:
        """Remove stock shader choices that the aperture shader supersedes."""

        shader_mappings: list[tuple[str, dict[str, Any]]] = [
            (config_key, camera_configs)
        ]
        shader_mappings.extend(
            (f"{config_key}.{camera_uid}", camera_config)
            for camera_uid, camera_config in camera_configs.items()
            if isinstance(camera_config, dict)
        )
        for config_path, camera_config in shader_mappings:
            if camera_config.get("shader_config") is not None:
                raise ValueError(
                    f"{config_path} supplies a custom shader_config, which cannot "
                    "be combined with rectangular window sunlight."
                )
            camera_config.pop("shader_config", None)
            shader_pack = camera_config.pop("shader_pack", None)
            if shader_pack not in {None, "minimal", "default"}:
                raise ValueError(
                    f"{config_path} selects unsupported shader_pack "
                    f"{shader_pack!r}; rectangular window sunlight currently "
                    "supports the minimal sensor shader and default viewer shader."
                )

    def _load_lighting(self, options: dict[str, Any]) -> None:
        """Load nominal or globally changed lighting for the selected layout."""

        lighting_spec = self.require_current_layout_spec().lighting
        if lighting_spec is None:
            raise RuntimeError(
                f"Layout {self.current_layout_id_value!r} does not declare nominal "
                "lighting. Every benchmark layout must define a LayoutLightingSpec."
            )

        if getattr(self, "_generalization_axis", None) is GeneralizationAxis.LIGHTING:
            samples = self._generalization_values(
                LIGHTING_PARAMETER_NAMES,
                cache_name="_lighting_generalization_values",
            )
            load_layout_lighting_batch(
                self.scene,
                lighting_values(lighting_spec, samples),
            )
            return
        load_layout_lighting(self.scene, lighting_spec)

    @property
    def _default_sensor_configs(self) -> list[CameraConfig]:
        camera_rig = self.current_camera_rig_spec()
        return [
            self._camera_config_from_view(camera_rig.context_view),
            self._camera_config_from_view(camera_rig.interaction_view),
        ]

    @property
    def _default_human_render_camera_configs(self) -> CameraConfig:
        if CameraConfig is None or sapien_utils is None:
            raise ImportError("ManiSkill is required to build camera configs.")
        render_spec = self.human_render_camera_spec()
        viewer_pose_spec = self.human_viewer_camera_pose_spec()
        if viewer_pose_spec is None:
            eye_world_xyz, target_world_xyz = self.interaction_site_local_camera_points(
                render_spec.eye_local_xyz,
                render_spec.target_local_xyz,
            )
            camera_pose = sapien_utils.look_at(
                eye=eye_world_xyz.tolist(),
                target=target_world_xyz.tolist(),
            )
        else:
            camera_pose = pose_spec_to_sapien_pose(
                PoseSpec(
                    position_xyz=viewer_pose_spec.position_world_xyz,
                    quaternion_wxyz=viewer_pose_spec.quaternion_world_wxyz,
                )
            )
        return CameraConfig(
            uid="render_camera",
            pose=camera_pose,
            width=render_spec.width,
            height=render_spec.height,
            fov=render_spec.fov_rad,
            near=render_spec.near,
            far=render_spec.far,
        )

    def human_viewer_camera_pose_spec(
        self,
    ) -> HumanViewerCameraPoseSpec | None:
        """Return the current layout's exact free Human Viewer world pose."""

        camera_poses = self.HUMAN_VIEWER_CAMERA_POSES
        if not camera_poses:
            return None
        layout_id = self.current_layout_id_value
        if not layout_id:
            raise RuntimeError(
                "A layout must be selected before resolving the Human Viewer "
                "camera pose."
            )
        try:
            return camera_poses[layout_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Task {self.TASK_SETTINGS.task_id!r} has no Human Viewer camera "
                f"pose for layout {layout_id!r}. Configured layouts: "
                f"{sorted(camera_poses)!r}."
            ) from exc

    def human_render_camera_spec(self) -> RenderCameraSpec:
        """Return the task-specific default Human Viewer camera spec.

        Subclasses normally set ``HUMAN_RENDER_CAMERA_SPEC``. The context view
        fallback keeps existing tasks working until their render camera is
        individually tuned.
        """

        render_spec = self.HUMAN_RENDER_CAMERA_SPEC
        if render_spec is not None:
            return render_spec
        context_view = self.current_camera_rig_spec().context_view
        return RenderCameraSpec(
            eye_local_xyz=context_view.eye_local_xyz,
            target_local_xyz=context_view.target_local_xyz,
            fov_rad=context_view.fov_rad,
            width=512,
            height=512,
            near=context_view.near,
            far=context_view.far,
        )

    def interaction_site_local_camera_points(
        self,
        eye_local_xyz: tuple[float, float, float],
        target_local_xyz: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform an eye/target pair from site-local to world coordinates."""

        return (
            self.interaction_site_local_point_to_world(eye_local_xyz),
            self.interaction_site_local_point_to_world(target_local_xyz),
        )

    def _parse_layout_id(self, value: Any) -> LayoutId:
        if isinstance(value, LayoutId):
            return value
        if isinstance(value, str):
            try:
                return LayoutId(value)
            except ValueError as value_error:
                supported_values = ", ".join(layout.value for layout in LayoutId)
                raise ValueError(
                    f"Unknown layout_id {value!r}. Supported values: {supported_values}."
                ) from value_error
        raise TypeError(
            f"layout_id must be a string or LayoutId, got {type(value).__name__}."
        )

    def _select_episode_layout(
        self,
        seed: int | None,
        options: dict[str, Any] | None,
    ) -> None:
        supported_layouts = list(self.TASK_SETTINGS.supported_layouts)
        if not supported_layouts:
            self._selected_layout_id = None
            self._selected_layout_spec = None
            return

        requested_layout = (
            options.get("layout_id") if isinstance(options, dict) else None
        )
        if requested_layout is not None:
            layout_id = self._parse_layout_id(requested_layout)
            if layout_id not in supported_layouts:
                supported_values = ", ".join(
                    layout.value for layout in supported_layouts
                )
                raise ValueError(
                    f"Task {self.TASK_SETTINGS.task_id!r} does not support layout "
                    f"{layout_id.value!r}. Supported layouts: {supported_values}."
                )
        else:
            # Layout is an evaluator-controlled dimension, not implicit episode
            # randomness. The first scene layout is the deterministic default.
            layout_id = supported_layouts[0]

        self._selected_layout_id = layout_id
        self._selected_layout_spec = LayoutRegistry.get(layout_id)

    def _validate_reset_layout_request(self, options: dict[str, Any] | None) -> None:
        if not isinstance(options, dict) or "layout_id" not in options:
            return
        requested_layout = self._parse_layout_id(options["layout_id"])
        if self._selected_layout_id is None:
            self._select_episode_layout(seed=None, options=options)
            return
        if requested_layout != self._selected_layout_id:
            raise ValueError(
                "layout_id cannot change after the ManiSkill scene has been built. "
                f"This env was initialized with {self._selected_layout_id.value!r}, "
                f"but reset requested {requested_layout.value!r}. Create a new env "
                "instance for a different layout."
            )

    def _build_observation(self, base_obs: dict | None = None) -> Observation:
        base_obs_dict = self._base_obs_dict(base_obs)
        obs_mode = getattr(self, "obs_mode", self.TASK_SETTINGS.obs_mode)
        use_pointcloud = "pointcloud" in str(obs_mode).lower()
        return Observation(
            cameras=(
                {} if use_pointcloud else self._build_camera_observations(base_obs_dict)
            ),
            manipulators=self._build_manipulator_observations(base_obs_dict),
            pointcloud=(base_obs_dict.get("pointcloud") if use_pointcloud else None),
        )

    def _build_policy_lowdim_observation(self) -> dict[str, Any]:
        """Build only the manipulator subtree used by policy adapters.

        ``ManiSkillBaseEnv.get_obs`` couples proprioception with sensor capture
        for visual observation modes.  CPU policy evaluation needs a cheaper
        path for state/tactile history updates between camera frames, so this
        method reads proprioception directly and deliberately omits cameras.
        """

        base_obs = {"agent": self._get_obs_agent()}
        manipulators = self._build_manipulator_observations(base_obs)
        return {
            "manipulators": {
                role.value: observation.to_dict()
                for role, observation in manipulators.items()
            }
        }

    def _extract_observation(self, observation: Observation) -> dict[str, Any]:
        obs_mode = getattr(self, "obs_mode", self.TASK_SETTINGS.obs_mode)
        use_pointcloud = "pointcloud" in str(obs_mode).lower()
        return observation.to_dict(use_pointcloud=use_pointcloud)

    def _latch_task_success(self, success_now: Any) -> Any:
        """Latch task completion for both SR and the final StageScore predicate."""

        current = self._as_per_env_tensor(success_now, dtype=torch.bool)
        if not hasattr(self, "_task_success"):
            self._task_success = torch.zeros_like(current)
        self._task_success |= current
        return self._task_success.clone()

    def compute_stage_score(
        self,
        evaluation_result: Mapping[str, Any],
    ) -> Any:
        if "stage_score" in evaluation_result:
            explicit_score = self._as_per_env_tensor(
                evaluation_result["stage_score"], dtype=torch.float32
            ).clamp(0.0, 1.0)
            return self._latch_explicit_stage_score(explicit_score)
        evaluation = self.TASK_SETTINGS.evaluation
        stages = evaluation.stages
        if not stages:
            success = evaluation_result.get("success", False)
            return self._as_per_env_tensor(success, dtype=torch.float32)

        for stage_index, stage in enumerate(stages):
            if stage.name not in evaluation_result:
                raise KeyError(
                    f"Task {self.TASK_SETTINGS.task_id!r} evaluate() result is "
                    f"missing configured stage predicate {stage.name!r}."
                )
            reached = self._as_per_env_tensor(
                evaluation_result[stage.name], dtype=torch.bool
            )
            previous_stages_achieved = (
                torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                if stage_index == 0
                else self._achieved_stages[:, :stage_index].all(dim=1)
            )
            independent = evaluation.independent_stages
            if independent and stage.score == 1.0:
                success = self._as_per_env_tensor(
                    evaluation_result["success"], dtype=torch.bool
                )
                if not torch.equal(reached, success):
                    raise ValueError(
                        f"Task {self.TASK_SETTINGS.task_id!r} final stage "
                        "must equal success."
                    )
            reached_in_order = (
                reached if independent else reached & previous_stages_achieved
            )
            persistence = getattr(
                stage,
                "persistence",
                StagePersistence.PERMANENT,
            )
            if persistence is StagePersistence.UNTIL_NEXT:
                next_stage_achieved = self._achieved_stages[:, stage_index + 1]
                self._achieved_stages[:, stage_index] = (
                    next_stage_achieved | reached_in_order
                )
            elif independent and stage.score == 1.0:
                # Task success is already latched by the task implementation.
                self._achieved_stages[:, stage_index] = reached
            else:
                self._achieved_stages[:, stage_index] |= reached_in_order

        # A provisional stage becomes permanent as soon as its following stage
        # is reached. This second pass also commits two stages reached in one
        # evaluate() call.
        for stage_index, stage in enumerate(stages[:-1]):
            if (
                getattr(stage, "persistence", StagePersistence.PERMANENT)
                is StagePersistence.UNTIL_NEXT
            ):
                self._achieved_stages[:, stage_index] |= self._achieved_stages[
                    :, stage_index + 1
                ]

        stage_scores = torch.as_tensor(
            [stage.score for stage in stages],
            dtype=torch.float32,
            device=self.device,
        )
        # [B, S] x [S] -> [B]: retain the highest achieved stage.
        return torch.where(
            self._achieved_stages,
            stage_scores.unsqueeze(0),
            torch.zeros_like(stage_scores).unsqueeze(0),
        ).amax(dim=1)

    def _latch_explicit_stage_score(self, explicit_score: Any) -> Any:
        stages = self.TASK_SETTINGS.evaluation.stages
        if not stages:
            return explicit_score
        for stage_index, stage in enumerate(stages):
            self._achieved_stages[:, stage_index] |= explicit_score >= stage.score
        stage_scores = torch.as_tensor(
            [stage.score for stage in stages],
            dtype=torch.float32,
            device=self.device,
        )
        return torch.where(
            self._achieved_stages,
            stage_scores.unsqueeze(0),
            torch.zeros_like(stage_scores).unsqueeze(0),
        ).amax(dim=1)

    def _build_camera_observations(
        self,
        base_obs: Mapping[str, Any],
    ) -> dict[CameraRole, CameraObservation]:
        sensor_obs = base_obs.get("sensor_data", {})
        if not isinstance(sensor_obs, Mapping):
            raise TypeError(
                "ManiSkill sensor_data must be a mapping, got "
                f"{type(sensor_obs).__name__}."
            )
        return {
            camera_role: CameraObservation(
                rgb=self._lookup_obs(
                    sensor_obs,
                    self.camera_sensor_uid(camera_role),
                    "rgb",
                ),
                depth=self._lookup_obs(
                    sensor_obs,
                    self.camera_sensor_uid(camera_role),
                    "depth",
                ),
            )
            for camera_role in self.embodiment_spec.camera_roles
        }

    def _agent_observation_for_role(
        self,
        base_obs: Mapping[str, Any],
        role: ManipulatorRole,
    ) -> Mapping[str, Any]:
        agent_obs = base_obs.get("agent", base_obs.get("proprioception", {}))
        if not isinstance(agent_obs, Mapping):
            raise TypeError(
                "ManiSkill agent observation must be a mapping, got "
                f"{type(agent_obs).__name__}."
            )
        binding = self.embodiment_spec.binding_for_role(role)
        if len(self.embodiment_spec.robot_uids) == 1:
            physical_agent_obs = agent_obs
        else:
            physical_agent_obs = agent_obs.get(self._multi_agent_key(role))
        if not isinstance(physical_agent_obs, Mapping):
            raise KeyError(
                "ManiSkill MultiAgent observation is missing key "
                f"{self._multi_agent_key(role)!r} for role {role.value!r}."
            )
        if binding.manipulator_key is None:
            return physical_agent_obs
        manipulator_obs = physical_agent_obs.get("manipulators")
        if not isinstance(manipulator_obs, Mapping):
            raise KeyError("Shared-agent observation is missing 'manipulators'.")
        role_obs = manipulator_obs.get(binding.manipulator_key)
        if not isinstance(role_obs, Mapping):
            raise KeyError(
                "Shared-agent observation is missing key "
                f"{binding.manipulator_key!r}."
            )
        return role_obs

    def _reshape_fingertip_poses(
        self,
        value: Any,
        role: ManipulatorRole,
    ) -> Any:
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        if shape is None or len(shape) == 0:
            raise ValueError(
                f"{role.value} fingertip poses must be an array or tensor."
            )
        fingertip_count = len(
            getattr(self.agent_for_role(role), "hand_tip_link_names", ())
        )
        expected_flat_dim = fingertip_count * 7
        if fingertip_count <= 0 or shape[-1] != expected_flat_dim:
            raise ValueError(
                f"{role.value} fingertip poses must end in "
                f"{expected_flat_dim} values ({fingertip_count} poses), got "
                f"shape {tuple(shape)}."
            )
        # [..., F*7] -> [..., F, 7], where F is the fingertip count.
        return value.reshape(*shape[:-1], fingertip_count, 7)

    def _build_manipulator_observations(
        self,
        base_obs: Mapping[str, Any],
    ) -> dict[ManipulatorRole, ManipulatorObservation]:
        observations: dict[ManipulatorRole, ManipulatorObservation] = {}
        for binding in self.embodiment_spec.manipulators:
            role = binding.role
            agent_obs = self._agent_observation_for_role(base_obs, role)
            observations[role] = ManipulatorObservation(
                joint_position=agent_obs.get("qpos"),
                joint_velocity=agent_obs.get("qvel"),
                end_effector_pose=self._get_eef_pose(role),
                palm_pose=agent_obs.get("palm_pose"),
                fingertip_poses=self._reshape_fingertip_poses(
                    agent_obs.get("tip_poses"),
                    role,
                ),
                taxel_normal_force=self._get_taxel_normal_force(role),
            )
        return observations

    def _lookup_obs(self, obs: Mapping[str, Any], *keys: str) -> Any:
        value: Any = obs
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                return None
            value = value[key]
        return value

    def _get_eef_pose(self, role: ManipulatorRole) -> Any:
        tcp = getattr(self.agent_for_role(role), "tcp", None)
        return getattr(tcp, "pose", None)

    def _get_taxel_normal_force(self, role: ManipulatorRole) -> Any:
        agent = self.agent_for_role(role)
        if not getattr(self, "_policy_tactile_observation_enabled", True):
            taxel_count = len(getattr(agent, "taxel_link_names", ()))
            return torch.zeros(
                (self.num_envs, taxel_count),
                dtype=torch.float32,
                device=self.device,
            )
        tactile_reader = getattr(
            agent,
            "get_net_taxel_normal_forces",
            None,
        )
        if not callable(tactile_reader):
            raise RuntimeError(
                f"Robot {type(agent).__name__} for role {role.value!r} must "
                "implement "
                "get_net_taxel_normal_forces() to provide model tactile input."
            )
        normal_forces = tactile_reader()
        taxel_count = len(getattr(agent, "taxel_link_names", ()))
        expected_num_envs = int(
            getattr(
                getattr(self, "scene", None),
                "num_envs",
                normal_forces.shape[0],
            )
        )
        expected_shape = (expected_num_envs, taxel_count)
        if tuple(normal_forces.shape) != expected_shape:
            raise ValueError(
                "get_net_taxel_normal_forces() must return [B, N] with "
                f"shape {expected_shape}, got {tuple(normal_forces.shape)}."
            )
        if not normal_forces.dtype.is_floating_point:
            raise TypeError(
                "taxel_normal_force must use a floating dtype, got "
                f"{normal_forces.dtype}."
            )
        expected_device = torch.device(
            getattr(
                getattr(self, "scene", None),
                "device",
                normal_forces.device,
            )
        )
        if expected_device.type == "cuda" and expected_device.index is None:
            # torch.device("cuda") denotes the current CUDA device, while
            # tensors report its concrete index (for example cuda:0).
            expected_device = normal_forces.device
        if normal_forces.device != expected_device:
            raise ValueError(
                f"taxel_normal_force must be on {expected_device}, got "
                f"{normal_forces.device}."
            )
        if not bool(torch.all(torch.isfinite(normal_forces))):
            raise ValueError("taxel_normal_force contains NaN or infinity.")
        if bool(torch.any(normal_forces < 0.0)):
            raise ValueError("taxel_normal_force must be non-negative.")
        return normal_forces
