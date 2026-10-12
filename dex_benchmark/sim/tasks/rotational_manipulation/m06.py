from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from dex_benchmark.sim.sensors.cameras import HumanViewerCameraPoseSpec
from dex_benchmark.sim.envs.base_env import DexVLABenchBaseEnv
from dex_benchmark.sim.generalization import (
    AxisRandomizationSpec,
    GeneralizationAxis,
    GeneralizationParameterSpec,
    TaskGeneralizationSpec,
)
from dex_benchmark.sim.layout_types import LayoutId
from dex_benchmark.sim.scenes.layouts import LayoutBuilder
from dex_benchmark.sim.tasks.registry import TaskRegistry

try:
    import sapien
    import torch
    from mani_skill.utils.registration import register_env
    from mani_skill.utils.structs import Articulation
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - depends on the optional simulator stack.
    sapien = None
    torch = None
    Articulation = None
    Pose = None

    def register_env(*args: Any, **kwargs: Any):
        def decorator(cls):
            return cls

        return decorator


M06_SETTINGS = TaskRegistry.register(
    TaskRegistry.load("m06_v1")
)


@dataclass(frozen=True)
class TurnOnFaucetSuccessSpec:
    small_stream_open_fraction: float = 0.20
    midpoint_open_fraction: float = 0.50
    large_stream_open_fraction: float = 0.50
    target_open_fraction: float = 0.80
    contact_force_n: float = 0.40


M06_SUCCESS_SPEC = TurnOnFaucetSuccessSpec()

FAUCET_TARGET_JOINT_NAME = "handle_temp_joint"
FAUCET_FIXED_HANDLE_JOINT_NAME = "handle_joint"
FAUCET_SPOUT_JOINT_NAME = "spout_joint"
FAUCET_SPOUT_INITIAL_QPOS_RAD_BY_LAYOUT: dict[LayoutId, float] = {
    LayoutId.KITCHEN_L1: 0.32,
    LayoutId.KITCHEN_L2: 0.25,
    LayoutId.KITCHEN_L3: -0.36,
}

# The original RoboCasa asset gives handle_temp_joint damping=50, which makes the
# visible handle effectively immovable under normal hand contact. M06 overrides the
# passive drive while retaining the authored +/- temperature-rotation limits.
FAUCET_TARGET_JOINT_FRICTION = 0.15
FAUCET_TARGET_JOINT_DAMPING = 1.50
FAUCET_TARGET_JOINT_FORCE_LIMIT = 1e10

# handle_joint is an orthogonal serial joint in the RoboCasa faucet. It is not part
# of this task, so lock it at zero through both a zero-width limit and a position
# drive. Keeping the DOF in the articulation avoids modifying the shared asset.
FAUCET_FIXED_JOINT_STIFFNESS = 1000.0
FAUCET_FIXED_JOINT_DAMPING = 100.0
FAUCET_FIXED_JOINT_FORCE_LIMIT = 1000.0

# SAPIEN cylinders are authored along local X. This +90-degree world-Y rotation
# maps local +X to world -Z, so the visual stream always points downward even
# while the spout rotates around its own joint.
FAUCET_WATER_STREAM_WORLD_QUATERNION = (
    math.sqrt(0.5),
    0.0,
    math.sqrt(0.5),
    0.0,
)
FAUCET_WATER_STREAM_HALF_LENGTH_M = 0.11
FAUCET_WATER_STREAM_HIDDEN_Z_M = -20.0
# (radius_m, alpha), ordered as small then large water flow.
FAUCET_WATER_STREAM_LEVEL_SPECS: tuple[tuple[float, float], ...] = (
    (0.003, 0.35),
    (0.006, 0.55),
)

M06_SPATIAL_PARAMETER_NAMES = (
    "handle_initial_open_fraction",
    "robot_forward_offset_m",
    "robot_side_offset_m",
    "robot_yaw_offset_rad",
)
M06_PHYSICAL_PARAMETER_NAMES = (
    "handle_temp_joint_friction",
    "handle_temp_joint_damping",
)
M06_GENERALIZATION_SPEC = TaskGeneralizationSpec(
    spatial=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "handle_initial_open_fraction",
                0.00,
                0.20,
            ),
            GeneralizationParameterSpec(
                "robot_forward_offset_m",
                -0.02,
                0.02,
            ),
            GeneralizationParameterSpec(
                "robot_side_offset_m",
                -0.02,
                0.02,
            ),
            GeneralizationParameterSpec(
                "robot_yaw_offset_rad",
                -math.radians(2.0),
                math.radians(2.0),
            ),
        ),
        requires_rebuild=False,
    ),
    physical=AxisRandomizationSpec(
        parameters=(
            GeneralizationParameterSpec(
                "handle_temp_joint_friction",
                0.13,
                0.30,
            ),
            GeneralizationParameterSpec(
                "handle_temp_joint_damping",
                1.41,
                1.55,
            ),
        ),
        requires_rebuild=True,
    ),
)

M06_INITIAL_MAX_HANDLE_SPEED_RAD_S = 0.02
M06_INITIAL_MAX_ROOT_POSITION_ERROR_M = 1e-4
M06_INITIAL_MAX_ROOT_ROTATION_ERROR_RAD = math.radians(0.1)


@register_env(
    M06_SETTINGS.env_id,
    max_episode_steps=M06_SETTINGS.max_steps,
)
class M06Env(DexVLABenchBaseEnv):
    """Rotate a faucet handle to the opened position."""

    TASK_SETTINGS = M06_SETTINGS
    SUCCESS_SPEC = M06_SUCCESS_SPEC
    GENERALIZATION_SPEC = M06_GENERALIZATION_SPEC
    SUPPORTED_ROBOTS = [TASK_SETTINGS.maniskill_robot_uids]
    INTERACTION_SITE_ID = "sink"
    HUMAN_VIEWER_CAMERA_POSES: dict[str, HumanViewerCameraPoseSpec] = {
        LayoutId.KITCHEN_L1.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(2.865329, -2.237481, 1.125760),
            quaternion_world_wxyz=(0.955616, 0.041832, 0.167401, -0.238800),
        ),
        LayoutId.KITCHEN_L2.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(0.285256, -1.818040, 1.227597),
            quaternion_world_wxyz=(0.477551, 0.216024, 0.122406, -0.842791),
        ),
        LayoutId.KITCHEN_L3.value: HumanViewerCameraPoseSpec(
            position_world_xyz=(0.648519, -2.400344, 1.290693),
            quaternion_world_wxyz=(0.141206, -0.262346, 0.038839, 0.953796),
        ),
    }

    def __init__(
        self,
        *args: Any,
        robot_uids: str | tuple[str, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _robot_base_pose(self, env_idx: Any | None = None) -> Any:
        nominal_pose = self.interaction_site_robot_base_sapien_pose()
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return nominal_pose

        values = self._generalization_values(
            M06_SPATIAL_PARAMETER_NAMES,
            cache_name="_turn_on_faucet_spatial_values",
        )
        if env_idx is None:
            value_indices = np.arange(len(values["robot_forward_offset_m"]))
        else:
            value_indices = env_idx.detach().cpu().numpy()

        forward_offset = values["robot_forward_offset_m"][value_indices]
        side_offset = values["robot_side_offset_m"][value_indices]
        yaw_offset = values["robot_yaw_offset_rad"][value_indices]
        local_position = np.stack(
            [forward_offset, side_offset, np.zeros_like(forward_offset)],
            axis=1,
        )
        half_yaw = 0.5 * yaw_offset
        local_quaternion = np.stack(
            [
                np.cos(half_yaw),
                np.zeros_like(half_yaw),
                np.zeros_like(half_yaw),
                np.sin(half_yaw),
            ],
            axis=1,
        )
        return Pose.create(nominal_pose) * Pose.create_from_pq(
            p=local_position,
            q=local_quaternion,
        )

    def _require_robocasa_sink_articulation(self) -> Any:
        external_builders = self.layout_scene.external_scene_builders

        if not external_builders:
            raise RuntimeError(
                f"Layout {self.current_layout_id_value!r} has no external "
                "RoboCasa scene builder."
            )

        if len(external_builders) != 1:
            raise RuntimeError(
                "TurnOnFaucet currently expects exactly one RoboCasa "
                f"external builder, got {list(external_builders)}."
            )

        interaction_site = self.require_current_interaction_site()
        target_entity_name = interaction_site.target_entity_name
        if not target_entity_name:
            raise RuntimeError(
                "TurnOnFaucet requires its sink interaction site to name an "
                "external RoboCasa fixture."
            )

        scene_builder = next(iter(external_builders.values()))
        scene_data = getattr(scene_builder, "scene_data", None) or ()
        if not scene_data:
            raise RuntimeError(
                "RoboCasa scene builder contains no scene_data."
            )

        target_normalized = target_entity_name.casefold()
        sink_articulations: list[Any] = []
        sink_fixtures: list[Any] = []
        for scene_index, scene_entry in enumerate(scene_data):
            fixtures = (
                scene_entry.get("fixtures", {})
                if isinstance(scene_entry, dict)
                else {}
            )
            matches = [
                fixture
                for fixture_name, fixture in fixtures.items()
                if str(fixture_name).casefold() == target_normalized
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "TurnOnFaucet expected exactly one fixture named "
                    f"{target_entity_name!r} in RoboCasa scene "
                    f"{scene_index}, found {len(matches)}. Known fixtures: "
                    f"{sorted(map(str, fixtures))}."
                )

            sink_fixture = matches[0]
            sink_fixtures.append(sink_fixture)
            articulation = getattr(sink_fixture, "articulation", None)
            if articulation is None:
                raise RuntimeError(
                    f"RoboCasa sink fixture {target_entity_name!r} in scene "
                    f"{scene_index} is not articulated."
                )
            sink_articulations.append(articulation)

        if len(sink_articulations) != self.num_envs:
            raise RuntimeError(
                "TurnOnFaucet sink articulation count does not match the "
                f"environment count: articulations={len(sink_articulations)}, "
                f"num_envs={self.num_envs}."
            )
        water_outlet_local_positions = [
            self._water_outlet_local_position_from_fixture(sink_fixture)
            for sink_fixture in sink_fixtures
        ]
        # B rows [3] -> [B, 3], expressed in each spout link's local frame.
        self._spout_water_outlet_local_position = torch.as_tensor(
            np.stack(water_outlet_local_positions, axis=0),
            device=self.device,
            dtype=torch.float32,
        )
        if self.num_envs == 1:
            return sink_articulations[0]
        if Articulation is None:
            raise ImportError(
                "ManiSkill Articulation is required to merge batched "
                "RoboCasa sink fixtures."
            )

        expected_joint_names = tuple(
            sink_articulations[0].active_joints_map
        )
        for scene_index, articulation in enumerate(
            sink_articulations[1:],
            start=1,
        ):
            joint_names = tuple(articulation.active_joints_map)
            if joint_names != expected_joint_names:
                raise RuntimeError(
                    "TurnOnFaucet sink active joints differ across RoboCasa "
                    f"scenes: scene 0={expected_joint_names}, scene "
                    f"{scene_index}={joint_names}."
                )
        merged_articulation = Articulation.merge(
            sink_articulations,
            name="m06_sink",
            merge_links=True,
        )
        initial_pose_rows: list[torch.Tensor] = []
        for scene_index, articulation in enumerate(sink_articulations):
            initial_pose = articulation.initial_pose
            if initial_pose is None:
                raise RuntimeError(
                    "RoboCasa sink articulation has no authored initial pose "
                    f"in scene {scene_index}."
                )
            if initial_pose.raw_pose.shape != (1, 7):
                raise RuntimeError(
                    "Expected one authored sink pose per RoboCasa scene, got "
                    f"{tuple(initial_pose.raw_pose.shape)} in scene "
                    f"{scene_index}."
                )
            initial_pose_rows.append(initial_pose.raw_pose)
        # Articulation.merge does not preserve this build-time metadata.
        # B rows [1, 7] -> [B, 7].
        merged_articulation.initial_pose = Pose.create(
            torch.vstack(initial_pose_rows)
        )
        return merged_articulation

    @staticmethod
    def _water_outlet_local_position_from_fixture(
        sink_fixture: Any,
    ) -> np.ndarray:
        """Read the authored water-site top endpoint in spout-link coordinates."""

        loader = getattr(sink_fixture, "loader", None)
        xml_root = getattr(loader, "xml", None)
        if xml_root is None:
            raise RuntimeError(
                "M06 requires the RoboCasa sink fixture loader to expose its "
                "parsed MJCF XML for water-outlet calibration."
            )

        spout_body = xml_root.find(".//body[@name='spout']")
        if spout_body is None:
            raise RuntimeError(
                "M06 requires the RoboCasa sink MJCF to contain a body named "
                "'spout'."
            )
        water_sites = [
            site
            for site in spout_body.findall("site")
            if str(site.attrib.get("name", "")).casefold().endswith("water")
        ]
        if len(water_sites) != 1:
            raise RuntimeError(
                "M06 expected exactly one water site directly under the "
                f"RoboCasa spout body, found {len(water_sites)}."
            )

        water_site = water_sites[0]
        if water_site.attrib.get("type", "sphere") != "cylinder":
            raise RuntimeError(
                "M06 requires the RoboCasa spout water site to be a cylinder, "
                f"got {water_site.attrib.get('type', 'sphere')!r}."
            )
        unsupported_orientation_keys = {
            "axisangle",
            "euler",
            "fromto",
            "quat",
            "xyaxes",
            "zaxis",
        }.intersection(water_site.attrib)
        if unsupported_orientation_keys:
            raise RuntimeError(
                "M06 water-outlet calibration currently requires an unrotated "
                "MJCF cylinder site; found orientation attributes "
                f"{sorted(unsupported_orientation_keys)}."
            )

        site_position = np.fromstring(
            water_site.attrib.get("pos", "0 0 0"),
            sep=" ",
            dtype=np.float32,
        )
        site_size = np.fromstring(
            water_site.attrib.get("size", ""),
            sep=" ",
            dtype=np.float32,
        )
        if site_position.shape != (3,) or site_size.shape != (2,):
            raise RuntimeError(
                "M06 water site must provide pos='x y z' and cylinder "
                "size='radius half_length'; got "
                f"pos={water_site.attrib.get('pos')!r}, "
                f"size={water_site.attrib.get('size')!r}."
            )

        loader_scale = np.asarray(
            getattr(loader, "scale", 1.0),
            dtype=np.float32,
        )
        if loader_scale.ndim == 0:
            loader_scale = np.full(3, float(loader_scale), dtype=np.float32)
        if loader_scale.shape != (3,) or not np.all(
            np.isfinite(loader_scale) & (loader_scale > 0.0)
        ):
            raise RuntimeError(
                "M06 requires a finite positive scalar or XYZ RoboCasa loader "
                f"scale, got {loader_scale}."
            )

        # MJCF cylinder sites are Z-aligned. Their +Z endpoint is authored at
        # the faucet mouth, while the remaining cylinder extends into the basin.
        outlet_position = site_position.copy()
        outlet_position[2] += site_size[1]
        return outlet_position * loader_scale

    def _load_agent(self, options: dict[str, Any]) -> None:
        super()._load_agent(options, self._robot_base_pose())

    def _load_scene(self, options: dict[str, Any]) -> None:
        self.layout_scene = LayoutBuilder(
            env=self,
            layout_spec=self.current_layout_spec,
        ).build()

        self.faucet = self._require_robocasa_sink_articulation()

        self._print_available_faucet_joints()
        self.target_handle_joint = self._require_faucet_joint(
            FAUCET_TARGET_JOINT_NAME,
            role="target handle",
        )
        self.fixed_handle_joint = self._require_faucet_joint(
            FAUCET_FIXED_HANDLE_JOINT_NAME,
            role="fixed handle",
        )
        self.spout_joint = self._require_faucet_joint(
            FAUCET_SPOUT_JOINT_NAME,
            role="spout",
        )
        self.spout_link = self.spout_joint.child_link
        self.handle_link = self._require_handle_link()
        self._prepare_handle_tactile_query()

        active_joint_names = tuple(self.faucet.active_joints_map)
        self._target_handle_joint_qpos_index = active_joint_names.index(
            self.target_handle_joint.name
        )
        self._fixed_handle_joint_qpos_index = active_joint_names.index(
            self.fixed_handle_joint.name
        )
        self._spout_joint_qpos_index = active_joint_names.index(
            self.spout_joint.name
        )

        print(
            "Configured M06 faucet joints:",
            f"target={self.target_handle_joint.name!r}",
            f"(qpos_index={self._target_handle_joint_qpos_index}),",
            f"fixed={self.fixed_handle_joint.name!r}",
            f"(qpos_index={self._fixed_handle_joint_qpos_index}),",
            f"spout={self.spout_joint.name!r}",
            f"(qpos_index={self._spout_joint_qpos_index}),",
            f"link={self.handle_link.name!r},",
        )

        self._configure_faucet_joint_physics()

        self.sink_fixture = self.faucet

        # D3 covers the complete robot against the complete articulated sink/faucet,
        # not only a hand-picked subset of arm/palm links. This includes the handle,
        # spout, basin/housing links and any other collidable links exposed by the
        # RoboCasa sink articulation.
        self._protected_force_links = self._unique_links(
            tuple(self.faucet.links)
            + self._entity_contact_links(self.sink_fixture)
        )
        self._protected_force_query = self.agent.prepare_pairwise_contact_query(
            tuple(self.agent.robot.links),
            self._protected_force_links,
        )

        self._nominal_faucet_root_position = (
            self.faucet.initial_pose.p.to(device=self.device).clone()
        )
        self._nominal_faucet_root_quaternion = (
            self.faucet.initial_pose.q.to(device=self.device).clone()
        )

        self.water_streams = self._build_water_stream_actors()

    def _prepare_handle_tactile_query(self) -> None:
        """Bind the real faucet handle to the active hand's taxel array."""

        self._handle_tactile_groups: tuple[
            tuple[str, tuple[Any, ...]], ...
        ] = (("faucet_handle", (self.handle_link,)),)
        if self.gpu_sim_enabled:
            return
        self.agent.prepare_cpu_object_taxel_query(
            self._handle_tactile_groups
        )

    def _configure_faucet_joint_physics(self) -> None:
        """Lock the unused handle axis and make the target axis hand-operable."""

        zero_limits = torch.zeros(
            (self.num_envs, 2),
            device=self.device,
            dtype=torch.float32,
        )
        zero_targets = torch.zeros(
            self.num_envs,
            device=self.device,
            dtype=torch.float32,
        )
        self.fixed_handle_joint.set_limits(zero_limits)
        self.fixed_handle_joint.set_friction(0.0)
        self.fixed_handle_joint.set_drive_properties(
            stiffness=FAUCET_FIXED_JOINT_STIFFNESS,
            damping=FAUCET_FIXED_JOINT_DAMPING,
            force_limit=FAUCET_FIXED_JOINT_FORCE_LIMIT,
        )
        self.fixed_handle_joint.set_drive_target(zero_targets)

        self.target_handle_joint.set_friction(FAUCET_TARGET_JOINT_FRICTION)
        self.target_handle_joint.set_drive_properties(
            stiffness=0.0,
            damping=FAUCET_TARGET_JOINT_DAMPING,
            force_limit=FAUCET_TARGET_JOINT_FORCE_LIMIT,
        )

        if self.generalization_axis is not GeneralizationAxis.PHYSICAL:
            return

        values = self._generalization_values(
            M06_PHYSICAL_PARAMETER_NAMES,
            cache_name="_turn_on_faucet_physical_values",
        )
        friction_values = values["handle_temp_joint_friction"]
        damping_values = values["handle_temp_joint_damping"]

        # ArticulationJoint is a batched wrapper. Each _objs element owns the
        # concrete SAPIEN joint for one sub-environment.
        joint_objects = tuple(getattr(self.target_handle_joint, "_objs", ()))
        if not joint_objects:
            raise RuntimeError(
                f"Target handle joint {self.target_handle_joint.name!r} does "
                "not expose underlying SAPIEN joint objects through _objs."
            )
        if not (
            len(joint_objects)
            == len(friction_values)
            == len(damping_values)
        ):
            raise RuntimeError(
                "Physical generalization value count does not match the "
                "number of target handle joints: "
                f"joints={len(joint_objects)}, "
                f"friction_values={len(friction_values)}, "
                f"damping_values={len(damping_values)}."
            )

        for env_index, (joint_object, friction, damping) in enumerate(
            zip(
                joint_objects,
                friction_values,
                damping_values,
                strict=True,
            )
        ):
            friction_value = float(friction)
            damping_value = float(damping)
            joint_object.set_friction(friction=friction_value)
            joint_object.set_drive_properties(
                stiffness=0.0,
                damping=damping_value,
                force_limit=FAUCET_TARGET_JOINT_FORCE_LIMIT,
            )
            print(
                f"Applied M06 target-handle physical values for env "
                f"{env_index}: joint={self.target_handle_joint.name!r}, "
                f"friction={friction_value:.6f}, "
                f"damping={damping_value:.6f}"
            )

    def _print_available_faucet_joints(self) -> None:
        active_joints = self.faucet.active_joints_map
        if not active_joints:
            raise RuntimeError(
                "The selected RoboCasa sink articulation exposes "
                "no active joints."
            )
        print("\nAvailable faucet active joints:")
        for joint_name, joint in active_joints.items():
            child_link = getattr(joint, "child_link", None)
            print(
                f"  joint={joint_name!r}, "
                f"type={joint.type!r}, "
                f"child_link="
                f"{getattr(child_link, 'name', None)!r}, "
                f"limits={joint.limits}"
            )

    def _require_faucet_joint(self, joint_name: str, *, role: str) -> Any:
        joint = self.faucet.active_joints_map.get(joint_name)
        if joint is None:
            known_joint_names = ", ".join(
                sorted(self.faucet.active_joints_map)
            )
            raise RuntimeError(
                f"M06 requires {role} joint {joint_name!r}. Known active "
                f"joints: {known_joint_names}."
            )
        joint_type = (
            joint.type[0]
            if isinstance(joint.type, (tuple, list))
            else joint.type
        )
        if joint_type not in ("revolute", "revolute_unwrapped", "continuous"):
            raise RuntimeError(
                f"M06 {role} joint {joint.name!r} must be revolute or "
                f"continuous, "
                f"got {joint.type!r}."
            )
        if getattr(joint, "child_link", None) is None:
            raise RuntimeError(
                f"M06 {role} joint {joint.name!r} has no child link."
            )
        return joint

    def _require_handle_link(self) -> Any:
        """Return the visible and collidable handle, not an MJCF dummy link."""

        handle_link = self.faucet.links_map.get("handle")
        if handle_link is not None:
            return handle_link
        known_link_names = ", ".join(sorted(self.faucet.links_map))
        raise RuntimeError(
            "M06 requires the RoboCasa faucet's visible 'handle' link. "
            f"Known links: {known_link_names}."
        )

    def _build_water_stream_actors(self) -> tuple[Any, ...]:
        """Build fixed-material actors for small and large water-flow levels."""

        if sapien is None:
            raise ImportError("SAPIEN is required to build the faucet water effect.")
        streams: list[Any] = []
        for level_index, (radius_m, alpha) in enumerate(
            FAUCET_WATER_STREAM_LEVEL_SPECS,
            start=1,
        ):
            builder = self.scene.create_actor_builder()
            material = sapien.render.RenderMaterial(
                base_color=[0.15, 0.55, 1.00, float(alpha)],
            )
            builder.add_cylinder_visual(
                radius=radius_m,
                half_length=FAUCET_WATER_STREAM_HALF_LENGTH_M,
                pose=sapien.Pose(),
                material=material,
            )
            builder.initial_pose = sapien.Pose(
                p=[0.0, 0.0, FAUCET_WATER_STREAM_HIDDEN_Z_M]
            )
            streams.append(
                builder.build_kinematic(
                    name=f"turn_on_faucet_water_stream_{level_index}"
                )
            )
        return tuple(streams)

    def _set_robot_to_rest_keyframe(self, batch_size: int) -> None:
        keyframes = getattr(self.agent, "keyframes", None)
        rest_keyframe = keyframes.get("rest") if keyframes else None
        if rest_keyframe is None or getattr(rest_keyframe, "qpos", None) is None:
            raise RuntimeError(
                "TurnOnFaucet requires the configured robot to expose a "
                "'rest' keyframe."
            )
        robot_qpos = torch.as_tensor(
            rest_keyframe.qpos,
            device=self.device,
            dtype=torch.float32,
        )
        if robot_qpos.ndim == 1:
            robot_qpos = robot_qpos.unsqueeze(0).repeat(batch_size, 1)
        self.agent.robot.set_qpos(robot_qpos)
        self.agent.robot.set_qvel(torch.zeros_like(robot_qpos))
        self.agent.controller.reset()

    def _initial_handle_open_fraction(self, env_idx: Any) -> Any:
        if self.generalization_axis is not GeneralizationAxis.SPATIAL:
            return torch.zeros(
                len(env_idx),
                device=self.device,
                dtype=torch.float32,
            )
        values = self._generalization_values(
            M06_SPATIAL_PARAMETER_NAMES,
            cache_name="_turn_on_faucet_spatial_values",
        )
        value_indices = env_idx.detach().cpu().numpy()
        return torch.as_tensor(
            values["handle_initial_open_fraction"][value_indices],
            device=self.device,
            dtype=torch.float32,
        )

    def _reset_faucet_joints(self, env_idx: Any) -> None:
        target_limits = self.target_handle_joint.limits.reshape(
            self.num_envs, -1, 2
        )[env_idx, 0]
        lower_limit = target_limits[:, 0]
        upper_limit = target_limits[:, 1]
        if not torch.all(
            torch.isfinite(lower_limit)
            & torch.isfinite(upper_limit)
            & (lower_limit <= 0.0)
            & (upper_limit > 0.0)
        ):
            raise RuntimeError(
                "M06 target handle joint must have finite limits spanning "
                f"zero with a positive upper limit; got {target_limits}."
            )
        initial_fraction = self._initial_handle_open_fraction(env_idx)
        initial_qpos = initial_fraction * upper_limit

        layout_id = self.current_layout_id
        if layout_id not in FAUCET_SPOUT_INITIAL_QPOS_RAD_BY_LAYOUT:
            raise RuntimeError(
                "M06 has no spout initial qpos configured for layout "
                f"{self.current_layout_id_value!r}."
            )
        spout_initial_qpos = FAUCET_SPOUT_INITIAL_QPOS_RAD_BY_LAYOUT[layout_id]
        spout_limits = self.spout_joint.limits.reshape(
            self.num_envs, -1, 2
        )[env_idx, 0]
        spout_qpos = torch.full_like(
            spout_limits[:, 0],
            spout_initial_qpos,
        )
        finite_lower = torch.isfinite(spout_limits[:, 0])
        finite_upper = torch.isfinite(spout_limits[:, 1])
        spout_in_bounds = (
            (~finite_lower | (spout_qpos >= spout_limits[:, 0]))
            & (~finite_upper | (spout_qpos <= spout_limits[:, 1]))
        )
        if not torch.all(spout_in_bounds):
            raise RuntimeError(
                f"M06 spout initial qpos {spout_initial_qpos} "
                f"is outside joint limits {spout_limits}."
            )

        qpos = self.faucet.get_qpos()[env_idx].clone()
        qpos[:, self._target_handle_joint_qpos_index] = initial_qpos
        qpos[:, self._fixed_handle_joint_qpos_index] = 0.0
        qpos[:, self._spout_joint_qpos_index] = spout_qpos
        self.faucet.set_qpos(qpos)
        self.faucet.set_qvel(torch.zeros_like(qpos))

    def _reset_faucet_root_pose(self, env_idx: Any) -> None:
        """Restore the RoboCasa sink fixture to its nominal root pose.

        The faucet belongs to the complete RoboCasa Sink articulation, so spatial
        randomization must not move its root independently from the countertop.
        """

        current_position = self.faucet.root_pose.p.clone()
        current_quaternion = self.faucet.root_pose.q.clone()

        current_position[env_idx] = (
            self._nominal_faucet_root_position[env_idx]
        )
        current_quaternion[env_idx] = (
            self._nominal_faucet_root_quaternion[env_idx]
        )

        self.faucet.set_pose(
            Pose.create_from_pq(
                p=current_position,
                q=current_quaternion,
            )
        )

    def _initialize_episode(
        self,
        env_idx: Any,
        options: dict[str, Any],
    ) -> None:
        with torch.device(self.device):
            self._invalidate_contact_force_cache()
            batch_size = len(env_idx)

            self.layout_scene.initialize_episode(
                env_idx=env_idx,
                batched_episode_rng=self._batched_episode_rng,
                device=self.device,
            )
            self._reset_faucet_root_pose(env_idx)
            self._reset_faucet_joints(env_idx)

            self.agent.robot.set_pose(sapien.Pose(p=[0.0, 0.0, -20.0]))
            self.layout_scene.settle(env_idx=env_idx)

            # Settle is allowed to move passive articulations. Re-apply the authored
            # sink root pose and this episode's sampled handle state afterwards so
            # every episode starts from the requested generalization value with zero
            # velocity instead of inheriting settle drift.
            self._reset_faucet_root_pose(env_idx)
            self._reset_faucet_joints(env_idx)

            self.agent.robot.set_pose(self._robot_base_pose(env_idx))
            self._set_robot_to_rest_keyframe(batch_size)

            self._ensure_episode_state()
            self._handle_was_contacted[env_idx] = False
            self._midpoint_was_reached[env_idx] = False
            self._faucet_was_opened[env_idx] = False
            self._water_level[env_idx] = 0

            self._initial_faucet_root_position[env_idx] = (
                self.faucet.root_pose.p[env_idx]
            )
            self._initial_sink_position[env_idx] = (
                self._entity_pose(self.sink_fixture).p[env_idx]
            )
            self._update_water_stream_visual()

    def _ensure_episode_state(self) -> None:
        if hasattr(self, "_handle_was_contacted"):
            return
        self._handle_was_contacted = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._midpoint_was_reached = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._faucet_was_opened = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._water_level = torch.zeros(
            self.num_envs,
            dtype=torch.int64,
            device=self.device,
        )
        self._initial_faucet_root_position = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )
        self._initial_sink_position = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
        )

    def _invalidate_contact_force_cache(self) -> None:
        self._cached_handle_tactile_force_norms: Any | None = None

    def _before_control_step(self) -> None:
        super()._before_control_step()
        self._invalidate_contact_force_cache()

    def _handle_open_fraction(self) -> Any:
        qpos = self.target_handle_joint.qpos.reshape(
            self.num_envs, -1
        )[:, 0]
        limits = self.target_handle_joint.limits.reshape(
            self.num_envs, -1, 2
        )[:, 0]
        upper_limit = limits[:, 1]
        return (qpos / upper_limit.clamp_min(1e-6)).clamp(0.0, 1.0)

    def _handle_opening_velocity(self) -> Any:
        return self.target_handle_joint.qvel.reshape(
            self.num_envs, -1
        )[:, 0]

    def _handle_tactile_force_norms(self) -> Any:
        if getattr(self, "_cached_handle_tactile_force_norms", None) is None:
            tactile_forces = self.agent.get_object_taxel_world_forces(
                self._handle_tactile_groups
            )["faucet_handle"]
            expected_prefix = (self.num_envs,)
            if (
                tactile_forces.ndim != 3
                or tuple(tactile_forces.shape[:1]) != expected_prefix
                or tactile_forces.shape[1] == 0
                or tactile_forces.shape[2] != 3
            ):
                raise RuntimeError(
                    "Faucet-handle tactile query must return non-empty "
                    "world forces shaped [B, N, 3]; got "
                    f"{tuple(tactile_forces.shape)} for B={self.num_envs}."
                )
            self._cached_handle_tactile_force_norms = torch.linalg.norm(
                tactile_forces,
                dim=-1,
            )
        force_norms = self._cached_handle_tactile_force_norms
        assert force_norms is not None
        return force_norms

    def _handle_has_tactile_contact(self) -> Any:
        return (
            self._handle_tactile_force_norms()
            >= self.SUCCESS_SPEC.contact_force_n
        ).any(dim=1)

    def _water_stream_open_pose(self) -> Any:
        outlet_local_pose = Pose.create_from_pq(
            p=self._spout_water_outlet_local_position,
        )
        outlet_world_position = (
            self.spout_link.pose * outlet_local_pose
        ).p
        stream_center_position = outlet_world_position.clone()
        stream_center_position[:, 2] -= FAUCET_WATER_STREAM_HALF_LENGTH_M
        stream_world_quaternion = torch.as_tensor(
            FAUCET_WATER_STREAM_WORLD_QUATERNION,
            device=self.device,
            dtype=torch.float32,
        ).unsqueeze(0).repeat(self.num_envs, 1)
        return Pose.create_from_pq(
            p=stream_center_position,
            q=stream_world_quaternion,
        )

    def _update_water_stream_visual(self) -> None:
        open_pose = self._water_stream_open_pose()
        for level_index, water_stream in enumerate(
            self.water_streams,
            start=1,
        ):
            position = open_pose.p.clone()
            position[self._water_level != level_index, 2] = (
                FAUCET_WATER_STREAM_HIDDEN_Z_M
            )
            water_stream.set_pose(
                Pose.create_from_pq(
                    p=position,
                    q=open_pose.q,
                )
            )

    def _water_level_from_open_fraction(self, open_fraction: Any) -> Any:
        """Map positive target-handle opening to none/small/large levels."""

        water_level = torch.zeros_like(open_fraction, dtype=torch.int64)
        water_level = torch.where(
            open_fraction >= self.SUCCESS_SPEC.small_stream_open_fraction,
            torch.ones_like(water_level),
            water_level,
        )
        return torch.where(
            open_fraction >= self.SUCCESS_SPEC.large_stream_open_fraction,
            torch.full_like(water_level, 2),
            water_level,
        )

    def _handle_is_static(self) -> Any:
        return (
            torch.abs(self._handle_opening_velocity())
            <= M06_INITIAL_MAX_HANDLE_SPEED_RAD_S
        )

    def _faucet_root_matches_nominal_pose(self) -> Any:
        position_error = torch.linalg.norm(
            self.faucet.root_pose.p - self._nominal_faucet_root_position,
            dim=1,
        )
        quaternion_dot = torch.abs(
            torch.sum(
                self.faucet.root_pose.q
                * self._nominal_faucet_root_quaternion,
                dim=1,
            )
        ).clamp(max=1.0)
        rotation_error = 2.0 * torch.acos(quaternion_dot)
        return (
            position_error <= M06_INITIAL_MAX_ROOT_POSITION_ERROR_M
        ) & (
            rotation_error <= M06_INITIAL_MAX_ROOT_ROTATION_ERROR_RAD
        )

    def _check_generalization_validity(
        self,
        axis: GeneralizationAxis | None,
    ) -> Any:
        if axis in (
            GeneralizationAxis.SPATIAL,
            GeneralizationAxis.PHYSICAL,
        ):
            # Spatial randomization changes only the handle's initial open fraction
            # and robot base. The sink/faucet root must remain exactly attached to its
            # authored RoboCasa fixture rather than being randomized independently.
            return (
                self._handle_open_fraction()
                < self.SUCCESS_SPEC.midpoint_open_fraction
            ) & self._handle_is_static() & self._faucet_root_matches_nominal_pose()
        return super()._check_generalization_validity(axis)

    def evaluate(self) -> dict[str, Any]:
        self._ensure_episode_state()
        handle_contacted = self._handle_has_tactile_contact()
        open_fraction = self._handle_open_fraction()
        midpoint_reached = (
            open_fraction >= self.SUCCESS_SPEC.midpoint_open_fraction
        )
        faucet_opened = (
            open_fraction >= self.SUCCESS_SPEC.target_open_fraction
        )

        self._handle_was_contacted |= handle_contacted
        self._midpoint_was_reached |= midpoint_reached
        self._faucet_was_opened |= faucet_opened
        self._water_level = self._water_level_from_open_fraction(open_fraction)
        self._update_water_stream_visual()

        return {
            "success": faucet_opened,
            "faucet_handle_contacted": handle_contacted,
            "faucet_handle_midpoint_reached": midpoint_reached,
            "faucet_opened": faucet_opened,
        }


    @staticmethod
    def _entity_pose(entity: Any) -> Any:
        if hasattr(entity, "root_pose"):
            return entity.root_pose
        if hasattr(entity, "pose"):
            return entity.pose
        raise TypeError(
            f"Protected entity {type(entity).__name__} exposes neither "
            "root_pose nor pose."
        )

    @staticmethod
    def _entity_contact_links(entity: Any) -> tuple[Any, ...]:
        links = getattr(entity, "links", None)
        if links is not None:
            return tuple(links)
        return (entity,)

    @staticmethod
    def _unique_links(links: tuple[Any, ...]) -> tuple[Any, ...]:
        """Deduplicate simulator link wrappers by identity while preserving order."""

        unique_links: list[Any] = []
        seen_ids: set[int] = set()
        for link in links:
            link_id = id(link)
            if link_id in seen_ids:
                continue
            seen_ids.add(link_id)
            unique_links.append(link)
        return tuple(unique_links)
