from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
from typing import Any
from xml.etree import ElementTree

import numpy as np

from dex_benchmark.sim.scenes.layouts.spec import (
    ArticulatedAssetSpec,
    ExternalSceneSpec,
    JointPositionLatchSpec,
    LayoutBuildItemSpec,
    LayoutSpec,
    MJCFActorSpec,
    PhysicalMaterialSpec,
    PoseSpec,
    ProceduralTableSpec,
    StaticBoxSpec,
    StaticMeshSpec,
    TexturedPlaneSpec,
    YCBActorSpec,
    quat_multiply_wxyz,
)

try:
    import sapien
    import torch
    from mani_skill import ASSET_DIR as MANISKILL_ASSET_DIR
    from mani_skill.utils.building import actors as mani_skill_actors
    from mani_skill.utils.building.ground import build_ground
    from mani_skill.utils.structs import Articulation
    from mani_skill.utils.structs.pose import Pose
except Exception:  # pragma: no cover - depends on optional simulator stack.
    sapien = None
    torch = None
    MANISKILL_ASSET_DIR = None
    mani_skill_actors = None
    build_ground = None
    Articulation = None
    Pose = None


@dataclass
class LayoutBuildResult:
    layout_spec: LayoutSpec
    actors: dict[str, Any] = field(default_factory=dict)
    articulations: dict[str, Any] = field(default_factory=dict)
    external_scene_builders: dict[str, Any] = field(default_factory=dict)
    ground: Any | None = None
    actor_values: dict[str, "ActorBuildValues"] = field(default_factory=dict)
    articulation_values: dict[str, "ArticulationBuildValues"] = field(
        default_factory=dict
    )
    _env: Any | None = field(default=None, repr=False)
    _gpu_materials_rebound: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def _set_body_material(body: Any, material: Any) -> None:
        shape_getter = getattr(body, "get_collision_shapes", None)
        shapes = (
            tuple(shape_getter())
            if callable(shape_getter)
            else tuple(getattr(body, "collision_shapes", ()))
        )
        for shape in shapes:
            shape.set_physical_material(material)

    def _rebind_gpu_physical_materials(self) -> None:
        """Upload layout materials again after SAPIEN has initialized GPU state.

        Some SAPIEN GPU builds expose the material assigned during scene build on
        each collision shape but leave the GPU solver using its stale default.
        Rebinding once from the authoritative layout specs makes CPU and GPU
        contact friction agree without changing any configured coefficient.
        """

        if (
            self._gpu_materials_rebound
            or self._env is None
            or not self._env.gpu_sim_enabled
        ):
            return

        for build_item in self.layout_spec.build_items:
            if build_item.name in self.actors:
                actor = self.actors[build_item.name]
                raw_bodies = tuple(getattr(actor, "_bodies", ()))
                values = self.actor_values.get(build_item.name)
                materials = (
                    values.physical_materials
                    if values is not None and values.physical_materials is not None
                    else None
                )
                for env_index, raw_body in enumerate(raw_bodies):
                    material_spec = (
                        materials[env_index]
                        if materials is not None
                        else getattr(build_item, "physical_material", None)
                    )
                    if material_spec is not None:
                        self._set_body_material(
                            raw_body,
                            physical_material_spec_to_sapien(material_spec),
                        )
                continue

            if build_item.name not in self.articulations:
                continue
            articulation = self.articulations[build_item.name]
            raw_articulations = tuple(getattr(articulation, "_objs", ()))
            values = self.articulation_values.get(build_item.name)
            materials = (
                values.physical_materials
                if values is not None and values.physical_materials is not None
                else None
            )
            for env_index, raw_articulation in enumerate(raw_articulations):
                material_spec = (
                    materials[env_index]
                    if materials is not None
                    else getattr(build_item, "physical_material", None)
                )
                if material_spec is None:
                    continue
                material = physical_material_spec_to_sapien(material_spec)
                links = tuple(getattr(raw_articulation, "links", ()))
                if not links:
                    link_getter = getattr(raw_articulation, "get_links", None)
                    links = tuple(link_getter()) if callable(link_getter) else ()
                for link in links:
                    self._set_body_material(link, material)

        self._gpu_materials_rebound = True

    def require_entity(self, entity_name: str) -> Any:
        """Return one layout-owned actor or articulation by its physical name."""

        in_actors = entity_name in self.actors
        in_articulations = entity_name in self.articulations
        if in_actors and in_articulations:
            raise ValueError(
                f"Layout {self.layout_spec.layout_id.value!r} exposes entity "
                f"{entity_name!r} as both an actor and an articulation. Entity "
                "names must be unique across both collections."
            )
        if in_actors:
            return self.actors[entity_name]
        if in_articulations:
            return self.articulations[entity_name]

        known_entities = ", ".join(
            sorted(set(self.actors) | set(self.articulations))
        ) or "<empty>"
        raise ValueError(
            f"Layout {self.layout_spec.layout_id.value!r} does not expose entity "
            f"{entity_name!r}. Known entities: {known_entities}."
        )

    def _joint_position_latch_bindings(
        self,
    ) -> tuple[tuple[Any, JointPositionLatchSpec], ...]:
        bindings: list[tuple[Any, JointPositionLatchSpec]] = []
        for build_item in self.layout_spec.build_items:
            if not isinstance(build_item, ArticulatedAssetSpec):
                continue
            if not build_item.joint_position_latches:
                continue
            articulation = self.articulations[build_item.name]
            joint_by_name = {
                str(joint.name): joint for joint in articulation.active_joints
            }
            for joint_name, latch_spec in (
                build_item.joint_position_latches.items()
            ):
                try:
                    joint = joint_by_name[joint_name]
                except KeyError as exc:
                    known_joint_names = ", ".join(sorted(joint_by_name))
                    raise RuntimeError(
                        f"Joint position latch for {build_item.name!r} references "
                        f"unknown joint {joint_name!r}. Known joints: "
                        f"{known_joint_names}."
                    ) from exc
                bindings.append((joint, latch_spec))
        return tuple(bindings)

    def reset_joint_position_latches(self, env_idx: Any) -> None:
        """Latch configured joints at their current reset positions."""

        if torch is None:
            raise ImportError(
                "PyTorch is required to reset layout joint position latches."
            )
        for joint, _ in self._joint_position_latch_bindings():
            # One active revolute joint: [B, 1] or [B] -> [B].
            current_qpos = joint.qpos.reshape(-1)
            joint.set_drive_target(current_qpos[env_idx])

    def update_joint_position_latches(self) -> None:
        """Advance finite-torque latch anchors after externally driven motion."""

        updated_any_target = False
        for joint, latch_spec in self._joint_position_latch_bindings():
            if not latch_spec.relatch:
                continue
            # One active revolute joint: [B, 1] or [B] -> [B].
            current_qpos = joint.qpos.reshape(-1)
            current_target = joint.drive_target.reshape(-1)
            if current_qpos.shape != current_target.shape:
                raise RuntimeError(
                    f"Joint position latch state shape mismatch for {joint.name!r}: "
                    f"qpos={tuple(current_qpos.shape)}, "
                    f"target={tuple(current_target.shape)}."
                )
            position_error = current_qpos - current_target
            clamped_error = position_error.clamp(
                min=-latch_spec.yield_angle_rad,
                max=latch_spec.yield_angle_rad,
            )
            # [B] - [B] -> [B]: the anchor follows only after drive saturation,
            # preserving a bounded elastic error that behaves like static friction.
            updated_target = current_qpos - clamped_error
            joint.set_drive_target(updated_target)
            updated_any_target = True

        if (
            updated_any_target
            and self._env is not None
            and self._env.gpu_sim_enabled
        ):
            self._env.scene.px.gpu_apply_articulation_target_position()

    def initialize_episode(
        self,
        env_idx: Any,
        batched_episode_rng: Any,
        device: Any,
    ) -> None:
        """Reset layout-owned movable objects to their canonical jittered poses."""
        if torch is None or Pose is None:
            raise ImportError(
                "ManiSkill and PyTorch are required to initialize layout objects."
            )

        batch_size = len(env_idx)
        if batch_size == 0:
            return
        if batched_episode_rng is None:
            raise RuntimeError(
                "Layout episode randomization requires ManiSkill's batched episode RNG."
            )
        episode_rng = batched_episode_rng[env_idx]

        # Scene construction precedes gpu_init(). Rebind once here, after GPU
        # initialization and before any settling/contact step.
        self._rebind_gpu_physical_materials()

        for build_item in self.layout_spec.build_items:
            position_jitter_xy = getattr(
                build_item,
                "episode_position_jitter_xy",
                None,
            )
            is_resettable_actor = isinstance(
                build_item,
                (StaticBoxSpec, StaticMeshSpec, YCBActorSpec),
            ) and (
                position_jitter_xy is not None
                or build_item.body_type == "dynamic"
            )
            is_resettable_articulation = isinstance(
                build_item,
                ArticulatedAssetSpec,
            ) and (
                position_jitter_xy is not None
                or not build_item.fix_root_link
                or build_item.initial_qpos is not None
                or bool(build_item.joint_initial_qpos)
            )
            if not is_resettable_actor and not is_resettable_articulation:
                continue
            if position_jitter_xy is None:
                position_jitter_xy = (0.0, 0.0)
            if position_jitter_xy[0] < 0.0 or position_jitter_xy[1] < 0.0:
                raise ValueError(
                    f"Episode XY jitter for layout item {build_item.name!r} must "
                    f"be non-negative, got {position_jitter_xy}."
                )

            # [3] -> [B, 3]: replicate the manually tuned canonical world position.
            position_xyz = torch.as_tensor(
                build_item.pose.position_xyz,
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(batch_size, 1)
            if position_jitter_xy != (0.0, 0.0):
                # [B, 2] * [2] -> [B, 2]: apply only bounded world-XY jitter.
                unit_xy_noise = torch.as_tensor(
                    episode_rng.uniform(-1.0, 1.0, size=(2,)),
                    device=device,
                    dtype=torch.float32,
                )
                jitter_limit_xy = torch.as_tensor(
                    position_jitter_xy,
                    device=device,
                    dtype=torch.float32,
                )
                position_xyz[:, :2] += unit_xy_noise * jitter_limit_xy

            if isinstance(
                build_item,
                (StaticBoxSpec, StaticMeshSpec, YCBActorSpec),
            ):
                spawn_probability = build_item.episode_spawn_probability
                if not 0.0 <= spawn_probability <= 1.0:
                    raise ValueError(
                        f"Episode spawn probability for layout item "
                        f"{build_item.name!r} must be in [0, 1], got "
                        f"{spawn_probability}."
                    )
                if spawn_probability < 1.0:
                    # [B]: inactive clutter is parked below the physical scene.
                    spawn_mask = torch.as_tensor(
                        episode_rng.uniform(0.0, 1.0),
                        device=device,
                        dtype=torch.float32,
                    ) < spawn_probability
                    position_xyz[~spawn_mask, 2] = -10.0

            # [4] -> [B, 4]: keep the manually tuned canonical orientation fixed.
            quaternion_wxyz = torch.as_tensor(
                build_item.pose.quaternion_wxyz,
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0).repeat(batch_size, 1)
            episode_pose = Pose.create_from_pq(
                p=position_xyz,
                q=quaternion_wxyz,
            )

            if isinstance(
                build_item,
                (StaticBoxSpec, StaticMeshSpec, YCBActorSpec),
            ):
                if build_item.body_type == "static":
                    raise ValueError(
                        f"Static layout item {build_item.name!r} cannot be reset "
                        "with episode_position_jitter_xy. Use kinematic or dynamic."
                    )
                actor = self.actors[build_item.name]
                actor.set_pose(episode_pose)
                if build_item.body_type == "dynamic":
                    zero_velocity = torch.zeros(
                        (batch_size, 3),
                        device=device,
                        dtype=torch.float32,
                    )
                    actor.set_linear_velocity(zero_velocity)
                    actor.set_angular_velocity(zero_velocity)
                continue

            if isinstance(build_item, ArticulatedAssetSpec):
                articulation = self.articulations[build_item.name]
                articulation.set_pose(episode_pose)
                if not build_item.fix_root_link:
                    zero_root_velocity = torch.zeros(
                        (batch_size, 3),
                        device=device,
                        dtype=torch.float32,
                    )
                    articulation.set_root_linear_velocity(zero_root_velocity)
                    articulation.set_root_angular_velocity(zero_root_velocity)
                if build_item.initial_qpos is not None:
                    # [D] -> [B, D]: restore articulated target joint state.
                    initial_qpos = torch.as_tensor(
                        build_item.initial_qpos,
                        device=device,
                        dtype=torch.float32,
                    ).unsqueeze(0).repeat(batch_size, 1)
                    articulation.set_qpos(initial_qpos)
                    articulation.set_qvel(torch.zeros_like(initial_qpos))
                elif build_item.joint_initial_qpos:
                    current_qpos = articulation.get_qpos()[env_idx].clone()
                    joint_index_by_name = {
                        joint.name: joint_index
                        for joint_index, joint in enumerate(
                            articulation.active_joints
                        )
                    }
                    for joint_name, joint_qpos in (
                        build_item.joint_initial_qpos.items()
                    ):
                        try:
                            joint_index = joint_index_by_name[joint_name]
                        except KeyError as exc:
                            known_joint_names = ", ".join(
                                sorted(joint_index_by_name)
                            )
                            raise ValueError(
                                f"Layout item {build_item.name!r} configures "
                                f"unknown joint {joint_name!r}. Known joints: "
                                f"{known_joint_names}."
                            ) from exc
                        current_qpos[:, joint_index] = float(joint_qpos)
                    articulation.set_qpos(current_qpos)
                    articulation.set_qvel(torch.zeros_like(current_qpos))
                continue

            raise TypeError(
                f"Layout item {build_item.name!r} defines episode XY jitter, but "
                f"{type(build_item).__name__} does not support episode resets."
            )
        self.reset_joint_position_latches(env_idx)

    def settle(self, env_idx: Any) -> None:
        """Advance physics before the first observation so dynamic layout items settle."""
        settle_sim_steps = self.layout_spec.settle_sim_steps
        if settle_sim_steps < 0:
            raise ValueError(
                f"Layout settle_sim_steps must be non-negative, got {settle_sim_steps}."
            )
        if settle_sim_steps == 0:
            return
        if self._env is None:
            raise RuntimeError("Layout settling requires the environment build context.")
        if len(env_idx) != self._env.num_envs:
            raise RuntimeError(
                "Pre-observation layout settling currently requires a full vector "
                "environment reset because PhysX scene.step() advances every sub-scene."
            )

        scene = self._env.scene
        if self._env.gpu_sim_enabled:
            scene._gpu_apply_all()
        for _ in range(settle_sim_steps):
            scene.step()
        if self._env.gpu_sim_enabled:
            scene._gpu_fetch_all()

        # Settling is an initialization operation. Do not leak residual rigid
        # body or articulation velocity into the first policy/replay step.
        zero_velocity = torch.zeros(
            (self._env.num_envs, 3),
            device=self._env.device,
            dtype=torch.float32,
        )
        for build_item in self.layout_spec.build_items:
            actor = self.actors.get(build_item.name)
            if (
                actor is not None
                and getattr(build_item, "body_type", None) == "dynamic"
            ):
                actor.set_linear_velocity(zero_velocity)
                actor.set_angular_velocity(zero_velocity)
            articulation = self.articulations.get(build_item.name)
            if articulation is not None and not getattr(
                build_item, "fix_root_link", True
            ):
                articulation.set_root_linear_velocity(zero_velocity)
                articulation.set_root_angular_velocity(zero_velocity)
                articulation.set_qvel(torch.zeros_like(articulation.get_qvel()))
        if self._env.gpu_sim_enabled:
            scene._gpu_apply_all()
            scene._gpu_fetch_all()


def pose_spec_to_sapien_pose(pose_spec: PoseSpec) -> Any:
    if sapien is None:
        raise ImportError("SAPIEN is required to convert Layout PoseSpec values.")
    return sapien.Pose(
        p=list(pose_spec.position_xyz),
        q=list(pose_spec.quaternion_wxyz),
    )


def physical_material_spec_to_sapien(
    material_spec: PhysicalMaterialSpec | None,
) -> Any | None:
    if material_spec is None:
        return None
    if sapien is None:
        raise ImportError("SAPIEN is required to build layout physical materials.")
    return sapien.physx.PhysxMaterial(
        static_friction=material_spec.static_friction,
        dynamic_friction=material_spec.dynamic_friction,
        restitution=material_spec.restitution,
    )


def render_texture_path_for_sapien(texture_path: Path, item_name: str) -> Path:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional image stack.
        raise ImportError(
            f"Pillow is required to normalize texture {texture_path} for layout "
            f"item {item_name!r}. Install pillow or use an RGB/RGBA texture."
        ) from exc
    with Image.open(texture_path) as image:
        if image.mode in {"RGB", "RGBA"}:
            return texture_path
        texture_cache_dir = Path(tempfile.gettempdir()) / "evodexbench_texture_cache"
        texture_cache_dir.mkdir(parents=True, exist_ok=True)
        texture_stat = texture_path.stat()
        cache_key_source = (
            f"{texture_path.resolve()}:{texture_stat.st_mtime_ns}:"
            f"{texture_stat.st_size}"
        )
        cache_key = hashlib.sha1(cache_key_source.encode("utf-8")).hexdigest()
        converted_texture_path = texture_cache_dir / f"{cache_key}_rgb.png"
        if converted_texture_path.is_file():
            try:
                with Image.open(converted_texture_path) as cached:
                    cached.load()
                    if cached.mode == "RGB" and cached.size == image.size:
                        return converted_texture_path
            except (OSError, ValueError):
                pass
        # Publish only a complete PNG. Multiple simulator workers can request
        # the same conversion concurrently, including on shared storage.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=texture_cache_dir, suffix=".png", delete=False) as stream:
                temporary = Path(stream.name)
                image.convert("RGB").save(stream, format="PNG")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, converted_texture_path)
            except FileExistsError:
                valid = False
                try:
                    with Image.open(converted_texture_path) as cached:
                        cached.load()
                        valid = cached.mode == "RGB" and cached.size == image.size
                except (OSError, ValueError):
                    pass
                if not valid:
                    os.replace(temporary, converted_texture_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return converted_texture_path


@dataclass(frozen=True)
class ActorBuildValues:
    """Per-environment physical values for one dynamic layout actor."""

    physical_materials: tuple[PhysicalMaterialSpec, ...] | None = None
    mass_scales: tuple[float, ...] | None = None

    def validate(self, *, item_name: str, num_envs: int) -> None:
        value_groups = {
            "physical_materials": self.physical_materials,
            "mass_scales": self.mass_scales,
        }
        for value_name, values in value_groups.items():
            if values is not None and len(values) != num_envs:
                raise ValueError(
                    f"Actor values for {item_name!r} field {value_name} must "
                    f"contain {num_envs} values, got {len(values)}."
                )
        if self.mass_scales is not None and any(
            not np.isfinite(scale) or scale <= 0.0
            for scale in self.mass_scales
        ):
            raise ValueError(
                f"Mass scales for {item_name!r} must be finite and positive."
            )


@dataclass(frozen=True)
class ArticulationBuildValues:
    """Per-environment physical values for one articulated layout item."""

    physical_materials: tuple[PhysicalMaterialSpec, ...] | None = None
    mass_scales: tuple[float, ...] | None = None
    joint_limits: dict[
        str, tuple[tuple[float, float], ...]
    ] = field(default_factory=dict)
    joint_stiffness: dict[str, tuple[float, ...]] = field(default_factory=dict)
    joint_force_limit: dict[str, tuple[float, ...]] = field(default_factory=dict)
    joint_friction: dict[str, tuple[float, ...]] = field(default_factory=dict)
    joint_damping: dict[str, tuple[float, ...]] = field(default_factory=dict)

    def validate(self, *, item_name: str, num_envs: int) -> None:
        value_groups = {
            "physical_materials": self.physical_materials,
            "mass_scales": self.mass_scales,
            **{
                f"joint_friction[{joint_name!r}]": values
                for joint_name, values in self.joint_friction.items()
            },
            **{
                f"joint_damping[{joint_name!r}]": values
                for joint_name, values in self.joint_damping.items()
            },
            **{
                f"joint_limits[{joint_name!r}]": values
                for joint_name, values in self.joint_limits.items()
            },
            **{
                f"joint_stiffness[{joint_name!r}]": values
                for joint_name, values in self.joint_stiffness.items()
            },
            **{
                f"joint_force_limit[{joint_name!r}]": values
                for joint_name, values in self.joint_force_limit.items()
            },
        }
        for value_name, values in value_groups.items():
            if values is not None and len(values) != num_envs:
                raise ValueError(
                    f"Articulation values for {item_name!r} field {value_name} "
                    f"must contain {num_envs} values, got {len(values)}."
                )
        if self.mass_scales is not None:
            if any(
                not np.isfinite(scale) or scale <= 0.0
                for scale in self.mass_scales
            ):
                raise ValueError(
                    f"Mass scales for {item_name!r} must be finite and positive."
                )
        for joint_name, values in self.joint_friction.items():
            if any(not np.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(
                    f"Joint friction for {item_name!r}/{joint_name!r} must be "
                    "finite and non-negative."
                )
        for joint_name, values in self.joint_damping.items():
            if any(not np.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(
                    f"Joint damping for {item_name!r}/{joint_name!r} must be "
                    "finite and non-negative."
                )
        for joint_name, limits_by_env in self.joint_limits.items():
            for lower, upper in limits_by_env:
                if (
                    not np.isfinite(lower)
                    or not np.isfinite(upper)
                    or lower >= upper
                ):
                    raise ValueError(
                        f"Joint limits for {item_name!r}/{joint_name!r} must be "
                        "finite and satisfy lower < upper."
                    )
        for field_name, values_by_joint in (
            ("stiffness", self.joint_stiffness),
            ("force limit", self.joint_force_limit),
        ):
            for joint_name, values in values_by_joint.items():
                if any(
                    not np.isfinite(value) or value < 0.0 for value in values
                ):
                    raise ValueError(
                        f"Joint {field_name} for {item_name!r}/{joint_name!r} "
                        "must be finite and non-negative."
                    )


class LayoutBuilder:
    def __init__(
        self,
        env: Any,
        layout_spec: LayoutSpec,
        *,
        actor_values: dict[str, ActorBuildValues] | None = None,
        articulation_values: dict[str, ArticulationBuildValues] | None = None,
    ) -> None:
        self.env = env
        self.layout_spec = layout_spec
        self.actor_values = dict(actor_values or {})
        self.articulation_values = dict(articulation_values or {})

    @property
    def scene(self) -> Any:
        return self.env.scene

    def build(self) -> LayoutBuildResult:
        if sapien is None or build_ground is None:
            raise ImportError("ManiSkill and SAPIEN are required to build layouts.")

        articulated_item_names = {
            item.name
            for item in self.layout_spec.build_items
            if isinstance(item, ArticulatedAssetSpec)
        }
        dynamic_actor_item_names = {
            item.name
            for item in self.layout_spec.build_items
            if isinstance(
                item,
                (StaticBoxSpec, StaticMeshSpec, MJCFActorSpec, YCBActorSpec),
            )
            and item.body_type == "dynamic"
        }
        unknown_actor_item_names = sorted(
            set(self.actor_values) - dynamic_actor_item_names
        )
        if unknown_actor_item_names:
            raise ValueError(
                "Physical actor values reference unknown or non-dynamic layout "
                "items: "
                + ", ".join(unknown_actor_item_names)
            )
        for item_name, values in self.actor_values.items():
            values.validate(item_name=item_name, num_envs=self.env.num_envs)

        unknown_item_names = sorted(
            set(self.articulation_values) - articulated_item_names
        )
        if unknown_item_names:
            raise ValueError(
                "Physical articulation values reference unknown layout items: "
                + ", ".join(unknown_item_names)
            )
        for item_name, values in self.articulation_values.items():
            values.validate(item_name=item_name, num_envs=self.env.num_envs)

        actors: dict[str, Any] = {}
        articulations: dict[str, Any] = {}
        external_scene_builders: dict[str, Any] = {}
        for build_item in self.layout_spec.build_items:
            self._build_item(
                build_item,
                actors=actors,
                articulations=articulations,
                external_scene_builders=external_scene_builders,
            )

        ground = None
        if self.layout_spec.add_default_ground:
            floor_width = 500 if self.scene.parallel_in_single_scene else 100
            ground = build_ground(self.scene, floor_width=floor_width, altitude=-0.75)

        return LayoutBuildResult(
            layout_spec=self.layout_spec,
            actors=actors,
            articulations=articulations,
            external_scene_builders=external_scene_builders,
            ground=ground,
            actor_values=self.actor_values,
            articulation_values=self.articulation_values,
            _env=self.env,
        )

    def _build_item(
        self,
        build_item: LayoutBuildItemSpec,
        actors: dict[str, Any],
        articulations: dict[str, Any],
        external_scene_builders: dict[str, Any],
    ) -> None:
        if isinstance(build_item, StaticBoxSpec):
            actor = self._build_static_box(build_item)
            self._apply_actor_values(build_item.name, actor)
            actors[build_item.name] = actor
            return
        if isinstance(build_item, ProceduralTableSpec):
            actors[build_item.name] = self._build_procedural_table(build_item)
            return
        if isinstance(build_item, StaticMeshSpec):
            actor = self._build_static_mesh(build_item)
            self._apply_actor_values(build_item.name, actor)
            actors[build_item.name] = actor
            return
        if isinstance(build_item, MJCFActorSpec):
            actor = self._build_mjcf_actor(build_item)
            self._apply_actor_values(build_item.name, actor)
            actors[build_item.name] = actor
            return
        if isinstance(build_item, YCBActorSpec):
            actor = self._build_ycb_actor(build_item)
            self._apply_actor_values(build_item.name, actor)
            actors[build_item.name] = actor
            return
        if isinstance(build_item, TexturedPlaneSpec):
            actors[build_item.name] = self._build_textured_plane(build_item)
            return
        if isinstance(build_item, ArticulatedAssetSpec):
            values = self.articulation_values.get(build_item.name)
            articulations[build_item.name] = self._build_articulated_asset(
                build_item,
                values=values,
            )
            return
        if isinstance(build_item, ExternalSceneSpec):
            external_scene_builders[build_item.name] = self._build_external_scene(
                build_item
            )
            return
        raise TypeError(
            f"Unsupported layout build item type {type(build_item).__name__}."
        )

    def _apply_actor_values(self, item_name: str, actor: Any) -> None:
        values = self.actor_values.get(item_name)
        if values is None:
            return

        raw_bodies = tuple(getattr(actor, "_bodies", ()))
        if len(raw_bodies) != self.env.num_envs:
            raise ValueError(
                f"Physical values for actor {item_name!r} expect "
                f"{self.env.num_envs} raw bodies, got {len(raw_bodies)}."
            )
        for env_index, raw_body in enumerate(raw_bodies):
            if raw_body is None or bool(getattr(raw_body, "kinematic", False)):
                raise ValueError(
                    f"Physical values for actor {item_name!r} require a dynamic "
                    "rigid body."
                )
            if values.mass_scales is not None:
                mass_scale = values.mass_scales[env_index]
                nominal_mass = float(raw_body.mass)
                nominal_inertia = np.asarray(raw_body.inertia, dtype=np.float64)
                if (
                    not np.isfinite(nominal_mass)
                    or nominal_mass <= 0.0
                    or not np.isfinite(nominal_inertia).all()
                ):
                    raise ValueError(
                        f"Actor {item_name!r} has invalid nominal mass or inertia."
                    )
                raw_body.mass = nominal_mass * mass_scale
                # [3] * scalar -> [3]: preserve the nominal mass/inertia relation.
                raw_body.inertia = nominal_inertia * mass_scale
            if values.physical_materials is not None:
                material = physical_material_spec_to_sapien(
                    values.physical_materials[env_index]
                )
                for collision_shape in raw_body.collision_shapes:
                    collision_shape.set_physical_material(material)

    def _build_static_box(self, static_box: StaticBoxSpec) -> Any:
        builder = self.scene.create_actor_builder()
        local_pose = sapien.Pose()
        if static_box.add_collision:
            builder.add_box_collision(
                half_size=static_box.half_size_xyz,
                pose=local_pose,
                material=physical_material_spec_to_sapien(
                    static_box.physical_material
                ),
            )
        material = sapien.render.RenderMaterial(
            base_color=list(static_box.color_rgba),
        )
        builder.add_box_visual(
            half_size=static_box.half_size_xyz,
            pose=local_pose,
            material=material,
        )
        builder.initial_pose = pose_spec_to_sapien_pose(static_box.pose)
        return self._build_actor_with_body_type(
            builder=builder,
            name=static_box.name,
            body_type=static_box.body_type,
        )

    def _build_procedural_table(self, table_spec: ProceduralTableSpec) -> Any:
        builder = self.scene.create_actor_builder()
        tabletop_half_size = table_spec.tabletop_half_size_xyz
        tabletop_pose = sapien.Pose(
            p=[0.0, 0.0, -tabletop_half_size[2]],
        )
        tabletop_material = sapien.render.RenderMaterial(
            base_color=list(table_spec.tabletop_color_rgba),
        )
        builder.add_box_collision(
            half_size=tabletop_half_size,
            pose=tabletop_pose,
        )
        builder.add_box_visual(
            half_size=tabletop_half_size,
            pose=tabletop_pose,
            material=tabletop_material,
        )

        leg_half_size = table_spec.leg_half_size_xyz
        leg_center_z = -2.0 * tabletop_half_size[2] - leg_half_size[2]
        leg_x_offset = tabletop_half_size[0] - table_spec.leg_inset_xy[0]
        leg_y_offset = tabletop_half_size[1] - table_spec.leg_inset_xy[1]
        leg_material = sapien.render.RenderMaterial(
            base_color=list(table_spec.leg_color_rgba),
        )
        for leg_x_sign in (-1.0, 1.0):
            for leg_y_sign in (-1.0, 1.0):
                leg_pose = sapien.Pose(
                    p=[
                        leg_x_sign * leg_x_offset,
                        leg_y_sign * leg_y_offset,
                        leg_center_z,
                    ]
                )
                builder.add_box_collision(
                    half_size=leg_half_size,
                    pose=leg_pose,
                )
                builder.add_box_visual(
                    half_size=leg_half_size,
                    pose=leg_pose,
                    material=leg_material,
                )

        builder.initial_pose = pose_spec_to_sapien_pose(table_spec.pose)
        return self._build_actor_with_body_type(
            builder=builder,
            name=table_spec.name,
            body_type=table_spec.body_type,
        )

    def _build_static_mesh(self, static_mesh: StaticMeshSpec) -> Any:
        mesh_path = Path(static_mesh.mesh_path).expanduser()
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"Static mesh asset for layout item {static_mesh.name!r} was not "
                f"found at {mesh_path}."
            )

        collision_mesh_path = Path(
            static_mesh.collision_mesh_path or mesh_path
        ).expanduser()
        if (
            static_mesh.collision_type
            not in {"none", "compound_boxes", "compound_primitives"}
            and not collision_mesh_path.is_file()
        ):
            raise FileNotFoundError(
                f"Collision mesh for layout item {static_mesh.name!r} was not "
                f"found at {collision_mesh_path}."
            )
        if (
            static_mesh.body_type == "dynamic"
            and static_mesh.collision_type == "nonconvex"
        ):
            raise ValueError(
                f"Dynamic layout mesh {static_mesh.name!r} cannot use a "
                "nonconvex triangle-mesh collision. Use a convex or "
                "pre-decomposed multiple_convex collision mesh."
            )
        if (
            static_mesh.collision_type == "compound_boxes"
            and not static_mesh.collision_boxes
        ):
            raise ValueError(
                f"Static mesh {static_mesh.name!r} uses compound_boxes but "
                "defines no collision_boxes."
            )
        if (
            static_mesh.collision_type not in {"compound_boxes", "compound_primitives"}
            and static_mesh.collision_boxes
        ):
            raise ValueError(
                f"Static mesh {static_mesh.name!r} defines collision_boxes but "
                f"uses collision_type={static_mesh.collision_type!r}."
            )
        if (
            static_mesh.collision_type != "compound_primitives"
            and static_mesh.collision_cylinders
        ):
            raise ValueError(
                f"Static mesh {static_mesh.name!r} defines collision_cylinders but "
                f"uses collision_type={static_mesh.collision_type!r}."
            )
        if (
            static_mesh.collision_type == "compound_primitives"
            and not (static_mesh.collision_boxes or static_mesh.collision_cylinders)
        ):
            raise ValueError(
                f"Static mesh {static_mesh.name!r} uses compound_primitives but "
                "defines no collision primitives."
            )

        builder = self.scene.create_actor_builder()
        scale = list(static_mesh.scale_xyz)
        physical_material = physical_material_spec_to_sapien(
            static_mesh.physical_material
        )
        builder.add_visual_from_file(filename=str(mesh_path), scale=scale)
        if static_mesh.collision_type == "nonconvex":
            builder.add_nonconvex_collision_from_file(
                filename=str(collision_mesh_path),
                scale=scale,
                material=physical_material,
            )
        elif static_mesh.collision_type == "convex":
            builder.add_convex_collision_from_file(
                filename=str(collision_mesh_path),
                scale=scale,
                material=physical_material,
            )
        elif static_mesh.collision_type == "multiple_convex":
            builder.add_multiple_convex_collisions_from_file(
                filename=str(collision_mesh_path),
                scale=scale,
                material=physical_material,
            )
        elif static_mesh.collision_type in {"compound_boxes", "compound_primitives"}:
            scale_array = np.asarray(scale, dtype=np.float64)
            for collision_box in static_mesh.collision_boxes:
                # [3] * [3] -> [3]: scale box dimensions with the visual mesh.
                half_size = (
                    np.asarray(collision_box.half_size_xyz, dtype=np.float64)
                    * scale_array
                )
                # [3] * [3] -> [3]: scale the box center in the same mesh frame.
                position = (
                    np.asarray(
                        collision_box.pose.position_xyz,
                        dtype=np.float64,
                    )
                    * scale_array
                )
                box_pose = pose_spec_to_sapien_pose(
                    PoseSpec(
                        position_xyz=tuple(float(value) for value in position),
                        quaternion_wxyz=collision_box.pose.quaternion_wxyz,
                    )
                )
                builder.add_box_collision(
                    pose=box_pose,
                    half_size=tuple(float(value) for value in half_size),
                    material=physical_material,
                )
            if static_mesh.collision_cylinders and not np.isclose(
                scale_array[0], scale_array[1]
            ):
                raise ValueError(
                    f"Static mesh {static_mesh.name!r} uses Z-axis collision "
                    "cylinders, which require equal X/Y scale values."
                )
            sapien_x_to_local_z = (
                math.sqrt(0.5),
                0.0,
                -math.sqrt(0.5),
                0.0,
            )
            for collision_cylinder in static_mesh.collision_cylinders:
                # [3] * [3] -> [3]: scale the cylinder center in the mesh frame.
                position = (
                    np.asarray(
                        collision_cylinder.pose.position_xyz,
                        dtype=np.float64,
                    )
                    * scale_array
                )
                cylinder_pose = pose_spec_to_sapien_pose(
                    PoseSpec(
                        position_xyz=tuple(float(value) for value in position),
                        quaternion_wxyz=quat_multiply_wxyz(
                            collision_cylinder.pose.quaternion_wxyz,
                            sapien_x_to_local_z,
                        ),
                    )
                )
                builder.add_cylinder_collision(
                    pose=cylinder_pose,
                    radius=float(collision_cylinder.radius_m * scale_array[0]),
                    half_length=float(
                        collision_cylinder.half_length_m * scale_array[2]
                    ),
                    material=physical_material,
                )
        elif static_mesh.collision_type != "none":
            raise ValueError(
                f"Unsupported collision_type {static_mesh.collision_type!r} for "
                f"static mesh {static_mesh.name!r}."
            )

        builder.initial_pose = pose_spec_to_sapien_pose(static_mesh.pose)
        return self._build_actor_with_body_type(
            builder=builder,
            name=static_mesh.name,
            body_type=static_mesh.body_type,
        )

    def _build_mjcf_actor(self, mjcf_actor: MJCFActorSpec) -> Any:
        asset_path = Path(mjcf_actor.asset_path).expanduser()
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"MJCF asset for layout item {mjcf_actor.name!r} was not found "
                f"at {asset_path}."
            )
        if any(scale <= 0.0 for scale in mjcf_actor.scale_xyz):
            raise ValueError(
                f"MJCF scale_xyz for layout item {mjcf_actor.name!r} must contain "
                f"only positive values, got {mjcf_actor.scale_xyz}."
            )
        if not mjcf_actor.visual_groups:
            raise ValueError(
                f"MJCF visual_groups for layout item {mjcf_actor.name!r} cannot "
                "be empty."
            )

        loader = self.scene.create_mjcf_loader()
        loader.name = mjcf_actor.name
        scale_xyz = np.asarray(mjcf_actor.scale_xyz, dtype=np.float64)
        if np.all(scale_xyz == scale_xyz[0]):
            # ManiSkill's MJCF primitive sizes contain one or two entries;
            # uniform scaling must therefore use its scalar loader path.
            loader.scale = float(scale_xyz[0])
        else:
            loader.scale = scale_xyz.tolist()
        loader.visual_groups = list(mjcf_actor.visual_groups)
        parsed_asset = loader.parse(
            str(asset_path),
            package_dir=str(asset_path.parent),
        )
        actor_builders = parsed_asset.get("actor_builders", [])
        articulation_builders = parsed_asset.get("articulation_builders", [])
        if len(actor_builders) != 1 or articulation_builders:
            raise ValueError(
                f"Expected MJCF layout item {mjcf_actor.name!r} to contain exactly "
                f"one actor and no articulations, got {len(actor_builders)} actors "
                f"and {len(articulation_builders)} articulations."
            )

        actor_builder = actor_builders[0]
        actor_builder.initial_pose = pose_spec_to_sapien_pose(mjcf_actor.pose)
        return self._build_actor_with_body_type(
            builder=actor_builder,
            name=mjcf_actor.name,
            body_type=mjcf_actor.body_type,
        )

    def _build_ycb_actor(self, ycb_actor: YCBActorSpec) -> Any:
        if mani_skill_actors is None or MANISKILL_ASSET_DIR is None:
            raise ImportError("ManiSkill is required to build YCB layout actors.")
        if not ycb_actor.model_id:
            raise ValueError(
                f"YCB model_id for layout item {ycb_actor.name!r} cannot be empty."
            )
        if ycb_actor.scale_multiplier <= 0.0:
            raise ValueError(
                f"YCB scale_multiplier for layout item {ycb_actor.name!r} must "
                f"be positive, got {ycb_actor.scale_multiplier}."
            )
        if (
            not np.isfinite(ycb_actor.density_multiplier)
            or ycb_actor.density_multiplier <= 0.0
        ):
            raise ValueError(
                f"YCB density_multiplier for layout item {ycb_actor.name!r} must "
                f"be finite and positive, got {ycb_actor.density_multiplier}."
            )
        scale_xyz_multiplier = np.asarray(
            ycb_actor.scale_xyz_multiplier,
            dtype=np.float64,
        )
        if (
            scale_xyz_multiplier.shape != (3,)
            or not np.isfinite(scale_xyz_multiplier).all()
            or (scale_xyz_multiplier <= 0.0).any()
        ):
            raise ValueError(
                f"YCB scale_xyz_multiplier for layout item {ycb_actor.name!r} "
                "must contain three finite positive values, got "
                f"{ycb_actor.scale_xyz_multiplier}."
            )

        ycb_dataset_dir = (
            Path(MANISKILL_ASSET_DIR) / "assets" / "mani_skill2_ycb"
        )
        metadata_path = ycb_dataset_dir / "info_pick_v0.json"
        model_dir = ycb_dataset_dir / "models" / ycb_actor.model_id
        required_asset_paths = [metadata_path]
        if ycb_actor.add_collision:
            required_asset_paths.append(model_dir / "collision.ply")
        if ycb_actor.add_visual:
            required_asset_paths.append(model_dir / "textured.obj")
        missing_asset_paths = [
            asset_path
            for asset_path in required_asset_paths
            if not asset_path.is_file()
        ]
        if missing_asset_paths:
            missing_paths = "\n".join(
                f"  - {asset_path}" for asset_path in missing_asset_paths
            )
            raise FileNotFoundError(
                f"YCB assets for layout item {ycb_actor.name!r} and model "
                f"{ycb_actor.model_id!r} are incomplete:\n{missing_paths}\n"
                "Download them with `python -m mani_skill.utils.download_asset "
                "ycb -y` and verify MS_ASSET_DIR."
            )

        physical_material = physical_material_spec_to_sapien(
            ycb_actor.physical_material
        )

        if (
            ycb_actor.scale_multiplier == 1.0
            and np.all(scale_xyz_multiplier == 1.0)
            and ycb_actor.density_multiplier == 1.0
        ):
            builder = mani_skill_actors.get_actor_builder(
                self.scene,
                id=f"ycb:{ycb_actor.model_id}",
                add_collision=ycb_actor.add_collision,
                add_visual=ycb_actor.add_visual,
            )
            if physical_material is not None:
                for collision_record in builder.collision_records:
                    collision_record.material = physical_material
        else:
            with metadata_path.open(encoding="utf-8") as metadata_file:
                dataset_metadata: Any = json.load(metadata_file)
            if not isinstance(dataset_metadata, dict):
                raise ValueError(
                    f"YCB metadata at {metadata_path} must contain a JSON object."
                )
            try:
                model_metadata = dataset_metadata[ycb_actor.model_id]
            except KeyError as exc:
                raise ValueError(
                    f"YCB model {ycb_actor.model_id!r} required by layout item "
                    f"{ycb_actor.name!r} is missing from {metadata_path}."
                ) from exc
            if not isinstance(model_metadata, dict):
                raise ValueError(
                    f"YCB metadata for model {ycb_actor.model_id!r} must be an object."
                )
            model_scales = model_metadata.get("scales", [1.0])
            if not isinstance(model_scales, list) or not model_scales:
                raise ValueError(
                    f"YCB metadata for model {ycb_actor.model_id!r} must provide "
                    "at least one scale."
                )
            effective_scale = float(model_scales[0]) * ycb_actor.scale_multiplier
            density = (
                float(model_metadata.get("density", 1000.0))
                * ycb_actor.density_multiplier
            )
            # [3] * scalar -> [3]: preserve the source model scale while
            # allowing task-specific thickening without excessive lengthening.
            scale_xyz = (scale_xyz_multiplier * effective_scale).tolist()

            # Match ManiSkill's official YCB builder while allowing one layout
            # instance to uniformly scale both its collision and visual meshes.
            builder = self.scene.create_actor_builder()
            if ycb_actor.add_collision:
                builder.add_multiple_convex_collisions_from_file(
                    filename=str(model_dir / "collision.ply"),
                    scale=scale_xyz,
                    material=physical_material,
                    density=density,
                )
            if ycb_actor.add_visual:
                builder.add_visual_from_file(
                    filename=str(model_dir / "textured.obj"),
                    scale=scale_xyz,
                )
        builder.initial_pose = pose_spec_to_sapien_pose(ycb_actor.pose)
        return self._build_actor_with_body_type(
            builder=builder,
            name=ycb_actor.name,
            body_type=ycb_actor.body_type,
        )

    def _build_textured_plane(self, textured_plane: TexturedPlaneSpec) -> Any:
        builder = self.scene.create_actor_builder()
        # With render_backend="none" ManiSkill does not install a render
        # system. Constructing a RenderTexture2D in that state can segfault in
        # SAPIEN, so state-only policies keep only the plane's physical body.
        if getattr(self.env, "_render_device", None) is not None:
            texture_path = Path(textured_plane.texture_path).expanduser()
            if not texture_path.is_file():
                raise FileNotFoundError(
                    f"Texture asset for layout item {textured_plane.name!r} was not "
                    f"found at {texture_path}."
                )
            render_texture_path = render_texture_path_for_sapien(
                texture_path=texture_path,
                item_name=textured_plane.name,
            )
            material = sapien.render.RenderMaterial(
                base_color=[1.0, 1.0, 1.0, 1.0]
            )
            material.base_color_texture = sapien.render.RenderTexture2D(
                filename=str(render_texture_path),
                mipmap_levels=textured_plane.mipmap_levels,
            )
            builder.add_plane_repeated_visual(
                pose=pose_spec_to_sapien_pose(textured_plane.visual_pose),
                half_size=list(textured_plane.half_size_xy),
                mat=material,
                texture_repeat=list(textured_plane.texture_repeat_uv),
            )
        if textured_plane.collision_half_size_xyz is not None:
            builder.add_box_collision(
                half_size=textured_plane.collision_half_size_xyz,
                pose=sapien.Pose(),
            )
        builder.initial_pose = pose_spec_to_sapien_pose(textured_plane.pose)
        return self._build_actor_with_body_type(
            builder=builder,
            name=textured_plane.name,
            body_type=textured_plane.body_type,
        )

    def _build_articulated_asset(
        self,
        articulated_asset: ArticulatedAssetSpec,
        *,
        values: ArticulationBuildValues | None = None,
    ) -> Any:
        asset_path = Path(articulated_asset.asset_path).expanduser()
        if not asset_path.is_file():
            raise FileNotFoundError(
                f"Articulated asset for layout item {articulated_asset.name!r} was "
                f"not found at {asset_path}."
            )

        if values is not None:
            return self._build_articulated_asset_per_env(
                articulated_asset=articulated_asset,
                asset_path=asset_path,
                values=values,
            )

        loader = self._create_articulation_loader(
            articulated_asset=articulated_asset,
            loader_name=articulated_asset.name,
            physical_material=articulated_asset.physical_material,
        )
        articulation_builder = self._parse_articulation_builder(
            loader=loader,
            asset_path=asset_path,
            item_name=articulated_asset.name,
        )
        articulation_builder.initial_pose = pose_spec_to_sapien_pose(
            articulated_asset.pose
        )
        articulation = articulation_builder.build(
            name=articulated_asset.name,
            fix_root_link=articulated_asset.fix_root_link,
        )
        self._apply_articulation_defaults(articulation, articulated_asset)
        return articulation

    def _create_articulation_loader(
        self,
        *,
        articulated_asset: ArticulatedAssetSpec,
        loader_name: str,
        physical_material: PhysicalMaterialSpec | None,
    ) -> Any:
        if articulated_asset.asset_format == "urdf":
            loader = self.scene.create_urdf_loader()
        elif articulated_asset.asset_format == "mjcf":
            loader = self.scene.create_mjcf_loader()
        else:
            raise ValueError(
                f"Unsupported asset_format {articulated_asset.asset_format!r} for "
                f"layout item {articulated_asset.name!r}."
            )
        loader.name = loader_name
        if hasattr(loader, "fix_root_link"):
            loader.fix_root_link = articulated_asset.fix_root_link
        if hasattr(loader, "scale"):
            loader.scale = (
                articulated_asset.scale
                if articulated_asset.scale_xyz is None
                else list(articulated_asset.scale_xyz)
            )
        if hasattr(loader, "load_multiple_collisions_from_file"):
            loader.load_multiple_collisions_from_file = (
                articulated_asset.load_multiple_collisions_from_file
            )
        if hasattr(loader, "disable_self_collisions"):
            loader.disable_self_collisions = articulated_asset.disable_self_collisions
        sapien_material = physical_material_spec_to_sapien(physical_material)
        if sapien_material is not None:
            if not hasattr(loader, "set_material"):
                raise AttributeError(
                    f"{articulated_asset.asset_format.upper()} loader for layout item "
                    f"{articulated_asset.name!r} does not support a global physical "
                    "material."
                )
            loader.set_material(
                sapien_material.static_friction,
                sapien_material.dynamic_friction,
                sapien_material.restitution,
            )
        return loader

    def _parse_articulation_builder(
        self,
        *,
        loader: Any,
        asset_path: Path,
        item_name: str,
    ) -> Any:
        parse_path = asset_path
        temporary_path: Path | None = None
        if (
            getattr(self.env, "_render_device", None) is None
            and asset_path.suffix.lower() == ".urdf"
        ):
            # SAPIEN's URDF loader creates render components unconditionally.
            # When no render system exists, textured visuals can crash in
            # native code. Parse a temporary physics-only URDF while retaining
            # the original package directory for collision mesh resolution.
            root = ElementTree.parse(asset_path).getroot()
            for link in root.iter("link"):
                for visual in tuple(link.findall("visual")):
                    link.remove(visual)
            for mesh in root.iter("mesh"):
                filename = mesh.get("filename")
                if not filename or "://" in filename:
                    continue
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    mesh.set(
                        "filename",
                        str((asset_path.parent / mesh_path).resolve()),
                    )
            temporary_file = tempfile.NamedTemporaryFile(
                prefix="evodex-physics-only-",
                suffix=".urdf",
                delete=False,
            )
            temporary_file.close()
            temporary_path = Path(temporary_file.name)
            ElementTree.ElementTree(root).write(
                temporary_path,
                encoding="utf-8",
                xml_declaration=True,
            )
            parse_path = temporary_path
        try:
            parsed_asset = loader.parse(
                str(parse_path), package_dir=str(asset_path.parent)
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        articulation_builders = parsed_asset.get("articulation_builders", [])
        if len(articulation_builders) != 1:
            raise ValueError(
                f"Expected articulated asset {item_name!r} to contain "
                f"exactly one articulation, got {len(articulation_builders)}."
            )
        return articulation_builders[0]

    def _apply_articulation_defaults(
        self,
        articulation: Any,
        articulated_asset: ArticulatedAssetSpec,
    ) -> None:
        # GPU articulation state buffers are created only after every scene item
        # has been built. LayoutBuildResult.initialize_episode() applies these
        # same initial states after gpu_init().
        can_set_build_time_qpos = not self.env.gpu_sim_enabled
        if articulated_asset.initial_qpos is not None and can_set_build_time_qpos:
            self._set_articulation_initial_qpos(
                articulation=articulation,
                qpos=articulated_asset.initial_qpos,
                name=articulated_asset.name,
            )
        if articulated_asset.joint_initial_qpos and can_set_build_time_qpos:
            self._set_articulation_joint_initial_qpos(
                articulation=articulation,
                joint_initial_qpos=articulated_asset.joint_initial_qpos,
                name=articulated_asset.name,
            )
        if (
            articulated_asset.default_joint_friction is not None
            or articulated_asset.joint_friction
        ):
            self._set_articulation_joint_friction(
                articulation=articulation,
                default_joint_friction=articulated_asset.default_joint_friction,
                joint_friction=articulated_asset.joint_friction,
                name=articulated_asset.name,
            )
        if articulated_asset.joint_position_latches:
            self._set_articulation_joint_position_latches(
                articulation=articulation,
                joint_position_latches=(
                    articulated_asset.joint_position_latches
                ),
                name=articulated_asset.name,
            )
        if articulated_asset.link_disable_gravity:
            self._set_articulation_link_disable_gravity(
                articulation=articulation,
                link_disable_gravity=articulated_asset.link_disable_gravity,
                name=articulated_asset.name,
            )

    def _build_articulated_asset_per_env(
        self,
        *,
        articulated_asset: ArticulatedAssetSpec,
        asset_path: Path,
        values: ArticulationBuildValues,
    ) -> Any:
        if Articulation is None:
            raise ImportError("ManiSkill is required for per-environment articulations.")

        built_articulations: list[Any] = []
        for env_index in range(self.env.num_envs):
            item_name = f"{articulated_asset.name}_generalized_{env_index}"
            material = (
                values.physical_materials[env_index]
                if values.physical_materials is not None
                else articulated_asset.physical_material
            )
            loader = self._create_articulation_loader(
                articulated_asset=articulated_asset,
                loader_name=item_name,
                physical_material=material,
            )
            articulation_builder = self._parse_articulation_builder(
                loader=loader,
                asset_path=asset_path,
                item_name=articulated_asset.name,
            )
            articulation_builder.set_scene_idxs([env_index])
            articulation_builder.initial_pose = pose_spec_to_sapien_pose(
                articulated_asset.pose
            )
            self._apply_articulation_builder_joint_limits(
                articulation_builder=articulation_builder,
                values=values,
                env_index=env_index,
                item_name=articulated_asset.name,
            )
            articulation = articulation_builder.build(
                name=item_name,
                fix_root_link=articulated_asset.fix_root_link,
            )
            self._apply_articulation_defaults(articulation, articulated_asset)
            if values.mass_scales is not None:
                self._scale_articulation_mass(
                    articulation,
                    scale=values.mass_scales[env_index],
                    item_name=articulated_asset.name,
                )
            self._apply_articulation_joint_values(
                articulation,
                values=values,
                env_index=env_index,
                item_name=articulated_asset.name,
                initial_qpos_by_joint=self._configured_initial_qpos_by_joint(
                    articulation=articulation,
                    articulated_asset=articulated_asset,
                ),
                apply_joint_limits=False,
            )
            self.scene.remove_from_state_dict_registry(articulation)
            built_articulations.append(articulation)

        merged_articulation = Articulation.merge(
            built_articulations,
            name=articulated_asset.name,
            merge_links=True,
        )
        self.scene.add_to_state_dict_registry(merged_articulation)
        return merged_articulation

    @staticmethod
    def _apply_articulation_builder_joint_limits(
        *,
        articulation_builder: Any,
        values: ArticulationBuildValues,
        env_index: int,
        item_name: str,
    ) -> None:
        """Apply limits before PhysX entities enter a scene.

        PhysX rejects joint-motion changes after an articulation has been added to
        a scene. ManiSkill's articulation builder stores the source joint records,
        so per-environment limits must be written here, before ``build()``.
        """

        link_builder_by_joint_name = {
            str(link_builder.joint_record.name): link_builder
            for link_builder in articulation_builder.link_builders
            if str(link_builder.joint_record.name)
        }
        unknown_joint_names = sorted(
            set(values.joint_limits) - set(link_builder_by_joint_name)
        )
        if unknown_joint_names:
            raise ValueError(
                f"Joint limits for {item_name!r} reference unknown joints: "
                + ", ".join(unknown_joint_names)
            )
        for joint_name, limits_by_env in values.joint_limits.items():
            lower, upper = limits_by_env[env_index]
            # [1, 2]: one scalar-DoF limit interval expected by SAPIEN.
            link_builder_by_joint_name[joint_name].joint_record.limits = np.asarray(
                [[lower, upper]], dtype=np.float32
            )

    @staticmethod
    def _configured_initial_qpos_by_joint(
        *,
        articulation: Any,
        articulated_asset: ArticulatedAssetSpec,
    ) -> dict[str, float]:
        joint_names = tuple(
            str(joint.name)
            for joint in tuple(getattr(articulation, "active_joints", ()))
        )
        qpos = np.zeros(len(joint_names), dtype=np.float64)
        if articulated_asset.initial_qpos is not None:
            qpos = np.asarray(
                articulated_asset.initial_qpos, dtype=np.float64
            ).copy()
        for joint_index, joint_name in enumerate(joint_names):
            if joint_name in articulated_asset.joint_initial_qpos:
                qpos[joint_index] = float(
                    articulated_asset.joint_initial_qpos[joint_name]
                )
        return {
            joint_name: float(qpos[joint_index])
            for joint_index, joint_name in enumerate(joint_names)
        }

    @staticmethod
    def _scale_articulation_mass(
        articulation: Any,
        *,
        scale: float,
        item_name: str,
    ) -> None:
        raw_articulations = tuple(getattr(articulation, "_objs", ()))
        if len(raw_articulations) != 1:
            raise ValueError(
                f"Mass scaling for {item_name!r} expects one sub-scene articulation, "
                f"got {len(raw_articulations)}."
            )
        for raw_link in raw_articulations[0].links:
            mass = float(raw_link.mass)
            inertia = np.asarray(raw_link.inertia, dtype=np.float64)
            if not np.isfinite(mass) or mass < 0.0 or not np.isfinite(inertia).all():
                raise ValueError(
                    f"Articulation {item_name!r} has invalid nominal mass or inertia."
                )
            raw_link.mass = mass * scale
            # [3] * scalar -> [3]: preserve the nominal mass/inertia relation.
            raw_link.inertia = inertia * scale

    @staticmethod
    def _apply_articulation_joint_values(
        articulation: Any,
        *,
        values: ArticulationBuildValues,
        env_index: int,
        item_name: str,
        initial_qpos_by_joint: dict[str, float] | None = None,
        apply_joint_limits: bool = True,
    ) -> None:
        joint_by_name = {
            str(getattr(joint, "name", "")): joint
            for joint in tuple(getattr(articulation, "active_joints", ()))
        }
        requested_joint_names = (
            set(values.joint_limits)
            | set(values.joint_stiffness)
            | set(values.joint_force_limit)
            | set(values.joint_friction)
            | set(values.joint_damping)
        )
        unknown_joint_names = sorted(requested_joint_names - set(joint_by_name))
        if unknown_joint_names:
            raise ValueError(
                f"Physical values for {item_name!r} reference unknown joints: "
                + ", ".join(unknown_joint_names)
            )

        for joint_name, limits_by_env in values.joint_limits.items():
            joint = joint_by_name[joint_name]
            raw_joints = tuple(getattr(joint, "_objs", ()))
            if len(raw_joints) != 1:
                raise ValueError(
                    f"Joint limits for {item_name!r}/{joint_name!r} expect one "
                    f"raw joint, got {len(raw_joints)}."
                )
            raw_joint = raw_joints[0]
            lower, upper = limits_by_env[env_index]
            if initial_qpos_by_joint is not None:
                joint_qpos = initial_qpos_by_joint[joint_name]
            else:
                active_index = getattr(joint, "active_index", None)
                if active_index is None:
                    raise ValueError(
                        f"Joint limits for {item_name!r}/{joint_name!r} require an "
                        "active joint."
                    )
                flat_active_indices = np.asarray(
                    active_index.detach().cpu().tolist(), dtype=np.int64
                ).reshape(-1)
                if flat_active_indices.size != 1:
                    raise ValueError(
                        f"Joint limits for {item_name!r}/{joint_name!r} expect one "
                        f"active index, got {flat_active_indices.size}."
                    )
                raw_articulations = tuple(getattr(articulation, "_objs", ()))
                if len(raw_articulations) != 1:
                    raise ValueError(
                        f"Joint limits for {item_name!r}/{joint_name!r} expect one "
                        f"raw articulation, got {len(raw_articulations)}."
                    )
                current_qpos = np.asarray(
                    raw_articulations[0].qpos, dtype=np.float64
                ).reshape(-1)
                joint_qpos = float(
                    current_qpos[int(flat_active_indices[0])]
                )
            if not np.isfinite(joint_qpos) or not lower <= joint_qpos <= upper:
                raise ValueError(
                    f"Initial qpos {joint_qpos} for {item_name!r}/{joint_name!r} "
                    f"is outside requested limits ({lower}, {upper})."
                )
            if apply_joint_limits:
                raw_joint.limits = np.asarray(
                    [[lower, upper]], dtype=np.float32
                )

        for joint_name, friction_values in values.joint_friction.items():
            joint_by_name[joint_name].set_friction(
                float(friction_values[env_index])
            )

        drive_joint_names = (
            set(values.joint_stiffness)
            | set(values.joint_damping)
            | set(values.joint_force_limit)
        )
        for joint_name in drive_joint_names:
            joint = joint_by_name[joint_name]
            raw_joints = tuple(getattr(joint, "_objs", ()))
            if len(raw_joints) != 1:
                raise ValueError(
                    f"Joint drive values for {item_name!r}/{joint_name!r} expect "
                    f"one raw joint, got {len(raw_joints)}."
                )
            raw_joint = raw_joints[0]
            stiffness = (
                float(values.joint_stiffness[joint_name][env_index])
                if joint_name in values.joint_stiffness
                else float(raw_joint.stiffness)
            )
            damping = (
                float(values.joint_damping[joint_name][env_index])
                if joint_name in values.joint_damping
                else float(raw_joint.damping)
            )
            force_limit = (
                float(values.joint_force_limit[joint_name][env_index])
                if joint_name in values.joint_force_limit
                else float(raw_joint.force_limit)
            )
            raw_joint.set_drive_properties(
                stiffness,
                damping,
                force_limit,
                raw_joint.drive_mode,
            )

    def _set_articulation_initial_qpos(
        self,
        articulation: Any,
        qpos: tuple[float, ...],
        name: str,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to set articulation initial qpos.")
        max_dof = getattr(articulation, "max_dof", None)
        if max_dof is not None and len(qpos) != max_dof:
            raise ValueError(
                f"Initial qpos for articulated layout item {name!r} has length "
                f"{len(qpos)}, expected {max_dof}."
            )
        raw_articulations = tuple(getattr(articulation, "_objs", ()))
        if not raw_articulations:
            raise ValueError(
                f"Articulated layout item {name!r} has no raw articulations."
            )
        qpos_array = np.asarray(qpos, dtype=np.float32)
        # Build happens before gpu_init(), so CUDA articulation buffers do not
        # exist yet. Set each raw SAPIEN articulation's [D] initial state.
        for raw_articulation in raw_articulations:
            raw_articulation.qpos = qpos_array.copy()
            raw_articulation.qvel = np.zeros_like(qpos_array)

    def _set_articulation_joint_initial_qpos(
        self,
        articulation: Any,
        joint_initial_qpos: dict[str, float],
        name: str,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to set articulation joint qpos.")

        active_joints = tuple(getattr(articulation, "active_joints", ()))
        if not active_joints:
            raise ValueError(
                f"Articulated layout item {name!r} does not expose active joints, "
                "so named joint qpos cannot be applied."
            )

        joint_names = [str(getattr(joint, "name", "")) for joint in active_joints]
        if any(not joint_name for joint_name in joint_names):
            raise ValueError(
                f"Articulated layout item {name!r} has unnamed active joints, "
                "so named joint qpos cannot be applied safely."
            )
        if len(set(joint_names)) != len(joint_names):
            raise ValueError(
                f"Articulated layout item {name!r} has duplicate active joint names, "
                "so named joint qpos cannot be applied safely."
            )

        unknown_joint_names = sorted(set(joint_initial_qpos) - set(joint_names))
        if unknown_joint_names:
            known_joint_names = ", ".join(joint_names)
            raise ValueError(
                f"Articulated layout item {name!r} received qpos for unknown joints "
                f"{unknown_joint_names}. Known joints: {known_joint_names}."
            )

        raw_articulations = tuple(getattr(articulation, "_objs", ()))
        if not raw_articulations:
            raise ValueError(
                f"Articulated layout item {name!r} has no raw articulations."
            )
        for raw_articulation in raw_articulations:
            # [D] -> [D]: preserve unspecified build-time qpos values.
            qpos_array = np.asarray(raw_articulation.qpos, dtype=np.float32).copy()
            for joint_index, joint_name in enumerate(joint_names):
                if joint_name in joint_initial_qpos:
                    qpos_array[joint_index] = float(
                        joint_initial_qpos[joint_name]
                    )
            raw_articulation.qpos = qpos_array
            raw_articulation.qvel = np.zeros_like(qpos_array)

    def _set_articulation_joint_friction(
        self,
        articulation: Any,
        default_joint_friction: float | None,
        joint_friction: dict[str, float],
        name: str,
    ) -> None:
        active_joints = tuple(getattr(articulation, "active_joints", ()))
        joint_by_name = {
            str(getattr(joint, "name", "")): joint for joint in active_joints
        }

        if len(joint_by_name) != len(active_joints):
            raise ValueError(
                f"Articulated layout item {name!r} has duplicate or unnamed active "
                "joints, so joint friction cannot be applied safely."
            )

        unknown_joint_names = sorted(set(joint_friction) - set(joint_by_name))
        if unknown_joint_names:
            known_joint_names = ", ".join(sorted(joint_by_name))
            raise ValueError(
                f"Articulated layout item {name!r} received friction for unknown "
                f"joints {unknown_joint_names}. Known joints: {known_joint_names}."
            )

        friction_by_joint_name = (
            {
                joint_name: default_joint_friction
                for joint_name in joint_by_name
            }
            if default_joint_friction is not None
            else {}
        )
        friction_by_joint_name.update(joint_friction)
        for joint_name, friction in friction_by_joint_name.items():
            joint = joint_by_name[joint_name]
            if not hasattr(joint, "set_friction"):
                raise AttributeError(
                    f"Active joint {joint_name!r} on articulated layout item {name!r} "
                    "does not support set_friction()."
                )
            joint.set_friction(float(friction))

    def _set_articulation_joint_position_latches(
        self,
        articulation: Any,
        joint_position_latches: dict[str, JointPositionLatchSpec],
        name: str,
    ) -> None:
        active_joints = tuple(getattr(articulation, "active_joints", ()))
        joint_by_name = {
            str(getattr(joint, "name", "")): joint for joint in active_joints
        }
        unknown_joint_names = sorted(
            set(joint_position_latches) - set(joint_by_name)
        )
        if unknown_joint_names:
            known_joint_names = ", ".join(sorted(joint_by_name))
            raise ValueError(
                f"Articulated layout item {name!r} received position latches for "
                f"unknown joints {unknown_joint_names}. Known joints: "
                f"{known_joint_names}."
            )

        for joint_name, latch_spec in joint_position_latches.items():
            joint = joint_by_name[joint_name]
            joint.set_drive_properties(
                stiffness=latch_spec.stiffness,
                damping=latch_spec.damping,
                force_limit=latch_spec.force_limit,
                mode="force",
            )
            if not self.env.gpu_sim_enabled:
                # One active revolute joint: [B, 1] or [B] -> [B].
                joint.set_drive_target(joint.qpos.reshape(-1))

    def _set_articulation_link_disable_gravity(
        self,
        articulation: Any,
        link_disable_gravity: tuple[str, ...],
        name: str,
    ) -> None:
        links = tuple(getattr(articulation, "links", ()) or ())
        if not links and hasattr(articulation, "get_links"):
            links = tuple(articulation.get_links())

        link_by_name = {str(getattr(link, "name", "")): link for link in links}

        if len(link_by_name) != len(links):
            raise ValueError(
                f"Articulated layout item {name!r} has duplicate or unnamed links, "
                "so link disable_gravity cannot be applied safely."
            )

        unknown_link_names = sorted(set(link_disable_gravity) - set(link_by_name))
        if unknown_link_names:
            known_link_names = ", ".join(sorted(link_by_name))
            raise ValueError(
                f"Articulated layout item {name!r} received disable_gravity for "
                f"unknown links {unknown_link_names}. Known links: {known_link_names}."
            )

        for link_name in link_disable_gravity:
            link = link_by_name[link_name]
            link_component = self._resolve_articulation_link_component(link)
            if hasattr(link_component, "set_disable_gravity"):
                link_component.set_disable_gravity(True)
            elif hasattr(link_component, "disable_gravity"):
                link_component.disable_gravity = True
            else:
                raise AttributeError(
                    f"Link {link_name!r} on articulated layout item {name!r} does not "
                    "support disable_gravity."
                )

    def _resolve_articulation_link_component(self, link: Any) -> Any:
        if hasattr(link, "find_component_by_type") and sapien is not None:
            component = link.find_component_by_type(
                sapien.physx.PhysxArticulationLinkComponent
            )
            if component is not None:
                return component
        return link

    def _build_external_scene(self, external_scene: ExternalSceneSpec) -> Any:
        build_config_idxs = None
        if external_scene.build_config_idx is not None:
            build_config_idxs = [external_scene.build_config_idx] * self.env.num_envs

        if external_scene.backend == "robocasa":
            try:
                from dex_benchmark.sim.scenes.robocasa.scene_builder import (
                    RoboCasaSceneBuilder,
                )
            except Exception as exc:  # pragma: no cover - optional asset stack.
                raise ImportError(
                    "RoboCasaSceneBuilder is required for ExternalSceneSpec backend "
                    "'robocasa'. Install ManiSkill with RoboCasa support and download "
                    "the RoboCasa assets first."
                ) from exc

            scene_builder = RoboCasaSceneBuilder(
                self.env,
                init_robot_base_pos=external_scene.init_robot_base_pos,
            )

            scene_builder.remove_fixture_names = set(
                external_scene.remove_fixture_names
            )

            scene_builder.build(build_config_idxs=build_config_idxs)
            return scene_builder

        if external_scene.backend == "replicacad":
            try:
                from mani_skill.utils.scene_builder.replicacad.scene_builder import (
                    ReplicaCADSceneBuilder,
                )
            except Exception as exc:  # pragma: no cover - optional asset stack.
                raise ImportError(
                    "ReplicaCADSceneBuilder is required for ExternalSceneSpec backend "
                    "'replicacad'. Install ManiSkill with ReplicaCAD support and download "
                    "the ReplicaCAD assets first."
                ) from exc

            scene_builder = ReplicaCADSceneBuilder(self.env)
            scene_builder.build(build_config_idxs=build_config_idxs)
            return scene_builder

        if external_scene.backend == "ai2thor":
            try:
                from mani_skill.utils.scene_builder.ai2thor.variants import (
                    iTHORSceneBuilder,
                )
            except Exception as exc:  # pragma: no cover - optional asset stack.
                raise ImportError(
                    "iTHORSceneBuilder is required for ExternalSceneSpec backend "
                    "'ai2thor'. Install ManiSkill with AI2-THOR support and download "
                    "the AI2-THOR assets first."
                ) from exc

            scene_builder = iTHORSceneBuilder(self.env)
            scene_builder.build(build_config_idxs=build_config_idxs)
            return scene_builder

        raise ValueError(
            f"Unsupported external scene backend {external_scene.backend!r}."
        )

    def _build_actor_with_body_type(
        self,
        builder: Any,
        name: str,
        body_type: str,
    ) -> Any:
        if body_type == "kinematic":
            return builder.build_kinematic(name=name)
        if body_type == "static":
            return builder.build_static(name=name)
        if body_type == "dynamic":
            return builder.build_dynamic(name=name)
        raise ValueError(
            f"Unsupported body_type {body_type!r} for layout actor {name!r}."
        )
