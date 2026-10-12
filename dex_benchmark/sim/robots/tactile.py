from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any

import numpy as np
import torch

from .shadow_hand.taxel_layout import TaxelSpec
from .virtual_taxel import (
    TaxelImpulseDistribution,
    distribute_contact_impulses_to_taxels,
    distribute_contact_impulses_to_taxels_with_diagnostics,
    rotate_vectors_by_wxyz_quaternions,
)


TAXEL_ASSIGNMENT_MARGIN_M = 0.001
TAXEL_ASSIGNMENT_NORMAL_TOLERANCE_M = 0.010
GPU_PARENT_CONTACT_POINTS_PER_ENV = 512


@dataclass(frozen=True)
class TactileQueryCounters:
    """Observable query counts used by performance and cache tests."""

    body_query_builds: int
    body_query_executions: int
    pair_query_builds: int
    pair_query_executions: int
    cache_invalidations: int


@dataclass(frozen=True)
class TactileContactDiagnostics:
    """Control-period contact mapping checks for one batched environment."""

    raw_total_force_n: float
    raw_covered_force_n: float
    mapped_force_n: float
    force_error: float
    raw_covered_torque_nm: float
    mapped_torque_nm: float
    torque_error_nm: float
    taxel_vector_force_n: float
    taxel_scalar_force_n: float
    taxel_force_norm_sum_n: float
    covered_contact_count: int
    total_contact_count: int
    coverage_ratio: float
    force_coverage_ratio: float
    centroid_error_mean_m: float
    centroid_error_max_m: float


@dataclass(frozen=True)
class TactileContactState:
    """Control-period contact vectors used by action-side safety filters."""

    parent_force_world: torch.Tensor
    parent_normal_force_world: torch.Tensor
    finger_force_world: torch.Tensor
    taxel_force_world: torch.Tensor
    hand_force_world: torch.Tensor
    parent_link_names: tuple[str, ...]
    finger_names: tuple[str, ...]
    parent_finger_indices: tuple[int, ...]


@dataclass
class _PairQuery:
    handle: Any | None
    target_count: int
    target_group_slices: tuple[tuple[str, slice], ...]


@dataclass(frozen=True)
class CpuPairwiseContactQueryHandle:
    """Opaque handle for one persistent CPU link-pair force query."""

    signature: tuple[object, ...]


@dataclass(frozen=True)
class PairwiseContactQueryHandle:
    """Opaque handle for one persistent CPU/GPU link-pair force query."""

    signature: tuple[object, ...]


@dataclass
class _CpuObjectQuery:
    target_count: int
    target_group_slices: tuple[tuple[str, slice], ...]
    target_entities: dict[Any, tuple[int, int]]
    world_impulses: np.ndarray | None = None


@dataclass
class _CpuPairQuery:
    first_count: int
    second_count: int
    first_entities: dict[Any, tuple[int, int]]
    second_entities: dict[Any, tuple[int, int]]
    pair_impulses: np.ndarray | None = None


@dataclass
class _GpuPairQuery:
    first_links: tuple[Any, ...]
    second_links: tuple[Any, ...]
    first_count: int
    second_count: int
    handle: Any | None = None
    cached_forces: torch.Tensor | None = None


class TactileQueryManager:
    """Own persistent PhysX taxel queries and per-physics-state results."""

    def __init__(
        self,
        *,
        scene: Any,
        robot_links: Sequence[Any],
        taxel_parent_links: Sequence[Any],
        taxel_specs: Sequence[TaxelSpec],
        force_sign: float = -1.0,
        assignment_margin_m: float = TAXEL_ASSIGNMENT_MARGIN_M,
        normal_tolerance_m: float = TAXEL_ASSIGNMENT_NORMAL_TOLERANCE_M,
    ) -> None:
        self.scene = scene
        self.robot_links = tuple(robot_links)
        self.taxel_parent_links = tuple(taxel_parent_links)
        self.taxel_specs = tuple(taxel_specs)
        self.force_sign = float(force_sign)
        self.assignment_margin_m = float(assignment_margin_m)
        self.normal_tolerance_m = float(normal_tolerance_m)
        if self.force_sign not in (-1.0, 1.0):
            raise ValueError(
                f"force_sign must be -1.0 or 1.0, got {self.force_sign}."
            )
        if not self.robot_links:
            raise ValueError("robot_links must not be empty.")
        if self.assignment_margin_m < 0.0:
            raise ValueError("assignment_margin_m must be non-negative.")
        if self.normal_tolerance_m <= 0.0:
            raise ValueError("normal_tolerance_m must be positive.")
        if len(self.taxel_parent_links) != len(self.taxel_specs):
            raise ValueError(
                "Taxel parent/spec count mismatch: "
                f"{len(self.taxel_parent_links)} parents and "
                f"{len(self.taxel_specs)} specs."
            )
        actual_parent_names = tuple(
            parent_link.name for parent_link in self.taxel_parent_links
        )
        expected_parent_names = tuple(
            spec.parent_link_name for spec in self.taxel_specs
        )
        if actual_parent_names != expected_parent_names:
            raise ValueError(
                "Taxel parent links are not in canonical layout order. "
                f"Expected {expected_parent_names}, got {actual_parent_names}."
            )
        self._body_query: Any | None = None
        self._pair_queries: dict[tuple[object, ...], _PairQuery] = {}
        self._cached_net_normal_forces: torch.Tensor | None = None
        self._cached_parent_force_world: torch.Tensor | None = None
        self._cached_parent_normal_force_world: torch.Tensor | None = None
        self._cached_finger_force_world: torch.Tensor | None = None
        self._cached_hand_force_world: torch.Tensor | None = None
        self._cached_object_normal_forces: dict[
            tuple[object, ...],
            dict[str, torch.Tensor],
        ] = {}
        self._cached_object_world_forces: dict[
            tuple[object, ...],
            dict[str, torch.Tensor],
        ] = {}
        self._cached_world_normals: torch.Tensor | None = None
        self._cached_world_positions: torch.Tensor | None = None
        self._cached_world_tangent_u: torch.Tensor | None = None
        self._cached_world_tangent_v: torch.Tensor | None = None
        self._cached_cpu_contacts: list[Any] | None = None
        self._cpu_contact_frame_populated = False
        self._cpu_pair_frame_populated = False
        self._cpu_net_world_impulses: np.ndarray | None = None
        self._cpu_object_queries: dict[tuple[object, ...], _CpuObjectQuery] = {}
        self._cpu_pair_queries: dict[tuple[object, ...], _CpuPairQuery] = {}
        self._cpu_pair_routes: dict[
            tuple[Any, Any],
            list[tuple[tuple[object, ...], int, int, int, float]],
        ] = {}
        self._gpu_pair_queries: dict[tuple[object, ...], _GpuPairQuery] = {}
        self._body_query_builds = 0
        self._body_query_executions = 0
        self._pair_query_builds = 0
        self._pair_query_executions = 0
        self._cache_invalidations = 0
        self._control_world_impulse_sum: torch.Tensor | None = None
        self._control_sim_step_count = 0
        self._control_accumulation_active = False
        self._control_average_ready = False
        self._cpu_control_profile: dict[str, float | int] = {}
        # Mapping diagnostics are opt-in because they add contact geometry,
        # torque, and coverage reductions to every physics-step query.
        self._contact_diagnostics_enabled = False
        self._current_step_diagnostics: dict[str, torch.Tensor] | None = None
        self._control_diagnostics: dict[str, torch.Tensor] | None = None

        self._local_normals = torch.as_tensor(
            tuple(spec.local_normal for spec in self.taxel_specs),
            device=self.scene.device,
            dtype=torch.float32,
        )
        self._local_positions = torch.as_tensor(
            tuple(spec.local_position_m for spec in self.taxel_specs),
            device=self.scene.device,
            dtype=torch.float32,
        )
        self._local_tangent_u = torch.as_tensor(
            tuple(spec.tangent_u for spec in self.taxel_specs),
            device=self.scene.device,
            dtype=torch.float32,
        )
        self._local_tangent_v = torch.linalg.cross(
            self._local_normals,
            self._local_tangent_u,
            dim=-1,
        )
        self._taxel_half_sizes = 0.5 * torch.as_tensor(
            tuple(spec.patch_size_m for spec in self.taxel_specs),
            device=self.scene.device,
            dtype=torch.float32,
        )

        unique_parent_links: list[Any] = []
        unique_parent_index_by_id: dict[int, int] = {}
        taxel_parent_indices: list[int] = []
        for parent_link in self.taxel_parent_links:
            parent_id = id(parent_link)
            parent_index = unique_parent_index_by_id.get(parent_id)
            if parent_index is None:
                parent_index = len(unique_parent_links)
                unique_parent_index_by_id[parent_id] = parent_index
                unique_parent_links.append(parent_link)
            taxel_parent_indices.append(parent_index)
        self._unique_parent_links = tuple(unique_parent_links)
        parent_finger_indices: list[int] = []
        for parent_link in self._unique_parent_links:
            matching_specs = [
                spec
                for spec, candidate in zip(
                    self.taxel_specs,
                    self.taxel_parent_links,
                    strict=True,
                )
                if candidate is parent_link
            ]
            if not matching_specs:
                raise ValueError(
                    f"No taxel metadata found for parent link {parent_link.name!r}."
                )
            finger_indices = {int(spec.finger_index) for spec in matching_specs}
            if len(finger_indices) != 1:
                raise ValueError(
                    f"Parent link {parent_link.name!r} maps to multiple fingers: "
                    f"{sorted(finger_indices)}."
                )
            parent_finger_indices.append(next(iter(finger_indices)))
        self._parent_finger_indices = torch.as_tensor(
            parent_finger_indices,
            device=self.scene.device,
            dtype=torch.int64,
        )
        self._taxel_parent_indices = torch.as_tensor(
            taxel_parent_indices,
            device=self.scene.device,
            dtype=torch.int64,
        )
        self._parent_entities: dict[Any, tuple[int, int]] = {}
        self._robot_entities: set[Any] = set()
        if not self.scene.gpu_sim_enabled:
            for robot_link in self.robot_links:
                for body in self._validated_bodies(robot_link):
                    self._robot_entities.add(body.entity)
            for parent_index, parent_link in enumerate(
                self._unique_parent_links
            ):
                for env_index, body in enumerate(
                    self._validated_bodies(parent_link)
                ):
                    entity = body.entity
                    if entity in self._parent_entities:
                        raise ValueError(
                            "A PhysX entity cannot represent two tactile parents."
                        )
                    self._parent_entities[entity] = (env_index, parent_index)

    @property
    def num_envs(self) -> int:
        return int(self.scene.num_envs)

    @property
    def num_taxels(self) -> int:
        return len(self.taxel_specs)

    @property
    def counters(self) -> TactileQueryCounters:
        return TactileQueryCounters(
            body_query_builds=self._body_query_builds,
            body_query_executions=self._body_query_executions,
            pair_query_builds=self._pair_query_builds,
            pair_query_executions=self._pair_query_executions,
            cache_invalidations=self._cache_invalidations,
        )

    def invalidate(self) -> None:
        """Drop result tensors while retaining PhysX query handles."""

        self._cached_net_normal_forces = None
        self._cached_parent_force_world = None
        self._cached_parent_normal_force_world = None
        self._cached_finger_force_world = None
        self._cached_hand_force_world = None
        self._cached_object_normal_forces.clear()
        self._cached_object_world_forces.clear()
        self._cached_world_normals = None
        self._cached_world_positions = None
        self._cached_world_tangent_u = None
        self._cached_world_tangent_v = None
        self._cached_cpu_contacts = None
        self._cpu_contact_frame_populated = False
        self._cpu_pair_frame_populated = False
        self._cpu_net_world_impulses = None
        for query in self._cpu_object_queries.values():
            query.world_impulses = None
        for query in self._cpu_pair_queries.values():
            query.pair_impulses = None
        for query in self._gpu_pair_queries.values():
            query.cached_forces = None
        self._cache_invalidations += 1

    def reset_control_step_accumulation(self) -> None:
        """Discard any current or completed control-period impulse sum."""

        self._control_world_impulse_sum = None
        self._control_sim_step_count = 0
        self._control_accumulation_active = False
        self._control_average_ready = False
        self._current_step_diagnostics = None
        self._control_diagnostics = None
        self.invalidate()

    def set_contact_diagnostics_enabled(self, enabled: bool) -> None:
        """Enable expensive force/moment mapping diagnostics for test runs."""

        self._contact_diagnostics_enabled = bool(enabled)
        self._current_step_diagnostics = None
        self._control_diagnostics = None

    def begin_control_step_accumulation(self) -> None:
        """Start a fresh sum of per-physics-step taxel impulses."""

        self._control_world_impulse_sum = torch.zeros(
            (self.num_envs, self.num_taxels, 3),
            device=self.scene.device,
            dtype=torch.float32,
        )
        self._control_sim_step_count = 0
        self._control_accumulation_active = True
        self._control_average_ready = False
        self._cpu_control_profile = {
            "total_s": 0.0,
            "fetch_s": 0.0,
            "mapping_s": 0.0,
            "contact_count": 0,
            "tactile_point_count": 0,
            "max_tactile_points_per_sim_step": 0,
        }
        self._control_diagnostics = {
            "raw_total_impulse": torch.zeros(
                (self.num_envs, 3),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "raw_covered_impulse": torch.zeros(
                (self.num_envs, 3),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "mapped_impulse": torch.zeros(
                (self.num_envs, 3),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "raw_covered_torque_impulse": torch.zeros(
                (self.num_envs, 3),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "mapped_torque_impulse": torch.zeros(
                (self.num_envs, 3),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "covered_contact_count": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.int64,
            ),
            "total_contact_count": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.int64,
            ),
            "centroid_error_sum_m": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "centroid_error_max_m": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.float32,
            ),
        } if self._contact_diagnostics_enabled else None
        self._current_step_diagnostics = None
        self.invalidate()

    def accumulate_sim_step_impulses(self) -> None:
        """Add the just-completed PhysX step's net taxel impulses."""

        if not self._control_accumulation_active:
            raise RuntimeError(
                "begin_control_step_accumulation() must be called before "
                "accumulate_sim_step_impulses()."
            )
        impulse_sum = self._control_world_impulse_sum
        if impulse_sum is None:
            raise RuntimeError("Control-step impulse storage is not initialized.")

        # Each PhysX step produces a new contact state. Retain query handles,
        # but drop the previous step's cached tensors before reading impulses.
        self.invalidate()
        world_impulses = (
            self._read_gpu_net_world_impulses()
            if self.scene.gpu_sim_enabled
            else self._read_cpu_net_world_impulses()
        )
        expected_shape = (self.num_envs, self.num_taxels, 3)
        if tuple(world_impulses.shape) != expected_shape:
            raise ValueError(
                "Unexpected per-step taxel impulse shape: "
                f"expected {expected_shape}, got {tuple(world_impulses.shape)}."
            )
        control_diagnostics = self._control_diagnostics
        if self._contact_diagnostics_enabled and (
            self._current_step_diagnostics is None
            or control_diagnostics is None
        ):
            raise RuntimeError(
                "Tactile contact diagnostics were not initialized for the "
                "current control step."
            )
        # [B, N, 3] + [B, N, 3] -> [B, N, 3].
        impulse_sum.add_(world_impulses)
        if self._contact_diagnostics_enabled:
            step_diagnostics = self._current_step_diagnostics
            assert step_diagnostics is not None
            assert control_diagnostics is not None
            for name in (
                "raw_total_impulse",
                "raw_covered_impulse",
                "mapped_impulse",
                "raw_covered_torque_impulse",
                "mapped_torque_impulse",
                "centroid_error_sum_m",
            ):
                control_diagnostics[name].add_(step_diagnostics[name])
            for name in ("covered_contact_count", "total_contact_count"):
                control_diagnostics[name].add_(step_diagnostics[name])
            control_diagnostics["centroid_error_max_m"].copy_(
                torch.maximum(
                    control_diagnostics["centroid_error_max_m"],
                    step_diagnostics["centroid_error_max_m"],
                )
            )
        self._control_sim_step_count += 1

    def end_control_step_accumulation(
        self,
        *,
        expected_sim_steps: int,
    ) -> None:
        """Finalize the impulse sum used by the next net-force read."""

        if expected_sim_steps <= 0:
            raise ValueError("expected_sim_steps must be positive.")
        if not self._control_accumulation_active:
            raise RuntimeError("No tactile control-step accumulation is active.")
        if self._control_sim_step_count != expected_sim_steps:
            raise RuntimeError(
                "Tactile impulse accumulation step mismatch: "
                f"expected {expected_sim_steps}, got "
                f"{self._control_sim_step_count}."
            )
        self._control_accumulation_active = False
        self._control_average_ready = True
        self.invalidate()

    def get_control_contact_diagnostics(
        self,
    ) -> TactileContactDiagnostics | None:
        """Return the latest completed force/moment mapping check.

        If no completed control-period snapshot exists, return ``None``
        instead of raising or fabricating a diagnostic value.
        """

        diagnostics = self._control_diagnostics
        if (
            not self._control_average_ready
            or diagnostics is None
            or self._control_sim_step_count <= 0
        ):
            return None
        impulse_sum = self._control_world_impulse_sum
        if impulse_sum is None:
            return None
        control_dt_s = (
            self._control_sim_step_count * float(self.scene.px.timestep)
        )
        raw_force = diagnostics["raw_covered_impulse"][0] / control_dt_s
        raw_total_force = diagnostics["raw_total_impulse"][0] / control_dt_s
        mapped_force = diagnostics["mapped_impulse"][0] / control_dt_s
        raw_torque = diagnostics["raw_covered_torque_impulse"][0] / control_dt_s
        mapped_torque = diagnostics["mapped_torque_impulse"][0] / control_dt_s
        taxel_world_forces = impulse_sum / control_dt_s
        taxel_normal_forces = self._project_normal_forces(taxel_world_forces)
        raw_force_norm = float(torch.linalg.vector_norm(raw_force))
        raw_total_force_norm = float(torch.linalg.vector_norm(raw_total_force))
        mapped_force_norm = float(torch.linalg.vector_norm(mapped_force))
        raw_torque_norm = float(torch.linalg.vector_norm(raw_torque))
        mapped_torque_norm = float(torch.linalg.vector_norm(mapped_torque))
        force_error = float(
            torch.linalg.vector_norm(mapped_force - raw_force)
            / max(raw_force_norm, 1e-8)
        )
        covered_count = int(diagnostics["covered_contact_count"][0].item())
        total_count = int(diagnostics["total_contact_count"][0].item())
        centroid_mean = (
            float(diagnostics["centroid_error_sum_m"][0]) / covered_count
            if covered_count > 0
            else 0.0
        )
        return TactileContactDiagnostics(
            raw_total_force_n=raw_total_force_norm,
            raw_covered_force_n=raw_force_norm,
            mapped_force_n=mapped_force_norm,
            force_error=force_error,
            raw_covered_torque_nm=raw_torque_norm,
            mapped_torque_nm=mapped_torque_norm,
            torque_error_nm=float(torch.linalg.vector_norm(mapped_torque - raw_torque)),
            taxel_vector_force_n=mapped_force_norm,
            taxel_scalar_force_n=float(torch.sum(taxel_normal_forces[0])),
            taxel_force_norm_sum_n=float(
                torch.sum(torch.linalg.vector_norm(taxel_world_forces[0], dim=-1))
            ),
            covered_contact_count=covered_count,
            total_contact_count=total_count,
            coverage_ratio=(covered_count / total_count if total_count else 1.0),
            force_coverage_ratio=(
                raw_force_norm / raw_total_force_norm
                if raw_total_force_norm > 1e-8
                else 1.0
            ),
            centroid_error_mean_m=centroid_mean,
            centroid_error_max_m=float(diagnostics["centroid_error_max_m"][0]),
        )

    def _validated_bodies(self, managed_object: Any) -> tuple[Any, ...]:
        bodies = tuple(managed_object._bodies)
        if len(bodies) != self.num_envs:
            raise ValueError(
                f"Managed object {managed_object.name!r} has {len(bodies)} "
                f"bodies, expected one for each of {self.num_envs} environments."
            )
        return bodies

    def _cpu_contacts(self) -> list[Any]:
        if self._cached_cpu_contacts is None:
            self._cached_cpu_contacts = list(self.scene.px.get_contacts())
        return self._cached_cpu_contacts

    def current_cpu_contacts(self) -> tuple[Any, ...]:
        """Expose the shared current-state CPU contact pass to task diagnostics."""

        if self.scene.gpu_sim_enabled:
            raise RuntimeError("CPU contacts are unavailable in GPU simulation.")
        return tuple(self._cpu_contacts())

    @staticmethod
    def _contact_impulse(contact: Any) -> np.ndarray:
        impulse = np.zeros(3, dtype=np.float32)
        for contact_point in contact.points:
            impulse += np.asarray(contact_point.impulse, dtype=np.float32)
        return impulse

    def _reset_current_step_diagnostics(self) -> None:
        zero_vector = torch.zeros(
            (self.num_envs, 3),
            device=self.scene.device,
            dtype=torch.float32,
        )
        self._current_step_diagnostics = {
            "raw_total_impulse": zero_vector.clone(),
            "raw_covered_impulse": zero_vector.clone(),
            "mapped_impulse": zero_vector.clone(),
            "raw_covered_torque_impulse": zero_vector.clone(),
            "mapped_torque_impulse": zero_vector.clone(),
            "covered_contact_count": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.int64,
            ),
            "total_contact_count": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.int64,
            ),
            "centroid_error_sum_m": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.float32,
            ),
            "centroid_error_max_m": torch.zeros(
                (self.num_envs,),
                device=self.scene.device,
                dtype=torch.float32,
            ),
        }

    def _record_contact_mapping_diagnostics(
        self,
        *,
        contact_positions: torch.Tensor,
        parent_world_impulses: torch.Tensor,
        contact_env_indices: torch.Tensor,
        contact_parent_indices: torch.Tensor,
        distribution: TaxelImpulseDistribution,
    ) -> None:
        diagnostics = self._current_step_diagnostics
        if diagnostics is None:
            raise RuntimeError("Current tactile diagnostics are not initialized.")
        contact_count = contact_positions.shape[0]
        if contact_count == 0:
            return

        positions, _, _, _ = self._world_taxel_geometry()
        # Use one reference per environment so torques from multiple parent
        # links are comparable before they are accumulated.
        reference_positions = self.robot_links[0].pose.p
        references = reference_positions[contact_env_indices]
        contact_taxel_positions = positions[contact_env_indices]
        weights = distribution.contact_weights
        weighted_centers = torch.sum(
            weights[..., None] * contact_taxel_positions,
            dim=1,
        )
        centroid_errors = torch.linalg.vector_norm(
            weighted_centers - contact_positions,
            dim=-1,
        )
        raw_torque = torch.linalg.cross(
            contact_positions - references,
            parent_world_impulses,
            dim=-1,
        )
        mapped_taxel_impulses = (
            weights[..., None] * parent_world_impulses[:, None, :]
        )
        mapped_torque = torch.sum(
            torch.linalg.cross(
                contact_taxel_positions - references[:, None, :],
                mapped_taxel_impulses,
                dim=-1,
            ),
            dim=1,
        )
        covered = distribution.covered_contact_mask
        covered_env_indices = contact_env_indices[covered]
        diagnostics["raw_total_impulse"].index_add_(
            0,
            contact_env_indices,
            parent_world_impulses,
        )
        if bool(torch.any(covered)):
            diagnostics["raw_covered_impulse"].index_add_(
                0,
                covered_env_indices,
                parent_world_impulses[covered],
            )
            diagnostics["raw_covered_torque_impulse"].index_add_(
                0,
                covered_env_indices,
                raw_torque[covered],
            )
            diagnostics["centroid_error_sum_m"].index_add_(
                0,
                covered_env_indices,
                centroid_errors[covered],
            )
            for env_index in torch.unique(covered_env_indices).tolist():
                env_mask = covered & (contact_env_indices == env_index)
                diagnostics["centroid_error_max_m"][env_index] = torch.maximum(
                    diagnostics["centroid_error_max_m"][env_index],
                    centroid_errors[env_mask].max(),
                )
        diagnostics["mapped_impulse"].index_add_(
            0,
            contact_env_indices,
            mapped_taxel_impulses.sum(dim=1),
        )
        diagnostics["mapped_torque_impulse"].index_add_(
            0,
            contact_env_indices,
            mapped_torque,
        )
        diagnostics["covered_contact_count"].index_add_(
            0,
            covered_env_indices,
            torch.ones(
                (covered_env_indices.shape[0],),
                device=self.scene.device,
                dtype=torch.int64,
            ),
        )
        diagnostics["total_contact_count"].index_add_(
            0,
            contact_env_indices,
            torch.ones(
                (contact_env_indices.shape[0],),
                device=self.scene.device,
                dtype=torch.int64,
            ),
        )

    def _distribute_contact_points_with_diagnostics(
        self,
        *,
        contact_positions: torch.Tensor,
        parent_world_impulses: torch.Tensor,
        contact_env_indices: torch.Tensor,
        contact_parent_indices: torch.Tensor,
    ) -> TaxelImpulseDistribution:
        positions, normals, tangent_u, tangent_v = self._world_taxel_geometry()
        distribution = distribute_contact_impulses_to_taxels_with_diagnostics(
            contact_positions=contact_positions,
            parent_world_impulses=parent_world_impulses,
            contact_env_indices=contact_env_indices,
            contact_parent_indices=contact_parent_indices,
            taxel_world_positions=positions,
            taxel_world_normals=normals,
            taxel_world_tangent_u=tangent_u,
            taxel_world_tangent_v=tangent_v,
            taxel_parent_indices=self._taxel_parent_indices,
            taxel_half_sizes=self._taxel_half_sizes,
            assignment_margin_m=self.assignment_margin_m,
            normal_tolerance_m=self.normal_tolerance_m,
        )
        self._record_contact_mapping_diagnostics(
            contact_positions=contact_positions,
            parent_world_impulses=parent_world_impulses,
            contact_env_indices=contact_env_indices,
            contact_parent_indices=contact_parent_indices,
            distribution=distribution,
        )
        return distribution

    def _distribute_cpu_contact_point_batch(
        self,
        *,
        contact_positions: torch.Tensor,
        parent_world_impulses: torch.Tensor,
        contact_env_indices: torch.Tensor,
        contact_parent_indices: torch.Tensor,
    ) -> TaxelImpulseDistribution:
        """Map one CPU physics step's tactile points in a single tensor pass.

        CPU PhysX exposes contacts as Python wrappers.  Keep their extraction on
        the CPU, then cross the NumPy/Torch boundary once for the complete
        tactile-parent subset.  The returned contact weights retain per-point
        attribution for object queries without repeating the taxel scan.
        """

        positions, normals, tangent_u, tangent_v = self._world_taxel_geometry()
        contact_count = contact_positions.shape[0]
        weights = torch.zeros(
            (contact_count, self.num_taxels),
            device=self.scene.device,
            dtype=torch.float32,
        )
        covered = torch.zeros(
            (contact_count,),
            device=self.scene.device,
            dtype=torch.bool,
        )
        taxel_world_impulses = torch.zeros_like(positions)

        # A contact point can only activate taxels on its own physical parent.
        # Partition before geometric mapping so a fingertip contact scans that
        # fingertip's patches, rather than every taxel on both Shadow Hands.
        for parent_index in torch.unique(contact_parent_indices).tolist():
            contact_indices = torch.nonzero(
                contact_parent_indices == parent_index,
                as_tuple=False,
            )[:, 0]
            taxel_indices = torch.nonzero(
                self._taxel_parent_indices == parent_index,
                as_tuple=False,
            )[:, 0]
            if taxel_indices.numel() == 0:
                raise RuntimeError(
                    f"Tactile parent index {parent_index} has no logical taxels."
                )
            parent_distribution = (
                distribute_contact_impulses_to_taxels_with_diagnostics(
                    contact_positions=contact_positions[contact_indices],
                    parent_world_impulses=parent_world_impulses[contact_indices],
                    contact_env_indices=contact_env_indices[contact_indices],
                    contact_parent_indices=contact_parent_indices[contact_indices],
                    taxel_world_positions=positions[:, taxel_indices, :],
                    taxel_world_normals=normals[:, taxel_indices, :],
                    taxel_world_tangent_u=tangent_u[:, taxel_indices, :],
                    taxel_world_tangent_v=tangent_v[:, taxel_indices, :],
                    taxel_parent_indices=self._taxel_parent_indices[taxel_indices],
                    taxel_half_sizes=self._taxel_half_sizes[taxel_indices],
                    assignment_margin_m=self.assignment_margin_m,
                    normal_tolerance_m=self.normal_tolerance_m,
                )
            )
            weights[
                contact_indices[:, None],
                taxel_indices[None, :],
            ] = parent_distribution.contact_weights
            covered[contact_indices] = parent_distribution.covered_contact_mask
            taxel_world_impulses[:, taxel_indices, :] = (
                parent_distribution.taxel_world_impulses
            )

        distribution = TaxelImpulseDistribution(
            taxel_world_impulses=taxel_world_impulses,
            contact_weights=weights,
            covered_contact_mask=covered,
        )
        if self._contact_diagnostics_enabled:
            self._record_contact_mapping_diagnostics(
                contact_positions=contact_positions,
                parent_world_impulses=parent_world_impulses,
                contact_env_indices=contact_env_indices,
                contact_parent_indices=contact_parent_indices,
                distribution=distribution,
            )
        return distribution

    def _read_gpu_net_world_impulses(self) -> torch.Tensor:
        if self._contact_diagnostics_enabled:
            self._reset_current_step_diagnostics()
        if self._body_query is None:
            create_query = getattr(
                self.scene.px,
                "gpu_create_contact_point_body_query",
                None,
            )
            if create_query is None:
                raise RuntimeError(
                    "The installed SAPIEN build does not provide GPU "
                    "contact-point queries required by virtual taxels. "
                    "Run with sim_backend='cpu'."
                )
            # Parent-major, environment-minor order: [P][B] -> [P * B].
            parent_bodies = [
                body
                for parent_link in self._unique_parent_links
                for body in self._validated_bodies(parent_link)
            ]
            ignored_bodies = [
                body
                for robot_link in self.robot_links
                for body in self._validated_bodies(robot_link)
            ]
            self._body_query = create_query(
                parent_bodies,
                ignored_bodies,
                GPU_PARENT_CONTACT_POINTS_PER_ENV * self.num_envs,
            )
            self._body_query_builds += 1

        self.scene.px.gpu_query_contact_points(self._body_query)
        self._body_query_executions += 1
        contact_count = int(self._body_query.cuda_count.torch()[0].item())
        overflow = int(self._body_query.cuda_overflow.torch()[0].item())
        if overflow != 0:
            raise RuntimeError(
                "SAPIEN GPU contact-point query overflowed its fixed "
                f"capacity; increase {GPU_PARENT_CONTACT_POINTS_PER_ENV=}."
            )
        positions = self._body_query.cuda_positions.torch()[:contact_count].clone()
        impulses = self._body_query.cuda_impulses.torch()[:contact_count].clone()
        body_indices = (
            self._body_query.cuda_body_indices.torch()[:contact_count]
            .to(dtype=torch.int64)
            .clone()
        )
        # Query order is [P, B]. Recover the logical parent and environment.
        contact_parent_indices = torch.div(
            body_indices,
            self.num_envs,
            rounding_mode="floor",
        )
        contact_env_indices = body_indices.remainder(self.num_envs)
        scene_offsets = self.scene.scene_offsets[contact_env_indices]
        # PhysX contact points include per-environment scene offsets, while
        # ManiSkill link poses do not: [C, 3] - [C, 3] -> [C, 3].
        positions = positions - scene_offsets
        if self._contact_diagnostics_enabled:
            return self._distribute_contact_points_with_diagnostics(
                contact_positions=positions,
                parent_world_impulses=impulses,
                contact_env_indices=contact_env_indices,
                contact_parent_indices=contact_parent_indices,
            ).taxel_world_impulses
        return self._distribute_contact_points(
            contact_positions=positions,
            parent_world_impulses=impulses,
            contact_env_indices=contact_env_indices,
            contact_parent_indices=contact_parent_indices,
        )

    def _read_cpu_net_world_impulses(self) -> torch.Tensor:
        self._populate_cpu_contact_frame()
        world_impulses = self._cpu_net_world_impulses
        assert world_impulses is not None
        return torch.as_tensor(
            world_impulses,
            device=self.scene.device,
            dtype=torch.float32,
        )

    def _instantaneous_net_world_forces(self) -> torch.Tensor:
        world_impulses = (
            self._read_gpu_net_world_impulses()
            if self.scene.gpu_sim_enabled
            else self._read_cpu_net_world_impulses()
        )
        return world_impulses / float(self.scene.px.timestep)

    def _control_average_net_world_forces(self) -> torch.Tensor:
        impulse_sum = self._control_world_impulse_sum
        if impulse_sum is None or self._control_sim_step_count <= 0:
            raise RuntimeError(
                "Completed tactile control-step accumulation is empty."
            )
        control_dt_s = (
            self._control_sim_step_count * float(self.scene.px.timestep)
        )
        return impulse_sum / control_dt_s

    def _world_taxel_geometry(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._cached_world_positions is not None
            and self._cached_world_normals is not None
            and self._cached_world_tangent_u is not None
            and self._cached_world_tangent_v is not None
        ):
            return (
                self._cached_world_positions,
                self._cached_world_normals,
                self._cached_world_tangent_u,
                self._cached_world_tangent_v,
            )

        unique_parent_positions = torch.stack(
            [parent_link.pose.p for parent_link in self._unique_parent_links],
            dim=1,
        )
        unique_parent_quaternions = torch.stack(
            [parent_link.pose.q for parent_link in self._unique_parent_links],
            dim=1,
        )
        # [B, P, 4] -> [B, N, 4], where P is the unique parent count.
        quaternions = unique_parent_quaternions.index_select(
            dim=1,
            index=self._taxel_parent_indices,
        )
        # [B, P, 3] -> [B, N, 3].
        parent_positions = unique_parent_positions.index_select(
            dim=1,
            index=self._taxel_parent_indices,
        )
        expected_shape = (self.num_envs, self.num_taxels, 4)
        if tuple(quaternions.shape) != expected_shape:
            raise ValueError(
                "Unexpected taxel quaternion shape: "
                f"expected {expected_shape}, got {tuple(quaternions.shape)}."
            )

        def expand_local(local_vectors: torch.Tensor) -> torch.Tensor:
            # [N, 3] -> [B, N, 3].
            return local_vectors.unsqueeze(0).expand(self.num_envs, -1, -1)

        self._cached_world_positions = parent_positions + (
            rotate_vectors_by_wxyz_quaternions(
                quaternions,
                expand_local(self._local_positions),
            )
        )
        self._cached_world_normals = rotate_vectors_by_wxyz_quaternions(
            quaternions,
            expand_local(self._local_normals),
        )
        self._cached_world_tangent_u = rotate_vectors_by_wxyz_quaternions(
            quaternions,
            expand_local(self._local_tangent_u),
        )
        self._cached_world_tangent_v = rotate_vectors_by_wxyz_quaternions(
            quaternions,
            expand_local(self._local_tangent_v),
        )
        return (
            self._cached_world_positions,
            self._cached_world_normals,
            self._cached_world_tangent_u,
            self._cached_world_tangent_v,
        )

    def _world_normals(self) -> torch.Tensor:
        return self._world_taxel_geometry()[1]

    def get_taxel_world_positions(self) -> torch.Tensor:
        """Return logical taxel centers as ``[B, N, 3]``."""

        return self._world_taxel_geometry()[0]

    def _distribute_contact_points(
        self,
        *,
        contact_positions: torch.Tensor,
        parent_world_impulses: torch.Tensor,
        contact_env_indices: torch.Tensor,
        contact_parent_indices: torch.Tensor,
    ) -> torch.Tensor:
        positions, normals, tangent_u, tangent_v = (
            self._world_taxel_geometry()
        )
        return distribute_contact_impulses_to_taxels(
            contact_positions=contact_positions,
            parent_world_impulses=parent_world_impulses,
            contact_env_indices=contact_env_indices,
            contact_parent_indices=contact_parent_indices,
            taxel_world_positions=positions,
            taxel_world_normals=normals,
            taxel_world_tangent_u=tangent_u,
            taxel_world_tangent_v=tangent_v,
            taxel_parent_indices=self._taxel_parent_indices,
            taxel_half_sizes=self._taxel_half_sizes,
            assignment_margin_m=self.assignment_margin_m,
            normal_tolerance_m=self.normal_tolerance_m,
        )

    def _project_normal_forces(
        self,
        world_forces: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = (self.num_envs, self.num_taxels, 3)
        if tuple(world_forces.shape) != expected_shape:
            raise ValueError(
                f"Expected world taxel forces shaped {expected_shape}, got "
                f"{tuple(world_forces.shape)}."
            )
        world_normals = self._world_normals()
        # [B, N, 3] * [B, N, 3] -> [B, N].
        normal_forces = self.force_sign * torch.sum(
            world_forces * world_normals,
            dim=-1,
        )
        return normal_forces.clamp_min(0.0)

    @staticmethod
    def _apply_sensor_model(
        raw_normal_forces: torch.Tensor,
    ) -> torch.Tensor:
        """Identity hook reserved for a future calibrated sensor model."""

        return raw_normal_forces

    def get_net_taxel_normal_forces(self) -> torch.Tensor:
        """Return cached all-object normal forces as ``[B, N]``.

        After an environment control step, this is the sum of all PhysX-step
        impulses divided by the full control period. Outside that lifecycle it
        falls back to the current single-physics-step force.
        """

        if self._cached_net_normal_forces is None:
            world_forces = (
                self._control_average_net_world_forces()
                if self._control_average_ready
                else self._instantaneous_net_world_forces()
            )
            self._cached_net_normal_forces = self._apply_sensor_model(
                self._project_normal_forces(world_forces)
            )
        return self._cached_net_normal_forces

    def _current_net_world_forces(self) -> torch.Tensor:
        return (
            self._control_average_net_world_forces()
            if self._control_average_ready
            else self._instantaneous_net_world_forces()
        )

    def get_contact_state(self) -> TactileContactState:
        """Return parent-link, finger, and whole-hand resultant forces.

        The returned vectors are world-frame forces in Newtons. Taxel vectors
        are first summed by their rigid parent link, then by anatomical finger;
        no individual taxel threshold is used by the contact controller.
        """

        world_forces = self._current_net_world_forces()
        normal_forces = self._project_normal_forces(world_forces)
        world_normals = self._world_normals()
        # [B, N] -> [B, N, 1], then restore the signed world direction of the
        # contact reaction represented by each taxel's outward normal.
        normal_force_vectors = (
            self.force_sign * normal_forces.unsqueeze(-1) * world_normals
        )
        if self._cached_parent_force_world is None:
            self._cached_parent_force_world = self.aggregate_taxel_world_forces_by_parent(
                world_forces
            )
        if self._cached_parent_normal_force_world is None:
            self._cached_parent_normal_force_world = (
                self.aggregate_taxel_world_forces_by_parent(normal_force_vectors)
            )
        if self._cached_finger_force_world is None:
            finger_count = 5
            finger_forces = torch.zeros(
                (self.num_envs, finger_count, 3),
                device=world_forces.device,
                dtype=world_forces.dtype,
            )
            parent_finger_indices = self._parent_finger_indices.to(
                device=world_forces.device
            )
            for finger_index in range(finger_count):
                selected = parent_finger_indices == finger_index
                if bool(torch.any(selected)):
                    finger_forces[:, finger_index, :] = self._cached_parent_force_world[
                        :, selected, :
                    ].sum(dim=1)
            self._cached_finger_force_world = finger_forces
        if self._cached_hand_force_world is None:
            self._cached_hand_force_world = self._cached_parent_force_world.sum(dim=1)
        return TactileContactState(
            parent_force_world=self._cached_parent_force_world,
            parent_normal_force_world=self._cached_parent_normal_force_world,
            finger_force_world=self._cached_finger_force_world,
            taxel_force_world=world_forces,
            hand_force_world=self._cached_hand_force_world,
            parent_link_names=tuple(link.name for link in self._unique_parent_links),
            finger_names=("th", "ff", "mf", "rf", "lf"),
            parent_finger_indices=tuple(int(value) for value in self._parent_finger_indices.tolist()),
        )

    def _normalize_target_groups(
        self,
        target_groups: Sequence[tuple[str, Sequence[Any]]],
    ) -> tuple[
        tuple[Any, ...],
        tuple[tuple[str, slice], ...],
        tuple[object, ...],
    ]:
        if not target_groups:
            raise ValueError("At least one tactile target group is required.")
        names: set[str] = set()
        targets: list[Any] = []
        slices: list[tuple[str, slice]] = []
        signature_parts: list[object] = []
        for group_name, group_targets in target_groups:
            if not group_name or group_name in names:
                raise ValueError(
                    f"Tactile target group names must be unique and non-empty, "
                    f"got {group_name!r}."
                )
            names.add(group_name)
            target_start = len(targets)
            normalized_targets = tuple(group_targets)
            if not normalized_targets:
                raise ValueError(
                    f"Tactile target group {group_name!r} must not be empty."
                )
            targets.extend(normalized_targets)
            slices.append((group_name, slice(target_start, len(targets))))
            signature_parts.extend((group_name, len(normalized_targets)))
            for target in normalized_targets:
                bodies = self._validated_bodies(target)
                signature_parts.append(
                    tuple(id(body) for body in bodies)
                )
        return (
            tuple(targets),
            tuple(slices),
            tuple(signature_parts),
        )

    def prepare_cpu_object_query(
        self,
        target_groups: Sequence[tuple[str, Sequence[Any]]],
    ) -> None:
        """Build persistent CPU entity bindings before the first state query."""

        if self.scene.gpu_sim_enabled:
            return
        targets, target_group_slices, signature = self._normalize_target_groups(
            target_groups
        )
        self._prepare_cpu_object_query(
            targets=targets,
            target_group_slices=target_group_slices,
            signature=signature,
        )

    def _prepare_cpu_object_query(
        self,
        *,
        targets: tuple[Any, ...],
        target_group_slices: tuple[tuple[str, slice], ...],
        signature: tuple[object, ...],
    ) -> None:
        if signature in self._cpu_object_queries:
            return

        target_entities: dict[Any, tuple[int, int]] = {}
        for target_index, target in enumerate(targets):
            for env_index, body in enumerate(self._validated_bodies(target)):
                entity = body.entity
                if entity in target_entities:
                    raise ValueError(
                        "A PhysX entity cannot appear twice in one tactile "
                        "object query."
                    )
                target_entities[entity] = (env_index, target_index)
        self._cpu_object_queries[signature] = _CpuObjectQuery(
            target_count=len(targets),
            target_group_slices=target_group_slices,
            target_entities=target_entities,
        )
        if self._cpu_contact_frame_populated:
            # A late registration remains correct, but callers should prepare all
            # queries after scene load to retain the one-traversal invariant.
            self._cpu_contact_frame_populated = False

    @staticmethod
    def _pair_query_signature(
        first_links: Sequence[Any],
        second_links: Sequence[Any],
    ) -> tuple[object, ...]:
        return (
            "link_pair_query",
            tuple(id(link) for link in first_links),
            tuple(id(link) for link in second_links),
        )

    @staticmethod
    def _normalize_pairwise_links(
        first_links: Sequence[Any],
        second_links: Sequence[Any],
    ) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        normalized_first_links = tuple(first_links)
        normalized_second_links = tuple(second_links)
        if not normalized_first_links or not normalized_second_links:
            raise ValueError(
                "Pairwise contact queries require two non-empty link groups."
            )
        return normalized_first_links, normalized_second_links

    def _prepare_cpu_pairwise_contact_query(
        self,
        *,
        first_links: tuple[Any, ...],
        second_links: tuple[Any, ...],
        signature: tuple[object, ...],
    ) -> None:
        if signature in self._cpu_pair_queries:
            return

        first_entities: dict[Any, tuple[int, int]] = {}
        for link_index, link in enumerate(first_links):
            for env_index, body in enumerate(self._validated_bodies(link)):
                entity = body.entity
                if entity in first_entities:
                    raise ValueError(
                        "A PhysX entity cannot appear twice in the first "
                        "pair-query group."
                    )
                first_entities[entity] = (env_index, link_index)
        second_entities: dict[Any, tuple[int, int]] = {}
        for link_index, link in enumerate(second_links):
            for env_index, body in enumerate(self._validated_bodies(link)):
                entity = body.entity
                if entity in second_entities:
                    raise ValueError(
                        "A PhysX entity cannot appear twice in the second "
                        "pair-query group."
                    )
                second_entities[entity] = (env_index, link_index)
        self._cpu_pair_queries[signature] = _CpuPairQuery(
            first_count=len(first_links),
            second_count=len(second_links),
            first_entities=first_entities,
            second_entities=second_entities,
        )
        for first_entity, (first_env, first_link) in first_entities.items():
            for second_entity, (second_env, second_link) in second_entities.items():
                if first_env != second_env:
                    continue
                forward = (signature, first_env, first_link, second_link, 1.0)
                reverse = (signature, first_env, first_link, second_link, -1.0)
                self._cpu_pair_routes.setdefault(
                    (first_entity, second_entity), []
                ).append(forward)
                self._cpu_pair_routes.setdefault(
                    (second_entity, first_entity), []
                ).append(reverse)
        if self._cpu_contact_frame_populated or self._cpu_pair_frame_populated:
            self._cpu_contact_frame_populated = False
            self._cpu_pair_frame_populated = False

    def _populate_cpu_pair_frame(self) -> None:
        """Aggregate registered link-pair forces without taxel projection."""

        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "CPU pairwise contacts are unavailable in GPU simulation."
            )
        if self._cpu_pair_frame_populated:
            return
        for query in self._cpu_pair_queries.values():
            query.pair_impulses = np.zeros(
                (
                    self.num_envs,
                    query.first_count,
                    query.second_count,
                    3,
                ),
                dtype=np.float32,
            )
        for contact in self._cpu_contacts():
            body0_entity = contact.bodies[0].entity
            body1_entity = contact.bodies[1].entity
            routes = self._cpu_pair_routes.get(
                (body0_entity, body1_entity), ()
            )
            for signature, env_index, first_link, second_link, direction in routes:
                query = self._cpu_pair_queries[signature]
                assert query.pair_impulses is not None
                query.pair_impulses[
                    env_index,
                    first_link,
                    second_link,
                ] += direction * self._contact_impulse(contact)
        self._cpu_pair_frame_populated = True

    def _prepare_gpu_pairwise_contact_query(
        self,
        *,
        first_links: tuple[Any, ...],
        second_links: tuple[Any, ...],
        signature: tuple[object, ...],
    ) -> None:
        if signature in self._gpu_pair_queries:
            return

        self._gpu_pair_queries[signature] = _GpuPairQuery(
            first_links=first_links,
            second_links=second_links,
            first_count=len(first_links),
            second_count=len(second_links),
        )

    def prepare_pairwise_contact_query(
        self,
        first_links: Sequence[Any],
        second_links: Sequence[Any],
    ) -> PairwiseContactQueryHandle:
        """Build a persistent link-pair query for the active backend."""

        normalized_first_links, normalized_second_links = (
            self._normalize_pairwise_links(first_links, second_links)
        )
        signature = self._pair_query_signature(
            normalized_first_links,
            normalized_second_links,
        )
        if self.scene.gpu_sim_enabled:
            self._prepare_gpu_pairwise_contact_query(
                first_links=normalized_first_links,
                second_links=normalized_second_links,
                signature=signature,
            )
        else:
            self._prepare_cpu_pairwise_contact_query(
                first_links=normalized_first_links,
                second_links=normalized_second_links,
                signature=signature,
            )
        return PairwiseContactQueryHandle(signature=signature)

    def prepare_cpu_pairwise_contact_query(
        self,
        first_links: Sequence[Any],
        second_links: Sequence[Any],
    ) -> CpuPairwiseContactQueryHandle:
        """Build a persistent CPU link-pair binding used by the shared pass."""

        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "CPU pairwise contact queries are unavailable in GPU simulation."
            )
        normalized_first_links, normalized_second_links = (
            self._normalize_pairwise_links(first_links, second_links)
        )
        signature = self._pair_query_signature(
            normalized_first_links,
            normalized_second_links,
        )
        self._prepare_cpu_pairwise_contact_query(
            first_links=normalized_first_links,
            second_links=normalized_second_links,
            signature=signature,
        )
        return CpuPairwiseContactQueryHandle(signature=signature)

    def _populate_cpu_contact_frame(self) -> None:
        """Aggregate every registered CPU contact consumer in one list pass."""

        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "The CPU contact frame is unavailable in GPU simulation."
            )
        if self._cpu_contact_frame_populated:
            return

        profile_started_s = time.perf_counter()
        if self._contact_diagnostics_enabled:
            self._reset_current_step_diagnostics()

        # Contact impulses: [B, N, 3].
        self._cpu_net_world_impulses = np.zeros(
            (self.num_envs, self.num_taxels, 3),
            dtype=np.float32,
        )
        for query in self._cpu_object_queries.values():
            # [B, N, M, 3].
            query.world_impulses = np.zeros(
                (
                    self.num_envs,
                    self.num_taxels,
                    query.target_count,
                    3,
                ),
                dtype=np.float32,
            )
        populate_pair_queries = not self._cpu_pair_frame_populated
        if populate_pair_queries:
            for query in self._cpu_pair_queries.values():
                # [B, R, D, 3].
                query.pair_impulses = np.zeros(
                    (
                        self.num_envs,
                        query.first_count,
                        query.second_count,
                        3,
                    ),
                    dtype=np.float32,
                )

        # First retain only contact points involving a tactile parent. Pairwise
        # consumers share this traversal only when they have not already used
        # the cheaper pair-only path. Scene contacts unrelated to tactile
        # parents never enter the 123/246-taxel mapping path.
        tactile_positions: list[np.ndarray] = []
        tactile_impulses: list[np.ndarray] = []
        tactile_env_indices: list[np.ndarray] = []
        tactile_parent_indices: list[np.ndarray] = []
        tactile_other_entities: list[Any] = []

        fetch_started_s = time.perf_counter()
        contacts = self._cpu_contacts()
        fetch_elapsed_s = time.perf_counter() - fetch_started_s
        for contact in contacts:
            body0_entity = contact.bodies[0].entity
            body1_entity = contact.bodies[1].entity
            body0_parent = self._parent_entities.get(body0_entity)
            body1_parent = self._parent_entities.get(body1_entity)

            parent_contact: tuple[int, int, Any, float] | None = None
            if body0_parent is not None and body1_entity not in self._robot_entities:
                parent_contact = (*body0_parent, body1_entity, 1.0)
            elif (
                body1_parent is not None
                and body0_entity not in self._robot_entities
            ):
                parent_contact = (*body1_parent, body0_entity, -1.0)

            if parent_contact is not None and contact.points:
                env_index, parent_index, other_entity, direction = parent_contact
                point_count = len(contact.points)
                tactile_positions.append(
                    np.asarray(
                        [point.position for point in contact.points], dtype=np.float32
                    )
                )
                tactile_impulses.append(
                    direction
                    * np.asarray(
                        [point.impulse for point in contact.points],
                        dtype=np.float32,
                    )
                )
                tactile_env_indices.append(
                    np.full((point_count,), env_index, dtype=np.int64)
                )
                tactile_parent_indices.append(
                    np.full((point_count,), parent_index, dtype=np.int64)
                )
                tactile_other_entities.extend((other_entity,) * point_count)

            if populate_pair_queries:
                routes = self._cpu_pair_routes.get(
                    (body0_entity, body1_entity), ()
                )
                for signature, env_index, first_link, second_link, direction in routes:
                    query = self._cpu_pair_queries[signature]
                    assert query.pair_impulses is not None
                    query.pair_impulses[
                        env_index,
                        first_link,
                        second_link,
                    ] += direction * self._contact_impulse(contact)

        tactile_point_count = 0
        mapping_elapsed_s = 0.0
        if tactile_positions:
            mapping_started_s = time.perf_counter()
            positions_array = np.concatenate(tactile_positions, axis=0)
            impulses_array = np.concatenate(tactile_impulses, axis=0)
            env_indices_array = np.concatenate(tactile_env_indices, axis=0)
            parent_indices_array = np.concatenate(tactile_parent_indices, axis=0)
            contact_positions = torch.as_tensor(
                positions_array,
                device=self.scene.device,
                dtype=torch.float32,
            )
            parent_world_impulses = torch.as_tensor(
                impulses_array,
                device=self.scene.device,
                dtype=torch.float32,
            )
            contact_env_indices = torch.as_tensor(
                env_indices_array,
                device=self.scene.device,
                dtype=torch.int64,
            )
            contact_parent_indices = torch.as_tensor(
                parent_indices_array,
                device=self.scene.device,
                dtype=torch.int64,
            )
            distribution = self._distribute_cpu_contact_point_batch(
                contact_positions=contact_positions,
                parent_world_impulses=parent_world_impulses,
                contact_env_indices=contact_env_indices,
                contact_parent_indices=contact_parent_indices,
            )

            # Preserve per-contact object attribution using the weights already
            # computed by the batched map.  This is the only Torch -> NumPy
            # transfer for all tactile points in the physics substep.
            per_point_impulses = (
                distribution.contact_weights[..., None]
                * parent_world_impulses[:, None, :]
            ).detach().cpu().numpy()
            assert self._cpu_net_world_impulses is not None
            np.add.at(
                self._cpu_net_world_impulses,
                env_indices_array,
                per_point_impulses,
            )

            for query in self._cpu_object_queries.values():
                matched_point_indices: list[int] = []
                matched_env_indices: list[int] = []
                matched_target_indices: list[int] = []
                for point_index, (env_index, other_entity) in enumerate(
                    zip(
                        env_indices_array,
                        tactile_other_entities,
                        strict=True,
                    )
                ):
                    target = query.target_entities.get(other_entity)
                    if target is None or target[0] != int(env_index):
                        continue
                    matched_point_indices.append(point_index)
                    matched_env_indices.append(int(env_index))
                    matched_target_indices.append(target[1])
                if not matched_point_indices:
                    continue
                assert query.world_impulses is not None
                np.add.at(
                    query.world_impulses,
                    (
                        np.asarray(matched_env_indices, dtype=np.int64),
                        slice(None),
                        np.asarray(matched_target_indices, dtype=np.int64),
                        slice(None),
                    ),
                    per_point_impulses[
                        np.asarray(matched_point_indices, dtype=np.int64)
                    ],
                )
            tactile_point_count = int(positions_array.shape[0])
            mapping_elapsed_s = time.perf_counter() - mapping_started_s

        self._cpu_contact_frame_populated = True
        self._cpu_pair_frame_populated = True
        profile = self._cpu_control_profile
        if profile:
            profile["total_s"] = float(profile["total_s"]) + (
                time.perf_counter() - profile_started_s
            )
            profile["fetch_s"] = float(profile["fetch_s"]) + fetch_elapsed_s
            profile["mapping_s"] = (
                float(profile["mapping_s"]) + mapping_elapsed_s
            )
            profile["contact_count"] = int(profile["contact_count"]) + len(contacts)
            profile["tactile_point_count"] = (
                int(profile["tactile_point_count"]) + tactile_point_count
            )
            profile["max_tactile_points_per_sim_step"] = max(
                int(profile["max_tactile_points_per_sim_step"]),
                tactile_point_count,
            )

    def get_cpu_control_step_profile(self) -> Mapping[str, float | int]:
        """Return low-overhead timing/counts for the latest CPU control step."""

        if self.scene.gpu_sim_enabled:
            return {}
        return dict(self._cpu_control_profile)

    def get_cpu_pairwise_contact_forces(
        self,
        query_handle: CpuPairwiseContactQueryHandle,
    ) -> torch.Tensor:
        """Return one prepared CPU query as [B, R, D, 3] forces."""

        if self.scene.gpu_sim_enabled:
            raise RuntimeError(
                "CPU pairwise contact forces are unavailable in GPU simulation."
            )
        try:
            query = self._cpu_pair_queries[query_handle.signature]
        except KeyError as exc:
            raise KeyError("Unknown CPU pairwise contact query handle.") from exc
        self._populate_cpu_pair_frame()
        pair_impulses = query.pair_impulses
        assert pair_impulses is not None
        return torch.as_tensor(
            pair_impulses,
            device=self.scene.device,
            dtype=torch.float32,
        ) / float(self.scene.px.timestep)

    def get_pairwise_contact_forces(
        self,
        query_handle: PairwiseContactQueryHandle,
    ) -> torch.Tensor:
        """Return one prepared query as [B, R, D, 3] forces."""

        if not self.scene.gpu_sim_enabled:
            return self.get_cpu_pairwise_contact_forces(
                CpuPairwiseContactQueryHandle(
                    signature=query_handle.signature
                )
            )

        try:
            query = self._gpu_pair_queries[query_handle.signature]
        except KeyError as exc:
            raise KeyError("Unknown GPU pairwise contact query handle.") from exc
        if query.handle is None:
            # Query registration may happen during _load_scene(), before ManiSkill
            # calls gpu_init(). Build the PhysX handle lazily on the first state
            # query, in [R][D][B] order matching the reshape below.
            body_pairs = [
                (first_body, second_body)
                for first_link in query.first_links
                for second_link in query.second_links
                for first_body, second_body in zip(
                    self._validated_bodies(first_link),
                    self._validated_bodies(second_link),
                    strict=True,
                )
            ]
            query.handle = (
                self.scene.px.gpu_create_contact_pair_impulse_query(
                    body_pairs
                )
            )
            self._pair_query_builds += 1
        if query.cached_forces is None:
            self.scene.px.gpu_query_contact_pair_impulses(query.handle)
            self._pair_query_executions += 1
            impulses = query.handle.cuda_impulses.torch().clone()
            expected_shape = (
                query.first_count
                * query.second_count
                * self.num_envs,
                3,
            )
            if tuple(impulses.shape) != expected_shape:
                raise ValueError(
                    "Unexpected GPU link-pair query shape: "
                    f"expected {expected_shape}, got {tuple(impulses.shape)}."
                )
            # [R * D * B, 3] -> [R, D, B, 3] -> [B, R, D, 3].
            query.cached_forces = (
                impulses.reshape(
                    query.first_count,
                    query.second_count,
                    self.num_envs,
                    3,
                ).permute(2, 0, 1, 3)
                / float(self.scene.px.timestep)
            )
        return query.cached_forces

    def _gpu_object_world_forces(
        self,
        *,
        targets: tuple[Any, ...],
        target_group_slices: tuple[tuple[str, slice], ...],
        signature: tuple[object, ...],
    ) -> torch.Tensor:
        query = self._pair_queries.get(signature)
        if query is None:
            query = _PairQuery(
                handle=None,
                target_count=len(targets),
                target_group_slices=target_group_slices,
            )
            self._pair_queries[signature] = query
        elif query.target_group_slices != target_group_slices:
            raise RuntimeError(
                "Tactile pair-query signature collided with different groups."
            )

        if query.handle is None:
            create_query = getattr(
                self.scene.px,
                "gpu_create_contact_point_pair_query",
                None,
            )
            if create_query is None:
                raise RuntimeError(
                    "The installed SAPIEN build does not provide GPU "
                    "contact-point pair queries required by virtual taxels. "
                    "Run with sim_backend='cpu'."
                )
            # Pair order: [parent, target, environment].
            body_pairs = [
                (parent_body, target_body)
                for parent_link in self._unique_parent_links
                for target in targets
                for parent_body, target_body in zip(
                    self._validated_bodies(parent_link),
                    self._validated_bodies(target),
                    strict=True,
                )
            ]
            query.handle = create_query(
                body_pairs,
                GPU_PARENT_CONTACT_POINTS_PER_ENV * self.num_envs,
            )
            self._pair_query_builds += 1

        self.scene.px.gpu_query_contact_points(query.handle)
        self._pair_query_executions += 1
        contact_count = int(query.handle.cuda_count.torch()[0].item())
        overflow = int(query.handle.cuda_overflow.torch()[0].item())
        if overflow != 0:
            raise RuntimeError(
                "SAPIEN GPU contact-point pair query overflowed its fixed "
                f"capacity; increase {GPU_PARENT_CONTACT_POINTS_PER_ENV=}."
            )
        positions = query.handle.cuda_positions.torch()[:contact_count].clone()
        impulses = query.handle.cuda_impulses.torch()[:contact_count].clone()
        pair_indices = (
            query.handle.cuda_pair_indices.torch()[:contact_count]
            .to(dtype=torch.int64)
            .clone()
        )
        target_env_stride = query.target_count * self.num_envs
        contact_parent_indices = torch.div(
            pair_indices,
            target_env_stride,
            rounding_mode="floor",
        )
        pair_remainder = pair_indices.remainder(target_env_stride)
        contact_target_indices = torch.div(
            pair_remainder,
            self.num_envs,
            rounding_mode="floor",
        )
        contact_env_indices = pair_remainder.remainder(self.num_envs)
        positions = positions - self.scene.scene_offsets[contact_env_indices]

        target_impulses: list[torch.Tensor] = []
        for target_index in range(query.target_count):
            target_mask = contact_target_indices == target_index
            target_impulses.append(
                self._distribute_contact_points(
                    contact_positions=positions[target_mask],
                    parent_world_impulses=impulses[target_mask],
                    contact_env_indices=contact_env_indices[target_mask],
                    contact_parent_indices=contact_parent_indices[target_mask],
                )
            )
        # M * [B, N, 3] -> [B, N, M, 3].
        return torch.stack(target_impulses, dim=2) / float(
            self.scene.px.timestep
        )

    def _cpu_object_world_forces(
        self,
        *,
        signature: tuple[object, ...],
    ) -> torch.Tensor:
        query = self._cpu_object_queries[signature]
        self._populate_cpu_contact_frame()
        world_impulses = query.world_impulses
        assert world_impulses is not None
        return torch.as_tensor(
            world_impulses,
            device=self.scene.device,
            dtype=torch.float32,
        ) / float(self.scene.px.timestep)

    def get_object_taxel_world_forces(
        self,
        target_groups: Sequence[tuple[str, Sequence[Any]]],
    ) -> Mapping[str, torch.Tensor]:
        """Return target-attributed world forces, one cached [B, N, 3] per group."""

        targets, target_group_slices, signature = self._normalize_target_groups(
            target_groups
        )
        if not self.scene.gpu_sim_enabled:
            self._prepare_cpu_object_query(
                targets=targets,
                target_group_slices=target_group_slices,
                signature=signature,
            )
        cached_result = self._cached_object_world_forces.get(signature)
        if cached_result is not None:
            return cached_result

        target_world_forces = (
            self._gpu_object_world_forces(
                targets=targets,
                target_group_slices=target_group_slices,
                signature=signature,
            )
            if self.scene.gpu_sim_enabled
            else self._cpu_object_world_forces(signature=signature)
        )
        expected_shape = (
            self.num_envs,
            self.num_taxels,
            len(targets),
            3,
        )
        if tuple(target_world_forces.shape) != expected_shape:
            raise ValueError(
                f"Expected object taxel forces shaped {expected_shape}, got "
                f"{tuple(target_world_forces.shape)}."
            )

        result: dict[str, torch.Tensor] = {}
        for group_name, target_slice in target_group_slices:
            # [B, N, M_group, 3] -> [B, N, 3].
            result[group_name] = target_world_forces[
                :,
                :,
                target_slice,
                :,
            ].sum(dim=2)
        self._cached_object_world_forces[signature] = result
        return result

    def get_object_taxel_normal_forces(
        self,
        target_groups: Sequence[tuple[str, Sequence[Any]]],
    ) -> Mapping[str, torch.Tensor]:
        """Return target-attributed normal forces, one cached [B, N] per group."""

        _, _, signature = self._normalize_target_groups(target_groups)
        cached_result = self._cached_object_normal_forces.get(signature)
        if cached_result is not None:
            return cached_result

        object_world_forces = self.get_object_taxel_world_forces(target_groups)
        result = {
            group_name: self._apply_sensor_model(
                self._project_normal_forces(group_world_forces)
            )
            for group_name, group_world_forces in object_world_forces.items()
        }
        self._cached_object_normal_forces[signature] = result
        return result

    def aggregate_taxel_world_forces_by_parent(
        self,
        taxel_world_forces: torch.Tensor,
    ) -> torch.Tensor:
        """Sum taxel force vectors on each physical parent link.

        Adjacent taxels on the same rigid link are samples of one physical
        contact surface. Summing their vectors before taking a magnitude
        preserves the former per-link resultant-force semantics.
        """

        expected_shape = (self.num_envs, self.num_taxels, 3)
        if tuple(taxel_world_forces.shape) != expected_shape:
            raise ValueError(
                "Expected taxel world forces shaped "
                f"{expected_shape}, got {tuple(taxel_world_forces.shape)}."
            )

        parent_count = len(self._unique_parent_links)
        # [B, N, 3] -> [B, P, 3] by rigid taxel parent index.
        parent_world_forces = torch.zeros(
            (self.num_envs, parent_count, 3),
            device=taxel_world_forces.device,
            dtype=taxel_world_forces.dtype,
        )
        parent_indices = self._taxel_parent_indices.reshape(1, -1, 1).expand(
            self.num_envs,
            -1,
            3,
        )
        parent_world_forces.scatter_add_(
            dim=1,
            index=parent_indices,
            src=taxel_world_forces,
        )
        return parent_world_forces


def aggregate_distal_taxels_by_finger(
    taxel_normal_forces: torch.Tensor,
    *,
    finger_indices: torch.Tensor,
    distal_mask: torch.Tensor,
    finger_count: int,
) -> torch.Tensor:
    """Temporarily aggregate distal taxel scalars into anatomical fingers."""

    if taxel_normal_forces.ndim != 2:
        raise ValueError(
            "Taxel normal forces must have shape [B, N], got "
            f"{tuple(taxel_normal_forces.shape)}."
        )
    taxel_count = taxel_normal_forces.shape[1]
    if tuple(finger_indices.shape) != (taxel_count,):
        raise ValueError(
            f"finger_indices must have shape ({taxel_count},), got "
            f"{tuple(finger_indices.shape)}."
        )
    if tuple(distal_mask.shape) != (taxel_count,):
        raise ValueError(
            f"distal_mask must have shape ({taxel_count},), got "
            f"{tuple(distal_mask.shape)}."
        )
    if finger_count <= 0:
        raise ValueError(f"finger_count must be positive, got {finger_count}.")

    selected_forces = taxel_normal_forces[:, distal_mask]
    selected_finger_indices = finger_indices[distal_mask]
    if bool(torch.any(selected_finger_indices < 0)) or bool(
        torch.any(selected_finger_indices >= finger_count)
    ):
        raise ValueError("Distal taxels contain an invalid finger index.")

    # [B, N_distal] -> [B, F] by semantic finger index.
    finger_forces = torch.zeros(
        (taxel_normal_forces.shape[0], finger_count),
        device=taxel_normal_forces.device,
        dtype=taxel_normal_forces.dtype,
    )
    finger_forces.scatter_add_(
        dim=1,
        index=selected_finger_indices.unsqueeze(0).expand(
            taxel_normal_forces.shape[0],
            -1,
        ),
        src=selected_forces,
    )
    return finger_forces
