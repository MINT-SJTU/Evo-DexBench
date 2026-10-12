from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaxelImpulseDistribution:
    """Per-contact mapping result used by tactile conservation diagnostics."""

    taxel_world_impulses: torch.Tensor
    contact_weights: torch.Tensor
    covered_contact_mask: torch.Tensor


def rotate_vectors_by_wxyz_quaternions(
    quaternions: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Rotate matching 3D vectors by ``wxyz`` quaternions.

    Both inputs use broadcast-compatible leading dimensions. The final shapes
    are ``[..., 4]`` and ``[..., 3]`` respectively.
    """

    if quaternions.shape[-1] != 4:
        raise ValueError(
            "Quaternion input must end in dimension 4, got "
            f"{tuple(quaternions.shape)}."
        )
    if vectors.shape[-1] != 3:
        raise ValueError(
            "Vector input must end in dimension 3, got "
            f"{tuple(vectors.shape)}."
        )
    quaternion_vector = quaternions[..., 1:]
    quaternion_scalar = quaternions[..., :1]
    doubled_cross = 2.0 * torch.linalg.cross(
        quaternion_vector,
        vectors,
        dim=-1,
    )
    return (
        vectors
        + quaternion_scalar * doubled_cross
        + torch.linalg.cross(quaternion_vector, doubled_cross, dim=-1)
    )


def transform_world_points_to_local(
    *,
    frame_positions: torch.Tensor,
    frame_quaternions: torch.Tensor,
    world_points: torch.Tensor,
) -> torch.Tensor:
    """Transform ``[B, N, 3]`` world points into batched local frames."""

    if world_points.ndim != 3 or world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [B, N, 3].")
    batch_size, point_count, _ = world_points.shape
    if tuple(frame_positions.shape) != (batch_size, 3):
        raise ValueError("frame_positions must have shape [B, 3].")
    if tuple(frame_quaternions.shape) != (batch_size, 4):
        raise ValueError("frame_quaternions must have shape [B, 4].")
    inverse_quaternions = frame_quaternions.clone()
    inverse_quaternions[:, 1:] = -inverse_quaternions[:, 1:]
    # [B, N, 3] - [B, 1, 3] -> [B, N, 3].
    position_delta = world_points - frame_positions[:, None, :]
    return rotate_vectors_by_wxyz_quaternions(
        inverse_quaternions[:, None, :].expand(-1, point_count, -1),
        position_delta,
    )


def _distribute_contact_impulses(
    *,
    contact_positions: torch.Tensor,
    parent_world_impulses: torch.Tensor,
    contact_env_indices: torch.Tensor,
    contact_parent_indices: torch.Tensor,
    taxel_world_positions: torch.Tensor,
    taxel_world_normals: torch.Tensor,
    taxel_world_tangent_u: torch.Tensor,
    taxel_world_tangent_v: torch.Tensor,
    taxel_parent_indices: torch.Tensor,
    taxel_half_sizes: torch.Tensor,
    assignment_margin_m: float,
    normal_tolerance_m: float,
    include_diagnostics: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Distribute parent-link contact impulses onto covered logical taxels.

    A contact can only activate taxels on the same parent link. Valid taxels
    are selected by their finite rectangular surface patch and normal-distance
    tolerance. Compact spatial weights are normalized per contact, so every
    covered contact preserves its complete vector impulse without duplication.
    Uncovered contacts intentionally produce no tactile signal.
    """

    if assignment_margin_m < 0.0:
        raise ValueError("assignment_margin_m must be non-negative.")
    if normal_tolerance_m <= 0.0:
        raise ValueError("normal_tolerance_m must be positive.")
    if contact_positions.ndim != 2 or contact_positions.shape[1] != 3:
        raise ValueError(
            "contact_positions must have shape [C, 3], got "
            f"{tuple(contact_positions.shape)}."
        )
    if parent_world_impulses.shape != contact_positions.shape:
        raise ValueError(
            "parent_world_impulses must match contact_positions, got "
            f"{tuple(parent_world_impulses.shape)} and "
            f"{tuple(contact_positions.shape)}."
        )

    env_count, taxel_count, coordinate_count = taxel_world_positions.shape
    expected_world_shape = (env_count, taxel_count, 3)
    if coordinate_count != 3:
        raise ValueError(
            "taxel_world_positions must have shape [B, N, 3], got "
            f"{tuple(taxel_world_positions.shape)}."
        )
    for tensor_name, tensor in (
        ("taxel_world_normals", taxel_world_normals),
        ("taxel_world_tangent_u", taxel_world_tangent_u),
        ("taxel_world_tangent_v", taxel_world_tangent_v),
    ):
        if tuple(tensor.shape) != expected_world_shape:
            raise ValueError(
                f"{tensor_name} must have shape {expected_world_shape}, got "
                f"{tuple(tensor.shape)}."
            )
    contact_count = contact_positions.shape[0]
    if tuple(contact_env_indices.shape) != (contact_count,):
        raise ValueError("contact_env_indices must have shape [C].")
    if tuple(contact_parent_indices.shape) != (contact_count,):
        raise ValueError("contact_parent_indices must have shape [C].")
    if tuple(taxel_parent_indices.shape) != (taxel_count,):
        raise ValueError("taxel_parent_indices must have shape [N].")
    if tuple(taxel_half_sizes.shape) != (taxel_count, 2):
        raise ValueError("taxel_half_sizes must have shape [N, 2].")
    if contact_count == 0:
        return (
            torch.zeros_like(taxel_world_positions),
            (
                torch.zeros(
                    (0, taxel_count),
                    device=taxel_world_positions.device,
                    dtype=taxel_world_positions.dtype,
                )
                if include_diagnostics
                else None
            ),
            (
                torch.zeros(
                    (0,),
                    device=taxel_world_positions.device,
                    dtype=torch.bool,
                )
                if include_diagnostics
                else None
            ),
        )
    if bool(torch.any(contact_env_indices < 0)) or bool(
        torch.any(contact_env_indices >= env_count)
    ):
        raise ValueError("contact_env_indices contains an invalid environment.")

    # [B, N, 3] indexed by [C] -> [C, N, 3].
    centers = taxel_world_positions[contact_env_indices]
    normals = taxel_world_normals[contact_env_indices]
    tangent_u = taxel_world_tangent_u[contact_env_indices]
    tangent_v = taxel_world_tangent_v[contact_env_indices]
    # [C, 3] -> [C, N, 3].
    position_delta = contact_positions[:, None, :] - centers
    local_u = torch.sum(position_delta * tangent_u, dim=-1).abs()
    local_v = torch.sum(position_delta * tangent_v, dim=-1).abs()
    local_normal = torch.sum(position_delta * normals, dim=-1).abs()

    half_width = taxel_half_sizes[:, 0].unsqueeze(0)
    half_height = taxel_half_sizes[:, 1].unsqueeze(0)
    support_width = half_width + assignment_margin_m
    support_height = half_height + assignment_margin_m
    same_parent = (
        contact_parent_indices[:, None] == taxel_parent_indices[None, :]
    )
    compression = -torch.sum(
        parent_world_impulses[:, None, :] * normals,
        dim=-1,
    )
    valid = (
        same_parent
        & (local_u <= support_width)
        & (local_v <= support_height)
        & (local_normal <= normal_tolerance_m)
        & (compression > 0.0)
    )

    # [C, N]: compact weights prefer the closest patch while remaining smooth.
    weight_u = (1.0 - local_u / support_width).clamp_min(0.0)
    weight_v = (1.0 - local_v / support_height).clamp_min(0.0)
    weight_normal = (
        1.0 - local_normal / normal_tolerance_m
    ).clamp_min(0.0)
    weights = weight_u * weight_v * weight_normal * valid
    weight_sum = weights.sum(dim=1, keepdim=True)
    weights = torch.where(
        weight_sum > 0.0,
        weights / weight_sum.clamp_min(torch.finfo(weights.dtype).eps),
        torch.zeros_like(weights),
    )
    covered_contact_mask = weight_sum[:, 0] > 0.0 if include_diagnostics else None

    # [C, N] * [C, 1, 3] -> [C, N, 3], then C -> B accumulation.
    per_contact_taxel_impulses = (
        weights[..., None] * parent_world_impulses[:, None, :]
    )
    taxel_world_impulses = torch.zeros_like(taxel_world_positions)
    taxel_world_impulses.index_add_(
        dim=0,
        index=contact_env_indices,
        source=per_contact_taxel_impulses,
    )
    return (
        taxel_world_impulses,
        weights if include_diagnostics else None,
        covered_contact_mask,
    )


def distribute_contact_impulses_to_taxels_with_diagnostics(
    *,
    contact_positions: torch.Tensor,
    parent_world_impulses: torch.Tensor,
    contact_env_indices: torch.Tensor,
    contact_parent_indices: torch.Tensor,
    taxel_world_positions: torch.Tensor,
    taxel_world_normals: torch.Tensor,
    taxel_world_tangent_u: torch.Tensor,
    taxel_world_tangent_v: torch.Tensor,
    taxel_parent_indices: torch.Tensor,
    taxel_half_sizes: torch.Tensor,
    assignment_margin_m: float,
    normal_tolerance_m: float,
) -> TaxelImpulseDistribution:
    """Return mapped impulses together with per-contact assignment details."""

    taxel_impulses, contact_weights, covered_contact_mask = (
        _distribute_contact_impulses(
            contact_positions=contact_positions,
            parent_world_impulses=parent_world_impulses,
            contact_env_indices=contact_env_indices,
            contact_parent_indices=contact_parent_indices,
            taxel_world_positions=taxel_world_positions,
            taxel_world_normals=taxel_world_normals,
            taxel_world_tangent_u=taxel_world_tangent_u,
            taxel_world_tangent_v=taxel_world_tangent_v,
            taxel_parent_indices=taxel_parent_indices,
            taxel_half_sizes=taxel_half_sizes,
            assignment_margin_m=assignment_margin_m,
            normal_tolerance_m=normal_tolerance_m,
            include_diagnostics=True,
        )
    )
    assert contact_weights is not None
    assert covered_contact_mask is not None
    return TaxelImpulseDistribution(
        taxel_world_impulses=taxel_impulses,
        contact_weights=contact_weights,
        covered_contact_mask=covered_contact_mask,
    )


def distribute_contact_impulses_to_taxels(
    *,
    contact_positions: torch.Tensor,
    parent_world_impulses: torch.Tensor,
    contact_env_indices: torch.Tensor,
    contact_parent_indices: torch.Tensor,
    taxel_world_positions: torch.Tensor,
    taxel_world_normals: torch.Tensor,
    taxel_world_tangent_u: torch.Tensor,
    taxel_world_tangent_v: torch.Tensor,
    taxel_parent_indices: torch.Tensor,
    taxel_half_sizes: torch.Tensor,
    assignment_margin_m: float,
    normal_tolerance_m: float,
) -> torch.Tensor:
    """Return mapped taxel impulses without exposing mapping diagnostics."""

    taxel_impulses, _, _ = _distribute_contact_impulses(
        contact_positions=contact_positions,
        parent_world_impulses=parent_world_impulses,
        contact_env_indices=contact_env_indices,
        contact_parent_indices=contact_parent_indices,
        taxel_world_positions=taxel_world_positions,
        taxel_world_normals=taxel_world_normals,
        taxel_world_tangent_u=taxel_world_tangent_u,
        taxel_world_tangent_v=taxel_world_tangent_v,
        taxel_parent_indices=taxel_parent_indices,
        taxel_half_sizes=taxel_half_sizes,
        assignment_margin_m=assignment_margin_m,
        normal_tolerance_m=normal_tolerance_m,
        include_diagnostics=False,
    )
    return taxel_impulses
