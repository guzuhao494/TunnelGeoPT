"""Deterministic nodal recovery of element-wise constant planar stress.

The recovery operator is geometry-only: each nodal value is obtained from the
constant stresses of incident triangles, and query values are then interpolated
with barycentric coordinates.  Consequently the complete mapping is linear in
``element_stress``.  This module defines a numerical operator contract; it does
not claim that recovery reduces discretization error for a particular mesh.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


class StressRecoveryError(ValueError):
    """Raised when stress-recovery geometry or array contracts are invalid."""


def _validated_mesh(
    nodes_yz: ArrayLike,
    elements: ArrayLike,
) -> tuple[FloatArray, IntArray, FloatArray, FloatArray]:
    nodes = np.asarray(nodes_yz, dtype=np.float64)
    raw_elements = np.asarray(elements)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3:
        raise StressRecoveryError("nodes_yz must have shape [N, 2] with N >= 3")
    if not np.isfinite(nodes).all():
        raise StressRecoveryError("nodes_yz must contain only finite values")
    if np.unique(nodes, axis=0).shape[0] != nodes.shape[0]:
        raise StressRecoveryError("nodes_yz contains duplicate coordinates")
    if raw_elements.ndim != 2 or raw_elements.shape[1] != 3 or raw_elements.shape[0] == 0:
        raise StressRecoveryError("elements must have shape [M, 3] with M > 0")
    if raw_elements.dtype.kind not in "iu" or raw_elements.dtype.kind == "b":
        raise StressRecoveryError("elements must contain integer node indices")
    triangles = np.asarray(raw_elements, dtype=np.int64)
    if triangles.min() < 0 or triangles.max() >= nodes.shape[0]:
        raise StressRecoveryError("elements contain an out-of-range node index")
    if np.any(np.diff(np.sort(triangles, axis=1), axis=1) == 0):
        raise StressRecoveryError("an element repeats a node index")
    canonical = np.sort(triangles, axis=1)
    if np.unique(canonical, axis=0).shape[0] != triangles.shape[0]:
        raise StressRecoveryError("elements contain a duplicate triangle")
    if np.unique(triangles).size != nodes.shape[0]:
        raise StressRecoveryError("every node must be referenced by at least one element")

    coordinates = nodes[triangles]
    edge_1 = coordinates[:, 1] - coordinates[:, 0]
    edge_2 = coordinates[:, 2] - coordinates[:, 0]
    determinants = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    span = float(np.max(np.ptp(nodes, axis=0)))
    if not np.isfinite(span) or span <= 0.0:
        raise StressRecoveryError("nodes_yz must span a two-dimensional mesh")
    area_tolerance = 64.0 * np.finfo(np.float64).eps * span**2
    if np.any(np.abs(determinants) <= area_tolerance):
        raise StressRecoveryError("elements contain a degenerate triangle")

    edges = np.sort(
        np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]], axis=0),
        axis=1,
    )
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(edge_counts > 2):
        raise StressRecoveryError("elements contain a non-manifold edge")
    areas = 0.5 * np.abs(determinants)
    centroids = coordinates.mean(axis=1)
    return nodes, triangles, areas, centroids


def _validated_element_stress(element_stress: ArrayLike, element_count: int) -> FloatArray:
    stress = np.asarray(element_stress, dtype=np.float64)
    if stress.shape != (element_count, 3):
        raise StressRecoveryError("element_stress must have shape [M, 3]")
    if not np.isfinite(stress).all():
        raise StressRecoveryError("element_stress must contain only finite values")
    return stress


def _recover_nodal_stress(
    nodes: FloatArray,
    elements: IntArray,
    element_stress: FloatArray,
    areas: FloatArray,
    centroids: FloatArray,
    *,
    rank_tolerance: float,
) -> FloatArray:
    """Recover nodal stresses using incident-element affine least squares."""

    nodal = np.empty((nodes.shape[0], 3), dtype=np.float64)
    mesh_scale = float(np.max(np.ptp(nodes, axis=0)))
    distance_floor = 64.0 * np.finfo(np.float64).eps * mesh_scale
    for node_index, node in enumerate(nodes):
        incident = np.flatnonzero(np.any(elements == node_index, axis=1))
        offsets = centroids[incident] - node
        distances = np.linalg.norm(offsets, axis=1)
        weights = areas[incident] / np.maximum(distances, distance_floor)
        local_scale = max(float(np.max(distances)), distance_floor)
        design = np.column_stack([np.ones(incident.size, dtype=np.float64), offsets / local_scale])
        sqrt_weights = np.sqrt(weights)
        weighted_design = design * sqrt_weights[:, None]
        weighted_stress = element_stress[incident] * sqrt_weights[:, None]
        coefficients, _, rank, _ = np.linalg.lstsq(
            weighted_design,
            weighted_stress,
            rcond=rank_tolerance,
        )
        if rank == 3:
            nodal[node_index] = coefficients[0]
        else:
            # The fallback is still geometry-weighted and linear in stress.
            nodal[node_index] = np.average(
                element_stress[incident],
                axis=0,
                weights=weights,
            )
    return nodal


def _barycentric_coordinates(
    nodes: FloatArray,
    elements: IntArray,
    points: FloatArray,
    element_ids: IntArray,
) -> FloatArray:
    coordinates = nodes[elements[element_ids]]
    edge_1 = coordinates[:, 1] - coordinates[:, 0]
    edge_2 = coordinates[:, 2] - coordinates[:, 0]
    relative = points - coordinates[:, 0]
    denominator = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    weight_1 = (relative[:, 0] * edge_2[:, 1] - relative[:, 1] * edge_2[:, 0]) / denominator
    weight_2 = (edge_1[:, 0] * relative[:, 1] - edge_1[:, 1] * relative[:, 0]) / denominator
    return np.column_stack([1.0 - weight_1 - weight_2, weight_1, weight_2])


def _locate_elements(
    nodes: FloatArray,
    elements: IntArray,
    points: FloatArray,
    *,
    tolerance: float,
) -> IntArray:
    located = np.full(points.shape[0], -1, dtype=np.int64)
    # This dependency-free locator is intentionally simple.  Formal generation
    # can pass its already verified element_ids to avoid the O(P*M) search.
    for element_index in range(elements.shape[0]):
        pending = np.flatnonzero(located < 0)
        if pending.size == 0:
            break
        ids = np.full(pending.size, element_index, dtype=np.int64)
        weights = _barycentric_coordinates(nodes, elements, points[pending], ids)
        inside = np.isfinite(weights).all(axis=1) & (weights >= -tolerance).all(axis=1)
        located[pending[inside]] = element_index
    if np.any(located < 0):
        rows = np.flatnonzero(located < 0)
        preview = ", ".join(str(int(row)) for row in rows[:8])
        suffix = "..." if rows.size > 8 else ""
        raise StressRecoveryError(
            f"query_points_yz contains {rows.size} point(s) outside the mesh "
            f"(row indices: {preview}{suffix})"
        )
    return located


def _validated_element_ids(
    element_ids: ArrayLike,
    *,
    point_count: int,
    element_count: int,
) -> IntArray:
    raw = np.asarray(element_ids)
    if raw.shape != (point_count,):
        raise StressRecoveryError("element_ids must have shape [P]")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise StressRecoveryError("element_ids must contain integer indices")
    result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= element_count):
        raise StressRecoveryError("element_ids contain an out-of-range element index")
    return result


def recover_stress_at_queries(
    nodes_yz: ArrayLike,
    elements: ArrayLike,
    element_stress: ArrayLike,
    query_points_yz: ArrayLike,
    element_ids: ArrayLike | None = None,
    *,
    barycentric_tolerance: float = 1e-10,
    rank_tolerance: float = 1e-12,
) -> FloatArray:
    """Recover and sample a continuous P1 stress field at physical queries.

    At each node, incident element-centroid stresses are fitted by weighted
    affine least squares.  The weight is triangle area divided by centroid-to-
    node distance.  A rank-deficient affine system falls back to the same
    geometry-weighted constant average.  The three recovered nodal components
    are interpolated within the containing triangle using barycentric weights.

    ``element_ids`` may be supplied by an independently verified point locator.
    When omitted, a deterministic NumPy locator is used.  Supplied identifiers
    are revalidated against the query coordinates before interpolation.
    """

    if (
        not np.isfinite(barycentric_tolerance)
        or barycentric_tolerance < 0.0
        or barycentric_tolerance >= 1.0
    ):
        raise StressRecoveryError("barycentric_tolerance must lie in [0, 1)")
    if not np.isfinite(rank_tolerance) or not 0.0 < rank_tolerance < 1.0:
        raise StressRecoveryError("rank_tolerance must lie in (0, 1)")

    nodes, triangles, areas, centroids = _validated_mesh(nodes_yz, elements)
    stress = _validated_element_stress(element_stress, triangles.shape[0])
    points = np.asarray(query_points_yz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise StressRecoveryError("query_points_yz must have shape [P, 2]")
    if not np.isfinite(points).all():
        raise StressRecoveryError("query_points_yz must contain only finite values")

    if element_ids is None:
        ids = _locate_elements(
            nodes,
            triangles,
            points,
            tolerance=float(barycentric_tolerance),
        )
    else:
        ids = _validated_element_ids(
            element_ids,
            point_count=points.shape[0],
            element_count=triangles.shape[0],
        )
    barycentric = _barycentric_coordinates(nodes, triangles, points, ids)
    accepted = np.isfinite(barycentric).all(axis=1) & (
        barycentric >= -float(barycentric_tolerance)
    ).all(axis=1)
    if not np.all(accepted):
        rows = np.flatnonzero(~accepted)
        preview = ", ".join(str(int(row)) for row in rows[:8])
        suffix = "..." if rows.size > 8 else ""
        raise StressRecoveryError(
            f"query_points_yz disagrees with supplied element_ids (row indices: {preview}{suffix})"
        )

    nodal_stress = _recover_nodal_stress(
        nodes,
        triangles,
        stress,
        areas,
        centroids,
        rank_tolerance=float(rank_tolerance),
    )
    recovered = np.einsum("pi,pij->pj", barycentric, nodal_stress[triangles[ids]])
    if recovered.shape != (points.shape[0], 3) or not np.isfinite(recovered).all():
        raise RuntimeError("stress recovery produced an invalid output")
    return recovered


def preserve_baseline_traction_with_tangential_correction(
    baseline_stress: ArrayLike,
    recovered_stress: ArrayLike,
    normals_yz: ArrayLike,
    *,
    normal_tolerance: float = 1e-8,
) -> FloatArray:
    """Keep baseline traction while accepting the recovered tangential stress.

    At a wall-near query, write ``t`` for the unit tangent obtained from the
    supplied unit normal ``n``.  Only the tangential-tangential component of
    the recovery increment is retained:

    ``S_out = S_base + (t.T @ (S_rec - S_base) @ t) * outer(t, t)``.

    Therefore ``(S_out - S_base) @ n == 0`` up to roundoff, while
    ``t.T @ S_out @ t == t.T @ S_rec @ t``.  The operator is linear in both
    stress arguments and uses no fine/ultrafine label.  It preserves the
    baseline traction; it does not assert that an offset query is itself a
    traction-free boundary point.
    """

    baseline = np.asarray(baseline_stress, dtype=np.float64)
    recovered = np.asarray(recovered_stress, dtype=np.float64)
    normals = np.asarray(normals_yz, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[1] != 3 or baseline.shape[0] < 1:
        raise StressRecoveryError("baseline_stress must have shape [P, 3] with P > 0")
    if recovered.shape != baseline.shape:
        raise StressRecoveryError("recovered_stress must match baseline_stress shape [P, 3]")
    if normals.shape != (baseline.shape[0], 2):
        raise StressRecoveryError("normals_yz must have shape [P, 2]")
    if not all(np.isfinite(value).all() for value in (baseline, recovered, normals)):
        raise StressRecoveryError("stress and normal arrays must contain only finite values")
    if not np.isfinite(normal_tolerance) or not 0.0 <= float(normal_tolerance) < 1.0:
        raise StressRecoveryError("normal_tolerance must lie in [0, 1)")
    lengths = np.linalg.norm(normals, axis=1)
    if not np.allclose(lengths, 1.0, rtol=0.0, atol=float(normal_tolerance)):
        raise StressRecoveryError("normals_yz rows must be unit length")
    unit_normals = normals / lengths[:, None]
    tangents = np.column_stack([-unit_normals[:, 1], unit_normals[:, 0]])

    increment = recovered - baseline
    tensor_increment = np.empty((baseline.shape[0], 2, 2), dtype=np.float64)
    tensor_increment[:, 0, 0] = increment[:, 0]
    tensor_increment[:, 1, 1] = increment[:, 1]
    tensor_increment[:, 0, 1] = increment[:, 2]
    tensor_increment[:, 1, 0] = increment[:, 2]
    tangential_increment = np.einsum("pi,pij,pj->p", tangents, tensor_increment, tangents)
    tangent_outer = np.einsum("pi,pj->pij", tangents, tangents)
    accepted_increment = tangential_increment[:, None, None] * tangent_outer

    output = baseline.copy()
    output[:, 0] += accepted_increment[:, 0, 0]
    output[:, 1] += accepted_increment[:, 1, 1]
    output[:, 2] += accepted_increment[:, 0, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("traction-preserving recovery produced an invalid output")
    return output


__all__ = [
    "StressRecoveryError",
    "preserve_baseline_traction_with_tangential_correction",
    "recover_stress_at_queries",
]
