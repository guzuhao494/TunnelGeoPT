"""Locate common physical points and sample element-wise tunnel fields.

The elastic solver stores stress as one constant vector per first-order
triangle.  Multi-fidelity comparisons therefore need the *same physical query
points* located independently in each mesh; element indices are never assumed
to correspond between coarse and fine meshes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class ElementLookup:
    """Validated triangle table plus a scikit-fem spatial finder."""

    nodes_yz: FloatArray
    elements: IntArray
    finder: Callable[..., IntArray]

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes_yz, dtype=np.float64)
        elements = np.asarray(self.elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
            raise ValueError("nodes_yz must be a finite array with shape [N, 2]")
        if elements.ndim != 2 or elements.shape[1] != 3 or elements.shape[0] == 0:
            raise ValueError("elements must have shape [M, 3] with M > 0")
        if elements.min() < 0 or elements.max() >= nodes.shape[0]:
            raise ValueError("elements contain an out-of-range node index")
        triangles = nodes[elements]
        twice_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
            triangles[:, 2, 1] - triangles[:, 0, 1]
        ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
        scale = max(float(np.ptp(nodes, axis=0).max()), 1.0)
        if np.any(np.abs(twice_area) <= 64.0 * np.finfo(float).eps * scale**2):
            raise ValueError("elements contain a degenerate triangle")
        if not callable(self.finder):
            raise TypeError("finder must be callable")
        object.__setattr__(self, "nodes_yz", nodes)
        object.__setattr__(self, "elements", elements)

    @classmethod
    def from_mesh(cls, mesh: Any) -> ElementLookup:
        """Build a lookup from a ``TunnelMesh`` or scikit-fem ``MeshTri``."""

        skfem_mesh = getattr(mesh, "skfem_mesh", mesh)
        if not hasattr(skfem_mesh, "p") or not hasattr(skfem_mesh, "t"):
            raise TypeError("mesh must be a TunnelMesh or a scikit-fem triangular mesh")
        nodes = np.asarray(skfem_mesh.p.T, dtype=np.float64)
        elements = np.asarray(skfem_mesh.t.T, dtype=np.int64)
        return cls(nodes, elements, skfem_mesh.element_finder())

    @classmethod
    def from_arrays(cls, nodes_yz: ArrayLike, elements: ArrayLike) -> ElementLookup:
        """Build a lookup from explicit nodes/connectivity using scikit-fem."""

        try:
            from skfem import MeshTri  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency boundary
            raise RuntimeError(
                "array-based element lookup requires scikit-fem; install elastic dependencies"
            ) from exc
        nodes = np.asarray(nodes_yz, dtype=np.float64)
        triangles = np.asarray(elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes_yz must have shape [N, 2]")
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("elements must have shape [M, 3]")
        mesh = MeshTri(nodes.T, triangles.T, validate=True, sort_t=False)
        return cls.from_mesh(mesh)


def _finder_with_outside(
    lookup: ElementLookup, points: FloatArray, *, chunk_size: int
) -> NDArray[np.int64]:
    element_ids = np.full(points.shape[0], -1, dtype=np.int64)

    def locate_interval(start: int, stop: int) -> None:
        if start >= stop:
            return
        try:
            found = np.asarray(
                lookup.finder(points[start:stop, 0], points[start:stop, 1]), dtype=np.int64
            )
        except ValueError as exc:
            # scikit-fem raises for a whole vector when any point is outside.
            # Bisecting retains vectorized lookups for all-inside subgroups.
            if "outside" not in str(exc).lower():
                raise
            if stop - start == 1:
                return
            midpoint = start + (stop - start) // 2
            locate_interval(start, midpoint)
            locate_interval(midpoint, stop)
            return
        if found.shape != (stop - start,):
            raise RuntimeError("scikit-fem element_finder returned an unexpected shape")
        element_ids[start:stop] = found

    for start in range(0, points.shape[0], chunk_size):
        locate_interval(start, min(start + chunk_size, points.shape[0]))
    return element_ids


def _barycentric_revalidate(
    lookup: ElementLookup,
    points: FloatArray,
    element_ids: NDArray[np.int64],
    *,
    tolerance: float,
) -> NDArray[np.int64]:
    valid_rows = np.flatnonzero(element_ids >= 0)
    if valid_rows.size == 0:
        return element_ids
    triangles = lookup.nodes_yz[lookup.elements[element_ids[valid_rows]]]
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    relative = points[valid_rows] - triangles[:, 0]
    denominator = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
    weight_1 = (relative[:, 0] * edge_2[:, 1] - relative[:, 1] * edge_2[:, 0]) / denominator
    weight_2 = (edge_1[:, 0] * relative[:, 1] - edge_1[:, 1] * relative[:, 0]) / denominator
    weight_0 = 1.0 - weight_1 - weight_2
    weights = np.column_stack([weight_0, weight_1, weight_2])
    accepted = np.isfinite(weights).all(axis=1) & (weights >= -tolerance).all(axis=1)
    element_ids[valid_rows[~accepted]] = -1
    return element_ids


def locate_elements(
    lookup_or_nodes: ElementLookup | Any | ArrayLike,
    elements_or_points: ArrayLike,
    points_yz: ArrayLike | None = None,
    *,
    chunk_size: int = 4096,
    barycentric_tolerance: float = 1e-10,
    raise_outside: bool = False,
) -> NDArray[np.int64]:
    """Locate query points independently in a triangular mesh.

    Two call forms are supported: ``locate_elements(lookup_or_mesh, points)``
    and ``locate_elements(nodes, elements, points)``.  Domain-exterior points
    receive ``-1`` unless ``raise_outside=True``.  Every finder result is
    independently checked with barycentric coordinates, including edge points.
    """

    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not np.isfinite(barycentric_tolerance) or barycentric_tolerance < 0.0:
        raise ValueError("barycentric_tolerance must be finite and non-negative")

    if points_yz is None:
        points = np.asarray(elements_or_points, dtype=np.float64)
        lookup = (
            lookup_or_nodes
            if isinstance(lookup_or_nodes, ElementLookup)
            else ElementLookup.from_mesh(lookup_or_nodes)
        )
    else:
        points = np.asarray(points_yz, dtype=np.float64)
        lookup = ElementLookup.from_arrays(lookup_or_nodes, elements_or_points)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points_yz must have shape [P, 2]")
    if not np.isfinite(points).all():
        raise ValueError("points_yz contains a non-finite value")

    element_ids = _finder_with_outside(lookup, points, chunk_size=chunk_size)
    element_ids = _barycentric_revalidate(
        lookup,
        points,
        element_ids,
        tolerance=float(barycentric_tolerance),
    )
    outside = np.flatnonzero(element_ids < 0)
    if raise_outside and outside.size:
        preview = ", ".join(str(int(index)) for index in outside[:8])
        suffix = "..." if outside.size > 8 else ""
        raise ValueError(
            f"{outside.size} point(s) lie outside the triangular mesh "
            f"(row indices: {preview}{suffix})"
        )
    return element_ids


def sample_piecewise_constant(
    element_values: ArrayLike,
    element_ids: ArrayLike,
    *,
    allow_outside: bool = False,
    fill_value: float = np.nan,
) -> NDArray[Any]:
    """Gather one scalar/vector/tensor value per located triangle."""

    values = np.asarray(element_values)
    ids = np.asarray(element_ids, dtype=np.int64)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("element_values must have shape [M, ...] with M > 0")
    if ids.ndim != 1:
        raise ValueError("element_ids must have shape [P]")
    if np.any(ids >= values.shape[0]):
        raise ValueError("element_ids contain an out-of-range element index")
    outside = ids < 0
    if outside.any() and not allow_outside:
        raise ValueError("element_ids contain outside points (-1)")
    if not outside.any():
        return values[ids]

    dtype = np.result_type(values.dtype, np.asarray(fill_value).dtype)
    sampled = np.full((ids.shape[0], *values.shape[1:]), fill_value, dtype=dtype)
    sampled[~outside] = values[ids[~outside]]
    return sampled


__all__ = ["ElementLookup", "locate_elements", "sample_piecewise_constant"]
