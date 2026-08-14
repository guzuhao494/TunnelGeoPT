"""Audited zero-width slit meshes for SENT and SENS fracture benchmarks.

The repository coordinate convention is ``(y, z)``: ``y`` is vertical and
``z`` is horizontal.  The benchmark coupon is the unit square in millimetres
with a horizontal slit at ``y = 0.5`` from ``z = 0`` to ``z = 0.5``.

The slit is represented by two coincident, topologically distinct boundary
curves.  Their nodes are distinct everywhere except at the shared crack tip.
This is deliberately not a finite-width slot.  Every generated mesh is
audited before it is returned; Gmsh merging the two faces is a hard failure.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from .mesh import (
    _GMSH_LOCK,
    _extract_first_order_elements,
    _facets_from_edges,
    _triangle_quality,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

BULK = "bulk"
TOP = "top"
BOTTOM = "bottom"
LEFT_UPPER = "left_upper"
LEFT_LOWER = "left_lower"
RIGHT = "right"
NOTCH_UPPER = "notch_upper"
NOTCH_LOWER = "notch_lower"
NOTCH_TIP = "notch_tip"

BOUNDARY_LABELS = (
    TOP,
    BOTTOM,
    LEFT_UPPER,
    LEFT_LOWER,
    RIGHT,
    NOTCH_UPPER,
    NOTCH_LOWER,
)
BOUNDARY_NODE_LABELS = (NOTCH_TIP,)
PHYSICAL_LABELS = (BULK, *BOUNDARY_LABELS, *BOUNDARY_NODE_LABELS)
MESH_TIERS = ("coarse", "medium", "fine")
LOADING_MODES = ("sent", "sens")

_TARGET_H_MM = MappingProxyType(
    {
        "coarse": 0.0075,
        "medium": 0.00375,
        "fine": 0.001875,
    }
)
_FARFIELD_H_MM = MappingProxyType(
    {
        "coarse": 0.03,
        "medium": 0.015,
        "fine": 0.0075,
    }
)
_NOTCH_BAND_HALF_WIDTH_MM = 0.05
_PROPAGATION_CORRIDOR_HALF_WIDTH_MM = MappingProxyType(
    {
        "sent": 0.10,
        "sens": 0.15,
    }
)
_CORRIDOR_TRANSITION_MM = 0.10
_GMSH_TARGET_FACTOR = 0.50
_CORRIDOR_HMAX_FACTOR = 1.15
_COORDINATE_TOLERANCE_MM = 2.0e-12

_PHYSICAL_TAGS = MappingProxyType(
    {
        BULK: 1,
        TOP: 11,
        BOTTOM: 12,
        LEFT_UPPER: 13,
        LEFT_LOWER: 14,
        RIGHT: 15,
        NOTCH_UPPER: 16,
        NOTCH_LOWER: 17,
        NOTCH_TIP: 18,
    }
)


@dataclass(frozen=True)
class FractureBenchmarkMeshPlan:
    """Frozen meshing plan for one loading mode and refinement tier."""

    loading: str
    tier: str
    target_h_mm: float
    farfield_h_mm: float
    notch_band_half_width_mm: float
    propagation_corridor_half_width_mm: float
    corridor_transition_mm: float
    notch_polyline_yz_mm: tuple[tuple[float, float], ...]
    propagation_corridor_polyline_yz_mm: tuple[tuple[float, float], ...]
    plan_sha256: str


@dataclass(frozen=True)
class FractureBenchmarkMesh:
    """MeshTri-compatible coupon mesh with immutable identity metadata."""

    mesh: Any
    nodes: FloatArray
    elements: IntArray
    boundary_facets: Mapping[str, IntArray]
    boundary_nodes: Mapping[str, IntArray]
    facet_markers: NDArray[np.int32]
    cell_markers: NDArray[np.int32]
    physical_tags: Mapping[str, int]
    physical_entity_tags: Mapping[str, tuple[int, ...]]
    plan: FractureBenchmarkMeshPlan
    identity: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 2:
            raise ValueError("nodes must have shape [N, 2] in (y, z) order")
        if self.elements.ndim != 2 or self.elements.shape[1] != 3:
            raise ValueError("elements must have shape [M, 3]")
        if set(self.boundary_facets) != set(BOUNDARY_LABELS):
            raise ValueError("boundary_facets do not have the frozen physical labels")
        if set(self.boundary_nodes) != set(BOUNDARY_NODE_LABELS):
            raise ValueError("boundary_nodes do not have the frozen point labels")
        if tuple(self.physical_tags) != PHYSICAL_LABELS:
            raise ValueError("physical_tags do not have the frozen label order")

    @property
    def skfem_mesh(self) -> Any:
        """Alias for the wrapped :class:`skfem.MeshTri` object."""

        return self.mesh

    @property
    def nodes_yz_mm(self) -> FloatArray:
        return self.nodes

    @property
    def triangles(self) -> IntArray:
        return self.elements

    def recompute_topology_sha256(self) -> str:
        """Recompute the digest from the exact frozen object consumed downstream."""

        return _canonical_mesh_sha256(
            nodes=np.asarray(self.mesh.p.T, dtype=np.float64),
            elements=np.asarray(self.mesh.t.T, dtype=np.int64),
            mesh=self.mesh,
            boundary_facets=self.boundary_facets,
            boundary_nodes=self.boundary_nodes,
            physical_tags=self.physical_tags,
            physical_entity_tags=self.physical_entity_tags,
        )


def _require_dependencies() -> tuple[Any, Any]:
    try:
        import gmsh  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("benchmark meshing requires gmsh 4.x") from exc
    try:
        from skfem import MeshTri  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("benchmark meshing requires scikit-fem 12.x") from exc
    return gmsh, MeshTri


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def benchmark_mesh_plan(
    *,
    loading: str = "sent",
    tier: str = "coarse",
) -> FractureBenchmarkMeshPlan:
    """Return the immutable, predeclared meshing plan.

    ``sent`` refines the complete horizontal slit/ligament line.  ``sens``
    refines the slit followed by a down-right diagonal from the crack tip.
    No arbitrary corridor is accepted: this keeps three-grid comparisons from
    silently changing geometry between runs.
    """

    normalized_loading = str(loading).strip().lower()
    normalized_tier = str(tier).strip().lower()
    if normalized_loading not in LOADING_MODES:
        raise ValueError(f"loading must be one of {LOADING_MODES}")
    if normalized_tier not in MESH_TIERS:
        raise ValueError(f"tier must be one of {MESH_TIERS}")

    notch = ((0.5, 0.0), (0.5, 0.5))
    if normalized_loading == "sent":
        propagation_corridor = ((0.5, 0.5), (0.5, 1.0))
    else:
        propagation_corridor = ((0.5, 0.5), (0.0, 1.0))
    payload = {
        "schema": "tunnelgeopt-fracture-benchmark-mesh-plan-v1",
        "coordinate_order": ["y", "z"],
        "coordinate_unit": "mm",
        "domain_yz_mm": [[0.0, 0.0], [1.0, 1.0]],
        "slit_y_mm": 0.5,
        "slit_z_interval_mm": [0.0, 0.5],
        "loading": normalized_loading,
        "tier": normalized_tier,
        "target_h_mm": _TARGET_H_MM[normalized_tier],
        "farfield_h_mm": _FARFIELD_H_MM[normalized_tier],
        "notch_band_half_width_mm": _NOTCH_BAND_HALF_WIDTH_MM,
        "propagation_corridor_half_width_mm": (
            _PROPAGATION_CORRIDOR_HALF_WIDTH_MM[normalized_loading]
        ),
        "corridor_transition_mm": _CORRIDOR_TRANSITION_MM,
        "notch_polyline_yz_mm": [list(point) for point in notch],
        "propagation_corridor_polyline_yz_mm": [list(point) for point in propagation_corridor],
        "gmsh_target_factor": _GMSH_TARGET_FACTOR,
        "corridor_hmax_factor": _CORRIDOR_HMAX_FACTOR,
    }
    return FractureBenchmarkMeshPlan(
        loading=normalized_loading,
        tier=normalized_tier,
        target_h_mm=float(_TARGET_H_MM[normalized_tier]),
        farfield_h_mm=float(_FARFIELD_H_MM[normalized_tier]),
        notch_band_half_width_mm=_NOTCH_BAND_HALF_WIDTH_MM,
        propagation_corridor_half_width_mm=(
            _PROPAGATION_CORRIDOR_HALF_WIDTH_MM[normalized_loading]
        ),
        corridor_transition_mm=_CORRIDOR_TRANSITION_MM,
        notch_polyline_yz_mm=notch,
        propagation_corridor_polyline_yz_mm=propagation_corridor,
        plan_sha256=_canonical_json_sha256(payload),
    )


def _compact_nodes(
    nodes: FloatArray,
    elements: IntArray,
    edge_groups: Mapping[str, IntArray],
) -> tuple[FloatArray, IntArray, dict[str, IntArray]]:
    used_blocks = [elements.ravel(), *(edges.ravel() for edges in edge_groups.values())]
    used = np.unique(np.concatenate(used_blocks))
    remap = np.full(nodes.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    return (
        np.asarray(nodes[used], dtype=np.float64),
        np.asarray(remap[elements], dtype=np.int64),
        {label: np.asarray(remap[edges], dtype=np.int64) for label, edges in edge_groups.items()},
    )


def _point_to_open_polyline_distance(
    points: FloatArray,
    polyline: Sequence[Sequence[float]],
) -> FloatArray:
    line = np.asarray(polyline, dtype=np.float64)
    starts = line[:-1]
    directions = line[1:] - starts
    length_squared = np.sum(directions**2, axis=1)
    distances = np.full(points.shape[0], np.inf, dtype=np.float64)
    for start, direction, denominator in zip(starts, directions, length_squared, strict=True):
        fractions = np.clip((points - start) @ direction / denominator, 0.0, 1.0)
        closest = start + fractions[:, None] * direction
        distances = np.minimum(distances, np.linalg.norm(points - closest, axis=1))
    return distances


def _facet_node_sets(mesh: Any, facets: IntArray) -> set[int]:
    return {int(value) for value in np.asarray(mesh.facets)[:, facets].ravel()}


def _assert_coordinate_label(
    nodes: FloatArray,
    mesh: Any,
    facets: IntArray,
    *,
    axis: int,
    value: float,
    label: str,
) -> None:
    coordinates = nodes[np.asarray(mesh.facets)[:, facets], axis]
    if not np.allclose(coordinates, value, rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
        raise RuntimeError(f"physical boundary {label!r} has off-geometry nodes")


def _audit_zero_width_slit(
    *,
    mesh: Any,
    nodes: FloatArray,
    elements: IntArray,
    boundary_facets: Mapping[str, IntArray],
    boundary_nodes: Mapping[str, IntArray],
    interface_facets: IntArray,
    plan: FractureBenchmarkMeshPlan,
) -> dict[str, Any]:
    """Fail closed unless the generated topology is an exact double-face slit."""

    if set(boundary_facets) != set(BOUNDARY_LABELS):
        raise RuntimeError("benchmark boundary labels are incomplete or unexpected")
    if set(boundary_nodes) != set(BOUNDARY_NODE_LABELS):
        raise RuntimeError("benchmark point-boundary labels are incomplete or unexpected")
    for label in BOUNDARY_LABELS:
        if np.asarray(boundary_facets[label]).size == 0:
            raise RuntimeError(f"physical boundary {label!r} has no facets")

    concatenated = np.concatenate([np.asarray(boundary_facets[label]) for label in BOUNDARY_LABELS])
    if np.unique(concatenated).size != concatenated.size:
        raise RuntimeError("physical boundary facet groups overlap")
    actual_boundary = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    if not np.array_equal(np.sort(concatenated), np.sort(actual_boundary)):
        raise RuntimeError("physical labels do not exactly cover the mesh boundary")

    _assert_coordinate_label(nodes, mesh, boundary_facets[TOP], axis=0, value=1.0, label=TOP)
    _assert_coordinate_label(nodes, mesh, boundary_facets[BOTTOM], axis=0, value=0.0, label=BOTTOM)
    _assert_coordinate_label(
        nodes, mesh, boundary_facets[LEFT_UPPER], axis=1, value=0.0, label=LEFT_UPPER
    )
    _assert_coordinate_label(
        nodes, mesh, boundary_facets[LEFT_LOWER], axis=1, value=0.0, label=LEFT_LOWER
    )
    _assert_coordinate_label(nodes, mesh, boundary_facets[RIGHT], axis=1, value=1.0, label=RIGHT)
    for label in (NOTCH_UPPER, NOTCH_LOWER):
        _assert_coordinate_label(
            nodes, mesh, boundary_facets[label], axis=0, value=0.5, label=label
        )
        notch_z = nodes[np.asarray(mesh.facets)[:, boundary_facets[label]], 1]
        if notch_z.min() < -_COORDINATE_TOLERANCE_MM or notch_z.max() > (
            0.5 + _COORDINATE_TOLERANCE_MM
        ):
            raise RuntimeError(f"physical boundary {label!r} extends outside the frozen slit")

    upper_nodes = _facet_node_sets(mesh, boundary_facets[NOTCH_UPPER])
    lower_nodes = _facet_node_sets(mesh, boundary_facets[NOTCH_LOWER])
    shared_nodes = upper_nodes & lower_nodes
    if len(shared_nodes) != 1:
        raise RuntimeError("Gmsh merged slit faces or failed to preserve one shared crack tip")
    tip_index = next(iter(shared_nodes))
    if not np.allclose(nodes[tip_index], (0.5, 0.5), rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
        raise RuntimeError("the sole shared slit-face node is not the crack tip")
    point_tip_nodes = np.asarray(boundary_nodes[NOTCH_TIP], dtype=np.int64)
    if point_tip_nodes.shape != (1,) or int(point_tip_nodes[0]) != tip_index:
        raise RuntimeError("0D physical notch_tip must be exactly the shared crack-tip node")

    left_upper_nodes = _facet_node_sets(mesh, boundary_facets[LEFT_UPPER])
    left_lower_nodes = _facet_node_sets(mesh, boundary_facets[LEFT_LOWER])
    if left_upper_nodes & left_lower_nodes:
        raise RuntimeError("left_upper and left_lower must use distinct slit-mouth nodes")
    for left_label, notch_label, left_nodes, notch_nodes in (
        (LEFT_UPPER, NOTCH_UPPER, left_upper_nodes, upper_nodes),
        (LEFT_LOWER, NOTCH_LOWER, left_lower_nodes, lower_nodes),
    ):
        contact = left_nodes & notch_nodes
        if len(contact) != 1:
            raise RuntimeError(f"{left_label} must meet {notch_label} at one slit-mouth node")
        mouth_index = next(iter(contact))
        if not np.allclose(nodes[mouth_index], (0.5, 0.0), rtol=0.0, atol=_COORDINATE_TOLERANCE_MM):
            raise RuntimeError(f"{left_label}/{notch_label} contact is not the slit mouth")
    if left_upper_nodes & lower_nodes or left_lower_nodes & upper_nodes:
        raise RuntimeError("a left boundary segment touches the opposite slit face")
    left_upper_y = nodes[np.asarray(sorted(left_upper_nodes), dtype=np.int64), 0]
    left_lower_y = nodes[np.asarray(sorted(left_lower_nodes), dtype=np.int64), 0]
    if left_upper_y.min() < 0.5 - _COORDINATE_TOLERANCE_MM:
        raise RuntimeError("left_upper extends below the slit mouth")
    if left_lower_y.max() > 0.5 + _COORDINATE_TOLERANCE_MM:
        raise RuntimeError("left_lower extends above the slit mouth")

    upper_distinct = sorted(upper_nodes - {tip_index}, key=lambda index: nodes[index, 1])
    lower_distinct = sorted(lower_nodes - {tip_index}, key=lambda index: nodes[index, 1])
    if len(upper_distinct) < 2 or len(upper_distinct) != len(lower_distinct):
        raise RuntimeError("slit faces do not have paired, independently meshed node chains")
    upper_coordinates = nodes[np.asarray(upper_distinct, dtype=np.int64)]
    lower_coordinates = nodes[np.asarray(lower_distinct, dtype=np.int64)]
    if not np.allclose(
        upper_coordinates,
        lower_coordinates,
        rtol=0.0,
        atol=_COORDINATE_TOLERANCE_MM,
    ):
        raise RuntimeError("upper and lower slit-face nodes are not geometrically coincident")
    if any(upper == lower for upper, lower in zip(upper_distinct, lower_distinct, strict=True)):
        raise RuntimeError("Gmsh reused a node across the open part of the slit")

    f2t = np.asarray(mesh.f2t)
    notch_facets = np.concatenate([boundary_facets[NOTCH_UPPER], boundary_facets[NOTCH_LOWER]])
    if not np.all(np.sum(f2t[:, notch_facets] >= 0, axis=0) == 1):
        raise RuntimeError("each slit-face facet must border exactly one triangle")
    if interface_facets.size == 0 or not np.all(np.sum(f2t[:, interface_facets] >= 0, axis=0) == 2):
        raise RuntimeError("the ligament ahead of the crack tip is not a shared interior interface")
    if np.intersect1d(interface_facets, actual_boundary).size:
        raise RuntimeError("the intact ligament was incorrectly classified as a free boundary")

    area, quality = _triangle_quality(nodes, elements)
    if area.size == 0 or np.any(area <= 0.0) or not np.isfinite(quality).all():
        raise RuntimeError("benchmark mesh contains a non-positive or non-finite triangle")
    total_area = float(area.sum())
    if not math.isclose(total_area, 1.0, rel_tol=0.0, abs_tol=2.0e-12):
        raise RuntimeError(f"benchmark mesh area is {total_area:.17g}, expected 1 mm^2")

    triangles = nodes[elements]
    centroids = triangles.mean(axis=1)
    notch_distance = _point_to_open_polyline_distance(centroids, plan.notch_polyline_yz_mm)
    propagation_distance = _point_to_open_polyline_distance(
        centroids, plan.propagation_corridor_polyline_yz_mm
    )
    notch_mask = notch_distance <= plan.notch_band_half_width_mm
    propagation_mask = propagation_distance <= plan.propagation_corridor_half_width_mm
    corridor_mask = notch_mask | propagation_mask
    if not np.any(corridor_mask):
        raise RuntimeError("refinement-corridor audit selected no triangles")
    edge_lengths = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
    actual_hmax = float(edge_lengths[corridor_mask].max())
    roundoff = 128.0 * np.finfo(np.float64).eps
    hmax_limit = float(_CORRIDOR_HMAX_FACTOR * plan.target_h_mm + roundoff)
    if actual_hmax > hmax_limit:
        raise RuntimeError(
            "refinement-corridor maximum edge audit failed: "
            f"actual={actual_hmax:.17g}, limit={hmax_limit:.17g}"
        )

    return {
        "topology_audit_passed": True,
        "positive_triangle_audit_passed": True,
        "boundary_coverage_audit_passed": True,
        "zero_width_double_face_slit_audit_passed": True,
        "slit_face_shared_node_count": 1,
        "slit_shared_tip_node_index": int(tip_index),
        "notch_tip_point_entity_audit_passed": True,
        "left_split_mouth_contact_audit_passed": True,
        "slit_distinct_coincident_node_pair_count": len(upper_distinct),
        "notch_upper_facet_count": int(boundary_facets[NOTCH_UPPER].size),
        "notch_lower_facet_count": int(boundary_facets[NOTCH_LOWER].size),
        "intact_ligament_facet_count": int(interface_facets.size),
        "minimum_element_area_mm2": float(area.min()),
        "total_element_area_mm2": total_area,
        "minimum_triangle_quality": float(quality.min()),
        "corridor_cell_selection": ("triangle_centroid_in_notch_band_union_propagation_corridor"),
        "notch_band_audited_element_count": int(np.count_nonzero(notch_mask)),
        "propagation_corridor_audited_element_count": int(np.count_nonzero(propagation_mask)),
        "corridor_audited_element_count": int(np.count_nonzero(corridor_mask)),
        "corridor_actual_hmax_mm": actual_hmax,
        "corridor_hmax_limit_mm": hmax_limit,
        "corridor_hmax_factor": _CORRIDOR_HMAX_FACTOR,
        "corridor_hmax_audit_passed": True,
    }


def _canonical_mesh_sha256(
    *,
    nodes: FloatArray,
    elements: IntArray,
    mesh: Any,
    boundary_facets: Mapping[str, IntArray],
    boundary_nodes: Mapping[str, IntArray],
    physical_tags: Mapping[str, int],
    physical_entity_tags: Mapping[str, tuple[int, ...]],
) -> str:
    """Hash a canonical node numbering, connectivity, and physical identity."""

    membership = np.zeros(nodes.shape[0], dtype=np.uint16)
    for bit, label in enumerate(BOUNDARY_LABELS):
        indices = np.unique(np.asarray(mesh.facets)[:, boundary_facets[label]])
        membership[indices] |= np.uint16(1 << bit)
    for offset, label in enumerate(BOUNDARY_NODE_LABELS, start=len(BOUNDARY_LABELS)):
        indices = np.asarray(boundary_nodes[label], dtype=np.int64)
        membership[indices] |= np.uint16(1 << offset)
    order = np.lexsort((membership, nodes[:, 1], nodes[:, 0]))
    inverse = np.empty(nodes.shape[0], dtype=np.int64)
    inverse[order] = np.arange(nodes.shape[0], dtype=np.int64)

    canonical_nodes = np.column_stack(
        [nodes[order, 0], nodes[order, 1], membership[order].astype(np.float64)]
    ).astype("<f8", copy=False)
    canonical_elements = np.sort(inverse[elements], axis=1)
    canonical_elements = canonical_elements[
        np.lexsort(
            (
                canonical_elements[:, 2],
                canonical_elements[:, 1],
                canonical_elements[:, 0],
            )
        )
    ].astype("<i8", copy=False)

    digest = hashlib.sha256()
    for label, array in (("nodes", canonical_nodes), ("elements", canonical_elements)):
        digest.update(label.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes(order="C"))
    for label in BOUNDARY_LABELS:
        edges = inverse[np.asarray(mesh.facets)[:, boundary_facets[label]].T]
        edges = np.sort(edges, axis=1)
        edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))].astype("<i8", copy=False)
        digest.update(label.encode("ascii"))
        digest.update(np.asarray(edges.shape, dtype="<i8").tobytes())
        digest.update(edges.tobytes(order="C"))
    for label in BOUNDARY_NODE_LABELS:
        canonical_points = np.sort(
            inverse[np.asarray(boundary_nodes[label], dtype=np.int64)]
        ).astype("<i8", copy=False)
        digest.update(label.encode("ascii"))
        digest.update(b"\0point-nodes\0")
        digest.update(np.asarray(canonical_points.shape, dtype="<i8").tobytes())
        digest.update(canonical_points.tobytes())
    for label in PHYSICAL_LABELS:
        dimension = 2 if label == BULK else (0 if label in BOUNDARY_NODE_LABELS else 1)
        digest.update(label.encode("ascii"))
        digest.update(b"\0physical-identity\0")
        digest.update(np.asarray([dimension, physical_tags[label]], dtype="<i8").tobytes())
        entities = np.asarray(physical_entity_tags[label], dtype="<i8")
        digest.update(np.asarray(entities.shape, dtype="<i8").tobytes())
        digest.update(entities.tobytes())
    return digest.hexdigest()


def _freeze_meshtri(mesh: Any) -> None:
    """Freeze the MeshTri arrays and marker maps backing the public wrapper."""

    for attribute in ("doflocs", "t", "_facets", "_t2f", "_f2t"):
        values = getattr(mesh, attribute, None)
        if isinstance(values, np.ndarray):
            values.setflags(write=False)
    for attribute in ("_boundaries", "_subdomains"):
        values = getattr(mesh, attribute, None)
        if values is None:
            continue
        frozen: dict[str, NDArray[Any]] = {}
        for label, indices in values.items():
            array = np.asarray(indices)
            array.setflags(write=False)
            frozen[str(label)] = array
        setattr(mesh, attribute, MappingProxyType(frozen))


def _readonly_array(values: NDArray[Any], *, dtype: Any) -> NDArray[Any]:
    result = np.asarray(values, dtype=dtype)
    result.setflags(write=False)
    return result


def generate_fracture_benchmark_mesh(
    *,
    loading: str = "sent",
    tier: str = "coarse",
    gmsh_algorithm: int = 6,
    verbose: bool = False,
) -> FractureBenchmarkMesh:
    """Generate and audit a first-order triangular SENT/SENS coupon mesh.

    This function validates geometry and mesh topology only.  It does not run
    a fracture trajectory and does not constitute numerical benchmark
    validation.
    """

    plan = benchmark_mesh_plan(loading=loading, tier=tier)
    if not isinstance(gmsh_algorithm, int) or gmsh_algorithm <= 0:
        raise ValueError("gmsh_algorithm must be a positive integer")
    gmsh, MeshTri = _require_dependencies()

    target = plan.target_h_mm
    gmsh_target = _GMSH_TARGET_FACTOR * target
    model_name = f"tunnelgeopt_fracture_benchmark_{uuid4().hex}"
    initialized_here = False
    model_added = False
    previous_model: str | None = None

    with _GMSH_LOCK:
        try:
            if not gmsh.isInitialized():
                gmsh.initialize()
                initialized_here = True
            else:
                previous_model = gmsh.model.getCurrent() or None
            gmsh.option.setNumber("General.Terminal", 1.0 if verbose else 0.0)
            gmsh.option.setNumber("General.NumThreads", 1.0)
            gmsh.option.setNumber("Mesh.ElementOrder", 1.0)
            gmsh.option.setNumber("Mesh.Algorithm", float(gmsh_algorithm))
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0.0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0)
            gmsh.model.add(model_name)
            model_added = True
            gmsh.model.setCurrent(model_name)

            # Coordinates below are (Gmsh x, Gmsh y) == repository (y, z).
            point = gmsh.model.geo.addPoint
            bottom_left = point(0.0, 0.0, 0.0, plan.farfield_h_mm)
            mouth_lower = point(0.5, 0.0, 0.0, gmsh_target)
            mouth_upper = point(0.5, 0.0, 0.0, gmsh_target)
            top_left = point(1.0, 0.0, 0.0, plan.farfield_h_mm)
            bottom_right = point(0.0, 1.0, 0.0, plan.farfield_h_mm)
            right_mid = point(0.5, 1.0, 0.0, gmsh_target)
            top_right = point(1.0, 1.0, 0.0, plan.farfield_h_mm)
            tip = point(0.5, 0.5, 0.0, gmsh_target)

            line = gmsh.model.geo.addLine
            left_lower = line(bottom_left, mouth_lower)
            notch_lower = line(mouth_lower, tip)
            intact_interface = line(tip, right_mid)
            right_lower = line(right_mid, bottom_right)
            bottom = line(bottom_right, bottom_left)

            left_upper = line(mouth_upper, top_left)
            top = line(top_left, top_right)
            right_upper = line(top_right, right_mid)
            notch_upper = line(mouth_upper, tip)

            lower_loop = gmsh.model.geo.addCurveLoop(
                [left_lower, notch_lower, intact_interface, right_lower, bottom]
            )
            upper_loop = gmsh.model.geo.addCurveLoop(
                [left_upper, top, right_upper, -intact_interface, -notch_upper]
            )
            lower_surface = gmsh.model.geo.addPlaneSurface([lower_loop])
            upper_surface = gmsh.model.geo.addPlaneSurface([upper_loop])

            # Identical transfinite counts make coincident-face pairing an
            # explicit contract rather than a fortunate Gmsh side effect.
            notch_point_count = math.ceil(0.5 / gmsh_target) + 1
            gmsh.model.geo.mesh.setTransfiniteCurve(notch_lower, notch_point_count)
            gmsh.model.geo.mesh.setTransfiniteCurve(notch_upper, notch_point_count)

            if plan.loading == "sent":
                propagation_curve = intact_interface
            else:
                down_right = line(tip, bottom_right)
                propagation_curve = down_right

            gmsh.model.geo.synchronize()

            physical_entity_tags: dict[str, tuple[int, ...]] = {
                BULK: (lower_surface, upper_surface),
                TOP: (top,),
                BOTTOM: (bottom,),
                LEFT_UPPER: (left_upper,),
                LEFT_LOWER: (left_lower,),
                RIGHT: (right_lower, right_upper),
                NOTCH_UPPER: (notch_upper,),
                NOTCH_LOWER: (notch_lower,),
                NOTCH_TIP: (tip,),
            }
            for label in PHYSICAL_LABELS:
                dim = 2 if label == BULK else (0 if label == NOTCH_TIP else 1)
                entities = physical_entity_tags[label]
                gmsh.model.addPhysicalGroup(dim, list(entities), _PHYSICAL_TAGS[label])
                gmsh.model.setPhysicalName(dim, _PHYSICAL_TAGS[label], label)

            threshold_fields: list[int] = []
            for refinement_curves, half_width in (
                ([notch_lower, notch_upper], plan.notch_band_half_width_mm),
                ([propagation_curve], plan.propagation_corridor_half_width_mm),
            ):
                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", refinement_curves)
                gmsh.model.mesh.field.setNumber(distance_field, "Sampling", 200.0)
                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", gmsh_target)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMax", plan.farfield_h_mm)
                # Keep two target lengths inside the constant fine-size
                # region so audited cells cannot straddle the transition.
                transition_start = half_width + 2.0 * target
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", transition_start)
                gmsh.model.mesh.field.setNumber(
                    threshold_field,
                    "DistMax",
                    transition_start + plan.corridor_transition_mm,
                )
                threshold_fields.append(threshold_field)
            minimum_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(minimum_field, "FieldsList", threshold_fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)
            gmsh.model.mesh.generate(2)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            node_tags = np.asarray(node_tags, dtype=np.int64)
            raw_nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
            if node_tags.size == 0 or node_tags.size != raw_nodes.shape[0]:
                raise RuntimeError("Gmsh returned an invalid node table")
            node_index = {int(tag): index for index, tag in enumerate(node_tags)}
            raw_elements = _extract_first_order_elements(
                gmsh,
                dim=2,
                entity_tags=[lower_surface, upper_surface],
                node_index=node_index,
            )
            entity_edges = {
                TOP: _extract_first_order_elements(
                    gmsh, dim=1, entity_tags=[top], node_index=node_index
                ),
                BOTTOM: _extract_first_order_elements(
                    gmsh, dim=1, entity_tags=[bottom], node_index=node_index
                ),
                LEFT_UPPER: _extract_first_order_elements(
                    gmsh,
                    dim=1,
                    entity_tags=[left_upper],
                    node_index=node_index,
                ),
                LEFT_LOWER: _extract_first_order_elements(
                    gmsh,
                    dim=1,
                    entity_tags=[left_lower],
                    node_index=node_index,
                ),
                RIGHT: _extract_first_order_elements(
                    gmsh,
                    dim=1,
                    entity_tags=[right_lower, right_upper],
                    node_index=node_index,
                ),
                NOTCH_UPPER: _extract_first_order_elements(
                    gmsh, dim=1, entity_tags=[notch_upper], node_index=node_index
                ),
                NOTCH_LOWER: _extract_first_order_elements(
                    gmsh, dim=1, entity_tags=[notch_lower], node_index=node_index
                ),
                "intact_interface": _extract_first_order_elements(
                    gmsh, dim=1, entity_tags=[intact_interface], node_index=node_index
                ),
            }
        finally:
            if gmsh.isInitialized():
                if initialized_here:
                    gmsh.finalize()
                elif model_added:
                    gmsh.model.setCurrent(model_name)
                    gmsh.model.remove()
                    if previous_model is not None:
                        gmsh.model.setCurrent(previous_model)

    nodes, elements, entity_edges = _compact_nodes(raw_nodes, raw_elements, entity_edges)
    mesh = MeshTri(nodes.T, elements.T, validate=True, sort_t=False)
    boundary_facets = {
        label: _facets_from_edges(mesh, entity_edges[label], label=label)
        for label in BOUNDARY_LABELS
    }
    interface_facets = _facets_from_edges(
        mesh, entity_edges["intact_interface"], label="intact_interface"
    )
    mesh = mesh.with_boundaries(boundary_facets)
    mesh = mesh.with_subdomains({BULK: np.arange(elements.shape[0], dtype=np.int64)})

    # Use the exact representation downstream solvers receive.
    nodes = np.asarray(mesh.p.T, dtype=np.float64)
    elements = np.asarray(mesh.t.T, dtype=np.int64)
    tip_nodes = np.flatnonzero(
        np.all(
            np.isclose(nodes, (0.5, 0.5), rtol=0.0, atol=_COORDINATE_TOLERANCE_MM),
            axis=1,
        )
    ).astype(np.int64)
    if tip_nodes.size != 1:
        raise RuntimeError("physical point notch_tip must map to exactly one mesh node")
    boundary_nodes = {NOTCH_TIP: tip_nodes}
    audits = _audit_zero_width_slit(
        mesh=mesh,
        nodes=nodes,
        elements=elements,
        boundary_facets=boundary_facets,
        boundary_nodes=boundary_nodes,
        interface_facets=interface_facets,
        plan=plan,
    )
    topology_sha256 = _canonical_mesh_sha256(
        nodes=nodes,
        elements=elements,
        mesh=mesh,
        boundary_facets=boundary_facets,
        boundary_nodes=boundary_nodes,
        physical_tags=_PHYSICAL_TAGS,
        physical_entity_tags=physical_entity_tags,
    )

    facet_markers = np.zeros(mesh.facets.shape[1], dtype=np.int32)
    for label in BOUNDARY_LABELS:
        facet_markers[boundary_facets[label]] = _PHYSICAL_TAGS[label]
    cell_markers = np.full(elements.shape[0], _PHYSICAL_TAGS[BULK], dtype=np.int32)

    identity = MappingProxyType(
        {
            "schema": "tunnelgeopt-fracture-benchmark-mesh-v1",
            "coordinate_order": ("y", "z"),
            "coordinate_unit": "mm",
            "domain": "unit_square_with_zero_width_left_edge_slit",
            "domain_bounds_yz_mm": (0.0, 1.0, 0.0, 1.0),
            "slit_y_mm": 0.5,
            "slit_z_interval_mm": (0.0, 0.5),
            "slit_topology": "distinct_coincident_faces_shared_tip_only",
            "physical_entity_dimensions": MappingProxyType(
                {
                    BULK: 2,
                    **{label: 1 for label in BOUNDARY_LABELS},
                    NOTCH_TIP: 0,
                }
            ),
            "physical_tags": MappingProxyType(dict(_PHYSICAL_TAGS)),
            "physical_entity_tags": MappingProxyType(dict(physical_entity_tags)),
            "plan_sha256": plan.plan_sha256,
            "topology_sha256": topology_sha256,
        }
    )
    metadata = MappingProxyType(
        {
            "generator": "gmsh-python-api",
            "gmsh_version": str(getattr(gmsh, "__version__", "unknown")),
            "element_order": 1,
            "loading": plan.loading,
            "tier": plan.tier,
            "target_h_mm": plan.target_h_mm,
            "farfield_h_mm": plan.farfield_h_mm,
            "gmsh_target_h_mm": gmsh_target,
            "gmsh_target_factor": _GMSH_TARGET_FACTOR,
            "notch_polyline_yz_mm": plan.notch_polyline_yz_mm,
            "notch_band_half_width_mm": plan.notch_band_half_width_mm,
            "propagation_corridor_polyline_yz_mm": (plan.propagation_corridor_polyline_yz_mm),
            "propagation_corridor_half_width_mm": (plan.propagation_corridor_half_width_mm),
            "corridor_transition_mm": plan.corridor_transition_mm,
            "node_count": int(nodes.shape[0]),
            "element_count": int(elements.shape[0]),
            "boundary_facet_labels": BOUNDARY_LABELS,
            "boundary_node_labels": BOUNDARY_NODE_LABELS,
            "notch_tip_node_count": 1,
            "notch_tip_node_index": int(tip_nodes[0]),
            "notch_tip_physical_tag": _PHYSICAL_TAGS[NOTCH_TIP],
            "notch_tip_gmsh_entity_tag": physical_entity_tags[NOTCH_TIP][0],
            **audits,
        }
    )

    readonly_boundaries: dict[str, IntArray] = {}
    for label in BOUNDARY_LABELS:
        readonly_boundaries[label] = _readonly_array(boundary_facets[label], dtype=np.int64)
    readonly_nodes = _readonly_array(nodes, dtype=np.float64)
    readonly_elements = _readonly_array(elements, dtype=np.int64)
    readonly_facet_markers = _readonly_array(facet_markers, dtype=np.int32)
    readonly_cell_markers = _readonly_array(cell_markers, dtype=np.int32)
    _freeze_meshtri(mesh)

    return FractureBenchmarkMesh(
        mesh=mesh,
        nodes=readonly_nodes,
        elements=readonly_elements,
        boundary_facets=MappingProxyType(readonly_boundaries),
        boundary_nodes=MappingProxyType(
            {NOTCH_TIP: _readonly_array(boundary_nodes[NOTCH_TIP], dtype=np.int64)}
        ),
        facet_markers=readonly_facet_markers,
        cell_markers=readonly_cell_markers,
        physical_tags=_PHYSICAL_TAGS,
        physical_entity_tags=MappingProxyType(physical_entity_tags),
        plan=plan,
        identity=identity,
        metadata=metadata,
    )


__all__ = [
    "BOTTOM",
    "BOUNDARY_LABELS",
    "BOUNDARY_NODE_LABELS",
    "BULK",
    "LEFT_LOWER",
    "LEFT_UPPER",
    "LOADING_MODES",
    "MESH_TIERS",
    "NOTCH_LOWER",
    "NOTCH_TIP",
    "NOTCH_UPPER",
    "PHYSICAL_LABELS",
    "RIGHT",
    "TOP",
    "FractureBenchmarkMesh",
    "FractureBenchmarkMeshPlan",
    "benchmark_mesh_plan",
    "generate_fracture_benchmark_mesh",
]
