"""Gmsh-backed two-dimensional tunnel meshes for the B-elastic layer.

Coordinates are ordered ``(y, z)`` throughout this module.  The meshed domain
is a finite rectangle of rock with one closed tunnel cross-section removed.
The two boundary labels are part of the public contract:

``wall``
    The excavation boundary.  Its outward normal with respect to the *rock*
    domain points into the cavity.
``farfield``
    All four sides of the exterior rectangle.

Gmsh and scikit-fem are imported lazily so that the dependency-light A-layer
can still be imported when the optional B-layer solver stack is unavailable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import TunnelGeometry

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]

WALL = "wall"
FARFIELD = "farfield"
ROCK = "rock"

# The Gmsh Python API owns process-global model state and is not thread-safe.
_GMSH_LOCK = Lock()

# Gmsh's size field is a target rather than a mathematical upper bound.  The
# public near-field value below is therefore audited against the generated
# connectivity with a small, documented allowance for mesher discretization.
_NEARFIELD_AUDIT_RELATIVE_TOLERANCE = 0.02
# Empirically conservative conversion from the public maximum-edge contract to
# Gmsh's characteristic-length target.  The generated connectivity remains the
# authority: exceeding the public cap still raises instead of being accepted.
_NEARFIELD_GMSH_TARGET_FACTOR = 0.5


@dataclass(frozen=True)
class TunnelMesh:
    """A first-order triangular rock mesh and its explicit physical markers."""

    mesh: Any
    nodes: FloatArray
    elements: IntArray
    boundary_facets: Mapping[str, IntArray]
    facet_markers: IntArray
    cell_markers: IntArray
    physical_tags: Mapping[str, int]
    outer_bounds: tuple[float, float, float, float]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes)
        elements = np.asarray(self.elements)
        if nodes.ndim != 2 or nodes.shape[1] != 2:
            raise ValueError("nodes must have shape [N, 2] in (y, z) order")
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError("elements must have shape [M, 3]")
        if not np.isfinite(nodes).all():
            raise ValueError("nodes contain non-finite values")
        if elements.size and (elements.min() < 0 or elements.max() >= nodes.shape[0]):
            raise ValueError("elements contain an out-of-range node index")
        required = {WALL, FARFIELD}
        if set(self.boundary_facets) != required:
            raise ValueError(f"boundary_facets must contain exactly {sorted(required)}")

    @property
    def skfem_mesh(self) -> Any:
        """Alias that makes the wrapped scikit-fem object explicit."""

        return self.mesh

    @property
    def nodes_yz(self) -> FloatArray:
        return self.nodes

    @property
    def triangles(self) -> IntArray:
        return self.elements


def _require_mesh_dependencies() -> tuple[Any, Any]:
    try:
        import gmsh  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without optional stack
        raise RuntimeError(
            "Tunnel meshing requires gmsh 4.x; install the B-elastic dependencies first"
        ) from exc
    try:
        from skfem import MeshTri  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only without optional stack
        raise RuntimeError(
            "Tunnel meshing requires scikit-fem 12.x; install the B-elastic dependencies first"
        ) from exc
    return gmsh, MeshTri


def _coerce_boundary(geometry: TunnelGeometry | ArrayLike) -> tuple[FloatArray, float]:
    if isinstance(geometry, TunnelGeometry):
        boundary = np.asarray(geometry.boundary_yz, dtype=np.float64)
        characteristic_radius = float(geometry.characteristic_radius)
    else:
        boundary = np.asarray(geometry, dtype=np.float64)
        if boundary.ndim != 2 or boundary.shape[1] != 2:
            raise ValueError("boundary must have shape [N, 2]")
        extent = np.ptp(boundary, axis=0)
        characteristic_radius = 0.5 * float(np.max(extent))

    if boundary.ndim != 2 or boundary.shape[1] != 2 or boundary.shape[0] < 8:
        raise ValueError("boundary must have shape [N, 2] with N >= 8")
    if not np.isfinite(boundary).all():
        raise ValueError("boundary contains non-finite values")
    if np.linalg.norm(boundary[0] - boundary[-1]) <= 1e-12:
        boundary = boundary[:-1]
    segment_lengths = np.linalg.norm(np.roll(boundary, -1, axis=0) - boundary, axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("boundary contains a zero-length segment")
    signed_area = _signed_polygon_area(boundary)
    if abs(signed_area) <= np.finfo(float).eps * max(characteristic_radius**2, 1.0):
        raise ValueError("boundary has zero signed area")

    # A hole loop is clockwise in the y-z plane; this also agrees with the
    # TunnelGeometry convention.  Reorient arbitrary user input deterministically.
    if signed_area > 0.0:
        boundary = boundary[::-1].copy()
    return boundary, characteristic_radius


def _signed_polygon_area(points: FloatArray) -> float:
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
    )


def _resolve_outer_bounds(
    boundary: FloatArray,
    *,
    domain_scale: float,
    outer_bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if outer_bounds is None:
        if not np.isfinite(domain_scale) or domain_scale <= 1.05:
            raise ValueError("domain_scale must be finite and greater than 1.05")
        lower = boundary.min(axis=0)
        upper = boundary.max(axis=0)
        center = 0.5 * (lower + upper)
        half_extent = 0.5 * (upper - lower) * float(domain_scale)
        bounds = (
            float(center[0] - half_extent[0]),
            float(center[0] + half_extent[0]),
            float(center[1] - half_extent[1]),
            float(center[1] + half_extent[1]),
        )
    else:
        if len(outer_bounds) != 4:
            raise ValueError("outer_bounds must be (y_min, y_max, z_min, z_max)")
        bounds = tuple(float(value) for value in outer_bounds)
        if not np.isfinite(bounds).all():
            raise ValueError("outer_bounds contains a non-finite value")

    y_min, y_max, z_min, z_max = bounds
    if not y_min < y_max or not z_min < z_max:
        raise ValueError("outer_bounds minima must be smaller than maxima")
    tolerance = 1e-10 * max(y_max - y_min, z_max - z_min, 1.0)
    if (
        boundary[:, 0].min() <= y_min + tolerance
        or boundary[:, 0].max() >= y_max - tolerance
        or boundary[:, 1].min() <= z_min + tolerance
        or boundary[:, 1].max() >= z_max - tolerance
    ):
        raise ValueError("the tunnel boundary must lie strictly inside outer_bounds")
    return bounds


def _triangle_quality(nodes: FloatArray, elements: IntArray) -> tuple[FloatArray, FloatArray]:
    triangles = nodes[elements]
    twice_signed_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    area = 0.5 * twice_signed_area
    edge_sq = np.sum((triangles - np.roll(triangles, -1, axis=1)) ** 2, axis=2)
    quality = 4.0 * np.sqrt(3.0) * area / np.maximum(edge_sq.sum(axis=1), 1e-300)
    return area, quality


def _point_to_polyline_distance(points: FloatArray, polyline: FloatArray) -> FloatArray:
    """Return minimum Euclidean distance to a closed piecewise-linear curve."""

    starts = polyline
    ends = np.roll(polyline, -1, axis=0)
    directions = ends - starts
    length_sq = np.sum(directions**2, axis=1)
    distances_sq = np.empty(points.shape[0], dtype=np.float64)
    # Chunk both axes so auditing a production mesh does not allocate an array
    # proportional to every cell times every wall segment.
    for point_first in range(0, points.shape[0], 16_384):
        point_chunk = points[point_first : point_first + 16_384]
        point_distances_sq = np.full(point_chunk.shape[0], np.inf, dtype=np.float64)
        for segment_first in range(0, starts.shape[0], 64):
            segment_start = starts[segment_first : segment_first + 64]
            segment_direction = directions[segment_first : segment_first + 64]
            segment_length_sq = length_sq[segment_first : segment_first + 64]
            offsets = point_chunk[:, None, :] - segment_start[None, :, :]
            projection = np.sum(offsets * segment_direction[None, :, :], axis=2)
            projection /= segment_length_sq[None, :]
            projection = np.clip(projection, 0.0, 1.0)
            closest = (
                segment_start[None, :, :] + projection[:, :, None] * segment_direction[None, :, :]
            )
            chunk_distance_sq = np.sum((point_chunk[:, None, :] - closest) ** 2, axis=2)
            point_distances_sq = np.minimum(
                point_distances_sq,
                chunk_distance_sq.min(axis=1),
            )
        distances_sq[point_first : point_first + point_chunk.shape[0]] = point_distances_sq
    return np.sqrt(distances_sq)


def _resolve_nearfield_parameters(
    *,
    nearfield_distance: float | None,
    nearfield_mesh_size: float | None,
    fracture_length_scale: float | None,
    nearfield_transition_width: float | None,
) -> tuple[float, float, float, float] | None:
    supplied = (
        nearfield_distance,
        nearfield_mesh_size,
        fracture_length_scale,
        nearfield_transition_width,
    )
    if all(value is None for value in supplied):
        return None
    if any(value is None for value in supplied[:3]):
        raise ValueError(
            "nearfield_distance, nearfield_mesh_size, and fracture_length_scale "
            "must be supplied together"
        )

    assert nearfield_distance is not None
    assert nearfield_mesh_size is not None
    assert fracture_length_scale is not None

    distance = float(nearfield_distance)
    size = float(nearfield_mesh_size)
    length_scale = float(fracture_length_scale)
    transition = (
        distance if nearfield_transition_width is None else float(nearfield_transition_width)
    )
    if not all(
        np.isfinite(value) and value > 0.0 for value in (distance, size, length_scale, transition)
    ):
        raise ValueError("all near-field parameters must be finite and positive")
    return distance, size, length_scale, transition


def _facets_from_edges(mesh: Any, edges: IntArray, *, label: str) -> IntArray:
    facet_lookup = {
        (int(edge[0]), int(edge[1])): index
        for index, edge in enumerate(np.sort(np.asarray(mesh.facets).T, axis=1))
    }
    indices: list[int] = []
    for edge in np.sort(np.asarray(edges, dtype=np.int64), axis=1):
        key = (int(edge[0]), int(edge[1]))
        if key not in facet_lookup:
            raise RuntimeError(f"Gmsh {label!r} edge {key} is absent from the triangle facets")
        indices.append(facet_lookup[key])
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def _extract_first_order_elements(
    gmsh: Any,
    *,
    dim: int,
    entity_tags: list[int],
    node_index: Mapping[int, int],
) -> IntArray:
    expected_nodes = 3 if dim == 2 else 2
    blocks: list[NDArray[np.int64]] = []
    for entity_tag in entity_tags:
        element_types, _, connectivity_blocks = gmsh.model.mesh.getElements(dim, entity_tag)
        for element_type, connectivity in zip(element_types, connectivity_blocks, strict=True):
            _, element_dim, order, num_nodes, _, num_primary = gmsh.model.mesh.getElementProperties(
                int(element_type)
            )
            if element_dim != dim or order != 1 or num_nodes != expected_nodes:
                continue
            if num_primary != expected_nodes:
                continue
            tags = np.asarray(connectivity, dtype=np.int64).reshape(-1, expected_nodes)
            try:
                local = np.fromiter(
                    (node_index[int(tag)] for tag in tags.ravel()),
                    dtype=np.int64,
                    count=tags.size,
                ).reshape(tags.shape)
            except KeyError as exc:
                raise RuntimeError("Gmsh returned connectivity for an unknown node tag") from exc
            blocks.append(local)
    if not blocks:
        kind = "triangles" if dim == 2 else "boundary edges"
        raise RuntimeError(f"Gmsh did not generate first-order {kind}")
    return np.vstack(blocks)


def generate_tunnel_mesh(
    geometry: TunnelGeometry | ArrayLike,
    *,
    domain_scale: float = 5.0,
    outer_bounds: tuple[float, float, float, float] | None = None,
    mesh_size: float | None = None,
    wall_mesh_size: float | None = None,
    farfield_mesh_size: float | None = None,
    nearfield_distance: float | None = None,
    nearfield_mesh_size: float | None = None,
    fracture_length_scale: float | None = None,
    nearfield_transition_width: float | None = None,
    gmsh_algorithm: int = 6,
    verbose: bool = False,
) -> TunnelMesh:
    """Mesh a rectangle of rock with exactly one polygonal tunnel cavity.

    Parameters
    ----------
    geometry:
        A :class:`~tunnelgeopt.geometry.TunnelGeometry` or an ``[N, 2]`` closed
        polygon in ``(y, z)`` order.  A repeated final point is accepted and
        removed.
    domain_scale:
        Exterior half-width divided by the corresponding cavity half-width
        when ``outer_bounds`` is not supplied.
    outer_bounds:
        ``(y_min, y_max, z_min, z_max)``.  The cavity must be strictly inside.
    mesh_size, wall_mesh_size, farfield_mesh_size:
        Target first-order edge sizes.  The wall defaults to half the global
        size; no statement about high fidelity is implied by these defaults.
    nearfield_distance, nearfield_mesh_size, fracture_length_scale:
        Optional fracture-band contract.  All three must be supplied together.
        A Gmsh Distance/Threshold field constructs a conservative target from
        ``nearfield_mesh_size`` through the band, and the generated triangle
        edges are then audited against it as a hard upper bound with a
        two-percent mesher tolerance.  ``fracture_length_scale`` records the
        requested and realized ``h/ell`` ratios.
    nearfield_transition_width:
        Distance over which the background size ramps from the near-field size
        to the far-field target.  It defaults to ``nearfield_distance`` when the
        fracture-band contract is enabled.
    """

    gmsh, MeshTri = _require_mesh_dependencies()
    boundary, characteristic_radius = _coerce_boundary(geometry)
    bounds = _resolve_outer_bounds(boundary, domain_scale=domain_scale, outer_bounds=outer_bounds)
    base_size = (
        float(mesh_size)
        if mesh_size is not None
        else max(characteristic_radius / 3.0, np.finfo(float).eps)
    )
    wall_size = float(wall_mesh_size) if wall_mesh_size is not None else 0.5 * base_size
    farfield_size = float(farfield_mesh_size) if farfield_mesh_size is not None else base_size
    if not all(
        np.isfinite(value) and value > 0.0 for value in (base_size, wall_size, farfield_size)
    ):
        raise ValueError("all mesh sizes must be finite and positive")
    if not isinstance(gmsh_algorithm, int) or gmsh_algorithm <= 0:
        raise ValueError("gmsh_algorithm must be a positive integer")
    nearfield = _resolve_nearfield_parameters(
        nearfield_distance=nearfield_distance,
        nearfield_mesh_size=nearfield_mesh_size,
        fracture_length_scale=fracture_length_scale,
        nearfield_transition_width=nearfield_transition_width,
    )
    if nearfield is not None and nearfield[1] >= farfield_size:
        raise ValueError("nearfield_mesh_size must be smaller than farfield_mesh_size")

    model_name = f"tunnelgeopt_{uuid4().hex}"
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
            gmsh.option.setNumber("Mesh.ElementOrder", 1.0)
            gmsh.option.setNumber("Mesh.Algorithm", float(gmsh_algorithm))
            gmsh.model.add(model_name)
            model_added = True
            gmsh.model.setCurrent(model_name)

            y_min, y_max, z_min, z_max = bounds
            outer_coordinates = np.asarray(
                [[y_min, z_min], [y_max, z_min], [y_max, z_max], [y_min, z_max]],
                dtype=np.float64,
            )
            outer_points = [
                gmsh.model.geo.addPoint(float(y), float(z), 0.0, farfield_size)
                for y, z in outer_coordinates
            ]
            wall_points = [
                gmsh.model.geo.addPoint(float(y), float(z), 0.0, wall_size) for y, z in boundary
            ]
            outer_lines = [
                gmsh.model.geo.addLine(outer_points[i], outer_points[(i + 1) % 4]) for i in range(4)
            ]
            wall_lines = [
                gmsh.model.geo.addLine(wall_points[i], wall_points[(i + 1) % len(wall_points)])
                for i in range(len(wall_points))
            ]
            outer_loop = gmsh.model.geo.addCurveLoop(outer_lines)
            wall_loop = gmsh.model.geo.addCurveLoop(wall_lines)
            rock_surface = gmsh.model.geo.addPlaneSurface([outer_loop, wall_loop])
            gmsh.model.geo.synchronize()

            physical_tags = {ROCK: 1, WALL: 1, FARFIELD: 2}
            gmsh.model.addPhysicalGroup(2, [rock_surface], physical_tags[ROCK])
            gmsh.model.setPhysicalName(2, physical_tags[ROCK], ROCK)
            gmsh.model.addPhysicalGroup(1, wall_lines, physical_tags[WALL])
            gmsh.model.setPhysicalName(1, physical_tags[WALL], WALL)
            gmsh.model.addPhysicalGroup(1, outer_lines, physical_tags[FARFIELD])
            gmsh.model.setPhysicalName(1, physical_tags[FARFIELD], FARFIELD)
            if nearfield is not None:
                distance, size, _, transition = nearfield
                gmsh_target_size = size * _NEARFIELD_GMSH_TARGET_FACTOR
                distance_field = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(distance_field, "CurvesList", wall_lines)
                threshold_field = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold_field, "InField", distance_field)
                gmsh.model.mesh.field.setNumber(threshold_field, "SizeMin", gmsh_target_size)
                gmsh.model.mesh.field.setNumber(
                    threshold_field,
                    "SizeMax",
                    max(base_size, wall_size, farfield_size),
                )
                gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", distance)
                gmsh.model.mesh.field.setNumber(
                    threshold_field,
                    "DistMax",
                    distance + transition,
                )
                gmsh.model.mesh.field.setAsBackgroundMesh(threshold_field)
            gmsh.model.mesh.generate(2)

            node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
            node_tags = np.asarray(node_tags, dtype=np.int64)
            nodes = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
            if node_tags.size != nodes.shape[0] or node_tags.size == 0:
                raise RuntimeError("Gmsh returned an invalid node table")
            node_index = {int(tag): index for index, tag in enumerate(node_tags)}
            elements = _extract_first_order_elements(
                gmsh, dim=2, entity_tags=[rock_surface], node_index=node_index
            )
            wall_edges = _extract_first_order_elements(
                gmsh, dim=1, entity_tags=wall_lines, node_index=node_index
            )
            farfield_edges = _extract_first_order_elements(
                gmsh, dim=1, entity_tags=outer_lines, node_index=node_index
            )
        finally:
            if gmsh.isInitialized():
                if initialized_here:
                    gmsh.finalize()
                elif model_added:
                    # Remove only the temporary model and restore the caller's
                    # model when Gmsh had already been initialized externally.
                    gmsh.model.setCurrent(model_name)
                    gmsh.model.remove()
                    if previous_model is not None:
                        gmsh.model.setCurrent(previous_model)

    # Gmsh emits consistently counter-clockwise triangles.  Disabling MeshTri's
    # vertex-index sorting preserves that signed-area invariant; arbitrary
    # index sorting can flip otherwise valid elements.
    mesh = MeshTri(nodes.T, elements.T, validate=True, sort_t=False)
    wall_facets = _facets_from_edges(mesh, wall_edges, label=WALL)
    farfield_facets = _facets_from_edges(mesh, farfield_edges, label=FARFIELD)
    if np.intersect1d(wall_facets, farfield_facets).size:
        raise RuntimeError("wall and farfield facet markers overlap")
    marked_boundary = np.union1d(wall_facets, farfield_facets)
    actual_boundary = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    if not np.array_equal(np.sort(marked_boundary), np.sort(actual_boundary)):
        missing = np.setdiff1d(actual_boundary, marked_boundary)
        extra = np.setdiff1d(marked_boundary, actual_boundary)
        raise RuntimeError(
            f"boundary marker coverage failed (missing={missing.size}, extra={extra.size})"
        )
    mesh = mesh.with_boundaries({WALL: wall_facets, FARFIELD: farfield_facets})
    mesh = mesh.with_subdomains({ROCK: np.arange(elements.shape[0], dtype=np.int64)})

    # Retain the exact connectivity of the object consumed by scikit-fem in
    # every downstream result.
    elements = np.asarray(mesh.t.T, dtype=np.int64)
    nodes = np.asarray(mesh.p.T, dtype=np.float64)
    area, quality = _triangle_quality(nodes, elements)
    if area.size == 0 or np.any(area <= 0.0) or not np.isfinite(quality).all():
        raise RuntimeError("generated mesh contains a degenerate or non-finite triangle")

    nearfield_metadata: dict[str, Any] = {}
    if nearfield is not None:
        distance, size, length_scale, transition = nearfield
        triangles = nodes[elements]
        centroids = triangles.mean(axis=1)
        centroid_wall_distance = _point_to_polyline_distance(centroids, boundary)
        band_mask = centroid_wall_distance <= distance
        if not np.any(band_mask):
            raise RuntimeError("near-field audit selected no triangles")
        edge_lengths = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
        band_edge_max = float(edge_lengths[band_mask].max())
        absolute_tolerance = float(
            64.0
            * np.finfo(np.float64).eps
            * max(
                characteristic_radius,
                size,
                1.0,
            )
        )
        audit_limit = float(size * (1.0 + _NEARFIELD_AUDIT_RELATIVE_TOLERANCE) + absolute_tolerance)
        if band_edge_max > audit_limit:
            raise RuntimeError(
                "near-field maximum edge audit failed: "
                f"actual={band_edge_max:.17g}, limit={audit_limit:.17g}, "
                f"requested={size:.17g}"
            )
        nearfield_metadata = {
            "nearfield_enabled": True,
            "nearfield_distance": distance,
            "nearfield_mesh_size": size,
            "nearfield_gmsh_target_mesh_size": size * _NEARFIELD_GMSH_TARGET_FACTOR,
            "nearfield_gmsh_target_factor": _NEARFIELD_GMSH_TARGET_FACTOR,
            "fracture_length_scale": length_scale,
            "nearfield_transition_width": transition,
            "nearfield_transition_end_distance": distance + transition,
            "nearfield_audit_cell_selection": (
                "triangle_centroid_distance_to_input_wall_polyline_le_nearfield_distance"
            ),
            "nearfield_audited_element_count": int(np.count_nonzero(band_mask)),
            "nearfield_actual_maximum_edge": band_edge_max,
            "nearfield_requested_h_over_ell": size / length_scale,
            "nearfield_actual_maximum_h_over_ell": band_edge_max / length_scale,
            "nearfield_audit_relative_tolerance": _NEARFIELD_AUDIT_RELATIVE_TOLERANCE,
            "nearfield_audit_absolute_tolerance": absolute_tolerance,
            "nearfield_audit_limit": audit_limit,
            "nearfield_audit_passed": True,
            "nearfield_gmsh_fields": ["Distance", "Threshold"],
        }

    facet_markers = np.zeros(mesh.facets.shape[1], dtype=np.int32)
    facet_markers[wall_facets] = physical_tags[WALL]
    facet_markers[farfield_facets] = physical_tags[FARFIELD]
    cell_markers = np.full(elements.shape[0], physical_tags[ROCK], dtype=np.int32)
    metadata: dict[str, Any] = {
        "generator": "gmsh-python-api",
        "gmsh_version": str(getattr(gmsh, "__version__", "unknown")),
        "element_order": 1,
        "coordinate_order": ["y", "z"],
        "domain": "outer_rectangle_minus_single_closed_cavity",
        "node_count": int(nodes.shape[0]),
        "element_count": int(elements.shape[0]),
        "wall_facet_count": int(wall_facets.size),
        "farfield_facet_count": int(farfield_facets.size),
        "minimum_element_area": float(area.min()),
        "minimum_triangle_quality": float(quality.min()),
        "mesh_size": base_size,
        "wall_mesh_size": wall_size,
        "farfield_mesh_size": farfield_size,
        "gmsh_algorithm": gmsh_algorithm,
    }
    metadata.update(nearfield_metadata)
    boundary_facets = {WALL: wall_facets, FARFIELD: farfield_facets}
    return TunnelMesh(
        mesh=mesh,
        nodes=nodes,
        elements=elements,
        boundary_facets=boundary_facets,
        facet_markers=facet_markers,
        cell_markers=cell_markers,
        physical_tags=physical_tags,
        outer_bounds=bounds,
        metadata=metadata,
    )


__all__ = [
    "FARFIELD",
    "ROCK",
    "WALL",
    "TunnelMesh",
    "generate_tunnel_mesh",
]
