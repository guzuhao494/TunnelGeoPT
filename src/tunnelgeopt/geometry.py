"""Procedural 2-D tunnel cross-sections embedded in GeoPT-compatible 3-D coordinates.

The cross-section coordinate order is ``(y, z)``.  The tunnel axis is ``x``;
therefore a cross-section point is embedded as ``(0, y, z)`` when serialized.
The routines are deliberately dependency-light so the geometry/data contract can
be validated before a high-fidelity fracture solver is installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


_SHAPE_PARAMETER_SPECS: Mapping[str, Mapping[str, tuple[float, float, float]]] = MappingProxyType(
    {
        "circle": MappingProxyType(
            {
                # name: (default, inclusive lower bound, inclusive upper bound)
                "axis_ratio": (1.0, 0.65, 1.55),
                "superellipse_exponent": (2.0, 1.6, 4.0),
            }
        ),
        "horseshoe": MappingProxyType(
            {
                "span_height_ratio": (0.82, 0.65, 1.05),
                "sidewall_height_ratio": (1.0, 0.75, 1.25),
                "crown_shape": (2.0, 1.6, 3.5),
            }
        ),
        "straight_wall_arch": MappingProxyType(
            {
                "span_height_ratio": (1.0, 0.75, 1.25),
                "springline_height_ratio": (0.2, -0.1, 0.45),
                "crown_rise_span": (0.8, 0.55, 1.1),
            }
        ),
    }
)


@dataclass(frozen=True)
class TunnelGeometry:
    """One closed tunnel cavity boundary in the ``y-z`` cross-section."""

    shape: str
    boundary_yz: FloatArray
    characteristic_radius: float
    roughness_amplitude: float
    seed: int
    shape_parameters: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        boundary = np.asarray(self.boundary_yz)
        if boundary.ndim != 2 or boundary.shape[1] != 2 or boundary.shape[0] < 8:
            raise ValueError("boundary_yz must have shape [N, 2] with N >= 8")
        if not np.isfinite(boundary).all():
            raise ValueError("boundary_yz contains non-finite values")
        if self.characteristic_radius <= 0:
            raise ValueError("characteristic_radius must be positive")
        if not np.isfinite(self.characteristic_radius):
            raise ValueError("characteristic_radius must be finite")
        if not np.isfinite(self.roughness_amplitude):
            raise ValueError("roughness_amplitude must be finite")
        if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        parameters = {str(name): float(value) for name, value in self.shape_parameters.items()}
        if not np.isfinite(list(parameters.values())).all():
            raise ValueError("shape_parameters contain a non-finite value")
        # Keep this a plain dict so ``dataclasses.asdict`` and JSON-oriented
        # dataset writers remain backward-compatible with TunnelGeometry.
        object.__setattr__(self, "shape_parameters", parameters)


def shape_parameter_bounds(shape: str) -> dict[str, tuple[float, float]]:
    """Return inclusive, dimensionless parameter bounds for one shape family."""

    if shape not in _SHAPE_PARAMETER_SPECS:
        raise ValueError(
            f"unknown shape {shape!r}; expected circle, horseshoe, or straight_wall_arch"
        )
    return {
        name: (float(lower), float(upper))
        for name, (_, lower, upper) in _SHAPE_PARAMETER_SPECS[shape].items()
    }


def canonical_shape_parameters(shape: str) -> dict[str, float]:
    """Return the parameters reproducing the historical normalized boundary."""

    if shape not in _SHAPE_PARAMETER_SPECS:
        raise ValueError(
            f"unknown shape {shape!r}; expected circle, horseshoe, or straight_wall_arch"
        )
    return {name: float(default) for name, (default, _, _) in _SHAPE_PARAMETER_SPECS[shape].items()}


def _validated_shape_parameters(
    shape: str, parameters: Mapping[str, float] | None
) -> dict[str, float]:
    resolved = canonical_shape_parameters(shape)
    if parameters is None:
        return resolved
    unknown = set(parameters) - set(resolved)
    if unknown:
        expected = ", ".join(resolved)
        raise ValueError(
            f"unknown {shape!r} shape parameter(s) {sorted(unknown)}; expected {expected}"
        )
    bounds = shape_parameter_bounds(shape)
    for name, raw_value in parameters.items():
        if not isinstance(raw_value, Real) or isinstance(raw_value, bool):
            raise TypeError(f"shape parameter {name!r} must be a real number")
        value = float(raw_value)
        lower, upper = bounds[name]
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(
                f"shape parameter {name!r} must be finite and lie in [{lower}, {upper}]"
            )
        resolved[name] = value
    return resolved


def _drop_consecutive_duplicates(points: FloatArray) -> FloatArray:
    delta = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.r_[True, delta > 1e-12]
    return points[keep]


def resample_closed_polyline(points: FloatArray, n_points: int) -> FloatArray:
    """Resample a closed polygon at approximately equal arc-length spacing."""

    if n_points < 8:
        raise ValueError("n_points must be at least 8")
    points = _drop_consecutive_duplicates(np.asarray(points, dtype=np.float64))
    if np.linalg.norm(points[0] - points[-1]) < 1e-12:
        points = points[:-1]
    closed = np.vstack([points, points[0]])
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("boundary contains a zero-length segment")
    cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
    targets = np.linspace(0.0, cumulative[-1], n_points, endpoint=False)
    segment_ids = np.searchsorted(cumulative, targets, side="right") - 1
    local = (targets - cumulative[segment_ids]) / segment_lengths[segment_ids]
    return closed[segment_ids] + local[:, None] * (closed[segment_ids + 1] - closed[segment_ids])


def _superellipse_coordinate(values: FloatArray, exponent: float) -> FloatArray:
    """Map sine/cosine samples to a signed superellipse coordinate."""

    return np.sign(values) * np.abs(values) ** (2.0 / exponent)


def _base_boundary(
    shape: str, n_points: int, parameters: Mapping[str, float] | None = None
) -> FloatArray:
    parameters = _validated_shape_parameters(shape, parameters)
    if shape == "circle":
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        axis_ratio = parameters["axis_ratio"]
        exponent = parameters["superellipse_exponent"]
        vertical_scale = np.sqrt(axis_ratio)
        horizontal_scale = 1.0 / vertical_scale
        return np.column_stack(
            [
                vertical_scale * _superellipse_coordinate(np.sin(theta), exponent),
                horizontal_scale * _superellipse_coordinate(np.cos(theta), exponent),
            ]
        )

    if shape == "horseshoe":
        width = parameters["span_height_ratio"]
        spring_y = -1.0 + parameters["sidewall_height_ratio"]
        crown_shape = parameters["crown_shape"]
        bottom = np.column_stack(
            [np.full(32, -1.0), np.linspace(-width, width, 32, endpoint=False)]
        )
        right_wall = np.column_stack(
            [np.linspace(-1.0, spring_y, 24, endpoint=False), np.full(24, width)]
        )
        theta = np.linspace(0.0, np.pi, 96, endpoint=False)
        arch = np.column_stack(
            [
                spring_y + width * _superellipse_coordinate(np.sin(theta), crown_shape),
                width * _superellipse_coordinate(np.cos(theta), crown_shape),
            ]
        )
        left_wall = np.column_stack(
            [np.linspace(spring_y, -1.0, 24, endpoint=False), np.full(24, -width)]
        )
        return resample_closed_polyline(np.vstack([bottom, right_wall, arch, left_wall]), n_points)

    if shape == "straight_wall_arch":
        width = parameters["span_height_ratio"]
        spring_y = parameters["springline_height_ratio"]
        crown_rise = width * parameters["crown_rise_span"]
        bottom = np.column_stack(
            [np.full(36, -1.0), np.linspace(-width, width, 36, endpoint=False)]
        )
        right_wall = np.column_stack(
            [np.linspace(-1.0, spring_y, 28, endpoint=False), np.full(28, width)]
        )
        theta = np.linspace(0.0, np.pi, 96, endpoint=False)
        arch = np.column_stack([spring_y + crown_rise * np.sin(theta), width * np.cos(theta)])
        left_wall = np.column_stack(
            [np.linspace(spring_y, -1.0, 28, endpoint=False), np.full(28, -width)]
        )
        return resample_closed_polyline(np.vstack([bottom, right_wall, arch, left_wall]), n_points)

    raise ValueError(f"unknown shape {shape!r}; expected circle, horseshoe, or straight_wall_arch")


def _segments_intersect(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray) -> bool:
    scale = max(float(np.ptp(np.vstack([a, b, c, d]), axis=0).max()), 1.0)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale**2

    def orient(p: FloatArray, q: FloatArray, r: FloatArray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def on_segment(p: FloatArray, q: FloatArray, r: FloatArray) -> bool:
        return bool(
            min(p[0], r[0]) - tolerance <= q[0] <= max(p[0], r[0]) + tolerance
            and min(p[1], r[1]) - tolerance <= q[1] <= max(p[1], r[1]) + tolerance
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if ((o1 > tolerance and o2 < -tolerance) or (o1 < -tolerance and o2 > tolerance)) and (
        (o3 > tolerance and o4 < -tolerance) or (o3 < -tolerance and o4 > tolerance)
    ):
        return True
    return bool(
        (abs(o1) <= tolerance and on_segment(a, c, b))
        or (abs(o2) <= tolerance and on_segment(a, d, b))
        or (abs(o3) <= tolerance and on_segment(c, a, d))
        or (abs(o4) <= tolerance and on_segment(c, b, d))
    )


def _validate_simple_closed_boundary(boundary: FloatArray) -> None:
    if not np.isfinite(boundary).all():
        raise ValueError("generated boundary contains a non-finite value")
    edges = np.roll(boundary, -1, axis=0) - boundary
    scale = max(float(np.ptp(boundary, axis=0).max()), 1.0)
    if np.any(np.linalg.norm(edges, axis=1) <= 64.0 * np.finfo(float).eps * scale):
        raise ValueError("generated boundary contains a zero-length segment")
    signed_area = 0.5 * float(
        np.sum(
            boundary[:, 0] * np.roll(boundary[:, 1], -1)
            - np.roll(boundary[:, 0], -1) * boundary[:, 1]
        )
    )
    if abs(signed_area) <= 64.0 * np.finfo(float).eps * scale**2:
        raise ValueError("generated boundary has zero signed area")

    n_points = boundary.shape[0]
    for first in range(n_points):
        a = boundary[first]
        b = boundary[(first + 1) % n_points]
        for second in range(first + 1, n_points):
            if second == first or second == first + 1 or (first == 0 and second == n_points - 1):
                continue
            c = boundary[second]
            d = boundary[(second + 1) % n_points]
            if _segments_intersect(a, b, c, d):
                raise ValueError(
                    f"generated boundary self-intersects at segments {first} and {second}"
                )


def make_tunnel_boundary(
    shape: str,
    *,
    n_points: int = 256,
    radius: float = 1.0,
    roughness_amplitude: float = 0.0,
    seed: int = 0,
) -> TunnelGeometry:
    """Create one normalized tunnel cavity boundary.

    ``roughness_amplitude`` is a dimensionless radial perturbation relative to
    the characteristic radius.  It is a geometry augmentation, not a calibrated
    representation of field overbreak.

    This historical API delegates to :func:`make_parametric_tunnel_boundary`
    with the canonical parameter set, preserving its boundary convention.
    """

    return make_parametric_tunnel_boundary(
        shape,
        n_points=n_points,
        radius=radius,
        roughness_amplitude=roughness_amplitude,
        seed=seed,
    )


def make_parametric_tunnel_boundary(
    shape: str,
    *,
    parameters: Mapping[str, float] | None = None,
    n_points: int = 256,
    radius: float = 1.0,
    roughness_amplitude: float = 0.0,
    seed: int = 0,
) -> TunnelGeometry:
    """Create a tunnel boundary from validated continuous macro-parameters.

    ``parameters`` may contain any subset of :func:`canonical_shape_parameters`;
    unspecified entries retain their canonical values.  The accepted intervals
    are returned by :func:`shape_parameter_bounds`.  Bounds are deliberately
    conservative so every supported combination remains a finite, simple cavity.
    """

    if not isinstance(radius, Real) or isinstance(radius, bool):
        raise TypeError("radius must be a real number")
    radius = float(radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not isinstance(roughness_amplitude, Real) or isinstance(roughness_amplitude, bool):
        raise TypeError("roughness_amplitude must be a real number")
    roughness_amplitude = float(roughness_amplitude)
    if not np.isfinite(roughness_amplitude) or not 0.0 <= roughness_amplitude <= 0.08:
        raise ValueError("roughness_amplitude must be finite and lie in [0, 0.08]")
    if not isinstance(n_points, Integral) or isinstance(n_points, bool) or n_points < 8:
        raise ValueError("n_points must be an integer of at least 8")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    n_points = int(n_points)
    seed = int(seed)
    resolved_parameters = _validated_shape_parameters(shape, parameters)
    boundary = _base_boundary(shape, n_points, resolved_parameters).astype(np.float64)
    if roughness_amplitude:
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        perturbation = roughness_amplitude * (
            0.6 * np.sin(3.0 * t + phase) + 0.4 * np.sin(7.0 * t - phase)
        )
        centroid = boundary.mean(axis=0)
        boundary = centroid + (boundary - centroid) * (1.0 + perturbation[:, None])
        boundary = resample_closed_polyline(boundary, n_points)
    boundary *= radius
    _validate_simple_closed_boundary(boundary)
    return TunnelGeometry(
        shape=shape,
        boundary_yz=boundary,
        characteristic_radius=radius,
        roughness_amplitude=roughness_amplitude,
        seed=seed,
        shape_parameters=resolved_parameters,
    )


def points_inside_polygon(points_yz: FloatArray, boundary_yz: FloatArray) -> NDArray[np.bool_]:
    """Vectorized even-odd containment test for a closed polygon."""

    points = np.asarray(points_yz, dtype=np.float64)
    boundary = np.asarray(boundary_yz, dtype=np.float64)
    a = points[:, 0]
    b = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    a0, b0 = boundary[-1]
    for a1, b1 in boundary:
        crosses = (b1 > b) != (b0 > b)
        denominator = b0 - b1
        safe_denominator = denominator if abs(denominator) > 1e-15 else 1e-15
        a_intersection = (a0 - a1) * (b - b1) / safe_denominator + a1
        inside ^= crosses & (a < a_intersection)
        a0, b0 = a1, b1
    return inside


def sample_rock_points(
    geometry: TunnelGeometry,
    n_points: int,
    *,
    domain_scale: float = 3.0,
    seed: int = 0,
) -> FloatArray:
    """Uniformly sample the rock domain: finite box minus the tunnel cavity."""

    if n_points <= 0:
        raise ValueError("n_points must be positive")
    if domain_scale <= 1.1:
        raise ValueError("domain_scale must be greater than 1.1")
    rng = np.random.default_rng(seed)
    boundary = geometry.boundary_yz
    center = 0.5 * (boundary.min(axis=0) + boundary.max(axis=0))
    half_extent = 0.5 * (boundary.max(axis=0) - boundary.min(axis=0)) * domain_scale
    accepted: list[FloatArray] = []
    total = 0
    for _ in range(100):
        batch_size = max(1024, 2 * (n_points - total))
        batch = rng.uniform(center - half_extent, center + half_extent, size=(batch_size, 2))
        outside = batch[~points_inside_polygon(batch, boundary)]
        accepted.append(outside)
        total += outside.shape[0]
        if total >= n_points:
            return np.vstack(accepted)[:n_points]
    raise RuntimeError(f"could not sample {n_points} rock points after 100 batches")


def nearest_boundary_vectors(
    points_yz: FloatArray,
    boundary_yz: FloatArray,
    *,
    chunk_size: int = 4096,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return distance, unit vector ``point-nearest``, and nearest boundary point."""

    points = np.asarray(points_yz, dtype=np.float64)
    boundary = np.asarray(boundary_yz, dtype=np.float64)
    p0 = boundary
    p1 = np.roll(boundary, -1, axis=0)
    segments = p1 - p0
    length_sq = np.sum(segments * segments, axis=1)
    distances_out: list[FloatArray] = []
    directions_out: list[FloatArray] = []
    nearest_out: list[FloatArray] = []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        relative = chunk[:, None, :] - p0[None, :, :]
        projection = np.sum(relative * segments[None, :, :], axis=2) / length_sq[None, :]
        projection = np.clip(projection, 0.0, 1.0)
        candidates = p0[None, :, :] + projection[:, :, None] * segments[None, :, :]
        delta = chunk[:, None, :] - candidates
        distance_sq = np.sum(delta * delta, axis=2)
        indices = np.argmin(distance_sq, axis=1)
        nearest = candidates[np.arange(chunk.shape[0]), indices]
        vector = chunk - nearest
        distance = np.linalg.norm(vector, axis=1)
        direction = vector / np.maximum(distance[:, None], 1e-12)
        distances_out.append(distance)
        directions_out.append(direction)
        nearest_out.append(nearest)
    return (
        np.concatenate(distances_out),
        np.vstack(directions_out),
        np.vstack(nearest_out),
    )


def surface_points_and_normals(
    geometry: TunnelGeometry, n_points: int
) -> tuple[FloatArray, FloatArray]:
    """Sample cavity-wall points and outward normals pointing into the rock."""

    points = resample_closed_polyline(geometry.boundary_yz, n_points)
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
    signed_area = 0.5 * np.sum(
        points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1]
    )
    if signed_area > 0:
        normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    else:
        normals = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return points, normals


def embed_yz(points_yz: FloatArray) -> FloatArray:
    """Embed ``(y,z)`` cross-section coordinates as ``(x,y,z)`` with ``x=0``."""

    points = np.asarray(points_yz)
    return np.column_stack([np.zeros(points.shape[0]), points])
