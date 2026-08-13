"""Procedural 2-D tunnel cross-sections embedded in GeoPT-compatible 3-D coordinates.

The cross-section coordinate order is ``(y, z)``.  The tunnel axis is ``x``;
therefore a cross-section point is embedded as ``(0, y, z)`` when serialized.
The routines are deliberately dependency-light so the geometry/data contract can
be validated before a high-fidelity fracture solver is installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TunnelGeometry:
    """One closed tunnel cavity boundary in the ``y-z`` cross-section."""

    shape: str
    boundary_yz: FloatArray
    characteristic_radius: float
    roughness_amplitude: float
    seed: int

    def __post_init__(self) -> None:
        boundary = np.asarray(self.boundary_yz)
        if boundary.ndim != 2 or boundary.shape[1] != 2 or boundary.shape[0] < 8:
            raise ValueError("boundary_yz must have shape [N, 2] with N >= 8")
        if not np.isfinite(boundary).all():
            raise ValueError("boundary_yz contains non-finite values")
        if self.characteristic_radius <= 0:
            raise ValueError("characteristic_radius must be positive")


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


def _base_boundary(shape: str, n_points: int) -> FloatArray:
    if shape == "circle":
        theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
        return np.column_stack([np.sin(theta), np.cos(theta)])

    if shape == "horseshoe":
        width = 0.82
        bottom = np.column_stack(
            [np.full(32, -1.0), np.linspace(-width, width, 32, endpoint=False)]
        )
        right_wall = np.column_stack(
            [np.linspace(-1.0, 0.0, 24, endpoint=False), np.full(24, width)]
        )
        theta = np.linspace(0.0, np.pi, 96, endpoint=False)
        arch = np.column_stack([width * np.sin(theta), width * np.cos(theta)])
        left_wall = np.column_stack(
            [np.linspace(0.0, -1.0, 24, endpoint=False), np.full(24, -width)]
        )
        return resample_closed_polyline(np.vstack([bottom, right_wall, arch, left_wall]), n_points)

    if shape == "straight_wall_arch":
        width = 1.0
        spring_y = 0.2
        bottom = np.column_stack(
            [np.full(36, -1.0), np.linspace(-width, width, 36, endpoint=False)]
        )
        right_wall = np.column_stack(
            [np.linspace(-1.0, spring_y, 28, endpoint=False), np.full(28, width)]
        )
        theta = np.linspace(0.0, np.pi, 96, endpoint=False)
        arch = np.column_stack([spring_y + 0.8 * np.sin(theta), width * np.cos(theta)])
        left_wall = np.column_stack(
            [np.linspace(spring_y, -1.0, 28, endpoint=False), np.full(28, -width)]
        )
        return resample_closed_polyline(np.vstack([bottom, right_wall, arch, left_wall]), n_points)

    raise ValueError(f"unknown shape {shape!r}; expected circle, horseshoe, or straight_wall_arch")


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
    """

    if radius <= 0:
        raise ValueError("radius must be positive")
    if not 0.0 <= roughness_amplitude <= 0.08:
        raise ValueError("roughness_amplitude must lie in [0, 0.08]")
    boundary = _base_boundary(shape, n_points).astype(np.float64)
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
    return TunnelGeometry(
        shape=shape,
        boundary_yz=boundary,
        characteristic_radius=float(radius),
        roughness_amplitude=float(roughness_amplitude),
        seed=int(seed),
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
