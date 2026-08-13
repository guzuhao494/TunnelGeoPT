"""Leakage-safe coarse-to-fine elastic data for the v0.3 learning layer.

The contract in this module is deliberately narrower than "rockburst".  A
sample contains two solutions of the *same* homogeneous, plane-strain elastic
boundary-value problem on two mesh resolutions.  The coarse solution is a
permitted model input.  The fine solution is a supervised label guarded by a
split-aware access audit.

All stresses use the solver's tension-positive ``[yy, zz, yz]`` convention.
The normalization scale is computed from the prescribed far-field tensor; it
never inspects either numerical solution.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elasticity import solve_plane_strain_excavation
from .field_sampling import locate_elements, sample_piecewise_constant
from .geometry import (
    TunnelGeometry,
    make_parametric_tunnel_boundary,
    nearest_boundary_vectors,
    points_inside_polygon,
    surface_points_and_normals,
)
from .mesh import generate_tunnel_mesh

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
BoolArray = NDArray[np.bool_]

SPLIT_NAMES = ("train", "dev", "locked_test")
STRESS_COMPONENT_ORDER = ("yy", "zz", "yz")
SIGN_CONVENTION = "tension_positive"


class MultiFidelityContractError(ValueError):
    """Raised when a multi-fidelity identity, split, or access rule is broken."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MultiFidelityContractError(f"value is not canonical JSON: {exc}") from exc


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _array_digest(value: ArrayLike) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_hash(value: str, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise MultiFidelityContractError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _normalise_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return json.loads(_canonical_json(dict(value or {})))


def _coerce_sigma_inf(sigma_inf: ArrayLike) -> FloatArray:
    stress = np.asarray(sigma_inf, dtype=np.float64)
    if stress.shape == (3,):
        stress = np.asarray([[stress[0], stress[2]], [stress[2], stress[1]]])
    if stress.shape != (2, 2) or not np.isfinite(stress).all():
        raise MultiFidelityContractError("sigma_inf must be finite with shape [2,2] or [3]")
    tolerance = 1e-12 * max(float(np.max(np.abs(stress))), 1.0)
    if not np.allclose(stress, stress.T, rtol=0.0, atol=tolerance):
        raise MultiFidelityContractError("sigma_inf must be symmetric")
    stress = 0.5 * (stress + stress.T)
    if not np.any(stress):
        raise MultiFidelityContractError("sigma_inf must not be the zero tensor")
    return stress


def farfield_stress_scale(sigma_inf_tension_positive: ArrayLike) -> float:
    """Return the in-plane tensor Frobenius norm using only known far field."""

    stress = _coerce_sigma_inf(sigma_inf_tension_positive)
    scale = math.sqrt(float(stress[0, 0] ** 2 + stress[1, 1] ** 2 + 2.0 * stress[0, 1] ** 2))
    if not math.isfinite(scale) or scale <= np.finfo(np.float64).tiny:
        raise MultiFidelityContractError("far-field stress scale is not positive and finite")
    return scale


def geometry_group_id(
    boundary_yz: TunnelGeometry | ArrayLike,
    *,
    shape: str | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> str:
    """Hash the frozen boundary, rather than a mutable generator description."""

    generation: dict[str, Any] = {}
    if isinstance(boundary_yz, TunnelGeometry):
        geometry = boundary_yz
        boundary = np.asarray(geometry.boundary_yz, dtype=np.float64)
        if shape is not None and str(shape) != geometry.shape:
            raise MultiFidelityContractError("shape disagrees with TunnelGeometry.shape")
        shape = geometry.shape
        if parameters is None:
            parameters = geometry.shape_parameters
        generation = {
            "n_boundary_points": int(boundary.shape[0]),
            "radius": float(geometry.characteristic_radius),
            "roughness_amplitude": float(geometry.roughness_amplitude),
            "seed": int(geometry.seed),
        }
    else:
        boundary = np.asarray(boundary_yz, dtype=np.float64)
        if not shape:
            raise MultiFidelityContractError("shape is required for an array boundary")
    if boundary.ndim != 2 or boundary.shape[1] != 2 or boundary.shape[0] < 8:
        raise MultiFidelityContractError("boundary_yz must have shape [N,2], N >= 8")
    if not np.isfinite(boundary).all():
        raise MultiFidelityContractError("boundary_yz contains non-finite values")
    return _sha256_payload(
        {
            "identity": "tunnelgeopt.geometry.v1",
            "shape": str(shape),
            "parameters": _normalise_mapping(parameters),
            "generation": generation,
            "boundary_float64_sha256": _array_digest(boundary),
        }
    )


def load_group_id(
    sigma_inf_tension_positive: ArrayLike,
    *,
    sigma_xx_inf_tension_positive: float | None = None,
) -> str:
    """Hash one prescribed far-field load in the tension-positive convention."""

    stress = _coerce_sigma_inf(sigma_inf_tension_positive)
    axial = None if sigma_xx_inf_tension_positive is None else float(sigma_xx_inf_tension_positive)
    if axial is not None and not math.isfinite(axial):
        raise MultiFidelityContractError("sigma_xx_inf_tension_positive must be finite")
    return _sha256_payload(
        {
            "identity": "tunnelgeopt.elastic_load.v1",
            "sign_convention": SIGN_CONVENTION,
            "sigma_inf_yy_zz_yz": [
                float(stress[0, 0]),
                float(stress[1, 1]),
                float(stress[0, 1]),
            ],
            "sigma_xx_inf": axial,
        }
    )


def case_group_id(
    geometry_id: str,
    load_id: str,
    *,
    young_modulus: float,
    poisson_ratio: float,
) -> str:
    """Hash the physical case; mesh fidelity is intentionally excluded."""

    geometry_id = _require_hash(geometry_id, "geometry_id")
    load_id = _require_hash(load_id, "load_id")
    young = float(young_modulus)
    poisson = float(poisson_ratio)
    if not math.isfinite(young) or young <= 0.0:
        raise MultiFidelityContractError("young_modulus must be positive and finite")
    if not math.isfinite(poisson) or not -1.0 < poisson < 0.5:
        raise MultiFidelityContractError("poisson_ratio must lie in (-1, 0.5)")
    return _sha256_payload(
        {
            "identity": "tunnelgeopt.multifidelity_case.v1",
            "geometry_group_id": geometry_id,
            "load_group_id": load_id,
            "material": {"young_modulus": young, "poisson_ratio": poisson},
        }
    )


@dataclass(frozen=True)
class GeometryDataSpec:
    """Reproducible generator inputs for one frozen cross-section boundary."""

    shape: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    n_boundary_points: int = 96
    radius: float = 1.0
    roughness_amplitude: float = 0.0
    seed: int = 0
    outer_domain_scale: float = 4.0

    def build(self) -> TunnelGeometry:
        return make_parametric_tunnel_boundary(
            self.shape,
            parameters=_normalise_mapping(self.parameters),
            n_points=int(self.n_boundary_points),
            radius=float(self.radius),
            roughness_amplitude=float(self.roughness_amplitude),
            seed=int(self.seed),
        )

    def identity_parameters(self) -> dict[str, Any]:
        """Return preregistered generator/domain values in addition to boundary hash."""

        return {
            "shape_parameters": _normalise_mapping(self.parameters),
            "n_boundary_points": int(self.n_boundary_points),
            "radius": float(self.radius),
            "roughness_amplitude": float(self.roughness_amplitude),
            "seed": int(self.seed),
            "outer_domain_rule": {
                "kind": "boundary_extent_rectangle",
                "domain_scale": float(self.outer_domain_scale),
            },
        }

    def geometry_group_id(self, geometry: TunnelGeometry | None = None) -> str:
        built = self.build() if geometry is None else geometry
        return geometry_group_id(built, parameters=self.identity_parameters())


@dataclass(frozen=True)
class GeometrySplitSpec:
    """An explicit, geometry-level train/dev/locked-test assignment."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    locked_test: tuple[str, ...]

    def __post_init__(self) -> None:
        groups: list[str] = []
        for name in SPLIT_NAMES:
            values = tuple(getattr(self, name))
            if len(set(values)) != len(values):
                raise MultiFidelityContractError(f"split {name!r} repeats a geometry_group_id")
            for value in values:
                groups.append(_require_hash(value, f"{name} geometry_group_id"))
        if len(groups) != len(set(groups)):
            raise MultiFidelityContractError("a geometry_group_id appears in multiple splits")

    @property
    def geometry_count(self) -> int:
        return len(self.train) + len(self.dev) + len(self.locked_test)

    def split_for(self, geometry_id: str) -> str:
        geometry_id = _require_hash(geometry_id, "geometry_id")
        for split in SPLIT_NAMES:
            if geometry_id in getattr(self, split):
                return split
        raise MultiFidelityContractError("geometry_group_id is absent from the frozen split spec")

    def as_dict(self) -> dict[str, Any]:
        return {
            "unit": "geometry_group_id",
            "train": list(self.train),
            "dev": list(self.dev),
            "locked_test": list(self.locked_test),
            "counts": {name: len(getattr(self, name)) for name in SPLIT_NAMES},
        }


def freeze_geometry_splits(
    geometry_ids: Sequence[str], *, train_count: int, dev_count: int, locked_test_count: int
) -> GeometrySplitSpec:
    """Freeze deterministic hash-ordered splits before any elastic solve."""

    identifiers = [_require_hash(value, "geometry_id") for value in geometry_ids]
    if len(identifiers) != len(set(identifiers)):
        raise MultiFidelityContractError("geometry_ids must be unique")
    counts = [int(train_count), int(dev_count), int(locked_test_count)]
    if any(count < 1 for count in counts) or sum(counts) != len(identifiers):
        raise MultiFidelityContractError("positive split counts must sum to geometry count")
    ordered = sorted(identifiers)
    train_end = counts[0]
    dev_end = train_end + counts[1]
    return GeometrySplitSpec(
        train=tuple(ordered[:train_end]),
        dev=tuple(ordered[train_end:dev_end]),
        locked_test=tuple(ordered[dev_end:]),
    )


@dataclass(frozen=True)
class MeshFidelitySpec:
    """One first-order mesh resolution; physics is not part of this object."""

    mesh_size: float
    wall_mesh_size: float
    farfield_mesh_size: float

    def __post_init__(self) -> None:
        values = (self.mesh_size, self.wall_mesh_size, self.farfield_mesh_size)
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
            raise MultiFidelityContractError("mesh sizes must be positive and finite")

    def kwargs(self) -> dict[str, float]:
        return {
            "mesh_size": float(self.mesh_size),
            "wall_mesh_size": float(self.wall_mesh_size),
            "farfield_mesh_size": float(self.farfield_mesh_size),
        }


@dataclass(frozen=True)
class ElasticQueryGrid:
    """Fixed-P geometry query shared by every load and both fidelity tiers."""

    geometry_group_id: str
    points_yz: FloatArray
    x: FloatArray
    nearfield_mask: BoolArray
    wall_offset_mask: BoolArray
    farfield_mask: BoolArray
    area_weights: FloatArray
    arc_weights: FloatArray
    characteristic_radius: float
    normalization_center_yz: FloatArray
    query_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_hash(self.geometry_group_id, "geometry_group_id")
        _require_hash(self.query_hash, "query_hash")
        points = np.asarray(self.points_yz)
        x = np.asarray(self.x)
        if points.ndim != 2 or points.shape[1] != 2 or x.shape != (points.shape[0], 7):
            raise MultiFidelityContractError("query grid must have points [P,2] and x [P,7]")
        masks = [
            np.asarray(self.nearfield_mask),
            np.asarray(self.wall_offset_mask),
            np.asarray(self.farfield_mask),
        ]
        if any(mask.shape != (points.shape[0],) or mask.dtype != np.bool_ for mask in masks):
            raise MultiFidelityContractError("query masks must be boolean with shape [P]")
        membership = sum(mask.astype(np.int8) for mask in masks)
        if not np.all(membership == 1):
            raise MultiFidelityContractError("every query must belong to exactly one region")
        for name, weights in (
            ("area_weights", self.area_weights),
            ("arc_weights", self.arc_weights),
        ):
            values = np.asarray(weights)
            if values.shape != (points.shape[0],) or not np.isfinite(values).all():
                raise MultiFidelityContractError(f"{name} must be finite with shape [P]")
            if np.any(values < 0.0):
                raise MultiFidelityContractError(f"{name} must be non-negative")
        if not np.isclose(np.asarray(self.area_weights).sum(), 1.0):
            raise MultiFidelityContractError("area_weights must sum to one")
        if not np.isclose(np.asarray(self.arc_weights).sum(), 1.0):
            raise MultiFidelityContractError("arc_weights must sum to one")
        if not np.isfinite(points).all() or not np.isfinite(x).all():
            raise MultiFidelityContractError("query grid contains non-finite values")

    @property
    def point_count(self) -> int:
        return int(self.points_yz.shape[0])


def _radical_inverse(index: int, base: int) -> float:
    result = 0.0
    factor = 1.0 / base
    while index:
        index, digit = divmod(index, base)
        result += digit * factor
        factor /= base
    return result


def _halton(count: int, dimension: int, seed: int) -> FloatArray:
    bases = (2, 3, 5, 7)
    if count < 1 or dimension < 1 or dimension > len(bases):
        raise MultiFidelityContractError("invalid Halton shape")
    start = max(int(seed), 0) * 997 + 1
    return np.asarray(
        [
            [_radical_inverse(start + row, bases[column]) for column in range(dimension)]
            for row in range(count)
        ],
        dtype=np.float64,
    )


def _rectangle_perimeter_points(
    center: FloatArray, half_extent: FloatArray, count: int, seed: int
) -> FloatArray:
    phase = _halton(count, 1, seed)[:, 0]
    perimeter_coordinate = 4.0 * phase
    result = np.empty((count, 2), dtype=np.float64)
    for index, value in enumerate(perimeter_coordinate):
        side = min(int(value), 3)
        local = value - side
        if side == 0:
            result[index] = [-half_extent[0] + 2.0 * half_extent[0] * local, -half_extent[1]]
        elif side == 1:
            result[index] = [half_extent[0], -half_extent[1] + 2.0 * half_extent[1] * local]
        elif side == 2:
            result[index] = [half_extent[0] - 2.0 * half_extent[0] * local, half_extent[1]]
        else:
            result[index] = [-half_extent[0], half_extent[1] - 2.0 * half_extent[1] * local]
    return result + center


def build_elastic_query_grid(
    geometry: TunnelGeometry,
    *,
    geometry_parameters: Mapping[str, Any] | None = None,
    nearfield_points: int = 128,
    wall_offset_points: int = 64,
    farfield_points: int = 32,
    nearfield_scale: float = 3.2,
    farfield_scale: float = 3.0,
    nearfield_min_distance_over_radius: float = 0.05,
    nearfield_max_distance_over_radius: float = 2.0,
    wall_offset_over_radius: float = 0.08,
    seed: int = 0,
    outer_domain_scale: float | None = None,
) -> ElasticQueryGrid:
    """Build a deterministic fixed-P rock-side grid for one frozen boundary."""

    counts = (int(nearfield_points), int(wall_offset_points), int(farfield_points))
    if any(count < 1 for count in counts):
        raise MultiFidelityContractError("all query-region point counts must be positive")
    if counts[1] < 8:
        raise MultiFidelityContractError("wall_offset_points must be at least eight")
    radius = float(geometry.characteristic_radius)
    if float(nearfield_scale) <= 1.1 or float(farfield_scale) <= 1.1:
        raise MultiFidelityContractError("nearfield_scale and farfield_scale must exceed 1.1")
    minimum_distance = float(nearfield_min_distance_over_radius) * radius
    maximum_distance = float(nearfield_max_distance_over_radius) * radius
    if not 0.0 < minimum_distance < maximum_distance:
        raise MultiFidelityContractError("require 0 < nearfield minimum < maximum distance")
    if not 0.005 <= float(wall_offset_over_radius) <= 0.25:
        raise MultiFidelityContractError("wall offset must lie in [0.005, 0.25] radii")
    boundary = np.asarray(geometry.boundary_yz, dtype=np.float64)
    parameters = geometry.shape_parameters if geometry_parameters is None else geometry_parameters
    geometry_id = geometry_group_id(geometry, parameters=parameters)
    lower = boundary.min(axis=0)
    upper = boundary.max(axis=0)
    center = 0.5 * (lower + upper)
    base_half_extent = 0.5 * (upper - lower)

    near_half_extent = base_half_extent * float(nearfield_scale)
    accepted: list[FloatArray] = []
    total = 0
    batch_number = 0
    while total < counts[0] and batch_number < 100:
        unit = _halton(max(4 * counts[0], 128), 2, int(seed) + 11 + batch_number)
        candidates = center + (2.0 * unit - 1.0) * near_half_extent
        outside = ~points_inside_polygon(candidates, boundary)
        distance, _, _ = nearest_boundary_vectors(candidates, boundary)
        valid = outside & (distance >= minimum_distance) & (distance <= maximum_distance)
        chosen = candidates[valid]
        accepted.append(chosen)
        total += chosen.shape[0]
        batch_number += 1
    if total < counts[0]:
        raise MultiFidelityContractError("could not construct the requested near-field queries")
    nearfield = np.vstack(accepted)[: counts[0]]

    wall, normals = surface_points_and_normals(geometry, counts[1])
    wall_offset = wall + float(wall_offset_over_radius) * radius * normals
    if np.any(points_inside_polygon(wall_offset, boundary)):
        raise MultiFidelityContractError("wall-offset query construction entered the cavity")

    far_half_extent = base_half_extent * float(farfield_scale)
    if outer_domain_scale is not None and not float(farfield_scale) < float(outer_domain_scale):
        raise MultiFidelityContractError(
            "farfield_scale must be strictly inside the preregistered outer_domain_scale"
        )
    farfield = _rectangle_perimeter_points(center, far_half_extent, counts[2], int(seed) + 29)
    if np.any(points_inside_polygon(farfield, boundary)):
        raise MultiFidelityContractError("far-field query construction entered the cavity")

    points = np.vstack([nearfield, wall_offset, farfield])
    distance, wall_to_point, _ = nearest_boundary_vectors(points, boundary)
    point_to_wall = -wall_to_point
    normalized_points = (points - center) / radius
    x = np.column_stack(
        [
            np.zeros(points.shape[0]),
            normalized_points,
            distance / radius,
            np.zeros(points.shape[0]),
            point_to_wall,
        ]
    ).astype(np.float32)
    near_mask = np.zeros(points.shape[0], dtype=bool)
    wall_mask = np.zeros(points.shape[0], dtype=bool)
    far_mask = np.zeros(points.shape[0], dtype=bool)
    near_mask[: counts[0]] = True
    wall_mask[counts[0] : counts[0] + counts[1]] = True
    far_mask[counts[0] + counts[1] :] = True
    area_weights = np.zeros(points.shape[0], dtype=np.float64)
    area_weights[near_mask] = 1.0 / counts[0]
    wall_points = points[wall_mask]
    segment_length = np.linalg.norm(np.roll(wall_points, -1, axis=0) - wall_points, axis=1)
    local_arc = 0.5 * (segment_length + np.roll(segment_length, 1))
    arc_weights = np.zeros(points.shape[0], dtype=np.float64)
    arc_weights[wall_mask] = local_arc / local_arc.sum()
    metadata = {
        "identity": "tunnelgeopt.elastic_query.v1",
        "seed": int(seed),
        "point_counts": {
            "nearfield": counts[0],
            "wall_offset": counts[1],
            "farfield": counts[2],
            "total": sum(counts),
        },
        "nearfield_scale": float(nearfield_scale),
        "farfield_scale": float(farfield_scale),
        "outer_domain_scale": (None if outer_domain_scale is None else float(outer_domain_scale)),
        "nearfield_distance_over_radius": [
            float(nearfield_min_distance_over_radius),
            float(nearfield_max_distance_over_radius),
        ],
        "wall_offset_over_radius": float(wall_offset_over_radius),
        "area_weight_normalization": "nearfield_sum_one",
        "arc_weight_normalization": "wall_offset_sum_one",
    }
    query_hash = _sha256_payload(
        {
            "metadata": metadata,
            "geometry_group_id": geometry_id,
            "points_sha256": _array_digest(points),
            "x_sha256": _array_digest(x),
            "area_weights_sha256": _array_digest(area_weights),
            "arc_weights_sha256": _array_digest(arc_weights),
        }
    )
    return ElasticQueryGrid(
        geometry_group_id=geometry_id,
        points_yz=points,
        x=x,
        nearfield_mask=near_mask,
        wall_offset_mask=wall_mask,
        farfield_mask=far_mask,
        area_weights=area_weights,
        arc_weights=arc_weights,
        characteristic_radius=radius,
        normalization_center_yz=center,
        query_hash=query_hash,
        metadata=metadata,
    )


def elastic_condition_vector(
    sigma_inf_tension_positive: ArrayLike,
    *,
    poisson_ratio: float,
    sigma_xx_inf_tension_positive: float | None = None,
) -> FloatArray:
    """Return normalized ``[yy, zz, yz, xx]`` known far-field components."""

    stress = _coerce_sigma_inf(sigma_inf_tension_positive)
    scale = farfield_stress_scale(stress)
    axial = (
        float(poisson_ratio) * float(stress[0, 0] + stress[1, 1])
        if sigma_xx_inf_tension_positive is None
        else float(sigma_xx_inf_tension_positive)
    )
    if not math.isfinite(axial):
        raise MultiFidelityContractError("far-field axial stress must be finite")
    return np.asarray(
        [stress[0, 0] / scale, stress[1, 1] / scale, stress[0, 1] / scale, axial / scale],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class MultiFidelitySample:
    """One common-query coarse/fine case; fine labels are intentionally private."""

    geometry_group_id: str
    load_group_id: str
    case_group_id: str
    split: str
    grid: ElasticQueryGrid
    condition: FloatArray
    stress_scale: float
    coarse_stress_normalized: FloatArray
    _fine_stress_normalized: FloatArray = field(repr=False)
    coarse_element_ids: IntArray
    fine_element_ids: IntArray
    coarse_mesh_metadata: Mapping[str, Any]
    fine_mesh_metadata: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_hash(self.geometry_group_id, "geometry_group_id")
        _require_hash(self.load_group_id, "load_group_id")
        _require_hash(self.case_group_id, "case_group_id")
        if self.split not in SPLIT_NAMES:
            raise MultiFidelityContractError(f"unknown split {self.split!r}")
        if self.grid.geometry_group_id != self.geometry_group_id:
            raise MultiFidelityContractError("query grid belongs to a different frozen boundary")
        point_count = self.grid.point_count
        if np.asarray(self.condition).shape != (4,):
            raise MultiFidelityContractError("condition must have shape [4]")
        for name, values in (
            ("coarse_stress_normalized", self.coarse_stress_normalized),
            ("fine_stress_normalized", self._fine_stress_normalized),
        ):
            array = np.asarray(values)
            if array.shape != (point_count, 3) or not np.isfinite(array).all():
                raise MultiFidelityContractError(f"{name} must be finite with shape [P,3]")
        for name, values in (
            ("coarse_element_ids", self.coarse_element_ids),
            ("fine_element_ids", self.fine_element_ids),
        ):
            array = np.asarray(values)
            if array.shape != (point_count,) or not np.issubdtype(array.dtype, np.integer):
                raise MultiFidelityContractError(f"{name} must be integer with shape [P]")
            if np.any(array < 0):
                raise MultiFidelityContractError(f"{name} contains a query outside the mesh")
        if not math.isfinite(float(self.stress_scale)) or float(self.stress_scale) <= 0.0:
            raise MultiFidelityContractError("stress_scale must be positive and finite")

    @property
    def model_features(self) -> FloatArray:
        """Return the public ``x7 + condition4 + coarse3`` tensor ``[P,14]``."""

        condition = np.repeat(
            np.asarray(self.condition, dtype=np.float32)[None, :], self.grid.point_count, axis=0
        )
        result = np.concatenate(
            [
                np.asarray(self.grid.x, dtype=np.float32),
                condition,
                np.asarray(self.coarse_stress_normalized, dtype=np.float32),
            ],
            axis=1,
        )
        if result.shape != (self.grid.point_count, 14):
            raise MultiFidelityContractError("model feature tensor violates the [P,14] contract")
        return result

    @property
    def fine_stress_normalized(self) -> FloatArray:
        raise MultiFidelityContractError(
            "fine labels are private; use MultiFidelityDataset.fine_labels_for()"
        )


def solve_multifidelity_case(
    geometry: TunnelGeometry,
    grid: ElasticQueryGrid,
    *,
    split: str,
    sigma_inf_tension_positive: ArrayLike,
    young_modulus: float,
    poisson_ratio: float,
    coarse_mesh: MeshFidelitySpec,
    fine_mesh: MeshFidelitySpec,
    domain_scale: float = 4.0,
    sigma_xx_inf_tension_positive: float | None = None,
    geometry_identity_parameters: Mapping[str, Any] | None = None,
) -> MultiFidelitySample:
    """Solve one frozen boundary/load twice, changing mesh resolution only."""

    if split not in SPLIT_NAMES:
        raise MultiFidelityContractError(f"unknown split {split!r}")
    expected_geometry_id = geometry_group_id(
        geometry,
        parameters=(
            geometry.shape_parameters
            if geometry_identity_parameters is None
            else geometry_identity_parameters
        ),
    )
    if expected_geometry_id != grid.geometry_group_id:
        raise MultiFidelityContractError(
            "grid/frozen-boundary mismatch (pass the same geometry parameters to both identities)"
        )
    coarse_values = np.asarray(
        [coarse_mesh.mesh_size, coarse_mesh.wall_mesh_size, coarse_mesh.farfield_mesh_size]
    )
    fine_values = np.asarray(
        [fine_mesh.mesh_size, fine_mesh.wall_mesh_size, fine_mesh.farfield_mesh_size]
    )
    if np.any(fine_values > coarse_values) or np.allclose(fine_values, coarse_values):
        raise MultiFidelityContractError(
            "fine mesh sizes must be no larger than coarse and at least one must be smaller"
        )
    stress = _coerce_sigma_inf(sigma_inf_tension_positive)
    scale = farfield_stress_scale(stress)
    load_id = load_group_id(stress, sigma_xx_inf_tension_positive=sigma_xx_inf_tension_positive)
    case_id = case_group_id(
        expected_geometry_id,
        load_id,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
    )
    common_outer_bounds = (
        float(
            grid.normalization_center_yz[0]
            - 0.5 * np.ptp(geometry.boundary_yz[:, 0]) * domain_scale
        ),
        float(
            grid.normalization_center_yz[0]
            + 0.5 * np.ptp(geometry.boundary_yz[:, 0]) * domain_scale
        ),
        float(
            grid.normalization_center_yz[1]
            - 0.5 * np.ptp(geometry.boundary_yz[:, 1]) * domain_scale
        ),
        float(
            grid.normalization_center_yz[1]
            + 0.5 * np.ptp(geometry.boundary_yz[:, 1]) * domain_scale
        ),
    )
    tolerance = 1e-12 * max(
        common_outer_bounds[1] - common_outer_bounds[0],
        common_outer_bounds[3] - common_outer_bounds[2],
        1.0,
    )
    points = np.asarray(grid.points_yz)
    inside_outer_domain = (
        (points[:, 0] > common_outer_bounds[0] + tolerance)
        & (points[:, 0] < common_outer_bounds[1] - tolerance)
        & (points[:, 1] > common_outer_bounds[2] + tolerance)
        & (points[:, 1] < common_outer_bounds[3] - tolerance)
    )
    if not np.all(inside_outer_domain):
        raise MultiFidelityContractError(
            "query farfield/nearfield points must lie strictly inside the actual solve domain"
        )
    meshes = [
        generate_tunnel_mesh(geometry, outer_bounds=common_outer_bounds, **coarse_mesh.kwargs()),
        generate_tunnel_mesh(geometry, outer_bounds=common_outer_bounds, **fine_mesh.kwargs()),
    ]
    results = [
        solve_plane_strain_excavation(
            mesh,
            young_modulus=float(young_modulus),
            poisson_ratio=float(poisson_ratio),
            sigma_inf=stress,
            sigma_xx_inf=sigma_xx_inf_tension_positive,
        )
        for mesh in meshes
    ]
    element_ids = [
        locate_elements(result.nodes, result.elements, grid.points_yz, raise_outside=True)
        for result in results
    ]
    sampled = [
        sample_piecewise_constant(result.total_stress, indices)
        for result, indices in zip(results, element_ids, strict=True)
    ]
    normalized = [np.asarray(values, dtype=np.float64) / scale for values in sampled]
    diagnostics = {
        "sign_convention": SIGN_CONVENTION,
        "stress_component_order": list(STRESS_COMPONENT_ORDER),
        "stress_scale_source": "prescribed_in_plane_farfield_frobenius_norm",
        "common_query_hash": grid.query_hash,
        "same_frozen_boundary": True,
        "same_outer_bounds": meshes[0].outer_bounds == meshes[1].outer_bounds,
        "coarse": {
            "algebraic_residual": float(results[0].algebraic_residual),
            "energy_closure": float(results[0].energy_closure),
        },
        "fine": {
            "algebraic_residual": float(results[1].algebraic_residual),
            "energy_closure": float(results[1].energy_closure),
        },
    }
    return MultiFidelitySample(
        geometry_group_id=expected_geometry_id,
        load_group_id=load_id,
        case_group_id=case_id,
        split=split,
        grid=grid,
        condition=elastic_condition_vector(
            stress,
            poisson_ratio=poisson_ratio,
            sigma_xx_inf_tension_positive=sigma_xx_inf_tension_positive,
        ),
        stress_scale=scale,
        coarse_stress_normalized=normalized[0],
        _fine_stress_normalized=normalized[1],
        coarse_element_ids=np.asarray(element_ids[0], dtype=np.int64),
        fine_element_ids=np.asarray(element_ids[1], dtype=np.int64),
        coarse_mesh_metadata=dict(meshes[0].metadata),
        fine_mesh_metadata=dict(meshes[1].metadata),
        diagnostics=diagnostics,
    )


@dataclass
class MultiFidelityAccessAudit:
    coarse_feature_case_reads: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES}
    )
    fine_label_case_reads: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES}
    )
    denied_locked_fine_accesses: int = 0
    locked_test_unlocked: bool = False
    frozen_checkpoint_count_authorized: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "coarse_feature_case_reads": dict(self.coarse_feature_case_reads),
            "fine_label_case_reads": dict(self.fine_label_case_reads),
            "denied_locked_fine_accesses": int(self.denied_locked_fine_accesses),
            "locked_test_unlocked": bool(self.locked_test_unlocked),
            "frozen_checkpoint_count_authorized": int(self.frozen_checkpoint_count_authorized),
            "events": [dict(event) for event in self.events],
        }


@dataclass
class MultiFidelityDataset:
    """Samples plus an auditable fine-label access boundary."""

    samples: tuple[MultiFidelitySample, ...]
    split_spec: GeometrySplitSpec
    access_audit: MultiFidelityAccessAudit = field(default_factory=MultiFidelityAccessAudit)

    def __post_init__(self) -> None:
        if not self.samples:
            raise MultiFidelityContractError("dataset requires at least one sample")
        case_ids = [sample.case_group_id for sample in self.samples]
        if len(case_ids) != len(set(case_ids)):
            raise MultiFidelityContractError("dataset repeats a case_group_id")
        point_counts = {sample.grid.point_count for sample in self.samples}
        if len(point_counts) != 1:
            raise MultiFidelityContractError("every sample must use the same fixed point count P")
        split_by_geometry: dict[str, str] = {}
        query_by_geometry: dict[str, str] = {}
        for sample in self.samples:
            expected = self.split_spec.split_for(sample.geometry_group_id)
            if sample.split != expected:
                raise MultiFidelityContractError("sample split disagrees with geometry split spec")
            previous = split_by_geometry.setdefault(sample.geometry_group_id, sample.split)
            if previous != sample.split:
                raise MultiFidelityContractError("loads of one geometry crossed split boundaries")
            query = query_by_geometry.setdefault(sample.geometry_group_id, sample.grid.query_hash)
            if query != sample.grid.query_hash:
                raise MultiFidelityContractError("loads of one geometry use different queries")

    def indices(self, split: str) -> IntArray:
        if split not in SPLIT_NAMES:
            raise MultiFidelityContractError(f"unknown split {split!r}")
        return np.asarray(
            [index for index, sample in enumerate(self.samples) if sample.split == split],
            dtype=np.int64,
        )

    def _validate_indices(self, indices: Sequence[int]) -> list[int]:
        result = [int(index) for index in indices]
        if not result:
            raise MultiFidelityContractError("array access requires at least one case")
        if any(index < 0 or index >= len(self.samples) for index in result):
            raise MultiFidelityContractError("array access contains an out-of-range case index")
        return result

    def features_for(self, indices: Sequence[int], *, purpose: str) -> FloatArray:
        """Read public features; locked-test coarse input is intentionally allowed."""

        requested = self._validate_indices(indices)
        for split in {self.samples[index].split for index in requested}:
            self.access_audit.coarse_feature_case_reads[split] += sum(
                self.samples[index].split == split for index in requested
            )
        self.access_audit.events.append(
            {"event": "coarse_features_read", "purpose": str(purpose), "count": len(requested)}
        )
        return np.stack([self.samples[index].model_features for index in requested]).astype(
            np.float32
        )

    def authorize_locked_test(
        self, frozen_checkpoint_ids: Sequence[str], *, expected_checkpoint_count: int
    ) -> None:
        identities = [str(value) for value in frozen_checkpoint_ids]
        if expected_checkpoint_count <= 0 or len(identities) != expected_checkpoint_count:
            raise MultiFidelityContractError(
                "locked-test authorization requires every expected frozen checkpoint"
            )
        if len(set(identities)) != len(identities) or any(not value for value in identities):
            raise MultiFidelityContractError("checkpoint identities must be unique and non-empty")
        if self.access_audit.fine_label_case_reads["locked_test"]:
            raise MultiFidelityContractError("locked fine labels were read before authorization")
        self.access_audit.locked_test_unlocked = True
        self.access_audit.frozen_checkpoint_count_authorized = len(identities)
        self.access_audit.events.append(
            {"event": "locked_test_authorized", "checkpoint_count": len(identities)}
        )

    def _fine_arrays_for(
        self, indices: Sequence[int], *, purpose: str
    ) -> tuple[list[int], FloatArray]:
        requested = self._validate_indices(indices)
        locked_count = sum(self.samples[index].split == "locked_test" for index in requested)
        if locked_count and not self.access_audit.locked_test_unlocked:
            self.access_audit.denied_locked_fine_accesses += 1
            self.access_audit.events.append(
                {
                    "event": "denied_locked_fine_read",
                    "purpose": str(purpose),
                    "requested_locked_case_count": locked_count,
                }
            )
            raise MultiFidelityContractError(
                "locked_test fine labels cannot be read before checkpoints are frozen"
            )
        for split in {self.samples[index].split for index in requested}:
            self.access_audit.fine_label_case_reads[split] += sum(
                self.samples[index].split == split for index in requested
            )
        self.access_audit.events.append(
            {"event": "fine_labels_read", "purpose": str(purpose), "count": len(requested)}
        )
        values = np.stack(
            [self.samples[index]._fine_stress_normalized for index in requested]
        ).astype(np.float32)
        return requested, values

    def fine_labels_for(self, indices: Sequence[int], *, purpose: str) -> FloatArray:
        """Read normalized fine stress through the label gate."""

        return self._fine_arrays_for(indices, purpose=purpose)[1]

    def residual_labels_for(self, indices: Sequence[int], *, purpose: str) -> FloatArray:
        """Read normalized ``fine - coarse`` targets through the same label gate."""

        requested, fine = self._fine_arrays_for(indices, purpose=purpose)
        coarse = np.stack(
            [self.samples[index].coarse_stress_normalized for index in requested]
        ).astype(np.float32)
        return fine - coarse

    def access_snapshot(self) -> dict[str, Any]:
        return self.access_audit.snapshot()


def reconstruct_fine_stress(
    coarse_stress_normalized: ArrayLike,
    predicted_residual_normalized: ArrayLike,
    *,
    stress_scale: float | ArrayLike | None = None,
) -> FloatArray:
    """Reconstruct fine stress; omit scale for normalized output."""

    coarse = np.asarray(coarse_stress_normalized)
    residual = np.asarray(predicted_residual_normalized)
    if coarse.shape != residual.shape or coarse.shape[-1:] != (3,):
        raise MultiFidelityContractError("coarse and residual must share shape [...,3]")
    if not np.isfinite(coarse).all() or not np.isfinite(residual).all():
        raise MultiFidelityContractError("reconstruction inputs contain non-finite values")
    result = coarse + residual
    if stress_scale is not None:
        scale = np.asarray(stress_scale, dtype=np.float64)
        if not np.isfinite(scale).all() or np.any(scale <= 0.0):
            raise MultiFidelityContractError("stress_scale must be positive and finite")
        while scale.ndim < result.ndim:
            scale = np.expand_dims(scale, axis=-1)
        result = result * scale
    return np.asarray(result)


__all__ = [
    "SIGN_CONVENTION",
    "SPLIT_NAMES",
    "STRESS_COMPONENT_ORDER",
    "ElasticQueryGrid",
    "GeometryDataSpec",
    "GeometrySplitSpec",
    "MeshFidelitySpec",
    "MultiFidelityAccessAudit",
    "MultiFidelityContractError",
    "MultiFidelityDataset",
    "MultiFidelitySample",
    "build_elastic_query_grid",
    "case_group_id",
    "elastic_condition_vector",
    "farfield_stress_scale",
    "freeze_geometry_splits",
    "geometry_group_id",
    "load_group_id",
    "reconstruct_fine_stress",
    "solve_multifidelity_case",
]
