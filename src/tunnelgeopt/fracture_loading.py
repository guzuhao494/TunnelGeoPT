"""Audited load histories for the frozen fracture Phase-1 development pilot.

The protocol stores positive principal *compression* controls.  This module
interpolates those stored controls first and only then converts them to a
tension-positive stress tensor in ``(y, z)`` order.  It also evaluates the P4
release field on the actual wall facets of one mesh.  It does not apply loads
to the fracture solver or execute a trajectory.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fracture_validation import LOAD_PATH_IDS, validate_fracture_phase1_config

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

WALL_ZONE_IDS = ("crown", "right_sidewall", "invert", "left_sidewall")
_TRANSITION_TOTAL_WIDTH_DEG = 5.0
_TRANSITION_HALF_WIDTH_DEG = 0.5 * _TRANSITION_TOTAL_WIDTH_DEG
_ZONE_BOUNDARIES_DEG = (45.0, 135.0, 225.0, 315.0)


class FractureLoadError(ValueError):
    """Raised when a Phase-1 load state cannot be compiled unambiguously."""


def _readonly_float_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> FloatArray:
    array = np.array(value, dtype=np.float64, order="C", copy=True)
    if array.ndim != ndim:
        raise FractureLoadError(f"{name} must have {ndim} dimensions")
    if shape is not None and any(
        expected is not None and array.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise FractureLoadError(f"{name} has shape {array.shape}, expected {shape}")
    if not np.isfinite(array).all():
        raise FractureLoadError(f"{name} contains a non-finite value")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.float64)
    return immutable.reshape(array.shape)


def _readonly_int_array(
    value: ArrayLike,
    *,
    name: str,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> IntArray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
        raise FractureLoadError(f"{name} must contain integer identifiers")
    array = np.array(raw, dtype=np.int64, order="C", copy=True)
    if array.ndim != ndim:
        raise FractureLoadError(f"{name} must have {ndim} dimensions")
    if shape is not None and any(
        expected is not None and array.shape[index] != expected
        for index, expected in enumerate(shape)
    ):
        raise FractureLoadError(f"{name} has shape {array.shape}, expected {shape}")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=np.int64)
    return immutable.reshape(array.shape)


def _finite_scalar(value: Any, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise FractureLoadError(f"{name} must be a real scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise FractureLoadError(f"{name} must be finite")
    return scalar


@dataclass(frozen=True, slots=True)
class FractureLoadState:
    """One immutable, facet-aligned state of a frozen Phase-1 load history."""

    path_id: str
    s: float
    ucs_scale: float
    sigma1_over_UCS: float
    sigma3_over_sigma1: float
    principal_angle_deg: float
    farfield_stress_tension_positive_yz: FloatArray
    wall_facet_ids: IntArray
    wall_zone_ids: tuple[str, ...]
    wall_zone_release: FloatArray
    wall_zone_weights: FloatArray
    wall_release: FloatArray

    def __post_init__(self) -> None:
        if self.path_id not in LOAD_PATH_IDS:
            raise FractureLoadError(f"unknown Phase-1 path_id {self.path_id!r}")
        if tuple(self.wall_zone_ids) != WALL_ZONE_IDS:
            raise FractureLoadError(f"wall_zone_ids must equal {WALL_ZONE_IDS}")
        object.__setattr__(self, "wall_zone_ids", WALL_ZONE_IDS)

        s = _finite_scalar(self.s, name="s")
        if not 0.0 <= s <= 1.0:
            raise FractureLoadError("s must lie in [0, 1]")
        ucs_scale = _finite_scalar(self.ucs_scale, name="ucs_scale")
        if ucs_scale <= 0.0:
            raise FractureLoadError("ucs_scale must be positive")
        sigma1 = _finite_scalar(self.sigma1_over_UCS, name="sigma1_over_UCS")
        ratio = _finite_scalar(self.sigma3_over_sigma1, name="sigma3_over_sigma1")
        angle = _finite_scalar(self.principal_angle_deg, name="principal_angle_deg")
        if sigma1 <= 0.0 or not 0.0 < ratio <= 1.0:
            raise FractureLoadError("principal compression controls are invalid")
        object.__setattr__(self, "s", s)
        object.__setattr__(self, "ucs_scale", ucs_scale)
        object.__setattr__(self, "sigma1_over_UCS", sigma1)
        object.__setattr__(self, "sigma3_over_sigma1", ratio)
        object.__setattr__(self, "principal_angle_deg", angle)

        stress = _readonly_float_array(
            self.farfield_stress_tension_positive_yz,
            name="farfield_stress_tension_positive_yz",
            ndim=2,
            shape=(2, 2),
        )
        if not np.allclose(stress, stress.T, rtol=0.0, atol=1e-13 * ucs_scale):
            raise FractureLoadError("farfield stress tensor must be symmetric")
        expected_stress = _tension_positive_principal_tensor_yz(
            sigma1_compression=sigma1 * ucs_scale,
            sigma3_compression=sigma1 * ratio * ucs_scale,
            principal_angle_deg=angle,
        )
        stress_tolerance = 32.0 * np.finfo(np.float64).eps * ucs_scale
        if not np.allclose(stress, expected_stress, rtol=2e-14, atol=stress_tolerance):
            raise FractureLoadError(
                "farfield stress tensor is inconsistent with ucs_scale and principal controls"
            )
        facet_ids = _readonly_int_array(self.wall_facet_ids, name="wall_facet_ids", ndim=1)
        if facet_ids.size == 0 or np.unique(facet_ids).size != facet_ids.size:
            raise FractureLoadError("wall_facet_ids must be non-empty and unique")
        zone_release = _readonly_float_array(
            self.wall_zone_release,
            name="wall_zone_release",
            ndim=1,
            shape=(len(WALL_ZONE_IDS),),
        )
        zone_weights = _readonly_float_array(
            self.wall_zone_weights,
            name="wall_zone_weights",
            ndim=2,
            shape=(facet_ids.size, len(WALL_ZONE_IDS)),
        )
        wall_release = _readonly_float_array(
            self.wall_release,
            name="wall_release",
            ndim=1,
            shape=(facet_ids.size,),
        )
        if np.any(zone_release < 0.0) or np.any(zone_release > 1.0):
            raise FractureLoadError("wall_zone_release must lie in [0, 1]")
        if np.any(zone_weights < 0.0) or np.any(zone_weights > 1.0):
            raise FractureLoadError("wall_zone_weights must lie in [0, 1]")
        if not np.allclose(zone_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-14):
            raise FractureLoadError("wall_zone_weights must form a convex partition")
        expected_release = zone_weights @ zone_release
        if not np.allclose(wall_release, expected_release, rtol=0.0, atol=1e-14):
            raise FractureLoadError("wall_release is not aligned with its facet-zone weights")

        object.__setattr__(self, "farfield_stress_tension_positive_yz", stress)
        object.__setattr__(self, "wall_facet_ids", facet_ids)
        object.__setattr__(self, "wall_zone_release", zone_release)
        object.__setattr__(self, "wall_zone_weights", zone_weights)
        object.__setattr__(self, "wall_release", wall_release)


@dataclass(frozen=True, slots=True)
class Phase1LoadSchedule:
    """Immutable interpolation data for one load path on one wall mesh."""

    path_id: str
    ucs_scale: float
    wall_facet_ids: IntArray
    wall_facet_midpoints_yz: FloatArray
    wall_perimeter_centroid_yz: FloatArray
    wall_zone_ids: tuple[str, ...]
    wall_zone_weights: FloatArray
    _control_s: FloatArray
    _principal_controls: FloatArray
    _zone_release_controls: FloatArray

    def __post_init__(self) -> None:
        if self.path_id not in LOAD_PATH_IDS:
            raise FractureLoadError(f"unknown Phase-1 path_id {self.path_id!r}")
        if tuple(self.wall_zone_ids) != WALL_ZONE_IDS:
            raise FractureLoadError(f"wall_zone_ids must equal {WALL_ZONE_IDS}")
        object.__setattr__(self, "wall_zone_ids", WALL_ZONE_IDS)
        ucs_scale = _finite_scalar(self.ucs_scale, name="ucs_scale")
        if ucs_scale <= 0.0:
            raise FractureLoadError("ucs_scale must be positive")
        object.__setattr__(self, "ucs_scale", ucs_scale)

        facet_ids = _readonly_int_array(self.wall_facet_ids, name="wall_facet_ids", ndim=1)
        if facet_ids.size == 0 or np.unique(facet_ids).size != facet_ids.size:
            raise FractureLoadError("wall_facet_ids must be non-empty and unique")
        midpoints = _readonly_float_array(
            self.wall_facet_midpoints_yz,
            name="wall_facet_midpoints_yz",
            ndim=2,
            shape=(facet_ids.size, 2),
        )
        centroid = _readonly_float_array(
            self.wall_perimeter_centroid_yz,
            name="wall_perimeter_centroid_yz",
            ndim=1,
            shape=(2,),
        )
        zone_weights = _readonly_float_array(
            self.wall_zone_weights,
            name="wall_zone_weights",
            ndim=2,
            shape=(facet_ids.size, len(WALL_ZONE_IDS)),
        )
        if np.any(zone_weights < 0.0) or np.any(zone_weights > 1.0):
            raise FractureLoadError("wall_zone_weights must lie in [0, 1]")
        if not np.allclose(zone_weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-14):
            raise FractureLoadError("wall_zone_weights must form a convex partition")

        control_s = _readonly_float_array(self._control_s, name="control_s", ndim=1)
        if control_s.size < 2 or not np.array_equal(control_s, np.linspace(0.0, 1.0, 5)):
            raise FractureLoadError("control_s must be the five frozen Phase-1 knots")
        principal = _readonly_float_array(
            self._principal_controls,
            name="principal_controls",
            ndim=2,
            shape=(control_s.size, 3),
        )
        releases = _readonly_float_array(
            self._zone_release_controls,
            name="zone_release_controls",
            ndim=2,
            shape=(control_s.size, len(WALL_ZONE_IDS)),
        )
        if np.any(releases < 0.0) or np.any(releases > 1.0):
            raise FractureLoadError("zone release controls must lie in [0, 1]")
        if np.any(np.diff(releases, axis=0) < 0.0):
            raise FractureLoadError("zone release controls must be monotone")

        object.__setattr__(self, "wall_facet_ids", facet_ids)
        object.__setattr__(self, "wall_facet_midpoints_yz", midpoints)
        object.__setattr__(self, "wall_perimeter_centroid_yz", centroid)
        object.__setattr__(self, "wall_zone_weights", zone_weights)
        object.__setattr__(self, "_control_s", control_s)
        object.__setattr__(self, "_principal_controls", principal)
        object.__setattr__(self, "_zone_release_controls", releases)

    def state_at(self, s: Real) -> FractureLoadState:
        """Interpolate the stored controls at any finite ``s`` in ``[0, 1]``."""

        parameter = _finite_scalar(s, name="s")
        if not 0.0 <= parameter <= 1.0:
            raise FractureLoadError("s must lie in [0, 1]")
        principal = np.asarray(
            [
                np.interp(parameter, self._control_s, self._principal_controls[:, index])
                for index in range(3)
            ],
            dtype=np.float64,
        )
        sigma1_over_ucs, sigma3_over_sigma1, principal_angle_deg = principal
        zone_release = np.asarray(
            [
                np.interp(parameter, self._control_s, self._zone_release_controls[:, index])
                for index in range(len(WALL_ZONE_IDS))
            ],
            dtype=np.float64,
        )
        wall_release = self.wall_zone_weights @ zone_release
        stress = _tension_positive_principal_tensor_yz(
            sigma1_compression=float(sigma1_over_ucs * self.ucs_scale),
            sigma3_compression=float(sigma1_over_ucs * sigma3_over_sigma1 * self.ucs_scale),
            principal_angle_deg=float(principal_angle_deg),
        )
        return FractureLoadState(
            path_id=self.path_id,
            s=parameter,
            ucs_scale=self.ucs_scale,
            sigma1_over_UCS=float(sigma1_over_ucs),
            sigma3_over_sigma1=float(sigma3_over_sigma1),
            principal_angle_deg=float(principal_angle_deg),
            farfield_stress_tension_positive_yz=stress,
            wall_facet_ids=self.wall_facet_ids,
            wall_zone_ids=WALL_ZONE_IDS,
            wall_zone_release=zone_release,
            wall_zone_weights=self.wall_zone_weights,
            wall_release=wall_release,
        )


def _tension_positive_principal_tensor_yz(
    *, sigma1_compression: float, sigma3_compression: float, principal_angle_deg: float
) -> FloatArray:
    angle = math.radians(principal_angle_deg)
    major_direction_yz = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)
    minor_direction_yz = np.asarray([-math.sin(angle), math.cos(angle)], dtype=np.float64)
    stress = -(
        sigma1_compression * np.outer(major_direction_yz, major_direction_yz)
        + sigma3_compression * np.outer(minor_direction_yz, minor_direction_yz)
    )
    return np.asarray(stress, dtype=np.float64)


def _extract_wall_geometry(mesh: Any) -> tuple[IntArray, FloatArray, FloatArray]:
    try:
        nodes_raw = mesh.nodes
        boundary_facets = mesh.boundary_facets
        skfem_mesh = mesh.mesh
        facets_raw = skfem_mesh.facets
    except AttributeError as exc:
        raise FractureLoadError(
            "mesh must expose TunnelMesh nodes, mesh.facets, and boundary_facets"
        ) from exc
    nodes = np.asarray(nodes_raw, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise FractureLoadError("mesh.nodes must be finite with shape [N, 2] in (y, z) order")
    if not isinstance(boundary_facets, Mapping) or "wall" not in boundary_facets:
        raise FractureLoadError("mesh.boundary_facets must contain a 'wall' marker")
    wall_ids = _readonly_int_array(
        boundary_facets["wall"], name="mesh.boundary_facets['wall']", ndim=1
    )
    if wall_ids.size == 0 or np.unique(wall_ids).size != wall_ids.size:
        raise FractureLoadError("wall facet identifiers must be non-empty and unique")
    facets = np.asarray(facets_raw)
    if facets.ndim != 2 or facets.shape[0] != 2:
        raise FractureLoadError("mesh.mesh.facets must have shape [2, F]")
    if not np.issubdtype(facets.dtype, np.integer):
        raise FractureLoadError("mesh.mesh.facets must contain integer node identifiers")
    if wall_ids.min() < 0 or wall_ids.max() >= facets.shape[1]:
        raise FractureLoadError("wall facet identifier is outside mesh.mesh.facets")
    edges = np.asarray(facets[:, wall_ids].T, dtype=np.int64)
    if edges.min() < 0 or edges.max() >= nodes.shape[0]:
        raise FractureLoadError("wall facet contains an out-of-range node identifier")
    start = nodes[edges[:, 0]]
    end = nodes[edges[:, 1]]
    lengths = np.linalg.norm(end - start, axis=1)
    scale = max(float(np.ptp(nodes, axis=0).max()), 1.0)
    if np.any(lengths <= 64.0 * np.finfo(np.float64).eps * scale):
        raise FractureLoadError("wall facets must have positive finite length")
    midpoints = 0.5 * (start + end)
    perimeter_centroid = np.sum(midpoints * lengths[:, None], axis=0) / float(lengths.sum())
    radial_distance = np.linalg.norm(midpoints - perimeter_centroid, axis=1)
    if np.any(radial_distance <= 64.0 * np.finfo(np.float64).eps * scale):
        raise FractureLoadError("a wall facet midpoint coincides with the perimeter centroid")
    return (
        wall_ids,
        np.asarray(midpoints, dtype=np.float64),
        np.asarray(perimeter_centroid, dtype=np.float64),
    )


def _wall_zone_weights(midpoints_yz: FloatArray, perimeter_centroid_yz: FloatArray) -> FloatArray:
    relative = midpoints_yz - perimeter_centroid_yz
    theta_deg = np.mod(np.degrees(np.arctan2(relative[:, 1], relative[:, 0])), 360.0)
    hard_zone = np.floor(np.mod(theta_deg + 45.0, 360.0) / 90.0).astype(np.int64)
    weights = np.zeros((theta_deg.size, len(WALL_ZONE_IDS)), dtype=np.float64)
    weights[np.arange(theta_deg.size), hard_zone] = 1.0
    for left_zone, boundary_deg in enumerate(_ZONE_BOUNDARIES_DEG):
        right_zone = (left_zone + 1) % len(WALL_ZONE_IDS)
        signed_offset = np.mod(theta_deg - boundary_deg + 180.0, 360.0) - 180.0
        transition = np.abs(signed_offset) <= _TRANSITION_HALF_WIDTH_DEG
        if not np.any(transition):
            continue
        weights[transition] = 0.0
        weights[transition, left_zone] = (
            _TRANSITION_HALF_WIDTH_DEG - signed_offset[transition]
        ) / _TRANSITION_TOTAL_WIDTH_DEG
        weights[transition, right_zone] = (
            _TRANSITION_HALF_WIDTH_DEG + signed_offset[transition]
        ) / _TRANSITION_TOTAL_WIDTH_DEG
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _path_controls(
    config: Mapping[str, Any], path_id: str
) -> tuple[FloatArray, FloatArray, FloatArray]:
    paths: Sequence[Mapping[str, Any]] = config["load_paths"]["paths"]
    path = next((item for item in paths if item["id"] == path_id), None)
    if path is None:
        raise FractureLoadError(f"unknown Phase-1 path_id {path_id!r}")
    control_s: list[float] = []
    principal: list[list[float]] = []
    releases: list[list[float]] = []
    for knot in path["control_knots"]:
        control_s.append(float(knot["s"]))
        principal.append(
            [
                float(knot["sigma1_over_UCS"]),
                float(knot["sigma3_over_sigma1"]),
                float(knot["principal_angle_deg"]),
            ]
        )
        wall_release = knot["wall_release"]
        if path_id == "p4":
            releases.append([float(wall_release[zone]) for zone in WALL_ZONE_IDS])
        else:
            releases.append([float(wall_release["all"])] * len(WALL_ZONE_IDS))
    return (
        np.asarray(control_s, dtype=np.float64),
        np.asarray(principal, dtype=np.float64),
        np.asarray(releases, dtype=np.float64),
    )


def compile_phase1_load_schedule(
    config: Mapping[str, Any], path_id: str, ucs_scale: Real, mesh: Any
) -> Phase1LoadSchedule:
    """Compile one frozen path against the actual, ID-aligned wall facets.

    ``ucs_scale`` supplies the dimensional positive stress scale used to turn
    the normalized principal controls into the returned solver tensor.
    """

    validate_fracture_phase1_config(config)
    if path_id not in LOAD_PATH_IDS:
        raise FractureLoadError(f"unknown Phase-1 path_id {path_id!r}")
    resolved_ucs = _finite_scalar(ucs_scale, name="ucs_scale")
    if resolved_ucs <= 0.0:
        raise FractureLoadError("ucs_scale must be positive")
    wall_ids, midpoints, centroid = _extract_wall_geometry(mesh)
    weights = _wall_zone_weights(midpoints, centroid)
    control_s, principal, releases = _path_controls(config, path_id)
    return Phase1LoadSchedule(
        path_id=path_id,
        ucs_scale=resolved_ucs,
        wall_facet_ids=wall_ids,
        wall_facet_midpoints_yz=midpoints,
        wall_perimeter_centroid_yz=centroid,
        wall_zone_ids=WALL_ZONE_IDS,
        wall_zone_weights=weights,
        _control_s=control_s,
        _principal_controls=principal,
        _zone_release_controls=releases,
    )


__all__ = [
    "WALL_ZONE_IDS",
    "FractureLoadError",
    "FractureLoadState",
    "Phase1LoadSchedule",
    "compile_phase1_load_schedule",
]
