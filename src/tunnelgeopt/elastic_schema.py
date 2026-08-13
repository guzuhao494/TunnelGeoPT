"""Strict persistence contract for the independent B-elastic data layer.

The GeoPT-compatible A-layer intentionally has a small fixed-width schema.
This module keeps the plane-strain finite-element labels in a separate,
lossless record.  It serializes one case as ``arrays.npz`` plus ``meta.json``
and validates both integrity and the physical component conventions on every
load.

The schema contains linear-elastic fields only.  Dynamic, damage, and
inelastic-energy labels are rejected instead of being represented by
placeholder zero arrays.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

ARRAYS_FILENAME = "arrays.npz"
META_FILENAME = "meta.json"
SCHEMA_NAME = "tunnelgeopt.b_elastic"
SCHEMA_VERSION = 1

FLOAT64 = np.dtype(np.float64)
FLOAT32 = np.dtype(np.float32)
SUPPORTED_PUBLICATION_DTYPES = (FLOAT64, FLOAT32)

COORDINATE_ORDER = ("y", "z")
STRAIN_COMPONENT_ORDER = ("yy", "zz", "gamma_yz")
STRESS_COMPONENT_ORDER = ("yy", "zz", "yz")
SIGN_CONVENTION = "tension_positive"

SI_UNITS: dict[str, str] = {
    "nodes": "m",
    "u": "m",
    "strain": "1",
    "stress": "Pa",
    "delta_stress": "Pa",
    "sigma_inf": "Pa",
    "sigma_xx": "Pa",
    "energy_density": "J/m^3",
    "area": "m^2",
    "centers": "m",
    "energy": "J/m",
    "external_work": "J/m",
    "residual_norm": "N/m",
    "relative_diagnostics": "1",
}

DIAGNOSTIC_KEYS = (
    "energy",
    "external_work",
    "algebraic_residual",
    "residual_norm",
    "energy_closure",
    "energy_discretization_error",
    "stiffness_symmetry_error",
)

ARRAY_KEYS = (
    "nodes",
    "elements",
    "wall_facets",
    "farfield_facets",
    "u",
    "strain",
    "stress",
    "delta_stress",
    "sigma_inf",
    "sigma_xx",
    "energy_density",
    "area",
    "centers",
)

_FLOAT_ARRAY_KEYS = frozenset(
    {
        "nodes",
        "u",
        "strain",
        "stress",
        "delta_stress",
        "sigma_inf",
        "sigma_xx",
        "energy_density",
        "area",
        "centers",
    }
)
_INTEGER_ARRAY_KEYS = frozenset({"elements", "wall_facets", "farfield_facets"})
_FORBIDDEN_FIELD_TOKENS = ("damage", "velocity", "dissipation")
_HEX_DIGITS = frozenset("0123456789abcdef")

_META_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "dtype",
        "case_group_id",
        "mesh_id",
        "mesh_content_sha256",
        "config_hash",
        "coordinate_order",
        "strain_component_order",
        "stress_component_order",
        "sign_convention",
        "units",
        "diagnostics",
        "sigma_xx_inf",
        "material",
        "physical_tags",
        "mesh_metadata",
        "env",
        "meta",
        "array_manifest",
        "arrays_file_sha256",
        "content_sha256",
    }
)


class ElasticSchemaValidationError(ValueError):
    """Raised when a B-elastic record violates the persistence contract."""


@dataclass(frozen=True)
class ElasticRecordPaths:
    """Resolved files for one independently persisted elastic case."""

    case_dir: Path
    arrays: Path
    meta: Path


@dataclass(frozen=True)
class ElasticRecord:
    """Validated, SI-unit plane-strain fields for one mesh and load case.

    ``wall_facets`` and ``farfield_facets`` are explicit undirected node-index
    pairs with shape ``[B, 2]``.  This is deliberately more portable than the
    solver-specific global facet-number arrays held by :class:`ElasticResult`.
    """

    nodes: np.ndarray
    elements: np.ndarray
    wall_facets: np.ndarray
    farfield_facets: np.ndarray
    u: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    delta_stress: np.ndarray
    sigma_inf: np.ndarray
    sigma_xx: np.ndarray
    energy_density: np.ndarray
    area: np.ndarray
    centers: np.ndarray
    diagnostics: Mapping[str, float]
    case_group_id: str
    mesh_id: str
    config_hash: str
    env: Mapping[str, Any]
    meta: Mapping[str, Any]
    sigma_xx_inf: float
    material: Mapping[str, float]
    physical_tags: Mapping[str, int]
    mesh_metadata: Mapping[str, Any]
    coordinate_order: tuple[str, str] = COORDINATE_ORDER
    strain_component_order: tuple[str, str, str] = STRAIN_COMPONENT_ORDER
    stress_component_order: tuple[str, str, str] = STRESS_COMPONENT_ORDER
    sign_convention: str = SIGN_CONVENTION
    units: Mapping[str, str] = field(default_factory=lambda: dict(SI_UNITS))

    @property
    def dtype(self) -> np.dtype:
        return self.nodes.dtype

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_elements(self) -> int:
        return int(self.elements.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the exact array payload used by ``arrays.npz``."""

        return {name: np.asarray(getattr(self, name)) for name in ARRAY_KEYS}

    def validate(self, *, expected_dtype: Any | None = FLOAT64) -> None:
        """Fully validate topology, values, conventions, units, and provenance."""

        validate_elastic_record(self, expected_dtype=expected_dtype)


def elastic_record_paths(case_dir: str | os.PathLike[str]) -> ElasticRecordPaths:
    root = Path(case_dir)
    return ElasticRecordPaths(root, root / ARRAYS_FILENAME, root / META_FILENAME)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ElasticSchemaValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _normalise_json(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ElasticSchemaValidationError(f"{path} contains a non-finite number")
        return 0 if number == 0.0 else number
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ElasticSchemaValidationError(f"{path} mapping keys must be non-empty strings")
            lowered = key.casefold()
            if any(token in lowered for token in _FORBIDDEN_FIELD_TOKENS):
                raise ElasticSchemaValidationError(
                    f"{path}.{key} is outside the linear-elastic schema"
                )
            result[key] = _normalise_json(item, path=f"{path}.{key}")
        return dict(sorted(result.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalise_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ElasticSchemaValidationError(
        f"{path} contains unsupported JSON type {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalise_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _require_float_array(
    name: str, value: Any, shape: tuple[int | None, ...], dtype: np.dtype
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ElasticSchemaValidationError(f"{name} must be a numpy.ndarray")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        expected_shape = "[" + ",".join("*" if item is None else str(item) for item in shape) + "]"
        raise ElasticSchemaValidationError(
            f"{name} must have shape {expected_shape}; got {value.shape}"
        )
    if value.dtype != dtype:
        raise ElasticSchemaValidationError(
            f"{name} must use the shared dtype {dtype.name}; got {value.dtype}"
        )
    if not np.isfinite(value).all():
        raise ElasticSchemaValidationError(f"{name} contains a non-finite value")
    return value


def _require_index_array(
    name: str, value: Any, shape: tuple[int | None, ...], upper_bound: int
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ElasticSchemaValidationError(f"{name} must be a numpy.ndarray")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise ElasticSchemaValidationError(f"{name} has invalid shape {value.shape}")
    if value.dtype.kind not in "iu" or value.dtype.kind == "b":
        raise ElasticSchemaValidationError(f"{name} must use an integer dtype")
    if value.shape[0] <= 0:
        raise ElasticSchemaValidationError(f"{name} must not be empty")
    if value.min() < 0 or value.max() >= upper_bound:
        raise ElasticSchemaValidationError(f"{name} contains an index outside [0, {upper_bound})")
    return value


def _all_facets_with_counts(elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.concatenate([elements[:, [0, 1]], elements[:, [1, 2]], elements[:, [2, 0]]], axis=0)
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    return np.unique(edges, axis=0, return_counts=True)


def _normalise_facets(facets: np.ndarray) -> np.ndarray:
    return np.unique(np.sort(np.asarray(facets, dtype=np.int64), axis=1), axis=0)


def _relative_tolerances(dtype: np.dtype) -> tuple[float, float]:
    if dtype == FLOAT32:
        return 5.0e-5, 2.0e-6
    return 2.0e-11, 2.0e-13


def _validate_identifiers(record: ElasticRecord) -> None:
    _require_sha256(record.case_group_id, "case_group_id")
    _require_sha256(record.mesh_id, "mesh_id")
    _require_sha256(record.config_hash, "config_hash")


def _validate_conventions(record: ElasticRecord) -> None:
    if tuple(record.coordinate_order) != COORDINATE_ORDER:
        raise ElasticSchemaValidationError(
            f"coordinate_order must be {COORDINATE_ORDER}, in tunnel-axis-transverse order"
        )
    if tuple(record.strain_component_order) != STRAIN_COMPONENT_ORDER:
        raise ElasticSchemaValidationError(
            f"strain_component_order must be {STRAIN_COMPONENT_ORDER}"
        )
    if tuple(record.stress_component_order) != STRESS_COMPONENT_ORDER:
        raise ElasticSchemaValidationError(
            f"stress_component_order must be {STRESS_COMPONENT_ORDER}"
        )
    if record.sign_convention != SIGN_CONVENTION:
        raise ElasticSchemaValidationError(
            f"sign_convention must be {SIGN_CONVENTION!r}; compression is negative"
        )
    normalised_units = _normalise_json(record.units, path="$.units")
    if normalised_units != SI_UNITS:
        raise ElasticSchemaValidationError(
            "units must exactly match the B-elastic SI unit contract"
        )


def _validate_metadata(record: ElasticRecord) -> tuple[dict[str, float], dict[str, float]]:
    diagnostics = _normalise_json(record.diagnostics, path="$.diagnostics")
    if set(diagnostics) != set(DIAGNOSTIC_KEYS):
        raise ElasticSchemaValidationError(
            "diagnostics must contain exactly " + ", ".join(DIAGNOSTIC_KEYS)
        )
    for name in DIAGNOSTIC_KEYS:
        value = diagnostics[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ElasticSchemaValidationError(f"diagnostics.{name} must be numeric")
        if value < 0.0:
            raise ElasticSchemaValidationError(f"diagnostics.{name} must be non-negative")

    material = _normalise_json(record.material, path="$.material")
    required_material = {
        "young_modulus",
        "poisson_ratio",
        "lame_lambda",
        "shear_modulus",
    }
    if not required_material.issubset(material):
        missing = sorted(required_material - set(material))
        raise ElasticSchemaValidationError(f"material is missing required fields: {missing}")
    for key in required_material:
        if isinstance(material[key], bool) or not isinstance(material[key], (int, float)):
            raise ElasticSchemaValidationError(f"material.{key} must be numeric")
    young_modulus = float(material["young_modulus"])
    poisson_ratio = float(material["poisson_ratio"])
    if young_modulus <= 0.0 or not -1.0 < poisson_ratio < 0.5:
        raise ElasticSchemaValidationError("material E/nu lies outside the elastic domain")
    expected_mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    expected_lambda = (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    if not math.isclose(
        float(material["shear_modulus"]), expected_mu, rel_tol=2e-12, abs_tol=0.0
    ) or not math.isclose(
        float(material["lame_lambda"]), expected_lambda, rel_tol=2e-12, abs_tol=0.0
    ):
        raise ElasticSchemaValidationError("material Lame parameters do not match E and nu")

    physical_tags = _normalise_json(record.physical_tags, path="$.physical_tags")
    if set(physical_tags) != {"rock", "wall", "farfield"} or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in physical_tags.values()
    ):
        raise ElasticSchemaValidationError(
            "physical_tags must contain positive integer rock/wall/farfield tags"
        )
    _normalise_json(record.mesh_metadata, path="$.mesh_metadata")
    _normalise_json(record.env, path="$.env")
    _normalise_json(record.meta, path="$.meta")
    if not isinstance(record.sigma_xx_inf, Real) or isinstance(record.sigma_xx_inf, bool):
        raise ElasticSchemaValidationError("sigma_xx_inf must be a real scalar")
    if not np.isfinite(float(record.sigma_xx_inf)):
        raise ElasticSchemaValidationError("sigma_xx_inf must be finite")
    return diagnostics, material


def validate_elastic_record(record: ElasticRecord, *, expected_dtype: Any | None = FLOAT64) -> None:
    """Validate the complete B-elastic record without mutating it."""

    if not isinstance(record, ElasticRecord):
        raise TypeError("record must be an ElasticRecord")
    dtype = record.dtype
    if dtype not in SUPPORTED_PUBLICATION_DTYPES:
        raise ElasticSchemaValidationError(
            f"floating arrays must use float64 or explicitly published float32; got {dtype}"
        )
    if expected_dtype is not None:
        required_dtype = np.dtype(expected_dtype)
        if required_dtype not in SUPPORTED_PUBLICATION_DTYPES:
            raise ValueError("expected_dtype must be float64, float32, or None")
        if dtype != required_dtype:
            raise ElasticSchemaValidationError(
                f"expected {required_dtype.name} publication; got {dtype.name}"
            )

    _validate_identifiers(record)
    _validate_conventions(record)
    diagnostics, material = _validate_metadata(record)

    nodes = _require_float_array("nodes", record.nodes, (None, 2), dtype)
    if nodes.shape[0] < 3:
        raise ElasticSchemaValidationError("nodes must contain at least three points")
    elements = _require_index_array("elements", record.elements, (None, 3), nodes.shape[0])
    if np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0):
        raise ElasticSchemaValidationError("elements contain a repeated local node")
    triangles = nodes[elements]
    twice_signed_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    scale = np.maximum(np.max(np.abs(triangles), axis=(1, 2)) ** 2, 1.0)
    if np.any(np.abs(twice_signed_area) <= 1.0e-14 * scale):
        raise ElasticSchemaValidationError("elements contain a degenerate triangle")

    element_count = elements.shape[0]
    wall = _require_index_array("wall_facets", record.wall_facets, (None, 2), nodes.shape[0])
    farfield = _require_index_array(
        "farfield_facets", record.farfield_facets, (None, 2), nodes.shape[0]
    )
    wall_normalised = _normalise_facets(wall)
    farfield_normalised = _normalise_facets(farfield)
    if (
        wall_normalised.shape[0] != wall.shape[0]
        or farfield_normalised.shape[0] != farfield.shape[0]
    ):
        raise ElasticSchemaValidationError("boundary facets must be unique undirected edges")
    if np.intersect1d(
        wall_normalised.view([("a", wall_normalised.dtype), ("b", wall_normalised.dtype)]),
        farfield_normalised.view(
            [("a", farfield_normalised.dtype), ("b", farfield_normalised.dtype)]
        ),
    ).size:
        raise ElasticSchemaValidationError("wall and farfield facets overlap")
    all_facets, facet_counts = _all_facets_with_counts(elements)
    boundary = all_facets[facet_counts == 1]
    supplied_boundary = np.unique(np.vstack([wall_normalised, farfield_normalised]), axis=0)
    if not np.array_equal(boundary, supplied_boundary):
        raise ElasticSchemaValidationError(
            "wall and farfield facets must be disjoint and cover the complete mesh boundary"
        )

    u = _require_float_array("u", record.u, (nodes.shape[0], 2), dtype)
    strain = _require_float_array("strain", record.strain, (element_count, 3), dtype)
    stress = _require_float_array("stress", record.stress, (element_count, 3), dtype)
    delta_stress = _require_float_array(
        "delta_stress", record.delta_stress, (element_count, 3), dtype
    )
    sigma_inf = _require_float_array("sigma_inf", record.sigma_inf, (2, 2), dtype)
    sigma_xx = _require_float_array("sigma_xx", record.sigma_xx, (element_count,), dtype)
    energy_density = _require_float_array(
        "energy_density", record.energy_density, (element_count,), dtype
    )
    area = _require_float_array("area", record.area, (element_count,), dtype)
    centers = _require_float_array("centers", record.centers, (element_count, 2), dtype)
    if not np.allclose(sigma_inf, sigma_inf.T, rtol=0.0, atol=0.0):
        raise ElasticSchemaValidationError("sigma_inf must be exactly symmetric")
    if np.any(area <= 0.0):
        raise ElasticSchemaValidationError("area must be strictly positive")

    rtol, atol = _relative_tolerances(dtype)
    geometric_area = 0.5 * np.abs(twice_signed_area)
    if not np.allclose(area, geometric_area, rtol=rtol, atol=atol * max(float(area.max()), 1.0)):
        raise ElasticSchemaValidationError("area does not match nodes/elements geometry")
    geometric_centers = triangles.mean(axis=1)
    if not np.allclose(centers, geometric_centers, rtol=rtol, atol=atol):
        raise ElasticSchemaValidationError("centers do not match nodes/elements geometry")

    sigma_vector = np.asarray([sigma_inf[0, 0], sigma_inf[1, 1], sigma_inf[0, 1]], dtype=dtype)
    if not np.allclose(stress, delta_stress + sigma_vector, rtol=rtol, atol=atol):
        raise ElasticSchemaValidationError(
            "stress must be total [yy,zz,yz] stress = delta_stress + sigma_inf"
        )
    expected_energy_density = 0.5 * np.sum(strain * delta_stress, axis=1)
    energy_scale = max(float(np.max(np.abs(expected_energy_density))), 1.0)
    if not np.allclose(
        energy_density,
        expected_energy_density,
        rtol=rtol,
        atol=atol * energy_scale,
    ):
        raise ElasticSchemaValidationError(
            "energy_density must equal 0.5 * strain:delta_stress using engineering shear"
        )
    expected_sigma_xx = float(record.sigma_xx_inf) + float(material["lame_lambda"]) * (
        strain[:, 0] + strain[:, 1]
    )
    sigma_scale = max(float(np.max(np.abs(expected_sigma_xx))), 1.0)
    if not np.allclose(sigma_xx, expected_sigma_xx, rtol=rtol, atol=atol * sigma_scale):
        raise ElasticSchemaValidationError(
            "sigma_xx is inconsistent with plane strain, material, and sigma_xx_inf"
        )
    integrated_energy = float(np.sum(energy_density.astype(np.float64) * area))
    if not math.isclose(
        float(diagnostics["energy"]),
        integrated_energy,
        rel_tol=max(rtol, 2.0e-6),
        abs_tol=atol * max(abs(integrated_energy), 1.0),
    ):
        raise ElasticSchemaValidationError(
            "diagnostics.energy does not equal the integrated element energy"
        )
    # Read the variable so accidental replacement with a scalar cannot evade
    # the array validation above; no kinematic-to-strain reconstruction is
    # imposed here because imported solvers may project strains differently.
    _ = u


def _facets_from_result(result: Any, elements: np.ndarray, label: str) -> np.ndarray:
    try:
        raw = np.asarray(result.boundary_facets[label])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ElasticSchemaValidationError(
            f"ElasticResult.boundary_facets is missing {label!r}"
        ) from exc
    if raw.ndim == 2 and raw.shape[1] == 2:
        return _normalise_facets(raw)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.size == 0:
        raise ElasticSchemaValidationError(
            f"ElasticResult boundary {label!r} must contain facet ids or [B,2] edges"
        )
    all_facets, _ = _all_facets_with_counts(elements)
    ids = np.asarray(raw, dtype=np.int64)
    if ids.min() < 0 or ids.max() >= all_facets.shape[0]:
        raise ElasticSchemaValidationError(f"ElasticResult {label!r} facet id is out of range")
    return all_facets[ids]


def compute_mesh_content_sha256(
    nodes: np.ndarray,
    elements: np.ndarray,
    wall_facets: np.ndarray,
    farfield_facets: np.ndarray,
) -> str:
    """Hash mesh topology and coordinates with a deterministic array encoding."""

    return _semantic_array_sha256(
        {
            "nodes": np.asarray(nodes),
            "elements": np.asarray(elements),
            "wall_facets": _normalise_facets(wall_facets),
            "farfield_facets": _normalise_facets(farfield_facets),
        }
    )


def elastic_record_from_result(
    result: Any,
    *,
    case_group_id: str,
    config_hash: str,
    env: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
    mesh_id: str | None = None,
    publication_dtype: Any = FLOAT64,
) -> ElasticRecord:
    """Convert an :class:`elasticity.ElasticResult` into the strict record.

    Computation outputs are retained as ``float64`` by default.  Passing
    ``publication_dtype=np.float32`` is the only supported down-cast route and
    makes that publication decision explicit at the call site.
    """

    dtype = np.dtype(publication_dtype)
    if dtype not in SUPPORTED_PUBLICATION_DTYPES:
        raise ValueError("publication_dtype must be float64 or float32")

    def floating(name: str, source_name: str | None = None) -> np.ndarray:
        try:
            value = getattr(result, source_name or name)
        except AttributeError as exc:
            raise ElasticSchemaValidationError(
                f"ElasticResult is missing required field {source_name or name!r}"
            ) from exc
        converted = np.ascontiguousarray(np.asarray(value, dtype=dtype))
        if not np.isfinite(converted).all():
            raise ElasticSchemaValidationError(
                f"ElasticResult field {source_name or name!r} is non-finite after publication cast"
            )
        return converted

    nodes = floating("nodes")
    elements = np.ascontiguousarray(np.asarray(result.elements, dtype=np.int64))
    wall_facets = _facets_from_result(result, elements, "wall")
    farfield_facets = _facets_from_result(result, elements, "farfield")
    derived_mesh_hash = compute_mesh_content_sha256(nodes, elements, wall_facets, farfield_facets)
    resolved_mesh_id = derived_mesh_hash if mesh_id is None else mesh_id

    diagnostics = {name: float(getattr(result, name)) for name in DIAGNOSTIC_KEYS}
    record = ElasticRecord(
        nodes=nodes,
        elements=elements,
        wall_facets=np.ascontiguousarray(wall_facets, dtype=np.int64),
        farfield_facets=np.ascontiguousarray(farfield_facets, dtype=np.int64),
        u=floating("u", "displacement"),
        strain=floating("strain"),
        stress=floating("stress", "total_stress"),
        delta_stress=floating("delta_stress"),
        sigma_inf=floating("sigma_inf"),
        sigma_xx=floating("sigma_xx"),
        energy_density=floating("energy_density"),
        area=floating("area", "element_area"),
        centers=floating("centers", "element_centers"),
        diagnostics=diagnostics,
        case_group_id=case_group_id,
        mesh_id=resolved_mesh_id,
        config_hash=config_hash,
        env=dict(env),
        meta=dict(meta or {}),
        sigma_xx_inf=float(result.sigma_xx_inf),
        material=dict(result.material),
        physical_tags=dict(result.physical_tags),
        mesh_metadata=dict(result.mesh_metadata),
    )
    record.validate(expected_dtype=dtype)
    return record


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_array_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        header = _canonical_json(
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)}
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    manifest: dict[str, Any] = {}
    for name in ARRAY_KEYS:
        value = np.ascontiguousarray(arrays[name])
        manifest[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _semantic_array_sha256({name: value}),
        }
    return manifest


def _record_meta(record: ElasticRecord, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    mesh_content_hash = compute_mesh_content_sha256(
        record.nodes,
        record.elements,
        record.wall_facets,
        record.farfield_facets,
    )
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dtype": record.dtype.name,
        "case_group_id": record.case_group_id,
        "mesh_id": record.mesh_id,
        "mesh_content_sha256": mesh_content_hash,
        "config_hash": record.config_hash,
        "coordinate_order": list(record.coordinate_order),
        "strain_component_order": list(record.strain_component_order),
        "stress_component_order": list(record.stress_component_order),
        "sign_convention": record.sign_convention,
        "units": _normalise_json(record.units, path="$.units"),
        "diagnostics": _normalise_json(record.diagnostics, path="$.diagnostics"),
        "sigma_xx_inf": float(record.sigma_xx_inf),
        "material": _normalise_json(record.material, path="$.material"),
        "physical_tags": _normalise_json(record.physical_tags, path="$.physical_tags"),
        "mesh_metadata": _normalise_json(record.mesh_metadata, path="$.mesh_metadata"),
        "env": _normalise_json(record.env, path="$.env"),
        "meta": _normalise_json(record.meta, path="$.meta"),
        "array_manifest": _array_manifest(arrays),
    }


def _content_sha256(meta: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in meta.items()
        if key not in {"arrays_file_sha256", "content_sha256"}
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@contextmanager
def _writer_lock(case_dir: Path) -> Iterator[None]:
    lock_path = case_dir / ".elastic-schema.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"another elastic-record writer holds {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _write_npz_temp(directory: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", suffix=".npz", prefix=".arrays.", dir=directory, delete=False
        ) as stream:
            path = Path(stream.name)
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def _write_json_temp(directory: Path, value: Mapping[str, Any]) -> Path:
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".meta.",
            suffix=".json",
            dir=directory,
            delete=False,
        ) as stream:
            path = Path(stream.name)
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise


def save_elastic_record(
    case_dir: str | os.PathLike[str],
    record: ElasticRecord,
    *,
    overwrite: bool = False,
    expected_dtype: Any | None = FLOAT64,
) -> ElasticRecordPaths:
    """Validate and atomically replace each file of one elastic record.

    Existing records are protected unless ``overwrite=True``.  The array file
    is published before metadata, so an interrupted overwrite is detected as a
    hash mismatch instead of being accepted as a mixed record.
    """

    record.validate(expected_dtype=expected_dtype)
    paths = elastic_record_paths(case_dir)
    paths.case_dir.mkdir(parents=True, exist_ok=True)
    arrays = record.arrays()
    npz_temp: Path | None = None
    json_temp: Path | None = None
    with _writer_lock(paths.case_dir):
        existing = [path for path in (paths.arrays, paths.meta) if path.exists()]
        if existing and not overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"elastic record already has protected file(s): {names}; "
                "pass overwrite=True to replace both files"
            )
        try:
            npz_temp = _write_npz_temp(paths.case_dir, arrays)
            metadata = _record_meta(record, arrays)
            metadata["arrays_file_sha256"] = _hash_file(npz_temp)
            metadata["content_sha256"] = _content_sha256(metadata)
            json_temp = _write_json_temp(paths.case_dir, metadata)
            os.replace(npz_temp, paths.arrays)
            npz_temp = None
            os.replace(json_temp, paths.meta)
            json_temp = None
        finally:
            if npz_temp is not None:
                npz_temp.unlink(missing_ok=True)
            if json_temp is not None:
                json_temp.unlink(missing_ok=True)
    return paths


def save_elastic_result(
    case_dir: str | os.PathLike[str],
    result: Any,
    *,
    case_group_id: str,
    config_hash: str,
    env: Mapping[str, Any],
    meta: Mapping[str, Any] | None = None,
    mesh_id: str | None = None,
    publication_dtype: Any = FLOAT64,
    overwrite: bool = False,
) -> ElasticRecordPaths:
    """Convert, fully validate, and persist an ``ElasticResult``."""

    record = elastic_record_from_result(
        result,
        case_group_id=case_group_id,
        config_hash=config_hash,
        env=env,
        meta=meta,
        mesh_id=mesh_id,
        publication_dtype=publication_dtype,
    )
    return save_elastic_record(
        case_dir,
        record,
        overwrite=overwrite,
        expected_dtype=np.dtype(publication_dtype),
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required elastic metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ElasticSchemaValidationError(f"could not load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ElasticSchemaValidationError("meta.json root must be an object")
    if set(value) != _META_KEYS:
        missing = sorted(_META_KEYS - set(value))
        extra = sorted(set(value) - _META_KEYS)
        raise ElasticSchemaValidationError(
            f"meta.json key mismatch; missing={missing}, extra={extra}"
        )
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required elastic arrays: {path}")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(ARRAY_KEYS):
                missing = sorted(set(ARRAY_KEYS) - set(archive.files))
                extra = sorted(set(archive.files) - set(ARRAY_KEYS))
                raise ElasticSchemaValidationError(
                    f"arrays.npz key mismatch; missing={missing}, extra={extra}"
                )
            return {name: np.asarray(archive[name]) for name in ARRAY_KEYS}
    except ElasticSchemaValidationError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise ElasticSchemaValidationError(f"could not load {path}: {exc}") from exc


def _verify_manifest(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    expected = _array_manifest(arrays)
    if _normalise_json(metadata["array_manifest"], path="$.array_manifest") != expected:
        raise ElasticSchemaValidationError("array_manifest does not match arrays.npz")
    mesh_hash = compute_mesh_content_sha256(
        arrays["nodes"],
        arrays["elements"],
        arrays["wall_facets"],
        arrays["farfield_facets"],
    )
    if metadata["mesh_content_sha256"] != mesh_hash:
        raise ElasticSchemaValidationError("mesh_content_sha256 does not match the mesh arrays")


def load_elastic_record(
    case_dir: str | os.PathLike[str], *, expected_dtype: Any | None = FLOAT64
) -> ElasticRecord:
    """Load a record, verify both hashes, then repeat every semantic check."""

    paths = elastic_record_paths(case_dir)
    metadata = _load_json(paths.meta)
    if metadata.get("schema") != SCHEMA_NAME or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ElasticSchemaValidationError("unsupported B-elastic schema name or version")
    _require_sha256(metadata.get("arrays_file_sha256"), "arrays_file_sha256")
    _require_sha256(metadata.get("content_sha256"), "content_sha256")
    if _hash_file(paths.arrays) != metadata["arrays_file_sha256"]:
        raise ElasticSchemaValidationError("arrays.npz SHA-256 does not match meta.json")
    if _content_sha256(metadata) != metadata["content_sha256"]:
        raise ElasticSchemaValidationError("record content_sha256 does not match meta.json")

    arrays = _load_arrays(paths.arrays)
    _verify_manifest(arrays, metadata)
    try:
        record = ElasticRecord(
            **arrays,
            diagnostics=metadata["diagnostics"],
            case_group_id=metadata["case_group_id"],
            mesh_id=metadata["mesh_id"],
            config_hash=metadata["config_hash"],
            env=metadata["env"],
            meta=metadata["meta"],
            sigma_xx_inf=metadata["sigma_xx_inf"],
            material=metadata["material"],
            physical_tags=metadata["physical_tags"],
            mesh_metadata=metadata["mesh_metadata"],
            coordinate_order=tuple(metadata["coordinate_order"]),
            strain_component_order=tuple(metadata["strain_component_order"]),
            stress_component_order=tuple(metadata["stress_component_order"]),
            sign_convention=metadata["sign_convention"],
            units=metadata["units"],
        )
    except (TypeError, KeyError) as exc:
        raise ElasticSchemaValidationError(
            f"metadata cannot construct ElasticRecord: {exc}"
        ) from exc
    if metadata["dtype"] != record.dtype.name:
        raise ElasticSchemaValidationError("meta.json dtype does not match floating arrays")
    record.validate(expected_dtype=expected_dtype)
    return record


__all__ = [
    "ARRAYS_FILENAME",
    "ARRAY_KEYS",
    "COORDINATE_ORDER",
    "DIAGNOSTIC_KEYS",
    "FLOAT32",
    "FLOAT64",
    "META_FILENAME",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SIGN_CONVENTION",
    "SI_UNITS",
    "STRAIN_COMPONENT_ORDER",
    "STRESS_COMPONENT_ORDER",
    "SUPPORTED_PUBLICATION_DTYPES",
    "ElasticRecord",
    "ElasticRecordPaths",
    "ElasticSchemaValidationError",
    "compute_mesh_content_sha256",
    "elastic_record_from_result",
    "elastic_record_paths",
    "load_elastic_record",
    "save_elastic_record",
    "save_elastic_result",
    "validate_elastic_record",
]
