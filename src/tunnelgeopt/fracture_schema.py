"""Strict persistence contract for C-fracture trajectories.

This module is deliberately independent of the GeoPT compatibility arrays and
the B-elastic record.  A trajectory is a sequence of *accepted* quasi-static
AT2 states on one two-dimensional P1 triangular mesh.  It is published as an
exact ``arrays.npz`` payload plus ``meta.json`` and is revalidated on load.

The contract fixes ``d=0`` as intact and ``d=1`` as broken.  It has no storage
slots for elastic-only labels, acoustic-emission signals, dynamics, fragments,
or contact.  Those concepts must not be represented by zero-valued placeholders
or caller metadata.
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
from itertools import pairwise
from numbers import Integral, Real
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

ARRAYS_FILENAME = "arrays.npz"
META_FILENAME = "meta.json"
SCHEMA_NAME = "tunnelgeopt.c_fracture_trajectory"
SCHEMA_VERSION = 2

FLOAT64 = np.dtype(np.float64)
FLOAT32 = np.dtype(np.float32)
SUPPORTED_PUBLICATION_DTYPES = (FLOAT64, FLOAT32)

COORDINATE_ORDER = ("y", "z")
STRAIN_COMPONENT_ORDER = ("yy", "zz", "gamma_yz")
STRESS_COMPONENT_ORDER = ("yy", "zz", "yz")
SIGN_CONVENTION = "tension_positive"
DAMAGE_CONVENTION = "d=0_intact,d=1_broken"
FORMULATION = "2d_plane_strain_p1_displacement_p1_damage_at2"

EQUILIBRIUM_RELATIVE_TOLERANCE = 1.0e-6
KKT_RELATIVE_TOLERANCE = 1.0e-6
STATE_MONOTONICITY_TOLERANCE = 1.0e-10
ENERGY_IMBALANCE_TOLERANCE = 5.0e-2

SI_UNITS: dict[str, str] = {
    "nodes": "m",
    "node_ids": "id",
    "displacement_dof_ids": "id",
    "damage_dof_ids": "id",
    "elements": "index",
    "wall_facets": "index",
    "farfield_facets": "index",
    "farfield_dirichlet_dofs": "index",
    "area": "m^2",
    "centers": "m",
    "load_parameter": "1",
    "farfield_stress": "Pa",
    "wall_release_by_facet": "1",
    "u": "m",
    "internal_nodal_force": "N/m",
    "wall_nodal_force": "N/m",
    "farfield_prescribed_displacement": "m",
    "farfield_reaction_on_rock": "N/m",
    "damage": "1",
    "strain": "1",
    "stress": "Pa",
    "sigma_xx": "Pa",
    "psi_plus": "J/m^3",
    "psi_minus": "J/m^3",
    "history": "J/m^3",
    "elastic_energy": "J/m",
    "fracture_energy": "J/m",
    "neumann_load_functional": "J/m",
    "wall_work_increment": "J/m",
    "farfield_work_increment": "J/m",
    "cumulative_external_work": "J/m",
    "equilibrium_force_normalization_floor": "N/m",
    "energy_balance_normalization_floor": "J/m",
    "total_potential_energy": "J/m",
    "damage_area": "m^2",
    "crack_density_integral": "m",
    "damage_connectivity": "1",
    "displacement_residual": "1",
    "damage_residual": "1",
    "equilibrium_relative_residual": "1",
    "kkt_relative_residual": "1",
    "complementarity_relative_residual": "1",
    "damage_irreversibility_violation": "1",
    "damage_range_violation": "1",
    "history_monotonicity_violation": "J/m^3",
    "relative_energy_imbalance": "1",
    "newton_iterations": "count",
    "active_set_iterations": "count",
    "staggered_iterations": "count",
    "step_halvings": "count",
    "retry_count": "count",
    "elastic_basis_stress": "Pa",
    "nonlinear_stress_residual": "Pa",
}

FLOAT_ARRAY_KEYS = (
    "nodes",
    "area",
    "centers",
    "load_parameter",
    "farfield_stress",
    "wall_release_by_facet",
    "u",
    "internal_nodal_force",
    "wall_nodal_force",
    "farfield_prescribed_displacement",
    "farfield_reaction_on_rock",
    "damage",
    "strain",
    "stress",
    "sigma_xx",
    "psi_plus",
    "psi_minus",
    "history",
    "elastic_energy",
    "fracture_energy",
    "neumann_load_functional",
    "wall_work_increment",
    "farfield_work_increment",
    "cumulative_external_work",
    "total_potential_energy",
    "damage_area",
    "crack_density_integral",
    "damage_connectivity",
    "displacement_residual",
    "damage_residual",
    "equilibrium_relative_residual",
    "kkt_relative_residual",
    "complementarity_relative_residual",
    "damage_irreversibility_violation",
    "damage_range_violation",
    "history_monotonicity_violation",
    "relative_energy_imbalance",
)
INDEX_ARRAY_KEYS = (
    "node_ids",
    "displacement_dof_ids",
    "damage_dof_ids",
    "elements",
    "wall_facets",
    "farfield_facets",
    "farfield_dirichlet_dofs",
    "newton_iterations",
    "active_set_iterations",
    "staggered_iterations",
    "step_halvings",
    "retry_count",
)
ARRAY_KEYS = FLOAT_ARRAY_KEYS + INDEX_ARRAY_KEYS
OPTIONAL_ARRAY_KEYS = ("elastic_basis_stress", "nonlinear_stress_residual")

ATTEMPT_LEDGER_KEYS = frozenset(
    {
        "step_index",
        "attempt_index",
        "load_parameter_start",
        "load_parameter_target",
        "accepted",
        "failure_code",
        "failure_message",
        "newton_iterations",
        "active_set_iterations",
        "staggered_iterations",
        "step_halvings",
        "equilibrium_relative_residual",
        "kkt_relative_residual",
        "complementarity_relative_residual",
        "damage_irreversibility_violation",
        "damage_range_violation",
        "relative_energy_imbalance",
        "load_state_sha256",
        "neumann_load_functional",
        "wall_work_increment",
        "farfield_work_increment",
        "cumulative_external_work",
    }
)

_FORBIDDEN_KEY_PHRASES = (
    "elastic_only",
    "linear_elastic_only",
    "b_elastic_record",
    "elastic_record",
    "damage_disabled",
    "acoustic_emission",
    "ae_event",
    "ae_waveform",
    "microseismic",
    "waveform",
    "contact",
    "friction",
    "ejection",
    "fragment",
    "velocity",
    "acceleration",
    "inertia",
    "kinetic_energy",
    "plasticity",
    "plastic_strain",
)
_FORBIDDEN_EXACT_KEYS = frozenset({"ae"})
_HEX_DIGITS = frozenset("0123456789abcdef")

_META_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "dtype",
        "trajectory_id",
        "case_id",
        "mesh_id",
        "geometry_id",
        "material_id",
        "load_path_id",
        "config_hash",
        "solver_hash",
        "elastic_basis_id",
        "elastic_basis_sha256",
        "completed",
        "accepted",
        "mesh_content_sha256",
        "identity_content_sha256",
        "equilibrium_force_normalization_floor",
        "energy_balance_normalization_floor",
        "coordinate_order",
        "strain_component_order",
        "stress_component_order",
        "sign_convention",
        "damage_convention",
        "formulation",
        "units",
        "material",
        "geometry",
        "load_path",
        "physical_tags",
        "mesh_metadata",
        "solver",
        "env",
        "meta",
        "attempt_ledger",
        "array_manifest",
        "arrays_file_sha256",
        "content_sha256",
    }
)


class FractureSchemaValidationError(ValueError):
    """Raised when a C-fracture trajectory violates the persistence contract."""


@dataclass(frozen=True)
class FractureTrajectoryPaths:
    """Resolved files for one independently persisted fracture trajectory."""

    trajectory_dir: Path
    arrays: Path
    meta: Path


def _immutable_array(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FractureSchemaValidationError(f"{name} must be a numpy.ndarray")
    contiguous = np.ascontiguousarray(value)
    # A bytes-backed view cannot be made writable again with setflags().  This
    # is stronger than merely clearing the flag on an owning ndarray.
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _normalised_key(key: str) -> str:
    return "_".join(
        part for part in "".join(c.lower() if c.isalnum() else " " for c in key).split()
    )


def _reject_forbidden_key(key: str, path: str) -> None:
    normalised = _normalised_key(key)
    if normalised in _FORBIDDEN_EXACT_KEYS or any(
        phrase in normalised for phrase in _FORBIDDEN_KEY_PHRASES
    ):
        raise FractureSchemaValidationError(
            f"{path}.{key} is outside the quasi-static C-fracture schema"
        )


def _freeze_json(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise FractureSchemaValidationError(f"{path} contains a non-finite number")
        return 0.0 if number == 0.0 else number
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise FractureSchemaValidationError(
                    f"{path} mapping keys must be non-empty strings"
                )
            _reject_forbidden_key(key, path)
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise FractureSchemaValidationError(
        f"{path} contains unsupported JSON type {type(value).__name__}"
    )


def _normalise_json(value: Any, *, path: str = "$") -> Any:
    frozen = _freeze_json(value, path=path)
    if isinstance(frozen, Mapping):
        return {key: _normalise_json(item, path=f"{path}.{key}") for key, item in frozen.items()}
    if isinstance(frozen, tuple):
        return [_normalise_json(item, path=f"{path}[{index}]") for index, item in enumerate(frozen)]
    return frozen


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalise_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass(frozen=True)
class FractureTrajectory:
    """Immutable accepted AT2 trajectory on one 2D P1 triangular mesh.

    Every input array is copied into a bytes-backed, read-only snapshot during
    construction.  Caller mappings and the attempt ledger are recursively
    frozen.  Call :meth:`validate` before use; save and load always do so.
    """

    nodes: np.ndarray
    node_ids: np.ndarray
    displacement_dof_ids: np.ndarray
    damage_dof_ids: np.ndarray
    elements: np.ndarray
    wall_facets: np.ndarray
    farfield_facets: np.ndarray
    farfield_dirichlet_dofs: np.ndarray
    area: np.ndarray
    centers: np.ndarray
    load_parameter: np.ndarray
    farfield_stress: np.ndarray
    wall_release_by_facet: np.ndarray
    u: np.ndarray
    internal_nodal_force: np.ndarray
    wall_nodal_force: np.ndarray
    farfield_prescribed_displacement: np.ndarray
    farfield_reaction_on_rock: np.ndarray
    damage: np.ndarray
    strain: np.ndarray
    stress: np.ndarray
    sigma_xx: np.ndarray
    psi_plus: np.ndarray
    psi_minus: np.ndarray
    history: np.ndarray
    elastic_energy: np.ndarray
    fracture_energy: np.ndarray
    neumann_load_functional: np.ndarray
    wall_work_increment: np.ndarray
    farfield_work_increment: np.ndarray
    cumulative_external_work: np.ndarray
    total_potential_energy: np.ndarray
    damage_area: np.ndarray
    crack_density_integral: np.ndarray
    damage_connectivity: np.ndarray
    displacement_residual: np.ndarray
    damage_residual: np.ndarray
    equilibrium_relative_residual: np.ndarray
    kkt_relative_residual: np.ndarray
    complementarity_relative_residual: np.ndarray
    damage_irreversibility_violation: np.ndarray
    damage_range_violation: np.ndarray
    history_monotonicity_violation: np.ndarray
    relative_energy_imbalance: np.ndarray
    newton_iterations: np.ndarray
    active_set_iterations: np.ndarray
    staggered_iterations: np.ndarray
    step_halvings: np.ndarray
    retry_count: np.ndarray
    attempt_ledger: Sequence[Mapping[str, Any]]
    trajectory_id: str
    case_id: str
    mesh_id: str
    geometry_id: str
    material_id: str
    load_path_id: str
    config_hash: str
    solver_hash: str
    equilibrium_force_normalization_floor: float
    energy_balance_normalization_floor: float
    material: Mapping[str, Any]
    geometry: Mapping[str, Any]
    load_path: Mapping[str, Any]
    physical_tags: Mapping[str, int]
    mesh_metadata: Mapping[str, Any]
    solver: Mapping[str, Any]
    env: Mapping[str, Any]
    meta: Mapping[str, Any]
    elastic_basis_stress: np.ndarray | None = None
    nonlinear_stress_residual: np.ndarray | None = None
    elastic_basis_id: str | None = None
    elastic_basis_sha256: str | None = None
    completed: bool = True
    accepted: bool = True
    coordinate_order: tuple[str, str] = COORDINATE_ORDER
    strain_component_order: tuple[str, str, str] = STRAIN_COMPONENT_ORDER
    stress_component_order: tuple[str, str, str] = STRESS_COMPONENT_ORDER
    sign_convention: str = SIGN_CONVENTION
    damage_convention: str = DAMAGE_CONVENTION
    formulation: str = FORMULATION
    units: Mapping[str, str] = field(default_factory=lambda: dict(SI_UNITS))

    def __post_init__(self) -> None:
        for name in ARRAY_KEYS + OPTIONAL_ARRAY_KEYS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _immutable_array(value, name))
        for name in (
            "material",
            "geometry",
            "load_path",
            "physical_tags",
            "mesh_metadata",
            "solver",
            "env",
            "meta",
            "units",
        ):
            object.__setattr__(self, name, _freeze_json(getattr(self, name), path=f"$.{name}"))
        if not isinstance(self.attempt_ledger, Sequence) or isinstance(
            self.attempt_ledger, (str, bytes, bytearray)
        ):
            raise FractureSchemaValidationError("attempt_ledger must be a sequence of mappings")
        object.__setattr__(
            self,
            "attempt_ledger",
            tuple(
                _freeze_json(item, path=f"$.attempt_ledger[{index}]")
                for index, item in enumerate(self.attempt_ledger)
            ),
        )
        object.__setattr__(self, "coordinate_order", tuple(self.coordinate_order))
        object.__setattr__(self, "strain_component_order", tuple(self.strain_component_order))
        object.__setattr__(self, "stress_component_order", tuple(self.stress_component_order))

    @property
    def dtype(self) -> np.dtype:
        return self.nodes.dtype

    @property
    def num_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def num_elements(self) -> int:
        return int(self.elements.shape[0])

    @property
    def num_steps(self) -> int:
        return int(self.load_parameter.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        """Return the exact immutable array payload used by ``arrays.npz``."""

        arrays = {name: getattr(self, name) for name in ARRAY_KEYS}
        if self.elastic_basis_stress is not None:
            arrays["elastic_basis_stress"] = self.elastic_basis_stress
            arrays["nonlinear_stress_residual"] = self.nonlinear_stress_residual
        return arrays

    def validate(self, *, expected_dtype: Any | None = FLOAT64) -> None:
        """Validate topology, states, mechanics, ledger, units, and identity."""

        validate_fracture_trajectory(self, expected_dtype=expected_dtype)


def fracture_trajectory_paths(
    trajectory_dir: str | os.PathLike[str],
) -> FractureTrajectoryPaths:
    root = Path(trajectory_dir)
    return FractureTrajectoryPaths(root, root / ARRAYS_FILENAME, root / META_FILENAME)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise FractureSchemaValidationError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _require_identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise FractureSchemaValidationError(f"{name} must be a non-empty trimmed string")
    if len(value) > 256 or any(ord(character) < 32 for character in value):
        raise FractureSchemaValidationError(f"{name} contains an invalid character or is too long")
    return value


def _require_float_array(
    name: str, value: Any, shape: tuple[int | None, ...], dtype: np.dtype
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FractureSchemaValidationError(f"{name} must be a numpy.ndarray")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        expected_shape = "[" + ",".join("*" if item is None else str(item) for item in shape) + "]"
        raise FractureSchemaValidationError(
            f"{name} must have shape {expected_shape}; got {value.shape}"
        )
    if value.dtype != dtype:
        raise FractureSchemaValidationError(
            f"{name} must use the shared dtype {dtype.name}; got {value.dtype}"
        )
    if value.flags.writeable:
        raise FractureSchemaValidationError(f"{name} must be immutable/read-only")
    if not np.isfinite(value).all():
        raise FractureSchemaValidationError(f"{name} contains a non-finite value")
    return value


def _require_index_array(
    name: str,
    value: Any,
    shape: tuple[int | None, ...],
    *,
    upper_bound: int | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise FractureSchemaValidationError(f"{name} must be a numpy.ndarray")
    if value.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(value.shape, shape, strict=True)
    ):
        raise FractureSchemaValidationError(f"{name} has invalid shape {value.shape}")
    if value.dtype != np.dtype(np.int64):
        raise FractureSchemaValidationError(f"{name} must use int64")
    if value.flags.writeable:
        raise FractureSchemaValidationError(f"{name} must be immutable/read-only")
    if upper_bound is not None and value.size and (value.min() < 0 or value.max() >= upper_bound):
        raise FractureSchemaValidationError(f"{name} contains an index outside [0, {upper_bound})")
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


def _validate_identity(record: FractureTrajectory) -> None:
    for name in (
        "trajectory_id",
        "case_id",
        "mesh_id",
        "geometry_id",
        "material_id",
        "load_path_id",
    ):
        _require_identifier(getattr(record, name), name)
    _require_sha256(record.config_hash, "config_hash")
    _require_sha256(record.solver_hash, "solver_hash")
    linked = (record.elastic_basis_id is not None, record.elastic_basis_sha256 is not None)
    if linked[0] != linked[1]:
        raise FractureSchemaValidationError(
            "elastic_basis_id and elastic_basis_sha256 must be supplied together"
        )
    if record.elastic_basis_id is not None:
        _require_identifier(record.elastic_basis_id, "elastic_basis_id")
        _require_sha256(record.elastic_basis_sha256, "elastic_basis_sha256")


def _validate_conventions(record: FractureTrajectory) -> None:
    if tuple(record.coordinate_order) != COORDINATE_ORDER:
        raise FractureSchemaValidationError(f"coordinate_order must be {COORDINATE_ORDER}")
    if tuple(record.strain_component_order) != STRAIN_COMPONENT_ORDER:
        raise FractureSchemaValidationError(
            f"strain_component_order must be {STRAIN_COMPONENT_ORDER}"
        )
    if tuple(record.stress_component_order) != STRESS_COMPONENT_ORDER:
        raise FractureSchemaValidationError(
            f"stress_component_order must be {STRESS_COMPONENT_ORDER}"
        )
    if record.sign_convention != SIGN_CONVENTION:
        raise FractureSchemaValidationError(
            f"sign_convention must be {SIGN_CONVENTION!r}; compression is negative"
        )
    if record.damage_convention != DAMAGE_CONVENTION:
        raise FractureSchemaValidationError(f"damage_convention must be {DAMAGE_CONVENTION!r}")
    if record.formulation != FORMULATION:
        raise FractureSchemaValidationError(f"formulation must be {FORMULATION!r}")
    if _normalise_json(record.units, path="$.units") != SI_UNITS:
        raise FractureSchemaValidationError(
            "units must exactly match the C-fracture SI unit contract"
        )


def _require_numeric_mapping_value(
    mapping: Mapping[str, Any], name: str, *, positive: bool = False
) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FractureSchemaValidationError(f"material.{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and strictly positive" if positive else "finite"
        raise FractureSchemaValidationError(f"material.{name} must be {qualifier}")
    return number


def _numeric_schedule_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return bool(value) and all(_numeric_schedule_value(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_numeric_schedule_value(item) for item in value)
    return False


def _validate_load_path(load_path: Mapping[str, Any]) -> None:
    required = {"path_parameter", "interpolation", "control_knots"}
    if not required.issubset(load_path):
        missing = sorted(required - set(load_path))
        raise FractureSchemaValidationError(f"load_path is missing required fields: {missing}")
    descriptor = load_path["path_parameter"]
    if not isinstance(descriptor, str) or not descriptor.strip():
        raise FractureSchemaValidationError("load_path.path_parameter must be a non-empty string")
    interpolation = load_path["interpolation"]
    if not isinstance(interpolation, str) or not interpolation.strip():
        raise FractureSchemaValidationError("load_path.interpolation must be a non-empty string")
    if "parameter_bounds" in load_path and load_path["parameter_bounds"] not in (
        [0, 1],
        [0.0, 1.0],
    ):
        raise FractureSchemaValidationError("load_path.parameter_bounds must be [0,1]")
    if "monotone" in load_path and load_path["monotone"] is not True:
        raise FractureSchemaValidationError("load_path.monotone must be true when declared")

    knots = load_path["control_knots"]
    if (
        not isinstance(knots, list)
        or len(knots) < 2
        or any(not isinstance(knot, dict) for knot in knots)
    ):
        raise FractureSchemaValidationError(
            "load_path.control_knots must contain at least two mappings"
        )
    if descriptor in knots[0]:
        parameter_name = descriptor
    else:
        candidates = [key for key in knots[0] if descriptor.startswith(f"{key}_in_")]
        if len(candidates) != 1:
            raise FractureSchemaValidationError(
                "load_path.path_parameter must identify the coordinate key in every control knot"
            )
        parameter_name = candidates[0]

    knot_keys = set(knots[0])
    if parameter_name not in knot_keys or "wall_release" not in knot_keys:
        raise FractureSchemaValidationError(
            "every control knot must explicitly contain the path coordinate and wall_release"
        )
    farfield_keys = knot_keys - {parameter_name, "wall_release"}
    if not farfield_keys:
        raise FractureSchemaValidationError(
            "every control knot must explicitly contain a far-field schedule"
        )

    coordinates: list[float] = []
    wall_zone_keys: set[str] | None = None
    wall_zone_values: dict[str, list[float]] = {}
    for index, knot in enumerate(knots):
        if set(knot) != knot_keys:
            raise FractureSchemaValidationError(
                "all load_path.control_knots must use one explicit schedule key set"
            )
        coordinate = knot[parameter_name]
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            raise FractureSchemaValidationError(
                f"load_path.control_knots[{index}].{parameter_name} must be numeric"
            )
        coordinate_value = float(coordinate)
        if not math.isfinite(coordinate_value) or not 0.0 <= coordinate_value <= 1.0:
            raise FractureSchemaValidationError(
                f"load_path.control_knots[{index}].{parameter_name} must lie in [0,1]"
            )
        coordinates.append(coordinate_value)

        release = knot["wall_release"]
        if not isinstance(release, dict) or not release:
            raise FractureSchemaValidationError(
                f"load_path.control_knots[{index}].wall_release must be a non-empty mapping"
            )
        if wall_zone_keys is None:
            wall_zone_keys = set(release)
            wall_zone_values = {zone: [] for zone in wall_zone_keys}
        if set(release) != wall_zone_keys:
            raise FractureSchemaValidationError(
                "all control knots must use the same explicit wall-release zones"
            )
        for zone, value in release.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FractureSchemaValidationError(
                    f"load_path.control_knots[{index}].wall_release.{zone} must be numeric"
                )
            release_value = float(value)
            if not math.isfinite(release_value) or not 0.0 <= release_value <= 1.0:
                raise FractureSchemaValidationError(
                    f"load_path.control_knots[{index}].wall_release.{zone} must lie in [0,1]"
                )
            wall_zone_values[zone].append(release_value)
        for key in farfield_keys:
            if not _numeric_schedule_value(knot[key]):
                raise FractureSchemaValidationError(
                    f"load_path.control_knots[{index}].{key} must be an explicit numeric schedule"
                )

    if not math.isclose(coordinates[0], 0.0, abs_tol=0.0) or not math.isclose(
        coordinates[-1], 1.0, abs_tol=0.0
    ):
        raise FractureSchemaValidationError(
            "load_path.control_knots must cover the normalized path from 0 to 1"
        )
    if any(later <= earlier for earlier, later in pairwise(coordinates)):
        raise FractureSchemaValidationError(
            "load_path control-knot coordinates must be strictly increasing"
        )
    for zone, values in wall_zone_values.items():
        if any(later < earlier for earlier, later in pairwise(values)):
            raise FractureSchemaValidationError(
                f"load_path wall release for zone {zone!r} must be monotone"
            )

    assert wall_zone_keys is not None
    if wall_zone_keys != {"all"}:
        zone_definition = next(
            (
                load_path[name]
                for name in ("wall_zone_definition", "wall_zones", "wall_zones_for_p4")
                if name in load_path
            ),
            None,
        )
        if not isinstance(zone_definition, dict) or not zone_definition:
            raise FractureSchemaValidationError(
                "spatially staged wall release requires an explicit wall-zone definition"
            )


def _validate_metadata(record: FractureTrajectory) -> tuple[dict[str, Any], dict[str, Any]]:
    for name in (
        "equilibrium_force_normalization_floor",
        "energy_balance_normalization_floor",
    ):
        floor = getattr(record, name)
        if isinstance(floor, bool) or not isinstance(floor, Real):
            raise FractureSchemaValidationError(f"{name} must be numeric")
        if not math.isfinite(float(floor)) or float(floor) <= 0.0:
            raise FractureSchemaValidationError(f"{name} must be finite and strictly positive")

    material = _normalise_json(record.material, path="$.material")
    required_material = {
        "young_modulus",
        "poisson_ratio",
        "fracture_energy",
        "length_scale",
        "residual_stiffness",
        "fracture_model",
        "energy_split",
    }
    if not required_material.issubset(material):
        missing = sorted(required_material - set(material))
        raise FractureSchemaValidationError(f"material is missing required fields: {missing}")
    young_modulus = _require_numeric_mapping_value(material, "young_modulus", positive=True)
    poisson_ratio = _require_numeric_mapping_value(material, "poisson_ratio")
    _require_numeric_mapping_value(material, "fracture_energy", positive=True)
    _require_numeric_mapping_value(material, "length_scale", positive=True)
    residual_stiffness = _require_numeric_mapping_value(material, "residual_stiffness")
    if young_modulus <= 0.0 or not -1.0 < poisson_ratio < 0.5:
        raise FractureSchemaValidationError("material E/nu lies outside the elastic domain")
    if not 0.0 <= residual_stiffness < 1.0:
        raise FractureSchemaValidationError("material.residual_stiffness must lie in [0,1)")
    if material["fracture_model"] != "AT2":
        raise FractureSchemaValidationError("material.fracture_model must be 'AT2'")
    allowed_splits = {"spectral_strain_3d_plane_strain", "volumetric_deviatoric"}
    if material["energy_split"] not in allowed_splits:
        raise FractureSchemaValidationError(
            "material.energy_split must explicitly name a supported split"
        )

    geometry = _normalise_json(record.geometry, path="$.geometry")
    if not isinstance(geometry, dict) or not geometry:
        raise FractureSchemaValidationError("geometry must be a non-empty mapping")

    load_path = _normalise_json(record.load_path, path="$.load_path")
    _validate_load_path(load_path)

    physical_tags = _normalise_json(record.physical_tags, path="$.physical_tags")
    if set(physical_tags) != {"rock", "wall", "farfield"} or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in physical_tags.values()
    ):
        raise FractureSchemaValidationError(
            "physical_tags must contain positive integer rock/wall/farfield tags"
        )
    mesh_metadata = _normalise_json(record.mesh_metadata, path="$.mesh_metadata")
    required_mesh = {
        "element_type": "triangle_p1",
        "displacement_interpolation": "P1",
        "damage_interpolation": "P1",
    }
    for key, expected in required_mesh.items():
        if mesh_metadata.get(key) != expected:
            raise FractureSchemaValidationError(f"mesh_metadata.{key} must be {expected!r}")

    solver = _normalise_json(record.solver, path="$.solver")
    if not isinstance(solver.get("name"), str) or not solver["name"].strip():
        raise FractureSchemaValidationError("solver.name must be a non-empty string")
    _normalise_json(record.env, path="$.env")
    _normalise_json(record.meta, path="$.meta")
    if record.completed is not True or record.accepted is not True:
        raise FractureSchemaValidationError(
            "FractureTrajectory stores only completed, accepted trajectories"
        )
    return material, load_path


def _validate_mesh(record: FractureTrajectory, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    nodes = _require_float_array("nodes", record.nodes, (None, 2), dtype)
    if nodes.shape[0] < 3:
        raise FractureSchemaValidationError("nodes must contain at least three points")
    elements = _require_index_array(
        "elements", record.elements, (None, 3), upper_bound=nodes.shape[0]
    )
    if elements.shape[0] == 0:
        raise FractureSchemaValidationError("elements must not be empty")
    if np.any(np.diff(np.sort(elements, axis=1), axis=1) == 0):
        raise FractureSchemaValidationError("elements contain a repeated local node")
    triangles = nodes[elements]
    twice_signed_area = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    scale = np.maximum(np.max(np.abs(triangles), axis=(1, 2)) ** 2, 1.0)
    if np.any(np.abs(twice_signed_area) <= 1.0e-14 * scale):
        raise FractureSchemaValidationError("elements contain a degenerate triangle")

    wall = _require_index_array(
        "wall_facets", record.wall_facets, (None, 2), upper_bound=nodes.shape[0]
    )
    farfield = _require_index_array(
        "farfield_facets", record.farfield_facets, (None, 2), upper_bound=nodes.shape[0]
    )
    if wall.shape[0] == 0 or farfield.shape[0] == 0:
        raise FractureSchemaValidationError("wall and farfield facets must not be empty")
    wall_normalised = _normalise_facets(wall)
    farfield_normalised = _normalise_facets(farfield)
    if (
        wall_normalised.shape[0] != wall.shape[0]
        or farfield_normalised.shape[0] != farfield.shape[0]
    ):
        raise FractureSchemaValidationError("boundary facets must be unique undirected edges")
    wall_keys = {tuple(edge) for edge in wall_normalised.tolist()}
    farfield_keys = {tuple(edge) for edge in farfield_normalised.tolist()}
    if wall_keys & farfield_keys:
        raise FractureSchemaValidationError("wall and farfield facets overlap")
    all_facets, counts = _all_facets_with_counts(elements)
    boundary = all_facets[counts == 1]
    supplied = np.unique(np.vstack([wall_normalised, farfield_normalised]), axis=0)
    if not np.array_equal(boundary, supplied):
        raise FractureSchemaValidationError(
            "wall and farfield facets must be disjoint and cover the complete mesh boundary"
        )

    area = _require_float_array("area", record.area, (elements.shape[0],), dtype)
    centers = _require_float_array("centers", record.centers, (elements.shape[0], 2), dtype)
    if np.any(area <= 0.0):
        raise FractureSchemaValidationError("area must be strictly positive")
    rtol, atol = _relative_tolerances(dtype)
    geometric_area = 0.5 * np.abs(twice_signed_area)
    if not np.allclose(
        area,
        geometric_area,
        rtol=rtol,
        atol=atol * max(float(area.max()), 1.0),
    ):
        raise FractureSchemaValidationError("area does not match nodes/elements geometry")
    if not np.allclose(centers, triangles.mean(axis=1), rtol=rtol, atol=atol):
        raise FractureSchemaValidationError("centers do not match nodes/elements geometry")
    return triangles, twice_signed_area


def _validate_discrete_identity(record: FractureTrajectory) -> np.ndarray:
    """Validate row-stable node, field-DOF, and constrained-DOF identities."""

    node_count = record.num_nodes
    node_ids = _require_index_array("node_ids", record.node_ids, (node_count,))
    displacement_ids = _require_index_array(
        "displacement_dof_ids", record.displacement_dof_ids, (node_count, 2)
    )
    damage_ids = _require_index_array("damage_dof_ids", record.damage_dof_ids, (node_count,))
    for name, values in (
        ("node_ids", node_ids),
        ("displacement_dof_ids", displacement_ids),
        ("damage_dof_ids", damage_ids),
    ):
        if np.any(values < 0) or np.unique(values).size != values.size:
            raise FractureSchemaValidationError(
                f"{name} must contain unique non-negative field-local identifiers"
            )
    expected_displacement_ids = np.arange(2 * node_count, dtype=np.int64).reshape(node_count, 2)
    if not np.array_equal(displacement_ids, expected_displacement_ids):
        raise FractureSchemaValidationError(
            "displacement_dof_ids must equal node-major [2*i, 2*i+1] identifiers"
        )
    dirichlet = _require_index_array(
        "farfield_dirichlet_dofs",
        record.farfield_dirichlet_dofs,
        (None,),
        upper_bound=2 * node_count,
    )
    if dirichlet.size == 0:
        raise FractureSchemaValidationError("farfield_dirichlet_dofs must not be empty")
    if dirichlet.size > 1 and np.any(np.diff(dirichlet) <= 0):
        raise FractureSchemaValidationError(
            "farfield_dirichlet_dofs must be unique and strictly increasing"
        )
    farfield_nodes = np.unique(record.farfield_facets)
    if not np.all(np.isin(dirichlet // 2, farfield_nodes)):
        raise FractureSchemaValidationError(
            "farfield_dirichlet_dofs may reference only farfield-facet nodes"
        )
    return dirichlet


def _p1_damage_integrals(
    nodes: np.ndarray,
    elements: np.ndarray,
    area: np.ndarray,
    damage: np.ndarray,
    length_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    element_damage = damage[:, elements]
    mean_damage = element_damage.mean(axis=2)
    squares = np.sum(element_damage * element_damage, axis=2)
    pairs = (
        element_damage[:, :, 0] * element_damage[:, :, 1]
        + element_damage[:, :, 1] * element_damage[:, :, 2]
        + element_damage[:, :, 2] * element_damage[:, :, 0]
    )
    mean_damage_squared = (squares + pairs) / 6.0

    triangles = nodes[elements]
    determinant = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    gradients = np.empty((elements.shape[0], 3, 2), dtype=np.float64)
    gradients[:, 0, 0] = (triangles[:, 1, 1] - triangles[:, 2, 1]) / determinant
    gradients[:, 0, 1] = (triangles[:, 2, 0] - triangles[:, 1, 0]) / determinant
    gradients[:, 1, 0] = (triangles[:, 2, 1] - triangles[:, 0, 1]) / determinant
    gradients[:, 1, 1] = (triangles[:, 0, 0] - triangles[:, 2, 0]) / determinant
    gradients[:, 2, 0] = (triangles[:, 0, 1] - triangles[:, 1, 1]) / determinant
    gradients[:, 2, 1] = (triangles[:, 1, 0] - triangles[:, 0, 0]) / determinant
    damage_gradient = np.einsum("tmi,mij->tmj", element_damage, gradients)
    gradient_squared = np.sum(damage_gradient * damage_gradient, axis=2)

    damage_area = np.sum(mean_damage * area[None, :], axis=1)
    crack_density = np.sum(
        area[None, :]
        * (mean_damage_squared / (2.0 * length_scale) + 0.5 * length_scale * gradient_squared),
        axis=1,
    )
    return mean_damage, mean_damage_squared, np.stack([damage_area, crack_density], axis=1)


def _validate_state_arrays(
    record: FractureTrajectory,
    dtype: np.dtype,
    material: Mapping[str, Any],
    dirichlet_dofs: np.ndarray,
) -> None:
    step_count = record.num_steps
    node_count = record.num_nodes
    element_count = record.num_elements
    if step_count <= 0:
        raise FractureSchemaValidationError("load_parameter must contain at least one step")
    load = _require_float_array("load_parameter", record.load_parameter, (step_count,), dtype)
    if np.any(load < 0.0) or np.any(load > 1.0):
        raise FractureSchemaValidationError("load_parameter must lie in [0,1]")
    if float(load[0]) != 0.0:
        raise FractureSchemaValidationError(
            "the first accepted state must define the s=0 path-work reference"
        )
    if step_count > 1 and np.any(np.diff(load) <= 0.0):
        raise FractureSchemaValidationError(
            "load_parameter must be strictly increasing across accepted steps"
        )

    _require_float_array("farfield_stress", record.farfield_stress, (step_count, 3), dtype)
    wall_release = _require_float_array(
        "wall_release_by_facet",
        record.wall_release_by_facet,
        (step_count, record.wall_facets.shape[0]),
        dtype,
    )
    if np.any(wall_release < 0.0) or np.any(wall_release > 1.0):
        raise FractureSchemaValidationError("wall_release_by_facet must lie in [0,1]")
    if step_count > 1 and np.any(np.diff(wall_release.astype(np.float64), axis=0) < 0.0):
        raise FractureSchemaValidationError(
            "wall_release_by_facet must be monotone for every ordered wall facet"
        )

    u = _require_float_array("u", record.u, (step_count, node_count, 2), dtype)
    internal_force = _require_float_array(
        "internal_nodal_force",
        record.internal_nodal_force,
        (step_count, 2 * node_count),
        dtype,
    )
    wall_force = _require_float_array(
        "wall_nodal_force", record.wall_nodal_force, (step_count, 2 * node_count), dtype
    )
    prescribed = _require_float_array(
        "farfield_prescribed_displacement",
        record.farfield_prescribed_displacement,
        (step_count, dirichlet_dofs.size),
        dtype,
    )
    reaction = _require_float_array(
        "farfield_reaction_on_rock",
        record.farfield_reaction_on_rock,
        (step_count, dirichlet_dofs.size),
        dtype,
    )
    rtol, atol = _relative_tolerances(dtype)
    flattened_u = u.reshape(step_count, 2 * node_count)
    if not np.allclose(prescribed, flattened_u[:, dirichlet_dofs], rtol=rtol, atol=atol):
        raise FractureSchemaValidationError(
            "farfield_prescribed_displacement must equal u at farfield_dirichlet_dofs"
        )

    full_residual = internal_force.astype(np.float64) - wall_force.astype(np.float64)
    expected_reaction = full_residual[:, dirichlet_dofs]
    force_scale = max(
        float(np.max(np.abs(internal_force))),
        float(np.max(np.abs(wall_force))),
        float(np.max(np.abs(expected_reaction))),
        1.0,
    )
    if not np.allclose(
        reaction,
        expected_reaction,
        rtol=rtol,
        atol=atol * force_scale,
    ):
        raise FractureSchemaValidationError(
            "farfield_reaction_on_rock must equal (internal_nodal_force - "
            "wall_nodal_force) at farfield_dirichlet_dofs"
        )

    wall_nodes = np.unique(record.wall_facets)
    nonwall_mask = np.ones(node_count, dtype=bool)
    nonwall_mask[wall_nodes] = False
    if np.any(nonwall_mask) and not np.allclose(
        wall_force.reshape(step_count, node_count, 2)[:, nonwall_mask, :],
        0.0,
        rtol=0.0,
        atol=atol * force_scale,
    ):
        raise FractureSchemaValidationError(
            "wall_nodal_force may be nonzero only at nodes in ordered wall_facets"
        )

    free_mask = np.ones(2 * node_count, dtype=bool)
    free_mask[dirichlet_dofs] = False
    free_norm = np.linalg.norm(full_residual[:, free_mask], axis=1)
    free_internal_norm = np.linalg.norm(internal_force[:, free_mask].astype(np.float64), axis=1)
    free_wall_norm = np.linalg.norm(wall_force[:, free_mask].astype(np.float64), axis=1)
    equilibrium_denominator = np.maximum.reduce(
        [
            free_internal_norm,
            free_wall_norm,
            np.full(step_count, float(record.equilibrium_force_normalization_floor)),
        ]
    )
    expected_equilibrium_relative_residual = free_norm / equilibrium_denominator

    damage = _require_float_array("damage", record.damage, (step_count, node_count), dtype)
    if np.any(damage < 0.0) or np.any(damage > 1.0):
        raise FractureSchemaValidationError(
            "damage must satisfy d=0 intact, d=1 broken, and 0<=d<=1"
        )
    actual_irreversibility = np.zeros(step_count, dtype=np.float64)
    if step_count > 1:
        actual_irreversibility[1:] = np.maximum(
            0.0, -np.min(np.diff(damage.astype(np.float64), axis=0), axis=1)
        )
        if np.any(actual_irreversibility > STATE_MONOTONICITY_TOLERANCE):
            raise FractureSchemaValidationError("damage is not irreversible across accepted steps")

    _require_float_array("strain", record.strain, (step_count, element_count, 3), dtype)
    _require_float_array("stress", record.stress, (step_count, element_count, 3), dtype)
    _require_float_array("sigma_xx", record.sigma_xx, (step_count, element_count), dtype)
    psi_plus = _require_float_array("psi_plus", record.psi_plus, (step_count, element_count), dtype)
    psi_minus = _require_float_array(
        "psi_minus", record.psi_minus, (step_count, element_count), dtype
    )
    history = _require_float_array("history", record.history, (step_count, element_count), dtype)
    if np.any(psi_plus < 0.0) or np.any(psi_minus < 0.0) or np.any(history < 0.0):
        raise FractureSchemaValidationError("psi_plus, psi_minus, and history must be non-negative")

    energy_scale = max(float(np.max(history)), float(np.max(psi_plus)), 1.0)
    history_tolerance = STATE_MONOTONICITY_TOLERANCE * energy_scale
    actual_history_violation = np.zeros(step_count, dtype=np.float64)
    actual_history_violation[:] = np.maximum(
        0.0, np.max(psi_plus.astype(np.float64) - history, axis=1)
    )
    if step_count > 1:
        actual_history_violation[1:] = np.maximum(
            actual_history_violation[1:],
            np.maximum(0.0, -np.min(np.diff(history.astype(np.float64), axis=0), axis=1)),
        )
    if np.any(actual_history_violation > history_tolerance):
        raise FractureSchemaValidationError("history must be monotone and no smaller than psi_plus")

    scalar_names = (
        "elastic_energy",
        "fracture_energy",
        "neumann_load_functional",
        "wall_work_increment",
        "farfield_work_increment",
        "cumulative_external_work",
        "total_potential_energy",
        "damage_area",
        "crack_density_integral",
        "damage_connectivity",
        "displacement_residual",
        "damage_residual",
        "equilibrium_relative_residual",
        "kkt_relative_residual",
        "complementarity_relative_residual",
        "damage_irreversibility_violation",
        "damage_range_violation",
        "history_monotonicity_violation",
        "relative_energy_imbalance",
    )
    scalars = {
        name: _require_float_array(name, getattr(record, name), (step_count,), dtype)
        for name in scalar_names
    }
    expected_neumann = np.einsum(
        "ti,ti->t", wall_force.astype(np.float64), flattened_u.astype(np.float64)
    )
    expected_wall_increment = np.zeros(step_count, dtype=np.float64)
    expected_farfield_increment = np.zeros(step_count, dtype=np.float64)
    if step_count > 1:
        expected_wall_increment[1:] = 0.5 * np.einsum(
            "ti,ti->t",
            wall_force[:-1].astype(np.float64) + wall_force[1:].astype(np.float64),
            flattened_u[1:].astype(np.float64) - flattened_u[:-1].astype(np.float64),
        )
        expected_farfield_increment[1:] = 0.5 * np.einsum(
            "ti,ti->t",
            reaction[:-1].astype(np.float64) + reaction[1:].astype(np.float64),
            prescribed[1:].astype(np.float64) - prescribed[:-1].astype(np.float64),
        )
    expected_cumulative_work = np.cumsum(expected_wall_increment + expected_farfield_increment)
    for name, expected in (
        ("neumann_load_functional", expected_neumann),
        ("wall_work_increment", expected_wall_increment),
        ("farfield_work_increment", expected_farfield_increment),
        ("cumulative_external_work", expected_cumulative_work),
    ):
        scale = max(float(np.max(np.abs(expected))), 1.0)
        if not np.allclose(scalars[name], expected, rtol=rtol, atol=atol * scale):
            raise FractureSchemaValidationError(
                f"{name} does not match recomputed accepted-state boundary work"
            )

    total_recoverable_energy = scalars["elastic_energy"].astype(np.float64) + scalars[
        "fracture_energy"
    ].astype(np.float64)
    energy_increment = np.zeros(step_count, dtype=np.float64)
    if step_count > 1:
        energy_increment[1:] = np.diff(total_recoverable_energy)
    external_increment = expected_wall_increment + expected_farfield_increment
    denominator = np.maximum.reduce(
        [
            np.abs(energy_increment),
            np.abs(external_increment),
            np.full(step_count, float(record.energy_balance_normalization_floor)),
        ]
    )
    expected_energy_imbalance = np.abs(energy_increment - external_increment) / denominator
    expected_energy_imbalance[0] = 0.0
    if not np.allclose(
        scalars["relative_energy_imbalance"],
        expected_energy_imbalance,
        rtol=rtol,
        atol=atol,
    ):
        raise FractureSchemaValidationError(
            "relative_energy_imbalance must be recomputed from recoverable-energy and "
            "accepted boundary-work increments"
        )
    if np.any(expected_energy_imbalance > ENERGY_IMBALANCE_TOLERANCE):
        raise FractureSchemaValidationError("relative_energy_imbalance exceeds 5%")

    for name in (
        "elastic_energy",
        "fracture_energy",
        "damage_area",
        "crack_density_integral",
        "displacement_residual",
        "damage_residual",
        "equilibrium_relative_residual",
        "kkt_relative_residual",
        "complementarity_relative_residual",
        "damage_irreversibility_violation",
        "damage_range_violation",
        "history_monotonicity_violation",
        "relative_energy_imbalance",
    ):
        if np.any(scalars[name] < 0.0):
            raise FractureSchemaValidationError(f"{name} must be non-negative")
    if np.any(scalars["damage_connectivity"] < 0.0) or np.any(scalars["damage_connectivity"] > 1.0):
        raise FractureSchemaValidationError("damage_connectivity must lie in [0,1]")

    if not np.allclose(
        scalars["equilibrium_relative_residual"],
        expected_equilibrium_relative_residual,
        rtol=rtol,
        atol=atol,
    ):
        raise FractureSchemaValidationError(
            "equilibrium_relative_residual must equal the recomputed free-DOF force residual"
        )
    if np.any(scalars["equilibrium_relative_residual"] > EQUILIBRIUM_RELATIVE_TOLERANCE):
        raise FractureSchemaValidationError("equilibrium relative residual exceeds 1e-6")
    if np.any(scalars["kkt_relative_residual"] > KKT_RELATIVE_TOLERANCE):
        raise FractureSchemaValidationError("KKT relative residual exceeds 1e-6")
    if np.any(scalars["complementarity_relative_residual"] > KKT_RELATIVE_TOLERANCE):
        raise FractureSchemaValidationError("complementarity relative residual exceeds 1e-6")
    for name in ("damage_irreversibility_violation", "damage_range_violation"):
        if np.any(scalars[name] > STATE_MONOTONICITY_TOLERANCE):
            raise FractureSchemaValidationError(f"{name} exceeds 1e-10")
    if np.any(scalars["history_monotonicity_violation"] > history_tolerance):
        raise FractureSchemaValidationError("history_monotonicity_violation exceeds tolerance")
    if np.any(
        scalars["damage_irreversibility_violation"] + STATE_MONOTONICITY_TOLERANCE
        < actual_irreversibility
    ):
        raise FractureSchemaValidationError(
            "damage_irreversibility_violation understates the stored state"
        )
    if np.any(
        scalars["history_monotonicity_violation"] + history_tolerance < actual_history_violation
    ):
        raise FractureSchemaValidationError(
            "history_monotonicity_violation understates the stored state"
        )

    rtol, atol = _relative_tolerances(dtype)
    mean_damage, mean_damage_squared, integrals = _p1_damage_integrals(
        record.nodes,
        record.elements,
        record.area,
        damage,
        float(material["length_scale"]),
    )
    expected_damage_area = integrals[:, 0]
    total_area = float(np.sum(record.area))
    if not np.allclose(
        scalars["damage_area"],
        expected_damage_area,
        rtol=rtol,
        atol=atol * max(total_area, 1.0),
    ):
        raise FractureSchemaValidationError("damage_area does not equal the P1 integral of damage")
    expected_crack_density = integrals[:, 1]
    if not np.allclose(
        scalars["crack_density_integral"],
        expected_crack_density,
        rtol=rtol,
        atol=atol * max(float(np.max(expected_crack_density)), 1.0),
    ):
        raise FractureSchemaValidationError(
            "crack_density_integral does not equal the AT2 P1 integral"
        )
    expected_fracture_energy = float(material["fracture_energy"]) * expected_crack_density
    if not np.allclose(
        scalars["fracture_energy"],
        expected_fracture_energy,
        rtol=rtol,
        atol=atol * max(float(np.max(expected_fracture_energy)), 1.0),
    ):
        raise FractureSchemaValidationError(
            "fracture_energy must equal Gc times crack_density_integral"
        )

    degradation_mean = (
        1.0 - 2.0 * mean_damage + mean_damage_squared + float(material["residual_stiffness"])
    )
    expected_elastic_energy = np.sum(
        record.area[None, :] * (degradation_mean * psi_plus.astype(np.float64) + psi_minus),
        axis=1,
    )
    if not np.allclose(
        scalars["elastic_energy"],
        expected_elastic_energy,
        rtol=rtol,
        atol=atol * max(float(np.max(expected_elastic_energy)), 1.0),
    ):
        raise FractureSchemaValidationError(
            "elastic_energy does not match the explicit P1 AT2 element integral"
        )
    expected_potential = (
        scalars["elastic_energy"] + scalars["fracture_energy"] - scalars["neumann_load_functional"]
    )
    if not np.allclose(
        scalars["total_potential_energy"],
        expected_potential,
        rtol=rtol,
        atol=atol * max(float(np.max(np.abs(expected_potential))), 1.0),
    ):
        raise FractureSchemaValidationError(
            "total_potential_energy must equal elastic + fracture - instantaneous "
            "neumann_load_functional"
        )

    paired = (
        record.elastic_basis_stress is not None,
        record.nonlinear_stress_residual is not None,
    )
    if paired[0] != paired[1]:
        raise FractureSchemaValidationError(
            "elastic_basis_stress and nonlinear_stress_residual must be supplied together"
        )
    if paired[0]:
        if record.elastic_basis_id is None or record.elastic_basis_sha256 is None:
            raise FractureSchemaValidationError(
                "basis arrays require elastic_basis_id and elastic_basis_sha256"
            )
        basis = _require_float_array(
            "elastic_basis_stress",
            record.elastic_basis_stress,
            (step_count, element_count, 3),
            dtype,
        )
        residual = _require_float_array(
            "nonlinear_stress_residual",
            record.nonlinear_stress_residual,
            (step_count, element_count, 3),
            dtype,
        )
        stress_scale = max(float(np.max(np.abs(record.stress))), 1.0)
        if not np.allclose(
            record.stress,
            basis + residual,
            rtol=rtol,
            atol=atol * stress_scale,
        ):
            raise FractureSchemaValidationError(
                "stress must equal elastic_basis_stress + nonlinear_stress_residual"
            )
    elif record.elastic_basis_id is not None:
        raise FractureSchemaValidationError(
            "an elastic-basis link requires basis and nonlinear residual arrays"
        )


def _ledger_integer(entry: Mapping[str, Any], name: str, index: int) -> int:
    value = entry[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FractureSchemaValidationError(
            f"attempt_ledger[{index}].{name} must be a non-negative integer"
        )
    return value


def _ledger_float(entry: Mapping[str, Any], name: str, index: int) -> float:
    value = entry[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FractureSchemaValidationError(f"attempt_ledger[{index}].{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FractureSchemaValidationError(f"attempt_ledger[{index}].{name} must be finite")
    return number


def _validate_attempt_ledger(record: FractureTrajectory) -> None:
    ledger = _normalise_json(record.attempt_ledger, path="$.attempt_ledger")
    if not isinstance(ledger, list) or not ledger:
        raise FractureSchemaValidationError("attempt_ledger must not be empty")
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {
        step: [] for step in range(record.num_steps)
    }
    observed_order: list[tuple[int, int]] = []
    for position, entry in enumerate(ledger):
        if not isinstance(entry, dict) or set(entry) != ATTEMPT_LEDGER_KEYS:
            missing = sorted(ATTEMPT_LEDGER_KEYS - set(entry if isinstance(entry, dict) else {}))
            extra = sorted(set(entry if isinstance(entry, dict) else {}) - ATTEMPT_LEDGER_KEYS)
            raise FractureSchemaValidationError(
                f"attempt_ledger[{position}] key mismatch; missing={missing}, extra={extra}"
            )
        step = _ledger_integer(entry, "step_index", position)
        attempt = _ledger_integer(entry, "attempt_index", position)
        if step >= record.num_steps:
            raise FractureSchemaValidationError(
                f"attempt_ledger[{position}].step_index is out of range"
            )
        start = _ledger_float(entry, "load_parameter_start", position)
        target = _ledger_float(entry, "load_parameter_target", position)
        if not 0.0 <= start <= target <= 1.0:
            raise FractureSchemaValidationError(
                f"attempt_ledger[{position}] has invalid load interval"
            )
        expected_start = 0.0 if step == 0 else float(record.load_parameter[step - 1])
        if not math.isclose(start, expected_start, rel_tol=0.0, abs_tol=1.0e-14):
            raise FractureSchemaValidationError(
                f"attempt_ledger[{position}] does not start at the previous accepted load"
            )
        if not isinstance(entry["accepted"], bool):
            raise FractureSchemaValidationError(
                f"attempt_ledger[{position}].accepted must be boolean"
            )
        for name in (
            "newton_iterations",
            "active_set_iterations",
            "staggered_iterations",
            "step_halvings",
        ):
            _ledger_integer(entry, name, position)
        for name in (
            "equilibrium_relative_residual",
            "kkt_relative_residual",
            "complementarity_relative_residual",
            "damage_irreversibility_violation",
            "damage_range_violation",
            "relative_energy_imbalance",
        ):
            if _ledger_float(entry, name, position) < 0.0:
                raise FractureSchemaValidationError(
                    f"attempt_ledger[{position}].{name} must be non-negative"
                )
        code = entry["failure_code"]
        message = entry["failure_message"]
        if entry["accepted"]:
            if code is not None or message is not None:
                raise FractureSchemaValidationError(
                    f"accepted attempt_ledger[{position}] cannot carry a failure"
                )
        elif (
            not isinstance(code, str)
            or not code.strip()
            or not isinstance(message, str)
            or not message.strip()
        ):
            raise FractureSchemaValidationError(
                f"rejected attempt_ledger[{position}] needs failure_code and failure_message"
            )

        work_names = (
            "neumann_load_functional",
            "wall_work_increment",
            "farfield_work_increment",
            "cumulative_external_work",
        )
        if entry["accepted"]:
            state_hash = _require_sha256(
                entry["load_state_sha256"],
                f"attempt_ledger[{position}].load_state_sha256",
            )
            expected_state_hash = compute_load_state_sha256(
                record.load_parameter[step : step + 1],
                record.farfield_stress[step],
                record.wall_release_by_facet[step],
                record.wall_facets,
            )
            if state_hash != expected_state_hash:
                raise FractureSchemaValidationError(
                    f"attempt_ledger[{position}].load_state_sha256 does not bind the "
                    "accepted load parameter, stress, and ordered wall-facet release"
                )
            rtol, atol = _relative_tolerances(record.dtype)
            for name in work_names:
                value = _ledger_float(entry, name, position)
                expected_value = float(getattr(record, name)[step])
                if not math.isclose(value, expected_value, rel_tol=rtol, abs_tol=atol):
                    raise FractureSchemaValidationError(
                        f"accepted attempt {name} for step {step} does not match the step array"
                    )
        elif entry["load_state_sha256"] is not None or any(
            entry[name] is not None for name in work_names
        ):
            raise FractureSchemaValidationError(
                f"rejected attempt_ledger[{position}] must not publish a load-state hash "
                "or accepted-state work"
            )
        grouped[step].append((attempt, entry))
        observed_order.append((step, attempt))

    if observed_order != sorted(observed_order):
        raise FractureSchemaValidationError(
            "attempt_ledger must be ordered by step_index then attempt_index"
        )

    for step, attempts in grouped.items():
        if not attempts:
            raise FractureSchemaValidationError(f"attempt_ledger has no entry for step {step}")
        attempts.sort(key=lambda item: item[0])
        if [item[0] for item in attempts] != list(range(len(attempts))):
            raise FractureSchemaValidationError(
                f"attempt indices for step {step} must be contiguous from zero"
            )
        entries = [item[1] for item in attempts]
        if any(entry["accepted"] for entry in entries[:-1]) or not entries[-1]["accepted"]:
            raise FractureSchemaValidationError(
                f"step {step} must end with exactly one accepted attempt"
            )
        accepted_target = float(entries[-1]["load_parameter_target"])
        if not math.isclose(
            accepted_target,
            float(record.load_parameter[step]),
            rel_tol=0.0,
            abs_tol=1.0e-14,
        ):
            raise FractureSchemaValidationError(
                f"accepted attempt for step {step} does not match load_parameter"
            )
        accepted_diagnostics = entries[-1]
        rtol, atol = _relative_tolerances(record.dtype)
        for name in (
            "equilibrium_relative_residual",
            "kkt_relative_residual",
            "complementarity_relative_residual",
            "damage_irreversibility_violation",
            "damage_range_violation",
            "relative_energy_imbalance",
        ):
            if not math.isclose(
                float(accepted_diagnostics[name]),
                float(getattr(record, name)[step]),
                rel_tol=rtol,
                abs_tol=atol,
            ):
                raise FractureSchemaValidationError(
                    f"accepted attempt {name} for step {step} does not match the step array"
                )
        expected_retry = len(entries) - 1
        if int(record.retry_count[step]) != expected_retry:
            raise FractureSchemaValidationError(
                f"retry_count[{step}] does not match rejected attempts"
            )
        for array_name, ledger_name in (
            ("newton_iterations", "newton_iterations"),
            ("active_set_iterations", "active_set_iterations"),
            ("staggered_iterations", "staggered_iterations"),
        ):
            total = sum(int(entry[ledger_name]) for entry in entries)
            if int(getattr(record, array_name)[step]) != total:
                raise FractureSchemaValidationError(
                    f"{array_name}[{step}] does not equal the complete attempt-ledger total"
                )
        halvings = [int(entry["step_halvings"]) for entry in entries]
        if any(later < earlier for earlier, later in pairwise(halvings)):
            raise FractureSchemaValidationError(
                f"step-halving counts for step {step} must be non-decreasing"
            )
        if int(record.step_halvings[step]) != halvings[-1]:
            raise FractureSchemaValidationError(
                f"step_halvings[{step}] does not match the accepted attempt"
            )


def validate_fracture_trajectory(
    trajectory: FractureTrajectory, *, expected_dtype: Any | None = FLOAT64
) -> None:
    """Validate a complete immutable C-fracture trajectory without mutation."""

    if not isinstance(trajectory, FractureTrajectory):
        raise TypeError("trajectory must be a FractureTrajectory")
    dtype = trajectory.dtype
    if dtype not in SUPPORTED_PUBLICATION_DTYPES:
        raise FractureSchemaValidationError(
            f"floating arrays must use float64 or explicitly published float32; got {dtype}"
        )
    if expected_dtype is not None:
        required_dtype = np.dtype(expected_dtype)
        if required_dtype not in SUPPORTED_PUBLICATION_DTYPES:
            raise ValueError("expected_dtype must be float64, float32, or None")
        if dtype != required_dtype:
            raise FractureSchemaValidationError(
                f"expected {required_dtype.name} publication; got {dtype.name}"
            )

    _validate_identity(trajectory)
    _validate_conventions(trajectory)
    material, _ = _validate_metadata(trajectory)
    _validate_mesh(trajectory, dtype)
    dirichlet_dofs = _validate_discrete_identity(trajectory)
    _validate_state_arrays(trajectory, dtype, material, dirichlet_dofs)

    for name in (
        "newton_iterations",
        "active_set_iterations",
        "staggered_iterations",
        "step_halvings",
        "retry_count",
    ):
        value = _require_index_array(name, getattr(trajectory, name), (trajectory.num_steps,))
        if np.any(value < 0):
            raise FractureSchemaValidationError(f"{name} must be non-negative")
    _validate_attempt_ledger(trajectory)


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


def compute_load_state_sha256(
    load_parameter: np.ndarray,
    farfield_stress: np.ndarray,
    wall_release_by_facet: np.ndarray,
    ordered_wall_facets: np.ndarray,
) -> str:
    """Bind one accepted coordinate to its stress and ordered wall-facet release."""

    return _semantic_array_sha256(
        {
            "load_parameter": np.asarray(load_parameter),
            "farfield_stress": np.asarray(farfield_stress),
            "wall_release_by_facet": np.asarray(wall_release_by_facet),
            "ordered_wall_facets": np.asarray(ordered_wall_facets),
        }
    )


def compute_mesh_content_sha256(
    nodes: np.ndarray,
    elements: np.ndarray,
    wall_facets: np.ndarray,
    farfield_facets: np.ndarray,
) -> str:
    """Hash mesh topology and coordinates with a deterministic encoding."""

    return _semantic_array_sha256(
        {
            "nodes": np.asarray(nodes),
            "elements": np.asarray(elements),
            "wall_facets": _normalise_facets(wall_facets),
            "farfield_facets": _normalise_facets(farfield_facets),
        }
    )


def compute_identity_content_sha256(
    nodes: np.ndarray,
    node_ids: np.ndarray,
    displacement_dof_ids: np.ndarray,
    damage_dof_ids: np.ndarray,
    elements: np.ndarray,
    wall_facets: np.ndarray,
    farfield_facets: np.ndarray,
    farfield_dirichlet_dofs: np.ndarray,
) -> str:
    """Bind geometry/topology to exact row-ordered node, boundary, and DOF identities."""

    return _semantic_array_sha256(
        {
            "nodes": np.asarray(nodes),
            "node_ids": np.asarray(node_ids),
            "displacement_dof_ids": np.asarray(displacement_dof_ids),
            "damage_dof_ids": np.asarray(damage_dof_ids),
            "elements": np.asarray(elements),
            "wall_facets": np.asarray(wall_facets),
            "farfield_facets": np.asarray(farfield_facets),
            "farfield_dirichlet_dofs": np.asarray(farfield_dirichlet_dofs),
        }
    )


def _array_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {
            "dtype": np.ascontiguousarray(value).dtype.str,
            "shape": list(value.shape),
            "sha256": _semantic_array_sha256({name: value}),
        }
        for name, value in sorted(arrays.items())
    }


def _record_meta(
    trajectory: FractureTrajectory, arrays: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "dtype": trajectory.dtype.name,
        "trajectory_id": trajectory.trajectory_id,
        "case_id": trajectory.case_id,
        "mesh_id": trajectory.mesh_id,
        "geometry_id": trajectory.geometry_id,
        "material_id": trajectory.material_id,
        "load_path_id": trajectory.load_path_id,
        "config_hash": trajectory.config_hash,
        "solver_hash": trajectory.solver_hash,
        "elastic_basis_id": trajectory.elastic_basis_id,
        "elastic_basis_sha256": trajectory.elastic_basis_sha256,
        "completed": trajectory.completed,
        "accepted": trajectory.accepted,
        "mesh_content_sha256": compute_mesh_content_sha256(
            trajectory.nodes,
            trajectory.elements,
            trajectory.wall_facets,
            trajectory.farfield_facets,
        ),
        "identity_content_sha256": compute_identity_content_sha256(
            trajectory.nodes,
            trajectory.node_ids,
            trajectory.displacement_dof_ids,
            trajectory.damage_dof_ids,
            trajectory.elements,
            trajectory.wall_facets,
            trajectory.farfield_facets,
            trajectory.farfield_dirichlet_dofs,
        ),
        "equilibrium_force_normalization_floor": float(
            trajectory.equilibrium_force_normalization_floor
        ),
        "energy_balance_normalization_floor": float(trajectory.energy_balance_normalization_floor),
        "coordinate_order": list(trajectory.coordinate_order),
        "strain_component_order": list(trajectory.strain_component_order),
        "stress_component_order": list(trajectory.stress_component_order),
        "sign_convention": trajectory.sign_convention,
        "damage_convention": trajectory.damage_convention,
        "formulation": trajectory.formulation,
        "units": _normalise_json(trajectory.units, path="$.units"),
        "material": _normalise_json(trajectory.material, path="$.material"),
        "geometry": _normalise_json(trajectory.geometry, path="$.geometry"),
        "load_path": _normalise_json(trajectory.load_path, path="$.load_path"),
        "physical_tags": _normalise_json(trajectory.physical_tags, path="$.physical_tags"),
        "mesh_metadata": _normalise_json(trajectory.mesh_metadata, path="$.mesh_metadata"),
        "solver": _normalise_json(trajectory.solver, path="$.solver"),
        "env": _normalise_json(trajectory.env, path="$.env"),
        "meta": _normalise_json(trajectory.meta, path="$.meta"),
        "attempt_ledger": _normalise_json(trajectory.attempt_ledger, path="$.attempt_ledger"),
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
def _writer_lock(trajectory_dir: Path) -> Iterator[None]:
    lock_path = trajectory_dir / ".fracture-schema.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(f"another fracture-trajectory writer holds {lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


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


def save_fracture_trajectory(
    trajectory_dir: str | os.PathLike[str],
    trajectory: FractureTrajectory,
    *,
    overwrite: bool = False,
    expected_dtype: Any | None = FLOAT64,
) -> FractureTrajectoryPaths:
    """Validate and atomically publish one protected fracture trajectory.

    Metadata is replaced after the NPZ.  Therefore an interrupted overwrite is
    detected as a hash mismatch rather than admitted as a mixed record.
    """

    trajectory.validate(expected_dtype=expected_dtype)
    paths = fracture_trajectory_paths(trajectory_dir)
    paths.trajectory_dir.mkdir(parents=True, exist_ok=True)
    arrays = trajectory.arrays()
    npz_temp: Path | None = None
    json_temp: Path | None = None
    with _writer_lock(paths.trajectory_dir):
        existing = [path for path in (paths.arrays, paths.meta) if path.exists()]
        if existing and not overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"fracture trajectory already has protected file(s): {names}; "
                "pass overwrite=True to replace both files"
            )
        try:
            npz_temp = _write_npz_temp(paths.trajectory_dir, arrays)
            metadata = _record_meta(trajectory, arrays)
            metadata["arrays_file_sha256"] = _hash_file(npz_temp)
            metadata["content_sha256"] = _content_sha256(metadata)
            json_temp = _write_json_temp(paths.trajectory_dir, metadata)
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required fracture metadata: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FractureSchemaValidationError(f"could not load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FractureSchemaValidationError("meta.json root must be an object")
    if set(value) != _META_KEYS:
        missing = sorted(_META_KEYS - set(value))
        extra = sorted(set(value) - _META_KEYS)
        raise FractureSchemaValidationError(
            f"meta.json key mismatch; missing={missing}, extra={extra}"
        )
    return value


def _load_arrays(path: Path, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing required fracture arrays: {path}")
    manifest = metadata.get("array_manifest")
    if not isinstance(manifest, Mapping):
        raise FractureSchemaValidationError("array_manifest must be an object")
    expected_keys = set(manifest)
    if not set(ARRAY_KEYS).issubset(expected_keys):
        missing = sorted(set(ARRAY_KEYS) - expected_keys)
        raise FractureSchemaValidationError(f"array_manifest is missing required arrays: {missing}")
    extras = expected_keys - set(ARRAY_KEYS)
    if extras not in (set(), set(OPTIONAL_ARRAY_KEYS)):
        raise FractureSchemaValidationError(
            f"array_manifest has unsupported optional arrays: {sorted(extras)}"
        )
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                missing = sorted(expected_keys - set(archive.files))
                extra = sorted(set(archive.files) - expected_keys)
                raise FractureSchemaValidationError(
                    f"arrays.npz key mismatch; missing={missing}, extra={extra}"
                )
            return {name: np.asarray(archive[name]) for name in sorted(expected_keys)}
    except FractureSchemaValidationError:
        raise
    except (OSError, ValueError, EOFError) as exc:
        raise FractureSchemaValidationError(f"could not load {path}: {exc}") from exc


def _verify_manifest(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
    expected = _array_manifest(arrays)
    if _normalise_json(metadata["array_manifest"], path="$.array_manifest") != expected:
        raise FractureSchemaValidationError("array_manifest does not match arrays.npz")
    mesh_hash = compute_mesh_content_sha256(
        arrays["nodes"],
        arrays["elements"],
        arrays["wall_facets"],
        arrays["farfield_facets"],
    )
    if metadata["mesh_content_sha256"] != mesh_hash:
        raise FractureSchemaValidationError("mesh_content_sha256 does not match the mesh arrays")
    identity_hash = compute_identity_content_sha256(
        arrays["nodes"],
        arrays["node_ids"],
        arrays["displacement_dof_ids"],
        arrays["damage_dof_ids"],
        arrays["elements"],
        arrays["wall_facets"],
        arrays["farfield_facets"],
        arrays["farfield_dirichlet_dofs"],
    )
    if metadata["identity_content_sha256"] != identity_hash:
        raise FractureSchemaValidationError(
            "identity_content_sha256 does not match geometry/topology/node/DOF arrays"
        )


def load_fracture_trajectory(
    trajectory_dir: str | os.PathLike[str], *, expected_dtype: Any | None = FLOAT64
) -> FractureTrajectory:
    """Verify both hashes, load, freeze, and semantically revalidate a trajectory."""

    paths = fracture_trajectory_paths(trajectory_dir)
    metadata = _load_json(paths.meta)
    if metadata.get("schema") != SCHEMA_NAME or metadata.get("schema_version") != SCHEMA_VERSION:
        raise FractureSchemaValidationError("unsupported C-fracture schema name or version")
    _require_sha256(metadata.get("arrays_file_sha256"), "arrays_file_sha256")
    _require_sha256(metadata.get("content_sha256"), "content_sha256")
    if _hash_file(paths.arrays) != metadata["arrays_file_sha256"]:
        raise FractureSchemaValidationError("arrays.npz SHA-256 does not match meta.json")
    if _content_sha256(metadata) != metadata["content_sha256"]:
        raise FractureSchemaValidationError("trajectory content_sha256 does not match meta.json")
    arrays = _load_arrays(paths.arrays, metadata)
    _verify_manifest(arrays, metadata)

    optional = {name: arrays.pop(name, None) for name in OPTIONAL_ARRAY_KEYS}
    try:
        trajectory = FractureTrajectory(
            **arrays,
            **optional,
            attempt_ledger=metadata["attempt_ledger"],
            trajectory_id=metadata["trajectory_id"],
            case_id=metadata["case_id"],
            mesh_id=metadata["mesh_id"],
            geometry_id=metadata["geometry_id"],
            material_id=metadata["material_id"],
            load_path_id=metadata["load_path_id"],
            config_hash=metadata["config_hash"],
            solver_hash=metadata["solver_hash"],
            equilibrium_force_normalization_floor=metadata["equilibrium_force_normalization_floor"],
            energy_balance_normalization_floor=metadata["energy_balance_normalization_floor"],
            material=metadata["material"],
            geometry=metadata["geometry"],
            load_path=metadata["load_path"],
            physical_tags=metadata["physical_tags"],
            mesh_metadata=metadata["mesh_metadata"],
            solver=metadata["solver"],
            env=metadata["env"],
            meta=metadata["meta"],
            elastic_basis_id=metadata["elastic_basis_id"],
            elastic_basis_sha256=metadata["elastic_basis_sha256"],
            completed=metadata["completed"],
            accepted=metadata["accepted"],
            coordinate_order=tuple(metadata["coordinate_order"]),
            strain_component_order=tuple(metadata["strain_component_order"]),
            stress_component_order=tuple(metadata["stress_component_order"]),
            sign_convention=metadata["sign_convention"],
            damage_convention=metadata["damage_convention"],
            formulation=metadata["formulation"],
            units=metadata["units"],
        )
    except (TypeError, KeyError) as exc:
        raise FractureSchemaValidationError(
            f"metadata cannot construct FractureTrajectory: {exc}"
        ) from exc
    if metadata["dtype"] != trajectory.dtype.name:
        raise FractureSchemaValidationError("meta.json dtype does not match floating arrays")
    trajectory.validate(expected_dtype=expected_dtype)
    return trajectory


__all__ = [
    "ARRAYS_FILENAME",
    "ARRAY_KEYS",
    "ATTEMPT_LEDGER_KEYS",
    "COORDINATE_ORDER",
    "DAMAGE_CONVENTION",
    "ENERGY_IMBALANCE_TOLERANCE",
    "EQUILIBRIUM_RELATIVE_TOLERANCE",
    "FLOAT32",
    "FLOAT64",
    "FORMULATION",
    "KKT_RELATIVE_TOLERANCE",
    "META_FILENAME",
    "OPTIONAL_ARRAY_KEYS",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "SIGN_CONVENTION",
    "SI_UNITS",
    "STATE_MONOTONICITY_TOLERANCE",
    "STRAIN_COMPONENT_ORDER",
    "STRESS_COMPONENT_ORDER",
    "SUPPORTED_PUBLICATION_DTYPES",
    "FractureSchemaValidationError",
    "FractureTrajectory",
    "FractureTrajectoryPaths",
    "compute_identity_content_sha256",
    "compute_load_state_sha256",
    "compute_mesh_content_sha256",
    "fracture_trajectory_paths",
    "load_fracture_trajectory",
    "save_fracture_trajectory",
    "validate_fracture_trajectory",
]
