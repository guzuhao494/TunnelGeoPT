"""Confirm fixed-geometry load-axis factorization with new direct FEM solves.

``validate-plan`` is deliberately solver-free and creates no artifact.  The
``run`` command is separately guarded by a clean, pushed Git checkout, an
explicit expected commit, frozen implementation hashes, and an absent output
directory.  A successful run uses three basis and five held-out direct solves
on each of three fixed geometry/mesh/query triples (24 solves total).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tunnelgeopt.elasticity import solve_plane_strain_excavation
from tunnelgeopt.field_sampling import locate_elements, sample_piecewise_constant
from tunnelgeopt.geometry import points_inside_polygon
from tunnelgeopt.load_basis import fit_linear_stress_response_basis
from tunnelgeopt.mesh import generate_tunnel_mesh
from tunnelgeopt.multifidelity import (
    GeometryDataSpec,
    MeshFidelitySpec,
    build_elastic_query_grid,
    case_group_id,
    load_group_id,
)

DEFAULT_CONFIG = ROOT / "configs" / "load_basis_confirmation.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "confirmation" / "linear-load-basis-v0.5.0"
CONFIRMED_CLASSIFICATION = "LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED"
STOP_CLASSIFICATION = "STOP_BASIS_CONFIRMATION"
INVALID_CLASSIFICATION = "ABSTAIN_INVALID"
EXECUTION_ACKNOWLEDGEMENT = "RUN_24_NEW_DIRECT_FEM_SOLVES"
HEX64 = frozenset("0123456789abcdef")


class ConfirmationError(RuntimeError):
    """Raised when the frozen plan, execution preflight, or result fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ConfirmationError(f"non-finite persisted float: {name}")
    return number


def _finite_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ConfirmationError(f"non-finite response array: {name}")
    return array


def _json_safe(value: Any, path: str = "$") -> tuple[Any, list[dict[str, str]]]:
    """Replace non-finite floats with null and return explicit validity issues."""

    issues: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        safe: dict[str, Any] = {}
        for key, child in value.items():
            child_safe, child_issues = _json_safe(child, f"{path}.{key}")
            safe[str(key)] = child_safe
            issues.extend(child_issues)
        return safe, issues
    if isinstance(value, (list, tuple)):
        safe_list: list[Any] = []
        for index, child in enumerate(value):
            child_safe, child_issues = _json_safe(child, f"{path}[{index}]")
            safe_list.append(child_safe)
            issues.extend(child_issues)
        return safe_list, issues
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if math.isfinite(number):
            return number, issues
        observed = (
            "nan"
            if math.isnan(number)
            else ("positive_infinity" if number > 0 else "negative_infinity")
        )
        issues.append(
            {
                "path": path,
                "reason": "non_finite_float_replaced_with_null",
                "observed": observed,
            }
        )
        return None, issues
    if isinstance(value, np.integer):
        return int(value), issues
    if isinstance(value, np.bool_):
        return bool(value), issues
    if value is None or isinstance(value, (str, bool, int)):
        return value, issues
    raise ConfirmationError(f"unsupported artifact value at {path}: {type(value).__name__}")


def _array_sha256(array: Any, dtype: str) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in HEX64 for character in text):
        raise ConfirmationError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _require_git_commit(value: Any, name: str) -> str:
    text = str(value)
    if len(text) not in (40, 64) or any(character not in HEX64 for character in text):
        raise ConfirmationError(f"{name} must be a lowercase 40- or 64-hex Git commit")
    return text


def _relative_repository_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ConfirmationError("all confirmation paths must remain inside the repository") from exc


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfirmationError(f"could not read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfirmationError(f"{name} must be a JSON object")
    return value


def _resolve_source(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    _relative_repository_path(path)
    if not path.is_file():
        raise ConfirmationError(f"identity source is missing: {relative}")
    return path


def _load_hashed_json(source: Mapping[str, Any], name: str) -> dict[str, Any]:
    path = _resolve_source(str(source["path"]))
    expected = _require_sha256(source["sha256"], f"{name}.sha256")
    if _file_sha256(path) != expected:
        raise ConfirmationError(f"identity source hash changed: {source['path']}")
    return _read_json(path, name)


def _collect_named_hashes(value: Any, names: frozenset[str]) -> set[str]:
    result: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key) in names:
                    if isinstance(child, str) and len(child) == 64:
                        result.add(child)
                    elif isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
                        result.update(str(entry) for entry in child if isinstance(entry, str))
                visit(child)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            for child in item:
                visit(child)

    visit(value)
    return result


def _load_excluded_identities(
    config: Mapping[str, Any],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    sources = config["identity_exclusions"]
    legacy = _load_hashed_json(sources["legacy_aggregate"], "legacy identity aggregate")
    required = (
        "geometry_group_ids",
        "boundary_float64_sha256",
        "case_group_ids",
        "load_group_ids",
    )
    if not all(isinstance(legacy.get(key), list) for key in required):
        raise ConfirmationError("legacy aggregate does not contain the four identity lists")
    excluded = {
        "geometry_group_id": set(map(str, legacy["geometry_group_ids"])),
        "boundary_float64_sha256": set(map(str, legacy["boundary_float64_sha256"])),
        "case_group_id": set(map(str, legacy["case_group_ids"])),
        "load_group_id": set(map(str, legacy["load_group_ids"])),
        "query_hash": set(),
    }

    query_source_audit: list[dict[str, Any]] = []
    for index, source in enumerate(sources["legacy_query_sources"]):
        payload = _load_hashed_json(source, f"legacy query source {index}")
        hashes = _collect_named_hashes(payload, frozenset({"query_hash", "common_query_hash"}))
        excluded["query_hash"].update(hashes)
        query_source_audit.append(
            {
                "path": str(source["path"]),
                "sha256": str(source["sha256"]),
                "query_identity_count": len(hashes),
            }
        )

    formal_manifest_source = sources["v03_formal_manifest"]
    formal_manifest = _load_hashed_json(formal_manifest_source, "v0.3 formal manifest")
    public_source = sources["v03_public_identity_store"]
    public_path = _resolve_source(str(public_source["path"]))
    public_hash = _require_sha256(public_source["sha256"], "v03_public_identity_store.sha256")
    if _file_sha256(public_path) != public_hash:
        raise ConfirmationError("v0.3 public identity store hash changed")
    if formal_manifest.get("files", {}).get(public_path.name) != public_hash:
        raise ConfirmationError("v0.3 manifest does not authenticate the public identity store")
    with np.load(public_path, allow_pickle=False) as archive:
        array_map = {
            "geometry_group_id": "geometry_group_ids",
            "boundary_float64_sha256": "boundary_float64_sha256",
            "case_group_id": "case_group_ids",
            "load_group_id": "load_group_ids",
            "query_hash": "query_hashes",
        }
        if not set(array_map.values()).issubset(archive.files):
            raise ConfirmationError("v0.3 public identity store is missing identity arrays")
        formal_counts: dict[str, int] = {}
        for key, array_name in array_map.items():
            values = set(np.asarray(archive[array_name]).astype(str).tolist())
            excluded[key].update(values)
            formal_counts[key] = len(values)

    audit = {
        "legacy_aggregate": {
            "path": str(sources["legacy_aggregate"]["path"]),
            "sha256": str(sources["legacy_aggregate"]["sha256"]),
        },
        "legacy_query_sources": query_source_audit,
        "v03_formal_manifest": {
            "path": str(formal_manifest_source["path"]),
            "sha256": str(formal_manifest_source["sha256"]),
        },
        "v03_public_identity_store": {
            "path": str(public_source["path"]),
            "sha256": public_hash,
            "unique_identity_counts": formal_counts,
        },
        "union_unique_identity_counts": {
            key: len(values) for key, values in sorted(excluded.items())
        },
    }
    return excluded, audit


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _read_json(path, "load-basis confirmation config")
    if config.get("schema") != "tunnelgeopt.load_basis_confirmation.config.v1":
        raise ConfirmationError("unexpected load-basis confirmation schema")
    if config.get("status") != "frozen_plan_not_executed":
        raise ConfirmationError("confirmation config must remain frozen_plan_not_executed")
    if config.get("classification_if_all_gates_pass") != CONFIRMED_CLASSIFICATION:
        raise ConfirmationError("the only permitted positive classification changed")
    if config.get("post_solve_classifications") != {
        "invalid_protocol_or_qc": INVALID_CLASSIFICATION,
        "valid_protocol_numerical_gate_failed": STOP_CLASSIFICATION,
        "all_gates_passed": CONFIRMED_CLASSIFICATION,
    }:
        raise ConfirmationError("post-solve classification contract changed")
    if config.get("solve_contract") != {
        "geometry_count": 3,
        "basis_loads_per_geometry": 3,
        "heldout_loads_per_geometry": 5,
        "total_direct_fem_solves": 24,
    }:
        raise ConfirmationError("confirmation must remain a 3 x (3 + 5) = 24 solve design")
    if config.get("execution_evidence_contract") != {
        "reserve_output_directory_atomically_before_first_mesh_or_solve": True,
        "record_each_solve_before_solver_call": True,
        "persist_planned_attempted_solver_returned_validated_completed_failed_and_not_attempted_counts": True,
        "post_reservation_mesh_solver_or_qc_failure_writes_abstain_invalid": True,
        "nonfinite_persisted_float_becomes_null_with_explicit_reason_and_abstain_invalid": True,
        "artifact_write": ("exclusive_tmp_file_then_atomic_replace_in_reserved_output_directory"),
    }:
        raise ConfirmationError("execution evidence contract changed")
    if config.get("response_contract") != {
        "primary": "query_total_in_plane_stress",
        "required_auxiliary": [
            "nodal_displacement",
            "element_delta_stress",
            "query_delta_stress",
        ],
        "secondary_report_only": "element_sigma_xx",
        "primary_and_stress_component_order": ["yy", "zz", "yz"],
    }:
        raise ConfirmationError("linear-response output contract changed")

    claim = config.get("claim_scope", {})
    expected_scope = {
        "confirmed": [
            "fixed_geometry",
            "fixed_material",
            "fixed_mesh",
            "fixed_query",
            "two_dimensional_small_strain_linear_elasticity",
            "linear_factorization_of_in_plane_farfield_load_axis",
        ],
        "excluded": [
            "geometry_generalization",
            "mesh_generalization",
            "material_generalization",
            "fracture",
            "damage",
            "plasticity",
            "rockburst",
            "micro_to_field_transfer",
            "field_prediction",
            "engineering_truth",
        ],
    }
    if claim != expected_scope:
        raise ConfirmationError("claim scope must remain the narrow fixed-system elastic scope")

    basis = np.asarray(config.get("basis_loads_tension_positive"), dtype=np.float64)
    heldout = np.asarray(config.get("heldout_loads_tension_positive"), dtype=np.float64)
    if basis.shape != (3, 3) or heldout.shape != (5, 3):
        raise ConfirmationError("basis and held-out load arrays must have shapes [3,3] and [5,3]")
    if not np.isfinite(basis).all() or not np.isfinite(heldout).all():
        raise ConfirmationError("all prescribed load values must be finite")
    tensor_norms = np.sqrt(basis[:, 0] ** 2 + basis[:, 1] ** 2 + 2.0 * basis[:, 2] ** 2)
    if not np.allclose(tensor_norms, 1.0, rtol=0.0, atol=1e-14):
        raise ConfirmationError("each basis load must have unit tensor-Frobenius norm")
    if int(np.linalg.matrix_rank(basis)) != 3:
        raise ConfirmationError("basis loads must have rank three")
    if not np.isclose(np.linalg.cond(basis), math.sqrt(2.0), rtol=0.0, atol=1e-14):
        raise ConfirmationError("basis load condition number must equal sqrt(2)")
    load_rows = [tuple(float(x) for x in row) for row in np.vstack([basis, heldout])]
    if len(set(load_rows)) != 8 or any(not any(row) for row in load_rows):
        raise ConfirmationError("the eight basis/held-out loads must be nonzero and unique")

    geometries = config.get("geometries")
    if not isinstance(geometries, list) or len(geometries) != 3:
        raise ConfirmationError("exactly three explicit geometries are required")
    families = [str(entry.get("section_family")) for entry in geometries]
    if families != ["circle", "horseshoe", "straight_wall_arch"]:
        raise ConfirmationError("geometry order/families must remain circle, horseshoe, arch")
    if len({str(entry.get("name")) for entry in geometries}) != 3:
        raise ConfirmationError("geometry names must be unique")
    if any(float(entry.get("roughness_amplitude", 0.0)) <= 0.0 for entry in geometries):
        raise ConfirmationError("every confirmation geometry must freeze positive roughness")

    query = config.get("query", {})
    if (
        sum(
            int(query.get(key, 0))
            for key in ("nearfield_points", "wall_offset_points", "farfield_points")
        )
        != 512
    ):
        raise ConfirmationError("the confirmation query must contain exactly 512 points")
    gates = config.get("gates", {})
    if (
        float(gates.get("median_relative_l2_max", math.inf)) != 1e-11
        or float(gates.get("maximum_relative_l2_max", math.inf)) != 1e-9
        or float(gates.get("maximum_auxiliary_response_relative_l2", math.inf)) != 1e-9
        or float(gates.get("maximum_solver_algebraic_residual", math.inf)) != 1e-9
        or float(gates.get("maximum_solver_energy_closure", math.inf)) != 1e-9
    ):
        raise ConfirmationError("confirmation numerical gates changed")
    _require_sha256(config.get("expected_plan_sha256"), "expected_plan_sha256")
    for relative, digest in (
        config.get("implementation_preflight", {}).get("required_path_sha256", {}).items()
    ):
        _relative_repository_path(ROOT / str(relative))
        _require_sha256(digest, f"implementation hash for {relative}")
    return config


@dataclass(frozen=True)
class FrozenGeometryRuntime:
    name: str
    section_family: str
    spec: GeometryDataSpec
    geometry: Any
    grid: Any
    boundary_sha256: str


@dataclass(frozen=True)
class ConfirmationPlan:
    identity: Mapping[str, Any]
    geometries: tuple[FrozenGeometryRuntime, ...]
    basis_loads: np.ndarray
    heldout_loads: np.ndarray


def _build_grid(spec: GeometryDataSpec, geometry: Any, query: Mapping[str, Any], seed: int) -> Any:
    return build_elastic_query_grid(
        geometry,
        geometry_parameters=spec.identity_parameters(),
        nearfield_points=int(query["nearfield_points"]),
        wall_offset_points=int(query["wall_offset_points"]),
        farfield_points=int(query["farfield_points"]),
        nearfield_scale=float(query["nearfield_scale"]),
        farfield_scale=float(query["farfield_scale"]),
        nearfield_min_distance_over_radius=float(query["nearfield_distance_over_radius"][0]),
        nearfield_max_distance_over_radius=float(query["nearfield_distance_over_radius"][1]),
        wall_offset_over_radius=float(query["wall_offset_over_radius"]),
        seed=int(seed),
        outer_domain_scale=float(spec.outer_domain_scale),
    )


def build_confirmation_plan(
    config: Mapping[str, Any], *, enforce_expected_hash: bool = True
) -> ConfirmationPlan:
    excluded, exclusion_sources = _load_excluded_identities(config)
    material = config["material"]
    young = float(material["young_modulus_over_reference_stress"])
    poisson = float(material["poisson_ratio"])
    query = config["query"]
    runtime: list[FrozenGeometryRuntime] = []
    geometry_records: list[dict[str, Any]] = []
    new_identities: dict[str, list[str]] = {
        "geometry_group_id": [],
        "boundary_float64_sha256": [],
        "case_group_id": [],
        "load_group_id": [],
        "query_hash": [],
    }

    for entry in config["geometries"]:
        spec = GeometryDataSpec(
            shape=str(entry["section_family"]),
            parameters=dict(entry["continuous_parameters"]),
            n_boundary_points=int(entry["boundary_points"]),
            radius=float(entry["characteristic_radius"]),
            roughness_amplitude=float(entry["roughness_amplitude"]),
            seed=int(entry["roughness_seed"]),
            outer_domain_scale=float(entry["outer_domain_scale"]),
        )
        geometry = spec.build()
        geometry_id = spec.geometry_group_id(geometry)
        boundary_hash = _array_sha256(geometry.boundary_yz, "<f8")
        grid = _build_grid(spec, geometry, query, int(entry["query_seed"]))
        if grid.geometry_group_id != geometry_id:
            raise ConfirmationError("query grid changed a frozen geometry identity")
        actual_expected = {
            "geometry_group_id": geometry_id,
            "boundary_float64_sha256": boundary_hash,
            "query_hash": grid.query_hash,
        }
        if entry.get("expected_identities") != actual_expected:
            raise ConfirmationError(f"frozen identity changed for geometry {entry['name']}")
        runtime.append(
            FrozenGeometryRuntime(
                name=str(entry["name"]),
                section_family=str(entry["section_family"]),
                spec=spec,
                geometry=geometry,
                grid=grid,
                boundary_sha256=boundary_hash,
            )
        )
        new_identities["geometry_group_id"].append(geometry_id)
        new_identities["boundary_float64_sha256"].append(boundary_hash)
        new_identities["query_hash"].append(grid.query_hash)
        geometry_records.append(
            {
                "name": str(entry["name"]),
                "section_family": str(entry["section_family"]),
                "continuous_parameters": dict(entry["continuous_parameters"]),
                "boundary_points": int(entry["boundary_points"]),
                "characteristic_radius": float(entry["characteristic_radius"]),
                "roughness_amplitude": float(entry["roughness_amplitude"]),
                "roughness_seed": int(entry["roughness_seed"]),
                "query_seed": int(entry["query_seed"]),
                **actual_expected,
            }
        )

    basis = np.asarray(config["basis_loads_tension_positive"], dtype=np.float64)
    heldout = np.asarray(config["heldout_loads_tension_positive"], dtype=np.float64)
    load_records: list[dict[str, Any]] = []
    for role, loads in (("basis", basis), ("heldout", heldout)):
        for index, vector in enumerate(loads):
            load_id = load_group_id(vector)
            new_identities["load_group_id"].append(load_id)
            load_records.append(
                {
                    "role": role,
                    "role_index": index,
                    "sigma_inf_yy_zz_yz_tension_positive": vector.tolist(),
                    "tensor_frobenius_norm": _finite_float(
                        np.sqrt(vector[0] ** 2 + vector[1] ** 2 + 2.0 * vector[2] ** 2),
                        f"{role}_load[{index}].tensor_frobenius_norm",
                    ),
                    "load_group_id": load_id,
                }
            )
    if config.get("expected_load_group_ids") != [row["load_group_id"] for row in load_records]:
        raise ConfirmationError("frozen load identities changed")

    case_records: list[dict[str, Any]] = []
    for geometry in runtime:
        for load in load_records:
            case_id = case_group_id(
                geometry.grid.geometry_group_id,
                str(load["load_group_id"]),
                young_modulus=young,
                poisson_ratio=poisson,
            )
            new_identities["case_group_id"].append(case_id)
            case_records.append(
                {
                    "geometry_name": geometry.name,
                    "role": load["role"],
                    "role_index": load["role_index"],
                    "geometry_group_id": geometry.grid.geometry_group_id,
                    "load_group_id": load["load_group_id"],
                    "case_group_id": case_id,
                    "query_hash": geometry.grid.query_hash,
                }
            )

    expected_cases = config.get("expected_case_group_ids")
    if expected_cases != [row["case_group_id"] for row in case_records]:
        raise ConfirmationError("frozen case identities changed")
    uniqueness_expectations = {
        "geometry_group_id": 3,
        "boundary_float64_sha256": 3,
        "query_hash": 3,
        "load_group_id": 8,
        "case_group_id": 24,
    }
    uniqueness_checks = {
        key: len(set(new_identities[key])) == count
        for key, count in uniqueness_expectations.items()
    }
    intersections = {
        key: sorted(set(values) & excluded[key]) for key, values in new_identities.items()
    }
    zero_intersection_checks = {key: not values for key, values in intersections.items()}
    if not all(uniqueness_checks.values()) or not all(zero_intersection_checks.values()):
        raise ConfirmationError("new-plan identity uniqueness/exclusion audit failed")

    identity_without_hash = {
        "schema": "tunnelgeopt.load_basis_confirmation.plan.v1",
        "run_id": str(config["run_id"]),
        "status": "validated_not_executed",
        "sign_convention": "tension_positive_internal_negative_values_are_compression",
        "material": dict(config["material"]),
        "fine_mesh": dict(config["fine_mesh"]),
        "query": dict(config["query"]),
        "basis": {
            "load_vectors": basis.tolist(),
            "tensor_frobenius_unit_norm": True,
            "rank": int(np.linalg.matrix_rank(basis)),
            "condition_number": _finite_float(np.linalg.cond(basis), "plan.basis.condition_number"),
        },
        "heldout_load_count": int(heldout.shape[0]),
        "geometry_count": len(runtime),
        "direct_fem_solve_count": len(case_records),
        "solve_contract": dict(config["solve_contract"]),
        "execution_evidence_contract": dict(config["execution_evidence_contract"]),
        "response_contract": dict(config["response_contract"]),
        "post_solve_classifications": dict(config["post_solve_classifications"]),
        "geometries": geometry_records,
        "loads": load_records,
        "cases": case_records,
        "identity_audit": {
            "new_unique_identity_counts": {
                key: len(set(values)) for key, values in sorted(new_identities.items())
            },
            "uniqueness_checks": uniqueness_checks,
            "excluded_identity_sources": exclusion_sources,
            "intersections": intersections,
            "zero_intersection_checks": zero_intersection_checks,
            "passed": all(uniqueness_checks.values()) and all(zero_intersection_checks.values()),
        },
        "gates": dict(config["gates"]),
        "claim_scope": dict(config["claim_scope"]),
    }
    plan_hash = _value_sha256(identity_without_hash)
    if enforce_expected_hash and plan_hash != config["expected_plan_sha256"]:
        raise ConfirmationError("frozen plan SHA-256 changed")
    identity = {**identity_without_hash, "plan_sha256": plan_hash}
    return ConfirmationPlan(
        identity=identity,
        geometries=tuple(runtime),
        basis_loads=basis,
        heldout_loads=heldout,
    )


def validate_plan(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate and return the complete frozen plan without solving or writing."""

    config = load_config(config_path)
    plan = build_confirmation_plan(config)
    return dict(plan.identity)


def tensor_frobenius_relative_l2(prediction: Any, reference: Any) -> float:
    """All-query relative L2 using the symmetric-tensor Frobenius norm."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        prediction_array.shape != reference_array.shape
        or prediction_array.ndim != 2
        or prediction_array.shape[1] != 3
        or not np.isfinite(prediction_array).all()
        or not np.isfinite(reference_array).all()
    ):
        raise ConfirmationError("stress comparison requires finite aligned [P,3] arrays")
    component_weights = np.asarray([1.0, 1.0, 2.0])
    numerator = float(np.sum(component_weights * (prediction_array - reference_array) ** 2))
    denominator = float(np.sum(component_weights * reference_array**2))
    if denominator <= np.finfo(float).tiny:
        raise ConfirmationError("direct held-out reference has zero tensor norm")
    return _finite_float(np.sqrt(numerator / denominator), "tensor_frobenius_relative_l2")


def evaluate_response_arrays(
    basis_loads: Any,
    basis_responses: Any,
    heldout_loads: Any,
    heldout_responses: Any,
) -> dict[str, Any]:
    """Fit three response fields and compare predictions with five direct fields."""

    basis_load_array = np.asarray(basis_loads, dtype=np.float64)
    heldout_load_array = np.asarray(heldout_loads, dtype=np.float64)
    direct = np.asarray(heldout_responses, dtype=np.float64)
    fitted_basis = fit_linear_stress_response_basis(basis_load_array, basis_responses)
    prediction = fitted_basis.predict(heldout_load_array)
    if direct.shape != prediction.shape:
        raise ConfirmationError("held-out direct response shape changed")
    errors = [tensor_frobenius_relative_l2(prediction[i], direct[i]) for i in range(5)]
    return {
        "basis_rank": int(fitted_basis.load_rank),
        "basis_condition_number": _finite_float(
            fitted_basis.load_condition_number, "basis_condition_number"
        ),
        "basis_relative_fit_residual": _finite_float(
            fitted_basis.relative_fit_residual, "basis_relative_fit_residual"
        ),
        "heldout_relative_l2": errors,
        "heldout_median_relative_l2": _finite_float(
            np.median(errors), "heldout_median_relative_l2"
        ),
        "heldout_maximum_relative_l2": _finite_float(np.max(errors), "heldout_maximum_relative_l2"),
    }


def _relative_l2(prediction: Any, reference: Any) -> float:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if (
        prediction_array.shape != reference_array.shape
        or prediction_array.size == 0
        or not np.isfinite(prediction_array).all()
        or not np.isfinite(reference_array).all()
    ):
        raise ConfirmationError("response comparison requires finite aligned nonempty arrays")
    denominator = float(np.linalg.norm(reference_array))
    if denominator <= np.finfo(float).tiny:
        raise ConfirmationError("direct held-out response has zero norm")
    return _finite_float(
        np.linalg.norm(prediction_array - reference_array) / denominator,
        "response_relative_l2",
    )


def evaluate_linear_response(
    basis_loads: Any,
    basis_responses: Any,
    heldout_loads: Any,
    heldout_responses: Any,
    *,
    tensor_frobenius: bool = False,
) -> dict[str, Any]:
    """Fit/evaluate one arbitrary response tensor along the three load axes."""

    loads = np.asarray(basis_loads, dtype=np.float64)
    basis = np.asarray(basis_responses, dtype=np.float64)
    heldout = np.asarray(heldout_responses, dtype=np.float64)
    heldout_load_array = np.asarray(heldout_loads, dtype=np.float64)
    if (
        loads.shape != (3, 3)
        or heldout_load_array.shape != (5, 3)
        or basis.shape[0] != 3
        or heldout.shape[0] != 5
        or basis.shape[1:] != heldout.shape[1:]
        or basis.size == 0
        or not np.isfinite(basis).all()
        or not np.isfinite(heldout).all()
    ):
        raise ConfirmationError("linear-response arrays do not satisfy the 3+5 contract")
    flattened = basis.reshape(3, -1)
    coefficients, _, rank, _ = np.linalg.lstsq(loads, flattened, rcond=None)
    if int(rank) != 3:
        raise ConfirmationError("linear-response basis lost rank three")
    predicted = (heldout_load_array @ coefficients).reshape(heldout.shape)
    if tensor_frobenius:
        if heldout.ndim != 3 or heldout.shape[-1] != 3:
            raise ConfirmationError("tensor-Frobenius response must have shape [5,P,3]")
        errors = [tensor_frobenius_relative_l2(predicted[i], heldout[i]) for i in range(5)]
    else:
        errors = [_relative_l2(predicted[i], heldout[i]) for i in range(5)]
    return {
        "basis_rank": int(rank),
        "basis_condition_number": _finite_float(
            np.linalg.cond(loads), "auxiliary_basis_condition_number"
        ),
        "heldout_relative_l2": errors,
        "heldout_median_relative_l2": _finite_float(
            np.median(errors), "auxiliary_heldout_median_relative_l2"
        ),
        "heldout_maximum_relative_l2": _finite_float(
            np.max(errors), "auxiliary_heldout_maximum_relative_l2"
        ),
    }


def classify_gate_checks(checks: Mapping[str, bool]) -> tuple[str, set[str], set[str]]:
    """Apply the frozen validity-before-effect three-way decision rule."""

    protocol_keys = {
        "exact_direct_fem_solve_count_24",
        "exact_heldout_comparison_count_15",
        "basis_rank_three_every_geometry",
        "basis_condition_sqrt_two_every_geometry",
        "solver_algebraic_residual",
        "solver_energy_closure",
        "mesh_query_boundary_fixed_per_geometry",
        "all_query_points_located",
        "explicit_boundary_tags",
        "no_element_centroid_inside_cavity",
        "new_identity_zero_intersection",
    }
    numerical_keys = set(checks) - protocol_keys
    if set(checks) != protocol_keys | numerical_keys or not numerical_keys:
        raise ConfirmationError("gate-check set is incomplete")
    if not all(bool(checks.get(key, False)) for key in protocol_keys):
        return INVALID_CLASSIFICATION, protocol_keys, numerical_keys
    if not all(bool(checks[key]) for key in numerical_keys):
        return STOP_CLASSIFICATION, protocol_keys, numerical_keys
    return CONFIRMED_CLASSIFICATION, protocol_keys, numerical_keys


def _git(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ConfirmationError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def real_execution_preflight(
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    expected_head: str,
) -> dict[str, Any]:
    """Require immutable pushed implementation provenance before any solve."""

    expected = _require_git_commit(expected_head, "expected_head")
    output_resolved = output_dir.resolve()
    output_relative = _relative_repository_path(output_resolved)
    if output_resolved.exists():
        raise ConfirmationError("confirmation output must be absent before the real run")
    if _git(["diff", "--name-only", "--"]) or _git(["diff", "--cached", "--name-only", "--"]):
        raise ConfirmationError("tracked worktree/index must be clean before the real run")
    head = _git(["rev-parse", "HEAD"])
    if head != expected:
        raise ConfirmationError("HEAD does not match the explicitly approved commit")
    upstream_commit = _git(["rev-parse", "@{upstream}"])
    if head != upstream_commit:
        raise ConfirmationError("HEAD must equal its pushed upstream before the real run")
    upstream_ref = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"])

    config_relative = _relative_repository_path(config_path)
    required = dict(config["implementation_preflight"]["required_path_sha256"])
    required_names = {
        "scripts/run_load_basis_confirmation.py",
        "src/tunnelgeopt/load_basis.py",
    }
    if set(required) != required_names:
        raise ConfirmationError("implementation preflight critical source set changed")
    if not _git(["ls-files", "--error-unmatch", "--", config_relative]):
        raise ConfirmationError("active confirmation config is not tracked")
    source_hashes: dict[str, str] = {}
    for relative, digest in sorted(required.items()):
        path = _resolve_source(str(relative))
        expected_digest = _require_sha256(digest, f"implementation hash for {relative}")
        actual = _file_sha256(path)
        if actual != expected_digest:
            raise ConfirmationError(f"implementation source hash changed: {relative}")
        if not _git(["ls-files", "--error-unmatch", "--", str(relative)]):
            raise ConfirmationError(f"implementation source is not tracked: {relative}")
        source_hashes[str(relative)] = actual
    return {
        "git_head": head,
        "git_upstream_commit": upstream_commit,
        "git_upstream_ref": upstream_ref,
        "tracked_clean": True,
        "head_equals_upstream": True,
        "expected_head_matched": True,
        "config_path": config_relative,
        "config_sha256": _file_sha256(config_path),
        "critical_source_sha256": source_hashes,
        "output_relative_path": output_relative,
        "output_absent_before_execution": True,
    }


def _outer_bounds(geometry: Any, grid: Any, domain_scale: float) -> tuple[float, ...]:
    center = np.asarray(grid.normalization_center_yz, dtype=np.float64)
    extent = np.ptp(np.asarray(geometry.boundary_yz, dtype=np.float64), axis=0)
    return (
        float(center[0] - 0.5 * extent[0] * domain_scale),
        float(center[0] + 0.5 * extent[0] * domain_scale),
        float(center[1] - 0.5 * extent[1] * domain_scale),
        float(center[1] + 0.5 * extent[1] * domain_scale),
    )


def _mesh_identity(mesh: Any) -> str:
    return _value_sha256(
        {
            "nodes_float64_sha256": _array_sha256(mesh.nodes, "<f8"),
            "elements_int64_sha256": _array_sha256(mesh.elements, "<i8"),
            "wall_facets_int64_sha256": _array_sha256(mesh.boundary_facets["wall"], "<i8"),
            "farfield_facets_int64_sha256": _array_sha256(mesh.boundary_facets["farfield"], "<i8"),
            "outer_bounds": [
                _finite_float(value, f"mesh.outer_bounds[{index}]")
                for index, value in enumerate(mesh.outer_bounds)
            ],
        }
    )


def _safe_error_text(error: BaseException) -> str:
    text = str(error).replace(str(ROOT.resolve()), "<repository>")
    text = text.replace(str(ROOT.resolve()).replace("\\", "/"), "<repository>")
    return text[:2000]


def _reserve_output_dir(output_dir: Path) -> None:
    """Atomically claim the absent run directory before any mesh or solve."""

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise ConfirmationError("confirmation output reservation lost an existence race") from exc


def _write_confirmation_artifact(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise ConfirmationError("confirmation output was not reserved before artifact write")
    safe_payload, serialization_issues = _json_safe(payload)
    if not isinstance(safe_payload, dict):
        raise ConfirmationError("confirmation artifact payload must be an object")
    if serialization_issues:
        previous = str(safe_payload.get("classification"))
        safe_payload["classification"] = INVALID_CLASSIFICATION
        safe_payload["all_gates_passed"] = False
        safe_payload["serialization_validity"] = {
            "passed": False,
            "classification_before_validation": previous,
            "issues": serialization_issues,
        }
    else:
        safe_payload["serialization_validity"] = {"passed": True, "issues": []}
    encoded = _canonical_bytes(safe_payload) + b"\n"
    artifact_path = output_dir / "confirmation.json"
    temporary = output_dir / "confirmation.json.tmp"
    if artifact_path.exists():
        raise ConfirmationError("reserved confirmation output already contains an artifact")
    with temporary.open("xb") as stream:
        stream.write(encoded)
    os.replace(temporary, artifact_path)
    return {
        "status": "completed",
        "classification": str(safe_payload["classification"]),
        "artifact": _relative_repository_path(artifact_path),
        "artifact_sha256": _file_sha256(artifact_path),
    }


def _sigma_matrix(vector: np.ndarray) -> np.ndarray:
    return np.asarray([[vector[0], vector[2]], [vector[2], vector[1]]], dtype=np.float64)


def _solve_geometry(
    frozen: FrozenGeometryRuntime,
    config: Mapping[str, Any],
    all_loads: np.ndarray,
    execution_records: list[dict[str, Any]],
) -> dict[str, Any]:
    mesh_values = config["fine_mesh"]
    radius = float(frozen.spec.radius)
    mesh_spec = MeshFidelitySpec(
        mesh_size=radius * float(mesh_values["mesh_size_over_radius"]),
        wall_mesh_size=radius * float(mesh_values["wall_size_over_radius"]),
        farfield_mesh_size=radius * float(mesh_values["farfield_size_over_radius"]),
    )
    bounds = _outer_bounds(frozen.geometry, frozen.grid, frozen.spec.outer_domain_scale)
    mesh = generate_tunnel_mesh(frozen.geometry, outer_bounds=bounds, **mesh_spec.kwargs())
    _finite_array(mesh.nodes, f"{frozen.name}.mesh.nodes")
    minimum_element_area = _finite_float(
        mesh.metadata["minimum_element_area"], f"{frozen.name}.minimum_element_area"
    )
    minimum_triangle_quality = _finite_float(
        mesh.metadata["minimum_triangle_quality"],
        f"{frozen.name}.minimum_triangle_quality",
    )
    mesh_hash = _mesh_identity(mesh)
    element_ids = locate_elements(
        mesh.nodes,
        mesh.elements,
        frozen.grid.points_yz,
        raise_outside=True,
    )
    centroids = np.asarray(mesh.nodes)[np.asarray(mesh.elements)].mean(axis=1)
    inside_count = int(
        np.sum(points_inside_polygon(centroids, np.asarray(frozen.geometry.boundary_yz)))
    )
    query_total_stress: list[np.ndarray] = []
    query_delta_stress: list[np.ndarray] = []
    nodal_displacement: list[np.ndarray] = []
    element_delta_stress: list[np.ndarray] = []
    element_sigma_xx: list[np.ndarray] = []
    solver_records: list[dict[str, Any]] = []
    material = config["material"]
    started = time.perf_counter()
    for load_index, vector in enumerate(all_loads):
        solve_started = time.perf_counter()
        load_id = load_group_id(vector)
        case_id = case_group_id(
            frozen.grid.geometry_group_id,
            load_id,
            young_modulus=float(material["young_modulus_over_reference_stress"]),
            poisson_ratio=float(material["poisson_ratio"]),
        )
        record: dict[str, Any] = {
            "planned_solve_index": len(execution_records),
            "geometry_name": frozen.name,
            "section_family": frozen.section_family,
            "geometry_group_id": frozen.grid.geometry_group_id,
            "load_index_within_geometry": load_index,
            "load_role": "basis" if load_index < 3 else "heldout",
            "load_role_index": load_index if load_index < 3 else load_index - 3,
            "load_group_id": load_id,
            "case_group_id": case_id,
            "attempted": True,
            "solver_returned": False,
            "validated_complete": False,
            "status": "attempted",
        }
        execution_records.append(record)
        solver_records.append(record)
        try:
            result = solve_plane_strain_excavation(
                mesh,
                young_modulus=float(material["young_modulus_over_reference_stress"]),
                poisson_ratio=float(material["poisson_ratio"]),
                sigma_inf=_sigma_matrix(vector),
            )
            record["solver_returned"] = True
            result_nodes = _finite_array(result.nodes, f"solve[{len(execution_records) - 1}].nodes")
            displacement = _finite_array(
                result.displacement,
                f"solve[{len(execution_records) - 1}].nodal_displacement",
            )
            delta_stress = _finite_array(
                result.delta_stress,
                f"solve[{len(execution_records) - 1}].element_delta_stress",
            )
            total_stress = _finite_array(
                result.total_stress,
                f"solve[{len(execution_records) - 1}].element_total_stress",
            )
            sigma_xx = _finite_array(
                result.sigma_xx,
                f"solve[{len(execution_records) - 1}].element_sigma_xx",
            )
            algebraic_residual = _finite_float(
                result.algebraic_residual,
                f"solve[{len(execution_records) - 1}].algebraic_residual",
            )
            energy_closure = _finite_float(
                result.energy_closure,
                f"solve[{len(execution_records) - 1}].energy_closure",
            )
            result_mesh_hash = _value_sha256(
                {
                    "nodes_float64_sha256": _array_sha256(result_nodes, "<f8"),
                    "elements_int64_sha256": _array_sha256(result.elements, "<i8"),
                    "wall_facets_int64_sha256": _array_sha256(
                        result.boundary_facets["wall"], "<i8"
                    ),
                    "farfield_facets_int64_sha256": _array_sha256(
                        result.boundary_facets["farfield"], "<i8"
                    ),
                    "outer_bounds": [
                        _finite_float(value, f"result.outer_bounds[{index}]")
                        for index, value in enumerate(mesh.outer_bounds)
                    ],
                }
            )
            sampled_total = _finite_array(
                sample_piecewise_constant(total_stress, element_ids),
                f"solve[{len(execution_records) - 1}].query_total_stress",
            )
            sampled_delta = _finite_array(
                sample_piecewise_constant(delta_stress, element_ids),
                f"solve[{len(execution_records) - 1}].query_delta_stress",
            )
            if (
                sampled_total.shape != (frozen.grid.point_count, 3)
                or sampled_delta.shape != sampled_total.shape
            ):
                raise ConfirmationError("direct solver returned an invalid query stress shape")
            query_total_stress.append(sampled_total)
            query_delta_stress.append(sampled_delta)
            nodal_displacement.append(displacement)
            element_delta_stress.append(delta_stress)
            element_sigma_xx.append(sigma_xx)
            record.update(
                {
                    "mesh_identity_sha256": result_mesh_hash,
                    "query_hash": frozen.grid.query_hash,
                    "boundary_float64_sha256": frozen.boundary_sha256,
                    "algebraic_residual": algebraic_residual,
                    "energy_closure": energy_closure,
                    "solver_seconds": _finite_float(
                        time.perf_counter() - solve_started,
                        f"solve[{len(execution_records) - 1}].solver_seconds",
                    ),
                    "validated_complete": True,
                    "status": "completed",
                }
            )
        except Exception as error:
            record.update(
                {
                    "validated_complete": False,
                    "status": "failed",
                    "failure": {
                        "error_type": type(error).__name__,
                        "reason": _safe_error_text(error),
                    },
                }
            )
            raise
    return {
        "responses": {
            "nodal_displacement": np.asarray(nodal_displacement, dtype=np.float64),
            "element_delta_stress": np.asarray(element_delta_stress, dtype=np.float64),
            "query_delta_stress": np.asarray(query_delta_stress, dtype=np.float64),
            "query_total_in_plane_stress": np.asarray(query_total_stress, dtype=np.float64),
            "element_sigma_xx_secondary": np.asarray(element_sigma_xx, dtype=np.float64),
        },
        "mesh_identity_sha256": mesh_hash,
        "node_count": int(mesh.nodes.shape[0]),
        "element_count": int(mesh.elements.shape[0]),
        "minimum_element_area": minimum_element_area,
        "minimum_triangle_quality": minimum_triangle_quality,
        "wall_facet_count": int(mesh.metadata["wall_facet_count"]),
        "farfield_facet_count": int(mesh.metadata["farfield_facet_count"]),
        "all_query_points_located": bool(np.all(element_ids >= 0)),
        "element_centroids_inside_cavity": inside_count,
        "solver_records": solver_records,
        "geometry_seconds": _finite_float(
            time.perf_counter() - started, f"{frozen.name}.geometry_seconds"
        ),
    }


def _execution_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    attempted = sum(bool(record.get("attempted")) for record in records)
    solver_returned = sum(bool(record.get("solver_returned")) for record in records)
    completed = sum(bool(record.get("validated_complete")) for record in records)
    failed = sum(str(record.get("status")) == "failed" for record in records)
    return {
        "planned_direct_fem_solve_count": 24,
        "attempted_direct_fem_solve_count": int(attempted),
        "solver_returned_count": int(solver_returned),
        "completed_validated_direct_fem_solve_count": int(completed),
        "failed_direct_fem_solve_count": int(failed),
        "not_attempted_direct_fem_solve_count": int(24 - attempted),
    }


def run_confirmation(
    config_path: Path,
    output_dir: Path,
    *,
    expected_head: str,
    acknowledgement: str,
) -> dict[str, Any]:
    """Run 24 solves and atomically persist every post-solve classification."""

    if acknowledgement != EXECUTION_ACKNOWLEDGEMENT:
        raise ConfirmationError("explicit 24-solve execution acknowledgement is required")
    config = load_config(config_path)
    preflight = real_execution_preflight(config, config_path, output_dir, expected_head)
    plan = build_confirmation_plan(config)
    _reserve_output_dir(output_dir)
    preflight = {
        **preflight,
        "output_directory_reserved_before_first_mesh_or_solve": True,
        "output_reservation_protocol": "atomic_mkdir_exist_ok_false",
    }
    all_loads = np.vstack([plan.basis_loads, plan.heldout_loads])
    geometry_results: list[dict[str, Any]] = []
    all_errors: list[float] = []
    execution_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    active_geometry: str | None = None
    try:
        for frozen in plan.geometries:
            active_geometry = frozen.name
            solved = _solve_geometry(frozen, config, all_loads, execution_records)
            responses = solved["responses"]
            query_total_analysis = evaluate_response_arrays(
                plan.basis_loads,
                responses["query_total_in_plane_stress"][:3],
                plan.heldout_loads,
                responses["query_total_in_plane_stress"][3:],
            )
            response_analyses = {
                "nodal_displacement": evaluate_linear_response(
                    plan.basis_loads,
                    responses["nodal_displacement"][:3],
                    plan.heldout_loads,
                    responses["nodal_displacement"][3:],
                ),
                "element_delta_stress": evaluate_linear_response(
                    plan.basis_loads,
                    responses["element_delta_stress"][:3],
                    plan.heldout_loads,
                    responses["element_delta_stress"][3:],
                    tensor_frobenius=True,
                ),
                "query_delta_stress": evaluate_linear_response(
                    plan.basis_loads,
                    responses["query_delta_stress"][:3],
                    plan.heldout_loads,
                    responses["query_delta_stress"][3:],
                    tensor_frobenius=True,
                ),
                "query_total_in_plane_stress": query_total_analysis,
                "element_sigma_xx_secondary": evaluate_linear_response(
                    plan.basis_loads,
                    responses["element_sigma_xx_secondary"][:3],
                    plan.heldout_loads,
                    responses["element_sigma_xx_secondary"][3:],
                ),
            }
            all_errors.extend(query_total_analysis["heldout_relative_l2"])
            identity_checks = {
                "one_mesh_identity_for_all_eight_solves": len(
                    {record["mesh_identity_sha256"] for record in solved["solver_records"]}
                )
                == 1,
                "one_query_identity_for_all_eight_solves": len(
                    {record["query_hash"] for record in solved["solver_records"]}
                )
                == 1,
                "one_boundary_identity_for_all_eight_solves": len(
                    {record["boundary_float64_sha256"] for record in solved["solver_records"]}
                )
                == 1,
                "solver_result_matches_generated_mesh": all(
                    record["mesh_identity_sha256"] == solved["mesh_identity_sha256"]
                    for record in solved["solver_records"]
                ),
            }
            geometry_results.append(
                {
                    "geometry_name": frozen.name,
                    "section_family": frozen.section_family,
                    "geometry_group_id": frozen.grid.geometry_group_id,
                    "boundary_float64_sha256": frozen.boundary_sha256,
                    "query_hash": frozen.grid.query_hash,
                    "mesh_identity_sha256": solved["mesh_identity_sha256"],
                    "node_count": solved["node_count"],
                    "element_count": solved["element_count"],
                    "minimum_element_area": solved["minimum_element_area"],
                    "minimum_triangle_quality": solved["minimum_triangle_quality"],
                    "wall_facet_count": solved["wall_facet_count"],
                    "farfield_facet_count": solved["farfield_facet_count"],
                    "all_query_points_located": solved["all_query_points_located"],
                    "element_centroids_inside_cavity": solved["element_centroids_inside_cavity"],
                    "response_analyses": response_analyses,
                    "identity_checks": identity_checks,
                    "solver_records": solved["solver_records"],
                    "geometry_seconds": solved["geometry_seconds"],
                }
            )
    except Exception as error:  # noqa: BLE001 - a post-preflight failure is durable evidence
        invalid_payload = {
            "schema": "tunnelgeopt.load_basis_confirmation.result.v1",
            "run_id": config["run_id"],
            "classification": INVALID_CLASSIFICATION,
            "classification_rule": {
                INVALID_CLASSIFICATION: (
                    "any_identity_hash_direct_path_or_solver_mesh_qc_gate_fails"
                ),
                STOP_CLASSIFICATION: ("protocol_valid_but_any_numerical_reconstruction_gate_fails"),
                CONFIRMED_CLASSIFICATION: "every_protocol_and_numerical_gate_passes",
            },
            "claim_scope": config["claim_scope"],
            "plan": plan.identity,
            "execution_preflight": preflight,
            "execution_failure": {
                "active_geometry": active_geometry,
                "error_type": type(error).__name__,
                "error": _safe_error_text(error),
                "completed_geometry_count": len(geometry_results),
            },
            "execution_summary": _execution_summary(execution_records),
            "solve_execution_records": execution_records,
            "partial_geometry_results": geometry_results,
            "all_gates_passed": False,
            "runtime_seconds": _finite_float(
                time.perf_counter() - started, "invalid_run.runtime_seconds"
            ),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        return _write_confirmation_artifact(output_dir, invalid_payload)

    gates = config["gates"]
    checks = {
        "exact_direct_fem_solve_count_24": (
            _execution_summary(execution_records)["completed_validated_direct_fem_solve_count"]
            == 24
        ),
        "exact_heldout_comparison_count_15": len(all_errors) == 15,
        "basis_rank_three_every_geometry": all(
            all(response["basis_rank"] == 3 for response in row["response_analyses"].values())
            for row in geometry_results
        ),
        "basis_condition_sqrt_two_every_geometry": all(
            all(
                np.isclose(
                    response["basis_condition_number"],
                    math.sqrt(2.0),
                    rtol=0.0,
                    atol=1e-14,
                )
                for response in row["response_analyses"].values()
            )
            for row in geometry_results
        ),
        "median_relative_l2": _finite_float(np.median(all_errors), "primary.median_relative_l2")
        <= float(gates["median_relative_l2_max"]),
        "maximum_relative_l2": _finite_float(np.max(all_errors), "primary.maximum_relative_l2")
        <= float(gates["maximum_relative_l2_max"]),
        "nodal_displacement_maximum_relative_l2": max(
            row["response_analyses"]["nodal_displacement"]["heldout_maximum_relative_l2"]
            for row in geometry_results
        )
        <= float(gates["maximum_auxiliary_response_relative_l2"]),
        "element_delta_stress_maximum_relative_l2": max(
            row["response_analyses"]["element_delta_stress"]["heldout_maximum_relative_l2"]
            for row in geometry_results
        )
        <= float(gates["maximum_auxiliary_response_relative_l2"]),
        "query_delta_stress_maximum_relative_l2": max(
            row["response_analyses"]["query_delta_stress"]["heldout_maximum_relative_l2"]
            for row in geometry_results
        )
        <= float(gates["maximum_auxiliary_response_relative_l2"]),
        "element_sigma_xx_secondary_maximum_relative_l2": max(
            row["response_analyses"]["element_sigma_xx_secondary"]["heldout_maximum_relative_l2"]
            for row in geometry_results
        )
        <= float(gates["maximum_auxiliary_response_relative_l2"]),
        "solver_algebraic_residual": max(
            record["algebraic_residual"] for record in execution_records
        )
        <= float(gates["maximum_solver_algebraic_residual"]),
        "solver_energy_closure": max(record["energy_closure"] for record in execution_records)
        <= float(gates["maximum_solver_energy_closure"]),
        "mesh_query_boundary_fixed_per_geometry": all(
            all(row["identity_checks"].values()) for row in geometry_results
        ),
        "all_query_points_located": all(
            row["all_query_points_located"] for row in geometry_results
        ),
        "explicit_boundary_tags": all(
            row["wall_facet_count"] > 0 and row["farfield_facet_count"] > 0
            for row in geometry_results
        ),
        "no_element_centroid_inside_cavity": all(
            row["element_centroids_inside_cavity"] == 0 for row in geometry_results
        ),
        "new_identity_zero_intersection": bool(plan.identity["identity_audit"]["passed"]),
    }

    classification, protocol_keys, numerical_keys = classify_gate_checks(checks)
    protocol_valid = all(checks[key] for key in protocol_keys)
    numerical_passed = all(checks[key] for key in numerical_keys)

    payload = {
        "schema": "tunnelgeopt.load_basis_confirmation.result.v1",
        "run_id": config["run_id"],
        "classification": classification,
        "classification_rule": {
            INVALID_CLASSIFICATION: "any_identity_hash_direct_path_or_solver_mesh_qc_gate_fails",
            STOP_CLASSIFICATION: "protocol_valid_but_any_numerical_reconstruction_gate_fails",
            CONFIRMED_CLASSIFICATION: "every_protocol_and_numerical_gate_passes",
        },
        "claim_scope": config["claim_scope"],
        "plan": plan.identity,
        "execution_preflight": preflight,
        "evaluation": {
            "direct_fem_solve_count": len(execution_records),
            "heldout_comparison_count": len(all_errors),
            "relative_l2_definition": (
                "all_512_queries_symmetric_tensor_frobenius_relative_l2_with_shear_weight_two"
            ),
            "heldout_relative_l2": all_errors,
            "heldout_median_relative_l2": _finite_float(
                np.median(all_errors), "evaluation.heldout_median_relative_l2"
            ),
            "heldout_maximum_relative_l2": _finite_float(
                np.max(all_errors), "evaluation.heldout_maximum_relative_l2"
            ),
            "maximum_solver_algebraic_residual": _finite_float(
                max(record["algebraic_residual"] for record in execution_records),
                "evaluation.maximum_solver_algebraic_residual",
            ),
            "maximum_solver_energy_closure": _finite_float(
                max(record["energy_closure"] for record in execution_records),
                "evaluation.maximum_solver_energy_closure",
            ),
            "execution_summary": _execution_summary(execution_records),
            "solve_execution_records": execution_records,
            "geometry_results": geometry_results,
        },
        "gate_checks": checks,
        "protocol_gate_keys": sorted(protocol_keys),
        "numerical_gate_keys": sorted(numerical_keys),
        "protocol_valid": protocol_valid,
        "numerical_reconstruction_passed": numerical_passed,
        "all_gates_passed": all(checks.values()),
        "runtime_seconds": _finite_float(
            time.perf_counter() - started, "completed_run.runtime_seconds"
        ),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    return _write_confirmation_artifact(output_dir, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate-plan", help="solver-free plan validation")
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute = subparsers.add_parser("run", help="run the guarded 24-solve confirmation")
    execute.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    execute.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    execute.add_argument("--expected-head", required=True)
    execute.add_argument("--acknowledge", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "validate-plan":
            result = validate_plan(arguments.config)
        else:
            result = run_confirmation(
                arguments.config,
                arguments.output,
                expected_head=arguments.expected_head,
                acknowledgement=arguments.acknowledge,
            )
    except (ConfirmationError, ValueError) as exc:
        print(
            _canonical_bytes({"status": "failed", "error": str(exc)}).decode("utf-8"),
            file=sys.stderr,
        )
        return 2
    print(_canonical_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
