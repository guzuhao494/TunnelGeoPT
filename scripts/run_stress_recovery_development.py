#!/usr/bin/env python3
"""Run a deterministic stress-recovery diagnostic on already-seen v0.3 cases.

Every v0.3 identity used here is explicitly treated as seen.  The runner opens
fine and ultrafine values only for development diagnosis and can never emit a
formal or independent-test effect claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tunnelgeopt.elasticity import solve_plane_strain_excavation
from tunnelgeopt.field_sampling import locate_elements, sample_piecewise_constant
from tunnelgeopt.formal_generation import (
    FORMAL_PARTITIONS,
    FormalGenerationOverrides,
    PlannedCase,
    PlannedGeometry,
    build_formal_generation_plan,
)
from tunnelgeopt.geometry import points_inside_polygon, surface_points_and_normals
from tunnelgeopt.mesh import generate_tunnel_mesh
from tunnelgeopt.multifidelity import (
    MeshFidelitySpec,
    build_elastic_query_grid,
    farfield_stress_scale,
)
from tunnelgeopt.stress_recovery import recover_stress_at_queries

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "stress_recovery_development.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "development" / "stress-recovery-v0.5-dev"
EXPECTED_SECTIONS = ("circle", "horseshoe", "straight_wall_arch")
EXPECTED_PARTITIONS = (
    "train_id",
    "dev_id",
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
)
SOURCE_FILES = (
    "configs/multifidelity_formal.json",
    "configs/stress_recovery_development.json",
    "scripts/run_stress_recovery_development.py",
    "src/tunnelgeopt/elasticity.py",
    "src/tunnelgeopt/field_sampling.py",
    "src/tunnelgeopt/formal_generation.py",
    "src/tunnelgeopt/geometry.py",
    "src/tunnelgeopt/mesh.py",
    "src/tunnelgeopt/multifidelity.py",
    "src/tunnelgeopt/stress_recovery.py",
)


class DevelopmentRunError(RuntimeError):
    """Raised when the frozen development-only contract is violated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_canonical_bytes(value).decode("utf-8") + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentRunError(f"could not load stress-recovery config: {exc}") from exc
    required = {
        "schema_version",
        "config_name",
        "run_id",
        "status",
        "scope",
        "claim_exclusions",
        "source",
        "selection",
        "operator",
        "metrics",
        "quality_control",
        "exploratory_gates",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise DevelopmentRunError("stress-recovery config key set changed")
    if config["schema_version"] != "tunnelgeopt.stress_recovery.development.v1":
        raise DevelopmentRunError("unsupported stress-recovery schema")
    if config["status"] != "frozen_development_only_before_execution":
        raise DevelopmentRunError("stress-recovery config must be frozen before execution")
    if config["scope"] != (
        "development_only_deterministic_stress_recovery_diagnostic_on_seen_v03_cases"
    ):
        raise DevelopmentRunError("stress-recovery scope must remain development-only")
    source = config["source"]
    if (
        source.get("all_v03_cases_declared_seen") is not True
        or source.get("fine_and_ultrafine_labels_are_development_only") is not True
        or source.get("locked_or_independent_label_claim_allowed") is not False
        or source.get("effect_claim_allowed") is not False
    ):
        raise DevelopmentRunError("v0.3 identities and labels must remain seen/development-only")
    selection = config["selection"]
    if tuple(selection.get("section_families", ())) != EXPECTED_SECTIONS:
        raise DevelopmentRunError("the three section families must remain frozen")
    if tuple(selection.get("partitions", ())) != EXPECTED_PARTITIONS:
        raise DevelopmentRunError("the five seen v0.3 partitions must remain frozen")
    if int(selection.get("cases_per_partition_section", 0)) != 1:
        raise DevelopmentRunError("exactly one case per partition/section is frozen")
    expected_count = len(EXPECTED_SECTIONS) * len(EXPECTED_PARTITIONS)
    if int(selection.get("expected_case_count", -1)) != expected_count:
        raise DevelopmentRunError("expected_case_count disagrees with the frozen design")
    if selection.get("selection_must_not_use_solver_or_label_values") is not True:
        raise DevelopmentRunError("selection must be independent of solver and label values")
    operator = config["operator"]
    if (
        operator.get("parameter_status") != "frozen_default_no_tuning"
        or operator.get("linearity_in_element_stress_required") is not True
    ):
        raise DevelopmentRunError("the default recovery operator must remain frozen and linear")
    gates = config["exploratory_gates"]
    if (
        gates.get("interpretation") != "development_routing_only_not_a_formal_effect_gate"
        or gates.get("effect_claim_allowed") is not False
    ):
        raise DevelopmentRunError("exploratory gates may not authorize an effect claim")
    return config


def _load_formal_config(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    relative = Path(str(config["source"]["formal_config"]))
    path = relative if relative.is_absolute() else ROOT / relative
    if path.resolve() == config_path.resolve():
        raise DevelopmentRunError("source formal config cannot be the recovery config")
    try:
        formal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentRunError(f"could not load source formal config: {exc}") from exc
    if formal.get("schema_version") != "tunnelgeopt.multifidelity.formal.v1":
        raise DevelopmentRunError("source is not the v0.3 formal schema")
    if formal.get("run_id") != "mf-residual-formal-v0.3.0":
        raise DevelopmentRunError("source formal run identity changed")
    return formal


def build_seen_v03_plan(formal: Mapping[str, Any]) -> Any:
    """Rebuild the complete v0.3 plan without treating any identity as locked."""

    # Passing all partitions explicitly activates the non-formal override while
    # preserving every original count, seed, identity and case order.
    plan = build_formal_generation_plan(
        formal,
        FormalGenerationOverrides(partitions=tuple(FORMAL_PARTITIONS)),
    )
    if plan.formal_eligible:
        raise DevelopmentRunError("development plan unexpectedly remained formal-eligible")
    if len(plan.geometries) != 195 or len(plan.cases) != 705:
        raise DevelopmentRunError("full v0.3 plan did not reconstruct 195 parents / 705 cases")
    return plan


def select_cases(
    plan: Any,
    selection: Mapping[str, Any],
) -> tuple[PlannedCase, ...]:
    """Select one case in each frozen cell using metadata-only hashes."""

    salt = str(selection["salt"])
    count = int(selection["cases_per_partition_section"])
    selected: list[PlannedCase] = []
    for partition in selection["partitions"]:
        for section in selection["section_families"]:
            bucket = [
                case
                for case in plan.cases
                if case.formal_partition == partition and case.section_family == section
            ]
            if len(bucket) < count:
                raise DevelopmentRunError(f"empty selection cell: {partition}/{section}")
            ranked = sorted(
                bucket,
                key=lambda case: _sha256_bytes(
                    _canonical_bytes(
                        {
                            "salt": salt,
                            "partition": partition,
                            "section": section,
                            "case_group_id": case.case_group_id,
                        }
                    )
                ),
            )
            selected.extend(ranked[:count])
    if len(selected) != int(selection["expected_case_count"]):
        raise DevelopmentRunError("selected case count changed")
    if len({case.case_group_id for case in selected}) != len(selected):
        raise DevelopmentRunError("selection repeated a case identity")
    return tuple(selected)


def _mesh_spec(formal: Mapping[str, Any], tier: str) -> MeshFidelitySpec:
    values = formal["mesh"]["tiers"][tier]
    radius = float(formal["geometry"]["characteristic_radius"])
    return MeshFidelitySpec(
        mesh_size=radius * float(values["mesh_size_over_radius"]),
        wall_mesh_size=radius * float(values["wall_size_over_radius"]),
        farfield_mesh_size=radius * float(values["farfield_size_over_radius"]),
    )


def _build_grid(geometry_entry: PlannedGeometry, formal: Mapping[str, Any]) -> Any:
    geometry = geometry_entry.spec.build()
    query = formal["query"]
    grid = build_elastic_query_grid(
        geometry,
        geometry_parameters=geometry_entry.spec.identity_parameters(),
        nearfield_points=int(query["nearfield_volume"]),
        wall_offset_points=int(query["wall_offset"]),
        farfield_points=int(query["farfield"]),
        nearfield_scale=float(query["nearfield_scale"]),
        farfield_scale=float(query["farfield_scale"]),
        nearfield_min_distance_over_radius=float(query["nearfield_distance_over_radius"][0]),
        nearfield_max_distance_over_radius=float(query["nearfield_distance_over_radius"][1]),
        wall_offset_over_radius=float(query["wall_offset_over_radius"]),
        seed=int(geometry_entry.query_seed),
        outer_domain_scale=float(geometry_entry.spec.outer_domain_scale),
    )
    if grid.geometry_group_id != geometry_entry.geometry_group_id:
        raise DevelopmentRunError("rebuilt query grid changed the geometry identity")
    return geometry, grid


def _outer_bounds(geometry: Any, grid: Any, domain_scale: float) -> tuple[float, ...]:
    center = np.asarray(grid.normalization_center_yz, dtype=np.float64)
    boundary = np.asarray(geometry.boundary_yz, dtype=np.float64)
    extent = np.ptp(boundary, axis=0)
    return (
        float(center[0] - 0.5 * extent[0] * domain_scale),
        float(center[0] + 0.5 * extent[0] * domain_scale),
        float(center[1] - 0.5 * extent[1] * domain_scale),
        float(center[1] + 0.5 * extent[1] * domain_scale),
    )


def _solve_tier(
    geometry: Any,
    grid: Any,
    sigma_inf: np.ndarray,
    mesh_spec: MeshFidelitySpec,
    formal: Mapping[str, Any],
    outer_bounds: tuple[float, ...],
) -> dict[str, Any]:
    started = time.perf_counter()
    mesh = generate_tunnel_mesh(geometry, outer_bounds=outer_bounds, **mesh_spec.kwargs())
    result = solve_plane_strain_excavation(
        mesh,
        young_modulus=float(formal["material_and_loads"]["young_modulus_over_reference_stress"]),
        poisson_ratio=float(formal["material_and_loads"]["poisson_ratio"]),
        sigma_inf=sigma_inf,
    )
    element_ids = locate_elements(
        result.nodes,
        result.elements,
        grid.points_yz,
        raise_outside=True,
    )
    sampled = np.asarray(
        sample_piecewise_constant(result.total_stress, element_ids), dtype=np.float64
    )
    centroids = np.asarray(result.nodes)[np.asarray(result.elements)].mean(axis=1)
    inside_count = int(np.sum(points_inside_polygon(centroids, np.asarray(geometry.boundary_yz))))
    elapsed = time.perf_counter() - started
    return {
        "mesh": mesh,
        "result": result,
        "element_ids": np.asarray(element_ids, dtype=np.int64),
        "sampled": sampled,
        "seconds": float(elapsed),
        "qc": {
            "node_count": int(result.nodes.shape[0]),
            "element_count": int(result.elements.shape[0]),
            "minimum_element_area": float(mesh.metadata["minimum_element_area"]),
            "minimum_triangle_quality": float(mesh.metadata["minimum_triangle_quality"]),
            "wall_facet_count": int(mesh.metadata["wall_facet_count"]),
            "farfield_facet_count": int(mesh.metadata["farfield_facet_count"]),
            "explicit_wall_and_farfield_tags": bool(
                mesh.boundary_facets["wall"].size > 0
                and mesh.boundary_facets["farfield"].size > 0
                and set(mesh.physical_tags) == {"rock", "wall", "farfield"}
            ),
            "element_centroids_inside_cavity": inside_count,
            "all_query_points_located": bool(np.all(element_ids >= 0)),
            "algebraic_residual": float(result.algebraic_residual),
            "energy_closure": float(result.energy_closure),
            "outer_bounds": [float(value) for value in mesh.outer_bounds],
            "solver_seconds": float(elapsed),
        },
    }


def relative_tensor_error(
    prediction: np.ndarray,
    reference: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Area-weighted [yy, zz, yz] tensor Frobenius relative L2."""

    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if (
        prediction.shape != reference.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 3
        or weights.shape != (prediction.shape[0],)
    ):
        raise DevelopmentRunError("relative-error arrays do not align")
    if (
        not all(np.isfinite(value).all() for value in (prediction, reference, weights))
        or np.any(weights < 0.0)
        or float(weights.sum()) <= 0.0
    ):
        raise DevelopmentRunError("relative-error inputs are invalid")
    components = np.asarray([1.0, 1.0, 2.0])
    numerator = float(np.sum(weights[:, None] * components * (prediction - reference) ** 2))
    denominator = float(np.sum(weights[:, None] * components * reference**2))
    if denominator <= np.finfo(float).tiny:
        raise DevelopmentRunError("reference stress norm is zero")
    return float(np.sqrt(numerator / denominator))


def wall_offset_discrepancy(
    prediction: np.ndarray,
    reference: np.ndarray,
    arc_weights: np.ndarray,
    normals_yz: np.ndarray,
) -> tuple[float, float]:
    """Return normalized traction and resultant discrepancies at wall offsets."""

    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    weights = np.asarray(arc_weights, dtype=np.float64)
    normals = np.asarray(normals_yz, dtype=np.float64)
    if (
        prediction.shape != reference.shape
        or prediction.ndim != 2
        or prediction.shape[1] != 3
        or weights.shape != (prediction.shape[0],)
        or normals.shape != (prediction.shape[0], 2)
    ):
        raise DevelopmentRunError("wall-offset arrays do not align")
    if (
        not all(np.isfinite(value).all() for value in (prediction, reference, weights, normals))
        or np.any(weights < 0.0)
        or not np.isclose(float(weights.sum()), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise DevelopmentRunError("wall-offset inputs are invalid")
    difference = prediction - reference
    tensor = np.empty((difference.shape[0], 2, 2), dtype=np.float64)
    tensor[:, 0, 0] = difference[:, 0]
    tensor[:, 1, 1] = difference[:, 1]
    tensor[:, 0, 1] = difference[:, 2]
    tensor[:, 1, 0] = difference[:, 2]
    traction = np.einsum("pij,pj->pi", tensor, normals)
    d_t = float(np.sqrt(np.sum(weights * np.sum(traction**2, axis=1))))
    d_r = float(np.linalg.norm(np.sum(weights[:, None] * traction, axis=0)))
    return d_t, d_r


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise DevelopmentRunError("cannot summarize an empty or non-finite distribution")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _ratio_center(records: Sequence[Mapping[str, Any]], numerator: str, denominator: str) -> float:
    top = np.asarray([float(record["metrics"][numerator]) for record in records])
    bottom = np.asarray([float(record["metrics"][denominator]) for record in records])
    mean_bottom = float(np.mean(bottom))
    if mean_bottom <= np.finfo(float).tiny:
        raise DevelopmentRunError("ratio denominator center is zero")
    return float(np.mean(top) / mean_bottom)


def _wall_ratio_center(
    records: Sequence[Mapping[str, Any]],
    reference: str,
    diagnostic: str,
) -> float:
    recovered = np.asarray(
        [record["wall_offset"][reference]["recovered"][diagnostic] for record in records],
        dtype=np.float64,
    )
    raw = np.asarray(
        [record["wall_offset"][reference]["raw_coarse"][diagnostic] for record in records],
        dtype=np.float64,
    )
    if float(np.mean(raw)) <= np.finfo(float).tiny:
        raise DevelopmentRunError("wall ratio denominator center is zero")
    return float(np.mean(recovered) / np.mean(raw))


def aggregate_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if len(records) != int(config["selection"]["expected_case_count"]):
        raise DevelopmentRunError("aggregate requires the complete frozen case set")
    raw_ultra = [float(record["metrics"]["raw_coarse_vs_ultrafine"]) for record in records]
    recovered_ultra = [
        float(record["metrics"]["recovered_coarse_vs_ultrafine"]) for record in records
    ]
    raw_fine = [float(record["metrics"]["raw_coarse_vs_fine"]) for record in records]
    recovered_fine = [float(record["metrics"]["recovered_coarse_vs_fine"]) for record in records]
    case_ratios = [float(record["metrics"]["recovery_raw_ratio_ultrafine"]) for record in records]
    section_summary: dict[str, Any] = {}
    for section in config["selection"]["section_families"]:
        subset = [record for record in records if record["section_family"] == section]
        section_summary[section] = {
            "case_count": len(subset),
            "ultrafine_center_ratio": _ratio_center(
                subset, "recovered_coarse_vs_ultrafine", "raw_coarse_vs_ultrafine"
            ),
            "fine_center_ratio": _ratio_center(
                subset, "recovered_coarse_vs_fine", "raw_coarse_vs_fine"
            ),
            "case_ratio_ultrafine": _distribution(
                [record["metrics"]["recovery_raw_ratio_ultrafine"] for record in subset]
            ),
        }
    partition_summary: dict[str, Any] = {}
    for partition in config["selection"]["partitions"]:
        subset = [record for record in records if record["formal_partition"] == partition]
        partition_summary[partition] = {
            "case_count": len(subset),
            "ultrafine_center_ratio": _ratio_center(
                subset, "recovered_coarse_vs_ultrafine", "raw_coarse_vs_ultrafine"
            ),
            "fine_center_ratio": _ratio_center(
                subset, "recovered_coarse_vs_fine", "raw_coarse_vs_fine"
            ),
        }
    all_qc = [
        record["solver_mesh_qc"][tier]
        for record in records
        for tier in ("coarse", "fine", "ultrafine")
    ]
    quality = config["quality_control"]
    qc_checks = {
        "complete_case_count": len(records) == int(config["selection"]["expected_case_count"]),
        "maximum_algebraic_residual": max(row["algebraic_residual"] for row in all_qc)
        <= float(quality["maximum_free_dof_algebraic_residual"]),
        "maximum_energy_closure": max(row["energy_closure"] for row in all_qc)
        <= float(quality["maximum_clapeyron_relative_energy_error"]),
        "minimum_triangle_quality": min(row["minimum_triangle_quality"] for row in all_qc)
        >= float(quality["minimum_triangle_quality"]),
        "all_query_points_located": all(row["all_query_points_located"] for row in all_qc),
        "explicit_wall_and_farfield_tags": all(
            row["explicit_wall_and_farfield_tags"] for row in all_qc
        ),
        "zero_element_centroids_inside_cavity": all(
            row["element_centroids_inside_cavity"] == 0 for row in all_qc
        ),
        "same_boundary_outer_bounds_and_query": all(
            record["identity_qc"]["passed"] for record in records
        ),
    }
    gates = config["exploratory_gates"]
    overall_ratio = float(np.mean(recovered_ultra) / np.mean(raw_ultra))
    wall_traction_ratio = _wall_ratio_center(records, "ultrafine", "traction")
    wall_resultant_ratio = _wall_ratio_center(records, "ultrafine", "resultant")
    exploratory_checks = {
        "overall_ultrafine_center_ratio": overall_ratio
        <= float(gates["overall_ultrafine_center_ratio_max"]),
        "each_section_ultrafine_center_ratio": all(
            row["ultrafine_center_ratio"] <= float(gates["each_section_ultrafine_center_ratio_max"])
            for row in section_summary.values()
        ),
        "wall_traction_ultrafine_center_ratio": wall_traction_ratio
        <= float(gates["wall_traction_ultrafine_center_ratio_max"]),
        "wall_resultant_ultrafine_center_ratio": wall_resultant_ratio
        <= float(gates["wall_resultant_ultrafine_center_ratio_max"]),
        "effect_claim_allowed": False,
    }
    positive_exploratory = {
        key: value for key, value in exploratory_checks.items() if key != "effect_claim_allowed"
    }
    return {
        "schema": "tunnelgeopt.stress_recovery.development.summary.v1",
        "run_id": config["run_id"],
        "scope": config["scope"],
        "all_v03_cases_declared_seen": True,
        "effect_claim_allowed": False,
        "case_count": len(records),
        "primary_against_ultrafine": {
            "raw_coarse_error": _distribution(raw_ultra),
            "recovered_error": _distribution(recovered_ultra),
            "center_ratio_recovered_over_raw": overall_ratio,
            "case_ratio_distribution": _distribution(case_ratios),
        },
        "secondary_against_fine": {
            "raw_coarse_error": _distribution(raw_fine),
            "recovered_error": _distribution(recovered_fine),
            "center_ratio_recovered_over_raw": float(np.mean(recovered_fine) / np.mean(raw_fine)),
        },
        "fine_ultrafine_reference_discrepancy": _distribution(
            [record["metrics"]["fine_vs_ultrafine"] for record in records]
        ),
        "by_section": section_summary,
        "by_partition": partition_summary,
        "wall_offset_center_ratios": {
            reference: {
                diagnostic: _wall_ratio_center(records, reference, diagnostic)
                for diagnostic in ("traction", "resultant")
            }
            for reference in ("fine", "ultrafine")
        },
        "solver_mesh_qc": {
            "maximum_algebraic_residual": float(max(row["algebraic_residual"] for row in all_qc)),
            "maximum_energy_closure": float(max(row["energy_closure"] for row in all_qc)),
            "minimum_triangle_quality": float(
                min(row["minimum_triangle_quality"] for row in all_qc)
            ),
            "checks": qc_checks,
            "passed": all(qc_checks.values()),
        },
        "runtime": {
            "total_solver_seconds": float(
                sum(
                    record["solver_mesh_qc"][tier]["solver_seconds"]
                    for record in records
                    for tier in ("coarse", "fine", "ultrafine")
                )
            ),
            "mean_case_seconds": float(np.mean([record["case_seconds"] for record in records])),
            "maximum_case_seconds": float(np.max([record["case_seconds"] for record in records])),
        },
        "exploratory_checks": exploratory_checks,
        "development_routing": (
            "PROMISING_FOR_NEW_UNSEEN_CONFIRMATORY_DESIGN"
            if all(positive_exploratory.values()) and all(qc_checks.values())
            else "STOP_OR_REDESIGN_BEFORE_ANY_NEW_UNSEEN_CONFIRMATORY_RUN"
        ),
        "claim_boundary": (
            "Development-only deterministic diagnostic on already-seen v0.3 identities. "
            "No formal effect, independent test, fracture, rockburst, field, or transfer claim."
        ),
    }


def _case_record(
    case: PlannedCase,
    geometry_entry: PlannedGeometry,
    formal: Mapping[str, Any],
    recovery_config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    geometry, grid = _build_grid(geometry_entry, formal)
    domain_scale = float(geometry_entry.spec.outer_domain_scale)
    bounds = _outer_bounds(geometry, grid, domain_scale)
    sigma_inf = np.asarray(case.sigma_inf_tension_positive, dtype=np.float64)
    solved = {
        tier: _solve_tier(
            geometry,
            grid,
            sigma_inf,
            _mesh_spec(formal, tier),
            formal,
            bounds,
        )
        for tier in ("coarse", "fine", "ultrafine_audit")
    }
    solved["ultrafine"] = solved.pop("ultrafine_audit")
    scale = farfield_stress_scale(sigma_inf)
    normalized = {
        tier: np.asarray(payload["sampled"], dtype=np.float64) / scale
        for tier, payload in solved.items()
    }
    operator = recovery_config["operator"]
    recovered = (
        recover_stress_at_queries(
            solved["coarse"]["result"].nodes,
            solved["coarse"]["result"].elements,
            solved["coarse"]["result"].total_stress,
            grid.points_yz,
            solved["coarse"]["element_ids"],
            barycentric_tolerance=float(operator["barycentric_tolerance"]),
            rank_tolerance=float(operator["rank_tolerance"]),
        )
        / scale
    )
    weights = np.asarray(grid.area_weights, dtype=np.float64)
    metrics = {
        "raw_coarse_vs_ultrafine": relative_tensor_error(
            normalized["coarse"], normalized["ultrafine"], weights
        ),
        "recovered_coarse_vs_ultrafine": relative_tensor_error(
            recovered, normalized["ultrafine"], weights
        ),
        "raw_coarse_vs_fine": relative_tensor_error(
            normalized["coarse"], normalized["fine"], weights
        ),
        "recovered_coarse_vs_fine": relative_tensor_error(recovered, normalized["fine"], weights),
        "fine_vs_ultrafine": relative_tensor_error(
            normalized["fine"], normalized["ultrafine"], weights
        ),
    }
    metrics["recovery_raw_ratio_ultrafine"] = (
        metrics["recovered_coarse_vs_ultrafine"] / metrics["raw_coarse_vs_ultrafine"]
    )
    metrics["recovery_raw_ratio_fine"] = (
        metrics["recovered_coarse_vs_fine"] / metrics["raw_coarse_vs_fine"]
    )
    wall_mask = np.asarray(grid.wall_offset_mask, dtype=bool)
    _, normals = surface_points_and_normals(geometry, int(np.sum(wall_mask)))
    arc_weights = np.asarray(grid.arc_weights[wall_mask], dtype=np.float64)
    wall: dict[str, Any] = {}
    for reference in ("fine", "ultrafine"):
        wall[reference] = {}
        for method, prediction in (
            ("raw_coarse", normalized["coarse"]),
            ("recovered", recovered),
        ):
            traction, resultant = wall_offset_discrepancy(
                prediction[wall_mask],
                normalized[reference][wall_mask],
                arc_weights,
                normals,
            )
            wall[reference][method] = {
                "traction": traction,
                "resultant": resultant,
            }
    qc = {tier: payload["qc"] for tier, payload in solved.items()}
    common_bounds = all(
        np.allclose(qc[tier]["outer_bounds"], bounds, rtol=0.0, atol=1e-12)
        for tier in ("coarse", "fine", "ultrafine")
    )
    elapsed = time.perf_counter() - started
    return {
        "case_group_id": case.case_group_id,
        "geometry_group_id": case.geometry_group_id,
        "boundary_float64_sha256": case.boundary_float64_sha256,
        "load_group_id": case.load_group_id,
        "formal_partition": case.formal_partition,
        "section_family": case.section_family,
        "parent_index": int(case.parent_index),
        "load_index": int(case.load_index),
        "load_subtype": case.load_subtype,
        "sigma1": float(case.sigma1),
        "sigma3_over_sigma1": float(case.sigma3_over_sigma1),
        "principal_angle_deg": float(case.principal_angle_deg),
        "sigma_inf_tension_positive": sigma_inf.tolist(),
        "query_hash": grid.query_hash,
        "query_point_count": int(grid.point_count),
        "seen_development_only": True,
        "metrics": metrics,
        "wall_offset": wall,
        "solver_mesh_qc": qc,
        "identity_qc": {
            "same_frozen_boundary": geometry_entry.geometry_group_id == case.geometry_group_id,
            "same_query_across_fidelities": True,
            "same_outer_bounds_across_fidelities": common_bounds,
            "passed": bool(
                geometry_entry.geometry_group_id == case.geometry_group_id and common_bounds
            ),
        },
        "case_seconds": float(elapsed),
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for distribution in ("numpy", "scipy", "scikit-fem", "gmsh", "tunnelgeopt"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision = None
        status = []
    return {
        "captured_at_utc": _now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable_name": Path(sys.executable).name,
        "packages": packages,
        "git_head_at_execution": revision,
        "git_worktree_status_at_execution": status,
        "source_file_sha256": {
            relative: _file_sha256(ROOT / relative) for relative in SOURCE_FILES
        },
    }


def run(config: Mapping[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    cases_path = output_dir / "case_metrics.jsonl"
    progress_path.write_text("", encoding="utf-8")
    cases_path.write_text("", encoding="utf-8")
    formal = _load_formal_config(config, config_path)
    plan = build_seen_v03_plan(formal)
    selected = select_cases(plan, config["selection"])
    geometry_by_id = {entry.geometry_group_id: entry for entry in plan.geometries}
    selection_manifest = {
        "schema": "tunnelgeopt.stress_recovery.development.selection.v1",
        "run_id": config["run_id"],
        "created_at_utc": _now(),
        "protocol": config["selection"]["protocol"],
        "salt_sha256": _sha256_bytes(str(config["selection"]["salt"]).encode("utf-8")),
        "source_formal_config_canonical_sha256": _sha256_bytes(_canonical_bytes(formal)),
        "full_seen_plan": {
            "parent_geometry_count": len(plan.geometries),
            "case_count": len(plan.cases),
            "formal_eligible": plan.formal_eligible,
            "all_v03_cases_declared_seen": True,
        },
        "selected_case_count": len(selected),
        "selected_cases": [
            {
                "case_group_id": case.case_group_id,
                "geometry_group_id": case.geometry_group_id,
                "formal_partition": case.formal_partition,
                "section_family": case.section_family,
                "parent_index": int(case.parent_index),
                "load_index": int(case.load_index),
            }
            for case in selected
        ],
        "selection_used_solver_or_label_values": False,
        "effect_claim_allowed": False,
    }
    _write_json_atomic(output_dir / "config_snapshot.json", dict(config))
    _write_json_atomic(output_dir / "selection_manifest.json", selection_manifest)
    _write_json_atomic(output_dir / "environment.json", _environment())
    records: list[dict[str, Any]] = []
    _append_jsonl(
        progress_path,
        {
            "event": "run_started",
            "at_utc": _now(),
            "selected_case_count": len(selected),
            "all_v03_cases_declared_seen": True,
            "effect_claim_allowed": False,
        },
    )
    for index, case in enumerate(selected):
        record = _case_record(case, geometry_by_id[case.geometry_group_id], formal, config)
        records.append(record)
        _append_jsonl(cases_path, record)
        _append_jsonl(
            progress_path,
            {
                "event": "seen_development_case_completed",
                "at_utc": _now(),
                "completed_case_count": len(records),
                "total_case_count": len(selected),
                "case_group_id": case.case_group_id,
                "formal_partition": case.formal_partition,
                "section_family": case.section_family,
                "case_seconds": record["case_seconds"],
                "raw_coarse_vs_ultrafine": record["metrics"]["raw_coarse_vs_ultrafine"],
                "recovered_coarse_vs_ultrafine": record["metrics"]["recovered_coarse_vs_ultrafine"],
                "recovery_raw_ratio_ultrafine": record["metrics"]["recovery_raw_ratio_ultrafine"],
                "selection_index": index,
            },
        )
    summary = aggregate_records(records, config)
    summary["runtime"]["wall_clock_seconds"] = float(time.perf_counter() - started)
    summary["completed_at_utc"] = _now()
    _write_json_atomic(output_dir / "summary.json", summary)
    _append_jsonl(
        progress_path,
        {
            "event": "run_completed",
            "at_utc": _now(),
            "case_count": len(records),
            "development_routing": summary["development_routing"],
            "effect_claim_allowed": False,
        },
    )
    artifact_hashes = {
        path.name: _file_sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json_atomic(
        output_dir / "artifact_manifest.json",
        {
            "schema": "tunnelgeopt.stress_recovery.development.artifacts.v1",
            "run_id": config["run_id"],
            "created_at_utc": _now(),
            "effect_claim_allowed": False,
            "files_sha256": artifact_hashes,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the frozen config and reconstruct/select the 705-case plan without solving",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.validate_only:
        formal = _load_formal_config(config, config_path)
        plan = build_seen_v03_plan(formal)
        selected = select_cases(plan, config["selection"])
        print(
            json.dumps(
                {
                    "status": "valid",
                    "all_v03_cases_declared_seen": True,
                    "effect_claim_allowed": False,
                    "full_plan_case_count": len(plan.cases),
                    "selected_case_count": len(selected),
                    "selected_case_ids": [case.case_group_id for case in selected],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    summary = run(config, config_path, args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
