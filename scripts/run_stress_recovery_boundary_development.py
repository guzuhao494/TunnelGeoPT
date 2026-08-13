#!/usr/bin/env python3
"""Run the post-v0.5 traction-preserving recovery redesign on seen cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from tunnelgeopt.geometry import surface_points_and_normals
from tunnelgeopt.multifidelity import farfield_stress_scale
from tunnelgeopt.stress_recovery import (
    preserve_baseline_traction_with_tangential_correction,
    recover_stress_at_queries,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "stress_recovery_boundary_development.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "development" / "stress-recovery-boundary-v0.5.1-dev"
EXPECTED_SECTIONS = ("circle", "horseshoe", "straight_wall_arch")
EXPECTED_PARTITIONS = (
    "train_id",
    "dev_id",
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
)
NEW_SOURCE_FILES = (
    "configs/stress_recovery_boundary_development.json",
    "scripts/run_stress_recovery_boundary_development.py",
    "tests/test_stress_recovery_boundary_development.py",
)


class BoundaryDevelopmentError(RuntimeError):
    """Raised when the v0.5.1 development-only contract is violated."""


def _load_base_runner() -> ModuleType:
    path = ROOT / "scripts" / "run_stress_recovery_development.py"
    specification = importlib.util.spec_from_file_location(
        "tunnelgeopt_v05_stress_recovery_runner", path
    )
    if specification is None or specification.loader is None:
        raise BoundaryDevelopmentError("could not load the v0.5 development runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


BASE = _load_base_runner()


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
        raise BoundaryDevelopmentError(
            f"could not load boundary-development config: {exc}"
        ) from exc
    required = {
        "schema_version",
        "config_name",
        "run_id",
        "status",
        "scope",
        "redesign_provenance",
        "claim_exclusions",
        "source",
        "selection",
        "operator",
        "metrics",
        "quality_control",
        "development_routing_gates",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise BoundaryDevelopmentError("boundary-development config key set changed")
    if config["schema_version"] != "tunnelgeopt.stress_recovery.boundary_development.v1":
        raise BoundaryDevelopmentError("unsupported boundary-development schema")
    if config["status"] != "frozen_post_v05_failure_development_only_before_execution":
        raise BoundaryDevelopmentError("v0.5.1 must remain frozen post-failure development")
    if config["scope"] != (
        "development_only_traction_preserving_recovery_redesign_on_same_seen_v03_cases"
    ):
        raise BoundaryDevelopmentError("v0.5.1 scope must remain development-only")
    provenance = config["redesign_provenance"]
    if (
        provenance.get("developed_after_observing_predecessor_results") is not True
        or provenance.get("confirmatory_status") != "not_confirmatory_post_hoc_development_redesign"
        or provenance.get("new_case_or_label_independence_claim_allowed") is not False
    ):
        raise BoundaryDevelopmentError("post-v0.5 redesign provenance must be explicit")
    source = config["source"]
    if (
        source.get("all_v03_cases_declared_seen") is not True
        or source.get("fine_and_ultrafine_labels_are_development_only") is not True
        or source.get("effect_claim_allowed") is not False
    ):
        raise BoundaryDevelopmentError("all v0.3 identities must remain seen/development-only")
    selection = config["selection"]
    if (
        tuple(selection.get("section_families", ())) != EXPECTED_SECTIONS
        or tuple(selection.get("partitions", ())) != EXPECTED_PARTITIONS
        or int(selection.get("cases_per_partition_section", 0)) != 1
        or int(selection.get("expected_case_count", 0)) != 15
        or selection.get("require_exact_predecessor_case_ids") is not True
        or selection.get("selection_must_not_use_solver_or_label_values") is not True
    ):
        raise BoundaryDevelopmentError("the same frozen 15-case selection must be retained")
    operator = config["operator"]
    if (
        operator.get("parameter_status")
        != "frozen_post_v05_structural_redesign_no_v051_result_tuning"
        or operator.get("linearity_in_raw_and_recovered_stress_required") is not True
        or float(operator.get("maximum_traction_increment_norm", -1.0)) != 1e-12
    ):
        raise BoundaryDevelopmentError("traction-preserving operator contract changed")
    gates = config["development_routing_gates"]
    if (
        gates.get("passing_route") != "READY_FOR_NEW_CONFIRMATORY_PREREGISTRATION"
        or gates.get("effect_claim_allowed") is not False
    ):
        raise BoundaryDevelopmentError("routing may not authorize an effect claim")
    return config


def _tree_hashes(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise BoundaryDevelopmentError(f"required predecessor artifact is missing: {path.name}")
    return {
        file.relative_to(path).as_posix(): _file_sha256(file)
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _predecessor(
    config: Mapping[str, Any], selected_case_ids: Sequence[str]
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    relative = Path(str(config["redesign_provenance"]["predecessor_artifact"]))
    path = relative if relative.is_absolute() else ROOT / relative
    before_hashes = _tree_hashes(path)
    try:
        selection = json.loads((path / "selection_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryDevelopmentError(f"could not verify predecessor artifact: {exc}") from exc
    predecessor_ids = [row["case_group_id"] for row in selection["selected_cases"]]
    if list(selected_case_ids) != predecessor_ids:
        raise BoundaryDevelopmentError("v0.5.1 cases differ from the v0.5 predecessor")
    if (
        summary.get("run_id") != config["redesign_provenance"]["predecessor_run_id"]
        or summary.get("development_routing")
        != "STOP_OR_REDESIGN_BEFORE_ANY_NEW_UNSEEN_CONFIRMATORY_RUN"
        or summary.get("effect_claim_allowed") is not False
    ):
        raise BoundaryDevelopmentError("predecessor is not the frozen v0.5 STOP result")
    evidence = {
        "artifact_tree_sha256": before_hashes,
        "selection_manifest_sha256": _file_sha256(path / "selection_manifest.json"),
        "summary_sha256": _file_sha256(path / "summary.json"),
        "observed_ultrafine_wall_center_ratios": summary["wall_offset_center_ratios"]["ultrafine"],
    }
    return path, before_hashes, evidence


def apply_boundary_preserving_recovery(
    raw_coarse: np.ndarray,
    unconstrained_recovered: np.ndarray,
    wall_mask: np.ndarray,
    wall_normals_yz: np.ndarray,
    *,
    normal_tolerance: float,
) -> np.ndarray:
    """Use recovery off-wall and its traction-preserving projection on-wall."""

    raw = np.asarray(raw_coarse, dtype=np.float64)
    recovered = np.asarray(unconstrained_recovered, dtype=np.float64)
    mask = np.asarray(wall_mask, dtype=bool)
    normals = np.asarray(wall_normals_yz, dtype=np.float64)
    if raw.shape != recovered.shape or raw.ndim != 2 or raw.shape[1] != 3:
        raise BoundaryDevelopmentError("raw and recovered stresses must align as [P,3]")
    if mask.shape != (raw.shape[0],) or normals.shape != (int(np.sum(mask)), 2):
        raise BoundaryDevelopmentError("wall mask and normals do not align")
    output = recovered.copy()
    output[mask] = preserve_baseline_traction_with_tangential_correction(
        raw[mask],
        recovered[mask],
        normals,
        normal_tolerance=float(normal_tolerance),
    )
    return output


def _traction_increment_norms(
    candidate: np.ndarray,
    baseline: np.ndarray,
    normals_yz: np.ndarray,
) -> np.ndarray:
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    normals = np.asarray(normals_yz, dtype=np.float64)
    tensor = np.empty((difference.shape[0], 2, 2), dtype=np.float64)
    tensor[:, 0, 0] = difference[:, 0]
    tensor[:, 1, 1] = difference[:, 1]
    tensor[:, 0, 1] = difference[:, 2]
    tensor[:, 1, 0] = difference[:, 2]
    return np.linalg.norm(np.einsum("pij,pj->pi", tensor, normals), axis=1)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise BoundaryDevelopmentError("cannot summarize empty or non-finite values")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _center_ratio(records: Sequence[Mapping[str, Any]], numerator: str, denominator: str) -> float:
    top = np.asarray([record["nearfield"][numerator] for record in records], dtype=np.float64)
    bottom = np.asarray([record["nearfield"][denominator] for record in records], dtype=np.float64)
    if float(np.mean(bottom)) <= np.finfo(float).tiny:
        raise BoundaryDevelopmentError("nearfield center denominator is zero")
    return float(np.mean(top) / np.mean(bottom))


def _wall_center_ratio(
    records: Sequence[Mapping[str, Any]],
    reference: str,
    method: str,
    diagnostic: str,
) -> float:
    top = np.asarray(
        [record["wall_offset"][reference][method][diagnostic] for record in records],
        dtype=np.float64,
    )
    raw = np.asarray(
        [record["wall_offset"][reference]["raw_coarse"][diagnostic] for record in records],
        dtype=np.float64,
    )
    if float(np.mean(raw)) <= np.finfo(float).tiny:
        raise BoundaryDevelopmentError("wall center denominator is zero")
    return float(np.mean(top) / np.mean(raw))


def _case_record(
    case: Any,
    geometry_entry: Any,
    formal: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    geometry, grid = BASE._build_grid(geometry_entry, formal)
    bounds = BASE._outer_bounds(geometry, grid, float(geometry_entry.spec.outer_domain_scale))
    sigma_inf = np.asarray(case.sigma_inf_tension_positive, dtype=np.float64)
    solved = {
        tier: BASE._solve_tier(
            geometry,
            grid,
            sigma_inf,
            BASE._mesh_spec(formal, source_tier),
            formal,
            bounds,
        )
        for tier, source_tier in (
            ("coarse", "coarse"),
            ("fine", "fine"),
            ("ultrafine", "ultrafine_audit"),
        )
    }
    scale = farfield_stress_scale(sigma_inf)
    normalized = {
        tier: np.asarray(payload["sampled"], dtype=np.float64) / scale
        for tier, payload in solved.items()
    }
    operator = config["operator"]
    unconstrained = (
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
    wall_mask = np.asarray(grid.wall_offset_mask, dtype=bool)
    _, wall_normals = surface_points_and_normals(geometry, int(np.sum(wall_mask)))
    candidate = apply_boundary_preserving_recovery(
        normalized["coarse"],
        unconstrained,
        wall_mask,
        wall_normals,
        normal_tolerance=float(operator["normal_tolerance"]),
    )
    near_weights = np.asarray(grid.area_weights, dtype=np.float64)
    nearfield = {
        "raw_coarse_vs_ultrafine": BASE.relative_tensor_error(
            normalized["coarse"], normalized["ultrafine"], near_weights
        ),
        "unconstrained_vs_ultrafine": BASE.relative_tensor_error(
            unconstrained, normalized["ultrafine"], near_weights
        ),
        "boundary_preserving_vs_ultrafine": BASE.relative_tensor_error(
            candidate, normalized["ultrafine"], near_weights
        ),
        "raw_coarse_vs_fine": BASE.relative_tensor_error(
            normalized["coarse"], normalized["fine"], near_weights
        ),
        "unconstrained_vs_fine": BASE.relative_tensor_error(
            unconstrained, normalized["fine"], near_weights
        ),
        "boundary_preserving_vs_fine": BASE.relative_tensor_error(
            candidate, normalized["fine"], near_weights
        ),
        "fine_vs_ultrafine": BASE.relative_tensor_error(
            normalized["fine"], normalized["ultrafine"], near_weights
        ),
    }
    nearfield["boundary_raw_ratio_ultrafine"] = (
        nearfield["boundary_preserving_vs_ultrafine"] / nearfield["raw_coarse_vs_ultrafine"]
    )
    nearfield_identity_error = max(
        abs(
            nearfield["boundary_preserving_vs_ultrafine"] - nearfield["unconstrained_vs_ultrafine"]
        ),
        abs(nearfield["boundary_preserving_vs_fine"] - nearfield["unconstrained_vs_fine"]),
    )
    arc_weights = np.asarray(grid.arc_weights[wall_mask], dtype=np.float64)
    methods = {
        "raw_coarse": normalized["coarse"],
        "unconstrained_recovery": unconstrained,
        "boundary_preserving": candidate,
    }
    wall: dict[str, Any] = {}
    for reference in ("fine", "ultrafine"):
        wall[reference] = {}
        for name, prediction in methods.items():
            traction, resultant = BASE.wall_offset_discrepancy(
                prediction[wall_mask],
                normalized[reference][wall_mask],
                arc_weights,
                wall_normals,
            )
            wall[reference][name] = {
                "traction": traction,
                "resultant": resultant,
                "full_stress_relative_l2": BASE.relative_tensor_error(
                    prediction[wall_mask],
                    normalized[reference][wall_mask],
                    arc_weights,
                ),
            }
    traction_increment = _traction_increment_norms(
        candidate[wall_mask], normalized["coarse"][wall_mask], wall_normals
    )
    qc = {tier: payload["qc"] for tier, payload in solved.items()}
    common_bounds = all(
        np.allclose(qc[tier]["outer_bounds"], bounds, rtol=0.0, atol=1e-12)
        for tier in ("coarse", "fine", "ultrafine")
    )
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
        "query_hash": grid.query_hash,
        "query_point_count": int(grid.point_count),
        "seen_development_only": True,
        "post_v05_redesign": True,
        "nearfield": nearfield,
        "wall_offset": wall,
        "projection_contract": {
            "maximum_traction_increment_norm": float(np.max(traction_increment)),
            "weighted_rms_traction_increment_norm": float(
                np.sqrt(np.sum(arc_weights * traction_increment**2))
            ),
            "nearfield_metric_identity_error_vs_unconstrained": float(nearfield_identity_error),
        },
        "solver_mesh_qc": qc,
        "identity_qc": {
            "same_frozen_boundary": geometry_entry.geometry_group_id == case.geometry_group_id,
            "same_query_across_fidelities": True,
            "same_outer_bounds_across_fidelities": common_bounds,
            "passed": bool(
                geometry_entry.geometry_group_id == case.geometry_group_id and common_bounds
            ),
        },
        "case_seconds": float(time.perf_counter() - started),
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    if len(records) != int(config["selection"]["expected_case_count"]):
        raise BoundaryDevelopmentError("aggregate requires all 15 frozen cases")
    raw = [record["nearfield"]["raw_coarse_vs_ultrafine"] for record in records]
    candidate = [record["nearfield"]["boundary_preserving_vs_ultrafine"] for record in records]
    ratios = [record["nearfield"]["boundary_raw_ratio_ultrafine"] for record in records]
    overall_ratio = float(np.mean(candidate) / np.mean(raw))
    sections: dict[str, Any] = {}
    for section in config["selection"]["section_families"]:
        subset = [record for record in records if record["section_family"] == section]
        sections[section] = {
            "case_count": len(subset),
            "ultrafine_nearfield_center_ratio": _center_ratio(
                subset,
                "boundary_preserving_vs_ultrafine",
                "raw_coarse_vs_ultrafine",
            ),
            "fine_nearfield_center_ratio": _center_ratio(
                subset,
                "boundary_preserving_vs_fine",
                "raw_coarse_vs_fine",
            ),
        }
    wall_ratios = {
        reference: {
            method: {
                diagnostic: _wall_center_ratio(records, reference, method, diagnostic)
                for diagnostic in ("traction", "resultant", "full_stress_relative_l2")
            }
            for method in ("unconstrained_recovery", "boundary_preserving")
        }
        for reference in ("fine", "ultrafine")
    }
    all_qc = [
        record["solver_mesh_qc"][tier]
        for record in records
        for tier in ("coarse", "fine", "ultrafine")
    ]
    quality = config["quality_control"]
    qc_checks = {
        "complete_case_count": len(records) == 15,
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
    gates = config["development_routing_gates"]
    tolerance = float(gates["ratio_numerical_tolerance"])
    required_references = tuple(gates["required_wall_references"])
    wall_nonworsening_checks = {
        f"{reference}_{diagnostic}": wall_ratios[reference]["boundary_preserving"][diagnostic]
        <= float(gates["wall_diagnostic_center_ratio_max"]) + tolerance
        for reference in required_references
        for diagnostic in ("traction", "resultant", "full_stress_relative_l2")
    }
    max_increment = float(
        max(record["projection_contract"]["maximum_traction_increment_norm"] for record in records)
    )
    max_nearfield_identity_error = float(
        max(
            record["projection_contract"]["nearfield_metric_identity_error_vs_unconstrained"]
            for record in records
        )
    )
    routing_checks = {
        "solver_mesh_qc": all(qc_checks.values()),
        "overall_ultrafine_nearfield_center_ratio": overall_ratio
        <= float(gates["overall_ultrafine_nearfield_center_ratio_max"]),
        "each_section_ultrafine_nearfield_center_ratio": all(
            row["ultrafine_nearfield_center_ratio"]
            <= float(gates["each_section_ultrafine_nearfield_center_ratio_max"])
            for row in sections.values()
        ),
        "all_case_traction_increment_norm": max_increment
        <= float(gates["all_case_traction_increment_norm_max"]),
        "nearfield_benefit_identical_to_unconstrained": max_nearfield_identity_error <= 1e-15,
        "wall_nonworsening": all(wall_nonworsening_checks.values()),
        "effect_claim_allowed": False,
    }
    positive = {
        key: value for key, value in routing_checks.items() if key != "effect_claim_allowed"
    }
    return {
        "schema": "tunnelgeopt.stress_recovery.boundary_development.summary.v1",
        "run_id": config["run_id"],
        "scope": config["scope"],
        "developed_after_observing_v05_failure": True,
        "all_v03_cases_declared_seen": True,
        "confirmatory_status": "not_confirmatory_post_hoc_development_redesign",
        "effect_claim_allowed": False,
        "case_count": len(records),
        "nearfield_against_ultrafine": {
            "raw_coarse_error": _distribution(raw),
            "boundary_preserving_error": _distribution(candidate),
            "center_ratio_boundary_over_raw": overall_ratio,
            "case_ratio_distribution": _distribution(ratios),
        },
        "nearfield_against_fine": {
            "center_ratio_boundary_over_raw": _center_ratio(
                records, "boundary_preserving_vs_fine", "raw_coarse_vs_fine"
            )
        },
        "by_section": sections,
        "wall_offset_center_ratios_over_raw": wall_ratios,
        "projection_contract": {
            "maximum_traction_increment_norm": max_increment,
            "maximum_nearfield_metric_identity_error_vs_unconstrained": (
                max_nearfield_identity_error
            ),
            "required_tolerance": float(gates["all_case_traction_increment_norm_max"]),
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
        "wall_nonworsening_checks": wall_nonworsening_checks,
        "routing_checks": routing_checks,
        "development_routing": (
            gates["passing_route"] if all(positive.values()) else gates["failing_route"]
        ),
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
        "claim_boundary": (
            "Post-v0.5 development redesign on exactly the same 15 seen synthetic cases. "
            "READY means only that a new, separately preregistered unseen confirmation is "
            "scientifically worth considering; it is not an effect or generalization claim."
        ),
    }


def _environment() -> dict[str, Any]:
    environment = BASE._environment()
    hashes = dict(environment["source_file_sha256"])
    for relative in NEW_SOURCE_FILES:
        hashes[relative] = _file_sha256(ROOT / relative)
    environment["source_file_sha256"] = hashes
    environment["campaign"] = "post_v05_boundary_development_only"
    return environment


def run(config: Mapping[str, Any], config_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    formal = BASE._load_formal_config(config, config_path)
    plan = BASE.build_seen_v03_plan(formal)
    selected = BASE.select_cases(plan, config["selection"])
    selected_ids = [case.case_group_id for case in selected]
    predecessor_path, predecessor_hashes, predecessor_evidence = _predecessor(config, selected_ids)
    geometry_by_id = {entry.geometry_group_id: entry for entry in plan.geometries}
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    case_path = output_dir / "case_metrics.jsonl"
    progress_path.write_text("", encoding="utf-8")
    case_path.write_text("", encoding="utf-8")
    _write_json_atomic(output_dir / "config_snapshot.json", dict(config))
    _write_json_atomic(
        output_dir / "selection_manifest.json",
        {
            "schema": "tunnelgeopt.stress_recovery.boundary_development.selection.v1",
            "run_id": config["run_id"],
            "created_at_utc": _now(),
            "selected_case_count": len(selected),
            "selected_case_ids": selected_ids,
            "exact_predecessor_case_ids": True,
            "all_v03_cases_declared_seen": True,
            "selection_used_solver_or_label_values": False,
            "predecessor_evidence": predecessor_evidence,
            "effect_claim_allowed": False,
        },
    )
    _write_json_atomic(output_dir / "environment.json", _environment())
    _append_jsonl(
        progress_path,
        {
            "event": "post_v05_seen_redesign_started",
            "at_utc": _now(),
            "selected_case_count": len(selected),
            "effect_claim_allowed": False,
        },
    )
    records: list[dict[str, Any]] = []
    for index, case in enumerate(selected):
        record = _case_record(case, geometry_by_id[case.geometry_group_id], formal, config)
        records.append(record)
        _append_jsonl(case_path, record)
        _append_jsonl(
            progress_path,
            {
                "event": "boundary_development_case_completed",
                "at_utc": _now(),
                "selection_index": index,
                "completed_case_count": len(records),
                "total_case_count": len(selected),
                "case_group_id": case.case_group_id,
                "formal_partition": case.formal_partition,
                "section_family": case.section_family,
                "nearfield_ratio_ultrafine": record["nearfield"]["boundary_raw_ratio_ultrafine"],
                "maximum_traction_increment_norm": record["projection_contract"][
                    "maximum_traction_increment_norm"
                ],
                "case_seconds": record["case_seconds"],
            },
        )
    after_hashes = _tree_hashes(predecessor_path)
    if after_hashes != predecessor_hashes:
        raise BoundaryDevelopmentError("v0.5 predecessor artifact changed during v0.5.1")
    summary = aggregate_records(records, config)
    summary["predecessor_artifact_unchanged"] = True
    summary["runtime"]["wall_clock_seconds"] = float(time.perf_counter() - started)
    summary["completed_at_utc"] = _now()
    _write_json_atomic(output_dir / "summary.json", summary)
    _append_jsonl(
        progress_path,
        {
            "event": "boundary_development_run_completed",
            "at_utc": _now(),
            "case_count": len(records),
            "development_routing": summary["development_routing"],
            "predecessor_artifact_unchanged": True,
            "effect_claim_allowed": False,
        },
    )
    hashes = {
        path.name: _file_sha256(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    _write_json_atomic(
        output_dir / "artifact_manifest.json",
        {
            "schema": "tunnelgeopt.stress_recovery.boundary_development.artifacts.v1",
            "run_id": config["run_id"],
            "created_at_utc": _now(),
            "effect_claim_allowed": False,
            "files_sha256": hashes,
        },
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    formal = BASE._load_formal_config(config, config_path)
    plan = BASE.build_seen_v03_plan(formal)
    selected = BASE.select_cases(plan, config["selection"])
    selected_ids = [case.case_group_id for case in selected]
    if args.validate_only:
        _, _, predecessor = _predecessor(config, selected_ids)
        print(
            json.dumps(
                {
                    "status": "valid",
                    "post_v05_redesign": True,
                    "confirmatory_status": "not_confirmatory",
                    "effect_claim_allowed": False,
                    "full_seen_plan_case_count": len(plan.cases),
                    "selected_case_count": len(selected),
                    "exact_predecessor_case_ids": True,
                    "predecessor_summary_sha256": predecessor["summary_sha256"],
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
