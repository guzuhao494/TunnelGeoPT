"""Verify the nine-channel load basis on already-seen v0.3 elastic labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tunnelgeopt.load_basis import (
    LoadBasisError,
    fit_linear_stress_response_basis,
)

DEFAULT_CONFIG = ROOT / "configs" / "load_basis_development.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "development" / "linear-load-basis-v0.5-development"


class LoadBasisDevelopmentError(RuntimeError):
    """Raised when seen-data authentication or analysis fails closed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)
    return _file_sha256(path)


def _load_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LoadBasisDevelopmentError(f"cannot read development config: {exc}") from exc
    if value.get("status") != "development_seen_data_only":
        raise LoadBasisDevelopmentError("config must remain development_seen_data_only")
    if value.get("effect_claim_allowed") is not False:
        raise LoadBasisDevelopmentError("development config may not authorize an effect claim")
    protocol = value.get("protocol", {})
    if protocol.get("all_v03_partitions_are_seen") is not True:
        raise LoadBasisDevelopmentError("all v0.3 partitions must be marked seen")
    if protocol.get("new_solver_calls") != 0 or protocol.get("new_locked_cases") != 0:
        raise LoadBasisDevelopmentError(
            "load-basis development may not create solver or locked data"
        )
    return value, _value_sha256(value)


def _authenticate_sources(config: dict[str, Any]) -> tuple[Path, dict[str, str]]:
    source_root = (ROOT / str(config["source_root"])).resolve()
    hashes: dict[str, str] = {}
    for relative, expected in config["source_hashes"].items():
        path = source_root / relative
        if not path.is_file():
            raise LoadBasisDevelopmentError(f"required seen-data source is missing: {relative}")
        actual = _file_sha256(path)
        if actual != expected:
            raise LoadBasisDevelopmentError(f"seen-data source hash mismatch: {relative}")
        hashes[str(relative)] = actual
    return source_root, hashes


def _load_seen_arrays(source_root: Path) -> dict[str, np.ndarray]:
    public_path = source_root / "data" / "public_inputs_and_coarse_fields.npz"
    with np.load(public_path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    case_count = int(data["case_group_ids"].shape[0])
    fine = np.full((case_count, data["base_features"].shape[1], 3), np.nan, dtype=np.float64)
    occupied = np.zeros(case_count, dtype=bool)
    label_paths = [source_root / "data" / "train_dev_fine_labels.npz"] + sorted(
        (source_root / "data" / ".sealed_generator_store").glob("*.npz")
    )
    for path in label_paths:
        with np.load(path, allow_pickle=False) as archive:
            indices = np.asarray(archive["indices"], dtype=np.int64)
            labels = np.asarray(archive["fine_stress"], dtype=np.float64)
            case_ids = np.asarray(archive["case_group_ids"]).astype(str)
        if labels.shape != (indices.size, fine.shape[1], 3):
            raise LoadBasisDevelopmentError(f"fine-label shape mismatch in {path.name}")
        if np.any(indices < 0) or np.any(indices >= case_count) or np.any(occupied[indices]):
            raise LoadBasisDevelopmentError(
                f"fine-label indices overlap or escape range in {path.name}"
            )
        if not np.array_equal(case_ids, np.asarray(data["case_group_ids"])[indices].astype(str)):
            raise LoadBasisDevelopmentError(f"fine-label case identities disagree in {path.name}")
        fine[indices] = labels
        occupied[indices] = True
    if not np.all(occupied) or not np.isfinite(fine).all():
        raise LoadBasisDevelopmentError(
            "seen fine labels do not cover all public cases exactly once"
        )
    data["fine_stress"] = fine
    return data


def _case_error(prediction: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 3:
        raise LoadBasisDevelopmentError("case stress arrays must have shape [P, 3]")
    weight = np.asarray(weights, dtype=np.float64)
    if weight.shape != (prediction.shape[0],) or np.any(weight < 0.0) or not np.any(weight > 0.0):
        raise LoadBasisDevelopmentError("metric weights must be nonnegative with positive mass")
    error = prediction - target
    numerator = np.sum(weight * (error[:, 0] ** 2 + error[:, 1] ** 2 + 2.0 * error[:, 2] ** 2))
    denominator = np.sum(weight * (target[:, 0] ** 2 + target[:, 1] ** 2 + 2.0 * target[:, 2] ** 2))
    if denominator <= np.finfo(float).tiny:
        raise LoadBasisDevelopmentError("case target has zero weighted tensor norm")
    return float(np.sqrt(numerator / denominator))


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise LoadBasisDevelopmentError("cannot summarize an empty or non-finite metric")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def analyze_leave_one_load_out(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fit three loads and predict the fourth within each eligible seen geometry."""

    required = {
        "base_features",
        "coarse_stress",
        "fine_stress",
        "metric_weights",
        "case_group_ids",
        "geometry_group_ids",
        "query_hashes",
        "partitions",
        "section_families",
    }
    if not required.issubset(data):
        raise LoadBasisDevelopmentError("analysis input field set is incomplete")
    case_count = int(np.asarray(data["case_group_ids"]).shape[0])
    if any(np.asarray(data[name]).shape[0] != case_count for name in required):
        raise LoadBasisDevelopmentError("analysis inputs disagree on case count")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, geometry_id in enumerate(np.asarray(data["geometry_group_ids"]).astype(str)):
        groups[geometry_id].append(index)

    per_case: list[dict[str, Any]] = []
    skipped_group_counts: dict[str, int] = defaultdict(int)
    for geometry_id, indices_list in sorted(groups.items()):
        indices = np.asarray(indices_list, dtype=np.int64)
        if indices.size != 4:
            skipped_group_counts[f"load_count_{indices.size}"] += 1
            continue
        query_hashes = np.asarray(data["query_hashes"])[indices].astype(str)
        if np.unique(query_hashes).size != 1:
            raise LoadBasisDevelopmentError("one geometry group contains different query hashes")
        partitions = np.asarray(data["partitions"])[indices].astype(str)
        sections = np.asarray(data["section_families"])[indices].astype(str)
        if np.unique(partitions).size != 1 or np.unique(sections).size != 1:
            raise LoadBasisDevelopmentError("one geometry group crosses partition or section")
        loads = np.asarray(data["base_features"])[indices, 0, 7:10].astype(np.float64)
        fine = np.asarray(data["fine_stress"])[indices].astype(np.float64)
        for local_holdout, case_index in enumerate(indices):
            fit_mask = np.arange(4) != local_holdout
            try:
                basis = fit_linear_stress_response_basis(loads[fit_mask], fine[fit_mask])
            except LoadBasisError as exc:
                raise LoadBasisDevelopmentError(
                    f"rank-three fit failed for geometry {geometry_id}"
                ) from exc
            prediction = basis.predict(loads[local_holdout])[0]
            weights = np.asarray(data["metric_weights"])[case_index].astype(np.float64)
            basis_error = _case_error(prediction, fine[local_holdout], weights)
            coarse_error = _case_error(
                np.asarray(data["coarse_stress"])[case_index].astype(np.float64),
                fine[local_holdout],
                weights,
            )
            per_case.append(
                {
                    "case_group_id": str(np.asarray(data["case_group_ids"])[case_index]),
                    "geometry_group_id": geometry_id,
                    "partition": str(partitions[0]),
                    "section_family": str(sections[0]),
                    "held_out_local_load_index": int(local_holdout),
                    "fit_load_condition_number": basis.load_condition_number,
                    "fit_relative_residual": basis.relative_fit_residual,
                    "basis_relative_error": basis_error,
                    "raw_coarse_relative_error": coarse_error,
                    "basis_to_raw_coarse_ratio": basis_error / coarse_error,
                }
            )
    if not per_case:
        raise LoadBasisDevelopmentError("no four-load geometry group is eligible")

    def slice_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        basis_values = [float(row["basis_relative_error"]) for row in rows]
        coarse_values = [float(row["raw_coarse_relative_error"]) for row in rows]
        ratios = [float(row["basis_to_raw_coarse_ratio"]) for row in rows]
        conditions = [float(row["fit_load_condition_number"]) for row in rows]
        residuals = [float(row["fit_relative_residual"]) for row in rows]
        return {
            "basis_relative_error": _stats(basis_values),
            "raw_coarse_relative_error": _stats(coarse_values),
            "basis_to_raw_coarse_ratio": _stats(ratios),
            "fit_load_condition_number": _stats(conditions),
            "fit_relative_residual": _stats(residuals),
            "ratio_of_mean_errors": float(np.mean(basis_values) / np.mean(coarse_values)),
        }

    partitions = sorted({str(row["partition"]) for row in per_case})
    sections = sorted({str(row["section_family"]) for row in per_case})
    parent_count = len({str(row["geometry_group_id"]) for row in per_case})
    return {
        "schema": "tunnelgeopt.load_basis_leave_one_load_out.v1",
        "evidence_scope": "development_only_all_v03_labels_seen",
        "effect_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "evaluated_parent_geometries": parent_count,
        "evaluated_cases": len(per_case),
        "skipped_parent_groups": dict(sorted(skipped_group_counts.items())),
        "overall": slice_stats(per_case),
        "by_partition": {
            partition: slice_stats([row for row in per_case if row["partition"] == partition])
            for partition in partitions
        },
        "by_section": {
            section: slice_stats([row for row in per_case if row["section_family"] == section])
            for section in sections
        },
        "per_case": per_case,
    }


def analyze_canonical_basis_plan(data: dict[str, np.ndarray]) -> dict[str, Any]:
    """Precompute the numerically stable three-load basis production plan."""

    conditions = np.asarray(data["base_features"], dtype=np.float64)[:, 0, 7:10]
    if conditions.ndim != 2 or conditions.shape[1] != 3 or not np.isfinite(conditions).all():
        raise LoadBasisDevelopmentError("normalized load conditions must have shape [K, 3]")
    # The dataset normalizes a symmetric tensor with
    # sqrt(yy**2 + zz**2 + 2*yz**2).  A unit-norm pure-shear load therefore
    # has yz=1/sqrt(2), not yz=1.
    canonical_loads = np.diag([1.0, 1.0, 1.0 / np.sqrt(2.0)]).astype(np.float64)
    condition = float(np.linalg.cond(canonical_loads))
    tensor_norms = np.sqrt(
        conditions[:, 0] ** 2 + conditions[:, 1] ** 2 + 2.0 * conditions[:, 2] ** 2
    )
    return {
        "schema": "tunnelgeopt.canonical_linear_load_basis_plan.v1",
        "basis_load_vectors_normalized": canonical_loads.tolist(),
        "basis_load_semantics": [
            "unit_tensor_norm_sigma_yy",
            "unit_tensor_norm_sigma_zz",
            "unit_tensor_norm_pure_tau_yz",
        ],
        "load_rank": int(np.linalg.matrix_rank(canonical_loads)),
        "load_condition_number": condition,
        "observed_tensor_frobenius_load_norm": _stats([float(row) for row in tensor_norms]),
        "rule": (
            "solve exactly these three unit-tensor-norm canonical loads on one fixed "
            "geometry/mesh/query; "
            "store the three-by-three response tensor per point; compose all later loads linearly"
        ),
        "uses_seen_fine_labels_to_choose_basis": False,
        "claim_boundary": "homogeneous fixed-geometry small-strain linear elasticity only",
    }


def _decision(analysis: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    thresholds = config["development_verification_thresholds"]
    overall = analysis["overall"]
    checks = {
        "parent_count": analysis["evaluated_parent_geometries"]
        >= int(thresholds["minimum_evaluated_parent_geometries"]),
        "case_count": analysis["evaluated_cases"] >= int(thresholds["minimum_evaluated_cases"]),
        "median_error": overall["basis_relative_error"]["median"]
        <= float(thresholds["maximum_median_relative_error"]),
        "p95_error": overall["basis_relative_error"]["p95"]
        <= float(thresholds["maximum_p95_relative_error"]),
        "maximum_error": overall["basis_relative_error"]["maximum"]
        <= float(thresholds["maximum_case_relative_error"]),
        "mean_ratio_to_coarse": overall["ratio_of_mean_errors"]
        <= float(thresholds["maximum_mean_ratio_to_raw_coarse"]),
    }
    passed = all(checks.values())
    return {
        "schema": "tunnelgeopt.load_basis_development_decision.v1",
        "classification": (
            "DEVELOPMENT_LINEAR_LOAD_BASIS_VERIFIED"
            if passed
            else "DEVELOPMENT_LINEAR_LOAD_BASIS_NOT_VERIFIED"
        ),
        "passed": passed,
        "checks": checks,
        "effect_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "scientific_interpretation": (
            "three independent load responses identify the nine-channel response basis "
            "for the current fixed-geometry linear-elastic layer"
            if passed
            else "the current seen-data load basis did not meet its development checks"
        ),
        "claim_boundary": list(config["claim_boundary"]),
    }


def execute(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config, config_hash = _load_config(config_path)
    source_root, authenticated = _authenticate_sources(config)
    data = _load_seen_arrays(source_root)
    analysis = analyze_leave_one_load_out(data)
    canonical_plan = analyze_canonical_basis_plan(data)
    decision = _decision(analysis, config)
    output = output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise LoadBasisDevelopmentError("output directory must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema": "tunnelgeopt.load_basis_development_provenance.v1",
        "run_id": config["run_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_hash,
        "source_run": config["source_run"],
        "authenticated_source_hashes": authenticated,
        "new_solver_calls": 0,
        "new_locked_cases": 0,
        "all_source_labels_seen": True,
    }
    hashes = {
        "config_snapshot.json": _atomic_json(output / "config_snapshot.json", config),
        "provenance.json": _atomic_json(output / "provenance.json", provenance),
        "leave_one_load_out_metrics.json": _atomic_json(
            output / "leave_one_load_out_metrics.json", analysis
        ),
        "canonical_basis_plan.json": _atomic_json(
            output / "canonical_basis_plan.json", canonical_plan
        ),
        "decision.json": _atomic_json(output / "decision.json", decision),
    }
    manifest = {
        "schema": "tunnelgeopt.load_basis_development_manifest.v1",
        "run_id": config["run_id"],
        "classification": decision["classification"],
        "artifact_sha256": hashes,
    }
    manifest_hash = _atomic_json(output / "manifest.json", manifest)
    return {
        "status": "completed",
        "classification": decision["classification"],
        "passed": decision["passed"],
        "effect_claim_allowed": False,
        "evaluated_parent_geometries": analysis["evaluated_parent_geometries"],
        "evaluated_cases": analysis["evaluated_cases"],
        "median_relative_error": analysis["overall"]["basis_relative_error"]["median"],
        "p95_relative_error": analysis["overall"]["basis_relative_error"]["p95"],
        "maximum_relative_error": analysis["overall"]["basis_relative_error"]["maximum"],
        "ratio_of_mean_errors_to_raw_coarse": analysis["overall"]["ratio_of_mean_errors"],
        "manifest_sha256": manifest_hash,
        "output_dir": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args(argv)
    try:
        result = execute(arguments.config, arguments.output)
    except (LoadBasisDevelopmentError, LoadBasisError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
