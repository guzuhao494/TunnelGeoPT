#!/usr/bin/env python3
"""Audit the preregistered fine/ultrafine tiers on development-only cases.

This runner creates a seed namespace that is disjoint from every smoke/formal
test namespace.  It never opens a smoke pseudo-test file or any locked label.
The output calibrates mesh adequacy only; it is not model-effect evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tunnelgeopt.geometry import shape_parameter_bounds
from tunnelgeopt.multifidelity import (
    GeometryDataSpec,
    GeometrySplitSpec,
    MeshFidelitySpec,
    MultiFidelityDataset,
    build_elastic_query_grid,
    solve_multifidelity_case,
)
from tunnelgeopt.multifidelity_learning import case_weighted_stress_error

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "multifidelity_convergence_dev.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "analysis" / "mf-convergence-dev-v0.3.0"
SECTION_NAMES = ("circle", "horseshoe", "straight_wall_arch")


@dataclass(frozen=True)
class DevelopmentGeometry:
    spec: GeometryDataSpec
    geometry_id: str
    section: str
    local_index: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json_atomic(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
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


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not load convergence config: {exc}") from exc
    required = {
        "schema_version",
        "config_name",
        "status",
        "run_id",
        "scope",
        "claim_exclusions",
        "data_access",
        "geometry",
        "loads",
        "mesh",
        "query",
        "quality_gates",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise RuntimeError("convergence config key set changed")
    if config["status"] != "frozen_development_calibration_before_execution":
        raise RuntimeError("convergence config must be frozen before execution")
    if config["scope"] != "development_only_mesh_calibration_no_model_effect_claim":
        raise RuntimeError("convergence scope must remain development-only")
    access = config["data_access"]
    if (
        access.get("allowed_split") != "dev"
        or access.get("locked_or_pseudo_test_labels_allowed") is not False
    ):
        raise RuntimeError("locked or pseudo-test labels are forbidden")
    if tuple(config["geometry"].get("section_families", ())) != SECTION_NAMES:
        raise RuntimeError("convergence section families changed")
    if config["quality_gates"].get("effect_claim_allowed") is not False:
        raise RuntimeError("development convergence may not authorize an effect claim")
    query = config["query"]
    if int(query["points_per_case"]) != sum(
        int(query[name]) for name in ("nearfield_volume", "wall_offset", "farfield")
    ):
        raise RuntimeError("query region counts do not sum to points_per_case")
    expected_cases = (
        len(SECTION_NAMES)
        * int(config["geometry"]["parents_per_family"])
        * int(config["loads"]["per_geometry"])
    )
    gates = config["quality_gates"]
    if expected_cases != int(gates["required_case_count"]):
        raise RuntimeError("required_case_count disagrees with generation counts")
    if int(config["geometry"]["parents_per_family"]) * int(config["loads"]["per_geometry"]) != int(
        gates["required_cases_per_family"]
    ):
        raise RuntimeError("required_cases_per_family disagrees with generation counts")
    for tier in ("fine", "ultrafine"):
        if set(config["mesh"][tier]) != {
            "mesh_size_over_radius",
            "wall_size_over_radius",
            "farfield_size_over_radius",
        }:
            raise RuntimeError(f"{tier} mesh does not freeze all three size controls")
    return config


def _parameter_values(
    section: str, count: int, *, seed: int, quantile_range: tuple[float, float]
) -> list[dict[str, float]]:
    bounds = shape_parameter_bounds(section)
    rng = np.random.default_rng(int(seed))
    low, high = quantile_range
    values: list[dict[str, float]] = []
    for _ in range(int(count)):
        parameters = {}
        for name, (lower, upper) in bounds.items():
            quantile = float(rng.uniform(low, high))
            parameters[name] = float(lower + quantile * (upper - lower))
        values.append(parameters)
    return values


def _development_geometries(config: dict[str, Any]) -> list[DevelopmentGeometry]:
    geometry_config = config["geometry"]
    count = int(geometry_config["parents_per_family"])
    quantiles = tuple(map(float, geometry_config["parameter_quantile_range"]))
    roughness_low, roughness_high = map(float, geometry_config["roughness_amplitude_range"])
    master_rng = np.random.default_rng(int(geometry_config["generator_seed"]))
    generated: list[DevelopmentGeometry] = []
    boundary_digests: set[str] = set()
    for section_index, section in enumerate(SECTION_NAMES):
        parameters = _parameter_values(
            section,
            count,
            seed=int(geometry_config["generator_seed"]) + 1009 * section_index,
            quantile_range=(quantiles[0], quantiles[1]),
        )
        for local_index in range(count):
            spec = GeometryDataSpec(
                shape=section,
                parameters=parameters[local_index],
                n_boundary_points=int(geometry_config["boundary_points"]),
                radius=float(geometry_config["radius"]),
                roughness_amplitude=float(master_rng.uniform(roughness_low, roughness_high)),
                seed=int(geometry_config["generator_seed"]) + 10_000 * section_index + local_index,
                outer_domain_scale=float(config["mesh"]["outer_half_width_over_radius"]),
            )
            geometry = spec.build()
            digest = _sha256_bytes(
                np.ascontiguousarray(geometry.boundary_yz, dtype="<f8").tobytes()
            )
            if digest in boundary_digests:
                raise RuntimeError("development generator produced a duplicate exact boundary")
            boundary_digests.add(digest)
            generated.append(
                DevelopmentGeometry(
                    spec=spec,
                    geometry_id=spec.geometry_group_id(geometry),
                    section=section,
                    local_index=local_index,
                )
            )
    return sorted(generated, key=lambda item: (item.section, item.geometry_id))


def _load_tensor(config: dict[str, Any], geometry_id: str, load_index: int) -> np.ndarray:
    load_config = config["loads"]
    namespace = config["data_access"]["seed_namespace"]
    seed_text = f"{namespace}:{load_config['generator_seed']}:{geometry_id}:{load_index}"
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    sigma1 = float(rng.uniform(*map(float, load_config["sigma1_over_reference_stress"])))
    ratio = float(rng.uniform(*map(float, load_config["sigma3_over_sigma1"])))
    angle = math.radians(float(rng.uniform(*map(float, load_config["principal_angle_deg"]))))
    direction = np.asarray([math.cos(angle), math.sin(angle)])
    transverse = np.asarray([-math.sin(angle), math.cos(angle)])
    return -sigma1 * np.outer(direction, direction) - sigma1 * ratio * np.outer(
        transverse, transverse
    )


def _mesh_spec(config: dict[str, Any], tier: str) -> MeshFidelitySpec:
    values = config["mesh"][tier]
    return MeshFidelitySpec(
        mesh_size=float(values["mesh_size_over_radius"]),
        wall_mesh_size=float(values["wall_size_over_radius"]),
        farfield_mesh_size=float(values["farfield_size_over_radius"]),
    )


def _environment() -> dict[str, Any]:
    packages = {}
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
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        worktree_status = subprocess.run(
            ["git", "status", "--short"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        worktree_status = []
    source_paths = (
        "src/tunnelgeopt/elasticity.py",
        "src/tunnelgeopt/field_sampling.py",
        "src/tunnelgeopt/geometry.py",
        "src/tunnelgeopt/mesh.py",
        "src/tunnelgeopt/multifidelity.py",
        "src/tunnelgeopt/multifidelity_learning.py",
    )
    return {
        "captured_at_utc": _now(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "packages": packages,
        "git_head_at_execution": revision,
        "git_worktree_status_at_execution": worktree_status,
        "source_file_sha256": {
            relative: _file_sha256(ROOT / relative) for relative in source_paths
        },
    }


def _aggregate(case_records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    errors = np.asarray([record["stress_frobenius_rell2"] for record in case_records])
    by_family = {
        section: np.asarray(
            [
                record["stress_frobenius_rell2"]
                for record in case_records
                if record["section_family"] == section
            ],
            dtype=np.float64,
        )
        for section in SECTION_NAMES
    }
    all_residuals = [
        float(record["solver_qc"][tier]["algebraic_residual"])
        for record in case_records
        for tier in ("fine", "ultrafine")
    ]
    all_energy = [
        float(record["solver_qc"][tier]["energy_closure"])
        for record in case_records
        for tier in ("fine", "ultrafine")
    ]
    all_quality = [
        float(record["solver_qc"][tier]["minimum_triangle_quality"])
        for record in case_records
        for tier in ("fine", "ultrafine")
    ]
    all_located = all(
        bool(record["query_qc"][tier]["all_points_located"])
        for record in case_records
        for tier in ("fine", "ultrafine")
    )
    all_identity = all(
        record["identity_qc"][name] is True
        for record in case_records
        for name in ("same_frozen_boundary", "same_outer_bounds", "same_query_hash")
    )
    gates = config["quality_gates"]
    family_medians = {section: float(np.median(values)) for section, values in by_family.items()}
    counts_by_family = {section: int(values.size) for section, values in by_family.items()}
    checks = {
        "required_case_count": len(case_records) == int(gates["required_case_count"]),
        "required_cases_per_family": all(
            count == int(gates["required_cases_per_family"]) for count in counts_by_family.values()
        ),
        "stress_case_median": float(np.median(errors))
        <= float(gates["stress_frobenius_rell2_case_median_max"]),
        "stress_case_p95": float(np.quantile(errors, 0.95))
        <= float(gates["stress_frobenius_rell2_case_p95_max"]),
        "stress_each_family_median": max(family_medians.values())
        <= float(gates["stress_frobenius_rell2_each_family_median_max"]),
        "algebraic_residual": max(all_residuals) <= float(gates["algebraic_residual_max"]),
        "energy_closure": max(all_energy) <= float(gates["energy_closure_max"]),
        "minimum_triangle_quality": min(all_quality)
        >= float(gates["minimum_triangle_quality_min"]),
        "all_query_points_located": all_located,
        "same_boundary_outer_domain_and_query": all_identity,
        "effect_claim_allowed": False,
    }
    positive = {key: value for key, value in checks.items() if key != "effect_claim_allowed"}
    decision = (
        "current_tiers_eligible_for_formal_freeze"
        if all(value is True for value in positive.values())
        and checks["effect_claim_allowed"] is False
        else "do_not_start_formal_with_current_tiers"
    )
    return {
        "primary_metric": {
            "name": "nearfield_area_weighted_stress_tensor_frobenius_relative_l2",
            "shear_multiplier": 2.0,
            "case_weighting": "each case equal for distribution summaries",
            "case_count": int(errors.size),
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p90": float(np.quantile(errors, 0.90)),
            "p95": float(np.quantile(errors, 0.95)),
            "maximum": float(np.max(errors)),
            "family_case_counts": counts_by_family,
            "family_medians": family_medians,
        },
        "solver_qc": {
            "maximum_algebraic_residual": max(all_residuals),
            "maximum_energy_closure": max(all_energy),
            "minimum_triangle_quality": min(all_quality),
            "all_query_points_located": all_located,
            "same_boundary_outer_domain_and_query": all_identity,
        },
        "checks": checks,
        "decision": decision,
        "effect_claim_allowed": False,
        "claim_boundary": (
            "Development-only mesh calibration; no model, locked-test, fracture, rockburst, "
            "field-validity, or transfer claim is authorized."
        ),
    }


def run(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    geometries = _development_geometries(config)
    fine_mesh = _mesh_spec(config, "fine")
    ultrafine_mesh = _mesh_spec(config, "ultrafine")
    query_config = config["query"]
    samples = []
    sample_sections: list[str] = []
    load_tensors: list[list[list[float]]] = []
    case_seconds: list[float] = []
    for geometry_index, entry in enumerate(geometries):
        geometry = entry.spec.build()
        grid = build_elastic_query_grid(
            geometry,
            geometry_parameters=entry.spec.identity_parameters(),
            nearfield_points=int(query_config["nearfield_volume"]),
            wall_offset_points=int(query_config["wall_offset"]),
            farfield_points=int(query_config["farfield"]),
            nearfield_min_distance_over_radius=float(
                query_config["nearfield_distance_over_radius"][0]
            ),
            nearfield_max_distance_over_radius=float(
                query_config["nearfield_distance_over_radius"][1]
            ),
            wall_offset_over_radius=float(query_config["wall_offset_over_radius"]),
            seed=int(query_config["generator_seed"]) + geometry_index,
            outer_domain_scale=float(entry.spec.outer_domain_scale),
        )
        if grid.geometry_group_id != entry.geometry_id:
            raise RuntimeError("development geometry identity changed while building the grid")
        for load_index in range(int(config["loads"]["per_geometry"])):
            sigma_inf = _load_tensor(config, entry.geometry_id, load_index)
            case_start = time.perf_counter()
            sample = solve_multifidelity_case(
                geometry,
                grid,
                split="dev",
                sigma_inf_tension_positive=sigma_inf,
                young_modulus=float(config["loads"]["young_modulus_over_reference_stress"]),
                poisson_ratio=float(config["loads"]["poisson_ratio"]),
                coarse_mesh=fine_mesh,
                fine_mesh=ultrafine_mesh,
                domain_scale=float(entry.spec.outer_domain_scale),
                geometry_spec=entry.spec,
            )
            elapsed = time.perf_counter() - case_start
            samples.append(sample)
            sample_sections.append(entry.section)
            load_tensors.append(np.asarray(sigma_inf, dtype=np.float64).tolist())
            case_seconds.append(elapsed)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    _canonical_json(
                        {
                            "event": "development_case_solved",
                            "at_utc": _now(),
                            "case_group_id": sample.case_group_id,
                            "geometry_group_id": sample.geometry_group_id,
                            "section_family": entry.section,
                            "load_index": load_index,
                            "completed_case_count": len(samples),
                            "solver_seconds": elapsed,
                        }
                    )
                    + "\n"
                )

    split_spec = GeometrySplitSpec(
        train=(),
        dev=tuple(sorted(entry.geometry_id for entry in geometries)),
        locked_test=(),
        protocol="development_only_explicit_v1",
        section_by_geometry={entry.geometry_id: entry.section for entry in geometries},
    )
    dataset = MultiFidelityDataset(tuple(samples), split_spec)
    dev_indices = dataset.indices("dev")
    ultrafine = dataset.fine_labels_for(
        dev_indices, purpose="development_fine_ultrafine_mesh_calibration"
    )
    fine = np.stack([sample.coarse_stress_normalized for sample in samples]).astype(np.float32)
    weights = np.stack([sample.grid.area_weights for sample in samples]).astype(np.float32)
    errors = case_weighted_stress_error(fine, ultrafine, weights)
    case_records = []
    for index, (sample, section, error, sigma_inf, elapsed) in enumerate(
        zip(samples, sample_sections, errors, load_tensors, case_seconds, strict=True)
    ):
        case_records.append(
            {
                "case_index": index,
                "case_group_id": sample.case_group_id,
                "geometry_group_id": sample.geometry_group_id,
                "load_group_id": sample.load_group_id,
                "section_family": section,
                "split": sample.split,
                "sigma_inf_tension_positive": sigma_inf,
                "query_hash": sample.grid.query_hash,
                "point_count": sample.grid.point_count,
                "stress_frobenius_rell2": float(error),
                "solver_seconds": elapsed,
                "element_counts": {
                    "fine": int(sample.coarse_mesh_metadata["element_count"]),
                    "ultrafine": int(sample.fine_mesh_metadata["element_count"]),
                },
                "solver_qc": {
                    "fine": {
                        "algebraic_residual": float(
                            sample.diagnostics["coarse"]["algebraic_residual"]
                        ),
                        "energy_closure": float(sample.diagnostics["coarse"]["energy_closure"]),
                        "minimum_triangle_quality": float(
                            sample.coarse_mesh_metadata["minimum_triangle_quality"]
                        ),
                    },
                    "ultrafine": {
                        "algebraic_residual": float(
                            sample.diagnostics["fine"]["algebraic_residual"]
                        ),
                        "energy_closure": float(sample.diagnostics["fine"]["energy_closure"]),
                        "minimum_triangle_quality": float(
                            sample.fine_mesh_metadata["minimum_triangle_quality"]
                        ),
                    },
                },
                "query_qc": {
                    "fine": {"all_points_located": bool(np.all(sample.coarse_element_ids >= 0))},
                    "ultrafine": {"all_points_located": bool(np.all(sample.fine_element_ids >= 0))},
                },
                "identity_qc": {
                    "same_frozen_boundary": bool(sample.diagnostics["same_frozen_boundary"]),
                    "same_outer_bounds": bool(sample.diagnostics["same_outer_bounds"]),
                    "same_query_hash": sample.diagnostics["common_query_hash"]
                    == sample.grid.query_hash,
                },
            }
        )

    metrics = _aggregate(case_records, config)
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["solver_seconds"] = {
        "total": float(sum(case_seconds)),
        "median_per_case": float(np.median(case_seconds)),
        "maximum_per_case": float(max(case_seconds)),
    }
    metrics["data_access_audit"] = dataset.access_snapshot()
    metrics["design_inventory"] = {
        "split": "dev",
        "parent_geometry_count": len(geometries),
        "case_count": len(case_records),
        "case_count_by_family": {
            section: sum(record["section_family"] == section for record in case_records)
            for section in SECTION_NAMES
        },
        "mesh_tiers_over_radius": config["mesh"],
        "common_query_hash_count": len({record["query_hash"] for record in case_records}),
        "same_query_hash_for_fine_and_ultrafine_every_case": all(
            record["identity_qc"]["same_query_hash"] for record in case_records
        ),
    }
    config_hash = _sha256_bytes(_canonical_json(config).encode("utf-8"))
    environment = _environment()
    environment["config_sha256"] = config_hash
    environment["runner_sha256"] = _file_sha256(Path(__file__).resolve())

    file_hashes = {}
    file_hashes["config.snapshot.json"] = _write_json_atomic(
        output_dir / "config.snapshot.json", config
    )
    file_hashes["environment.json"] = _write_json_atomic(
        output_dir / "environment.json", environment
    )
    file_hashes["case_metrics.json"] = _write_json_atomic(
        output_dir / "case_metrics.json", {"cases": case_records}
    )
    file_hashes["metrics.json"] = _write_json_atomic(output_dir / "metrics.json", metrics)
    file_hashes["progress.jsonl"] = _file_sha256(progress_path)
    manifest = {
        "schema_version": "tunnelgeopt.multifidelity_convergence_manifest.v1",
        "run_id": config["run_id"],
        "created_at_utc": _now(),
        "scope": config["scope"],
        "status": "complete",
        "config_sha256": config_hash,
        "runner_sha256": environment["runner_sha256"],
        "files": file_hashes,
        "geometry_count": len(geometries),
        "case_count": len(case_records),
        "case_distribution": {
            "split": "dev",
            "by_family": metrics["design_inventory"]["case_count_by_family"],
        },
        "mesh_tiers_over_radius": config["mesh"],
        "common_query_contract": {
            "unique_parent_query_hash_count": metrics["design_inventory"][
                "common_query_hash_count"
            ],
            "same_hash_for_fine_and_ultrafine_every_case": metrics["design_inventory"][
                "same_query_hash_for_fine_and_ultrafine_every_case"
            ],
        },
        "section_families": list(SECTION_NAMES),
        "seed_namespace": config["data_access"]["seed_namespace"],
        "locked_or_pseudo_test_labels_read": False,
        "locked_test_fine_label_case_reads": metrics["data_access_audit"]["fine_label_case_reads"][
            "locked_test"
        ],
        "effect_claim_allowed": False,
        "decision": metrics["decision"],
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    return {"manifest": manifest, "metrics": metrics}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--validate-only", action="store_true", help="validate the frozen contract without FEM"
    )
    args = parser.parse_args()
    config = _load_config(args.config.resolve())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "run_id": config["run_id"],
                    "config_sha256": _sha256_bytes(_canonical_json(config).encode("utf-8")),
                },
                indent=2,
            )
        )
        return 0
    result = run(config, args.output.resolve())
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
    return 0 if result["metrics"]["decision"] == "current_tiers_eligible_for_formal_freeze" else 2


if __name__ == "__main__":
    raise SystemExit(main())
