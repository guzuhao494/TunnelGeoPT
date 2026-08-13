#!/usr/bin/env python3
"""Run the preregistered v0.3 coarse-to-fine pipeline smoke.

The runner deliberately stores public inputs, train/dev labels, and sealed
pseudo-test labels in three different files.  The sealed file is not opened
until every expected CPU checkpoint has been written and hashed.  This smoke
checks plumbing and auditability only; its model errors are not effect evidence.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
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
from tunnelgeopt.multifidelity_learning import (
    LearningBatch,
    case_weighted_stress_error,
    load_model_from_checkpoint,
    make_model,
    method_arrays,
    mismatched_coarse_indices,
    nested_geometry_subsets,
    predict,
    reconstruct_fine_prediction,
    save_checkpoint_atomic,
    section_balanced_geometry_mean,
    train_with_dev_selection,
    write_json_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "multifidelity_smoke.json"
DEFAULT_DATA = ROOT / "outputs" / "mf-residual-smoke-v0.3.0-data"
DEFAULT_RUN = ROOT / "artifacts" / "experiment" / "mf-residual-smoke-v0.3.0"
SECTION_NAMES = ("circle", "horseshoe", "straight_wall_arch")
METHOD_MAP = {
    "scratch_100": ("scratch", 1.0),
    "direct_coarse_100": ("direct_coarse", 1.0),
    "residual_coarse_50": ("residual_coarse", 0.5),
    "residual_coarse_100": ("residual_coarse", 1.0),
    "mismatched_coarse_50": ("mismatched_coarse", 0.5),
}


@dataclass(frozen=True)
class GeneratedGeometry:
    spec: GeometryDataSpec
    geometry_id: str
    section: str
    split: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_npz(path: Path, **arrays: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"could not load smoke config: {exc}") from exc
    required = {
        "schema_version",
        "config_name",
        "status",
        "run_id",
        "scope",
        "split_salt",
        "claim_exclusions",
        "geometry",
        "loads",
        "mesh",
        "query",
        "methods",
        "model",
        "optimization",
        "smoke_success",
    }
    if not isinstance(config, dict) or set(config) != required:
        raise RuntimeError("smoke config key set changed")
    if config["scope"] != "pipeline_only_no_effect_claim":
        raise RuntimeError("smoke scope must remain pipeline-only")
    if tuple(config["geometry"]["section_families"]) != SECTION_NAMES:
        raise RuntimeError("smoke section families changed")
    if set(config["methods"]) != set(METHOD_MAP):
        raise RuntimeError("smoke methods changed")
    if config["smoke_success"].get("effect_claim_allowed") is not False:
        raise RuntimeError("the smoke may not authorize an effect claim")
    return config


def _smoke_decision(checks: dict[str, bool]) -> str:
    """Aggregate positive checks while requiring the claim permission to stay false."""

    if set(checks) != {
        "all_solver_cases",
        "all_query_points_located_in_both_meshes",
        "no_cross_split_parent",
        "locked_label_denied_before_authorization",
        "all_methods_one_checkpoint",
        "finite_metrics",
        "effect_claim_allowed",
    }:
        raise RuntimeError("smoke check key set changed")
    positive = {key: value for key, value in checks.items() if key != "effect_claim_allowed"}
    return (
        "pipeline_go"
        if all(value is True for value in positive.values())
        and checks["effect_claim_allowed"] is False
        else "pipeline_no_go"
    )


def _parameter_values(section: str, count: int, seed: int) -> list[dict[str, float]]:
    bounds = shape_parameter_bounds(section)
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(count):
        parameters = {}
        for name, (lower, upper) in bounds.items():
            q = rng.uniform(0.15, 0.85)
            parameters[name] = float(lower + q * (upper - lower))
        values.append(parameters)
    return values


def _geometry_specs(config: dict[str, Any]) -> tuple[list[GeneratedGeometry], GeometrySplitSpec]:
    geometry_config = config["geometry"]
    counts = geometry_config["parent_geometry_counts_per_family"]
    count_per_section = int(counts["train"] + counts["dev"] + counts["pseudo_test"])
    roughness_low, roughness_high = map(float, geometry_config["roughness_amplitude_range"])
    rng = np.random.default_rng(int(geometry_config["generator_seed"]))
    provisional: dict[str, list[tuple[GeometryDataSpec, str]]] = {}
    seen_boundaries: set[str] = set()
    for section_index, section in enumerate(SECTION_NAMES):
        parameters = _parameter_values(
            section,
            count_per_section,
            int(geometry_config["generator_seed"]) + 1009 * section_index,
        )
        section_values = []
        for local_index in range(count_per_section):
            spec = GeometryDataSpec(
                shape=section,
                parameters=parameters[local_index],
                n_boundary_points=int(geometry_config["boundary_points"]),
                radius=float(geometry_config["radius"]),
                roughness_amplitude=float(rng.uniform(roughness_low, roughness_high)),
                seed=int(geometry_config["generator_seed"]) + 10_000 * section_index + local_index,
                outer_domain_scale=float(config["mesh"]["outer_half_width_over_radius"]),
            )
            geometry = spec.build()
            geometry_id = spec.geometry_group_id(geometry)
            boundary_digest = hashlib.sha256(
                np.ascontiguousarray(geometry.boundary_yz, dtype="<f8").tobytes()
            ).hexdigest()
            if boundary_digest in seen_boundaries:
                raise RuntimeError("duplicate exact boundary generated")
            seen_boundaries.add(boundary_digest)
            section_values.append((spec, geometry_id))
        provisional[section] = sorted(section_values, key=lambda value: value[1])

    generated: list[GeneratedGeometry] = []
    split_ids: dict[str, list[str]] = {"train": [], "dev": [], "locked_test": []}
    for section in SECTION_NAMES:
        cursor = 0
        for split, config_name in (
            ("train", "train"),
            ("dev", "dev"),
            ("locked_test", "pseudo_test"),
        ):
            take = int(counts[config_name])
            for spec, geometry_id in provisional[section][cursor : cursor + take]:
                generated.append(GeneratedGeometry(spec, geometry_id, section, split))
                split_ids[split].append(geometry_id)
            cursor += take
    split_spec = GeometrySplitSpec(
        train=tuple(sorted(split_ids["train"])),
        dev=tuple(sorted(split_ids["dev"])),
        locked_test=tuple(sorted(split_ids["locked_test"])),
    )
    return sorted(generated, key=lambda value: value.geometry_id), split_spec


def _load_tensor(config: dict[str, Any], geometry_id: str, load_index: int) -> np.ndarray:
    load_config = config["loads"]
    seed_text = f"{load_config['generator_seed']}:{geometry_id}:{load_index}"
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    sigma1 = rng.uniform(*map(float, load_config["sigma1_over_reference_stress"]))
    ratio = rng.uniform(*map(float, load_config["sigma3_over_sigma1"]))
    angle = np.deg2rad(rng.uniform(*map(float, load_config["principal_angle_deg"])))
    direction = np.asarray([np.cos(angle), np.sin(angle)])
    transverse = np.asarray([-np.sin(angle), np.cos(angle)])
    # The solver is tension-positive; prescribed rock compression is negative.
    return -sigma1 * np.outer(direction, direction) - sigma1 * ratio * np.outer(
        transverse, transverse
    )


def _mesh_spec(config: dict[str, Any], tier: str) -> MeshFidelitySpec:
    values = config["mesh"][tier]
    return MeshFidelitySpec(
        mesh_size=float(values["farfield_size_over_radius"]),
        wall_mesh_size=float(values["wall_size_over_radius"]),
        farfield_mesh_size=float(values["farfield_size_over_radius"]),
    )


def generate_dataset(
    config: dict[str, Any], data_dir: Path, *, progress_path: Path
) -> dict[str, Any]:
    start = time.perf_counter()
    geometries, split_spec = _geometry_specs(config)
    query_config = config["query"]
    loads_per_geometry = int(config["loads"]["per_geometry"])
    samples = []
    geometry_records = []
    case_records = []
    coarse_mesh = _mesh_spec(config, "coarse")
    fine_mesh = _mesh_spec(config, "fine")
    with progress_path.open("a", encoding="utf-8") as progress:
        for geometry_index, entry in enumerate(geometries):
            geometry = entry.spec.build()
            identity_parameters = entry.spec.identity_parameters()
            grid = build_elastic_query_grid(
                geometry,
                geometry_parameters=identity_parameters,
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
                raise RuntimeError("geometry identity changed while building its query grid")
            geometry_records.append(
                {
                    "geometry_group_id": entry.geometry_id,
                    "section_family": entry.section,
                    "split": entry.split,
                    "query_hash": grid.query_hash,
                    "generator": identity_parameters,
                }
            )
            for load_index in range(loads_per_geometry):
                sigma_inf = _load_tensor(config, entry.geometry_id, load_index)
                case_start = time.perf_counter()
                sample = solve_multifidelity_case(
                    geometry,
                    grid,
                    split=entry.split,
                    sigma_inf_tension_positive=sigma_inf,
                    young_modulus=float(config["loads"]["young_modulus_over_reference_stress"]),
                    poisson_ratio=float(config["loads"]["poisson_ratio"]),
                    coarse_mesh=coarse_mesh,
                    fine_mesh=fine_mesh,
                    domain_scale=float(entry.spec.outer_domain_scale),
                    geometry_identity_parameters=identity_parameters,
                )
                samples.append(sample)
                case_records.append(
                    {
                        "case_group_id": sample.case_group_id,
                        "geometry_group_id": entry.geometry_id,
                        "load_group_id": sample.load_group_id,
                        "section_family": entry.section,
                        "split": entry.split,
                        "load_index": load_index,
                        "query_hash": grid.query_hash,
                        "coarse_elements": sample.coarse_mesh_metadata["element_count"],
                        "fine_elements": sample.fine_mesh_metadata["element_count"],
                        "solver_seconds": time.perf_counter() - case_start,
                        "diagnostics": sample.diagnostics,
                    }
                )
                progress.write(
                    _canonical_json(
                        {
                            "event": "case_generated",
                            "at_utc": _now(),
                            "case_group_id": sample.case_group_id,
                            "split": entry.split,
                            "section": entry.section,
                            "index": len(samples),
                        }
                    )
                    + "\n"
                )
                progress.flush()

    dataset = MultiFidelityDataset(tuple(samples), split_spec)
    # This is an actual data-layer denial in the trusted generation process.
    denied = False
    try:
        dataset.fine_labels_for(dataset.indices("locked_test"), purpose="smoke_premature_probe")
    except ValueError:
        denied = True
    if not denied:
        raise RuntimeError("locked fine-label access was not denied")
    generation_access_audit = dataset.access_snapshot()

    base = np.stack([sample.model_features[:, :11] for sample in samples]).astype(np.float32)
    coarse = np.stack([sample.coarse_stress_normalized for sample in samples]).astype(np.float32)
    fine = np.stack([sample._fine_stress_normalized for sample in samples]).astype(np.float32)
    weights = np.stack([sample.grid.area_weights for sample in samples]).astype(np.float32)
    case_ids = np.asarray([sample.case_group_id for sample in samples], dtype="U64")
    geometry_ids = np.asarray([sample.geometry_group_id for sample in samples], dtype="U64")
    sections = np.asarray(
        [
            next(
                record["section_family"]
                for record in case_records
                if record["case_group_id"] == sample.case_group_id
            )
            for sample in samples
        ],
        dtype="U32",
    )
    splits = np.asarray([sample.split for sample in samples], dtype="U16")
    train_dev = np.flatnonzero(splits != "locked_test")
    locked = np.flatnonzero(splits == "locked_test")
    data_dir.mkdir(parents=True, exist_ok=True)
    public_path = data_dir / "public_inputs.npz"
    train_path = data_dir / "train_dev_labels.npz"
    sealed_path = data_dir / "sealed_pseudo_test_labels.npz"
    hashes = {
        "public_inputs.npz": _atomic_npz(
            public_path,
            base_features=base,
            coarse_stress=coarse,
            weights=weights,
            case_group_ids=case_ids,
            geometry_group_ids=geometry_ids,
            section_families=sections,
            splits=splits,
        ),
        "train_dev_labels.npz": _atomic_npz(
            train_path,
            indices=train_dev,
            fine_stress=fine[train_dev],
            case_group_ids=case_ids[train_dev],
        ),
        "sealed_pseudo_test_labels.npz": _atomic_npz(
            sealed_path, indices=locked, fine_stress=fine[locked], case_group_ids=case_ids[locked]
        ),
    }
    manifest = {
        "format_version": 1,
        "generated_at_utc": _now(),
        "scope": config["scope"],
        "effect_claim_allowed": False,
        "split_spec": split_spec.as_dict(),
        "geometry_records": geometry_records,
        "case_records": case_records,
        "counts": {
            "geometries": len(geometries),
            "cases": len(samples),
            "points_per_case": int(base.shape[1]),
            "train_cases": int(np.sum(splits == "train")),
            "dev_cases": int(np.sum(splits == "dev")),
            "pseudo_test_cases": int(np.sum(splits == "locked_test")),
        },
        "files": hashes,
        "generation_checks": {
            "all_solver_cases": len(samples)
            == len(geometries) * int(config["loads"]["per_geometry"]),
            "all_query_points_located_in_both_meshes": all(
                np.all(sample.coarse_element_ids >= 0) and np.all(sample.fine_element_ids >= 0)
                for sample in samples
            ),
            "no_cross_split_parent": True,
            "locked_label_denied_before_authorization": denied,
        },
        "generation_access_audit": generation_access_audit,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json_atomic(data_dir / "dataset_manifest.json", manifest)
    return manifest


def _load_public(data_dir: Path) -> dict[str, np.ndarray]:
    with np.load(data_dir / "public_inputs.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _labels_for_indices(data_dir: Path, filename: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(data_dir / filename, allow_pickle=False) as archive:
        return np.asarray(archive["indices"], dtype=np.int64), np.asarray(
            archive["fine_stress"], dtype=np.float32
        )


def _learning_batch(
    public: dict[str, np.ndarray], indices: np.ndarray, fine: np.ndarray
) -> LearningBatch:
    if fine.shape[0] != indices.size:
        raise RuntimeError("fine label rows do not align with selected public cases")
    return LearningBatch(
        base_features=public["base_features"][indices],
        coarse_stress=public["coarse_stress"][indices],
        fine_stress=fine,
        weights=public["weights"][indices],
        geometry_group_ids=tuple(str(value) for value in public["geometry_group_ids"][indices]),
        section_families=tuple(str(value) for value in public["section_families"][indices]),
        case_group_ids=tuple(str(value) for value in public["case_group_ids"][indices]),
        splits=tuple(str(value) for value in public["splits"][indices]),
    )


def _select_batch(batch: LearningBatch, indices: np.ndarray) -> LearningBatch:
    return LearningBatch(
        base_features=batch.base_features[indices],
        coarse_stress=batch.coarse_stress[indices],
        fine_stress=batch.fine_stress[indices],
        weights=batch.weights[indices],
        geometry_group_ids=tuple(batch.geometry_group_ids[int(index)] for index in indices),
        section_families=tuple(batch.section_families[int(index)] for index in indices),
        case_group_ids=tuple(batch.case_group_ids[int(index)] for index in indices),
        splits=tuple(batch.splits[int(index)] for index in indices),
    )


def _method_arrays(
    batch: LearningBatch, method: str, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mismatch = None
    if method == "mismatched_coarse":
        mismatch = mismatched_coarse_indices(batch.section_families, seed=seed)
    return method_arrays(batch, method, mismatch_indices=mismatch)


def train_and_evaluate(
    config: dict[str, Any], data_dir: Path, run_dir: Path, *, device: str
) -> dict[str, Any]:
    import torch

    start = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = data_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_sha256 = _sha256_bytes(_canonical_json(config).encode())
    public = _load_public(data_dir)
    dataset_manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest_hash = write_json_atomic(run_dir / "dataset_manifest.json", dataset_manifest)
    train_dev_indices, train_dev_fine = _labels_for_indices(data_dir, "train_dev_labels.npz")
    # The sealed path exists but is not opened anywhere above this line.
    train_dev = _learning_batch(public, train_dev_indices, train_dev_fine)
    train_local = np.flatnonzero(np.asarray(train_dev.splits) == "train")
    dev_local = np.flatnonzero(np.asarray(train_dev.splits) == "dev")
    train_full = _select_batch(train_dev, train_local)
    dev_batch = _select_batch(train_dev, dev_local)
    subsets = nested_geometry_subsets(
        train_full.geometry_group_ids,
        train_full.section_families,
        fractions=(0.5, 1.0),
        salt=config["split_salt"],
    )
    optimization = config["optimization"]
    training_seed = int(optimization["training_seeds"][0])
    checkpoint_records = []
    training_records = []
    progress_path = run_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    for method_name in config["methods"]:
        method, fraction = METHOD_MAP[method_name]
        selected_geometries = set(subsets[fraction])
        selected = np.asarray(
            [
                index
                for index, geometry_id in enumerate(train_full.geometry_group_ids)
                if geometry_id in selected_geometries
            ],
            dtype=np.int64,
        )
        method_train = _select_batch(train_full, selected)
        train_x, train_y, _ = _method_arrays(method_train, method, seed=training_seed + 17)
        dev_x, _, dev_base = _method_arrays(dev_batch, method, seed=training_seed + 29)
        model = make_model(config["model"], seed=training_seed, device=device)

        def record_progress(value: dict[str, Any], current_method: str = method_name) -> None:
            with progress_path.open("a", encoding="utf-8") as progress:
                progress.write(
                    _canonical_json({"method": current_method, "at_utc": _now(), **value}) + "\n"
                )

        outcome = train_with_dev_selection(
            model,
            train_x,
            train_y,
            method_train.weights,
            dev_x,
            dev_batch.fine_stress,
            dev_base,
            dev_batch.weights,
            seed=training_seed,
            device=device,
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            batch_size=int(optimization["case_batch_size"]),
            max_epochs=int(optimization["max_epochs"]),
            patience=int(optimization["early_stopping_patience"]),
            min_delta=float(optimization["early_stopping_min_delta"]),
            progress=record_progress,
        )
        checkpoint_path = checkpoint_dir / f"{method_name}-seed-{training_seed}.pt"
        digest = save_checkpoint_atomic(
            outcome,
            checkpoint_path,
            method=method,
            fraction=fraction,
            seed=training_seed,
            model_config=config["model"],
            config_sha256=config_sha256,
            train_geometry_ids=tuple(sorted(selected_geometries)),
        )
        checkpoint_records.append(
            {
                "method_name": method_name,
                "method": method,
                "fraction": fraction,
                "seed": training_seed,
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": digest,
            }
        )
        training_records.append(
            {
                "method_name": method_name,
                "train_cases": int(selected.size),
                "train_geometries": len(selected_geometries),
                "best_epoch": outcome.best_epoch,
                "epochs_run": outcome.epochs_run,
                "best_dev_error": outcome.best_dev_error,
            }
        )
        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected = len(config["methods"])
    frozen_ids = [record["sha256"] for record in checkpoint_records]
    if len(frozen_ids) != expected or len(set(frozen_ids)) != expected:
        raise RuntimeError("the complete unique checkpoint set was not frozen")
    checkpoint_index_hash = write_json_atomic(
        run_dir / "checkpoint_index.json",
        {
            "expected": expected,
            "frozen_before_pseudo_labels": True,
            "checkpoints": checkpoint_records,
        },
    )

    # This is the only open of the sealed label file in the training phase.
    locked_indices, locked_fine = _labels_for_indices(data_dir, "sealed_pseudo_test_labels.npz")
    locked_batch = _learning_batch(public, locked_indices, locked_fine)
    results = []
    for record in checkpoint_records:
        model, payload = load_model_from_checkpoint(ROOT / record["path"], device=device)
        features, _, base = _method_arrays(
            locked_batch, payload["method"], seed=int(payload["seed"]) + 43
        )
        raw = predict(
            model,
            features,
            batch_size=int(optimization["case_batch_size"]),
            device=device,
        )
        prediction = reconstruct_fine_prediction(raw, base)
        errors = case_weighted_stress_error(
            prediction, locked_batch.fine_stress, locked_batch.weights
        )
        overall, geometry_means, section_means = section_balanced_geometry_mean(
            errors,
            locked_batch.geometry_group_ids,
            locked_batch.section_families,
        )
        results.append(
            {
                "method_name": record["method_name"],
                "evaluation_calls": 1,
                "case_errors": [float(value) for value in errors],
                "overall_section_balanced_geometry_mean": overall,
                "geometry_means": geometry_means,
                "section_means": section_means,
                "nonfinite_predictions": int(np.size(prediction) - np.isfinite(prediction).sum()),
            }
        )
        model.to("cpu")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    coarse_errors = case_weighted_stress_error(
        locked_batch.coarse_stress, locked_batch.fine_stress, locked_batch.weights
    )
    coarse_overall, _, coarse_sections = section_balanced_geometry_mean(
        coarse_errors,
        locked_batch.geometry_group_ids,
        locked_batch.section_families,
    )
    metrics = {
        "scope": config["scope"],
        "effect_claim_allowed": False,
        "warning": "smoke metrics are pipeline diagnostics, not model-effect evidence",
        "pseudo_test_case_count": int(locked_indices.size),
        "coarse_only": {
            "overall_section_balanced_geometry_mean": coarse_overall,
            "section_means": coarse_sections,
        },
        "methods": results,
    }
    write_json_atomic(run_dir / "metrics.json", metrics)
    access_audit = {
        "sealed_label_file_opened_before_all_checkpoints_frozen": False,
        "train_dev_label_case_reads": int(train_dev_indices.size),
        "pseudo_test_label_case_reads_before_freeze": 0,
        "frozen_checkpoint_count_authorized": expected,
        "pseudo_test_label_case_reads_after_freeze": int(locked_indices.size),
        "pseudo_test_evaluation_calls": expected,
        "evaluation_calls_per_checkpoint": 1,
    }
    write_json_atomic(run_dir / "access_audit.json", access_audit)
    generation_checks = dataset_manifest["generation_checks"]
    success = {
        "all_solver_cases": generation_checks["all_solver_cases"] is True,
        "all_query_points_located_in_both_meshes": generation_checks[
            "all_query_points_located_in_both_meshes"
        ]
        is True,
        "no_cross_split_parent": generation_checks["no_cross_split_parent"] is True,
        "locked_label_denied_before_authorization": generation_checks[
            "locked_label_denied_before_authorization"
        ]
        is True,
        "all_methods_one_checkpoint": len(checkpoint_records) == expected,
        "finite_metrics": all(
            result["nonfinite_predictions"] == 0
            and np.isfinite(result["overall_section_balanced_geometry_mean"])
            for result in results
        ),
        "effect_claim_allowed": False,
    }
    decision = _smoke_decision(success)
    run_manifest = {
        "run_id": config["run_id"],
        "status": "completed",
        "decision": decision,
        "scope": config["scope"],
        "effect_claim_allowed": False,
        "started_from_config_sha256": config_sha256,
        "completed_at_utc": _now(),
        "command": " ".join(sys.argv),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "checks": success,
        "checkpoint_index_sha256": checkpoint_index_hash,
        "dataset_manifest_sha256": dataset_manifest_hash,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write_json_atomic(run_dir / "manifest.json", run_manifest)
    return run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--phase", choices=("generate", "train", "all"), default="all")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    arguments = parser.parse_args(argv)
    config_path = arguments.config.resolve()
    data_dir = arguments.data_dir.resolve()
    run_dir = arguments.run_dir.resolve()
    config = _load_config(config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = json.loads(_canonical_json(config))
    write_json_atomic(run_dir / "config.snapshot.json", config_snapshot)
    if arguments.phase in {"generate", "all"}:
        progress_path = run_dir / "generation_progress.jsonl"
        progress_path.write_text("", encoding="utf-8")
        generate_dataset(config, data_dir, progress_path=progress_path)
    if arguments.phase in {"train", "all"}:
        if arguments.device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = arguments.device
        manifest = train_and_evaluate(config, data_dir, run_dir, device=device)
        print(_canonical_json(manifest))
        return 0 if manifest["decision"] == "pipeline_go" else 2
    print(_canonical_json({"status": "generated", "data_dir": str(data_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
