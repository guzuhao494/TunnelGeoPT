"""Fail-closed scientific decision engine for the v0.3 formal experiment.

The evaluator is deliberately independent from data generation, training, and
sealed-label I/O.  Its only inputs are already materialized JSON-like values:
the frozen config, sealed per-case metrics, the dataset manifest, and the
authenticated access state.  Missing evidence is never filled with a default;
it makes the scientific result ``ABSTAIN``.

The primary statistic follows the preregistration exactly: average all loads
inside a parent geometry, average parents inside each section family, then give
the section families equal weight.  Confidence intervals use a paired,
two-level bootstrap over training seeds and parent geometries within section.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

PRIMARY_PARTITIONS = (
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
)
LOCKED_PARTITIONS = (*PRIMARY_PARTITIONS, "locked_joint_ood")
REQUIRED_DATA_PARTITIONS = ("train_id", "dev_id", *LOCKED_PARTITIONS)
SHA256_HEX = frozenset("0123456789abcdef")
GIT_HEX = SHA256_HEX
IMPLEMENTATION_SOURCE_PATHS = frozenset(
    {
        "configs/multifidelity_formal.json",
        "configs/multifidelity_formal_approval.json",
        "configs/multifidelity_seen_identity_exclusions.json",
        "scripts/run_multifidelity_formal.py",
        "scripts/run_multifidelity_train_worker.py",
        "src/tunnelgeopt/__init__.py",
        "src/tunnelgeopt/cases.py",
        "src/tunnelgeopt/elastic_schema.py",
        "src/tunnelgeopt/elastic_validation.py",
        "src/tunnelgeopt/elasticity.py",
        "src/tunnelgeopt/field_sampling.py",
        "src/tunnelgeopt/formal_analysis.py",
        "src/tunnelgeopt/formal_generation.py",
        "src/tunnelgeopt/geometry.py",
        "src/tunnelgeopt/kirsch.py",
        "src/tunnelgeopt/lift.py",
        "src/tunnelgeopt/mesh.py",
        "src/tunnelgeopt/multifidelity.py",
        "src/tunnelgeopt/multifidelity_learning.py",
        "src/tunnelgeopt/schema.py",
    }
)
IMPLEMENTATION_ENVIRONMENT_FIELDS = frozenset(
    {
        "cuda_available",
        "cuda_runtime",
        "device_name",
        "device_requested",
        "device_total_memory_bytes",
        "driver_version",
        "gmsh",
        "numpy",
        "platform",
        "python",
        "scipy",
        "skfem",
        "torch",
    }
)


class FormalAnalysisError(ValueError):
    """Raised internally when a required formal-evidence field is invalid."""


@dataclass(frozen=True)
class CheckpointMetric:
    """Authenticated per-case errors for one method/fraction/seed."""

    key: str
    method: str
    fraction: float
    seed: int
    checkpoint_sha256: str
    case_errors: FloatArray


@dataclass(frozen=True)
class PartitionMetric:
    """Validated sealed metrics for one locked partition."""

    name: str
    case_group_ids: tuple[str, ...]
    geometry_group_ids: tuple[str, ...]
    section_families: tuple[str, ...]
    load_subtypes: tuple[str, ...]
    coarse_errors: FloatArray
    checkpoints: Mapping[tuple[str, float, int], CheckpointMetric]
    wall_offset: Mapping[str, Any]


def canonical_sha256(value: Any) -> str:
    """Return the frozen UTF-8 canonical-JSON digest used by the formal run."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalAnalysisError(f"value is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_file_sha256(value: Any) -> str:
    """Digest the runner's canonical JSON file encoding, including its newline."""

    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FormalAnalysisError(f"value is not canonical-JSON serializable: {exc}") from exc
    return hashlib.sha256(payload + b"\n").hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value).issubset(SHA256_HEX)


def _is_git_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value).issubset(GIT_HEX)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FormalAnalysisError(f"{name} must be an object")
    return value


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FormalAnalysisError(f"{name} must be an array")
    return value


def _require_finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalAnalysisError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise FormalAnalysisError(f"{name} is outside its finite range")
    return result


def _require_integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise FormalAnalysisError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalAnalysisError(f"{name} must be an integer") from exc
    if result != value or (minimum is not None and result < minimum):
        raise FormalAnalysisError(f"{name} is outside its integer range")
    return result


def _require_float_array(
    value: Any,
    name: str,
    *,
    length: int | None = None,
    nonnegative: bool = True,
) -> FloatArray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise FormalAnalysisError(f"{name} must be a numeric array") from exc
    if result.ndim != 1 or (length is not None and result.size != length):
        raise FormalAnalysisError(f"{name} has the wrong one-dimensional shape")
    if not np.isfinite(result).all() or (nonnegative and np.any(result < 0.0)):
        raise FormalAnalysisError(f"{name} contains invalid or non-finite values")
    result = result.copy()
    result.setflags(write=False)
    return result


def _checkpoint_key(method: str, fraction: float, seed: int) -> str:
    return f"{method}__f{round(float(fraction) * 100):03d}__seed{int(seed)}"


def _expected_checkpoint_specs(config: Mapping[str, Any]) -> tuple[tuple[str, float, int], ...]:
    learning = _require_mapping(config.get("learning"), "config.learning")
    matrix = _require_mapping(
        learning.get("method_fraction_matrix"), "config.learning.method_fraction_matrix"
    )
    seeds = tuple(
        _require_integer(value, "training seed")
        for value in _require_sequence(
            learning.get("training_seeds"), "config.learning.training_seeds"
        )
    )
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise FormalAnalysisError("formal analysis requires exactly five unique training seeds")
    expected_matrix = {
        "scratch": (1.0,),
        "direct_coarse": (1.0,),
        "residual_coarse": (0.25, 0.5, 0.75, 1.0),
        "mismatched_coarse": (0.5,),
    }
    normalized = {
        str(method): tuple(float(value) for value in _require_sequence(fractions, str(method)))
        for method, fractions in matrix.items()
    }
    if normalized != expected_matrix:
        raise FormalAnalysisError(
            "method/fraction matrix differs from the frozen 7-per-seed design"
        )
    result = tuple(
        (method, fraction, seed)
        for seed in seeds
        for method, fractions in expected_matrix.items()
        for fraction in fractions
    )
    if len(result) != 35 or _require_integer(
        learning.get("expected_checkpoint_count"), "expected checkpoint count"
    ) != len(result):
        raise FormalAnalysisError("the frozen checkpoint count must be 35")
    return result


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "tunnelgeopt.multifidelity.formal.v1":
        raise FormalAnalysisError("unsupported formal config schema")
    sections = tuple(
        str(value)
        for value in _require_sequence(
            _require_mapping(config.get("geometry"), "config.geometry").get("section_families"),
            "config.geometry.section_families",
        )
    )
    if len(sections) != 3 or len(set(sections)) != 3 or any(not value for value in sections):
        raise FormalAnalysisError("formal analysis requires three unique section families")
    bootstrap = _require_mapping(
        _require_mapping(config.get("evaluation"), "config.evaluation").get("bootstrap"),
        "config.evaluation.bootstrap",
    )
    if (
        _require_integer(bootstrap.get("replicates"), "bootstrap replicates") != 20_000
        or bootstrap.get("paired") is not True
        or tuple(bootstrap.get("levels", ())) != ("training_seed", "parent_geometry_within_section")
        or bootstrap.get("all_parent_loads_remain_together") is not True
        or _require_finite(
            bootstrap.get("one_sided_upper_confidence_level"), "one-sided confidence"
        )
        != 0.95
        or _require_finite(bootstrap.get("two_sided_interval_level"), "interval level") != 0.95
        or _require_finite(
            bootstrap.get("max_primary_ratio_interval_total_width"), "max interval width"
        )
        != 0.10
    ):
        raise FormalAnalysisError("bootstrap contract differs from the frozen 20k paired design")
    specs = _expected_checkpoint_specs(config)
    if tuple(dict.fromkeys(spec[2] for spec in specs)) != (103, 211, 307, 401, 509):
        raise FormalAnalysisError("training seeds differ from the frozen five-seed design")
    gates = _require_mapping(config.get("scientific_decision"), "scientific_decision")
    if tuple(_require_mapping(gates.get("upper_95_ci_gates"), "upper CI gates")) != (
        *PRIMARY_PARTITIONS,
    ):
        raise FormalAnalysisError("primary partition gate set or order changed")
    if gates["upper_95_ci_gates"] != {
        "locked_iid": {"R_s": 1.02, "R_d": 1.02, "R_c": 0.70},
        "locked_geometry_ood": {"R_s": 1.05, "R_d": 1.05, "R_c": 0.80},
        "locked_load_ood": {"R_s": 1.05, "R_d": 1.05, "R_c": 0.80},
    }:
        raise FormalAnalysisError("primary upper-95 gates differ from the frozen thresholds")
    if gates.get("seed_stability") != {
        "minimum_passing_seeds": 4,
        "total_seeds": 5,
        "iid_max_R_s_and_R_d": 1.05,
        "geometry_ood_max_R_s_and_R_d": 1.10,
        "load_ood_max_R_s_and_R_d": 1.10,
    }:
        raise FormalAnalysisError("seed-stability gates differ from the frozen thresholds")
    section_gates = _require_mapping(gates.get("section_robustness"), "section robustness gates")
    for key, expected in {
        "iid_max_any_section": 1.10,
        "iid_max_for_at_least_two_sections": 1.02,
        "minimum_iid_sections_at_strict_gate": 2,
        "ood_subtype_max_point_ratio_to_each_full_label_baseline": 1.15,
    }.items():
        if section_gates.get(key) != expected:
            raise FormalAnalysisError(f"section robustness gate {key} changed")
    partitions = _require_mapping(
        _require_mapping(config.get("dataset"), "config.dataset").get("partitions"),
        "config.dataset.partitions",
    )
    if tuple(partitions) != REQUIRED_DATA_PARTITIONS:
        raise FormalAnalysisError("dataset partition set or order changed")
    quality = _require_mapping(config.get("quality_control"), "quality_control")
    solver = _require_mapping(quality.get("solver_and_mesh"), "solver_and_mesh")
    frozen_solver = {
        "max_nonfinite_fraction": 0.0,
        "max_free_dof_algebraic_residual": 1e-9,
        "max_clapeyron_relative_energy_error": 1e-9,
        "min_triangle_signed_area_over_radius_squared": 1e-12,
        "min_triangle_quality": 0.02,
        "minimum_valid_case_fraction_per_partition_section": 0.95,
    }
    if any(solver.get(key) != value for key, value in frozen_solver.items()) or tuple(
        solver.get("required_fidelities_per_formal_case", ())
    ) != ("coarse", "fine"):
        raise FormalAnalysisError("solver/mesh QC thresholds differ from the frozen contract")
    fine_ultrafine = _require_mapping(quality.get("fine_ultrafine"), "fine_ultrafine")
    if any(
        fine_ultrafine.get(key) != value
        for key, value in {
            "formal_audit_fraction": 0.2,
            "minimum_selected_cases_per_partition_section": 3,
            "expected_formal_audit_cases": 144,
            "max_overall_median": 0.03,
            "max_overall_p95": 0.05,
            "max_any_section_median": 0.04,
        }.items()
    ):
        raise FormalAnalysisError("fine-ultrafine thresholds differ from the frozen contract")
    wall = _require_mapping(
        _require_mapping(config.get("evaluation"), "evaluation").get("wall_offset_physics"),
        "wall_offset_physics",
    )
    if wall.get("coarse_nonworsening") != {
        "max_multiplier": 1.1,
        "traction_additive_margin_over_S_inf": 0.005,
        "resultant_additive_margin_over_S_inf": 0.0025,
        "traction_gate_formula": "D_t(M,F)<=1.10*D_t(C,F)+0.005",
        "resultant_gate_formula": "D_r(M,F)<=1.10*D_r(C,F)+0.0025",
    } or wall.get("absolute_caps") != {
        "locked_iid": {
            "max_traction_discrepancy": 0.10,
            "max_resultant_discrepancy": 0.05,
        },
        "locked_geometry_ood": {
            "max_traction_discrepancy": 0.15,
            "max_resultant_discrepancy": 0.08,
        },
        "locked_load_ood": {
            "max_traction_discrepancy": 0.15,
            "max_resultant_discrepancy": 0.08,
        },
        "locked_joint_ood_report_only": {
            "max_traction_discrepancy": 0.20,
            "max_resultant_discrepancy": 0.10,
        },
    }:
        raise FormalAnalysisError("wall-offset physics thresholds differ from the frozen contract")
    return {
        "sections": sections,
        "specs": specs,
        "seeds": tuple(dict.fromkeys(spec[2] for spec in specs)),
        "config_sha256": canonical_sha256(config),
        "bootstrap": bootstrap,
        "gates": gates,
        "partitions": partitions,
    }


def _validate_hashes_and_access(
    config: Mapping[str, Any],
    sealed_metrics: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    access_state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    digest = metadata["config_sha256"]
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise FormalAnalysisError("config.run_id is missing")
    for name, value in (
        ("sealed_metrics", sealed_metrics),
        ("dataset_manifest", dataset_manifest),
        ("access_state", access_state),
    ):
        if value.get("run_id") != run_id or value.get("config_sha256") != digest:
            raise FormalAnalysisError(f"{name} is not bound to the run/config hash")
    if (
        sealed_metrics.get("backend") != "formal"
        or sealed_metrics.get("effect_claim_allowed") is not True
    ):
        raise FormalAnalysisError("sealed metrics are not formal-effect evidence")

    hashes = _require_mapping(access_state.get("hashes"), "access_state.hashes")
    computed = {
        "config_canonical_sha256": digest,
        "dataset_manifest_canonical_sha256": canonical_sha256(dataset_manifest),
        "sealed_metrics_canonical_sha256": canonical_sha256(sealed_metrics),
    }
    for key, expected in computed.items():
        if hashes.get(key) != expected:
            raise FormalAnalysisError(f"{key} mismatch")
    for key in (
        "config_file_sha256",
        "dataset_manifest_file_sha256",
        "sealed_metrics_file_sha256",
        "access_log_file_sha256",
        "checkpoint_manifest_file_sha256",
        "checkpoint_registry_file_sha256",
        "implementation_manifest_file_sha256",
        "prepare_manifest_file_sha256",
    ):
        if not _is_sha256(hashes.get(key)):
            raise FormalAnalysisError(f"{key} is missing or invalid")

    implementation = _require_mapping(
        access_state.get("implementation_manifest"), "access_state.implementation_manifest"
    )
    expected_implementation_fields = {
        "schema",
        "run_id",
        "config_sha256",
        "effect_claim_allowed",
        "recorded_at_utc",
        "source_provenance",
        "environment",
    }
    if set(implementation) != expected_implementation_fields:
        raise FormalAnalysisError("implementation manifest field set changed")
    if (
        implementation.get("schema") != "tunnelgeopt.formal_implementation_manifest.v1"
        or implementation.get("run_id") != run_id
        or implementation.get("config_sha256") != digest
        or implementation.get("effect_claim_allowed") is not True
        or not isinstance(implementation.get("recorded_at_utc"), str)
        or not str(implementation["recorded_at_utc"]).strip()
    ):
        raise FormalAnalysisError("implementation manifest is not formal-run bound")
    if _canonical_json_file_sha256(implementation) != hashes["implementation_manifest_file_sha256"]:
        raise FormalAnalysisError("implementation manifest file digest mismatch")

    provenance = _require_mapping(
        implementation.get("source_provenance"), "implementation source provenance"
    )
    expected_provenance_fields = {
        "git_head",
        "upstream_ref",
        "upstream_head",
        "head_matches_upstream",
        "worktree_clean_before_prepare",
        "remote_url_sanitized",
        "all_sources_tracked",
        "source_sha256",
    }
    if set(provenance) != expected_provenance_fields:
        raise FormalAnalysisError("source provenance field set changed")
    git_head = provenance.get("git_head")
    if (
        not _is_git_commit(git_head)
        or not _is_git_commit(provenance.get("upstream_head"))
        or git_head != provenance.get("upstream_head")
        or provenance.get("head_matches_upstream") is not True
        or provenance.get("worktree_clean_before_prepare") is not True
        or provenance.get("all_sources_tracked") is not True
        or not isinstance(provenance.get("upstream_ref"), str)
        or not str(provenance["upstream_ref"]).strip()
        or not isinstance(provenance.get("remote_url_sanitized"), str)
        or not str(provenance["remote_url_sanitized"]).strip()
    ):
        raise FormalAnalysisError("source provenance is not a clean pushed Git revision")
    source_sha256 = _require_mapping(
        provenance.get("source_sha256"), "source provenance source_sha256"
    )
    if set(source_sha256) != IMPLEMENTATION_SOURCE_PATHS or any(
        not _is_sha256(value) for value in source_sha256.values()
    ):
        raise FormalAnalysisError("critical implementation source hash set is incomplete")
    if source_sha256["configs/multifidelity_formal.json"] != hashes["config_file_sha256"]:
        raise FormalAnalysisError("source provenance config file digest mismatch")

    environment = _require_mapping(implementation.get("environment"), "implementation environment")
    if set(environment) != IMPLEMENTATION_ENVIRONMENT_FIELDS:
        raise FormalAnalysisError("implementation environment field set changed")
    required_strings = IMPLEMENTATION_ENVIRONMENT_FIELDS - {
        "cuda_available",
        "device_total_memory_bytes",
    }
    if (
        environment.get("cuda_available") is not True
        or _require_integer(
            environment.get("device_total_memory_bytes"),
            "implementation device total memory",
            minimum=1,
        )
        < 1
        or any(
            not isinstance(environment.get(key), str) or not str(environment[key]).strip()
            for key in required_strings
        )
        or not str(environment["device_requested"]).startswith("cuda")
    ):
        raise FormalAnalysisError("formal CUDA implementation environment is incomplete")

    registry = _require_mapping(
        access_state.get("checkpoint_registry"), "access_state.checkpoint_registry"
    )
    if (
        registry.get("frozen") is not True
        or registry.get("config_sha256") != digest
        or _require_integer(registry.get("checkpoint_count"), "registry checkpoint count") != 35
        or not _is_sha256(registry.get("registry_hash"))
    ):
        raise FormalAnalysisError("checkpoint registry is not frozen and config-bound")
    records = _require_mapping(registry.get("checkpoints"), "checkpoint registry records")
    expected_keys = {
        _checkpoint_key(method, fraction, seed) for method, fraction, seed in metadata["specs"]
    }
    if set(records) != expected_keys:
        raise FormalAnalysisError("checkpoint registry does not contain the exact frozen design")
    checkpoint_digests: list[str] = []
    for key, raw in records.items():
        record = _require_mapping(raw, f"registry checkpoint {key}")
        if not _is_sha256(record.get("sha256")) or not _is_sha256(
            record.get("training_contract_sha256")
        ):
            raise FormalAnalysisError(f"checkpoint {key} lacks checkpoint/contract hashes")
        if record.get("config_sha256") != digest:
            raise FormalAnalysisError(f"checkpoint {key} is not config-bound")
        checkpoint_digests.append(str(record["sha256"]))
    if len(set(checkpoint_digests)) != 35:
        raise FormalAnalysisError("all formal checkpoint SHA-256 values must be unique")
    ordered_checkpoint_digests = [
        records[_checkpoint_key(method, fraction, seed)]["sha256"]
        for method, fraction, seed in metadata["specs"]
    ]
    expected_registry_hash = canonical_sha256(
        {
            "identity": "tunnelgeopt.checkpoint_registry.v1",
            "checkpoint_ids": ordered_checkpoint_digests,
        }
    )
    if registry.get("registry_hash") != expected_registry_hash:
        raise FormalAnalysisError("checkpoint registry hash does not authenticate its ordered IDs")
    if sealed_metrics.get("registry_hash") != expected_registry_hash:
        raise FormalAnalysisError("sealed metrics are not bound to the frozen checkpoint registry")

    if (
        access_state.get("config_frozen_before_generation") is not True
        or access_state.get("locked_labels_opened_before_checkpoint_freeze") is not False
        or access_state.get("locked_labels_used_for_tuning") is not False
        or access_state.get("trainer_received_locked_label_path") is not False
        or access_state.get("access_log_append_only") is not True
        or _require_integer(
            access_state.get("denied_premature_sealed_accesses"),
            "denied premature access count",
            minimum=0,
        )
        != 0
    ):
        raise FormalAnalysisError("sealed-label leakage/access preconditions failed")

    open_counts = _require_mapping(
        access_state.get("sealed_partition_open_counts"), "sealed partition open counts"
    )
    if set(open_counts) != set(LOCKED_PARTITIONS) or any(
        _require_integer(value, "sealed open count") != 1 for value in open_counts.values()
    ):
        raise FormalAnalysisError("each locked partition must be opened exactly once")
    evaluation_counts = _require_mapping(
        access_state.get("checkpoint_evaluation_counts"), "checkpoint evaluation counts"
    )
    expected_evaluations = {
        f"{partition}:{key}" for partition in LOCKED_PARTITIONS for key in expected_keys
    }
    if set(evaluation_counts) != expected_evaluations or any(
        _require_integer(value, "checkpoint evaluation count") != 1
        for value in evaluation_counts.values()
    ):
        raise FormalAnalysisError("every checkpoint/locked-partition pair must be evaluated once")
    return {
        "registry": registry,
        "registry_records": records,
        "run_id": run_id,
        "implementation_manifest": implementation,
    }


def _solver_record_valid(record: Mapping[str, Any], thresholds: Mapping[str, Any]) -> bool:
    required_fidelities = tuple(thresholds["required_fidelities_per_formal_case"])
    fidelities = _require_mapping(record.get("fidelities"), "solver record fidelities")
    if set(fidelities) != set(required_fidelities):
        return False
    for fidelity in required_fidelities:
        metrics = _require_mapping(fidelities[fidelity], f"{fidelity} solver metrics")
        if (
            _require_finite(metrics.get("nonfinite_fraction"), "nonfinite fraction", minimum=0)
            > float(thresholds["max_nonfinite_fraction"])
            or _require_finite(
                metrics.get("free_dof_algebraic_residual"), "algebraic residual", minimum=0
            )
            > float(thresholds["max_free_dof_algebraic_residual"])
            or _require_finite(
                metrics.get("clapeyron_relative_energy_error"), "energy error", minimum=0
            )
            > float(thresholds["max_clapeyron_relative_energy_error"])
            or _require_finite(
                metrics.get("min_triangle_signed_area_over_radius_squared"),
                "triangle signed area",
            )
            < float(thresholds["min_triangle_signed_area_over_radius_squared"])
            or _require_finite(metrics.get("min_triangle_quality"), "triangle quality")
            < float(thresholds["min_triangle_quality"])
            or metrics.get("explicit_wall_and_farfield_tags") is not True
            or metrics.get("no_element_centroid_inside_cavity") is not True
            or metrics.get("same_boundary_hash_and_outer_bounds") is not True
            or metrics.get("all_query_points_located") is not True
        ):
            return False
    return record.get("valid") is True


def _validate_dataset_qc(
    config: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    sealed_metrics: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    identities = _require_mapping(dataset_manifest.get("identities"), "manifest.identities")
    required_identity_true = (
        "cross_partition_zero_intersection",
        "legacy_v0_2_locked_test_zero_intersection",
        "normalization_fit_train_only",
        "no_result_conditioned_replacement",
    )
    if any(identities.get(key) is not True for key in required_identity_true):
        raise FormalAnalysisError("dataset identity/split/leakage audit failed")
    artifact_hashes = _require_mapping(
        dataset_manifest.get("artifact_hashes"), "manifest.artifact_hashes"
    )
    required_artifact_hashes = {
        "geometry_manifest",
        "case_manifest",
        "query_manifest",
        "public_input_store",
        "train_dev_label_store",
        *(f"sealed_{partition}_label_store" for partition in LOCKED_PARTITIONS),
    }
    if set(artifact_hashes) < required_artifact_hashes or any(
        not _is_sha256(artifact_hashes[key]) for key in required_artifact_hashes
    ):
        raise FormalAnalysisError("dataset artifact hash set is incomplete")

    quality = _require_mapping(dataset_manifest.get("solver_mesh_qc"), "manifest.solver_mesh_qc")
    if quality.get("no_silent_case_replacement") is not True:
        raise FormalAnalysisError("silent solver-case replacement was not excluded")
    records = _require_sequence(quality.get("records"), "solver_mesh_qc.records")
    thresholds = _require_mapping(
        _require_mapping(config.get("quality_control"), "quality_control").get("solver_and_mesh"),
        "quality_control.solver_and_mesh",
    )
    seen: set[str] = set()
    valid_case_ids: dict[str, list[str]] = {name: [] for name in REQUIRED_DATA_PARTITIONS}
    invalid_cases: list[dict[str, str]] = []
    grouped: dict[tuple[str, str], list[bool]] = {}
    sections = set(metadata["sections"])
    expected_counts = {
        name: _require_integer(spec["cases"], f"{name} expected cases")
        for name, spec in metadata["partitions"].items()
    }
    observed_counts = {name: 0 for name in REQUIRED_DATA_PARTITIONS}
    for raw in records:
        record = _require_mapping(raw, "solver_mesh_qc record")
        case_id = str(record.get("case_group_id", ""))
        partition = str(record.get("partition", ""))
        section = str(record.get("section_family", ""))
        if (
            not case_id
            or case_id in seen
            or partition not in expected_counts
            or section not in sections
        ):
            raise FormalAnalysisError("solver QC record identity is invalid or duplicated")
        seen.add(case_id)
        observed_counts[partition] += 1
        valid = _solver_record_valid(record, thresholds)
        grouped.setdefault((partition, section), []).append(valid)
        if valid:
            valid_case_ids[partition].append(case_id)
        else:
            invalid_cases.append(
                {
                    "case_group_id": case_id,
                    "partition": partition,
                    "section_family": section,
                }
            )
    if observed_counts != expected_counts:
        raise FormalAnalysisError("solver QC records do not cover every planned formal case")
    expected_groups = {
        (partition, section)
        for partition in REQUIRED_DATA_PARTITIONS
        for section in metadata["sections"]
    }
    if set(grouped) != expected_groups:
        raise FormalAnalysisError("solver QC lacks a partition-by-section cell")
    minimum_valid = float(thresholds["minimum_valid_case_fraction_per_partition_section"])
    valid_rates = {
        f"{partition}:{section}": float(np.mean(values))
        for (partition, section), values in sorted(grouped.items())
    }
    if any(value < minimum_valid for value in valid_rates.values()):
        raise FormalAnalysisError("solver/mesh valid fraction is below 95% in a partition-section")

    selection = _require_mapping(
        dataset_manifest.get("fine_ultrafine_selection"),
        "manifest.fine_ultrafine_selection",
    )
    audit_config = _require_mapping(
        _require_mapping(config["quality_control"], "quality_control").get("fine_ultrafine"),
        "quality_control.fine_ultrafine",
    )
    if (
        selection.get("selected_before_any_ultrafine_label") is not True
        or selection.get("case_values_exposed_before_checkpoint_freeze") is not False
        or selection.get("selection_protocol") != audit_config["selection_protocol"]
        or selection.get("selection_unit") != audit_config["selection_unit"]
        or _require_finite(selection.get("formal_audit_fraction"), "audit fraction", minimum=0)
        != float(audit_config["formal_audit_fraction"])
    ):
        raise FormalAnalysisError("fine-ultrafine selection/access contract failed")
    selected_ids = tuple(
        str(value)
        for value in _require_sequence(selection.get("selected_case_ids"), "selected case IDs")
    )
    expected_selected = int(audit_config["expected_formal_audit_cases"])
    if (
        len(selected_ids) != len(set(selected_ids))
        or len(selected_ids) != expected_selected
        or not set(selected_ids).issubset(
            {case_id for values in valid_case_ids.values() for case_id in values}
        )
    ):
        raise FormalAnalysisError("fine-ultrafine audit selection is incomplete or duplicated")

    audit = _require_mapping(
        sealed_metrics.get("fine_ultrafine_audit"), "sealed_metrics.fine_ultrafine_audit"
    )
    audit_ids = tuple(
        str(value) for value in _require_sequence(audit.get("case_group_ids"), "audit case IDs")
    )
    audit_sections = tuple(
        str(value) for value in _require_sequence(audit.get("section_families"), "audit sections")
    )
    audit_partitions = tuple(
        str(value) for value in _require_sequence(audit.get("partitions"), "audit partitions")
    )
    errors = _require_float_array(audit.get("relative_errors"), "fine-ultrafine errors")
    if (
        audit_ids != selected_ids
        or len(audit_sections) != len(audit_ids)
        or len(audit_partitions) != len(audit_ids)
        or errors.size != len(audit_ids)
        or set(audit_sections) != sections
        or not set(audit_partitions).issubset(expected_counts)
    ):
        raise FormalAnalysisError("sealed fine-ultrafine values do not match frozen selection")
    minimum_selected = int(audit_config["minimum_selected_cases_per_partition_section"])
    for partition in REQUIRED_DATA_PARTITIONS:
        for section in metadata["sections"]:
            count = sum(
                observed_partition == partition and observed_section == section
                for observed_partition, observed_section in zip(
                    audit_partitions, audit_sections, strict=True
                )
            )
            planned = int(metadata["partitions"][partition]["parents_per_section"]) * int(
                metadata["partitions"][partition]["loads_per_parent"]
            )
            expected_in_cell = max(
                minimum_selected,
                math.ceil(float(audit_config["formal_audit_fraction"]) * planned),
            )
            if count != expected_in_cell:
                raise FormalAnalysisError(
                    "fine-ultrafine selection does not match the frozen ceil rule in one cell"
                )
    overall_median = float(np.median(errors))
    overall_p95 = float(np.quantile(errors, 0.95))
    section_medians = {
        section: float(np.median(errors[np.asarray(audit_sections, dtype=object) == section]))
        for section in metadata["sections"]
    }
    if (
        overall_median > float(audit_config["max_overall_median"])
        or overall_p95 > float(audit_config["max_overall_p95"])
        or max(section_medians.values()) > float(audit_config["max_any_section_median"])
    ):
        raise FormalAnalysisError("fine-ultrafine 3%/5%/4% convergence gate failed")
    return {
        "_valid_case_ids_by_partition": {
            key: tuple(values) for key, values in valid_case_ids.items()
        },
        "invalid_cases": invalid_cases,
        "valid_case_fraction_by_partition_section": valid_rates,
        "fine_ultrafine": {
            "selected_case_count": len(audit_ids),
            "overall_median": overall_median,
            "overall_p95": overall_p95,
            "section_medians": section_medians,
        },
    }


def _parse_partition(
    name: str,
    raw: Mapping[str, Any],
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    registry_records: Mapping[str, Any],
    valid_case_ids: Sequence[str],
) -> PartitionMetric:
    case_ids = tuple(
        str(value) for value in _require_sequence(raw.get("case_group_ids"), "case IDs")
    )
    geometry_ids = tuple(
        str(value) for value in _require_sequence(raw.get("geometry_group_ids"), "geometry IDs")
    )
    sections = tuple(
        str(value) for value in _require_sequence(raw.get("section_families"), "sections")
    )
    subtypes = tuple(
        str(value) for value in _require_sequence(raw.get("load_subtypes"), "load subtypes")
    )
    size = len(case_ids)
    expected_case_ids = set(valid_case_ids)
    if (
        size != len(expected_case_ids)
        or set(case_ids) != expected_case_ids
        or len(set(case_ids)) != size
        or any(not value for value in case_ids)
        or any(len(values) != size for values in (geometry_ids, sections, subtypes))
        or set(sections) != set(metadata["sections"])
    ):
        raise FormalAnalysisError(f"{name} case identities/slices do not match the config")
    geometry_sections: dict[str, str] = {}
    for geometry, section in zip(geometry_ids, sections, strict=True):
        previous = geometry_sections.setdefault(geometry, section)
        if previous != section:
            raise FormalAnalysisError(f"{name} parent geometry crosses section families")
    coarse = _require_float_array(
        raw.get("coarse_only_case_errors"), f"{name} coarse errors", length=size
    )
    if np.any(coarse <= 0.0):
        raise FormalAnalysisError(f"{name} coarse reference contains zero error")
    checkpoint_records = _require_mapping(raw.get("checkpoints"), f"{name} checkpoints")
    expected_keys = {
        _checkpoint_key(method, fraction, seed) for method, fraction, seed in metadata["specs"]
    }
    if set(checkpoint_records) != expected_keys:
        raise FormalAnalysisError(f"{name} checkpoint metrics are incomplete")
    parsed: dict[tuple[str, float, int], CheckpointMetric] = {}
    for key, value in checkpoint_records.items():
        record = _require_mapping(value, f"{name}:{key}")
        method = str(record.get("method", ""))
        fraction = _require_finite(record.get("fine_fraction"), f"{key} fraction", minimum=0)
        seed = _require_integer(record.get("seed"), f"{key} seed")
        identity = (method, fraction, seed)
        if identity not in metadata["specs"] or key != _checkpoint_key(*identity):
            raise FormalAnalysisError(f"{name}:{key} method/fraction/seed identity changed")
        checkpoint_digest = str(record.get("checkpoint_sha256", ""))
        if checkpoint_digest != registry_records[key]["sha256"]:
            raise FormalAnalysisError(f"{name}:{key} checkpoint digest is unauthenticated")
        if (
            _require_integer(
                record.get("nonfinite_prediction_count"), f"{key} nonfinite count", minimum=0
            )
            != 0
        ):
            raise FormalAnalysisError(f"{name}:{key} has non-finite predictions")
        errors = _require_float_array(
            record.get("case_errors"), f"{name}:{key} case errors", length=size
        )
        parsed[identity] = CheckpointMetric(
            key=key,
            method=method,
            fraction=fraction,
            seed=seed,
            checkpoint_sha256=checkpoint_digest,
            case_errors=errors,
        )
    if any(
        _require_integer(value, f"{name} evaluation count") != 1
        for value in _require_mapping(
            raw.get("checkpoint_evaluation_counts"), f"{name} evaluation counts"
        ).values()
    ):
        raise FormalAnalysisError(f"{name} includes a repeated checkpoint evaluation")
    if set(raw["checkpoint_evaluation_counts"]) != expected_keys:
        raise FormalAnalysisError(f"{name} checkpoint evaluation count set changed")
    wall = _require_mapping(raw.get("wall_offset_physics"), f"{name} wall-offset physics")
    return PartitionMetric(
        name=name,
        case_group_ids=case_ids,
        geometry_group_ids=geometry_ids,
        section_families=sections,
        load_subtypes=subtypes,
        coarse_errors=coarse,
        checkpoints=parsed,
        wall_offset=wall,
    )


def _validate_resources(
    sealed_metrics: Mapping[str, Any], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    resources = _require_mapping(sealed_metrics.get("resource_usage"), "resource_usage")
    generation = _require_mapping(resources.get("generation"), "resource_usage.generation")
    training = _require_mapping(resources.get("training"), "resource_usage.training")
    evaluation = _require_mapping(resources.get("evaluation"), "resource_usage.evaluation")
    expected_keys = {
        _checkpoint_key(method, fraction, seed) for method, fraction, seed in metadata["specs"]
    }
    if set(training) != expected_keys or set(evaluation) != set(LOCKED_PARTITIONS):
        raise FormalAnalysisError("runtime/memory reports do not cover training and evaluation")

    def one(value: Any, name: str) -> dict[str, float | int]:
        item = _require_mapping(value, name)
        return {
            "runtime_seconds": _require_finite(
                item.get("runtime_seconds"), f"{name} runtime", minimum=0
            ),
            "peak_memory_bytes": _require_integer(
                item.get("peak_memory_bytes"), f"{name} peak memory", minimum=0
            ),
        }

    return {
        "generation": one(generation, "generation"),
        "training": {key: one(value, f"training {key}") for key, value in training.items()},
        "evaluation": {key: one(value, f"evaluation {key}") for key, value in evaluation.items()},
    }


def _aggregate_parent_section(
    values: FloatArray,
    geometry_ids: Sequence[str],
    sections: Sequence[str],
    section_order: Sequence[str],
) -> tuple[float, dict[str, float], tuple[str, ...], tuple[str, ...], FloatArray]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != len(geometry_ids):
        raise FormalAnalysisError("case values do not align as [seed, case]")
    parents = tuple(dict.fromkeys(str(value) for value in geometry_ids))
    parent_sections: list[str] = []
    parent_values = np.empty((array.shape[0], len(parents)), dtype=np.float64)
    geometry_array = np.asarray(geometry_ids, dtype=object)
    section_array = np.asarray(sections, dtype=object)
    for index, parent in enumerate(parents):
        mask = geometry_array == parent
        observed_sections = {str(value) for value in section_array[mask]}
        if len(observed_sections) != 1:
            raise FormalAnalysisError("one parent geometry belongs to multiple sections")
        parent_sections.append(observed_sections.pop())
        parent_values[:, index] = array[:, mask].mean(axis=1)
    section_means: dict[str, float] = {}
    parent_section_array = np.asarray(parent_sections, dtype=object)
    for section in section_order:
        mask = parent_section_array == section
        if not np.any(mask):
            raise FormalAnalysisError(f"aggregation slice lacks section {section}")
        section_means[section] = float(parent_values[:, mask].mean())
    return (
        float(np.mean(tuple(section_means.values()))),
        section_means,
        parents,
        tuple(parent_sections),
        parent_values,
    )


def _bootstrap_ratio(
    candidate: FloatArray,
    reference: FloatArray,
    geometry_ids: Sequence[str],
    sections: Sequence[str],
    section_order: Sequence[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    candidate_center, _, parents, parent_sections, candidate_parent = _aggregate_parent_section(
        candidate, geometry_ids, sections, section_order
    )
    reference_center, _, reference_parents, reference_sections, reference_parent = (
        _aggregate_parent_section(reference, geometry_ids, sections, section_order)
    )
    if parents != reference_parents or parent_sections != reference_sections:
        raise FormalAnalysisError("paired bootstrap parent identities do not align")
    if candidate_parent.shape != reference_parent.shape or candidate_parent.shape[0] != 5:
        raise FormalAnalysisError("paired bootstrap requires five aligned training seeds")
    if np.any(reference_parent <= 0.0):
        raise FormalAnalysisError("bootstrap reference error must be positive")
    parent_sections_array = np.asarray(parent_sections, dtype=object)
    section_indices = [
        np.flatnonzero(parent_sections_array == section) for section in section_order
    ]
    rng = np.random.default_rng(seed)
    ratios = np.empty(replicates, dtype=np.float64)
    chunk_size = 2_000
    cursor = 0
    while cursor < replicates:
        size = min(chunk_size, replicates - cursor)
        sampled_seeds = rng.integers(0, 5, size=(size, 5))
        candidate_sections = []
        reference_sections_draw = []
        for indices in section_indices:
            sampled_parents = indices[rng.integers(0, len(indices), size=(size, len(indices)))]
            candidate_sections.append(
                candidate_parent[sampled_seeds[:, :, None], sampled_parents[:, None, :]].mean(
                    axis=(1, 2)
                )
            )
            reference_sections_draw.append(
                reference_parent[sampled_seeds[:, :, None], sampled_parents[:, None, :]].mean(
                    axis=(1, 2)
                )
            )
        numerator = np.stack(candidate_sections, axis=1).mean(axis=1)
        denominator = np.stack(reference_sections_draw, axis=1).mean(axis=1)
        if np.any(denominator <= 0.0):
            raise FormalAnalysisError("bootstrap produced a zero reference denominator")
        ratios[cursor : cursor + size] = numerator / denominator
        cursor += size
    return {
        "center_ratio": candidate_center / reference_center,
        "lower_95": float(np.quantile(ratios, 0.025)),
        "upper_95": float(np.quantile(ratios, 0.975)),
        "one_sided_upper_95": float(np.quantile(ratios, 0.95)),
        "interval_width": float(np.quantile(ratios, 0.975) - np.quantile(ratios, 0.025)),
        "replicates": int(replicates),
    }


def _comparison_seed(partition: str, ratio: str) -> int:
    payload = f"tunnelgeopt.formal_analysis.v1:{partition}:{ratio}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _method_matrix(
    partition: PartitionMetric, metadata: Mapping[str, Any]
) -> dict[tuple[str, float], FloatArray]:
    result: dict[tuple[str, float], FloatArray] = {}
    for method, fraction, _ in metadata["specs"]:
        key = (method, fraction)
        if key in result:
            continue
        result[key] = np.stack(
            [
                partition.checkpoints[(method, fraction, seed)].case_errors
                for seed in metadata["seeds"]
            ]
        )
    return result


def _summary(values: FloatArray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _slice_ratio(
    candidate: FloatArray,
    reference: FloatArray,
    partition: PartitionMetric,
    metadata: Mapping[str, Any],
    mask: NDArray[np.bool_] | None = None,
    *,
    section_order: Sequence[str] | None = None,
) -> float:
    if mask is None:
        mask = np.ones(len(partition.case_group_ids), dtype=bool)
    geometry = tuple(
        value for value, keep in zip(partition.geometry_group_ids, mask, strict=True) if keep
    )
    sections = tuple(
        value for value, keep in zip(partition.section_families, mask, strict=True) if keep
    )
    aggregation_sections = tuple(section_order or metadata["sections"])
    numerator, *_ = _aggregate_parent_section(
        candidate[:, mask], geometry, sections, aggregation_sections
    )
    denominator, *_ = _aggregate_parent_section(
        reference[:, mask], geometry, sections, aggregation_sections
    )
    if denominator <= 0.0:
        raise FormalAnalysisError("slice reference error is zero")
    return numerator / denominator


def _wall_aggregate(
    values: Any,
    partition: PartitionMetric,
    metadata: Mapping[str, Any],
    name: str,
    *,
    seeded: bool,
) -> tuple[float, float]:
    payload = _require_mapping(values, name)
    if seeded:
        by_seed = _require_mapping(payload.get("by_seed"), f"{name}.by_seed")
        if set(by_seed) != {str(seed) for seed in metadata["seeds"]}:
            raise FormalAnalysisError(f"{name} does not cover all five seeds")
        traction = np.stack(
            [
                _require_float_array(
                    _require_mapping(by_seed[str(seed)], f"{name}:{seed}").get(
                        "traction_discrepancy_by_case"
                    ),
                    f"{name}:{seed}:traction",
                    length=len(partition.case_group_ids),
                )
                for seed in metadata["seeds"]
            ]
        )
        resultant = np.stack(
            [
                _require_float_array(
                    _require_mapping(by_seed[str(seed)], f"{name}:{seed}").get(
                        "resultant_discrepancy_by_case"
                    ),
                    f"{name}:{seed}:resultant",
                    length=len(partition.case_group_ids),
                )
                for seed in metadata["seeds"]
            ]
        )
    else:
        traction = _require_float_array(
            payload.get("traction_discrepancy_by_case"),
            f"{name}:traction",
            length=len(partition.case_group_ids),
        )
        resultant = _require_float_array(
            payload.get("resultant_discrepancy_by_case"),
            f"{name}:resultant",
            length=len(partition.case_group_ids),
        )
    traction_value, *_ = _aggregate_parent_section(
        traction,
        partition.geometry_group_ids,
        partition.section_families,
        metadata["sections"],
    )
    resultant_value, *_ = _aggregate_parent_section(
        resultant,
        partition.geometry_group_ids,
        partition.section_families,
        metadata["sections"],
    )
    return traction_value, resultant_value


def _analyze_effects(
    config: Mapping[str, Any],
    partitions: Mapping[str, PartitionMetric],
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    effect_failures: list[dict[str, str]] = []
    results: dict[str, Any] = {}
    reports: dict[str, Any] = {"partitions": {}, "learning_curves": {}}
    gates = metadata["gates"]
    max_width = float(metadata["bootstrap"]["max_primary_ratio_interval_total_width"])
    validity_width_failures: list[dict[str, str]] = []

    for partition_name in LOCKED_PARTITIONS:
        partition = partitions[partition_name]
        matrix = _method_matrix(partition, metadata)
        residual = matrix[("residual_coarse", 0.5)]
        scratch = matrix[("scratch", 1.0)]
        direct = matrix[("direct_coarse", 1.0)]
        coarse = np.tile(partition.coarse_errors, (5, 1))
        references = {"R_s": scratch, "R_d": direct, "R_c": coarse}
        comparisons = {
            ratio: _bootstrap_ratio(
                residual,
                reference,
                partition.geometry_group_ids,
                partition.section_families,
                metadata["sections"],
                replicates=20_000,
                seed=_comparison_seed(partition_name, ratio),
            )
            for ratio, reference in references.items()
        }
        for ratio, comparison in comparisons.items():
            if partition_name in PRIMARY_PARTITIONS and comparison["interval_width"] > max_width:
                validity_width_failures.append(
                    {
                        "code": "BOOTSTRAP_INTERVAL_TOO_WIDE",
                        "detail": f"{partition_name}:{ratio} width={comparison['interval_width']:.6g}",
                    }
                )
        if partition_name in PRIMARY_PARTITIONS:
            thresholds = gates["upper_95_ci_gates"][partition_name]
            for ratio, threshold in thresholds.items():
                if comparisons[ratio]["one_sided_upper_95"] > float(threshold):
                    effect_failures.append(
                        {
                            "code": "PRIMARY_UPPER95_GATE_FAILED",
                            "detail": (
                                f"{partition_name}:{ratio} upper95="
                                f"{comparisons[ratio]['one_sided_upper_95']:.6g}>"
                                f"{float(threshold):.6g}"
                            ),
                        }
                    )

        seed_ratios: dict[str, list[float]] = {}
        for ratio, reference in references.items():
            values = []
            for seed_index in range(5):
                numerator, *_ = _aggregate_parent_section(
                    residual[seed_index],
                    partition.geometry_group_ids,
                    partition.section_families,
                    metadata["sections"],
                )
                denominator, *_ = _aggregate_parent_section(
                    reference[seed_index],
                    partition.geometry_group_ids,
                    partition.section_families,
                    metadata["sections"],
                )
                values.append(numerator / denominator)
            seed_ratios[ratio] = values

        section_ratios: dict[str, dict[str, float]] = {}
        section_array = np.asarray(partition.section_families, dtype=object)
        for section in metadata["sections"]:
            mask = section_array == section
            section_ratios[section] = {}
            for ratio, reference in references.items():
                section_ratios[section][ratio] = _slice_ratio(
                    residual,
                    reference,
                    partition,
                    metadata,
                    mask,
                    section_order=(section,),
                )

        wall_config = config["evaluation"]["wall_offset_physics"]
        coarse_wall = _wall_aggregate(
            partition.wall_offset.get("coarse_only"),
            partition,
            metadata,
            f"{partition_name}.coarse wall",
            seeded=False,
        )
        residual_wall = _wall_aggregate(
            partition.wall_offset.get("residual_coarse_0.5"),
            partition,
            metadata,
            f"{partition_name}.residual50 wall",
            seeded=True,
        )
        caps_key = (
            "locked_joint_ood_report_only"
            if partition_name == "locked_joint_ood"
            else partition_name
        )
        caps = wall_config["absolute_caps"][caps_key]
        nonworsening = wall_config["coarse_nonworsening"]
        wall_pass = (
            residual_wall[0] <= float(caps["max_traction_discrepancy"])
            and residual_wall[1] <= float(caps["max_resultant_discrepancy"])
            and residual_wall[0]
            <= float(nonworsening["max_multiplier"]) * coarse_wall[0]
            + float(nonworsening["traction_additive_margin_over_S_inf"])
            and residual_wall[1]
            <= float(nonworsening["max_multiplier"]) * coarse_wall[1]
            + float(nonworsening["resultant_additive_margin_over_S_inf"])
        )
        if partition_name in PRIMARY_PARTITIONS and not wall_pass:
            effect_failures.append(
                {
                    "code": "WALL_OFFSET_GATE_FAILED",
                    "detail": f"{partition_name}: residual50 Dt/Dr caps or coarse nonworsening failed",
                }
            )

        method_reports: dict[str, Any] = {"coarse_only": _summary(partition.coarse_errors)}
        learning_curve: list[dict[str, Any]] = []
        partition_section_array = np.asarray(partition.section_families, dtype=object)
        for (method, fraction), values in sorted(matrix.items()):
            aggregate, section_summary, *_ = _aggregate_parent_section(
                values,
                partition.geometry_group_ids,
                partition.section_families,
                metadata["sections"],
            )
            label = f"{method}@{fraction:g}"
            method_reports[label] = {
                **_summary(values),
                "parent_section_equal_mean": aggregate,
                "section_means": section_summary,
                "by_seed": {
                    str(seed): _summary(values[seed_index])
                    for seed_index, seed in enumerate(metadata["seeds"])
                },
                "by_section": {
                    section: _summary(values[:, partition_section_array == section])
                    for section in metadata["sections"]
                },
            }
            learning_curve.append(
                {
                    "method": method,
                    "fine_fraction": fraction,
                    "parent_section_equal_mean": aggregate,
                    **_summary(values),
                }
            )
        subtype_reports: dict[str, Any] = {}
        subtype_array = np.asarray(partition.load_subtypes, dtype=object)
        for subtype in sorted(set(partition.load_subtypes)):
            mask = subtype_array == subtype
            subtype_reports[subtype] = {
                "case_count": int(mask.sum()),
                "methods": {
                    f"{method}@{fraction:g}": _summary(values[:, mask])
                    for (method, fraction), values in sorted(matrix.items())
                },
                "coarse_only": _summary(partition.coarse_errors[mask]),
            }
        results[partition_name] = {
            "comparisons": comparisons,
            "seed_ratios": seed_ratios,
            "section_ratios": section_ratios,
            "wall_offset": {
                "coarse": {"D_t": coarse_wall[0], "D_r": coarse_wall[1]},
                "residual50": {"D_t": residual_wall[0], "D_r": residual_wall[1]},
                "gate_pass": wall_pass if partition_name in PRIMARY_PARTITIONS else None,
                "report_only": partition_name == "locked_joint_ood",
            },
            "load_subtypes": subtype_reports,
            "report_only": partition_name == "locked_joint_ood",
        }
        reports["partitions"][partition_name] = {
            "methods": method_reports,
            "by_section": section_ratios,
            "by_seed": seed_ratios,
            "by_load_subtype": subtype_reports,
        }
        reports["learning_curves"][partition_name] = learning_curve

    seed_config = gates["seed_stability"]
    seed_thresholds = {
        "locked_iid": float(seed_config["iid_max_R_s_and_R_d"]),
        "locked_geometry_ood": float(seed_config["geometry_ood_max_R_s_and_R_d"]),
        "locked_load_ood": float(seed_config["load_ood_max_R_s_and_R_d"]),
    }
    seed_gate: dict[str, Any] = {}
    for partition, threshold in seed_thresholds.items():
        passing = [
            int(seed)
            for index, seed in enumerate(metadata["seeds"])
            if results[partition]["seed_ratios"]["R_s"][index] <= threshold
            and results[partition]["seed_ratios"]["R_d"][index] <= threshold
        ]
        passed = len(passing) >= int(seed_config["minimum_passing_seeds"])
        seed_gate[partition] = {
            "threshold": threshold,
            "passing_seeds": passing,
            "passing_count": len(passing),
            "passed": passed,
        }
        if not passed:
            effect_failures.append(
                {
                    "code": "SEED_STABILITY_GATE_FAILED",
                    "detail": f"{partition}: only {len(passing)}/5 seeds pass same-seed R_s and R_d",
                }
            )

    section_config = gates["section_robustness"]
    iid_maxima = {
        section: max(values["R_s"], values["R_d"])
        for section, values in results["locked_iid"]["section_ratios"].items()
    }
    if any(value > float(section_config["iid_max_any_section"]) for value in iid_maxima.values()):
        effect_failures.append(
            {
                "code": "IID_SECTION_MAX_GATE_FAILED",
                "detail": "at least one IID section has max(R_s,R_d)>1.10",
            }
        )
    strict_count = sum(
        value <= float(section_config["iid_max_for_at_least_two_sections"])
        for value in iid_maxima.values()
    )
    if strict_count < int(section_config["minimum_iid_sections_at_strict_gate"]):
        effect_failures.append(
            {
                "code": "IID_SECTION_STRICT_COUNT_GATE_FAILED",
                "detail": f"only {strict_count}/3 IID sections satisfy max(R_s,R_d)<=1.02",
            }
        )

    load_partition = partitions["locked_load_ood"]
    load_matrix = _method_matrix(load_partition, metadata)
    residual = load_matrix[("residual_coarse", 0.5)]
    subtype_array = np.asarray(load_partition.load_subtypes, dtype=object)
    expected_subtypes = set(config["material_and_loads"]["load_ood_subtypes"])
    if set(load_partition.load_subtypes) != expected_subtypes:
        raise FormalAnalysisError("locked load-OOD does not report every frozen subtype")
    subtype_gate: dict[str, Any] = {}
    for subtype in sorted(expected_subtypes):
        mask = subtype_array == subtype
        if set(np.asarray(load_partition.section_families, dtype=object)[mask]) != set(
            metadata["sections"]
        ):
            raise FormalAnalysisError(f"load-OOD subtype {subtype} lacks a section family")
        ratios = {
            "to_scratch100": _slice_ratio(
                residual,
                load_matrix[("scratch", 1.0)],
                load_partition,
                metadata,
                mask,
            ),
            "to_direct100": _slice_ratio(
                residual,
                load_matrix[("direct_coarse", 1.0)],
                load_partition,
                metadata,
                mask,
            ),
        }
        passed = max(ratios.values()) <= float(
            section_config["ood_subtype_max_point_ratio_to_each_full_label_baseline"]
        )
        subtype_gate[subtype] = {**ratios, "passed": passed}
        if not passed:
            effect_failures.append(
                {
                    "code": "LOAD_OOD_SUBTYPE_GATE_FAILED",
                    "detail": f"{subtype}: residual50 point ratio exceeds 1.15",
                }
            )

    results["seed_stability"] = seed_gate
    results["iid_section_robustness"] = {
        "max_Rs_Rd": iid_maxima,
        "strict_section_count": strict_count,
    }
    results["load_ood_subtype_gate"] = subtype_gate
    return (
        results,
        effect_failures,
        {**reports, "interval_validity_failures": validity_width_failures},
    )


def _abstain_decision(
    config: Mapping[str, Any],
    failure: Exception | str,
    *,
    partial: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    detail = str(failure)
    decision: dict[str, Any] = {
        "schema": "tunnelgeopt.multifidelity.formal_decision.v1",
        "run_id": config.get("run_id"),
        "config_sha256": None,
        "classification": "ABSTAIN",
        "effect_claim_allowed": False,
        "validity_failures": [{"code": "FORMAL_EVIDENCE_INVALID", "detail": detail}],
        "effect_failures": [],
        "gate_summary": {"validity_pass": False, "effect_pass": False},
        "results": dict(partial or {}),
        "mandatory_reports": {"complete": False, "missing_or_invalid": detail},
        "claim_scope": None,
    }
    try:
        decision["config_sha256"] = canonical_sha256(config)
    except FormalAnalysisError:
        pass
    decision["decision_payload_sha256"] = canonical_sha256(decision)
    return decision


def evaluate_formal_decision(
    config: Mapping[str, Any],
    sealed_metrics: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    access_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every frozen v0.3 validity and effect gate.

    ``ABSTAIN`` has precedence over ``NO_GO``.  It is returned for missing or
    unauthenticated evidence, invalid solver/mesh/convergence data, non-finite
    predictions, fewer than five seeds, an interval wider than 0.10, missing
    mandatory reports, or a violated sealed-evaluation contract.  ``NO_GO`` is
    used only when the experiment is valid but an effect/robustness/physics
    threshold fails.  ``GO`` is possible only when every required gate passes.
    Joint OOD is mandatory to report but never changes an otherwise valid
    effect decision.
    """

    try:
        config = _require_mapping(config, "config")
        sealed_metrics = _require_mapping(sealed_metrics, "sealed_metrics")
        dataset_manifest = _require_mapping(dataset_manifest, "dataset_manifest")
        access_state = _require_mapping(access_state, "access_state")
        metadata = _validate_config(config)
        access = _validate_hashes_and_access(
            config, sealed_metrics, dataset_manifest, access_state, metadata
        )
        qc = _validate_dataset_qc(config, dataset_manifest, sealed_metrics, metadata)
        valid_case_ids = qc.pop("_valid_case_ids_by_partition")
        raw_partitions = _require_mapping(
            sealed_metrics.get("partitions"), "sealed_metrics.partitions"
        )
        if set(raw_partitions) != set(LOCKED_PARTITIONS):
            raise FormalAnalysisError("sealed metrics must report all four locked partitions")
        partitions = {
            name: _parse_partition(
                name,
                _require_mapping(raw_partitions[name], f"partition {name}"),
                config,
                metadata,
                access["registry_records"],
                valid_case_ids[name],
            )
            for name in LOCKED_PARTITIONS
        }
        resources = _validate_resources(sealed_metrics, metadata)
        results, effect_failures, reports = _analyze_effects(config, partitions, metadata)
        validity_failures = list(reports.pop("interval_validity_failures"))
        if access_state.get("abstain_reasons") not in (None, []):
            validity_failures.append(
                {
                    "code": "RUN_STATE_ABSTAIN_REASON",
                    "detail": "; ".join(str(value) for value in access_state["abstain_reasons"]),
                }
            )
        classification = "ABSTAIN" if validity_failures else ("NO_GO" if effect_failures else "GO")
        reports["resource_usage"] = resources
        reports["dataset_qc"] = qc
        reports["implementation_manifest"] = access["implementation_manifest"]
        reports["complete"] = True
        decision: dict[str, Any] = {
            "schema": "tunnelgeopt.multifidelity.formal_decision.v1",
            "run_id": access["run_id"],
            "config_sha256": metadata["config_sha256"],
            "classification": classification,
            "effect_claim_allowed": classification == "GO",
            "validity_failures": validity_failures,
            "effect_failures": effect_failures,
            "gate_summary": {
                "validity_pass": not validity_failures,
                "effect_pass": not effect_failures,
                "joint_ood_report_only": True,
            },
            "results": results,
            "mandatory_reports": reports,
            "input_hashes": dict(access_state["hashes"]),
            "claim_scope": (
                config["scientific_decision"]["passing_claim"] if classification == "GO" else None
            ),
        }
        decision["decision_payload_sha256"] = canonical_sha256(decision)
        return decision
    except (FormalAnalysisError, KeyError, TypeError, ValueError) as exc:
        return _abstain_decision(config if isinstance(config, Mapping) else {}, exc)


__all__ = [
    "LOCKED_PARTITIONS",
    "PRIMARY_PARTITIONS",
    "FormalAnalysisError",
    "canonical_sha256",
    "evaluate_formal_decision",
]
