#!/usr/bin/env python3
"""Run the v0.4 development-only cross-fit architecture gate.

This runner intentionally has no dataset generator and no sealed-evaluation
phase.  Every v0.3 label consumed here is permanently *seen* development data.
A passing result only authorizes drafting a future preregistration with new
identities; it is never a confirmatory model-effect result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tunnelgeopt.multifidelity_learning import (
    LearningBatch,
    aggregate_case_errors_by_parent,
    case_weighted_stress_error,
    hierarchical_paired_bootstrap,
    load_model_from_checkpoint,
    make_model,
    method_arrays,
    reconstruct_fine_prediction,
    train_with_dev_selection,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "multifidelity_v04_development.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "analysis" / "mf-structured-dev-v0.4.0"
SECTIONS = ("circle", "horseshoe", "straight_wall_arch")
SOURCE_PARTITIONS = (
    "train_id",
    "dev_id",
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
    "locked_joint_ood",
)
DEVELOPMENT_PARTITIONS = (
    "train_id",
    "dev_id",
    "seen_iid",
    "seen_geometry_ood",
    "seen_load_ood",
    "seen_joint_ood",
)
PRIMARY_PARTITIONS = ("seen_iid", "seen_geometry_ood", "seen_load_ood")
PARTITION_RENAME = {
    "train_id": "train_id",
    "dev_id": "dev_id",
    "locked_iid": "seen_iid",
    "locked_geometry_ood": "seen_geometry_ood",
    "locked_load_ood": "seen_load_ood",
    "locked_joint_ood": "seen_joint_ood",
}
METHODS = (
    "structured_linear_residual",
    "ablate_strict_load_linearity",
    "ablate_local_tensor_frame",
    "ablate_zero_init_coarse_gate",
    "generic_residual50_v03",
    "scratch_full_available",
    "direct_full_available",
)
ABLATIONS = (
    "ablate_strict_load_linearity",
    "ablate_local_tensor_frame",
    "ablate_zero_init_coarse_gate",
)
SAME_FOLD_LEARNED_COMPARATORS = ("scratch_full_available", "direct_full_available")


class DevelopmentProtocolError(RuntimeError):
    """Raised when a development-only integrity contract is violated."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise DevelopmentProtocolError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DevelopmentProtocolError(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DevelopmentProtocolError(f"invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise DevelopmentProtocolError(f"{label} must be a JSON object")
    return payload


def _load_npz(path: Path, label: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as exc:
        raise DevelopmentProtocolError(f"invalid {label}: {path}") from exc


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_bytes(dict(payload)) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return hashlib.sha256(encoded).hexdigest()


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(dict(payload)) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _resolve_inside(base: Path, relative: str, label: str) -> Path:
    value = Path(str(relative))
    if value.is_absolute():
        raise DevelopmentProtocolError(f"{label} must be relative to its declared root")
    path = (base / value).resolve()
    try:
        path.relative_to(base.resolve())
    except ValueError as exc:
        raise DevelopmentProtocolError(f"{label} escapes its declared root") from exc
    return path


def _require_exact_float(value: Any, expected: float, label: str) -> None:
    if not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12):
        raise DevelopmentProtocolError(f"{label} changed from the frozen value {expected}")


def _validate_architecture(config: Mapping[str, Any]) -> None:
    architecture = config["architecture"]
    champion = architecture["champion"]
    expected = {
        "strict_load_linearity": True,
        "local_tensor_frame": True,
        "exact_zero_init_coarse_gate": True,
    }
    if champion.get("name") != "structured_linear_residual" or any(
        champion.get(key) is not value for key, value in expected.items()
    ):
        raise DevelopmentProtocolError("structured champion contract changed")
    ablations = architecture.get("ablations")
    if not isinstance(ablations, list) or len(ablations) != 3:
        raise DevelopmentProtocolError("exactly three one-switch diagnostics are required")
    expected_names = tuple(ABLATIONS)
    if tuple(str(value.get("name")) for value in ablations) != expected_names:
        raise DevelopmentProtocolError("ablation names or order changed")
    removed = (
        "strict_load_linearity",
        "local_tensor_frame",
        "exact_zero_init_coarse_gate",
    )
    for candidate, removed_key in zip(ablations, removed, strict=True):
        for key, value in expected.items():
            required = False if key == removed_key else value
            if candidate.get(key) is not required:
                raise DevelopmentProtocolError(
                    f"{candidate.get('name')} is not a one-switch diagnostic"
                )
    reference = architecture.get("same_fold_reference", {})
    if reference.get("name") != "generic_residual50_v03":
        raise DevelopmentProtocolError("same-fold generic Residual50 reference is required")
    same_fold = tuple(value.get("name") for value in architecture.get("same_fold_comparators", ()))
    if same_fold != SAME_FOLD_LEARNED_COMPARATORS:
        raise DevelopmentProtocolError("same-fold available-budget comparators are required")
    baselines = tuple(
        value.get("name") for value in architecture.get("former_locked_stress_baselines", ())
    )
    if baselines != (
        "frozen_v03_scratch100",
        "frozen_v03_direct100",
        "frozen_v03_generic_residual50",
    ):
        raise DevelopmentProtocolError("authenticated v0.3 seen-stress comparators are required")


def load_development_config(path: Path) -> tuple[dict[str, Any], str]:
    """Load and fail-closed validate the frozen development protocol."""

    config = _read_json(Path(path), "v0.4 development config")
    if config.get("schema_version") != "tunnelgeopt.multifidelity_v04_development.v1":
        raise DevelopmentProtocolError("unknown v0.4 development schema")
    if config.get("status") != "implementation_stop_pending_pivot":
        raise DevelopmentProtocolError("development status must remain stopped pending a pivot")
    authorization = config.get("execution_authorization", {})
    if (
        authorization.get("validate_only_authorized") is not True
        or authorization.get("tiny_mock_authorized") is not True
        or authorization.get("real_cross_fit_authorized") is not False
    ):
        raise DevelopmentProtocolError("real development execution must remain unauthorized")
    if config.get("effect_claim_allowed") is not False:
        raise DevelopmentProtocolError("development config may not authorize an effect claim")
    if config.get("independent_validation_claim_allowed") is not False:
        raise DevelopmentProtocolError("seen-label development is not independent validation")

    seen = config.get("seen_data_contract", {})
    if (
        tuple(seen.get("development_partitions", ())) != DEVELOPMENT_PARTITIONS
        or seen.get("former_locked_partitions_are_seen") is not True
        or seen.get("new_locked_partition_count") != 0
        or seen.get("generator_invocation_allowed") is not False
        or seen.get("split_unit") != "geometry_group_id"
        or seen.get("all_loads_and_points_follow_parent") is not True
        or seen.get("cross_fit_source_partition") != "train_id"
        or seen.get("fixed_early_stopping_partition") != "dev_id"
        or seen.get("former_locked_training_access_allowed") is not False
    ):
        raise DevelopmentProtocolError("seen-data or no-new-locked contract changed")

    cross_fit = config.get("cross_fit", {})
    expected_same_fold_models = (
        "structured_linear_residual",
        "three_one_switch_diagnostics",
        "generic_residual50_v03",
        "scratch_full_available",
        "direct_full_available",
    )
    if (
        int(cross_fit.get("fold_count", 0)) != 5
        or not str(cross_fit.get("fold_salt", ""))
        or cross_fit.get("require_each_parent_exactly_one_oof_fold") is not True
        or cross_fit.get("oof_rows_never_used_for_optimizer_normalization_or_early_stopping")
        is not True
        or cross_fit.get("outer_oof_parent_count") != 72
        or cross_fit.get("optimizer_parent_count_per_fold") != 36
        or cross_fit.get("optimizer_parents_per_section_per_fold") != 12
        or tuple(cross_fit.get("same_fold_models", ())) != expected_same_fold_models
    ):
        raise DevelopmentProtocolError("five-fold parent-level cross-fit contract changed")
    _require_exact_float(
        cross_fit.get("fine_label_fraction_for_structured_and_generic_residual"),
        0.5,
        "cross-fit fine-label fraction",
    )
    seeds = tuple(int(value) for value in config.get("training_seeds", ()))
    if seeds != (103, 211, 307, 401, 509):
        raise DevelopmentProtocolError("five frozen training seeds changed")
    _validate_architecture(config)
    model = config.get("model", {})
    if (
        model.get("point_input_width") != 14
        or model.get("base_feature_width") != 11
        or model.get("coarse_feature_width") != 3
        or model.get("hidden_width") != 64
        or model.get("global_context_blocks") != 3
        or model.get("output_width") != 3
    ):
        raise DevelopmentProtocolError("frozen 14-to-3, 64x3 model shape contract changed")

    gates = config.get("launch_gates", {})
    if tuple(gates.get("primary_partitions", ())) != PRIMARY_PARTITIONS:
        raise DevelopmentProtocolError("primary development partition set changed")
    if gates.get("all_three_one_switch_diagnostics_must_be_reported") is not True:
        raise DevelopmentProtocolError("all three diagnostics must remain mandatory reports")
    architecture_gate = gates.get("cross_fit_architecture_gate", {})
    if (
        architecture_gate.get("partition") != "train_id_oof"
        or architecture_gate.get("candidate_and_generic_optimizer_parent_count") != 36
        or architecture_gate.get("candidate_and_generic_optimizer_parents_per_section") != 12
        or architecture_gate.get("fixed_early_stopping_parent_count") != 18
        or architecture_gate.get("fixed_early_stopping_parents_per_section") != 6
    ):
        raise DevelopmentProtocolError("cross-fit label-budget contract changed")
    _require_exact_float(
        architecture_gate.get("champion_to_same_fold_generic_residual50_max_point_ratio"),
        0.98,
        "generic-reference margin",
    )
    _require_exact_float(
        architecture_gate.get("champion_to_each_ablation_max_point_ratio"),
        0.99,
        "ablation margin",
    )
    _require_exact_float(
        architecture_gate.get("champion_to_each_same_fold_learned_comparator_max_point_ratio"),
        0.95,
        "same-fold learned-comparator margin",
    )
    _require_exact_float(
        architecture_gate.get("paired_parent_group_one_sided_upper_ratio_max"),
        1.0,
        "paired-parent development upper margin",
    )
    if (
        architecture_gate.get("paired_parent_group_interval_role")
        != "development_heuristic_only_fold_dependence_precludes_formal_ci_claim"
    ):
        raise DevelopmentProtocolError("fold-dependent interval may not be called a formal CI")
    stress_gate = gates.get("post_selection_seen_stress_gate", {})
    if (
        stress_gate.get("old_locked_labels_are_seen") is not True
        or stress_gate.get("champion_frozen_before_stress_test") is not True
        or stress_gate.get("absolute_seed_stability_minimum") != 4
        or stress_gate.get("absolute_seed_stability_total") != 5
        or stress_gate.get("section_gate_uses_partition_absolute_margins") is not True
    ):
        raise DevelopmentProtocolError("post-selection seen-stress boundary changed")
    _require_exact_float(
        stress_gate.get("champion_to_frozen_v03_generic_residual50_max_point_ratio"),
        0.98,
        "seen-stress generic-reference margin",
    )
    disclosures = config["architecture"].get("comparison_disclosures", {})
    if (
        disclosures.get("parameter_counts")
        != {
            "structured_linear_residual": 40685,
            "generic_residual50_v03": 38787,
            "ablate_strict_load_linearity": 38598,
            "ablate_local_tensor_frame": 40813,
            "ablate_zero_init_coarse_gate": 40685,
        }
        or disclosures.get("equal_parameter_count") is not False
        or disclosures.get("generic_models_receive_original_common_14_channels") is not True
        or disclosures.get("structured_model_receives_17_packed_channels") is not True
        or disclosures.get("structured_model_has_additional_derived_information") is not True
        or disclosures.get("structured_training_loss") != "residual_relative_per_case"
        or disclosures.get("generic_training_loss") != "legacy_weighted_mse"
        or disclosures.get("loss_functions_matched") is not False
        or disclosures.get("whole_candidate_comparison_allowed") is not True
        or disclosures.get("causal_component_attribution_allowed") is not False
        or disclosures.get("ablation_role") != "diagnostic_only_not_component_necessity_evidence"
    ):
        raise DevelopmentProtocolError("candidate-comparison confound disclosures changed")
    expected_margins = {
        "seen_iid": (0.95, 0.68),
        "seen_geometry_ood": (0.98, 0.78),
        "seen_load_ood": (0.98, 0.78),
    }
    margins = gates.get("absolute_safety_margins", {})
    for partition, (learned, coarse) in expected_margins.items():
        record = margins.get(partition, {})
        _require_exact_float(
            record.get("max_point_ratio_to_each_learned_baseline"),
            learned,
            f"{partition} learned-baseline margin",
        )
        _require_exact_float(
            record.get("max_one_sided_upper_ratio_to_coarse"),
            coarse,
            f"{partition} coarse margin",
        )
    stability = gates.get("seed_stability", {})
    if stability.get("minimum_passing_seeds") != 4 or stability.get("total_seeds") != 5:
        raise DevelopmentProtocolError("4-of-5 seed gate changed")
    sections = gates.get("cross_fit_section_robustness", {})
    if tuple(sections.get("required_sections", ())) != SECTIONS:
        raise DevelopmentProtocolError("three-section robustness gate changed")
    _require_exact_float(
        sections.get("max_ratio_to_any_comparator"), 1.02, "section robustness margin"
    )
    if (
        gates.get("post_selection_wall_all_primary_partitions_must_pass_original_v03_gates")
        is not True
    ):
        raise DevelopmentProtocolError("original wall-offset gate must remain mandatory")

    final_fit = config.get("final_development_fit", {})
    if (
        final_fit.get("run_only_after_all_launch_gates_pass") is not True
        or final_fit.get("expected_optimizer_parent_count") != 36
        or final_fit.get("expected_optimizer_parents_per_section") != 12
        or final_fit.get("expected_early_stopping_parent_count") != 18
        or final_fit.get("expected_early_stopping_parents_per_section") != 6
        or final_fit.get("required_v03_selection_sha256")
        != "0f62e6e197f96f582115d1c63888fb9852112debf38662b40f3ff33f07dd2ab9"
        or tuple(final_fit.get("forbidden_final_fit_partitions", ()))
        != PRIMARY_PARTITIONS + ("seen_joint_ood",)
        or final_fit.get(
            "forbid_former_locked_labels_in_optimizer_normalization_and_early_stopping"
        )
        is not True
    ):
        raise DevelopmentProtocolError("final 36-train/18-dev leakage boundary changed")
    output = config.get("output_contract", {})
    if output.get("create_new_locked_data") is not False:
        raise DevelopmentProtocolError("development runner may not create new locked data")
    if any("new_locked" in str(name).lower() for name in output.get("allowed_artifacts", ())):
        raise DevelopmentProtocolError("allowed output names disclose a new locked artifact")
    return config, _sha256_value(config)


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise DevelopmentProtocolError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def production_preflight(
    config: Mapping[str, Any], config_sha256: str, *, device: str
) -> dict[str, Any]:
    """Require a clean, pushed, tracked implementation before real GPU compute."""

    preflight = config["output_contract"]["real_execution_preflight"]
    critical = tuple(str(value) for value in preflight["critical_sources"])
    paths = {relative: _resolve_inside(ROOT, relative, relative) for relative in critical}
    if any(not path.is_file() for path in paths.values()):
        raise DevelopmentProtocolError("a critical development source is missing")
    tracked = set(_git_output("ls-files", "--", *critical).splitlines())
    if tracked != set(critical):
        raise DevelopmentProtocolError("real development run requires all critical sources tracked")
    status = _git_output("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise DevelopmentProtocolError("real development run requires a clean git worktree")
    head = _git_output("rev-parse", "HEAD")
    upstream_ref = _git_output("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    upstream = _git_output("rev-parse", "@{upstream}")
    if head != upstream:
        raise DevelopmentProtocolError("real development run requires HEAD equal to upstream")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise DevelopmentProtocolError("real development run requires PyTorch") from exc
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise DevelopmentProtocolError("real development run requires an available CUDA device")
    try:
        device_name = str(torch.cuda.get_device_name(device))
        device_memory = int(torch.cuda.get_device_properties(device).total_memory)
    except (AssertionError, RuntimeError, ValueError) as exc:
        raise DevelopmentProtocolError("could not authenticate the requested CUDA device") from exc
    return {
        "schema": "tunnelgeopt.multifidelity_v04_production_preflight.v1",
        "config_sha256": config_sha256,
        "git_head": head,
        "upstream_ref": upstream_ref,
        "upstream_head": upstream,
        "head_matches_upstream": True,
        "worktree_clean": True,
        "all_critical_sources_tracked": True,
        "source_sha256": {relative: _file_sha256(path) for relative, path in sorted(paths.items())},
        "device_requested": str(device),
        "cuda_available": True,
        "device_name": device_name,
        "device_total_memory_bytes": device_memory,
        "torch_version": str(torch.__version__),
        "passed": True,
    }


@dataclass(frozen=True)
class DevelopmentData:
    """All 705 v0.3 rows after formerly locked labels are marked seen."""

    base_features: np.ndarray
    coarse_stress: np.ndarray
    fine_stress: np.ndarray
    training_weights: np.ndarray
    metric_weights: np.ndarray
    arc_weights: np.ndarray
    wall_normals: np.ndarray
    wall_offset_mask: np.ndarray
    stress_scales: np.ndarray
    case_group_ids: tuple[str, ...]
    geometry_group_ids: tuple[str, ...]
    section_families: tuple[str, ...]
    source_partitions: tuple[str, ...]
    development_partitions: tuple[str, ...]
    label_roles: tuple[str, ...]
    fine_available: np.ndarray

    @property
    def case_count(self) -> int:
        return len(self.case_group_ids)

    @property
    def parent_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.geometry_group_ids)))


def _validate_source_file_records(config: Mapping[str, Any]) -> dict[str, Path]:
    source = config["source_experiment"]
    source_root = _resolve_inside(ROOT, str(source["root"]), "source experiment root")
    records = source.get("files")
    expected_roles = {
        "dataset_manifest",
        "public_inputs",
        "train_dev_labels",
        "seen_iid_labels",
        "seen_geometry_ood_labels",
        "seen_load_ood_labels",
        "seen_joint_ood_labels",
        "v03_decision",
        "v03_checkpoint_manifest",
    }
    if not isinstance(records, Mapping) or set(records) != expected_roles:
        raise DevelopmentProtocolError("source artifact role set changed")
    paths: dict[str, Path] = {}
    for role, record in records.items():
        if not isinstance(record, Mapping):
            raise DevelopmentProtocolError(f"source record {role} is invalid")
        path = _resolve_inside(source_root, str(record.get("path")), role)
        expected = _require_sha256(record.get("sha256"), f"{role} sha256")
        if not path.is_file() or _file_sha256(path) != expected:
            raise DevelopmentProtocolError(f"source artifact hash mismatch: {role}")
        paths[str(role)] = path
    return paths


def _aligned_label_rows(
    public: Mapping[str, np.ndarray],
    archive: Mapping[str, np.ndarray],
    *,
    expected_partition: str | None,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    required = {"indices", "fine_stress", "case_group_ids"}
    if not required.issubset(archive):
        raise DevelopmentProtocolError(f"{label} omits required arrays")
    indices = np.asarray(archive["indices"], dtype=np.int64)
    if indices.ndim != 1 or len(set(indices.tolist())) != indices.size:
        raise DevelopmentProtocolError(f"{label} indices must be unique one-dimensional rows")
    if np.any(indices < 0) or np.any(indices >= len(public["case_group_ids"])):
        raise DevelopmentProtocolError(f"{label} contains an out-of-range public row")
    case_ids = np.asarray(public["case_group_ids"])[indices]
    if not np.array_equal(case_ids, np.asarray(archive["case_group_ids"])):
        raise DevelopmentProtocolError(f"{label} fine labels are identity-misaligned")
    if expected_partition is not None:
        partitions = np.asarray(public["partitions"])[indices]
        if set(map(str, partitions.tolist())) != {expected_partition}:
            raise DevelopmentProtocolError(f"{label} contains another source partition")
    fine = np.asarray(archive["fine_stress"], dtype=np.float32)
    expected_shape = (indices.size, public["base_features"].shape[1], 3)
    if fine.shape != expected_shape or not np.isfinite(fine).all():
        raise DevelopmentProtocolError(f"{label} fine tensor is invalid")
    return indices, fine


def audit_and_load_inputs(
    config: Mapping[str, Any], config_sha256: str
) -> tuple[DevelopmentData, dict[str, Any], dict[str, Any]]:
    """Authenticate v0.3, then combine all labels under explicit seen roles."""

    paths = _validate_source_file_records(config)
    source = config["source_experiment"]
    decision = _read_json(paths["v03_decision"], "v0.3 scientific decision")
    if (
        decision.get("classification") != source["required_classification"]
        or decision.get("effect_claim_allowed") is not source["required_effect_claim_allowed"]
        or decision.get("config_sha256") != source["config_sha256"]
    ):
        raise DevelopmentProtocolError("v0.3 decision boundary differs from the frozen source")
    manifest = _read_json(paths["dataset_manifest"], "v0.3 dataset manifest")
    checkpoint_manifest = _read_json(paths["v03_checkpoint_manifest"], "v0.3 checkpoint manifest")
    if (
        manifest.get("config_sha256") != source["config_sha256"]
        or checkpoint_manifest.get("config_sha256") != source["config_sha256"]
        or checkpoint_manifest.get("completed_checkpoint_count") != 35
    ):
        raise DevelopmentProtocolError("v0.3 manifests do not share the frozen config identity")
    public = _load_npz(paths["public_inputs"], "v0.3 public input archive")
    required_public = {
        "base_features",
        "coarse_stress",
        "training_weights",
        "metric_weights",
        "arc_weights",
        "wall_rock_outward_normals_yz",
        "wall_offset_mask",
        "stress_scales",
        "case_group_ids",
        "geometry_group_ids",
        "section_families",
        "partitions",
    }
    if not required_public.issubset(public):
        raise DevelopmentProtocolError(
            f"public archive omits arrays: {sorted(required_public - set(public))}"
        )
    case_count = len(public["case_group_ids"])
    if case_count != 705 or len(set(map(str, public["case_group_ids"]))) != case_count:
        raise DevelopmentProtocolError("v0.3 public archive must contain 705 unique cases")
    if set(map(str, public["partitions"])) != set(SOURCE_PARTITIONS):
        raise DevelopmentProtocolError("v0.3 source partition set changed")

    fine = np.full_like(np.asarray(public["coarse_stress"], dtype=np.float32), np.nan)
    filled = np.zeros(case_count, dtype=bool)
    label_roles = np.full(case_count, "", dtype="U32")
    train_dev = _load_npz(paths["train_dev_labels"], "v0.3 train/dev labels")
    indices, values = _aligned_label_rows(
        public, train_dev, expected_partition=None, label="train/dev labels"
    )
    if set(map(str, np.asarray(public["partitions"])[indices])) != {"train_id", "dev_id"}:
        raise DevelopmentProtocolError("train/dev label store covers another partition")
    fine[indices] = values
    filled[indices] = True
    label_roles[indices] = "original_train_dev"

    seen_roles = (
        ("seen_iid_labels", "locked_iid", "seen_iid"),
        ("seen_geometry_ood_labels", "locked_geometry_ood", "seen_geometry_ood"),
        ("seen_load_ood_labels", "locked_load_ood", "seen_load_ood"),
        ("seen_joint_ood_labels", "locked_joint_ood", "seen_joint_ood"),
    )
    for role, source_partition, development_partition in seen_roles:
        record = source["files"][role]
        if (
            record.get("original_partition") != source_partition
            or record.get("development_partition") != development_partition
        ):
            raise DevelopmentProtocolError(f"{role} is not explicitly relabeled as seen")
        indices = np.flatnonzero(np.asarray(public["partitions"]) == source_partition)
        if np.any(filled[indices]):
            raise DevelopmentProtocolError("a fine-label row appears in multiple source stores")
        # Hash authentication does not open former-locked fine values.  The
        # values are opened only after final checkpoints have been frozen.
        label_roles[indices] = development_partition
    if np.any(label_roles == ""):
        raise DevelopmentProtocolError("every case must have an explicit seen-label role")

    source_partitions = tuple(str(value) for value in public["partitions"])
    development_partitions = tuple(PARTITION_RENAME[value] for value in source_partitions)
    geometry_records: dict[str, tuple[str, str]] = {}
    for geometry_id, partition, section in zip(
        public["geometry_group_ids"],
        development_partitions,
        public["section_families"],
        strict=True,
    ):
        identity = str(geometry_id)
        record = (str(partition), str(section))
        if geometry_records.setdefault(identity, record) != record:
            raise DevelopmentProtocolError("one parent crosses partition or section strata")
    if len(geometry_records) != 195 or {value[1] for value in geometry_records.values()} != set(
        SECTIONS
    ):
        raise DevelopmentProtocolError("v0.3 parent or section count changed")

    data = DevelopmentData(
        base_features=np.asarray(public["base_features"], dtype=np.float32),
        coarse_stress=np.asarray(public["coarse_stress"], dtype=np.float32),
        fine_stress=fine,
        training_weights=np.asarray(public["training_weights"], dtype=np.float32),
        metric_weights=np.asarray(public["metric_weights"], dtype=np.float32),
        arc_weights=np.asarray(public["arc_weights"], dtype=np.float32),
        wall_normals=np.asarray(public["wall_rock_outward_normals_yz"], dtype=np.float32),
        wall_offset_mask=np.asarray(public["wall_offset_mask"], dtype=bool),
        stress_scales=np.asarray(public["stress_scales"], dtype=np.float64),
        case_group_ids=tuple(str(value) for value in public["case_group_ids"]),
        geometry_group_ids=tuple(str(value) for value in public["geometry_group_ids"]),
        section_families=tuple(str(value) for value in public["section_families"]),
        source_partitions=source_partitions,
        development_partitions=development_partitions,
        label_roles=tuple(str(value) for value in label_roles),
        fine_available=filled,
    )
    input_audit = {
        "schema": "tunnelgeopt.multifidelity_v04_input_audit.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "source_run_id": source["run_id"],
        "source_classification": decision["classification"],
        "source_effect_claim_allowed": decision["effect_claim_allowed"],
        "former_locked_partitions_are_seen": True,
        "independent_validation_claim_allowed": False,
        "new_locked_partition_count": 0,
        "generator_invocation_count": 0,
        "case_count": data.case_count,
        "parent_count": len(data.parent_ids),
        "source_file_sha256": {role: _file_sha256(path) for role, path in sorted(paths.items())},
        "label_role_counts": {
            role: int(np.sum(np.asarray(data.label_roles) == role))
            for role in sorted(set(data.label_roles))
        },
        "development_partition_case_counts": {
            partition: int(np.sum(np.asarray(data.development_partitions) == partition))
            for partition in DEVELOPMENT_PARTITIONS
        },
        "all_former_locked_labels_classified_as_seen": True,
        "train_dev_fine_label_case_reads": int(np.sum(filled)),
        "former_locked_seen_fine_label_case_reads": 0,
        "former_locked_seen_values_opened": False,
        "passed": True,
    }
    return data, checkpoint_manifest, input_audit


def open_seen_stress_labels(
    data: DevelopmentData, config: Mapping[str, Any]
) -> tuple[DevelopmentData, dict[str, Any]]:
    """Open former-locked values only after caller has frozen final checkpoints."""

    paths = _validate_source_file_records(config)
    public = _load_npz(paths["public_inputs"], "v0.3 public input archive")
    fine = np.array(data.fine_stress, copy=True)
    available = np.array(data.fine_available, copy=True)
    seen_roles = (
        ("seen_iid_labels", "locked_iid", "seen_iid"),
        ("seen_geometry_ood_labels", "locked_geometry_ood", "seen_geometry_ood"),
        ("seen_load_ood_labels", "locked_load_ood", "seen_load_ood"),
        ("seen_joint_ood_labels", "locked_joint_ood", "seen_joint_ood"),
    )
    opened: dict[str, Any] = {}
    for role, source_partition, development_partition in seen_roles:
        archive = _load_npz(paths[role], role)
        indices, values = _aligned_label_rows(
            public,
            archive,
            expected_partition=source_partition,
            label=role,
        )
        if np.any(available[indices]):
            raise DevelopmentProtocolError("a seen-stress row was already available before open")
        fine[indices] = values
        available[indices] = True
        opened[development_partition] = {
            "case_count": int(indices.size),
            "source_sha256": _file_sha256(paths[role]),
            "opened_at_utc": _now(),
            "role": "seen_post_selection_stress_only",
        }
    if not np.all(available) or not np.isfinite(fine).all():
        raise DevelopmentProtocolError("post-selection seen-label open is incomplete")
    return replace(data, fine_stress=fine, fine_available=available), opened


def _rank(salt: str, *values: str) -> str:
    return hashlib.sha256(":".join((salt, *values)).encode("utf-8")).hexdigest()


def _parent_records(data: DevelopmentData) -> dict[str, tuple[str, str]]:
    records: dict[str, tuple[str, str]] = {}
    for parent, partition, section in zip(
        data.geometry_group_ids,
        data.development_partitions,
        data.section_families,
        strict=True,
    ):
        record = (partition, section)
        if records.setdefault(parent, record) != record:
            raise DevelopmentProtocolError("one parent has inconsistent stratification metadata")
    return records


def _stratified_fold_assignment(
    data: DevelopmentData,
    *,
    fold_count: int,
    salt: str,
    eligible_parents: Sequence[str] | None = None,
) -> dict[str, int]:
    records = _parent_records(data)
    if eligible_parents is not None:
        eligible = set(map(str, eligible_parents))
        if not eligible or not eligible.issubset(records):
            raise DevelopmentProtocolError("fold eligibility contains an unknown parent")
        records = {parent: record for parent, record in records.items() if parent in eligible}
    strata: dict[tuple[str, str], list[str]] = {}
    for parent, stratum in records.items():
        strata.setdefault(stratum, []).append(parent)
    assignment: dict[str, int] = {}
    for (partition, section), parents in sorted(strata.items()):
        ordered = sorted(
            parents,
            key=lambda parent: _rank(salt, partition, section, parent),
        )
        if len(ordered) < fold_count:
            raise DevelopmentProtocolError(
                f"stratum {partition}/{section} cannot support {fold_count} folds"
            )
        for index, parent in enumerate(ordered):
            assignment[parent] = index % fold_count
    if set(assignment) != set(records):
        raise DevelopmentProtocolError("fold assignment omitted a parent")
    return assignment


def _stratified_holdout(
    parents: Sequence[str],
    records: Mapping[str, tuple[str, str]],
    *,
    fraction: float,
    salt: str,
) -> tuple[str, ...]:
    strata: dict[tuple[str, str], list[str]] = {}
    for parent in parents:
        strata.setdefault(records[parent], []).append(parent)
    selected: set[str] = set()
    for (partition, section), values in sorted(strata.items()):
        ordered = sorted(values, key=lambda parent: _rank(salt, partition, section, parent))
        count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * fraction)))
        if count <= 0:
            raise DevelopmentProtocolError("inner holdout emptied a development stratum")
        selected.update(ordered[:count])
    return tuple(sorted(selected))


def _stratified_fraction(
    parents: Sequence[str],
    records: Mapping[str, tuple[str, str]],
    *,
    fraction: float,
    salt: str,
) -> tuple[str, ...]:
    strata: dict[tuple[str, str], list[str]] = {}
    for parent in parents:
        strata.setdefault(records[parent], []).append(parent)
    selected: set[str] = set()
    for (partition, section), values in sorted(strata.items()):
        ordered = sorted(values, key=lambda parent: _rank(salt, partition, section, parent))
        count = max(1, math.floor(len(ordered) * fraction))
        selected.update(ordered[:count])
    return tuple(sorted(selected))


def build_fold_manifest(
    data: DevelopmentData, config: Mapping[str, Any], config_sha256: str
) -> dict[str, Any]:
    """Build leakage-safe outer folds, inner dev sets and fine50 optimizer sets."""

    cross_fit = config["cross_fit"]
    count = int(cross_fit["fold_count"])
    records = _parent_records(data)
    train_parents = tuple(
        sorted(parent for parent, (partition, _) in records.items() if partition == "train_id")
    )
    dev_parents = tuple(
        sorted(parent for parent, (partition, _) in records.items() if partition == "dev_id")
    )
    if len(train_parents) != 72 or len(dev_parents) != 18:
        raise DevelopmentProtocolError("cross-fit requires original 72 train and 18 dev parents")
    assignment = _stratified_fold_assignment(
        data,
        fold_count=count,
        salt=str(cross_fit["fold_salt"]),
        eligible_parents=train_parents,
    )
    all_train_parents = set(train_parents)
    folds: list[dict[str, Any]] = []
    oof_occurrences = {parent: 0 for parent in all_train_parents}
    for fold in range(count):
        oof = tuple(sorted(parent for parent, value in assignment.items() if value == fold))
        training_pool = tuple(sorted(all_train_parents - set(oof)))
        # The helper's generic fraction rule is deliberately replaced by the
        # frozen exact label budget: twelve parents in each section.
        optimizer_fine50_set: set[str] = set()
        for section in SECTIONS:
            candidates = [parent for parent in training_pool if records[parent][1] == section]
            ordered = sorted(
                candidates,
                key=lambda parent: _rank(
                    str(cross_fit["fine_parent_salt"]), str(fold), section, parent
                ),
            )
            optimizer_fine50_set.update(ordered[:12])
        optimizer_fine50 = tuple(sorted(optimizer_fine50_set))
        if (
            set(oof) & set(training_pool)
            or set(oof) & set(dev_parents)
            or set(oof) & set(optimizer_fine50)
            or set(dev_parents) & set(optimizer_fine50)
            or not set(optimizer_fine50).issubset(training_pool)
        ):
            raise DevelopmentProtocolError("outer/inner/optimizer parent sets overlap")
        section_counts = {
            section: sum(records[parent][1] == section for parent in optimizer_fine50)
            for section in SECTIONS
        }
        if len(optimizer_fine50) != 36 or set(section_counts.values()) != {12}:
            raise DevelopmentProtocolError("each fold must fit exactly 36 parents, 12 per section")
        for parent in oof:
            oof_occurrences[parent] += 1

        def counts(values: Sequence[str]) -> dict[str, int]:
            output: dict[str, int] = {}
            for parent in values:
                partition, section = records[parent]
                key = f"{partition}:{section}"
                output[key] = output.get(key, 0) + 1
            return dict(sorted(output.items()))

        folds.append(
            {
                "fold": fold,
                "oof_parent_ids": list(oof),
                "fixed_dev_parent_ids": list(dev_parents),
                "available_non_oof_train_parent_ids": list(training_pool),
                "optimizer_fine50_parent_ids": list(optimizer_fine50),
                "normalization_fit_parent_ids": list(optimizer_fine50),
                "role_counts": {
                    "oof": counts(oof),
                    "fixed_dev": counts(dev_parents),
                    "available_non_oof_train": counts(training_pool),
                    "optimizer_fine50": counts(optimizer_fine50),
                },
                "role_identity_sha256": {
                    "oof": _sha256_value(list(oof)),
                    "fixed_dev": _sha256_value(list(dev_parents)),
                    "available_non_oof_train": _sha256_value(list(training_pool)),
                    "optimizer_fine50": _sha256_value(list(optimizer_fine50)),
                },
                "leakage_checks": {
                    "oof_optimizer_zero_intersection": True,
                    "oof_normalization_zero_intersection": True,
                    "oof_fixed_dev_zero_intersection": True,
                    "fixed_dev_optimizer_zero_intersection": True,
                    "former_locked_optimizer_zero_intersection": True,
                    "former_locked_normalization_zero_intersection": True,
                    "former_locked_early_stopping_zero_intersection": True,
                    "all_loads_follow_parent": True,
                },
            }
        )
    if set(oof_occurrences.values()) != {1}:
        raise DevelopmentProtocolError("each parent must occur in exactly one OOF fold")
    return {
        "schema": "tunnelgeopt.multifidelity_v04_fold_manifest.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "split_unit": "geometry_group_id",
        "fold_count": count,
        "fold_salt": cross_fit["fold_salt"],
        "outer_oof_parent_count": len(all_train_parents),
        "fixed_early_stopping_parent_count": len(dev_parents),
        "each_parent_exactly_one_oof_fold": True,
        "all_labels_are_seen_development_data": True,
        "independent_validation_claim_allowed": False,
        "assignment_sha256": _sha256_value(dict(sorted(assignment.items()))),
        "folds": folds,
    }


def _row_indices(data: DevelopmentData, parents: Sequence[str]) -> np.ndarray:
    selected = set(map(str, parents))
    rows = np.asarray(
        [index for index, parent in enumerate(data.geometry_group_ids) if parent in selected],
        dtype=np.int64,
    )
    if not selected or set(np.asarray(data.geometry_group_ids)[rows].tolist()) != selected:
        raise DevelopmentProtocolError("parent selection did not resolve to exact case rows")
    return rows


def _batch_for_rows(
    data: DevelopmentData,
    rows: Sequence[int],
    *,
    split: str,
) -> LearningBatch:
    indices = np.asarray(rows, dtype=np.int64)
    if not np.all(data.fine_available[indices]):
        raise DevelopmentProtocolError("attempted to build a batch from unopened fine labels")
    return LearningBatch(
        base_features=data.base_features[indices],
        coarse_stress=data.coarse_stress[indices],
        fine_stress=data.fine_stress[indices],
        weights=data.training_weights[indices],
        geometry_group_ids=tuple(data.geometry_group_ids[index] for index in indices),
        section_families=tuple(data.section_families[index] for index in indices),
        case_group_ids=tuple(data.case_group_ids[index] for index in indices),
        splits=tuple(split for _ in indices),
    )


def _combined_train_dev_batch(
    data: DevelopmentData,
    train_parents: Sequence[str],
    dev_parents: Sequence[str],
) -> LearningBatch:
    train_rows = _row_indices(data, train_parents)
    dev_rows = _row_indices(data, dev_parents)
    if set(train_rows.tolist()) & set(dev_rows.tolist()):
        raise DevelopmentProtocolError("optimizer and early-stopping case rows overlap")
    rows = np.concatenate((train_rows, dev_rows))
    if not np.all(data.fine_available[rows]):
        raise DevelopmentProtocolError("optimizer/dev batch requested an unopened seen label")
    split = tuple("train" for _ in train_rows) + tuple("dev" for _ in dev_rows)
    return LearningBatch(
        base_features=data.base_features[rows],
        coarse_stress=data.coarse_stress[rows],
        fine_stress=data.fine_stress[rows],
        weights=data.training_weights[rows],
        geometry_group_ids=tuple(data.geometry_group_ids[index] for index in rows),
        section_families=tuple(data.section_families[index] for index in rows),
        case_group_ids=tuple(data.case_group_ids[index] for index in rows),
        splits=split,
    )


def _architecture_by_name(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    architecture = config["architecture"]
    values = [
        architecture["champion"],
        *architecture["ablations"],
        architecture["same_fold_reference"],
        *architecture["same_fold_comparators"],
        *architecture["former_locked_stress_baselines"],
    ]
    matches = [dict(value) for value in values if value.get("name") == name]
    if len(matches) != 1:
        raise DevelopmentProtocolError(f"architecture record is missing or duplicated: {name}")
    return matches[0]


def _generic_method(name: str) -> str:
    if name == "generic_residual50_v03":
        return "residual_coarse"
    if name in {"scratch_full_available", "frozen_v03_scratch100"}:
        return "scratch"
    if name in {"direct_full_available", "frozen_v03_direct100"}:
        return "direct_coarse"
    if name == "frozen_v03_generic_residual50":
        return "residual_coarse"
    raise DevelopmentProtocolError(f"{name} is not a generic v0.3 model")


def _predict_torch_model(
    model: Any, features: np.ndarray, *, batch_size: int, device: str
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - learn extra is required for real runs
        raise DevelopmentProtocolError("PyTorch is required for a production run") from exc
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            values = torch.as_tensor(
                features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            outputs.append(model(values).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _structured_module() -> Any:
    try:
        from tunnelgeopt import structured_residual
    except ImportError as exc:
        raise DevelopmentProtocolError("structured residual implementation is unavailable") from exc
    required = {
        "make_structured_residual_model",
        "pack_structured_features",
    }
    if not required.issubset(vars(structured_residual)):
        raise DevelopmentProtocolError("structured residual API is incomplete")
    return structured_residual


def _model_mapping(config: Mapping[str, Any], architecture: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    if int(model["hidden_width"]) != 64 or int(model["global_context_blocks"]) != 3:
        raise DevelopmentProtocolError("structured candidate must retain the frozen 64x3 shape")
    return {
        "hidden_width": 64,
        "global_context_blocks": 3,
        "strict_load_linearity": bool(architecture["strict_load_linearity"]),
        "local_tensor_frame": bool(architecture["local_tensor_frame"]),
        "exact_zero_init_coarse_gate": bool(architecture["exact_zero_init_coarse_gate"]),
    }


def _structured_features(data: DevelopmentData, rows: np.ndarray) -> np.ndarray:
    module = _structured_module()
    features14 = np.concatenate(
        (data.base_features[rows], data.coarse_stress[rows]), axis=-1
    ).astype(np.float32)
    packed = module.pack_structured_features(
        features14,
        data.wall_normals[rows],
        data.wall_offset_mask[rows],
    )
    values = np.asarray(packed, dtype=np.float32)
    if values.shape != (*features14.shape[:-1], 17) or not np.isfinite(values).all():
        raise DevelopmentProtocolError(
            "structured feature packing violated the 17-channel contract"
        )
    return values


def _fit_predict_one(
    data: DevelopmentData,
    config: Mapping[str, Any],
    *,
    method_name: str,
    seed: int,
    train_parents: Sequence[str],
    dev_parents: Sequence[str],
    predict_parents: Sequence[str],
    device: str,
    progress: Any = None,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    """Fit one development model and predict only its held-out parent set."""

    train_rows = _row_indices(data, train_parents)
    dev_rows = _row_indices(data, dev_parents)
    predict_rows = _row_indices(data, predict_parents)
    if (
        set(train_rows.tolist()) & set(dev_rows.tolist())
        or set(train_rows.tolist()) & set(predict_rows.tolist())
        or set(dev_rows.tolist()) & set(predict_rows.tolist())
    ):
        raise DevelopmentProtocolError("fit/predict roles overlap at case-row level")
    if not np.all(data.fine_available[np.concatenate((train_rows, dev_rows, predict_rows))]):
        raise DevelopmentProtocolError("fit/predict requested a label not yet available")

    optimization = config["optimization"]
    batch_size = int(optimization["case_batch_size"])
    if method_name.startswith(("structured_", "ablate_")):
        architecture = _architecture_by_name(config, method_name)
        module = _structured_module()
        mapping = _model_mapping(config, architecture)
        model = module.make_structured_residual_model(mapping, seed=int(seed), device=device)
        train_features = _structured_features(data, train_rows)
        dev_features = _structured_features(data, dev_rows)
        predict_features = _structured_features(data, predict_rows)
        train_targets = data.fine_stress[train_rows] - data.coarse_stress[train_rows]
        dev_base = data.coarse_stress[dev_rows]
        predict_base = data.coarse_stress[predict_rows]
        if mapping["exact_zero_init_coarse_gate"]:
            initial_raw = _predict_torch_model(
                model,
                train_features[: min(2, train_features.shape[0])],
                batch_size=batch_size,
                device=device,
            )
            if not np.array_equal(initial_raw, np.zeros_like(initial_raw)):
                raise DevelopmentProtocolError("zero-init coarse gate is not exactly identity")
    else:
        generic_method = _generic_method(method_name)
        train_batch = _batch_for_rows(data, train_rows, split="train")
        dev_batch = _batch_for_rows(data, dev_rows, split="dev")
        predict_batch = _batch_for_rows(data, predict_rows, split="oof")
        train_features, train_targets, _ = method_arrays(train_batch, generic_method)
        dev_features, _, dev_base = method_arrays(dev_batch, generic_method)
        predict_features, _, predict_base = method_arrays(predict_batch, generic_method)
        model_config = {
            "point_input_width": 14,
            "hidden_width": 64,
            "global_context_blocks": 3,
            "output_width": 3,
        }
        model = make_model(model_config, seed=int(seed), device=device)

    if method_name.startswith(("structured_", "ablate_")):
        outcome = module.train_structured_with_dev_selection(
            model,
            np.asarray(train_features, dtype=np.float32),
            np.asarray(train_targets, dtype=np.float32),
            data.training_weights[train_rows],
            np.asarray(dev_features, dtype=np.float32),
            data.fine_stress[dev_rows],
            np.asarray(dev_base, dtype=np.float32),
            data.metric_weights[dev_rows],
            seed=int(seed),
            device=device,
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            batch_size=batch_size,
            max_epochs=int(optimization["max_epochs"]),
            patience=int(optimization["early_stopping_patience"]),
            min_delta=float(optimization["early_stopping_min_delta"]),
            loss_mode="residual_relative_per_case",
        )
        if progress is not None:
            for record in outcome.history:
                progress(record)
    else:
        outcome = train_with_dev_selection(
            model,
            np.asarray(train_features, dtype=np.float32),
            np.asarray(train_targets, dtype=np.float32),
            data.training_weights[train_rows],
            np.asarray(dev_features, dtype=np.float32),
            data.fine_stress[dev_rows],
            np.asarray(dev_base, dtype=np.float32),
            data.metric_weights[dev_rows],
            seed=int(seed),
            device=device,
            learning_rate=float(optimization["learning_rate"]),
            weight_decay=float(optimization["weight_decay"]),
            batch_size=batch_size,
            max_epochs=int(optimization["max_epochs"]),
            patience=int(optimization["early_stopping_patience"]),
            min_delta=float(optimization["early_stopping_min_delta"]),
            progress=progress,
        )
    raw = _predict_torch_model(
        model,
        np.asarray(predict_features, dtype=np.float32),
        batch_size=batch_size,
        device=device,
    )
    prediction = reconstruct_fine_prediction(raw, predict_base)
    return (
        prediction,
        model,
        {
            "best_epoch": int(outcome.best_epoch),
            "epochs_run": int(outcome.epochs_run),
            "best_dev_error": float(outcome.best_dev_error),
            "train_parent_count": len(set(train_parents)),
            "dev_parent_count": len(set(dev_parents)),
            "prediction_parent_count": len(set(predict_parents)),
            "train_parent_ids_sha256": _sha256_value(sorted(set(train_parents))),
            "dev_parent_ids_sha256": _sha256_value(sorted(set(dev_parents))),
            "prediction_parent_ids_sha256": _sha256_value(sorted(set(predict_parents))),
        },
    )


def _case_errors(data: DevelopmentData, rows: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return case_weighted_stress_error(
        prediction,
        data.fine_stress[rows],
        data.metric_weights[rows],
    )


def run_cross_fit(
    data: DevelopmentData,
    config: Mapping[str, Any],
    fold_manifest: Mapping[str, Any],
    *,
    device: str,
    output_dir: Path,
    tiny_mock: bool,
) -> dict[str, Any]:
    """Create complete OOF errors without ever opening former-locked labels."""

    train_rows = np.flatnonzero(np.asarray(data.development_partitions) == "train_id")
    if train_rows.size != 288 or not np.all(data.fine_available[train_rows]):
        raise DevelopmentProtocolError("cross-fit must see exactly 288 original train cases")
    former_locked = np.isin(
        np.asarray(data.development_partitions),
        np.asarray(("seen_iid", "seen_geometry_ood", "seen_load_ood", "seen_joint_ood")),
    )
    if np.any(data.fine_available[former_locked]):
        raise DevelopmentProtocolError("former-locked labels opened before cross-fit")
    local_row = {int(row): index for index, row in enumerate(train_rows)}
    seeds = tuple(int(value) for value in config["training_seeds"])
    errors = {
        method: np.full((len(seeds), train_rows.size), np.nan, dtype=np.float64)
        for method in METHODS
    }
    fold_by_case = np.full(train_rows.size, -1, dtype=np.int64)
    training_records: list[dict[str, Any]] = []
    progress_path = output_dir / "training_progress.jsonl"
    for fold_record in fold_manifest["folds"]:
        fold = int(fold_record["fold"])
        oof_parents = tuple(map(str, fold_record["oof_parent_ids"]))
        fine50 = tuple(map(str, fold_record["optimizer_fine50_parent_ids"]))
        full_available = tuple(map(str, fold_record["available_non_oof_train_parent_ids"]))
        dev_parents = tuple(map(str, fold_record["fixed_dev_parent_ids"]))
        oof_rows = _row_indices(data, oof_parents)
        for row in oof_rows:
            fold_by_case[local_row[int(row)]] = fold
        for seed_index, seed in enumerate(seeds):
            coarse_error = _case_errors(data, oof_rows, data.coarse_stress[oof_rows])
            for method in METHODS:
                train_parents = (
                    full_available if method in SAME_FOLD_LEARNED_COMPARATORS else fine50
                )
                if tiny_mock:
                    factors = {
                        "structured_linear_residual": 0.50,
                        "ablate_strict_load_linearity": 0.58,
                        "ablate_local_tensor_frame": 0.59,
                        "ablate_zero_init_coarse_gate": 0.57,
                        "generic_residual50_v03": 0.61,
                        "scratch_full_available": 0.67,
                        "direct_full_available": 0.64,
                    }
                    jitter = 1.0 + 0.002 * seed_index
                    values = coarse_error * factors[method] * jitter
                    record = {
                        "best_epoch": 0,
                        "epochs_run": 1,
                        "best_dev_error": float(np.mean(values)),
                        "train_parent_count": len(set(train_parents)),
                        "dev_parent_count": len(set(dev_parents)),
                        "prediction_parent_count": len(set(oof_parents)),
                        "mock": True,
                    }
                else:

                    def progress(
                        values: Mapping[str, Any],
                        *,
                        fold_value: int = fold,
                        seed_value: int = seed,
                        method_value: str = method,
                    ) -> None:
                        _append_jsonl(
                            progress_path,
                            {
                                "at_utc": _now(),
                                "phase": "cross_fit",
                                "fold": fold_value,
                                "seed": seed_value,
                                "method": method_value,
                                **dict(values),
                            },
                        )

                    prediction, _, record = _fit_predict_one(
                        data,
                        config,
                        method_name=method,
                        seed=seed,
                        train_parents=train_parents,
                        dev_parents=dev_parents,
                        predict_parents=oof_parents,
                        device=device,
                        progress=progress,
                    )
                    values = _case_errors(data, oof_rows, prediction)
                positions = np.asarray([local_row[int(row)] for row in oof_rows], dtype=np.int64)
                errors[method][seed_index, positions] = values
                training_records.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "optimizer_role": (
                            "all_non_oof_train_id"
                            if method in SAME_FOLD_LEARNED_COMPARATORS
                            else "fixed_36_train_id"
                        ),
                        **record,
                    }
                )
    if np.any(fold_by_case < 0) or any(not np.isfinite(values).all() for values in errors.values()):
        raise DevelopmentProtocolError("OOF prediction coverage is incomplete")
    return {
        "schema": "tunnelgeopt.multifidelity_v04_cross_fit_metrics.v1",
        "scope": "development_only_original_train_id_oof",
        "effect_claim_allowed": False,
        "formal_confidence_interval_claim_allowed": False,
        "fixed_dev_reused_across_all_folds": True,
        "fixed_dev_reuse_disclosure": (
            "the same 18 original dev parents were repeatedly used for early stopping; "
            "all intervals are development heuristics"
        ),
        "former_locked_seen_label_reads": 0,
        "case_group_ids": [data.case_group_ids[index] for index in train_rows],
        "geometry_group_ids": [data.geometry_group_ids[index] for index in train_rows],
        "section_families": [data.section_families[index] for index in train_rows],
        "fold_by_case": fold_by_case.tolist(),
        "seeds": list(seeds),
        "methods": {
            method: {"case_errors_by_seed": values.tolist()} for method, values in errors.items()
        },
        "training_records": training_records,
    }


def _parent_arrays(
    case_errors: np.ndarray,
    geometry_group_ids: Sequence[str],
    section_families: Sequence[str],
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...]]:
    values, parents, sections = aggregate_case_errors_by_parent(
        case_errors,
        geometry_group_ids,
        section_families,
    )
    if values.ndim != 2:
        raise DevelopmentProtocolError("seeded parent errors must have shape [seed,parent]")
    return values, parents, sections


def _section_equal_mean(values: np.ndarray, sections: Sequence[str]) -> float:
    array = np.asarray(values, dtype=np.float64)
    section_array = np.asarray(sections, dtype=object)
    if array.ndim not in (1, 2) or array.shape[-1] != section_array.size:
        raise DevelopmentProtocolError("section aggregation arrays do not align")
    return float(np.mean([np.mean(array[..., section_array == section]) for section in SECTIONS]))


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    seeds: Sequence[int],
    parents: Sequence[str],
    sections: Sequence[str],
    *,
    replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    point = _section_equal_mean(candidate, sections) / _section_equal_mean(reference, sections)
    heuristic = hierarchical_paired_bootstrap(
        candidate,
        reference,
        seeds,
        parents,
        sections,
        replicates=replicates,
        confidence=0.95,
        bootstrap_seed=bootstrap_seed,
    )
    section_array = np.asarray(sections, dtype=object)
    section_ratios = {
        section: float(
            np.mean(candidate[:, section_array == section])
            / np.mean(reference[:, section_array == section])
        )
        for section in SECTIONS
    }
    seed_ratios = []
    for index in range(len(seeds)):
        seed_ratios.append(
            _section_equal_mean(candidate[index], sections)
            / _section_equal_mean(reference[index], sections)
        )
    return {
        "point_ratio": float(point),
        "paired_parent_group_one_sided_upper_95": float(heuristic["one_sided_upper"]),
        "paired_parent_group_two_sided_95": [
            float(heuristic["lower"]),
            float(heuristic["upper"]),
        ],
        "interval_role": "development_heuristic_only_fold_dependence_precludes_formal_ci_claim",
        "section_ratios": section_ratios,
        "seed_ratios": [float(value) for value in seed_ratios],
    }


def analyze_cross_fit(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply frozen development heuristics to original-train OOF predictions."""

    seeds = tuple(int(value) for value in metrics["seeds"])
    geometry = tuple(map(str, metrics["geometry_group_ids"]))
    sections = tuple(map(str, metrics["section_families"]))
    methods = metrics["methods"]
    parent_values: dict[str, np.ndarray] = {}
    parent_ids: tuple[str, ...] | None = None
    parent_sections: tuple[str, ...] | None = None
    for method in METHODS:
        case_values = np.asarray(methods[method]["case_errors_by_seed"], dtype=np.float64)
        values, parents, values_sections = _parent_arrays(case_values, geometry, sections)
        if parent_ids is None:
            parent_ids, parent_sections = parents, values_sections
        elif parents != parent_ids or values_sections != parent_sections:
            raise DevelopmentProtocolError("method OOF parent identities do not align")
        parent_values[method] = values
    assert parent_ids is not None and parent_sections is not None
    gates = config["launch_gates"]["cross_fit_architecture_gate"]
    thresholds = {
        "generic_residual50_v03": float(
            gates["champion_to_same_fold_generic_residual50_max_point_ratio"]
        ),
        **{
            ablation: float(gates["champion_to_each_ablation_max_point_ratio"])
            for ablation in ABLATIONS
        },
        **{
            comparator: float(
                gates["champion_to_each_same_fold_learned_comparator_max_point_ratio"]
            )
            for comparator in SAME_FOLD_LEARNED_COMPARATORS
        },
    }
    comparisons: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    champion = parent_values["structured_linear_residual"]
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    for index, (reference, threshold) in enumerate(thresholds.items()):
        result = _comparison(
            champion,
            parent_values[reference],
            seeds,
            parent_ids,
            parent_sections,
            replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + index,
        )
        point_pass = result["point_ratio"] <= threshold
        upper_pass = result["paired_parent_group_one_sided_upper_95"] <= float(
            gates["paired_parent_group_one_sided_upper_ratio_max"]
        )
        section_pass = all(
            value
            <= float(
                config["launch_gates"]["cross_fit_section_robustness"][
                    "max_ratio_to_any_comparator"
                ]
            )
            for value in result["section_ratios"].values()
        )
        result.update(
            {
                "point_threshold": threshold,
                "point_pass": point_pass,
                "heuristic_upper_threshold": float(
                    gates["paired_parent_group_one_sided_upper_ratio_max"]
                ),
                "heuristic_upper_pass": upper_pass,
                "section_pass": section_pass,
            }
        )
        comparisons[reference] = result
        if not point_pass:
            failures.append(
                {
                    "code": "CROSS_FIT_POINT_MARGIN_FAILED",
                    "reference": reference,
                    "value": result["point_ratio"],
                    "threshold": threshold,
                }
            )
        if not upper_pass:
            failures.append(
                {
                    "code": "CROSS_FIT_HEURISTIC_UPPER_FAILED",
                    "reference": reference,
                    "value": result["paired_parent_group_one_sided_upper_95"],
                    "threshold": float(gates["paired_parent_group_one_sided_upper_ratio_max"]),
                }
            )
        if not section_pass:
            failures.append(
                {
                    "code": "CROSS_FIT_SECTION_FAILED",
                    "reference": reference,
                    "section_ratios": result["section_ratios"],
                }
            )
    seed_passes: list[bool] = []
    for seed_index in range(len(seeds)):
        seed_passes.append(
            all(
                comparisons[reference]["seed_ratios"][seed_index] <= threshold
                for reference, threshold in thresholds.items()
            )
        )
    minimum = int(config["launch_gates"]["seed_stability"]["minimum_passing_seeds"])
    if sum(seed_passes) < minimum:
        failures.append(
            {
                "code": "CROSS_FIT_SEED_STABILITY_FAILED",
                "passing_seed_count": sum(seed_passes),
                "minimum": minimum,
            }
        )
    return {
        "schema": "tunnelgeopt.multifidelity_v04_cross_fit_gate.v1",
        "scope": "development_only_architecture_gate",
        "effect_claim_allowed": False,
        "formal_confidence_interval_claim_allowed": False,
        "comparisons": comparisons,
        "seed_pass": {str(seed): passed for seed, passed in zip(seeds, seed_passes, strict=True)},
        "passing_seed_count": int(sum(seed_passes)),
        "minimum_passing_seed_count": minimum,
        "failures": failures,
        "passed": not failures,
    }


def final_fit_identity_contract(
    data: DevelopmentData,
    checkpoint_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Recover the exact original Residual50 36-train/18-dev identity contract."""

    seeds = tuple(int(value) for value in config["training_seeds"])
    expected_selection = config["final_development_fit"]["required_v03_selection_sha256"]
    contracts = checkpoint_manifest.get("contracts")
    if not isinstance(contracts, Mapping):
        raise DevelopmentProtocolError("v0.3 checkpoint manifest omits training contracts")
    train_sets: list[tuple[str, ...]] = []
    dev_sets: list[tuple[str, ...]] = []
    source_records: dict[str, Any] = {}
    for seed in seeds:
        key = f"residual_coarse__f050__seed{seed}"
        record = contracts.get(key)
        if not isinstance(record, Mapping):
            raise DevelopmentProtocolError(f"v0.3 Residual50 contract is missing: {key}")
        if (
            record.get("selection_sha256") != expected_selection
            or record.get("method") != "residual_coarse"
            or float(record.get("fine_fraction", -1.0)) != 0.5
            or record.get("selected_train_geometry_count") != 36
        ):
            raise DevelopmentProtocolError(f"v0.3 Residual50 identity drifted: {key}")
        train_sets.append(tuple(sorted(map(str, record["train_geometry_ids"]))))
        dev_sets.append(tuple(sorted(map(str, record["dev_geometry_ids"]))))
        source_records[str(seed)] = {
            "checkpoint_key": key,
            "selection_sha256": record["selection_sha256"],
            "contract_sha256": record["contract_sha256"],
        }
    if len(set(train_sets)) != 1 or len(set(dev_sets)) != 1:
        raise DevelopmentProtocolError("v0.3 Residual50 seeds used different parent identities")
    train_ids = train_sets[0]
    dev_ids = dev_sets[0]
    parent_records = _parent_records(data)
    if (
        len(train_ids) != 36
        or len(dev_ids) != 18
        or set(train_ids) & set(dev_ids)
        or any(parent_records[parent][0] != "train_id" for parent in train_ids)
        or any(parent_records[parent][0] != "dev_id" for parent in dev_ids)
    ):
        raise DevelopmentProtocolError("final fit identities violate train/dev roles")
    train_sections = {
        section: sum(parent_records[parent][1] == section for parent in train_ids)
        for section in SECTIONS
    }
    dev_sections = {
        section: sum(parent_records[parent][1] == section for parent in dev_ids)
        for section in SECTIONS
    }
    if set(train_sections.values()) != {12} or set(dev_sections.values()) != {6}:
        raise DevelopmentProtocolError("final fit must retain 12/section train and 6/section dev")
    forbidden = {
        parent for parent, (partition, _) in parent_records.items() if partition.startswith("seen_")
    }
    if forbidden & (set(train_ids) | set(dev_ids)):
        raise DevelopmentProtocolError("former-locked parent entered final fit identities")
    return {
        "schema": "tunnelgeopt.multifidelity_v04_final_identity_contract.v1",
        "source": "authenticated_v03_residual50_training_contract",
        "selection_sha256": expected_selection,
        "optimizer_parent_ids": list(train_ids),
        "optimizer_parent_count": 36,
        "optimizer_parents_per_section": train_sections,
        "normalization_fit_parent_ids": list(train_ids),
        "normalization_role": "identity_physics_normalization_no_seen_statistics_fitted",
        "early_stopping_parent_ids": list(dev_ids),
        "early_stopping_parent_count": 18,
        "early_stopping_parents_per_section": dev_sections,
        "former_locked_optimizer_intersection_count": 0,
        "former_locked_normalization_intersection_count": 0,
        "former_locked_early_stopping_intersection_count": 0,
        "source_seed_contracts": source_records,
        "passed": True,
    }


def _atomic_torch_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise DevelopmentProtocolError("PyTorch is required to save final checkpoints") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_sha256(path)


def run_final_development_fit(
    data: DevelopmentData,
    config: Mapping[str, Any],
    config_sha256: str,
    identity: Mapping[str, Any],
    *,
    device: str,
    output_dir: Path,
    tiny_mock: bool,
) -> dict[str, Any]:
    """Freeze five newly initialized development checkpoints before seen stress labels open."""

    if np.any(
        data.fine_available[
            np.isin(
                np.asarray(data.development_partitions),
                np.asarray(("seen_iid", "seen_geometry_ood", "seen_load_ood", "seen_joint_ood")),
            )
        ]
    ):
        raise DevelopmentProtocolError("former-locked fine labels opened before final fit")
    train_ids = tuple(map(str, identity["optimizer_parent_ids"]))
    dev_ids = tuple(map(str, identity["early_stopping_parent_ids"]))
    remaining_train = tuple(
        sorted(
            {
                parent
                for parent, partition in zip(
                    data.geometry_group_ids, data.development_partitions, strict=True
                )
                if partition == "train_id"
            }
            - set(train_ids)
        )
    )
    if len(remaining_train) != 36:
        raise DevelopmentProtocolError("unexpected remaining original-train parent count")
    checkpoint_dir = output_dir / "development_checkpoints"
    checkpoints: dict[str, Any] = {}
    seeds = tuple(int(value) for value in config["training_seeds"])
    architecture = _architecture_by_name(config, "structured_linear_residual")
    model_mapping = _model_mapping(config, architecture)
    for seed in seeds:
        path = checkpoint_dir / f"structured_linear_residual__seed{seed}.pt"
        if tiny_mock:
            payload = {
                "schema": "tunnelgeopt.multifidelity_v04_tiny_mock_checkpoint.v1",
                "effect_claim_allowed": False,
                "seed": seed,
                "config_sha256": config_sha256,
                "model_config": model_mapping,
                "optimizer_parent_ids_sha256": _sha256_value(list(train_ids)),
                "early_stopping_parent_ids_sha256": _sha256_value(list(dev_ids)),
            }
            path = path.with_suffix(".json")
            digest = _atomic_json(path, payload)
            training_record = {
                "best_epoch": 0,
                "epochs_run": 1,
                "best_dev_error": 0.0,
                "mock": True,
            }
        else:

            def progress(values: Mapping[str, Any], *, seed_value: int = seed) -> None:
                _append_jsonl(
                    output_dir / "training_progress.jsonl",
                    {
                        "at_utc": _now(),
                        "phase": "final_development_fit",
                        "seed": seed_value,
                        **dict(values),
                    },
                )

            _, model, training_record = _fit_predict_one(
                data,
                config,
                method_name="structured_linear_residual",
                seed=seed,
                train_parents=train_ids,
                dev_parents=dev_ids,
                predict_parents=remaining_train,
                device=device,
                progress=progress,
            )
            payload = {
                "schema": "tunnelgeopt.multifidelity_v04_development_checkpoint.v1",
                "effect_claim_allowed": False,
                "checkpoint_scope": "development_candidate_only_not_independent_validation",
                "seed": seed,
                "config_sha256": config_sha256,
                "model_config": model_mapping,
                "state_dict": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "optimizer_parent_ids": list(train_ids),
                "normalization_fit_parent_ids": list(train_ids),
                "early_stopping_parent_ids": list(dev_ids),
                "former_locked_label_reads_before_freeze": 0,
                "training_record": training_record,
            }
            digest = _atomic_torch_checkpoint(path, payload)
        checkpoints[str(seed)] = {
            "file": path.name,
            "sha256": digest,
            "seed": seed,
            "model_config": model_mapping,
            "optimizer_parent_count": 36,
            "early_stopping_parent_count": 18,
            "former_locked_label_reads_before_freeze": 0,
            "training_record": training_record,
        }
    if len(checkpoints) != 5 or len({record["sha256"] for record in checkpoints.values()}) != 5:
        raise DevelopmentProtocolError("final development checkpoint set is incomplete or aliased")
    return {
        "schema": "tunnelgeopt.multifidelity_v04_final_fit_manifest.v1",
        "scope": "development_candidate_only_not_independent_validation",
        "effect_claim_allowed": False,
        "config_sha256": config_sha256,
        "identity_contract_sha256": _sha256_value(dict(identity)),
        "identity_contract": dict(identity),
        "checkpoint_count": 5,
        "checkpoints_frozen_before_seen_stress_labels_open": True,
        "former_locked_label_reads_before_checkpoint_freeze": 0,
        "checkpoints": checkpoints,
        "frozen_at_utc": _now(),
    }


def _load_final_structured_model(
    output_dir: Path,
    record: Mapping[str, Any],
    *,
    device: str,
) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise DevelopmentProtocolError("PyTorch is required for seen-stress evaluation") from exc
    path = _resolve_inside(
        output_dir / "development_checkpoints", str(record["file"]), "checkpoint"
    )
    if not path.is_file() or _file_sha256(path) != record.get("sha256"):
        raise DevelopmentProtocolError("final structured checkpoint hash mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DevelopmentProtocolError("could not open final structured checkpoint") from exc
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != "tunnelgeopt.multifidelity_v04_development_checkpoint.v1"
        or payload.get("effect_claim_allowed") is not False
        or int(payload.get("seed", -1)) != int(record["seed"])
        or payload.get("config_sha256") is None
    ):
        raise DevelopmentProtocolError("final structured checkpoint envelope is invalid")
    module = _structured_module()
    model = module.make_structured_residual_model(
        payload["model_config"], seed=int(payload["seed"]), device="cpu"
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _load_frozen_v03_model(
    config: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    *,
    key: str,
    device: str,
) -> Any:
    source_root = _resolve_inside(ROOT, str(config["source_experiment"]["root"]), "source root")
    checkpoint_root = (source_root / "checkpoints").resolve()
    checkpoint = checkpoint_manifest.get("checkpoints", {}).get(key)
    contract = checkpoint_manifest.get("contracts", {}).get(key)
    if not isinstance(checkpoint, Mapping) or not isinstance(contract, Mapping):
        raise DevelopmentProtocolError(f"frozen v0.3 checkpoint record is missing: {key}")
    path = _resolve_inside(checkpoint_root, str(checkpoint.get("file")), key)
    if not path.is_file() or _file_sha256(path) != checkpoint.get("sha256"):
        raise DevelopmentProtocolError(f"frozen v0.3 checkpoint hash mismatch: {key}")
    model, payload = load_model_from_checkpoint(
        path,
        device=device,
        expected_config_sha256=str(config["source_experiment"]["config_sha256"]),
        expected_selection_sha256=str(contract["selection_sha256"]),
        require_formal=True,
    )
    if payload.get("training_contract_sha256") != contract.get("contract_sha256"):
        raise DevelopmentProtocolError(f"frozen v0.3 training contract mismatch: {key}")
    return model


def _wall_offset_discrepancy(
    prediction: np.ndarray,
    fine: np.ndarray,
    arc_weights: np.ndarray,
    normals_yz: np.ndarray,
    stress_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    fine = np.asarray(fine, dtype=np.float64)
    weights = np.asarray(arc_weights, dtype=np.float64)
    normals = np.asarray(normals_yz, dtype=np.float64)
    scales = np.asarray(stress_scales, dtype=np.float64)
    if (
        prediction.shape != fine.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 3
        or weights.shape != prediction.shape[:2]
        or normals.shape != (*prediction.shape[:2], 2)
        or scales.shape != (prediction.shape[0],)
    ):
        raise DevelopmentProtocolError("wall-offset arrays do not align")
    if (
        not all(np.isfinite(value).all() for value in (prediction, fine, weights, normals, scales))
        or np.any(weights < 0.0)
        or np.any(scales <= 0.0)
        or not np.allclose(weights.sum(axis=1), 1.0)
    ):
        raise DevelopmentProtocolError("wall-offset arrays are invalid")
    difference = (prediction - fine) * scales[:, None, None]
    tensor = np.empty((*difference.shape[:2], 2, 2), dtype=np.float64)
    tensor[..., 0, 0] = difference[..., 0]
    tensor[..., 1, 1] = difference[..., 1]
    tensor[..., 0, 1] = difference[..., 2]
    tensor[..., 1, 0] = difference[..., 2]
    traction = np.einsum("cpij,cpj->cpi", tensor, normals)
    d_t = np.sqrt(np.sum(weights * np.sum(traction**2, axis=-1), axis=1)) / scales
    resultant = np.sum(weights[..., None] * traction, axis=1)
    d_r = np.linalg.norm(resultant, axis=1) / scales
    if not np.isfinite(d_t).all() or not np.isfinite(d_r).all():
        raise DevelopmentProtocolError("wall-offset diagnostic is non-finite")
    return d_t, d_r


def _aggregate_wall(
    values: np.ndarray,
    geometry: Sequence[str],
    sections: Sequence[str],
) -> float:
    array = np.asarray(values, dtype=np.float64)
    parent_values, _, parent_sections = aggregate_case_errors_by_parent(array, geometry, sections)
    return _section_equal_mean(parent_values, parent_sections)


def _predict_frozen_generic(
    data: DevelopmentData,
    rows: np.ndarray,
    model: Any,
    method: str,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    batch = _batch_for_rows(data, rows, split="seen_stress")
    features, _, base = method_arrays(batch, method)
    raw = _predict_torch_model(model, features, batch_size=batch_size, device=device)
    return reconstruct_fine_prediction(raw, base)


def run_seen_stress_evaluation(
    data: DevelopmentData,
    config: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    final_manifest: Mapping[str, Any],
    *,
    device: str,
    output_dir: Path,
    tiny_mock: bool,
) -> dict[str, Any]:
    """Evaluate frozen final models once on all former-locked, now-seen labels."""

    if final_manifest.get("checkpoints_frozen_before_seen_stress_labels_open") is not True:
        raise DevelopmentProtocolError("final checkpoints were not frozen before seen stress")
    if not np.all(data.fine_available):
        raise DevelopmentProtocolError("seen-stress fine labels have not been opened")
    seeds = tuple(int(value) for value in config["training_seeds"])
    batch_size = int(config["optimization"]["case_batch_size"])
    partitions: dict[str, Any] = {}
    for partition in (*PRIMARY_PARTITIONS, "seen_joint_ood"):
        rows = np.flatnonzero(np.asarray(data.development_partitions) == partition)
        if rows.size == 0:
            raise DevelopmentProtocolError(f"seen-stress partition is empty: {partition}")
        coarse_errors = _case_errors(data, rows, data.coarse_stress[rows])
        coarse_dt, coarse_dr = _wall_offset_discrepancy(
            data.coarse_stress[rows],
            data.fine_stress[rows],
            data.arc_weights[rows],
            data.wall_normals[rows],
            data.stress_scales[rows],
        )
        case_errors = {
            "structured_linear_residual": np.empty((len(seeds), rows.size), dtype=np.float64),
            "frozen_v03_scratch100": np.empty((len(seeds), rows.size), dtype=np.float64),
            "frozen_v03_direct100": np.empty((len(seeds), rows.size), dtype=np.float64),
            "frozen_v03_generic_residual50": np.empty((len(seeds), rows.size), dtype=np.float64),
        }
        wall_dt = np.empty((len(seeds), rows.size), dtype=np.float64)
        wall_dr = np.empty((len(seeds), rows.size), dtype=np.float64)
        for seed_index, seed in enumerate(seeds):
            if tiny_mock:
                factors = {
                    "structured_linear_residual": 0.50,
                    "frozen_v03_scratch100": 0.72,
                    "frozen_v03_direct100": 0.70,
                    "frozen_v03_generic_residual50": 0.62,
                }
                for method, factor in factors.items():
                    case_errors[method][seed_index] = (
                        coarse_errors * factor * (1.0 + 0.002 * seed_index)
                    )
                wall_dt[seed_index] = 0.50 * coarse_dt
                wall_dr[seed_index] = 0.50 * coarse_dr
                continue
            final_record = final_manifest["checkpoints"][str(seed)]
            champion_model = _load_final_structured_model(output_dir, final_record, device=device)
            packed = _structured_features(data, rows)
            raw = _predict_torch_model(champion_model, packed, batch_size=batch_size, device=device)
            champion_prediction = reconstruct_fine_prediction(raw, data.coarse_stress[rows])
            case_errors["structured_linear_residual"][seed_index] = _case_errors(
                data, rows, champion_prediction
            )
            wall_dt[seed_index], wall_dr[seed_index] = _wall_offset_discrepancy(
                champion_prediction,
                data.fine_stress[rows],
                data.arc_weights[rows],
                data.wall_normals[rows],
                data.stress_scales[rows],
            )
            frozen_specs = (
                ("frozen_v03_scratch100", "scratch", 1.0),
                ("frozen_v03_direct100", "direct_coarse", 1.0),
                ("frozen_v03_generic_residual50", "residual_coarse", 0.5),
            )
            for label, method, fraction in frozen_specs:
                key = f"{method}__f{round(fraction * 100):03d}__seed{seed}"
                model = _load_frozen_v03_model(
                    config,
                    checkpoint_manifest,
                    key=key,
                    device=device,
                )
                prediction = _predict_frozen_generic(
                    data,
                    rows,
                    model,
                    method,
                    device=device,
                    batch_size=batch_size,
                )
                case_errors[label][seed_index] = _case_errors(data, rows, prediction)
        if any(not np.isfinite(values).all() for values in case_errors.values()):
            raise DevelopmentProtocolError("seen-stress model errors are incomplete")
        partitions[partition] = {
            "case_group_ids": [data.case_group_ids[index] for index in rows],
            "geometry_group_ids": [data.geometry_group_ids[index] for index in rows],
            "section_families": [data.section_families[index] for index in rows],
            "coarse_case_errors": coarse_errors.tolist(),
            "methods": {
                method: {"case_errors_by_seed": values.tolist()}
                for method, values in case_errors.items()
            },
            "wall_offset": {
                "coarse": {
                    "traction_case_values": coarse_dt.tolist(),
                    "resultant_case_values": coarse_dr.tolist(),
                },
                "structured_linear_residual": {
                    "traction_case_values_by_seed": wall_dt.tolist(),
                    "resultant_case_values_by_seed": wall_dr.tolist(),
                },
            },
        }
    return {
        "schema": "tunnelgeopt.multifidelity_v04_seen_stress_metrics.v1",
        "scope": "post_selection_former_locked_now_seen_development_stress_test",
        "effect_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "former_locked_labels_are_seen": True,
        "final_checkpoint_count": 5,
        "checkpoint_freeze_preceded_seen_label_open": True,
        "seeds": list(seeds),
        "partitions": partitions,
    }


def analyze_seen_stress(metrics: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Apply absolute, seed, section and wall gates to the seen stress test."""

    seeds = tuple(int(value) for value in metrics["seeds"])
    launch = config["launch_gates"]
    stress_gate = launch["post_selection_seen_stress_gate"]
    failures: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    for partition in (*PRIMARY_PARTITIONS, "seen_joint_ood"):
        record = metrics["partitions"][partition]
        geometry = tuple(map(str, record["geometry_group_ids"]))
        sections = tuple(map(str, record["section_families"]))
        candidate_case = np.asarray(
            record["methods"]["structured_linear_residual"]["case_errors_by_seed"],
            dtype=np.float64,
        )
        candidate, parents, parent_sections = _parent_arrays(candidate_case, geometry, sections)
        coarse_case = np.asarray(record["coarse_case_errors"], dtype=np.float64)
        coarse_parent_one, coarse_parents, coarse_sections = aggregate_case_errors_by_parent(
            coarse_case, geometry, sections
        )
        if coarse_parents != parents or coarse_sections != parent_sections:
            raise DevelopmentProtocolError("coarse and champion parent identities do not align")
        coarse = np.repeat(coarse_parent_one[None, :], len(seeds), axis=0)
        references: dict[str, np.ndarray] = {"coarse_only": coarse}
        for method in (
            "frozen_v03_scratch100",
            "frozen_v03_direct100",
            "frozen_v03_generic_residual50",
        ):
            values, method_parents, method_sections = _parent_arrays(
                np.asarray(record["methods"][method]["case_errors_by_seed"], dtype=np.float64),
                geometry,
                sections,
            )
            if method_parents != parents or method_sections != parent_sections:
                raise DevelopmentProtocolError("seen-stress method parent identities do not align")
            references[method] = values

        comparisons: dict[str, Any] = {}
        for index, (reference_name, reference) in enumerate(references.items()):
            comparisons[reference_name] = _comparison(
                candidate,
                reference,
                seeds,
                parents,
                parent_sections,
                replicates=int(config["evaluation"]["bootstrap_replicates"]),
                bootstrap_seed=int(config["evaluation"]["bootstrap_seed"]) + 100 + index,
            )

        primary = partition in PRIMARY_PARTITIONS
        if primary:
            margin = launch["absolute_safety_margins"][partition]
            learned_threshold = float(margin["max_point_ratio_to_each_learned_baseline"])
            coarse_threshold = float(margin["max_one_sided_upper_ratio_to_coarse"])
            generic_threshold = float(
                stress_gate["champion_to_frozen_v03_generic_residual50_max_point_ratio"]
            )
            for reference_name in ("frozen_v03_scratch100", "frozen_v03_direct100"):
                value = comparisons[reference_name]["point_ratio"]
                if value > learned_threshold:
                    failures.append(
                        {
                            "code": "SEEN_STRESS_LEARNED_MARGIN_FAILED",
                            "partition": partition,
                            "reference": reference_name,
                            "value": value,
                            "threshold": learned_threshold,
                        }
                    )
            generic_value = comparisons["frozen_v03_generic_residual50"]["point_ratio"]
            if generic_value > generic_threshold:
                failures.append(
                    {
                        "code": "SEEN_STRESS_GENERIC_MARGIN_FAILED",
                        "partition": partition,
                        "value": generic_value,
                        "threshold": generic_threshold,
                    }
                )
            coarse_upper = comparisons["coarse_only"]["paired_parent_group_one_sided_upper_95"]
            if coarse_upper > coarse_threshold:
                failures.append(
                    {
                        "code": "SEEN_STRESS_COARSE_UPPER_FAILED",
                        "partition": partition,
                        "value": coarse_upper,
                        "threshold": coarse_threshold,
                        "interval_role": "development_heuristic_only_seen_data_not_formal_ci",
                    }
                )

            seed_pass: list[bool] = []
            for seed_index in range(len(seeds)):
                seed_pass.append(
                    comparisons["frozen_v03_scratch100"]["seed_ratios"][seed_index]
                    <= learned_threshold
                    and comparisons["frozen_v03_direct100"]["seed_ratios"][seed_index]
                    <= learned_threshold
                    and comparisons["frozen_v03_generic_residual50"]["seed_ratios"][seed_index]
                    <= generic_threshold
                    and comparisons["coarse_only"]["seed_ratios"][seed_index] <= coarse_threshold
                )
            minimum = int(stress_gate["absolute_seed_stability_minimum"])
            if sum(seed_pass) < minimum:
                failures.append(
                    {
                        "code": "SEEN_STRESS_SEED_STABILITY_FAILED",
                        "partition": partition,
                        "passing_seed_count": sum(seed_pass),
                        "minimum": minimum,
                    }
                )

            section_pass: dict[str, bool] = {}
            for section in SECTIONS:
                section_pass[section] = (
                    comparisons["frozen_v03_scratch100"]["section_ratios"][section]
                    <= learned_threshold
                    and comparisons["frozen_v03_direct100"]["section_ratios"][section]
                    <= learned_threshold
                    and comparisons["frozen_v03_generic_residual50"]["section_ratios"][section]
                    <= generic_threshold
                    and comparisons["coarse_only"]["section_ratios"][section] <= coarse_threshold
                )
            if not all(section_pass.values()):
                failures.append(
                    {
                        "code": "SEEN_STRESS_SECTION_FAILED",
                        "partition": partition,
                        "section_pass": section_pass,
                    }
                )
        else:
            seed_pass = [False] * len(seeds)
            section_pass = {section: False for section in SECTIONS}

        wall = record["wall_offset"]
        coarse_dt = _aggregate_wall(
            np.asarray(wall["coarse"]["traction_case_values"], dtype=np.float64),
            geometry,
            sections,
        )
        coarse_dr = _aggregate_wall(
            np.asarray(wall["coarse"]["resultant_case_values"], dtype=np.float64),
            geometry,
            sections,
        )
        candidate_dt = _aggregate_wall(
            np.asarray(
                wall["structured_linear_residual"]["traction_case_values_by_seed"],
                dtype=np.float64,
            ),
            geometry,
            sections,
        )
        candidate_dr = _aggregate_wall(
            np.asarray(
                wall["structured_linear_residual"]["resultant_case_values_by_seed"],
                dtype=np.float64,
            ),
            geometry,
            sections,
        )
        wall_config = config["evaluation"]["wall_offset"]
        caps = wall_config["absolute_caps"][partition]
        wall_pass = (
            candidate_dt <= float(caps["max_traction_discrepancy"])
            and candidate_dr <= float(caps["max_resultant_discrepancy"])
            and candidate_dt
            <= float(wall_config["coarse_nonworsening_multiplier"]) * coarse_dt
            + float(wall_config["traction_additive_margin_over_S_inf"])
            and candidate_dr
            <= float(wall_config["coarse_nonworsening_multiplier"]) * coarse_dr
            + float(wall_config["resultant_additive_margin_over_S_inf"])
        )
        if primary and not wall_pass:
            failures.append(
                {
                    "code": "SEEN_STRESS_WALL_FAILED",
                    "partition": partition,
                    "candidate_D_t": candidate_dt,
                    "candidate_D_r": candidate_dr,
                    "coarse_D_t": coarse_dt,
                    "coarse_D_r": coarse_dr,
                }
            )
        results[partition] = {
            "role": "primary_seen_stress_gate" if primary else "seen_joint_report_only",
            "comparisons": comparisons,
            "seed_pass": {str(seed): passed for seed, passed in zip(seeds, seed_pass, strict=True)},
            "section_pass": section_pass,
            "wall_offset": {
                "coarse": {"D_t": coarse_dt, "D_r": coarse_dr},
                "structured_linear_residual": {
                    "D_t": candidate_dt,
                    "D_r": candidate_dr,
                },
                "passed": wall_pass,
            },
        }
    return {
        "schema": "tunnelgeopt.multifidelity_v04_seen_stress_gate.v1",
        "scope": "development_only_former_locked_now_seen",
        "effect_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "formal_confidence_interval_claim_allowed": False,
        "results": results,
        "failures": failures,
        "passed": not failures,
    }


def make_launch_decision(
    config: Mapping[str, Any],
    config_sha256: str,
    *,
    cross_fit_gate: Mapping[str, Any],
    seen_stress_gate: Mapping[str, Any] | None,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Return STOP or READY, neither of which is a confirmatory effect decision."""

    cross_fit_pass = cross_fit_gate.get("passed") is True
    stress_pass = seen_stress_gate is not None and seen_stress_gate.get("passed") is True
    real_authorized = config["execution_authorization"]["real_cross_fit_authorized"] is True
    ready = bool(cross_fit_pass and stress_pass and real_authorized)
    return {
        "schema": "tunnelgeopt.multifidelity_v04_launch_decision.v1",
        "protocol_id": config["protocol_id"],
        "config_sha256": config_sha256,
        "classification": (
            config["launch_gates"]["pass_classification"]
            if ready
            else config["launch_gates"]["fail_classification"]
        ),
        "effect_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "confirmatory_go_no_go_claim_allowed": False,
        "all_v03_labels_are_seen": True,
        "new_locked_data_created": False,
        "generator_invocation_count": 0,
        "cross_fit_gate_passed": cross_fit_pass,
        "final_fit_and_seen_stress_executed": seen_stress_gate is not None,
        "seen_stress_gate_passed": stress_pass,
        "real_cross_fit_authorized": real_authorized,
        "implementation_stop_pending_pivot": not real_authorized,
        "may_draft_new_locked_preregistration": ready,
        "future_preregistration_requirements": {
            "new_version": True,
            "new_salt": True,
            "new_identities": True,
            "exclude_all_705_v03_case_and_parent_identities": True,
            "threshold_rewrite_after_new_locked_open": False,
        },
        "artifact_sha256": dict(sorted(artifact_hashes.items())),
        "decided_at_utc": _now(),
    }


def _ensure_fresh_output(output_dir: Path) -> Path:
    root = Path(output_dir).resolve()
    if root == ROOT.resolve() or root == Path(root.anchor):
        raise DevelopmentProtocolError(
            "development output may not be a repository or filesystem root"
        )
    if root.exists() and any(root.iterdir()):
        raise DevelopmentProtocolError("development output directory must be new or empty")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_write(
    output_dir: Path,
    name: str,
    payload: Mapping[str, Any],
    artifact_hashes: dict[str, str],
) -> str:
    digest = _atomic_json(output_dir / name, payload)
    artifact_hashes[name] = digest
    return digest


def execute(
    *,
    config_path: Path,
    output_dir: Path,
    device: str,
    tiny_mock: bool = False,
    validate_only: bool = False,
) -> dict[str, Any]:
    """Execute validation or the complete fail-closed development workflow."""

    config, config_sha256 = load_development_config(config_path)
    if (
        not validate_only
        and not tiny_mock
        and config["execution_authorization"]["real_cross_fit_authorized"] is not True
    ):
        raise DevelopmentProtocolError(
            "real cross-fit is stopped pending an explicit pivot and new authorization"
        )
    data, checkpoint_manifest, input_audit = audit_and_load_inputs(config, config_sha256)
    fold_manifest = build_fold_manifest(data, config, config_sha256)
    if validate_only:
        return {
            "status": "validated",
            "protocol_id": config["protocol_id"],
            "config_sha256": config_sha256,
            "case_count": data.case_count,
            "parent_count": len(data.parent_ids),
            "cross_fit_outer_parent_count": fold_manifest["outer_oof_parent_count"],
            "former_locked_seen_fine_label_case_reads": 0,
            "new_locked_data_created": False,
            "effect_claim_allowed": False,
            "real_cross_fit_authorized": False,
            "implementation_status": config["status"],
        }

    preflight = (
        {
            "schema": "tunnelgeopt.multifidelity_v04_tiny_mock_preflight.v1",
            "config_sha256": config_sha256,
            "effect_claim_allowed": False,
            "production_compute": False,
            "passed": True,
        }
        if tiny_mock
        else production_preflight(config, config_sha256, device=device)
    )
    output = _ensure_fresh_output(output_dir)
    allowed = set(config["output_contract"]["allowed_artifacts"])
    expected_outputs = {
        "config_snapshot.json",
        "production_preflight.json",
        "input_audit.json",
        "fold_manifest.json",
        "cross_fit_metrics.json",
        "cross_fit_gate.json",
        "seen_stress_open_audit.json",
        "seen_stress_metrics.json",
        "seen_stress_gate.json",
        "launch_decision.json",
        "final_fit_manifest.json",
        "access_audit.jsonl",
        "development_checkpoints",
    }
    if not expected_outputs.issubset(allowed):
        raise DevelopmentProtocolError("output contract omits a runner artifact")
    artifact_hashes: dict[str, str] = {}
    access_log = output / "access_audit.jsonl"

    def event(name: str, **values: Any) -> None:
        _append_jsonl(
            access_log,
            {
                "at_utc": _now(),
                "protocol_id": config["protocol_id"],
                "config_sha256": config_sha256,
                "event": name,
                **values,
            },
        )

    _artifact_write(output, "config_snapshot.json", config, artifact_hashes)
    _artifact_write(output, "production_preflight.json", preflight, artifact_hashes)
    _artifact_write(output, "input_audit.json", input_audit, artifact_hashes)
    _artifact_write(output, "fold_manifest.json", fold_manifest, artifact_hashes)
    event(
        "development_inputs_authenticated",
        former_locked_seen_fine_label_case_reads=0,
        new_locked_data_created=False,
        generator_invocation_count=0,
    )
    cross_metrics = run_cross_fit(
        data,
        config,
        fold_manifest,
        device=device,
        output_dir=output,
        tiny_mock=tiny_mock,
    )
    _artifact_write(output, "cross_fit_metrics.json", cross_metrics, artifact_hashes)
    cross_gate = analyze_cross_fit(cross_metrics, config)
    _artifact_write(output, "cross_fit_gate.json", cross_gate, artifact_hashes)
    event(
        "cross_fit_completed",
        passed=cross_gate["passed"],
        former_locked_seen_fine_label_case_reads=0,
    )
    if not cross_gate["passed"]:
        decision = make_launch_decision(
            config,
            config_sha256,
            cross_fit_gate=cross_gate,
            seen_stress_gate=None,
            artifact_hashes=artifact_hashes,
        )
        _artifact_write(output, "launch_decision.json", decision, artifact_hashes)
        event("development_stopped_before_final_fit", classification=decision["classification"])
        return {
            "status": "completed",
            "classification": decision["classification"],
            "effect_claim_allowed": False,
            "output_dir": str(output),
            "launch_decision_sha256": artifact_hashes["launch_decision.json"],
        }

    identity = final_fit_identity_contract(data, checkpoint_manifest, config)
    final_manifest = run_final_development_fit(
        data,
        config,
        config_sha256,
        identity,
        device=device,
        output_dir=output,
        tiny_mock=tiny_mock,
    )
    _artifact_write(output, "final_fit_manifest.json", final_manifest, artifact_hashes)
    event(
        "final_development_checkpoints_frozen",
        checkpoint_count=5,
        former_locked_seen_fine_label_case_reads=0,
        final_fit_manifest_sha256=artifact_hashes["final_fit_manifest.json"],
    )
    data_with_seen, open_audit = open_seen_stress_labels(data, config)
    open_payload = {
        "schema": "tunnelgeopt.multifidelity_v04_seen_stress_open_audit.v1",
        "scope": "post_selection_former_locked_now_seen",
        "checkpoints_frozen_before_open": True,
        "former_locked_values_opened_before_cross_fit": False,
        "former_locked_values_opened_before_final_fit": False,
        "opened_partitions": open_audit,
        "opened_case_count": int(
            np.sum(data_with_seen.fine_available) - np.sum(data.fine_available)
        ),
        "independent_validation_claim_allowed": False,
        "passed": True,
    }
    _artifact_write(output, "seen_stress_open_audit.json", open_payload, artifact_hashes)
    event(
        "former_locked_seen_labels_opened",
        opened_case_count=open_payload["opened_case_count"],
        role="post_selection_seen_stress_only",
    )
    seen_metrics = run_seen_stress_evaluation(
        data_with_seen,
        config,
        checkpoint_manifest,
        final_manifest,
        device=device,
        output_dir=output,
        tiny_mock=tiny_mock,
    )
    _artifact_write(output, "seen_stress_metrics.json", seen_metrics, artifact_hashes)
    seen_gate = analyze_seen_stress(seen_metrics, config)
    _artifact_write(output, "seen_stress_gate.json", seen_gate, artifact_hashes)
    decision = make_launch_decision(
        config,
        config_sha256,
        cross_fit_gate=cross_gate,
        seen_stress_gate=seen_gate,
        artifact_hashes=artifact_hashes,
    )
    _artifact_write(output, "launch_decision.json", decision, artifact_hashes)
    event(
        "development_completed",
        classification=decision["classification"],
        effect_claim_allowed=False,
        independent_validation_claim_allowed=False,
        new_locked_data_created=False,
    )
    artifact_hashes["access_audit.jsonl"] = _file_sha256(access_log)
    return {
        "status": "completed",
        "classification": decision["classification"],
        "effect_claim_allowed": False,
        "independent_validation_claim_allowed": False,
        "new_locked_data_created": False,
        "output_dir": str(output),
        "launch_decision_sha256": artifact_hashes["launch_decision.json"],
        "access_audit_sha256": artifact_hashes["access_audit.jsonl"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="authenticate config/data/folds without writing or training",
    )
    parser.add_argument(
        "--tiny-mock",
        action="store_true",
        help="exercise the complete state/gate chain without scientific compute",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = execute(
            config_path=arguments.config,
            output_dir=arguments.output,
            device=str(arguments.device),
            tiny_mock=bool(arguments.tiny_mock),
            validate_only=bool(arguments.validate_only),
        )
    except (DevelopmentProtocolError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "failed", "error": str(exc)},
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
