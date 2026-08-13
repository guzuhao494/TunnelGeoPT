from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "multifidelity_formal.json"
SECTIONS = ("circle", "horseshoe", "straight_wall_arch")
PARTITIONS = (
    "train_id",
    "dev_id",
    "locked_iid",
    "locked_geometry_ood",
    "locked_load_ood",
    "locked_joint_ood",
)


def _reject_duplicate_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_config() -> dict[str, Any]:
    return json.loads(
        CONFIG_PATH.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        object_pairs_hook=_reject_duplicate_pairs,
    )


def _walk_numbers(value: Any, path: str = "$") -> Iterable[tuple[str, float | int]]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield path, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_numbers(child, f"{path}[{index}]")


def _dig(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        value = value[key]
    return value


def test_formal_config_is_strict_finite_json_with_reproducible_hash_strategy() -> None:
    config = _strict_config()
    assert config["schema_version"] == "tunnelgeopt.multifidelity.formal.v1"
    assert config["hashing"]["algorithm"] == "sha256"
    assert config["hashing"]["embedded_digest"] is False

    numeric_values = list(_walk_numbers(config))
    assert numeric_values
    assert all(math.isfinite(float(value)) for _, value in numeric_values)

    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")
    assert config["hashing"]["digest_storage"] == "formal_manifest.config_sha256"


def test_frozen_status_allows_generation_but_claims_no_formal_run_or_locked_read() -> None:
    status = _strict_config()["status"]
    assert status == {
        "state": "frozen_preregistered_pre_generation",
        "formal_data_generated": False,
        "formal_effect_computation_started": False,
        "locked_labels_opened": False,
        "requires_dev_convergence_pass": False,
        "eligible_to_generate_formal_data": True,
        "test_driven_rewrite_prohibited": True,
    }
    assert _strict_config()["dataset"]["formal_data_manifest_status"] == "not_generated"


def test_partition_counts_are_exact_and_balanced_across_three_sections() -> None:
    config = _strict_config()
    partitions = config["dataset"]["partitions"]
    assert tuple(partitions) == PARTITIONS
    assert tuple(config["geometry"]["section_families"]) == SECTIONS

    parents = 0
    cases = 0
    for partition in partitions.values():
        assert partition["parent_geometries"] == 3 * partition["parents_per_section"]
        assert partition["cases"] == (
            partition["parent_geometries"] * partition["loads_per_parent"]
        )
        parents += partition["parent_geometries"]
        cases += partition["cases"]
    assert parents == config["dataset"]["total_parent_geometries"] == 195
    assert cases == config["dataset"]["total_cases"] == 705


def test_frozen_seeds_nested_fractions_and_35_checkpoint_matrix_are_exact() -> None:
    config = _strict_config()
    assert config["identity_and_split"]["split_salt"] == (
        "tunnelgeopt-v0.3-mf-residual-20260813-v1"
    )
    assert config["dataset"]["generator_seeds"] == {
        "train_dev": 310031,
        "locked_iid": 310037,
        "locked_geometry_ood": 310049,
        "locked_load_ood": 310061,
        "locked_joint_ood": 310073,
    }

    learning = config["learning"]
    assert learning["training_seeds"] == [103, 211, 307, 401, 509]
    assert learning["fine_train_fractions"] == [0.25, 0.5, 0.75, 1.0]
    assert learning["strictly_nested_parent_subsets"] is True
    previous_total = 0
    previous_per_section = 0
    for fraction in learning["fine_train_fractions"]:
        counts = learning["nested_parent_counts"][str(fraction)]
        assert counts["total"] == round(72 * fraction)
        assert counts["per_section"] == round(24 * fraction)
        assert counts["total"] > previous_total
        assert counts["per_section"] > previous_per_section
        previous_total = counts["total"]
        previous_per_section = counts["per_section"]

    matrix = learning["method_fraction_matrix"]
    assert matrix == {
        "scratch": [1.0],
        "direct_coarse": [1.0],
        "residual_coarse": [0.25, 0.5, 0.75, 1.0],
        "mismatched_coarse": [0.5],
    }
    checkpoints_per_seed = sum(len(fractions) for fractions in matrix.values())
    assert checkpoints_per_seed == 7
    assert learning["expected_checkpoint_count"] == checkpoints_per_seed * 5 == 35
    assert config["sealed_evaluation"]["checkpoint_count_required_before_authorization"] == 35


def test_mesh_query_solver_and_convergence_thresholds_are_complete() -> None:
    config = _strict_config()
    tiers = config["mesh"]["tiers"]
    assert tuple(tiers) == ("coarse", "fine", "ultrafine_audit")
    for tier in tiers.values():
        assert set(tier) == {
            "mesh_size_over_radius",
            "wall_size_over_radius",
            "farfield_size_over_radius",
        }
        assert all(float(value) > 0.0 for value in tier.values())
    assert tiers["ultrafine_audit"]["farfield_size_over_radius"] <= 0.25

    query = config["query"]
    assert query["points_per_case"] == (
        query["nearfield_volume"] + query["wall_offset"] + query["farfield"]
    )
    assert query["wall_offset_over_radius"] == pytest.approx(0.02)

    required = {
        "quality_control.solver_and_mesh.max_nonfinite_fraction": 0.0,
        "quality_control.solver_and_mesh.max_free_dof_algebraic_residual": 1e-9,
        "quality_control.solver_and_mesh.max_clapeyron_relative_energy_error": 1e-9,
        "quality_control.solver_and_mesh.min_triangle_signed_area_over_radius_squared": 1e-12,
        "quality_control.solver_and_mesh.min_triangle_quality": 0.02,
        "quality_control.solver_and_mesh.minimum_valid_case_fraction_per_partition_section": 0.95,
        "quality_control.fine_ultrafine.formal_audit_fraction": 0.2,
        "quality_control.fine_ultrafine.max_overall_median": 0.03,
        "quality_control.fine_ultrafine.max_overall_p95": 0.05,
        "quality_control.fine_ultrafine.max_any_section_median": 0.04,
    }
    for path, expected in required.items():
        value = _dig(config, path)
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        assert math.isfinite(float(value))
        assert value == pytest.approx(expected)

    audit = config["quality_control"]["fine_ultrafine"]
    assert audit["independent_dev_convergence_pass_recorded"] is True
    expected_audit_cases = 0
    for partition in config["dataset"]["partitions"].values():
        cases_per_section = partition["parents_per_section"] * partition["loads_per_parent"]
        expected_audit_cases += 3 * math.ceil(audit["formal_audit_fraction"] * cases_per_section)
    assert expected_audit_cases == audit["expected_formal_audit_cases"] == 144
    assert audit["failure_classification"] == "ABSTAIN"


def test_dev_only_evidence_is_hashed_and_optimization_cap_is_frozen_preformal() -> None:
    config = _strict_config()
    evidence = config["preformal_development_evidence"]
    convergence = evidence["mesh_convergence"]
    assert convergence["run_id"] == "mf-convergence-dev-v0.3.0"
    for name in (
        "config_canonical_sha256",
        "config_snapshot_file_sha256",
        "manifest_file_sha256",
        "metrics_file_sha256",
    ):
        digest = convergence[name]
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    assert convergence["case_count"] == 24
    assert convergence["locked_test_fine_label_case_reads"] == 0
    assert convergence["overall_median"] <= 0.03
    assert convergence["overall_p95"] <= 0.05
    assert max(convergence["family_medians"].values()) <= 0.04
    assert convergence["decision"] == "current_tiers_eligible_for_formal_freeze"

    calibration = evidence["optimization_calibration"]
    assert calibration["formal_locked_label_reads"] == 0
    assert calibration["candidate_max_epochs"] == 200
    assert calibration["observed_best_epochs"] == {
        "scratch_100": 194,
        "direct_coarse_100": 198,
        "residual_coarse_50": 138,
    }
    assert calibration["observed_residual_coarse_50_stop_epoch"] == 164
    assert calibration["all_other_optimization_fields_unchanged"] is True
    assert calibration["effect_gate_changed"] is False
    optimization = config["learning"]["optimization"]
    assert optimization["max_epochs"] == 300
    assert optimization["early_stopping_patience"] == 35
    assert optimization["early_stopping_min_delta"] == pytest.approx(1e-5)


def test_wall_offset_diagnostic_is_relative_to_fine_not_false_wall_equilibrium() -> None:
    config = _strict_config()
    physics = config["evaluation"]["wall_offset_physics"]
    assert physics["scope"].startswith("wall_offset_queries_at_0.02R")
    assert set(physics["explicitly_not"]) == {
        "absolute_traction_free_residual",
        "exact_wall_traction",
        "global_domain_equilibrium_residual",
    }
    assert "Sigma_M_i-Sigma_F_i" in physics["traction_discrepancy_formula"]
    assert "Sigma_M_i-Sigma_F_i" in physics["resultant_discrepancy_formula"]
    assert physics["require_both_absolute_and_coarse_nonworsening"] is True
    nonworsening = physics["coarse_nonworsening"]
    assert nonworsening["max_multiplier"] == pytest.approx(1.1)
    assert nonworsening["traction_additive_margin_over_S_inf"] == pytest.approx(0.005)
    assert nonworsening["resultant_additive_margin_over_S_inf"] == pytest.approx(0.0025)
    for caps in physics["absolute_caps"].values():
        assert 0.0 < caps["max_traction_discrepancy"] <= 0.2
        assert 0.0 < caps["max_resultant_discrepancy"] <= 0.1


def test_scientific_gates_and_abstain_semantics_have_no_numeric_holes() -> None:
    config = _strict_config()
    gates = config["scientific_decision"]
    assert gates["status"] == "not_run"
    assert gates["upper_95_ci_gates"] == {
        "locked_iid": {"R_s": 1.02, "R_d": 1.02, "R_c": 0.7},
        "locked_geometry_ood": {"R_s": 1.05, "R_d": 1.05, "R_c": 0.8},
        "locked_load_ood": {"R_s": 1.05, "R_d": 1.05, "R_c": 0.8},
    }
    assert gates["seed_stability"]["minimum_passing_seeds"] == 4
    assert gates["seed_stability"]["total_seeds"] == 5
    assert gates["section_robustness"]["minimum_iid_sections_at_strict_gate"] == 2
    assert gates["classification"] == {
        "GO": "all_validity_and_effect_gates_pass",
        "NO_GO": "experiment_valid_but_one_or_more_effect_or_robustness_gates_fail",
        "ABSTAIN": (
            "leakage_test_tuning_convergence_solver_mesh_qc_seed_count_interval_width_or_"
            "evaluation_contract_invalid"
        ),
    }
    assert gates["after_locked_labels_open"]["threshold_or_method_rewrite_allowed"] is False
    assert gates["after_locked_labels_open"]["debug_on_locked_values_allowed"] is False


def test_file_level_sealing_training_contract_and_registry_are_mandatory() -> None:
    config = _strict_config()
    sealed = config["sealed_evaluation"]
    assert "file_store" in sealed["security_boundary"]
    assert sealed["trainer_must_not_receive_locked_label_path"] is True
    assert sealed["max_evaluation_calls_per_locked_partition"] == 1
    assert sealed["individual_locked_metrics_available_to_training_process"] is False
    integrity = config["learning"]["formal_integrity"]
    assert integrity["required_training_contract_type"] == "TrainingContract"
    assert integrity["required_checkpoint_registry_type"] == "CheckpointRegistry"
    assert integrity["require_checkpoint_sha256_unique"] is True
    assert integrity["require_atomic_checkpoint_write"] is True
