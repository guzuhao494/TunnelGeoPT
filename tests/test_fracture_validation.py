from __future__ import annotations

import copy
import math

import pytest

from tunnelgeopt.fracture_validation import (
    PROTOCOL_ID,
    SCHEMA_VERSION,
    FracturePhase1ContractError,
    enumerate_fracture_phase1_cases,
    enumerate_ultrafine_audits,
    evaluate_trajectory_qc,
    load_fracture_phase1_config,
    validate_fracture_phase1_config,
)


def _passing_metrics(*, audit: bool = False) -> dict[str, float | int | bool]:
    metrics: dict[str, float | int | bool] = {
        "completed": True,
        "required_output_state_count": 41,
        "stored_accepted_step_count": 41,
        "required_output_s_complete": True,
        "nonfinite_fraction": 0.0,
        "max_equilibrium_relative_residual": 1e-6,
        "max_kkt_complementarity_relative_residual": 1e-6,
        "max_damage_irreversibility_violation": 1e-10,
        "max_damage_range_violation": 1e-10,
        "max_history_monotonicity_violation": 1e-10,
        "max_relative_energy_imbalance": 0.05,
        "min_incremental_dissipation_over_UCS_R2": -1e-10,
        "max_damage_on_refined_region_outer_edge": 1e-4,
        "all_steps_converged": True,
        "failure_ledger_complete": True,
        "retry_budget_exhausted": False,
        "replacement_attempts": 0,
    }
    if audit:
        metrics.update(
            {
                "fine_to_ultrafine_peak_reaction_relative_change": 0.05,
                "fine_to_ultrafine_total_fracture_energy_relative_change": 0.05,
            }
        )
    return metrics


def test_frozen_config_enumerates_exact_cross_product_and_ids() -> None:
    config = load_fracture_phase1_config()
    assert SCHEMA_VERSION == "tunnelgeopt.fracture.phase1.v3"
    assert PROTOCOL_ID == "fracture-phase1-development-pilot-v3"
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["protocol_id"] == PROTOCOL_ID
    cases = enumerate_fracture_phase1_cases(config)
    assert len(cases) == 36
    assert len({case.case_id for case in cases}) == 36
    assert cases[0].case_id == "fp1-circle-m1-p1"
    assert cases[-1].case_id == "fp1-straight_wall_arch-m3-p4"
    assert {case.section_family for case in cases} == {
        "circle",
        "horseshoe",
        "straight_wall_arch",
    }
    assert {case.material_level_id for case in cases} == {"m1", "m2", "m3"}
    assert {case.load_path_id for case in cases} == {"p1", "p2", "p3", "p4"}
    assert all(case.primary_mesh_tier == "fine" for case in cases)


def test_ultrafine_selection_covers_section_path_cells_with_balanced_latin_cycle() -> None:
    audits = enumerate_ultrafine_audits(load_fracture_phase1_config())
    assert len(audits) == 12
    assert len({(case.section_family, case.load_path_id) for case in audits}) == 12
    assert all(case.ultrafine_audit for case in audits)
    assert {
        material: sum(case.material_level_id == material for case in audits)
        for material in ("m1", "m2", "m3")
    } == {
        "m1": 4,
        "m2": 4,
        "m3": 4,
    }
    for case in audits:
        assert case.material_index == (case.section_index + case.load_path_index) % 3


def test_material_levels_preserve_explicit_dimensionless_strength_coupling() -> None:
    config = load_fracture_phase1_config()
    materials = config["materials"]
    ratio = materials["coupling"]["Gc_over_UCS_ell"]
    for level in materials["levels"]:
        assert level["Gc_over_UCS_R"] / level["ell_over_R"] == pytest.approx(ratio)
    expected_peak = math.sqrt(
        (27.0 / 256.0) * materials["fixed"]["youngs_modulus_over_UCS"] * ratio
    )
    assert materials["coupling"]["implied_sigma_peak_over_UCS"] == pytest.approx(expected_peak)
    assert config["mesh"]["tiers"]["fine"]["max_h_over_ell_in_potential_fracture_region"] <= 0.25


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("lifted_dynamics_per_train_case", 8),
        ("reference_velocity", "compressional_wave_speed_cp"),
        ("intermediate_stress_b", 0.5),
        ("max_cfl", 0.5),
        ("joint_family_count", 1),
        ("roughness_amplitude_over_radius", 0.0),
        ("random_field_seed", 7),
    ],
)
def test_legacy_stress_lift_3d_and_dynamic_fields_fail_closed(
    field_name: str, field_value: object
) -> None:
    config = load_fracture_phase1_config()
    changed = copy.deepcopy(config)
    changed["materials"][field_name] = field_value
    with pytest.raises(FracturePhase1ContractError, match="forbidden legacy"):
        validate_fracture_phase1_config(changed)


def test_scope_and_no_replacement_flags_are_not_relaxable() -> None:
    config = load_fracture_phase1_config()
    changed = copy.deepcopy(config)
    changed["scope"]["spatial_dimension"] = 3
    with pytest.raises(FracturePhase1ContractError, match="spatial_dimension"):
        validate_fracture_phase1_config(changed)

    changed = copy.deepcopy(config)
    changed["design"]["result_conditioned_replacement_allowed"] = True
    with pytest.raises(FracturePhase1ContractError, match="replacement"):
        validate_fracture_phase1_config(changed)


def test_moose_reference_and_same_problem_cross_check_are_independent_gates() -> None:
    config = load_fracture_phase1_config()
    prerequisites = config["quality_control"]["solver_prerequisites_before_pilot"]
    reference_key = "require_pinned_MOOSE_crack2d_iso_reference_self_test"
    cross_check_key = "require_local_vs_MOOSE_same_problem_cross_check"
    assert prerequisites[reference_key] is True
    assert prerequisites[cross_check_key] is True
    assert "require_MOOSE_crack2d_iso_cross_check" not in prerequisites

    for key in (reference_key, cross_check_key):
        changed = copy.deepcopy(config)
        changed["quality_control"]["solver_prerequisites_before_pilot"][key] = False
        with pytest.raises(FracturePhase1ContractError, match=key):
            validate_fracture_phase1_config(changed)

        changed = copy.deepcopy(config)
        del changed["quality_control"]["solver_prerequisites_before_pilot"][key]
        with pytest.raises(FracturePhase1ContractError, match="missing"):
            validate_fracture_phase1_config(changed)

    changed = copy.deepcopy(config)
    del changed["quality_control"]["solver_prerequisites_before_pilot"][reference_key]
    del changed["quality_control"]["solver_prerequisites_before_pilot"][cross_check_key]
    changed["quality_control"]["solver_prerequisites_before_pilot"][
        "require_MOOSE_crack2d_iso_cross_check"
    ] = True
    with pytest.raises(FracturePhase1ContractError, match="extra"):
        validate_fracture_phase1_config(changed)


def test_load_histories_and_required_outputs_are_actual_and_frozen() -> None:
    config = load_fracture_phase1_config()
    paths = config["load_paths"]["paths"]
    histories = [path["control_knots"] for path in paths]
    assert len(histories) == 4
    assert all(len(history) == 5 for history in histories)
    assert histories[0] != histories[1] != histories[2] != histories[3]
    assert set(histories[3][-1]["wall_release"]) == {
        "crown",
        "right_sidewall",
        "invert",
        "left_sidewall",
    }
    outputs = config["time_discretization"]["required_output_s"]
    assert len(outputs) == 41
    assert outputs == pytest.approx([index / 40.0 for index in range(41)])

    changed = copy.deepcopy(config)
    changed["load_paths"]["paths"][0]["control_knots"][2]["wall_release"]["all"] = 0.49
    with pytest.raises(FracturePhase1ContractError, match="frozen value"):
        validate_fracture_phase1_config(changed)


@pytest.mark.parametrize(
    ("old_key", "old_value"),
    [
        (
            "stress_components",
            "principal_compression_magnitudes_reported_positive_then_converted_to_solver_tension_positive_tensor",
        ),
        ("interpolation", "piecewise_linear_between_control_knots"),
    ],
)
def test_v3_rejects_old_ambiguous_load_path_fields(old_key: str, old_value: str) -> None:
    config = load_fracture_phase1_config()
    changed = copy.deepcopy(config)
    changed["load_paths"][old_key] = old_value
    with pytest.raises(FracturePhase1ContractError, match="extra"):
        validate_fracture_phase1_config(changed)

    changed = copy.deepcopy(config)
    del changed["load_paths"]["principal_angle_rule"]
    changed["load_paths"]["coordinate_rule"] = (
        "polar_angle_about_boundary_centroid_measured_from_positive_z_toward_positive_y"
    )
    with pytest.raises(FracturePhase1ContractError, match="missing.*principal_angle_rule"):
        validate_fracture_phase1_config(changed)


def test_v3_rejects_ambiguous_p4_transition_and_centroid_fields() -> None:
    config = load_fracture_phase1_config()
    zones = config["load_paths"]["wall_zones_for_p4"]
    assert zones["transition_total_width_deg"] == pytest.approx(5.0)
    assert "plus_or_minus_2.5deg" in zones["transition_rule"]

    changed = copy.deepcopy(config)
    old_zones = changed["load_paths"]["wall_zones_for_p4"]
    del old_zones["transition_total_width_deg"]
    old_zones["transition_blend_deg"] = 5.0
    with pytest.raises(FracturePhase1ContractError, match="transition_total_width_deg"):
        validate_fracture_phase1_config(changed)

    changed = copy.deepcopy(config)
    changed["load_paths"]["wall_zones_for_p4"]["centroid_rule"] = "vertex_arithmetic_mean"
    with pytest.raises(FracturePhase1ContractError, match="centroid_rule"):
        validate_fracture_phase1_config(changed)


def test_per_trajectory_qc_accepts_threshold_boundaries_without_replacement() -> None:
    config = load_fracture_phase1_config()
    result = evaluate_trajectory_qc(config, "fp1-circle-m1-p2", _passing_metrics())
    assert result.passed is True
    assert result.failed_checks == ()
    assert result.ultrafine_audit is False
    assert result.replacement_allowed is False
    with pytest.raises(TypeError):
        result.checks["finite"] = False  # type: ignore[index]


def test_per_trajectory_qc_records_failures_and_never_replaces_case() -> None:
    config = load_fracture_phase1_config()
    metrics = _passing_metrics()
    metrics["max_equilibrium_relative_residual"] = 1.01e-6
    metrics["retry_budget_exhausted"] = True
    metrics["replacement_attempts"] = 1
    result = evaluate_trajectory_qc(config, "fp1-circle-m1-p2", metrics)
    assert result.passed is False
    assert set(result.failed_checks) == {
        "equilibrium",
        "retry_budget_not_exhausted",
        "zero_replacement_attempts",
    }
    assert result.case_id == "fp1-circle-m1-p2"
    assert result.replacement_allowed is False


def test_qc_allows_stored_adaptive_substeps_beyond_required_states() -> None:
    config = load_fracture_phase1_config()
    metrics = _passing_metrics()
    metrics["stored_accepted_step_count"] = 44
    result = evaluate_trajectory_qc(config, "fp1-circle-m1-p2", metrics)
    assert result.passed
    assert result.checks["required_output_step_count"]
    assert result.checks["stored_all_required_and_adaptive_steps"]


def test_preselected_audit_requires_and_gates_both_refinement_metrics() -> None:
    config = load_fracture_phase1_config()
    audit_id = enumerate_ultrafine_audits(config)[0].case_id
    with pytest.raises(FracturePhase1ContractError, match="audit metrics are missing"):
        evaluate_trajectory_qc(config, audit_id, _passing_metrics())

    metrics = _passing_metrics(audit=True)
    metrics["fine_to_ultrafine_total_fracture_energy_relative_change"] = 0.0500001
    result = evaluate_trajectory_qc(config, audit_id, metrics)
    assert result.passed is False
    assert result.failed_checks == ("fine_to_ultrafine_fracture_energy",)


def test_qc_rejects_unknown_identity_and_missing_diagnostics() -> None:
    config = load_fracture_phase1_config()
    with pytest.raises(FracturePhase1ContractError, match="unknown or replacement"):
        evaluate_trajectory_qc(config, "fp1-circle-m1-p5", _passing_metrics())
    metrics = _passing_metrics()
    del metrics["failure_ledger_complete"]
    with pytest.raises(FracturePhase1ContractError, match="missing required keys"):
        evaluate_trajectory_qc(config, "fp1-circle-m1-p2", metrics)
