from __future__ import annotations

import copy
import json
from itertools import pairwise

import pytest

from tunnelgeopt.fracture_benchmark_validation import (
    FROZEN_CANONICAL_SHA256,
    PROTOCOL_ID,
    SCHEMA_VERSION,
    FractureBenchmarkContractError,
    enumerate_fracture_benchmark_cases,
    load_fracture_sent_sens_config,
    prescribed_displacements,
    validate_fracture_sent_sens_config,
)


def test_frozen_mesh_verified_solver_not_run_status_and_six_case_order() -> None:
    config = load_fracture_sent_sens_config()
    assert SCHEMA_VERSION == "tunnelgeopt.fracture.sent_sens.development_protocol.v1"
    assert PROTOCOL_ID == "miehe-sent-sens-three-grid-development-v1.1"
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["protocol_id"] == PROTOCOL_ID
    assert len(FROZEN_CANONICAL_SHA256) == 64
    assert config["status"] == {
        "state": "protocol_frozen_mesh_contract_verified_solver_not_run",
        "development_only": True,
        "mesh_contract_implemented_and_audited": True,
        "mesh_contract_verification_scope": (
            "six_Gmsh_meshes_SENT_SENS_x_coarse_medium_fine_topology_labels_and_size_contract_only"
        ),
        "fracture_solver_runs_completed": False,
        "benchmark_results_available": False,
        "current_decision": "ABSTAIN_NOT_RUN",
        "phase1_pilot_authorized": False,
        "formal_or_paper_claim_allowed": False,
    }
    cases = enumerate_fracture_benchmark_cases(config)
    assert [case.case_id for case in cases] == [
        "fss1-sent-coarse",
        "fss1-sent-medium",
        "fss1-sent-fine",
        "fss1-sens-coarse",
        "fss1-sens-medium",
        "fss1-sens-fine",
    ]
    assert len({case.case_id for case in cases}) == 6


def test_yz_coordinate_slit_and_boundary_conditions_are_not_swapped() -> None:
    config = load_fracture_sent_sens_config()
    assert config["coordinate_system"]["order"] == "y_vertical_then_z_horizontal"
    assert config["coordinate_system"]["domain_bounds_mm"] == {
        "y": [0.0, 1.0],
        "z": [0.0, 1.0],
    }
    notch = config["geometry"]["notch"]
    assert notch["line_mm"] == {"y": 0.5, "z": [0.0, 0.5]}
    assert notch["upper_and_lower_face_nodes_distinct_except_shared_tip"] is True
    assert notch["initial_damage_d0"] == 0.0
    assert notch["initial_history_H0_kN_per_mm2"] == 0.0

    sent, sens = config["loading"]["benchmarks"]
    assert sent["top_y1"] == {"u_y_mm": "U", "u_z_mm": 0.0}
    assert sent["reaction_component"] == "sum_top_reaction_y_kN"
    assert sens["top_y1"] == {"u_y_mm": 0.0, "u_z_mm": "U"}
    assert sens["reaction_component"] == "sum_top_reaction_z_kN"
    assert config["loading"]["common"]["bottom_y0"] == {
        "u_y_mm": 0.0,
        "u_z_mm": 0.0,
    }


def test_primary_facts_supplemental_choices_and_digitized_windows_are_separate() -> None:
    config = load_fracture_sent_sens_config()
    evidence = config["evidence_basis"]
    primary = evidence["primary_source"]
    assert primary["doi"] == "10.1016/j.cma.2010.04.011"
    assert "lame_lambda_121.15_kN_per_mm2" in primary["facts_used"]
    assert "residual_stiffness_k_1e-8" in evidence["tunnelgeopt_supplemental_conventions"]
    assert evidence["digitized_sanity_envelopes"]["not_reference_arrays_or_exact_gold"] is True
    assert evidence["public_implementations"]["role"] == (
        "implementation_cross_read_only_not_numerical_gold"
    )
    assert config["scope"]["claim_boundary"] == (
        "development_validation_only_not_exact_Miehe_reproduction"
    )


def test_material_model_is_frozen_with_supplemental_residual_stiffness() -> None:
    config = load_fracture_sent_sens_config()
    assert config["material"] == {
        "homogeneous_isotropic": True,
        "lame_lambda_kN_per_mm2": 121.15,
        "shear_modulus_kN_per_mm2": 80.77,
        "critical_fracture_energy_kN_per_mm": 0.0027,
        "regularization_length_ell_mm": 0.015,
        "viscosity_eta": 0.0,
    }
    assert config["fracture_model"]["residual_stiffness_k"] == 1e-8
    assert config["fracture_model"]["model"] == "AT2"


def test_formal_displacement_grids_have_exact_transition_and_endpoint_counts() -> None:
    config = load_fracture_sent_sens_config()
    sent = prescribed_displacements(config, "sent")
    sens = prescribed_displacements(config, "sens")
    assert len(sent) == 2001
    assert sent[0] == 0.0
    assert sent[500] == 0.005
    assert sent[501] - sent[500] == pytest.approx(1e-6)
    assert sent[-1] == 0.0065
    assert len(sens) == 1501
    assert sens[0] == 0.0
    assert sens[1] - sens[0] == pytest.approx(1e-5)
    assert sens[-1] == 0.015
    assert all(right > left for left, right in pairwise(sent))


def test_compute_preflight_grids_are_exactly_ten_times_coarser() -> None:
    config = load_fracture_sent_sens_config()
    sent = prescribed_displacements(config, "sent", compute_preflight=True)
    sens = prescribed_displacements(config, "sens", compute_preflight=True)
    assert len(sent) == 201
    assert sent[50] == 0.005
    assert sent[51] - sent[50] == pytest.approx(1e-5)
    assert sent[-1] == 0.0065
    assert len(sens) == 151
    assert sens[1] - sens[0] == pytest.approx(1e-4)
    assert sens[-1] == 0.015


def test_three_mesh_tiers_follow_ell_and_bulk_rules() -> None:
    config = load_fracture_sent_sens_config()
    cases = enumerate_fracture_benchmark_cases(config)
    sent_cases = [case for case in cases if case.benchmark_id == "sent"]
    assert [case.h_target_over_ell for case in sent_cases] == [0.5, 0.25, 0.125]
    assert [case.h_target_mm for case in sent_cases] == [0.0075, 0.00375, 0.001875]
    assert [case.bulk_h_target_mm for case in sent_cases] == [0.03, 0.015, 0.0075]
    assert config["mesh"]["refined_corridor_max_edge_over_target"] == 1.15
    assert config["mesh"]["damage_component_threshold"] == 0.5
    assert config["mesh"]["damage_escape_action"] == "STOP_INVALID"
    sent, sens = config["loading"]["benchmarks"]
    assert sent["refined_corridor"] == {
        "definition": "buffer_around_horizontal_tip_to_right_boundary_segment",
        "centerline_yz_mm": [[0.5, 0.5], [0.5, 1.0]],
        "half_width_mm": 0.1,
        "notch_face_and_tip_refinement_distance_mm": 0.05,
    }
    assert sens["refined_corridor"] == {
        "definition": "buffer_around_tip_to_lower_right_diagonal_segment",
        "centerline_yz_mm": [[0.5, 0.5], [0.0, 1.0]],
        "half_width_mm": 0.15,
        "notch_face_and_tip_refinement_distance_mm": 0.05,
    }


def test_topology_and_per_tier_qc_fail_closed_before_convergence_claim() -> None:
    config = load_fracture_sent_sens_config()
    topology = config["topology_qc"]
    assert topology["any_failure_action"] == "STOP_INVALID"
    assert config["mesh"]["damage_escape_action"] == topology["any_failure_action"]
    assert "no_triangle_crosses_the_open_slit" in topology["required"]
    assert (
        "every_open_notch_face_facet_has_exactly_one_adjacent_rock_element" in topology["required"]
    )
    qc = config["per_tier_qc"]
    assert qc["max_equilibrium_relative_residual"] == 1e-8
    assert qc["max_kkt_complementarity_relative_residual"] == 1e-8
    assert qc["max_relative_displacement_increment"] == 1e-8
    assert qc["max_relative_damage_increment"] == 1e-8
    assert qc["max_relative_potential_energy_increment"] == 1e-8
    assert qc["max_damage_irreversibility_violation"] == 1e-12
    assert qc["max_global_force_relative_imbalance"] == 1e-8
    assert qc["max_global_moment_relative_imbalance"] == 1e-8
    assert qc["max_path_energy_relative_imbalance"] == 0.05
    assert qc["require_no_accepted_nonconverged_step"] is True


def test_three_grid_curve_path_and_monotonicity_gates_are_frozen() -> None:
    config = load_fracture_sent_sens_config()
    convergence = config["three_grid_convergence"]
    medium_fine = convergence["medium_to_fine"]
    assert medium_fine == {
        "max_peak_reaction_relative_change": 0.05,
        "max_final_fracture_energy_relative_change": 0.05,
        "max_reaction_curve_relative_L2": 0.05,
        "max_peak_displacement_relative_change": 0.05,
        "max_symmetric_crack_path_Hausdorff_over_ell": 0.5,
    }
    assert convergence["monotonicity"]["required_for_each_metric"] is True
    assert convergence["any_failure_action"] == "STOP_NUMERICAL"
    assert "d_eq_0.95" in convergence["crack_path_rule"]


def test_digitized_windows_are_sanity_screens_not_exact_gold() -> None:
    windows = load_fracture_sent_sens_config()["digitized_sanity_windows"]
    assert windows["sent"] == {
        "peak_reaction_kN": [0.6, 0.82],
        "peak_displacement_mm": [0.005, 0.0061],
    }
    assert windows["sens"] == {
        "peak_reaction_kN": [0.44, 0.62],
        "peak_displacement_mm": [0.0075, 0.0115],
    }
    assert "not_exact_gold" in windows["role"]
    assert windows["mismatch_action"] == "ABSTAIN_REFERENCE_AMBIGUITY"


def test_compute_gate_uses_measured_same_tier_steps_and_no_invented_dof_exponent() -> None:
    compute = load_fracture_sent_sens_config()["compute_preflight"]
    assert compute["projection_method"] == (
        "linear_in_formal_load_increment_count_times_measured_median_accepted_step_wall_time_"
        "on_same_benchmark_and_tier"
    )
    assert compute["unmeasured_DOF_scaling_exponent_allowed"] is False
    assert compute["timing_probe"]["minimum_measured_accepted_steps_per_benchmark_and_tier"] == 10
    assert "explicit_fixed_10_step_probe" in compute["timing_probe"]["medium_source"]
    assert compute["max_projected_single_medium_case_wall_hours"] == 12.0
    assert compute["max_projected_all_six_cases_wall_hours"] == 72.0
    assert compute["threshold_failure_action"] == "ABSTAIN_COMPUTE_OPTIMIZE_BEFORE_RERUN"


def test_decision_routes_do_not_promote_benchmark_readiness_to_paper_go() -> None:
    decision = load_fracture_sent_sens_config()["decision"]
    assert decision["precedence"] == [
        "STOP_INVALID",
        "ABSTAIN_COMPUTE_OPTIMIZE_BEFORE_RERUN",
        "ABSTAIN_REFERENCE_AMBIGUITY",
        "STOP_NUMERICAL",
        "READY_FOR_PHASE1_PILOT",
    ]
    assert decision["go_family_code"] == "READY_FOR_PHASE1_PILOT"
    assert decision["go_is_not_paper_or_field_GO"] is True
    assert decision["not_run_code"] == "ABSTAIN_NOT_RUN"


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update({"unexpected": 1}), "top-level"),
        (lambda value: value["coordinate_system"].update({"order": "z_then_y"}), "coordinate"),
        (lambda value: value["material"].update({"regularization_length_ell_mm": 0.02}), "ell"),
        (
            lambda value: value["status"].update({"benchmark_results_available": True}),
            "unrun fracture benchmarks",
        ),
        (
            lambda value: value["compute_preflight"].update(
                {"unmeasured_DOF_scaling_exponent_allowed": True}
            ),
            "DOF",
        ),
        (
            lambda value: value["mesh"].update(
                {"damage_escape_action": "STOP_REMESH_BEFORE_RERUN"}
            ),
            "damage corridor escape",
        ),
    ],
)
def test_semantic_mutations_fail_closed(mutator: object, match: str) -> None:
    changed = copy.deepcopy(load_fracture_sent_sens_config())
    mutator(changed)  # type: ignore[operator]
    with pytest.raises(FractureBenchmarkContractError, match=match):
        validate_fracture_sent_sens_config(changed)


def test_any_other_nested_change_is_rejected_by_canonical_hash() -> None:
    changed = copy.deepcopy(load_fracture_sent_sens_config())
    changed["evidence_basis"]["primary_source"]["citation"] += " changed"
    with pytest.raises(FractureBenchmarkContractError, match="canonical"):
        validate_fracture_sent_sens_config(changed)


def test_nonfinite_and_unknown_benchmark_fail_closed() -> None:
    config = load_fracture_sent_sens_config()
    changed = copy.deepcopy(config)
    changed["mesh"]["tiers"][0]["h_target_mm"] = float("nan")
    with pytest.raises(FractureBenchmarkContractError, match="finite"):
        validate_fracture_sent_sens_config(changed)
    with pytest.raises(FractureBenchmarkContractError, match="benchmark_id"):
        prescribed_displacements(config, "unknown")


def test_loader_rejects_a_changed_file(tmp_path: object) -> None:
    config = load_fracture_sent_sens_config()
    config["decision"]["go_is_not_paper_or_field_GO"] = False
    path = tmp_path / "changed.json"  # type: ignore[operator]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FractureBenchmarkContractError, match="paper GO"):
        load_fracture_sent_sens_config(path)
