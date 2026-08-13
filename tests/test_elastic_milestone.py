from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest

from scripts.run_elastic_milestone import (
    DEFAULT_CONFIG,
    MilestoneConfigError,
    build_physical_cases,
    compression_positive_inplane_tensor,
    evaluate_kirsch_gate,
    load_config,
    prepare_run_contract,
    solver_tension_positive_stress,
    validate_frozen_config,
)
from tunnelgeopt.cases import load_case_manifest


def test_frozen_config_expands_exactly_eighteen_deterministic_physical_cases() -> None:
    config, first_hash = load_config(DEFAULT_CONFIG)
    cases = build_physical_cases(config)
    repeated = build_physical_cases(config)

    assert cases == repeated
    assert len(cases) == 18
    assert len({json.dumps(case, sort_keys=True) for case in cases}) == 18
    assert len(first_hash) == 64
    for family in ("circle", "horseshoe", "straight_wall_arch"):
        section = [case for case in cases if case["section_family"] == family]
        assert len(section) == 6
        for field in (
            "roughness_amplitude_over_radius",
            "young_modulus_over_reference_stress",
            "poisson_ratio",
        ):
            if field == "roughness_amplitude_over_radius":
                values = [case["section_parameters"][field] for case in section]
            else:
                values = [case["dimensionless_material_parameters"][field] for case in section]
            assert values.count(min(values)) == 2
            assert len(set(values)) == 3


def test_stress_rotation_and_compression_to_tension_conversion_are_explicit() -> None:
    compression = compression_positive_inplane_tensor(0.7, 0.21, 45.0)
    solver = solver_tension_positive_stress(compression)

    assert np.allclose(np.linalg.eigvalsh(compression), [0.21, 0.7])
    assert np.allclose(solver, -compression)
    assert np.trace(solver) < 0.0
    assert np.allclose(solver, solver.T)


def test_relaxed_frozen_gates_are_rejected() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    relaxed_kirsch = deepcopy(config)
    relaxed_kirsch["kirsch_validation"]["max_fine_area_weighted_stress_relative_l2"] = 0.081
    with pytest.raises(MilestoneConfigError, match="may not exceed 0.08"):
        validate_frozen_config(relaxed_kirsch)

    relaxed_qc = deepcopy(config)
    relaxed_qc["quality_control"]["max_free_dof_algebraic_residual"] = 1.1e-9
    with pytest.raises(MilestoneConfigError, match="may not be relaxed"):
        validate_frozen_config(relaxed_qc)


def test_prepare_writes_verified_parent_and_derived_manifest_before_solver(tmp_path) -> None:
    prepared = prepare_run_contract(DEFAULT_CONFIG, tmp_path / "frozen-run")
    manifest = load_case_manifest(prepared.case_manifest_path)

    assert prepared.config_snapshot_path.is_file()
    assert prepared.environment_path.is_file()
    assert len(manifest["cases"]) == 18
    assert len(manifest["derived_records"]) == 18
    assert manifest["metadata"]["written_before_any_solver_call"] is True
    assert {item["mesh_tier"] for item in manifest["derived_records"]} == {"medium"}
    parent_split = {item["case_group_id"]: item["split"] for item in manifest["cases"]}
    assert all(
        item["split"] == parent_split[item["case_group_id"]] for item in manifest["derived_records"]
    )
    for family in ("circle", "horseshoe", "straight_wall_arch"):
        section = [item for item in manifest["cases"] if item["section_family"] == family]
        assert [item["split"] for item in section].count("train") == 4
        assert [item["split"] for item in section].count("dev") == 1
        assert [item["split"] for item in section].count("locked_test") == 1


def _kirsch_record(load: str, tier: str, error: float, *, scf: float = 0.02) -> dict:
    return {
        "load_case": load,
        "mesh_tier": tier,
        "status": "completed",
        "metrics": {
            "annulus_stress_relative_l2": error,
            "peak_hoop_relative_error": scf,
        },
        "generic_qc": {"passed": True},
    }


def test_kirsch_gate_requires_each_level_to_improve_and_strict_fine_limits() -> None:
    config, _ = load_config(DEFAULT_CONFIG)
    records = [
        _kirsch_record(load, tier, error)
        for load in ("uniaxial", "equal_biaxial", "pure_shear")
        for tier, error in zip(("coarse", "medium", "fine"), (0.07, 0.05, 0.03), strict=True)
    ]
    assert evaluate_kirsch_gate(records, config)["passed"] is True

    nonmonotonic = deepcopy(records)
    next(
        item
        for item in nonmonotonic
        if item["load_case"] == "pure_shear" and item["mesh_tier"] == "medium"
    )["metrics"]["annulus_stress_relative_l2"] = 0.071
    gate = evaluate_kirsch_gate(nonmonotonic, config)
    assert gate["passed"] is False
    assert gate["loads"]["pure_shear"]["monotonic_coarse_to_fine"] is False

    boundary = deepcopy(records)
    next(
        item
        for item in boundary
        if item["load_case"] == "equal_biaxial" and item["mesh_tier"] == "fine"
    )["metrics"]["annulus_stress_relative_l2"] = 0.08
    assert evaluate_kirsch_gate(boundary, config)["passed"] is False

    scf_boundary = deepcopy(records)
    next(
        item
        for item in scf_boundary
        if item["load_case"] == "uniaxial" and item["mesh_tier"] == "fine"
    )["metrics"]["peak_hoop_relative_error"] = 0.10
    assert evaluate_kirsch_gate(scf_boundary, config)["passed"] is False
