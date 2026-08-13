from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_load_basis_development.py"
SPEC = importlib.util.spec_from_file_location("run_load_basis_development", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _synthetic_data(parent_count: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(813)
    cases = parent_count * 4
    points = 9
    loads = np.asarray(
        [[1.0, 0.2, 0.1], [0.2, 1.0, -0.1], [0.4, 0.3, 0.8], [-0.2, 0.5, 0.6]],
        dtype=np.float64,
    )
    base = np.zeros((cases, points, 11), dtype=np.float64)
    coarse = np.empty((cases, points, 3), dtype=np.float64)
    fine = np.empty_like(coarse)
    weights = np.repeat((np.ones(points) / points)[None, :], cases, axis=0)
    geometry_ids: list[str] = []
    query_hashes: list[str] = []
    partitions: list[str] = []
    sections: list[str] = []
    for parent in range(parent_count):
        coefficients = rng.normal(size=(points, 3, 3))
        response = np.einsum("ki,poi->kpo", loads, coefficients)
        rows = slice(4 * parent, 4 * parent + 4)
        base[rows, :, 7:10] = loads[:, None, :]
        fine[rows] = response
        coarse[rows] = response + 0.02 * rng.normal(size=response.shape)
        geometry_ids.extend([f"geometry-{parent}"] * 4)
        query_hashes.extend([f"query-{parent}"] * 4)
        partitions.extend(["train_id"] * 4)
        sections.extend([["circle", "horseshoe", "straight_wall_arch"][parent % 3]] * 4)
    return {
        "base_features": base,
        "coarse_stress": coarse,
        "fine_stress": fine,
        "metric_weights": weights,
        "case_group_ids": np.asarray([f"case-{index}" for index in range(cases)]),
        "geometry_group_ids": np.asarray(geometry_ids),
        "query_hashes": np.asarray(query_hashes),
        "partitions": np.asarray(partitions),
        "section_families": np.asarray(sections),
    }


def test_leave_one_load_out_recovers_each_synthetic_geometry() -> None:
    result = runner.analyze_leave_one_load_out(_synthetic_data())
    assert result["evaluated_parent_geometries"] == 3
    assert result["evaluated_cases"] == 12
    assert result["overall"]["basis_relative_error"]["maximum"] < 1e-13
    assert result["overall"]["ratio_of_mean_errors"] < 1e-11
    assert set(result["by_section"]) == {"circle", "horseshoe", "straight_wall_arch"}


def test_three_load_groups_are_reported_as_ineligible_not_fitted() -> None:
    data = _synthetic_data()
    keep = np.arange(data["case_group_ids"].shape[0]) != 3
    trimmed = {name: value[keep] for name, value in data.items()}
    result = runner.analyze_leave_one_load_out(trimmed)
    assert result["evaluated_parent_geometries"] == 2
    assert result["skipped_parent_groups"] == {"load_count_3": 1}


def test_config_remains_seen_only_and_solver_free() -> None:
    config, digest = runner._load_config(ROOT / "configs" / "load_basis_development.json")
    assert len(digest) == 64
    assert config["effect_claim_allowed"] is False
    assert config["independent_validation_claim_allowed"] is False
    assert config["protocol"]["all_v03_partitions_are_seen"] is True
    assert config["protocol"]["new_solver_calls"] == 0
    assert config["protocol"]["new_locked_cases"] == 0


def test_canonical_basis_plan_is_rank_three_and_well_conditioned() -> None:
    plan = runner.analyze_canonical_basis_plan(_synthetic_data())
    expected = np.diag([1.0, 1.0, 1.0 / np.sqrt(2.0)])
    np.testing.assert_allclose(plan["basis_load_vectors_normalized"], expected, rtol=0.0, atol=0.0)
    assert plan["load_rank"] == 3
    assert plan["load_condition_number"] == pytest.approx(np.sqrt(2.0))
    assert plan["observed_tensor_frobenius_load_norm"]["count"] == 12
    assert plan["uses_seen_fine_labels_to_choose_basis"] is False
