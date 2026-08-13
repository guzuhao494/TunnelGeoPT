from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_stress_recovery_development.py"
CONFIG_PATH = ROOT / "configs" / "stress_recovery_development.json"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stress_recovery_development_runner", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_config_rebuilds_full_seen_plan_and_selects_one_case_per_cell() -> None:
    runner = _runner()
    config = runner.load_config(CONFIG_PATH)
    formal = runner._load_formal_config(config, CONFIG_PATH)
    plan = runner.build_seen_v03_plan(formal)

    first = runner.select_cases(plan, config["selection"])
    second = runner.select_cases(plan, config["selection"])

    assert len(plan.geometries) == 195
    assert len(plan.cases) == 705
    assert plan.formal_eligible is False
    assert len(first) == 15
    assert [case.case_group_id for case in first] == [case.case_group_id for case in second]
    cells = {(case.formal_partition, case.section_family) for case in first}
    assert cells == {
        (partition, section)
        for partition in runner.EXPECTED_PARTITIONS
        for section in runner.EXPECTED_SECTIONS
    }


def test_config_rejects_any_effect_claim_or_non_seen_interpretation(tmp_path: Path) -> None:
    runner = _runner()
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["source"]["all_v03_cases_declared_seen"] = False
    payload["exploratory_gates"]["effect_claim_allowed"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.DevelopmentRunError, match="seen/development-only"):
        runner.load_config(path)


def test_relative_tensor_error_uses_engineering_shear_multiplier_two() -> None:
    runner = _runner()
    reference = np.asarray([[2.0, -1.0, 0.5], [1.0, 3.0, -0.25]])
    prediction = reference.copy()
    prediction[:, 2] += np.asarray([1.0, -2.0])
    weights = np.asarray([0.25, 0.75])

    observed = runner.relative_tensor_error(prediction, reference, weights)
    numerator = 0.25 * 2.0 * 1.0**2 + 0.75 * 2.0 * 2.0**2
    denominator = np.sum(weights[:, None] * np.asarray([1.0, 1.0, 2.0]) * reference**2)

    assert observed == pytest.approx(np.sqrt(numerator / denominator))


def test_wall_offset_discrepancy_matches_direct_tensor_traction() -> None:
    runner = _runner()
    reference = np.zeros((2, 3), dtype=np.float64)
    prediction = np.asarray([[2.0, 4.0, 1.0], [3.0, -2.0, -0.5]])
    normals = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    weights = np.asarray([0.4, 0.6])

    traction, resultant = runner.wall_offset_discrepancy(prediction, reference, weights, normals)
    expected_vectors = np.asarray([[2.0, 1.0], [-0.5, -2.0]])

    assert traction == pytest.approx(np.sqrt(np.sum(weights * np.sum(expected_vectors**2, axis=1))))
    assert resultant == pytest.approx(
        np.linalg.norm(np.sum(weights[:, None] * expected_vectors, axis=0))
    )


def test_aggregate_center_is_ratio_of_means_and_remains_development_only() -> None:
    runner = _runner()
    config = runner.load_config(CONFIG_PATH)
    records = []
    case_index = 0
    for partition in runner.EXPECTED_PARTITIONS:
        for section in runner.EXPECTED_SECTIONS:
            raw = 0.02 + 0.001 * case_index
            recovered = 0.8 * raw
            qc = {
                "algebraic_residual": 1e-13,
                "energy_closure": 2e-14,
                "minimum_triangle_quality": 0.5,
                "all_query_points_located": True,
                "explicit_wall_and_farfield_tags": True,
                "element_centroids_inside_cavity": 0,
                "solver_seconds": 1.0,
            }
            records.append(
                {
                    "formal_partition": partition,
                    "section_family": section,
                    "metrics": {
                        "raw_coarse_vs_ultrafine": raw,
                        "recovered_coarse_vs_ultrafine": recovered,
                        "raw_coarse_vs_fine": 0.9 * raw,
                        "recovered_coarse_vs_fine": 0.72 * raw,
                        "fine_vs_ultrafine": 0.01,
                        "recovery_raw_ratio_ultrafine": 0.8,
                    },
                    "wall_offset": {
                        reference: {
                            "raw_coarse": {"traction": 0.10, "resultant": 0.05},
                            "recovered": {"traction": 0.08, "resultant": 0.04},
                        }
                        for reference in ("fine", "ultrafine")
                    },
                    "solver_mesh_qc": {tier: dict(qc) for tier in ("coarse", "fine", "ultrafine")},
                    "identity_qc": {"passed": True},
                    "case_seconds": 3.0,
                }
            )
            case_index += 1

    summary = runner.aggregate_records(records, config)

    assert summary["primary_against_ultrafine"]["center_ratio_recovered_over_raw"] == pytest.approx(
        0.8
    )
    assert summary["effect_claim_allowed"] is False
    assert summary["solver_mesh_qc"]["passed"] is True
    assert summary["development_routing"] == "PROMISING_FOR_NEW_UNSEEN_CONFIRMATORY_DESIGN"
