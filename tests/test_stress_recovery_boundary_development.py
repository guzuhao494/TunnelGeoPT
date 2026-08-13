from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_stress_recovery_boundary_development.py"
CONFIG_PATH = ROOT / "configs" / "stress_recovery_boundary_development.json"


def _runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "stress_recovery_boundary_development_runner", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_config_and_selection_reuse_exact_v05_seen_cases() -> None:
    runner = _runner()
    config = runner.load_config(CONFIG_PATH)
    formal = runner.BASE._load_formal_config(config, CONFIG_PATH)
    plan = runner.BASE.build_seen_v03_plan(formal)
    selected = runner.BASE.select_cases(plan, config["selection"])
    selected_ids = [case.case_group_id for case in selected]

    _, _, predecessor = runner._predecessor(config, selected_ids)

    assert len(plan.cases) == 705
    assert len(selected_ids) == 15
    assert predecessor["observed_ultrafine_wall_center_ratios"]["traction"] > 1.0
    assert predecessor["observed_ultrafine_wall_center_ratios"]["resultant"] > 1.0


def test_config_rejects_hiding_posthoc_redesign_provenance(tmp_path: Path) -> None:
    runner = _runner()
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["redesign_provenance"]["developed_after_observing_predecessor_results"] = False
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(runner.BoundaryDevelopmentError, match="provenance"):
        runner.load_config(path)


def test_boundary_recovery_uses_unconstrained_offwall_and_preserves_wall_traction() -> None:
    runner = _runner()
    rng = np.random.default_rng(521)
    raw = rng.normal(size=(7, 3))
    recovered = rng.normal(size=(7, 3))
    wall_mask = np.asarray([False, True, False, True, True, False, False])
    angles = np.asarray([0.1, 1.2, -0.7])
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    candidate = runner.apply_boundary_preserving_recovery(
        raw, recovered, wall_mask, normals, normal_tolerance=1e-8
    )
    increments = runner._traction_increment_norms(candidate[wall_mask], raw[wall_mask], normals)

    np.testing.assert_allclose(candidate[~wall_mask], recovered[~wall_mask], rtol=0.0, atol=0.0)
    assert float(np.max(increments)) <= 2e-15


def test_aggregate_ready_route_is_only_development_routing() -> None:
    runner = _runner()
    config = runner.load_config(CONFIG_PATH)
    records = []
    for partition in runner.EXPECTED_PARTITIONS:
        for section in runner.EXPECTED_SECTIONS:
            qc = {
                "algebraic_residual": 1e-13,
                "energy_closure": 2e-14,
                "minimum_triangle_quality": 0.6,
                "all_query_points_located": True,
                "explicit_wall_and_farfield_tags": True,
                "element_centroids_inside_cavity": 0,
                "solver_seconds": 1.0,
            }
            wall = {}
            for reference in ("fine", "ultrafine"):
                wall[reference] = {
                    "raw_coarse": {
                        "traction": 0.10,
                        "resultant": 0.05,
                        "full_stress_relative_l2": 0.20,
                    },
                    "unconstrained_recovery": {
                        "traction": 0.12,
                        "resultant": 0.08,
                        "full_stress_relative_l2": 0.11,
                    },
                    "boundary_preserving": {
                        "traction": 0.10,
                        "resultant": 0.05,
                        "full_stress_relative_l2": 0.12,
                    },
                }
            records.append(
                {
                    "formal_partition": partition,
                    "section_family": section,
                    "nearfield": {
                        "raw_coarse_vs_ultrafine": 0.03,
                        "boundary_preserving_vs_ultrafine": 0.015,
                        "boundary_raw_ratio_ultrafine": 0.5,
                        "raw_coarse_vs_fine": 0.025,
                        "boundary_preserving_vs_fine": 0.0125,
                    },
                    "wall_offset": wall,
                    "projection_contract": {
                        "maximum_traction_increment_norm": 2e-16,
                        "nearfield_metric_identity_error_vs_unconstrained": 0.0,
                    },
                    "solver_mesh_qc": {tier: dict(qc) for tier in ("coarse", "fine", "ultrafine")},
                    "identity_qc": {"passed": True},
                    "case_seconds": 3.0,
                }
            )

    summary = runner.aggregate_records(records, config)

    assert summary["development_routing"] == "READY_FOR_NEW_CONFIRMATORY_PREREGISTRATION"
    assert summary["effect_claim_allowed"] is False
    assert summary["confirmatory_status"] == "not_confirmatory_post_hoc_development_redesign"
