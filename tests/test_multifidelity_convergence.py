from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_multifidelity_convergence.py"
SPEC = importlib.util.spec_from_file_location("run_multifidelity_convergence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _config() -> dict:
    path = ROOT / "configs" / "multifidelity_convergence_dev.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_contract_is_development_only_and_has_24_balanced_cases() -> None:
    config = runner._load_config(ROOT / "configs" / "multifidelity_convergence_dev.json")
    geometries = runner._development_geometries(config)
    assert len(geometries) == 12
    assert {item.section for item in geometries} == set(runner.SECTION_NAMES)
    for section in runner.SECTION_NAMES:
        values = [item for item in geometries if item.section == section]
        assert len(values) == 4
        assert len({item.geometry_id for item in values}) == 4
        assert all(item.spec.roughness_amplitude > 0.0 for item in values)
    assert len(geometries) * config["loads"]["per_geometry"] == 24
    assert config["data_access"]["locked_or_pseudo_test_labels_allowed"] is False


def test_development_generation_and_loads_are_deterministic() -> None:
    config = _config()
    first = runner._development_geometries(config)
    second = runner._development_geometries(config)
    assert [item.geometry_id for item in first] == [item.geometry_id for item in second]
    load_a = runner._load_tensor(config, first[0].geometry_id, 0)
    load_b = runner._load_tensor(config, first[0].geometry_id, 0)
    assert (load_a == load_b).all()
    assert (load_a == load_a.T).all()


def test_config_rejects_any_locked_or_pseudo_test_permission(tmp_path: Path) -> None:
    config = _config()
    config["data_access"]["locked_or_pseudo_test_labels_allowed"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        runner._load_config(path)


def test_config_rejects_implicit_mesh_size(tmp_path: Path) -> None:
    config = _config()
    del config["mesh"]["ultrafine"]["mesh_size_over_radius"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="freeze all three"):
        runner._load_config(path)


def test_aggregate_requires_effect_claim_to_remain_false() -> None:
    config = _config()
    records = []
    for section in runner.SECTION_NAMES:
        for _ in range(8):
            records.append(
                {
                    "section_family": section,
                    "stress_frobenius_rell2": 0.01,
                    "solver_qc": {
                        tier: {
                            "algebraic_residual": 1e-12,
                            "energy_closure": 1e-12,
                            "minimum_triangle_quality": 0.2,
                        }
                        for tier in ("fine", "ultrafine")
                    },
                    "query_qc": {
                        tier: {"all_points_located": True} for tier in ("fine", "ultrafine")
                    },
                    "identity_qc": {
                        "same_frozen_boundary": True,
                        "same_outer_bounds": True,
                        "same_query_hash": True,
                    },
                }
            )
    metrics = runner._aggregate(records, config)
    assert metrics["decision"] == "current_tiers_eligible_for_formal_freeze"
    assert metrics["checks"]["effect_claim_allowed"] is False
    records[0]["stress_frobenius_rell2"] = 0.2
    records[1]["stress_frobenius_rell2"] = 0.2
    metrics = runner._aggregate(records, config)
    assert metrics["decision"] == "do_not_start_formal_with_current_tiers"
