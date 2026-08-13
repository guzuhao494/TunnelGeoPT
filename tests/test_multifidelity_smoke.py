from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_multifidelity_smoke.py"
SPEC = importlib.util.spec_from_file_location("run_multifidelity_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _config() -> dict:
    return json.loads((ROOT / "configs" / "multifidelity_smoke.json").read_text(encoding="utf-8"))


def test_frozen_config_and_section_stratified_parent_splits() -> None:
    config = _config()
    geometries, split = runner._geometry_specs(config)
    assert len(geometries) == 30
    assert len(split.train) == 18
    assert len(split.dev) == 6
    assert len(split.locked_test) == 6
    for section in runner.SECTION_NAMES:
        section_rows = [value for value in geometries if value.section == section]
        assert sum(value.split == "train" for value in section_rows) == 6
        assert sum(value.split == "dev" for value in section_rows) == 2
        assert sum(value.split == "locked_test" for value in section_rows) == 2
        assert all(value.spec.roughness_amplitude > 0.0 for value in section_rows)
        assert len({value.geometry_id for value in section_rows}) == 10


def test_loads_are_finite_compressive_symmetric_and_case_specific() -> None:
    config = _config()
    values = [runner._load_tensor(config, "a" * 64, index) for index in range(2)]
    assert not np.array_equal(values[0], values[1])
    for stress in values:
        assert stress.shape == (2, 2)
        assert np.isfinite(stress).all()
        np.testing.assert_allclose(stress, stress.T)
        assert np.all(np.linalg.eigvalsh(stress) < 0.0)


def test_smoke_decision_does_not_treat_forbidden_claim_as_a_failed_check() -> None:
    checks = {
        "all_solver_cases": True,
        "all_query_points_located_in_both_meshes": True,
        "no_cross_split_parent": True,
        "locked_label_denied_before_authorization": True,
        "all_methods_one_checkpoint": True,
        "finite_metrics": True,
        "effect_claim_allowed": False,
    }
    assert runner._smoke_decision(checks) == "pipeline_go"
    checks["finite_metrics"] = False
    assert runner._smoke_decision(checks) == "pipeline_no_go"
    checks["finite_metrics"] = True
    checks["effect_claim_allowed"] = True
    assert runner._smoke_decision(checks) == "pipeline_no_go"


def test_public_inputs_and_sealed_labels_are_separate_files(tmp_path) -> None:
    public = tmp_path / "public_inputs.npz"
    sealed = tmp_path / "sealed_pseudo_test_labels.npz"
    runner._atomic_npz(public, base_features=np.zeros((2, 3, 11)))
    runner._atomic_npz(sealed, indices=np.asarray([1]), fine_stress=np.ones((1, 3, 3)))
    with np.load(public, allow_pickle=False) as archive:
        assert "fine_stress" not in archive.files
    with np.load(sealed, allow_pickle=False) as archive:
        assert archive.files == ["indices", "fine_stress"]


def test_config_rejects_effect_claim_permission(tmp_path) -> None:
    config = _config()
    config["smoke_success"]["effect_claim_allowed"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="may not authorize"):
        runner._load_config(path)
