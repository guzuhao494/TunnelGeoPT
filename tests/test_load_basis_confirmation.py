from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_load_basis_confirmation.py"
CONFIG_PATH = ROOT / "configs" / "load_basis_confirmation.json"


def _load_runner():
    specification = importlib.util.spec_from_file_location(
        "tunnelgeopt_load_basis_confirmation_runner",
        RUNNER_PATH,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_frozen_plan_is_solver_free_new_and_exactly_24_solves(monkeypatch) -> None:
    runner = _load_runner()

    def forbid_solver(*args, **kwargs):
        raise AssertionError("validate-plan must never call the FEM solver")

    monkeypatch.setattr(runner, "solve_plane_strain_excavation", forbid_solver)
    plan = runner.validate_plan(CONFIG_PATH)
    assert plan["status"] == "validated_not_executed"
    assert plan["plan_sha256"] == "cf91a557bae100545ad84bec121cc6bbcdcc09e1ae6a7fd3da98c7d9cf463ef5"
    assert plan["geometry_count"] == 3
    assert plan["heldout_load_count"] == 5
    assert plan["direct_fem_solve_count"] == 24
    assert plan["basis"]["load_vectors"] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0 / math.sqrt(2.0)],
    ]
    assert plan["basis"]["rank"] == 3
    assert plan["basis"]["condition_number"] == pytest.approx(math.sqrt(2.0))


def test_identity_audit_is_unique_and_zero_intersection() -> None:
    runner = _load_runner()
    plan = runner.validate_plan(CONFIG_PATH)
    audit = plan["identity_audit"]
    assert audit["passed"] is True
    assert audit["new_unique_identity_counts"] == {
        "boundary_float64_sha256": 3,
        "case_group_id": 24,
        "geometry_group_id": 3,
        "load_group_id": 8,
        "query_hash": 3,
    }
    assert all(audit["uniqueness_checks"].values())
    assert all(audit["zero_intersection_checks"].values())
    assert all(not values for values in audit["intersections"].values())
    # The formal v0.3 identity store contains all 195 geometry/query parents.
    source_counts = audit["excluded_identity_sources"]["v03_public_identity_store"][
        "unique_identity_counts"
    ]
    assert source_counts["geometry_group_id"] == 195
    assert source_counts["query_hash"] == 195


def test_three_basis_fields_reconstruct_five_independent_linear_fields() -> None:
    runner = _load_runner()
    random = np.random.default_rng(20260813)
    basis_loads = np.diag([1.0, 1.0, 1.0 / math.sqrt(2.0)])
    heldout_loads = np.asarray(
        [
            [-0.62, -0.37, -0.11],
            [-0.48, -0.71, 0.13],
            [-0.83, -0.29, -0.07],
            [-0.34, -0.91, 0.16],
            [-0.74, -0.55, 0.0],
        ]
    )
    coefficients = random.normal(size=(37, 3, 3))
    basis_responses = np.einsum("ki,poi->kpo", basis_loads, coefficients)
    heldout_responses = np.einsum("ki,poi->kpo", heldout_loads, coefficients)
    result = runner.evaluate_response_arrays(
        basis_loads,
        basis_responses,
        heldout_loads,
        heldout_responses,
    )
    assert result["basis_rank"] == 3
    assert result["basis_condition_number"] == pytest.approx(math.sqrt(2.0))
    assert result["heldout_median_relative_l2"] <= 1e-14
    assert result["heldout_maximum_relative_l2"] <= 1e-14


@pytest.mark.parametrize(
    ("shape", "tensor_frobenius"),
    [((41, 2), False), ((73, 3), True), ((41,), False)],
)
def test_auxiliary_linear_responses_reconstruct_heldout_fields(
    shape: tuple[int, ...], tensor_frobenius: bool
) -> None:
    runner = _load_runner()
    random = np.random.default_rng(8021 + len(shape))
    basis_loads = np.diag([1.0, 1.0, 1.0 / math.sqrt(2.0)])
    heldout_loads = np.asarray(
        [
            [-0.62, -0.37, -0.11],
            [-0.48, -0.71, 0.13],
            [-0.83, -0.29, -0.07],
            [-0.34, -0.91, 0.16],
            [-0.74, -0.55, 0.0],
        ]
    )
    coefficients = random.normal(size=(3, *shape))
    basis_response = np.einsum("ki,i...->k...", basis_loads, coefficients)
    heldout_response = np.einsum("ki,i...->k...", heldout_loads, coefficients)
    result = runner.evaluate_linear_response(
        basis_loads,
        basis_response,
        heldout_loads,
        heldout_response,
        tensor_frobenius=tensor_frobenius,
    )
    assert result["basis_rank"] == 3
    assert result["heldout_maximum_relative_l2"] <= 1e-14


def test_tensor_relative_l2_counts_engineering_shear_twice() -> None:
    runner = _load_runner()
    reference = np.asarray([[1.0, 0.0, 1.0]])
    prediction = np.asarray([[1.0, 0.0, 2.0]])
    assert runner.tensor_frobenius_relative_l2(prediction, reference) == pytest.approx(
        math.sqrt(2.0 / 3.0)
    )


def test_config_rejects_basis_or_claim_scope_drift(tmp_path: Path) -> None:
    runner = _load_runner()
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["basis_loads_tension_positive"][2][2] = -1.0 / math.sqrt(2.0)
    invalid_basis = tmp_path / "invalid_basis.json"
    invalid_basis.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.ConfirmationError, match="condition number|unit|frozen load"):
        # Loading checks unit/rank/condition; plan validation also authenticates frozen IDs.
        config = runner.load_config(invalid_basis)
        runner.build_confirmation_plan(config)

    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["claim_scope"]["confirmed"].append("geometry_generalization")
    invalid_claim = tmp_path / "invalid_claim.json"
    invalid_claim.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(runner.ConfirmationError, match="claim scope"):
        runner.load_config(invalid_claim)


def test_real_run_requires_explicit_acknowledgement_and_writes_nothing(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "must-not-exist"
    with pytest.raises(runner.ConfirmationError, match="acknowledgement"):
        runner.run_confirmation(
            CONFIG_PATH,
            output,
            expected_head="a" * 64,
            acknowledgement="NO",
        )
    assert not output.exists()


def test_post_solve_classification_is_validity_first_and_three_way() -> None:
    runner = _load_runner()
    checks = {
        "exact_direct_fem_solve_count_24": True,
        "exact_heldout_comparison_count_15": True,
        "basis_rank_three_every_geometry": True,
        "basis_condition_sqrt_two_every_geometry": True,
        "solver_algebraic_residual": True,
        "solver_energy_closure": True,
        "mesh_query_boundary_fixed_per_geometry": True,
        "all_query_points_located": True,
        "explicit_boundary_tags": True,
        "no_element_centroid_inside_cavity": True,
        "new_identity_zero_intersection": True,
        "median_relative_l2": True,
        "maximum_relative_l2": True,
    }
    classification, _, _ = runner.classify_gate_checks(checks)
    assert classification == runner.CONFIRMED_CLASSIFICATION
    checks["maximum_relative_l2"] = False
    classification, _, _ = runner.classify_gate_checks(checks)
    assert classification == runner.STOP_CLASSIFICATION
    checks["solver_energy_closure"] = False
    classification, _, _ = runner.classify_gate_checks(checks)
    assert classification == runner.INVALID_CLASSIFICATION


def test_real_preflight_requires_clean_pushed_head_and_hashes(monkeypatch) -> None:
    runner = _load_runner()
    config = runner.load_config(CONFIG_PATH)
    config["implementation_preflight"]["required_path_sha256"][
        "scripts/run_load_basis_confirmation.py"
    ] = runner._file_sha256(RUNNER_PATH)
    expected_head = "a" * 40
    output = ROOT / "artifacts" / "confirmation" / "unit-test-output-must-be-absent"
    assert not output.exists()

    def fake_git(arguments):
        key = tuple(arguments)
        if key in {
            ("diff", "--name-only", "--"),
            ("diff", "--cached", "--name-only", "--"),
        }:
            return ""
        if key in {("rev-parse", "HEAD"), ("rev-parse", "@{upstream}")}:
            return expected_head
        if key == ("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"):
            return "origin/codex/v0.4-structured-residual"
        if key[:2] == ("ls-files", "--error-unmatch"):
            return str(key[-1])
        raise AssertionError(f"unexpected git call: {arguments}")

    monkeypatch.setattr(runner, "_git", fake_git)
    preflight = runner.real_execution_preflight(
        config,
        CONFIG_PATH,
        output,
        expected_head,
    )
    assert preflight["tracked_clean"] is True
    assert preflight["head_equals_upstream"] is True
    assert preflight["expected_head_matched"] is True
    assert preflight["output_absent_before_execution"] is True
    assert set(preflight["critical_source_sha256"]) == {
        "scripts/run_load_basis_confirmation.py",
        "src/tunnelgeopt/load_basis.py",
    }
    assert not output.exists()


def test_canonical_json_is_sorted_compact_and_rejects_nan() -> None:
    runner = _load_runner()
    assert runner._canonical_bytes({"z": 2, "a": 1}) == b'{"a":1,"z":2}'
    with pytest.raises(ValueError):
        runner._canonical_bytes({"bad": float("nan")})


def test_nonfinite_qc_writes_serializable_invalid_artifact(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "reserved"
    monkeypatch.setattr(runner, "_relative_repository_path", lambda path: Path(path).name)
    runner._reserve_output_dir(output)
    result = runner._write_confirmation_artifact(
        output,
        {
            "classification": runner.CONFIRMED_CLASSIFICATION,
            "qc": {"algebraic_residual": float("nan"), "energy_closure": float("inf")},
            "all_gates_passed": True,
        },
    )
    assert result["classification"] == runner.INVALID_CLASSIFICATION
    payload = json.loads((output / "confirmation.json").read_text(encoding="utf-8"))
    assert payload["classification"] == runner.INVALID_CLASSIFICATION
    assert payload["qc"] == {"algebraic_residual": None, "energy_closure": None}
    assert payload["serialization_validity"]["passed"] is False
    assert {issue["observed"] for issue in payload["serialization_validity"]["issues"]} == {
        "nan",
        "positive_infinity",
    }


def test_second_solve_failure_preserves_attempted_and_completed_records(monkeypatch) -> None:
    runner = _load_runner()
    plan = runner.build_confirmation_plan(runner.load_config(CONFIG_PATH))
    frozen = plan.geometries[0]
    node_count = 5
    element_count = 2
    mesh = SimpleNamespace(
        nodes=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]),
        elements=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
        boundary_facets={
            "wall": np.asarray([[0, 1]], dtype=np.int64),
            "farfield": np.asarray([[1, 3]], dtype=np.int64),
        },
        outer_bounds=(-2.0, 2.0, -2.0, 2.0),
        metadata={
            "minimum_element_area": 0.5,
            "minimum_triangle_quality": 0.8,
            "wall_facet_count": 1,
            "farfield_facet_count": 1,
        },
    )
    monkeypatch.setattr(runner, "generate_tunnel_mesh", lambda *args, **kwargs: mesh)
    monkeypatch.setattr(
        runner,
        "locate_elements",
        lambda *args, **kwargs: np.zeros(frozen.grid.point_count, dtype=np.int64),
    )
    monkeypatch.setattr(
        runner,
        "points_inside_polygon",
        lambda points, boundary: np.zeros(points.shape[0], dtype=bool),
    )
    calls = 0

    def fake_solve(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second solve failure")
        return SimpleNamespace(
            nodes=mesh.nodes,
            elements=mesh.elements,
            boundary_facets=mesh.boundary_facets,
            displacement=np.full((node_count, 2), 0.25),
            delta_stress=np.full((element_count, 3), 0.5),
            total_stress=np.full((element_count, 3), 1.0),
            sigma_xx=np.full(element_count, 0.75),
            algebraic_residual=1e-13,
            energy_closure=1e-14,
        )

    monkeypatch.setattr(runner, "solve_plane_strain_excavation", fake_solve)
    execution_records: list[dict] = []
    all_loads = np.vstack([plan.basis_loads, plan.heldout_loads])
    with pytest.raises(RuntimeError, match="second solve"):
        runner._solve_geometry(
            frozen, runner.load_config(CONFIG_PATH), all_loads, execution_records
        )
    assert calls == 2
    assert len(execution_records) == 2
    assert execution_records[0]["status"] == "completed"
    assert execution_records[0]["validated_complete"] is True
    assert execution_records[1]["status"] == "failed"
    assert execution_records[1]["attempted"] is True
    assert execution_records[1]["validated_complete"] is False
    summary = runner._execution_summary(execution_records)
    assert summary == {
        "planned_direct_fem_solve_count": 24,
        "attempted_direct_fem_solve_count": 2,
        "solver_returned_count": 1,
        "completed_validated_direct_fem_solve_count": 1,
        "failed_direct_fem_solve_count": 1,
        "not_attempted_direct_fem_solve_count": 22,
    }


def test_output_reservation_is_atomic_and_detects_race(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "confirmation"
    runner._reserve_output_dir(output)
    assert output.is_dir()
    with pytest.raises(runner.ConfirmationError, match="reservation lost"):
        runner._reserve_output_dir(output)
    assert list(output.iterdir()) == []


def test_run_reserves_output_before_first_geometry_work(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "confirmation"
    plan = runner.build_confirmation_plan(runner.load_config(CONFIG_PATH))
    monkeypatch.setattr(
        runner,
        "real_execution_preflight",
        lambda *args, **kwargs: {"output_absent_before_execution": True},
    )
    monkeypatch.setattr(runner, "build_confirmation_plan", lambda config: plan)
    monkeypatch.setattr(runner, "_relative_repository_path", lambda path: Path(path).name)

    def fail_after_reservation(frozen, config, all_loads, records):
        assert output.is_dir()
        raise RuntimeError("injected first geometry failure")

    monkeypatch.setattr(runner, "_solve_geometry", fail_after_reservation)
    result = runner.run_confirmation(
        CONFIG_PATH,
        output,
        expected_head="a" * 40,
        acknowledgement=runner.EXECUTION_ACKNOWLEDGEMENT,
    )
    assert result["classification"] == runner.INVALID_CLASSIFICATION
    payload = json.loads((output / "confirmation.json").read_text(encoding="utf-8"))
    assert (
        payload["execution_preflight"]["output_directory_reserved_before_first_mesh_or_solve"]
        is True
    )
    assert payload["execution_summary"]["planned_direct_fem_solve_count"] == 24
    assert payload["execution_summary"]["attempted_direct_fem_solve_count"] == 0
