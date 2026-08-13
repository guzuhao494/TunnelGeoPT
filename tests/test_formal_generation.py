from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import tunnelgeopt.formal_generation as formal_generation_module
from tunnelgeopt.formal_generation import (
    FORMAL_PARTITIONS,
    LOCKED_PARTITIONS,
    FormalGenerationError,
    FormalGenerationOverrides,
    FrozenIdentityExclusions,
    _failed_qc,
    _load,
    _train_dev_candidate_assignments,
    _valid_learning_rows,
    _validity_cell,
    build_formal_generation_plan,
    generate_formal_dataset,
    training_data_paths,
    trusted_locked_label_path,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads((ROOT / "configs" / "multifidelity_formal.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny_override() -> FormalGenerationOverrides:
    return FormalGenerationOverrides(
        partitions=("train_id", "locked_iid"),
        section_families=("circle", "horseshoe", "straight_wall_arch"),
        parents_per_section={"train_id": 1, "locked_iid": 1},
        loads_per_parent={"train_id": 1, "locked_iid": 1},
        boundary_points=32,
        query_region_counts=(16, 8, 8),
        audit_fraction=1.0,
        audit_minimum_per_partition_section=1,
    )


@pytest.fixture(scope="module")
def frozen_exclusions() -> FrozenIdentityExclusions:
    payload = json.loads(
        (ROOT / "configs" / "multifidelity_seen_identity_exclusions.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["counts"] == {
        "geometry_group_ids": 42,
        "boundary_float64_sha256": 43,
        "case_group_ids": 102,
        "load_group_ids": 84,
        "all_unique_identity_records": 271,
    }
    return FrozenIdentityExclusions(
        geometry_group_ids=frozenset(payload["geometry_group_ids"]),
        boundary_float64_sha256=frozenset(payload["boundary_float64_sha256"]),
        case_group_ids=frozenset(payload["case_group_ids"]),
        load_group_ids=frozenset(payload["load_group_ids"]),
        source_artifact_sha256=payload["source_artifact_sha256"],
        source_record_count=payload["source_record_count"],
    )


def test_formal_plan_has_exact_counts_balance_ranges_and_unique_identities(
    config: dict, frozen_exclusions: FrozenIdentityExclusions
) -> None:
    plan = build_formal_generation_plan(config, forbidden_identities=frozen_exclusions)
    assert plan.formal_eligible is True
    assert len(plan.geometries) == 195
    assert len(plan.cases) == 705
    assert len(plan.audit_case_ids) == 144
    assert plan.identity_report["section_exact_balance"] is True
    assert all(plan.identity_report["cross_partition_zero_intersection"].values())
    for left_index, left in enumerate(FORMAL_PARTITIONS):
        left_loads = {case.load_group_id for case in plan.cases if case.formal_partition == left}
        for right in FORMAL_PARTITIONS[left_index + 1 :]:
            right_loads = {
                case.load_group_id for case in plan.cases if case.formal_partition == right
            }
            assert left_loads.isdisjoint(right_loads)
    assert plan.identity_report["legacy_exclusion_artifact"] == {
        "sha256": frozen_exclusions.source_artifact_sha256,
        "source_record_count": 271,
        "identity_counts": {
            "geometry_group_id": 42,
            "boundary_float64_sha256": 43,
            "case_group_id": 102,
            "load_group_id": 84,
        },
    }
    assert all(plan.identity_report["legacy_zero_intersection"].values())

    id_partitions = {"train_id", "dev_id", "locked_iid", "locked_load_ood"}
    for geometry in plan.geometries:
        assert geometry.spec.roughness_amplitude > 0.0
        positions = geometry.normalized_parameter_positions
        if geometry.formal_partition in id_partitions:
            assert all(0.15 <= value <= 0.85 for value in positions.values())
        else:
            assert geometry.ood_parameter in positions
            value = positions[str(geometry.ood_parameter)]
            assert value <= 0.1 or value >= 0.9
    for case in plan.cases:
        assert case.sigma1 >= 0.3 and case.sigma1 <= 0.8
        if case.load_subtype == "id":
            assert 0.45 <= case.sigma3_over_sigma1 <= 0.85
            assert -45.0 <= case.principal_angle_deg <= 45.0
        elif case.load_subtype == "low_lateral_ratio":
            assert 0.25 <= case.sigma3_over_sigma1 <= 0.35
            assert -45.0 <= case.principal_angle_deg <= 45.0
        elif case.load_subtype == "large_rotation":
            assert abs(case.principal_angle_deg) >= 60.0
        else:
            assert case.load_subtype == "joint_low_lateral_large_rotation"
            assert 0.25 <= case.sigma3_over_sigma1 <= 0.35
            assert abs(case.principal_angle_deg) >= 60.0


def test_small_override_is_deterministic_nonformal_and_checks_forbidden_identity(
    config: dict, tiny_override: FormalGenerationOverrides
) -> None:
    left = build_formal_generation_plan(config, tiny_override)
    right = build_formal_generation_plan(config, tiny_override)
    assert left.formal_eligible is False
    assert [case.case_group_id for case in left.cases] == [
        case.case_group_id for case in right.cases
    ]
    assert len(left.geometries) == 6
    assert len(left.cases) == 6
    assert len(left.audit_case_ids) == 6
    with pytest.raises(FormalGenerationError, match="forbidden geometry_group_id"):
        build_formal_generation_plan(
            config,
            tiny_override,
            FrozenIdentityExclusions(
                geometry_group_ids=frozenset({left.geometries[0].geometry_group_id})
            ),
        )


def test_full_formal_plan_rejects_empty_legacy_exclusions(config: dict) -> None:
    with pytest.raises(FormalGenerationError, match="legacy identity exclusion artifact"):
        build_formal_generation_plan(config)


def test_train_dev_shared_pool_is_invariant_to_salted_assignment(config: dict) -> None:
    left = _train_dev_candidate_assignments(
        config,
        section="circle",
        train_count=24,
        dev_count=6,
        boundary_points=32,
        split_salt="left-assignment",
    )
    right = _train_dev_candidate_assignments(
        config,
        section="circle",
        train_count=24,
        dev_count=6,
        boundary_points=32,
        split_salt="right-assignment",
    )
    assert len(left) == len(right) == 30
    assert {item[3] for item in left} == {item[3] for item in right}
    assert sum(item[0] == "train_id" for item in left) == 24
    assert sum(item[0] == "dev_id" for item in left) == 6
    assert {item[3] for item in left if item[0] == "train_id"} != {
        item[3] for item in right if item[0] == "train_id"
    }

    # The formal partition remains part of the load stream, guaranteeing load
    # identity separation even though train/dev share geometry candidates.
    left_load = _load(
        config,
        partition="train_id",
        section="circle",
        parent_index=7,
        load_index=2,
        seed=config["dataset"]["generator_seeds"]["train_dev"],
    )
    right_load = _load(
        config,
        partition="dev_id",
        section="circle",
        parent_index=7,
        load_index=2,
        seed=config["dataset"]["generator_seeds"]["train_dev"],
    )
    assert left_load[0] == right_load[0] == "id"
    assert not np.array_equal(left_load[4], right_load[4])


def test_trainer_path_api_never_returns_locked_path(tmp_path: Path) -> None:
    paths = training_data_paths(tmp_path)
    rendered = repr(paths)
    assert ".sealed_generator_store" not in rendered
    assert set(paths.__dataclass_fields__) == {
        "public_inputs_path",
        "train_dev_fine_labels_path",
        "dataset_manifest_path",
    }
    for partition in LOCKED_PARTITIONS:
        sealed = trusted_locked_label_path(tmp_path, partition)
        assert sealed.parent.name == ".sealed_generator_store"
        assert sealed.name == f"{partition}.npz"
    with pytest.raises(FormalGenerationError, match="not a locked"):
        trusted_locked_label_path(tmp_path, "train_id")


def test_tiny_real_fem_generation_seals_labels_resumes_and_preserves_qc(
    config: dict, tiny_override: FormalGenerationOverrides, tmp_path: Path
) -> None:
    progress: list[dict] = []
    first = generate_formal_dataset(
        config,
        tmp_path,
        overrides=tiny_override,
        progress_callback=lambda event: progress.append(dict(event)),
    )
    assert first.solved_cases == 6
    assert first.resumed_cases == 0
    assert len(progress) == 6
    assert first.audit_summary["case_values_exposed_before_checkpoint_freeze"] is False

    with np.load(first.public_inputs_path, allow_pickle=False) as public:
        assert public["base_features"].shape == (6, 32, 11)
        assert public["coarse_stress"].shape == (6, 32, 3)
        assert np.allclose(public["training_weights"].sum(axis=1), 1.0)
        assert np.allclose(public["metric_weights"].sum(axis=1), 1.0)
        assert np.allclose(public["arc_weights"].sum(axis=1), 1.0)
        assert set(public["partitions"].tolist()) == {"train_id", "locked_iid"}
        assert np.isfinite(public["wall_rock_outward_normals_yz"]).all()
    with np.load(first.train_dev_labels_path, allow_pickle=False) as labels:
        assert labels["fine_stress"].shape == (3, 32, 3)
        assert labels["audit_case_group_ids"].size == 3
        assert set(labels["audit_partitions"].tolist()) == {"train_id"}
    with np.load(trusted_locked_label_path(tmp_path, "locked_iid"), allow_pickle=False) as sealed:
        assert sealed["fine_stress"].shape == (3, 32, 3)
        assert sealed["audit_case_group_ids"].size == 3
        assert set(sealed["audit_partitions"].tolist()) == {"locked_iid"}

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["files"]) == {
        "public_inputs_and_coarse_fields.npz",
        "train_dev_fine_labels.npz",
    }
    assert all("sealed" not in key for key in manifest["files"])
    assert len(manifest["solver_mesh_qc"]["records"]) == 6
    assert manifest["solver_mesh_qc"]["no_silent_case_replacement"] is True
    assert manifest["fine_ultrafine_selection"]["selection_unit"] == "case_group_id"
    assert manifest["fine_ultrafine_selection"]["selected_before_any_ultrafine_label"] is True
    assert "relative_errors" not in json.dumps(manifest["fine_ultrafine_selection"])
    assert manifest["resource_usage"]["runtime_seconds"] > 0.0
    assert manifest["resource_usage"]["peak_memory_bytes"] > 0

    for store_id, digest in manifest["opaque_sealed_stores"].items():
        assert len(store_id) == len(digest) == 64
        assert set(store_id + digest) <= set("0123456789abcdef")
    expected_store_id = hashlib.sha256(
        json.dumps(
            {"partition": "locked_iid", "run_id": config["run_id"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert (
        manifest["opaque_sealed_stores"][expected_store_id]
        == hashlib.sha256(
            trusted_locked_label_path(tmp_path, "locked_iid").read_bytes()
        ).hexdigest()
    )

    second = generate_formal_dataset(config, tmp_path, overrides=tiny_override, resume=True)
    assert second.solved_cases == 0
    assert second.resumed_cases == 6
    assert (
        second.public_file_hashes[first.public_inputs_path.name]
        == first.public_file_hashes[first.public_inputs_path.name]
    )


def test_partition_names_are_frozen() -> None:
    assert FORMAL_PARTITIONS == (
        "train_id",
        "dev_id",
        "locked_iid",
        "locked_geometry_ood",
        "locked_load_ood",
        "locked_joint_ood",
    )


def test_nineteen_of_twenty_valid_passes_without_replacement() -> None:
    cell = _validity_cell([True] * 19 + [False], 0.95)
    assert cell == {
        "planned_cases": 20,
        "attempted_cases": 20,
        "valid_cases": 19,
        "valid_fraction": 0.95,
        "passed": True,
        "replacement_count": 0,
    }
    failed = _failed_qc(RuntimeError("deterministic solver failure"))
    assert failed["passed"] is False
    assert failed["failure"]["label_written"] is False
    assert failed["failure"]["replacement_attempted"] is False
    records = [{"qc": {"passed": index < 19}, "label": index} for index in range(20)]
    assert [row["label"] for row in _valid_learning_rows(records)] == list(range(19))


def test_eighteen_of_twenty_valid_fails_closed_without_replacement() -> None:
    cell = _validity_cell([True] * 18 + [False, False], 0.95)
    assert cell["planned_cases"] == cell["attempted_cases"] == 20
    assert cell["valid_cases"] == 18
    assert cell["valid_fraction"] == pytest.approx(0.9)
    assert cell["passed"] is False
    assert cell["replacement_count"] == 0
    records = [{"qc": {"passed": index < 18}, "label": index} for index in range(20)]
    assert len(_valid_learning_rows(records)) == 18


def _passing_qc() -> dict[str, Any]:
    fidelity = {
        "nonfinite_fraction": 0.0,
        "free_dof_algebraic_residual": 0.0,
        "clapeyron_relative_energy_error": 0.0,
        "min_triangle_signed_area_over_radius_squared": 1.0,
        "min_triangle_quality": 1.0,
        "all_query_points_located": True,
        "explicit_wall_and_farfield_tags": True,
        "no_element_centroid_inside_cavity": True,
        "same_boundary_hash_and_outer_bounds": True,
        "passed": True,
    }
    return {
        "fidelities": {"coarse": dict(fidelity), "fine": dict(fidelity)},
        "identity": {
            "same_boundary": True,
            "same_outer_bounds": True,
            "same_query_hash": True,
        },
        "passed": True,
    }


def _install_twenty_case_fake_solver(
    monkeypatch: pytest.MonkeyPatch,
    plan: Any,
    *,
    invalid_count: int,
) -> None:
    non_audit = [case for case in plan.cases if case.case_group_id not in plan.audit_case_ids]
    failed_ids = {case.case_group_id for case in non_audit[:invalid_count]}
    ordered_cases = iter(plan.cases)

    def fake_solve(geometry: Any, grid: Any, **kwargs: Any) -> Any:
        case = next(ordered_cases)
        if case.case_group_id in failed_ids:
            raise RuntimeError("injected deterministic solver failure")
        point_count = int(grid.point_count)
        return SimpleNamespace(
            case_group_id=case.case_group_id,
            model_features=np.zeros((point_count, 11), dtype=np.float32),
            coarse_stress_normalized=np.zeros((point_count, 3), dtype=np.float32),
            _fine_stress_normalized=np.zeros((point_count, 3), dtype=np.float32),
            stress_scale=1.0,
        )

    def fake_audit(*, case: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "case_group_id": case.case_group_id,
            "formal_partition": case.formal_partition,
            "section_family": case.section_family,
            "error": 0.0,
            "fine_repeat_consistent": True,
            "ultrafine_qc_passed": True,
        }

    monkeypatch.setattr(formal_generation_module, "solve_multifidelity_case", fake_solve)
    monkeypatch.setattr(
        formal_generation_module,
        "_independent_mesh_geometry_qc",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        formal_generation_module,
        "_qc_record",
        lambda *args, **kwargs: _passing_qc(),
    )
    monkeypatch.setattr(formal_generation_module, "_execute_ultrafine_audit", fake_audit)


def _twenty_case_override() -> FormalGenerationOverrides:
    return FormalGenerationOverrides(
        partitions=("train_id",),
        section_families=("circle",),
        parents_per_section={"train_id": 1},
        loads_per_parent={"train_id": 20},
        boundary_points=32,
        query_region_counts=(16, 8, 8),
        audit_fraction=0.2,
        audit_minimum_per_partition_section=3,
    )


def test_one_of_twenty_failed_cases_writes_only_nineteen_learning_rows(
    config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _twenty_case_override()
    plan = build_formal_generation_plan(config, override)
    _install_twenty_case_fake_solver(monkeypatch, plan, invalid_count=1)
    result = generate_formal_dataset(config, tmp_path, overrides=override, resume=False)
    with np.load(result.public_inputs_path, allow_pickle=False) as public:
        assert public["case_group_ids"].size == 19
    with np.load(result.train_dev_labels_path, allow_pickle=False) as labels:
        assert labels["fine_stress"].shape[0] == 19
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["counts"]["valid_cases"] == 19
    assert manifest["counts"]["invalid_cases"] == 1
    assert len(manifest["solver_mesh_qc"]["records"]) == 20
    assert manifest["solver_mesh_qc"]["partition_section_summary"]["passed"] is True


def test_two_of_twenty_failed_cases_write_eighteen_rows_then_abstain(
    config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _twenty_case_override()
    plan = build_formal_generation_plan(config, override)
    _install_twenty_case_fake_solver(monkeypatch, plan, invalid_count=2)
    with pytest.raises(FormalGenerationError, match="complete evidence written"):
        generate_formal_dataset(config, tmp_path, overrides=override, resume=False)
    with np.load(tmp_path / "public_inputs_and_coarse_fields.npz", allow_pickle=False) as public:
        assert public["case_group_ids"].size == 18
    with np.load(tmp_path / "train_dev_fine_labels.npz", allow_pickle=False) as labels:
        assert labels["fine_stress"].shape[0] == 18
    manifest = json.loads((tmp_path / "formal_dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["generation_status"] == "ABSTAIN"
    assert manifest["counts"]["valid_cases"] == 18
    assert manifest["counts"]["invalid_cases"] == 2
    assert len(manifest["solver_mesh_qc"]["records"]) == 20
    assert manifest["solver_mesh_qc"]["partition_section_summary"]["passed"] is False
    assert all(
        record.get("failure", {}).get("replacement_attempted") is False
        for record in manifest["solver_mesh_qc"]["records"]
        if not record["valid"]
    )


def test_zero_valid_cases_still_write_complete_abstain_evidence(
    config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = FormalGenerationOverrides(
        partitions=("train_id",),
        section_families=("circle",),
        parents_per_section={"train_id": 1},
        loads_per_parent={"train_id": 1},
        boundary_points=32,
        query_region_counts=(16, 8, 8),
        audit_fraction=1.0,
        audit_minimum_per_partition_section=1,
    )

    def fail_solve(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected deterministic solver failure")

    monkeypatch.setattr(formal_generation_module, "solve_multifidelity_case", fail_solve)
    with pytest.raises(FormalGenerationError, match="zero valid cases"):
        generate_formal_dataset(config, tmp_path, overrides=override, resume=False)
    manifest_path = tmp_path / "formal_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation_status"] == "ABSTAIN"
    assert manifest["counts"] == {
        "parent_geometries": 1,
        "planned_cases": 1,
        "valid_cases": 0,
        "invalid_cases": 1,
    }
    records = manifest["solver_mesh_qc"]["records"]
    assert len(records) == 1 and records[0]["valid"] is False
    assert records[0]["failure"]["replacement_attempted"] is False
    assert manifest["files"] == manifest["artifact_hashes"] == {}
    assert not (tmp_path / "public_inputs_and_coarse_fields.npz").exists()
    assert not (tmp_path / ".sealed_generator_store").exists()
