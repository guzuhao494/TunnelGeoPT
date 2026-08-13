from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tunnelgeopt.transfer import (
    ALL_METHODS,
    PRETRAIN_METHODS,
    TransferContractError,
    build_analytic_dataset,
    build_finetuning_arrays,
    build_load_cases,
    build_pretraining_arrays,
    build_query_grid,
    case_metrics,
    checkpoint_identity,
    deterministic_derangement,
    dry_run_contract,
    evaluate_locked_test,
    load_checkpoint_payload,
    load_model_checkpoint,
    load_transfer_config,
    make_model,
    nested_train_indices,
    paired_stratified_bootstrap,
    resolve_device,
    save_cpu_checkpoint_atomic,
    static_geometry_case,
    stress_frobenius_relative_l2,
    stress_lift_case,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "analytic_transfer_smoke.json"


@pytest.fixture(scope="module")
def config():
    return load_transfer_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def dataset(config):
    return build_analytic_dataset(config)


def test_strict_config_and_canonical_case_split(config) -> None:
    cases_first = build_load_cases(config)
    cases_second = build_load_cases(config)
    assert cases_first == cases_second
    assert len(cases_first) == 240
    assert len({case.case_group_id for case in cases_first}) == 240
    assert {len(case.case_group_id) for case in cases_first} == {64}
    assert {
        split: sum(case.split == split for case in cases_first)
        for split in ("train", "dev", "locked_test")
    } == {"train": 168, "dev": 36, "locked_test": 36}
    assert {case.stratum for case in cases_first} == set(range(16))


def test_config_mutation_is_rejected(tmp_path, config) -> None:
    changed = json.loads(json.dumps(config))
    changed["split"]["counts"]["train"] = 169
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(TransferContractError, match="168/36/36"):
        load_transfer_config(path)


def test_query_grid_and_kirsch_labels_have_exact_case_shapes(dataset) -> None:
    grid = dataset.grid
    assert grid.x.shape == (512, 7)
    assert grid.points_yz.shape == (512, 2)
    assert grid.annulus_mask.sum() == 384
    assert grid.wall_mask.sum() == 64
    assert grid.farfield_mask.sum() == 64
    assert dataset.expected_label_shape == (240, 512, 3)
    audit = dataset.access_snapshot()
    assert audit["materialized_cases"] == {"train": 168, "dev": 36, "locked_test": 0}
    assert audit["label_case_reads"] == {"train": 0, "dev": 0, "locked_test": 0}
    assert audit["locked_test_unlocked"] is False
    assert np.max(np.abs(np.linalg.norm(grid.points_yz[grid.wall_mask], axis=1) - 1.0)) < 1e-14


def test_query_grid_is_deterministic(config) -> None:
    first = build_query_grid(config)
    second = build_query_grid(config)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.points_yz, second.points_yz)


def test_stress_lift_formula_condition_and_three_step_sticking(dataset) -> None:
    case = dataset.cases[int(dataset.indices("train")[0])]
    condition, target = stress_lift_case(case, dataset.grid)
    assert condition.shape == (512, 4)
    assert target.shape == (512, 9)
    np.testing.assert_allclose(condition[0], case.condition_vector)
    assert np.all(condition[dataset.grid.wall_mask, 3] == 0.0)
    t0 = target[:, :3][:, 1:3]
    np.testing.assert_allclose(t0, dataset.grid.vector_distance, atol=1e-7)
    distance = np.stack(
        [np.linalg.norm(target[:, step : step + 3][:, 1:3], axis=1) for step in (0, 3, 6)]
    )
    assert np.all(distance[1:] <= distance[:-1] + 1e-7)
    assert np.all(distance[:, dataset.grid.wall_mask] == 0.0)
    stuck_at_step_1 = distance[1] < 1e-7
    assert np.all(distance[2, stuck_at_step_1] < 1e-7)


def test_static_repeats_t0_and_derangement_has_no_fixed_point(dataset) -> None:
    case = dataset.cases[int(dataset.indices("train")[0])]
    condition, target = static_geometry_case(case, dataset.grid)
    assert np.count_nonzero(condition) == 0
    np.testing.assert_array_equal(target[:, :3], target[:, 3:6])
    np.testing.assert_array_equal(target[:, :3], target[:, 6:9])
    first = deterministic_derangement(168, 17)
    second = deterministic_derangement(168, 17)
    np.testing.assert_array_equal(first, second)
    assert np.all(first != np.arange(168))
    assert len(np.unique(first)) == 168


def test_pretraining_is_train_only_and_shuffled_only_changes_condition(dataset) -> None:
    train = dataset.indices("train")
    stress_x, stress_y, stress_meta = build_pretraining_arrays(
        dataset, train, "stress_lift_80", seed=17
    )
    shuffled_x, shuffled_y, shuffled_meta = build_pretraining_arrays(
        dataset, train, "shuffled_stress_lift_80", seed=17
    )
    assert stress_x.shape == (168, 512, 11)
    assert stress_y.shape == (168, 512, 9)
    assert stress_meta["splits_read"] == ["train"]
    assert shuffled_meta["splits_read"] == ["train"]
    np.testing.assert_array_equal(shuffled_y, stress_y)
    np.testing.assert_array_equal(shuffled_x[:, :, :7], stress_x[:, :, :7])
    assert not np.array_equal(shuffled_x[:, :, 7:], stress_x[:, :, 7:])
    permutation = np.asarray(shuffled_meta["derangement"])
    assert np.all(permutation != np.arange(len(permutation)))

    contaminated = np.r_[train[:-1], dataset.indices("dev")[0]]
    with pytest.raises(TransferContractError, match="non-train"):
        build_pretraining_arrays(dataset, contaminated, "stress_lift_80", seed=17)


def test_random_control_matches_case_magnitude_marginal(dataset) -> None:
    train = dataset.indices("train")[:8]
    features, _, _ = build_pretraining_arrays(dataset, train, "random_lift_80", seed=29)
    for position, index in enumerate(train):
        expected = 1.0 - dataset.cases[int(index)].sigma_ratio
        nonwall = ~dataset.grid.wall_mask
        np.testing.assert_allclose(features[position, nonwall, 10], expected, atol=1e-7)
        directions = features[position, nonwall, 8:10]
        np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)


def test_nested_case_subsets_and_feature_contract(dataset) -> None:
    train_80 = nested_train_indices(dataset, 0.8)
    train_100 = nested_train_indices(dataset, 1.0)
    assert len(train_80) == 134
    assert len(train_100) == 168
    assert set(train_80).issubset(set(train_100))
    x, y = build_finetuning_arrays(dataset, train_80[:3])
    assert x.shape == (3, 512, 11)
    assert y.shape == (3, 512, 3)


def test_case_metrics_and_paired_stratified_bootstrap(dataset) -> None:
    train = dataset.indices("train")[:36]
    target = dataset.labels_for(train, purpose="unit_metric_test")
    perfect = case_metrics(target, target, dataset.grid)
    for values in perfect.values():
        np.testing.assert_allclose(values, 0.0, atol=1e-15)
    perturbed = target * 1.1
    rel = stress_frobenius_relative_l2(perturbed, target)
    np.testing.assert_allclose(rel, 0.1, rtol=2e-6)
    interval = paired_stratified_bootstrap(
        rel,
        np.full_like(rel, 0.2),
        np.asarray([dataset.cases[int(index)].stratum for index in train]),
        replicates=200,
        confidence=0.95,
        seed=20260813,
    )
    assert interval["center_ratio"] == pytest.approx(0.5, rel=1e-6)
    assert interval["lower"] == pytest.approx(0.5, rel=1e-6)
    assert interval["upper"] == pytest.approx(0.5, rel=1e-6)


def test_locked_test_access_fails_before_authorization_and_is_counted(config) -> None:
    isolated = build_analytic_dataset(config)
    test = isolated.indices("locked_test")
    with pytest.raises(TransferContractError, match="before every checkpoint"):
        isolated.labels_for(test, purpose="forbidden_test")
    audit = isolated.access_snapshot()
    assert audit["materialized_cases"]["locked_test"] == 0
    assert audit["label_case_reads"]["locked_test"] == 0
    assert audit["denied_locked_test_accesses"] == 1
    with pytest.raises(TransferContractError, match="every expected"):
        isolated.authorize_locked_test(["only-one"], expected_checkpoint_count=18)


def test_evaluate_locked_test_regression_returns_all_finite_case_metrics(config) -> None:
    pytest.importorskip("torch")
    dataset = build_analytic_dataset(config)
    dataset.authorize_locked_test(
        [f"checkpoint-{index}" for index in range(18)], expected_checkpoint_count=18
    )
    dataset.materialize_split("locked_test", purpose="post_freeze_unit_evaluation")
    model = make_model(config, output_width=3, seed=17, device="cpu")
    result = evaluate_locked_test(model, dataset, config, device="cpu")
    assert len(result["case_group_ids"]) == 36
    assert len(result["strata"]) == 36
    expected = {
        "stress_frobenius_relative_l2",
        "wall_traction_relative_l2",
        "farfield_stress_relative_l2",
        "peak_wall_hoop_stress_relative_error",
        "rotation_equivariance_relative_error",
    }
    assert set(result["per_case"]) == expected
    for name in expected:
        values = np.asarray(result["per_case"][name])
        assert values.shape == (36,)
        assert np.isfinite(values).all()
        assert np.isfinite(result["means"][name])
    assert result["access_counts"] == {
        "evaluation_calls": 1,
        "locked_test_label_case_reads": 36,
        "locked_test_model_forward_passes": 3,
        "locked_test_model_forward_batches": 15,
    }
    audit = dataset.access_snapshot()
    assert audit["materialized_cases"]["locked_test"] == 36
    assert audit["label_case_reads"]["locked_test"] == 36


def test_cpu_checkpoint_round_trip_is_atomic_and_portable(tmp_path, config) -> None:
    torch = pytest.importorskip("torch")
    model = make_model(config, output_width=3, seed=17, device="cpu")
    path = tmp_path / "method__seed-17.pt"
    digest = save_cpu_checkpoint_atomic(
        model,
        path,
        config,
        seed=17,
        metadata={"method": "scratch_80", "seed": 17, "config_sha256": "abc"},
    )
    assert digest == checkpoint_identity(path)
    assert not path.with_suffix(".pt.tmp").exists()
    payload = load_checkpoint_payload(path)
    assert payload["metadata"]["method"] == "scratch_80"
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())
    loaded, metadata = load_model_checkpoint(
        path,
        config,
        device="cpu",
        expected_metadata={"method": "scratch_80", "seed": 17, "config_sha256": "abc"},
    )
    assert metadata["seed"] == 17
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_dry_run_exercises_all_methods_without_locked_test(dataset, config) -> None:
    pytest.importorskip("torch")
    report = dry_run_contract(dataset, config, device=resolve_device("cpu"))
    assert report["status"] == "dry_run_passed"
    assert report["split_counts"] == {"train": 168, "dev": 36, "locked_test": 36}
    assert report["locked_test_inference_count"] == 0
    assert set(report["methods"]) == set(ALL_METHODS)
    for method, result in report["methods"].items():
        assert np.isfinite(result["finetuning_loss_one_batch"])
        assert result["output_width_after_setup"] == 3
        if method in PRETRAIN_METHODS:
            assert result["pretraining_splits_read"] == ["train"]
            assert np.isfinite(result["pretraining_loss_one_batch"])
            assert result["head_replaced"] is True
    assert report["methods"]["shuffled_stress_lift_80"]["derangement_fixed_points"] == 0


def test_case_level_access_guard_rejects_dev_even_if_case_is_copied(dataset) -> None:
    train = list(dataset.cases)
    index = int(dataset.indices("train")[0])
    train[index] = replace(train[index], split="dev")
    contaminated = replace(dataset, cases=tuple(train))
    with pytest.raises(TransferContractError, match="non-train"):
        build_pretraining_arrays(contaminated, [index], "static_geometry_80", seed=17)
