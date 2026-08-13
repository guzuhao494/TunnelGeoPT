from dataclasses import replace

import numpy as np
import pytest

from tunnelgeopt.multifidelity_learning import (
    LearningBatch,
    LearningContractError,
    aggregate_case_errors_by_parent,
    build_training_contract,
    build_training_selection,
    case_weighted_stress_error,
    checkpoint_payload,
    hierarchical_paired_bootstrap,
    load_formal_model_from_checkpoint,
    make_model,
    method_arrays,
    mismatched_coarse_indices,
    nested_geometry_subsets,
    reconstruct_fine_prediction,
    save_checkpoint_atomic,
    save_formal_checkpoint_atomic,
    section_balanced_geometry_mean,
    train_formal_with_dev_selection,
    train_with_dev_selection,
    validate_training_selection,
)


def _batch(case_count: int = 6, point_count: int = 5) -> LearningBatch:
    rng = np.random.default_rng(4)
    coarse = rng.normal(size=(case_count, point_count, 3)).astype(np.float32)
    fine = coarse + 0.1 * rng.normal(size=coarse.shape).astype(np.float32)
    sections = tuple("a" if index < case_count // 2 else "b" for index in range(case_count))
    return LearningBatch(
        base_features=rng.normal(size=(case_count, point_count, 11)).astype(np.float32),
        coarse_stress=coarse,
        fine_stress=fine,
        weights=np.ones((case_count, point_count), dtype=np.float32),
        geometry_group_ids=tuple(f"g{index // 2}" for index in range(case_count)),
        section_families=sections,
        case_group_ids=tuple(f"c{index}" for index in range(case_count)),
        splits=tuple("train" for _ in range(case_count)),
    )


def _formal_batch(point_count: int = 5) -> LearningBatch:
    rng = np.random.default_rng(44)
    geometry = (
        "ta0",
        "ta0",
        "ta1",
        "ta1",
        "tb0",
        "tb0",
        "tb1",
        "tb1",
        "da",
        "db",
        "locked",
    )
    sections = ("a", "a", "a", "a", "b", "b", "b", "b", "a", "b", "a")
    splits = (*("train" for _ in range(8)), "dev", "dev", "locked_test")
    coarse = rng.normal(size=(len(geometry), point_count, 3)).astype(np.float32)
    return LearningBatch(
        base_features=rng.normal(size=(len(geometry), point_count, 11)).astype(np.float32),
        coarse_stress=coarse,
        fine_stress=coarse + 0.1 * rng.normal(size=coarse.shape).astype(np.float32),
        weights=np.ones((len(geometry), point_count), dtype=np.float32),
        geometry_group_ids=geometry,
        section_families=sections,
        case_group_ids=tuple(f"case-{index}" for index in range(len(geometry))),
        splits=tuple(splits),
    )


def test_method_contract_and_residual_reconstruction() -> None:
    batch = _batch()
    direct_x, direct_y, direct_base = method_arrays(batch, "direct_coarse")
    residual_x, residual_y, residual_base = method_arrays(batch, "residual_coarse")
    assert direct_x.shape == residual_x.shape == (6, 5, 14)
    np.testing.assert_allclose(direct_x, residual_x)
    np.testing.assert_allclose(direct_y, batch.fine_stress)
    np.testing.assert_allclose(direct_base, 0.0)
    np.testing.assert_allclose(
        reconstruct_fine_prediction(residual_y, residual_base), batch.fine_stress
    )
    scratch_x, _, _ = method_arrays(batch, "scratch")
    np.testing.assert_allclose(scratch_x[..., 11:], 0.0)


def test_mismatched_control_is_within_section_and_has_no_fixed_points() -> None:
    sections = ("a", "a", "a", "b", "b", "b")
    permutation = mismatched_coarse_indices(sections, seed=13)
    assert np.all(permutation != np.arange(6))
    assert all(sections[index] == sections[int(other)] for index, other in enumerate(permutation))
    batch = _batch()
    features, target, base = method_arrays(batch, "mismatched_coarse", mismatch_indices=permutation)
    np.testing.assert_allclose(features[..., 11:], batch.coarse_stress[permutation])
    np.testing.assert_allclose(target + base, batch.fine_stress, atol=2e-7)


def test_nested_geometry_subsets_are_balanced_and_nested() -> None:
    geometry = tuple(f"{section}{index}" for section in "abc" for index in range(8))
    sections = tuple(section for section in "abc" for _ in range(8))
    subsets = nested_geometry_subsets(geometry, sections, salt="frozen")
    assert [len(subsets[value]) for value in (0.25, 0.5, 0.75, 1.0)] == [6, 12, 18, 24]
    assert set(subsets[0.25]) < set(subsets[0.5]) < set(subsets[0.75]) < set(subsets[1.0])


def test_case_metric_and_section_balancing() -> None:
    target = np.ones((4, 3, 3))
    prediction = 1.1 * target
    error = case_weighted_stress_error(prediction, target, np.ones((4, 3)))
    np.testing.assert_allclose(error, 0.1)
    overall, geometry, section = section_balanced_geometry_mean(
        np.asarray([1.0, 3.0, 10.0, 10.0]),
        ("g1", "g1", "g2", "g3"),
        ("a", "a", "b", "b"),
    )
    assert geometry == {"g1": 2.0, "g2": 10.0, "g3": 10.0}
    assert section == {"a": 2.0, "b": 10.0}
    assert overall == 6.0


def test_geometry_cannot_cross_splits() -> None:
    batch = _batch()
    with pytest.raises(LearningContractError, match="crosses"):
        LearningBatch(
            base_features=batch.base_features,
            coarse_stress=batch.coarse_stress,
            fine_stress=batch.fine_stress,
            weights=batch.weights,
            geometry_group_ids=batch.geometry_group_ids,
            section_families=batch.section_families,
            case_group_ids=batch.case_group_ids,
            splits=("train", "dev", "train", "train", "train", "train"),
        )


def test_formal_selection_is_derived_from_real_splits_and_fraction() -> None:
    batch = _formal_batch()
    selection = build_training_selection(
        batch,
        ("ta0", "tb0"),
        expected_fine_fraction=0.5,
    )
    assert selection.fine_fraction == 0.5
    assert selection.train_geometry_ids == ("ta0", "tb0")
    assert selection.train_case_ids == ("case-0", "case-1", "case-4", "case-5")
    assert all(batch.splits[index] == "train" for index in selection.train_indices)
    assert all(batch.splits[index] == "dev" for index in selection.dev_indices)
    assert selection.section_geometry_counts == {"a": 1, "b": 1}

    with pytest.raises(LearningContractError, match="declared fine fraction disagrees"):
        build_training_selection(
            batch,
            ("ta0", "ta1", "tb0", "tb1"),
            expected_fine_fraction=0.5,
        )
    with pytest.raises(LearningContractError, match="outside the train split"):
        build_training_selection(batch, ("ta0", "db"))

    forged = replace(selection, train_indices=(*selection.train_indices, 8))
    with pytest.raises(LearningContractError, match="not derived"):
        validate_training_selection(batch, forged)


def test_parent_aggregation_precedes_unique_geometry_bootstrap() -> None:
    case_values = np.asarray(
        [
            [1.0, 3.0, 4.0, 6.0],
            [2.0, 4.0, 5.0, 7.0],
        ]
    )
    geometry = ("g1", "g1", "g2", "g2")
    sections = ("a", "a", "b", "b")
    with pytest.raises(LearningContractError, match="must be unique"):
        hierarchical_paired_bootstrap(
            0.8 * case_values,
            case_values,
            (3, 5),
            geometry,
            sections,
            replicates=20,
            confidence=0.95,
            bootstrap_seed=7,
        )
    reference, parent_ids, parent_sections = aggregate_case_errors_by_parent(
        case_values, geometry, sections
    )
    assert parent_ids == ("g1", "g2")
    assert parent_sections == ("a", "b")
    np.testing.assert_allclose(reference, [[2.0, 5.0], [3.0, 6.0]])
    result = hierarchical_paired_bootstrap(
        0.8 * reference,
        reference,
        (3, 5),
        parent_ids,
        parent_sections,
        replicates=50,
        confidence=0.95,
        bootstrap_seed=7,
    )
    assert result["center_ratio"] == pytest.approx(0.8)
    with pytest.raises(LearningContractError, match="seeds must be non-empty and unique"):
        hierarchical_paired_bootstrap(
            0.8 * reference,
            reference,
            (3, 3),
            parent_ids,
            parent_sections,
            replicates=20,
            confidence=0.95,
            bootstrap_seed=7,
        )


def test_hierarchical_bootstrap_preserves_known_ratio() -> None:
    reference = np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 4.0, 3.0]])
    result = hierarchical_paired_bootstrap(
        0.8 * reference,
        reference,
        (3, 5),
        ("g1", "g2", "g3", "g4"),
        ("a", "a", "b", "b"),
        replicates=200,
        confidence=0.95,
        bootstrap_seed=7,
    )
    assert result["center_ratio"] == pytest.approx(0.8)
    assert result["lower"] == pytest.approx(0.8)
    assert result["upper"] == pytest.approx(0.8)


def test_short_training_and_cpu_checkpoint_roundtrip(tmp_path) -> None:
    pytest.importorskip("torch")
    batch = _batch(case_count=8, point_count=8)
    train = np.arange(6)
    dev = np.arange(6, 8)
    features, targets, base = method_arrays(batch, "residual_coarse")
    model_config = {
        "point_input_width": 14,
        "hidden_width": 16,
        "global_context_blocks": 1,
        "output_width": 3,
    }
    model = make_model(model_config, seed=2, device="cpu")
    outcome = train_with_dev_selection(
        model,
        features[train],
        targets[train],
        batch.weights[train],
        features[dev],
        batch.fine_stress[dev],
        base[dev],
        batch.weights[dev],
        seed=2,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=3,
        max_epochs=2,
        patience=2,
        min_delta=0.0,
    )
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint_atomic(
        outcome,
        path,
        method="residual_coarse",
        fraction=0.5,
        seed=2,
        model_config=model_config,
        config_sha256="a" * 64,
        train_geometry_ids=("g1", "g2"),
    )
    assert len(digest) == 64
    payload = checkpoint_payload(path)
    assert payload["method"] == "residual_coarse"
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())


def test_formal_training_checkpoint_binds_actual_selection_and_config(tmp_path) -> None:
    pytest.importorskip("torch")
    batch = _formal_batch(point_count=8)
    config_hash = "a" * 64
    contract = build_training_contract(
        batch,
        method="residual_coarse",
        config_sha256=config_hash,
        train_geometry_selector=("ta0", "tb0"),
        expected_fine_fraction=0.5,
    )
    model_config = {
        "point_input_width": 14,
        "hidden_width": 16,
        "global_context_blocks": 1,
        "output_width": 3,
    }
    model = make_model(model_config, seed=9, device="cpu")
    outcome = train_formal_with_dev_selection(
        model,
        batch,
        contract,
        seed=9,
        device="cpu",
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=2,
        max_epochs=2,
        patience=2,
        min_delta=0.0,
    )
    path = tmp_path / "formal.pt"
    digest = save_formal_checkpoint_atomic(
        outcome,
        path,
        contract=contract,
        seed=9,
        model_config=model_config,
    )
    assert len(digest) == 64
    payload = checkpoint_payload(
        path,
        expected_config_sha256=config_hash,
        expected_selection_sha256=contract.selection.selection_sha256,
        require_formal=True,
    )
    assert payload["format_version"] == 2
    assert payload["fine_fraction"] == 0.5
    assert tuple(payload["train_geometry_ids"]) == ("ta0", "tb0")
    assert tuple(payload["train_case_ids"]) == (
        "case-0",
        "case-1",
        "case-4",
        "case-5",
    )
    loaded, loaded_payload = load_formal_model_from_checkpoint(
        path,
        contract=contract,
        device="cpu",
    )
    assert loaded is not None
    assert loaded_payload["training_contract_sha256"] == contract.contract_sha256
    with pytest.raises(LearningContractError, match="config hash"):
        checkpoint_payload(
            path,
            expected_config_sha256="b" * 64,
            expected_selection_sha256=contract.selection.selection_sha256,
            require_formal=True,
        )
    with pytest.raises(LearningContractError, match="selection hash"):
        checkpoint_payload(
            path,
            expected_config_sha256=config_hash,
            expected_selection_sha256="b" * 64,
            require_formal=True,
        )
    with pytest.raises(LearningContractError, match="different formal contract"):
        save_formal_checkpoint_atomic(
            outcome,
            tmp_path / "wrong.pt",
            contract=replace(contract, contract_sha256="b" * 64),
            seed=9,
            model_config=model_config,
        )
