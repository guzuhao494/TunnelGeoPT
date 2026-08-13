from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from tunnelgeopt.multifidelity import (
    CheckpointRegistry,
    GeometryDataSpec,
    GeometrySplitRecord,
    GeometrySplitSpec,
    MeshFidelitySpec,
    MultiFidelityContractError,
    MultiFidelityDataset,
    MultiFidelitySample,
    build_elastic_query_grid,
    case_group_id,
    elastic_condition_vector,
    farfield_stress_scale,
    freeze_geometry_splits,
    freeze_stratified_geometry_splits,
    load_group_id,
    reconstruct_fine_stress,
    solve_multifidelity_case,
)


def _synthetic_sample(grid, split: str, diagonal_load: float) -> MultiFidelitySample:
    sigma = np.asarray([[-diagonal_load, 0.1 * diagonal_load], [0.1 * diagonal_load, -0.7]])
    load_id = load_group_id(sigma)
    case_id = case_group_id(
        grid.geometry_group_id,
        load_id,
        young_modulus=30.0e9,
        poisson_ratio=0.24,
    )
    coarse = np.full((grid.point_count, 3), diagonal_load, dtype=np.float64)
    fine = coarse + np.asarray([0.2, -0.1, 0.05])
    return MultiFidelitySample(
        geometry_group_id=grid.geometry_group_id,
        load_group_id=load_id,
        case_group_id=case_id,
        split=split,
        grid=grid,
        condition=elastic_condition_vector(sigma, poisson_ratio=0.24),
        stress_scale=farfield_stress_scale(sigma),
        coarse_stress_normalized=coarse,
        _fine_stress_normalized=fine,
        coarse_element_ids=np.arange(grid.point_count, dtype=np.int64),
        fine_element_ids=np.arange(grid.point_count, dtype=np.int64),
        coarse_mesh_metadata={"tier": "coarse"},
        fine_mesh_metadata={"tier": "fine"},
        diagnostics={"sign_convention": "tension_positive"},
    )


def _small_grid(seed: int = 7):
    spec = GeometryDataSpec(
        shape="horseshoe",
        parameters={"span_height_ratio": 0.9, "sidewall_height_ratio": 0.82},
        n_boundary_points=48,
        roughness_amplitude=0.01,
        seed=12,
        outer_domain_scale=3.0,
    )
    geometry = spec.build()
    grid = build_elastic_query_grid(
        geometry,
        geometry_parameters=spec.identity_parameters(),
        nearfield_points=12,
        wall_offset_points=8,
        farfield_points=4,
        nearfield_scale=1.8,
        farfield_scale=2.5,
        seed=seed,
        outer_domain_scale=spec.outer_domain_scale,
    )
    return geometry, grid


def test_identity_split_and_query_grid_are_boundary_level_and_deterministic() -> None:
    geometry, first = _small_grid()
    _, second = _small_grid()
    spec = GeometryDataSpec(
        shape="horseshoe",
        parameters={"span_height_ratio": 0.9, "sidewall_height_ratio": 0.82},
        n_boundary_points=48,
        roughness_amplitude=0.01,
        seed=12,
        outer_domain_scale=3.0,
    )
    other_spec = GeometryDataSpec(
        shape="horseshoe",
        parameters={"span_height_ratio": 1.04, "sidewall_height_ratio": 0.82},
        n_boundary_points=48,
        roughness_amplitude=0.01,
        seed=12,
        outer_domain_scale=3.0,
    )
    other = other_spec.build()
    other_grid = build_elastic_query_grid(
        other,
        geometry_parameters=other_spec.identity_parameters(),
        nearfield_points=12,
        wall_offset_points=8,
        farfield_points=4,
        nearfield_scale=1.8,
        farfield_scale=2.5,
        seed=7,
        outer_domain_scale=other_spec.outer_domain_scale,
    )

    assert spec.geometry_group_id(geometry) == first.geometry_group_id
    assert first.query_hash == second.query_hash
    assert np.array_equal(first.points_yz, second.points_yz)
    assert first.geometry_group_id != other_grid.geometry_group_id
    assert first.query_hash != other_grid.query_hash
    assert first.x.shape == (24, 7)
    assert first.nearfield_mask.sum() == 12
    assert first.wall_offset_mask.sum() == 8
    assert first.farfield_mask.sum() == 4
    assert first.area_weights.sum() == pytest.approx(1.0)
    assert first.arc_weights.sum() == pytest.approx(1.0)
    assert np.all(first.area_weights[~first.nearfield_mask] == 0.0)
    assert np.all(first.arc_weights[~first.wall_offset_mask] == 0.0)
    boundary_distance = first.x[:, 3]
    assert np.all(boundary_distance[first.nearfield_mask] >= 0.05 - 1e-6)
    assert np.all(boundary_distance[first.nearfield_mask] <= 2.0 + 1e-6)

    with pytest.raises(MultiFidelityContractError, match="strictly inside"):
        build_elastic_query_grid(
            geometry,
            nearfield_points=8,
            wall_offset_points=8,
            farfield_points=8,
            farfield_scale=3.0,
            outer_domain_scale=3.0,
        )

    split = freeze_geometry_splits(
        [first.geometry_group_id, other_grid.geometry_group_id, "f" * 64],
        train_count=1,
        dev_count=1,
        locked_test_count=1,
    )
    assert split.geometry_count == 3
    assert split.formal_eligible is False
    assert set(split.as_dict()["counts"]) == {"train", "dev", "locked_test"}
    with pytest.raises(MultiFidelityContractError, match="multiple splits"):
        GeometrySplitSpec(
            train=(first.geometry_group_id,),
            dev=(first.geometry_group_id,),
            locked_test=(),
        )


def test_formal_split_is_salted_section_stratified_and_salt_changes_assignment() -> None:
    records = tuple(
        GeometrySplitRecord(hashlib.sha256(f"{section}:{index}".encode()).hexdigest(), section)
        for section in ("circle", "horseshoe")
        for index in range(9)
    )
    counts = {
        section: {"train": 3, "dev": 3, "locked_test": 3} for section in ("circle", "horseshoe")
    }
    first = freeze_stratified_geometry_splits(
        records, salt="formal-v0.3-a", counts_per_section=counts
    )
    second = freeze_stratified_geometry_splits(
        records, salt="formal-v0.3-b", counts_per_section=counts
    )

    assert first.formal_eligible is True
    assert first.protocol == "salted_stratified_geometry_v1"
    assert first.salt_sha256 == hashlib.sha256(b"formal-v0.3-a").hexdigest()
    assert (first.train, first.dev, first.locked_test) != (
        second.train,
        second.dev,
        second.locked_test,
    )
    for section in counts:
        for split in ("train", "dev", "locked_test"):
            assert (
                sum(
                    first.section_by_geometry[identifier] == section
                    for identifier in getattr(first, split)
                )
                == 3
            )


def test_model_array_residual_reconstruction_and_stress_normalization_contract() -> None:
    _, grid = _small_grid()
    sample = _synthetic_sample(grid, "train", 1.0)
    assert sample.model_features.shape == (grid.point_count, 14)
    assert np.array_equal(sample.model_features[:, :7], grid.x)
    assert np.allclose(sample.model_features[:, 11:14], sample.coarse_stress_normalized)
    with pytest.raises(MultiFidelityContractError, match="fine labels are private"):
        _ = sample.fine_stress_normalized
    frozen_hash = grid.query_hash
    for array in (
        grid.points_yz,
        grid.x,
        grid.nearfield_mask,
        grid.area_weights,
        sample.condition,
        sample.coarse_stress_normalized,
        sample._fine_stress_normalized,
        sample.coarse_element_ids,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0
    with pytest.raises(TypeError, match="frozen mapping"):
        grid.metadata["seed"] = 999
    with pytest.raises(TypeError, match="frozen mapping"):
        sample.diagnostics["tampered"] = True
    assert grid.query_hash == frozen_hash

    points_alias = np.array(grid.points_yz, copy=True)
    detached_grid = replace(grid, points_yz=points_alias)
    points_alias[0, 0] += 100.0
    assert np.array_equal(detached_grid.points_yz, grid.points_yz)
    coarse_alias = np.array(sample.coarse_stress_normalized, copy=True)
    detached_sample = replace(sample, coarse_stress_normalized=coarse_alias)
    coarse_alias[0, 0] += 100.0
    assert np.array_equal(detached_sample.coarse_stress_normalized, sample.coarse_stress_normalized)
    with pytest.raises(MultiFidelityContractError, match="query_hash"):
        replace(grid, x=np.asarray(grid.x) + 0.01)

    sigma = np.asarray([[-12.0, 2.0], [2.0, -5.0]])
    assert farfield_stress_scale(sigma) == pytest.approx(np.sqrt(12.0**2 + 5.0**2 + 2 * 2.0**2))
    condition = elastic_condition_vector(sigma, poisson_ratio=0.25)
    assert condition.shape == (4,)
    assert np.isfinite(condition).all()
    residual = np.full((2, 5, 3), 0.25)
    coarse = np.full((2, 5, 3), 2.0)
    assert np.allclose(reconstruct_fine_stress(coarse, residual), 2.25)
    physical = reconstruct_fine_stress(coarse, residual, stress_scale=np.asarray([10.0, 20.0]))
    assert np.allclose(physical[0], 22.5)
    assert np.allclose(physical[1], 45.0)


def test_all_loads_inherit_geometry_split_and_locked_fine_access_is_denied() -> None:
    _, grid = _small_grid()
    first = _synthetic_sample(grid, "locked_test", 1.0)
    second = _synthetic_sample(grid, "locked_test", 1.2)
    spec = GeometrySplitSpec(train=(), dev=(), locked_test=(grid.geometry_group_id,))
    dataset = MultiFidelityDataset((first, second), spec)
    locked = dataset.indices("locked_test")

    # Coarse at inference is a declared model input and is not a label leak.
    features = dataset.features_for(locked, purpose="checkpoint_inference")
    assert features.shape == (2, grid.point_count, 14)
    with pytest.raises(MultiFidelityContractError, match="checkpoints are frozen"):
        dataset.fine_labels_for(locked, purpose="premature_model_selection")
    audit = dataset.access_snapshot()
    assert audit["coarse_feature_case_reads"]["locked_test"] == 2
    assert audit["fine_label_case_reads"]["locked_test"] == 0
    assert audit["denied_locked_fine_accesses"] == 1
    with pytest.raises(MultiFidelityContractError, match="CheckpointRegistry"):
        dataset.authorize_locked_test(["only-one"])
    with pytest.raises(MultiFidelityContractError, match="SHA-256"):
        CheckpointRegistry(("not-a-hash",))
    checkpoint_ids = tuple(
        hashlib.sha256(f"checkpoint-{index}".encode()).hexdigest() for index in range(2)
    )
    registry = CheckpointRegistry(checkpoint_ids)
    dataset.authorize_locked_test(registry)
    residual = dataset.residual_labels_for(locked, purpose="single_post_freeze_evaluation")
    fine = reconstruct_fine_stress(
        np.stack([first.coarse_stress_normalized, second.coarse_stress_normalized]), residual
    )
    assert fine.shape == (2, grid.point_count, 3)
    authorized = dataset.access_snapshot()
    assert authorized["fine_label_case_reads"]["locked_test"] == 2
    assert authorized["authorized_checkpoint_ids"] == list(checkpoint_ids)
    assert authorized["checkpoint_registry_hash"] == registry.registry_hash
    assert authorized["frozen_checkpoint_count_authorized"] == registry.checkpoint_count

    with pytest.raises(MultiFidelityContractError, match="sample split disagrees"):
        MultiFidelityDataset((replace(first, split="train"),), spec)


def test_different_frozen_boundary_is_rejected_before_any_solver_call() -> None:
    geometry, grid = _small_grid()
    other_spec = GeometryDataSpec(
        shape=geometry.shape,
        parameters={"span_height_ratio": 1.03, "sidewall_height_ratio": 0.82},
        n_boundary_points=48,
        roughness_amplitude=0.01,
        seed=12,
        outer_domain_scale=3.0,
    )
    other = other_spec.build()
    with pytest.raises(MultiFidelityContractError, match="grid/frozen-boundary mismatch"):
        solve_multifidelity_case(
            other,
            grid,
            split="train",
            sigma_inf_tension_positive=np.asarray([[-10.0, 0.0], [0.0, -6.0]]),
            young_modulus=30.0e9,
            poisson_ratio=0.24,
            coarse_mesh=MeshFidelitySpec(0.7, 0.35, 0.7),
            fine_mesh=MeshFidelitySpec(0.4, 0.2, 0.4),
            domain_scale=3.0,
            geometry_spec=other_spec,
        )

    original_spec = GeometryDataSpec(
        shape="horseshoe",
        parameters={"span_height_ratio": 0.9, "sidewall_height_ratio": 0.82},
        n_boundary_points=48,
        roughness_amplitude=0.01,
        seed=12,
        outer_domain_scale=3.0,
    )
    with pytest.raises(MultiFidelityContractError, match="actual domain_scale"):
        solve_multifidelity_case(
            geometry,
            grid,
            split="train",
            sigma_inf_tension_positive=np.asarray([[-10.0, 0.0], [0.0, -6.0]]),
            young_modulus=30.0e9,
            poisson_ratio=0.24,
            coarse_mesh=MeshFidelitySpec(0.7, 0.35, 0.7),
            fine_mesh=MeshFidelitySpec(0.4, 0.2, 0.4),
            domain_scale=3.1,
            geometry_spec=original_spec,
        )


def test_tiny_real_common_query_coarse_fine_e2e() -> None:
    pytest.importorskip("gmsh")
    pytest.importorskip("skfem")
    geometry_spec = GeometryDataSpec(
        shape="circle",
        parameters={"axis_ratio": 1.08},
        n_boundary_points=32,
        seed=2,
        outer_domain_scale=3.0,
    )
    geometry = geometry_spec.build()
    grid = build_elastic_query_grid(
        geometry,
        geometry_parameters=geometry_spec.identity_parameters(),
        nearfield_points=8,
        wall_offset_points=8,
        farfield_points=4,
        nearfield_scale=1.7,
        farfield_scale=2.4,
        wall_offset_over_radius=0.12,
        seed=23,
        outer_domain_scale=geometry_spec.outer_domain_scale,
    )
    sample = solve_multifidelity_case(
        geometry,
        grid,
        split="train",
        sigma_inf_tension_positive=np.asarray([[-10.0e6, 1.0e6], [1.0e6, -6.0e6]]),
        young_modulus=30.0e9,
        poisson_ratio=0.24,
        coarse_mesh=MeshFidelitySpec(0.65, 0.32, 0.65),
        fine_mesh=MeshFidelitySpec(0.38, 0.19, 0.38),
        domain_scale=3.0,
        geometry_spec=geometry_spec,
    )
    spec = GeometrySplitSpec(train=(grid.geometry_group_id,), dev=(), locked_test=())
    dataset = MultiFidelityDataset((sample,), spec)
    train = dataset.indices("train")
    fine = dataset.fine_labels_for(train, purpose="unit_train_label")
    residual = dataset.residual_labels_for(train, purpose="unit_train_residual")

    assert sample.coarse_mesh_metadata["element_count"] < sample.fine_mesh_metadata["element_count"]
    assert np.all(sample.coarse_element_ids >= 0)
    assert np.all(sample.fine_element_ids >= 0)
    assert sample.diagnostics["same_frozen_boundary"] is True
    assert sample.diagnostics["same_outer_bounds"] is True
    expected_bounds = tuple(sample.diagnostics["actual_outer_bounds"])
    assert tuple(sample.coarse_mesh_metadata["actual_outer_bounds"]) == expected_bounds
    assert tuple(sample.fine_mesh_metadata["actual_outer_bounds"]) == expected_bounds
    assert sample.diagnostics["actual_domain_scale"] == 3.0
    assert sample.diagnostics["sign_convention"] == "tension_positive"
    assert sample.model_features.shape == (20, 14)
    assert fine.shape == residual.shape == (1, 20, 3)
    assert np.allclose(
        reconstruct_fine_stress(sample.coarse_stress_normalized[None, ...], residual), fine
    )
