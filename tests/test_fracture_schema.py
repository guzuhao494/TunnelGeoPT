from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from tunnelgeopt.fracture_schema import (
    ARRAY_KEYS,
    DAMAGE_CONVENTION,
    OPTIONAL_ARRAY_KEYS,
    SCHEMA_VERSION,
    SI_UNITS,
    FractureSchemaValidationError,
    FractureTrajectory,
    compute_load_state_sha256,
    load_fracture_trajectory,
    save_fracture_trajectory,
)

CONFIG_HASH = "c" * 64
SOLVER_HASH = "d" * 64
BASIS_HASH = "e" * 64


def _trajectory(
    dtype: type[np.floating] = np.float64,
    *,
    with_basis: bool = False,
) -> FractureTrajectory:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=dtype)
    elements = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    wall_facets = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
    farfield_facets = np.asarray([[0, 3], [2, 3]], dtype=np.int64)
    area = np.asarray([0.5, 0.5], dtype=dtype)
    centers = np.asarray(nodes[elements].mean(axis=1), dtype=dtype)
    load_parameter = np.asarray([0.0, 0.5, 1.0], dtype=dtype)
    farfield_stress = np.asarray(
        [[-10.0, -7.0, 0.0], [-10.0, -7.0, 0.0], [-10.0, -7.0, 0.0]],
        dtype=dtype,
    )
    wall_release_by_facet = np.repeat(load_parameter[:, None], wall_facets.shape[0], axis=1)
    u = np.stack([value * nodes * 0.01 for value in load_parameter]).astype(dtype)
    farfield_dirichlet_dofs = np.asarray([0, 1, 4, 5, 6, 7], dtype=np.int64)
    flattened_u = u.reshape(3, -1)
    farfield_prescribed_displacement = flattened_u[:, farfield_dirichlet_dofs]
    wall_nodal_force = np.zeros_like(flattened_u)
    farfield_reaction_on_rock = np.zeros((3, farfield_dirichlet_dofs.size), dtype=dtype)
    damage_level = np.asarray([0.0, 0.1, 0.2], dtype=dtype)
    damage = np.repeat(damage_level[:, None], nodes.shape[0], axis=1)
    strain = np.zeros((3, 2, 3), dtype=dtype)
    stress = np.zeros((3, 2, 3), dtype=dtype)
    sigma_xx = np.zeros((3, 2), dtype=dtype)
    psi_plus = np.repeat(np.asarray([[0.0], [1.0], [2.0]], dtype=dtype), 2, axis=1)
    psi_minus = np.full((3, 2), 0.5, dtype=dtype)
    history = psi_plus.copy()

    length_scale = 0.2
    fracture_toughness = 10.0
    residual_stiffness = 1.0e-4
    damage_area = damage_level.copy()
    crack_density = damage_level**2 / (2.0 * length_scale)
    fracture_energy = fracture_toughness * crack_density
    degradation = (1.0 - damage_level) ** 2 + residual_stiffness
    elastic_energy = degradation * np.asarray([0.0, 1.0, 2.0], dtype=dtype) + 0.5
    recoverable_energy = elastic_energy + fracture_energy
    farfield_work_increment = np.asarray(
        [
            0.0,
            recoverable_energy[1] - recoverable_energy[0],
            recoverable_energy[2] - recoverable_energy[1],
        ],
        dtype=dtype,
    )
    farfield_reaction_on_rock[1, 2] = 2.0 * farfield_work_increment[1] / 0.005
    farfield_reaction_on_rock[2, 2] = (
        2.0 * farfield_work_increment[2] / 0.005 - farfield_reaction_on_rock[1, 2]
    )
    internal_nodal_force = np.zeros_like(flattened_u)
    internal_nodal_force[:, farfield_dirichlet_dofs] = farfield_reaction_on_rock
    neumann_load_functional = np.zeros(3, dtype=dtype)
    wall_work_increment = np.zeros(3, dtype=dtype)
    cumulative_external_work = np.cumsum(farfield_work_increment).astype(dtype)
    total_potential = recoverable_energy - neumann_load_functional

    load_state_hashes = [
        compute_load_state_sha256(
            load_parameter[index : index + 1],
            farfield_stress[index],
            wall_release_by_facet[index],
            wall_facets,
        )
        for index in range(load_parameter.size)
    ]

    attempt_ledger = [
        {
            "step_index": 0,
            "attempt_index": 0,
            "load_parameter_start": 0.0,
            "load_parameter_target": 0.0,
            "accepted": True,
            "failure_code": None,
            "failure_message": None,
            "newton_iterations": 0,
            "active_set_iterations": 0,
            "staggered_iterations": 0,
            "step_halvings": 0,
            "equilibrium_relative_residual": 0.0,
            "kkt_relative_residual": 0.0,
            "complementarity_relative_residual": 0.0,
            "damage_irreversibility_violation": 0.0,
            "damage_range_violation": 0.0,
            "relative_energy_imbalance": 0.0,
            "load_state_sha256": load_state_hashes[0],
            "neumann_load_functional": 0.0,
            "wall_work_increment": 0.0,
            "farfield_work_increment": 0.0,
            "cumulative_external_work": 0.0,
        },
        {
            "step_index": 1,
            "attempt_index": 0,
            "load_parameter_start": 0.0,
            "load_parameter_target": 1.0,
            "accepted": False,
            "failure_code": "EQUILIBRIUM_NOT_CONVERGED",
            "failure_message": "trial increment exceeded the equilibrium tolerance",
            "newton_iterations": 2,
            "active_set_iterations": 3,
            "staggered_iterations": 1,
            "step_halvings": 0,
            "equilibrium_relative_residual": 1.0e-3,
            "kkt_relative_residual": 1.0e-5,
            "complementarity_relative_residual": 1.0e-5,
            "damage_irreversibility_violation": 0.0,
            "damage_range_violation": 0.0,
            "relative_energy_imbalance": 0.1,
            "load_state_sha256": None,
            "neumann_load_functional": None,
            "wall_work_increment": None,
            "farfield_work_increment": None,
            "cumulative_external_work": None,
        },
        {
            "step_index": 1,
            "attempt_index": 1,
            "load_parameter_start": 0.0,
            "load_parameter_target": 0.5,
            "accepted": True,
            "failure_code": None,
            "failure_message": None,
            "newton_iterations": 4,
            "active_set_iterations": 5,
            "staggered_iterations": 2,
            "step_halvings": 1,
            "equilibrium_relative_residual": 0.0,
            "kkt_relative_residual": 1.0e-8,
            "complementarity_relative_residual": 1.0e-8,
            "damage_irreversibility_violation": 0.0,
            "damage_range_violation": 0.0,
            "relative_energy_imbalance": 0.0,
            "load_state_sha256": load_state_hashes[1],
            "neumann_load_functional": 0.0,
            "wall_work_increment": 0.0,
            "farfield_work_increment": float(farfield_work_increment[1]),
            "cumulative_external_work": float(cumulative_external_work[1]),
        },
        {
            "step_index": 2,
            "attempt_index": 0,
            "load_parameter_start": 0.5,
            "load_parameter_target": 1.0,
            "accepted": True,
            "failure_code": None,
            "failure_message": None,
            "newton_iterations": 4,
            "active_set_iterations": 5,
            "staggered_iterations": 2,
            "step_halvings": 0,
            "equilibrium_relative_residual": 0.0,
            "kkt_relative_residual": 1.0e-8,
            "complementarity_relative_residual": 1.0e-8,
            "damage_irreversibility_violation": 0.0,
            "damage_range_violation": 0.0,
            "relative_energy_imbalance": 0.0,
            "load_state_sha256": load_state_hashes[2],
            "neumann_load_functional": 0.0,
            "wall_work_increment": 0.0,
            "farfield_work_increment": float(farfield_work_increment[2]),
            "cumulative_external_work": float(cumulative_external_work[2]),
        },
    ]

    optional: dict[str, object] = {}
    if with_basis:
        optional = {
            "elastic_basis_stress": np.zeros_like(stress),
            "nonlinear_stress_residual": np.zeros_like(stress),
            "elastic_basis_id": "basis-circle-0001",
            "elastic_basis_sha256": BASIS_HASH,
        }

    return FractureTrajectory(
        nodes=nodes,
        node_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        displacement_dof_ids=np.arange(8, dtype=np.int64).reshape(4, 2),
        damage_dof_ids=np.arange(20, 24, dtype=np.int64),
        elements=elements,
        wall_facets=wall_facets,
        farfield_facets=farfield_facets,
        farfield_dirichlet_dofs=farfield_dirichlet_dofs,
        area=area,
        centers=centers,
        load_parameter=load_parameter,
        farfield_stress=farfield_stress,
        wall_release_by_facet=wall_release_by_facet,
        u=u,
        internal_nodal_force=internal_nodal_force,
        wall_nodal_force=wall_nodal_force,
        farfield_prescribed_displacement=farfield_prescribed_displacement,
        farfield_reaction_on_rock=farfield_reaction_on_rock,
        damage=damage,
        strain=strain,
        stress=stress,
        sigma_xx=sigma_xx,
        psi_plus=psi_plus,
        psi_minus=psi_minus,
        history=history,
        elastic_energy=np.asarray(elastic_energy, dtype=dtype),
        fracture_energy=np.asarray(fracture_energy, dtype=dtype),
        neumann_load_functional=neumann_load_functional,
        wall_work_increment=wall_work_increment,
        farfield_work_increment=farfield_work_increment,
        cumulative_external_work=cumulative_external_work,
        total_potential_energy=np.asarray(total_potential, dtype=dtype),
        damage_area=np.asarray(damage_area, dtype=dtype),
        crack_density_integral=np.asarray(crack_density, dtype=dtype),
        damage_connectivity=np.asarray([0.0, 0.5, 1.0], dtype=dtype),
        displacement_residual=np.asarray([0.0, 1.0e-9, 1.0e-9], dtype=dtype),
        damage_residual=np.asarray([0.0, 1.0e-9, 1.0e-9], dtype=dtype),
        equilibrium_relative_residual=np.zeros(3, dtype=dtype),
        kkt_relative_residual=np.asarray([0.0, 1.0e-8, 1.0e-8], dtype=dtype),
        complementarity_relative_residual=np.asarray([0.0, 1.0e-8, 1.0e-8], dtype=dtype),
        damage_irreversibility_violation=np.zeros(3, dtype=dtype),
        damage_range_violation=np.zeros(3, dtype=dtype),
        history_monotonicity_violation=np.zeros(3, dtype=dtype),
        relative_energy_imbalance=np.zeros(3, dtype=dtype),
        newton_iterations=np.asarray([0, 6, 4], dtype=np.int64),
        active_set_iterations=np.asarray([0, 8, 5], dtype=np.int64),
        staggered_iterations=np.asarray([0, 3, 2], dtype=np.int64),
        step_halvings=np.asarray([0, 1, 0], dtype=np.int64),
        retry_count=np.asarray([0, 1, 0], dtype=np.int64),
        attempt_ledger=attempt_ledger,
        trajectory_id="trajectory-circle-0001",
        case_id="case-circle-0001",
        mesh_id="mesh-circle-fine-0001",
        geometry_id="geometry-circle-0001",
        material_id="material-level-01",
        load_path_id="load-path-p1",
        config_hash=CONFIG_HASH,
        solver_hash=SOLVER_HASH,
        equilibrium_force_normalization_floor=1.0e-12,
        energy_balance_normalization_floor=1.0e-12,
        material={
            "young_modulus": 30.0e9,
            "poisson_ratio": 0.2,
            "fracture_energy": fracture_toughness,
            "length_scale": length_scale,
            "residual_stiffness": residual_stiffness,
            "fracture_model": "AT2",
            "energy_split": "spectral_strain_3d_plane_strain",
        },
        geometry={"section_family": "circle", "radius": 1.0},
        load_path={
            "path_parameter": "s",
            "parameter_bounds": [0, 1],
            "monotone": True,
            "interpolation": "piecewise_linear_between_control_knots",
            "control_knots": [
                {
                    "s": 0.0,
                    "sigma1": 10.0,
                    "sigma3": 7.0,
                    "principal_angle_deg": 0.0,
                    "wall_release": {"all": 0.0},
                },
                {
                    "s": 1.0,
                    "sigma1": 10.0,
                    "sigma3": 7.0,
                    "principal_angle_deg": 0.0,
                    "wall_release": {"all": 1.0},
                },
            ],
        },
        physical_tags={"rock": 1, "wall": 2, "farfield": 3},
        mesh_metadata={
            "element_type": "triangle_p1",
            "displacement_interpolation": "P1",
            "damage_interpolation": "P1",
            "mesh_tier": "fine",
        },
        solver={"name": "synthetic-schema-test", "version": "0"},
        env={"python": "3.13", "numpy": np.__version__},
        meta={"section_family": "circle", "split": "development"},
        **optional,
    )


@pytest.fixture
def trajectory() -> FractureTrajectory:
    return _trajectory()


def test_contract_is_immutable_and_contains_complete_step_fields(
    trajectory: FractureTrajectory,
) -> None:
    trajectory.validate()
    assert trajectory.num_steps == 3
    assert trajectory.u.shape == (3, 4, 2)
    assert trajectory.damage.shape == (3, 4)
    assert trajectory.strain.shape == trajectory.stress.shape == (3, 2, 3)
    assert trajectory.psi_plus.shape == trajectory.history.shape == (3, 2)
    assert set(trajectory.arrays()) == set(ARRAY_KEYS)
    assert trajectory.damage_convention == DAMAGE_CONVENTION
    assert trajectory.units == SI_UNITS

    with pytest.raises(ValueError, match="WRITEABLE"):
        trajectory.damage.setflags(write=True)
    with pytest.raises(TypeError):
        trajectory.material["fracture_energy"] = 20.0
    with pytest.raises(FrozenInstanceError):
        trajectory.accepted = False


def test_constructor_snapshots_caller_arrays_and_metadata() -> None:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    metadata = {"section_family": "circle", "split": "development"}
    trajectory = replace(_trajectory(), nodes=nodes, meta=metadata)
    nodes[0, 0] = 99.0
    metadata["split"] = "locked"
    assert trajectory.nodes[0, 0] == 0.0
    assert trajectory.meta["split"] == "development"


def test_roundtrip_verifies_file_semantic_array_and_mesh_hashes(
    tmp_path, trajectory: FractureTrajectory
) -> None:
    paths = save_fracture_trajectory(tmp_path / "trajectory", trajectory)
    metadata = json.loads(paths.meta.read_text(encoding="utf-8"))
    assert len(metadata["arrays_file_sha256"]) == 64
    assert len(metadata["content_sha256"]) == 64
    assert len(metadata["mesh_content_sha256"]) == 64
    assert len(metadata["identity_content_sha256"]) == 64
    assert metadata["schema_version"] == SCHEMA_VERSION == 2
    assert set(metadata["array_manifest"]) == set(ARRAY_KEYS)

    loaded = load_fracture_trajectory(paths.trajectory_dir)
    assert loaded.trajectory_id == trajectory.trajectory_id
    assert loaded.attempt_ledger == trajectory.attempt_ledger
    for name in ARRAY_KEYS:
        assert np.array_equal(getattr(loaded, name), getattr(trajectory, name))
        assert not getattr(loaded, name).flags.writeable


def test_v1_metadata_is_explicitly_rejected_without_implicit_migration(
    tmp_path, trajectory: FractureTrajectory
) -> None:
    paths = save_fracture_trajectory(tmp_path / "v1", trajectory)
    metadata = json.loads(paths.meta.read_text(encoding="utf-8"))
    metadata["schema_version"] = 1
    paths.meta.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FractureSchemaValidationError, match="unsupported"):
        load_fracture_trajectory(paths.trajectory_dir)


def test_work_reaction_and_total_potential_are_recomputed_from_raw_arrays(
    trajectory: FractureTrajectory,
) -> None:
    trajectory.validate()
    assert np.any(trajectory.cumulative_external_work[1:] != 0.0)
    assert np.array_equal(trajectory.neumann_load_functional, np.zeros(3))
    assert np.allclose(
        trajectory.total_potential_energy,
        trajectory.elastic_energy + trajectory.fracture_energy,
    )
    assert not np.allclose(
        trajectory.total_potential_energy,
        trajectory.elastic_energy
        + trajectory.fracture_energy
        - trajectory.cumulative_external_work,
    )

    bad_reaction = np.asarray(trajectory.farfield_reaction_on_rock).copy()
    bad_reaction[1, 0] += 1.0
    with pytest.raises(FractureSchemaValidationError, match="must equal .*internal_nodal_force"):
        replace(trajectory, farfield_reaction_on_rock=bad_reaction).validate()

    bad_internal = np.asarray(trajectory.internal_nodal_force).copy()
    bad_internal[1, 2] += 1.0
    with pytest.raises(FractureSchemaValidationError, match="recomputed free-DOF"):
        replace(trajectory, internal_nodal_force=bad_internal).validate()

    bad_prescribed = np.asarray(trajectory.farfield_prescribed_displacement).copy()
    bad_prescribed[1, 0] += 1.0e-5
    with pytest.raises(FractureSchemaValidationError, match="must equal u"):
        replace(trajectory, farfield_prescribed_displacement=bad_prescribed).validate()

    bad_work = np.asarray(trajectory.farfield_work_increment).copy()
    bad_work[1] += 1.0
    with pytest.raises(FractureSchemaValidationError, match="recomputed accepted-state"):
        replace(trajectory, farfield_work_increment=bad_work).validate()

    bad_cumulative = np.asarray(trajectory.cumulative_external_work).copy()
    bad_cumulative[2] += 1.0
    with pytest.raises(FractureSchemaValidationError, match="recomputed accepted-state"):
        replace(trajectory, cumulative_external_work=bad_cumulative).validate()

    bad_imbalance = np.asarray(trajectory.relative_energy_imbalance).copy()
    bad_imbalance[1] = 1.0e-2
    with pytest.raises(FractureSchemaValidationError, match="must be recomputed"):
        replace(trajectory, relative_energy_imbalance=bad_imbalance).validate()

    cumulative_potential = (
        trajectory.elastic_energy + trajectory.fracture_energy - trajectory.cumulative_external_work
    )
    with pytest.raises(FractureSchemaValidationError, match="instantaneous"):
        replace(trajectory, total_potential_energy=cumulative_potential).validate()


def test_load_state_and_discrete_identity_are_order_bound(
    trajectory: FractureTrajectory,
) -> None:
    changed_stress = np.asarray(trajectory.farfield_stress).copy()
    changed_stress[1, 0] -= 1.0
    with pytest.raises(FractureSchemaValidationError, match="load_state_sha256"):
        replace(trajectory, farfield_stress=changed_stress).validate()

    changed_release = np.asarray(trajectory.wall_release_by_facet).copy()
    changed_release[1, 0] = 0.4
    with pytest.raises(FractureSchemaValidationError, match="load_state_sha256"):
        replace(trajectory, wall_release_by_facet=changed_release).validate()

    reversed_facets = np.asarray(trajectory.wall_facets)[::-1].copy()
    with pytest.raises(FractureSchemaValidationError, match="load_state_sha256"):
        replace(trajectory, wall_facets=reversed_facets).validate()

    duplicate_node_ids = np.asarray(trajectory.node_ids).copy()
    duplicate_node_ids[1] = duplicate_node_ids[0]
    with pytest.raises(FractureSchemaValidationError, match="unique non-negative"):
        replace(trajectory, node_ids=duplicate_node_ids).validate()

    swapped_displacement_ids = np.asarray(trajectory.displacement_dof_ids).copy()
    swapped_displacement_ids[[0, 1]] = swapped_displacement_ids[[1, 0]]
    with pytest.raises(FractureSchemaValidationError, match="node-major"):
        replace(trajectory, displacement_dof_ids=swapped_displacement_ids).validate()

    unsorted_dofs = np.asarray(trajectory.farfield_dirichlet_dofs)[::-1].copy()
    with pytest.raises(FractureSchemaValidationError, match="strictly increasing"):
        replace(trajectory, farfield_dirichlet_dofs=unsorted_dofs).validate()


def test_rejected_attempts_cannot_publish_accepted_work_or_load_identity(
    trajectory: FractureTrajectory,
) -> None:
    ledger = [dict(entry) for entry in trajectory.attempt_ledger]
    ledger[1]["farfield_work_increment"] = 1.0
    with pytest.raises(FractureSchemaValidationError, match="rejected attempt_ledger"):
        replace(trajectory, attempt_ledger=ledger).validate()

    ledger = [dict(entry) for entry in trajectory.attempt_ledger]
    ledger[1]["load_state_sha256"] = "a" * 64
    with pytest.raises(FractureSchemaValidationError, match="rejected attempt_ledger"):
        replace(trajectory, attempt_ledger=ledger).validate()


@pytest.mark.parametrize("floor", [0.0, -1.0, np.inf, np.nan])
@pytest.mark.parametrize(
    "field_name",
    ["equilibrium_force_normalization_floor", "energy_balance_normalization_floor"],
)
def test_normalization_floors_are_explicit_positive_and_finite(
    trajectory: FractureTrajectory, floor: float, field_name: str
) -> None:
    with pytest.raises(FractureSchemaValidationError, match="finite and strictly positive"):
        replace(trajectory, **{field_name: floor}).validate()


def test_free_dof_equilibrium_is_not_diluted_by_large_constraint_reactions(
    trajectory: FractureTrajectory,
) -> None:
    internal = np.asarray(trajectory.internal_nodal_force).copy()
    wall = np.asarray(trajectory.wall_nodal_force).copy()
    reaction = np.asarray(trajectory.farfield_reaction_on_rock).copy()
    residual = np.asarray(trajectory.equilibrium_relative_residual).copy()

    constrained = np.asarray(trajectory.farfield_dirichlet_dofs)
    # This constrained component has zero prescribed displacement at all steps,
    # so making its reaction huge does not alter the independently recomputed work.
    internal[:, constrained[0]] = 1.0e12
    reaction[:, 0] = internal[:, constrained[0]] - wall[:, constrained[0]]
    free_dof = int(np.flatnonzero(~np.isin(np.arange(internal.shape[1]), constrained))[0])
    internal[1, free_dof] = wall[1, free_dof] + 1.0
    residual[1] = 1.0

    with pytest.raises(FractureSchemaValidationError, match="exceeds 1e-6"):
        replace(
            trajectory,
            internal_nodal_force=internal,
            farfield_reaction_on_rock=reaction,
            equilibrium_relative_residual=residual,
        ).validate()


def test_optional_elastic_basis_is_hash_linked_and_decomposes_stress(tmp_path) -> None:
    trajectory = _trajectory(with_basis=True)
    trajectory.validate()
    assert set(trajectory.arrays()) == set(ARRAY_KEYS) | set(OPTIONAL_ARRAY_KEYS)
    paths = save_fracture_trajectory(tmp_path / "basis", trajectory)
    loaded = load_fracture_trajectory(paths.trajectory_dir)
    assert loaded.elastic_basis_id == "basis-circle-0001"
    assert np.array_equal(
        loaded.stress,
        loaded.elastic_basis_stress + loaded.nonlinear_stress_residual,
    )


def test_float32_requires_explicit_publication_at_save_and_load(tmp_path) -> None:
    trajectory = _trajectory(np.float32)
    with pytest.raises(FractureSchemaValidationError, match="expected float64"):
        trajectory.validate()
    trajectory.validate(expected_dtype=np.float32)

    paths = save_fracture_trajectory(tmp_path / "float32", trajectory, expected_dtype=np.float32)
    with pytest.raises(FractureSchemaValidationError, match="expected float64"):
        load_fracture_trajectory(paths.trajectory_dir)
    assert load_fracture_trajectory(
        paths.trajectory_dir, expected_dtype=np.float32
    ).dtype == np.dtype(np.float32)


def test_existing_record_is_protected_and_overwrite_replaces_both_files(
    tmp_path, trajectory: FractureTrajectory
) -> None:
    trajectory_dir = tmp_path / "trajectory"
    save_fracture_trajectory(trajectory_dir, trajectory)
    changed = replace(trajectory, env={"solver": "replacement"})
    with pytest.raises(FileExistsError, match="protected"):
        save_fracture_trajectory(trajectory_dir, changed)

    save_fracture_trajectory(trajectory_dir, changed, overwrite=True)
    assert load_fracture_trajectory(trajectory_dir).env == {"solver": "replacement"}
    assert not (trajectory_dir / ".fracture-schema.lock").exists()


def test_arrays_corruption_and_metadata_tampering_fail_closed(
    tmp_path, trajectory: FractureTrajectory
) -> None:
    paths = save_fracture_trajectory(tmp_path / "arrays", trajectory)
    with paths.arrays.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(FractureSchemaValidationError, match="SHA-256"):
        load_fracture_trajectory(paths.trajectory_dir)

    paths = save_fracture_trajectory(tmp_path / "metadata", trajectory)
    metadata = json.loads(paths.meta.read_text(encoding="utf-8"))
    metadata["solver"]["version"] = "tampered"
    paths.meta.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(FractureSchemaValidationError, match="content_sha256"):
        load_fracture_trajectory(paths.trajectory_dir)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda item: replace(
                item, load_parameter=np.asarray([0.0, 0.5, 0.4], dtype=np.float64)
            ),
            "strictly increasing",
        ),
        (
            lambda item: replace(
                item,
                damage=np.asarray([[0.0] * 4, [0.1] * 4, [0.05] * 4], dtype=np.float64),
            ),
            "not irreversible",
        ),
        (
            lambda item: replace(
                item,
                damage=np.asarray([[0.0] * 4, [0.1] * 4, [1.01] * 4], dtype=np.float64),
            ),
            "0<=d<=1",
        ),
        (
            lambda item: replace(
                item,
                stress=np.full((3, 2, 3), np.nan, dtype=np.float64),
            ),
            "non-finite",
        ),
        (
            lambda item: replace(
                item,
                equilibrium_relative_residual=np.asarray([0.0, 2.0e-6, 0.0], dtype=np.float64),
            ),
            "recomputed free-DOF",
        ),
        (
            lambda item: replace(
                item,
                fracture_energy=np.asarray(item.fracture_energy) + 1.0,
            ),
            "Gc times",
        ),
        (
            lambda item: replace(item, units={**SI_UNITS, "stress": "MPa"}),
            "SI unit contract",
        ),
        (
            lambda item: replace(item, damage_convention="d=1_intact,d=0_broken"),
            "damage_convention",
        ),
    ],
)
def test_shape_finite_monotonic_physics_residual_and_unit_checks(
    trajectory: FractureTrajectory, mutator, message: str
) -> None:
    invalid = mutator(trajectory)
    with pytest.raises(FractureSchemaValidationError, match=message):
        invalid.validate()


def test_attempt_ledger_must_account_for_retries_iterations_and_halvings(
    trajectory: FractureTrajectory,
) -> None:
    invalid = replace(
        trajectory,
        retry_count=np.asarray([0, 0, 0], dtype=np.int64),
    )
    with pytest.raises(FractureSchemaValidationError, match=r"retry_count\[1\]"):
        invalid.validate()

    ledger = [dict(entry) for entry in trajectory.attempt_ledger]
    ledger[1]["failure_message"] = None
    invalid = replace(trajectory, attempt_ledger=ledger)
    with pytest.raises(FractureSchemaValidationError, match="needs failure_code"):
        invalid.validate()

    ledger = list(trajectory.attempt_ledger)
    invalid = replace(trajectory, attempt_ledger=[ledger[0], ledger[3], ledger[1], ledger[2]])
    with pytest.raises(FractureSchemaValidationError, match="must be ordered"):
        invalid.validate()


def test_phase1_s_paths_support_farfield_changes_and_spatial_wall_release(
    trajectory: FractureTrajectory,
) -> None:
    config_path = Path(__file__).parents[1] / "configs" / "fracture_phase1_pilot.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    path_config = config["load_paths"]
    for candidate in path_config["paths"]:
        load_path = {
            "path_parameter": path_config["path_parameter"],
            "parameter_bounds": [0, 1],
            "monotone": True,
            "coordinate_order": path_config["coordinate_order"],
            "principal_angle_rule": path_config["principal_angle_rule"],
            "stress_sign_rule": path_config["stress_sign_rule"],
            "interpolation": path_config["interpolation_rule"],
            "control_knots": candidate["control_knots"],
        }
        if candidate["id"] == "p4":
            load_path["wall_zones_for_p4"] = path_config["wall_zones_for_p4"]
        replace(trajectory, load_path=load_path).validate()


def test_load_path_rejects_implicit_farfield_or_undefined_spatial_zones(
    trajectory: FractureTrajectory,
) -> None:
    implicit_farfield = {
        "path_parameter": "s",
        "interpolation": "linear",
        "control_knots": [
            {"s": 0.0, "wall_release": {"all": 0.0}},
            {"s": 1.0, "wall_release": {"all": 1.0}},
        ],
    }
    with pytest.raises(FractureSchemaValidationError, match="far-field schedule"):
        replace(trajectory, load_path=implicit_farfield).validate()

    undefined_zones = {
        "path_parameter": "s",
        "interpolation": "linear",
        "control_knots": [
            {"s": 0.0, "sigma1": 10.0, "wall_release": {"crown": 0.0}},
            {"s": 1.0, "sigma1": 10.0, "wall_release": {"crown": 1.0}},
        ],
    }
    with pytest.raises(FractureSchemaValidationError, match="wall-zone definition"):
        replace(trajectory, load_path=undefined_zones).validate()


@pytest.mark.parametrize(
    "forbidden",
    ["elastic_only", "ae_waveform", "contact_pressure", "fragment_velocity"],
)
def test_elastic_only_ae_and_contact_placeholders_are_rejected(
    trajectory: FractureTrajectory, forbidden: str
) -> None:
    with pytest.raises(FractureSchemaValidationError, match="outside"):
        replace(trajectory, meta={forbidden: 0.0})
