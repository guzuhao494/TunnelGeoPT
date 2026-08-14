from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
skfem = pytest.importorskip("skfem")

from tunnelgeopt.fracture import AT2Material, FractureSolverOptions, at2_fracture_energy
from tunnelgeopt.fracture_bvp import (
    PrescribedDisplacementState,
    prescribed_displacement_mesh_identity,
    solve_at2_dirichlet_path,
    solve_fixed_damage_displacement_bvp,
)


def _mesh() -> object:
    return skfem.MeshTri.init_sqsymmetric()


def _boundary_nodes(mesh: object) -> np.ndarray:
    facets = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    return np.unique(np.asarray(mesh.facets, dtype=np.int64)[:, facets])


def _node_major_dofs(nodes: np.ndarray) -> np.ndarray:
    return np.sort(np.column_stack((2 * nodes, 2 * nodes + 1)).ravel())


def _state(
    mesh: object,
    *,
    identity: str = "state-000",
    sequence_index: int = 0,
    path_parameter: float = 0.0,
    dirichlet_dofs: np.ndarray | None = None,
    dirichlet_values: np.ndarray | None = None,
    mesh_identity: str | None = None,
) -> PrescribedDisplacementState:
    nodes = np.asarray(mesh.p.T, dtype=np.float64)
    z_max_nodes = np.flatnonzero(np.isclose(nodes[:, 1], np.max(nodes[:, 1])))
    z_min_nodes = np.flatnonzero(np.isclose(nodes[:, 1], np.min(nodes[:, 1])))
    dofs = _node_major_dofs(_boundary_nodes(mesh)) if dirichlet_dofs is None else dirichlet_dofs
    values = np.zeros(dofs.size) if dirichlet_values is None else dirichlet_values
    return PrescribedDisplacementState(
        identity=identity,
        mesh_identity=mesh_identity or prescribed_displacement_mesh_identity(mesh),
        sequence_index=sequence_index,
        path_parameter=path_parameter,
        dirichlet_dofs=dofs,
        dirichlet_values=values,
        external_force=np.zeros(2 * nodes.shape[0]),
        reaction_groups={
            "z_max_u_z": np.sort(2 * z_max_nodes + 1),
            "z_min_u_z": np.sort(2 * z_min_nodes + 1),
        },
        driven_group="z_max_u_z",
    )


def _material(*, fracture_toughness: float = 1.0e6) -> AT2Material:
    return AT2Material(
        young_modulus=100.0,
        poisson_ratio=0.25,
        fracture_toughness=fracture_toughness,
        length_scale=0.1,
        residual_stiffness=0.0,
    )


def _options() -> FractureSolverOptions:
    return FractureSolverOptions(
        equilibrium_tolerance=1.0e-10,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=1.0e-9,
        energy_tolerance=1.0e-9,
    )


def test_affine_patch_reaction_sign_and_immutable_contract() -> None:
    mesh = _mesh()
    nodes = np.asarray(mesh.p.T, dtype=np.float64)
    boundary_nodes = _boundary_nodes(mesh)
    dofs = _node_major_dofs(boundary_nodes)
    affine = np.column_stack(
        (
            0.010 * nodes[:, 0] + 0.002 * nodes[:, 1],
            -0.003 * nodes[:, 0] + 0.020 * nodes[:, 1],
        )
    )
    state = _state(mesh, dirichlet_dofs=dofs, dirichlet_values=affine.ravel()[dofs])
    result = solve_fixed_damage_displacement_bvp(
        mesh, _material(), state, damage=0.0, options=_options()
    )

    assert result.converged
    assert np.allclose(result.displacement, affine, rtol=0.0, atol=5.0e-15)
    assert np.allclose(
        result.strain,
        np.asarray([0.010, 0.020, -0.001]),
        rtol=0.0,
        atol=3.0e-15,
    )
    assert np.allclose(result.reaction, result.internal_force - result.external_force)
    assert result.generalized_load == pytest.approx(
        np.sum(result.reaction[state.reaction_groups[state.driven_group]])
    )
    reaction_vectors = result.reaction.reshape(-1, 2)
    assert np.linalg.norm(np.sum(reaction_vectors, axis=0)) < 2.0e-13
    assert result.neumann_load_functional == 0.0
    assert result.total_potential_energy == result.elastic_energy

    # Applied force on a prescribed DOF does not change the imposed solution;
    # it subtracts exactly from the support-on-rock reaction.
    applied = np.zeros(2 * nodes.shape[0])
    applied[state.reaction_groups["z_max_u_z"]] = 0.25
    loaded_state = replace(state, identity="state-loaded", external_force=applied)
    loaded = solve_fixed_damage_displacement_bvp(
        mesh, _material(), loaded_state, damage=0.0, options=_options()
    )
    assert np.array_equal(loaded.displacement, result.displacement)
    assert np.allclose(loaded.reaction, result.reaction - applied, rtol=0.0, atol=2.0e-15)
    assert loaded.total_potential_energy == pytest.approx(
        loaded.elastic_energy - applied @ affine.ravel()
    )

    for array in (
        state.dirichlet_dofs,
        state.dirichlet_values,
        state.external_force,
        state.reaction_groups["z_max_u_z"],
        result.displacement,
        result.reaction,
        result.reaction_groups["z_min_u_z"],
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0
    with pytest.raises(TypeError):
        state.reaction_groups["new"] = np.asarray([1])  # type: ignore[index]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("duplicate", "exactly increasing"),
        ("out_of_range", "missing from external_force"),
        ("nonfinite_value", "finite"),
        ("group_outside", "subset"),
        ("missing_driven", "driven_group"),
    ],
)
def test_state_contract_fails_closed(mutation: str, match: str) -> None:
    mesh = _mesh()
    nodes = np.asarray(mesh.p.T)
    dofs = _node_major_dofs(_boundary_nodes(mesh))
    values = np.zeros(dofs.size)
    groups: dict[str, np.ndarray] = {"drive": np.asarray([dofs[-1]])}
    driven = "drive"
    if mutation == "duplicate":
        dofs = np.insert(dofs, 1, dofs[0])
        values = np.zeros(dofs.size)
    elif mutation == "out_of_range":
        dofs = np.append(dofs, 2 * nodes.shape[0])
        values = np.zeros(dofs.size)
    elif mutation == "nonfinite_value":
        values[0] = np.nan
    elif mutation == "group_outside":
        missing = next(index for index in range(2 * nodes.shape[0]) if index not in dofs)
        groups = {"drive": np.asarray([missing])}
    elif mutation == "missing_driven":
        driven = "absent"

    with pytest.raises((TypeError, ValueError), match=match):
        PrescribedDisplacementState(
            identity="bad",
            mesh_identity=prescribed_displacement_mesh_identity(mesh),
            sequence_index=0,
            path_parameter=0.0,
            dirichlet_dofs=dofs,
            dirichlet_values=values,
            external_force=np.zeros(2 * nodes.shape[0]),
            reaction_groups=groups,
            driven_group=driven,
        )


def test_mesh_identity_and_missing_rigid_constraints_are_rejected() -> None:
    mesh = _mesh()
    good = _state(mesh)
    with pytest.raises(ValueError, match="mesh_identity"):
        solve_fixed_damage_displacement_bvp(
            mesh,
            _material(),
            replace(good, mesh_identity="0" * 64),
            damage=0.0,
        )

    nodes = np.asarray(mesh.p.T)
    insufficient = PrescribedDisplacementState(
        identity="underconstrained",
        mesh_identity=prescribed_displacement_mesh_identity(mesh),
        sequence_index=0,
        path_parameter=0.0,
        dirichlet_dofs=np.asarray([0, 1]),
        dirichlet_values=np.zeros(2),
        external_force=np.zeros(2 * nodes.shape[0]),
        reaction_groups={"drive": np.asarray([1])},
        driven_group="drive",
    )
    with pytest.raises(ValueError, match="rigid-body mode"):
        solve_fixed_damage_displacement_bvp(mesh, _material(), insufficient, damage=0.0)


def _tension_state(mesh: object, index: int, displacement: float) -> PrescribedDisplacementState:
    nodes = np.asarray(mesh.p.T, dtype=np.float64)
    z_max_nodes = np.flatnonzero(np.isclose(nodes[:, 1], np.max(nodes[:, 1])))
    z_min_nodes = np.flatnonzero(np.isclose(nodes[:, 1], np.min(nodes[:, 1])))
    z_min_dofs = _node_major_dofs(z_min_nodes)
    z_max_u_z = np.sort(2 * z_max_nodes + 1)
    dofs = np.unique(np.concatenate((z_min_dofs, z_max_u_z)))
    values = np.zeros(dofs.size)
    values[np.isin(dofs, z_max_u_z)] = displacement
    return PrescribedDisplacementState(
        identity=f"tension-{index:03d}",
        mesh_identity=prescribed_displacement_mesh_identity(mesh),
        sequence_index=index,
        path_parameter=float(index),
        dirichlet_dofs=dofs,
        dirichlet_values=values,
        external_force=np.zeros(2 * nodes.shape[0]),
        reaction_groups={
            "z_max_u_z": z_max_u_z,
            "z_min_u_z": np.sort(2 * z_min_nodes + 1),
        },
        driven_group="z_max_u_z",
    )


def test_at2_path_initial_state_irreversibility_reaction_work_and_energy() -> None:
    mesh = _mesh()
    initial_damage = np.full(mesh.p.shape[1], 0.2)
    initial_history = np.full(mesh.t.shape[1], 0.3)
    states = (_tension_state(mesh, 0, 0.0), _tension_state(mesh, 1, 1.0e-4))
    result = solve_at2_dirichlet_path(
        mesh,
        _material(),
        states,
        initial_damage=initial_damage,
        initial_history=initial_history,
        options=_options(),
    )

    assert len(result.steps) == 2
    assert all(step.converged for step in result.steps)
    first, second = result.steps
    assert np.all(first.damage >= initial_damage)
    assert np.all(second.damage >= first.damage)
    assert np.all(first.history >= initial_history)
    assert np.all(second.history >= first.history)
    assert first.path_work_increment == 0.0
    assert first.path_work == 0.0

    dofs = states[0].dirichlet_dofs
    delta_u = second.displacement.ravel()[dofs] - first.displacement.ravel()[dofs]
    expected_increment = 0.5 * (first.reaction[dofs] + second.reaction[dofs]) @ delta_u
    assert second.path_work_increment == pytest.approx(expected_increment, rel=1.0e-14)
    assert second.path_work == pytest.approx(expected_increment, rel=1.0e-14)
    assert second.generalized_load == pytest.approx(
        np.sum(second.reaction[second.reaction_groups[second.driven_group]])
    )
    for step in result.steps:
        assert step.neumann_load_functional == 0.0
        assert step.total_potential_energy == pytest.approx(
            step.elastic_energy + step.fracture_energy, rel=1.0e-14
        )
        assert step.fracture_energy == pytest.approx(
            at2_fracture_energy(mesh, _material(), step.damage), rel=1.0e-14
        )
        assert step.energy_change <= _options().energy_tolerance
        assert step.staggered_iterations >= 2
        assert step.irreversibility_violation == 0.0


def test_path_sequence_and_identity_contract_fails_before_solving() -> None:
    mesh = _mesh()
    zero = _tension_state(mesh, 0, 0.0)
    one = _tension_state(mesh, 1, 1.0e-4)

    with pytest.raises(ValueError, match="sequence_index"):
        solve_at2_dirichlet_path(mesh, _material(), (replace(zero, sequence_index=1),))
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_at2_dirichlet_path(
            mesh,
            _material(),
            (zero, replace(one, path_parameter=zero.path_parameter)),
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        solve_at2_dirichlet_path(mesh, _material(), (zero, replace(one, identity=zero.identity)))
    altered_dofs = one.dirichlet_dofs[:-1]
    altered = replace(
        one,
        dirichlet_dofs=altered_dofs,
        dirichlet_values=one.dirichlet_values[:-1],
        reaction_groups={"z_min_u_z": one.reaction_groups["z_min_u_z"]},
        driven_group="z_min_u_z",
    )
    with pytest.raises(ValueError, match="identical dirichlet_dofs"):
        solve_at2_dirichlet_path(mesh, _material(), (zero, altered))
