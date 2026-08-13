from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("scipy")
skfem = pytest.importorskip("skfem")

import tunnelgeopt.fracture as fracture_module
from tunnelgeopt.elasticity import (
    plane_strain_sigma_xx,
    plane_strain_stress,
    solve_plane_strain_excavation,
)
from tunnelgeopt.fracture import (
    AT2LoadPath,
    AT2Material,
    FractureSolverOptions,
    _evaluate_fixed_damage_displacement_at_load_state,
    _evaluate_fixed_damage_displacement_state,
    assemble_at2_damage_system,
    degradation,
    miehe_spectral_response,
    plane_strain_spectral_split,
    solve_at2_damage,
    solve_at2_fracture,
    solve_at2_fracture_schedule,
    solve_fixed_damage_displacement,
    solve_fixed_damage_displacement_at_load_state,
    update_history,
)
from tunnelgeopt.fracture_loading import compile_phase1_load_schedule
from tunnelgeopt.fracture_validation import load_fracture_phase1_config
from tunnelgeopt.fracture_work import BoundaryEquilibriumState
from tunnelgeopt.mesh import FARFIELD, WALL, TunnelMesh


def _square_annulus_mesh() -> object:
    """Return eight P1 triangles around a square hole for a tiny solver check."""

    nodes = np.asarray(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [2.0, 2.0],
            [-2.0, 2.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    elements = np.asarray(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    mesh = skfem.MeshTri(nodes.T, elements.T, validate=True, sort_t=False)
    boundary = np.asarray(mesh.boundary_facets(), dtype=np.int64)
    boundary_nodes = np.asarray(mesh.facets[:, boundary], dtype=np.int64)
    wall = boundary[np.all(np.isin(boundary_nodes, [4, 5, 6, 7]), axis=0)]
    farfield = boundary[np.all(np.isin(boundary_nodes, [0, 1, 2, 3]), axis=0)]
    return mesh.with_boundaries({WALL: wall, FARFIELD: farfield})


def _square_annulus_tunnel_mesh() -> TunnelMesh:
    mesh = _square_annulus_mesh()
    wall = np.asarray(mesh.boundaries[WALL], dtype=np.int64)
    farfield = np.asarray(mesh.boundaries[FARFIELD], dtype=np.int64)
    return TunnelMesh(
        mesh=mesh,
        nodes=np.asarray(mesh.p.T, dtype=np.float64),
        elements=np.asarray(mesh.t.T, dtype=np.int64),
        boundary_facets={WALL: wall, FARFIELD: farfield},
        facet_markers=np.zeros(mesh.facets.shape[1], dtype=np.int64),
        cell_markers=np.zeros(mesh.t.shape[1], dtype=np.int64),
        physical_tags={WALL: 1, FARFIELD: 2},
        outer_bounds=(-2.0, 2.0, -2.0, 2.0),
        metadata={"test_fixture": "square_annulus"},
    )


def _intact_material() -> AT2Material:
    return AT2Material(
        young_modulus=100.0,
        poisson_ratio=0.25,
        fracture_toughness=1.0e6,
        length_scale=0.5,
        residual_stiffness=0.0,
    )


def _strict_options() -> FractureSolverOptions:
    return FractureSolverOptions(
        equilibrium_tolerance=1.0e-10,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=1.0e-9,
    )


@pytest.mark.parametrize("value", [0.0, -1.0e-8, np.inf, -np.inf, np.nan])
def test_energy_tolerance_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="energy_tolerance must be finite and positive"):
        FractureSolverOptions(energy_tolerance=value)


def test_three_dimensional_plane_strain_spectral_reconstruction_and_energy() -> None:
    material = AT2Material(
        young_modulus=120.0,
        poisson_ratio=0.23,
        fracture_toughness=2.5,
        length_scale=0.15,
        residual_stiffness=0.0,
    )
    strain = np.asarray(
        [
            [0.030, -0.012, 0.040],
            [-0.010, 0.025, -0.018],
            [0.004, 0.009, 0.000],
        ]
    )

    full, positive, negative, principal = plane_strain_spectral_split(strain)
    response = miehe_spectral_response(strain, material)

    assert full.shape == (3, 3, 3)
    assert np.allclose(positive + negative, full, rtol=0.0, atol=2.0e-15)
    assert np.all(np.linalg.eigvalsh(positive) >= -2.0e-15)
    assert np.all(np.linalg.eigvalsh(negative) <= 2.0e-15)
    assert np.allclose(principal, np.linalg.eigvalsh(full), rtol=0.0, atol=2.0e-15)

    inplane_stress = plane_strain_stress(
        strain,
        young_modulus=material.young_modulus,
        poisson_ratio=material.poisson_ratio,
    )
    sigma_xx = plane_strain_sigma_xx(
        strain,
        young_modulus=material.young_modulus,
        poisson_ratio=material.poisson_ratio,
    )
    assert np.allclose(response.stress[..., 1, 1], inplane_stress[:, 0], atol=2.0e-13)
    assert np.allclose(response.stress[..., 2, 2], inplane_stress[:, 1], atol=2.0e-13)
    assert np.allclose(response.stress[..., 1, 2], inplane_stress[:, 2], atol=2.0e-13)
    assert np.allclose(response.stress[..., 0, 0], sigma_xx, atol=2.0e-13)
    direct_energy = 0.5 * np.sum(strain * inplane_stress, axis=1)
    assert np.allclose(
        response.psi_positive + response.psi_negative,
        direct_energy,
        rtol=2.0e-14,
        atol=2.0e-15,
    )


def test_pure_compression_has_no_tensile_drive_and_rigid_strain_is_zero() -> None:
    material = AT2Material(100.0, 0.25, 2.0, 0.2, residual_stiffness=1.0e-7)
    compression = np.asarray([-0.020, -0.010, 0.0])
    intact = miehe_spectral_response(compression, material, damage=0.0)
    damaged = miehe_spectral_response(compression, material, damage=0.85)

    # The plane-strain x eigenvalue is exactly zero: it is neither positive nor
    # negative and must not create an artificial compression damage drive.
    assert intact.principal_strains[-1] == pytest.approx(0.0, abs=1.0e-15)
    assert intact.psi_positive == pytest.approx(0.0, abs=1.0e-15)
    assert np.array_equal(intact.stress_positive, np.zeros((3, 3)))
    assert intact.psi_negative > 0.0
    assert np.allclose(damaged.stress, intact.stress, rtol=0.0, atol=2.0e-14)

    rigid = miehe_spectral_response(np.zeros(3), material, damage=0.4)
    assert rigid.psi_positive == 0.0
    assert rigid.psi_negative == 0.0
    assert np.array_equal(rigid.strain_positive, np.zeros((3, 3)))
    assert np.array_equal(rigid.strain_negative, np.zeros((3, 3)))
    assert np.array_equal(rigid.stress, np.zeros((3, 3)))
    assert degradation(0.0, residual_stiffness=0.0) == pytest.approx(1.0)
    assert degradation(1.0, residual_stiffness=1.0e-7) == pytest.approx(1.0e-7)


def test_history_update_is_monotone_and_shape_checked() -> None:
    previous = np.asarray([0.2, 0.5, 0.1])
    current = np.asarray([0.3, 0.4, 0.1])
    updated = update_history(previous, current)
    assert np.array_equal(updated, np.asarray([0.3, 0.5, 0.1]))
    with pytest.raises(ValueError, match="broadcast-compatible"):
        update_history(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError, match="nonnegative"):
        update_history(np.asarray([-1.0]), np.asarray([0.0]))


def test_linear_damage_solution_irreversibility_and_kkt_diagnostics() -> None:
    mesh = skfem.MeshTri.init_sqsymmetric()
    material = AT2Material(100.0, 0.25, 2.0, 0.2)
    history_value = 0.8
    system = assemble_at2_damage_system(mesh, material, history_value)
    result = solve_at2_damage(system, damage_old=0.0)

    expected = (
        2.0
        * history_value
        / (material.fracture_toughness / material.length_scale + 2.0 * history_value)
    )
    assert result.converged
    assert np.allclose(result.damage, expected, rtol=2.0e-13, atol=2.0e-15)
    assert result.kkt_residual < 1.0e-12
    assert result.stationarity_residual < 1.0e-12
    assert result.complementarity_residual < 1.0e-14
    assert result.primal_violation == 0.0
    assert result.irreversibility_violation == 0.0
    assert result.range_violation == 0.0

    lower_bound = np.full(mesh.p.shape[1], expected + 0.2)
    unloading_system = assemble_at2_damage_system(mesh, material, 0.0)
    irreversible = solve_at2_damage(unloading_system, damage_old=lower_bound)
    assert irreversible.converged
    assert np.array_equal(irreversible.damage, lower_bound)
    assert irreversible.active_lower_count == mesh.p.shape[1]
    assert irreversible.kkt_residual < 1.0e-12
    assert irreversible.complementarity_residual == 0.0
    assert irreversible.irreversibility_violation == 0.0

    refined = skfem.MeshTri.init_sqsymmetric().refined(1)
    nonuniform_history = np.linspace(0.01, 20.0, refined.t.shape[1])
    nonuniform_lower = np.clip(0.6 + 0.15 * np.sin(np.arange(refined.p.shape[1])), 0.0, 0.95)
    mixed = solve_at2_damage(
        assemble_at2_damage_system(refined, material, nonuniform_history),
        damage_old=nonuniform_lower,
    )
    assert mixed.converged
    assert 0 < mixed.active_lower_count < refined.p.shape[1]
    assert np.any(mixed.damage > nonuniform_lower + 1.0e-8)
    assert np.all(mixed.damage >= nonuniform_lower)
    assert np.all(mixed.damage <= 1.0)
    assert mixed.kkt_residual < 1.0e-12

    scaled_system = type(system)(
        stiffness=1.0e9 * system.stiffness,
        load=1.0e9 * system.load,
        nodes=system.nodes,
        elements=system.elements,
        history=system.history,
        element_area=system.element_area,
    )
    scaled_result = solve_at2_damage(scaled_system, damage_old=0.0)
    assert np.allclose(scaled_result.damage, result.damage, rtol=2.0e-13, atol=2.0e-15)
    assert scaled_result.kkt_residual < 1.0e-12


def test_tiny_total_field_staggered_smoke_and_intact_elastic_regression() -> None:
    mesh = _square_annulus_mesh()
    sigma_inf = np.asarray([[-1.0, 0.0], [0.0, -1.0]])
    material = AT2Material(
        young_modulus=100.0,
        poisson_ratio=0.25,
        fracture_toughness=1.0e6,
        length_scale=0.5,
        residual_stiffness=0.0,
    )
    options = FractureSolverOptions(
        equilibrium_tolerance=1.0e-10,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=1.0e-9,
    )

    fixed = solve_fixed_damage_displacement(
        mesh,
        material,
        sigma_inf,
        load_parameter=1.0,
        damage=0.0,
        options=options,
    )
    elastic = solve_plane_strain_excavation(
        mesh,
        young_modulus=material.young_modulus,
        poisson_ratio=material.poisson_ratio,
        sigma_inf=sigma_inf,
    )
    assert fixed.converged
    assert fixed.equilibrium_residual < 1.0e-12
    assert np.allclose(
        fixed.correction_displacement,
        elastic.displacement,
        rtol=2.0e-13,
        atol=2.0e-16,
    )
    assert np.allclose(fixed.stress, elastic.total_stress, rtol=2.0e-13, atol=2.0e-13)

    result = solve_at2_fracture(
        mesh,
        material,
        sigma_inf,
        load_path=AT2LoadPath((0.0, 0.5, 1.0)),
        options=options,
    )
    assert result.converged
    assert [step.load_parameter for step in result.steps] == [0.0, 0.5, 1.0]
    assert all(step.converged for step in result.steps)
    assert all(step.equilibrium_residual < 1.0e-10 for step in result.steps)
    assert all(step.kkt_residual < 1.0e-10 for step in result.steps)
    assert all(step.irreversibility_violation == 0.0 for step in result.steps)
    assert all(step.range_violation == 0.0 for step in result.steps)
    assert np.all(np.diff(np.stack([step.damage for step in result.steps]), axis=0) >= -1.0e-14)
    assert np.allclose(result.final.correction_displacement, elastic.displacement, atol=2.0e-16)


def test_staggered_acceptance_reassembles_the_post_damage_state() -> None:
    mesh = _square_annulus_mesh()
    sigma_inf = np.asarray([[-1.0, 0.0], [0.0, -0.1]])
    material = AT2Material(
        young_modulus=100.0,
        poisson_ratio=0.25,
        fracture_toughness=1.0e-4,
        length_scale=0.5,
        residual_stiffness=1.0e-8,
    )
    permissive_change_options = FractureSolverOptions(
        max_staggered_iterations=1,
        equilibrium_tolerance=1.0e-10,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=2.0,
        raise_on_nonconvergence=False,
    )

    result = solve_at2_fracture(
        mesh,
        material,
        sigma_inf,
        load_path=AT2LoadPath((1.0,)),
        options=permissive_change_options,
    )
    step = result.final
    assert np.max(step.damage) > 0.8
    assert not result.converged
    assert not step.converged
    assert np.isinf(step.energy_change)
    assert step.equilibrium_residual > permissive_change_options.equilibrium_tolerance

    recomputed = _evaluate_fixed_damage_displacement_state(
        mesh,
        material,
        sigma_inf,
        load_parameter=step.load_parameter,
        damage=step.damage,
        displacement=step.displacement,
        options=permissive_change_options,
    )
    assert step.equilibrium_residual == pytest.approx(
        recomputed.equilibrium_residual, rel=2.0e-14, abs=1.0e-15
    )
    assert np.allclose(step.stress, recomputed.stress, rtol=2.0e-14, atol=2.0e-15)
    assert step.elastic_energy == pytest.approx(recomputed.elastic_energy, rel=2.0e-14)
    assert step.external_work == pytest.approx(recomputed.external_work, rel=2.0e-14)
    assert step.total_potential_energy == pytest.approx(
        recomputed.elastic_energy + step.fracture_energy - recomputed.external_work,
        rel=2.0e-14,
    )

    strict_options = FractureSolverOptions(
        max_staggered_iterations=1,
        equilibrium_tolerance=1.0e-10,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=2.0,
    )
    with pytest.raises(RuntimeError, match="AT2 staggered solve did not converge"):
        solve_at2_fracture(
            mesh,
            material,
            sigma_inf,
            load_path=AT2LoadPath((1.0,)),
            options=strict_options,
        )

    continuing_options = FractureSolverOptions(
        max_staggered_iterations=20,
        equilibrium_tolerance=1.0e-10,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=1.0e-9,
    )
    continued = solve_at2_fracture(
        mesh,
        material,
        sigma_inf,
        load_path=AT2LoadPath((1.0,)),
        options=continuing_options,
    ).final
    assert continued.converged
    assert continued.staggered_iterations > 1
    assert continued.equilibrium_residual <= continuing_options.equilibrium_tolerance


def test_potential_energy_gate_is_independent_for_legacy_and_scheduled_p1() -> None:
    mesh = _square_annulus_tunnel_mesh()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p1", 1.0, mesh)
    stress = schedule.state_at(1.0).farfield_stress_tension_positive_yz
    material = AT2Material(
        young_modulus=100.0,
        poisson_ratio=0.25,
        fracture_toughness=1.0e-4,
        length_scale=0.5,
        residual_stiffness=1.0e-8,
    )
    blocked_options = FractureSolverOptions(
        max_staggered_iterations=2,
        equilibrium_tolerance=1.0e-4,
        kkt_tolerance=1.0e-10,
        staggered_tolerance=2.0,
        energy_tolerance=1.0e-8,
        raise_on_nonconvergence=False,
    )

    legacy = solve_at2_fracture(
        mesh,
        material,
        stress,
        load_path=AT2LoadPath((1.0,)),
        options=blocked_options,
    ).final
    scheduled = solve_at2_fracture_schedule(
        mesh,
        material,
        schedule,
        load_path=AT2LoadPath((1.0,)),
        options=blocked_options,
    ).final

    for step in (legacy, scheduled):
        assert not step.converged
        assert step.displacement_change <= blocked_options.staggered_tolerance
        assert step.damage_change <= blocked_options.staggered_tolerance
        assert step.equilibrium_residual <= blocked_options.equilibrium_tolerance
        assert step.kkt_residual <= blocked_options.kkt_tolerance
        assert np.isfinite(step.energy_change)
        assert step.energy_change > blocked_options.energy_tolerance
    assert scheduled.energy_change == legacy.energy_change

    raising_options = replace(blocked_options, raise_on_nonconvergence=True)
    with pytest.raises(RuntimeError, match=r"potential_energy_change=[0-9.e+-]+"):
        solve_at2_fracture(
            mesh,
            material,
            stress,
            load_path=AT2LoadPath((1.0,)),
            options=raising_options,
        )
    with pytest.raises(RuntimeError, match=r"potential_energy_change=[0-9.e+-]+"):
        solve_at2_fracture_schedule(
            mesh,
            material,
            schedule,
            load_path=AT2LoadPath((1.0,)),
            options=raising_options,
        )

    continuing_options = replace(
        blocked_options,
        max_staggered_iterations=3,
        raise_on_nonconvergence=True,
    )
    legacy_continued = solve_at2_fracture(
        mesh,
        material,
        stress,
        load_path=AT2LoadPath((1.0,)),
        options=continuing_options,
    ).final
    scheduled_continued = solve_at2_fracture_schedule(
        mesh,
        material,
        schedule,
        load_path=AT2LoadPath((1.0,)),
        options=continuing_options,
    ).final
    for step in (legacy_continued, scheduled_continued):
        assert step.converged
        assert step.staggered_iterations == 3
        assert step.energy_change <= continuing_options.energy_tolerance
    assert scheduled_continued.energy_change == legacy_continued.energy_change


def test_energy_change_uses_total_potential_with_instantaneous_neumann_term() -> None:
    mesh = _square_annulus_mesh()
    stress = np.asarray([[-1.0, 0.0], [0.0, -0.1]])
    material = AT2Material(100.0, 0.25, 1.0e-4, 0.5, residual_stiffness=1.0e-8)
    base_options = FractureSolverOptions(
        equilibrium_tolerance=2.0,
        kkt_tolerance=1.0,
        staggered_tolerance=2.0,
        energy_tolerance=1.0e-30,
        raise_on_nonconvergence=False,
    )
    first = solve_at2_fracture(
        mesh,
        material,
        stress,
        load_path=AT2LoadPath((0.5,)),
        options=replace(base_options, max_staggered_iterations=1),
    ).final
    second = solve_at2_fracture(
        mesh,
        material,
        stress,
        load_path=AT2LoadPath((0.5,)),
        options=replace(base_options, max_staggered_iterations=2),
    ).final

    expected = 2.0 * abs(second.total_potential_energy - first.total_potential_energy)
    expected /= abs(second.total_potential_energy) + abs(first.total_potential_energy)
    internal_first = first.elastic_energy + first.fracture_energy
    internal_second = second.elastic_energy + second.fracture_energy
    internal_only_change = 2.0 * abs(internal_second - internal_first)
    internal_only_change /= abs(internal_second) + abs(internal_first)

    assert first.external_work != 0.0
    assert np.isinf(first.energy_change)
    assert second.energy_change == pytest.approx(expected, rel=2.0e-14)
    assert second.energy_change != pytest.approx(internal_only_change, rel=1.0e-6)


def test_p1_load_state_is_bitwise_identical_to_legacy_uniform_release() -> None:
    mesh = _square_annulus_tunnel_mesh()
    material = _intact_material()
    options = _strict_options()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p1", 1.0, mesh)

    for parameter in (0.0, 0.25, 0.5, 0.75, 1.0):
        state = schedule.state_at(parameter)
        legacy = solve_fixed_damage_displacement(
            mesh,
            material,
            state.farfield_stress_tension_positive_yz,
            load_parameter=parameter,
            damage=0.0,
            options=options,
        )
        scheduled = solve_fixed_damage_displacement_at_load_state(
            mesh, material, schedule, state, damage=0.0, options=options
        )
        for field in (
            "displacement",
            "correction_displacement",
            "strain",
            "stress",
            "psi_positive",
            "psi_negative",
            "internal_force",
            "wall_nodal_force",
            "dirichlet_dofs",
            "farfield_prescribed_displacement",
        ):
            assert np.array_equal(getattr(scheduled, field), getattr(legacy, field)), field
        for field in (
            "elastic_energy",
            "external_work",
            "residual_norm",
            "equilibrium_residual",
            "iterations",
            "converged",
        ):
            assert getattr(scheduled, field) == getattr(legacy, field), field
        assert scheduled.neumann_load_functional == scheduled.external_work


def test_p1_scheduled_trajectory_is_bitwise_identical_to_legacy_trajectory() -> None:
    mesh = _square_annulus_tunnel_mesh()
    material = AT2Material(100.0, 0.25, 100.0, 0.5, residual_stiffness=0.0)
    options = _strict_options()
    path = AT2LoadPath((0.0, 0.5, 1.0))
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p1", 1.0, mesh)
    state = schedule.state_at(0.0)
    legacy = solve_at2_fracture(
        mesh,
        material,
        state.farfield_stress_tension_positive_yz,
        load_path=path,
        options=options,
    )
    scheduled = solve_at2_fracture_schedule(
        mesh, material, schedule, load_path=path, options=options
    )

    assert legacy.converged and scheduled.converged
    for scheduled_step, legacy_step in zip(scheduled.steps, legacy.steps, strict=True):
        for field in (
            "displacement",
            "correction_displacement",
            "damage",
            "strain",
            "stress",
            "psi_positive",
            "psi_negative",
            "history",
            "internal_force",
            "wall_nodal_force",
            "dirichlet_dofs",
            "farfield_prescribed_displacement",
        ):
            assert np.array_equal(getattr(scheduled_step, field), getattr(legacy_step, field)), (
                field
            )
        for field in (
            "load_parameter",
            "elastic_energy",
            "fracture_energy",
            "external_work",
            "total_potential_energy",
            "equilibrium_residual",
            "kkt_residual",
            "complementarity_residual",
            "irreversibility_violation",
            "range_violation",
            "displacement_change",
            "damage_change",
            "energy_change",
            "staggered_iterations",
            "displacement_iterations",
            "damage_iterations",
            "converged",
        ):
            assert getattr(scheduled_step, field) == getattr(legacy_step, field), field


@pytest.mark.parametrize("path_id, parameter", [("p2", 0.5), ("p3", 0.75)])
def test_p2_p3_use_current_farfield_stress_and_uniform_release(
    path_id: str, parameter: float
) -> None:
    mesh = _square_annulus_tunnel_mesh()
    material = _intact_material()
    options = _strict_options()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), path_id, 1.0, mesh)
    initial_state = schedule.state_at(0.0)
    current_state = schedule.state_at(parameter)
    scheduled = solve_fixed_damage_displacement_at_load_state(
        mesh, material, schedule, current_state, damage=0.0, options=options
    )
    current_stress_legacy = solve_fixed_damage_displacement(
        mesh,
        material,
        current_state.farfield_stress_tension_positive_yz,
        load_parameter=float(current_state.wall_release[0]),
        damage=0.0,
        options=options,
    )

    assert not np.array_equal(
        current_state.farfield_stress_tension_positive_yz,
        initial_state.farfield_stress_tension_positive_yz,
    )
    assert np.array_equal(
        scheduled.farfield_prescribed_displacement,
        current_stress_legacy.farfield_prescribed_displacement,
    )
    assert np.array_equal(scheduled.displacement, current_stress_legacy.displacement)
    assert np.array_equal(scheduled.wall_nodal_force, current_stress_legacy.wall_nodal_force)
    if path_id == "p3":
        assert current_state.farfield_stress_tension_positive_yz[0, 1] != 0.0


def test_scheduled_trajectory_carries_previous_correction_to_current_affine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _square_annulus_tunnel_mesh()
    material = _intact_material()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p2", 1.0, mesh)
    captured_first_correction: dict[float, np.ndarray] = {}
    original = fracture_module.solve_fixed_damage_displacement_at_load_state

    def capture_initial_correction(*args: object, **kwargs: object) -> object:
        load_state = args[3]
        assert hasattr(load_state, "s")
        parameter = float(load_state.s)  # type: ignore[attr-defined]
        correction = np.asarray(kwargs["initial_correction_displacement"])
        captured_first_correction.setdefault(parameter, correction.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(
        fracture_module,
        "solve_fixed_damage_displacement_at_load_state",
        capture_initial_correction,
    )
    result = solve_at2_fracture_schedule(
        mesh,
        material,
        schedule,
        load_path=AT2LoadPath((0.0, 0.5)),
        options=_strict_options(),
    )

    assert result.converged
    assert np.array_equal(captured_first_correction[0.0], np.zeros_like(mesh.nodes))
    assert np.array_equal(captured_first_correction[0.5], result.steps[0].correction_displacement)
    assert not np.array_equal(
        result.steps[0].farfield_prescribed_displacement,
        result.steps[1].farfield_prescribed_displacement,
    )


def test_p4_nonuniform_release_differs_from_same_mean_and_aligns_by_facet_id() -> None:
    mesh = _square_annulus_tunnel_mesh()
    material = _intact_material()
    options = _strict_options()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p4", 1.0, mesh)
    state = schedule.state_at(0.375)
    mean_release = float(np.mean(state.wall_release))
    nonuniform = solve_fixed_damage_displacement_at_load_state(
        mesh, material, schedule, state, damage=0.0, options=options
    )
    uniform = solve_fixed_damage_displacement(
        mesh,
        material,
        state.farfield_stress_tension_positive_yz,
        load_parameter=mean_release,
        damage=0.0,
        options=options,
    )

    assert not np.allclose(nonuniform.wall_nodal_force, uniform.wall_nodal_force)
    assert not np.allclose(nonuniform.displacement, uniform.displacement)

    reverse = np.arange(state.wall_facet_ids.size - 1, -1, -1)
    reordered_state = replace(
        state,
        wall_facet_ids=state.wall_facet_ids[reverse],
        wall_zone_weights=state.wall_zone_weights[reverse],
        wall_release=state.wall_release[reverse],
    )
    reordered = solve_fixed_damage_displacement_at_load_state(
        mesh, material, schedule, reordered_state, damage=0.0, options=options
    )
    assert np.array_equal(reordered.wall_nodal_force, nonuniform.wall_nodal_force)
    assert np.array_equal(reordered.displacement, nonuniform.displacement)

    missing_facet_state = replace(
        state,
        wall_facet_ids=state.wall_facet_ids[:-1],
        wall_zone_weights=state.wall_zone_weights[:-1],
        wall_release=state.wall_release[:-1],
    )
    with pytest.raises(ValueError, match="inconsistent with its load schedule"):
        solve_fixed_damage_displacement_at_load_state(
            mesh, material, schedule, missing_facet_state, damage=0.0, options=options
        )


def test_load_schedule_rejects_same_facet_ids_from_different_wall_geometry() -> None:
    mesh = _square_annulus_tunnel_mesh()
    foreign_nodes = mesh.nodes.copy()
    foreign_nodes[[4, 7], 0] *= 0.7
    foreign_nodes[[5, 6], 0] *= 1.3
    foreign_mesh = replace(mesh, nodes=foreign_nodes)
    foreign_schedule = compile_phase1_load_schedule(
        load_fracture_phase1_config(), "p4", 1.0, foreign_mesh
    )
    foreign_state = foreign_schedule.state_at(0.375)

    assert np.array_equal(foreign_schedule.wall_facet_ids, mesh.boundary_facets[WALL])
    with pytest.raises(ValueError, match="wall facet endpoints do not match"):
        solve_fixed_damage_displacement_at_load_state(
            mesh,
            _intact_material(),
            foreign_schedule,
            foreign_state,
            damage=0.0,
            options=_strict_options(),
        )


def test_load_schedule_rejects_changed_wall_endpoints_with_same_midpoints() -> None:
    mesh = _square_annulus_tunnel_mesh()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p4", 1.0, mesh)
    state = schedule.state_at(0.375)
    changed = mesh.nodes.copy()
    wall_facet_id = int(mesh.boundary_facets[WALL][0])
    edge_nodes = np.asarray(mesh.mesh.facets)[:, wall_facet_id]
    direction = changed[edge_nodes[1]] - changed[edge_nodes[0]]
    delta = 0.1 * direction
    changed[edge_nodes[0]] -= delta
    changed[edge_nodes[1]] += delta
    changed_mesh_object = mesh.mesh.copy()
    changed_mesh_object.p[:, :] = changed.T
    changed_mesh = replace(mesh, mesh=changed_mesh_object, nodes=changed)

    original_midpoint = np.asarray(schedule.wall_facet_midpoints_yz)[0]
    changed_midpoint = changed[edge_nodes].mean(axis=0)
    assert changed_midpoint == pytest.approx(original_midpoint)
    with pytest.raises(ValueError, match="wall facet endpoints do not match"):
        solve_fixed_damage_displacement_at_load_state(
            changed_mesh,
            _intact_material(),
            schedule,
            state,
            damage=0.0,
            options=_strict_options(),
        )


def test_scheduled_final_forces_are_reassembled_from_the_same_u_d_state() -> None:
    mesh = _square_annulus_tunnel_mesh()
    material = AT2Material(100.0, 0.25, 100.0, 0.5, residual_stiffness=0.0)
    options = _strict_options()
    schedule = compile_phase1_load_schedule(load_fracture_phase1_config(), "p3", 1.0, mesh)
    result = solve_at2_fracture_schedule(
        mesh,
        material,
        schedule,
        load_path=AT2LoadPath((0.0, 0.5, 1.0)),
        options=options,
    )
    step = result.final
    recomputed = _evaluate_fixed_damage_displacement_at_load_state(
        mesh,
        material,
        schedule,
        step.load_state,
        damage=step.damage,
        displacement=step.displacement,
        options=options,
    )
    for field in (
        "internal_force",
        "wall_nodal_force",
        "dirichlet_dofs",
        "farfield_prescribed_displacement",
        "strain",
        "stress",
    ):
        assert np.array_equal(getattr(step, field), getattr(recomputed, field)), field
    boundary_state = BoundaryEquilibriumState(
        displacement=step.displacement,
        internal_force=step.internal_force,
        wall_nodal_force=step.wall_nodal_force,
        dirichlet_dofs=step.dirichlet_dofs,
        farfield_prescribed_displacement=step.farfield_prescribed_displacement,
    )
    assert np.linalg.norm(boundary_state.free_residual) == pytest.approx(
        recomputed.residual_norm, rel=2.0e-14, abs=1.0e-15
    )
    assert step.neumann_load_functional == step.external_work
