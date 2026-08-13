from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")
skfem = pytest.importorskip("skfem")

from tunnelgeopt.elasticity import (
    plane_strain_sigma_xx,
    plane_strain_stress,
    solve_plane_strain_excavation,
)
from tunnelgeopt.fracture import (
    AT2LoadPath,
    AT2Material,
    FractureSolverOptions,
    _evaluate_fixed_damage_displacement_state,
    assemble_at2_damage_system,
    degradation,
    miehe_spectral_response,
    plane_strain_spectral_split,
    solve_at2_damage,
    solve_at2_fracture,
    solve_fixed_damage_displacement,
    update_history,
)
from tunnelgeopt.mesh import FARFIELD, WALL


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
