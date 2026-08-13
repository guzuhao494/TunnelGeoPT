from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

from tunnelgeopt.elasticity import (
    assemble_plane_strain_system,
    compute_element_strain,
    plane_strain_lame_parameters,
    plane_strain_stress,
    solve_plane_strain_excavation,
)
from tunnelgeopt.geometry import make_tunnel_boundary
from tunnelgeopt.kirsch import kirsch_stress
from tunnelgeopt.mesh import FARFIELD, WALL, TunnelMesh, generate_tunnel_mesh


@pytest.fixture(scope="module")
def kirsch_meshes() -> dict[str, TunnelMesh]:
    geometry = make_tunnel_boundary("circle", n_points=128, radius=1.0)
    tiers = {
        "coarse": {"wall_mesh_size": 0.25, "farfield_mesh_size": 0.80},
        "medium": {"wall_mesh_size": 0.125, "farfield_mesh_size": 0.60},
        "fine": {"wall_mesh_size": 0.0625, "farfield_mesh_size": 0.40},
    }
    return {
        name: generate_tunnel_mesh(
            geometry,
            domain_scale=8.0,
            mesh_size=settings["farfield_mesh_size"],
            **settings,
        )
        for name, settings in tiers.items()
    }


@pytest.fixture(scope="module")
def circular_mesh(kirsch_meshes: dict[str, TunnelMesh]) -> TunnelMesh:
    return kirsch_meshes["medium"]


def test_affine_patch_strain_and_lambda_zero_constitutive_consistency() -> None:
    nodes = np.asarray([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    elements = np.asarray([[0, 1, 2], [0, 2, 3]])
    gradient = np.asarray([[0.03, -0.02], [0.04, 0.01]])
    translation = np.asarray([3.0, -7.0])
    displacement = nodes @ gradient.T + translation

    strain, area = compute_element_strain(nodes, elements, displacement)
    expected_strain = np.asarray([0.03, 0.01, 0.02])
    assert np.allclose(strain, expected_strain[None, :], rtol=0.0, atol=2e-15)
    assert np.allclose(area, 1.0)
    shifted_strain, shifted_area = compute_element_strain(nodes + 1.0e8, elements, displacement)
    assert np.allclose(shifted_strain, expected_strain[None, :], rtol=0.0, atol=2e-15)
    assert np.array_equal(shifted_area, area)

    lame_lambda, shear_modulus = plane_strain_lame_parameters(100.0, 0.0)
    stress = plane_strain_stress(strain, young_modulus=100.0, poisson_ratio=0.0)
    assert lame_lambda == 0.0
    assert shear_modulus == 50.0
    assert np.allclose(
        stress,
        np.asarray([[3.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
        rtol=0.0,
        atol=2e-13,
    )


def test_stiffness_is_symmetric_finite_and_constraints_are_explicit(
    circular_mesh: TunnelMesh,
) -> None:
    system = assemble_plane_strain_system(
        circular_mesh,
        young_modulus=30.0e9,
        poisson_ratio=0.24,
        sigma_inf=np.asarray([[-12.0e6, 1.5e6], [1.5e6, -7.0e6]]),
    )

    assert system.stiffness.shape[0] == 2 * circular_mesh.nodes.shape[0]
    assert np.isfinite(system.stiffness.data).all()
    assert np.isfinite(system.load).all()
    assert system.stiffness_symmetry_error < 1e-14
    assert system.dirichlet_dofs.size > 0
    assert system.free_dofs.size > 0
    assert np.intersect1d(system.dirichlet_dofs, system.free_dofs).size == 0
    assert system.dirichlet_dofs.size + system.free_dofs.size == system.stiffness.shape[0]


def test_excavation_solution_contract_residual_energy_and_linear_scaling(
    circular_mesh: TunnelMesh,
) -> None:
    sigma_inf = np.asarray([[-12.0e6, 1.5e6], [1.5e6, -7.0e6]])
    result = solve_plane_strain_excavation(
        circular_mesh,
        young_modulus=30.0e9,
        poisson_ratio=0.24,
        sigma_inf=sigma_inf,
    )
    scale = 2.75
    scaled = solve_plane_strain_excavation(
        circular_mesh,
        young_modulus=30.0e9,
        poisson_ratio=0.24,
        sigma_inf=scale * sigma_inf,
    )

    node_count = circular_mesh.nodes.shape[0]
    element_count = circular_mesh.elements.shape[0]
    assert result.nodes.shape == (node_count, 2)
    assert result.elements.shape == (element_count, 3)
    assert result.u.shape == (node_count, 2)
    assert result.strain.shape == (element_count, 3)
    assert result.total_stress.shape == (element_count, 3)
    assert result.sigma_xx.shape == (element_count,)
    assert result.energy_density.shape == (element_count,)
    assert set(result.boundary_facets) == {WALL, FARFIELD}
    assert np.isfinite(result.u).all()
    assert np.isfinite(result.strain).all()
    assert np.isfinite(result.total_stress).all()
    assert np.isfinite(result.sigma_xx).all()
    assert np.isfinite(result.energy_density).all()
    assert result.energy > 0.0
    assert result.algebraic_residual < 1e-10
    assert result.energy_closure < 1e-10
    assert result.energy_discretization_error < 1e-10
    assert result.stiffness_symmetry_error < 1e-14

    sigma_vector = np.asarray([sigma_inf[0, 0], sigma_inf[1, 1], sigma_inf[0, 1]])
    assert np.allclose(result.total_stress, result.delta_stress + sigma_vector)
    assert np.allclose(scaled.u, scale * result.u, rtol=2e-11, atol=1e-16)
    assert np.allclose(scaled.strain, scale * result.strain, rtol=2e-11, atol=1e-16)
    assert np.allclose(scaled.total_stress, scale * result.total_stress, rtol=2e-11, atol=1e-5)
    assert np.allclose(scaled.sigma_xx, scale * result.sigma_xx, rtol=2e-11, atol=1e-5)
    assert scaled.energy == pytest.approx(scale**2 * result.energy, rel=2e-11)


def test_three_load_kirsch_errors_improve_monotonically_across_mesh_tiers(
    kirsch_meshes: dict[str, TunnelMesh],
) -> None:
    loads = {
        "uniaxial": (1.0, 0.0, 0.0),
        "equal_biaxial": (1.0, 1.0, 0.0),
        "pure_shear": (0.0, 0.0, 1.0),
    }
    errors = {name: [] for name in loads}
    uniaxial_fine = None

    for tier_name in ("coarse", "medium", "fine"):
        tunnel_mesh = kirsch_meshes[tier_name]
        for load_name, (sigma_y, sigma_z, tau_yz) in loads.items():
            sigma_inf = np.asarray([[sigma_y, tau_yz], [tau_yz, sigma_z]])
            result = solve_plane_strain_excavation(
                tunnel_mesh,
                young_modulus=100.0,
                poisson_ratio=0.25,
                sigma_inf=sigma_inf,
            )
            coordinates = result.element_centers
            radius = np.linalg.norm(coordinates, axis=1)
            comparison = (radius >= 1.25) & (radius <= 3.0)
            analytical = kirsch_stress(
                coordinates[comparison, 0],
                coordinates[comparison, 1],
                radius=1.0,
                sigma_x=sigma_y,
                sigma_y=sigma_z,
                tau_xy=tau_yz,
                return_cartesian=True,
            )
            target = np.column_stack(
                [
                    analytical["sigma_xx"],
                    analytical["sigma_yy"],
                    analytical["tau_xy"],
                ]
            )
            difference = result.total_stress[comparison] - target
            area = result.element_area[comparison]
            difference_sq = difference[:, 0] ** 2 + difference[:, 1] ** 2
            difference_sq += 2.0 * difference[:, 2] ** 2
            target_sq = target[:, 0] ** 2 + target[:, 1] ** 2
            target_sq += 2.0 * target[:, 2] ** 2
            relative_l2 = np.sqrt(np.sum(area * difference_sq) / np.sum(area * target_sq))
            errors[load_name].append(relative_l2)

            assert comparison.sum() > 300
            assert result.algebraic_residual < 1e-9
            assert result.energy_closure < 1e-9
            if tier_name == "fine" and load_name == "uniaxial":
                uniaxial_fine = result

    # These are configured synthetic B-elastic gates, not fracture or
    # high-fidelity rockburst evidence.  All three independent Kirsch load
    # cases must improve at both refinements and pass the fine-mesh bound.
    for load_errors in errors.values():
        assert load_errors[0] > load_errors[1] > load_errors[2]
        assert load_errors[2] < 0.08

    assert uniaxial_fine is not None
    fine_mesh = kirsch_meshes["fine"]
    adjacent = np.unique(fine_mesh.mesh.f2t[:, fine_mesh.boundary_facets[WALL]])
    adjacent = adjacent[adjacent >= 0]
    centers = uniaxial_fine.element_centers[adjacent]
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    tangent = np.column_stack([-np.sin(theta), np.cos(theta)])
    stress = uniaxial_fine.total_stress[adjacent]
    hoop = stress[:, 0] * tangent[:, 0] ** 2 + stress[:, 1] * tangent[:, 1] ** 2
    hoop += 2.0 * stress[:, 2] * tangent[:, 0] * tangent[:, 1]
    peak_stress_concentration_error = abs(float(hoop.max()) - 3.0) / 3.0
    assert peak_stress_concentration_error < 0.10
