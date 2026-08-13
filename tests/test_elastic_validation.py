from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gmsh")
pytest.importorskip("skfem")

from tunnelgeopt.elastic_validation import (
    kirsch_metrics,
    stress_vectors_to_matrices,
    tensor_frobenius_relative_l2,
    validate_elastic_result,
)
from tunnelgeopt.elasticity import solve_plane_strain_excavation
from tunnelgeopt.geometry import make_tunnel_boundary
from tunnelgeopt.mesh import generate_tunnel_mesh


def test_tensor_metric_counts_symmetric_shear_twice() -> None:
    target = np.asarray([[1.0, 2.0, 3.0]])
    prediction = np.asarray([[1.0, 2.0, 4.0]])
    expected = np.sqrt(2.0 / (1.0 + 4.0 + 18.0))
    assert tensor_frobenius_relative_l2(prediction, target) == pytest.approx(expected)
    matrices = stress_vectors_to_matrices(target)
    assert np.array_equal(matrices, np.asarray([[[1.0, 3.0], [3.0, 2.0]]]))


def test_kirsch_and_generic_metrics_are_finite_and_bounded() -> None:
    geometry = make_tunnel_boundary("circle", n_points=96, radius=1.0)
    mesh = generate_tunnel_mesh(
        geometry,
        domain_scale=8.0,
        mesh_size=0.6,
        wall_mesh_size=0.125,
        farfield_mesh_size=0.6,
    )
    result = solve_plane_strain_excavation(
        mesh,
        young_modulus=100.0,
        poisson_ratio=0.25,
        sigma_inf=np.asarray([[1.0, 0.0], [0.0, 0.0]]),
    )
    metrics = kirsch_metrics(result, mesh, radius=1.0)
    assert metrics.annulus_element_count > 500
    assert metrics.annulus_stress_relative_l2 < 0.08
    # Cellwise P1 stress is discontinuous at the polygonal wall.  This is a
    # coarse regression guard; the metric remains reported rather than being
    # promoted to the preregistered Kirsch field-error gate.
    assert metrics.wall_traction_relative_l2 < 0.15
    assert metrics.peak_hoop_relative_error < 0.10

    validation = validate_elastic_result(
        result,
        max_symmetry_error=1e-12,
        max_algebraic_residual=1e-9,
        max_energy_closure=1e-9,
    )
    assert validation["passed"] is True
    assert validation["not_applicable"] == [
        "damage_irreversibility",
        "negative_dissipation",
        "joint_penetration",
        "cfl",
    ]
