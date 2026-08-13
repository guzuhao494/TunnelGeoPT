"""Physics validation helpers for the B-elastic tunnel solver.

The functions in this module report numerical evidence; they do not relabel a
linear-elastic calculation as damage, fracture, or rockburst physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elasticity import ElasticResult, compute_element_strain, plane_strain_stress
from .kirsch import kirsch_stress
from .mesh import WALL, TunnelMesh

FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class KirschMetrics:
    """Area- and edge-weighted diagnostics for one circular-opening solve."""

    annulus_stress_relative_l2: float
    wall_traction_relative_l2: float
    peak_hoop_stress: float
    analytical_peak_hoop_stress: float
    peak_hoop_relative_error: float
    annulus_element_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "annulus_stress_relative_l2": self.annulus_stress_relative_l2,
            "wall_traction_relative_l2": self.wall_traction_relative_l2,
            "peak_hoop_stress": self.peak_hoop_stress,
            "analytical_peak_hoop_stress": self.analytical_peak_hoop_stress,
            "peak_hoop_relative_error": self.peak_hoop_relative_error,
            "annulus_element_count": self.annulus_element_count,
        }


def run_affine_patch_test(
    *,
    young_modulus: float = 100.0,
    poisson_ratio: float = 0.25,
    refinement: int = 3,
) -> dict[str, float | int | bool]:
    """Solve an affine displacement patch and report stress recovery error."""

    if refinement < 1:
        raise ValueError("refinement must be at least one")
    try:
        from skfem import Basis, ElementTriP1, ElementVector, MeshTri, asm, condense, solve
        from skfem.models.elasticity import linear_elasticity
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("patch validation requires scikit-fem and SciPy") from exc

    grid = np.linspace(-1.0, 1.0, 2**refinement + 1)
    mesh = MeshTri.init_tensor(grid, grid)
    basis = Basis(mesh, ElementVector(ElementTriP1()))
    from .elasticity import plane_strain_lame_parameters

    lame_lambda, shear_modulus = plane_strain_lame_parameters(young_modulus, poisson_ratio)
    stiffness = asm(linear_elasticity(Lambda=lame_lambda, Mu=shear_modulus), basis).tocsr()
    gradient = np.asarray([[0.017, -0.011], [0.023, 0.009]], dtype=np.float64)
    translation = np.asarray([0.31, -0.27], dtype=np.float64)
    nodal_displacement = np.asarray(mesh.p.T, dtype=np.float64) @ gradient.T + translation
    full = np.zeros(stiffness.shape[0], dtype=np.float64)
    nodal_dofs = np.asarray(basis.nodal_dofs, dtype=np.int64)
    full[nodal_dofs] = nodal_displacement.T
    boundary_dofs = np.asarray(basis.get_dofs().flatten(), dtype=np.int64)
    solution = np.asarray(
        solve(*condense(stiffness, np.zeros(stiffness.shape[0]), x=full, D=boundary_dofs)),
        dtype=np.float64,
    )
    recovered_u = solution[nodal_dofs].T
    strain, _ = compute_element_strain(mesh.p.T, mesh.t.T, recovered_u)
    expected_strain = np.asarray(
        [gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]],
        dtype=np.float64,
    )
    target_stress = plane_strain_stress(
        expected_strain, young_modulus=young_modulus, poisson_ratio=poisson_ratio
    )
    recovered_stress = plane_strain_stress(
        strain, young_modulus=young_modulus, poisson_ratio=poisson_ratio
    )
    stress_error = float(
        np.linalg.norm(recovered_stress - target_stress[None, :])
        / np.linalg.norm(np.broadcast_to(target_stress, recovered_stress.shape))
    )
    free = np.setdiff1d(np.arange(stiffness.shape[0]), boundary_dofs)
    # For a prescribed-displacement patch, the free-DOF right-hand side is
    # the condensed boundary contribution ``-K_fc u_c``.  Normalizing the
    # cancellation error by an absolute ``1.0`` makes the check depend on the
    # units and Young's-modulus scale (for example Pa versus MPa).  Compare the
    # two condensed equilibrium terms instead so the reported residual is
    # dimensionless and scale invariant.
    free_lhs = np.asarray(stiffness[free][:, free] @ solution[free], dtype=np.float64)
    free_rhs = np.asarray(
        -(stiffness[free][:, boundary_dofs] @ solution[boundary_dofs]),
        dtype=np.float64,
    )
    free_residual_norm = float(np.linalg.norm(free_lhs - free_rhs))
    free_residual_scale = max(
        float(np.linalg.norm(free_lhs)) + float(np.linalg.norm(free_rhs)),
        np.finfo(float).tiny,
    )
    free_residual = free_residual_norm / free_residual_scale
    return {
        "passed": stress_error <= 1e-9 and free_residual <= 1e-9,
        "stress_relative_l2": stress_error,
        "free_dof_residual": free_residual,
        "node_count": int(mesh.p.shape[1]),
        "element_count": int(mesh.t.shape[1]),
    }


def stress_vectors_to_matrices(stress: ArrayLike) -> FloatArray:
    """Convert ``[..., yy, zz, yz]`` to symmetric ``[..., 2, 2]`` matrices."""

    values = np.asarray(stress, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError("stress must end in [yy, zz, yz]")
    if not np.isfinite(values).all():
        raise ValueError("stress contains a non-finite value")
    matrices = np.empty((*values.shape[:-1], 2, 2), dtype=np.float64)
    matrices[..., 0, 0] = values[..., 0]
    matrices[..., 1, 1] = values[..., 1]
    matrices[..., 0, 1] = values[..., 2]
    matrices[..., 1, 0] = values[..., 2]
    return matrices


def tensor_frobenius_relative_l2(
    prediction: ArrayLike,
    target: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> float:
    """Relative L2 for symmetric 2-D stresses with the shear term counted twice."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape or prediction_array.shape[-1] != 3:
        raise ValueError("prediction and target must have equal shape [..., 3]")
    if not np.isfinite(prediction_array).all() or not np.isfinite(target_array).all():
        raise ValueError("prediction and target must be finite")
    flat_prediction = prediction_array.reshape(-1, 3)
    flat_target = target_array.reshape(-1, 3)
    if weights is None:
        weight_array = np.ones(flat_target.shape[0], dtype=np.float64)
    else:
        weight_array = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weight_array.shape[0] != flat_target.shape[0]:
            raise ValueError("weights must have one value per stress vector")
        if not np.isfinite(weight_array).all() or np.any(weight_array < 0.0):
            raise ValueError("weights must be finite and non-negative")
    multiplier = np.asarray([1.0, 1.0, 2.0], dtype=np.float64)
    numerator = np.sum(weight_array[:, None] * multiplier * (flat_prediction - flat_target) ** 2)
    denominator = np.sum(weight_array[:, None] * multiplier * flat_target**2)
    if denominator <= np.finfo(float).tiny:
        raise ValueError("target norm is zero")
    return float(np.sqrt(numerator / denominator))


def _outward_facet_normals(
    tunnel_mesh: TunnelMesh, facets: NDArray[np.integer]
) -> tuple[FloatArray, FloatArray, NDArray[np.integer]]:
    edges = np.asarray(tunnel_mesh.mesh.facets[:, facets].T, dtype=np.int64)
    start = tunnel_mesh.nodes[edges[:, 0]]
    end = tunnel_mesh.nodes[edges[:, 1]]
    edge_vector = end - start
    length = np.linalg.norm(edge_vector, axis=1)
    if np.any(length <= 0.0):
        raise ValueError("boundary contains a zero-length facet")
    normal = np.column_stack([edge_vector[:, 1], -edge_vector[:, 0]]) / length[:, None]
    cells = np.asarray(tunnel_mesh.mesh.f2t[0, facets], dtype=np.int64)
    centers = tunnel_mesh.nodes[tunnel_mesh.elements[cells]].mean(axis=1)
    midpoint = 0.5 * (start + end)
    points_into_cell = np.sum(normal * (centers - midpoint), axis=1) > 0.0
    normal[points_into_cell] *= -1.0
    return normal, length, cells


def wall_traction_relative_l2(result: ElasticResult, tunnel_mesh: TunnelMesh) -> float:
    """Compute the total-stress traction residual along the cavity wall."""

    facets = np.asarray(tunnel_mesh.boundary_facets[WALL], dtype=np.int64)
    normal, length, cells = _outward_facet_normals(tunnel_mesh, facets)
    stress = stress_vectors_to_matrices(result.total_stress[cells])
    traction = np.einsum("nij,nj->ni", stress, normal)
    numerator = np.sum(length * np.sum(traction**2, axis=1))
    reference = float(np.linalg.norm(result.sigma_inf, ord="fro"))
    denominator = reference**2 * float(np.sum(length))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("sigma_inf norm is zero")
    return float(np.sqrt(numerator / denominator))


def kirsch_metrics(
    result: ElasticResult,
    tunnel_mesh: TunnelMesh,
    *,
    radius: float,
    annulus: tuple[float, float] = (1.25, 3.0),
) -> KirschMetrics:
    """Compare one circular-opening result to the analytical Kirsch solution."""

    if radius <= 0.0 or not 1.0 < annulus[0] < annulus[1]:
        raise ValueError("radius and annulus bounds are invalid")
    centers = np.asarray(result.element_centers, dtype=np.float64)
    radial_distance = np.linalg.norm(centers, axis=1)
    selected = (radial_distance >= annulus[0] * radius) & (radial_distance <= annulus[1] * radius)
    if not np.any(selected):
        raise ValueError("comparison annulus contains no elements")
    analytical = kirsch_stress(
        centers[selected, 0],
        centers[selected, 1],
        radius=radius,
        sigma_x=float(result.sigma_inf[0, 0]),
        sigma_y=float(result.sigma_inf[1, 1]),
        tau_xy=float(result.sigma_inf[0, 1]),
        return_cartesian=True,
    )
    target = np.column_stack([analytical["sigma_xx"], analytical["sigma_yy"], analytical["tau_xy"]])
    field_error = tensor_frobenius_relative_l2(
        result.total_stress[selected], target, weights=result.element_area[selected]
    )

    wall_facets = np.asarray(tunnel_mesh.boundary_facets[WALL], dtype=np.int64)
    normal, _, cells = _outward_facet_normals(tunnel_mesh, wall_facets)
    tangent = np.column_stack([-normal[:, 1], normal[:, 0]])
    matrices = stress_vectors_to_matrices(result.total_stress[cells])
    hoop = np.einsum("ni,nij,nj->n", tangent, matrices, tangent)
    numerical_peak = float(np.max(np.abs(hoop)))
    theta = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    exact_wall = kirsch_stress(
        radius * np.cos(theta),
        radius * np.sin(theta),
        radius=radius,
        sigma_x=float(result.sigma_inf[0, 0]),
        sigma_y=float(result.sigma_inf[1, 1]),
        tau_xy=float(result.sigma_inf[0, 1]),
    )
    analytical_peak = float(np.max(np.abs(exact_wall["sigma_tt"])))
    if analytical_peak <= np.finfo(float).tiny:
        raise ValueError("analytical peak hoop stress is zero")
    peak_error = abs(numerical_peak - analytical_peak) / analytical_peak
    return KirschMetrics(
        annulus_stress_relative_l2=field_error,
        wall_traction_relative_l2=wall_traction_relative_l2(result, tunnel_mesh),
        peak_hoop_stress=numerical_peak,
        analytical_peak_hoop_stress=analytical_peak,
        peak_hoop_relative_error=float(peak_error),
        annulus_element_count=int(np.count_nonzero(selected)),
    )


def validate_elastic_result(
    result: ElasticResult,
    *,
    max_symmetry_error: float,
    max_algebraic_residual: float,
    max_energy_closure: float,
) -> dict[str, Any]:
    """Apply only solver-generic, preregistered static-elastic checks."""

    arrays = (
        result.nodes,
        result.displacement,
        result.strain,
        result.total_stress,
        result.sigma_xx,
        result.energy_density,
        result.element_area,
    )
    nonfinite_fraction = float(
        sum(np.size(array) - np.count_nonzero(np.isfinite(array)) for array in arrays)
        / sum(np.size(array) for array in arrays)
    )
    checks = {
        "finite": nonfinite_fraction == 0.0,
        "matrix_symmetry": result.stiffness_symmetry_error <= max_symmetry_error,
        "algebraic_residual": result.algebraic_residual <= max_algebraic_residual,
        "energy_closure": result.energy_closure <= max_energy_closure,
        "nonnegative_element_energy": bool(np.all(result.energy_density >= -1e-12)),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "nonfinite_fraction": nonfinite_fraction,
            "stiffness_symmetry_relative_error": result.stiffness_symmetry_error,
            "free_dof_algebraic_residual": result.algebraic_residual,
            "clapeyron_relative_error": result.energy_closure,
            "energy_discretization_relative_error": result.energy_discretization_error,
        },
        "not_applicable": [
            "damage_irreversibility",
            "negative_dissipation",
            "joint_penetration",
            "cfl",
        ],
    }


__all__ = [
    "KirschMetrics",
    "kirsch_metrics",
    "run_affine_patch_test",
    "stress_vectors_to_matrices",
    "tensor_frobenius_relative_l2",
    "validate_elastic_result",
    "wall_traction_relative_l2",
]
