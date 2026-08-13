"""P1 plane-strain elasticity for synthetic tunnel-excavation labels.

The solver uses an excavation-increment formulation.  A uniform in-situ
stress ``Sigma_inf`` exists before excavation.  The increment problem imposes

* zero incremental displacement on the exterior ``farfield`` boundary, and
* ``-Sigma_inf @ n`` on the ``wall`` boundary, where ``n`` is the outward
  normal of the finite-element rock domain and therefore points into the hole.

Consequently ``total_stress = Sigma_inf + delta_stress`` and the ideal cavity
wall is traction-free in the continuum problem.  Stresses are tension-positive
internally.  Compression-positive rock-mechanics inputs must be negated by the
caller and recorded in case metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .mesh import FARFIELD, ROCK, WALL, TunnelMesh

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class PlaneStrainMaterial:
    """Homogeneous isotropic material parameters."""

    young_modulus: float
    poisson_ratio: float

    def __post_init__(self) -> None:
        plane_strain_lame_parameters(self.young_modulus, self.poisson_ratio)

    @property
    def lame_lambda(self) -> float:
        return plane_strain_lame_parameters(self.young_modulus, self.poisson_ratio)[0]

    @property
    def shear_modulus(self) -> float:
        return plane_strain_lame_parameters(self.young_modulus, self.poisson_ratio)[1]


@dataclass(frozen=True)
class ElasticSystem:
    """Assembled linear system before essential-boundary condensation."""

    stiffness: Any
    load: FloatArray
    basis: Any
    dirichlet_dofs: IntArray
    free_dofs: IntArray
    material: PlaneStrainMaterial
    sigma_inf: FloatArray
    stiffness_symmetry_error: float

    @property
    def K(self) -> Any:
        return self.stiffness

    @property
    def f(self) -> FloatArray:
        return self.load


@dataclass(frozen=True)
class ElasticResult:
    """Elementwise B-layer labels and auditable numerical diagnostics."""

    nodes: FloatArray
    elements: IntArray
    displacement: FloatArray
    strain: FloatArray
    delta_stress: FloatArray
    total_stress: FloatArray
    sigma_xx: FloatArray
    energy_density: FloatArray
    element_area: FloatArray
    element_centers: FloatArray
    energy: float
    external_work: float
    algebraic_residual: float
    residual_norm: float
    energy_closure: float
    energy_discretization_error: float
    stiffness_symmetry_error: float
    boundary_facets: Mapping[str, IntArray]
    facet_markers: IntArray
    cell_markers: IntArray
    physical_tags: Mapping[str, int]
    material: Mapping[str, float]
    sigma_inf: FloatArray
    sigma_xx_inf: float
    mesh_metadata: Mapping[str, Any]

    @property
    def u(self) -> FloatArray:
        """Nodal displacement alias with shape ``[N, 2]`` in ``(y,z)`` order."""

        return self.displacement

    @property
    def stress(self) -> FloatArray:
        """Alias for total in-plane stress ``[yy, zz, yz]``."""

        return self.total_stress

    @property
    def algebraic_residual_relative(self) -> float:
        return self.algebraic_residual

    @property
    def energy_closure_error(self) -> float:
        return self.energy_closure

    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def as_dict(self) -> dict[str, Any]:
        """Return a schema-explicit view suitable for a downstream serializer."""

        return {
            "nodes": self.nodes,
            "elements": self.elements,
            "u": self.displacement,
            "strain": self.strain,
            "delta_stress": self.delta_stress,
            "total_stress": self.total_stress,
            "sigma_xx": self.sigma_xx,
            "energy_density": self.energy_density,
            "element_area": self.element_area,
            "element_centers": self.element_centers,
            "energy": self.energy,
            "external_work": self.external_work,
            "algebraic_residual": self.algebraic_residual,
            "residual_norm": self.residual_norm,
            "energy_closure": self.energy_closure,
            "energy_discretization_error": self.energy_discretization_error,
            "stiffness_symmetry_error": self.stiffness_symmetry_error,
            "boundary_facets": self.boundary_facets,
            "facet_markers": self.facet_markers,
            "cell_markers": self.cell_markers,
            "physical_tags": self.physical_tags,
            "material": self.material,
            "sigma_inf": self.sigma_inf,
            "sigma_xx_inf": self.sigma_xx_inf,
            "mesh_metadata": self.mesh_metadata,
            "strain_component_order": ("yy", "zz", "gamma_yz"),
            "stress_component_order": ("yy", "zz", "yz"),
            "sign_convention": "tension_positive",
        }


def _require_elasticity_dependencies() -> dict[str, Any]:
    try:
        from skfem import (  # type: ignore[import-not-found]
            Basis,
            ElementTriP1,
            ElementVector,
            FacetBasis,
            LinearForm,
            asm,
            condense,
            solve,
        )
        from skfem.helpers import dot  # type: ignore[import-not-found]
        from skfem.models.elasticity import (  # type: ignore[import-not-found]
            linear_elasticity,
        )
    except ImportError as exc:  # pragma: no cover - exercised only without optional stack
        raise RuntimeError("Plane-strain solving requires scikit-fem 12.x and SciPy") from exc
    return {
        "Basis": Basis,
        "ElementTriP1": ElementTriP1,
        "ElementVector": ElementVector,
        "FacetBasis": FacetBasis,
        "LinearForm": LinearForm,
        "asm": asm,
        "condense": condense,
        "solve": solve,
        "dot": dot,
        "linear_elasticity": linear_elasticity,
    }


def plane_strain_lame_parameters(young_modulus: float, poisson_ratio: float) -> tuple[float, float]:
    """Return ``(lambda, mu)`` for three-dimensional isotropic plane strain.

    ``nu = 0`` is intentionally valid; it supplies a useful lambda-zero
    constitutive consistency check.  Negative Poisson ratios are mathematically
    allowed down to ``-1`` even though the initial hard-rock sampling envelope
    is normally positive.
    """

    young_modulus = float(young_modulus)
    poisson_ratio = float(poisson_ratio)
    if not np.isfinite(young_modulus) or young_modulus <= 0.0:
        raise ValueError("young_modulus must be finite and positive")
    if not np.isfinite(poisson_ratio) or not -1.0 < poisson_ratio < 0.5:
        raise ValueError("poisson_ratio must lie strictly between -1 and 0.5")
    shear_modulus = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )
    return lame_lambda, shear_modulus


def _coerce_strain(strain: ArrayLike) -> FloatArray:
    values = np.asarray(strain, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("strain must end in [yy, zz, gamma_yz]")
    if not np.isfinite(values).all():
        raise ValueError("strain contains a non-finite value")
    return values


def plane_strain_stress(
    strain: ArrayLike, *, young_modulus: float, poisson_ratio: float
) -> FloatArray:
    """Map engineering strain ``[yy,zz,gamma_yz]`` to stress ``[yy,zz,yz]``."""

    values = _coerce_strain(strain)
    lame_lambda, shear_modulus = plane_strain_lame_parameters(young_modulus, poisson_ratio)
    trace = values[..., 0] + values[..., 1]
    stress = np.empty_like(values, dtype=np.float64)
    stress[..., 0] = lame_lambda * trace + 2.0 * shear_modulus * values[..., 0]
    stress[..., 1] = lame_lambda * trace + 2.0 * shear_modulus * values[..., 1]
    stress[..., 2] = shear_modulus * values[..., 2]
    return stress


def plane_strain_sigma_xx(
    strain: ArrayLike, *, young_modulus: float, poisson_ratio: float
) -> FloatArray:
    """Return the out-of-plane stress increment for ``epsilon_xx = 0``."""

    values = _coerce_strain(strain)
    lame_lambda, _ = plane_strain_lame_parameters(young_modulus, poisson_ratio)
    return lame_lambda * (values[..., 0] + values[..., 1])


def compute_element_strain(
    nodes: ArrayLike, elements: ArrayLike, displacement: ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Compute exact constant P1 strains and triangle areas.

    This independent reconstruction is also the affine patch-test surface; it
    avoids relying on quadrature-axis ordering in a particular scikit-fem
    release.
    """

    nodes_array = np.asarray(nodes, dtype=np.float64)
    elements_array = np.asarray(elements, dtype=np.int64)
    displacement_array = np.asarray(displacement, dtype=np.float64)
    if nodes_array.ndim != 2 or nodes_array.shape[1] != 2:
        raise ValueError("nodes must have shape [N, 2]")
    if elements_array.ndim != 2 or elements_array.shape[1] != 3:
        raise ValueError("elements must have shape [M, 3]")
    if displacement_array.shape != nodes_array.shape:
        raise ValueError("displacement must have the same [N, 2] shape as nodes")
    if not np.isfinite(nodes_array).all() or not np.isfinite(displacement_array).all():
        raise ValueError("nodes and displacement must be finite")
    if elements_array.size and (
        elements_array.min() < 0 or elements_array.max() >= nodes_array.shape[0]
    ):
        raise ValueError("elements contain an out-of-range node index")

    triangles = nodes_array[elements_array]
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    determinants = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
    scale = np.maximum(
        np.maximum(np.sum(first_edge**2, axis=1), np.sum(second_edge**2, axis=1)),
        np.finfo(float).tiny,
    )
    if np.any(np.abs(determinants) <= 1e-14 * scale):
        raise ValueError("elements contain a degenerate triangle")
    shape_gradients = np.empty((triangles.shape[0], 2, 3), dtype=np.float64)
    shape_gradients[:, 0, 0] = (triangles[:, 1, 1] - triangles[:, 2, 1]) / determinants
    shape_gradients[:, 0, 1] = (triangles[:, 2, 1] - triangles[:, 0, 1]) / determinants
    shape_gradients[:, 0, 2] = (triangles[:, 0, 1] - triangles[:, 1, 1]) / determinants
    shape_gradients[:, 1, 0] = (triangles[:, 2, 0] - triangles[:, 1, 0]) / determinants
    shape_gradients[:, 1, 1] = (triangles[:, 0, 0] - triangles[:, 2, 0]) / determinants
    shape_gradients[:, 1, 2] = (triangles[:, 1, 0] - triangles[:, 0, 0]) / determinants
    nodal_u = displacement_array[elements_array]  # [element, local_node, component]
    gradient_u = np.einsum("eni,ejn->eij", nodal_u, shape_gradients)
    strain = np.column_stack(
        [
            gradient_u[:, 0, 0],
            gradient_u[:, 1, 1],
            gradient_u[:, 0, 1] + gradient_u[:, 1, 0],
        ]
    )
    area = 0.5 * np.abs(determinants)
    return strain, area


def _coerce_sigma_inf(sigma_inf: ArrayLike) -> FloatArray:
    stress = np.asarray(sigma_inf, dtype=np.float64)
    if stress.shape == (3,):
        stress = np.asarray([[stress[0], stress[2]], [stress[2], stress[1]]], dtype=np.float64)
    if stress.shape != (2, 2):
        raise ValueError("sigma_inf must be a symmetric 2x2 matrix or [yy, zz, yz]")
    if not np.isfinite(stress).all():
        raise ValueError("sigma_inf contains a non-finite value")
    tolerance = 1e-12 * max(float(np.max(np.abs(stress))), 1.0)
    if not np.allclose(stress, stress.T, rtol=0.0, atol=tolerance):
        raise ValueError("sigma_inf must be symmetric")
    return 0.5 * (stress + stress.T)


def _mesh_payload(
    tunnel_mesh: TunnelMesh | Any,
) -> tuple[
    Any,
    FloatArray,
    IntArray,
    dict[str, IntArray],
    IntArray,
    IntArray,
    dict[str, int],
    dict[str, Any],
]:
    if isinstance(tunnel_mesh, TunnelMesh):
        return (
            tunnel_mesh.mesh,
            np.asarray(tunnel_mesh.nodes, dtype=np.float64),
            np.asarray(tunnel_mesh.elements, dtype=np.int64),
            {
                name: np.asarray(facets, dtype=np.int64)
                for name, facets in tunnel_mesh.boundary_facets.items()
            },
            np.asarray(tunnel_mesh.facet_markers, dtype=np.int32),
            np.asarray(tunnel_mesh.cell_markers, dtype=np.int32),
            dict(tunnel_mesh.physical_tags),
            dict(tunnel_mesh.metadata),
        )

    mesh = tunnel_mesh
    if not hasattr(mesh, "p") or not hasattr(mesh, "t"):
        raise TypeError("tunnel_mesh must be TunnelMesh or a scikit-fem MeshTri")
    boundaries = getattr(mesh, "boundaries", None)
    if boundaries is None or WALL not in boundaries or FARFIELD not in boundaries:
        raise ValueError("the scikit-fem mesh must have named wall and farfield boundaries")
    boundary_facets = {
        WALL: np.asarray(boundaries[WALL], dtype=np.int64),
        FARFIELD: np.asarray(boundaries[FARFIELD], dtype=np.int64),
    }
    physical_tags = {ROCK: 1, WALL: 1, FARFIELD: 2}
    facet_markers = np.zeros(mesh.facets.shape[1], dtype=np.int32)
    facet_markers[boundary_facets[WALL]] = physical_tags[WALL]
    facet_markers[boundary_facets[FARFIELD]] = physical_tags[FARFIELD]
    elements = np.asarray(mesh.t.T, dtype=np.int64)
    cell_markers = np.full(elements.shape[0], physical_tags[ROCK], dtype=np.int32)
    return (
        mesh,
        np.asarray(mesh.p.T, dtype=np.float64),
        elements,
        boundary_facets,
        facet_markers,
        cell_markers,
        physical_tags,
        {"generator": "external-skfem-mesh", "coordinate_order": ["y", "z"]},
    )


def assemble_plane_strain_system(
    tunnel_mesh: TunnelMesh | Any,
    *,
    young_modulus: float,
    poisson_ratio: float,
    sigma_inf: ArrayLike,
) -> ElasticSystem:
    """Assemble the P1 vector elasticity matrix and excavation wall load."""

    dependencies = _require_elasticity_dependencies()
    material = PlaneStrainMaterial(young_modulus, poisson_ratio)
    stress = _coerce_sigma_inf(sigma_inf)
    mesh, _, _, boundary_facets, _, _, _, _ = _mesh_payload(tunnel_mesh)
    if boundary_facets[WALL].size == 0 or boundary_facets[FARFIELD].size == 0:
        raise ValueError("wall and farfield boundary marker sets must both be non-empty")

    element = dependencies["ElementVector"](dependencies["ElementTriP1"]())
    basis = dependencies["Basis"](mesh, element)
    wall_basis = dependencies["FacetBasis"](mesh, element, facets=boundary_facets[WALL])
    dot = dependencies["dot"]
    LinearForm = dependencies["LinearForm"]

    @LinearForm
    def excavation_load(test_function: Any, quadrature: Any) -> Any:
        traction = -np.einsum("ij,j...->i...", stress, quadrature.n)
        return dot(traction, test_function)

    stiffness = dependencies["asm"](
        dependencies["linear_elasticity"](Lambda=material.lame_lambda, Mu=material.shear_modulus),
        basis,
    ).tocsr()
    load = np.asarray(dependencies["asm"](excavation_load, wall_basis), dtype=np.float64)
    if not np.isfinite(stiffness.data).all() or not np.isfinite(load).all():
        raise RuntimeError("assembled stiffness or load contains a non-finite value")

    dirichlet_dofs = np.asarray(basis.get_dofs(FARFIELD).flatten(), dtype=np.int64)
    all_dofs = np.arange(stiffness.shape[0], dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs, assume_unique=False)
    if dirichlet_dofs.size == 0 or free_dofs.size == 0:
        raise RuntimeError("farfield constraints produced an empty constrained or free set")
    skew = stiffness - stiffness.T
    numerator = float(np.linalg.norm(skew.data)) if skew.nnz else 0.0
    denominator = max(float(np.linalg.norm(stiffness.data)), np.finfo(float).tiny)
    symmetry_error = numerator / denominator
    return ElasticSystem(
        stiffness=stiffness,
        load=load,
        basis=basis,
        dirichlet_dofs=dirichlet_dofs,
        free_dofs=free_dofs,
        material=material,
        sigma_inf=stress,
        stiffness_symmetry_error=symmetry_error,
    )


def solve_plane_strain_excavation(
    tunnel_mesh: TunnelMesh | Any,
    *,
    young_modulus: float,
    poisson_ratio: float,
    sigma_inf: ArrayLike,
    sigma_xx_inf: float | None = None,
) -> ElasticResult:
    """Solve one homogeneous elastic excavation increment.

    ``sigma_xx_inf`` defaults to ``nu * (Sigma_yy + Sigma_zz)``, which is the
    out-of-plane far-field stress compatible with isotropic plane strain.  It
    may be overridden when a separately specified axial in-situ stress is part
    of the synthetic case definition.
    """

    dependencies = _require_elasticity_dependencies()
    system = assemble_plane_strain_system(
        tunnel_mesh,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        sigma_inf=sigma_inf,
    )
    condensed = dependencies["condense"](
        system.stiffness,
        system.load,
        D=system.dirichlet_dofs,
    )
    solution = np.asarray(dependencies["solve"](*condensed), dtype=np.float64)
    if not np.isfinite(solution).all():
        raise RuntimeError("elastic solution contains a non-finite value")

    (
        _,
        nodes,
        elements,
        boundary_facets,
        facet_markers,
        cell_markers,
        physical_tags,
        mesh_metadata,
    ) = _mesh_payload(tunnel_mesh)
    nodal_dofs = np.asarray(system.basis.nodal_dofs, dtype=np.int64)
    if nodal_dofs.shape != (2, nodes.shape[0]):
        raise RuntimeError("unexpected scikit-fem vector P1 nodal DOF layout")
    displacement = solution[nodal_dofs].T
    strain, element_area = compute_element_strain(nodes, elements, displacement)
    delta_stress = plane_strain_stress(
        strain, young_modulus=young_modulus, poisson_ratio=poisson_ratio
    )
    sigma_vector = np.asarray(
        [system.sigma_inf[0, 0], system.sigma_inf[1, 1], system.sigma_inf[0, 1]],
        dtype=np.float64,
    )
    total_stress = delta_stress + sigma_vector[None, :]
    if sigma_xx_inf is None:
        sigma_xx_initial = float(poisson_ratio) * float(
            system.sigma_inf[0, 0] + system.sigma_inf[1, 1]
        )
    else:
        sigma_xx_initial = float(sigma_xx_inf)
        if not np.isfinite(sigma_xx_initial):
            raise ValueError("sigma_xx_inf must be finite")
    sigma_xx = sigma_xx_initial + plane_strain_sigma_xx(
        strain, young_modulus=young_modulus, poisson_ratio=poisson_ratio
    )

    energy_density = 0.5 * (
        strain[:, 0] * delta_stress[:, 0]
        + strain[:, 1] * delta_stress[:, 1]
        + strain[:, 2] * delta_stress[:, 2]
    )
    element_energy = float(np.sum(energy_density * element_area))
    algebraic_energy = 0.5 * float(solution @ (system.stiffness @ solution))
    external_work = float(solution @ system.load)
    energy_scale = max(abs(element_energy), abs(algebraic_energy), np.finfo(float).tiny)
    energy_discretization_error = abs(element_energy - algebraic_energy) / energy_scale
    closure_scale = max(abs(2.0 * algebraic_energy), abs(external_work))
    energy_closure = (
        0.0
        if closure_scale <= 100.0 * np.finfo(float).tiny
        else abs(2.0 * algebraic_energy - external_work) / closure_scale
    )

    residual = np.asarray(system.stiffness @ solution - system.load, dtype=np.float64)
    free_residual = residual[system.free_dofs]
    residual_norm = float(np.linalg.norm(free_residual))
    residual_scale = max(
        float(np.linalg.norm(system.load[system.free_dofs])),
        float(np.linalg.norm((system.stiffness @ solution)[system.free_dofs])),
    )
    algebraic_residual = (
        0.0 if residual_scale <= 100.0 * np.finfo(float).tiny else residual_norm / residual_scale
    )
    if not all(
        np.isfinite(value)
        for value in (
            element_energy,
            external_work,
            energy_closure,
            energy_discretization_error,
            residual_norm,
            algebraic_residual,
        )
    ):
        raise RuntimeError("elastic diagnostics contain a non-finite value")

    element_centers = nodes[elements].mean(axis=1)
    material_metadata = {
        "young_modulus": float(young_modulus),
        "poisson_ratio": float(poisson_ratio),
        "lame_lambda": system.material.lame_lambda,
        "shear_modulus": system.material.shear_modulus,
    }
    result_mesh_metadata = dict(mesh_metadata)
    result_mesh_metadata.update(
        {
            "formulation": "P1_vector_small_strain_plane_strain",
            "incremental_farfield_displacement": 0.0,
            "wall_incremental_traction": "-Sigma_inf@n_domain",
            "normal_convention": "rock_domain_outward_wall_normal_points_into_cavity",
            "stress_sign": "tension_positive",
            "strain_component_order": ["yy", "zz", "gamma_yz"],
            "stress_component_order": ["yy", "zz", "yz"],
            "energy_measure": "per_unit_tunnel_axis_thickness",
        }
    )
    return ElasticResult(
        nodes=nodes,
        elements=elements,
        displacement=displacement,
        strain=strain,
        delta_stress=delta_stress,
        total_stress=total_stress,
        sigma_xx=np.asarray(sigma_xx, dtype=np.float64),
        energy_density=energy_density,
        element_area=element_area,
        element_centers=element_centers,
        energy=element_energy,
        external_work=external_work,
        algebraic_residual=algebraic_residual,
        residual_norm=residual_norm,
        energy_closure=energy_closure,
        energy_discretization_error=energy_discretization_error,
        stiffness_symmetry_error=system.stiffness_symmetry_error,
        boundary_facets=boundary_facets,
        facet_markers=facet_markers,
        cell_markers=cell_markers,
        physical_tags=physical_tags,
        material=material_metadata,
        sigma_inf=system.sigma_inf,
        sigma_xx_inf=sigma_xx_initial,
        mesh_metadata=result_mesh_metadata,
    )


# Short aliases for callers that already state the formulation in their config.
solve_elastic_excavation = solve_plane_strain_excavation
solve_plane_strain = solve_plane_strain_excavation


__all__ = [
    "ElasticResult",
    "ElasticSystem",
    "PlaneStrainMaterial",
    "assemble_plane_strain_system",
    "compute_element_strain",
    "plane_strain_lame_parameters",
    "plane_strain_sigma_xx",
    "plane_strain_stress",
    "solve_elastic_excavation",
    "solve_plane_strain",
    "solve_plane_strain_excavation",
]
