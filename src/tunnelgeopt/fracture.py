"""Small-strain plane-strain AT2 phase-field fracture kernel.

This module is an internal numerical kernel, not a validated fracture-label
generator.  It implements the candidate formulation documented in
``paper/FRACTURE_SOLVER_BLUEPRINT.md``:

* ``d = 0`` is intact and ``d = 1`` is fully damaged;
* the Miehe split is a three-dimensional spectral strain split evaluated with
  ``epsilon_xx = 0``;
* the tensile history is elementwise for P1 displacement fields;
* the scalar P1 AT2 problem is solved as a bound-constrained quadratic problem,
  including irreversibility and explicit KKT diagnostics; and
* excavation uses the total field ``u = epsilon_inf x + w`` with wall traction
  ``(1 - load_parameter) Sigma_inf n``.

The P1 element strain is constant.  Elastic and history terms are therefore
integrated exactly per triangle, while the quadratic P1 damage terms use the
exact consistent mass matrix and constant gradient.  The displacement tangent
uses the analytic isotropic tensor in the intact limit and a centered numerical
constitutive tangent otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .elasticity import compute_element_strain, plane_strain_lame_parameters
from .mesh import FARFIELD, WALL, TunnelMesh

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


@dataclass(frozen=True)
class AT2Material:
    """Homogeneous isotropic AT2 material parameters.

    ``fracture_toughness`` is :math:`G_c`, ``length_scale`` is :math:`ell`,
    and ``residual_stiffness`` is the additive ``k`` in
    ``g(d) = (1 - d)**2 + k``.  Setting ``k=0`` and fixing ``d=0`` exposes the
    exact intact regression path to the linear elastic excavation solver.
    """

    young_modulus: float
    poisson_ratio: float
    fracture_toughness: float
    length_scale: float
    residual_stiffness: float = 1.0e-8

    def __post_init__(self) -> None:
        plane_strain_lame_parameters(self.young_modulus, self.poisson_ratio)
        if not np.isfinite(self.fracture_toughness) or self.fracture_toughness <= 0.0:
            raise ValueError("fracture_toughness must be finite and positive")
        if not np.isfinite(self.length_scale) or self.length_scale <= 0.0:
            raise ValueError("length_scale must be finite and positive")
        if (
            not np.isfinite(self.residual_stiffness)
            or self.residual_stiffness < 0.0
            or self.residual_stiffness >= 1.0
        ):
            raise ValueError("residual_stiffness must be finite and lie in [0, 1)")

    @property
    def lame_lambda(self) -> float:
        return plane_strain_lame_parameters(self.young_modulus, self.poisson_ratio)[0]

    @property
    def shear_modulus(self) -> float:
        return plane_strain_lame_parameters(self.young_modulus, self.poisson_ratio)[1]


@dataclass(frozen=True)
class FractureSolverOptions:
    """Numerical controls for the local active-set and staggered solves."""

    max_staggered_iterations: int = 40
    max_displacement_iterations: int = 30
    max_active_set_iterations: int = 100
    staggered_tolerance: float = 1.0e-7
    equilibrium_tolerance: float = 1.0e-8
    kkt_tolerance: float = 1.0e-8
    active_set_tolerance: float = 1.0e-10
    line_search_steps: int = 16
    tangent_perturbation: float = 1.0e-7
    raise_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_staggered_iterations",
            "max_displacement_iterations",
            "max_active_set_iterations",
            "line_search_steps",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "staggered_tolerance",
            "equilibrium_tolerance",
            "kkt_tolerance",
            "active_set_tolerance",
            "tangent_perturbation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class AT2LoadPath:
    """Strictly increasing scalar wall-release parameters in ``[0, 1]``."""

    load_parameters: tuple[float, ...] = (0.0, 1.0)

    def __post_init__(self) -> None:
        values = np.asarray(self.load_parameters, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("load_parameters must be a non-empty one-dimensional sequence")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("load_parameters must be finite and lie in [0, 1]")
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError("load_parameters must be strictly increasing")
        object.__setattr__(self, "load_parameters", tuple(float(value) for value in values))


@dataclass(frozen=True)
class SplitResponse:
    """Three-dimensional Miehe split response for plane-strain inputs."""

    strain_tensor: FloatArray
    strain_positive: FloatArray
    strain_negative: FloatArray
    principal_strains: FloatArray
    psi_positive: FloatArray
    psi_negative: FloatArray
    stress_positive: FloatArray
    stress_negative: FloatArray
    stress: FloatArray


@dataclass(frozen=True)
class DamageSystem:
    """Unconstrained scalar P1 AT2 system before applying box constraints."""

    stiffness: Any
    load: FloatArray
    nodes: FloatArray
    elements: IntArray
    history: FloatArray
    element_area: FloatArray

    @property
    def K(self) -> Any:
        return self.stiffness

    @property
    def f(self) -> FloatArray:
        return self.load


@dataclass(frozen=True)
class DamageSolveResult:
    """Bound-constrained AT2 damage solution and auditable KKT diagnostics."""

    damage: FloatArray
    gradient: FloatArray
    kkt_residual: float
    stationarity_residual: float
    complementarity_residual: float
    primal_violation: float
    irreversibility_violation: float
    range_violation: float
    active_lower_count: int
    active_upper_count: int
    iterations: int
    converged: bool


@dataclass(frozen=True)
class DisplacementSolveResult:
    """Fixed-damage total-field displacement equilibrium result."""

    displacement: FloatArray
    correction_displacement: FloatArray
    strain: FloatArray
    stress: FloatArray
    psi_positive: FloatArray
    psi_negative: FloatArray
    elastic_energy: float
    external_work: float
    residual_norm: float
    equilibrium_residual: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class AT2StepResult:
    """One accepted (or explicitly nonconverged) load-step state."""

    load_parameter: float
    displacement: FloatArray
    correction_displacement: FloatArray
    damage: FloatArray
    strain: FloatArray
    stress: FloatArray
    psi_positive: FloatArray
    psi_negative: FloatArray
    history: FloatArray
    elastic_energy: float
    fracture_energy: float
    external_work: float
    total_potential_energy: float
    equilibrium_residual: float
    kkt_residual: float
    complementarity_residual: float
    irreversibility_violation: float
    range_violation: float
    displacement_change: float
    damage_change: float
    staggered_iterations: int
    displacement_iterations: int
    damage_iterations: int
    converged: bool


@dataclass(frozen=True)
class AT2Result:
    """Local AT2 trajectory; no external-validation status is implied."""

    nodes: FloatArray
    elements: IntArray
    steps: tuple[AT2StepResult, ...]
    material: AT2Material
    load_path: AT2LoadPath
    options: FractureSolverOptions
    sigma_inf: FloatArray

    @property
    def final(self) -> AT2StepResult:
        return self.steps[-1]

    @property
    def converged(self) -> bool:
        return bool(self.steps) and all(step.converged for step in self.steps)


def degradation(damage: ArrayLike, *, residual_stiffness: float = 0.0) -> FloatArray:
    """Return ``g(d) = (1-d)^2 + k`` for finite damage in ``[0, 1]``."""

    values = np.asarray(damage, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("damage contains a non-finite value")
    tolerance = 32.0 * np.finfo(float).eps
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("damage must lie in [0, 1]")
    residual = float(residual_stiffness)
    if not np.isfinite(residual) or residual < 0.0 or residual >= 1.0:
        raise ValueError("residual_stiffness must be finite and lie in [0, 1)")
    bounded = np.clip(values, 0.0, 1.0)
    return np.asarray((1.0 - bounded) ** 2 + residual, dtype=np.float64)


def _coerce_engineering_strain(strain: ArrayLike) -> FloatArray:
    values = np.asarray(strain, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("strain must end in [yy, zz, gamma_yz]")
    if not np.isfinite(values).all():
        raise ValueError("strain contains a non-finite value")
    return values


def plane_strain_spectral_split(
    strain: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return full, positive and negative 3-D strains and their eigenvalues.

    The input uses engineering components ``[yy, zz, gamma_yz]``.  Returned
    tensors use coordinate order ``(x, y, z)`` and enforce ``epsilon_xx=0``.
    At a zero eigenvalue both the positive and negative contribution are zero.
    """

    values = _coerce_engineering_strain(strain)
    tensor = np.zeros(values.shape[:-1] + (3, 3), dtype=np.float64)
    tensor[..., 1, 1] = values[..., 0]
    tensor[..., 2, 2] = values[..., 1]
    tensor[..., 1, 2] = 0.5 * values[..., 2]
    tensor[..., 2, 1] = 0.5 * values[..., 2]
    principal, directions = np.linalg.eigh(tensor)
    positive_values = np.maximum(principal, 0.0)
    negative_values = np.minimum(principal, 0.0)
    positive = np.einsum(
        "...ik,...k,...jk->...ij", directions, positive_values, directions, optimize=True
    )
    negative = np.einsum(
        "...ik,...k,...jk->...ij", directions, negative_values, directions, optimize=True
    )
    return tensor, positive, negative, principal


def miehe_spectral_response(
    strain: ArrayLike,
    material: AT2Material,
    *,
    damage: ArrayLike = 0.0,
) -> SplitResponse:
    """Evaluate split energies and stresses under three-dimensional plane strain."""

    tensor, positive, negative, principal = plane_strain_spectral_split(strain)
    trace = np.trace(tensor, axis1=-2, axis2=-1)
    trace_positive = np.maximum(trace, 0.0)
    trace_negative = np.minimum(trace, 0.0)
    mu = material.shear_modulus
    lame_lambda = material.lame_lambda
    psi_positive = 0.5 * lame_lambda * trace_positive**2
    psi_positive += mu * np.einsum("...ij,...ij->...", positive, positive, optimize=True)
    psi_negative = 0.5 * lame_lambda * trace_negative**2
    psi_negative += mu * np.einsum("...ij,...ij->...", negative, negative, optimize=True)
    identity = np.eye(3, dtype=np.float64)
    stress_positive = lame_lambda * trace_positive[..., None, None] * identity
    stress_positive = stress_positive + 2.0 * mu * positive
    stress_negative = lame_lambda * trace_negative[..., None, None] * identity
    stress_negative = stress_negative + 2.0 * mu * negative
    damage_values = np.asarray(damage, dtype=np.float64)
    try:
        damage_values = np.broadcast_to(damage_values, psi_positive.shape)
    except ValueError as exc:
        raise ValueError("damage is not broadcast-compatible with strain") from exc
    factor = degradation(damage_values, residual_stiffness=material.residual_stiffness)
    stress = factor[..., None, None] * stress_positive + stress_negative
    return SplitResponse(
        strain_tensor=tensor,
        strain_positive=np.asarray(positive, dtype=np.float64),
        strain_negative=np.asarray(negative, dtype=np.float64),
        principal_strains=np.asarray(principal, dtype=np.float64),
        psi_positive=np.asarray(psi_positive, dtype=np.float64),
        psi_negative=np.asarray(psi_negative, dtype=np.float64),
        stress_positive=np.asarray(stress_positive, dtype=np.float64),
        stress_negative=np.asarray(stress_negative, dtype=np.float64),
        stress=np.asarray(stress, dtype=np.float64),
    )


def update_history(previous_history: ArrayLike, psi_positive: ArrayLike) -> FloatArray:
    """Apply the irreversible tensile-history update ``max(H_old, psi_plus)``."""

    previous = np.asarray(previous_history, dtype=np.float64)
    current = np.asarray(psi_positive, dtype=np.float64)
    try:
        previous, current = np.broadcast_arrays(previous, current)
    except ValueError as exc:
        raise ValueError("history and psi_positive are not broadcast-compatible") from exc
    if not np.isfinite(previous).all() or not np.isfinite(current).all():
        raise ValueError("history inputs must be finite")
    tolerance = (
        64.0
        * np.finfo(float).eps
        * max(
            float(np.max(np.abs(previous), initial=0.0)),
            float(np.max(np.abs(current), initial=0.0)),
            1.0,
        )
    )
    if np.any(previous < -tolerance) or np.any(current < -tolerance):
        raise ValueError("history and psi_positive must be nonnegative")
    return np.maximum(np.maximum(previous, 0.0), np.maximum(current, 0.0))


def _require_scipy() -> dict[str, Any]:
    try:
        from scipy.sparse import coo_matrix  # type: ignore[import-not-found]
        from scipy.sparse.linalg import spsolve  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional numerical stack
        raise RuntimeError("AT2 solving requires SciPy 1.15 or newer") from exc
    return {"coo_matrix": coo_matrix, "spsolve": spsolve}


def _mesh_arrays(mesh_like: TunnelMesh | Any) -> tuple[Any, FloatArray, IntArray]:
    mesh = mesh_like.mesh if isinstance(mesh_like, TunnelMesh) else mesh_like
    if not hasattr(mesh, "p") or not hasattr(mesh, "t"):
        raise TypeError("mesh must be TunnelMesh or a scikit-fem MeshTri")
    nodes = np.asarray(mesh.p.T, dtype=np.float64)
    elements = np.asarray(mesh.t.T, dtype=np.int64)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("mesh nodes must have shape [N, 2]")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError("mesh elements must have shape [M, 3]")
    if not np.isfinite(nodes).all():
        raise ValueError("mesh nodes contain a non-finite value")
    return mesh, nodes, elements


def _element_geometry(
    nodes: FloatArray, elements: IntArray
) -> tuple[FloatArray, FloatArray, FloatArray]:
    triangles = nodes[elements]
    first_edge = triangles[:, 1] - triangles[:, 0]
    second_edge = triangles[:, 2] - triangles[:, 0]
    determinants = first_edge[:, 0] * second_edge[:, 1]
    determinants -= first_edge[:, 1] * second_edge[:, 0]
    scale = np.maximum(
        np.maximum(np.sum(first_edge**2, axis=1), np.sum(second_edge**2, axis=1)),
        np.finfo(float).tiny,
    )
    if np.any(np.abs(determinants) <= 1.0e-14 * scale):
        raise ValueError("mesh contains a degenerate triangle")
    gradients = np.empty((elements.shape[0], 2, 3), dtype=np.float64)
    gradients[:, 0, 0] = (triangles[:, 1, 1] - triangles[:, 2, 1]) / determinants
    gradients[:, 0, 1] = (triangles[:, 2, 1] - triangles[:, 0, 1]) / determinants
    gradients[:, 0, 2] = (triangles[:, 0, 1] - triangles[:, 1, 1]) / determinants
    gradients[:, 1, 0] = (triangles[:, 2, 0] - triangles[:, 1, 0]) / determinants
    gradients[:, 1, 1] = (triangles[:, 0, 0] - triangles[:, 2, 0]) / determinants
    gradients[:, 1, 2] = (triangles[:, 1, 0] - triangles[:, 0, 0]) / determinants
    return gradients, 0.5 * np.abs(determinants), determinants


def assemble_at2_damage_system(
    mesh: TunnelMesh | Any,
    material: AT2Material,
    history: ArrayLike,
) -> DamageSystem:
    """Assemble the exact scalar P1 AT2 system for elementwise history."""

    scipy = _require_scipy()
    _, nodes, elements = _mesh_arrays(mesh)
    history_values = np.asarray(history, dtype=np.float64)
    if history_values.ndim == 0:
        history_values = np.full(elements.shape[0], float(history_values), dtype=np.float64)
    if history_values.shape != (elements.shape[0],):
        raise ValueError("history must be scalar or have shape [element_count]")
    if not np.isfinite(history_values).all() or np.any(history_values < 0.0):
        raise ValueError("history must be finite and nonnegative")
    gradients, area, _ = _element_geometry(nodes, elements)
    local_mass_template = np.asarray(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]],
        dtype=np.float64,
    )
    reaction = material.fracture_toughness / material.length_scale + 2.0 * history_values
    local_mass = reaction[:, None, None] * area[:, None, None] / 12.0
    local_mass = local_mass * local_mass_template[None, :, :]
    local_gradient = material.fracture_toughness * material.length_scale
    local_gradient = local_gradient * area[:, None, None]
    local_gradient = local_gradient * np.einsum("eia,eib->eab", gradients, gradients, optimize=True)
    local_stiffness = local_mass + local_gradient
    rows = np.repeat(elements, 3, axis=1).ravel()
    columns = np.tile(elements, (1, 3)).ravel()
    stiffness = scipy["coo_matrix"](
        (local_stiffness.ravel(), (rows, columns)),
        shape=(nodes.shape[0], nodes.shape[0]),
    ).tocsr()
    load = np.zeros(nodes.shape[0], dtype=np.float64)
    np.add.at(load, elements.ravel(), np.repeat(2.0 * history_values * area / 3.0, 3))
    if not np.isfinite(stiffness.data).all() or not np.isfinite(load).all():
        raise RuntimeError("assembled damage system contains a non-finite value")
    return DamageSystem(
        stiffness=stiffness,
        load=load,
        nodes=nodes,
        elements=elements,
        history=history_values,
        element_area=area,
    )


def _damage_kkt_diagnostics(
    stiffness: Any,
    load: FloatArray,
    damage: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    active_tolerance: float,
) -> tuple[FloatArray, float, float, float, float, int, int]:
    gradient = np.asarray(stiffness @ damage - load, dtype=np.float64)
    active_lower = damage <= lower + active_tolerance
    active_upper = damage >= upper - active_tolerance
    multiplier_lower = np.where(active_lower, np.maximum(gradient, 0.0), 0.0)
    multiplier_upper = np.where(active_upper, np.maximum(-gradient, 0.0), 0.0)
    stationarity = gradient - multiplier_lower + multiplier_upper
    stationarity_residual = float(np.linalg.norm(stationarity, ord=np.inf))
    complementarity_residual = float(
        max(
            np.max(np.abs(multiplier_lower * (damage - lower)), initial=0.0),
            np.max(np.abs(multiplier_upper * (upper - damage)), initial=0.0),
        )
    )
    primal_violation = float(
        max(
            np.max(lower - damage, initial=0.0),
            np.max(damage - upper, initial=0.0),
            0.0,
        )
    )
    diagonal = np.asarray(stiffness.diagonal(), dtype=np.float64)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.0):
        raise RuntimeError("damage stiffness must have a finite positive diagonal")
    projected = damage - np.clip(damage - gradient / diagonal, lower, upper)
    scale = max(
        float(np.linalg.norm(load, ord=np.inf)),
        float(np.linalg.norm(np.asarray(stiffness @ damage), ord=np.inf)),
        np.finfo(float).tiny,
    )
    kkt_residual = max(
        float(np.linalg.norm(projected, ord=np.inf)),
        stationarity_residual / scale,
        complementarity_residual / scale,
        primal_violation,
    )
    return (
        gradient,
        kkt_residual,
        stationarity_residual / scale,
        complementarity_residual / scale,
        primal_violation,
        int(np.count_nonzero(active_lower)),
        int(np.count_nonzero(active_upper)),
    )


def solve_at2_damage(
    system_or_mesh: DamageSystem | TunnelMesh | Any,
    material: AT2Material | None = None,
    history: ArrayLike | None = None,
    *,
    damage_old: ArrayLike,
    initial_damage: ArrayLike | None = None,
    options: FractureSolverOptions | None = None,
) -> DamageSolveResult:
    """Solve the irreversibility-constrained damage quadratic by active set.

    The lower bound is the previous accepted damage, not a clipped
    unconstrained solution.  The returned residual is a scaled projected/KKT
    residual and the separate primal/complementarity diagnostics remain
    available for auditing.
    """

    controls = options or FractureSolverOptions()
    if isinstance(system_or_mesh, DamageSystem):
        if material is not None or history is not None:
            raise ValueError("material/history must be omitted when passing a DamageSystem")
        system = system_or_mesh
    else:
        if material is None or history is None:
            raise ValueError("material and history are required when passing a mesh")
        system = assemble_at2_damage_system(system_or_mesh, material, history)
    lower = np.asarray(damage_old, dtype=np.float64)
    if lower.ndim == 0:
        lower = np.full(system.nodes.shape[0], float(lower), dtype=np.float64)
    if lower.shape != (system.nodes.shape[0],):
        raise ValueError("damage_old must be scalar or have shape [node_count]")
    if not np.isfinite(lower).all() or np.any(lower < 0.0) or np.any(lower > 1.0):
        raise ValueError("damage_old must be finite and lie in [0, 1]")
    upper = np.ones_like(lower)
    if initial_damage is None:
        damage = lower.copy()
    else:
        initial = np.asarray(initial_damage, dtype=np.float64)
        if initial.shape != lower.shape or not np.isfinite(initial).all():
            raise ValueError("initial_damage must be finite with shape [node_count]")
        damage = np.clip(initial, lower, upper)

    scipy = _require_scipy()
    diagonal = np.asarray(system.stiffness.diagonal(), dtype=np.float64)
    if not np.isfinite(diagonal).all() or np.any(diagonal <= 0.0):
        raise RuntimeError("damage stiffness must have a finite positive diagonal")
    converged = False
    iterations = 0
    diagnostics: tuple[FloatArray, float, float, float, float, int, int] | None = None
    for iterations in range(1, controls.max_active_set_iterations + 1):
        gradient = np.asarray(system.stiffness @ damage - system.load, dtype=np.float64)
        # Primal-dual active-set classification is the semismooth Newton form
        # of x = projection_[lower, upper](x - D^-1 grad).  Unlike clipping an
        # unconstrained answer, each active set defines and solves its own KKT
        # linear system.  Intermediate iterates may be infeasible; feasibility
        # is part of the reported convergence diagnostic.
        projected_argument = damage - gradient / diagonal
        fixed_lower = projected_argument <= lower + controls.active_set_tolerance
        fixed_upper = projected_argument >= upper - controls.active_set_tolerance
        fixed_upper &= ~fixed_lower
        fixed = fixed_lower | fixed_upper
        free = ~fixed
        trial = np.empty_like(damage)
        trial[fixed_lower] = lower[fixed_lower]
        trial[fixed_upper] = upper[fixed_upper]
        if np.any(free):
            free_indices = np.flatnonzero(free)
            fixed_indices = np.flatnonzero(fixed)
            rhs = system.load[free_indices].copy()
            if fixed_indices.size:
                rhs -= system.stiffness[free_indices][:, fixed_indices] @ trial[fixed_indices]
            free_stiffness = system.stiffness[free_indices][:, free_indices]
            free_solution = np.asarray(scipy["spsolve"](free_stiffness, rhs), dtype=np.float64)
            if not np.isfinite(free_solution).all():
                raise RuntimeError("active-set free solve produced a non-finite value")
            trial[free_indices] = free_solution
        damage = trial

        diagnostics = _damage_kkt_diagnostics(
            system.stiffness,
            system.load,
            damage,
            lower,
            upper,
            active_tolerance=controls.active_set_tolerance,
        )
        if diagnostics[1] <= controls.kkt_tolerance:
            converged = True
            break

    assert diagnostics is not None
    gradient, kkt, stationarity, complementarity, primal, lower_count, upper_count = diagnostics
    irreversibility_violation = float(max(np.max(lower - damage, initial=0.0), 0.0))
    range_violation = float(
        max(np.max(-damage, initial=0.0), np.max(damage - 1.0, initial=0.0), 0.0)
    )
    if not converged and controls.raise_on_nonconvergence:
        raise RuntimeError(
            "AT2 active-set solve did not converge "
            f"(iterations={iterations}, kkt_residual={kkt:.3e})"
        )
    return DamageSolveResult(
        damage=damage,
        gradient=gradient,
        kkt_residual=float(kkt),
        stationarity_residual=float(stationarity),
        complementarity_residual=float(complementarity),
        primal_violation=float(primal),
        irreversibility_violation=irreversibility_violation,
        range_violation=range_violation,
        active_lower_count=lower_count,
        active_upper_count=upper_count,
        iterations=iterations,
        converged=converged,
    )


def _coerce_sigma_inf(sigma_inf: ArrayLike) -> FloatArray:
    stress = np.asarray(sigma_inf, dtype=np.float64)
    if stress.shape == (3,):
        stress = np.asarray([[stress[0], stress[2]], [stress[2], stress[1]]], dtype=np.float64)
    if stress.shape != (2, 2):
        raise ValueError("sigma_inf must be a symmetric 2x2 matrix or [yy, zz, yz]")
    if not np.isfinite(stress).all():
        raise ValueError("sigma_inf contains a non-finite value")
    tolerance = 1.0e-12 * max(float(np.max(np.abs(stress))), 1.0)
    if not np.allclose(stress, stress.T, rtol=0.0, atol=tolerance):
        raise ValueError("sigma_inf must be symmetric")
    return 0.5 * (stress + stress.T)


def _farfield_engineering_strain(sigma_inf: FloatArray, material: AT2Material) -> FloatArray:
    lame_lambda = material.lame_lambda
    mu = material.shear_modulus
    normal_matrix = np.asarray(
        [[lame_lambda + 2.0 * mu, lame_lambda], [lame_lambda, lame_lambda + 2.0 * mu]],
        dtype=np.float64,
    )
    normal_strain = np.linalg.solve(normal_matrix, np.diag(sigma_inf))
    return np.asarray([normal_strain[0], normal_strain[1], sigma_inf[0, 1] / mu], dtype=np.float64)


def _affine_displacement(nodes: FloatArray, strain: FloatArray) -> FloatArray:
    tensor = np.asarray(
        [[strain[0], 0.5 * strain[2]], [0.5 * strain[2], strain[1]]], dtype=np.float64
    )
    return nodes @ tensor.T


def _named_boundaries(mesh: Any) -> tuple[IntArray, IntArray]:
    boundaries = getattr(mesh, "boundaries", None)
    if boundaries is None or WALL not in boundaries or FARFIELD not in boundaries:
        raise ValueError("the scikit-fem mesh must have named wall and farfield boundaries")
    wall = np.asarray(boundaries[WALL], dtype=np.int64)
    farfield = np.asarray(boundaries[FARFIELD], dtype=np.int64)
    if wall.size == 0 or farfield.size == 0:
        raise ValueError("wall and farfield boundary marker sets must both be non-empty")
    if np.intersect1d(wall, farfield).size:
        raise ValueError("wall and farfield boundary markers overlap")
    return wall, farfield


def _integrated_wall_load(
    mesh: Any,
    nodes: FloatArray,
    elements: IntArray,
    wall_facets: IntArray,
    sigma_inf: FloatArray,
) -> FloatArray:
    load = np.zeros((nodes.shape[0], 2), dtype=np.float64)
    facets = np.asarray(mesh.facets, dtype=np.int64)
    f2t = np.asarray(mesh.f2t, dtype=np.int64)
    for facet_index in wall_facets:
        edge_nodes = facets[:, facet_index]
        adjacent = f2t[:, facet_index]
        adjacent = adjacent[adjacent >= 0]
        if adjacent.size != 1:
            raise RuntimeError("a wall facet must have exactly one adjacent rock element")
        edge = nodes[edge_nodes[1]] - nodes[edge_nodes[0]]
        length = float(np.linalg.norm(edge))
        if length <= 0.0:
            raise RuntimeError("wall boundary contains a zero-length facet")
        normal = np.asarray([-edge[1], edge[0]], dtype=np.float64) / length
        midpoint = 0.5 * (nodes[edge_nodes[0]] + nodes[edge_nodes[1]])
        centroid = nodes[elements[int(adjacent[0])]].mean(axis=0)
        if float(normal @ (midpoint - centroid)) < 0.0:
            normal = -normal
        traction = sigma_inf @ normal
        load[edge_nodes] += 0.5 * length * traction
    return load.ravel()


def _strain_displacement_matrices(gradients: FloatArray) -> FloatArray:
    matrices = np.zeros((gradients.shape[0], 3, 6), dtype=np.float64)
    matrices[:, 0, 0::2] = gradients[:, 0, :]
    matrices[:, 1, 1::2] = gradients[:, 1, :]
    matrices[:, 2, 0::2] = gradients[:, 1, :]
    matrices[:, 2, 1::2] = gradients[:, 0, :]
    return matrices


def _average_degradation(local_damage: FloatArray, material: AT2Material) -> FloatArray:
    mean_damage = np.mean(local_damage, axis=1)
    sum_squares = np.sum(local_damage**2, axis=1)
    pair_sum = (
        local_damage[:, 0] * local_damage[:, 1]
        + local_damage[:, 0] * local_damage[:, 2]
        + local_damage[:, 1] * local_damage[:, 2]
    )
    mean_square_damage = (sum_squares + pair_sum) / 6.0
    return 1.0 - 2.0 * mean_damage + mean_square_damage + material.residual_stiffness


def _inplane_stress(tensor: FloatArray) -> FloatArray:
    return np.asarray([tensor[1, 1], tensor[2, 2], tensor[1, 2]], dtype=np.float64)


def _constitutive_response(
    strain: FloatArray, material: AT2Material, degradation_factor: float
) -> tuple[FloatArray, float, float]:
    response = miehe_spectral_response(
        strain,
        AT2Material(
            material.young_modulus,
            material.poisson_ratio,
            material.fracture_toughness,
            material.length_scale,
            0.0,
        ),
    )
    stress_tensor = degradation_factor * response.stress_positive + response.stress_negative
    return (
        _inplane_stress(stress_tensor),
        float(response.psi_positive),
        float(response.psi_negative),
    )


def _isotropic_tangent(material: AT2Material) -> FloatArray:
    lame_lambda = material.lame_lambda
    mu = material.shear_modulus
    return np.asarray(
        [
            [lame_lambda + 2.0 * mu, lame_lambda, 0.0],
            [lame_lambda, lame_lambda + 2.0 * mu, 0.0],
            [0.0, 0.0, mu],
        ],
        dtype=np.float64,
    )


def _numerical_constitutive_tangent(
    strain: FloatArray,
    material: AT2Material,
    degradation_factor: float,
    perturbation: float,
) -> FloatArray:
    if abs(degradation_factor - 1.0) <= 64.0 * np.finfo(float).eps:
        return _isotropic_tangent(material)
    tangent = np.empty((3, 3), dtype=np.float64)
    for column in range(3):
        step = perturbation * max(1.0, abs(float(strain[column])))
        positive = strain.copy()
        negative = strain.copy()
        positive[column] += step
        negative[column] -= step
        stress_positive = _constitutive_response(positive, material, degradation_factor)[0]
        stress_negative = _constitutive_response(negative, material, degradation_factor)[0]
        tangent[:, column] = (stress_positive - stress_negative) / (2.0 * step)
    return 0.5 * (tangent + tangent.T)


def _assemble_displacement_state(
    nodes: FloatArray,
    elements: IntArray,
    area: FloatArray,
    matrices: FloatArray,
    displacement: FloatArray,
    damage: FloatArray,
    material: AT2Material,
    *,
    tangent_perturbation: float,
    assemble_tangent: bool,
) -> tuple[FloatArray, Any | None, FloatArray, FloatArray, FloatArray, FloatArray, float]:
    scipy = _require_scipy()
    strain, _ = compute_element_strain(nodes, elements, displacement)
    degradation_average = _average_degradation(damage[elements], material)
    stress = np.empty_like(strain)
    psi_positive = np.empty(elements.shape[0], dtype=np.float64)
    psi_negative = np.empty(elements.shape[0], dtype=np.float64)
    internal = np.zeros(2 * nodes.shape[0], dtype=np.float64)
    local_tangents = (
        np.empty((elements.shape[0], 6, 6), dtype=np.float64) if assemble_tangent else None
    )
    for index in range(elements.shape[0]):
        stress[index], psi_positive[index], psi_negative[index] = _constitutive_response(
            strain[index], material, float(degradation_average[index])
        )
        local_internal = area[index] * (matrices[index].T @ stress[index])
        dofs = np.column_stack((2 * elements[index], 2 * elements[index] + 1)).ravel()
        internal[dofs] += local_internal
        if local_tangents is not None:
            constitutive = _numerical_constitutive_tangent(
                strain[index],
                material,
                float(degradation_average[index]),
                tangent_perturbation,
            )
            local_tangents[index] = area[index] * (
                matrices[index].T @ constitutive @ matrices[index]
            )
    tangent = None
    if local_tangents is not None:
        element_dofs = np.empty((elements.shape[0], 6), dtype=np.int64)
        element_dofs[:, 0::2] = 2 * elements
        element_dofs[:, 1::2] = 2 * elements + 1
        rows = np.repeat(element_dofs, 6, axis=1).ravel()
        columns = np.tile(element_dofs, (1, 6)).ravel()
        tangent = scipy["coo_matrix"](
            (local_tangents.ravel(), (rows, columns)),
            shape=(2 * nodes.shape[0], 2 * nodes.shape[0]),
        ).tocsr()
    elastic_energy = float(np.sum(area * (degradation_average * psi_positive + psi_negative)))
    return internal, tangent, strain, stress, psi_positive, psi_negative, elastic_energy


def _evaluate_fixed_damage_displacement_state(
    mesh_like: TunnelMesh | Any,
    material: AT2Material,
    sigma_inf: ArrayLike,
    *,
    load_parameter: float,
    damage: ArrayLike,
    displacement: ArrayLike,
    options: FractureSolverOptions,
    iterations: int = 0,
) -> DisplacementSolveResult:
    """Reassemble one immutable ``(u, d)`` state without taking a Newton step.

    The staggered solver calls this immediately after every damage update.  It
    prevents equilibrium, stress and energy diagnostics from being inherited
    from the preceding displacement solve, which used the pre-update damage.
    """

    parameter = float(load_parameter)
    if not np.isfinite(parameter) or not 0.0 <= parameter <= 1.0:
        raise ValueError("load_parameter must be finite and lie in [0, 1]")
    mesh, nodes, elements = _mesh_arrays(mesh_like)
    wall_facets, farfield_facets = _named_boundaries(mesh)
    material_stress = _coerce_sigma_inf(sigma_inf)
    damage_values = np.asarray(damage, dtype=np.float64)
    if damage_values.ndim == 0:
        damage_values = np.full(nodes.shape[0], float(damage_values), dtype=np.float64)
    if damage_values.shape != (nodes.shape[0],):
        raise ValueError("damage must be scalar or have shape [node_count]")
    if (
        not np.isfinite(damage_values).all()
        or np.any(damage_values < 0.0)
        or np.any(damage_values > 1.0)
    ):
        raise ValueError("damage must be finite and lie in [0, 1]")
    displacement_values = np.asarray(displacement, dtype=np.float64).copy()
    if displacement_values.shape != nodes.shape or not np.isfinite(displacement_values).all():
        raise ValueError("displacement must be finite with shape [node_count, 2]")

    gradients, area, _ = _element_geometry(nodes, elements)
    matrices = _strain_displacement_matrices(gradients)
    affine = _affine_displacement(nodes, _farfield_engineering_strain(material_stress, material))
    farfield_nodes = np.unique(np.asarray(mesh.facets)[:, farfield_facets])
    displacement_values[farfield_nodes] = affine[farfield_nodes]
    fixed_dofs = np.column_stack((2 * farfield_nodes, 2 * farfield_nodes + 1)).ravel()
    all_dofs = np.arange(2 * nodes.shape[0], dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=False)
    if not free_dofs.size:
        raise RuntimeError("farfield constraints leave no free displacement degrees of freedom")

    full_wall_load = _integrated_wall_load(mesh, nodes, elements, wall_facets, material_stress)
    external_load = (1.0 - parameter) * full_wall_load
    state = _assemble_displacement_state(
        nodes,
        elements,
        area,
        matrices,
        displacement_values,
        damage_values,
        material,
        tangent_perturbation=options.tangent_perturbation,
        assemble_tangent=False,
    )
    residual = state[0] - external_load
    residual_norm = float(np.linalg.norm(residual[free_dofs]))
    scale = max(
        float(np.linalg.norm(state[0][free_dofs])),
        float(np.linalg.norm(external_load[free_dofs])),
        float(np.linalg.norm(full_wall_load[free_dofs])),
        np.finfo(float).tiny,
    )
    equilibrium_residual = residual_norm / scale
    return DisplacementSolveResult(
        displacement=displacement_values,
        correction_displacement=displacement_values - affine,
        strain=state[2],
        stress=state[3],
        psi_positive=state[4],
        psi_negative=state[5],
        elastic_energy=float(state[6]),
        external_work=float(external_load @ displacement_values.ravel()),
        residual_norm=residual_norm,
        equilibrium_residual=equilibrium_residual,
        iterations=iterations,
        converged=equilibrium_residual <= options.equilibrium_tolerance,
    )


def solve_fixed_damage_displacement(
    mesh_like: TunnelMesh | Any,
    material: AT2Material,
    sigma_inf: ArrayLike,
    *,
    load_parameter: float,
    damage: ArrayLike,
    initial_displacement: ArrayLike | None = None,
    options: FractureSolverOptions | None = None,
) -> DisplacementSolveResult:
    """Solve total-field equilibrium for fixed nodal damage.

    At ``load_parameter=0`` the applied wall traction is ``Sigma_inf n``; at
    one it is zero.  Far-field nodes always follow the affine displacement.
    For ``damage=0`` and ``residual_stiffness=0``, the returned correction field
    at one is the same mathematical linear problem as
    :func:`tunnelgeopt.elasticity.solve_plane_strain_excavation`.
    """

    controls = options or FractureSolverOptions()
    parameter = float(load_parameter)
    if not np.isfinite(parameter) or not 0.0 <= parameter <= 1.0:
        raise ValueError("load_parameter must be finite and lie in [0, 1]")
    mesh, nodes, elements = _mesh_arrays(mesh_like)
    wall_facets, farfield_facets = _named_boundaries(mesh)
    material_stress = _coerce_sigma_inf(sigma_inf)
    damage_values = np.asarray(damage, dtype=np.float64)
    if damage_values.ndim == 0:
        damage_values = np.full(nodes.shape[0], float(damage_values), dtype=np.float64)
    if damage_values.shape != (nodes.shape[0],):
        raise ValueError("damage must be scalar or have shape [node_count]")
    if (
        not np.isfinite(damage_values).all()
        or np.any(damage_values < 0.0)
        or np.any(damage_values > 1.0)
    ):
        raise ValueError("damage must be finite and lie in [0, 1]")
    gradients, area, _ = _element_geometry(nodes, elements)
    matrices = _strain_displacement_matrices(gradients)
    farfield_strain = _farfield_engineering_strain(material_stress, material)
    affine = _affine_displacement(nodes, farfield_strain)
    if initial_displacement is None:
        displacement = affine.copy()
    else:
        displacement = np.asarray(initial_displacement, dtype=np.float64).copy()
        if displacement.shape != nodes.shape or not np.isfinite(displacement).all():
            raise ValueError("initial_displacement must be finite with shape [node_count, 2]")
    farfield_nodes = np.unique(np.asarray(mesh.facets)[:, farfield_facets])
    fixed_dofs = np.column_stack((2 * farfield_nodes, 2 * farfield_nodes + 1)).ravel()
    all_dofs = np.arange(2 * nodes.shape[0], dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=False)
    if not free_dofs.size:
        raise RuntimeError("farfield constraints leave no free displacement degrees of freedom")
    displacement[farfield_nodes] = affine[farfield_nodes]
    full_wall_load = _integrated_wall_load(mesh, nodes, elements, wall_facets, material_stress)
    external_load = (1.0 - parameter) * full_wall_load
    scipy = _require_scipy()
    equilibrium_residual = np.inf
    residual_norm = np.inf
    state: tuple[FloatArray, Any | None, FloatArray, FloatArray, FloatArray, FloatArray, float]
    iterations = 0
    for iterations in range(1, controls.max_displacement_iterations + 1):
        state = _assemble_displacement_state(
            nodes,
            elements,
            area,
            matrices,
            displacement,
            damage_values,
            material,
            tangent_perturbation=controls.tangent_perturbation,
            assemble_tangent=True,
        )
        internal, tangent = state[0], state[1]
        assert tangent is not None
        residual = internal - external_load
        residual_norm = float(np.linalg.norm(residual[free_dofs]))
        scale = max(
            float(np.linalg.norm(internal[free_dofs])),
            float(np.linalg.norm(external_load[free_dofs])),
            float(np.linalg.norm(full_wall_load[free_dofs])),
            np.finfo(float).tiny,
        )
        equilibrium_residual = residual_norm / scale
        if equilibrium_residual <= controls.equilibrium_tolerance:
            break
        free_tangent = tangent[free_dofs][:, free_dofs]
        increment_free = np.asarray(
            scipy["spsolve"](free_tangent, -residual[free_dofs]), dtype=np.float64
        )
        if not np.isfinite(increment_free).all():
            raise RuntimeError("fixed-damage displacement solve produced a non-finite increment")
        base_energy = state[6] - float(external_load @ displacement.ravel())
        base_residual = residual_norm
        accepted = False
        step = 1.0
        for _ in range(controls.line_search_steps):
            candidate = displacement.copy().ravel()
            candidate[free_dofs] += step * increment_free
            candidate = candidate.reshape(nodes.shape)
            candidate[farfield_nodes] = affine[farfield_nodes]
            candidate_state = _assemble_displacement_state(
                nodes,
                elements,
                area,
                matrices,
                candidate,
                damage_values,
                material,
                tangent_perturbation=controls.tangent_perturbation,
                assemble_tangent=False,
            )
            candidate_residual = candidate_state[0] - external_load
            candidate_norm = float(np.linalg.norm(candidate_residual[free_dofs]))
            candidate_energy = candidate_state[6] - float(external_load @ candidate.ravel())
            if candidate_norm < base_residual or candidate_energy < base_energy:
                displacement = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    final_result = _evaluate_fixed_damage_displacement_state(
        mesh,
        material,
        material_stress,
        load_parameter=parameter,
        damage=damage_values,
        displacement=displacement,
        options=controls,
        iterations=iterations,
    )
    if not final_result.converged and controls.raise_on_nonconvergence:
        raise RuntimeError(
            "fixed-damage displacement solve did not converge "
            f"(iterations={iterations}, "
            f"equilibrium_residual={final_result.equilibrium_residual:.3e})"
        )
    return final_result


def at2_fracture_energy(mesh: TunnelMesh | Any, material: AT2Material, damage: ArrayLike) -> float:
    """Return the exact P1 AT2 crack-density energy per unit tunnel thickness."""

    _, nodes, elements = _mesh_arrays(mesh)
    values = np.asarray(damage, dtype=np.float64)
    if values.shape != (nodes.shape[0],) or not np.isfinite(values).all():
        raise ValueError("damage must be finite with shape [node_count]")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("damage must lie in [0, 1]")
    gradients, area, _ = _element_geometry(nodes, elements)
    local = values[elements]
    sum_squares = np.sum(local**2, axis=1)
    pair_sum = local[:, 0] * local[:, 1] + local[:, 0] * local[:, 2]
    pair_sum += local[:, 1] * local[:, 2]
    integrated_square = area * (sum_squares + pair_sum) / 6.0
    damage_gradient = np.einsum("eia,ea->ei", gradients, local, optimize=True)
    integrated_gradient_square = area * np.sum(damage_gradient**2, axis=1)
    energy = material.fracture_toughness / (2.0 * material.length_scale)
    energy *= float(np.sum(integrated_square))
    energy += (
        0.5
        * material.fracture_toughness
        * material.length_scale
        * float(np.sum(integrated_gradient_square))
    )
    return float(energy)


def _relative_change(new: FloatArray, old: FloatArray) -> float:
    return float(
        np.linalg.norm(new - old)
        / max(float(np.linalg.norm(new)), float(np.linalg.norm(old)), np.finfo(float).tiny)
    )


def solve_at2_fracture(
    mesh: TunnelMesh | Any,
    material: AT2Material,
    sigma_inf: ArrayLike,
    *,
    load_path: AT2LoadPath | None = None,
    options: FractureSolverOptions | None = None,
) -> AT2Result:
    """Run a local staggered quasistatic AT2 wall-release trajectory.

    This routine has no adaptive increment retry and no external benchmark
    validation.  With ``raise_on_nonconvergence=True`` (the default), a failed
    step raises instead of being returned as an accepted trajectory.
    """

    path = load_path or AT2LoadPath()
    controls = options or FractureSolverOptions()
    _, nodes, elements = _mesh_arrays(mesh)
    stress = _coerce_sigma_inf(sigma_inf)
    affine = _affine_displacement(nodes, _farfield_engineering_strain(stress, material))
    damage_old = np.zeros(nodes.shape[0], dtype=np.float64)
    history_old = np.zeros(elements.shape[0], dtype=np.float64)
    displacement_old = affine.copy()
    accepted_steps: list[AT2StepResult] = []

    for parameter in path.load_parameters:
        displacement_iterate = displacement_old.copy()
        damage_iterate = damage_old.copy()
        displacement_change = np.inf
        damage_change = np.inf
        displacement_iterations = 0
        damage_iterations = 0
        converged = False
        displacement_result: DisplacementSolveResult | None = None
        damage_result: DamageSolveResult | None = None
        evaluated_result: DisplacementSolveResult | None = None
        candidate_history = history_old.copy()
        staggered_iterations = 0
        for staggered_iterations in range(1, controls.max_staggered_iterations + 1):
            previous_displacement = displacement_iterate.copy()
            previous_damage = damage_iterate.copy()
            displacement_result = solve_fixed_damage_displacement(
                mesh,
                material,
                stress,
                load_parameter=parameter,
                damage=damage_iterate,
                initial_displacement=displacement_iterate,
                options=controls,
            )
            displacement_iterations += displacement_result.iterations
            displacement_iterate = displacement_result.displacement
            candidate_history = update_history(history_old, displacement_result.psi_positive)
            damage_system = assemble_at2_damage_system(mesh, material, candidate_history)
            damage_result = solve_at2_damage(
                damage_system,
                damage_old=damage_old,
                initial_damage=damage_iterate,
                options=controls,
            )
            damage_iterations += damage_result.iterations
            damage_iterate = damage_result.damage
            evaluated_result = _evaluate_fixed_damage_displacement_state(
                mesh,
                material,
                stress,
                load_parameter=parameter,
                damage=damage_iterate,
                displacement=displacement_iterate,
                options=controls,
                iterations=displacement_result.iterations,
            )
            candidate_history = update_history(history_old, evaluated_result.psi_positive)
            displacement_change = _relative_change(displacement_iterate, previous_displacement)
            damage_change = _relative_change(damage_iterate, previous_damage)
            converged = (
                displacement_result.converged
                and evaluated_result.converged
                and damage_result.converged
                and evaluated_result.equilibrium_residual <= controls.equilibrium_tolerance
                and damage_result.kkt_residual <= controls.kkt_tolerance
                and displacement_change <= controls.staggered_tolerance
                and damage_change <= controls.staggered_tolerance
            )
            if converged:
                break

        assert (
            displacement_result is not None
            and damage_result is not None
            and evaluated_result is not None
        )
        if not converged and controls.raise_on_nonconvergence:
            raise RuntimeError(
                "AT2 staggered solve did not converge "
                f"(load_parameter={parameter:.6g}, iterations={staggered_iterations}, "
                f"equilibrium={evaluated_result.equilibrium_residual:.3e}, "
                f"kkt={damage_result.kkt_residual:.3e}, "
                f"du={displacement_change:.3e}, dd={damage_change:.3e})"
            )
        fracture_energy = at2_fracture_energy(mesh, material, damage_iterate)
        total_potential = (
            evaluated_result.elastic_energy + fracture_energy - evaluated_result.external_work
        )
        step = AT2StepResult(
            load_parameter=parameter,
            displacement=displacement_iterate.copy(),
            correction_displacement=evaluated_result.correction_displacement.copy(),
            damage=damage_iterate.copy(),
            strain=evaluated_result.strain.copy(),
            stress=evaluated_result.stress.copy(),
            psi_positive=evaluated_result.psi_positive.copy(),
            psi_negative=evaluated_result.psi_negative.copy(),
            history=candidate_history.copy(),
            elastic_energy=evaluated_result.elastic_energy,
            fracture_energy=fracture_energy,
            external_work=evaluated_result.external_work,
            total_potential_energy=total_potential,
            equilibrium_residual=evaluated_result.equilibrium_residual,
            kkt_residual=damage_result.kkt_residual,
            complementarity_residual=damage_result.complementarity_residual,
            irreversibility_violation=damage_result.irreversibility_violation,
            range_violation=damage_result.range_violation,
            displacement_change=displacement_change,
            damage_change=damage_change,
            staggered_iterations=staggered_iterations,
            displacement_iterations=displacement_iterations,
            damage_iterations=damage_iterations,
            converged=converged,
        )
        accepted_steps.append(step)
        if not converged:
            break
        displacement_old = displacement_iterate.copy()
        damage_old = damage_iterate.copy()
        history_old = candidate_history.copy()

    return AT2Result(
        nodes=nodes,
        elements=elements,
        steps=tuple(accepted_steps),
        material=material,
        load_path=path,
        options=controls,
        sigma_inf=stress,
    )


# Concise aliases for callers that already encode AT2 in their configuration.
solve_damage = solve_at2_damage
solve_fracture = solve_at2_fracture


__all__ = [
    "AT2LoadPath",
    "AT2Material",
    "AT2Result",
    "AT2StepResult",
    "DamageSolveResult",
    "DamageSystem",
    "DisplacementSolveResult",
    "FractureSolverOptions",
    "SplitResponse",
    "assemble_at2_damage_system",
    "at2_fracture_energy",
    "degradation",
    "miehe_spectral_response",
    "plane_strain_spectral_split",
    "solve_at2_damage",
    "solve_at2_fracture",
    "solve_damage",
    "solve_fixed_damage_displacement",
    "solve_fracture",
    "update_history",
]
