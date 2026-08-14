"""Development-only prescribed-displacement AT2 boundary-value problems.

This module deliberately sits beside, rather than inside, the tunnel
wall-release solver.  It provides the explicit displacement-control contract
needed by coupon benchmarks while leaving the established tunnel schedule and
its output types untouched.

Mesh-node columns follow the repository convention ``[y, z]``.  All
displacement degrees of freedom use its node-major ordering:
``u_y(node i) = 2*i`` and ``u_z(node i) = 2*i + 1``.  The generic solver acts
on component 0/1 indices; a benchmark adapter assigns their boundary meaning.
Reactions use the support-on-rock sign convention
``r = f_internal - f_applied``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .fracture import (
    AT2Material,
    DamageSolveResult,
    FractureSolverOptions,
    _assemble_displacement_state,
    _element_geometry,
    _mesh_arrays,
    _relative_change,
    _strain_displacement_matrices,
    _symmetric_relative_energy_change,
    assemble_at2_damage_system,
    at2_fracture_energy,
    solve_at2_damage,
    update_history,
)
from .mesh import TunnelMesh

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]


def _readonly_float(value: ArrayLike, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _readonly_int(value: ArrayLike, *, name: str) -> IntArray:
    source = np.asarray(value)
    if source.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if source.dtype.kind not in "iu" or source.dtype.kind == "b":
        raise TypeError(f"{name} must contain integer DOF indices")
    array = np.array(source, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _validate_identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def prescribed_displacement_mesh_identity(mesh_like: TunnelMesh | Any) -> str:
    """Hash the exact ordered nodes and triangles used by the BVP solver.

    This is a discrete-mesh identity, not a geometry-family identity.  Node or
    element reordering intentionally changes the digest because it changes the
    node-major displacement and damage DOF identities.
    """

    _, nodes, elements = _mesh_arrays(mesh_like)
    digest = hashlib.sha256()
    digest.update(b"tunnelgeopt.prescribed-displacement-mesh.v1\0")
    for array, dtype in ((nodes, "<f8"), (elements, "<i8")):
        canonical = np.ascontiguousarray(array, dtype=dtype)
        shape = np.asarray(canonical.shape, dtype="<i8")
        digest.update(np.asarray([canonical.ndim], dtype="<i8").tobytes())
        digest.update(shape.tobytes())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _freeze_groups(
    groups: Mapping[str, ArrayLike], dirichlet_dofs: IntArray
) -> Mapping[str, IntArray]:
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("reaction_groups must be a non-empty mapping")
    frozen: dict[str, IntArray] = {}
    for key, values in groups.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("reaction group names must be non-empty strings")
        dofs = _readonly_int(values, name=f"reaction_groups[{key!r}]")
        if dofs.size == 0:
            raise ValueError(f"reaction_groups[{key!r}] must not be empty")
        if np.any(np.diff(dofs) <= 0):
            raise ValueError(f"reaction_groups[{key!r}] must be exactly increasing")
        if not np.all(np.isin(dofs, dirichlet_dofs)):
            raise ValueError(f"reaction_groups[{key!r}] must be a subset of dirichlet_dofs")
        frozen[key] = dofs
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class PrescribedDisplacementState:
    """One exact displacement-control state, independent of boundary names.

    ``external_force`` is the complete node-major applied-force vector.  For
    the pure-Dirichlet SENT/SENS use case it is exactly zero.  Every reaction
    group contains actual constrained DOF indices, and ``driven_group`` names
    the group whose reaction sum is reported as the scalar generalized load.
    """

    identity: str
    mesh_identity: str
    sequence_index: int
    path_parameter: float
    dirichlet_dofs: IntArray
    dirichlet_values: FloatArray
    external_force: FloatArray
    reaction_groups: Mapping[str, IntArray]
    driven_group: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _validate_identity(self.identity, "identity"))
        mesh_identity = _validate_identity(self.mesh_identity, "mesh_identity")
        if len(mesh_identity) != 64 or any(
            character not in "0123456789abcdef" for character in mesh_identity
        ):
            raise ValueError("mesh_identity must be a lowercase SHA-256 digest")
        object.__setattr__(self, "mesh_identity", mesh_identity)
        if (
            not isinstance(self.sequence_index, int)
            or isinstance(self.sequence_index, bool)
            or self.sequence_index < 0
        ):
            raise ValueError("sequence_index must be a nonnegative integer")
        parameter = float(self.path_parameter)
        if not np.isfinite(parameter):
            raise ValueError("path_parameter must be finite")
        object.__setattr__(self, "path_parameter", parameter)

        dofs = _readonly_int(self.dirichlet_dofs, name="dirichlet_dofs")
        if dofs.size == 0:
            raise ValueError("dirichlet_dofs must not be empty")
        if dofs[0] < 0 or np.any(np.diff(dofs) <= 0):
            raise ValueError("dirichlet_dofs must be nonnegative and exactly increasing")
        values = _readonly_float(self.dirichlet_values, name="dirichlet_values", ndim=1)
        if values.shape != dofs.shape:
            raise ValueError("dirichlet_values must align one-to-one with dirichlet_dofs")
        external = _readonly_float(self.external_force, name="external_force", ndim=1)
        if external.size == 0 or external.size % 2:
            raise ValueError("external_force must contain exactly two DOFs per node")
        if dofs[-1] >= external.size:
            raise ValueError("dirichlet_dofs contains an index missing from external_force")
        groups = _freeze_groups(self.reaction_groups, dofs)
        driven = _validate_identity(self.driven_group, "driven_group")
        if driven not in groups:
            raise ValueError("driven_group must name one of reaction_groups")

        object.__setattr__(self, "dirichlet_dofs", dofs)
        object.__setattr__(self, "dirichlet_values", values)
        object.__setattr__(self, "external_force", external)
        object.__setattr__(self, "reaction_groups", groups)
        object.__setattr__(self, "driven_group", driven)


@dataclass(frozen=True)
class FixedDamageDisplacementBVPResult:
    """Equilibrium and reaction audit for one fixed-damage BVP state."""

    state: PrescribedDisplacementState
    mesh_identity: str
    displacement: FloatArray
    damage: FloatArray
    strain: FloatArray
    stress: FloatArray
    psi_positive: FloatArray
    psi_negative: FloatArray
    elastic_energy: float
    neumann_load_functional: float
    total_potential_energy: float
    internal_force: FloatArray
    external_force: FloatArray
    reaction: FloatArray
    dirichlet_dofs: IntArray
    prescribed_displacement: FloatArray
    reaction_groups: Mapping[str, IntArray]
    driven_group: str
    generalized_load: float
    residual_norm: float
    equilibrium_residual: float
    iterations: int
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "displacement",
            "damage",
            "strain",
            "stress",
            "psi_positive",
            "psi_negative",
            "internal_force",
            "external_force",
            "reaction",
            "prescribed_displacement",
        ):
            object.__setattr__(self, name, _readonly_float(getattr(self, name), name=name))
        dofs = _readonly_int(self.dirichlet_dofs, name="dirichlet_dofs")
        object.__setattr__(self, "dirichlet_dofs", dofs)
        object.__setattr__(self, "reaction_groups", _freeze_groups(self.reaction_groups, dofs))

    @property
    def reaction_on_dirichlet_dofs(self) -> FloatArray:
        values = np.array(self.reaction[self.dirichlet_dofs], copy=True)
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class AT2DirichletStepResult:
    """One converged, or explicitly reported failed, staggered path state."""

    state: PrescribedDisplacementState
    displacement: FloatArray
    damage: FloatArray
    strain: FloatArray
    stress: FloatArray
    psi_positive: FloatArray
    psi_negative: FloatArray
    history: FloatArray
    elastic_energy: float
    fracture_energy: float
    neumann_load_functional: float
    total_potential_energy: float
    internal_force: FloatArray
    external_force: FloatArray
    reaction: FloatArray
    dirichlet_dofs: IntArray
    prescribed_displacement: FloatArray
    reaction_groups: Mapping[str, IntArray]
    driven_group: str
    generalized_load: float
    path_work_increment: float
    path_work: float
    equilibrium_residual: float
    kkt_residual: float
    complementarity_residual: float
    irreversibility_violation: float
    range_violation: float
    displacement_change: float
    damage_change: float
    energy_change: float
    staggered_iterations: int
    displacement_iterations: int
    damage_iterations: int
    converged: bool

    def __post_init__(self) -> None:
        for name in (
            "displacement",
            "damage",
            "strain",
            "stress",
            "psi_positive",
            "psi_negative",
            "history",
            "internal_force",
            "external_force",
            "reaction",
            "prescribed_displacement",
        ):
            object.__setattr__(self, name, _readonly_float(getattr(self, name), name=name))
        dofs = _readonly_int(self.dirichlet_dofs, name="dirichlet_dofs")
        object.__setattr__(self, "dirichlet_dofs", dofs)
        object.__setattr__(self, "reaction_groups", _freeze_groups(self.reaction_groups, dofs))

    @property
    def reaction_on_dirichlet_dofs(self) -> FloatArray:
        values = np.array(self.reaction[self.dirichlet_dofs], copy=True)
        values.setflags(write=False)
        return values


@dataclass(frozen=True)
class AT2DirichletPathResult:
    """Immutable development trajectory for prescribed-displacement states."""

    nodes: FloatArray
    elements: IntArray
    mesh_identity: str
    steps: tuple[AT2DirichletStepResult, ...]
    material: AT2Material
    options: FractureSolverOptions

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", _readonly_float(self.nodes, name="nodes", ndim=2))
        elements = np.array(self.elements, dtype=np.int64, copy=True)
        if elements.ndim != 2 or elements.shape[1] != 3:
            raise ValueError("elements must have shape [element_count, 3]")
        elements.setflags(write=False)
        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "steps", tuple(self.steps))

    @property
    def final(self) -> AT2DirichletStepResult:
        if not self.steps:
            raise RuntimeError("the path contains no solved state")
        return self.steps[-1]


def _coerce_damage(value: ArrayLike, node_count: int, *, name: str) -> FloatArray:
    damage = np.asarray(value, dtype=np.float64)
    if damage.ndim == 0:
        damage = np.full(node_count, float(damage), dtype=np.float64)
    if damage.shape != (node_count,):
        raise ValueError(f"{name} must be scalar or have shape [node_count]")
    if not np.isfinite(damage).all() or np.any(damage < 0.0) or np.any(damage > 1.0):
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return damage.copy()


def _coerce_history(value: ArrayLike, element_count: int) -> FloatArray:
    history = np.asarray(value, dtype=np.float64)
    if history.ndim == 0:
        history = np.full(element_count, float(history), dtype=np.float64)
    if history.shape != (element_count,):
        raise ValueError("initial_history must be scalar or have shape [element_count]")
    if not np.isfinite(history).all() or np.any(history < 0.0):
        raise ValueError("initial_history must be finite and nonnegative")
    return history.copy()


def _validate_state_on_mesh(
    state: PrescribedDisplacementState,
    mesh_identity: str,
    nodes: FloatArray,
) -> tuple[IntArray, IntArray]:
    if not isinstance(state, PrescribedDisplacementState):
        raise TypeError("state must be a PrescribedDisplacementState")
    if state.mesh_identity != mesh_identity:
        raise ValueError("state mesh_identity does not match the exact solver mesh")
    dof_count = 2 * nodes.shape[0]
    if state.external_force.shape != (dof_count,):
        raise ValueError("state external_force does not match the solver mesh DOF count")
    if state.dirichlet_dofs[-1] >= dof_count:
        raise ValueError("state dirichlet_dofs contains an index missing from the solver mesh")
    all_dofs = np.arange(dof_count, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, state.dirichlet_dofs, assume_unique=True)

    # A BVP with unconstrained rigid translation or rotation can report a zero
    # residual at a singular state.  Reject it before Newton or reaction use.
    centered = nodes - nodes.mean(axis=0)
    rigid_modes = np.zeros((dof_count, 3), dtype=np.float64)
    rigid_modes[0::2, 0] = 1.0
    rigid_modes[1::2, 1] = 1.0
    rigid_modes[0::2, 2] = -centered[:, 1]
    rigid_modes[1::2, 2] = centered[:, 0]
    mode_norms = np.linalg.norm(rigid_modes, axis=0)
    if np.any(mode_norms <= np.finfo(float).tiny):
        raise ValueError("mesh cannot define three independent planar rigid-body modes")
    rigid_modes /= mode_norms
    if np.linalg.matrix_rank(rigid_modes[state.dirichlet_dofs]) < 3:
        raise ValueError("dirichlet_dofs are missing constraints for a rigid-body mode")
    return state.dirichlet_dofs, free_dofs


def _evaluate_prescribed_displacement_state(
    mesh_like: TunnelMesh | Any,
    material: AT2Material,
    state: PrescribedDisplacementState,
    *,
    damage: ArrayLike,
    displacement: ArrayLike,
    options: FractureSolverOptions,
    iterations: int,
) -> FixedDamageDisplacementBVPResult:
    _, nodes, elements = _mesh_arrays(mesh_like)
    mesh_identity = prescribed_displacement_mesh_identity(mesh_like)
    fixed_dofs, free_dofs = _validate_state_on_mesh(state, mesh_identity, nodes)
    damage_values = _coerce_damage(damage, nodes.shape[0], name="damage")
    # Force C order: node-major DOF assignment must mutate the actual [N, 2]
    # array even when mesh coordinates originated from a Fortran-ordered p.T.
    displacement_values = np.array(displacement, dtype=np.float64, order="C", copy=True)
    if displacement_values.shape != nodes.shape or not np.isfinite(displacement_values).all():
        raise ValueError("displacement must be finite with shape [node_count, 2]")
    displacement_values.ravel()[fixed_dofs] = state.dirichlet_values

    gradients, area, _ = _element_geometry(nodes, elements)
    matrices = _strain_displacement_matrices(gradients)
    assembled = _assemble_displacement_state(
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
    internal = np.asarray(assembled[0], dtype=np.float64)
    external = np.asarray(state.external_force, dtype=np.float64)
    residual = internal - external
    residual_norm = float(np.linalg.norm(residual[free_dofs]))
    scale = max(
        float(np.linalg.norm(internal)),
        float(np.linalg.norm(external)),
        np.finfo(float).tiny,
    )
    equilibrium_residual = residual_norm / scale
    neumann = float(external @ displacement_values.ravel())
    elastic = float(assembled[6])
    driven_dofs = state.reaction_groups[state.driven_group]
    generalized_load = float(np.sum(residual[driven_dofs]))
    return FixedDamageDisplacementBVPResult(
        state=state,
        mesh_identity=mesh_identity,
        displacement=displacement_values,
        damage=damage_values,
        strain=assembled[2],
        stress=assembled[3],
        psi_positive=assembled[4],
        psi_negative=assembled[5],
        elastic_energy=elastic,
        neumann_load_functional=neumann,
        total_potential_energy=elastic - neumann,
        internal_force=internal,
        external_force=external,
        reaction=residual,
        dirichlet_dofs=fixed_dofs,
        prescribed_displacement=displacement_values.ravel()[fixed_dofs],
        reaction_groups=state.reaction_groups,
        driven_group=state.driven_group,
        generalized_load=generalized_load,
        residual_norm=residual_norm,
        equilibrium_residual=equilibrium_residual,
        iterations=iterations,
        converged=equilibrium_residual <= options.equilibrium_tolerance,
    )


def solve_fixed_damage_displacement_bvp(
    mesh_like: TunnelMesh | Any,
    material: AT2Material,
    state: PrescribedDisplacementState,
    *,
    damage: ArrayLike,
    initial_displacement: ArrayLike | None = None,
    options: FractureSolverOptions | None = None,
) -> FixedDamageDisplacementBVPResult:
    """Solve equilibrium for exact prescribed DOFs and fixed nodal damage.

    Applied nodal forces are allowed on free or prescribed DOFs.  The reported
    complete residual is the support-on-rock reaction convention; only its
    entries on prescribed DOFs are physical support reactions.
    """

    if not isinstance(material, AT2Material):
        raise TypeError("material must be an AT2Material")
    controls = options or FractureSolverOptions()
    _, nodes, elements = _mesh_arrays(mesh_like)
    mesh_identity = prescribed_displacement_mesh_identity(mesh_like)
    fixed_dofs, free_dofs = _validate_state_on_mesh(state, mesh_identity, nodes)
    damage_values = _coerce_damage(damage, nodes.shape[0], name="damage")
    if initial_displacement is None:
        displacement = np.zeros(nodes.shape, dtype=np.float64, order="C")
    else:
        displacement = np.array(initial_displacement, dtype=np.float64, order="C", copy=True)
        if displacement.shape != nodes.shape or not np.isfinite(displacement).all():
            raise ValueError("initial_displacement must be finite with shape [node_count, 2]")
    displacement.ravel()[fixed_dofs] = state.dirichlet_values

    gradients, area, _ = _element_geometry(nodes, elements)
    matrices = _strain_displacement_matrices(gradients)
    external = np.asarray(state.external_force, dtype=np.float64)
    iterations = 0
    for iterations in range(1, controls.max_displacement_iterations + 1):
        assembled = _assemble_displacement_state(
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
        internal, tangent = assembled[0], assembled[1]
        assert tangent is not None
        residual = internal - external
        residual_norm = float(np.linalg.norm(residual[free_dofs]))
        scale = max(
            float(np.linalg.norm(internal)),
            float(np.linalg.norm(external)),
            np.finfo(float).tiny,
        )
        equilibrium_residual = residual_norm / scale

        if free_dofs.size:
            free_tangent = tangent[free_dofs][:, free_dofs]
            # Factorization is required even for a zero residual so a singular
            # fixed-damage state cannot masquerade as equilibrium.
            try:
                from scipy.sparse.linalg import splu  # type: ignore[import-not-found]

                factor = splu(free_tangent.tocsc())
            except RuntimeError as exc:
                raise RuntimeError("prescribed-displacement tangent is singular") from exc
        else:
            factor = None

        if equilibrium_residual <= controls.equilibrium_tolerance:
            break
        assert factor is not None
        increment_free = np.asarray(factor.solve(-residual[free_dofs]), dtype=np.float64)
        if not np.isfinite(increment_free).all():
            raise RuntimeError("prescribed-displacement solve produced a non-finite increment")
        base_potential = float(assembled[6]) - float(external @ displacement.ravel())
        accepted = False
        step_length = 1.0
        for _ in range(controls.line_search_steps):
            candidate = displacement.copy().ravel()
            candidate[free_dofs] += step_length * increment_free
            candidate[fixed_dofs] = state.dirichlet_values
            candidate = candidate.reshape(nodes.shape)
            candidate_assembled = _assemble_displacement_state(
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
            candidate_residual = candidate_assembled[0] - external
            candidate_norm = float(np.linalg.norm(candidate_residual[free_dofs]))
            candidate_potential = float(candidate_assembled[6]) - float(
                external @ candidate.ravel()
            )
            if candidate_norm < residual_norm or candidate_potential < base_potential:
                displacement = candidate
                accepted = True
                break
            step_length *= 0.5
        if not accepted:
            break

    result = _evaluate_prescribed_displacement_state(
        mesh_like,
        material,
        state,
        damage=damage_values,
        displacement=displacement,
        options=controls,
        iterations=iterations,
    )
    if not result.converged and controls.raise_on_nonconvergence:
        raise RuntimeError(
            "prescribed-displacement solve did not converge "
            f"(identity={state.identity!r}, iterations={iterations}, "
            f"equilibrium_residual={result.equilibrium_residual:.3e})"
        )
    return result


def _same_group_contract(left: Mapping[str, IntArray], right: Mapping[str, IntArray]) -> bool:
    return left.keys() == right.keys() and all(
        np.array_equal(left[key], right[key]) for key in left
    )


def _validate_path_states(
    states: Sequence[PrescribedDisplacementState], mesh_identity: str, nodes: FloatArray
) -> tuple[PrescribedDisplacementState, ...]:
    path = tuple(states)
    if not path:
        raise ValueError("states must be a non-empty sequence")
    identities: set[str] = set()
    first: PrescribedDisplacementState | None = None
    previous_parameter: float | None = None
    for expected_index, state in enumerate(path):
        _validate_state_on_mesh(state, mesh_identity, nodes)
        if state.sequence_index != expected_index:
            raise ValueError("state sequence_index values must be exactly 0, 1, ...")
        if state.identity in identities:
            raise ValueError("state identities must be unique along a path")
        identities.add(state.identity)
        if previous_parameter is not None and state.path_parameter <= previous_parameter:
            raise ValueError("state path_parameter values must be strictly increasing")
        previous_parameter = state.path_parameter
        if first is None:
            first = state
            continue
        if not np.array_equal(state.dirichlet_dofs, first.dirichlet_dofs):
            raise ValueError("all path states must use identical dirichlet_dofs")
        if state.driven_group != first.driven_group or not _same_group_contract(
            state.reaction_groups, first.reaction_groups
        ):
            raise ValueError("all path states must use identical reaction groups and driven_group")
    return path


def solve_at2_dirichlet_path(
    mesh_like: TunnelMesh | Any,
    material: AT2Material,
    states: Sequence[PrescribedDisplacementState],
    *,
    initial_damage: ArrayLike | None = None,
    initial_history: ArrayLike | None = None,
    options: FractureSolverOptions | None = None,
) -> AT2DirichletPathResult:
    """Run a development-only staggered AT2 displacement-control path.

    Irreversibility is anchored to the preceding accepted state.  Fixed-load
    potential is ``elastic + fracture - external_force @ displacement`` and is
    therefore exactly ``elastic + fracture`` for zero applied force.  Path
    work is the trapezoidal integral of the complete prescribed-DOF reaction
    against the complete prescribed-displacement increment.
    """

    if not isinstance(material, AT2Material):
        raise TypeError("material must be an AT2Material")
    controls = options or FractureSolverOptions()
    _, nodes, elements = _mesh_arrays(mesh_like)
    mesh_identity = prescribed_displacement_mesh_identity(mesh_like)
    path = _validate_path_states(states, mesh_identity, nodes)
    damage_old = (
        np.zeros(nodes.shape[0], dtype=np.float64)
        if initial_damage is None
        else _coerce_damage(initial_damage, nodes.shape[0], name="initial_damage")
    )
    history_old = (
        np.zeros(elements.shape[0], dtype=np.float64)
        if initial_history is None
        else _coerce_history(initial_history, elements.shape[0])
    )
    displacement_old = np.zeros(nodes.shape, dtype=np.float64, order="C")
    accepted_steps: list[AT2DirichletStepResult] = []
    cumulative_work = 0.0

    for prescribed_state in path:
        displacement_iterate = displacement_old.copy()
        displacement_iterate.ravel()[prescribed_state.dirichlet_dofs] = (
            prescribed_state.dirichlet_values
        )
        damage_iterate = damage_old.copy()
        displacement_change = np.inf
        damage_change = np.inf
        energy_change = np.inf
        previous_potential: float | None = None
        fracture_energy = np.inf
        displacement_iterations = 0
        damage_iterations = 0
        converged = False
        displacement_result: FixedDamageDisplacementBVPResult | None = None
        evaluated_result: FixedDamageDisplacementBVPResult | None = None
        damage_result: DamageSolveResult | None = None
        candidate_history = history_old.copy()
        staggered_iterations = 0

        for staggered_iterations in range(1, controls.max_staggered_iterations + 1):
            previous_displacement = displacement_iterate.copy()
            previous_damage = damage_iterate.copy()
            displacement_result = solve_fixed_damage_displacement_bvp(
                mesh_like,
                material,
                prescribed_state,
                damage=damage_iterate,
                initial_displacement=displacement_iterate,
                options=controls,
            )
            displacement_iterations += displacement_result.iterations
            displacement_iterate = np.asarray(displacement_result.displacement).copy()
            candidate_history = update_history(history_old, displacement_result.psi_positive)
            damage_result = solve_at2_damage(
                assemble_at2_damage_system(mesh_like, material, candidate_history),
                damage_old=damage_old,
                initial_damage=damage_iterate,
                options=controls,
            )
            damage_iterations += damage_result.iterations
            damage_iterate = damage_result.damage
            evaluated_result = _evaluate_prescribed_displacement_state(
                mesh_like,
                material,
                prescribed_state,
                damage=damage_iterate,
                displacement=displacement_iterate,
                options=controls,
                iterations=displacement_result.iterations,
            )
            candidate_history = update_history(history_old, evaluated_result.psi_positive)
            displacement_change = _relative_change(displacement_iterate, previous_displacement)
            damage_change = _relative_change(damage_iterate, previous_damage)
            fracture_energy = at2_fracture_energy(mesh_like, material, damage_iterate)
            current_potential = (
                evaluated_result.elastic_energy
                + fracture_energy
                - evaluated_result.neumann_load_functional
            )
            energy_change = _symmetric_relative_energy_change(current_potential, previous_potential)
            previous_potential = current_potential
            converged = (
                displacement_result.converged
                and evaluated_result.converged
                and damage_result.converged
                and evaluated_result.equilibrium_residual <= controls.equilibrium_tolerance
                and damage_result.kkt_residual <= controls.kkt_tolerance
                and displacement_change <= controls.staggered_tolerance
                and damage_change <= controls.staggered_tolerance
                and energy_change <= controls.energy_tolerance
            )
            if converged:
                break

        assert evaluated_result is not None and damage_result is not None
        assert previous_potential is not None
        if not converged and controls.raise_on_nonconvergence:
            raise RuntimeError(
                "prescribed-displacement AT2 staggered solve did not converge "
                f"(identity={prescribed_state.identity!r}, "
                f"iterations={staggered_iterations}, "
                f"equilibrium={evaluated_result.equilibrium_residual:.3e}, "
                f"kkt={damage_result.kkt_residual:.3e}, "
                f"du={displacement_change:.3e}, dd={damage_change:.3e}, "
                f"potential_energy_change={energy_change:.3e})"
            )

        work_increment = 0.0
        if accepted_steps:
            previous = accepted_steps[-1]
            fixed_dofs = prescribed_state.dirichlet_dofs
            displacement_increment = (
                evaluated_result.displacement.ravel()[fixed_dofs]
                - previous.displacement.ravel()[fixed_dofs]
            )
            average_reaction = 0.5 * (
                evaluated_result.reaction[fixed_dofs] + previous.reaction[fixed_dofs]
            )
            work_increment = float(average_reaction @ displacement_increment)
            cumulative_work += work_increment

        step = AT2DirichletStepResult(
            state=prescribed_state,
            displacement=evaluated_result.displacement,
            damage=damage_iterate,
            strain=evaluated_result.strain,
            stress=evaluated_result.stress,
            psi_positive=evaluated_result.psi_positive,
            psi_negative=evaluated_result.psi_negative,
            history=candidate_history,
            elastic_energy=evaluated_result.elastic_energy,
            fracture_energy=fracture_energy,
            neumann_load_functional=evaluated_result.neumann_load_functional,
            total_potential_energy=previous_potential,
            internal_force=evaluated_result.internal_force,
            external_force=evaluated_result.external_force,
            reaction=evaluated_result.reaction,
            dirichlet_dofs=evaluated_result.dirichlet_dofs,
            prescribed_displacement=evaluated_result.prescribed_displacement,
            reaction_groups=evaluated_result.reaction_groups,
            driven_group=evaluated_result.driven_group,
            generalized_load=evaluated_result.generalized_load,
            path_work_increment=work_increment,
            path_work=cumulative_work,
            equilibrium_residual=evaluated_result.equilibrium_residual,
            kkt_residual=damage_result.kkt_residual,
            complementarity_residual=damage_result.complementarity_residual,
            irreversibility_violation=damage_result.irreversibility_violation,
            range_violation=damage_result.range_violation,
            displacement_change=displacement_change,
            damage_change=damage_change,
            energy_change=energy_change,
            staggered_iterations=staggered_iterations,
            displacement_iterations=displacement_iterations,
            damage_iterations=damage_iterations,
            converged=converged,
        )
        accepted_steps.append(step)
        if not converged:
            break
        displacement_old = np.asarray(evaluated_result.displacement).copy()
        damage_old = np.asarray(damage_iterate).copy()
        history_old = np.asarray(candidate_history).copy()

    return AT2DirichletPathResult(
        nodes=nodes,
        elements=elements,
        mesh_identity=mesh_identity,
        steps=tuple(accepted_steps),
        material=material,
        options=controls,
    )


__all__ = [
    "AT2DirichletPathResult",
    "AT2DirichletStepResult",
    "FixedDamageDisplacementBVPResult",
    "PrescribedDisplacementState",
    "prescribed_displacement_mesh_identity",
    "solve_at2_dirichlet_path",
    "solve_fixed_damage_displacement_bvp",
]
