"""Pure boundary-equilibrium and quasi-static work bookkeeping primitives.

The functions in this module intentionally do not assemble finite-element
forces or advance a fracture solve.  They turn already assembled nodal force
vectors and accepted displacement states into auditable reactions, external
work increments, and energy-balance diagnostics.

The sign convention is force *on the rock*.  With ``f_int`` the assembled
internal-force vector and ``f_wall`` the prescribed wall force on the rock,
the far-field support force on the rock is

``r_D = (f_int - f_wall)_D``.

Displacements are flattened in node-major order:
``[u_0,0, u_0,1, u_1,0, u_1,1, ...]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _readonly_float_array(value: ArrayLike, name: str, *, ndim: int) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return immutable.reshape(contiguous.shape)


def _readonly_int_array(value: ArrayLike, name: str) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    contiguous = np.ascontiguousarray(raw, dtype=np.int64)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.int64)
    return immutable.reshape(contiguous.shape)


def _readonly_computed_float_array(value: ArrayLike) -> FloatArray:
    contiguous = np.ascontiguousarray(value, dtype=np.float64)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.float64)
    return immutable.reshape(contiguous.shape)


def _readonly_computed_int_array(value: ArrayLike) -> IntArray:
    contiguous = np.ascontiguousarray(value, dtype=np.int64)
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=np.int64)
    return immutable.reshape(contiguous.shape)


def _finite_scalar(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclass(frozen=True, slots=True)
class BoundaryEquilibriumState:
    """One attempted boundary-equilibrium state with immutable array storage.

    ``farfield_prescribed_displacement`` is aligned one-to-one with the
    canonical, strictly increasing ``dirichlet_dofs`` array.  It is stored
    explicitly so an accidental mismatch between solver constraints and the
    displacement vector fails at construction instead of silently corrupting
    far-field work.

    ``accepted`` records only whether the nonlinear/load-step controller
    accepted this attempted state.  It has no effect on equilibrium
    diagnostics; cumulative path work omits rejected states.
    """

    displacement: FloatArray
    internal_force: FloatArray
    wall_nodal_force: FloatArray
    dirichlet_dofs: IntArray
    farfield_prescribed_displacement: FloatArray
    accepted: bool = True
    full_equilibrium_residual: FloatArray = field(init=False, repr=False)
    reaction_on_dirichlet_dofs: FloatArray = field(init=False, repr=False)
    reaction_full: FloatArray = field(init=False, repr=False)
    free_dofs: IntArray = field(init=False, repr=False)
    free_residual: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        displacement = _readonly_float_array(self.displacement, "displacement", ndim=2)
        if displacement.shape[1] != 2:
            raise ValueError("displacement must have shape [N, 2]")
        dof_count = 2 * displacement.shape[0]

        internal_force = _readonly_float_array(self.internal_force, "internal_force", ndim=1)
        wall_nodal_force = _readonly_float_array(self.wall_nodal_force, "wall_nodal_force", ndim=1)
        if internal_force.shape != (dof_count,):
            raise ValueError("internal_force must have shape [2N]")
        if wall_nodal_force.shape != (dof_count,):
            raise ValueError("wall_nodal_force must have shape [2N]")

        dirichlet_dofs = _readonly_int_array(self.dirichlet_dofs, "dirichlet_dofs")
        if dirichlet_dofs.size == 0:
            raise ValueError("dirichlet_dofs must be non-empty for a far-field state")
        if np.any(dirichlet_dofs < 0) or np.any(dirichlet_dofs >= dof_count):
            raise ValueError("dirichlet_dofs contains an out-of-range index")
        if dirichlet_dofs.size > 1 and np.any(np.diff(dirichlet_dofs) <= 0):
            raise ValueError("dirichlet_dofs must be unique and strictly increasing")

        prescribed = _readonly_float_array(
            self.farfield_prescribed_displacement,
            "farfield_prescribed_displacement",
            ndim=1,
        )
        if prescribed.shape != dirichlet_dofs.shape:
            raise ValueError(
                "farfield_prescribed_displacement must align one-to-one with dirichlet_dofs"
            )
        flattened_displacement = displacement.reshape(-1)
        if not np.array_equal(prescribed, flattened_displacement[dirichlet_dofs]):
            raise ValueError(
                "farfield_prescribed_displacement does not match displacement at dirichlet_dofs"
            )
        if not isinstance(self.accepted, (bool, np.bool_)):
            raise TypeError("accepted must be boolean")

        residual = internal_force - wall_nodal_force
        reaction = residual[dirichlet_dofs]
        reaction_full = np.zeros(dof_count, dtype=np.float64)
        reaction_full[dirichlet_dofs] = reaction
        free_mask = np.ones(dof_count, dtype=bool)
        free_mask[dirichlet_dofs] = False
        free_dofs = np.flatnonzero(free_mask)

        object.__setattr__(self, "displacement", displacement)
        object.__setattr__(self, "internal_force", internal_force)
        object.__setattr__(self, "wall_nodal_force", wall_nodal_force)
        object.__setattr__(self, "dirichlet_dofs", dirichlet_dofs)
        object.__setattr__(self, "farfield_prescribed_displacement", prescribed)
        object.__setattr__(self, "accepted", bool(self.accepted))
        object.__setattr__(
            self, "full_equilibrium_residual", _readonly_computed_float_array(residual)
        )
        object.__setattr__(
            self, "reaction_on_dirichlet_dofs", _readonly_computed_float_array(reaction)
        )
        object.__setattr__(self, "reaction_full", _readonly_computed_float_array(reaction_full))
        object.__setattr__(self, "free_dofs", _readonly_computed_int_array(free_dofs))
        object.__setattr__(
            self, "free_residual", _readonly_computed_float_array(residual[free_dofs])
        )

    @property
    def flattened_displacement(self) -> FloatArray:
        """Return a read-only node-major view of the displacement vector."""

        flattened = self.displacement.reshape(-1)
        flattened.setflags(write=False)
        return flattened


@dataclass(frozen=True, slots=True)
class AcceptedStepWorkIncrement:
    """Trapezoidal external-work increments between two accepted states."""

    wall_work: float
    farfield_work: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "wall_work", _finite_scalar(self.wall_work, "wall_work"))
        object.__setattr__(
            self, "farfield_work", _finite_scalar(self.farfield_work, "farfield_work")
        )

    @property
    def external_work(self) -> float:
        return self.wall_work + self.farfield_work


@dataclass(frozen=True, slots=True)
class CumulativeWorkHistory:
    """Immutable work history indexed only by accepted input states.

    Increment arrays have one entry per accepted state.  The first entry is
    zero because the first accepted state defines the path-work origin.
    """

    accepted_input_indices: IntArray
    wall_increment: FloatArray
    farfield_increment: FloatArray
    external_increment: FloatArray
    cumulative_wall_work: FloatArray
    cumulative_farfield_work: FloatArray
    cumulative_external_work: FloatArray

    def __post_init__(self) -> None:
        indices = _readonly_int_array(self.accepted_input_indices, "accepted_input_indices")
        if np.any(indices < 0):
            raise ValueError("accepted_input_indices must be nonnegative")
        if indices.size > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("accepted_input_indices must be unique and strictly increasing")
        arrays: dict[str, FloatArray] = {}
        for name in (
            "wall_increment",
            "farfield_increment",
            "external_increment",
            "cumulative_wall_work",
            "cumulative_farfield_work",
            "cumulative_external_work",
        ):
            value = _readonly_float_array(getattr(self, name), name, ndim=1)
            if value.shape != indices.shape:
                raise ValueError(f"{name} must align with accepted_input_indices")
            arrays[name] = value
        object.__setattr__(self, "accepted_input_indices", indices)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)
        if not np.array_equal(
            arrays["external_increment"], arrays["wall_increment"] + arrays["farfield_increment"]
        ):
            raise ValueError("external_increment must equal wall_increment + farfield_increment")
        if not np.array_equal(arrays["cumulative_wall_work"], np.cumsum(arrays["wall_increment"])):
            raise ValueError("cumulative_wall_work is inconsistent with wall_increment")
        if not np.array_equal(
            arrays["cumulative_farfield_work"], np.cumsum(arrays["farfield_increment"])
        ):
            raise ValueError("cumulative_farfield_work is inconsistent with farfield_increment")
        if not np.array_equal(
            arrays["cumulative_external_work"], np.cumsum(arrays["external_increment"])
        ):
            raise ValueError("cumulative_external_work is inconsistent with external_increment")


@dataclass(frozen=True, slots=True)
class EnergyIncrementDiagnostic:
    """Quasi-static total-energy increment compared with external work."""

    previous_total_energy: float
    current_total_energy: float
    external_work_increment: float
    normalization_floor: float
    total_energy_increment: float = field(init=False)
    signed_imbalance: float = field(init=False)
    absolute_imbalance: float = field(init=False)
    relative_imbalance: float = field(init=False)

    def __post_init__(self) -> None:
        previous = _finite_scalar(self.previous_total_energy, "previous_total_energy")
        current = _finite_scalar(self.current_total_energy, "current_total_energy")
        work = _finite_scalar(self.external_work_increment, "external_work_increment")
        normalization_floor = _finite_scalar(self.normalization_floor, "normalization_floor")
        if previous < 0.0 or current < 0.0:
            raise ValueError("total energies must be nonnegative")
        if normalization_floor <= 0.0:
            raise ValueError("normalization_floor must be positive")
        energy_increment = current - previous
        signed_imbalance = energy_increment - work
        absolute_imbalance = abs(signed_imbalance)
        scale = max(abs(energy_increment), abs(work), normalization_floor)
        relative_imbalance = absolute_imbalance / scale
        object.__setattr__(self, "previous_total_energy", previous)
        object.__setattr__(self, "current_total_energy", current)
        object.__setattr__(self, "external_work_increment", work)
        object.__setattr__(self, "normalization_floor", normalization_floor)
        object.__setattr__(self, "total_energy_increment", energy_increment)
        object.__setattr__(self, "signed_imbalance", signed_imbalance)
        object.__setattr__(self, "absolute_imbalance", absolute_imbalance)
        object.__setattr__(self, "relative_imbalance", relative_imbalance)


def _require_step_alignment(
    previous: BoundaryEquilibriumState, current: BoundaryEquilibriumState
) -> None:
    if previous.displacement.shape != current.displacement.shape:
        raise ValueError("accepted states must have matching displacement shapes")
    if not np.array_equal(previous.dirichlet_dofs, current.dirichlet_dofs):
        raise ValueError("accepted states must have identical aligned dirichlet_dofs")


def accepted_step_work_increment(
    previous: BoundaryEquilibriumState,
    current: BoundaryEquilibriumState,
) -> AcceptedStepWorkIncrement:
    """Integrate wall and far-field work over one accepted step.

    The rule is trapezoidal in the nodal generalized forces.  It is exact for a
    linear one-degree-of-freedom spring under monotone prescribed displacement,
    but is only a convergent quadrature for general nonlinear paths.
    """

    if not previous.accepted or not current.accepted:
        raise ValueError("accepted_step_work_increment requires two accepted states")
    _require_step_alignment(previous, current)
    displacement_increment = current.flattened_displacement - previous.flattened_displacement
    wall_work = 0.5 * np.dot(
        previous.wall_nodal_force + current.wall_nodal_force,
        displacement_increment,
    )
    farfield_increment = (
        current.farfield_prescribed_displacement - previous.farfield_prescribed_displacement
    )
    farfield_work = 0.5 * np.dot(
        previous.reaction_on_dirichlet_dofs + current.reaction_on_dirichlet_dofs,
        farfield_increment,
    )
    return AcceptedStepWorkIncrement(float(wall_work), float(farfield_work))


def cumulative_accepted_work(
    attempted_states: Iterable[BoundaryEquilibriumState],
) -> CumulativeWorkHistory:
    """Recompute path work from accepted states without mutating prior history.

    Rejected attempts are discarded before adjacent accepted states are paired.
    Consequently a failed trial can be rolled back without subtracting an
    already accumulated increment or retaining a trial-force contribution.
    """

    states = tuple(attempted_states)
    if any(not isinstance(state, BoundaryEquilibriumState) for state in states):
        raise TypeError("attempted_states must contain BoundaryEquilibriumState values")
    accepted_pairs = tuple((index, state) for index, state in enumerate(states) if state.accepted)
    accepted_indices = np.asarray([index for index, _ in accepted_pairs], dtype=np.int64)
    accepted_states = tuple(state for _, state in accepted_pairs)
    count = len(accepted_states)
    wall = np.zeros(count, dtype=np.float64)
    farfield = np.zeros(count, dtype=np.float64)
    for index in range(1, count):
        increment = accepted_step_work_increment(accepted_states[index - 1], accepted_states[index])
        wall[index] = increment.wall_work
        farfield[index] = increment.farfield_work
    external = wall + farfield
    return CumulativeWorkHistory(
        accepted_input_indices=accepted_indices,
        wall_increment=wall,
        farfield_increment=farfield,
        external_increment=external,
        cumulative_wall_work=np.cumsum(wall),
        cumulative_farfield_work=np.cumsum(farfield),
        cumulative_external_work=np.cumsum(external),
    )


def energy_increment_diagnostic(
    previous_total_energy: float,
    current_total_energy: float,
    external_work_increment: float,
    *,
    normalization_floor: float,
) -> EnergyIncrementDiagnostic:
    """Return a scale-free quasi-static energy-balance diagnostic.

    ``total_energy`` must be the caller's consistently assembled recoverable
    elastic plus fracture-surface energy.  The relative denominator is the
    larger magnitude of the energy increment, external work, and an explicitly
    supplied positive dimensional floor.  The caller, rather than this module,
    owns the physical scale used for that floor.
    """

    previous = _finite_scalar(previous_total_energy, "previous_total_energy")
    current = _finite_scalar(current_total_energy, "current_total_energy")
    work = _finite_scalar(external_work_increment, "external_work_increment")
    floor = _finite_scalar(normalization_floor, "normalization_floor")
    if previous < 0.0 or current < 0.0:
        raise ValueError("total energies must be nonnegative")
    if floor <= 0.0:
        raise ValueError("normalization_floor must be positive")
    return EnergyIncrementDiagnostic(
        previous_total_energy=previous,
        current_total_energy=current,
        external_work_increment=work,
        normalization_floor=floor,
    )


def neumann_load_functional(state: BoundaryEquilibriumState) -> float:
    """Return the instantaneous wall functional ``f_wall . u``.

    This is deliberately separate from accepted-step work: when wall forces
    vary along a path, the endpoint functional is not cumulative external work.
    """

    if not isinstance(state, BoundaryEquilibriumState):
        raise TypeError("state must be a BoundaryEquilibriumState")
    return float(np.dot(state.wall_nodal_force, state.flattened_displacement))
