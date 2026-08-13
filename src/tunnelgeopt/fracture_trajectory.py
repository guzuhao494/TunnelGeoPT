"""Strict Phase-1 scheduled-solver to C-fracture-schema development adapter.

This module intentionally does not authorize formal Phase-1 label generation.
It closes the mechanical bookkeeping gap between ``ScheduledAT2Result`` and
the v3 ``FractureTrajectory`` contract for one development trajectory:

* every attempted coordinate is evaluated by ``Phase1LoadSchedule.state_at``;
* stress, exact wall-facet ID order, and facet release are compared bitwise;
* a rejected target is retried by halving the increment from the last accepted
  coordinate;
* the default solver is invoked on a fresh complete accepted prefix for every
  attempt, so a rejected iterate cannot mutate accepted ``u``, ``d`` or ``H``;
* wall and far-field work is integrated over every accepted internal step;
* the 41 required output coordinates are retained as an explicit index map;
* reactions and global force/moment residuals are derived from assembled force
  arrays, never supplied as placeholders; and
* the frozen potential-energy convergence tolerance is passed to the solver,
  independently rechecked on every accepted state, and recorded in metadata;
  and
* the resulting immutable schema-v3 record is validated before it is returned.

The returned record remains development evidence only.  The fixed-state
same-mesh MOOSE cross-check is now closed independently, but the fine-mesh,
coupled SENT/SENS fracture-benchmark, and protocol-scale resource/publication
prerequisites still prevent formal Phase-1 label generation.
"""

from __future__ import annotations

import json
import math
import platform
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .elasticity import compute_element_strain
from .fracture import (
    AT2LoadPath,
    AT2Material,
    FractureSolverOptions,
    ScheduledAT2Result,
    ScheduledAT2StepResult,
    miehe_spectral_response,
    solve_at2_fracture_schedule,
)
from .fracture_loading import FractureLoadState, Phase1LoadSchedule
from .fracture_schema import (
    FractureTrajectory,
    FractureTrajectoryPaths,
    compute_load_state_sha256,
    load_fracture_trajectory,
    save_fracture_trajectory,
)
from .fracture_validation import validate_fracture_phase1_config
from .fracture_work import BoundaryEquilibriumState, accepted_step_work_increment

_FLOAT_EPS = np.finfo(np.float64).eps
_FORMAL_LABELS_ALLOWED = False
_SOLVER_ENERGY_INCREMENT_RESIDUAL_AVAILABLE = True


class FractureTrajectoryAdapterError(RuntimeError):
    """Raised when solver output cannot be adapted without inventing evidence."""


class FractureTrajectoryRunFailed(FractureTrajectoryAdapterError):
    """Raised when a development trajectory exhausts or cannot enter retry."""

    def __init__(
        self,
        message: str,
        *,
        attempt_ledger: Sequence[Mapping[str, Any]],
        accepted_load_parameters: Sequence[float],
    ) -> None:
        super().__init__(message)
        self.attempt_ledger = tuple(MappingProxyType(dict(entry)) for entry in attempt_ledger)
        self.accepted_load_parameters = tuple(float(value) for value in accepted_load_parameters)


@dataclass(frozen=True, slots=True)
class Phase1TrajectoryIdentity:
    """Caller-owned identity and descriptive metadata for one development run."""

    trajectory_id: str
    case_id: str
    mesh_id: str
    geometry_id: str
    material_id: str
    solver_hash: str
    geometry: Mapping[str, Any]
    solver: Mapping[str, Any]
    env: Mapping[str, Any] | None = None
    meta: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "trajectory_id",
            "case_id",
            "mesh_id",
            "geometry_id",
            "material_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if (
            not isinstance(self.solver_hash, str)
            or len(self.solver_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.solver_hash)
        ):
            raise ValueError("solver_hash must be a lowercase SHA-256 digest")
        if not isinstance(self.geometry, Mapping) or not self.geometry:
            raise ValueError("geometry must be a non-empty mapping")
        if not isinstance(self.solver, Mapping) or not str(self.solver.get("name", "")).strip():
            raise ValueError("solver must be a mapping with a non-empty name")
        for name in ("env", "meta"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping or None")


@dataclass(frozen=True, slots=True)
class GlobalBalanceDiagnostics:
    """Global nodal force and moment residuals for all accepted internal states."""

    force_residual_yz: np.ndarray
    force_relative_residual: np.ndarray
    moment_residual: np.ndarray
    moment_relative_residual: np.ndarray
    reference_yz: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "force_residual_yz": (self.force_residual_yz, 2),
            "force_relative_residual": (self.force_relative_residual, 1),
            "moment_residual": (self.moment_residual, 1),
            "moment_relative_residual": (self.moment_relative_residual, 1),
            "reference_yz": (self.reference_yz, 1),
        }
        copied: dict[str, np.ndarray] = {}
        for name, (raw, ndim) in arrays.items():
            value = np.asarray(raw, dtype=np.float64)
            if value.ndim != ndim or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
            immutable = np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=np.float64)
            copied[name] = immutable.reshape(value.shape)
        count = copied["force_residual_yz"].shape[0]
        if copied["force_residual_yz"].shape != (count, 2):
            raise ValueError("force_residual_yz must have shape [T,2]")
        for name in (
            "force_relative_residual",
            "moment_residual",
            "moment_relative_residual",
        ):
            if copied[name].shape != (count,):
                raise ValueError(f"{name} must have shape [T]")
        if copied["reference_yz"].shape != (2,):
            raise ValueError("reference_yz must have shape [2]")
        if np.any(copied["force_relative_residual"] < 0.0) or np.any(
            copied["moment_relative_residual"] < 0.0
        ):
            raise ValueError("relative balance residuals must be non-negative")
        for name, value in copied.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class Phase1DevelopmentRun:
    """Validated development trajectory plus required-output and balance indices."""

    trajectory: FractureTrajectory
    required_output_indices: np.ndarray
    balance: GlobalBalanceDiagnostics
    formal_labels_allowed: bool = _FORMAL_LABELS_ALLOWED
    solver_energy_increment_residual_available: bool = _SOLVER_ENERGY_INCREMENT_RESIDUAL_AVAILABLE

    def __post_init__(self) -> None:
        if not isinstance(self.trajectory, FractureTrajectory):
            raise TypeError("trajectory must be a FractureTrajectory")
        indices = np.asarray(self.required_output_indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("required_output_indices must be a one-dimensional integer array")
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size != 41 or np.any(np.diff(indices) <= 0):
            raise ValueError("required_output_indices must contain 41 increasing entries")
        if indices[0] < 0 or indices[-1] >= self.trajectory.num_steps:
            raise ValueError("required_output_indices contains an out-of-range entry")
        if self.balance.force_residual_yz.shape[0] != self.trajectory.num_steps:
            raise ValueError("balance diagnostics must align with every accepted internal step")
        immutable = np.frombuffer(indices.tobytes(), dtype=np.int64).reshape(indices.shape)
        object.__setattr__(self, "required_output_indices", immutable)
        if self.formal_labels_allowed is not False:
            raise ValueError("the development adapter cannot authorize formal labels")
        if self.solver_energy_increment_residual_available is not True:
            raise ValueError("the scheduled solver energy-iteration residual must be available")

    @property
    def required_output_s(self) -> np.ndarray:
        values = self.trajectory.load_parameter[self.required_output_indices]
        values.setflags(write=False)
        return values


SolvePrefix = Callable[..., ScheduledAT2Result]


@dataclass(frozen=True, slots=True)
class _CandidateDiagnostics:
    boundary: BoundaryEquilibriumState
    wall_work_increment: float
    farfield_work_increment: float
    cumulative_external_work: float
    relative_energy_imbalance: float
    equilibrium_relative_residual: float
    history_monotonicity_violation: float
    damage_irreversibility_violation: float
    global_force_residual_yz: np.ndarray
    global_force_relative_residual: float
    global_moment_residual: float
    global_moment_relative_residual: float


def canonical_config_sha256(config: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 over one JSON-compatible Phase-1 config."""

    validate_fracture_phase1_config(config)
    payload = json.dumps(
        config,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        nodes = np.asarray(mesh.nodes, dtype=np.float64)
        elements = np.asarray(mesh.elements, dtype=np.int64)
        facets = np.asarray(mesh.mesh.facets, dtype=np.int64)
        wall_ids = np.asarray(mesh.boundary_facets["wall"], dtype=np.int64)
        farfield_ids = np.asarray(mesh.boundary_facets["farfield"], dtype=np.int64)
    except (AttributeError, KeyError, TypeError) as exc:
        raise FractureTrajectoryAdapterError(
            "mesh must expose nodes, elements, mesh.facets, and wall/farfield facet IDs"
        ) from exc
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise FractureTrajectoryAdapterError("mesh nodes must be finite with shape [N,2]")
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise FractureTrajectoryAdapterError("mesh elements must have shape [M,3]")
    if facets.ndim != 2 or facets.shape[0] != 2:
        raise FractureTrajectoryAdapterError("mesh.mesh.facets must have shape [2,F]")
    if wall_ids.ndim != 1 or farfield_ids.ndim != 1:
        raise FractureTrajectoryAdapterError("boundary facet identifiers must be vectors")
    if wall_ids.size == 0 or farfield_ids.size == 0:
        raise FractureTrajectoryAdapterError("wall and farfield facet sets must be non-empty")
    return (
        nodes,
        elements,
        np.asarray(facets[:, wall_ids].T, dtype=np.int64),
        np.asarray(facets[:, farfield_ids].T, dtype=np.int64),
    )


def _strict_load_state(schedule: Phase1LoadSchedule, step: ScheduledAT2StepResult) -> None:
    """Require exact schedule evaluation including exact facet row order."""

    expected = schedule.state_at(step.load_parameter)
    actual = step.load_state
    if not isinstance(actual, FractureLoadState):
        raise FractureTrajectoryAdapterError("scheduled step has no FractureLoadState")
    scalar_names = (
        "s",
        "ucs_scale",
        "sigma1_over_UCS",
        "sigma3_over_sigma1",
        "principal_angle_deg",
    )
    if actual.path_id != expected.path_id or actual.wall_zone_ids != expected.wall_zone_ids:
        raise FractureTrajectoryAdapterError("scheduled load-state path or zone IDs differ")
    for name in scalar_names:
        if getattr(actual, name) != getattr(expected, name):
            raise FractureTrajectoryAdapterError(
                f"scheduled load-state scalar {name} differs from state_at(s)"
            )
    array_names = (
        "farfield_stress_tension_positive_yz",
        "wall_facet_ids",
        "wall_zone_release",
        "wall_zone_weights",
        "wall_release",
    )
    for name in array_names:
        if not np.array_equal(
            np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))
        ):
            raise FractureTrajectoryAdapterError(
                f"scheduled load-state {name} differs bitwise or in facet row order"
            )


_STEP_ARRAY_FIELDS = (
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
)
_STEP_SCALAR_FIELDS = (
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
)


def _copy_step(step: ScheduledAT2StepResult) -> ScheduledAT2StepResult:
    arguments = {name: np.asarray(getattr(step, name)).copy() for name in _STEP_ARRAY_FIELDS}
    arguments.update({name: getattr(step, name) for name in _STEP_SCALAR_FIELDS})
    arguments["load_state"] = step.load_state
    return ScheduledAT2StepResult(**arguments)


def _validate_step_payload(
    step: ScheduledAT2StepResult, node_count: int, element_count: int
) -> None:
    shapes = {
        "displacement": (node_count, 2),
        "correction_displacement": (node_count, 2),
        "damage": (node_count,),
        "strain": (element_count, 3),
        "stress": (element_count, 3),
        "psi_positive": (element_count,),
        "psi_negative": (element_count,),
        "history": (element_count,),
        "internal_force": (2 * node_count,),
        "wall_nodal_force": (2 * node_count,),
    }
    for name, expected_shape in shapes.items():
        value = np.asarray(getattr(step, name))
        if value.shape != expected_shape or not np.isfinite(value).all():
            raise FractureTrajectoryAdapterError(
                f"scheduled step {name} must be finite with shape {expected_shape}"
            )
    dirichlet = np.asarray(step.dirichlet_dofs)
    prescribed = np.asarray(step.farfield_prescribed_displacement)
    if (
        dirichlet.ndim != 1
        or not np.issubdtype(dirichlet.dtype, np.integer)
        or dirichlet.size == 0
        or np.any(dirichlet < 0)
        or np.any(dirichlet >= 2 * node_count)
        or (dirichlet.size > 1 and np.any(np.diff(dirichlet) <= 0))
    ):
        raise FractureTrajectoryAdapterError(
            "scheduled step dirichlet_dofs must be increasing in [0,2N)"
        )
    if prescribed.shape != dirichlet.shape or not np.isfinite(prescribed).all():
        raise FractureTrajectoryAdapterError(
            "scheduled step prescribed displacement must align with dirichlet_dofs"
        )
    for name in (
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
    ):
        if not math.isfinite(float(getattr(step, name))):
            raise FractureTrajectoryAdapterError(f"scheduled step {name} must be finite")
    energy_change = float(step.energy_change)
    if not (
        (math.isfinite(energy_change) and energy_change >= 0.0)
        or (math.isinf(energy_change) and energy_change > 0.0 and not step.converged)
    ):
        raise FractureTrajectoryAdapterError(
            "scheduled step energy_change must be non-negative and finite when converged"
        )
    for name in (
        "staggered_iterations",
        "displacement_iterations",
        "damage_iterations",
    ):
        value = getattr(step, name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FractureTrajectoryAdapterError(
                f"scheduled step {name} must be a non-negative integer"
            )
    if not isinstance(step.converged, (bool, np.bool_)):
        raise FractureTrajectoryAdapterError("scheduled step converged must be boolean")


def _validate_prefix_result(
    result: ScheduledAT2Result,
    mesh_nodes: np.ndarray,
    mesh_elements: np.ndarray,
    schedule: Phase1LoadSchedule,
    material: AT2Material,
    options: FractureSolverOptions,
    parameters: Sequence[float],
    accepted_steps: Sequence[ScheduledAT2StepResult],
) -> ScheduledAT2StepResult:
    if not isinstance(result, ScheduledAT2Result):
        raise FractureTrajectoryAdapterError("solve_prefix must return ScheduledAT2Result")
    if result.load_schedule is not schedule:
        raise FractureTrajectoryAdapterError("solver returned a different load-schedule object")
    if result.material != material or result.options != options:
        raise FractureTrajectoryAdapterError("solver returned different material or options")
    if not np.array_equal(result.nodes, mesh_nodes) or not np.array_equal(
        result.elements, mesh_elements
    ):
        raise FractureTrajectoryAdapterError("solver result mesh differs from the scheduled mesh")
    if tuple(result.load_path.load_parameters) != tuple(parameters):
        raise FractureTrajectoryAdapterError(
            "solver result load path differs from attempted prefix"
        )
    if len(result.steps) != len(parameters):
        raise FractureTrajectoryAdapterError(
            "solver result does not contain the full attempted prefix"
        )
    for parameter, step in zip(parameters, result.steps, strict=True):
        _validate_step_payload(step, mesh_nodes.shape[0], mesh_elements.shape[0])
        if step.load_parameter != parameter:
            raise FractureTrajectoryAdapterError(
                "solver step coordinate differs from attempted prefix"
            )
        _strict_load_state(schedule, step)
    if accepted_steps:
        # Compare every state quantity that could otherwise leak from a rejected
        # solve.  The public solver is deterministic for a fixed fresh prefix.
        if len(result.steps) < len(accepted_steps):
            raise FractureTrajectoryAdapterError("solver result omitted accepted prefix states")
        for index, expected in enumerate(accepted_steps):
            actual = result.steps[index]
            for name in _STEP_ARRAY_FIELDS:
                if not np.array_equal(
                    np.asarray(getattr(expected, name)), np.asarray(getattr(actual, name))
                ):
                    raise FractureTrajectoryAdapterError(
                        f"fresh retry changed accepted prefix field {name} at index {index}"
                    )
            for name in _STEP_SCALAR_FIELDS:
                if getattr(expected, name) != getattr(actual, name):
                    raise FractureTrajectoryAdapterError(
                        f"fresh retry changed accepted prefix scalar {name} at index {index}"
                    )
    return result.steps[-1]


def _element_geometry(nodes: np.ndarray, elements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = nodes[elements]
    determinant = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    area = 0.5 * np.abs(determinant)
    if np.any(area <= 0.0):
        raise FractureTrajectoryAdapterError("mesh contains a degenerate triangle")
    return area, triangles.mean(axis=1)


def _damage_integrals(
    nodes: np.ndarray,
    elements: np.ndarray,
    area: np.ndarray,
    damage: np.ndarray,
    length_scale: float,
) -> tuple[float, float, np.ndarray]:
    local = damage[elements]
    mean = local.mean(axis=1)
    mean_square = (
        np.sum(local * local, axis=1)
        + local[:, 0] * local[:, 1]
        + local[:, 1] * local[:, 2]
        + local[:, 2] * local[:, 0]
    ) / 6.0
    triangles = nodes[elements]
    determinant = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    gradients = np.empty((elements.shape[0], 3, 2), dtype=np.float64)
    gradients[:, 0, 0] = (triangles[:, 1, 1] - triangles[:, 2, 1]) / determinant
    gradients[:, 0, 1] = (triangles[:, 2, 0] - triangles[:, 1, 0]) / determinant
    gradients[:, 1, 0] = (triangles[:, 2, 1] - triangles[:, 0, 1]) / determinant
    gradients[:, 1, 1] = (triangles[:, 0, 0] - triangles[:, 2, 0]) / determinant
    gradients[:, 2, 0] = (triangles[:, 0, 1] - triangles[:, 1, 1]) / determinant
    gradients[:, 2, 1] = (triangles[:, 1, 0] - triangles[:, 0, 0]) / determinant
    damage_gradient = np.einsum("mi,mij->mj", local, gradients)
    gradient_square = np.sum(damage_gradient * damage_gradient, axis=1)
    damage_area = float(np.sum(area * mean))
    crack_density = float(
        np.sum(area * (mean_square / (2.0 * length_scale) + 0.5 * length_scale * gradient_square))
    )
    degradation_average = 1.0 - 2.0 * mean + mean_square
    return damage_area, crack_density, degradation_average


def _assembled_internal_force(
    nodes: np.ndarray,
    elements: np.ndarray,
    area: np.ndarray,
    stress: np.ndarray,
) -> np.ndarray:
    """Independently assemble P1 internal nodal forces from element stress."""

    triangles = nodes[elements]
    determinant = (triangles[:, 1, 0] - triangles[:, 0, 0]) * (
        triangles[:, 2, 1] - triangles[:, 0, 1]
    ) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1])
    gradients = np.empty((elements.shape[0], 3, 2), dtype=np.float64)
    gradients[:, 0, 0] = (triangles[:, 1, 1] - triangles[:, 2, 1]) / determinant
    gradients[:, 0, 1] = (triangles[:, 2, 0] - triangles[:, 1, 0]) / determinant
    gradients[:, 1, 0] = (triangles[:, 2, 1] - triangles[:, 0, 1]) / determinant
    gradients[:, 1, 1] = (triangles[:, 0, 0] - triangles[:, 2, 0]) / determinant
    gradients[:, 2, 0] = (triangles[:, 0, 1] - triangles[:, 1, 1]) / determinant
    gradients[:, 2, 1] = (triangles[:, 1, 0] - triangles[:, 0, 0]) / determinant
    matrices = np.zeros((elements.shape[0], 3, 6), dtype=np.float64)
    matrices[:, 0, 0::2] = gradients[:, :, 0]
    matrices[:, 1, 1::2] = gradients[:, :, 1]
    matrices[:, 2, 0::2] = gradients[:, :, 1]
    matrices[:, 2, 1::2] = gradients[:, :, 0]
    local_force = area[:, None] * np.einsum("mij,mi->mj", matrices, stress)
    internal = np.zeros(2 * nodes.shape[0], dtype=np.float64)
    element_dofs = np.empty((elements.shape[0], 6), dtype=np.int64)
    element_dofs[:, 0::2] = 2 * elements
    element_dofs[:, 1::2] = 2 * elements + 1
    np.add.at(internal, element_dofs.ravel(), local_force.ravel())
    return internal


def _schedule_affine_displacement(
    nodes: np.ndarray, material: AT2Material, load_state: FractureLoadState
) -> np.ndarray:
    stress = np.asarray(load_state.farfield_stress_tension_positive_yz, dtype=np.float64)
    lame_lambda = material.lame_lambda
    mu = material.shear_modulus
    normal_matrix = np.asarray(
        [[lame_lambda + 2.0 * mu, lame_lambda], [lame_lambda, lame_lambda + 2.0 * mu]],
        dtype=np.float64,
    )
    normal_strain = np.linalg.solve(normal_matrix, np.diag(stress))
    strain = np.asarray([normal_strain[0], normal_strain[1], stress[0, 1] / mu], dtype=np.float64)
    tensor = np.asarray(
        [[strain[0], 0.5 * strain[2]], [0.5 * strain[2], strain[1]]], dtype=np.float64
    )
    return nodes @ tensor.T


def damage_graph_connectivity(
    elements: np.ndarray,
    wall_facets: np.ndarray,
    farfield_facets: np.ndarray,
    damage: np.ndarray,
) -> float:
    """Return widest-path wall-to-farfield damage connectivity in ``[0,1]``.

    Each mesh edge has capacity ``min(d_i, d_j)``.  The returned value is the
    maximum, over all wall-to-farfield graph paths, of the minimum capacity on
    that path.  It is therefore zero for an intact cut and one only for a fully
    damaged connected nodal path.  This is a deterministic topology-derived
    diagnostic, not a supplied or thresholded placeholder.
    """

    topology = np.asarray(elements, dtype=np.int64)
    values = np.asarray(damage, dtype=np.float64)
    if topology.ndim != 2 or topology.shape[1] != 3:
        raise ValueError("elements must have shape [M,3]")
    if (
        values.ndim != 1
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("damage must be a finite vector in [0,1]")
    if topology.size and (topology.min() < 0 or topology.max() >= values.size):
        raise ValueError("elements contain an out-of-range node")
    edges = np.unique(
        np.sort(
            np.concatenate([topology[:, [0, 1]], topology[:, [1, 2]], topology[:, [2, 0]]], axis=0),
            axis=1,
        ),
        axis=0,
    )
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(values.size)]
    for left, right in edges.tolist():
        capacity = float(min(values[left], values[right]))
        adjacency[left].append((right, capacity))
        adjacency[right].append((left, capacity))
    sources = np.unique(np.asarray(wall_facets, dtype=np.int64))
    targets = set(np.unique(np.asarray(farfield_facets, dtype=np.int64)).tolist())
    capacity = np.full(values.size, -1.0, dtype=np.float64)
    unvisited = np.ones(values.size, dtype=bool)
    capacity[sources] = values[sources]
    while np.any(unvisited):
        available = np.where(unvisited, capacity, -np.inf)
        node = int(np.argmax(available))
        if not np.isfinite(available[node]) or available[node] < 0.0:
            break
        unvisited[node] = False
        if node in targets:
            return float(capacity[node])
        for neighbor, edge_capacity in adjacency[node]:
            if unvisited[neighbor]:
                candidate = min(float(capacity[node]), edge_capacity, float(values[neighbor]))
                capacity[neighbor] = max(capacity[neighbor], candidate)
    return 0.0


def _global_balance(
    nodes: np.ndarray,
    state: BoundaryEquilibriumState,
    reference_yz: np.ndarray,
    force_floor: float,
) -> tuple[np.ndarray, float, float, float]:
    residual = (state.internal_force - state.wall_nodal_force - state.reaction_full).reshape(-1, 2)
    force = residual.sum(axis=0)
    internal = state.internal_force.reshape(-1, 2)
    wall = state.wall_nodal_force.reshape(-1, 2)
    reaction = state.reaction_full.reshape(-1, 2)
    force_scale = max(
        float(np.sum(np.linalg.norm(internal, axis=1))),
        float(np.sum(np.linalg.norm(wall, axis=1))),
        float(np.sum(np.linalg.norm(reaction, axis=1))),
        force_floor,
    )
    relative_force = float(np.linalg.norm(force) / force_scale)
    lever = nodes - reference_yz

    def moments(forces: np.ndarray) -> np.ndarray:
        return lever[:, 0] * forces[:, 1] - lever[:, 1] * forces[:, 0]

    residual_moment = float(np.sum(moments(residual)))
    length_scale = max(float(np.max(np.linalg.norm(lever, axis=1))), 1.0)
    moment_scale = max(
        float(np.sum(np.abs(moments(internal)))),
        float(np.sum(np.abs(moments(wall)))),
        float(np.sum(np.abs(moments(reaction)))),
        force_floor * length_scale,
    )
    relative_moment = abs(residual_moment) / moment_scale
    return force, relative_force, residual_moment, relative_moment


def _candidate_diagnostics(
    nodes: np.ndarray,
    step: ScheduledAT2StepResult,
    previous_step: ScheduledAT2StepResult | None,
    previous_boundary: BoundaryEquilibriumState | None,
    cumulative_external_work: float,
    reference_yz: np.ndarray,
    force_floor: float,
    energy_floor: float,
) -> _CandidateDiagnostics:
    boundary = BoundaryEquilibriumState(
        displacement=step.displacement,
        internal_force=step.internal_force,
        wall_nodal_force=step.wall_nodal_force,
        dirichlet_dofs=step.dirichlet_dofs,
        farfield_prescribed_displacement=step.farfield_prescribed_displacement,
        accepted=True,
    )
    if previous_boundary is None:
        wall_work = 0.0
        farfield_work = 0.0
        relative_energy = 0.0
        history_violation = max(0.0, float(np.max(step.psi_positive - step.history)))
        irreversibility = max(0.0, float(step.irreversibility_violation))
    else:
        work = accepted_step_work_increment(previous_boundary, boundary)
        wall_work = work.wall_work
        farfield_work = work.farfield_work
        assert previous_step is not None
        energy_increment = (step.elastic_energy + step.fracture_energy) - (
            previous_step.elastic_energy + previous_step.fracture_energy
        )
        external_increment = wall_work + farfield_work
        relative_energy = abs(energy_increment - external_increment) / max(
            abs(energy_increment), abs(external_increment), energy_floor
        )
        history_violation = max(
            0.0,
            float(np.max(step.psi_positive - step.history)),
            float(-np.min(step.history - previous_step.history)),
        )
        irreversibility = max(
            0.0,
            float(step.irreversibility_violation),
            float(-np.min(step.damage - previous_step.damage)),
        )
    free_norm = float(np.linalg.norm(boundary.free_residual))
    free_internal = float(np.linalg.norm(boundary.internal_force[boundary.free_dofs]))
    free_wall = float(np.linalg.norm(boundary.wall_nodal_force[boundary.free_dofs]))
    equilibrium = free_norm / max(free_internal, free_wall, force_floor)
    force, force_relative, moment, moment_relative = _global_balance(
        nodes, boundary, reference_yz, force_floor
    )
    return _CandidateDiagnostics(
        boundary=boundary,
        wall_work_increment=float(wall_work),
        farfield_work_increment=float(farfield_work),
        cumulative_external_work=float(cumulative_external_work + wall_work + farfield_work),
        relative_energy_imbalance=float(relative_energy),
        equilibrium_relative_residual=float(equilibrium),
        history_monotonicity_violation=float(history_violation),
        damage_irreversibility_violation=float(irreversibility),
        global_force_residual_yz=np.asarray(force, dtype=np.float64),
        global_force_relative_residual=float(force_relative),
        global_moment_residual=float(moment),
        global_moment_relative_residual=float(moment_relative),
    )


def _failure_reason(
    step: ScheduledAT2StepResult,
    diagnostics: _CandidateDiagnostics,
    config: Mapping[str, Any],
) -> tuple[str, str] | None:
    solver = config["solver"]
    qc = config["quality_control"]["per_trajectory"]
    checks = (
        (not step.converged, "SOLVER_NONCONVERGENCE", "scheduled staggered solve did not converge"),
        (
            step.energy_change > float(solver["relative_energy_increment_tolerance"]),
            "ENERGY_INCREMENT_NOT_CONVERGED",
            "staggered potential-energy change exceeds the frozen solver tolerance",
        ),
        (
            diagnostics.equilibrium_relative_residual
            > float(solver["equilibrium_relative_residual_tolerance"]),
            "EQUILIBRIUM_NOT_CONVERGED",
            "recomputed free-DOF equilibrium residual exceeds the frozen tolerance",
        ),
        (
            max(step.kkt_residual, step.complementarity_residual)
            > float(solver["kkt_complementarity_relative_residual_tolerance"]),
            "DAMAGE_KKT_NOT_CONVERGED",
            "damage KKT or complementarity residual exceeds the frozen tolerance",
        ),
        (
            diagnostics.damage_irreversibility_violation
            > float(qc["max_damage_irreversibility_violation"]),
            "DAMAGE_IRREVERSIBILITY_VIOLATION",
            "candidate damage decreases relative to the last accepted state",
        ),
        (
            step.range_violation > float(qc["max_damage_range_violation"]),
            "DAMAGE_RANGE_VIOLATION",
            "candidate damage violates the frozen range tolerance",
        ),
        (
            diagnostics.history_monotonicity_violation
            > float(qc["max_history_monotonicity_violation"])
            * max(float(np.max(step.history)), float(np.max(step.psi_positive)), 1.0),
            "HISTORY_MONOTONICITY_VIOLATION",
            "candidate history decreases or is smaller than psi_plus",
        ),
        (
            diagnostics.relative_energy_imbalance > float(qc["max_relative_energy_imbalance"]),
            "ENERGY_IMBALANCE",
            "accepted-step recoverable-energy increment disagrees with boundary work",
        ),
        (
            diagnostics.global_force_relative_residual
            > float(solver["equilibrium_relative_residual_tolerance"]),
            "GLOBAL_FORCE_IMBALANCE",
            "global resultant residual exceeds the equilibrium tolerance",
        ),
        (
            diagnostics.global_moment_relative_residual
            > float(solver["equilibrium_relative_residual_tolerance"]),
            "GLOBAL_MOMENT_IMBALANCE",
            "global moment residual exceeds the equilibrium tolerance",
        ),
    )
    return next(((code, message) for failed, code, message in checks if failed), None)


def _ledger_entry(
    *,
    step_index: int,
    attempt_index: int,
    start: float,
    target: float,
    accepted: bool,
    step: ScheduledAT2StepResult,
    diagnostics: _CandidateDiagnostics,
    halvings: int,
    wall_facets: np.ndarray,
    cumulative_external_work: float,
    failure: tuple[str, str] | None,
) -> dict[str, Any]:
    load_state_hash = None
    neumann = None
    wall_work = None
    farfield_work = None
    cumulative = None
    staggered_potential_energy_change = None
    if accepted:
        stress = step.load_state.farfield_stress_tension_positive_yz
        load_state_hash = compute_load_state_sha256(
            np.asarray([target], dtype=np.float64),
            np.asarray([stress[0, 0], stress[1, 1], stress[0, 1]], dtype=np.float64),
            np.asarray(step.load_state.wall_release, dtype=np.float64),
            wall_facets,
        )
        neumann = float(np.dot(step.wall_nodal_force, step.displacement.reshape(-1)))
        wall_work = diagnostics.wall_work_increment
        farfield_work = diagnostics.farfield_work_increment
        cumulative = cumulative_external_work
        staggered_potential_energy_change = float(step.energy_change)
    return {
        "step_index": int(step_index),
        "attempt_index": int(attempt_index),
        "load_parameter_start": float(start),
        "load_parameter_target": float(target),
        "accepted": bool(accepted),
        "failure_code": None if failure is None else failure[0],
        "failure_message": None if failure is None else failure[1],
        "newton_iterations": int(step.displacement_iterations),
        "active_set_iterations": int(step.damage_iterations),
        "staggered_iterations": int(step.staggered_iterations),
        "step_halvings": int(halvings),
        "equilibrium_relative_residual": diagnostics.equilibrium_relative_residual,
        "kkt_relative_residual": float(step.kkt_residual),
        "complementarity_relative_residual": float(step.complementarity_residual),
        "damage_irreversibility_violation": diagnostics.damage_irreversibility_violation,
        "damage_range_violation": float(step.range_violation),
        "relative_energy_imbalance": diagnostics.relative_energy_imbalance,
        "staggered_potential_energy_change": staggered_potential_energy_change,
        "load_state_sha256": load_state_hash,
        "neumann_load_functional": neumann,
        "wall_work_increment": wall_work,
        "farfield_work_increment": farfield_work,
        "cumulative_external_work": cumulative,
    }


def _phase1_load_path_metadata(
    config: Mapping[str, Any], schedule: Phase1LoadSchedule
) -> dict[str, Any]:
    path = next(item for item in config["load_paths"]["paths"] if item["id"] == schedule.path_id)
    result: dict[str, Any] = {
        "path_parameter": "s",
        "parameter_bounds": [0.0, 1.0],
        "monotone": True,
        "interpolation": config["load_paths"]["interpolation_rule"],
        "control_knots": path["control_knots"],
    }
    if schedule.path_id == "p4":
        result["wall_zone_definition"] = config["load_paths"]["wall_zones_for_p4"]
    return result


def _default_env() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _schema_record(
    *,
    mesh: Any,
    material: AT2Material,
    schedule: Phase1LoadSchedule,
    config: Mapping[str, Any],
    identity: Phase1TrajectoryIdentity,
    accepted_steps: Sequence[ScheduledAT2StepResult],
    boundaries: Sequence[BoundaryEquilibriumState],
    diagnostics: Sequence[_CandidateDiagnostics],
    ledger: Sequence[Mapping[str, Any]],
    required_output_indices: np.ndarray,
    force_floor: float,
    energy_floor: float,
) -> tuple[FractureTrajectory, GlobalBalanceDiagnostics]:
    nodes, elements, wall_facets, farfield_facets = _mesh_arrays(mesh)
    area, centers = _element_geometry(nodes, elements)
    step_count = len(accepted_steps)
    if not step_count or len(boundaries) != step_count or len(diagnostics) != step_count:
        raise FractureTrajectoryAdapterError("accepted state bookkeeping is incomplete")
    dirichlet = np.asarray(accepted_steps[0].dirichlet_dofs, dtype=np.int64)
    if any(not np.array_equal(step.dirichlet_dofs, dirichlet) for step in accepted_steps[1:]):
        raise FractureTrajectoryAdapterError(
            "farfield Dirichlet DOFs changed across the trajectory"
        )
    farfield_nodes = np.unique(farfield_facets)
    expected_dirichlet = np.column_stack((2 * farfield_nodes, 2 * farfield_nodes + 1)).ravel()
    if not np.array_equal(dirichlet, expected_dirichlet):
        raise FractureTrajectoryAdapterError(
            "farfield Dirichlet DOFs must contain both node-major components of every "
            "farfield-facet node"
        )

    load_parameter = np.asarray([step.load_parameter for step in accepted_steps], dtype=np.float64)
    farfield_stress = np.asarray(
        [
            [
                step.load_state.farfield_stress_tension_positive_yz[0, 0],
                step.load_state.farfield_stress_tension_positive_yz[1, 1],
                step.load_state.farfield_stress_tension_positive_yz[0, 1],
            ]
            for step in accepted_steps
        ],
        dtype=np.float64,
    )
    wall_release = np.asarray(
        [step.load_state.wall_release for step in accepted_steps], dtype=np.float64
    )
    u = np.asarray([step.displacement for step in accepted_steps], dtype=np.float64)
    internal = np.asarray([step.internal_force for step in accepted_steps], dtype=np.float64)
    wall_force = np.asarray([step.wall_nodal_force for step in accepted_steps], dtype=np.float64)
    prescribed = np.asarray(
        [step.farfield_prescribed_displacement for step in accepted_steps], dtype=np.float64
    )
    reaction = np.asarray(
        [boundary.reaction_on_dirichlet_dofs for boundary in boundaries], dtype=np.float64
    )
    damage = np.asarray([step.damage for step in accepted_steps], dtype=np.float64)
    strain = np.asarray([step.strain for step in accepted_steps], dtype=np.float64)
    stress = np.asarray([step.stress for step in accepted_steps], dtype=np.float64)
    psi_plus = np.asarray([step.psi_positive for step in accepted_steps], dtype=np.float64)
    psi_minus = np.asarray([step.psi_negative for step in accepted_steps], dtype=np.float64)
    history = np.asarray([step.history for step in accepted_steps], dtype=np.float64)

    damage_area = np.empty(step_count, dtype=np.float64)
    crack_density = np.empty(step_count, dtype=np.float64)
    fracture_energy = np.empty(step_count, dtype=np.float64)
    elastic_energy = np.empty(step_count, dtype=np.float64)
    sigma_xx = np.empty((step_count, elements.shape[0]), dtype=np.float64)
    connectivity = np.empty(step_count, dtype=np.float64)
    intact_material = AT2Material(
        material.young_modulus,
        material.poisson_ratio,
        material.fracture_toughness,
        material.length_scale,
        0.0,
    )
    for index, step in enumerate(accepted_steps):
        expected_affine = _schedule_affine_displacement(nodes, material, step.load_state)
        displacement_scale = max(float(np.max(np.abs(expected_affine))), 1.0)
        if not np.allclose(
            step.displacement[farfield_nodes],
            expected_affine[farfield_nodes],
            rtol=2.0e-11,
            atol=2.0e-13 * displacement_scale,
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} farfield displacement differs from schedule stress"
            )
        if not np.allclose(
            step.correction_displacement,
            step.displacement - expected_affine,
            rtol=2.0e-11,
            atol=2.0e-13 * displacement_scale,
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} correction displacement is inconsistent"
            )
        expected_strain, expected_area = compute_element_strain(nodes, elements, step.displacement)
        if not np.array_equal(expected_area, area):
            raise FractureTrajectoryAdapterError("independent element areas changed")
        strain_scale = max(float(np.max(np.abs(expected_strain))), 1.0)
        if not np.allclose(
            step.strain,
            expected_strain,
            rtol=2.0e-11,
            atol=2.0e-13 * strain_scale,
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} strain is inconsistent with nodal displacement"
            )
        damage_area[index], crack_density[index], degradation_no_k = _damage_integrals(
            nodes, elements, area, step.damage, material.length_scale
        )
        degradation_average = degradation_no_k + material.residual_stiffness
        response = miehe_spectral_response(step.strain, intact_material)
        expected_stress_tensor = (
            degradation_average[:, None, None] * response.stress_positive + response.stress_negative
        )
        expected_stress = np.column_stack(
            (
                expected_stress_tensor[:, 1, 1],
                expected_stress_tensor[:, 2, 2],
                expected_stress_tensor[:, 1, 2],
            )
        )
        scale = max(float(np.max(np.abs(expected_stress))), 1.0)
        if not np.allclose(step.stress, expected_stress, rtol=2.0e-11, atol=2.0e-13 * scale):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} stress is inconsistent with strain and P1 damage"
            )
        if not np.allclose(
            step.psi_positive, response.psi_positive, rtol=2.0e-11, atol=2.0e-13
        ) or not np.allclose(step.psi_negative, response.psi_negative, rtol=2.0e-11, atol=2.0e-13):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} split energy is inconsistent with its strain"
            )
        expected_internal = _assembled_internal_force(nodes, elements, area, expected_stress)
        force_scale = max(float(np.max(np.abs(expected_internal))), 1.0)
        if not np.allclose(
            step.internal_force,
            expected_internal,
            rtol=2.0e-11,
            atol=2.0e-13 * force_scale,
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} internal force is inconsistent with element stress"
            )
        expected_neumann = float(np.dot(step.wall_nodal_force, step.displacement.reshape(-1)))
        if not math.isclose(
            step.neumann_load_functional,
            expected_neumann,
            rel_tol=2.0e-11,
            abs_tol=2.0e-13 * max(abs(expected_neumann), 1.0),
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} Neumann functional is inconsistent"
            )
        sigma_xx[index] = expected_stress_tensor[:, 0, 0]
        elastic_energy[index] = float(
            np.sum(area * (degradation_average * step.psi_positive + step.psi_negative))
        )
        fracture_energy[index] = material.fracture_toughness * crack_density[index]
        energy_scale = max(abs(elastic_energy[index]), abs(fracture_energy[index]), 1.0)
        if not math.isclose(
            elastic_energy[index],
            step.elastic_energy,
            rel_tol=2.0e-11,
            abs_tol=2.0e-13 * energy_scale,
        ) or not math.isclose(
            fracture_energy[index],
            step.fracture_energy,
            rel_tol=2.0e-11,
            abs_tol=2.0e-13 * energy_scale,
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} energy is inconsistent with stored state fields"
            )
        expected_potential = elastic_energy[index] + fracture_energy[index] - expected_neumann
        if not math.isclose(
            step.total_potential_energy,
            expected_potential,
            rel_tol=2.0e-11,
            abs_tol=2.0e-13 * max(abs(expected_potential), 1.0),
        ):
            raise FractureTrajectoryAdapterError(
                f"accepted step {index} total potential is inconsistent"
            )
        connectivity[index] = damage_graph_connectivity(
            elements, wall_facets, farfield_facets, step.damage
        )

    wall_work = np.asarray([item.wall_work_increment for item in diagnostics], dtype=np.float64)
    farfield_work = np.asarray(
        [item.farfield_work_increment for item in diagnostics], dtype=np.float64
    )
    cumulative_work = np.asarray(
        [item.cumulative_external_work for item in diagnostics], dtype=np.float64
    )
    neumann = np.einsum("ti,ti->t", wall_force, u.reshape(step_count, -1))
    total_potential = elastic_energy + fracture_energy - neumann
    equilibrium = np.asarray(
        [item.equilibrium_relative_residual for item in diagnostics], dtype=np.float64
    )
    history_violation = np.asarray(
        [item.history_monotonicity_violation for item in diagnostics], dtype=np.float64
    )
    irreversibility = np.asarray(
        [item.damage_irreversibility_violation for item in diagnostics], dtype=np.float64
    )
    relative_energy = np.asarray(
        [item.relative_energy_imbalance for item in diagnostics], dtype=np.float64
    )
    staggered_potential_energy_change = np.asarray(
        [step.energy_change for step in accepted_steps], dtype=np.float64
    )

    grouped_ledger: list[list[Mapping[str, Any]]] = [[] for _ in range(step_count)]
    for entry in ledger:
        grouped_ledger[int(entry["step_index"])].append(entry)
    newton_iterations = np.asarray(
        [sum(int(entry["newton_iterations"]) for entry in group) for group in grouped_ledger],
        dtype=np.int64,
    )
    active_iterations = np.asarray(
        [sum(int(entry["active_set_iterations"]) for entry in group) for group in grouped_ledger],
        dtype=np.int64,
    )
    staggered_iterations = np.asarray(
        [sum(int(entry["staggered_iterations"]) for entry in group) for group in grouped_ledger],
        dtype=np.int64,
    )
    step_halvings = np.asarray(
        [int(group[-1]["step_halvings"]) for group in grouped_ledger], dtype=np.int64
    )
    retries = np.asarray([len(group) - 1 for group in grouped_ledger], dtype=np.int64)

    force_values = np.asarray(
        [item.global_force_residual_yz for item in diagnostics], dtype=np.float64
    )
    force_relative = np.asarray(
        [item.global_force_relative_residual for item in diagnostics], dtype=np.float64
    )
    moments = np.asarray([item.global_moment_residual for item in diagnostics], dtype=np.float64)
    moment_relative = np.asarray(
        [item.global_moment_relative_residual for item in diagnostics], dtype=np.float64
    )
    balance = GlobalBalanceDiagnostics(
        force_residual_yz=force_values,
        force_relative_residual=force_relative,
        moment_residual=moments,
        moment_relative_residual=moment_relative,
        reference_yz=schedule.wall_perimeter_centroid_yz,
    )

    mesh_metadata = dict(getattr(mesh, "metadata", {}))
    mesh_metadata.update(
        {
            "element_type": "triangle_p1",
            "displacement_interpolation": "P1",
            "damage_interpolation": "P1",
            "accepted_internal_steps_stored": True,
        }
    )
    environment = _default_env()
    environment.update(dict(identity.env or {}))
    caller_meta = dict(identity.meta or {})
    caller_meta.update(
        {
            "development_only": True,
            "formal_labels_allowed": False,
            "solver_energy_increment_residual_available": True,
            "solver_energy_increment_definition": (
                "symmetric_relative_total_potential_change_between_complete_staggered_iterates"
            ),
            "solver_energy_increment_tolerance": float(
                config["solver"]["relative_energy_increment_tolerance"]
            ),
            "solver_energy_increment_residual": [
                float(step.energy_change) for step in accepted_steps
            ],
            "required_output_indices": required_output_indices.tolist(),
            "required_output_s": load_parameter[required_output_indices].tolist(),
            "global_balance": {
                "reference_yz": balance.reference_yz.tolist(),
                "force_residual_yz": balance.force_residual_yz.tolist(),
                "force_relative_residual": balance.force_relative_residual.tolist(),
                "moment_residual": balance.moment_residual.tolist(),
                "moment_relative_residual": balance.moment_relative_residual.tolist(),
            },
            "damage_connectivity_definition": (
                "widest_path_wall_to_farfield_with_edge_capacity_min_endpoint_damage"
            ),
            "rollback_strategy": "fresh_complete_accepted_prefix_per_attempt",
        }
    )
    solver_metadata = dict(identity.solver)
    frozen_energy_tolerance = float(config["solver"]["relative_energy_increment_tolerance"])
    declared_energy_tolerance = solver_metadata.get("relative_energy_increment_tolerance")
    if declared_energy_tolerance is not None and float(declared_energy_tolerance) != (
        frozen_energy_tolerance
    ):
        raise FractureTrajectoryAdapterError(
            "identity solver energy tolerance differs from the frozen Phase-1 protocol"
        )
    solver_metadata["relative_energy_increment_tolerance"] = frozen_energy_tolerance
    physical_tags = {key: int(value) for key, value in dict(mesh.physical_tags).items()}
    trajectory = FractureTrajectory(
        nodes=nodes,
        node_ids=np.arange(nodes.shape[0], dtype=np.int64),
        displacement_dof_ids=np.arange(2 * nodes.shape[0], dtype=np.int64).reshape(-1, 2),
        damage_dof_ids=np.arange(nodes.shape[0], dtype=np.int64),
        elements=elements,
        wall_facets=wall_facets,
        farfield_facets=farfield_facets,
        farfield_dirichlet_dofs=dirichlet,
        area=area,
        centers=centers,
        load_parameter=load_parameter,
        farfield_stress=farfield_stress,
        wall_release_by_facet=wall_release,
        u=u,
        internal_nodal_force=internal,
        wall_nodal_force=wall_force,
        farfield_prescribed_displacement=prescribed,
        farfield_reaction_on_rock=reaction,
        damage=damage,
        strain=strain,
        stress=stress,
        sigma_xx=sigma_xx,
        psi_plus=psi_plus,
        psi_minus=psi_minus,
        history=history,
        elastic_energy=elastic_energy,
        fracture_energy=fracture_energy,
        neumann_load_functional=neumann,
        wall_work_increment=wall_work,
        farfield_work_increment=farfield_work,
        cumulative_external_work=cumulative_work,
        total_potential_energy=total_potential,
        damage_area=damage_area,
        crack_density_integral=crack_density,
        damage_connectivity=connectivity,
        displacement_residual=np.asarray(
            [step.equilibrium_residual for step in accepted_steps], dtype=np.float64
        ),
        damage_residual=np.asarray(
            [step.kkt_residual for step in accepted_steps], dtype=np.float64
        ),
        equilibrium_relative_residual=equilibrium,
        kkt_relative_residual=np.asarray(
            [step.kkt_residual for step in accepted_steps], dtype=np.float64
        ),
        complementarity_relative_residual=np.asarray(
            [step.complementarity_residual for step in accepted_steps], dtype=np.float64
        ),
        damage_irreversibility_violation=irreversibility,
        damage_range_violation=np.asarray(
            [step.range_violation for step in accepted_steps], dtype=np.float64
        ),
        history_monotonicity_violation=history_violation,
        relative_energy_imbalance=relative_energy,
        staggered_potential_energy_change=staggered_potential_energy_change,
        newton_iterations=newton_iterations,
        active_set_iterations=active_iterations,
        staggered_iterations=staggered_iterations,
        step_halvings=step_halvings,
        retry_count=retries,
        attempt_ledger=ledger,
        trajectory_id=identity.trajectory_id,
        case_id=identity.case_id,
        mesh_id=identity.mesh_id,
        geometry_id=identity.geometry_id,
        material_id=identity.material_id,
        load_path_id=schedule.path_id,
        config_hash=canonical_config_sha256(config),
        solver_hash=identity.solver_hash,
        equilibrium_force_normalization_floor=float(force_floor),
        energy_balance_normalization_floor=float(energy_floor),
        material={
            "young_modulus": material.young_modulus,
            "poisson_ratio": material.poisson_ratio,
            "fracture_energy": material.fracture_toughness,
            "length_scale": material.length_scale,
            "residual_stiffness": material.residual_stiffness,
            "fracture_model": "AT2",
            "energy_split": "spectral_strain_3d_plane_strain",
        },
        geometry=dict(identity.geometry),
        load_path=_phase1_load_path_metadata(config, schedule),
        physical_tags=physical_tags,
        mesh_metadata=mesh_metadata,
        solver=solver_metadata,
        env=environment,
        meta=caller_meta,
    )
    trajectory.validate()
    return trajectory, balance


def _validated_solver_options(
    config: Mapping[str, Any], options: FractureSolverOptions | None
) -> FractureSolverOptions:
    frozen = config["solver"]
    if options is None:
        return FractureSolverOptions(
            max_staggered_iterations=int(frozen["max_staggered_iterations"]),
            max_active_set_iterations=int(frozen["max_active_set_iterations"]),
            staggered_tolerance=min(
                float(frozen["relative_displacement_increment_tolerance"]),
                float(frozen["relative_damage_increment_tolerance"]),
            ),
            energy_tolerance=float(frozen["relative_energy_increment_tolerance"]),
            equilibrium_tolerance=float(frozen["equilibrium_relative_residual_tolerance"]),
            kkt_tolerance=float(frozen["kkt_complementarity_relative_residual_tolerance"]),
            raise_on_nonconvergence=False,
        )
    if options.max_staggered_iterations > int(frozen["max_staggered_iterations"]):
        raise ValueError("options exceed the frozen maximum staggered iterations")
    if options.max_active_set_iterations > int(frozen["max_active_set_iterations"]):
        raise ValueError("options exceed the frozen maximum active-set iterations")
    if options.staggered_tolerance > min(
        float(frozen["relative_displacement_increment_tolerance"]),
        float(frozen["relative_damage_increment_tolerance"]),
    ):
        raise ValueError("options use a looser staggered tolerance than the frozen protocol")
    if options.energy_tolerance > float(frozen["relative_energy_increment_tolerance"]):
        raise ValueError("options use a looser energy tolerance than the frozen protocol")
    if options.equilibrium_tolerance > float(frozen["equilibrium_relative_residual_tolerance"]):
        raise ValueError("options use a looser equilibrium tolerance than the frozen protocol")
    if options.kkt_tolerance > float(frozen["kkt_complementarity_relative_residual_tolerance"]):
        raise ValueError("options use a looser KKT tolerance than the frozen protocol")
    return replace(options, raise_on_nonconvergence=False)


def _validate_phase1_material_identity(
    material: AT2Material,
    schedule: Phase1LoadSchedule,
    config: Mapping[str, Any],
    identity: Phase1TrajectoryIdentity,
) -> None:
    section = identity.geometry.get("section_family")
    if section not in config["design"]["section_families"]:
        raise ValueError("identity.geometry.section_family is outside the frozen design")
    if identity.material_id not in config["design"]["material_level_ids"]:
        raise ValueError("identity.material_id is outside the frozen design")
    expected_case_id = f"fp1-{section}-{identity.material_id}-{schedule.path_id}"
    if identity.case_id != expected_case_id:
        raise ValueError(f"identity.case_id must equal frozen case identity {expected_case_id!r}")
    radius = float(config["geometry"]["characteristic_radius_R"])
    fixed = config["materials"]["fixed"]
    level = next(
        item for item in config["materials"]["levels"] if item["id"] == identity.material_id
    )
    expected = {
        "young_modulus": float(fixed["youngs_modulus_over_UCS"]) * schedule.ucs_scale,
        "poisson_ratio": float(fixed["poisson_ratio"]),
        "fracture_toughness": float(level["Gc_over_UCS_R"]) * schedule.ucs_scale * radius,
        "length_scale": float(level["ell_over_R"]) * radius,
        "residual_stiffness": float(config["fracture_model"]["residual_stiffness_k"]),
    }
    for name, target in expected.items():
        actual = float(getattr(material, name))
        if not math.isclose(
            actual, target, rel_tol=2.0e-13, abs_tol=2.0e-15 * max(abs(target), 1.0)
        ):
            raise ValueError(
                f"material.{name}={actual:.17g} differs from frozen Phase-1 value {target:.17g}"
            )


def run_phase1_development_trajectory(
    mesh: Any,
    material: AT2Material,
    schedule: Phase1LoadSchedule,
    config: Mapping[str, Any],
    identity: Phase1TrajectoryIdentity,
    *,
    equilibrium_force_normalization_floor: float,
    energy_balance_normalization_floor: float,
    options: FractureSolverOptions | None = None,
    solve_prefix: SolvePrefix = solve_at2_fracture_schedule,
) -> Phase1DevelopmentRun:
    """Run one adaptive, development-only Phase-1 trajectory.

    A retry always invokes ``solve_prefix`` on ``accepted_s + [candidate_s]``
    from a fresh zero state.  Previously accepted fields are compared bitwise
    with the recomputed prefix before the candidate is considered.  This costs
    more than a future stateful restart API but gives a strong rollback
    guarantee without modifying the numerical kernel.
    """

    validate_fracture_phase1_config(config)
    if not isinstance(material, AT2Material):
        raise TypeError("material must be AT2Material")
    if not isinstance(schedule, Phase1LoadSchedule):
        raise TypeError("schedule must be Phase1LoadSchedule")
    if schedule.path_id not in config["design"]["load_path_ids"]:
        raise ValueError("schedule path is outside the frozen Phase-1 design")
    _validate_phase1_material_identity(material, schedule, config, identity)
    force_floor = float(equilibrium_force_normalization_floor)
    energy_floor = float(energy_balance_normalization_floor)
    if not math.isfinite(force_floor) or force_floor <= 0.0:
        raise ValueError("equilibrium_force_normalization_floor must be finite and positive")
    if not math.isfinite(energy_floor) or energy_floor <= 0.0:
        raise ValueError("energy_balance_normalization_floor must be finite and positive")
    runtime_options = _validated_solver_options(config, options)
    nodes, elements, wall_facets, _ = _mesh_arrays(mesh)
    if not np.array_equal(wall_facets, np.asarray(mesh.mesh.facets)[:, schedule.wall_facet_ids].T):
        raise FractureTrajectoryAdapterError("ordered wall facets differ from the load schedule")

    required = np.asarray(config["time_discretization"]["required_output_s"], dtype=np.float64)
    if required.shape != (41,) or required[0] != 0.0 or required[-1] != 1.0:
        raise FractureTrajectoryAdapterError("frozen required output grid is not 41 states")
    solver_controls = config["solver"]
    max_retries = int(solver_controls["max_step_retries_per_required_output_interval"])
    minimum_increment = float(solver_controls["minimum_s_increment"])
    retry_factor = float(solver_controls["step_retry_factor"])
    if retry_factor != 0.5:
        raise FractureTrajectoryAdapterError("development runner implements frozen factor 0.5")

    accepted_steps: list[ScheduledAT2StepResult] = []
    accepted_boundaries: list[BoundaryEquilibriumState] = []
    accepted_diagnostics: list[_CandidateDiagnostics] = []
    ledger: list[dict[str, Any]] = []
    required_indices: list[int] = []
    cumulative_external = 0.0
    reference = np.asarray(schedule.wall_perimeter_centroid_yz, dtype=np.float64)

    def solve_attempt(target: float) -> tuple[ScheduledAT2StepResult, _CandidateDiagnostics]:
        parameters = tuple([step.load_parameter for step in accepted_steps] + [float(target)])
        try:
            result = solve_prefix(
                mesh,
                material,
                schedule,
                load_path=AT2LoadPath(parameters),
                options=runtime_options,
            )
        except Exception as exc:
            raise FractureTrajectoryRunFailed(
                "solver raised before returning auditable retry diagnostics: "
                f"{type(exc).__name__}: {exc}",
                attempt_ledger=ledger,
                accepted_load_parameters=[step.load_parameter for step in accepted_steps],
            ) from exc
        candidate = _validate_prefix_result(
            result,
            nodes,
            elements,
            schedule,
            material,
            runtime_options,
            parameters,
            accepted_steps,
        )
        diagnostics = _candidate_diagnostics(
            nodes,
            candidate,
            accepted_steps[-1] if accepted_steps else None,
            accepted_boundaries[-1] if accepted_boundaries else None,
            cumulative_external,
            reference,
            force_floor,
            energy_floor,
        )
        return candidate, diagnostics

    # The initial state defines the work origin and cannot be obtained by
    # halving an earlier increment.
    initial_step, initial_diagnostics = solve_attempt(0.0)
    initial_failure = _failure_reason(initial_step, initial_diagnostics, config)
    ledger.append(
        _ledger_entry(
            step_index=0,
            attempt_index=0,
            start=0.0,
            target=0.0,
            accepted=initial_failure is None,
            step=initial_step,
            diagnostics=initial_diagnostics,
            halvings=0,
            wall_facets=wall_facets,
            cumulative_external_work=0.0,
            failure=initial_failure,
        )
    )
    if initial_failure is not None:
        raise FractureTrajectoryRunFailed(
            f"initial s=0 state failed: {initial_failure[0]}",
            attempt_ledger=ledger,
            accepted_load_parameters=(),
        )
    accepted_steps.append(_copy_step(initial_step))
    accepted_boundaries.append(initial_diagnostics.boundary)
    accepted_diagnostics.append(initial_diagnostics)
    required_indices.append(0)

    for required_target in required[1:]:
        rejected_in_interval = 0
        while accepted_steps[-1].load_parameter < required_target:
            start = float(accepted_steps[-1].load_parameter)
            candidate_target = float(required_target)
            attempt_index = 0
            halvings = 0
            while True:
                candidate, candidate_diagnostics = solve_attempt(candidate_target)
                failure = _failure_reason(candidate, candidate_diagnostics, config)
                step_index = len(accepted_steps)
                if failure is None:
                    cumulative_external = candidate_diagnostics.cumulative_external_work
                    ledger.append(
                        _ledger_entry(
                            step_index=step_index,
                            attempt_index=attempt_index,
                            start=start,
                            target=candidate_target,
                            accepted=True,
                            step=candidate,
                            diagnostics=candidate_diagnostics,
                            halvings=halvings,
                            wall_facets=wall_facets,
                            cumulative_external_work=cumulative_external,
                            failure=None,
                        )
                    )
                    accepted_steps.append(_copy_step(candidate))
                    accepted_boundaries.append(candidate_diagnostics.boundary)
                    accepted_diagnostics.append(candidate_diagnostics)
                    break

                ledger.append(
                    _ledger_entry(
                        step_index=step_index,
                        attempt_index=attempt_index,
                        start=start,
                        target=candidate_target,
                        accepted=False,
                        step=candidate,
                        diagnostics=candidate_diagnostics,
                        halvings=halvings,
                        wall_facets=wall_facets,
                        cumulative_external_work=cumulative_external,
                        failure=failure,
                    )
                )
                rejected_in_interval += 1
                if rejected_in_interval > max_retries:
                    raise FractureTrajectoryRunFailed(
                        f"retry budget exhausted before required output s={required_target:.17g}",
                        attempt_ledger=ledger,
                        accepted_load_parameters=[step.load_parameter for step in accepted_steps],
                    )
                next_target = start + retry_factor * (candidate_target - start)
                if next_target - start < minimum_increment - 64.0 * _FLOAT_EPS:
                    raise FractureTrajectoryRunFailed(
                        "halved increment is below the frozen minimum_s_increment",
                        attempt_ledger=ledger,
                        accepted_load_parameters=[step.load_parameter for step in accepted_steps],
                    )
                candidate_target = float(next_target)
                attempt_index += 1
                halvings += 1
        if accepted_steps[-1].load_parameter != float(required_target):
            raise FractureTrajectoryAdapterError("required output target was not reached exactly")
        required_indices.append(len(accepted_steps) - 1)

    required_index_array = np.asarray(required_indices, dtype=np.int64)
    trajectory, balance = _schema_record(
        mesh=mesh,
        material=material,
        schedule=schedule,
        config=config,
        identity=identity,
        accepted_steps=accepted_steps,
        boundaries=accepted_boundaries,
        diagnostics=accepted_diagnostics,
        ledger=ledger,
        required_output_indices=required_index_array,
        force_floor=force_floor,
        energy_floor=energy_floor,
    )
    if not np.array_equal(trajectory.load_parameter[required_index_array], required):
        raise FractureTrajectoryAdapterError("required output extraction differs from frozen grid")
    return Phase1DevelopmentRun(
        trajectory=trajectory,
        required_output_indices=required_index_array,
        balance=balance,
    )


def save_and_verify_phase1_development_run(
    trajectory_dir: str | Path,
    run: Phase1DevelopmentRun,
    *,
    overwrite: bool = False,
) -> tuple[FractureTrajectoryPaths, FractureTrajectory]:
    """Persist, reload, and byte-compare every schema array for one run."""

    if not isinstance(run, Phase1DevelopmentRun):
        raise TypeError("run must be a Phase1DevelopmentRun")
    paths = save_fracture_trajectory(
        trajectory_dir, run.trajectory, overwrite=overwrite, expected_dtype=np.float64
    )
    loaded = load_fracture_trajectory(paths.trajectory_dir, expected_dtype=np.float64)
    for name, expected in run.trajectory.arrays().items():
        if not np.array_equal(expected, loaded.arrays()[name]):
            raise FractureTrajectoryAdapterError(f"save-load E2E changed schema array {name}")
    if dict(loaded.meta) != dict(run.trajectory.meta):
        raise FractureTrajectoryAdapterError("save-load E2E changed development metadata")
    return paths, loaded


__all__ = [
    "FractureTrajectoryAdapterError",
    "FractureTrajectoryRunFailed",
    "GlobalBalanceDiagnostics",
    "Phase1DevelopmentRun",
    "Phase1TrajectoryIdentity",
    "canonical_config_sha256",
    "damage_graph_connectivity",
    "run_phase1_development_trajectory",
    "save_and_verify_phase1_development_run",
]
