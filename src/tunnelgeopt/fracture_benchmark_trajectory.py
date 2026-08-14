"""Fail-closed infrastructure for bounded SENT/SENS coupled prefixes.

This module is development-only.  It does not run a finite-element solve by
itself and it does not authorize a formal SENT/SENS trajectory.  Its purpose
is to freeze the state, retry, QC, and ledger semantics that a future solver
adapter must satisfy.  The exact current v1.2 protocol is accepted only after
its strict canonical validator passes; the pinned legacy v1.1 shape remains
rejected because it lacks complete solver and adaptive controls.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .fracture import FractureSolverOptions
from .fracture_benchmark_validation import validate_fracture_sent_sens_config

FloatArray = NDArray[np.float64]

READY_DEVELOPMENT_PREFIX_ONLY = "READY_DEVELOPMENT_PREFIX_ONLY"
NOT_READY_MISSING_FROZEN_CONTROLS = "NOT_READY_MISSING_FROZEN_CONTROLS"
NOT_READY_PROTOCOL_EXTENSION_REQUIRED = "NOT_READY_PROTOCOL_EXTENSION_REQUIRED"
NOT_READY_INVALID_FROZEN_CONTROLS = "NOT_READY_INVALID_FROZEN_CONTROLS"

STOP_INVALID = "STOP_INVALID"
STOP_NUMERICAL = "STOP_NUMERICAL"
DEVELOPMENT_PREFIX_ACCEPTED = "DEVELOPMENT_PREFIX_ACCEPTED"

_RETRYABLE_NUMERICAL_CODES = (
    "QC_NONCONVERGED",
    "QC_EQUILIBRIUM",
    "QC_KKT",
    "QC_DU",
    "QC_DD",
    "QC_DPI",
    "QC_PATH_ENERGY",
)
_NONRETRYABLE_INVALID_CODES = (
    "QC_NONFINITE",
    "QC_IRREVERSIBILITY",
    "QC_RANGE",
    "QC_GLOBAL_FORCE",
    "QC_GLOBAL_MOMENT",
    "QC_REACTION",
    "SOLVER_EXCEPTION",
    STOP_INVALID,
)

# These literals identify only the immutable validate-only v1.1 artifact.  Do
# not import the validator's current identity here: when that validator moves
# to v1.2, an imported constant could silently turn the new protocol into the
# legacy protocol that this runner must reject.
_LEGACY_V11_PROTOCOL_ID = "miehe-sent-sens-three-grid-development-v1.1"
_LEGACY_V11_CANONICAL_SHA256 = "d10036cfe1a0fa54600acae5d5f04425014074ec3d8ebace9e8f284251d8a20d"
_MISSING = object()
_FROZEN_CONTROL_PATHS = (
    "solver.max_displacement_iterations",
    "solver.line_search_steps",
    "solver.active_set_tolerance",
    "solver.tangent_perturbation",
    "solver.raise_on_nonconvergence",
    "solver.adaptive_bisection.factor",
    "solver.adaptive_bisection.max_retry_depth",
    "solver.adaptive_bisection.minimum_increment_mm",
    "solver.adaptive_bisection.retryable_codes",
    "solver.adaptive_bisection.retry_exhausted_action",
    "solver.adaptive_bisection.max_rejected_attempts_per_required_interval",
    "per_tier_qc.max_damage_range_violation",
    "per_tier_qc.force_balance_normalization_floor_kN",
    "per_tier_qc.moment_balance_normalization_floor_kN_mm",
    "per_tier_qc.path_energy_normalization_floor_kN_mm",
    "per_tier_qc.global_moment_origin_yz_mm",
)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lookup(config: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _readonly_float(value: Any, *, name: str, ndim: int | None = None) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    # A normal owning ndarray can undo ``setflags(write=False)``.  Rebuild the
    # copy on an immutable ``bytes`` buffer so neither callers nor downstream
    # code can make it writeable and silently stale a content hash.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float64).reshape(array.shape)


def _finite_number(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _nonnegative_number(value: Any, *, name: str) -> float:
    result = _finite_number(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _options_and_policy_sha256(
    options: FractureSolverOptions, policy: AdaptiveBisectionPolicy
) -> str:
    payload = {
        "schema": "tunnelgeopt.fracture.runtime_controls.v1",
        "solver_options": asdict(options),
        "adaptive_bisection": asdict(policy),
    }
    return _canonical_sha256(payload)


@dataclass(frozen=True)
class AdaptiveBisectionPolicy:
    """All retry controls, sourced only from a versioned protocol."""

    factor: float
    max_retry_depth: int
    minimum_increment_mm: float
    retryable_codes: tuple[str, ...]
    retry_exhausted_action: str
    max_rejected_attempts_per_required_interval: int

    def __post_init__(self) -> None:
        if self.factor != 0.5:
            raise ValueError("adaptive_bisection.factor must be exactly 0.5")
        _nonnegative_int(self.max_retry_depth, name="adaptive_bisection.max_retry_depth")
        _finite_number(
            self.minimum_increment_mm,
            name="adaptive_bisection.minimum_increment_mm",
            positive=True,
        )
        if self.retryable_codes != _RETRYABLE_NUMERICAL_CODES:
            raise ValueError(
                "adaptive_bisection.retryable_codes must equal the frozen seven-code order"
            )
        if self.retry_exhausted_action != STOP_NUMERICAL:
            raise ValueError("adaptive_bisection.retry_exhausted_action must be STOP_NUMERICAL")
        _positive_int(
            self.max_rejected_attempts_per_required_interval,
            name="adaptive_bisection.max_rejected_attempts_per_required_interval",
        )


@dataclass(frozen=True)
class FrozenTrajectoryQC:
    """Frozen thresholds and normalization conventions for one step."""

    max_nonfinite_fraction: float
    max_equilibrium_relative_residual: float
    max_kkt_complementarity_relative_residual: float
    max_relative_displacement_increment: float
    max_relative_damage_increment: float
    max_relative_potential_energy_increment: float
    max_damage_irreversibility_violation: float
    max_damage_range_violation: float
    max_global_force_relative_imbalance: float
    max_global_moment_relative_imbalance: float
    max_path_energy_relative_imbalance: float
    force_balance_normalization_floor_kN: float
    moment_balance_normalization_floor_kN_mm: float
    path_energy_normalization_floor_kN_mm: float
    global_moment_origin_yz_mm: tuple[float, float]


@dataclass(frozen=True)
class CoupledPrefixPreflight:
    """Structured refusal or exact runtime controls for a bounded prefix."""

    status: str
    protocol_sha256: str
    missing_controls: tuple[str, ...] = ()
    detail: str = ""
    options: FractureSolverOptions | None = None
    options_sha256: str | None = None
    adaptive_policy: AdaptiveBisectionPolicy | None = None
    qc: FrozenTrajectoryQC | None = None

    @property
    def ready(self) -> bool:
        return self.status == READY_DEVELOPMENT_PREFIX_ONLY


def preflight_coupled_coarse_prefix(config: Mapping[str, Any]) -> CoupledPrefixPreflight:
    """Build every runtime control explicitly, or return a fail-closed status."""

    try:
        protocol_sha = _canonical_sha256(config)
    except (TypeError, ValueError) as exc:
        return CoupledPrefixPreflight(
            status=NOT_READY_INVALID_FROZEN_CONTROLS,
            protocol_sha256="",
            detail=f"protocol is not canonical finite JSON: {exc}",
        )

    is_legacy_v11 = config.get("protocol_id") == _LEGACY_V11_PROTOCOL_ID
    missing = tuple(path for path in _FROZEN_CONTROL_PATHS if _lookup(config, path) is _MISSING)
    if is_legacy_v11 and missing:
        return CoupledPrefixPreflight(
            status=NOT_READY_MISSING_FROZEN_CONTROLS,
            protocol_sha256=protocol_sha,
            missing_controls=missing,
            detail="protocol extension required before any coupled prefix solve",
        )

    if is_legacy_v11:
        detail = "the v1.1 identity is immutable; adding controls requires a new protocol version"
        if protocol_sha == _LEGACY_V11_CANONICAL_SHA256:
            detail = "v1.1 cannot be coupled-ready without a versioned protocol extension"
        return CoupledPrefixPreflight(
            status=NOT_READY_PROTOCOL_EXTENSION_REQUIRED,
            protocol_sha256=protocol_sha,
            detail=detail,
        )

    # Non-legacy execution is permitted only for the validator's exact current
    # protocol ID, semantics, and canonical hash.  Keep the legacy identity
    # literal-pinned above while letting the strict validator own the moving
    # current identity.  Never expose config contents or host paths in detail.
    try:
        validate_fracture_sent_sens_config(config)
    except Exception:  # noqa: BLE001 - arbitrary mappings must fail closed
        return CoupledPrefixPreflight(
            status=NOT_READY_INVALID_FROZEN_CONTROLS,
            protocol_sha256=protocol_sha,
            detail="current SENT/SENS protocol validation failed",
        )

    if missing:  # defensive: the strict current validator should make this unreachable
        return CoupledPrefixPreflight(
            status=NOT_READY_INVALID_FROZEN_CONTROLS,
            protocol_sha256=protocol_sha,
            detail="current SENT/SENS protocol validation failed",
        )

    try:
        solver = config["solver"]
        displacement_tolerance = _finite_number(
            solver["relative_displacement_increment_tolerance"],
            name="solver.relative_displacement_increment_tolerance",
            positive=True,
        )
        damage_tolerance = _finite_number(
            solver["relative_damage_increment_tolerance"],
            name="solver.relative_damage_increment_tolerance",
            positive=True,
        )
        options = FractureSolverOptions(
            max_staggered_iterations=_positive_int(
                solver["max_staggered_iterations"],
                name="solver.max_staggered_iterations",
            ),
            max_displacement_iterations=_positive_int(
                solver["max_displacement_iterations"],
                name="solver.max_displacement_iterations",
            ),
            max_active_set_iterations=_positive_int(
                solver["max_active_set_iterations"],
                name="solver.max_active_set_iterations",
            ),
            staggered_tolerance=min(displacement_tolerance, damage_tolerance),
            energy_tolerance=_finite_number(
                solver["relative_potential_energy_increment_tolerance"],
                name="solver.relative_potential_energy_increment_tolerance",
                positive=True,
            ),
            equilibrium_tolerance=_finite_number(
                solver["equilibrium_relative_residual_tolerance"],
                name="solver.equilibrium_relative_residual_tolerance",
                positive=True,
            ),
            kkt_tolerance=_finite_number(
                solver["kkt_complementarity_relative_residual_tolerance"],
                name="solver.kkt_complementarity_relative_residual_tolerance",
                positive=True,
            ),
            active_set_tolerance=_finite_number(
                solver["active_set_tolerance"],
                name="solver.active_set_tolerance",
                positive=True,
            ),
            line_search_steps=_positive_int(
                solver["line_search_steps"], name="solver.line_search_steps"
            ),
            tangent_perturbation=_finite_number(
                solver["tangent_perturbation"],
                name="solver.tangent_perturbation",
                positive=True,
            ),
            raise_on_nonconvergence=solver["raise_on_nonconvergence"],
        )
        if not isinstance(solver["raise_on_nonconvergence"], bool):
            raise TypeError("solver.raise_on_nonconvergence must be boolean")
        if solver["raise_on_nonconvergence"]:
            raise ValueError(
                "solver.raise_on_nonconvergence must be false so failed attempts can be ledgered"
            )
        if solver.get("accepted_unconverged_step_allowed") is not False:
            raise ValueError("solver.accepted_unconverged_step_allowed must remain false")

        adaptive = solver["adaptive_bisection"]
        retryable = adaptive["retryable_codes"]
        if isinstance(retryable, (str, bytes)) or not isinstance(retryable, Sequence):
            raise TypeError("solver.adaptive_bisection.retryable_codes must be an array")
        policy = AdaptiveBisectionPolicy(
            factor=_finite_number(
                adaptive["factor"], name="adaptive_bisection.factor", positive=True
            ),
            max_retry_depth=_nonnegative_int(
                adaptive["max_retry_depth"], name="adaptive_bisection.max_retry_depth"
            ),
            minimum_increment_mm=_finite_number(
                adaptive["minimum_increment_mm"],
                name="adaptive_bisection.minimum_increment_mm",
                positive=True,
            ),
            retryable_codes=tuple(retryable),
            retry_exhausted_action=str(adaptive["retry_exhausted_action"]),
            max_rejected_attempts_per_required_interval=_positive_int(
                adaptive["max_rejected_attempts_per_required_interval"],
                name=("adaptive_bisection.max_rejected_attempts_per_required_interval"),
            ),
        )

        frozen_qc = config["per_tier_qc"]
        origin = frozen_qc["global_moment_origin_yz_mm"]
        if isinstance(origin, (str, bytes)) or not isinstance(origin, Sequence) or len(origin) != 2:
            raise ValueError("per_tier_qc.global_moment_origin_yz_mm must have length two")
        qc = FrozenTrajectoryQC(
            max_nonfinite_fraction=_nonnegative_number(
                frozen_qc["max_nonfinite_fraction"],
                name="per_tier_qc.max_nonfinite_fraction",
            ),
            max_equilibrium_relative_residual=_nonnegative_number(
                frozen_qc["max_equilibrium_relative_residual"],
                name="per_tier_qc.max_equilibrium_relative_residual",
            ),
            max_kkt_complementarity_relative_residual=_nonnegative_number(
                frozen_qc["max_kkt_complementarity_relative_residual"],
                name="per_tier_qc.max_kkt_complementarity_relative_residual",
            ),
            max_relative_displacement_increment=_nonnegative_number(
                frozen_qc["max_relative_displacement_increment"],
                name="per_tier_qc.max_relative_displacement_increment",
            ),
            max_relative_damage_increment=_nonnegative_number(
                frozen_qc["max_relative_damage_increment"],
                name="per_tier_qc.max_relative_damage_increment",
            ),
            max_relative_potential_energy_increment=_nonnegative_number(
                frozen_qc["max_relative_potential_energy_increment"],
                name="per_tier_qc.max_relative_potential_energy_increment",
            ),
            max_damage_irreversibility_violation=_nonnegative_number(
                frozen_qc["max_damage_irreversibility_violation"],
                name="per_tier_qc.max_damage_irreversibility_violation",
            ),
            max_damage_range_violation=_nonnegative_number(
                frozen_qc["max_damage_range_violation"],
                name="per_tier_qc.max_damage_range_violation",
            ),
            max_global_force_relative_imbalance=_nonnegative_number(
                frozen_qc["max_global_force_relative_imbalance"],
                name="per_tier_qc.max_global_force_relative_imbalance",
            ),
            max_global_moment_relative_imbalance=_nonnegative_number(
                frozen_qc["max_global_moment_relative_imbalance"],
                name="per_tier_qc.max_global_moment_relative_imbalance",
            ),
            max_path_energy_relative_imbalance=_nonnegative_number(
                frozen_qc["max_path_energy_relative_imbalance"],
                name="per_tier_qc.max_path_energy_relative_imbalance",
            ),
            force_balance_normalization_floor_kN=_finite_number(
                frozen_qc["force_balance_normalization_floor_kN"],
                name="per_tier_qc.force_balance_normalization_floor_kN",
                positive=True,
            ),
            moment_balance_normalization_floor_kN_mm=_finite_number(
                frozen_qc["moment_balance_normalization_floor_kN_mm"],
                name="per_tier_qc.moment_balance_normalization_floor_kN_mm",
                positive=True,
            ),
            path_energy_normalization_floor_kN_mm=_finite_number(
                frozen_qc["path_energy_normalization_floor_kN_mm"],
                name="per_tier_qc.path_energy_normalization_floor_kN_mm",
                positive=True,
            ),
            global_moment_origin_yz_mm=(
                _finite_number(origin[0], name="per_tier_qc.global_moment_origin_yz_mm[0]"),
                _finite_number(origin[1], name="per_tier_qc.global_moment_origin_yz_mm[1]"),
            ),
        )
        if qc.max_nonfinite_fraction != 0.0:
            raise ValueError("per_tier_qc.max_nonfinite_fraction must remain zero")
    except (KeyError, TypeError, ValueError) as exc:
        return CoupledPrefixPreflight(
            status=NOT_READY_INVALID_FROZEN_CONTROLS,
            protocol_sha256=protocol_sha,
            detail=str(exc),
        )

    return CoupledPrefixPreflight(
        status=READY_DEVELOPMENT_PREFIX_ONLY,
        protocol_sha256=protocol_sha,
        detail="bounded coarse prefix only; no formal-run authorization",
        options=options,
        options_sha256=_options_and_policy_sha256(options, policy),
        adaptive_policy=policy,
        qc=qc,
    )


def _hash_array(hasher: Any, name: str, value: FloatArray) -> None:
    hasher.update(name.encode("utf-8"))
    hasher.update(str(value.dtype).encode("ascii"))
    hasher.update(json.dumps(value.shape, separators=(",", ":")).encode("ascii"))
    hasher.update(value.tobytes(order="C"))


@dataclass(frozen=True)
class RestartCheckpoint:
    """Immutable accepted-state restart payload with a content hash."""

    U_mm: float
    displacement_yz_mm: FloatArray
    damage: FloatArray
    history_kN_per_mm2: FloatArray
    reaction_yz_kN: FloatArray
    path_work_kN_mm: float
    total_potential_energy_kN_mm: float
    mesh_sha256: str
    protocol_sha256: str
    options_sha256: str
    checkpoint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        U_mm = _finite_number(self.U_mm, name="U_mm")
        if U_mm < 0.0:
            raise ValueError("U_mm must be finite and nonnegative")
        path_work = _finite_number(self.path_work_kN_mm, name="path_work_kN_mm")
        total_potential = _finite_number(
            self.total_potential_energy_kN_mm,
            name="total_potential_energy_kN_mm",
        )
        displacement = _readonly_float(self.displacement_yz_mm, name="displacement_yz_mm", ndim=2)
        reaction = _readonly_float(self.reaction_yz_kN, name="reaction_yz_kN", ndim=2)
        damage = _readonly_float(self.damage, name="damage", ndim=1)
        history = _readonly_float(self.history_kN_per_mm2, name="history_kN_per_mm2", ndim=1)
        if displacement.shape[1] != 2 or reaction.shape != displacement.shape:
            raise ValueError("displacement_yz_mm and reaction_yz_kN must have shape [N,2]")
        if damage.shape != (displacement.shape[0],):
            raise ValueError("damage must have one entry per node")
        for name, array in (
            ("displacement_yz_mm", displacement),
            ("damage", damage),
            ("history_kN_per_mm2", history),
            ("reaction_yz_kN", reaction),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        for name in ("mesh_sha256", "protocol_sha256", "options_sha256"):
            digest = getattr(self, name)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
        object.__setattr__(self, "U_mm", U_mm)
        object.__setattr__(self, "path_work_kN_mm", path_work)
        object.__setattr__(self, "total_potential_energy_kN_mm", total_potential)
        object.__setattr__(self, "displacement_yz_mm", displacement)
        object.__setattr__(self, "reaction_yz_kN", reaction)
        object.__setattr__(self, "damage", damage)
        object.__setattr__(self, "history_kN_per_mm2", history)

        hasher = hashlib.sha256()
        metadata = {
            "schema": "tunnelgeopt.fracture.sent_sens.restart.v1",
            "U_mm": float(self.U_mm),
            "path_work_kN_mm": float(self.path_work_kN_mm),
            "total_potential_energy_kN_mm": float(self.total_potential_energy_kN_mm),
            "mesh_sha256": self.mesh_sha256,
            "protocol_sha256": self.protocol_sha256,
            "options_sha256": self.options_sha256,
        }
        hasher.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        _hash_array(hasher, "displacement_yz_mm", displacement)
        _hash_array(hasher, "damage", damage)
        _hash_array(hasher, "history_kN_per_mm2", history)
        _hash_array(hasher, "reaction_yz_kN", reaction)
        object.__setattr__(self, "checkpoint_sha256", hasher.hexdigest())


@dataclass(frozen=True)
class CoupledStepCandidate:
    """One injectable solver result before the runner decides acceptance."""

    checkpoint: RestartCheckpoint
    applied_nodal_force_yz_kN: FloatArray
    converged: bool
    equilibrium_relative_residual: float
    kkt_relative_residual: float
    complementarity_relative_residual: float
    relative_displacement_increment: float
    relative_damage_increment: float
    relative_potential_energy_increment: float
    reported_irreversibility_violation: float
    reported_range_violation: float
    generalized_reaction_magnitude_kN: float
    peak_rss_bytes: int

    def __post_init__(self) -> None:
        applied = _readonly_float(
            self.applied_nodal_force_yz_kN,
            name="applied_nodal_force_yz_kN",
            ndim=2,
        )
        if applied.shape != self.checkpoint.reaction_yz_kN.shape:
            raise ValueError("applied_nodal_force_yz_kN must have shape [N,2]")
        if not np.isfinite(applied).all():
            raise ValueError("applied_nodal_force_yz_kN must contain only finite values")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be boolean")
        if (
            not isinstance(self.peak_rss_bytes, int)
            or isinstance(self.peak_rss_bytes, bool)
            or self.peak_rss_bytes < 0
        ):
            raise ValueError("peak_rss_bytes must be a nonnegative integer")
        for name in (
            "equilibrium_relative_residual",
            "kkt_relative_residual",
            "complementarity_relative_residual",
            "relative_displacement_increment",
            "relative_damage_increment",
            "relative_potential_energy_increment",
            "reported_irreversibility_violation",
            "reported_range_violation",
            "generalized_reaction_magnitude_kN",
        ):
            value = _nonnegative_number(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "applied_nodal_force_yz_kN", applied)


@dataclass(frozen=True)
class StepQCEvaluation:
    """Every independent per-step gate and the formula-derived metrics."""

    evaluated: bool
    finite_passed: bool
    converged_passed: bool
    equilibrium_passed: bool
    kkt_passed: bool
    displacement_increment_passed: bool
    damage_increment_passed: bool
    potential_energy_increment_passed: bool
    irreversibility_passed: bool
    range_passed: bool
    global_force_passed: bool
    global_moment_passed: bool
    path_energy_passed: bool
    reaction_passed: bool
    corridor_passed: bool
    irreversibility_violation: float
    range_violation: float
    global_force_relative_imbalance: float
    global_moment_relative_imbalance: float
    path_energy_relative_imbalance: float
    failed_codes: tuple[str, ...]

    @property
    def all_passed(self) -> bool:
        return self.evaluated and not self.failed_codes

    @classmethod
    def not_evaluated(cls) -> StepQCEvaluation:
        return cls(
            evaluated=False,
            finite_passed=False,
            converged_passed=False,
            equilibrium_passed=False,
            kkt_passed=False,
            displacement_increment_passed=False,
            damage_increment_passed=False,
            potential_energy_increment_passed=False,
            irreversibility_passed=False,
            range_passed=False,
            global_force_passed=False,
            global_moment_passed=False,
            path_energy_passed=False,
            reaction_passed=False,
            corridor_passed=False,
            irreversibility_violation=math.inf,
            range_violation=math.inf,
            global_force_relative_imbalance=math.inf,
            global_moment_relative_imbalance=math.inf,
            path_energy_relative_imbalance=math.inf,
            failed_codes=("NOT_EVALUATED",),
        )


CorridorCallback = Callable[[RestartCheckpoint, float], bool]
SingleStepSolver = Callable[[RestartCheckpoint, float, FractureSolverOptions], CoupledStepCandidate]


def evaluate_step_qc(
    previous: RestartCheckpoint,
    candidate: CoupledStepCandidate,
    nodes_yz_mm: FloatArray,
    controls: FrozenTrajectoryQC,
    *,
    damage_component_threshold: float,
    corridor_callback: CorridorCallback,
) -> StepQCEvaluation:
    """Evaluate all frozen gates without mutating either restart checkpoint."""

    nodes = np.asarray(nodes_yz_mm, dtype=np.float64)
    if nodes.shape != candidate.checkpoint.reaction_yz_kN.shape:
        raise ValueError("nodes_yz_mm must match checkpoint reaction shape [N,2]")
    threshold = _finite_number(damage_component_threshold, name="damage_component_threshold")
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("damage_component_threshold must lie in [0,1]")
    if not callable(corridor_callback):
        raise TypeError("corridor_callback must be callable")

    checkpoint = candidate.checkpoint
    scalar_metrics = np.asarray(
        [
            checkpoint.U_mm,
            checkpoint.path_work_kN_mm,
            checkpoint.total_potential_energy_kN_mm,
            candidate.equilibrium_relative_residual,
            candidate.kkt_relative_residual,
            candidate.complementarity_relative_residual,
            candidate.relative_displacement_increment,
            candidate.relative_damage_increment,
            candidate.relative_potential_energy_increment,
            candidate.reported_irreversibility_violation,
            candidate.reported_range_violation,
            candidate.generalized_reaction_magnitude_kN,
        ],
        dtype=np.float64,
    )
    finite_passed = bool(
        np.isfinite(nodes).all()
        and np.isfinite(checkpoint.displacement_yz_mm).all()
        and np.isfinite(checkpoint.damage).all()
        and np.isfinite(checkpoint.history_kN_per_mm2).all()
        and np.isfinite(checkpoint.reaction_yz_kN).all()
        and np.isfinite(candidate.applied_nodal_force_yz_kN).all()
        and np.isfinite(scalar_metrics).all()
    )

    equilibrium_passed = bool(
        finite_passed
        and candidate.equilibrium_relative_residual <= controls.max_equilibrium_relative_residual
    )
    kkt_metric = max(candidate.kkt_relative_residual, candidate.complementarity_relative_residual)
    kkt_passed = bool(
        finite_passed and kkt_metric <= controls.max_kkt_complementarity_relative_residual
    )
    du_passed = bool(
        finite_passed
        and candidate.relative_displacement_increment
        <= controls.max_relative_displacement_increment
    )
    dd_passed = bool(
        finite_passed
        and candidate.relative_damage_increment <= controls.max_relative_damage_increment
    )
    dpi_passed = bool(
        finite_passed
        and candidate.relative_potential_energy_increment
        <= controls.max_relative_potential_energy_increment
    )

    computed_irreversibility = float(
        np.maximum(previous.damage - checkpoint.damage, 0.0).max(initial=0.0)
    )
    irreversibility = max(
        computed_irreversibility, float(candidate.reported_irreversibility_violation)
    )
    irreversibility_passed = bool(
        finite_passed and irreversibility <= controls.max_damage_irreversibility_violation
    )
    computed_range = float(
        max(
            np.maximum(-checkpoint.damage, 0.0).max(initial=0.0),
            np.maximum(checkpoint.damage - 1.0, 0.0).max(initial=0.0),
        )
    )
    range_violation = max(computed_range, float(candidate.reported_range_violation))
    range_passed = bool(finite_passed and range_violation <= controls.max_damage_range_violation)

    support = np.asarray(checkpoint.reaction_yz_kN)
    applied = np.asarray(candidate.applied_nodal_force_yz_kN)
    resultant = support + applied
    force_numerator = float(np.linalg.norm(resultant.sum(axis=0)))
    force_denominator = max(
        float(np.linalg.norm(support, axis=1).sum() + np.linalg.norm(applied, axis=1).sum()),
        controls.force_balance_normalization_floor_kN,
    )
    global_force = force_numerator / force_denominator

    relative_nodes = nodes - np.asarray(controls.global_moment_origin_yz_mm)
    support_moments = relative_nodes[:, 0] * support[:, 1] - relative_nodes[:, 1] * support[:, 0]
    applied_moments = relative_nodes[:, 0] * applied[:, 1] - relative_nodes[:, 1] * applied[:, 0]
    moment_numerator = abs(float((support_moments + applied_moments).sum()))
    moment_denominator = max(
        float(np.abs(support_moments).sum() + np.abs(applied_moments).sum()),
        controls.moment_balance_normalization_floor_kN_mm,
    )
    global_moment = moment_numerator / moment_denominator
    global_force_passed = bool(
        finite_passed and global_force <= controls.max_global_force_relative_imbalance
    )
    global_moment_passed = bool(
        finite_passed and global_moment <= controls.max_global_moment_relative_imbalance
    )

    potential_increment = (
        checkpoint.total_potential_energy_kN_mm - previous.total_potential_energy_kN_mm
    )
    work_increment = checkpoint.path_work_kN_mm - previous.path_work_kN_mm
    path_energy = abs(potential_increment - work_increment) / max(
        abs(potential_increment),
        abs(work_increment),
        controls.path_energy_normalization_floor_kN_mm,
    )
    path_energy_passed = bool(
        finite_passed and path_energy <= controls.max_path_energy_relative_imbalance
    )
    reaction_passed = bool(finite_passed and candidate.generalized_reaction_magnitude_kN >= 0.0)
    corridor_passed = bool(corridor_callback(checkpoint, threshold))

    checks = (
        (finite_passed, "QC_NONFINITE"),
        (candidate.converged, "QC_NONCONVERGED"),
        (equilibrium_passed, "QC_EQUILIBRIUM"),
        (kkt_passed, "QC_KKT"),
        (du_passed, "QC_DU"),
        (dd_passed, "QC_DD"),
        (dpi_passed, "QC_DPI"),
        (irreversibility_passed, "QC_IRREVERSIBILITY"),
        (range_passed, "QC_RANGE"),
        (global_force_passed, "QC_GLOBAL_FORCE"),
        (global_moment_passed, "QC_GLOBAL_MOMENT"),
        (path_energy_passed, "QC_PATH_ENERGY"),
        (reaction_passed, "QC_REACTION"),
        (corridor_passed, STOP_INVALID),
    )
    return StepQCEvaluation(
        evaluated=True,
        finite_passed=finite_passed,
        converged_passed=bool(candidate.converged),
        equilibrium_passed=equilibrium_passed,
        kkt_passed=kkt_passed,
        displacement_increment_passed=du_passed,
        damage_increment_passed=dd_passed,
        potential_energy_increment_passed=dpi_passed,
        irreversibility_passed=irreversibility_passed,
        range_passed=range_passed,
        global_force_passed=global_force_passed,
        global_moment_passed=global_moment_passed,
        path_energy_passed=path_energy_passed,
        reaction_passed=reaction_passed,
        corridor_passed=corridor_passed,
        irreversibility_violation=irreversibility,
        range_violation=range_violation,
        global_force_relative_imbalance=global_force,
        global_moment_relative_imbalance=global_moment,
        path_energy_relative_imbalance=path_energy,
        failed_codes=tuple(code for passed, code in checks if not passed),
    )


@dataclass(frozen=True)
class AttemptLedgerEntry:
    """Complete immutable audit record for one attempted scalar target."""

    attempt_id: str
    required_state_index: int
    required_target_U_mm: float
    attempted_U_mm: float
    attempted_dU_mm: float
    retry_depth: int
    parent_attempt_id: str | None
    start_checkpoint_sha256: str
    result_checkpoint_sha256: str | None
    wall_seconds: float
    peak_rss_bytes: int
    exception_type: str | None
    exception_message: str | None
    code: str
    qc: StepQCEvaluation
    accepted: bool
    accepted_as_required_state: bool
    rejected_attempt_count_for_required_state: int


@dataclass(frozen=True)
class CoupledPrefixResult:
    """Bounded development result; required and adaptive states stay separate."""

    status: str
    detail: str
    benchmark_id: str
    preflight: CoupledPrefixPreflight
    required_targets_U_mm: tuple[float, ...]
    required_checkpoints: tuple[RestartCheckpoint, ...]
    accepted_checkpoints: tuple[RestartCheckpoint, ...]
    attempt_ledger: tuple[AttemptLedgerEntry, ...]
    authorizes_formal_run: bool = False


def _segment_grid(config: Mapping[str, Any], benchmark_id: str) -> tuple[float, ...]:
    source = config["compute_preflight"]["coarse_subsampled_loading"][benchmark_id]
    values: list[float] = []
    for segment_index, segment in enumerate(source["segments"]):
        start = _finite_number(segment["start_U_mm"], name=f"segments[{segment_index}].start_U_mm")
        end = _finite_number(segment["end_U_mm"], name=f"segments[{segment_index}].end_U_mm")
        increment = _finite_number(
            segment["increment_mm"],
            name=f"segments[{segment_index}].increment_mm",
            positive=True,
        )
        count = _positive_int(
            segment["accepted_increment_count"],
            name=f"segments[{segment_index}].accepted_increment_count",
        )
        expected_end = start + count * increment
        if not math.isclose(expected_end, end, rel_tol=0.0, abs_tol=2.0e-15):
            raise ValueError(f"segments[{segment_index}] endpoint/increment/count disagree")
        if not isinstance(segment["include_start"], bool) or not isinstance(
            segment["include_end"], bool
        ):
            raise TypeError("segment include_start/include_end must be boolean")
        indices = range(0 if segment["include_start"] else 1, count + 1)
        segment_values = [start + index * increment for index in indices]
        if not segment["include_end"]:
            segment_values = segment_values[:-1]
        values.extend(segment_values)
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or array[0] != 0.0 or np.any(np.diff(array) <= 0.0):
        raise ValueError("coarse required grid must start at zero and be strictly increasing")
    if int(source["required_state_count"]) != array.size:
        raise ValueError("coarse required_state_count differs from the segment grid")
    return tuple(float(value) for value in array)


def _validate_candidate_identity(
    candidate: CoupledStepCandidate,
    start: RestartCheckpoint,
    target_U_mm: float,
) -> None:
    if not isinstance(candidate, CoupledStepCandidate):
        raise TypeError("single_step_solver must return CoupledStepCandidate")
    checkpoint = candidate.checkpoint
    if checkpoint.U_mm != target_U_mm:
        raise ValueError("candidate checkpoint U_mm differs from attempted target")
    if checkpoint.U_mm < start.U_mm:
        raise ValueError("candidate checkpoint moves backward in displacement")
    for name in ("mesh_sha256", "protocol_sha256", "options_sha256"):
        if getattr(checkpoint, name) != getattr(start, name):
            raise ValueError(f"candidate checkpoint {name} differs from its restart")
    if checkpoint.displacement_yz_mm.shape != start.displacement_yz_mm.shape:
        raise ValueError("candidate displacement shape differs from its restart")
    if checkpoint.damage.shape != start.damage.shape:
        raise ValueError("candidate damage shape differs from its restart")
    if checkpoint.history_kN_per_mm2.shape != start.history_kN_per_mm2.shape:
        raise ValueError("candidate history shape differs from its restart")


def _default_peak_rss_reader() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError):
        return 0


def run_coupled_coarse_prefix(
    config: Mapping[str, Any],
    *,
    benchmark_id: str,
    initial_checkpoint: RestartCheckpoint,
    nodes_yz_mm: FloatArray,
    required_prefix_count: int,
    single_step_solver: SingleStepSolver,
    corridor_callback: CorridorCallback,
    clock: Callable[[], float] = time.perf_counter,
    peak_rss_reader: Callable[[], int] = _default_peak_rss_reader,
) -> CoupledPrefixResult:
    """Attempt an exact coarse-grid prefix with rollback-safe bisection.

    Adaptive states may be inserted between two required targets, but they are
    never stored as substitutes for a required state.  A rejected candidate is
    ledgered and discarded; every child attempt restarts from the last accepted
    checkpoint.
    """

    preflight = preflight_coupled_coarse_prefix(config)
    if not preflight.ready:
        return CoupledPrefixResult(
            status=preflight.status,
            detail=preflight.detail,
            benchmark_id=benchmark_id,
            preflight=preflight,
            required_targets_U_mm=(),
            required_checkpoints=(),
            accepted_checkpoints=(),
            attempt_ledger=(),
        )
    assert preflight.options is not None
    assert preflight.options_sha256 is not None
    assert preflight.adaptive_policy is not None
    assert preflight.qc is not None
    if not callable(single_step_solver):
        raise TypeError("single_step_solver must be callable")
    if not callable(corridor_callback):
        raise TypeError("corridor_callback must be callable")
    if benchmark_id not in {entry["id"] for entry in config["loading"]["benchmarks"]}:
        raise ValueError("benchmark_id is not present in the protocol")
    full_grid = _segment_grid(config, benchmark_id)
    prefix_count = _positive_int(required_prefix_count, name="required_prefix_count")
    if prefix_count > len(full_grid):
        raise ValueError("required_prefix_count exceeds the exact coarse required grid")
    required_targets = full_grid[:prefix_count]
    if initial_checkpoint.U_mm != required_targets[0]:
        raise ValueError("initial checkpoint must be the U=0 rollback seed")
    if initial_checkpoint.protocol_sha256 != preflight.protocol_sha256:
        raise ValueError("initial checkpoint protocol hash differs from preflight")
    if initial_checkpoint.options_sha256 != preflight.options_sha256:
        raise ValueError("initial checkpoint options hash differs from preflight")
    nodes = np.asarray(nodes_yz_mm, dtype=np.float64)
    if nodes.shape != initial_checkpoint.displacement_yz_mm.shape or not np.isfinite(nodes).all():
        raise ValueError("nodes_yz_mm must be finite and match checkpoint shape [N,2]")
    damage_threshold = _finite_number(
        config["mesh"]["damage_component_threshold"],
        name="mesh.damage_component_threshold",
    )
    if not 0.0 <= damage_threshold <= 1.0:
        raise ValueError("mesh.damage_component_threshold must lie in [0,1]")
    if config["mesh"].get("damage_escape_action") != STOP_INVALID:
        raise ValueError("mesh.damage_escape_action must remain STOP_INVALID")

    ledger: list[AttemptLedgerEntry] = []
    accepted: list[RestartCheckpoint] = []
    required: list[RestartCheckpoint] = []
    policy = preflight.adaptive_policy
    terminal_status: str | None = None
    terminal_detail = ""
    rejected_attempt_counts: dict[int, int] = {}

    def record_attempt(
        start: RestartCheckpoint,
        target: float,
        required_index: int,
        required_target: float,
        depth: int,
        parent_attempt_id: str | None,
        is_required_target: bool,
    ) -> tuple[RestartCheckpoint | None, str, str]:
        nonlocal terminal_status, terminal_detail
        attempt_id = f"attempt-{len(ledger) + 1:06d}"
        started = float(clock())
        before_rss = int(peak_rss_reader())
        candidate: CoupledStepCandidate | None = None
        returned_candidate: Any = None
        solver_returned = False
        exception_type: str | None = None
        exception_message: str | None = None
        qc = StepQCEvaluation.not_evaluated()
        code = "SOLVER_EXCEPTION"
        try:
            returned_candidate = single_step_solver(start, target, preflight.options)
            solver_returned = True
            _validate_candidate_identity(returned_candidate, start, target)
            candidate = returned_candidate
            qc = evaluate_step_qc(
                start,
                candidate,
                nodes,
                preflight.qc,
                damage_component_threshold=damage_threshold,
                corridor_callback=corridor_callback,
            )
            if qc.all_passed:
                code = "ACCEPTED_REQUIRED" if is_required_target else "ACCEPTED_ADAPTIVE"
            elif STOP_INVALID in qc.failed_codes:
                # Invalid geometry/topology/corridor evidence always outranks
                # retryable numerical failures from the same candidate.
                code = STOP_INVALID
            else:
                invalid_code = next(
                    (
                        failure
                        for failure in _NONRETRYABLE_INVALID_CODES
                        if failure in qc.failed_codes
                    ),
                    None,
                )
                retryable_code = next(
                    (
                        failure
                        for failure in _RETRYABLE_NUMERICAL_CODES
                        if failure in qc.failed_codes
                    ),
                    None,
                )
                # Unknown failure codes are invalid, never implicitly
                # retryable.  Preserve a known concrete invalid code in the
                # ledger when one is available.
                code = invalid_code or retryable_code or STOP_INVALID
        # Arbitrary exceptions are part of the injectable solver contract and
        # must be ledgered before the frozen retry policy can classify them.
        except Exception as exc:  # noqa: BLE001
            exception_type = type(exc).__name__
            exception_message = str(exc)
            code = STOP_INVALID if solver_returned else "SOLVER_EXCEPTION"
        completed = float(clock())
        wall_seconds = completed - started
        if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
            raise RuntimeError("clock produced an invalid attempt duration")
        after_rss = int(peak_rss_reader())
        peak_rss = max(
            before_rss,
            after_rss,
            candidate.peak_rss_bytes if candidate is not None else 0,
        )
        accepted_attempt = candidate is not None and qc.all_passed and code.startswith("ACCEPTED_")
        rejected_count = rejected_attempt_counts.get(required_index, 0)
        if not accepted_attempt:
            rejected_count += 1
            rejected_attempt_counts[required_index] = rejected_count
        ledger.append(
            AttemptLedgerEntry(
                attempt_id=attempt_id,
                required_state_index=required_index,
                required_target_U_mm=required_target,
                attempted_U_mm=target,
                attempted_dU_mm=target - start.U_mm,
                retry_depth=depth,
                parent_attempt_id=parent_attempt_id,
                start_checkpoint_sha256=start.checkpoint_sha256,
                result_checkpoint_sha256=(
                    candidate.checkpoint.checkpoint_sha256 if candidate is not None else None
                ),
                wall_seconds=wall_seconds,
                peak_rss_bytes=peak_rss,
                exception_type=exception_type,
                exception_message=exception_message,
                code=code,
                qc=qc,
                accepted=accepted_attempt,
                accepted_as_required_state=accepted_attempt and is_required_target,
                rejected_attempt_count_for_required_state=rejected_count,
            )
        )
        if accepted_attempt:
            assert candidate is not None
            accepted.append(candidate.checkpoint)
            return candidate.checkpoint, attempt_id, code
        if code in _NONRETRYABLE_INVALID_CODES or code not in policy.retryable_codes:
            terminal_status = STOP_INVALID
            terminal_detail = f"attempt {attempt_id} failed closed with nonretryable {code}"
        elif rejected_count >= policy.max_rejected_attempts_per_required_interval:
            terminal_status = policy.retry_exhausted_action
            terminal_detail = (
                f"required state {required_index} reached the frozen rejected-attempt limit "
                f"{policy.max_rejected_attempts_per_required_interval}"
            )
        return None, attempt_id, code

    def advance(
        start: RestartCheckpoint,
        target: float,
        required_index: int,
        required_target: float,
        depth: int,
        parent_attempt_id: str | None,
        is_required_target: bool,
    ) -> RestartCheckpoint | None:
        nonlocal terminal_status, terminal_detail
        checkpoint, attempt_id, code = record_attempt(
            start,
            target,
            required_index,
            required_target,
            depth,
            parent_attempt_id,
            is_required_target,
        )
        if checkpoint is not None or terminal_status is not None:
            return checkpoint
        increment = target - start.U_mm
        child_increment = policy.factor * increment
        retry_allowed = (
            code in policy.retryable_codes
            and increment > 0.0
            and depth < policy.max_retry_depth
            and child_increment >= policy.minimum_increment_mm
            and increment - child_increment >= policy.minimum_increment_mm
        )
        if not retry_allowed:
            terminal_status = policy.retry_exhausted_action
            terminal_detail = (
                f"attempt {attempt_id} failed with {code}; bisection is not allowed or exhausted"
            )
            return None
        midpoint = start.U_mm + child_increment
        if not start.U_mm < midpoint < target:
            terminal_status = STOP_INVALID
            terminal_detail = "adaptive midpoint did not lie strictly inside the failed interval"
            return None
        midpoint_checkpoint = advance(
            start,
            midpoint,
            required_index,
            required_target,
            depth + 1,
            attempt_id,
            False,
        )
        if midpoint_checkpoint is None:
            return None
        return advance(
            midpoint_checkpoint,
            target,
            required_index,
            required_target,
            depth + 1,
            attempt_id,
            is_required_target,
        )

    for required_index, target in enumerate(required_targets):
        start = accepted[-1] if accepted else initial_checkpoint
        checkpoint = advance(
            start,
            target,
            required_index,
            target,
            0,
            None,
            True,
        )
        if checkpoint is None:
            break
        if checkpoint.U_mm != target:
            terminal_status = STOP_INVALID
            terminal_detail = "adaptive checkpoint attempted to replace a required state"
            break
        required.append(checkpoint)

    if terminal_status is not None:
        status = terminal_status
        detail = terminal_detail
    elif len(required) != len(required_targets):
        status = STOP_INVALID
        detail = "required state accounting is incomplete"
    elif len({checkpoint.U_mm for checkpoint in required}) != len(required):
        status = STOP_INVALID
        detail = "a required state was stored more than once"
    else:
        status = DEVELOPMENT_PREFIX_ACCEPTED
        detail = "bounded coarse development prefix accepted; formal run remains unauthorized"
    return CoupledPrefixResult(
        status=status,
        detail=detail,
        benchmark_id=benchmark_id,
        preflight=preflight,
        required_targets_U_mm=required_targets,
        required_checkpoints=tuple(required),
        accepted_checkpoints=tuple(accepted),
        attempt_ledger=tuple(ledger),
    )


__all__ = [
    "DEVELOPMENT_PREFIX_ACCEPTED",
    "NOT_READY_INVALID_FROZEN_CONTROLS",
    "NOT_READY_MISSING_FROZEN_CONTROLS",
    "NOT_READY_PROTOCOL_EXTENSION_REQUIRED",
    "READY_DEVELOPMENT_PREFIX_ONLY",
    "STOP_INVALID",
    "STOP_NUMERICAL",
    "AdaptiveBisectionPolicy",
    "AttemptLedgerEntry",
    "CoupledPrefixPreflight",
    "CoupledPrefixResult",
    "CoupledStepCandidate",
    "FrozenTrajectoryQC",
    "RestartCheckpoint",
    "StepQCEvaluation",
    "evaluate_step_qc",
    "preflight_coupled_coarse_prefix",
    "run_coupled_coarse_prefix",
]
