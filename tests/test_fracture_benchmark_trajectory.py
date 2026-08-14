from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import tunnelgeopt.fracture_benchmark_trajectory as trajectory_module
from tunnelgeopt.fracture_benchmark_trajectory import (
    DEVELOPMENT_PREFIX_ACCEPTED,
    NOT_READY_INVALID_FROZEN_CONTROLS,
    NOT_READY_MISSING_FROZEN_CONTROLS,
    READY_DEVELOPMENT_PREFIX_ONLY,
    STOP_INVALID,
    CoupledStepCandidate,
    RestartCheckpoint,
    evaluate_step_qc,
    preflight_coupled_coarse_prefix,
    run_coupled_coarse_prefix,
)


def _base_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "fracture_sent_sens_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_v11_config() -> dict:
    """Reconstruct the exact pinned v1.1 control shape from current v1.2."""

    config = copy.deepcopy(_base_config())
    config["protocol_id"] = "miehe-sent-sens-three-grid-development-v1.1"
    for key in (
        "max_displacement_iterations",
        "line_search_steps",
        "active_set_tolerance",
        "tangent_perturbation",
        "raise_on_nonconvergence",
        "adaptive_bisection",
    ):
        del config["solver"][key]
    for key in (
        "max_damage_range_violation",
        "force_balance_normalization_floor_kN",
        "moment_balance_normalization_floor_kN_mm",
        "path_energy_normalization_floor_kN_mm",
        "global_moment_origin_yz_mm",
    ):
        del config["per_tier_qc"][key]
    return config


def _checkpoint(
    U_mm: float,
    protocol_sha256: str,
    options_sha256: str,
    *,
    work: float | None = None,
    potential: float | None = None,
    damage: np.ndarray | None = None,
    reaction: np.ndarray | None = None,
) -> RestartCheckpoint:
    return RestartCheckpoint(
        U_mm=U_mm,
        displacement_yz_mm=np.asarray([[U_mm, 0.0], [U_mm, 0.0]]),
        damage=np.zeros(2) if damage is None else damage,
        history_kN_per_mm2=np.zeros(1),
        reaction_yz_kN=(np.asarray([[-1.0, 0.0], [1.0, 0.0]]) if reaction is None else reaction),
        path_work_kN_mm=U_mm if work is None else work,
        total_potential_energy_kN_mm=U_mm if potential is None else potential,
        mesh_sha256="a" * 64,
        protocol_sha256=protocol_sha256,
        options_sha256=options_sha256,
    )


def _candidate(
    start: RestartCheckpoint,
    target: float,
    *,
    equilibrium: float = 0.0,
    converged: bool = True,
    work: float | None = None,
    potential: float | None = None,
    damage: np.ndarray | None = None,
    reaction: np.ndarray | None = None,
    generalized_reaction: float = 1.0,
    **diagnostics: float,
) -> CoupledStepCandidate:
    checkpoint = _checkpoint(
        target,
        start.protocol_sha256,
        start.options_sha256,
        work=work,
        potential=potential,
        damage=damage,
        reaction=reaction,
    )
    return CoupledStepCandidate(
        checkpoint=checkpoint,
        applied_nodal_force_yz_kN=np.zeros((2, 2)),
        converged=converged,
        equilibrium_relative_residual=equilibrium,
        kkt_relative_residual=diagnostics.get("kkt", 0.0),
        complementarity_relative_residual=diagnostics.get("complementarity", 0.0),
        relative_displacement_increment=diagnostics.get("du", 0.0),
        relative_damage_increment=diagnostics.get("dd", 0.0),
        relative_potential_energy_increment=diagnostics.get("dpi", 0.0),
        reported_irreversibility_violation=diagnostics.get("irreversibility", 0.0),
        reported_range_violation=diagnostics.get("range_violation", 0.0),
        generalized_reaction_magnitude_kN=generalized_reaction,
        peak_rss_bytes=1234,
    )


def test_v11_preflight_lists_missing_controls_without_options() -> None:
    legacy = _legacy_v11_config()
    canonical_sha256 = hashlib.sha256(
        json.dumps(
            legacy,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    preflight = preflight_coupled_coarse_prefix(legacy)

    assert preflight.status == NOT_READY_MISSING_FROZEN_CONTROLS
    assert canonical_sha256 == "d10036cfe1a0fa54600acae5d5f04425014074ec3d8ebace9e8f284251d8a20d"
    assert preflight.protocol_sha256 == canonical_sha256
    assert preflight.options is None
    assert preflight.adaptive_policy is None
    assert "solver.max_displacement_iterations" in preflight.missing_controls
    assert "solver.line_search_steps" in preflight.missing_controls
    assert "solver.active_set_tolerance" in preflight.missing_controls
    assert "solver.tangent_perturbation" in preflight.missing_controls
    assert "solver.adaptive_bisection.factor" in preflight.missing_controls
    assert (
        "solver.adaptive_bisection.max_rejected_attempts_per_required_interval"
        in preflight.missing_controls
    )
    assert "per_tier_qc.max_damage_range_violation" in preflight.missing_controls


def test_current_frozen_v12_config_preflight_is_ready() -> None:
    preflight = preflight_coupled_coarse_prefix(_base_config())

    assert preflight.status == READY_DEVELOPMENT_PREFIX_ONLY
    assert preflight.missing_controls == ()
    assert preflight.options is not None
    assert preflight.options.max_displacement_iterations == 30
    assert preflight.options.line_search_steps == 16
    assert preflight.options.active_set_tolerance == 1.0e-10
    assert preflight.options.tangent_perturbation == 1.0e-7
    assert preflight.options.raise_on_nonconvergence is False
    assert preflight.adaptive_policy is not None
    assert preflight.adaptive_policy.max_rejected_attempts_per_required_interval == 6


def test_current_preflight_builds_every_option_explicitly() -> None:
    preflight = preflight_coupled_coarse_prefix(_base_config())

    assert preflight.status == READY_DEVELOPMENT_PREFIX_ONLY
    assert preflight.options is not None
    assert preflight.options.max_staggered_iterations == 100
    assert preflight.options.max_displacement_iterations == 30
    assert preflight.options.max_active_set_iterations == 100
    assert preflight.options.staggered_tolerance == 1.0e-8
    assert preflight.options.energy_tolerance == 1.0e-8
    assert preflight.options.equilibrium_tolerance == 1.0e-8
    assert preflight.options.kkt_tolerance == 1.0e-8
    assert preflight.options.active_set_tolerance == 1.0e-10
    assert preflight.options.line_search_steps == 16
    assert preflight.options.tangent_perturbation == 1.0e-7
    assert preflight.options.raise_on_nonconvergence is False
    assert {item.name for item in fields(preflight.options)} == {
        "max_staggered_iterations",
        "max_displacement_iterations",
        "max_active_set_iterations",
        "staggered_tolerance",
        "energy_tolerance",
        "equilibrium_tolerance",
        "kkt_tolerance",
        "active_set_tolerance",
        "line_search_steps",
        "tangent_perturbation",
        "raise_on_nonconvergence",
    }


def test_unknown_future_identity_has_no_validator_and_is_not_ready() -> None:
    config = _base_config()
    config["protocol_id"] = "miehe-sent-sens-three-grid-development-v1.3"

    preflight = preflight_coupled_coarse_prefix(config)

    assert preflight.status == NOT_READY_INVALID_FROZEN_CONTROLS
    assert preflight.detail == "current SENT/SENS protocol validation failed"


def test_preflight_refuses_when_adaptive_policy_is_not_frozen() -> None:
    config = _base_config()
    del config["solver"]["adaptive_bisection"]

    preflight = preflight_coupled_coarse_prefix(config)

    assert preflight.status == NOT_READY_INVALID_FROZEN_CONTROLS
    assert preflight.missing_controls == ()
    assert preflight.detail == "current SENT/SENS protocol validation failed"


def test_runtime_control_hash_binds_adaptive_rejection_policy() -> None:
    original = preflight_coupled_coarse_prefix(_base_config())
    assert original.options is not None and original.adaptive_policy is not None
    changed_policy = replace(
        original.adaptive_policy,
        max_rejected_attempts_per_required_interval=5,
    )
    changed_hash = trajectory_module._options_and_policy_sha256(original.options, changed_policy)

    assert original.status == READY_DEVELOPMENT_PREFIX_ONLY
    assert original.options_sha256 != changed_hash


def test_current_v12_control_or_hash_drift_is_not_ready() -> None:
    changed_control = _base_config()
    changed_control["solver"]["max_displacement_iterations"] = 31
    hash_only_drift = _base_config()
    hash_only_drift["evidence_basis"]["public_implementations"]["phasefieldx_url"] = (
        "https://invalid.example/drift"
    )

    for config in (changed_control, hash_only_drift):
        preflight = preflight_coupled_coarse_prefix(config)
        assert preflight.status == NOT_READY_INVALID_FROZEN_CONTROLS
        assert preflight.options is None
        assert preflight.detail == "current SENT/SENS protocol validation failed"
        assert "invalid.example" not in preflight.detail


def test_restart_checkpoint_arrays_are_bytes_backed_and_hash_stays_stable() -> None:
    source_displacement = np.zeros((2, 2))
    source_damage = np.zeros(2)
    source_history = np.zeros(1)
    source_reaction = np.zeros((2, 2))
    checkpoint = RestartCheckpoint(
        U_mm=0.0,
        displacement_yz_mm=source_displacement,
        damage=source_damage,
        history_kN_per_mm2=source_history,
        reaction_yz_kN=source_reaction,
        path_work_kN_mm=0.0,
        total_potential_energy_kN_mm=0.0,
        mesh_sha256="a" * 64,
        protocol_sha256="b" * 64,
        options_sha256="c" * 64,
    )
    stable_hash = checkpoint.checkpoint_sha256
    source_displacement[0, 0] = 99.0
    source_damage[0] = 0.5
    source_history[0] = 7.0
    source_reaction[0, 0] = 3.0

    arrays = (
        checkpoint.displacement_yz_mm,
        checkpoint.damage,
        checkpoint.history_kN_per_mm2,
        checkpoint.reaction_yz_kN,
    )
    assert all(np.count_nonzero(array) == 0 for array in arrays)
    assert all(array.flags.writeable is False for array in arrays)
    for array in arrays:
        with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
            array.setflags(write=True)
    assert checkpoint.checkpoint_sha256 == stable_hash


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("U_mm", math.nan),
        ("displacement_yz_mm", np.asarray([[math.nan, 0.0], [0.0, 0.0]])),
        ("damage", np.asarray([0.0, math.inf])),
        ("history_kN_per_mm2", np.asarray([math.nan])),
        ("reaction_yz_kN", np.asarray([[0.0, 0.0], [0.0, -math.inf]])),
        ("path_work_kN_mm", math.nan),
        ("total_potential_energy_kN_mm", math.inf),
    ],
)
def test_restart_checkpoint_rejects_every_nonfinite_persistent_field(
    field_name: str, invalid_value: object
) -> None:
    values = {
        "U_mm": 0.0,
        "displacement_yz_mm": np.zeros((2, 2)),
        "damage": np.zeros(2),
        "history_kN_per_mm2": np.zeros(1),
        "reaction_yz_kN": np.zeros((2, 2)),
        "path_work_kN_mm": 0.0,
        "total_potential_energy_kN_mm": 0.0,
        "mesh_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "options_sha256": "c" * 64,
    }
    values[field_name] = invalid_value

    with pytest.raises(ValueError, match="finite"):
        RestartCheckpoint(**values)


def test_step_candidate_applied_force_is_bytes_backed_and_input_isolated() -> None:
    checkpoint = _checkpoint(0.0, "b" * 64, "c" * 64)
    source = np.zeros((2, 2))
    candidate = CoupledStepCandidate(
        checkpoint=checkpoint,
        applied_nodal_force_yz_kN=source,
        converged=True,
        equilibrium_relative_residual=0.0,
        kkt_relative_residual=0.0,
        complementarity_relative_residual=0.0,
        relative_displacement_increment=0.0,
        relative_damage_increment=0.0,
        relative_potential_energy_increment=0.0,
        reported_irreversibility_violation=0.0,
        reported_range_violation=0.0,
        generalized_reaction_magnitude_kN=0.0,
        peak_rss_bytes=0,
    )
    source[0, 0] = 99.0

    assert np.count_nonzero(candidate.applied_nodal_force_yz_kN) == 0
    assert candidate.applied_nodal_force_yz_kN.flags.writeable is False
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        candidate.applied_nodal_force_yz_kN.setflags(write=True)


@pytest.mark.parametrize(
    "field_name",
    [
        "equilibrium_relative_residual",
        "kkt_relative_residual",
        "complementarity_relative_residual",
        "relative_displacement_increment",
        "relative_damage_increment",
        "relative_potential_energy_increment",
        "reported_irreversibility_violation",
        "reported_range_violation",
        "generalized_reaction_magnitude_kN",
    ],
)
@pytest.mark.parametrize("invalid_value", [-1.0, math.nan, math.inf])
def test_step_candidate_rejects_invalid_nonnegative_diagnostics(
    field_name: str, invalid_value: float
) -> None:
    valid = _candidate(_checkpoint(0.0, "b" * 64, "c" * 64), 0.0)

    with pytest.raises(ValueError):
        replace(valid, **{field_name: invalid_value})


def test_step_candidate_rejects_nonfinite_applied_force() -> None:
    valid = _candidate(_checkpoint(0.0, "b" * 64, "c" * 64), 0.0)
    applied = np.zeros((2, 2))
    applied[0, 0] = math.nan

    with pytest.raises(ValueError, match="finite"):
        replace(valid, applied_nodal_force_yz_kN=applied)


def test_not_ready_runner_never_calls_solver() -> None:
    calls = 0

    def solver(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    seed = _checkpoint(0.0, "b" * 64, "c" * 64)
    result = run_coupled_coarse_prefix(
        _legacy_v11_config(),
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda _checkpoint, _threshold: True,
    )

    assert result.status == NOT_READY_MISSING_FROZEN_CONTROLS
    assert result.attempt_ledger == ()
    assert calls == 0


def test_rejected_step_rolls_back_and_required_state_is_not_replaced() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    failed_full_target = False
    starts: list[tuple[float, float, float]] = []

    def solver(start, target, _options):
        nonlocal failed_full_target
        starts.append((start.U_mm, start.path_work_kN_mm, target))
        if target == 0.0001 and start.U_mm == 0.0 and not failed_full_target:
            failed_full_target = True
            return _candidate(
                start,
                target,
                equilibrium=1.0,
                work=999.0,
                potential=999.0,
            )
        return _candidate(start, target)

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda _checkpoint, _threshold: True,
        peak_rss_reader=lambda: 100,
    )

    assert result.status == DEVELOPMENT_PREFIX_ACCEPTED
    assert [item.U_mm for item in result.required_checkpoints] == [0.0, 0.0001]
    assert [item.U_mm for item in result.accepted_checkpoints] == [0.0, 0.00005, 0.0001]
    assert [item.accepted_as_required_state for item in result.attempt_ledger] == [
        True,
        False,
        False,
        True,
    ]
    assert result.attempt_ledger[1].accepted is False
    assert result.attempt_ledger[1].code == "QC_EQUILIBRIUM"
    assert result.attempt_ledger[2].parent_attempt_id == result.attempt_ledger[1].attempt_id
    assert result.attempt_ledger[3].parent_attempt_id == result.attempt_ledger[1].attempt_id
    assert starts[-1] == (0.00005, 0.00005, 0.0001)
    assert all(start_work != 999.0 for _, start_work, _ in starts)
    assert result.required_checkpoints[-1].path_work_kN_mm == 0.0001
    assert all(item.peak_rss_bytes >= 1234 for item in result.attempt_ledger)


def test_sixth_retryable_rejection_stops_without_a_seventh_attempt() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    calls: list[float] = []

    def solver(start, target, _options):
        calls.append(target)
        return _candidate(start, target, converged=target == 0.0)

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda _checkpoint, _threshold: True,
    )

    rejected = [entry for entry in result.attempt_ledger if not entry.accepted]
    assert result.status == "STOP_NUMERICAL"
    assert len(rejected) == 6
    assert len(calls) == 7  # one accepted U=0 plus exactly six rejected attempts
    assert rejected[-1].rejected_attempt_count_for_required_state == 6
    assert [entry.required_state_index for entry in rejected] == [1] * 6
    assert all(entry.code == "QC_NONCONVERGED" for entry in rejected)


def test_global_force_failure_is_stop_invalid_and_never_retried() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    calls: list[float] = []

    def solver(start, target, _options):
        calls.append(target)
        if target == 0.0:
            return _candidate(start, target)
        return _candidate(
            start,
            target,
            reaction=np.asarray([[1.0, 0.0], [0.0, 0.0]]),
        )

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda _checkpoint, _threshold: True,
    )

    assert result.status == STOP_INVALID
    assert calls == [0.0, 0.0001]
    assert result.attempt_ledger[-1].code == "QC_GLOBAL_FORCE"
    assert result.attempt_ledger[-1].rejected_attempt_count_for_required_state == 1


def test_solver_exception_is_stop_invalid_and_never_retried() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    calls: list[float] = []

    def solver(start, target, _options):
        calls.append(target)
        if target == 0.0:
            return _candidate(start, target)
        raise RuntimeError("synthetic solver failure")

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda _checkpoint, _threshold: True,
    )

    assert result.status == STOP_INVALID
    assert calls == [0.0, 0.0001]
    assert result.attempt_ledger[-1].code == "SOLVER_EXCEPTION"
    assert result.attempt_ledger[-1].exception_type == "RuntimeError"
    assert result.attempt_ledger[-1].exception_message == "synthetic solver failure"


def test_nonfinite_candidate_construction_is_solver_exception_and_never_retried() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    calls: list[float] = []

    def solver(start, target, _options):
        calls.append(target)
        return _candidate(start, target, equilibrium=0.0 if target == 0.0 else math.nan)

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda _checkpoint, _threshold: True,
    )

    assert result.status == STOP_INVALID
    assert calls == [0.0, 0.0001]
    assert result.attempt_ledger[-1].code == "SOLVER_EXCEPTION"
    assert result.attempt_ledger[-1].exception_type == "ValueError"


@pytest.mark.parametrize("malformed", [object(), {"checkpoint": "not-a-candidate"}])
def test_malformed_solver_return_is_ledgered_stop_invalid(malformed: object) -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=1,
        single_step_solver=lambda _start, _target, _options: malformed,
        corridor_callback=lambda _checkpoint, _threshold: True,
    )

    assert result.status == STOP_INVALID
    assert result.accepted_checkpoints == ()
    assert len(result.attempt_ledger) == 1
    entry = result.attempt_ledger[0]
    assert entry.code == STOP_INVALID
    assert entry.accepted is False
    assert entry.result_checkpoint_sha256 is None
    assert entry.exception_type == "TypeError"
    assert entry.rejected_attempt_count_for_required_state == 1


def test_damage_corridor_escape_stops_invalid_without_bisection() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    calls: list[float] = []

    def solver(start, target, _options):
        calls.append(target)
        return _candidate(start, target)

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda checkpoint, _threshold: checkpoint.U_mm == 0.0,
    )

    assert result.status == STOP_INVALID
    assert calls == [0.0, 0.0001]
    assert result.attempt_ledger[-1].code == STOP_INVALID
    assert result.attempt_ledger[-1].accepted is False
    assert [item.U_mm for item in result.required_checkpoints] == [0.0]


def test_corridor_stop_invalid_outranks_simultaneous_retryable_failure() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None
    seed = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    calls: list[float] = []

    def solver(start, target, _options):
        calls.append(target)
        return _candidate(
            start,
            target,
            converged=target == 0.0,
        )

    result = run_coupled_coarse_prefix(
        config,
        benchmark_id="sent",
        initial_checkpoint=seed,
        nodes_yz_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        required_prefix_count=2,
        single_step_solver=solver,
        corridor_callback=lambda checkpoint, _threshold: checkpoint.U_mm == 0.0,
    )

    assert result.status == STOP_INVALID
    assert calls == [0.0, 0.0001]
    assert result.attempt_ledger[-1].qc.failed_codes == (
        "QC_NONCONVERGED",
        STOP_INVALID,
    )
    assert result.attempt_ledger[-1].code == STOP_INVALID
    assert result.attempt_ledger[-1].retry_depth == 0
    assert len(result.attempt_ledger) == 2


def test_qc_formulas_cover_all_required_gates() -> None:
    config = _base_config()
    preflight = preflight_coupled_coarse_prefix(config)
    assert preflight.options_sha256 is not None and preflight.qc is not None
    previous = _checkpoint(0.0, preflight.protocol_sha256, preflight.options_sha256)
    bad = _candidate(
        previous,
        0.0001,
        equilibrium=1.0,
        converged=False,
        work=1.0,
        potential=0.0,
        damage=np.asarray([-0.1, 0.0]),
        reaction=np.asarray([[0.0, 0.0], [0.0, 1.0]]),
        generalized_reaction=1.0,
        kkt=1.0,
        complementarity=1.0,
        du=1.0,
        dd=1.0,
        dpi=1.0,
        irreversibility=0.2,
        range_violation=0.1,
    )

    qc = evaluate_step_qc(
        previous,
        bad,
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
        preflight.qc,
        damage_component_threshold=0.5,
        corridor_callback=lambda _checkpoint, _threshold: False,
    )

    assert {
        "QC_NONCONVERGED",
        "QC_EQUILIBRIUM",
        "QC_KKT",
        "QC_DU",
        "QC_DD",
        "QC_DPI",
        "QC_IRREVERSIBILITY",
        "QC_RANGE",
        "QC_GLOBAL_FORCE",
        "QC_GLOBAL_MOMENT",
        "QC_PATH_ENERGY",
        STOP_INVALID,
    } <= set(qc.failed_codes)
    assert qc.irreversibility_violation == pytest.approx(0.2)
    assert qc.range_violation == pytest.approx(0.1)
    assert qc.global_force_relative_imbalance > 0.0
    assert qc.global_moment_relative_imbalance > 0.0
    assert qc.path_energy_relative_imbalance == pytest.approx(1.0)
