from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tunnelgeopt.fracture_benchmark_trajectory import (
    DEVELOPMENT_PREFIX_ACCEPTED,
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


def _extended_config() -> dict:
    config = copy.deepcopy(_base_config())
    config["protocol_id"] = "miehe-sent-sens-three-grid-development-v1.2-test"
    config["solver"].update(
        {
            "max_displacement_iterations": 37,
            "line_search_steps": 19,
            "active_set_tolerance": 2.0e-11,
            "tangent_perturbation": 3.0e-8,
            "raise_on_nonconvergence": False,
            "adaptive_bisection": {
                "factor": 0.5,
                "max_retry_depth": 4,
                "minimum_increment_mm": 1.0e-8,
                "retryable_codes": [
                    "QC_NONCONVERGED",
                    "QC_EQUILIBRIUM",
                    "SOLVER_EXCEPTION",
                ],
                "retry_exhausted_action": "STOP_NUMERICAL",
            },
        }
    )
    config["per_tier_qc"].update(
        {
            "max_damage_range_violation": 1.0e-12,
            "force_balance_normalization_floor_kN": 1.0e-15,
            "moment_balance_normalization_floor_kN_mm": 1.0e-15,
            "path_energy_normalization_floor_kN_mm": 1.0e-18,
            "global_moment_origin_yz_mm": [0.0, 0.0],
        }
    )
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
    preflight = preflight_coupled_coarse_prefix(_base_config())

    assert preflight.status == NOT_READY_MISSING_FROZEN_CONTROLS
    assert preflight.options is None
    assert preflight.adaptive_policy is None
    assert "solver.max_displacement_iterations" in preflight.missing_controls
    assert "solver.line_search_steps" in preflight.missing_controls
    assert "solver.active_set_tolerance" in preflight.missing_controls
    assert "solver.tangent_perturbation" in preflight.missing_controls
    assert "solver.adaptive_bisection.factor" in preflight.missing_controls
    assert "per_tier_qc.max_damage_range_violation" in preflight.missing_controls


def test_extended_preflight_builds_every_option_explicitly() -> None:
    preflight = preflight_coupled_coarse_prefix(_extended_config())

    assert preflight.status == READY_DEVELOPMENT_PREFIX_ONLY
    assert preflight.options is not None
    assert preflight.options.max_staggered_iterations == 100
    assert preflight.options.max_displacement_iterations == 37
    assert preflight.options.max_active_set_iterations == 100
    assert preflight.options.staggered_tolerance == 1.0e-8
    assert preflight.options.energy_tolerance == 1.0e-8
    assert preflight.options.equilibrium_tolerance == 1.0e-8
    assert preflight.options.kkt_tolerance == 1.0e-8
    assert preflight.options.active_set_tolerance == 2.0e-11
    assert preflight.options.line_search_steps == 19
    assert preflight.options.tangent_perturbation == 3.0e-8
    assert preflight.options.raise_on_nonconvergence is False


def test_preflight_refuses_when_adaptive_policy_is_not_frozen() -> None:
    config = _extended_config()
    del config["solver"]["adaptive_bisection"]

    preflight = preflight_coupled_coarse_prefix(config)

    assert preflight.status == NOT_READY_MISSING_FROZEN_CONTROLS
    assert "solver.adaptive_bisection.factor" in preflight.missing_controls


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


def test_not_ready_runner_never_calls_solver() -> None:
    calls = 0

    def solver(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("must not run")

    seed = _checkpoint(0.0, "b" * 64, "c" * 64)
    result = run_coupled_coarse_prefix(
        _base_config(),
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
    config = _extended_config()
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


def test_damage_corridor_escape_stops_invalid_without_bisection() -> None:
    config = _extended_config()
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


def test_qc_formulas_cover_all_required_gates() -> None:
    config = _extended_config()
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
        generalized_reaction=-1.0,
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
        "QC_REACTION",
        STOP_INVALID,
    } <= set(qc.failed_codes)
    assert qc.irreversibility_violation == pytest.approx(0.2)
    assert qc.range_violation == pytest.approx(0.1)
    assert qc.global_force_relative_imbalance > 0.0
    assert qc.global_moment_relative_imbalance > 0.0
    assert qc.path_energy_relative_imbalance == pytest.approx(1.0)
