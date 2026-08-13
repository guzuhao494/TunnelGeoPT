from __future__ import annotations

import numpy as np
import pytest

from tunnelgeopt.fracture_work import (
    AcceptedStepWorkIncrement,
    BoundaryEquilibriumState,
    CumulativeWorkHistory,
    accepted_step_work_increment,
    cumulative_accepted_work,
    energy_increment_diagnostic,
    neumann_load_functional,
)


def _state(
    displacement: np.ndarray,
    internal_force: np.ndarray,
    wall_force: np.ndarray,
    dirichlet_dofs: np.ndarray,
    *,
    accepted: bool = True,
) -> BoundaryEquilibriumState:
    flattened = np.asarray(displacement, dtype=np.float64).reshape(-1)
    dofs = np.asarray(dirichlet_dofs, dtype=np.int64)
    return BoundaryEquilibriumState(
        displacement=displacement,
        internal_force=internal_force,
        wall_nodal_force=wall_force,
        dirichlet_dofs=dofs,
        farfield_prescribed_displacement=flattened[dofs],
        accepted=accepted,
    )


def test_one_dof_linear_spring_farfield_work_is_exact() -> None:
    stiffness = 7.5
    displacement = 0.4
    initial = _state(
        np.asarray([[0.0, 0.0]]),
        np.asarray([0.0, 0.0]),
        np.zeros(2),
        np.asarray([0]),
    )
    final = _state(
        np.asarray([[displacement, 0.0]]),
        np.asarray([stiffness * displacement, 0.0]),
        np.zeros(2),
        np.asarray([0]),
    )

    work = accepted_step_work_increment(initial, final)
    exact_energy = 0.5 * stiffness * displacement**2
    assert work.wall_work == 0.0
    assert work.farfield_work == pytest.approx(exact_energy, rel=0.0, abs=2.0e-16)
    diagnostic = energy_increment_diagnostic(
        0.0, exact_energy, work.external_work, normalization_floor=1.0e-12
    )
    assert diagnostic.absolute_imbalance == pytest.approx(0.0, abs=2.0e-16)
    assert diagnostic.relative_imbalance == pytest.approx(0.0, abs=3.0e-16)


def test_fixed_farfield_has_zero_work_even_when_reaction_changes() -> None:
    initial = _state(
        np.asarray([[0.2, 0.0]]),
        np.asarray([3.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0]),
    )
    final = _state(
        np.asarray([[0.2, 0.1]]),
        np.asarray([9.0, 2.0]),
        np.asarray([0.0, 2.0]),
        np.asarray([0]),
    )

    work = accepted_step_work_increment(initial, final)
    assert work.wall_work == pytest.approx(0.15)
    assert work.farfield_work == 0.0


def test_force_reaction_and_free_residual_use_full_balance_vector() -> None:
    state = _state(
        np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        np.asarray([10.0, 2.0, -3.0, 7.0]),
        np.asarray([1.0, 2.5, -4.0, 1.0]),
        np.asarray([0, 3]),
    )

    assert np.array_equal(state.full_equilibrium_residual, np.asarray([9.0, -0.5, 1.0, 6.0]))
    assert np.array_equal(state.reaction_on_dirichlet_dofs, np.asarray([9.0, 6.0]))
    assert np.array_equal(state.reaction_full, np.asarray([9.0, 0.0, 0.0, 6.0]))
    assert np.array_equal(state.free_dofs, np.asarray([1, 2]))
    assert np.array_equal(state.free_residual, np.asarray([-0.5, 1.0]))
    assert neumann_load_functional(state) == pytest.approx(-0.2)
    with pytest.raises(ValueError, match="read-only"):
        state.reaction_full[0] = 0.0


def test_state_copies_inputs_and_exposes_read_only_arrays() -> None:
    displacement = np.asarray([[0.1, 0.0]])
    internal_force = np.asarray([2.0, 0.0])
    state = _state(displacement, internal_force, np.zeros(2), np.asarray([0]))
    displacement[0, 0] = 99.0
    internal_force[0] = 99.0

    assert state.displacement[0, 0] == pytest.approx(0.1)
    assert state.internal_force[0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="read-only"):
        state.displacement[0, 0] = 1.0
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        state.displacement.setflags(write=True)


def test_omitting_farfield_work_is_detected_by_energy_negative_control() -> None:
    stiffness = 4.0
    end_displacement = 0.5
    initial = _state(np.zeros((1, 2)), np.zeros(2), np.zeros(2), np.asarray([0]))
    final = _state(
        np.asarray([[end_displacement, 0.0]]),
        np.asarray([stiffness * end_displacement, 0.0]),
        np.zeros(2),
        np.asarray([0]),
    )
    stored_energy = 0.5 * stiffness * end_displacement**2
    work = accepted_step_work_increment(initial, final)

    correct = energy_increment_diagnostic(
        0.0, stored_energy, work.external_work, normalization_floor=1.0e-12
    )
    missing_farfield = energy_increment_diagnostic(
        0.0, stored_energy, work.wall_work, normalization_floor=1.0e-12
    )
    assert correct.relative_imbalance == pytest.approx(0.0, abs=1.0e-15)
    assert missing_farfield.relative_imbalance == pytest.approx(1.0)
    assert missing_farfield.signed_imbalance == pytest.approx(stored_energy)


def _nonlinear_path_work(step_count: int, path: str) -> float:
    states: list[BoundaryEquilibriumState] = []
    for load_parameter in np.linspace(0.0, 1.0, step_count + 1):
        if path == "P2":
            displacement = np.asarray([[load_parameter, 0.0]])
            reaction = np.asarray([load_parameter**2, 0.0])
            dofs = np.asarray([0])
        else:
            displacement = np.asarray([[load_parameter, load_parameter**2]])
            reaction = np.asarray([load_parameter**2, 0.5 + load_parameter**2], dtype=np.float64)
            dofs = np.asarray([0, 1])
        states.append(_state(displacement, reaction, np.zeros(2), dofs))
    return float(cumulative_accepted_work(states).cumulative_external_work[-1])


@pytest.mark.parametrize(("path", "exact"), [("P2", 1.0 / 3.0), ("P3", 4.0 / 3.0)])
def test_p2_p3_nonlinear_path_toy_shows_quadrature_convergence(path: str, exact: float) -> None:
    # These are quadrature toys, not claims that a P2/P3 fracture solve is exact.
    errors = [abs(_nonlinear_path_work(step_count, path) - exact) for step_count in (8, 16, 32)]
    assert errors[2] < errors[1] < errors[0]
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.08)
    assert errors[1] / errors[2] == pytest.approx(4.0, rel=0.08)
    assert errors[2] > np.finfo(np.float64).eps


def test_rejected_attempt_is_omitted_without_mutating_accepted_path() -> None:
    accepted_initial = _state(np.zeros((1, 2)), np.zeros(2), np.zeros(2), np.asarray([0]))
    rejected_trial = _state(
        np.asarray([[10.0, 0.0]]),
        np.asarray([100.0, 0.0]),
        np.zeros(2),
        np.asarray([0]),
        accepted=False,
    )
    accepted_final = _state(
        np.asarray([[1.0, 0.0]]),
        np.asarray([2.0, 0.0]),
        np.zeros(2),
        np.asarray([0]),
    )

    history = cumulative_accepted_work([accepted_initial, rejected_trial, accepted_final])
    direct = accepted_step_work_increment(accepted_initial, accepted_final)
    assert np.array_equal(history.accepted_input_indices, np.asarray([0, 2]))
    assert np.array_equal(history.external_increment, np.asarray([0.0, direct.external_work]))
    assert history.cumulative_external_work[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="two accepted states"):
        accepted_step_work_increment(accepted_initial, rejected_trial)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"displacement": np.asarray([[np.nan, 0.0]])}, "finite"),
        ({"internal_force": np.zeros(3)}, "shape"),
        ({"wall_nodal_force": np.asarray([0.0, np.inf])}, "finite"),
        (
            {
                "dirichlet_dofs": np.asarray([], dtype=np.int64),
                "farfield_prescribed_displacement": np.asarray([]),
            },
            "non-empty",
        ),
        ({"dirichlet_dofs": np.asarray([0, 0])}, "unique"),
        ({"dirichlet_dofs": np.asarray([2])}, "out-of-range"),
        ({"farfield_prescribed_displacement": np.asarray([0.25])}, "does not match"),
        ({"accepted": 1}, "boolean"),
    ],
)
def test_state_nonfinite_shape_index_and_alignment_errors_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "displacement": np.zeros((1, 2)),
        "internal_force": np.zeros(2),
        "wall_nodal_force": np.zeros(2),
        "dirichlet_dofs": np.asarray([0]),
        "farfield_prescribed_displacement": np.asarray([0.0]),
        "accepted": True,
    }
    arguments.update(overrides)
    exception = TypeError if message == "boolean" else ValueError
    with pytest.raises(exception, match=message):
        BoundaryEquilibriumState(**arguments)  # type: ignore[arg-type]


def test_cross_step_dof_and_mesh_mismatches_fail_closed() -> None:
    initial = _state(np.zeros((1, 2)), np.zeros(2), np.zeros(2), np.asarray([0]))
    changed_dofs = _state(np.zeros((1, 2)), np.zeros(2), np.zeros(2), np.asarray([1]))
    changed_mesh = _state(np.zeros((2, 2)), np.zeros(4), np.zeros(4), np.asarray([0]))

    with pytest.raises(ValueError, match="identical aligned"):
        accepted_step_work_increment(initial, changed_dofs)
    with pytest.raises(ValueError, match="matching displacement shapes"):
        accepted_step_work_increment(initial, changed_mesh)


def test_energy_and_work_diagnostics_reject_nonfinite_or_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        energy_increment_diagnostic(0.0, 1.0, np.nan, normalization_floor=1.0e-12)
    with pytest.raises(ValueError, match="nonnegative"):
        energy_increment_diagnostic(-1.0, 1.0, 2.0, normalization_floor=1.0e-12)
    with pytest.raises(ValueError, match="positive"):
        energy_increment_diagnostic(0.0, 1.0, 1.0, normalization_floor=0.0)
    with pytest.raises(ValueError, match="finite"):
        energy_increment_diagnostic(0.0, 1.0, 1.0, normalization_floor=np.inf)
    with pytest.raises(ValueError, match="finite"):
        AcceptedStepWorkIncrement(np.inf, 0.0)
    with pytest.raises(ValueError, match="external_increment"):
        CumulativeWorkHistory(
            accepted_input_indices=np.asarray([0, 1]),
            wall_increment=np.asarray([0.0, 1.0]),
            farfield_increment=np.asarray([0.0, 2.0]),
            external_increment=np.asarray([0.0, 99.0]),
            cumulative_wall_work=np.asarray([0.0, 1.0]),
            cumulative_farfield_work=np.asarray([0.0, 2.0]),
            cumulative_external_work=np.asarray([0.0, 99.0]),
        )


def test_near_zero_energy_imbalance_uses_explicit_dimensional_floor() -> None:
    diagnostic = energy_increment_diagnostic(
        0.0,
        2.0e-15,
        1.0e-15,
        normalization_floor=1.0e-12,
    )
    assert diagnostic.absolute_imbalance == pytest.approx(1.0e-15)
    assert diagnostic.relative_imbalance == pytest.approx(1.0e-3)
    assert diagnostic.normalization_floor == pytest.approx(1.0e-12)
