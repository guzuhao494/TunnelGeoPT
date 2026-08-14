# SENT/SENS bounded coupled-prefix contract

`fracture_benchmark_trajectory.py` freezes the minimum restart, retry, QC, and
ledger semantics for a future coupled coarse SENT/SENS prefix. It is
development-only infrastructure. It does not contain a finite-element solver,
does not run either coupon, does not authorize a complete coarse trajectory,
and does not provide paper evidence.

## Current decision

The checked-in v1.2 protocol freezes the complete controls below and returns
`READY_DEVELOPMENT_PREFIX_ONLY`. This only enables a bounded, injected coarse
development prefix; it does not authorize a real FEM campaign or formal run.
Every non-legacy mapping must pass the strict current SENT/SENS validator,
including its exact protocol ID, semantic contract, and canonical SHA-256;
unknown future IDs and any field/hash drift return
`NOT_READY_INVALID_FROZEN_CONTROLS` with a path-free generic detail.
The exact pinned v1.1 shape, reconstructed in regression tests by removing the
v1.2 additions, still returns `NOT_READY_MISSING_FROZEN_CONTROLS`. The coupled
runner will not construct `FractureSolverOptions` from library defaults and
will not choose an adaptive policy locally. A coupled-ready protocol must
freeze all of the following paths:

- `solver.max_displacement_iterations`
- `solver.line_search_steps`
- `solver.active_set_tolerance`
- `solver.tangent_perturbation`
- `solver.raise_on_nonconvergence`
- `solver.adaptive_bisection.factor`
- `solver.adaptive_bisection.max_retry_depth`
- `solver.adaptive_bisection.minimum_increment_mm`
- `solver.adaptive_bisection.retryable_codes`
- `solver.adaptive_bisection.retry_exhausted_action`
- `solver.adaptive_bisection.max_rejected_attempts_per_required_interval`
- `per_tier_qc.max_damage_range_violation`
- `per_tier_qc.force_balance_normalization_floor_kN`
- `per_tier_qc.moment_balance_normalization_floor_kN_mm`
- `per_tier_qc.path_energy_normalization_floor_kN_mm`
- `per_tier_qc.global_moment_origin_yz_mm`

The extension must have a new protocol identity. Adding fields while retaining
the immutable v1.1 ID is rejected as
`NOT_READY_PROTOCOL_EXTENSION_REQUIRED`. No numerical value for any missing
control is proposed or implied by this module. The legacy guard pins both the
literal v1.1 protocol ID and its canonical SHA-256 locally; it intentionally
does not import a validator's moving "current protocol" identity, so a future
v1.2 cannot drift into the v1.1 rejection branch.

When a new protocol supplies those fields, every `FractureSolverOptions` field
is passed explicitly. The existing displacement and damage tolerances define
the single staggered tolerance conservatively through their minimum; the two
original thresholds are still checked independently in QC.

## Restart and rollback

Every accepted state is an immutable `RestartCheckpoint` containing:

- scalar displacement `U`;
- complete displacement, damage, history, and reaction arrays;
- cumulative path work and total potential energy;
- mesh, protocol, and exact solver-options-plus-adaptive-policy SHA-256
  identities; and
- a content SHA-256 over the metadata and all array bytes.

Checkpoint construction rejects non-finite `U`, displacement, damage,
history, reaction, path-work, or potential-energy values before a solver can
receive a restart seed. Candidate construction likewise requires finite
applied nodal force and finite, nonnegative residual, increment, violation,
and reaction-magnitude diagnostics. A candidate-construction failure inside
the injected solver is ledgered as nonretryable `SOLVER_EXCEPTION` and routes
to `STOP_INVALID`.

The injectable one-step solver receives only the last accepted checkpoint,
the attempted scalar target, and the explicitly constructed options. A
rejected candidate is written to the attempt ledger but is never appended to
accepted state and is never used as the parent of another solve. This is the
rollback guarantee.

Adaptive bisection is allowed only when its failure code is listed by the new
protocol and its frozen depth and minimum-increment gates permit both child
intervals. The factor is exactly `0.5`. Rejections are counted independently
for each `required_state_index`; v1.2 stops with `STOP_NUMERICAL` immediately
on the sixth rejection, before a seventh attempt can be issued. A midpoint is
an adaptive state only; the original required target remains pending and must
later be accepted exactly once. `required_prefix_count` truncates the exact
coarse grid by count; callers cannot provide replacement displacement values.
The restart `options_sha256` binds both the explicit
`FractureSolverOptions` and the complete adaptive policy, including this
rejection budget.

Each ledger row records the required target and index, attempted `U` and
`dU`, retry depth and parent, start/result checkpoint hashes, wall time, peak
RSS, exception type/message, all QC results, terminal/acceptance code, and
whether the accepted state is a required output. It also records the cumulative
rejected-attempt count for that required state.

## Per-attempt QC formulas

All fields and diagnostics must be finite because the frozen nonfinite fraction
is zero. The independent scalar gates are:

- solver convergence;
- equilibrium relative residual;
- maximum of KKT and complementarity relative residuals;
- final staggered relative `du`, `dd`, and `dPi`;
- irreversibility violation, computed as the maximum of the solver report and
  `max(d_previous-d_candidate,0)`;
- range violation, computed as the maximum of the solver report,
  `max(-d,0)`, and `max(d-1,0)`; and
- a finite, nonnegative reaction magnitude under the protocol's positive
  opening/shear curve convention.

Global force uses complete support reaction `r` and applied nodal force `f`:

`||sum_i(r_i+f_i)|| / max(sum_i||r_i||+sum_i||f_i||, force_floor)`.

With the protocol-frozen moment origin `(y0,z0)`, moment uses
`m_i=(y_i-y0)F_zi-(z_i-z0)F_yi` and is checked as
`|sum_i(m_i^r+m_i^f)| / max(sum_i|m_i^r|+sum_i|m_i^f|, moment_floor)`.

Path energy uses consecutive accepted checkpoints:

`|(Pi_n-Pi_previous)-(W_n-W_previous)| /
max(|Pi_n-Pi_previous|,|W_n-W_previous|,energy_floor)`.

Finally, the caller must inject a damage-corridor callback evaluated with the
protocol's `d>=0.5` component threshold. Callback failure, malformed identity,
or corridor escape produces `STOP_INVALID` and is never bisected. This has
explicit highest precedence even when the same candidate also reports a
retryable convergence or residual failure. The only retryable codes are the
frozen seven: `QC_NONCONVERGED`, `QC_EQUILIBRIUM`, `QC_KKT`, `QC_DU`,
`QC_DD`, `QC_DPI`, and `QC_PATH_ENERGY`; their depth, increment, or rejection
budget exhaustion produces `STOP_NUMERICAL`. `QC_NONFINITE`,
`QC_IRREVERSIBILITY`, `QC_RANGE`, `QC_GLOBAL_FORCE`, `QC_GLOBAL_MOMENT`,
`QC_REACTION`, `SOLVER_EXCEPTION`, unknown codes, and `STOP_INVALID` all route
immediately to `STOP_INVALID` and are never bisected.

The mock tests exercise preflight refusal, explicit option construction,
checkpoint immutability, rejected-state rollback, midpoint insertion without
required-state replacement, complete ledger linkage, every QC family, and
fail-closed `STOP_INVALID` on corridor escape. They do not execute real FEM.
