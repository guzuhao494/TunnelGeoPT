# Audited boundary reaction and path-work primitives

## Status and scope

`src/tunnelgeopt/fracture_work.py` is an isolated bookkeeping layer for the
planned quasi-static AT2 fracture pipeline. It does **not** yet connect to
`fracture.py`, generate Phase-1 labels, or establish agreement with MOOSE.
Its purpose is narrower: make the force signs, constrained reactions, accepted
path work, and energy-balance diagnostic explicit and independently testable.

## Boundary state and force sign

A `BoundaryEquilibriumState` stores immutable copies of:

- displacement `u` with shape `[N, 2]`;
- assembled internal force `f_int` with shape `[2N]`;
- prescribed wall force on the rock `f_wall` with shape `[2N]`;
- unique, strictly increasing constrained DOF indices `D`;
- prescribed far-field displacement values aligned one-to-one with `D`; and
- whether the attempted nonlinear/load step was accepted.

Node-major DOF ordering is fixed as
`[u_0,0, u_0,1, u_1,0, u_1,1, ...]`. The constructor rejects non-finite data,
shape mismatches, an empty far-field constraint set, duplicate/out-of-range
DOFs, and prescribed values that do not exactly match `u[D]`.

All external-force signs mean **force applied to the rock**. Static balance is

```text
f_int = f_wall + r_D,
r_D = (f_int - f_wall)_D.
```

Thus `reaction_on_dirichlet_dofs` is the far-field support force on the rock,
not the opposite rock-on-support force sometimes reported by FE postprocessors.
`reaction_full` embeds that vector into `[2N]` and is zero away from `D`.
`free_residual` is `(f_int - f_wall)` on unconstrained DOFs and should approach
zero for an equilibrated state.

## Accepted-step external work

For two adjacent accepted states `0` and `1`, the work increments are

```text
Delta W_wall = 0.5 (f_wall,0 + f_wall,1) . (u_1 - u_0)
Delta W_far  = 0.5 (r_D,0 + r_D,1) . (u_D,1 - u_D,0)
Delta W_ext  = Delta W_wall + Delta W_far.
```

The constrained DOF set and its ordering must be identical across the step.
Fixed far-field DOFs contribute exactly zero work even if their reactions
change. The trapezoidal rule is exact for the tested linear one-DOF spring; it
is only a convergent quadrature for a general nonlinear P2/P3-like path. Tests
therefore check mesh-in-load-step convergence for nonlinear toys rather than
claiming machine-exact nonlinear work.

`cumulative_accepted_work` is pure and rollback-safe. It first removes rejected
attempts, then recomputes increments between consecutive accepted states. A
rejected trial force is never accumulated, and no mutable accumulator needs to
be undone.

`neumann_load_functional(state) = f_wall . u` remains a separate instantaneous
endpoint functional. It is not a substitute for cumulative wall work when the
wall load varies.

## Energy diagnostic

For caller-supplied total quasi-static energy `E` (recoverable elastic plus
consistently assembled AT2 fracture-surface energy), the module reports

```text
Delta E       = E_1 - E_0
imbalance     = Delta E - Delta W_ext
relative      = |imbalance| / max(|Delta E|, |Delta W_ext|, E_floor).
```

`E_floor` must be supplied explicitly as a finite positive quantity. The
planned integration uses `1e-12 * UCS * R^2`; the pure module does not guess a
physical scale or silently substitute a dimensionless epsilon. The all-zero
case still has relative imbalance zero because its numerator is zero. Omitting
moving far-field work from a prescribed-displacement spring produces a relative
imbalance of one in the negative-control test. This catches a bookkeeping
omission; it does not by itself validate element energies, force assembly,
nonlinear convergence, or a fracture model.

## Scientific boundaries before integration

- The primitives require the solver to supply consistent internal and wall
  nodal forces at the same converged `(u, d)` state.
- `free_residual` and `r_D` are nodal algebraic equilibrium diagnostics only.
  Global resultant force and moment checks require mesh coordinates, boundary
  geometry, and an explicit integration/origin convention; those checks remain
  for the later solver integration and are not inferred in this pure module.
- Only accepted states may enter path work; retry/halving logic remains solver
  responsibility.
- Units are generalized force times displacement, per unit out-of-plane
  thickness for the present 2D plane-strain formulation.
- No kinetic energy, damping, rate dependence, contact work, or dynamic
  rockburst energy is represented.
- Passing these unit tests is not SENT/SENS validation and is not a local-vs-
  MOOSE same-problem cross-check.
- Integration into the trajectory schema must preserve wall and far-field
  contributions separately instead of storing only an opaque scalar total.
