# Phase-1 scheduled trajectory adapter

`tunnelgeopt.fracture_trajectory` is the development-only bridge from the
scheduled local AT2 solver to the strict C-fracture schema v3. It does not
authorize the 36-case pilot and cannot turn the local prototype into a
validated label generator.

## API and evidence boundary

The primary API is:

```python
run = run_phase1_development_trajectory(
    mesh,
    material,
    schedule,
    config,
    identity,
    equilibrium_force_normalization_floor=force_floor,
    energy_balance_normalization_floor=energy_floor,
)

paths, loaded = save_and_verify_phase1_development_run(output_dir, run)
```

In this adapter revision, `Phase1DevelopmentRun.formal_labels_allowed` remains
fixed to `False`. The scheduled solver now publishes the symmetric relative
change of total potential energy between its final two complete staggered
iterates, so `solver_energy_increment_residual_available` is fixed to `True`.
The adapter passes the frozen `1e-8` tolerance into the solver, rechecks every
accepted step, and stores `staggered_potential_energy_change` as a hash-bound
schema-v3 array and accepted-ledger field. Rejected attempts carry `null` for
that accepted-state quantity. Closing this numerical gate does not authorize
formal labels.

The adapter stores every accepted internal state in `FractureTrajectory`. The
41 frozen output coordinates are not reconstructed by interpolation. They are
exact accepted states selected by the immutable `required_output_indices` map,
also recorded in metadata. Therefore internal half steps contribute to work
and history, while a consumer can extract the exact 41 required outputs.

## Exact schedule audit

For every attempted `s`, the adapter explicitly calls
`Phase1LoadSchedule.state_at(s)` and requires exact equality of:

- path ID, `s`, UCS scale, normalized principal controls, and angle;
- the tension-positive `2 x 2` far-field stress tensor;
- wall-facet IDs in exact row order, not merely as a set;
- wall-zone IDs, releases, and facet-zone weights; and
- per-facet wall release in the same exact row order.

Before schema construction it additionally recomputes the affine far-field
displacement from the accepted stress, requires both node-major displacement
DOFs on every far-field node, and checks the solver's correction field.
Independent P1 reconstruction checks strain from `u`, constitutive stress from
strain and P1 damage, internal nodal force from element stress, the Neumann
functional, split energies, fracture energy, and total potential.

## Adaptive retry and rollback

For a required interval `(s_old, s_required]`, the first candidate is
`s_required`. A failed candidate is recorded and the increment is multiplied
by the frozen factor `0.5`. At most six rejected candidates are allowed in the
whole required-output interval, and no attempted increment may fall below the
frozen minimum. After an accepted internal half step, the controller again
targets the original required output. Accepted coordinates are strictly
increasing; a required output is indexed only after it has been reached
exactly.

The numerical kernel currently has no restart-state input. To make rollback
unambiguous, each attempt starts a fresh solver invocation on the complete
prefix

```text
[all previously accepted s] + [candidate s]
```

and the adapter compares every recomputed accepted-prefix `u`, `d`, `H`, force,
energy, residual, iteration count, and load-state field bitwise. A rejected
candidate never enters accepted arrays or work. This is deterministic and
stronger than retaining a possibly mutated in-memory iterate, but it has
quadratic prefix-recomputation cost. A future explicit restart-state API can
replace this strategy only after rollback-equivalence tests exist.

The adapter forces `raise_on_nonconvergence=False`, so a returned
nonconverged candidate becomes a rejected ledger entry. If the solver raises
before returning a step, no reaction, residual, or iteration diagnostic exists
to record without fabrication. The runner therefore fails closed with the
accepted-prefix ledger; it does not manufacture a schema-shaped rejected
entry.

The initial `s=0` state is solved separately and defines the zero path-work
origin. It cannot be recovered by halving an earlier increment and must pass
all candidate gates.

## Work, reaction, and global balance

Every accepted internal state constructs `BoundaryEquilibriumState` directly
from the solver's `u`, internal nodal force, wall nodal force, constrained DOFs,
and prescribed far-field displacement. Far-field reaction on the rock is
derived, never supplied:

```text
r_D = (f_int - f_wall)_D
```

Adjacent accepted internal states contribute trapezoidal wall and far-field
work. Rejected trials do not alter cumulative work. The schema then recomputes
the same increments on validation and binds each accepted load state to its
stress, ordered wall facets, and release row with SHA-256.

Global balance is reported about the length-weighted wall-perimeter centroid.
After inserting the derived support reaction, the resultant and scalar
out-of-plane moment are

```text
q = f_int - f_wall - r_full
F_res = sum_i q_i
M_res = sum_i ((y_i-y_ref) q_z,i - (z_i-z_ref) q_y,i)
```

The force denominator is the largest L1-of-nodal-vector magnitude of internal,
wall, or reaction forces and the explicit force floor. The moment denominator
uses the corresponding sums of absolute nodal moments and `force_floor` times
the maximum reference radius. Both relative diagnostics must meet the frozen
equilibrium tolerance before acceptance.

`damage_connectivity` is also derived. Each mesh edge has capacity equal to
the smaller endpoint damage, and the stored scalar is the widest-path capacity
from any wall node to any far-field node. No reaction or connectivity value is
filled with a zero placeholder.

## Two independent energy gates

The adapter and schema compute the accepted-path balance

```text
abs(Delta(elastic + fracture) - Delta W_external)
--------------------------------------------------
max(abs(Delta energy), abs(Delta work), energy floor)
```

and require at most `5%`. This checks boundary-work accounting between
accepted states.

Separately, the solver requires the symmetric relative change of

```text
Pi = elastic_energy + fracture_energy - neumann_load_functional
```

between two complete staggered iterates to be at most the frozen `1e-8`.
The first iterate has infinite change by definition, so a converged step always
contains at least two complete staggered iterates. Passing either the `5%` path
balance or the `1e-8` fixed-load potential-energy gate cannot substitute for
the other.

## Development command

The command is a dry run unless explicitly acknowledged:

```powershell
.venv-gpu\Scripts\python.exe scripts\run_fracture_phase1_development.py

.venv-gpu\Scripts\python.exe scripts\run_fracture_phase1_development.py `
  --section circle --material m1 --path p1 `
  --execute-development-only
```

It runs exactly one coarse identity and writes only under
`artifacts/development/fracture-phase1` by default. The mesh is deliberately a
coarse adapter diagnostic. It does not satisfy the fine fracture-band mesh or
coupled SENT/SENS three-grid contracts. The separate fixed-state same-mesh
MOOSE comparison has passed, but this coarse trajectory is not that evidence.
Fresh-prefix execution may also be expensive.

Formal 36-case generation remains blocked until the coupled fracture-benchmark
gates, fine-mesh contract, protocol-scale resource gate, and reviewed
publication adapter are closed. The current fresh-prefix retry is deliberately
retained as a development-only, quadratic-cost rollback oracle.
