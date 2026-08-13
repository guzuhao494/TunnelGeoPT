# Phase-1 scheduled loading in the local fracture solver

Status: **development only**.  The APIs described here do not implement adaptive
load-step retry, cumulative external-work integration, fracture-schema output,
or any validation that would make the resulting trajectory a usable label.

## APIs

`solve_fixed_damage_displacement_at_load_state(mesh, material, load_schedule,
load_state, ...)` solves one fixed-damage equilibrium state.  Both the schedule
and its state are required.  A state cannot be passed by itself because wall
facet IDs are not a mesh identity.

`solve_at2_fracture_schedule(mesh, material, load_schedule, ...)` runs the local
staggered AT2 kernel at the requested scalar `AT2LoadPath` values.  It returns a
`ScheduledAT2Result`, which is deliberately separate from the legacy result and
from the fracture dataset schema.

The legacy `solve_fixed_damage_displacement` and `solve_at2_fracture` APIs keep
their original uniform wall-release contract.

## Boundary-load contract

For every state, the solver:

1. checks the schedule's wall-facet ID set against the solver marker set;
2. checks each ID-aligned undirected wall-facet endpoint pair, midpoint, and the
   length-weighted wall-perimeter centroid against the current mesh;
3. checks that the supplied state is exactly the schedule state at the same
   `s`, allowing only a joint permutation of facet-aligned rows;
4. converts the state's tension-positive far-field tensor to the current affine
   displacement; and
5. assembles `(1 - release_f) Sigma_inf n` separately on every wall facet.

The geometry checks prevent a P4 state compiled on a different geometry from
being silently accepted merely because both meshes happen to reuse the same
integer facet IDs.

When the far-field tensor changes, the initial total field is

`u_initial = w_previous + epsilon_inf(Sigma_current) x`.

The fixed-state API therefore accepts an optional previous **correction** field,
not a previous total field.  It also checks that the correction vanishes on the
far-field Dirichlet nodes.

## Auditable force output

`DisplacementSolveResult` and each trajectory step expose the final, jointly
reassembled state:

- `internal_force` with shape `[2N]`;
- `wall_nodal_force` with shape `[2N]`;
- sorted `dirichlet_dofs`; and
- `farfield_prescribed_displacement` aligned with those DOFs.

These arrays are sufficient to construct `BoundaryEquilibriumState` from
`fracture_work.py`.  They are assembled after the final damage update from the
same `(u, d)` used for the returned strain, stress, and energy diagnostics.

The historical `external_work` field remains for compatibility.  Its precise
meaning is the instantaneous Neumann load functional `f_wall . u`, also exposed
as `neumann_load_functional`.  It is **not** cumulative trajectory work.  Path
work must be integrated only over accepted states and must include both wall
work and far-field reaction work.

## Regression boundary

The P1 schedule is tested bit for bit against the legacy uniform-release solver
for fixed-damage states and a complete tiny staggered trajectory.  P2 and P3
tests check that the current far-field tensor and prescribed displacement are
used.  P4 tests demonstrate that nonuniform release differs from a uniform
release with the same arithmetic mean and that joint facet-row reordering does
not alter the assembled solution.
