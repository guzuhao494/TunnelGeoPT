# Prescribed-displacement AT2 BVP contract

Status: **development-only numerical infrastructure**.  This module enables
future coupon benchmarks such as single-edge-notched tension or shear.  It is
not itself a SENT/SENS result, an external validation, a tunnel-fracture label,
or a rockburst claim.

## Why this is separate from the tunnel solver

`tunnelgeopt.fracture` implements the existing tunnel far-field and excavation
wall-release semantics.  Coupon tests instead prescribe selected displacement
components and measure support reactions.  The implementation therefore
lives in `tunnelgeopt.fracture_bvp` and does not reinterpret or refactor the
tunnel load schedule.

Mesh nodes follow the repository coordinate convention `nodes[:, :] = [y, z]`.
Every displacement index is node-major:

```text
u_y(node i) = component 0 = 2 i
u_z(node i) = component 1 = 2 i + 1
```

The generic BVP solver operates on component 0/1 DOF indices and does not infer
physical boundary names.  A SENT/SENS adapter is responsible for assigning
the explicit y/z boundary semantics and unambiguous reaction-group names.

## Exact state contract

`PrescribedDisplacementState` contains:

- a unique state `identity`, exact `sequence_index`, and finite monotone path
  coordinate `path_parameter`;
- `mesh_identity`, the SHA-256 digest of ordered float64 nodes and ordered
  int64 triangles;
- exactly increasing `dirichlet_dofs` and one aligned value per DOF;
- the complete node-major `external_force` vector;
- immutable `reaction_groups`, each an exactly increasing subset of the
  prescribed DOFs; and
- a `driven_group` key used for the scalar load-displacement curve.

All arrays are copied and made read-only.  Reaction-group mappings are also
read-only.  The solver rejects duplicate or missing DOFs, non-finite values,
mesh-identity mismatches, underconstrained rigid-body modes, duplicate state
identities, nonconsecutive sequence indices, non-increasing path coordinates,
or changing prescribed/reaction-group topology along a path.

Use `prescribed_displacement_mesh_identity(mesh)` to obtain the required exact
discrete identity.  Reordering nodes or elements intentionally changes this
identity because it changes the DOF contract.

## Equilibrium, reactions, load, and work

For internal nodal force `f_int` and explicitly applied nodal force `f_ext`,
the reported complete residual is

```text
r = f_int - f_ext .
```

On prescribed DOFs this is the support-on-rock reaction.  On free DOFs it is a
numerical equilibrium residual and must pass the configured tolerance.  The
reported generalized load is deliberately simple and auditable:

```text
Q_n = sum(r_n[d] for d in reaction_groups[driven_group]).
```

Between two accepted states, the complete prescribed-DOF reaction is used for
trapezoidal path work:

```text
Delta W_D,n = 0.5 (r_D,n-1 + r_D,n) dot (u_D,n - u_D,n-1)
W_D,n       = W_D,n-1 + Delta W_D,n .
```

Thus fixed supports contribute exactly zero through their zero displacement
increment.  `Q_n` is a plotting scalar; it is not substituted for the full
reaction vector in the work integral.  The first solved state has zero work
increment because no prior accepted state is defined.

## Fixed-damage and staggered APIs

`solve_fixed_damage_displacement_bvp` solves one displacement equilibrium for
an exact state and fixed nodal damage.  It reports the full internal, applied,
and reaction vectors; named reaction groups; strain, stress, split energies;
and free-DOF equilibrium diagnostics.

`solve_at2_dirichlet_path` alternates the displacement solve with the existing
bound-constrained P1 AT2 damage solve.  Optional `initial_damage` and
elementwise `initial_history` are accepted only when finite and admissible.
Damage is bounded below by the preceding accepted damage, while history is the
elementwise maximum of preceding history and current tensile energy.  A failed
state raises by default and is never silently treated as an accepted state.

At a fixed prescribed state, the staggered potential is

```text
Pi = E_elastic + E_fracture - f_ext dot u .
```

For the intended pure-Dirichlet coupon setup `f_ext = 0` exactly, so
`Pi = E_elastic + E_fracture`.  The energy-increment gate is independent of
displacement change, damage change, free-DOF equilibrium, and damage KKT
gates.  As in the tunnel solver, the first complete staggered iterate has an
infinite relative energy change, forcing at least two complete iterations
before convergence.

## Verified local tests and remaining boundary

The local test suite covers an affine displacement patch, node-major DOF
application, reaction sign and global balance, immutable state/result arrays,
mesh and sequence rejection, rigid-mode rejection, initial damage/history,
irreversibility, full-reaction trapezoidal work, and potential-energy
accounting.  Existing tunnel-fracture tests are run separately to guard the
unchanged tunnel semantics.

No notch geometry, literature benchmark curve, crack path, mesh-convergence
result, or external solver comparison is asserted here.  Those require a
separately frozen SENT/SENS protocol and executed evidence.
