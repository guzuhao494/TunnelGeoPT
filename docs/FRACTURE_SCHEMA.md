# C-fracture trajectory persistence schema

`tunnelgeopt.fracture_schema` is the independent persistence and validation
contract for candidate accepted, quasi-static AT2 fracture trajectories. It
does not extend the fixed-width GeoPT A-layer or the linear B-elastic schema,
and no Phase-1 trajectory has yet been admitted for publication.

The current contract is **schema version 2**. It is intentionally incompatible
with version 1: the reader rejects v1 instead of inferring reactions or path
work from the former ambiguous scalar fields.

One trajectory occupies exactly two schema files:

```text
trajectory_directory/
  arrays.npz
  meta.json
```

The computation/publication default is `float64`. `float32` is supported only
when the caller explicitly selects it at validation, save, and load boundaries.
All integer arrays use `int64`.

## Physical and discretization contract

The schema fixes the following conventions:

- two-dimensional, small-strain plane strain;
- P1 vector displacement and P1 scalar damage on three-node triangles;
- coordinates and vector components in `(y,z)` order, with `x` the tunnel axis;
- engineering strain order `[yy, zz, gamma_yz]`;
- stress order `[yy, zz, yz]`, plus the separate plane-strain `sigma_xx`;
- tension-positive stress;
- `d=0` intact and `d=1` fully broken;
- AT2 fracture energy and an explicitly named spectral-plane-strain or
  volumetric/deviatoric energy split;
- a named normalized main-path parameter in `[0,1]` (Phase-1 uses `s`),
  strictly increasing after the first stored state.

The trajectory is a successful label, not a solver checkpoint. Both
`completed` and `accepted` must be true. Failed trial increments remain in the
attempt ledger, but a final unconverged iterate cannot be published as an
accepted step.

## Exact array contract

`N` is the node count, `M` the element count, and `T` the number of accepted
states. The NPZ member set is exact: unknown members, missing members, wrong
shapes, mixed floating dtypes, non-finite values, or writable in-memory arrays
are rejected.

### Mesh and accepted states

| Array | Shape | Meaning | Unit |
|---|---:|---|---|
| `nodes` | `[N,2]` | nodal `(y,z)` coordinates | m |
| `node_ids` | `[N]` | immutable row-aligned node identity | id |
| `displacement_dof_ids` | `[N,2]` | row-aligned `(y,z)` displacement-field DOF identity | id |
| `damage_dof_ids` | `[N]` | row-aligned damage-field DOF identity | id |
| `elements` | `[M,3]` | P1 triangular connectivity | index |
| `wall_facets` | `[Bw,2]` | excavation-wall boundary edges | index |
| `farfield_facets` | `[Bf,2]` | exterior boundary edges | index |
| `farfield_dirichlet_dofs` | `[D]` | increasing node-major constrained displacement positions | index |
| `area` | `[M]` | element area | m2 |
| `centers` | `[M,2]` | element centroids | m |
| `load_parameter` | `[T]` | accepted normalized main-path coordinate (`s` in Phase-1) | 1 |
| `farfield_stress` | `[T,3]` | accepted `(yy,zz,yz)` far-field tensor, tension positive | Pa |
| `wall_release_by_facet` | `[T,Bw]` | release aligned to the exact ordered `wall_facets` rows | 1 |
| `u` | `[T,N,2]` | total nodal displacement `(u_y,u_z)` | m |
| `internal_nodal_force` | `[T,2N]` | assembled internal force, node-major | N/m |
| `wall_nodal_force` | `[T,2N]` | wall force applied to the rock, node-major | N/m |
| `farfield_prescribed_displacement` | `[T,D]` | prescribed values aligned to constrained positions | m |
| `farfield_reaction_on_rock` | `[T,D]` | support force applied to the rock at constrained positions | N/m |
| `damage` | `[T,N]` | nodal P1 phase field | 1 |
| `strain` | `[T,M,3]` | element engineering strain | 1 |
| `stress` | `[T,M,3]` | element total in-plane stress | Pa |
| `sigma_xx` | `[T,M]` | element out-of-plane stress | Pa |
| `psi_plus` | `[T,M]` | undegraded tensile elastic energy density | J/m3 |
| `psi_minus` | `[T,M]` | compressive elastic energy density | J/m3 |
| `history` | `[T,M]` | tensile history field `H` | J/m3 |

Boundary facets are explicit, unique undirected node pairs. Wall and far-field
sets must be non-empty, disjoint, and cover the complete geometric boundary.
Element areas and centroids are recomputed from the mesh on every validation.
All node and field-local DOF IDs are non-negative and unique within their own
field. Because raw displacement forces are flattened node-major throughout the
contract, `displacement_dof_ids[i]` must equal `[2*i, 2*i+1]`; arbitrary stable
external node IDs remain in `node_ids`, while damage IDs are identity labels and
need not start at zero. Constrained positions may refer only to far-field-facet
nodes.

### Energies and response

| Array | Shape | Meaning | Unit |
|---|---:|---|---|
| `elastic_energy` | `[T]` | integrated degraded elastic energy | J/m |
| `fracture_energy` | `[T]` | integrated AT2 fracture energy | J/m |
| `neumann_load_functional` | `[T]` | instantaneous endpoint functional `f_wall . u` | J/m |
| `wall_work_increment` | `[T]` | accepted-step trapezoidal wall work | J/m |
| `farfield_work_increment` | `[T]` | accepted-step trapezoidal support work | J/m |
| `cumulative_external_work` | `[T]` | cumulative accepted wall + far-field work | J/m |
| `total_potential_energy` | `[T]` | elastic + fracture - instantaneous Neumann functional | J/m |
| `damage_area` | `[T]` | exact P1 integral of `d` | m2 |
| `crack_density_integral` | `[T]` | AT2 crack-density integral | m |
| `damage_connectivity` | `[T]` | frozen `[0,1]` connectivity metric | 1 |

Validation does not trust the totals blindly. For each P1 element it integrates
`d`, `d^2`, and `grad(d)` analytically. It then checks

```text
crack_density_integral = integral[d^2/(2 ell) + ell/2 |grad d|^2] dA
fracture_energy = Gc * crack_density_integral

elastic_energy = integral[
    ((1-d)^2 + k) psi_plus + psi_minus
] dA

neumann_load_functional = wall_nodal_force . u
total_potential_energy = elastic_energy + fracture_energy
                         - neumann_load_functional
```

Thus an element-centre damage substitution cannot masquerade as explicit P1
integration. `damage_area` is also recomputed and cannot exceed the mesh area.

`cumulative_external_work` is deliberately absent from the total-potential
definition. It is a path integral, not an instantaneous load potential.

### Accepted-step diagnostics and work accounting

| Array | Shape | Meaning | Unit |
|---|---:|---|---|
| `displacement_residual` | `[T]` | staggered displacement update residual | 1 |
| `damage_residual` | `[T]` | staggered damage update residual | 1 |
| `equilibrium_relative_residual` | `[T]` | relative force-balance residual | 1 |
| `kkt_relative_residual` | `[T]` | relative bound-constrained KKT residual | 1 |
| `complementarity_relative_residual` | `[T]` | relative active-set complementarity residual | 1 |
| `damage_irreversibility_violation` | `[T]` | maximum accepted-state damage decrease | 1 |
| `damage_range_violation` | `[T]` | maximum bound violation | 1 |
| `history_monotonicity_violation` | `[T]` | maximum history decrease/shortfall | J/m3 |
| `relative_energy_imbalance` | `[T]` | path energy imbalance | 1 |
| `newton_iterations` | `[T]` | all displacement/Newton iterations, including retries | count |
| `active_set_iterations` | `[T]` | all damage active-set iterations, including retries | count |
| `staggered_iterations` | `[T]` | all staggered iterations, including retries | count |
| `step_halvings` | `[T]` | cumulative halvings before the accepted state | count |
| `retry_count` | `[T]` | rejected attempts before acceptance | count |

The accepted-state gates from the solver blueprint are enforced: equilibrium,
KKT, and complementarity relative residuals may not exceed `1e-6`; damage
irreversibility and range violations may not exceed `1e-10`; relative energy
imbalance may not exceed `5%`. Stored damage must remain within `[0,1]`, be
irreversible to `1e-10`, and have a monotone history field satisfying
`H >= psi_plus` to the declared tolerance.

None of the work, reaction, equilibrium, or energy-balance diagnostics is
trusted as an opaque solver scalar. With `D` the stored constrained positions,
version 2 recomputes

```text
r_D = (f_internal - f_wall)[D]
free_residual = (f_internal - f_wall)[not D]

Delta W_wall[i] = 0.5 (f_wall[i-1] + f_wall[i])
                  . (u[i] - u[i-1])
Delta W_far[i]  = 0.5 (r_D[i-1] + r_D[i])
                  . (u_D[i] - u_D[i-1])
W_external[i]  = cumulative sum of both accepted increments
```

The force sign is force **on the rock**. All forces and energies are per unit
out-of-plane thickness. The first stored state is exactly `s=0` and defines
zero wall increment, zero far-field increment, and zero cumulative path work.
The prescribed-displacement array must exactly match `u` at `D`; wall nodal
force must vanish away from wall-facet nodes. The free-DOF relative residual is
recomputed from these raw arrays before the `1e-6` gate is applied.

For each later accepted state, validation recomputes

```text
Delta E = Delta (elastic_energy + fracture_energy)
Delta W = wall_work_increment + farfield_work_increment
relative_energy_imbalance = |Delta E - Delta W|
                            / max(|Delta E|, |Delta W|, E_floor)
```

`equilibrium_force_normalization_floor` and
`energy_balance_normalization_floor` are required finite positive metadata
scalars in N/m and J/m, respectively. Free-DOF equilibrium is normalized only
by the free internal force, free wall force, and the explicit force floor;
large constrained reactions cannot dilute a bad free residual. The schema
never guesses either floor, and the first energy imbalance is fixed to zero
because that state is the path-work reference.

These schema checks admit a record to the C-fracture data layer. They do not
replace mesh-convergence, benchmark, tangent, or cross-implementation evidence.

## Complete attempt ledger

`meta.json` stores an ordered `attempt_ledger`. Every accepted step has one or
more entries with this exact key set:

```text
step_index
attempt_index
load_parameter_start
load_parameter_target
accepted
failure_code
failure_message
newton_iterations
active_set_iterations
staggered_iterations
step_halvings
equilibrium_relative_residual
kkt_relative_residual
complementarity_relative_residual
damage_irreversibility_violation
damage_range_violation
relative_energy_imbalance
load_state_sha256
neumann_load_functional
wall_work_increment
farfield_work_increment
cumulative_external_work
```

Attempt indices are contiguous from zero within each step. Rejected attempts
must precede the single accepted attempt and must carry a non-empty failure code
and message. Accepted attempts carry no failure. The accepted target equals the
stored load parameter, while rejected targets may record a larger increment
that caused step halving.

The per-step Newton, active-set, and staggered counts must equal the sums over
all attempts, not just the accepted solve. Retry counts equal the number of
rejected attempts, and the accepted attempt's halving count and diagnostics
must match the corresponding step arrays. This preserves failed work for cost
accounting and prevents a successful label from hiding retries.

Only an accepted ledger entry carries the load-state hash and the four work
values; all five fields must be `null` on a rejected entry. Accepted values must
match the corresponding arrays to dtype-aware tolerance. Therefore a failed
trial cannot contribute load identity, an increment, or cumulative work.

## Identity, model metadata, and units

The record carries stable non-empty identifiers for the trajectory, parent
case, mesh, geometry, material, and load path. It also requires lowercase
SHA-256 values for the exact configuration and solver source/environment
snapshot. The metadata includes:

- AT2 material values `young_modulus`, `poisson_ratio`, `fracture_energy` (`Gc`),
  `length_scale` (`ell`), `residual_stiffness` (`k`), and an explicit split;
- the geometry definition;
- the normalized main-path coordinate, interpolation rule, and complete
  control-knot schedule for both far-field state and wall release;
- finite positive free-equilibrium and energy-balance normalization floors in
  N/m and J/m;
- positive physical tags for rock, wall, and far field;
- explicit triangle/P1 displacement/P1 damage mesh declarations;
- solver, environment, and caller metadata;
- an exact unit mapping covering every required and optional array.

The schema supports an optional EBR-DNO decomposition. It is all-or-nothing:
`elastic_basis_stress` and `nonlinear_stress_residual`, both `[T,M,3]`, require
an `elastic_basis_id` and SHA-256 link, and validation checks

```text
stress = elastic_basis_stress + nonlinear_stress_residual.
```

No zero placeholder is written when the basis is not used; both optional arrays
are absent from the NPZ.

### Phase-1 load-path representation

The schema does not equate the stored main-path coordinate with uniform wall
release. `load_path.path_parameter` names the coordinate (`s` is canonical for
the Phase-1 pilot), while `control_knots` explicitly store each knot's:

- path coordinate;
- complete numeric far-field state, in any frozen representation such as
  principal magnitudes and angle or tensor components;
- `wall_release` mapping, either the uniform `all` zone or multiple named wall
  zones.

Control-knot coordinates must start at zero, end at one, and be strictly
increasing. Every knot must use one far-field key set and one wall-zone key set.
Each wall-zone release is bounded in `[0,1]` and monotone, but far-field values
may change arbitrarily along the path. Spatially staged release also requires
an explicit wall-zone definition (including any transition/blending rule).

This represents all four Phase-1 paths without pretending they are uniform:
P1 is fixed far-field plus uniform release; P2 changes stress magnitude and
ratio; P3 rotates the principal stress; and P4 independently releases crown,
sidewalls, and invert. Accepted internal substeps need not coincide with control
knots; their `load_parameter` values still increase along the same `s` path.

Every accepted state also stores the evaluated tension-positive stress tensor
and release value for each actual wall facet. `load_state_sha256` commits to the
step's load parameter, stress tensor, release row, and exact ordered wall-facet
rows. It is recomputed rather than trusted. The combination of these arrays and
the complete control-knot metadata is sufficient for a separate schedule
evaluator to audit the load actually applied at every accepted state without
reducing P4 to a uniform scalar.

The generic v2 schema deliberately does **not** infer the physical meaning or
units of every possible numeric far-field representation in `control_knots`.
Consequently, the per-state hash proves internal byte-level binding to the
stored evaluated state; it does not by itself prove that a caller evaluated the
control knots correctly. Phase-1 publication must therefore pass the stricter
`Phase1LoadSchedule.state_at(s)` comparison in the trajectory runner before
schema save. That runner integration remains a label-generation blocker.

## Immutability and integrity

`FractureTrajectory` is a frozen dataclass, but the implementation goes beyond
attribute freezing. Construction copies every array into a bytes-backed view
whose write flag cannot be re-enabled. All mappings, nested mappings, sequences,
and ledger entries are recursively copied and frozen. Mutation of a caller's
source array or dictionary after construction therefore cannot change the
validated trajectory or the bytes later hashed by save.

`meta.json` contains:

- an exact dtype/shape/SHA-256 manifest for every NPZ member;
- a deterministic mesh-content SHA-256;
- an identity-content SHA-256 over coordinates, topology, exact ordered
  wall/far-field facet rows, row-aligned node/field-DOF IDs, and constrained
  node-major DOF positions;
- the exact `arrays.npz` file SHA-256;
- a semantic `content_sha256` covering all metadata and committing to every
  array through its manifest hash.

These are integrity and identity hashes, not signatures of authorship. Load
checks the file hash and semantic hash before reading arrays, verifies the
manifest and mesh hash, reconstructs a new immutable trajectory, and repeats
all semantic checks.

## Atomicity and conflicts

Save validates before publication and uses a per-directory exclusive lock.
Existing `arrays.npz` or `meta.json` files raise `FileExistsError` unless
`overwrite=True`. Temporary files are written in the destination directory,
flushed and `fsync`ed, then individually installed with `os.replace`; metadata
is installed last.

The pair is not claimed to be a transactional database. If a process or machine
fails between replacements, the old metadata cannot validate the new NPZ, so a
reader fails closed rather than accepting a mixed record.

## Excluded and rejected fields

The exact array set has no place for a B-elastic-only record, acoustic-emission
or microseismic events, waveforms, velocity, acceleration, inertia, fragments,
ejection, contact, friction, plasticity, or kinetic energy. Reserved metadata
keys for those concepts are rejected recursively, including zero-valued
placeholders. They require separate solvers, validation, and versioned schemas.

A trajectory with identically zero damage is not rejected merely for being
intact: a legitimate high-toughness or pre-localization AT2 path can do that.
It must nevertheless supply the AT2 material/split, bound-constrained residuals,
accepted-state ledger, explicit AT2 energy integrals, and all fracture identity
metadata. The schema therefore distinguishes evidence by contract rather than
claiming that nonzero damage alone proves a fracture solve.

## Public API

The module exports:

- `FractureTrajectory` and `FractureTrajectoryPaths`;
- `FractureSchemaValidationError`;
- `validate_fracture_trajectory`;
- `save_fracture_trajectory` and `load_fracture_trajectory`;
- `fracture_trajectory_paths`;
- `compute_mesh_content_sha256`, `compute_identity_content_sha256`, and
  `compute_load_state_sha256`;
- schema, convention, unit, dtype, key-set, and gate constants.

Typical persistence is intentionally small:

```python
from tunnelgeopt.fracture_schema import (
    load_fracture_trajectory,
    save_fracture_trajectory,
)

# `trajectory` has already been explicitly adapted from solver results.
trajectory.validate()
save_fracture_trajectory("data/c_fracture/trajectory-0001", trajectory)

published = load_fracture_trajectory("data/c_fracture/trajectory-0001")
assert not published.damage.flags.writeable
```

The schema deliberately does not provide a duck-typed solver-result converter.
Solver adapters must explicitly map their displacement, energy-split, residual,
and iteration definitions into this contract. That prevents similarly named but
differently normalized quantities from being silently accepted.

## Limitations

- This is a persistence and accepted-state validation layer, not an AT2 solver.
- It does not reconstruct strain/stress from displacement or independently
  recompute the spectral split; those belong in solver/benchmark validation.
- `psi_plus`, `psi_minus`, and `history` are one stored value per element. The
  schema exactly integrates the P1 damage factors against those element values,
  but it cannot recover a solver's unarchived higher-order quadrature variation.
- The schema intentionally does not define a one-number reaction observable.
  Any paper-facing reduction or baseline subtraction must be separately frozen;
  the publication record retains the constrained reaction vector needed to
  reproduce such a reduction.
- The solver/trajectory adapter must expose converged internal and wall nodal
  forces, prescribed far-field values, constrained positions, and reactions at
  the same `(u,d)` state. An instantaneous solver compatibility alias named
  `external_work` is not a v2 schema field and cannot stand in for path work.
- `damage_connectivity` is validated only as a finite `[0,1]` metric. Its
  threshold and graph definition must be frozen before formal generation.
- Hashes detect corruption and mixed records but do not prove scientific
  correctness, solver provenance, or authorship.
- The contract supports only synthetic 2D quasi-static brittle phase-field
  fracture. It provides no evidence for compression-shear fragmentation,
  dynamic rockburst, fragment contact/ejection, AE waveforms, or field validity.
