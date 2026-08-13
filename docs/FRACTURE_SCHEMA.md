# C-fracture trajectory persistence schema

`tunnelgeopt.fracture_schema` is the independent publication contract for
accepted, quasi-static AT2 fracture trajectories. It does not extend the
fixed-width GeoPT A-layer or the linear B-elastic schema.

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
| `elements` | `[M,3]` | P1 triangular connectivity | index |
| `wall_facets` | `[Bw,2]` | excavation-wall boundary edges | index |
| `farfield_facets` | `[Bf,2]` | exterior boundary edges | index |
| `area` | `[M]` | element area | m2 |
| `centers` | `[M,2]` | element centroids | m |
| `load_parameter` | `[T]` | accepted normalized main-path coordinate (`s` in Phase-1) | 1 |
| `u` | `[T,N,2]` | total nodal displacement `(u_y,u_z)` | m |
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

### Energies and response

| Array | Shape | Meaning | Unit |
|---|---:|---|---|
| `elastic_energy` | `[T]` | integrated degraded elastic energy | J/m |
| `fracture_energy` | `[T]` | integrated AT2 fracture energy | J/m |
| `external_work` | `[T]` | external work with the solver's documented sign | J/m |
| `total_potential_energy` | `[T]` | elastic + fracture - external work | J/m |
| `reaction` | `[T]` | signed scalar load/reaction observable | N/m |
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

total_potential_energy = elastic_energy + fracture_energy - external_work
```

Thus an element-centre damage substitution cannot masquerade as explicit P1
integration. `damage_area` is also recomputed and cannot exceed the mesh area.

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
- `compute_mesh_content_sha256`;
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
- `reaction` is one signed scalar observable; its direction and aggregation must
  be frozen in the load-path metadata.
- The current local `AT2StepResult` exposes `elastic_energy`, `fracture_energy`,
  `external_work`, and `total_potential_energy`, so those schema arrays can be
  mapped without inventing values for the existing uniform wall-release solve.
  It does not expose the required scalar reaction. Reaction must be computed
  and archived by the solver/runner before a trajectory can satisfy this
  schema; `external_work` is not a reaction surrogate.
- The current solver's `external_work` is the instantaneous uniform wall-load
  functional `f_wall . u` at fixed far-field stress. It is not yet cumulative
  path work and does not include a general non-proportional far-field schedule.
  Therefore it cannot by itself substantiate Phase-1 P2/P3 path-energy balance;
  a path-aware work/reaction implementation is still required.
- `damage_connectivity` is validated only as a finite `[0,1]` metric. Its
  threshold and graph definition must be frozen before formal generation.
- Hashes detect corruption and mixed records but do not prove scientific
  correctness, solver provenance, or authorship.
- The contract supports only synthetic 2D quasi-static brittle phase-field
  fracture. It provides no evidence for compression-shear fragmentation,
  dynamic rockburst, fragment contact/ejection, AE waveforms, or field validity.
