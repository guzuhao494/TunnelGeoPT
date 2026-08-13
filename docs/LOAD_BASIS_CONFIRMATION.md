# Independent direct-FEM load-basis confirmation

## Status

The frozen confirmation completed on 2026-08-14 from clean, pushed
implementation commit `44d244e344a0e40dbf33fdaa21cc823b8f46a85a`.
Its classification is `LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED`.
All 24 direct solves completed, all 17 gates passed, and the public result is
stored under `artifacts/confirmation/linear-load-basis-v0.5.0/`.

This protocol tests one narrow proposition: for each fixed geometry, material,
fine mesh, and query grid in the present two-dimensional small-strain
plane-strain linear-elastic solver, the response to an arbitrary in-plane
far-field stress is factorizable along three load axes. It is not evidence of
geometry, material, or mesh generalization, and it says nothing about damage,
fracture, rockburst, micro-to-field transfer, field prediction, or engineering
truth.

## Frozen design

There are three explicit new rough geometries, one each from `circle`,
`horseshoe`, and `straight_wall_arch`. Continuous shape parameters, roughness
amplitudes, roughness seeds, and query seeds are literal values in
`configs/load_basis_confirmation.json`; their geometry, boundary, and query
SHA-256 identities are also frozen.

For each geometry, the runner generates one fine P1 mesh and locates one
512-point query grid once. It then reuses that same in-memory mesh, boundary,
and query for all eight direct solves:

- tensor-Frobenius unit basis loads `[1,0,0]`, `[0,1,0]`, and
  `[0,0,1/sqrt(2)]`, in `[sigma_yy,sigma_zz,tau_yz]` order;
- five fixed held-out physical far-field loads, including positive and
  negative shear;
- tension-positive internal convention, so negative normal values mean
  compression.

The total is `3 geometries x (3 basis + 5 held-out) = 24` direct FEM solves.
The physical-coordinate basis has rank 3 and condition number `sqrt(2)`.

The three basis responses are fitted with
`fit_linear_stress_response_basis`; the resulting maps predict all 15
held-out query total-stress fields. The runner additionally checks independent
linear reconstruction of:

- nodal displacement;
- element and query incremental in-plane stress;
- query total in-plane stress (the primary metric);
- element `sigma_xx` as a secondary diagnostic, without widening the claim
  beyond in-plane far-field load-axis factorization.

Every response has a held-out maximum relative L2 gate of `1e-9`. Query total
stress uses the symmetric-tensor Frobenius norm (shear weight 2) and also has a
primary median gate of `1e-11`. Solver algebraic residual and energy closure
must each be at most `1e-9` for all 24 solves.

## Identity exclusion

Plan validation hashes and reads the v0.2/v0.3 legacy exclusion aggregate, the
legacy smoke and convergence query records, and the v0.3 formal public identity
store authenticated by its formal manifest. The three geometry IDs, three
boundary hashes, three query hashes, eight load IDs, and 24 case IDs must be
unique at the appropriate level and have zero intersection with those sources.

The current solver-free plan identity is:

`cf91a557bae100545ad84bec121cc6bbcdcc09e1ae6a7fd3da98c7d9cf463ef5`

## Solver-free validation

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_load_basis_confirmation.py validate-plan
```

This command rebuilds boundaries and queries deterministically, authenticates
the exclusion sources, verifies all frozen identities, and emits canonical
JSON to stdout. It does not generate a mesh, call the elasticity solver, or
write an artifact.

## Real-run preflight and classifications

The real run is intentionally unavailable from a dirty or merely local
checkout. Before the first solve it requires:

- tracked worktree and index clean;
- `HEAD` equal to its configured upstream (therefore already pushed);
- an explicit `--expected-head` exactly equal to `HEAD`;
- frozen SHA-256 values for the runner and load-basis implementation;
- active config tracked, with its current file SHA recorded at runtime;
- output directory absent;
- explicit `--acknowledge RUN_24_NEW_DIRECT_FEM_SOLVES`.

Immediately after this preflight and before any mesh generation or solve, the
runner atomically reserves the absent output directory with an exclusive
`mkdir`. A concurrent creator therefore fails before computation rather than
racing with the final artifact write.

Each solve receives a durable in-memory execution record before the solver is
called. The record is then marked solver-returned and validated-complete, or
failed with a sanitized reason. The final evidence reports exactly 24 planned
solves plus attempted, solver-returned, validated-complete, failed, and
not-attempted counts. A failure on the second solve therefore retains the
completed first-solve record and the failed second-solve record.

After output reservation, any mesh/solver/QC failure writes an
`ABSTAIN_INVALID` artifact rather than losing the negative result. Every float
destined for the artifact is checked for finiteness. As a final fail-safe,
unexpected non-finite values are replaced by JSON `null`, with their exact
paths and reasons recorded under `serialization_validity`, and classification
is forced to `ABSTAIN_INVALID`. The artifact remains canonical, compact,
`allow_nan=false` JSON and is published by temporary-file plus atomic replace.
Its classification is exactly one of:

- `ABSTAIN_INVALID`: identity/hash/direct-path or solver/mesh/QC validity gate
  failed;
- `STOP_BASIS_CONFIRMATION`: the protocol is valid, but at least one numerical
  reconstruction gate failed;
- `LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED`: every validity and
  numerical gate passed.

Only a plan/preflight or output-reservation failure before any mesh/solve
creates no artifact. The guarded real command, to be used only after committing
and pushing the frozen hashes, is:

```powershell
.\.venv\Scripts\python.exe scripts\run_load_basis_confirmation.py run `
  --expected-head <pushed-commit> `
  --acknowledge RUN_24_NEW_DIRECT_FEM_SOLVES
```

Do not run this command from the implementation worktree before the critical
source hashes in the config have been frozen to the pushed files.

## Confirmed result

The run completed all `24/24` planned solves with zero failures and evaluated
all `15` held-out loads. The primary query total in-plane stress RelL2 was:

- median: `4.885724690966474e-15`;
- maximum: `5.882054174085674e-15`.

Maximum auxiliary reconstruction errors were `9.5927e-16` for nodal
displacement, `1.2359e-14` for element incremental stress, `1.2921e-14` for
query incremental stress, and `4.6708e-15` for the report-only `sigma_xx`.
The maximum algebraic residual was `2.0815e-13`; maximum energy-closure error
was `1.3998e-14`. An independent result audit recomputed every aggregate gate,
verified the 24-case ledger and basis/held-out separation, and found no
non-finite values, absolute machine paths, or identity intersections with the
frozen v0.2/v0.3 exclusion sources.

The SHA-256 of `confirmation.json` is
`b86efe9e283f4ee1f00cb2d8cb01d754475fb9a8ddf1e34338d01ecd5c89e736`.
The artifact retains the complete execution ledger and per-held-out errors,
but not the full field tensors; reproducing field-level RelL2 from raw arrays
therefore requires rerunning the direct solves.
