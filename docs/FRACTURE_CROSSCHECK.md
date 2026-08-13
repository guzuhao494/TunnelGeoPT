# Local-MOOSE same-problem cross-check (v1)

## Purpose and claim boundary

This gate asks a narrow question: when the local kernel and a pinned MOOSE
binary solve the same P1 plane-strain equilibrium problem on the exact same
TRI3 mesh, do their displacement, element strain/stress, out-of-plane stress,
elastic energy, and far-field reaction agree to a primary relative-L2 tolerance
of `1e-6`?

A pass supports only fixed-damage quasi-static equilibrium on this mesh. It
does **not** validate coupled phase-field evolution, irreversibility, adaptive
retry, trajectory work, SENT/SENS crack benchmarks, dynamic rockburst, field
transfer, or production labels.

## Frozen identity

- Mesh file: `moose/fracture_crosscheck/canonical_square_annulus_v1.msh`.
- Format: ASCII Gmsh 2.2, first-order line boundaries and TRI3 rock cells.
- Physical groups: `rock=1`, `wall=2`, `farfield=3`.
- MOOSE reads a byte-identical per-case copy with
  `FileMeshGenerator/allow_renumbering=false`.
- The runner hashes the complete mesh file and a canonical structure payload
  containing node tags/coordinates, cell tags/connectivity, and ordered
  boundary-line tags/connectivity.
- MOOSE IDs are not trusted. Nodes are mapped by a unique coordinate bijection;
  elements are mapped by a unique centroid bijection only after the identical
  input-mesh hash has been proved.
- The manifest conservatively binds every `src/tunnelgeopt/*.py` package file,
  plus the runner, configuration, both MOOSE templates, and packaging contract,
  by per-file SHA-256 and byte size. It also
  records Python/dependency versions plus the repository HEAD, upstream, and an
  honest pre-run dirty-state summary. A dirty development tree is allowed only
  because every relevant implementation file is independently content-bound.

The coordinate map is

```text
MOOSE x = local y
MOOSE y = local z
MOOSE z = tunnel axis
```

Thus `disp_x -> u_y`, `disp_y -> u_z`, MOOSE `(stress_xx, stress_yy,
stress_xy) -> local (sigma_yy, sigma_zz, sigma_yz)`, and MOOSE `stress_zz ->
local sigma_xx`. MOOSE tensor `strain_xy` is multiplied by two before it is
compared with local engineering shear `gamma_yz`. Both solvers use
tension-positive stress.

## Gates

1. **Intact gate.** `d=0`, `k=0`, three linearly independent far-field stress
   bases, affine displacement on `farfield`, and natural zero traction on the
   fully released `wall`.
2. **Fixed-damage gate.** Runs only after all three intact cases pass. Damage is
   the same nonuniform global P1 function in both solvers. Degradation is
   exactly `(1-d)^2 + k`; the runner does not substitute MOOSE's normalized
   eta form.

Both use small strain, `PLANE_STRAIN`, MOOSE out-of-plane direction `z`, P1
displacement, replicated mesh, one thread, PETSc direct LU, and explicitly
`THIRD`-order Gaussian quadrature in the valid
`[Executioner]/[Quadrature]` block (`order`, `element_order`, and `side_order`;
no `[Problem] quadrature_order`). For fixed P1 damage this integrates the
quadratic `(1-d)^2` factor exactly on TRI3. Both fixed-damage
`DerivativeParsedMaterial` objects set `enable_jit=false`,
freezing their bytecode evaluator and avoiding compiled JIT/toolchain
dependence. MOOSE still creates a small symbolic-derivative `.jitcache` for
these parsed materials. The runner removes only that exact generated directory
from each fresh case after execution and records its file count and byte size
as excluded ephemeral state; a curated evidence directory must contain no
`.jitcache`. Element fields are constant-MONOMIAL AuxKernel projections
(JxW/volume averages over all volume
quadrature points) and are checked against the local analytical element
averages rather than assumed equivalent.

For the pinned fixed-damage material,
`ComputeLinearElasticPFFractureStress::computeQpStress` does not populate the
`elastic_strain` property consumed by `ElasticEnergyAux`; the resulting raw
CSV energy column is therefore retained and hashed only as a non-comparable
diagnostic. It is excluded from every pass/fail metric. Energy is instead
recomputed offline from the exported MOOSE element strain, the exported and
coordinate-aligned MOOSE nodal P1 damage, and the frozen local
`g*psi_plus + psi_minus` formula. The intact gate additionally checks the
MOOSE-reported linear-elastic energy density as an auxiliary consistency test.

## Fail-closed checks

The run is invalid if any of the following occurs:

- config keys or frozen semantics differ;
- the mesh hash changes, connectivity is malformed, or boundary groups do not
  partition the boundary;
- MOOSE cannot be executed and hashed, exits nonzero, or omits a sampler CSV;
- CSV headers are ambiguous, IDs repeat, coordinates do not form a bijection,
  or any numeric value is non-finite;
- any required field exceeds the frozen primary threshold;
- the intact gate fails (the fixed-damage gate is then skipped).

Near-zero fields use a declared absolute tolerance only to scale their primary
error; this rule prevents meaningless division by a zero reference while
retaining the `1e-6` primary gate.

## Commands

Local preparation (does not claim cross-solver validation):

```powershell
.venv-gpu\Scripts\python.exe scripts\run_fracture_crosscheck.py `
  --artifact-dir artifacts\development\moose-local-same-problem-v1\local-only
```

Real WSL MOOSE run:

```powershell
.venv-gpu\Scripts\python.exe scripts\run_fracture_crosscheck.py `
  --run-moose `
  --artifact-dir artifacts\development\moose-local-same-problem-v1\real-run-YYYYMMDD-HHMMSS
```

The artifact directory must be new and empty. Existing output is never reused:
each node/element CSV filename, byte size, and SHA-256 is bound into the case
manifest and then into the comparison report. This prevents a successful
process exit from accidentally validating stale sampler output.

Every case retains the rendered input, byte-identical mesh, local NPZ,
sanitized MOOSE log, raw MOOSE CSV files, a case manifest, and a comparison
report. The top-level manifest records config/mesh/MOOSE hashes and the final
scope boundary. An official MOOSE self-test is useful environment evidence but
is not a substitute for this same-problem comparison.

## Recorded development result

The curated artifact at
[`artifacts/development/moose-local-same-problem-v1`](../artifacts/development/moose-local-same-problem-v1)
contains one final-v7 run. All six cases passed the frozen `1e-6` gate. The
largest primary error was `2.0954757928848265e-12` for the intact cases and
`2.83634290099144e-9` for the fixed-damage cases. The artifact contains 51
files (145,266 bytes); its manifest SHA-256 is
`ad4b8992714c2165b8b9f7691c1f8b144f9f3d6956b96394bdfbf4b8e27ec23a`.

This result confirms only same-mesh fixed-displacement equilibrium under the
frozen intact and fixed-nonuniform-damage states. It does not authorize coupled
fracture trajectories or Phase-1 labels.
