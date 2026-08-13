# TunnelGeoPT v0.3 formal result

## Decision

The preregistered decision is **ABSTAIN**, not `GO` and not `NO_GO`.
`effect_claim_allowed=false` and `claim_scope=null`.  Two validity gates failed
because the paired hierarchical-bootstrap interval for
`Residual50 / Scratch100` was wider than the frozen maximum of `0.10`:

- locked IID: width `0.131422`;
- locked geometry OOD: width `0.112845`.

Validity failure takes precedence over the 11 observed effect/robustness gate
failures.  Those failures are useful diagnostics, but they cannot be promoted
to a confirmatory `NO_GO` result for this run.

The formal run completed on the clean, already-pushed implementation commit
`0f4cc0d504b35092928eb33e43bbbca0d213b545`.  This evidence directory is
published by a later child commit; it does not rewrite the implementation HEAD
recorded by the run.  `PUBLIC_EVIDENCE_SHA256.txt` authenticates the curated
public evidence files.

Key identities are:

- run ID: `mf-residual-formal-v0.3.0`;
- canonical config SHA-256:
  `5564a06529b9895f4577a4ad0489611e1ebc319662ceb52e26651675ba10b5e7`;
- final decision payload SHA-256:
  `f074bf5933b4525d04719693a778df8361cddc0ed30bf9bcdf7a454fa609174f`;
- decision file SHA-256:
  `80db392622a9a706fdafcc430d43607ec3c7b10e30a34f25e15a121de96899f6`;
- checkpoint-registry file SHA-256:
  `884d87cce5dc5ee15d37379c68da9fb324742e2a1ff5eedf6f7469cca7083074`.

All five phases completed on their first attempt.  UTC intervals were:

| Phase | Started | Completed |
|---|---|---|
| prepare | 2026-08-13 13:13:40 | 2026-08-13 13:13:40 |
| generate | 2026-08-13 13:14:13 | 2026-08-13 13:58:16 |
| train | 2026-08-13 13:59:45 | 2026-08-13 14:24:20 |
| evaluate | 2026-08-13 14:25:27 | 2026-08-13 14:25:51 |
| analyze | 2026-08-13 14:26:10 | 2026-08-13 14:26:10 |

## Data and integrity

- `195` parent geometries and `705/705` valid cases; `0` invalid cases and `0`
  result-conditioned replacements; `512` common query points per case.
- Partition case counts: train `288`, dev `72`, IID `120`, geometry OOD `90`,
  load OOD `90`, and joint OOD `45`.
- All `705` FEM/mesh QC records passed.  Maximum free-DOF residual was
  `1.34315e-13`, maximum energy-closure error `1.78758e-14`, and minimum
  triangle quality `0.660192`.
- The preselected fine-ultrafine audit covered `144` cases.  Its overall median
  and p95 discrepancies were `2.30697%` and `3.14749%`; section medians were
  `1.91184%`, `2.42697%`, and `2.48098%`.
- Exactly `35` checkpoints were frozen before evaluation.  The trainer received
  no locked-label path, made zero sealed-store opens, and read zero locked labels
  before checkpoint freeze.  Each of four sealed partitions was opened once;
  `35 x 4 = 140` checkpoint-partition evaluations were recorded exactly once.
- All cross-partition identities were disjoint and all formal identities were
  disjoint from the frozen v0.2 seen-identity exclusion set.
- Formal generation took `2601.77 s` (`43.36 min`) and reported peak process
  memory of `861,528,887` bytes (`0.802 GiB`).  Training all 35 checkpoints took
  about `24.54 min`; the end-to-end five-phase wall interval was about
  `72.50 min`, including deliberate inter-phase checks.
- The recorded environment was Windows 11, Python `3.12.13`, NumPy `2.5.2`,
  SciPy `1.18.0`, scikit-fem `12.0.2`, Gmsh `4.15.2`, and PyTorch
  `2.11.0+cu128` / CUDA `12.8` on an NVIDIA GeForce RTX 5070 Ti Laptop GPU
  (`12,820,480,000` bytes; driver `596.49`).  The implementation manifest
  freezes hashes for 20 executable/configuration sources.

## Primary comparisons

Ratios below use lower-is-better error.  `R_s=Residual50/Scratch100`,
`R_d=Residual50/Direct100`, and `R_c=Residual50/CoarseOnly`.  `U95` is the
frozen one-sided 95% upper confidence bound; `width` is the two-sided 95%
interval width.

| Partition | Ratio | Estimate | U95 | Width | Gate result |
|---|---:|---:|---:|---:|---|
| IID | R_s | 1.065973 | 1.120417 | 0.131422 | fail effect and width |
| IID | R_d | 1.116097 | 1.146392 | 0.067553 | fail effect |
| IID | R_c | 0.989452 | 1.010603 | 0.049395 | fail effect |
| Geometry OOD | R_s | 0.977016 | 1.024499 | 0.112845 | pass effect, fail width |
| Geometry OOD | R_d | 1.101263 | 1.128509 | 0.060852 | fail effect |
| Geometry OOD | R_c | 0.990437 | 1.010627 | 0.046835 | fail effect |
| Load OOD | R_s | 0.545602 | 0.571584 | 0.060270 | pass |
| Load OOD | R_d | 0.904881 | 0.943677 | 0.088788 | pass |
| Load OOD | R_c | 1.074985 | 1.088680 | 0.031423 | fail effect |
| Joint OOD | R_s | 0.510359 | 0.544390 | 0.077688 | report only |
| Joint OOD | R_d | 0.871662 | 0.913412 | 0.096027 | report only |
| Joint OOD | R_c | 1.081804 | 1.096386 | 0.033428 | report only |

Mean near-field tensor relative-L2 errors were:

| Partition | Coarse | Scratch100 | Direct100 | Residual50 | Residual100 | Mismatch50 |
|---|---:|---:|---:|---:|---:|---:|
| IID | 3.356% | 3.115% | 2.975% | 3.320% | 2.999% | 4.579% |
| Geometry OOD | 3.441% | 3.489% | 3.095% | 3.408% | 3.119% | 5.301% |
| Load OOD | 3.860% | 7.604% | 4.585% | 4.149% | 4.202% | 9.929% |
| Joint OOD | 3.794% | 8.042% | 4.709% | 4.105% | 4.275% | 10.349% |

## Interpretation boundary

The exploratory pattern is clear but is not a confirmatory effect claim:

1. `Residual50` did not demonstrate 50% fine-label efficiency on IID or
   geometry OOD; it was worse than `Direct100` and did not reduce raw coarse
   error by the preregistered amount.
2. It strongly stabilized load/joint OOD relative to the learned baselines,
   but remained worse than the raw coarse solver and failed the load-OOD
   wall-offset/coarse-improvement gates.
3. `Mismatch50` was worst in all four partitions, which validates that paired
   coarse-fine correspondence contains useful information; it does not rescue
   the primary claim.
4. `Residual100` approached `Direct100` on IID and geometry OOD, suggesting
   that this residual formulation may need close to full fine-label coverage.

This experiment concerns two-dimensional, homogeneous, isotropic, small-strain
linear elasticity.  It is **not** evidence of fracture, damage evolution,
dynamic rockburst, micro-to-field transfer, or engineering warning ability.
The current locked identities are permanently seen.  Any powered replication
must use a new version, new salt, new identities, and unchanged scientific
thresholds.

## Public/private artifact boundary

The Git publication contains the 13 immutable JSON files listed in
`PUBLIC_EVIDENCE_SHA256.txt`, this summary, and that checksum list.  Local
`*.npz` labels/fields, `*.pt` checkpoints, per-case solver caches, progress
streams, and five path-bearing raw audit files remain private.  They are kept
unchanged locally so their original hashes remain meaningful; they were not
silently sanitized for publication.

The append-only raw access log has 61 records.  The analyzer authenticates the
first 60 records; record 61 is the post-analysis `phase_completed` event that
stores the decision/access-state hashes themselves.  This explains the
expected difference between the final whole-log hash and the analyzer's
60-record prefix hash.  The `effect_claim_allowed=true` field in the sealed
execution artifact means only that the access contract allowed analysis; the
authoritative scientific decision is the later `decision.json` value
`effect_claim_allowed=false`.
