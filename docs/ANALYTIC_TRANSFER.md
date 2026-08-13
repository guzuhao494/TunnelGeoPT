# Kirsch analytic-transfer smoke

This experiment is a deliberately narrow intermediate milestone. It asks whether a
principal-stress-conditioned geometric lifting task can improve data efficiency for a
shared DeepSets surrogate of the closed-form Kirsch stress field around a circular opening.

It does **not** establish transfer to a horseshoe or straight-wall-arch tunnel, nonlinear
material response, damage, fracture, rockburst, laboratory measurements, or field validity.

## Frozen contract

The runner reads `configs/analytic_transfer_smoke.json` as strict JSON and rejects changes to
the critical contract:

- 240 canonical load cases from a deterministic scrambled Sobol design;
- 16 load strata (four stress-ratio bins by four principal-azimuth sectors);
- SHA-256 case identities and case-level 168/36/36 train/dev/locked-test assignment;
- 512 labels per case: 384 annulus points, 64 wall points, and 64 far-field points;
- a shared 11-input DeepSets backbone and a replaced 9-to-3 output head for transfer;
- equal optimizer, case batch, maximum epoch, and dev-only early-stopping rules;
- three fixed training seeds and all six preregistered methods.

The 80% training subset contains `floor(168 * 0.8) = 134` whole cases and is the hash-ordered
prefix of the 168-case set. No point-level split is implemented.

## Stress-Lift construction

For a query point, `n` is the rock-side unit normal from the closest circular wall point to
the query, `K = sigma3 / sigma1`, and `e1` is the principal stress direction. The anisotropic
factor and velocity are

```text
q = K + (1 - K) (e1 dot n)^2
smax = R (1 - K)
v = -smax q n
condition = [0, sin(alpha), cos(alpha), 1 - K]
```

Three vector-distance targets are recorded before successive steps. A trajectory that first
intersects the wall is placed exactly on the wall and remains there. Wall points are fixed.

Controls are intentionally diagnostic:

- **Static** repeats the initial vector-distance field in all nine output channels.
- **Random** retains each case's `1-K` step-magnitude marginal but samples independent uniform
  directions.
- **Shuffled Stress-Lift** uses a deterministic no-fixed-point derangement. Only conditions are
  permuted; the original targets stay in place, creating a contradictory negative control.
- **Scratch 80/100** use the same downstream network without pretraining.

All pretraining methods read the 168 training cases only. Full-backbone fine-tuning uses the
134-case nested subset for `*_80` methods and all 168 cases for `scratch_100`. Development
cases select the early-stopping checkpoint. Initial dataset construction materializes only
train/dev labels. The data layer rejects and counts any locked-test label access until all 18
checkpoints have been atomically stored as CPU state dictionaries and explicitly authorize the
test phase. Only then are the 36 locked-test labels generated.

Each checkpoint is written independently and indexed for crash recovery, then its model is
moved off the GPU and released. Final evaluation reloads one checkpoint at a time. One logical
evaluation call makes three complete locked-test forward passes: the primary prediction, the
unrotated prediction used by the equivariance diagnostic, and its rotated counterpart. The
manifest reports evaluation calls, actual forward passes, forward batches, and label case reads
separately.

## Evaluation

The primary metric is first calculated per case:

```text
||[dyy, dzz, sqrt(2) dyz]||_2 / ||[yy, zz, sqrt(2) yz]||_2
```

Only then are cases equally averaged. Secondary metrics cover normalized wall traction,
far-field stress, peak absolute wall hoop stress, and circle/load rotation equivariance.
The candidate/reference uncertainty is a paired bootstrap at `case_group_id` level, stratified
by the frozen load stratum. There is no point bootstrap.

The smoke Go/No-Go gate remains exactly the one in the configuration. The formal five-seed,
1.02 upper-confidence-bound gate is explicitly not evaluated by this smoke.

## Frozen result: No-Go

The full CUDA run completed all six methods and three seeds in 416.9 seconds. Stress-Lift@80%
relative to Scratch@100% produced error ratios of 1.166, 0.638, and 1.618 for seeds 17, 29,
and 43. Only one seed passed the 1.05 upper-confidence-bound gate, below the required two.
Stress-Lift also had a worse three-seed mean than Random-Lift, while the shuffled control had
the best three-seed mean and passed in the same seed 29. The preregistered status is therefore
`NO-GO`; no 20% label-saving claim is made. Full interpretation and integrity checks are in
`docs/MILESTONE_V0.2.md`.

## Commands

Wiring-only GPU dry run (does not infer on locked test and is not an effect result):

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_analytic_transfer.py --mode dry-run --device auto
```

Full preregistered smoke:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_analytic_transfer.py --mode full --device auto
```

Durable outputs are written below `artifacts/experiment/analytic-transfer-v0.2.0/`, including
the exact configuration hash, environment snapshot, progress log, training-access audit,
CPU checkpoint index, data-layer access audit, per-case locked-test metrics, bootstrap gate,
and final manifest.
