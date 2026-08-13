# TunnelGeoPT v0.4 development-only protocol

## Current decision: implementation stop pending a pivot

The v0.4 runner is implemented for validation and end-to-end mock auditing,
but production cross-fit is deliberately **not authorized**.  Early structured
residual prototypes did not meet the frozen launch margins consistently enough
to justify a large GPU campaign.  The configuration therefore freezes
`real_cross_fit_authorized=false`; a normal invocation fails before reading
experiment inputs or creating an output directory.

This is a scientific stop, not a software failure.  It avoids spending compute
after an unfavorable prototype and prevents a development exercise from being
reported as independent validation.  The supported commands are:

```powershell
# Read-only authentication of config, v0.3 artifacts and fold identities.
& .venv-gpu/Scripts/python.exe scripts/run_v04_development.py --validate-only --device cpu

# Complete state-machine/gate exercise with deterministic mock predictions.
& .venv-gpu/Scripts/python.exe scripts/run_v04_development.py `
  --tiny-mock --device cpu --output <new-empty-output-directory>
```

The following command is intentionally rejected until the protocol is revised
after an explicit pivot decision:

```powershell
& .venv-gpu/Scripts/python.exe scripts/run_v04_development.py `
  --device cuda --output artifacts/analysis/mf-structured-dev-v0.4.0
```

Neither `READY_FOR_NEW_LOCKED_PREREGISTRATION` nor
`IMPLEMENTATION_STOP_PENDING_PIVOT` is a confirmatory `GO`/`NO_GO` effect
classification.  At the current authorization state, even a passing mock gate
remains `IMPLEMENTATION_STOP_PENDING_PIVOT`.

## What the protocol is for

The v0.3 formal experiment completed correctly but classified `ABSTAIN` because
two preregistered bootstrap intervals were too wide.  Its values have now been
observed, so all 705 v0.3 cases are permanently seen.  v0.4 is designed only to
answer a development question: is a physics-structured residual architecture
promising enough to preregister a future experiment on entirely new
identities?

The candidate is a 64-wide, three-block structured linear residual with three
simultaneous constraints:

1. strict linear action in load/stress variables;
2. a deterministic local tensor frame;
3. an exact zero-initialized gate that initially returns the coarse solution.

Three one-switch diagnostics remove exactly one constraint each.  The same
fold also fits a 64-wide, three-block generic v0.3 Residual50 reference.  The
runner rejects a missing, reordered or multi-switch diagnostic, but these
comparisons are not causal component-necessity evidence.

Model shape is similar but parameter count is not equal:

| Model | Trainable parameters |
|---|---:|
| structured candidate | 40,685 |
| generic Residual50 | 38,787 |
| no strict load linearity | 38,598 |
| no local tensor frame | 40,813 |
| no zero-init gate | 40,685 |

There are two further confounds.  The structured candidate trains with
`residual_relative_per_case`, whereas generic models retain the legacy
weighted-MSE objective.  It also receives a 17-channel packed representation,
whereas generic models receive the original 14 channels.  Therefore the
protocol may compare the complete candidate pipelines as development systems;
it may not attribute a difference causally to linearity, frame choice, zero
initialization, parameter count, feature information or loss in isolation.

This remains a two-dimensional homogeneous, isotropic, small-strain linear
elastic discretization-correction study.  It does not model fracture, damage,
dynamic rockburst, micro-to-field transfer, or engineering warning.

## Seen-label and access boundary

The input manifest binds nine v0.3 artifacts by SHA-256: the public archive,
train/dev label store, four former sealed stores, dataset manifest, scientific
decision and checkpoint manifest.  Hashing a former sealed file authenticates
its bytes but does not materialize its fine-stress array.

At input audit time:

- the 288 `train_id` cases and 72 `dev_id` cases are opened;
- the four old locked partitions are renamed `seen_iid`,
  `seen_geometry_ood`, `seen_load_ood` and `seen_joint_ood`;
- their 345 fine-label case values remain unopened;
- no generator is imported or invoked;
- no new locked partition or identity is created.

The runner's batch constructor rejects an unopened fine-label row.  Former
locked values can be opened only by the explicit post-selection function,
after the final five checkpoint hashes have been frozen.  The access log records
the checkpoint-freeze event before the seen-label-open event.

## Fair cross-fit identity design

Architecture cross-fit uses only the original 72 `train_id` parent geometries,
stratified by the three section families.  It never trains on the old locked
partitions.  Salted SHA-256 ordering followed by five-fold round-robin gives:

| Fold | OOF parents | Non-OOF train parents | Candidate/reference optimizer | Fixed dev |
|---:|---:|---:|---:|---:|
| 0 | 15 | 57 | 36 (12/section) | 18 (6/section) |
| 1 | 15 | 57 | 36 (12/section) | 18 (6/section) |
| 2 | 15 | 57 | 36 (12/section) | 18 (6/section) |
| 3 | 15 | 57 | 36 (12/section) | 18 (6/section) |
| 4 | 12 | 60 | 36 (12/section) | 18 (6/section) |

Every parent occurs in exactly one OOF fold, and all loads and query points
follow the parent.  Candidate, generic Residual50 and each ablation use exactly
36 fine parents in every fold.  The same-fold Scratch and Direct comparators
use every available non-OOF train parent (57 or 60), because they represent
the full-label comparator under that fold, not the frozen 72-parent v0.3 model.

The same original 18 `dev_id` parents are reused for early stopping in all five
folds.  This is allowed for architecture development but creates fold
dependence.  Consequently the paired parent-group bootstrap upper bounds are
explicitly development heuristics, not formal confidence intervals.

OOF parents have zero intersection with optimizer, normalization and dev
parents.  Former locked parents have zero intersection with every cross-fit
optimizer, normalizer and early-stopping set.

## Structured feature contract

Generic models receive the common original 14 channels.  The structured model
packs those channels to 17 by appending a deterministic frame `(n_y,n_z)` and a
wall-role bit:

- at wall-offset points, the frame uses the stored physical wall normal from
  the frozen public geometry;
- at other points, it uses the negative nearest-distance gradient;
- the frame performs a defined tensor-coordinate transformation and is not a
  freely learned physical input.

The structured encoder remains 64-wide with three global-context blocks, but
its parameter count still differs from the generic model.  Its additional
frame and wall-mask channels are deterministic information derived from frozen
public geometry, yet they are still additional derived information.  The
structured branch also uses a different relative-residual loss with the
correct symmetric-tensor weight on shear.  These design differences are
disclosed rather than normalized away.

## Two distinct comparator sets

The protocol deliberately keeps two comparator sets separate:

1. **Architecture cross-fit comparators.** Same-fold generic Residual50 uses
   the same 36-parent budget; Scratch and Direct use all 57/60 non-OOF parents.
2. **Post-selection seen-stress comparators.** These are the authenticated,
   already frozen v0.3 Scratch100, Direct100 and generic Residual50 checkpoints.
   They are evaluated against a newly initialized final structured candidate.

The final candidate identity is not resampled.  It is recovered from the
authenticated v0.3 Residual50 contract:

- optimizer and any fitted normalization: exactly the original 36 `train_id`
  parents, 12 per section;
- early stopping: exactly the original 18 `dev_id` parents, 6 per section;
- five fresh initializations: seeds 103, 211, 307, 401 and 509;
- old locked labels: zero optimizer, normalization or early-stopping access.

## Frozen development gates

### OOF architecture gate

On original-train OOF parents, the champion must satisfy all of the following:

- point ratio to same-fold generic Residual50 at most 0.98;
- point ratio to each one-switch diagnostic at most 0.99;
- point ratio to same-fold Scratch and Direct at most 0.95;
- parent-group paired one-sided upper heuristic at most 1.00 for every
  comparison;
- all three section ratios at most 1.02;
- at least four of five seeds pass their point thresholds.

If this gate fails, execution stops before final fitting and before any former
locked fine value is opened.

A passing or failing diagnostic ratio describes the whole implemented
pipeline.  It cannot establish that the toggled component is necessary or
sufficient because parameter count, loss and derived feature information are
not all matched.

### Final seen-stress gate

If the OOF gate passes, five final 36/18 checkpoints are frozen before opening
the former locked values.  The three primary seen partitions then require:

| Seen partition | Point ratio to each frozen learned baseline | One-sided upper heuristic to coarse |
|---|---:|---:|
| IID | <= 0.95 | <= 0.68 |
| Geometry shift | <= 0.98 | <= 0.78 |
| Load shift | <= 0.98 | <= 0.78 |

The ratio to frozen v0.3 generic Residual50 must additionally be at most 0.98.
At least four of five seeds and every section must meet the corresponding
absolute margins.  The original v0.3 wall-offset conditions are unchanged:

- `D_t(candidate) <= 1.10 D_t(coarse) + 0.005`;
- `D_r(candidate) <= 1.10 D_r(coarse) + 0.0025`;
- the original per-partition absolute caps must also pass.

`seen_joint_ood` is reported but is never a primary launch gate.  These are
all seen-data development heuristics; the word OOD here describes how v0.3
generated the partition, not an independent v0.4 validation claim.

## Production preflight and artifacts

Even after a future pivot authorizes real execution, the preflight requires:

- a clean Git worktree;
- `HEAD` equal to its configured upstream;
- tracked config, runner and structured-model sources;
- SHA-256 for each critical source;
- an available explicit CUDA device and recorded PyTorch/GPU provenance.

The runner writes canonical JSON with a SHA-256 chain for the config,
preflight, input audit, folds, OOF metrics/gate, final-fit manifest, seen-open
audit, seen metrics/gate and launch decision.  Checkpoints are development-only
and carry `effect_claim_allowed=false`.  Output names containing a new locked
artifact are forbidden.

## Verification

`tests/test_v04_development.py` covers the protocol's main counterexamples:

- source hash drift;
- a multi-factor or silently ineffective ablation;
- launch-threshold drift;
- any new locked partition;
- unauthorized production execution;
- a parent crossing OOF/optimizer/dev roles;
- a label budget other than 36 train and 18 dev;
- a former locked label entering a training batch;
- final identities differing from the authenticated v0.3 contract;
- a complete tiny workflow that verifies checkpoint freeze precedes seen-label
  open and still returns the implementation-stop classification.

The current implementation is therefore ready for protocol inspection and a
pivot decision, not for a production result claim.
