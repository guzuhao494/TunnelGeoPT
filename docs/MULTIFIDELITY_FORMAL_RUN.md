# v0.3 multi-fidelity formal runner

`scripts/run_multifidelity_formal.py` is the execution harness for the frozen
coarse-to-fine static linear-elastic experiment.  It is not a fracture,
damage, rockburst, three-dimensional dynamic, field-transfer, or engineering
truth experiment.

## Status and execution boundary

The five phases are separate invocations:

```text
prepare -> generate -> train -> evaluate -> analyze
```

There is deliberately no `all` command.  The execution approval is an external
JSON document bound to the canonical SHA-256 of the complete config.  `prepare`
checks the frozen config, the completed development convergence/ultrafine
gate, and authorization before it creates state.  Each later phase verifies
its predecessors and hashes.  Completed phases are resumable no-ops; training
resumes only from individually verified checkpoints.  Evaluation is
fail-closed: an interrupted evaluation becomes `ABSTAIN`, because reopening a
sealed partition would violate the one-open contract.

This is a process/code-path boundary, not a claim of operating-system sandbox
security.  The trainer-facing API returns only public inputs/coarse fields and
the train/dev fine-label store.  It does not return a sealed path.  The
evaluation phase alone calls the trusted locked-label path helper after the
checkpoint registry is frozen.  Four independent sealed files are each read
into memory exactly once; all 35 checkpoints are then evaluated against those
in-memory labels.

## Frozen model set

Five seeds produce exactly seven v2 formal checkpoints per seed:

- Scratch at 100% labels;
- Direct+Coarse at 100%;
- Residual+Coarse at 25%, 50%, 75%, and 100%;
- Mismatched-Coarse at 50%.

The 25/50/75/100 parent sets are section-balanced and nested.  Every checkpoint
uses a batch-derived `TrainingSelection`, a hashed `TrainingContract`, the
formal train/dev entry point, CPU state tensors, and atomic replacement.  A
35-member `CheckpointRegistry` is frozen before sealed evaluation.  A caller
cannot self-report a fine fraction: the contract recomputes it from the actual
selected parent geometries and rejects disagreement.

Training uses region weights with masses nearfield/wall-offset/farfield =
0.80/0.15/0.05.  The primary metric uses a different array with nonzero mass
only in the nearfield.  Wall-offset diagnostics use their own arc weights,
rock-side normals, and prescribed far-field stress scales.

## Commands

The real run requires the frozen config and approval.  Run one command in a
fresh process, inspect the phase result, then invoke the next phase:

```powershell
$cfg = "configs/multifidelity_formal.json"
$approval = "configs/multifidelity_formal_approval.json"
$out = "artifacts/experiment/mf-residual-formal-v0.3.0"

python scripts/run_multifidelity_formal.py prepare  --config $cfg --approval $approval --output $out
python scripts/run_multifidelity_formal.py generate --config $cfg --approval $approval --output $out
python scripts/run_multifidelity_formal.py train    --config $cfg --approval $approval --output $out --device cuda
python scripts/run_multifidelity_formal.py evaluate --config $cfg --approval $approval --output $out --device cuda
python scripts/run_multifidelity_formal.py analyze  --config $cfg --approval $approval --output $out
```

Do not run `generate` until the repository development audit has converged and
the user has authorized formal generation.  The runner never generates or
opens locked labels during tests.

`--backend tiny-mock` exists only for automated state-machine tests.  Its
config must have a `tiny-mock-...` run id and a non-scientific scope, and its
analysis is unconditionally `ABSTAIN` with `effect_claim_allowed=false`.

## Artifacts and audit

The run directory contains atomic JSON state, append-only access/progress
events, a dataset manifest, public and train/dev stores, v2 checkpoints,
checkpoint manifest/registry, sealed metrics, and a final decision.  Phase
state exposes only public/train/manifest generation artifacts; sealed paths
remain inside the trusted generator/evaluator boundary.

The generator manifest authenticates split/section/load identities and file
hashes.  The analysis checks case -> parent -> equal-section aggregation,
paired seed/parent bootstrap, the frozen one-sided confidence gates, 4/5 seed
stability, IID section robustness, OOD load-subtype ratios, and wall-offset
`D_t`/`D_r` absolute and coarse-nonworsening gates.  Joint OOD is mandatory
report-only and does not make the run abstain solely for a wide confidence
interval.

`ABSTAIN` takes precedence over `NO_GO`: leakage/access-count failure,
incomplete seeds, non-finite predictions, solver/mesh valid fraction below
95%, failed or missing fine-ultrafine evidence, an interrupted sealed phase,
or an over-wide primary interval invalidates effect interpretation.  If the
experiment remains valid but an effect/robustness threshold fails, the result
is `NO_GO`.  `GO` requires every validity and effect gate.

Mandatory checkpoint/partition records include per-case errors, mean/median/
p90, section and load-subtype summaries, non-finite count, wall-offset
traction/resultant discrepancies, runtime, Python peak memory, and GPU peak
memory where CUDA is used.  The evaluation phase releases each model and
empties the CUDA cache before loading the next checkpoint.

## Tiny/mock verification scope

`tests/test_multifidelity_formal.py` covers:

- hash-bound approval and the exact 35-checkpoint matrix;
- denied premature sealed access;
- rejection of a fabricated fine fraction;
- completed-phase/checkpoint recovery;
- exactly one open per sealed partition and one evaluation per checkpoint;
- invariance of checkpoint hashes to changed test labels;
- fail-closed interrupted evaluation;
- permanent `ABSTAIN` for mock results.

These tests exercise the integrity state machine without generating or reading
any formal locked/test label and without making a scientific claim.
