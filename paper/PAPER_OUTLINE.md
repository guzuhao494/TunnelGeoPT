# Paper outline and evidence contract

## Paper view

### One-sentence idea

Exact representation of the inexpensive intact elastic response may let scarce
fracture trajectories be spent on learning only irreversible nonlinear
deviations, rather than forcing a neural operator to relearn both regimes.

### Story spine

- **Problem:** high-resolution brittle-fracture trajectories around varied
  tunnel sections are expensive, and data-driven rollout models must learn a
  large intact response before seeing sparse crack evolution.
- **Gap:** geometry-aware and multi-fidelity operators do not normally exploit
  the exact three-dimensional in-plane load space of the intact tunnel system.
- **Method:** compute a nine-channel elastic response basis per geometry, then
  condition a causal irreversible damage/stress-residual operator on that basis.
- **Main result required:** with 50% fracture trajectories, the method matches a
  same-architecture Scratch100 model on new geometries and load paths while
  passing physical and cost gates.
- **Limit:** 2D, quasi-static, synthetic phase-field fracture; no dynamic
  rockburst, fragments, monitoring signal, or field validation.

### Scoped claims

#### C1 - Label efficiency

Within the frozen synthetic phase-field benchmark, the elastic-basis method
uses half the fracture trajectories without materially degrading parent-level
damage-rollout accuracy relative to Scratch100.

- Evidence needed: powered locked IID, geometry-OOD, and load-path-OOD ratios;
  five seeds; cost accounting.
- Falsified by: any simultaneous upper bound above `1.05`, physics failure, or
  cost ratio above `0.65`.

#### C2 - Mechanism

The gain comes from a correct intact mechanical factorization rather than an
arbitrary extra field, model size, or optimization artifact.

- Evidence needed: shuffled basis, zero basis, equal-capacity direct model,
  stage-wise error, and elastic-to-crack transition analysis.
- Falsified by: shuffled/zero bases matching the complete method or gains only
  before crack initiation.

#### C3 - Robustness boundary

The method has a characterized, not universal, robustness envelope across
section, material, and load-path regimes.

- Evidence needed: parent-balanced subgroup results, joint-OOD mandatory report,
  failure taxonomy, uncertainty/calibration and convergence analysis.
- Falsified by: severe unreported section collapse, leakage, or a solver error
  floor that explains the apparent gain.

### Closest-neighbor and novelty boundary

Closest neighbors are geometry-aware neural operators, GeoPT-style lifted
geometry pretraining, multi-fidelity residual operators, and neural operators
for brittle-fracture coupons. The proposed novelty is their mechanics-specific
combination: an independently verified tunnel elastic response basis is treated
as a nonlearned backbone for a causal fracture residual model and tested under
strict geometry/material/load-path identities. The paper will not claim that
load superposition itself is novel.

## Planned manuscript

1. **Introduction**
   - fracture-trajectory label bottleneck;
   - why intact and damaged regimes should not be learned identically;
   - contribution and explicit scope.
2. **Related work**
   - general-geometry neural operators;
   - physics and geometry pretraining;
   - multi-fidelity mechanics;
   - fracture surrogates and rockburst prediction.
3. **Problem formulation**
   - phase-field state, geometry/material/load path, rollout task;
   - parent identities and allowed claim.
4. **Verified C-fracture benchmark**
   - formulation, solver validation, schema, split, convergence, QC.
5. **Elastic-basis residual damage operator**
   - three-axis basis;
   - causal state transition and irreversibility;
   - losses and reconstruction.
6. **Experimental protocol**
   - baselines, label budgets, locked partitions, metrics, statistics, cost.
7. **Main results**
   - label-efficiency comparison and OOD table;
   - physical validity and cost.
8. **Analysis**
   - mechanism, sensitivity, stage, subgroup, failure and uncertainty analyses.
9. **Limitations and implications**
   - synthetic 2D phase-field scope;
   - path toward specimen and field calibration.
10. **Conclusion**

## Reviewer-facing analysis plan

| ID | Analysis | Reviewer question | Main failure interpretation |
|---|---|---|---|
| A1 | equal-capacity basis ablations | Is the basis, rather than capacity, causal? | downgrade C2 |
| A2 | label-efficiency curve 25/50/75/100% | Is 50% cherry-picked? | narrow or remove C1 |
| A3 | error by intact/initiation/localization/post-peak stage | Where does the method help? | if only intact, method misses fracture |
| A4 | section/material/load-path subgroup | Does one easy family dominate? | characterize or downgrade C3 |
| A5 | mesh/time-step/length-scale sensitivity | Is gain below solver uncertainty? | `ABSTAIN_INVALID` if unresolved |
| A6 | crack topology and energy analysis | Does low field error preserve physics? | restrict to field emulation |
| A7 | cost, memory and storage Pareto | Is the data-saving claim real end-to-end? | remove efficiency claim |
| A8 | worst-case/failure taxonomy and calibration | Are failures detectable? | add operational limitation |

## Likely reviewer objections

1. **The load basis is just linear superposition.** Answer: agree; present it as
   a verified lemma and test novelty only in fracture-label allocation.
2. **The phase-field simulator is not real rockburst.** Answer: constrain the
   title/claims to quasi-static synthetic brittle fracture and add independent
   benchmark/solver or specimen evidence.
3. **The method sees extra information.** Answer: equal-capacity baselines,
   deterministic basis cost accounting, shuffled/zero controls, and identical
   optimization budgets.
4. **Mesh error is comparable to model error.** Answer: fine-ultrafine audits,
   convergence gates, solver uncertainty display, and invalidation if the model
   gain is below the numerical floor.
5. **Geometry split leakage is easy.** Answer: parent identity includes exact
   boundary, defect/material field, mesh family and load path; all time steps
   and fidelities remain in one split.

## Evidence view

Existing evidence that may be cited in Methods/Appendix:

- v0.5 load-basis confirmation: exact intact-response lemma;
- v0.3 formal run: prior negative multi-fidelity baseline and split/audit design;
- B-elastic/Kirsch validation: elastic solver verification after a clean rerun;
- v0.5/v0.5.1 stress recovery: negative design history, likely appendix only.

Missing evidence before prose may be finalized:

- all C-fracture solver and schema results;
- development margin and power analysis;
- formal checkpoints and locked results;
- figures, external benchmark, citations and reproducibility bundle.
