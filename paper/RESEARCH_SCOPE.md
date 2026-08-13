# Stage 1 research scope

## Working title

**Elastic-Basis-Conditioned Neural Operators for Data-Efficient Quasi-Static
Brittle-Fracture Prediction around Hard-Rock Tunnels**

The title is provisional. It must be weakened if the independent fracture
experiment does not pass.

## Background and gap

Geometry-aware neural operators and geometry pretraining can reduce reliance on
expensive simulation labels, but they generally ask a network to relearn both
the inexpensive intact response and the expensive nonlinear failure response.
For homogeneous small-strain tunnel elasticity, the present codebase has
independently confirmed that three canonical far-field load axes span the full
in-plane response of each fixed system to machine precision. That property is
not a paper novelty by itself, but it permits a sharper learning question:
whether exact intact mechanics can be separated from the scarce trajectory
labels needed for brittle damage.

The geomechanics literature on rockburst prediction is dominated by tabular
classification and monitoring signals, while operator-learning work on fracture
usually studies coupons or generic domains. The missing controlled experiment is
therefore not another generic geometry encoder. It is a tunnel-specific test of
physics factorization under geometry, material, and loading-path shifts.

## Research questions

### Primary RQ

Can an elastic-basis-conditioned causal damage operator trained with 50% of the
high-fidelity fracture trajectories match an equal-architecture end-to-end
operator trained with 100%, on unseen tunnel geometries and loading paths?

### Secondary RQs

1. Does a correct elastic basis improve rollout accuracy beyond capacity,
   optimization, or an arbitrary extra input field?
2. Where does factorization help: intact loading, crack initiation, localization,
   or post-peak propagation?
3. Does the label saving remain after accounting for three elastic solves per
   geometry and all data-generation core-hours?
4. Which geometry, material, and loading regimes break the factorization?

## Scope

Included in the first paper:

- two-dimensional plane strain;
- quasi-static AT2-style brittle phase-field fracture with tensile/compressive
  split and damage irreversibility;
- homogeneous or weakly heterogeneous isotropic material fields;
- circle, horseshoe, and straight-wall-arch section families with continuous
  shape parameters;
- multiple in-situ stress ratios, orientations, and monotone/non-proportional
  quasi-static unloading paths;
- damage, stress, displacement, load-response, and fracture-energy trajectories;
- synthetic solver truth with explicit convergence and physical QC.

Excluded from the first paper:

- dynamic ejection, fragment contact, plasticity, explicit joints, 3D faces,
  excavation advance, AE or microseismic waveforms;
- laboratory-to-field or specimen-to-tunnel transfer unless an independently
  licensed dataset is later added and passes a separate protocol;
- engineering rockburst warning or probability calibration.

## Method hypothesis

For geometry `g` and normalized far-field load state `lambda_t`, the intact
stress is computed, not learned:

```text
sigma_elastic(t) = B(g) @ lambda(t)
```

where `B(g)` is the nine-channel response matrix obtained from three canonical
elastic solves. A causal neural operator receives geometry, material, load
history, the current damage state, and the elastic field, then predicts only
the irreversible damage increment and nonlinear stress correction:

```text
(delta_damage, delta_stress) = F_theta(
    geometry, material, load_history, damage_t, sigma_elastic_t
)
damage_(t+1) = clip(max(damage_t, damage_t + softplus(delta_damage)), 0, 1)
sigma_(t+1) = sigma_elastic_(t+1) + delta_stress
```

This mechanism is a hypothesis, not yet a result.

## Stage plan

### Solver and schema pilot

- benchmark intact elasticity and single-edge-notch tension/shear;
- use three mesh levels and at least four elements per regularization length at
  the high-fidelity tier;
- verify monotone damage, nonnegative dissipation, load-step convergence,
  energy balance, and pre-damage agreement with the elastic basis;
- generate 36 development-only trajectories, 12 per section family;
- measure CPU time, memory, storage, nonlinear iterations, and variance before
  freezing a formal sample size.

### Formal experiment candidate

The current minimum candidate is 240 parent trajectories:

| Partition | Parents | Role |
|---|---:|---|
| train | 96 | nested 25/50/100% label subsets |
| dev | 24 | early stopping and fixed threshold selection |
| locked IID | 24 | interpolation |
| locked geometry OOD | 24 | extreme shape parameters |
| locked load-path OOD | 24 | unseen stress ratios/orientations/path changes |
| locked material OOD | 24 | fracture-energy/length-scale shifts |
| locked joint OOD | 24 | mandatory report only |

The pilot must estimate whether this makes the main ratio interval width no
larger than `0.12`. Otherwise the locked sample size is increased before labels
are generated.

## Baselines

1. coarse phase-field solver;
2. Scratch50 and Scratch100 using the same causal operator architecture;
3. direct coarse-conditioned operator at 50% and 100%;
4. geometry-pretrained/GeoPT-inspired initialization at 50%;
5. full elastic-basis residual method at 50%;
6. within-section shuffled-basis negative control at 50%;
7. one parameter- and optimization-matched public strong operator baseline,
   selected before locked generation.

## Metrics and statistics

Primary metric: parent-level balanced damage-trajectory error, with damage and
non-damage spatial regions each receiving 50% mass and the threshold selected
on train/dev only.

Mandatory secondary metrics:

- crack-initiation load error;
- damage-set Dice/IoU and normalized symmetric crack distance;
- peak/post-peak load-response curve error;
- cumulative fracture energy and per-step dissipation error;
- near-field stress tensor RelL2;
- rollout error by stage;
- `0 <= d <= 1`, irreversibility, nonnegative dissipation, energy, equilibrium,
  and boundary residual violations;
- solver and learned-inference runtime, CPU/GPU hours, memory, and storage.

The statistical unit is the parent trajectory, never a mesh point or time
step. Formal evaluation uses five fixed training seeds and 20,000 paired,
section-stratified bootstrap replicates; simultaneous upper bounds or a frozen
family-wise correction are required for the three primary partitions.

## Falsification and decision rule

The intended positive claim requires all of the following:

- the simultaneous one-sided 95% upper bound of
  `EBR-DNO50/Scratch100 <= 1.05` in IID, geometry OOD, and load-path OOD;
- the upper bound of `EBR-DNO50/Scratch50 <= 0.85`;
- at least four of five training seeds pass the directional criterion;
- no failure of irreversibility, energy, boundary, or stability gates;
- total data-generation core-hours, including the three elastic solves, at
  most `0.65` of the Scratch100 data path.

`GO_METHOD` requires every validity, effect, physics, and cost gate.
`NO_GO` means the protocol is valid but an effect gate fails.
`ABSTAIN_INVALID` means leakage, solver/QC failure, insufficient powered sample,
or evaluation-contract failure. No failed gate may be removed after locked
evaluation.

## Deliverables

- validated C-fracture solver and independent benchmark report;
- versioned trajectory schema and identity/split manifest;
- development pilot and power analysis;
- preregistered formal configuration and access audit;
- reproducible model/checkpoint pipeline;
- 5-10 paper-facing result/analysis groups;
- figures, tables, English manuscript source, references, appendix, and PDF;
- public evidence package with hashes and private-label boundary documented.

The implementation candidate and validation ladder are specified separately in
[`FRACTURE_SOLVER_BLUEPRINT.md`](FRACTURE_SOLVER_BLUEPRINT.md).

## Sources already verified at scope stage

- GeoPT, ICLR 2026 paper and official code;
- Geo-FNO, JMLR 2023;
- GINO, NeurIPS 2023;
- Transolver, ICML 2024;
- PI-GANO, CMAME 2025;
- multi-fidelity DeepONet work in Physical Review Research 2022 and AMSES 2023;
- variational DeepONet for brittle fracture, CMAME 2022;
- recent fracture-surrogate and rockburst-AI reviews.

The related-work matrix will retain publication status and primary URLs. No
unverified arXiv item will be described as peer reviewed.
