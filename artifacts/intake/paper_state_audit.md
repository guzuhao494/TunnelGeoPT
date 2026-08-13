# Paper-state intake audit

Date: 2026-08-14

## Intake classification

The repository is `baseline_ready + analysis_ready`, but it is not
`main_result_ready` for a fracture paper and not `paper_ready`.

The existing evidence is unusually well audited for a research prototype. It
contains a verified elastic solver, a formal multi-fidelity experiment, several
negative development results, and a narrow independent confirmation of linear
load-axis factorization. None of these assets contains a real damage or fracture
trajectory.

## Trust-ranked assets

| Asset | Trust | Paper role | Reusable fact | Boundary |
|---|---|---|---|---|
| v0.3 formal multi-fidelity run | trusted | comparator and negative evidence | 195 parents, 705 valid cases, 35 checkpoints, 144 fine-ultrafine audits, final `ABSTAIN` | no 50% label-efficiency claim; all identities are permanently seen |
| v0.5 independent load-basis confirmation | trusted | method lemma and elastic backbone | 24/24 solves; 15 held-out reconstructions; median/max stress RelL2 `4.886e-15/5.882e-15` | fixed geometry/material/mesh/query linear systems only; near-tautological linear superposition |
| v0.3 fine-ultrafine convergence | trusted | mesh-tier rationale | 24 cases; median/p95 `2.116%/3.036%` | development mesh evidence, not a model result |
| B-elastic solver validation | usable with verification | methods/appendix | patch test and Kirsch refinement are physically coherent | five text artifact hashes no longer match raw bytes; clean rerun or canonical re-publication needed before final paper |
| v0.5 stress recovery | trusted as development evidence | negative result | near-field error `3.1617% -> 1.3953%` on 15 seen cases | wall traction/resultant worsened; STOP |
| v0.5.1 traction-preserving recovery | trusted as development evidence | negative result | traction increment preserved to `2.355e-16` | full wall-stress error still worsened; post-hoc seen-data STOP |
| v0.4 structured residual | reference only | internal history | a plausible mechanism was screened out | conversation-only, unreplayable, capacity/loss/input confounds |
| GeoPT source paper | trusted as an external idea source | related work and baseline motivation | lifted geometry pretraining improved industrial simulation data efficiency | steady-state pretraining; no tunnel fracture, scale transfer, or rockburst evidence |

## Claims that are currently allowed

1. The project has a validated two-dimensional plane-strain elastic data layer
   with strict identity and leakage controls.
2. In fixed linear systems, three independent in-plane far-field load axes
   reconstruct displacement and stress responses to machine precision.
3. A generic coarse-to-fine residual learner did not establish the
   preregistered 50% label-efficiency claim; the formal result was `ABSTAIN`.
4. Existing stress-recovery candidates improved interior accuracy but failed
   predeclared wall diagnostics.

## Claims that are not currently allowed

- learned rock damage, crack initiation, crack growth, fragmentation, or
  fracture dissipation;
- rockburst dynamics, AE or microseismic waveforms, or engineering warning;
- micro/specimen-to-tunnel or simulation-to-field transfer;
- geometry, material, mesh, solver, or field-truth generalization;
- public release of a complete trainable high-fidelity dataset;
- a positive high-fidelity label-efficiency result.

## Missing paper-critical evidence

1. A genuine nonlinear fracture generator and an independent physical
   validation chain.
2. A separate C-fracture schema with time-indexed damage, stress, displacement,
   fracture energy, and convergence diagnostics.
3. A nontrivial learned rollout task with equal-budget baselines and negative
   controls.
4. Powered, parent-level, unseen geometry/material/load-path evaluation.
5. A cost-versus-accuracy study that includes the three elastic-basis solves.
6. Preferably, a second solver or laboratory specimen curve/crack image as an
   external anchor.

## Intake decision

Do not draft a result-complete paper from the current elastic evidence. Reuse
the confirmed three-load basis as a fixed elastic backbone and move to a small,
fully seen C-fracture solver pilot. The next durable decision is whether that
pilot is physically valid and whether an elastic-basis-conditioned damage
operator has enough development margin to justify new locked data.
