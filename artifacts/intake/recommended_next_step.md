# Recommended next step

## Route

Implement and validate a small C-fracture layer before running any new neural
operator campaign.

The proposed scientific question is:

> In synthetic two-dimensional quasi-static brittle fracture around hard-rock
> tunnels, can a neural operator that receives an exact three-load elastic
> response basis and learns only irreversible damage/stress residuals match a
> full-label end-to-end operator while using 50% of the fracture trajectories?

## Why this route

The load basis is exact for the intact linear problem and cheap to generate,
but it cannot describe damage. This creates a clean factorization rather than a
metaphor: deterministic elasticity is handled outside the network; scarce
labels are reserved for path-dependent nonlinear deviations. The hypothesis is
useful only if it survives equal-capacity baselines, shuffled-basis controls,
unseen geometries and load paths, physical rollout gates, and total-cost
accounting.

## Immediate sequence

1. Validate a quasi-static AT2 phase-field solver on intact elasticity,
   single-edge-notch tension/shear, mesh refinement, irreversibility, and energy
   balance.
2. Define a C-fracture schema that cannot be confused with the elastic schema.
3. Generate 36 development trajectories: 12 per section family, balanced over
   material and load-path regimes, with no locked labels.
4. Fit a small `EBR-DNO` and matched Scratch50/Scratch100 models on development
   folds.
5. Stop unless the solver passes all gates and the candidate shows consistent
   margin before any new locked dataset is created.

## Stop conditions before formal data

- fewer than four elements per phase-field length scale at high fidelity;
- fine-ultrafine changes above 5% for critical load or fracture energy;
- nonlinear residual above `1e-6`, energy imbalance above 5%, or damage/
  dissipation violations above 0.1%;
- fracture benchmark path or load curve outside frozen tolerances;
- `EBR-DNO@50%` not better than `Scratch@100%` in development mean, not at least
  15% better than `Scratch@50%`, or physics gates deteriorate.
