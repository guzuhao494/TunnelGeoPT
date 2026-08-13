# Experiment records

This directory stores durable run contracts, manifests, per-case metrics, and
decisions. The initial `smoke-v0.1.0` record validates executable paths and
schemas only. The v0.2 records add two verified milestones:

- `b-elastic-v0.2.0`: GO for homogeneous isotropic small-strain plane-strain
  solver-emulation data only.
- `analytic-transfer-v0.2.0`: preregistered NO-GO for the current circle-only
  Stress-Lift label-efficiency hypothesis.

Generated `.npz` fields and `.pt` checkpoints remain local by default to avoid
growing Git history. Their content hashes and semantic metadata are retained in
the committed manifests, and the scripts regenerate them. The committed JSON
records contain the configuration, case-level split, seeds, environment,
per-case metrics, access audit, and final decision. See
`docs/MILESTONE_V0.2.md` for the claim boundary.
