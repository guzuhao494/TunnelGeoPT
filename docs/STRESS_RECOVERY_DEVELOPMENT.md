# Stress-recovery development diagnostic (v0.5)

This campaign tests one fixed, deterministic post-processing operator before
spending new unseen-case budget. It reconstructs the complete v0.3 generation
plan, declares all 705 v0.3 identities **seen**, and selects one case by a
metadata-only hash from each of 15 partition-by-section cells.

The selected cells are the three section families crossed with `train_id`,
`dev_id`, former `locked_iid`, former `locked_geometry_ood`, and former
`locked_load_ood`. The word `locked` in those source partition names is only a
historical v0.3 name. No identity or fine/ultrafine value used by this campaign
is an independent locked test.

## Frozen operator

The operator is
`src/tunnelgeopt/stress_recovery.py::recover_stress_at_queries` with its default
rank and barycentric tolerances. It performs geometry-weighted affine recovery
from incident element stresses to mesh nodes and then P1 barycentric
interpolation to the common physical query points. Its parameters are not
tuned on campaign results, and its mapping is linear in element stress.

For every selected case, the runner rebuilds the original geometry and query
grid and independently solves the original coarse, fine, and ultrafine mesh
tiers. It reports:

- near-field area-weighted tensor Frobenius relative L2 for raw coarse and
  recovered coarse against ultrafine (primary development diagnostic);
- the same two errors against fine;
- fine-to-ultrafine discrepancy;
- wall-offset traction and resultant discrepancies against both references;
- exact case IDs, query hashes, source hashes, mesh/solver QC, and runtime.

The center ratio is the mean recovered error divided by the mean raw error.
Section and partition centers are also reported. Exploratory thresholds route
the next engineering decision only; they cannot authorize an effect claim.

## Run

Validate the frozen selection without solving:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_stress_recovery_development.py --validate-only
```

Run all 15 real three-tier cases:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_stress_recovery_development.py
```

Outputs are written under
`artifacts/development/stress-recovery-v0.5-dev/`. The per-case JSONL is the
raw metric record; `summary.json` contains aggregate routing, not a formal
scientific effect decision.

## Claim boundary

This campaign can support only a statement about observed behavior of one
fixed stress-recovery operator on 15 already-seen synthetic, homogeneous,
two-dimensional, plane-strain linear-elastic cases. It cannot support a formal
effect, independent generalization, fracture, damage, rockburst, 3D dynamics,
micro-to-field transfer, field prediction, or engineering-truth claim.
