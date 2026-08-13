# Traction-preserving recovery redesign (v0.5.1 development)

This is a post-hoc development redesign created **after** v0.5 showed a strong
near-field improvement but worse wall-offset traction and resultant metrics.
It deliberately reuses the exact same 15 already-seen v0.3 cases. Therefore it
is not a confirmation, a new test, or evidence of independent generalization.

## Structural change

Away from wall-offset queries, v0.5.1 retains the original recovered stress.
At a wall-offset query with unit normal `n` and tangent `t`, it applies

```text
S_bc = S_raw + (t^T (S_recovered - S_raw) t) t t^T.
```

Consequently `(S_bc - S_raw) n = 0` up to floating-point roundoff. The
candidate retains the recovered tangential-tangential stress but exactly
preserves the raw coarse traction vector. This is a geometry-based projection;
it does not read a fine or ultrafine label and does not claim that the offset
query is the exact traction-free boundary.

The implementation checks the maximum traction increment against `1e-12` for
every case. It also reports wall-offset traction discrepancy, resultant
discrepancy, and full stress tensor relative L2 against both fine and
ultrafine. Because the primary near-field weights exclude wall-offset points,
the near-field result must be numerically identical to unconstrained recovery.

## Frozen routing rule

`READY_FOR_NEW_CONFIRMATORY_PREREGISTRATION` requires all of the following:

- all solver, mesh, identity, and query QC passes;
- the overall and each-section ultrafine near-field development ratios pass;
- the traction-preservation residual is at most `1e-12` in every case;
- wall traction, resultant, and full-stress center ratios do not worsen raw
  coarse against either fine or ultrafine;
- the near-field metric is identical to unconstrained recovery.

Even a READY route means only that a genuinely new, versioned, salted and
preregistered unseen experiment is worth designing. `effect_claim_allowed`
remains false.

## Run

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_stress_recovery_boundary_development.py --validate-only
.\.venv-gpu\Scripts\python.exe scripts\run_stress_recovery_boundary_development.py
```

Outputs are isolated under
`artifacts/development/stress-recovery-boundary-v0.5.1-dev/`. The runner hashes
the v0.5 predecessor tree before and after execution and aborts if any original
artifact changes.
