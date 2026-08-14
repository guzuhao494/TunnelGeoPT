# Paired coarse intact-probe campaign

`scripts/run_fracture_benchmark_campaign.py` is the immutable paired execution
lane for the first bounded SENT/SENS timing and equilibrium-QC measurement. It
is not a coupled fracture trajectory. Damage remains fixed at `d=0`; therefore
the measured cost is only a lower-bound triage signal.

## Frozen execution contract

- one clean, pushed Git snapshot is captured before the outer run leaf exists;
- the source closure must contain this campaign runner;
- SENT then SENS are generated and solved serially on their real coarse meshes;
- both cases use exactly `U = [0, 1e-5, 2e-5] mm`, re-derived at runtime from
  the frozen formal grids (six fixed-damage equilibrium solves in total);
- CPU thread-control environment variables are set to one before numerical
  packages are lazily imported and recorded with the honest limitation that
  this does not prove the operating-system scheduler used one thread;
- process-lifetime peak RSS is recorded when the platform exposes it, otherwise
  a typed `UNAVAILABLE` record is emitted;
- all five fixed artifact paths are checked individually with `git check-ignore`
  before the outer leaf is created or either solver is called;
- every positive-displacement state must return a finite, strictly positive
  generalized reaction magnitude; the zero state must remain within the
  existing `1e-12 kN` numerical-zero tolerance;
- the outer run leaf remains empty while both results are held in memory;
- one postflight rechecks HEAD/upstream/source/cleanliness, still with an empty
  leaf, before any evidence file is written.

Publication is fail-closed and never resumes an interrupted leaf. The writer
creates, in order:

1. `cases/sent/result.json`
2. `cases/sens/result.json`
3. `implementation_manifest.json`
4. `campaign_result.json`
5. `artifact_manifest.json` (completion marker, linked last)

The final manifest hashes the other four files and is not self-hashed. Every
file is created exclusively, all JSON floats must be finite, and local project
paths are rejected. A reserved leaf without the final manifest is incomplete
and must not be interpreted or reused.

## Claim boundary

Successful completion means only that a real paired coarse intact probe ran on
one shared pushed implementation and passed the QC fields applicable to fixed
damage. The artifacts set `real_paired_probe_completed=true`, but also freeze:

- `timing_and_qc_triage_only=true`;
- `coupled_damage_evolution=false`;
- `formal_fracture_trajectory=false`;
- `authorizes_coupled_fracture_run=false`;
- `authorizes_medium_fine_or_formal_run=false`;
- `paper_effect_evidence=false`.

Thus the campaign cannot support a fracture, benchmark-reproduction, tunnel,
rockburst, or learned-model effect claim.

## Invocation

Run only from a clean branch whose HEAD is already at its upstream:

```powershell
$headSha = git rev-parse HEAD
.venv\Scripts\python.exe scripts\run_fracture_benchmark_campaign.py `
  --output artifacts/development/<new-unique-run-leaf> `
  --expected-project-head $headSha `
  --run-paired-intact-probe `
  --approved-development-probe
```

Choose a new, non-ignored output leaf each time. Never delete or complete a
partially published leaf in place.
