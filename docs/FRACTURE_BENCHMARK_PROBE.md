# SENT/SENS development adapter and probe

`fracture_benchmark.py` is the fail-closed bridge between the frozen v1
protocol, the explicit-slit Gmsh mesh, and the generic prescribed-displacement
BVP. It is development infrastructure, not a Miehe reproduction and not a
Phase-1 readiness result.

The adapter freezes these conventions:

- coordinates are `[y,z]`, where the top is `y=1` and bottom is `y=0`;
- node-major displacement DOFs are even `u_y` and odd `u_z`;
- the bottom has both components fixed and the top has both components
  prescribed; SENT drives top `u_y`, while SENS drives top `u_z`;
- protocol, plan, topology, and ordered BVP-mesh hashes must agree before and
  after a probe;
- the mesh tier, target sizes, notch band, and propagation corridor are read
  from the frozen config and compared against the generated plan.

The default CLI path is validate-only and never calls a solver:

```powershell
$probeHead = git rev-parse HEAD
python scripts/run_fracture_benchmark_probe.py --benchmark sent --validate-only `
  --expected-project-head $probeHead
```

Every invocation requires the full expected commit. Before mesh generation the
runner requires `HEAD == @{upstream} == --expected-project-head` and a fully
clean worktree. This makes an unpushed commit, staged edit, unstaged edit, or
untracked file a hard preflight failure.

The implemented probe holds `d=0` and solves only a few explicitly supplied
displacement states. It requires an approval flag and writes canonical JSON
atomically:

```powershell
python scripts/run_fracture_benchmark_probe.py --benchmark sent `
  --run-intact-probe --approved-development-probe `
  --expected-project-head $probeHead `
  --u-mm 0 --u-mm 0.000001 `
  --output artifacts/development/sent-intact-probe-<unique-run-id>
```

`--output` is a unique run directory, not a JSON filename. The runner reserves
the leaf with `exist_ok=False`; it never overwrites or resumes an existing or
partially written leaf. It opens both files exclusively and emits:

- `result.json`, containing the labelled probe result, UTC start/completion,
  default solver controls, selected package versions, platform and thread
  settings, and a host-path-free reconstructed command;
- `artifact_manifest.json`, containing the SHA-256 and byte count of
  `result.json` plus the pushed project/source identity. The CLI reports the
  SHA-256 of both `result.json` and `artifact_manifest.json`; a manifest cannot
  self-embed its own hash without a circular definition.

The conservative source closure is every sorted top-level
`src/tunnelgeopt/*.py` file (including `__init__.py`), this runner, the actual
config passed to it, `pyproject.toml`, `.gitignore`, `.gitattributes`, and any
root `*.lock`/`pylock.toml`.
Every member must be Git tracked. After solving, the runner rechecks
`HEAD == @{upstream} == expected` and recomputes the exact closure before it
creates either result file. It also rechecks the globally clean worktree: the
empty leaf reserved before mesh generation is invisible to Git until evidence
is published. All numeric artifact fields are recursively rejected if any
value is NaN or infinite.

This checkpoint is intentionally a single-case runner. Its durable result is
labelled `single_case_only=true`, `real_probe_allowed=false`, and
`paper_evidence_eligible=false`, where “real probe” means the paired SENT+SENS
campaign needed for paper-facing timing evidence. Running SENT and SENS as two
independent invocations at the same commit is not supported: the first
artifact makes the worktree dirty and correctly blocks the second preflight.
Before any real paired timing run, add one campaign-level invocation that
reserves one leaf, runs both cases under one snapshot, and completes one
manifest. The present runner may only produce an explicitly approved,
non-authorizing, single-case development diagnostic.

No command in this runner starts a formal trajectory. Medium/fine probes need
an additional explicit flag. The projected formal case time is labelled an
intact fixed-damage lower bound and cannot authorize a medium, fine, or formal
run.

For QC, only constrained residual entries count as support reactions; free-DOF
residual is handled by the separate equilibrium gate. Global force is the
resultant of constrained support reactions plus the complete applied nodal
force, normalized by the sum of their nodal magnitudes with a `1e-15 kN`
floor.
Moment is evaluated about `(y,z)=(0,0)` as
`|sum_i(y_i r_zi-z_i r_yi)| / max(sum_i|y_i r_zi-z_i r_yi|,1e-15 kN mm)`.
Path energy uses the `U=0` initial elastic-energy baseline (the first state is
required to be zero) and trapezoidal reaction work with denominator
`max(|delta E|,|W|,1e-18 kN mm)`. These diagnostic values do not execute or
substitute for the formal protocol thresholds. Since the built-in
probe fixes `d=0`, its damage-component status is explicitly not applicable;
absence of a tip-seeded `d>=0.5` component is never recorded as a benchmark
pass.
