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
python scripts/run_fracture_benchmark_probe.py --benchmark sent --validate-only
```

The implemented probe holds `d=0` and solves only a few explicitly supplied
displacement states. It requires an approval flag and writes canonical JSON
atomically:

```powershell
python scripts/run_fracture_benchmark_probe.py --benchmark sent `
  --run-intact-probe --approved-development-probe `
  --u-mm 0 --u-mm 0.000001 --output artifacts/development/sent-probe.json
```

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
