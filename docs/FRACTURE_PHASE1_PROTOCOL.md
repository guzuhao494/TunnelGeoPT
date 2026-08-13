# Fracture Phase-1 development protocol

Status: frozen v3 development-only contract. A local P1 AT2 kernel and an
audited P1-P4 load-schedule adapter exist for isolated development, but the
adapter is not yet integrated into the solver and trajectory labels do not yet
exist. Passing this protocol is not evidence of field rockburst prediction.

The machine-readable source of truth is
[`configs/fracture_phase1_pilot.json`](../configs/fracture_phase1_pilot.json).
`tunnelgeopt.fracture_validation` rejects a changed or legacy-shaped config
before case generation. This document explains that contract; it does not
authorize a formal experiment.

## Fixed scientific scope

The pilot is homogeneous, isotropic, two-dimensional plane strain and
quasi-static. It uses AT2 brittle phase-field fracture with a 3-D spectral
strain split evaluated at the plane-strain state `epsilon_xx = 0`, P1
displacement, P1 damage, explicit energy/history quadrature, and constrained
damage irreversibility. The convention is `d=0` intact and `d=1` fully damaged.

The total-field boundary formulation is mandatory:

```text
u(x,s) = epsilon_infinity(s) x + w(x,s)
t_wall(x,s) = (1 - r_wall(x,s)) sigma_infinity(s) n
```

For P1, `r_wall=lambda=s` is uniform. P2 and P3 change the far-field stress
tensor along `s`; P4 uses the frozen crown/sidewall/invert release field. A
fixed `sigma_infinity` plus one uniform wall scalar is therefore only the P1
debugging subset, not an implementation of this protocol.

The residual stiffness is `k=1e-8` in
`g(d)=(1-d)^2+k`. An unconstrained damage solve followed by clipping is not an
allowed substitute for a bound-constrained solve.

Explicitly excluded are Stress-Lift or lifted-dynamics inputs, 3-D faces,
excavation advance, inertia/ejection, joints, geometric roughness, random or
heterogeneous material fields, AE/microseismic waveforms, fragment contact,
plasticity, and field or engineering-warning claims. The validator also rejects
legacy config keys for these concepts instead of silently ignoring them.

## Frozen 36-trajectory cross-product

There is exactly one deterministic trajectory in every cell of:

```text
3 canonical section families x 3 coupled material levels x 4 load histories
= 36 trajectories
```

The section order is `circle`, `horseshoe`, `straight_wall_arch`. Each uses the
canonical parameters written in the config, 256 boundary points, no boundary
perturbation, and characteristic radius `R=1`. Every case and mesh tier uses the
same centered square outer domain `[-8R,8R] x [-8R,8R]`; the outer boundary is
not rescaled to the section.

Case identities are
`fp1-{section_family}-{material_level_id}-{load_path_id}`. Mesh tier, retries,
and solver attempts are deliberately absent from the physical identity. The
canonical enumeration order is section, then material, then path.

### Dimensionless material coupling

`E/UCS=500` and `nu=0.25` are fixed. The three levels vary regularization
length and toughness together:

| Level | `ell/R` | `Gc/(UCS R)` | `Gc/(UCS ell)` |
|---|---:|---:|---:|
| m1 | 0.04 | 0.000008 | 0.0002 |
| m2 | 0.06 | 0.000012 | 0.0002 |
| m3 | 0.08 | 0.000016 | 0.0002 |

Thus the pilot is a regularization-sensitivity check at one fixed
dimensionless AT2 homogeneous strength scale, not three arbitrary independent
material draws. The declared scale is

```text
sigma_peak/UCS = sqrt[(27/256) (E/UCS) (Gc/(UCS ell))]
               = 0.1026979795
```

This value is not called a crack-initiation threshold: AT2 does not provide a
strict finite onset threshold in that sense.

### Four actual load histories

Every path has five explicit control knots in `s in [0,1]`. Coordinates are
always `(y,z)`, with `y` vertical and `z` horizontal. A principal angle is
measured from positive `y` toward positive `z`. The stored controls
`sigma1/UCS`, `sigma3/sigma1`, angle, and releases are each interpolated
piecewise-linearly first; only then are the positive compression magnitudes
converted to the solver's tension-positive tensor. For angle `alpha`,

```text
e1 = (cos alpha, sin alpha)
e3 = (-sin alpha, cos alpha)
sigma_infinity = -UCS [(sigma1/UCS) e1 tensor e1
                       + (sigma1/UCS)(sigma3/sigma1) e3 tensor e3]
```

The runtime `ucs_scale` passed to the adapter must be explicit, finite, and
strictly positive; the adapter does not guess stress units.

| Path | Frozen history |
|---|---|
| p1 | Fixed `sigma1/UCS=0.45`, ratio `0.70`, angle `0 deg`; uniform release `0, .25, .50, .75, 1` |
| p2 | `sigma1/UCS: .45 -> .65`, ratio `.80 -> .35`; non-proportional uniform release `0, .15, .40, .70, 1` |
| p3 | Fixed `sigma1/UCS=.55`, ratio `.55`; angle `-30 -> 0 -> 30 deg` during uniform release |
| p4 | Fixed `sigma1/UCS=.55`, ratio `.45`, angle `15 deg`; crown, then sidewalls, then invert are released |

P4 is evaluated on the actual wall facets of the generated mesh, not on the
input boundary vertices. The wall centroid is the length-weighted perimeter
centroid of those facets. For each facet midpoint `(y_f,z_f)`,
`theta=atan2(z_f-c_z,y_f-c_y) mod 360 deg`. Crown, right sidewall, invert, and
left sidewall occupy `[-45,45]`, `(45,135)`, `[135,225]`, and `(225,315)`
degrees respectively away from transitions. Every zone boundary has one
linear convex transition of total width exactly `5 deg` (`+-2.5 deg`): the two
neighboring weights vary from `(1,0)` to `(0,1)` and all other weights are
zero. Facet identifiers remain in the mesh marker order, so every returned
release stays aligned with its source facet. Reordering facets or reversing
edge endpoints cannot change the physical result. All zone releases are
monotone and reach one at `s=1`.

Required output states are `s=0, 0.025, ..., 1` (41 states). Adaptive accepted
substeps may be inserted between them and must also be stored. Six step retries
with a factor of one half are allowed per required interval; exhausting the
budget invalidates the trajectory.

## Mesh and deterministic ultrafine audits

The potential fracture region is the rock within `2R` wall distance. Fine
meshes for all 36 cases require `h/ell <= 0.25` there. The ultrafine audit tier
requires `h/ell <= 0.125`. A trajectory is invalid if damage reaches the outer
edge of this refined region, because a crack may not be judged converged after
leaving the resolved band.

Twelve cases are selected before solver results. Each section-by-path cell has
one audit, with zero-based material index
`(section_index + path_index) mod 3`:

| Section | p1 | p2 | p3 | p4 |
|---|---|---|---|---|
| circle | m1 | m2 | m3 | m1 |
| horseshoe | m2 | m3 | m1 | m2 |
| straight_wall_arch | m3 | m1 | m2 | m3 |

This covers all 12 section-path combinations and assigns four audits to each
material level. Audit selection cannot change after a convergence failure.

## Solver and trajectory gates

The staggered displacement/history/bound-constrained-damage solve uses at most
100 staggered and 100 active-set iterations. Displacement, damage, and energy
increment tolerances are `1e-8`; equilibrium and KKT/complementarity relative
residuals are `1e-6`.

Every trajectory must store all required output states and pass all of these
gates:

- zero non-finite values;
- equilibrium and KKT/complementarity residuals at most `1e-6`;
- damage irreversibility, damage range, and history monotonicity violations at
  most `1e-10`;
- relative path-energy imbalance at most `5%`;
- incremental dissipation no less than `-1e-10 UCS R^2` (roundoff allowance);
- damage at the refined-region outer edge at most `1e-4`;
- all stored steps converged, retry budget not exhausted, complete attempt and
  failure ledger, and zero replacement attempts.

Each of the 12 preselected audits additionally requires both peak reaction and
total fracture-energy fine-to-ultrafine changes to be at most `5%`.

Before launching the 36 trajectories, the element/spectral/KKT tests, intact
elastic regression, three-grid single-edge-notch tension and shear benchmarks,
and both MOOSE-related gates listed in the config must pass:

1. the pinned official MOOSE `crack2d_iso` reference self-test verifies only
   that the external reference environment executes its own regression; and
2. a separate local-vs-MOOSE same-problem comparison must cover intact
   elasticity and pre-notched tension/shear with matched parameters, boundary
   conditions, and output definitions.

The first gate cannot satisfy the second. These prerequisites are not waived by
trajectory-level QC.

The pilot requires a 100% pass fraction. A failed identity remains in the
ledger and is excluded from successful labels; it is never replaced by a newly
sampled case. A failed gate means the 36-case protocol did not pass, rather than
that the denominator has become smaller.

## Validation API

The public module-level API is intentionally independent of the future solver
and trajectory schema:

```python
from tunnelgeopt.fracture_validation import (
    enumerate_fracture_phase1_cases,
    enumerate_ultrafine_audits,
    evaluate_trajectory_qc,
    load_fracture_phase1_config,
    validate_fracture_phase1_config,
)

config = load_fracture_phase1_config()
cases = enumerate_fracture_phase1_cases(config)  # 36 frozen identities
audits = enumerate_ultrafine_audits(config)  # 12 preselected identities
result = evaluate_trajectory_qc(config, cases[0].case_id, trajectory_metrics)
```

The v3 load adapter is a separate, solver-independent API:

```python
from tunnelgeopt.fracture_loading import compile_phase1_load_schedule

schedule = compile_phase1_load_schedule(
    config,
    path_id="p4",
    ucs_scale=ucs_in_consistent_stress_units,
    mesh=tunnel_mesh,
)
state = schedule.state_at(0.375)  # any finite s in [0,1], not only output knots

sigma_yz = state.farfield_stress_tension_positive_yz
facet_ids = state.wall_facet_ids
facet_release = state.wall_release
```

`FractureLoadState` is frozen and owns read-only copies of its numeric arrays.
The adapter computes controls, tensors, zone weights, and facet-aligned release
values only. It does not assemble tractions, advance damage, integrate work, or
close any solver-validation prerequisite.

`evaluate_trajectory_qc` consumes reported aggregate diagnostics. It does not
derive those values from raw fields, execute a solver, or validate a trajectory
file. Audit identities require both refinement metrics. All results report
`replacement_allowed=False`.
