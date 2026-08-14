# SENT/SENS three-grid development protocol

Status: **protocol frozen; real-Gmsh mesh contract verified; fracture solver not
run**. The machine-readable source of truth is
[`configs/fracture_sent_sens_v1.json`](../configs/fracture_sent_sens_v1.json).
All six SENT/SENS x coarse/medium/fine mesh contracts have been generated with
the real Gmsh path and audited for their frozen topology, physical labels, and
size contract. This verifies the mesh layer only: no coupled-fracture
trajectory, reaction curve, fracture energy, crack path, or three-grid
benchmark result exists. The only valid benchmark decision remains
`ABSTAIN_NOT_RUN`. The dedicated real-Gmsh mesh tests and their narrower
evidence boundary are documented in
[`FRACTURE_BENCHMARK_MESH.md`](FRACTURE_BENCHMARK_MESH.md).

This protocol is a narrow prerequisite for attempting the 36-case tunnel
Phase-1 pilot. It is not an exact reproduction of Miehe et al., an experimental
validation, a tunnel-fracture result, or evidence for dynamic or field
rockburst prediction.

## Evidence layers

The protocol keeps three evidence layers separate.

### 1. Primary-source facts

The benchmark family is based on:

> C. Miehe, M. Hofacker and F. Welschinger, *A phase field model for
> rate-independent crack propagation: Robust algorithmic implementation based
> on operator splits*, CMAME 199 (2010), 2765-2778.

- DOI: <https://doi.org/10.1016/j.cma.2010.04.011>
- Publisher record:
  <https://www.sciencedirect.com/science/article/pii/S0045782510001283>

The facts used from that source are the unit-square coupon with a horizontal
left-edge notch ending at the centre, the tension and shear loading families,
`lambda=121.15 kN/mm^2`, `mu=80.77 kN/mm^2`,
`Gc=2.7e-3 kN/mm`, the reported `ell=0.015/0.0075 mm` studies, the
rate-independent `eta=0` results, and the reported load-increment schedules.
The present three-grid protocol selects `ell=0.015 mm` only.

### 2. TunnelGeoPT supplemental conventions

The source paper does not uniquely freeze every implementation decision needed
for this repository. TunnelGeoPT therefore declares, rather than hides, these
supplemental choices:

- repository coordinates are `(y,z)`, with `y` vertical and `z` horizontal;
- P1 displacement and P1 damage are evaluated on first-order triangles under
  plane strain;
- the notch is an explicit zero-width double-face slit, not a row of seeded
  damaged material;
- rock starts with `d0=0` and `H0=0`;
- residual stiffness is `k=1e-8`;
- the three mesh tiers, refinement corridors, solver residuals, topology tests,
  convergence metrics, and compute gate are repository conventions.

These choices are why a successful run may be called a **Miehe-type
development benchmark**, but never an exact Miehe reproduction.

Public implementations may be cross-read for implementation ideas, but their
arrays and curves are not accepted as gold data. In particular, the protocol
records the [PhaseFieldX example](https://phasefieldx.readthedocs.io/en/latest/auto_examples/PhaseFieldFracture/plot_1712.html)
and the [PhAST setup guide](https://github.com/CEMS-Lab/PhAST/blob/main/docs/user_guide/setup_problems.md)
only in that limited role.

### 3. Digitized sanity envelopes

The following inclusive windows were manually read from primary-source plots:

| Benchmark | peak reaction | displacement at peak |
|---|---:|---:|
| SENT | `0.60-0.82 kN` | `0.0050-0.0061 mm` |
| SENS | `0.44-0.62 kN` | `0.0075-0.0115 mm` |

They are deliberately broad sanity screens. They are not reference arrays,
they cannot replace three-grid convergence, and they must not be used to tune
a run into the plotted band. A mismatch is `ABSTAIN_REFERENCE_AMBIGUITY`, not
permission to edit the window after seeing results.

## Geometry and coordinate contract

The material domain is

```text
y in [0,1] mm    (vertical)
z in [0,1] mm    (horizontal)
```

The horizontal slit is

```text
y = 0.5 mm
z in [0,0.5] mm
```

Upper- and lower-face nodes and facets are distinct along the open slit and
share only the tip at `(y,z)=(0.5,0.5) mm`. Each open face facet has exactly one
adjacent rock element. No triangle may bridge the slit. Treating `z` as the
vertical coordinate would silently turn the benchmark by 90 degrees and is a
contract failure.

Required boundary labels are `bottom`, `top`, `left_upper`, `left_lower`,
`right`, `notch_upper`, `notch_lower`, and `notch_tip`. Labels must be nonempty
and disjoint where the geometry requires it.

## Material and fracture model

Both benchmarks use the same homogeneous material:

| Quantity | Frozen value |
|---|---:|
| Lame lambda | `121.15 kN/mm^2` |
| shear modulus | `80.77 kN/mm^2` |
| critical fracture energy | `2.7e-3 kN/mm` |
| regularization length | `0.015 mm` |
| viscosity | `0` |
| residual stiffness | `1e-8` |

The local model is quasi-static plane-strain AT2 with `d=0` intact and `d=1`
fully damaged, `g(d)=(1-d)^2+k`, the repository's 3-D spectral strain split
evaluated at `epsilon_xx=0`, quadrature history
`H_n=max(H_previous,psi_plus_current)`, and bound-constrained irreversible
damage. An unconstrained solve followed by clipping does not satisfy the
protocol.

## Boundary conditions and load grids

For both benchmarks, the bottom boundary `y=0` has `u_y=u_z=0`. Lateral and
notch faces are traction-free. The entire top `y=1` is displacement controlled.

### SENT

```text
top: u_y=U, u_z=0
reaction: positive magnitude of the summed top y reaction
```

The required states are:

1. `U=0 ... 0.005 mm` in `1e-5 mm` increments (500 increments);
2. then `U=0.005001 ... 0.0065 mm` in `1e-6 mm` increments (1500 increments).

The transition state at `0.005 mm` is stored once, giving 2001 states.

### SENS

```text
top: u_y=0, u_z=U
reaction: positive magnitude of the summed top z reaction
```

`U=0 ... 0.015 mm` in `1e-5 mm` increments gives 1501 states.
Every prescribed state must be stored. Adaptive retries may add states but may
not remove or replace prescribed states.

## Three mesh tiers

The selected `ell=0.015 mm` gives:

| Tier | target `h/ell` in corridor | `h_target` | bulk target `min(4h,0.04 mm)` |
|---|---:|---:|---:|
| coarse | `1/2` | `0.0075 mm` | `0.030 mm` |
| medium | `1/4` | `0.00375 mm` | `0.015 mm` |
| fine | `1/8` | `0.001875 mm` | `0.0075 mm` |

The SENT refinement corridor is a `0.10 mm` half-width buffer around the
horizontal segment from the notch tip to the right boundary. The SENS corridor
uses a `0.15 mm` half-width around the tip-to-lower-right segment from
`[0.5,0.5]` to `[0,1]` in `[y,z]` order. Both benchmarks also refine points
within `0.05 mm` of the open notch faces and tip. These corridors are supplemental
meshing conventions, not claims about an exact crack trajectory.

Within the tagged corridor, every edge must satisfy
`h_max <= 1.15 h_target`. The connected `d>=0.5` component seeded at the notch
tip must stay inside the corridor. Escape yields `STOP_REMESH_BEFORE_RERUN`;
the result cannot be called a mesh-convergence failure while the crack has left
the resolved region.

Before any solve, each tier must pass all slit adjacency, no-crossing,
physical-label, positive-oriented-area, and corridor-size checks. Any topology
failure is `STOP_INVALID`.

## Per-tier solver and balance gates

Every accepted state of all six cases must satisfy:

- equilibrium relative residual `<=1e-8`;
- KKT/complementarity relative residual `<=1e-8`;
- relative displacement increment `<=1e-8`;
- relative damage increment `<=1e-8`;
- relative potential-energy increment `<=1e-8`;
- damage irreversibility violation `<=1e-12`;
- global force relative imbalance `<=1e-8`;
- global moment relative imbalance `<=1e-8`;
- path-energy relative imbalance `<=5%`;
- zero non-finite values, all prescribed states, a complete attempt ledger,
  and no accepted unconverged state.

The alternate-minimization and active-set iteration caps are both 100. A
failure of topology, completeness, balance, convergence, or ledger integrity
makes the run invalid (`STOP_INVALID`); it cannot be counted as numerical
evidence by dropping failed states.

## Three-grid convergence gates

For SENT and SENS separately, medium-to-fine differences must satisfy all five
gates:

| Metric | Maximum medium-to-fine error |
|---|---:|
| peak reaction relative change | `5%` |
| final fracture-energy relative change | `5%` |
| full reaction-curve relative L2 | `5%` |
| peak-displacement relative change | `5%` |
| symmetric final crack-path Hausdorff distance / `ell` | `0.5` |

Scalar relative changes use the fine value as the denominator with a declared
`1e-12` quantity-unit floor. The reaction-curve error is the trapezoidal L2
ratio on the frozen common `U` grid. If a plateau has several equal global
reaction maxima, `U_peak` is the smallest such displacement.

The crack path is the notch-tip-seeded component of the final continuous P1
`d=0.95` contour. Comparing arbitrary disconnected damaged islands is not
allowed.

Each of the five medium-to-fine errors must also be no greater than its
corresponding coarse-to-medium error. This monotonicity gate prevents a lucky
medium/fine pair from hiding oscillatory mesh behaviour. A valid run that
fails one of these gates is `STOP_NUMERICAL`.

## Compute preflight for the 12 GB laptop GPU environment

The benchmark currently has a CPU finite-element path; the 12 GB RTX 5070 Ti
does not by itself make the fine solve affordable. Compute authorization is
therefore based on measured wall time, not an invented GPU speedup or a guessed
degree-of-freedom exponent.

The coarse subsampled load grids are exactly ten times the formal increments:

- SENT: `0 ... 0.005` by `1e-4`, then to `0.0065` by `1e-5` (201 states);
- SENS: `0 ... 0.015` by `1e-4` (151 states).

At least ten accepted steps must be timed for each benchmark and tier. The
coarse-only stage reports only a coarse formal-step projection. Before any
formal medium or fine trajectory, that same tier must complete an explicit
fixed ten-step timing probe. The only default projection is

```text
projected case hours = formal increment count
                     * measured median accepted-step seconds
                     / 3600
```

The median must come from the same benchmark and tier. No unmeasured
coarse-to-medium DOF exponent is allowed. All raw step times, medians, and the
resulting projections must be recorded before authorization.

If one projected medium case exceeds 12 wall-hours, or the sum for all six
formal cases exceeds 72 wall-hours, the route is
`ABSTAIN_COMPUTE_OPTIMIZE_BEFORE_RERUN`. Optimize and re-freeze before spending
the full compute budget.

## Decision precedence

Decision routing is fail-closed and ordered:

1. `STOP_INVALID`: invalid config, topology, mesh, per-tier QC, balance,
   completeness, or ledger;
2. `ABSTAIN_COMPUTE_OPTIMIZE_BEFORE_RERUN`: the 12/72-hour compute gate fails;
3. `ABSTAIN_REFERENCE_AMBIGUITY`: digitized sanity windows disagree or a
   primary-source figure/definition is too ambiguous for comparison;
4. `STOP_NUMERICAL`: the run is valid, but a three-grid or monotonicity gate
   fails;
5. `READY_FOR_PHASE1_PILOT`: every hard gate and sanity screen passes.

The fifth route is the protocol's limited GO-family outcome. Its only meaning
is that the local development solver may attempt the 36-case Phase-1 pilot. It
is explicitly **not** a paper GO, field GO, hard-rock tunnel validation, or
rockburst claim.

## Validate-only API

```python
from tunnelgeopt.fracture_benchmark_validation import (
    enumerate_fracture_benchmark_cases,
    load_fracture_sent_sens_config,
    prescribed_displacements,
)

config = load_fracture_sent_sens_config()
cases = enumerate_fracture_benchmark_cases(config)  # six identities
sent_U = prescribed_displacements(config, "sent")  # 2001 states
sens_U = prescribed_displacements(config, "sens")  # 1501 states
coarse_probe_U = prescribed_displacements(
    config, "sent", compute_preflight=True
)  # 201 states
```

The validator pins the canonical JSON hash in code. Any nested value, key, or
decision change fails closed and requires a new protocol version. The API does
not import the mesh or fracture solver and cannot accidentally turn validation
of the frozen plan into fracture-solver or numerical-benchmark evidence.
