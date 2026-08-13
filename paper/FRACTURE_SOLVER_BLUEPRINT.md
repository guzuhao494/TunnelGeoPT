# Candidate C-fracture solver blueprint

Status: Stage-1 design candidate; not implemented, validated, or authorized for
formal label generation.

## Reuse boundary

The current code can reuse:

- `geometry.py` for the three continuous tunnel-section families;
- `mesh.py` for `TunnelMesh`, first-order triangles, wall/far-field tags and mesh QC;
- `elasticity.py` for plane-strain material utilities and element strain recovery;
- `field_sampling.py` for projection to a fixed query set;
- the residual, normal and tensor-metric patterns in `elastic_validation.py`.

The elastic schema must not be extended with damage fields. The fracture layer
needs independent modules, tests and storage semantics:

```text
src/tunnelgeopt/fracture.py
src/tunnelgeopt/fracture_schema.py
src/tunnelgeopt/fracture_validation.py
scripts/run_fracture_benchmarks.py
tests/test_fracture.py
tests/test_fracture_benchmarks.py
```

## Field and loading convention

Use `d=0` for intact material and `d=1` for fully damaged material. The tunnel
problem must be formulated in the total field, not by degrading the current
incremental excavation energy.

For a prescribed far-field strain, write

```text
u(x) = epsilon_infinity x + w(x).
```

The far field follows the affine displacement. The cavity-support traction is
released along a path parameter `lambda`:

```text
t_wall(lambda) = (1 - lambda) sigma_infinity n,
lambda=0: uniform in-situ state,
lambda=1: fully excavated wall.
```

With damage disabled, the `lambda=1` correction field must regress to the
existing plane-strain excavation solver within a frozen tolerance. Load path P4
will require spatially staged wall release rather than a single scalar factor.

## Candidate AT2 energy

The initial implementation candidate uses P1 displacement and P1 damage with
residual stiffness `k`:

```text
g(d) = (1 - d)^2 + k

Pi(u,d) = integral[
    g(d) psi_plus(epsilon(u)) + psi_minus(epsilon(u))
  + Gc/(2 ell) d^2 + Gc ell/2 |grad d|^2
] dOmega - W_external.
```

The preferred split is a three-dimensional spectral strain split evaluated
under plane strain (`epsilon_xx=0`). This choice requires independent tensor
reconstruction and directional-derivative tests. An easier volumetric/
deviatoric split may be implemented only as a named alternative, not silently
substituted.

At load step `n`, store the tensile history field at integration points:

```text
H_n = max(H_(n-1), psi_plus(epsilon_n)).
```

The AT2 damage weak equation is

```text
integral[(Gc/ell + 2 H_n) d q + Gc ell grad(d).grad(q)] dOmega
  = integral[2 H_n q] dOmega.
```

Irreversibility is a constrained solve, not postprocessing:

```text
d_n >= d_(n-1),  0 <= d_n <= 1.
```

A primal-dual active set or another auditable bound-constrained method must
report KKT/complementarity residuals. Clipping an unconstrained result is not
acceptable evidence.

## Staggered solve

For each accepted wall-release increment:

1. initialize from the previous accepted `u`, `d`, and history;
2. hold damage fixed and solve displacement equilibrium;
3. update tensile history;
4. hold displacement/history fixed and solve constrained damage;
5. test displacement, damage, equilibrium, KKT and energy convergence;
6. accept the state or halve the load increment and retry.

Exhausting the retry budget invalidates the trajectory. A final unconverged
iterate must never be stored as a successful label.

The spectral consistent tangent is the largest implementation risk. A numerical
element tangent may be used first, provided directional derivatives are tested;
an analytic tangent can replace it only after regression equivalence. Energy and
history integration must use explicit quadrature rather than silently treating
one element-center value as a complete integral.

## Minimum result schema

Each accepted step should store or hash-link:

- load parameter and boundary schedule;
- nodal displacement and damage;
- integration/element strain, stress, positive/negative elastic energy and history;
- elastic, fracture and external-work totals;
- reaction/load curve, damage area, crack-density integral and connectivity metrics;
- equilibrium, KKT, irreversibility, range and energy residuals;
- Newton, active-set and staggered iteration counts;
- step subdivisions/retries and a complete failure ledger;
- mesh, material, geometry, path and trajectory identities;
- elastic basis and the nonlinear stress residual when used for EBR-DNO.

## Solver gates

Before any 36-trajectory pilot:

- `h/ell <= 0.25` throughout every potential fracture region at the fine tier;
- equilibrium and KKT relative residuals `<=1e-6`;
- damage/history monotonicity and range violation `<=1e-10`;
- path energy imbalance `<=5%`;
- fine-to-ultrafine peak reaction and fracture-energy changes `<=5%`;
- the intact/disabled-damage regression passes;
- every frozen case remains in the ledger, with no result-conditioned replacement.

The current tunnel mesher needs a near-field/background size field before these
requirements can be claimed once a crack leaves the wall.

## Validation ladder

1. **Element/unit tests:** rigid-body zero energy, spectral reconstruction,
   pure-compression driving-energy behavior, tangent directional derivative,
   constrained-damage KKT.
2. **Intact regression:** fixed damage or very large toughness reduces to the
   existing elastic solution.
3. **MOOSE `crack2d_iso`:** reproduce the official two-dimensional phase-field
   fracture example and archive version, input and output hashes.
4. **Miehe single-edge-notch tension:** three-grid reaction curve and straight
   crack path.
5. **Single-edge-notch shear:** curved crack path, reaction and energy convergence.
6. **Cross-implementation check:** compare the local solver with MOOSE on fixed
   geometry/material/load schedules before using local trajectories as labels.

AT2 has no strict finite onset threshold in the same sense as a cohesive law.
The paper must not call the first nonzero damage value the crack-initiation load;
it should report peak load or a threshold-crossing definition frozen on
development data.

## Resource and claim boundary

The estimated auditable implementation is about 1,500-2,200 lines across solver,
schema, validation, benchmarks and tests, excluding the MOOSE installation and
cross-validation work. The principal risks are the spectral tangent, external
work sign under wall unloading, crack-band resolution, active-set correctness
and finite-domain effects.

Passing this blueprint would support a synthetic, two-dimensional,
quasi-static brittle-fracture study. It would still not support compression-
shear fragmentation, contact/ejection, inertia-driven rockburst, AE waveforms or
field validity.
