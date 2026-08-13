# v0.3 multi-fidelity elastic data contract

This layer asks a deliberately bounded question: can a learned residual correct
a cheap coarse-mesh **linear-elastic** tunnel stress field toward the same
boundary-value problem solved on a finer mesh? It is not fracture, damage,
ejection, rockburst, or field-validation evidence.

## Identity and split unit

Three SHA-256 identities prevent accidental pairing and leakage:

- `geometry_group_id`: shape family, continuous shape parameters, and the exact
  frozen `float64` boundary;
- `load_group_id`: prescribed far-field tensor (tension positive) and optional
  axial stress;
- `case_group_id`: geometry, load, and elastic material. Mesh size is excluded,
  because coarse and fine must be two discretizations of one physical case.

The split unit is `geometry_group_id`, not load or mesh. Every load associated
with a boundary inherits that boundary's `train`, `dev`, or `locked_test`
assignment. `freeze_geometry_splits` sorts hashes and writes an explicit
`GeometrySplitSpec`; it should run before the first elastic solve.

## Common physical query

Each geometry owns one deterministic `ElasticQueryGrid` with fixed point count
`P`. It contains near-field rock points, rock-side wall-offset points, and a
far-field ring. The same physical `(y,z)` points are located independently in
the coarse and fine triangle meshes. Element indices are never paired across
meshes. A query hash covers the boundary identity, coordinates, `x[7]`, region
weights, point counts, scales, and seed.

The frozen near-field distance interval is explicit and defaults to
`[0.05R, 2.0R]`. The far-field rectangle scale and optional actual outer-domain
scale are also hashed. Grid construction rejects a far-field scale on or beyond
the preregistered outer boundary, and the paired solver checks every query lies
strictly inside the actual domain.

`area_weights` are normalized over near-field points; `arc_weights` are local
polyline quadrature weights normalized over wall-offset points. The masks keep
these two measures explicit instead of silently mixing area and boundary
statistics.

## Arrays and conventions

For every case:

| Array | Shape | Meaning |
| --- | --- | --- |
| `grid.x` | `[P,7]` | normalized GeoPT geometry coordinates/distance/direction |
| `condition` | `[4]` | normalized far-field `[yy,zz,yz,xx]` |
| `coarse_stress_normalized` | `[P,3]` | coarse total stress `[yy,zz,yz]` |
| fine label | `[P,3]` | fine total stress `[yy,zz,yz]` |
| residual label | `[P,3]` | `fine - coarse` |
| model features | `[P,14]` | `x7 + repeated condition4 + coarse3` |

Stress is tension positive throughout. The stress scale is the tensor
Frobenius norm of the *prescribed in-plane far field*, so normalization cannot
peek at a coarse or fine solution. Reconstruction is
`fine_normalized = coarse_normalized + predicted_residual`; multiplying by the
known far-field scale returns pascals.

Coarse and fine solves use the same frozen boundary, outer bounds, material,
far-field load, first-order plane-strain formulation, and sign convention.
Only the three mesh-size controls differ, and fine sizes must be no larger than
coarse sizes with at least one strict refinement.

## Locked-label access

`MultiFidelityDataset.features_for` may read locked-test features, including
the permitted coarse solver input. Fine stress and residual targets are private
and only available through audited methods. Any locked fine-label read before
the complete unique checkpoint set is frozen is rejected and counted. After
authorization, evaluation reads are counted separately by split and purpose.
Direct `sample.fine_stress_normalized` access deliberately fails.

The data object records actual reads; the training runner must persist
`access_snapshot()` alongside checkpoint identities. This access boundary does
not by itself guarantee sound model selection, so the experimental protocol
must still prohibit test-derived hyperparameter or gate changes.

## Known scope limits

- fine FEM is a numerical reference, not an exact solution;
- piecewise-constant P1 element stress has discretization error near the wall;
- a finer mesh may still be insufficient for peak stress convergence;
- homogeneous elasticity contains no strength, damage, crack topology,
  discontinuity, energy release, dynamic ejection, or monitoring uncertainty;
- passing a coarse-to-fine gate would justify the next fracture-data stage,
  not an engineering rockburst prediction claim.
