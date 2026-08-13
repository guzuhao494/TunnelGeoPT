# v0.5 independent load-basis confirmation

Date: 2026-08-14

## Decision

`LINEAR_ELASTIC_LOAD_AXIS_FACTORIZATION_CONFIRMED`

This is a narrow numerical-physics result. In three new identities relative to
the frozen v0.2/v0.3 exclusion sources, one per tunnel-section family, the
current two-dimensional small-strain plane-strain linear-elastic solver was run
on three canonical basis loads and five independent direct-FEM held-out loads
per geometry. Each geometry retained one fixed material, fine mesh, boundary,
and query grid across all eight loads.

## Evidence

- implementation HEAD and upstream: `44d244e344a0e40dbf33fdaa21cc823b8f46a85a`;
- frozen plan SHA-256: `cf91a557bae100545ad84bec121cc6bbcdcc09e1ae6a7fd3da98c7d9cf463ef5`;
- runner SHA-256: `efac59c4ea3dabad630c3da8f6bb6535b3191b52c62a1accc6f99fb4deabada3`;
- config SHA-256: `b83849dd7b8940c7e63eabf145d3ac281df5f0e0e5e60effae3f3e2b9eb6eaaa`;
- result artifact SHA-256: `b86efe9e283f4ee1f00cb2d8cb01d754475fb9a8ddf1e34338d01ecd5c89e736`;
- `24/24` direct solves completed, zero failed, `15` held-out comparisons;
- all `17/17` frozen validity and numerical gates passed.

Primary query total in-plane stress tensor RelL2:

| Metric | Value |
|---|---:|
| Median over 15 held-out loads | `4.885724690966474e-15` |
| Maximum over 15 held-out loads | `5.882054174085674e-15` |

Auxiliary global maximum relative errors:

| Response | Maximum RelL2 |
|---|---:|
| Nodal displacement | `9.5927e-16` |
| Element incremental in-plane stress | `1.2359e-14` |
| Query incremental in-plane stress | `1.2921e-14` |
| Element `sigma_xx` (report-only) | `4.6708e-15` |

The maximum solver algebraic residual was `2.0815e-13`, and maximum energy
closure error was `1.3998e-14`. An independent audit recomputed the artifact
hash, plan hash, ledger counts, raw-error aggregates, all gates, and identity
intersections; it found no basis/held-out mixing, non-finite values, duplicate
JSON keys, machine-local paths, or usernames.

## Allowed claim

For these three new, individually fixed geometry/material/fine-mesh/query
systems, the current 2D small-strain plane-strain linear-elastic mapping from
in-plane far-field stress to displacement and stress responses is reproduced
by a three-load-axis basis to machine precision.

This does **not** establish geometry, mesh, or material generalization. It does
not model plasticity, damage, crack initiation or propagation, contact,
rockburst dynamics, micro-to-field transfer, field prediction, or engineering
truth. The nine channels are the coefficients of a `3 x 3` linear response
matrix, not nine physical load axes and not GeoPT's three-step vector-distance
targets.

## Reproducibility boundary

`confirmation.json` contains all per-held-out errors and the complete 24-solve
ledger, so every published aggregate gate can be recomputed without rerunning
FEM. It intentionally does not retain the full field tensors; recomputing the
field-level relative errors from raw arrays requires rerunning the frozen
direct solves.
