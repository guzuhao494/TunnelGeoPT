# Deterministic P1 stress-recovery operator

`stress_recovery.py` converts one constant planar stress vector
`[yy, zz, yz]` per triangular element into a continuous, piecewise-linear
query field. It is a pure NumPy post-processing operator and does not alter the
finite-element solve.

For each mesh node, the operator gathers incident triangles and evaluates their
centroids and areas. It fits all three stress components together with the
geometry-only weighted affine model

```text
sigma(centroid) = sigma(node) + gradient_y * dy + gradient_z * dz,
weight = triangle_area / distance(node, centroid).
```

If the local design has rank three, its intercept is the recovered nodal
stress. If the patch is rank deficient, the operator uses the same weights for
a constant average. Query values are the barycentric interpolation of the
three recovered nodal values in the containing triangle.

## API and validation

```python
from tunnelgeopt.stress_recovery import recover_stress_at_queries

query_stress = recover_stress_at_queries(
    nodes_yz,          # [N, 2]
    elements,          # [M, 3] integer connectivity
    element_stress,    # [M, 3] tension-positive [yy, zz, yz]
    query_points_yz,   # [P, 2]
    element_ids=None,  # optional [P]
)
```

The function returns a finite `float64` array `[P, 3]`. If `element_ids` are
omitted, a deterministic dependency-free locator is used. Supplied identifiers
are still checked by recomputing barycentric coordinates. The implementation
rejects non-finite arrays, invalid shapes, non-integer or out-of-range
connectivity, duplicate or degenerate triangles, unreferenced nodes,
non-manifold edges, exterior queries, and query/element mismatches.

All regression, fallback, and interpolation weights depend only on mesh and
query geometry. There is no stress-dependent branch, clipping, activation, or
regularization. The resulting operator is therefore homogeneous and additive
in `element_stress`; tests verify constant exactness, affine reproduction on
full-rank interior patches, and numerical superposition.

## Evidence boundary

These tests establish the deterministic operator contract only. They do not
show that recovered stress is closer to an analytic, ultrafine, experimental,
or field reference than the original piecewise-constant P1 stress. Such an
accuracy claim requires a separately frozen comparison on unseen geometries,
loads, query points, and mesh resolutions.

## Wall-compatible correction

`preserve_baseline_traction_with_tangential_correction` is a second,
geometry-only operator for wall-near queries. It retains only the
tangential-tangential component of the recovered-minus-baseline increment.
The corrected field therefore keeps the original coarse traction exactly while
accepting the recovered tangential stress. This is a baseline-nonworsening
boundary safeguard, not an assertion that an offset query is traction-free.
It remains linear in both stress inputs. Its usefulness must be evaluated as a
new development candidate because it was introduced after the unconstrained
recovery's wall diagnostic failed.
