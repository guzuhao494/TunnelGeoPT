# Structured linear residual operator (v0.4 development)

This module is a development candidate for the existing two-dimensional
linear-elastic coarse-to-fine task.  It is not a fracture, damage, rockburst,
micro-scale-transfer, or field-warning model.

For each query point, a nonlinear encoder sees only static geometry.  It emits
a matrix `A(g)`.  The correction is a matrix action on normalized far-field
and coarse-stress tensors:

```text
delta_sigma_normalized = A(static_geometry) * dynamic_stress_normalized
fine_prediction = coarse + delta_sigma
```

The strict action contains no bias, activation, LayerNorm, dynamic-dependent
attention, or dynamic mean centering.  The optional case mean is a fixed linear
aggregation.  If cases use scales `S1` and `S2`, physical superposition is
tested after de-normalization: the combined normalized input is
`(S1*d1 + S2*d2)/S12`, and the dimensional predictions must add.  A direct
zero-load call is not used because the public stress-scale contract correctly
rejects a zero tensor.

## Frame contract

`pack_structured_features` preserves the shared 14 channels and appends a
deterministic frame plus a wall-ring flag.  At the frozen wall-offset ring, the
generator's stored rock-outward physical normal is used exactly.  Elsewhere,
the frame uses normalized `-g_yz`, the nearest-distance direction.  The latter
is intentionally not called an exact physical wall normal.  Symmetric stress
rotation treats `[yy,zz,yz]` as tensor components, including the factor of two
in normal-stress transformations; it never rotates the three channels as an
ordinary vector.

The local-frame geometry input uses radial/tangential coordinates, distance,
and the wall-ring flag.  This makes the strict branch equivariant to proper
two-dimensional rotations when points, wall normals, loads, and stress tensors
are rotated together.  Reflections and three-dimensional tunnel rotations are
outside the claim.

## Controlled ablations

The development protocol changes one switch at a time:

- remove strict load linearity;
- remove the local tensor frame;
- remove exact zero initialization of the coarse-preserving correction gate.

All use the same hidden width and number of context blocks.  Exact-zero
initialization makes the initial fine prediction identical to the coarse field.
It does not guarantee non-worsening after training; the frozen wall and error
gates still decide that empirically.

The entire v0.3 data set is now seen development material.  Results from it can
only decide whether drafting a new preregistration is worthwhile.  They are not
an independent validation result or an effect claim.
