# SENT/SENS zero-width slit mesh contract

This document freezes only the geometry and meshing contract for the later
single-edge-notch tension (SENT) and single-edge-notch shear (SENS) runs.  A
mesh passing these checks is **not** evidence that the fracture solver matches
a published load-displacement curve or crack path.

## Coordinates and topology

Repository coordinates are `[y, z]`: `y` is vertical and `z` is horizontal;
all coordinates in this benchmark are millimetres.  The domain is the unit
square.  The slit lies on `y = 0.5`, from the left edge at `z = 0` to the crack
tip `[0.5, 0.5]`.

The Gmsh construction uses two plane surfaces.  They share one curve only on
the intact ligament from the crack tip to the right edge.  On the open part of
the slit, `notch_upper` and `notch_lower` are distinct coincident curves with
distinct nodes; the crack-tip node alone is shared.  No duplicate-removal
operation is called.  This realizes two traction-free faces of zero geometric
width rather than a finite-width slot.

The immutable physical identity is:

| label | meaning |
| --- | --- |
| `bulk` | both two-dimensional coupon surfaces |
| `top` | `y = 1` |
| `bottom` | `y = 0` |
| `left_upper` | `z = 0`, `0.5 <= y <= 1`, ending at the upper slit mouth |
| `left_lower` | `z = 0`, `0 <= y <= 0.5`, ending at the lower slit mouth |
| `right` | `z = 1` |
| `notch_upper` | upper face of the open slit |
| `notch_lower` | lower face of the open slit |
| `notch_tip` | the single shared crack-tip node `[0.5,0.5]` |

The seven entries from `top` through `notch_lower` are one-dimensional facet
groups returned through `boundary_facets`.  `notch_tip` is a zero-dimensional
physical group returned through `boundary_nodes`; it is never represented as
a degenerate facet.  The two left groups have distinct coincident mouth nodes.
Each left group touches only its corresponding notch face at that one mouth
node.

## Frozen refinement plans

The three predeclared corridor maximum-edge targets and matching bulk targets
are:

| tier | corridor target `h` (mm) | bulk target (mm) | `h/ell`, `ell=0.015 mm` |
| --- | ---: | ---: | ---: |
| coarse | 0.007500 | 0.0300 | 0.500 |
| medium | 0.003750 | 0.0150 | 0.250 |
| fine | 0.001875 | 0.0075 | 0.125 |

Both loading modes refine all cells whose centroids are within `0.05 mm` of
the open notch.  The separate propagation corridors are:

| mode | centreline `[y,z]` | half-width |
| --- | --- | ---: |
| SENT | `[0.5,0.5] -> [0.5,1.0]` | `0.10 mm` |
| SENS | `[0.5,0.5] -> [0.0,1.0]` | `0.15 mm` |

Distance/Threshold fields for the notch band and propagation corridor are
combined using Gmsh's `Min` field.  The requested Gmsh size is one half of the
public target.  The generated connectivity remains authoritative: for every
triangle whose centroid belongs to the union of the frozen notch band and
propagation corridor, every triangle edge must satisfy

`h <= 1.15 * target_h`.

Exceeding that limit raises an exception; the factor is not used to relabel a
failing mesh as passing.

## Audits and provenance

Before returning a scikit-fem `MeshTri`, generation verifies:

1. all triangles have positive area and their total area is `1 mm^2`;
2. the seven facet-boundary groups are disjoint and exactly cover all boundary
   facets;
3. each notch-face facet is adjacent to one triangle;
4. each intact-ligament facet is adjacent to two triangles and is not a free
   boundary;
5. upper/lower notch nodes are distinct but geometrically paired, except for
   exactly one shared crack-tip node;
6. the 0D `notch_tip` identity contains exactly that one shared node;
7. `left_upper` and `left_lower` meet only their corresponding notch face at
   distinct slit-mouth nodes, and occupy the correct vertical half-edge;
8. physical coordinates match the frozen square/slit geometry;
9. the realized corridor maximum edge passes the hard limit above.

The plan receives a canonical JSON SHA-256.  The mesh receives a second
SHA-256 over canonicalized coordinates, connectivity, physical boundary
edges, numeric physical tags, Gmsh entity tags, and the dimensioned 0D
crack-tip identity.  `recompute_topology_sha256()` binds the recorded digest
back to the exact returned object.  A repeated Gmsh generation test requires
the same topology hash for the same plan.  Identity, physical tags,
physical-entity tags, audit metadata, wrapper arrays, and the backing
`MeshTri.p/t/facets/f2t` arrays are read-only; boundary and subdomain maps are
also immutable.

Real Gmsh tests cover both loading modes at all three tiers.  They are mesh
tests only: no displacement loading, crack evolution, reaction curve, or
published-reference comparison is executed here.
