# Fracture-band mesh contract

`generate_tunnel_mesh` keeps its original mesh path when the fracture-band
arguments are omitted.  Fracture refinement is enabled only when these three
values are supplied together:

- `nearfield_distance`: audited band width measured from the input polygonal
  tunnel wall;
- `nearfield_mesh_size`: public upper bound for a triangle edge in that band;
- `fracture_length_scale`: phase-field length scale `ell`, used to record both
  requested and realized `h/ell`.

`nearfield_transition_width` is optional.  It defaults to
`nearfield_distance`.  The Gmsh background field uses `Distance` on every wall
curve and `Threshold` with a two-edge guard outside the audited band:

- `guard_width = 2 * nearfield_mesh_size`;
- `DistMin = nearfield_distance + guard_width`;
- `DistMax = nearfield_distance + guard_width + nearfield_transition_width`;
- a linear characteristic-length transition between those distances.

The guard is necessary because a triangle with its centroid just inside the
audited band can have vertices outside it.  Starting the transition at the
audit boundary made such triangles sensitive to Gmsh version and platform.
The guard keeps the public band within the constant-size field without changing
which triangles the post-generation audit selects.

Gmsh characteristic length is a target, not an edge-length guarantee.  The
internal `SizeMin` is conservatively set to 0.5 times the public cap, and wall
point sizes are capped at that same target while refinement is enabled.  This
factor is only a meshing control; it is not evidence that the requested cap was
met.  The generated connectivity is always audited afterward.

## Hard audit

The auditable band population is defined exactly as triangles whose centroid
distance to the *input wall polyline* is no greater than
`nearfield_distance`.  For those triangles, the implementation recomputes all
three edge lengths and requires

```text
actual maximum edge
    <= nearfield_mesh_size * (1 + 0.02) + floating-point tolerance
```

The floating-point term is `64 * eps * max(characteristic_radius,
nearfield_mesh_size, 1)`.  The two-percent allowance covers normal Gmsh
discretization around a characteristic-length target.  A missing band, an
incomplete parameter group, a non-positive or non-finite value, or an edge
above the limit raises an exception; no mesh object is returned.

The returned metadata records the requested cap, Gmsh target, band and
transition distances, audited element count, realized maximum edge, requested
and realized `h/ell`, tolerances, audit limit, and pass flag.  The explicit
centroid criterion should be used when independently reproducing the audit; it
does not claim that every triangle merely touching the band has the same cap.

## Phase-1 protocol mapping

The development protocol calls for a `2R` near-field band.  For the most
restrictive `ell/R = 0.04` material setting, `h/ell <= 0.25` maps to
`nearfield_mesh_size/R <= 0.01`, and the ultrafine audit maps to `<= 0.005`.
These settings can be expensive and are intentionally not instantiated in unit
tests.  Tests use a geometrically similar but coarser ratio to verify the real
Gmsh field, metadata, rejection rules, and fail-closed audit path.
