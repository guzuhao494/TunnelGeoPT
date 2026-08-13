# B-elastic persistence schema

`tunnelgeopt.elastic_schema` is the independent persistence contract for the
plane-strain finite-element layer.  It does not reuse the fixed-width GeoPT
A-layer arrays, and it does not imply that fracture, damage, or dynamics have
been simulated.

Each case is one directory containing exactly two schema files:

```text
case_directory/
  arrays.npz
  meta.json
```

The computation/publication default is `float64`.  A caller may explicitly
choose `float32` at conversion, save, and load time; the default loader rejects
such a record so that an accidental precision reduction is visible.

## Array contract

All coordinates use `(y,z)`, where the omitted `x` direction is the tunnel
axis.  Indices are zero-based.  The NPZ member set is exact: an unknown or
missing member is a hard error.

| Array | Shape | Meaning | Unit |
|---|---:|---|---|
| `nodes` | `[N,2]` | nodal `(y,z)` coordinates | m |
| `elements` | `[M,3]` | triangular node indices | index |
| `wall_facets` | `[Bw,2]` | excavation-wall boundary edges | index |
| `farfield_facets` | `[Bf,2]` | exterior boundary edges | index |
| `u` | `[N,2]` | nodal displacement `(u_y,u_z)` | m |
| `strain` | `[M,3]` | engineering strain `[yy,zz,gamma_yz]` | 1 |
| `stress` | `[M,3]` | total stress `[yy,zz,yz]` | Pa |
| `delta_stress` | `[M,3]` | excavation-induced stress increment | Pa |
| `sigma_inf` | `[2,2]` | symmetric in-situ `(y,z)` stress tensor | Pa |
| `sigma_xx` | `[M]` | total out-of-plane plane-strain stress | Pa |
| `energy_density` | `[M]` | incremental elastic energy density | J/m3 |
| `area` | `[M]` | element area | m2 |
| `centers` | `[M,2]` | element centroids | m |

Boundary facets are explicit undirected node pairs, not scikit-fem's
process-local facet numbers.  Conversion from `ElasticResult` resolves its
facet-number arrays against the deterministic edge table.  Validation requires
the wall and far-field sets to be non-empty, unique, disjoint, and to cover the
entire geometric mesh boundary.

The schema fixes these physical conventions:

- strain order: `[yy, zz, gamma_yz]`;
- stress order: `[yy, zz, yz]`;
- stress sign: tension positive, hence compressive rock stress is negative;
- `stress = delta_stress + [sigma_inf_yy, sigma_inf_zz, sigma_inf_yz]`;
- `energy_density = 0.5 * strain : delta_stress`, with engineering shear;
- `sigma_xx = sigma_xx_inf + lambda * (epsilon_yy + epsilon_zz)`.

`area` and `centers` are recomputed from `nodes/elements` during validation.
Element/facet indices, repeated or degenerate triangles, non-finite values,
inconsistent constitutive quantities, and a changed component/sign/unit
convention are rejected.

## Metadata and provenance

`meta.json` contains:

- schema name/version and publication dtype;
- `case_group_id`, `mesh_id`, and `config_hash`, each a lowercase SHA-256;
- exact component, sign, and SI-unit declarations;
- material parameters, `sigma_xx_inf`, physical tags, and mesh metadata;
- `diagnostics`: energy, external work, algebraic residual, residual norm,
  energy closure, energy discretization error, and stiffness symmetry error;
- caller-provided `env` and `meta` JSON mappings;
- per-array dtype/shape/SHA-256 manifest;
- `mesh_content_sha256`, exact `arrays.npz` file SHA-256, and complete semantic
  `content_sha256`.

The semantic content hash covers all metadata except the exact NPZ file hash
and the content hash itself.  It still commits to every array through the
per-array hashes.  The separate file hash detects byte-level archive changes.
These are integrity/audit hashes, not cryptographic signatures of authorship.

`mesh_id` may be supplied by a dataset manifest.  If it is omitted during
`ElasticResult` conversion, the implementation uses `mesh_content_sha256` as
the mesh identity.  Both values remain present so an external mesh catalogue
can coexist with independently verifiable mesh content.

## Atomicity and conflicts

Save performs all validation before publishing.  It writes temporary files in
the destination directory, flushes them, and uses `os.replace` separately for
`arrays.npz` and `meta.json`.  A lock file prevents two schema writers from
publishing concurrently.  Existing schema files cause `FileExistsError` unless
`overwrite=True`; an overwrite replaces both files and publishes metadata last.

The two-file directory is not claimed to be a transactional database.  If a
process or machine fails between the two atomic replacements, the old metadata
cannot validate the new NPZ hash, so readers fail closed instead of accepting a
mixed record.

Every load verifies the exact file hash, semantic content hash, array manifest,
and mesh hash before constructing `ElasticRecord`; it then repeats the complete
shape, topology, finite-value, component, sign, unit, constitutive, and energy
validation.  Hash agreement alone is therefore insufficient to admit a
physically inconsistent record.

## Scope boundary

The NPZ member set deliberately has no fields for nonlinear damage, particle
velocity, or dissipated energy.  Keys containing those concepts are also
rejected recursively in caller metadata.  Zero arrays must never be inserted
to make a linear-elastic record look like a fracture or dynamic simulation.
Those labels require a separately versioned high-fidelity schema and real
solver evidence.

## Usage

```python
import numpy as np

from tunnelgeopt.elastic_schema import (
    load_elastic_record,
    save_elastic_result,
)

save_elastic_result(
    "data/b_elastic/case-0001",
    result,
    case_group_id=case_group_id,
    mesh_id=mesh_id,
    config_hash=config_hash,
    env=environment_snapshot,
    meta={"section_family": "circle", "split": "train"},
)

# Strict float64 is the default.
record = load_elastic_record("data/b_elastic/case-0001")

# A reduced-precision publication is always explicit at every boundary.
save_elastic_result(
    "data/b_elastic/case-0001-f32",
    result,
    case_group_id=case_group_id,
    config_hash=config_hash,
    env=environment_snapshot,
    publication_dtype=np.float32,
)
record32 = load_elastic_record(
    "data/b_elastic/case-0001-f32", expected_dtype=np.float32
)
```
