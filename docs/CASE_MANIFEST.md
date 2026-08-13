# B-elastic case identity and frozen split contract

This document defines the executable dataset-identity contract implemented in
`tunnelgeopt.cases`.  It is a leakage-control mechanism, not evidence that an
elastic or fracture solver has run.

## Parent physical identity

Every independent physical case contains exactly these identity components:

1. `section_family`: `circle`, `horseshoe`, or `straight_wall_arch`;
2. `section_parameters`;
3. `material_field_seed`;
4. `joint_network_seed`;
5. `dimensionless_material_parameters`;
6. symmetric `initial_stress_tensor` (3x3 or six-value Voigt input);
7. `stress_orientation`;
8. `excavation_schedule`;
9. `unloading_schedule`.

The values are validated, normalized into finite JSON-native values, serialized
as compact UTF-8 JSON with sorted keys, then hashed with SHA-256.  This digest is
the `case_group_id`.  Dictionary order, `1` versus `1.0`, `-0.0` versus `0`, and
3x3 versus equivalent Voigt stress input do not change the identity.  The scalar
principal-stress axis is canonical modulo 180 degrees (`-45` equals `135`).  A
change to any non-equivalent physical component does.

Mesh size/topology, numerical fidelity, solver name/version, time-step choice,
restart attempt, output sampling, and augmentation are **not** parent identity
fields.  They belong to derived records carrying the parent's `case_group_id`.
This prevents low- and high-fidelity views of one physical problem from being
miscounted as independent cases.

Validation rejects missing fields, unsupported sections, non-finite numbers,
invalid seeds, non-positive size/modulus/strength-like parameters, roughness
outside `[0, 0.08]`, Poisson ratio outside `(-1, 0.5)`, non-symmetric stress,
axis orientation outside `[-180, 180)`, schedule fractions outside `[0, 1]`, and non-increasing
explicit schedule times.  Generic dimensionless values are bounded to
`[-1e6, 1e6]` to catch unit leaks and corrupt records.

## Frozen split

Cases are stratified independently inside each section family, sorted by
`case_group_id`, and assigned in `train`, `dev`, `locked_test` order.  `locked_test` is locked:
it cannot be used for pre-training, tuning, checkpoint selection, or early
stopping.

Integer allocations use Hamilton's largest-remainder method with ratios
`70/15/15`.  Floors are assigned first, then remaining places go to the largest
fractional remainders; exact ties follow `train`, `dev`, `locked_test` order.  Therefore:

- six cases in one section become `4/1/1`;
- 128 cases in one section become `90/19/19`.

The assignment is deterministic under input reordering.  An identical parent
appearing twice is a hard error, even if the two inputs have different meshes or
fidelities.  Every derived record inherits the parent split; an orphan, a split
override, or a duplicate derived identity is a hard error.

## Hash layers

The manifest intentionally exposes several audit layers:

- `case_group_id`: canonical physical identity;
- parent/derived `content_hash`: complete frozen record including inherited split;
- top-level `content_hash`: semantic manifest body, excluding both top-level hashes;
- `manifest_hash`: the whole envelope including `content_hash`, excluding only
  `manifest_hash` itself.

`verify_case_manifest` recomputes identities, section-stratified allocation,
derived inheritance, split counts, record hashes, semantic content hash, and
envelope hash.  `write_case_manifest` verifies before an atomic canonical JSON
write; `load_case_manifest` verifies again after reading.

Minimal usage:

```python
from tunnelgeopt.cases import build_case_manifest, write_case_manifest

manifest = build_case_manifest(parent_cases, derived_records=solver_runs)
write_case_manifest("data/manifests/b_elastic_v1.json", manifest)
```
