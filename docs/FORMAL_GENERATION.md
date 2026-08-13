# v0.3 formal FEM generation contract

`tunnelgeopt.formal_generation` is the trusted data-generator boundary for the
frozen multi-fidelity study. It builds the complete identity plan before any
FEM label is solved, runs genuine gmsh/scikit-fem coarse/fine pairs on the same
512 query points, and runs a second fine/ultrafine pair for the frozen 20%
audit. It does not authorize fracture, damage, rockburst, 3-D, or field claims.

## Public API

- `build_formal_generation_plan(config, forbidden_identities=...)` derives 195
  parent geometries, 705 physical cases and 144 audit cases. A formal plan
  requires a hashed non-empty legacy/seen identity exclusion artifact; tiny
  overrides are deliberately `formal_eligible=False`.
- `generate_formal_dataset(...)` supports SHA-256-verified per-case caches,
  deterministic progress callbacks and resume. A selected audit case is cached
  only after its ultrafine solve succeeds.
- `training_data_paths(root)` returns only the public input archive, train/dev
  label archive and dataset manifest. It never returns a locked label path.
- `trusted_locked_label_path(root, partition)` is evaluator-only. The formal
  runner resolves it after checkpoint freeze; an isolated training worker gets
  only explicit public/train paths and cannot import the generator module.

## Frozen sampling

Each locked partition and section uses one scrambled Sobol sequence seeded by
the SHA-256 substream rule. ID shape coordinates are in `[0.15, 0.85]`.
Geometry-OOD parents anchor at least one shape coordinate in `[0, 0.1]` or
`[0.9, 1]`; roughness is always positive in `[0.008R, 0.025R]`.

Train/dev are not two separately generated distributions. Each section first
creates one 30-parent candidate pool from the shared `train_dev` seed. The
generator then sorts candidates by SHA-256 of canonical
`{salt, section, geometry_group_id}` and assigns 24 to train and 6 to dev.
Changing only the split salt can change the assignment but not the candidate
identity set.

Load streams are partition-, section-, parent- and load-index-specific. ID,
low-lateral, large-rotation and joint-OOD ranges come directly from
`configs/multifidelity_formal.json`; loads are stored internally using the
tension-positive sign convention.

## Stored fields and sealing

The public archive contains `base_features`, `coarse_stress`, separate
`training_weights` and `metric_weights`, `arc_weights`, wall rock-side normals,
stress scales, three region masks, query hashes, all four physical identities,
formal partitions, sections and load subtypes. The train/dev archive contains
aligned fine labels and its non-locked fine/ultrafine audit rows. Each locked
partition has one independent sealed archive with aligned fine labels and only
that partition's audit rows.

The trainer-visible manifest `files` map contains only public and train/dev
filenames. Locked store digests appear under `opaque_sealed_stores`, keyed by
the SHA-256 of canonical `{run_id, partition}`; no sealed filename or path is
published to training. Detailed locked audit values remain inside sealed
archives until evaluation.

## QC and failure semantics

Every planned case gets one solver/mesh record. The record covers coarse and
fine non-finite fraction, free-DOF residual, Clapeyron closure, signed triangle
area, triangle quality, explicit wall/farfield tags, independently checked
cavity centroids, boundary/outer-domain agreement, and query location. Mesh
identity checks remesh the same registered geometry independently; they do not
assume that a successful solve proves the cavity-centroid condition.

A failed solve or QC check is recorded as `valid=false` with a structured
failure; no NaN label is fabricated and no replacement is attempted. Only
valid cases enter learning stores. The frozen 95% validity gate is evaluated
per partition/section: 19/20 passes and 18/20 fails. Partition/load-subtype
rates are reported as diagnostics only; they are not an additional gate. A
failed selected fine/ultrafine case is an immediate audit failure.
When any validity/audit gate fails, the complete manifest and already sealed
evidence are written with `generation_status=ABSTAIN`, then generation raises;
the runner must not enter training.

## Performance expectation

The six-case real-FEM test (three section families, one train and one locked
case per family, all selected for ultrafine audit) completes in roughly 30-45 s
on the current laptop after independent mesh-QC remeshing. The formal plan has
705 coarse/fine pairs plus 144 fine/ultrafine audits; runtime is expected to be
on the order of tens of minutes, with per-case verified cache/resume protecting
completed work. This is an estimate, not a completed formal run.
