# Formal scientific decision engine

`src/tunnelgeopt/formal_analysis.py` is the independent, fail-closed decision
layer for the frozen v0.3 coarse-to-fine elastic experiment. It does not read
files, generate data, train a model, or open sealed labels. Its public entry
point is:

```python
from tunnelgeopt.formal_analysis import evaluate_formal_decision

decision = evaluate_formal_decision(
    config,
    sealed_metrics,
    dataset_manifest,
    access_state,
)
```

All arguments are already-materialized JSON-like mappings. The returned value
is canonical-JSON serializable and includes a digest over the decision payload.
Any absent, non-finite, unauthenticated, or internally inconsistent required
field yields `ABSTAIN`; no evidence is silently inferred or filled with a
default.

## Aggregation and uncertainty

For every method, ratio, seed, section, and OOD slice, all loads are averaged
inside their parent geometry first. Parent geometries are then averaged inside
each section family, and the three section families receive equal weight. The
primary intervals use exactly 20,000 paired hierarchical bootstrap draws: five
training seeds are paired and resampled, then parent geometries are paired and
resampled independently inside each section. Point- or load-level bootstrap is
not available.

The decision checks the one-sided 95% upper bounds for `R_s`, `R_d`, and `R_c`
on locked IID, geometry OOD, and load OOD. A two-sided 95% primary-ratio
interval wider than 0.10 is an invalidity condition (`ABSTAIN`). Joint OOD is
always computed and reported, but neither its effect ratios nor its interval
width can alter the primary scientific classification.

## Evidence contract

The runner must build four objects with these groups of fields:

- `config`: the complete frozen formal config. Its canonical SHA-256 is derived
  inside the decision module.
- `dataset_manifest`: config/run identity, artifact hashes, split and leakage
  booleans, one solver/mesh QC record per planned case, and the fine-ultrafine
  selection made before label access.
- `sealed_metrics`: all 35 checkpoint case errors for all four locked
  partitions; identities, section and load-subtype slices; wall-offset
  `D_t/D_r` arrays for raw coarse and Residual50; sealed fine-ultrafine values;
  generation, per-checkpoint training, and per-partition evaluation runtime and
  peak-memory records.
- `access_state`: hashes of all inputs and durable files, the frozen 35-entry
  checkpoint registry with unique checkpoint and training-contract hashes,
  leakage booleans, exactly one open per locked partition, and exactly one
  evaluation per checkpoint-partition pair. It also carries the complete
  implementation manifest: a clean 40-hex Git `HEAD` equal to its configured
  upstream, tracked-source status, and an exact 20-file SHA-256 closure. That
  closure covers the runner, isolated worker, generator, analyzer, package
  initializer, learning/core FEM modules and their imported schema, validation,
  case, Kirsch and lift dependencies, plus the three frozen config/approval/
  exclusion inputs. Python/scientific-library/CUDA/device provenance, the
  canonical implementation-manifest file hash, and the prepare-manifest file
  hash are also mandatory. A dirty, unpushed, untracked, incomplete, CPU-only,
  or hash-mismatched implementation is `ABSTAIN`.

The dataset validity path verifies every partition-by-section cell has all
planned case records and at least 95% valid cases. A valid case must meet the
frozen non-finite, algebraic residual, Clapeyron energy, signed-area, triangle
quality, boundary-tag, cavity-centroid, boundary/outer-domain, and query-location
conditions on every required fidelity. The selected fine-ultrafine audit must
cover at least 20% with at least three cases in every partition-by-section cell
and pass overall median/p95/any-section median limits of 3%/5%/4%.

## Classification precedence

1. `ABSTAIN`: leakage, access, hash, checkpoint, evaluation-count,
   solver/mesh, convergence, non-finite, seed-count, CI-width, or mandatory
   report evidence is missing or invalid.
2. `NO_GO`: the experiment is valid, but at least one primary upper-CI, 4/5
   same-seed stability, IID section, load-OOD subtype, or wall-offset effect
   gate fails.
3. `GO`: every validity and every effect gate passes. Only the frozen synthetic
   linear-elastic multi-fidelity label-efficiency claim is then permitted.

`tests/test_formal_analysis.py` constructs a complete synthetic passing
fixture, mutates each gate independently, and separately exercises the real
20,000-draw seed/parent bootstrap. The fixture is decision-logic QA only; it is
not scientific evidence.
