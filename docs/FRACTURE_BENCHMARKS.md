# C-fracture external benchmark gate

`scripts/run_fracture_benchmarks.py` currently owns one deliberately narrow
external check: the official MOOSE `crack2d_iso` regression test. It is a
reference-environment gate, not a validation of TunnelGeoPT's AT2 kernel.

## Pinned execution

Build MOOSE's combined application in its official conda environment, then run
the exact test name. Both repositories must be clean, and the execution command
requires the exact pushed Git heads:

```bash
python scripts/run_fracture_benchmarks.py run-moose \
  --moose-root /home/user/projects/moose \
  --expected-moose-head <40-hex-moose-head> \
  --expected-project-head <40-hex-tunnelgeopt-head> \
  --output artifacts/development/moose-crack2d-iso-v1
```

The runner invokes only:

```text
run_tests --re '^test:phase_field_fracture\.crack2d_iso$' -j 1 --no-color
```

It rejects zero-test runs, near-name matches, `NOT OK`, `FAILED`, `SKIPPED`,
dirty tracked sources, missing upstream provenance, wrong heads, and reused
output directories. The result records hashes for the official input, gold
output, test specification, test harness, executable, stdout, and stderr.

## Evidence boundary

A pass supports only this statement:

> The pinned MOOSE build executed its own pinned `crack2d_iso` regression test.

It does **not** establish local-solver equivalence, tunnel-fracture validity,
SENT/SENS convergence, dynamic rockburst behavior, or field validity. Those are
separate gates in `configs/fracture_phase1_pilot.json`.

Official references:

- <https://mooseframework.inl.gov/getting_started/installation/conda.html>
- <https://mooseframework.inl.gov/source/materials/ComputeLinearElasticPFFractureStress.html>
