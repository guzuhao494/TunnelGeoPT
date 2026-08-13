# B-elastic v0.2.0 milestone

`scripts/run_elastic_milestone.py` is the evidence-producing runner for the
first numerical-physics layer.  Its scope is deliberately narrow: homogeneous,
isotropic, two-dimensional small-strain plane-strain elasticity around a
traction-free tunnel opening.

## Frozen execution order

1. Read and validate `configs/elastic_milestone.json`.  The runner rejects a
   relaxation of the preregistered numerical gates.
2. Materialize an exact config snapshot and environment snapshot.
3. Deterministically expand three section families times six physical cases.
4. Use `tunnelgeopt.cases` to freeze the 18 parent cases, leakage-safe 4/1/1
   per-section splits, and the 18 planned medium-mesh derived records.  Write
   and reload the manifest before any solver call.
5. Run the affine patch test, then all nine Kirsch combinations (three mesh
   tiers times uniaxial/equal-biaxial/pure-shear loading).
6. Run every one of the 18 planned medium-mesh cases exactly once.  Input rock
   stresses are compression-positive; the runner performs one explicit sign
   conversion to the solver's tension-positive convention.
7. Validate generic numerical and geometric QC, serialize every successful
   solve through the strict independent `elastic_schema`, reload it, and retain
   every failure without generating replacement samples.
8. Write raw metrics, a go/no-go decision, the append-only run log, and a
   SHA-256 file inventory.

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_elastic_milestone.py
```

The immutable evidence directory is
`artifacts/experiment/b-elastic-v0.2.0/`.  A successful gate permits generation
of elastic solver-emulation data only.  It does not validate fracture, damage,
dynamic instability, field prediction, or a rockburst mechanism.
