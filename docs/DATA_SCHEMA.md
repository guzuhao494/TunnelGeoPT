# TunnelGeoPT data schema

TunnelGeoPT keeps a narrow compatibility layer for GeoPT's released lifted-
geometry data format.  That layer is intentionally separate from the future
high-fidelity rock-mechanics layer: passing schema validation means that a
sample can enter the GeoPT-shaped pipeline, not that it contains a physical
law of fracture or rockburst.

## Official-compatible lifted-geometry layer

One geometry case is a directory containing a shared geometry array and one
or more indexed trajectory pairs:

```text
case_000123/
├── x.npy
├── condition_0.npy
├── supervise_0.npy
├── condition_1.npy
├── supervise_1.npy
└── meta.json             # optional
```

The three required arrays have a common point count `N` and a common floating
dtype:

| File | Shape | Columns | Official-compatible dtype |
| --- | --- | --- | --- |
| `x.npy` | `[N, 7]` | normalized `xyz`, distance to geometry, volume-point direction `(closest-point)/distance` or surface normal `gxyz` | `float16` |
| `condition_k.npy` | `[N, 4]` | unit transport direction `vxyz`, step length/magnitude | `float16` |
| `supervise_k.npy` | `[N, 9]` | three concatenated vector-distance targets `dvec_t0`, `dvec_t1`, `dvec_t2` | `float16` |

The GeoPT release uses 32,768 volume points and 4,096 surface points, so its
default is `N = 36,864`.  TunnelGeoPT validates the interface rather than
hard-coding that sampling budget; smaller smoke tests and later sampling
studies may use another positive `N` as long as all three arrays agree.

For a tunnel-cavity adaptation, `x` should describe rock-side points inside a
finite surrounding box but outside the closed excavation cavity.  A vector
target at step `t` follows the released generator's sign convention:

```text
dvec_t = particle_position_t - closest_surface_point_t
```

Surface tracking points remain fixed by assigning zero step length.  None of
these columns should be reinterpreted as crack damage or material strength.

### Python API

```python
import numpy as np
from tunnelgeopt.schema import load_sample, save_sample, validate_arrays

validate_arrays(x, condition, supervise)  # strict float16 compatibility

save_sample(
    "pretrain/case_000123",
    x,
    condition,
    supervise,
    trajectory_index=0,
    meta={
        "case_id": "case_000123",
        "num_points": int(x.shape[0]),
        "dtype": "float16",
        "axis_convention": {"x": "tunnel", "y": "up", "z": "transverse"},
        "normalization": {"characteristic_length": 5.0},
        "random_seed": 1234,
    },
)

sample = load_sample(
    "pretrain/case_000123",
    trajectory_index=0,
    require_meta=True,
)
```

`save_sample` casts to `float16` by default and writes each file atomically.
It permits another indexed trajectory to reuse an identical `x.npy` and
identical `meta.json`, but protects existing trajectory files unless
`overwrite=True` is explicit.  Pass `dtype=np.float32` and load with
`expected_dtype=np.float32` only when a higher-precision variant is a
deliberate project decision; it is not byte-for-byte compatible with the
released float16 dataset.

Validation rejects:

- missing or non-NumPy arrays;
- incorrect rank or fixed width;
- zero points or inconsistent point counts;
- integer, float64, object, or mixed dtypes;
- a dtype different from the requested compatibility dtype;
- `NaN` or positive/negative infinity;
- malformed, non-object, non-finite, or non-JSON-serializable metadata;
- optional metadata declarations for `num_points` or `dtype` that disagree
  with the arrays.

The validator deliberately does not impose physical ranges on coordinates,
directions, or step lengths.  Those checks depend on the selected coordinate
normalization and dynamics sampler and belong in the generator's domain QA.

## `meta.json`

Metadata is optional for compatibility but strongly recommended for every
generated tunnel case.  It is UTF-8 JSON and should record information needed
to reverse preprocessing and reproduce generation, such as:

- stable geometry/case identifier and source provenance;
- random seed and generator version or commit;
- axis convention and normalization transform;
- original tunnel dimensions and characteristic length;
- geometry family and synthetic-parameter values;
- point counts and stored dtype;
- mesh quality checks and exclusions.

Metadata remains case-level information and is not automatically consumed by
the model.  Do not place a large point field in JSON.

## Future high-fidelity rock-mechanics layer

The fixed-width compatibility arrays encode geometry-bounded synthetic
transport.  They do **not** have enough channels or supervision to represent:

- the full initial stress tensor and excavation/load history;
- elastic, plastic, damage, brittleness, anisotropy, and rate parameters;
- joints and microcracks with two-sided, open/closed, frictional state;
- evolving displacement, stress, strain, damage, fracture topology, and
  released energy through time;
- acoustic-emission, ejection, or field-monitoring observations and their
  uncertainty.

Those quantities will use a versioned high-fidelity schema with explicit
units, masks, topology, time coordinates, and train/validation/test provenance.
They must not be compressed into `condition[:, 3]`, smuggled into `meta.json`
as model input, or labeled as GeoPT-compatible vector-distance supervision.
The compatibility layer may initialize a geometry-aware backbone; calibrated
simulation or experimental labels are still required before making claims
about fracture or rockburst mechanisms.
