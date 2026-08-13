# v0.4 exploratory prototype stop record

This note is the human-readable companion to
`artifacts/analysis/v04-structured-prototype-stop/exploratory_record.json`.
It preserves the evidence that caused the v0.4 implementation stop without
pretending that an unlogged prototype is reproducible.

## Provenance boundary

The prototype result was reported during the development conversation. No
independently retained raw training log, checkpoint set or original metric
export exists in the repository; the numbers below are a manual transcription
of that conversation. Its provenance is therefore `conversation_record_only`,
and it is not replayable from this record. Any unrecorded value must not be
reconstructed or inferred.

The manually migrated structured-residual probe used the
`pointwise_global_mean` module, the exact original v0.3 36-train-parent and
18-dev-parent identities, and normalized near-field tensor relative L2.  Its
values were:

| Seed | Train | Dev | Seen IID | Seen geometry | Seen load | Seen joint |
|---:|---:|---:|---:|---:|---:|---:|
| 103 | .032483 | .032815 | .032092 | .033226 | .037250 | .037148 |
| 211 | .032777 | .033052 | .032226 | .033268 | .037394 | .037008 |
| 307 | .031894 | .032393 | .031944 | .033440 | .036908 | .037677 |
| 401 | .031229 | .032278 | .031661 | .034175 | .037442 | .038317 |
| 509 | .029552 | .031190 | .030610 | .032927 | .037120 | .038025 |
| Mean | .031587 | .032346 | .031707 | .033407 | .037223 | .037635 |

Coarse means were `.033556`, `.034412` and `.038596` on seen IID,
geometry and load respectively.  The approximate candidate/coarse point ratios
were therefore `.945`, `.971` and `.965`, far above the proposed `.68`, `.78`
and `.78` launch targets.

The seed-103 geometry-attention probe with key width 8 produced:

| Heads | Seen IID | Seen geometry | Seen load | Seen joint |
|---:|---:|---:|---:|---:|
| 2 | .031840 | .033312 | .036922 | .037470 |
| 4 | .032026 | .033440 | .037410 | .037521 |
| 8 | .033131 | .034114 | .038031 | .037880 |

The best two-head variant was only a modest change and did not justify a
production cross-fit.  No production v0.4 cross-fit was run and no new locked
case was generated.

These observations support only
`IMPLEMENTATION_STOP_PENDING_PIVOT`.  They support no effect, independent
validation, formal `GO`/`NO_GO`, fracture, rockburst or engineering claim.

## Numbers verified from the current implementation

Parameter counts were recomputed locally by summing model parameter tensor
sizes; unlike the prototype metrics, these values are code-derived:

| Implementation | Parameters |
|---|---:|
| structured linear residual | 40,685 |
| generic v0.3 Residual50 | 38,787 |
| no strict load linearity | 38,598 |
| no local tensor frame | 40,813 |
| no zero-init gate | 40,685 |

The comparisons are not parameter matched.  They are also not loss matched:
the structured path uses a per-case relative residual loss, while the generic
path uses legacy weighted MSE.  Finally, structured input packing has 17
channels and includes derived frame/wall information, whereas the generic path
uses 14 channels.  The one-switch variants are therefore diagnostic probes of
whole pipelines, not causal evidence that any component is necessary.

## Durable decision

Real cross-fit remains disabled in
`configs/multifidelity_v04_development.json`.  Validation and deterministic
tiny-mock auditing are allowed; production training requires a later explicit
pivot, a revised frozen protocol and fresh authorization.
