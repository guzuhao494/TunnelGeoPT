# Current board packet

Date: 2026-08-14

- `current_mainline`: build and validate a two-dimensional quasi-static brittle
  fracture layer, then test an elastic-basis-conditioned damage operator.
- `incumbent`: no fracture surrogate exists. The strongest learned incumbent is
  the v0.3 direct fine-field operator; its data are now development-only.
- `latest_decisive_result`: the elastic load-axis factorization is independently
  confirmed, but it is an implementation/physics lemma rather than a paper-level
  learning contribution.
- `active_blocker`: no validated C-fracture solver, schema, trajectories, or
  fracture-model baselines exist.
- `stale_routes_to_ignore`: presenting v0.3 `ABSTAIN` as success; treating v0.4
  conversation-only prototypes as a reproducible ablation; reviving either
  failed stress-recovery version by deleting wall gates; calling GeoPT-style
  vector-distance transport a fracture model.
- `next_decision_scope`: authorize a 36-trajectory, development-only fracture
  pilot after solver benchmarks pass. No new locked data may be created yet.
- `budget_class`: high scientific effort, moderate local compute for the pilot,
  unknown formal CPU cost until measured. The 12 GB RTX 5070 Ti is sufficient
  for the proposed 1-5 M parameter rollout model; fracture generation is
  primarily CPU-bound.
- `target_scope`: first paper is synthetic, 2D, quasi-static, plane-strain,
  phase-field brittle fracture around three tunnel-section families. It is not
  a dynamic rockburst or field-validation paper.
- `recommended_venue`: Computers and Geotechnics first; CMAME only if the
  mechanics/method contribution and external validation become materially
  stronger.
