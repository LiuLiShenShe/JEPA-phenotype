# M3.5 bounded implementation plan

Baseline commit: `5cf34b8d90b5699597f2875a63e40ab9a02dbe37`; initial working tree clean.
Date: October 12, 2026. Existing Sugar4D V4 only; no download, M4, JEPA,
diffusion, commit, merge or push.

## Sequence and acceptance

1. **Audit and protect** (about 5 minutes): read M3 deliverables, raw CSVs,
   decoder and models; hash all pre-existing code, results, processed data and
   local Sugar4D raw files. Save protocol hash before training.
2. **Task/feasibility construction** (about 5 minutes): add
   `src/datasets/sugar4d_organ_dataset.py` and
   `scripts/analyze_m3_5_feasibility.py`; verify all raw PLY/CSV identities and
   centroids, construct a complete missing-aware grid and three-consecutive-
   observation windows, and cumulative first-observation events. Gate training
   on train/validation independent plants and both-class coverage, never test
   performance. Audit labels against manual reference without inventing an area
   validation or a longitudinal noise floor.
3. **Current representation first** (about 10 minutes): train existing M3
   PointNet encoder/head with zero time slots on current XYZ; compare raw
   geometry proxies, train-calibrated geometry Ridge, DAS-only Ridge, and a
   geometry-augmented PointNet. Same partitions and three seeds; diagnose on
   validation before future model training.
4. **Fixed light baselines** (about 10 minutes): add
   `src/models/organ_growth_baselines.py`,
   `src/models/geometry_residual_forecaster.py`, and
   `scripts/train_m3_5.py`. Run C0/C1, C2/C3/C4 joint morphology/visibility
   predictors, C5 current/history × absolute/residual whole-plant predictors,
   and frequency/DAS, logistic, MLP, GRU, repeated-GRU event baselines for each
   feasible task. Equal real/repeated initialization, training budget and
   plant-balanced losses. Save checkpoints and train/validation predictions
   only; no test inference until the fixed selection registry is locked.
5. **Evaluation and delivery** (about 10 minutes): add
   `scripts/evaluate_m3_5.py`, plant-level paired bootstrap statistics,
   per-seed/fold/visibility/phase analyses and source-backed figures. Reload
   checkpoints for final exploratory test inference. Generate the report and
   deterministic recommendation from locked validation rules, not a
   retrospectively best test subtask.
6. **Verify and stop** (about 5 minutes): add `tests/test_m3_5.py` for association,
   missingness, actual scan horizons, leakage exclusions, controls, plant
   weighting, data gates, reload and immutable hashes; run focused and full
   suites, preserve all M0–M3 files, report actual resource use and reproduction
   commands, and end without starting M4.

## Risks and mitigations

- 48 plants are biological repeats, not thousands of independent leaves/windows.
- Official test was used in M3; four-fold models share plants and their
  validation partitions can include official-test plants. All M3.5 design is
  fixed before any such scores, and intervals remain descriptive/exploratory.
- First observed is not biological emergence; disappearance is not confirmed
  senescence. Predict and report future visibility, including missing targets.
- Official height needs a manual growth point and semantic preprocessing;
  XYZ-only proxies deliberately differ. Never read growth points or annotations
  while computing C5 features.
- Published leaf width/length have imperfect destructive-reference agreement;
  area has no manual validation. Report numeric gains and these limits, rather
  than treating every small improvement as biological evidence.
- Do not touch legacy scripts to reuse them; import their audited functions and
  place all new products inside M3.5 paths.

Sources: local files above; public retrieval attempted via Bocha (HTTP 403),
official Sugar4D page, and current Context7 primary documentation saved in
`reports/m3_5/sources/`. No unsourced scientific result is assumed.
