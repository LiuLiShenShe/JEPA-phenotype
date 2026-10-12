# M5 implementation plan

Date: 2026-10-12. Scope: one bounded, exploratory representation experiment.

## Progress

- Historical review and checkpoint protection: completed.
- Actual Crops3D and Bonn acquisition/cache integrity: completed; physical
  identity/native-unit completeness explicitly remains PARTIAL.
- M5-A 0/120/500/2000-step diagnostic: completed.
- Protocol locked before formal comparisons; documented train-only budget
  amendment fixes 500 steps and enables one conditional P4 loss.
- Formal matched matrix: completed (45 SSL, 180 new downstream runs plus
  120 unchanged historical controls, 75 probes). All formal SSL runs reach
  the locked 500 steps; no extra loss/weight trial is run.
- Final regression: 147 tests pass; 39 M4/M5 contracts and 18 focused M5 tests
  pass. All 180 new prediction checkpoints reload successfully.
- Post-training external-source/species diagnostic: 15 official encoders,
  three seeds and all five conditions, completed read-only. No model selection,
  protocol gate or training configuration changes follow these diagnostics.
- Final paired/representation/semantic decision: **JEPA-STOP**. Seven editable
  quantitative figures, complete report, acceptance, provenance and protected
  45,614-file verification are complete. Engineering remains PARTIAL due to
  source native-unit/physical-identity uncertainty and disclosed fold roles.
- M5 is stopped. No Temporal-JEPA, future prediction, Diffusion, next-stage
  task, Git commit, merge or push is performed.

## Dependencies

- Preserve all pre-existing data, code, checkpoints, splits, and reports.
- Reuse the M4 384-dimensional, 12-layer encoder and 192-dimensional,
  6-layer predictor; 2048 XYZ points, 64 FPS groups, 32 neighbours.
- Obtain actual external geometry with an explicit licence and integrity check.
- Lock the M5 evaluation and conditional regularization rules before P3/P4.
- Use the existing isolated `.venv-m4` environment and physical GPU 0 only.

## Steps

1. **Historical audit and protection** (`scripts/prepare_m5.py`,
   `reports/m5/HISTORY_AUDIT.md`). Recheck M4 checkpoints, EMA, dimensions,
   sampling, split provenance, and archival results. Verify protected-file hashes.
   Estimated duration: 5–15 minutes, including hashing.
2. **External source preparation** (`src/datasets/plant_ssl_dataset.py`,
   `scripts/prepare_m5_data.py`). Save official metadata and acquisition logs;
   download with published checksums; identify canonical scans, physical identity
   uncertainty, crop species, acquisition methods, label-assisted cropping, and
   duplicated derived versions. Exclude known/repeated geometry and use only
   XYZ in SSL caches. Estimated duration: network dependent; bounded retries.
3. **M5-A budget diagnostic** (`src/m5/pretrain.py`,
   `src/m5/diagnostics.py`). Run one fixed training split/seed at 120, 500,
   and 2000 steps with intermediate full state and encoder checkpoints. Report
   centred/uncentred similarities by identity category, variance, covariance,
   pooled/token effective ranks, and loss. Stop at a predeclared degradation
   boundary rather than indefinitely extending. Estimated duration: measured
   from GPU throughput, not guessed as a scientific budget.
4. **Lock and controlled M5-B/C runs** (`configs/m5/protocol.json`,
   `reports/m5/protocol_lock.json`, `src/m5/downstream.py`). Fix one SSL step
   budget using only M5-A training diagnostics. Compare historical P0/P2,
   matched-step Sugar4D-only SSL, P3, and one conditional P4. Match downstream
   architecture, subsets, initial heads, seeds, and training budget. Do not
   select pretraining checkpoints using downstream outcomes. Estimated duration:
   actual measured SSL/downstream throughput times the locked matrix.
5. **Evaluation and delivery** (`src/m5/evaluation.py`,
   `scripts/evaluate_m5.py`, `tests/test_m5.py`). Plant-paired bootstrap, three
   seeds, four-fold directions, nested 6/12/24 labels, class-specific semantic
   IoU, train-only leaf-identity geometry-matched representation probe,
   checkpoint reload, quantitative figures with source CSVs, full report,
   acceptance and final protection check. Stop after M5. Estimated duration:
   15–30 minutes plus inference and regression tests.

## Risks and safeguards

- Bocha search is currently quota-exhausted (HTTP 403). Record failure, do not
  pretend it succeeded; verify user-provided official endpoints directly.
- Crops3D has raw, FPS, corruption, and plot copies. These are not independent
  plants. Do not multiply the data count using derived views or augmentations.
- Physical plant identities or units may not be completely documented. Preserve
  uncertainty; do not label 1230 scans as 1230 verified independent plants.
- BonnBeetClouds3D crops may depend on plant-instance annotations. Disclose
  annotation-assisted preprocessing and exclude it if safe identity resolution
  or acquisition is infeasible within this bounded run.
- Previously inspected Sugar4D outcomes cannot become an untouched confirmation
  set. Explicitly label M5 as a newly locked exploratory hypothesis.
- High cosine alone is anisotropy, not proof of complete collapse. Loss decline
  alone is not success. Use centred features, variance, ranks, leaf probes,
  and downstream behaviour jointly.
- No Temporal-JEPA, future plant prediction, Diffusion, git merge or push.

## Verification

Tests cover split leakage rejection, XYZ-only SSL inputs, exact/canonical
duplicates, balance sampling, finite statistics and constant-feature collapse,
EMA/feature shapes, locked protocol tamper detection, conditional-loss gating,
plant-level paired statistics, checkpoint predictions, and archive protection.
All incomplete experiments remain explicitly unavailable; no placeholder metrics
or synthesized independent plants are permitted.
