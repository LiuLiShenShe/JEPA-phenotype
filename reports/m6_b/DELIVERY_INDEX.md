# M6-B delivery index

- `M6_B_REPORT.md` and `decision.json`: actual supervised results and one research recommendation.
- `protocol_lock.json`, `configs/m6_b/protocol.json`: pretraining frozen metrics/budgets/selection/continuation gates.
- `SOURCE_AUDIT.md`, `sources/`, `organ3d_feasibility.json`: fixed official revisions, licensing, native blocker and checkpoint hashes.
- `DATA_AUDIT.md`, `data_audit.json`, `data_frames.csv`, `data_leaves.csv`: complete768-frame labels, splits, small/proximity proxies and physical metres.
- `training_summary.csv`, `validation_learning_curves.csv`: all seven actual60-epoch runs and loading/budget/convergence evidence.
- `segmentation_summary.csv`, `segmentation_per_plant.csv`, `segmentation_frame_metrics.csv`, `segmentation_leaf_errors.csv`: full-resolution macro scores and every legal leaf's errors.
- `paired_plant_intervals.csv`, `seed_mean_plant_pq.csv`, `error_strata_summary.csv`, `pq_by_das.csv`, `pq_by_leaf_count.csv`: plant bootstrap, seed handling and registered strata.
- `tracking_per_plant.csv`, `tracking_summary.csv`, `tracking_paired_intervals.csv`, `tracking_error_decomposition.csv`: frozen tracker and identical measurer propagation.
- `first_observation_events.csv`, `first_observation_strata_per_plant.csv`, `first_observation_strata_summary.csv`: every legal new-leaf event, detection-vs-identity loss and evaluation-only size/core strata.
- `tracking_geometry_equivalence.json`, `tracking_replay_equivalence.json`: exact384-frame XYZ/sample-label agreement and24-plant historical S0 numerical replay.
- `inference_runtime.csv`, `inference_microbenchmark.csv`, `inference_repeatability.json`: actual shared-host timing/VRAM; same-device read-only repeat check when executed.
- `figures/`, `FIGURE_CONTRACT.md`, `FIGURE_QA.md`, `figure_manifest.json`, `figure_source_preflight.json`: source-grounded PNG/SVG/PDF and direct-image QA.
- `TESTING.md`, test/build logs, `protection_verification.json`, `final_verification.json`: focused/historical regression, unchanged files/environments/HEAD and completion checks.
- `REPRODUCE.md`, `RESOURCE_SCHEDULING.md`, `scripts/*m6_b*`, `src/m6b/`, `tests/test_m6_b.py`: actual reproduction and scheduling/lock rules.

Large/local-only outputs: `outputs/m6_b/S*/seed_*/` contains `training_manifest.json`, every-update `batches.csv`, per-epoch `epochs.jsonl`, latest/best/evaluation-epoch checkpoints, full-resolution predictionNPZ, postprocessing selection and full observations/events/measurement errors. Every legal point is included in scored full-resolution labels; displayed subsets do not replace them.
`outputs/m6_b/data/` holds current-frame sample indices/local supervision and reversible centre/scale; `tracker_geometry/` holds384 geometry-only indices.
Upstream source/checkpoints remain ignored under `data/raw/m6_a` / `data/raw/m6_b`, and derived weights/predictions remain ignored under `outputs/m6_b`. No redistribution, commit, branch, merge or push.
