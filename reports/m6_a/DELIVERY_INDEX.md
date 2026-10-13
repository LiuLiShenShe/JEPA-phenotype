# M6-A delivery index

## Interpretation
- `M6_A_REPORT.md`: main Chinese report, answers to five research questions and unique `INSTANCE-FIRST` recommendation.
- `decision.json`: fixed-primary failed acceptance, no training/cross-species claim, final stop boundary.
- `configs/m6_a/protocol.json` / `protocol_lock.json`: original pre-inference protocol and hashes. Never replaced by later diagnostics.
- `NORMALIZATION_DIAGNOSTIC.md`: transparently explains post-lock official preprocessing discrepancy; segmentation-only diagnostic, not a new primary.

## Resource/data audit
- `SOURCE_AUDIT.md`, `source_audit.csv`, `downloaded_files.csv`, `data_feasibility.json`.
- `sources/`: official metadata/README/tree/commit, source blobs, licenses404, actual Bocha403 and Pepper network failures, Context7 docs.
- `COMPATIBILITY.md`: numerical and interface changes, plus rejected non-equivalent optimizations.

## Results
- `baseline_metrics.csv`, `per_plant.csv`, `frame_metrics.csv`, `paired_plant_intervals.csv`.
- `observations.csv`, `predicted_organs.csv`, `new_leaf_events.csv`, `observation_status_grid.csv`.
- `measurement_errors.csv`, `error_decomposition.csv`, `transition_failure_types.csv`, `severe_failures.csv`.
- `propagation_common_support_summary.csv` / `propagation_common_support_per_plant.csv`: explicitly post-hoc artifact diagnosis; does not replace main acceptance.
- `manual_length_comparison.csv` / `manual_length_per_plant.csv` / `manual_length_summary.csv`: final-DAS manual vs baseline length, with coverage and automatic/manual reference discrepancy.
- `confidence_summary.csv` / `confidence_per_plant.csv`, `prediction_confidence.csv`, `confidence_selective_risk.csv`.
- `official_normalization_frame_metrics.csv` / `official_normalization_per_plant.csv` / `official_normalization_metrics.csv`: same models/samples under official preprocessing, full-resolution segmentation only.

## Figures and verification
- `figures/`: six figures in PNG300dpi, SVG and PDF; actual point clouds and numeric identities, not AI-generated anatomy.
- `FIGURE_CONTRACT.md`, `FIGURE_QA.md`, `figure_source_preflight.json`, `figure_manifest.json`.
- `tests_m6_a.log/xml`, `tests_all.log/xml`, `tests_m1_original_environment.log/xml`, `TESTING.md`.
- `final_verification.json`, `real_frame_repeatability.json`, `cache_manifest.csv`, `protection_verification.json`, `historical_snapshot.json`.
- `environment.json`, `environment-lock.txt`, `REPRODUCE.md`, original runtime/inference logs.

Large derived caches stay in ignored `outputs/m6_a`; upstream code/weights remain ignored under `data/raw/m6_a/official`. No M0–M5 file, original environment, Git commit/branch or remote state is changed.
