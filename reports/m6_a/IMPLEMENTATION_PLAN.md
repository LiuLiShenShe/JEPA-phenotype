# M6-A implementation plan

This plan and `configs/m6_a/protocol.json` precede baseline inference and evaluation.

1. **Protect and audit** (about 15 minutes): snapshot historical paths, preserve V4, pin official repository trees/checkpoints, record live license and download evidence. New writes are restricted to M6-A paths; no legacy environment installs.
2. **Freeze contracts** (about 10 minutes): use unchanged official plant split; train/validation/test remain 24/12/12 plants. Freeze zero-shot PlantNet, identity/event metrics, measurement definitions, missingness penalties, thresholds and recommendation precedence. No result-driven model selection or training.
3. **Adapt official baselines** (about 30 minutes): isolated environment; load official networks strictly; reuse official CPD, costs, extended matching and 3DEPS. Replace hard-coded filesystem scripts with array adapters. Densify sparse per-frame labels for numerical compatibility, never inject GT temporal IDs. B2 uses B1's exact instances; B2_GT is an explicitly separate diagnostic.
4. **Evaluate and decompose** (about 30–60 minutes): full point-resolution segmentation, all 16 actual scans, identity matching in evaluator only, first-observation events, static/dynamic geometry proxy discrepancies and coverage. Compare GT masks/oracle links, predicted masks/oracle links and tracked links. Aggregate at plant level and retain failures.
5. **Test and deliver** (about 20 minutes): synthetic metric/missingness/new-event tests, GT-ID permutation invariance, repeated inference/label-free contract checks, historical protection verification; generate data-driven tracking/event plots, machine-readable tables and final Chinese report.

## Risks and boundaries
- Official repositories do not expose an explicit code/weight license in GitHub metadata. Public downloadability does not imply a redistribution license: upstream materials stay local under `data/raw/m6_a`.
- Pepper's Google Drive endpoint may be unreachable. Record the actual exception and do not claim pepper or cross-species experiments.
- Official source checkpoints were trained on tobacco/tomato/sorghum. Sugar beet zero-shot transfer is a specific baseline, not a claim that a supervised target-species segmenter cannot work.
- Original TrackPlant3D operates on 256 sampled points; small organs may have no sampled support. Record sampling-drop failures instead of supplying GT-assisted points.
- Sugar4D automatic length/area and PCA proxies are not identical estimands. Report these discrepancies as measurement limitations, not precise biological errors or confirmed deaths.
- M3/M4 previously inspected the official test cohort. These plants are unseen by these fixed models, not an untouched confirmatory dataset.
- GPU work is inference only; do not interrupt unrelated running processes or modify existing environments.

## Verification
New tests live in `tests/test_m6_a.py`. Reproduction entry point is `scripts/run_m6_a.py`; evaluation and source audit outputs live only in `reports/m6_a`, caches only in `outputs/m6_a`.
