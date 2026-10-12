# M4 implementation plan

Start date: 2026-10-12. Scope stops before Temporal-JEPA.

## Dependencies and protected scope

- Read audited Sugar4D V4 metadata and M3 plant assignments without changing them.
- Use a new `.venv-m4`; retain the installed driver, system CUDA and old environments.
- Pin official Point-JEPA source and inspect the real checkpoint before promising P1.
- Official Point-JEPA/Point-MAE are MIT; preserve their licence with source snapshots.
- No existing M0--M3.5 source, data, outputs or reports may be modified.

## Steps and verification

1. **Audit resources and freeze protocol.** Save actual upstream source, DOI metadata,
   dependencies, licence, download failures, protection hashes, plant splits and nested
   6/12/24-plant label budgets. Lock primary validation MAE, practical thresholds and
   selection before any Sugar4D model fitting. Cost: network and data reads.
2. **Port official architecture in isolation.** Preserve official tensor names/shapes,
   patch PointNet, positional embeddings, transformer and latent predictor; replace
   old PyTorch3D grouping and Lightning orchestration with current PyTorch equivalents.
   Test forward/backward, masks, target EMA, shapes, checkpoint/RNG resume, deterministic
   runs, a fixed-mask tiny overfit and upstream numerical parity. Cost: minutes.
3. **Prepare current-scan data.** Separate XYZ-only SSL cache from supervised labels;
   current-scan centering/radius transforms are invertible, physical XYZ descriptors
   remain available. Test plant isolation, absent SSL labels and label-budget nesting.
4. **Run train-only SSL.** Three seeds and five independent partitions; fixed final
   epoch, not test-selected. Save full states, curves, std/cosine/effective-rank
   diagnostics, steps/time/VRAM/sample counts. Cost: GPU-dependent, measured rather
   than extrapolated as completed work.
5. **Evaluate representations.** P0/P1/P2 use identical heads and budgets; leaf-count
   frozen linear probe and equal-budget full fine-tuning at 25/50/100% plants. Run
   three-class point segmentation with XYZ-only interpolation plus a supervised
   PointNet. Include unchanged M3.5 PointNet+Geometry and geometric controls. Three
   seeds, four folds and official validation; no test fitting or selection.
6. **Finalize evidence.** Aggregate independent-plant metrics, locked paired comparisons,
   stage/failure cases, real-data plots and segmentation visualizations. Verify all
   old artifact hashes and automated tests. Report limitations/missing conditions
   explicitly; never turn missing runs or declining SSL loss into success.

## Risks and handling

- Google Drive checkpoint may be inaccessible: record P1 as unavailable, never random
   weights labelled official. Report engineering acceptance as incomplete if required.
- Legacy source bugs and CUDA nondeterminism: minimal isolated fixes with explicit
   equivalence tests and a compatibility ledger, not a new unreported architecture.
- Both GPUs already have jobs: use only one and never stop unrelated processes.
- Small independent sample size and reused official test: prioritize locked validation
   and plant-level uncertainty; do not claim a new untouched confirmatory test.
- Full official backbone is expensive: run the preregistered finite budget, document
   actual costs; no open-ended architecture/test-driven search.
- Leaf instance segmentation is an explicit deferred extension, not semantic mIoU.

## Deliverables

`src/models/point_jepa_*.py`, `src/datasets/sugar4d_ssl_dataset.py`, isolated
`src/m4/` orchestration, M4 scripts/config/tests, checkpoints/predictions under
`outputs/m4/`, and `reports/m4/M4_REPORT.md` plus all requested tables and plots.
