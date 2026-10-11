# M1 implementation plan

1. Freeze the M0 file inventory; validate every window, original RGB/mask hash,
   plant assignment, actual timestamp difference and geometry. Do not rebuild M0.
2. Create `.venv-m1` with Python 3.11 and the official PyTorch 2.8 / torchvision
   0.23 CUDA 12.8 pair. Verify both A6000 devices, tensor operations and real models.
3. Extract official frozen DINOv2 ViT-S/14 embeddings with aspect-preserving
   bicubic letterboxing, 224 square inputs and ImageNet normalization. Bind cache
   rows to RGB SHA-256, source commit, weight SHA-256 and preprocessing hash.
4. Run persistence, actual-time linear extrapolation and train-fitted polynomial
   time-only regression. Tune time-only regularization on validation plants only.
5. Tune the two RGB heads on horizon 4 validation only, lock the choices, then
   train separate horizon 2/4/8 models with seeds 42/43/44 on one A6000. Use
   train-only target/time scaling, Huber loss and inverse plant-window weighting.
6. Freeze model selection before test inference. Save configs, checkpoints,
   prediction CSVs, per-run metrics and plant-cluster bootstrap confidence intervals.
7. Test leakage boundaries, cache integrity, shapes, devices, checkpoint reloads,
   repeated inference and metric definitions. Recheck M0 hashes, plot all seven
   requested diagnostics, document failures and give PASS/PARTIAL/FAIL and
   CONTINUE/REVIEW/STOP independently. Commit only on the M1 branch.

## Boundaries and risks

- No JEPA, 3-D, raw-data downloads, new splits or exception-policy changes.
- Elapsed observation time is not plant age since sowing. Future wall-clock
  difference is an explicit query condition, not a fixed mapping from horizon.
- Leaf count and projected image-occupancy ratio are the two targets. Ratio is
  not physical leaf area or LAI. Preserve difficult plants and all M0 windows.
- Overlapping windows are not independent; confidence intervals resample plants.
- B0/B1 are oracle-phenotype models and must remain distinct from RGB-only heads.
- M0 reports raw observations as 17,082, of which exactly 17,056 are valid.
- Official documentation is fetched with Context7 and official source inspection.
  Bocha search returned HTTP 403; no data pixels are sent to external services.
- Expected runtime: environment/source setup minutes; cached encoder inference
  minutes; small head sweeps tens of minutes. Record actual durations, not estimates.

## Verification map

- Data: full local metadata assertions, RGB/mask manifest hashes, M0 snapshot hashes.
- Features: unique paths, row hashes, all-finite float32 matrix, real re-extraction.
- Models: 3-frame order, latest-only MLP, independent labels, train-only normalizers.
- Selection: validation-only candidates and explicit immutable selection registry.
- Outputs: 5 models x 3 horizons, 18 neural runs, 46 unseen test plants per run.
- Figures: metric source CSVs and plant-based CIs; deterministic plant selection.
