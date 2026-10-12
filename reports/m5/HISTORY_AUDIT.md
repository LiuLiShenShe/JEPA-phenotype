# M0–M4 historical audit

M4 decision remains **LIMITED-BENEFIT**, engineering **PARTIAL** because official
P1 weights were unavailable. No M4 report/result is overwritten. M3.5 remains
REPRESENTATION-FIRST and does not authorize temporal-JEPA automatically.

- Verified all 30 historical P0/P2 encoders (5 partitions × 3 seeds × 2
  conditions): strict weight load, saved encoder hashes match full-train
  diagnostic hashes; 384-dimensional token and 768-dimensional mean/max pooled
  features. Count head adds 27 current XYZ descriptors (795 inputs); 21,839,004
  count parameters, no 384/768 dimension mismatch.
- M4 P2: 15 real SSL runs, each 24 train plants × 16 scans = 384 scans,
  batch64, 20 epochs, **120 optimizer steps**, total1800. Encoder remains 12
  blocks/6 heads; predictor192/6/6;64 FPS×32 KNN on2048 points.
- Teacher is the EMA Transformer, not an independently updated full tokenizer.
  Shared tokenizer/position target path is stop-gradient. This matches the
  official port. Teacher has no gradient, EMA equation/count and frozen encoder
  buffers pass rerun tests. No observed EMA implementation defect explains M4.
- SSL/downstream use the same fixed current-scan XYZ cache and normalization;
  full current scan mean/radius, no future transforms. Sampling and patch
  tokenization are XYZ-only. Downstream geometry scaling fits labelled train
  plants only. No observed preprocessing mismatch.
- Keep Sugar4D V4, original48 independent plants,16 scans/plant; original
  official24/12/12 and four fold-local24/12/12 assignments are hash-protected.
  Fold role rotation is not a new globally independent test cohort.
- Full-training M4 mean P0/P2 pooled rank21.155/15.480; token std0.3228/0.0435;
  uncentred global cosine0.9902/0.9996. These show variance/rank contraction and
  anisotropy risk, not a sufficient proof of complete collapse. M5 adds centred
  similarities, identity categories, covariance spectrum and leaf-token probe.
- 50% finetune M4 MAE1.9413→1.5743 across five splits/three seeds was an auxiliary
  finding. M5's50% main target is a **new exploratory hypothesis** informed by
  that observation. Neither historical validation nor previously evaluated test
  plants are described as untouched independent confirmation.
- M4 semantic finetune mIoU0.5539→0.4714; core/taproot negative transfer is a
  prespecified risk. Semantic mIoU is not an instance segmentation metric.

Machine-readable details: `history_audit.json`, `historical_checkpoints.csv`.
Source files: `../m4/M4_REPORT.md`, `../m4/decision.json`,
`../m4/UPSTREAM_AUDIT.md`, `../m4/representation_full_train_diagnostics.json`,
`../m4/label_efficiency.csv`, `../m4/segmentation_metrics.csv`,
`../m3_5/M3_5_REPORT.md`. Current documentation snapshot:
`sources/pytorch_context7.txt`. `/home/test/.codex/RTK.md` was not present.
