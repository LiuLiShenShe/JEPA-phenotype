# Resource and compatibility audit (2026-10-12)

## Point-JEPA

- Official repository: https://github.com/Ayumu-J-S/Point-JEPA
- Retrieved revision: `c36b51ca20a07f3fa3c75f6b9d3bc0e29b45401c` (GitHub tree API).
- Paper: https://openaccess.thecvf.com/content/WACV2025/html/Saito_Point-JEPA_A_Joint_Embedding_Predictive_Architecture_for_Self-Supervised_Learning_on_WACV_2025_paper.html
- Licence: MIT; complete upstream licence and source snapshot in `third_party/m4/point_jepa/`.
- Actual entry: `python -m pointjepa fit -c configs/Point-JEPA/pretraining/shapenet.yaml`.
- `install_py.sh`: PyTorch 1.13.1 / torchvision 0.14.1; pinned PyTorch3D commit
  `799c1cd21beff84e50ac4ab7a480e715780da2de`; requirements specify Lightning 1.9.5,
  timm 0.6.7, lightning-bolts 0.7.0, NumPy 1.23, W&B. README asks Python 3.10.4,
  CUDA >=11.6, cuDNN >=8.4, GCC 6--11.2.1. Do not run installer globally.
- Official encoder: 64 FPS/KNN patches, 32 points each, mini PointNet, position MLP,
  384 dimensions, 12 transformer blocks, 6 heads; predictor 192/6/6. ShapeNet uses
  1024 input points and 500 epochs. This M4 protocol uses 2048 and 20 epochs.
- Weight URL is Google Drive; Lightning format normally `state_dict` with `tokenizer`,
  `positional_encoding`, `student`, `predictor`, and teacher/EMA state. Actual official
  checkpoint download failed. No checkpoint format/coverage is claimed as observed.
- README identifies released weights as the best of ten runs, not a typical seed.
- Raw source retrieved via codeload because GitHub git transport timed out.

## Minimal isolated port and explicit departures

- Port PointNet, attention/MLP/transformer and predictor equations with exact state
  names. CPU reference tests verify bit-exact eval output and strict state matching
  against the actual source for PointNet, transformer and predictor.
- Replace PyTorch3D FPS/KNN with deterministic pure PyTorch FPS/KNN. Start FPS at
  farthest-from-origin, instead of upstream random-start FPS. Preserve patch geometry.
  Cache label-independent group indices for computational comparability.
- Replace timm DropPath with the same Bernoulli residual-drop operation; no timm or
  torchvision import is needed. Replace Lightning/W&B/bolts with explicit local loops.
- Remove hardcoded `.half()` in context forward: FP32 works on CPU and GPU. No
  misleading full official training reproduction is claimed.
- Shared patch tokenizer and position MLP follow official code; EMA applies only to
  context transformer. Targets use last pre-final-norm block then functional layer
  norm, stop-gradient, teacher eval. Predictor's unused `predictor_norm` is retained
  for checkpoint compatibility (same as upstream).
- Reject empty context instead of official context sampler's illegal-index fallback.
  The unused upstream random-context branch returns indices relative to valid-array
  positions rather than original token positions; this port uses registered contiguous
  selection only and tests disjointness. Target blocks may overlap each other.
- EMA updates after each optimizer step, with a locked monotone 0.9995--0.99999
  schedule. Upstream EMA has initial-copy/update bookkeeping; this simplified timing
  is disclosed, checkpointed and equation-tested, not asserted identical.
- Normalize only current scan, preserve inverse transform and full-scan physical
  geometry; rotation is about plant vertical Z, not object-dataset Y.
- No pixel/XYZ reconstruction, variance regularizer or new self-supervision objective.
  Collapse is diagnosed, not assumed impossible from EMA alone.
- Weighted point cross-entropy in `B x C x N` dispatches to a CUDA 2D NLL kernel
  without deterministic implementation in PyTorch 2.8. The initial segmentation
  smoke run stopped rather than disabling determinism. Flattening to `(B*N) x C`
  and `(B*N)` is mathematically equivalent and passes deterministic GPU regression.
- Frozen segmentation point features are held on the one selected GPU when they
  fit (~1.8 GiB Train+Val), avoiding repeated large CPU indexing/PCIe copies. Dtype,
  batch order, number of optimizer steps and head equations are unchanged. This is
  only a transport optimization, not an extra learned feature or new supervision.
- Early downstream smoke checkpoints had a final-state capture placement error;
  validation-selected `best.pt` and predictions were correct and are never replaced.
  `scripts/repair_m4_state_capture.py` deterministically replays identical settings,
  requires identical best epoch/predictions, preserves the original training snapshot,
  and then fills the corresponding final weights+optimizer state. Extra replay cost
  is separately logged, not counted as more scientific seeds or observations.

## Point-MAE

- Paper: https://arxiv.org/abs/2203.06604
- Official repo: https://github.com/Pang-Yatian/Point-MAE ; MIT licence retained in
  `third_party/m4/point_mae/` with README and requirements.
- Actual entry `python main.py --config cfgs/pretrain.yaml`; reconstruction pretrainer
  is `models/Point_MAE.py`, with `tools/runner_pretrain.py` and custom KNN/Chamfer CUDA
  extensions. Segmentation has its own `segmentation/main.py` and pretrained loading.
- Official model-zoo `.pth` releases are not interchangeable with Point-JEPA Lightning
  checkpoints. Different key names/objectives require a separate verified adapter.
- Inspected upstream `tools/builder.py`: actual saved training dictionary keys include
  `base_model`, `optimizer`, `epoch`, `metrics`, `best_metrics`; encoder weights in
  `models/Point_MAE.py` include `MAE_encoder` and decoder/reconstruction-specific keys.
  No actual Point-MAE weight download/loading is claimed in this M4 run.
- Optional P3 is omitted: custom extension compatibility and comparable budget were
  not established. No Point-MAE number is invented.

## Plant-MAE

- DOI: https://doi.org/10.1016/j.plaphe.2025.100049
- Crossref verified title: *Automated 3D Segmentation of Plant Organs via the Plant-MAE:
  A Self-Supervised Learning Framework*, 2025, Kai Xie et al.
- Publisher full text returns HTTP 403. This prevents full method/result verification.
- Candidate https://github.com/tigeraBCv/Plant-MAE at
  `a834caa6cbe5c70286f8ef54b5534dfeee26897a` contains only an 11-byte README and no
  code/weights/licence/dependencies. Relation to paper cannot be authenticated from
  this placeholder. It is not a reproducible implementation and is not reused.
- Plant-organ masked pretraining therefore already has relevant prior work; do not
  claim M4 is the first plant 3D SSL or infer unverified quantitative superiority.

## Sugar4D

- https://doi.org/10.60507/FK2/IS8YBZ
- Authoritative API: https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ
- Live metadata confirms version 4, CC BY 4.0; retained in `sources/sugar4d_v4_metadata.json`.
- Authors: Jonas Bömer, Elias Marks, Facundo Ramón Ispizua Yamati, Cyrill Stachniss,
  Stefan Paulus and Anne-Katrin Mahlein. Existing V4 audit and raw data remain unchanged.
- 48 independent plants, 16 scans each. Official Train/Val/Test = 24/12/12 plants;
  M3 four-fold assignments reused independently, not pooled SSL pretraining.
- Semantic 0/1/2 = taproot/core/leaf. Current visible count = unique annotated leaf
  instances in this scan; not botanical total leaf number or future organ identity.

## Search/documentation limitations

- User-requested Bocha returned HTTP 403, insufficient money/package quota. Sources
  were fetched from official repositories/papers/DOI APIs, not inferred from memory.
- Context7 resolved PyTorch `/pytorch/pytorch`; version-specific 2.8 docs preserved in
  `sources/pytorch_context7.txt`. Old global Python/CUDA/driver were not modified.
- User-referenced `/home/test/.codex/RTK.md` does not exist in this environment.
- Quantitative plots are deterministic charts of real outputs, not generated imagery.
