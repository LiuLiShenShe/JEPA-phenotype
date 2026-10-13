# M6-B source and runtime audit

Checked on **2026-10-13**, using official HTTP retrieval and local byte hashes.
Bocha search returned HTTP403 "You do not have enough money or package quota";
`sources/bocha_search.json` preserves the response. No claim of a successful
general search, and no source/runtime claims based solely on training memory.
The referenced `/home/test/.codex/RTK.md` does not exist on this host.

## Sugar4D V4
- Official page: https://www.ipb.uni-bonn.de/data/sugar4d/index.html
- DOI: https://doi.org/10.60507/FK2/IS8YBZ
- Live metadata: https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ
- Retain the same local V4 as M3–M6-A (2026-08-27 release), CC BY 4.0.
  No upgrade or re-extraction over historical data. All 768 PLY files checked.
- 48 plants ×16 scans, official train24/val12/test12. The held-out Test has been
  used repeatedly before this stage; it is not an untouched blind cohort.
- Current semantics: taproot0/core1/leaf2; actual negative pairs are taproot/-2
  and core/-1. Every legal nonnegative leaf is retained.

## PlantNet and PSegNet
- Official repo: https://github.com/Huang2002200/PlantNet-and-PSegNet
- Fixed commit **5efe9ced49473f0d59c618b7e140ffeeb692d9b1**, identical to M6-A.
  Current GitHub commit query confirms the same revision. Source and both epoch199
  checkpoints are reused read-only; file SHA256s in `sources/retrieval_audit.json`.
- Repository/license API returns no explicit code license. Do not publicly
  redistribute source copies or derived checkpoints without permission. Local
  upstream files under ignored `data/raw/`; generated weights under ignored
  `outputs/m6_b/`. No commit/push performed.
- Source six-way semantic head is three species stem/leaf, not Sugar4D classes.
  PlantNet CONV5 and PSegNet CONV10 are three-way newly initialized heads.
  PlantNet has 4,442,978 parameters and PSegNet 3,339,824 after adaptation.
- Official design and state shapes are retained. S1/S3 use Xavier scratch init;
  S2 loads every compatible state entry, including instance embedding, and
  randomizes only CONV5 weight/bias. No frozen layers; no optimizer state transfer.
- Target environment independently installs Python3.11.15 / Torch2.8.0+cu128,
  not the source Python3.8/Torch2.0.1. Full lock and real batch4/4096-point
  forward/backward/Adam evidence are included. No legacy environment writes.
- Official loss includes CE×10, L1 discriminative×10 and similarity double hinge.
  Retain source's unusual same-instance relu(80-D); do not silently fix a possible
  source bug. Explicit adaptations: leaf-only instance supervision, separate
  taproot/core CE and per-frame group-mask weights. Original helper's batch-global
  local-ID counts are inappropriate for independently remapped frames.
- S2 source pretraining uses tobacco/tomato/sorghum point clouds. Source reports
  546 point clouds, not 546 independently verified biological plants. Exact
  independent pretraining plant/annotation count is unavailable. S0 vs S2 mainly
  tests target-domain supervision, not architecture novelty.

## Organ3DNet
- Official repo: https://github.com/Huang2002200/Organ3DNet
- Fixed commit **06f711fc3d9884933c61766c009b2bcd86837139**, downloaded as official
  archive, fully hashed locally. No explicit repository code license.
- Paper: https://doi.org/10.1016/j.aiia.2025.10.011 . Crossref resolves the exact
  title "Organ3DNet: A deep network for segmenting organ semantics and instances
  from dense plant point clouds". DOI's 2025 suffix is not the journal issue date:
  Crossref reports March2026; its OA license starts 2025-10-26.
- Source requirements: Python3.10.9, Torch1.12.1+cu113, Lightning1.7.2,
  torch-scatter, detectron2 and native PointNet2/MinkowskiEngine. ME fixed revision
  **02fc608bea4c0549b0a7b00ca1bf15dee4a0b228** matches official README.
- Separate `.venv-m6-b-organ3d` installs Python3.10 and exact old Torch/CUDA11.3.
  Both real official native builds fail with available nvcc12.6 vs Torch11.3.
  `organ3d_pointnet2_build.log` and `organ3d_minkowski_build.log` capture full
  compiler diagnostics. No mismatch-check bypass or replacement architecture.
- Official checkpoint: https://zenodo.org/records/17446444 . Metadata lists
  `last-epoch.ckpt`, 481,057,352 bytes, MD5 e3266d804ab27d6ab89d7c0a61cc8ade,
  record CC BY4.0. This checkpoint record license does not license repo code.
  Local download/verification results appear in `organ3d_feasibility.json`.
- **S4 BLOCKED**: official example, strict model loading and target training not
  completed because native dependencies do not build. Report no S4 PQ and no
  invented parameter/training-budget result. Do not port indefinitely.
- Source README reports 889 point clouds (635/254) from five species, not 889
  independent plants. Its recipe includes 10× crop augmentation; these are source
  claims, not Sugar4D training samples.

## Unchanged tracking and measurement
- Tracker: https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- Fixed M6-A commit **3c04919dc86d75acd1a3159a1e1f74893e0bc5e4**. No retraining.
  Keep M6-A background matching, CPD, dummy costs and 256-point geometry sampling.
- Recompute label-independent sample indices and verify exact normalized XYZ
  equality before using M6-A CPD cache. Newly predicted frame IDs are independently
  randomized, and background remains0 just as in M6-A. GT temporal labels appear
  only in evaluator reference links and identity metrics.
- Reproduction found six initial historical frames had the original boundary
  radius10, not the later adaptive radius. `tracking_sampling_diagnostic.json`
  records mismatches from mistakenly using the newer radius. Only those six
  geometry calls force the original radius, then require exact cached XYZ and
  sampled-label equality. Historical caches and scored baselines are never edited.
- Full384-frame geometry equivalence passes; read-only S0 mask replay with new
  independent local numbering exactly reproduces original24-plant PQ/IDF1/ID
  switches/continuity/eventF1/dynamic coverage and severe rates. See
  `tracking_geometry_equivalence.json` and `tracking_replay_equivalence.json`.
- Same `m6a.metrics.measurements` and `m6a.experiment.measurement_losses` for all
  current and historical variants; missing-organs remain failures. Absolute
  length and area remain biased, and reliable masks do not certify phenotyping.

## Current documentation
Context7 resolve→docs requests succeeded for PyTorch2.8 and MinkowskiEngine:
https://github.com/pytorch/pytorch/tree/v2.8.0/docs and
https://github.com/NVIDIA/MinkowskiEngine . Responses are saved in `sources/`.
No credentials are included in queries, saved HTTP evidence or source scripts.
