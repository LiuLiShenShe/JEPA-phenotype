# Official implementation compatibility ledger

Upstream code and weights are kept unedited in `data/raw/m6_a/official` and are not redistributed.

| Component | Actual issue | Adapter / modification | Consequence |
|---|---|---|---|
| PlantNet/PSegNet PyTorch | Original test scripts contain blank paths, hard-coded CUDA, H5 test/GT loading and output paths | Direct strict state-dict load, array-only XYZ inference; preserve model, six classes, five-dimensional instance embedding, MeanShift 0.6 and public mean-instance filter | No weight training, no target GT in prediction; changed input species and geometry-only uniform sampling explicitly fixed in protocol |
| PlantNet class mapping | Public labels are species-specific: tobacco 0/1, tomato 2/3, sorghum 4/5 | Collapse leaf classes 1/3/5; keep majority cluster leaf labels, consistent with official instance/semantic output | Zero-shot target-species transfer, not target fine-tuning |
| PlantNet input scaling | Official H5 converters were only retrieved successfully after primary partial evaluation; they use bounding-box midpoint/largest positive coordinate rather than frozen unit sphere | Preserve original primary; run separately locked full-resolution segmentation-only official-normalization diagnostic | Report method-fidelity limitation; no original IDF1 replacement, no confirmatory acceptance based on diagnostic |
| TrackPlant3D script paths | Hard-coded four source crops and scan filename parsing | Array CPD + official `build_matrix`, `extend_matrix` + Hungarian; remap each current scan's integer mask IDs and active track IDs to compact local indices | Removes numeric identity/order information; creates unique persistent output IDs outside official sparse dummy columns |
| TrackPlant3D sparse IDs | Released utility sizes cost matrix by max label, potentially hundreds of empty slots | Compact active track IDs before constructing official matrix | Avoids empty-label dummy artifacts; weights and dummy policy unchanged |
| 3DEPS boundary runtime | Radius=10 m exceeds the entire Sugar4D plant; tensor hybrid search spends 82–111 seconds on 195k–225k points | Radius becomes `min(10, 1.001*max(all-point 20th-neighbor distance)+1e-7)`; the exact same 20-neighbor hybrid search is requested within the radius | Geometry-only, mathematically equivalent max_nn=20 neighborhood; record mask agreement check before continuing. Initial six cached frames use original radius |
| 3DEPS random pools | Source scripts seed edge/core inconsistently with directory order | Fixed per-plant/per-DAS seed, 4096 draws per edge/core, original FPS 128+128 and current-scan normalization | Reproducibility adaptation; no organ-aware sampling |
| 3DEPS empty partition | Boundary/core may be empty | Whole-cloud fallback only when a partition is empty, record explicit flag | Engineering fallback, never GT-assisted supplementation |
| 256-point sampling | Some full-resolution masks receive no tracking support | Leave track ID=-1 and count failure; no label-stratified resampling | B0 is not a perfect tracking upper bound |
| B2 | Needs non-learning baseline on same instances | Full-mask current centroids; gated Hungarian 0.25, no CPD | B2/B1 share exact prediction masks; different geometric tracker |
| Environment | Python 3.8/Torch 2.0.1 vs local Python 3.11/Torch 2.8; Open3D missing | `.venv-m6-a`, local Open3D 0.19 and frozen geometry dependencies, read-only access to historical torch via `.pth` | Historical environment packages are untouched; smoke tests check strict checkpoint load and deterministic inference |

No upstream copyright/license headers or algorithm weights were changed. All edits are in new M6-A adapter files.

A trial vectorized boundary implementation reproduced the 15,033-point mask but differed at 8/93,957 points on a second scan (neighbor ties/rounding). It was rejected before scored evaluation and removed; all experiment caches use Open3D's original boundary kernel. It is not used as a faster new method.

A later check with Open3D KNN found 10/93,957 and7/224,602 differing boundary flags because KNN and hybrid neighbor sets differ in tied-distance cases (`boundary_numerical_analysis.json`). This alternative was also rejected. Baseline caches continue to use the original hybrid-search boundary kernel, not a vectorized replacement.

Tracking label0 is the per-frame aggregate non-leaf/background organ (GT taproot/core combined for B0, non-leaf/rejected clusters for B1). It is retained as geometric context in matching, consistent with the official whole-plant organ tracker, but not counted as a leaf in PQ/identity/events/phenotypes. Its numeric ID is not a leaf time-identity truth.
