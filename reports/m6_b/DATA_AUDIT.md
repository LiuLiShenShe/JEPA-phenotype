# Sugar4D supervised label/normalization audit

## Complete official cohort
`data_audit.json` and `data_frames.csv`: 768 frames, 48 whole plants, 24/12/12
plant split with 384/192/192 scans. TrainingDataset filters exactly `split=train`.
No plant crosses subsets. All legal points, leaves, scans and plants are retained.
The source and empirical annotation mapping is `0=taproot/-2`, `1=core/-1`,
`2=leaf/nonnegative ID`; leaf0 is legal, not background.

## Explicit target mapping
The model input is only float32 XYZ. CE supervises distinct taproot/core/leaf
classes on the current training frame. Both nonleaf classes remain in training
and normalization. They are not merged into one semantic stem class.
Instance supervision is leaf-only; frame-local IDs 0..K−1 are independently
permuted each epoch and nonleaves get sentinel−1. Source species0/1,2/3,4/5 are
never passed through as target semantic IDs. Original temporal IDs are not cached
as training tensors. No future frame, true GT leaf count, phenotype, normal or
evaluation segmentation enters the model.

## Geometry/physical reference
Raw metres remain available in original PLY. Retain full-cloud bounding-box
midpoint `center_m` and half maximum span `scale_m`, with exact inverse
`xyz_m=normalized*scale_m+center_m`. This differs explicitly from M6-A's original
centroid/unit sphere S0 and from official source sample-box normalization.
S1/S2 use the same full-box transform; do not relabel this controlled target-domain
adapter as a perfect official end-to-end reproduction.

## Sampling
4096-point uniform sampling, never label-guided. Training uses a fixed at-most
32768-point uniform pool per frame, fresh epoch/seed subsampling and z rotation.
All 384 training frames are visited every epoch. Pools are saved with raw indices,
local labels, center/scale and original vertex count. No pool leaf disappeared;
some 4096-point frame samples can nevertheless miss very small leaves, counted
separately in the error audit. Do not inject GT points to repair these omissions.
Inference samples uniformly from the original full point cloud with the same
plant/time seed as S0, restores nearest-neighbor predicted labels to every vertex,
and scores full-resolution masks with strict IoU>0.5 Hungarian matches.

## Early/small/core-proximity audits
Small threshold is train-only q25 GT leaf-point fraction **0.04588683197817431**.
Early is DAS≤47. Core proximity is a declared **evaluation-only proxy**: at least
10% of leaf points within5mm of current GT core. It is not a manually confirmed
unseparated-leaf diagnosis. Every proxy-positive leaf is retained and its instance
miss/split outcome reported. Lack of core points makes proximity unknown, not a
reason to remove an organ. Held-out labels may be inspected for evaluation/audit,
but are never used for training, augmentation, weight initialization or inference.

## Tests and sources
`tests/test_m6_b.py` checks plant isolation, current local IDs, ignored nonleaf
embedding gradients, independent ID permutation, inverse metres, full-resolution
restoration and rejection of label-bearing inputs. Source: released V4 PLY/readme
and live official metadata in `sources/sugar4d_metadata.json`; official page:
https://www.ipb.uni-bonn.de/data/sugar4d/index.html .
