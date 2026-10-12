# M3.5 sources and provenance

## Scientific/data sources

- Official Sugar4D page, fetched October 12, 2026 (HTTP200):
  https://www.ipb.uni-bonn.de/data/sugar4d/index.html ; local receipt
  `sources/official_page.txt`.
- Fixed existing V4 DOI: https://doi.org/10.60507/FK2/IS8YBZ . Archive,
  byte count and author-supplied MD5 are inherited from the immutable M3 audit,
  not inferred from current website claims and not redownloaded.
- Descriptor and official morphology extraction source: preserved in
  `reports/m3/SOURCES.md`, `reports/m3/sources/paper.txt` and
  `reports/m3/sources/morphology_definitions.json`; this milestone verifies
  raw measurement identities/values and all 768 current PLY associations.
- Local empirical claims derive from `feasibility.json`, saved predictions,
  plant-level error CSVs, validation selection lock and final acceptance.
  Original measurement files/PLYs are unchanged.

## Documentation

Context7 resolution followed by docs retrieval (no hidden training-data API
guess). Exact fetched passages and source URLs are preserved locally:

- PyTorch installed 2.8.0, `/pytorch/pytorch/v2_8_0`;
  `sources/pytorch_docs.txt`, including
  https://github.com/pytorch/pytorch/blob/v2.8.0/docs/source/notes/modules.rst
  and serialization docs.
- scikit-learn installed 1.7.2, documentation nearest available 1.7.1,
  `/scikit-learn/scikit-learn/1.7.1`; `sources/sklearn_docs.txt` including
  https://github.com/scikit-learn/scikit-learn/blob/1.7.1/doc/getting_started.rst .
- SciPy `/scipy/scipy`, `sources/scipy_docs.txt`, official spatial tutorial
  and implementation; installed version recorded in `environment.json`.
- Matplotlib `/matplotlib/matplotlib`, `sources/matplotlib_docs.txt`,
  official figure/axes/savefig documentation and source.

## Retrieval/vision limitations

Bocha live search returned **HTTP403, insufficient balance/package quota**,
recorded in `sources/bocha_search.json`. Search was attempted, not claimed
successful; scientific evidence is direct official retrieval and real local
files. The Dataverse V4 metadata endpoint timed out on this date; version
provenance uses the existing verified M3 V4 receipt instead. Missing optional
`/home/test/.codex/RTK.md` and absent `.codegraph/` index were recorded, not
silently assumed present.

Quantitative figures are deterministic Python plots of real measurements and
predictions, not generated-image output. External visual QA, when available,
uses the user-authorized Agnes API, with the explicit attribution
**图片内容由 Agnes API 识别**. Failed calls are saved and never treated as
successful visual inspection.

Sugar4D CC BY4.0 attribution: Jonas Bömer, Elias Marks, Facundo Ramón Ispizua
Yamati, Cyrill Stachniss, Stefan Paulus and Anne-Katrin Mahlein. Original data
remain untouched; derived trajectories, geometric proxies, task labels,
train-only calibration and forecasts are this project's adaptations. No
endorsement by data authors is implied.
