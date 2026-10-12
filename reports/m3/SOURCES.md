# M3 primary sources and audit provenance

1. Sugar4D official project page: https://www.ipb.uni-bonn.de/data/sugar4d/index.html
2. Dataset DOI: https://doi.org/10.60507/FK2/IS8YBZ
3. Version metadata: https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ
4. V4 archive: https://bonndata.uni-bonn.de/api/access/datafile/54550
5. V4 readme: https://bonndata.uni-bonn.de/api/access/datafile/58163
6. Descriptor: Bömer et al., *A Dataset of Temporally Consistent Instance Annotations for 4D Plant Phenotyping*, Scientific Data 13, 1325 (September 18, 2026), https://www.nature.com/articles/s41597-026-08326-5
7. Licence legal text: https://creativecommons.org/licenses/by/4.0/legalcode.en
8. Official morphology implementation: Bömer et al., Precision Agriculture 27, 50 (2026), supplementary code archive, https://static-content.springer.com/esm/art%3A10.1007%2Fs11119-026-10337-6/MediaObjects/11119_2026_10337_MOESM2_ESM.zip
9. PyTorch 2.8 documentation fetched with Context7 library resolution `/pytorch/pytorch`, version `/pytorch/pytorch/v2_8_0`; query output preserved in `sources/pytorch_docs.txt`.
10. Matplotlib documentation fetched with Context7 library resolution `/websites/matplotlib_stable`; query output preserved in `sources/matplotlib_docs.txt`.

## Attribution / CC BY 4.0

Sugar4D authors: Jonas Bömer, Elias Marks, Facundo Ramón Ispizua Yamati, Cyrill
Stachniss, Stefan Paulus and Anne-Katrin Mahlein. Original data remain unchanged.
Derived metadata, metre-to-centimetre target conversions, subsampling, anchor
translations and forecasts are this repository's adaptations. Licence permits
sharing/adaptation subject to appropriate attribution, a licence link and
indication of changes; no endorsement is implied. Raw archive is not committed.

## Search and vision limitations

- The requested Bocha live search returned HTTP/API 403: insufficient funds or
  quota. It is recorded in `sources/bocha_search.json`; no hidden search fallback
  to remembered facts was used. Direct official pages, repository APIs, the
  readme, PDF and actual released files were fetched instead.
- 图片内容由 Agnes API 识别。The official layout and targeted crops were sent to
  the user-specified vision service. API outputs are preserved without secrets.
  Plant 020/021 mapping remains provisional: source-image reads both as G2,
  while older supplementary morphology suggests plant020=G3 by trajectory
  matching. This diagnostic is not used to infer/correct ground-truth identities.
- No generated-image service is used for quantitative figures: charts are
  reproducible Python/Matplotlib plots of real measurements and forecasts.
