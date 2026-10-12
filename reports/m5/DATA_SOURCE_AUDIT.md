# M5 data source audit

Date:2026-10-12. Real external acquisition completed; no augmentation copies are
counted as new independent plants. Data-source identity/unit completeness is
**PARTIAL**, while published-byte/CRC/cache integrity checks are PASS.

## Official sources and fixed versions

- Crops3D: https://github.com/clawCa/Crops3D , fixed utility revision
  `cf5ed2ab45fc78d84cad6ec4fe296c822eb5d759`. Official data DOI https://doi.org/10.6084/m9.figshare.27313272 .
  Article https://www.nature.com/articles/s41597-024-04290-0 . DataCite confirms
  **CC0-1.0**, distinct from article copyright. Pin downloaded **article v1**;
  do not claim it is the latest unobserved release. Native `Crops3D.zip` exactly
  9,289,027,713 bytes matches DataCite. Outer download 9,289,027,995 bytes contains
  that file; outer is not another dataset. Full inner ZIP CRC pass, archive
  SHA256 in data_preparation.json. Metadata API repeatedly403, official article
  downloader200. Published Figshare MD5 could not be retrieved; no invented MD5.
- BonnBeetClouds3D: https://doi.org/10.60507/FK2/34W30T ; official Bonndata **V1.0**
  released2024-01-29, file`BonnBeetClouds3Dv0.zip`, CC0-1.0. Official MD5
  `a8f7aa642db818e5e4e02e9e3d16f584` matches actual. 11,149,363,555 bytes; SHA256 fixed in preparation.
  The paper https://arxiv.org/abs/2312.14706 is currentlyv2(2025-08-11), not a
  licence/version change to the downloaded dataset.
- Sugar4D: originalV4/CC BY4.0/DOI10.60507/FK2/IS8YBZ is not redownloaded or
  relabelled.48 physical plants×16 scans. Original official/fold assignments and
  raw files are protected. Terrestrial LiDAR,2021-10-26—2021-12-21,Göttingen.

## Admitted geometry and physical identity

| source | species | ssl_objects | independent_physical_plants | method | training_probability |
| --- | --- | --- | --- | --- | --- |
| BonnBeetClouds3D | sugar_beet | 128 | 128 released plant IDs; one fragment each | UAV camera + SfM | 0.2500 |
| Crops3D | Cabbage | 137 | 14 identified across repeated scans | structured light / MVS | 0.0312 |
| Crops3D | Cotton | 123 | unknown; not equal to scan count | FARO Focus S70 TLS | 0.0312 |
| Crops3D | Maize | 157 | unknown; not equal to scan count | FARO Focus S70 TLS | 0.0312 |
| Crops3D | Potato | 82 | unknown; not equal to scan count | FARO Focus S70 TLS | 0.0312 |
| Crops3D | Rapeseed | 105 | unknown; not equal to scan count | FARO Focus S70 TLS | 0.0312 |
| Crops3D | Rice | 58 | unknown; not equal to scan count | FARO Focus S70 TLS | 0.0312 |
| Crops3D | Tomato | 58 | unknown; not equal to scan count | SfM greenhouse | 0.0312 |
| Crops3D | Wheat | 103 | unknown; not equal to scan count | FARO Focus S70 TLS | 0.0312 |

Total external951objects=823Crops3D Train scans+128Bonn Train identity fragments.
The Crops3D archive has1180canonical raw single-plant PLY, plus50plot PLY; these
explain1230total samples without implying1230independent plants. Canonical Train
list has823/1180.10k/FPS copies,corruption HDF5,plot-scale IS and all externalTest
files are excluded from SSL; no test annotations are used for model selection.

**Cabbage:**14plants were scanned weekly2022-09-01—2022-12-05 with Wiiboox
Reeyee-Pro structured light and enhanced CasMVSNet MVS/Canon77D. Filename
`mvs/sl_date_plantID` lets M5 group date/mode scans of the same plant.137Train
scans remain technical observations; physical IDs are sampled uniformly before
choosing a scan. Other crops lack a verified identity schema; numeric filename
identifiers must not be called independently verified biological replicates.
Tomato:83raw/58Train samples; repeated2020-08—10greenhouse acquisitions in
Liaocheng,Shandong via SfM/3DPhenoMVS. Other6species useFARO Focus S70TLS; maize
multi-stage acquisition can revisit plants. Record unknown identities, rather
than fabricate a full independent-plant total.

**Bonn:**UAV imagery/SfM, real field breeding trial,2022-07,Germany.149Train
1m-overlapping patch files contain128releasedplant_ids, matching official Train
count. Stored`plot` is constant99, therefore is not a useful varying identity
disambiguator in this release. Use plant_ids as released identities and at most
one point cloud per ID. Initial cross-patch merging produced6m-spanning objects
because overlapping/local fragments were not universally safe to fuse; before
formal locking it was replaced with each ID's largest single-patch XYZ fragment.
Resulting radii0.181—0.444m. These are possibly partial plants, not guaranteed
whole-plant reconstructions, and no patch fragments are counted independently.

## Annotation-assisted preprocessing

Bonn single-plant extraction is explicitly **annotation-assisted preprocessing**:
plant_ids provide the crop; point count only picks the largestTrain fragment.
No leaf_ids,organ labels,keypoints,phenotypic_traits or future labels enter JEPA
loss, inputs or patch sampling. The data preparation is **not fully unsupervised**.
Crops3D single-plant crop/denoise/annotations were produced by authors; M5 uses
the released authors' single-plant objects without further label-based cropping,
still does not characterize the entire upstream pipeline as label-free.

## Overlap and preprocessing

Within/admitted exact geometry duplicate rows:0.
Cross-Sugar4D exact normalized geometry matches:0.
Hashes sort uniqueXYZ rounded at1e-5after translation/radius normalization;
this catches exact/quantized identical geometry and label-order variants, not
all similar or resampled objects. Sugar4D2021GermanLiDAR and Bonn2022UAV field
trial, Crops3D China's documented2020tomato/2022cabbage differ in acquisition;
no official known overlap is reported. Shared Bonn authors are not proof of
shared plants. Absence of exact hashes does not prove no near duplicates; we
disclose unknownphysicalIDs and never calculate fictitious independent totals.
SugarVal/TestXYZ is read only for this audit, not training or selection.

XYZ-only2048uniform samples after current-cloud mean/max-radius normalization;
deterministic64FPS/32KNN. Cache has points/indices/native transform only, no
semantic/instance labels. Sugar4D and Bonn coordinates are metres; Crops3D
explicit native units are unresolved. Recordnative radius/extent without
inventing conversion; normalizedinput is dimensionless and comparable, but
no cross-sourcephysical-scale claim is made. Sugar4D downstream lawful current
metric descriptors and train-only standardization are unchanged.

Source weightsSugar0.5,Crops0.25,Bonn0.25, external species uniform (Crops each
0.03125). Actual exposure CSV/JSON is recordedperrun; largeCrops species cannot
overwhelm small species. Known cabbageIDsbalanced; unknown otherphysicalIDs
are not masqueraded as resolved. Sources have no external label-based tuning.

## Retrieval audit and reproducibility

Bocha searchHTTP403quota exhausted. Official URL pages/DOI APIs were directly
retrieved, preserved in sources/; no successful search is claimed. Context7
officialPyTorch2.8documentation saved. Network requests include no credentials
in recorded queries/URLs. Crops Range attempt was rejected as200rather than206;
subsequent ordinary complete download and CRC validated the final bytes.
Bonn exactRange downloads were206and officialMD5 passed. Acquisition logs are
not scientific measurements. Unresolved license of GitHub utility code (no
LICENSE) means code is audited/read, not copied into the M5 implementation.

Preparation reproduction:
`.venv-m4/bin/python scripts/prepare_m5_data.py --include-beet`;
already prepared caches retain hashes in external_data_manifest.csv and the
formal protocol will reject changes. Raw data/checkpoints stay local and are
not git source artifacts. External manifests/source snapshots are retained.
