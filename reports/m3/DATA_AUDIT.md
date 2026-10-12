# Sugar4D V4 independent data audit

## Source and immutable inputs

- DOI: `10.60507/FK2/IS8YBZ`; Bonndata release V4, August 27, 2026.
- ZIP bytes: 6,270,780,517; MD5: `27e44210d95a85cb04b93513f0ef2beb`.
- ZIP SHA256: `69e25b87b470173d30a70fbac9d4caf2ae968082cd83378cd87d93601b370908`.
- Readme MD5: `3959c3c687cf10f5b51b41bb03f95bd3`.
- All 916 archive members pass CRC; 891 non-directory extracted files are
  individually SHA256-hashed and final verification detects zero changes.
- CC BY 4.0 attribution/adaptation duties and primary source links are in
  `SOURCES.md`; raw files and large derived caches are excluded from Git.

## Plant, time and PLY audit

- 48 unique global plant IDs, exactly 16 actual scans each, 768 decoded plant PLYs.
- Actual DAS: 37,40,44,47,51,54,58,61,65,72,75,79,82,86,89,93.
- Every scan joins one plant-level CSV row and the filename DAS/ID. Official TXT
  membership is plant-disjoint: 24 train, 12 validation, 12 test plants. The exact
  same legal cohort is retained for all models; no hard test plant is deleted.
- Binary little-endian PLY has eight vertex fields: float32 XYZ and normals;
  int32 semantic/instance labels. All coordinates/normals are finite, all normals
  have length within 0.01 of unity, and every semantic/instance pair is valid.
- Semantic 0/1/2 means taproot/core/leaf; non-leaf instances -2/-1. Released
  leaf IDs are 0–18 in actual files (official schema permits 0–19).
- Raw counts: minimum 15,033; median 363,595.5; maximum 1,454,217; sum 305,838,181.
  `Points Quantity` in the morphology CSV is a **voxel-reduced descriptor**, not
  the raw PLY vertex count; it is therefore not an integrity mismatch.

## Physical coordinates and preprocessing

- Source data are registered to a fixed floor-target Cartesian frame; z is
  vertical positive and coordinates are metres. This is supported by the
  descriptor, reference-growth-point comparisons and the 10m experimental layout.
- The model cache uses one fixed, label-free x/y translation derived from the
  first actual scan at 37 DAS. z remains unchanged; the same rigid translation
  is used for every later observation. It does not contain phenotype/instance GT.
- No date is normalized to a unit sphere; no per-date scaling, rotation, future
  reference point, future organ identity or future phenotype is used as input.
- A deterministic 8192-point pool per frame is stored separately; 2048/4096
  samples preserve metre scale. Epoch-varying training sampling is seed-recorded;
  inference sampling is stable across window orders and models.
- Publisher registration validation is millimetre-scale, but leaf attitude,
  occlusion and scanner density remain limitations. Local QC retains 19 >10cm
  leaf-centroid displacements without assuming that they are mislabeled leaves.

## Plant targets and validation

- `plant_related.csv`: 768 rows × 61 columns, three identifiers plus 58 released
  automatic morphology fields. Target Height Maximum [m] is z percentile99 minus
  fixed manually annotated growth point, after label-wise 2mm downsampling and
  taproot exclusion; Width Major Axis [m] is maximal projected-hull distance.
- Target units are consistently changed m→cm in **derived metadata only**.
  Raw CSVs are never rewritten. Ranges: height 7.111–30.752 cm; width
  10.334–51.789 cm; visible annotated leaves 2–17.
- Visible count is checked independently from unique nonnegative instance IDs
  on semantic leaf points and equals released `Leaf Quantity` for every scan.
- 15 printed-plant scans validate automatic height/width versus the reference
  digital model: MAE 0.566/0.523 cm. This is not manual real-plant ground truth.
- Source anomaly: plant001/37 DAS `Normal Angle Mean` contains a numeric value
  followed by `[°]`. This original string is recorded and is not a target.
- 355/720 height and 245/720 width transitions decrease. Future phenotype is not
  necessarily monotonic growth or biomass; these declines are legal observations.

## Organ association and observation status

- `leaf_related.csv`: 6778 rows × 8 columns, three identity fields plus total
  length, blade length, petiole length, blade width (cm) and leaf area (cm²).
- All 6778 rows join PLY-derived plant/instance/DAS triples one-to-one; zero
  missing, foreign or duplicate associations. There are 675 unique leaf identities.
- Full 675×16 observation grid stores `observed` or `not_observed` (missing is
  not silently converted to zero morphology). 198 identities appear at 37 DAS,
  477 are first observed later, 127 disappear before the last scan, zero
  disappear/reappear sequences under the missing-actual-scan definition.
- 450 manually measured leaves at 93 DAS join automatic measurements. MAE in cm:
  total length 1.918, blade length 1.868, petiole length 2.383, blade width 1.086.
  Leaf area has no released manual validation. Organ prediction is therefore
  deferred from the first baseline output rather than treated as reliable GT.
- First observed DAS is **not** a biological organ birth time; emergence,
  occlusion, density and the unlabeled-resolution core can delay observation.

## Genotype caveat

图片内容由 Agnes API 识别。The official layout is repeatedly read as plant020=G2
and plant021=G2, conflicting with the fourfold-per-genotype design. A secondary
Precision Agriculture supplementary trajectory match suggests plant020=G3,
plant021=G2; this diagnostic is **not** verified identity and is not used to alter
source labels or model inputs. The preserved provisional mapping has G2=5,
G3=3 and other groups=4. Genotype summaries/fold balance are explicitly provisional,
while all independent units remain disjoint plants. Obtain author confirmation
before making genotype-specific conclusions.

## Evidence files

`data_audit.json`, `detailed_data_audit.json`, `raw_file_manifest.json`,
`actual_field_schema.json`, `geometry_audit.csv`, `data_anomalies.csv`,
`organ_csv_mismatches.csv`, `leaf_reappearance.json`, `manual_leaf_validation.csv`,
`reference_plant_validation.csv`, `centroid_qc_flags.csv`, `final_verification.json`.
