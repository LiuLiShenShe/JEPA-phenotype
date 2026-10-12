# M3 Report — Sugar4D 3D plant growth prediction baseline

> **Engineering decision: PASS. Research decision: REVIEW. M4 is not started.**
> All scores below are from the downloaded Sugar4D archive and recorded model outputs; no synthetic data or synthetic results are used.

## A. Data source, version, licence and integrity
- Official source: Sugar4D project page, Dataverse DOI `10.60507/FK2/IS8YBZ`, Nature Scientific Data descriptor published September 18, 2026.
- Download fixed to Dataverse released V4 (release date August 27, 2026): `sugar4d.zip`, 6,270,780,517 bytes, MD5 `27e44210d95a85cb04b93513f0ef2beb`; ZIP CRC check passed with 916 members.
- Licence: CC BY 4.0. The source archive and readme are retained locally and excluded from Git; raw point clouds and generated caches are ignored.
- Extracted audit: 48 plants, 768 observations, 891 extracted raw files hashed; all 768 PLY files decoded as binary little-endian and valid.
- Coordinates remain in the released metre-scale registered Cartesian frame. The model cache subtracts one fixed x/y anchor from each plant defined at 37 DAS and leaves z unchanged; it never unit-spheres or independently rescales dates.

## B. Cohort, timepoints, morphology and label quality
- 48 plants × 16 observations = 768 scans; DAS = `[37, 40, 44, 47, 51, 54, 58, 61, 65, 72, 75, 79, 82, 86, 89, 93]`. The intervals are real 3/4-day gaps except 65→72 DAS = 7 days.
- PLY fields: `x,y,z,nx,ny,nz,label_semantic,label_instance`; semantic values 0=taproot, 1=core, 2=leaf; instance values -2/-1 for non-leaf classes and 0–19 for leaf labels. All scans contain both label fields and no unexpected label values.
- 6,778 PLY-derived leaf observations correspond exactly to 6,778 `leaf_related.csv` rows and 675 unique plant/leaf-instance identities. No reappearance candidate was found under the explicit missing-scan criterion.
- Actual CSV schemas: `plant_related.csv` is 768 × 61 (three identifiers plus 58 parameter fields); `leaf_related.csv` is 6,778 × 8 (three identifiers plus five morphology fields); the manual leaf file is 450 × 7. Full field names/units/null/nonnumeric checks are saved in `actual_field_schema.json`.
- Plant Height is the released `Height Maximum [m]`: the official extraction implementation uses the 99th percentile of z minus a manually annotated fixed growth-point z after label-wise 2 mm downsampling and removal of taproot. It is not raw maximum-minus-minimum. Plant Width is `Width Major Axis [m]`, the largest pairwise horizontal projected-hull distance. Targets are merely converted m→cm; originals are untouched.
- Visible Annotated Leaf Count is the number of unique nonnegative instance IDs on semantic class leaf, independently checked equal to `Leaf Quantity` in all 768 scans. It is a visible annotation count, not total emerged botanical leaves; the core class can hide unresolvable young leaves. These targets are automatic/annotation-derived, not manually measured real-plant height/width.
- Official reference checks: 15 printed-plant scans give height MAE 0.566 cm and width MAE 0.523 cm against the digital reference model. At 93 DAS, 450 leaf records join manual destructive references; automatic-vs-manual MAE is 1.918 cm total length, 1.868 cm blade length, 2.383 cm petiole length and 1.086 cm blade width. Leaf area has no manual validation in the release.
- One nonnumeric source field is recorded as an audit anomaly: `Normal Angle Mean` embeds `[°]`; the original CSV is untouched and it is not a target.
- Actual ranges: height 7.111–30.752 cm, width 10.334–51.789 cm, visible count 2–17. PLY leaf IDs actually span 0–18 (official format permits 0–19). There are 305,838,181 raw plant points in total; median scan has 363,595.5 points.
- Of 675 leaf identities, 198 are already observed at 37 DAS and 477 first appear later. 127 cease to be observed before 93 DAS; none disappear and then reappear. 6,103 successive-instance centroid moves have median 1.235 cm, 95th percentile 4.579 cm; 19 moves exceed 10 cm and are retained as QC flags, not automatically deleted or called annotation errors.
- Across the 720 adjacent plant observations, released height decreases in 355 and width in 245. These labels include leaf attitude/turgor and extraction variability, so neither is a monotone biomass-growth proxy. The task predicts future measured morphology, not necessarily a scalar biological growth rate.
- `plant_020/plant_021` genotype readings are both provisionally G2 in the source layout image, although this conflicts with the paper’s fourfold-genotype design. A separate old-release trajectory-match diagnostic suggests plant_020=G3, but is not a verified identity and is not used to rewrite metadata. Genotype is never a model input and stratified results are explicitly provisional.

## C. Splits and prediction tasks
- The official plant-level partition is retained: 24 train, 12 validation and 12 test plants; no plant crosses sets. Four additional plant-level cyclic folds (12/12/24) are used as out-of-fold robustness checks; each fold is balanced by plant count, not assumed unseen-genotype generalization.
- Every window uses three consecutive real historical observations and predicts the next actual scan (horizon 1) or the second next actual scan (horizon 2). Each record stores history DAS, actual gaps, target DAS, target labels and plant UID.
- Primary task locked before training: horizon 2 Plant Height. Continue requires at least 10% improvement over both B3-2 and B3-4, improvement for all three seeds, paired plant-bootstrap upper 95% difference < 0 for both controls, and the same ≥10%/≥3-of-4 direction in folds. Otherwise REVIEW after engineering PASS.
- B3-0 and B3-1 are explicitly `Oracle-phenotype`: they read historical released target values. Neural models read only XYZ and legal DAS/time-gap conditions; future point clouds, future labels and future organ IDs are never inputs.

## D. Baselines and training configuration
- B3-0 Persistence: latest historical target repeated.
- B3-1 Growth Trend: least-squares line through the three historical targets against actual DAS, extrapolated to target DAS.
- B3-2 Single-Frame PointNet: latest XYZ cloud, Conv1d/BatchNorm/ReLU pointwise encoder, max+mean pooling, MLP and seven legal time conditions; 61,699 parameters.
- B3-3 PointNet+GRU: shared encoder on three chronological XYZ clouds, one-layer batch-first GRU, MLP head; 160,771 parameters.
- B3-4 Repeated-Latest: identical B3-3 capacity and training budget, but the latest encoded cloud is repeated three times inside the model; independently initialized and trained.
- Neural settings: 2048 sampled XYZ points, fixed cache pool 8192, no geometry augmentation, AdamW, learning rate 0.001, weight decay 0.0001, batch 64, 50 epochs, gradient clip 5.0, seeds `[17, 42, 93]`, train-standardized MSE, validation-only selection.
- Deterministic PyTorch settings and checkpoint fingerprints are recorded. No test score selected model architecture or epoch.

## E. Actual performance and statistics
- Primary and supplementary scores use plant-level macro MAE/RMSE: windows are first averaged within plant, then plants are equally weighted. Bootstrap resampling unit is plant UID, never point cloud or overlapping window.

### Official test plant-macro MAE (cm/count units)
| Model | h=1 height | h=2 height | h=1 width | h=2 width | h=1 leaves | h=2 leaves |
|---|---:|---:|---:|---:|---:|---:|
| B3-0 | 0.855 | 1.312 | 1.915 | 3.017 | 0.686 | 1.056 |
| B3-1 | 1.028 | 1.792 | 2.097 | 3.962 | 0.812 | 1.345 |
| B3-2 | 1.853 | 2.145 | 2.349 | 2.665 | 1.760 | 1.817 |
| B3-3 | 1.889 | 2.164 | 2.375 | 2.654 | 1.798 | 1.875 |
| B3-4 | 1.752 | 2.019 | 2.332 | 2.678 | 1.819 | 1.889 |

### Official test plant-macro RMSE
| Model | h=1 height | h=2 height | h=1 width | h=2 width |
|---|---:|---:|---:|---:|
| B3-0 | 1.207 | 1.763 | 2.645 | 4.170 |
| B3-1 | 1.395 | 2.306 | 2.878 | 5.360 |
| B3-2 | 2.120 | 2.398 | 2.796 | 3.101 |
| B3-3 | 2.171 | 2.438 | 2.801 | 3.066 |
| B3-4 | 2.019 | 2.299 | 2.828 | 3.100 |

- RMSE is the mean of within-plant root mean-square errors, not the square root of pooled MSE. Detailed per-seed values and 95% plant-bootstrap intervals are in `baseline_comparison.csv`, `per_seed_metrics.csv`, `plant_level_errors.csv` and `paired_comparisons.csv`.
- The paired primary comparisons below use B3-3 minus control: negative is favorable.

| Control | h=2 height difference | 95% CI | fold direction |
|---|---:|---|---:|
| B3-2 | 0.019 (0.9%) | [-0.163, 0.192] | 1/4 |
| B3-4 | 0.145 (7.2%) | [-0.005, 0.305] | 2/4 |
- Exploratory paired CSVs additionally include Bonferroni-family intervals across all 48 displayed model/target/horizon contrasts; the locked primary rule is unchanged. No test-derived claim is promoted by selectively ignoring other comparisons.

### Four-fold out-of-fold primary task (48 independent held-out plants)
| Model | h=2 height MAE | seed SD | 95% plant CI |
|---|---:|---:|---|
| B3-0 | 1.220 | 0.000 | [1.088, 1.362] |
| B3-1 | 1.663 | 0.000 | [1.507, 1.832] |
| B3-2 | 1.795 | 0.070 | [1.583, 2.021] |
| B3-3 | 1.900 | 0.067 | [1.670, 2.143] |
| B3-4 | 1.836 | 0.045 | [1.597, 2.087] |
- OOF uncertainty is descriptive: held-out plant predictions are distinct, but fitted fold models share training plants. It is not a fully independent external replication.

### Supplementary analyses (exploratory)
- B3-3 h=2 width MAE is 2.654 cm versus Persistence 3.017 cm (−12.0%, paired 95% CI [−0.852, 0.159] cm) and Growth Trend 3.962 cm (−33.0%, CI [−1.678, −0.936] cm). This demonstrates usable width prediction but does not show extra temporal value: single-frame is 2.665 and repeated-latest is 2.678 cm; those pairwise intervals include zero.
- Stage errors: B3-3 h=2 height MAE is 2.328 cm at target DAS≤58, 1.989 cm at DAS61–75, and 2.205 cm at DAS≥79. Corresponding leaf-count MAE rises from 1.259 to 1.678 to 2.402; label-free random samples struggle most with late visible count, not necessarily chronological biological growth.
- Actual-gap errors are reported separately at h=1 (3, 4 or 7 days) and h=2 (7, 10 or 11 days). Seven-day acquisition interruption is preserved and never interpreted as a fixed scan-day schedule.
- Organ-first-observation relation: B3-3 h=2 width MAE is 2.855 cm in windows with ≥1 newly first-observed leaf versus 2.376 without, but cohorts differ (12 vs 10 plants). This is a descriptive association confounded by stage/growth, not evidence of biological birth causing prediction errors.
- Genotype-stratified CSVs cover all available groups and report independent plant counts, but official test has only one plant per source-layout genotype and two training identities are provisional; no confirmatory genotype claim is made.

| Model | 2048-point h=2 height MAE | 4096-point h=2 height MAE |
|---|---:|---:|
| B3-2 | 2.145 | 2.179 |
| B3-3 | 2.164 | 2.199 |
| B3-4 | 2.019 | 2.126 |
- 4096-point models are independently retrained for the same three seeds and budgets. More samples do not reverse the history-control conclusion; this is a predeclared sensitivity, not a new model-selection opportunity.

## F. Incremental value of 3D history
- Research decision is **REVIEW**, not a declaration that the hypothesis is proven.
- Against B3-2: official primary improvement -0.9%; all seeds improve = `False`; OOF improvement -5.9%; folds improving = 1/4; locked criterion passed = `False`.
- Against B3-4: official primary improvement -7.2%; all seeds improve = `False`; OOF improvement -3.5%; folds improving = 2/4; locked criterion passed = `False`.
- Genotype, DAS-stage, actual-gap, organ-first-observation and point-count analyses are exploratory because 48 plants provide limited independent units and genotype labels have one unresolved source-layout imbalance.

## G. Failure cases and limitations
- Worst official B3-3 h=2 height plant is `plant_026` under plant-level MAE; its forecast trace is Figure 05 and the CSV diagnostics.
- The dataset is a single controlled greenhouse experiment with 48 plants and automatic morphology. Point density ranges from 15,033 to 1,454,217 raw points; fixed subsampling may discard density information, while the point cache is deliberately label-free.
- First observed leaf-instance DAS is an annotation observation, not a biological birth time; the report uses it only as an observation proxy.
- Official split is plant-disjoint but genotypes are represented across sets; provisional layout-image genotype mapping should be confirmed by the dataset authors before genotype claims.
- The delivered task predicts released phenotype measurements rather than complete future point clouds. This is intentionally a baseline and does not justify a JEPA launch by itself.

## H. M4 recommendation
- **Do not automatically start M4.** Engineering is complete, but the locked research rule decides whether to CONTINUE or REVIEW. If REVIEW, first resolve genotype metadata and assess whether any observed 3D-history gain is practically meaningful and robust to plant-level folds; only then consider 3D Temporal-JEPA.

## Reproducibility and files
- Protocol: `configs/m3/protocol.json`; dataset audit: `reports/m3/data_audit.json`, `reports/m3/detailed_data_audit.json`; tests: `reports/m3/tests_initial_real.log` and `reports/m3/tests_initial_real.xml`.
- Code: `src/datasets/sugar4d_dataset.py`, `src/models/pointnet_forecaster.py`, `src/models/pointnet_gru_forecaster.py`, `scripts/prepare_sugar4d.py`, `scripts/train_m3.py`, `scripts/finalize_m3.py`.
- Figures are generated from saved predictions and source metadata under `reports/m3/figures/`; all raw data and large caches remain outside Git.

## Acceptance evidence
- 54 completed neural runs: 9 official 2048-point + 36 four-fold + 9 official 4096-point; all run for 50 epochs/500 steps (27,000 optimizer steps). Deterministic oracle baselines require no stochastic training; duplicating identical oracle runs across seeds would not increase evidence.
- Complete repository tests: 80 passed, 0 failed/0 skipped; logs and JUnit XML are under `reports/m3/tests_all.*`. The M3-only suite includes actual-data decoding/sampling/batching smoke coverage.
- All 18 official checkpoints were reloaded and produced exactly identical repeated inference outputs; saved float32 predictions agree within 1e-5. GRU/repeat initial tensor-state hashes, capacities, optimizers and steps match for each seed/partition.
- Legacy protection checked 3647 previous files; raw protection checked 891 extracted data files; zero changes. Raw archive MD5/CRC and SHA256 and readme hashes are recorded in `data_audit.json`.
- Eight figures are exported as PNG/PDF/SVG/TIFF; plot-source QA passes hard checks. Figure titles, n, error-bar meanings and sampling/crop operations are documented in `FIGURE_CONTRACT.md` and figure QA files.
- No future cloud, semantic/instance labels, oracle phenotypes or genotype are passed to neural forward functions. GT is used only for target construction, audit/visualization and the explicitly Oracle baselines.

## Sources and attribution
- Sugar4D project: https://www.ipb.uni-bonn.de/data/sugar4d/index.html
- Data: https://doi.org/10.60507/FK2/IS8YBZ ; descriptor: https://www.nature.com/articles/s41597-026-08326-5
- Licence: https://creativecommons.org/licenses/by/4.0/legalcode.en ; detailed extraction-code/doc provenance and CC BY author attribution: `reports/m3/SOURCES.md`.
- Requested Bocha search returned quota/funds error; official pages/API/readme/actual files were directly fetched. 图片内容由 Agnes API 识别; outputs and unresolved layout evidence are recorded in `sources/`.
- M3 remains a bounded baseline milestone; M0/M1/M1.5 are unchanged and M4 has not started.
