# Figure QA — actual local inspection

Backend: Python/matplotlib. The current model directly viewed the rendered PNGs with the local image tool; no external recognition or image-generation API was used. This is an engineering inspection, not human scientific sign-off.

## Data provenance and integrity
- Quantitative panels come from `per_plant.csv`, `error_decomposition.csv`, `confidence_selective_risk.csv` and event tables, not simulated observations.
- Source cases are the worst/best B1 test IDF1 plants (plant041/plant046), explicitly named in titles; not selected for successful appearance.
- Point-cloud panels show the same geometry-only 4096 sampled XYZ points at DAS37/40 of plant041. Only label coloring and GT first-observation rings are overlaid; geometry is not fabricated, inpainted or sent to an external API.
- GT identities appear only in evaluation axes/annotation. Model/tracker inputs remain distinct and label leakage is tested separately.
- Biological repeat definition is n=12 plants; 16 repeated scans per plant are not used as independent replicates.

## Directly checked outputs
- `baseline_metrics.png`: all four conditions, twelve plant dots each, fixed gates, plant-bootstrap intervals and actual failed B1 event scores are readable.
- `plant_041_identity_timeline.png` and `plant_046_identity_timeline.png`: numeric output IDs are drawn on their observed cells; blank=GT not observed, red cross=observed GT but instance/identity lost, black star=GT first observation. Repeated categorical colors cannot silently hide ID changes because numbers remain visible.
- `tracking_new_leaf_pointcloud.png`: early new leaves 4/5 at DAS40 are ringed in all four rows; coordinates are cm from the real globally positioned PLY. GT panel and output tracker panel colors are not presented as a common identity index.
- `segmentation_tracking_measurement_decomposition.png`: positive increments stack above zero, negative increments below zero; black diamonds show actual total. Negative B1/B2 increments are explicitly labeled possible wrong-link substitution artifacts, not improvement. The accompanying common-support table preserves this limitation.
- `new_leaf_events_and_confidence.png`: false rebirth and spurious birth are shown alongside true-new missed instances; confidence curves are marked rankings rather than calibrated probabilities. Errors are not hidden by an arbitrary retained-coverage choice.

## Export and automated preflight
- All six figures have PNG300dpi, editable SVG and PDF with TrueType text; paths and SHA256 hashes are in `figure_manifest.json`.
- Static source preflight: 11 PASS, 0 FAIL, 3 WARN. Warnings are accepted for this engineering report: PNG rather than TIFF, 300 rather than600dpi, and wide report layouts rather than fixed89/183mm manuscript columns. No Nature submission-ready claim is made.
- No raw images were uploaded; no generative content substitutes for experimental point data. Visual checks do not validate automatic phenotype reference truth.
