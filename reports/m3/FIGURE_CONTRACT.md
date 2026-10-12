# M3 figure evidence contract

- Core question: does real three-scan 3D history improve future plant morphology over single-scan and independently retrained repeated-latest controls?
- Backend: Python / Matplotlib only; recorded experimental data, not generated image content.
- Evidence: 01 fixed-physical-scale temporal scans; 02 released morphology curves; 03 model/horizon error; 04 paired retrained control; 05 worst independent plant; 06 official/fold plant partition; 07 actual DAS-gap errors; 08 predeclared 2048/4096 sampling sensitivity.
- Independent unit: plant UID. Main test n=12; four-fold OOF n=48, not 1200 independent windows. Neural mean and seed SD from three seeds; oracle baselines deterministic, one run. Paired intervals bootstrap seed-averaged plant errors, 10,000 resamples.
- Coordinates: PLY metres converted to centimetres for displayed axes only, same x/y translation and same axis limits across dates. Organ-instance pseudo-colors are ground-truth visualization only; never model inputs. Raw files are unchanged.
- Source tables: metadata/window/cache manifests, official/fold prediction CSVs, plant scores, baseline/paired comparisons and diagnostics. Figures include editable SVG, PDF, TIFF and PNG previews.
- Risks to state: source-layout genotype discrepancy for plant_021; small single-environment cohort; automatic target measurement; future requested DAS is a legal schedule condition; leaf first observation is not biological birth.
