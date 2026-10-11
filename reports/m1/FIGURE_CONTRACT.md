# M1 quantitative figure contract

- Conclusion: measure whether three observed RGBs deliver robust plant-level
  benefit over the latest RGB under identical frozen features and time queries.
- Archetype: quantitative model comparison with temporal robustness diagnostics.
- Primary evidence: horizon-specific seed-averaged **plant-macro MAE** and paired
  plant-cluster intervals. Supporting panels show genotype heterogeneity, curves,
  validation optimization and the actual elapsed-time relationship.
- Independent biological unit: 46 held-out plants; windows overlap. Neural runs
  have three seeds, not three additional biological replicates. Each CI resamples
  plants, after averaging errors across seeds, with 2,000 bootstrap replicates.
- Oracle B0/B1 are visually and textually identified; B1b is time-only. No plot
  implies that two-dimensional occupancy is physical area, LAI or sowing age.
- Curve selection: six plants selected by SHA-256(seed, plant_uid), independent
  of errors; failure curves additionally show the worst plants under an explicit
  fixed +8 GRU normalized-error ranking. Neither selection changes evaluation.
- Source data: comparison CSV, all-plant error CSV, per-run predictions and losses.
  Displayed curve predictions average seeds; summary metrics average seed errors,
  which is deliberately different from evaluating an ensemble.
- Plotting exclusively uses Python/matplotlib. White backgrounds, non-rainbow
  palette, minimum 7 pt text, editable SVG and PDF, and 300 dpi PNG previews.
- These are experimental report figures, not a journal submission claim. No
  images of raw plants or masks are included or transmitted to external APIs.
- Exploratory intervals are not multiplicity-corrected; no significance claims
  are based solely on an uncorrected interval for one target/horizon.
