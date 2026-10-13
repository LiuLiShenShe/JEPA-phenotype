# Figure QA scope and source preflight

Matplotlib source and metric CSVs are auditable. Final rendering/direct image
inspection must occur after actual trained predictions exist, not from mock data.
`figure_source_preflight.json` is a static audit, not numerical validation.

Accepted static warnings:
- PNG300dpi plus editable SVG/PDF is an engineering-report preview bundle, not
  requested journal TIFF submission. No claim of submission-ready final dimensions.
- Wide three-metric panels are report-readable rather than fixed89/183mm plates.
- `np.random.default_rng` only selects deterministic **real point indices** for
  display. No synthetic coordinates, experimental values or future leaves.
- Categorical tab20 colors identify actual instances; matched GT/pred colors are
  evaluation-only visualization, never tracker identity input.

Final checks: real sourceCSV hashes, strict matching, biological n12, seed labels,
fixed Validation/Test roles, physical centimetre axes, readable legends/CI,
all missing errors retained, case selection rule and original scan provenance.
Uses current model's direct local image viewing; no external vision/generation API.

Source: original point files and executed result tables, not web-generated imagery.

## Final direct-image review —2026-10-13

The current model directly opened all8 local PNG outputs (no Agnes vision or
image-generation API). Confirmed the displayed pointclouds are the actual
GT/S0/B3 outputs from deterministic minimum/maximum Validation B3 plants004/019;
the early DAS44 and late DAS79 views are not hand-picked success-only cases.
All quantitative axes,12-plant bootstrap intervals, seed-mean labels and fixed
tracking threshold lines agree with the exported result CSVs. The learning
curves show all7 real runs and the60-epoch budget. Full-resolution scoring is
not replaced by the deterministic displayed18k/12k point subsets.

Two display-only findings were fixed and re-viewed: the observation-error
vertical label was too long and is now shortened with diagnostic context in the
title; early3D tick labels were crowded and now use four-bin locators. Physical
cm axes remain unchanged, and no metric, point geometry or prediction changed.
Remaining3D perspective/occlusion and repeated categorical colors limit visual
interpretation; actual numeric per-leaf errors and full-resolution NPZ remain
authoritative. Plots are engineering-report figures, not certified journal
submission plates or human-approved scientific images.

PNG300dpi plus editable SVG/PDF exports exist. Source preflight has no FAIL;
the documented wide-report/sampling/export warnings are accepted. Quantitative
data and rendered PNG SHA256s are recorded in `figure_manifest.json`. This visual
review does not certify the absolute phenotype measurement system.
