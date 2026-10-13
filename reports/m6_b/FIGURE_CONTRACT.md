# M6-B figure contract

Backend: saved Python/matplotlib preference, matching original repository flow.

- Main claim: target-domain supervision may improve instances, but the claim of
  tracking readiness depends on locked PQ/IDF1/event gates, not semantic accuracy.
- Quantitative evidence: plant-level macro Validation PQ/SBD/recall with n12
  independent plants, seed means and10,000-resample plant bootstrap95% intervals.
  S1/S2 have3 seeds; S3 only1; scans and seeds do not inflate biological n.
- Learning curves: six prespecified Validation epochs, exact60-epoch budget;
  semantic-only high accuracy must not substitute for instance PQ.
- Propagation: unchanged tracker/measurer, GT B0 and historical S0 B1 retained;
  B3 fixed seed17 of Validation-selected family, B4 supervised PSegNet seed17.
- Error archetype: paired bar/point summaries and stacked diagnostic categories
  separate instance misses from sampled support/identity failures. Stacked totals
  are GT observation fractions, not scan-level confidence intervals.
- Actual point-cloud plates: whole-plant Validation worst/best B3 PQ selection;
  fixed DAS44/79, real current points, XY projection in cm, no synthetic leaves.
  Predictions are IoU-color-associated solely for visual evaluation; unassociated
  colors do not imply temporal matching. Deterministic18,000-point geometry-only
  visualization subset does not replace full-resolution metric calculations.
- Additional early DAS44 3D views use12,000 real geometry-only points, identical
  camera angles, XYZ axes in cm and actual full-resolution PQ. No depth-generated
  or interpolated geometry; full cloud is retained in numerical evaluation.
- Integrity: no generative image API, external raw-data upload, hidden cropping,
  local brightness adjustments or successful-case cherry-picking.
- Export: white-background PNG300dpi, editable-text SVG/PDF, source CSV hashes,
  source preflight and direct local image viewing before final delivery.
- Scope: engineering research report previews, not declared final Nature plate
  dimensions. Categorical instance palette is not a continuous rainbow metric.
- QA: check labels, CI, biological n, cohort roles, missing outcomes, units,
  clipping/overlap and exact numeric traceability; disclosed proxy proximity
  is not confirmed biological leaf-emergence/death annotation.

Sources: actual V4 PLY and executed CSV/checkpoint artifacts; upstream provenance
in `SOURCE_AUDIT.md`. Quantitative figures are data rendering, not generated images.
