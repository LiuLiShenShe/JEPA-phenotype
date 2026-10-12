# M4 figure QA

- Python source preflight is saved as `figure_source_qa.json`. It audits syntax and
  exports, not scientific validity. No FAIL remains after PNG was raised to 300 dpi.
- Remaining submission-format warnings are accepted for this **engineering report**:
  report-scale widths are larger than a journal column; PNG is a 300-dpi preview;
  editable SVG/PDF are the authoritative resizing assets; no journal TIFF was requested.
- Log-loss scale explicitly checks that recorded losses are positive and finite.
- All figures export source CSV. Three seed error bars are SD, not plant-level CI.
- Label efficiency folds overlap; descriptive aggregate bars are never independent
  biological replicates. Paired inference stays at official validation plant level.
- Stage ranges were locked before train; DAS93 is included as a separately named extra
  stage instead of silently dropping observations or amending the primary protocol.
- GT/predicted segmentation plots use the exact saved sample and physically recoverable
  current coordinates. Typical and worst selection is display-only. No organ instance
  performance is inferred from semantic colours.
- Visual inspection must use the user-requested Agnes vision service, not guessed
  image descriptions. `scripts/qa_m4_figures.py` reads credentials only from environment;
  its results/failures are saved in `figure_visual_qa.json` without credentials.
- Initial Agnes pass identified a count legend overlapping bars and crowded titles;
  final Python export uses explicit top margins, outside legend, and median/worst
  panel names. Initial incomplete-fold figures were not final scientific outputs.
- Near-identical pooled embedding cosines all land in the final broad histogram bin;
  show the actual saved 0/5/50/95/100% quantiles on a disclosed zoomed similarity axis
  instead of hiding fine differences in coarse bins. Full 40-bin distributions remain
  available in diagnostic JSON. Numerical values come from JSON, not vision estimates.
- Agnes descriptions are untrusted visual aids, not metric/source truth. Its reported
  `Efective rank` typo is not present in the actual plotting source (`Effective rank`).
  Its claim of point geometry disappearing/changing between GT/prediction is also
  not accepted: both panels use the identical points and camera, only labels change.
  Occlusion and dominant leaf colours can visually hide rare classes; use saved
  point predictions/confusions for interpretation. Separate plants use per-scan
  physical cm axes, not a shared cross-plant screen scale; within-pair axes match.
