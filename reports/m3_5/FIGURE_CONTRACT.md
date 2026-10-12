# Figure contract

Core claim: use a locked primary organ task to test historical increment while
first diagnosing how reliably XYZ recovers the current published plant state.
Backend: existing Python/Matplotlib workflow. Archetype: quantitative grid.
All panels use actual local V4 data and saved forecasts, never generated values.

1. Real validation leaf trajectory; history-coverage selection, not accuracy.
2. The same leaf, next-two-scan length/area forecasts from all Oracle baselines.
3. Hero panel: locked h2 total-length GRU-control paired plant differences in
   official/four-fold validation, 10,000 plant-bootstrap 95% intervals and the
   prespecified practical -0.5cm threshold. Overlapping fold training acknowledged.
4. First observation counts/probabilities for the same validation plant; no
   botanical birth interpretation and no future identity inputs.
5. Raw XYZ proxies versus official traits and trained current estimates,
   explicitly different height preprocessing/reference-point definitions.
6. Post-hoc highest validation-error plant/leaf diagnosis, not model selection.

Export: PNG at 300dpi plus editable SVG/PDF, white background, restrained
colorblind-safe palette, approximately 183mm width, readable 8pt text. Caption,
source CSV/SHA256, actual sample scope and automated renderer text bounds are
retained in figure_manifest.json. Scientific inference uses mean per-seed
errors, not the seed-mean lines displayed for clarity.
