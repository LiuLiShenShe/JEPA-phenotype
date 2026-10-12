# Output-only implementation correction

Date: 2026-10-12, after all 45 formal SSL runs, 180 new downstream runs and
75 train-only probes completed. No model is retrained or selected again.

The first complete evaluation failed when exporting `decision.json` because
`pandas.Series.all()` returned a NumPy boolean for matrix completeness; the
historical M4 JSON writer deliberately accepts only native JSON types. Metric
CSVs had already been computed correctly. The failure did not change a metric,
comparison, hypothesis, threshold, split or candidate.

`scripts/evaluate_m5.py` now supplies `m5.artifacts.write_json` at the output
boundary. It converts NumPy scalar values to their exact native values (and
arrays to lists), continues to reject NaN, and otherwise uses the same JSON
options. The locked `m5.evaluation` source and all locked inputs are **unchanged**.
Tests cover a true/false round trip and invalid nonfinite values. This is not
a post-result amendment to the statistical rules; only JSON serialization is
corrected. Use `scripts/evaluate_m5.py` as the canonical evaluation entry point.
