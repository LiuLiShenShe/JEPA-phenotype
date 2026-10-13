# Figure contract

Backend: Python/matplotlib, matching the supplied repository workflow and saved preference.

- Core claim: identify where segmentation, identity and measurement fail; no fabricated success or biological death.
- Independent unit: plant. Test plots use n=12 plants; repeated scans are not biological replicates.
- Main quantitative figure: plant-level PQ, observation IDF1 and new-first-observation F1, fixed gates and plant-bootstrap 95% intervals.
- Image plate: actual input point samples, GT shown evaluation-only, two actual consecutive scans around an observed new-organ event. Labels refer to output identity, not generated future clouds.
- Temporal heatmaps: full actual DAS grid; blank means unavailable matching, not zero morphology. GT identity axis is evaluation-only.
- Error decomposition: all eligible reference transitions, shared clipped loss and explicit missing penalty; additive differences are not orthogonal variance components.
- Confidence: scores are heuristic rankings, not calibrated probabilities; retain missed-output coverage caveat.
- Cases: worst and best B1 test IDF1; selection rule disclosed, no success-only manual selection.
- Export: white background, PNG300dpi for local inspection, editable SVG and vector-text PDF; no private raw points uploaded to external APIs.
- Preflight: plotting source checks, source-data hash manifest, direct image viewing and annotation audit before delivery.
