# M5 quantitative figure contract

Backend: Python/matplotlib, existing saved preference. Figures are deterministic
plots of real source CSVs, not generative imagery.

1. Budget ladder: SSL loss is not success; effective-rank concentration can occur
   while variance grows. One seed on official training plants.
2. Locked 50% primary: all conditions, all seeds, all splits, not cherry-picked.
3. Label efficiency: nested 6/12/24 plant budgets; 50% remains the primary.
4. Semantic transfer: sampled-point three-class IoU, not leaf instance PQ/SBD.
5. Similarity controls: centred/uncentred, identity and local-patch contrasts.
6. Leaf-token probe: train-plant evaluation-only labels, not unseen instance
   segmentation; geometry-distance matching exposes proximity confounding.
7. Official primary paired intervals: all fixed P3/P4 control comparisons,
   plant-level 95% CI and predeclared 0.2-leaf absolute improvement threshold.
   CI alone does not waive the relative/seed/fold/semantic/representation gates.

Biological replicate = plant. Repeated scans and training seeds are not new
biological replicates. SD whiskers are descriptive; inferential intervals are
paired plant bootstrap after averaging seeds. Official and cross-fold validation
plants overlap and are never pooled as 60 unique biological samples.

Exports: editable SVG/PDF and 300 dpi PNG, source CSV, per-figure JSON contract.
Preflight: plotting syntax and export checks, canvas text bounds, quantitative
source coverage and direct local image inspection. Model visual QA is not human
review/signoff and cannot replace numerical/source checks.
