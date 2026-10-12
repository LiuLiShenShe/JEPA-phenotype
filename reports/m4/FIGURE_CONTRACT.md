# M4 figure contract

Backend: Python/Matplotlib, as required by the Python experiment workflow. All
figures come from saved real measurements/predictions; no generated pseudo-data.

## Scientific claim and evidence map

Claim to test, not assume: train-only JEPA pretraining improves current unseen-plant
leaf-count MAE and useful local organ representation beyond identical random weights.

1. Training/collapse grid: loss, dimensionwise std, effective rank. A declining loss
   alone cannot establish the claim. Diagnostic embeddings are 96 fixed **training**
   scans; scanning order/plant composition are disclosed, not random independence.
2. Current count grid: same P0/P2 heads and unchanged strong historical PointNet +
   Geometry, official plus four validation folds, three seed SD bars.
3. Plant-label efficiency: 6/12/24 independently labelled plants, frozen linear head
   versus full fine-tuning. Fold/seed aggregate SD is descriptive, not a bootstrap CI.
4. DAS grid: preregistered current stages; the audited data also contain 93 DAS, which
   is retained as an explicitly extra stage rather than discarded or silently changing
   the locked stage boundaries.
5. Segmentation grid: plant-macro mIoU and per-class IoU on all fixed 2048-point samples.
   No instance PQ/SBD claim. Missing classes are undefined, not filled with perfect IoU.
6. Failure grid: worst current-count validation plant/scans, averaging seeds.
7. Local semantic point plate: median and worst P2/seed17 official-validation scans,
   GT versus predicted labels on the same unaltered 2048 XYZ points, physical cm scale.

## Export and integrity

- Archetypes: quantitative grids; one image/point plate with semantic predictions.
- SVG/PDF editable text, PNG previews; all panels also export companion source CSV.
- Source figures follow all recorded partitions/seeds/budgets. Top failures and median/
  worst plate selection are display-only and identified in captions, not training rules.
- Seed SD cannot substitute for plant-level paired inference. Locked primary uses a
  paired bootstrap of independent official-validation plant MAE averaged over seeds.
- Five partitions reuse plants; never count them as 5×48 independent plants.
- Current-scan labels appear only in supervised loss, metrics and GT visualization.
- Plot exclusions: P1 unavailable (explicit table rows), P3 optional not run. No fabricated
  bar or imputed point is plotted for unavailable official weights.
- Report-scale figures, not a claim of finished Nature layout compliance. Vector outputs
  and source data permit later journal resizing without changing the evidence.
