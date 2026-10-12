# M3.5 — Organ prediction and current 3D representation diagnosis

Bounded experiment on **existing Sugar4D V4**, not a JEPA/diffusion launch.
All M0–M3 inputs, code and results remain immutable. New code and artifacts use
M3.5 paths. The locked protocol specifies the h2 leaf-total-length primary task,
three seeds, official and four plant-level partitions, feasibility gates,
geometry/current diagnosis and an explicit research branch rule.

## Reproduce

Use the existing `.venv-m1` environment; exact dependencies are in
`requirements-m3_5.txt` and actual runtime metadata in
`reports/m3_5/environment.json`. Raw V4 data and M3 point pools must already exist.
Do not download again or overwrite M3 outputs.

```bash
.venv-m1/bin/python scripts/analyze_m3_5_feasibility.py
.venv-m1/bin/python scripts/train_m3_5.py --stage current --device cuda:0
.venv-m1/bin/python scripts/evaluate_m3_5.py --stage current
.venv-m1/bin/python scripts/train_m3_5.py --stage future --device cuda:0
.venv-m1/bin/python scripts/evaluate_m3_5.py --stage select
.venv-m1/bin/python scripts/evaluate_m3_5.py --stage final --device cuda:0
.venv-m1/bin/python -m pytest -q --junitxml=reports/m3_5/tests_all.xml
.venv-m1/bin/python scripts/verify_m3_5.py
```

Completed checkpoints are reused. To retrain from scratch use a separate
experiment copy with the same local-data layout and no M3.5 checkpoints; never
delete or modify previous milestones. Raw point/organ files and checkpoints
are local and uncommitted; small report tables remain available for auditing.

## Outputs and interpretation

- `reports/m3_5/FEASIBILITY.md`, `feasibility.json`: numerical label-task
  feasibility, independent plant/class coverage and morphology-quality limits.
- `reports/m3_5/M3_5_REPORT.md`: current-state errors, organ Oracle baselines,
  label-free whole-plant geometry absolute/residual forecasts, events,
  visibility, plant-level uncertainty and fixed branch recommendation.
- `reports/m3_5/selection_lock.json`: all selected checkpoints and validation
  thresholds locked before final exploratory Test inference.
- `outputs/m3_5/data/organ_trajectories.csv`: original units, missing morphology
  as NaN, current-instance centroids; future IDs/observability only form labels.
- `reports/m3_5/figures/`: six real-data quantitative PNG/PDF/SVG figures,
  source CSVs, captions and reproducible Python plotting code.
- `reports/m3_5/acceptance.json`: tests, source/input protection, equal-capacity
  control training, exact checkpoint replay and completed-run counts.

Biological independent unit is **plant**, not leaf/window. First observed is
not precise botanical birth. C0–C4 use historical GT organs (Oracle); C5 uses
only current/history XYZ whole-plant descriptors (non-Oracle). Official height
requires a manual reference and semantic preprocessing; our XYZ proxies are
not that official measurement. Official Test was used in M3: M3.5 results are
exploratory, not independent confirmation. No automatic M4, merge or push.

Source: local fixed V4 audit and actual saved experiment outputs; official
Sugar4D page https://www.ipb.uni-bonn.de/data/sugar4d/index.html . Bocha search
returned HTTP403 quota; documentation was fetched from primary sources via
Context7 and saved under `reports/m3_5/sources/`.
