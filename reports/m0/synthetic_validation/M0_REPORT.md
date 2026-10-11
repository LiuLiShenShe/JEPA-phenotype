# M0 Report — Plant Temporal-JEPA

**M0 status: PARTIAL — synthetic code-path demonstration only**

This report is generated from local execution. No real-data statistic is inferred when the official archive is unavailable.

## Execution
- Real dataset available in this report: `False` (all 168 observations below are synthetic)
- Official download status: `not_run`
- Official repository commit: `0d9a24ff146d12cc8fca6bdfa2538d196754f456`
- Frame rows scanned: `168`
- Data issues recorded: `0`

## QA checklist
| Check | Status | Details |
|---|---|---|
| RGB/mask row uniqueness | PASS | duplicate rows=0 |
| RGB path uniqueness | PASS | duplicate paths=0 |
| RGB/mask files exist | PASS | missing_or_corrupt=0 |
| RGB/mask dimensions | PASS | mismatches=0 |
| Projected area ratio in [0,1] | PASS | invalid_rows=0 |
| Palette validation | PASS | frames_with_invalid_colors=0 |
| Within-plant time order | PASS | problem_plants=0 |
| Duplicate timestamps | PASS | duplicates=0 |
| Plant IDs disjoint | PASS | intersections=0 |
| Each image belongs to one split | PASS | multi-split-images=0 |
| Fixed-seed split reproducibility | PASS | same_assignments=True |
| All usable frames assigned | PASS | unassigned=0 |
| Temporal window IDs unique | PASS | duplicates=0 |
| Window history/target same plant | PASS | invalid_windows=0 |
| Targets strictly future | PASS | non_future=0 |
| Window split matches plant split | PASS | mismatches=0 |

## Data-use constraints
- Dataset source: `https://huggingface.co/datasets/rick77a/PhenoLeaf_TS`.
- License: La Trobe University Use Terms; non-commercial academic research/teaching only; do not redistribute raw images or masks.
- The official image-level split is intentionally not used; this project creates a plant-level split using `plant_uid = genotype/replicate`.

## Limitations
- A real-data M0 PASS requires a successful official archive download and a non-empty full scan.
- Synthetic fixture tests validate the code path only and are not evidence about the real dataset's counts or annotation quality.
