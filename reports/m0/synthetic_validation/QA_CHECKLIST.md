# M0 QA Checklist

Overall M0 status: **PARTIAL** — synthetic checks only; not real-dataset evidence.

| Requirement | Result | Evidence |
|---|---|---|
| RGB/mask row uniqueness | **PASS** | duplicate rows=0 |
| RGB path uniqueness | **PASS** | duplicate paths=0 |
| RGB/mask files exist | **PASS** | missing_or_corrupt=0 |
| RGB/mask dimensions | **PASS** | mismatches=0 |
| Projected area ratio in [0,1] | **PASS** | invalid_rows=0 |
| Palette validation | **PASS** | frames_with_invalid_colors=0 |
| Within-plant time order | **PASS** | problem_plants=0 |
| Duplicate timestamps | **PASS** | duplicates=0 |
| Plant IDs disjoint | **PASS** | intersections=0 |
| Each image belongs to one split | **PASS** | multi-split-images=0 |
| Fixed-seed split reproducibility | **PASS** | same_assignments=True |
| All usable frames assigned | **PASS** | unassigned=0 |
| Temporal window IDs unique | **PASS** | duplicates=0 |
| Window history/target same plant | **PASS** | invalid_windows=0 |
| Targets strictly future | **PASS** | non_future=0 |
| Window split matches plant split | **PASS** | mismatches=0 |
