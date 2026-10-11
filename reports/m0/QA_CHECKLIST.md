# M0 QA Checklist

Overall status: **PASS**

| Requirement | Result | Evidence |
|---|---|---|
| Nonempty metadata | PASS | validation_results.json |
| Plant UID and file paths | PASS | validation_results.json |
| RGB/mask one-to-one row uniqueness | PASS | validation_results.json |
| Plant inventory matches frame inventory | PASS | validation_results.json |
| Strict verified within-plant time order | PASS | validation_results.json |
| Projected area ratios in [0,1] | PASS | validation_results.json |
| Illegal colors recorded and quarantined | PASS | validation_results.json |
| Phenotypes and canonical frames agree | PASS | validation_results.json |
| Nonempty plant-level split with seed 42 | PASS | validation_results.json |
| Plant split IDs disjoint and assignments consistent | PASS | validation_results.json |
| All usable images in exactly one plant split | PASS | validation_results.json |
| Seed 42 split reproducibility | PASS | validation_results.json |
| Quarantined images excluded from split index | PASS | validation_results.json |
| Nonempty unique windows for horizons 2/4/8 | PASS | validation_results.json |
| All windows: same plant, future GT, original offsets | PASS | validation_results.json |
| All eligible windows generated exactly once | PASS | validation_results.json |
| Full-file independent audit | PASS | validation_results.json |
| Audit evidence matches current metadata | PASS | validation_results.json |
| Every frame and mask audited | PASS | validation_results.json |
| Byte-identical full pipeline rerun | PASS | validation_results.json |
| Full archive identity, extraction CRCs and provenance | PASS | validation_results.json |
| Official repository commit pinned | PASS | validation_results.json |
| Automated pytest suite | PASS | validation_results.json |
| Local review figures and selection coverage | PASS | validation_results.json |
| Acquisition identity independently confirms directory plant UID | PASS | validation_results.json |
| Real Dataset samples and DataLoader on CPU | PASS | validation_results.json |
