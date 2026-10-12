# Sugar4D M3 baseline reproduction

This milestone is isolated from M0/M1/M1.5. Do not overwrite their data, outputs,
splits or reports. The environment used is `.venv-m1`; versions are recorded in
`reports/m3/environment.json` and `requirements-m3.txt`.

## Data

Sugar4D V4 archive file 54550 is available from Bonndata. The archive must match
MD5 `27e44210d95a85cb04b93513f0ef2beb`, size 6,270,780,517 bytes. The readme is
file 58163, MD5 `3959c3c687cf10f5b51b41bb03f95bd3`. Both belong to DOI
`10.60507/FK2/IS8YBZ` and are CC BY 4.0.

```bash
mkdir -p data/raw/sugar4d
curl -L --fail --retry 3 -C - https://bonndata.uni-bonn.de/api/access/datafile/54550 -o data/raw/sugar4d/sugar4d.zip
curl -L --fail https://bonndata.uni-bonn.de/api/access/datafile/58163 -o data/raw/sugar4d/sugar4d_readme.txt
.venv-m1/bin/python scripts/prepare_sugar4d.py --detailed
.venv-m1/bin/python -m pytest -q tests/test_m3_dataset.py tests/test_m3_models.py tests/test_m3_metrics.py
```

The released genotype layout was machine-read using the specified vision service.
Plant 021 was read as G2, inconsistent with the stated fourfold-genotype cohort.
The mapping and conflicting evidence are recorded under `reports/m3/sources/`.
Do not silently change it or claim confirmed balanced-genotype performance;
genotype is not a neural model input.

## Training and evaluation

The task and hypothesis rule are fixed in `configs/m3/protocol.json`.

```bash
.venv-m1/bin/python scripts/train_m3.py --partition official --device cuda:0
for fold in fold0 fold1 fold2 fold3; do
  .venv-m1/bin/python scripts/train_m3.py --partition "$fold" --device cuda:1
done
.venv-m1/bin/python scripts/train_m3.py --partition official --points 4096 --device cuda:0
.venv-m1/bin/python scripts/replay_m3.py --device cuda:0
.venv-m1/bin/python -m pytest -q --junitxml=reports/m3/tests_all.xml > reports/m3/tests_all.log 2>&1
.venv-m1/bin/python scripts/verify_m3.py --archive
.venv-m1/bin/python scripts/evaluate_m3.py --sensitivity
```

Each run lasts 50 epochs with validation-only checkpoint selection. Do not alter
architecture or hyperparameters based on test results. This workflow stops before
M4; use `reports/m3/M3_REPORT.md` for the final decision.
