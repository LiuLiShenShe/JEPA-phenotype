# M5 bounded plant Point-JEPA verification

This milestone extends **current**, not future, plant representation experiments.
It does not train Temporal-JEPA, predict future plants, or use Diffusion.

## Artifacts

- `reports/m5/M5_REPORT.md`: complete results, research decision and limitations.
- `reports/m5/DATA_SOURCE_AUDIT.md`: actual source/licence/acquisition audit.
- `reports/m5/protocol_lock.json`: exploratory target, exact input/implementation
  hashes, unchanged nested labelled-plant budgets.
- `reports/m5/BUDGET_SELECTION_NOTE.md`: training-only 2000→500-step amendment;
  not an undeclared downstream tuning sweep.
- `reports/m5/*metrics.csv`, paired statistics, diagnostics and editable figures.
- `outputs/m5/`: local XYZ caches, full checkpoints and predictions; gitignored.

## Locked experiment

M4 architecture is unchanged: 384-dimensional/12-block/6-head encoder,
192-dimensional/6-block/6-head predictor; 2048 points, 64 FPS patches × 32 KNN.
P0/P2 are unmodified archived M4 controls. New S500 = Sugar4D-only 500 steps;
P3 = Sugar4D Train + 823 Crops3D Train scans + 128 Bonn train-ID partial fragments;
P4 applies one preregistered context-token variance/covariance loss, conditional
on observed M5-A degeneration. No future/semantic/leaf labels in SSL target.
Bonn plant-ID cropping is **annotation-assisted preprocessing**.
Some Crops3D physical identities/native units remain unknown; 951 objects are not
951 verified independent plants. Source/species sampling is explicitly balanced.

Primary = **50% labelled plants, current visible leaf-count plant-macro MAE**;
same architecture/subsets/head initialization/seeds/epochs. Three seeds, original
official and four folds; plant-paired bootstrap. No untouched confirmatory test.
Every encoder excludes its own partition's Val/Test; the official encoder uses
only original official Train. Existing folds rotate official roles, so the fold
collection is not a global official holdout. See `split_provenance.json`.
Semantic IoU and train-only read-only leaf-token identity probes are auxiliary.

## Reproduce

Use `.venv-m4` (PyTorch2.8.0+cu128). Preserve all M0–M4 inputs and results.
Prepare external source files/caches with `prepare_m5_data.py`, audit with
`audit_m5_sources.py`; actual final hashes must match the lock. Training/evaluation
refuses mutated inputs. Download metadata and archived source responses in
`reports/m5/sources/`;no credentials are stored in reproduction artifacts.

```bash
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/pretrain_m5.py
.venv-m4/bin/python scripts/run_m5.py --stage lock
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/run_m5.py --stage run --partitions official,fold0,fold1
CUDA_VISIBLE_DEVICES=1 .venv-m4/bin/python scripts/run_m5.py --stage run --partitions fold2,fold3
.venv-m4/bin/python scripts/evaluate_m5.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/diagnose_m5_external.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/restore_m5.py
.venv-m4/bin/python scripts/verify_m5.py
.venv-m1/bin/python -m pytest tests -q
.venv-m4/bin/python scripts/prepare_m5.py --stage verify
.venv-m4/bin/python scripts/report_m5.py
```

The two jobs have disjoint partition write sets, not distributed training. Saved
completed runs are not overwritten. SSL state includes optimizer/scheduler/all RNG/
sampler; downstream selected/final states are independent clones. Checkpoint
prediction restore is verified. If source identity or units remain incomplete,
engineering acceptance remains PARTIAL, even when the executed matrix is complete.

At M5 completion **stop**. No next-stage task, Git merge, commit or push is authorized.
