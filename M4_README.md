# M4: train-only Point-JEPA plant representations

This isolated milestone does **not** predict future clouds or start Temporal-JEPA.
Full results and engineering/scientific caveats: `reports/m4/M4_REPORT.md`.

## Conditions

- P0: official-size Point-JEPA backbone, random initialization.
- P1: official ShapeNet weights; network unavailable in the recorded run, not fabricated.
- P2: 20-epoch JEPA SSL on each partition's 24 training plants, from scratch.
- Reuse original M3 official + four plant folds; seeds 17/42/93.
- Current annotated leaf count: 6/12/24 labelled plants, frozen linear probe and fine-tune.
- 2048-point taproot/core/leaf semantic segmentation: frozen head, full fine-tune, PointNet.
- No test prediction or test-based checkpoint/architecture selection.

## Reproduce

```bash
bash scripts/setup_m4_environment.sh
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python -m pytest tests/test_m4.py tests/test_m4_protocol.py -q
.venv-m4/bin/python scripts/prepare_m4.py --stage lock
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/verify_m4.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/prepare_m4.py --stage cache
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/pretrain_m4.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/train_m4_downstream.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/diagnose_m4_representations.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/evaluate_m4.py --verify-checkpoints
.venv-m4/bin/python scripts/prepare_m4.py --stage verify
```

Local audited raw Sugar4D V4/M3 artifacts are required; untracked large artifacts
stay under `outputs/m4/`. New environment does not change M0--M3.5 dependencies.
Completed runs safely skip with provenance checks; SSL resumes exact full state.
Inference reconstruction supports both frozen compact heads and fine-tuned models.
Metric CSVs are validation-only, plant-level. Test remains unavailable to M4 loaders.
25/50/100% budgets reduce independent plants, never random scans within one plant.

The protocol hash is locked before training. P1 unavailability was fixed before
SSL fitting; no insertion of newly acquired official weights into this matrix.
Do not treat architectural compatibility tests as actual official checkpoint coverage.
Do not treat semantic mIoU as leaf-instance PQ/SBD.
