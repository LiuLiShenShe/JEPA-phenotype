# M1: 2-D temporal phenotype prediction

M0 files, raw observations, quarantines and the seed-42 plant split are read-only.
This extension does not train JEPA and has no three-dimensional experiment.
See `reports/m1/M1_REPORT.md` for the actual result rather than assuming success.

## Isolated environment

Prerequisites: Linux, NVIDIA driver compatible with CUDA 12.8, `uv`, Python 3.11,
existing audited M0 data and enough disk space for a CUDA environment. M0 `.venv`
remains unchanged. The tested pair is PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128.
Run `bash scripts/setup_m1_environment.sh` only for a fresh M1 environment.

Large artifacts remain local. For fresh clones, exclude `.venv-m1/`,
`third_party/dinov2/` and `outputs/m1/` in `.git/info/exclude`; never upload raw data.
The checked-in output/figure `.gitignore` files prevent generated artifact commits.

## Reproduce

From the repository root, with M0 data already present:

```bash
bash scripts/setup_m1_encoder.sh
.venv-m1/bin/python scripts/check_m1_data.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/extract_dinov2_features.py
.venv-m1/bin/python scripts/run_m1_baselines.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/train_m1.py --stage tune
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/train_m1.py --stage main
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/evaluate_m1.py
.venv-m1/bin/python scripts/visualize_m1.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python -m pytest tests -q
.venv-m1/bin/python scripts/check_m1_data.py --final
.venv-m1/bin/python scripts/summarize_m1.py
```

Extractor refuses to overwrite a cache. Training reuses completed identical runs
and refuses conflicting configs; tuning refuses to overwrite a locked selection.
To reproduce fresh training, use a new artifact directory/worktree and the same
M0 inputs. Do not delete the evidence from the delivered run or change test plants.

## Contracts

- B0/B1 use historical mask-derived phenotypes: **Oracle-phenotype**, not RGB-only.
- B1b is polynomial ridge regression of observed time metadata; "age" means time
  since first observation, not sowing age. No genotype/plant identity is a feature.
- B2a/B2b use exactly the same frozen official DINO cache and time conditions.
  No historical phenotype, target RGB or target mask enters the neural models.
- Letterboxing avoids leaf cropping and uses official ImageNet normalization.
  Encoder output dimension 384 is verified by real GPU execution.
- Three separate horizons (+2/+4/+8 original observation offsets), three seeds
  (42/43/44) and train-only target/time normalization. The true requested future
  wall-clock interval is an explicit query condition and saved per prediction.
- Hyperparameters tuned at +4 using validation plants; checkpoints selected by
  mean target-normalized validation plant-macro MAE, never by test performance.
- Main metric: mean within-plant MAE across 46 test plants. Macro RMSE is the
  mean of each plant's RMSE; window RMSE is reported separately. Clustered CIs
  resample plants after seed-averaging errors, not windows or ensemble predictions.
- `repeat_latest` and `reverse_history` are post-hoc input interventions on fixed
  GRU checkpoints, not separately trained capacity-controlled models.

Code and summary reports are committed on `m1/plant-2d-baseline`, never merged
into main or pushed without a separate user request. Weights, embeddings,
checkpoints and individual predictions are in `outputs/m1/` on this machine.
Verified external sources and unavailable search routes: `reports/m1/docs/SOURCES.md`.
