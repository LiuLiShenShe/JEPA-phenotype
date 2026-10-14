#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
for method in single_RF history_RF single_PointNet PointNet_GRU forecast_RF C1 C2; do
  for seed in 17 29 43; do
    .venv-m6-b/bin/python -u scripts/train_birth_detector.py --method "$method" --seed "$seed" --device cuda:1 >> "reports/m7/train_${method}_${seed}.log" 2>&1
  done
done
.venv-m6-b/bin/python -u scripts/evaluate_birth.py --split val > reports/m7/evaluate_val.log 2>&1
.venv-m6-b/bin/python -u scripts/evaluate_birth.py --split test > reports/m7/evaluate_test.log 2>&1
