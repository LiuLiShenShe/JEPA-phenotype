#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1
run_seed() {
    local seed="$1"
    .venv-m6-b/bin/python scripts/run_m6_b.py train --method S2 --seed "$seed" > "reports/m6_b/train_S2_${seed}_gpu1.log" 2>&1
    .venv-m6-b/bin/python scripts/run_m6_b.py infer --method S2 --seed "$seed" --splits val > "reports/m6_b/infer_S2_${seed}_val_gpu1.log" 2>&1
}
processes=()
for seed in 17 29 43; do
    run_seed "$seed" &
    processes+=("$!")
done
for process in "${processes[@]}"; do
    wait "$process"
done
