#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1
run_seed() {
    local seed="$1"
    for method in S1 S2; do
        .venv-m6-b/bin/python scripts/run_m6_b.py train --method "$method" --seed "$seed" > "reports/m6_b/train_${method}_${seed}.log" 2>&1
        .venv-m6-b/bin/python scripts/run_m6_b.py infer --method "$method" --seed "$seed" --splits val > "reports/m6_b/infer_${method}_${seed}_val.log" 2>&1
    done
}
run_seed 17 &
process_17=$!
run_seed 29 &
process_29=$!
run_seed 43 &
process_43=$!
wait "$process_17"
wait "$process_29"
wait "$process_43"
if ! test -f outputs/m6_b/S3/seed_17/training_complete.json; then
    .venv-m6-b/bin/python scripts/run_m6_b.py train --method S3 --seed 17 > reports/m6_b/train_S3_17.log 2>&1
fi
if ! test -f outputs/m6_b/S3/seed_17/inference_val.json; then
    .venv-m6-b/bin/python scripts/run_m6_b.py infer --method S3 --seed 17 --splits val > reports/m6_b/infer_S3_17_val.log 2>&1
fi
for method in S1 S2 S3; do
    .venv-m6-b/bin/python scripts/run_m6_b.py infer --method "$method" --seed 17 --splits test > "reports/m6_b/infer_${method}_17_test.log" 2>&1
done
