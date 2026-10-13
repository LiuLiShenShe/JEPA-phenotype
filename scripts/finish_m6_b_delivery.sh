#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
while ! test -f outputs/m6_b/S3/seed_17/inference_test.json; do
    sleep 60
done
PYTHONPATH=src .venv-m6-b/bin/python -c 'from pathlib import Path; from m6b.analysis import select_downstream; print(select_downstream(Path.cwd()))' > reports/m6_b/downstream_selection.log 2>&1
for method in S1 S2 S3; do
    .venv-m6-b/bin/python scripts/run_m6_b.py track --method "$method" --seed 17 --splits val test > "reports/m6_b/track_${method}_17.log" 2>&1
done
.venv-m6-b/bin/python scripts/run_m6_b.py verify > reports/m6_b/protection_verification.log 2>&1
PYTHONPATH=src .venv-m6-b/bin/python -m pytest -q -p no:cacheprovider tests/test_m6_b.py tests/test_m6_a.py --junitxml=reports/m6_b/tests_m6_ab_final.xml > reports/m6_b/tests_m6_ab_final.log 2>&1
.venv-m6-b/bin/python scripts/benchmark_m6_b_inference.py > reports/m6_b/inference_microbenchmark.log 2>&1
.venv-m6-b/bin/python scripts/run_m6_b.py report > reports/m6_b/report.log 2>&1
.venv-m6-b/bin/python scripts/verify_m6_b.py > reports/m6_b/final_verification.log 2>&1
.venv-m6-b/bin/python scripts/run_m6_b.py manifest
printf 'Finished M6-B numerical delivery; figures require direct visual review.\n'
