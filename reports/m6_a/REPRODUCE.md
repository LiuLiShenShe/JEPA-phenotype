# M6-A reproduction

No command trains a model, merges, commits or pushes. Run from the repository root.

## Preconditions
- Audited Sugar4D V4 raw files and M3 official plant split must be present and hash-correct.
- Two official epoch199 models and TrackPlant3D code are local in ignored `data/raw/m6_a/official`. Source audit can retrieve them, but no upstream redistribution license is assumed.
- Linux, Python3.11, CUDA-compatible GPU for the unchanged PlantNet model with hard-coded CUDA layers; approximately 4GB additional environment/cache disk space. Historical environments are not modified.

```bash
bash scripts/setup_m6_a_environment.sh
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/audit_m6_a_sources.py --download
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py freeze
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py smoke
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-m6-a/bin/python scripts/run_m6_a.py prepare
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-m6-a/bin/python scripts/run_m6_a.py evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py report
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-m6-a/bin/python -m pytest tests/test_m6_a.py -q
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py verify
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/verify_m6_a.py --repeat
```

## Optional two-GPU inference

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 taskset -c 0-15 .venv-m6-a/bin/python scripts/run_m6_a.py prepare --splits val > reports/m6_a/prepare_val.log 2>&1 &
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 taskset -c 32-47 .venv-m6-a/bin/python scripts/run_m6_a.py prepare --splits test > reports/m6_a/prepare_test.log 2>&1 &
wait
```

CPU affinity is optional and host-specific; it is not an algorithm parameter. Val/test have disjoint cache paths. Evaluator may run `evaluate --wait-for-cache` while inference proceeds, never evaluating a partial plant.

`prepare` uses cached full-resolution predictions only if their protocol lock hash agrees. Changing protocol/checkpoint raises instead of reusing stale results. `freeze` never overwrites an existing mismatching lock. New upstream source revisions are not substituted for fixed model weights.

`setup_m6_a_environment.sh` uses read-only `.pth` access to existing torch, plus local pinned geometry dependencies. The `.pth` path is host-specific; `environment.json` and `environment-lock.txt` describe the actual runtime. For an independent machine, install torch2.8.0+cu128 and the listed packages in its own environment instead of copying the host path.

`audit_m6_a_sources.py --search` reads `BOCHA_API_KEY` from environment; no credential is stored in source or delivered logs. On this host search API returned403; official GitHub/Dataverse direct evidence is saved. Pepper Drive requests failed and do not become downloaded datasets.
