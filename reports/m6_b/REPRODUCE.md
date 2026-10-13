# M6-B reproduction (no commit or push)

Run from `/fj/JEPA-表型`; default replay selectsGPU0, never kill other GPU jobs.
Keep upstream/licensed sources and all model weights local.

```bash
bash scripts/setup_m6_b_environment.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0
.venv-m6-b/bin/python scripts/audit_m6_b_sources.py
.venv-m6-b/bin/python scripts/run_m6_b.py snapshot
.venv-m6-b/bin/python scripts/run_m6_b.py smoke
.venv-m6-b/bin/python scripts/run_m6_b.py freeze
.venv-m6-b/bin/python scripts/run_m6_b.py data
.venv-m6-b/bin/python scripts/run_m6_b.py training-cache
.venv-m6-b/bin/python scripts/prepare_m6_b_tracking_geometry.py
bash scripts/train_m6_b_matrix.sh
PYTHONPATH=src .venv-m6-b/bin/python -m pytest -q -p no:cacheprovider tests/test_m6_b.py
```

Actual execution later used the verified newly idle identical GPU1 for S2 via
`bash scripts/train_m6_b_finetune_gpu1.sh`; `RESOURCE_SCHEDULING.md` records this
wall-clock scheduling change. The per-run CLI lock serializes duplicated launches;
completed epoch60 resumes take zero extra training updates. Do not choose a GPU
that is currently running unrelated jobs without considering shared contention.
Actual S2 resumes onGPU0 at intact epoch37 boundaries after shared GPU1 became
contended; checkpoint hashes/full-state evidence are in `S2_device_migration.json`.

All fixed resources need to match the original lock. A new live retrieval may
change response hashes; do not overwrite a completed execution lock or claim it
is the same registered run. For exact local replay, use already verified sources
and skip audit/download, smoke and snapshot. `freeze` refuses changed locked inputs.

Training resumes from `outputs/m6_b/S*/seed_*/latest.pth` at epoch boundaries;
all native-model state, optimizer and RNG states are preserved. Completed runs
have `training_complete.json`. Logs include every optimizer step and epoch.
Delete/relocate only the **new M6-B run directory** to deliberately reproduce
from scratch; never remove historical M0–M6-A artifacts. Three seeds are not
three independent biological cohorts.

After all Validation runs finish, select the family by three-seed mean PQ:

```bash
PYTHONPATH=src .venv-m6-b/bin/python -c 'from pathlib import Path; from m6b.analysis import select_downstream; print(select_downstream(Path.cwd()))'
for method in S1 S2 S3; do
  .venv-m6-b/bin/python scripts/run_m6_b.py infer --method "$method" --seed 17 --splits test
done
.venv-m6-b/bin/python scripts/run_m6_b.py track --method S1 --seed 17 --splits val test
.venv-m6-b/bin/python scripts/run_m6_b.py track --method S2 --seed 17 --splits val test
.venv-m6-b/bin/python scripts/run_m6_b.py track --method S3 --seed 17 --splits val test
.venv-m6-b/bin/python scripts/run_m6_b.py verify
CUDA_VISIBLE_DEVICES=0 .venv-m6-b/bin/python scripts/benchmark_m6_b_inference.py
.venv-m6-b/bin/python scripts/run_m6_b.py report
.venv-m6-b/bin/python scripts/verify_m6_b.py
.venv-m6-b/bin/python scripts/run_m6_b.py manifest
```

The final report maps B3 only to the Validation-selected family and fixed seed17.
Tracking both families is an additional disclosed diagnostic, not Test selection.
Test inference happens only after Validation checkpoint and bandwidth selection.
It is an exploratory reused cohort, not untouched confirmation.

Organ3DNet dependency attempt (separate environment):

```bash
uv venv --python 3.10 .venv-m6-b-organ3d
uv pip install --python .venv-m6-b-organ3d/bin/python --link-mode copy --index-strategy unsafe-best-match torch==1.12.1+cu113 torchvision==0.13.1+cu113 numpy==1.23.5 hydra-core==1.0.5 ninja 'setuptools<70' --extra-index-url https://download.pytorch.org/whl/cu113
.venv-m6-b/bin/python scripts/smoke_m6_b_organ3d.py
```

Do not bypass CUDA major-version mismatch. Source: official repository README at
https://github.com/Huang2002200/Organ3DNet ; details in `SOURCE_AUDIT.md` and actual
failed build logs. S4 stays BLOCKED rather than using a simplified architecture.
