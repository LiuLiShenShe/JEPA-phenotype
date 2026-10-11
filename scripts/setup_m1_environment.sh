#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [ -e .venv-m1 ]; then
  printf 'M1 environment already exists; refusing to replace it.\n' >&2
  exit 1
fi
nvidia-smi
uv venv --python 3.11 .venv-m1
uv pip install --python .venv-m1/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-m1/bin/python -r requirements-m1-lock.txt
uv pip check --python .venv-m1/bin/python
.venv-m1/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), torch.ones(4,device="cuda:0").sum().item())'
