#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ ! -x .venv-m4/bin/python ]]; then
  uv venv --python 3.11 .venv-m4
fi
uv pip install --python .venv-m4/bin/python -r requirements-m4.txt
.venv-m4/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
