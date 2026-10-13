#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-m6-b/bin/python ]; then
    uv venv --python .venv-m4/bin/python .venv-m6-b
fi
uv pip install --python .venv-m6-b/bin/python --link-mode copy --index-strategy unsafe-best-match \
    torch==2.8.0 numpy==2.2.6 scipy==1.17.1 pandas==2.2.3 scikit-learn==1.7.2 \
    matplotlib==3.10.6 open3d==0.19.0 pytest==8.4.2 requests==2.34.2 h5py==3.15.1 \
    openpyxl==3.1.5 --extra-index-url https://download.pytorch.org/whl/cu128
uv pip freeze --python .venv-m6-b/bin/python > reports/m6_b/environment-lock.txt
