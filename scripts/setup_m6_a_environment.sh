#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv-m6-a/bin/python ]; then
    uv venv --python .venv-m4/bin/python .venv-m6-a
fi
site_dir="$(.venv-m6-a/bin/python -c 'import site; print(site.getsitepackages()[0])')"
legacy_site="$(.venv-m4/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "$legacy_site" > "$site_dir/m6_readonly_legacy.pth"
uv pip install --python .venv-m6-a/bin/python open3d==0.19.0 numpy==2.2.6 pandas==2.2.3 scipy==1.17.1 scikit-learn==1.7.2 matplotlib==3.10.6
uv pip freeze --python .venv-m6-a/bin/python > reports/m6_a/environment-lock.txt
