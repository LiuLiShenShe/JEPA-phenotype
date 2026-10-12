#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m4.common import write_json


def capture(root):
    root = Path(root)
    packages = sorted((distribution.metadata["Name"], distribution.version)
                      for distribution in importlib.metadata.distributions())
    (root / "reports/m5/environment_freeze.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in packages) + "\n")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                             "--format=csv,noheader"], capture_output=True, text=True, check=True)
    record = {"date": "2026-10-12", "python": platform.python_version(), "platform": platform.platform(),
              "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "cudnn": torch.backends.cudnn.version(), "gpus": result.stdout.strip().splitlines(),
              "environment": ".venv-m4", "packages": dict(packages), "shared_gpu_load": True,
              "determinism": "m4.common.seed_all: Python/NumPy/torch/CUDA seeds; deterministic algorithms; one torch CPU thread"}
    write_json(root / "reports/m5/environment.json", record)
    print(json.dumps({name: record[name] for name in ("python", "pytorch", "cuda_runtime", "gpus")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    arguments = parser.parse_args()
    capture(arguments.root)
