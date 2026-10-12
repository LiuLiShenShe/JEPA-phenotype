#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m5.pretrain import budget_diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    budget_diagnostic(arguments.root, torch.device(arguments.device))
