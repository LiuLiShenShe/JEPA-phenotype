#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m4.common import load_protocol
from m4.pretrain import run_pretrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--partitions", default="official,fold0,fold1,fold2,fold3")
    parser.add_argument("--seeds", default="17,42,93")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    for partition in arguments.partitions.split(","):
        for seed in map(int, arguments.seeds.split(",")):
            run_pretrain(Path(arguments.root), partition, seed, torch.device(arguments.device))
