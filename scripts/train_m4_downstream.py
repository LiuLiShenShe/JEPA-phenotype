#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m4.downstream import run_count, run_segmentation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--partitions", default="official,fold0,fold1,fold2,fold3")
    parser.add_argument("--seeds", default="17,42,93")
    parser.add_argument("--conditions", default="P0,P2")
    parser.add_argument("--fractions", default="0.25,0.5,1.0")
    parser.add_argument("--modes", default="probe,finetune")
    parser.add_argument("--tasks", default="count,segmentation")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    device, root = torch.device(arguments.device), Path(arguments.root)
    for partition in arguments.partitions.split(","):
        for seed in map(int, arguments.seeds.split(",")):
            for condition in arguments.conditions.split(","):
                for mode in arguments.modes.split(","):
                    if "count" in arguments.tasks.split(","):
                        for fraction in map(float, arguments.fractions.split(",")):
                            run_count(root, partition, seed, condition, fraction, mode, device)
                    if "segmentation" in arguments.tasks.split(","):
                        run_segmentation(root, partition, seed, condition, mode, device)
            if "segmentation" in arguments.tasks.split(","):
                run_segmentation(root, partition, seed, "PointNet", "finetune", device)
