#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4.evaluation import summarize
from m4.figures import make_figures
from m4.report import generate_report
from m4.inference import verify_matrix_restore


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    root = Path(arguments.root)
    if arguments.verify_checkpoints:
        import torch
        verify_matrix_restore(root, torch.device(arguments.device))
    _, _, _, _, coverage, decision = summarize(root)
    if not arguments.no_figures:
        make_figures(root)
    generate_report(root)
    print(f"{decision['decision']} count={coverage['count_completed']}/{coverage['count_expected']} segmentation={coverage['segmentation_completed']}/{coverage['segmentation_expected']}")
