#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5 import evaluation
from m5.artifacts import write_json
from m5.figures import make_figures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    arguments = parser.parse_args()
    evaluation.write_json = write_json
    evaluation.summarize(arguments.root)
    make_figures(arguments.root)
