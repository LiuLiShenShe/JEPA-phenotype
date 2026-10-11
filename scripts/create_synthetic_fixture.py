#!/usr/bin/env python3
"""Create a small local-only fixture for validating the M0 pipeline."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw
from openpyxl import Workbook

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.phenoleaf_utils import OFFICIAL_PALETTE  # noqa: E402


def create_fixture(output_root: str | Path) -> Path:
    output_root = Path(output_root)
    dataset_root = output_root / "leaf_dataset_colour"
    raw_root = dataset_root / "01_raw_dataset"
    mask_root = dataset_root / "02_leaf_labels"
    start = datetime(2022, 5, 1, 12, 0, 0)
    for genotype_index, genotype in enumerate(("G1", "G2", "G3")):
        for replicate in range(1, 5):
            replicate_name = f"rep_{replicate:02d}"
            raw_dir = raw_root / genotype / replicate_name
            mask_dir = mask_root / genotype / replicate_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in range(14):
                timestamp = start + timedelta(hours=6 * frame_index + genotype_index)
                filename = (
                    f"1000{genotype_index}_{timestamp:%Y_%m_%d_%H_%M_%S}-"
                    f"{frame_index}-{genotype}-{replicate_name}.png"
                )
                rgb = Image.new("RGB", (64, 64), (30 + genotype_index, 40 + replicate, 50))
                ImageDraw.Draw(rgb).rectangle((0, 0, frame_index + 1, 2), fill=(255, 255, 255))
                rgb.save(raw_dir / filename)
                mask = Image.new("RGB", (64, 64), (0, 0, 0))
                mask_draw = ImageDraw.Draw(mask)
                leaf_count = min(4 + (frame_index + 1) // 2, 11)
                for leaf_id in range(leaf_count):
                    x = 3 + (leaf_id % 6) * 10
                    y = 5 + (leaf_id // 6) * 22
                    mask_draw.rectangle((x, y, x + 5 + frame_index % 3, y + 5), fill=OFFICIAL_PALETTE[leaf_id])
                mask.save(mask_dir / filename)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["leaf_id", "R", "G", "B"])
    for leaf_id, color in enumerate(OFFICIAL_PALETTE, start=1):
        sheet.append([leaf_id, *color])
    workbook.save(dataset_root / "leaves_unique_colours.xlsx")
    return dataset_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="data/synthetic_m0")
    args = parser.parse_args()
    print(create_fixture(args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
