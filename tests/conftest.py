from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from openpyxl import Workbook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from datasets.phenoleaf_utils import OFFICIAL_PALETTE  # noqa: E402


@pytest.fixture
def synthetic_project(tmp_path: Path) -> Path:
    """Create a compact multi-genotype temporal dataset for all M0 tests."""

    dataset_root = tmp_path / "data" / "leaf_dataset_colour"
    raw_root = dataset_root / "01_raw_dataset"
    mask_root = dataset_root / "02_leaf_labels"
    start = datetime(2022, 5, 1, 12, 0, 0)
    genotypes = ("G1", "G2", "G3")
    replicates = range(1, 5)
    frame_count = 14
    for genotype_index, genotype in enumerate(genotypes):
        for replicate in replicates:
            replicate_name = f"rep_{replicate:02d}"
            raw_dir = raw_root / genotype / replicate_name
            mask_dir = mask_root / genotype / replicate_name
            raw_dir.mkdir(parents=True, exist_ok=True)
            mask_dir.mkdir(parents=True, exist_ok=True)
            for frame_index in range(frame_count):
                timestamp = start + timedelta(hours=6 * frame_index + genotype_index)
                filename = (
                    f"1000{genotype_index}_{timestamp:%Y_%m_%d_%H_%M_%S}-"
                    f"{frame_index}-{genotype}-{replicate_name}.png"
                )
                rgb = Image.new("RGB", (64, 64), (30 + genotype_index, 40 + replicate, 50))
                rgb_draw = ImageDraw.Draw(rgb)
                rgb_draw.rectangle((0, 0, frame_index + 1, 2), fill=(255, 255, 255))
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
    return tmp_path


@pytest.fixture
def workflow(synthetic_project):
    from build_phenotypes import build_phenotypes
    from build_temporal_windows import build_windows
    from create_splits import create_splits
    from prepare_metadata import scan_dataset

    metadata_dir = synthetic_project / "data" / "metadata"
    reports_dir = synthetic_project / "reports" / "m0"
    split_path = synthetic_project / "splits" / "phenoleaf_plant_split_seed42.json"
    windows_path = synthetic_project / "data" / "processed" / "temporal_windows.csv"
    scan_stats = scan_dataset(
        synthetic_project / "data" / "leaf_dataset_colour",
        metadata_dir,
        project_root=synthetic_project,
    )
    phenotype_stats = build_phenotypes(
        metadata_dir / "frames.csv",
        metadata_dir,
        project_root=synthetic_project,
        palette_path=synthetic_project / "data" / "leaf_dataset_colour" / "leaves_unique_colours.xlsx",
    )
    split_stats = create_splits(
        metadata_dir / "frames.csv",
        split_path,
        reports_dir / "split_statistics.csv",
        reports_dir / "split_distribution.png",
    )
    window_stats = build_windows(
        metadata_dir / "frames.csv",
        split_path,
        windows_path,
        reports_dir / "temporal_window_statistics.json",
    )
    return {
        "root": synthetic_project,
        "metadata_dir": metadata_dir,
        "split_path": split_path,
        "windows_path": windows_path,
        "scan_stats": scan_stats,
        "phenotype_stats": phenotype_stats,
        "split_stats": split_stats,
        "window_stats": window_stats,
    }
