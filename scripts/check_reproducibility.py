#!/usr/bin/env python3
"""Rebuild all M0 indexes and require byte-identical outputs with seed 42."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_phenotypes import build_phenotypes
from build_temporal_windows import build_windows
from create_splits import create_splits
from prepare_metadata import scan_dataset


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    metadata = PROJECT_ROOT / "data/metadata"
    report_dir = PROJECT_ROOT / "reports/m0"
    frames, split, windows = metadata / "frames.csv", PROJECT_ROOT / "splits/phenoleaf_plant_split_seed42.json", PROJECT_ROOT / "data/processed/temporal_windows.csv"
    paths = [metadata / name for name in ("frames.csv", "plants.csv", "phenotypes.csv", "leaf_instances.csv", "data_issues.csv", "dataset_statistics.json")]
    paths += [split, windows, report_dir / "split_statistics.csv", report_dir / "phenotype_statistics.json"]
    before = {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in paths}
    scan_dataset(PROJECT_ROOT / "data/raw", metadata, project_root=PROJECT_ROOT)
    build_phenotypes(frames, metadata, project_root=PROJECT_ROOT)
    create_splits(frames, split, report_dir / "split_statistics.csv", report_dir / "split_distribution.png", seed=42)
    build_windows(frames, split, windows, report_dir / "temporal_window_statistics.json", history_length=3, horizons=(2, 4, 8))
    after = {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in paths}
    changed = [path for path in before if before[path] != after[path]]
    report = {"status": "PASS" if not changed else "FAIL", "seed": 42, "changed_files": changed, "sha256_before": before, "sha256_after": after}
    (report_dir / "reproducibility.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if not key.startswith("sha256_")}, indent=2))
    return 0 if not changed else 1


if __name__ == "__main__":
    raise SystemExit(main())
