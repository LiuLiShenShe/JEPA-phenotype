#!/usr/bin/env python3
"""Read-only full M0 hash audit and actual-time M1 data verification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.m1_feature_dataset import M1Index
from training.common import file_hash, write_json


def audit(root: Path, final: bool = False):
    directory = root / "reports/m1"
    snapshot_path = directory / "m0_snapshot.json"
    if not snapshot_path.exists():
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        paths = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
        paths.extend(["data/metadata/frames.csv", "data/metadata/file_manifest.csv",
                      "data/processed/temporal_windows.csv", "splits/phenoleaf_plant_split_seed42.json"])
        snapshot = {"base_commit": base_commit, "files": {path: file_hash(root / path) for path in paths if (root / path).is_file()}}
        write_json(snapshot_path, snapshot)
    snapshot = json.loads(snapshot_path.read_text())
    changed = [path for path, digest in snapshot["files"].items() if not (root / path).is_file() or file_hash(root / path) != digest]
    if changed:
        raise RuntimeError(f"M0 protected files changed: {changed}")
    index = M1Index(root)
    manifest = pd.read_csv(root / "data/metadata/file_manifest.csv")
    jobs = [(path, digest) for row in manifest.itertuples() for path, digest in [(row.rgb_path, row.rgb_sha256), (row.mask_path, row.mask_sha256)]]
    def verify(job):
        path, expected = job
        if file_hash(root / path) != expected:
            return path
        return None
    with ThreadPoolExecutor(max_workers=8) as executor:
        mismatches = [path for path in executor.map(verify, jobs) if path is not None]
    if mismatches:
        raise RuntimeError(f"Original data no longer agrees with M0 hashes: {mismatches[:10]}")
    window_counts = index.rows.groupby(["split", "plant_uid", "genotype", "forecast_horizon"]).size().rename("windows").reset_index()
    window_counts.to_csv(directory / "windows_per_plant.csv", index=False)
    time_distribution = index.rows.groupby(["split", "forecast_horizon"]).target_time_delta_seconds.describe(percentiles=[.05,.25,.5,.75,.95])
    time_distribution.to_csv(directory / "actual_time_distribution.csv")
    valid = index.frames.loc[index.frames.quality_status.eq("valid")]
    counts = {split: {"plants": len(index.split[f"{split}_plants"]), "images": int(valid.plant_uid.map(index.split["assignments"]).eq(split).sum()), "windows": int(index.rows.split.eq(split).sum())} for split in ("train", "val", "test")}
    expected = {"train": {"plants":219,"images":11982,"windows":31564},"val":{"plants":53,"images":2815,"windows":7385},"test":{"plants":46,"images":2259,"windows":5857}}
    if counts != expected or len(index.rows) != 44806 or len(valid) != 17056:
        raise RuntimeError(f"Local state differs from required M0 version: {counts}")
    report = {
        "status": "PASS", "phase": "final" if final else "initial", "m0_unchanged": True,
        "original_files_hashed": len(jobs), "raw_hash_mismatches": mismatches,
        "protected_files_hashed": len(snapshot["files"]), "m0_base_commit": snapshot["base_commit"],
        "frames":len(index.frames), "valid_frames":len(valid), "quarantined_frames":len(index.frames)-len(valid),
        "plants":int(index.frames.plant_uid.nunique()), "genotypes":int(index.frames.genotype.nunique()),
        "windows":len(index.rows), "counts":counts,
        "horizons":{str(key):int(value) for key,value in index.rows.forecast_horizon.value_counts().sort_index().items()},
        "unique_image_sizes":int(valid.groupby(["image_width","image_height"]).ngroups),
        "plants_with_variable_size":valid.groupby("plant_uid").apply(lambda group: len(group[["image_width","image_height"]].drop_duplicates()), include_groups=False).loc[lambda series:series>1].index.tolist(),
        "horizon_unit":"original observation offset; never fixed days",
        "time_unit":"uncalibrated timezone-naive wall-clock seconds",
        "age_definition":"time since first recorded observation; sowing age unavailable",
        "targets":["target_leaf_count","target_projected_area_ratio"],
        "time_conditions_are_query_metadata":True,
    }
    write_json(directory / ("data_audit_final.json" if final else "data_audit.json"), report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    audit(args.root, args.final)
