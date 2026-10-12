#!/usr/bin/env python
"""Validate immutable inputs, legacy protection and complete M3 training contracts."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m3.common import file_hash, tensor_state_hash, write_json


def validate_training(root):
    protocol = json.loads((root / "configs/m3/protocol.json").read_text())
    lock = json.loads((root / "reports/m3/protocol_lock.json").read_text())
    if file_hash(root / "configs/m3/protocol.json") != lock["sha256"]:
        raise ValueError("Protocol changed after lock")
    checks = []
    for partition in ("official", "fold0", "fold1", "fold2", "fold3"):
        for point_count in (2048, 4096) if partition == "official" else (2048,):
            directory = root / "outputs/m3" / partition / f"points{point_count}"
            for seed in protocol["seeds"]:
                records = {}
                for model in ("B3-2", "B3-3", "B3-4"):
                    record_path = directory / f"{model}_seed{seed}_training.json"
                    record = json.loads(record_path.read_text())
                    if record["epochs"] != protocol["epochs"] or record["optimizer_steps"] != 500 or len(record["history"]) != protocol["epochs"]:
                        raise ValueError("Incomplete training budget")
                    if set(record["training_plants"]) & set(record["validation_plants"]):
                        raise ValueError("Train/validation plant leak")
                    if not np.isfinite([value for row in record["history"] for value in (row["train_mse"], row["validation_macro_normalized_mae"])]).all():
                        raise ValueError("Nonfinite training history")
                    initial = torch.load(directory / f"{model}_seed{seed}_initial.pt", map_location="cpu", weights_only=True)
                    if tensor_state_hash(initial) != record["initial_tensor_sha256"]:
                        raise ValueError("Initial tensor state changed")
                    for path, expected in record["fingerprints"].items():
                        if file_hash(root / path) != expected:
                            raise ValueError(f"Training source/input changed after run: {path}")
                    records[model] = record
                    checks.append({key: record[key] for key in ("partition", "model", "seed", "points", "parameters", "epochs", "optimizer_steps", "best_epoch", "seconds", "peak_cuda_bytes", "initial_tensor_sha256")})
                if records["B3-3"]["parameters"] != records["B3-4"]["parameters"] or records["B3-3"]["initial_tensor_sha256"] != records["B3-4"]["initial_tensor_sha256"]:
                    raise ValueError("Real/repeat capacity or initialization not matched")
    pd.DataFrame(checks).to_csv(root / "reports/m3/training_resources.csv", index=False)
    return {"neural_runs": len(checks), "runs_each_complete": True, "matched_real_repeat": True, "protocol_unchanged": True,
            "total_optimizer_steps": sum(row["optimizer_steps"] for row in checks), "total_training_seconds": sum(row["seconds"] for row in checks)}


def validate_immutable(root):
    legacy = json.loads((root / "reports/m3/legacy_snapshot.json").read_text())
    failures = []
    for path, expected in legacy["files"].items():
        source = root / path
        if not source.is_file() or file_hash(source) != expected["sha256"]:
            failures.append(path)
    raw = json.loads((root / "reports/m3/raw_file_manifest.json").read_text())
    raw_failures = []
    for record in raw["files"]:
        if file_hash(root / record["path"]) != record["sha256"]:
            raw_failures.append(record["path"])
    if failures or raw_failures:
        raise ValueError(f"Immutable file changed: legacy={failures}, raw={raw_failures}")
    return {"legacy_files_checked": len(legacy["files"]), "legacy_changes": failures, "raw_files_checked": len(raw["files"]), "raw_changes": raw_failures}


def validate_archive(root):
    archive_path = root / "data/raw/sugar4d/sugar4d.zip"
    readme_path = root / "data/raw/sugar4d/sugar4d_readme.txt"
    if archive_path.stat().st_size != 6270780517 or file_hash(archive_path, "md5") != "27e44210d95a85cb04b93513f0ef2beb":
        raise ValueError("Official archive integrity failed")
    if file_hash(readme_path, "md5") != "3959c3c687cf10f5b51b41bb03f95bd3":
        raise ValueError("Official readme checksum failed")
    with zipfile.ZipFile(archive_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC mismatch: {bad}")
        members = len(archive.infolist())
    sha256 = file_hash(archive_path)
    for audit_path in (root / "reports/m3/data_audit.json", root / "data/processed/sugar4d/audit_report.json"):
        payload = json.loads(audit_path.read_text())
        payload["archive"]["sha256"] = sha256
        payload["extract"] = {"members": members, "zip_test": "pass", "reused_verified_extraction": True}
        payload["readme"] = {"md5": file_hash(readme_path, "md5"), "sha256": file_hash(readme_path), "bytes": readme_path.stat().st_size}
        write_json(audit_path, payload)
    return {"archive_sha256": sha256, "members_crc_checked": members, "readme_md5_verified": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    result = validate_immutable(ROOT)
    if args.archive:
        result["archive"] = validate_archive(ROOT)
    if not args.skip_training:
        result["training"] = validate_training(ROOT)
    result["status"] = "PASS"
    write_json(ROOT / "reports/m3/final_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
