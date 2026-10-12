#!/usr/bin/env python
"""Verify, extract, audit and index a local Sugar4D archive."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m3.common import file_hash, write_json
from datasets.sugar4d_dataset import audit_dataset, build_windows
from m3.audit import detailed_audit, make_folds


def verify_archive(path: Path, expected_md5: str, expected_size: int):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Archive size mismatch: {path.stat().st_size} != {expected_size}")
    digest = file_hash(path, "md5")
    if digest != expected_md5:
        raise ValueError(f"Archive MD5 mismatch: {digest} != {expected_md5}")
    return {"path": str(path), "bytes": path.stat().st_size, "md5": digest, "sha256": file_hash(path)}


def safe_extract(path: Path, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)
        return {"members": len(archive.infolist()), "zip_test": "pass"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/sugar4d/sugar4d.zip"))
    parser.add_argument("--extract-root", type=Path, default=Path("data/raw/sugar4d/extracted_v4"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/sugar4d"))
    parser.add_argument("--expected-md5", default="27e44210d95a85cb04b93513f0ef2beb")
    parser.add_argument("--expected-bytes", type=int, default=6270780517)
    parser.add_argument("--mapping", type=Path, default=Path("reports/m3/sources/genotype_mapping.json"))
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()
    archive_report = verify_archive(args.archive, args.expected_md5, args.expected_bytes)
    if (args.extract_root / "point_clouds").exists():
        with zipfile.ZipFile(args.archive) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP CRC failure: {bad}")
            extract_report = {"members": len(archive.infolist()), "zip_test": "pass", "reused_verified_extraction": True}
    else:
        extract_report = safe_extract(args.archive, args.extract_root)
    metadata, leaves, report = audit_dataset(args.extract_root, mapping_path=args.mapping)
    protocol = json.loads((ROOT / "configs/m3/protocol.json").read_text())
    if args.detailed:
        metadata, detailed = detailed_audit(ROOT, metadata, leaves, protocol)
        report["detailed"] = detailed
        if report["genotype_mapping_complete"]:
            make_folds(metadata, None, protocol, ROOT / "reports/m3")
    windows = build_windows(metadata)
    args.processed_root.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(args.processed_root / "plant_metadata.csv", index=False)
    leaves.to_csv(args.processed_root / "leaf_metadata.csv", index=False)
    windows.to_csv(args.processed_root / "forecast_windows.csv", index=False)
    report.update({"archive": archive_report, "extract": extract_report, "windows": {"rows": len(windows), "by_split": windows.groupby("split").size().to_dict(), "by_horizon": windows.groupby("horizon").size().to_dict()}})
    write_json(args.processed_root / "audit_report.json", report)
    write_json(ROOT / "reports/m3/data_audit.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
