#!/usr/bin/env python3
"""Independently audit every extracted frame against ZIP CRCs and CSV labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from datasets.phenoleaf_utils import OFFICIAL_PALETTE, resolve_project_path, write_csv


def read_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_dataset(metadata_dir, project_root, archive_path, report_dir, workers=8):
    metadata_dir, project_root, report_dir = Path(metadata_dir), Path(project_root), Path(report_dir)
    frames = read_rows(metadata_dir / "frames.csv")
    if not frames:
        raise ValueError("Cannot audit an empty dataset")
    leaves = defaultdict(dict)
    for row in read_rows(metadata_dir / "leaf_instances.csv"):
        key = (row["plant_uid"], int(row["frame_idx"]))
        leaf_id = int(row["leaf_id"])
        if leaf_id in leaves[key]:
            raise ValueError(f"Duplicate leaf instance row: {key}, leaf {leaf_id}")
        leaves[key][leaf_id] = int(row["area_px"])
    palette_codes = {
        (color[0] << 16) | (color[1] << 8) | color[2]: leaf_id
        for leaf_id, color in enumerate(OFFICIAL_PALETTE, 1)
    }
    zip_manifest = {}
    archive_path = Path(archive_path)
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                key = str((archive_path.parent / member.filename).resolve())
                if key in zip_manifest:
                    raise ValueError("Archive contains duplicate member names")
                zip_manifest[key] = (member.file_size, member.CRC)

    def audit_frame(row):
        errors = []
        result = {"plant_uid": row["plant_uid"], "frame_idx": row["frame_idx"], "quality_status": row["quality_status"]}
        decoded = {}
        for kind in ("rgb", "mask"):
            path = resolve_project_path(row[f"{kind}_path"], project_root)
            result[f"{kind}_path"] = row[f"{kind}_path"]
            try:
                payload = path.read_bytes()
                result[f"{kind}_sha256"] = hashlib.sha256(payload).hexdigest()
                result[f"{kind}_bytes"] = len(payload)
                result[f"{kind}_crc32"] = f"{zlib.crc32(payload):08x}"
                if zip_manifest and zip_manifest.get(str(path.resolve())) != (len(payload), zlib.crc32(payload)):
                    errors.append(f"{kind}_differs_from_archive")
                with Image.open(path) as image:
                    image.load()
                    decoded[kind] = np.asarray(image.convert("RGB"))
                if kind == "rgb":
                    result["rgb_pixels_sha256"] = hashlib.sha256(decoded[kind].tobytes()).hexdigest()
            except Exception as exc:
                errors.append(f"{kind}_decode_error:{type(exc).__name__}")
        if set(decoded) == {"rgb", "mask"}:
            rgb, mask = decoded["rgb"], decoded["mask"]
            if rgb.shape != mask.shape:
                errors.append("dimension_mismatch")
            if rgb.shape[:2] != (int(row["image_height"]), int(row["image_width"])):
                errors.append("metadata_dimension_mismatch")
            codes = (mask[..., 0].astype(np.uint32) << 16) | (mask[..., 1].astype(np.uint32) << 8) | mask[..., 2]
            unique_codes, counts = np.unique(codes, return_counts=True)
            instances = {
                palette_codes[int(code)]: int(count)
                for code, count in zip(unique_codes, counts) if int(code) in palette_codes
            }
            invalid = sum(int(count) for code, count in zip(unique_codes, counts) if int(code) != 0 and int(code) not in palette_codes)
            area = int(np.count_nonzero(codes))
            ratio = area / (mask.shape[0] * mask.shape[1])
            if int(row["leaf_count"]) != len(instances):
                errors.append("leaf_count_mismatch")
            if int(row["projected_area_px"]) != area or int(row["known_projected_area_px"]) != sum(instances.values()):
                errors.append("area_mismatch")
            if abs(float(row["projected_area_ratio"]) - ratio) > 1e-12:
                errors.append("area_ratio_mismatch")
            if int(row["invalid_color_pixel_count"]) != invalid:
                errors.append("illegal_color_count_mismatch")
            if leaves[(row["plant_uid"], int(row["frame_idx"]))] != instances:
                errors.append("per_leaf_area_mismatch")
            result["invalid_color_pixel_count"] = invalid
            result["leaf_count"] = len(instances)
            if invalid and row["quality_status"] == "valid":
                errors.append("illegal_color_not_quarantined")
        result["errors"] = errors
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(audit_frame, frames))
    rgb_duplicates, pixel_duplicates, mask_duplicates = {}, {}, {}
    for key, destination in (("rgb_sha256", rgb_duplicates), ("rgb_pixels_sha256", pixel_duplicates), ("mask_sha256", mask_duplicates)):
        groups = defaultdict(list)
        for row in results:
            if key in row:
                groups[row[key]].append({"plant_uid": row["plant_uid"], "frame_idx": int(row["frame_idx"]), "path": row.get("rgb_path", "")})
        destination.update({digest: rows for digest, rows in sorted(groups.items()) if len(rows) > 1})
    cross_plant_duplicates = {
        digest: rows for digest, rows in pixel_duplicates.items()
        if len({row["plant_uid"] for row in rows}) > 1
    }
    archive_digest = None
    extra_file_errors = []
    if archive_path.exists():
        archive_hasher = hashlib.sha256()
        with archive_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                archive_hasher.update(chunk)
        archive_digest = archive_hasher.hexdigest()
        image_paths = {str(resolve_project_path(row[kind], project_root).resolve()) for row in frames for kind in ("rgb_path", "mask_path")}
        for file_path, expected in zip_manifest.items():
            if file_path in image_paths:
                continue
            path = Path(file_path)
            if not path.exists():
                extra_file_errors.append({"path": file_path, "error": "missing_archive_member"})
                continue
            payload = path.read_bytes()
            if (len(payload), zlib.crc32(payload) & 0xFFFFFFFF) != expected:
                extra_file_errors.append({"path": file_path, "error": "archive_member_changed"})
    report = {
        "metadata_sha256": hashlib.sha256((metadata_dir / "frames.csv").read_bytes()).hexdigest(),
        "leaf_instances_sha256": hashlib.sha256((metadata_dir / "leaf_instances.csv").read_bytes()).hexdigest(),
        "status": "PASS" if not any(row["errors"] for row in results) and not cross_plant_duplicates and not extra_file_errors else "FAIL",
        "frames_audited": len(results),
        "masks_independently_recomputed": len(results),
        "archive_crc_comparison_enabled": bool(zip_manifest),
        "archive_sha256": archive_digest,
        "archive_member_files": len(zip_manifest),
        "extra_archive_file_errors": extra_file_errors,
        "frame_errors": [{"plant_uid": row["plant_uid"], "frame_idx": row["frame_idx"], "errors": row["errors"]} for row in results if row["errors"]],
        "invalid_color_frames": sum(int(row.get("invalid_color_pixel_count", 0)) > 0 for row in results),
        "invalid_color_pixels": sum(int(row.get("invalid_color_pixel_count", 0)) for row in results),
        "quarantined_frames": sum(row["quality_status"] != "valid" for row in results),
        "duplicate_rgb_file_groups": len(rgb_duplicates),
        "duplicate_rgb_pixel_groups": len(pixel_duplicates),
        "duplicate_mask_file_groups": len(mask_duplicates),
        "cross_plant_duplicate_rgb_groups": len(cross_plant_duplicates),
        "duplicates": {"rgb_files": rgb_duplicates, "rgb_pixels": pixel_duplicates, "masks": mask_duplicates},
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "full_dataset_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["plant_uid", "frame_idx", "quality_status", "rgb_path", "mask_path", "rgb_bytes", "mask_bytes", "rgb_crc32", "mask_crc32", "rgb_sha256", "mask_sha256", "rgb_pixels_sha256", "leaf_count", "invalid_color_pixel_count", "errors"]
    write_csv(metadata_dir / "file_manifest.csv", results, fields)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--archive", default="data/raw/PhenoLeaf-TS_DS.zip")
    parser.add_argument("--report-dir", default="reports/m0")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    report = audit_dataset(args.metadata_dir, args.project_root, args.archive, args.report_dir, args.workers)
    print(json.dumps({key: value for key, value in report.items() if key != "duplicates"}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
