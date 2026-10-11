#!/usr/bin/env python3
"""Scan PhenoLeaf-TS RGB/mask pairs and build deterministic metadata tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.phenoleaf_utils import (  # noqa: E402
    compute_mask_phenotype,
    is_usable_quality_status,
    json_dumps,
    load_palette,
    parse_timestamp,
    inspect_image,
    serializable_path,
    write_csv,
)


FRAME_FIELDS = [
    "plant_uid",
    "genotype",
    "replicate",
    "frame_idx",
    "time_index",
    "timestamp",
    "timestamp_status",
    "timestamp_reliable",
    "time_order_method",
    "raw_filename",
    "rgb_path",
    "mask_path",
    "image_width",
    "image_height",
    "leaf_count",
    "projected_area_px",
    "known_projected_area_px",
    "projected_area_ratio",
    "invalid_color_pixel_count",
    "invalid_colors",
    "growth_stage_id",
    "growth_stage",
    "quality_status",
    "quality_flags",
]

PLANT_FIELDS = [
    "plant_uid",
    "genotype",
    "replicate",
    "n_rgb_files",
    "n_frames",
    "n_usable_frames",
    "n_invalid_frames",
    "first_timestamp",
    "last_timestamp",
    "timestamp_reliable",
    "time_order_method",
    "quality_status",
]

ISSUE_FIELDS = [
    "issue_type",
    "severity",
    "plant_uid",
    "genotype",
    "replicate",
    "raw_filename",
    "path",
    "message",
    "details",
]

AREA_JUMP_FACTOR = 3.0


def locate_dataset_root(data_root: str | Path) -> Path | None:
    """Resolve either the documented dataset root or its parent directory."""

    candidate = Path(data_root)
    if (candidate / "01_raw_dataset").is_dir() and (candidate / "02_leaf_labels").is_dir():
        return candidate
    for name in ("leaf_dataset_colour", "PhenoLeaf-TS_DS"):
        nested = candidate / name
        if (nested / "01_raw_dataset").is_dir() and (nested / "02_leaf_labels").is_dir():
            return nested
    return None


def add_issue(
    issues: list[dict[str, Any]],
    issue_type: str,
    severity: str,
    message: str,
    *,
    plant_uid: str = "",
    genotype: str = "",
    replicate: str = "",
    raw_filename: str = "",
    path: str = "",
    details: Any = None,
) -> None:
    issues.append(
        {
            "issue_type": issue_type,
            "severity": severity,
            "plant_uid": plant_uid,
            "genotype": genotype,
            "replicate": replicate,
            "raw_filename": raw_filename,
            "path": path,
            "message": message,
            "details": details if details is not None else {},
        }
    )


def image_files(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}],
        key=lambda path: path.name,
    )


def _empty_frame(rgb_path: Path, mask_path: Path, genotype: str, replicate: str, project_root: Path) -> dict[str, Any]:
    return {
        "plant_uid": f"{genotype}/{replicate}",
        "genotype": genotype,
        "replicate": replicate,
        "frame_idx": "",
        "time_index": "",
        "timestamp": "",
        "timestamp_status": "",
        "timestamp_reliable": False,
        "time_order_method": "",
        "raw_filename": rgb_path.name,
        "rgb_path": serializable_path(rgb_path, project_root),
        "mask_path": serializable_path(mask_path, project_root),
        "image_width": "",
        "image_height": "",
        "leaf_count": "",
        "projected_area_px": "",
        "known_projected_area_px": "",
        "projected_area_ratio": "",
        "invalid_color_pixel_count": "",
        "invalid_colors": [],
        "growth_stage_id": "",
        "growth_stage": "unknown",
        "quality_status": "",
        "quality_flags": [],
    }


def _scan_pair(
    rgb_path: Path,
    mask_path: Path,
    genotype: str,
    replicate: str,
    palette: list[tuple[int, int, int]],
    project_root: Path,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    plant_uid = f"{genotype}/{replicate}"
    row = _empty_frame(rgb_path, mask_path, genotype, replicate, project_root)
    parsed_dt, timestamp, timestamp_status = parse_timestamp(rgb_path.name)
    row["timestamp"] = timestamp
    row["timestamp_status"] = timestamp_status
    row["_timestamp_dt"] = parsed_dt
    row["_filename_sort_key"] = rgb_path.name
    flags: list[str] = []

    try:
        rgb_width, rgb_height, _ = inspect_image(rgb_path)
    except Exception as exc:
        row["quality_status"] = "corrupt_rgb"
        flags.append("corrupt_rgb")
        add_issue(
            issues,
            "corrupt_rgb",
            "error",
            f"RGB image cannot be decoded: {type(exc).__name__}",
            plant_uid=plant_uid,
            genotype=genotype,
            replicate=replicate,
            raw_filename=rgb_path.name,
            path=str(rgb_path),
        )
        row["quality_flags"] = flags
        return row

    row["image_width"] = rgb_width
    row["image_height"] = rgb_height
    if not mask_path.exists():
        row["quality_status"] = "missing_mask"
        flags.append("missing_mask")
        add_issue(
            issues,
            "missing_mask",
            "error",
            "RGB image has no corresponding mask with the same filename",
            plant_uid=plant_uid,
            genotype=genotype,
            replicate=replicate,
            raw_filename=rgb_path.name,
            path=str(rgb_path),
        )
        row["quality_flags"] = flags
        return row

    try:
        mask_width, mask_height, _ = inspect_image(mask_path)
    except Exception as exc:
        row["quality_status"] = "corrupt_mask"
        flags.append("corrupt_mask")
        add_issue(
            issues,
            "corrupt_mask",
            "error",
            f"Mask cannot be decoded: {type(exc).__name__}",
            plant_uid=plant_uid,
            genotype=genotype,
            replicate=replicate,
            raw_filename=rgb_path.name,
            path=str(mask_path),
        )
        row["quality_flags"] = flags
        return row

    if (rgb_width, rgb_height) != (mask_width, mask_height):
        row["quality_status"] = "dimension_mismatch"
        flags.append("dimension_mismatch")
        add_issue(
            issues,
            "dimension_mismatch",
            "error",
            "RGB and mask dimensions differ",
            plant_uid=plant_uid,
            genotype=genotype,
            replicate=replicate,
            raw_filename=rgb_path.name,
            path=str(mask_path),
            details={"rgb": [rgb_width, rgb_height], "mask": [mask_width, mask_height]},
        )
        row["quality_flags"] = flags
        return row

    # Exact palette phenotype extraction is intentionally performed once in
    # build_phenotypes.py.  Keeping this scan focused on file integrity makes
    # the full 17k-frame pass practical while preserving the same final rows.
    row["quality_status"] = "valid"
    row["quality_flags"] = flags
    return row


def _order_plant_rows(
    rows: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    plant_uid: str,
) -> tuple[list[dict[str, Any]], bool, str]:
    parsed = [row["_timestamp_dt"] for row in rows]
    all_parsed = all(value is not None for value in parsed)
    unique_timestamps = len({value for value in parsed if value is not None}) == len(parsed)
    if all_parsed and unique_timestamps:
        ordered = sorted(rows, key=lambda row: (row["_timestamp_dt"], row["raw_filename"]))
        method = "timestamp"
        reliable = True
        filename_order = [row["raw_filename"] for row in sorted(rows, key=lambda row: row["raw_filename"])]
        timestamp_order = [row["raw_filename"] for row in ordered]
        if filename_order != timestamp_order:
            add_issue(
                issues,
                "filename_timestamp_order_difference",
                "warning",
                "Timestamp order differs from lexicographic filename order; timestamp order is used",
                plant_uid=plant_uid,
                details={"filename_order_differs": True},
            )
    else:
        ordered = sorted(rows, key=lambda row: row["raw_filename"])
        method = "filename"
        reliable = False
        if not all_parsed:
            add_issue(
                issues,
                "unparseable_timestamp",
                "warning",
                "At least one timestamp is missing or invalid; filename order is retained",
                plant_uid=plant_uid,
            )
        if all_parsed and not unique_timestamps:
            add_issue(
                issues,
                "duplicate_timestamp",
                "error",
                "Duplicate timestamps exist within one plant sequence",
                plant_uid=plant_uid,
                details={"timestamps": [row["timestamp"] for row in rows]},
            )
    for frame_idx, row in enumerate(ordered):
        row["frame_idx"] = frame_idx
        row["time_index"] = frame_idx
        row["timestamp_reliable"] = reliable
        row["time_order_method"] = method
        if not reliable and row["quality_status"] == "valid":
            row["quality_status"] = "invalid_time_order"
            row["quality_flags"].append("unverified_time_order")
        row.pop("_timestamp_dt", None)
        row.pop("_filename_sort_key", None)
    return ordered, reliable, method


def _detect_sequence_anomalies(rows: list[dict[str, Any]], issues: list[dict[str, Any]], plant_uid: str) -> None:
    valid_rows = [row for row in rows if is_usable_quality_status(str(row["quality_status"]))]
    present_history: dict[int, list[int]] = defaultdict(list)
    for row in valid_rows:
        phenotype_path = row.get("mask_path", "")
        if not phenotype_path or not row.get("leaf_count", ""):
            continue
        try:
            colors = json.loads(row.get("invalid_colors", "[]"))
        except (TypeError, json.JSONDecodeError):
            colors = []
        del colors
        # The instance IDs are represented by leaf count in frames.csv.  The
        # full colour-level continuity check is performed in build_phenotypes.
        if row.get("projected_area_px", "") not in {"", None}:
            present_history[-1].append(int(row["projected_area_px"]))

    areas = present_history.get(-1, [])
    for previous, current in zip(areas, areas[1:]):
        if previous == 0 or current == 0:
            continue
        ratio = max(previous, current) / min(previous, current)
        if ratio > AREA_JUMP_FACTOR:
            add_issue(
                issues,
                "projected_area_jump",
                "warning",
                f"Adjacent usable frames change projected area by a factor of {ratio:.2f}",
                plant_uid=plant_uid,
                details={"previous_area_px": previous, "current_area_px": current, "factor": ratio},
            )


def scan_dataset(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    allow_empty: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_dir
    issues: list[dict[str, Any]] = []
    dataset_root = locate_dataset_root(data_root)
    if dataset_root is None:
        message = f"Documented dataset structure not found below {Path(data_root)}"
        add_issue(issues, "dataset_root_missing", "error", message, path=str(data_root))
        if not allow_empty:
            raise FileNotFoundError(message)
        frames: list[dict[str, Any]] = []
        plants: list[dict[str, Any]] = []
        stats = {
            "status": "blocked_no_dataset",
            "real_data_available": False,
            "dataset_root": str(Path(data_root)),
            "counts": {"rgb_files": 0, "frame_rows": 0, "usable_frames": 0, "plants": 0, "genotypes": 0},
            "issues": {"total": len(issues), "by_type": {"dataset_root_missing": 1}},
            "palette": {"source": "not_loaded", "size": 0},
        }
        write_csv(metadata_dir / "frames.csv", frames, FRAME_FIELDS)
        write_csv(metadata_dir / "plants.csv", plants, PLANT_FIELDS)
        write_csv(metadata_dir / "data_issues.csv", issues, ISSUE_FIELDS)
        (metadata_dir / "dataset_statistics.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return stats

    raw_dir = dataset_root / "01_raw_dataset"
    mask_dir = dataset_root / "02_leaf_labels"
    palette_path = dataset_root / "leaves_unique_colours.xlsx"
    palette, palette_source, palette_issues = load_palette(palette_path)
    for message in palette_issues:
        add_issue(issues, "palette_validation", "warning", message, path=str(palette_path))

    all_rows: list[dict[str, Any]] = []
    plants_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rgb_paths_seen: set[str] = set()
    rgb_file_count = 0
    for genotype_dir in sorted((path for path in raw_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
        genotype = genotype_dir.name
        for replicate_dir in sorted((path for path in genotype_dir.iterdir() if path.is_dir()), key=lambda path: path.name):
            replicate = replicate_dir.name
            plant_uid = f"{genotype}/{replicate}"
            rgb_paths = image_files(replicate_dir)
            rgb_file_count += len(rgb_paths)
            plant_rows: list[dict[str, Any]] = []
            for rgb_path in rgb_paths:
                path_key = str(rgb_path.resolve())
                if path_key in rgb_paths_seen:
                    add_issue(issues, "duplicate_rgb_path", "error", "RGB path appeared more than once", plant_uid=plant_uid, path=str(rgb_path))
                rgb_paths_seen.add(path_key)
                mask_path = mask_dir / genotype / replicate / rgb_path.name
                row = _scan_pair(rgb_path, mask_path, genotype, replicate, palette, Path(project_root), issues)
                plant_rows.append(row)
            ordered, reliable, method = _order_plant_rows(plant_rows, issues, plant_uid)
            _detect_sequence_anomalies(ordered, issues, plant_uid)
            for row in ordered:
                all_rows.append(row)
            plants_by_uid[plant_uid] = ordered
            if not ordered:
                add_issue(issues, "empty_plant_sequence", "warning", "Plant directory contains no supported RGB images", plant_uid=plant_uid)

    expected_mask_paths: set[str] = set()
    for row in all_rows:
        if row["mask_path"]:
            expected_mask_paths.add(str((Path(project_root) / row["mask_path"]).resolve()))
    if mask_dir.exists():
        for mask_path in sorted(mask_dir.rglob("*")):
            if not mask_path.is_file() or mask_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            if str(mask_path.resolve()) not in expected_mask_paths:
                add_issue(issues, "orphan_mask", "warning", "Mask has no corresponding RGB file", path=str(mask_path))

    plants: list[dict[str, Any]] = []
    for plant_uid in sorted(plants_by_uid):
        rows = plants_by_uid[plant_uid]
        genotype, replicate = plant_uid.split("/", 1)
        usable = [row for row in rows if is_usable_quality_status(str(row["quality_status"]))]
        valid_timestamps = [row["timestamp"] for row in rows if row["timestamp"]]
        statuses = {str(row["quality_status"]) for row in rows}
        plant_quality = "valid" if statuses and statuses <= {"valid"} else "warning" if usable else "invalid"
        plants.append(
            {
                "plant_uid": plant_uid,
                "genotype": genotype,
                "replicate": replicate,
                "n_rgb_files": len(rows),
                "n_frames": len(rows),
                "n_usable_frames": len(usable),
                "n_invalid_frames": len(rows) - len(usable),
                "first_timestamp": valid_timestamps[0] if valid_timestamps else "",
                "last_timestamp": valid_timestamps[-1] if valid_timestamps else "",
                "timestamp_reliable": bool(rows and rows[0]["timestamp_reliable"]),
                "time_order_method": rows[0]["time_order_method"] if rows else "",
                "quality_status": plant_quality,
            }
        )

    all_rows.sort(key=lambda row: (row["plant_uid"], int(row["frame_idx"]) if row["frame_idx"] != "" else 10**9, row["raw_filename"]))
    write_csv(metadata_dir / "frames.csv", all_rows, FRAME_FIELDS)
    write_csv(metadata_dir / "plants.csv", plants, PLANT_FIELDS)
    write_csv(metadata_dir / "data_issues.csv", issues, ISSUE_FIELDS)

    issue_counts: dict[str, int] = defaultdict(int)
    for issue in issues:
        issue_counts[issue["issue_type"]] += 1
    usable_frames = sum(is_usable_quality_status(str(row["quality_status"])) for row in all_rows)
    stats = {
        "status": "ok" if all_rows else "empty_dataset",
        "real_data_available": bool(all_rows),
        "dataset_root": serializable_path(dataset_root, project_root),
        "raw_dir": serializable_path(raw_dir, project_root),
        "mask_dir": serializable_path(mask_dir, project_root),
        "palette_path": serializable_path(palette_path, project_root) if palette_path.exists() else "",
        "palette": {"source": palette_source, "size": len(palette), "colors": [list(color) for color in palette]},
        "counts": {
            "rgb_files": rgb_file_count,
            "frame_rows": len(all_rows),
            "usable_frames": int(usable_frames),
            "invalid_frames": len(all_rows) - int(usable_frames),
            "plants": len(plants),
            "genotypes": len({plant["genotype"] for plant in plants}),
        },
        "timestamp": {
            "reliable_plants": sum(bool(plant["timestamp_reliable"]) for plant in plants),
            "unreliable_plants": sum(not bool(plant["timestamp_reliable"]) for plant in plants),
        },
        "issues": {"total": len(issues), "by_type": dict(sorted(issue_counts.items()))},
        "official_reference": {
            "repository": "https://github.com/pisyntor/PhenoLeaf-TS",
            "commit": "0d9a24ff146d12cc8fca6bdfa2538d196754f456",
            "dataset": "https://huggingface.co/datasets/rick77a/PhenoLeaf_TS",
        },
    }
    (metadata_dir / "dataset_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/leaf_dataset_colour")
    parser.add_argument("--output-dir", default="data/metadata")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--allow-empty", action="store_true", help="Write blocked empty schemas if real data is absent")
    args = parser.parse_args()
    stats = scan_dataset(args.data_root, args.output_dir, project_root=args.project_root, allow_empty=args.allow_empty)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
