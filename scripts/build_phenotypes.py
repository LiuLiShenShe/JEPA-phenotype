#!/usr/bin/env python3
"""Recompute mask-derived phenotypes and per-leaf instance tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.phenoleaf_utils import (  # noqa: E402
    compute_mask_phenotype,
    is_usable_quality_status,
    load_palette,
    resolve_project_path,
    serializable_path,
    write_csv,
)


PHENOTYPE_FIELDS = [
    "plant_uid",
    "genotype",
    "replicate",
    "frame_idx",
    "time_index",
    "timestamp",
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

LEAF_FIELDS = [
    "plant_uid",
    "genotype",
    "replicate",
    "frame_idx",
    "timestamp",
    "leaf_id",
    "colour_r",
    "colour_g",
    "colour_b",
    "area_px",
    "bbox_x",
    "bbox_y",
    "bbox_width",
    "bbox_height",
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


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def discover_palette_path(project_root: str | Path) -> Path:
    root = Path(project_root)
    candidates = (
        root / "data/raw/leaf_dataset_colour/leaves_unique_colours.xlsx",
        root / "data/raw/PhenoLeaf-TS_DS/leaves_unique_colours.xlsx",
        root / "data/raw/leaves_unique_colours.xlsx",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def add_issue(issues: list[dict[str, Any]], issue_type: str, severity: str, message: str, row: dict[str, str], details: Any = None) -> None:
    issues.append(
        {
            "issue_type": issue_type,
            "severity": severity,
            "plant_uid": row.get("plant_uid", ""),
            "genotype": row.get("genotype", ""),
            "replicate": row.get("replicate", ""),
            "raw_filename": row.get("raw_filename", ""),
            "path": row.get("mask_path", ""),
            "message": message,
            "details": details or {},
        }
    )


def deduplicate_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for issue in issues:
        details = issue.get("details", {})
        while isinstance(details, str):
            try:
                decoded = json.loads(details)
            except json.JSONDecodeError:
                break
            if decoded == details:
                break
            details = decoded
        issue["details"] = details
        details_key = json.dumps(details, ensure_ascii=False, sort_keys=True)
        key = (
            str(issue.get("issue_type", "")),
            str(issue.get("plant_uid", "")),
            str(issue.get("raw_filename", "")),
            str(issue.get("message", "")),
            details_key,
        )
        unique[key] = issue
    return list(unique.values())


def build_phenotypes(
    frames_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    palette_path: str | Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_csv(frames_path)
    issues_path = Path(frames_path).parent / "data_issues.csv"
    issues = read_csv(issues_path) if issues_path.exists() else []
    derived_types = {
        "palette_validation", "invalid_palette_colors", "zero_leaf_count", "zero_projected_area",
        "leaf_count_outside_dataset_card", "leaf_identity_reappears", "leaf_identity_order_gap",
        "projected_area_jump", "phenotype_read_error",
    }
    issues = [issue for issue in issues if issue["issue_type"] not in derived_types]
    data_root = Path(palette_path) if palette_path else discover_palette_path(project_root)
    palette, palette_source, palette_issues = load_palette(data_root)
    for message in palette_issues:
        issues.append(
            {
                "issue_type": "palette_validation",
                "severity": "warning",
                "plant_uid": "",
                "genotype": "",
                "replicate": "",
                "raw_filename": "",
                "path": str(data_root),
                "message": message,
                "details": {},
            }
        )

    phenotype_rows: list[dict[str, Any]] = []
    leaf_rows: list[dict[str, Any]] = []
    frame_lookup: dict[tuple[str, str], dict[str, str]] = {}
    valid_phenotypes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_color_frames = 0
    for row in rows:
        key = (row.get("plant_uid", ""), row.get("raw_filename", ""))
        frame_lookup[key] = row
        base = {field: row.get(field, "") for field in PHENOTYPE_FIELDS}
        if row.get("quality_status", "") in {
            "corrupt_rgb",
            "missing_mask",
            "corrupt_mask",
            "dimension_mismatch",
            "phenotype_error",
        }:
            phenotype_rows.append(base)
            continue
        mask_path = resolve_project_path(row.get("mask_path", ""), project_root)
        try:
            phenotype = compute_mask_phenotype(mask_path, palette)
        except Exception as exc:
            add_issue(issues, "phenotype_read_error", "error", f"Could not extract phenotype: {type(exc).__name__}", row)
            row["quality_status"] = "phenotype_error"
            row["quality_flags"] = json.dumps(["phenotype_read_error"])
            phenotype_rows.append({field: row.get(field, "") for field in PHENOTYPE_FIELDS})
            continue

        flags = []
        if phenotype["invalid_color_pixel_count"]:
            flags.append("invalid_palette_colors")
            invalid_color_frames += 1
            add_issue(
                issues,
                "invalid_palette_colors",
                "warning",
                "Mask contains non-black colors outside the validated palette",
                row,
                {
                    "pixel_count": phenotype["invalid_color_pixel_count"],
                    "colors": phenotype["invalid_colors"],
                    "regions": phenotype["invalid_color_regions"],
                },
            )
        if phenotype["leaf_count"] == 0:
            flags.append("zero_leaf_count")
            add_issue(issues, "zero_leaf_count", "warning", "No visible legal palette instances", row)
        if phenotype["projected_area_px"] == 0:
            flags.append("zero_projected_area")
            add_issue(issues, "zero_projected_area", "warning", "All-black mask", row)
        if phenotype["leaf_count"] < 4 or phenotype["leaf_count"] > 26:
            add_issue(
                issues, "leaf_count_outside_dataset_card", "warning",
                "Visible leaf count outside the descriptive 4-26 range; retained as observational warning",
                row, {"leaf_count": phenotype["leaf_count"]},
            )
        row.update(
            {
                "image_width": str(phenotype["image_width"]),
                "image_height": str(phenotype["image_height"]),
                "leaf_count": str(phenotype["leaf_count"]),
                "projected_area_px": str(phenotype["projected_area_px"]),
                "known_projected_area_px": str(phenotype["known_projected_area_px"]),
                "projected_area_ratio": str(phenotype["projected_area_ratio"]),
                "invalid_color_pixel_count": str(phenotype["invalid_color_pixel_count"]),
                "invalid_colors": json.dumps(phenotype["invalid_colors"], ensure_ascii=False),
                "growth_stage_id": "" if phenotype["growth_stage_id"] is None else str(phenotype["growth_stage_id"]),
                "growth_stage": phenotype["growth_stage"],
                "quality_status": (
                    "invalid_time_order"
                    if row.get("timestamp_reliable", "").lower() != "true"
                    else "valid_with_warnings" if flags else "valid"
                ),
                "quality_flags": json.dumps(flags, ensure_ascii=False),
            }
        )
        phenotype_row = {field: row.get(field, "") for field in PHENOTYPE_FIELDS}
        phenotype_rows.append(phenotype_row)
        valid_phenotypes[row["plant_uid"]].append(
            {
                "row": row,
                "phenotype": phenotype,
            }
        )
        for instance in phenotype["instances"]:
            leaf_rows.append(
                {
                    "plant_uid": row.get("plant_uid", ""),
                    "genotype": row.get("genotype", ""),
                    "replicate": row.get("replicate", ""),
                    "frame_idx": row.get("frame_idx", ""),
                    "timestamp": row.get("timestamp", ""),
                    **instance,
                }
            )

    _check_leaf_identity_continuity(valid_phenotypes, issues)
    issues = deduplicate_issues(issues)
    phenotype_rows.sort(key=lambda row: (row.get("plant_uid", ""), int(row["frame_idx"]) if row.get("frame_idx", "").isdigit() else 10**9))
    leaf_rows.sort(key=lambda row: (row["plant_uid"], int(row["frame_idx"]) if str(row["frame_idx"]).isdigit() else 10**9, int(row["leaf_id"])))
    write_csv(output_dir / "phenotypes.csv", phenotype_rows, PHENOTYPE_FIELDS)
    write_csv(output_dir / "leaf_instances.csv", leaf_rows, LEAF_FIELDS)
    write_csv(Path(frames_path).parent / "data_issues.csv", issues, ISSUE_FIELDS)

    # Keep frames.csv as the canonical one-row-per-observation table while
    # ensuring its measurements come from this dedicated phenotype pass.
    frame_rows = list(frame_lookup.values())
    frame_rows.sort(key=lambda row: (row.get("plant_uid", ""), int(row["frame_idx"]) if row.get("frame_idx", "").isdigit() else 10**9))
    frame_fields = list(rows[0].keys()) if rows else [
        "plant_uid", "genotype", "replicate", "frame_idx", "time_index", "timestamp", "timestamp_status",
        "timestamp_reliable", "time_order_method", "raw_filename", "rgb_path", "mask_path", "image_width",
        "image_height", "leaf_count", "projected_area_px", "known_projected_area_px", "projected_area_ratio",
        "invalid_color_pixel_count", "invalid_colors", "growth_stage_id", "growth_stage", "quality_status", "quality_flags",
    ]
    write_csv(frames_path, frame_rows, frame_fields)

    leaf_counts = [int(row["leaf_count"]) for row in phenotype_rows if str(row.get("leaf_count", "")).isdigit()]
    areas = [float(row["projected_area_px"]) for row in phenotype_rows if row.get("projected_area_px", "") not in {"", None}]
    ratios = [float(row["projected_area_ratio"]) for row in phenotype_rows if row.get("projected_area_ratio", "") not in {"", None}]
    usable_rows = [row for row in phenotype_rows if is_usable_quality_status(row.get("quality_status", ""))]
    stats = {
        "status": "ok" if phenotype_rows else "blocked_no_frames",
        "palette": {"source": palette_source, "size": len(palette), "colors": [list(color) for color in palette]},
        "counts": {
            "phenotype_rows": len(phenotype_rows),
            "leaf_instance_rows": len(leaf_rows),
            "valid_phenotype_rows": len(usable_rows),
            "warning_phenotype_rows": len(phenotype_rows) - len(usable_rows),
            "invalid_color_frames": invalid_color_frames,
            "issue_rows": len(issues),
        },
        "leaf_count": {
            "min": min(leaf_counts) if leaf_counts else None,
            "max": max(leaf_counts) if leaf_counts else None,
            "mean": mean(leaf_counts) if leaf_counts else None,
        },
        "projected_area_px": {
            "min": min(areas) if areas else None,
            "max": max(areas) if areas else None,
            "mean": mean(areas) if areas else None,
        },
        "projected_area_ratio": {
            "min": min(ratios) if ratios else None,
            "max": max(ratios) if ratios else None,
            "mean": mean(ratios) if ratios else None,
        },
        "definitions": {
            "leaf_count": "number of unique exact palette colours present in the mask",
            "projected_area_px": "all non-black mask pixels, including invalid-colour pixels for transparent accounting",
            "projected_area_ratio": "projected_area_px / (image_height * image_width)",
            "not_physical_leaf_area": True,
        },
    }
    report_dir = Path(project_root) / "reports/m0"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "phenotype_statistics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    dataset_statistics_path = Path(frames_path).parent / "dataset_statistics.json"
    if dataset_statistics_path.exists():
        dataset_statistics = json.loads(dataset_statistics_path.read_text(encoding="utf-8"))
    else:
        dataset_statistics = {}
    dataset_statistics["status"] = "ok"
    dataset_statistics["real_data_available"] = bool(phenotype_rows)
    dataset_statistics.setdefault("counts", {})
    dataset_statistics["counts"].update(
        {
            "frame_rows": len(phenotype_rows),
            "usable_frames": len(usable_rows),
            "invalid_frames": len(phenotype_rows) - len(usable_rows),
            "leaf_instance_rows": len(leaf_rows),
        }
    )
    dataset_statistics["issues"] = {
        "total": len(issues),
        "by_type": dict(sorted(Counter(issue["issue_type"] for issue in issues).items())),
    }
    dataset_statistics["quality_policy"] = {
        "usable_statuses": ["valid"],
        "isolated_statuses": ["valid_with_warnings"],
        "isolated_reason": "non-palette mask colors are not used for split/window construction",
    }
    dataset_statistics_path.write_text(
        json.dumps(dataset_statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plants_path = Path(frames_path).parent / "plants.csv"
    if plants_path.exists():
        plants = read_csv(plants_path)
        usable_counts = Counter(row["plant_uid"] for row in usable_rows)
        for plant in plants:
            plant["n_usable_frames"] = usable_counts[plant["plant_uid"]]
            plant["n_invalid_frames"] = int(plant["n_frames"]) - plant["n_usable_frames"]
            if plant["n_invalid_frames"]:
                plant["quality_status"] = "warning" if plant["n_usable_frames"] else "invalid"
        if plants:
            write_csv(plants_path, plants, list(plants[0]))
    return stats


def _check_leaf_identity_continuity(
    plant_phenotypes: dict[str, list[dict[str, Any]]],
    issues: list[dict[str, Any]],
) -> None:
    for plant_uid, entries in plant_phenotypes.items():
        entries.sort(key=lambda entry: int(entry["row"].get("frame_idx", 0)))
        for previous, current in zip(entries, entries[1:]):
            previous_area = int(previous["phenotype"]["projected_area_px"])
            current_area = int(current["phenotype"]["projected_area_px"])
            if previous_area and current_area:
                factor = max(previous_area, current_area) / min(previous_area, current_area)
                if factor > 3.0:
                    row = current["row"]
                    issues.append(
                        {
                            "issue_type": "projected_area_jump",
                            "severity": "warning",
                            "plant_uid": plant_uid,
                            "genotype": row.get("genotype", ""),
                            "replicate": row.get("replicate", ""),
                            "raw_filename": row.get("raw_filename", ""),
                            "path": row.get("mask_path", ""),
                            "message": "Adjacent valid frames change projected area by more than 3x",
                            "details": {
                                "previous_area_px": previous_area,
                                "current_area_px": current_area,
                                "factor": factor,
                            },
                        }
                    )
        presence = [
            {int(instance["leaf_id"]) for instance in entry["phenotype"]["instances"]}
            for entry in entries
        ]
        first_seen = {}
        for position, ids in enumerate(presence):
            for leaf_id in sorted(ids):
                first_seen.setdefault(leaf_id, position)
        for leaf_id in sorted(first_seen):
            seen = False
            absent_since = None
            previous_present_frame = None
            for position, ids in enumerate(presence):
                if leaf_id in ids:
                    if absent_since is not None:
                        row = entries[position]["row"]
                        add_issue(
                            issues, "leaf_identity_reappears", "warning",
                            "A colour-coded leaf disappears and reappears; may reflect occlusion, not ID corruption",
                            row,
                            {
                                "leaf_id": leaf_id,
                                "last_visible_frame_idx": previous_present_frame,
                                "first_absent_frame_idx": absent_since,
                                "reappeared_frame_idx": int(row["frame_idx"]),
                            },
                        )
                    seen = True
                    absent_since = None
                    previous_present_frame = int(entries[position]["row"]["frame_idx"])
                elif seen and absent_since is None:
                    absent_since = int(entries[position]["row"]["frame_idx"])
        expected = list(range(1, max(first_seen, default=0) + 1))
        missing = [leaf_id for leaf_id in expected if leaf_id not in first_seen]
        if missing:
            row = entries[0]["row"]
            issues.append(
                {
                    "issue_type": "leaf_identity_order_gap",
                    "severity": "warning",
                    "plant_uid": plant_uid,
                    "genotype": row.get("genotype", ""),
                    "replicate": row.get("replicate", ""),
                    "raw_filename": row.get("raw_filename", ""),
                    "path": row.get("mask_path", ""),
                    "message": "First appearances of leaf colours contain an ID gap",
                    "details": {"missing_leaf_ids": missing},
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/metadata/frames.csv")
    parser.add_argument("--output-dir", default="data/metadata")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--palette-path", default=None)
    args = parser.parse_args()
    stats = build_phenotypes(
        args.frames,
        args.output_dir,
        project_root=args.project_root,
        palette_path=args.palette_path,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
