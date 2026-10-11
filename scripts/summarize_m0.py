#!/usr/bin/env python3
"""Report geometry, timing, annotation quarantine and split label distributions."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize() -> dict:
    frames = read_rows(PROJECT_ROOT / "data/metadata/frames.csv")
    windows = read_rows(PROJECT_ROOT / "data/processed/temporal_windows.csv")
    assignments = json.loads((PROJECT_ROOT / "splits/phenoleaf_plant_split_seed42.json").read_text())["assignments"]
    by_plant = defaultdict(list)
    for row in frames:
        by_plant[row["plant_uid"]].append(row)
    identity_rows, source_to_plants = [], defaultdict(set)
    suffix_mismatch_frames = 0
    genotype_suffix_conflicts = 0
    for plant_uid, rows in sorted(by_plant.items()):
        source_ids = sorted({row["raw_filename"].split("_", 1)[0] for row in rows})
        suffixes = sorted({Path(row["raw_filename"]).stem.rsplit("_", 1)[-1] for row in rows})
        suffix_mismatch_frames += sum(not Path(row["raw_filename"]).stem.endswith(f"{row['genotype']}-{row['replicate'].removeprefix('rep_')}") for row in rows)
        genotype_suffix_conflicts += sum(Path(row["raw_filename"]).stem.rsplit("_", 1)[-1].rsplit("-", 1)[0] != row["genotype"] for row in rows)
        for source_id in source_ids:
            source_to_plants[source_id].add(plant_uid)
        identity_rows.append({"plant_uid": plant_uid, "source_acquisition_ids": json.dumps(source_ids), "filename_plant_suffixes": json.dumps(suffixes), "directory_replicate": rows[0]["replicate"], "directory_matches_filename_suffix": all(Path(row["raw_filename"]).stem.endswith(f"{row['genotype']}-{row['replicate'].removeprefix('rep_')}") for row in rows), "frames": len(rows)})
    with (PROJECT_ROOT / "data/metadata/plant_identity_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identity_rows[0]))
        writer.writeheader()
        writer.writerows(identity_rows)
    intervals, geometry_changes = [], []
    for plant_uid, rows in by_plant.items():
        rows.sort(key=lambda row: int(row["frame_idx"]))
        for previous, current in zip(rows, rows[1:]):
            intervals.append((datetime.fromisoformat(current["timestamp"]) - datetime.fromisoformat(previous["timestamp"])).total_seconds())
            if (previous["image_width"], previous["image_height"]) != (current["image_width"], current["image_height"]):
                geometry_changes.append({"plant_uid": plant_uid, "previous_frame_idx": int(previous["frame_idx"]), "frame_idx": int(current["frame_idx"]), "previous_size_wh": [int(previous["image_width"]), int(previous["image_height"])], "size_wh": [int(current["image_width"]), int(current["image_height"])]})
    quarantined = [row for row in frames if row["quality_status"] != "valid"]
    colors = Counter()
    for row in quarantined:
        colors.update(tuple(color) for color in json.loads(row["invalid_colors"]))
    windows_per_split_horizon = Counter((row["split"], int(row["forecast_horizon"])) for row in windows)
    split_labels = {}
    for split in ("train", "val", "test"):
        rows = [row for row in frames if row["quality_status"] == "valid" and assignments[row["plant_uid"]] == split]
        split_labels[split] = {
            "leaf_count_distribution": dict(sorted(Counter(int(row["leaf_count"]) for row in rows).items())),
            "growth_stage_distribution": dict(Counter(row["growth_stage"] for row in rows)),
            "windows_by_horizon": {str(horizon): windows_per_split_horizon[(split, horizon)] for horizon in (2, 4, 8)},
        }
    horizon_seconds = {}
    for horizon in (2, 4, 8):
        values = sorted(float(row["target_time_delta_seconds"]) for row in windows if int(row["forecast_horizon"]) == horizon)
        horizon_seconds[str(horizon)] = {"min_seconds": min(values), "median_seconds": values[len(values) // 2], "max_seconds": max(values)}
    potential_windows = Counter()
    for rows in by_plant.values():
        indices = {int(row["frame_idx"]) for row in rows}
        for end_idx in indices:
            if all(frame_idx in indices for frame_idx in (end_idx - 2, end_idx - 1, end_idx)):
                for horizon in (2, 4, 8):
                    if end_idx + horizon in indices:
                        potential_windows[horizon] += 1
    actual_windows = Counter(int(row["forecast_horizon"]) for row in windows)
    zero_mask_frames = sum(float(row["projected_area_px"]) == 0 for row in frames)
    result = {
        "data_scope": "real_dataset",
        "identity": {
            "source_acquisition_ids": len(source_to_plants),
            "source_ids_shared_across_plant_uids": {source_id: sorted(plant_uids) for source_id, plant_uids in source_to_plants.items() if len(plant_uids) > 1},
            "plants_with_multiple_source_ids": [row["plant_uid"] for row in identity_rows if len(json.loads(row["source_acquisition_ids"])) != 1],
            "filename_genotype_conflict_frames": genotype_suffix_conflicts,
            "filename_replicate_differs_from_directory_frames": suffix_mismatch_frames,
            "plants_with_filename_replicate_alias": sum(not row["directory_matches_filename_suffix"] for row in identity_rows),
            "policy": "Use curated genotype/replicate directory identity. Filename suffix replicate numbers can be reused across acquisitions; acquisition-prefix IDs are one-to-one with directory plant_uids.",
        },
        "distinct_image_sizes": len({(row['image_width'], row['image_height']) for row in frames}),
        "plant_lengths": {"min": min(map(len, by_plant.values())), "max": max(map(len, by_plant.values())), "distribution": dict(sorted(Counter(map(len, by_plant.values())).items()))},
        "plants_with_size_changes": sorted({event["plant_uid"] for event in geometry_changes}),
        "size_change_events": geometry_changes,
        "observation_time_range": {"first": min(row['timestamp'] for row in frames), "last": max(row['timestamp'] for row in frames), "timezone": "not provided by source"},
        "adjacent_intervals": {"min_seconds": min(intervals), "median_seconds": sorted(intervals)[len(intervals) // 2], "max_seconds": max(intervals)},
        "horizon_elapsed_seconds": horizon_seconds,
        "quarantine": {"frames": len(quarantined), "plants": sorted({row['plant_uid'] for row in quarantined}), "frame_indices": {plant_uid: [int(row['frame_idx']) for row in quarantined if row['plant_uid'] == plant_uid] for plant_uid in sorted({row['plant_uid'] for row in quarantined})}, "invalid_color_pixels": sum(int(row['invalid_color_pixel_count']) for row in quarantined), "distinct_illegal_colors": len(colors), "illegal_rgb_colors": [list(color) for color in sorted(colors)]},
        "split_labels": split_labels,
        "quarantine_window_impact": {
            "potential_windows_without_quality_exclusion": sum(potential_windows.values()),
            "actual_windows": len(windows),
            "excluded_windows": sum(potential_windows.values()) - len(windows),
            "excluded_by_horizon": {str(horizon): potential_windows[horizon] - actual_windows[horizon] for horizon in (2, 4, 8)},
        },
        "zero_area_frames": zero_mask_frames,
        "physical_area_warning": "Original dimensions vary across and within plants; projected pixels/ratios are image-space measurements, not calibrated physical area or LAI. Resize only historical RGB tensors, never GT areas.",
    }
    path = PROJECT_ROOT / "reports/m0/geometry_time_and_label_statistics.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    result = summarize()
    print(json.dumps({key: result[key] for key in ("distinct_image_sizes", "plants_with_size_changes", "observation_time_range", "adjacent_intervals", "quarantine")}, ensure_ascii=False, indent=2))
