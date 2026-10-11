#!/usr/bin/env python3
"""Build history-3 to future-observation temporal prediction windows."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.phenoleaf_utils import is_usable_quality_status, json_dumps, write_csv  # noqa: E402


WINDOW_FIELDS = [
    "window_id",
    "plant_uid",
    "genotype",
    "replicate",
    "split",
    "history_length",
    "input_image_paths",
    "input_frame_indices",
    "input_timestamps",
    "time_deltas",
    "timestamp_reliable",
    "target_frame_idx",
    "target_timestamp",
    "target_time_delta_seconds",
    "forecast_horizon",
    "target_leaf_count",
    "target_projected_area",
    "target_projected_area_ratio",
    "target_growth_stage",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _float_or_blank(value: str) -> float | str:
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def build_windows(
    frames_path: str | Path,
    split_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    *,
    history_length: int = 3,
    horizons: Iterable[int] = (2, 4, 8),
    allow_empty: bool = False,
) -> dict[str, Any]:
    if history_length < 1:
        raise ValueError("history_length must be positive")
    horizon_values = tuple(sorted({int(value) for value in horizons}))
    if not horizon_values or any(value < 1 for value in horizon_values):
        raise ValueError("horizons must contain positive observation offsets")
    rows = read_csv(frames_path)
    split_payload = json.loads(Path(split_path).read_text(encoding="utf-8")) if Path(split_path).exists() else {}
    assignments = split_payload.get("assignments", {})
    usable = [row for row in rows if is_usable_quality_status(row.get("quality_status", ""))]
    if not usable and not allow_empty:
        raise ValueError("No usable frames found; run metadata and phenotype extraction first")

    by_plant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_plant[row["plant_uid"]].append(row)
    windows: list[dict[str, Any]] = []
    skipped_target_labels = 0
    skipped_invalid_history = 0
    skipped_invalid_targets = 0
    for plant_uid in sorted(by_plant):
        sequence = sorted(by_plant[plant_uid], key=lambda row: int(row["frame_idx"]))
        frame_lookup = {int(row["frame_idx"]): row for row in sequence}
        if len(frame_lookup) != len(sequence):
            raise ValueError(f"Duplicate observation indices in {plant_uid}")
        for history_end in sorted(frame_lookup):
            input_frame_indices = list(range(history_end - history_length + 1, history_end + 1))
            if input_frame_indices[0] < 0:
                continue
            if not all(frame_index in frame_lookup for frame_index in input_frame_indices):
                continue
            history = [frame_lookup[frame_index] for frame_index in input_frame_indices]
            if not all(is_usable_quality_status(row.get("quality_status", "")) for row in history):
                skipped_invalid_history += 1
                continue
            for horizon in horizon_values:
                target_frame_idx = history_end + horizon
                if target_frame_idx not in frame_lookup:
                    continue
                target = frame_lookup[target_frame_idx]
                if not is_usable_quality_status(target.get("quality_status", "")):
                    skipped_invalid_targets += 1
                    continue
                try:
                    target_leaf_count = int(float(target["leaf_count"]))
                    target_area = float(target["projected_area_px"])
                    target_ratio = float(target["projected_area_ratio"])
                except (KeyError, TypeError, ValueError):
                    skipped_target_labels += 1
                    continue
                if not all(frame_idx < target_frame_idx for frame_idx in input_frame_indices):
                    raise AssertionError(f"Non-future target in plant {plant_uid}")
                timestamp_values = [parse_iso(row.get("timestamp", "")) for row in history]
                target_timestamp_value = parse_iso(target.get("timestamp", ""))
                reliable = all(timestamp_values) and target_timestamp_value is not None and all(
                    bool(row.get("timestamp_reliable", "").lower() == "true") for row in history + [target]
                )
                if reliable:
                    if any(current <= previous for previous, current in zip(timestamp_values, timestamp_values[1:])):
                        raise AssertionError(f"Non-increasing input timestamps in plant {plant_uid}")
                    if target_timestamp_value <= timestamp_values[-1]:
                        raise AssertionError(f"Target timestamp is not later than history in plant {plant_uid}")
                    time_deltas: list[float | None] = [0.0]
                    time_deltas.extend(
                        (current - previous).total_seconds()
                        for previous, current in zip(timestamp_values, timestamp_values[1:])
                    )
                    target_delta = (target_timestamp_value - timestamp_values[-1]).total_seconds()
                else:
                    time_deltas = [None for _ in history]
                    target_delta = None
                split = assignments.get(plant_uid, "unassigned")
                if split not in {"train", "val", "test"}:
                    raise ValueError(f"Usable window has no plant-level split: {plant_uid}")
                window_id = f"{plant_uid}__end{input_frame_indices[-1]}__h{horizon}"
                windows.append(
                    {
                        "window_id": window_id,
                        "plant_uid": plant_uid,
                        "genotype": history[0]["genotype"],
                        "replicate": history[0]["replicate"],
                        "split": split,
                        "history_length": history_length,
                        "input_image_paths": [row["rgb_path"] for row in history],
                        "input_frame_indices": input_frame_indices,
                        "input_timestamps": [row.get("timestamp", "") for row in history],
                        "time_deltas": time_deltas,
                        "timestamp_reliable": reliable,
                        "target_frame_idx": target_frame_idx,
                        "target_timestamp": target.get("timestamp", ""),
                        "target_time_delta_seconds": target_delta if target_delta is not None else "",
                        "forecast_horizon": horizon,
                        "target_leaf_count": target_leaf_count,
                        "target_projected_area": target_area,
                        "target_projected_area_ratio": target_ratio,
                        "target_growth_stage": target.get("growth_stage", ""),
                    }
                )

    windows.sort(key=lambda row: (row["split"], row["plant_uid"], row["target_frame_idx"], row["forecast_horizon"]))
    write_csv(output_path, windows, WINDOW_FIELDS)
    windows_per_plant = Counter(row["plant_uid"] for row in windows)
    windows_per_horizon = Counter(int(row["forecast_horizon"]) for row in windows)
    stats = {
        "status": "ok" if windows else "blocked_no_windows",
        "frames_path": str(frames_path),
        "split_path": str(split_path),
        "history_length": history_length,
        "horizons": list(horizon_values),
        "horizon_unit": "observation index offset, not days",
        "counts": {
            "usable_frames": len(usable),
            "plants_with_usable_frames": len(by_plant),
            "windows": len(windows),
            "plants_with_windows": len(windows_per_plant),
            "skipped_target_labels": skipped_target_labels,
            "skipped_invalid_history_endpoints": skipped_invalid_history,
            "skipped_invalid_targets": skipped_invalid_targets,
        },
        "windows_by_split": dict(sorted(Counter(row["split"] for row in windows).items())),
        "windows_by_horizon": {str(key): value for key, value in sorted(windows_per_horizon.items())},
        "overlap": {
            "allowed": True,
            "note": "Windows from the same plant overlap by design; split assignment remains plant-level.",
            "mean_windows_per_plant": (sum(windows_per_plant.values()) / len(windows_per_plant)) if windows_per_plant else 0.0,
            "max_windows_per_plant": max(windows_per_plant.values()) if windows_per_plant else 0,
        },
    }
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/metadata/frames.csv")
    parser.add_argument("--split", default="splits/phenoleaf_plant_split_seed42.json")
    parser.add_argument("--output", default="data/processed/temporal_windows.csv")
    parser.add_argument("--report", default="reports/m0/temporal_window_statistics.json")
    parser.add_argument("--history-length", type=int, default=3)
    parser.add_argument("--horizons", default="2,4,8")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    horizons = [int(value.strip()) for value in args.horizons.split(",") if value.strip()]
    stats = build_windows(
        args.frames,
        args.split,
        args.output,
        args.report,
        history_length=args.history_length,
        horizons=horizons,
        allow_empty=args.allow_empty,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
