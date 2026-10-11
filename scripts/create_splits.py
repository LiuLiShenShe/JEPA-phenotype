#!/usr/bin/env python3
"""Create a deterministic genotype-stratified plant-level split."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.phenoleaf_utils import is_usable_quality_status, write_csv  # noqa: E402


SPLITS = ("train", "val", "test")
SPLIT_FIELDS = [
    "split",
    "genotype",
    "plant_count",
    "image_count",
    "leaf_count_min",
    "leaf_count_max",
    "leaf_count_mean",
    "projected_area_px_min",
    "projected_area_px_max",
    "projected_area_px_mean",
]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _allocate_counts(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0
    raw = [count * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remainder = count - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remainder]:
        counts[index] += 1
    if count >= 3:
        for index in (1, 2):
            if counts[index] == 0:
                donor = 0 if counts[0] > 1 else next(candidate for candidate in (1, 2) if counts[candidate] > 1)
                counts[donor] -= 1
                counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def assign_plants(rows: list[dict[str, str]], seed: int = 42) -> dict[str, str]:
    """Assign each usable plant UID once, stratified independently by genotype."""

    plant_to_genotype: dict[str, str] = {}
    for row in rows:
        if not is_usable_quality_status(row.get("quality_status", "")):
            continue
        plant_uid = row["plant_uid"]
        genotype = row["genotype"]
        previous = plant_to_genotype.setdefault(plant_uid, genotype)
        if previous != genotype:
            raise ValueError(f"Plant UID collision: {plant_uid} maps to {previous} and {genotype}")

    by_genotype: dict[str, list[str]] = defaultdict(list)
    for plant_uid, genotype in plant_to_genotype.items():
        by_genotype[genotype].append(plant_uid)
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    for genotype in sorted(by_genotype):
        plant_uids = sorted(by_genotype[genotype])
        rng.shuffle(plant_uids)
        counts = _allocate_counts(len(plant_uids), (0.70, 0.15, 0.15))
        cursor = 0
        for split, count in zip(SPLITS, counts):
            for plant_uid in plant_uids[cursor : cursor + count]:
                assignments[plant_uid] = split
            cursor += count
    return assignments


def _numeric_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values = []
    for row in rows:
        try:
            values.append(float(row[field]))
        except (KeyError, TypeError, ValueError):
            continue
    return values


def write_split_plot(rows: list[dict[str, str]], assignments: dict[str, str], output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    genotypes = sorted({row["genotype"] for row in rows if row.get("plant_uid") in assignments})
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    split_colors = {"train": "#4C78A8", "val": "#F58518", "test": "#E45756"}
    totals = {split: sum(1 for plant_uid in assignments if assignments[plant_uid] == split) for split in SPLITS}
    image_totals = {
        split: sum(1 for row in rows if assignments.get(row.get("plant_uid")) == split) for split in SPLITS
    }
    positions = np.arange(len(SPLITS))
    axes[0].bar(positions - 0.18, [totals[split] for split in SPLITS], width=0.36, label="plants")
    axes[0].bar(positions + 0.18, [image_totals[split] for split in SPLITS], width=0.36, label="images")
    axes[0].set_xticks(positions, [split.title() for split in SPLITS])
    axes[0].set_title("Plant-level split totals")
    axes[0].legend()
    bottom = np.zeros(len(SPLITS))
    for genotype in genotypes:
        values = [
            sum(1 for plant_uid, split in assignments.items() if split == split_name and plant_uid.startswith(f"{genotype}/"))
            for split_name in SPLITS
        ]
        axes[1].bar(SPLITS, values, bottom=bottom, label=genotype)
        bottom += np.asarray(values)
    axes[1].set_title("Genotype distribution by plants")
    axes[1].set_ylabel("plants")
    if len(genotypes) <= 24:
        axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(axis="y", alpha=0.25)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def create_splits(
    frames_path: str | Path,
    output_path: str | Path,
    statistics_path: str | Path,
    plot_path: str | Path,
    *,
    seed: int = 42,
    allow_empty: bool = False,
) -> dict[str, Any]:
    rows = read_csv(frames_path)
    usable_rows = [row for row in rows if is_usable_quality_status(row.get("quality_status", ""))]
    if not usable_rows and not allow_empty:
        raise ValueError("No usable frames found; run metadata and phenotype extraction first")
    assignments = assign_plants(usable_rows, seed=seed)
    split_rows: list[dict[str, Any]] = []
    for split in SPLITS:
        split_plant_uids = sorted(plant_uid for plant_uid, assigned in assignments.items() if assigned == split)
        for genotype in sorted({row["genotype"] for row in usable_rows}):
            selected = [
                row for row in usable_rows
                if assignments.get(row["plant_uid"]) == split and row["genotype"] == genotype
            ]
            leaf_counts = _numeric_values(selected, "leaf_count")
            areas = _numeric_values(selected, "projected_area_px")
            split_rows.append(
                {
                    "split": split,
                    "genotype": genotype,
                    "plant_count": len({row["plant_uid"] for row in selected}),
                    "image_count": len(selected),
                    "leaf_count_min": min(leaf_counts) if leaf_counts else "",
                    "leaf_count_max": max(leaf_counts) if leaf_counts else "",
                    "leaf_count_mean": mean(leaf_counts) if leaf_counts else "",
                    "projected_area_px_min": min(areas) if areas else "",
                    "projected_area_px_max": max(areas) if areas else "",
                    "projected_area_px_mean": mean(areas) if areas else "",
                }
            )
    split_rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["genotype"]))
    write_csv(statistics_path, split_rows, SPLIT_FIELDS)
    if assignments:
        write_split_plot(usable_rows, assignments, plot_path)

    image_to_split = {}
    for row in usable_rows:
        key = row.get("rgb_path", "")
        if key in image_to_split and image_to_split[key] != assignments[row["plant_uid"]]:
            raise AssertionError(f"Image assigned to multiple splits: {key}")
        image_to_split[key] = assignments[row["plant_uid"]]
    payload = {
        "status": "ok" if assignments else "blocked_no_usable_frames",
        "unit": "plant_uid = genotype/replicate",
        "seed": seed,
        "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "strategy": "genotype-stratified plant-level random assignment",
        "train_plants": sorted(plant_uid for plant_uid, split in assignments.items() if split == "train"),
        "val_plants": sorted(plant_uid for plant_uid, split in assignments.items() if split == "val"),
        "test_plants": sorted(plant_uid for plant_uid, split in assignments.items() if split == "test"),
        "assignments": dict(sorted(assignments.items())),
        "image_to_split": dict(sorted(image_to_split.items())),
        "excluded_frame_count": len(rows) - len(usable_rows),
        "excluded_frame_paths": sorted(row.get("rgb_path", "") for row in rows if row not in usable_rows),
        "counts": {
            split: {
                "plants": sum(assigned == split for assigned in assignments.values()),
                "images": sum(assignments.get(row["plant_uid"]) == split for row in usable_rows),
                "genotypes": len({row["genotype"] for row in usable_rows if assignments.get(row["plant_uid"]) == split}),
            }
            for split in SPLITS
        },
        "official_split_not_used": True,
        "official_reference": {
            "published_split": "image-level random-within-replicate; intentionally not used",
            "seed": 42,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/metadata/frames.csv")
    parser.add_argument("--output", default="splits/phenoleaf_plant_split_seed42.json")
    parser.add_argument("--statistics", default="reports/m0/split_statistics.csv")
    parser.add_argument("--plot", default="reports/m0/split_distribution.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    payload = create_splits(args.frames, args.output, args.statistics, args.plot, seed=args.seed, allow_empty=args.allow_empty)
    print(json.dumps({key: payload[key] for key in ("status", "seed", "counts", "excluded_frame_count")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
