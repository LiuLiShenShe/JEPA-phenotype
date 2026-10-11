#!/usr/bin/env python3
"""Generate local-only M0 quality-control figures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets.phenoleaf_utils import load_palette, resolve_project_path  # noqa: E402


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def spread(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if len(rows) <= count:
        return rows
    indices = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indices]


def discover_palette_path(project_root: str | Path) -> Path:
    root = Path(project_root)
    candidates = (
        root / "data/raw/leaf_dataset_colour/leaves_unique_colours.xlsx",
        root / "data/raw/PhenoLeaf-TS_DS/leaves_unique_colours.xlsx",
        root / "data/raw/leaves_unique_colours.xlsx",
    )
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def _figure_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def visualize(
    frames_path: str | Path,
    split_path: str | Path,
    windows_path: str | Path,
    output_dir: str | Path,
    *,
    project_root: str | Path,
    palette_path: str | Path | None = None,
    allow_empty: bool = False,
) -> list[str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frames = read_csv(frames_path)
    windows = read_csv(windows_path) if Path(windows_path).exists() else []
    if not frames:
        if not allow_empty:
            raise ValueError("No frames available for visualization")
        (output / "README.md").write_text(
            "Real-data figures were not generated because the official dataset was unavailable.\n",
            encoding="utf-8",
        )
        return []
    usable = [row for row in frames if row.get("quality_status") == "valid"]
    if not usable:
        raise ValueError("No usable frames available for visualization")
    plt = _figure_setup()
    generated: list[str] = []
    selection_manifest = {"data_scope": "local_only", "overlays": [], "growth_curves": [], "windows": []}
    split_payload = json.loads(Path(split_path).read_text(encoding="utf-8")) if Path(split_path).exists() else {}
    assignments = split_payload.get("assignments", {})

    genotype_counts: dict[str, int] = {}
    for row in usable:
        genotype_counts[row["genotype"]] = genotype_counts.get(row["genotype"], 0) + 1
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(list(sorted(genotype_counts)), [genotype_counts[key] for key in sorted(genotype_counts)])
    axis.set_title("Usable RGB/mask observations by genotype")
    axis.set_xlabel("genotype")
    axis.set_ylabel("frames")
    axis.tick_params(axis="x", rotation=60)
    path = output / "01_dataset_genotype_distribution.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    plant_counts: dict[str, int] = {}
    for row in usable:
        plant_counts[row["plant_uid"]] = plant_counts.get(row["plant_uid"], 0) + 1
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(list(plant_counts.values()), bins=min(20, max(1, len(set(plant_counts.values())))), edgecolor="white")
    axis.set_title("Observations per plant")
    axis.set_xlabel("frames")
    axis.set_ylabel("plants")
    path = output / "02_frames_per_plant.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    leaf_counts = [int(float(row["leaf_count"])) for row in usable if row.get("leaf_count", "")]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(leaf_counts, bins=range(min(leaf_counts), max(leaf_counts) + 2), align="left", rwidth=0.85)
    axis.set_title("Visible leaf-count distribution")
    axis.set_xlabel("leaf count")
    axis.set_ylabel("frames")
    path = output / "03_leaf_count_distribution.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    areas = [float(row["projected_area_px"]) for row in usable if row.get("projected_area_px", "")]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(areas, bins=30)
    axis.set_title("Projected plant area distribution")
    axis.set_xlabel("mask pixels")
    axis.set_ylabel("frames")
    path = output / "04_projected_area_distribution.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    by_plant: dict[str, list[dict[str, str]]] = {}
    for row in usable:
        by_plant.setdefault(row["plant_uid"], []).append(row)
    figure, axis = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, max(1, len(by_plant)))))
    genotype_names = sorted({row["genotype"] for row in usable})
    selected_genotypes = [item["genotype"] for item in spread([{"genotype": name} for name in genotype_names], 6)]
    flagged_plants = sorted({row["plant_uid"] for row in frames if row.get("quality_status") == "valid_with_warnings"})
    curve_plants = []
    for genotype in selected_genotypes:
        candidates = sorted(plant_uid for plant_uid in by_plant if plant_uid.startswith(genotype + "/"))
        if candidates:
            curve_plants.append(candidates[0])
    if flagged_plants and flagged_plants[0] in by_plant and flagged_plants[0] not in curve_plants:
        curve_plants[-1] = flagged_plants[0]
    for color, plant_name in zip(colors, curve_plants):
        plant_uid = {"plant_uid": plant_name}
        sequence = sorted(by_plant[plant_uid["plant_uid"]], key=lambda row: int(row["frame_idx"]))
        axis.plot([int(row["frame_idx"]) for row in sequence], [int(float(row["leaf_count"])) for row in sequence], marker=".", label=plant_uid["plant_uid"], color=color)
        selection_manifest["growth_curves"].append(plant_name)
    axis.set_title("Leaf-count growth curves (six plants)")
    axis.set_xlabel("observation index")
    axis.set_ylabel("visible leaf count")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    path = output / "05_growth_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(12, 7))
    for plant_name in curve_plants:
        sequence = sorted(by_plant[plant_name], key=lambda row: int(row["frame_idx"]))
        axis.plot([int(row["frame_idx"]) for row in sequence], [float(row["projected_area_px"]) for row in sequence], marker=".", label=plant_name)
    axis.set_title("Projected-area growth curves (same six plants)")
    axis.set_xlabel("original observation index (not days)")
    axis.set_ylabel("projected area (pixels)")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    path = output / "05b_area_growth_curves.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    palette, _, _ = load_palette(palette_path or discover_palette_path(project_root))
    warning_rows = [row for row in frames if row.get("quality_status") == "valid_with_warnings"]
    overlay_rows = spread(warning_rows, 3)
    overlay_rows += spread(sorted(usable, key=lambda row: (row["genotype"], int(row["frame_idx"]))), 12 - len(overlay_rows))
    figure, axes = plt.subplots(12, 4, figsize=(14, 36), constrained_layout=True)
    for axis_row, row in zip(axes, overlay_rows):
        rgb_path = resolve_project_path(row["rgb_path"], project_root)
        mask_path = resolve_project_path(row["mask_path"], project_root)
        with Image.open(rgb_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("RGB"))
        colored = np.zeros_like(mask)
        for color in palette:
            colored[np.all(mask == np.asarray(color), axis=-1)] = np.asarray(color)
        invalid_pixels = np.any(mask != 0, axis=-1) & ~np.any(colored != 0, axis=-1)
        colored[invalid_pixels] = (255, 0, 255)
        selection_manifest["overlays"].append({"plant_uid": row["plant_uid"], "frame_idx": row["frame_idx"], "quality_status": row["quality_status"], "growth_stage": row["growth_stage"]})
        axis_row[0].imshow(rgb)
        axis_row[0].set_title(f"RGB\n{row['plant_uid']} f{row['frame_idx']}", fontsize=7)
        axis_row[1].imshow(mask)
        axis_row[1].set_title("label mask", fontsize=7)
        axis_row[2].imshow(rgb)
        axis_row[2].imshow(colored, alpha=0.45)
        axis_row[2].set_title("overlay", fontsize=7)
        axis_row[3].axis("off")
        axis_row[3].text(0.02, 0.95, f"leaves={row['leaf_count']}\narea={row['projected_area_px']}\nquality={row['quality_status']}\nillegal pixels={row['invalid_color_pixel_count']}\nmagenta = illegal color", va="top", fontsize=8)
        for axis in axis_row[:3]:
            axis.axis("off")
    for axis_row in axes[len(overlay_rows) :]:
        for axis in axis_row:
            axis.axis("off")
    path = output / "06_rgb_mask_overlays_12.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))

    figure, axis = plt.subplots(figsize=(8, 5))
    counts = {split: sum(assigned == split for assigned in assignments.values()) for split in ("train", "val", "test")}
    image_counts = {split: sum(1 for row in usable if assignments.get(row["plant_uid"]) == split) for split in counts}
    x = np.arange(3)
    axis.bar(x - 0.18, [counts[split] for split in counts], width=0.36, label="plants")
    axis.bar(x + 0.18, [image_counts[split] for split in counts], width=0.36, label="frames")
    axis.set_xticks(x, ["Train", "Val", "Test"])
    axis.set_title("Plant-level split size comparison")
    axis.legend()
    path = output / "07_split_comparison.png"
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    generated.append(str(path))

    frame_index = {(row["plant_uid"], int(row["frame_idx"])): row for row in usable}
    window_rows = spread(windows, 6)
    figure, axes = plt.subplots(max(1, len(window_rows)), 4, figsize=(12, max(4, len(window_rows) * 3)), squeeze=False)
    for index, row in enumerate(window_rows):
        selection_manifest["windows"].append({"window_id": row["window_id"], "plant_uid": row["plant_uid"], "split": row["split"], "horizon": row["forecast_horizon"]})
        input_indices = json.loads(row["input_frame_indices"])
        for column, frame_idx in enumerate(input_indices):
            frame = frame_index.get((row["plant_uid"], int(frame_idx)))
            if frame is None:
                axes[index][column].axis("off")
                continue
            with Image.open(resolve_project_path(frame["rgb_path"], project_root)) as image:
                axes[index][column].imshow(image.convert("RGB"))
            axes[index][column].set_title(f"{row['plant_uid']}\nhistory f{frame_idx}\n{frame['timestamp']}", fontsize=8)
            axes[index][column].axis("off")
        target = frame_index.get((row["plant_uid"], int(row["target_frame_idx"])))
        if target is not None:
            with Image.open(resolve_project_path(target["rgb_path"], project_root)) as image:
                axes[index][3].imshow(image.convert("RGB"))
        axes[index][3].set_title(f"target f{row['target_frame_idx']} h{row['forecast_horizon']} obs\n{row['target_timestamp']}\nleaves={row['target_leaf_count']}", fontsize=8)
        axes[index][3].axis("off")
    if not window_rows:
        axes[0][0].text(0.05, 0.5, "No temporal windows", fontsize=12)
        axes[0][0].axis("off")
    path = output / "08_temporal_windows_6.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    generated.append(str(path))
    (output / "selection_manifest.json").write_text(json.dumps(selection_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="data/metadata/frames.csv")
    parser.add_argument("--split", default="splits/phenoleaf_plant_split_seed42.json")
    parser.add_argument("--windows", default="data/processed/temporal_windows.csv")
    parser.add_argument("--output-dir", default="reports/m0/figures")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--palette-path", default=None)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    generated = visualize(
        args.frames,
        args.split,
        args.windows,
        args.output_dir,
        project_root=args.project_root,
        palette_path=args.palette_path,
        allow_empty=args.allow_empty,
    )
    print(json.dumps({"generated": generated}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
