"""Data-derived diagnostic figures, not generative or illustrative images."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.common import file_hash, write_json


def create_figures(root):
    root = Path(root)
    report = root / "reports/m1_5"
    destination = report / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(report / "paired_comparisons.csv")
    summary = pd.read_csv(report / "baseline_comparison.csv")
    feasibility = pd.read_csv(report / "calendar_feasibility.csv")
    difficulty = pd.read_csv(report / "difficulty_stratified_errors.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
                         "legend.frameon": False, "savefig.dpi": 300})
    tasks = ["offset2", "offset4", "offset8", "day3", "day5", "day7"]
    records = []
    def export(figure, name, sources, panels):
        figure.canvas.draw()
        panel_checks = []
        for axis in figure.axes:
            limits = axis.get_ylim()
            centers = [np.asarray(line.get_ydata(), dtype=float) for line in axis.lines if len(line.get_ydata())]
            finite_centers = np.concatenate(centers) if centers else np.asarray([])
            finite_centers = finite_centers[np.isfinite(finite_centers)]
            if len(finite_centers) and (finite_centers.min() < limits[0] - 1e-12 or finite_centers.max() > limits[1] + 1e-12):
                raise ValueError(f"Data point or confidence-bound clipping detected in {name}")
            panel_checks.append({"ylim": list(limits), "finite_artist_y_min": float(finite_centers.min()) if len(finite_centers) else None,
                                 "finite_artist_y_max": float(finite_centers.max()) if len(finite_centers) else None, "within_axes": True})
            point_lines = [container.lines[0] for container in axis.containers if hasattr(container, "lines") and container.lines[0] is not None]
            if len(point_lines) >= 2 and name == "time_conditions":
                first_positions = np.asarray([axis.transData.transform((line.get_xdata()[0], line.get_ydata()[0]))[0] for line in point_lines])
                minimum = float(np.diff(np.sort(first_positions)).min())
                marker_width = max(line.get_markersize() for line in point_lines) * figure.dpi / 72
                if minimum <= marker_width:
                    raise ValueError("Adjacent time-condition markers overlap horizontally")
                panel_checks[-1].update({"minimum_model_x_separation_pixels": minimum, "marker_width_pixels": marker_width,
                                         "marker_centers_are_distinguishable": True})
        for extension in ("svg", "pdf", "png"):
            figure.savefig(destination / f"{name}.{extension}", bbox_inches="tight")
        records.append({"figure": name, "panels": panels, "source_csvs": sources,
                        "files": {extension: file_hash(destination / f"{name}.{extension}") for extension in ("svg", "pdf", "png")},
                        "data_derived": True, "raw_image_manipulation": False, "panel_bounds": panel_checks})
        plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), layout="constrained")
    for axis, target, label in zip(axes, ("leaf_count", "projected_area_ratio"), ("Visible leaf count", "Projected area ratio")):
        for second, offset, color, name in (("gru_repeat", -.12, "#8172B3", "Genuine GRU - repeated-frame GRU"),
                                            ("mlp_full", .12, "#4C72B0", "Genuine GRU - single-frame MLP")):
            selected = paired.loc[paired.family.eq("capacity") & paired.target.eq(target) & paired.second.eq(second)].set_index("task").reindex(tasks)
            means = selected.delta_mae.to_numpy()
            error = np.vstack([means - selected.ci95_low, selected.ci95_high - means])
            if not np.isfinite(error).all() or (error < 0).any():
                raise ValueError("Capacity plot interval integrity failed")
            axis.errorbar(np.arange(len(tasks)) + offset, means, yerr=error, fmt="o", color=color, capsize=3, label=name)
        bounds = paired.loc[paired.family.eq("capacity") & paired.target.eq(target), ["ci95_low", "ci95_high"]]
        lower, upper = bounds.ci95_low.min(), bounds.ci95_high.max()
        margin = (upper - lower) * .12
        axis.set_ylim(lower - margin, upper + margin)
        axis.axhline(0, color=".4", linewidth=.8, linestyle="--")
        axis.set_xticks(np.arange(len(tasks)), tasks)
        axis.set_ylabel(f"{label}\nPaired MAE difference")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Historical-information test: identical plant pairing, 95% plant-bootstrap CI")
    axes[1].set_xlabel("Negative favors genuine history; seed-averaged errors, not an ensemble")
    export(figure, "capacity_control", ["paired_comparisons.csv"], 2)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout="constrained")
    methods = (("mlp_rgb", "RGB only", "#4C72B0"), ("mlp_full", "RGB + known actual interval", "#55A868"),
               ("mlp_offset", "RGB + observed time", "#8172B3"), ("mlp_relative", "RGB + relative time, no calendar", "#C4A264"),
               ("time_full", "Time only neural", "#C44E52"))
    for axis, target, label in zip(axes, ("leaf_count", "projected_area_ratio"), ("Leaf-count MAE", "Area-ratio MAE")):
        for method_position, (model, name, color) in enumerate(methods):
            selected = summary.loc[summary.split.eq("test") & summary.model.eq(model) & summary.target.eq(target) & summary.task.isin(tasks[:3])].set_index("task").reindex(tasks[:3])
            means = selected.plant_macro_mae_mean.to_numpy()
            error = np.vstack([means - selected.ci95_low, selected.ci95_high - means])
            jitter = (method_position - 2) * .13
            axis.errorbar(np.arange(3) + jitter, means, yerr=error, color=color, fmt="o-", capsize=2, linewidth=1, label=name)
        axis.set_xlabel("Original observation offset (not days)")
        axis.set_ylabel(label)
        axis.set_ylim(0, summary.loc[summary.split.eq("test") & summary.model.isin([entry[0] for entry in methods])
                                     & summary.target.eq(target) & summary.task.isin(tasks[:3]), "ci95_high"].max() * 1.18)
        axis.set_xticks([0, 1, 2], ["+2", "+4", "+8"])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=6, loc="outside lower center", ncol=2)
    figure.suptitle("Time conditions: same MLP, three seeds; display offsets only within each task")
    export(figure, "time_conditions", ["baseline_comparison.csv"], 2)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for split_position, (split, color) in enumerate((("train", "#4C72B0"), ("val", "#55A868"), ("test", "#8172B3"))):
        selected = feasibility.loc[feasibility.split.eq(split)].sort_values("days")
        axes[0].plot(selected.days, selected.plants, "o-", color=color, label=split)
        for percentile, style, marker in (("p95", "-", "o"), ("max", ":", "s")):
            axes[1].plot(selected.days + (split_position - 1) * .12, selected[f"absolute_match_error_hours_{percentile}"], marker + style, color=color,
                         linewidth=1, label=f"{split} {percentile}")
    axes[0].set_ylabel("Independent plants with matched windows")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(fontsize=7)
    axes[1].axhline(6, color=".4", linewidth=.8, linestyle="--", label="Fixed tolerance")
    axes[1].set_ylabel("Absolute target matching error (hours)")
    axes[1].set_yscale("symlog", linthresh=.01)
    axes[1].set_yticks([0, .01, .1, 1, 6], ["0", "0.01", "0.1", "1", "6"])
    axes[1].set_ylim(0, 9)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, fontsize=6, loc="outside lower center", ncol=4)
    for axis in axes:
        axis.set_xlabel("Requested future days")
        axis.set_xticks([3, 5, 7])
    figure.suptitle("Calendar feasibility: test performance not used; matching-error axis is symlog")
    export(figure, "calendar_feasibility", ["calendar_feasibility.csv"], 2)
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), layout="constrained")
    specifications = (("history_stage", ["early", "intermediate", "mature"], "Observed leaf-count stage"),
                      ("visible_leaf_decline", ["False", "True"], "Future visible-leaf decline"),
                      ("rapid_leaf_change", ["False", "True"], "Rapid change (train Q75)"))
    for axis, (factor, levels, title) in zip(axes, specifications):
        maximum = 0.
        for model, shift, color in (("mlp_full", -.2, "#4C72B0"), ("gru_full", 0, "#55A868"), ("gru_repeat", .2, "#8172B3")):
            selected = difficulty.loc[difficulty.task.eq("day5") & difficulty.target.eq("leaf_count") & difficulty.factor.eq(factor) & difficulty.model.eq(model)].copy()
            selected["level"] = selected.level.astype(str)
            selected = selected.set_index("level").reindex(levels)
            means = selected.plant_macro_mae.to_numpy()
            error = np.vstack([means - selected.ci95_low, selected.ci95_high - means])
            positions = np.arange(len(levels)) + shift
            finite = np.isfinite(error).all(axis=0)
            axis.plot(positions, means, "o", color=color, label=model)
            axis.errorbar(positions[finite], means[finite], yerr=error[:, finite], fmt="none", capsize=2, color=color)
            maximum = max(maximum, float(np.nanmax(means)), float(np.nanmax(selected.ci95_high)))
        axis.set_xticks(np.arange(len(levels)), [level if len(level) < 9 else "mid" for level in levels])
        axis.set_ylim(0, maximum * 1.15)
        axis.set_title(title, fontsize=7)
        axis.set_ylabel("Plant-macro leaf-count MAE")
        count_rows = difficulty.loc[difficulty.task.eq("day5") & difficulty.target.eq("leaf_count") & difficulty.factor.eq(factor) & difficulty.model.eq("gru_full")].set_index("level")
        counts = [int(count_rows.loc[level, "plants"]) for level in levels]
        axis.set_xlabel("Plant n: " + ", ".join(map(str, counts)))
    axes[0].legend(fontsize=6)
    figure.suptitle("Day5 difficulty associations: strata overlap in plants; no causal attribution")
    export(figure, "difficulty_day5", ["difficulty_stratified_errors.csv"], 3)
    write_json(report / "figure_source_qa.json", {"status": "PASS", "figures": records, "source_unit": "independent plant",
               "visual_qa": "separate report; source and rendering success do not certify visual inspection"})
