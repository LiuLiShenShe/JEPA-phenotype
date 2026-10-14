from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from birth_detection.common import write_json
from m6b.data import read_frame


COLORS = {"Current": "#767676", "C1": "#3775BA", "C2": "#42949E", "proposed": "#9A4D8E",
          "detection_failure": "#E9A6A1", "identity_failure": "#AADCA9", "measurement_failure": "#CFCECE", "correct": "#3775BA"}


def export(root, figure, name):
    directory = root / "reports/m7/figures"
    directory.mkdir(exist_ok=True)
    for extension in ["svg", "pdf", "png"]:
        figure.savefig(directory / (name + "." + extension), dpi=200, bbox_inches="tight")
    plt.close(figure)


def benchmark_figure(root):
    events = pd.read_csv(root / "reports/m7/organ_birth_events.csv")
    events = events[events.previous_observation_available]
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), layout="constrained")
    days = sorted(events.birth_time_visible.unique())
    for split, color in [("train", "#CFCECE"), ("val", "#3775BA"), ("test", "#AADCA9")]:
        counts = events[events.split == split].groupby("birth_time_visible").size().reindex(days, fill_value=0)
        axes[0].plot(days, counts, "o-", markersize=3, color=color, label=split)
    axes[0].set(xlabel="DAS at first released observation", ylabel="New visible organs", title="a  Event timing (477 organs)")
    axes[0].legend(fontsize=8)
    fractions = events.groupby("split")[["small", "core_adjacent"]].mean().reindex(["train", "val", "test"])
    positions = np.arange(3)
    axes[1].bar(positions - .18, fractions.small, width=.36, color="#3775BA", label="Small by point fraction")
    axes[1].bar(positions + .18, fractions.core_adjacent, width=.36, color="#CFCECE", label="Core distance <=5 mm")
    axes[1].set(xticks=positions, xticklabels=fractions.index, ylim=(0, 1), ylabel="Fraction of new organs", title="b  Initial geometric difficulty")
    axes[1].legend(fontsize=7, loc="upper left")
    delay = events.birth_time_segmentable - events.birth_time_visible
    valid = delay.dropna()
    bins = np.r_[[-.5], np.arange(2.5, max(valid.max() + 6, 6), 3)]
    axes[2].hist(valid, bins=bins, color="#42949E", edgecolor="white")
    axes[2].set(xlabel="Days to first frozen S2 match", ylabel="Organs (uncensored only)", title="c  Model-dependent segmentation delay")
    axes[2].text(.95, .94, f"Right-censored: {delay.isna().sum()}\nVisible != biological birth", ha="right", va="top", transform=axes[2].transAxes, fontsize=8)
    export(root, figure, "birth_benchmark")


def temporal_figure(root):
    paired = pd.read_csv(root / "reports/m7/temporal_paired_plants.csv").query("split == 'val' and metric == 'birth_f1'")
    intervals = pd.read_csv(root / "reports/m7/plant_intervals.csv").query("split == 'val'")
    decision = json.loads((root / "reports/m7/decision.json").read_text())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for position, (comparison, group) in enumerate(paired.groupby("comparison", sort=False)):
        axes[0].scatter(position + np.linspace(-.12, .12, len(group)), group.delta, color=["#3775BA", "#42949E"][position], s=24)
        axes[0].plot([position - .18, position + .18], [group.delta.mean()] * 2, color="black", linewidth=2)
    axes[0].axhline(0, color="#767676", linewidth=.8)
    axes[0].set(xticks=[0, 1], xticklabels=["History RF\n- single-feature RF", "PointNet-GRU\n- current PointNet"],
                ylabel="Within-plant event F1 difference", title="a  Temporal contrasts, 12 Validation plants")
    methods = ["Current", "C1", "C2"]
    if decision["chosen"] not in methods:
        methods.append(decision["chosen"])
    for method in methods:
        row = intervals[intervals.method == method].set_index("metric")
        horizontal, vertical = row.loc["new_event_f1"], row.loc["idf1"]
        color = COLORS.get(method, COLORS["proposed"])
        axes[1].errorbar(horizontal["mean"], vertical["mean"],
                         xerr=np.array([[horizontal["mean"] - horizontal.lower], [horizontal.upper - horizontal["mean"]]]),
                         yerr=np.array([[vertical["mean"] - vertical.lower], [vertical.upper - vertical["mean"]]]), fmt="o", color=color, capsize=3, label=method)
    axes[1].axvline(.5, color="#CFCECE", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Plant-macro exact first-observation F1", ylabel="Plant-macro observation IDF1", xlim=(0, 1), ylim=(0, 1),
                title="b  Discovery versus identity (95% plant CI)")
    axes[1].legend(fontsize=8, loc="lower right")
    figure.suptitle("Selected Validation evidence; intervals are exploratory, not blind-test inference", fontsize=10)
    export(root, figure, "temporal_evidence")


def failure_figure(root):
    event = pd.read_csv(root / "reports/m7/event_failure_contributions.csv").query("split == 'val'")
    measurement = pd.read_csv(root / "reports/m7/measurement_failure_contributions.csv").query("split == 'val' and kind == 'static' and trait == 'length'")
    chosen = json.loads((root / "reports/m7/decision.json").read_text())["chosen"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for axis, table, methods, status_key, title in [
        (axes[0], event, ["Current", chosen], "error_type", "a  At first visible scan (107 new organs)"),
        (axes[1], measurement, ["Oracle_GT_masks_GT_identity", "Current", chosen], "status", "b  Static length: all annotated observations")]:
        bottom = np.zeros(len(methods))
        for status in ["detection_failure", "identity_failure", "measurement_failure", "correct"]:
            values = table[table[status_key] == status].groupby("method").fraction.mean().reindex(methods, fill_value=0).to_numpy()
            axis.bar(np.arange(len(methods)), values, bottom=bottom, color=COLORS[status], label=status.replace("_", " "))
            bottom += values
        axis.set(xticks=np.arange(len(methods)), xticklabels=["GT masks\n+ GT identity" if name.startswith("Oracle") else name for name in methods],
                 ylim=(0, 1), ylabel="Fraction (seed average)", title=title)
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    export(root, figure, "failure_contributions")


def failure_clouds(root):
    decision = json.loads((root / "reports/m7/decision.json").read_text())
    cases = pd.read_csv(root / "reports/m7/event_level_results.csv")
    events = pd.read_csv(root / "reports/m7/organ_birth_events.csv")
    cases = cases[(cases.split == "val") & (cases.seed == 17) & (cases.method == decision["chosen"])]
    cases = cases.merge(events[["plant_id", "leaf_instance_id", "initial_point_count"]], on=["plant_id", "leaf_instance_id"], validate="many_to_one")
    selected = []
    for status in ["detection_failure", "identity_failure"]:
        rows = cases[cases.error_type == status].sort_values(["initial_point_count", "plant_id", "DAS"])
        if len(rows):
            selected.append(rows.iloc[len(rows) // 2])
    if not selected:
        selected.append(cases.sort_values(["plant_id", "DAS"]).iloc[0])
    metadata = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    figure = plt.figure(figsize=(9, 4 * len(selected)), layout="constrained")
    audit = []
    for position, event in enumerate(selected):
        source = metadata[(metadata.plant_uid == event.plant_id) & (metadata.DAS == event.DAS)].iloc[0]
        frame = read_frame(root / source.pointcloud_path)
        xyz = frame["xyz"]
        gt = np.where(frame["semantic"] == 2, frame["original_instance_evaluation_only"] + 1, 0)
        with np.load(root / f"outputs/m7/tracking/{decision['chosen']}/seed_17/val/predictions/{event.plant_id}_{event.DAS}.npz") as prediction:
            labels = prediction["labels"]
        generator = np.random.default_rng(20261013)
        indices = np.unique(np.r_[generator.choice(len(xyz), min(9000, len(xyz)), replace=False), np.flatnonzero(gt == event.leaf_instance_id + 1)])
        center = xyz.mean(axis=0)
        display = (xyz[indices] - center) * 100
        for column, values, title in [(0, gt[indices] == event.leaf_instance_id + 1, "GT first-visible leaf (red)"),
                                      (1, labels[indices], f"{decision['chosen']}: local instances")]:
            axis = figure.add_subplot(len(selected), 2, position * 2 + column + 1, projection="3d")
            colors = np.where(values, "#B64342", "#CFCECE") if column == 0 else plt.get_cmap("tab20")((values % 20) / 20)
            if column == 1:
                colors[values == 0] = matplotlib.colors.to_rgba("#CFCECE")
            axis.scatter(display[:, 0], display[:, 1], display[:, 2], c=colors, s=1, rasterized=True)
            axis.set(xlabel="x (cm, centered)", ylabel="y (cm, centered)", zlabel="z (cm, centered)",
                     title=f"{event.plant_id}, {int(event.DAS)} DAS, leaf {int(event.leaf_instance_id)}\n{event.error_type}; {title}")
            axis.view_init(elev=28, azim=45)
            axis.set_box_aspect(np.maximum(np.ptp(display, axis=0), 1e-6))
        audit.append({"plant_id": event.plant_id, "DAS": int(event.DAS), "leaf_instance_id": int(event.leaf_instance_id),
                      "failure": event.error_type, "selection": "median initial point count within failure type, Validation seed17",
                      "source_points": len(xyz), "display_points": len(indices), "full_GT_target_preserved_in_display": True})
    pd.DataFrame(audit).to_csv(root / "reports/m7/figures/case_index.csv", index=False)
    export(root, figure, "birth_failure_pointclouds")


def generate_figures(root):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    benchmark_figure(root)
    temporal_figure(root)
    failure_figure(root)
    failure_clouds(root)
    write_json(root / "reports/m7/figures/export_audit.json", {"source": "actual M7 CSV and original point clouds", "formats": ["SVG", "PDF", "PNG"],
               "backend": "Python matplotlib", "SVG_text_editable": True, "display_subsampling_disclosed": True,
               "synthetic_data": False, "human_signoff": False})
