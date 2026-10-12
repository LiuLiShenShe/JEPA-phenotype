from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3.common import TARGETS, read_ply, write_json, xyz_from_vertices


COLORS = {"B3-0": "#888888", "B3-1": "#B6A06A", "B3-2": "#4B79A1", "B3-3": "#2C8B79", "B3-4": "#B96C75"}
TARGET_LABELS = {"plant_height": "Plant height (cm)", "plant_width": "Plant width (cm)", "visible_organ_count": "Visible annotated leaves"}


def save(figure, destination):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(destination.with_suffix("." + extension), dpi=600 if extension == "png" else None, bbox_inches="tight")
    figure.savefig(destination.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def generate(root, metadata, predictions, comparison, per_seed, plants, paired, diagnostics, splits, sensitivity=None):
    root = Path(root)
    directory = root / "reports/m3/figures"
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "axes.spines.right": False, "axes.spines.top": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False, "figure.dpi": 130})
    generator = np.random.default_rng(20261011)
    figure = plt.figure(figsize=(13.8, 4.0), layout="constrained")
    selection = metadata.loc[metadata.plant_uid.eq("plant_002") & metadata.DAS.isin([37, 51, 65, 79, 93])]
    clouds = []
    for record in selection.itertuples(index=False):
        vertices, _ = read_ply(root / record.pointcloud_path)
        xyz = xyz_from_vertices(vertices)
        indices = generator.choice(len(xyz), min(2500, len(xyz)), replace=False)
        clouds.append((record, xyz[indices], vertices["label_instance"][indices]))
    base = np.concatenate([cloud for _, cloud, _ in clouds])
    origin = np.array([np.median(clouds[0][1][:, 0]), np.median(clouds[0][1][:, 1]), 0.0])
    lower, upper = (base.min(axis=0) - origin) * 100, (base.max(axis=0) - origin) * 100
    center = (lower + upper) / 2
    span = float(np.max(upper - lower))
    for index, (record, xyz, labels) in enumerate(clouds):
        axis = figure.add_subplot(1, 5, index + 1, projection="3d")
        xyz = (xyz - origin) * 100
        axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=labels, cmap="tab20", vmin=-2, vmax=19, s=0.7, rasterized=True)
        axis.set(xlim=(center[0] - span/2, center[0] + span/2), ylim=(center[1] - span/2, center[1] + span/2), zlim=(center[2] - span/2, center[2] + span/2))
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=-50)
        axis.set_title(f"DAS {record.DAS}", fontsize=10)
        axis.set_xlabel("x (cm)", labelpad=-1)
        axis.set_ylabel("y (cm)", labelpad=-1)
        if index == 0: axis.set_zlabel("z (cm)", labelpad=-1)
        axis.tick_params(labelsize=6, pad=0)
    figure.suptitle("a  Plant 002, fixed physical scale; organ GT colors only for visualization", fontsize=12)
    save(figure, directory / "01_temporal_pointclouds")

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), layout="constrained")
    for axis, target in zip(axes, TARGETS):
        for plant in ("plant_002", "plant_007", "plant_010", "plant_012"):
            group = metadata.loc[metadata.plant_uid.eq(plant)].sort_values("DAS")
            axis.plot(group.DAS, group[target], marker="o", markersize=3, linewidth=1, label=plant.replace("plant_", "Plant "))
        axis.set(xlabel="Actual days after sowing (DAS)", ylabel=TARGET_LABELS[target])
        axis.set_ylim(bottom=0)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", fontsize=8, ncol=4)
    figure.suptitle("b  Released automatic morphology / annotated visible count", fontsize=12)
    save(figure, directory / "02_growth_curves")

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), layout="constrained")
    for axis, target in zip(axes, TARGETS):
        for model in COLORS:
            group = comparison.loc[comparison.model.eq(model) & comparison.target.eq(target)].sort_values("horizon")
            axis.errorbar(group.horizon, group.plant_macro_mae, yerr=group.seed_mae_std, color=COLORS[model], marker="o", capsize=3, linewidth=1.4, label=model)
        axis.set(xlabel="Future actual scan offset", ylabel=f"MAE: {TARGET_LABELS[target]}", xticks=[1, 2])
        axis.set_ylim(bottom=0)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", fontsize=9, ncol=5)
    figure.suptitle("c  Official test: 12 independent plants; mean ± SD of 3 training seeds", fontsize=12)
    save(figure, directory / "03_mae_by_horizon")

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), layout="constrained")
    control_pairs = paired.loc[paired.left.eq("B3-3") & paired.right.eq("B3-4")]
    shared_cm_limit = float(control_pairs.loc[control_pairs.target.isin(["plant_height", "plant_width"]), ["ci95_low", "ci95_high"]].abs().to_numpy().max()) * 1.12
    for axis, target in zip(axes, TARGETS):
        group = paired.loc[paired.left.eq("B3-3") & paired.right.eq("B3-4") & paired.target.eq(target)].sort_values("horizon")
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.errorbar(group.horizon, group.left_minus_right_mae, yerr=[group.left_minus_right_mae-group.ci95_low, group.ci95_high-group.left_minus_right_mae],
                      marker="o", color=COLORS["B3-3"], capsize=4)
        axis.set(xlabel="Future actual scan offset", ylabel=f"Real minus repeat MAE\n{TARGET_LABELS[target]}", xticks=[1, 2])
        limit = shared_cm_limit if target in ("plant_height", "plant_width") else float(group[["ci95_low", "ci95_high"]].abs().to_numpy().max()) * 1.15
        axis.set_ylim(-limit, limit)
    figure.suptitle("d  Paired plant-bootstrap 95% CI; negative favors history, positive favors repeat", fontsize=12)
    save(figure, directory / "04_real_vs_repeat")

    worst = plants.loc[plants.model.eq("B3-3") & plants.horizon.eq(2)].groupby("plant_uid").plant_height_mae.mean().idxmax()
    selected = predictions.loc[predictions.plant_uid.eq(worst) & predictions.horizon.eq(2)]
    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), layout="constrained")
    for axis, target in zip(axes, TARGETS):
        truth = selected.groupby("target_DAS")[f"true_{target}"].first()
        axis.plot(truth.index, truth, color="#111111", marker="o", linewidth=2, label="Released target")
        for model in COLORS:
            values = selected.loc[selected.model.eq(model)].groupby("target_DAS")[f"pred_{target}"].mean()
            axis.plot(values.index, values, color=COLORS[model], linewidth=1, label=model)
        axis.set(xlabel="Target DAS", ylabel=TARGET_LABELS[target])
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", fontsize=8, ncol=6)
    figure.suptitle(f"e  Worst official test plant for GRU height h=2: {worst}; all legal windows retained", fontsize=12)
    save(figure, directory / "05_worst_plant")

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), layout="constrained")
    distribution = pd.read_csv(root / "reports/m3/split_genotype_distribution.csv")
    for position, partition in enumerate(["official", "fold0"]):
        counts = distribution.loc[distribution.partition.eq(partition)].pivot(index="genotype", columns="split", values="plants").fillna(0)
        counts = counts.reindex(sorted(counts.index, key=lambda value:int(value[1:])))
        bottom = np.zeros(len(counts))
        for split, color in zip(["train", "val", "test"], ["#A7B4BD", "#C5AB69", "#5B819F"]):
            axes[position].bar(np.arange(len(counts)), counts[split], bottom=bottom, label=split, color=color)
            bottom += counts[split].to_numpy()
        axes[position].set(xticks=np.arange(len(counts)), xticklabels=counts.index, xlabel="Genotype read from source layout (provisional)", ylabel="Independent plants", title=partition)
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    figure.suptitle("f  Plant-disjoint 24/12/12 splits; plant_021 genotype ambiguity retained", fontsize=12)
    save(figure, directory / "06_split_statistics")

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), layout="constrained")
    stratified = pd.read_csv(root / "reports/m3/stratified_errors.csv")
    for axis, target in zip(axes, TARGETS):
        for model in COLORS:
            values = stratified.loc[stratified.dimension.eq("forecast_delta_days") & stratified.model.eq(model) & stratified.target.eq(target) & stratified.horizon.eq(2)].copy()
            values["days"] = values.value.astype(int)
            values = values.sort_values("days")
            axis.plot(values.days, values.plant_macro_mae, marker="o", color=COLORS[model], label=model)
        axis.set(xlabel="Actual prediction gap (days), h=2", ylabel=f"MAE: {TARGET_LABELS[target]}")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", fontsize=9, ncol=5)
    figure.suptitle("g  Actual time gaps, not assumed fixed days", fontsize=12)
    save(figure, directory / "07_actual_gap_errors")

    if sensitivity is not None:
        figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), layout="constrained")
        for axis, target in zip(axes, TARGETS):
            for model in ("B3-2", "B3-3", "B3-4"):
                values = sensitivity.loc[sensitivity.target.eq(target) & sensitivity.model.eq(model) & sensitivity.horizon.eq(2)].sort_values("points")
                axis.errorbar(values.points, values.plant_macro_mae, yerr=values.seed_mae_std, marker="o", color=COLORS[model], label=model, capsize=3)
            axis.set(xlabel="Sampled XYZ points", ylabel=f"MAE: {TARGET_LABELS[target]}", xticks=[2048, 4096])
        handles, labels = axes[-1].get_legend_handles_labels()
        figure.legend(handles, labels, loc="outside lower center", fontsize=9, ncol=3)
        figure.suptitle("h  Predeclared point-count sensitivity; retrained, three seeds, h=2", fontsize=12)
        save(figure, directory / "08_sampling_sensitivity")
    write_json(root / "reports/m3/figure_manifest.json", {"source": "actual Sugar4D observations and recorded forecasts, never synthetic model scores", "worst_plant_selection": {"model": "B3-3", "target": "plant_height", "horizon": 2, "plant_uid": worst},
               "cloud_example": "fixed plant_002; dates 37,51,65,79,93; equal axis limits across dates", "error_bars": "model MAE: across-training-seed SD; paired control: plant-bootstrap 95% CI", "files": [str(path.relative_to(root)) for path in sorted(directory.glob("*")) if path.suffix in (".png", ".svg", ".pdf", ".tiff")]})
    return worst
