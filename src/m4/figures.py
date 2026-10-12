from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"P0": "#526d82", "P2": "#cd6b34", "PointNet": "#577e45", "A1-PointNet": "#a6a6a6", "A3-PointNet-Geometry": "#577e45"}


def save_figure(figure, path, source):
    figure.tight_layout(rect=(0, 0, 1, 0.89), pad=1.4)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    source.to_csv(path.with_name(path.stem + "_source.csv"), index=False)


def make_figures(root):
    root = Path(root)
    report, output = root / "reports/m4", root / "reports/m4/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    count = pd.read_csv(report / "label_efficiency.csv")
    segmentation = pd.read_csv(report / "segmentation_metrics.csv")
    baseline = pd.read_csv(report / "strong_baseline_validation.csv")
    histories = []
    for path in (root / "outputs/m4/checkpoints").glob("*/ssl_seed*/history.csv"):
        frame = pd.read_csv(path)
        frame["partition"] = path.parent.parent.name
        frame["seed"] = int(path.parent.name.removeprefix("ssl_seed"))
        histories.append(frame)
    history = pd.concat(histories, ignore_index=True)
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for partition, values in history.groupby("partition"):
        curve = values.groupby("epoch")[["jepa_loss", "token_std", "effective_rank"]].mean()
        for axis, column, label in zip(axes, ("jepa_loss", "token_std", "effective_rank"), ("Train JEPA loss", "Token std mean", "Embedding effective rank")):
            axis.plot(curve.index, curve[column], label=partition)
            axis.set(xlabel="SSL epoch", ylabel=label)
    if not (history.jepa_loss > 0).all():
        raise ValueError("Log loss plot requires positive finite recorded loss")
    axes[0].set_yscale("log")
    for axis in axes:
        axis.legend(fontsize=8)
    figure.suptitle("Train-only SSL monitoring (mean of seeds; loss is not representation success)")
    save_figure(figure, output / "01_pretraining_collapse.png", history)
    selected = count.loc[count.label_fraction.eq(1.0) & count["mode"].eq("finetune")]
    plotted = pd.concat([selected[["partition", "seed", "condition", "plant_macro_mae"]],
                         baseline.loc[baseline.condition.isin(("A1-PointNet", "A3-PointNet-Geometry")), ["partition", "seed", "condition", "plant_macro_mae"]]])
    figure, axis = plt.subplots(figsize=(9, 4))
    partitions = sorted(plotted.partition.unique(), key=lambda name: (name != "official", name))
    conditions = [name for name in ("A1-PointNet", "A3-PointNet-Geometry", "P0", "P2") if name in plotted.condition.unique()]
    width = 0.8 / len(conditions)
    for offset, condition in enumerate(conditions):
        values = plotted.loc[plotted.condition.eq(condition)].groupby("partition").plant_macro_mae.agg(["mean", "std"]).reindex(partitions)
        axis.bar(np.arange(len(partitions)) - 0.4 + width * (offset + 0.5), values["mean"], width,
                 yerr=values["std"].fillna(0), color=COLORS[condition], label=condition, capsize=3)
    axis.set(xticks=np.arange(len(partitions)), xticklabels=partitions, ylabel="Validation plant-macro leaf-count MAE")
    axis.legend(fontsize=8, loc="upper left", ncol=4, frameon=False)
    axis.set_ylim(0, float(plotted.plant_macro_mae.max()) * 1.45)
    axis.set_title("Current visible annotated leaf count; error bars = seed SD")
    save_figure(figure, output / "02_count_strong_baselines.png", plotted)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, mode in zip(axes, ("probe", "finetune")):
        for condition in ("P0", "P2"):
            values = count.loc[count["mode"].eq(mode) & count.condition.eq(condition)].groupby("label_fraction").plant_macro_mae.agg(["mean", "std"])
            if len(values):
                axis.errorbar(values.index * 100, values["mean"], yerr=values["std"].fillna(0), marker="o", color=COLORS[condition], label=condition, capsize=3)
        axis.set(xlabel="Independently labelled training plants (%)", title="Frozen linear probe" if mode == "probe" else "Full fine-tuning", xticks=[25, 50, 100])
        axis.legend()
    axes[0].set_ylabel("Validation plant-macro leaf-count MAE")
    figure.suptitle("Label efficiency (5 partitions × 3 seeds; SD is descriptive, folds overlap)")
    save_figure(figure, output / "03_label_efficiency_probe_vs_finetune.png", count)
    stage = pd.read_csv(report / "stage_metrics.csv")
    selected_stage = stage.loc[stage["mode"].eq("finetune") & stage.label_fraction.eq(1.0)]
    figure, axis = plt.subplots(figsize=(9, 4))
    stage_names = [name for name in ("early", "middle", "late", "outside_registered_stages_DAS93") if name in selected_stage.stage.unique()]
    for condition in ("P0", "P2"):
        values = selected_stage.loc[selected_stage.condition.eq(condition)].groupby("stage").plant_macro_mae.agg(["mean", "std"]).reindex(stage_names)
        axis.errorbar(np.arange(len(stage_names)), values["mean"], yerr=values["std"].fillna(0), marker="o", label=condition, color=COLORS[condition], capsize=3)
    axis.set(xticks=np.arange(len(stage_names)), xticklabels=["37–54 DAS", "58–72 DAS", "75–89 DAS", "93 DAS (extra)"], ylabel="Plant-macro leaf-count MAE", title="Current-scan stage errors, fixed primary count task")
    axis.legend()
    save_figure(figure, output / "04_stage_errors.png", selected_stage)
    if not segmentation.empty:
        selected_segment = segmentation.loc[segmentation["mode"].eq("finetune")]
        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        conditions = [name for name in ("PointNet", "P0", "P2") if name in selected_segment.condition.unique()]
        values = selected_segment.groupby("condition").plant_macro_miou.agg(["mean", "std"]).reindex(conditions)
        axes[0].bar(conditions, values["mean"], yerr=values["std"].fillna(0), color=[COLORS[name] for name in conditions], capsize=3)
        axes[0].set(ylabel="Validation plant-macro mIoU", ylim=(0, 1))
        for offset, condition in enumerate(conditions):
            values = selected_segment.loc[selected_segment.condition.eq(condition), ["plant_macro_iou_taproot", "plant_macro_iou_core", "plant_macro_iou_leaf"]].mean()
            axes[1].bar(np.arange(3) + (offset - 1) * 0.24, values, width=0.24, label=condition, color=COLORS[condition])
        axes[1].set(xticks=[0, 1, 2], xticklabels=["taproot", "core", "leaf"], ylabel="Plant-macro class IoU", ylim=(0, 1))
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=3, frameon=False)
        figure.suptitle("2048-point organ semantic segmentation, not leaf instance segmentation")
        save_figure(figure, output / "05_segmentation_metrics.png", selected_segment)
    failures = pd.read_csv(report / "failure_cases.csv")
    figure, axis = plt.subplots(figsize=(9, 4))
    plotted = failures.head(12).sort_values("mae")
    axis.barh(np.arange(len(plotted)), plotted.mae, color=COLORS["P2"])
    axis.set(yticks=np.arange(len(plotted)), yticklabels=[f"{row.partition}/{row.plant_uid}/{row.DAS} DAS" for row in plotted.itertuples()], xlabel="Mean seed absolute count error", title="P2 failure cases on validation; selection fixed before these plots")
    save_figure(figure, output / "06_failure_cases.png", plotted)
    segmentation_visualizations(root, output)
    full_diagnostics(root, output)
    return sorted(path.name for path in output.glob("*.png"))


def full_diagnostics(root, output):
    path = root / "reports/m4/representation_full_train_diagnostics.json"
    if not path.exists():
        return
    summary = json.loads(path.read_text())
    source_rows = []
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    displayed = []
    for condition in ("P0", "P2"):
        records = [value[condition] for value in summary.values()]
        probability_points = np.asarray([0, 0.05, 0.5, 0.95, 1])
        for axis, field, title in zip(axes[:2], ("cosine_quantiles_0_05_50_95_100", "embedding_cosine_quantiles_0_05_50_95_100"),
                                      ("Token cosine quantiles", "Global cosine quantiles")):
            quantiles = np.stack([np.asarray(value[field]) for value in records])
            axis.plot(quantiles.mean(axis=0), probability_points, marker="o", color=COLORS[condition], label=condition)
            axis.set(xlabel="Cosine similarity", ylabel="Quantile probability", title=title)
            displayed.append((axis, quantiles))
            for record_name, value in summary.items():
                for similarity, probability in zip(value[condition][field], probability_points):
                    source_rows.append({"partition_seed": record_name, "condition": condition, "field": field,
                                        "cosine_similarity": similarity, "probability": probability})
        axes[2].scatter([0 if condition == "P0" else 1] * len(records), [value["effective_rank"] for value in records],
                        color=COLORS[condition], alpha=0.7, label=condition)
    for axis in axes[:2]:
        values = np.concatenate([quantiles.flatten() for plotted_axis, quantiles in displayed if plotted_axis is axis])
        lower = max(-1, float(values.min()) - 0.002)
        axis.set_xlim(lower, 1.0002)
        axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        axis.legend(loc="upper left")
    axes[2].set(xticks=[0, 1], xticklabels=["P0", "P2"], xlabel="Pretraining condition", ylabel="Effective rank", title="All 384 Train scans (24 plants)")
    axes[2].legend()
    figure.suptitle("Post-training representation diagnostics; no validation/test data")
    save_figure(figure, output / "08_representation_similarity_distributions.png", pd.DataFrame(source_rows))


def segmentation_visualizations(root, output):
    source = root / "outputs/m4/predictions/official_segmentation_P2_finetune_labels100_seed17_points.npz"
    if not source.exists():
        return
    archive = np.load(source, allow_pickle=False)
    matrices = archive["confusion"]
    from m4.metrics import confusion_scores
    scores = np.asarray([confusion_scores(matrix)["miou"] for matrix in matrices])
    worst = int(scores.argmin())
    typical = int(np.argsort(scores)[len(scores) // 2])
    selected = [typical, worst]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), subplot_kw={"projection": "3d"})
    source_rows = []
    palette = np.asarray(["#8a6f52", "#d6a83b", "#4f944e"])
    for row_index, index in enumerate(selected):
        plant, das = str(archive["plant_uid"][index]), int(archive["das"][index])
        scan = root / "outputs/m4/data/xyz" / f"{das}_das_{plant}.npz"
        with np.load(scan, allow_pickle=False) as cloud:
            points = cloud["points"] * cloud["radius_m"] * 100
        for column_index, field in enumerate(("truth", "prediction")):
            axis = axes[row_index, column_index]
            labels = archive[field][index]
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=palette[labels], s=2, alpha=0.8, rasterized=True)
            label = "Median scan" if row_index == 0 else "Worst scan"
            axis.set(xlabel="X (cm)", ylabel="Y (cm)", zlabel="Z (cm)", title=f"{label}: {plant}, {das} DAS, {'GT' if field == 'truth' else 'P2 prediction'}")
            axis.view_init(elev=20, azim=35)
            axis.set_box_aspect(np.maximum(np.ptp(points, axis=0), 0.1))
            for point, label in zip(points, labels):
                source_rows.append({"plant_uid": plant, "DAS": das, "field": field, "x_cm": point[0], "y_cm": point[1], "z_cm": point[2], "semantic": int(label)})
    figure.suptitle("Typical / worst sampled semantic segmentation; root=brown, core=yellow, leaf=green\nCurrent physical scale retained; no future identities in model")
    save_figure(figure, output / "07_segmentation_visualization.png", pd.DataFrame(source_rows))
