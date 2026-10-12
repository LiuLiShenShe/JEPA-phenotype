from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {"P0": "#65758B", "P2": "#A98A60", "S500": "#518E80", "P3": "#456DB4", "P4": "#C86D53"}


def export(figure, folder, name, source, contract, qa):
    drawn_text = []
    original_draws = []

    def tracked_draw(artist, original):
        def draw(renderer):
            result = original(renderer)
            if artist.get_visible() and artist.get_text():
                drawn_text.append(artist)
            return result
        return draw

    for text in figure.findobj(matplotlib.text.Text):
        original_draws.append((text, text.draw))
        text.draw = tracked_draw(text, text.draw)
    try:
        figure.canvas.draw()
    finally:
        for text, original in original_draws:
            text.draw = original
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()
    outside = []
    for text in dict.fromkeys(drawn_text):
        bbox = text.get_window_extent(renderer)
        if bbox.x0 < -2 or bbox.y0 < -2 or bbox.x1 > width + 2 or bbox.y1 > height + 2:
            outside.append(text.get_text())
    qa.append({"figure": name, "text_outside_canvas": outside, "drawn_text_artists": len(set(drawn_text)),
               "rule": "only text whose draw method was invoked by the renderer, excluding dormant out-of-limits tick artists",
               "status": "PASS" if not outside else "REVIEW"})
    figure.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(folder / f"{name}.svg", bbox_inches="tight")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    source.to_csv(folder / f"{name}_source.csv", index=False)
    (folder / f"{name}_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    plt.close(figure)


def make_figures(root):
    root = Path(root)
    report, folder = root / "reports/m5", root / "reports/m5/figures"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    width = 183 / 25.4
    qa = []
    budget = pd.read_csv(report / "ssl_budget_comparison.csv")
    figure, axes = plt.subplots(1, 3, figsize=(width, 2.3), constrained_layout=True)
    for axis, column, label in zip(axes, ("ssl_loss", "pooled_effective_rank", "pooled_variance_mean"),
                                  ("JEPA loss", "Centred pooled effective rank", "Pooled variance")):
        values = budget.loc[budget[column].notna()]
        axis.plot(values.step, values[column], marker="o", color=COLORS["S500"])
        axis.set(xlabel="Optimizer steps", ylabel=label)
        axis.axvline(500, color="#777777", linestyle="--", linewidth=.7)
        if column == "ssl_loss":
            if not (values[column] > 0).all():
                raise ValueError("Log loss needs positive observed data")
            axis.set_yscale("log")
    figure.suptitle("M5-A: loss decreases while representation concentrates in fewer dimensions", fontsize=8)
    export(figure, folder, "01_ssl_budget", budget, {"claim": "Training loss does not establish representation success",
           "n": "24 training plants, 384 technical scans, one seed; different points are not independent plants",
           "budget": "2000-step horizon diagnostics; formal 500-step runs use a separately locked schedule"}, qa)
    metrics = pd.read_csv(report / "downstream_metrics.csv")
    count = metrics.loc[metrics.task.eq("count")]
    primary = count.loc[count.label_fraction.eq(.5)]
    figure, axis = plt.subplots(figsize=(width, 2.8), constrained_layout=True)
    partitions = ["official", "fold0", "fold1", "fold2", "fold3"]
    conditions = [value for value in COLORS if value in primary.condition.unique()]
    bar_width = .8 / len(conditions)
    for index, condition in enumerate(conditions):
        group = primary.loc[primary.condition.eq(condition)].groupby("partition").plant_macro_mae.agg(["mean", "std"]).reindex(partitions)
        axis.bar(np.arange(len(partitions)) - .4 + (index + .5) * bar_width, group["mean"], width=bar_width,
                 yerr=group["std"], capsize=2, color=COLORS[condition], label=condition)
    axis.set(xticks=np.arange(len(partitions)), xticklabels=partitions, ylabel="Plant-macro leaf-count MAE", ylim=(0, primary.plant_macro_mae.max() * 1.3))
    axis.legend(ncol=5, frameon=False, loc="upper center")
    axis.set_title("Locked primary: 12 labelled plants (50%); bars mean, whiskers seed SD")
    export(figure, folder, "02_primary_leaf_count", primary, {"claim": "Evaluate all conditions at locked 50% label budget",
           "n": "12 validation plants per split; three random seeds; four fold directions, not independent partitions",
           "historical": "P0/P2 are unchanged M4 controls; new exploratory target, not untouched confirmation"}, qa)
    figure, axes = plt.subplots(1, 2, figsize=(width, 2.7), constrained_layout=True)
    for axis, partition in zip(axes, ("official", "four_folds")):
        for condition in conditions:
            selection = count.condition.eq(condition) & (count.partition.eq(partition) if partition == "official" else count.partition.str.startswith("fold"))
            group = count.loc[selection].groupby("label_fraction").plant_macro_mae.agg(["mean", "std"])
            axis.errorbar(group.index * 100, group["mean"], yerr=group["std"], color=COLORS[condition], label=condition, marker="o", capsize=2)
        axis.set(xlabel="Labelled training plants (%)", ylabel="Plant-macro MAE", xticks=[25, 50, 100], title=partition)
        axis.legend(ncol=3, frameon=False, fontsize=6)
    export(figure, folder, "03_label_efficiency", count, {"claim": "No selecting a best label fraction after evaluation",
           "budgets": "nested 6,12,24 physical training plants", "spread": "seed SD or descriptive fold-by-seed SD; not confidence interval"}, qa)
    segmentation = metrics.loc[metrics.task.eq("segmentation") & metrics.partition.eq("official")]
    figure, axes = plt.subplots(1, 4, figsize=(width, 2.7), constrained_layout=True)
    for axis, metric, title in zip(axes, ("plant_macro_miou", "plant_macro_iou_taproot", "plant_macro_iou_core", "plant_macro_iou_leaf"), ("mIoU", "Taproot IoU", "Core IoU", "Leaf IoU")):
        group = segmentation.groupby("condition")[metric].agg(["mean", "std"]).reindex(conditions)
        axis.bar(np.arange(len(conditions)), group["mean"], yerr=group["std"], color=[COLORS[value] for value in conditions], capsize=2)
        axis.set(xticks=np.arange(len(conditions)), xticklabels=conditions, ylabel=title, ylim=(0, 1.1))
    figure.suptitle("Auxiliary semantic segmentation: 2048 sampled points; not leaf instance segmentation", fontsize=8)
    export(figure, folder, "04_semantic_transfer", segmentation, {"claim": "Negative semantic transfer is a research risk, not a substitute primary",
           "n": "official 12 validation plants, three seeds, 24 labelled train plants", "scope": "three-class semantic IoU only, no PQ/SBD claim"}, qa)
    diagnostics = json.loads((report / "representation_diagnostics.json").read_text())
    rows = []
    for identifier, checkpoints in diagnostics.items():
        if "M5A" in identifier:
            continue
        candidates = [key for key in checkpoints if key.startswith("diagnostics_step")]
        latest = max(candidates, key=lambda value: int(value.removeprefix("diagnostics_step"))) if candidates else "posttraining_probe"
        record = checkpoints[latest]
        for category, variants in record["similarities"].items():
            for mode, values in variants.items():
                rows.append({"identifier": identifier, "condition": record["condition"], "category": category, "mode": mode,
                             "median_cosine": values["quantiles_0_05_50_95_100"][2], "pooled_rank": record["pooled"]["effective_rank"],
                             "token_rank": record["token"]["effective_rank"]})
    similarity = pd.DataFrame(rows)
    figure, axes = plt.subplots(1, 2, figsize=(width, 2.6), constrained_layout=True)
    categories = ["different_plants", "same_plant_different_scans", "same_scan_different_patches"]
    for axis, mode in zip(axes, ("uncentred", "centred")):
        for condition in ("S500", "P3", "P4"):
            group = similarity.loc[similarity.condition.eq(condition) & similarity["mode"].eq(mode)].groupby("category").median_cosine.mean().reindex(categories)
            axis.plot(np.arange(3), group, marker="o", color=COLORS[condition], label=condition)
        axis.set(xticks=np.arange(3), xticklabels=["Different plants", "Same plant\nother scans", "Same scan\nother patches"], ylabel="Mean of run median cosine", title=mode, ylim=(-.25, 1.05))
        axis.legend(frameon=False)
    export(figure, folder, "05_similarity_controls", similarity, {"claim": "Separate mean-direction anisotropy from local/identity similarity",
           "n": "384 train scans/24 plants per run; pooled pairs capped at 8192; patch pairs=8192",
           "centering": "global feature mean for representation diagnostics; same scan local centering in leaf probe"}, qa)
    probe = pd.read_csv(report / "leaf_structure_probe.csv")
    probe = probe.loc[probe.partition.eq("official")]
    figure, axes = plt.subplots(1, 2, figsize=(width, 2.5), constrained_layout=True)
    for axis, metric, label in zip(axes, ("centred_pair_auc", "centred_distance_matched_margin"), ("Same-leaf pair AUC", "Geometry-matched cosine margin")):
        group = probe.groupby("condition")[metric].agg(["mean", "std"]).reindex(conditions)
        axis.bar(np.arange(len(conditions)), group["mean"], yerr=group["std"], color=[COLORS[value] for value in conditions], capsize=2)
        axis.set(xticks=np.arange(len(conditions)), xticklabels=conditions, ylabel=label)
        if metric.endswith("auc"):
            axis.axhline(.5, color="#777777", linestyle="--", linewidth=.7)
            axis.set_ylim(0, 1.05)
    figure.suptitle("Read-only training-plant leaf-token probe: not an unseen-plant instance result", fontsize=8)
    export(figure, folder, "06_leaf_structure_probe", probe, {"claim": "Same-leaf cohesion and different-leaf separability need explicit checks",
           "n": "24 official train plants, three seeds; macro over plants then seeds", "labels": "instance labels only for evaluation; token purity>=0.8; no instance-supervised encoder update",
           "confound_control": "nearest geometric-distance different-leaf pair match; residual distance difference reported"}, qa)
    comparisons = pd.read_csv(report / "paired_comparisons.csv")
    comparison = comparisons.loc[comparisons.partition.eq("official") & comparisons.task.eq("count") &
                                 comparisons.label_fraction.eq(.5) & comparisons.left.isin(("P3", "P4")) &
                                 comparisons.right.isin(("P0", "P2", "S500"))].sort_values(["left", "right"])
    figure, axis = plt.subplots(figsize=(width, 2.8), constrained_layout=True)
    for position, row in enumerate(comparison.itertuples(index=False)):
        axis.plot([row.ci95_low, row.ci95_high], [position, position], color=COLORS[row.left], linewidth=1.3)
        axis.plot(row.mean_delta, position, marker="o", color=COLORS[row.left], markersize=4)
    axis.axvline(0, color="#777777", linewidth=.8)
    axis.axvline(-.2, color="#777777", linewidth=.8, linestyle="--", label="0.2-leaf absolute threshold")
    axis.set(yticks=np.arange(len(comparison)), yticklabels=[f"{row.left} minus {row.right}" for row in comparison.itertuples(index=False)],
             xlabel="Plant-paired MAE difference (negative favours expanded model)", title="Official 50% primary: seed-averaged plant bootstrap 95% CI")
    axis.invert_yaxis()
    axis.legend(frameon=False, loc="best")
    export(figure, folder, "07_primary_paired_intervals", comparison, {"claim": "Report all fixed controls, uncertainty and the prespecified absolute threshold",
           "n": "12 official validation plants, seed differences averaged across three seeds before bootstrap",
           "interval": "10000 paired plant resamples; seed20261012; conditional on validation-selected downstream models",
           "success": "also requires 10% relative improvement, all seed directions, >=3 folds and semantic/representation gates; interval alone is insufficient"}, qa)
    status = "PASS" if all(record["status"] == "PASS" for record in qa) else "REVIEW"
    (report / "figure_source_qa.json").write_text(json.dumps({"status": status, "figures": len(qa), "records": qa}, indent=2) + "\n")
