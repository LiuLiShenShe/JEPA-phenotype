"""Six real-data M3.5 quantitative figures with source tables and render QA."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3.common import file_hash, write_json


COLORS = {"C0-Persistence": "#7B7B7B", "C1-Linear-Growth": "#E69F00", "C2-MLP": "#56B4E9", "C3-GRU": "#0072B2", "C4-Repeated-GRU": "#CC79A7"}


def render_figures(root, predictions, protocol):
    root = Path(root)
    report, output = root / "reports/m3_5", root / "reports/m3_5/figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
                         "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    manifest = []

    def save(figure, name, source, caption):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        bounds = figure.bbox
        outside = []
        undrawn_ticks = set()
        for axis in figure.axes:
            for coordinate, limits in ((axis.xaxis, axis.get_xlim()), (axis.yaxis, axis.get_ylim())):
                lower, upper = sorted(limits)
                for tick in [*coordinate.get_major_ticks(), *coordinate.get_minor_ticks()]:
                    if tick.get_loc() < lower or tick.get_loc() > upper:
                        undrawn_ticks.update((id(tick.label1), id(tick.label2)))
        for text in figure.findobj(matplotlib.text.Text):
            if not text.get_visible() or not text.get_text() or id(text) in undrawn_ticks:
                continue
            box = text.get_window_extent(renderer)
            if box.x0 < bounds.x0 - 2 or box.y0 < bounds.y0 - 2 or box.x1 > bounds.x1 + 2 or box.y1 > bounds.y1 + 2:
                outside.append(text.get_text())
        figure.savefig(output / f"{name}.png", dpi=300, facecolor="white")
        figure.savefig(output / f"{name}.svg", facecolor="white")
        figure.savefig(output / f"{name}.pdf", facecolor="white")
        source.to_csv(output / f"{name}_source.csv", index=False)
        manifest.append({"figure": name, "caption": caption, "source_csv": f"figures/{name}_source.csv", "source_sha256": file_hash(output / f"{name}_source.csv"),
                         "all_data_real": True, "text_outside_canvas": outside, "rendered": True, "dpi": 300})
        plt.close(figure)

    grid = pd.read_csv(root / "outputs/m3_5/data/organ_trajectories.csv")
    eligible = grid.loc[grid.split.eq("val") & grid.observation_status.eq("observed")].groupby(["plant_uid", "leaf_instance_id"]).size()
    maximum = eligible.max()
    plant, leaf = sorted(eligible.loc[eligible.eq(maximum)].index)[0]
    trajectory = grid.loc[grid.plant_uid.eq(plant) & grid.leaf_instance_id.eq(leaf)].sort_values("DAS")
    selected = predictions.loc[predictions.task.eq("organ") & predictions.partition.eq("official") & predictions.split.eq("val") & predictions.plant_uid.eq(plant) & predictions.leaf_instance_id.eq(leaf) & predictions.horizon.eq(2)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), layout="constrained")
    for axis, target, label in zip(axes, ("leaf_total_length", "leaf_blade_width", "leaf_area"), ("Total length (cm)", "Blade width (cm)", "Area (cm$^2$)")):
        axis.plot(trajectory.DAS, trajectory[target], "o-", color="#333333", markersize=3)
        axis.set(xlabel="Days after sowing (DAS)", ylabel=label, xticks=[37, 51, 65, 79, 93])
    axes[0].set_yticks([10, 15, 20, 25, 30, 35])
    axes[1].set_yticks([0, 5, 10, 15, 20, 25])
    fig.suptitle(f"a  Published organ trajectory\n{plant}, leaf {leaf} (validation)")
    save(fig, "01_leaf_trajectory", trajectory, "Selection: lexically first longest observed validation trajectory, based on history coverage not model errors. Missing observations are gaps, never zero.")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    source_records = []
    for axis, target, label in zip(axes, ("leaf_total_length", "leaf_area"), ("Total length (cm)", "Area (cm$^2$)")):
        axis.plot(trajectory.DAS, trajectory[target], "o-", color="#222222", markersize=3, label="Published measurement")
        source_records.extend({"model": "published_measurement", "target": target, "target_DAS": int(row.DAS), "predicted": float(getattr(row, target)), "plant_uid": plant, "leaf_instance_id": int(leaf)} for row in trajectory.itertuples(index=False))
        for name in COLORS:
            group = selected.loc[selected.model.eq(name)].groupby("target_DAS")[f"pred_{target}"].mean()
            axis.plot(group.index, group, ".--", color=COLORS[name], linewidth=1, label=name)
            source_records.extend({"model": name, "target": target, "target_DAS": int(day), "predicted": float(value), "plant_uid": plant, "leaf_instance_id": int(leaf)} for day, value in group.items())
        axis.set(xlabel="Target DAS", ylabel=label)
    axes[1].legend(fontsize=6.5, loc="upper left", ncol=2)
    fig.suptitle("b  Two-scan-ahead organ forecasts; lines show seed means only")
    save(fig, "02_leaf_forecasts", pd.DataFrame(source_records), "Same preselected validation leaf. Forecast curves average predictions for visual clarity; all inferential metrics average per-seed errors, not ensemble forecasts. C0–C4 are Oracle organ-state conditions.")

    pairs = pd.read_csv(report / "organ_validation_pairs.csv")
    primary = pairs.loc[pairs.split.eq("val") & pairs.target.eq(protocol["primary_task"]["target"]) & pairs.horizon.eq(2) & pairs.left.eq("C3-GRU") & pairs.right.isin(["C2-MLP", "C4-Repeated-GRU"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    for axis, control in zip(axes, ("C2-MLP", "C4-Repeated-GRU")):
        subset = primary.loc[primary.right.eq(control)].set_index("partition").reindex(protocol["partitions"])
        values = subset.left_minus_right_mae.to_numpy()
        axis.errorbar(values, np.arange(len(subset)), xerr=np.vstack((values - subset.ci95_low, subset.ci95_high - values)), fmt="o", color="#0072B2", capsize=3, label="Mean + paired plant 95% CI")
        axis.axvline(0, color="#7B7B7B", linestyle=":", label="No increment")
        axis.axvline(-0.5, color="#E69F00", linestyle="--", linewidth=1, label="Practical threshold (-0.5 cm)")
        axis.set(yticks=np.arange(len(subset)), yticklabels=protocol["partitions"], xlabel="GRU minus control MAE (cm)", title=f"Against {control}")
    fig.suptitle("c  Locked primary task: h=2 leaf total length; paired plant 95% CIs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6.5)
    save(fig, "03_organ_history_increment", primary, "Negative differences favor real history. Independent unit: plant, 12 validation plants per split; errors averaged over three seeds then paired plant bootstrap (10,000 draws). Orange -0.5cm line is prespecified practical threshold; secondary tasks do not choose the branch. Fold training sets overlap.")

    event = predictions.loc[predictions.task.eq("event") & predictions.partition.eq("official") & predictions.split.eq("val") & predictions.plant_uid.eq(plant) & predictions.horizon.eq(1)]
    observed = event.drop_duplicates("window_id").sort_values("target_DAS")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.5), layout="constrained", sharex=True)
    axes[0].step(observed.target_DAS, observed.new_count, where="mid", color="#333333", label="First-observed count")
    for name, color in (("E0-DAS", "#7B7B7B"), ("E1-Logistic", "#E69F00"), ("E2-MLP", "#56B4E9"), ("E3-GRU", "#0072B2"), ("E4-Repeated-GRU", "#CC79A7")):
        group = event.loc[event.model.eq(name)].groupby("target_DAS")[["pred_new_count", "pred_event"]].mean()
        axes[0].plot(group.index, group.pred_new_count, ".--", color=color, linewidth=1, label=name)
        axes[1].plot(group.index, group.pred_event, ".-", color=color, linewidth=1, label=name)
    axes[0].set(ylabel="New visible identities")
    axes[1].scatter(observed.target_DAS, observed.event, facecolors="none", edgecolors="#333333", label="Observed event")
    axes[1].set(xlabel="Target DAS", ylabel="Event probability", ylim=(-0.05, 1.05))
    handles, labels = axes[0].get_legend_handles_labels()
    observed_handles, observed_labels = axes[1].get_legend_handles_labels()
    handles.append(observed_handles[-1])
    labels.append(observed_labels[-1])
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.5)
    fig.suptitle(f"d  New-leaf first observation: {plant} (validation; not biological birth)")
    save(fig, "04_first_observation_events", event, "Same history-coverage-selected validation plant; cumulative event labels include scans through target. Future IDs never enter model inputs. Displayed probability curves are seed means, not the locked per-seed threshold decisions.")

    current = predictions.loc[predictions.task.eq("current") & predictions.partition.eq("official") & predictions.split.eq("val")]
    geometry = pd.read_csv(root / "outputs/m3_5/data/geometry.csv")
    meta = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    geo_source = meta.merge(geometry, on=["plant_uid", "DAS"], validate="one_to_one")
    geo_source = geo_source.loc[geo_source.split.eq("val")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), layout="constrained")
    for column, target, proxy, label in ((0, "plant_height", "height_quantile_proxy_cm", "Height (cm)"), (1, "plant_width", "hull_diameter_cm", "Width (cm)")):
        axes[0, column].scatter(geo_source[target], geo_source[proxy], s=8, alpha=0.55, color="#7B7B7B")
        axes[0, column].set(xlabel=f"Published {label}", ylabel="Raw XYZ proxy (cm)")
        for name, color in (("A1-PointNet", "#0072B2"), ("A2-Geometry-Ridge", "#E69F00")):
            group = current.loc[current.model.eq(name)].groupby(["plant_uid", "DAS"])[[f"true_{target}", f"pred_{target}"]].mean()
            axes[1, column].scatter(group[f"true_{target}"], group[f"pred_{target}"], s=8, alpha=0.5, color=color, label=name)
        axes[1, column].set(xlabel=f"Published {label}", ylabel="Current estimate (cm)")
        for row in range(2):
            axis = axes[row, column]
            low = min(axis.get_xlim()[0], axis.get_ylim()[0])
            high = max(axis.get_xlim()[1], axis.get_ylim()[1])
            axis.plot([low, high], [low, high], ":", color="#333333", linewidth=1)
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7)
    fig.suptitle("e  Current geometry vs released targets: 12 validation plants")
    geo_plot_source = current.merge(geo_source, on=["plant_uid", "DAS"], how="left", suffixes=("", "_source"), validate="many_to_one")
    save(fig, "05_current_geometry_vs_targets", geo_plot_source, "Raw height quantile span is not official Height Maximum: no manual growth point, no semantic taproot removal or label-wise voxel reduction. Raw hull diameter also differs from official downsampled/excluded geometry. Ridge uses train-only calibration; no future correction.")

    errors = pd.read_csv(report / "organ_plant_errors.csv")
    worst_plants = errors.loc[errors.partition.eq("official") & errors.split.eq("val") & errors.model.eq("C3-GRU") & errors.horizon.eq(2) & errors.target.eq("leaf_total_length")].groupby("plant_uid").mae.mean()
    worst_plant = worst_plants.sort_values(ascending=False).index[0]
    source = predictions.loc[predictions.task.eq("organ") & predictions.partition.eq("official") & predictions.split.eq("val") & predictions.model.eq("C3-GRU") & predictions.plant_uid.eq(worst_plant) & predictions.horizon.eq(2)].copy()
    per_leaf = (source.pred_leaf_total_length - source.true_leaf_total_length).abs().groupby(source.leaf_instance_id).mean()
    worst_leaf = int(per_leaf.sort_values(ascending=False).index[0])
    failed = grid.loc[grid.plant_uid.eq(worst_plant) & grid.leaf_instance_id.eq(worst_leaf)].sort_values("DAS")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    for axis, target, label in zip(axes, ("leaf_total_length", "leaf_area"), ("Length (cm)", "Area (cm$^2$)")):
        axis.plot(failed.DAS, failed[target], "o-", color="#333333", markersize=3, label="Published measurement")
        group = source.loc[source.leaf_instance_id.eq(worst_leaf)].groupby("target_DAS")[f"pred_{target}"].mean()
        axis.plot(group.index, group, ".--", color="#0072B2", label="C3-GRU")
        axis.set(xlabel="DAS", ylabel=label)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"f  Validation failure case: {worst_plant}, leaf {worst_leaf} (post-hoc QC)")
    failure_source = source.loc[source.leaf_instance_id.eq(worst_leaf)].merge(failed.rename(columns={"DAS": "target_DAS"}), on=["plant_uid", "leaf_instance_id", "target_DAS"], how="left", suffixes=("", "_trajectory"), validate="many_to_one")
    save(fig, "06_failure_case", failure_source, "Post-hoc highest validation primary MAE plant and organ; diagnostic only, never used to alter task/model/branch. Missing future targets are excluded from morphology metrics but remain in visibility assessment; full measurement trace is in outputs/m3_5/data/organ_trajectories.csv.")
    write_json(report / "figure_manifest.json", {"backend": "Python/Matplotlib", "figures": manifest,
                                               "all_text_inside_canvas": all(not item["text_outside_canvas"] for item in manifest), "AI_generated_quantitative_content": False})
