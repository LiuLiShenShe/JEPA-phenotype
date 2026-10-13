from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from m6a.common import digest, write_json
from m6a.experiment import cache_path, metadata
from m6a.metrics import frame_metrics
from m6b.analysis import interval, paired_statistics, prediction_errors, run_inventory, select_downstream
from m6b.data import read_frame
from m6b.evaluation import prediction_path, run_dir

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 10,
                     "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "figure.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})


def table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def tracking_tables(root, selection):
    historical = pd.read_csv(root / "reports/m6_a/per_plant.csv")
    baseline = historical[historical.pipeline.isin(["B0", "B1"])].copy()
    baseline["method"] = baseline.pipeline.map({"B0": "GT", "B1": "S0"})
    baseline["seed"] = 0
    baseline["source"] = "historical M6-A unchanged"
    parts = [baseline]
    for method in ["S1", "S2", "S3"]:
        for split in ["val", "test"]:
            path = run_dir(root, method, 17) / f"tracking_plants_{split}.csv"
            if path.exists():
                current = pd.read_csv(path)
                current["pipeline"] = "B3" if method == selection["B3_method"] else "B4" if method == "S3" else "diagnostic_" + method
                current["source"] = "current supervised segmentation; unchanged tracker/measurer"
                parts.append(current)
    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(root / "reports/m6_b/tracking_per_plant.csv", index=False)
    metrics = ["pq", "sbd", "idf1", "id_switches", "continuity", "new_event_f1", "new_event_precision", "new_event_recall",
               "length_change_tracked_coverage", "area_change_tracked_coverage", "length_change_tracked_severe", "area_change_tracked_severe",
               "length_static_tracked_loss", "area_static_tracked_loss", "new_missed_instance", "new_attached_old_identity"]
    summaries = []
    for (split, pipeline, method), group in combined.groupby(["split", "pipeline", "method"]):
        for metric in metrics:
            summaries.append({"split": split, "pipeline": pipeline, "method": method, "metric": metric, **interval(group[metric])})
    pd.DataFrame(summaries).to_csv(root / "reports/m6_b/tracking_summary.csv", index=False)
    pairs = []
    for (split, pipeline, method), group in combined[combined.pipeline.isin(["B3", "B4"])].groupby(["split", "pipeline", "method"]):
        for baseline_name in ["B0", "B1"]:
            reference = combined[(combined.split == split) & (combined.pipeline == baseline_name)].set_index("plant_uid")
            for metric in metrics:
                aligned = group.set_index("plant_uid")
                differences = aligned[metric] - reference[metric].reindex(aligned.index)
                pairs.append({"split": split, "pipeline": pipeline, "method": method, "comparison": pipeline + "-" + baseline_name,
                              "metric": metric, **interval(differences), "positive_plants": int(np.sum(differences > 0))})
    pd.DataFrame(pairs).to_csv(root / "reports/m6_b/tracking_paired_intervals.csv", index=False)
    errors = []
    old = pd.read_csv(root / "reports/m6_a/observations.csv")
    for (split, pipeline, plant), group in old[old.pipeline.isin(["B0", "B1"])].groupby(["split", "pipeline", "plant_uid"]):
        errors.append({"split": split, "pipeline": pipeline, "plant_uid": plant,
                       "instance_missed": int((group.local_id < 0).sum()),
                       "sampling_or_identity_lost": int(((group.local_id >= 0) & (group.track_id < 0)).sum()),
                       "wrong_identity": int(((group.local_id >= 0) & (group.track_id >= 0) & ~group.evaluation_identity_correct).sum()),
                       "correct": int(group.evaluation_identity_correct.sum()), "gt_observations": len(group)})
    for method in [selection["B3_method"], "S3"]:
        for split in ["val", "test"]:
            path = run_dir(root, method, 17) / f"observations_{split}.csv"
            if not path.exists():
                continue
            current = pd.read_csv(path)
            for plant, group in current.groupby("plant_uid"):
                counts = group.error_type.value_counts()
                errors.append({"split": split, "pipeline": "B3" if method != "S3" else "B4", "plant_uid": plant,
                               **{name: int(counts.get(name, 0)) for name in ["instance_missed", "sampling_or_identity_lost", "wrong_identity", "correct"]},
                               "gt_observations": len(group)})
    pd.DataFrame(errors).to_csv(root / "reports/m6_b/tracking_error_decomposition.csv", index=False)
    return combined


def first_observation_audit(root, selection):
    historical = pd.read_csv(root / "reports/m6_a/new_leaf_events.csv")
    baseline_events = historical[historical.pipeline.isin(["B0", "B1"])].copy()
    baseline_events["method"] = baseline_events.pipeline.map({"B0": "GT", "B1": "S0"})
    baseline_events["seed"] = 0
    events = baseline_events.to_dict("records")
    for method in ["S1", "S2", "S3"]:
        for split in ["val", "test"]:
            directory = run_dir(root, method, 17)
            observations_path = directory / f"observations_{split}.csv"
            if not observations_path.exists():
                continue
            observations = pd.read_csv(observations_path)
            births = pd.read_csv(directory / f"new_track_events_{split}.csv")
            birth_keys = set(zip(births.plant_uid, births.frame_index, births.local_id))
            pipeline = "B3" if method == selection["B3_method"] else "B4" if method == "S3" else "diagnostic_" + method
            for (plant, gt_id), group in observations.groupby(["plant_uid", "gt_id"]):
                first = group.sort_values("frame_index").iloc[0]
                if first.frame_index == 0:
                    continue
                state = "missed_instance" if first.local_id < 0 else "sampling_or_identity_lost" if first.track_id < 0 else (
                    "correct_first_observation" if (plant, int(first.frame_index), int(first.local_id)) in birth_keys else "attached_to_old_identity")
                detected = group[group.local_id >= 0]
                events.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "method": method, "seed": 17,
                               "gt_id_evaluation_only": int(gt_id), "first_DAS": int(first.DAS), "event_state": state,
                               "first_instance_detection_delay_days": float(detected.DAS.min() - first.DAS) if len(detected) else np.nan,
                               "meaning": "first observation, not botanical birth"})
    events = pd.DataFrame(events)
    leaves = pd.read_csv(root / "reports/m6_b/data_leaves.csv")
    threshold = json.loads((root / "reports/m6_b/data_audit.json").read_text())["small_leaf_fraction_threshold_train_q25"]
    records = metadata(root).set_index(["plant_uid", "DAS"])
    features = []
    for (plant, das), group in events.groupby(["plant_uid", "first_DAS"]):
        record = records.loc[(plant, int(das))]
        data = read_frame(root / record.pointcloud_path)
        frame_leaves = leaves[(leaves.plant_uid == plant) & (leaves.DAS == das)].set_index("local_id")
        for gt_id in group.gt_id_evaluation_only.unique():
            matched = np.flatnonzero(data["leaf_original_ids_evaluation_only"] + 1 == gt_id)
            if len(matched) != 1:
                raise ValueError("First-observation evaluator identity missing in legal current-frame leaves")
            leaf = frame_leaves.loc[int(matched[0])]
            features.append({"plant_uid": plant, "first_DAS": int(das), "gt_id_evaluation_only": int(gt_id),
                             "gt_local_id_evaluation_only": int(matched[0]) + 1,
                             "first_leaf_points": int(leaf.points), "first_leaf_fraction": float(leaf.point_fraction),
                             "small": bool(leaf.point_fraction <= threshold), "early": bool(das <= 47),
                             "core_adjacent": bool(np.isfinite(leaf.core_distance_p10_m) and leaf.core_distance_p10_m <= .005)})
    events = events.merge(pd.DataFrame(features), on=["plant_uid", "first_DAS", "gt_id_evaluation_only"], validate="many_to_one")
    error_path = root / "reports/m6_b/segmentation_leaf_errors.csv"
    if error_path.exists():
        leaf_errors = pd.read_csv(error_path).rename(columns={"DAS": "first_DAS", "gt_local_id": "gt_local_id_evaluation_only"})
        keys = ["method", "seed", "split", "plant_uid", "first_DAS", "gt_local_id_evaluation_only"]
        events = events.merge(leaf_errors[keys + ["merge_error", "split_error", "sampling_loss", "sampled_semantic_leaf_recall"]],
                              on=keys, how="left", validate="many_to_one")
    events.to_csv(root / "reports/m6_b/first_observation_events.csv", index=False)
    strata = []
    for category in ["all", "small", "early", "core_adjacent"]:
        current = events if category == "all" else events[events[category]]
        for (split, pipeline, plant), group in current.groupby(["split", "pipeline", "plant_uid"]):
            strata.append({"split": split, "pipeline": pipeline, "plant_uid": plant, "category": category,
                           "events": len(group), "first_observation_recall": float((group.event_state == "correct_first_observation").mean()),
                           "first_instance_miss_rate": float((group.event_state == "missed_instance").mean()),
                           "old_identity_rate": float((group.event_state == "attached_to_old_identity").mean()),
                           "sampling_loss_rate": float((group.event_state == "sampling_or_identity_lost").mean())})
    strata = pd.DataFrame(strata)
    strata.to_csv(root / "reports/m6_b/first_observation_strata_per_plant.csv", index=False)
    summary = []
    for (split, pipeline, category), group in strata.groupby(["split", "pipeline", "category"]):
        for metric in ["first_observation_recall", "first_instance_miss_rate", "old_identity_rate", "sampling_loss_rate"]:
            summary.append({"split": split, "pipeline": pipeline, "category": category, "metric": metric,
                            "event_observations": int(group.events.sum()), **interval(group[metric])})
    pd.DataFrame(summary).to_csv(root / "reports/m6_b/first_observation_strata_summary.csv", index=False)
    return events


def training_table(root):
    rows, validation = [], []
    for method, seed in run_inventory(root):
        if method == "S0":
            continue
        directory = run_dir(root, method, seed)
        complete = json.loads((directory / "training_complete.json").read_text())
        manifest = json.loads((directory / "training_manifest.json").read_text())
        post = json.loads((directory / "postprocessing_selection.json").read_text())
        epochs = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
        tail = epochs[-10:]
        head = epochs[:10]
        selected = [row for row in epochs if "val_pq" in row]
        for row in selected:
            validation.append({"method": method, "seed": seed, **row})
        rows.append({"method": method, "seed": seed, "parameters": manifest["parameters"], "epochs": complete["epochs"],
                     "updates": complete["optimizer_steps"], "frame_presentations": complete["frame_presentations"],
                     "loaded_entries": len(manifest["loaded_state_keys"]), "random_entries": len(manifest["randomly_initialized_state_keys"]),
                     "selected_epoch": post["checkpoint_epoch"], "bandwidth": post["bandwidth"],
                     "train_seconds": sum(row["training_seconds"] for row in epochs),
                     "first10_loss": float(np.mean([row["loss"] for row in head])), "last10_loss": float(np.mean([row["loss"] for row in tail])),
                     "last10_discriminative": float(np.mean([row["discriminative"] for row in tail])),
                     "validation_final_minus_50": selected[-1]["val_pq"] - selected[-2]["val_pq"],
                     "checkpoint_sha256": complete["best_sha256"]})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "reports/m6_b/training_summary.csv", index=False)
    pd.DataFrame(validation).to_csv(root / "reports/m6_b/validation_learning_curves.csv", index=False)
    return frame


def make_decision(root, plants, paired, tracking, training, selection):
    protocol = json.loads((root / "configs/m6_b/protocol.json").read_text())["continuation"]
    strata = pd.read_csv(root / "reports/m6_b/error_strata_per_plant.csv")
    outcomes = []
    for pipeline, method in [("B3", selection["B3_method"]), ("B4", "S3")]:
        observed = tracking[(tracking.split == "val") & (tracking.pipeline == pipeline)]
        seg = plants[(plants.split == "val") & (plants.method == method) & (plants.seed == 17)]
        if observed.empty or seg.empty:
            continue
        early = strata[(strata.split == "val") & (strata.method == method) & (strata.seed == 17) & (strata.category == "early_small")]
        improvement = paired[(paired.split == "val") & (paired.method == method) & (paired["seed"].astype(str) == "17") & (paired.metric == "pq")]
        positive = bool(len(improvement) and improvement.iloc[0].ci95_low > 0 and improvement.iloc[0].improved_plants >= 9)
        if method in ["S1", "S2"]:
            mean_improvement = paired[(paired.split == "val") & (paired.method == method) & (paired["seed"].astype(str) == "mean_seeds") & (paired.metric == "pq") & (paired.comparison == method + "-S0")]
            positive = positive and bool(len(mean_improvement) and mean_improvement.iloc[0].ci95_low > 0 and mean_improvement.iloc[0].improved_plants >= 9)
        outcome = {"pipeline": pipeline, "method": method, "pq": float(seg.pq.mean()), "idf1": float(observed.idf1.mean()),
                   "new_event_f1": float(observed.new_event_f1.mean()), "stable_improvement": positive,
                   "early_small_recall": float(early.recall.mean()) if len(early) else None,
                   "new_missed_instance": float(observed.new_missed_instance.mean()) if "new_missed_instance" in observed else 0.,
                   "new_attached_old_identity": float(observed.new_attached_old_identity.mean()) if "new_attached_old_identity" in observed else 0.}
        outcome["passes"] = bool(outcome["pq"] >= protocol["PQ_min"] and outcome["idf1"] >= protocol["IDF1_min"]
                                 and outcome["new_event_f1"] >= protocol["new_first_observation_F1_min"] and positive
                                 and len(early) and outcome["early_small_recall"] >= protocol["early_small_recall_min"])
        outcomes.append(outcome)
    convergence_adequate = bool(len(training) == 7 and (training.validation_final_minus_50.abs() < .03).all()
                                and (training.last10_discriminative < .25).all())
    b3 = next((row for row in outcomes if row["pipeline"] == "B3"), None)
    instance_bottleneck = bool(b3 is not None and (
        b3["pq"] < protocol["PQ_min"]
        or (b3["early_small_recall"] is not None and b3["early_small_recall"] < protocol["early_small_recall_min"])
        or (b3["new_event_f1"] < protocol["new_first_observation_F1_min"]
            and b3["new_missed_instance"] > b3["new_attached_old_identity"])))
    if any(row["passes"] for row in outcomes):
        recommendation = "INSTANCE-SOLVED-FOR-TRACKING"
        rationale = "A locked supervised baseline passes Validation segmentation, tracking, first-observation and stable-direction gates; early-small recall also passes."
    elif convergence_adequate and b3 is not None and b3["stable_improvement"] and instance_bottleneck:
        recommendation = "INSTANCE-METHOD-NEEDED"
        rationale = "A converged bounded supervised baseline improves independent plants stably but fails locked gates with low PQ/early-small recall or predominantly instance-missed first observations; a tracking-only failure is not attributed to segmentation. No dense strong-baseline superiority is claimed."
    else:
        recommendation = "DATA-LIMITED"
        rationale = "The 12-plant Validation evidence and bounded/convergence or strong-baseline completeness cannot establish a stable solved-vs-method-needed judgment; stop tuning and improve independent data/validation evidence."
    decision = {"recommendation": recommendation, "rationale": rationale, "validation_gates": outcomes,
                "convergence_adequate_for_method_needed": convergence_adequate,
                "instance_bottleneck_diagnostic": instance_bottleneck,
                "attribution_diagnostic": "post-training diagnosis, not a changed success gate: first-observation instance-missed count compared with old-identity-attached count; low PQ or early-small recall also identifies an instance failure",
                "convergence_diagnostic": "disclosed post-training adequacy diagnostic, not an added success gate: all final-vs50 PQ changes <.03 and last10 embedding loss <.25",
                "measurement_system_certified": False, "S4": "BLOCKED", "test_not_used_for_decision": True,
                "exactly_one": True, "no_automatic_next_method": True}
    write_json(root / "reports/m6_b/decision.json", decision)
    return decision


def save_figure(root, figure, name, manifests):
    destination = root / "reports/m6_b/figures"
    destination.mkdir(exist_ok=True)
    figure.savefig(destination / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(destination / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(destination / f"{name}.svg", bbox_inches="tight")
    manifests.append({"name": name, "data_source": "real Sugar4D PLY and scored predictions; no generative pixels",
                      "png_sha256": digest(destination / f"{name}.png")})
    plt.close(figure)


def figures(root, plants, tracking, selection):
    manifests = []
    methods = ["S0", "S1", "S2", "S3"]
    colors = ["#6c757d", "#247ba0", "#70ad47", "#d58936"]
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric in zip(axes, ["pq", "sbd", "recall"]):
        for position, (method, color) in enumerate(zip(methods, colors)):
            group = plants[(plants.split == "val") & (plants.method == method)]
            values = group.groupby("plant_uid")[metric].mean()
            ci = interval(values)
            if ci["mean"] is not None:
                axis.bar(position, ci["mean"], color=color, width=.6)
                axis.errorbar(position, ci["mean"], yerr=[[ci["mean"] - ci["ci95_low"]], [ci["ci95_high"] - ci["mean"]]], color="black", capsize=4)
                axis.scatter(position + np.linspace(-.18, .18, len(values)), values, color="black", s=12, alpha=.5)
        axis.set_xticks(range(4), methods)
        axis.set_ylim(0, 1)
        axis.set_title("Validation " + metric.upper())
        axis.grid(axis="y", alpha=.2)
    axes[0].axhline(.7, color="red", ls="--", lw=1)
    figure.suptitle("12 independent plants; S1/S2 seed-mean; plant bootstrap 95% intervals")
    save_figure(root, figure, "validation_instance_summary", manifests)
    curves = pd.read_csv(root / "reports/m6_b/validation_learning_curves.csv")
    figure, axis = plt.subplots(figsize=(7, 4))
    for (method, seed), group in curves.groupby(["method", "seed"]):
        axis.plot(group.epoch, group.val_pq, marker="o", label=f"{method} seed{seed}")
    axis.axhline(.7, color="red", ls="--", lw=1)
    axis.set(xlabel="Epoch (fixed budget 60)", ylabel="Full-resolution plant-macro PQ", ylim=(0, 1))
    axis.legend(fontsize=8, ncol=2)
    save_figure(root, figure, "validation_learning_curves", manifests)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    pipelines = ["B0", "B1", "B3", "B4"]
    for axis, metric, threshold in zip(axes, ["idf1", "new_event_f1", "length_change_tracked_coverage"], [.8, .7, None]):
        for position, pipeline in enumerate(pipelines):
            values = tracking[(tracking.split == "val") & (tracking.pipeline == pipeline)][metric]
            ci = interval(values)
            if ci["mean"] is not None:
                axis.bar(position, ci["mean"], color=colors[position])
                axis.errorbar(position, ci["mean"], yerr=[[ci["mean"] - ci["ci95_low"]], [ci["ci95_high"] - ci["mean"]]], color="black", capsize=4)
        axis.set_xticks(range(4), pipelines)
        axis.set_title(metric)
        axis.set_ylim(0, 1)
        if threshold is not None:
            axis.axhline(threshold, color="red", ls="--", lw=1)
    figure.suptitle(f"Fixed tracker/measurer; B3={selection['B3_method']} seed17; B4=PSegNet seed17")
    save_figure(root, figure, "segmentation_to_tracking", manifests)
    errors = pd.read_csv(root / "reports/m6_b/tracking_error_decomposition.csv")
    summary = errors[errors.split == "val"].groupby("pipeline")[["instance_missed", "sampling_or_identity_lost", "wrong_identity", "correct", "gt_observations"]].sum()
    figure, axis = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(pipelines))
    for error, color in zip(["correct", "instance_missed", "sampling_or_identity_lost", "wrong_identity"], ["#70ad47", "#ba4444", "#c7a83b", "#7c62a3"]):
        values = (summary.loc[pipelines, error] / summary.loc[pipelines, "gt_observations"]).to_numpy()
        axis.bar(pipelines, values, bottom=bottom, label=error, color=color)
        bottom += values
    axis.set(ylabel="Fraction of GT observations", ylim=(0, 1), title="Observation errors; diagnostic, not independent replicates")
    axis.legend(fontsize=8, bbox_to_anchor=(1, 1))
    save_figure(root, figure, "detection_vs_identity_errors", manifests)
    chosen = plants[(plants.method == selection["B3_method"]) & (plants.seed == 17) & (plants.split == "val")].sort_values("pq")
    cases = [chosen.iloc[0].plant_uid, chosen.iloc[-1].plant_uid]
    metadata_frame = metadata(root)
    for plant in cases:
        figure, axes = plt.subplots(2, 3, figsize=(10, 8))
        for row_index, das in enumerate([44, 79]):
            record = metadata_frame[(metadata_frame.plant_uid == plant) & (metadata_frame.DAS == das)].iloc[0]
            data = read_frame(root / record.pointcloud_path)
            gt = np.where(data["local_instance"] >= 0, data["local_instance"] + 1, 0)
            with np.load(cache_path(root, plant, das)) as original:
                zero = original["labels"].copy()
            with np.load(prediction_path(root, selection["B3_method"], 17, plant, das)) as current:
                supervised = current["labels"].copy()
            indices = np.random.default_rng(17 + das).choice(len(gt), min(18000, len(gt)), replace=False)
            cloud = data["xyz"][indices] * 100
            for axis, label, name in zip(axes[row_index], [gt, zero, supervised], ["GT", "S0", "B3 supervised"]):
                score, matches = frame_metrics(gt, label)
                colors_current = np.full((len(indices), 4), [.7, .7, .7, 1])
                assigned = {predicted: actual for actual, (predicted, unused_iou) in matches.items()}
                assigned[0] = 0
                palette = plt.get_cmap("tab20")
                for local in np.unique(label[indices]):
                    if local > 0:
                        color_id = assigned.get(int(local), int(local) + 20)
                        colors_current[label[indices] == local] = palette((color_id - 1) % 20)
                axis.scatter(cloud[:, 0], cloud[:, 1], c=colors_current, s=.4, rasterized=True)
                axis.set_aspect("equal")
                axis.set_title(f"DAS{das} {name}; PQ={score['pq']:.3f}", fontsize=10)
                axis.set(xlabel="x (cm)", ylabel="y (cm)")
        figure.suptitle(f"{plant}: deterministic {'worst' if plant == cases[0] else 'best'} B3 Validation plant; IoU-matched colors")
        save_figure(root, figure, plant + "_instance_cases", manifests)
        record = metadata_frame[(metadata_frame.plant_uid == plant) & (metadata_frame.DAS == 44)].iloc[0]
        data = read_frame(root / record.pointcloud_path)
        gt = np.where(data["local_instance"] >= 0, data["local_instance"] + 1, 0)
        with np.load(cache_path(root, plant, 44)) as original:
            zero = original["labels"].copy()
        with np.load(prediction_path(root, selection["B3_method"], 17, plant, 44)) as current:
            supervised = current["labels"].copy()
        indices = np.random.default_rng(61).choice(len(gt), min(12000, len(gt)), replace=False)
        cloud = data["xyz"][indices] * 100
        figure = plt.figure(figsize=(13, 4.8))
        for panel_index, (label, name) in enumerate(zip([gt, zero, supervised], ["GT", "S0", "B3 supervised"])):
            axis = figure.add_subplot(1, 3, panel_index + 1, projection="3d")
            score, matches = frame_metrics(gt, label)
            associated = {predicted: actual for actual, (predicted, unused_iou) in matches.items()}
            point_colors = np.full((len(indices), 4), [.7, .7, .7, 1])
            palette = plt.get_cmap("tab20")
            for local in np.unique(label[indices]):
                if local > 0:
                    color_id = associated.get(int(local), int(local) + 20)
                    point_colors[label[indices] == local] = palette((color_id - 1) % 20)
            axis.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=point_colors, s=.7, depthshade=False, rasterized=True)
            axis.view_init(elev=24, azim=-58)
            axis.set_box_aspect(np.ptp(cloud, axis=0))
            axis.set(xlabel="x (cm)", ylabel="y (cm)", zlabel="z (cm)", title=f"{name}; PQ={score['pq']:.3f}")
            for coordinate_axis in [axis.xaxis, axis.yaxis, axis.zaxis]:
                coordinate_axis.set_major_locator(MaxNLocator(nbins=4))
            axis.tick_params(axis="both", labelsize=8)
        figure.suptitle(f"{plant} DAS44: real early 3D leaf instances; evaluation-only matched colors")
        save_figure(root, figure, plant + "_early_instances_3d", manifests)
    write_json(root / "reports/m6_b/figure_manifest.json", {"selection": "minimum/maximum B3 Validation plant PQ, no hand-picked successes",
               "figures": manifests, "vision_review_required": True,
               "source_csv_sha256": {name: digest(root / "reports/m6_b" / name) for name in
                   ["segmentation_per_plant.csv", "tracking_per_plant.csv", "validation_learning_curves.csv", "tracking_error_decomposition.csv"]}})


def deliver(root):
    selection = select_downstream(root)
    training = training_table(root)
    plants = prediction_errors(root)
    paired = paired_statistics(root, plants)
    tracking = tracking_tables(root, selection)
    events = first_observation_audit(root, selection)
    decision = make_decision(root, plants, paired, tracking, training, selection)
    figures(root, plants, tracking, selection)
    report = root / "reports/m6_b"
    summary = pd.read_csv(report / "segmentation_summary.csv")
    track_summary = pd.read_csv(report / "tracking_summary.csv")
    segmentation_table = plants.groupby(["split", "method", "seed"])[["pq", "sbd", "precision", "recall", "miss_rate", "merge_rate", "split_rate"]].mean().reset_index()
    seed_means = plants.groupby(["method", "split", "plant_uid"])[["pq", "sbd"]].mean().reset_index()
    seed_summary = []
    for (method, split), group in seed_means.groupby(["method", "split"]):
        seed_summary.append({"method": method, "split": split, "aggregation": "plant means after seed averaging", **interval(group.pq)})
    pd.DataFrame(seed_summary).to_csv(report / "seed_mean_plant_pq.csv", index=False)
    tracking_table = tracking[tracking.pipeline.isin(["B0", "B1", "B3", "B4"])].groupby(["split", "pipeline", "method"])[
        ["pq", "idf1", "id_switches", "continuity", "new_event_f1", "length_change_tracked_coverage", "area_change_tracked_coverage", "length_change_tracked_severe", "area_change_tracked_severe"]].mean().reset_index()
    strata = pd.read_csv(report / "error_strata_per_plant.csv")
    strata_table = strata.groupby(["split", "method", "seed", "category"]).agg(
        recall=("recall", "mean"), split_rate=("split_rate", "mean"), merge_rate=("merge_rate", "mean"),
        sampling_loss=("sampling_loss", "mean"), independent_plants=("plant_uid", "nunique"), leaf_observations=("leaves", "sum")).reset_index()
    runtime = []
    for method, seed in run_inventory(root):
        if method == "S0":
            continue
        for split in ["val", "test"]:
            path = run_dir(root, method, seed) / f"segmentation_frames_{split}.csv"
            if path.exists():
                current = pd.read_csv(path)
                total = current.forward_seconds + current.cluster_restore_seconds
                runtime.append({"method": method, "seed": seed, "split": split, "frames": len(current),
                                "forward_seconds_mean": current.forward_seconds.mean(), "cluster_restore_seconds_mean": current.cluster_restore_seconds.mean(),
                                "frames_per_second_total": float(1 / total.mean()), "peak_vram_GB": float(current.peak_vram_bytes.max() / 1e9),
                                "note": "concurrent GPU0 training/inference may affect speed; VRAM is per-process allocator, not whole-card peak"})
    pd.DataFrame(runtime).to_csv(report / "inference_runtime.csv", index=False)
    runtime_table = pd.DataFrame(runtime)
    runtime_table = runtime_table[runtime_table.split == "val"][["method", "seed", "forward_seconds_mean", "cluster_restore_seconds_mean", "frames_per_second_total", "peak_vram_GB"]]
    benchmark_path = report / "inference_microbenchmark.csv"
    microbenchmark = table(pd.read_csv(benchmark_path)[["method", "DAS", "forward_seconds_median", "frames_per_second_forward", "peak_allocated_vram_GB", "exact_repeat_logits_embeddings"]]) if benchmark_path.exists() else "单设备重复前向microbenchmark尚未执行；实际逐帧推理统计已经保存。"
    block = json.loads((report / "organ3d_feasibility.json").read_text())
    val_delta = paired[(paired.split == "val") & (paired.metric == "pq")]
    b3_gate = next(row for row in decision["validation_gates"] if row["pipeline"] == "B3")
    new_events = events[(events.split == "val") & (events.pipeline == "B3")]
    missed_new = new_events[new_events.event_state == "missed_instance"]
    adjacent = strata[(strata.split == "val") & (strata.method == selection["B3_method"]) & (strata.seed == 17) & (strata.category == "core_adjacent")]
    event_diagnostic_table = events[events.split == "val"].groupby(["pipeline", "event_state"]).agg(
        event_observations=("event_state", "size"), small_event_observations=("small", "sum"),
        independent_plants=("plant_uid", "nunique")).reset_index()
    report_text = f"""# M6-B：Sugar4D 监督式叶片实例分割与下游追踪

## 0. 唯一研究建议

**{decision['recommendation']}**。{decision['rationale']}

本阶段已实际训练 S1/S2 各三个种子、S3 一个种子，全部60轮、每轮384训练帧。
S4 **BLOCKED**，没有伪造 Organ3DNet 推理/训练指标。没有训练JEPA、生成模型、追踪器或动态预测网络。
Validation为模型选择/继续判断主集合；旧Test仅探索性固定方案复核，不是未使用盲测。
当前长度/面积测量器明显有偏；本建议不认证整套表型系统。
当前固定B3={selection['B3_method']} seed17：Validation PQ={b3_gate['pq']:.4f}、IDF1={b3_gate['idf1']:.4f}均超过原门槛，首观测F1={b3_gate['new_event_f1']:.4f}仍低于.70。
Validation共{len(new_events)}次新叶首观测，{len(missed_new)}次无合格当前帧实例匹配，其中{int(missed_new.small.sum())}次属于Train阈值定义的小叶；另{int((new_events.event_state == 'attached_to_old_identity').sum())}次已匹配叶被接到旧身份。芯部邻近代理的植株宏召回仅{adjacent.recall.mean():.4f}（{adjacent.plant_uid.nunique()}株），该失败类型支持针对性实例研究，但不证明整套主干架构无效或优于未完成的Organ3DNet。
这{len(missed_new)}次首帧漏检中，合并代理{int(missed_new.merge_error.sum())}次、分裂代理{int(missed_new.split_error.sum())}次（可重叠），4096点采样完全漏掉该叶的次数为{int(missed_new.sampling_loss.sum())}；不是简单把未采到点误称为网络失败。

## 1. 冻结协议、公平性与数据

- 协议 `configs/m6_b/protocol.json` SHA256 `{digest(root / 'configs/m6_b/protocol.json')}`。
- 锁定时间在 `protocol_lock.json`，早于真实目标域训练。原官方24/12/12整株划分不改动；每株16个DAS：37,40,44,47,51,54,58,61,65,72,75,79,82,86,89,93。
- 主指标：全分辨率逐帧 strict IoU>0.5 PQ，先按植物平均，再植物宏平均；不是把192验证扫描当192株。所有植株、合法叶片都保留。
- CE监督taproot/core/leaf三类；负实例仅作非叶实例哨兵，叶0是合法叶。当前叶实例每帧连续重映射、每轮随机置换；模型输入只有XYZ。
- 无跨时间GT身份、未来扫描、真叶数、GT选点、Validation/Test训练。训练24株384帧不变，4096点，固定32768训练池、逐轮重采样、z旋转，中心/尺度/原始索引可恢复到米。
- S0完整沿用M6-A：原checkpoint、单位球归一化、MeanShift .6、源物种均值过滤和预测缓存，历史PQ/SBD/追踪不覆盖。M6-A官方归一化诊断亦保留，不用来择优替代S0。
- S1/S2相同官方PlantNet主干、同监督量/学习率/损失/验证时点/后处理网格。S2只随机三类语义输出头，加载其它全部兼容状态且全解冻。S1不加载任何源物种权重。
- S1/S2有目标域标签、S0无目标域标签：提升首先是监督适配收益，不是架构创新。目标归一化、叶-only损失接口、后处理也是公开的适配变化，不孤立归因于权重微调。
- Seed17固定作下游，不选最好种子。B3方法族按三种子Validation PQ平均选择为 **{selection['B3_method']}**；B4为可运行PSegNet seed17，不能冒称已复现密集查询Organ3DNet。

## 2. 实际训练和加载审计

独立`.venv-m6-b`，Python3.11/Torch2.8+cu128，历史环境不改。实际批4/4096点前反向和Adam烟测通过。
最初只用GPU0；GPU1原任务结束并核验空闲后，S2用相同A6000 GPU1并发。共享GPU1后来出现外部作业，S2在完整第37轮保存边界迁回GPU0，从模型、Adam和RNG状态精确恢复；每run仍只有5760更新。只改资源调度，数据/种子/预算/参数不变；CLI逐run锁防重复，实际控制台日志分`train_S2_*_gpu1.log`及`train_S2_*.log`。共享主机非独占，见`RESOURCE_SCHEDULING.md`和`S2_device_migration.json`。
统一Adam lr.001、weight_decay.001、60轮余弦名义下限.00002（最后实际使用约.00002067，逐更新lr有完整记录）、FP32、clip10，全监督帧每轮一遍；每个run5760更新/23040帧呈现。
监督来源为24株384帧、3419次当前帧叶实例观测；每run训练输入点标签呈现94,371,840次（384×60×4096），不是这些数量的独立植物。每帧训练池取min(32768,该PLY点数)统一随机点，共12,555,297点（2帧原云较小），池覆盖全部合法Train叶；完整Train原始点标签共150,776,654点，不声称每轮用到了完整PLY的每个标签点。
Validation每10轮在完整12株192帧计算PQ择checkpoint，后处理中带宽.4/.6/.8只看Validation。
保留官方CE×10、L1 embedding×10及double-hinge相似性；core/taproot分开CE、实例项仅叶。
特别披露：官方同实例相似性项为relu(80-D)，没有悄悄改符号；局部ID权重改为逐帧，避免批内不同植物身份混同。
原官方预算200轮，本轮60轮是有界目标监督验证；不因Test好坏延长预算。

{table(training.drop(columns=['checkpoint_sha256']).round(5))}

逐run成功/随机状态键、冻结列表、源checkpoint及源码hash在`outputs/m6_b/S*/seed_*/training_manifest.json`；逐更新`batches.csv`、逐轮`epochs.jsonl`、最后/验证轮/最佳checkpoint均保留。
外部预训练源烟草/番茄/高粱（源码报告546点云，不可当546独立植株）；S1/S3无外部预训练。
训练预算充分性诊断：{decision['convergence_adequate_for_method_needed']}。末轮-v50 PQ和末10轮embedding loss逐run公开，不能把不收敛或预算不足证明为架构缺陷。

## 3. Validation主结果与探索性Test

{table(segmentation_table)}

Plant seed-mean PQ及植物95%bootstrap区间（三个种子不是36个独立植物）：
Validation的S1/S2各包含3个种子；Test仅seed17固定方案复核，S3和S0仅一个固定模型。表中平均只覆盖实际存在的seed，不把Test一seed当三seed稳健性证据。

{table(pd.DataFrame(seed_summary))}

Validation配对植物PQ差值（mean_seeds亦先按植物平均）：

{table(val_delta[['method','seed','comparison','mean','ci95_low','ci95_high','plants','improved_plants']])}

全部逐株/逐帧、各指标区间、不同DAS及叶数条件在`segmentation_per_plant.csv`、`segmentation_frame_metrics.csv`、`segmentation_summary.csv`、`pq_by_das.csv`、`pq_by_leaf_count.csv`。
均为保留全部合法叶片的完整PLY点评价；恢复后污染背景也计入预测实例并影响IoU。

### 实际推理速度与显存

{table(runtime_table)}

以上逐帧时长为实际共享GPU推理，forward不含PLY加载，cluster_restore含MeanShift和全分辨率最近邻及评分开销；实际推理加载了带优化器的训练checkpoint，分配峰值不能冒称最小部署显存。与同时发生的训练/其他任务存在竞争，不作公平架构速度排名。
单设备、同一真实植物DAS37/79、4096点、3次预热10次前向重复（checkpointCPU加载，不把optimizer放GPU）：

{microbenchmark}

microbenchmark仅forward，不含I/O/聚类/回传；记录共享主机GPU快照并检查logits/embedding逐次精确一致，不宣称独占GPU性能。

## 4. 明确实例错误与早期小叶

漏检：无IoU>0.5匹配；合并：一预测实例覆盖多个GT叶各>10%；分裂：一GT叶被多个预测实例覆盖各>10%。
漏检/分裂分母是GT实例，合并分母是预测实例；逐株算比例再平均。小叶阈值仅取Train四分位数0.0458868，early DAS≤47。
core-near：GT叶距离GT core的p10≤5mm，是可复核的评价代理，不是人工确认的“芯部未分离”标签。不存在core的帧距离未知，不删除叶片。

{table(strata_table)}

`segmentation_leaf_errors.csv`保留每一个合法叶的漏检、分裂、合并、采样丢失、proxy标签及IoU；不凭预测困难删除它们。`error_strata_summary.csv`报告各层植物级CI、实际独立植物数和叶观测数，观测数不是独立重复数。

## 5. 固定TrackPlant3D传播验证

每帧预测实例独立随机重新编号后才进入追踪器。保留M6-A背景0匹配、256点3DEPS、CPD及官方扩展匹配代价，不重训、不重调追踪参数。
重建同一XYZ采样索引、核验与M6-A缓存normalized XYZ逐点精确相等，复用label-independent CPD变形。
GT时序ID只在评价器中作IoU关联/IDF1全局对齐，不输入追踪器。

{table(tracking_table)}

`tracking_summary.csv`给植物CI，`tracking_paired_intervals.csv`给B3/B4相对B0/B1配对CI。
`tracking_error_decomposition.csv`独立统计 instance_missed、sampling_or_identity_lost、wrong_identity及correct。
真实首次观测为GT器官首个实际扫描，事件必须同帧IoU关联到新器官；新预测track不是生物学死亡/出生真值。
`new_missed_instance`和`new_attached_old_identity`明确区分没检测到与已检测但接成旧身份。完整observations、new_track_events和predicted_organs保存在各run中。
`first_observation_events.csv`逐新叶记录首次帧面积点占比、当前帧GT局部标签对应、small/early/core邻近分层及检测延迟；`first_observation_strata_summary.csv`给植株级区间。事件分层不改变原始严格同帧首观测F1。

{table(event_diagnostic_table)}

更高PQ不会保证所有终端指标都更高：Validation预先按PQ选择B3，不按追踪/测量后验改选。B3和B4的覆盖率、continuity及严重误差仍有差异，完整表及逐株配对结果保留；不能把全链路收益仅归因于分割PQ一个数。

测量完全复用原函数、缺失损失2及严重动态误差规则，覆盖率含全部适格GT转移。
同测量器减少的是缺失/错连传播，不会自动修正绝对叶长/面积。GT对照仍显著长度偏差，平面凸包面积非人工叶面积真值；**不宣布表型测量系统可靠**。

## 6. PSegNet / Organ3DNet及边界

S3是原PSegNet监督从头基线，真实训练60轮/一个固定种子；只有一个种子，不声称与S1/S2相同随机稳定性证据。
S4官方commit `{block['official_commit']}`；原Python3.10/Torch1.12.1+cu113隔离环境实际安装；官方PointNet2和固定ME扩展均在CUDA12.6/11.3不匹配处失败。
官方checkpoint {block['checkpoint']['bytes']} bytes下载且MD5核验，但**未严格加载官方模型、未跑官方示例、目标监督步数0**；无S4 PQ、无简化网络、停止移植。
repo代码无明确许可证，checkpoint Zenodo记录CC BY4.0单独注明；本地上游代码/源及衍生权重不公开再分发。

## 7. 继续标准和唯一建议

冻结Validation门槛PQ≥.70、IDF1≥.80、新叶首观测F1≥.70；分割提升代表seed和seed-mean均≥9/12株同方向且配对CI下界>0；早期小叶召回≥.50。
不要求当前偏置测量器全指标达标。Test结果不用于选择或最终判定。

{table(pd.DataFrame(decision['validation_gates']))}

唯一建议 **{decision['recommendation']}**。{decision['rationale']}
错误归因附加诊断（不改成功门槛）：{decision['instance_bottleneck_diagnostic']}。若仅追踪失败而分割/早期小叶达标，不能直接判为需要新分割器；若首观测缺失以当前帧实例漏检为主，则是公开的实例瓶颈证据。芯部邻近和首观测分层是GT评价代理，仍需更多独立植物/外部数据检验泛化。
若证据不足，不继续Test驱动调参；补独立植株/跨来源数据或更严格外部验证。若实例错误仍需方法研究，只由公开的失败类型推动，未自动开发新算法。

## 8. 工程交付、保护及复现

协议/配置、源码/标签审计、训练日志/权重/全分辨率预测、按植株区间、错误分解和真实点云案例均实际存在。
`figures/`中case按Validation B3最差/最好植株固定选择，非手选成功；包含early/late同一植物GT/S0/B3对照及指标/训练曲线/检测-身份错误。
`REPRODUCE.md`给命令，`tests_m6_ab_final.log`保留35项当前/历史接口回归通过；初始完整套件175通过/1失败是M6-B隔离环境缺少历史RGB torchvision，同一历史测试在原`.venv-m1`通过，不修改旧环境或代码。`protection_verification.json`检查历史文件/环境fingerprint、tracked SHA256和891原始文件SHA256，HEAD必须保持原样。
无commit、merge、push，M0–M6-A不改；本阶段完成后停止，不自动开发方法。

## 9. 信息来源

Bocha一般检索不可用（403配额不足），公开方法信息来自实际取得的官方源码/HTTP记录；本地结果来自真实训练/评价，不基于训练数据记忆填数。
- Sugar4D: https://www.ipb.uni-bonn.de/data/sugar4d/index.html ; https://doi.org/10.60507/FK2/IS8YBZ
- PlantNet/PSegNet: https://github.com/Huang2002200/PlantNet-and-PSegNet
- Organ3DNet: https://github.com/Huang2002200/Organ3DNet ; https://doi.org/10.1016/j.aiia.2025.10.011 ; https://zenodo.org/records/17446444
- TrackPlant3D: https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- PyTorch/ME官方文档由Context7实际获取，见`sources/ctx7_*`；全部具体固定commit/许可证/依赖在`SOURCE_AUDIT.md`。
"""
    (report / "M6_B_REPORT.md").write_text(report_text)
    (report / "DELIVERY_INDEX.md").write_text("""# M6-B delivery index

- `M6_B_REPORT.md` and `decision.json`: actual supervised results and one research recommendation.
- `protocol_lock.json`, `configs/m6_b/protocol.json`: pretraining frozen metrics/budgets/selection/continuation gates.
- `SOURCE_AUDIT.md`, `sources/`, `organ3d_feasibility.json`: fixed official revisions, licensing, native blocker and checkpoint hashes.
- `DATA_AUDIT.md`, `data_audit.json`, `data_frames.csv`, `data_leaves.csv`: complete768-frame labels, splits, small/proximity proxies and physical metres.
- `training_summary.csv`, `validation_learning_curves.csv`: all seven actual60-epoch runs and loading/budget/convergence evidence.
- `segmentation_summary.csv`, `segmentation_per_plant.csv`, `segmentation_frame_metrics.csv`, `segmentation_leaf_errors.csv`: full-resolution macro scores and every legal leaf's errors.
- `paired_plant_intervals.csv`, `seed_mean_plant_pq.csv`, `error_strata_summary.csv`, `pq_by_das.csv`, `pq_by_leaf_count.csv`: plant bootstrap, seed handling and registered strata.
- `tracking_per_plant.csv`, `tracking_summary.csv`, `tracking_paired_intervals.csv`, `tracking_error_decomposition.csv`: frozen tracker and identical measurer propagation.
- `first_observation_events.csv`, `first_observation_strata_per_plant.csv`, `first_observation_strata_summary.csv`: every legal new-leaf event, detection-vs-identity loss and evaluation-only size/core strata.
- `tracking_geometry_equivalence.json`, `tracking_replay_equivalence.json`: exact384-frame XYZ/sample-label agreement and24-plant historical S0 numerical replay.
- `inference_runtime.csv`, `inference_microbenchmark.csv`, `inference_repeatability.json`: actual shared-host timing/VRAM; same-device read-only repeat check when executed.
- `figures/`, `FIGURE_CONTRACT.md`, `FIGURE_QA.md`, `figure_manifest.json`, `figure_source_preflight.json`: source-grounded PNG/SVG/PDF and direct-image QA.
- `TESTING.md`, test/build logs, `protection_verification.json`, `final_verification.json`: focused/historical regression, unchanged files/environments/HEAD and completion checks.
- `REPRODUCE.md`, `RESOURCE_SCHEDULING.md`, `scripts/*m6_b*`, `src/m6b/`, `tests/test_m6_b.py`: actual reproduction and scheduling/lock rules.

Large/local-only outputs: `outputs/m6_b/S*/seed_*/` contains `training_manifest.json`, every-update `batches.csv`, per-epoch `epochs.jsonl`, latest/best/evaluation-epoch checkpoints, full-resolution predictionNPZ, postprocessing selection and full observations/events/measurement errors. Every legal point is included in scored full-resolution labels; displayed subsets do not replace them.
`outputs/m6_b/data/` holds current-frame sample indices/local supervision and reversible centre/scale; `tracker_geometry/` holds384 geometry-only indices.
Upstream source/checkpoints remain ignored under `data/raw/m6_a` / `data/raw/m6_b`, and derived weights/predictions remain ignored under `outputs/m6_b`. No redistribution, commit, branch, merge or push.
""")
    delivery_manifest(root)
    print("DELIVERED", decision["recommendation"], flush=True)


def delivery_manifest(root):
    report = root / "reports/m6_b"
    decision = json.loads((report / "decision.json").read_text())
    training = pd.read_csv(report / "training_summary.csv")
    manifest = []
    for folder in [report, root / "outputs/m6_b"]:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.name != "delivery_manifest.json":
                manifest.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)})
    implementations = list((root / "src/m6b").glob("*.py")) + list((root / "configs/m6_b").glob("*")) + list((root / "scripts").glob("*m6_b*")) + [root / "tests/test_m6_b.py"]
    for path in sorted(implementations):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)})
    write_json(report / "delivery_manifest.json", {"recommendation": decision["recommendation"], "files": manifest,
               "weights_predictions_local_only": True, "full_resolution_predictions": True, "supervised_runs": len(training),
               "S4": "BLOCKED", "historical_results_unchanged": "see protection_verification.json"})
