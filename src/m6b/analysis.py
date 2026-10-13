from __future__ import annotations

import json

import numpy as np
import pandas as pd

from m6a.common import digest, write_json
from m6a.experiment import cache_path, metadata
from m6a.metrics import frame_metrics
from m6b.data import read_frame
from m6b.evaluation import prediction_path, run_dir


def interval(values, seed=20261013):
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": None, "ci95_low": None, "ci95_high": None, "plants": 0}
    generator = np.random.default_rng(seed)
    resampled = array[generator.integers(0, len(array), size=(10000, len(array)))].mean(axis=1)
    return {"mean": float(array.mean()), "ci95_low": float(np.quantile(resampled, .025)),
            "ci95_high": float(np.quantile(resampled, .975)), "plants": len(array)}


def run_inventory(root):
    inventory = [("S0", 0)]
    for method in ["S1", "S2", "S3"]:
        for seed in ([17] if method == "S3" else [17, 29, 43]):
            if (run_dir(root, method, seed) / "inference_val.json").exists():
                inventory.append((method, seed))
    return inventory


def prediction_errors(root):
    records = metadata(root)
    leaves = pd.read_csv(root / "reports/m6_b/data_leaves.csv")
    audit = json.loads((root / "reports/m6_b/data_audit.json").read_text())
    small_threshold = audit["small_leaf_fraction_threshold_train_q25"]
    scores, errors = [], []
    inventory = run_inventory(root)
    report = root / "reports/m6_b"
    for method, seed in inventory:
        for split in ["val", "test"]:
            if method != "S0" and not (run_dir(root, method, seed) / f"inference_{split}.json").exists():
                continue
            for record in records.query("split == @split").itertuples(index=False):
                data = read_frame(root / record.pointcloud_path)
                gt = np.where(data["local_instance"] >= 0, data["local_instance"] + 1, 0).astype(np.int32)
                source = cache_path(root, record.plant_uid, record.DAS) if method == "S0" else prediction_path(root, method, seed, record.plant_uid, record.DAS)
                with np.load(source) as prediction:
                    predicted = prediction["labels"]
                    sample_indices = prediction["sample_indices"]
                    sample_leaf = predicted[sample_indices] > 0
                score, matches = frame_metrics(gt, predicted)
                gt_sizes = np.bincount(gt, minlength=int(gt.max()) + 1)
                score.update({"method": method, "seed": seed, "split": split, "plant_uid": record.plant_uid, "DAS": int(record.DAS),
                              "stage": "early" if record.DAS <= 47 else "middle" if record.DAS <= 65 else "late",
                              "leaf_count_bin": "<=6" if score["gt_count"] <= 6 else "7-10" if score["gt_count"] <= 10 else ">10"})
                scores.append(score)
                frame_leaves = leaves[(leaves.plant_uid == record.plant_uid) & (leaves.DAS == record.DAS)]
                for leaf in frame_leaves.itertuples(index=False):
                    gt_id = int(leaf.local_id) + 1
                    mask = gt == gt_id
                    predicted_ids, intersections = np.unique(predicted[mask], return_counts=True)
                    part = intersections / mask.sum()
                    is_split = int(np.sum(part[predicted_ids > 0] > .1) > 1)
                    merged = False
                    for predicted_id in predicted_ids[(predicted_ids > 0) & (part > .1)]:
                        overlapping_ids, overlap = np.unique(gt[predicted == predicted_id], return_counts=True)
                        significant = overlapping_ids[(overlapping_ids > 0) & (overlap / np.maximum(gt_sizes[overlapping_ids], 1) > .1)]
                        merged = merged or len(significant) > 1
                    localized = data["local_instance"][sample_indices] == leaf.local_id
                    errors.append({"method": method, "seed": seed, "plant_uid": record.plant_uid, "split": split, "DAS": int(record.DAS),
                                   "gt_local_id": gt_id, "leaf_points": int(leaf.points), "fraction": float(leaf.point_fraction),
                                   "detected": gt_id in matches, "iou": matches.get(gt_id, (-1, 0))[1], "split_error": is_split,
                                   "small": bool(leaf.point_fraction <= small_threshold), "early": bool(record.DAS <= 47),
                                   "merge_error": merged,
                                   "core_adjacent": bool(np.isfinite(leaf.core_distance_p10_m) and leaf.core_distance_p10_m <= .005),
                                   "sampling_loss": not localized.any(),
                                   "sampled_semantic_leaf_recall": float(sample_leaf[localized].mean()) if localized.any() else None,
                                   "core_distance_p10_m": float(leaf.core_distance_p10_m) if np.isfinite(leaf.core_distance_p10_m) else None})
        print("ERROR AUDIT", method, seed, flush=True)
    frames = pd.DataFrame(scores)
    leaf_errors = pd.DataFrame(errors)
    frames.to_csv(report / "segmentation_frame_metrics.csv", index=False)
    leaf_errors.to_csv(report / "segmentation_leaf_errors.csv", index=False)
    grouped = frames.groupby(["method", "seed", "split", "plant_uid"])
    plants = grouped.agg(pq=("pq", "mean"), sbd=("sbd", "mean"), tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
                         gt_instances=("gt_count", "sum"), predicted_instances=("pred_count", "sum"),
                         merged_predictions=("merge_instances", "sum"), split_gt=("split_gt_instances", "sum")).reset_index()
    plants["precision"] = plants.tp / (plants.tp + plants.fp).clip(lower=1)
    plants["recall"] = plants.tp / plants.gt_instances.clip(lower=1)
    plants["miss_rate"] = 1 - plants.recall
    plants["merge_rate"] = plants.merged_predictions / plants.predicted_instances.clip(lower=1)
    plants["split_rate"] = plants.split_gt / plants.gt_instances.clip(lower=1)
    plants.to_csv(report / "segmentation_per_plant.csv", index=False)
    intervals = []
    for (method, seed, split), group in plants.groupby(["method", "seed", "split"]):
        for metric in ["pq", "sbd", "precision", "recall", "miss_rate", "merge_rate", "split_rate"]:
            intervals.append({"method": method, "seed": seed, "split": split, "metric": metric, **interval(group[metric])})
    pd.DataFrame(intervals).to_csv(report / "segmentation_summary.csv", index=False)
    strata = []
    for category in ["small", "early", "core_adjacent"]:
        selected = leaf_errors[leaf_errors[category]]
        group = selected.groupby(["method", "seed", "split", "plant_uid"]).agg(recall=("detected", "mean"),
                split_rate=("split_error", "mean"), merge_rate=("merge_error", "mean"), sampling_loss=("sampling_loss", "mean"), leaves=("detected", "size")).reset_index()
        group["category"] = category
        strata.append(group)
    selected = leaf_errors[leaf_errors.small & leaf_errors.early]
    group = selected.groupby(["method", "seed", "split", "plant_uid"]).agg(recall=("detected", "mean"),
            split_rate=("split_error", "mean"), merge_rate=("merge_error", "mean"), sampling_loss=("sampling_loss", "mean"), leaves=("detected", "size")).reset_index()
    group["category"] = "early_small"
    strata.append(group)
    error_strata = pd.concat(strata, ignore_index=True)
    error_strata.to_csv(report / "error_strata_per_plant.csv", index=False)
    stratum_summary = []
    for (method, seed, split, category), group in error_strata.groupby(["method", "seed", "split", "category"]):
        for metric in ["recall", "split_rate", "merge_rate", "sampling_loss"]:
            stratum_summary.append({"method": method, "seed": seed, "split": split, "category": category, "metric": metric,
                                    "leaf_observations": int(group.leaves.sum()), **interval(group[metric])})
    pd.DataFrame(stratum_summary).to_csv(report / "error_strata_summary.csv", index=False)
    for column, filename in [("DAS", "pq_by_das.csv"), ("leaf_count_bin", "pq_by_leaf_count.csv")]:
        stratified = frames.groupby(["method", "seed", "split", "plant_uid", column])[["pq", "sbd", "fn", "merge_instances", "split_gt_instances"]].mean().reset_index()
        stratified.groupby(["method", "seed", "split", column]).agg(pq=("pq", "mean"), sbd=("sbd", "mean"),
             independent_plants=("plant_uid", "nunique"), fn_per_scan=("fn", "mean"),
             merge_per_scan=("merge_instances", "mean"), split_per_scan=("split_gt_instances", "mean")).reset_index().to_csv(report / filename, index=False)
    return plants


def paired_statistics(root, plants):
    rows = []
    metrics = ["pq", "sbd", "recall", "merge_rate", "split_rate"]
    baseline = plants[plants.method == "S0"].set_index(["split", "plant_uid"])
    for (method, seed, split), group in plants[plants.method != "S0"].groupby(["method", "seed", "split"]):
        for metric in metrics:
            paired = group.set_index("plant_uid")
            differences = paired[metric] - baseline.loc[split, metric].reindex(paired.index)
            rows.append({"method": method, "seed": seed, "split": split, "comparison": method + "-S0", "metric": metric,
                         **interval(differences), "improved_plants": int(np.sum(differences > 0))})
    seed_mean = plants.groupby(["method", "split", "plant_uid"])[metrics].mean().reset_index()
    for (method, split), group in seed_mean[seed_mean.method.isin(["S1", "S2"])].groupby(["method", "split"]):
        for metric in metrics:
            paired = group.set_index("plant_uid")
            differences = paired[metric] - baseline.loc[split, metric].reindex(paired.index)
            rows.append({"method": method, "seed": "mean_seeds", "split": split, "comparison": method + "-S0", "metric": metric,
                         **interval(differences), "improved_plants": int(np.sum(differences > 0))})
    pair = seed_mean[seed_mean.method.isin(["S1", "S2"])].pivot(index=["split", "plant_uid"], columns="method", values="pq")
    if {"S1", "S2"}.issubset(pair.columns):
        for split, group in pair.groupby(level="split"):
            differences = group.S2 - group.S1
            rows.append({"method": "S2", "seed": "mean_seeds", "split": split, "comparison": "S2-S1", "metric": "pq",
                         **interval(differences), "improved_plants": int(np.sum(differences > 0))})
    pd.DataFrame(rows).to_csv(root / "reports/m6_b/paired_plant_intervals.csv", index=False)
    return pd.DataFrame(rows)


def select_downstream(root):
    scores = []
    for method in ["S1", "S2"]:
        for seed in [17, 29, 43]:
            manifest = json.loads((run_dir(root, method, seed) / "inference_val.json").read_text())
            scores.append({"method": method, "seed": seed, "pq": manifest["pq"]})
    family = pd.DataFrame(scores).groupby("method").pq.mean().sort_values(ascending=False)
    selected = {"B3_method": family.index[0], "B3_seed": 17, "B4_method": "S3", "B4_seed": 17,
                "selection": "mean over three Validation seeds; downstream seed17 fixed before outcomes", "family_validation_pq": family.to_dict(),
                "no_test_selection": True}
    write_json(root / "reports/m6_b/downstream_selection.json", selected)
    return selected
