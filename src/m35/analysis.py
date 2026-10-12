"""Plant-level evaluation, validation-locked decisions and final test replay."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from datasets.sugar4d_organ_dataset import ORGAN_FEATURES, ORGAN_TARGETS, PLANT_TARGETS, Sugar4DCurrentDataset
from m3.common import file_hash, tensor_state_hash, write_json
from m35.data import GEOMETRY_FEATURES, verify_protection
from m35.engine import balanced_weights, base_prediction_frame, current_arrays, current_inference, decode_tabular, inverse, logistic_predict, ridge_predict, tabular_arrays, tabular_inference, tabular_model, tabular_prediction_frame, transform
from models.geometry_residual_forecaster import CurrentPointNetRegressor
from models.organ_growth_baselines import linear_growth, persistence


def bootstrap_mean(values, protocol):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        return None, None
    indices = np.random.default_rng(protocol["bootstrap_seed"]).integers(0, len(values), (protocol["bootstrap_replicates"], len(values)))
    low, high = np.quantile(values[indices].mean(axis=1), [0.025, 0.975])
    return float(low), float(high)


def condition(model):
    if model == "A2-DAS-Ridge":
        return "DAS-only control; no geometry input"
    if model == "E0-Frequency":
        return "Train event/count frequency; no current-state input"
    if model.startswith(("C0", "C1", "C2", "C3", "C4")):
        return "Oracle historical organ phenotype; historical identity association"
    if model.startswith("E"):
        return "Oracle historical annotation/organ-state; first observed, not birth"
    return "Non-Oracle XYZ geometry; no phenotype or instance inputs"


def morphology_scores(predictions, targets, protocol):
    records = []
    dimensions = ["partition", "split", "task", "model", "seed", "horizon", "plant_uid"]
    for keys, group in predictions.groupby(dimensions, sort=True):
        metadata = dict(zip(dimensions, keys))
        for target in targets:
            valid = group[[f"pred_{target}", f"true_{target}"]].notna().all(axis=1)
            subset = group.loc[valid]
            if subset.empty:
                continue
            error = subset[f"pred_{target}"].to_numpy() - subset[f"true_{target}"].to_numpy()
            records.append({**metadata, "target": target, "windows": len(subset), "eligible_windows": len(group),
                            "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.square(error).mean())),
                            "delta_mae": float((subset[f"pred_delta_{target}"] - subset[f"true_delta_{target}"]).abs().mean()) if f"pred_delta_{target}" in subset else np.nan})
    plants = pd.DataFrame(records)
    summaries, seed_rows = [], []
    for keys, group in plants.groupby(["partition", "split", "task", "model", "horizon", "target"]):
        metadata = dict(zip(("partition", "split", "task", "model", "horizon", "target"), keys))
        seeds = group.groupby("seed").mae.mean()
        for seed, value in seeds.items():
            seed_rows.append({**metadata, "seed": int(seed), "plant_macro_mae": float(value)})
        independent = group.groupby("plant_uid").mae.mean()
        low, high = bootstrap_mean(independent, protocol)
        summaries.append({**metadata, "condition": condition(metadata["model"]), "plants": len(independent), "seeds": len(seeds),
                          "plant_macro_mae": float(independent.mean()), "seed_mae_std": float(seeds.std(ddof=0)),
                          "plant_macro_rmse": float(group.groupby("plant_uid").rmse.mean().mean()),
                          "delta_phenotype_mae": float(group.delta_mae.mean()) if group.delta_mae.notna().any() else np.nan,
                          "ci95_low": low, "ci95_high": high, "unit": "cm2" if metadata["target"] == "leaf_area" else ("leaves" if metadata["target"] == "visible_organ_count" else "cm")})
    return pd.DataFrame(summaries), plants, pd.DataFrame(seed_rows)


def paired_scores(plants, pairs, protocol):
    records = []
    for keys, group in plants.groupby(["partition", "split", "task", "horizon", "target"]):
        metadata = dict(zip(("partition", "split", "task", "horizon", "target"), keys))
        independent = group.groupby(["model", "plant_uid"]).mae.mean().unstack("model")
        for left, right in pairs:
            if left not in independent or right not in independent:
                continue
            subset = independent[[left, right]].dropna()
            difference = subset[left] - subset[right]
            low, high = bootstrap_mean(difference, protocol)
            records.append({**metadata, "left": left, "right": right, "plants": len(subset), "left_minus_right_mae": float(difference.mean()),
                            "improvement_pct": float(-100 * difference.mean() / subset[right].mean()), "ci95_low": low, "ci95_high": high,
                            "aggregation": "per-seed error averaged within plant; paired plant bootstrap, not ensemble or record bootstrap"})
    return pd.DataFrame(records)


def current_diagnostics(root, protocol):
    root = Path(root)
    frames = [pd.read_csv(path) for path in sorted((root / "outputs/m3_5").glob("*/current/*train_val.csv"))]
    if not frames:
        raise ValueError("No current-state predictions")
    predictions = pd.concat(frames, ignore_index=True)
    for partition, assignment in json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"].items():
        rows = current_arrays(root, assignment)
        rows = rows.loc[rows.split.ne("test")]
        for name, field in (("A2-Z-Range-Proxy", "z_range_cm"), ("A2-Z-Quantile-Proxy", "height_quantile_proxy_cm"), ("A2-Hull-Proxy", "hull_diameter_cm")):
            target = "plant_width" if field == "hull_diameter_cm" else "plant_height"
            result = base_prediction_frame(rows, name, -1, partition, "current")
            for target_name in PLANT_TARGETS:
                result[f"true_{target_name}"] = rows[target_name].to_numpy()
                result[f"pred_{target_name}"] = rows[field].to_numpy() if target_name == target else np.nan
            predictions = pd.concat([predictions, result], ignore_index=True)
    summary, plants, seeds = morphology_scores(predictions, PLANT_TARGETS, protocol)
    report = root / "reports/m3_5"
    summary.to_csv(report / "current_state_diagnostics.csv", index=False)
    plants.to_csv(report / "current_state_plant_errors.csv", index=False)
    seeds.to_csv(report / "current_state_seed_metrics.csv", index=False)
    pairs = paired_scores(plants, [("A1-PointNet", "A2-Geometry-Ridge"), ("A3-PointNet-Geometry", "A1-PointNet")], protocol)
    pairs.to_csv(report / "current_state_paired.csv", index=False)
    return summary, predictions


def research_decision(current, organ_summary, organ_pairs, seeds, protocol):
    thresholds = protocol["current_state_reliability"]
    official = current.loc[current.partition.eq("official") & current.split.eq("val")]
    representation = []
    for target, threshold in (("plant_height", thresholds["height_mae_cm"]), ("plant_width", thresholds["width_mae_cm"])):
        subset = official.loc[official.target.eq(target)].set_index("model")
        pn, geometry = float(subset.loc["A1-PointNet", "plant_macro_mae"]), float(subset.loc["A2-Geometry-Ridge", "plant_macro_mae"])
        representation.append({"target": target, "pointnet_validation_mae": pn, "geometry_validation_mae": geometry,
                               "threshold_cm": threshold, "ratio": pn / geometry,
                               "unreliable": pn > threshold and pn >= thresholds["pointnet_to_geometry_mae_ratio"] * geometry})
    task = protocol["primary_task"]
    rule = protocol["temporal_signal_rule"]
    controls = []
    if not organ_pairs.empty:
        for control in rule["controls"]:
            row = organ_pairs.loc[organ_pairs.partition.eq("official") & organ_pairs.split.eq("val") & organ_pairs.target.eq(task["target"]) & organ_pairs.horizon.eq(task["horizon"]) & organ_pairs.left.eq("C3-GRU") & organ_pairs.right.eq(control)].iloc[0]
            subset = seeds.loc[seeds.partition.eq("official") & seeds.split.eq("val") & seeds.target.eq(task["target"]) & seeds.horizon.eq(task["horizon"])].pivot(index="seed", columns="model", values="plant_macro_mae")
            subset = subset.dropna(subset=["C3-GRU", control])
            all_seeds = bool((subset["C3-GRU"] < subset[control]).all()) and len(subset) == 3
            improving = 0
            folds = []
            for fold in ("fold0", "fold1", "fold2", "fold3"):
                pair = organ_pairs.loc[organ_pairs.partition.eq(fold) & organ_pairs.split.eq("val") & organ_pairs.target.eq(task["target"]) & organ_pairs.horizon.eq(task["horizon"]) & organ_pairs.left.eq("C3-GRU") & organ_pairs.right.eq(control)].iloc[0]
                improving += int(pair.left_minus_right_mae < 0)
                folds.append({"fold": fold, "validation_difference_cm": float(pair.left_minus_right_mae)})
            passed = bool(row.improvement_pct >= 100 * rule["minimum_relative_improvement"] and -row.left_minus_right_mae >= rule["minimum_absolute_improvement_cm"] and row.ci95_high < 0 and all_seeds and improving >= rule["minimum_directionally_improving_validation_folds"])
            controls.append({"control": control, "validation_improvement_pct": float(row.improvement_pct), "validation_absolute_improvement_cm": float(-row.left_minus_right_mae),
                             "validation_difference_ci95": [float(row.ci95_low), float(row.ci95_high)], "all_three_seeds_improve": all_seeds,
                             "improving_validation_folds": improving, "folds": folds, "passed": passed})
    signal = len(controls) == 2 and all(control["passed"] for control in controls)
    recommendation = "REPRESENTATION-FIRST" if any(item["unreliable"] for item in representation) else ("ORGAN-TEMPORAL-JEPA" if signal else "PIVOT")
    return {"recommendation": recommendation, "current_representation": representation, "primary_task": task, "organ_history_controls": controls,
            "reliable_practical_organ_temporal_signal": signal, "selection_evidence": "official validation plus locked four-fold validation checks only",
            "test_is_exploratory": True, "fold_validation_not_external_confirmation": True, "start_m4_automatically": False,
            "rule": protocol["decision_priority"], "area_alone_cannot_justify_continuation": True}


def classification_metrics(frame, label, probability, threshold, protocol):
    truth = frame[label].astype(int).to_numpy()
    score = np.clip(frame[probability].to_numpy(), 1e-7, 1 - 1e-7)
    prediction = (score >= threshold).astype(int)
    weights = balanced_weights(frame)
    both_classes = len(np.unique(truth)) == 2
    plant_rows = []
    for plant, positions in frame.groupby("plant_uid").indices.items():
        positions = np.asarray(positions)
        actual, predicted, continuous = truth[positions], prediction[positions], score[positions]
        plant_rows.append({"plant_uid": plant, "precision": float(precision_score(actual, predicted, zero_division=0)),
                           "recall": float(recall_score(actual, predicted, zero_division=0)), "f1": float(f1_score(actual, predicted, zero_division=0)),
                           "brier": float(np.square(continuous - actual).mean()),
                           "pr_auc": float(average_precision_score(actual, continuous)) if len(np.unique(actual)) == 2 else np.nan,
                           "positive": int(actual.sum()), "negative": int((1 - actual).sum())})
    plants = pd.DataFrame(plant_rows)
    output = {"plants": len(plants), "samples": len(frame), "positive": int(truth.sum()), "negative": int((1 - truth).sum()), "threshold": threshold,
              "positive_plants": int((plants.positive > 0).sum()), "negative_plants": int((plants.negative > 0).sum()),
              "plant_macro_precision": float(plants.precision.mean()), "plant_macro_recall": float(plants.recall.mean()),
              "plant_macro_f1": float(plants.f1.mean()), "plant_macro_brier": float(plants.brier.mean()),
              "plant_balanced_pr_auc": float(average_precision_score(truth, score, sample_weight=weights)) if both_classes else np.nan,
              "plant_macro_pr_auc_eligible": float(plants.pr_auc.mean()) if plants.pr_auc.notna().any() else np.nan,
              "pr_auc_eligible_plants": int(plants.pr_auc.notna().sum()), "pooled_plant_weighted_precision": float(precision_score(truth, prediction, sample_weight=weights, zero_division=0)),
              "pooled_plant_weighted_recall": float(recall_score(truth, prediction, sample_weight=weights, zero_division=0)),
              "pooled_plant_weighted_f1": float(f1_score(truth, prediction, sample_weight=weights, zero_division=0))}
    output["f1_ci95_low"], output["f1_ci95_high"] = bootstrap_mean(plants.f1, protocol)
    if "pred_new_count" in frame:
        errors = (frame.pred_new_count - frame.new_count).abs()
        output["new_count_plant_macro_mae"] = float(errors.groupby(frame.plant_uid).mean().mean())
    return output, plants


def cluster_pr_auc_interval(frame, label, probability, protocol):
    contributions = []
    for _, group in frame.groupby("seed"):
        scores = np.clip(group[probability].to_numpy(dtype=float), 0, 1)
        labels = group[label].to_numpy(dtype=int)
        score_values = np.unique(scores)[::-1]
        positions = np.searchsorted(-score_values, -scores)
        plants = sorted(group.plant_uid.unique())
        positive = np.zeros((len(plants), len(score_values)))
        negative = np.zeros_like(positive)
        for plant_index, plant in enumerate(plants):
            mask = group.plant_uid.eq(plant).to_numpy()
            per_record_weight = 1 / mask.sum()
            np.add.at(positive[plant_index], positions[mask], labels[mask] * per_record_weight)
            np.add.at(negative[plant_index], positions[mask], (1 - labels[mask]) * per_record_weight)
        contributions.append((plants, positive, negative))
    if not contributions or any(item[0] != contributions[0][0] for item in contributions):
        raise ValueError("Seed plants do not match for PR-AUC interval")
    plant_count = len(contributions[0][0])
    draws = np.random.default_rng(protocol["bootstrap_seed"]).integers(0, plant_count, (protocol["bootstrap_replicates"], plant_count))
    multiplicities = np.stack([np.bincount(draw, minlength=plant_count) for draw in draws])
    values = []
    for start in range(0, len(multiplicities), 256):
        weights = multiplicities[start:start + 256]
        by_seed = []
        for _, positive, negative in contributions:
            positive_counts, negative_counts = weights @ positive, weights @ negative
            total_positive, total_negative = positive_counts.sum(axis=1), negative_counts.sum(axis=1)
            cumulative_positive = np.cumsum(positive_counts, axis=1)
            cumulative_all = np.cumsum(positive_counts + negative_counts, axis=1)
            precision = np.divide(cumulative_positive, cumulative_all, out=np.zeros_like(cumulative_positive), where=cumulative_all > 0)
            ap = np.divide((positive_counts * precision).sum(axis=1), total_positive, out=np.full(len(weights), np.nan), where=(total_positive > 0) & (total_negative > 0))
            by_seed.append(ap)
        values.extend(np.mean(np.stack(by_seed), axis=0).tolist())
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return None, None, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), len(finite)


def select_thresholds(predictions, probability, label, protocol):
    records = {}
    for (partition, model, seed, horizon), group in predictions.loc[predictions.split.eq("val")].groupby(["partition", "model", "seed", "horizon"]):
        best = None
        for threshold in protocol["event_thresholds"]:
            truth, pred = group[label].astype(int).to_numpy(), (group[probability].to_numpy() >= threshold).astype(int)
            score = f1_score(truth, pred, sample_weight=balanced_weights(group), zero_division=0)
            if best is None or score > best[0] or (score == best[0] and abs(threshold - 0.5) < abs(best[1] - 0.5)):
                best = (float(score), threshold)
        records[f"{partition}|{model}|{int(seed)}|{int(horizon)}"] = {"threshold": best[1], "validation_plant_weighted_f1": best[0], "selection_split": "val"}
    return records


def summarize_classification(predictions, probability, label, thresholds, protocol):
    records, plant_records = [], []
    for keys, group in predictions.groupby(["partition", "split", "model", "seed", "horizon"]):
        partition, split, model, seed, horizon = keys
        key = f"{partition}|{model}|{int(seed)}|{int(horizon)}"
        metrics, plants = classification_metrics(group.reset_index(drop=True), label, probability, thresholds[key]["threshold"], protocol)
        records.append({"partition": partition, "split": split, "model": model, "seed": seed, "horizon": horizon, "condition": condition(model), **metrics})
        plants = plants.assign(partition=partition, split=split, model=model, seed=seed, horizon=horizon)
        if "pred_new_count" in group:
            group = group.reset_index(drop=True)
            errors = (group.pred_new_count - group.new_count).abs().groupby(group.plant_uid).mean()
            plants["count_mae"] = plants.plant_uid.map(errors)
        plant_records.append(plants)
    per_seed = pd.DataFrame(records)
    metadata = ["partition", "split", "model", "horizon", "condition"]
    averaged = per_seed.groupby(metadata).mean(numeric_only=True).reset_index().drop(columns="seed")
    all_plants = pd.concat(plant_records, ignore_index=True)
    for index, row in averaged.iterrows():
        subset = all_plants.loc[all_plants.partition.eq(row.partition) & all_plants.split.eq(row.split) & all_plants.model.eq(row.model) & all_plants.horizon.eq(row.horizon)]
        independent = subset.groupby("plant_uid").f1.mean()
        averaged.loc[index, "f1_ci95_low"], averaged.loc[index, "f1_ci95_high"] = bootstrap_mean(independent, protocol)
        averaged.loc[index, "seeds"] = subset.seed.nunique()
        averaged.loc[index, "f1_seed_std"] = subset.groupby("seed").f1.mean().std(ddof=0)
        prediction_subset = predictions.loc[predictions.partition.eq(row.partition) & predictions.split.eq(row.split) & predictions.model.eq(row.model) & predictions.horizon.eq(row.horizon)]
        if row.split in ("val", "test") and row.partition == "official":
            low, high, valid_draws = cluster_pr_auc_interval(prediction_subset, label, probability, protocol)
            averaged.loc[index, "pr_auc_cluster_ci95_low"], averaged.loc[index, "pr_auc_cluster_ci95_high"] = low, high
            averaged.loc[index, "pr_auc_valid_bootstrap_draws"] = valid_draws
        if "count_mae" in subset:
            count_values = subset.groupby("plant_uid").count_mae.mean()
            averaged.loc[index, "new_count_ci95_low"], averaged.loc[index, "new_count_ci95_high"] = bootstrap_mean(count_values, protocol)
    averaged["aggregation"] = "Metrics computed per seed using plant units, then averaged over seeds; no forecast ensemble"
    return averaged, per_seed, all_plants


def collect_train_validation(root, task):
    paths = sorted((Path(root) / "outputs/m3_5").glob(f"*/{task}/*train_val.csv"))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True) if paths else pd.DataFrame()


def lock_selection(root, protocol):
    root = Path(root)
    current, current_predictions = current_diagnostics(root, protocol)
    organ, event = collect_train_validation(root, "organ"), collect_train_validation(root, "event")
    summary, plants, seeds = morphology_scores(organ, ORGAN_TARGETS, protocol)
    pairs = paired_scores(plants, [("C3-GRU", "C2-MLP"), ("C3-GRU", "C4-Repeated-GRU"), ("C3-GRU", "C0-Persistence"), ("C1-Linear-Growth", "C0-Persistence")], protocol)
    report = root / "reports/m3_5"
    summary.to_csv(report / "organ_validation_metrics.csv", index=False)
    pairs.to_csv(report / "organ_validation_pairs.csv", index=False)
    seeds.to_csv(report / "organ_validation_seed_metrics.csv", index=False)
    decision = research_decision(current, summary, pairs, seeds, protocol)
    write_json(report / "research_decision.json", decision)
    thresholds = select_thresholds(event, "pred_event", "event", protocol)
    visibility = organ.copy()
    visibility["disappearance"] = 1 - visibility.future_observed.astype(int)
    visibility["pred_disappearance"] = 1 - visibility.pred_visibility
    visibility_thresholds = select_thresholds(visibility, "pred_disappearance", "disappearance", protocol)
    artifacts = {str(path.relative_to(root)): file_hash(path) for path in sorted((root / "outputs/m3_5").glob("*/*/*")) if path.is_file() and (path.suffix == ".pt" or path.name.endswith("fits.json"))}
    expected = len(protocol["partitions"]) * len(protocol["seeds"]) * 12
    records = list((root / "outputs/m3_5").glob("*/*/*training.json"))
    if len(records) != expected:
        raise ValueError(f"Required {expected} completed neural runs, got {len(records)}")
    lock = {"protocol_sha256": file_hash(root / "configs/m3_5/protocol.json"), "locked_before_test_inference": True,
            "event_thresholds": thresholds, "visibility_disappearance_thresholds": visibility_thresholds,
            "research_decision_before_test": decision, "artifact_fingerprints": artifacts,
            "neural_runs": len(records), "test_used_for_selection": False, "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_source_sha256": file_hash(root / "src/m35/analysis.py"),
            "training_entrypoint_sha256": file_hash(root / "scripts/train_m3_5.py")}
    write_json(report / "selection_lock.json", lock)
    return lock


def replay_test(root, protocol, device, lock):
    root = Path(root)
    results, repeat_checks = [], []
    splits = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]
    for partition, assignment in splits.items():
        directory = root / "outputs/m3_5" / partition
        fits = json.loads((directory / "current/ridge_fits.json").read_text())
        current_rows = current_arrays(root, assignment)
        test_current = current_rows.loc[current_rows.split.eq("test")].reset_index(drop=True)
        for name, fit in fits.items():
            inputs = test_current[["DAS"]] if name == "A2-DAS-Ridge" else test_current[list(GEOMETRY_FEATURES)]
            prediction = np.maximum(ridge_predict(inputs, fit), 0)
            frame = base_prediction_frame(test_current, name, -1, partition, "current")
            for position, target in enumerate(PLANT_TARGETS):
                frame[f"pred_{target}"] = prediction[:, position]
                frame[f"true_{target}"] = test_current[target].to_numpy()
            results.append(frame)
        for name, field, target in (("A2-Z-Range-Proxy", "z_range_cm", "plant_height"), ("A2-Z-Quantile-Proxy", "height_quantile_proxy_cm", "plant_height"), ("A2-Hull-Proxy", "hull_diameter_cm", "plant_width")):
            frame = base_prediction_frame(test_current, name, -1, partition, "current")
            for target_name in PLANT_TARGETS:
                frame[f"pred_{target_name}"] = test_current[field].to_numpy() if target == target_name else np.nan
                frame[f"true_{target_name}"] = test_current[target_name].to_numpy()
            results.append(frame)
        organ = tabular_arrays(root, "organ", assignment)
        indices = np.flatnonzero(organ["rows"].split.eq("test"))
        target_positions = [ORGAN_FEATURES.index(target) for target in ORGAN_TARGETS]
        for name in ("C0-Persistence", "C1-Linear-Growth"):
            prediction = []
            for index in indices:
                row = organ["rows"].iloc[index]
                history = organ["sequence"][index][:, target_positions]
                prediction.append(persistence(history) if name.startswith("C0") else linear_growth(history, json.loads(row.history_DAS), row.target_DAS))
            regression = np.asarray(prediction) - organ["baseline"][indices]
            results.append(tabular_prediction_frame(organ, indices, regression, np.ones(len(indices)), name, -1, partition, "organ"))
        event = tabular_arrays(root, "event", assignment)
        event_fits = json.loads((directory / "event/simple_fits.json").read_text())
        for horizon in protocol["horizons"]:
            indices = np.flatnonzero(event["rows"].split.eq("test") & event["rows"].horizon.eq(horizon))
            rows, fit = event["rows"].iloc[indices], event_fits[str(horizon)]
            stages = np.digitize(rows.current_DAS, [58, 75])
            inputs = np.column_stack((rows.current_visible_count, rows.current_DAS / 100, rows.forecast_delta_days / 10, rows.target_DAS / 100))
            for name in ("E0-Frequency", "E0-DAS", "E1-Logistic"):
                if name == "E1-Logistic":
                    probability, counts = logistic_predict(inputs, fit["logistic"]), np.maximum(ridge_predict(inputs, fit["count_ridge"]), 0)
                elif name == "E0-Frequency":
                    probability, counts = np.full(len(indices), fit["frequency"]["probability"]), np.full((len(indices), 1), fit["frequency"]["count"])
                else:
                    probability = np.asarray([fit["stages"][str(stage)]["probability"] for stage in stages])
                    counts = np.asarray([[fit["stages"][str(stage)]["count"]] for stage in stages])
                results.append(tabular_prediction_frame(event, indices, counts, probability, name, -1, partition, "event"))
        for path in sorted(directory.glob("*/*.pt")):
            if path.name.endswith("_initial.pt"):
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint["test_used_for_selection"] or checkpoint["selection_split"] != "val":
                raise ValueError("Invalid checkpoint selection")
            name, task, seed = checkpoint["model_name"], checkpoint["task"], checkpoint["seed"]
            for source, expected in checkpoint["fingerprints"].items():
                if file_hash(root / source) != expected:
                    raise ValueError(f"Training source changed: {source}")
            if task == "current":
                model = CurrentPointNetRegressor(len(GEOMETRY_FEATURES) if name.startswith("A3") else 0,
                                                 protocol["pointnet_feature_dim"], protocol["pointnet_hidden_dim"], protocol["dropout"]).to(device)
                model.load_state_dict(checkpoint["model"])
                dataset = Sugar4DCurrentDataset(test_current, root / "data/processed/sugar4d/point_cache", protocol["points"], protocol["sampling_seed"])
                loader = DataLoader(dataset, batch_size=protocol["pointnet_batch_size"], num_workers=0)
                geometry = torch.tensor(transform(test_current[list(GEOMETRY_FEATURES)], checkpoint["geometry_scaler"]), device=device)
                raw = current_inference(model, loader, geometry, device)
                repeated = current_inference(model, loader, geometry, device)
                absolute = np.maximum(inverse(raw, checkpoint["target_scaler"]), 0)
                frame = base_prediction_frame(test_current, name, seed, partition, "current")
                for position, target in enumerate(PLANT_TARGETS):
                    frame[f"pred_{target}"] = absolute[:, position]
                    frame[f"true_{target}"] = test_current[target].to_numpy()
            else:
                data = organ if task == "organ" else (event if task == "event" else tabular_arrays(root, task, assignment, fits, include_test=True))
                indices = np.flatnonzero(data["rows"].split.eq("test"))
                model = tabular_model(name, task, protocol).to(device)
                model.load_state_dict(checkpoint["model"])
                tensors = {"sequence": torch.tensor(transform(data["sequence"], checkpoint["sequence_scaler"]), device=device),
                           "times": torch.tensor(transform(data["times"], checkpoint["time_scaler"]), device=device)}
                raw, repeated = tabular_inference(model, tensors, indices), tabular_inference(model, tensors, indices)
                regression, probability = decode_tabular(raw, checkpoint["label_scaler"], task)
                frame = tabular_prediction_frame(data, indices, regression, probability, name, seed, partition, task)
            if not np.array_equal(raw, repeated):
                raise ValueError("Repeated inference differs")
            repeat_checks.append({"partition": partition, "model": name, "seed": seed, "identical": True, "max_difference": 0.0})
            frame.to_csv(path.parent / f"{name}_seed{seed}_test.csv", index=False)
            results.append(frame)
        print(f"REPLAY {partition} completed locked test inference", flush=True)
    predictions = pd.concat(results, ignore_index=True)
    predictions.to_csv(root / "outputs/m3_5/test_predictions.csv", index=False)
    write_json(root / "reports/m3_5/inference_repeatability.json", {"runs": repeat_checks, "all_identical": True})
    return predictions


def training_audit(root, protocol):
    root = Path(root)
    records = [json.loads(path.read_text()) for path in sorted((root / "outputs/m3_5").glob("*/*/*training.json"))]
    table = pd.DataFrame([{key: value for key, value in record.items() if key not in ("history", "fingerprints", "training_plants", "validation_plants")} for record in records])
    for record in records:
        if record["epochs"] != protocol["epochs"] or len(record["history"]) != protocol["epochs"] or record["test_used_for_selection"]:
            raise ValueError("Incomplete budget or illegal selection")
        if set(record["training_plants"]) & set(record["validation_plants"]):
            raise ValueError("Plant split leakage")
        destination = root / "outputs/m3_5" / record["partition"] / record["task"]
        initial = torch.load(destination / f"{record['model']}_seed{record['seed']}_initial.pt", weights_only=True, map_location="cpu")
        if tensor_state_hash(initial) != record["initial_tensor_sha256"]:
            raise ValueError("Initial state fingerprint mismatch")
    for partition in protocol["partitions"]:
        for seed in protocol["seeds"]:
            for task, real, repeat in (("organ", "C3-GRU", "C4-Repeated-GRU"), ("event", "E3-GRU", "E4-Repeated-GRU")):
                subset = table.loc[table.partition.eq(partition) & table.seed.eq(seed) & table.task.eq(task)].set_index("model")
                for field in ("initial_tensor_sha256", "parameters", "optimizer_steps", "epochs"):
                    if subset.loc[real, field] != subset.loc[repeat, field]:
                        raise ValueError("Real/repeated control training not matched")
    table.to_csv(root / "reports/m3_5/training_resources.csv", index=False)
    return {"neural_runs": len(records), "total_optimizer_steps": int(table.optimizer_steps.sum()), "total_training_seconds": float(table.seconds.sum()),
            "all_fixed_budget_completed": True, "real_repeat_matched": True, "test_used_for_selection": False}


def final_analysis(root, protocol, lock, test):
    root = Path(root)
    report = root / "reports/m3_5"
    current_summary_before_test, current_train_val = current_diagnostics(root, protocol)
    train_validation = pd.concat([current_train_val, *[collect_train_validation(root, task) for task in ("organ", "event", "geometry_absolute", "geometry_residual")]], ignore_index=True)
    all_predictions = pd.concat([train_validation, test], ignore_index=True)
    organ = all_predictions.loc[all_predictions.task.eq("organ")].copy()
    summary, plants, seeds = morphology_scores(organ, ORGAN_TARGETS, protocol)
    summary.to_csv(report / "organ_forecast_metrics.csv", index=False)
    plants.to_csv(report / "organ_plant_errors.csv", index=False)
    seeds.to_csv(report / "organ_seed_metrics.csv", index=False)
    pairs = paired_scores(plants, [("C3-GRU", "C2-MLP"), ("C3-GRU", "C4-Repeated-GRU"), ("C3-GRU", "C0-Persistence"), ("C1-Linear-Growth", "C0-Persistence")], protocol)
    pairs.to_csv(report / "organ_paired_comparisons.csv", index=False)
    current = all_predictions.loc[all_predictions.task.eq("current")]
    current_summary, current_plants, current_seeds = morphology_scores(current, PLANT_TARGETS, protocol)
    current_summary.to_csv(report / "current_state_final_metrics.csv", index=False)
    m3_predictions = pd.read_csv(root / "outputs/m3/official/points2048/neural_predictions_points2048.csv")
    matched_records = []
    a1_test = current.loc[current.partition.eq("official") & current.split.eq("test") & current.model.eq("A1-PointNet")]
    future = m3_predictions.loc[m3_predictions.model.eq("B3-2")]
    for horizon in protocol["horizons"]:
        requested = future.loc[future.horizon.eq(horizon), ["plant_uid", "seed", "target_DAS"]].rename(columns={"target_DAS": "DAS"})
        matched = a1_test.merge(requested, on=["plant_uid", "seed", "DAS"], validate="one_to_one")
        for target in PLANT_TARGETS:
            current_error = (matched[f"pred_{target}"] - matched[f"true_{target}"]).abs().groupby([matched.plant_uid, matched.seed]).mean().groupby("plant_uid").mean()
            future_subset = future.loc[future.horizon.eq(horizon)]
            future_error = (future_subset[f"pred_{target}"] - future_subset[f"true_{target}"]).abs().groupby([future_subset.plant_uid, future_subset.seed]).mean().groupby("plant_uid").mean()
            difference = future_error - current_error
            low, high = bootstrap_mean(difference, protocol)
            matched_records.append({"target": target, "horizon": horizon, "plants": len(current_error), "current_target_frame_estimation_mae": float(current_error.mean()),
                                    "M3_past_XYZ_future_mae": float(future_error.mean()), "future_minus_current_task_gap": float(difference.mean()),
                                    "gap_ci95_low": low, "gap_ci95_high": high, "scope": "Same requested target scans; current estimation has target-time XYZ and is not a forecasting competitor"})
    pd.DataFrame(matched_records).to_csv(report / "current_vs_M3_matched_target_scans.csv", index=False)
    geometry = all_predictions.loc[all_predictions.task.str.startswith("geometry", na=False)]
    geometry_summary, geometry_plants, geometry_seeds = morphology_scores(geometry, PLANT_TARGETS, protocol)
    geometry_summary.to_csv(report / "geometry_forecast_metrics.csv", index=False)
    geometry_plants.to_csv(report / "geometry_plant_errors.csv", index=False)
    geometry_pairs = paired_scores(geometry_plants.assign(task="geometry"), [("C5-History-Residual", "C5-Current-Residual"), ("C5-History-Absolute", "C5-Current-Absolute"), ("C5-History-Residual", "C5-History-Absolute"), ("C5-Current-Residual", "C5-Current-Absolute")], protocol)
    geometry_pairs.to_csv(report / "geometry_paired_comparisons.csv", index=False)
    event = all_predictions.loc[all_predictions.task.eq("event")].copy()
    event_summary, event_seeds, event_plants = summarize_classification(event, "pred_event", "event", lock["event_thresholds"], protocol)
    event_summary.to_csv(report / "new_leaf_event_metrics.csv", index=False)
    event_seeds.to_csv(report / "new_leaf_event_seed_metrics.csv", index=False)
    event_plants.to_csv(report / "new_leaf_event_plant_metrics.csv", index=False)
    event_errors = pd.concat([event_plants.assign(task="event", target=target, mae=event_plants[field]) for target, field in (("event_brier", "brier"), ("new_count", "count_mae"))], ignore_index=True)
    event_pairs = paired_scores(event_errors, [("E3-GRU", "E2-MLP"), ("E3-GRU", "E4-Repeated-GRU"), ("E3-GRU", "E1-Logistic"), ("E1-Logistic", "E0-DAS")], protocol)
    event_pairs.to_csv(report / "new_leaf_event_paired.csv", index=False)
    organ["disappearance"] = 1 - organ.future_observed.astype(int)
    organ["pred_disappearance"] = 1 - organ.pred_visibility
    visibility_summary, visibility_seeds, visibility_plants = summarize_classification(organ, "pred_disappearance", "disappearance", lock["visibility_disappearance_thresholds"], protocol)
    visibility_summary.to_csv(report / "future_visibility_metrics.csv", index=False)
    visibility_seeds.to_csv(report / "future_visibility_seed_metrics.csv", index=False)
    visibility_plants.to_csv(report / "future_visibility_plant_metrics.csv", index=False)
    old_plants = pd.read_csv(root / "reports/m3/official/plant_level_errors.csv")
    old_records = []
    for target in PLANT_TARGETS:
        records = old_plants[["model", "seed", "horizon", "plant_uid", f"{target}_mae"]].rename(columns={f"{target}_mae": "mae"})
        old_records.append(records.assign(partition="official", split="test", task="geometry", target=target))
    legacy_comparison = pd.concat([geometry_plants.loc[geometry_plants.partition.eq("official") & geometry_plants.split.eq("test")].assign(task="geometry"), *old_records], ignore_index=True)
    legacy_pairs = paired_scores(legacy_comparison, [(name, "B3-2") for name in ("C5-Current-Absolute", "C5-Current-Residual", "C5-History-Absolute", "C5-History-Residual")], protocol)
    legacy_pairs.to_csv(report / "geometry_vs_M3_paired.csv", index=False)
    visibility_coverage = organ.drop_duplicates(["partition", "split", "window_id"]).groupby(["partition", "split", "horizon"]).agg(eligible_windows=("window_id", "size"), observed_target_windows=("future_observed", "sum"), plants=("plant_uid", "nunique")).reset_index()
    visibility_coverage["not_observed_target_windows"] = visibility_coverage.eligible_windows - visibility_coverage.observed_target_windows
    visibility_coverage.to_csv(report / "visibility_evaluation_coverage.csv", index=False)
    failures = []
    for keys, group in organ.loc[organ.partition.eq("official") & organ.split.eq("val") & organ.model.eq("C3-GRU") & organ.horizon.eq(2)].groupby(["plant_uid", "leaf_instance_id"]):
        plant, leaf = keys
        for target in ORGAN_TARGETS:
            error = (group[f"pred_{target}"] - group[f"true_{target}"]).abs()
            by_seed = error.groupby(group.seed).mean()
            failures.append({"plant_uid": plant, "leaf_instance_id": leaf, "target": target, "horizon": 2,
                             "validation_mae_seed_mean": float(by_seed.mean()), "eligible_windows_per_seed": group.window_id.nunique(),
                             "not_observed_target_windows_per_seed": group.loc[~group.future_observed.astype(bool)].window_id.nunique(),
                             "posthoc_diagnostic_only": True})
    pd.DataFrame(failures).sort_values(["target", "validation_mae_seed_mean"], ascending=[True, False]).to_csv(report / "organ_failure_cases.csv", index=False)
    strata = []
    for task, predictions, targets in (("organ", organ, ORGAN_TARGETS), ("geometry", geometry, PLANT_TARGETS)):
        predictions = predictions.loc[predictions.partition.eq("official") & predictions.split.eq("test")].copy()
        predictions["stage"] = pd.cut(predictions.target_DAS, [0, 58, 75, 100], labels=["early", "middle", "late"]).astype(str)
        for dimension in ("stage", "forecast_delta_days"):
            for keys, group in predictions.groupby(["model", "horizon", dimension]):
                model, horizon, value = keys
                for target in targets:
                    eligible = group.loc[group[f"true_{target}"].notna()]
                    error = (eligible[f"pred_{target}"] - eligible[f"true_{target}"]).abs()
                    independent = error.groupby([eligible.plant_uid, eligible.seed]).mean().groupby("plant_uid").mean()
                    strata.append({"task": task, "model": model, "horizon": horizon, "dimension": dimension, "value": str(value),
                                   "target": target, "plants": len(independent), "plant_macro_mae": float(independent.mean())})
    pd.DataFrame(strata).to_csv(report / "stratified_errors.csv", index=False)
    oof = all_predictions.loc[all_predictions.partition.str.startswith("fold") & all_predictions.split.eq("test")].copy()
    if oof.groupby(["task", "model", "seed", "plant_uid"]).partition.nunique().max() != 1:
        raise ValueError("Plant appears in multiple OOF test partitions")
    for task, targets in (("organ", ORGAN_TARGETS), ("current", PLANT_TARGETS), ("geometry_absolute", PLANT_TARGETS), ("geometry_residual", PLANT_TARGETS)):
        subset = oof.loc[oof.task.eq(task)].assign(partition="oof")
        table, independent, by_seed = morphology_scores(subset, targets, protocol)
        table.to_csv(report / f"{task}_oof_metrics.csv", index=False)
    write_json(report / "evaluation_complete.json", {"selection_lock_sha256": file_hash(report / "selection_lock.json"), "all_fixed_test_inference_after_lock": True,
                "test_scope": "Exploratory: same official Test used by M3; folds share fitted training sets", "plant_statistics": True, "test_predictions": len(test)})
    return all_predictions
