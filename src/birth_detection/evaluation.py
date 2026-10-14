from __future__ import annotations

import json

import numpy as np
import pandas as pd

from birth_detection.common import protocol, strict_matches, write_json
from birth_detection.dataset import frame_path


def select_indices(masks, indices, probabilities):
    selected = []
    for candidate in sorted(indices, key=lambda index: (-float(probabilities[index]), index)):
        mask = masks[candidate]
        if not any(len(np.intersect1d(mask, masks[old], assume_unique=True)) /
                   max(len(mask) + len(masks[old]) - len(np.intersect1d(mask, masks[old], assume_unique=True)), 1) > .5 for old in selected):
            selected.append(candidate)
    return selected


def masks_from_frame(frame):
    return [frame["indices"][start:end] for start, end in zip(frame["offsets"][:-1], frame["offsets"][1:])]


def candidate_evaluation_cache(root, split):
    frames = pd.read_csv(root / "reports/m7/frame_audit.csv").query("split == @split and frame_index > 0")
    cache = {"candidates": pd.read_csv(root / "outputs/m7/candidates.csv")}
    for row in frames.itertuples(index=False):
        with np.load(frame_path(root, row.plant_id, row.DAS)) as frame:
            cached = {key: frame[key] for key in ["indices", "offsets", "candidate_global_indices", "gt_evaluation_only", "gt_new_ids_evaluation_only"]}
        unused_matches, scores, gt_ids = strict_matches(cached["gt_evaluation_only"], masks_from_frame(cached))
        cached["_iou_scores"], cached["_gt_ids"] = scores, gt_ids
        cache[(row.plant_id, int(row.DAS))] = cached
    return cache


def oracle_birth_indices(frame, masks):
    from scipy.optimize import linear_sum_assignment

    if "_iou_scores" in frame:
        scores, gt_ids = frame["_iou_scores"], frame["_gt_ids"]
    else:
        unused_matches, scores, gt_ids = strict_matches(frame["gt_evaluation_only"], masks)
    scores = scores[np.isin(gt_ids, frame["gt_new_ids_evaluation_only"])]
    if not scores.size:
        return []
    rows, columns = linear_sum_assignment(-scores)
    return [int(column) for row, column in zip(rows, columns) if scores[row, column] > .5]


def frame_event_result(frame, selected):
    masks = masks_from_frame(frame)
    if "_iou_scores" in frame:
        from scipy.optimize import linear_sum_assignment

        scores = frame["_iou_scores"][:, selected]
        matches = {}
        if scores.size:
            rows, columns = linear_sum_assignment(-scores)
            matches = {int(frame["_gt_ids"][row]): (int(column), float(scores[row, column]))
                       for row, column in zip(rows, columns) if scores[row, column] > .5}
    else:
        matches, unused_scores, unused_ids = strict_matches(frame["gt_evaluation_only"], [masks[index] for index in selected])
    current_new = set(map(int, frame["gt_new_ids_evaluation_only"]))
    true_positive = sum(label in matches for label in current_new)
    return {"tp": true_positive, "fp": len(selected) - true_positive, "fn": len(current_new) - true_positive,
            "events": len(current_new), "predictions": len(selected)}, matches


def evaluate_candidates(root, method, seed, split, threshold=None, save=False, cache=None):
    candidates = cache["candidates"] if cache is not None else pd.read_csv(root / "outputs/m7/candidates.csv")
    if method in {"difference", "candidate_oracle"}:
        values = np.ones(len(candidates))
    else:
        values = pd.read_csv(root / f"outputs/m7/birth/{method}/seed_{seed}/candidate_probabilities.csv").probability.to_numpy()
    frames = pd.read_csv(root / "reports/m7/frame_audit.csv").query("split == @split and frame_index > 0")
    events = pd.read_csv(root / "reports/m7/organ_birth_events.csv")
    details, predictions, plant_rows = [], [], []
    for plant, group in frames.groupby("plant_id"):
        totals = {"tp": 0, "fp": 0, "fn": 0, "events": 0, "predictions": 0}
        small_tp, small_count = 0, 0
        for record in group.itertuples(index=False):
            def evaluate_frame(frame):
                masks = masks_from_frame(frame)
                global_indices = frame["candidate_global_indices"]
                local_values = values[global_indices]
                if method == "difference":
                    sources = candidates.iloc[global_indices].source.to_numpy()
                    eligible = np.flatnonzero(sources == "change")
                elif method == "candidate_oracle":
                    eligible = oracle_birth_indices(frame, masks)
                else:
                    eligible = np.flatnonzero(local_values >= threshold)
                selected = list(eligible) if method == "candidate_oracle" else select_indices(masks, eligible, local_values)
                score, matches = frame_event_result(frame, selected)
                for name in totals:
                    totals[name] += score[name]
                current_events = events.query("plant_id == @plant and birth_time_visible == @record.DAS and previous_observation_available")
                for event in current_events.itertuples(index=False):
                    found = int(event.leaf_instance_id) + 1 in matches
                    details.append({"method": method, "seed": seed, "threshold": threshold, "split": split, "plant_id": plant,
                                    "DAS": int(record.DAS), "leaf_instance_id": int(event.leaf_instance_id), "detected": found,
                                    "small": bool(event.small), "core_adjacent": bool(event.core_adjacent),
                                    "outcome": "correct_birth_region" if found else "birth_region_missed"})
                for selected_index, local in enumerate(selected):
                    matched = next((label for label, (column, unused_iou) in matches.items() if column == selected_index), -1)
                    predictions.append({"method": method, "seed": seed, "split": split, "plant_id": plant, "DAS": int(record.DAS),
                                        "candidate_index": int(global_indices[local]), "probability": float(local_values[local]),
                                        "matched_GT_evaluation_only": int(matched), "correct_visible_birth": matched in set(frame["gt_new_ids_evaluation_only"])})
                return sum(int(detail["small"] and detail["detected"]) for detail in details if detail["plant_id"] == plant and detail["DAS"] == int(record.DAS)), int(current_events.small.sum())
            if cache is None:
                with np.load(frame_path(root, plant, record.DAS)) as frame:
                    local_small_tp, local_small_count = evaluate_frame(frame)
            else:
                local_small_tp, local_small_count = evaluate_frame(cache[(plant, int(record.DAS))])
            small_tp += local_small_tp
            small_count += local_small_count
        denominator = 2 * totals["tp"] + totals["fp"] + totals["fn"]
        plant_rows.append({"method": method, "seed": seed, "threshold": threshold, "split": split, "plant_id": plant,
                           **totals, "birth_f1": 2 * totals["tp"] / denominator if denominator else 1,
                           "birth_precision": totals["tp"] / max(totals["predictions"], 1),
                           "birth_recall": totals["tp"] / max(totals["events"], 1),
                           "small_recall": small_tp / small_count if small_count else np.nan, "small_events": small_count})
    result = pd.DataFrame(plant_rows)
    if save:
        directory = root / f"outputs/m7/birth_evaluation/{method}/seed_{seed}/{split}"
        directory.mkdir(parents=True, exist_ok=True)
        result.to_csv(directory / "plants.csv", index=False)
        pd.DataFrame(details).to_csv(directory / "event_results.csv", index=False)
        pd.DataFrame(predictions).to_csv(directory / "predictions.csv", index=False)
    return result


def select_birth(root):
    settings = protocol(root)
    destination = root / "reports/m7/birth_selection.json"
    if destination.exists():
        return json.loads(destination.read_text())
    cache = candidate_evaluation_cache(root, "val")
    all_rows, selected = [], {}
    for method in ["single_RF", "history_RF", "single_PointNet", "PointNet_GRU"]:
        choices = []
        for threshold in settings["birth_training"]["thresholds"]:
            for seed in settings["seeds"]:
                result = evaluate_candidates(root, method, seed, "val", threshold, cache=cache)
                all_rows.extend(result.to_dict("records"))
                choices.append({"threshold": threshold, "seed": seed, "score": float(result.birth_f1.mean())})
        table = pd.DataFrame(choices).groupby("threshold").score.mean().reset_index().sort_values(["score", "threshold"], ascending=[False, False])
        selected[method] = {"threshold": float(table.iloc[0].threshold), "validation_mean_seed_plant_F1": float(table.iloc[0].score)}
        print("BIRTH SELECT", method, selected[method], flush=True)
    selected["selection_split"] = "val"
    selected["representative_seed"] = 17
    selected["chosen_method"] = max(["history_RF", "PointNet_GRU"], key=lambda name: (selected[name]["validation_mean_seed_plant_F1"], name))
    pd.DataFrame(all_rows).to_csv(root / "reports/m7/birth_threshold_validation.csv", index=False)
    write_json(destination, selected)
    return selected


def evaluate_forecast(root, split):
    settings = protocol(root)
    selection_path = root / "reports/m7/forecast_selection.json"
    if not selection_path.exists():
        if split != "val":
            raise ValueError("Select forecast threshold on Validation first")
        choices = []
        for threshold in settings["birth_training"]["thresholds"]:
            scores = []
            for seed in settings["seeds"]:
                rows = pd.read_csv(root / f"outputs/m7/forecast/seed_{seed}/probabilities.csv").query("split == 'val'")
                for unused_plant, group in rows.groupby("plant_id"):
                    predicted = group.probability.to_numpy() >= threshold
                    target = group.forecast_target_new_visible.to_numpy(dtype=bool)
                    score = 2 * np.sum(target & predicted) / max(np.sum(target) + np.sum(predicted), 1)
                    scores.append(score)
            choices.append({"threshold": threshold, "mean_F1": float(np.mean(scores))})
        best = sorted(choices, key=lambda row: (row["mean_F1"], row["threshold"]), reverse=True)[0]
        write_json(selection_path, {**best, "selection_split": "val", "target": "next-scan plant event, not current-scan localization"})
    threshold = json.loads(selection_path.read_text())["threshold"]
    result = []
    for seed in settings["seeds"]:
        rows = pd.read_csv(root / f"outputs/m7/forecast/seed_{seed}/probabilities.csv").query("split == @split")
        rows["prediction"] = rows.probability >= threshold
        rows.to_csv(root / f"outputs/m7/forecast/seed_{seed}/event_predictions_{split}.csv", index=False)
        for plant, group in rows.groupby("plant_id"):
            predicted = group.prediction.to_numpy()
            target = group.forecast_target_new_visible.to_numpy(dtype=bool)
            result.append({"method": "next_scan_plant_forecast_RF", "seed": seed, "plant_id": plant, "split": split,
                           "F1": 2 * np.sum(target & predicted) / max(np.sum(target) + np.sum(predicted), 1),
                           "Brier": float(np.mean((group.probability - target) ** 2)), "threshold": threshold})
    return pd.DataFrame(result)
