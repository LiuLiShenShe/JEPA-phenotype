from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from birth_detection.common import paired_interval, write_json


def sign_flip_probability(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    signs = np.asarray(list(itertools.product([-1, 1], repeat=len(values))))
    statistic = abs(values.mean())
    return float(np.mean(np.abs((signs * values).mean(axis=1)) >= statistic - 1e-12))


def holm_adjust(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    maximum = 0.
    for rank, index in enumerate(order):
        maximum = max(maximum, min(1., (len(values) - rank) * values[index]))
        adjusted[index] = maximum
    return adjusted


def cohort_statistics(root):
    events = pd.read_csv(root / "reports/m7/organ_birth_events.csv")
    rows = []
    for subset, selected in [("all_first_observations", events), ("new_after_initial_scan", events[events.previous_observation_available])]:
        for strata in ["split", "birth_time_visible", "genotype"]:
            keys = ["split"] if strata == "split" else ["split", strata]
            for key, group in selected.groupby(keys):
                key = key if isinstance(key, tuple) else (key,)
                delay = group.birth_time_segmentable - group.birth_time_visible
                track_delay = group.birth_time_trackable - group.birth_time_visible
                rows.append({"subset": subset, "split": key[0], "stratum": strata, "value": "all" if len(key) == 1 else key[1],
                             "plants": group.plant_id.nunique(), "events": len(group), "small_events": int(group.small.sum()),
                             "small_fraction": group.small.mean(), "core_adjacent_events": int(group.core_adjacent.sum()),
                             "core_fraction": group.core_adjacent.mean(), "median_initial_points": group.initial_point_count.median(),
                             "median_initial_area_cm2": group.initial_leaf_area.median(),
                             "segmentable_at_visible_fraction": delay.eq(0).mean(), "segmentable_right_censored": int(group.segmentable_right_censored.sum()),
                             "median_segment_delay_observed_days": delay.median() if delay.notna().any() else np.nan,
                             "trackability_evaluable": int(group.trackability_evaluable.sum()),
                             "median_track_delay_observed_days": track_delay.median() if track_delay.notna().any() else np.nan})
    pd.DataFrame(rows).to_csv(root / "reports/m7/birth_cohort_statistics.csv", index=False)
    frames = pd.read_csv(root / "reports/m7/frame_audit.csv")
    upper = frames.groupby(["split", "plant_id"])[["current_new_events", "new_candidates_recoverable"]].sum().reset_index()
    upper["candidate_recall_ceiling"] = upper.new_candidates_recoverable / upper.current_new_events
    upper.to_csv(root / "reports/m7/candidate_recall_ceiling.csv", index=False)
    gaps = []
    for plant, group in frames.groupby("plant_id"):
        appearances = {}
        for frame in group.itertuples(index=False):
            with np.load(root / f"outputs/m7/frames/{plant}_{frame.DAS}.npz") as cache:
                for gt_id in cache["gt_measures"][:, 0].astype(int):
                    appearances.setdefault(int(gt_id), []).append((int(frame.frame_index), int(frame.DAS)))
        for gt_id, times in appearances.items():
            missing_scans = sum(current[0] - previous[0] - 1 for previous, current in zip(times, times[1:]))
            gaps.append({"plant_id": plant, "split": group.iloc[0].split, "leaf_instance_id": gt_id - 1,
                         "observations": len(times), "first_DAS": times[0][1], "last_DAS": times[-1][1],
                         "interior_missing_scans": missing_scans, "reappears_after_missing_scan": missing_scans > 0,
                         "annotation_vs_occlusion_cause_known": False})
    pd.DataFrame(gaps).to_csv(root / "reports/m7/label_stability_audit.csv", index=False)


def temporal_comparisons(root):
    table = pd.read_csv(root / "reports/m7/birth_metrics.csv")
    rows, paired = [], []
    for split in ["val", "test"]:
        selected = table[table.split == split]
        for current, temporal in [("single_RF", "history_RF"), ("single_PointNet", "PointNet_GRU")]:
            first = selected[selected.method == current].groupby("plant_id").mean(numeric_only=True)
            second = selected[selected.method == temporal].groupby("plant_id").mean(numeric_only=True)
            for metric in ["birth_f1", "birth_recall", "small_recall"]:
                values = (second[metric] - first[metric]).dropna()
                rows.append({"split": split, "comparison": temporal + " - " + current, "metric": metric,
                             **paired_interval(values), "positive_plants": int((values > 0).sum()),
                             "sign_flip_p_two_sided": sign_flip_probability(values), "holm_p_for_two_validation_F1_comparisons": np.nan,
                             "selection_aware": False, "exploratory_selected_thresholds": True})
                paired.extend({"split": split, "comparison": temporal + " - " + current, "metric": metric,
                               "plant_id": plant, "single": first.loc[plant, metric], "temporal": second.loc[plant, metric], "delta": value}
                              for plant, value in values.items())
    frame = pd.DataFrame(rows)
    selected = (frame.split == "val") & (frame.metric == "birth_f1")
    frame.loc[selected, "holm_p_for_two_validation_F1_comparisons"] = holm_adjust(frame.loc[selected, "sign_flip_p_two_sided"])
    frame.to_csv(root / "reports/m7/temporal_comparisons.csv", index=False)
    pd.DataFrame(paired).to_csv(root / "reports/m7/temporal_paired_plants.csv", index=False)
    return frame


def training_budgets(root):
    rows = []
    for path in (root / "outputs/m7").glob("**/completed.json"):
        if "invalid_source_leak_runs" in path.parts:
            continue
        record = json.loads(path.read_text())
        method = record.get("method", "forecast_RF")
        epochs = pd.read_csv(path.parent / "epochs.csv") if (path.parent / "epochs.csv").exists() else None
        rows.append({"method": method, "seed": record["seed"], "epochs": record.get("epochs", np.nan),
                     "trees": record.get("trees", np.nan), "parameters": record.get("parameters", np.nan),
                     "training_plants": record.get("train_plants", record.get("plants", record.get("training_plants"))),
                     "training_examples": record.get("training_pairs", record.get("training_candidates", record.get("training_transitions"))),
                     "positive_examples": record.get("positive_pairs", record.get("positive_candidates", np.nan)),
                     "optimizer_updates": record.get("updates", np.nan),
                     "final_train_loss": epochs.iloc[-1].train_loss if epochs is not None else np.nan,
                     "elapsed_seconds": record.get("seconds", epochs.iloc[-1].elapsed_seconds if epochs is not None else np.nan),
                     "checkpoint_sha256": record["checkpoint_sha256"], "pretraining": "none; frozen S2 proposals shared",
                     "path": str(path.parent.relative_to(root))})
    pd.DataFrame(rows).sort_values(["method", "seed"]).to_csv(root / "reports/m7/training_budgets.csv", index=False)


def failure_contributions(root):
    table = pd.read_csv(root / "reports/m7/measurement_failure_cases.csv")
    counts = table.groupby(["split", "method", "seed", "trait", "kind", "status"]).size().rename("count").reset_index()
    counts["fraction"] = counts["count"] / counts.groupby(["split", "method", "seed", "trait", "kind"])["count"].transform("sum")
    counts.to_csv(root / "reports/m7/measurement_failure_contributions.csv", index=False)
    events = pd.read_csv(root / "reports/m7/event_level_results.csv")
    rows = []
    for (split, method, seed), group in events.groupby(["split", "method", "seed"]):
        for status in ["detection_failure", "identity_failure", "correct"]:
            rows.append({"split": split, "method": method, "seed": seed, "error_type": status,
                         "events": len(group), "count": int((group.error_type == status).sum()),
                         "fraction": (group.error_type == status).mean()})
    pd.DataFrame(rows).to_csv(root / "reports/m7/event_failure_contributions.csv", index=False)
    write_json(root / "reports/m7/statistical_scope.json", {"unit": "plant", "three_seed_average_before_resampling": True,
               "confidence": .95, "bootstrap_repetitions": 10000, "tests": "two-sided exact plant sign flips; Holm for two Validation F1 contrasts",
               "post_selection_bias": "Validation thresholds and methods were selected on these plants; confidence intervals and p values are exploratory, not unbiased confirmatory inference",
               "Test_already_used_in_history": True, "new_external_independent_test": False})


def paired_tracking_plants(root):
    table = pd.read_csv(root / "reports/m7/tracking_metrics.csv")
    metrics = ["idf1", "new_event_f1", "correct_link_coverage", "continuity", "id_switches"]
    rows = []
    for split, subset in table.groupby("split"):
        current = subset[subset.method == "Current"].set_index("plant_id")
        for method, group in subset[subset.method != "Current"].groupby("method"):
            plants = group.groupby("plant_id")[metrics].mean()
            for plant, values in plants.iterrows():
                for metric in metrics:
                    rows.append({"split": split, "plant_id": plant, "method": method, "metric": metric,
                                 "current": current.loc[plant, metric], "proposed_seed_mean": values[metric],
                                 "delta": values[metric] - current.loc[plant, metric],
                                 "seeds_averaged_not_independent_plants": 3})
    pd.DataFrame(rows).to_csv(root / "reports/m7/tracking_paired_plants.csv", index=False)


def forecast_controls(root):
    forecasts = [pd.read_csv(root / f"reports/m7/forecast_metrics_{split}.csv") for split in ["val", "test"]]
    samples = pd.read_csv(root / "outputs/m7/forecast_rows.csv")
    controls, predictions = [], []
    for (split, plant), group in samples[samples.split != "train"].groupby(["split", "plant_id"]):
        target = group.forecast_target_new_visible.to_numpy(dtype=bool)
        for method, value in [("next_scan_always_new", True), ("next_scan_never_new", False)]:
            predicted = np.full(len(group), value)
            controls.append({"method": method, "seed": 0, "split": split, "plant_id": plant,
                             "F1": 2 * np.sum(target & predicted) / max(np.sum(target) + np.sum(predicted), 1),
                             "Brier": float(np.mean((predicted.astype(float) - target) ** 2)), "threshold": np.nan})
            predictions.extend({"method": method, "split": split, "plant_id": plant, "DAS": int(row.DAS),
                                "forecast_target_DAS": int(row.forecast_target_DAS), "prediction": value,
                                "target_evaluation_only": bool(row.forecast_target_new_visible)} for row in group.itertuples(index=False))
    metrics = pd.concat(forecasts + [pd.DataFrame(controls)], ignore_index=True)
    metrics.to_csv(root / "reports/m7/forecast_metrics.csv", index=False)
    metrics.groupby(["method", "split"])[["F1", "Brier"]].mean().reset_index().to_csv(root / "reports/m7/forecast_summary.csv", index=False)
    pd.DataFrame(predictions).to_csv(root / "outputs/m7/forecast/control_predictions.csv", index=False)


def track_level_results(root):
    directories = [root / "outputs/m7/baseline_import"]
    directories.extend(sorted((root / "outputs/m7/tracking").glob("*/seed_*/*")))
    predicted_rows, reference_rows = [], []
    for directory in directories:
        tracks = pd.read_csv(directory / "tracks.csv")
        observations = pd.read_csv(directory / "observations.csv")
        detected = observations[observations.local_id > 0]
        matched = tracks.merge(detected[["plant_id", "split", "method", "seed", "frame_index", "local_id", "gt_id", "identity_correct"]],
                               on=["plant_id", "split", "method", "seed", "frame_index", "local_id"],
                               how="left", validate="one_to_one")
        keys = ["plant_id", "split", "method", "seed"]
        for key, group in matched.groupby(keys + ["track_id"]):
            times = sorted(group.frame_index.unique())
            identities = group.gt_id.dropna()
            dominant = int(identities.value_counts().index[0]) if len(identities) else None
            predicted_rows.append({**dict(zip(keys + ["track_id"], key)), "first_DAS": int(group.DAS.min()),
                                   "last_DAS": int(group.DAS.max()), "observed_scans": len(times),
                                   "interior_missing_scans": times[-1] - times[0] + 1 - len(times),
                                   "GT_matched_scans_evaluation_only": len(identities),
                                   "identity_correct_scans": int(group.identity_correct.eq(True).sum()),
                                   "distinct_GT_leaves_evaluation_only": int(identities.nunique()),
                                   "dominant_GT_id_evaluation_only": dominant,
                                   "identity_purity_among_matched": float((identities == dominant).mean()) if len(identities) else np.nan})
        for key, group in observations.groupby(keys + ["gt_id"]):
            visible = group[group.local_id > 0]
            identities = visible.track_id.unique()
            reference_rows.append({**dict(zip(keys + ["gt_id"], key)), "GT_observations": len(group),
                                   "detected_observations": len(visible), "missed_observations": int((group.local_id <= 0).sum()),
                                   "identity_correct_observations": int(group.identity_correct.sum()),
                                   "distinct_predicted_tracks": len(identities),
                                   "first_visible_DAS": int(group.DAS.min()),
                                   "first_detected_DAS": int(visible.DAS.min()) if len(visible) else np.nan,
                                   "reference_identity_evaluation_only": True})
    pd.DataFrame(predicted_rows).to_csv(root / "reports/m7/track_level_results.csv", index=False)
    pd.DataFrame(reference_rows).to_csv(root / "reports/m7/gt_track_level_results.csv", index=False)


def analyze(root):
    cohort_statistics(root)
    temporal_comparisons(root)
    training_budgets(root)
    failure_contributions(root)
    paired_tracking_plants(root)
    forecast_controls(root)
    track_level_results(root)
