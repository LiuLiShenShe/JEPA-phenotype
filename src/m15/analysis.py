"""Seed-averaged paired plant inference and strictly descriptive difficulty analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from datasets.m1_feature_dataset import M1Index
from m15.artifacts import save_once
from training.common import TARGET_NAMES, write_json
from training.metrics import cluster_interval


def paired_statistics(differences, settings):
    values = np.asarray(differences, dtype=np.float64)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Paired inference requires finite matched independent plant differences")
    lower, upper = cluster_interval(values, seed=settings["seed"], replicates=settings["bootstrap_replicates"])
    generator = np.random.default_rng(settings["seed"])
    observed = abs(values.mean())
    signs = generator.choice([-1., 1.], size=(settings["permutation_replicates"], len(values)))
    null = np.abs((signs * values).mean(axis=1))
    probability = (np.count_nonzero(null >= observed - 1e-15) + 1) / (len(null) + 1)
    return {"delta_mae": float(values.mean()), "ci95_low": lower, "ci95_high": upper,
            "paired_sign_flip_p_two_sided": float(probability), "plants": len(values),
            "plants_with_gain": int((values < 0).sum())}


def holm_adjust(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities, kind="stable")
    adjusted = np.empty(len(probabilities))
    running = 0.
    for rank, position in enumerate(order):
        running = max(running, (len(probabilities) - rank) * probabilities[position])
        adjusted[position] = min(1., running)
    return adjusted


def bootstrap_spearman(first, second, settings):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if len(first) < 5 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return {"rho": None, "ci95_low": None, "ci95_high": None, "bootstrap_valid": 0}
    rho = float(spearmanr(first, second).statistic)
    generator = np.random.default_rng(settings["seed"])
    selected = generator.integers(0, len(first), size=(settings["bootstrap_replicates"], len(first)))
    ranked_first = rankdata(first[selected], axis=1)
    ranked_second = rankdata(second[selected], axis=1)
    centered_first = ranked_first - ranked_first.mean(axis=1, keepdims=True)
    centered_second = ranked_second - ranked_second.mean(axis=1, keepdims=True)
    denominator = np.sqrt((centered_first ** 2).sum(axis=1) * (centered_second ** 2).sum(axis=1))
    valid = denominator > 0
    correlations = (centered_first[valid] * centered_second[valid]).sum(axis=1) / denominator[valid]
    lower, upper = np.quantile(correlations, [.025, .975]) if len(correlations) else (None, None)
    return {"rho": rho, "ci95_low": lower, "ci95_high": upper, "bootstrap_valid": int(valid.sum())}


def collect(root, lock):
    output = root / "outputs/m1_5"
    protocol = lock["protocol"]
    expected = set()
    for family, horizons, models in (("offset", protocol["offsets"], protocol["offset_models"]),
                                    ("day", lock["eligible_days"], protocol["day_models"])):
        expected.update((f"{family}{horizon}", model, seed) for horizon in horizons for model in models for seed in protocol["seeds"])
    expected.update((f"day{horizon}", model, 0) for horizon in lock["eligible_days"] for model in protocol["day_simple_baselines"])
    actual = set()
    plant_frames, summaries, window_frames = [], [], []
    coverage = {}
    for directory in sorted((output / "predictions").iterdir()):
        if not directory.is_dir():
            continue
        task, model, seed_text = directory.name.split("__")
        seed = int(seed_text[1:])
        actual.add((task, model, seed))
        for split in ("val", "test"):
            frame = pd.read_csv(directory / f"{split}.csv")
            ids = frozenset(frame.window_id)
            key = task, split
            if key in coverage and ids != coverage[key]:
                raise ValueError("Models must evaluate identical task windows, with no plant selection")
            coverage[key] = ids
            plants = pd.read_csv(directory / f"{split}_plants.csv")
            metrics = json.loads((directory / f"{split}_metrics.json").read_text())
            plant_frames.append(plants)
            for name in TARGET_NAMES:
                summaries.append({"task": task, "model": model, "seed": seed, "split": split, "target": name,
                                  "plants": metrics["plants"], "windows": metrics["windows"],
                                  **{metric: metrics[name][metric] for metric in ("plant_macro_mae", "plant_macro_rmse", "window_mae", "window_rmse")}})
            if split == "test":
                window_frames.append(frame[["task", "model", "seed", "window_id", "plant_uid", *[f"abs_error_{name}" for name in TARGET_NAMES]]])
    if expected != actual:
        raise ValueError(f"Incomplete result registry. Missing: {sorted(expected - actual)[:10]}; extra: {sorted(actual - expected)[:10]}")
    return pd.concat(plant_frames, ignore_index=True), pd.DataFrame(summaries), pd.concat(window_frames, ignore_index=True)


def summarize(plants, seeds, settings):
    records = []
    for (split, task, model, target), group in seeds.groupby(["split", "task", "model", "target"]):
        independent = plants.loc[plants.split.eq(split) & plants.task.eq(task) & plants.model.eq(model) & plants.target.eq(target)].groupby("plant_uid").mae.mean()
        lower, upper = cluster_interval(independent, seed=settings["seed"], replicates=settings["bootstrap_replicates"])
        record = {"split": split, "task": task, "model": model, "target": target, "seeds": len(group),
                  "plants": int(group.plants.iloc[0]), "windows": int(group.windows.iloc[0]),
                  "ci95_low": lower, "ci95_high": upper}
        for metric in ("plant_macro_mae", "plant_macro_rmse", "window_mae", "window_rmse"):
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.
        records.append(record)
    return pd.DataFrame(records)


def paired_comparisons(plants, settings):
    comparisons = []
    test = plants.loc[plants.split.eq("test")]
    for task in sorted(test.task.unique()):
        pairs = [("capacity", "gru_full", "gru_repeat"), ("capacity", "gru_full", "mlp_full"),
                 ("differences", "diff_full", "mlp_full"), ("rgb_temporal", "gru_rgb", "mlp_rgb"),
                 ("time", "mlp_full", "mlp_rgb"), ("time", "gru_full", "gru_rgb"), ("time", "mlp_full", "time_full")]
        if task.startswith("offset"):
            pairs.extend([("time", "mlp_full", "mlp_offset"), ("time", "gru_full", "gru_offset"),
                          ("time", "time_full", "time_offset"), ("time", "mlp_offset", "mlp_relative"),
                          ("time", "gru_offset", "gru_relative"), ("time", "time_offset", "time_relative")])
        for family, first, second in pairs:
            for target in TARGET_NAMES:
                selected = test.loc[test.task.eq(task) & test.target.eq(target) & test.model.isin([first, second])]
                matched = selected.groupby(["plant_uid", "model"]).mae.mean().unstack("model")
                if matched.isna().any().any() or set(matched.columns) != {first, second}:
                    raise ValueError("Paired comparisons cannot discard unmatched plants")
                results = paired_statistics(matched[first] - matched[second], settings)
                seed_values = selected.groupby(["seed", "model"]).mae.mean().unstack("model")
                differences = seed_values[first] - seed_values[second]
                comparisons.append({"task": task, "target": target, "family": family, "first": first, "second": second,
                                    "definition": "first minus second; negative favors first", **results,
                                    "relative_gain": float((matched[second].mean() - matched[first].mean()) / matched[second].mean()),
                                    "seed42_delta": float(differences.loc[42]), "seed43_delta": float(differences.loc[43]),
                                    "seed44_delta": float(differences.loc[44]), "all_seeds_favor_first": bool((differences < 0).all())})
    result = pd.DataFrame(comparisons)
    result["holm_p"] = 1.
    for family, positions in result.groupby("family").groups.items():
        result.loc[positions, "holm_p"] = holm_adjust(result.loc[positions, "paired_sign_flip_p_two_sided"])
        result.loc[positions, "family_comparisons"] = len(positions)
    return result


def controlled_training_audit(root, lock):
    rows = []
    for family, horizons in (("offset", lock["protocol"]["offsets"]), ("day", lock["eligible_days"])):
        for horizon in horizons:
            for seed in lock["protocol"]["seeds"]:
                task = f"{family}{horizon}"
                first = root / "outputs/m1_5/checkpoints" / f"{task}__gru_full__s{seed}"
                second = root / "outputs/m1_5/checkpoints" / f"{task}__gru_repeat__s{seed}"
                first_resources = json.loads((first / "resources.json").read_text())
                second_resources = json.loads((second / "resources.json").read_text())
                keys = ("initial_state_sha256", "batch_order_sha256", "optimizer_steps", "trainable_parameters", "epochs")
                same = all(first_resources[key] == second_resources[key] for key in keys)
                first_config = json.loads((first / "config.json").read_text())
                second_config = json.loads((second / "config.json").read_text())
                ignored = {"model", "strategy"}
                same_config = {key: value for key, value in first_config.items() if key not in ignored} == {key: value for key, value in second_config.items() if key not in ignored}
                if not same or not same_config:
                    raise ValueError("A2/A3 training capacity, initialization, batch order or budget was not matched")
                rows.append({"task": task, "seed": seed, "matched": True, "optimizer_steps": first_resources["optimizer_steps"],
                             "parameters": first_resources["trainable_parameters"], "epochs": first_resources["epochs"],
                             "initial_state_sha256": first_resources["initial_state_sha256"], "batch_order_sha256": first_resources["batch_order_sha256"]})
    return pd.DataFrame(rows)


def diagnostic_exposures(index, calendar, protocol):
    frames = index.frame_lookup
    variable = index.frames.groupby("plant_uid").apply(lambda group: len(group[["image_width", "image_height"]].drop_duplicates()) > 1, include_groups=False)
    sources = [(f"offset{horizon}", index.rows.loc[index.rows.forecast_horizon.eq(horizon)]) for horizon in protocol["offsets"]]
    sources += [(f"day{horizon}", calendar.loc[calendar.forecast_horizon.eq(horizon)]) for horizon in protocol["day_horizons"]]
    exposures, thresholds = [], {}
    for task, rows in sources:
        if not len(rows):
            continue
        rows = rows.reset_index(drop=True)
        indices = np.asarray([json.loads(value) for value in rows.input_frame_indices])
        histories = [frames.loc[pd.MultiIndex.from_arrays([rows.plant_uid, indices[:, position]])].reset_index(drop=True) for position in range(3)]
        target = frames.loc[pd.MultiIndex.from_arrays([rows.plant_uid, rows.target_frame_idx])].reset_index(drop=True)
        times = np.asarray([pd.to_datetime(json.loads(value)).to_numpy() for value in rows.input_timestamps])
        relative = (times - times[:, :1]) / np.timedelta64(1, "D")
        centered = relative - relative.mean(axis=1, keepdims=True)
        area = np.column_stack([frame.projected_area_ratio for frame in histories])
        history_leaves = np.column_stack([frame.leaf_count for frame in histories])
        slope = ((area - area.mean(axis=1, keepdims=True)) * centered).sum(axis=1) / (centered ** 2).sum(axis=1)
        actual_days = rows.target_time_delta_seconds.to_numpy() / 86400
        change = target.leaf_count.to_numpy() - histories[-1].leaf_count.to_numpy()
        elapsed = (times[:, -1] - rows.plant_uid.map(index.start_times).to_numpy()) / np.timedelta64(1, "D")
        width = np.column_stack([frame.image_width for frame in histories] + [target.image_width])
        height = np.column_stack([frame.image_height for frame in histories] + [target.image_height])
        records = pd.DataFrame({"task": task, "window_id": rows.window_id, "plant_uid": rows.plant_uid, "split": rows.split,
                                "genotype": rows.genotype, "actual_hours": actual_days * 24, "signed_leaf_change": change,
                                "absolute_leaf_change_per_day": np.abs(change) / actual_days,
                                "visible_leaf_decline": change < 0, "history_stage": histories[-1].growth_stage,
                                "target_stage": target.growth_stage, "elapsed_days": elapsed, "history_area_rate": slope,
                                "history_leaf_change": history_leaves[:, -1] - history_leaves[:, 0],
                                "history_leaf_constant": np.ptp(history_leaves, axis=1) == 0,
                                "history_area_change": area[:, -1] - area[:, 0],
                                "future_area_change": target.projected_area_ratio.to_numpy() - area[:, -1],
                                "plant_size_varies": rows.plant_uid.map(variable).to_numpy(),
                                "window_size_changes": (np.ptp(width, axis=1) > 0) | (np.ptp(height, axis=1) > 0),
                                "last_width": width[:, -2], "last_height": height[:, -2]})
        train_rows = records.loc[records.split.eq("train")]
        threshold = float(train_rows.absolute_leaf_change_per_day.quantile(.75))
        age_cuts = train_rows.elapsed_days.quantile([1 / 3, 2 / 3]).to_numpy()
        interval_cuts = train_rows.actual_hours.quantile([1 / 3, 2 / 3]).to_numpy()
        area_cuts = train_rows.history_area_rate.quantile([1 / 3, 2 / 3]).to_numpy()
        records["rapid_leaf_change"] = records.absolute_leaf_change_per_day > threshold
        for name, values, cuts in (("observed_age_stage", records.elapsed_days, age_cuts),
                                   ("interval_band", records.actual_hours, interval_cuts),
                                   ("area_rate_band", records.history_area_rate, area_cuts)):
            records[name] = np.asarray(["low", "middle", "high"])[np.searchsorted(cuts, values.to_numpy(), side="left")]
        thresholds[task] = {"fit_split": "train", "rapid_leaf_change_per_day_q75": threshold,
                            "elapsed_days_tertiles": age_cuts.tolist(), "actual_hours_tertiles": interval_cuts.tolist(),
                            "history_area_rate_tertiles": area_cuts.tolist()}
        exposures.append(records)
    return pd.concat(exposures, ignore_index=True), thresholds


def difficulty_analysis(exposures, windows, settings):
    averaged = windows.groupby(["task", "model", "window_id", "plant_uid"])[[f"abs_error_{target}" for target in TARGET_NAMES]].mean().reset_index()
    merged = averaged.merge(exposures.loc[exposures.split.eq("test")], on=["task", "window_id", "plant_uid"], validate="many_to_one")
    models = ["mlp_full", "gru_full", "gru_repeat", "diff_full", "time_full", "mlp_offset", "gru_offset"]
    merged = merged.loc[merged.model.isin(models)]
    strata = ("rapid_leaf_change", "visible_leaf_decline", "history_stage", "target_stage", "observed_age_stage",
              "plant_size_varies", "window_size_changes", "interval_band", "area_rate_band")
    summaries, paired = [], []
    for task, task_rows in merged.groupby("task"):
        for factor in strata:
            for level, selected in task_rows.groupby(factor):
                for target in TARGET_NAMES:
                    errors = selected.groupby(["model", "plant_uid"])[f"abs_error_{target}"].mean().unstack("model")
                    for model in errors:
                        values = errors[model].dropna()
                        lower, upper = cluster_interval(values, seed=settings["seed"], replicates=settings["bootstrap_replicates"]) if len(values) >= 5 else (None, None)
                        summaries.append({"task": task, "target": target, "factor": factor, "level": str(level), "model": model,
                                          "plants": len(values), "windows": selected.loc[selected.model.eq(model), "window_id"].nunique(),
                                          "plant_macro_mae": float(values.mean()), "ci95_low": lower, "ci95_high": upper,
                                          "warning": "too few plants for CI" if len(values) < 5 else "descriptive subgroup; not for selection"})
                    for second in ("gru_repeat", "mlp_full"):
                        if "gru_full" not in errors or second not in errors:
                            continue
                        differences = errors.gru_full - errors[second]
                        if differences.isna().any():
                            raise ValueError("Subgroup pairing lost identical model coverage")
                        lower, upper = cluster_interval(differences, seed=settings["seed"], replicates=settings["bootstrap_replicates"]) if len(differences) >= 5 else (None, None)
                        paired.append({"task": task, "target": target, "factor": factor, "level": str(level), "second": second,
                                       "plants": len(differences), "gru_minus_second": float(differences.mean()), "ci95_low": lower, "ci95_high": upper})
    associations = []
    variables = ("absolute_leaf_change_per_day", "signed_leaf_change", "actual_hours", "elapsed_days", "history_area_rate")
    for (task, model), selected in merged.groupby(["task", "model"]):
        plant_values = selected.groupby("plant_uid")[[*variables, *[f"abs_error_{target}" for target in TARGET_NAMES]]].mean()
        for variable in variables:
            for target in TARGET_NAMES:
                correlation = bootstrap_spearman(plant_values[variable], plant_values[f"abs_error_{target}"], settings)
                associations.append({"task": task, "model": model, "target": target, "variable": variable, "plants": len(plant_values),
                                     **correlation, "unit": "plant means; descriptive association, not mechanism"})
    return pd.DataFrame(summaries), pd.DataFrame(paired), pd.DataFrame(associations)


def label_and_schedule_tables(root, index, exposures):
    frames = index.frames.copy()
    frames["split"] = frames.plant_uid.map(index.split["assignments"])
    frames["recording_month"] = frames.datetime.dt.to_period("M").astype(str)
    cohort = frames.groupby("plant_uid").agg(genotype=("genotype", "first"), split=("split", "first"),
                                               first_record=("datetime", "min"), frames=("frame_idx", "size")).reset_index()
    cohort["first_record_month"] = cohort.first_record.dt.to_period("M").astype(str)
    schedule = cohort.groupby(["split", "genotype", "first_record_month"]).size().rename("plants").reset_index()
    schedule.to_csv(root / "reports/m1_5/schedule_genotype_coverage.csv", index=False)
    cohort.to_csv(root / "reports/m1_5/plant_recording_cohorts.csv", index=False)
    transitions = []
    for plant_uid, group in frames.groupby("plant_uid"):
        group = group.sort_values("frame_idx")
        previous = group.shift(1)
        eligible = group.quality_status.eq("valid") & previous.quality_status.eq("valid") & group.frame_idx.sub(previous.frame_idx).eq(1)
        for position in np.flatnonzero(eligible.to_numpy()):
            last = group.iloc[position]
            before = previous.iloc[position]
            transitions.append({"plant_uid": plant_uid, "genotype": last.genotype, "split": last.split, "stage": before.growth_stage,
                                "leaf_change": last.leaf_count - before.leaf_count,
                                "delta_hours": (last.datetime - before.datetime).total_seconds() / 3600})
    transitions = pd.DataFrame(transitions)
    transitions.to_csv(root / "outputs/m1_5/adjacent_label_transitions.csv", index=False)
    summary = transitions.groupby(["split", "stage"]).agg(plants=("plant_uid", "nunique"), transitions=("leaf_change", "size"),
                 decline_fraction=("leaf_change", lambda series: float((series < 0).mean())),
                 unchanged_fraction=("leaf_change", lambda series: float((series == 0).mean())),
                 absolute_change_mean=("leaf_change", lambda series: float(series.abs().mean()))).reset_index()
    summary.to_csv(root / "reports/m1_5/label_transition_summary.csv", index=False)
    test_coverage = exposures.loc[exposures.split.eq("test")].groupby("task").agg(
        plants=("plant_uid", "nunique"), windows=("window_id", "size"), declining_windows=("visible_leaf_decline", "sum"),
        rapid_change_windows=("rapid_leaf_change", "sum"), variable_size_plants=("plant_size_varies", "sum"),
        size_changed_windows=("window_size_changes", "sum")).reset_index()
    variable_counts = exposures.loc[exposures.split.eq("test") & exposures.plant_size_varies].groupby("task").plant_uid.nunique()
    test_coverage["variable_size_plants"] = test_coverage.task.map(variable_counts).fillna(0).astype(int)
    test_coverage.to_csv(root / "reports/m1_5/diagnostic_test_coverage.csv", index=False)
    dynamic = exposures.groupby(["task", "split"]).agg(
        plants=("plant_uid", "nunique"), windows=("window_id", "size"),
        history_leaf_constant_fraction=("history_leaf_constant", "mean"),
        future_leaf_unchanged_fraction=("signed_leaf_change", lambda values: float((values == 0).mean())),
        future_leaf_change_median=("signed_leaf_change", "median"),
        history_absolute_area_change_median=("history_area_change", lambda values: float(values.abs().median())),
        future_absolute_area_change_median=("future_area_change", lambda values: float(values.abs().median()))).reset_index()
    dynamic.to_csv(root / "reports/m1_5/dynamic_information_summary.csv", index=False)


def decision(pairs, protocol, eligible_days):
    settings = protocol["statistics"]
    primary = pairs.loc[pairs.task.eq(settings["primary_task"]) & pairs.target.eq(settings["primary_target"]) & pairs.family.eq("capacity")]
    continuation = (5 in eligible_days and len(primary) == 2 and (primary.delta_mae <= -settings["minimum_leaf_gain"]).all()
                    and (primary.relative_gain >= settings["minimum_relative_gain"]).all() and primary.all_seeds_favor_first.all()
                    and (primary.ci95_high < 0).all() and (primary.holm_p < .05).all())
    core = pairs.loc[pairs.family.isin(["capacity", "differences", "rgb_temporal"])].copy()
    thresholds = core.target.map({"leaf_count": settings["minimum_leaf_gain"], "projected_area_ratio": settings["minimum_area_gain"]})
    practical_excluded = bool((core.ci95_low > -thresholds).all())
    verdict = "CONTINUE" if continuation else ("PIVOT" if practical_excluded else "REVIEW")
    return {"research_judgment": verdict, "primary_task": settings["primary_task"], "primary_target": settings["primary_target"],
            "primary_meets_all_rules": bool(continuation), "all_tasks_exclude_minimum_gain": practical_excluded,
            "primary_comparisons": primary.to_dict("records"), "test_was_inspected_in_m1": True,
            "claim_level": "exploratory M2 recommendation, not independent confirmation", "jepa_trained": False, "three_dimensional_experiments": False}


def analyze(root):
    root = Path(root)
    report = root / "reports/m1_5"
    lock = json.loads((report / "protocol_lock.json").read_text())
    settings = lock["protocol"]["statistics"]
    write_json(report / "time_condition_contracts.json", {
        "shared_observed_slots": ["elapsed_since_first_record_days", "last_original_observation_index", "history_gap1_days", "history_gap2_days"],
        "calendar_slots": ["year_day_sin", "year_day_cos", "hour_sin", "hour_cos"],
        "offset_full": {"forecast_slot": "actual future interval from target metadata", "known_at_deployment": "only if scheduled/requested in advance; not demonstrated by dataset", "task_slot": "fixed original observation offset"},
        "offset_query": {"forecast_slot": "zero; actual future time unavailable", "task_slot": "fixed original observation offset", "observed_calendar": True},
        "relative_offset_query": {"forecast_slot": "zero", "task_slot": "fixed original observation offset", "calendar_slots": "all zero"},
        "days_full": {"forecast_slot": "requested 3/5/7 days, known before prediction", "task_slot": "fixed day-task identity, NOT actual target observation offset", "actual_matching_interval_or_residual": "never input"},
        "rgb_only": {"all_ten_time_slots": "zero", "task_identity": "separate per-task model; no variable future time metadata"},
        "time_only_neural": {"all_rgb_slots": "zero", "time_slots": "same as matching RGB condition"},
    })
    plants, seeds, windows = collect(root, lock)
    plants.to_csv(report / "plant_level_errors.csv", index=False)
    seeds.to_csv(report / "per_seed_metrics.csv", index=False)
    summary = summarize(plants, seeds, settings)
    summary.to_csv(report / "baseline_comparison.csv", index=False)
    pairs = paired_comparisons(plants, settings)
    pairs.to_csv(report / "paired_comparisons.csv", index=False)
    controlled = controlled_training_audit(root, lock)
    controlled.to_csv(report / "capacity_training_audit.csv", index=False)
    index = M1Index(root)
    calendar = pd.read_csv(root / "outputs/m1_5/calendar_windows.csv")
    exposures, thresholds = diagnostic_exposures(index, calendar, lock["protocol"])
    exposures.to_csv(root / "outputs/m1_5/window_diagnostics.csv", index=False)
    history = pd.concat([index.rows.assign(task="offset" + index.rows.forecast_horizon.astype(str)),
                         calendar.assign(task="day" + calendar.forecast_horizon.astype(str))], ignore_index=True)
    first_times = pd.to_datetime(history.input_timestamps.map(lambda value: json.loads(value)[0]))
    last_times = pd.to_datetime(history.input_timestamps.map(lambda value: json.loads(value)[-1]))
    history["history_span_hours"] = (last_times - first_times).dt.total_seconds() / 3600
    history.groupby(["task", "split"]).history_span_hours.agg(["min", "median", "max"]).reset_index().to_csv(report / "history_span_summary.csv", index=False)
    save_once(report / "diagnostic_train_thresholds.json", thresholds)
    difficulty, subgroup_pairs, associations = difficulty_analysis(exposures, windows, settings)
    difficulty.to_csv(report / "difficulty_stratified_errors.csv", index=False)
    subgroup_pairs.to_csv(report / "difficulty_paired.csv", index=False)
    associations.to_csv(report / "difficulty_associations.csv", index=False)
    val_windows = []
    for directory in sorted((root / "outputs/m1_5/predictions").iterdir()):
        if directory.is_dir():
            frame = pd.read_csv(directory / "val.csv")
            val_windows.append(frame[["task", "model", "seed", "window_id", "plant_uid", *[f"abs_error_{name}" for name in TARGET_NAMES]]])
    val_exposures = exposures.loc[exposures.split.eq("val")].copy()
    val_exposures["split"] = "test"
    val_difficulty, val_pairs, val_associations = difficulty_analysis(val_exposures, pd.concat(val_windows, ignore_index=True), settings)
    val_difficulty.assign(split="val", note="validation already used for epoch selection; not independent confirmation").to_csv(report / "difficulty_validation_errors.csv", index=False)
    variable_size = index.frames.groupby("plant_uid").apply(lambda group: len(group[["image_width", "image_height"]].drop_duplicates()) > 1, include_groups=False)
    geometry = []
    for plant_uid in variable_size.loc[variable_size].index:
        frames = index.frames.loc[index.frames.plant_uid.eq(plant_uid)]
        geometry.append({"plant_uid": plant_uid, "split": index.split["assignments"][plant_uid],
                         "sizes": json.dumps(frames[["image_width", "image_height"]].drop_duplicates().values.tolist()),
                         "valid_frames": int(frames.quality_status.eq("valid").sum()),
                         "association_evaluable_on_test": index.split["assignments"][plant_uid] == "test"})
    pd.DataFrame(geometry).to_csv(report / "geometry_inventory.csv", index=False)
    label_and_schedule_tables(root, index, exposures)
    verdict = decision(pairs, lock["protocol"], lock["eligible_days"])
    write_json(report / "decision.json", verdict)
    write_json(report / "metrics.json", {"primary_metric": "Plant-level Macro MAE", "independent_unit": "plant_uid", "seed_aggregation": "mean per-seed errors, not prediction ensemble",
               "bootstrap_replicates": settings["bootstrap_replicates"], "paired_sign_flip_replicates": settings["permutation_replicates"],
               "summary": summary.to_dict("records"), "paired_comparisons": pairs.to_dict("records"), "decision": verdict})
    print(json.dumps(verdict, ensure_ascii=False, indent=2), flush=True)
