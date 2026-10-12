from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3.common import TARGETS, write_json
from m3.metrics import aggregate_metrics, aggregate_pairs, summarize


def analyze_prediction_set(predictions, destination):
    destination.mkdir(parents=True, exist_ok=True)
    per_seed, plants = summarize(predictions)
    comparison = aggregate_metrics(per_seed, plants)
    paired = aggregate_pairs(plants)
    per_seed.to_csv(destination / "per_seed_metrics.csv", index=False)
    plants.to_csv(destination / "plant_level_errors.csv", index=False)
    comparison.to_csv(destination / "baseline_comparison.csv", index=False)
    paired.to_csv(destination / "paired_comparisons.csv", index=False)
    return comparison, per_seed, plants, paired


def stratified_errors(predictions, first_observations, windows, destination):
    enriched = predictions.merge(windows[["window_id", "history_DAS"]], on="window_id", how="left", validate="many_to_one")
    new_counts = []
    for record in enriched.itertuples(index=False):
        history_last = json.loads(record.history_DAS)[-1]
        arrivals = first_observations.loc[first_observations.plant_uid.eq(record.plant_uid)]
        new_counts.append(int(((arrivals.first_observed_DAS > history_last) & (arrivals.first_observed_DAS <= record.target_DAS)).sum()))
    enriched["new_organ_first_observations"] = new_counts
    enriched["organ_arrival_group"] = np.where(enriched.new_organ_first_observations > 0, "at_least_one_first_observed", "none_first_observed")
    enriched["DAS_stage"] = pd.cut(enriched.target_DAS, bins=[0, 58, 75, 100], labels=["early_47_58", "middle_61_75", "late_79_93"]).astype(str)
    records = []
    for dimension in ("DAS_stage", "forecast_delta_days", "genotype", "organ_arrival_group"):
        for (model, horizon, value), subset in enriched.groupby(["model", "horizon", dimension]):
            for target in TARGETS:
                subset = subset.copy()
                subset["error"] = (subset[f"pred_{target}"] - subset[f"true_{target}"]).abs()
                independent = subset.groupby(["plant_uid", "seed"]).error.mean().groupby("plant_uid").mean()
                records.append({"dimension": dimension, "value": str(value), "model": model, "horizon": horizon, "target": target,
                                "plants": len(independent), "windows_across_seeds": len(subset), "plant_macro_mae": float(independent.mean())})
    pd.DataFrame(records).to_csv(destination / "stratified_errors.csv", index=False)
    enriched.to_csv(destination / "window_error_diagnostics.csv", index=False)
    return enriched


def research_decision(official, official_seed, official_pairs, oof, oof_pairs, fold_tables, protocol):
    task = protocol["primary_task"]
    target = task["target"]
    horizon = task["horizon"]
    failures = []
    details = []
    for control in ("B3-2", "B3-4"):
        primary = official_pairs.loc[official_pairs.left.eq("B3-3") & official_pairs.right.eq(control) & official_pairs.target.eq(target) & official_pairs.horizon.eq(horizon)].iloc[0]
        oof_primary = oof_pairs.loc[oof_pairs.left.eq("B3-3") & oof_pairs.right.eq(control) & oof_pairs.target.eq(target) & oof_pairs.horizon.eq(horizon)].iloc[0]
        by_seed = official_seed.loc[official_seed.target.eq(target) & official_seed.horizon.eq(horizon)].pivot(index="seed", columns="model", values="plant_macro_mae")
        by_seed = by_seed.dropna(subset=["B3-3", control])
        consistent = bool((by_seed["B3-3"] < by_seed[control]).all())
        fold_improvements = 0
        for comparison in fold_tables:
            subset = comparison.loc[comparison.target.eq(target) & comparison.horizon.eq(horizon)].set_index("model")
            fold_improvements += int(subset.loc["B3-3", "plant_macro_mae"] < subset.loc[control, "plant_macro_mae"])
        passed = bool(primary.relative_change_pct <= -10 and primary.ci95_high < 0 and consistent and oof_primary.relative_change_pct <= -10 and fold_improvements >= 3)
        details.append({"control": control, "official_improvement_pct": -float(primary.relative_change_pct), "official_ci95_upper_difference": float(primary.ci95_high),
                        "all_three_seeds_improve": consistent, "oof_improvement_pct": -float(oof_primary.relative_change_pct), "improving_folds": fold_improvements, "passed": passed})
        if not passed:
            failures.append(control)
    return {"research_decision": "CONTINUE" if not failures else "REVIEW", "primary_task": task, "locked_rule": protocol["continue_rule"], "controls": details,
            "scope": "small single-greenhouse, within-genotype/inter-plant prediction; OOF CIs descriptive with overlapping training folds; no biological-birth claim", "start_m4_automatically": False}
