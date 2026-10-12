from __future__ import annotations

import numpy as np
import pandas as pd


TARGETS = ("plant_height", "plant_width", "visible_organ_count")


def plant_level_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    for target in TARGETS:
        if not np.isfinite(predictions[[f"pred_{target}", f"true_{target}"]].to_numpy()).all():
            raise ValueError("Nonfinite metric inputs")
    rows = []
    for (model, seed, horizon, plant_uid), group in predictions.groupby(["model", "seed", "horizon", "plant_uid"], sort=True):
        record = {"model": model, "seed": seed, "horizon": horizon, "plant_uid": plant_uid, "genotype": group.genotype.iloc[0], "windows": len(group)}
        for target in TARGETS:
            error = group[f"pred_{target}"].to_numpy() - group[f"true_{target}"].to_numpy()
            record[f"{target}_mae"] = float(np.abs(error).mean())
            record[f"{target}_rmse"] = float(np.sqrt(np.square(error).mean()))
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap(values, seed=20261011, replicates=10000, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all() or not 0 < confidence < 1 or replicates < 1:
        raise ValueError("Bootstrap values must be finite and non-empty")
    generator = np.random.default_rng(seed)
    samples = values[generator.integers(0, len(values), size=(replicates, len(values)))].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return float(np.quantile(samples, tail)), float(np.quantile(samples, 1 - tail))


def summarize(predictions: pd.DataFrame, *, bootstrap_seed=20261011, replicates=10000):
    plants = plant_level_scores(predictions)
    records = []
    for (model, seed, horizon), group in plants.groupby(["model", "seed", "horizon"], sort=True):
        for target in TARGETS:
            values = group[f"{target}_mae"].to_numpy()
            rmse = group[f"{target}_rmse"].to_numpy()
            low, high = bootstrap(values, seed=bootstrap_seed, replicates=replicates)
            records.append({"model": model, "seed": seed, "horizon": horizon, "target": target, "plants": len(group),
                            "plant_macro_mae": float(values.mean()), "plant_macro_rmse": float(rmse.mean()),
                            "mae_ci95_low": low, "mae_ci95_high": high})
    return pd.DataFrame(records), plants


def paired_difference(predictions: pd.DataFrame, left: str, right: str, target: str, *, replicates=10000):
    left_plants = plant_level_scores(predictions.loc[predictions.model.eq(left)]).groupby(["seed", "horizon", "plant_uid"])[f"{target}_mae"].mean()
    right_plants = plant_level_scores(predictions.loc[predictions.model.eq(right)]).groupby(["seed", "horizon", "plant_uid"])[f"{target}_mae"].mean()
    if set(left_plants.index) != set(right_plants.index):
        raise ValueError("Paired models must cover exactly the same plants, seeds and horizons")
    joined = left_plants.rename("left").to_frame().join(right_plants.rename("right"), how="inner")
    joined["difference_left_minus_right"] = joined.left - joined.right
    rows = []
    for (seed, horizon), group in joined.reset_index().groupby(["seed", "horizon"]):
        low, high = bootstrap(group.difference_left_minus_right.to_numpy(), seed=20261011 + int(seed) + int(horizon), replicates=replicates)
        rows.append({"left": left, "right": right, "target": target, "seed": seed, "horizon": horizon, "plants": len(group),
                     "left_minus_right_mae": float(group.difference_left_minus_right.mean()), "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(rows)


def aggregate_metrics(per_seed, plant_scores, *, replicates=10000):
    records = []
    for (model, horizon, target), group in per_seed.groupby(["model", "horizon", "target"]):
        independent = plant_scores.loc[plant_scores.model.eq(model) & plant_scores.horizon.eq(horizon)].groupby("plant_uid")[[f"{target}_mae", f"{target}_rmse"]].mean()
        lower, upper = bootstrap(independent[f"{target}_mae"], replicates=replicates)
        records.append({"model": model, "horizon": horizon, "target": target, "seeds": len(group), "plants": len(independent),
                        "plant_macro_mae": float(group.plant_macro_mae.mean()), "seed_mae_std": float(group.plant_macro_mae.std(ddof=1)) if len(group) > 1 else 0.0,
                        "plant_macro_rmse": float(group.plant_macro_rmse.mean()), "mae_ci95_low": lower, "mae_ci95_high": upper,
                        "condition": "Oracle historical phenotype" if model in ("B3-0", "B3-1") else "XYZ+scheduled DAS; no label input"})
    return pd.DataFrame(records)


def aggregate_pairs(plants, *, replicates=10000):
    records = []
    averaged = plants.groupby(["model", "horizon", "plant_uid"])[[f"{target}_mae" for target in TARGETS]].mean()
    pairs = [("B3-3", "B3-2"), ("B3-3", "B3-4")]
    pairs += [(model, oracle) for model in ("B3-2", "B3-3", "B3-4") for oracle in ("B3-0", "B3-1")]
    for left, right in pairs:
        for horizon in (1, 2):
            for target in TARGETS:
                left_values = averaged.loc[(left, horizon), f"{target}_mae"]
                right_values = averaged.loc[(right, horizon), f"{target}_mae"]
                if set(left_values.index) != set(right_values.index):
                    raise ValueError("Paired models must cover all identical independent plants")
                difference = left_values - right_values.reindex(left_values.index)
                low, high = bootstrap(difference, replicates=replicates)
                family_low, family_high = bootstrap(difference, replicates=replicates, confidence=1 - 0.05 / (len(pairs) * 2 * len(TARGETS)))
                records.append({"left": left, "right": right, "horizon": horizon, "target": target, "plants": len(difference),
                                "left_minus_right_mae": float(difference.mean()), "relative_change_pct": float(100 * difference.mean() / right_values.mean()),
                                "ci95_low": low, "ci95_high": high, "family_adjusted_ci_low": family_low, "family_adjusted_ci_high": family_high,
                                "family_comparisons": len(pairs) * 2 * len(TARGETS),
                                "aggregation": "average per-seed errors within plant, then paired plant bootstrap; not ensemble forecasts"})
    return pd.DataFrame(records)
