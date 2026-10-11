"""Window metrics, equal-plant macro metrics and paired plant-cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from training.common import TARGET_NAMES


def plant_scores(rows: pd.DataFrame, targets: np.ndarray, predictions: np.ndarray):
    if targets.shape != predictions.shape or targets.shape != (len(rows),2):
        raise ValueError("Metric shapes disagree")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("Non-finite metric inputs")
    records = []
    for plant_uid, positions in rows.reset_index(drop=True).groupby("plant_uid").indices.items():
        errors = predictions[positions] - targets[positions]
        genotype = rows.iloc[positions[0]].genotype
        for target_position, name in enumerate(TARGET_NAMES):
            records.append({"plant_uid":plant_uid,"genotype":genotype,"target":name,"windows":len(positions),
                            "mae":float(np.abs(errors[:,target_position]).mean()),
                            "rmse":float(np.sqrt(np.square(errors[:,target_position]).mean()))})
    return pd.DataFrame(records)


def cluster_interval(values, *, seed=2026, replicates=2000, confidence=0.95):
    array = np.asarray(values,dtype=np.float64)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap needs finite independent plant scores")
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    generator = np.random.default_rng(seed)
    samples = array[generator.integers(0,len(array),size=(replicates,len(array)))].mean(axis=1)
    tail = (1-confidence)/2
    return np.quantile(samples,[tail,1-tail]).tolist()


def evaluate(rows, targets, predictions, *, bootstrap_replicates=2000):
    plants = plant_scores(rows,targets,predictions)
    errors = predictions-targets
    metrics = {"windows":len(rows),"plants":int(rows.plant_uid.nunique()),"bootstrap_unit":"plant_uid",
               "bootstrap_replicates":bootstrap_replicates,"plant_rmse_definition":"mean of within-plant RMSE"}
    for target_position, name in enumerate(TARGET_NAMES):
        group = plants.loc[plants.target.eq(name)]
        metrics[name] = {
            "window_mae":float(np.abs(errors[:,target_position]).mean()),
            "window_rmse":float(np.sqrt(np.square(errors[:,target_position]).mean())),
            "plant_macro_mae":float(group.mae.mean()),"plant_macro_rmse":float(group.rmse.mean()),
            "plant_macro_mae_ci95":cluster_interval(group.mae,replicates=bootstrap_replicates),
            "plant_macro_rmse_ci95":cluster_interval(group.rmse,replicates=bootstrap_replicates),
        }
    metrics["genotypes"] = {
        genotype:{name:{"plants":int(group.loc[group.target.eq(name)].plant_uid.nunique()),
                        "plant_macro_mae":float(group.loc[group.target.eq(name)].mae.mean()),
                        "plant_macro_rmse":float(group.loc[group.target.eq(name)].rmse.mean())} for name in TARGET_NAMES}
        for genotype,group in plants.groupby("genotype")
    }
    return metrics, plants


def selection_score(rows, targets, predictions, target_scale):
    plants = plant_scores(rows,targets,predictions)
    return float(np.mean([plants.loc[plants.target.eq(name)].mae.mean()/target_scale[position]
                          for position,name in enumerate(TARGET_NAMES)]))


def prediction_frame(rows, targets, predictions, *, model, seed):
    columns = ["window_id","plant_uid","genotype","split","forecast_horizon","target_frame_idx",
               "target_timestamp","target_time_delta_seconds"]
    result = rows[columns].copy().reset_index(drop=True)
    result["model"] = model
    result["seed"] = seed
    for position,name in enumerate(TARGET_NAMES):
        result[f"true_{name}"] = targets[:,position]
        result[f"pred_{name}"] = predictions[:,position]
        result[f"abs_error_{name}"] = np.abs(predictions[:,position]-targets[:,position])
    return result
