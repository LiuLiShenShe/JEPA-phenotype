from __future__ import annotations

import numpy as np
import pandas as pd


def count_metrics(predictions):
    frame = predictions.copy()
    frame["absolute_error"] = np.abs(frame.prediction - frame.truth)
    frame["squared_error"] = np.square(frame.prediction - frame.truth)
    per_plant = frame.groupby("plant_uid", sort=True).agg(mae=("absolute_error", "mean"), mse=("squared_error", "mean"))
    per_plant["rmse"] = np.sqrt(per_plant.mse)
    return {"plant_macro_mae": float(per_plant.mae.mean()), "plant_macro_rmse": float(per_plant.rmse.mean()),
            "pooled_rmse": float(np.sqrt(frame.squared_error.mean())), "plants": len(per_plant),
            "scans": len(frame)}, per_plant.reset_index()


def confusion_scores(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    diagonal = np.diag(confusion)
    union = confusion.sum(axis=0) + confusion.sum(axis=1) - diagonal
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    present = union > 0
    iou = np.divide(diagonal, union, out=np.full(3, np.nan), where=present)
    f1 = np.divide(2 * diagonal, denominator, out=np.full(3, np.nan), where=denominator > 0)
    return {"miou": float(np.nanmean(iou)), "macro_f1": float(np.nanmean(f1)),
            **{f"iou_{name}": float(value) if np.isfinite(value) else None
               for name, value in zip(("taproot", "core", "leaf"), iou)}}


def segmentation_metrics(scan_confusions, rows):
    plants = []
    for plant in sorted(rows.plant_uid.unique()):
        mask = rows.plant_uid.eq(plant).to_numpy()
        matrix = np.asarray(scan_confusions)[mask].sum(axis=0)
        plants.append({"plant_uid": plant, **confusion_scores(matrix)})
    plant_frame = pd.DataFrame(plants)
    metrics = {f"plant_macro_{name}": float(plant_frame[name].mean()) for name in ("miou", "macro_f1", "iou_taproot", "iou_core", "iou_leaf")}
    metrics.update({f"pooled_{name}": value for name, value in confusion_scores(np.asarray(scan_confusions).sum(axis=0)).items()})
    return metrics, plant_frame


def paired_interval(differences, replicates=10000, seed=20261012):
    differences = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = differences[generator.integers(len(differences), size=(replicates, len(differences)))].mean(axis=1)
    return {"mean_delta": float(differences.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975)), "independent_plants": len(differences)}
