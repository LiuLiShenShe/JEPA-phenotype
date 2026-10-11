"""Calendar-time matching without labels, and explicitly separated query contracts."""

import json

import numpy as np
import pandas as pd


def calendar_windows(frames, assignments, horizons, tolerance_hours):
    frames = frames.copy()
    frames["datetime"] = pd.to_datetime(frames.timestamp)
    records = []
    for plant_uid, group in frames.groupby("plant_uid", sort=True):
        group = group.sort_values("frame_idx").reset_index(drop=True)
        valid = group.quality_status.eq("valid") & group.timestamp_reliable
        future = group.loc[valid].sort_values(["datetime", "frame_idx"])
        future_times = future.datetime.to_numpy()
        for end_position in range(2, len(group)):
            history = group.iloc[end_position - 2:end_position + 1]
            if not valid.iloc[end_position - 2:end_position + 1].all():
                continue
            if not np.all(np.diff(history.frame_idx) == 1) or not np.all(np.diff(history.datetime) > np.timedelta64(0, "s")):
                continue
            last = history.iloc[-1]
            for days in horizons:
                requested = last.datetime + pd.Timedelta(days=days)
                differences = np.abs((future_times - requested.to_datetime64()) / np.timedelta64(1, "s"))
                eligible = (future_times > last.datetime.to_datetime64()) & (future.frame_idx.to_numpy() > last.frame_idx)
                candidates = np.flatnonzero(eligible & (differences <= tolerance_hours * 3600))
                if not len(candidates):
                    continue
                chosen = candidates[np.argmin(differences[candidates])]
                target = future.iloc[chosen]
                actual = (target.datetime - last.datetime).total_seconds()
                records.append({
                    "window_id": f"{plant_uid}__end{last.frame_idx}__day{days}",
                    "plant_uid": plant_uid, "genotype": last.genotype, "replicate": last.replicate,
                    "split": assignments[plant_uid], "history_length": 3,
                    "input_image_paths": json.dumps(history.rgb_path.tolist()),
                    "input_frame_indices": json.dumps(history.frame_idx.tolist()),
                    "input_timestamps": json.dumps(history.timestamp.tolist()),
                    "time_deltas": json.dumps([0.0, *np.diff(history.datetime).astype("timedelta64[s]").astype(float).tolist()]),
                    "timestamp_reliable": True, "target_frame_idx": int(target.frame_idx),
                    "target_timestamp": target.timestamp, "target_time_delta_seconds": actual,
                    "forecast_horizon": days, "requested_delta_days": days,
                    "matching_error_hours": (actual - days * 86400) / 3600,
                    "target_leaf_count": target.leaf_count, "target_projected_area": target.projected_area_px,
                    "target_projected_area_ratio": target.projected_area_ratio,
                    "target_growth_stage": target.growth_stage, "history_growth_stage": last.growth_stage,
                })
    if not records:
        raise ValueError("No calendar-time windows under the fixed tolerance")
    rows = pd.DataFrame(records)
    if rows.window_id.duplicated().any():
        raise ValueError("Each history can match at most one target per task")
    return rows.sort_values(["forecast_horizon", "split", "plant_uid", "window_id"]).reset_index(drop=True)


def feasibility(rows, assignments, protocol):
    counts = rows.groupby(["forecast_horizon", "split", "plant_uid", "genotype"]).size().rename("windows").reset_index()
    grid = pd.MultiIndex.from_product([protocol["day_horizons"], sorted(assignments)], names=["forecast_horizon", "plant_uid"]).to_frame(index=False)
    grid["split"] = grid.plant_uid.map(assignments)
    full = grid.merge(counts, on=["forecast_horizon", "split", "plant_uid"], how="left")
    full["windows"] = full.windows.fillna(0).astype(int)
    summary = []
    eligible = []
    criteria = protocol["matching"]
    for days in protocol["day_horizons"]:
        split_stats = {}
        for split in ("train", "val", "test"):
            selection = rows.loc[rows.forecast_horizon.eq(days) & rows.split.eq(split)]
            plant_counts = counts.loc[counts.forecast_horizon.eq(days) & counts.split.eq(split), "windows"]
            record = {
                "days": days, "split": split, "plants": int(selection.plant_uid.nunique()),
                "windows": len(selection), "genotypes": int(selection.genotype.nunique()),
                "windows_per_plant_min": int(plant_counts.min()) if len(plant_counts) else 0,
                "windows_per_plant_median": float(plant_counts.median()) if len(plant_counts) else 0,
                "windows_per_plant_max": int(plant_counts.max()) if len(plant_counts) else 0,
            }
            for name, values in (("actual_hours", selection.target_time_delta_seconds / 3600),
                                 ("signed_match_error_hours", selection.matching_error_hours),
                                 ("absolute_match_error_hours", selection.matching_error_hours.abs())):
                for label, quantile in (("min", 0), ("p05", .05), ("median", .5), ("p95", .95), ("max", 1)):
                    record[f"{name}_{label}"] = float(values.quantile(quantile)) if len(values) else None
            summary.append(record)
            split_stats[split] = record
        usable = (split_stats["train"]["plants"] >= criteria["minimum_train_plants"]
                  and split_stats["val"]["plants"] >= criteria["minimum_val_plants"]
                  and split_stats["train"]["windows_per_plant_median"] >= criteria["minimum_train_median_windows"]
                  and split_stats["val"]["windows_per_plant_median"] >= criteria["minimum_val_median_windows"])
        if usable:
            eligible.append(days)
    coverage = rows.groupby(["forecast_horizon", "split", "genotype", "history_growth_stage", "target_growth_stage"]).agg(
        plants=("plant_uid", "nunique"), windows=("window_id", "size")).reset_index()
    return pd.DataFrame(summary), full, coverage, eligible


def time_conditions(rows, frames, task_family, condition):
    timestamps = np.asarray([pd.to_datetime(json.loads(value)).to_numpy() for value in rows.input_timestamps])
    indices = np.asarray([json.loads(value) for value in rows.input_frame_indices])
    last = pd.DatetimeIndex(timestamps[:, -1])
    starts = pd.to_datetime(frames.timestamp).groupby(frames.plant_uid).min()
    elapsed = (timestamps[:, -1] - rows.plant_uid.map(starts).to_numpy()) / np.timedelta64(1, "D")
    gaps = np.diff(timestamps, axis=1) / np.timedelta64(1, "D")
    calendar = 2 * np.pi * last.dayofyear.to_numpy() / 365.25
    clock = 2 * np.pi * (last.hour.to_numpy() + last.minute.to_numpy() / 60) / 24
    query = rows.forecast_horizon.to_numpy() if task_family == "days" else rows.target_time_delta_seconds.to_numpy() / 86400
    values = np.column_stack([elapsed, indices[:, -1], gaps, query, rows.forecast_horizon,
                              np.sin(calendar), np.cos(calendar), np.sin(clock), np.cos(clock)]).astype(np.float32)
    if condition == "rgb":
        values[:] = 0
    elif condition == "offset":
        values[:, 4] = 0
    elif condition == "relative":
        values[:, [4, 6, 7, 8, 9]] = 0
    elif condition != "full":
        raise ValueError(f"Unknown time condition: {condition}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite time inputs")
    return values


class DiagnosticDataset:
    def __init__(self, rows, frames, store, *, task_family, condition, features=None):
        self.rows = rows.reset_index(drop=True)
        paths = [json.loads(value) for value in self.rows.input_image_paths]
        self.features = np.array(store.features[store.row_indices(paths)], dtype=np.float32, copy=True) if features is None else features
        self.times = time_conditions(self.rows, frames, task_family, condition)
        self.targets = self.rows[["target_leaf_count", "target_projected_area_ratio"]].to_numpy(dtype=np.float32)
        if self.features.shape != (len(rows), 3, 384):
            raise ValueError("Diagnostic model must only receive three history RGB features")

    def __len__(self):
        return len(self.rows)
