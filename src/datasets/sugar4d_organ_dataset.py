"""Missing-aware organ trajectories and legal historical Sugar4D inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LEAF_FIELDS = {
    "total length [cm]": "leaf_total_length",
    "leaf blade length [cm]": "leaf_blade_length",
    "petiole length [cm]": "petiole_length",
    "leaf blade width [cm]": "leaf_blade_width",
    "leaf area [cm2]": "leaf_area",
}
ORGAN_FEATURES = tuple(LEAF_FIELDS.values())
ORGAN_TARGETS = ("leaf_total_length", "leaf_blade_width", "leaf_area")
PLANT_TARGETS = ("plant_height", "plant_width", "visible_organ_count")
KEYS = ["plant_uid", "leaf_instance_id", "DAS"]
EVENT_FEATURES = ("visible_count", "seen_count", "mean_length", "mean_blade_length", "mean_petiole_length", "mean_width", "mean_area", "sum_area")


def make_trajectories(leaves, organs, metadata):
    if leaves.duplicated(KEYS).any() or organs.duplicated(KEYS).any():
        raise ValueError("Duplicate organ identity/time association")
    joined = leaves.merge(organs, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("PLY and leaf CSV identities disagree")
    joined = joined.drop(columns="_merge").rename(columns=LEAF_FIELDS)
    if not np.isfinite(joined[list(ORGAN_FEATURES)].to_numpy()).all():
        raise ValueError("Observed morphology must be finite")
    if (joined[list(ORGAN_FEATURES)] <= 0).any().any():
        raise ValueError("Observed morphology must be positive")
    if metadata.groupby("plant_uid").split.nunique().max() != 1:
        raise ValueError("Plant crosses split boundaries")
    records = []
    for (plant, leaf), group in joined.groupby(["plant_uid", "leaf_instance_id"], sort=True):
        scans = metadata.loc[metadata.plant_uid.eq(plant)].sort_values("DAS")
        if scans.DAS.duplicated().any() or scans.empty:
            raise ValueError("Invalid actual plant scan schedule")
        indexed = group.set_index("DAS")
        for scan in scans.itertuples(index=False):
            record = {"plant_uid": plant, "plant_id": int(scan.plant_id), "leaf_instance_id": int(leaf),
                      "DAS": int(scan.DAS), "split": scan.split, "observation_status": "not_observed"}
            if scan.DAS in indexed.index:
                source = indexed.loc[scan.DAS]
                record.update({field: float(source[field]) for field in ORGAN_FEATURES})
                record.update({field: float(source[field]) for field in ("centroid_x_m", "centroid_y_m", "centroid_z_m", "point_count")})
                record["observation_status"] = "observed"
            records.append(record)
    result = pd.DataFrame(records).sort_values(KEYS).reset_index(drop=True)
    missing = result.observation_status.ne("observed")
    if result.loc[missing, list(ORGAN_FEATURES)].notna().any().any():
        raise ValueError("Unobserved morphology is not missing")
    return result


def time_vector(days, target_das, first_seen=None):
    history = np.asarray(days, dtype=np.float32)
    if len(history) != 3 or not (np.diff(history) > 0).all() or target_das <= history[-1]:
        raise ValueError("Expected chronological past and a strictly future query")
    values = [*list(history / 100), *list(np.diff(history) / 10), (target_das - history[-1]) / 10, target_das / 100]
    if first_seen is not None:
        if first_seen > history[-1]:
            raise ValueError("Future first-observation time is forbidden")
        values.append((history[-1] - first_seen) / 100)
    return np.asarray(values, dtype=np.float32)


def build_organ_windows(trajectories, horizons=(1, 2)):
    records = []
    for (plant, leaf), group in trajectories.groupby(["plant_uid", "leaf_instance_id"], sort=True):
        group = group.sort_values("DAS").reset_index(drop=True)
        if group.split.nunique() != 1 or group.DAS.duplicated().any():
            raise ValueError("Invalid organ split or scan identity")
        observed = group.observation_status.eq("observed").to_numpy()
        observed_days = group.loc[observed, "DAS"].to_numpy()
        if not len(observed_days):
            continue
        for index in range(2, len(group)):
            if not observed[index - 2:index + 1].all():
                continue
            history = group.iloc[index - 2:index + 1]
            for horizon in horizons:
                if index + horizon >= len(group):
                    continue
                target = group.iloc[index + horizon]
                days = history.DAS.astype(int).tolist()
                first_seen = int(observed_days[observed_days <= days[-1]].min())
                record = {"window_id": f"{plant}_leaf{leaf:02d}_h{horizon}_i{index:02d}", "plant_uid": plant,
                          "leaf_instance_id": int(leaf), "split": group.split.iloc[0], "horizon": int(horizon),
                          "history_DAS": json.dumps(days), "current_DAS": days[-1], "target_DAS": int(target.DAS),
                          "forecast_delta_days": int(target.DAS - days[-1]), "first_seen_so_far_DAS": first_seen,
                          "history_values": json.dumps(history[list(ORGAN_FEATURES)].to_numpy().tolist()),
                          "future_observed": bool(observed[index + horizon])}
                for field in ORGAN_TARGETS:
                    record[f"current_{field}"] = float(history[field].iloc[-1])
                    record[f"target_{field}"] = float(target[field]) if observed[index + horizon] else np.nan
                records.append(record)
    return pd.DataFrame(records)


def build_event_windows(trajectories, metadata, horizons=(1, 2)):
    records, states = [], []
    for plant, scans in metadata.groupby("plant_uid", sort=True):
        scans = scans.sort_values("DAS").reset_index(drop=True)
        if scans.split.nunique() != 1:
            raise ValueError("Plant crosses split boundaries")
        leaves = trajectories.loc[trajectories.plant_uid.eq(plant) & trajectories.observation_status.eq("observed")]
        first = leaves.groupby("leaf_instance_id").DAS.min()
        state_values = []
        for scan in scans.itertuples(index=False):
            current = leaves.loc[leaves.DAS.eq(scan.DAS)]
            if current.empty:
                raise ValueError("Missing whole-plant observation")
            seen = int((first <= scan.DAS).sum())
            values = np.asarray([len(current), seen, *current[list(ORGAN_FEATURES)].mean().tolist(), current.leaf_area.sum()], dtype=float)
            if len(current) != scan.visible_organ_count:
                raise ValueError("Visible leaf quantity disagrees with plant target")
            state_values.append(values)
            states.append({"plant_uid": plant, "DAS": int(scan.DAS), **dict(zip(EVENT_FEATURES, values))})
        for index in range(2, len(scans)):
            for horizon in horizons:
                if index + horizon >= len(scans):
                    continue
                days = scans.DAS.iloc[index - 2:index + 1].astype(int).tolist()
                target_das = int(scans.DAS.iloc[index + horizon])
                count = int(((first > days[-1]) & (first <= target_das)).sum())
                records.append({"window_id": f"{plant}_event_h{horizon}_i{index:02d}", "plant_uid": plant,
                                "split": scans.split.iloc[0], "horizon": int(horizon), "history_DAS": json.dumps(days),
                                "current_DAS": days[-1], "target_DAS": target_das,
                                "forecast_delta_days": target_das - days[-1], "current_visible_count": float(state_values[index][0]),
                                "history_values": json.dumps(np.stack(state_values[index - 2:index + 1]).tolist()),
                                "event": int(count > 0), "new_count": count})
    return pd.DataFrame(records), pd.DataFrame(states)


class Sugar4DOrganDataset(Dataset):
    def __init__(self, windows):
        self.windows = windows.reset_index(drop=True).copy()
        self.sequences = np.asarray([json.loads(value) for value in self.windows.history_values], dtype=np.float32)
        self.times = np.stack([time_vector(json.loads(row.history_DAS), row.target_DAS, row.first_seen_so_far_DAS)
                               for row in self.windows.itertuples(index=False)])
        if self.sequences.shape[1:] != (3, 5) or not np.isfinite(self.sequences).all():
            raise ValueError("Invalid historical organ state")
        for row in self.windows.itertuples(index=False):
            if row.current_DAS != json.loads(row.history_DAS)[-1] or row.target_DAS - row.current_DAS != row.forecast_delta_days:
                raise ValueError("Illegal forecast timing")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        row = self.windows.iloc[index]
        return {"sequence": torch.from_numpy(self.sequences[index]), "time_conditions": torch.from_numpy(self.times[index]),
                "window_id": row.window_id, "plant_uid": row.plant_uid}


class Sugar4DCurrentDataset(Dataset):
    def __init__(self, frames, cache_root, points=2048, sampling_seed=20261011, training=False):
        self.frames = frames.reset_index(drop=True).copy()
        self.cache_root, self.points = Path(cache_root), int(points)
        self.sampling_seed, self.training, self.epoch = sampling_seed, training, 0
        self.clouds = {Path(path).stem: np.load(self.cache_root / (Path(path).stem + ".npy"), allow_pickle=False)
                       for path in self.frames.pointcloud_path.unique()}
        if self.frames.duplicated(["plant_uid", "DAS"]).any() or self.frames.groupby("plant_uid").split.nunique().max() != 1:
            raise ValueError("Invalid current observation identity/split")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        row = self.frames.iloc[index]
        name = Path(row.pointcloud_path).stem
        token = f"{self.sampling_seed}:{self.epoch if self.training else 0}:{name}"
        seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
        cloud = self.clouds[name]
        selection = np.random.default_rng(seed).choice(len(cloud), self.points, replace=len(cloud) < self.points)
        return {"points": torch.from_numpy(cloud[selection].astype(np.float32)), "index": index}
