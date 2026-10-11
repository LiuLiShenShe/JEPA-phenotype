"""Strict read-only M1 window index and label-free cached RGB input interface."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from training.common import file_hash, object_hash


TIME_NAMES = (
    "elapsed_since_first_observation_days", "last_original_observation_index",
    "history_gap_1_days", "history_gap_2_days", "forecast_delta_days",
    "forecast_observation_offset", "calendar_day_sin", "calendar_day_cos",
    "wall_clock_hour_sin", "wall_clock_hour_cos",
)


class M1Index:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.frames = pd.read_csv(self.root / "data/metadata/frames.csv")
        self.rows = pd.read_csv(self.root / "data/processed/temporal_windows.csv")
        self.split = json.loads((self.root / "splits/phenoleaf_plant_split_seed42.json").read_text())
        self.frames["datetime"] = pd.to_datetime(self.frames.timestamp)
        self.frame_lookup = self.frames.set_index(["plant_uid", "frame_idx"])
        self.start_times = self.frames.groupby("plant_uid").datetime.min()
        self.paths = np.asarray([json.loads(value) for value in self.rows.input_image_paths])
        self.indices = np.asarray([json.loads(value) for value in self.rows.input_frame_indices], dtype=np.int64)
        self.timestamps = np.asarray([pd.to_datetime(json.loads(value)).to_numpy() for value in self.rows.input_timestamps])
        self.validate()
        self.times = self._time_conditions()
        self.targets = self.rows[["target_leaf_count", "target_projected_area_ratio"]].to_numpy(dtype=np.float32)

    def validate(self):
        if self.rows.window_id.duplicated().any() or self.frame_lookup.index.duplicated().any():
            raise ValueError("Duplicate window or observation identity")
        if self.indices.shape != (len(self.rows), 3) or self.paths.shape != self.indices.shape:
            raise ValueError("Expected exactly three history observations")
        if not np.all(np.diff(self.indices, axis=1) == 1):
            raise ValueError("History is not consecutive in original observation order")
        if not np.all(np.diff(self.timestamps, axis=1) > np.timedelta64(0, "s")):
            raise ValueError("History timestamps are not strictly increasing")
        history_deltas = np.column_stack([np.zeros(len(self.rows)),np.diff(self.timestamps,axis=1)/np.timedelta64(1,"s")])
        if not np.allclose(history_deltas,np.asarray([json.loads(value) for value in self.rows.time_deltas]),rtol=0,atol=1e-6):
            raise ValueError("History time differences disagree with timestamps")
        if set(self.rows.forecast_horizon) != {2, 4, 8}:
            raise ValueError("M1 expects horizons 2, 4, 8")
        if not np.all(self.rows.target_frame_idx.to_numpy() == self.indices[:, -1] + self.rows.forecast_horizon.to_numpy()):
            raise ValueError("Future target does not match original observation offset")
        if self.rows.groupby("plant_uid").split.nunique().max() != 1:
            raise ValueError("Plant crosses splits")
        if not self.rows.split.eq(self.rows.plant_uid.map(self.split["assignments"])).all():
            raise ValueError("Window split differs from frozen M0 assignment")
        for history_position in range(3):
            keys = pd.MultiIndex.from_arrays([self.rows.plant_uid, self.indices[:, history_position]])
            history = self.frame_lookup.loc[keys]
            if not np.all(history.rgb_path.to_numpy() == self.paths[:, history_position]):
                raise ValueError("History contains a foreign RGB path")
            if not history.quality_status.eq("valid").all():
                raise ValueError("Quarantined history observation")
            if not np.all(history.datetime.to_numpy() == self.timestamps[:, history_position]):
                raise ValueError("History timestamps disagree with original frame")
        target_keys = pd.MultiIndex.from_arrays([self.rows.plant_uid, self.rows.target_frame_idx])
        target = self.frame_lookup.loc[target_keys]
        if not target.quality_status.eq("valid").all():
            raise ValueError("Quarantined future observation")
        target_times = pd.to_datetime(self.rows.target_timestamp).to_numpy()
        if not np.all(target.datetime.to_numpy() == target_times):
            raise ValueError("Target timestamp disagrees with metadata")
        actual_delta = (target_times - self.timestamps[:, -1]) / np.timedelta64(1, "s")
        if not np.all(actual_delta > 0) or not np.allclose(actual_delta, self.rows.target_time_delta_seconds, rtol=0, atol=1e-6):
            raise ValueError("Invalid future wall-clock interval")
        if not np.array_equal(target.leaf_count.to_numpy(), self.rows.target_leaf_count.to_numpy()):
            raise ValueError("Future leaf labels disagree with metadata")
        if not np.allclose(target.projected_area_ratio, self.rows.target_projected_area_ratio, rtol=0, atol=1e-12):
            raise ValueError("Future ratio labels disagree with metadata")
        if not self.rows.timestamp_reliable.all():
            raise ValueError("Unreliable wall-clock time requires an explicit new protocol")
        valid_paths = set(self.frames.loc[self.frames.quality_status.eq("valid"), "rgb_path"])
        if valid_paths != set(self.split["image_to_split"]):
            raise ValueError("Valid image set differs from M0 split image inventory")
        for frame in self.frames.loc[self.frames.quality_status.eq("valid")].itertuples():
            if self.split["image_to_split"][frame.rgb_path] != self.split["assignments"][frame.plant_uid]:
                raise ValueError("Image split is inconsistent with plant split")

    def _time_conditions(self):
        last_time = pd.DatetimeIndex(self.timestamps[:, -1])
        start = self.rows.plant_uid.map(self.start_times).to_numpy()
        elapsed = (self.timestamps[:, -1] - start) / np.timedelta64(1, "D")
        gaps = np.diff(self.timestamps, axis=1) / np.timedelta64(1, "D")
        year_phase = 2 * np.pi * last_time.dayofyear.to_numpy() / 365.25
        hour_phase = 2 * np.pi * (last_time.hour.to_numpy() + last_time.minute.to_numpy() / 60) / 24
        values = np.column_stack([
            elapsed, self.indices[:, -1], gaps,
            self.rows.target_time_delta_seconds.to_numpy() / 86400,
            self.rows.forecast_horizon.to_numpy(),
            np.sin(year_phase), np.cos(year_phase), np.sin(hour_phase), np.cos(hour_phase),
        ]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite time condition")
        return values

    def select(self, split: str, horizon: int):
        return np.flatnonzero(self.rows.split.eq(split) & self.rows.forecast_horizon.eq(horizon))

    def oracle_history(self, selection):
        rows = self.rows.iloc[selection]
        values = []
        for position in range(3):
            keys = pd.MultiIndex.from_arrays([rows.plant_uid, self.indices[selection, position]])
            values.append(self.frame_lookup.loc[keys, ["leaf_count", "projected_area_ratio"]].to_numpy(dtype=np.float64))
        return np.stack(values, axis=1)


class FeatureStore:
    def __init__(self, directory: str | Path, *, root: str | Path | None = None):
        self.directory = Path(directory)
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        if self.metadata["status"] != "complete":
            raise ValueError("Feature cache is incomplete")
        self.manifest = pd.read_csv(self.directory / "manifest.csv")
        self.features = np.load(self.directory / "features.npy", mmap_mode="r")
        if self.manifest.rgb_path.duplicated().any():
            raise ValueError("Feature manifest contains duplicate RGB paths")
        if not np.array_equal(self.manifest.feature_row.to_numpy(), np.arange(len(self.manifest))):
            raise ValueError("Feature manifest row alignment is invalid")
        if self.features.shape != (len(self.manifest), self.metadata["feature_dimension"]):
            raise ValueError("Feature shape does not match manifest")
        if not np.isfinite(self.features).all():
            raise ValueError("Non-finite cached features")
        if file_hash(self.directory / "features.npy") != self.metadata["features_sha256"]:
            raise ValueError("Feature matrix hash differs from its metadata")
        if file_hash(self.directory / "manifest.csv") != self.metadata["manifest_sha256"]:
            raise ValueError("Feature manifest hash differs from its metadata")
        for record in self.manifest.itertuples():
            row_hash = hashlib.sha256(self.features[record.feature_row].tobytes()).hexdigest()
            if row_hash != record.feature_sha256:
                raise ValueError("Cached feature is associated with the wrong image row")
        if object_hash(self.metadata["preprocessing"]) != self.metadata["preprocessing_sha256"]:
            raise ValueError("Feature preprocessing version mismatch")
        self.lookup = dict(zip(self.manifest.rgb_path, self.manifest.feature_row))
        if root is not None:
            audit = pd.read_csv(Path(root) / "data/metadata/file_manifest.csv").set_index("rgb_path")
            for record in self.manifest.itertuples():
                if audit.loc[record.rgb_path, "rgb_sha256"] != record.rgb_sha256:
                    raise ValueError("Cache RGB identity differs from M0 audit")

    def row_indices(self, paths):
        try:
            return np.asarray([[self.lookup[path] for path in history] for history in paths], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f"Missing history RGB feature: {error}") from error


class M1FeatureDataset(Dataset):
    def __init__(self, index: M1Index, store: FeatureStore, split: str, horizon: int):
        self.selection = index.select(split, horizon)
        self.rows = index.rows.iloc[self.selection].reset_index(drop=True)
        self.feature_indices = store.row_indices(index.paths[self.selection])
        self.features = store.features
        self.times = index.times[self.selection]
        self.targets = index.targets[self.selection]

    def inputs(self, position):
        return {
            "features": torch.from_numpy(np.array(self.features[self.feature_indices[position]], copy=True)),
            "time_conditions": torch.from_numpy(self.times[position].copy()),
        }

    def __getitem__(self, position):
        return {**self.inputs(position), "targets": torch.from_numpy(self.targets[position].copy())}

    def __len__(self):
        return len(self.selection)
