from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from m4.common import standardize


IDENTITY_COLUMNS = ["plant_uid", "DAS", "pointcloud_path"]


def partition_rows(root, partition, split):
    if split not in ("train", "val"):
        raise ValueError("M4 training and evaluation do not open test observations")
    root = Path(root)
    assignment = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"][partition]
    frame = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    frame["split"] = frame.plant_uid.map(assignment)
    frame = frame.loc[frame.split.eq(split)].reset_index(drop=True)
    if frame.duplicated(["plant_uid", "DAS"]).any() or len(frame) != frame.plant_uid.nunique() * 16:
        raise ValueError("Expected 16 current observations per independent plant")
    return frame


def labelled_plants(plants, fraction, partition, seed=20261012):
    if fraction not in (0.25, 0.5, 1.0):
        raise ValueError("Unregistered label budget")
    plants = np.asarray(sorted(set(plants)))
    digest = int.from_bytes(hashlib.sha256(f"{seed}:{partition}".encode()).digest()[:8], "little")
    permutation = np.random.default_rng(digest).permutation(plants)
    count = max(1, int(round(len(plants) * fraction)))
    return sorted(permutation[:count].tolist())


def current_transform(xyz, center=None, radius=None):
    xyz = np.asarray(xyz, dtype=np.float32)
    center = xyz.mean(axis=0, dtype=np.float64).astype(np.float32) if center is None else np.asarray(center, dtype=np.float32)
    radius = float(np.linalg.norm(xyz - center, axis=1).max()) if radius is None else float(radius)
    if not np.isfinite(xyz).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid current-scan geometry")
    return (xyz - center) / radius, center, radius


class Sugar4DSSLDataset(Dataset):
    def __init__(self, rows, cache_root, training=True):
        if not len(rows) or set(rows.split) != {"train"}:
            raise ValueError("SSL must contain only this partition's training plants")
        if rows.duplicated(["plant_uid", "DAS"]).any():
            raise ValueError("Duplicate current scans")
        self.rows = rows[IDENTITY_COLUMNS].reset_index(drop=True).copy()
        self.cache_root, self.training = Path(cache_root), training
        self.clouds = []
        for row in self.rows.itertuples(index=False):
            with np.load(self.cache_root / f"{Path(row.pointcloud_path).stem}.npz", allow_pickle=False) as archive:
                self.clouds.append({name: archive[name].copy() for name in ("points", "center_indices", "neighbors", "center_m", "radius_m", "extent_m")})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = self.clouds[index]
        return {name: torch.from_numpy(value.copy()) for name, value in sample.items()} | {"index": index}


class Sugar4DLabelledDataset(Dataset):
    def __init__(self, rows, cache_root, label_root, geometry=None, geometry_scaler=None):
        if not len(rows) or not set(rows.split).issubset({"train", "val"}):
            raise ValueError("Only train/validation downstream observations are admitted")
        self.rows, self.clouds, self.labels = rows.reset_index(drop=True).copy(), [], []
        for row in self.rows.itertuples(index=False):
            name = Path(row.pointcloud_path).stem
            with np.load(Path(cache_root) / f"{name}.npz", allow_pickle=False) as archive:
                self.clouds.append({key: archive[key].copy() for key in ("points", "center_indices", "neighbors")})
            with np.load(Path(label_root) / f"{name}.npz", allow_pickle=False) as archive:
                self.labels.append(archive["semantic"].copy())
        self.geometry = None if geometry is None else standardize(geometry, geometry_scaler)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        sample = {key: torch.from_numpy(value.copy()) for key, value in self.clouds[index].items()}
        sample.update({"semantic": torch.from_numpy(self.labels[index].astype(np.int64)),
                       "count": torch.tensor(float(self.rows.iloc[index].visible_organ_count), dtype=torch.float32),
                       "index": index})
        if self.geometry is not None:
            sample["geometry"] = torch.from_numpy(self.geometry[index])
        return sample
