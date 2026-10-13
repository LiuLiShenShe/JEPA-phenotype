from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from m3.common import read_ply
from m6a.common import write_json
from m6a.experiment import metadata


def read_frame(path):
    vertices, info = read_ply(path)
    xyz = np.column_stack([vertices[name] for name in ["x", "y", "z"]]).astype(np.float32)
    semantic = vertices["label_semantic"].astype(np.int64)
    original_instance = vertices["label_instance"].astype(np.int64)
    if not np.isfinite(xyz).all() or not set(np.unique(semantic)).issubset({0, 1, 2}):
        raise ValueError("Invalid Sugar4D geometry/semantic labels")
    leaf = (semantic == 2) & (original_instance >= 0)
    if np.any((semantic == 2) & (original_instance < 0)):
        raise ValueError("Leaf with invalid negative instance; do not silently delete")
    local = np.full(len(xyz), -1, dtype=np.int64)
    leaf_ids, inverse = np.unique(original_instance[leaf], return_inverse=True)
    local[leaf] = inverse
    return {"xyz": xyz, "semantic": semantic, "local_instance": local,
            "original_instance_evaluation_only": original_instance, "leaf_original_ids_evaluation_only": leaf_ids,
            "info": info}


def transform(xyz):
    minimum, maximum = xyz.min(axis=0), xyz.max(axis=0)
    center = (minimum + maximum) / 2
    scale = max(float((maximum - minimum).max()) / 2, 1e-8)
    return ((xyz - center) / scale).astype(np.float32), center, scale


def restore(normalized, center, scale):
    return np.asarray(normalized) * scale + center


def cache_file(root, plant, das):
    return root / f"outputs/m6_b/data/{plant}_{das}.npz"


def prepare_training_cache(root):
    records = metadata(root).query("split == 'train'")
    protocol = json.loads((root / "configs/m6_b/protocol.json").read_text())
    pool_size = protocol["sampling"]["training_pool_points"]
    full_records = metadata(root).reset_index(drop=True)
    for record in records.itertuples(index=False):
        path = cache_file(root, record.plant_uid, record.DAS)
        if path.exists():
            continue
        data = read_frame(root / record.pointcloud_path)
        normalized, center, scale = transform(data["xyz"])
        record_index = full_records.index[(full_records.plant_uid == record.plant_uid) & (full_records.DAS == record.DAS)][0]
        generator = np.random.default_rng(20261013 + record_index)
        indices = generator.choice(len(normalized), min(pool_size, len(normalized)), replace=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".ready.npz")
        np.savez(temporary, xyz=normalized[indices], semantic=data["semantic"][indices],
                 local_instance=data["local_instance"][indices], raw_indices=indices,
                 center_m=center, scale_m=scale, full_points=len(normalized))
        temporary.replace(path)
    write_json(root / "reports/m6_b/training_cache_ready.json", {"train_frames": len(records), "train_plants": records.plant_uid.nunique(), "no_validation_test_training": True})


def prepare_data(root):
    records = metadata(root)
    protocol = json.loads((root / "configs/m6_b/protocol.json").read_text())
    pool_size = protocol["sampling"]["training_pool_points"]
    training_leaf_sizes = []
    audit = []
    for index, record in enumerate(records.itertuples(index=False)):
        data = read_frame(root / record.pointcloud_path)
        normalized, center, scale = transform(data["xyz"])
        generator = np.random.default_rng(20261013 + index)
        indices = generator.choice(len(normalized), min(pool_size, len(normalized)), replace=False)
        local = data["local_instance"]
        ids, sizes = np.unique(local[local >= 0], return_counts=True)
        core = data["semantic"] == 1
        distances = cKDTree(data["xyz"][core]).query(data["xyz"], workers=4)[0] if core.any() else np.full(len(local), np.inf)
        leaf_rows = []
        for leaf_id, size in zip(ids, sizes):
            mask = local == leaf_id
            leaf_rows.append({"plant_uid": record.plant_uid, "split": record.split, "DAS": int(record.DAS),
                              "local_id": int(leaf_id), "points": int(size), "point_fraction": float(size / len(local)),
                              "core_distance_p10_m": float(np.quantile(distances[mask], .1)) if core.any() else None,
                              "extent_m": float(np.ptp(data["xyz"][mask], axis=0).max()),
                              "pool_sample_points": int(np.sum(local[indices] == leaf_id))})
            if record.split == "train":
                training_leaf_sizes.append(size / len(local))
        path = cache_file(root, record.plant_uid, record.DAS)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            np.savez(path, xyz=normalized[indices], semantic=data["semantic"][indices], local_instance=local[indices],
                     raw_indices=indices, center_m=center, scale_m=scale, full_points=len(normalized))
        audit.append({"plant_uid": record.plant_uid, "split": record.split, "DAS": int(record.DAS),
                      "points": len(local), "taproot_points": int(np.sum(data["semantic"] == 0)),
                      "core_points": int(np.sum(core)), "leaf_points": int(np.sum(data["semantic"] == 2)),
                      "negative_nonleaf_instances": int(np.sum((local < 0))), "leaf_instances": len(ids),
                      "raw_semantic_instance_pairs": json.dumps(np.unique(np.column_stack([data["semantic"], data["original_instance_evaluation_only"]]), axis=0).tolist()),
                      "pool_lost_leaves": sum(row["pool_sample_points"] == 0 for row in leaf_rows)})
        pd.DataFrame(leaf_rows).to_csv(path.with_suffix(".csv"), index=False)
        if (index + 1) % 32 == 0:
            print("DATA", index + 1, "/", len(records), flush=True)
    pd.DataFrame(audit).to_csv(root / "reports/m6_b/data_frames.csv", index=False)
    leaves = pd.concat([pd.read_csv(path) for path in (root / "outputs/m6_b/data").glob("*.csv")], ignore_index=True)
    leaves.to_csv(root / "reports/m6_b/data_leaves.csv", index=False)
    write_json(root / "reports/m6_b/data_audit.json", {
        "frames": len(records), "plants": records.plant_uid.nunique(), "split_plants": records.groupby("split").plant_uid.nunique().to_dict(),
        "split_frames": records.groupby("split").size().to_dict(), "coordinate_unit": "metres; retained centre and scale in metres",
        "semantic_mapping": {"0": "taproot", "1": "core", "2": "leaf"},
        "nonleaf_instance_policy": "negative IDs never become leaves; all taproot/core points have separate semantic supervision; no leaf embedding loss",
        "leaf_policy": "frame-local contiguous 0..K-1; raw temporal identities excluded from model and cache inputs",
        "small_leaf_fraction_threshold_train_q25": float(np.quantile(training_leaf_sizes, .25)),
        "pool_size": pool_size, "pool_lost_leaves": int(pd.DataFrame(audit).pool_lost_leaves.sum()),
        "sampling_uses_labels": False, "future_inputs": False})


class TrainingDataset:
    def __init__(self, root, seed, points=4096):
        self.root = root
        self.seed = seed
        self.points = points
        self.epoch = 0
        self.records = metadata(root).query("split == 'train'").reset_index(drop=True)
        self.arrays = []
        for record in self.records.itertuples(index=False):
            with np.load(cache_file(root, record.plant_uid, record.DAS)) as data:
                self.arrays.append({name: data[name].copy() for name in ["xyz", "semantic", "local_instance"]})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        data = self.arrays[index]
        generator = np.random.default_rng(self.seed * 1000003 + self.epoch * 1009 + index)
        indices = generator.choice(len(data["xyz"]), self.points, replace=len(data["xyz"]) < self.points)
        xyz = data["xyz"][indices].copy()
        angle = generator.uniform(-np.pi, np.pi)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=np.float32)
        xyz = xyz @ rotation.T
        local = data["local_instance"][indices].copy()
        ids = np.unique(local[local >= 0])
        relabeled = np.full(len(local), -1, dtype=np.int64)
        permutation = generator.permutation(len(ids))
        for source, target in zip(ids, permutation):
            relabeled[local == source] = target
        return xyz, data["semantic"][indices], relabeled
