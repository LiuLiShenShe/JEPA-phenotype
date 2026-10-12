"""Strict Sugar4D metadata, point-cloud and three-observation forecasting datasets."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from m3.common import TARGETS, TIME_NAMES, read_ply, xyz_from_vertices


def _find_dataset_root(path: str | Path) -> Path:
    root = Path(path)
    if (root / "point_clouds").is_dir():
        return root
    candidates = list(root.rglob("point_clouds"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one Sugar4D point_clouds directory under {root}")
    return candidates[0].parent


def _split_names(dataset_root: Path) -> dict[str, set[int]]:
    result = {}
    for split in ("train", "val", "test"):
        names = (dataset_root / "point_clouds" / "splits" / f"{split}.txt").read_text().splitlines()
        result[split] = {int(name.rsplit("_", 1)[-1].split(".")[0]) for name in names if name.strip()}
        if len(result[split]) * 16 != len([name for name in names if name.strip()]):
            raise ValueError(f"{split} split does not contain exactly 16 observations per plant")
    if set().union(*result.values()) != set(range(1, 49)) or sum(map(len, result.values())) != 48:
        raise ValueError("Official Sugar4D split is not a disjoint partition of plants 1..48")
    if set(result["train"]) & set(result["val"]) or set(result["train"]) & set(result["test"]) or set(result["val"]) & set(result["test"]):
        raise ValueError("Official Sugar4D split leaks plants")
    return result


def _load_genotypes(dataset_root: Path, mapping_path: str | Path | None = None) -> dict[int, str]:
    if mapping_path is not None and Path(mapping_path).exists():
        payload = json.loads(Path(mapping_path).read_text())
        mapping = {int(key): str(value) for key, value in payload.items()}
        if set(mapping) == set(range(1, 49)):
            return mapping
    candidate = dataset_root / "supplements" / "genotype_mapping.json"
    if candidate.exists():
        payload = json.loads(candidate.read_text())
        mapping = {int(key): str(value) for key, value in payload.items()}
        if set(mapping) == set(range(1, 49)):
            return mapping
    return {plant: "unknown" for plant in range(1, 49)}


def _target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["DAS"] = result["Days after sowing"].astype(int)
    result["plant_id"] = result["Plant ID"].astype(int)
    result["plant_height"] = result["Height Maximum [m]"].astype(float) * 100.0
    result["plant_width"] = result["Width Major Axis [m]"].astype(float) * 100.0
    result["visible_organ_count"] = result["Leaf Quantity"].astype(float)
    return result


def audit_dataset(dataset_root: str | Path, *, mapping_path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    root = _find_dataset_root(dataset_root)
    measurements = _target_columns(pd.read_csv(root / "measurements" / "plant_related.csv"))
    leaves = pd.read_csv(root / "measurements" / "leaf_related.csv")
    splits = _split_names(root)
    genotypes = _load_genotypes(root, mapping_path)
    rows = []
    ply_errors = []
    for record in measurements.itertuples(index=False):
        plant_id = int(record.plant_id)
        das = int(record.DAS)
        path = root / "point_clouds" / f"{das}_das" / f"{das}_das_plant_{plant_id:03d}.ply"
        semantic_status = "missing"
        instance_status = "missing"
        point_count = 0
        try:
            vertices, info = read_ply(path)
            point_count = int(info["vertex_count"])
            properties = set(info["properties"])
            semantic_status = "present" if "label_semantic" in properties else "missing"
            instance_status = "present" if "label_instance" in properties else "missing"
            if not {"x", "y", "z"}.issubset(properties):
                raise ValueError("missing xyz")
            xyz = xyz_from_vertices(vertices)
            if len(xyz) == 0 or not np.isfinite(xyz).all():
                raise ValueError("empty or non-finite xyz")
            if semantic_status == "present":
                semantic = np.asarray(vertices["label_semantic"])
                if not np.isin(semantic, (0, 1, 2)).all():
                    raise ValueError("unexpected semantic labels")
            if instance_status == "present":
                instance = np.asarray(vertices["label_instance"])
                if not np.isin(instance, np.arange(-2, 20)).all():
                    raise ValueError("unexpected instance labels")
            quality = "valid"
        except Exception as error:
            quality = "invalid"
            ply_errors.append({"plant_uid": f"plant_{plant_id:03d}", "DAS": das, "path": str(path), "error": repr(error)})
        split = next(name for name, plants in splits.items() if plant_id in plants)
        rows.append({
            "plant_uid": f"plant_{plant_id:03d}", "plant_id": plant_id, "genotype": genotypes[plant_id], "timepoint": f"{das}_das", "DAS": das,
            "pointcloud_path": str(path), "point_count": point_count, "semantic_label_status": semantic_status,
            "instance_label_status": instance_status, "plant_height": float(record.plant_height), "plant_width": float(record.plant_width),
            "visible_organ_count": float(record.visible_organ_count), "quality_status": quality, "split": split,
            "plant_height_source": "Height Maximum [m] * 100 -> cm", "plant_width_source": "Width Major Axis [m] * 100 -> cm",
            "visible_organ_count_source": "Leaf Quantity (automatic plant_related.csv)",
        })
    metadata = pd.DataFrame(rows).sort_values(["plant_id", "DAS"]).reset_index(drop=True)
    leaf_frame = leaves.rename(columns={"days after sowing": "DAS", "plant": "plant_id", "instance id": "leaf_instance_id"}).copy()
    leaf_frame["plant_uid"] = leaf_frame["plant_id"].map(lambda value: f"plant_{int(value):03d}")
    leaf_frame["genotype"] = leaf_frame["plant_id"].map(genotypes)
    leaf_frame["observation_status"] = "observed"
    leaf_frame = leaf_frame.sort_values(["plant_id", "leaf_instance_id", "DAS"]).reset_index(drop=True)
    instance_counts = leaf_frame.groupby(["plant_uid", "leaf_instance_id"]).DAS.nunique()
    reappearance = 0
    for (_, _), group in leaf_frame.groupby(["plant_uid", "leaf_instance_id"]):
        values = sorted(group.DAS.unique())
        if len(values) > 2 and any(right - left > 7 for left, right in zip(values, values[1:])):
            reappearance += 1
    report = {
        "dataset_root": str(root), "plants": int(metadata.plant_uid.nunique()), "observations": int(len(metadata)),
        "timepoints": sorted(metadata.DAS.unique().tolist()), "genotypes": sorted(metadata.genotype.unique().tolist()),
        "point_count": {"min": int(metadata.point_count.min()), "median": float(metadata.point_count.median()), "max": int(metadata.point_count.max())},
        "semantic_labels": metadata.semantic_label_status.value_counts().to_dict(), "instance_labels": metadata.instance_label_status.value_counts().to_dict(),
        "quality_status": metadata.quality_status.value_counts().to_dict(), "leaf_rows": int(len(leaf_frame)),
        "unique_leaf_plant_instance": int(len(instance_counts)), "leaf_instance_reappearance_candidate_count": int(reappearance),
        "manual_leaf_reference_rows": int(len(pd.read_csv(root / "measurements" / "reference" / "leaf_related_manual.csv"))),
        "ply_errors": ply_errors,
        "genotype_mapping_complete": all(value != "unknown" for value in genotypes.values()),
        "units": {"coordinates": "metres (per released m-valued morphology and physical coordinate system; verified from PLY/header and reference files)", "plant_height": "cm", "plant_width": "cm", "leaf_count": "count"},
    }
    return metadata, leaf_frame, report


def build_windows(metadata: pd.DataFrame, *, horizons=(1, 2), history=3, split_column="split") -> pd.DataFrame:
    if history != 3 or metadata.duplicated(["plant_uid", "DAS"]).any():
        raise ValueError("Three distinct historical observations are required")
    if metadata.groupby("plant_uid")[split_column].nunique().max() != 1:
        raise ValueError("A plant crosses split boundaries")
    rows = []
    for plant_uid, group in metadata.sort_values(["plant_uid", "DAS"]).groupby("plant_uid", sort=True):
        group = group.reset_index(drop=True)
        if len(group) != 16 or not group.DAS.is_monotonic_increasing or not (np.diff(group.DAS) > 0).all():
            raise ValueError(f"Expected 16 chronological observations for {plant_uid}")
        for target_offset in horizons:
            for latest_index in range(history - 1, len(group) - target_offset):
                history_rows = group.iloc[latest_index - history + 1: latest_index + 1]
                target = group.iloc[latest_index + target_offset]
                if not history_rows.quality_status.eq("valid").all() or target.quality_status != "valid":
                    raise ValueError("Invalid observations must be audited, not silently removed")
                if target[split_column] != history_rows[split_column].iloc[0] or not (history_rows[split_column] == target[split_column]).all():
                    raise ValueError("A window crosses a plant split")
                row = {"window_id": f"{plant_uid}_h{target_offset}_i{latest_index:02d}", "plant_uid": plant_uid, "plant_id": int(target.plant_id),
                       "genotype": target.genotype, "split": target[split_column], "horizon": target_offset,
                       "history_DAS": json.dumps(history_rows.DAS.astype(int).tolist()), "history_paths": json.dumps(history_rows.pointcloud_path.tolist()),
                       "history_deltas_days": json.dumps(np.diff(history_rows.DAS).astype(int).tolist()),
                       "target_DAS": int(target.DAS), "target_path": target.pointcloud_path, "forecast_delta_days": int(target.DAS - history_rows.DAS.iloc[-1]),
                       "target_plant_height": float(target.plant_height), "target_plant_width": float(target.plant_width), "target_visible_organ_count": float(target.visible_organ_count),
                       "history_plant_height": json.dumps(history_rows.plant_height.astype(float).tolist()), "history_plant_width": json.dumps(history_rows.plant_width.astype(float).tolist()),
                       "history_visible_organ_count": json.dumps(history_rows.visible_organ_count.astype(float).tolist())}
                rows.append(row)
    return pd.DataFrame(rows)


class Sugar4DDataset(Dataset):
    def __init__(self, windows: pd.DataFrame, *, points=2048, cache_root: str | Path | None = None, sampling_seed=20261011,
                 training=False, metadata: pd.DataFrame | None = None):
        self.windows = windows.reset_index(drop=True).copy()
        self.points = int(points)
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.sampling_seed = int(sampling_seed)
        self.training = training
        self.epoch = 0
        self.clouds = {}
        self.anchors = {}
        self.validate(metadata)
        for path in sorted({path for value in self.windows.history_paths for path in json.loads(value)}):
            source = Path(path)
            if self.cache_root is not None:
                cache_path = self.cache_root / (source.stem + ".npy")
                if not cache_path.exists():
                    raise FileNotFoundError(cache_path)
                xyz = np.load(cache_path, allow_pickle=False)
            else:
                vertices, _ = read_ply(source)
                xyz = xyz_from_vertices(vertices)
                plant = source.stem.rsplit("_", 1)[-1]
                if plant not in self.anchors:
                    first_path = source.parent.parent / "37_das" / f"37_das_plant_{plant}.ply"
                    first_vertices, _ = read_ply(first_path)
                    first_xyz = xyz_from_vertices(first_vertices)
                    self.anchors[plant] = np.asarray([np.median(first_xyz[:, 0]), np.median(first_xyz[:, 1]), 0], dtype=np.float32)
                xyz -= self.anchors[plant]
            if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
                raise ValueError("Invalid XYZ cache")
            self.clouds[path] = xyz.astype(np.float32, copy=False)

    def validate(self, metadata=None):
        if self.points < 1 or self.windows.empty or self.windows.window_id.duplicated().any():
            raise ValueError("Invalid point count or window identities")
        if self.windows.groupby("plant_uid").split.nunique().max() != 1:
            raise ValueError("Plant leakage")
        lookup = metadata.set_index(["plant_uid", "DAS"]) if metadata is not None else None
        for row in self.windows.itertuples(index=False):
            paths = json.loads(row.history_paths)
            days = json.loads(row.history_DAS)
            if len(paths) != 3 or len(days) != 3 or not (np.diff(days) > 0).all():
                raise ValueError("History must contain exactly three chronological real scans")
            if row.horizon not in (1, 2) or row.target_DAS <= days[-1] or row.forecast_delta_days != row.target_DAS - days[-1]:
                raise ValueError("Invalid actual forecast time")
            if hasattr(row, "history_deltas_days") and json.loads(row.history_deltas_days) != np.diff(days).tolist():
                raise ValueError("History gaps disagree with DAS")
            for path, das in zip(paths, days):
                if Path(path).name != f"{das}_das_{row.plant_uid}.ply":
                    raise ValueError("Foreign plant or future cloud in history")
                if lookup is not None:
                    actual = lookup.loc[(row.plant_uid, das)]
                    if str(actual.pointcloud_path) != path or actual.split != row.split or actual.quality_status != "valid":
                        raise ValueError("History disagrees with audited metadata")
            if lookup is not None:
                actual = lookup.loc[(row.plant_uid, row.target_DAS)]
                if actual.split != row.split or actual.quality_status != "valid":
                    raise ValueError("Future target disagrees with metadata")
                for target in TARGETS:
                    if not np.isclose(actual[target], getattr(row, f"target_{target}"), rtol=0, atol=1e-6):
                        raise ValueError("Future target labels disagree")
            labels = [getattr(row, f"target_{target}") for target in TARGETS]
            if not np.isfinite(labels).all():
                raise ValueError("Nonfinite supervision")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.windows)

    def _load(self, path, row_index, position):
        xyz = self.clouds[path]
        token = f"{self.sampling_seed}:{self.epoch if self.training else 0}:{Path(path).stem}"
        stable_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
        generator = np.random.default_rng(stable_seed)
        indices = generator.choice(len(xyz), self.points, replace=len(xyz) < self.points)
        return xyz[indices].astype(np.float32, copy=False)

    def __getitem__(self, index):
        row = self.windows.iloc[index]
        paths = json.loads(row.history_paths)
        das = np.asarray(json.loads(row.history_DAS), dtype=np.float32)
        points = np.stack([self._load(path, index, position) for position, path in enumerate(paths)])
        target_das = float(row.target_DAS)
        gaps = np.diff(das)
        time_conditions = np.asarray([das[0] / 100.0, das[1] / 100.0, das[2] / 100.0, gaps[0] / 10.0, gaps[1] / 10.0,
                                      (target_das - das[-1]) / 10.0, target_das / 100.0], dtype=np.float32)
        target = np.asarray([row.target_plant_height, row.target_plant_width, row.target_visible_organ_count], dtype=np.float32)
        return {"points": torch.from_numpy(points), "time_conditions": torch.from_numpy(time_conditions), "target": torch.from_numpy(target),
                "window_id": row.window_id, "plant_uid": row.plant_uid, "genotype": row.genotype, "split": row.split, "horizon": int(row.horizon),
                "history_DAS": tuple(int(value) for value in das), "target_DAS": int(row.target_DAS), "forecast_delta_days": int(row.forecast_delta_days)}
