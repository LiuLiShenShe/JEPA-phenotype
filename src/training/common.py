"""Shared artifact, normalization and reproducibility contracts."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


TARGET_NAMES = ("leaf_count", "projected_area_ratio")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path: str | Path, payload) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def seed_everything(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def require_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
    return device


def clip_predictions(predictions: np.ndarray) -> np.ndarray:
    result = np.array(predictions, dtype=np.float64, copy=True)
    result[:, 0] = np.maximum(result[:, 0], 0)
    result[:, 1] = np.clip(result[:, 1], 0, 1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite predictions")
    return result


class TrainStandardizer:
    def __init__(self, mean, scale, provenance: dict):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.provenance = provenance
        if provenance["split"] != "train" or np.any(self.scale <= 0):
            raise ValueError("Standardizer must be train-fitted with positive scales")

    @classmethod
    def fit(cls, values: np.ndarray, rows):
        if len(rows) != len(values) or set(rows["split"]) != {"train"}:
            raise ValueError("Only training rows may fit a standardizer")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite training values")
        return cls(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-6), {
            "split": "train", "windows": len(rows),
            "plants": int(rows.plant_uid.nunique()),
            "window_ids_sha256": object_hash(sorted(rows.window_id.tolist())),
        })

    def transform(self, values):
        return (values - self.mean) / self.scale

    def inverse(self, values):
        return values * self.scale + self.mean

    def state(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "provenance": self.provenance}

    @classmethod
    def from_state(cls, state):
        return cls(**state)
