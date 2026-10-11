"""PyTorch map-style Dataset for future plant phenotype prediction windows."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .phenoleaf_utils import resolve_project_path


class PlantTemporalDataset(Dataset):
    """Load history image tensors and future ground-truth phenotype labels.

    ``temporal_windows.csv`` is intentionally the source of targets.  The
    target RGB image is never opened by ``__getitem__``; this prevents an
    accidental future-image shortcut during baseline or JEPA experiments.
    """

    def __init__(
        self,
        metadata_path: str | Path = "data/metadata/frames.csv",
        split_path: str | Path = "splits/phenoleaf_plant_split_seed42.json",
        split: str = "train",
        history_length: int = 3,
        horizon: int = 4,
        *,
        windows_path: str | Path | None = None,
        transform: Callable[[Image.Image], Any] | None = None,
        project_root: str | Path | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be positive")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.metadata_path = Path(metadata_path)
        self.split_path = Path(split_path)
        self.split = split
        self.history_length = history_length
        self.horizon = horizon
        self.transform = transform
        self.image_size = image_size
        if image_size is not None and (len(image_size) != 2 or min(image_size) < 1):
            raise ValueError("image_size must be a positive (height, width) pair")
        self.project_root = Path(project_root) if project_root is not None else self.metadata_path.resolve().parents[2]
        self.windows_path = Path(windows_path) if windows_path is not None else self.metadata_path.parent.parent / "processed/temporal_windows.csv"
        if not self.windows_path.exists():
            raise FileNotFoundError(
                f"Temporal window index not found: {self.windows_path}. "
                "Run scripts/build_temporal_windows.py first."
            )
        self._windows = self._read_windows()
        self._validate_index()

    def _validate_index(self) -> None:
        payload = json.loads(self.split_path.read_text(encoding="utf-8"))
        assignments = payload.get("assignments", {})
        with self.metadata_path.open(newline="", encoding="utf-8") as handle:
            frames = list(csv.DictReader(handle))
        frame_lookup = {(row["plant_uid"], int(row["frame_idx"])): row for row in frames}
        for row in self._windows:
            indices = self._json_column(row, "input_frame_indices")
            paths = self._json_column(row, "input_image_paths")
            if len(indices) != self.history_length or len(paths) != len(indices):
                raise ValueError("Window history length differs from its schema")
            if any(int(current) != int(previous) + 1 for previous, current in zip(indices, indices[1:])):
                raise ValueError("Window history is not consecutive in original observation indices")
            if int(row["target_frame_idx"]) != int(indices[-1]) + self.horizon:
                raise ValueError("Window horizon differs from original observation index offset")
            if assignments.get(row["plant_uid"]) != self.split:
                raise ValueError("Window plant is not assigned to the requested split")
            for frame_index, path in zip(indices, paths):
                frame = frame_lookup.get((row["plant_uid"], int(frame_index)))
                if frame is None or frame["quality_status"] != "valid" or frame["rgb_path"] != path:
                    raise ValueError("Window contains a foreign, missing, or quarantined history frame")
            target = frame_lookup.get((row["plant_uid"], int(row["target_frame_idx"])))
            if target is None or target["quality_status"] != "valid":
                raise ValueError("Window contains a missing or quarantined target")
            if int(float(row["target_leaf_count"])) != int(target["leaf_count"]):
                raise ValueError("Future leaf-count label disagrees with metadata")
            if float(row["target_projected_area"]) != float(target["projected_area_px"]):
                raise ValueError("Future projected-area label disagrees with metadata")

    def _read_windows(self) -> list[dict[str, str]]:
        with self.windows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        selected = [
            row
            for row in rows
            if row.get("split") == self.split
            and int(row.get("history_length", 0)) == self.history_length
            and int(row.get("forecast_horizon", 0)) == self.horizon
        ]
        selected.sort(key=lambda row: row.get("window_id", ""))
        return selected

    @staticmethod
    def _json_column(row: dict[str, str], key: str) -> list[Any]:
        try:
            value = json.loads(row[key])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON column {key!r} in window {row.get('window_id', '')}") from exc
        if not isinstance(value, list):
            raise ValueError(f"Expected list in {key!r} for window {row.get('window_id', '')}")
        return value

    def _load_image_tensor(self, path_value: str) -> torch.Tensor:
        path = resolve_project_path(path_value, self.project_root)
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.image_size is not None:
                image = image.resize((self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR)
            if self.transform is not None:
                transformed = self.transform(image)
                if torch.is_tensor(transformed):
                    tensor = transformed
                else:
                    array = np.asarray(transformed)
                    tensor = torch.from_numpy(array)
                    if tensor.ndim == 3 and tensor.shape[-1] in {1, 3, 4}:
                        tensor = tensor.permute(2, 0, 1)
            else:
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
        if tensor.ndim != 3:
            raise ValueError(f"Image transform must return a 3D tensor/array, got shape {tuple(tensor.shape)}")
        return tensor.float()

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._windows[index]
        input_paths = [str(value) for value in self._json_column(row, "input_image_paths")]
        input_frame_indices = [int(value) for value in self._json_column(row, "input_frame_indices")]
        input_timestamps = [str(value) for value in self._json_column(row, "input_timestamps")]
        time_deltas = self._json_column(row, "time_deltas")
        if len(input_paths) != self.history_length:
            raise ValueError(f"Window {row.get('window_id', '')} has wrong history length")
        tensors = [self._load_image_tensor(path) for path in input_paths]
        spatial_shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
        if len(spatial_shapes) != 1:
            raise ValueError(f"Window {row.get('window_id', '')} contains different image sizes: {spatial_shapes}")
        images = torch.stack(tensors, dim=0)
        return {
            "plant_uid": row["plant_uid"],
            "images": images,
            "input_images": images,
            "input_image_paths": input_paths,
            "input_frame_indices": input_frame_indices,
            "input_timestamps": input_timestamps,
            "timestamps": input_timestamps,
            "time_deltas": time_deltas,
            "timestamp_reliable": row.get("timestamp_reliable", "False").lower() == "true",
            "target_frame_idx": int(row["target_frame_idx"]),
            "forecast_horizon": int(row["forecast_horizon"]),
            "target_leaf_count": int(float(row["target_leaf_count"])),
            "target_projected_area": float(row["target_projected_area"]),
            "target_projected_area_ratio": float(row.get("target_projected_area_ratio", "nan")),
            "target_growth_stage": row.get("target_growth_stage", ""),
            "target_timestamp": row.get("target_timestamp", ""),
            "target_time_delta_seconds": float(row["target_time_delta_seconds"]) if row.get("target_time_delta_seconds") else float("nan"),
            "split": row["split"],
            "window_id": row.get("window_id", ""),
        }
