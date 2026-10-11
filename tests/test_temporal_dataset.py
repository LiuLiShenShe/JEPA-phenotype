from __future__ import annotations

import torch
import csv
import json

import pytest
from torch.utils.data import DataLoader
from build_temporal_windows import build_windows
from datasets.phenoleaf_utils import write_csv

from datasets.plant_temporal_dataset import PlantTemporalDataset


def test_temporal_dataset_smoke_returns_history_tensor_and_future_target(workflow):
    dataset = PlantTemporalDataset(
        metadata_path=workflow["metadata_dir"] / "frames.csv",
        split_path=workflow["split_path"],
        split="train",
        history_length=3,
        horizon=4,
        windows_path=workflow["windows_path"],
        project_root=workflow["root"],
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["plant_uid"]
    assert sample["images"].shape == (3, 3, 64, 64)
    assert torch.is_floating_point(sample["images"])
    assert len(sample["input_frame_indices"]) == 3
    assert sample["input_frame_indices"][-1] < sample["target_frame_idx"]
    assert sample["forecast_horizon"] == 4
    assert isinstance(sample["target_leaf_count"], int)
    assert isinstance(sample["target_projected_area"], float)


def test_quarantined_frames_do_not_compress_observation_horizons(workflow):
    path = workflow["metadata_dir"] / "frames.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    plant_uid = rows[0]["plant_uid"]
    next(row for row in rows if row["plant_uid"] == plant_uid and row["frame_idx"] == "5")["quality_status"] = "valid_with_warnings"
    write_csv(path, rows, list(rows[0]))
    build_windows(path, workflow["split_path"], workflow["windows_path"], workflow["root"] / "window_report.json")
    with workflow["windows_path"].open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    for row in windows:
        indices = json.loads(row["input_frame_indices"])
        assert int(row["target_frame_idx"]) - indices[-1] == int(row["forecast_horizon"])
        assert indices == list(range(indices[0], indices[0] + 3))
        if row["plant_uid"] == plant_uid:
            assert 5 not in indices and int(row["target_frame_idx"]) != 5


def test_getitem_never_opens_future_rgb_or_mask(workflow, monkeypatch):
    from PIL import Image

    dataset = PlantTemporalDataset(
        workflow["metadata_dir"] / "frames.csv", workflow["split_path"],
        windows_path=workflow["windows_path"], project_root=workflow["root"],
    )
    expected_paths = json.loads(dataset._windows[0]["input_image_paths"])
    opened = []
    original_open = Image.open

    def tracked_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked_open)
    dataset[0]
    assert opened == [str(workflow["root"] / path) for path in expected_paths]


def test_dataloader_batches_resized_history_images(workflow):
    dataset = PlantTemporalDataset(
        workflow["metadata_dir"] / "frames.csv", workflow["split_path"],
        windows_path=workflow["windows_path"], project_root=workflow["root"], image_size=(32, 48),
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["images"].shape == (2, 3, 3, 32, 48)


def test_forged_cross_plant_window_is_rejected(workflow):
    path = workflow["windows_path"]
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in rows if row["split"] == "train" and row["forecast_horizon"] == "4")
    row["input_image_paths"] = json.dumps(["other_plant.png"] * 3)
    write_csv(path, rows, list(rows[0]))
    with pytest.raises(ValueError, match="foreign"):
        PlantTemporalDataset(
            workflow["metadata_dir"] / "frames.csv", workflow["split_path"],
            windows_path=path, project_root=workflow["root"],
        )
