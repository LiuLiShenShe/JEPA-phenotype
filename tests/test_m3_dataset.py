import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from datasets.sugar4d_dataset import Sugar4DDataset, build_windows
from m3.common import read_ply


def fixture_project(tmp_path):
    rows = []
    days = [37, 40, 44, 47, 51, 54, 58, 61, 65, 72, 75, 79, 82, 86, 89, 93]
    cache = tmp_path / "cache"
    cache.mkdir()
    for index, day in enumerate(days):
        name = f"{day}_das_plant_001.ply"
        np.save(cache / (Path(name).stem + ".npy"), np.arange(96, dtype=np.float32).reshape(32, 3) / 100 + index / 100)
        rows.append({"plant_uid": "plant_001", "plant_id": 1, "genotype": "G1", "DAS": day, "pointcloud_path": str(tmp_path / name),
                     "split": "train", "quality_status": "valid", "plant_height": day / 2, "plant_width": day, "visible_organ_count": index + 2})
    metadata = pd.DataFrame(rows)
    return metadata, build_windows(metadata), cache


def test_binary_ply_decoder_and_truncation(tmp_path):
    data = np.zeros(2, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("label_instance", "<i4")])
    data["z"] = [0.1, 0.3]
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty int label_instance\nend_header\n"
    path = tmp_path / "tiny.ply"
    path.write_bytes(header + data.tobytes())
    vertices, info = read_ply(path)
    assert info["vertex_count"] == 2
    np.testing.assert_allclose(vertices["z"], [0.1, 0.3])
    with pytest.raises(ValueError, match="read-only"):
        vertices["z"][0] = 5
    path.write_bytes(header + data.tobytes()[:5])
    with pytest.raises(ValueError, match="Truncated"):
        read_ply(path)


def test_actual_das_windows(tmp_path):
    metadata, windows, cache = fixture_project(tmp_path)
    assert len(windows) == 25
    long_gap = windows.loc[windows.target_DAS.eq(72) & windows.horizon.eq(1)].iloc[0]
    assert long_gap.forecast_delta_days == 7
    assert json.loads(long_gap.history_DAS) == [58, 61, 65]
    dataset = Sugar4DDataset(windows, cache_root=cache, points=16, metadata=metadata)
    batch = next(iter(DataLoader(dataset, batch_size=4)))
    assert batch["points"].shape == (4, 3, 16, 3)
    assert batch["time_conditions"].shape == (4, 7)
    assert batch["target"].shape == (4, 3)
    assert not any("instance" in key or "semantic" in key or key == "target_path" for key in batch)


def test_sampling_stable_across_window_order_and_nested_points(tmp_path):
    metadata, windows, cache = fixture_project(tmp_path)
    first = Sugar4DDataset(windows, cache_root=cache, points=16)
    second = Sugar4DDataset(windows.iloc[::-1], cache_root=cache, points=16)
    torch.testing.assert_close(first[0]["points"], second[len(second) - 1]["points"])
    first.training = True
    unchanged = first[0]["points"].clone()
    first.set_epoch(1)
    assert not torch.equal(unchanged, first[0]["points"])
    assert first[0]["points"].abs().max() < 1.2


def test_foreign_future_or_cross_plant_inputs_rejected(tmp_path):
    metadata, windows, cache = fixture_project(tmp_path)
    corrupt = windows.copy()
    paths = json.loads(corrupt.loc[0, "history_paths"])
    paths[-1] = paths[-1].replace("44_das", "47_das")
    corrupt.loc[0, "history_paths"] = json.dumps(paths)
    with pytest.raises(ValueError, match="Foreign plant or future"):
        Sugar4DDataset(corrupt, cache_root=cache)
    corrupt = windows.copy()
    corrupt.loc[0, "forecast_delta_days"] = 999
    with pytest.raises(ValueError, match="forecast time"):
        Sugar4DDataset(corrupt, cache_root=cache)
    metadata.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="split boundaries"):
        build_windows(metadata)


def test_real_sugar4d_dataset_smoke():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/sugar4d/forecast_windows.csv"
    if not path.exists() or not (path.parent / "point_cache").is_dir():
        pytest.skip("Real dataset not downloaded/prepared")
    metadata = pd.read_csv(path.parent / "plant_metadata.csv")
    windows = pd.read_csv(path)
    assert metadata.plant_uid.nunique() == 48 and len(metadata) == 768
    assert metadata.quality_status.eq("valid").all()
    dataset = Sugar4DDataset(windows.loc[windows.split.eq("test")], cache_root=path.parent / "point_cache", points=2048, metadata=metadata)
    assert len(dataset) == 300
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    assert batch["points"].shape == (2, 3, 2048, 3)
    assert torch.isfinite(batch["points"]).all()
    assert len(windows.loc[windows.horizon.eq(1)]) == 624
    assert len(windows.loc[windows.horizon.eq(2)]) == 576
