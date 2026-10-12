from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.sugar4d_ssl_dataset import Sugar4DLabelledDataset, partition_rows
from m4.common import artifact_protocol_matches, load_protocol, seed_all, standardize, write_json
from m4.downstream import count_predict, geometry_rows, segmentation_predict, source_encoder
from models.point_jepa_downstream import PointJEPACount, PointJEPASegmentation, PointNetSegmentation


def load_downstream(root, checkpoint_path, device):
    root = Path(root)
    protocol, lock = load_protocol(root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    record = checkpoint["metadata"]
    if not artifact_protocol_matches(root, record["protocol_sha256"], lock) or record["test_used"]:
        raise ValueError("Unverified downstream protocol/selection provenance")
    seed_all(record["seed"])
    frozen = record["mode"] == "probe"
    if record["condition"] == "PointNet":
        model = PointNetSegmentation()
    else:
        encoder, _ = source_encoder(root, protocol, record["partition"], record["seed"], record["condition"])
        model = (PointJEPACount(encoder, len(checkpoint["geometry_columns"]), frozen=frozen)
                 if record["task"] == "count" else PointJEPASegmentation(encoder, frozen=frozen))
    if frozen:
        model.head.load_state_dict(checkpoint["head"], strict=True)
    else:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def verify_prediction_restore(root, checkpoint_path, device):
    import pandas as pd

    root = Path(root)
    protocol, _ = load_protocol(root)
    model, checkpoint = load_downstream(root, checkpoint_path, device)
    record = checkpoint["metadata"]
    rows = partition_rows(root, record["partition"], "val")
    geometry = geometry_rows(root, rows)[0] if record["task"] == "count" else None
    dataset = Sugar4DLabelledDataset(rows, root / "outputs/m4/data/xyz", root / "outputs/m4/data/labels", geometry,
                                     checkpoint.get("geometry_scaler"))
    loader = DataLoader(dataset, batch_size=protocol["downstream"]["batch_size"], shuffle=False)
    if record["task"] == "count":
        if record["mode"] == "probe":
            features = []
            for batch in loader:
                pooled = model.encoder(batch["points"].to(device), batch["center_indices"].to(device), batch["neighbors"].to(device))
                features.append(torch.cat((pooled, batch["geometry"].to(device)), dim=-1).cpu().numpy())
            normalized = standardize(np.concatenate(features), checkpoint["feature_scaler"])
            predicted = model.head(torch.from_numpy(normalized).to(device)).flatten().cpu().numpy()
            values = np.maximum(predicted * checkpoint["target_scaler"]["scale"][0] + checkpoint["target_scaler"]["mean"][0], 0)
        else:
            values = count_predict(model, loader, checkpoint["target_scaler"], device)
        expected = pd.read_csv(root / "outputs/m4/predictions" / f"{record['id']}.csv").prediction.to_numpy()
        deviation = float(np.abs(values - expected).max())
        if not np.allclose(values, expected, rtol=0, atol=2e-6):
            raise ValueError(f"Restored count prediction differs: {record['id']} delta {deviation}")
        result = {"id": record["id"], "task": "count", "maximum_absolute_difference": deviation, "status": "PASS"}
    else:
        matrices, prediction = segmentation_predict(model, loader, device)
        with np.load(root / "outputs/m4/predictions" / f"{record['id']}_points.npz", allow_pickle=False) as saved:
            exact = bool(np.array_equal(prediction, saved["prediction"]) and np.array_equal(matrices, saved["confusion"]))
        if not exact:
            raise ValueError(f"Restored segmentation prediction differs: {record['id']}")
        result = {"id": record["id"], "task": "segmentation", "exact_same_predictions_and_confusion": True, "status": "PASS"}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def verify_matrix_restore(root, device):
    root = Path(root)
    results = []
    for path in sorted((root / "outputs/m4/checkpoints").glob("*/*/best.pt")):
        results.append(verify_prediction_restore(root, path, device))
    write_json(root / "reports/m4/checkpoint_prediction_restore.json", {"status": "PASS", "checkpoints": len(results), "records": results})
    return results
