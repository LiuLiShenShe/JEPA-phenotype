#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from m4.common import state_hash, write_json
from m4.downstream import run_count, run_segmentation


def repair(root, device):
    root = Path(root)
    issues, checked = [], 0
    for path in sorted((root / "outputs/m4/checkpoints").glob("*/*/training_state.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        metadata, saved = checkpoint["metadata"], checkpoint.get("model")
        issue = not saved
        if saved:
            if metadata["mode"] == "probe":
                head = saved.get("head", saved)
            else:
                head = {name.removeprefix("head."): value for name, value in saved.items() if name.startswith("head.")}
            if state_hash(head) == metadata["initial_head_hash"]:
                issue = True
            if metadata["mode"] == "probe" and metadata["best_epoch"] < metadata["epochs"]:
                best = torch.load(path.parent / "best.pt", map_location="cpu", weights_only=False)
                if state_hash(head) == state_hash(best["head"]):
                    issue = True
        checked += 1
        if issue:
            issues.append((path, metadata))
    repaired = []
    for path, record in issues:
        if record["task"] == "count":
            replay = run_count(root, record["partition"], record["seed"], record["condition"], record["label_fraction"],
                               record["mode"], device, replay_state=True)
            replay_folder = root / "outputs/m4/state_replays" / record["id"]
            expected = pd.read_csv(root / "outputs/m4/predictions" / f"{record['id']}.csv").prediction.to_numpy()
            actual = pd.read_csv(replay_folder / "predictions.csv").prediction.to_numpy()
            if not np.allclose(expected, actual, rtol=0, atol=2e-6):
                raise ValueError("State replay changed selected scientific count predictions; refusing overwrite")
            comparison = {"maximum_prediction_difference": float(np.abs(expected - actual).max())}
        else:
            replay = run_segmentation(root, record["partition"], record["seed"], record["condition"], record["mode"], device,
                                      replay_state=True)
            replay_folder = root / "outputs/m4/state_replays" / record["id"]
            with np.load(root / "outputs/m4/predictions" / f"{record['id']}_points.npz") as expected, np.load(replay_folder / "points.npz") as actual:
                if not np.array_equal(expected["prediction"], actual["prediction"]):
                    raise ValueError("State replay changed selected scientific segmentation predictions")
            comparison = {"exact_selected_semantic_predictions": True}
        if replay["best_epoch"] != record["best_epoch"]:
            raise ValueError("Deterministic replay altered model selection")
        shutil.copyfile(path, replay_folder / "original_training_state.pt")
        shutil.copyfile(replay_folder / "training_state.pt", path)
        repaired.append({"id": record["id"], "reason": "early engineering snapshot lacked or mismatched final weights",
                         "scientific_predictions_unchanged": True, "seconds": replay["seconds"], **comparison})
        write_json(root / "reports/m4/state_capture_repairs.json", {"status": "PASS", "checked_checkpoints": checked,
                                                                   "repaired": repaired, "original_best_and_predictions_overwritten": False})
    if not repaired:
        write_json(root / "reports/m4/state_capture_repairs.json", {"status": "PASS", "checked_checkpoints": checked,
                                                                   "repaired": [], "original_best_and_predictions_overwritten": False})
    return repaired


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    repair(Path(arguments.root), torch.device(arguments.device))
