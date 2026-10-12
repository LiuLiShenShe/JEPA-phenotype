#!/usr/bin/env python
"""Reload official checkpoints and verify deterministic real-data forecasts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m3.common import TARGETS, seed_everything, write_json
from train_m3 import build_model, loader, predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    protocol = json.loads((ROOT / "configs/m3/protocol.json").read_text())
    windows = pd.read_csv(ROOT / "data/processed/sugar4d/forecast_windows.csv")
    windows = windows.loc[windows.split.eq("test")]
    cache = ROOT / "data/processed/sugar4d/point_cache"
    records = []
    for point_count in (2048, 4096):
        directory = ROOT / "outputs/m3/official" / f"points{point_count}"
        for model_name in ("B3-2", "B3-3", "B3-4"):
            for seed in protocol["seeds"]:
                seed_everything(seed)
                checkpoint = torch.load(directory / f"{model_name}_seed{seed}_points{point_count}.pt", map_location="cpu", weights_only=True)
                model = build_model(model_name, protocol).to(args.device)
                model.load_state_dict(checkpoint["model"])
                batches = loader(windows, point_count, seed, protocol["batch_size"], False, cache)
                expected = pd.read_csv(directory / f"{model_name}_seed{seed}_test_predictions.csv").set_index("window_id")
                first = predict(model, batches, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["scale"]), args.device).set_index("window_id")
                second = predict(model, batches, np.asarray(checkpoint["mean"]), np.asarray(checkpoint["scale"]), args.device).set_index("window_id")
                columns = [f"pred_{target}" for target in TARGETS]
                expected = expected.reindex(first.index)
                replay_error = float(np.abs(first[columns].to_numpy() - expected[columns].to_numpy()).max())
                repeat_error = float(np.abs(first[columns].to_numpy() - second[columns].to_numpy()).max())
                if replay_error > 1e-5 or repeat_error > 0:
                    raise ValueError(f"Inference repeatability failed: {model_name} {seed} {point_count} {replay_error} {repeat_error}")
                records.append({"model": model_name, "seed": seed, "points": point_count, "windows": len(first), "replay_max_abs_difference": replay_error, "repeat_max_abs_difference": repeat_error})
                del model, batches
                torch.cuda.empty_cache()
    write_json(ROOT / "reports/m3/inference_repeatability.json", {"status": "PASS", "device": args.device, "runs": records})
    print(f"PASS: {len(records)} checkpoint replays; all repeated inference outputs identical")


if __name__ == "__main__":
    main()
