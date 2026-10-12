#!/usr/bin/env python
"""Train Sugar4D point-cloud forecasting baselines without test selection."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.sugar4d_dataset import Sugar4DDataset
from m3.common import TARGETS, file_hash, seed_everything, tensor_state_hash, write_json
from models.pointnet_forecaster import PointNetForecaster
from models.pointnet_gru_forecaster import PointNetGRUForecaster


def target_matrix(frame):
    return frame[[f"target_{name}" for name in TARGETS]].to_numpy(dtype=np.float32)


def loader(windows, points, seed, batch_size, shuffle, cache_root):
    dataset = Sugar4DDataset(windows, points=points, sampling_seed=20261011, cache_root=cache_root, training=shuffle)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def standardizer(train_windows):
    values = target_matrix(train_windows)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def build_model(name, protocol):
    if name == "B3-2":
        return PointNetForecaster(protocol["feature_dim"], protocol["hidden_dim"], 7, protocol["dropout"])
    if name not in ("B3-3", "B3-4"):
        raise ValueError("Unknown neural baseline")
    return PointNetGRUForecaster(protocol["feature_dim"], protocol["hidden_dim"], 7, protocol["dropout"], repeated_latest=name == "B3-4")


def epoch(model, batches, optimizer, mean, scale, device):
    model.train()
    loss_sum = 0.0
    count = 0
    criterion = torch.nn.MSELoss()
    for batch in batches:
        points = batch["points"].to(device)
        times = batch["time_conditions"].to(device)
        target = (batch["target"].to(device) - mean) / scale
        optimizer.zero_grad(set_to_none=True)
        prediction = model(points, times)
        loss = criterion(prediction, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(points)
        count += len(points)
    return loss_sum / max(1, count)


@torch.no_grad()
def predict(model, batches, mean, scale, device):
    model.eval()
    records = []
    for batch in batches:
        prediction = model(batch["points"].to(device), batch["time_conditions"].to(device)).cpu().numpy() * scale + mean
        for index in range(len(prediction)):
            record = {"window_id": batch["window_id"][index], "plant_uid": batch["plant_uid"][index], "genotype": batch["genotype"][index], "split": batch["split"][index], "horizon": int(batch["horizon"][index]), "target_DAS": int(batch["target_DAS"][index]), "forecast_delta_days": int(batch["forecast_delta_days"][index])}
            for position, target in enumerate(TARGETS):
                record[f"pred_{target}"] = float(prediction[index, position])
                record[f"true_{target}"] = float(batch["target"][index, position])
            records.append(record)
    return pd.DataFrame(records)


def validation_score(predictions, scale):
    values = []
    for horizon, horizon_group in predictions.groupby("horizon"):
        for position, target in enumerate(TARGETS):
            absolute = (horizon_group[f"pred_{target}"] - horizon_group[f"true_{target}"]).abs()
            macro = absolute.groupby(horizon_group.plant_uid).mean().mean()
            values.append(float(macro / scale[position]))
    return float(np.mean(values))


def run(args):
    root = Path(args.root).resolve()
    protocol = json.loads((root / "configs/m3/protocol.json").read_text())
    windows = pd.read_csv(root / "data/processed/sugar4d/forecast_windows.csv")
    splits = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]
    if args.partition not in splits:
        raise ValueError("Unknown plant partition")
    windows["split"] = windows.plant_uid.map(splits[args.partition])
    if windows.groupby("plant_uid").split.nunique().max() != 1 or windows.split.isna().any():
        raise ValueError("Plant split invalid")
    output = root / "outputs/m3" / args.partition / f"points{args.points}"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    all_predictions = []
    training_records = []
    cache_root = root / "data/processed/sugar4d/point_cache"
    source_files = ["configs/m3/protocol.json", "src/datasets/sugar4d_dataset.py", "src/models/pointnet_forecaster.py", "src/models/pointnet_gru_forecaster.py", "scripts/train_m3.py", "data/processed/sugar4d/forecast_windows.csv", "reports/m3/plant_splits.json"]
    fingerprints = {path: file_hash(root / path) for path in source_files}
    models = args.models.split(",")
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else protocol["seeds"]
    for model_name in models:
        for seed in seeds:
            seed_everything(seed)
            train_windows = windows.loc[windows.split.eq("train")]
            val_windows = windows.loc[windows.split.eq("val")]
            mean_np, scale_np = standardizer(train_windows)
            mean = torch.from_numpy(mean_np).to(device)
            scale = torch.from_numpy(scale_np).to(device)
            train_loader = loader(train_windows, args.points, seed, protocol["batch_size"], True, cache_root)
            val_loader = loader(val_windows, args.points, seed, protocol["batch_size"], False, cache_root)
            test_loader = loader(windows.loc[windows.split.eq("test")], args.points, seed, protocol["batch_size"], False, cache_root)
            model = build_model(model_name, protocol).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
            best_loss = float("inf")
            best_state = None
            best_epoch = 0
            history = []
            initial_path = output / f"{model_name}_seed{seed}_initial.pt"
            torch.save(model.state_dict(), initial_path)
            initial_hash = file_hash(initial_path)
            initial_tensor_hash = tensor_state_hash(model.state_dict())
            started = time.monotonic()
            if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
            print(f"START {args.partition} points={args.points} {model_name} seed={seed} parameters={sum(parameter.numel() for parameter in model.parameters())}", flush=True)
            for epoch_index in range(1, protocol["epochs"] + 1):
                train_loader.dataset.set_epoch(epoch_index)
                train_loss = epoch(model, train_loader, optimizer, mean, scale, device)
                val_predictions = predict(model, val_loader, mean_np, scale_np, device)
                val_loss = validation_score(val_predictions, scale_np)
                history.append({"epoch": epoch_index, "train_mse": train_loss, "validation_macro_normalized_mae": val_loss})
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch_index
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                if epoch_index % 10 == 0:
                    print(f"EPOCH {epoch_index} train={train_loss:.5f} val={val_loss:.5f} seconds={time.monotonic()-started:.1f}", flush=True)
            model.load_state_dict(best_state)
            checkpoint = output / f"{model_name}_seed{seed}_points{args.points}.pt"
            torch.save({"model": model.state_dict(), "mean": mean_np.tolist(), "scale": scale_np.tolist(), "protocol": protocol, "best_epoch": best_epoch, "model_name": model_name, "seed": seed, "points": args.points, "fingerprints": fingerprints, "partition": args.partition}, checkpoint)
            prediction = predict(model, test_loader, mean_np, scale_np, device)
            prediction["model"] = model_name
            prediction["seed"] = seed
            prediction["points"] = args.points
            prediction["partition"] = args.partition
            prediction.to_csv(output / f"{model_name}_seed{seed}_test_predictions.csv", index=False)
            val_predictions = predict(model, val_loader, mean_np, scale_np, device)
            val_predictions.to_csv(output / f"{model_name}_seed{seed}_validation_predictions.csv", index=False)
            all_predictions.append(prediction)
            record = {"model": model_name, "seed": seed, "partition": args.partition, "points": args.points, "best_epoch": best_epoch, "best_val_loss": best_loss,
                      "initial_state_file_sha256": initial_hash, "parameters": sum(parameter.numel() for parameter in model.parameters()), "history": history, "fingerprints": fingerprints,
                      "initial_tensor_sha256": initial_tensor_hash,
                      "seconds": time.monotonic() - started, "epochs": protocol["epochs"], "optimizer_steps": len(train_loader) * protocol["epochs"],
                      "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                      "training_plants": sorted(train_windows.plant_uid.unique()), "validation_plants": sorted(val_windows.plant_uid.unique()), "normalization_mean": mean_np.tolist(), "normalization_std": scale_np.tolist()}
            training_records.append(record)
            write_json(output / f"{model_name}_seed{seed}_training.json", record)
            print(f"DONE {model_name} seed={seed} best_epoch={best_epoch} seconds={record['seconds']:.1f}", flush=True)
            del model, optimizer
            if device.type == "cuda": torch.cuda.empty_cache()
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(output / f"neural_predictions_points{args.points}.csv", index=False)
    write_json(output / f"training_points{args.points}.json", {"device": str(device), "records": training_records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--partition", default="official")
    parser.add_argument("--models", default="B3-2,B3-3,B3-4")
    parser.add_argument("--seeds", default=None)
    args = parser.parse_args()
    run(args)
