"""Fixed-budget deterministic training with validation-only best-epoch selection."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from m15.artifacts import save_once
from m15.models import DiagnosticHead
from training.common import TrainStandardizer, clip_predictions, require_device, seed_everything, write_json
from training.losses import standardized_huber


def plant_weights(rows):
    counts = rows.groupby("plant_uid").window_id.transform("count").to_numpy()
    return len(rows) / (rows.plant_uid.nunique() * counts)


def training_inputs(dataset, scaler, device):
    return (torch.as_tensor(dataset.features, device=device),
            torch.as_tensor(scaler.transform(dataset.times), device=device))


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def fast_selection_score(predictions, targets, target_scale, weights):
    errors = torch.abs(predictions - targets) / target_scale
    return float((errors.mean(dim=1) * weights).mean())


def train(train_data, val_data, config, destination, device_name):
    if set(train_data.rows.split) != {"train"} or set(val_data.rows.split) != {"val"}:
        raise ValueError("Train/Validation only: test cannot enter training or epoch selection")
    if set(train_data.rows.plant_uid) & set(val_data.rows.plant_uid):
        raise ValueError("Independent plant split violated")
    destination = Path(destination)
    complete = destination / "resources.json"
    if complete.exists():
        if json.loads((destination / "config.json").read_text()) != config:
            raise ValueError("Completed run has a different locked configuration")
        if not (destination / "checkpoint.pt").exists():
            raise ValueError("Completed run lacks its checkpoint")
        return json.loads(complete.read_text())
    if (destination / "checkpoint.pt").exists():
        raise FileExistsError("Incomplete checkpoint exists; refusing to overwrite")
    destination.mkdir(parents=True, exist_ok=True)
    save_once(destination / "config.json", config)
    seed_everything(config["seed"])
    torch.set_num_threads(4)
    device = require_device(device_name)
    target_scaler = TrainStandardizer.fit(train_data.targets, train_data.rows)
    time_scaler = TrainStandardizer.fit(train_data.times, train_data.rows)
    train_features, train_times = training_inputs(train_data, time_scaler, device)
    val_features, val_times = training_inputs(val_data, time_scaler, device)
    targets = torch.as_tensor(target_scaler.transform(train_data.targets), device=device)
    val_targets = torch.as_tensor(val_data.targets, device=device)
    target_mean = torch.as_tensor(target_scaler.mean, device=device)
    target_scale = torch.as_tensor(target_scaler.scale, device=device)
    weights = torch.as_tensor(plant_weights(train_data.rows), dtype=torch.float32, device=device)
    val_weights = torch.as_tensor(plant_weights(val_data.rows), dtype=torch.float32, device=device)
    model = DiagnosticHead(config).to(device)
    initial_digest = state_digest(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"], eta_min=config["learning_rate"] * .1)
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])
    ordering_digest = hashlib.sha256()
    best_score = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    steps = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        order_cpu = torch.randperm(len(train_data), generator=generator)
        ordering_digest.update(order_cpu.numpy().tobytes())
        order = order_cpu.to(device)
        total_loss = torch.zeros((), device=device)
        for start in range(0, len(train_data), config["batch_size"]):
            batch = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            output = model(train_features[batch], train_times[batch])
            loss = standardized_huber(output, targets[batch], weights[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total_loss += loss.detach() * len(batch)
            steps += 1
        model.eval()
        with torch.inference_mode():
            predicted = model(val_features, val_times) * target_scale + target_mean
            predicted[:, 0].clamp_(min=0)
            predicted[:, 1].clamp_(0, 1)
            score = fast_selection_score(predicted, val_targets, target_scale, val_weights)
        if not np.isfinite(score) or not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite loss or validation score")
        history.append({"epoch": epoch, "train_weighted_huber": float(total_loss / len(train_data)),
                        "val_plant_macro_normalized_mae": score, "learning_rate": optimizer.param_groups[0]["lr"]})
        if score < best_score - 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        scheduler.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    resources = {
        "seconds": time.perf_counter() - started, "epochs": config["max_epochs"], "best_epoch": best_epoch,
        "optimizer_steps": steps, "val_selection_score": best_score, "device": str(device),
        "model_parameter_device": str(next(model.parameters()).device),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "initial_state_sha256": initial_digest, "batch_order_sha256": ordering_digest.hexdigest(),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "selection_split": "val", "test_used_for_selection": False, "initialization": "from scratch, no legacy weights",
    }
    checkpoint = {
        "config": config, "model_state": best_state, "time_scaler": time_scaler.state(), "target_scaler": target_scaler.state(),
        "best_epoch": best_epoch, "selection_split": "val", "test_used_for_selection": False,
        "train_plants": sorted(train_data.rows.plant_uid.unique().tolist()),
        "val_plants": sorted(val_data.rows.plant_uid.unique().tolist()), "resources": resources,
    }
    torch.save(checkpoint, destination / "checkpoint.pt")
    pd.DataFrame(history).to_csv(destination / "training_history.csv", index=False)
    write_json(complete, resources)
    print(json.dumps({"run": destination.name, **resources}), flush=True)
    return resources


def predict(path, dataset, device_name):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["selection_split"] != "val" or checkpoint["test_used_for_selection"]:
        raise ValueError("Unsafe checkpoint selection provenance")
    seed_everything(checkpoint["config"]["seed"])
    device = require_device(device_name)
    model = DiagnosticHead(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    inputs = training_inputs(dataset, TrainStandardizer.from_state(checkpoint["time_scaler"]), device)
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(dataset), 2048):
            predictions.append(model(inputs[0][start:start + 2048], inputs[1][start:start + 2048]).cpu().numpy())
    scaler = TrainStandardizer.from_state(checkpoint["target_scaler"])
    return clip_predictions(scaler.inverse(np.concatenate(predictions)))
