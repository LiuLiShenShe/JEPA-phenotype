"""Fixed train/validation-only fitting, checkpoint replay and balanced losses."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from datasets.sugar4d_organ_dataset import EVENT_FEATURES, ORGAN_FEATURES, ORGAN_TARGETS, PLANT_TARGETS, Sugar4DCurrentDataset, Sugar4DOrganDataset, time_vector
from m3.common import file_hash, seed_everything, tensor_state_hash, write_json
from m35.data import GEOMETRY_FEATURES
from models.geometry_residual_forecaster import CurrentPointNetRegressor, GeometryResidualForecaster
from models.organ_growth_baselines import OrganStateGRU, OrganStateMLP, linear_growth, persistence


def balanced_weights(frame, valid=None):
    selected = np.ones(len(frame), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    result = np.zeros(len(frame), dtype=np.float64)
    subset = frame.loc[selected]
    keys = ["plant_uid"] + (["horizon"] if "horizon" in frame else [])
    counts = subset.groupby(keys).window_id.transform("size").to_numpy()
    result[selected] = 1 / counts
    if result.sum() == 0:
        raise ValueError("No valid plant-balanced samples")
    result *= len(frame) / result.sum()
    return result.astype(np.float32)


def scaler_fit(values, weights=None):
    values = np.asarray(values, dtype=np.float64)
    mean = np.average(values, axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.average(np.square(values - mean), axis=0, weights=weights)), 1e-6)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "fit_split": "train"}


def transform(values, state):
    if state["fit_split"] != "train":
        raise ValueError("Scaler must be train-only")
    return ((np.asarray(values) - np.asarray(state["mean"])) / np.asarray(state["scale"])).astype(np.float32)


def inverse(values, state):
    return np.asarray(values) * np.asarray(state["scale"]) + np.asarray(state["mean"])


def macro_mae(frame, prediction, truth):
    values = np.abs(np.asarray(prediction) - np.asarray(truth))
    if values.ndim == 1:
        values = values[:, None]
    weights = balanced_weights(frame)
    return np.average(values, axis=0, weights=weights)


def current_arrays(root, assignment):
    root = Path(root)
    frames = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    geometry = pd.read_csv(root / "outputs/m3_5/data/geometry.csv")
    frames = frames.merge(geometry, on=["plant_uid", "DAS"], validate="one_to_one")
    frames["split"] = frames.plant_uid.map(assignment)
    frames["window_id"] = frames.plant_uid + "_current_" + frames.DAS.astype(str)
    frames["horizon"] = 0
    return frames


def base_prediction_frame(rows, model, seed, partition, task):
    columns = [name for name in ("window_id", "plant_uid", "leaf_instance_id", "split", "horizon", "current_DAS", "target_DAS", "DAS", "forecast_delta_days", "future_observed", "event", "new_count") if name in rows]
    result = rows[columns].reset_index(drop=True).copy()
    result["model"], result["seed"], result["partition"], result["task"] = model, seed, partition, task
    return result


def save_ridge_fit(train, validation, inputs, targets, protocol):
    train_mask, val_mask = train.index.to_numpy(), validation.index.to_numpy()
    weights = balanced_weights(train)
    x_state = scaler_fit(inputs[train_mask], weights)
    y_state = scaler_fit(targets[train_mask], weights)
    normalized = transform(inputs, x_state)
    scaled_target = transform(targets[train_mask], y_state)
    best = None
    for alpha in protocol["ridge_alphas"]:
        model = Ridge(alpha=alpha).fit(normalized[train_mask], scaled_target, sample_weight=weights)
        raw_prediction = np.asarray(model.predict(normalized[val_mask])).reshape(len(val_mask), len(y_state["scale"]))
        prediction = inverse(raw_prediction, y_state)
        score = float(np.mean(macro_mae(validation, prediction, targets[val_mask]) / np.asarray(y_state["scale"])))
        if best is None or score < best["validation_score"]:
            best = {"coef": model.coef_.tolist(), "intercept": np.asarray(model.intercept_).tolist(), "alpha": alpha,
                    "validation_score": score, "input_scaler": x_state, "target_scaler": y_state}
    return best


def ridge_predict(values, fit):
    normalized = transform(values, fit["input_scaler"])
    prediction = normalized @ np.asarray(fit["coef"]).T + np.asarray(fit["intercept"])
    prediction = np.asarray(prediction).reshape(len(normalized), len(fit["target_scaler"]["scale"]))
    return inverse(prediction, fit["target_scaler"])


def current_simple(root, partition, assignment, protocol):
    rows = current_arrays(root, assignment)
    train, validation = rows.loc[rows.split.eq("train")], rows.loc[rows.split.eq("val")]
    truth = rows[list(PLANT_TARGETS)].to_numpy()
    geometry = rows[list(GEOMETRY_FEATURES)].to_numpy()
    outputs, fits = [], {}
    for name, inputs in (("A2-Geometry-Ridge", geometry), ("A2-DAS-Ridge", rows[["DAS"]].to_numpy())):
        fit = save_ridge_fit(train, validation, inputs, truth, protocol)
        fits[name] = fit
        mask = rows.split.ne("test").to_numpy()
        prediction = np.maximum(ridge_predict(inputs[mask], fit), 0)
        result = base_prediction_frame(rows.loc[mask], name, -1, partition, "current")
        for position, target in enumerate(PLANT_TARGETS):
            result[f"pred_{target}"] = prediction[:, position]
            result[f"true_{target}"] = truth[mask, position]
        outputs.append(result)
    destination = Path(root) / "outputs/m3_5" / partition / "current"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "ridge_fits.json", fits)
    pd.concat(outputs).to_csv(destination / "simple_train_val.csv", index=False)
    return fits


def source_fingerprints(root):
    root = Path(root)
    paths = [root / "configs/m3_5/protocol.json", root / "src/datasets/sugar4d_organ_dataset.py", root / "src/models/organ_growth_baselines.py",
             root / "src/models/geometry_residual_forecaster.py", root / "src/m35/engine.py", root / "src/m35/data.py", root / "reports/m3/plant_splits.json"]
    paths += sorted((root / "outputs/m3_5/data").glob("*.csv"))
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def current_train(root, partition, assignment, protocol, device, seeds):
    rows = current_arrays(root, assignment)
    train = rows.loc[rows.split.eq("train")].reset_index(drop=True)
    validation = rows.loc[rows.split.eq("val")].reset_index(drop=True)
    destination = Path(root) / "outputs/m3_5" / partition / "current"
    destination.mkdir(parents=True, exist_ok=True)
    target_state = scaler_fit(train[list(PLANT_TARGETS)], balanced_weights(train))
    geometry_state = scaler_fit(train[list(GEOMETRY_FEATURES)], balanced_weights(train))
    target_scale = np.asarray(target_state["scale"])
    cache = Path(root) / "data/processed/sugar4d/point_cache"
    for name in ("A1-PointNet", "A3-PointNet-Geometry"):
        for seed in seeds:
            checkpoint_path = destination / f"{name}_seed{seed}.pt"
            if checkpoint_path.exists():
                continue
            seed_everything(seed)
            model = CurrentPointNetRegressor(len(GEOMETRY_FEATURES) if name.startswith("A3") else 0,
                                             protocol["pointnet_feature_dim"], protocol["pointnet_hidden_dim"], protocol["dropout"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
            initial_hash = tensor_state_hash(model.state_dict())
            torch.save(model.state_dict(), destination / f"{name}_seed{seed}_initial.pt")
            loaders = {}
            for split, frame in (("train", train), ("val", validation)):
                dataset = Sugar4DCurrentDataset(frame, cache, protocol["points"], protocol["sampling_seed"], split == "train")
                loaders[split] = DataLoader(dataset, batch_size=protocol["pointnet_batch_size"], shuffle=split == "train",
                                            generator=torch.Generator().manual_seed(seed), num_workers=0)
            tensors = {split: (torch.tensor(transform(frame[list(GEOMETRY_FEATURES)], geometry_state), device=device),
                               torch.tensor(transform(frame[list(PLANT_TARGETS)], target_state), device=device))
                       for split, frame in (("train", train), ("val", validation))}
            best_score, best_state, best_epoch, history = float("inf"), None, 0, []
            started = time.monotonic()
            print(f"START {partition} {name} seed={seed}", flush=True)
            for epoch_index in range(1, protocol["epochs"] + 1):
                model.train()
                loaders["train"].dataset.set_epoch(epoch_index)
                losses = []
                for batch in loaders["train"]:
                    indices = batch["index"].to(device)
                    prediction = model(batch["points"].to(device), tensors["train"][0][indices])
                    loss = functional.mse_loss(prediction, tensors["train"][1][indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"])
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                val_prediction = current_inference(model, loaders["val"], tensors["val"][0], device)
                score = float(np.mean(macro_mae(validation, inverse(val_prediction, target_state), validation[list(PLANT_TARGETS)]) / target_scale))
                history.append({"epoch": epoch_index, "train_mse": float(np.mean(losses)), "validation_score": score})
                if score < best_score:
                    best_score, best_epoch = score, epoch_index
                    best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            model.load_state_dict(best_state)
            checkpoint = {"model": best_state, "model_name": name, "task": "current", "seed": seed, "partition": partition,
                          "geometry_scaler": geometry_state, "target_scaler": target_state, "protocol": protocol,
                          "selection_split": "val", "test_used_for_selection": False, "fingerprints": source_fingerprints(root)}
            torch.save(checkpoint, checkpoint_path)
            records = []
            for split, frame in (("train", train), ("val", validation)):
                loaders[split].dataset.training = False
                prediction = inverse(current_inference(model, loaders[split], tensors[split][0], device), target_state)
                result = base_prediction_frame(frame, name, seed, partition, "current")
                for position, target in enumerate(PLANT_TARGETS):
                    result[f"pred_{target}"] = np.maximum(prediction[:, position], 0)
                    result[f"true_{target}"] = frame[target].to_numpy()
                records.append(result)
            pd.concat(records).to_csv(destination / f"{name}_seed{seed}_train_val.csv", index=False)
            write_json(destination / f"{name}_seed{seed}_training.json", {"model": name, "task": "current", "partition": partition, "seed": seed,
                       "parameters": sum(parameter.numel() for parameter in model.parameters()), "initial_tensor_sha256": initial_hash,
                       "epochs": protocol["epochs"], "optimizer_steps": len(loaders["train"]) * protocol["epochs"],
                       "best_epoch": best_epoch, "validation_score": best_score, "seconds": time.monotonic() - started, "device": str(device),
                       "training_plants": sorted(train.plant_uid.unique()), "validation_plants": sorted(validation.plant_uid.unique()),
                       "history": history, "fingerprints": checkpoint["fingerprints"], "selection_split": "val", "test_used_for_selection": False})
            print(f"DONE {partition} {name} seed={seed} score={best_score:.4f} seconds={time.monotonic()-started:.1f}", flush=True)


@torch.no_grad()
def current_inference(model, loader, geometry, device):
    model.eval()
    predictions = np.zeros((len(loader.dataset), 3), dtype=np.float32)
    for batch in loader:
        indices = batch["index"].to(device)
        predictions[batch["index"].numpy()] = model(batch["points"].to(device), geometry[indices]).cpu().numpy()
    return predictions


def tabular_arrays(root, task, assignment, fits=None, include_test=False):
    root = Path(root)
    if task == "organ":
        rows = pd.read_csv(root / "outputs/m3_5/data/organ_windows.csv")
        dataset = Sugar4DOrganDataset(rows)
        sequence, times = dataset.sequences, dataset.times
        current = rows[[f"current_{target}" for target in ORGAN_TARGETS]].to_numpy(dtype=np.float32)
        truth = rows[[f"target_{target}" for target in ORGAN_TARGETS]].to_numpy(dtype=np.float32)
        valid = rows.future_observed.to_numpy(dtype=bool)
        label = np.where(valid[:, None], truth - current, np.nan)
        visibility = valid.astype(np.float32)
        baseline = current
        targets = ORGAN_TARGETS
    elif task == "event":
        rows = pd.read_csv(root / "outputs/m3_5/data/event_windows.csv")
        sequence = np.asarray([json.loads(value) for value in rows.history_values], dtype=np.float32)
        times = np.stack([time_vector(json.loads(row.history_DAS), row.target_DAS) for row in rows.itertuples(index=False)])
        label = rows[["new_count"]].to_numpy(dtype=np.float32)
        truth, baseline = label, np.zeros_like(label)
        valid, visibility = np.ones(len(rows), dtype=bool), rows.event.to_numpy(dtype=np.float32)
        targets = ("new_count",)
    else:
        rows = pd.read_csv(root / "data/processed/sugar4d/forecast_windows.csv")
        geometry = pd.read_csv(root / "outputs/m3_5/data/geometry.csv").set_index(["plant_uid", "DAS"])
        sequence = np.stack([geometry.loc[[(row.plant_uid, day) for day in json.loads(row.history_DAS)], list(GEOMETRY_FEATURES)].to_numpy() for row in rows.itertuples(index=False)]).astype(np.float32)
        times = np.stack([time_vector(json.loads(row.history_DAS), row.target_DAS) for row in rows.itertuples(index=False)])
        rows["current_DAS"] = rows.history_DAS.map(lambda value: json.loads(value)[-1])
        rows["split"] = rows.plant_uid.map(assignment)
        truth = rows[[f"target_{target}" for target in PLANT_TARGETS]].to_numpy(dtype=np.float32)
        baseline = np.zeros_like(truth)
        included = np.ones(len(rows), dtype=bool) if include_test else rows.split.ne("test").to_numpy()
        baseline[included] = np.maximum(ridge_predict(sequence[included, -1], fits["A2-Geometry-Ridge"]), 0).astype(np.float32)
        label = truth if task.endswith("absolute") else truth - baseline
        valid, visibility = np.ones(len(rows), dtype=bool), None
        targets = PLANT_TARGETS
    rows["split"] = rows.plant_uid.map(assignment)
    if rows.split.isna().any() or rows.groupby("plant_uid").split.nunique().max() != 1:
        raise ValueError("Illegal partition")
    return {"rows": rows, "sequence": sequence, "times": times, "label": label, "truth": truth, "baseline": baseline,
            "valid": valid, "visibility": visibility, "targets": targets}


def tabular_model(name, task, protocol):
    hidden, dropout = protocol["tabular_hidden_dim"], protocol["dropout"]
    if task.startswith("geometry"):
        return GeometryResidualForecaster(len(GEOMETRY_FEATURES), 7, hidden, "History" in name, dropout)
    input_dim, time_dim, output_dim = (5, 8, 4) if task == "organ" else (len(EVENT_FEATURES), 7, 2)
    if "MLP" in name:
        return OrganStateMLP(input_dim, time_dim, hidden, output_dim, dropout)
    return OrganStateGRU(input_dim, time_dim, hidden, output_dim, dropout, "Repeated" in name)


def tabular_score(data, indices, regression, probabilities, scales, task):
    rows = data["rows"].iloc[indices]
    valid = data["valid"][indices]
    score = float(np.mean(macro_mae(rows.loc[valid], regression[valid], data["label"][indices][valid]) / scales))
    if probabilities is not None:
        truth = data["visibility"][indices]
        error = -(truth * np.log(np.clip(probabilities, 1e-7, 1)) + (1 - truth) * np.log(np.clip(1 - probabilities, 1e-7, 1)))
        bce = np.average(error, weights=balanced_weights(rows))
        score += (0.2 if task == "organ" else 1.0) * bce
    return float(score)


def train_tabular(root, partition, data, task, names, protocol, device, seeds):
    rows = data["rows"]
    train_indices = np.flatnonzero(rows.split.eq("train"))
    val_indices = np.flatnonzero(rows.split.eq("val"))
    train_rows = rows.iloc[train_indices]
    weights = balanced_weights(train_rows)
    sequence_state = scaler_fit(data["sequence"][train_indices].reshape(-1, data["sequence"].shape[-1]), np.repeat(weights, 3))
    time_state = scaler_fit(data["times"][train_indices], weights)
    valid_train = train_indices[data["valid"][train_indices]]
    label_state = scaler_fit(data["label"][valid_train], balanced_weights(rows.iloc[valid_train]))
    morphology_weights = balanced_weights(train_rows, data["valid"][train_indices])
    normalized_label = transform(np.nan_to_num(data["label"], nan=0), label_state)
    tensors = {"sequence": torch.tensor(transform(data["sequence"], sequence_state), device=device),
               "times": torch.tensor(transform(data["times"], time_state), device=device),
               "label": torch.tensor(normalized_label, device=device), "weights": torch.tensor(weights, device=device),
               "morphology_weights": torch.tensor(morphology_weights, device=device)}
    if data["visibility"] is not None:
        tensors["visibility"] = torch.tensor(data["visibility"], device=device)
    destination = Path(root) / "outputs/m3_5" / partition / task
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        for seed in seeds:
            checkpoint_path = destination / f"{name}_seed{seed}.pt"
            if checkpoint_path.exists():
                continue
            seed_everything(seed)
            model = tabular_model(name, task, protocol).to(device)
            initial_hash = tensor_state_hash(model.state_dict())
            torch.save(model.state_dict(), destination / f"{name}_seed{seed}_initial.pt")
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rate"], weight_decay=protocol["weight_decay"])
            rng = np.random.default_rng(seed)
            best_score, best_state, best_epoch, history = float("inf"), None, 0, []
            started = time.monotonic()
            output_dim = len(data["targets"])
            print(f"START {partition} {task} {name} seed={seed}", flush=True)
            for epoch_index in range(1, protocol["epochs"] + 1):
                model.train()
                losses = []
                order = rng.permutation(len(train_indices))
                for start in range(0, len(order), protocol["batch_size"]):
                    local = torch.tensor(order[start:start + protocol["batch_size"]], device=device)
                    indices = torch.tensor(train_indices[order[start:start + protocol["batch_size"]]], device=device)
                    raw = model(tensors["sequence"][indices], tensors["times"][indices])
                    regression = raw[:, :output_dim]
                    if task == "event":
                        regression = functional.softplus(regression) / torch.tensor(label_state["scale"], device=device) - torch.tensor(label_state["mean"], device=device) / torch.tensor(label_state["scale"], device=device)
                    error = torch.square(regression - tensors["label"][indices]).mean(dim=1)
                    loss = (error * tensors["morphology_weights"][local]).mean()
                    if data["visibility"] is not None:
                        bce = functional.binary_cross_entropy_with_logits(raw[:, -1], tensors["visibility"][indices], reduction="none")
                        loss = loss + (protocol["visibility_loss_weight"] if task == "organ" else 1.0) * (bce * tensors["weights"][local]).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["gradient_clip"])
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                raw = tabular_inference(model, tensors, val_indices)
                regression, probability = decode_tabular(raw, label_state, task)
                score = tabular_score(data, val_indices, regression, probability, np.asarray(label_state["scale"]), task)
                history.append({"epoch": epoch_index, "train_loss": float(np.mean(losses)), "validation_score": score})
                if score < best_score:
                    best_score, best_epoch = score, epoch_index
                    best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
            model.load_state_dict(best_state)
            checkpoint = {"model": best_state, "model_name": name, "task": task, "seed": seed, "partition": partition,
                          "sequence_scaler": sequence_state, "time_scaler": time_state, "label_scaler": label_state, "protocol": protocol,
                          "selection_split": "val", "test_used_for_selection": False, "fingerprints": source_fingerprints(root)}
            torch.save(checkpoint, checkpoint_path)
            records = []
            for indices in (train_indices, val_indices):
                raw = tabular_inference(model, tensors, indices)
                regression, probability = decode_tabular(raw, label_state, task)
                records.append(tabular_prediction_frame(data, indices, regression, probability, name, seed, partition, task))
            pd.concat(records).to_csv(destination / f"{name}_seed{seed}_train_val.csv", index=False)
            record = {"model": name, "task": task, "partition": partition, "seed": seed, "parameters": sum(parameter.numel() for parameter in model.parameters()),
                      "initial_tensor_sha256": initial_hash, "epochs": protocol["epochs"], "optimizer_steps": math.ceil(len(train_indices) / protocol["batch_size"]) * protocol["epochs"],
                      "best_epoch": best_epoch, "validation_score": best_score, "seconds": time.monotonic() - started, "device": str(device),
                      "training_plants": sorted(train_rows.plant_uid.unique()), "validation_plants": sorted(rows.iloc[val_indices].plant_uid.unique()),
                      "history": history, "fingerprints": checkpoint["fingerprints"], "selection_split": "val", "test_used_for_selection": False}
            write_json(destination / f"{name}_seed{seed}_training.json", record)
            print(f"DONE {partition} {task} {name} seed={seed} score={best_score:.4f} seconds={record['seconds']:.1f}", flush=True)


@torch.no_grad()
def tabular_inference(model, tensors, indices):
    model.eval()
    return np.concatenate([model(tensors["sequence"][selection], tensors["times"][selection]).cpu().numpy()
                           for selection in np.array_split(np.asarray(indices), max(1, math.ceil(len(indices) / 512)))])


def decode_tabular(raw, label_state, task):
    output_dim = len(label_state["scale"])
    if task == "event":
        regression = np.logaddexp(0, raw[:, :output_dim])
    else:
        regression = inverse(raw[:, :output_dim], label_state)
    probability = 1 / (1 + np.exp(-np.clip(raw[:, -1], -40, 40))) if task in ("organ", "event") else None
    return regression, probability


def tabular_prediction_frame(data, indices, regression, probability, name, seed, partition, task):
    rows = data["rows"].iloc[indices]
    result = base_prediction_frame(rows, name, seed, partition, task)
    absolute = regression if task.endswith("absolute") or task == "event" else regression + data["baseline"][indices]
    absolute = np.maximum(absolute, 0)
    for position, target in enumerate(data["targets"]):
        result[f"pred_{target}"] = absolute[:, position]
        result[f"true_{target}"] = data["truth"][indices, position]
        result[f"baseline_{target}"] = data["baseline"][indices, position]
        result[f"pred_delta_{target}"] = absolute[:, position] - data["baseline"][indices, position]
        result[f"true_delta_{target}"] = data["truth"][indices, position] - data["baseline"][indices, position]
    if probability is not None:
        result["pred_visibility" if task == "organ" else "pred_event"] = probability
    return result


def organ_simple(root, partition, data):
    rows = data["rows"]
    all_results = []
    positions = [ORGAN_FEATURES.index(target) for target in ORGAN_TARGETS]
    for name in ("C0-Persistence", "C1-Linear-Growth"):
        selected = np.flatnonzero(rows.split.ne("test"))
        prediction = []
        for index in selected:
            row = rows.iloc[index]
            history = data["sequence"][index][:, positions]
            prediction.append(persistence(history) if name.startswith("C0") else linear_growth(history, json.loads(row.history_DAS), row.target_DAS))
        prediction = np.asarray(prediction)
        all_results.append(tabular_prediction_frame(data, selected, prediction - data["baseline"][selected], np.ones(len(selected)), name, -1, partition, "organ"))
    destination = Path(root) / "outputs/m3_5" / partition / "organ"
    destination.mkdir(parents=True, exist_ok=True)
    pd.concat(all_results).to_csv(destination / "simple_train_val.csv", index=False)


def event_simple(root, partition, data, protocol):
    rows = data["rows"]
    records, fits = [], {}
    for horizon in protocol["horizons"]:
        selected = np.flatnonzero(rows.horizon.eq(horizon) & rows.split.ne("test"))
        train_indices = np.flatnonzero(rows.horizon.eq(horizon) & rows.split.eq("train"))
        validation_indices = np.flatnonzero(rows.horizon.eq(horizon) & rows.split.eq("val"))
        train_rows, val_rows = rows.iloc[train_indices], rows.iloc[validation_indices]
        weights = balanced_weights(train_rows)
        probability = float(np.average(train_rows.event, weights=weights))
        count = float(np.average(train_rows.new_count, weights=weights))
        stages = np.digitize(rows.current_DAS.to_numpy(), [58, 75])
        stage_fit = {}
        for stage in range(3):
            mask = stages[train_indices] == stage
            stage_fit[str(stage)] = {"probability": float(np.average(train_rows.event.to_numpy()[mask], weights=weights[mask])),
                                     "count": float(np.average(train_rows.new_count.to_numpy()[mask], weights=weights[mask]))}
        fits[str(horizon)] = {"frequency": {"probability": probability, "count": count}, "stages": stage_fit}
        for name in ("E0-Frequency", "E0-DAS"):
            prob = np.full(len(selected), probability) if name.endswith("Frequency") else np.asarray([stage_fit[str(stage)]["probability"] for stage in stages[selected]])
            counts = np.full((len(selected), 1), count) if name.endswith("Frequency") else np.asarray([[stage_fit[str(stage)]["count"]] for stage in stages[selected]])
            keep = rows.iloc[selected].split.ne("test").to_numpy()
            records.append(tabular_prediction_frame(data, selected[keep], counts[keep], prob[keep], name, -1, partition, "event"))
        inputs = np.column_stack((rows.current_visible_count, rows.current_DAS / 100, rows.forecast_delta_days / 10, rows.target_DAS / 100))
        scaler = scaler_fit(inputs[train_indices], weights)
        normalized = transform(inputs, scaler)
        best = None
        for regularization in protocol["logistic_C"]:
            model = LogisticRegression(C=regularization, max_iter=1000, random_state=17).fit(normalized[train_indices], train_rows.event, sample_weight=weights)
            prob = model.predict_proba(normalized[validation_indices])[:, 1]
            label = val_rows.event.to_numpy()
            score = float(np.average(-(label * np.log(prob) + (1 - label) * np.log(1 - prob)), weights=balanced_weights(val_rows)))
            if best is None or score < best["validation_score"]:
                best = {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(), "C": regularization, "validation_score": score, "input_scaler": scaler}
        count_fit = save_ridge_fit(train_rows, val_rows, inputs, rows[["new_count"]].to_numpy(), protocol)
        fits[str(horizon)]["logistic"], fits[str(horizon)]["count_ridge"] = best, count_fit
        prob = logistic_predict(inputs[selected], best)
        counts = np.maximum(ridge_predict(inputs[selected], count_fit), 0)
        keep = rows.iloc[selected].split.ne("test").to_numpy()
        records.append(tabular_prediction_frame(data, selected[keep], counts[keep], prob[keep], "E1-Logistic", -1, partition, "event"))
    destination = Path(root) / "outputs/m3_5" / partition / "event"
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "simple_fits.json", fits)
    pd.concat(records).to_csv(destination / "simple_train_val.csv", index=False)


def logistic_predict(inputs, fit):
    logit = transform(inputs, fit["input_scaler"]) @ np.asarray(fit["coef"]).T + np.asarray(fit["intercept"])
    return (1 / (1 + np.exp(-logit)))[:, 0]
