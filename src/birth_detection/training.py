from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.nn import functional

from birth_detection.common import digest, initialize, protocol, write_json
from temporal_encoder.models import BirthEncoder, PairMatcher


def normalize_features(values, train_mask):
    mean = values[train_mask].mean(axis=0)
    scale = values[train_mask].std(axis=0)
    scale[scale < 1e-6] = 1
    return np.clip((values - mean) / scale, -10, 10).astype(np.float32), mean, scale


def save_prediction(path, rows, values):
    result = rows.copy()
    result["probability"] = values
    result.to_csv(path, index=False)


def forecast_features(rows):
    names = ["points", "voxel_count", "new_fraction", "predicted_leaf_count", "height_m", "width_m", "core_predicted_points"]
    result = np.zeros((len(rows), len(names) * 3), dtype=np.float32)
    for unused_plant, group in rows.groupby("plant_id"):
        indices = group.index.to_numpy()
        current = rows.loc[indices, names].to_numpy(dtype=np.float32)
        for position, index in enumerate(indices):
            window = [current[max(0, position - offset)] for offset in [2, 1, 0]]
            result[index] = np.concatenate(window)
    return result


def train_birth(root, method, seed, device="cuda:1"):
    settings = protocol(root)
    initialize(seed)
    directory = root / f"outputs/m7/birth/{method}/seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists():
        return
    candidates = pd.read_csv(root / "outputs/m7/candidates.csv")
    data = np.load(root / "outputs/m7/birth_dataset.npz")
    train_mask = (candidates.split == "train").to_numpy() & (candidates.frame_index > 0).to_numpy()
    feature_name = "single_features" if method in {"single_RF", "single_PointNet"} else "temporal_features"
    values, mean, scale = normalize_features(data[feature_name], train_mask)
    targets = data["targets"]
    patch_data = data["patches"] if method not in {"history_RF", "single_RF"} else None
    timing_data = data["timing"] if patch_data is not None else None
    started = time.perf_counter()
    training_manifest = {"method": method, "seed": seed, "split": "train", "plants": candidates.loc[train_mask, "plant_id"].nunique(),
                         "training_candidates": int(train_mask.sum()), "positive_candidates": int(targets[train_mask].sum()),
                         "target": "current scan visible birth, not next-scan forecast", "GT_identity_input": False,
                         "candidate_pool": "shared causal proposals; single-feature controls do not remove proposal-level temporal information",
                         "code_sha256": {name: digest(root / name) for name in ["src/birth_detection/training.py", "src/temporal_encoder/models.py"]}}
    if method in {"history_RF", "single_RF"}:
        forest = RandomForestClassifier(n_estimators=settings["birth_training"]["random_forest_trees"],
                                        max_depth=settings["birth_training"]["random_forest_max_depth"], min_samples_leaf=3,
                                        class_weight="balanced", random_state=seed, n_jobs=4)
        forest.fit(values[train_mask], targets[train_mask])
        joblib.dump({"model": forest, "mean": mean, "scale": scale, "manifest": training_manifest}, directory / "model.joblib")
        probabilities = np.full(len(values), np.nan)
        observable = (candidates.split != "test").to_numpy()
        probabilities[observable] = forest.predict_proba(values[observable])[:, list(forest.classes_).index(1)]
        training_manifest["trees"] = len(forest.estimators_)
        training_manifest["checkpoint_sha256"] = digest(directory / "model.joblib")
    else:
        model = BirthEncoder(values.shape[1], method == "PointNet_GRU").to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings["birth_training"]["learning_rate"], weight_decay=settings["birth_training"]["weight_decay"])
        indices = np.flatnonzero(train_mask)
        generator = np.random.default_rng(seed)
        positive_weight = float((len(indices) - targets[indices].sum()) / max(targets[indices].sum(), 1))
        positive_weight = min(positive_weight, 30)
        logs, updates = [], 0
        for epoch in range(1, settings["birth_training"]["epochs"] + 1):
            model.train()
            losses = []
            for batch in np.array_split(generator.permutation(indices), int(np.ceil(len(indices) / settings["birth_training"]["batch_size"]))):
                patch = torch.from_numpy(patch_data[batch]).to(device)
                timing = torch.from_numpy(timing_data[batch]).to(device)
                feature = torch.from_numpy(values[batch]).to(device)
                target = torch.from_numpy(targets[batch]).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(patch, timing, feature)
                loss = functional.binary_cross_entropy_with_logits(logits, target, pos_weight=target.new_tensor(positive_weight))
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite birth loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                losses.append(float(loss.detach()))
                updates += 1
            record = {"epoch": epoch, "updates": updates, "train_loss": float(np.mean(losses)), "elapsed_seconds": time.perf_counter() - started}
            logs.append(record)
            print("BIRTH", method, seed, record, flush=True)
            pd.DataFrame(logs).to_csv(directory / "epochs.csv", index=False)
        model.eval()
        probabilities = np.full(len(values), np.nan)
        observable = np.flatnonzero((candidates.split != "test").to_numpy())
        with torch.inference_mode():
            for start in range(0, len(observable), 256):
                batch = observable[start:start + 256]
                logits = model(torch.from_numpy(patch_data[batch]).to(device), torch.from_numpy(timing_data[batch]).to(device), torch.from_numpy(values[batch]).to(device))
                probabilities[batch] = torch.sigmoid(logits).cpu().numpy()
        torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "mean": torch.from_numpy(mean),
                    "scale": torch.from_numpy(scale), "epoch": epoch, "updates": updates, "manifest": training_manifest}, directory / "model.pth")
        training_manifest.update({"epochs": epoch, "updates": updates, "parameters": sum(parameter.numel() for parameter in model.parameters()),
                                  "checkpoint_sha256": digest(directory / "model.pth")})
    save_prediction(directory / "candidate_probabilities.csv", candidates, probabilities)
    training_manifest["seconds"] = time.perf_counter() - started
    write_json(directory / "completed.json", training_manifest)


def train_forecast(root, seed):
    settings = protocol(root)
    initialize(seed)
    directory = root / f"outputs/m7/forecast/seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists():
        return
    rows = pd.read_csv(root / "outputs/m7/forecast_rows.csv")
    values = forecast_features(rows)
    mask = (rows.split == "train").to_numpy()
    targets = rows.forecast_target_new_visible.astype(int).to_numpy()
    model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=3, class_weight="balanced", random_state=seed, n_jobs=4)
    model.fit(values[mask], targets[mask])
    probabilities = np.full(len(values), np.nan)
    observable = (rows.split != "test").to_numpy()
    probabilities[observable] = model.predict_proba(values[observable])[:, list(model.classes_).index(1)]
    joblib.dump(model, directory / "model.joblib")
    save_prediction(directory / "probabilities.csv", rows, probabilities)
    write_json(directory / "completed.json", {"seed": seed, "trees": 200, "training_plants": 24, "training_transitions": int(mask.sum()),
               "target": "plant-level presence of next scheduled scan new visible organ; localization not claimed",
               "input_max_time": "DAS <= current DAS", "checkpoint_sha256": digest(directory / "model.joblib")})


def train_matcher(root, method, seed, device="cuda:1"):
    settings = protocol(root)
    initialize(seed)
    directory = root / f"outputs/m7/matching/{method}/seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists():
        return
    rows = pd.read_csv(root / "outputs/m7/pairs.csv")
    data = np.load(root / "outputs/m7/pair_dataset.npz")
    organs = np.load(root / "outputs/m7/organ_dataset.npz")
    organ_points = organs["points"] if method == "C2" else None
    previous_indices = data["previous_indices"]
    current_indices = data["current_indices"]
    mask = (rows.split == "train").to_numpy()
    values, mean, scale = normalize_features(data["features"], mask)
    targets = data["targets"]
    model = PairMatcher(values.shape[1], method == "C2").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["tracking_training"]["learning_rate"], weight_decay=settings["tracking_training"]["weight_decay"])
    generator = np.random.default_rng(seed)
    indices = np.flatnonzero(mask)
    positive_weight = float((len(indices) - targets[indices].sum()) / max(targets[indices].sum(), 1))
    started = time.perf_counter()
    updates, logs = 0, []
    for epoch in range(1, settings["tracking_training"]["epochs"] + 1):
        losses, metric_losses = [], []
        model.train()
        for batch in np.array_split(generator.permutation(indices), int(np.ceil(len(indices) / settings["tracking_training"]["batch_size"]))):
            feature = torch.from_numpy(values[batch]).to(device)
            previous = torch.from_numpy(organ_points[previous_indices[batch]]).to(device) if method == "C2" else None
            current = torch.from_numpy(organ_points[current_indices[batch]]).to(device) if method == "C2" else None
            target = torch.from_numpy(targets[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, distance = model(feature, previous, current)
            metric = (target * distance.square() + (1 - target) * functional.relu(.8 - distance).square()).mean() if method == "C2" else distance.mean()
            loss = functional.binary_cross_entropy_with_logits(logits, target, pos_weight=target.new_tensor(positive_weight)) + settings["tracking_training"]["metric_loss_weight"] * metric
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite matching loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
            metric_losses.append(float(metric.detach()))
            updates += 1
        record = {"epoch": epoch, "updates": updates, "train_loss": float(np.mean(losses)), "metric_loss": float(np.mean(metric_losses)), "elapsed_seconds": time.perf_counter() - started}
        logs.append(record)
        print("MATCH", method, seed, record, flush=True)
        pd.DataFrame(logs).to_csv(directory / "epochs.csv", index=False)
    model.eval()
    probabilities = np.full(len(values), np.nan)
    observable = np.flatnonzero((rows.split != "test").to_numpy())
    with torch.inference_mode():
        for start in range(0, len(observable), 512):
            batch = observable[start:start + 512]
            previous = torch.from_numpy(organ_points[previous_indices[batch]]).to(device) if method == "C2" else None
            current = torch.from_numpy(organ_points[current_indices[batch]]).to(device) if method == "C2" else None
            logits, unused_distance = model(torch.from_numpy(values[batch]).to(device), previous, current)
            probabilities[batch] = torch.sigmoid(logits).cpu().numpy()
    predicted = rows.copy()
    predicted["previous_organ_index"] = previous_indices
    predicted["current_organ_index"] = current_indices
    predicted["probability"] = probabilities
    predicted.to_csv(directory / "pair_probabilities.csv", index=False)
    manifest = {"method": method, "seed": seed, "train_plants": 24, "training_pairs": int(mask.sum()), "positive_pairs": int(targets[mask].sum()),
                "epochs": epoch, "updates": updates, "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "seconds": time.perf_counter() - started, "GT_ID_supervision_train_only": True, "JEPA": False,
                "code_sha256": {name: digest(root / name) for name in ["src/birth_detection/training.py", "src/temporal_encoder/models.py"]}}
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "mean": torch.from_numpy(mean),
                "scale": torch.from_numpy(scale), "epoch": epoch, "updates": updates, "manifest": manifest}, directory / "model.pth")
    manifest["checkpoint_sha256"] = digest(directory / "model.pth")
    write_json(directory / "completed.json", manifest)


def predict_test(root):
    settings = protocol(root)
    if not (root / "reports/m7/system_selection.json").exists():
        raise ValueError("Test inference requires locked Validation selection")
    data = np.load(root / "outputs/m7/birth_dataset.npz")
    patch_data = data["patches"]
    timing_data = data["timing"]
    for method in ["single_RF", "history_RF", "single_PointNet", "PointNet_GRU"]:
        for seed in settings["seeds"]:
            initialize(seed)
            directory = root / f"outputs/m7/birth/{method}/seed_{seed}"
            rows = pd.read_csv(directory / "candidate_probabilities.csv")
            selected = np.flatnonzero((rows.split == "test").to_numpy())
            feature_name = "single_features" if method in {"single_RF", "single_PointNet"} else "temporal_features"
            if method.endswith("RF"):
                checkpoint = joblib.load(directory / "model.joblib")
                values = np.clip((data[feature_name][selected] - checkpoint["mean"]) / checkpoint["scale"], -10, 10)
                probabilities = checkpoint["model"].predict_proba(values)[:, list(checkpoint["model"].classes_).index(1)]
            else:
                checkpoint = torch.load(directory / "model.pth", map_location="cpu", weights_only=True)
                values = np.clip((data[feature_name][selected] - checkpoint["mean"].numpy()) / checkpoint["scale"].numpy(), -10, 10).astype(np.float32)
                model = BirthEncoder(values.shape[1], method == "PointNet_GRU")
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model.eval()
                probabilities = []
                with torch.inference_mode():
                    for start in range(0, len(selected), 256):
                        batch = selected[start:start + 256]
                        logits = model(torch.from_numpy(patch_data[batch]), torch.from_numpy(timing_data[batch]), torch.from_numpy(values[start:start + 256]))
                        probabilities.extend(torch.sigmoid(logits).numpy())
            rows.loc[selected, "probability"] = probabilities
            rows.to_csv(directory / "candidate_probabilities.csv", index=False)
    for seed in settings["seeds"]:
        directory = root / f"outputs/m7/forecast/seed_{seed}"
        rows = pd.read_csv(directory / "probabilities.csv")
        selected = (rows.split == "test").to_numpy()
        model = joblib.load(directory / "model.joblib")
        rows.loc[selected, "probability"] = model.predict_proba(forecast_features(rows)[selected])[:, list(model.classes_).index(1)]
        rows.to_csv(directory / "probabilities.csv", index=False)
    write_json(root / "reports/m7/test_inference_gate.json", {"validation_system_selection_sha256": digest(root / "reports/m7/system_selection.json"),
               "all_Test_inference_after_Validation_selection": True, "models_retrained": False})
