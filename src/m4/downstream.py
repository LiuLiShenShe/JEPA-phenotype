from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.sugar4d_ssl_dataset import Sugar4DLabelledDataset, partition_rows
from m4.common import artifact_protocol_matches, load_protocol, rng_state, seed_all, sha256, standardize, standardize_fit, state_hash, write_json
from m4.metrics import count_metrics, segmentation_metrics
from models.point_jepa_downstream import PointJEPACount, PointJEPASegmentation, PointNetSegmentation
from models.point_jepa_encoder import PointJEPAEncoder, load_encoder_checkpoint


def source_encoder(root, protocol, partition, seed, condition):
    if partition not in protocol["partitions"] or seed not in protocol["seeds"]:
        raise ValueError("Unregistered downstream partition/seed")
    seed_all(seed)
    encoder = PointJEPAEncoder(**protocol["model"])
    reference = None
    if condition == "P1":
        reference = root / "outputs/m4/external/point_jepa_shapenet.ckpt"
        if not reference.exists():
            raise FileNotFoundError("P1 official checkpoint unavailable")
        load_encoder_checkpoint(encoder, reference)
    elif condition == "P2":
        reference = root / "outputs/m4/checkpoints" / partition / f"ssl_seed{seed}" / "encoder.pt"
        if not reference.exists():
            raise FileNotFoundError(f"P2 pretraining is not complete: {reference}")
        checkpoint = torch.load(reference, map_location="cpu", weights_only=False)
        expected = sorted(partition_rows(root, partition, "train").plant_uid.unique())
        if checkpoint["metadata"]["train_plants"] != expected or checkpoint["metadata"]["test_used"]:
            raise ValueError("SSL checkpoint plant provenance mismatch")
        load_encoder_checkpoint(encoder, reference)
    elif condition != "P0":
        raise ValueError("Unknown pretraining condition")
    return encoder, str(reference.relative_to(root)) if reference else None


def geometry_rows(root, rows):
    table = pd.read_csv(root / "outputs/m3_5/data/geometry.csv")
    columns = [name for name in table.columns if name not in ("plant_uid", "DAS")]
    merged = rows[["plant_uid", "DAS"]].merge(table, on=["plant_uid", "DAS"], validate="one_to_one", sort=False)
    if len(merged) != len(rows) or not np.isfinite(merged[columns].to_numpy()).all():
        raise ValueError("Missing lawful current XYZ geometry")
    return merged[columns].to_numpy(dtype=np.float32), columns


def datasets(root, train, validation, geometry_state=None):
    xyz_root, label_root = root / "outputs/m4/data/xyz", root / "outputs/m4/data/labels"
    result = {}
    for split, frame in (("train", train), ("val", validation)):
        geometry = geometry_rows(root, frame)[0] if geometry_state else None
        result[split] = Sugar4DLabelledDataset(frame, xyz_root, label_root, geometry, geometry_state)
    return result


@torch.no_grad()
def count_predict(model, loader, target_state, device):
    model.eval()
    outputs = []
    for batch in loader:
        prediction = model(batch["points"].to(device), batch["geometry"].to(device),
                           batch["center_indices"].to(device), batch["neighbors"].to(device))
        outputs.append(prediction.cpu().numpy())
    values = np.concatenate(outputs) * target_state["scale"][0] + target_state["mean"][0]
    return np.maximum(values, 0)


def count_frame(rows, values):
    frame = rows[["plant_uid", "DAS"]].reset_index(drop=True).copy()
    frame["truth"], frame["prediction"] = rows.visible_organ_count.to_numpy(), values
    return frame


def run_identifier(partition, seed, condition, fraction, mode, task):
    return f"{partition}_{task}_{condition}_{mode}_labels{int(100 * fraction)}_seed{seed}"


def run_count(root, partition, seed, condition, fraction, mode, device, replay_state=False):
    root = Path(root)
    protocol, lock = load_protocol(root)
    train = partition_rows(root, partition, "train")
    validation = partition_rows(root, partition, "val")
    selected = lock["label_budgets"][partition][str(fraction)]
    train = train.loc[train.plant_uid.isin(selected)].reset_index(drop=True)
    identifier = run_identifier(partition, seed, condition, fraction, mode, "count")
    output = (root / "outputs/m4/state_replays" / identifier if replay_state else
              root / "outputs/m4/checkpoints" / partition / identifier)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.json").exists():
        record = json.loads((output / "training.json").read_text())
        if not artifact_protocol_matches(root, record["protocol_sha256"], lock) or record["labelled_plants"] != selected:
            raise ValueError("Existing downstream checkpoint split/protocol mismatch")
        return record
    encoder, reference = source_encoder(root, protocol, partition, seed, condition)
    geometry, geometry_columns = geometry_rows(root, train)
    geometry_state = standardize_fit(geometry)
    target_state = standardize_fit(train.visible_organ_count.to_numpy()[:, None])
    dataset = datasets(root, train, validation, geometry_state)
    generator = torch.Generator().manual_seed(seed)
    loader = {split: DataLoader(value, batch_size=protocol["downstream"]["batch_size"], shuffle=split == "train",
                                generator=generator if split == "train" else None, num_workers=0)
              for split, value in dataset.items()}
    frozen = mode == "probe"
    if mode not in ("probe", "finetune"):
        raise ValueError("Unknown transfer mode")
    model = PointJEPACount(encoder, len(geometry_columns), frozen=frozen).to(device)
    initial_encoder_hash, initial_head_hash = state_hash(encoder.state_dict()), state_hash(model.head.state_dict())
    epochs = protocol["downstream"]["probe_epochs" if frozen else "epochs"]
    learning_rate = protocol["downstream"]["probe_learning_rate" if frozen else "learning_rate"]
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=learning_rate,
                                  weight_decay=protocol["downstream"]["weight_decay"])
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best_state, best_score, best_epoch, steps = [], None, float("inf"), 0, 0
    print(f"COUNT START {identifier} plants={len(selected)} mode={mode}", flush=True)
    feature_state = None
    if frozen:
        features = {}
        encoder.eval()
        for split in ("train", "val"):
            ordered = DataLoader(dataset[split], batch_size=protocol["downstream"]["batch_size"], shuffle=False)
            pooled = []
            with torch.no_grad():
                for batch in ordered:
                    values = encoder(batch["points"].to(device), batch["center_indices"].to(device), batch["neighbors"].to(device))
                    pooled.append(torch.cat((values, batch["geometry"].to(device)), dim=-1).cpu().numpy())
            features[split] = np.concatenate(pooled)
        feature_state = standardize_fit(features["train"])
        tensors = {split: torch.from_numpy(standardize(features[split], feature_state)).to(device) for split in features}
        target = torch.from_numpy(standardize(train.visible_organ_count.to_numpy()[:, None], target_state)).to(device).flatten()
        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(len(train), generator=generator)
            losses = []
            for start in range(0, len(train), protocol["downstream"]["batch_size"]):
                indices = permutation[start:start + protocol["downstream"]["batch_size"]].to(device)
                loss = nn.functional.mse_loss(model.head(tensors["train"][indices]).flatten(), target[indices])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), protocol["downstream"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
            with torch.no_grad():
                prediction = np.maximum(model.head(tensors["val"]).flatten().cpu().numpy() * target_state["scale"][0] + target_state["mean"][0], 0)
            score = count_metrics(count_frame(validation, prediction))[0]["plant_macro_mae"]
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_plant_macro_mae": score})
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()}
        final_state = {"head": {name: value.detach().cpu().clone() for name, value in model.head.state_dict().items()}}
        model.head.load_state_dict(best_state, strict=True)
        with torch.no_grad():
            values = np.maximum(model.head(tensors["val"]).flatten().cpu().numpy() * target_state["scale"][0] + target_state["mean"][0], 0)
        if state_hash(encoder.state_dict()) != initial_encoder_hash:
            raise ValueError("Frozen encoder/BatchNorm drifted")
        state = {"head": best_state, "encoder_reference": reference, "random_encoder_seed": seed if condition == "P0" else None}
    else:
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for batch in loader["train"]:
                target = (batch["count"].to(device) - target_state["mean"][0]) / target_state["scale"][0]
                prediction = model(batch["points"].to(device), batch["geometry"].to(device),
                                   batch["center_indices"].to(device), batch["neighbors"].to(device))
                loss = nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["downstream"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
            values = count_predict(model, loader["val"], target_state, device)
            score = count_metrics(count_frame(validation, values))[0]["plant_macro_mae"]
            history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "val_plant_macro_mae": score})
            if score < best_score:
                best_score, best_epoch = score, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        model.load_state_dict(best_state, strict=True)
        values = count_predict(model, loader["val"], target_state, device)
        state = {"model": best_state}
    predictions = count_frame(validation, values)
    record = {"id": identifier, "partition": partition, "seed": seed, "condition": condition, "label_fraction": fraction,
              "mode": mode, "task": "count", "labelled_plants": selected, "labelled_scans": len(train),
              "validation_plants": sorted(validation.plant_uid.unique()), "encoder_reference": reference,
              "protocol_sha256": lock["protocol_sha256"], "initial_encoder_hash": initial_encoder_hash,
              "initial_head_hash": initial_head_hash, "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
              "total_parameters": sum(value.numel() for value in model.parameters()), "epochs": epochs,
              "best_epoch": best_epoch, "optimizer_steps": steps, "seconds": time.monotonic() - started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "test_used": False, "selection": "val plant-macro MAE", "status": "COMPLETED",
              **count_metrics(predictions)[0]}
    for name in ("partition", "seed", "condition", "label_fraction", "mode"):
        predictions[name] = record[name]
    prediction_path = output / "predictions.csv" if replay_state else root / "outputs/m4/predictions" / f"{identifier}.csv"
    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save(state | {"metadata": record, "model_config": protocol["model"], "geometry_scaler": geometry_state,
                       "feature_scaler": feature_state, "target_scaler": target_state, "geometry_columns": geometry_columns}, output / "best.pt")
    torch.save({"model": final_state,
                "optimizer": optimizer.state_dict(), "rng": rng_state(), "loader_rng": generator.get_state(),
                "final_epoch": epochs, "steps": steps, "metadata": record}, output / "training_state.pt")
    write_json(output / "training.json", record)
    print(f"COUNT DONE {identifier} MAE={record['plant_macro_mae']:.4f} seconds={record['seconds']:.1f}", flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


@torch.no_grad()
def segmentation_predict(model, loader, device):
    model.eval()
    matrices, predictions = [], []
    for batch in loader:
        logits = model(batch["points"].to(device), batch["center_indices"].to(device), batch["neighbors"].to(device))
        inferred = logits.argmax(dim=-1).cpu().numpy()
        for truth, prediction in zip(batch["semantic"].numpy(), inferred):
            matrices.append(np.bincount(3 * truth + prediction, minlength=9).reshape(3, 3))
            predictions.append(prediction)
    return np.stack(matrices), np.stack(predictions)


def run_segmentation(root, partition, seed, condition, mode, device, replay_state=False):
    root = Path(root)
    protocol, lock = load_protocol(root)
    if partition not in protocol["partitions"] or seed not in protocol["seeds"] or mode not in ("probe", "finetune"):
        raise ValueError("Unregistered segmentation partition/seed/mode")
    if condition == "PointNet" and mode != "finetune":
        raise ValueError("Only fully supervised PointNet segmentation was registered")
    train, validation = partition_rows(root, partition, "train"), partition_rows(root, partition, "val")
    identifier = run_identifier(partition, seed, condition, 1.0, mode, "segmentation")
    output = (root / "outputs/m4/state_replays" / identifier if replay_state else
              root / "outputs/m4/checkpoints" / partition / identifier)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training.json").exists():
        record = json.loads((output / "training.json").read_text())
        if not artifact_protocol_matches(root, record["protocol_sha256"], lock):
            raise ValueError("Existing segmentation checkpoint protocol mismatch")
        return record
    frozen = mode == "probe"
    if condition == "PointNet":
        seed_all(seed)
        model, reference = PointNetSegmentation().to(device), None
    else:
        encoder, reference = source_encoder(root, protocol, partition, seed, condition)
        model = PointJEPASegmentation(encoder, frozen=frozen).to(device)
    dataset = datasets(root, train, validation)
    generator = torch.Generator().manual_seed(seed)
    loader = {split: DataLoader(value, batch_size=protocol["downstream"]["batch_size"], shuffle=split == "train",
                                generator=generator if split == "train" else None, num_workers=0) for split, value in dataset.items()}
    frequencies = np.bincount(np.concatenate(dataset["train"].labels), minlength=3).astype(np.float64)
    weights = np.minimum(10.0, np.sqrt(frequencies.max() / np.maximum(frequencies, 1)))
    weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    epochs = protocol["downstream"]["probe_epochs" if frozen else "epochs"]
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                  lr=protocol["downstream"]["probe_learning_rate" if frozen else "learning_rate"],
                                  weight_decay=protocol["downstream"]["weight_decay"])
    initial_head_hash = state_hash(model.head.state_dict())
    initial_encoder_hash = state_hash(model.encoder.state_dict()) if condition != "PointNet" else None
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, best_state, best_score, best_epoch, steps = [], None, -float("inf"), 0, 0
    print(f"SEG START {identifier}", flush=True)
    cached = None
    if frozen:
        cached = {}
        model.eval()
        with torch.no_grad():
            for split in ("train", "val"):
                ordered = DataLoader(dataset[split], batch_size=protocol["downstream"]["batch_size"], shuffle=False)
                values = []
                for batch in ordered:
                    features = model.point_features(batch["points"].to(device), batch["center_indices"].to(device), batch["neighbors"].to(device))
                    values.append(features.cpu())
                cached[split] = torch.cat(values).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        if frozen:
            permutation = torch.randperm(len(train), generator=generator)
            labels = torch.from_numpy(np.stack(dataset["train"].labels)).to(device)
            batches = [(permutation[start:start + protocol["downstream"]["batch_size"]])
                       for start in range(0, len(train), protocol["downstream"]["batch_size"])]
            for indices in batches:
                indices = indices.to(device)
                logits = model.head(cached["train"][indices])
                loss = criterion(logits.reshape(-1, 3), labels[indices].reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
        else:
            for batch in loader["train"]:
                logits = model(batch["points"].to(device), batch["center_indices"].to(device), batch["neighbors"].to(device))
                loss = criterion(logits.reshape(-1, 3), batch["semantic"].to(device).reshape(-1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                losses.append(float(loss.detach()))
        if frozen:
            with torch.no_grad():
                predicted = torch.cat([model.head(values.to(device)).argmax(dim=-1).cpu() for values in cached["val"].split(protocol["downstream"]["batch_size"])]).numpy()
            matrices = np.stack([np.bincount(3 * truth + prediction, minlength=9).reshape(3, 3)
                                 for truth, prediction in zip(dataset["val"].labels, predicted)])
        else:
            matrices, predicted = segmentation_predict(model, loader["val"], device)
        score = segmentation_metrics(matrices, validation)[0]["plant_macro_miou"]
        history.append({"epoch": epoch, "train_ce": float(np.mean(losses)), "val_plant_macro_miou": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            selected_state = model.head.state_dict() if frozen else model.state_dict()
            best_state = {name: value.detach().cpu().clone() for name, value in selected_state.items()}
    final_state = {name: value.detach().cpu().clone() for name, value in (model.head.state_dict() if frozen else model.state_dict()).items()}
    if frozen:
        model.head.load_state_dict(best_state, strict=True)
        with torch.no_grad():
            predicted = torch.cat([model.head(values.to(device)).argmax(dim=-1).cpu() for values in cached["val"].split(protocol["downstream"]["batch_size"])]).numpy()
        matrices = np.stack([np.bincount(3 * truth + prediction, minlength=9).reshape(3, 3)
                             for truth, prediction in zip(dataset["val"].labels, predicted)])
        if state_hash(model.encoder.state_dict()) != initial_encoder_hash:
            raise ValueError("Frozen segmentation encoder changed")
        state = {"head": best_state, "encoder_reference": reference, "random_encoder_seed": seed if condition == "P0" else None}
    else:
        model.load_state_dict(best_state, strict=True)
        matrices, predicted = segmentation_predict(model, loader["val"], device)
        state = {"model": best_state}
    metrics, plants = segmentation_metrics(matrices, validation)
    record = {"id": identifier, "partition": partition, "seed": seed, "condition": condition, "label_fraction": 1.0,
              "mode": mode, "task": "segmentation", "labelled_plants": sorted(train.plant_uid.unique()),
              "validation_plants": sorted(validation.plant_uid.unique()), "labelled_scans": len(train),
              "encoder_reference": reference, "initial_head_hash": initial_head_hash, "initial_encoder_hash": initial_encoder_hash,
              "protocol_sha256": lock["protocol_sha256"], "trainable_parameters": sum(value.numel() for value in model.parameters() if value.requires_grad),
              "total_parameters": sum(value.numel() for value in model.parameters()), "class_weights": weights.tolist(),
              "epochs": epochs, "best_epoch": best_epoch, "optimizer_steps": steps, "seconds": time.monotonic() - started,
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "test_used": False, "selection": "val plant-macro mIoU", "status": "COMPLETED", **metrics}
    for name in ("partition", "seed", "condition", "mode"):
        plants[name] = record[name]
    plants_path = output / "plants.csv" if replay_state else root / "outputs/m4/predictions" / f"{identifier}_plants.csv"
    points_path = output / "points.npz" if replay_state else root / "outputs/m4/predictions" / f"{identifier}_points.npz"
    plants.to_csv(plants_path, index=False)
    np.savez_compressed(points_path, prediction=predicted.astype(np.int8),
                        truth=np.stack(dataset["val"].labels).astype(np.int8), confusion=matrices,
                        plant_uid=validation.plant_uid.to_numpy(dtype=str), das=validation.DAS.to_numpy())
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    torch.save(state | {"metadata": record, "model_config": protocol["model"], "class_weights": weights.tolist()}, output / "best.pt")
    torch.save({"model": final_state, "optimizer": optimizer.state_dict(), "rng": rng_state(), "loader_rng": generator.get_state(),
                "final_epoch": epochs, "steps": steps, "metadata": record}, output / "training_state.pt")
    write_json(output / "training.json", record)
    print(f"SEG DONE {identifier} mIoU={metrics['plant_macro_miou']:.4f} seconds={record['seconds']:.1f}", flush=True)
    del model, optimizer, cached
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record
