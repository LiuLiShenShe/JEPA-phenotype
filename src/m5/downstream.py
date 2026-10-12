from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.sugar4d_ssl_dataset import partition_rows
from m4.common import rng_state, seed_all, standardize_fit, state_hash, write_json
from m4.downstream import count_frame, count_predict, datasets, geometry_rows, segmentation_predict
from m4.downstream import source_encoder as old_encoder
from m4.metrics import count_metrics, segmentation_metrics
from m5.common import load_protocol
from models.point_jepa_downstream import PointJEPACount, PointJEPASegmentation
from models.point_jepa_encoder import PointJEPAEncoder, load_encoder_checkpoint


def source_encoder(root, protocol, partition, seed, condition):
    root = Path(root)
    if condition in ("P0", "P2"):
        return old_encoder(root, protocol, partition, seed, condition)
    if condition not in protocol["conditions"] or partition not in protocol["partitions"] or seed not in protocol["seeds"]:
        raise ValueError("Unregistered M5 condition")
    seed_all(seed)
    encoder = PointJEPAEncoder(**protocol["model"])
    reference = root / "outputs/m5/checkpoints" / partition / f"{condition}_seed{seed}" / "encoder.pt"
    checkpoint = torch.load(reference, map_location="cpu", weights_only=False)
    expected = sorted(partition_rows(root, partition, "train").plant_uid.unique())
    record = checkpoint["metadata"]
    _, lock = load_protocol(root)
    if record["train_plants"] != expected or record["test_used"] or record["validation_used"] or record["specification_sha256"] != lock["protocol_sha256"]:
        raise ValueError("M5 SSL provenance mismatch")
    load_encoder_checkpoint(encoder, reference)
    return encoder, str(reference.relative_to(root))


def identifier(partition, seed, condition, fraction, task):
    return f"{partition}_{task}_{condition}_finetune_labels{int(fraction * 100)}_seed{seed}"


def run_downstream(root, partition, seed, condition, fraction, task, device):
    root = Path(root)
    protocol, lock = load_protocol(root)
    if task not in ("count", "segmentation") or fraction not in protocol["label_fractions"]:
        raise ValueError("Unregistered downstream task/budget")
    if task == "segmentation" and fraction != 1.0:
        raise ValueError("Semantic auxiliary task is fixed to 100% labels")
    selected = lock["label_budgets"][partition][str(fraction)]
    train_rows, validation = partition_rows(root, partition, "train"), partition_rows(root, partition, "val")
    train_rows = train_rows.loc[train_rows.plant_uid.isin(selected)].reset_index(drop=True)
    name = identifier(partition, seed, condition, fraction, task)
    folder = root / "outputs/m5/downstream" / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "training.json").exists():
        record = json.loads((folder / "training.json").read_text())
        if record["protocol_sha256"] != lock["protocol_sha256"]:
            raise ValueError("Downstream protocol mismatch")
        return record
    encoder, reference = source_encoder(root, protocol, partition, seed, condition)
    geometry, geometry_columns = geometry_rows(root, train_rows)
    geometry_state = standardize_fit(geometry)
    target_state = standardize_fit(train_rows.visible_organ_count.to_numpy()[:, None])
    dataset = datasets(root, train_rows, validation, geometry_state if task == "count" else None)
    generator = torch.Generator().manual_seed(seed)
    loaders = {split: DataLoader(value, batch_size=protocol["downstream"]["batch_size"], shuffle=split == "train",
                                 generator=generator if split == "train" else None, num_workers=0)
               for split, value in dataset.items()}
    model = (PointJEPACount(encoder, len(geometry_columns)) if task == "count" else PointJEPASegmentation(encoder)).to(device)
    head_hash, encoder_hash = state_hash(model.head.state_dict()), state_hash(model.encoder.state_dict())
    weights = None
    if task == "segmentation":
        frequencies = np.bincount(np.concatenate(dataset["train"].labels), minlength=3).astype(np.float64)
        weights = np.minimum(10., np.sqrt(frequencies.max() / np.maximum(frequencies, 1)))
        weights /= weights.mean()
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["downstream"]["learning_rate"],
                                  weight_decay=protocol["downstream"]["weight_decay"])
    history, best_state, best_epoch, steps = [], None, 0, 0
    best_score = float("inf") if task == "count" else -float("inf")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, protocol["downstream"]["epochs"] + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            points, centers, neighbors = (batch[field].to(device) for field in ("points", "center_indices", "neighbors"))
            if task == "count":
                predicted = model(points, batch["geometry"].to(device), centers, neighbors)
                target = (batch["count"].to(device) - target_state["mean"][0]) / target_state["scale"][0]
                loss = torch.nn.functional.mse_loss(predicted, target)
            else:
                logits = model(points, centers, neighbors)
                loss = criterion(logits.reshape(-1, 3), batch["semantic"].to(device).reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), protocol["downstream"]["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            losses.append(float(loss.detach()))
        if task == "count":
            values = count_predict(model, loaders["val"], target_state, device)
            score = count_metrics(count_frame(validation, values))[0]["plant_macro_mae"]
            improved = score < best_score
        else:
            matrices, values = segmentation_predict(model, loaders["val"], device)
            score = segmentation_metrics(matrices, validation)[0]["plant_macro_miou"]
            improved = score > best_score
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation_score": score})
        if improved:
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state, strict=True)
    if task == "count":
        values = count_predict(model, loaders["val"], target_state, device)
        predictions = count_frame(validation, values)
        metrics, plants = count_metrics(predictions)
        predictions.to_csv(folder / "predictions.csv", index=False)
    else:
        matrices, values = segmentation_predict(model, loaders["val"], device)
        metrics, plants = segmentation_metrics(matrices, validation)
        np.savez_compressed(folder / "predictions.npz", prediction=values.astype(np.int8), confusion=matrices,
                            truth=np.stack(dataset["val"].labels).astype(np.int8))
    record = {"id": name, "partition": partition, "seed": seed, "condition": condition, "label_fraction": fraction,
              "task": task, "mode": "finetune", "labelled_plants": selected, "labelled_scans": len(train_rows),
              "validation_plants": sorted(validation.plant_uid.unique()), "encoder_reference": reference,
              "initial_head_hash": head_hash, "initial_encoder_hash": encoder_hash, "protocol_sha256": lock["protocol_sha256"],
              "trainable_parameters": sum(value.numel() for value in model.parameters()),
              "epochs": protocol["downstream"]["epochs"], "optimizer_steps": steps, "best_epoch": best_epoch,
              "seconds": time.monotonic() - started, "test_used": False, "status": "COMPLETED",
              "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "historical": False, **metrics}
    torch.save({"format": "m5-downstream-v1", "model": best_state, "metadata": record, "model_config": protocol["model"],
                "geometry_columns": geometry_columns, "geometry_scaler": geometry_state, "target_scaler": target_state,
                "class_weights": weights.tolist() if weights is not None else None}, folder / "best.pt")
    torch.save({"format": "m5-downstream-training-v1", "model": final_state, "optimizer": optimizer.state_dict(),
                "rng": rng_state(), "loader_rng": generator.get_state(), "step": steps, "epoch": protocol["downstream"]["epochs"],
                "metadata": record}, folder / "last.pt")
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    plants.to_csv(folder / "plants.csv", index=False)
    write_json(folder / "training.json", record)
    score_name = "plant_macro_mae" if task == "count" else "plant_macro_miou"
    print(f"DOWNSTREAM {name} {score_name}={metrics[score_name]:.5f}", flush=True)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


@torch.no_grad()
def verify_restore(root, folder, device):
    root, folder = Path(root), Path(folder)
    protocol, lock = load_protocol(root)
    checkpoint = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
    record = checkpoint["metadata"]
    if record["protocol_sha256"] != lock["protocol_sha256"]:
        raise ValueError("Restore provenance mismatch")
    encoder, _ = source_encoder(root, protocol, record["partition"], record["seed"], record["condition"])
    model = (PointJEPACount(encoder, len(checkpoint["geometry_columns"])) if record["task"] == "count" else PointJEPASegmentation(encoder))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    train_rows, validation = partition_rows(root, record["partition"], "train"), partition_rows(root, record["partition"], "val")
    dataset = datasets(root, train_rows, validation, checkpoint["geometry_scaler"] if record["task"] == "count" else None)["val"]
    loader = DataLoader(dataset, batch_size=protocol["downstream"]["batch_size"], shuffle=False)
    if record["task"] == "count":
        predicted = count_predict(model, loader, checkpoint["target_scaler"], device)
        expected = pd.read_csv(folder / "predictions.csv").prediction.to_numpy()
        difference = float(np.abs(predicted - expected).max())
        if difference > 2e-6:
            raise ValueError("Count checkpoint predictions changed")
        return {"id": record["id"], "status": "PASS", "maximum_absolute_difference": difference}
    matrices, predicted = segmentation_predict(model, loader, device)
    with np.load(folder / "predictions.npz") as saved:
        if not np.array_equal(predicted, saved["prediction"]) or not np.array_equal(matrices, saved["confusion"]):
            raise ValueError("Segmentation checkpoint predictions changed")
    return {"id": record["id"], "status": "PASS", "exact": True}
