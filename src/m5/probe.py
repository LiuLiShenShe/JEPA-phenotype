from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.sugar4d_ssl_dataset import Sugar4DSSLDataset, partition_rows
from m3.common import read_ply
from m4.common import write_json
from m4.metrics import paired_interval
from m5.common import load_protocol
from m5.downstream import source_encoder
from m5.diagnostics import representation_diagnostics


def token_leaf_labels(neighbors, semantic, instance, purity=.8):
    result = np.full(len(neighbors), -1, dtype=np.int64)
    for index, members in enumerate(neighbors):
        valid = (semantic[members] == 2) & (instance[members] >= 0)
        if valid.mean() < purity:
            continue
        values, counts = np.unique(instance[members][valid], return_counts=True)
        maximum = int(np.argmax(counts))
        if counts[maximum] / len(members) >= purity:
            result[index] = values[maximum]
    return result


def pair_metrics(features, centers, labels, seed=20261012):
    valid = np.flatnonzero(labels >= 0)
    if len(valid) < 4 or len(np.unique(labels[valid])) < 2:
        return None
    features, centers, labels = features[valid], centers[valid], labels[valid]
    first, second = np.triu_indices(len(valid), 1)
    same = labels[first] == labels[second]
    if not same.any() or same.all():
        return None
    distances = np.linalg.norm(centers[first] - centers[second], axis=1)
    generator = np.random.default_rng(seed)
    matches = []
    for same_index in np.flatnonzero(same):
        candidates = np.flatnonzero(~same)
        differences = np.abs(distances[candidates] - distances[same_index])
        minimum = differences.min()
        selected = generator.choice(candidates[np.isclose(differences, minimum, atol=1e-12)])
        matches.append((same_index, int(selected)))
    records = {}
    for mode, values in (("uncentred", features), ("centred", features - features.mean(0))):
        normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        cosine = (normalized[first] * normalized[second]).sum(1)
        same_cosine, different_cosine = cosine[same], cosine[~same]
        from sklearn.metrics import roc_auc_score
        margin = np.asarray([cosine[same_index] - cosine[different_index] for same_index, different_index in matches])
        records[mode] = {"same_leaf_cosine": float(same_cosine.mean()), "different_leaf_cosine": float(different_cosine.mean()),
                         "pair_auc": float(roc_auc_score(same.astype(np.int64), cosine)),
                         "distance_matched_margin": float(margin.mean()),
                         "distance_matched_accuracy": float((margin > 0).mean() + .5 * (margin == 0).mean())}
    records["pairs_same"] = int(same.sum())
    records["pairs_different"] = int((~same).sum())
    records["matched_mean_distance_difference"] = float(np.mean([abs(distances[left] - distances[right]) for left, right in matches]))
    return records


def probe_scans(root, partition):
    root = Path(root)
    destination = root / "outputs/m5/data/probe" / f"{partition}.npz"
    rows = partition_rows(root, partition, "train")
    if destination.exists():
        return rows, destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    for row in rows.itertuples(index=False):
        name = Path(row.pointcloud_path).stem
        vertices, _ = read_ply(root / row.pointcloud_path)
        with np.load(root / "outputs/m4/data/xyz" / f"{name}.npz") as cache:
            indices = cache["source_indices"]
            semantic = np.asarray(vertices["label_semantic"])[indices]
            instance = np.asarray(vertices["label_instance"])[indices]
            labels.append(token_leaf_labels(cache["neighbors"], semantic, instance))
    np.savez_compressed(destination, token_leaf_instance=np.stack(labels))
    return rows, destination


@torch.no_grad()
def run_probe(root, partition, seed, condition, device):
    root = Path(root)
    protocol, lock = load_protocol(root)
    folder = root / "outputs/m5/probes" / f"{partition}_{condition}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "summary.json").exists():
        record = json.loads((folder / "summary.json").read_text())
        if record["protocol_sha256"] != lock["protocol_sha256"]:
            raise ValueError("Probe protocol mismatch")
        return record
    rows, annotation_path = probe_scans(root, partition)
    with np.load(annotation_path) as archive:
        instance = archive["token_leaf_instance"]
    dataset = Sugar4DSSLDataset(rows, root / "outputs/m4/data/xyz")
    encoder, _ = source_encoder(root, protocol, partition, seed, condition)
    encoder.to(device).eval()
    scans, pooled_features, token_features = [], [], []
    for start in range(0, len(dataset), 64):
        samples = [dataset[index] for index in range(start, min(start + 64, len(dataset)))]
        batch = {field: torch.stack([sample[field] for sample in samples]).to(device)
                 for field in ("points", "center_indices", "neighbors")}
        pooled, tokens, centers = encoder(**batch, return_tokens=True)
        pooled_features.append(pooled.cpu())
        token_features.append(tokens.cpu())
        for offset, (features, locations) in enumerate(zip(tokens.cpu().numpy(), centers.cpu().numpy())):
            index = start + offset
            metrics = pair_metrics(features, locations, instance[index])
            if metrics is not None:
                scans.append({"plant_uid": rows.iloc[index].plant_uid, "DAS": int(rows.iloc[index].DAS),
                              **{f"{mode}_{name}": value for mode in ("uncentred", "centred") for name, value in metrics[mode].items()},
                              **{name: metrics[name] for name in ("pairs_same", "pairs_different", "matched_mean_distance_difference")}})
    frame = pd.DataFrame(scans)
    if frame.empty:
        raise ValueError("No valid leaf-token probe observations")
    columns = [column for column in frame if column not in ("plant_uid", "DAS")]
    plants = frame.groupby("plant_uid")[columns].mean().reset_index()
    frame.to_csv(folder / "scans.csv", index=False)
    plants.to_csv(folder / "plants.csv", index=False)
    record = {"partition": partition, "condition": condition, "seed": seed, "protocol_sha256": lock["protocol_sha256"],
              "scans": len(frame), "train_plants": len(plants), "labels_used_to_train_encoder": False,
              "scope": "training-plant read-only instance probe, not generalization or formal leaf instance segmentation",
              "purity": .8, **{column: float(plants[column].mean()) for column in columns}, "status": "COMPLETED"}
    write_json(folder / "summary.json", record)
    diagnostic = representation_diagnostics(torch.cat(token_features), torch.cat(pooled_features), rows.plant_uid.to_numpy())
    diagnostic.update({"condition": condition, "partition": partition, "seed": seed, "selection_use": "none; posttraining evaluation only"})
    write_json(folder / "representation.json", diagnostic)
    print(f"PROBE {partition} {condition} seed={seed} centred_auc={record['centred_pair_auc']:.4f}", flush=True)
    return record
