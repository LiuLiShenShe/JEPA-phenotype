from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift

from m6a.common import digest, write_json
from m6a.experiment import metadata
from m6a.metrics import frame_metrics
from m6b.common import check_lock
from m6b.data import read_frame, transform


def run_dir(root, method, seed):
    return root / f"outputs/m6_b/{method}/seed_{seed}"


def prediction_path(root, method, seed, plant, das):
    return run_dir(root, method, seed) / f"predictions/{plant}_{das}.npz"


def sampled_forward(model, xyz, seed, device):
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Only XYZ may enter the segmenter")
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(xyz), min(4096, len(xyz)), replace=False)
    normalized, center, scale = transform(xyz)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        logits, embedding, similarity = model(torch.from_numpy(normalized[indices][None]).to(device))
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        features = embedding[0].cpu().numpy()
    torch.cuda.synchronize()
    return {"sample_indices": indices, "semantic_probability": probabilities, "embedding": features,
            "center_m": center, "scale_m": scale, "forward_seconds": time.perf_counter() - started,
            "peak_vram_bytes": torch.cuda.max_memory_allocated()}


def cluster_prediction(xyz, sampled, bandwidth):
    semantic = sampled["semantic_probability"].argmax(axis=1)
    leaf = semantic == 2
    labels = np.zeros(len(semantic), dtype=np.int32)
    if np.any(leaf):
        clusters = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=1).fit_predict(sampled["embedding"][leaf])
        labels[leaf] = clusters + 1
    neighbor = cKDTree(xyz[sampled["sample_indices"]]).query(xyz, workers=4)[1]
    return {"labels": labels[neighbor], "confidence": sampled["semantic_probability"].max(axis=1)[neighbor],
            "semantic": semantic[neighbor].astype(np.int8), "sample_labels": labels}


def gt_labels(data):
    return np.where(data["local_instance"] >= 0, data["local_instance"] + 1, 0).astype(np.int32)


def evaluate_model(root, model, device, split="val", bandwidths=(.6,), save=None):
    records = metadata(root).query("split == @split")
    model.eval()
    rows = []
    for record in records.itertuples(index=False):
        data = read_frame(root / record.pointcloud_path)
        seed = 20261012 + record.plant_id * 100 + record.DAS
        sampled = sampled_forward(model, data["xyz"], seed, device)
        for bandwidth in bandwidths:
            started = time.perf_counter()
            predicted = cluster_prediction(data["xyz"], sampled, bandwidth)
            score, matches = frame_metrics(gt_labels(data), predicted["labels"])
            row = {"plant_uid": record.plant_uid, "split": split, "DAS": int(record.DAS), "bandwidth": bandwidth,
                   **score, "forward_seconds": sampled["forward_seconds"], "cluster_restore_seconds": time.perf_counter() - started,
                   "peak_vram_bytes": sampled["peak_vram_bytes"]}
            rows.append(row)
            if save is not None and len(bandwidths) == 1:
                path = save / f"{record.plant_uid}_{record.DAS}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **predicted, sample_indices=sampled["sample_indices"],
                                    semantic_probability=sampled["semantic_probability"], embedding=sampled["embedding"],
                                    center_m=sampled["center_m"], scale_m=sampled["scale_m"], bandwidth=bandwidth)
    frame = pd.DataFrame(rows)
    plants = frame.groupby(["bandwidth", "plant_uid"])[["pq", "sbd"]].mean().reset_index()
    means = plants.groupby("bandwidth")[["pq", "sbd"]].mean().reset_index()
    return frame, plants, means


def infer(root, method, seed, device, splits):
    from m6b.models import official_model

    lock_hash = check_lock(root)
    directory = run_dir(root, method, seed)
    model_name = "PSegNet" if method == "S3" else "PlantNet"
    model, unused_loading = official_model(root, model_name, False, device)
    checkpoint_path = directory / "best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint["protocol_lock_sha256"] != lock_hash:
        raise ValueError("Checkpoint protocol mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    protocol = json.loads((root / "configs/m6_b/protocol.json").read_text())
    selection_path = directory / "postprocessing_selection.json"
    if not selection_path.exists():
        if "val" not in splits:
            raise ValueError("Must select postprocessing on Validation before Test")
        scores, plants, means = evaluate_model(root, model, device, "val", protocol["postprocessing"]["bandwidth_candidates"])
        means["tie_preference"] = (means.bandwidth != .6).astype(int)
        selected = means.sort_values(["pq", "tie_preference", "bandwidth"], ascending=[False, True, True]).iloc[0]
        scores.to_csv(directory / "bandwidth_validation_frames.csv", index=False)
        plants.to_csv(directory / "bandwidth_validation_plants.csv", index=False)
        means.to_csv(directory / "bandwidth_validation_means.csv", index=False)
        write_json(selection_path, {"selection_split": "val", "bandwidth": float(selected.bandwidth),
                                   "pq": float(selected.pq), "checkpoint_epoch": int(checkpoint["epoch"]),
                                   "checkpoint_sha256": digest(checkpoint_path), "protocol_lock_sha256": lock_hash,
                                   "no_test_selection": True})
    selection = json.loads(selection_path.read_text())
    if selection["checkpoint_sha256"] != digest(checkpoint_path) or selection["protocol_lock_sha256"] != lock_hash:
        raise ValueError("Postprocessing checkpoint mismatch")
    for split in splits:
        frames, plants, means = evaluate_model(root, model, device, split, (selection["bandwidth"],), directory / "predictions")
        frames.to_csv(directory / f"segmentation_frames_{split}.csv", index=False)
        plants.to_csv(directory / f"segmentation_plants_{split}.csv", index=False)
        write_json(directory / f"inference_{split}.json", {"method": method, "seed": seed, "split": split,
                   "frames": len(frames), "plants": len(plants), "checkpoint_epoch": int(checkpoint["epoch"]),
                   "checkpoint_sha256": digest(checkpoint_path), "protocol_lock_sha256": lock_hash,
                   "bandwidth": selection["bandwidth"], "pq": float(means.pq.iloc[0]), "sbd": float(means.sbd.iloc[0]),
                   "full_resolution": True, "test_exploratory": split == "test"})
        print(method, seed, split, means.to_dict(orient="records"), flush=True)
