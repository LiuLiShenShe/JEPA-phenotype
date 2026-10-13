from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from m6a.adapters import OfficialSegmenter
from m6a.common import digest, read_cloud, write_json
from m6a.experiment import cache_path, initialize_runtime, metadata
from m6a.metrics import frame_metrics, measurements


def run(splits, device, evaluate_only=False):
    import datetime

    configuration = "configs/m6_a/official_normalization_diagnostic.json"
    hashes = {configuration: digest(ROOT / configuration),
              "primary_protocol_lock": digest(ROOT / "reports/m6_a/protocol_lock.json"),
              "official_H5_source": digest(ROOT / "reports/m6_a/sources/002TXT2H5.py.source.txt")}
    lock = ROOT / "reports/m6_a/official_normalization_lock.json"
    if lock.exists():
        if json.loads(lock.read_text())["sha256"] != hashes:
            raise ValueError("Official-normalization diagnostic changed")
    else:
        write_json(lock, {"utc_before_diagnostic_inference": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sha256": hashes,
                          "status": "post-primary-lock diagnostic; never supersedes original primary"})
    initialize_runtime()
    records = metadata(ROOT)
    records = records[records.split.isin(splits)].sort_values(["plant_uid", "DAS"])
    model = None if evaluate_only else OfficialSegmenter(ROOT, "PlantNet", device)
    directory = ROOT / "outputs/m6_a/official_normalization"
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records.itertuples(index=False)):
        original = cache_path(ROOT, record.plant_uid, record.DAS)
        path = directory / "frames" / original.name
        if path.exists():
            continue
        if evaluate_only:
            raise FileNotFoundError(path)
        while not original.exists():
            time.sleep(5)
        with np.load(original) as cached:
            data = {key: cached[key] for key in cached.files}
        xyz, unused_gt, unused_info = read_cloud(ROOT / record.pointcloud_path)
        seed = 20261012 + record.plant_id * 100 + record.DAS
        prediction = model.predict(xyz, seed, official_normalization=True)
        if not np.array_equal(prediction["sample_indices"], data["sample_indices"]):
            raise ValueError("Diagnostic point sample changed")
        measure = measurements(xyz, prediction["labels"])
        ids = np.unique(prediction["labels"])
        from m6a.common import normalize

        norm = normalize(xyz)
        data.update(labels=prediction["labels"], confidence=prediction["confidence"], sample_labels=prediction["sample_labels"],
                    pred_measure_ids=np.array(list(measure)), pred_measure_values=np.array(list(measure.values())).reshape(-1, 2),
                    pred_centroid_ids=ids, pred_centroids=np.array([norm[prediction["labels"] == label].mean(axis=0) for label in ids]))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **data)
        print("OFFICIAL-NORM", index + 1, len(records), record.plant_uid, record.DAS, len(measure), flush=True)
    if not evaluate_only:
        return
    all_results = []
    for record in records.itertuples(index=False):
        with np.load(directory / "frames" / f"{record.plant_uid}_{record.DAS}.npz") as cached:
            labels = cached["labels"]
        unused_xyz, gt, unused_info = read_cloud(ROOT / record.pointcloud_path)
        score, unused_association = frame_metrics(gt, labels)
        all_results.append({"plant_uid": record.plant_uid, "split": record.split, "DAS": record.DAS, **score})
    scores = pd.DataFrame(all_results)
    scores.to_csv(ROOT / "reports/m6_a/official_normalization_frame_metrics.csv", index=False)
    results = scores.groupby(["plant_uid", "split"]).mean(numeric_only=True).reset_index()
    results.to_csv(ROOT / "reports/m6_a/official_normalization_per_plant.csv", index=False)
    results.groupby("split").mean(numeric_only=True).reset_index().to_csv(ROOT / "reports/m6_a/official_normalization_metrics.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    run(args.splits, args.device, args.evaluate_only)
