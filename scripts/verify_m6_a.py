from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from m6a.adapters import OfficialSegmenter, OfficialTracker
from m6a.common import digest, freeze, localize_gt, read_cloud, verify_protection, write_json
from m6a.experiment import initialize_runtime, metadata
from m6a.metrics import frame_metrics


def verify(repeat=False):
    freeze(ROOT)
    destination = ROOT / "reports/m6_a"
    records = metadata(ROOT)
    records = records[records.split.isin(["val", "test"])]
    results = pd.read_csv(destination / "per_plant.csv")
    frames = pd.read_csv(destination / "frame_metrics.csv")
    observed = pd.read_csv(destination / "observations.csv")
    errors = pd.read_csv(destination / "measurement_errors.csv")
    decomposition = pd.read_csv(destination / "error_decomposition.csv")
    assert len(results) == 96 and results.plant_uid.nunique() == 24
    assert len(frames) == 1536 and frames.groupby(["plant_uid", "pipeline"]).size().eq(16).all()
    assert not results.duplicated(["plant_uid", "pipeline"]).any()
    assert not observed.duplicated(["plant_uid", "pipeline", "DAS", "gt_id"]).any()
    assert not errors.duplicated(["plant_uid", "pipeline", "DAS", "gt_id", "kind", "trait", "stage"]).any()
    assert decomposition.additivity_residual.abs().max() < 1e-12
    for metric in ["pq", "sbd", "idf1", "new_event_f1", "continuity"]:
        assert results[metric].between(0, 1).all()
    assert results[results.pipeline == "B0"].pq.eq(1).all()
    b1 = frames[frames.pipeline == "B1"].set_index(["plant_uid", "DAS"])
    b2 = frames[frames.pipeline == "B2"].set_index(["plant_uid", "DAS"])
    assert np.array_equal(b1.pq, b2.pq)
    assert np.array_equal(b1.sbd, b2.sbd)
    cache_manifest = []
    protocol_hash = digest(destination / "protocol_lock.json")
    for record in records.itertuples(index=False):
        path = ROOT / f"outputs/m6_a/frames/{record.plant_uid}_{record.DAS}.npz"
        with np.load(path) as cached:
            assert cached["lock_hash"].item() == protocol_hash
            assert cached["sample_xyz"].shape == (4096, 3)
            assert cached["tracking_xyz"].shape == (256, 3)
            assert len(cached["labels"]) == record.point_count
            assert len(cached["gt_local"]) == record.point_count
        cache_manifest.append({"plant_uid": record.plant_uid, "DAS": record.DAS, "file": str(path.relative_to(ROOT)), "sha256": digest(path)})
    repeats = []
    repeat_destination = destination / "real_frame_repeatability.json"
    if not repeat and repeat_destination.exists():
        repeats = json.loads(repeat_destination.read_text())
    if repeat:
        initialize_runtime()
        model = OfficialSegmenter(ROOT)
        tracker = OfficialTracker(ROOT)
        for record in [records[(records.plant_uid == "plant_001") & (records.DAS == 37)].iloc[0],
                       records[(records.plant_uid == "plant_046") & (records.DAS == 93)].iloc[0]]:
            cloud, gt, unused_info = read_cloud(ROOT / record.pointcloud_path)
            seed = 20261012 + int(record.plant_id) * 100 + int(record.DAS)
            first = model.predict(cloud, seed)
            sample_indices, unused_sampling = tracker.sample_indices(cloud, seed)
            path = ROOT / f"outputs/m6_a/frames/{record.plant_uid}_{record.DAS}.npz"
            with np.load(path) as cached:
                result = {"plant_uid": record.plant_uid, "DAS": int(record.DAS),
                          "labels_repeat_exact": bool(np.array_equal(first["labels"], cached["labels"])),
                          "confidence_repeat_exact": bool(np.array_equal(first["confidence"], cached["confidence"])),
                          "sample_indices_repeat_exact": bool(np.array_equal(first["sample_indices"], cached["sample_indices"])),
                          "track_sampled_gt_masks_repeat_exact": bool(np.array_equal(localize_gt(gt, seed)[sample_indices], cached["tracking_gt"])),
                          "relabelled_gt_pq": float(frame_metrics(gt, localize_gt(gt, seed + 1))[0]["pq"])}
                assert result["labels_repeat_exact"] and result["sample_indices_repeat_exact"]
                assert result["track_sampled_gt_masks_repeat_exact"]
                repeats.append(result)
        write_json(repeat_destination, repeats)
    pd.DataFrame(cache_manifest).to_csv(destination / "cache_manifest.csv", index=False)
    protection = verify_protection(ROOT)
    write_json(destination / "final_verification.json", {"full_resolution_frames": len(records), "independent_plants": 24,
               "all_prespecified_plants_retained": True, "per_plant_pipeline_rows": len(results), "metric_bounds_valid": True,
               "B1_B2_same_masks_and_PQ": True, "all_cache_hashes_valid": True, "decomposition_additivity_valid": True,
               "real_frame_repeats": repeats, "historical_protection": protection,
               "protocol_sha256": digest(ROOT / "configs/m6_a/protocol.json"), "training_steps": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    verify(args.repeat)
