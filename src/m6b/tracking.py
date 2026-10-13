from __future__ import annotations

import json

import numpy as np
import pandas as pd

from m6a.adapters import OfficialTracker
from m6a.common import digest, localize_gt, write_json
from m6a.experiment import cache_path, measurement_losses, metadata, summarize_losses
from m6a.metrics import event_metrics, frame_metrics, measurements, observation_identity
from m6b.common import check_lock
from m6b.data import read_frame
from m6b.evaluation import prediction_path, run_dir


def track(root, method, seed, splits):
    lock_hash = check_lock(root)
    tracker = OfficialTracker(root)
    directory = run_dir(root, method, seed)
    prediction_manifest = json.loads((directory / "postprocessing_selection.json").read_text())
    leaf = pd.read_csv(root / "data/processed/sugar4d/leaf_metadata.csv")
    reference_by_plant = {}
    for plant, group in leaf.groupby("plant_uid"):
        reference_by_plant[plant] = {(int(row["DAS"]), int(row["leaf_instance_id"]) + 1):
                                    (float(row["total length [cm]"]), float(row["leaf area [cm2]"]))
                                     for unused_index, row in group.iterrows()}
    source_lock_hash = digest(root / "reports/m6_a/protocol_lock.json")
    records = metadata(root)
    records = records[records.split.isin(splits)]
    for split in splits:
        results, observations_all, organs_all, errors_all, frames_all, births_all = [], [], [], [], [], []
        for plant, scans in records.query("split == @split").groupby("plant_uid"):
            completed = directory / f"tracking/{plant}.json"
            if completed.exists():
                content = json.loads(completed.read_text())
                if content["protocol_lock_sha256"] != lock_hash or content["checkpoint_sha256"] != prediction_manifest["checkpoint_sha256"]:
                    raise ValueError("Tracking cache provenance mismatch")
            else:
                frames, observations, organs, births, frame_scores = [], [], [], [], []
                next_track = 0
                for frame_index, record in enumerate(scans.itertuples(index=False)):
                    data = read_frame(root / record.pointcloud_path)
                    gt = np.where((data["semantic"] == 2) & (data["original_instance_evaluation_only"] >= 0),
                                  data["original_instance_evaluation_only"] + 1, 0).astype(np.int32)
                    with np.load(prediction_path(root, method, seed, plant, record.DAS)) as prediction:
                        source_labels = prediction["labels"].copy()
                        confidence_point = prediction["confidence"].copy()
                    local_seed = 20261013 + record.plant_id * 100 + record.DAS
                    labels = localize_gt(source_labels, local_seed)
                    with np.load(cache_path(root, plant, record.DAS)) as cache:
                        if cache["lock_hash"].item() != source_lock_hash:
                            raise ValueError("Historical geometry cache hash mismatch")
                        indices = tracker_indices(root, record)
                        current_xyz = cache["tracking_xyz"].copy()
                        deformed = cache["deformed_xyz"].copy()
                    sample_labels = labels[indices]
                    current_labels = sample_labels
                    sampled_ids = np.unique(current_labels)
                    if frame_index == 0:
                        tracks = {int(label): next_track + offset for offset, label in enumerate(sampled_ids)}
                        margins = {int(label): 1.0 for label in sampled_ids}
                        born = set(tracks)
                        next_track += len(sampled_ids)
                    else:
                        previous = frames[-1]
                        previous_tracks = np.array([previous["tracks"].get(int(label), -1) for label in previous["input_labels"]])
                        tracks, margins, born, next_track = tracker.assign(previous["input_xyz"], previous_tracks, deformed, current_labels, next_track)
                    for label in np.unique(labels):
                        tracks.setdefault(int(label), -1)
                    frame_score, matches = frame_metrics(gt, labels)
                    association = {int(gt_id): matches.get(int(gt_id), (-1, 0.0)) for gt_id in np.unique(gt[gt > 0])}
                    gt_association = {int(gt_id): (int(gt_id), 1.0) for gt_id in np.unique(gt[gt > 0])}
                    measure, gt_measure = measurements(data["xyz"], labels), measurements(data["xyz"], gt)
                    confidences = {int(local): float(confidence_point[labels == local].mean()) * margins.get(int(local), 0)
                                   for local in measure}
                    for local_id in measure:
                        organs.append({"plant_uid": plant, "split": split, "method": method, "seed": seed, "frame_index": frame_index,
                                       "DAS": int(record.DAS), "local_id": local_id, "track_id": tracks[local_id],
                                       "length_cm": measure[local_id][0], "area_cm2": measure[local_id][1],
                                       "new_track": local_id in born, "sample_support": int(np.sum(sample_labels == local_id))})
                        if local_id in born:
                            births.append({"frame_index": frame_index, "local_id": local_id, "track_id": tracks[local_id]})
                    for gt_id, (local_id, iou) in association.items():
                        observations.append({"plant_uid": plant, "split": split, "method": method, "seed": seed,
                                             "frame_index": frame_index, "DAS": int(record.DAS), "gt_id": gt_id, "local_id": local_id,
                                             "track_id": tracks.get(local_id, -1), "iou": iou,
                                             "state": "instance_missed" if local_id < 0 else "sampling_or_identity_lost" if tracks.get(local_id, -1) < 0 else "detected"})
                    frame_score.update({"plant_uid": plant, "split": split, "method": method, "seed": seed, "DAS": int(record.DAS),
                                        "sampling_drop_instances": sum(tracks[local] < 0 for local in measure)})
                    frame_scores.append(frame_score)
                    frames.append({"DAS": int(record.DAS), "association": association, "gt_association": gt_association,
                                   "measure": measure, "gt_measure": gt_measure, "tracks": tracks, "confidence": confidences,
                                   "input_xyz": current_xyz, "input_labels": sample_labels})
                identity, alignment = observation_identity(observations, sum(row["pred_count"] for row in frame_scores))
                events = event_metrics(observations, births)
                losses = measurement_losses(plant, split, method, frames, reference_by_plant[plant])
                for row in observations:
                    row["evaluation_identity_correct"] = row["track_id"] >= 0 and alignment.get(row["gt_id"]) == row["track_id"]
                    row["error_type"] = "instance_missed" if row["local_id"] < 0 else "sampling_or_identity_lost" if row["track_id"] < 0 else "wrong_identity" if not row["evaluation_identity_correct"] else "correct"
                result = {"plant_uid": plant, "split": split, "method": method, "seed": seed,
                          "pq": float(np.mean([row["pq"] for row in frame_scores])), "sbd": float(np.mean([row["sbd"] for row in frame_scores])),
                          **identity, **events, **summarize_losses(losses)}
                content = {"protocol_lock_sha256": lock_hash, "checkpoint_sha256": prediction_manifest["checkpoint_sha256"],
                           "result": result, "observations": observations, "organs": organs,
                           "frames": frame_scores, "births": births, "losses": losses}
                directory.joinpath("tracking").mkdir(exist_ok=True)
                serializable = json.loads(json.dumps(content, default=lambda value: value.item() if isinstance(value, np.generic) else value))
                completed.write_text(json.dumps(serializable, allow_nan=True))
            results.append(content["result"])
            observations_all.extend(content["observations"])
            organs_all.extend(content["organs"])
            frames_all.extend(content["frames"])
            errors_all.extend(content["losses"])
            births_all.extend([{**row, "plant_uid": plant, "split": split, "method": method, "seed": seed} for row in content["births"]])
            print("TRACK", method, seed, split, plant, content["result"]["idf1"], flush=True)
        for name, rows in [("tracking_plants", results), ("observations", observations_all), ("predicted_organs", organs_all),
                           ("tracking_frames", frames_all), ("measurement_errors", errors_all), ("new_track_events", births_all)]:
            pd.DataFrame(rows).to_csv(directory / f"{name}_{split}.csv", index=False)
        write_json(directory / f"tracking_manifest_{split}.json", {"protocol_lock_sha256": lock_hash,
                   "plants": len(results), "M6_A_geometry_lock_sha256": source_lock_hash,
                   "no_GT_temporal_ID_input": True, "independent_frame_renumbering": True,
                   "measurement_function_unchanged": "m6a.experiment.measurement_losses; m6a.metrics.measurements",
                   "tracker_configuration_unchanged": True})


def tracker_indices(root, record):
    path = root / f"outputs/m6_b/tracker_geometry/{record.plant_uid}_{record.DAS}.npy"
    if path.exists():
        return np.load(path)
    tracker = OfficialTracker(root)
    data = read_frame(root / record.pointcloud_path)
    with np.load(cache_path(root, record.plant_uid, record.DAS)) as cache:
        legacy_radius = "equivalent_neighbor_radius" not in json.loads(cache["timing"].item())
    if legacy_radius:
        import open3d as open3d
        from unittest.mock import patch

        original_boundary = open3d.t.geometry.PointCloud.compute_boundary_points

        def fixed_legacy_boundary(pointcloud, **kwargs):
            return original_boundary(pointcloud, **{**kwargs, "radius": 10.0})

        with patch.object(open3d.t.geometry.PointCloud, "compute_boundary_points", fixed_legacy_boundary):
            indices, unused_info = tracker.sample_indices(data["xyz"], 20261012 + record.plant_id * 100 + record.DAS)
    else:
        indices, unused_info = tracker.sample_indices(data["xyz"], 20261012 + record.plant_id * 100 + record.DAS)
    from m6a.common import normalize
    with np.load(cache_path(root, record.plant_uid, record.DAS)) as cache:
        if not np.array_equal(normalize(data["xyz"][indices]), cache["tracking_xyz"]):
            raise ValueError("Tracking sampling is not identical to M6-A")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, indices)
    return indices
