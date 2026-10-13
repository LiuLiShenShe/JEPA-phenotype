from __future__ import annotations

import datetime
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from m6a.adapters import OfficialSegmenter, OfficialTracker, simple_assign
from m6a.common import digest, freeze, localize_gt, normalize, paired_interval, read_cloud, write_json
from m6a.metrics import event_metrics, frame_metrics, geometry_measure, measurements, observation_identity, penalized_error


PIPELINES = ["B0", "B1", "B2", "B2_GT"]


def cache_path(root, plant, das):
    return root / f"outputs/m6_a/frames/{plant}_{das}.npz"


def metadata(root):
    frame = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    fixed = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]["official"]
    if not all(frame["split"] == frame["plant_uid"].map(fixed)):
        raise ValueError("Plant split differs from frozen historical official split")
    if frame.groupby("plant_uid").size().to_dict() != {name: 16 for name in fixed}:
        raise ValueError("Actual 16-scan grid changed")
    return frame


def initialize_runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def smoke(root, device):
    freeze(root)
    initialize_runtime()
    rng = np.random.default_rng(20261012)
    synthetic = rng.normal(size=(4096, 3)).astype(np.float32)
    reports = {}
    for name in ["PlantNet", "PSegNet"]:
        started = time.perf_counter()
        try:
            model = OfficialSegmenter(root, name, device)
            first = model.predict(synthetic, 20261012)
            second = model.predict(synthetic, 20261012)
            reports[name] = {"runnable": True, "checkpoint_epoch": model.epoch,
                             "strict_state_dict_load": True, "input_columns": 3, "synthetic_points": 4096,
                             "labels_repeat_exact": bool(np.array_equal(first["labels"], second["labels"])),
                             "confidence_repeat_exact": bool(np.array_equal(first["confidence"], second["confidence"])),
                             "elapsed_seconds": time.perf_counter() - started}
        except Exception as error:
            reports[name] = {"runnable": False, "error": repr(error)}
        print(name, reports[name], flush=True)
    tracker = OfficialTracker(root)
    pointcloud = rng.normal(size=(512, 3)).astype(np.float32)
    selected, boundary_info = tracker.sample_indices(pointcloud, 1)
    transformed, registration_info = tracker.deformation(normalize(pointcloud[selected]), normalize(pointcloud[selected]))
    reports["TrackPlant3D"] = {"runnable": bool(np.isfinite(transformed).all()), "boundary": boundary_info,
                               "registration": registration_info, "official_cost_and_extended_matrix": True}
    write_json(root / "reports/m6_a/official_runtime_smoke.json", reports)
    if not reports["PlantNet"]["runnable"]:
        raise RuntimeError("Frozen primary segmenter is not runnable")


def prepare(root, device, splits=("val", "test")):
    freeze(root)
    initialize_runtime()
    frame = metadata(root)
    frame = frame[frame["split"].isin(splits)].sort_values(["plant_uid", "DAS"])
    model = OfficialSegmenter(root, "PlantNet", device)
    tracker = OfficialTracker(root)
    lock_hash = digest(root / "reports/m6_a/protocol_lock.json")
    previous_samples = {}
    timings = []
    for index, record in enumerate(frame.itertuples(index=False)):
        path = cache_path(root, record.plant_uid, record.DAS)
        seed = 20261012 + record.plant_id * 100 + record.DAS
        if path.exists():
            with np.load(path) as cached:
                if cached["lock_hash"].item() != lock_hash:
                    raise ValueError("Cached frame uses a different frozen protocol")
                previous_samples[record.plant_uid] = cached["tracking_xyz"]
                timings.append(json.loads(cached["timing"].item()))
            continue
        started = time.perf_counter()
        xyz, gt, info = read_cloud(root / record.pointcloud_path)
        gt_local = localize_gt(gt, seed)
        prediction = model.predict(xyz, seed)
        inference_seconds = time.perf_counter() - started
        sample_started = time.perf_counter()
        indices, sample_info = tracker.sample_indices(xyz, seed)
        normalized = normalize(xyz[indices])
        tracking_gt = gt_local[indices]
        tracking_pred = prediction["labels"][indices]
        sampling_seconds = time.perf_counter() - sample_started
        registration_started = time.perf_counter()
        if record.plant_uid in previous_samples:
            deformed, registration_info = tracker.deformation(previous_samples[record.plant_uid], normalized)
        else:
            deformed = normalized.copy()
            registration_info = {"iterations": 0, "objective_difference": 0.0}
        previous_samples[record.plant_uid] = normalized
        registration_seconds = time.perf_counter() - registration_started
        measure_started = time.perf_counter()
        gt_measure = measurements(xyz, gt_local)
        pred_measure = measurements(xyz, prediction["labels"])
        normalized_full = normalize(xyz)
        gt_ids = np.array([0] + sorted(gt_measure), dtype=int)
        pred_ids = np.array([0] + sorted(pred_measure), dtype=int)
        gt_centroids = np.array([normalized_full[gt_local == label].mean(axis=0) if np.any(gt_local == label) else np.zeros(3) for label in gt_ids])
        pred_centroids = np.array([normalized_full[prediction["labels"] == label].mean(axis=0) if np.any(prediction["labels"] == label) else np.zeros(3) for label in pred_ids])
        gt_centroid_present = np.array([np.any(gt_local == label) for label in gt_ids])
        pred_centroid_present = np.array([np.any(prediction["labels"] == label) for label in pred_ids])
        timing = {"plant_uid": record.plant_uid, "DAS": record.DAS, "split": record.split,
                  "raw_points": info["vertex_count"], "inference_seconds": inference_seconds,
                  "sampling_seconds": sampling_seconds, "registration_seconds": registration_seconds,
                  "measurement_seconds": time.perf_counter() - measure_started, **sample_info, **registration_info}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, lock_hash=lock_hash, labels=prediction["labels"], confidence=prediction["confidence"],
                            gt_local=gt_local, sample_indices=prediction["sample_indices"],
                            sample_xyz=xyz[prediction["sample_indices"]], sample_labels=prediction["sample_labels"],
                            tracking_xyz=normalized, deformed_xyz=deformed, tracking_gt=tracking_gt, tracking_pred=tracking_pred,
                            gt_measure_ids=np.array(list(gt_measure)), gt_measure_values=np.array(list(gt_measure.values())).reshape(-1, 2),
                            pred_measure_ids=np.array(list(pred_measure)), pred_measure_values=np.array(list(pred_measure.values())).reshape(-1, 2),
                            gt_centroid_ids=gt_ids[gt_centroid_present], gt_centroids=gt_centroids[gt_centroid_present],
                            pred_centroid_ids=pred_ids[pred_centroid_present], pred_centroids=pred_centroids[pred_centroid_present],
                            timing=json.dumps(timing))
        timings.append(timing)
        pd.DataFrame(timings).to_csv(root / ("reports/m6_a/runtime_" + "_".join(splits) + ".csv"), index=False)
        print(f"FRAME {index + 1}/{len(frame)} {record.plant_uid} DAS={record.DAS} split={record.split} "
              f"points={len(xyz)} pred_leaves={len(pred_measure)} seconds={time.perf_counter() - started:.2f}", flush=True)
    write_json(root / ("reports/m6_a/inference_manifest_" + "_".join(splits) + ".json"), {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "frames": len(frame), "plants": frame.plant_uid.nunique(), "splits": list(splits), "training_steps": 0,
               "lock_sha256": lock_hash,
               "checkpoint_sha256": digest(root / "data/raw/m6_a/official/plantnet/PlantNet/PlantNet_pytorch/models/checkpoints/model_epoch199.pth"),
               "input": "XYZ only; GT current masks confined to B0 and B2_GT; no GT temporal identities in any tracker"})


def summarize_losses(losses):
    result = {}
    for trait in ["length", "area"]:
        for kind in ["static", "change"]:
            for stage in ["GT_MASK_ORACLE_LINK", "PRED_MASK_ORACLE_LINK", "TRACKED"]:
                subset = [row for row in losses if row["trait"] == trait and row["kind"] == kind and row["stage"] == stage]
                prefix = f"{trait}_{kind}_{stage.lower()}"
                result[prefix + "_loss"] = float(np.mean([row["loss"] for row in subset])) if subset else np.nan
                result[prefix + "_coverage"] = float(np.mean([row["available"] for row in subset])) if subset else np.nan
                result[prefix + "_severe"] = float(np.mean([row["severe"] for row in subset])) if subset else np.nan
                conditional = [row["absolute_error"] for row in subset if row["available"]]
                result[prefix + "_mae"] = float(np.mean(conditional)) if conditional else np.nan
                rate = [row["rate_absolute_error"] for row in subset if row["available"] and kind == "change"]
                if kind == "change":
                    result[prefix + "_rate_mae"] = float(np.mean(rate)) if rate else np.nan
                result[prefix + "_eligible"] = len(subset)
    return result


def measurement_losses(plant, split, pipeline, frames, reference):
    losses = []
    for frame_index, current in enumerate(frames):
        for gt_id, association in current["association"].items():
            current_reference = reference[(current["DAS"], gt_id)]
            local_id = association[0]
            gt_estimate = current["gt_measure"].get(current["gt_association"][gt_id][0], (np.nan, np.nan))
            pred_estimate = current["measure"].get(local_id, (np.nan, np.nan)) if local_id >= 0 else (np.nan, np.nan)
            for trait_index, trait in enumerate(["length", "area"]):
                actual = current_reference[trait_index]
                for stage, estimate in [("GT_MASK_ORACLE_LINK", gt_estimate[trait_index]), ("PRED_MASK_ORACLE_LINK", pred_estimate[trait_index]), ("TRACKED", pred_estimate[trait_index])]:
                    available = bool(np.isfinite(estimate))
                    relative = abs(estimate - actual) / max(actual, 1.) if available else np.inf
                    losses.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "DAS": current["DAS"],
                                   "gt_id": gt_id, "kind": "static", "trait": trait, "stage": stage,
                                   "reference": actual, "estimate": estimate, "available": available,
                                   "absolute_error": abs(estimate - actual) if available else np.nan,
                                   "relative_error": relative if available else np.nan,
                                   "loss": penalized_error(estimate, actual), "severe": bool(relative > .5),
                                   "status": "measured" if available else "instance_missed", "confidence": current["confidence"].get(local_id, np.nan),
                                   "rate_absolute_error": np.nan})
            if frame_index == 0:
                continue
            previous = frames[frame_index - 1]
            if gt_id not in previous["association"]:
                continue
            previous_reference = reference[(previous["DAS"], gt_id)]
            previous_gt = previous["gt_measure"][previous["gt_association"][gt_id][0]]
            previous_local = previous["association"][gt_id][0]
            previous_pred = previous["measure"].get(previous_local, (np.nan, np.nan))
            current_track = current["tracks"].get(local_id, -1)
            tracked_previous_local = next((old_local for old_local, old_track in previous["tracks"].items() if old_local > 0 and old_track == current_track), -1) if current_track >= 0 else -1
            tracked_previous = previous["measure"].get(tracked_previous_local, (np.nan, np.nan))
            if local_id < 0 or previous_local < 0:
                status = "instance_missed"
            elif current_track < 0 or tracked_previous_local < 0:
                status = "identity_lost"
            elif tracked_previous_local != previous_local:
                status = "wrong_match"
            else:
                status = "correct_link"
            gap = current["DAS"] - previous["DAS"]
            for trait_index, trait in enumerate(["length", "area"]):
                actual = current_reference[trait_index] - previous_reference[trait_index]
                estimates = [("GT_MASK_ORACLE_LINK", gt_estimate[trait_index] - previous_gt[trait_index]),
                             ("PRED_MASK_ORACLE_LINK", pred_estimate[trait_index] - previous_pred[trait_index]),
                             ("TRACKED", pred_estimate[trait_index] - tracked_previous[trait_index])]
                for stage, estimate in estimates:
                    available = bool(np.isfinite(estimate))
                    relative = abs(estimate - actual) / max(previous_reference[trait_index], 1.) if available else np.inf
                    losses.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "DAS": current["DAS"],
                                   "gt_id": gt_id, "kind": "change", "trait": trait, "stage": stage,
                                   "reference": actual, "estimate": estimate, "available": available,
                                   "absolute_error": abs(estimate - actual) if available else np.nan,
                                   "relative_error": relative if available else np.nan,
                                   "loss": penalized_error(estimate, actual, previous_reference[trait_index]), "severe": bool(relative > .5),
                                   "status": status if stage == "TRACKED" else ("measured" if available else "instance_missed"),
                                   "confidence": current["confidence"].get(local_id, np.nan),
                                   "rate_absolute_error": abs(estimate - actual) / gap if available else np.nan})
    return losses


def evaluate_plant(root, plant, records, tracker, reference):
    pipeline_frames = {name: [] for name in PIPELINES}
    observations = {name: [] for name in PIPELINES}
    births = {name: [] for name in PIPELINES}
    next_track = {name: 0 for name in PIPELINES}
    frame_scores = {name: [] for name in PIPELINES}
    predicted_organs = []
    for frame_index, record in enumerate(records.itertuples(index=False)):
        with np.load(cache_path(root, plant, record.DAS)) as cached:
            cache = {key: cached[key] for key in cached.files}
        xyz, gt, unused_info = read_cloud(root / record.pointcloud_path)
        gt_frame_score, gt_association = frame_metrics(gt, cache["gt_local"])
        pred_frame_score, pred_association = frame_metrics(gt, cache["labels"])
        gt_measure = {int(key): value for key, value in zip(cache["gt_measure_ids"], cache["gt_measure_values"])}
        pred_measure = {int(key): value for key, value in zip(cache["pred_measure_ids"], cache["pred_measure_values"])}
        for name in PIPELINES:
            use_gt = name in ["B0", "B2_GT"]
            labels = cache["gt_local"] if use_gt else cache["labels"]
            measure = gt_measure if use_gt else pred_measure
            source_association = gt_association if use_gt else pred_association
            association = {int(gt_id): source_association.get(int(gt_id), (-1, 0.)) for gt_id in np.unique(gt) if gt_id > 0}
            frame_score = dict(gt_frame_score if use_gt else pred_frame_score)
            sample_labels = cache["tracking_gt"] if use_gt else cache["tracking_pred"]
            simple = name in ["B2", "B2_GT"]
            current_labels = cache["gt_centroid_ids"] if use_gt and simple else cache["pred_centroid_ids"] if simple else sample_labels
            current_xyz = cache["gt_centroids"] if use_gt and simple else cache["pred_centroids"] if simple else cache["tracking_xyz"]
            if frame_index == 0:
                sampled_ids = np.unique(current_labels)
                tracks = {int(label): offset for offset, label in enumerate(sampled_ids)}
                margins = {int(label): 1.0 for label in sampled_ids}
                born = set(tracks)
                next_track[name] = len(sampled_ids)
            else:
                previous = pipeline_frames[name][-1]
                previous_tracks = np.array([previous["tracks"][int(label)] for label in previous["input_labels"]])
                if simple:
                    tracks, margins, born, next_track[name] = simple_assign(previous["input_xyz"], previous_tracks, current_xyz, current_labels, next_track[name])
                else:
                    tracks, margins, born, next_track[name] = tracker.assign(previous["input_xyz"], previous_tracks, cache["deformed_xyz"], current_labels, next_track[name])
            for label in np.unique(labels):
                tracks.setdefault(int(label), -1)
            confidence = {}
            for local_id in measure:
                semantic_confidence = 1.0 if use_gt else float(cache["confidence"][labels == local_id].mean())
                confidence[local_id] = semantic_confidence * margins.get(local_id, 0.)
                inverse_gt = next((gt_id for gt_id, value in association.items() if value[0] == local_id), -1)
                predicted_organs.append({"plant_uid": plant, "split": record.split, "pipeline": name,
                                         "frame_index": frame_index, "DAS": record.DAS, "local_id": local_id,
                                         "track_id": tracks[local_id], "gt_id_evaluation_only": inverse_gt,
                                         "confidence": confidence[local_id], "semantic_confidence": semantic_confidence,
                                         "new_track": local_id in born, "length_cm": measure[local_id][0], "area_cm2": measure[local_id][1],
                                         "wrong_instance": inverse_gt < 0, "sampled_support": int(np.sum(sample_labels == local_id))})
                if local_id in born:
                    births[name].append({"frame_index": frame_index, "local_id": local_id, "track_id": tracks[local_id]})
            for gt_id, value in association.items():
                local_id, iou = value
                observations[name].append({"plant_uid": plant, "split": record.split, "pipeline": name, "frame_index": frame_index,
                                           "DAS": record.DAS, "gt_id": gt_id, "local_id": local_id,
                                           "track_id": tracks.get(local_id, -1), "iou": iou,
                                           "confidence": confidence.get(local_id, np.nan),
                                           "state": "instance_missed" if local_id < 0 else "sampling_or_identity_lost" if tracks.get(local_id, -1) < 0 else "detected"})
            frame_score.update({"plant_uid": plant, "split": record.split, "pipeline": name, "DAS": record.DAS,
                                "sampled_missing_leaf_instances": sum(tracks[local_id] < 0 for local_id in measure)})
            frame_scores[name].append(frame_score)
            pipeline_frames[name].append({"DAS": record.DAS, "association": association, "gt_association": gt_association,
                                          "measure": measure, "gt_measure": gt_measure, "tracks": tracks, "confidence": confidence,
                                          "input_xyz": current_xyz, "input_labels": current_labels})
    results = []
    all_losses = []
    for name in PIPELINES:
        identity, alignment = observation_identity(observations[name], sum(item["pred_count"] for item in frame_scores[name]))
        event = event_metrics(observations[name], births[name])
        losses = measurement_losses(plant, records.iloc[0].split, name, pipeline_frames[name], reference)
        all_losses.extend(losses)
        result = {"plant_uid": plant, "split": records.iloc[0].split, "pipeline": name, "scans": len(records),
                  "pq": np.mean([item["pq"] for item in frame_scores[name]]), "sbd": np.mean([item["sbd"] for item in frame_scores[name]]),
                  "detection_recall": sum(item["tp"] for item in frame_scores[name]) / max(sum(item["gt_count"] for item in frame_scores[name]), 1),
                  "merge_instances": sum(item["merge_instances"] for item in frame_scores[name]),
                  "split_gt_instances": sum(item["split_gt_instances"] for item in frame_scores[name]),
                  "sampling_drop_instances": sum(item["sampled_missing_leaf_instances"] for item in frame_scores[name]),
                  **identity, **event, **summarize_losses(losses)}
        for row in observations[name]:
            row["evaluation_identity_correct"] = row["track_id"] >= 0 and alignment.get(row["gt_id"]) == row["track_id"]
        results.append(result)
    return results, [row for values in observations.values() for row in values], [row for values in frame_scores.values() for row in values], all_losses, predicted_organs


def reliable(row, protocol):
    rule = protocol["acceptance"]
    return bool(row["pq"] >= rule["leaf_pq_min"] and row["idf1"] >= rule["idf1_min"] and
                row["new_event_f1"] >= rule["new_event_f1_min"] and
                row["length_static_tracked_coverage"] >= rule["measurement_coverage_min"] and
                row["area_static_tracked_coverage"] >= rule["measurement_coverage_min"] and
                row["length_change_tracked_coverage"] >= rule["measurement_coverage_min"] and
                row["area_change_tracked_coverage"] >= rule["measurement_coverage_min"] and
                row["length_static_tracked_loss"] <= rule["length_penalized_relative_error_max"] and
                row["area_static_tracked_loss"] <= rule["area_penalized_relative_error_max"] and
                row["length_change_tracked_severe"] <= rule["severe_change_failure_rate_max"] and
                row["area_change_tracked_severe"] <= rule["severe_change_failure_rate_max"])


def finalize_tables(root):
    destination = root / "reports/m6_a"
    results = pd.read_csv(destination / "per_plant.csv")
    protocol = json.loads((root / "configs/m6_a/protocol.json").read_text())
    results["reliable"] = results.apply(lambda row: reliable(row, protocol), axis=1)
    results.to_csv(destination / "per_plant.csv", index=False)
    numeric = results.select_dtypes(include=["number", "bool"]).columns
    summary = results.groupby(["split", "pipeline"])[numeric].mean().reset_index()
    summary["independent_plants"] = summary.apply(lambda row: len(results[(results.split == row.split) & (results.pipeline == row.pipeline)]), axis=1)
    summary.to_csv(destination / "baseline_metrics.csv", index=False)
    comparisons = []
    for split in results.split.unique():
        selected = results[results.split == split]
        for left, right in [("B1", "B0"), ("B1", "B2"), ("B0", "B2_GT")]:
            paired = selected[selected.pipeline == left].set_index("plant_uid").join(selected[selected.pipeline == right].set_index("plant_uid"), lsuffix="_left", rsuffix="_right")
            for metric in ["pq", "idf1", "link_recall", "new_event_f1", "length_change_tracked_loss", "area_change_tracked_loss"]:
                comparisons.append({"split": split, "comparison": left + " minus " + right, "metric": metric,
                                    **paired_interval(paired[metric + "_left"] - paired[metric + "_right"])})
        for pipeline in PIPELINES:
            current = selected[selected.pipeline == pipeline]
            for trait in ["length", "area"]:
                comparisons.append({"split": split, "comparison": pipeline + " tracking minus evaluation-only oracle links",
                                    "metric": trait + "_change_loss",
                                    **paired_interval(current[trait + "_change_tracked_loss"] - current[trait + "_change_pred_mask_oracle_link_loss"])})
    pd.DataFrame(comparisons).to_csv(destination / "paired_plant_intervals.csv", index=False)
    decomposition = []
    for row in results.itertuples(index=False):
        for trait in ["length", "area"]:
            baseline = getattr(row, trait + "_change_gt_mask_oracle_link_loss")
            segmentation = getattr(row, trait + "_change_pred_mask_oracle_link_loss")
            tracked = getattr(row, trait + "_change_tracked_loss")
            decomposition.append({"plant_uid": row.plant_uid, "split": row.split, "pipeline": row.pipeline, "trait": trait,
                                  "measurement_reference_discrepancy": baseline, "segmentation_increment": segmentation - baseline,
                                  "tracking_increment": tracked - segmentation, "total": tracked,
                                  "additivity_residual": tracked - baseline - (segmentation - baseline) - (tracked - segmentation)})
    pd.DataFrame(decomposition).to_csv(destination / "error_decomposition.csv", index=False)
    primary = summary[summary.split == "test"].set_index("pipeline")
    delta = float(primary.loc["B1", "idf1"] - primary.loc["B0", "idf1"])
    if len(results[results.split == "test"]) != 48 or not np.isfinite(primary.idf1).all():
        recommendation = "PIVOT"
    elif primary.loc["B1", "pq"] < .7 and delta <= -.15:
        recommendation = "INSTANCE-FIRST"
    elif primary.loc["B0", "idf1"] < .8 and primary.loc["B0", "length_change_tracked_loss"] - primary.loc["B0", "length_change_pred_mask_oracle_link_loss"] >= .05:
        recommendation = "TRACKING-FIRST"
    else:
        recommendation = "MEASUREMENT-FIRST"
    passed = reliable(primary.loc["B1"], protocol) and primary.loc["B1", "reliable"] >= .8
    write_json(destination / "decision.json", {"recommendation": recommendation, "baseline_acceptance_passed": bool(passed),
               "test_independent_plants": 12, "primary_metric": "plant-macro observation IDF1",
               "B1_minus_B0_idf1": delta, "protocol_sha256": digest(root / "configs/m6_a/protocol.json"),
               "no_training": True, "no_cross_species_validation": True, "stop_after_delivery": True})


def evaluate(root, splits=("val", "test"), wait_for_cache=False):
    freeze(root)
    frame = metadata(root)
    frame = frame[frame.split.isin(splits)]
    leaf = pd.read_csv(root / "data/processed/sugar4d/leaf_metadata.csv")
    tracker = OfficialTracker(root)
    tables = {"per_plant": [], "observations": [], "frame_metrics": [], "measurement_errors": [], "predicted_organs": []}
    for plant, records in frame.groupby("plant_uid", sort=True):
        records = records.sort_values("DAS")
        while not all(cache_path(root, plant, record.DAS).exists() for record in records.itertuples(index=False)):
            if not wait_for_cache:
                raise FileNotFoundError("Incomplete fixed cohort: " + plant)
            time.sleep(5)
        selected = leaf[leaf.plant_uid == plant]
        reference = {(int(row["DAS"]), int(row["leaf_instance_id"]) + 1): (float(row["total length [cm]"]), float(row["leaf area [cm2]"])) for unused_index, row in selected.iterrows()}
        result = evaluate_plant(root, plant, records, tracker, reference)
        plant_destination = root / f"outputs/m6_a/evaluation/{plant}"
        plant_destination.mkdir(parents=True, exist_ok=True)
        for key, values in zip(tables, result):
            tables[key].extend(values)
            pd.DataFrame(values).to_csv(plant_destination / (key + ".csv"), index=False)
        for key, values in tables.items():
            pd.DataFrame(values).to_csv(root / f"reports/m6_a/{key}.csv", index=False)
        print("EVALUATED", plant, records.iloc[0].split, "B0/B1/B2/B2_GT", flush=True)
    finalize_tables(root)
