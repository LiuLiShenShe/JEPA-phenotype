from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from birth_detection.common import digest, initialize, protocol, strict_matches, window_indices, write_json
from m6a.experiment import metadata
from m6a.metrics import geometry_measure
from m6b.data import read_frame
from m6b.evaluation import cluster_prediction, prediction_path, sampled_forward


def frame_path(root, plant, das):
    return root / f"outputs/m7/frames/{plant}_{das}.npz"


def resample(points, count, generator):
    if not len(points):
        return np.zeros((count, 3), dtype=np.float32)
    indices = generator.choice(len(points), count, replace=len(points) < count)
    return points[indices].astype(np.float32)


def shape_features(points, total_points, core_tree=None):
    if not len(points):
        return np.zeros(25, dtype=np.float32)
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(points), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    orientation = eigenvectors[:, -1]
    if orientation[np.abs(orientation).argmax()] < 0:
        orientation = -orientation
    extent = np.ptp(points, axis=0)
    length, area = geometry_measure(points)
    radial = np.linalg.norm(centered, axis=1)
    core_distance = float(np.quantile(core_tree.query(points[::max(1, len(points) // 512)])[0], .1)) if core_tree is not None else .25
    return np.asarray([*center, np.log1p(len(points)), len(points) / max(total_points, 1),
                       *extent, np.prod(extent), *eigenvalues, *orientation,
                       eigenvalues[0] / max(eigenvalues.sum(), 1e-10),
                       np.nan_to_num(length), np.nan_to_num(area), core_distance,
                       *np.quantile(radial, [0, .25, .5, .75, 1]), float(radial.std())], dtype=np.float32)


def voxelize(xyz, size):
    keys, indices, inverse, counts = np.unique(np.floor(xyz / size).astype(np.int32), axis=0,
                                               return_index=True, return_inverse=True, return_counts=True)
    return xyz[indices], inverse, counts


def change_regions(xyz, previous_xyz, settings):
    points, inverse, counts = voxelize(xyz, settings["voxel_m"])
    current_tree = cKDTree(points)
    if previous_xyz is None:
        return [], np.zeros(len(xyz), dtype=bool), points
    previous_tree = cKDTree(previous_xyz)
    distances = previous_tree.query(points, workers=4)[0]
    current_density = current_tree.query_ball_point(points, settings["density_radius_m"], return_length=True, workers=4)
    previous_density = previous_tree.query_ball_point(points, settings["density_radius_m"], return_length=True, workers=4)
    novel = (distances > settings["novel_distance_m"]) | ((current_density + 1) / (previous_density + 1) > settings["density_ratio"])
    selected = np.flatnonzero(novel)
    regions = []
    if len(selected) >= settings["dbscan_min_samples"]:
        clusters = DBSCAN(eps=settings["dbscan_eps_m"], min_samples=settings["dbscan_min_samples"], n_jobs=4).fit_predict(points[selected])
        voxel_labels = np.full(len(points), -1, dtype=np.int32)
        voxel_labels[selected] = clusters
        full_labels = voxel_labels[inverse]
        for cluster in np.unique(clusters[clusters >= 0]):
            indices = np.flatnonzero(full_labels == cluster)
            if len(indices) >= settings["minimum_region_points"]:
                regions.append(indices)
    return regions, novel[inverse], points


def frozen_prediction(root, record, data, model, device):
    if record.split != "train":
        path = prediction_path(root, "S2", 17, record.plant_uid, record.DAS)
    else:
        path = root / f"outputs/m7/frozen_train_predictions/{record.plant_uid}_{record.DAS}.npz"
    if path.exists():
        with np.load(path) as values:
            return {name: values[name].copy() for name in ["labels", "semantic", "confidence"]}
    if record.split != "train":
        raise ValueError("Missing old predictions; do not reevaluate old methods")
    sampled = sampled_forward(model, data["xyz"], 20261012 + record.plant_id * 100 + record.DAS, device)
    bandwidth = json.loads((root / "outputs/m6_b/S2/seed_17/postprocessing_selection.json").read_text())["bandwidth"]
    prediction = cluster_prediction(data["xyz"], sampled, bandwidth)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **prediction, **{key: sampled[key] for key in ["sample_indices", "semantic_probability", "embedding", "center_m", "scale_m"]})
    return prediction


def prepare(root, device="cuda:1"):
    settings = protocol(root)
    if (root / "reports/m7/dataset_manifest.json").exists():
        return
    initialize(20261013)
    from m6b.models import official_model

    model, unused_loading = official_model(root, "PlantNet", False, device)
    checkpoint = torch.load(root / settings["segmentation"]["frozen_checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    records = metadata(root).sort_values(["plant_uid", "DAS"])
    leaf_reference = pd.read_csv(root / "data/processed/sugar4d/leaf_metadata.csv")
    reference = {(row["plant_uid"], int(row["DAS"]), int(row["leaf_instance_id"]) + 1):
                 (float(row["total length [cm]"]), float(row["leaf area [cm2]"]))
                 for unused_index, row in leaf_reference.iterrows()}
    frame_rows, candidate_rows, event_rows, organ_rows, forecast_rows = [], [], [], [], []
    features, single_features, patches, time_features, candidate_targets = [], [], [], [], []
    organ_points, organ_descriptors, organ_gt_labels = [], [], []
    for plant, scans in records.groupby("plant_uid", sort=True):
        previous_voxels = None
        history_clouds, history_days, seen = [], [], set()
        initial_anchor = None
        historical = None
        old_path = root / f"outputs/m6_b/S2/seed_17/tracking/{plant}.json"
        if old_path.exists():
            historical = json.loads(old_path.read_text())
        historical_observations = {(int(row["DAS"]), int(row["gt_id"])): row for row in historical["observations"]} if historical else {}
        plant_events, plant_forecasts = [], []
        for frame_index, record in enumerate(scans.itertuples(index=False)):
            destination = frame_path(root, plant, record.DAS)
            if destination.exists():
                raise ValueError("Preparation is atomic at dataset level; use completed dataset or remove only incomplete M7 outputs")
            data = read_frame(root / record.pointcloud_path)
            xyz = data["xyz"]
            gt = np.where(data["semantic"] == 2, data["original_instance_evaluation_only"] + 1, 0).astype(np.int32)
            prediction = frozen_prediction(root, record, data, model, device)
            labels = prediction["labels"].astype(np.int32)
            core_xyz = xyz[prediction["semantic"] == 1]
            if not len(core_xyz):
                core_xyz = xyz[xyz[:, 2] <= np.quantile(xyz[:, 2], .1)]
            predicted_core = cKDTree(core_xyz)
            if initial_anchor is None:
                initial_anchor = xyz[xyz[:, 2] <= np.quantile(xyz[:, 2], .1)].mean(axis=0)
            regions, novel, voxels = change_regions(xyz, previous_voxels, settings["candidates"])
            previous_voxels = voxels
            generator = np.random.default_rng(20261013 + record.plant_id * 100 + record.DAS)
            sampled_cloud = resample(xyz, 16384, generator)
            history_clouds.append(sampled_cloud)
            history_days.append(record.DAS)
            predicted_ids = np.unique(labels[labels > 0])
            masks = [np.flatnonzero(labels == label) for label in predicted_ids]
            sources = ["segmentation"] * len(masks)
            mask_sizes = {int(label): len(mask) for label, mask in zip(predicted_ids, masks)}
            for region in regions:
                overlap_ids, overlap_counts = np.unique(labels[region], return_counts=True)
                repeated = any(label > 0 and count / max(len(region) + mask_sizes[int(label)] - count, 1) > .8
                               for label, count in zip(overlap_ids, overlap_counts))
                if not repeated:
                    masks.append(region)
                    sources.append("change")
            matches, scores, gt_ids = strict_matches(gt, masks)
            base_masks = masks[:len(predicted_ids)]
            base_matches, unused_scores, unused_ids = strict_matches(gt, base_masks)
            by_pred = {column: label for label, (column, unused_iou) in base_matches.items()}
            current_new = set(map(int, gt_ids)) - seen
            evaluation_core = cKDTree(xyz[data["semantic"] == 1]) if np.any(data["semantic"] == 1) else None
            gt_measures = []
            for gt_id in gt_ids:
                indices = np.flatnonzero(gt == gt_id)
                gt_measures.append([int(gt_id), *geometry_measure(xyz[indices])])
                if int(gt_id) in current_new:
                    distance = float(np.quantile(evaluation_core.query(xyz[indices], workers=4)[0], .1)) if evaluation_core is not None else np.nan
                    event = {"plant_id": plant, "leaf_instance_id": int(gt_id) - 1, "genotype": record.genotype,
                             "split": record.split, "birth_time_visible": int(record.DAS), "birth_time_segmentable": np.nan,
                             "birth_time_trackable": np.nan, "previous_observation_available": frame_index > 0,
                             "neighbor_core_distance": distance, "initial_leaf_area": reference[(plant, int(record.DAS), int(gt_id))][1],
                             "initial_point_count": len(indices), "initial_point_fraction": len(indices) / len(xyz),
                             "small": len(indices) / len(xyz) <= .04588683197817431,
                             "core_adjacent": bool(np.isfinite(distance) and distance <= .005),
                             "left_censored": frame_index == 0, "visible_interval_left_DAS": history_days[-2] if frame_index else np.nan,
                             "visible_interval_right_DAS": int(record.DAS), "biological_birth_observed": False,
                             "segmentability_source": "frozen_S2_train_in_sample" if record.split == "train" else "imported_M6B_frozen_S2",
                             "trackability_source": "not_available_for_train_no_baseline_reexecution" if record.split == "train" else "imported_M6B_fixed_TrackPlant3D"}
                    plant_events.append(event)
            for event in plant_events:
                gt_id = event["leaf_instance_id"] + 1
                if np.isnan(event["birth_time_segmentable"]) and gt_id in base_matches:
                    event["birth_time_segmentable"] = int(record.DAS)
            seen.update(map(int, gt_ids))
            offsets = [0]
            all_indices = []
            candidate_global = []
            for candidate_index, (indices, source) in enumerate(zip(masks, sources)):
                points = xyz[indices]
                descriptor = shape_features(points, len(xyz), predicted_core)
                center = points.mean(axis=0)
                descriptor[:3] -= initial_anchor
                current = np.r_[descriptor, float(source == "change"), float(novel[indices].mean()),
                                float(prediction["confidence"][indices].mean()), float(np.mean(prediction["semantic"][indices] == 2))].astype(np.float32)
                patch_sequence, statistics, timing = [], [], []
                for history_index in window_indices(frame_index):
                    cloud = history_clouds[history_index]
                    nearby = cloud[np.linalg.norm(cloud - center, axis=1) <= settings["candidates"]["patch_radius_m"]]
                    patch_sequence.append(resample(nearby - center, settings["candidates"]["patch_points"], generator) / settings["candidates"]["patch_scale_m"])
                    local = shape_features(nearby, len(cloud))
                    local[:3] -= center
                    statistics.append(local)
                    timing.append([(record.DAS - history_days[history_index]) / 10, float(history_index != 0 or frame_index == 0)])
                temporal = np.r_[current, *statistics, np.asarray(timing).reshape(-1)].astype(np.float32)
                single = np.r_[current[:25], current[27:], statistics[-1], timing[-1]].astype(np.float32)
                matched_gt = next((label for label, (column, unused_iou) in matches.items() if column == candidate_index), -1)
                positive = frame_index > 0 and matched_gt in current_new
                global_index = len(candidate_rows)
                candidate_global.append(global_index)
                best_row = int(scores[:, candidate_index].argmax()) if len(gt_ids) else -1
                candidate_rows.append({"candidate_index": global_index, "plant_id": plant, "split": record.split, "DAS": int(record.DAS),
                                       "frame_index": frame_index, "frame_candidate_index": candidate_index, "source": source,
                                       "point_count": len(indices), "target_new_visible": bool(positive),
                                       "GT_best_id_evaluation_only": int(gt_ids[best_row]) if best_row >= 0 else -1,
                                       "GT_best_iou_evaluation_only": float(scores[best_row, candidate_index]) if best_row >= 0 else 0,
                                       "GT_matched_id_evaluation_only": int(matched_gt), "change_fraction": float(novel[indices].mean())})
                features.append(temporal)
                single_features.append(single)
                patches.append(patch_sequence)
                time_features.append(timing)
                candidate_targets.append(positive)
                all_indices.append(indices.astype(np.int32))
                offsets.append(offsets[-1] + len(indices))
            organ_global = []
            predicted_measure = []
            for local_index, (predicted_id, indices) in enumerate(zip(predicted_ids, base_masks)):
                points = xyz[indices]
                descriptor = shape_features(points, len(xyz), predicted_core)
                global_index = len(organ_rows)
                organ_global.append(global_index)
                organ_rows.append({"organ_index": global_index, "plant_id": plant, "split": record.split, "DAS": int(record.DAS),
                                   "frame_index": frame_index, "local_id": int(predicted_id), "points": len(indices),
                                   "GT_matched_id_supervision_or_evaluation_only": int(by_pred.get(local_index, -1))})
                organ_points.append(resample(points - points.mean(axis=0), settings["candidates"]["patch_points"], generator) / .12)
                organ_descriptors.append(descriptor)
                organ_gt_labels.append(by_pred.get(local_index, -1))
                predicted_measure.append([int(predicted_id), *geometry_measure(points)])
            forecast = {"plant_id": plant, "split": record.split, "DAS": int(record.DAS), "frame_index": frame_index,
                        "points": len(xyz), "voxel_count": len(voxels), "new_fraction": float(novel.mean()),
                        "predicted_leaf_count": len(predicted_ids), "height_m": float(np.ptp(xyz[:, 2])),
                        "width_m": float(np.ptp(xyz[:, :2], axis=0).max()), "core_predicted_points": len(core_xyz),
                        "new_visible_count_label_only": len(current_new) if frame_index > 0 else 0}
            plant_forecasts.append(forecast)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination, gt_evaluation_only=gt, base_labels=labels,
                                indices=np.concatenate(all_indices) if all_indices else np.empty(0, dtype=np.int32), offsets=np.asarray(offsets),
                                candidate_global_indices=np.asarray(candidate_global), organ_global_indices=np.asarray(organ_global),
                                base_local_ids=predicted_ids, gt_measures=np.asarray(gt_measures), predicted_measures=np.asarray(predicted_measure),
                                sample_xyz=sampled_cloud, gt_new_ids_evaluation_only=np.asarray(sorted(current_new) if frame_index else [], dtype=np.int32))
            frame_rows.append({"plant_id": plant, "split": record.split, "DAS": int(record.DAS), "frame_index": frame_index,
                               "points": len(xyz), "GT_leaf_count_evaluation_only": len(gt_ids), "candidate_count": len(masks),
                               "positive_candidates": sum(candidate_targets[index] for index in candidate_global),
                               "current_new_events": len(current_new) if frame_index else 0,
                               "new_candidates_recoverable": sum(label in matches for label in current_new) if frame_index else 0,
                               "pointcloud_path": record.pointcloud_path})
            print("PREPARE", plant, record.DAS, len(xyz), len(masks), frame_rows[-1]["new_candidates_recoverable"], flush=True)
        for event in plant_events:
            gt_id = event["leaf_instance_id"] + 1
            days = history_days
            if historical:
                for position in range(len(days) - 2):
                    sequence = [historical_observations.get((int(day), gt_id)) for day in days[position:position + 3]]
                    if all(row is not None and row["local_id"] >= 0 and row["track_id"] >= 0 for row in sequence) and len({row["track_id"] for row in sequence}) == 1:
                        event["birth_time_trackable"] = int(days[position])
                        break
            event["segmentable_right_censored"] = bool(np.isnan(event["birth_time_segmentable"]))
            event["trackable_right_censored"] = bool(np.isnan(event["birth_time_trackable"]))
            event["trackability_evaluable"] = historical is not None
        for position, row in enumerate(plant_forecasts[:-1]):
            row["forecast_target_DAS"] = plant_forecasts[position + 1]["DAS"]
            row["forecast_target_new_visible"] = plant_forecasts[position + 1]["new_visible_count_label_only"] > 0
            forecast_rows.append(row)
        event_rows.extend(plant_events)
    pd.DataFrame(event_rows).to_csv(root / "reports/m7/organ_birth_events.csv", index=False)
    pd.DataFrame(frame_rows).to_csv(root / "reports/m7/frame_audit.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(root / "outputs/m7/candidates.csv", index=False)
    pd.DataFrame(organ_rows).to_csv(root / "outputs/m7/organs.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(root / "outputs/m7/forecast_rows.csv", index=False)
    np.savez_compressed(root / "outputs/m7/birth_dataset.npz", temporal_features=np.asarray(features), single_features=np.asarray(single_features),
                        patches=np.asarray(patches), timing=np.asarray(time_features, dtype=np.float32), targets=np.asarray(candidate_targets, dtype=np.float32))
    np.savez_compressed(root / "outputs/m7/organ_dataset.npz", points=np.asarray(organ_points), descriptors=np.asarray(organ_descriptors),
                        GT_labels_supervision_or_evaluation_only=np.asarray(organ_gt_labels))
    build_pairs(root)
    events = pd.DataFrame(event_rows)
    events["segment_delay_days"] = events.birth_time_segmentable - events.birth_time_visible
    events["track_delay_days"] = events.birth_time_trackable - events.birth_time_visible
    events.groupby(["split", "plant_id", "genotype"]).agg(new_events=("previous_observation_available", "sum"),
        total_leaves=("leaf_instance_id", "size"), small_fraction=("small", "mean"), core_fraction=("core_adjacent", "mean"),
        segment_delay_days=("segment_delay_days", "mean"), track_delay_days=("track_delay_days", "mean")).reset_index().to_csv(root / "reports/m7/birth_statistics.csv", index=False)
    write_json(root / "reports/m7/dataset_manifest.json", {"frames": len(frame_rows), "plants": 48, "splits": records.groupby("split").plant_uid.nunique().to_dict(),
               "events": len(event_rows), "non_left_censored_events": int(events.previous_observation_available.sum()),
               "candidates": len(candidate_rows), "Train_predictions_in_sample": True, "no_GT_features": True,
               "future_inputs": False, "train_trackable_annotation_unavailable": True,
               "segmentation_checkpoint_sha256": digest(root / settings["segmentation"]["frozen_checkpoint"]),
               "feature_dimensions": {"temporal": len(features[0]), "single": len(single_features[0])}})


def pair_features(previous, current, gap_days):
    displacement = current[:3] - previous[:3]
    ratios = np.log((np.abs(current[[4, 16, 17]]) + 1e-4) / (np.abs(previous[[4, 16, 17]]) + 1e-4))
    shape_difference = np.abs(current[3:] - previous[3:])
    return np.r_[displacement / .25, np.linalg.norm(displacement) / .25, gap_days / 10,
                 ratios, abs(float(np.dot(current[12:15], previous[12:15]))), shape_difference,
                 previous[3:], current[3:]].astype(np.float32)


def build_pairs(root):
    organs = pd.read_csv(root / "outputs/m7/organs.csv")
    with np.load(root / "outputs/m7/organ_dataset.npz") as data:
        descriptors = data["descriptors"]
    features, targets, previous_indices, current_indices, rows = [], [], [], [], []
    for plant, group in organs.groupby("plant_id"):
        for frame_index, current in group.groupby("frame_index"):
            for gap in [1, 2]:
                previous = group.query("frame_index == @frame_index - @gap")
                for old in previous.itertuples(index=False):
                    for new in current.itertuples(index=False):
                        old_gt = old.GT_matched_id_supervision_or_evaluation_only
                        new_gt = new.GT_matched_id_supervision_or_evaluation_only
                        positive = old_gt > 0 and old_gt == new_gt
                        features.append(pair_features(descriptors[old.organ_index], descriptors[new.organ_index], new.DAS - old.DAS))
                        targets.append(positive)
                        previous_indices.append(old.organ_index)
                        current_indices.append(new.organ_index)
                        rows.append({"plant_id": plant, "split": new.split, "DAS": new.DAS, "gap_scans": gap, "positive": positive})
    np.savez_compressed(root / "outputs/m7/pair_dataset.npz", features=np.asarray(features), targets=np.asarray(targets, dtype=np.float32),
                        previous_indices=np.asarray(previous_indices), current_indices=np.asarray(current_indices))
    pd.DataFrame(rows).to_csv(root / "outputs/m7/pairs.csv", index=False)
