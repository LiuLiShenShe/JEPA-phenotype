from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from birth_detection.common import initialize, protocol, strict_matches, write_json
from birth_detection.dataset import frame_path, pair_features, resample, shape_features
from birth_detection.evaluation import masks_from_frame, select_indices
from m6a.metrics import event_metrics, geometry_measure, observation_identity
from m6b.data import read_frame
from temporal_encoder.models import PairMatcher


def assign_tracks(probabilities, previous_tracks, next_track, threshold, forced_births=None):
    probabilities = np.asarray(probabilities)
    previous_tracks = np.asarray(previous_tracks, dtype=int)
    current_count = probabilities.shape[0]
    forced_births = set() if forced_births is None else set(forced_births)
    costs = np.full((current_count, len(previous_tracks) + current_count), 1e6)
    if len(previous_tracks):
        costs[:, :len(previous_tracks)] = -probabilities
    for current in range(current_count):
        costs[current, len(previous_tracks) + current] = -threshold
        if current in forced_births:
            costs[current, :len(previous_tracks)] = 1e6
    tracks, born = {}, set()
    if current_count:
        rows, columns = linear_sum_assignment(costs)
        for row, column in zip(rows, columns):
            if column < len(previous_tracks) and probabilities[row, column] > threshold:
                tracks[int(row)] = int(previous_tracks[column])
            else:
                tracks[int(row)] = next_track
                born.add(int(row))
                next_track += 1
    return tracks, born, next_track


def load_matcher(root, method, seed):
    path = root / f"outputs/m7/matching/{method}/seed_{seed}/model.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = PairMatcher(len(checkpoint["mean"]), method == "C2")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint["mean"].numpy(), checkpoint["scale"].numpy()


def score_pairs(model, mean, scale, current, previous, gap_days):
    if not current or not previous:
        return np.empty((len(current), len(previous)))
    features, old_points, new_points = [], [], []
    for now in current:
        for old in previous:
            features.append(pair_features(old["descriptor"], now["descriptor"], gap_days(old)))
            if model.metric:
                old_points.append(old["points"])
                new_points.append(now["points"])
    normalized = np.clip((np.asarray(features) - mean) / scale, -10, 10).astype(np.float32)
    with torch.inference_mode():
        logits, unused_distance = model(torch.from_numpy(normalized),
                                       torch.from_numpy(np.asarray(old_points)) if model.metric else None,
                                       torch.from_numpy(np.asarray(new_points)) if model.metric else None)
    return torch.sigmoid(logits).numpy().reshape(len(current), len(previous))


def prepared_organs(root, frame, plant, das, seed, organ_data, candidate_values=None, birth_threshold=None):
    labels = frame["base_labels"].copy() if candidate_values is not None else None
    ids = list(map(int, frame["base_local_ids"]))
    descriptors = organ_data["descriptors"][frame["organ_global_indices"]]
    points = organ_data["points"][frame["organ_global_indices"]]
    forced_labels = set()
    if candidate_values is not None:
        masks = masks_from_frame(frame)
        probabilities = candidate_values[frame["candidate_global_indices"]]
        selected = select_indices(masks, np.flatnonzero(probabilities >= birth_threshold), probabilities)
        next_label = max(ids, default=0) + 1
        for candidate in selected:
            indices = masks[candidate]
            overlapping = labels[indices]
            values, counts = np.unique(overlapping[overlapping > 0], return_counts=True)
            already = int(values[counts.argmax()]) if len(values) else -1
            if already > 0 and counts.max() / max(len(indices) + np.sum(labels == already) - counts.max(), 1) > .8:
                forced_labels.add(already)
            else:
                labels[indices] = next_label
                forced_labels.add(next_label)
                next_label += 1
        record = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv").query("plant_uid == @plant and DAS == @das").iloc[0]
        cloud = read_frame(root / record.pointcloud_path)["xyz"]
        with np.load(root / f"outputs/m6_b/S2/seed_17/predictions/{plant}_{das}.npz") as prediction:
            core = cloud[prediction["semantic"] == 1]
        if not len(core):
            core = cloud[cloud[:, 2] <= np.quantile(cloud[:, 2], .1)]
        core_tree = cKDTree(core) if len(core) else None
        ids = list(map(int, np.unique(labels[labels > 0])))
        generator = np.random.default_rng(seed * 1000 + das)
        descriptors = np.asarray([shape_features(cloud[labels == local], len(cloud), core_tree) for local in ids])
        points = np.asarray([resample(cloud[labels == local] - cloud[labels == local].mean(axis=0), 128, generator) / .12 for local in ids])
        measured = {local: geometry_measure(cloud[labels == local]) for local in ids}
    else:
        measured = {int(row[0]): tuple(row[1:]) for row in frame["predicted_measures"]}
    generator = np.random.default_rng(seed * 100000 + das)
    renumber = dict(zip(ids, generator.permutation(np.arange(1, len(ids) + 1))))
    local_labels = np.zeros_like(labels) if labels is not None else None
    organs = []
    forced = set()
    for index, local in enumerate(ids):
        remapped = int(renumber[local])
        if local_labels is not None:
            local_labels[labels == local] = remapped
        organs.append({"local_id": remapped, "base_local_id": local, "descriptor": descriptors[index], "points": points[index], "measure": measured[local]})
        if local in forced_labels:
            forced.add(index)
    return local_labels, organs, forced


def run_tracking(root, method, seed, split, threshold, birth_method=None, birth_threshold=None, save=True):
    initialize(seed)
    settings = protocol(root)
    model, mean, scale = load_matcher(root, method, seed)
    candidates = pd.read_csv(root / "outputs/m7/candidates.csv")
    pair_rows = pd.read_csv(root / f"outputs/m7/matching/{method}/seed_{seed}/pair_probabilities.csv")
    pair_lookup = {(int(row.previous_organ_index), int(row.current_organ_index)): float(row.probability) for row in pair_rows.itertuples(index=False)}
    with np.load(root / "outputs/m7/organ_dataset.npz") as cached_organs:
        organ_data = {key: cached_organs[key] for key in ["descriptors", "points", "GT_labels_supervision_or_evaluation_only"]}
    candidate_values = None
    name = method
    if birth_method:
        candidate_values = pd.read_csv(root / f"outputs/m7/birth/{birth_method}/seed_{seed}/candidate_probabilities.csv").probability.to_numpy()
        name = f"{birth_method}+{method}"
    frames = pd.read_csv(root / "reports/m7/frame_audit.csv").query("split == @split")
    prediction_directory = root / f"outputs/m7/tracking/{name}/seed_{seed}/{split}/predictions"
    if save:
        prediction_directory.mkdir(parents=True, exist_ok=True)
    results, all_observations, all_births, all_tracks, all_links = [], [], [], [], []
    for plant, scans in frames.groupby("plant_id"):
        memory, next_track = {}, 0
        observations, births, track_rows, links = [], [], [], []
        previous_association = {}
        predicted_count = 0
        for row in scans.itertuples(index=False):
            with np.load(frame_path(root, plant, row.DAS)) as frame:
                labels, current, forced = prepared_organs(root, frame, plant, row.DAS, seed, organ_data,
                                                          candidate_values if row.frame_index > 0 else None, birth_threshold)
                for index, organ in enumerate(current):
                    if candidate_values is None:
                        organ["global_index"] = int(frame["organ_global_indices"][index])
                active = [value for value in memory.values() if row.frame_index - value["frame_index"] <= settings["tracking_training"]["maximum_gap_scans"]]
                if candidate_values is None:
                    probabilities = np.asarray([[pair_lookup[(old["global_index"], now["global_index"])] for old in active] for now in current]).reshape(len(current), len(active))
                else:
                    probabilities = score_pairs(model, mean, scale, current, active, lambda old: row.DAS - old["DAS"])
                assignments, born, next_track = assign_tracks(probabilities, [old["track_id"] for old in active], next_track, threshold, forced)
                if save:
                    output_labels = labels
                    if output_labels is None:
                        output_labels = np.zeros_like(frame["base_labels"])
                        for organ in current:
                            output_labels[frame["base_labels"] == organ["base_local_id"]] = organ["local_id"]
                    np.savez_compressed(prediction_directory / f"{plant}_{row.DAS}.npz", labels=output_labels,
                                        local_ids=np.asarray([organ["local_id"] for organ in current], dtype=np.int32),
                                        track_ids=np.asarray([assignments[index] for index in range(len(current))], dtype=np.int32),
                                        birth_local_ids=np.asarray([current[index]["local_id"] for index in sorted(born)], dtype=np.int32))
                for current_index, organ in enumerate(current):
                    track_id = assignments[current_index]
                    memory[track_id] = {**organ, "track_id": track_id, "frame_index": row.frame_index, "DAS": row.DAS}
                    track_rows.append({"plant_id": plant, "split": split, "method": name, "seed": seed, "DAS": row.DAS,
                                       "frame_index": row.frame_index, "local_id": organ["local_id"], "track_id": track_id,
                                       "length_cm": organ["measure"][0], "area_cm2": organ["measure"][1],
                                       "new_track": current_index in born, "forced_birth": current_index in forced})
                    if current_index in born:
                        births.append({"frame_index": row.frame_index, "local_id": organ["local_id"], "track_id": track_id})
                if labels is not None:
                    predicted_ids = sorted(map(int, np.unique(labels[labels > 0])))
                    masks = [np.flatnonzero(labels == local) for local in predicted_ids]
                    matches, unused_scores, gt_ids = strict_matches(frame["gt_evaluation_only"], masks)
                else:
                    predicted_ids = [organ["local_id"] for organ in current]
                    gt_ids = frame["gt_measures"][:, 0].astype(int)
                    matched_gt = organ_data["GT_labels_supervision_or_evaluation_only"][frame["organ_global_indices"]]
                    base_ious = candidates.iloc[frame["candidate_global_indices"][:len(current)]].GT_best_iou_evaluation_only.to_numpy()
                    matches = {int(label): (index, float(base_ious[index])) for index, label in enumerate(matched_gt) if label > 0}
                by_local = {organ["local_id"]: (assignments[index], organ["measure"]) for index, organ in enumerate(current)}
                current_association = {}
                gt_measure = {int(value[0]): tuple(value[1:]) for value in frame["gt_measures"]}
                for gt_id in gt_ids:
                    matched = matches.get(int(gt_id))
                    local = predicted_ids[matched[0]] if matched is not None else -1
                    track_id, measure = by_local.get(local, (-1, (np.nan, np.nan)))
                    observation = {"plant_id": plant, "split": split, "method": name, "seed": seed, "frame_index": row.frame_index,
                                   "DAS": row.DAS, "gt_id": int(gt_id), "local_id": local, "track_id": track_id,
                                   "iou": matched[1] if matched is not None else 0,
                                   "length_cm": measure[0], "area_cm2": measure[1]}
                    observations.append(observation)
                    current_association[int(gt_id)] = observation
                    if int(gt_id) in previous_association:
                        previous = previous_association[int(gt_id)]
                        detected = local > 0 and previous["local_id"] > 0
                        linked = detected and track_id >= 0 and previous["track_id"] == track_id
                        links.append({"plant_id": plant, "split": split, "method": name, "seed": seed, "gt_id": int(gt_id),
                                      "DAS": row.DAS, "previous_DAS": previous["DAS"], "detected_both": detected,
                                      "correct_link": linked, "error_type": "detection_failure" if not detected else "identity_failure" if not linked else "correct",
                                      "length_cm": measure[0], "area_cm2": measure[1],
                                      "previous_length_cm": previous["length_cm"], "previous_area_cm2": previous["area_cm2"]})
                previous_association = current_association
                predicted_count += len(current)
        identity, alignment = observation_identity(observations, predicted_count)
        event = event_metrics(observations, births)
        for row in observations:
            row["identity_correct"] = row["track_id"] >= 0 and alignment.get(row["gt_id"]) == row["track_id"]
            row["error_type"] = "detection_failure" if row["local_id"] < 0 else "identity_failure" if not row["identity_correct"] else "correct"
        results.append({"plant_id": plant, "split": split, "method": name, "seed": seed, "threshold": threshold,
                        **identity, **event, "correct_link_coverage": np.mean([row["correct_link"] for row in links]) if links else 0,
                        "detected_link_coverage": np.mean([row["detected_both"] for row in links]) if links else 0})
        all_observations.extend(observations)
        all_births.extend([{**row, "plant_id": plant, "split": split, "method": name, "seed": seed} for row in births])
        all_tracks.extend(track_rows)
        all_links.extend(links)
        print("TRACK", name, seed, split, plant, identity["idf1"], event["new_event_f1"], flush=True)
    result = pd.DataFrame(results)
    if save:
        directory = root / f"outputs/m7/tracking/{name}/seed_{seed}/{split}"
        directory.mkdir(parents=True, exist_ok=True)
        for filename, rows in [("plants", results), ("observations", all_observations), ("births", all_births), ("tracks", all_tracks), ("links", all_links)]:
            pd.DataFrame(rows).to_csv(directory / f"{filename}.csv", index=False)
        write_json(directory / "manifest.json", {"method": name, "seed": seed, "split": split, "GT_ID_inputs": False,
                   "independent_frame_IDs": True, "future_inputs": False, "segmentation_backbone_frozen": True,
                   "threshold": threshold, "birth_threshold": birth_threshold, "max_gap_scans": 2})
    return result


def select_tracking(root):
    settings = protocol(root)
    destination = root / "reports/m7/tracking_selection.json"
    if destination.exists():
        return json.loads(destination.read_text())
    rows, selected = [], {}
    for method in ["C1", "C2"]:
        for threshold in settings["tracking_training"]["pair_thresholds"]:
            for seed in settings["seeds"]:
                result = run_tracking(root, method, seed, "val", threshold, save=False)
                rows.extend(result.to_dict("records"))
        scores = pd.DataFrame(rows).query("method == @method").groupby("threshold").idf1.mean().reset_index().sort_values(["idf1", "threshold"], ascending=[False, False])
        selected[method] = {"threshold": float(scores.iloc[0].threshold), "mean_validation_IDF1": float(scores.iloc[0].idf1)}
    selected["selection_split"] = "val"
    selected["chosen_method"] = max(["C1", "C2"], key=lambda name: (selected[name]["mean_validation_IDF1"], name))
    pd.DataFrame(rows).to_csv(root / "reports/m7/tracking_threshold_validation.csv", index=False)
    write_json(destination, selected)
    return selected
