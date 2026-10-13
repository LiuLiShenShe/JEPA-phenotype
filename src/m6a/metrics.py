from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, QhullError


def frame_metrics(gt, predicted):
    gt_ids, gt_index = np.unique(gt, return_inverse=True)
    pred_ids, pred_index = np.unique(predicted, return_inverse=True)
    intersections = np.bincount(gt_index * len(pred_ids) + pred_index,
                                minlength=len(gt_ids) * len(pred_ids)).reshape(len(gt_ids), len(pred_ids))
    gt_counts = intersections.sum(axis=1)
    pred_counts = intersections.sum(axis=0)
    leaf_gt = gt_ids > 0
    leaf_pred = pred_ids > 0
    overlap = intersections[leaf_gt][:, leaf_pred]
    gt_counts = gt_counts[leaf_gt]
    pred_counts = pred_counts[leaf_pred]
    gt_ids = gt_ids[leaf_gt]
    pred_ids = pred_ids[leaf_pred]
    union = gt_counts[:, None] + pred_counts[None, :] - overlap
    iou = overlap / np.maximum(union, 1)
    dice = 2 * overlap / np.maximum(gt_counts[:, None] + pred_counts[None, :], 1)
    matches = {}
    if len(gt_ids) and len(pred_ids):
        rows, cols = linear_sum_assignment(-iou)
        matches = {int(gt_ids[row]): (int(pred_ids[col]), float(iou[row, col])) for row, col in zip(rows, cols) if iou[row, col] > .5}
        sbd = min(float(dice.max(axis=1).mean()), float(dice.max(axis=0).mean()))
    else:
        sbd = float(not len(gt_ids) and not len(pred_ids))
    true_positive = len(matches)
    denominator = .5 * (len(gt_ids) + len(pred_ids))
    pq = sum(value[1] for value in matches.values()) / denominator if denominator else 1.0
    merges = int(((overlap / np.maximum(gt_counts[:, None], 1) > .1).sum(axis=0) > 1).sum())
    splits = int(((overlap / np.maximum(gt_counts[:, None], 1) > .1).sum(axis=1) > 1).sum())
    return {"pq": pq, "sbd": sbd, "gt_count": len(gt_ids), "pred_count": len(pred_ids),
            "tp": true_positive, "fp": len(pred_ids) - true_positive, "fn": len(gt_ids) - true_positive,
            "merge_instances": merges, "split_gt_instances": splits}, matches


def geometry_measure(cloud):
    if len(cloud) < 4:
        return (float("nan"), float("nan"))
    sampled = cloud[np.linspace(0, len(cloud) - 1, min(len(cloud), 4096)).astype(int)].astype(float) * 100
    centered = sampled - sampled.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ axes[:2].T
    length = float(np.quantile(plane[:, 0], .99) - np.quantile(plane[:, 0], .01))
    try:
        area = float(ConvexHull(plane).volume)
    except QhullError:
        area = float("nan")
    return length, area


def measurements(cloud, labels):
    return {int(label): geometry_measure(cloud[labels == label]) for label in np.unique(labels) if label > 0}


def penalized_error(estimate, reference, scale=None):
    if not np.isfinite(estimate):
        return 2.0
    denominator = max(abs(reference if scale is None else scale), 1.0)
    return float(min(abs(estimate - reference) / denominator, 2.0))


def observation_identity(rows, predicted_count):
    gt_ids = sorted({record["gt_id"] for record in rows})
    track_ids = sorted({record["track_id"] for record in rows if record["track_id"] >= 0})
    counts = np.zeros((len(gt_ids), len(track_ids)), dtype=int)
    gt_lookup = {value: index for index, value in enumerate(gt_ids)}
    track_lookup = {value: index for index, value in enumerate(track_ids)}
    for record in rows:
        if record["track_id"] >= 0:
            counts[gt_lookup[record["gt_id"]], track_lookup[record["track_id"]]] += 1
    aligned = {}
    idtp = 0
    if counts.size:
        row_index, column_index = linear_sum_assignment(-counts)
        aligned = {gt_ids[row]: track_ids[column] for row, column in zip(row_index, column_index) if counts[row, column] > 0}
        idtp = int(counts[row_index, column_index].sum())
    detected = sum(record["track_id"] >= 0 for record in rows)
    switches = 0
    continuity = []
    complete = []
    correct_links = 0
    eligible_links = 0
    for gt_id in gt_ids:
        sequence = sorted([record for record in rows if record["gt_id"] == gt_id], key=lambda record: record["frame_index"])
        last_detected = None
        longest = 0
        run = 0
        previous = None
        observed_tracks = []
        for record in sequence:
            track = record["track_id"]
            if track >= 0:
                observed_tracks.append(track)
                if last_detected is not None and last_detected != track:
                    switches += 1
                last_detected = track
            if track >= 0:
                run = run + 1 if previous is not None and previous["frame_index"] == record["frame_index"] - 1 and previous["track_id"] == track else 1
            else:
                run = 0
            longest = max(longest, run)
            if previous is not None and previous["frame_index"] == record["frame_index"] - 1:
                eligible_links += 1
                correct_links += int(track >= 0 and previous["track_id"] == track)
            previous = record
        continuity.append(longest / len(sequence))
        complete.append(len(observed_tracks) == len(sequence) and len(set(observed_tracks)) == 1)
    denominator = len(rows) + predicted_count
    return {"idf1": 2 * idtp / denominator if denominator else 1.0,
            "identity_accuracy_detected": idtp / detected if detected else 0.0,
            "identity_observation_recall": idtp / len(rows) if rows else 1.0,
            "id_switches": switches, "id_switch_rate": switches / max(eligible_links, 1),
            "link_recall": correct_links / eligible_links if eligible_links else 1.0,
            "continuity": float(np.mean(continuity)) if continuity else 1.0,
            "complete_track_fraction": float(np.mean(complete)) if complete else 1.0,
            "gt_observations": len(rows), "predicted_observations": predicted_count,
            "idtp": idtp, "idfn": len(rows) - idtp, "idfp": predicted_count - idtp}, aligned


def event_metrics(rows, births):
    first = {}
    observations = {}
    for record in sorted(rows, key=lambda record: record["frame_index"]):
        first.setdefault(record["gt_id"], record)
        if record["local_id"] >= 0:
            observations[(record["frame_index"], record["local_id"])] = record
    new = [record for record in first.values() if record["frame_index"] > 0]
    birth_set = {(record["frame_index"], record["local_id"]) for record in births if record["frame_index"] > 0}
    true_positives = sum((record["frame_index"], record["local_id"]) in birth_set for record in new if record["local_id"] >= 0 and record["track_id"] >= 0)
    false_rebirth = 0
    spurious = 0
    for key in birth_set:
        if key not in observations:
            spurious += 1
        elif observations[key]["frame_index"] > first[observations[key]["gt_id"]]["frame_index"]:
            false_rebirth += 1
    precision = true_positives / len(birth_set) if birth_set else 0.0
    recall = true_positives / len(new) if new else 1.0
    return {"new_event_tp": true_positives, "new_event_fp": len(birth_set) - true_positives,
            "new_event_fn": len(new) - true_positives, "new_event_precision": precision,
            "new_event_recall": recall, "new_event_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "new_missed_instance": sum(record["local_id"] < 0 for record in new),
            "new_sampling_or_track_loss": sum(record["local_id"] >= 0 and record["track_id"] < 0 for record in new),
            "new_attached_old_identity": sum(record["local_id"] >= 0 and record["track_id"] >= 0 and (record["frame_index"], record["local_id"]) not in birth_set for record in new),
            "false_rebirth": false_rebirth, "spurious_birth": spurious, "new_gt_events": len(new)}
