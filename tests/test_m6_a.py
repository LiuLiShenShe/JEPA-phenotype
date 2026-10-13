import json
from pathlib import Path

import numpy as np
import pytest

from m6a.adapters import OfficialSegmenter, OfficialTracker, simple_assign
from m6a.common import digest, localize_gt, normalize, paired_interval
from m6a.metrics import event_metrics, frame_metrics, geometry_measure, observation_identity, penalized_error
from m6a.experiment import measurement_losses


ROOT = Path(__file__).resolve().parents[1]


def test_perfect_pq_sbd_ignores_numeric_instance_ids():
    gt = np.array([0, 0, 1, 1, 2, 2])
    prediction = np.array([0, 0, 72, 72, 14, 14])
    metrics, association = frame_metrics(gt, prediction)
    assert metrics["pq"] == 1.0
    assert metrics["sbd"] == 1.0
    assert association[1][0] == 72


def test_merged_masks_pq_and_sbd():
    gt = np.array([1, 1, 2, 2])
    predicted = np.array([1, 1, 1, 1])
    metrics, association = frame_metrics(gt, predicted)
    assert metrics["pq"] == 0.0
    assert not association
    assert metrics["sbd"] == pytest.approx(2 / 3)
    assert metrics["merge_instances"] == 1


def test_false_leaf_background_counts_in_iou():
    gt = np.array([1, 1, 0, 0, 0])
    predicted = np.ones(5, dtype=int)
    metrics, association = frame_metrics(gt, predicted)
    assert metrics["pq"] == 0
    assert not association


def test_missing_segmentation_not_excluded_from_identity():
    rows = [{"gt_id": 1, "track_id": 8, "frame_index": 0},
            {"gt_id": 1, "track_id": -1, "frame_index": 1},
            {"gt_id": 1, "track_id": 9, "frame_index": 2}]
    metrics, alignment = observation_identity(rows, predicted_count=2)
    assert metrics["idf1"] == .4
    assert metrics["id_switches"] == 1
    assert metrics["complete_track_fraction"] == 0
    assert metrics["continuity"] == pytest.approx(1 / 3)
    assert metrics["link_recall"] == 0


def test_identity_switch_and_false_rebirth():
    rows = [{"gt_id": 1, "track_id": 1, "local_id": 4, "frame_index": 0},
            {"gt_id": 1, "track_id": 2, "local_id": 7, "frame_index": 1},
            {"gt_id": 2, "track_id": 3, "local_id": 8, "frame_index": 1}]
    births = [{"frame_index": 0, "local_id": 4}, {"frame_index": 1, "local_id": 7}, {"frame_index": 1, "local_id": 8}]
    metrics = event_metrics(rows, births)
    assert metrics["new_event_recall"] == 1
    assert metrics["new_event_precision"] == .5
    assert metrics["false_rebirth"] == 1
    assert metrics["spurious_birth"] == 0


def test_new_organ_missed_or_attached_to_old_track():
    rows = [{"gt_id": 1, "track_id": 3, "local_id": 2, "frame_index": 0},
            {"gt_id": 2, "track_id": -1, "local_id": -1, "frame_index": 1},
            {"gt_id": 3, "track_id": 3, "local_id": 4, "frame_index": 1}]
    metrics = event_metrics(rows, [])
    assert metrics["new_missed_instance"] == 1
    assert metrics["new_attached_old_identity"] == 1
    assert metrics["new_event_fn"] == 2


def test_new_organ_sampling_loss_separate_from_segmentation():
    rows = [{"gt_id": 1, "track_id": -1, "local_id": 4, "frame_index": 1}]
    metrics = event_metrics(rows, [])
    assert metrics["new_sampling_or_track_loss"] == 1
    assert metrics["new_missed_instance"] == 0


def test_missing_measurement_penalty_not_zero_trait():
    assert penalized_error(np.nan, 12.) == 2.
    assert penalized_error(12., 12.) == 0.
    assert penalized_error(60., 12.) == 2.
    assert penalized_error(-3., -2., scale=10.) == .1


def test_geometry_units_and_planar_area():
    axis_x, axis_y = np.meshgrid(np.linspace(0, .1, 100), np.linspace(0, .05, 60))
    cloud = np.column_stack([axis_x.ravel(), axis_y.ravel(), np.zeros(axis_x.size)])
    length, area = geometry_measure(cloud)
    assert length == pytest.approx(10., abs=.3)
    assert area == pytest.approx(50., abs=.1)


def test_simple_tracking_rejects_distant_birth():
    previous_xyz = np.array([[0., 0, 0], [1., 0, 0]])
    previous_track = np.array([9, 14])
    current_xyz = np.array([[0.01, 0, 0], [2., 0, 0]])
    current_local = np.array([3, 7])
    assigned, confidence, births, next_id = simple_assign(previous_xyz, previous_track, current_xyz, current_local, 20)
    assert assigned == {3: 9, 7: 20}
    assert births == {7}
    assert next_id == 21
    assert confidence[3] > .9


def test_simple_tracking_handles_no_detected_current_organs():
    previous_xyz = np.array([[0., 0, 0]])
    assigned, confidence, births, next_id = simple_assign(previous_xyz, np.array([1]), np.empty((0, 3)), np.empty(0, dtype=int), 5)
    assert assigned == {} and confidence == {} and births == set() and next_id == 5


def test_local_gt_relabeling_destoys_temporal_numeric_identity():
    gt = np.tile(np.arange(0, 10), 3)
    first = localize_gt(gt, 1)
    second = localize_gt(gt, 2)
    assert not np.array_equal(first, second)
    assert frame_metrics(first, second)[0]["pq"] == 1.
    assert np.all(first[gt == 0] == 0)


def test_official_tracker_matching_equivariant_to_local_label_permutation():
    tracker = OfficialTracker(ROOT)
    previous_xyz = np.array([[0, 0, 0], [.01, 0, 0], [1, 0, 0], [1.01, 0, 0]], dtype=float)
    previous_tracks = np.array([20, 20, 80, 80])
    current_xyz = previous_xyz + .01
    first = tracker.assign(previous_xyz, previous_tracks, current_xyz, np.array([0, 0, 1, 1]), 90)
    second = tracker.assign(previous_xyz, previous_tracks, current_xyz, np.array([7, 7, 2, 2]), 90)
    assert first[0][0] == second[0][7]
    assert first[0][1] == second[0][2]
    assert first[2] == second[2] == set()


def test_xyz_only_contract_rejects_labels_before_loading_model():
    segmenter = object.__new__(OfficialSegmenter)
    with pytest.raises(ValueError, match="XYZ only"):
        segmenter.predict(np.zeros((4096, 8)), 3)


def test_plant_bootstrap_is_not_scan_bootstrap():
    result = paired_interval([.3] * 12)
    assert result["plants"] == 12
    assert result["mean"] == pytest.approx(.3)
    assert result["ci95_low"] == pytest.approx(.3)


def test_protocol_hash_and_plant_disjoint_split():
    lock = json.loads((ROOT / "reports/m6_a/protocol_lock.json").read_text())
    assert digest(ROOT / "configs/m6_a/protocol.json") == lock["sha256"]["configs/m6_a/protocol.json"]
    splits = json.loads((ROOT / "reports/m3/plant_splits.json").read_text())["assignments"]["official"]
    assert len(splits) == 48
    assert list(splits.values()).count("train") == 24
    assert list(splits.values()).count("val") == 12
    assert list(splits.values()).count("test") == 12
    assert json.loads((ROOT / "configs/m6_a/protocol.json").read_text())["segmentation"]["training"] is False


def test_wrong_tracking_propagates_but_oracle_links_stay_evaluation_only():
    previous = {"DAS": 37, "association": {1: (10, 1.), 2: (20, 1.)}, "gt_association": {1: (10, 1.), 2: (20, 1.)},
                "measure": {10: (10., 10.), 20: (30., 30.)}, "gt_measure": {10: (10., 10.), 20: (30., 30.)},
                "tracks": {10: 1, 20: 2}, "confidence": {10: .9, 20: .9}}
    current = {"DAS": 40, "association": {1: (11, 1.), 2: (21, 1.)}, "gt_association": {1: (11, 1.), 2: (21, 1.)},
               "measure": {11: (12., 12.), 21: (32., 32.)}, "gt_measure": {11: (12., 12.), 21: (32., 32.)},
               "tracks": {11: 2, 21: 1}, "confidence": {11: .9, 21: .9}}
    reference = {(37, 1): (10., 10.), (37, 2): (30., 30.), (40, 1): (12., 12.), (40, 2): (32., 32.)}
    rows = measurement_losses("synthetic", "test", "B1", [previous, current], reference)
    oracle = [row for row in rows if row["kind"] == "change" and row["stage"] == "PRED_MASK_ORACLE_LINK"]
    tracked = [row for row in rows if row["kind"] == "change" and row["stage"] == "TRACKED"]
    assert all(row["loss"] == 0 for row in oracle)
    assert all(row["status"] == "wrong_match" for row in tracked)
    assert all(row["loss"] > .5 for row in tracked)


def test_identity_loss_penalized_even_with_perfect_static_measurement():
    first = {"DAS": 37, "association": {1: (10, 1.)}, "gt_association": {1: (10, 1.)},
             "measure": {10: (10., 10.)}, "gt_measure": {10: (10., 10.)}, "tracks": {10: 1}, "confidence": {10: 1.}}
    second = {"DAS": 40, "association": {1: (20, 1.)}, "gt_association": {1: (20, 1.)},
              "measure": {20: (12., 12.)}, "gt_measure": {20: (12., 12.)}, "tracks": {20: 2}, "confidence": {20: 1.}}
    rows = measurement_losses("synthetic", "test", "B1", [first, second], {(37, 1): (10., 10.), (40, 1): (12., 12.)})
    tracked = [row for row in rows if row["kind"] == "change" and row["stage"] == "TRACKED"]
    assert all(row["loss"] == 2 and row["severe"] for row in tracked)
    assert all(row["status"] == "identity_lost" for row in tracked)
