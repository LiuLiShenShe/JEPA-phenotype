from pathlib import Path

import numpy as np
import pytest
import torch

from birth_detection.common import paired_interval, strict_matches, window_indices
from birth_detection.analysis import holm_adjust, sign_flip_probability, track_level_results
from birth_detection.dataset import pair_features, resample, shape_features
from birth_detection.evaluation import frame_event_result, oracle_birth_indices, select_indices
from birth_detection.training import forecast_features, normalize_features
from birth_detection.reporting import temporal_event_rows
from organ_tracking.engine import assign_tracks
from temporal_encoder.models import BirthEncoder, PairMatcher


def test_temporal_encoder_accepts_float64_timing():
    model = BirthEncoder(110, temporal=True)
    output = model(torch.zeros(2, 3, 128, 3), torch.zeros(2, 3, 2, dtype=torch.float64), torch.zeros(2, 110))
    output.sum().backward()
    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_candidate_oracle_is_evaluator_only_and_selects_new_ids():
    frame = {"gt_evaluation_only": np.array([1, 1, 1, 2, 2, 2]), "gt_new_ids_evaluation_only": np.array([2])}
    masks = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([3, 4, 5])]
    selected = oracle_birth_indices(frame, masks)
    assert len(selected) == 1
    assert selected[0] in {1, 2}


def test_cached_event_iou_exactly_matches_direct_evaluation():
    frame = {"gt_evaluation_only": np.array([1, 1, 1, 2, 2, 2, 0]), "gt_new_ids_evaluation_only": np.array([2]),
             "indices": np.array([0, 1, 2, 3, 4, 5, 3, 4]), "offsets": np.array([0, 3, 6, 8])}
    masks = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([3, 4])]
    unused_matches, scores, gt_ids = strict_matches(frame["gt_evaluation_only"], masks)
    cached = {**frame, "_iou_scores": scores, "_gt_ids": gt_ids}
    for selected in [[], [0], [1], [2], [2, 0], [0, 1, 2]]:
        assert frame_event_result(frame, selected) == frame_event_result(cached, selected)
    assert oracle_birth_indices(frame, masks) == oracle_birth_indices(cached, masks)


def test_exact_plant_sign_flip_does_not_count_seeds_as_plants():
    assert sign_flip_probability(np.ones(12)) == pytest.approx(2 / 4096)
    assert sign_flip_probability(np.zeros(12)) == 1.


def test_holm_adjustment_is_bounded_and_monotone():
    assert np.allclose(holm_adjust([.04, .01]), [.04, .02])
    assert np.allclose(holm_adjust([.9, .8]), [1., 1.])


def test_single_frame_control_excludes_proposal_source_and_novelty():
    path = Path("outputs/m7/birth_dataset.npz")
    if not path.exists():
        pytest.skip("M7 integration cache is not present")
    data = np.load(path)
    assert data["single_features"].shape[1] == 54
    expected = np.concatenate([data["temporal_features"][:, :25], data["temporal_features"][:, 27:29],
                               data["temporal_features"][:, 79:104], data["temporal_features"][:, 108:110]], axis=1)
    assert np.allclose(data["single_features"], expected)


def test_causal_window():
    assert window_indices(0) == [0, 0, 0]
    assert window_indices(1) == [0, 0, 1]
    for index in range(16):
        assert max(window_indices(index)) == index


def test_strict_iou_and_leaf_zero_remapping():
    gt = np.array([1, 1, 1, 1, 0, 0])
    matches, unused_scores, unused_ids = strict_matches(gt, [np.array([0, 1])])
    assert not matches
    matches, unused_scores, unused_ids = strict_matches(gt, [np.array([0, 1, 2])])
    assert matches[1][1] == .75


def test_one_to_one_assignment():
    assignments, born, next_track = assign_tracks(np.array([[.9], [.8]]), [7], 8, .5)
    assert assignments == {0: 7, 1: 8}
    assert born == {1}
    assert next_track == 9


def test_explicit_birth_never_inherits_old_identity():
    assignments, born, next_track = assign_tracks(np.array([[.99]]), [3], 4, .5, {0})
    assert assignments[0] == 4 and born == {0} and next_track == 5


def test_empty_active_tracks():
    assignments, born, next_track = assign_tracks(np.empty((2, 0)), [], 0, .5)
    assert assignments == {0: 0, 1: 1} and born == {0, 1} and next_track == 2


def test_shape_features_physical_and_finite():
    generator = np.random.default_rng(17)
    points = generator.normal(size=(128, 3)).astype(np.float32) * .01
    features = shape_features(points, 1000)
    assert features.shape == (25,) and np.isfinite(features).all()
    assert features[4] == pytest.approx(.128)
    assert len(pair_features(features, features, 3)) == 75


def test_normalizer_train_only():
    values = np.array([[1., 2.], [3., 4.], [1000., 2000.]], dtype=np.float32)
    unused_normalized, mean, scale = normalize_features(values, np.array([True, True, False]))
    assert mean.tolist() == [2., 3.]
    assert scale.tolist() == [1., 1.]


def test_sampling_does_not_require_gt():
    result = resample(np.zeros((0, 3)), 16, np.random.default_rng(0))
    assert result.shape == (16, 3)


def test_nms_no_double_event():
    masks = [np.array([0, 1, 2]), np.array([0, 1, 2]), np.array([3, 4, 5])]
    assert select_indices(masks, [0, 1, 2], [.7, .9, .5]) == [1, 2]


@pytest.mark.parametrize("temporal", [False, True])
def test_birth_encoder_real_backward(temporal):
    torch.manual_seed(17)
    model = BirthEncoder(8, temporal)
    output = model(torch.randn(4, 3, 16, 3), torch.randn(4, 3, 2), torch.randn(4, 8))
    output.square().mean().backward()
    assert output.shape == (4,) and all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_supervised_metric_matcher_backward():
    model = PairMatcher(75, True)
    logits, distance = model(torch.randn(4, 75), torch.randn(4, 16, 3), torch.randn(4, 16, 3))
    (logits.square().mean() + distance.mean()).backward()
    assert distance.shape == (4,) and torch.isfinite(logits).all()


def test_plant_bootstrap_not_frames():
    interval = paired_interval([.1] * 12)
    assert interval["plants"] == 12 and interval["lower"] == pytest.approx(.1)


def test_forecast_no_future_feature():
    import pandas as pd

    rows = pd.DataFrame({"plant_id": ["a"] * 3, "points": [1, 2, 999], "voxel_count": [1, 2, 999],
                         "new_fraction": [0, 1, 999], "predicted_leaf_count": [1, 2, 999], "height_m": [1, 2, 999],
                         "width_m": [1, 2, 999], "core_predicted_points": [1, 2, 999]})
    features = forecast_features(rows)
    assert features[1].max() <= 2


def test_track_report_keeps_unmatched_scans_and_missed_leaves(tmp_path):
    import pandas as pd

    directory = tmp_path / "outputs/m7/baseline_import"
    directory.mkdir(parents=True)
    (tmp_path / "reports/m7").mkdir(parents=True)
    common = {"plant_id": "a", "split": "val", "method": "Current", "seed": 17}
    tracks = [{**common, "frame_index": index, "DAS": index + 37, "local_id": 1, "track_id": 2} for index in range(3)]
    observations = [{**common, "frame_index": 0, "DAS": 37, "local_id": 1, "track_id": 2, "gt_id": 1, "identity_correct": True},
                    {**common, "frame_index": 1, "DAS": 38, "local_id": 1, "track_id": 2, "gt_id": 2, "identity_correct": False},
                    {**common, "frame_index": 2, "DAS": 39, "local_id": -1, "track_id": -1, "gt_id": 1, "identity_correct": False}]
    pd.DataFrame(tracks).to_csv(directory / "tracks.csv", index=False)
    pd.DataFrame(observations).to_csv(directory / "observations.csv", index=False)
    track_level_results(tmp_path)
    predicted = pd.read_csv(tmp_path / "reports/m7/track_level_results.csv").iloc[0]
    reference = pd.read_csv(tmp_path / "reports/m7/gt_track_level_results.csv").set_index("gt_id")
    assert predicted.observed_scans == 3 and predicted.GT_matched_scans_evaluation_only == 2
    assert predicted.identity_purity_among_matched == .5 and predicted.identity_correct_scans == 1
    assert reference.loc[1, "missed_observations"] == 1


def test_event_export_includes_unselected_combined_system(tmp_path):
    import pandas as pd

    (tmp_path / "reports/m7").mkdir(parents=True)
    pd.DataFrame([{"split": "val", "plant_id": "a", "leaf_instance_id": 0, "birth_time_visible": 40,
                   "previous_observation_available": True, "small": True, "core_adjacent": False}]).to_csv(tmp_path / "reports/m7/organ_birth_events.csv", index=False)
    observations = pd.DataFrame([{"split": "val", "plant_id": "a", "gt_id": 1, "frame_index": 1, "DAS": 40, "local_id": 1}])
    births = pd.DataFrame([{"split": "val", "plant_id": "a", "frame_index": 1, "local_id": 1}])
    directories = [tmp_path / "outputs/m7/baseline_import"]
    directories.extend(tmp_path / f"outputs/m7/tracking/{method}/seed_{seed}/val" for method in ["C1", "PointNet_GRU+C1"] for seed in [17, 29, 43])
    for directory in directories:
        directory.mkdir(parents=True)
        observations.to_csv(directory / "observations.csv", index=False)
        births.to_csv(directory / "births.csv", index=False)
    rows = temporal_event_rows(tmp_path, "val", "C1")
    assert len(rows) == 7 and rows.correct_birth.all()
    assert set(rows.method) == {"Current", "C1", "PointNet_GRU+C1"}
