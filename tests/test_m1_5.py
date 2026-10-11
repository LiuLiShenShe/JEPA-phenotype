"""M1.5 timestamp matching, task-information and fair-training contracts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from m15.data import DiagnosticDataset, calendar_windows, time_conditions
from m15.analysis import bootstrap_spearman, decision, holm_adjust, paired_statistics
from m15.engine import fast_selection_score, plant_weights, predict, state_digest, train
from m15.models import DiagnosticHead, model_spec
from training.metrics import selection_score


PROTOCOL = json.loads((Path(__file__).resolve().parents[1] / "configs/m1_5/protocol.json").read_text())


def make_frames():
    timestamps = pd.to_datetime(["2022-05-01 00:00", "2022-05-01 06:00", "2022-05-01 12:00",
                                 "2022-05-04 06:00", "2022-05-04 18:00", "2022-05-07 12:00"])
    return pd.DataFrame({"plant_uid": ["g/p1"] * 6, "genotype": ["g"] * 6, "replicate": ["p1"] * 6,
                         "frame_idx": list(range(6)), "timestamp": timestamps.astype(str), "quality_status": ["valid"] * 6,
                         "timestamp_reliable": [True] * 6, "rgb_path": [f"rgb{position}" for position in range(6)],
                         "leaf_count": [5, 5, 6, 7, 8, 10], "projected_area_px": [10, 11, 12, 13, 14, 15],
                         "projected_area_ratio": [.1, .11, .12, .13, .14, .15], "growth_stage": ["early"] * 6})


def test_unique_nearest_timestamp_tie_and_tolerance():
    rows = calendar_windows(make_frames(), {"g/p1": "train"}, [3], 6)
    selected = rows.loc[rows.window_id.eq("g/p1__end2__day3")].iloc[0]
    assert selected.target_frame_idx == 3
    assert selected.matching_error_hours == -6
    assert not rows.window_id.duplicated().any()
    with pytest.raises(ValueError, match="No calendar-time"):
        calendar_windows(make_frames().iloc[:5], {"g/p1": "train"}, [3], 5.9)


def test_matching_is_independent_of_label_value_and_order():
    frames = make_frames()
    before = calendar_windows(frames, {"g/p1": "train"}, [3], 6)
    frames["leaf_count"] = [100, 90, 80, 70, 60, 50]
    after = calendar_windows(frames.sample(frac=1, random_state=7), {"g/p1": "train"}, [3], 6)
    assert before.window_id.tolist() == after.window_id.tolist()
    assert before.target_frame_idx.tolist() == after.target_frame_idx.tolist()


def test_quarantine_and_missing_original_index_not_compressed():
    frames = make_frames()
    frames.loc[1, "quality_status"] = "quarantine"
    with pytest.raises(ValueError, match="No calendar-time"):
        calendar_windows(frames.iloc[:5], {"g/p1": "train"}, [3], 6)
    frames = make_frames().drop(index=1)
    rows = calendar_windows(frames, {"g/p1": "train"}, [3], 6)
    assert "g/p1__end2__day3" not in set(rows.window_id)


def test_day_and_offset_inputs_cannot_reveal_unknown_future():
    frames = make_frames()
    rows = calendar_windows(frames, {"g/p1": "train"}, [3], 6)
    altered = rows.assign(target_time_delta_seconds=9999999, target_frame_idx=9999,
                          target_timestamp="2099-01-01", target_leaf_count=500)
    for condition in ("full", "rgb"):
        np.testing.assert_array_equal(time_conditions(rows, frames, "days", condition), time_conditions(altered, frames, "days", condition))
    for condition in ("offset", "relative", "rgb"):
        np.testing.assert_array_equal(time_conditions(rows, frames, "offsets", condition), time_conditions(altered, frames, "offsets", condition))
    assert np.all(time_conditions(rows, frames, "offsets", "rgb") == 0)
    assert np.all(time_conditions(rows, frames, "offsets", "relative")[:, [4, 6, 7, 8, 9]] == 0)


def test_gru_repeat_same_capacity_and_initialization_but_different_input():
    torch.manual_seed(42)
    genuine = DiagnosticHead(model_spec("gru_full", PROTOCOL))
    torch.manual_seed(42)
    repeated = DiagnosticHead(model_spec("gru_repeat", PROTOCOL))
    assert state_digest(genuine) == state_digest(repeated)
    assert sum(parameter.numel() for parameter in genuine.parameters()) == sum(parameter.numel() for parameter in repeated.parameters()) == 561922
    genuine.eval()
    repeated.eval()
    features = torch.randn(2, 3, 384)
    times = torch.randn(2, 10)
    assert not torch.allclose(genuine(features, times), repeated(features, times))
    torch.testing.assert_close(repeated(features, times), genuine(features[:, -1:].expand(-1, 3, -1), times))


def test_time_only_and_single_frame_ignore_excluded_visuals():
    features = torch.randn(2, 3, 384)
    times = torch.randn(2, 10)
    model = DiagnosticHead(model_spec("time_full", PROTOCOL)).eval()
    torch.testing.assert_close(model(features, times), model(features * 100, times))
    model = DiagnosticHead(model_spec("mlp_full", PROTOCOL)).eval()
    altered = features.clone()
    altered[:, :2] *= 100
    torch.testing.assert_close(model(features, times), model(altered, times))


def test_vectorized_validation_score_matches_existing_plant_macro():
    rows = pd.DataFrame({"plant_uid": ["p1", "p1", "p2"], "genotype": ["g"] * 3, "window_id": ["a", "b", "c"]})
    targets = np.array([[5., .1], [7., .2], [6., .3]])
    prediction = targets + [[1., .01], [2., .02], [3., .03]]
    scale = np.array([2., .1])
    fast = fast_selection_score(torch.tensor(prediction), torch.tensor(targets), torch.tensor(scale), torch.tensor(plant_weights(rows)))
    assert fast == pytest.approx(selection_score(rows, targets, prediction, scale), abs=1e-12)


def tiny_data(split, plant, seed):
    class Data:
        def __len__(self):
            return len(self.rows)
    data = Data()
    generator = np.random.default_rng(seed)
    data.rows = pd.DataFrame({"split": [split] * 8, "plant_uid": [plant] * 8, "window_id": [f"{plant}_{position}" for position in range(8)]})
    data.features = generator.normal(size=(8, 3, 384)).astype(np.float32)
    data.times = generator.normal(size=(8, 10)).astype(np.float32)
    data.targets = generator.uniform(size=(8, 2)).astype(np.float32)
    return data


def test_both_grus_retrained_from_scratch_equal_optimizer_steps(tmp_path):
    train_data = tiny_data("train", "p1", 2)
    val_data = tiny_data("val", "p2", 3)
    results = []
    for name in ("gru_full", "gru_repeat"):
        config = {**model_spec(name, PROTOCOL), "seed": 42, "max_epochs": 2, "batch_size": 4}
        results.append(train(train_data, val_data, config, tmp_path / name, "cpu"))
    for key in ("initial_state_sha256", "batch_order_sha256", "optimizer_steps", "trainable_parameters", "epochs"):
        assert results[0][key] == results[1][key]
    assert results[0]["optimizer_steps"] == 4
    assert not results[0]["test_used_for_selection"]
    original_prediction = predict(tmp_path / "gru_full/checkpoint.pt", val_data, "cpu")
    val_data.targets[:] = 1000000
    np.testing.assert_array_equal(original_prediction, predict(tmp_path / "gru_full/checkpoint.pt", val_data, "cpu"))
    with pytest.raises(ValueError, match="test cannot enter"):
        train(tiny_data("test", "p3", 4), val_data, config, tmp_path / "bad", "cpu")


def test_holm_and_paired_plant_interval_not_window_bootstrap():
    np.testing.assert_allclose(holm_adjust([.04, .001, .02]), [.04, .003, .04])
    settings = {**PROTOCOL["statistics"], "bootstrap_replicates": 1000, "permutation_replicates": 1000}
    result = paired_statistics([-.1, -.2, -.3, -.4, -.5], settings)
    assert result["plants"] == 5
    assert result["delta_mae"] == pytest.approx(-.3)
    assert result["ci95_high"] < 0
    assert result["plants_with_gain"] == 5
    assert 0 < result["paired_sign_flip_p_two_sided"] <= 1
    with pytest.raises(ValueError, match="finite matched"):
        paired_statistics([float("nan")], settings)


def test_spearman_small_groups_do_not_fabricate_uncertainty():
    assert bootstrap_spearman([1, 1, 1, 1, 1], [1, 2, 3, 4, 5], PROTOCOL["statistics"])["rho"] is None
    result = bootstrap_spearman(range(5), range(5), {**PROTOCOL["statistics"], "bootstrap_replicates": 100})
    assert result["rho"] == pytest.approx(1)
    assert result["ci95_low"] == pytest.approx(1)


def test_decision_requires_both_controls_and_each_seed():
    records = [{"task": "day5", "target": "leaf_count", "family": "capacity", "delta_mae": -.15,
                "relative_gain": .1, "all_seeds_favor_first": True, "ci95_low": -.2, "ci95_high": -.1, "holm_p": .01,
                "second": name} for name in ("mlp_full", "gru_repeat")]
    rows = pd.DataFrame(records)
    assert decision(rows, PROTOCOL, [3, 5, 7])["research_judgment"] == "CONTINUE"
    rows.loc[0, "all_seeds_favor_first"] = False
    assert decision(rows, PROTOCOL, [3, 5, 7])["research_judgment"] == "REVIEW"
    assert decision(rows.iloc[:1], PROTOCOL, [3, 5, 7])["research_judgment"] != "CONTINUE"


def test_real_task_eligibility_does_not_use_test_counts():
    from m15.data import feasibility
    rows = calendar_windows(make_frames(), {"g/p1": "train"}, [3], 6)
    other = rows.copy().assign(plant_uid="g/p2", split="val")
    both = pd.concat([rows, other], ignore_index=True)
    minimal = {**PROTOCOL, "day_horizons": [3], "matching": {**PROTOCOL["matching"], "minimum_train_plants": 1,
               "minimum_val_plants": 1, "minimum_train_median_windows": 1, "minimum_val_median_windows": 1}}
    summary, per_plant, coverage, eligible = feasibility(both, {"g/p1": "train", "g/p2": "val", "g/p3": "test"}, minimal)
    assert eligible == [3]
    assert summary.loc[summary.split.eq("test"), "plants"].iloc[0] == 0
    assert per_plant.loc[per_plant.plant_uid.eq("g/p3"), "windows"].iloc[0] == 0


def test_completed_real_m15_runs_have_three_seeds_and_budget():
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/m1_5/training_complete.json"
    if not path.exists() or json.loads(path.read_text())["status"] != "complete":
        pytest.skip("Requires complete real M1.5 training")
    from m15.analysis import controlled_training_audit
    from training.common import file_hash
    lock_path = root / "reports/m1_5/protocol_lock.json"
    lock = json.loads(lock_path.read_text())
    checked = controlled_training_audit(root, lock)
    assert len(checked) == 18
    runs = pd.read_csv(root / "reports/m1_5/training_resources.csv")
    assert len(runs) == 180
    assert set(runs.seed) == {42, 43, 44}
    assert set(runs.epochs) == {100}
    assert set(runs.protocol_lock_sha256) == {file_hash(lock_path)}
    for _, group in runs.groupby(["task", "model"]):
        assert len(group) == 3
    for source, digest in json.loads((root / "reports/m1_5/training_source_fingerprints.json").read_text()).items():
        if source == "src/m15/engine.py":
            correction = json.loads((root / "reports/m1_5/inference_precision_correction.json").read_text())
            assert file_hash(root / correction["original_source_snapshot"]) == digest
            assert file_hash(root / source) == correction["corrected_engine_sha256"]
            import ast
            before = ast.parse((root / correction["original_source_snapshot"]).read_text())
            after = ast.parse((root / source).read_text())
            before_train = next(node for node in before.body if isinstance(node, ast.FunctionDef) and node.name == "train")
            after_train = next(node for node in after.body if isinstance(node, ast.FunctionDef) and node.name == "train")
            assert ast.dump(before_train) == ast.dump(after_train)
        else:
            assert file_hash(root / source) == digest


def test_completed_real_m15_prediction_coverage_is_identical():
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/m1_5/evaluation_complete.json"
    if not path.exists() or json.loads(path.read_text())["status"] != "complete":
        pytest.skip("Requires complete real M1.5 evaluation")
    from m15.analysis import collect
    lock = json.loads((root / "reports/m1_5/protocol_lock.json").read_text())
    plants, seeds, windows = collect(root, lock)
    assert set(plants.loc[plants.task.eq("day7") & plants.split.eq("test")].plant_uid) == set(
        pd.read_csv(root / "outputs/m1_5/calendar_windows.csv").query("forecast_horizon == 7 and split == 'test'").plant_uid)
    assert windows.loc[windows.task.eq("day7")].plant_uid.nunique() == 44
    assert windows.loc[windows.task.eq("offset8")].plant_uid.nunique() == 46
    assert set(seeds.loc[seeds.model.str.startswith(("mlp", "gru", "diff", "time")), "seed"]) == {42, 43, 44}


def test_cached_day_dataset_inputs_independent_of_targets():
    frames = make_frames()
    rows = calendar_windows(frames, {"g/p1": "train"}, [3], 6)
    class Store:
        features = np.arange(6 * 384, dtype=np.float32).reshape(6, 384)
        def row_indices(self, paths):
            return np.asarray([[int(path.removeprefix("rgb")) for path in history] for history in paths])
    before = DiagnosticDataset(rows, frames, Store(), task_family="days", condition="full")
    altered = rows.assign(target_leaf_count=500, target_projected_area_ratio=.99, target_frame_idx=9999,
                          target_time_delta_seconds=9999999, target_timestamp="2099-01-01")
    after = DiagnosticDataset(altered, frames, Store(), task_family="days", condition="full")
    np.testing.assert_array_equal(before.features, after.features)
    np.testing.assert_array_equal(before.times, after.times)
    assert not np.array_equal(before.targets, after.targets)


def test_authorized_inference_precision_patch_allows_replay_without_retraining():
    from m15.artifacts import save_once
    from training.common import file_hash
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/m1_5/training_source_fingerprints.json"
    if not path.exists() or not (root / "reports/m1_5/inference_precision_correction.json").exists():
        pytest.skip("Requires recorded precision correction")
    before = path.read_bytes()
    sources = json.loads(path.read_text())
    save_once(path, {source: file_hash(root / source) for source in sources})
    assert path.read_bytes() == before


def test_prediction_precision_matches_training_configuration(tmp_path):
    data = tiny_data("train", "p1", 2)
    val = tiny_data("val", "p2", 3)
    config = {**model_spec("mlp_full", PROTOCOL), "seed": 42, "max_epochs": 1, "batch_size": 8}
    train(data, val, config, tmp_path / "model", "cpu")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    predict(tmp_path / "model/checkpoint.pt", val, "cpu")
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32
    assert torch.are_deterministic_algorithms_enabled()


def test_actual_m15_checkpoint_scalers_fit_only_train_windows():
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/m1_5/training_complete.json"
    if not path.exists() or json.loads(path.read_text())["status"] != "complete":
        pytest.skip("Requires all actual M1.5 checkpoints")
    from datasets.m1_feature_dataset import M1Index
    from training.common import object_hash
    index = M1Index(root)
    calendar = pd.read_csv(root / "outputs/m1_5/calendar_windows.csv")
    expected = {}
    for directory in sorted((root / "outputs/m1_5/checkpoints").iterdir()):
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        key = config["task"], config["condition"]
        if key not in expected:
            source = index.rows if config["task_family"] == "offsets" else calendar
            rows = source.loc[source.forecast_horizon.eq(config["horizon"]) & source.split.eq("train")].reset_index(drop=True)
            expected[key] = (rows, time_conditions(rows, index.frames, config["task_family"], config["condition"]))
        rows, times = expected[key]
        targets = rows[["target_leaf_count", "target_projected_area_ratio"]].to_numpy(dtype=np.float32)
        for name, values in (("time_scaler", times), ("target_scaler", targets)):
            scaler = checkpoint[name]
            assert scaler["provenance"]["split"] == "train"
            assert scaler["provenance"]["window_ids_sha256"] == object_hash(sorted(rows.window_id.tolist()))
            assert scaler["provenance"]["plants"] == rows.plant_uid.nunique()
            np.testing.assert_array_equal(np.asarray(scaler["mean"], dtype=np.float32), values.mean(axis=0))
            np.testing.assert_array_equal(np.asarray(scaler["scale"], dtype=np.float32), np.maximum(values.std(axis=0), 1e-6))
        assert set(checkpoint["train_plants"]) == set(rows.plant_uid)
        assert set(checkpoint["val_plants"]).issubset(set(index.split["val_plants"]))
        assert not set(checkpoint["train_plants"]) & set(index.split["test_plants"])
