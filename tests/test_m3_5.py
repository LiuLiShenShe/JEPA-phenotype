import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.sugar4d_organ_dataset import LEAF_FIELDS, ORGAN_FEATURES, ORGAN_TARGETS, Sugar4DOrganDataset, build_event_windows, build_organ_windows, make_trajectories, time_vector
from m3.common import seed_everything, tensor_state_hash
from m35.data import classification_gate, geometry_descriptors
from m35.engine import balanced_weights, macro_mae, scaler_fit, transform
from m35.analysis import bootstrap_mean, classification_metrics, morphology_scores, paired_scores, research_decision
from models.geometry_residual_forecaster import CurrentPointNetRegressor, GeometryResidualForecaster
from models.organ_growth_baselines import OrganStateGRU, OrganStateMLP, linear_growth


def organ_fixture():
    days = [37, 40, 44, 47, 51, 58]
    leaves, organs, scans = [], [], []
    for index, day in enumerate(days):
        identifiers = [0] if index < 3 else ([0, 1] if index < 5 else [1])
        scans.append({"plant_uid": "plant_001", "plant_id": 1, "DAS": day, "split": "train", "visible_organ_count": len(identifiers)})
        for identifier in identifiers:
            leaves.append({"plant_uid": "plant_001", "plant_id": 1, "leaf_instance_id": identifier, "DAS": day,
                           **{field: day + identifier for field in LEAF_FIELDS}})
            organs.append({"plant_uid": "plant_001", "leaf_instance_id": identifier, "DAS": day, "point_count": 8,
                           "centroid_x_m": 1, "centroid_y_m": 2, "centroid_z_m": 0.1})
    return pd.DataFrame(leaves), pd.DataFrame(organs), pd.DataFrame(scans)


def test_exact_join_grid_and_missing_values_never_zero():
    leaves, organs, scans = organ_fixture()
    trajectories = make_trajectories(leaves, organs, scans)
    assert len(trajectories) == 12
    missing = trajectories.observation_status.eq("not_observed")
    assert trajectories.loc[missing, list(ORGAN_FEATURES)].isna().all().all()
    with pytest.raises(ValueError, match="identities disagree"):
        make_trajectories(leaves, organs.iloc[:-1], scans)
    with pytest.raises(ValueError, match="Duplicate"):
        make_trajectories(pd.concat([leaves, leaves.iloc[:1]]), organs, scans)


def test_actual_scan_horizons_visibility_and_label_mask():
    leaves, organs, scans = organ_fixture()
    trajectories = make_trajectories(leaves, organs, scans)
    windows = build_organ_windows(trajectories)
    sample = windows.loc[windows.current_DAS.eq(44) & windows.horizon.eq(2)].iloc[0]
    assert sample.target_DAS == 51 and sample.forecast_delta_days == 7
    missing = windows.loc[~windows.future_observed]
    assert len(missing) > 0
    assert missing[[f"target_{target}" for target in ORGAN_TARGETS]].isna().all().all()
    dataset = Sugar4DOrganDataset(windows)
    assert set(dataset[0]) == {"sequence", "time_conditions", "window_id", "plant_uid"}
    changed = windows.copy()
    changed.loc[:, "future_observed"] = False
    changed.loc[:, "target_leaf_area"] = 99999
    alternative = Sugar4DOrganDataset(changed)
    torch.testing.assert_close(dataset[0]["sequence"], alternative[0]["sequence"])
    torch.testing.assert_close(dataset[0]["time_conditions"], alternative[0]["time_conditions"])


def test_reappearance_does_not_compress_missing_actual_scans():
    leaves, organs, scans = organ_fixture()
    keep = ~(leaves.leaf_instance_id.eq(0) & leaves.DAS.eq(40))
    leaves = leaves.loc[keep]
    organs = organs.loc[~(organs.leaf_instance_id.eq(0) & organs.DAS.eq(40))]
    scans.loc[scans.DAS.eq(40), "visible_organ_count"] = 0
    trajectories = make_trajectories(leaves, organs, scans)
    windows = build_organ_windows(trajectories)
    assert not ((windows.leaf_instance_id == 0) & (windows.current_DAS == 44)).any()
    assert not ((windows.leaf_instance_id == 0) & (windows.current_DAS == 47)).any()
    assert ((windows.leaf_instance_id == 0) & (windows.current_DAS == 51)).any()


def test_new_event_includes_first_observed_between_current_and_target():
    leaves, organs, scans = organ_fixture()
    identifiers = ["plant_uid", "plant_id", "leaf_instance_id", "DAS"]
    leaf = {"plant_uid": "plant_001", "plant_id": 1, "leaf_instance_id": 2, "DAS": 47,
            **{field: 5 for field in LEAF_FIELDS}}
    organ = {"plant_uid": "plant_001", "leaf_instance_id": 2, "DAS": 47, "point_count": 8,
             "centroid_x_m": 0.1, "centroid_y_m": 0.2, "centroid_z_m": 0.1}
    leaves = pd.concat([leaves, pd.DataFrame([leaf])], ignore_index=True)
    organs = pd.concat([organs, pd.DataFrame([organ])], ignore_index=True)
    scans.loc[scans.DAS.eq(47), "visible_organ_count"] += 1
    events, states = build_event_windows(make_trajectories(leaves, organs, scans), scans)
    row = events.loc[events.current_DAS.eq(44) & events.horizon.eq(2)].iloc[0]
    assert row.target_DAS == 51 and row.new_count == 2


def test_first_observation_cumulative_events_not_net_count():
    leaves, organs, scans = organ_fixture()
    trajectories = make_trajectories(leaves, organs, scans)
    events, states = build_event_windows(trajectories, scans)
    assert events.loc[events.current_DAS.eq(44) & events.horizon.eq(1), "new_count"].iloc[0] == 1
    assert events.loc[events.current_DAS.eq(47) & events.horizon.eq(2), "new_count"].iloc[0] == 0
    assert events.loc[events.current_DAS.eq(47) & events.horizon.eq(2), "event"].iloc[0] == 0
    assert not any("leaf_instance_id" in name or "first" in name for name in events.columns)
    assert states.loc[states.DAS.eq(58), "seen_count"].iloc[0] == 2


def test_illegal_time_and_cross_split_rejected():
    with pytest.raises(ValueError, match="strictly future"):
        time_vector([37, 40, 44], 44)
    with pytest.raises(ValueError, match="Future first"):
        time_vector([37, 40, 44], 51, 47)
    leaves, organs, scans = organ_fixture()
    scans.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="split boundaries"):
        make_trajectories(leaves, organs, scans)


def test_linear_actual_DAS_and_not_nominal_step():
    prediction = linear_growth(np.array([[2, 1, 10], [5, 2.5, 25], [9, 4.5, 45]]), [37, 40, 44], 51)
    np.testing.assert_allclose(prediction, [16, 8, 80])


def test_geometry_is_label_free_in_metres_and_degenerate_safe():
    cloud = np.array([[0, 0, 0.1], [0.1, 0, 0.3], [0, 0.2, 0.2], [0.1, 0.2, 0.25]])
    geometry = geometry_descriptors(cloud)
    assert geometry["z_range_cm"] == pytest.approx(20)
    assert geometry["hull_diameter_cm"] == pytest.approx(np.sqrt(5) * 10)
    assert geometry["hull_area_cm2"] == pytest.approx(200)
    assert np.isfinite(list(geometry_descriptors(np.column_stack([np.zeros(4), np.zeros(4), np.arange(4)])).values())).all()


def test_plant_balancing_not_leaf_window_weighted():
    frame = pd.DataFrame({"plant_uid": ["one"] * 100 + ["two"], "window_id": [str(index) for index in range(101)], "horizon": 1})
    weights = balanced_weights(frame)
    assert weights[:100].sum() == pytest.approx(weights[-1])
    score = macro_mae(frame, np.array([1] * 100 + [10]), np.zeros(101))
    assert score[0] == pytest.approx(5.5)
    state = scaler_fit([[0], [2]], [1, 1])
    assert state["fit_split"] == "train"
    state["fit_split"] = "test"
    with pytest.raises(ValueError, match="train-only"):
        transform([[0]], state)


def test_single_output_ridge_prediction_shape():
    from m35.engine import ridge_predict
    fit = {"input_scaler": {"mean": [0, 0], "scale": [1, 1], "fit_split": "train"},
           "target_scaler": {"mean": [0], "scale": [1], "fit_split": "train"}, "coef": [1, 2], "intercept": 0}
    prediction = ridge_predict(np.array([[1, 2], [3, 4]]), fit)
    assert prediction.shape == (2, 1)
    np.testing.assert_allclose(prediction[:, 0], [5, 11])


def test_single_output_ridge_validation_does_not_broadcast_records():
    from m35.engine import save_ridge_fit, ridge_predict
    rows = pd.DataFrame({"plant_uid": ["train"] * 4 + ["val"] * 4, "window_id": [str(index) for index in range(8)],
                         "split": ["train"] * 4 + ["val"] * 4, "horizon": 1})
    inputs = np.arange(8, dtype=float).reshape(-1, 1)
    targets = (2 * inputs + 1)
    fit = save_ridge_fit(rows.iloc[:4], rows.iloc[4:], inputs, targets, {"ridge_alphas": [0.0001, 1, 100]})
    prediction = ridge_predict(inputs[4:], fit)
    score = float(np.mean(macro_mae(rows.iloc[4:], prediction, targets[4:]) / np.asarray(fit["target_scaler"]["scale"])))
    assert fit["validation_score"] == pytest.approx(score, abs=1e-6)
    assert fit["validation_score"] < 0.001


def test_event_train_save_path_and_no_test_inference(tmp_path, monkeypatch):
    from m35.engine import train_tabular
    rng = np.random.default_rng(17)
    rows = []
    for split in ("train", "val", "test"):
        for plant in range(2):
            for index in range(4):
                rows.append({"window_id": f"{split}{plant}{index}", "plant_uid": f"{split}{plant}", "split": split,
                             "horizon": 1, "event": index % 2, "new_count": index % 3, "target_DAS": 47 + index,
                             "current_DAS": 44, "forecast_delta_days": 3 + index})
    rows = pd.DataFrame(rows)
    data = {"rows": rows, "sequence": rng.normal(size=(len(rows), 3, 8)).astype(np.float32),
            "times": rng.normal(size=(len(rows), 7)).astype(np.float32), "label": rows[["new_count"]].to_numpy(dtype=np.float32),
            "truth": rows[["new_count"]].to_numpy(dtype=np.float32), "baseline": np.zeros((len(rows), 1)),
            "visibility": rows.event.to_numpy(dtype=np.float32), "valid": np.ones(len(rows), dtype=bool), "targets": ("new_count",)}
    protocol = {"tabular_hidden_dim": 8, "dropout": 0.1, "learning_rate": 0.001, "weight_decay": 0.0001,
                "epochs": 2, "batch_size": 4, "gradient_clip": 5.0}
    monkeypatch.setattr("m35.engine.source_fingerprints", lambda root: {})
    train_tabular(tmp_path, "scratch", data, "event", ["E2-MLP"], protocol, torch.device("cpu"), [17])
    destination = tmp_path / "outputs/m3_5/scratch/event"
    record = json.loads((destination / "E2-MLP_seed17_training.json").read_text())
    assert isinstance(record["validation_score"], float)
    assert record["optimizer_steps"] == 4 and record["epochs"] == 2
    predictions = pd.read_csv(destination / "E2-MLP_seed17_train_val.csv")
    assert set(predictions.split) == {"train", "val"}
    assert predictions.pred_new_count.ge(0).all()
    assert predictions.pred_event.between(0, 1).all()


def test_real_repeat_exact_initialization_capacity_and_input_control():
    seed_everything(17)
    real = OrganStateGRU(hidden_dim=16).eval()
    seed_everything(17)
    repeated = OrganStateGRU(hidden_dim=16, repeated_latest=True).eval()
    assert tensor_state_hash(real.state_dict()) == tensor_state_hash(repeated.state_dict())
    assert sum(value.numel() for value in real.parameters()) == sum(value.numel() for value in repeated.parameters())
    sequence, times = torch.randn(4, 3, 5), torch.randn(4, 8)
    changed = sequence.clone()
    changed[:, :-1] += 5
    torch.testing.assert_close(repeated(sequence, times), repeated(changed, times))
    assert not torch.equal(real(sequence, times), real(changed, times))
    mlp = OrganStateMLP(hidden_dim=16).eval()
    torch.testing.assert_close(mlp(sequence, times), mlp(changed, times))


def test_small_model_shapes_and_state_reload(tmp_path):
    for model, inputs in ((CurrentPointNetRegressor(), (torch.randn(2, 16, 3), torch.zeros(2, 27))),
                          (GeometryResidualForecaster(27), (torch.randn(2, 3, 27), torch.randn(2, 7)))):
        model.eval()
        prediction = model(*inputs)
        assert prediction.shape == (2, 3)
        path = tmp_path / "checkpoint.pt"
        torch.save(model.state_dict(), path)
        model.load_state_dict(torch.load(path, weights_only=True))
        torch.testing.assert_close(prediction, model(*inputs))


def test_class_gate_not_influenced_by_test():
    rules = {"train_class_samples": 3, "validation_class_samples": 2, "train_class_plants": 2, "validation_class_plants": 2, "minimum_class_fraction": 0.01}
    rows = []
    for split in ("train", "val", "test"):
        for plant in range(3):
            for label in (0, 1):
                rows.append({"plant_uid": f"{split}{plant}", "split": split, "event": label})
    frame = pd.DataFrame(rows)
    before = classification_gate(frame, "event", rules)
    frame.loc[frame.split.eq("test"), "event"] = 0
    assert before == classification_gate(frame, "event", rules)
    frame.loc[frame.split.eq("val"), "event"] = 1
    assert not classification_gate(frame, "event", rules)["feasible"]


def test_real_m35_data_and_protocol_contract():
    root = Path(__file__).resolve().parents[1]
    data = root / "outputs/m3_5/data"
    if not (data / "organ_windows.csv").exists():
        pytest.skip("Run M3.5 feasibility preparation first")
    grid = pd.read_csv(data / "organ_trajectories.csv")
    windows = pd.read_csv(data / "organ_windows.csv")
    events = pd.read_csv(data / "event_windows.csv")
    assert grid.plant_uid.nunique() == 48 and len(grid) == 10800
    assert grid.observation_status.eq("observed").sum() == 6778
    assert len(windows) == 9386 and windows.future_observed.sum() == 9016
    assert len(events) == 1200
    assert windows.groupby("plant_uid").split.nunique().max() == 1
    Sugar4DOrganDataset(windows)
    feasibility = json.loads((root / "reports/m3_5/feasibility.json").read_text())
    assert feasibility["association"]["mismatches"] == 0
    assert feasibility["reappearing_tracks"] == 0
    assert feasibility["measurement_agreement"]["leaf_area"]["manual_validation_available"] is False


def test_plant_intervals_units_and_delta_identity():
    protocol = {"bootstrap_seed": 17, "bootstrap_replicates": 1000}
    records = []
    for plant, windows, error in (("one", 100, 1), ("two", 1, 10)):
        for index in range(windows):
            records.append({"partition": "official", "split": "val", "task": "organ", "model": "C3-GRU", "seed": 17,
                            "horizon": 2, "plant_uid": plant, "window_id": f"{plant}{index}", "pred_leaf_total_length": 20 + error,
                            "true_leaf_total_length": 20, "pred_delta_leaf_total_length": 5 + error, "true_delta_leaf_total_length": 5})
    summary, plants, seeds = morphology_scores(pd.DataFrame(records), ["leaf_total_length"], protocol)
    assert summary.iloc[0].plant_macro_mae == pytest.approx(5.5)
    assert summary.iloc[0].unit == "cm"
    assert summary.iloc[0].delta_phenotype_mae == pytest.approx(summary.iloc[0].plant_macro_mae)
    lower, upper = bootstrap_mean([1, 10], protocol)
    assert 1 <= lower <= upper <= 10


def test_classification_units_plant_weights_and_minority_visibility():
    frame = pd.DataFrame({"plant_uid": ["one"] * 100 + ["two"] * 2, "window_id": [str(index) for index in range(102)],
                          "horizon": 1, "event": [0] * 100 + [1, 0], "pred_event": [0.1] * 100 + [0.9, 0.1]})
    metrics, plants = classification_metrics(frame, "event", "pred_event", 0.5, {"bootstrap_seed": 1, "bootstrap_replicates": 100})
    assert metrics["plants"] == 2
    assert metrics["plant_macro_recall"] == pytest.approx(0.5)
    assert metrics["plant_balanced_pr_auc"] == pytest.approx(1)
    assert metrics["pr_auc_eligible_plants"] == 1


def test_seed_average_intervals_do_not_treat_seeds_as_plants():
    from m35.analysis import summarize_classification
    rows = []
    for seed in (17, 42, 93):
        for plant in ("one", "two"):
            for label in (0, 1):
                rows.append({"partition": "official", "split": "val", "model": "E3-GRU", "seed": seed, "horizon": 1,
                             "plant_uid": plant, "window_id": f"{plant}{label}", "event": label, "pred_event": 0.1 + 0.8 * label,
                             "pred_new_count": label, "new_count": label})
    frame = pd.DataFrame(rows)
    thresholds = {f"official|E3-GRU|{seed}|1": {"threshold": 0.5} for seed in (17, 42, 93)}
    summary, by_seed, plants = summarize_classification(frame, "pred_event", "event", thresholds,
                                                       {"bootstrap_seed": 1, "bootstrap_replicates": 100})
    assert summary.iloc[0].plants == 2 and summary.iloc[0].seeds == 3
    assert summary.iloc[0].plant_macro_f1 == 1
    assert summary.iloc[0].f1_ci95_low == 1 and summary.iloc[0].f1_ci95_high == 1
    assert len(plants) == 6 and len(by_seed) == 3


def test_PR_AUC_bootstrap_resamples_whole_plants_preserves_ties():
    from m35.analysis import cluster_pr_auc_interval
    frame = pd.DataFrame({"seed": [17] * 4, "plant_uid": ["one", "one", "two", "two"], "event": [0, 1, 0, 1], "pred_event": [0.1, 0.9, 0.1, 0.9]})
    low, high, draws = cluster_pr_auc_interval(frame, "event", "pred_event", {"bootstrap_seed": 17, "bootstrap_replicates": 100})
    assert low == 1 and high == 1 and draws == 100
    frame["pred_event"] = 0.5
    low, high, draws = cluster_pr_auc_interval(frame, "event", "pred_event", {"bootstrap_seed": 17, "bootstrap_replicates": 100})
    assert low == 0.5 and high == 0.5


def test_real_m35_current_and_future_inference_inputs_independent_of_targets():
    from m35.engine import current_arrays, tabular_arrays
    root = Path(__file__).resolve().parents[1]
    if not (root / "outputs/m3_5/official/current/ridge_fits.json").exists():
        pytest.skip("Run current diagnostics first")
    assignment = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]["official"]
    data = tabular_arrays(root, "organ", assignment)
    assert data["sequence"].shape == (9386, 3, 5)
    assert data["times"].shape == (9386, 8)
    assert not np.isfinite(data["label"][~data["valid"]]).any()
    assert np.isfinite(data["sequence"]).all()
    events = tabular_arrays(root, "event", assignment)
    assert events["sequence"].shape == (1200, 3, 8)
    assert len(events["times"][0]) == 7
    fits = json.loads((root / "outputs/m3_5/official/current/ridge_fits.json").read_text())
    geometry = tabular_arrays(root, "geometry_residual", assignment, fits)
    assert geometry["sequence"].shape == (1200, 3, 27)
    assert np.isfinite(geometry["baseline"]).all()
    test_mask = geometry["rows"].split.eq("test").to_numpy()
    assert (geometry["baseline"][test_mask] == 0).all()
    final_geometry = tabular_arrays(root, "geometry_residual", assignment, fits, include_test=True)
    assert (final_geometry["baseline"][test_mask] > 0).all()
    for target in ("plant_height", "plant_width", "visible_organ_count"):
        assert target not in __import__('m35.data', fromlist=['GEOMETRY_FEATURES']).GEOMETRY_FEATURES


def test_research_branch_uses_validation_only_and_primary_rule():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/m3_5/protocol.json").read_text())
    current = pd.DataFrame([{"partition": "official", "split": "val", "model": model, "target": target, "plant_macro_mae": value}
                            for target in ("plant_height", "plant_width") for model, value in (("A1-PointNet", 2), ("A2-Geometry-Ridge", 0.3))])
    decision = research_decision(current, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), protocol)
    assert decision["recommendation"] == "REPRESENTATION-FIRST"
    assert not decision["reliable_practical_organ_temporal_signal"]
    current.loc[current.model.eq("A1-PointNet"), "plant_macro_mae"] = 0.8
    alternative = research_decision(current, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), protocol)
    assert alternative["recommendation"] == "PIVOT"


def test_temporal_branch_requires_both_controls_all_seeds_and_folds():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / "configs/m3_5/protocol.json").read_text())
    current = pd.DataFrame([{"partition": "official", "split": "val", "model": model, "target": target, "plant_macro_mae": value}
                            for target in ("plant_height", "plant_width") for model, value in (("A1-PointNet", 0.8), ("A2-Geometry-Ridge", 0.5))])
    pairs = []
    for partition in protocol["partitions"]:
        for control in ("C2-MLP", "C4-Repeated-GRU"):
            pairs.append({"partition": partition, "split": "val", "target": "leaf_total_length", "horizon": 2,
                          "left": "C3-GRU", "right": control, "left_minus_right_mae": -0.6,
                          "improvement_pct": 20, "ci95_low": -0.9, "ci95_high": -0.3})
    pairs = pd.DataFrame(pairs)
    seeds = pd.DataFrame([{"partition": "official", "split": "val", "target": "leaf_total_length", "horizon": 2, "seed": seed,
                          "model": model, "plant_macro_mae": value} for seed in (17, 42, 93)
                         for model, value in (("C3-GRU", 2.4), ("C2-MLP", 3), ("C4-Repeated-GRU", 3))])
    result = research_decision(current, pd.DataFrame(), pairs, seeds, protocol)
    assert result["recommendation"] == "ORGAN-TEMPORAL-JEPA"
    deterministic = pd.DataFrame([{"partition": "official", "split": "val", "target": "leaf_total_length", "horizon": 2,
                                   "seed": -1, "model": "C0-Persistence", "plant_macro_mae": 4}])
    mixed = pd.concat([seeds, deterministic], ignore_index=True)
    assert research_decision(current, pd.DataFrame(), pairs, mixed, protocol)["recommendation"] == "ORGAN-TEMPORAL-JEPA"
    seeds.loc[seeds.seed.eq(93) & seeds.model.eq("C3-GRU"), "plant_macro_mae"] = 3.1
    assert research_decision(current, pd.DataFrame(), pairs, seeds, protocol)["recommendation"] == "PIVOT"
    seeds.loc[seeds.model.eq("C3-GRU"), "plant_macro_mae"] = 2.4
    pairs.loc[pairs.partition.eq("official") & pairs.right.eq("C4-Repeated-GRU"), "ci95_high"] = 0.01
    assert research_decision(current, pd.DataFrame(), pairs, seeds, protocol)["recommendation"] == "PIVOT"


def test_real_history_and_future_labels_match_actual_scan_grid():
    root = Path(__file__).resolve().parents[1]
    grid_path = root / "outputs/m3_5/data/organ_trajectories.csv"
    if not grid_path.exists():
        pytest.skip("Run M3.5 preparation first")
    grid = pd.read_csv(grid_path).set_index(["plant_uid", "leaf_instance_id", "DAS"])
    windows = pd.read_csv(root / "outputs/m3_5/data/organ_windows.csv")
    for row in windows.itertuples(index=False):
        days = json.loads(row.history_DAS)
        history = grid.loc[[(row.plant_uid, row.leaf_instance_id, day) for day in days]]
        assert history.observation_status.eq("observed").all()
        np.testing.assert_allclose(history[list(ORGAN_FEATURES)].to_numpy(), json.loads(row.history_values), rtol=1e-12)
        target = grid.loc[(row.plant_uid, row.leaf_instance_id, row.target_DAS)]
        assert row.future_observed == (target.observation_status == "observed")
        if row.future_observed:
            assert row.target_leaf_area == pytest.approx(target.leaf_area)
        else:
            assert np.isnan(row.target_leaf_area)


def test_completed_runs_selection_and_equal_control_budget():
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "reports/m3_5/selection_lock.json"
    if not lock_path.exists():
        pytest.skip("Run all M3.5 models and lock selection first")
    lock = json.loads(lock_path.read_text())
    assert lock["locked_before_test_inference"] and not lock["test_used_for_selection"]
    assert lock["neural_runs"] == 180
    table = pd.read_csv(root / "reports/m3_5/training_resources.csv")
    assert len(table) == 180 and table.epochs.eq(50).all()
    for partition in ("official", "fold0", "fold1", "fold2", "fold3"):
        for seed in (17, 42, 93):
            for task, real, repeat in (("organ", "C3-GRU", "C4-Repeated-GRU"), ("event", "E3-GRU", "E4-Repeated-GRU")):
                subset = table.loc[table.partition.eq(partition) & table.seed.eq(seed) & table.task.eq(task)].set_index("model")
                for field in ("initial_tensor_sha256", "parameters", "optimizer_steps", "epochs"):
                    assert subset.loc[real, field] == subset.loc[repeat, field]


def test_real_checkpoint_scalers_only_fit_training_plants():
    from m35.engine import tabular_arrays
    root = Path(__file__).resolve().parents[1]
    checkpoint_path = root / "outputs/m3_5/official/organ/C3-GRU_seed17.pt"
    if not checkpoint_path.exists():
        pytest.skip("Run organ GRU training first")
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    assignment = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]["official"]
    data = tabular_arrays(root, "organ", assignment)
    mask = data["rows"].split.eq("train").to_numpy()
    indices = np.flatnonzero(mask)
    train_rows = data["rows"].loc[mask]
    weights = balanced_weights(train_rows)
    expected = scaler_fit(data["sequence"][indices].reshape(-1, 5), np.repeat(weights, 3))
    np.testing.assert_allclose(checkpoint["sequence_scaler"]["mean"], expected["mean"], rtol=1e-12)
    np.testing.assert_allclose(checkpoint["sequence_scaler"]["scale"], expected["scale"], rtol=1e-12)
    valid = indices[data["valid"][indices]]
    labels = scaler_fit(data["label"][valid], balanced_weights(data["rows"].iloc[valid]))
    np.testing.assert_allclose(checkpoint["label_scaler"]["scale"], labels["scale"], rtol=1e-12)


def test_final_predictions_missingness_oof_units_and_replay():
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs/m3_5/test_predictions.csv"
    if not path.exists():
        pytest.skip("Run locked final evaluation first")
    predictions = pd.read_csv(path, dtype={"future_observed": "boolean"}, low_memory=False)
    organ = predictions.loc[predictions.task.eq("organ")]
    missing = organ.loc[~organ.future_observed.astype(bool)]
    official_missing = missing.loc[missing.partition.eq("official")]
    assert official_missing.window_id.nunique() == 96
    assert missing.true_leaf_area.isna().all()
    assert missing.pred_visibility.notna().all()
    oof = predictions.loc[predictions.partition.str.startswith("fold")]
    assert oof.groupby(["task", "model", "seed", "plant_uid"]).partition.nunique().max() == 1
    repeat = json.loads((root / "reports/m3_5/inference_repeatability.json").read_text())
    assert repeat["all_identical"] and len(repeat["runs"]) == 180


def test_legacy_protection_snapshot_unchanged():
    from m3.common import file_hash
    root = Path(__file__).resolve().parents[1]
    path = root / "reports/m3_5/initial_protection.json"
    if not path.exists():
        pytest.skip("Run M3.5 protection first")
    snapshot = json.loads(path.read_text())
    for name, digest in snapshot["files"].items():
        if name.endswith((".py", ".json", ".md", ".csv")):
            assert file_hash(root / name) == digest
