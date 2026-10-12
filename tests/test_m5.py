from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.plant_ssl_dataset import BalancedPlantSSLDataset
from m4.common import sha256, write_json
from m5.common import load_protocol
from m5.diagnostics import degradation_stop, feature_statistics, representation_diagnostics
from m5.evaluation import decide, paired_comparison
from m5.pretrain import regularization
from m5.probe import pair_metrics, token_leaf_labels


ROOT = Path(__file__).resolve().parents[1]


def test_feature_statistics_constant_features_are_collapsed():
    values = torch.ones(32, 8)
    statistics = feature_statistics(values)
    assert statistics["effective_rank"] == 0 and statistics["variance_mean"] == 0
    diagnostic = representation_diagnostics(values[:, None].repeat(1, 4, 1), values, np.repeat(np.arange(8), 4))
    assert diagnostic["hard_collapse"]


def test_output_json_serializes_numpy_flags_without_changing_values(tmp_path):
    from m5.artifacts import write_json as write_output

    destination = tmp_path / "result.json"
    write_output(destination, {"complete": np.bool_(True), "rank": np.float64(2.5), "steps": np.int64(500),
                               "nested": {"passed": np.bool_(False)}, "values": np.array([1., 2.])})
    record = json.loads(destination.read_text())
    assert record == {"complete": True, "rank": 2.5, "steps": 500, "nested": {"passed": False}, "values": [1., 2.]}
    with pytest.raises(ValueError):
        write_output(destination, {"invalid": np.float64(np.nan)})


def test_figure_bounds_ignore_dormant_ticks_but_detect_actual_outside_text(tmp_path):
    from m5.figures import export, plt

    records = []
    figure, axis = plt.subplots(figsize=(4, 3), constrained_layout=True)
    axis.plot([0, 1], [.2, .8])
    axis.set_ylim(.1, .9)
    export(figure, tmp_path, "inside", pd.DataFrame({"value": [.2, .8]}), {}, records)
    assert records[-1]["status"] == "PASS"
    figure, axis = plt.subplots(figsize=(4, 3), constrained_layout=True)
    axis.plot([0, 1], [.2, .8])
    figure.text(1.5, .5, "outside")
    export(figure, tmp_path, "outside", pd.DataFrame({"value": [.2, .8]}), {}, records)
    assert records[-1]["status"] == "REVIEW" and "outside" in records[-1]["text_outside_canvas"]


def test_high_cosine_is_not_equivalent_to_collapse():
    generator = torch.Generator().manual_seed(3)
    features = torch.randn(64, 12, generator=generator) + 100
    tokens = torch.randn(64, 8, 12, generator=generator) + 100
    diagnostic = representation_diagnostics(tokens, features, np.repeat(np.arange(8), 8))
    assert not diagnostic["hard_collapse"]
    assert diagnostic["pooled"]["effective_rank"] > 8
    pairs = diagnostic["similarities"]["different_plants"]
    assert pairs["uncentred"]["mean"] > .99 and abs(pairs["centred"]["mean"]) < .15


def test_balanced_sampler_does_not_let_large_species_dominate(tmp_path):
    records = []
    for index in range(101):
        filename = f"{index}.npz"
        np.savez(tmp_path / filename, points=np.zeros((8, 3), dtype=np.float32), center_indices=np.arange(4), neighbors=np.zeros((4, 2), dtype=np.int64), semantic=np.ones(8))
        records.append({"source": "Crops3D", "species": "rare" if index == 0 else "large", "split": "train", "cache_file": filename, "geometry_sha256": str(index)})
    sugar = [{"points": torch.zeros(8, 3), "center_indices": torch.arange(4), "neighbors": torch.zeros(4, 2, dtype=torch.long)}] * 4
    dataset = BalancedPlantSSLDataset(sugar, pd.DataFrame(records), tmp_path, {"Sugar4D": .5, "Crops3D": .5})
    sampled = dataset.sample_indices(10000, torch.Generator().manual_seed(5))
    assert 4500 < sum(index < 4 for index in sampled) < 5500
    assert 2100 < dataset.exposures["Crops3D"]["rare"] < 2900
    assert set(dataset[4]) == {"points", "center_indices", "neighbors"}


def test_external_duplicate_and_nontraining_scan_are_rejected(tmp_path):
    rows = pd.DataFrame([{"source": "x", "species": "x", "split": "test", "cache_file": "x", "geometry_sha256": "same"}])
    with pytest.raises(ValueError, match="training"):
        BalancedPlantSSLDataset([], rows, tmp_path, {})
    rows["split"] = "train"
    with pytest.raises(ValueError, match="Duplicate"):
        BalancedPlantSSLDataset([], pd.concat([rows, rows]), tmp_path, {})


def test_regularization_is_differentiable_and_penalizes_flat_features():
    specification = {"standard_deviation_floor": .1, "epsilon": .0001, "variance_weight": 1., "covariance_weight": .01}
    constant = torch.ones(24, 8, requires_grad=True)
    loss, variance, covariance = regularization(constant, specification)
    assert variance > 0 and covariance == 0
    loss.backward()
    assert constant.grad is not None and torch.isfinite(constant.grad).all()
    diverse = torch.randn(500, 8) * .3
    assert regularization(diverse, specification)[0] < loss


def test_probe_purity_excludes_mixed_nonleaf_tokens():
    neighbors = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    semantic = np.array([2] * 5 + [0, 2, 2, 2, 2])
    instance = np.array([1] * 5 + [-1, 1, 2, 2, 3])
    labels = token_leaf_labels(neighbors, semantic, instance)
    assert labels.tolist() == [1, -1]


def test_leaf_probe_reports_separability_without_instance_iou():
    labels = np.repeat(np.arange(3), 4)
    features = np.eye(3)[labels].astype(np.float32)
    centers = np.random.default_rng(9).normal(size=(12, 3))
    record = pair_metrics(features, centers, labels)
    assert record["centred"]["pair_auc"] == 1
    assert record["centred"]["distance_matched_margin"] > 0
    assert "miou" not in record and "pq" not in record


def test_plant_pair_bootstrap_averages_seeds_before_ci():
    rows = []
    for seed in [17, 42, 93]:
        for plant in ["a", "b", "c"]:
            for condition, error in [("P0", 2.), ("P3", 1.5)]:
                rows.append({"seed": seed, "plant_uid": plant, "condition": condition, "mae": error,
                             "partition": "official", "label_fraction": .5, "task": "count"})
    record = paired_comparison(pd.DataFrame(rows), "P3", "P0", "official", .5)
    assert record["independent_plants"] == 3 and record["seeds"] == 3
    assert record["mean_delta"] == -.5 and record["ci95_high"] == -.5


def test_decision_cannot_switch_to_better_auxiliary_label_fraction():
    protocol = json.loads((ROOT / "configs/m5/protocol.json").read_text())
    rows = []
    for condition in ("P3", "P4"):
        for right in ("P0", "P2", "S500"):
            for partition in protocol["partitions"]:
                for fraction in (.25, .5, 1.):
                    rows.append({"left": condition, "right": right, "partition": partition, "label_fraction": fraction, "task": "count",
                                 "absolute_improvement": .01 if fraction == .5 else 1., "relative_improvement": .01 if fraction == .5 else .5,
                                 "ci95_low": -.03, "ci95_high": .01 if fraction == .5 else -.3, "seeds": 3, "improving_seeds": 2})
    decision = decide(protocol, pd.DataFrame(rows), True, [])
    assert decision["decision"] != "JEPA-CONTINUE"


def decision_fixture():
    protocol = json.loads((ROOT / "configs/m5/protocol.json").read_text())
    rows = []
    for condition in ("P3", "P4"):
        for right in ("P0", "P2", "S500"):
            for partition in protocol["partitions"]:
                rows.append({"left": condition, "right": right, "partition": partition,
                             "label_fraction": .5, "task": "count", "absolute_improvement": .4,
                             "relative_improvement": .2, "ci95_low": -.5, "ci95_high": -.3,
                             "seeds": 3, "improving_seeds": 3})
        rows.append({"left": condition, "right": "P0", "partition": "official",
                     "label_fraction": 1., "task": "segmentation", "mean_delta": -.01,
                     "absolute_improvement": -.01, "ci95_high": .01})
    risks = [{"partition": "official", "condition": condition, "seed": seed,
              "pooled_rank": 10., "token_rank": 10., "hard_collapse": False}
             for condition in ("P3", "P4") for seed in protocol["seeds"]]
    return protocol, pd.DataFrame(rows), risks


def test_decision_requires_complete_matrix_and_representation_gate():
    protocol, comparisons, risks = decision_fixture()
    assert decide(protocol, comparisons, True, risks)["decision"] == "JEPA-CONTINUE"
    assert decide(protocol, comparisons, False, risks)["decision"] != "JEPA-CONTINUE"
    for record in risks:
        record["pooled_rank"] = 1.5
    result = decide(protocol, comparisons, True, risks)
    assert result["decision"] != "JEPA-CONTINUE"
    assert not any(candidate["representation_gate"] for candidate in result["candidate_checks"])


def test_decision_semantic_negative_transfer_veto_is_not_auxiliary_success():
    protocol, comparisons, risks = decision_fixture()
    selection = comparisons.task.eq("segmentation")
    comparisons.loc[selection, "mean_delta"] = -.1
    comparisons.loc[selection, "ci95_high"] = -.05
    result = decide(protocol, comparisons, True, risks)
    assert result["decision"] == "JEPA-STOP"
    assert all(candidate["semantic_negative_transfer"] for candidate in result["candidate_checks"])


def test_m5a_stop_is_training_only_and_requires_joint_evidence():
    initial = {"hard_collapse": False, "pooled": {"effective_rank": 20., "variance_mean": 1.}}
    previous = {"hard_collapse": False, "pooled": {"effective_rank": 10., "variance_mean": .4}}
    current = {"hard_collapse": False, "pooled": {"effective_rank": 3., "variance_mean": .1}}
    assert degradation_stop(current, previous, initial, 500)
    assert not degradation_stop(current, previous, initial, 120)


def test_protocol_rejects_post_lock_metric_change(tmp_path, monkeypatch):
    import m5.common as common

    monkeypatch.setattr(common, "m4_protocol", lambda root: (None, None))
    path = tmp_path / "configs/m5/protocol.json"
    write_json(path, {"primary": "50 percent plant MAE"})
    write_json(tmp_path / "reports/m5/protocol_lock.json", {"protocol_sha256": sha256(path), "inputs": {}})
    assert load_protocol(tmp_path)[0]["primary"] == "50 percent plant MAE"
    write_json(path, {"primary": "best label ratio"})
    with pytest.raises(ValueError, match="Locked"):
        load_protocol(tmp_path)


def test_tiny_m5_ssl_downstream_and_checkpoint_restore(tmp_path):
    from datasets.sugar4d_ssl_dataset import Sugar4DSSLDataset, partition_rows
    from m5.downstream import run_downstream, verify_restore
    from m5.pretrain import train
    from models.point_jepa_encoder import patch_indices

    old_protocol = json.loads((ROOT / "configs/m4/protocol.json").read_text())
    old_protocol["model"].update({"num_groups": 8, "group_size": 8, "encoder_dim": 24, "encoder_depth": 2,
                                  "encoder_heads": 3, "predictor_dim": 12, "predictor_depth": 1, "predictor_heads": 3,
                                  "dropout": 0., "attention_dropout": 0., "drop_path_rate": 0.})
    old_protocol["partitions"], old_protocol["seeds"] = ["official"], [17]
    write_json(tmp_path / "configs/m4/protocol.json", old_protocol)
    write_json(tmp_path / "reports/m4/protocol_lock.json", {"protocol_sha256": sha256(tmp_path / "configs/m4/protocol.json"), "inputs": {}})
    protocol = json.loads((ROOT / "configs/m5/protocol.json").read_text())
    protocol["model"] = old_protocol["model"]
    protocol["partitions"], protocol["seeds"] = ["official"], [17]
    protocol["conditions"] = ["S500"]
    protocol["downstream"].update({"epochs": 2, "batch_size": 8})
    write_json(tmp_path / "configs/m5/protocol.json", protocol)
    lock_hash = sha256(tmp_path / "configs/m5/protocol.json")
    write_json(tmp_path / "reports/m5/protocol_lock.json", {"protocol_sha256": lock_hash, "inputs": {},
               "label_budgets": {"official": {"1.0": ["a", "b"]}}})
    assignments = {"a": "train", "b": "train", "c": "val", "d": "test"}
    write_json(tmp_path / "reports/m3/plant_splits.json", {"assignments": {"official": assignments}})
    generator = np.random.default_rng(3)
    metadata, geometry = [], []
    cache, labels = tmp_path / "outputs/m4/data/xyz", tmp_path / "outputs/m4/data/labels"
    cache.mkdir(parents=True)
    labels.mkdir(parents=True)
    for plant, split in assignments.items():
        for scan in range(16):
            name = f"{plant}_{scan}"
            points = generator.normal(size=(32, 3)).astype(np.float32)
            selected, neighbors = patch_indices(torch.from_numpy(points)[None], 8, 8)
            np.savez(cache / f"{name}.npz", points=points, center_indices=selected[0].numpy(), neighbors=neighbors[0].numpy(),
                     center_m=np.zeros(3, dtype=np.float32), radius_m=np.array(1., dtype=np.float32), extent_m=np.ones(3, dtype=np.float32))
            np.savez(labels / f"{name}.npz", semantic=generator.integers(0, 3, size=32, dtype=np.int64))
            metadata.append({"plant_uid": plant, "DAS": scan, "pointcloud_path": name + ".ply", "split": split, "visible_organ_count": 4 + scan // 4})
            geometry.append({"plant_uid": plant, "DAS": scan, "extent": scan + 1.})
    (tmp_path / "data/processed/sugar4d").mkdir(parents=True)
    (tmp_path / "outputs/m3_5/data").mkdir(parents=True)
    pd.DataFrame(metadata).to_csv(tmp_path / "data/processed/sugar4d/plant_metadata.csv", index=False)
    pd.DataFrame(geometry).to_csv(tmp_path / "outputs/m3_5/data/geometry.csv", index=False)
    rows = partition_rows(tmp_path, "official", "train")
    dataset = Sugar4DSSLDataset(rows, cache)
    specification = {"learning_rate": .001, "weight_decay": .05, "maximum_steps": 2, "warmup_steps": 1,
                     "batch_size": 8, "checkpoints": [0, 1, 2]}
    record = train(tmp_path, "official", 17, "S500", specification, dataset, dataset, torch.device("cpu"), lock_hash)
    assert record["optimizer_steps"] == 2 and not record["test_used"]
    for task in ("count", "segmentation"):
        record = run_downstream(tmp_path, "official", 17, "S500", 1., task, torch.device("cpu"))
        folder = tmp_path / "outputs/m5/downstream" / record["id"]
        assert verify_restore(tmp_path, folder, torch.device("cpu"))["status"] == "PASS"
        state = torch.load(folder / "last.pt", weights_only=False)
        assert state["model"] and state["optimizer"] and state["rng"] and state["loader_rng"] is not None


def test_ssl_full_state_resume_repeats_exact_next_step(tmp_path):
    from m4.common import restore_rng, seed_all, state_hash
    from m5.pretrain import save_checkpoint
    from models.point_jepa_predictor import PointJEPA

    model_config = {"num_groups": 8, "group_size": 8, "encoder_dim": 24, "encoder_depth": 2,
                    "encoder_heads": 3, "predictor_dim": 12, "predictor_depth": 1, "predictor_heads": 3,
                    "dropout": .1, "attention_dropout": .05, "drop_path_rate": .1}
    ssl_config = {"target_ratio": [.15, .2], "context_ratio": [.4, .75], "target_blocks": 4, "loss_beta": 2.}
    seed_all(17)
    model = PointJEPA(model_config, ssl_config)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=.001)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    sampler = torch.Generator().manual_seed(17)
    save_checkpoint(tmp_path / "state.pt", model, optimizer, scheduler, sampler, 0, {}, [])

    def update(model, optimizer, scheduler, sampler):
        optimizer.zero_grad(set_to_none=True)
        samples = torch.randn(4, 32, 3)
        indices = torch.randint(32, (3,), generator=sampler)
        loss = model(samples)["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.update_target(.99)
        return loss.detach(), indices

    expected_loss, expected_indices = update(model, optimizer, scheduler, sampler)
    expected_hash = state_hash(model.state_dict())
    checkpoint = torch.load(tmp_path / "state.pt", weights_only=False)
    resumed = PointJEPA(model_config, ssl_config)
    resumed_optimizer = torch.optim.AdamW([value for value in resumed.parameters() if value.requires_grad], lr=.001)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda step: 1 / (step + 1))
    resumed_sampler = torch.Generator()
    resumed.load_state_dict(checkpoint["model"], strict=True)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    resumed_sampler.set_state(checkpoint["sampler_rng"])
    restore_rng(checkpoint["rng"])
    observed_loss, observed_indices = update(resumed, resumed_optimizer, resumed_scheduler, resumed_sampler)
    assert torch.equal(observed_loss, expected_loss) and torch.equal(observed_indices, expected_indices)
    assert state_hash(resumed.state_dict()) == expected_hash


def test_data_parser_reads_only_expected_scalar_vertices_and_xyz_cache(tmp_path):
    specification = importlib.util.spec_from_file_location("prepare_m5_test_data", ROOT / "scripts/prepare_m5_data.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    header = b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nproperty int label\nend_header\n"
    vertices = module.read_vertices(header + b"1 2 3 8\n4 5 6 9\n")
    assert vertices["x"].tolist() == [1., 4.] and vertices["label"].tolist() == [8, 9]
    xyz = np.random.default_rng(5).normal(size=(256, 3)).astype(np.float32)
    record = module.cache_geometry(xyz, tmp_path / "cache.npz", "plant-a")
    with np.load(tmp_path / "cache.npz") as archive:
        assert archive["points"].shape == (2048, 3)
        assert archive["neighbors"].shape == (64, 32)
        assert not {"label", "instance", "semantic"}.intersection(archive.files)
        assert np.isclose(np.linalg.norm(archive["points"], axis=1).max(), 1., atol=.02)
    assert record["cache_sha256"] == sha256(tmp_path / "cache.npz")
