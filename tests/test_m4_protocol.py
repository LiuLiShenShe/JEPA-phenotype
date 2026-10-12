from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import torch

from datasets.sugar4d_ssl_dataset import partition_rows
from m4.common import artifact_protocol_matches, load_protocol, sha256, write_json
from m4.evaluation import decide, fairness_checks
from m4.downstream import run_count, run_segmentation
from m4.inference import verify_prediction_restore
from m4.pretrain import run_pretrain
from models.point_jepa_encoder import patch_indices


ROOT = Path(__file__).resolve().parents[1]


def test_lock_covers_original_splits_metadata_and_geometry_and_no_test_loader():
    protocol, lock = load_protocol(ROOT)
    assert lock["protocol_sha256"] == sha256(ROOT / "configs/m4/protocol.json")
    assert protocol["primary"]["task"] == "current visible annotated leaf count"
    assert protocol["stop_before_temporal_jepa"]
    for partition in protocol["partitions"]:
        train = partition_rows(ROOT, partition, "train")
        validation = partition_rows(ROOT, partition, "val")
        assert train.plant_uid.nunique() == 24 and validation.plant_uid.nunique() == 12
        assert not set(train.plant_uid) & set(validation.plant_uid)
        assert len(train) == 384 and len(validation) == 192
        with pytest.raises(ValueError):
            partition_rows(ROOT, partition, "test")
        budgets = lock["label_budgets"][partition]
        assert set(budgets["0.25"]) < set(budgets["0.5"]) < set(budgets["1.0"]) == set(train.plant_uid)


def test_annotation_only_provenance_alias_cannot_change_any_experiment_setting(tmp_path):
    protocol, lock = load_protocol(ROOT)
    alias = "bf79e05a5d2aaaac7f06319247f4e5f680416847f3565b19f418eb50c56f234d"
    assert artifact_protocol_matches(ROOT, alias, lock)
    assert not artifact_protocol_matches(ROOT, "not-a-valid-hash", lock)
    (tmp_path / "configs/m4").mkdir(parents=True)
    (tmp_path / "reports/m4/sources").mkdir(parents=True)
    (tmp_path / "configs/m4/protocol.json").write_text(json.dumps(protocol))
    altered = protocol | {"notes": ["looks cosmetic"]}
    altered["seeds"] = [999]
    archive = tmp_path / "reports/m4/sources/protocol_annotations_only.json"
    archive.write_text(json.dumps(altered))
    assert not artifact_protocol_matches(tmp_path, sha256(archive), lock)


def test_fairness_checks_detect_head_or_budget_changes():
    base = {"partition": "official", "seed": 17, "task": "count", "mode": "finetune", "label_fraction": 1.0,
            "initial_head_hash": "same", "trainable_parameters": 10, "optimizer_steps": 20, "labelled_plants": ["plant_a"]}
    table = pd.DataFrame([base | {"condition": "P0"}, base | {"condition": "P2"}])
    assert fairness_checks(table)["status"] == "PASS"
    table.loc[1, "optimizer_steps"] = 100
    assert fairness_checks(table)["status"] == "FAIL"


def test_decision_cannot_switch_to_segmentation_or_use_one_good_seed():
    protocol, _ = load_protocol(ROOT)
    good = {"partition": "official", "mode": "finetune", "label_fraction": 1.0,
            "p0_mae": 2.0, "p2_mae": 1.5, "absolute_improvement": 0.5, "relative_improvement": 0.25,
            "paired_seeds": 3, "improving_seeds": 3, "ci95_high": -0.1}
    comparisons = [good] + [good | {"partition": f"fold{index}"} for index in range(4)]
    baseline = pd.DataFrame({"partition": ["official"], "condition": ["A3-PointNet-Geometry"], "plant_macro_mae": [1.6]})
    diagnostics = {"test": {"final": {"collapse_warning": False}}}
    count = pd.DataFrame({"condition": ["P0", "P2"], "mode": ["finetune"] * 2, "label_fraction": [1.0] * 2, "plant_macro_mae": [2.0, 1.5]})
    segment = pd.DataFrame({"condition": ["P0", "P2", "PointNet"], "mode": ["finetune"] * 3, "plant_macro_miou": [0.6, 0.9, 0.5]})
    coverage = {"p0_p2_matrix_complete": True}
    fairness = {"status": "PASS"}
    result = decide(protocol, comparisons, count, segment, baseline, diagnostics, coverage, fairness)
    assert result["decision"] == "REPRESENTATION-SUCCESS"
    comparisons[0] = good | {"improving_seeds": 1}
    assert decide(protocol, comparisons, count, segment, baseline, diagnostics, coverage, fairness)["decision"] == "LIMITED-BENEFIT"
    comparisons[0] = good
    coverage["p0_p2_matrix_complete"] = False
    assert decide(protocol, comparisons, count, segment, baseline, diagnostics, coverage, fairness)["decision"] == "LIMITED-BENEFIT"


def test_random_and_ssl_checkpoint_provenance_and_no_validation_pretraining():
    protocol, lock = load_protocol(ROOT)
    for partition in protocol["partitions"]:
        train = set(partition_rows(ROOT, partition, "train").plant_uid)
        validation = set(partition_rows(ROOT, partition, "val").plant_uid)
        for seed in protocol["seeds"]:
            path = ROOT / "outputs/m4/checkpoints" / partition / f"ssl_seed{seed}/training.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            assert set(record["train_plants"]) == train and not train & validation
            assert record["setting"] == "Sugar4D-train-only-from-scratch"
            assert record["validation_used"] is False and record["test_used"] is False
            assert artifact_protocol_matches(ROOT, record["protocol_sha256"], lock)


def test_tiny_isolated_training_pipeline_and_saved_inference_roundtrip(tmp_path):
    protocol = json.loads((ROOT / "configs/m4/protocol.json").read_text())
    protocol["model"].update({"num_groups": 8, "group_size": 8, "encoder_dim": 24, "encoder_depth": 2,
                              "encoder_heads": 3, "predictor_dim": 12, "predictor_depth": 1, "predictor_heads": 3,
                              "dropout": 0.0, "attention_dropout": 0.0, "drop_path_rate": 0.0})
    protocol["points"] = 32
    protocol["ssl"].update({"epochs": 2, "warmup_epochs": 1, "batch_size": 8})
    protocol["downstream"].update({"epochs": 2, "probe_epochs": 3, "batch_size": 8})
    protocol["seeds"] = [17]
    protocol["partitions"] = ["official"]
    write_json(tmp_path / "configs/m4/protocol.json", protocol)
    assignments = {"plant_a": "train", "plant_b": "train", "plant_c": "val", "plant_d": "test"}
    write_json(tmp_path / "reports/m3/plant_splits.json", {"assignments": {"official": assignments}})
    write_json(tmp_path / "reports/m4/m4a_correctness.json", {"status": "PASS"})
    write_json(tmp_path / "reports/m4/external_pretraining_lock.json", {"p1_status": "UNAVAILABLE"})
    cache = tmp_path / "outputs/m4/data/xyz"
    labels = tmp_path / "outputs/m4/data/labels"
    cache.mkdir(parents=True)
    labels.mkdir(parents=True)
    (tmp_path / "outputs/m4/predictions").mkdir(parents=True)
    generator = np.random.default_rng(20261012)
    metadata, geometry = [], []
    for plant, split in assignments.items():
        for das in range(16):
            name = f"{das}_das_{plant}"
            metadata.append({"plant_uid": plant, "DAS": das, "pointcloud_path": name + ".ply", "visible_organ_count": float(4 + das // 4), "split": split})
            geometry.append({"plant_uid": plant, "DAS": das, "extent_z": das + 2, "extent_x": float(generator.random())})
            points = generator.normal(size=(32, 3)).astype(np.float32)
            centers, neighbors = patch_indices(torch.from_numpy(points)[None], 8, 8)
            np.savez(cache / (name + ".npz"), points=points, center_indices=centers[0].numpy(), neighbors=neighbors[0].numpy(),
                     center_m=np.zeros(3, dtype=np.float32), radius_m=np.array(1, dtype=np.float32), extent_m=np.ones(3, dtype=np.float32))
            np.savez(labels / (name + ".npz"), semantic=generator.integers(3, size=32, dtype=np.int64))
    (tmp_path / "data/processed/sugar4d").mkdir(parents=True)
    (tmp_path / "outputs/m3_5/data").mkdir(parents=True)
    pd.DataFrame(metadata).to_csv(tmp_path / "data/processed/sugar4d/plant_metadata.csv", index=False)
    pd.DataFrame(geometry).to_csv(tmp_path / "outputs/m3_5/data/geometry.csv", index=False)
    write_json(tmp_path / "reports/m4/protocol_lock.json", {"protocol_sha256": sha256(tmp_path / "configs/m4/protocol.json"),
                                                           "inputs": {}, "label_budgets": {"official": {"1.0": ["plant_a", "plant_b"]}}})
    device = torch.device("cpu")
    ssl = run_pretrain(tmp_path, "official", 17, device)
    assert ssl["optimizer_steps"] == 8 and ssl["test_used"] is False
    heads = []
    for condition in ("P0", "P2"):
        for mode in ("probe", "finetune"):
            count = run_count(tmp_path, "official", 17, condition, 1.0, mode, device)
            segment = run_segmentation(tmp_path, "official", 17, condition, mode, device)
            for record in (count, segment):
                checkpoint = tmp_path / "outputs/m4/checkpoints/official" / record["id"] / "best.pt"
                assert verify_prediction_restore(tmp_path, checkpoint, device)["status"] == "PASS"
                state = torch.load(checkpoint.parent / "training_state.pt", weights_only=False)
                assert state["model"] and state["optimizer"] and state["rng"]
            heads.append((condition, mode, count["initial_head_hash"]))
    for mode in ("probe", "finetune"):
        assert len({value for _, method, value in heads if method == mode}) == 1
