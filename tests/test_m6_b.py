from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from m6a.common import localize_gt
from m6a.experiment import metadata
from m6a.metrics import frame_metrics, observation_identity, event_metrics
from m6b.data import TrainingDataset, read_frame, restore, transform
from m6b.evaluation import cluster_prediction, sampled_forward
from m6b.models import discriminative_leaf_loss, official_joint_loss

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("pq,idf1,event_f1,small_recall,stable,expected", [
    (.9, .9, .8, .9, True, "INSTANCE-SOLVED-FOR-TRACKING"),
    (.9, .7, .4, .9, True, "DATA-LIMITED"),
    (.65, .7, .4, .9, True, "INSTANCE-METHOD-NEEDED"),
    (.9, .7, .4, .4, True, "INSTANCE-METHOD-NEEDED"),
    (.65, .7, .4, .4, False, "DATA-LIMITED"),
])
def test_research_decision_does_not_blame_segmentation_for_tracking_only_failure(tmp_path, pq, idf1, event_f1, small_recall, stable, expected):
    from m6b.reporting import make_decision

    (tmp_path / "configs/m6_b").mkdir(parents=True)
    (tmp_path / "reports/m6_b").mkdir(parents=True)
    (tmp_path / "configs/m6_b/protocol.json").write_bytes((ROOT / "configs/m6_b/protocol.json").read_bytes())
    selection = {"B3_method": "S1"}
    plants = pd.DataFrame([{"split": "val", "method": method, "seed": 17, "pq": pq} for method in ["S1", "S3"]])
    strata = pd.DataFrame([{"split": "val", "method": method, "seed": 17, "category": "early_small", "recall": small_recall} for method in ["S1", "S3"]])
    strata.to_csv(tmp_path / "reports/m6_b/error_strata_per_plant.csv", index=False)
    paired = pd.DataFrame([{"split": "val", "method": method, "seed": seed, "metric": "pq", "comparison": method + "-S0",
                            "ci95_low": .1 if stable else -.1, "improved_plants": 12 if stable else 5}
                           for method, seed in [("S1", 17), ("S1", "mean_seeds"), ("S3", 17)]])
    tracking = pd.DataFrame([{"split": "val", "pipeline": pipeline, "idf1": idf1, "new_event_f1": event_f1} for pipeline in ["B3", "B4"]])
    training = pd.DataFrame([{"validation_final_minus_50": .001, "last10_discriminative": .1}] * 7)
    decision = make_decision(tmp_path, plants, paired, tracking, training, selection)
    assert decision["recommendation"] == expected
    assert decision["exactly_one"] and not decision["measurement_system_certified"]


def test_first_observation_instance_omissions_identify_segmentation_bottleneck(tmp_path):
    from m6b.reporting import make_decision

    (tmp_path / "configs/m6_b").mkdir(parents=True)
    (tmp_path / "reports/m6_b").mkdir(parents=True)
    (tmp_path / "configs/m6_b/protocol.json").write_bytes((ROOT / "configs/m6_b/protocol.json").read_bytes())
    pd.DataFrame([{"split": "val", "method": "S1", "seed": 17, "category": "early_small", "recall": .9}]).to_csv(
        tmp_path / "reports/m6_b/error_strata_per_plant.csv", index=False)
    plants = pd.DataFrame([{"split": "val", "method": "S1", "seed": 17, "pq": .9}])
    paired = pd.DataFrame([{"split": "val", "method": "S1", "seed": seed, "metric": "pq", "comparison": "S1-S0",
                            "ci95_low": .1, "improved_plants": 12} for seed in [17, "mean_seeds"]])
    tracking = pd.DataFrame([{"split": "val", "pipeline": "B3", "idf1": .85, "new_event_f1": .3,
                              "new_missed_instance": 4., "new_attached_old_identity": 2.}])
    training = pd.DataFrame([{"validation_final_minus_50": .001, "last10_discriminative": .1}] * 7)
    decision = make_decision(tmp_path, plants, paired, tracking, training, {"B3_method": "S1"})
    assert decision["recommendation"] == "INSTANCE-METHOD-NEEDED"
    assert decision["instance_bottleneck_diagnostic"]


def test_independent_official_plants():
    records = metadata(ROOT)
    assert records.groupby("plant_uid").split.nunique().max() == 1
    assert records.groupby("plant_uid").size().eq(16).all()
    assert records.groupby("split").plant_uid.nunique().to_dict() == {"test": 12, "train": 24, "val": 12}
    fixed = json.loads((ROOT / "reports/m3/plant_splits.json").read_text())["assignments"]["official"]
    assert all(records.split == records.plant_uid.map(fixed))


def test_current_frame_semantics_and_negative_ids():
    record = metadata(ROOT).iloc[0]
    frame = read_frame(ROOT / record.pointcloud_path)
    assert set(np.unique(frame["semantic"])) <= {0, 1, 2}
    assert np.all(frame["local_instance"][frame["semantic"] != 2] == -1)
    ids = np.unique(frame["local_instance"][frame["local_instance"] >= 0])
    np.testing.assert_array_equal(ids, np.arange(len(ids)))
    assert len(frame["xyz"]) == int(record.point_count)


def test_physical_scale_inverse():
    xyz = np.random.default_rng(3).normal(size=(100, 3)).astype(np.float32) * .2 + np.array([.4, -.8, 1])
    normalized, center, scale = transform(xyz)
    np.testing.assert_allclose(restore(normalized, center, scale), xyz, atol=1e-7)
    assert scale > 0


def test_frame_local_embedding_invariant_and_nonleaf_mask():
    generator = torch.Generator().manual_seed(5)
    embedding = torch.randn(2, 9, 5, generator=generator, requires_grad=True)
    labels = torch.tensor([[0, 0, 0, 1, 1, 2, 2, -1, -1], [0, 0, 1, 1, 1, 1, 1, -1, -1]])
    permuted = labels.clone()
    permuted[0, labels[0] == 0] = 70
    permuted[0, labels[0] == 1] = 8
    permuted[0, labels[0] == 2] = 99
    permuted[1, labels[1] == 0] = 12
    permuted[1, labels[1] == 1] = 2
    loss = discriminative_leaf_loss(embedding, labels)
    torch.testing.assert_close(loss, discriminative_leaf_loss(embedding, permuted))
    loss.backward()
    assert torch.count_nonzero(embedding.grad[labels < 0]) == 0


def test_joint_loss_frame_local_permutation():
    logits = torch.randn(2, 9, 3)
    embedding = torch.randn(2, 9, 5)
    similarity = torch.rand(2, 9, 9) * 100
    semantic = torch.tensor([[2] * 7 + [0, 1], [2] * 7 + [0, 1]])
    instances = torch.tensor([[0, 0, 0, 1, 1, 2, 2, -1, -1], [0, 0, 1, 1, 1, 1, 1, -1, -1]])
    permuted = instances.clone()
    permuted[0, instances[0] == 0] = 30
    permuted[0, instances[0] == 1] = 8
    permuted[0, instances[0] == 2] = 2
    permuted[1, instances[1] == 0] = 50
    permuted[1, instances[1] == 1] = 90
    original, components = official_joint_loss(logits, embedding, similarity, semantic, instances)
    renamed, other = official_joint_loss(logits, embedding, similarity, semantic, permuted)
    torch.testing.assert_close(original, renamed)
    assert set(components) == {"semantic", "discriminative", "similarity"}


def test_all_nonleaf_has_finite_joint_loss():
    logits = torch.randn(1, 8, 3, requires_grad=True)
    embeddings = torch.randn(1, 8, 5, requires_grad=True)
    similarity = torch.rand(1, 8, 8, requires_grad=True)
    loss, components = official_joint_loss(logits, embeddings, similarity, torch.ones(1, 8, dtype=torch.long), torch.full((1, 8), -1))
    assert torch.isfinite(loss)
    assert components["discriminative"] == 0 and components["similarity"] == 0
    loss.backward()


def test_full_resolution_restore_and_no_gt_filtering():
    xyz = np.array([[0., 0, 0], [.01, 0, 0], [1., 0, 0], [1.01, 0, 0], [2., 0, 0]])
    sampled = {"sample_indices": np.array([0, 2, 4]), "semantic_probability": np.array([[.01, .01, .98], [.01, .01, .98], [.1, .8, .1]]),
               "embedding": np.array([[0., 0, 0, 0, 0], [4., 4, 4, 4, 4], [0., 0, 0, 0, 0]])}
    result = cluster_prediction(xyz, sampled, .6)
    assert len(result["labels"]) == 5 and result["labels"][-1] == 0
    assert len(np.unique(result["labels"][result["labels"] > 0])) == 2
    score, matches = frame_metrics(np.array([1, 1, 2, 2, 0]), result["labels"])
    assert score["pq"] == 1 and len(matches) == 2


def test_reject_label_bearing_model_input():
    with pytest.raises(ValueError, match="XYZ"):
        sampled_forward(None, np.zeros((16, 5)), 17, "cpu")


def test_tracking_local_ids_and_missing_vs_identity():
    first = np.array([1, 1, 2, 2, 3, 3, 0])
    localized = localize_gt(first, 6)
    score, matches = frame_metrics(first, localized)
    assert score["pq"] == 1 and len(matches) == 3
    rows = [{"gt_id": 1, "frame_index": 0, "local_id": 4, "track_id": 0},
            {"gt_id": 1, "frame_index": 1, "local_id": 9, "track_id": 0},
            {"gt_id": 2, "frame_index": 1, "local_id": -1, "track_id": -1}]
    identity, unused = observation_identity(rows, 2)
    events = event_metrics(rows, [{"frame_index": 0, "local_id": 4, "track_id": 0}])
    assert identity["idfn"] == 1 and events["new_missed_instance"] == 1


def test_frozen_training_policy():
    protocol = json.loads((ROOT / "configs/m6_b/protocol.json").read_text())
    assert protocol["training"]["seeds"] == [17, 29, 43]
    assert protocol["evaluation"]["main_split"] == "val"
    assert "exploratory" in protocol["evaluation"]["test_role"]
    assert protocol["training"]["epochs"] == 60
    assert protocol["continuation"]["exactly_one"]


def test_training_dataset_only_train():
    if not (ROOT / "reports/m6_b/training_cache_ready.json").exists():
        pytest.skip("Full training cache not prepared")
    dataset = TrainingDataset(ROOT, 17)
    assert len(dataset) == 384 and dataset.records.split.eq("train").all()
    xyz, semantic, instance = dataset[0]
    assert xyz.shape == (4096, 3)
    assert np.all(instance[semantic != 2] == -1)
    assert np.all(np.isfinite(xyz))
