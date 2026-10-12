from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.sugar4d_ssl_dataset import Sugar4DSSLDataset, current_transform, labelled_plants, partition_rows
from m4.common import representation_diagnostics, resume_training, save_training, seed_all, state_hash
from m4.metrics import count_metrics, segmentation_metrics
from models.point_jepa_downstream import PointJEPACount, PointJEPASegmentation, interpolate_tokens
from models.point_jepa_encoder import DropPath, PatchPointNet, PointJEPAEncoder, Transformer, greedy_order, load_encoder_checkpoint, patch_indices
from models.point_jepa_predictor import PointJEPA, PointJEPAPredictor, sample_masks


MODEL = {"num_groups": 8, "group_size": 8, "encoder_dim": 24, "encoder_depth": 2, "encoder_heads": 3,
         "dropout": 0.0, "attention_dropout": 0.0, "drop_path_rate": 0.0,
         "predictor_dim": 12, "predictor_depth": 1, "predictor_heads": 3}
SSL = {"target_ratio": [0.15, 0.2], "context_ratio": [0.4, 0.75], "target_blocks": 4, "loss_beta": 2.0}


def test_forward_backward_shapes_and_stop_gradient():
    seed_all(17)
    model = PointJEPA(MODEL, SSL)
    output = model(torch.randn(2, 32, 3))
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert output["prediction"].shape == output["target"].shape == (4, 2, 1, 24)
    assert not output["target"].requires_grad
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.encoder.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_ema_matches_equation_and_target_never_trains():
    model = PointJEPA(MODEL, SSL)
    model.train()
    assert not model.target_encoder.training
    before = {name: value.clone() for name, value in model.target_encoder.state_dict().items()}
    with torch.no_grad():
        for parameter in model.encoder.student.parameters():
            parameter.add_(0.25)
    model.update_target(0.8)
    for name, value in model.target_encoder.state_dict().items():
        torch.testing.assert_close(value, before[name] * 0.8 + model.encoder.student.state_dict()[name] * 0.2)
    assert int(model.ema_updates) == 1


def test_masks_never_overlap_and_never_illegally_fallback():
    for _ in range(100):
        context, targets = sample_masks(64, "cpu")
        assert len(context) > 0 and not torch.isin(context, targets).any()
        assert (targets[:, 1:] - targets[:, :-1] == 1).all()
    with pytest.raises(ValueError):
        sample_masks(2, "cpu", (1.0, 1.0))


def test_patch_geometry_shapes_permutation_and_reconstruction():
    seed_all(42)
    points = torch.randn(3, 32, 3)
    centers, neighbors = patch_indices(points, 8, 8)
    assert centers.shape == (3, 8) and neighbors.shape == (3, 8, 8)
    coordinates = points[torch.arange(3)[:, None], centers]
    order = greedy_order(coordinates)
    assert all(len(torch.unique(row)) == 8 for row in order)
    model = PointJEPAEncoder(**MODEL).eval()
    pooled, tokens, positions = model(points, centers, neighbors, return_tokens=True)
    assert tokens.shape == (3, 8, 24) and pooled.shape == (3, 48) and positions.shape == (3, 8, 3)
    normalized, center, radius = current_transform(points[0].numpy())
    np.testing.assert_allclose(normalized * radius + center, points[0].numpy(), atol=3e-7)


def test_checkpoint_exact_next_step_and_rng_resume(tmp_path):
    seed_all(93)
    model = PointJEPA(MODEL, SSL)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    generator = torch.Generator().manual_seed(93)
    points = torch.randn(2, 32, 3)
    optimizer.zero_grad()
    model(points)["loss"].backward()
    optimizer.step()
    model.update_target(0.9)
    scheduler.step()
    path = tmp_path / "training.pt"
    save_training(path, model, optimizer, scheduler, 1, 1, generator, {"train_plants": ["plant_a"]})

    def next_step():
        optimizer.zero_grad()
        output = model(points)
        output["loss"].backward()
        optimizer.step()
        model.update_target(0.9)
        return float(output["loss"].detach()), state_hash(model.state_dict()), np.random.rand(), torch.rand(1)

    first = next_step()
    restored = resume_training(path, model, optimizer, scheduler, generator)
    second = next_step()
    assert restored["epoch"] == 1 and restored["steps"] == 1
    assert first[:3] == second[:3]
    torch.testing.assert_close(first[3], second[3], rtol=0, atol=0)


def test_seed_repeatability_and_fixed_mask_tiny_loss_descends():
    losses, hashes = [], []
    for _ in range(2):
        seed_all(17)
        model = PointJEPA(MODEL, SSL)
        points = torch.randn(2, 32, 3)
        output = model(points)
        losses.append(float(output["loss"].detach()))
        hashes.append(state_hash(model.state_dict()))
    assert losses[0] == losses[1] and hashes[0] == hashes[1]
    model.eval()
    centers, neighbors = patch_indices(points, 8, 8)
    masks = (torch.tensor([0, 1, 2]), torch.tensor([[4], [5], [6], [7]]))
    optimizer = torch.optim.AdamW(model.predictor.parameters(), lr=0.005)
    initial = float(model(points, centers, neighbors, masks)["loss"].detach())
    for _ in range(40):
        optimizer.zero_grad()
        output = model(points, centers, neighbors, masks)
        output["loss"].backward()
        optimizer.step()
    assert float(model(points, centers, neighbors, masks)["loss"].detach()) < initial * 0.6


def test_frozen_encoder_parameters_and_buffers_unchanged():
    encoder = PointJEPAEncoder(**MODEL)
    model = PointJEPACount(encoder, 4, frozen=True)
    before = state_hash(encoder.state_dict())
    model.train()
    model(torch.randn(2, 32, 3), torch.randn(2, 4)).sum().backward()
    assert before == state_hash(encoder.state_dict())
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert model.head.weight.grad is not None


def test_segmentation_uses_tokens_without_annotation_input():
    seed_all(42)
    encoder = PointJEPAEncoder(**MODEL)
    model = PointJEPASegmentation(encoder)
    points = torch.randn(2, 32, 3)
    output = model(points)
    assert output.shape == (2, 32, 3)
    torch.nn.functional.cross_entropy(output.transpose(1, 2), torch.randint(3, (2, 32))).backward()
    assert encoder.student.blocks[0].attn.qkv.weight.grad is not None
    tokens = torch.ones(2, 8, 24)
    features = interpolate_tokens(points, points[:, :8], tokens)
    torch.testing.assert_close(features[:, :, :24], torch.ones(2, 32, 24))


def test_nested_independent_plant_label_budgets():
    plants = [f"plant_{index:03d}" for index in range(24)]
    quarter = labelled_plants(plants, 0.25, "official")
    half = labelled_plants(plants, 0.5, "official")
    full = labelled_plants(plants, 1.0, "official")
    assert len(quarter) == 6 and len(half) == 12 and len(full) == 24
    assert set(quarter) < set(half) < set(full)
    assert quarter == labelled_plants(plants[::-1], 0.25, "official")


def test_ssl_rejects_validation_and_has_no_labels(tmp_path):
    rows = pd.DataFrame({"plant_uid": ["plant_001"], "DAS": [37], "pointcloud_path": ["scan.ply"], "split": ["val"], "visible_organ_count": [99]})
    with pytest.raises(ValueError):
        Sugar4DSSLDataset(rows, tmp_path)
    rows["split"] = "train"
    np.savez(tmp_path / "scan.npz", points=np.ones((32, 3), dtype=np.float32), center_indices=np.arange(8),
             neighbors=np.tile(np.arange(8), (8, 1)), center_m=np.zeros(3, dtype=np.float32),
             radius_m=np.array(1, dtype=np.float32), extent_m=np.ones(3, dtype=np.float32), semantic=np.ones(32))
    dataset = Sugar4DSSLDataset(rows, tmp_path)
    assert set(dataset.rows.columns) == {"plant_uid", "DAS", "pointcloud_path"}
    assert not {"semantic", "count", "visible_organ_count", "instance", "DAS"} & set(dataset[0])


def test_strict_official_encoder_loading_and_foreign_checkpoint_rejection(tmp_path):
    encoder = PointJEPAEncoder(**MODEL)
    path = tmp_path / "official.ckpt"
    torch.save({"state_dict": encoder.state_dict() | {"predictor.fake": torch.zeros(3)}}, path)
    report = load_encoder_checkpoint(encoder, path, tmp_path / "report.json")
    assert report["parameter_coverage"] == 1 and report["strict_encoder_match"]
    torch.save({"base_model": {"pointnet.weight": torch.ones(1)}}, path)
    with pytest.raises(ValueError):
        load_encoder_checkpoint(encoder, path)


def test_metrics_are_plant_macro_not_scan_weighted_and_no_fake_instance_metric():
    frame = pd.DataFrame({"plant_uid": ["a", "b", "b", "b"], "prediction": [4, 0, 0, 0], "truth": [0, 0, 0, 0]})
    metrics, plants = count_metrics(frame)
    assert metrics["plant_macro_mae"] == 2 and len(plants) == 2
    matrices = np.stack([np.eye(3, dtype=int)] * 4)
    segmentation, _ = segmentation_metrics(matrices, frame)
    assert segmentation["plant_macro_miou"] == 1
    assert "pq" not in segmentation and "sbd" not in segmentation


def test_collapse_diagnostics_warn_on_constant_representations():
    collapsed = representation_diagnostics(torch.ones(4, 8, 24), torch.ones(4, 48))
    assert collapsed["collapse_warning"] and collapsed["effective_rank"] == 0
    normal = representation_diagnostics(torch.randn(8, 8, 24), torch.randn(8, 48))
    assert not normal["collapse_warning"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_weighted_point_cross_entropy_deterministic_cuda():
    seed_all(17)
    logits = torch.randn(2, 32, 3, device="cuda", requires_grad=True)
    labels = torch.randint(3, (2, 32), device="cuda")
    weights = torch.tensor([2.0, 0.5, 1.0], device="cuda")
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), labels.reshape(-1), weight=weights)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_upstream_transformer_pointnet_predictor_numerical_parity(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    original = root / "third_party/m4/point_jepa/pointjepa/modules"
    timm, models, layers = types.ModuleType("timm"), types.ModuleType("timm.models"), types.ModuleType("timm.models.layers")
    layers.DropPath = DropPath
    for name, value in (("timm", timm), ("timm.models", models), ("timm.models.layers", layers)):
        monkeypatch.setitem(sys.modules, name, value)
    specification = importlib.util.spec_from_file_location("m4_reference_transformer", original / "transformer.py")
    reference = importlib.util.module_from_spec(specification)
    monkeypatch.setitem(sys.modules, specification.name, reference)
    specification.loader.exec_module(reference)
    our = Transformer(24, 2, 3).eval()
    their = reference.TransformerEncoder(embed_dim=24, depth=2, num_heads=3, qkv_bias=True, add_pos_at_every_layer=True).eval()
    their.load_state_dict(our.state_dict(), strict=True)
    values, positions = torch.randn(2, 8, 24), torch.randn(2, 8, 24)
    torch.testing.assert_close(our(values, positions), their(values, positions).last_hidden_state, rtol=0, atol=0)
    specification = importlib.util.spec_from_file_location("m4_reference_pointnet", original / "Pointnet.py")
    pointnet_reference = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(pointnet_reference)
    patch = PatchPointNet(24).eval()
    original_patch = pointnet_reference.PointNetSmall(3, 24).eval()
    original_patch.load_state_dict(patch.state_dict(), strict=True)
    points = torch.randn(4, 8, 3)
    torch.testing.assert_close(patch(points), original_patch(points), rtol=0, atol=0)
    predictor = PointJEPAPredictor(**MODEL).eval()
    reference_predictor = reference.TransformerPredictor(embed_dim=24, predictor_embed_dim=12, depth=1,
                                                       num_heads=3, qkv_bias=True, attn_drop_rate=0, drop_path_rate=0,
                                                       add_target_pos=True).eval()
    reference_predictor.load_state_dict(predictor.state_dict(), strict=True)
    centers = torch.randn(2, 8, 3)
    torch.testing.assert_close(predictor(values[:, :4], centers[:, :4], centers[:, 4:]),
                               reference_predictor(values[:, :4], centers[:, :4], centers[:, 4:]), rtol=0, atol=0)
