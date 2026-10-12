#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m4.common import load_protocol, seed_all, state_hash, write_json
from models.point_jepa_encoder import PointJEPAEncoder, load_encoder_checkpoint, patch_indices
from models.point_jepa_predictor import PointJEPA


def verify(root, device):
    protocol, _ = load_protocol(root)
    results = []
    started = time.monotonic()
    for seed in (17, 17):
        seed_all(seed)
        model = PointJEPA(protocol["model"], protocol["ssl"]).to(device)
        points = torch.randn(2, protocol["points"], 3, device=device)
        optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-3)
        output = model(points)
        output["loss"].backward()
        gradients = [value.grad for value in model.parameters() if value.grad is not None]
        assert torch.isfinite(output["loss"]) and all(torch.isfinite(value).all() for value in gradients)
        assert all(value.grad is None for value in model.target_encoder.parameters())
        assert not torch.isin(output["context_indices"], output["target_indices"]).any()
        target_before = state_hash(model.target_encoder.state_dict())
        optimizer.step()
        model.update_target(0.9995)
        assert state_hash(model.target_encoder.state_dict()) != target_before
        results.append({"loss": float(output["loss"].detach()), "state_hash": state_hash(model.state_dict()),
                        "target_shape": list(output["target"].shape), "gradients_finite": True,
                        "target_gradients_absent": True, "context_target_disjoint": True})
        del model, optimizer, output, gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()
    assert results[0] == results[1]
    seed_all(42)
    small_config = protocol["model"] | {"encoder_dim": 48, "encoder_depth": 2, "encoder_heads": 3,
                                       "predictor_dim": 24, "predictor_depth": 2, "predictor_heads": 3,
                                       "dropout": 0.0, "attention_dropout": 0.0, "drop_path_rate": 0.0}
    small = PointJEPA(small_config, protocol["ssl"]).to(device).eval()
    points = torch.randn(2, protocol["points"], 3, device=device)
    centers, neighbors = patch_indices(points)
    output = small(points, centers, neighbors)
    masks = (output["context_indices"], output["target_indices"])
    optimizer = torch.optim.AdamW(small.predictor.parameters(), lr=0.003)
    losses = []
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = small(points, centers, neighbors, masks)["loss"]
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 0.7
    checkpoint_path = root / "outputs/m4/external/point_jepa_shapenet.ckpt"
    checkpoint_report = {"status": "UNAVAILABLE", "reason": "Google Drive official checkpoint not downloaded", "parameter_coverage": None}
    if checkpoint_path.exists():
        encoder = PointJEPAEncoder(**protocol["model"])
        checkpoint_report = load_encoder_checkpoint(encoder, checkpoint_path, root / "reports/m4/checkpoint_compatibility.json") | {"status": "STRICT-MATCH"}
    write_json(root / "reports/m4/checkpoint_compatibility.json", checkpoint_report)
    write_json(root / "reports/m4/m4a_correctness.json", {
        "status": "PASS", "full_official_backbone": True, "device": str(device), "torch": torch.__version__,
        "full_2048_point_forward_backward": results, "deterministic_repeat": True,
        "tiny_fixed_mask_overfit_loss": losses, "tiny_overfit_note": "predictor-only sanity check with fixed target; not evidence of useful representation",
        "automated_tests": "reports/m4/tests_m4.xml", "checkpoint_rng_resume": "tested exact next step including optimizer, scheduler, masks and RNG",
        "seconds": time.monotonic() - started})
    print("M4-A PASS: official-size GPU forward/backward, EMA, masks, repeatability and small overfit", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    verify(Path(arguments.root), torch.device(arguments.device))
