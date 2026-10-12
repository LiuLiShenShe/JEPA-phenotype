from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.sugar4d_ssl_dataset import Sugar4DSSLDataset, partition_rows
from m4.common import artifact_protocol_matches, load_protocol, representation_diagnostics, resume_training, save_training, seed_all, sha256, state_hash, write_json
from models.point_jepa_encoder import load_encoder_checkpoint
from models.point_jepa_predictor import PointJEPA


def rotate_z(points):
    angles = torch.rand(len(points), device=points.device) * (2 * math.pi)
    rotated = points.clone()
    rotated[:, :, 0] = points[:, :, 0] * angles.cos()[:, None] - points[:, :, 1] * angles.sin()[:, None]
    rotated[:, :, 1] = points[:, :, 0] * angles.sin()[:, None] + points[:, :, 1] * angles.cos()[:, None]
    return rotated


@torch.no_grad()
def diagnose(encoder, loader, device, maximum=96):
    encoder.eval()
    tokens, features = [], []
    for batch in loader:
        pooled, patches, _ = encoder(batch["points"].to(device), batch["center_indices"].to(device),
                                     batch["neighbors"].to(device), return_tokens=True)
        tokens.append(patches.cpu())
        features.append(pooled.cpu())
        if sum(len(value) for value in features) >= maximum:
            break
    return representation_diagnostics(torch.cat(tokens)[:maximum], torch.cat(features)[:maximum])


def run_pretrain(root, partition, seed, device):
    root = Path(root)
    protocol, lock = load_protocol(root)
    if json.loads((root / "reports/m4/m4a_correctness.json").read_text())["status"] != "PASS":
        raise ValueError("M4-A not passed")
    if partition not in protocol["partitions"] or seed not in protocol["seeds"]:
        raise ValueError("Unregistered partition/seed")
    folder = root / "outputs/m4/checkpoints" / partition / f"ssl_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    final_path, latest_path = folder / "encoder.pt", folder / "last.pt"
    if final_path.exists():
        record = json.loads((folder / "training.json").read_text())
        if not artifact_protocol_matches(root, record["protocol_sha256"], lock):
            raise ValueError("Existing checkpoint belongs to another protocol")
        return record
    seed_all(seed)
    rows = partition_rows(root, partition, "train")
    dataset = Sugar4DSSLDataset(rows, root / "outputs/m4/data/xyz")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=protocol["ssl"]["batch_size"], shuffle=True, generator=generator, num_workers=0)
    diagnostic_loader = DataLoader(dataset, batch_size=protocol["ssl"]["batch_size"], shuffle=False, num_workers=0)
    model = PointJEPA(protocol["model"], protocol["ssl"]).to(device)
    external = root / "outputs/m4/external/point_jepa_shapenet.ckpt"
    external_lock = json.loads((root / "reports/m4/external_pretraining_lock.json").read_text())
    if external.exists() and external_lock["p1_status"] == "UNAVAILABLE":
        raise ValueError("External weights were inserted after availability was locked; use a separately preregistered run")
    setting = "Sugar4D-train-only-from-scratch"
    external_digest = None
    if external.exists():
        load_encoder_checkpoint(model.encoder, external, root / "reports/m4" / f"load_{partition}_seed{seed}.json")
        model.reset_target()
        setting, external_digest = "ShapeNet-external-plus-Sugar4D-train-only-adaptation", sha256(external)
    initial_hash = state_hash(model.encoder.state_dict())
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=protocol["ssl"]["learning_rate"], weight_decay=protocol["ssl"]["weight_decay"])
    total_steps = len(loader) * protocol["ssl"]["epochs"]
    warmup = len(loader) * protocol["ssl"]["warmup_epochs"]

    def schedule(step):
        if step < warmup:
            return max(0.01, (step + 1) / warmup)
        return 0.001 + 0.999 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    metadata = {"partition": partition, "seed": seed, "protocol_sha256": lock["protocol_sha256"],
                "setting": setting, "external_checkpoint_sha256": external_digest,
                "train_plants": sorted(rows.plant_uid.unique()), "ssl_scans": len(rows), "validation_used": False,
                "test_used": False, "initial_encoder_hash": initial_hash, "points": protocol["points"]}
    history, steps, start_epoch = [], 0, 0
    prior_seconds = 0.0
    if latest_path.exists():
        checkpoint = resume_training(latest_path, model, optimizer, scheduler, generator)
        if not artifact_protocol_matches(root, checkpoint["metadata"]["protocol_sha256"], lock) or checkpoint["metadata"]["train_plants"] != metadata["train_plants"]:
            raise ValueError("Resume split/protocol mismatch")
        metadata = checkpoint["metadata"]
        start_epoch, steps = checkpoint["epoch"], checkpoint["steps"]
        history = pd.read_csv(folder / "history.csv").to_dict("records")[:start_epoch]
        prior_seconds = history[-1]["elapsed_seconds"] if history else 0.0
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    if start_epoch == 0:
        write_json(folder / "diagnostics_initial.json", diagnose(model.encoder, diagnostic_loader, device))
    print(f"SSL START {partition} seed={seed} train_plants={rows.plant_uid.nunique()} scans={len(rows)}", flush=True)
    for epoch in range(start_epoch + 1, protocol["ssl"]["epochs"] + 1):
        model.train()
        losses, target_std, prediction_std = [], [], []
        for batch in loader:
            points = rotate_z(batch["points"].to(device))
            optimizer.zero_grad(set_to_none=True)
            output = model(points, batch["center_indices"].to(device), batch["neighbors"].to(device))
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite JEPA loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 5.0,
                                                  error_if_nonfinite=True)
            optimizer.step()
            progress = steps / max(1, total_steps - 1)
            momentum = protocol["ssl"]["ema_start"] + progress * (protocol["ssl"]["ema_end"] - protocol["ssl"]["ema_start"])
            model.update_target(momentum)
            scheduler.step()
            steps += 1
            losses.append(float(loss.detach()))
            target_std.append(float(output["target"].detach().flatten(0, -2).std(dim=0, unbiased=False).mean()))
            prediction_std.append(float(output["prediction"].detach().flatten(0, -2).std(dim=0, unbiased=False).mean()))
        diagnostic = diagnose(model.encoder, diagnostic_loader, device)
        history.append({"epoch": epoch, "jepa_loss": float(np.mean(losses)), "target_std": float(np.mean(target_std)),
                        "prediction_std": float(np.mean(prediction_std)), "learning_rate": optimizer.param_groups[0]["lr"],
                        "ema_momentum": momentum, "gradient_norm_last": float(norm), "steps": steps,
                        "token_std": diagnostic["token_std_mean"], "embedding_std": diagnostic["embedding_std_mean"],
                        "effective_rank": diagnostic["effective_rank"], "cosine_median": diagnostic["cosine_quantiles_0_05_50_95_100"][2],
                        "elapsed_seconds": prior_seconds + time.monotonic() - started})
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        write_json(folder / "diagnostics_final.json", diagnostic)
        save_training(latest_path, model, optimizer, scheduler, epoch, steps, generator, metadata)
        print(f"SSL {partition} seed={seed} epoch={epoch} loss={history[-1]['jepa_loss']:.4f} rank={diagnostic['effective_rank']:.2f}", flush=True)
    record = metadata | {"epochs": protocol["ssl"]["epochs"], "optimizer_steps": steps,
                          "ssl_sample_exposures": steps * protocol["ssl"]["batch_size"],
                          "parameters": sum(parameter.numel() for parameter in model.parameters()),
                          "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                          "seconds": prior_seconds + time.monotonic() - started,
                          "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                          "encoder_hash": state_hash(model.encoder.state_dict()), "selection": "fixed-final-epoch",
                          "status": "COMPLETED"}
    torch.save({"format": "m4-encoder-v1", "encoder": {name: value.detach().cpu() for name, value in model.encoder.state_dict().items()},
                "metadata": record, "model_config": protocol["model"]}, final_path)
    write_json(folder / "training.json", record)
    return record
