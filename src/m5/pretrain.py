from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.sugar4d_ssl_dataset import Sugar4DSSLDataset, partition_rows
from m4.common import load_protocol, restore_rng, rng_state, seed_all, sha256, state_hash, write_json
from m4.pretrain import rotate_z
from m5.diagnostics import degradation_stop, diagnose_encoder
from models.point_jepa_predictor import PointJEPA


def regularization(features, specification):
    features = features.reshape(-1, features.shape[-1])
    centered = features - features.mean(0)
    variance = torch.relu(specification["standard_deviation_floor"] -
                          torch.sqrt(centered.square().mean(0) + specification["epsilon"])).mean()
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    covariance_loss = off_diagonal.square().sum() / features.shape[-1]
    return specification["variance_weight"] * variance + specification["covariance_weight"] * covariance_loss, variance, covariance_loss


def save_checkpoint(path, model, optimizer, scheduler, sampler, step, metadata, history):
    state = {"format": "m5-ssl-training-v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "sampler_rng": sampler.get_state(), "rng": rng_state(),
             "step": step, "metadata": metadata, "history": history}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(root, partition, seed, condition, specification, dataset, diagnostic_dataset, device, formal_lock=None):
    root = Path(root)
    protocol, _ = load_protocol(root)
    folder = root / "outputs/m5/checkpoints" / partition / f"{condition}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    final_path, last_path = folder / "encoder.pt", folder / "last.pt"
    if final_path.exists():
        record = json.loads((folder / "training.json").read_text())
        expected_lock = sha256(root / "configs/m5/budget_diagnostic.json") if formal_lock is None else formal_lock
        if record["specification_sha256"] != expected_lock:
            raise ValueError("Existing SSL protocol mismatch")
        return record
    seed_all(seed)
    model = PointJEPA(protocol["model"], protocol["ssl"]).to(device)
    sampler = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad],
                                  lr=specification["learning_rate"], weight_decay=specification["weight_decay"])
    total_steps, warmup = specification["maximum_steps"], specification["warmup_steps"]

    def schedule(step):
        if step < warmup:
            return max(.01, (step + 1) / warmup)
        return .001 + .999 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))) / 2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    train_rows = partition_rows(root, partition, "train")
    metadata = {"partition": partition, "seed": seed, "condition": condition,
                "train_plants": sorted(train_rows.plant_uid.unique()), "validation_used": False, "test_used": False,
                "architecture": protocol["model"], "initial_encoder_hash": state_hash(model.encoder.state_dict()),
                "specification_sha256": sha256(root / "configs/m5/budget_diagnostic.json") if formal_lock is None else formal_lock,
                "sampling": "uniform with replacement over train scans for Sugar4D; balanced source/species sampler for expanded data"}
    history, steps, prior_seconds = [], 0, 0.
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if checkpoint["metadata"]["specification_sha256"] != metadata["specification_sha256"]:
            raise ValueError("SSL resume protocol mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        sampler.set_state(checkpoint["sampler_rng"])
        restore_rng(checkpoint["rng"])
        history, steps, metadata = checkpoint["history"], checkpoint["step"], checkpoint["metadata"]
        if hasattr(dataset, "exposures"):
            dataset.exposures = metadata.get("sampling_exposures", dataset.exposures)
        prior_seconds = history[-1]["elapsed_seconds"] if history else 0.
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    diagnostic_ids = train_rows.plant_uid.to_numpy()
    initial_path = folder / "diagnostics_step0.json"
    if not initial_path.exists():
        initial = diagnose_encoder(model.encoder, diagnostic_dataset, diagnostic_ids, device)
        write_json(initial_path, initial)
    initial = json.loads(initial_path.read_text())
    previous = initial
    for checkpoint_step in sorted(specification["checkpoints"]):
        path = folder / f"diagnostics_step{checkpoint_step}.json"
        if path.exists() and 0 < checkpoint_step <= steps:
            previous = json.loads(path.read_text())
    checkpoints = set(specification["checkpoints"])
    stopped = False
    for steps in range(steps + 1, total_steps + 1):
        model.train()
        indices = dataset.sample_indices(specification["batch_size"], sampler) if hasattr(dataset, "sample_indices") else torch.randint(len(dataset), (specification["batch_size"],), generator=sampler).tolist()
        samples = [dataset[index] for index in indices]
        batch = {name: torch.stack([sample[name] for sample in samples]).to(device)
                 for name in ("points", "center_indices", "neighbors")}
        batch["points"] = rotate_z(batch["points"])
        optimizer.zero_grad(set_to_none=True)
        student_outputs = []
        hook = model.encoder.student.register_forward_hook(lambda module, inputs, features: student_outputs.append(features)) if condition == "P4" else None
        output = model(**batch)
        if hook is not None:
            hook.remove()
        variance, covariance, penalty = output["loss"].new_zeros(()), output["loss"].new_zeros(()), output["loss"].new_zeros(())
        if condition == "P4":
            penalty, variance, covariance = regularization(student_outputs[-1], specification["p4_candidate"])
        loss = output["loss"] + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SSL loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_([value for value in model.parameters() if value.requires_grad], 5., error_if_nonfinite=True)
        optimizer.step()
        momentum = protocol["ssl"]["ema_start"] + (steps - 1) / max(1, total_steps - 1) * (protocol["ssl"]["ema_end"] - protocol["ssl"]["ema_start"])
        model.update_target(momentum)
        scheduler.step()
        history.append({"step": steps, "ssl_loss": float(output["loss"].detach()), "total_loss": float(loss.detach()),
                        "variance_loss": float(variance.detach()), "covariance_loss": float(covariance.detach()),
                        "learning_rate": optimizer.param_groups[0]["lr"], "ema": momentum,
                        "gradient_norm": float(norm), "elapsed_seconds": prior_seconds + time.monotonic() - started})
        if steps % 100 == 0:
            print(f"SSL {partition} {condition} seed={seed} step={steps} loss={np.mean([entry['ssl_loss'] for entry in history[-100:]]):.5f}", flush=True)
        if steps in checkpoints:
            diagnostic = diagnose_encoder(model.encoder, diagnostic_dataset, diagnostic_ids, device)
            diagnostic.update({"step": steps, "ssl_loss_last_100": float(np.mean([entry["ssl_loss"] for entry in history[-100:]])),
                               "encoder_hash": state_hash(model.encoder.state_dict()), "condition": condition, "partition": partition, "seed": seed})
            write_json(folder / f"diagnostics_step{steps}.json", diagnostic)
            stopped = condition != "P4" and degradation_stop(diagnostic, previous, initial, steps)
            if hasattr(dataset, "exposures"):
                metadata["sampling_exposures"] = dataset.exposures
            save_checkpoint(folder / f"step{steps}.pt", model, optimizer, scheduler, sampler, steps, metadata, history)
            torch.save({"format": "m4-encoder-v1", "encoder": {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()},
                        "metadata": metadata | {"optimizer_steps": steps}, "model_config": protocol["model"]}, folder / f"encoder_step{steps}.pt")
            previous = diagnostic
            print(f"DIAGNOSTIC {condition} step={steps} pooled_rank={diagnostic['pooled']['effective_rank']:.2f} token_std={diagnostic['token']['std_mean']:.5f} stop={stopped}", flush=True)
        if steps % 100 == 0 or steps in checkpoints or steps == total_steps:
            if hasattr(dataset, "exposures"):
                metadata["sampling_exposures"] = dataset.exposures
            save_checkpoint(last_path, model, optimizer, scheduler, sampler, steps, metadata, history)
            pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        if stopped:
            break
    if int(model.ema_updates) != steps:
        raise ValueError("EMA update count mismatch")
    record = metadata | {"optimizer_steps": steps, "planned_steps": total_steps, "early_stopped": stopped,
                         "ssl_sample_exposures": steps * specification["batch_size"], "batch_size": specification["batch_size"],
                         "seconds": prior_seconds + time.monotonic() - started, "encoder_hash": state_hash(model.encoder.state_dict()),
                         "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                         "parameters": sum(value.numel() for value in model.parameters()), "status": "COMPLETED",
                         "dataset_size": len(dataset), "selection": "locked step or training-only degeneration stop, never downstream"}
    torch.save({"format": "m4-encoder-v1", "encoder": {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()},
                "metadata": record, "model_config": protocol["model"]}, final_path)
    write_json(folder / "training.json", record)
    if hasattr(dataset, "exposures"):
        write_json(folder / "sampling_exposures.json", dataset.exposures)
    return record


def budget_diagnostic(root, device):
    root = Path(root)
    specification = json.loads((root / "configs/m5/budget_diagnostic.json").read_text())
    rows = partition_rows(root, specification["partition"], "train")
    dataset = Sugar4DSSLDataset(rows, root / "outputs/m4/data/xyz")
    return train(root, specification["partition"], specification["seed"], "M5A", specification, dataset, dataset, device)


def formal_pretrain(root, partition, seed, condition, device):
    from datasets.plant_ssl_dataset import BalancedPlantSSLDataset
    from m5.common import load_protocol as m5_protocol

    root = Path(root)
    protocol, lock = m5_protocol(root)
    if condition not in ("S500", "P3", "P4") or condition not in protocol["conditions"]:
        raise ValueError("Unregistered formal SSL condition")
    if partition not in protocol["partitions"] or seed not in protocol["seeds"]:
        raise ValueError("Unregistered formal SSL partition/seed")
    if condition == "P4":
        risk = json.loads((root / "outputs/m5/checkpoints/official/M5A_seed17/diagnostics_step500.json").read_text())
        initial = json.loads((root / "outputs/m5/checkpoints/official/M5A_seed17/diagnostics_step0.json").read_text())
        triggered = ((risk["token"]["std_mean"] <= .5 * initial["token"]["std_mean"] or
                      risk["pooled"]["std_mean"] <= .5 * initial["pooled"]["std_mean"]) and
                     risk["pooled"]["effective_rank"] <= .9 * initial["pooled"]["effective_rank"])
        if not triggered:
            raise ValueError("P4 was not triggered by M5-A risk")
    rows = partition_rows(root, partition, "train")
    sugar = Sugar4DSSLDataset(rows, root / "outputs/m4/data/xyz")
    if condition == "S500":
        dataset = sugar
    else:
        external = pd.read_csv(root / "reports/m5/external_data_manifest.csv")
        dataset = BalancedPlantSSLDataset(sugar, external, root / "outputs/m5/data/external_xyz", protocol["ssl"]["source_proportions"])
    return train(root, partition, seed, condition, protocol["ssl"], dataset, sugar, device, lock["protocol_sha256"])
