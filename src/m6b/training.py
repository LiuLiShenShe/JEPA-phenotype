from __future__ import annotations

import csv
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from m6a.common import digest, write_json
from m6b.common import check_lock
from m6b.data import TrainingDataset
from m6b.evaluation import evaluate_model, run_dir
from m6b.models import official_joint_loss, official_model


def save_checkpoint(path, model, optimizer, epoch, best, lock_hash):
    temporary = path.with_suffix(".temporary")
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch, "best_val_pq": best, "protocol_lock_sha256": lock_hash,
                "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}, temporary)
    temporary.replace(path)


def train(root, method, seed, device):
    lock_hash = check_lock(root)
    protocol = json.loads((root / "configs/m6_b/protocol.json").read_text())
    config = protocol["training"]
    seeds = config["S3_seeds"] if method == "S3" else config["seeds"]
    if seed not in seeds:
        raise ValueError("Seed not in frozen protocol")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, loading = official_model(root, "PSegNet" if method == "S3" else "PlantNet", method == "S2", device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"][method], weight_decay=config["weight_decay"])
    directory = run_dir(root, method, seed)
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / "latest.pth"
    if not (directory / "training_manifest.json").exists():
        loading.update({"method": method, "seed": seed, "config": config, "protocol_lock_sha256": lock_hash,
                        "training_plants": 24, "training_frames": 384, "labels": "all current-frame semantics and local leaf masks",
                        "external_pretraining": "source tobacco/tomato/sorghum epoch199; individual biological plant count unknown" if method == "S2" else "none",
                        "code_sha256": {name: digest(root / name) for name in ["src/m6b/models.py", "src/m6b/data.py", "src/m6b/training.py", "src/m6b/evaluation.py"]}})
        write_json(directory / "training_manifest.json", loading)
    dataset = TrainingDataset(root, seed, protocol["sampling"]["points"])
    start_epoch, best = 0, -1.0
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=True)
        if state["protocol_lock_sha256"] != lock_hash:
            raise ValueError("Resume protocol mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch, best = state["epoch"], state["best_val_pq"]
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    fields = ["epoch", "batch", "optimizer_step", "loss", "semantic", "discriminative", "similarity", "semantic_accuracy", "lr", "seconds"]
    batch_file = directory / "batches.csv"
    with batch_file.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        for epoch_index in range(start_epoch, config["epochs"]):
            started = time.perf_counter()
            dataset.epoch = epoch_index
            generator = torch.Generator().manual_seed(seed + epoch_index * 1009)
            loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0,
                                drop_last=False, generator=generator)
            learning_rate = .00002 + .5 * (config["learning_rate"][method] - .00002) * (1 + math.cos(math.pi * epoch_index / config["epochs"]))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            model.train()
            losses = []
            for batch_index, (xyz, semantic, instances) in enumerate(loader):
                batch_started = time.perf_counter()
                xyz, semantic, instances = xyz.to(device), semantic.to(device), instances.to(device)
                logits, embedding, similarity = model(xyz)
                loss, components = official_joint_loss(logits, embedding, similarity, semantic, instances)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss; no samples silently removed")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
                row = {"epoch": epoch_index + 1, "batch": batch_index + 1,
                       "optimizer_step": epoch_index * len(loader) + batch_index + 1, "loss": float(loss.detach()),
                       **{name: float(value.detach()) for name, value in components.items()},
                       "semantic_accuracy": float((logits.argmax(-1) == semantic).float().mean()),
                       "lr": learning_rate, "seconds": time.perf_counter() - batch_started}
                writer.writerow(row)
                losses.append(row)
            stream.flush()
            epoch = epoch_index + 1
            summary = {"epoch": epoch, "training_seconds": time.perf_counter() - started,
                       **{name: float(np.mean([row[name] for row in losses])) for name in ["loss", "semantic", "discriminative", "similarity", "semantic_accuracy"]}}
            if epoch in config["validation_epochs"]:
                rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                frames, plants, means = evaluate_model(root, model, device, "val", (.6,))
                torch.set_rng_state(rng)
                torch.cuda.set_rng_state_all(cuda_rng)
                frames.to_csv(directory / f"validation_epoch_{epoch:03d}.csv", index=False)
                score = float(means.pq.iloc[0])
                summary.update({"val_pq": score, "val_sbd": float(means.sbd.iloc[0]), "val_plants": len(plants)})
                if score > best:
                    best = score
                    save_checkpoint(directory / "best.pth", model, optimizer, epoch, best, lock_hash)
                save_checkpoint(directory / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best, lock_hash)
            save_checkpoint(latest, model, optimizer, epoch, best, lock_hash)
            summary["best_val_pq"] = best
            with (directory / "epochs.jsonl").open("a") as log:
                log.write(json.dumps(summary) + "\n")
            print(method, seed, json.dumps(summary), flush=True)
    write_json(directory / "training_complete.json", {"method": method, "seed": seed, "epochs": config["epochs"],
               "optimizer_steps": config["epochs"] * len(dataset) // config["batch_size"],
               "frame_presentations": config["epochs"] * len(dataset), "independent_train_plants": 24,
               "best_val_pq": best, "latest_sha256": digest(latest), "best_sha256": digest(directory / "best.pth"),
               "protocol_lock_sha256": lock_hash})
