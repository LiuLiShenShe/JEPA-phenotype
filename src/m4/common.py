from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def save_training(path, model, optimizer, scheduler, epoch, steps, generator, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({"format": "m4-training-v1", "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "steps": steps, "rng": rng_state(),
                "loader_rng": generator.get_state(), "metadata": metadata}, temporary)
    temporary.replace(path)


def resume_training(path, model, optimizer, scheduler, generator):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "m4-training-v1":
        raise ValueError("Not a complete M4 training checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    generator.set_state(checkpoint["loader_rng"])
    restore_rng(checkpoint["rng"])
    return checkpoint


def load_protocol(root):
    root = Path(root)
    path = root / "configs/m4/protocol.json"
    lock = json.loads((root / "reports/m4/protocol_lock.json").read_text())
    if sha256(path) != lock["protocol_sha256"]:
        raise ValueError("Locked M4 protocol changed; do not silently retrain")
    for name, digest in lock["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Locked input changed: {name}")
    return json.loads(path.read_text()), lock


def artifact_protocol_matches(root, digest, lock):
    if digest == lock["protocol_sha256"]:
        return True
    archive = Path(root) / "reports/m4/sources/protocol_annotations_only.json"
    if not archive.exists() or sha256(archive) != digest:
        return False
    annotated = json.loads(archive.read_text())
    annotated.pop("notes", None)
    original = json.loads((Path(root) / "configs/m4/protocol.json").read_text())
    return annotated == original


def standardize_fit(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": values.mean(axis=0).tolist(),
            "scale": np.maximum(values.std(axis=0), 1e-6).tolist(), "fit_split": "train"}


def standardize(values, state):
    if state["fit_split"] != "train":
        raise ValueError("Non-training normalization is forbidden")
    return ((np.asarray(values) - np.asarray(state["mean"])) / np.asarray(state["scale"])).astype(np.float32)


@torch.no_grad()
def representation_diagnostics(tokens, pooled):
    flattened = tokens.detach().float().cpu().reshape(-1, tokens.shape[-1])
    embedding = pooled.detach().float().cpu()
    deviations = flattened.std(dim=0, unbiased=False)
    normalized = torch.nn.functional.normalize(flattened, dim=-1)
    generator = torch.Generator().manual_seed(20261012)
    first = torch.randint(len(flattened), (min(4096, len(flattened) * 4),), generator=generator)
    second = torch.randint(len(flattened), first.shape, generator=generator)
    second = (second + (first == second).long()) % len(flattened)
    similarities = (normalized[first] * normalized[second]).sum(dim=-1)
    centered = embedding - embedding.mean(dim=0)
    singular = torch.linalg.svdvals(centered)
    probabilities = singular.square() / singular.square().sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
    if float(singular.sum()) < 1e-10:
        effective_rank = 0.0
    quantiles = torch.quantile(similarities, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])).tolist()
    embedding_normalized = torch.nn.functional.normalize(embedding, dim=-1)
    embedding_cosines = (embedding_normalized @ embedding_normalized.T)[torch.triu(torch.ones(len(embedding), len(embedding), dtype=torch.bool), diagonal=1)]
    embedding_quantiles = torch.quantile(embedding_cosines, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0])).tolist()
    embedding_std = embedding.std(dim=0, unbiased=False)
    return {"token_std_per_dimension": deviations.tolist(), "token_std_mean": float(deviations.mean()),
            "embedding_std_per_dimension": embedding_std.tolist(), "embedding_std_mean": float(embedding_std.mean()),
            "fraction_token_std_below_0_01": float((deviations < 0.01).float().mean()),
            "cosine_quantiles_0_05_50_95_100": quantiles,
            "token_cosine_histogram_counts": torch.histc(similarities, bins=40, min=-1, max=1).int().tolist(),
            "embedding_cosine_quantiles_0_05_50_95_100": embedding_quantiles,
            "embedding_cosine_histogram_counts": torch.histc(embedding_cosines, bins=40, min=-1, max=1).int().tolist(),
            "cosine_histogram_edges": torch.linspace(-1, 1, 41).tolist(), "effective_rank": effective_rank,
            "algebraic_rank": int(torch.linalg.matrix_rank(centered)),
            "max_rank": min(len(embedding) - 1, embedding.shape[-1]),
            "singular_values": singular.tolist(), "samples": len(embedding),
            "collapse_warning": bool(float(deviations.mean()) < 0.01 or effective_rank < 2)}
