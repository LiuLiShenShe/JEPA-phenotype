from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def snapshot(root):
    destination = root / "reports/m6_a/historical_snapshot.json"
    if destination.exists():
        return
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines())
    paths = set(tracked)
    for folder in ["reports", "outputs", "src", "scripts", "configs", "tests", "splits", "data/processed", "data/metadata"]:
        paths.update(str(path.relative_to(root)) for path in (root / folder).rglob("*") if path.is_file())
    records = {}
    for name in sorted(paths):
        path = root / name
        if not path.is_file() or "__pycache__" in path.parts or "m6_a" in path.parts or "m6a" in path.parts or "m6_a" in path.name:
            continue
        info = path.stat()
        record = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
        if name in tracked:
            record["sha256"] = digest(path)
        records[name] = record
    raw = json.loads((root / "reports/m3/raw_file_manifest.json").read_text())["files"]
    for record in raw:
        if digest(root / record["path"]) != record["sha256"]:
            raise ValueError("Historical raw hash mismatch: " + record["path"])
    write_json(destination, {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                            "files": records, "raw_files": raw,
                            "policy": "all historical artifact size/mtime fingerprints; tracked artifacts and 891 Sugar4D raw files additionally SHA256"})


def verify_protection(root):
    original = json.loads((root / "reports/m6_a/historical_snapshot.json").read_text())
    changes = []
    for name, record in original["files"].items():
        path = root / name
        if not path.is_file():
            changes.append(name)
            continue
        info = path.stat()
        if info.st_size != record["bytes"] or info.st_mtime_ns != record["mtime_ns"] or ("sha256" in record and digest(path) != record["sha256"]):
            changes.append(name)
    for record in original["raw_files"]:
        if not (root / record["path"]).is_file() or digest(root / record["path"]) != record["sha256"]:
            changes.append(record["path"])
    result = {"historical_files": len(original["files"]), "sha256_tracked_files": sum("sha256" in item for item in original["files"].values()),
              "sha256_raw_files": len(original["raw_files"]), "changes": changes,
              "unchanged": not changes, "git_head_unchanged": original["commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    write_json(root / "reports/m6_a/protection_verification.json", result)
    if changes or not result["git_head_unchanged"]:
        raise ValueError("Historical artifacts changed")
    return result


def freeze(root):
    snapshot(root)
    paths = ["configs/m6_a/protocol.json", "reports/m3/plant_splits.json", "reports/m3/raw_file_manifest.json",
             "data/raw/m6_a/official/plantnet/PlantNet/PlantNet_pytorch/models/checkpoints/model_epoch199.pth"]
    hashes = {name: digest(root / name) for name in paths}
    lock = root / "reports/m6_a/protocol_lock.json"
    if lock.exists():
        if json.loads(lock.read_text())["sha256"] != hashes:
            raise ValueError("Frozen input changed; refusing result-driven protocol revision")
        return
    write_json(lock, {"utc_before_inference": datetime.datetime.now(datetime.timezone.utc).isoformat(), "sha256": hashes,
                      "inference_started": False, "training_steps": 0})


def read_cloud(path):
    from m3.common import read_ply

    vertices, info = read_ply(path)
    cloud = np.column_stack([vertices[name] for name in ["x", "y", "z"]]).astype(np.float32)
    labels = np.where((vertices["label_semantic"] == 2) & (vertices["label_instance"] >= 0), vertices["label_instance"] + 1, 0).astype(np.int32)
    if not np.isfinite(cloud).all():
        raise ValueError("Nonfinite current geometry")
    return cloud, labels, info


def normalize(cloud):
    centered = np.asarray(cloud, dtype=np.float64) - np.mean(cloud, axis=0)
    radius = np.linalg.norm(centered, axis=1).max()
    return (centered / max(radius, 1e-8)).astype(np.float32)


def localize_gt(gt, seed):
    unique = np.unique(gt[gt > 0])
    shuffled = np.random.default_rng(seed).permutation(np.arange(1, len(unique) + 1))
    result = np.zeros_like(gt)
    for old, new in zip(unique, shuffled):
        result[gt == old] = new
    return result


def paired_interval(values, seed=20261012):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"mean": None, "ci95_low": None, "ci95_high": None, "plants": 0}
    generator = np.random.default_rng(seed)
    means = finite[generator.integers(0, len(finite), size=(10000, len(finite)))].mean(axis=1)
    return {"mean": float(finite.mean()), "ci95_low": float(np.quantile(means, .025)),
            "ci95_high": float(np.quantile(means, .975)), "plants": len(finite)}
