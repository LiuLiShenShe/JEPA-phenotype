#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from datasets.sugar4d_ssl_dataset import current_transform, labelled_plants
from m3.common import read_ply, xyz_from_vertices
from m4.common import load_protocol, sha256, write_json
from models.point_jepa_encoder import patch_indices


def protection(root):
    result = {}
    for folder in ("src", "scripts", "tests", "configs", "splits", "reports", "outputs", "data"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or "m4" in path.relative_to(root).parts:
                continue
            relative = str(path.relative_to(root))
            if path.name.startswith(("prepare_m4", "pretrain_m4", "train_m4", "evaluate_m4", "verify_m4", "run_m4")) or "point_jepa" in path.name or path.name in ("sugar4d_ssl_dataset.py", "test_m4.py"):
                continue
            stat = path.stat()
            result[relative] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if "raw" not in path.parts:
                result[relative]["sha256"] = sha256(path)
    return result


def lock(root):
    protocol_path = root / "configs/m4/protocol.json"
    lock_path = root / "reports/m4/protocol_lock.json"
    if lock_path.exists():
        load_protocol(root)
        return
    if list((root / "outputs/m4/checkpoints").rglob("*.pt")):
        raise ValueError("Cannot preregister after model training has started")
    protocol = json.loads(protocol_path.read_text())
    splits = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]
    budgets = {}
    for partition, assignments in splits.items():
        plants = [plant for plant, split in assignments.items() if split == "train"]
        budgets[partition] = {str(fraction): labelled_plants(plants, fraction, partition, protocol["label_budget_seed"])
                              for fraction in protocol["label_fractions"]}
        if len(plants) != 24:
            raise ValueError("Unexpected old plant split")
    inputs = {name: sha256(root / name) for name in ("reports/m3/plant_splits.json", "data/processed/sugar4d/plant_metadata.csv",
                                                    "outputs/m3_5/data/geometry.csv")}
    write_json(root / "reports/m4/initial_protection.json", protection(root))
    write_json(lock_path, {"date": "2026-10-12", "protocol_sha256": sha256(protocol_path), "inputs": inputs,
                           "label_budgets": budgets, "test_opened": False,
                           "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()})
    print("LOCKED protocol, nested plant budgets and original-artifact snapshot", flush=True)


def build_cache(root, device):
    protocol, _ = load_protocol(root)
    gate = json.loads((root / "reports/m4/m4a_correctness.json").read_text())
    if gate["status"] != "PASS":
        raise ValueError("M4-A must pass before Sugar4D preprocessing/training")
    rows = pd.read_csv(root / "data/processed/sugar4d/plant_metadata.csv")
    xyz_root, label_root = root / "outputs/m4/data/xyz", root / "outputs/m4/data/labels"
    xyz_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for index, row in enumerate(rows.itertuples(index=False)):
        name = Path(row.pointcloud_path).stem
        destination, label_path = xyz_root / f"{name}.npz", label_root / f"{name}.npz"
        if not destination.exists() or not label_path.exists():
            vertices, _ = read_ply(root / row.pointcloud_path)
            full_xyz = xyz_from_vertices(vertices)
            normalized, center, radius = current_transform(full_xyz)
            identity_seed = int(sha256_bytes(f"{protocol['sampling_seed']}:{name}"), 16)
            indices = np.random.default_rng(identity_seed).choice(len(full_xyz), protocol["points"], replace=len(full_xyz) < protocol["points"])
            points = normalized[indices].astype(np.float32)
            selected, neighbors = patch_indices(torch.from_numpy(points)[None].to(device), protocol["model"]["num_groups"], protocol["model"]["group_size"])
            np.savez(destination, points=points, center_m=center, radius_m=np.array(radius, dtype=np.float32),
                     extent_m=np.ptp(full_xyz, axis=0).astype(np.float32), source_indices=indices,
                     center_indices=selected[0].cpu().numpy(), neighbors=neighbors[0].cpu().numpy())
            semantic = np.asarray(vertices["label_semantic"])[indices].astype(np.int64)
            if not np.isin(semantic, (0, 1, 2)).all():
                raise ValueError("Unexpected semantic label")
            annotated_count = len(np.unique(np.asarray(vertices["label_instance"])[np.asarray(vertices["label_semantic"]) == 2]))
            if annotated_count != row.visible_organ_count:
                raise ValueError("Current visible annotated leaf count disagrees with PLY")
            np.savez(label_path, semantic=semantic)
        manifests.append({"plant_uid": row.plant_uid, "DAS": row.DAS, "scan": name,
                          "xyz_sha256": sha256(destination), "label_sha256": sha256(label_path)})
        if (index + 1) % 96 == 0:
            print(f"CACHE {index + 1}/{len(rows)}", flush=True)
    pd.DataFrame(manifests).to_csv(root / "reports/m4/data_cache_manifest.csv", index=False)
    write_json(root / "reports/m4/data_contract.json", {
        "scans": len(rows), "independent_plants": rows.plant_uid.nunique(), "unit": "metre",
        "ssl_fields": ["points", "center_indices", "neighbors", "center_m", "radius_m", "extent_m"],
        "semantic_mapping": {"0": "taproot", "1": "core", "2": "leaf"},
        "transform": "normalized=(current_xyz-current_full_scan_mean)/current_full_scan_radius",
        "inverse": "current_xyz=normalized*radius_m+center_m", "future_transform": False,
        "cache_note": "All scans can be cached for separate fold assignments; cache presence is not training use. Datasets and checkpoint manifests enforce fold-local Train-only SSL, and Train/Val-only downstream.",
        "sampling": "fixed XYZ-only 2048 uniform sample; deterministic FPS 64, KNN 32; no annotation-stratified input sampling",
        "segmentation_scope": "2048 fixed sampled points per scan, not full-resolution IoU",
        "physical_head_inputs": "unchanged M3.5 current full-scan XYZ geometry descriptors, normalized on labelled training plants only"})


def sha256_bytes(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()[:16]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--stage", choices=("lock", "cache", "verify"), required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve()
    if arguments.stage == "lock":
        lock(root)
    elif arguments.stage == "cache":
        build_cache(root, torch.device(arguments.device))
    else:
        before = json.loads((root / "reports/m4/initial_protection.json").read_text())
        after = protection(root)
        differences = [name for name, record in before.items() if after.get(name) != record]
        write_json(root / "reports/m4/final_protection.json", {"status": "PASS" if not differences else "FAIL",
                                                              "protected_files": len(before), "differences": differences,
                                                              "raw_protection": "size and nanosecond mtime; all non-raw protected files SHA256"})
        if differences:
            raise ValueError(f"Original artifacts changed: {differences}")
