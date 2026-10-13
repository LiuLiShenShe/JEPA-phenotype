from __future__ import annotations

import datetime
import json
import subprocess
from pathlib import Path

from m6a.common import digest, write_json


def snapshot(root):
    destination = root / "reports/m6_b/historical_snapshot.json"
    if destination.exists():
        return
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines())
    records = {}
    for folder in ["reports", "outputs", "src", "scripts", "configs", "tests", "splits", "data", ".venv", ".venv-m1", ".venv-m4", ".venv-m6-a"]:
        for path in (root / folder).rglob("*"):
            name = str(path.relative_to(root))
            if not path.is_file() or any(part in {"m6_b", "m6b", "__pycache__"} for part in path.parts) or "m6_b" in path.name:
                continue
            info = path.stat()
            record = {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}
            if name in tracked:
                record["sha256"] = digest(path)
            records[name] = record
    raw = json.loads((root / "reports/m3/raw_file_manifest.json").read_text())["files"]
    write_json(destination, {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                            "files": records, "raw_files": raw,
                            "policy": "historical size/mtime, tracked SHA256, raw manifest SHA256; environment files included"})


def verify(root):
    original = json.loads((root / "reports/m6_b/historical_snapshot.json").read_text())
    changes = []
    for name, record in original["files"].items():
        path = root / name
        if not path.is_file():
            changes.append(name)
            continue
        info = path.stat()
        if info.st_size != record["bytes"] or info.st_mtime_ns != record["mtime_ns"] or ("sha256" in record and digest(path) != record["sha256"]):
            changes.append(name)
    raw_changes = [record["path"] for record in original["raw_files"]
                   if not (root / record["path"]).is_file() or digest(root / record["path"]) != record["sha256"]]
    result = {"historical_files": len(original["files"]), "tracked_sha256_files": sum("sha256" in value for value in original["files"].values()),
              "raw_sha256_files": len(original["raw_files"]), "changes": changes, "raw_changes": raw_changes,
              "unchanged": not changes and not raw_changes,
              "git_head_unchanged": original["commit"] == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    write_json(root / "reports/m6_b/protection_verification.json", result)
    if not result["unchanged"] or not result["git_head_unchanged"]:
        raise ValueError("Historical protection failed")
    return result


def freeze(root):
    snapshot(root)
    paths = ["configs/m6_b/protocol.json", "configs/m6_a/protocol.json", "reports/m3/plant_splits.json",
             "reports/m3/raw_file_manifest.json", "reports/m6_b/sources/retrieval_audit.json"]
    hashes = {name: digest(root / name) for name in paths}
    for name in ["PlantNet", "PSegNet"]:
        path = root / f"data/raw/m6_a/official/plantnet/{name}/{name}_pytorch/models/checkpoints/model_epoch199.pth"
        hashes[str(path.relative_to(root))] = digest(path)
    lock = root / "reports/m6_b/protocol_lock.json"
    if lock.exists():
        if json.loads(lock.read_text())["sha256"] != hashes:
            raise ValueError("Frozen inputs changed")
    else:
        write_json(lock, {"utc_before_training": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "sha256": hashes, "supervised_training_started": False})
    return digest(lock)


def check_lock(root):
    path = root / "reports/m6_b/protocol_lock.json"
    content = json.loads(path.read_text())
    for name, expected in content["sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen input changed: " + name)
    return digest(path)
