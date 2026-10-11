"""Read-only legacy audit and no-overwrite artifact contracts."""

from concurrent.futures import ThreadPoolExecutor
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import pandas as pd

from training.common import file_hash, write_json


def save_once(path, payload):
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload and path.name == "training_source_fingerprints.json":
            changes = {key for key in set(existing) | set(payload) if existing.get(key) != payload.get(key)}
            correction_path = path.parent / "inference_precision_correction.json"
            if changes == {"src/m15/engine.py"} and correction_path.exists():
                correction = json.loads(correction_path.read_text())
                root = path.parents[2]
                before = root / correction["original_source_snapshot"]
                after = root / "src/m15/engine.py"
                old_tree, new_tree = ast.parse(before.read_text()), ast.parse(after.read_text())
                old_tree.body = [node for node in old_tree.body if not isinstance(node, ast.FunctionDef) or node.name != "predict"]
                new_tree.body = [node for node in new_tree.body if not isinstance(node, ast.FunctionDef) or node.name != "predict"]
                if (file_hash(before) == existing["src/m15/engine.py"] == correction["original_engine_sha256"]
                        and file_hash(after) == payload["src/m15/engine.py"] == correction["corrected_engine_sha256"]
                        and ast.dump(old_tree) == ast.dump(new_tree)):
                    return
        if existing != payload:
            raise FileExistsError(f"Immutable artifact disagrees: {path}")
        return
    write_json(path, payload)


def snapshot(root):
    root = Path(root)
    destination = root / "reports/m1_5/legacy_snapshot.json"
    if destination.exists():
        return verify_snapshot(root)
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    paths = {path for path in tracked if not path.startswith(("src/m15/", "reports/m1_5/", "configs/m1_5/"))
             and "m1_5" not in path and (root / path).is_file()}
    for folder in ("reports/m0", "reports/m1", "outputs/m1", "data/metadata", "data/processed", "splits"):
        paths.update(str(path.relative_to(root)) for path in (root / folder).rglob("*") if path.is_file())
    records = {path: file_hash(root / path) for path in sorted(paths)}
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip(),
        "git_status_at_snapshot": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=root, text=True),
        "files": records,
    }
    save_once(destination, payload)
    return verify_snapshot(root)


def verify_snapshot(root, *, raw=False):
    root = Path(root)
    payload = json.loads((root / "reports/m1_5/legacy_snapshot.json").read_text())
    changed = [path for path, digest in payload["files"].items()
               if not (root / path).is_file() or file_hash(root / path) != digest]
    if changed:
        raise RuntimeError(f"Protected legacy artifacts changed: {changed[:10]}")
    original = json.loads((root / "reports/m1/m0_snapshot.json").read_text())
    if any(file_hash(root / path) != digest for path, digest in original["files"].items()):
        raise RuntimeError("M0 protected source or data changed")
    raw_count = 0
    if raw:
        manifest = pd.read_csv(root / "data/metadata/file_manifest.csv")
        jobs = [(row.rgb_path, row.rgb_sha256) for row in manifest.itertuples()]
        jobs += [(row.mask_path, row.mask_sha256) for row in manifest.itertuples()]
        def check(job):
            path, expected = job
            return path if file_hash(root / path) != expected else None
        with ThreadPoolExecutor(max_workers=8) as pool:
            mismatches = [path for path in pool.map(check, jobs) if path is not None]
        if mismatches:
            raise RuntimeError(f"M0 raw file mismatch: {mismatches[:10]}")
        raw_count = len(jobs)
    return {"status": "PASS", "protected_files": len(payload["files"]), "changed_files": [],
            "raw_files_checked": raw_count, "commit": payload["commit"], "branch": payload["branch"]}
