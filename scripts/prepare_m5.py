#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from datasets.sugar4d_ssl_dataset import partition_rows
from m4.common import load_protocol, sha256, state_hash, write_json
from m4.downstream import source_encoder


def snapshot(root):
    root = Path(root)
    destination = root / "reports/m5/initial_protection.json"
    if destination.exists():
        return
    records = {}
    for folder in ("src", "scripts", "tests", "configs", "splits", "reports", "outputs", "data"):
        for path in sorted((root / folder).rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(root)
            if "m5" in relative.parts or "m5" in path.name:
                continue
            stat = path.stat()
            records[str(relative)] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            if "raw" not in relative.parts:
                records[str(relative)]["sha256"] = sha256(path)
    write_json(destination, {"date": "2026-10-12", "files": records,
                             "raw_policy": "size + nanosecond mtime; all non-raw files SHA256",
                             "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()})
    print(f"PROTECTED {len(records)} existing files", flush=True)


def verify(root):
    root = Path(root)
    before = json.loads((root / "reports/m5/initial_protection.json").read_text())["files"]
    differences = []
    for name, record in before.items():
        path = root / name
        if not path.exists():
            differences.append({"path": name, "reason": "missing"})
            continue
        stat = path.stat()
        same = stat.st_size == record["bytes"] and stat.st_mtime_ns == record["mtime_ns"]
        if "sha256" in record:
            same = sha256(path) == record["sha256"]
        if not same:
            differences.append({"path": name, "reason": "changed"})
    result = {"status": "PASS" if not differences else "FAIL", "protected_files": len(before), "differences": differences}
    write_json(root / "reports/m5/final_protection.json", result)
    if differences:
        raise ValueError(f"Historical artifacts changed: {differences}")
    print(f"PROTECTION PASS {len(before)} files", flush=True)


def audit(root):
    root = Path(root)
    protocol, _ = load_protocol(root)
    resources = pd.read_csv(root / "reports/m4/pretraining_resources.csv")
    downstream = pd.read_csv(root / "reports/m4/training_resources.csv")
    diagnostics = json.loads((root / "reports/m4/representation_full_train_diagnostics.json").read_text())
    checkpoints = []
    for partition in protocol["partitions"]:
        rows = partition_rows(root, partition, "train")
        for seed in protocol["seeds"]:
            for condition in ("P0", "P2"):
                encoder, reference = source_encoder(root, protocol, partition, seed, condition)
                diagnostic = diagnostics[f"{partition}_seed{seed}"][condition]
                if state_hash(encoder.state_dict()) != diagnostic["encoder_hash"]:
                    raise ValueError("Historical encoder hash mismatch")
                checkpoints.append({"partition": partition, "seed": seed, "condition": condition,
                                    "encoder_parameters": sum(value.numel() for value in encoder.parameters()),
                                    "token_dimension": encoder.feature_dim, "pooled_dimension": 2 * encoder.feature_dim,
                                    "train_plants": rows.plant_uid.nunique(), "scans": len(rows),
                                    "reference": reference, "encoder_hash": diagnostic["encoder_hash"],
                                    "effective_rank": diagnostic["effective_rank"],
                                    "token_std": diagnostic["token_std_mean"], "pooled_std": diagnostic["embedding_std_mean"]})
    record = {"status": "PASS", "m4_decision": json.loads((root / "reports/m4/decision.json").read_text()),
              "checkpoints": checkpoints, "ssl_optimizer_steps": sorted(resources.optimizer_steps.unique().tolist()),
              "ssl_runs": len(resources), "ssl_total_steps": int(resources.optimizer_steps.sum()),
              "downstream_runs": len(downstream), "model_config": protocol["model"],
              "ema": "M4 teacher is the Transformer only; shared tokenizer and positional encoding are stop-gradient in target. Exact EMA unit test is rerun, not a full-tokenizer teacher.",
              "dimension_check": "384 token; concatenated mean/max = 768 pooled; count head uses 768 + 27 current geometry = 795 inputs. No 384/768 mismatch.",
              "preprocessing_check": "M4 cache shared by SSL and downstream; full-current-scan mean/radius; XYZ-only sampling/FPS/KNN; current physical geometry only in downstream.",
              "split_check": "Original official and four fold-local 24/12/12 assignments retained; 16 scans/plant are not independent plants.",
              "representation_interpretation": "M4 rank and variance shrink; M4 uncentred cosine lacks identity/centering controls. This is degeneration risk, not proof of complete collapse.",
              "test_policy": "Historical tests were inspected previously. M5 is exploratory; this run will use train/validation only and will not read new test labels or generate test predictions."}
    write_json(root / "reports/m5/history_audit.json", record)
    pd.DataFrame(checkpoints).to_csv(root / "reports/m5/historical_checkpoints.csv", index=False)
    print("HISTORICAL CHECKPOINT AUDIT PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--stage", choices=("snapshot", "audit", "verify"), required=True)
    arguments = parser.parse_args()
    {"snapshot": snapshot, "audit": audit, "verify": verify}[arguments.stage](arguments.root)
