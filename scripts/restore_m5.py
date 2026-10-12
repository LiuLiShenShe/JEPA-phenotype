#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m4.common import sha256, write_json
from m5.common import load_protocol
from m5.downstream import verify_restore


def artifacts(folder):
    record = json.loads((folder / "training.json").read_text())
    prediction = folder / ("predictions.csv" if record["task"] == "count" else "predictions.npz")
    return {"checkpoint_sha256": sha256(folder / "best.pt"), "prediction_sha256": sha256(prediction)}


def process_running(identifier):
    if identifier is None:
        return False
    try:
        os.kill(identifier, 0)
    except ProcessLookupError:
        return False
    return True


def restore(root, device, watch_pid=None):
    root = Path(root)
    protocol, lock = load_protocol(root)
    output = root / "reports/m5/checkpoint_restore.json"
    expected = len(protocol["partitions"]) * len(protocol["seeds"]) * sum(
        condition not in ("P0", "P2") for condition in protocol["conditions"]) * 4
    records = {}
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("protocol_sha256") != lock["protocol_sha256"]:
            raise ValueError("Restore protocol mismatch")
        records = {record["id"]: record for record in previous["records"]}
    while True:
        paths = sorted((root / "outputs/m5/downstream").glob("*/training.json"))
        for path in paths:
            hashes = artifacts(path.parent)
            if path.parent.name in records:
                if any(records[path.parent.name].get(name) != value for name, value in hashes.items()):
                    raise ValueError("Previously restored artifacts changed")
                continue
            record = verify_restore(root, path.parent, device) | hashes
            records[record["id"]] = record
            complete = len(records) == expected
            temporary = output.with_suffix(".tmp")
            write_json(temporary, {"status": "PASS" if complete else "IN_PROGRESS", "checkpoints": len(records),
                       "expected_checkpoints": expected, "protocol_sha256": lock["protocol_sha256"],
                       "records": sorted(records.values(), key=lambda row: row["id"])})
            temporary.replace(output)
            print(f"RESTORE {len(records)}/{expected} {record['id']} PASS", flush=True)
        if len(records) == expected:
            return
        if not process_running(watch_pid):
            raise ValueError(f"Incomplete checkpoint matrix: {len(records)}/{expected}")
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--watch-pid", type=int)
    arguments = parser.parse_args()
    restore(arguments.root, torch.device(arguments.device), arguments.watch_pid)
