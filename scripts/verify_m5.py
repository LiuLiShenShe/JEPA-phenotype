#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from m4.common import sha256, state_hash, write_json
from m5.common import load_protocol
from m5.downstream import source_encoder


def verify(root):
    root = Path(root)
    protocol, lock = load_protocol(root)
    records = []
    for partition in protocol["partitions"]:
        for seed in protocol["seeds"]:
            for condition in ("S500", "P3", "P4"):
                folder = root / "outputs/m5/checkpoints" / partition / f"{condition}_seed{seed}"
                record = json.loads((folder / "training.json").read_text())
                if record["optimizer_steps"] != protocol["ssl"]["maximum_steps"] or record["batch_size"] != protocol["ssl"]["batch_size"]:
                    raise ValueError("Unmatched optimizer budget")
                if record["test_used"] or record["validation_used"]:
                    raise ValueError("Nontraining SSL use")
                encoder, _ = source_encoder(root, protocol, partition, seed, condition)
                if state_hash(encoder.state_dict()) != record["encoder_hash"]:
                    raise ValueError("Final encoder tensor hash mismatch")
                state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
                if int(state["model"]["ema_updates"]) != record["optimizer_steps"]:
                    raise ValueError("EMA update count is not one per optimizer step")
                if not all(field in state for field in ("optimizer", "scheduler", "rng", "sampler_rng", "history")):
                    raise ValueError("Incomplete SSL resume state")
                if condition in ("P3", "P4"):
                    exposure = record["sampling_exposures"]
                    total = sum(sum(values.values()) for values in exposure.values())
                    if total != record["ssl_sample_exposures"]:
                        raise ValueError("Source exposures mismatch")
                    for source, target in protocol["ssl"]["source_proportions"].items():
                        actual = sum(exposure[source].values()) / total
                        if abs(actual - target) > .015:
                            raise ValueError("Unbalanced actual data source sampling")
                records.append({"partition": partition, "seed": seed, "condition": condition,
                                "encoder_hash": record["encoder_hash"], "encoder_checkpoint_sha256": sha256(folder / "encoder.pt"),
                                "steps": record["optimizer_steps"], "ema_updates": int(state["model"]["ema_updates"]), "status": "PASS"})
    metrics = pd.read_csv(root / "reports/m5/downstream_metrics.csv")
    expected = len(protocol["partitions"]) * len(protocol["seeds"]) * len(protocol["conditions"]) * 4
    if len(metrics) != expected or metrics.duplicated(["partition", "seed", "condition", "task", "label_fraction"]).any():
        raise ValueError("Incomplete/duplicate downstream matrix")
    probes = pd.read_csv(root / "reports/m5/leaf_structure_probe.csv")
    if len(probes) != len(protocol["partitions"]) * len(protocol["seeds"]) * len(protocol["conditions"]):
        raise ValueError("Incomplete leaf structure probes")
    external = json.loads((root / "reports/m5/external_representation_diagnostics.json").read_text())
    if external["protocol_sha256"] != lock["protocol_sha256"] or len(external["records"]) != len(protocol["seeds"]) * len(protocol["conditions"]):
        raise ValueError("Incomplete external read-only diagnostic matrix")
    test_candidates = [path for path in (root / "outputs/m5").rglob("*") if "test" in path.name.lower() and path.is_file()]
    if test_candidates:
        raise ValueError("Unexpected M5 test prediction artifact")
    lock_path = root / "reports/m5/protocol_lock.json"
    formal_initial_paths = [root / "outputs/m5/checkpoints" / record["partition"] /
                            f"{record['condition']}_seed{record['seed']}" / "diagnostics_step0.json" for record in records]
    first_formal_ns = min(path.stat().st_mtime_ns for path in formal_initial_paths)
    if lock_path.stat().st_mtime_ns >= first_formal_ns:
        raise ValueError("Protocol lock does not precede formal diagnostics")
    write_json(root / "reports/m5/preregistration_chronology.json", {
        "status": "PASS", "protocol_lock_mtime_ns": lock_path.stat().st_mtime_ns,
        "earliest_formal_initial_diagnostic_mtime_ns": first_formal_ns,
        "protocol_lock_sha256": sha256(lock_path), "formal_runs": len(records),
        "scope": "Local immutable hash lock before formal training; filesystem chronology, not public study registration or trusted external timestamp"})
    assignments = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]
    official = assignments["official"]
    split_records = []
    for partition in protocol["partitions"]:
        assignment = assignments[partition]
        selected = {plant for plant, role in assignment.items() if role == "train"}
        excluded = {plant for plant, role in assignment.items() if role in ("val", "test")}
        if selected & excluded or len(selected) != 24:
            raise ValueError("Invalid within-partition train exclusion")
        split_records.append({"partition": partition, "train_plants": sorted(selected),
                              "within_partition_validation_test_in_ssl": 0,
                              "original_official_roles_in_local_train": {
                                  role: sum(official[plant] == role for plant in selected)
                                  for role in ("train", "val", "test")}})
    write_json(root / "reports/m5/split_provenance.json", {
        "status": "PASS_WITH_FOLD_ROLE_ROTATION_DISCLOSURE", "records": split_records,
        "scope": "Original M4 partitions retained. Official primary encoder uses only original official Train. Every fold excludes its own Validation/Test; fold-local Train rotates official roles.",
        "global_official_holdout_across_all_fold_models": False,
        "untouched_confirmatory_test": False,
        "warning": "Fold evidence is not global official-holdout evidence. No new globally held-out test is claimed."})
    for path in (root / "outputs/m5/downstream").glob("*/training.json"):
        record = json.loads(path.read_text())
        expected_plants = lock["label_budgets"][record["partition"]][str(record["label_fraction"])]
        if record["labelled_plants"] != expected_plants or record["test_used"]:
            raise ValueError("Downstream labels/split provenance mismatch")
        state = torch.load(path.parent / "last.pt", map_location="cpu", weights_only=False)
        if not all(field in state for field in ("model", "optimizer", "rng", "loader_rng", "step", "epoch")):
            raise ValueError("Incomplete downstream training state")
    write_json(root / "reports/m5/verification.json", {"status": "PASS", "ssl_checkpoints": len(records),
               "downstream_runs": len(metrics), "probe_runs": len(probes), "test_prediction_files": 0,
               "external_diagnostic_encoders": len(external["records"]),
               "protocol_sha256": lock["protocol_sha256"], "records": records})
    print(f"M5 VERIFY PASS: {len(records)}SSL, {len(metrics)}downstream, {len(probes)}probes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    arguments = parser.parse_args()
    verify(arguments.root)
