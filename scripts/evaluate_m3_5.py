#!/usr/bin/env python
"""M3.5 validation diagnostics, selection locking and exploratory final test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m3.common import file_hash, seed_everything, write_json
from m35.analysis import current_diagnostics, final_analysis, lock_selection, replay_test, training_audit
from m35.data import verify_protection


def run(args):
    root = Path(args.root).resolve()
    protocol = json.loads((root / "configs/m3_5/protocol.json").read_text())
    seed_everything(protocol["seeds"][0])
    if args.stage == "current":
        summary, predictions = current_diagnostics(root, protocol)
        print(summary.loc[summary.partition.eq("official") & summary.split.eq("val")].to_string(index=False))
    elif args.stage == "select":
        audit = training_audit(root, protocol)
        lock = lock_selection(root, protocol)
        write_json(root / "reports/m3_5/training_complete.json", audit)
        print(json.dumps(lock["research_decision_before_test"], indent=2))
    else:
        lock = json.loads((root / "reports/m3_5/selection_lock.json").read_text())
        if lock["protocol_sha256"] != file_hash(root / "configs/m3_5/protocol.json"):
            raise ValueError("Protocol changed after selection")
        if lock["analysis_source_sha256"] != file_hash(root / "src/m35/analysis.py"):
            raise ValueError("Analysis rules changed after validation selection")
        if lock["training_entrypoint_sha256"] != file_hash(root / "scripts/train_m3_5.py"):
            raise ValueError("Training entrypoint changed after selection")
        for path, digest in lock["artifact_fingerprints"].items():
            if file_hash(root / path) != digest:
                raise ValueError("Model changed after selection")
        test = replay_test(root, protocol, torch.device(args.device), lock)
        predictions = final_analysis(root, protocol, lock, test)
        write_json(root / "reports/m3_5/final_protection.json", verify_protection(root))
        from m35.figures import render_figures
        from m35.report import write_report
        render_figures(root, predictions, protocol)
        write_report(root, protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--stage", choices=("current", "select", "final"), default="current")
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
