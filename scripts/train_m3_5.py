#!/usr/bin/env python
"""Run locked lightweight M3.5 baselines; never infer on test during fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m3.common import file_hash
from m35.engine import current_simple, current_train, event_simple, organ_simple, tabular_arrays, train_tabular


def run(args):
    root = Path(args.root).resolve()
    protocol = json.loads((root / "configs/m3_5/protocol.json").read_text())
    if file_hash(root / "configs/m3_5/protocol.json") != json.loads((root / "reports/m3_5/protocol_lock.json").read_text())["sha256"]:
        raise ValueError("Protocol lock mismatch")
    partitions = json.loads((root / "reports/m3/plant_splits.json").read_text())["assignments"]
    feasibility = json.loads((root / "reports/m3_5/feasibility.json").read_text())
    seeds = protocol["seeds"] if args.seeds is None else [int(value) for value in args.seeds.split(",")]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    for partition in args.partitions.split(","):
        assignment = partitions[partition]
        fits = current_simple(root, partition, assignment, protocol)
        if args.stage in ("current", "all"):
            current_train(root, partition, assignment, protocol, device, seeds)
        if args.stage in ("future", "all"):
            if not (root / "reports/m3_5/current_state_diagnostics.csv").exists():
                raise ValueError("Complete current-state diagnostics before future prediction")
            gates = feasibility["gates"][partition]
            data = tabular_arrays(root, "organ", assignment)
            allowed = [int(horizon) for horizon, tasks in gates.items() if tasks["morphology"]["feasible"]]
            if allowed:
                if allowed != protocol["horizons"]:
                    raise ValueError("Partial organ horizons require an explicit gated task implementation")
                organ_simple(root, partition, data)
                train_tabular(root, partition, data, "organ", ["C2-MLP", "C3-GRU", "C4-Repeated-GRU"], protocol, device, seeds)
            for task in ("geometry_absolute", "geometry_residual"):
                data = tabular_arrays(root, task, assignment, fits)
                names = [f"C5-{history}-{task.rsplit('_', 1)[-1].title()}" for history in ("Current", "History")]
                train_tabular(root, partition, data, task, names, protocol, device, seeds)
            if all(tasks["event"]["feasible"] for tasks in gates.values()):
                data = tabular_arrays(root, "event", assignment)
                event_simple(root, partition, data, protocol)
                train_tabular(root, partition, data, "event", ["E2-MLP", "E3-GRU", "E4-Repeated-GRU"], protocol, device, seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--stage", choices=("current", "future", "all"), default="current")
    parser.add_argument("--partitions", default="official,fold0,fold1,fold2,fold3")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
