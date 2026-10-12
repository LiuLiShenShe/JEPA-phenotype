#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from m4.common import write_json
from m5.common import load_protocol, lock_protocol
from m5.downstream import run_downstream, verify_restore
from m5.evaluation import summarize
from m5.pretrain import formal_pretrain
from m5.probe import run_probe


def run(root, partitions, device):
    root = Path(root)
    protocol, _ = load_protocol(root)
    for partition in partitions:
        for seed in protocol["seeds"]:
            for condition in ("S500", "P3", "P4"):
                if condition not in protocol["conditions"]:
                    continue
                formal_pretrain(root, partition, seed, condition, device)
                for fraction in (.5, .25, 1.):
                    run_downstream(root, partition, seed, condition, fraction, "count", device)
                run_downstream(root, partition, seed, condition, 1., "segmentation", device)
            for condition in protocol["conditions"]:
                run_probe(root, partition, seed, condition, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--stage", choices=("lock", "run", "evaluate", "restore"), required=True)
    parser.add_argument("--partitions", default="official,fold0,fold1,fold2,fold3")
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    root, device = Path(arguments.root), torch.device(arguments.device)
    if arguments.stage == "lock":
        lock_protocol(root)
        print("M5 FORMAL PROTOCOL LOCKED", flush=True)
    elif arguments.stage == "run":
        run(root, arguments.partitions.split(","), device)
    elif arguments.stage == "evaluate":
        print(summarize(root), flush=True)
    else:
        records = [verify_restore(root, path.parent, device) for path in sorted((root / "outputs/m5/downstream").glob("*/best.pt"))]
        write_json(root / "reports/m5/checkpoint_restore.json", {"status": "PASS", "checkpoints": len(records), "records": records})
