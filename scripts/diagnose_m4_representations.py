#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from datasets.sugar4d_ssl_dataset import Sugar4DSSLDataset, partition_rows
from m4.common import load_protocol, representation_diagnostics, state_hash, write_json
from m4.downstream import source_encoder


@torch.no_grad()
def diagnose(root, device):
    protocol, _ = load_protocol(root)
    summary = {}
    for partition in protocol["partitions"]:
        rows = partition_rows(root, partition, "train")
        dataset = Sugar4DSSLDataset(rows, root / "outputs/m4/data/xyz", training=False)
        loader = DataLoader(dataset, batch_size=protocol["ssl"]["batch_size"], shuffle=False)
        for seed in protocol["seeds"]:
            entry = {}
            for condition in ("P0", "P2"):
                encoder, _ = source_encoder(root, protocol, partition, seed, condition)
                encoder.to(device).eval()
                features, tokens = [], []
                for batch in loader:
                    pooled, patches, _ = encoder(batch["points"].to(device), batch["center_indices"].to(device),
                                                 batch["neighbors"].to(device), return_tokens=True)
                    features.append(pooled.cpu())
                    tokens.append(patches.cpu())
                record = representation_diagnostics(torch.cat(tokens), torch.cat(features))
                record.update({"diagnostic_split": "train", "independent_plants": rows.plant_uid.nunique(),
                               "all_training_scans": len(rows), "encoder_hash": state_hash(encoder.state_dict()),
                               "condition": condition, "selection_use": "none; post-training diagnostics only"})
                entry[condition] = record
                del encoder
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            summary[f"{partition}_seed{seed}"] = entry
            print(f"DIAGNOSTIC {partition} seed={seed} P0_rank={entry['P0']['effective_rank']:.2f} P2_rank={entry['P2']['effective_rank']:.2f}", flush=True)
    write_json(root / "reports/m4/representation_full_train_diagnostics.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    diagnose(Path(arguments.root), torch.device(arguments.device))
