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

from m4.common import sha256, write_json
from m5.common import load_protocol
from m5.diagnostics import feature_statistics, similarities
from m5.downstream import source_encoder


@torch.no_grad()
def diagnose(root, device):
    root = Path(root)
    protocol, lock = load_protocol(root)
    manifest = pd.read_csv(root / "reports/m5/external_data_manifest.csv")
    clouds = []
    for row in manifest.itertuples(index=False):
        path = root / "outputs/m5/data/external_xyz" / row.cache_file
        if sha256(path) != row.cache_sha256:
            raise ValueError("External diagnostic cache changed")
        with np.load(path, allow_pickle=False) as archive:
            clouds.append({field: torch.from_numpy(archive[field].copy())
                           for field in ("points", "center_indices", "neighbors")})
    groups = [("all", "all", np.arange(len(manifest)))]
    groups += [(source, "all", group.index.to_numpy()) for source, group in manifest.groupby("source")]
    groups += [(source, species, group.index.to_numpy()) for (source, species), group in manifest.groupby(["source", "species"])]
    output = root / "reports/m5/external_representation_diagnostics.json"
    diagnostic = {"protocol_sha256": lock["protocol_sha256"], "scope": "external Train geometry only, post-training read-only diagnostic",
                  "independent_plant_total": "unknown; scan/fragment counts are not verified independent plants",
                  "selection_use": "none; no training, loss, weight, checkpoint or success-gate changes", "records": {}}
    if output.exists():
        diagnostic = json.loads(output.read_text())
        if diagnostic["protocol_sha256"] != lock["protocol_sha256"]:
            raise ValueError("External diagnostic lock changed")
    for seed in protocol["seeds"]:
        for condition in protocol["conditions"]:
            key = f"official_{condition}_seed{seed}"
            if key in diagnostic["records"]:
                continue
            encoder, reference = source_encoder(root, protocol, "official", seed, condition)
            encoder.to(device).eval()
            pooled_parts, token_parts = [], []
            for start in range(0, len(clouds), 64):
                batch = {field: torch.stack([cloud[field] for cloud in clouds[start:start + 64]]).to(device)
                         for field in ("points", "center_indices", "neighbors")}
                pooled, tokens, _ = encoder(**batch, return_tokens=True)
                pooled_parts.append(pooled.cpu())
                token_parts.append(tokens.cpu())
            pooled, tokens = torch.cat(pooled_parts), torch.cat(token_parts)
            records = []
            generator = np.random.default_rng(20261012)
            for source, species, indices in groups:
                first, second = np.triu_indices(len(indices), 1)
                chosen = generator.choice(len(first), min(8192, len(first)), replace=False)
                selected_pooled, selected_tokens = pooled[indices], tokens[indices]
                records.append({"source": source, "species": species, "objects": len(indices),
                                "pooled": feature_statistics(selected_pooled), "token": feature_statistics(selected_tokens),
                                "distinct_scan_or_fragment_similarity": similarities(selected_pooled, first[chosen], second[chosen]),
                                "similarity_identity_caveat": "distinct objects, not necessarily distinct independent plants"})
            diagnostic["records"][key] = {"condition": condition, "seed": seed, "partition": "official",
                                          "encoder_reference": reference, "groups": records}
            write_json(output, diagnostic)
            print(f"EXTERNAL_DIAGNOSTIC {condition} seed={seed} PASS", flush=True)
    rows = []
    for record in diagnostic["records"].values():
        for group in record["groups"]:
            rows.append({"condition": record["condition"], "seed": record["seed"], "source": group["source"],
                         "species": group["species"], "objects": group["objects"],
                         **{f"{kind}_{field}": group[kind][field] for kind in ("pooled", "token")
                            for field in ("effective_rank", "std_mean", "variance_mean", "largest_eigenvalue_fraction")},
                         **{f"distinct_objects_cosine_{mode}": group["distinct_scan_or_fragment_similarity"][mode]["mean"]
                            for mode in ("uncentred", "centred")}})
    pd.DataFrame(rows).to_csv(root / "reports/m5/external_representation_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    diagnose(arguments.root, torch.device(arguments.device))
