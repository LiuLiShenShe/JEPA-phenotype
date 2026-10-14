import json

import numpy as np
import pandas as pd
import torch

from birth_detection.common import initialize, protocol, write_json
from organ_tracking.engine import load_matcher


def predict_test_pairs(root, device="cuda:1"):
    settings = protocol(root)
    if not (root / "reports/m7/system_selection.json").exists():
        raise ValueError("Validation system selection must precede Test")
    with np.load(root / "outputs/m7/pair_dataset.npz") as data:
        raw_features = data["features"]
        previous_indices, current_indices = data["previous_indices"], data["current_indices"]
    with np.load(root / "outputs/m7/organ_dataset.npz") as data:
        points = data["points"]
    for method in ["C1", "C2"]:
        for seed in settings["seeds"]:
            initialize(seed)
            model, mean, scale = load_matcher(root, method, seed)
            model.to(device)
            path = root / f"outputs/m7/matching/{method}/seed_{seed}/pair_probabilities.csv"
            rows = pd.read_csv(path)
            indices = np.flatnonzero((rows.split == "test").to_numpy())
            features = np.clip((raw_features - mean) / scale, -10, 10).astype(np.float32)
            probabilities = []
            with torch.inference_mode():
                for start in range(0, len(indices), 512):
                    batch = indices[start:start + 512]
                    previous = torch.from_numpy(points[previous_indices[batch]]).to(device) if method == "C2" else None
                    current = torch.from_numpy(points[current_indices[batch]]).to(device) if method == "C2" else None
                    logits, unused_distance = model(torch.from_numpy(features[batch]).to(device), previous, current)
                    probabilities.extend(torch.sigmoid(logits).cpu().numpy())
            rows.loc[indices, "probability"] = probabilities
            rows.to_csv(path, index=False)
            print("TEST PAIR INFER", method, seed, len(indices), flush=True)
