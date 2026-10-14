from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def protocol(root):
    lock = json.loads((root / "reports/m7/protocol_lock.json").read_text())
    for name, expected in lock["sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen M7 input changed: " + name)
    return json.loads((root / "configs/m7/protocol.json").read_text())


def initialize(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def window_indices(index, length=3):
    return [max(0, index - offset) for offset in reversed(range(length))]


def strict_matches(gt, masks):
    from scipy.optimize import linear_sum_assignment

    ids = np.unique(gt[gt > 0])
    scores = np.zeros((len(ids), len(masks)))
    sizes = {int(label): int(np.sum(gt == label)) for label in ids}
    for column, indices in enumerate(masks):
        values, counts = np.unique(gt[indices], return_counts=True)
        overlaps = dict(zip(values, counts))
        for row, label in enumerate(ids):
            intersection = overlaps.get(label, 0)
            scores[row, column] = intersection / max(sizes[int(label)] + len(indices) - intersection, 1)
    pairs = {}
    if scores.size:
        rows, columns = linear_sum_assignment(-scores)
        pairs = {int(ids[row]): (int(column), float(scores[row, column]))
                 for row, column in zip(rows, columns) if scores[row, column] > .5}
    return pairs, scores, ids


def paired_interval(values, iterations=10000, seed=20261013):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "lower": None, "upper": None, "plants": 0}
    generator = np.random.default_rng(seed)
    means = values[generator.integers(len(values), size=(iterations, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "lower": float(np.quantile(means, .025)),
            "upper": float(np.quantile(means, .975)), "plants": len(values)}
