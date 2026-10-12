from __future__ import annotations

import json
from pathlib import Path

from m4.common import load_protocol as m4_protocol, sha256, write_json


def load_protocol(root):
    root = Path(root)
    m4_protocol(root)
    path = root / "configs/m5/protocol.json"
    lock = json.loads((root / "reports/m5/protocol_lock.json").read_text())
    if sha256(path) != lock["protocol_sha256"]:
        raise ValueError("Locked M5 protocol changed")
    for name, digest in lock["inputs"].items():
        if sha256(root / name) != digest:
            raise ValueError(f"Locked M5 input changed: {name}")
    return json.loads(path.read_text()), lock


def lock_protocol(root):
    root = Path(root)
    path = root / "configs/m5/protocol.json"
    lock_path = root / "reports/m5/protocol_lock.json"
    if lock_path.exists():
        return load_protocol(root)
    if any(folder.name.startswith(("S500", "P3", "P4")) for folder in (root / "outputs/m5/checkpoints").glob("*/*")):
        raise ValueError("Cannot preregister after formal SSL training")
    protocol = json.loads(path.read_text())
    _, old_lock = m4_protocol(root)
    required = ["configs/m5/budget_diagnostic.json", "reports/m5/external_data_manifest.csv",
                "reports/m5/data_preparation.json", "reports/m5/history_audit.json",
                "reports/m3/plant_splits.json", "data/processed/sugar4d/plant_metadata.csv",
                "outputs/m3_5/data/geometry.csv", "src/m5/pretrain.py", "src/m5/diagnostics.py",
                "src/m5/downstream.py", "src/m5/evaluation.py", "src/m5/probe.py",
                "src/m5/common.py", "scripts/run_m5.py", "scripts/prepare_m5_data.py",
                "src/models/point_jepa_encoder.py", "src/models/point_jepa_predictor.py",
                "src/models/point_jepa_downstream.py", "src/datasets/sugar4d_ssl_dataset.py",
                "src/datasets/plant_ssl_dataset.py", "reports/m5/BUDGET_SELECTION_NOTE.md",
                "reports/m5/source_integrity.json", "reports/m5/DATA_SOURCE_AUDIT.md",
                "outputs/m5/checkpoints/official/M5A_seed17/diagnostics_step0.json",
                "outputs/m5/checkpoints/official/M5A_seed17/diagnostics_step500.json"]
    write_json(lock_path, {"date": "2026-10-12", "protocol_sha256": sha256(path),
                          "inputs": {name: sha256(root / name) for name in required},
                          "label_budgets": old_lock["label_budgets"], "test_opened": False,
                          "hypothesis_status": "new exploratory hypothesis informed by historical M4; not untouched confirmation",
                          "budget_selection": protocol["budget_selection"], "external_manifest_locked": True})
    return load_protocol(root)
