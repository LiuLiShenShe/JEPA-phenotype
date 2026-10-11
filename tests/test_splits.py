from __future__ import annotations

import csv
import json

from create_splits import create_splits


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_plant_level_split_is_disjoint_and_reproducible(workflow):
    payload = json.loads(workflow["split_path"].read_text(encoding="utf-8"))
    train_plants = set(payload["train_plants"])
    val_plants = set(payload["val_plants"])
    test_plants = set(payload["test_plants"])
    assert train_plants.isdisjoint(val_plants)
    assert train_plants.isdisjoint(test_plants)
    assert val_plants.isdisjoint(test_plants)
    assert train_plants | val_plants | test_plants == set(payload["assignments"])
    assert {plant.split("/")[0] for plant in val_plants} == {"G1", "G2", "G3"}
    assert {plant.split("/")[0] for plant in test_plants} == {"G1", "G2", "G3"}

    rows = read_csv(workflow["metadata_dir"] / "frames.csv")
    rerun_path = workflow["root"] / "splits" / "rerun.json"
    rerun = create_splits(
        workflow["metadata_dir"] / "frames.csv",
        rerun_path,
        workflow["root"] / "reports" / "m0" / "rerun_statistics.csv",
        workflow["root"] / "reports" / "m0" / "rerun_distribution.png",
        seed=42,
    )
    assert rerun["assignments"] == payload["assignments"]
    assert all(rerun["assignments"][row["plant_uid"]] in {"train", "val", "test"} for row in rows)
