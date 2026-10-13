from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m6a.common import digest, write_json
from m6a.experiment import metadata
from m6b.common import check_lock, verify
from m6b.evaluation import prediction_path, run_dir


def main():
    lock = check_lock(ROOT)
    protection = verify(ROOT)
    records = metadata(ROOT)
    checks = []
    report = ROOT / "reports/m6_b"
    for method in ["S1", "S2", "S3"]:
        for seed in ([17] if method == "S3" else [17, 29, 43]):
            directory = run_dir(ROOT, method, seed)
            training = json.loads((directory / "training_complete.json").read_text())
            assert training["epochs"] == 60 and training["optimizer_steps"] == 5760
            assert training["frame_presentations"] == 23040 and training["protocol_lock_sha256"] == lock
            assert digest(directory / "best.pth") == training["best_sha256"]
            manifest = json.loads((directory / "training_manifest.json").read_text())
            for source, expected in manifest["code_sha256"].items():
                assert digest(ROOT / source) == expected, source
            steps = pd.read_csv(directory / "batches.csv")
            assert len(steps) == 5760 and steps.optimizer_step.nunique() == 5760
            assert np.isfinite(steps[["loss", "semantic", "discriminative", "similarity"]].to_numpy()).all()
            post = json.loads((directory / "postprocessing_selection.json").read_text())
            assert post["selection_split"] == "val" and post["no_test_selection"]
            for split in (["val", "test"] if seed == 17 else ["val"]):
                inference = json.loads((directory / f"inference_{split}.json").read_text())
                assert inference["frames"] == 192 and inference["plants"] == 12
                assert inference["full_resolution"] and inference["checkpoint_sha256"] == training["best_sha256"]
                for record in records.query("split == @split").itertuples(index=False):
                    with np.load(prediction_path(ROOT, method, seed, record.plant_uid, record.DAS)) as predicted:
                        assert len(predicted["labels"]) == record.point_count
                        assert len(predicted["semantic"]) == record.point_count
                        assert len(predicted["sample_indices"]) == 4096
                        assert np.all(predicted["labels"] >= 0)
                if seed == 17:
                    tracked = pd.read_csv(directory / f"tracking_plants_{split}.csv")
                    assert len(tracked) == 12 and tracked.plant_uid.nunique() == 12
                    assert np.isfinite(tracked[["idf1", "new_event_f1", "continuity"]].to_numpy()).all()
            checks.append({"method": method, "seed": seed, "epochs": 60, "steps": 5760, "valid": True})
    decision = json.loads((report / "decision.json").read_text())
    assert decision["recommendation"] in {"INSTANCE-SOLVED-FOR-TRACKING", "INSTANCE-METHOD-NEEDED", "DATA-LIMITED"}
    assert decision["exactly_one"] and not decision["measurement_system_certified"]
    assert json.loads((report / "organ3d_feasibility.json").read_text())["status"] == "BLOCKED"
    errors = pd.read_csv(report / "segmentation_leaf_errors.csv")
    for (method, seed, split), group in errors.groupby(["method", "seed", "split"]):
        expected = int(pd.read_csv(report / "data_frames.csv").query("split == @split").leaf_instances.sum())
        assert len(group) == expected, (method, seed, split, len(group), expected)
    events = pd.read_csv(report / "first_observation_events.csv")
    tracking = pd.read_csv(report / "tracking_per_plant.csv")
    for (pipeline, split), group in events.groupby(["pipeline", "split"]):
        expected = int(tracking.query("pipeline == @pipeline and split == @split").new_gt_events.sum())
        assert len(group) == expected, (pipeline, split, len(group), expected)
        assert group.gt_local_id_evaluation_only.ge(1).all()
        assert not group.duplicated(["plant_uid", "gt_id_evaluation_only"]).any()
    result = {"protocol_lock_sha256": lock, "checks": checks, "all_required_supervised_runs_actual": True,
              "all_full_resolution_predictions_present": True, "all_legal_leaf_error_rows_present": True,
              "all_legal_first_observation_events_present": True,
              "historical_protection": protection["unchanged"], "HEAD_unchanged": protection["git_head_unchanged"],
              "S4": "BLOCKED without replacement", "recommendation": decision["recommendation"], "passed": True}
    write_json(report / "final_verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
