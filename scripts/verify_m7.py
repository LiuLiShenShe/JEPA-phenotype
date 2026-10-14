import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from birth_detection.common import digest, protocol, write_json

if __name__ == "__main__":
    settings = protocol(ROOT)
    historical = json.loads((ROOT / "reports/m7/historical_snapshot.json").read_text())
    changes = []
    for name, expected in historical["files"].items():
        path = ROOT / name
        if not path.is_file():
            changes.append(name)
            continue
        info = path.stat()
        if info.st_size != expected["bytes"] or ("mtime_ns" in expected and info.st_mtime_ns != expected["mtime_ns"]) or ("sha256" in expected and digest(path) != expected["sha256"]):
            changes.append(name)
    commit_unchanged = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == historical["commit"]
    protection = {"historical_files": len(historical["files"]), "changes": changes, "unchanged": not changes,
                  "git_HEAD_unchanged": commit_unchanged}
    write_json(ROOT / "reports/m7/protection_verification.json", protection)
    frames = pd.read_csv(ROOT / "reports/m7/frame_audit.csv")
    events = pd.read_csv(ROOT / "reports/m7/organ_birth_events.csv")
    candidates = pd.read_csv(ROOT / "outputs/m7/candidates.csv")
    training = []
    for path in (ROOT / "outputs/m7").glob("**/completed.json"):
        if "invalid_source_leak_runs" in path.parts:
            continue
        record = json.loads(path.read_text())
        for name, expected in record.get("code_sha256", {}).items():
            assert digest(ROOT / name) == expected
        if "epochs" in record:
            logs = pd.read_csv(path.parent / "epochs.csv")
            assert record["epochs"] == 40 and len(logs) == 40 and logs.epoch.tolist() == list(range(1, 41))
            assert np.isfinite(logs.train_loss).all()
        checkpoint = path.parent / ("model.pth" if (path.parent / "model.pth").exists() else "model.joblib")
        assert digest(checkpoint) == record["checkpoint_sha256"]
        training.append({"path": str(path.relative_to(ROOT)), **record})
    assert len(training) == 21
    for folder in ["birth", "matching", "forecast"]:
        probabilities = list((ROOT / "outputs/m7" / folder).glob("**/*probabilities.csv"))
        assert len(probabilities) == {"birth": 12, "matching": 6, "forecast": 3}[folder]
        for path in probabilities:
            table = pd.read_csv(path)
            assert np.isfinite(table.probability).all() and table.probability.between(0, 1).all()
            assert set(table.split) == {"train", "val", "test"}
            if folder == "birth":
                assert len(table) == len(candidates)
    with np.load(ROOT / "outputs/m7/birth_dataset.npz") as data:
        assert data["single_features"].shape[1] == 54
        expected = np.concatenate([data["temporal_features"][:, :25], data["temporal_features"][:, 27:29],
                                   data["temporal_features"][:, 79:104], data["temporal_features"][:, 108:110]], axis=1)
        assert np.allclose(data["single_features"], expected)
        assert data["timing"].dtype == np.float32
    assert frames.shape[0] == 768 and frames.groupby("plant_id").size().eq(16).all()
    assert frames.groupby("plant_id").split.nunique().eq(1).all()
    assert frames.groupby("split").plant_id.nunique().to_dict() == {"test": 12, "train": 24, "val": 12}
    assert not events.duplicated(["plant_id", "leaf_instance_id"]).any()
    assert events.query("left_censored").previous_observation_available.eq(False).all()
    assert events.query("segmentable_right_censored").birth_time_segmentable.isna().all()
    assert events.query("trackable_right_censored").birth_time_trackable.isna().all()
    prediction_files = list((ROOT / "outputs/m7/tracking").glob("*/seed_*/*/predictions/*.npz"))
    assert len(prediction_files) == 4 * 3 * 384
    for path in prediction_files:
        with np.load(path) as prediction:
            plant, day = path.stem.rsplit("_", 1)
            das = int(day)
            size = int(frames.query("plant_id == @plant and DAS == @das").iloc[0].points)
            assert len(prediction["labels"]) == size
            assert len(np.unique(prediction["track_ids"])) == len(prediction["track_ids"])
    selection = json.loads((ROOT / "reports/m7/system_selection.json").read_text())
    gate = json.loads((ROOT / "reports/m7/test_inference_gate.json").read_text())
    assert selection["selection_split"] == "val" and selection["no_test_selection"]
    assert gate["validation_system_selection_sha256"] == digest(ROOT / "reports/m7/system_selection.json")
    decision = json.loads((ROOT / "reports/m7/decision.json").read_text())
    assert decision["outcome"] in {"BIRTH-SUCCESS", "TRACKING-SUCCESS", "DATA-LIMITED"}
    assert decision["next_stage"] in {"BIRTH-MODELING", "ORGAN-TRACKING", "DYNAMIC-PHENOTYPING", "STOP"}
    decision_lock = json.loads((ROOT / "reports/m7/validation_decision_lock.json").read_text())
    assert decision_lock["validation_only"]
    assert decision_lock["system_selection_sha256"] == digest(ROOT / "reports/m7/system_selection.json")
    assert decision_lock["tracking_val_sha256"] == digest(ROOT / "reports/m7/tracking_metrics_val.csv")
    for name in ["outcome", "next_stage", "chosen", "birth_gates_pass", "tracking_gates_pass"]:
        assert decision[name] == decision_lock["decision"][name]
    for name in ["validation_mean_birth_F1", "validation_mean_IDF1"]:
        assert np.isclose(decision[name], decision_lock["decision"][name], atol=1e-12, rtol=0)
    equivalence = json.loads((ROOT / "reports/m7/evaluation_cache_equivalence.json").read_text())
    assert all(row["equal_direct_and_cached"] for row in equivalence["records"])
    assert protection["unchanged"] and commit_unchanged
    write_json(ROOT / "reports/m7/final_verification.json", {"passed": True, "real_training_runs": len(training),
               "training": training, "full_resolution_prediction_frames": len(prediction_files), "frames": len(frames),
               "events": len(events), "candidates": len(candidates), "historical_unchanged": True,
               "Test_after_validation_selection": True, "recommendation": decision["outcome"], "next_stage": decision["next_stage"]})
    paths = []
    for folder in ["configs/m7", "src/birth_detection", "src/organ_tracking", "src/temporal_encoder", "reports/m7", "outputs/m7"]:
        paths.extend(path for path in (ROOT / folder).rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.name != "delivery_manifest.json")
    paths.extend(ROOT / name for name in ["scripts/build_birth_dataset.py", "scripts/train_birth_detector.py", "scripts/evaluate_birth.py",
                 "scripts/train_m7_matrix.sh", "scripts/report_m7.py", "scripts/analyze_m7.py", "scripts/plot_m7.py", "scripts/verify_m7.py", "scripts/audit_m7_sources.py", "tests/test_m7.py"])
    write_json(ROOT / "reports/m7/delivery_manifest.json", {"files": [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)} for path in sorted(set(paths))]})
    print("M7 VERIFIED", len(training), len(prediction_files), decision["outcome"], decision["next_stage"])
