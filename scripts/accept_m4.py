#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m4.common import load_protocol, sha256, write_json


def run(root):
    root = Path(root)
    protocol, lock = load_protocol(root)
    report = root / "reports/m4"
    coverage = json.loads((report / "coverage.json").read_text())
    protection = json.loads((report / "final_protection.json").read_text())
    restored = json.loads((report / "checkpoint_prediction_restore.json").read_text())
    fairness = json.loads((report / "fairness.json").read_text())
    decision = json.loads((report / "decision.json").read_text())
    assert coverage["p0_p2_matrix_complete"] and coverage["count_completed"] == 180
    assert coverage["segmentation_completed"] == 75 and coverage["ssl_completed"] == 15
    assert restored["status"] == "PASS" and restored["checkpoints"] == 255
    assert fairness["status"] == "PASS" and fairness["paired_runs"] == 120
    assert protection["status"] == "PASS" and not protection["differences"]
    assert decision["temporal_jepa_started"] is False and protocol["stop_before_temporal_jepa"]
    tests = []
    for name in ("tests_m4.xml", "tests_all_original_environment.xml"):
        suites = ElementTree.parse(report / name).getroot()
        failures = sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites)
        assert failures == 0
        tests.append({"file": name, "tests": sum(int(suite.attrib.get("tests", 0)) for suite in suites), "failures": failures})
    visuals = json.loads((report / "figure_visual_qa.json").read_text())
    assert len(visuals["images"]) == 8 and all(row["status"] == "API_RECOGNIZED" for row in visuals["images"])
    required = ["M4_REPORT.md", "pretrained_comparison.csv", "label_efficiency.csv", "segmentation_metrics.csv",
                "representation_diagnostics.json", "representation_full_train_diagnostics.json", "paired_comparisons.csv",
                "failure_cases.csv", "stage_metrics.csv", "cross_fold_validation.csv", "condition_availability.csv"]
    manifest = {name: {"sha256": sha256(report / name), "bytes": (report / name).stat().st_size} for name in required}
    for path in sorted((report / "figures").glob("*")):
        if path.is_file():
            manifest[str(path.relative_to(report))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    source_paths = [root / "M4_README.md", root / "requirements-m4.txt", root / "requirements-m4-lock.txt"]
    source_paths += list((root / "src/m4").glob("*.py")) + list((root / "src/models").glob("point_jepa*.py"))
    source_paths += [root / "src/datasets/sugar4d_ssl_dataset.py"] + list((root / "scripts").glob("*m4*.py"))
    source_paths += list((root / "tests").glob("test_m4*.py"))
    source_manifest = {str(path.relative_to(root)): sha256(path) for path in source_paths}
    write_json(report / "artifact_manifest.json", {"protocol_sha256": lock["protocol_sha256"], "artifacts": manifest, "source_sha256": source_manifest,
                                                  "checkpoint_full_reload": restored["checkpoints"], "source_snapshot_note": "final code after isolated compatibility fixes"})
    write_json(report / "acceptance.json", {"m4a": "PASS", "feasible_p0_p2_experiments": "PASS", "overall_engineering": "PARTIAL",
                                            "blocker": "P1 official ShapeNet checkpoint unavailable; external transfer/weight coverage not measured",
                                            "science_decision": decision["decision"], "ssl_runs": 15, "count_runs": 180, "segmentation_runs": 75,
                                            "paired_fairness_checks": 120, "checkpoint_restores": 255, "tests": tests,
                                            "old_artifacts_unchanged": True, "protected_files": protection["protected_files"],
                                            "test_inference": False, "temporal_jepa": False, "source_data_and_figures": True,
                                            "image_content": "图片内容由 Agnes API 识别"})
    print("M4-A PASS | all feasible P0/P2 experiments PASS | overall PARTIAL (P1 unavailable) | " + decision["decision"])


if __name__ == "__main__":
    run(ROOT)
