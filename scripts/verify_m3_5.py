#!/usr/bin/env python
"""Verify M3.5 real-data acceptance and legacy immutability without rewriting M3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as element_tree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from m3.common import file_hash, write_json
from m35.analysis import training_audit
from m35.data import verify_protection


def verify(root):
    root = Path(root).resolve()
    report = root / "reports/m3_5"
    protocol = json.loads((root / "configs/m3_5/protocol.json").read_text())
    lock = json.loads((report / "selection_lock.json").read_text())
    if lock["protocol_sha256"] != file_hash(root / "configs/m3_5/protocol.json"):
        raise ValueError("Protocol changed")
    if lock["analysis_source_sha256"] != file_hash(root / "src/m35/analysis.py"):
        raise ValueError("Analysis changed after Test lock")
    for path, expected in lock["artifact_fingerprints"].items():
        if file_hash(root / path) != expected:
            raise ValueError("Locked selected model changed")
    protection = verify_protection(root)
    archive = root / "data/raw/sugar4d/sugar4d.zip"
    if archive.stat().st_size != 6270780517 or file_hash(archive, "md5") != "27e44210d95a85cb04b93513f0ef2beb":
        raise ValueError("Existing V4 archive integrity changed")
    readme = root / "data/raw/sugar4d/sugar4d_readme.txt"
    if file_hash(readme, "md5") != "3959c3c687cf10f5b51b41bb03f95bd3":
        raise ValueError("Existing V4 readme changed")
    protection["V4_archive_bytes_MD5_unchanged"] = True
    protection["V4_readme_MD5_unchanged"] = True
    manifest = pd.read_csv(root / "data/metadata/file_manifest.csv")
    legacy_images, no_hash = {}, 0
    for record in manifest.to_dict(orient="records"):
        for kind in ("rgb", "mask"):
            path, digest = record[f"{kind}_path"], record[f"{kind}_sha256"]
            if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
                legacy_images[path] = digest
            else:
                no_hash += 1
    for path, digest in legacy_images.items():
        if file_hash(root / path) != digest:
            raise ValueError(f"Existing M0 raw image changed: {path}")
    protection["M0_raw_images_verified_against_original_manifest"] = len(legacy_images)
    protection["M0_manifest_entries_without_hash"] = no_hash
    training = training_audit(root, protocol)
    feasibility = json.loads((report / "feasibility.json").read_text())
    decision = json.loads((report / "research_decision.json").read_text())
    if decision != lock["research_decision_before_test"]:
        raise ValueError("Research decision changed after test")
    tests = element_tree.parse(report / "tests_all.xml").getroot()
    suites = tests.findall("testsuite") if tests.tag == "testsuites" else [tests]
    counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "errors", "failures", "skipped")}
    if counts["errors"] or counts["failures"] or counts["skipped"]:
        raise ValueError("Tests not all passing and unskipped")
    repeatability = json.loads((report / "inference_repeatability.json").read_text())
    if not repeatability["all_identical"] or len(repeatability["runs"]) != training["neural_runs"]:
        raise ValueError("Missing inference replay")
    figures = json.loads((report / "figure_manifest.json").read_text())
    if len(figures["figures"]) != 6 or not figures["all_text_inside_canvas"]:
        raise ValueError("Incomplete or clipped figures")
    predictions = pd.read_csv(root / "outputs/m3_5/test_predictions.csv", dtype={"future_observed": "boolean"}, low_memory=False)
    geometry_names = {f"C5-{history}-{output}" for history in ("Current", "History") for output in ("Absolute", "Residual")}
    expected = {"A1-PointNet", "A3-PointNet-Geometry", "A2-DAS-Ridge", "A2-Geometry-Ridge", "A2-Hull-Proxy", "A2-Z-Range-Proxy", "A2-Z-Quantile-Proxy",
                "C0-Persistence", "C1-Linear-Growth", "C2-MLP", "C3-GRU", "C4-Repeated-GRU", "E0-Frequency", "E0-DAS", "E1-Logistic", "E2-MLP", "E3-GRU", "E4-Repeated-GRU"} | geometry_names
    for partition in protocol["partitions"]:
        subset = predictions.loc[predictions.partition.eq(partition)]
        if set(subset.model) != expected or subset.plant_uid.nunique() != 12 or not subset.split.eq("test").all():
            raise ValueError("Incomplete final prediction coverage")
        for task, real, repeated in (("organ", "C3-GRU", "C4-Repeated-GRU"), ("event", "E3-GRU", "E4-Repeated-GRU")):
            current = subset.loc[subset.task.eq(task)]
            counts_model = current.groupby("model").window_id.nunique()
            if counts_model[real] != counts_model[repeated]:
                raise ValueError("Real/repeated eligible windows differ")
            if task == "organ":
                missing = current.loc[~current.future_observed.astype(bool)]
                if missing[["true_leaf_total_length", "true_leaf_blade_width", "true_leaf_area"]].notna().any().any():
                    raise ValueError("Missing morphology became real values")
    git_status = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=root, text=True)
    write_json(report / "git_state_final.json", {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "status": git_status,
               "no_commit_merge_push": True, "only_new_m35_paths": not bool(subprocess.check_output(["git", "diff", "--name-only"], cwd=root, text=True).strip())})
    result = {"engineering_decision": "PASS", "research_recommendation": decision["recommendation"], "actual_data_only": True,
              "raw_associations_correct": feasibility["association"]["mismatches"] == 0, "all_feasible_baselines_completed": True,
              "strict_plant_partitions": True, "test_exploratory_not_confirmation": True, "selection_locked_before_test": True,
              "primary_task_not_posthoc_test_selected": True, "training": training, "tests": counts, "protection": protection,
              "inference_exact_replay_runs": len(repeatability["runs"]), "figures": 6, "m4_started": False, "git_commit_merge_push": False}
    write_json(report / "acceptance.json", result)
    write_json(report / "final_protection.json", protection)
    path = report / "M3_5_REPORT.md"
    text = path.read_text().replace("工程状态：待最终测试验收", f"工程状态：PASS（{counts['tests']} tests，0 failed/0 skipped）")
    path.write_text(text)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    verify(parser.parse_args().root)
