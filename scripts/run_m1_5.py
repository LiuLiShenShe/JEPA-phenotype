#!/usr/bin/env python3
"""Isolated M1.5 audit, feasibility, fixed-budget training and plant-level diagnosis."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.m1_feature_dataset import FeatureStore, M1Index
from m15.artifacts import save_once, snapshot, verify_snapshot
from m15.data import DiagnosticDataset, calendar_windows, feasibility
from m15.engine import predict, train
from m15.formatting import markdown_table
from m15.models import model_spec
from models.growth_baseline import LinearGrowth, TimeOnly
from training.common import TARGET_NAMES, file_hash, write_json
from training.metrics import evaluate, prediction_frame, selection_score

REPORT = ROOT / "reports/m1_5"
OUTPUT = ROOT / "outputs/m1_5"
PROTOCOL = ROOT / "configs/m1_5/protocol.json"


def review_legacy(index):
    original = pd.read_csv(ROOT / "reports/m1/baseline_comparison.csv")
    legacy_plants = pd.read_csv(ROOT / "reports/m1/plant_level_errors.csv")
    checks = []
    for record in original.itertuples():
        selected = legacy_plants.loc[legacy_plants.model.eq(record.model) & legacy_plants.horizon.eq(record.horizon) & legacy_plants.target.eq(record.target)]
        recalculated = selected.groupby("seed").mae.mean().mean()
        if not np.isclose(recalculated, record.plant_macro_mae_mean, rtol=0, atol=1e-12):
            raise ValueError("Original plant errors and comparison disagree")
        checks.append({"model": record.model, "horizon": record.horizon, "target": record.target,
                       "original_mae": record.plant_macro_mae_mean, "recalculated_mae": recalculated})
    predictions = []
    for directory in sorted((ROOT / "outputs/m1/predictions").iterdir()):
        if not directory.is_dir():
            continue
        rows = pd.read_csv(directory / "test.csv")
        horizon = int(rows.forecast_horizon.iloc[0])
        expected = index.rows.loc[index.rows.split.eq("test") & index.rows.forecast_horizon.eq(horizon)]
        if set(rows.window_id) != set(expected.window_id) or rows.plant_uid.nunique() != 46:
            raise ValueError("Legacy predictions have different test coverage")
        target = rows[[f"true_{name}" for name in TARGET_NAMES]].to_numpy(dtype=np.float32)
        predicted = rows[[f"pred_{name}" for name in TARGET_NAMES]].to_numpy()
        metrics, plants = evaluate(rows, target, predicted, bootstrap_replicates=10)
        saved = json.loads((directory / "test_metrics.json").read_text())
        for name in TARGET_NAMES:
            if not np.isclose(metrics[name]["plant_macro_mae"], saved[name]["plant_macro_mae"], atol=1e-12, rtol=0):
                raise ValueError("Legacy prediction metric mismatch")
        predictions.append({"run": directory.name, "windows": len(rows), "plants": metrics["plants"]})
    checkpoints = []
    for path in sorted((ROOT / "outputs/m1/checkpoints").glob("*/checkpoint.pt")):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["test_used_for_selection"] or checkpoint["selection_split"] != "val":
            raise ValueError("Legacy model selection provenance failed")
        if set(checkpoint["train_plants"]) & set(checkpoint["val_plants"]):
            raise ValueError("Legacy train/validation plant overlap")
        checkpoints.append({"run": path.parent.name, "sha256": file_hash(path), "best_epoch": checkpoint["best_epoch"]})
    if len(predictions) != 27 or len(checkpoints) != 18:
        raise ValueError("Legacy experiment inventory is incomplete")
    pd.DataFrame(checks).to_csv(REPORT / "legacy_metric_recheck.csv", index=False)
    return {"status": "PASS", "prediction_runs_checked": predictions, "neural_checkpoints_checked": checkpoints,
            "original_comparison_rows": len(original), "original_test_already_inspected": True}


def prepare():
    snapshot(ROOT)
    protocol = json.loads(PROTOCOL.read_text())
    locked_path = REPORT / "protocol_lock.json"
    if locked_path.exists():
        write_feasibility_document(load_lock())
        print("Reusing immutable M1.5 protocol and feasibility results", flush=True)
        return
    initial = verify_snapshot(ROOT, raw=True)
    write_json(REPORT / "initial_protection.json", initial)
    index = M1Index(ROOT)
    write_json(REPORT / "legacy_review.json", review_legacy(index))
    cache_info = json.loads((ROOT / "reports/m1/feature_cache.json").read_text())
    store = FeatureStore(ROOT / cache_info["directory"])
    rows = calendar_windows(index.frames, index.split["assignments"], protocol["day_horizons"], protocol["matching"]["tolerance_hours"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUTPUT / "calendar_windows.csv", index=False)
    summary, per_plant, coverage, eligible = feasibility(rows, index.split["assignments"], protocol)
    summary.to_csv(REPORT / "calendar_feasibility.csv", index=False)
    per_plant.to_csv(REPORT / "calendar_windows_per_plant.csv", index=False)
    coverage.to_csv(REPORT / "calendar_stage_genotype_coverage.csv", index=False)
    stage = rows.groupby(["forecast_horizon", "split", "history_growth_stage"]).agg(plants=("plant_uid", "nunique"), windows=("window_id", "size")).reset_index()
    stage.to_csv(REPORT / "calendar_stage_coverage.csv", index=False)
    genotype = rows.groupby(["forecast_horizon", "split", "genotype"]).agg(plants=("plant_uid", "nunique"), windows=("window_id", "size")).reset_index()
    genotype.to_csv(REPORT / "calendar_genotype_coverage.csv", index=False)
    fingerprint_paths = ["data/metadata/frames.csv", "data/processed/temporal_windows.csv", "splits/phenoleaf_plant_split_seed42.json",
                         "outputs/m1_5/calendar_windows.csv", "reports/m1_5/calendar_feasibility.csv", "reports/m1_5/calendar_windows_per_plant.csv",
                         "reports/m1_5/calendar_stage_genotype_coverage.csv", "reports/m1/feature_cache.json", "reports/m1/model_selection.json"]
    lock = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol, "protocol_sha256": file_hash(PROTOCOL),
        "eligible_days": eligible, "eligibility_uses": ["train", "val"], "test_model_performance_used": False,
        "test_was_inspected_in_m1": True, "data_fingerprints": {path: file_hash(ROOT / path) for path in fingerprint_paths},
        "feature_matrix_sha256": store.metadata["features_sha256"], "feature_dimension": store.metadata["feature_dimension"],
    }
    save_once(locked_path, lock)
    write_feasibility_document(lock)


def write_feasibility_document(lock):
    summary = pd.read_csv(REPORT / "calendar_feasibility.csv")
    eligible = lock["eligible_days"]
    table = summary[["days", "split", "plants", "windows", "genotypes", "windows_per_plant_median", "absolute_match_error_hours_p95"]]
    lines = ["# M1.5 真实时间尺度任务可行性（训练之前冻结）", "", f"锁定 UTC：{lock['created_utc']}",
             "", "固定 ±6 小时，按同株最近有效未来时间戳唯一匹配；等距取较早时间。只用 Train/Validation 判定是否可训练，Test 只列覆盖。",
             "独立植物阈值 Train≥100、Val≥25，二者每株窗口中位数≥5；这是工程可行性门槛，不是统计功效保证。",
             "", markdown_table(table), "", f"可训练跨度：{eligible} 天；所有方法共用同一窗口及请求时间，不输入匹配残差或目标观测编号。",
             "实际标签相对请求时刻有匹配误差；不能将其解释成精确指定时刻的无噪声标签。",
             "每株包括零窗口记录见 calendar_windows_per_plant.csv；阶段/基因型完整交叉覆盖见 calendar_stage_genotype_coverage.csv。"]
    (REPORT / "CALENDAR_FEASIBILITY.md").write_text("\n".join(lines) + "\n")
    write_json(REPORT / "environment.json", {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
               "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
               "gpu_names": [torch.cuda.get_device_name(position) for position in range(torch.cuda.device_count())],
               "reused_environment": ".venv-m1", "m0_environment_modified": False})
    print(table.to_string(index=False), flush=True)


def load_lock():
    lock = json.loads((REPORT / "protocol_lock.json").read_text())
    if file_hash(PROTOCOL) != lock["protocol_sha256"]:
        raise ValueError("Protocol was changed after feasibility lock")
    for path, digest in lock["data_fingerprints"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"Locked task/data artifact changed: {path}")
    return lock


def load_data():
    lock = load_lock()
    index = M1Index(ROOT)
    cache_info = json.loads((ROOT / "reports/m1/feature_cache.json").read_text())
    store = FeatureStore(ROOT / cache_info["directory"])
    calendar = pd.read_csv(OUTPUT / "calendar_windows.csv")
    return lock, index, store, calendar


def tasks(lock, index, calendar, families):
    for family in families:
        horizons = lock["protocol"]["offsets"] if family == "offsets" else lock["eligible_days"]
        source = index.rows if family == "offsets" else calendar
        for horizon in horizons:
            yield family, horizon, source.loc[source.forecast_horizon.eq(horizon)].reset_index(drop=True)


def train_all(device, families):
    lock, index, store, calendar = load_data()
    protocol = lock["protocol"]
    training_sources = ["src/m15/data.py", "src/m15/models.py", "src/m15/engine.py", "scripts/run_m1_5.py"]
    save_once(REPORT / "training_source_fingerprints.json", {path: file_hash(ROOT / path) for path in training_sources})
    for family, horizon, rows in tasks(lock, index, calendar, families):
        models = protocol["offset_models"] if family == "offsets" else protocol["day_models"]
        datasets = {}
        for model in models:
            spec = model_spec(model, protocol)
            condition = spec["condition"]
            for split in ("train", "val"):
                key = split, condition
                if key not in datasets:
                    full = datasets.get((split, "full"))
                    datasets[key] = DiagnosticDataset(rows.loc[rows.split.eq(split)], index.frames, store,
                                                      task_family=family, condition=condition, features=None if full is None else full.features)
            for seed in protocol["seeds"]:
                config = {**spec, "task": f"{'offset' if family == 'offsets' else 'day'}{horizon}", "task_family": family,
                          "horizon": horizon, "seed": seed, "protocol_lock_sha256": file_hash(REPORT / "protocol_lock.json"),
                          "feature_cache_signature": lock["feature_matrix_sha256"]}
                directory = OUTPUT / "checkpoints" / f"{config['task']}__{model}__s{seed}"
                train(datasets[("train", condition)], datasets[("val", condition)], config, directory, device)
    registry = []
    for path in sorted((OUTPUT / "checkpoints").glob("*/resources.json")):
        config = json.loads((path.parent / "config.json").read_text())
        registry.append({**config, **json.loads(path.read_text()), "checkpoint_sha256": file_hash(path.parent / "checkpoint.pt")})
    pd.DataFrame(registry).to_csv(REPORT / "training_resources.csv", index=False)
    expected = len(protocol["seeds"]) * (len(protocol["offsets"]) * len(protocol["offset_models"]) + len(lock["eligible_days"]) * len(protocol["day_models"]))
    write_json(REPORT / "training_complete.json", {"runs": len(registry), "expected_runs": expected,
               "status": "complete" if len(registry) == expected else "partial", "test_used_for_selection": False})


def record_prediction(rows, targets, predictions, *, model, seed, task, split):
    directory = OUTPUT / "predictions" / f"{task}__{model}__s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{split}.csv"
    if destination.exists():
        saved = pd.read_csv(destination)
        if not np.allclose(saved[[f"pred_{name}" for name in TARGET_NAMES]], predictions, rtol=0, atol=2e-6):
            raise ValueError("Existing M1.5 predictions disagree; refusing overwrite")
        return
    metrics, plants = evaluate(rows, targets, predictions, bootstrap_replicates=2000)
    frame = prediction_frame(rows, targets, predictions, model=model, seed=seed)
    frame["task"] = task
    frame["query_contract"] = "requested fixed days; actual matching interval hidden" if task.startswith("day") else "see model condition and protocol"
    if "requested_delta_days" in rows:
        frame["requested_delta_days"] = rows.requested_delta_days.to_numpy()
        frame["matching_error_hours"] = rows.matching_error_hours.to_numpy()
    frame.to_csv(destination, index=False)
    plants.assign(model=model, seed=seed, task=task, split=split).to_csv(directory / f"{split}_plants.csv", index=False)
    write_json(directory / f"{split}_metrics.json", {"task": task, "model": model, "seed": seed, "split": split, **metrics})


def evaluate_all(device, families):
    lock, index, store, calendar = load_data()
    protocol = lock["protocol"]
    for family, horizon, rows in tasks(lock, index, calendar, families):
        task = f"{'offset' if family == 'offsets' else 'day'}{horizon}"
        models = protocol["offset_models"] if family == "offsets" else protocol["day_models"]
        for model in models:
            condition = model_spec(model, protocol)["condition"]
            for split in ("val", "test"):
                dataset = DiagnosticDataset(rows.loc[rows.split.eq(split)], index.frames, store, task_family=family, condition=condition)
                for seed in protocol["seeds"]:
                    checkpoint = OUTPUT / "checkpoints" / f"{task}__{model}__s{seed}" / "checkpoint.pt"
                    prediction = predict(checkpoint, dataset, device)
                    record_prediction(dataset.rows, dataset.targets, prediction, model=model, seed=seed, task=task, split=split)
        if family == "days":
            simple_day_baselines(task, horizon, rows, index, store)
    write_json(REPORT / "evaluation_complete.json", {"status": "complete", "tasks": [f"offset{value}" for value in protocol["offsets"]]
               + [f"day{value}" for value in lock["eligible_days"]], "prediction_directories": len(list((OUTPUT / "predictions").glob("*"))),
               "test_used_for_selection": False})


def oracle_data(rows, index):
    indices = np.asarray([json.loads(value) for value in rows.input_frame_indices])
    history = np.stack([index.frame_lookup.loc[pd.MultiIndex.from_arrays([rows.plant_uid, indices[:, position]]),
                       ["leaf_count", "projected_area_ratio"]].to_numpy() for position in range(3)], axis=1)
    times = np.asarray([pd.to_datetime(json.loads(value)).to_numpy() for value in rows.input_timestamps])
    relative = (times - times[:, :1]) / np.timedelta64(1, "D")
    return history, relative


def simple_day_baselines(task, horizon, rows, index, store):
    train_data = DiagnosticDataset(rows.loc[rows.split.eq("train")], index.frames, store, task_family="days", condition="full")
    val_data = DiagnosticDataset(rows.loc[rows.split.eq("val")], index.frames, store, task_family="days", condition="full")
    candidates = []
    for alpha in (.1, 1., 10., 100.):
        model = TimeOnly(alpha).fit(train_data.times, train_data.targets, train_data.rows)
        score = selection_score(val_data.rows, val_data.targets, model.predict(val_data.times), model.target_scaler.scale)
        candidates.append((score, alpha, model))
    score, alpha, model = min(candidates, key=lambda entry: entry[0])
    save_once(OUTPUT / "simple" / f"{task}_B1b.json", {"selection_split": "val", "test_used_for_selection": False,
              "val_score": score, "alpha": alpha, "candidates": [{"alpha": entry[1], "val_score": entry[0]} for entry in candidates], "state": model.state()})
    for split in ("val", "test"):
        dataset = DiagnosticDataset(rows.loc[rows.split.eq(split)], index.frames, store, task_family="days", condition="full")
        history, times = oracle_data(dataset.rows, index)
        predictions = {"B0": history[:, -1], "B1": LinearGrowth().predict(history, times, np.full(len(dataset), horizon)),
                       "B1b": model.predict(dataset.times)}
        for name, prediction in predictions.items():
            record_prediction(dataset.rows, dataset.targets, prediction, model=name, seed=0, task=task, split=split)


def main(args):
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("train", "all"):
        train_all(args.device, args.families)
    if args.stage in ("evaluate", "all"):
        evaluate_all(args.device, args.families)
    if args.stage in ("analyze", "all"):
        from m15.analysis import analyze
        analyze(ROOT)
    if args.stage in ("report", "all"):
        from m15.report import render_report
        render_report(ROOT)
    if args.stage == "verify":
        write_json(REPORT / "final_protection.json", verify_snapshot(ROOT, raw=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "train", "evaluate", "analyze", "report", "verify", "all"], required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--families", nargs="+", choices=["offsets", "days"], default=["offsets", "days"])
    main(parser.parse_args())
