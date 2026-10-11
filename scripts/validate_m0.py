#!/usr/bin/env python3
"""Validate the complete, local M0 evidence bundle without training a model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from create_splits import assign_plants
from datasets.phenoleaf_utils import is_usable_quality_status, parse_timestamp
from datasets.plant_temporal_dataset import PlantTemporalDataset
from download_data import EXPECTED_ARCHIVE_SHA256, OFFICIAL_REPO_COMMIT


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name: str, passed: bool, details: Any, *, warning: bool = False) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "WARN" if warning else "FAIL", "details": details}


def validate_metadata(metadata_dir: Path, project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frames = read_csv(metadata_dir / "frames.csv") if (metadata_dir / "frames.csv").exists() else []
    plants = read_csv(metadata_dir / "plants.csv") if (metadata_dir / "plants.csv").exists() else []
    issues = read_csv(metadata_dir / "data_issues.csv") if (metadata_dir / "data_issues.csv").exists() else []
    checks = [check("Nonempty metadata", bool(frames), {"frames": len(frames), "plants": len(plants)})]
    invalid_ids, timing_errors, path_errors = [], [], []
    by_plant = defaultdict(list)
    for row in frames:
        by_plant[row["plant_uid"]].append(row)
        if row["plant_uid"] != f"{row['genotype']}/{row['replicate']}":
            invalid_ids.append(row["plant_uid"])
        rgb_path, mask_path = Path(row["rgb_path"]), Path(row["mask_path"])
        if (rgb_path.parent.name, rgb_path.parent.parent.name) != (row["replicate"], row["genotype"]):
            path_errors.append(row["rgb_path"])
        if rgb_path.name != mask_path.name or rgb_path.name != row["raw_filename"]:
            path_errors.append(row["rgb_path"])
    reliable_plants = 0
    for plant_uid, rows in by_plant.items():
        rows.sort(key=lambda row: int(row["frame_idx"]))
        if [int(row["frame_idx"]) for row in rows] != list(range(len(rows))):
            timing_errors.append(plant_uid)
        if any(int(row["time_index"]) != int(row["frame_idx"]) for row in rows):
            timing_errors.append(plant_uid)
        reliable = all(row["timestamp_reliable"].lower() == "true" for row in rows)
        if reliable:
            reliable_plants += 1
            try:
                dates = [datetime.fromisoformat(row["timestamp"]) for row in rows]
                if dates != [parse_timestamp(row["raw_filename"])[0] for row in rows]:
                    timing_errors.append(plant_uid)
                if any(later <= earlier for earlier, later in zip(dates, dates[1:])):
                    timing_errors.append(plant_uid)
            except ValueError:
                timing_errors.append(plant_uid)
        elif any(is_usable_quality_status(row["quality_status"]) for row in rows):
            timing_errors.append(plant_uid)
    keys = {(row["plant_uid"], row["raw_filename"]) for row in frames}
    rgb_paths = {row["rgb_path"] for row in frames}
    mask_paths = {row["mask_path"] for row in frames}
    checks.extend([
        check("Plant UID and file paths", not invalid_ids and not path_errors, {"id_errors": len(invalid_ids), "path_errors": len(path_errors)}),
        check("RGB/mask one-to-one row uniqueness", len(keys) == len(rgb_paths) == len(mask_paths) == len(frames), {"frames": len(frames), "unique_pairs": len(keys)}),
        check("Plant inventory matches frame inventory", len(plants) == len(by_plant) and {row['plant_uid'] for row in plants} == set(by_plant), len(by_plant)),
        check("Strict verified within-plant time order", not timing_errors, {"problem_plants": sorted(set(timing_errors)), "reliable_plants": reliable_plants}),
    ])
    ratios_valid = all(0 <= float(row["projected_area_ratio"]) <= 1 for row in frames if is_usable_quality_status(row["quality_status"]))
    checks.append(check("Projected area ratios in [0,1]", ratios_valid, "all usable frames"))
    illegal = [row for row in frames if int(row.get("invalid_color_pixel_count") or 0) > 0]
    recorded_illegal = {(row["plant_uid"], row["raw_filename"]) for row in issues if row["issue_type"] == "invalid_palette_colors"}
    checks.append(check("Illegal colors recorded and quarantined", all(row["quality_status"] != "valid" and (row["plant_uid"], row["raw_filename"]) in recorded_illegal for row in illegal), {"quarantined_illegal_color_frames": len(illegal)}))
    phenotypes = read_csv(metadata_dir / "phenotypes.csv") if (metadata_dir / "phenotypes.csv").exists() else []
    label_fields = ("leaf_count", "projected_area_px", "projected_area_ratio", "known_projected_area_px", "quality_status")
    frame_map = {(row["plant_uid"], row["frame_idx"]): row for row in frames}
    label_consistency = len(phenotypes) == len(frames) and len({(row["plant_uid"], row["frame_idx"]) for row in phenotypes}) == len(phenotypes)
    label_consistency = label_consistency and all(all(row[field] == frame_map.get((row["plant_uid"], row["frame_idx"]), {}).get(field) for field in label_fields) for row in phenotypes)
    checks.append(check("Phenotypes and canonical frames agree", label_consistency, {"phenotypes": len(phenotypes)}))
    return checks, {"frames": frames, "plants": plants, "issues": issues, "reliable_plants": reliable_plants}


def validate_splits(split_path: Path, frames: list[dict[str, str]], *_unused: Any) -> list[dict[str, Any]]:
    payload = read_json(split_path)
    assignments = payload.get("assignments", {})
    sets = {split: set(payload.get(f"{split}_plants", [])) for split in ("train", "val", "test")}
    disjoint = sets["train"].isdisjoint(sets["val"]) and sets["train"].isdisjoint(sets["test"]) and sets["val"].isdisjoint(sets["test"])
    lists_match = set.union(*sets.values()) == set(assignments) and all(assignments.get(plant_uid) == split for split, plant_uids in sets.items() for plant_uid in plant_uids)
    expected_images = {row["rgb_path"]: assignments.get(row["plant_uid"]) for row in frames if is_usable_quality_status(row["quality_status"])}
    recorded_images = payload.get("image_to_split", {})
    return [
        check("Nonempty plant-level split with seed 42", all(sets.values()) and payload.get("seed") == 42 and payload.get("official_split_not_used") is True, {split: len(plants) for split, plants in sets.items()}),
        check("Plant split IDs disjoint and assignments consistent", disjoint and lists_match, "train∩val = train∩test = val∩test = empty"),
        check("All usable images in exactly one plant split", expected_images == recorded_images and all(value in sets for value in expected_images.values()), {"assigned_images": len(recorded_images)}),
        check("Seed 42 split reproducibility", assign_plants(frames, seed=42) == assignments, "independently recomputed assignments"),
        check("Quarantined images excluded from split index", all(row["rgb_path"] not in recorded_images for row in frames if not is_usable_quality_status(row["quality_status"])), payload.get("excluded_frame_count", 0)),
    ]


def validate_windows(window_path: Path, split_path: Path, frames: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = read_csv(window_path) if window_path.exists() else []
    assignments = read_json(split_path).get("assignments", {})
    lookup = {(row["plant_uid"], int(row["frame_idx"])): row for row in frames}
    errors = Counter()
    observed_keys = set()
    for row in rows:
        try:
            indices = json.loads(row["input_frame_indices"])
            paths = json.loads(row["input_image_paths"])
            timestamps = json.loads(row["input_timestamps"])
            deltas = json.loads(row["time_deltas"])
            target_idx, horizon = int(row["target_frame_idx"]), int(row["forecast_horizon"])
            history = [lookup[(row["plant_uid"], int(index))] for index in indices]
            target = lookup[(row["plant_uid"], target_idx)]
            observed_keys.add((row["plant_uid"], indices[-1], horizon))
            if len(indices) != 3 or int(row["history_length"]) != 3 or indices != list(range(indices[-1] - 2, indices[-1] + 1)):
                errors["consecutive_history"] += 1
            if paths != [frame["rgb_path"] for frame in history] or timestamps != [frame["timestamp"] for frame in history]:
                errors["cross_plant_or_path_timestamp_mismatch"] += 1
            if target_idx != indices[-1] + horizon or horizon not in {2, 4, 8} or not all(index < target_idx for index in indices):
                errors["original_observation_horizon"] += 1
            if any(not is_usable_quality_status(frame["quality_status"]) for frame in history + [target]):
                errors["quarantine_leak"] += 1
            if assignments.get(row["plant_uid"]) != row["split"]:
                errors["split_mismatch"] += 1
            if int(row["target_leaf_count"]) != int(target["leaf_count"]) or float(row["target_projected_area"]) != float(target["projected_area_px"]) or float(row["target_projected_area_ratio"]) != float(target["projected_area_ratio"]):
                errors["future_gt_labels"] += 1
            dates = [datetime.fromisoformat(frame["timestamp"]) for frame in history]
            target_date = datetime.fromisoformat(target["timestamp"])
            expected_deltas = [0.0] + [(later - earlier).total_seconds() for earlier, later in zip(dates, dates[1:])]
            if deltas != expected_deltas or row["target_timestamp"] != target["timestamp"] or float(row["target_time_delta_seconds"]) != (target_date - dates[-1]).total_seconds() or target_date <= dates[-1]:
                errors["physical_time_information"] += 1
        except (ValueError, KeyError, TypeError, IndexError):
            errors["malformed_or_missing_frame"] += 1
    expected_keys = set()
    for plant_uid, end_idx in lookup:
        history = [lookup.get((plant_uid, frame_idx)) for frame_idx in range(end_idx - 2, end_idx + 1)]
        if any(frame is None or not is_usable_quality_status(frame["quality_status"]) for frame in history):
            continue
        for horizon in (2, 4, 8):
            target = lookup.get((plant_uid, end_idx + horizon))
            if target is not None and is_usable_quality_status(target["quality_status"]):
                expected_keys.add((plant_uid, end_idx, horizon))
    return [
        check("Nonempty unique windows for horizons 2/4/8", bool(rows) and len(rows) == len({row["window_id"] for row in rows}) and {int(row["forecast_horizon"]) for row in rows} == {2, 4, 8}, {"windows": len(rows)}),
        check("All windows: same plant, future GT, original offsets", not errors, dict(errors)),
        check("All eligible windows generated exactly once", observed_keys == expected_keys and len(observed_keys) == len(rows), {"expected": len(expected_keys), "observed": len(observed_keys)}),
    ]


def validate_evidence(metadata_dir: Path, report_dir: Path, download: dict[str, Any], *, synthetic: bool) -> list[dict[str, Any]]:
    audit = read_json(report_dir / "full_dataset_audit.json")
    checks = [check("Full-file independent audit", audit.get("status") == "PASS" and audit.get("frames_audited", 0) > 0, {key: value for key, value in audit.items() if key != "duplicates"})]
    fresh = audit.get("metadata_sha256") == digest(metadata_dir / "frames.csv") and audit.get("leaf_instances_sha256") == digest(metadata_dir / "leaf_instances.csv")
    checks.append(check("Audit evidence matches current metadata", fresh, "SHA-256 of canonical frames and leaf instances"))
    frame_count = len(read_csv(metadata_dir / "frames.csv"))
    checks.append(check("Every frame and mask audited", audit.get("frames_audited") == audit.get("masks_independently_recomputed") == frame_count, frame_count))
    reproducibility = read_json(report_dir / "reproducibility.json")
    reproducible = reproducibility.get("status") == "PASS" and all(digest(PROJECT_ROOT / path) == expected for path, expected in reproducibility.get("sha256_after", {}).items())
    checks.append(check("Byte-identical full pipeline rerun", reproducible, reproducibility.get("changed_files", "missing evidence")))
    if not synthetic:
        provenance_ok = download.get("status") == "ok" and download.get("archive_testzip") == "passed" and download.get("layout_verified") is True and download.get("sha256_matches_hf_lfs") is True
        lfs_oid = download.get("hub_metadata_verification", {}).get("archive", {}).get("lfs", {}).get("oid")
        provenance_ok = provenance_ok and download.get("archive_sha256") == lfs_oid == EXPECTED_ARCHIVE_SHA256 and audit.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256 and audit.get("archive_crc_comparison_enabled") is True
        checks.append(check("Full archive identity, extraction CRCs and provenance", provenance_ok, {"archive_bytes": download.get("archive_bytes"), "sha256": download.get("archive_sha256"), "source_mode": download.get("source_mode")}))
        try:
            commit = subprocess.check_output(["git", "-C", str(PROJECT_ROOT / "third_party/PhenoLeaf-TS"), "rev-parse", "HEAD"], text=True).strip()
        except subprocess.SubprocessError:
            commit = "unavailable"
        checks.append(check("Official repository commit pinned", commit == OFFICIAL_REPO_COMMIT, commit))
    return checks


def validate_tests(report_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    path = report_dir / "test_results.xml"
    cases = []
    if path.exists():
        for case in ElementTree.parse(path).getroot().iter("testcase"):
            status = "FAIL" if case.find("failure") is not None or case.find("error") is not None else "SKIP" if case.find("skipped") is not None else "PASS"
            cases.append({"name": f"{case.get('classname')}::{case.get('name')}", "status": status, "seconds": case.get("time", "")})
    return [check("Automated pytest suite", bool(cases) and all(case["status"] == "PASS" for case in cases), {"tests": len(cases), "results": dict(Counter(case['status'] for case in cases))})], cases


def validate_figures(report_dir: Path) -> list[dict[str, Any]]:
    figures = report_dir / "figures"
    manifest = read_json(figures / "selection_manifest.json")
    paths = sorted(figures.glob("*.png"))
    minimum_counts = len(manifest.get("overlays", [])) >= 12 and len(manifest.get("growth_curves", [])) >= 6 and len(manifest.get("windows", [])) >= 6
    from PIL import Image

    readable = True
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            readable = False
    return [check("Local review figures and selection coverage", len(paths) >= 9 and readable and minimum_counts, {"png_files": len(paths), "overlays": len(manifest.get('overlays', [])), "growth_plants": len(manifest.get('growth_curves', [])), "temporal_windows": len(manifest.get('windows', []))})]


def dataset_smoke(metadata_dir: Path, split_path: Path, windows_path: Path, project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results = []
    try:
        import torch
        from torch.utils.data import DataLoader

        torch.set_num_threads(2)
        for split in ("train", "val", "test"):
            for horizon in (2, 4, 8):
                dataset = PlantTemporalDataset(metadata_path=metadata_dir / "frames.csv", split_path=split_path, windows_path=windows_path, split=split, horizon=horizon, image_size=(224, 224), project_root=project_root)
                sample = dataset[0]
                assert tuple(sample["images"].shape) == (3, 3, 224, 224)
                assert bool(torch.isfinite(sample["images"]).all())
                assert float(sample["images"].min()) >= 0 and float(sample["images"].max()) <= 1
                assert sample["target_frame_idx"] == sample["input_frame_indices"][-1] + horizon
                assert sample["target_time_delta_seconds"] > 0
                results.append({"split": split, "horizon": horizon, "samples": len(dataset), "plant_uid": sample["plant_uid"], "shape": list(sample["images"].shape), "status": "PASS"})
        batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
        assert tuple(batch["images"].shape) == (2, 3, 3, 224, 224)
        results.append({"dataloader_batch_shape": list(batch["images"].shape), "device": str(batch["images"].device), "status": "PASS"})
    except Exception as error:
        results.append({"status": "FAIL", "error": f"{type(error).__name__}: {error}"})
    return [check("Real Dataset samples and DataLoader on CPU", len(results) == 10 and all(result["status"] == "PASS" for result in results), results)], results


def render_reports(checks: list[dict[str, Any]], summary: dict[str, Any], download: dict[str, Any], report_dir: Path, *, synthetic: bool = False) -> str:
    overall = "FAIL" if any(item["status"] == "FAIL" for item in checks) else "PARTIAL" if synthetic else "PASS"
    frames, issues = summary["frames"], summary["issues"]
    usable = [row for row in frames if row["quality_status"] == "valid"]
    plant_uids = {row["plant_uid"] for row in frames}
    split_payload = summary["split_payload"]
    windows = summary["windows"]
    issue_counts = dict(sorted(Counter(row["issue_type"] for row in issues).items()))
    excluded = [row for row in frames if row["quality_status"] != "valid"]
    counts = {"all_frame_rows": len(frames), "usable_frames": len(usable), "quarantined_frames": len(excluded), "plants": len(plant_uids), "genotypes": len({row["genotype"] for row in frames}), "issues": len(issues), "temporal_windows": len(windows)}
    stats = read_json(summary["metadata_dir"] / "dataset_statistics.json")
    geometry_stats = read_json(report_dir / "geometry_time_and_label_statistics.json")
    stats.update({"m0_status": overall, "data_scope": "synthetic_only" if synthetic else "real_dataset", "counts": counts, "split_counts": split_payload.get("counts", {}), "windows_by_split": dict(Counter(row["split"] for row in windows)), "windows_by_horizon": dict(Counter(row["forecast_horizon"] for row in windows)), "issue_counts": issue_counts, "m1_ready": overall == "PASS" and not synthetic})
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "validation_results.json").write_text(json.dumps({"status": overall, "data_scope": stats["data_scope"], "checks": checks}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "dataset_smoke.json").write_text(json.dumps(summary["smoke_results"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (report_dir / "test_results.json").write_text(json.dumps(summary["tests"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# M0 交付报告 — Plant Temporal-JEPA", "", f"**M0 状态：{overall}**", "", "## A. 执行结果", f"- 工作目录：`{summary['project_root']}`", f"- Python：`{platform.python_version()}`；PyTorch：`{version('torch')}`（CPU-only，无模型训练）。", f"- 官方仓库 commit：`{download.get('official_repository_commit', 'unavailable')}`", f"- Hugging Face dataset commit：`{download.get('dataset_commit', 'unavailable')}`", f"- 下载归档：{download.get('archive_bytes', 0):,} bytes；SHA-256：`{download.get('archive_sha256', 'unavailable')}`", f"- 全量配对观测：{len(frames):,}；有效观测：{len(usable):,}；隔离：{len(excluded):,}。", f"- 独立植株：{len(plant_uids)}；基因型：{counts['genotypes']}。", "", "| 集合 | 植株 | 有效图像 | 基因型 | 时间窗口 |", "|---|---:|---:|---:|---:|"]
    for split, split_counts in split_payload.get("counts", {}).items():
        lines.append(f"| {split} | {split_counts['plants']} | {split_counts['images']} | {split_counts['genotypes']} | {stats['windows_by_split'].get(split, 0)} |")
    lines.extend(["", f"- 时序窗口总数：{len(windows):,}；按跨度：`{stats['windows_by_horizon']}`。", "- 输入 3 个连续的原始观测；预测 +2/+4/+8 **观测索引偏移，不是天数**。", "", "## B. 数据质量", "- 原始图像与 mask 不作修改；所有文件完整解码、SHA-256、归档 CRC、RGB/mask 尺寸、精确调色板标签均由全量独立审计核对。", "- 审计详情、重复文件/像素组与异常计数：`full_dataset_audit.json`；逐文件证据：`../../data/metadata/file_manifest.csv`。", f"- {summary['reliable_plants']}/{len(plant_uids)} 株时间戳通过官方文件名格式、日历合法性、唯一性与严格排序验证。没有独立设备时钟校准或时区记录；保存无时区墙钟时间，不宣称 UTC 或真实天数预测。", "- 黑色背景不计叶片；叶片 ID 是官方 31 色调色板位置（1–31），仅在同一植株内解释身份。", "- 非法颜色不计作新叶片；诊断面积包含全部非黑像素，另存合法像素面积。所有含非法颜色的帧从 split 图像索引、历史输入和未来目标中隔离。", f"- 含隔离帧的植株（仅隔离帧，未删除整株）：`{sorted({row['plant_uid'] for row in excluded})}`。", "- 隔离不压缩观测编号；缺失/异常帧造成的历史窗口或目标缺口均跳过，不能把 +4 改成第 4 个剩余有效帧。", "- 叶片身份消失/重现、面积突变和描述范围外叶数只作观察性预警，可能来自遮挡/衰老，不直接删数据。", f"- 异常事件分类：`{issue_counts}`。同一帧可以对应多条事件，事件数不等于坏图像数。", "- `projected_area_ratio` 是图像面积比例，不是真实物理叶面积，也不是 LAI；原图尺寸不统一，跨植株原始像素面积解释需谨慎。", "- 高度重叠窗口合法，植株级划分避免跨集合泄漏；没有拟合全数据归一化统计量。仅 RGB /255 是固定缩放，未来需拟合的标准化只能使用 train。", "", "## C. 测试情况", "| 检查 | 状态 | 运行证据 |", "|---|---|---|"])
    for item in checks:
        details = json.dumps(item["details"], ensure_ascii=False, separators=(",", ":"))
        if len(details) > 1600:
            details = "见 validation_results.json（完整机器可读证据）"
        lines.append(f"| {item['name']} | {item['status']} | {details.replace('|', '/')} |")
    lines.extend(["", "### 自动化测试逐项结果", "| 测试 | 结果 | 秒 |", "|---|---|---:|"])
    for case in summary["tests"]:
        lines.append(f"| `{case['name']}` | {case['status']} | {case['seconds']} |")
    lines.extend(["", "### 额外数据风险统计", f"- 图像尺寸：{geometry_stats.get('distinct_image_sizes', 'not_run')} 种；同株尺寸变化植株：`{geometry_stats.get('plants_with_size_changes', [])}`。使用 `image_size=(224,224)` 批处理；跨裁剪变化的面积曲线不是物理生长证据。", f"- 非法像素总数：{geometry_stats.get('quarantine', {}).get('invalid_color_pixels', 'not_run')}；非法 RGB 色数：{geometry_stats.get('quarantine', {}).get('distinct_illegal_colors', 'not_run')}。", f"- 隔离观测索引：`{geometry_stats.get('quarantine', {}).get('frame_indices', {})}`。", f"- 同株相邻观测时间间隔：`{geometry_stats.get('adjacent_intervals', {})}`；因此观测偏移不能统一换算为天数。", "- 详情：`geometry_time_and_label_statistics.json`（尺寸变化逐事件、按集合叶数/阶段分布、预测跨度实际秒数）。"])
    lines.append(f"- 隔离对窗口的影响：`{geometry_stats.get('quarantine_window_impact', {})}`。")
    identity = geometry_stats.get("identity", {})
    lines.append(f"- 文件身份编码交叉核验：采集前缀 ID 数 {identity.get('source_acquisition_ids', 'not_run')}，跨目录复用采集 ID `{identity.get('source_ids_shared_across_plant_uids', {})}`；有 {identity.get('filename_replicate_differs_from_directory_frames', 'not_run')} 帧/{identity.get('plants_with_filename_replicate_alias', 'not_run')} 株的文件尾部重复编号与目录编号不同。原目录为权威 plant_uid，采集前缀与目录一一对应；不按文件尾号合并不同植株。逐株证据见 `../../data/metadata/plant_identity_audit.csv`。")
    lines.extend(["", "一条命令重新执行全部验收（包含全量审计、测试和真实 Dataset smoke）：", "```bash", ".venv/bin/python scripts/run_m0_acceptance.py", "```", "", "## D. 可视化与文件", "- CSV：`../../data/metadata/frames.csv`、`plants.csv`、`phenotypes.csv`、`leaf_instances.csv`、`data_issues.csv`、`file_manifest.csv`。", "- 索引：`../../splits/phenoleaf_plant_split_seed42.json`、`../../data/processed/temporal_windows.csv`。", "- 统计：`dataset_statistics.json`、`phenotype_statistics.json`、`split_statistics.csv`、`temporal_window_statistics.json`。", "- 可视化：`figures/` 的 9 张图；`figures/selection_manifest.json` 记录实际选取的植株、阶段、异常及窗口。", "- 完整机器证据：`download_report.json`、`full_dataset_audit.json`、`validation_results.json`、`dataset_smoke.json`、`test_results.xml/json`。", "- 数据来源、许可与网络替代证据：README 与 `SOURCE_AND_LICENSE.md`。所有 RGB/mask 相关可视化仅留本地，未上传第三方视觉服务或公开代码仓库。", "", "## E. 未解决问题与限制", "- 隔离帧的非法调色板颜色未获官方修正；保留原标注，不猜测这些颜色对应哪个叶片。具体位置、像素数量和颜色保存在 data_issues.csv。", "- 描述范围外叶数与身份间歇出现并不证明错误；保留预警，需要人工解释植物遮挡、衰老或标注一致性。", "- 时间仅验证文件编码与顺序，无时区/设备时钟证据；不构建按天跨度任务。", "- 二维投影像素面积依赖裁剪尺度，不当作物理面积或 LAI；RGB resize 仅在加载时进行，不改 GT 面积。", "- 本交付提供完整人工验收材料；不代替用户最终人工验收，不声称全部叶片边界已人工审核。", "- 官方站直连下载遇到网络问题，使用用户允许的镜像替代；归档 SHA-256 对照固定版本 Hugging Face LFS 对象，不能将镜像域名当作官方域名。", "", "## F. 是否具备进入 M1 的条件", f"**{'YES' if overall == 'PASS' and not synthetic else 'NO'}** — 全部验收条件以 validation_results.json 为准；只在真实数据、全量审计、无泄漏索引、Dataset smoke 与测试通过时允许进入 M1。未启动 DINOv2、GRU、JEPA 或任何训练。", "", "## 数据使用许可", "- 数据集：https://huggingface.co/datasets/rick77a/PhenoLeaf_TS", "- 官方定义与预处理：https://github.com/pisyntor/PhenoLeaf-TS", "- La Trobe University Use Terms：仅非商业学术/教学，必须引用；不得未经许可再分发原图/标注或公开托管。工具代码 Apache-2.0 不赋予数据再分发权限。"])
    (report_dir / "M0_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checklist = ["# M0 QA Checklist", "", f"Overall status: **{overall}**", "", "| Requirement | Result | Evidence |", "|---|---|---|"]
    checklist.extend(f"| {item['name']} | {item['status']} | validation_results.json |" for item in checks)
    (report_dir / "QA_CHECKLIST.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    return overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default="data/metadata")
    parser.add_argument("--split", default="splits/phenoleaf_plant_split_seed42.json")
    parser.add_argument("--windows", default="data/processed/temporal_windows.csv")
    parser.add_argument("--download-report", default="reports/m0/download_report.json")
    parser.add_argument("--report-dir", default="reports/m0")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--synthetic", action="store_true", help="Do not treat synthetic results as a real-data M0 PASS")
    args = parser.parse_args()
    metadata_dir, report_dir, project_root = Path(args.metadata_dir), Path(args.report_dir), Path(args.project_root).resolve()
    checks, summary = validate_metadata(metadata_dir, project_root)
    download = read_json(args.download_report)
    if summary["frames"]:
        checks.extend(validate_splits(Path(args.split), summary["frames"]))
        checks.extend(validate_windows(Path(args.windows), Path(args.split), summary["frames"]))
        checks.extend(validate_evidence(metadata_dir, report_dir, download, synthetic=args.synthetic))
    test_checks, cases = validate_tests(report_dir)
    checks.extend(test_checks)
    checks.extend(validate_figures(report_dir))
    geometry_stats = read_json(report_dir / "geometry_time_and_label_statistics.json")
    identity = geometry_stats.get("identity", {})
    checks.append(check("Acquisition identity independently confirms directory plant UID", identity.get("source_acquisition_ids") == len(summary["plants"]) and not identity.get("source_ids_shared_across_plant_uids", {"missing": True}) and not identity.get("plants_with_multiple_source_ids", ["missing"]) and identity.get("filename_genotype_conflict_frames") == 0, identity))
    smoke_checks, smoke = dataset_smoke(metadata_dir, Path(args.split), Path(args.windows), project_root)
    checks.extend(smoke_checks)
    summary.update({"metadata_dir": metadata_dir, "project_root": str(project_root), "split_payload": read_json(args.split), "windows": read_csv(args.windows) if Path(args.windows).exists() else [], "tests": cases, "smoke_results": smoke})
    overall = render_reports(checks, summary, download, report_dir, synthetic=args.synthetic)
    print(json.dumps({"status": overall, "checks": [{"name": item["name"], "status": item["status"]} for item in checks]}, ensure_ascii=False, indent=2))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
