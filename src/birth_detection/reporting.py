from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from birth_detection.common import digest, paired_interval, protocol, write_json
from birth_detection.dataset import frame_path
from birth_detection.analysis import analyze


def import_baseline(root):
    destination = root / "outputs/m7/baseline_import"
    destination.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(root / "reports/m6_b/tracking_per_plant.csv").query("pipeline == 'B3'").copy()
    rows, observations, tracks, births, links = [], [], [], [], []
    for result in original.itertuples(index=False):
        plant = result.plant_uid
        source = root / f"outputs/m6_b/S2/seed_17/tracking/{plant}.json"
        data = json.loads(source.read_text())
        by_key = {(row["frame_index"], row["gt_id"]): row for row in data["observations"]}
        organ_key = {(row["frame_index"], row["local_id"]): row for row in data["organs"]}
        correct_links = []
        for old in data["observations"]:
            measure = organ_key.get((old["frame_index"], old["local_id"]), {})
            current = {**old, "plant_id": plant, "method": "Current", "identity_correct": old["evaluation_identity_correct"],
                       "length_cm": measure.get("length_cm", np.nan), "area_cm2": measure.get("area_cm2", np.nan)}
            observations.append(current)
            previous = by_key.get((old["frame_index"] - 1, old["gt_id"]))
            if previous is not None:
                detected = old["local_id"] > 0 and previous["local_id"] > 0
                correct = detected and old["track_id"] >= 0 and old["track_id"] == previous["track_id"]
                correct_links.append(correct)
                previous_measure = organ_key.get((previous["frame_index"], previous["local_id"]), {})
                links.append({"plant_id": plant, "split": result.split, "method": "Current", "seed": 17,
                              "gt_id": old["gt_id"], "DAS": old["DAS"], "previous_DAS": previous["DAS"],
                              "detected_both": detected, "correct_link": correct,
                              "error_type": "detection_failure" if not detected else "identity_failure" if not correct else "correct",
                              "length_cm": measure.get("length_cm", np.nan), "area_cm2": measure.get("area_cm2", np.nan),
                              "previous_length_cm": previous_measure.get("length_cm", np.nan), "previous_area_cm2": previous_measure.get("area_cm2", np.nan)})
        rows.append({**data["result"], "plant_id": plant, "method": "Current", "seed": 17,
                     "correct_link_coverage": float(np.mean(correct_links)), "source": "imported unchanged M6B result, not rerun"})
        tracks.extend([{**row, "plant_id": plant, "method": "Current"} for row in data["organs"]])
        births.extend([{**row, "plant_id": plant, "method": "Current", "split": result.split, "seed": 17} for row in data["births"]])
    for name, values in [("plants", rows), ("observations", observations), ("tracks", tracks), ("births", births), ("links", links)]:
        pd.DataFrame(values).to_csv(destination / f"{name}.csv", index=False)
    write_json(root / "reports/m7/baseline_import.json", {"source": "reports/m6_b/tracking_per_plant.csv", "sha256": digest(root / "reports/m6_b/tracking_per_plant.csv"),
               "new_method_evaluation_of_old_segmenter": False, "old_tracking_reexecution": False,
               "M6B_F1_semantics_unchanged": True})
    return pd.DataFrame(rows)


def temporal_event_rows(root, split, chosen_method):
    events = pd.read_csv(root / "reports/m7/organ_birth_events.csv").query("split == @split and previous_observation_available")
    all_rows = []
    methods = ["Current"] + sorted(path.name for path in (root / "outputs/m7/tracking").iterdir() if path.is_dir())
    if chosen_method not in methods:
        methods.append(chosen_method)
    for method in methods:
        seeds = [17] if method == "Current" else [17, 29, 43]
        for seed in seeds:
            if method == "Current":
                directory = root / "outputs/m7/baseline_import"
                observations = pd.read_csv(directory / "observations.csv").query("split == @split")
                births = pd.read_csv(directory / "births.csv").query("split == @split")
            else:
                directory = root / f"outputs/m7/tracking/{method}/seed_{seed}/{split}"
                observations = pd.read_csv(directory / "observations.csv")
                births = pd.read_csv(directory / "births.csv")
            lookup = {(row.plant_id, int(row.DAS), int(row.gt_id)): row for row in observations.itertuples(index=False)}
            born_keys = {(row.plant_id, int(row.frame_index), int(row.local_id)) for row in births.itertuples(index=False)}
            for event in events.itertuples(index=False):
                row = lookup[(event.plant_id, int(event.birth_time_visible), int(event.leaf_instance_id) + 1)]
                detected = row.local_id > 0
                born = detected and (event.plant_id, int(row.frame_index), int(row.local_id)) in born_keys
                later = observations.query("plant_id == @event.plant_id and gt_id == @row.gt_id and DAS >= @row.DAS")
                matched = later.query("local_id > 0")
                delay = float(matched.DAS.min() - row.DAS) if len(matched) else np.nan
                all_rows.append({"method": method, "seed": seed, "split": split, "plant_id": event.plant_id,
                                 "leaf_instance_id": event.leaf_instance_id, "DAS": event.birth_time_visible,
                                 "detected_instance": detected, "correct_birth": born, "small": event.small,
                                 "core_adjacent": event.core_adjacent, "first_detection_delay_days": delay,
                                 "error_type": "detection_failure" if not detected else "identity_failure" if not born else "correct"})
    return pd.DataFrame(all_rows)


def growth_audit(root, method, seed, split, baseline=False):
    if baseline:
        directory = root / "outputs/m7/baseline_import"
    else:
        directory = root / f"outputs/m7/tracking/{method}/seed_{seed}/{split}"
    observations = pd.read_csv(directory / "observations.csv").query("split == @split")
    links = pd.read_csv(directory / "links.csv").query("split == @split")
    leaf = pd.read_csv(root / "data/processed/sugar4d/leaf_metadata.csv")
    reference = {(row["plant_uid"], int(row["DAS"]), int(row["leaf_instance_id"]) + 1):
                 (float(row["total length [cm]"]), float(row["leaf area [cm2]"])) for unused_index, row in leaf.iterrows()}
    rows, failures = [], []
    for plant, group in observations.groupby("plant_id"):
        for trait_index, trait in enumerate(["length", "area"]):
            estimate_name = trait + ("_cm" if trait == "length" else "_cm2")
            errors, squared, relative = [], [], []
            eligible = len(group)
            correct_observations = int(group.identity_correct.sum())
            available = 0
            for row in group.itertuples(index=False):
                estimate = getattr(row, estimate_name)
                actual = reference[(plant, int(row.DAS), int(row.gt_id))][trait_index]
                found = bool(np.isfinite(estimate) and row.identity_correct)
                available += int(found)
                if found:
                    error = abs(estimate - actual)
                    errors.append(error)
                    squared.append(error ** 2)
                    relative.append(error / max(actual, 1.))
                failures.append({"plant_id": plant, "split": split, "method": method, "seed": seed, "DAS": row.DAS,
                                 "gt_id_evaluation_only": row.gt_id, "trait": trait, "kind": "static",
                                 "reference": actual, "estimate": estimate, "identity_correct": row.identity_correct,
                                 "status": "detection_failure" if row.local_id < 0 else "identity_failure" if not row.identity_correct else "measurement_failure" if not np.isfinite(estimate) or abs(estimate - actual) / max(actual, 1.) > .5 else "correct"})
            rows.append({"plant_id": plant, "split": split, "method": method, "seed": seed, "trait": trait, "kind": "static",
                         "eligible": eligible, "available": available, "coverage": available / max(eligible, 1),
                         "identity_correct_fraction": correct_observations / max(eligible, 1),
                         "MAE": np.mean(errors) if errors else np.nan, "RMSE": np.sqrt(np.mean(squared)) if squared else np.nan,
                         "relative_error": np.mean(relative) if relative else np.nan,
                         "conditional_errors": True, "missing_not_dropped_from_coverage": True})
            plant_links = links.query("plant_id == @plant")
            errors, squared, relative = [], [], []
            for row in plant_links.itertuples(index=False):
                current = getattr(row, estimate_name)
                previous = getattr(row, "previous_" + estimate_name)
                actual = reference[(plant, int(row.DAS), int(row.gt_id))][trait_index] - reference[(plant, int(row.previous_DAS), int(row.gt_id))][trait_index]
                gap = row.DAS - row.previous_DAS
                found = row.correct_link and np.isfinite(current) and np.isfinite(previous)
                estimate = (current - previous) / gap if found else np.nan
                if found:
                    error = abs(estimate - actual / gap)
                    errors.append(error)
                    squared.append(error ** 2)
                    relative.append(abs(current - previous - actual) / max(reference[(plant, int(row.previous_DAS), int(row.gt_id))][trait_index], 1.))
                failures.append({"plant_id": plant, "split": split, "method": method, "seed": seed, "DAS": row.DAS,
                                 "gt_id_evaluation_only": row.gt_id, "trait": trait, "kind": "growth_rate",
                                 "reference": actual / gap, "estimate": estimate, "identity_correct": row.correct_link,
                                 "status": row.error_type if row.error_type != "correct" else "measurement_failure" if not found else "correct"})
            rows.append({"plant_id": plant, "split": split, "method": method, "seed": seed, "trait": trait, "kind": "growth_rate",
                         "eligible": len(plant_links), "available": len(errors), "coverage": len(errors) / max(len(plant_links), 1),
                         "identity_correct_fraction": plant_links.correct_link.mean(), "MAE": np.mean(errors) if errors else np.nan,
                         "RMSE": np.sqrt(np.mean(squared)) if squared else np.nan, "relative_error": np.mean(relative) if relative else np.nan,
                         "conditional_errors": True, "missing_not_dropped_from_coverage": True})
    return rows, failures


def oracle_growth_audit(root, split):
    frames = pd.read_csv(root / "reports/m7/frame_audit.csv").query("split == @split")
    reference = pd.read_csv(root / "data/processed/sugar4d/leaf_metadata.csv")
    reference_lookup = {(row["plant_uid"], int(row["DAS"]), int(row["leaf_instance_id"]) + 1):
                        [row["total length [cm]"], row["leaf area [cm2]"]] for unused_index, row in reference.iterrows()}
    rows, failures = [], []
    for plant, group in frames.groupby("plant_id"):
        observations, links = [], []
        previous = {}
        for row in group.itertuples(index=False):
            with np.load(frame_path(root, plant, row.DAS)) as frame:
                current = {int(value[0]): value[1:] for value in frame["gt_measures"]}
            for gt_id, measure in current.items():
                observations.append((row.DAS, gt_id, measure))
                if gt_id in previous:
                    old_day, old = previous[gt_id]
                    links.append((old_day, row.DAS, gt_id, old, measure))
            previous = {gt_id: (row.DAS, measure) for gt_id, measure in current.items()}
        for trait_index, trait in enumerate(["length", "area"]):
            for kind, values in [("static", observations), ("growth_rate", links)]:
                errors, squared, relative = [], [], []
                for value in values:
                    if kind == "static":
                        day, gt_id, measure = value
                        actual = reference_lookup[(plant, int(day), gt_id)][trait_index]
                        estimate = measure[trait_index]
                        scale = max(actual, 1.)
                    else:
                        old_day, day, gt_id, old, measure = value
                        actual = (reference_lookup[(plant, int(day), gt_id)][trait_index] - reference_lookup[(plant, int(old_day), gt_id)][trait_index]) / (day - old_day)
                        estimate = (measure[trait_index] - old[trait_index]) / (day - old_day)
                        scale = max(reference_lookup[(plant, int(old_day), gt_id)][trait_index], 1.) / (day - old_day)
                    if np.isfinite(estimate):
                        errors.append(abs(estimate - actual))
                        squared.append((estimate - actual) ** 2)
                        relative.append(abs(estimate - actual) / scale)
                    failures.append({"plant_id": plant, "split": split, "method": "Oracle_GT_masks_GT_identity", "seed": 0,
                                     "DAS": day, "gt_id_evaluation_only": gt_id, "trait": trait, "kind": kind,
                                     "reference": actual, "estimate": estimate, "identity_correct": True,
                                     "status": "measurement_failure" if not np.isfinite(estimate) or abs(estimate - actual) / scale > .5 else "correct"})
                rows.append({"plant_id": plant, "split": split, "method": "Oracle_GT_masks_GT_identity", "seed": 0, "trait": trait, "kind": kind,
                             "eligible": len(values), "available": len(errors), "coverage": len(errors) / max(len(values), 1),
                             "identity_correct_fraction": 1., "MAE": np.mean(errors), "RMSE": np.sqrt(np.mean(squared)),
                             "relative_error": np.mean(relative), "conditional_errors": True, "missing_not_dropped_from_coverage": True})
    return rows, failures


def recommendation(tracking, event_rows, baseline, chosen):
    current = baseline.query("split == 'val'").set_index("plant_id")
    proposed = tracking.query("split == 'val' and method == @chosen").groupby("plant_id").mean(numeric_only=True)
    paired = proposed.idf1 - current.idf1
    id_interval = paired_interval(paired)
    event_interval = paired_interval(proposed.new_event_f1 - current.new_event_f1)
    current_small = event_rows.query("split == 'val' and method == 'Current' and small").groupby("plant_id").correct_birth.mean()
    new_small = event_rows.query("split == 'val' and method == @chosen and small").groupby(["plant_id", "seed"]).correct_birth.mean().groupby("plant_id").mean()
    small_delta = new_small - current_small
    small_interval = paired_interval(small_delta)
    coverage_delta = proposed.correct_link_coverage - current.correct_link_coverage
    coverage_interval = paired_interval(coverage_delta)
    positive = int((proposed.new_event_f1 > current.new_event_f1).sum())
    birth_pass = proposed.new_event_f1.mean() > .5 and small_interval["lower"] is not None and small_interval["lower"] > 0 and paired.mean() >= 0 and positive >= 9
    track_pass = id_interval["lower"] > 0 and paired.mean() >= .02 and coverage_delta.mean() >= .02
    outcome = "BIRTH-SUCCESS" if birth_pass else "TRACKING-SUCCESS" if track_pass else "DATA-LIMITED"
    next_stage = "DYNAMIC-PHENOTYPING" if birth_pass and track_pass else "BIRTH-MODELING" if birth_pass else "ORGAN-TRACKING" if track_pass else "STOP"
    return {"outcome": outcome, "next_stage": next_stage, "chosen": chosen, "validation_mean_birth_F1": float(proposed.new_event_f1.mean()),
            "validation_mean_IDF1": float(proposed.idf1.mean()), "IDF1_paired": id_interval, "birth_F1_paired": event_interval,
            "small_birth_recall_paired": small_interval, "correct_link_coverage_paired": coverage_interval,
            "birth_positive_plants": positive, "birth_gates_pass": bool(birth_pass), "tracking_gates_pass": bool(track_pass),
            "absolute_measurements_certified": False}


def generate(root):
    settings = protocol(root)
    baseline = import_baseline(root)
    selected = json.loads((root / "reports/m7/system_selection.json").read_text())
    chosen = selected["chosen_method"]
    tracking = pd.concat([pd.read_csv(root / f"reports/m7/tracking_metrics_{split}.csv") for split in ["val", "test"]], ignore_index=True)
    combined_tracking = pd.concat([baseline, tracking], ignore_index=True)
    combined_tracking.to_csv(root / "reports/m7/tracking_metrics.csv", index=False)
    event_rows = pd.concat([temporal_event_rows(root, split, chosen) for split in ["val", "test"]], ignore_index=True)
    event_rows.to_csv(root / "reports/m7/event_level_results.csv", index=False)
    event_rows.query("error_type != 'correct'").to_csv(root / "reports/m7/failure_cases.csv", index=False)
    event_rows.groupby(["method", "split", "error_type"]).size().reset_index(name="events_including_three_seeds").to_csv(root / "reports/m7/error_decomposition.csv", index=False)
    birth_tables = []
    for path in (root / "outputs/m7/birth_evaluation").glob("*/seed_*/*/plants.csv"):
        birth_tables.append(pd.read_csv(path))
    birth_metrics = pd.concat(birth_tables, ignore_index=True)
    birth_metrics.to_csv(root / "reports/m7/birth_metrics.csv", index=False)
    decision = recommendation(tracking, event_rows, baseline, chosen)
    write_json(root / "reports/m7/decision.json", decision)
    growth_rows, growth_failures = [], []
    for split in ["val", "test"]:
        rows, failures = oracle_growth_audit(root, split)
        growth_rows.extend(rows)
        growth_failures.extend(failures)
        for method in ["Current", chosen]:
            for seed in ([17] if method == "Current" else settings["seeds"]):
                rows, failures = growth_audit(root, method, seed, split, method == "Current")
                growth_rows.extend(rows)
                growth_failures.extend(failures)
    growth = pd.DataFrame(growth_rows)
    growth["M7_D_full_recovery_gate_pass"] = decision["birth_gates_pass"] or decision["tracking_gates_pass"]
    growth.to_csv(root / "reports/m7/growth_metrics.csv", index=False)
    pd.DataFrame(growth_failures).to_csv(root / "reports/m7/measurement_failure_cases.csv", index=False)
    intervals = []
    for (method, split), group in combined_tracking.groupby(["method", "split"]):
        plants = group.groupby("plant_id").mean(numeric_only=True)
        for metric in ["idf1", "new_event_f1", "correct_link_coverage", "continuity", "id_switches"]:
            intervals.append({"method": method, "split": split, "metric": metric, **paired_interval(plants[metric])})
    pd.DataFrame(intervals).to_csv(root / "reports/m7/plant_intervals.csv", index=False)
    summary = combined_tracking.groupby(["method", "split"])[["idf1", "new_event_f1", "correct_link_coverage", "continuity", "id_switches"]].mean().reset_index()
    summary.to_csv(root / "reports/m7/tracking_summary.csv", index=False)
    static_summary = birth_metrics.groupby(["method", "split"])[["birth_f1", "birth_recall", "small_recall"]].mean().reset_index()
    static_summary.to_csv(root / "reports/m7/birth_summary.csv", index=False)
    growth_summary = growth.groupby(["method", "split", "trait", "kind"])[["coverage", "MAE", "RMSE", "relative_error"]].mean().reset_index()
    growth_summary.to_csv(root / "reports/m7/growth_summary.csv", index=False)
    analyze(root)
    write_report(root, decision, summary, static_summary, growth_summary)


def table_markdown(frame, digits=4):
    names = list(frame.columns)
    result = ["| " + " | ".join(names) + " |", "| " + " | ".join(["---"] * len(names)) + " |"]
    for unused_index, row in frame.iterrows():
        result.append("| " + " | ".join(f"{value:.{digits}f}" if isinstance(value, (float, np.floating)) else str(value) for value in row) + " |")
    return "\n".join(result)


def write_report(root, decision, tracking, birth, growth):
    manifest = json.loads((root / "reports/m7/dataset_manifest.json").read_text())
    events = pd.read_csv(root / "reports/m7/organ_birth_events.csv")
    split_events = events.query("previous_observation_available").groupby("split").size().to_dict()
    chosen = decision["chosen"]
    failures = pd.read_csv(root / "reports/m7/event_level_results.csv").query("split == 'val' and method == @chosen")
    failure_counts = failures.groupby("error_type").size().to_dict()
    selected = json.loads((root / "reports/m7/birth_selection.json").read_text())
    contrasts = pd.read_csv(root / "reports/m7/temporal_comparisons.csv")
    contrasts = contrasts.query("split == 'val' and metric == 'birth_f1'")
    cohort = pd.read_csv(root / "reports/m7/birth_cohort_statistics.csv").query("stratum == 'split' and subset == 'new_after_initial_scan'")
    ceiling = pd.read_csv(root / "reports/m7/candidate_recall_ceiling.csv").query("split == 'val'")
    ceiling_count = int(ceiling.new_candidates_recoverable.sum())
    event_contribution = pd.read_csv(root / "reports/m7/event_failure_contributions.csv").query("split == 'val'")
    current_events = event_contribution.query("method == 'Current'").set_index("error_type")["count"].to_dict()
    chosen_events = event_contribution.query("method == @chosen").groupby("error_type")["count"].mean().to_dict()
    measure_contribution = pd.read_csv(root / "reports/m7/measurement_failure_contributions.csv").query("split == 'val' and kind == 'static' and trait == 'length' and method == @chosen")
    measurement_errors = measure_contribution.groupby("status").fraction.mean()
    failing = measurement_errors.drop("correct", errors="ignore")
    largest_measurement = failing.idxmax() if len(failing) else "none"
    budgets = pd.read_csv(root / "reports/m7/training_budgets.csv")
    forecast = pd.read_csv(root / "reports/m7/forecast_summary.csv")
    seed_tracking = pd.read_csv(root / "reports/m7/tracking_metrics_val.csv").groupby(["method", "seed"])[["idf1", "new_event_f1", "correct_link_coverage"]].mean().reset_index()
    gaps = pd.read_csv(root / "reports/m7/label_stability_audit.csv")
    geometric_oracle = growth.query("split == 'val' and method == 'Oracle_GT_masks_GT_identity' and kind == 'static'")
    positive_temporal = int((contrasts.lower > 0).sum())
    time_answer = "至少一组匹配对照的探索性植株CI完全为正，但仍须外部验证" if positive_temporal else "没有建立稳定的时间编码优势：两组匹配对照均未得到完全为正的探索性植株CI"
    source = "https://www.ipb.uni-bonn.de/data/sugar4d/index.html"
    report = f"""# M7：Birth-aware Plant Organ Dynamics from Sparse Temporal Point Clouds

## 1. 唯一结论与下一阶段

**{decision['outcome']}**；唯一下一阶段建议 **{decision['next_stage']}**。
Validation锁定系统 `{decision['chosen']}`：新叶首次观测F1={decision['validation_mean_birth_F1']:.4f}，IDF1={decision['validation_mean_IDF1']:.4f}。
出生门槛通过={decision['birth_gates_pass']}；追踪门槛通过={decision['tracking_gates_pass']}。单独IDF1上升不等于TRACKING-SUCCESS，还要求正确相邻link覆盖增益至少0.02；单独候选分类F1上升也不等于自动系统的BIRTH-SUCCESS。
本结论只覆盖固定候选和轻量模型的实际证据，不是证明所有动态方法无效。未自动增加分割主干、大模型或JEPA。

## 2. M7-A：出生 benchmark

保留官方24/12/12整株划分，48株各16帧，共{manifest['frames']}帧；合法叶和整株不删除。
全部叶首次出现{manifest['events']}个，其中可比较的非左删失新叶{manifest['non_left_censored_events']}个，分集合为{split_events}。
`organ_birth_events.csv`包含全部要求字段，另记录删失、事件区间和代理出处。

- **Visible birth**：释放的PLY首次出现叶语义且非负实例，不是生物出生；真实可见时刻位于前次扫描与当前扫描之间，初始扫描左删失。
- **Segmentable birth**：固定S2 strict IoU>0.5首匹配，是依赖模型的操作代理，不能称天然可分割。一直未匹配为右删失；不是设为93DAS。
- **Trackable birth**：固定M6-B追踪中连续三个扫描匹配同一track的首帧，需要事后确认，仅benchmark标注使用，不输入模型。
- Train旧追踪未运行，trackable置缺失且 `trackability_evaluable=False`，不为了补字段重跑旧方法。Train S2预测是冻结模型训练内推理，非out-of-fold，限制明确。
- 初始面积是官方点云提取元数据cm²，**不是人工面积真值**，仅评价；小叶用M6-B Train阈值点比例<=.04588683197817431。core距离为GT leaf到GT core的p10米距离，<=5mm是邻近代理，不是人工证明未分离。

{table_markdown(cohort[['split', 'plants', 'events', 'small_events', 'core_adjacent_events', 'median_initial_points', 'median_initial_area_cm2', 'segmentable_right_censored']])}

小叶是**相对点数比例**，不是生物学幼叶或绝对点数很少；需结合面积和物理距离解释。GT同一ID在首次/末次之间有缺扫描的叶共{int(gaps.reappears_after_missing_scan.sum())}/{len(gaps)}个；缺扫描可能是遮挡或标注变化，不能无人工复核宣称标签错误。

## 3. M7-B：时间信息与发现

Baseline0完全读取已有M6-B S2 seed17分割和TrackPlant3D结果，不重评旧方法或修改历史指标。
Baseline1在米坐标用4mm voxel占据/局部密度差分、12mm DBSCAN，保留所有候选。已有叶增长、扫描漂移也可能造成差分；候选上限单列为评价Oracle，不是自动方法。
Baseline2历史几何RandomForest；Baseline3共享MLP PointNet+64维GRU，均三随机种子。加入单帧几何RF/PointNet匹配对照，候选集相同；单帧控制仍来自相同因果候选，因此结论仅是历史特征/编码增益，不能称纯单帧候选系统。
所有输入仅XYZ、预测语义/实例、<=t的物理几何、实际DAS间隔。全点候选评估 strict IoU>.5，不把GT叶中心/ID作为候选。

{table_markdown(birth.query("split == 'val'"))}

RF树数200深度12；PointNet/GRU40轮，Adam .001，batch128，train-only标准化。相同候选训练标签，三种子共同Validation选阈值；最终轮checkpoint，不以Test选结构或epoch。
另单独训练next-scan plant-level forecast_RF三种子，目标下一次扫描是否有新可见叶，只用截至t历史；见`forecast_metrics_val.csv`。这不是当前器官定位或生物出生预测，不能与定位F1混为一谈。
锁定新叶模型为 `{selected['chosen_method']}`；所有阈值和单种子结果完整交付，不只报最好植株/种子。

匹配对照按三种子先植株平均的差值如下（selected thresholds，CI/p是选择后探索性证据）：

{table_markdown(contrasts[['comparison', 'mean', 'lower', 'upper', 'plants', 'positive_plants', 'holm_p_for_two_validation_F1_comparisons']])}

Validation候选池GT选取的严格IoU召回乐观上限为{ceiling_count}/107事件，植株macro={ceiling.candidate_recall_ceiling.mean():.4f}。`candidate_oracle`使用GT选择正确候选、忽略自动辨别困难，不属于自动方法、不参与选择。若候选本身无法分离一片新叶，历史分类器只能改变候选概率，不能凭空恢复该叶mask。

## 4. M7-C：身份保持及集成

C0是原TrackPlant3D结果导入。C1监督MLP：物理质心位移、形状特征、面积/长度变化和方向；C2增加32维PointNet器官embedding及同叶/异叶监督contrastive距离。不是JEPA。
GT时间身份仅用于Train配对标签和evaluator；推理不使用GT。实例进入系统独立随机局部编号，Hungarian一对一匹配，最多两扫描gap，C1/C2相同生命周期。
C0保持官方采样、CPD及几何cost，不被替换为本轮学习matching。C1/C2与C0同时存在匹配评分和生命周期实现差异，因此系统增益不是纯粹的embedding因果消融；C2相对C1才共享同一生命周期。结果只验证身份打分/管理，不证明学习了可泛化的植物生理规律。
birth-aware集成将选中新区域/实例显式初始化新track；差分区域重叠会从旧mask中划分，不能假装它天然是GT叶，完整记录错误。没有优化分割主干或PQ。

{table_markdown(tracking.query("split == 'val'"))}

三种子平均后按植株配对：IDF1增益CI={decision['IDF1_paired']}；事件F1增益CI={decision['birth_F1_paired']}；小叶首观测召回增益CI={decision['small_birth_recall_paired']}；正确link覆盖增益CI={decision['correct_link_coverage_paired']}。
出生F1改善植株{decision['birth_positive_plants']}/12。bootstrap10000，95%CI，单位是12株而非192扫描或3种子×12独立株。
小叶首观测成功率的提高不等于小叶mask检测召回提高：C1/C2保持S2分割，改善可能仅来自正确初始化身份。IDF1平均提高也不能抵消正确相邻link覆盖下降；须同时查看上述配对区间。

三种子分别的全12株结果：

{table_markdown(seed_tracking)}

整株下一扫描forecast与当前扫描叶mask出生不是同一目标。另有三种子RF及不使用GT作为输入的always/never-new频率控制，禁止用其F1代替新叶定位结果：

{table_markdown(forecast)}

## 5. M7-D：测量与失败分解

完整恢复研究启动条件={decision['birth_gates_pass'] or decision['tracking_gates_pass']}。若未稳定，仅描述性覆盖/错误审计，不宣称时序表型恢复成功。
Oracle为GT mask+GT identity，不同于GT mask+自动tracker；Current为旧S2+固定TrackPlant3D；Proposed为Validation锁定自动系统。
所有方法复用 `m6a.metrics.geometry_measure`；面积平面凸包、长度PCA范围的绝对偏差不改动。MAE/RMSE条件于测量且身份正确，必须同时读丢失/错接覆盖率；缺失不当零误差。
参照`leaf_related.csv`为官方自动点云形态提取，不是每时刻人工真值；人工CSV只有93DAS长度/宽度450条、无人工面积或完整growth轨迹。以下误差相对官方提取参照，不称独立实测精度，见`DATA_INTERFACE_AUDIT.md`。

{table_markdown(growth.query("split == 'val'"))}

Validation选定系统新叶失败/正确共三种子计数：{failure_counts}。三种子不是新增生物样本。
`event_level_results.csv`、`failure_cases.csv`区分首帧检测失败、已检测但错身份；`measurement_failure_cases.csv`单列在正确身份上的测量器错误。不能用平均误差掩盖新叶漏检或错轨。

## 6. 四个最终问题

1. **新叶发现是否主要瓶颈**：是出生任务的主要瓶颈。固定Current在107事件的计数={current_events}；选定系统每种子平均={chosen_events}。候选乐观上限仅{ceiling_count}/107，限制集中在出生帧的可独立辨别几何，而非整株静态PQ。这个判断不意味着全部动态测量误差也由漏检主导。
2. **时间是否优于单帧**：{time_answer}。共享候选history_RF与single_RF、PointNet_GRU与single_PointNet是匹配特征对照；共同候选本身使用时间变化，所以不是纯单帧端到端系统消融。集成系统收益不能单独证明时间编码收益；下一扫描整株forecast F1也不是叶mask出生F1。
3. **最大错误贡献**：首观测事件中Current的detection failure={current_events.get('detection_failure', 0)}、identity failure={current_events.get('identity_failure', 0)}，因此漏检最大；选定系统见上方平均。在全部静态长度观测的分母上，选定系统最大失败项为`{largest_measurement}`，各项比例={measurement_errors.to_dict()}。这是不同分母、不同问题，不强行合成单一百分比。Oracle GT mask+GT identity仍有测量偏差：

{table_markdown(geometric_oracle[['trait', 'coverage', 'MAE', 'RMSE', 'relative_error']])}

测量错误分解使用相对误差>50%的描述性门限，静态分母floor=1cm/1cm²；不是M6-B原可靠性判据，也不用于方法选择。身份正确条件误差不能证明整套测量器可靠。
4. **下一阶段**：只建议 **{decision['next_stage']}**，不自动设计大模型或恢复JEPA。

## 7. Test、复现及保护

Test早已在历史阶段使用，只作固定方案探索性复核，不是独立盲测。Validation选择锁定后才进行Test概率推理、track及评价。三种子Test全部保留，不反向调整阈值。

{table_markdown(tracking.query("split == 'test'"))}

新增工程与训练日志/checkpoint/全部候选概率/事件结果/track/失败保存在`outputs/m7`和`reports/m7`。
`tracking_paired_plants.csv`保留每株三种子均值对Current的配对差；`track_level_results.csv`与`gt_track_level_results.csv`分别保留预测轨迹及参照器官的漏失、混合身份与连续性，不删除无匹配轨迹。
复现：`scripts/build_birth_dataset.py` → `scripts/train_m7_matrix.sh` → `scripts/report_m7.py`；详见`REPRODUCE.md`。
旧成果保护以任务开始HEAD及tracked SHA256、旧outputs/reports大小mtime审计；不运行旧report写回、不改历史环境。未提交/合并/推送。本阶段交付后停止。

共{len(budgets)}次有效正式拟合：7种方法×3种子，其中12个NN每个40轮，RF每个200树；最终轮而非Test最优checkpoint。模型参数、样本、正样本、梯度步和运行时间见`training_budgets.csv`。另有被作废的工程批次，全部保留在`invalid_source_leak_runs`，**不参与结果或选择**：GRU最初时间精度错误，之后发现早期构建缓存的单帧控制含source/novel两列。内容审计纠正为54维，不含两列，重新按原预算从头训练全部出生/匹配模型，无改门限、读Test调参或删样本；`GRU_dtype_failure.log`和作废缓存证据保留。
图见`figures/`（真实统计、配对证据、失败分解和点云实例案例；SVG/PDF/PNG），案例按失败类型中位初始点数选择，不是最优植株。

## 8. 来源与限制

一般博查检索实际HTTP403配额不足，不能声称成功搜索。事实来源是实际取得的官方网页/Dataverse元数据、Context7官方文档和本地实验产物，不凭训练记忆编数。
Sugar4D官方：{source} ; https://doi.org/10.60507/FK2/IS8YBZ 。PyTorch2.8和scikit-learn文档实际拉取，见`sources/ctx7_*`。
只有单一来源48株，时间间隔3–7天，visible是释放标注代理；没有生物出生真值、人工审定segmentability或外部植物验证。任何DATA-LIMITED只能说明该限定benchmark/模型未建立稳定优势，不能凭失败宣称事件数必然太少。
Train非左删失事件238个，不能随意说“完全无训练事件”；难度是24个独立训练植株、稀疏事件区间、出生候选辨识上限、in-sample S2 proposals以及未人工审定的可分离性。缺外部盲测与人工微小叶标注，停止本轮调参不等于证明一切出生模型不可行。
来源检索失败只限一般搜索；上述科研数值来自真实本地执行而不是训练数据记忆。完整来源、锁定哈希、依赖环境、delivery manifest与保护检查单独交付。
"""
    (root / "reports/m7/M7_REPORT.md").write_text(report)
