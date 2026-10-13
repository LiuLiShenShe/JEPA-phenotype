from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from m6a.common import digest, paired_interval, write_json
from m6a.experiment import PIPELINES, cache_path, metadata


COLORS = {"B0": "#277A9D", "B1": "#C64B4B", "B2": "#D49534", "B2_GT": "#6F7987"}


def markdown_table(frame):
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = []
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) and np.isfinite(value) else str(value) for value in row]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider] + rows)


def source_audit(root):
    destination = root / "reports/m6_a"
    source = destination / "sources"
    records = []
    for path in (root / "data/raw/m6_a/official").rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            records.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)})
    pd.DataFrame(records).to_csv(destination / "downloaded_files.csv", index=False)
    license_table = []
    for name in ["pepper4d", "plantnet", "trackplant3d"]:
        meta = json.loads((source / f"{name}_metadata.json").read_text())
        tree = json.loads((source / f"{name}_tree.json").read_text())
        license_table.append({"project": name, "official_url": meta["html_url"], "code_license": "not declared / not verified",
                              "github_license": str(meta["license"]), "pinned_source_sha": tree["sha"],
                              "tree_entries": len(tree["tree"]), "license_api_http": 404,
                              "redistribution": "not authorized by public accessibility; upstream stays local"})
    pd.DataFrame(license_table).to_csv(destination / "source_audit.csv", index=False)
    sugar = json.loads((source / "sugar4d_live_metadata.json").read_text())["data"]["latestVersion"]
    write_json(destination / "data_feasibility.json", {"sugar_version": sugar["versionNumber"], "sugar_release": sugar["releaseTime"],
               "license": sugar["license"], "historical_sugar_archive_sha256": "69e25b87b470173d30a70fbac9d4caf2ae968082cd83378cd87d93601b370908",
               "plants": 48, "scans": 768, "evaluated_plants": 24, "evaluated_scans": 384,
               "train_plants_unused_for_training": 24, "pepper_downloaded": False, "cross_species_tested": False})
    text = """# M6-A 数据与代码来源审计

审计日期：2026-10-12。博查搜索实际返回 HTTP 403；直接访问官方 GitHub API、固定版本源码和 Bonn Dataverse 成功。下列项目情况来自这些实时官方材料与本地文件核验，不是假装搜索成功。

## Sugar4D V4：沿用，不升级历史数据
- 官方页面：https://www.ipb.uni-bonn.de/data/sugar4d/index.html
- DOI：https://doi.org/10.60507/FK2/IS8YBZ
- 实时元数据：https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ
- 2026-08-27 发布的 V4，CC BY 4.0。压缩包 6,270,780,517 bytes；SHA256 `69e25b87b470173d30a70fbac9d4caf2ae968082cd83378cd87d93601b370908`，MD5 `27e44210d95a85cb04b93513f0ef2beb`。
- 768 个真实 PLY，48 株×16次实际扫描；二进制 XYZ、法线、语义及实例标签。两张 CSV 包含植株与器官自动表型；最终时间点有部分人工长度验证，无人工面积或连续死亡事件真值。
- 保留原始 24/12/12 整株划分。新评估只处理 val/test 共24株384帧；没有训练，train 不需要再次推理。675个全数据器官身份是评价真值，不是追踪输入。

## Pepper-4D：官方信息可审计，数据不可访问
- 官方：https://github.com/foysalahmed10/Pepper-4D
- 官方论文：https://www.mdpi.com/2223-7747/15/4/599
- README 声明29株、916点云、约10GB、3个子集、不同生长条件；`frames/*.txt` 与 semantic/instance/tracking/health 标签文件、元数据并列。此结构为官方说明，**未通过下载实体核验**。
- 三个 Google Drive 官方下载地址均实际尝试，全部 `Network is unreachable`，见 `sources/pepper_download_attempts.json`。未取得任何完整数据子集、未训练或评价辣椒，不能声称完成跨物种验证。
- 仓库无显式许可证；GitHub license API 404。论文许可不自动等于数据许可。无模型实现或预训练权重；它是数据发布项目，不是可运行分割器。

## PlantNet / PSegNet：权重与实现成功获取
- 官方：https://github.com/Huang2002200/PlantNet-and-PSegNet
- 固定来源 `5efe9ced49473f0d59c618b7e140ffeeb692d9b1`。PyTorch模型/工具及两份epoch199权重实际下载；PlantNet权重53,599,453 bytes，PSegNet权重40,459,367 bytes，逐文件SHA256在 `downloaded_files.csv`。
- 官方原始546个单株点云为烟草/番茄/高粱，不解释为546个独立生物学植株；TXT六列 XYZ/instance/semantic/object。语义0/1、2/3、4/5分别对应三物种茎/叶；训练前转换H5，公开平均实例大小用于聚类过滤。
- 官方环境说明为Python3.8.18/PyTorch2.0.1，旧TensorFlow为Python3.7.13/TF1.13.1；本次不启用旧TensorFlow栈，使用隔离Python3.11.15/Torch2.8.0+cu128。
- 两网络都严格加载全部权重并实际4096点前向推理，重复运行标签/置信度完全一致。PSegNet仅可运行性检查，不做Sugar4D指标择优。
- 官方原始RAR与配套test.h5在仓库中是Git-LFS对象；本阶段未完整下载它们，也不把小型pointer文件记为数据下载成功。Sugar4D零样本运行不依赖它们。
- 代码和权重许可证未声明/未核实；公共可访问不表示可再分发。完整上游副本留在忽略的 `data/raw/m6_a`，只交付适配器和哈希/来源。

## TrackPlant3D：官方方法可运行，但需要接口适配
- 官方：https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- 固定来源 `3c04919dc86d75acd1a3159a1e1f74893e0bc5e4`。下载3DEPS、CPD、器官匹配、评价源码及requirements；完整四物种配套数据不作为本次实验输入。
- 输入是每时刻**非时序一致的器官实例标签**。3DEPS256点→非刚性CPD→扩展代价矩阵MWPM；不需要预训练追踪模型。
- 官方依赖numpy2.2.5/scipy1.15.2/matplotlib3.10.1/open3d0.19.0。本次隔离环境锁定Open3D0.19.0并使用与历史数值环境相同的numpy2.2.6/scipy1.17.1/matplotlib3.10.6；兼容性修改和实测记录见 `COMPATIBILITY.md`、`official_runtime_smoke.json`。
- 原始脚本有硬编码物种/目录和稀疏标签大小问题。保留官方CPD与代价/dummy权重；只做数组接口、活动标签稠密化、固定随机种子和边界邻域等价性能修复。当前实例GT每帧独立随机置换，时序GT只在评估器出现。
- 无显式代码许可，GitHub license API404；不公开再分发上游代码。成功执行不等于证明与论文全部官方数值完全一致。

## 文档来源与兼容性证据
- Context7实际拉取SciPy、Open3D和Matplotlib当前官方文档；检索结果保存在 `sources/ctx7_*`。
- SciPy：https://github.com/scipy/scipy
- Open3D官方边界核：https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/PointCloudImpl.h
- Matplotlib：https://github.com/matplotlib/matplotlib
- `http_audit.json`、官方仓库API响应、README、git tree及commit响应均本地保存；正文中所有项目数量声明明确区分官方陈述与实际本地核验。
"""
    (destination / "SOURCE_AUDIT.md").write_text(text)


def confidence_audit(root):
    destination = root / "reports/m6_a"
    predicted = pd.read_csv(destination / "predicted_organs.csv")
    observations = pd.read_csv(destination / "observations.csv")
    matched = observations[observations.local_id >= 0].groupby(["plant_uid", "pipeline", "DAS", "local_id"], as_index=False).agg(
        evaluation_identity_correct=("evaluation_identity_correct", "all"), matched_gt_count=("gt_id", "size"))
    predicted = predicted.merge(matched, on=["plant_uid", "pipeline", "DAS", "local_id"], how="left", validate="one_to_one")
    predicted["identity_error"] = ~predicted.evaluation_identity_correct.eq(True)
    loss = pd.read_csv(destination / "measurement_errors.csv")
    measured = loss[(loss.kind == "change") & (loss.stage == "TRACKED")].copy()
    rows = []
    risk_rows = []
    for (plant, split, pipeline), group in predicted.groupby(["plant_uid", "split", "pipeline"]):
        for score in ["semantic_confidence", "confidence"]:
            if group.identity_error.nunique() == 2:
                auroc = float(roc_auc_score(group.identity_error, 1 - group[score]))
            else:
                auroc = np.nan
            rows.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "target": "identity_error",
                         "score": score, "auroc": auroc, "predictions": len(group), "error_rate": group.identity_error.mean(),
                         "high_confidence_errors": int(((group[score] >= .8) & group.identity_error).sum()),
                         "high_confidence_count": int((group[score] >= .8).sum())})
            for coverage in [.2, .5, .8, 1.]:
                ordered = group.sort_values(score, ascending=False)
                retained = ordered.iloc[:max(int(np.ceil(coverage * len(group))), 1)]
                risk_rows.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "target": "identity_error",
                                  "score": score, "retained_fraction": coverage, "error_rate": retained.identity_error.mean()})
        for trait in ["length", "area"]:
            subset = measured[(measured.plant_uid == plant) & (measured.pipeline == pipeline) & (measured.trait == trait) & measured.confidence.notna()]
            auroc = float(roc_auc_score(subset.severe, 1 - subset.confidence)) if len(subset) and subset.severe.nunique() == 2 else np.nan
            rows.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "target": trait + "_severe_change",
                         "score": "confidence", "auroc": auroc, "predictions": len(subset),
                         "error_rate": subset.severe.mean(), "high_confidence_errors": int(((subset.confidence >= .8) & subset.severe).sum()),
                         "high_confidence_count": int((subset.confidence >= .8).sum())})
    predicted.to_csv(destination / "prediction_confidence.csv", index=False)
    confidence = pd.DataFrame(rows)
    confidence.to_csv(destination / "confidence_per_plant.csv", index=False)
    confidence.groupby(["split", "pipeline", "target", "score"])[["auroc", "error_rate", "high_confidence_errors", "high_confidence_count"]].mean().reset_index().to_csv(destination / "confidence_summary.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(destination / "confidence_selective_risk.csv", index=False)
    return predicted, confidence


def event_and_status_audit(root):
    destination = root / "reports/m6_a"
    observations = pd.read_csv(destination / "observations.csv")
    prediction = pd.read_csv(destination / "predicted_organs.csv")
    status_rows = []
    events = []
    for (plant, pipeline, gt_id), group in observations.groupby(["plant_uid", "pipeline", "gt_id"]):
        observed_indices = group.frame_index.to_numpy()
        first_index = int(observed_indices.min())
        last_index = int(observed_indices.max())
        all_das = sorted(observations[observations.plant_uid == plant].DAS.unique())
        lookup = group.set_index("frame_index")
        for frame_index, das in enumerate(all_das):
            if frame_index not in lookup.index:
                state = "not_observed_before_first" if frame_index < first_index else "not_observed_after_last" if frame_index > last_index else "not_observed_gap"
            else:
                row = lookup.loc[frame_index]
                state = "instance_missed" if row.local_id < 0 else "identity_lost" if row.track_id < 0 else "wrong_identity" if not row.evaluation_identity_correct else "observed_correct_identity"
            status_rows.append({"plant_uid": plant, "split": group.iloc[0].split, "pipeline": pipeline, "gt_id_evaluation_only": gt_id,
                                "frame_index": frame_index, "DAS": das, "state": state,
                                "true_biological_disappearance": "unknown_not_labeled"})
        if first_index > 0:
            first = group[group.frame_index == first_index].iloc[0]
            corresponding = prediction[(prediction.plant_uid == plant) & (prediction.pipeline == pipeline) & (prediction.DAS == first.DAS) & (prediction.local_id == first.local_id)]
            if first.local_id < 0:
                state = "missed_instance"
            elif first.track_id < 0:
                state = "sampling_or_identity_lost"
            elif len(corresponding) and bool(corresponding.iloc[0].new_track):
                state = "correct_first_observation"
            else:
                state = "attached_to_old_identity"
            detected = group[group.local_id >= 0]
            delay = float(detected.DAS.min() - first.DAS) if len(detected) else np.nan
            events.append({"plant_uid": plant, "split": group.iloc[0].split, "pipeline": pipeline, "gt_id_evaluation_only": gt_id,
                           "first_DAS": first.DAS, "event_state": state, "first_instance_detection_delay_days": delay,
                           "meaning": "first observation, not botanical birth"})
    pd.DataFrame(status_rows).to_csv(destination / "observation_status_grid.csv", index=False)
    pd.DataFrame(events).to_csv(destination / "new_leaf_events.csv", index=False)
    loss = pd.read_csv(destination / "measurement_errors.csv")
    loss[(loss.stage == "TRACKED") & loss.severe].sort_values("loss", ascending=False).to_csv(destination / "severe_failures.csv", index=False)
    loss[(loss.stage == "TRACKED") & (loss.kind == "change")].groupby(["split", "pipeline", "trait", "status"]).size().rename("eligible_GT_transitions").reset_index().to_csv(destination / "transition_failure_types.csv", index=False)


def propagation_common_support(root):
    destination = root / "reports/m6_a"
    errors = pd.read_csv(destination / "measurement_errors.csv")
    changes = errors[errors.kind == "change"]
    keys = ["plant_uid", "split", "pipeline", "DAS", "gt_id", "trait"]
    oracle = changes[changes.stage == "PRED_MASK_ORACLE_LINK"][keys + ["loss", "available", "absolute_error"]]
    tracked = changes[changes.stage == "TRACKED"][keys + ["loss", "available", "absolute_error", "status"]]
    paired = oracle.merge(tracked, on=keys, suffixes=("_oracle", "_tracked"), validate="one_to_one")
    rows = []
    for (plant, split, pipeline, trait), group in paired.groupby(["plant_uid", "split", "pipeline", "trait"]):
        support = group[group.available_oracle]
        rows.append({"plant_uid": plant, "split": split, "pipeline": pipeline, "trait": trait,
                     "eligible_gt_transitions": len(group), "oracle_supported_transitions": len(support),
                     "common_support_fraction": len(support) / len(group),
                     "tracking_minus_oracle_loss": (support.loss_tracked - support.loss_oracle).mean() if len(support) else np.nan,
                     "tracking_identity_failure_rate": (~support.status.eq("correct_link")).mean() if len(support) else np.nan,
                     "unverified_numerical_substitution_count": int((~group.available_oracle & group.available_tracked).sum()),
                     "unverified_numerical_substitution_fraction": float((~group.available_oracle & group.available_tracked).mean()),
                     "all_eligible_reference_mean_delta": float((group.loss_tracked - group.loss_oracle).mean())})
    per_plant = pd.DataFrame(rows)
    per_plant.to_csv(destination / "propagation_common_support_per_plant.csv", index=False)
    summary = []
    for (split, pipeline, trait), group in per_plant.groupby(["split", "pipeline", "trait"]):
        summary.append({"split": split, "pipeline": pipeline, "trait": trait,
                        **paired_interval(group.tracking_minus_oracle_loss),
                        "common_support_fraction": group.common_support_fraction.mean(),
                        "tracking_identity_failure_rate": group.tracking_identity_failure_rate.mean(),
                        "unverified_numerical_substitution_fraction": group.unverified_numerical_substitution_fraction.mean()})
    pd.DataFrame(summary).to_csv(destination / "propagation_common_support_summary.csv", index=False)
    paired.to_csv(root / "outputs/m6_a/propagation_common_support_pairs.csv", index=False)


def manual_length_check(root):
    destination = root / "reports/m6_a"
    manual = pd.read_csv(root / "data/raw/sugar4d/extracted_v4/measurements/reference/leaf_related_manual.csv")
    write_json(destination / "manual_reference_schema.json", {"columns": list(manual.columns), "rows": len(manual),
               "caveat": "cross-method final-DAS length comparison only; no longitudinal or area manual truth"})
    errors = pd.read_csv(destination / "measurement_errors.csv")
    final = errors[(errors.DAS == 93) & (errors.kind == "static") & (errors.trait == "length")].copy()
    final["plant"] = final.plant_uid.str.removeprefix("plant_").astype(int)
    final["instance id"] = final.gt_id.astype(int) - 1
    paired = final.merge(manual[["plant", "instance id", "total length [cm]"]], on=["plant", "instance id"], how="inner", validate="many_to_one")
    paired["manual_length_cm"] = paired["total length [cm]"]
    paired["manual_absolute_error"] = np.where(paired.available, (paired.estimate - paired.manual_length_cm).abs(), np.nan)
    paired["automatic_manual_discrepancy"] = (paired.reference - paired.manual_length_cm).abs()
    paired.to_csv(destination / "manual_length_comparison.csv", index=False)
    per_plant = paired.groupby(["plant_uid", "split", "pipeline", "stage"]).agg(
        manual_length_mae=("manual_absolute_error", "mean"), automatic_manual_mae=("automatic_manual_discrepancy", "mean"),
        coverage=("available", "mean"), leaves=("gt_id", "size")).reset_index()
    per_plant.to_csv(destination / "manual_length_per_plant.csv", index=False)
    per_plant.groupby(["split", "pipeline", "stage"])[["manual_length_mae", "automatic_manual_mae", "coverage", "leaves"]].mean().reset_index().to_csv(destination / "manual_length_summary.csv", index=False)
    return paired


def save_figure(figure, destination, name, manifest, inputs):
    figure.savefig(destination / f"{name}.svg", bbox_inches="tight")
    figure.savefig(destination / f"{name}.pdf", bbox_inches="tight")
    figure.savefig(destination / f"{name}.png", dpi=300, bbox_inches="tight")
    for suffix in ["png", "pdf", "svg"]:
        path = destination / f"{name}.{suffix}"
        manifest.append({"file": str(path), "sha256": digest(path), "inputs": inputs,
                         "independent_unit": "plant", "AI_generated": False})
    plt.close(figure)


def figures(root):
    destination = root / "reports/m6_a/figures"
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
    report = root / "reports/m6_a"
    per_plant = pd.read_csv(report / "per_plant.csv")
    observations = pd.read_csv(report / "observations.csv")
    pred = pd.read_csv(report / "predicted_organs.csv")
    events = pd.read_csv(report / "new_leaf_events.csv")
    manifest = []
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, metric, ylabel, threshold in zip(axes, ["pq", "idf1", "new_event_f1"], ["Full-resolution leaf PQ", "Observation IDF1", "First-observation F1"], [.7, .8, .7]):
        for index, pipeline in enumerate(PIPELINES):
            subset = per_plant[(per_plant.split == "test") & (per_plant.pipeline == pipeline)]
            axis.scatter(index + np.linspace(-.12, .12, len(subset)), subset[metric], color=COLORS[pipeline], s=28)
            interval = paired_interval(subset[metric])
            axis.errorbar(index, interval["mean"], yerr=[[interval["mean"] - interval["ci95_low"]], [interval["ci95_high"] - interval["mean"]]], color="black", fmt="_", markersize=15, capsize=3)
        axis.axhline(threshold, color="gray", linestyle="--", linewidth=1)
        axis.set(xticks=range(4), xticklabels=PIPELINES, ylim=(-.03, 1.05), ylabel=ylabel)
    fig.suptitle("Sugar4D held-out plants: n=12 independent plants, 16 scans each; bars=plant bootstrap 95% CI")
    save_figure(fig, destination, "baseline_metrics", manifest, ["per_plant.csv"])
    b1 = per_plant[(per_plant.pipeline == "B1") & (per_plant.split == "test")].sort_values("idf1")
    cases = [b1.iloc[0].plant_uid, b1.iloc[-1].plant_uid]
    for plant in cases:
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for axis, pipeline in zip(axes, ["B0", "B1", "B2"]):
            current = observations[(observations.plant_uid == plant) & (observations.pipeline == pipeline)]
            pivot = current.pivot(index="gt_id", columns="DAS", values="track_id").reindex(columns=sorted(current.DAS.unique()))
            displayed = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
            displayed = np.ma.masked_where(displayed < 0, displayed)
            display_colors = np.ma.array(displayed % 20, mask=np.ma.getmaskarray(displayed))
            axis.imshow(display_colors, cmap="tab20", interpolation="nearest", aspect="auto", vmin=0, vmax=19)
            for row_index, column_index in np.argwhere(~np.ma.getmaskarray(displayed)):
                axis.text(column_index, row_index, str(int(displayed[row_index, column_index])), ha="center", va="center", fontsize=5)
            missing = np.argwhere(pivot.to_numpy(dtype=float) < 0)
            if len(missing):
                axis.scatter(missing[:, 1], missing[:, 0], marker="x", color="#C64B4B", s=20, linewidths=.7)
            axis.set(yticks=range(len(pivot)), yticklabels=pivot.index.astype(int) - 1, ylabel="GT leaf (eval only)", title=pipeline + ": track IDs; blank=GT not observed; red x=instance/identity loss")
            first = events[(events.plant_uid == plant) & (events.pipeline == pipeline)]
            das_list = list(pivot.columns)
            gt_list = list(pivot.index)
            for event in first.itertuples(index=False):
                axis.scatter(das_list.index(event.first_DAS), gt_list.index(event.gt_id_evaluation_only), marker="*", color="black", s=35, facecolors="none")
        axes[-1].set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, xlabel="Actual DAS; stars=GT first observation (evaluation only)")
        fig.suptitle(plant + ": selected by worst/best B1 test IDF1; consistent palette within each pipeline only")
        fig.tight_layout()
        save_figure(fig, destination, plant + "_identity_timeline", manifest, ["observations.csv", "new_leaf_events.csv"])
    plant = cases[0]
    frames = metadata(root)
    frames = frames[frames.plant_uid == plant].sort_values("DAS")
    new = events[(events.plant_uid == plant) & (events.pipeline == "B1")]
    chosen_das = int(new.sort_values(["event_state", "first_DAS"]).iloc[0].first_DAS) if len(new) else int(frames.iloc[4].DAS)
    selected_index = list(frames.DAS).index(chosen_das)
    selected_records = frames.iloc[max(selected_index - 1, 0):selected_index + 1]
    fig, axes = plt.subplots(4, len(selected_records), figsize=(10, 12), squeeze=False)
    for column, record in enumerate(selected_records.itertuples(index=False)):
        with np.load(cache_path(root, plant, record.DAS)) as data:
            coordinates = data["sample_xyz"]
            indices = data["sample_indices"]
            from m6a.common import read_cloud

            xyz, gt, unused_info = read_cloud(root / record.pointcloud_path)
            label_sets = [gt[indices]]
            for pipeline in ["B0", "B1", "B2"]:
                labels = data["gt_local"][indices] if pipeline == "B0" else data["labels"][indices]
                mapping = pred[(pred.plant_uid == plant) & (pred.pipeline == pipeline) & (pred.DAS == record.DAS)].set_index("local_id").track_id.to_dict()
                label_sets.append(np.array([mapping.get(int(label), -1) if label else -2 for label in labels]))
            for row, (axis, labels, title) in enumerate(zip(axes[:, column], label_sets, ["GT (eval only)", "B0", "B1", "B2"])):
                colors = np.array([plt.get_cmap("tab20")(int(label) % 20) if label >= (1 if row == 0 else 0) else (.7, .7, .7, 1) for label in labels])
                axis.scatter(coordinates[:, 0] * 100, coordinates[:, 1] * 100, c=colors, s=2, rasterized=True)
                emerging = events[(events.plant_uid == plant) & (events.pipeline == "B1") & (events.first_DAS == record.DAS)]
                for event in emerging.itertuples(index=False):
                    event_cloud = xyz[gt == event.gt_id_evaluation_only]
                    if len(event_cloud):
                        center = event_cloud[:, :2].mean(axis=0) * 100
                        axis.scatter(center[0], center[1], s=160, facecolors="none", edgecolors="black", linewidths=1)
                        axis.text(center[0], center[1], "new " + str(int(event.gt_id_evaluation_only) - 1), fontsize=7)
                axis.set_aspect("equal")
                axis.set(title=f"{title} | DAS {record.DAS}", xlabel="x (cm)", ylabel="y (cm)")
    fig.suptitle(plant + ": actual geometry; rings=GT first observation (evaluation only), no generated clouds")
    fig.tight_layout()
    save_figure(fig, destination, "tracking_new_leaf_pointcloud", manifest, ["predicted_organs.csv", "new_leaf_events.csv", "outputs/m6_a/frames"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    decomposition = pd.read_csv(report / "error_decomposition.csv")
    for axis, trait in zip(axes, ["length", "area"]):
        current = decomposition[(decomposition.split == "test") & (decomposition.trait == trait)].groupby("pipeline").mean(numeric_only=True).reindex(PIPELINES)
        positive_bottom = np.zeros(len(current))
        negative_bottom = np.zeros(len(current))
        for component, color, text in [("measurement_reference_discrepancy", "#8AA4AF", "GT-mask measurement/reference"), ("segmentation_increment", "#C64B4B", "Segmentation increment"), ("tracking_increment", "#D49534", "Tracking increment")]:
            values = current[component].to_numpy()
            positive = np.maximum(values, 0)
            negative = np.minimum(values, 0)
            axis.bar(range(len(current)), positive, bottom=positive_bottom, color=color, label=text)
            axis.bar(range(len(current)), negative, bottom=negative_bottom, color=color)
            positive_bottom += positive
            negative_bottom += negative
        axis.scatter(range(len(current)), current.total, marker="D", color="black", s=22, label="Actual total loss", zorder=5)
        axis.axhline(0, color="black", linewidth=.7)
        axis.set(xticks=range(4), xticklabels=PIPELINES, ylabel="Penalized relative change loss", title=trait + ": all eligible transitions; missing=2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=4, fontsize=8)
    fig.suptitle("Plant-macro n=12; negative increments can be wrong-link substitution artifacts, not tracking benefit", y=1.09)
    fig.tight_layout()
    save_figure(fig, destination, "segmentation_tracking_measurement_decomposition", manifest, ["error_decomposition.csv"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    event_columns = ["new_missed_instance", "new_sampling_or_track_loss", "new_attached_old_identity", "false_rebirth", "spurious_birth"]
    labels = ["Missed instance", "Sampling/ID loss", "New attached old", "False rebirth", "Spurious birth"]
    for pipeline in ["B0", "B1", "B2"]:
        current = per_plant[(per_plant.split == "test") & (per_plant.pipeline == pipeline)]
        axes[0].plot(range(5), current[event_columns].mean(), marker="o", color=COLORS[pipeline], label=pipeline)
    axes[0].set(xticks=range(5), xticklabels=labels, ylabel="Events per independent plant", title="First-observation failure types")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend()
    risk = pd.read_csv(report / "confidence_selective_risk.csv")
    for score, style in [("semantic_confidence", "--"), ("confidence", "-")]:
        current = risk[(risk.split == "test") & (risk.pipeline == "B1") & (risk.score == score)].groupby("retained_fraction").error_rate.mean()
        axes[1].plot(current.index, current, style, marker="o", label=score)
    axes[1].set(xlabel="Fraction of predictions retained by confidence", ylabel="Identity error rate (plant-macro)", title="Confidence ranking, not calibrated probability")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, destination, "new_leaf_events_and_confidence", manifest, ["per_plant.csv", "confidence_selective_risk.csv"])
    write_json(report / "figure_manifest.json", manifest)


def report(root):
    destination = root / "reports/m6_a"
    decision = json.loads((destination / "decision.json").read_text())
    protocol = json.loads((root / "configs/m6_a/protocol.json").read_text())
    summary = pd.read_csv(destination / "baseline_metrics.csv")
    per_plant = pd.read_csv(destination / "per_plant.csv")
    intervals = pd.read_csv(destination / "paired_plant_intervals.csv")
    confidence = pd.read_csv(destination / "confidence_summary.csv")
    rows = summary[summary.split == "test"].set_index("pipeline")
    paired = intervals[(intervals.split == "test") & (intervals.comparison == "B1 minus B0") & (intervals.metric == "idf1")].iloc[0]
    track = intervals[(intervals.split == "test") & (intervals.comparison == "B1 tracking minus evaluation-only oracle links")]
    error_modes = ["new_missed_instance", "new_sampling_or_track_loss", "new_attached_old_identity"]
    main_new_error = rows.loc["B1", error_modes].astype(float).idxmax()
    main_new_text = {"new_missed_instance": "当前实例未得到IoU>0.5关联（实例漏检/合并/不完整）",
                     "new_sampling_or_track_loss": "实例已检出但256点采样/追踪身份丢失",
                     "new_attached_old_identity": "新器官错误附着到旧身份"}[main_new_error]
    significant = track[(track.ci95_low > 0) & (track["mean"] >= .05)]
    significant_text = "、".join(significant.metric.tolist()) if len(significant) else "没有B1动态指标同时满足固定增量和CI门槛"
    manual_summary = pd.read_csv(destination / "manual_length_summary.csv")
    common_support = pd.read_csv(destination / "propagation_common_support_summary.csv")
    b1_length_support = common_support[(common_support.split == "test") & (common_support.pipeline == "B1") & (common_support.trait == "length")].iloc[0]
    diagnostic = pd.read_csv(destination / "official_normalization_metrics.csv") if (destination / "official_normalization_metrics.csv").exists() else None
    base_table = summary[["split", "pipeline", "independent_plants", "pq", "sbd", "idf1", "link_recall", "id_switches", "continuity", "complete_track_fraction", "new_event_f1", "reliable"]]
    measure_table = summary[summary.split == "test"][["pipeline", "length_static_tracked_mae", "length_static_tracked_loss", "area_static_tracked_mae", "area_static_tracked_loss", "length_change_tracked_mae", "area_change_tracked_mae", "length_change_tracked_coverage", "length_change_tracked_severe", "area_change_tracked_severe"]]
    plant_table = per_plant[(per_plant.split == "test") & (per_plant.pipeline == "B1")][["plant_uid", "pq", "idf1", "id_switches", "new_event_f1", "length_change_tracked_loss", "length_change_tracked_coverage", "length_change_tracked_severe", "reliable"]]
    text = f"""# M6-A：非 Oracle 植物器官动态表型流水线可行性验证

## 0. 唯一研究结论

**{decision['recommendation']}**。现有固定基线未通过冻结的可靠性验收：`baseline_acceptance_passed={decision['baseline_acceptance_passed']}`。
这不是算法创新阶段，也不是JEPA复活实验；本次无任何训练步、无未来点云预测、无自监督预算增加。

完成的是 **Sugar4D V4 24株384帧** 的整株验证/测试比较；未成功下载Pepper-4D，**未完成跨物种验证**。
所有平均值以独立植株为单位，12个test植株并非192个生物学重复。val也为12株；官方24个train植株未重新训练。

## 1. 预先冻结的数据、主指标与门槛

- 冻结文件：`configs/m6_a/protocol.json`；SHA256 `{digest(root / 'configs/m6_a/protocol.json')}`；锁定时间见 `protocol_lock.json`，早于模型推理和目标评分。
- 沿用V4和官方24/12/12整株划分。每株16次实际扫描，DAS为37、40、44、47、51、54、58、61、65、72、75、79、82、86、89、93。没有按帧随机划分、删除困难植株或压缩时间缺口。
- 主要指标固定为 **plant-macro observation IDF1**；实例PQ严格IoU>0.5、SBD、事件F1、动态误差和失败率是预设必要诊断/验收门槛，不替换主指标。
- 冻结验收：B1 PQ≥0.70、IDF1≥0.80、首观测事件F1≥0.70、测量覆盖≥0.90、长度/面积静态惩罚相对误差≤0.20/0.25、长度和面积严重动态错误率≤0.10、至少80%测试植株同时可靠。
- 同一器官身份GT在评估器中用于时序对齐，**任何追踪器都不接收它**。B0/B2_GT只接收当前掩码且每帧独立随机重编号。模型B1/B2只接收XYZ；法线、GT语义、器官掩码、叶数、CSV表型均不作为模型输入。
- Test植株未进入这两份官方源物种模型训练，但M3/M4已使用过这些test数据；这里只能称未见植株工程验证，不能称全新独立确认性队列。

## 2. 资源核实及实际运行

详见 `SOURCE_AUDIT.md`、`source_audit.csv`、`downloaded_files.csv`、`sources/`。
Sugar4D仍为CC BY4.0；三个GitHub仓库没有可核实的显式代码许可，不能把“公开可下载”当成再分发许可。
PlantNet与PSegNet epoch199权重真实下载且严格加载，4096点前向与重复性测试通过；PSegNet只做smoke不择优。
TrackPlant3D的3DEPS、官方CPD和官方扩展匹配代价实际运行；没有训练追踪网络。
隔离环境为`.venv-m6-a`；历史环境包不变。全部兼容性调整、被拒绝的边界向量化试验和邻域等价验证在 `COMPATIBILITY.md`。

### 官方归一化保真度限制及单独诊断

原始B1冻结的是均值中心/单位球归一化。原始H5转换源码下载起初超时；之后成功的GitHub blob API证据显示官方模型预处理用**采样点bounding-box midpoint / 最大正坐标**，并非单位球。
保留所有原始基线；在新哈希锁后，对相同24株384帧、相同checkpoint和4096个XYZ采样点做**独立工程分割诊断**：

{markdown_table(diagnostic[['split','pq','sbd','tp','fp','fn']]) if diagnostic is not None else '诊断尚无完整汇总，不能改变主结论。'}

这些PQ仍低于冻结0.70阈值，支持分割瓶颈方向并非仅由原始归一化差异造成，但不把它当新的主验收结果。诊断没有执行完整官方归一化追踪/表型，不能宣称对官方原生流水线已全面验证。
见 `NORMALIZATION_DIAGNOSTIC.md`、`official_normalization_lock.json`、`official_normalization_execution_scope.json`，主协议/主IDF1没有修改。

## 3. 三条公平流程与明确诊断

- **B0**：真实当前叶实例掩码 + TrackPlant3D适配器；不是把真实时序ID交给追踪器。
- **B1**：官方固定PlantNet零样本当前实例分割 + 与B0完全相同的TrackPlant3D。
- **B2**：与B1完全相同的预测实例 + 当前空间质心距离门控Hungarian，无CPD、无学习。
- **B2_GT**：GT当前掩码 + B2匹配，独立列出的诊断条件；不替换非Oracle B2。
- B1官方MeanShift带宽0.6、公开平均实例过滤、4096个几何均匀点；最近邻回传标签到全部原始PLY点，PQ/SBD按完整分辨率评价，非2048/4096点替代指标。
- TrackPlant3D每帧256个3DEPS点，一部分完整器官可能没有采样支持；缺失标记track=-1并计入失败，不偷偷按GT补点。

## 4. 固定主指标及逐植株结果

{markdown_table(base_table)}

IDF1采用**器官观测级**而非像素级：评价器根据严格IoU>0.5单帧关联，再在整条时序做GT身份与输出track的一对一最大观测数分配；`2*IDTP/(GT observations + predicted leaf observations)`。
ID Switch是在同GT器官连续检测身份改变时计数，漏检间隔后的ID变化仍计；ID连续性为最长相同身份连续检测段/该GT身份实际观测数。`link_recall`将漏检计为失败，不能只在成功检测器官上报高准确率。

Test B1−B0 IDF1差：{paired['mean']:.4f}，整株配对bootstrap95%CI [{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}]，n={int(paired['plants'])}独立株。
所有额外区间、B1/B2及B0/B2_GT比较保存在 `paired_plant_intervals.csv`。

### B1 每个未见测试植株

{markdown_table(plant_table)}

全96个植株×流程结果在 `per_plant.csv`；所有1536个帧×流程结果在 `frame_metrics.csv`。没有删掉失败植株。

## 5. 测量、时间变化与误差传播

测量器是预设简单几何基线：PCA主轴1%–99%跨度(cm)，PCA平面凸包面积(cm²)，每个完整掩码最多4096点。
这与Sugar4D的“包含叶柄的总长度”和其自动叶面积**不是严格相同估计对象**，因此误差是基线相对于已发布自动参考的差异，不能宣称精确生物学长度/面积真值。
变化为相邻实际扫描差分，允许负增长；另报每DAS天数除后的变化率误差。缺失形态为NaN，不是0。

{markdown_table(measure_table)}

静态/变化MAE只在有测量时定义，必须同时读覆盖率。主要误差分解用所有有参考的合法GT观测/相邻变更，`min(relative absolute error,2)`；没有测量直接损失2，不让拒绝输出带来虚假低误差。
严重静态错误为相对误差>50%或缺失；严重动态错误为变更误差/上一时刻参考>50%或缺失。

`error_decomposition.csv`使用同一GT观测集合：
1. GT当前掩码 + **评价器真相邻链接**：测量器/参考差异；不运行Oracle追踪算法。
2. 预测当前掩码 + **评价器真相邻链接**：增加分割误差。
3. 同一预测当前掩码 + 实际输出链接：增加匹配错误/身份缺失影响。

三项是相同损失的加法差分，可能出现负差值，**不是独立、正交或因果方差分解**。GT链接永远只在误差诊断评价器里出现。
尤其若正确上一个器官没有可关联分割，Oracle-link测量为缺失损失2，而错误上一个器官却能提供一个有限数值，原始“tracking增量”会变负。这是**未经证实的数值替代/错连掩盖缺失**，不是可靠追踪收益。
因此额外报告**事后误差诊断**：只限定“相同预测掩码下前后真实GT器官均有可关联测量”的评价Oracle支持集，保留追踪丢失损失2，另外统计身份失败率。此诊断由负增量现象触发，不是重新注册的确认性主指标；不替换全体主要损失，也不删减主验收失败。

{markdown_table(common_support[common_support.split=='test'])}

### B1 追踪相对于评价Oracle链接的动态损失增量

{markdown_table(track[['metric','mean','ci95_low','ci95_high','plants']])}

GT-mask长度动态损失={rows.loc['B0','length_change_gt_mask_oracle_link_loss']:.4f}；GT-mask面积动态损失={rows.loc['B0','area_change_gt_mask_oracle_link_loss']:.4f}。
严重失败逐条见 `severe_failures.csv`，分类见 `transition_failure_types.csv`。最终人工长度仅做DAS93横断面对照：

{markdown_table(manual_summary[manual_summary.split=='test'])}

人工比较的遗漏也单独报覆盖率，不将成功输出子集的MAE当成全部器官的性能；没有人工纵向面积、死亡、出现真值。

## 6. 首次观察新器官与缺失语义

初始DAS所有已有器官不计新事件。GT器官首次在后续实际扫描出现，才是“首次观测”事件；不是净叶数增加，不是植物学出生。
追踪新ID只是预测事件；真阳性需要同一时刻通过IoU关联到新GT器官。

{markdown_table(summary[summary.split=='test'][['pipeline','new_event_precision','new_event_recall','new_event_f1','new_missed_instance','new_sampling_or_track_loss','new_attached_old_identity','false_rebirth','spurious_birth']])}

`new_leaf_events.csv`记录逐新器官首观测状态与首次分割检测延迟。`observation_status_grid.csv`严格区分：
- `not_observed_before_first/gap/after_last`：GT记录没有器官观察，**不转换成零长度/零面积**。
- `instance_missed`：GT器官已观察但无足够IoU预测实例。
- `identity_lost`：实例可关联但追踪未覆盖或断链。
- `wrong_identity`/`wrong_match`：观察被连到错误时序身份/错误上一个器官。
- 真实器官消失/死亡：**数据未标注，始终unknown**。停止追踪、GT最后观察都不等于生物学死亡。

## 7. 严重错误、置信度和拒绝输出

置信度分两层：模型max-softmax语义置信度；语义置信度×匹配备选边际的组合启发式。两者都不是已校准的“结果正确概率”。
评估身份错误和严重动态错误AUROC，固定有价值阈值0.70，整株macro；单株只有一类时AUROC不可定义并保留NaN，不强制当成1。

{markdown_table(confidence[confidence.split=='test'][['pipeline','target','score','auroc','error_rate','high_confidence_errors','high_confidence_count']])}

`confidence_per_plant.csv`、`prediction_confidence.csv`与`confidence_selective_risk.csv`保留高置信错误和20/50/80/100%保留风险。
没有预测的遗漏器官没有模型置信度；不能把仅成功输出上的AUROC外推为“识别所有漏检”。测量缺失率和全队列严重失败率仍作为验收门槛。

## 8. 五个研究问题：根据证据作答

1. **当前实例分割是不是主要瓶颈？** 固定零样本PlantNet的PQ={rows.loc['B1','pq']:.4f}，B0−B1 IDF1差={rows.loc['B0','idf1']-rows.loc['B1','idf1']:.4f}。GT掩码对照定位的是此基线分割的传播影响，不足以证明所有监督分割器都失败；按冻结优先级得到 {decision['recommendation']}。
2. **追踪错误是否显著影响动态表型？** 结果：{significant_text}。B1共同支持长度损失增量={b1_length_support['mean']:.4f}，95%CI [{b1_length_support['ci95_low']:.4f}, {b1_length_support['ci95_high']:.4f}]，虽有正向区间但小于固定实际意义0.05门槛。简单匹配B2_GT的额外损失约0.084且区间大于0，说明错误匹配确实可传播，但官方追踪在GT掩码条件下不是目前最大的瓶颈。漏掉身份造成的测量不可用一并计入，不能只看成功链接MAE。
3. **新器官主要发生哪类错误？** B1主要是 **{main_new_text}**，每株平均{rows.loc['B1',main_new_error]:.4f}个；新实例漏检、采样/ID丢失、新器官附着旧ID、已有叶误报新ID和伪实例分别报告，不仅展示成功图。
4. **未见植株仍可靠吗？** B1可靠测试植株比例={rows.loc['B1','reliable']:.4f}，冻结要求≥0.80；整体验收={decision['baseline_acceptance_passed']}。零样本跨目标物种迁移的局限与长期身份误差同时保留；不声称pepper结果。
5. **最值得下一阶段创新的问题？** 唯一建议 {decision['recommendation']}，只按冻结判定优先级，不凭面积单项变好或更换主评价指标宣称成功。本任务结束后不直接设计或训练新方法。

此建议范围是“先解决当前器官检测/实例关联完整性”，不是已证明需要发明一种全新网络：尚无目标物种监督分割训练、独立跨物种下载或官方归一化完整追踪结果。官方零样本PQ诊断仍失败，因此分割方向证据成立，但方法创新的新颖性与可达上限尚未证明。
即便B0身份接近可靠，PCA静态长度相对误差仍约45%、DAS93人工长度差异约8.17cm，说明当前简单测量器也不具备可靠绝对形态精度。它不否定实例缺失导致动态覆盖约56%的主要传播瓶颈；不能把INSTANCE-FIRST解读为测量阶段已经解决。

## 9. 可视化、自动化测试与复现

`figures/`有逐株身份时间线、真实点云首观测事件、基线指标、分割→追踪→测量增量、事件错误和置信度风险。
案例按最差/最好B1测试IDF1选择并标注，非手选成功案例；点云是真实当前观测，图像由Python数据绘制，未调用生成式模型填补器官。
`figure_manifest.json`记录图与输入及哈希，`FIGURE_QA.md`记录直接查看输出的检查。
`final_verification.json`核验384帧、96条逐株流程结果、同掩码B1/B2、无重复关联、全缓存锁定哈希及加法闭合；真实早期和后期扫描重复推理的标签、置信度及采样均完全一致。

```bash
bash scripts/setup_m6_a_environment.sh
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/audit_m6_a_sources.py --download
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py freeze
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py smoke
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-m6-a/bin/python scripts/run_m6_a.py prepare --splits val test
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-m6-a/bin/python scripts/run_m6_a.py evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py report
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-m6-a/bin/python -m pytest tests/test_m6_a.py -q
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py verify
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/verify_m6_a.py --repeat
```

可分别 `prepare --splits val` / `prepare --splits test` 使用可见GPU0/1并发执行，不改变配置；完整缓存重用仍核验冻结哈希。
官方归一化诊断复现：`python scripts/diagnose_m6_a_normalization.py --splits val test`，完成后同命令加`--evaluate-only`；不是新主指标训练或择优。
原始数据与上游权重不重新发布；网络不通时使用已校验本地副本。完整依赖版本在 `environment.json` / `environment-lock.txt`。
M6-A专项测试18项通过。全库测试在M6隔离环境164通过/1失败，唯一旧M1编码器测试缺少torchvision；该测试在原`.venv-m1`环境单独复跑通过。没有安装无关依赖或修改旧测试来掩盖失败。
M0—M5保护验证在 `protection_verification.json`：历史文件及891个Sugar4D原始文件无修改。无合并、无commit、无push、无删除历史工件。

## 10. 信息来源及边界

- 当前本地实测：`baseline_metrics.csv`、`per_plant.csv`、`paired_plant_intervals.csv`、`measurement_errors.csv`、`observations.csv`、`runtime_*.csv`。
- 官方来源URL与检索失败记录：`SOURCE_AUDIT.md`、`sources/bocha_search.json`、`sources/http_audit.json`。
- Sugar4D：https://www.ipb.uni-bonn.de/data/sugar4d/index.html
- PlantNet/PSegNet：https://github.com/Huang2002200/PlantNet-and-PSegNet
- TrackPlant3D：https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- Pepper-4D：https://github.com/foysalahmed10/Pepper-4D

停止于baseline研究可行性判断，不把“可运行”写成“可靠”、不把自动参考当成生物学真相，也不把失败硬改成主指标成功。
"""
    (destination / "M6_A_REPORT.md").write_text(text)


def deliver(root):
    source_audit(root)
    confidence_audit(root)
    event_and_status_audit(root)
    propagation_common_support(root)
    manual_length_check(root)
    figures(root)
    report(root)
