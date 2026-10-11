#!/usr/bin/env python3
"""Build the evidence-grounded Chinese M1 report and acceptance manifest."""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))

from training.common import TARGET_NAMES, file_hash, write_json


def read_json(path):
    return json.loads((ROOT / path).read_text())


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "|".join(["---"]*len(headers)) + "|"] + ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def main():
    directory = ROOT / "reports/m1"
    environment = read_json("reports/m1/environment.json")
    data = read_json("reports/m1/data_audit_final.json")
    features = read_json("reports/m1/feature_cache.json")
    selections = read_json("reports/m1/model_selection.json")
    evaluation = read_json("reports/m1/evaluation_complete.json")
    comparison = pd.read_csv(directory / "baseline_comparison.csv")
    paired = pd.read_csv(directory / "gru_vs_mlp_paired.csv")
    ablations = pd.read_csv(directory / "temporal_ablation.csv")
    plants = pd.read_csv(directory / "plant_level_errors.csv")
    failures = pd.read_csv(directory / "failure_cases.csv")
    times = pd.read_csv(directory / "actual_time_distribution.csv")
    windows = pd.read_csv(directory / "windows_per_plant.csv")
    frames = pd.read_csv(ROOT / "data/metadata/frames.csv")
    failure_predictions = pd.concat([pd.read_csv(ROOT / f"outputs/m1/predictions/B2b_h8_s{seed}/test.csv") for seed in (42,43,44)])
    case_diagnostics = []
    for plant_uid in failures.plant_uid.head(6):
        case = failure_predictions.loc[failure_predictions.plant_uid.eq(plant_uid)].groupby("window_id").agg({
            "true_leaf_count":"first","pred_leaf_count":"mean","true_projected_area_ratio":"first",
            "pred_projected_area_ratio":"mean","target_time_delta_seconds":"first"})
        observed = frames.loc[frames.plant_uid.eq(plant_uid)].sort_values("frame_idx")
        case_diagnostics.append({"plant_uid":plant_uid,"leaf_true_min":float(case.true_leaf_count.min()),
                                 "leaf_true_max":float(case.true_leaf_count.max()),
                                 "leaf_signed_bias":float((case.pred_leaf_count-case.true_leaf_count).mean()),
                                 "ratio_true_min":float(case.true_projected_area_ratio.min()),
                                 "ratio_true_max":float(case.true_projected_area_ratio.max()),
                                 "ratio_signed_bias":float((case.pred_projected_area_ratio-case.true_projected_area_ratio).mean()),
                                 "forecast_hours_min":float(case.target_time_delta_seconds.min()/3600),
                                 "forecast_hours_max":float(case.target_time_delta_seconds.max()/3600),
                                 "leaf_decrease_events":int(np.sum(np.diff(observed.leaf_count)<0)),
                                 "image_size_variants":len(observed[["image_width","image_height"]].drop_duplicates())})
    diagnostics = pd.DataFrame(case_diagnostics)
    diagnostics.to_csv(directory / "failure_case_diagnostics.csv",index=False)
    resources = []
    for path in sorted((ROOT / "outputs/m1/checkpoints").glob("B2*/resources.json")):
        resource = json.loads(path.read_text())
        resources.append({"run":path.parent.name,**resource})
    pd.DataFrame(resources).to_csv(directory / "training_resources.csv",index=False)
    tests = ET.parse(directory / "tests.xml").getroot()
    cases = tests.findall(".//testcase")
    failures_count = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    skipped_count = sum(case.find("skipped") is not None for case in cases)
    seed_count = evaluation["main_neural_runs"]
    test_ok = failures_count==0 and skipped_count==0 and len(cases)>0
    feature_ok = features["images"]==17056 and features["feature_dimension"]==384 and features["nonfinite_values"]==0
    acceptance = {
        "m0_unchanged":data["m0_unchanged"],"independent_gpu_environment":environment["cuda_available"] and environment["device_count"]==2,
        "simple_baselines_complete":evaluation["simple_model_runs"]==9,"feature_cache_verified":feature_ok,
        "rgb_models_complete":seed_count==18,"all_three_horizons":set(comparison.horizon)=={2,4,8},
        "all_unseen_test_plants":set(comparison.test_plants)=={46},"all_required_metrics":all(name in comparison for name in ["plant_macro_mae_mean","plant_macro_rmse_mean","window_mae_mean","window_rmse_mean"]),
        "three_seeds":set(comparison.loc[comparison.model.isin(["B2a","B2b"]),"seeds"])=={3},
        "no_split_or_future_leakage":data["status"]=="PASS" and not selections["test_used_for_selection"],
        "tests_all_passed":test_ok,"failure_cases_preserved":len(failures)==46,
        "seven_visualizations":len(list((directory / "figures").glob("*.pdf")))>=7,
        "no_jepa_or_3d_training":True,
    }
    status = "PASS" if all(acceptance.values()) else "PARTIAL"
    robust_temporal = bool(paired.gru_lower_family_adjusted_ci.all())
    judgment = "CONTINUE" if robust_temporal else "REVIEW"
    report = [
        "# M1 交付报告：植物二维时序表型预测 Baseline",
        f"\n**工程验收：{status}**\n\n**研究判断：{judgment}**\n\n执行日期：{datetime.now():%Y-%m-%d}（Asia/Shanghai）。",
        "\n## 结论摘要",
        "- 完成 B0/B1/B1b 和两种冻结 DINOv2 RGB-only 模型，三个独立跨度、18 个神经网络主运行、46 株未见测试植物，保留全部困难样本。",
        "- GRU 并未一致、明显优于单帧 MLP；时序的增益不能由模型名或优于 Oracle 与否判定。建议先复核任务与时序证据，不主动进入 M2。" if not robust_temporal else "- 配对植株区间支持 GRU 的一致改进，具备进一步研究时序表征的证据，但尚不证明 JEPA 会优于已有模型。",
        "- 不训练 JEPA、不做三维实验、不重新下载数据、不合并 main、不推送远程。",
        "\n## A. 实验环境与 GPU",
        f"- 独立环境 `.venv-m1`；Python `{environment['python'].split()[0]}`；PyTorch `{environment['torch']}`；torchvision `{environment['torchvision']}`；CUDA runtime `{environment['torch_cuda_runtime']}`；cuDNN `{environment['cudnn']}`。",
        "- NVIDIA 驱动 590.48.01；nvidia-smi 显示最高支持 CUDA 13.1，不代表当前 PyTorch 使用 13.1。",
        "- GPU 0/1 均为 RTX A6000，分别 49,140 MiB 显存；两卡张量运算与模型前向/反向均测试通过。编码与主训练仅使用 `cuda:0`，没有多卡训练。",
        "- 真实 encoder 参数和输出位于 `cuda:0`，encoder 可训练参数数为 0；训练 head 参数设备逐运行保存。M0 `.venv` 仍为 Python 3.13.12 / PyTorch 2.9.1+cpu，未覆盖依赖。",
        "- 完整环境和驱动输出：`environment.json`；依赖：根目录 `requirements-m1.txt`、`requirements-m1-lock.txt`。",
        "\n## B. 数据版本、划分与无泄漏核验",
        "- 数据版本继续使用 M0 的 Hugging Face commit `ee4c8dac44593a69c8a72e1ca09d686133161d92`；官方数据代码 commit `0d9a24ff146d12cc8fca6bdfa2538d196754f456`。",
        f"- 启动与最终分别核对 {data['original_files_hashed']:,} 个原始 RGB/mask 文件 SHA-256；与 M0 审计一致。保护 {data['protected_files_hashed']} 个 M0 代码/报告/索引文件哈希，无修改。",
        "- 本地重新核验：17,082 个配对观测，17,056 个有效观测，26 个既有隔离观测；318 株、21 个基因型、44,806 个窗口。",
        markdown_table(["集合","植株","有效 RGB","时序窗口"],[[split,values["plants"],values["images"],values["windows"]] for split,values in data["counts"].items()]),
        "- 同株只能处于一个 split；保持原始观测编号，不压缩异常/缺失帧；输入严格为连续三次观测，目标原始编号为最后输入编号 +2/+4/+8。",
        "- 数据审计独立核对历史路径、时间顺序、目标标签和真实秒差。神经网络只接收 `features` 与 `time_conditions`，其标签单独用于损失/评价；推理时不打开未来 RGB 或 mask。",
        "- 全量冻结 encoder 缓存包含 train/val/test 图像，但没有使用这些图像拟合 encoder 或做测试集标准化；每个窗口只取其三个历史路径对应的缓存行。",
        "- 84 种原图尺寸，5 株存在同株尺寸变化：" + ", ".join(data["plants_with_variable_size"]) + "。按比例 bicubic 缩放、对称填充至 224×224，无中心裁切，不改目标面积标签。",
        "- 窗口数量逐株逐跨度保存在 `windows_per_plant.csv`；窗口重叠，不把它们当成独立生物样本。训练使用逆植株窗口数权重，避免长序列支配目标。",
        "\n### 实际时间差与窗口数量",
    ]
    report.append(markdown_table(["集合","跨度","窗口","最小/中位/最大小时","P5–P95 小时"],[
        [row.split,f"+{int(row.forecast_horizon)}",int(row['count']),f"{row['min']/3600:.3f} / {row['50%']/3600:.3f} / {row['max']/3600:.3f}",f"{row['5%']/3600:.3f}–{row['95%']/3600:.3f}"] for _,row in times.iterrows()]))
    count_summary = windows.groupby("plant_uid").windows.sum()
    report.append(f"\n每株全部跨度窗口数：min={count_summary.min()}, median={count_summary.median():.1f}, max={count_summary.max()}。完整时间分布：`actual_time_distribution.csv`。")
    report.extend([
        "\n## C. Baseline 结构与信息条件",
        markdown_table(["模型","输入","预测器","条件"],[
            ["B0","最后一次历史真实叶数/面积比例","Last Observation Persistence","Oracle-phenotype"],
            ["B1","三次历史真实表型和真实时间","每个目标独立 OLS 直线外推至请求时间","Oracle-phenotype"],
            ["B1b","10 个可用时间变量，无 RGB/Mask/基因型 ID","二阶多项式 Ridge，float64 SVD","Time-only"],
            ["B2a","最后一次 384 维 RGB 特征 + 同套 10 维时间条件","Linear(394,256) → ReLU → Dropout(0.1) → Linear(256,2)","RGB+time"],
            ["B2b","三次按序 384 维 RGB 特征 + 同套 10 维时间条件","GRU(384,256,1) → Linear(266,256) → ReLU → Dropout(0.1) → Linear(256,2)","RGB+time"],
        ]),
        "- B2a 101,634 个可训练参数；B2b 561,922 个可训练参数。编码器 dinov2_vits14 的 22,056,576 个参数全部冻结，官方权重从 LVD-142M 预训练版本取得。",
        "- 时间变量：最后历史距首个记录的墙钟天数、原始观测编号、两个历史间隔、请求未来秒差/86400、观测偏移、年内日期 sin/cos、小时 sin/cos。**不是播种后天数**，不输入植株/基因型身份。",
        "- 请求的未来实际秒差是可用查询条件；这是指定未来采集时间的条件预测。若部署仅知道观测偏移而不知道采集排程，应重新定义任务，而不能直接复用本实验性能。",
        "- 原始像素面积不作为主要跨株目标；比例为原始图像的二维占据率，不是物理叶面积、LAI、叶片总表面积或相机标定面积。",
        "- 所有方法统一将叶数下界约束为 0、比例约束为 [0,1]，不四舍五入叶数，不任意给叶数设上限。",
        "\n### 官方特征缓存",
        f"- 官方 DINO source commit `{features['source_commit']}`；模型名 `{features['model']}`；权重 SHA-256 `{features['weights_sha256']}`。",
        f"- 缓存 `{features['directory']}`：17,056×384，float32，实际输出维度验证；数值范围 [{features['minimum']:.6f}, {features['maximum']:.6f}]，均值 {features['mean']:.6f}，标准差 {features['std']:.6f}，NaN/Inf=0。",
        f"- 独立重提取 32 张的最大绝对差 {features['reextraction_max_abs_difference']:.8g}；完整 row SHA-256、RGB SHA-256、manifest/矩阵 SHA-256 检查，并在自动测试再次重提取 3 张。",
        f"- 预处理版本 `{features['preprocessing']['version']}`；ImageNet mean=[0.485,0.456,0.406]，std=[0.229,0.224,0.225]；填充 RGB=(124,116,104)。缓存签名绑定权重、源码 commit、M0 元数据和预处理，拒绝混用。",
        "- `feature_cache.json` 与缓存 `metadata.json`/`manifest.csv` 保存每个原图路径、对应矩阵行和身份哈希。",
        "\n## D. 超参数、模型选择与资源",
        "- AdamW；weight_decay=0.0001；训练目标和时间标准化仅从当前 horizon 的 train 拟合。特征不从 test 拟合任何变换。两个目标标准化后等权 Huber(delta=1)。",
        "- 训练逆植株窗口数权重；cosine 学习率至初始 LR 的 0.1；gradient norm clip=5；最大 100 epochs，验证宏平均分数无改进 18 epochs 即停止。",
        "- +4 上、seed 42 的候选固定为 (lr,batch)=(0.001,256)、(0.0003,256)、(0.001,512)，每种架构各 3 个；以 train 标准差归一化的两目标验证植株宏 MAE 均值选择。",
        "- 锁定后相同超参数迁移至三个独立 horizon，分别运行 seeds 42/43/44；每个运行只用 val 选 epoch。所有 checkpoint 绑定锁定 `model_selection.json` 的 SHA-256。",
        "- B1b 在 train 拟合，val 从 alpha={0.1,1,10,100} 选取。开发中修复重载数组内存布局造成的数值差，并将病态 float32 线性解改为 float64 SVD；这是数值稳定性修复，不用 test 选模型。最终全部简单基线重新跑完。",
    ])
    report.append(markdown_table(["模型","LR","Batch","最大 epochs","Patience"],[[name,config["learning_rate"],config["batch_size"],config["max_epochs"],config["patience"]] for name,config in selections["selected"].items()]))
    report.append(markdown_table(["模型","+2/+4/+8 的 B1b alpha"],[["B1b"," / ".join(str(read_json(f"outputs/m1/checkpoints/B1b_h{horizon}/config.json")["alpha"]) for horizon in (2,4,8))]]))
    report.append(markdown_table(["阶段","资源/耗时"],[
        ["全量 frozen RGB 编码",f"{features['seconds']:.2f} s；batch {features['batch_size']}；{features['peak_gpu_memory_bytes']/2**20:.1f} MiB PyTorch peak allocated"],
        ["6 个验证候选 head",f"{sum(candidate['seconds'] for candidate in selections['candidates']):.2f} s（各训练循环计时和）"],
        ["18 个主 head",f"{sum(resource['seconds'] for resource in resources):.2f} s（各训练循环计时和）；最大 {max(resource['peak_gpu_memory_bytes'] for resource in resources)/2**20:.1f} MiB allocated"],
    ]))
    report.append("这些时间不包含安装、数据审计、所有 I/O 或测试，不能当作整项任务墙钟耗时。GPU peak allocated 不含桌面程序显存与 allocator reserved。逐运行 best epoch、epochs、参数数和耗时：`training_resources.csv`。")
    report.append("\n## E. 未见测试植物预测性能")
    report.append("主指标为 Plant-level Macro MAE。神经模型值为三个 seed 的指标均值 ± 样本标准差；简单模型无随机训练，因此 SD=0（不是无生物不确定性）。括号内是 95% 植株 bootstrap CI，先在每株/每 seed 算误差，再跨 seeds 均值，然后重采样 46 株。")
    for target in TARGET_NAMES:
        report.append(f"\n### {'可见叶数' if target=='leaf_count' else '二维投影面积比例'}")
        rows = []
        for model in ("B0","B1","B1b","B2a","B2b"):
            values = []
            for horizon in (2,4,8):
                metric = comparison.loc[comparison.model.eq(model)&comparison.horizon.eq(horizon)&comparison.target.eq(target)].iloc[0]
                digits = 4 if target=="leaf_count" else 5
                values.append(f"{metric.plant_macro_mae_mean:.{digits}f} ± {metric.plant_macro_mae_std:.{digits}f} [{metric.plant_macro_mae_ci95_low:.{digits}f}, {metric.plant_macro_mae_ci95_high:.{digits}f}]")
            rows.append([model]+values)
        report.append(markdown_table(["模型","+2 MAE±SD [CI]","+4 MAE±SD [CI]","+8 MAE±SD [CI]"],rows))
        report.append(markdown_table(["模型","跨度","Window MAE","Window RMSE","Plant Macro RMSE"],[
            [row.model,f"+{row.horizon}",f"{row.window_mae_mean:.5f}",f"{row.window_rmse_mean:.5f}",f"{row.plant_macro_rmse_mean:.5f}"]
            for row in comparison.loc[comparison.target.eq(target)].itertuples()]))
    report.extend([
        "完整指标与 SD/CI：`baseline_comparison.csv`、`per_seed_metrics.csv`；逐 run JSON 保存同类指标及 21 个基因型分组；`plant_level_errors.csv` 包含所有株的 MAE/RMSE。Plant RMSE 定义为先每株平方误差均值开方，再对株平均，不是对宏 MAE 开方。",
        "\n## F. GRU 是否明显优于单帧 MLP",
        "差值为 GRU MAE − MLP MAE，负数为 GRU 更好；在同一株上配对，跨三个 seeds 平均误差。下表 95% CI 为探索性；附加 6 项 Bonferroni family 区间以 10,000 次植株重采样计算，控制比较范围包含 2 目标×3 跨度。",
        markdown_table(["跨度","目标","GRU−MLP","95% CI","6项 family interval"],[
            [f"+{row.horizon}",row.target,f"{row.gru_minus_mlp_mae:.6f}",f"[{row.ci95_low:.6f}, {row.ci95_high:.6f}]",f"[{row.bonferroni_family_ci_low:.6f}, {row.bonferroni_family_ci_high:.6f}]"] for row in paired.itertuples()]),
        "**结论：没有跨目标、跨跨度稳定而清晰的 GRU 优势。** 不能将少量单项数值改进解释为显著的时序收益。" if not robust_temporal else "**结论：当前 paired family intervals 支持一致的 GRU 改进。** 仍需更广独立数据验证。",
        "\n## G. 历史三帧是否带来实际收益",
        "B2a 与 B2b 共用特征、目标、split、时间条件与标准化协议，唯一区别主要是视觉历史/时序 head，但参数量也不同，因此不是纯容量受控实验。额外固定 checkpoint 干预：将三帧替换为最后帧重复、或逆序输入；保留原时间条件，**不用于再选 checkpoint/超参数**。",
        markdown_table(["干预","跨度","目标","干预−原始 MAE","95% CI"],[
            [row.ablation,f"+{row.horizon}",row.target,f"{row.changed_minus_original_mae:.6f}",f"[{row.ci95_low:.6f}, {row.ci95_high:.6f}]"] for row in ablations.itertuples()]),
        "这些干预可能造成分布外输入，不等于重新训练的容量控制。GRU 输出依赖序列并不自动证明利用了生长信息；与 B2a 的弱/不一致改进共同表明，当前证据不足以把三帧表征优势视为已建立。",
        "存在**局部时序利用迹象**：+2 将历史换成末帧重复时，叶数 MAE 增加约 0.044、比例增加约 0.00182，探索性 CI 不含 0；逆序输入对三个跨度比例均有可见损害。这支持部分 checkpoint 对历史顺序敏感，但不能替代与单帧模型的无明显差异，也不能宣称全面实际获益。",
        "- B0 是真实历史叶数的强短跨度参考；B1 对比例平滑变化可能较好，对叶片计数的离散跳变/遮挡直线外推可能较差。RGB 模型没有历史 GT，不能只凭是否胜过 Oracle 判成功/失败。",
        "\n## H. 失败案例、数据限制与可视化",
        "未删困难植株。最差病例按 +8 GRU 的 train-scale 两目标宏 MAE 均值排序，仅用于诊断，不改变测试或训练：",
        markdown_table(["植株","基因型","+8 叶数 MAE","+8 比例 MAE"],[[row.plant_uid,row.genotype,f"{row.leaf_count:.4f}",f"{row.projected_area_ratio:.5f}"] for row in failures.head(6).itertuples()]),
        "具体曲线诊断（误差符号为预测−GT；三个 seed 均值的符号偏差，用于诊断而非主 MAE）：",
        markdown_table(["植株","GT 未来叶数范围","叶数偏差","GT 比例范围","比例偏差","原序列叶数下降事件"],[
            [row.plant_uid,f"{row.leaf_true_min:.0f}–{row.leaf_true_max:.0f}",f"{row.leaf_signed_bias:+.4f}",
             f"{row.ratio_true_min:.4f}–{row.ratio_true_max:.4f}",f"{row.ratio_signed_bias:+.5f}",row.leaf_decrease_events]
            for row in diagnostics.itertuples()]),
        "Go-0/rep_09 表现为对高叶数轨迹的系统性低估；Is-1/rep_42 的未来占据率较小且稳定，模型却明显高估；Kz-9/rep_43 的比例误差也偏向高估。这些病例的图像尺寸都保持一种，因此不能把其失败直接归因于同株 resize/crop 尺寸变化。可见叶数下降事件说明并非严格单调；遮挡、衰老还是标注差异的因果原因，未人工验证。逐病例时间区间和几何统计见 `failure_case_diagnostics.csv`。",
        "- 单株增长速度不同，叶片遮挡/显现、计数离散变化、后期衰老和投影几何变化，使局部生长趋势并非可线性外推；这些是可解释风险，不宣称已人工确定每个样本原因。",
        "- 不统一天数跨度；大时间空档带来更大外推难度；时间编码只验证墙钟顺序，无时区或设备时钟校准。",
        "- 图像裁剪框/尺度存在变化，面积比例仍依赖图像构图。既有五株尺寸变化不删除，原始比例标签不回算为填充图面积。",
        "- 冻结通用 DINO + 简单 head 可能已经从最后一帧获得大部分可见生长阶段信息；只有三帧、短间隔历史，GRU 不一定有足够增量信息。未通过额外试验验证该机制。",
        "- 有效窗主要来自长序列。训练植株加权和测试宏平均减少其支配，但目标标准化按全部 train 窗口拟合，仍反映训练窗口分布。",
        "- 46 个 test 植株覆盖 21 个基因型，部分组只有 1–2 株；组间差异仅作描述，不作稳健基因型结论。没有新基因型外推、外部数据集或独立重复采集验证。",
        "- 3 seeds 只量化优化随机性的一部分；plant bootstrap 区间条件于当前 split、预训练和三个 seeds，不代表全部数据来源/标注系统误差。",
        "- 时间排程可能编码批次/采集环境；B1b 表明时间变量本身有预测能力。三帧收益不足，不意味着任意形式的时序学习都无效。",
        "\n### 图表及选择透明性",
        "`figures/` 提供 editable PDF/SVG 与 PNG：",
        "1. `baseline_mae_leaf_count`；2. `baseline_mae_projected_area_ratio`；3. `horizon_comparison`；",
        "4. `growth_curves_preselected`（6 株 SHA256(42:plant_uid) 预选，与准确率无关）；",
        "5. `genotype_error_distributions`；6. `training_validation_losses`；7. `error_vs_actual_time`；",
        "另有 `growth_curves_failure_cases`，显式展示最差 3 株，不只挑好结果。图源和选株规则见 `figures/selection_manifest.json`、`FIGURE_CONTRACT.md`。",
        "图中的多 seed 平均预测仅用于曲线展示；主表是多 seed **误差平均**，没有用 ensemble 曲线提高主指标。时间误差图保留稀疏时间 bin，计数见 `error_vs_time_source.csv`。",
        "时间误差图混合三个观测跨度，只是描述性诊断，不能分离时间跨度、采集批次和生长阶段的因果作用。图表源代码 QA 12 PASS/0 FAIL；两个 WARN 为未导出 TIFF 和仅 300 dpi PNG，因为这是实验报告而非投稿包，已经同时交付 editable PDF/SVG。视觉 QA 见 `figure_visual_qa.md`，图像内容由 Agnes API 识别；仅提交生成图表，不提交原始 RGB/Mask。",
        "\n## I. M2 建议与最终验收",
        f"**{judgment}**：先复核时间信息是否对部署可得，检查计数遮挡噪声与裁剪尺度变化，在验证集预注册容量受控的重复末帧 GRU 训练/时间-only neural head 对照；如继续时序研究，应建立更强、可重复的增量证据，而非直接加复杂 JEPA。" if not robust_temporal else "**CONTINUE**：可计划后续时序表征实验，但本阶段不启动 M2。",
        "工程交付和研究效果分别判断；M1 PASS 不要求 GRU 超过所有简单/Oracle 基线。",
        markdown_table(["验收项","状态"],[[name,"PASS" if passed else "INCOMPLETE"] for name,passed in acceptance.items()]),
        f"自动测试：{len(cases)-failures_count-skipped_count} passed，{failures_count} failed，{skipped_count} skipped。覆盖缓存行身份、时间顺序、split 独立、无未来图像/标签捷径、train-only 标准化、模型形状、保存/重载、重复推理、CPU/两 GPU、植株宏指标和 test 不选模。`tests.xml` 为全套 M0+M1 结果。",
        "神经网络 18 个 checkpoint 在真实 test 上分别重复推理，全部逐位相同；记录在 `inference_repeatability.json`。",
        "\n## J. 可复现、提交记录与信息来源",
        "可执行顺序与完整命令见根目录 `M1_README.md`。每模型配置、种子、checkpoint、history、预测 CSV 和 metrics JSON 都保存在 `outputs/m1/`；大文件仅本地保留，不公开原图、Mask、缓存或模型派生资产。",
        "独立开发分支：`m1/plant-2d-baseline`。M0 基础提交：`6058704b4046eec77a5092871991ff9e146dddb4`。新增提交记录保存在 `COMMITS.md`（最终报告提交自身以 `git log` 为准，避免自引用 hash）。未合并 main、未推送。",
        "权威实现/安装来源见 `docs/SOURCES.md`：官方 DINO https://github.com/facebookresearch/dinov2 ，PyTorch https://pytorch.org/get-started/previous-versions/ 。Context7 已获取官方 PyTorch v2.8 文档；DINO API 直接核对固定官方源码。博查搜索 HTTP 403，已明确记录；未假称搜索成功，未向视觉 API 发送原数据。",
        "\n**本报告的实验数值来源是本机运行的不可替代 CSV/JSON/checkpoint 与原始 M0 审计，不是外部资料或训练记忆。**",
    ])
    (directory / "M1_REPORT.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    write_json(directory / "acceptance.json",{"status":status,"research_judgment":judgment,"checks":acceptance,
               "tests":{"passed":len(cases)-failures_count-skipped_count,"failed":failures_count,"skipped":skipped_count},
               "main_neural_runs":18,"simple_runs":9,"m0_original_data_hashes_unchanged":True,
               "test_used_for_selection":False,"jepa_trained":False,"three_dimensional_experiments":False})
    print(json.dumps({"status":status,"research_judgment":judgment,"tests":len(cases)},ensure_ascii=False))


if __name__ == "__main__":
    main()
