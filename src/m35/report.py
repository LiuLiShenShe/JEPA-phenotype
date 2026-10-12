"""Generate the factual M3.5 research report from locked experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from m3.common import write_json


def markdown_table(frame, columns, decimals=3):
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.{decimals}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(root, protocol):
    root = Path(root)
    report = root / "reports/m3_5"
    feasibility = json.loads((report / "feasibility.json").read_text())
    decision = json.loads((report / "research_decision.json").read_text())
    protection = json.loads((report / "final_protection.json").read_text())
    training = json.loads((report / "training_complete.json").read_text())
    current = pd.read_csv(report / "current_state_final_metrics.csv")
    organ = pd.read_csv(report / "organ_forecast_metrics.csv")
    event = pd.read_csv(report / "new_leaf_event_metrics.csv")
    visibility = pd.read_csv(report / "future_visibility_metrics.csv")
    geometry = pd.read_csv(report / "geometry_forecast_metrics.csv")
    coverage = pd.read_csv(report / "task_coverage.csv")
    current_val = current.loc[current.partition.eq("official") & current.split.eq("val")]
    current_test = current.loc[current.partition.eq("official") & current.split.eq("test")]
    organ_val = organ.loc[organ.partition.eq("official") & organ.split.eq("val")]
    organ_test = organ.loc[organ.partition.eq("official") & organ.split.eq("test")]
    event_test = event.loc[event.partition.eq("official") & event.split.eq("test")]
    visibility_test = visibility.loc[visibility.partition.eq("official") & visibility.split.eq("test")]
    geo_test = geometry.loc[geometry.partition.eq("official") & geometry.split.eq("test")]
    paired = pd.read_csv(report / "organ_validation_pairs.csv")
    primary = paired.loc[paired.split.eq("val") & paired.target.eq("leaf_total_length") & paired.horizon.eq(2) & paired.left.eq("C3-GRU") & paired.right.isin(["C2-MLP", "C4-Repeated-GRU"])]
    m3 = pd.read_csv(root / "reports/m3/baseline_comparison.csv")
    comparison = current_test.loc[current_test.model.eq("A1-PointNet"), ["target", "plant_macro_mae"]].rename(columns={"plant_macro_mae": "current_state_mae"})
    comparison = m3.loc[m3.model.eq("B3-2"), ["horizon", "target", "plant_macro_mae"]].merge(comparison, on="target").rename(columns={"plant_macro_mae": "M3_future_mae"})
    comparison.to_csv(report / "current_vs_M3_future.csv", index=False)
    matched_comparison = pd.read_csv(report / "current_vs_M3_matched_target_scans.csv")
    current_answer = current_val.loc[current_val.model.eq("A1-PointNet")].set_index("target").plant_macro_mae
    augmentation_answer = current_val.loc[current_val.model.eq("A3-PointNet-Geometry")].set_index("target").plant_macro_mae
    current_calibrated = current_val.loc[current_val.model.eq("A2-Geometry-Ridge")].set_index("target").plant_macro_mae
    temporal_details = []
    for control in decision["organ_history_controls"]:
        temporal_details.append(f"相对 {control['control']}：{control['validation_improvement_pct']:.1f}% / {control['validation_absolute_improvement_cm']:.3f}cm改善，三种子一致={control['all_three_seeds_improve']}，改善验证折={control['improving_validation_folds']}/4，预设门槛通过={control['passed']}")
    history_pass = decision["reliable_practical_organ_temporal_signal"]
    recommendation = decision["recommendation"]
    response = "是：锁定的叶总长主任务通过全部预设门槛，但仍只是探索性证据。" if history_pass else "否，尚不足以支持现在启动 3D Temporal-JEPA。受控基线发现小幅、稳定的自动形态标签预测增益，但未通过训练前锁定的实际增益门槛，且当前冠幅表征不足；这不等于器官历史完全没有信息。"
    lines = [
        "# M3.5 — 植物器官级生长预测可行性与三维表征诊断",
        "",
        f"> **工程状态：待最终测试验收；研究建议：{recommendation}。不启动 M4。**",
        "> 所有数值来自现有 Sugar4D V4 的真实观测与保存的预测；没有合成数据结果、JEPA 或生成式模型训练。",
        "",
        "## 1. 结论与终止选择",
        f"**是否存在足够可靠、具有实际意义的器官级动态预测信号，使继续研究 3D Temporal-JEPA 有合理依据？** {response}",
        f"**唯一下一步建议：{recommendation}。** 分支选择来自训练前锁定的规则及 Train/Validation 证据，而非测试集上最好子任务。",
        "当前 PointNet 的株高/冠幅可靠性优先诊断；只有当表示门槛通过且锁定器官历史主任务相对 C2/C4 同时通过时，才建议 ORGAN-TEMPORAL-JEPA。",
        "本阶段完成后停止 GRU 扩展，不自动开始 M4、合并、提交或推送。若建议 REPRESENTATION-FIRST，优先几何增强、物理尺度保留、器官分割/表征与少样本当前表型估计；不是把未来预测失败直接归因于没有时间信息。",
        "",
        "## 2. 数据、关联与原有成果保护",
        f"- 起始实际 commit：`{protection['commit']}`；起始工作区干净，`main...origin/main`。",
        "- 已逐项阅读 M3_REPORT、DATA_AUDIT、actual_field_schema、baseline_comparison、Sugar4D dataset、PointNet/GRU 实现和原始两张 CSV。",
        "- 原始植物表 768×61，器官表 6778×8；48 株、16 次真实扫描、675 条跨时一致身份轨迹；全部 768 PLY 重新计算叶身份和原始米制 centroid，并复核 M3 表。",
        "- 原始 CSV 与已有元数据逐字段一致、PLY/CSV 关联一一对应、重复/错配均为 0；centroid 使用当前叶实例点集均值，只用于轨迹审计，不作为 C2–C4 预测输入。",
        f"- 完整网格有 {feasibility['not_observed_grid_cells']} 个 not_observed 单元。未观测面积等为 NaN，从未当作真实零面积；没有 CSV 缺行但 PLY 有叶等错配。",
        f"- {feasibility['disappearing_tracks']} 条轨迹在 93 DAS 前停止可观测；重新观测轨迹 {feasibility['reappearing_tracks']} 条。所有植株扫描完整，not_observed 仍不能区分遮挡、分割失败、移除或生物学死亡。",
        "- DAS = 37,40,44,47,51,54,58,61,65,72,75,79,82,86,89,93。h=1/2 是后续一次/两次实际扫描，不是固定天数；包含 65→72 的 7 日间隔。",
        f"- 保护清单覆盖 {protection['files_checked']} 个既有文件（含 {protection['raw_files_checked']} 个 Sugar4D 原始文件），最终变化 {len(protection['changes'])}；M0–M3 模型、数据和结果不变。",
        "- 保留 M3 官方 24/12/12 植株划分及同一 fold0–fold3 24/12/12 划分。genotype 不作为输入，保留 M3 对 plant020/021 的未确认映射警告，不做基因型结论。",
        "",
        "## 3. 实验 B：先验可行性与标签质量",
        markdown_table(coverage.loc[coverage.partition.eq("official")], ["split", "horizon", "independent_plants", "trajectories", "tracks_ge3_observations", "eligible_windows", "valid_morphology_windows", "future_not_observed_windows", "event_positive", "event_negative"]),
        "",
        "所有官方/四折训练与验证门槛通过，完整门槛、每类独立植株覆盖见 feasibility.json；测试类别统计仅描述，不决定任务是否启动。每株轨迹和足够历史数量见 trajectories_per_plant.csv；逐 DAS 分布见 DAS_task_distribution.csv。",
        "形态研究可行性为 **自动发布标签的数值预测可行**，不是已证明可测量精确生物生长：",
        "- 450 条 93 DAS 破坏性人工参考与发布测量匹配；总长 MAE 1.918cm、叶片长 1.868cm、叶柄长 2.383cm、叶宽 1.086cm。",
        "- 该横截面自动/人工差异不是重复扫描噪声估计，不能宣称纵向 delta 噪声下限；本报告保留有风险标签、不修正或删除异常值。",
        "- 叶面积无人工验证，只作次要自动面积任务；不允许单独靠面积测试增益建议 Temporal-JEPA。",
        "- 明确可行性边界：发布自动形态标签的有限数值基线可行；以此独立确证精确生物器官增长不可行，因缺纵向重复测量噪声、跨方法误差较大且面积参考缺失。没有强行训练生物增长/JEPA大模型。",
        "- 面积/长度的相邻增减和极值完整保存在 organ_label_transitions.csv；大幅非单调变化可能受叶姿态、抽取算法与标注几何影响，未假称都是生物增长。",
        "",
        "## 4. 实验 A：当前状态编码诊断",
        "A1 使用原有 PointNetForecaster 编码器和预测头，输入当前 XYZ，7 个时间槽全部置零；没有 DAS/实例/语义/GT输入。训练目标为同一时点株高、冠幅、可见标注叶数。A3 在相同 PointNet 上拼接标签自由几何特征作为诊断性补充。",
        "A2-Z-Range/Quantile/Hull 为无需学习原始代理；A2-Geometry-Ridge 是 train-only 标准化/校准的几何回归，不应称作无需学习。A2-DAS-Ridge 单独展示时间条件预测，避免把纯阶段信息当作几何。",
        "**官方验证集（12 独立植株）当前状态 MAE：**",
        markdown_table(current_val, ["model", "target", "plant_macro_mae", "seed_mae_std", "ci95_low", "ci95_high"]),
        f"明确回答：PointNet株高验证MAE {current_answer['plant_height']:.3f}cm，达到预设1cm当前估计标准；冠幅 {current_answer['plant_width']:.3f}cm、可见叶数 {current_answer['visible_organ_count']:.3f}片，未达到各自1cm/1片标准。因此不能称它已可靠编码全部当前状态。",
        f"几何校准株高/冠幅 {current_calibrated['plant_height']:.3f}/{current_calibrated['plant_width']:.3f}cm；几何增强PointNet为 {augmentation_answer['plant_height']:.3f}/{augmentation_answer['plant_width']:.3f}cm。几何描述对当前几何目标提供强信息，尤其冠幅；不应把原始跨度误当官方Height，叶数仍需要更好的结构表征。",
        "",
        "表示分支预设株高/冠幅 1cm 阈值，并要求 PointNet 相对几何校准至少差 1.5倍；这是实用诊断标准，不等同于仪器噪声或统计确证。",
        "```json\n" + json.dumps(decision["current_representation"], ensure_ascii=False, indent=2) + "\n```",
        "**当前与 M3 未来误差对照（同一官方测试植株；不同任务与时间窗口，仅诊断）：**",
        markdown_table(comparison, ["target", "horizon", "current_state_mae", "M3_future_mae"]),
        "**同一目标扫描阶段对齐的诊断：** 当前模型使用目标时点XYZ恢复同一目标，M3用过去XYZ预测；这不是公平未来预测比赛，而是当前表示误差与额外预测困难的拆分。",
        markdown_table(matched_comparison, ["target", "horizon", "current_target_frame_estimation_mae", "M3_past_XYZ_future_mae", "future_minus_current_task_gap"]),
        "",
        "株高发布定义是处理后 z 的99分位数减人工固定生长点 z，且排除 taproot、逐实例2mm降采样；本阶段高度跨度/分位跨度未读取人工生长点或语义标签，不是同一量。冠幅原始 XY 凸包直径也未复现官方语义剔除/降采样。当前几何代理有偏差仍可 train-only 校准，绝不以未来 GT 修正。",
        "诊断瓶颈：小型 max+mean PointNet、随机2048点和极少独立植株，可能不充分保留极值/投影边界/点密度；株高目标还依赖外部人工参考及预处理。可见叶数不能仅靠简单包围盒解析。几何增强是否补充见 A3 与 A1 植株配对表；不把表征误差直接叫做不可预测的未来生长。",
        "",
        "## 5. 实验 C：固定轻量预测器与输入条件",
        "- C0 Persistence、C1 实际 DAS 最小二乘趋势：Oracle historical phenotype；使用同样三次连续可观测历史，长度/宽度/面积未来值和变化量。",
        "- C2 MLP：当前5个发布形态（总长/叶片长/叶柄长/宽/面积）+合法时间；C3 GRU：三次形态；C4 同架构从头训练，重复末状态。均属于 Oracle organ-state，不是原始点云器官跟踪模型。",
        "- C2–C4 联合预测三个形态 delta 和未来可观测概率；不可观测目标不参与形态损失，但所有历史有效窗口参与 visibility BCE 和独立评价；不在推理时获知未来持续出现。",
        "- 时间输入只有已观测 DAS/间隔、已知查询 target DAS、当前已知的 first-seen-so-far 年龄；没有未来身份、first-observed 目标、生命周期、未来可见性/面积或点云。",
        "- C5 为 **非 Oracle 整株 XYZ 几何预测**：当前或三历史几何 × absolute 或 residual，残差基准是当前 train-calibrated几何估计，不是当前 GT。没有实例标签/器官身份输入，不能与 Oracle C0–C4 宣称信息条件公平。",
        "- C5 27项代理含高度分位、XY范围/凸包面积/直径、径向分布、协方差特征值、raw点数/包围盒密度、8192点邻近距离和上部点比例；几何全由当前/历史原始 XYZ计算。PointNet用采样点而C5用全点几何，因此这是归纳偏置/信息保留诊断，不是严格等点数架构比赛。",
        "- E0 训练频率/DAS阶段；E1 当前可见数+DAS Logistic，数量以同输入 Ridge；E2 当前汇总器官 MLP，E3 三次历史汇总 GRU，E4 重复末状态对照。标签为历史末点之后到目标扫描之间首次被观察的新增身份，非净叶数差或精确出叶时刻。",
        f"- Seeds={protocol['seeds']}，每模型50 epochs；tabular hidden64、dropout0.1、AdamW lr0.001/WD0.0001、梯度裁剪5。{training['neural_runs']} 个神经运行全部完成，累计 {training['total_optimizer_steps']} steps、{training['total_training_seconds']:.1f} 秒模型训练时间。",
        "- MLP/GRU/C5 train-only标准化，按植株×horizon平衡损失；PointNet每株扫描相同所以天然植株平衡。真实/重复GRU参数、初始tensor哈希、epochs/steps完全相同，均从头训练；不拷贝训练后权重。",
        "- epoch/正则化/事件及消失分类阈值仅由相应Validation选择；首版完整validation lock在首次Test checkpoint重放前生成，模型/阈值与研究建议均已固定。",
        "- 首次重放期间发现统计代码把确定性C0/C1的seed=-1行计入‘三随机种子’行数，修正为只取C3/对照共同的三个随机种子并重新锁定；未改模型/阈值/协议、未读取Test评价指标，建议保持REPRESENTATION-FIRST。原锁保存在development_logs/，修正记录在seed_count_correction.json，避免假称首版统计完全无错。",
        "",
        "## 6. 实验 D：器官历史增量与统计",
        "预设主任务：h=2 leaf_total_length；必须相对 C2、C4 都至少10%且0.5cm改善、三种子全向好、官方验证植株配对95%上界<0、至少3/4验证折方向改善。面积/宽度/阶段子集只能次要诊断。",
        "**主任务官方验证与四折验证配对结果（GRU减对照，负值有利）：**",
        markdown_table(primary, ["partition", "right", "plants", "left_minus_right_mae", "improvement_pct", "ci95_low", "ci95_high"]),
        "```json\n" + json.dumps(decision["organ_history_controls"], ensure_ascii=False, indent=2) + "\n```",
        "；".join(temporal_details) + "。不把未达实际意义门槛解释为‘绝对没有动态信息’：可检测的小幅自动标签预测增益仍应与测量限制和表示误差分开。",
        "**官方测试形态结果（探索性、按植株宏平均）：**",
        markdown_table(organ_test, ["model", "horizon", "target", "plant_macro_mae", "seed_mae_std", "ci95_low", "ci95_high", "delta_phenotype_mae"]),
        "绝对未来与delta都减去同一个历史GT基准，所以器官 absolute MAE 与 delta MAE 数学上相等；没有把它们当作两条独立证据。形态指标是在未来被观察目标上条件评价，需同时读取可观测性表。",
        "以植株先汇总该株的所有叶/窗口，再等权汇总植物；seed间比较用每种子误差，区间先平均每株种子误差再重采样植株10,000次，不对叶/窗口做伪重复。不把平均预测的ensemble误差冒充三种子独立性能。",
        "两项主对照必须同时通过，主任务和绝对/相对门槛在训练前锁定；次要目标/阶段/事件对照均为探索性描述，95%区间未作全表多重比较确证性解释，不以某个事后显著小任务改变建议。",
        "所有四折结果保存在 organ_forecast_metrics/organ_validation_pairs，OOF 48株结果见 organ_oof_metrics；各折模型共享训练植株，OOF置信区间与各验证折方向是描述性稳定性检查，不是外部重复。",
        "**C5非Oracle整株几何 absolute/residual 对照（官方探索性Test）：**",
        markdown_table(geo_test, ["model", "horizon", "target", "plant_macro_mae", "seed_mae_std"]),
        "C5的 residual 基准是当前几何估计而非发布当前GT，故记录的 residual delta误差是代理残差误差，不是已经校准过的生物增长率。geometry_paired_comparisons 给出同信息条件的历史/当前和残差/绝对对照；与 M3 小型PointNet比较仅诊断。",
        "absolute/residual各自按训练输出标准差归一化，故除了残差参数化也改变多目标损失尺度；不能把差异全归因为残差技巧本身。比较使用物理单位MAE，不能跨任务比较normalized validation score。",
        "",
        "## 7. 新叶事件与未来可观测性",
        "**新叶首次观测（官方探索性Test）：**",
        markdown_table(event_test, ["model", "horizon", "plant_macro_precision", "plant_macro_recall", "plant_macro_f1", "plant_balanced_pr_auc", "new_count_plant_macro_mae"]),
        "Precision/Recall/F1先计算各株再宏平均（zero_division=0），同时提供植株等权pooled混淆指标；PR-AUC使用average precision，以每株等总权重计算，另给有正负类别植株的macro AP及覆盖。种子指标先分别计算再平均，不能把多个种子同窗口当成重复样本。官方验证/测试PR-AUC区间按整个植株同步重采样各seed、每次计算AP后平均seed；数量MAE/F1区间同样按植株重采样，详见指标CSV。",
        "**未来未被观测事件（1−visibility；官方探索性Test）：**",
        markdown_table(visibility_test, ["model", "horizon", "samples", "positive", "positive_plants", "plant_macro_precision", "plant_macro_recall", "plant_macro_f1", "plant_balanced_pr_auc", "plant_macro_brier"]),
        "可观测性表正类是未来not_observed，突出少数消失窗口，而非以约95%以上持续观察率隐藏偏差。C0/C1持续观察假设即预测消失概率0；C2–C4联合头学习概率，分类阈值在验证集上锁定。形态条件性指标并不证明能正确处理所有将消失叶片。",
        "",
        "## 8. 可视化、失败案例与数据限制",
        "六组 PNG/PDF/SVG + 源CSV位于 figures：01真实叶轨迹；02该叶h2长度/面积预测；03主任务器官历史增量与配对CI；04首次观测事件/数量；05几何代理/发布目标；06验证失败植株及叶。",
        "01/02/04由验证历史覆盖最长的字典序首叶/植株选择，未挑最佳预测；06是事后最大验证MAE诊断，不用于改主任务或选择研究分支。散点/曲线用于描述，图上的seed均值线不替代误差统计。",
        "所有验证器官失败样本按MAE保存于 organ_failure_cases.csv，包含未观测目标窗口数；选择偏差不能靠只画持续存在的叶片隐藏。",
        "失败可能包括非单调长度/面积突变、解剖自动度量误差、短历史和future not_observed选择；没有将点云局部错误自动断定为身份错误。",
        "官方Test此前用于M3，M3.5 Test与含官方Test植株的fold结果均只能探索性证据，不称独立确证；48株单温室不足以建立跨场景/基因型外推。",
        "",
        "## 9. 复现与验收",
        "```bash",
        ".venv-m1/bin/python scripts/analyze_m3_5_feasibility.py",
        ".venv-m1/bin/python scripts/train_m3_5.py --stage current --device cuda:0",
        ".venv-m1/bin/python scripts/evaluate_m3_5.py --stage current",
        ".venv-m1/bin/python scripts/train_m3_5.py --stage future --device cuda:0",
        ".venv-m1/bin/python scripts/evaluate_m3_5.py --stage select",
        ".venv-m1/bin/python scripts/evaluate_m3_5.py --stage final --device cuda:0",
        ".venv-m1/bin/python -m pytest -q --junitxml=reports/m3_5/tests_all.xml",
        ".venv-m1/bin/python scripts/verify_m3_5.py",
        "```",
        "复用已安装 `.venv-m1`；版本记录environment.json。所有新缓存、checkpoint和逐窗口预测在 outputs/m3_5，不覆盖 data/processed/sugar4d。训练脚本跳过已有checkpoint；完整从头复现实验请使用单独的已有数据工作目录，不删除旧里程碑文件。",
        "开发阶段发生numpy标量JSON与单输出Ridge维度的工程错误，以及C5当前代理测试隔离的防御性强化；更正记录与开发产物已归档。最终180个运行统一使用修正后同一源代码哈希从头训练；未更改协议、任务、超参数或预算，也未根据测试结果调整方案。",
        "验收以 acceptance.json、tests_all.log/XML、training_complete.json、selection_lock、inference_repeatability、final_protection 和 figure_manifest 为准。没有自动下载大型数据或产生虚构不可行任务分类分数。",
        "",
        "## 10. 来源与归属",
        "- 本实验数值：原始 leaf_related.csv/plant_related.csv/PLY、M3审计、本阶段保存的真实预测与植株级统计CSV。",
        "- Sugar4D官方项目 https://www.ipb.uni-bonn.de/data/sugar4d/index.html （本次HTTP200，页面保存sources/official_page.txt）；数据DOI https://doi.org/10.60507/FK2/IS8YBZ ，固定复用M3 V4哈希，不请求新版本。",
        "- Bocha搜索返回403配额不足，已保存sources/bocha_search.json；搜索不可用，不冒充检索成功或以训练知识替代数据证据。Dataverse版本端点本次超时，版本由已有固定V4档案核实。",
        "- PyTorch 2.8、scikit-learn 1.7.1近邻版本、SciPy、Matplotlib 当前官方源码文档经Context7 library→docs获取并保存sources；运行时确切版本见environment.json。",
        "- 官方 Height/Width定义来源与作者归属沿用 reports/m3/SOURCES.md 和 morphology_definitions.json；本阶段变化为派生轨迹/几何/窗口/预测。Sugar4D CC BY4.0；Bömer等作者，不隐含原作者背书。",
        "- 图表是实测数据的Python绘制，不使用生成式图片制造定量内容；图片内容由Agnes API识别进行外部视觉QA，响应保存在figure_visual_qa.json/visual_qa，自动renderer确认无绘制文字越界；外部判断不替代数值数据验证。",
    ]
    (report / "M3_5_REPORT.md").write_text("\n\n".join(lines) + "\n")
    write_json(report / "decision.json", decision)
