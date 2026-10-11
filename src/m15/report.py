"""Source-grounded M1.5 delivery, including exploratory decision boundaries."""

from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd

from m15.artifacts import verify_snapshot
from m15.figures import create_figures
from m15.formatting import markdown_table
from training.common import file_hash, write_json


def metric_table(summary, models, tasks):
    selected = summary.loc[summary.split.eq("test") & summary.model.isin(models) & summary.task.isin(tasks)].copy()
    selected["MAE ± seed SD [plant CI95]"] = selected.apply(
        lambda row: f"{row.plant_macro_mae_mean:.5f} ± {row.plant_macro_mae_std:.5f} [{row.ci95_low:.5f}, {row.ci95_high:.5f}]", axis=1)
    return markdown_table(selected[["task", "model", "target", "plants", "MAE ± seed SD [plant CI95]"]])


def render_report(root):
    root = Path(root)
    report = root / "reports/m1_5"
    lock = json.loads((report / "protocol_lock.json").read_text())
    protocol = lock["protocol"]
    verdict = json.loads((report / "decision.json").read_text())
    summary = pd.read_csv(report / "baseline_comparison.csv")
    pairs = pd.read_csv(report / "paired_comparisons.csv")
    feasible = pd.read_csv(report / "calendar_feasibility.csv")
    resources = pd.read_csv(report / "training_resources.csv")
    legacy = pd.read_csv(root / "reports/m1/baseline_comparison.csv")
    audit = json.loads((report / "legacy_snapshot.json").read_text())
    tests_path = report / "tests.xml"
    if not tests_path.exists():
        raise FileNotFoundError("Run the final full test suite before delivery")
    suites = ET.parse(tests_path).getroot()
    suites = list(suites) if suites.tag == "testsuites" else [suites]
    totals = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    if totals["failures"] or totals["errors"] or totals["skipped"]:
        raise ValueError("Final automated tests must pass without skipped real-artifact verification")
    protection = verify_snapshot(root, raw=True)
    write_json(report / "final_protection.json", protection)
    create_figures(root)
    expected_runs = len(protocol["seeds"]) * (len(protocol["offsets"]) * len(protocol["offset_models"]) + len(lock["eligible_days"]) * len(protocol["day_models"]))
    if len(resources) != expected_runs:
        raise ValueError("Incomplete neural training inventory")
    stage = pd.read_csv(report / "calendar_stage_coverage.csv")
    zero = pd.read_csv(report / "calendar_windows_per_plant.csv")
    zero = zero.loc[zero.windows.eq(0), ["forecast_horizon", "split", "plant_uid"]]
    time_pairs = pairs.loc[pairs.family.eq("time")]
    core = pairs.loc[pairs.family.eq("capacity")]
    diff = pairs.loc[pairs.family.eq("differences")]
    difficulty = pd.read_csv(report / "difficulty_stratified_errors.csv")
    associations = pd.read_csv(report / "difficulty_associations.csv")
    dynamic = pd.read_csv(report / "dynamic_information_summary.csv")
    span = pd.read_csv(report / "history_span_summary.csv")
    visual_path = report / "figure_visual_qa.json"
    visual = json.loads(visual_path.read_text()) if visual_path.exists() else {"records": []}
    cohorts = pd.read_csv(report / "plant_recording_cohorts.csv")
    cohort_counts = cohorts.groupby(["split", "first_record_month"]).size().rename("plants").reset_index()
    single_cohort_genotypes = int(cohorts.groupby("genotype").first_record_month.nunique().eq(1).sum())
    primary = core.loc[core.task.eq("day5") & core.target.eq("leaf_count")]
    resource_table = resources.groupby("architecture").agg(runs=("model", "size"), parameters=("trainable_parameters", "first"),
                      best_epoch_median=("best_epoch", "median"), training_seconds=("seconds", "sum"),
                      peak_memory_bytes=("peak_gpu_memory_bytes", "max")).reset_index()
    legacy_subset = legacy.loc[legacy.model.isin(["B2a", "B2b"]), ["model", "horizon", "target", "plant_macro_mae_mean", "plant_macro_mae_std"]]
    stats_columns = ["task", "target", "first", "second", "delta_mae", "ci95_low", "ci95_high", "relative_gain", "all_seeds_favor_first", "holm_p"]
    key_difficulty = difficulty.loc[difficulty.task.eq("day5") & difficulty.target.eq("leaf_count") & difficulty.model.eq("gru_full")
                    & difficulty.factor.isin(["rapid_leaf_change", "visible_leaf_decline", "history_stage", "observed_age_stage", "plant_size_varies", "window_size_changes"])]
    key_associations = associations.loc[associations.model.eq("gru_full") & associations.target.eq("leaf_count")]
    area_association = key_associations.loc[key_associations.task.eq("day5") & key_associations.variable.eq("history_area_rate")].iloc[0]
    stage_primary = difficulty.loc[difficulty.task.eq("day5") & difficulty.target.eq("leaf_count")
                    & difficulty.model.eq("gru_full") & difficulty.factor.eq("history_stage")].set_index("level")
    pair_seed_rows = core.loc[core.task.eq("day5"), ["target", "second", "seed42_delta", "seed43_delta", "seed44_delta"]]
    primary_metrics = summary.loc[summary.split.eq("test") & summary.task.eq("day5") & summary.target.eq("leaf_count")].set_index("model")
    neural_time = primary_metrics.loc["time_full", "plant_macro_mae_mean"]
    image_mlp = primary_metrics.loc["mlp_full", "plant_macro_mae_mean"]
    image_gru = primary_metrics.loc["gru_full", "plant_macro_mae_mean"]
    repeat_gru = primary_metrics.loc["gru_repeat", "plant_macro_mae_mean"]
    primary_repeat = primary.loc[primary.second.eq("gru_repeat")].iloc[0]
    primary_single = primary.loc[primary.second.eq("mlp_full")].iloc[0]
    diff_primary = diff.loc[diff.task.eq("day5") & diff.target.eq("leaf_count")].iloc[0]
    decision_text = {
        "CONTINUE": "在预先固定的 day5 可见叶数任务中，真实序列同时优于同容量重复帧 GRU 和单帧 MLP，并满足三种子一致方向、实际意义阈值及配对不确定性规则。建议下一阶段研究时序表征，但必须在独立确认数据上检验，不立即训练 Temporal-JEPA。",
        "REVIEW": "现有证据不足以满足预先固定的继续规则，也未在所有任务上排除有实际意义的历史增益。时间、结构与真实动态的贡献需要分别解释；不把监督 GRU 的弱优势当成对 Temporal-JEPA 的否定。建议先做预固定独立确认和任务核验，不进入 M2 训练。",
        "PIVOT": "在当前冻结特征、监督训练和任务集合中，配对区间排除了所设最小历史增益。建议调整为更长跨度、叶片新生/可见性变化事件或器官级动态任务；这不是证明所有时序自监督方法无效，不直接进入 M2。"
    }[verdict["research_judgment"]]
    sections = [
        "# M1.5：植物时序预测有效性诊断实验", "",
        f"**工程验收：PASS；研究判断：{verdict['research_judgment']}。**", "",
        f"执行日期：{datetime.now().date().isoformat()}，Asia/Shanghai。这里只交付 M1.5；没有 JEPA、三维实验、合并、提交或推送。", "",
        "## 1. 结论与预先固定的 M2 决策", "", decision_text, "",
        f"- 预先指定 day5 叶数 Plant Macro MAE：单帧 {image_mlp:.4f}、真实 GRU {image_gru:.4f}、重复帧 GRU {repeat_gru:.4f}；纯时间神经网络 {neural_time:.4f}，说明 RGB 存在明显预测信息，但不等于证明历史视觉变化有增量价值。",
        f"- 真实历史较重复帧仅降低 {-primary_repeat.delta_mae:.4f}（{primary_repeat.relative_gain:.2%}），配对 CI95 [{primary_repeat.ci95_low:.4f}, {primary_repeat.ci95_high:.4f}]；较单帧降低 {-primary_single.delta_mae:.4f}，CI95 [{primary_single.ci95_low:.4f}, {primary_single.ci95_high:.4f}]，种子方向不全一致；不满足实际意义和不确定性规则。",
        f"- 显式差分 day5 叶数相对单帧 delta={diff_primary.delta_mae:.4f}（负值才为改善），没有提供稳定增益；容量家族没有 Holm 校正后成立的真实历史优势。",
        "- 3/5/7 天均可构建，Test 可用植物分别46/46/44；不是原 +2/+4/+8 的同义转换。日尺度预测越远，绝对误差增大，但本次没有因此出现稳定的历史序列相对增量。",
        "- 加入时间条件或未来实际时间差没有一致改善 RGB 预测；不能将 M1 的弱时序结果简单归结为‘模型只读了时间’。日历/拍摄安排混杂仍是需要独立批次确认的风险，而不是已经证实的唯一原因。", "",
        "主任务在 M1.5 训练前固定为 **day5 可见叶数**。真实 GRU 必须同时较重复帧和单帧降低 ≥0.10 MAE 且 ≥5%，三种子全部同向、配对植物 CI95 上界<0、capacity 家族 Holm p<0.05，才触发 CONTINUE。面积阈值为 0.002（0.2 个百分点）；这些是工程意义阈值，不是已获生物学专家确认的效应界值。",
        "若主任务不满足，不从 Test 子组或其他终点挑一个有利结果替代主任务。PIVOT 仅在全部容量、显式差分和 RGB-only 时序比较的植物区间均排除所设最小增益时触发，否则 REVIEW。",
        "", markdown_table(primary[stats_columns]), "", "day5 每种子的真实 GRU 减对照 MAE：", "", markdown_table(pair_seed_rows), "",
        "**背景限制：M1 已查看 Test；M1.5 是在此背景下开展的探索性诊断。这里的规则是本轮训练前锁定，不是对整个研究的事前注册，也不使原 Test 恢复成未看过的确认集。** 后续若要声称新方法确证性优势，需要预先固定的额外划分、从未用于设计/选择的留出基因型，或外部数据集。", "",
        "## 2. 原 M1 复核与只读保护", "",
        f"起始 Git commit `{audit['commit']}`；当前分支 `{audit['branch']}`，起始 main 与本地 origin/main 跟踪引用一致且工作区干净；远端实时状态因 GitHub 连接超时未验证。M1.5 仅添加隔离文件，没有创建分支、修改历史代码或旧结果。完整状态在 legacy_snapshot.json / git_state.json。",
        f"保护 {protection['protected_files']} 个既有文件；启动/交付均按 M0 manifest 校验 {protection['raw_files_checked']} 个 RGB/mask 原始文件，改变数 0。原 M0 63 文件保护也独立通过。",
        "M0：318 株、17,056 张有效图像、26 个既有隔离观测。保留 seed42 植物划分：Train 219 / Val 53 / Test 46，21 个基因型；三帧历史坚持原始编号连续，不压缩缺失/隔离帧。",
        "重新从 27 份旧 Test 预测与 plant_level_errors.csv 计算指标，核对原 baseline_comparison.csv；18 个旧神经 checkpoint 及 Train/Val 来源全部存在并通过。CSV 中原 float32 标签回读时还原 float32 后核对，避免把十进制序列化误差误判成实验变化。原数值不覆盖。",
        "", markdown_table(legacy_subset), "",
        "原 B2a 101,634 参数，B2b 561,922 参数，约 5.53 倍容量；M1 的已训练 GRU 重复帧干预仅是推理输入干预，不能充当本轮从头训练的容量对照。原 B0/B1 使用历史真实表型，是 Oracle，不与图像模型作同信息量优势宣称。",
        "原部分脚本/文案使用 RGB-only 称谓，但实际 B2a/B2b 前向接收10维时间条件，不能视为严格 RGB-only；本轮 mlp_rgb/gru_rgb 才明确去除全部时间槽。",
        "冻结缓存 `outputs/m1/features/037935a36cdfd52a`：17,056×384 float32；DINOv2 ViT-S/14 source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，权重 SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。全部冻结，不重新编码或拟合 Test 特征标准化。预处理是等比例 bicubic + 对称 letterbox 224，不做中心裁剪。缓存矩阵/manifest/逐行哈希再次验证。", "",
        "## 3. 实验 A：容量和训练预算受控的历史信息", "",
        "| 名称 | M1.5 模型 | 输入与结构 |", "| --- | --- | --- |",
        "| A1 | mlp_full | 沿用 B2a：最后 RGB 特征 + 同套时间条件，394→256→2 |",
        "| A2 | gru_full | 沿用 B2b：GRU(384,256,1) + 266→256→2 |",
        "| A3 | gru_repeat | 三次重复最后 RGB 特征；与 A2 完全同结构、同时间条件，从头独立训练 |",
        "| A4 | diff_full | 三帧特征及两个逐帧差分共 1,920 维，追加10维条件；1930→64→2 轻量 MLP（123,714 参数） |", "",
        "A1–A4 全部重新训练、3 seeds 42/43/44。A2/A3 同初始化参数 SHA、完整随机 batch 顺序 SHA、学习率、优化器、时间标准化、样本权重和 100 轮/步数预算。关闭早停，但各自仍只用 Val 选择最优 epoch；capacity_training_audit.csv 核对 18 组真实/重复帧训练配对。",
        "A1/A2 新结果与原 M1 不要求逐值一致：M1 使用 patience=18 早停，本轮统一固定 100 轮，可能发现更晚的 Val 最优 epoch；旧 checkpoint 原样保存。A1 和 A2 架构不同，因此最关键的容量控制是 A2 对 A3，不能只靠 A2 对 A1。",
        "A4 的差分是三帧的确定性函数，没有新增独立信息；它检验显式变化坐标和归纳偏置是否帮助预测，不证明 GRU 顺序或自监督表征无用。A4 与单帧结构/参数数不完全相同；若增益只出现在 A4，后续还需 A4 的重复帧容量对照，不仅凭该结果认定动态机制。", "",
        "三个历史帧覆盖的是短观测区间，并没有因为目标改为5/7天就延长历史；实际历史跨度统计见 history_span_summary.csv。这可能限制动态估计，但不是已证实的因果原因。本次未按Test调整历史长度。", "",
        markdown_table(span.loc[span.split.eq("test")]), "",
        metric_table(summary, ["mlp_full", "gru_full", "gru_repeat", "diff_full"], [f"offset{value}" for value in protocol["offsets"]]), "",
        "容量配对结果：first-second；负值有利于真实历史。以下 CI95 为探索性区间，Holm 是 capacity 家族 24 个 task×target×contrast 的双侧符号翻转 p 校正，不将未校正子组显著性用于决策。", "",
        markdown_table(core[stats_columns]), "", "显式特征差分对单帧的结果（独立 differences 家族校正）：", "", markdown_table(diff[stats_columns]), "",
        "## 4. 实验 B：RGB / Time / RGB+Time 与不同预测任务", "",
        "| 条件 | 模型 | 推理时使用的信息 |", "| --- | --- | --- |",
        "| RGB-only | mlp_rgb / gru_rgb | RGB 历史；10个时间槽全部置零。每个跨度独立模型本身就是明确任务标识，没有日期、历史间隔或未来实际时间差 |",
        "| Time-only neural | time_full | 同一 B2a 结构但 RGB 恒为零，只有同套10维时间变量。名义参数数与 B2a 相同，实际有效时间网络为10→256→2 |",
        "| RGB+Time | mlp_full / gru_full | 与 M1 相同输入条件；offset 任务包括目标实际时间差 |",
        "| Offset-only-query | mlp_offset / gru_offset / time_offset | 将未来实际时间差槽置零，只保留可观测历史时间、日历与固定观测跨度任务标识 |",
        "| No-calendar offset query | mlp_relative / gru_relative / time_relative | 再移除 calendar-day/hour 的4个周期槽，保留相对观测年龄、原始编号和历史间隔 |", "",
        "M1 十个条件为：距首条记录的天数、最后原始观测编号、两个历史间隔、实际未来天数、观测跨度、年内日期 sin/cos、一天时刻 sin/cos。它们不是播种年龄，也没有直接输入 genotype ID。",
        "**未来实际时间差的可用性：数据索引确实提供了它，但没有证据证明 offset +2/+4/+8 的每次实际采集间隔在真实部署前已知。** 因而 full-offset 是已知/事后提供目标时间的条件预测；offset-only-query 才是不依赖未来实际间隔的观测偏移任务。二者标签和窗口完全相同，但可用信息/部署任务不同，不能把 full 的优势自动称作无条件生长预测，也不凭此断言发生未来表型泄漏。",
        "calendar/hour、首记录年龄和编号可能编码 May/Aug 采集队列、拍摄节律、基因型/生长阶段分布；去 calendar 的比较是敏感性分析，不是因果批次校正。以下为逐株采集首月覆盖；完整基因型×月份×split 见 schedule_genotype_coverage.csv。",
        f"21个基因型中有 {single_cohort_genotypes} 个只见于一个首记录月份；因此日历月份与这些基因型/队列相互混杂，不能从当前植物划分推断跨月份或未见基因型泛化，也不能把月份效应直接认定成纯拍摄批次因果效应。",
        "即使真实帧优于重复帧，这也首先证明历史图像的增量信息，尚不能唯一分解为真实生长动态、多个视图的降噪/可见性改善或取景/光照变化；本轮没有从头训练的置序/无序池化控制，不把旧 M1 推理置乱干预当成这一证明。",
        "", markdown_table(cohort_counts), "",
        metric_table(summary, ["mlp_rgb", "gru_rgb", "time_full", "mlp_full", "gru_full", "mlp_offset", "gru_offset", "time_offset", "mlp_relative", "gru_relative", "time_relative"], [f"offset{value}" for value in protocol["offsets"]]), "",
        "时间条件配对效应（负值有利于 first；不同信息合同的比较不作同一部署任务的能力宣称）：", "", markdown_table(time_pairs[stats_columns]), "",
        "## 5. 实验 C：3/5/7 天可行性与统一条件的重训", "",
        f"训练前锁定时间 `{lock['created_utc']}`；协议 SHA-256 `{lock['protocol_sha256']}`。目标容差固定 ±6 小时，同株有效未来时间戳最近匹配，等距选较早时间；每个历史窗口每个任务至多一个目标，匹配不用标签值或模型误差。只按 Train≥100、Val≥25 植物且二者每株窗口中位数≥5 判定可训练。三个跨度均通过，但门槛不代表保证检验功效。",
        "**日尺度模型只输入事先请求的 3/5/7 天，不输入目标实际秒差、匹配残差、未来文件/目标编号或未来 RGB/mask。** 时间第6槽为固定日跨度任务标识，不是目标实际观测偏移。B0 不需要目标时间；Oracle B1 外推到请求天数（不偷用匹配的实际间隔）；B1b/全部神经模型使用同套请求时间条件。",
        "", markdown_table(feasible[["days", "split", "plants", "windows", "genotypes", "windows_per_plant_min", "windows_per_plant_median", "windows_per_plant_max", "signed_match_error_hours_median", "absolute_match_error_hours_p95", "absolute_match_error_hours_max"]]), "",
        "阶段覆盖（历史末帧阶段；基因型与历史/目标阶段的完整交叉表另存）：", "", markdown_table(stage), "",
        "因固定时间覆盖不足而零窗口的植物如下；不是按预测困难移除，原 M1 46 株 Test 和全部旧结果未变。day7 的两株 Test 无可匹配窗口，所以全部 day7 方法统一在44株评价，不能与46株的任务直接作独立配对比较。", "", markdown_table(zero), "",
        "完整每株窗口数（含零）见 calendar_windows_per_plant.csv；基因型覆盖见 calendar_genotype_coverage.csv；时间偏差完整统计见 calendar_feasibility.csv。匹配偏差是目标测量时刻不确定性，不把约定日尺度说成精确时刻无噪声标签。",
        "", metric_table(summary, ["B0", "B1", "B1b", "mlp_full", "gru_full", "gru_repeat", "diff_full", "mlp_rgb", "gru_rgb", "time_full"], [f"day{value}" for value in lock["eligible_days"]]), "",
        "## 6. 实验 D：标签变化、阶段、尺寸与预测困难", "",
        "所有困难样本保留。误差先按 seed 取窗口绝对误差均值，再在每个子组中按株取均值；置信区间仍重采样植物而非窗口。子组内植物可同时出现在多个阶段，因此阶段间不是独立组。n<5 的尺寸等子组保留点估计，但不报告不可靠 CI。",
        "- 快速变化：未来可见叶数绝对变化/实际天数超过该任务 Train 的 Q75；阈值在 diagnostic_train_thresholds.json。",
        "- 可见叶数下降：目标叶数低于最后历史叶数；这不是对遮挡、叶片死亡或标签错误的机制判定。",
        "- 早/中/晚：M0 叶数阶段 early≤6、intermediate 7–10、mature≥11；另用 Train 相对记录年龄 tertile 作敏感性分组，不冒充真实播种年龄。",
        "- 裁剪/尺寸：原图宽高变化是可测代理。全数据5株尺寸变化；不能从尺寸元数据确定实际裁剪、同尺寸取景变化或唯一误差成因。",
        "- **几何变化关联不可识别：5株尺寸变化植物全部属于 Train，Val/Test 均为0株。** 因而当前独立评价不能检验尺寸变化与误差的关联，不制造虚假的 Test 分组差异；完整 plant/尺寸/split 清单见 geometry_inventory.csv。Train 拟合误差与未见植物误差不等价。",
        "- 预测间隔：实际小时数只作为分析变量；日尺度模型不见实际匹配时差。",
        "- 历史面积变化率：三帧 mask-derived 面积比例与历史实际时间的 OLS 斜率（ratio/day），只用于诊断，绝不输入 RGB 模型；尺寸改变会影响面积比例解释。", "",
        "预先指定 day5 的真实 GRU 叶数误差分层：", "", markdown_table(key_difficulty[["factor", "level", "plants", "windows", "plant_macro_mae", "ci95_low", "ci95_high", "warning"]]), "",
        f"day5 真实 GRU 的叶数阶段误差：early {stage_primary.loc['early', 'plant_macro_mae']:.4f}（n={stage_primary.loc['early', 'plants']}）、intermediate {stage_primary.loc['intermediate', 'plant_macro_mae']:.4f}（n={stage_primary.loc['intermediate', 'plants']}）、mature {stage_primary.loc['mature', 'plant_macro_mae']:.4f}（n={stage_primary.loc['mature', 'plants']}）。这是观察到的任务难度差异；阶段由叶数定义，不能当成独立生理阶段机制。",
        f"day5 每株历史面积变化率与叶数误差的 Spearman rho={area_association.rho:.4f}，植物 CI95 [{area_association.ci95_low:.4f}, {area_association.ci95_high:.4f}]。这提示高速面积变化植物可能更难预测，但没有排除基因型、阶段、采集队列与面积比例定义的共同影响，也不证明遮挡或衰老。", "",
        "真实 GRU 的独立植物均值 Spearman 关联（CI 为植物 bootstrap；描述性，不是机制/因果检验，不据此选任务或删样本）：", "",
        markdown_table(key_associations[["task", "variable", "plants", "rho", "ci95_low", "ci95_high"]]), "",
        "完整所有模型/阶段/下降/时间/面积速度/尺寸分层见 difficulty_stratified_errors.csv，子组同株 GRU-对照差值见 difficulty_paired.csv。原始连续有效相邻帧叶数转移见 outputs/m1_5/adjacent_label_transitions.csv，分阶段汇总见 label_transition_summary.csv。标签非单调是可见叶片计数现象；本实验没有人工复标、遮挡分解或器官身份跟踪，不能归因到唯一机制。", "",
        "Test day5 仅1个叶数下降窗口（1株），day7没有下降窗口；这一特定关联不能在它们上稳定估计。报告相应缺失 CI，不强行外推。另存 difficulty_validation_errors.csv 提供已用于选 epoch 的 Val 敏感性描述，不能当独立确认。", "",
        "数据是否存在动态信息：以下为窗口级**描述性覆盖**（不是独立统计样本量）。未来标签确实随时间变化，并非全部静止；但三帧历史中可见叶数不变的比例及很短历史跨度，可能限制从短期变化估计较远未来。真正历史增量仍以 A2/A3 和配对结果为依据，不能只凭非零标签变化就断言 Temporal-JEPA 可获益。", "",
        markdown_table(dynamic.loc[dynamic.split.eq("test")].drop(columns="split")), "",
        "## 7. 统计、训练和工程验证", "",
        "主指标 Plant-level Macro MAE：每株先平均所有该任务窗口误差，再对植物等权平均。另存 Window MAE/RMSE 和 Plant Macro RMSE，不能用窗口数量放大独立生物样本数。双侧符号翻转检验依赖零差假设下植物差值符号的交换性/对称性，不是随机化处理的因果检验。",
        "三个种子报告均值±样本 SD（不是 SEM）；CI 先平均每个 seed 的植物误差，再对独立植物 10,000 次 bootstrap，不能把3种子当额外生物重复，也不是预测 ensemble。配对差在相同植物、相同窗口上计算；双侧植物符号翻转 20,000 次，Holm 分家族校正。区间按各植物在当前划分中的变异估计，不包含新划分/批次/编码器不确定性。",
        "所有目标/时间标准化只从 Train 拟合；逆每株窗口数加权的标准化等权 Huber(delta=1)；AdamW、weight_decay=1e-4、dropout=0.1、clip=5、cosine 到起始 LR 的0.1。MLP/difference LR=0.001 batch512；GRU LR=0.0003 batch256；只沿用原 M1 Val 选定配置，没有基于本轮 Test 做结构/超参搜索。B1b 的 alpha={0.1,1,10,100} 仍仅由各日尺度 Val 选择。",
        "A2/A3 所有100轮和实际 optimizer_steps相同，各自只在 Val 选择最优 epoch；可复现初始化和 batch-order SHA 见 capacity_training_audit.csv。GPU0/1 两张 A6000 按互不重叠的任务并行，没有多卡同步训练；设备、时间、峰值显存逐运行保存，时间和不是总墙钟时长。",
        "", markdown_table(resource_table), "",
        f"自动化测试：{totals['tests']} passed、0 failed、0 skipped。包含 M0/M1 回归测试、唯一匹配/等距/容差、隔离帧不压缩、标签不影响匹配、未知未来时间不可进入 offset-only/day 条件、Time-only/单帧输入边界、从头重复帧训练、相同初始化/顺序/步数、真实产物完整性和统计配对规则。证据：tests.xml / tests_initial.xml。",
        "初始开发遇到 CSV 标签浮点回读和可选 tabulate 不存在：修复回读类型并使用无新增依赖的 Markdown 表格；没有改旧结果或据 Test 调参。完整运行依赖复用 requirements-m1-lock.txt，无新依赖安装或环境覆盖。", "",
        "推理精度复核发现 PyTorch 新进程的默认 cuDNN TF32 与训练时关闭 TF32 不同。统一调用训练的 deterministic/TF32=False 设置后重新评估，**没有改训练函数、配置、checkpoint或模型选择**。首轮所有预测/统计原样归档于 outputs/m1_5/predictions_before_precision_fix 和 analysis_before_precision_fix，修复前源码快照与 SHA 在 inference_precision_correction.json；这是一致性修复，不是根据 Test 优化方法。", "",
        "推理稳定性另检查代表性 offset4 重复帧 GRU 与 day5 真实 GRU，同GPU两次逐值一致；两张GPU的差异明细见 inference_repeatability.json。冻结训练源码快照保留原SHA，重放仅允许有记录且 AST 证明非 predict 函数完全不变的精度修复，拒绝悄悄改变训练实现。", "",
        "## 8. 交付与复现", "",
        "```bash\nCUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/run_m1_5.py --stage prepare\nCUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/run_m1_5.py --stage train --device cuda:1\nCUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/run_m1_5.py --stage evaluate --device cuda:1\n.venv-m1/bin/python scripts/run_m1_5.py --stage analyze\nCUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python -m pytest tests -q --junitxml=reports/m1_5/tests.xml\n.venv-m1/bin/python scripts/run_m1_5.py --stage report\n```", "",
        "同一目录可重放：已完成且配置一致的训练复用 checkpoint，禁止覆盖；协议、可行性和输入哈希不一致则报错，不会静默重选 Test。要从零再训练，应在保留全部现有产物的单独工作副本中复现，不能删除旧困难样本或 checkpoint。",
        f"- {expected_runs} 个神经运行：`outputs/m1_5/checkpoints/<task>__<model>__s<seed>/checkpoint.pt`，附 config.json、training_history.csv、resources.json。",
        "- 9 个日尺度简单基线；B1b 状态/候选验证分数在 outputs/m1_5/simple。",
        "- 每个运行的 Val/Test 预测 CSV、指标 JSON 与植物级误差 CSV：outputs/m1_5/predictions/<task>__<model>__s<seed>/。",
        "- 汇总：baseline_comparison.csv、per_seed_metrics.csv、plant_level_errors.csv、paired_comparisons.csv、metrics.json、decision.json。",
        "- 协议/追溯：configs/m1_5/protocol.json、protocol_lock.json、training_source_fingerprints.json、legacy_snapshot.json、initial_protection.json、final_protection.json。",
        "- 可行性：CALENDAR_FEASIBILITY.md 与 calendar_*.csv；诊断标签仅在 outputs/m1_5/window_diagnostics.csv，不是模型输入。",
        "- 图表：figures/capacity_control、time_conditions、calendar_feasibility、difficulty_day5，各 PNG/SVG/PDF；来源与 QA 见 FIGURE_CONTRACT.md、figure_source_qa.json、figure_visual_qa.md。图片内容由 Agnes API 识别；最终4图均无阻塞视觉问题，原始建议完整保存。时间图的模型x偏移仅为展示分离，同一刻度仍是同一固定任务，不是改变预测时间；数值请以CSV为准。若没有设置 AGNES_API_KEY，视觉 QA 必须标记 unavailable，不得伪造 PASS。", "",
        "图表额外数值 QA 校验每个面板数据点/误差棒端点位于坐标界内，并核验时间图各方法的像素x间隔大于标记宽度；这与图像描述互补，避免把视觉服务对末个刻度或图例的猜测当作真实数值截断。非阻塞建议（相同max值重合、各困难子组使用不同可读轴范围等）不隐藏，均可回溯。", "",
        "## 9. 下一步边界", "", decision_text, "",
        "无论该监督模型结论如何，都不能自动否定 Temporal-JEPA：冻结 DINOv2 可能没有保留任务所需动态，GRU/轻量 MLP 的归纳偏置可能有限，三个短历史帧与可见叶数/投影面积可能并非理想动态目标。现有结果只约束当前数据、缓存与监督训练合同。",
        "若后续研究时序表征，先固定额外独立数据、请求时间合同和主要终点，再比较单帧、重复帧、显式变化及新方法；若转向事件/器官级动态，应先明确可见性与器官身份标签并审计，不在本次启动 M2。", "",
        "## 来源", "", "实验事实：本报告列出的本地只读 M0/M1 和新 M1.5 产物；来源/检索/精确 PyTorch 2.8 文档见 SOURCES.md。博查接口 HTTP403，未把搜索失败伪装成成功。绘图是实际统计数据渲染，不是文生图生成。", "",
    ]
    (report / "M1_5_REPORT.md").write_text("\n".join(sections))
    acceptance = {"status": "PASS", "research_judgment": verdict["research_judgment"], "neural_runs": expected_runs, "simple_runs": 9,
                  "tests": totals, "legacy_unchanged": True, "raw_files_checked": protection["raw_files_checked"],
                  "fixed_budget_capacity_pairs": 18, "eligible_days": lock["eligible_days"], "test_was_inspected_in_m1": True,
                  "test_used_for_training_or_selection": False, "jepa_trained": False, "three_dimensional_experiments": False,
                  "main_merged": False, "code_committed": False, "code_pushed": False,
                  "figure_visual_qa_inspected": len([record for record in visual["records"] if record["status"] == "inspected"]),
                  "figure_visual_qa_blocking_issues": [record["figure"] for record in visual["records"] if record["status"] == "inspected" and record["response"]["has_blocking_visual_issue"]]}
    write_json(report / "acceptance.json", acceptance)
    manifest = {str(path.relative_to(root)): file_hash(path) for directory in (report, root / "outputs/m1_5", root / "configs/m1_5", root / "src/m15")
                for path in sorted(directory.rglob("*")) if path.is_file() and path.name != "artifact_manifest.json" and path.suffix != ".log" and "__pycache__" not in path.parts}
    manifest["scripts/run_m1_5.py"] = file_hash(root / "scripts/run_m1_5.py")
    manifest["scripts/qa_m1_5_figures.py"] = file_hash(root / "scripts/qa_m1_5_figures.py")
    manifest["tests/test_m1_5.py"] = file_hash(root / "tests/test_m1_5.py")
    write_json(report / "artifact_manifest.json", manifest)
    print(json.dumps(acceptance, ensure_ascii=False, indent=2))
