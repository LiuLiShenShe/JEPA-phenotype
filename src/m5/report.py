from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from m4.common import sha256, write_json


def markdown_table(frame):
    frame = frame.copy()
    for column in frame:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: "NA" if pd.isna(value) else f"{value:.4f}")
    header = "| " + " | ".join(frame.columns) + " |\n"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |\n"
    return header + separator + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))


def generate_report(root):
    root = Path(root)
    folder = root / "reports/m5"
    decision = json.loads((folder / "decision.json").read_text())
    protocol = json.loads((root / "configs/m5/protocol.json").read_text())
    lock = json.loads((folder / "protocol_lock.json").read_text())
    data = json.loads((folder / "data_preparation.json").read_text())
    metrics = pd.read_csv(folder / "downstream_metrics.csv")
    paired = pd.read_csv(folder / "paired_comparisons.csv")
    budget = pd.read_csv(folder / "ssl_budget_comparison.csv")
    ssl = pd.read_csv(folder / "pretraining_dataset_comparison.csv")
    probe = pd.read_csv(folder / "leaf_structure_probe.csv")
    primary = metrics.loc[metrics.task.eq("count") & metrics.label_fraction.eq(.5)]
    primary_table = primary.groupby(["partition", "condition"]).plant_macro_mae.agg(["mean", "std"]).reset_index()
    primary_pairs = paired.loc[paired.task.eq("count") & paired.label_fraction.eq(.5) & paired.left.isin(("P3", "P4")) & paired.right.isin(("P0", "P2", "S500"))]
    efficiency = metrics.loc[metrics.task.eq("count")].groupby(["condition", "label_fraction"]).plant_macro_mae.agg(["mean", "std"]).reset_index()
    segmentation = metrics.loc[metrics.task.eq("segmentation")]
    segmentation_table = segmentation.groupby("condition")[["plant_macro_miou", "plant_macro_iou_taproot", "plant_macro_iou_core", "plant_macro_iou_leaf"]].mean().reset_index()
    semantic_pairs = paired.loc[paired.partition.eq("official") & paired.task.eq("segmentation") &
                                paired.left.isin(("P3", "P4")) & paired.right.eq("P0")]
    probe_table = probe.loc[probe.partition.eq("official")].groupby("condition")[["centred_pair_auc", "centred_distance_matched_margin", "centred_distance_matched_accuracy"]].mean().reset_index()
    formal = ssl.loc[ssl.condition.ne("M5A")]
    integrity = json.loads((folder / "final_protection.json").read_text()) if (folder / "final_protection.json").exists() else {"status": "NOT_RUN"}
    coverage = json.loads((folder / "evaluation_complete.json").read_text())
    fairness = json.loads((folder / "fairness.json").read_text())
    restores = json.loads((folder / "checkpoint_restore.json").read_text()) if (folder / "checkpoint_restore.json").exists() else {"status": "NOT_RUN", "checkpoints": 0}
    verification = json.loads((folder / "verification.json").read_text()) if (folder / "verification.json").exists() else {"status": "NOT_RUN"}
    figure_qa = json.loads((folder / "figure_source_qa.json").read_text()) if (folder / "figure_source_qa.json").exists() else {"status": "NOT_RUN"}
    visual_qa = json.loads((folder / "figure_visual_qa.json").read_text()) if (folder / "figure_visual_qa.json").exists() else {"status": "NOT_RUN"}
    diagnostics = json.loads((folder / "representation_diagnostics.json").read_text())
    representation_rows = []
    for checkpoints in diagnostics.values():
        if "posttraining_probe" not in checkpoints:
            continue
        record = checkpoints["posttraining_probe"]
        representation_rows.append({"partition": record["partition"], "condition": record["condition"], "seed": record["seed"],
                                    **{f"{kind}_{field}": record[kind][field] for kind in ("pooled", "token")
                                       for field in ("std_mean", "effective_rank", "variance_mean", "largest_eigenvalue_fraction")},
                                    "hard_collapse": record["hard_collapse"]})
    representation = pd.DataFrame(representation_rows)
    representation.to_csv(folder / "representation_summary.csv", index=False)
    representation_table = representation.loc[representation.partition.eq("official")].groupby("condition")[
        ["pooled_effective_rank", "token_effective_rank", "pooled_std_mean", "token_std_mean", "pooled_largest_eigenvalue_fraction"]].mean().reset_index()
    external_representation = pd.read_csv(folder / "external_representation_summary.csv")
    external_table = external_representation.loc[external_representation.species.eq("all")].groupby(["source", "condition"])[
        ["objects", "pooled_effective_rank", "token_effective_rank", "pooled_largest_eigenvalue_fraction"]].mean().reset_index()
    tests = {}
    import xml.etree.ElementTree as etree
    for filename in ("tests_m5.xml", "tests_contracts.xml", "tests_all.xml"):
        path = folder / filename
        if path.exists():
            suite = etree.parse(path).getroot()
            suites = [suite] if suite.tag == "testsuite" else list(suite)
            tests[filename] = {key: sum(int(item.get(key, 0)) for item in suites) for key in ("tests", "failures", "errors", "skipped")}
    source_status = "PARTIAL: Crops3D native metric unit and non-cabbage physical IDs incompletely documented; exact geometry and acquisition audits do not establish all physical identities"
    tests_pass = len(tests) == 3 and all(record["failures"] == 0 and record["errors"] == 0 for record in tests.values())
    probes_complete = coverage["completed_probe_runs"] == coverage["expected_probe_runs"]
    expected_restores = int((~metrics.historical.astype(bool)).sum())
    accepted = (coverage["complete"] and probes_complete and tests_pass and fairness["status"] == "PASS" and
                integrity["status"] == "PASS" and restores["status"] == "PASS" and restores["checkpoints"] == expected_restores and
                verification["status"] == "PASS" and figure_qa["status"] == "PASS" and visual_qa["status"] == "PASS")
    acceptance = {"milestone": "M5", "engineering_status": "PARTIAL", "completed_executable_matrix": accepted,
                  "research_decision": decision["decision"], "source_identity_unit_audit": source_status,
                  "checks": {"downstream_complete": coverage["complete"], "paired_fairness": fairness["status"],
                             "archive_protection": integrity["status"], "checkpoint_restore": restores["status"],
                             "tests": tests, "probe_complete": probes_complete, "state_provenance_verification": verification["status"],
                             "figure_source_qa": figure_qa["status"], "figure_visual_qa": visual_qa["status"],
                             "p3_p4_preregistered": True, "within_partition_sugar_validation_test_in_ssl": False,
                             "global_official_holdout_across_all_fold_models": False},
                  "limitations": ["Crops3D native units unresolved, input normalization dimensionless", "non-cabbage independent plant identities unresolved", "Bonn uses annotation-assisted partial-plant cropping", "no untouched confirmatory Sugar4D test", "M4 fold-local role rotation is retained; folds do not constitute a global official holdout"],
                  "next_stage_started": False, "temporal_jepa": False, "future_prediction": False, "diffusion": False,
                  "git_push_merge_commit": False}
    write_json(folder / "acceptance.json", acceptance)
    expanded = primary_pairs.loc[primary_pairs.partition.eq("official")]
    primary_lines = markdown_table(expanded[["left", "right", "left_metric", "right_metric", "absolute_improvement", "relative_improvement", "improving_seeds", "ci95_low", "ci95_high"]])
    data_pair = expanded.loc[expanded.left.eq("P3") & expanded.right.eq("S500")].iloc[0]
    rank_means = representation_table.set_index("condition")
    answer1 = (f"单独扩大数据的 P3 相对同 500 步 S500：official 50% MAE "
               f"{data_pair.right_metric:.4f}→{data_pair.left_metric:.4f}；"
               f"pooled rank {rank_means.loc['S500', 'pooled_effective_rank']:.3f}→"
               f"{rank_means.loc['P3', 'pooled_effective_rank']:.3f}。"
               f"P4 pooled rank 为 {rank_means.loc['P4', 'pooled_effective_rank']:.3f}，可缓解低秩风险，"
               "但未转化为优于三个控制的稳定主任务收益。"
               "P4 同时改变正则化，不能把 P4−S500 的差值全部归因于数据扩展；"
               "是否真正成功必须满足完整预设门槛。")
    stop_answer = ("应停止当前通用 Point-JEPA 研发分支：数据扩展与固定正则化未通过完整主任务门槛，且器官分割负迁移持续。"
                   if decision["decision"] == "JEPA-STOP" else
                   "是否继续仅由预设主指标/风险规则决定，不根据辅助任务挑选。")
    report = f"""# M5：扩展植物点云数据的 JEPA 表征恢复与验证

日期：2026-10-12。最终研究决策：**{decision['decision']}**。

工程状态：**PARTIAL**。真实下载、实验矩阵、评价与检查点已按下表记录；但
Crops3D 未完整公开物理植株身份/原生米制单位，不能冒称数据审计全部 PASS。
本阶段完成后停止，不自动启动下一阶段，不合并、提交或推送 Git。

## 1. 五个必须回答的问题

1. **更大的预训练数据是否改善表征？** {answer1} 下面同时报告方差、有效秩、
   中心化相似度、训练植株叶实例探针和全部下游结果，不能只凭 SSL loss 回答。
2. **更多训练步骤改善还是恶化退化？** M5-A pooled rank 从20.709→13.185→8.544→1.525，
   对应0/120/500/2000步。2000步方差上升却91.89%集中在第一特征方向，说明低维退化，
   不等于零方差完全坍塌。未再盲目加时；正式全部锁定500步。
3. **是否真正提高少标注结构任务性能？** 按50%当前可见叶数、三个同架构控制、
   最小实际改善、三种子、植株配对 CI、四折和分割风险共同判定为 **{decision['decision']}**。
   不从25%/100%、阶段、物种或辅助任务中挑最佳结果替换主任务。
4. **是否值得发展成未来生长预测模型？** 本次仅测试当前几何表征，未测试未来预测；
   当前证据不支持将此通用 Point-JEPA 分支推进为未来生长预测模型。即使当前主任务
   改善也不足以证明时序能力。M3.5的动态任务门槛仍未通过，不启动Temporal-JEPA。
5. **是否终止通用Point-JEPA路线？** {stop_answer} 这不代表植物表型预测研究停止。

## 2. 历史复核与成果保护

见 `HISTORY_AUDIT.md`、`history_audit.json`、`historical_checkpoints.csv`。
M4仍为LIMITED-BENEFIT/工程PARTIAL，P1官方权重仍未使用。
15份P2各24株/384扫描/120优化步，架构384×12×6、predictor192×6×6、64×32patch不变。
严格重载的30份P0/P2权重hash与原诊断一致；token384、pooled768、count795输入无维度错误；
EMA/stop-gradient/共享tokenizer及缓存归一化没有发现实现错误。
保护{integrity.get('protected_files', 'unknown')}个既有文件：{integrity['status']}，
不覆盖任何M0–M4报告、模型、划分或原始数据。raw使用大小/纳秒mtime，其他文件SHA256。

**探索性边界**：50%目标由已经看过的M4少标注辅助结果提出。保留官方Train/Val/Test
和四折植株级划分；每折仅该折 Train 参加 SSL。任何 Val/Test 都不参与该折 SSL。
这一规则是 **partition-local**：official 模型只用原 official 的 24 株 Train；既有四折
会轮换植株角色，其 Train 中分别包含原 official Val/Test 的 5/5、5/6、7/7、7/6 株。
见 `split_provenance.json`；不能声称所有折模型共同保持全局 official Val/Test 从未进入 SSL。
四折只是原 M4 划分兼容的探索性稳健性证据，不是全局 official 留出验证。
以前被使用过的Test不能重新称为独立验证；本次不生成Test预测、不用Test实例标签训练特征提取器。
所有新旧下游都用各自 Validation 选择最佳 epoch；因此主表是 validation-selected
探索性性能，CI 条件于这些已训练/已选模型，未校正模型选择偏差，不是新盲测集置信区间。
“不使用 Test 标签”的限制同样以各 partition 的角色为准；既有折角色轮换已明确披露。

## 3. 真实外部数据与许可

详细审计见`DATA_SOURCE_AUDIT.md`、`external_data_manifest.csv`、`data_preparation.json`。
外部实际采纳{data['admitted_scans']}份几何对象。来源计数：{json.dumps(data['sources'], ensure_ascii=False)}。
Crops3D固定Figshare article v1，CC0-1.0；官方823个Train单株文件，只用原始Crops3D目录，
不重复读取10k、腐蚀HDF5或50个田块作为独立植物。白菜按14株的文件身份平衡，
其他作物独立植株数未知；不能把1230样本或823Train扫描当作同数独立植株。
BonnBeetClouds3D V1.0 CC0-1.0，149 Train patch解析为128个released plant ID；
每身份仅取点数最多的单个Train patch内XYZ片段，可能是部分植株，标注辅助裁剪如实披露。
早期跨patch合并发现数米异常跨度，在正式训练前纠正，旧缓存不用于正式模型。
没有使用leaf_ids、器官语义、关键点、未来形态或外部Test标签构建JEPA目标。

比例锁定Sugar4D0.5、Crops3D0.25、Bonn0.25；外部物种均匀、已解析白菜身份均匀后扫描。
每run实际source/species曝光量保存在`sampling_exposures.json`，不会由大数据源主导。
S500/P3/P4 的总优化步、batch 及总点云曝光量匹配（每 run 500×64=32,000）；
P3/P4 中 Sugar4D 曝光量约减半。因此比较识别的是 **新增源＋固定混合策略** 的效果，
不是在完全保留 Sugar4D 曝光量的同时增加训练时间，也不能孤立归因于独立植株数量。
所有点云XYZ-only2048均匀采样，当前完整/选定片段均值中心化、最大半径缩放、64FPS×32KNN。
Sugar4D/Bonn米制保留；Crops3D保留native尺度字段，不编造native→metre换算，模型输入为无量纲。
不同采集系统均读取原 x/y/z 字段；未建立所有来源统一的重力方向/绝对姿态校准，
所以“统一 XYZ”不意味着统一物理朝向。沿用 M4 的 z 轴旋转增广，不事后修改预处理。
完整ZIP大小/CRC、SHA256固定；Bonn官方MD5匹配。跨源exact归一化几何重复
{data.get('cross_source_exact_geometry_duplicates', 'not recorded')}；无公开已知重叠采集，
但哈希无重复不能证明所有近重复或重复物理植株均不存在。

## 4. M5-A与预先锁定协议

协议SHA256：`{lock['protocol_sha256']}`。`protocol_lock.json`冻结原始输入、代码与label budgets。
`preregistration_chronology.json`核验本地锁文件早于所有正式模型初始诊断。
这是本地探索性协议锁，不是公开注册或可信第三方时间戳。
M5-A只用official Train、seed17；梯度、EMA、特征与中间完整状态真实保存。

{markdown_table(budget[['step','ssl_loss','token_std_mean','token_effective_rank','pooled_std_mean','pooled_effective_rank','pooled_variance_mean','pooled_largest_eigenvalue_fraction']])}

原停止规则要求方差与秩同时下降，没抓到高方差低秩集中；如实记为诊断规则不足。
M4 的120→正式500比较还改变了从逐 epoch 无放回到固定 step 有放回采样及调度 horizon；
它是训练预算配置比较，不是仅改步数的单变量因果实验。M5-A 同一2000步训练轨迹才是
本阶段主要的步数退化诊断；正式 S500 与 P3/P4 的 horizon/采样方式则匹配。
`BUDGET_SELECTION_NOTE.md`在任何正式训练之前仅依据Train诊断一次性修订到500步。
M5-A500不是正式S500：前者2000步cosine/EMA horizon，后者重新从头500步horizon。
M5-A120也不是精确M4复现；M4历史120步原结果作为真正P2。

P4条件在M5-A500满足：token std初值17.1%、pooled rank初值41.3%；仅用预写的
variance权重1.0、covariance0.01、std floor0.1、eps1e-4，正则化实际student context token。
没有第二次encoder/BN前向、没有损失/权重搜索、没有下游选SSL轮次。
每run固定500步，20epoch下游，三种子[17,42,93]。
两张A6000仅运行互斥partition任务，非独占速度benchmark，不终止其他进程。

| 条件 | SSL 几何 | SSL 步数/每 run | 额外损失 |
| --- | --- | --- | --- |
| P0 | 随机初始化历史控制，不预训练 | 0 | 无 |
| P2 | 原每折 24 株 Sugar4D Train / 384 扫描 | 120 | 原 JEPA |
| S500 | 同折 24 株 Sugar4D Train / 384 扫描 | 500 | 原 JEPA |
| P3 | 同折 Sugar4D Train ＋ 951 外部 Train 几何对象 | 500 | 原 JEPA |
| P4 | 与 P3 完全相同的数据及采样 | 500 | 同一预定方差/协方差正则化 |

## 5. 主任务：50%当前可见叶数

Label budget为12个Train独立植株及其全部扫描；不把部分扫描叫作少植株。
主指标先每株扫描平均absolute error，再植物macro；非扫描伪重复。
最小改善：同时0.2片和10%；必须分别优于P0、P2、S500，三种子同向，official
seed-averaged植物paired bootstrap10000次CI上界<0，至少3/4validation折方向改善。
三控制为交集门槛；P4是明确条件性探索救援，不冒充第二个独立确认实验。

各划分三种子均值/SD：

{markdown_table(primary_table)}

官方植株配对结果（delta=left−right，MAE负值有利）：

{primary_lines}

全部四折配对方向、种子delta与置信区间见`paired_comparisons.csv`，不把official和folds
拼成60株。四折Validation覆盖48株但重叠的模型训练和官方集不构成独立重复。
P0/P2未重新训练，原20epoch模型/预测原字节保留；S500/P3/P4复用相同下游实现。
head初值、可训练参数、epoch、step公平性：**{fairness['status']}**。

## 6. Label efficiency与语义/实例结构辅助证据

25/50/100%即6/12/24株，原nested预算；下面5划分×3seed的均值/SD仅描述，不是15份独立生物学重复。

{markdown_table(efficiency)}

语义分割：100%标签、同M4插值+XYZ/offset头，plant-macro三类IoU：

{markdown_table(segmentation_table)}

明显负迁移风险阈值：official mIoU下降至少0.03且植物配对CI上界<0；它只是否决风险，
不替代叶数主指标。2048点sampled IoU不是全分辨率，不是叶实例PQ/SBD。

Official paired semantic differences（left−P0，负值代表负迁移）：

{markdown_table(semantic_pairs[['left','right','mean_delta','ci95_low','ci95_high','independent_plants']])}

训练植株只读叶结构探针：token的32邻域必须>=80%是同一当前leaf实例；同叶/异叶
cosine、per-scan中心化AUC、距离最近匹配异叶pair margin，先扫描再植物macro。
标签仅用于本探针评价，不反传到encoder、不从Val/Test实例标签训练提取器。

{markdown_table(probe_table)}

此为Train侧器官可分性诊断，不称为未见植物实例分割。近几何距离可混杂pair相似性，
因此报告distance-matched结果及残余距离差。没有正式instance decoder/PQ/SBD结果。
随机 P0 的局部 patch/位置几何本身也能产生同叶相似性；高 AUC 不是已学到叶器官语义的
充分证据。距离匹配仅控制 pair 距离，不能排除局部曲率、token 纯度和空间布局的全部混杂。

## 7. 表征与成本

`representation_diagnostics.json`记录pooled/token每维std、协方差特征值/方差、
effective rank（covariance-eigenvalue entropy，与M4一致）、participation rank、
mean energy、same-plant/different-scan、different-plant及same-scan/different-patch
中心化/不中心化cosine分位数。高cosine本身不作完全坍塌结论，下降SSL loss也不是成功。

同一 official Train 384 份扫描、24 株植物，三个种子均值（仅诊断，不是独立验证）：

{markdown_table(representation_table)}

这是下游微调 **之前** 的各源编码器；P0/P2 与新模型使用同一诊断实现。
`representation_summary.csv` 包含全部划分和种子，不能用局部 token 相似性冒充器官实例性能。
P4 的方差/协方差损失在 train-mode context token 上实施；dropout/局部 patch 多样性可能使
损失满足阈值而 eval-mode pooled/token 仍各向异性，因此不据正则项下降断言全面恢复。

补充 **训练完成后的只读外部 Train 几何诊断**：official 编码器三种子，按来源单独统计，
不使用外部 Test 标签，不改变模型、损失、权重、checkpoint 或成功判定。

{markdown_table(external_table)}

外部 species 级完整结果见 `external_representation_summary.csv`，全维方差/协方差和
distinct-object 中心化/不中心化 cosine 见 `external_representation_diagnostics.json`。
混合来源的秩可能反映扫描系统/物种差异；distinct object 相似性不称为不同独立植株相似性，
也不能作为未见 Sugar4D 主任务成功的替代证据。

正式SSL完成{len(formal)}runs、{int(formal.optimizer_steps.sum())}steps，
实测累计{formal.seconds.sum():.1f}秒；M5-A另2000steps。
新下游{int((~metrics.historical.astype(bool)).sum())}runs；矩阵共
{coverage['completed_downstream_runs']}/{coverage['expected_downstream_runs']}（含原P0/P2）。
探针{coverage['completed_probe_runs']}/{coverage['expected_probe_runs']}。
显存、曝光量、wall time是实测共享负载run统计，不能作方法速度排名。
SSL/下游完整状态与best encoder保存于`outputs/m5/`；重载验证
{restores.get('checkpoints', 0)}个新下游checkpoints：{restores['status']}。
完整 SSL 优化器/学习率调度/EMA/随机数/采样器及矩阵来源验证：**{verification['status']}**。
环境版本、CUDA/驱动与依赖见 `environment.json`、`environment_freeze.txt`。

## 8. 决策与复现

```json
{json.dumps(decision['candidate_checks'], ensure_ascii=False, indent=2)}
```

最终唯一研究选择：**{decision['decision']}**。不事后换任务，不自动训练Temporal-JEPA、
未来植物预测或Diffusion，不合并/提交/推送Git。

自动测试：{json.dumps(tests, ensure_ascii=False)}。
评估输出曾遇到 NumPy boolean JSON 序列化错误，已仅在输出边界转换精确标量，
锁定评价实现、指标和判定不变；复核记录见 `OUTPUT_SERIALIZATION_NOTE.md`。
全套旧环境执行，保持隔离环境和历史依赖不被覆盖。
图在`figures/`，CSV可复算，editable PDF/SVG及PNG；直接查看图形质检不是人工签署。
图源/版面检查：{figure_qa['status']}；当前模型直接查看 PNG 的视觉检查：{visual_qa['status']}。

复现（锁定输入变化会拒绝运行）：

```bash
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/run_m5.py --stage run --partitions official,fold0,fold1 --device cuda:0
CUDA_VISIBLE_DEVICES=1 .venv-m4/bin/python scripts/run_m5.py --stage run --partitions fold2,fold3 --device cuda:0
.venv-m4/bin/python scripts/evaluate_m5.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/diagnose_m5_external.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/restore_m5.py
.venv-m4/bin/python scripts/verify_m5.py
.venv-m1/bin/python -m pytest tests -q
.venv-m4/bin/python scripts/prepare_m5.py --stage verify
.venv-m4/bin/python scripts/report_m5.py
```

## 9. 来源与限制

- Crops3D官方仓库：https://github.com/clawCa/Crops3D ，fixed revision见sources/crops_revision.json。
- Crops3D数据：https://doi.org/10.6084/m9.figshare.27313272 ，许可独立于文章版权。
- 数据论文：https://www.nature.com/articles/s41597-024-04290-0 ，Zhu等，2024。
- Bonn数据：https://doi.org/10.60507/FK2/34W30T ，Marks等，CC0-1.0。
- Bonn论文：https://arxiv.org/abs/2312.14706 ，current v2发布2025-08-11，数据固定V1.0。
- Sugar4D：https://doi.org/10.60507/FK2/IS8YBZ ，原V4/CC BY4.0/2021采集，Bömer等归属保留。
- 技术语法来自Context7 `/pytorch/pytorch/v2_8_0`官方文档快照；不是凭记忆写EMA API。
- 博查HTTP403配额不足，明确搜索不可用；上述结论来自实际官方URL正文、发布API及本地
  真实计算，而非把未搜索事实隐瞒。Figshare metadata API403，官方下载整包可用；
  DataCite确认CC0，ZIPCRC/SHA256与byte size固定，缺Figshare发布MD5不假称已验。
- 外部独立身份与native单位的不完整性限制因果解释、计数与物理尺度结论；数据扩展+来源
  混合的收益不能归因某单一物种。外部plant ID裁剪不是完全无监督的数据准备。
- 来源混合亦改变采集密度、遮挡和局部完整性；本阶段没有扫描系统/物种单独消融，
  因此否定结论针对当前通用 Point-JEPA、固定来源混合与训练设置，不证明所有植物 JEPA
  或任务专属器官表示方法均不可能有效。
"""
    (folder / "M5_REPORT.md").write_text(report)
    write_json(folder / "report_manifest.json", {"report_sha256": sha256(folder / "M5_REPORT.md"),
               "decision": decision["decision"], "protocol_sha256": lock["protocol_sha256"], "next_stage_started": False})
