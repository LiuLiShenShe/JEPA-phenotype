from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m4.common import load_protocol, write_json


def markdown_table(frame, float_format=".4f"):
    if frame.empty:
        return "No completed observations."
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(format(value, float_format) if np.isfinite(value) else "NA")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join(rows)


def generate_report(root):
    root = Path(root)
    protocol, lock = load_protocol(root)
    report = root / "reports/m4"
    count = pd.read_csv(report / "label_efficiency.csv")
    segment = pd.read_csv(report / "segmentation_metrics.csv")
    baseline = pd.read_csv(report / "strong_baseline_validation.csv")
    comparisons = pd.read_csv(report / "paired_comparisons.csv")
    coverage = json.loads((report / "coverage.json").read_text())
    decision = json.loads((report / "decision.json").read_text())
    fairness = json.loads((report / "fairness.json").read_text())
    diagnostics = json.loads((report / "representation_diagnostics.json").read_text())
    resources = pd.read_csv(report / "training_resources.csv")
    ssl = pd.read_csv(report / "pretraining_resources.csv")
    primary = count.loc[count.partition.eq("official") & count["mode"].eq("finetune") & count.label_fraction.eq(1.0)]
    primary_table = primary.groupby("condition").agg(MAE=("plant_macro_mae", "mean"), seed_SD=("plant_macro_mae", "std"),
                                                     RMSE=("plant_macro_rmse", "mean"), seeds=("seed", "nunique")).reset_index()
    official_baseline = baseline.loc[baseline.partition.eq("official")]
    baseline_table = official_baseline.groupby("condition").agg(MAE=("plant_macro_mae", "mean"), seed_SD=("plant_macro_mae", "std"),
                                                                RMSE=("plant_macro_rmse", "mean"), seeds=("seed", "nunique")).reset_index()
    fold_table = comparisons.loc[comparisons["mode"].eq("finetune") & comparisons.label_fraction.eq(1.0),
                                 ["partition", "p0_mae", "p2_mae", "absolute_improvement", "improving_seeds", "ci95_low", "ci95_high"]]
    efficiency_table = count.groupby(["mode", "label_fraction", "condition"]).agg(MAE=("plant_macro_mae", "mean"),
                                                                                  run_SD=("plant_macro_mae", "std"), runs=("seed", "count")).reset_index()
    semantic_table = segment.groupby(["mode", "condition"]).agg(mIoU=("plant_macro_miou", "mean"), run_SD=("plant_macro_miou", "std"),
                                                                 Macro_F1=("plant_macro_macro_f1", "mean"), taproot_IoU=("plant_macro_iou_taproot", "mean"),
                                                                 core_IoU=("plant_macro_iou_core", "mean"), leaf_IoU=("plant_macro_iou_leaf", "mean"),
                                                                 runs=("seed", "count")).reset_index()
    diagnosis_rows = [{"partition_seed": name, "token_std": value["final"]["token_std_mean"], "embedding_std": value["final"]["embedding_std_mean"],
                       "effective_rank": value["final"]["effective_rank"], "algebraic_rank": value["final"]["algebraic_rank"],
                       "cosine_median": value["final"]["cosine_quantiles_0_05_50_95_100"][2], "collapse_warning": value["final"]["collapse_warning"]}
                      for name, value in diagnostics.items()]
    diagnosis = pd.DataFrame(diagnosis_rows)
    diagnosis.to_csv(report / "representation_diagnostics_summary.csv", index=False)
    full_diagnostics_path = report / "representation_full_train_diagnostics.json"
    full_summary = pd.DataFrame()
    if full_diagnostics_path.exists():
        full = json.loads(full_diagnostics_path.read_text())
        full_rows = [{"partition_seed": name, "condition": condition, "token_std": value["token_std_mean"],
                      "embedding_std": value["embedding_std_mean"], "effective_rank": value["effective_rank"],
                      "global_cosine_median": value["embedding_cosine_quantiles_0_05_50_95_100"][2],
                      "token_cosine_median": value["cosine_quantiles_0_05_50_95_100"][2]}
                     for name, entry in full.items() for condition, value in entry.items()]
        full_frame = pd.DataFrame(full_rows)
        full_frame.to_csv(report / "representation_full_train_summary.csv", index=False)
        full_summary = full_frame.groupby("condition").agg(token_std=("token_std", "mean"), embedding_std=("embedding_std", "mean"),
                                                          effective_rank=("effective_rank", "mean"), global_cosine_median=("global_cosine_median", "mean"),
                                                          token_cosine_median=("token_cosine_median", "mean")).reset_index()
    conditions = pd.DataFrame([{"condition": "P0", "status": "COMPLETED" if coverage["p0_p2_matrix_complete"] else "PARTIAL", "pretraining_data": "none", "official_coverage": "N/A"},
                               {"condition": "P1", "status": "UNAVAILABLE", "pretraining_data": "ShapeNet external (not obtained)", "official_coverage": "not measurable"},
                               {"condition": "P2", "status": "COMPLETED" if coverage["p0_p2_matrix_complete"] else "PARTIAL", "pretraining_data": "each partition's 24 Train plants only, from scratch", "official_coverage": "N/A; no external weights"},
                               {"condition": "P3", "status": "NOT RUN (optional)", "pretraining_data": "Point-MAE CUDA dependencies/budget not established", "official_coverage": "N/A"}])
    conditions.to_csv(report / "condition_availability.csv", index=False)
    rule_table = pd.DataFrame([{"locked_rule": name, "passes": value} for name, value in decision["checks"].items()])
    stage = pd.read_csv(report / "stage_metrics.csv")
    stage = stage.loc[stage["mode"].eq("finetune") & stage.label_fraction.eq(1.0)].groupby(["stage", "condition"]).plant_macro_mae.mean().unstack().reset_index()
    protection = json.loads((report / "final_protection.json").read_text()) if (report / "final_protection.json").exists() else {"status": "PENDING", "protected_files": 0}
    primary_comparison = decision["primary_comparison"]
    efficiency_comment = "少标注量实验尚未齐全，暂不概括收益。"
    if coverage["p0_p2_matrix_complete"]:
        statements = []
        for mode in ("probe", "finetune"):
            for fraction in protocol["label_fractions"]:
                matched = comparisons.loc[comparisons["mode"].eq(mode) & comparisons.label_fraction.eq(fraction)]
                average = count.loc[count["mode"].eq(mode) & count.label_fraction.eq(fraction)].groupby("condition").plant_macro_mae.mean()
                if {"P0", "P2"}.issubset(average.index):
                    statements.append(f"{mode}/{int(100 * fraction)}%：平均 P0→P2 MAE {average.P0:.4f}→{average.P2:.4f}，改善方向划分 {int((matched.absolute_improvement > 0).sum())}/{len(matched)}")
        efficiency_comment = "；".join(statements) + "。这些是预定辅助证据，不能替代失败的主任务门槛。"
    replay_seconds = 0.0
    replay_path = report / "state_capture_repairs.json"
    if replay_path.exists():
        replay_seconds = sum(record["seconds"] for record in json.loads(replay_path.read_text())["repaired"])
    test_summary = []
    import xml.etree.ElementTree as ElementTree
    for name in ("tests_m4.xml", "tests_all_original_environment.xml"):
        test_path = report / name
        if test_path.exists():
            suites = ElementTree.parse(test_path).getroot()
            total = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
            failed = sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites)
            test_summary.append(f"`{name}`：{total} tests，{failed} failures/errors")
    restore_path = report / "checkpoint_prediction_restore.json"
    restore_note = "checkpoint 全矩阵重载校验尚未完成。"
    if restore_path.exists():
        restored = json.loads(restore_path.read_text())
        restore_note = f"checkpoint 重载校验 {restored['checkpoints']} 份，状态 {restored['status']}；count 浮点误差容许 2e-6，segmentation predictions/confusions 必须完全一致。"
    primary_sentence = "主任务配对矩阵尚未齐全，不能形成完整科学判断。"
    if primary_comparison is not None:
        primary_sentence = (f"官方 Validation 的 P0→P2 平均 MAE 为 {primary_comparison['p0_mae']:.4f}→{primary_comparison['p2_mae']:.4f}；"
                            f"改善 {primary_comparison['absolute_improvement']:.4f} 片（{100 * primary_comparison['relative_improvement']:.2f}%），"
                            f"{primary_comparison['improving_seeds']}/{primary_comparison['paired_seeds']} 个种子改善。"
                            f"配对植株 bootstrap 的 P2−P0 MAE 95% CI [{primary_comparison['ci95_low']:.4f}, {primary_comparison['ci95_high']:.4f}]。")
    answer = ("当前锁定配置下，尚不能证明 Point-JEPA 式预训练稳定改善未见植株的三维器官结构理解；不具备直接作为后续动态预测 Encoder 的充分证据。"
              if decision["decision"] != "REPRESENTATION-SUCCESS" else
              "当前锁定配置下，Point-JEPA 式预训练在未见植株主任务达到预定实际价值和稳定性标准，可作为后续动态预测 Encoder 的候选；外部预训练与独立确认仍有缺口。")
    document = f"""# M4：基于 Point-JEPA 的植物三维自监督表征学习

日期：2026-10-12。科学判断：**{decision['decision']}**。

工程状态：**M4-A PASS；P0/P2 {'全部可行实验完成' if coverage['p0_p2_matrix_complete'] else '实验尚未齐全'}；整体工程验收 PARTIAL，不冒称 PASS**。
缺项是必需的 P1 官方外部预训练权重不可下载。没有启动 Temporal-JEPA，没有预测未来点云。

## 1. 必须回答的问题

**Point-JEPA 式自监督预训练是否改善了对未见植株的三维器官结构理解，并且能否成为后续植物动态预测的有效 Encoder？**

{answer}

{primary_sentence}

这是有限数据、有限预算的验证集可行性研究，不是官方 ShapeNet 论文重现，不是对 JEPA 全部能力的否定。
不能把 SSL Loss 下降、语义 mIoU 或某个少标注比例更好，替换成原先锁定的主要成功标准。

## 2. 资源核验与隔离工程

完整来源、许可证、入口、版本和修复见 `UPSTREAM_AUDIT.md`；官方原代码快照保留 MIT LICENSE。
Point-JEPA 固定 revision `c36b51ca20a07f3fa3c75f6b9d3bc0e29b45401c`。
Point-MAE revision `7445a68068d707da6d959b66773bfef84c4e3e32`。
Plant-MAE DOI 元数据已核实，但出版商正文 HTTP 403；找到的候选仓库仅 README，不能作为官方复现代码。
Sugar4D 实时版本为 V4、CC BY 4.0，使用原审计数据而不修改。

新环境 `.venv-m4`：Python 3.11.15，PyTorch 2.8.0+cu128；完整依赖在 `requirements-m4-lock.txt`。
M4 训练和表征推理只使用物理 GPU 0（RTX A6000）；原有其他进程没有终止。
旧 regression suite 自带 GPU 1 的微型梯度 smoke test，保持原测试且未在第二卡训练 M4。
部分 SSL/下游初期运行共享同一张 GPU，
时间/显存是实测且受共享负载影响，不应当作模型速度的严格独占硬件 benchmark。
驱动、系统 CUDA、全局 Python、旧环境和 M0—M3.5 代码未覆盖。

模型保持官方 384 维、12 层、6 头编码器以及 192 维、6 层 predictor。
FPS/KNN 改为可复现的 PyTorch 实现，FP32 替换硬编码 half，Lightning/W&B 换成本地训练循环。
保留相同的参数键名/shape；实际上游 PointNet、Transformer、Predictor 前向与本地实现 bit-exact。
这不等于未下载的官方权重覆盖率。加载器严格拒绝缺失/shape 不匹配，不允许 PointNet/Point-MAE checkpoint 混用。

{markdown_table(conditions)}

P2 是 **Sugar4D Train-only 从零自监督预训练**，不是“加载官方权重后的领域适配”。
官方 Google Drive 两种下载入口及候选镜像检索均网络失败，详见 `external_pretraining_lock.json`。
没有为 P1 编造结果，实际覆盖率为不可测，而不是虚假的 100%。

## 3. 数据与冻结协议

- 48 株独立植株，16 扫描/株，768 扫描不视为 768 株。
- 官方 Train/Validation/Test=24/12/12；复用 M3 四折，每折各自 24/12/12，不合并 SSL 数据。
- 每折 P2 仅加载该折 Train 的 384 份 XYZ；Validation 不参与 SSL，Test 不参与该划分 SSL。
- 四折角色按照旧协议轮换；某株可在不同划分扮演不同角色，不能将不同划分看作全局新独立测试。
- 下游 25/50/100% 固定且嵌套地选择 6/12/24 训练植株，保留所选植株全部当前扫描。
- Validation 仅按预定叶数 plant-macro MAE / 分割 plant-macro mIoU 选择下游 epoch。
- SSL 固定第 20 epoch 导出，不按 Test、Validation 标签或某个下游指标挑预训练轮次。
- 所有 2048 点采样、64 FPS patches×32 KNN 点，只由当前 XYZ 决定，语义标签不用于采样/输入。
- 当前完整扫描均值/半径中心化；保存 `center_m/radius_m/extent_m` 可逆恢复。
- 下游同时保留当前物理 XYZ 几何 descriptors，标准化仅拟合当前所选有标签训练植株。
- 不读取未来扫描确定坐标变换；不读取未来器官 ID、时间、状态作为当前输入。
- 分割 taproot/core/leaf 对应标注 0/1/2；2048 点 IoU 不假称全分辨率性能。
- 当前叶数是当前扫描中可见且已标注的叶实例数，不是潜在全部生物学叶数。
- 本次不生成任何 Test 预测；以前官方/fold Test 已使用，仅视为探索性证据。

原锁定 SHA256：`{lock['protocol_sha256']}`。植株预算保存于 `protocol_lock.json`。
一次仅增加 notes 的临时元数据编辑已经按原字节恢复；三个保留原始 hash 的工件由逐字段
验证的 annotation-only alias 审计，详见 `PROTOCOL_PROVENANCE_NOTE.md`；任何实验参数变更仍拒绝。

## 4. M4-A 正确性与训练目标

`m4a_correctness.json` 实测完整主干、2048 点 GPU forward/backward、梯度有限、target 无梯度、
EMA 方程正确、context/target 无交叠、固定种子一致、小样本固定 mask predictor Loss 下降。
自动测试还覆盖完整 optimizer/scheduler/loader/Python/NumPy/Torch/CUDA RNG 精确续训、
冻结 encoder/BatchNorm 不漂移、上下游 shape、严格 checkpoint 加载、植株泄漏与主指标不可事后更换。

{'；'.join(test_summary)}。{restore_note}

算法：局部 patch mini PointNet → spatial greedy sequencing → contiguous target blocks（可互相重叠）→
排除全部 target 的 context block → context Transformer → 带 target position/mask 的 latent Predictor。
Teacher 读取全 token，经 last block functional LayerNorm 生成 stop-gradient latent targets；
仅 context Transformer 以 0.9995→0.99999 EMA 更新 teacher，符合官方共享 tokenizer 的设计。
损失是 SmoothL1 latent matching，完全不重建原始 XYZ/像素。

EMA、非对称 mask/predictor 与 target normalization 提供官方 anti-collapse 机制，但并非理论保证。
另监控 token 各维 std、pooled std、cosine quantile、effective/algebraic rank，且不把低 Loss 当成功。

分割首次 CUDA weighted cross-entropy 遇到 deterministic NLL2D 限制，已用完全等价的点展平版本修复，
没有关闭 determinism。完整回归在旧 `.venv-m1` 通过；M4 环境全套旧 DINOv2 测试因没有 torchvision
失败，这属于隔离环境依赖边界而不是原环境被损坏。

## 5. 当前可见叶片数：主要证据

所有 M4 count 模型使用相同的 `Linear(mean/max encoder tokens + lawful current geometry → count)` head。
P0/P2 每个 seed 的 head 初值、可训练参数、步数、所选训练植株均逐项核对，公平性：**{fairness['status']}**。
冻结 Linear Probe 和全量 fine-tuning 分开统计，不与外部预训练混合。
P0 与 P2 各自同样 20 epochs fine-tuning / 120 epochs cached frozen probe，不调预算追某个方法。

官方验证集：

{markdown_table(primary_table)}

认真比较的既有强基线（原预测文件只读；原 50-epoch 多表型监督，不是匹配容量的消融）：

{markdown_table(baseline_table)}

四折/官方配对方向，positive absolute improvement 表示 P2 更好：

{markdown_table(fold_table)}

独立评估单位是植株，MAE=先每株所有当前扫描平均，再植株平均。
RMSE 为 per-plant RMSE 的 macro mean，同时 CSV 单列提供 pooled RMSE，不混为同一指标。
CI 为官方验证植株的 seed-averaged paired error bootstrap，10,000 次；不是扫描重采样。
`cross_fold_validation.csv` 对 4 个不同 validation folds 的 48 株进行每株一次的汇总；
官方与 folds 不能拼成 60 个不同独立植株。

## 6. 少标注量与冻结/微调

下表均为 5 个划分 × 3 个 seed 的描述性均值/SD；不是 15 份独立生物学重复。
同一比例同一划分完全相同的植物预算，未用同一植株部分扫描冒充更少植株。
列 runs 明确当前真实完成量。

{markdown_table(efficiency_table)}

{efficiency_comment}

具体每折/每种子结果在 `label_efficiency.csv`；不能凭某个少标注比例最好而重定义主指标。
Frozen count probe 的 encoder/BN 不更新；feature/target scalers 仅从该预算 Train 拟合并保存在 checkpoint。

## 7. 植物器官语义分割

完整 64 patch tokens 经 3-nearest inverse-distance interpolation 映射回点，拼接当前 normalized XYZ
与 local offset 后用 pointwise MLP。没有 GT mask/instance ID/organ morphology 作为输入。
Frozen 模式缓存实际 encoder 点特征，仅训练轻量 head；fine-tune 更新全部 encoder 与相同 head。
另提供监督 PointNet pointwise-local + max-global baseline；并没有假装完成 PointNet++。
训练类权重只计算 Train 标签频率；validation 类 IoU、Macro F1 与各株结果完整保存。

{markdown_table(semantic_table)}

这是**语义** taproot/core/leaf 分割，不是叶实例分割。mIoU 不能作为叶实例 PQ/SBD。
模型确实保留局部 patch feature、可点级映射，因此具备做 instance decoder 的工程基础；
但尚未用实例监督/聚类验证同叶凝聚与不同叶分离，不能声称已能可靠区分叶实例。
实例分割、PQ/SBD/器官形态提取明确留作后续，不阻塞本阶段、也不自动启动 temporal 学习。

## 8. 坍塌诊断、训练成本与失败案例

监控只使用固定前 96 个 Train 扫描，其组成不等价于 96 独立植株；rank 上限为 95。
各 token 维 std 完整数组与 cosine 0/5/50/95/100% quantiles 在 JSON。

{markdown_table(diagnosis)}

事后只用 **全部 384 Train scans/24 plants** 的 P0/P2 对比（不用于重新选择模型）：

{markdown_table(full_summary, float_format='.6f')}

P2 的维度 std 和有效秩相对随机初始化收缩，global embedding 高度相似，提示表示多样性降低。
即便没有低于 `token std <0.01` 的硬警戒，也不能声称“没有坍塌所以已学到有效表示”；
须结合主任务和分割收益判断。直方图、pooled/token 各维 std、全部 cosine quantiles 保存于
`representation_full_train_diagnostics.json`，可复算而不是仅展示选中的低相似样本。

低有效秩/高 token cosine 即便未触发硬阈值，也不能解释为已学到优质器官结构。
SSL 全部实测累计 {ssl.seconds.sum():.1f} 秒、{int(ssl.optimizer_steps.sum())} 步；
下游实测累计 {resources.seconds.sum():.1f} 秒、{int(resources.optimizer_steps.sum())} 步。
checkpoint final-state deterministic repair 重复运行另耗 {replay_seconds:.1f} 秒，不列为独立科学实验。
这里是各 run 墙钟耗时之和，不是共享 GPU 下独占计算量；包含必要诊断/部分保存成本。
SSL 最大 allocated VRAM {ssl.peak_cuda_bytes.max() / 2**30:.2f} GiB；
下游最大 allocated VRAM {resources.peak_cuda_bytes.max() / 2**30:.2f} GiB。
精确参数量、steps、labelled/SSL scans、exposures、秒数和显存见 resources CSV 及逐 run JSON。

阶段 MAE（锁定主任务；93 DAS 原阶段范围之外，单列保留）：

{markdown_table(stage)}

`failure_cases.csv` 给出不同验证植株/阶段的 worst 当前 count error；语义图显示 median/worst 真实 scan。
图中错误可以显示叶/芯部/主根混淆，**不能从形状单独推断未测量的生物学原因**。
未见植物有限、leaf/core 局部边界、稀少 taproot、对象归一化与 plant 领域差异、预训练时间短均是
明确的研究限制，而不是已经验证的单一因果解释。

## 9. 锁定标准逐项判定

主指标固定为 100% 标签、full fine-tuning、官方 Validation current-count Plant Macro MAE。
要求至少改善 0.2 片且 10%，三官方 seed 同方向、至少三 validation fold 同方向、paired plant CI < 0，
并不比原 PointNet+Geometry 差超过 0.1 片。辅助 segmentation/少标注结果不能救活失败的主标准。

{markdown_table(rule_table)}

最终科学判断：**{decision['decision']}**。P0/P2 实验矩阵状态：
SSL {coverage['ssl_completed']}/{coverage['ssl_expected']}，count {coverage['count_completed']}/{coverage['count_expected']}，
semantic segmentation {coverage['segmentation_completed']}/{coverage['segmentation_expected']}。
**P1 缺失，所以整体工程 PASS 未达到；不制造官方迁移结果。**

原工件保护状态：**{protection['status']}**，保护 {protection['protected_files']} 个文件。
非 raw 文件 SHA256；大体积 raw 数据以 size/nanosecond mtime 核对，未虚称逐 raw SHA256 重审。
未写旧实验文件、未 git commit、未新建 branch。停止于 M4，不自动进入 Temporal-JEPA。

## 10. 交付与重现

```bash
uv venv --python 3.11 .venv-m4
uv pip install --python .venv-m4/bin/python -r requirements-m4.txt
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python -m pytest tests/test_m4.py tests/test_m4_protocol.py -q
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/verify_m4.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/prepare_m4.py --stage cache
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/pretrain_m4.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/train_m4_downstream.py
CUDA_VISIBLE_DEVICES=0 .venv-m4/bin/python scripts/diagnose_m4_representations.py
.venv-m4/bin/python scripts/evaluate_m4.py
.venv-m4/bin/python scripts/prepare_m4.py --stage verify
```

既有锁已存在时 `prepare_m4 --stage lock` 只验证而不重选植物；已有完成模型安全跳过。
SSL `last.pt` 保存 model/EMA/optimizer/scheduler/epoch/step/RNG 可精确续训；下游 `best.pt` 保存选定模型、
配置、train-only 标准化，`training_state.pt` 保存配套 final model/优化器/RNG。早期 smoke 的
state capture 问题通过相同 seed/配置的 deterministic replay 修复；原训练 state 另存，选中
best epoch 和 scientific predictions 必须完全一致后才补存 final state，见 `state_capture_repairs.json`。
没有重写原 best 模型和预测；重复运行成本单列，不混入“新独立实验”。
禁止把 P1 新权重静默插入这次矩阵，应另立明确外部数据设置。

主要工件：`pretrained_comparison.csv`、`label_efficiency.csv`、`segmentation_metrics.csv`、
`paired_comparisons.csv`、`representation_diagnostics.json`、`figures/`、`outputs/m4/checkpoints/`、
`outputs/m4/predictions/`。每张图提供 PNG/SVG/PDF 与 source CSV。

## 信息来源

- Point-JEPA 官方论文/代码：https://openaccess.thecvf.com/content/WACV2025/html/Saito_Point-JEPA_A_Joint_Embedding_Predictive_Architecture_for_Self-Supervised_Learning_on_WACV_2025_paper.html ; https://github.com/Ayumu-J-S/Point-JEPA
- Point-MAE：https://arxiv.org/abs/2203.06604 ; https://github.com/Pang-Yatian/Point-MAE
- Plant-MAE DOI/Crossref（正文不可用）：https://doi.org/10.1016/j.plaphe.2025.100049 ; https://api.crossref.org/works/10.1016/j.plaphe.2025.100049
- Sugar4D V4：https://doi.org/10.60507/FK2/IS8YBZ ; https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ
- PyTorch 2.8 Context7 current docs：`sources/pytorch_context7.txt`。
- 所有指标来自本地实际运行 CSV/JSON/checkpoints，不是训练知识或文献估计。
- 图像内容与视觉排版检查由 Agnes API 识别，记录在 `figure_visual_qa.json`；定量解释以真实 CSV/JSON 为准。
- 博查搜索 HTTP 403，未隐瞒；直接官方资源核验结果与限制见 `UPSTREAM_AUDIT.md`。
"""
    (report / "M4_REPORT.md").write_text(document)
    write_json(report / "report_manifest.json", {"status": "COMPLETE-FEASIBLE" if coverage["p0_p2_matrix_complete"] else "PARTIAL",
                                                "decision": decision["decision"], "primary_not_reselected": True,
                                                "external_weights_missing": True, "engineering_pass": False})
    return document
