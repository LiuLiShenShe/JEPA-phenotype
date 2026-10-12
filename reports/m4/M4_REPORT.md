# M4：基于 Point-JEPA 的植物三维自监督表征学习

日期：2026-10-12。科学判断：**LIMITED-BENEFIT**。

工程状态：**M4-A PASS；P0/P2 全部可行实验完成；整体工程验收 PARTIAL，不冒称 PASS**。
缺项是必需的 P1 官方外部预训练权重不可下载。没有启动 Temporal-JEPA，没有预测未来点云。

## 1. 必须回答的问题

**Point-JEPA 式自监督预训练是否改善了对未见植株的三维器官结构理解，并且能否成为后续植物动态预测的有效 Encoder？**

当前锁定配置下，尚不能证明 Point-JEPA 式预训练稳定改善未见植株的三维器官结构理解；不具备直接作为后续动态预测 Encoder 的充分证据。

官方 Validation 的 P0→P2 平均 MAE 为 1.2772→1.2234；改善 0.0538 片（4.21%），3/3 个种子改善。配对植株 bootstrap 的 P2−P0 MAE 95% CI [-0.1494, 0.0404]。

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

| condition | status | pretraining_data | official_coverage |
| --- | --- | --- | --- |
| P0 | COMPLETED | none | N/A |
| P1 | UNAVAILABLE | ShapeNet external (not obtained) | not measurable |
| P2 | COMPLETED | each partition's 24 Train plants only, from scratch | N/A; no external weights |
| P3 | NOT RUN (optional) | Point-MAE CUDA dependencies/budget not established | N/A |

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

原锁定 SHA256：`8a138270bb64727ed57cfcef2a8f4e0a0685311c151a04dc65c68c94435521c0`。植株预算保存于 `protocol_lock.json`。
一次仅增加 notes 的临时元数据编辑已经按原字节恢复；三个保留原始 hash 的工件由逐字段
验证的 annotation-only alias 审计，详见 `PROTOCOL_PROVENANCE_NOTE.md`；任何实验参数变更仍拒绝。

## 4. M4-A 正确性与训练目标

`m4a_correctness.json` 实测完整主干、2048 点 GPU forward/backward、梯度有限、target 无梯度、
EMA 方程正确、context/target 无交叠、固定种子一致、小样本固定 mask predictor Loss 下降。
自动测试还覆盖完整 optimizer/scheduler/loader/Python/NumPy/Torch/CUDA RNG 精确续训、
冻结 encoder/BatchNorm 不漂移、上下游 shape、严格 checkpoint 加载、植株泄漏与主指标不可事后更换。

`tests_m4.xml`：21 tests，0 failures/errors；`tests_all_original_environment.xml`：129 tests，0 failures/errors。checkpoint 重载校验 255 份，状态 PASS；count 浮点误差容许 2e-6，segmentation predictions/confusions 必须完全一致。

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
P0/P2 每个 seed 的 head 初值、可训练参数、步数、所选训练植株均逐项核对，公平性：**PASS**。
冻结 Linear Probe 和全量 fine-tuning 分开统计，不与外部预训练混合。
P0 与 P2 各自同样 20 epochs fine-tuning / 120 epochs cached frozen probe，不调预算追某个方法。

官方验证集：

| condition | MAE | seed_SD | RMSE | seeds |
| --- | --- | --- | --- | --- |
| P0 | 1.2772 | 0.0561 | 1.6145 | 3 |
| P2 | 1.2234 | 0.0937 | 1.5442 | 3 |

认真比较的既有强基线（原预测文件只读；原 50-epoch 多表型监督，不是匹配容量的消融）：

| condition | MAE | seed_SD | RMSE | seeds |
| --- | --- | --- | --- | --- |
| A1-PointNet | 1.4304 | 0.0940 | 1.7337 | 3 |
| A2-DAS-Ridge | 1.4374 | NA | 1.7138 | 1 |
| A2-Geometry-Ridge | 1.5452 | NA | 1.8452 | 1 |
| A3-PointNet-Geometry | 1.3164 | 0.0739 | 1.6063 | 3 |

四折/官方配对方向，positive absolute improvement 表示 P2 更好：

| partition | p0_mae | p2_mae | absolute_improvement | improving_seeds | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| fold0 | 1.3271 | 1.3756 | -0.0485 | 0 | -0.0588 | 0.1654 |
| fold1 | 1.1458 | 1.2638 | -0.1180 | 1 | 0.0194 | 0.2344 |
| fold2 | 1.3480 | 1.3472 | 0.0008 | 2 | -0.1413 | 0.1301 |
| fold3 | 1.4627 | 1.4780 | -0.0152 | 1 | -0.2215 | 0.2371 |
| official | 1.2772 | 1.2234 | 0.0538 | 3 | -0.1494 | 0.0404 |

独立评估单位是植株，MAE=先每株所有当前扫描平均，再植株平均。
RMSE 为 per-plant RMSE 的 macro mean，同时 CSV 单列提供 pooled RMSE，不混为同一指标。
CI 为官方验证植株的 seed-averaged paired error bootstrap，10,000 次；不是扫描重采样。
`cross_fold_validation.csv` 对 4 个不同 validation folds 的 48 株进行每株一次的汇总；
官方与 folds 不能拼成 60 个不同独立植株。

## 6. 少标注量与冻结/微调

下表均为 5 个划分 × 3 个 seed 的描述性均值/SD；不是 15 份独立生物学重复。
同一比例同一划分完全相同的植物预算，未用同一植株部分扫描冒充更少植株。
列 runs 明确当前真实完成量。

| mode | label_fraction | condition | MAE | run_SD | runs |
| --- | --- | --- | --- | --- | --- |
| finetune | 0.2500 | P0 | 1.9986 | 0.1729 | 15 |
| finetune | 0.2500 | P2 | 1.8593 | 0.1298 | 15 |
| finetune | 0.5000 | P0 | 1.9413 | 0.1726 | 15 |
| finetune | 0.5000 | P2 | 1.5743 | 0.1519 | 15 |
| finetune | 1.0000 | P0 | 1.3122 | 0.1243 | 15 |
| finetune | 1.0000 | P2 | 1.3376 | 0.1107 | 15 |
| probe | 0.2500 | P0 | 1.7579 | 0.2032 | 15 |
| probe | 0.2500 | P2 | 1.6851 | 0.1675 | 15 |
| probe | 0.5000 | P0 | 1.4971 | 0.1750 | 15 |
| probe | 0.5000 | P2 | 1.4637 | 0.1405 | 15 |
| probe | 1.0000 | P0 | 1.3178 | 0.1560 | 15 |
| probe | 1.0000 | P2 | 1.3322 | 0.0947 | 15 |

probe/25%：平均 P0→P2 MAE 1.7579→1.6851，改善方向划分 3/5；probe/50%：平均 P0→P2 MAE 1.4971→1.4637，改善方向划分 3/5；probe/100%：平均 P0→P2 MAE 1.3178→1.3322，改善方向划分 2/5；finetune/25%：平均 P0→P2 MAE 1.9986→1.8593，改善方向划分 5/5；finetune/50%：平均 P0→P2 MAE 1.9413→1.5743，改善方向划分 5/5；finetune/100%：平均 P0→P2 MAE 1.3122→1.3376，改善方向划分 2/5。这些是预定辅助证据，不能替代失败的主任务门槛。

具体每折/每种子结果在 `label_efficiency.csv`；不能凭某个少标注比例最好而重定义主指标。
Frozen count probe 的 encoder/BN 不更新；feature/target scalers 仅从该预算 Train 拟合并保存在 checkpoint。

## 7. 植物器官语义分割

完整 64 patch tokens 经 3-nearest inverse-distance interpolation 映射回点，拼接当前 normalized XYZ
与 local offset 后用 pointwise MLP。没有 GT mask/instance ID/organ morphology 作为输入。
Frozen 模式缓存实际 encoder 点特征，仅训练轻量 head；fine-tune 更新全部 encoder 与相同 head。
另提供监督 PointNet pointwise-local + max-global baseline；并没有假装完成 PointNet++。
训练类权重只计算 Train 标签频率；validation 类 IoU、Macro F1 与各株结果完整保存。

| mode | condition | mIoU | run_SD | Macro_F1 | taproot_IoU | core_IoU | leaf_IoU | runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| finetune | P0 | 0.5539 | 0.0219 | 0.6577 | 0.3407 | 0.3446 | 0.9763 | 15 |
| finetune | P2 | 0.4714 | 0.0188 | 0.5616 | 0.2102 | 0.2367 | 0.9672 | 15 |
| finetune | PointNet | 0.4364 | 0.0149 | 0.4927 | 0.0029 | 0.3077 | 0.9769 | 15 |
| probe | P0 | 0.4547 | 0.0065 | 0.5258 | 0.0729 | 0.3046 | 0.9749 | 15 |
| probe | P2 | 0.4424 | 0.0149 | 0.5062 | 0.0357 | 0.2976 | 0.9751 | 15 |

这是**语义** taproot/core/leaf 分割，不是叶实例分割。mIoU 不能作为叶实例 PQ/SBD。
模型确实保留局部 patch feature、可点级映射，因此具备做 instance decoder 的工程基础；
但尚未用实例监督/聚类验证同叶凝聚与不同叶分离，不能声称已能可靠区分叶实例。
实例分割、PQ/SBD/器官形态提取明确留作后续，不阻塞本阶段、也不自动启动 temporal 学习。

## 8. 坍塌诊断、训练成本与失败案例

监控只使用固定前 96 个 Train 扫描，其组成不等价于 96 独立植株；rank 上限为 95。
各 token 维 std 完整数组与 cosine 0/5/50/95/100% quantiles 在 JSON。

| partition_seed | token_std | embedding_std | effective_rank | algebraic_rank | cosine_median | collapse_warning |
| --- | --- | --- | --- | --- | --- | --- |
| fold0_seed17 | 0.0426 | 0.0173 | 11.1740 | 95 | 0.9984 | False |
| fold0_seed42 | 0.0402 | 0.0158 | 12.3165 | 95 | 0.9986 | False |
| fold0_seed93 | 0.0516 | 0.0209 | 8.3834 | 95 | 0.9977 | False |
| fold1_seed17 | 0.0405 | 0.0158 | 13.1926 | 95 | 0.9986 | False |
| fold1_seed42 | 0.0406 | 0.0152 | 15.9574 | 95 | 0.9985 | False |
| fold1_seed93 | 0.0500 | 0.0200 | 8.9250 | 95 | 0.9979 | False |
| fold2_seed17 | 0.0417 | 0.0164 | 12.4675 | 95 | 0.9985 | False |
| fold2_seed42 | 0.0405 | 0.0151 | 15.7848 | 95 | 0.9985 | False |
| fold2_seed93 | 0.0502 | 0.0198 | 9.5765 | 95 | 0.9978 | False |
| fold3_seed17 | 0.0413 | 0.0165 | 10.4501 | 95 | 0.9986 | False |
| fold3_seed42 | 0.0405 | 0.0157 | 12.0362 | 95 | 0.9986 | False |
| fold3_seed93 | 0.0499 | 0.0203 | 8.6456 | 95 | 0.9979 | False |
| official_seed17 | 0.0414 | 0.0148 | 15.2603 | 95 | 0.9985 | False |
| official_seed42 | 0.0407 | 0.0145 | 16.8050 | 95 | 0.9985 | False |
| official_seed93 | 0.0494 | 0.0180 | 11.1266 | 95 | 0.9978 | False |

事后只用 **全部 384 Train scans/24 plants** 的 P0/P2 对比（不用于重新选择模型）：

| condition | token_std | embedding_std | effective_rank | global_cosine_median | token_cosine_median |
| --- | --- | --- | --- | --- | --- |
| P0 | 0.322787 | 0.088868 | 21.155263 | 0.990243 | 0.900009 |
| P2 | 0.043488 | 0.017694 | 15.479784 | 0.999635 | 0.998354 |

P2 的维度 std 和有效秩相对随机初始化收缩，global embedding 高度相似，提示表示多样性降低。
即便没有低于 `token std <0.01` 的硬警戒，也不能声称“没有坍塌所以已学到有效表示”；
须结合主任务和分割收益判断。直方图、pooled/token 各维 std、全部 cosine quantiles 保存于
`representation_full_train_diagnostics.json`，可复算而不是仅展示选中的低相似样本。

低有效秩/高 token cosine 即便未触发硬阈值，也不能解释为已学到优质器官结构。
SSL 全部实测累计 1355.6 秒、1800 步；
下游实测累计 4282.1 秒、73200 步。
checkpoint final-state deterministic repair 重复运行另耗 375.2 秒，不列为独立科学实验。
这里是各 run 墙钟耗时之和，不是共享 GPU 下独占计算量；包含必要诊断/部分保存成本。
SSL 最大 allocated VRAM 3.75 GiB；
下游最大 allocated VRAM 4.52 GiB。
精确参数量、steps、labelled/SSL scans、exposures、秒数和显存见 resources CSV 及逐 run JSON。

阶段 MAE（锁定主任务；93 DAS 原阶段范围之外，单列保留）：

| stage | P0 | P2 |
| --- | --- | --- |
| early | 1.0169 | 0.9915 |
| late | 1.5762 | 1.6645 |
| middle | 1.1479 | 1.1476 |
| outside_registered_stages_DAS93 | 2.4206 | 2.5403 |

`failure_cases.csv` 给出不同验证植株/阶段的 worst 当前 count error；语义图显示 median/worst 真实 scan。
图中错误可以显示叶/芯部/主根混淆，**不能从形状单独推断未测量的生物学原因**。
未见植物有限、leaf/core 局部边界、稀少 taproot、对象归一化与 plant 领域差异、预训练时间短均是
明确的研究限制，而不是已经验证的单一因果解释。

## 9. 锁定标准逐项判定

主指标固定为 100% 标签、full fine-tuning、官方 Validation current-count Plant Macro MAE。
要求至少改善 0.2 片且 10%，三官方 seed 同方向、至少三 validation fold 同方向、paired plant CI < 0，
并不比原 PointNet+Geometry 差超过 0.1 片。辅助 segmentation/少标注结果不能救活失败的主标准。

| locked_rule | passes |
| --- | --- |
| primary_complete_three_seeds | True |
| absolute_threshold | False |
| relative_threshold | False |
| all_official_seeds_improve | True |
| plant_paired_ci | False |
| at_least_three_validation_folds_improve | False |
| strong_baseline_noninferiority | True |
| no_collapse_warning | True |
| paired_fairness | True |
| complete_p0_p2_matrix | True |

最终科学判断：**LIMITED-BENEFIT**。P0/P2 实验矩阵状态：
SSL 15/15，count 180/180，
semantic segmentation 75/75。
**P1 缺失，所以整体工程 PASS 未达到；不制造官方迁移结果。**

原工件保护状态：**PASS**，保护 42362 个文件。
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
