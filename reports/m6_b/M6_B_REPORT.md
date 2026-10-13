# M6-B：Sugar4D 监督式叶片实例分割与下游追踪

## 0. 唯一研究建议

**INSTANCE-METHOD-NEEDED**。A converged bounded supervised baseline improves independent plants stably but fails locked gates with low PQ/early-small recall or predominantly instance-missed first observations; a tracking-only failure is not attributed to segmentation. No dense strong-baseline superiority is claimed.

本阶段已实际训练 S1/S2 各三个种子、S3 一个种子，全部60轮、每轮384训练帧。
S4 **BLOCKED**，没有伪造 Organ3DNet 推理/训练指标。没有训练JEPA、生成模型、追踪器或动态预测网络。
Validation为模型选择/继续判断主集合；旧Test仅探索性固定方案复核，不是未使用盲测。
当前长度/面积测量器明显有偏；本建议不认证整套表型系统。
当前固定B3=S2 seed17：Validation PQ=0.8692、IDF1=0.8496均超过原门槛，首观测F1=0.3138仍低于.70。
Validation共107次新叶首观测，55次无合格当前帧实例匹配，其中51次属于Train阈值定义的小叶；另14次已匹配叶被接到旧身份。芯部邻近代理的植株宏召回仅0.3092（11株），该失败类型支持针对性实例研究，但不证明整套主干架构无效或优于未完成的Organ3DNet。
这55次首帧漏检中，合并代理46次、分裂代理26次（可重叠），4096点采样完全漏掉该叶的次数为0；不是简单把未采到点误称为网络失败。

## 1. 冻结协议、公平性与数据

- 协议 `configs/m6_b/protocol.json` SHA256 `39a31f73de48568998defea76715f538b9caf77a2cc4164e501848be2698ccd9`。
- 锁定时间在 `protocol_lock.json`，早于真实目标域训练。原官方24/12/12整株划分不改动；每株16个DAS：37,40,44,47,51,54,58,61,65,72,75,79,82,86,89,93。
- 主指标：全分辨率逐帧 strict IoU>0.5 PQ，先按植物平均，再植物宏平均；不是把192验证扫描当192株。所有植株、合法叶片都保留。
- CE监督taproot/core/leaf三类；负实例仅作非叶实例哨兵，叶0是合法叶。当前叶实例每帧连续重映射、每轮随机置换；模型输入只有XYZ。
- 无跨时间GT身份、未来扫描、真叶数、GT选点、Validation/Test训练。训练24株384帧不变，4096点，固定32768训练池、逐轮重采样、z旋转，中心/尺度/原始索引可恢复到米。
- S0完整沿用M6-A：原checkpoint、单位球归一化、MeanShift .6、源物种均值过滤和预测缓存，历史PQ/SBD/追踪不覆盖。M6-A官方归一化诊断亦保留，不用来择优替代S0。
- S1/S2相同官方PlantNet主干、同监督量/学习率/损失/验证时点/后处理网格。S2只随机三类语义输出头，加载其它全部兼容状态且全解冻。S1不加载任何源物种权重。
- S1/S2有目标域标签、S0无目标域标签：提升首先是监督适配收益，不是架构创新。目标归一化、叶-only损失接口、后处理也是公开的适配变化，不孤立归因于权重微调。
- Seed17固定作下游，不选最好种子。B3方法族按三种子Validation PQ平均选择为 **S2**；B4为可运行PSegNet seed17，不能冒称已复现密集查询Organ3DNet。

## 2. 实际训练和加载审计

独立`.venv-m6-b`，Python3.11/Torch2.8+cu128，历史环境不改。实际批4/4096点前反向和Adam烟测通过。
最初只用GPU0；GPU1原任务结束并核验空闲后，S2用相同A6000 GPU1并发。共享GPU1后来出现外部作业，S2在完整第37轮保存边界迁回GPU0，从模型、Adam和RNG状态精确恢复；每run仍只有5760更新。只改资源调度，数据/种子/预算/参数不变；CLI逐run锁防重复，实际控制台日志分`train_S2_*_gpu1.log`及`train_S2_*.log`。共享主机非独占，见`RESOURCE_SCHEDULING.md`和`S2_device_migration.json`。
统一Adam lr.001、weight_decay.001、60轮余弦名义下限.00002（最后实际使用约.00002067，逐更新lr有完整记录）、FP32、clip10，全监督帧每轮一遍；每个run5760更新/23040帧呈现。
监督来源为24株384帧、3419次当前帧叶实例观测；每run训练输入点标签呈现94,371,840次（384×60×4096），不是这些数量的独立植物。每帧训练池取min(32768,该PLY点数)统一随机点，共12,555,297点（2帧原云较小），池覆盖全部合法Train叶；完整Train原始点标签共150,776,654点，不声称每轮用到了完整PLY的每个标签点。
Validation每10轮在完整12株192帧计算PQ择checkpoint，后处理中带宽.4/.6/.8只看Validation。
保留官方CE×10、L1 embedding×10及double-hinge相似性；core/taproot分开CE、实例项仅叶。
特别披露：官方同实例相似性项为relu(80-D)，没有悄悄改符号；局部ID权重改为逐帧，避免批内不同植物身份混同。
原官方预算200轮，本轮60轮是有界目标监督验证；不因Test好坏延长预算。

| method | seed | parameters | epochs | updates | frame_presentations | loaded_entries | random_entries | selected_epoch | bandwidth | train_seconds | first10_loss | last10_loss | last10_discriminative | validation_final_minus_50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 17 | 4442978 | 60 | 5760 | 23040 | 0 | 266 | 50 | 0.6000 | 11161.9983 | 11.4141 | 1.6483 | 0.1335 | -0.0023 |
| S1 | 29 | 4442978 | 60 | 5760 | 23040 | 0 | 266 | 60 | 0.8000 | 11178.0773 | 7.2807 | 1.5536 | 0.1246 | 0.0078 |
| S1 | 43 | 4442978 | 60 | 5760 | 23040 | 0 | 266 | 50 | 0.8000 | 11188.6827 | 8.0873 | 1.5454 | 0.1239 | -0.0090 |
| S2 | 17 | 4442978 | 60 | 5760 | 23040 | 264 | 2 | 60 | 0.8000 | 11072.6968 | 3.4260 | 1.3027 | 0.1026 | 0.0058 |
| S2 | 29 | 4442978 | 60 | 5760 | 23040 | 264 | 2 | 50 | 0.8000 | 11069.8834 | 3.6892 | 1.3149 | 0.1034 | -0.0004 |
| S2 | 43 | 4442978 | 60 | 5760 | 23040 | 264 | 2 | 50 | 0.8000 | 11067.4390 | 3.5627 | 1.3275 | 0.1049 | -0.0053 |
| S3 | 17 | 3339824 | 60 | 5760 | 23040 | 0 | 364 | 60 | 0.6000 | 10213.2086 | 5.7055 | 1.6652 | 0.1314 | 0.0027 |

逐run成功/随机状态键、冻结列表、源checkpoint及源码hash在`outputs/m6_b/S*/seed_*/training_manifest.json`；逐更新`batches.csv`、逐轮`epochs.jsonl`、最后/验证轮/最佳checkpoint均保留。
外部预训练源烟草/番茄/高粱（源码报告546点云，不可当546独立植株）；S1/S3无外部预训练。
训练预算充分性诊断：True。末轮-v50 PQ和末10轮embedding loss逐run公开，不能把不收敛或预算不足证明为架构缺陷。

## 3. Validation主结果与探索性Test

| split | method | seed | pq | sbd | precision | recall | miss_rate | merge_rate | split_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | S0 | 0 | 0.4980 | 0.6188 | 0.6307 | 0.5574 | 0.4426 | 0.3367 | 0.2705 |
| test | S1 | 17 | 0.8672 | 0.8685 | 0.9109 | 0.9068 | 0.0932 | 0.1179 | 0.0622 |
| test | S2 | 17 | 0.8782 | 0.8826 | 0.9511 | 0.8884 | 0.1116 | 0.1343 | 0.0526 |
| test | S3 | 17 | 0.8616 | 0.8562 | 0.8817 | 0.9237 | 0.0763 | 0.0818 | 0.0465 |
| val | S0 | 0 | 0.5002 | 0.6382 | 0.6425 | 0.5932 | 0.4068 | 0.3564 | 0.3124 |
| val | S1 | 17 | 0.8562 | 0.8729 | 0.9041 | 0.9142 | 0.0858 | 0.1157 | 0.0792 |
| val | S1 | 29 | 0.8572 | 0.8763 | 0.9581 | 0.8684 | 0.1316 | 0.1791 | 0.0697 |
| val | S1 | 43 | 0.8538 | 0.8728 | 0.9619 | 0.8660 | 0.1340 | 0.1831 | 0.0704 |
| val | S2 | 17 | 0.8692 | 0.8883 | 0.9585 | 0.8928 | 0.1072 | 0.1482 | 0.0672 |
| val | S2 | 29 | 0.8698 | 0.8909 | 0.9508 | 0.8977 | 0.1023 | 0.1432 | 0.0712 |
| val | S2 | 43 | 0.8650 | 0.8831 | 0.9593 | 0.8822 | 0.1178 | 0.1586 | 0.0655 |
| val | S3 | 17 | 0.8545 | 0.8570 | 0.8872 | 0.9300 | 0.0700 | 0.0944 | 0.0658 |

Plant seed-mean PQ及植物95%bootstrap区间（三个种子不是36个独立植物）：
Validation的S1/S2各包含3个种子；Test仅seed17固定方案复核，S3和S0仅一个固定模型。表中平均只覆盖实际存在的seed，不把Test一seed当三seed稳健性证据。

| method | split | aggregation | mean | ci95_low | ci95_high | plants |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | test | plant means after seed averaging | 0.4980 | 0.4679 | 0.5270 | 12 |
| S0 | val | plant means after seed averaging | 0.5002 | 0.4665 | 0.5322 | 12 |
| S1 | test | plant means after seed averaging | 0.8672 | 0.8490 | 0.8835 | 12 |
| S1 | val | plant means after seed averaging | 0.8557 | 0.8331 | 0.8777 | 12 |
| S2 | test | plant means after seed averaging | 0.8782 | 0.8577 | 0.8978 | 12 |
| S2 | val | plant means after seed averaging | 0.8680 | 0.8453 | 0.8891 | 12 |
| S3 | test | plant means after seed averaging | 0.8616 | 0.8387 | 0.8826 | 12 |
| S3 | val | plant means after seed averaging | 0.8545 | 0.8336 | 0.8732 | 12 |

Validation配对植物PQ差值（mean_seeds亦先按植物平均）：

| method | seed | comparison | mean | ci95_low | ci95_high | plants | improved_plants |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 17 | S1-S0 | 0.3560 | 0.3361 | 0.3775 | 12 | 12 |
| S1 | 29 | S1-S0 | 0.3570 | 0.3345 | 0.3781 | 12 | 12 |
| S1 | 43 | S1-S0 | 0.3536 | 0.3340 | 0.3727 | 12 | 12 |
| S2 | 17 | S2-S0 | 0.3690 | 0.3467 | 0.3905 | 12 | 12 |
| S2 | 29 | S2-S0 | 0.3695 | 0.3486 | 0.3899 | 12 | 12 |
| S2 | 43 | S2-S0 | 0.3648 | 0.3411 | 0.3867 | 12 | 12 |
| S3 | 17 | S3-S0 | 0.3543 | 0.3300 | 0.3803 | 12 | 12 |
| S1 | mean_seeds | S1-S0 | 0.3555 | 0.3357 | 0.3748 | 12 | 12 |
| S2 | mean_seeds | S2-S0 | 0.3678 | 0.3458 | 0.3886 | 12 | 12 |
| S2 | mean_seeds | S2-S1 | 0.0123 | 0.0046 | 0.0199 | 12 | 10 |

全部逐株/逐帧、各指标区间、不同DAS及叶数条件在`segmentation_per_plant.csv`、`segmentation_frame_metrics.csv`、`segmentation_summary.csv`、`pq_by_das.csv`、`pq_by_leaf_count.csv`。
均为保留全部合法叶片的完整PLY点评价；恢复后污染背景也计入预测实例并影响IoU。

### 实际推理速度与显存

| method | seed | forward_seconds_mean | cluster_restore_seconds_mean | frames_per_second_total | peak_vram_GB |
| --- | --- | --- | --- | --- | --- |
| S1 | 17 | 1.1242 | 0.2071 | 0.7512 | 0.3194 |
| S1 | 29 | 1.1493 | 0.1836 | 0.7502 | 0.3194 |
| S1 | 43 | 1.1202 | 0.1773 | 0.7707 | 0.3194 |
| S2 | 17 | 0.6406 | 0.1801 | 1.2185 | 0.3194 |
| S2 | 29 | 0.6155 | 0.1821 | 1.2537 | 0.3194 |
| S2 | 43 | 0.6333 | 0.1818 | 1.2269 | 0.3194 |
| S3 | 17 | 0.3074 | 0.2006 | 1.9683 | 0.3017 |

以上逐帧时长为实际共享GPU推理，forward不含PLY加载，cluster_restore含MeanShift和全分辨率最近邻及评分开销；实际推理加载了带优化器的训练checkpoint，分配峰值不能冒称最小部署显存。与同时发生的训练/其他任务存在竞争，不作公平架构速度排名。
单设备、同一真实植物DAS37/79、4096点、3次预热10次前向重复（checkpointCPU加载，不把optimizer放GPU）：

| method | DAS | forward_seconds_median | frames_per_second_forward | peak_allocated_vram_GB | exact_repeat_logits_embeddings |
| --- | --- | --- | --- | --- | --- |
| S1 | 37 | 0.3208 | 3.0479 | 0.3090 | True |
| S1 | 79 | 0.3145 | 3.1241 | 0.3090 | True |
| S2 | 37 | 0.3143 | 3.1180 | 0.3090 | True |
| S2 | 79 | 0.3181 | 3.1258 | 0.3090 | True |
| S3 | 37 | 0.3003 | 3.2692 | 0.3044 | True |
| S3 | 79 | 0.3002 | 3.2202 | 0.3044 | True |

microbenchmark仅forward，不含I/O/聚类/回传；记录共享主机GPU快照并检查logits/embedding逐次精确一致，不宣称独占GPU性能。

## 4. 明确实例错误与早期小叶

漏检：无IoU>0.5匹配；合并：一预测实例覆盖多个GT叶各>10%；分裂：一GT叶被多个预测实例覆盖各>10%。
漏检/分裂分母是GT实例，合并分母是预测实例；逐株算比例再平均。小叶阈值仅取Train四分位数0.0458868，early DAS≤47。
core-near：GT叶距离GT core的p10≤5mm，是可复核的评价代理，不是人工确认的“芯部未分离”标签。不存在core的帧距离未知，不删除叶片。

| split | method | seed | category | recall | split_rate | merge_rate | sampling_loss | independent_plants | leaf_observations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | S0 | 0 | core_adjacent | 0.0000 | 0.3661 | 0.7069 | 0.0000 | 12 | 70 |
| test | S0 | 0 | early | 0.6725 | 0.3184 | 0.4704 | 0.0000 | 12 | 272 |
| test | S0 | 0 | early_small | 0.0926 | 0.3148 | 0.7407 | 0.0000 | 9 | 18 |
| test | S0 | 0 | small | 0.1408 | 0.3045 | 0.6764 | 0.0000 | 12 | 499 |
| test | S1 | 17 | core_adjacent | 0.4411 | 0.2109 | 0.5399 | 0.0000 | 12 | 70 |
| test | S1 | 17 | early | 0.9771 | 0.0245 | 0.0463 | 0.0000 | 12 | 272 |
| test | S1 | 17 | early_small | 0.8889 | 0.0556 | 0.1111 | 0.0000 | 9 | 18 |
| test | S1 | 17 | small | 0.6991 | 0.1640 | 0.4698 | 0.0000 | 12 | 499 |
| test | S2 | 17 | core_adjacent | 0.3886 | 0.1731 | 0.5799 | 0.0000 | 12 | 70 |
| test | S2 | 17 | early | 0.9684 | 0.0273 | 0.0760 | 0.0000 | 12 | 272 |
| test | S2 | 17 | early_small | 0.7407 | 0.1481 | 0.2963 | 0.0000 | 9 | 18 |
| test | S2 | 17 | small | 0.6392 | 0.1445 | 0.5131 | 0.0000 | 12 | 499 |
| test | S3 | 17 | core_adjacent | 0.5283 | 0.1673 | 0.5383 | 0.0000 | 12 | 70 |
| test | S3 | 17 | early | 0.9800 | 0.0381 | 0.0466 | 0.0000 | 12 | 272 |
| test | S3 | 17 | early_small | 0.8519 | 0.0556 | 0.1667 | 0.0000 | 9 | 18 |
| test | S3 | 17 | small | 0.7486 | 0.1141 | 0.3636 | 0.0000 | 12 | 499 |
| val | S0 | 0 | core_adjacent | 0.0368 | 0.4017 | 0.8530 | 0.0000 | 11 | 48 |
| val | S0 | 0 | early | 0.6302 | 0.3660 | 0.5385 | 0.0000 | 12 | 266 |
| val | S0 | 0 | early_small | 0.0455 | 0.2348 | 0.7652 | 0.0000 | 11 | 23 |
| val | S0 | 0 | small | 0.1057 | 0.3646 | 0.7694 | 0.0000 | 12 | 345 |
| val | S1 | 17 | core_adjacent | 0.5955 | 0.1898 | 0.4219 | 0.0000 | 11 | 48 |
| val | S1 | 17 | early | 0.9407 | 0.0298 | 0.0815 | 0.0000 | 12 | 266 |
| val | S1 | 17 | early_small | 0.7879 | 0.0000 | 0.1970 | 0.0000 | 11 | 23 |
| val | S1 | 17 | small | 0.6949 | 0.2237 | 0.4555 | 0.0000 | 12 | 345 |
| val | S1 | 29 | core_adjacent | 0.4461 | 0.2767 | 0.4435 | 0.0000 | 11 | 48 |
| val | S1 | 29 | early | 0.9075 | 0.0293 | 0.1563 | 0.0000 | 12 | 266 |
| val | S1 | 29 | early_small | 0.5152 | 0.1742 | 0.5076 | 0.0000 | 11 | 23 |
| val | S1 | 29 | small | 0.5391 | 0.2779 | 0.6062 | 0.0000 | 12 | 345 |
| val | S1 | 43 | core_adjacent | 0.4290 | 0.3474 | 0.5688 | 0.0000 | 11 | 48 |
| val | S1 | 43 | early | 0.9242 | 0.0167 | 0.1580 | 0.0000 | 12 | 266 |
| val | S1 | 43 | early_small | 0.6515 | 0.1212 | 0.3485 | 0.0000 | 11 | 23 |
| val | S1 | 43 | small | 0.5609 | 0.2451 | 0.5862 | 0.0000 | 12 | 345 |
| val | S2 | 17 | core_adjacent | 0.3092 | 0.1995 | 0.4689 | 0.0000 | 11 | 48 |
| val | S2 | 17 | early | 0.9213 | 0.0166 | 0.1227 | 0.0000 | 12 | 266 |
| val | S2 | 17 | early_small | 0.6364 | 0.1212 | 0.3182 | 0.0000 | 11 | 23 |
| val | S2 | 17 | small | 0.6227 | 0.2380 | 0.5111 | 0.0000 | 12 | 345 |
| val | S2 | 29 | core_adjacent | 0.4172 | 0.1414 | 0.4078 | 0.0000 | 11 | 48 |
| val | S2 | 29 | early | 0.9181 | 0.0195 | 0.1279 | 0.0000 | 12 | 266 |
| val | S2 | 29 | early_small | 0.5606 | 0.0909 | 0.2576 | 0.0000 | 11 | 23 |
| val | S2 | 29 | small | 0.6203 | 0.2308 | 0.5089 | 0.0000 | 12 | 345 |
| val | S2 | 43 | core_adjacent | 0.3411 | 0.2287 | 0.4608 | 0.0000 | 11 | 48 |
| val | S2 | 43 | early | 0.9055 | 0.0184 | 0.1387 | 0.0000 | 12 | 266 |
| val | S2 | 43 | early_small | 0.5000 | 0.1439 | 0.3485 | 0.0000 | 11 | 23 |
| val | S2 | 43 | small | 0.5823 | 0.2348 | 0.5358 | 0.0000 | 12 | 345 |
| val | S3 | 17 | core_adjacent | 0.5038 | 0.1465 | 0.3631 | 0.0000 | 11 | 48 |
| val | S3 | 17 | early | 0.9366 | 0.0349 | 0.0893 | 0.0000 | 12 | 266 |
| val | S3 | 17 | early_small | 0.6667 | 0.0000 | 0.1667 | 0.0000 | 11 | 23 |
| val | S3 | 17 | small | 0.7421 | 0.1934 | 0.3969 | 0.0000 | 12 | 345 |

`segmentation_leaf_errors.csv`保留每一个合法叶的漏检、分裂、合并、采样丢失、proxy标签及IoU；不凭预测困难删除它们。`error_strata_summary.csv`报告各层植物级CI、实际独立植物数和叶观测数，观测数不是独立重复数。

## 5. 固定TrackPlant3D传播验证

每帧预测实例独立随机重新编号后才进入追踪器。保留M6-A背景0匹配、256点3DEPS、CPD及官方扩展匹配代价，不重训、不重调追踪参数。
重建同一XYZ采样索引、核验与M6-A缓存normalized XYZ逐点精确相等，复用label-independent CPD变形。
GT时序ID只在评价器中作IoU关联/IDF1全局对齐，不输入追踪器。

| split | pipeline | method | pq | idf1 | id_switches | continuity | new_event_f1 | length_change_tracked_coverage | area_change_tracked_coverage | length_change_tracked_severe | area_change_tracked_severe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | B0 | GT | 1.0000 | 0.9756 | 1.3333 | 0.9840 | 0.9003 | 0.9944 | 0.9944 | 0.0062 | 0.0407 |
| test | B1 | S0 | 0.4980 | 0.5319 | 4.2500 | 0.3977 | 0.0088 | 0.5627 | 0.5627 | 0.4680 | 0.6179 |
| test | B3 | S2 | 0.8782 | 0.8507 | 3.8333 | 0.7390 | 0.3238 | 0.8800 | 0.8800 | 0.1233 | 0.1962 |
| test | B4 | S3 | 0.8616 | 0.8584 | 2.5000 | 0.8191 | 0.2362 | 0.9206 | 0.9206 | 0.0835 | 0.1487 |
| val | B0 | GT | 1.0000 | 0.9590 | 1.7500 | 0.9707 | 0.8349 | 0.9910 | 0.9910 | 0.0111 | 0.0598 |
| val | B1 | S0 | 0.5002 | 0.5408 | 4.6667 | 0.4187 | 0.0069 | 0.5926 | 0.5926 | 0.4434 | 0.5934 |
| val | B3 | S2 | 0.8692 | 0.8496 | 3.5833 | 0.7622 | 0.3138 | 0.8816 | 0.8816 | 0.1205 | 0.1968 |
| val | B4 | S3 | 0.8545 | 0.8462 | 3.0833 | 0.8351 | 0.1864 | 0.9243 | 0.9243 | 0.0782 | 0.1573 |

`tracking_summary.csv`给植物CI，`tracking_paired_intervals.csv`给B3/B4相对B0/B1配对CI。
`tracking_error_decomposition.csv`独立统计 instance_missed、sampling_or_identity_lost、wrong_identity及correct。
真实首次观测为GT器官首个实际扫描，事件必须同帧IoU关联到新器官；新预测track不是生物学死亡/出生真值。
`new_missed_instance`和`new_attached_old_identity`明确区分没检测到与已检测但接成旧身份。完整observations、new_track_events和predicted_organs保存在各run中。
`first_observation_events.csv`逐新叶记录首次帧面积点占比、当前帧GT局部标签对应、small/early/core邻近分层及检测延迟；`first_observation_strata_summary.csv`给植株级区间。事件分层不改变原始严格同帧首观测F1。

| pipeline | event_state | event_observations | small_event_observations | independent_plants |
| --- | --- | --- | --- | --- |
| B0 | attached_to_old_identity | 18 | 13 | 9 |
| B0 | correct_first_observation | 88 | 73 | 12 |
| B0 | sampling_or_identity_lost | 1 | 1 | 1 |
| B1 | attached_to_old_identity | 1 | 0 | 1 |
| B1 | correct_first_observation | 1 | 1 | 1 |
| B1 | missed_instance | 105 | 86 | 12 |
| B3 | attached_to_old_identity | 14 | 7 | 7 |
| B3 | correct_first_observation | 38 | 29 | 12 |
| B3 | missed_instance | 55 | 51 | 12 |
| B4 | attached_to_old_identity | 38 | 25 | 10 |
| B4 | correct_first_observation | 25 | 20 | 11 |
| B4 | missed_instance | 44 | 42 | 12 |
| diagnostic_S1 | attached_to_old_identity | 31 | 21 | 11 |
| diagnostic_S1 | correct_first_observation | 32 | 25 | 11 |
| diagnostic_S1 | missed_instance | 44 | 41 | 12 |

更高PQ不会保证所有终端指标都更高：Validation预先按PQ选择B3，不按追踪/测量后验改选。B3和B4的覆盖率、continuity及严重误差仍有差异，完整表及逐株配对结果保留；不能把全链路收益仅归因于分割PQ一个数。

测量完全复用原函数、缺失损失2及严重动态误差规则，覆盖率含全部适格GT转移。
同测量器减少的是缺失/错连传播，不会自动修正绝对叶长/面积。GT对照仍显著长度偏差，平面凸包面积非人工叶面积真值；**不宣布表型测量系统可靠**。

## 6. PSegNet / Organ3DNet及边界

S3是原PSegNet监督从头基线，真实训练60轮/一个固定种子；只有一个种子，不声称与S1/S2相同随机稳定性证据。
S4官方commit `06f711fc3d9884933c61766c009b2bcd86837139`；原Python3.10/Torch1.12.1+cu113隔离环境实际安装；官方PointNet2和固定ME扩展均在CUDA12.6/11.3不匹配处失败。
官方checkpoint 481057352 bytes下载且MD5核验，但**未严格加载官方模型、未跑官方示例、目标监督步数0**；无S4 PQ、无简化网络、停止移植。
repo代码无明确许可证，checkpoint Zenodo记录CC BY4.0单独注明；本地上游代码/源及衍生权重不公开再分发。

## 7. 继续标准和唯一建议

冻结Validation门槛PQ≥.70、IDF1≥.80、新叶首观测F1≥.70；分割提升代表seed和seed-mean均≥9/12株同方向且配对CI下界>0；早期小叶召回≥.50。
不要求当前偏置测量器全指标达标。Test结果不用于选择或最终判定。

| pipeline | method | pq | idf1 | new_event_f1 | stable_improvement | early_small_recall | new_missed_instance | new_attached_old_identity | passes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B3 | S2 | 0.8692 | 0.8496 | 0.3138 | True | 0.6364 | 4.5833 | 1.1667 | False |
| B4 | S3 | 0.8545 | 0.8462 | 0.1864 | True | 0.6667 | 3.6667 | 3.1667 | False |

唯一建议 **INSTANCE-METHOD-NEEDED**。A converged bounded supervised baseline improves independent plants stably but fails locked gates with low PQ/early-small recall or predominantly instance-missed first observations; a tracking-only failure is not attributed to segmentation. No dense strong-baseline superiority is claimed.
错误归因附加诊断（不改成功门槛）：True。若仅追踪失败而分割/早期小叶达标，不能直接判为需要新分割器；若首观测缺失以当前帧实例漏检为主，则是公开的实例瓶颈证据。芯部邻近和首观测分层是GT评价代理，仍需更多独立植物/外部数据检验泛化。
若证据不足，不继续Test驱动调参；补独立植株/跨来源数据或更严格外部验证。若实例错误仍需方法研究，只由公开的失败类型推动，未自动开发新算法。

## 8. 工程交付、保护及复现

协议/配置、源码/标签审计、训练日志/权重/全分辨率预测、按植株区间、错误分解和真实点云案例均实际存在。
`figures/`中case按Validation B3最差/最好植株固定选择，非手选成功；包含early/late同一植物GT/S0/B3对照及指标/训练曲线/检测-身份错误。
`REPRODUCE.md`给命令，`tests_m6_ab_final.log`保留35项当前/历史接口回归通过；初始完整套件175通过/1失败是M6-B隔离环境缺少历史RGB torchvision，同一历史测试在原`.venv-m1`通过，不修改旧环境或代码。`protection_verification.json`检查历史文件/环境fingerprint、tracked SHA256和891原始文件SHA256，HEAD必须保持原样。
无commit、merge、push，M0–M6-A不改；本阶段完成后停止，不自动开发方法。

## 9. 信息来源

Bocha一般检索不可用（403配额不足），公开方法信息来自实际取得的官方源码/HTTP记录；本地结果来自真实训练/评价，不基于训练数据记忆填数。
- Sugar4D: https://www.ipb.uni-bonn.de/data/sugar4d/index.html ; https://doi.org/10.60507/FK2/IS8YBZ
- PlantNet/PSegNet: https://github.com/Huang2002200/PlantNet-and-PSegNet
- Organ3DNet: https://github.com/Huang2002200/Organ3DNet ; https://doi.org/10.1016/j.aiia.2025.10.011 ; https://zenodo.org/records/17446444
- TrackPlant3D: https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- PyTorch/ME官方文档由Context7实际获取，见`sources/ctx7_*`；全部具体固定commit/许可证/依赖在`SOURCE_AUDIT.md`。
