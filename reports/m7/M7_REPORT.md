# M7：Birth-aware Plant Organ Dynamics from Sparse Temporal Point Clouds

## 1. 唯一结论与下一阶段

**DATA-LIMITED**；唯一下一阶段建议 **STOP**。
Validation锁定系统 `C2`：新叶首次观测F1=0.3329，IDF1=0.8765。
出生门槛通过=False；追踪门槛通过=False。单独IDF1上升不等于TRACKING-SUCCESS，还要求正确相邻link覆盖增益至少0.02；单独候选分类F1上升也不等于自动系统的BIRTH-SUCCESS。
本结论只覆盖固定候选和轻量模型的实际证据，不是证明所有动态方法无效。未自动增加分割主干、大模型或JEPA。

## 2. M7-A：出生 benchmark

保留官方24/12/12整株划分，48株各16帧，共768帧；合法叶和整株不删除。
全部叶首次出现675个，其中可比较的非左删失新叶477个，分集合为{'test': 132, 'train': 238, 'val': 107}。
`organ_birth_events.csv`包含全部要求字段，另记录删失、事件区间和代理出处。

- **Visible birth**：释放的PLY首次出现叶语义且非负实例，不是生物出生；真实可见时刻位于前次扫描与当前扫描之间，初始扫描左删失。
- **Segmentable birth**：固定S2 strict IoU>0.5首匹配，是依赖模型的操作代理，不能称天然可分割。一直未匹配为右删失；不是设为93DAS。
- **Trackable birth**：固定M6-B追踪中连续三个扫描匹配同一track的首帧，需要事后确认，仅benchmark标注使用，不输入模型。
- Train旧追踪未运行，trackable置缺失且 `trackability_evaluable=False`，不为了补字段重跑旧方法。Train S2预测是冻结模型训练内推理，非out-of-fold，限制明确。
- 初始面积是官方点云提取元数据cm²，**不是人工面积真值**，仅评价；小叶用M6-B Train阈值点比例<=.04588683197817431。core距离为GT leaf到GT core的p10米距离，<=5mm是邻近代理，不是人工证明未分离。

| split | plants | events | small_events | core_adjacent_events | median_initial_points | median_initial_area_cm2 | segmentable_right_censored |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 12 | 132 | 98 | 38 | 6650.0000 | 14.9515 | 23 |
| train | 24 | 238 | 187 | 60 | 6787.0000 | 15.2407 | 22 |
| val | 12 | 107 | 87 | 18 | 6754.0000 | 17.3643 | 9 |

小叶是**相对点数比例**，不是生物学幼叶或绝对点数很少；需结合面积和物理距离解释。GT同一ID在首次/末次之间有缺扫描的叶共0/675个；缺扫描可能是遮挡或标注变化，不能无人工复核宣称标签错误。

## 3. M7-B：时间信息与发现

Baseline0完全读取已有M6-B S2 seed17分割和TrackPlant3D结果，不重评旧方法或修改历史指标。
Baseline1在米坐标用4mm voxel占据/局部密度差分、12mm DBSCAN，保留所有候选。已有叶增长、扫描漂移也可能造成差分；候选上限单列为评价Oracle，不是自动方法。
Baseline2历史几何RandomForest；Baseline3共享MLP PointNet+64维GRU，均三随机种子。加入单帧几何RF/PointNet匹配对照，候选集相同；单帧控制仍来自相同因果候选，因此结论仅是历史特征/编码增益，不能称纯单帧候选系统。
所有输入仅XYZ、预测语义/实例、<=t的物理几何、实际DAS间隔。全点候选评估 strict IoU>.5，不把GT叶中心/ID作为候选。

| method | split | birth_f1 | birth_recall | small_recall |
| --- | --- | --- | --- | --- |
| PointNet_GRU | val | 0.3744 | 0.3309 | 0.2467 |
| candidate_oracle | val | 0.6731 | 0.5251 | 0.4549 |
| difference | val | 0.0120 | 0.1156 | 0.1013 |
| history_RF | val | 0.3709 | 0.4156 | 0.3470 |
| single_PointNet | val | 0.3195 | 0.3342 | 0.2483 |
| single_RF | val | 0.3575 | 0.4351 | 0.3633 |

RF树数200深度12；PointNet/GRU40轮，Adam .001，batch128，train-only标准化。相同候选训练标签，三种子共同Validation选阈值；最终轮checkpoint，不以Test选结构或epoch。
另单独训练next-scan plant-level forecast_RF三种子，目标下一次扫描是否有新可见叶，只用截至t历史；见`forecast_metrics_val.csv`。这不是当前器官定位或生物出生预测，不能与定位F1混为一谈。
锁定新叶模型为 `PointNet_GRU`；所有阈值和单种子结果完整交付，不只报最好植株/种子。

匹配对照按三种子先植株平均的差值如下（selected thresholds，CI/p是选择后探索性证据）：

| comparison | mean | lower | upper | plants | positive_plants | holm_p_for_two_validation_F1_comparisons |
| --- | --- | --- | --- | --- | --- | --- |
| history_RF - single_RF | 0.0134 | -0.0095 | 0.0343 | 12 | 8 | 0.2725 |
| PointNet_GRU - single_PointNet | 0.0549 | -0.0051 | 0.1221 | 12 | 7 | 0.2725 |

Validation候选池GT选取的严格IoU召回乐观上限为57/107事件，植株macro=0.5251。`candidate_oracle`使用GT选择正确候选、忽略自动辨别困难，不属于自动方法、不参与选择。若候选本身无法分离一片新叶，历史分类器只能改变候选概率，不能凭空恢复该叶mask。

## 4. M7-C：身份保持及集成

C0是原TrackPlant3D结果导入。C1监督MLP：物理质心位移、形状特征、面积/长度变化和方向；C2增加32维PointNet器官embedding及同叶/异叶监督contrastive距离。不是JEPA。
GT时间身份仅用于Train配对标签和evaluator；推理不使用GT。实例进入系统独立随机局部编号，Hungarian一对一匹配，最多两扫描gap，C1/C2相同生命周期。
C0保持官方采样、CPD及几何cost，不被替换为本轮学习matching。C1/C2与C0同时存在匹配评分和生命周期实现差异，因此系统增益不是纯粹的embedding因果消融；C2相对C1才共享同一生命周期。结果只验证身份打分/管理，不证明学习了可泛化的植物生理规律。
birth-aware集成将选中新区域/实例显式初始化新track；差分区域重叠会从旧mask中划分，不能假装它天然是GT叶，完整记录错误。没有优化分割主干或PQ。

| method | split | idf1 | new_event_f1 | correct_link_coverage | continuity | id_switches |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | val | 0.8603 | 0.3253 | 0.8287 | 0.7382 | 5.1111 |
| C2 | val | 0.8765 | 0.3329 | 0.8343 | 0.7443 | 4.4444 |
| Current | val | 0.8496 | 0.3138 | 0.8519 | 0.7622 | 3.5833 |
| PointNet_GRU+C1 | val | 0.8528 | 0.3144 | 0.8201 | 0.7329 | 6.0833 |
| PointNet_GRU+C2 | val | 0.8677 | 0.3188 | 0.8254 | 0.7387 | 5.4167 |

三种子平均后按植株配对：IDF1增益CI={'mean': 0.026917537836775113, 'lower': -0.006766767533514977, 'upper': 0.06013167538180498, 'plants': 12}；事件F1增益CI={'mean': 0.019102864229911448, 'lower': -0.025702853258179083, 'upper': 0.06895461282309726, 'plants': 12}；小叶首观测召回增益CI={'mean': 0.05732323232323233, 'lower': 0.0080808080808081, 'upper': 0.12071338383838376, 'plants': 12}；正确link覆盖增益CI={'mean': -0.017636613944865136, 'lower': -0.03288787356543567, 'upper': -0.004148075088908622, 'plants': 12}。
出生F1改善植株6/12。bootstrap10000，95%CI，单位是12株而非192扫描或3种子×12独立株。
小叶首观测成功率的提高不等于小叶mask检测召回提高：C1/C2保持S2分割，改善可能仅来自正确初始化身份。IDF1平均提高也不能抵消正确相邻link覆盖下降；须同时查看上述配对区间。

三种子分别的全12株结果：

| method | seed | idf1 | new_event_f1 | correct_link_coverage |
| --- | --- | --- | --- | --- |
| C1 | 17 | 0.8558 | 0.3132 | 0.8222 |
| C1 | 29 | 0.8668 | 0.3302 | 0.8293 |
| C1 | 43 | 0.8584 | 0.3325 | 0.8346 |
| C2 | 17 | 0.8765 | 0.3323 | 0.8318 |
| C2 | 29 | 0.8791 | 0.3393 | 0.8412 |
| C2 | 43 | 0.8739 | 0.3269 | 0.8298 |
| PointNet_GRU+C1 | 17 | 0.8503 | 0.3142 | 0.8158 |
| PointNet_GRU+C1 | 29 | 0.8594 | 0.3135 | 0.8220 |
| PointNet_GRU+C1 | 43 | 0.8488 | 0.3157 | 0.8225 |
| PointNet_GRU+C2 | 17 | 0.8720 | 0.3254 | 0.8269 |
| PointNet_GRU+C2 | 29 | 0.8698 | 0.3256 | 0.8323 |
| PointNet_GRU+C2 | 43 | 0.8613 | 0.3053 | 0.8169 |

整株下一扫描forecast与当前扫描叶mask出生不是同一目标。另有三种子RF及不使用GT作为输入的always/never-new频率控制，禁止用其F1代替新叶定位结果：

| method | split | F1 | Brier |
| --- | --- | --- | --- |
| next_scan_always_new | test | 0.7169 | 0.4333 |
| next_scan_always_new | val | 0.6272 | 0.5389 |
| next_scan_never_new | test | 0.0000 | 0.5667 |
| next_scan_never_new | val | 0.0000 | 0.4611 |
| next_scan_plant_forecast_RF | test | 0.7289 | 0.2112 |
| next_scan_plant_forecast_RF | val | 0.6534 | 0.2202 |

## 5. M7-D：测量与失败分解

完整恢复研究启动条件=False。若未稳定，仅描述性覆盖/错误审计，不宣称时序表型恢复成功。
Oracle为GT mask+GT identity，不同于GT mask+自动tracker；Current为旧S2+固定TrackPlant3D；Proposed为Validation锁定自动系统。
所有方法复用 `m6a.metrics.geometry_measure`；面积平面凸包、长度PCA范围的绝对偏差不改动。MAE/RMSE条件于测量且身份正确，必须同时读丢失/错接覆盖率；缺失不当零误差。
参照`leaf_related.csv`为官方自动点云形态提取，不是每时刻人工真值；人工CSV只有93DAS长度/宽度450条、无人工面积或完整growth轨迹。以下误差相对官方提取参照，不称独立实测精度，见`DATA_INTERFACE_AUDIT.md`。

| method | split | trait | kind | coverage | MAE | RMSE | relative_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C2 | val | area | growth_rate | 0.8343 | 3.8419 | 6.0867 | 0.2016 |
| C2 | val | area | static | 0.8470 | 13.8965 | 20.3397 | 0.1895 |
| C2 | val | length | growth_rate | 0.8343 | 0.4044 | 0.5899 | 0.0729 |
| C2 | val | length | static | 0.8470 | 9.9625 | 10.3754 | 0.4714 |
| Current | val | area | growth_rate | 0.8519 | 3.9850 | 6.4638 | 0.2071 |
| Current | val | area | static | 0.8210 | 14.3854 | 21.6339 | 0.1921 |
| Current | val | length | growth_rate | 0.8519 | 0.4137 | 0.6039 | 0.0748 |
| Current | val | length | static | 0.8210 | 9.9790 | 10.3960 | 0.4698 |
| Oracle_GT_masks_GT_identity | val | area | growth_rate | 1.0000 | 2.8763 | 4.9911 | 0.1562 |
| Oracle_GT_masks_GT_identity | val | area | static | 1.0000 | 12.2336 | 18.0379 | 0.1749 |
| Oracle_GT_masks_GT_identity | val | length | growth_rate | 1.0000 | 0.3619 | 0.5609 | 0.0688 |
| Oracle_GT_masks_GT_identity | val | length | static | 1.0000 | 9.5265 | 9.9738 | 0.4781 |

Validation选定系统新叶失败/正确共三种子计数：{'correct': 139, 'detection_failure': 165, 'identity_failure': 17}。三种子不是新增生物样本。
`event_level_results.csv`、`failure_cases.csv`区分首帧检测失败、已检测但错身份；`measurement_failure_cases.csv`单列在正确身份上的测量器错误。不能用平均误差掩盖新叶漏检或错轨。

## 6. 四个最终问题

1. **新叶发现是否主要瓶颈**：是出生任务的主要瓶颈。固定Current在107事件的计数={'detection_failure': 55, 'identity_failure': 14, 'correct': 38}；选定系统每种子平均={'correct': 46.333333333333336, 'detection_failure': 55.0, 'identity_failure': 5.666666666666667}。候选乐观上限仅57/107，限制集中在出生帧的可独立辨别几何，而非整株静态PQ。这个判断不意味着全部动态测量误差也由漏检主导。
2. **时间是否优于单帧**：没有建立稳定的时间编码优势：两组匹配对照均未得到完全为正的探索性植株CI。共享候选history_RF与single_RF、PointNet_GRU与single_PointNet是匹配特征对照；共同候选本身使用时间变化，所以不是纯单帧端到端系统消融。集成系统收益不能单独证明时间编码收益；下一扫描整株forecast F1也不是叶mask出生F1。
3. **最大错误贡献**：首观测事件中Current的detection failure=55、identity failure=14，因此漏检最大；选定系统见上方平均。在全部静态长度观测的分母上，选定系统最大失败项为`measurement_failure`，各项比例={'correct': 0.4844527363184079, 'detection_failure': 0.1069651741293532, 'identity_failure': 0.04270315091210607, 'measurement_failure': 0.36587893864013266}。这是不同分母、不同问题，不强行合成单一百分比。Oracle GT mask+GT identity仍有测量偏差：

| trait | coverage | MAE | RMSE | relative_error |
| --- | --- | --- | --- | --- |
| area | 1.0000 | 12.2336 | 18.0379 | 0.1749 |
| length | 1.0000 | 9.5265 | 9.9738 | 0.4781 |

测量错误分解使用相对误差>50%的描述性门限，静态分母floor=1cm/1cm²；不是M6-B原可靠性判据，也不用于方法选择。身份正确条件误差不能证明整套测量器可靠。
4. **下一阶段**：只建议 **STOP**，不自动设计大模型或恢复JEPA。

## 7. Test、复现及保护

Test早已在历史阶段使用，只作固定方案探索性复核，不是独立盲测。Validation选择锁定后才进行Test概率推理、track及评价。三种子Test全部保留，不反向调整阈值。

| method | split | idf1 | new_event_f1 | correct_link_coverage | continuity | id_switches |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | test | 0.8845 | 0.3809 | 0.8488 | 0.7337 | 3.6667 |
| C2 | test | 0.8877 | 0.3826 | 0.8534 | 0.7386 | 3.1944 |
| Current | test | 0.8507 | 0.3238 | 0.8563 | 0.7390 | 3.8333 |
| PointNet_GRU+C1 | test | 0.8780 | 0.4058 | 0.8414 | 0.7306 | 4.9167 |
| PointNet_GRU+C2 | test | 0.8839 | 0.4149 | 0.8468 | 0.7365 | 4.3611 |

新增工程与训练日志/checkpoint/全部候选概率/事件结果/track/失败保存在`outputs/m7`和`reports/m7`。
`tracking_paired_plants.csv`保留每株三种子均值对Current的配对差；`track_level_results.csv`与`gt_track_level_results.csv`分别保留预测轨迹及参照器官的漏失、混合身份与连续性，不删除无匹配轨迹。
复现：`scripts/build_birth_dataset.py` → `scripts/train_m7_matrix.sh` → `scripts/report_m7.py`；详见`REPRODUCE.md`。
旧成果保护以任务开始HEAD及tracked SHA256、旧outputs/reports大小mtime审计；不运行旧report写回、不改历史环境。未提交/合并/推送。本阶段交付后停止。

共21次有效正式拟合：7种方法×3种子，其中12个NN每个40轮，RF每个200树；最终轮而非Test最优checkpoint。模型参数、样本、正样本、梯度步和运行时间见`training_budgets.csv`。另有被作废的工程批次，全部保留在`invalid_source_leak_runs`，**不参与结果或选择**：GRU最初时间精度错误，之后发现早期构建缓存的单帧控制含source/novel两列。内容审计纠正为54维，不含两列，重新按原预算从头训练全部出生/匹配模型，无改门限、读Test调参或删样本；`GRU_dtype_failure.log`和作废缓存证据保留。
图见`figures/`（真实统计、配对证据、失败分解和点云实例案例；SVG/PDF/PNG），案例按失败类型中位初始点数选择，不是最优植株。

## 8. 来源与限制

一般博查检索实际HTTP403配额不足，不能声称成功搜索。事实来源是实际取得的官方网页/Dataverse元数据、Context7官方文档和本地实验产物，不凭训练记忆编数。
Sugar4D官方：https://www.ipb.uni-bonn.de/data/sugar4d/index.html ; https://doi.org/10.60507/FK2/IS8YBZ 。PyTorch2.8和scikit-learn文档实际拉取，见`sources/ctx7_*`。
只有单一来源48株，时间间隔3–7天，visible是释放标注代理；没有生物出生真值、人工审定segmentability或外部植物验证。任何DATA-LIMITED只能说明该限定benchmark/模型未建立稳定优势，不能凭失败宣称事件数必然太少。
Train非左删失事件238个，不能随意说“完全无训练事件”；难度是24个独立训练植株、稀疏事件区间、出生候选辨识上限、in-sample S2 proposals以及未人工审定的可分离性。缺外部盲测与人工微小叶标注，停止本轮调参不等于证明一切出生模型不可行。
来源检索失败只限一般搜索；上述科研数值来自真实本地执行而不是训练数据记忆。完整来源、锁定哈希、依赖环境、delivery manifest与保护检查单独交付。
