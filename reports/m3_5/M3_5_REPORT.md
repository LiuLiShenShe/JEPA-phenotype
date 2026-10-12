# M3.5 — 植物器官级生长预测可行性与三维表征诊断



> **工程状态：PASS（108 tests，0 failed/0 skipped）；研究建议：REPRESENTATION-FIRST。不启动 M4。**

> 所有数值来自现有 Sugar4D V4 的真实观测与保存的预测；没有合成数据结果、JEPA 或生成式模型训练。



## 1. 结论与终止选择

**是否存在足够可靠、具有实际意义的器官级动态预测信号，使继续研究 3D Temporal-JEPA 有合理依据？** 否，尚不足以支持现在启动 3D Temporal-JEPA。受控基线发现小幅、稳定的自动形态标签预测增益，但未通过训练前锁定的实际增益门槛，且当前冠幅表征不足；这不等于器官历史完全没有信息。

**唯一下一步建议：REPRESENTATION-FIRST。** 分支选择来自训练前锁定的规则及 Train/Validation 证据，而非测试集上最好子任务。

当前 PointNet 的株高/冠幅可靠性优先诊断；只有当表示门槛通过且锁定器官历史主任务相对 C2/C4 同时通过时，才建议 ORGAN-TEMPORAL-JEPA。

本阶段完成后停止 GRU 扩展，不自动开始 M4、合并、提交或推送。若建议 REPRESENTATION-FIRST，优先几何增强、物理尺度保留、器官分割/表征与少样本当前表型估计；不是把未来预测失败直接归因于没有时间信息。



## 2. 数据、关联与原有成果保护

- 起始实际 commit：`5cf34b8d90b5699597f2875a63e40ab9a02dbe37`；起始工作区干净，`main...origin/main`。

- 已逐项阅读 M3_REPORT、DATA_AUDIT、actual_field_schema、baseline_comparison、Sugar4D dataset、PointNet/GRU 实现和原始两张 CSV。

- 原始植物表 768×61，器官表 6778×8；48 株、16 次真实扫描、675 条跨时一致身份轨迹；全部 768 PLY 重新计算叶身份和原始米制 centroid，并复核 M3 表。

- 原始 CSV 与已有元数据逐字段一致、PLY/CSV 关联一一对应、重复/错配均为 0；centroid 使用当前叶实例点集均值，只用于轨迹审计，不作为 C2–C4 预测输入。

- 完整网格有 4022 个 not_observed 单元。未观测面积等为 NaN，从未当作真实零面积；没有 CSV 缺行但 PLY 有叶等错配。

- 127 条轨迹在 93 DAS 前停止可观测；重新观测轨迹 0 条。所有植株扫描完整，not_observed 仍不能区分遮挡、分割失败、移除或生物学死亡。

- DAS = 37,40,44,47,51,54,58,61,65,72,75,79,82,86,89,93。h=1/2 是后续一次/两次实际扫描，不是固定天数；包含 65→72 的 7 日间隔。

- 保护清单覆盖 5771 个既有文件（含 891 个 Sugar4D 原始文件），最终变化 0；M0–M3 模型、数据和结果不变。

- 保留 M3 官方 24/12/12 植株划分及同一 fold0–fold3 24/12/12 划分。genotype 不作为输入，保留 M3 对 plant020/021 的未确认映射警告，不做基因型结论。



## 3. 实验 B：先验可行性与标签质量

| split | horizon | independent_plants | trajectories | tracks_ge3_observations | eligible_windows | valid_morphology_windows | future_not_observed_windows | event_positive | event_negative |
|---|---|---|---|---|---|---|---|---|---|
| train | 1 | 24 | 340 | 316 | 2498 | 2431 | 67 | 151 | 161 |
| val | 1 | 12 | 157 | 150 | 1172 | 1144 | 28 | 67 | 89 |
| test | 1 | 12 | 178 | 164 | 1268 | 1236 | 32 | 81 | 75 |
| train | 2 | 24 | 340 | 316 | 2253 | 2127 | 126 | 218 | 70 |
| val | 2 | 12 | 157 | 150 | 1055 | 1002 | 53 | 98 | 46 |
| test | 2 | 12 | 178 | 164 | 1140 | 1076 | 64 | 109 | 35 |



所有官方/四折训练与验证门槛通过，完整门槛、每类独立植株覆盖见 feasibility.json；测试类别统计仅描述，不决定任务是否启动。每株轨迹和足够历史数量见 trajectories_per_plant.csv；逐 DAS 分布见 DAS_task_distribution.csv。

形态研究可行性为 **自动发布标签的数值预测可行**，不是已证明可测量精确生物生长：

- 450 条 93 DAS 破坏性人工参考与发布测量匹配；总长 MAE 1.918cm、叶片长 1.868cm、叶柄长 2.383cm、叶宽 1.086cm。

- 该横截面自动/人工差异不是重复扫描噪声估计，不能宣称纵向 delta 噪声下限；本报告保留有风险标签、不修正或删除异常值。

- 叶面积无人工验证，只作次要自动面积任务；不允许单独靠面积测试增益建议 Temporal-JEPA。

- 明确可行性边界：发布自动形态标签的有限数值基线可行；以此独立确证精确生物器官增长不可行，因缺纵向重复测量噪声、跨方法误差较大且面积参考缺失。没有强行训练生物增长/JEPA大模型。

- 面积/长度的相邻增减和极值完整保存在 organ_label_transitions.csv；大幅非单调变化可能受叶姿态、抽取算法与标注几何影响，未假称都是生物增长。



## 4. 实验 A：当前状态编码诊断

A1 使用原有 PointNetForecaster 编码器和预测头，输入当前 XYZ，7 个时间槽全部置零；没有 DAS/实例/语义/GT输入。训练目标为同一时点株高、冠幅、可见标注叶数。A3 在相同 PointNet 上拼接标签自由几何特征作为诊断性补充。

A2-Z-Range/Quantile/Hull 为无需学习原始代理；A2-Geometry-Ridge 是 train-only 标准化/校准的几何回归，不应称作无需学习。A2-DAS-Ridge 单独展示时间条件预测，避免把纯阶段信息当作几何。

**官方验证集（12 独立植株）当前状态 MAE：**

| model | target | plant_macro_mae | seed_mae_std | ci95_low | ci95_high |
|---|---|---|---|---|---|
| A1-PointNet | plant_height | 0.713 | 0.055 | 0.547 | 0.919 |
| A1-PointNet | plant_width | 1.680 | 0.064 | 1.379 | 2.058 |
| A1-PointNet | visible_organ_count | 1.430 | 0.077 | 1.240 | 1.638 |
| A2-DAS-Ridge | plant_height | 4.594 | 0.000 | 2.891 | 6.562 |
| A2-DAS-Ridge | plant_width | 5.405 | 0.000 | 4.909 | 5.967 |
| A2-DAS-Ridge | visible_organ_count | 1.437 | 0.000 | 1.181 | 1.703 |
| A2-Geometry-Ridge | plant_height | 0.450 | 0.000 | 0.350 | 0.574 |
| A2-Geometry-Ridge | plant_width | 0.342 | 0.000 | 0.264 | 0.421 |
| A2-Geometry-Ridge | visible_organ_count | 1.545 | 0.000 | 1.288 | 1.841 |
| A2-Hull-Proxy | plant_width | 0.076 | 0.000 | 0.042 | 0.119 |
| A2-Z-Quantile-Proxy | plant_height | 4.169 | 0.000 | 3.204 | 5.138 |
| A2-Z-Range-Proxy | plant_height | 5.059 | 0.000 | 3.312 | 6.735 |
| A3-PointNet-Geometry | plant_height | 0.509 | 0.041 | 0.394 | 0.642 |
| A3-PointNet-Geometry | plant_width | 1.450 | 0.083 | 1.172 | 1.744 |
| A3-PointNet-Geometry | visible_organ_count | 1.316 | 0.060 | 1.196 | 1.432 |

明确回答：PointNet株高验证MAE 0.713cm，达到预设1cm当前估计标准；冠幅 1.680cm、可见叶数 1.430片，未达到各自1cm/1片标准。因此不能称它已可靠编码全部当前状态。

几何校准株高/冠幅 0.450/0.342cm；几何增强PointNet为 0.509/1.450cm。几何描述对当前几何目标提供强信息，尤其冠幅；不应把原始跨度误当官方Height，叶数仍需要更好的结构表征。



表示分支预设株高/冠幅 1cm 阈值，并要求 PointNet 相对几何校准至少差 1.5倍；这是实用诊断标准，不等同于仪器噪声或统计确证。

```json
[
  {
    "target": "plant_height",
    "pointnet_validation_mae": 0.7126977973088154,
    "geometry_validation_mae": 0.44958099340743707,
    "threshold_cm": 1.0,
    "ratio": 1.5852489490429287,
    "unreliable": false
  },
  {
    "target": "plant_width",
    "pointnet_validation_mae": 1.680230555038909,
    "geometry_validation_mae": 0.34221661073764204,
    "threshold_cm": 1.0,
    "ratio": 4.909845116568716,
    "unreliable": true
  }
]
```

**当前与 M3 未来误差对照（同一官方测试植株；不同任务与时间窗口，仅诊断）：**

| target | horizon | current_state_mae | M3_future_mae |
|---|---|---|---|
| plant_height | 1 | 1.107 | 1.853 |
| plant_width | 1 | 1.642 | 2.349 |
| visible_organ_count | 1 | 1.735 | 1.760 |
| plant_height | 2 | 1.107 | 2.145 |
| plant_width | 2 | 1.642 | 2.665 |
| visible_organ_count | 2 | 1.735 | 1.817 |

**同一目标扫描阶段对齐的诊断：** 当前模型使用目标时点XYZ恢复同一目标，M3用过去XYZ预测；这不是公平未来预测比赛，而是当前表示误差与额外预测困难的拆分。

| target | horizon | current_target_frame_estimation_mae | M3_past_XYZ_future_mae | future_minus_current_task_gap |
|---|---|---|---|---|
| plant_height | 1 | 1.161 | 1.853 | 0.692 |
| plant_width | 1 | 1.670 | 2.349 | 0.679 |
| visible_organ_count | 1 | 1.924 | 1.760 | -0.164 |
| plant_height | 2 | 1.180 | 2.145 | 0.965 |
| plant_width | 2 | 1.696 | 2.665 | 0.970 |
| visible_organ_count | 2 | 1.986 | 1.817 | -0.169 |



株高发布定义是处理后 z 的99分位数减人工固定生长点 z，且排除 taproot、逐实例2mm降采样；本阶段高度跨度/分位跨度未读取人工生长点或语义标签，不是同一量。冠幅原始 XY 凸包直径也未复现官方语义剔除/降采样。当前几何代理有偏差仍可 train-only 校准，绝不以未来 GT 修正。

诊断瓶颈：小型 max+mean PointNet、随机2048点和极少独立植株，可能不充分保留极值/投影边界/点密度；株高目标还依赖外部人工参考及预处理。可见叶数不能仅靠简单包围盒解析。几何增强是否补充见 A3 与 A1 植株配对表；不把表征误差直接叫做不可预测的未来生长。



## 5. 实验 C：固定轻量预测器与输入条件

- C0 Persistence、C1 实际 DAS 最小二乘趋势：Oracle historical phenotype；使用同样三次连续可观测历史，长度/宽度/面积未来值和变化量。

- C2 MLP：当前5个发布形态（总长/叶片长/叶柄长/宽/面积）+合法时间；C3 GRU：三次形态；C4 同架构从头训练，重复末状态。均属于 Oracle organ-state，不是原始点云器官跟踪模型。

- C2–C4 联合预测三个形态 delta 和未来可观测概率；不可观测目标不参与形态损失，但所有历史有效窗口参与 visibility BCE 和独立评价；不在推理时获知未来持续出现。

- 时间输入只有已观测 DAS/间隔、已知查询 target DAS、当前已知的 first-seen-so-far 年龄；没有未来身份、first-observed 目标、生命周期、未来可见性/面积或点云。

- C5 为 **非 Oracle 整株 XYZ 几何预测**：当前或三历史几何 × absolute 或 residual，残差基准是当前 train-calibrated几何估计，不是当前 GT。没有实例标签/器官身份输入，不能与 Oracle C0–C4 宣称信息条件公平。

- C5 27项代理含高度分位、XY范围/凸包面积/直径、径向分布、协方差特征值、raw点数/包围盒密度、8192点邻近距离和上部点比例；几何全由当前/历史原始 XYZ计算。PointNet用采样点而C5用全点几何，因此这是归纳偏置/信息保留诊断，不是严格等点数架构比赛。

- E0 训练频率/DAS阶段；E1 当前可见数+DAS Logistic，数量以同输入 Ridge；E2 当前汇总器官 MLP，E3 三次历史汇总 GRU，E4 重复末状态对照。标签为历史末点之后到目标扫描之间首次被观察的新增身份，非净叶数差或精确出叶时刻。

- Seeds=[17, 42, 93]，每模型50 epochs；tabular hidden64、dropout0.1、AdamW lr0.001/WD0.0001、梯度裁剪5。180 个神经运行全部完成，累计 119400 steps、1375.1 秒模型训练时间。

- MLP/GRU/C5 train-only标准化，按植株×horizon平衡损失；PointNet每株扫描相同所以天然植株平衡。真实/重复GRU参数、初始tensor哈希、epochs/steps完全相同，均从头训练；不拷贝训练后权重。

- epoch/正则化/事件及消失分类阈值仅由相应Validation选择；首版完整validation lock在首次Test checkpoint重放前生成，模型/阈值与研究建议均已固定。

- 首次重放期间发现统计代码把确定性C0/C1的seed=-1行计入‘三随机种子’行数，修正为只取C3/对照共同的三个随机种子并重新锁定；未改模型/阈值/协议、未读取Test评价指标，建议保持REPRESENTATION-FIRST。原锁保存在development_logs/，修正记录在seed_count_correction.json，避免假称首版统计完全无错。



## 6. 实验 D：器官历史增量与统计

预设主任务：h=2 leaf_total_length；必须相对 C2、C4 都至少10%且0.5cm改善、三种子全向好、官方验证植株配对95%上界<0、至少3/4验证折方向改善。面积/宽度/阶段子集只能次要诊断。

**主任务官方验证与四折验证配对结果（GRU减对照，负值有利）：**

| partition | right | plants | left_minus_right_mae | improvement_pct | ci95_low | ci95_high |
|---|---|---|---|---|---|---|
| fold0 | C2-MLP | 12 | -0.034 | 5.516 | -0.058 | -0.014 |
| fold0 | C4-Repeated-GRU | 12 | -0.038 | 6.164 | -0.069 | -0.009 |
| fold1 | C2-MLP | 12 | -0.072 | 9.616 | -0.103 | -0.042 |
| fold1 | C4-Repeated-GRU | 12 | -0.082 | 10.824 | -0.119 | -0.047 |
| fold2 | C2-MLP | 12 | -0.054 | 8.221 | -0.071 | -0.038 |
| fold2 | C4-Repeated-GRU | 12 | -0.057 | 8.680 | -0.073 | -0.044 |
| fold3 | C2-MLP | 12 | -0.090 | 12.900 | -0.144 | -0.044 |
| fold3 | C4-Repeated-GRU | 12 | -0.099 | 14.067 | -0.152 | -0.056 |
| official | C2-MLP | 12 | -0.107 | 14.432 | -0.160 | -0.059 |
| official | C4-Repeated-GRU | 12 | -0.114 | 15.292 | -0.174 | -0.059 |

```json
[
  {
    "control": "C2-MLP",
    "validation_improvement_pct": 14.432210108511622,
    "validation_absolute_improvement_cm": 0.10662417013127641,
    "validation_difference_ci95": [
      -0.16026315425637142,
      -0.05942494536767752
    ],
    "all_three_seeds_improve": true,
    "improving_validation_folds": 4,
    "folds": [
      {
        "fold": "fold0",
        "validation_difference_cm": -0.03415652571495602
      },
      {
        "fold": "fold1",
        "validation_difference_cm": -0.07227806629989462
      },
      {
        "fold": "fold2",
        "validation_difference_cm": -0.054168525023406085
      },
      {
        "fold": "fold3",
        "validation_difference_cm": -0.08990569214272026
      }
    ],
    "passed": false
  },
  {
    "control": "C4-Repeated-GRU",
    "validation_improvement_pct": 15.292262444207863,
    "validation_absolute_improvement_cm": 0.11412526431361307,
    "validation_difference_ci95": [
      -0.17422206451676714,
      -0.059427675445670014
    ],
    "all_three_seeds_improve": true,
    "improving_validation_folds": 4,
    "folds": [
      {
        "fold": "fold0",
        "validation_difference_cm": -0.03842747402227293
      },
      {
        "fold": "fold1",
        "validation_difference_cm": -0.08246856236319204
      },
      {
        "fold": "fold2",
        "validation_difference_cm": -0.05748058096259475
      },
      {
        "fold": "fold3",
        "validation_difference_cm": -0.09937070919643716
      }
    ],
    "passed": false
  }
]
```

相对 C2-MLP：14.4% / 0.107cm改善，三种子一致=True，改善验证折=4/4，预设门槛通过=False；相对 C4-Repeated-GRU：15.3% / 0.114cm改善，三种子一致=True，改善验证折=4/4，预设门槛通过=False。不把未达实际意义门槛解释为‘绝对没有动态信息’：可检测的小幅自动标签预测增益仍应与测量限制和表示误差分开。

**官方测试形态结果（探索性、按植株宏平均）：**

| model | horizon | target | plant_macro_mae | seed_mae_std | ci95_low | ci95_high | delta_phenotype_mae |
|---|---|---|---|---|---|---|---|
| C0-Persistence | 1 | leaf_area | 5.485 | 0.000 | 4.353 | 6.836 | 5.485 |
| C0-Persistence | 1 | leaf_blade_width | 0.700 | 0.000 | 0.556 | 0.844 | 0.700 |
| C0-Persistence | 1 | leaf_total_length | 0.657 | 0.000 | 0.572 | 0.753 | 0.657 |
| C0-Persistence | 2 | leaf_area | 7.376 | 0.000 | 5.999 | 9.012 | 7.376 |
| C0-Persistence | 2 | leaf_blade_width | 0.858 | 0.000 | 0.686 | 1.045 | 0.858 |
| C0-Persistence | 2 | leaf_total_length | 1.034 | 0.000 | 0.895 | 1.188 | 1.034 |
| C1-Linear-Growth | 1 | leaf_area | 6.918 | 0.000 | 5.267 | 8.897 | 6.918 |
| C1-Linear-Growth | 1 | leaf_blade_width | 0.967 | 0.000 | 0.752 | 1.178 | 0.967 |
| C1-Linear-Growth | 1 | leaf_total_length | 0.766 | 0.000 | 0.674 | 0.867 | 0.766 |
| C1-Linear-Growth | 2 | leaf_area | 11.839 | 0.000 | 9.429 | 14.686 | 11.839 |
| C1-Linear-Growth | 2 | leaf_blade_width | 1.428 | 0.000 | 1.154 | 1.703 | 1.428 |
| C1-Linear-Growth | 2 | leaf_total_length | 1.558 | 0.000 | 1.372 | 1.760 | 1.558 |
| C2-MLP | 1 | leaf_area | 4.805 | 0.032 | 3.702 | 6.043 | 4.805 |
| C2-MLP | 1 | leaf_blade_width | 0.729 | 0.014 | 0.577 | 0.890 | 0.729 |
| C2-MLP | 1 | leaf_total_length | 0.458 | 0.007 | 0.389 | 0.532 | 0.458 |
| C2-MLP | 2 | leaf_area | 5.957 | 0.036 | 4.641 | 7.523 | 5.957 |
| C2-MLP | 2 | leaf_blade_width | 0.852 | 0.022 | 0.673 | 1.037 | 0.852 |
| C2-MLP | 2 | leaf_total_length | 0.667 | 0.002 | 0.564 | 0.780 | 0.667 |
| C3-GRU | 1 | leaf_area | 4.503 | 0.078 | 3.538 | 5.559 | 4.503 |
| C3-GRU | 1 | leaf_blade_width | 0.727 | 0.008 | 0.575 | 0.883 | 0.727 |
| C3-GRU | 1 | leaf_total_length | 0.420 | 0.006 | 0.348 | 0.495 | 0.420 |
| C3-GRU | 2 | leaf_area | 5.527 | 0.098 | 4.398 | 6.816 | 5.527 |
| C3-GRU | 2 | leaf_blade_width | 0.857 | 0.018 | 0.681 | 1.043 | 0.857 |
| C3-GRU | 2 | leaf_total_length | 0.597 | 0.003 | 0.499 | 0.698 | 0.597 |
| C4-Repeated-GRU | 1 | leaf_area | 4.965 | 0.083 | 3.770 | 6.290 | 4.965 |
| C4-Repeated-GRU | 1 | leaf_blade_width | 0.734 | 0.014 | 0.566 | 0.908 | 0.734 |
| C4-Repeated-GRU | 1 | leaf_total_length | 0.467 | 0.004 | 0.397 | 0.542 | 0.467 |
| C4-Repeated-GRU | 2 | leaf_area | 6.037 | 0.055 | 4.657 | 7.696 | 6.037 |
| C4-Repeated-GRU | 2 | leaf_blade_width | 0.847 | 0.004 | 0.672 | 1.032 | 0.847 |
| C4-Repeated-GRU | 2 | leaf_total_length | 0.678 | 0.003 | 0.573 | 0.792 | 0.678 |

绝对未来与delta都减去同一个历史GT基准，所以器官 absolute MAE 与 delta MAE 数学上相等；没有把它们当作两条独立证据。形态指标是在未来被观察目标上条件评价，需同时读取可观测性表。

以植株先汇总该株的所有叶/窗口，再等权汇总植物；seed间比较用每种子误差，区间先平均每株种子误差再重采样植株10,000次，不对叶/窗口做伪重复。不把平均预测的ensemble误差冒充三种子独立性能。

两项主对照必须同时通过，主任务和绝对/相对门槛在训练前锁定；次要目标/阶段/事件对照均为探索性描述，95%区间未作全表多重比较确证性解释，不以某个事后显著小任务改变建议。

所有四折结果保存在 organ_forecast_metrics/organ_validation_pairs，OOF 48株结果见 organ_oof_metrics；各折模型共享训练植株，OOF置信区间与各验证折方向是描述性稳定性检查，不是外部重复。

**C5非Oracle整株几何 absolute/residual 对照（官方探索性Test）：**

| model | horizon | target | plant_macro_mae | seed_mae_std |
|---|---|---|---|---|
| C5-Current-Absolute | 1 | plant_height | 1.593 | 0.077 |
| C5-Current-Absolute | 1 | plant_width | 2.030 | 0.164 |
| C5-Current-Absolute | 1 | visible_organ_count | 1.424 | 0.041 |
| C5-Current-Absolute | 2 | plant_height | 1.870 | 0.080 |
| C5-Current-Absolute | 2 | plant_width | 2.514 | 0.166 |
| C5-Current-Absolute | 2 | visible_organ_count | 1.475 | 0.055 |
| C5-History-Absolute | 1 | plant_height | 1.791 | 0.046 |
| C5-History-Absolute | 1 | plant_width | 2.237 | 0.042 |
| C5-History-Absolute | 1 | visible_organ_count | 1.491 | 0.040 |
| C5-History-Absolute | 2 | plant_height | 2.068 | 0.039 |
| C5-History-Absolute | 2 | plant_width | 2.575 | 0.028 |
| C5-History-Absolute | 2 | visible_organ_count | 1.530 | 0.024 |
| C5-Current-Residual | 1 | plant_height | 1.181 | 0.023 |
| C5-Current-Residual | 1 | plant_width | 1.599 | 0.044 |
| C5-Current-Residual | 1 | visible_organ_count | 1.482 | 0.025 |
| C5-Current-Residual | 2 | plant_height | 1.506 | 0.023 |
| C5-Current-Residual | 2 | plant_width | 2.443 | 0.042 |
| C5-Current-Residual | 2 | visible_organ_count | 1.523 | 0.027 |
| C5-History-Residual | 1 | plant_height | 1.158 | 0.006 |
| C5-History-Residual | 1 | plant_width | 1.593 | 0.012 |
| C5-History-Residual | 1 | visible_organ_count | 1.503 | 0.006 |
| C5-History-Residual | 2 | plant_height | 1.496 | 0.000 |
| C5-History-Residual | 2 | plant_width | 2.465 | 0.047 |
| C5-History-Residual | 2 | visible_organ_count | 1.533 | 0.011 |

C5的 residual 基准是当前几何估计而非发布当前GT，故记录的 residual delta误差是代理残差误差，不是已经校准过的生物增长率。geometry_paired_comparisons 给出同信息条件的历史/当前和残差/绝对对照；与 M3 小型PointNet比较仅诊断。

absolute/residual各自按训练输出标准差归一化，故除了残差参数化也改变多目标损失尺度；不能把差异全归因为残差技巧本身。比较使用物理单位MAE，不能跨任务比较normalized validation score。



## 7. 新叶事件与未来可观测性

**新叶首次观测（官方探索性Test）：**

| model | horizon | plant_macro_precision | plant_macro_recall | plant_macro_f1 | plant_balanced_pr_auc | new_count_plant_macro_mae |
|---|---|---|---|---|---|---|
| E0-DAS | 1 | 0.625 | 0.760 | 0.677 | 0.623 | 0.613 |
| E0-DAS | 2 | 0.757 | 1.000 | 0.852 | 0.821 | 0.769 |
| E0-Frequency | 1 | 0.519 | 1.000 | 0.675 | 0.519 | 0.650 |
| E0-Frequency | 2 | 0.757 | 1.000 | 0.852 | 0.757 | 0.822 |
| E1-Logistic | 1 | 0.694 | 0.673 | 0.666 | 0.697 | 0.598 |
| E1-Logistic | 2 | 0.772 | 0.930 | 0.837 | 0.908 | 0.761 |
| E2-MLP | 1 | 0.641 | 0.765 | 0.674 | 0.694 | 0.587 |
| E2-MLP | 2 | 0.798 | 0.921 | 0.842 | 0.916 | 0.729 |
| E3-GRU | 1 | 0.619 | 0.828 | 0.689 | 0.746 | 0.567 |
| E3-GRU | 2 | 0.823 | 0.935 | 0.865 | 0.932 | 0.722 |
| E4-Repeated-GRU | 1 | 0.624 | 0.874 | 0.712 | 0.701 | 0.582 |
| E4-Repeated-GRU | 2 | 0.787 | 0.932 | 0.841 | 0.911 | 0.733 |

Precision/Recall/F1先计算各株再宏平均（zero_division=0），同时提供植株等权pooled混淆指标；PR-AUC使用average precision，以每株等总权重计算，另给有正负类别植株的macro AP及覆盖。种子指标先分别计算再平均，不能把多个种子同窗口当成重复样本。官方验证/测试PR-AUC区间按整个植株同步重采样各seed、每次计算AP后平均seed；数量MAE/F1区间同样按植株重采样，详见指标CSV。

**未来未被观测事件（1−visibility；官方探索性Test）：**

| model | horizon | samples | positive | positive_plants | plant_macro_precision | plant_macro_recall | plant_macro_f1 | plant_balanced_pr_auc | plant_macro_brier |
|---|---|---|---|---|---|---|---|---|---|
| C0-Persistence | 1 | 1268.000 | 32.000 | 12.000 | 0.000 | 0.000 | 0.000 | 0.026 | 0.026 |
| C0-Persistence | 2 | 1140.000 | 64.000 | 12.000 | 0.000 | 0.000 | 0.000 | 0.057 | 0.057 |
| C1-Linear-Growth | 1 | 1268.000 | 32.000 | 12.000 | 0.000 | 0.000 | 0.000 | 0.026 | 0.026 |
| C1-Linear-Growth | 2 | 1140.000 | 64.000 | 12.000 | 0.000 | 0.000 | 0.000 | 0.057 | 0.057 |
| C2-MLP | 1 | 1268.000 | 32.000 | 12.000 | 0.352 | 0.468 | 0.364 | 0.402 | 0.019 |
| C2-MLP | 2 | 1140.000 | 64.000 | 12.000 | 0.598 | 0.590 | 0.545 | 0.626 | 0.034 |
| C3-GRU | 1 | 1268.000 | 32.000 | 12.000 | 0.276 | 0.359 | 0.275 | 0.355 | 0.020 |
| C3-GRU | 2 | 1140.000 | 64.000 | 12.000 | 0.594 | 0.553 | 0.502 | 0.580 | 0.038 |
| C4-Repeated-GRU | 1 | 1268.000 | 32.000 | 12.000 | 0.320 | 0.403 | 0.308 | 0.353 | 0.020 |
| C4-Repeated-GRU | 2 | 1140.000 | 64.000 | 12.000 | 0.525 | 0.625 | 0.511 | 0.578 | 0.036 |

可观测性表正类是未来not_observed，突出少数消失窗口，而非以约95%以上持续观察率隐藏偏差。C0/C1持续观察假设即预测消失概率0；C2–C4联合头学习概率，分类阈值在验证集上锁定。形态条件性指标并不证明能正确处理所有将消失叶片。



## 8. 可视化、失败案例与数据限制

六组 PNG/PDF/SVG + 源CSV位于 figures：01真实叶轨迹；02该叶h2长度/面积预测；03主任务器官历史增量与配对CI；04首次观测事件/数量；05几何代理/发布目标；06验证失败植株及叶。

01/02/04由验证历史覆盖最长的字典序首叶/植株选择，未挑最佳预测；06是事后最大验证MAE诊断，不用于改主任务或选择研究分支。散点/曲线用于描述，图上的seed均值线不替代误差统计。

所有验证器官失败样本按MAE保存于 organ_failure_cases.csv，包含未观测目标窗口数；选择偏差不能靠只画持续存在的叶片隐藏。

失败可能包括非单调长度/面积突变、解剖自动度量误差、短历史和future not_observed选择；没有将点云局部错误自动断定为身份错误。

官方Test此前用于M3，M3.5 Test与含官方Test植株的fold结果均只能探索性证据，不称独立确证；48株单温室不足以建立跨场景/基因型外推。



## 9. 复现与验收

```bash

.venv-m1/bin/python scripts/analyze_m3_5_feasibility.py

.venv-m1/bin/python scripts/train_m3_5.py --stage current --device cuda:0

.venv-m1/bin/python scripts/evaluate_m3_5.py --stage current

.venv-m1/bin/python scripts/train_m3_5.py --stage future --device cuda:0

.venv-m1/bin/python scripts/evaluate_m3_5.py --stage select

.venv-m1/bin/python scripts/evaluate_m3_5.py --stage final --device cuda:0

.venv-m1/bin/python -m pytest -q --junitxml=reports/m3_5/tests_all.xml

.venv-m1/bin/python scripts/verify_m3_5.py

```

复用已安装 `.venv-m1`；版本记录environment.json。所有新缓存、checkpoint和逐窗口预测在 outputs/m3_5，不覆盖 data/processed/sugar4d。训练脚本跳过已有checkpoint；完整从头复现实验请使用单独的已有数据工作目录，不删除旧里程碑文件。

开发阶段发生numpy标量JSON与单输出Ridge维度的工程错误，以及C5当前代理测试隔离的防御性强化；更正记录与开发产物已归档。最终180个运行统一使用修正后同一源代码哈希从头训练；未更改协议、任务、超参数或预算，也未根据测试结果调整方案。

验收以 acceptance.json、tests_all.log/XML、training_complete.json、selection_lock、inference_repeatability、final_protection 和 figure_manifest 为准。没有自动下载大型数据或产生虚构不可行任务分类分数。



## 10. 来源与归属

- 本实验数值：原始 leaf_related.csv/plant_related.csv/PLY、M3审计、本阶段保存的真实预测与植株级统计CSV。

- Sugar4D官方项目 https://www.ipb.uni-bonn.de/data/sugar4d/index.html （本次HTTP200，页面保存sources/official_page.txt）；数据DOI https://doi.org/10.60507/FK2/IS8YBZ ，固定复用M3 V4哈希，不请求新版本。

- Bocha搜索返回403配额不足，已保存sources/bocha_search.json；搜索不可用，不冒充检索成功或以训练知识替代数据证据。Dataverse版本端点本次超时，版本由已有固定V4档案核实。

- PyTorch 2.8、scikit-learn 1.7.1近邻版本、SciPy、Matplotlib 当前官方源码文档经Context7 library→docs获取并保存sources；运行时确切版本见environment.json。

- 官方 Height/Width定义来源与作者归属沿用 reports/m3/SOURCES.md 和 morphology_definitions.json；本阶段变化为派生轨迹/几何/窗口/预测。Sugar4D CC BY4.0；Bömer等作者，不隐含原作者背书。

- 图表是实测数据的Python绘制，不使用生成式图片制造定量内容；图片内容由Agnes API识别进行外部视觉QA，响应保存在figure_visual_qa.json/visual_qa，自动renderer确认无绘制文字越界；外部判断不替代数值数据验证。
