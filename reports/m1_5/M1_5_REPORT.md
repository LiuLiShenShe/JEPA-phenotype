# M1.5：植物时序预测有效性诊断实验

**工程验收：PASS；研究判断：REVIEW。**

执行日期：2026-10-11，Asia/Shanghai。这里只交付 M1.5；没有 JEPA、三维实验、合并、提交或推送。

## 1. 结论与预先固定的 M2 决策

现有证据不足以满足预先固定的继续规则，也未在所有任务上排除有实际意义的历史增益。时间、结构与真实动态的贡献需要分别解释；不把监督 GRU 的弱优势当成对 Temporal-JEPA 的否定。建议先做预固定独立确认和任务核验，不进入 M2 训练。

- 预先指定 day5 叶数 Plant Macro MAE：单帧 1.2398、真实 GRU 1.2259、重复帧 GRU 1.2784；纯时间神经网络 1.6832，说明 RGB 存在明显预测信息，但不等于证明历史视觉变化有增量价值。
- 真实历史较重复帧仅降低 0.0525（4.11%），配对 CI95 [-0.1095, 0.0010]；较单帧降低 0.0139，CI95 [-0.0539, 0.0271]，种子方向不全一致；不满足实际意义和不确定性规则。
- 显式差分 day5 叶数相对单帧 delta=0.0269（负值才为改善），没有提供稳定增益；容量家族没有 Holm 校正后成立的真实历史优势。
- 3/5/7 天均可构建，Test 可用植物分别46/46/44；不是原 +2/+4/+8 的同义转换。日尺度预测越远，绝对误差增大，但本次没有因此出现稳定的历史序列相对增量。
- 加入时间条件或未来实际时间差没有一致改善 RGB 预测；不能将 M1 的弱时序结果简单归结为‘模型只读了时间’。日历/拍摄安排混杂仍是需要独立批次确认的风险，而不是已经证实的唯一原因。

主任务在 M1.5 训练前固定为 **day5 可见叶数**。真实 GRU 必须同时较重复帧和单帧降低 ≥0.10 MAE 且 ≥5%，三种子全部同向、配对植物 CI95 上界<0、capacity 家族 Holm p<0.05，才触发 CONTINUE。面积阈值为 0.002（0.2 个百分点）；这些是工程意义阈值，不是已获生物学专家确认的效应界值。
若主任务不满足，不从 Test 子组或其他终点挑一个有利结果替代主任务。PIVOT 仅在全部容量、显式差分和 RGB-only 时序比较的植物区间均排除所设最小增益时触发，否则 REVIEW。

| task | target | first | second | delta_mae | ci95_low | ci95_high | relative_gain | all_seeds_favor_first | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| day5 | leaf_count | gru_full | gru_repeat | -0.05251 | -0.10953 | 0.00102 | 0.04107 | True | 1.00000 |
| day5 | leaf_count | gru_full | mlp_full | -0.01395 | -0.05387 | 0.02706 | 0.01125 | False | 1.00000 |

day5 每种子的真实 GRU 减对照 MAE：

| target | second | seed42_delta | seed43_delta | seed44_delta |
| --- | --- | --- | --- | --- |
| leaf_count | gru_repeat | -0.08218 | -0.02441 | -0.05094 |
| projected_area_ratio | gru_repeat | -0.00156 | -0.00105 | 0.00038 |
| leaf_count | mlp_full | -0.04424 | -0.00016 | 0.00254 |
| projected_area_ratio | mlp_full | -0.00094 | 0.00010 | -0.00029 |

**背景限制：M1 已查看 Test；M1.5 是在此背景下开展的探索性诊断。这里的规则是本轮训练前锁定，不是对整个研究的事前注册，也不使原 Test 恢复成未看过的确认集。** 后续若要声称新方法确证性优势，需要预先固定的额外划分、从未用于设计/选择的留出基因型，或外部数据集。

## 2. 原 M1 复核与只读保护

起始 Git commit `fb789b73e3a95ac677847251b6314b5dbb718c95`；当前分支 `main`，起始 main 与本地 origin/main 跟踪引用一致且工作区干净；远端实时状态因 GitHub 连接超时未验证。M1.5 仅添加隔离文件，没有创建分支、修改历史代码或旧结果。完整状态在 legacy_snapshot.json / git_state.json。
保护 497 个既有文件；启动/交付均按 M0 manifest 校验 34164 个 RGB/mask 原始文件，改变数 0。原 M0 63 文件保护也独立通过。
M0：318 株、17,056 张有效图像、26 个既有隔离观测。保留 seed42 植物划分：Train 219 / Val 53 / Test 46，21 个基因型；三帧历史坚持原始编号连续，不压缩缺失/隔离帧。
重新从 27 份旧 Test 预测与 plant_level_errors.csv 计算指标，核对原 baseline_comparison.csv；18 个旧神经 checkpoint 及 Train/Val 来源全部存在并通过。CSV 中原 float32 标签回读时还原 float32 后核对，避免把十进制序列化误差误判成实验变化。原数值不覆盖。

| model | horizon | target | plant_macro_mae_mean | plant_macro_mae_std |
| --- | --- | --- | --- | --- |
| B2a | 2 | leaf_count | 0.94909 | 0.00900 |
| B2a | 2 | projected_area_ratio | 0.00969 | 0.00038 |
| B2a | 4 | leaf_count | 0.96831 | 0.02320 |
| B2a | 4 | projected_area_ratio | 0.01317 | 0.00088 |
| B2a | 8 | leaf_count | 1.08662 | 0.00658 |
| B2a | 8 | projected_area_ratio | 0.02008 | 0.00026 |
| B2b | 2 | leaf_count | 0.91032 | 0.01133 |
| B2b | 2 | projected_area_ratio | 0.01007 | 0.00055 |
| B2b | 4 | leaf_count | 0.96271 | 0.01087 |
| B2b | 4 | projected_area_ratio | 0.01298 | 0.00046 |
| B2b | 8 | leaf_count | 1.11850 | 0.05949 |
| B2b | 8 | projected_area_ratio | 0.01958 | 0.00090 |

原 B2a 101,634 参数，B2b 561,922 参数，约 5.53 倍容量；M1 的已训练 GRU 重复帧干预仅是推理输入干预，不能充当本轮从头训练的容量对照。原 B0/B1 使用历史真实表型，是 Oracle，不与图像模型作同信息量优势宣称。
原部分脚本/文案使用 RGB-only 称谓，但实际 B2a/B2b 前向接收10维时间条件，不能视为严格 RGB-only；本轮 mlp_rgb/gru_rgb 才明确去除全部时间槽。
冻结缓存 `outputs/m1/features/037935a36cdfd52a`：17,056×384 float32；DINOv2 ViT-S/14 source commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`，权重 SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`。全部冻结，不重新编码或拟合 Test 特征标准化。预处理是等比例 bicubic + 对称 letterbox 224，不做中心裁剪。缓存矩阵/manifest/逐行哈希再次验证。

## 3. 实验 A：容量和训练预算受控的历史信息

| 名称 | M1.5 模型 | 输入与结构 |
| --- | --- | --- |
| A1 | mlp_full | 沿用 B2a：最后 RGB 特征 + 同套时间条件，394→256→2 |
| A2 | gru_full | 沿用 B2b：GRU(384,256,1) + 266→256→2 |
| A3 | gru_repeat | 三次重复最后 RGB 特征；与 A2 完全同结构、同时间条件，从头独立训练 |
| A4 | diff_full | 三帧特征及两个逐帧差分共 1,920 维，追加10维条件；1930→64→2 轻量 MLP（123,714 参数） |

A1–A4 全部重新训练、3 seeds 42/43/44。A2/A3 同初始化参数 SHA、完整随机 batch 顺序 SHA、学习率、优化器、时间标准化、样本权重和 100 轮/步数预算。关闭早停，但各自仍只用 Val 选择最优 epoch；capacity_training_audit.csv 核对 18 组真实/重复帧训练配对。
A1/A2 新结果与原 M1 不要求逐值一致：M1 使用 patience=18 早停，本轮统一固定 100 轮，可能发现更晚的 Val 最优 epoch；旧 checkpoint 原样保存。A1 和 A2 架构不同，因此最关键的容量控制是 A2 对 A3，不能只靠 A2 对 A1。
A4 的差分是三帧的确定性函数，没有新增独立信息；它检验显式变化坐标和归纳偏置是否帮助预测，不证明 GRU 顺序或自监督表征无用。A4 与单帧结构/参数数不完全相同；若增益只出现在 A4，后续还需 A4 的重复帧容量对照，不仅凭该结果认定动态机制。

三个历史帧覆盖的是短观测区间，并没有因为目标改为5/7天就延长历史；实际历史跨度统计见 history_span_summary.csv。这可能限制动态估计，但不是已证实的因果原因。本次未按Test调整历史长度。

| task | split | min | median | max |
| --- | --- | --- | --- | --- |
| day3 | test | 10.99528 | 13.00000 | 26.84500 |
| day5 | test | 10.99583 | 13.00000 | 26.84500 |
| day7 | test | 10.99639 | 12.99972 | 26.84500 |
| offset2 | test | 10.98139 | 13.00000 | 26.84500 |
| offset4 | test | 10.99028 | 13.00000 | 26.84500 |
| offset8 | test | 10.99083 | 13.00000 | 26.84500 |

| task | model | target | plants | MAE ± seed SD [plant CI95] |
| --- | --- | --- | --- | --- |
| offset2 | diff_full | leaf_count | 46 | 0.89971 ± 0.00793 [0.81478, 1.01130] |
| offset2 | diff_full | projected_area_ratio | 46 | 0.01068 ± 0.00017 [0.00943, 0.01205] |
| offset2 | gru_full | leaf_count | 46 | 0.90950 ± 0.01100 [0.81492, 1.03464] |
| offset2 | gru_full | projected_area_ratio | 46 | 0.01007 ± 0.00058 [0.00874, 0.01151] |
| offset2 | gru_repeat | leaf_count | 46 | 0.92731 ± 0.00910 [0.82348, 1.06817] |
| offset2 | gru_repeat | projected_area_ratio | 46 | 0.00933 ± 0.00013 [0.00825, 0.01053] |
| offset2 | mlp_full | leaf_count | 46 | 0.94507 ± 0.00698 [0.83964, 1.08656] |
| offset2 | mlp_full | projected_area_ratio | 46 | 0.00970 ± 0.00034 [0.00863, 0.01090] |
| offset4 | diff_full | leaf_count | 46 | 0.92410 ± 0.01777 [0.82620, 1.05142] |
| offset4 | diff_full | projected_area_ratio | 46 | 0.01366 ± 0.00096 [0.01174, 0.01566] |
| offset4 | gru_full | leaf_count | 46 | 0.96429 ± 0.00630 [0.86138, 1.10079] |
| offset4 | gru_full | projected_area_ratio | 46 | 0.01260 ± 0.00037 [0.01061, 0.01474] |
| offset4 | gru_repeat | leaf_count | 46 | 0.96534 ± 0.01661 [0.85715, 1.11633] |
| offset4 | gru_repeat | projected_area_ratio | 46 | 0.01234 ± 0.00066 [0.01056, 0.01424] |
| offset4 | mlp_full | leaf_count | 46 | 0.96526 ± 0.01257 [0.87201, 1.09080] |
| offset4 | mlp_full | projected_area_ratio | 46 | 0.01334 ± 0.00058 [0.01151, 0.01532] |
| offset8 | diff_full | leaf_count | 46 | 1.06753 ± 0.05127 [0.93111, 1.23237] |
| offset8 | diff_full | projected_area_ratio | 46 | 0.01995 ± 0.00044 [0.01624, 0.02426] |
| offset8 | gru_full | leaf_count | 46 | 1.10092 ± 0.00465 [0.95775, 1.27187] |
| offset8 | gru_full | projected_area_ratio | 46 | 0.01923 ± 0.00057 [0.01538, 0.02365] |
| offset8 | gru_repeat | leaf_count | 46 | 1.12030 ± 0.01752 [0.97271, 1.29723] |
| offset8 | gru_repeat | projected_area_ratio | 46 | 0.01931 ± 0.00031 [0.01550, 0.02363] |
| offset8 | mlp_full | leaf_count | 46 | 1.09999 ± 0.02418 [0.96714, 1.25272] |
| offset8 | mlp_full | projected_area_ratio | 46 | 0.01981 ± 0.00076 [0.01621, 0.02387] |

容量配对结果：first-second；负值有利于真实历史。以下 CI95 为探索性区间，Holm 是 capacity 家族 24 个 task×target×contrast 的双侧符号翻转 p 校正，不将未校正子组显著性用于决策。

| task | target | first | second | delta_mae | ci95_low | ci95_high | relative_gain | all_seeds_favor_first | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| day3 | leaf_count | gru_full | gru_repeat | 0.01595 | -0.01817 | 0.05205 | -0.01514 | False | 1.00000 |
| day3 | projected_area_ratio | gru_full | gru_repeat | 0.00054 | -0.00016 | 0.00130 | -0.03097 | False | 1.00000 |
| day3 | leaf_count | gru_full | mlp_full | 0.00714 | -0.03867 | 0.05760 | -0.00672 | False | 1.00000 |
| day3 | projected_area_ratio | gru_full | mlp_full | -0.00060 | -0.00130 | 0.00008 | 0.03233 | False | 1.00000 |
| day5 | leaf_count | gru_full | gru_repeat | -0.05251 | -0.10953 | 0.00102 | 0.04107 | True | 1.00000 |
| day5 | projected_area_ratio | gru_full | gru_repeat | -0.00074 | -0.00195 | 0.00041 | 0.02843 | False | 1.00000 |
| day5 | leaf_count | gru_full | mlp_full | -0.01395 | -0.05387 | 0.02706 | 0.01125 | False | 1.00000 |
| day5 | projected_area_ratio | gru_full | mlp_full | -0.00038 | -0.00177 | 0.00094 | 0.01468 | False | 1.00000 |
| day7 | leaf_count | gru_full | gru_repeat | 0.00986 | -0.04821 | 0.06900 | -0.00709 | False | 1.00000 |
| day7 | projected_area_ratio | gru_full | gru_repeat | 0.00064 | -0.00036 | 0.00158 | -0.01959 | False | 1.00000 |
| day7 | leaf_count | gru_full | mlp_full | 0.05085 | -0.03698 | 0.14074 | -0.03770 | False | 1.00000 |
| day7 | projected_area_ratio | gru_full | mlp_full | -0.00115 | -0.00327 | 0.00087 | 0.03330 | False | 1.00000 |
| offset2 | leaf_count | gru_full | gru_repeat | -0.01781 | -0.06248 | 0.02458 | 0.01921 | False | 1.00000 |
| offset2 | projected_area_ratio | gru_full | gru_repeat | 0.00075 | 0.00021 | 0.00131 | -0.07989 | False | 0.28559 |
| offset2 | leaf_count | gru_full | mlp_full | -0.03558 | -0.08736 | 0.01409 | 0.03764 | True | 1.00000 |
| offset2 | projected_area_ratio | gru_full | mlp_full | 0.00037 | -0.00020 | 0.00099 | -0.03865 | False | 1.00000 |
| offset4 | leaf_count | gru_full | gru_repeat | -0.00105 | -0.03422 | 0.02812 | 0.00109 | False | 1.00000 |
| offset4 | projected_area_ratio | gru_full | gru_repeat | 0.00026 | -0.00023 | 0.00076 | -0.02081 | False | 1.00000 |
| offset4 | leaf_count | gru_full | mlp_full | -0.00097 | -0.03464 | 0.03411 | 0.00100 | False | 1.00000 |
| offset4 | projected_area_ratio | gru_full | mlp_full | -0.00075 | -0.00136 | -0.00009 | 0.05594 | True | 0.62672 |
| offset8 | leaf_count | gru_full | gru_repeat | -0.01937 | -0.05108 | 0.01208 | 0.01729 | True | 1.00000 |
| offset8 | projected_area_ratio | gru_full | gru_repeat | -0.00007 | -0.00063 | 0.00049 | 0.00386 | False | 1.00000 |
| offset8 | leaf_count | gru_full | mlp_full | 0.00094 | -0.04678 | 0.04867 | -0.00085 | False | 1.00000 |
| offset8 | projected_area_ratio | gru_full | mlp_full | -0.00058 | -0.00129 | 0.00013 | 0.02936 | True | 1.00000 |

显式特征差分对单帧的结果（独立 differences 家族校正）：

| task | target | first | second | delta_mae | ci95_low | ci95_high | relative_gain | all_seeds_favor_first | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| day3 | leaf_count | diff_full | mlp_full | -0.01104 | -0.05326 | 0.03051 | 0.01039 | False | 1.00000 |
| day3 | projected_area_ratio | diff_full | mlp_full | 0.00032 | -0.00075 | 0.00142 | -0.01702 | False | 1.00000 |
| day5 | leaf_count | diff_full | mlp_full | 0.02694 | -0.02841 | 0.08830 | -0.02173 | False | 1.00000 |
| day5 | projected_area_ratio | diff_full | mlp_full | 0.00004 | -0.00119 | 0.00120 | -0.00155 | False | 1.00000 |
| day7 | leaf_count | diff_full | mlp_full | 0.01058 | -0.10027 | 0.12417 | -0.00784 | False | 1.00000 |
| day7 | projected_area_ratio | diff_full | mlp_full | -0.00062 | -0.00235 | 0.00104 | 0.01777 | False | 1.00000 |
| offset2 | leaf_count | diff_full | mlp_full | -0.04536 | -0.09828 | 0.00592 | 0.04800 | True | 1.00000 |
| offset2 | projected_area_ratio | diff_full | mlp_full | 0.00098 | 0.00032 | 0.00164 | -0.10072 | False | 0.06300 |
| offset4 | leaf_count | diff_full | mlp_full | -0.04116 | -0.06846 | -0.01139 | 0.04264 | True | 0.09350 |
| offset4 | projected_area_ratio | diff_full | mlp_full | 0.00032 | -0.00031 | 0.00095 | -0.02370 | False | 1.00000 |
| offset8 | leaf_count | diff_full | mlp_full | -0.03246 | -0.07498 | 0.01002 | 0.02951 | False | 1.00000 |
| offset8 | projected_area_ratio | diff_full | mlp_full | 0.00014 | -0.00088 | 0.00116 | -0.00704 | False | 1.00000 |

## 4. 实验 B：RGB / Time / RGB+Time 与不同预测任务

| 条件 | 模型 | 推理时使用的信息 |
| --- | --- | --- |
| RGB-only | mlp_rgb / gru_rgb | RGB 历史；10个时间槽全部置零。每个跨度独立模型本身就是明确任务标识，没有日期、历史间隔或未来实际时间差 |
| Time-only neural | time_full | 同一 B2a 结构但 RGB 恒为零，只有同套10维时间变量。名义参数数与 B2a 相同，实际有效时间网络为10→256→2 |
| RGB+Time | mlp_full / gru_full | 与 M1 相同输入条件；offset 任务包括目标实际时间差 |
| Offset-only-query | mlp_offset / gru_offset / time_offset | 将未来实际时间差槽置零，只保留可观测历史时间、日历与固定观测跨度任务标识 |
| No-calendar offset query | mlp_relative / gru_relative / time_relative | 再移除 calendar-day/hour 的4个周期槽，保留相对观测年龄、原始编号和历史间隔 |

M1 十个条件为：距首条记录的天数、最后原始观测编号、两个历史间隔、实际未来天数、观测跨度、年内日期 sin/cos、一天时刻 sin/cos。它们不是播种年龄，也没有直接输入 genotype ID。
**未来实际时间差的可用性：数据索引确实提供了它，但没有证据证明 offset +2/+4/+8 的每次实际采集间隔在真实部署前已知。** 因而 full-offset 是已知/事后提供目标时间的条件预测；offset-only-query 才是不依赖未来实际间隔的观测偏移任务。二者标签和窗口完全相同，但可用信息/部署任务不同，不能把 full 的优势自动称作无条件生长预测，也不凭此断言发生未来表型泄漏。
calendar/hour、首记录年龄和编号可能编码 May/Aug 采集队列、拍摄节律、基因型/生长阶段分布；去 calendar 的比较是敏感性分析，不是因果批次校正。以下为逐株采集首月覆盖；完整基因型×月份×split 见 schedule_genotype_coverage.csv。
21个基因型中有 13 个只见于一个首记录月份；因此日历月份与这些基因型/队列相互混杂，不能从当前植物划分推断跨月份或未见基因型泛化，也不能把月份效应直接认定成纯拍摄批次因果效应。
即使真实帧优于重复帧，这也首先证明历史图像的增量信息，尚不能唯一分解为真实生长动态、多个视图的降噪/可见性改善或取景/光照变化；本轮没有从头训练的置序/无序池化控制，不把旧 M1 推理置乱干预当成这一证明。

| split | first_record_month | plants |
| --- | --- | --- |
| test | 2022-05 | 25 |
| test | 2022-08 | 21 |
| train | 2022-05 | 99 |
| train | 2022-08 | 120 |
| val | 2022-05 | 26 |
| val | 2022-08 | 27 |

| task | model | target | plants | MAE ± seed SD [plant CI95] |
| --- | --- | --- | --- | --- |
| offset2 | gru_full | leaf_count | 46 | 0.90950 ± 0.01100 [0.81492, 1.03464] |
| offset2 | gru_full | projected_area_ratio | 46 | 0.01007 ± 0.00058 [0.00874, 0.01151] |
| offset2 | gru_offset | leaf_count | 46 | 0.91386 ± 0.02077 [0.82177, 1.03903] |
| offset2 | gru_offset | projected_area_ratio | 46 | 0.00969 ± 0.00021 [0.00845, 0.01103] |
| offset2 | gru_relative | leaf_count | 46 | 0.90649 ± 0.01916 [0.81147, 1.03300] |
| offset2 | gru_relative | projected_area_ratio | 46 | 0.00997 ± 0.00027 [0.00867, 0.01137] |
| offset2 | gru_rgb | leaf_count | 46 | 0.88615 ± 0.01768 [0.79621, 1.01057] |
| offset2 | gru_rgb | projected_area_ratio | 46 | 0.00993 ± 0.00019 [0.00867, 0.01131] |
| offset2 | mlp_full | leaf_count | 46 | 0.94507 ± 0.00698 [0.83964, 1.08656] |
| offset2 | mlp_full | projected_area_ratio | 46 | 0.00970 ± 0.00034 [0.00863, 0.01090] |
| offset2 | mlp_offset | leaf_count | 46 | 0.95327 ± 0.00843 [0.84400, 1.10251] |
| offset2 | mlp_offset | projected_area_ratio | 46 | 0.00956 ± 0.00021 [0.00851, 0.01073] |
| offset2 | mlp_relative | leaf_count | 46 | 0.94037 ± 0.02295 [0.83518, 1.08644] |
| offset2 | mlp_relative | projected_area_ratio | 46 | 0.00991 ± 0.00044 [0.00879, 0.01114] |
| offset2 | mlp_rgb | leaf_count | 46 | 0.94558 ± 0.01820 [0.84171, 1.08369] |
| offset2 | mlp_rgb | projected_area_ratio | 46 | 0.00978 ± 0.00029 [0.00865, 0.01105] |
| offset2 | time_full | leaf_count | 46 | 1.44416 ± 0.00404 [1.24877, 1.65228] |
| offset2 | time_full | projected_area_ratio | 46 | 0.03057 ± 0.00009 [0.02730, 0.03388] |
| offset2 | time_offset | leaf_count | 46 | 1.44855 ± 0.00527 [1.25003, 1.65906] |
| offset2 | time_offset | projected_area_ratio | 46 | 0.03048 ± 0.00020 [0.02727, 0.03375] |
| offset2 | time_relative | leaf_count | 46 | 1.51155 ± 0.00118 [1.28572, 1.76187] |
| offset2 | time_relative | projected_area_ratio | 46 | 0.02974 ± 0.00013 [0.02637, 0.03319] |
| offset4 | gru_full | leaf_count | 46 | 0.96429 ± 0.00630 [0.86138, 1.10079] |
| offset4 | gru_full | projected_area_ratio | 46 | 0.01260 ± 0.00037 [0.01061, 0.01474] |
| offset4 | gru_offset | leaf_count | 46 | 0.96263 ± 0.01254 [0.85911, 1.10358] |
| offset4 | gru_offset | projected_area_ratio | 46 | 0.01286 ± 0.00059 [0.01085, 0.01504] |
| offset4 | gru_relative | leaf_count | 46 | 0.95660 ± 0.01374 [0.85289, 1.09314] |
| offset4 | gru_relative | projected_area_ratio | 46 | 0.01316 ± 0.00067 [0.01110, 0.01542] |
| offset4 | gru_rgb | leaf_count | 46 | 0.93910 ± 0.01452 [0.83953, 1.06517] |
| offset4 | gru_rgb | projected_area_ratio | 46 | 0.01236 ± 0.00034 [0.01042, 0.01452] |
| offset4 | mlp_full | leaf_count | 46 | 0.96526 ± 0.01257 [0.87201, 1.09080] |
| offset4 | mlp_full | projected_area_ratio | 46 | 0.01334 ± 0.00058 [0.01151, 0.01532] |
| offset4 | mlp_offset | leaf_count | 46 | 0.96476 ± 0.03818 [0.86715, 1.09432] |
| offset4 | mlp_offset | projected_area_ratio | 46 | 0.01279 ± 0.00077 [0.01106, 0.01463] |
| offset4 | mlp_relative | leaf_count | 46 | 0.97730 ± 0.02844 [0.87456, 1.11867] |
| offset4 | mlp_relative | projected_area_ratio | 46 | 0.01309 ± 0.00104 [0.01133, 0.01498] |
| offset4 | mlp_rgb | leaf_count | 46 | 0.96243 ± 0.00749 [0.86484, 1.08992] |
| offset4 | mlp_rgb | projected_area_ratio | 46 | 0.01298 ± 0.00069 [0.01117, 0.01493] |
| offset4 | time_full | leaf_count | 46 | 1.47638 ± 0.00076 [1.27569, 1.68872] |
| offset4 | time_full | projected_area_ratio | 46 | 0.03234 ± 0.00003 [0.02879, 0.03592] |
| offset4 | time_offset | leaf_count | 46 | 1.47793 ± 0.00248 [1.27705, 1.69051] |
| offset4 | time_offset | projected_area_ratio | 46 | 0.03215 ± 0.00007 [0.02870, 0.03563] |
| offset4 | time_relative | leaf_count | 46 | 1.53967 ± 0.00386 [1.30920, 1.79488] |
| offset4 | time_relative | projected_area_ratio | 46 | 0.03140 ± 0.00021 [0.02786, 0.03500] |
| offset8 | gru_full | leaf_count | 46 | 1.10092 ± 0.00465 [0.95775, 1.27187] |
| offset8 | gru_full | projected_area_ratio | 46 | 0.01923 ± 0.00057 [0.01538, 0.02365] |
| offset8 | gru_offset | leaf_count | 46 | 1.10098 ± 0.05318 [0.96021, 1.26367] |
| offset8 | gru_offset | projected_area_ratio | 46 | 0.01957 ± 0.00018 [0.01551, 0.02422] |
| offset8 | gru_relative | leaf_count | 46 | 1.08410 ± 0.02746 [0.95170, 1.24326] |
| offset8 | gru_relative | projected_area_ratio | 46 | 0.01975 ± 0.00115 [0.01575, 0.02435] |
| offset8 | gru_rgb | leaf_count | 46 | 1.08417 ± 0.02741 [0.95027, 1.24272] |
| offset8 | gru_rgb | projected_area_ratio | 46 | 0.02003 ± 0.00119 [0.01581, 0.02486] |
| offset8 | mlp_full | leaf_count | 46 | 1.09999 ± 0.02418 [0.96714, 1.25272] |
| offset8 | mlp_full | projected_area_ratio | 46 | 0.01981 ± 0.00076 [0.01621, 0.02387] |
| offset8 | mlp_offset | leaf_count | 46 | 1.09565 ± 0.02098 [0.96260, 1.25554] |
| offset8 | mlp_offset | projected_area_ratio | 46 | 0.02025 ± 0.00047 [0.01649, 0.02454] |
| offset8 | mlp_relative | leaf_count | 46 | 1.10745 ± 0.01647 [0.97526, 1.26163] |
| offset8 | mlp_relative | projected_area_ratio | 46 | 0.02029 ± 0.00071 [0.01651, 0.02456] |
| offset8 | mlp_rgb | leaf_count | 46 | 1.09843 ± 0.02082 [0.96254, 1.26284] |
| offset8 | mlp_rgb | projected_area_ratio | 46 | 0.02041 ± 0.00058 [0.01670, 0.02458] |
| offset8 | time_full | leaf_count | 46 | 1.53745 ± 0.00384 [1.32234, 1.76365] |
| offset8 | time_full | projected_area_ratio | 46 | 0.03633 ± 0.00007 [0.03228, 0.04046] |
| offset8 | time_offset | leaf_count | 46 | 1.53867 ± 0.00305 [1.32251, 1.76582] |
| offset8 | time_offset | projected_area_ratio | 46 | 0.03606 ± 0.00008 [0.03207, 0.04012] |
| offset8 | time_relative | leaf_count | 46 | 1.60364 ± 0.00208 [1.34927, 1.87604] |
| offset8 | time_relative | projected_area_ratio | 46 | 0.03579 ± 0.00017 [0.03174, 0.03990] |

时间条件配对效应（负值有利于 first；不同信息合同的比较不作同一部署任务的能力宣称）：

| task | target | first | second | delta_mae | ci95_low | ci95_high | relative_gain | all_seeds_favor_first | holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| day3 | leaf_count | mlp_full | mlp_rgb | 0.00765 | -0.02881 | 0.04236 | -0.00725 | False | 1.00000 |
| day3 | projected_area_ratio | mlp_full | mlp_rgb | 0.00020 | -0.00043 | 0.00082 | -0.01064 | False | 1.00000 |
| day3 | leaf_count | gru_full | gru_rgb | 0.01347 | -0.01573 | 0.04336 | -0.01275 | False | 1.00000 |
| day3 | projected_area_ratio | gru_full | gru_rgb | -0.00020 | -0.00081 | 0.00043 | 0.01120 | False | 1.00000 |
| day3 | leaf_count | mlp_full | time_full | -0.46215 | -0.65468 | -0.26999 | 0.30315 | True | 0.00360 |
| day3 | projected_area_ratio | mlp_full | time_full | -0.01679 | -0.02119 | -0.01244 | 0.47514 | True | 0.00360 |
| day5 | leaf_count | mlp_full | mlp_rgb | -0.02371 | -0.04872 | 0.00148 | 0.01876 | True | 1.00000 |
| day5 | projected_area_ratio | mlp_full | mlp_rgb | -0.00055 | -0.00092 | -0.00021 | 0.02087 | True | 0.15369 |
| day5 | leaf_count | gru_full | gru_rgb | -0.03227 | -0.08729 | 0.01941 | 0.02565 | False | 1.00000 |
| day5 | projected_area_ratio | gru_full | gru_rgb | 0.00010 | -0.00100 | 0.00124 | -0.00402 | False | 1.00000 |
| day5 | leaf_count | mlp_full | time_full | -0.44335 | -0.66820 | -0.22544 | 0.26340 | True | 0.02790 |
| day5 | projected_area_ratio | mlp_full | time_full | -0.01445 | -0.01989 | -0.00892 | 0.35889 | True | 0.00360 |
| day7 | leaf_count | mlp_full | mlp_rgb | 0.00041 | -0.02496 | 0.02739 | -0.00031 | False | 1.00000 |
| day7 | projected_area_ratio | mlp_full | mlp_rgb | 0.00125 | 0.00058 | 0.00195 | -0.03730 | False | 0.05700 |
| day7 | leaf_count | gru_full | gru_rgb | 0.00186 | -0.03707 | 0.04171 | -0.00133 | False | 1.00000 |
| day7 | projected_area_ratio | gru_full | gru_rgb | 0.00003 | -0.00057 | 0.00064 | -0.00100 | False | 1.00000 |
| day7 | leaf_count | mlp_full | time_full | -0.55824 | -0.84772 | -0.27789 | 0.29271 | True | 0.03965 |
| day7 | projected_area_ratio | mlp_full | time_full | -0.01189 | -0.01874 | -0.00489 | 0.25531 | True | 0.11504 |
| offset2 | leaf_count | mlp_full | mlp_rgb | -0.00050 | -0.03466 | 0.03498 | 0.00053 | False | 1.00000 |
| offset2 | projected_area_ratio | mlp_full | mlp_rgb | -0.00008 | -0.00040 | 0.00023 | 0.00866 | False | 1.00000 |
| offset2 | leaf_count | gru_full | gru_rgb | 0.02335 | -0.01430 | 0.06272 | -0.02635 | False | 1.00000 |
| offset2 | projected_area_ratio | gru_full | gru_rgb | 0.00014 | -0.00029 | 0.00053 | -0.01438 | False | 1.00000 |
| offset2 | leaf_count | mlp_full | time_full | -0.49909 | -0.69551 | -0.30794 | 0.34559 | True | 0.00360 |
| offset2 | projected_area_ratio | mlp_full | time_full | -0.02087 | -0.02419 | -0.01758 | 0.68270 | True | 0.00360 |
| offset2 | leaf_count | mlp_full | mlp_offset | -0.00819 | -0.02377 | 0.00676 | 0.00859 | True | 1.00000 |
| offset2 | projected_area_ratio | mlp_full | mlp_offset | 0.00014 | -0.00010 | 0.00039 | -0.01450 | False | 1.00000 |
| offset2 | leaf_count | gru_full | gru_offset | -0.00436 | -0.02581 | 0.01826 | 0.00477 | False | 1.00000 |
| offset2 | projected_area_ratio | gru_full | gru_offset | 0.00039 | 0.00008 | 0.00071 | -0.03980 | False | 1.00000 |
| offset2 | leaf_count | time_full | time_offset | -0.00439 | -0.01567 | 0.00346 | 0.00303 | True | 1.00000 |
| offset2 | projected_area_ratio | time_full | time_offset | 0.00008 | -0.00010 | 0.00030 | -0.00269 | False | 1.00000 |
| offset2 | leaf_count | mlp_offset | mlp_relative | 0.01289 | -0.00733 | 0.03347 | -0.01371 | False | 1.00000 |
| offset2 | projected_area_ratio | mlp_offset | mlp_relative | -0.00035 | -0.00077 | 0.00002 | 0.03576 | False | 1.00000 |
| offset2 | leaf_count | gru_offset | gru_relative | 0.00736 | -0.01164 | 0.02553 | -0.00812 | False | 1.00000 |
| offset2 | projected_area_ratio | gru_offset | gru_relative | -0.00028 | -0.00066 | 0.00008 | 0.02833 | False | 1.00000 |
| offset2 | leaf_count | time_offset | time_relative | -0.06300 | -0.14925 | -0.00035 | 0.04168 | True | 1.00000 |
| offset2 | projected_area_ratio | time_offset | time_relative | 0.00074 | -0.00068 | 0.00235 | -0.02492 | False | 1.00000 |
| offset4 | leaf_count | mlp_full | mlp_rgb | 0.00283 | -0.02446 | 0.02778 | -0.00294 | False | 1.00000 |
| offset4 | projected_area_ratio | mlp_full | mlp_rgb | 0.00036 | 0.00004 | 0.00068 | -0.02792 | False | 1.00000 |
| offset4 | leaf_count | gru_full | gru_rgb | 0.02519 | -0.00554 | 0.05656 | -0.02682 | False | 1.00000 |
| offset4 | projected_area_ratio | gru_full | gru_rgb | 0.00023 | -0.00023 | 0.00069 | -0.01896 | False | 1.00000 |
| offset4 | leaf_count | mlp_full | time_full | -0.51112 | -0.70734 | -0.31960 | 0.34620 | True | 0.00360 |
| offset4 | projected_area_ratio | mlp_full | time_full | -0.01899 | -0.02268 | -0.01536 | 0.58736 | True | 0.00360 |
| offset4 | leaf_count | mlp_full | mlp_offset | 0.00050 | -0.01359 | 0.01437 | -0.00051 | False | 1.00000 |
| offset4 | projected_area_ratio | mlp_full | mlp_offset | 0.00056 | 0.00030 | 0.00082 | -0.04346 | False | 0.00360 |
| offset4 | leaf_count | gru_full | gru_offset | 0.00166 | -0.02030 | 0.02403 | -0.00172 | False | 1.00000 |
| offset4 | projected_area_ratio | gru_full | gru_offset | -0.00026 | -0.00064 | 0.00013 | 0.02002 | False | 1.00000 |
| offset4 | leaf_count | time_full | time_offset | -0.00155 | -0.01214 | 0.00697 | 0.00105 | False | 1.00000 |
| offset4 | projected_area_ratio | time_full | time_offset | 0.00019 | -0.00002 | 0.00045 | -0.00591 | False | 1.00000 |
| offset4 | leaf_count | mlp_offset | mlp_relative | -0.01254 | -0.03320 | 0.00640 | 0.01283 | True | 1.00000 |
| offset4 | projected_area_ratio | mlp_offset | mlp_relative | -0.00030 | -0.00051 | -0.00010 | 0.02320 | False | 0.36478 |
| offset4 | leaf_count | gru_offset | gru_relative | 0.00603 | -0.00832 | 0.02040 | -0.00630 | False | 1.00000 |
| offset4 | projected_area_ratio | gru_offset | gru_relative | -0.00031 | -0.00053 | -0.00008 | 0.02331 | False | 0.71116 |
| offset4 | leaf_count | time_offset | time_relative | -0.06174 | -0.15329 | 0.00512 | 0.04010 | True | 1.00000 |
| offset4 | projected_area_ratio | time_offset | time_relative | 0.00075 | -0.00081 | 0.00255 | -0.02373 | False | 1.00000 |
| offset8 | leaf_count | mlp_full | mlp_rgb | 0.00155 | -0.02647 | 0.02923 | -0.00141 | False | 1.00000 |
| offset8 | projected_area_ratio | mlp_full | mlp_rgb | -0.00060 | -0.00135 | 0.00012 | 0.02916 | True | 1.00000 |
| offset8 | leaf_count | gru_full | gru_rgb | 0.01676 | -0.01568 | 0.04900 | -0.01546 | False | 1.00000 |
| offset8 | projected_area_ratio | gru_full | gru_rgb | -0.00080 | -0.00156 | -0.00002 | 0.03990 | False | 1.00000 |
| offset8 | leaf_count | mlp_full | time_full | -0.43746 | -0.64320 | -0.23553 | 0.28454 | True | 0.01890 |
| offset8 | projected_area_ratio | mlp_full | time_full | -0.01652 | -0.02098 | -0.01203 | 0.45465 | True | 0.00360 |
| offset8 | leaf_count | mlp_full | mlp_offset | 0.00434 | -0.02181 | 0.02925 | -0.00396 | False | 1.00000 |
| offset8 | projected_area_ratio | mlp_full | mlp_offset | -0.00044 | -0.00082 | -0.00009 | 0.02154 | True | 1.00000 |
| offset8 | leaf_count | gru_full | gru_offset | -0.00006 | -0.01788 | 0.01946 | 0.00005 | False | 1.00000 |
| offset8 | projected_area_ratio | gru_full | gru_offset | -0.00034 | -0.00095 | 0.00024 | 0.01726 | False | 1.00000 |
| offset8 | leaf_count | time_full | time_offset | -0.00122 | -0.01521 | 0.00875 | 0.00079 | False | 1.00000 |
| offset8 | projected_area_ratio | time_full | time_offset | 0.00027 | 0.00005 | 0.00056 | -0.00759 | False | 1.00000 |
| offset8 | leaf_count | mlp_offset | mlp_relative | -0.01180 | -0.03269 | 0.01005 | 0.01066 | True | 1.00000 |
| offset8 | projected_area_ratio | mlp_offset | mlp_relative | -0.00004 | -0.00041 | 0.00031 | 0.00196 | False | 1.00000 |
| offset8 | leaf_count | gru_offset | gru_relative | 0.01688 | -0.00550 | 0.03907 | -0.01557 | False | 1.00000 |
| offset8 | projected_area_ratio | gru_offset | gru_relative | -0.00018 | -0.00099 | 0.00062 | 0.00921 | False | 1.00000 |
| offset8 | leaf_count | time_offset | time_relative | -0.06497 | -0.16108 | 0.00692 | 0.04052 | True | 1.00000 |
| offset8 | projected_area_ratio | time_offset | time_relative | 0.00027 | -0.00148 | 0.00233 | -0.00743 | False | 1.00000 |

## 5. 实验 C：3/5/7 天可行性与统一条件的重训

训练前锁定时间 `2026-10-11T12:38:02.407580+00:00`；协议 SHA-256 `50889f886870a180787686eee4bf5eaa3b20b638768935a63beb3683821e0c27`。目标容差固定 ±6 小时，同株有效未来时间戳最近匹配，等距选较早时间；每个历史窗口每个任务至多一个目标，匹配不用标签值或模型误差。只按 Train≥100、Val≥25 植物且二者每株窗口中位数≥5 判定可训练。三个跨度均通过，但门槛不代表保证检验功效。
**日尺度模型只输入事先请求的 3/5/7 天，不输入目标实际秒差、匹配残差、未来文件/目标编号或未来 RGB/mask。** 时间第6槽为固定日跨度任务标识，不是目标实际观测偏移。B0 不需要目标时间；Oracle B1 外推到请求天数（不偷用匹配的实际间隔）；B1b/全部神经模型使用同套请求时间条件。

| days | split | plants | windows | genotypes | windows_per_plant_min | windows_per_plant_median | windows_per_plant_max | signed_match_error_hours_median | absolute_match_error_hours_p95 | absolute_match_error_hours_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | train | 219 | 9574 | 21 | 4 | 46.00000 | 138 | 0.00000 | 0.00778 | 6.00000 |
| 3 | val | 53 | 2245 | 21 | 8 | 46.00000 | 94 | 0.00000 | 0.00556 | 6.00000 |
| 3 | test | 46 | 1775 | 21 | 7 | 31.50000 | 82 | 0.00000 | 0.00306 | 6.00000 |
| 5 | train | 218 | 8235 | 21 | 4 | 37.50000 | 130 | 0.00000 | 0.02619 | 6.00000 |
| 5 | val | 53 | 1929 | 21 | 4 | 39.00000 | 86 | 0.00000 | 0.01389 | 6.00000 |
| 5 | test | 46 | 1509 | 21 | 3 | 27.00000 | 74 | 0.00000 | 0.00417 | 6.00000 |
| 7 | train | 213 | 6867 | 21 | 1 | 32.00000 | 122 | 0.00000 | 0.03269 | 6.00000 |
| 7 | val | 50 | 1606 | 19 | 2 | 31.50000 | 78 | 0.00000 | 0.02667 | 6.00000 |
| 7 | test | 44 | 1241 | 20 | 2 | 24.00000 | 66 | 0.00000 | 0.00806 | 6.00000 |

阶段覆盖（历史末帧阶段；基因型与历史/目标阶段的完整交叉表另存）：

| forecast_horizon | split | history_growth_stage | plants | windows |
| --- | --- | --- | --- | --- |
| 3 | test | early | 24 | 398 |
| 3 | test | intermediate | 46 | 827 |
| 3 | test | mature | 37 | 550 |
| 3 | train | early | 133 | 2687 |
| 3 | train | intermediate | 219 | 4264 |
| 3 | train | mature | 194 | 2623 |
| 3 | val | early | 31 | 617 |
| 3 | val | intermediate | 52 | 985 |
| 3 | val | mature | 49 | 643 |
| 5 | test | early | 24 | 398 |
| 5 | test | intermediate | 46 | 771 |
| 5 | test | mature | 30 | 340 |
| 5 | train | early | 133 | 2687 |
| 5 | train | intermediate | 217 | 3943 |
| 5 | train | mature | 154 | 1605 |
| 5 | val | early | 31 | 617 |
| 5 | val | intermediate | 52 | 903 |
| 5 | val | mature | 34 | 409 |
| 7 | test | early | 24 | 398 |
| 7 | test | intermediate | 43 | 646 |
| 7 | test | mature | 21 | 197 |
| 7 | train | early | 133 | 2684 |
| 7 | train | intermediate | 212 | 3263 |
| 7 | train | mature | 92 | 920 |
| 7 | val | early | 31 | 617 |
| 7 | val | intermediate | 48 | 738 |
| 7 | val | mature | 20 | 251 |

因固定时间覆盖不足而零窗口的植物如下；不是按预测困难移除，原 M1 46 株 Test 和全部旧结果未变。day7 的两株 Test 无可匹配窗口，所以全部 day7 方法统一在44株评价，不能与46株的任务直接作独立配对比较。

| forecast_horizon | split | plant_uid |
| --- | --- | --- |
| 5 | train | Go-0/rep_07 |
| 7 | test | Bch-4/rep_01 |
| 7 | test | Cvi-0/rep_42 |
| 7 | val | Ei-6/rep_06 |
| 7 | train | Ei-6/rep_07 |
| 7 | train | Go-0/rep_07 |
| 7 | val | Go-0/rep_10 |
| 7 | train | Hs-0/rep_05 |
| 7 | train | Hs-0/rep_09 |
| 7 | val | Kz-9/rep_42 |
| 7 | train | Kz-9/rep_44 |
| 7 | train | Lz-0/rep_09 |

完整每株窗口数（含零）见 calendar_windows_per_plant.csv；基因型覆盖见 calendar_genotype_coverage.csv；时间偏差完整统计见 calendar_feasibility.csv。匹配偏差是目标测量时刻不确定性，不把约定日尺度说成精确时刻无噪声标签。

| task | model | target | plants | MAE ± seed SD [plant CI95] |
| --- | --- | --- | --- | --- |
| day3 | B0 | leaf_count | 46 | 1.90368 ± 0.00000 [1.75600, 2.05100] |
| day3 | B0 | projected_area_ratio | 46 | 0.05865 ± 0.00000 [0.05293, 0.06452] |
| day3 | B1 | leaf_count | 46 | 3.17238 ± 0.00000 [2.93289, 3.41226] |
| day3 | B1 | projected_area_ratio | 46 | 0.02270 ± 0.00000 [0.02072, 0.02473] |
| day3 | B1b | leaf_count | 46 | 1.51359 ± 0.00000 [1.30599, 1.72579] |
| day3 | B1b | projected_area_ratio | 46 | 0.03742 ± 0.00000 [0.03393, 0.04097] |
| day3 | diff_full | leaf_count | 46 | 1.05130 ± 0.02132 [0.92320, 1.20669] |
| day3 | diff_full | projected_area_ratio | 46 | 0.01886 ± 0.00048 [0.01586, 0.02207] |
| day3 | gru_full | leaf_count | 46 | 1.06948 ± 0.03580 [0.94196, 1.23573] |
| day3 | gru_full | projected_area_ratio | 46 | 0.01795 ± 0.00019 [0.01474, 0.02153] |
| day3 | gru_repeat | leaf_count | 46 | 1.05353 ± 0.01428 [0.93142, 1.20679] |
| day3 | gru_repeat | projected_area_ratio | 46 | 0.01741 ± 0.00032 [0.01460, 0.02046] |
| day3 | gru_rgb | leaf_count | 46 | 1.05602 ± 0.02404 [0.92692, 1.21769] |
| day3 | gru_rgb | projected_area_ratio | 46 | 0.01815 ± 0.00020 [0.01515, 0.02152] |
| day3 | mlp_full | leaf_count | 46 | 1.06234 ± 0.01786 [0.94835, 1.19950] |
| day3 | mlp_full | projected_area_ratio | 46 | 0.01855 ± 0.00046 [0.01541, 0.02201] |
| day3 | mlp_rgb | leaf_count | 46 | 1.05469 ± 0.01969 [0.94514, 1.18764] |
| day3 | mlp_rgb | projected_area_ratio | 46 | 0.01835 ± 0.00052 [0.01542, 0.02157] |
| day3 | time_full | leaf_count | 46 | 1.52449 ± 0.00368 [1.31288, 1.74734] |
| day3 | time_full | projected_area_ratio | 46 | 0.03534 ± 0.00005 [0.03157, 0.03911] |
| day5 | B0 | leaf_count | 46 | 3.03257 ± 0.00000 [2.76675, 3.29554] |
| day5 | B0 | projected_area_ratio | 46 | 0.09586 ± 0.00000 [0.08611, 0.10592] |
| day5 | B1 | leaf_count | 46 | 5.04055 ± 0.00000 [4.66717, 5.41167] |
| day5 | B1 | projected_area_ratio | 46 | 0.04698 ± 0.00000 [0.04225, 0.05202] |
| day5 | B1b | leaf_count | 46 | 1.69353 ± 0.00000 [1.46168, 1.93346] |
| day5 | B1b | projected_area_ratio | 46 | 0.04159 ± 0.00000 [0.03767, 0.04563] |
| day5 | diff_full | leaf_count | 46 | 1.26676 ± 0.02261 [1.09845, 1.45932] |
| day5 | diff_full | projected_area_ratio | 46 | 0.02585 ± 0.00068 [0.02063, 0.03167] |
| day5 | gru_full | leaf_count | 46 | 1.22588 ± 0.01210 [1.07043, 1.39879] |
| day5 | gru_full | projected_area_ratio | 46 | 0.02543 ± 0.00047 [0.02032, 0.03094] |
| day5 | gru_repeat | leaf_count | 46 | 1.27839 ± 0.02432 [1.10964, 1.47208] |
| day5 | gru_repeat | projected_area_ratio | 46 | 0.02617 ± 0.00057 [0.02092, 0.03188] |
| day5 | gru_rgb | leaf_count | 46 | 1.25815 ± 0.03962 [1.09421, 1.44466] |
| day5 | gru_rgb | projected_area_ratio | 46 | 0.02533 ± 0.00055 [0.02005, 0.03121] |
| day5 | mlp_full | leaf_count | 46 | 1.23983 ± 0.01624 [1.08967, 1.40914] |
| day5 | mlp_full | projected_area_ratio | 46 | 0.02581 ± 0.00042 [0.02072, 0.03143] |
| day5 | mlp_rgb | leaf_count | 46 | 1.26353 ± 0.01720 [1.11219, 1.43617] |
| day5 | mlp_rgb | projected_area_ratio | 46 | 0.02636 ± 0.00049 [0.02112, 0.03219] |
| day5 | time_full | leaf_count | 46 | 1.68317 ± 0.00227 [1.44314, 1.93350] |
| day5 | time_full | projected_area_ratio | 46 | 0.04026 ± 0.00006 [0.03590, 0.04466] |
| day7 | B0 | leaf_count | 44 | 4.27188 ± 0.00000 [3.86499, 4.66520] |
| day7 | B0 | projected_area_ratio | 44 | 0.13313 ± 0.00000 [0.11846, 0.14832] |
| day7 | B1 | leaf_count | 44 | 6.53725 ± 0.00000 [5.97934, 7.09022] |
| day7 | B1 | projected_area_ratio | 44 | 0.07933 ± 0.00000 [0.06981, 0.08973] |
| day7 | B1b | leaf_count | 44 | 1.87235 ± 0.00000 [1.58597, 2.16991] |
| day7 | B1b | projected_area_ratio | 44 | 0.04692 ± 0.00000 [0.04172, 0.05231] |
| day7 | diff_full | leaf_count | 44 | 1.35947 ± 0.06043 [1.15779, 1.58283] |
| day7 | diff_full | projected_area_ratio | 44 | 0.03405 ± 0.00074 [0.02778, 0.04106] |
| day7 | gru_full | leaf_count | 44 | 1.39974 ± 0.07121 [1.19716, 1.62716] |
| day7 | gru_full | projected_area_ratio | 44 | 0.03351 ± 0.00203 [0.02665, 0.04146] |
| day7 | gru_repeat | leaf_count | 44 | 1.38988 ± 0.04217 [1.19213, 1.60549] |
| day7 | gru_repeat | projected_area_ratio | 44 | 0.03287 ± 0.00173 [0.02587, 0.04091] |
| day7 | gru_rgb | leaf_count | 44 | 1.39788 ± 0.00745 [1.18727, 1.63981] |
| day7 | gru_rgb | projected_area_ratio | 44 | 0.03348 ± 0.00223 [0.02657, 0.04142] |
| day7 | mlp_full | leaf_count | 44 | 1.34889 ± 0.01700 [1.14700, 1.56699] |
| day7 | mlp_full | projected_area_ratio | 44 | 0.03467 ± 0.00238 [0.02803, 0.04207] |
| day7 | mlp_rgb | leaf_count | 44 | 1.34847 ± 0.05467 [1.14854, 1.56773] |
| day7 | mlp_rgb | projected_area_ratio | 44 | 0.03342 ± 0.00103 [0.02675, 0.04095] |
| day7 | time_full | leaf_count | 44 | 1.90712 ± 0.03485 [1.61483, 2.21489] |
| day7 | time_full | projected_area_ratio | 44 | 0.04655 ± 0.00003 [0.04104, 0.05215] |

## 6. 实验 D：标签变化、阶段、尺寸与预测困难

所有困难样本保留。误差先按 seed 取窗口绝对误差均值，再在每个子组中按株取均值；置信区间仍重采样植物而非窗口。子组内植物可同时出现在多个阶段，因此阶段间不是独立组。n<5 的尺寸等子组保留点估计，但不报告不可靠 CI。
- 快速变化：未来可见叶数绝对变化/实际天数超过该任务 Train 的 Q75；阈值在 diagnostic_train_thresholds.json。
- 可见叶数下降：目标叶数低于最后历史叶数；这不是对遮挡、叶片死亡或标签错误的机制判定。
- 早/中/晚：M0 叶数阶段 early≤6、intermediate 7–10、mature≥11；另用 Train 相对记录年龄 tertile 作敏感性分组，不冒充真实播种年龄。
- 裁剪/尺寸：原图宽高变化是可测代理。全数据5株尺寸变化；不能从尺寸元数据确定实际裁剪、同尺寸取景变化或唯一误差成因。
- **几何变化关联不可识别：5株尺寸变化植物全部属于 Train，Val/Test 均为0株。** 因而当前独立评价不能检验尺寸变化与误差的关联，不制造虚假的 Test 分组差异；完整 plant/尺寸/split 清单见 geometry_inventory.csv。Train 拟合误差与未见植物误差不等价。
- 预测间隔：实际小时数只作为分析变量；日尺度模型不见实际匹配时差。
- 历史面积变化率：三帧 mask-derived 面积比例与历史实际时间的 OLS 斜率（ratio/day），只用于诊断，绝不输入 RGB 模型；尺寸改变会影响面积比例解释。

预先指定 day5 的真实 GRU 叶数误差分层：

| factor | level | plants | windows | plant_macro_mae | ci95_low | ci95_high | warning |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rapid_leaf_change | False | 46 | 1239 | 1.11536 | 0.97977 | 1.26119 | descriptive subgroup; not for selection |
| rapid_leaf_change | True | 35 | 270 | 1.49463 | 1.19424 | 1.83724 | descriptive subgroup; not for selection |
| visible_leaf_decline | False | 46 | 1508 | 1.22558 | 1.07020 | 1.39871 | descriptive subgroup; not for selection |
| visible_leaf_decline | True | 1 | 1 | 1.89021 | NA | NA | too few plants for CI |
| history_stage | early | 24 | 398 | 0.82508 | 0.68018 | 0.99585 | descriptive subgroup; not for selection |
| history_stage | intermediate | 46 | 771 | 1.20013 | 1.06385 | 1.35369 | descriptive subgroup; not for selection |
| history_stage | mature | 30 | 340 | 1.52219 | 1.27672 | 1.78298 | descriptive subgroup; not for selection |
| observed_age_stage | high | 23 | 461 | 1.05234 | 0.85162 | 1.27619 | descriptive subgroup; not for selection |
| observed_age_stage | low | 46 | 536 | 1.06562 | 0.91249 | 1.23047 | descriptive subgroup; not for selection |
| observed_age_stage | middle | 36 | 512 | 1.33876 | 1.10285 | 1.61491 | descriptive subgroup; not for selection |
| plant_size_varies | False | 46 | 1509 | 1.22588 | 1.07043 | 1.39879 | descriptive subgroup; not for selection |
| window_size_changes | False | 46 | 1509 | 1.22588 | 1.07043 | 1.39879 | descriptive subgroup; not for selection |

day5 真实 GRU 的叶数阶段误差：early 0.8251（n=24）、intermediate 1.2001（n=46）、mature 1.5222（n=30）。这是观察到的任务难度差异；阶段由叶数定义，不能当成独立生理阶段机制。
day5 每株历史面积变化率与叶数误差的 Spearman rho=0.4964，植物 CI95 [0.2639, 0.6680]。这提示高速面积变化植物可能更难预测，但没有排除基因型、阶段、采集队列与面积比例定义的共同影响，也不证明遮挡或衰老。

真实 GRU 的独立植物均值 Spearman 关联（CI 为植物 bootstrap；描述性，不是机制/因果检验，不据此选任务或删样本）：

| task | variable | plants | rho | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| day3 | absolute_leaf_change_per_day | 46 | 0.24662 | -0.07130 | 0.54852 |
| day3 | signed_leaf_change | 46 | 0.22011 | -0.09718 | 0.52156 |
| day3 | actual_hours | 46 | 0.18980 | -0.08898 | 0.43093 |
| day3 | elapsed_days | 46 | -0.21825 | -0.49246 | 0.07549 |
| day3 | history_area_rate | 46 | 0.51933 | 0.28361 | 0.69048 |
| day5 | absolute_leaf_change_per_day | 46 | 0.09380 | -0.25232 | 0.42109 |
| day5 | signed_leaf_change | 46 | 0.09091 | -0.25550 | 0.41759 |
| day5 | actual_hours | 46 | 0.06334 | -0.21867 | 0.34136 |
| day5 | elapsed_days | 46 | -0.24391 | -0.53248 | 0.07399 |
| day5 | history_area_rate | 46 | 0.49639 | 0.26391 | 0.66801 |
| day7 | absolute_leaf_change_per_day | 44 | 0.34870 | -0.00209 | 0.64485 |
| day7 | signed_leaf_change | 44 | 0.36040 | 0.00678 | 0.65426 |
| day7 | actual_hours | 44 | 0.18186 | -0.09855 | 0.44943 |
| day7 | elapsed_days | 44 | -0.15131 | -0.45598 | 0.15891 |
| day7 | history_area_rate | 44 | 0.52981 | 0.31320 | 0.68974 |
| offset2 | absolute_leaf_change_per_day | 46 | 0.03719 | -0.27256 | 0.33939 |
| offset2 | signed_leaf_change | 46 | 0.23276 | -0.03673 | 0.48061 |
| offset2 | actual_hours | 46 | 0.34129 | 0.06303 | 0.57760 |
| offset2 | elapsed_days | 46 | -0.29386 | -0.53870 | -0.01305 |
| offset2 | history_area_rate | 46 | 0.17607 | -0.13443 | 0.46157 |
| offset4 | absolute_leaf_change_per_day | 46 | 0.19198 | -0.13215 | 0.50902 |
| offset4 | signed_leaf_change | 46 | 0.34965 | 0.06458 | 0.60084 |
| offset4 | actual_hours | 46 | 0.48045 | 0.22859 | 0.67496 |
| offset4 | elapsed_days | 46 | -0.32273 | -0.56977 | -0.03644 |
| offset4 | history_area_rate | 46 | 0.38378 | 0.10369 | 0.61646 |
| offset8 | absolute_leaf_change_per_day | 46 | 0.31508 | -0.00884 | 0.60627 |
| offset8 | signed_leaf_change | 46 | 0.48000 | 0.19930 | 0.70109 |
| offset8 | actual_hours | 46 | 0.39429 | 0.07020 | 0.67918 |
| offset8 | elapsed_days | 46 | -0.26340 | -0.52711 | 0.02410 |
| offset8 | history_area_rate | 46 | 0.41301 | 0.13414 | 0.63860 |

完整所有模型/阶段/下降/时间/面积速度/尺寸分层见 difficulty_stratified_errors.csv，子组同株 GRU-对照差值见 difficulty_paired.csv。原始连续有效相邻帧叶数转移见 outputs/m1_5/adjacent_label_transitions.csv，分阶段汇总见 label_transition_summary.csv。标签非单调是可见叶片计数现象；本实验没有人工复标、遮挡分解或器官身份跟踪，不能归因到唯一机制。

Test day5 仅1个叶数下降窗口（1株），day7没有下降窗口；这一特定关联不能在它们上稳定估计。报告相应缺失 CI，不强行外推。另存 difficulty_validation_errors.csv 提供已用于选 epoch 的 Val 敏感性描述，不能当独立确认。

数据是否存在动态信息：以下为窗口级**描述性覆盖**（不是独立统计样本量）。未来标签确实随时间变化，并非全部静止；但三帧历史中可见叶数不变的比例及很短历史跨度，可能限制从短期变化估计较远未来。真正历史增量仍以 A2/A3 和配对结果为依据，不能只凭非零标签变化就断言 Temporal-JEPA 可获益。

| task | plants | windows | history_leaf_constant_fraction | future_leaf_unchanged_fraction | future_leaf_change_median | history_absolute_area_change_median | future_absolute_area_change_median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| day3 | 46 | 1775 | 0.47549 | 0.10986 | 2.00000 | 0.00510 | 0.03930 |
| day5 | 46 | 1509 | 0.47647 | 0.02518 | 3.00000 | 0.00386 | 0.06582 |
| day7 | 44 | 1241 | 0.49879 | 0.00242 | 4.00000 | 0.00262 | 0.09352 |
| offset2 | 46 | 2075 | 0.46795 | 0.50892 | 0.00000 | 0.00640 | 0.00752 |
| offset4 | 46 | 1983 | 0.47504 | 0.32930 | 1.00000 | 0.00584 | 0.01580 |
| offset8 | 46 | 1799 | 0.47804 | 0.16342 | 1.00000 | 0.00507 | 0.02948 |

## 7. 统计、训练和工程验证

主指标 Plant-level Macro MAE：每株先平均所有该任务窗口误差，再对植物等权平均。另存 Window MAE/RMSE 和 Plant Macro RMSE，不能用窗口数量放大独立生物样本数。双侧符号翻转检验依赖零差假设下植物差值符号的交换性/对称性，不是随机化处理的因果检验。
三个种子报告均值±样本 SD（不是 SEM）；CI 先平均每个 seed 的植物误差，再对独立植物 10,000 次 bootstrap，不能把3种子当额外生物重复，也不是预测 ensemble。配对差在相同植物、相同窗口上计算；双侧植物符号翻转 20,000 次，Holm 分家族校正。区间按各植物在当前划分中的变异估计，不包含新划分/批次/编码器不确定性。
所有目标/时间标准化只从 Train 拟合；逆每株窗口数加权的标准化等权 Huber(delta=1)；AdamW、weight_decay=1e-4、dropout=0.1、clip=5、cosine 到起始 LR 的0.1。MLP/difference LR=0.001 batch512；GRU LR=0.0003 batch256；只沿用原 M1 Val 选定配置，没有基于本轮 Test 做结构/超参搜索。B1b 的 alpha={0.1,1,10,100} 仍仅由各日尺度 Val 选择。
A2/A3 所有100轮和实际 optimizer_steps相同，各自只在 Val 选择最优 epoch；可复现初始化和 batch-order SHA 见 capacity_training_audit.csv。GPU0/1 两张 A6000 按互不重叠的任务并行，没有多卡同步训练；设备、时间、峰值显存逐运行保存，时间和不是总墙钟时长。

| architecture | runs | parameters | best_epoch_median | training_seconds | peak_memory_bytes |
| --- | --- | --- | --- | --- | --- |
| diff | 18 | 123714 | 13.00000 | 126.63053 | 164375040 |
| gru | 72 | 561922 | 8.00000 | 1780.57162 | 233935872 |
| mlp | 90 | 101634 | 31.00000 | 631.20381 | 154744832 |

自动化测试：67 passed、0 failed、0 skipped。包含 M0/M1 回归测试、唯一匹配/等距/容差、隔离帧不压缩、标签不影响匹配、未知未来时间不可进入 offset-only/day 条件、Time-only/单帧输入边界、从头重复帧训练、相同初始化/顺序/步数、真实产物完整性和统计配对规则。证据：tests.xml / tests_initial.xml。
初始开发遇到 CSV 标签浮点回读和可选 tabulate 不存在：修复回读类型并使用无新增依赖的 Markdown 表格；没有改旧结果或据 Test 调参。完整运行依赖复用 requirements-m1-lock.txt，无新依赖安装或环境覆盖。

推理精度复核发现 PyTorch 新进程的默认 cuDNN TF32 与训练时关闭 TF32 不同。统一调用训练的 deterministic/TF32=False 设置后重新评估，**没有改训练函数、配置、checkpoint或模型选择**。首轮所有预测/统计原样归档于 outputs/m1_5/predictions_before_precision_fix 和 analysis_before_precision_fix，修复前源码快照与 SHA 在 inference_precision_correction.json；这是一致性修复，不是根据 Test 优化方法。

推理稳定性另检查代表性 offset4 重复帧 GRU 与 day5 真实 GRU，同GPU两次逐值一致；两张GPU的差异明细见 inference_repeatability.json。冻结训练源码快照保留原SHA，重放仅允许有记录且 AST 证明非 predict 函数完全不变的精度修复，拒绝悄悄改变训练实现。

## 8. 交付与复现

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/run_m1_5.py --stage prepare
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/run_m1_5.py --stage train --device cuda:1
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python scripts/run_m1_5.py --stage evaluate --device cuda:1
.venv-m1/bin/python scripts/run_m1_5.py --stage analyze
CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv-m1/bin/python -m pytest tests -q --junitxml=reports/m1_5/tests.xml
.venv-m1/bin/python scripts/run_m1_5.py --stage report
```

同一目录可重放：已完成且配置一致的训练复用 checkpoint，禁止覆盖；协议、可行性和输入哈希不一致则报错，不会静默重选 Test。要从零再训练，应在保留全部现有产物的单独工作副本中复现，不能删除旧困难样本或 checkpoint。
- 180 个神经运行：`outputs/m1_5/checkpoints/<task>__<model>__s<seed>/checkpoint.pt`，附 config.json、training_history.csv、resources.json。
- 9 个日尺度简单基线；B1b 状态/候选验证分数在 outputs/m1_5/simple。
- 每个运行的 Val/Test 预测 CSV、指标 JSON 与植物级误差 CSV：outputs/m1_5/predictions/<task>__<model>__s<seed>/。
- 汇总：baseline_comparison.csv、per_seed_metrics.csv、plant_level_errors.csv、paired_comparisons.csv、metrics.json、decision.json。
- 协议/追溯：configs/m1_5/protocol.json、protocol_lock.json、training_source_fingerprints.json、legacy_snapshot.json、initial_protection.json、final_protection.json。
- 可行性：CALENDAR_FEASIBILITY.md 与 calendar_*.csv；诊断标签仅在 outputs/m1_5/window_diagnostics.csv，不是模型输入。
- 图表：figures/capacity_control、time_conditions、calendar_feasibility、difficulty_day5，各 PNG/SVG/PDF；来源与 QA 见 FIGURE_CONTRACT.md、figure_source_qa.json、figure_visual_qa.md。图片内容由 Agnes API 识别；最终4图均无阻塞视觉问题，原始建议完整保存。时间图的模型x偏移仅为展示分离，同一刻度仍是同一固定任务，不是改变预测时间；数值请以CSV为准。若没有设置 AGNES_API_KEY，视觉 QA 必须标记 unavailable，不得伪造 PASS。

图表额外数值 QA 校验每个面板数据点/误差棒端点位于坐标界内，并核验时间图各方法的像素x间隔大于标记宽度；这与图像描述互补，避免把视觉服务对末个刻度或图例的猜测当作真实数值截断。非阻塞建议（相同max值重合、各困难子组使用不同可读轴范围等）不隐藏，均可回溯。

## 9. 下一步边界

现有证据不足以满足预先固定的继续规则，也未在所有任务上排除有实际意义的历史增益。时间、结构与真实动态的贡献需要分别解释；不把监督 GRU 的弱优势当成对 Temporal-JEPA 的否定。建议先做预固定独立确认和任务核验，不进入 M2 训练。

无论该监督模型结论如何，都不能自动否定 Temporal-JEPA：冻结 DINOv2 可能没有保留任务所需动态，GRU/轻量 MLP 的归纳偏置可能有限，三个短历史帧与可见叶数/投影面积可能并非理想动态目标。现有结果只约束当前数据、缓存与监督训练合同。
若后续研究时序表征，先固定额外独立数据、请求时间合同和主要终点，再比较单帧、重复帧、显式变化及新方法；若转向事件/器官级动态，应先明确可见性与器官身份标签并审计，不在本次启动 M2。

## 来源

实验事实：本报告列出的本地只读 M0/M1 和新 M1.5 产物；来源/检索/精确 PyTorch 2.8 文档见 SOURCES.md。博查接口 HTTP403，未把搜索失败伪装成成功。绘图是实际统计数据渲染，不是文生图生成。
