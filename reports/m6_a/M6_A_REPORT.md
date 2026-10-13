# M6-A：非 Oracle 植物器官动态表型流水线可行性验证

## 0. 唯一研究结论

**INSTANCE-FIRST**。现有固定基线未通过冻结的可靠性验收：`baseline_acceptance_passed=False`。
这不是算法创新阶段，也不是JEPA复活实验；本次无任何训练步、无未来点云预测、无自监督预算增加。

完成的是 **Sugar4D V4 24株384帧** 的整株验证/测试比较；未成功下载Pepper-4D，**未完成跨物种验证**。
所有平均值以独立植株为单位，12个test植株并非192个生物学重复。val也为12株；官方24个train植株未重新训练。

## 1. 预先冻结的数据、主指标与门槛

- 冻结文件：`configs/m6_a/protocol.json`；SHA256 `4df365b232acd61334c414fe387fd17d9ead42c2026765238cf571f276865bd6`；锁定时间见 `protocol_lock.json`，早于模型推理和目标评分。
- 沿用V4和官方24/12/12整株划分。每株16次实际扫描，DAS为37、40、44、47、51、54、58、61、65、72、75、79、82、86、89、93。没有按帧随机划分、删除困难植株或压缩时间缺口。
- 主要指标固定为 **plant-macro observation IDF1**；实例PQ严格IoU>0.5、SBD、事件F1、动态误差和失败率是预设必要诊断/验收门槛，不替换主指标。
- 冻结验收：B1 PQ≥0.70、IDF1≥0.80、首观测事件F1≥0.70、测量覆盖≥0.90、长度/面积静态惩罚相对误差≤0.20/0.25、长度和面积严重动态错误率≤0.10、至少80%测试植株同时可靠。
- 同一器官身份GT在评估器中用于时序对齐，**任何追踪器都不接收它**。B0/B2_GT只接收当前掩码且每帧独立随机重编号。模型B1/B2只接收XYZ；法线、GT语义、器官掩码、叶数、CSV表型均不作为模型输入。
- Test植株未进入这两份官方源物种模型训练，但M3/M4已使用过这些test数据；这里只能称未见植株工程验证，不能称全新独立确认性队列。

## 2. 资源核实及实际运行

详见 `SOURCE_AUDIT.md`、`source_audit.csv`、`downloaded_files.csv`、`sources/`。
Sugar4D仍为CC BY4.0；三个GitHub仓库没有可核实的显式代码许可，不能把“公开可下载”当成再分发许可。
PlantNet与PSegNet epoch199权重真实下载且严格加载，4096点前向与重复性测试通过；PSegNet只做smoke不择优。
TrackPlant3D的3DEPS、官方CPD和官方扩展匹配代价实际运行；没有训练追踪网络。
隔离环境为`.venv-m6-a`；历史环境包不变。全部兼容性调整、被拒绝的边界向量化试验和邻域等价验证在 `COMPATIBILITY.md`。

### 官方归一化保真度限制及单独诊断

原始B1冻结的是均值中心/单位球归一化。原始H5转换源码下载起初超时；之后成功的GitHub blob API证据显示官方模型预处理用**采样点bounding-box midpoint / 最大正坐标**，并非单位球。
保留所有原始基线；在新哈希锁后，对相同24株384帧、相同checkpoint和4096个XYZ采样点做**独立工程分割诊断**：

| split | pq | sbd | tp | fp | fn |
| --- | --- | --- | --- | --- | --- |
| test | 0.4903 | 0.6277 | 5.3229 | 3.9688 | 3.7969 |
| val | 0.5144 | 0.6274 | 5.2812 | 3.6146 | 3.0938 |

这些PQ仍低于冻结0.70阈值，支持分割瓶颈方向并非仅由原始归一化差异造成，但不把它当新的主验收结果。诊断没有执行完整官方归一化追踪/表型，不能宣称对官方原生流水线已全面验证。
见 `NORMALIZATION_DIAGNOSTIC.md`、`official_normalization_lock.json`、`official_normalization_execution_scope.json`，主协议/主IDF1没有修改。

## 3. 三条公平流程与明确诊断

- **B0**：真实当前叶实例掩码 + TrackPlant3D适配器；不是把真实时序ID交给追踪器。
- **B1**：官方固定PlantNet零样本当前实例分割 + 与B0完全相同的TrackPlant3D。
- **B2**：与B1完全相同的预测实例 + 当前空间质心距离门控Hungarian，无CPD、无学习。
- **B2_GT**：GT当前掩码 + B2匹配，独立列出的诊断条件；不替换非Oracle B2。
- B1官方MeanShift带宽0.6、公开平均实例过滤、4096个几何均匀点；最近邻回传标签到全部原始PLY点，PQ/SBD按完整分辨率评价，非2048/4096点替代指标。
- TrackPlant3D每帧256个3DEPS点，一部分完整器官可能没有采样支持；缺失标记track=-1并计入失败，不偷偷按GT补点。

## 4. 固定主指标及逐植株结果

| split | pipeline | independent_plants | pq | sbd | idf1 | link_recall | id_switches | continuity | complete_track_fraction | new_event_f1 | reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | B0 | 12 | 1.0000 | 1.0000 | 0.9756 | 0.9895 | 1.3333 | 0.9840 | 0.9110 | 0.9003 | 0.0000 |
| test | B1 | 12 | 0.4980 | 0.6188 | 0.5319 | 0.4903 | 4.2500 | 0.3977 | 0.0897 | 0.0088 | 0.0000 |
| test | B2 | 12 | 0.4980 | 0.6188 | 0.5082 | 0.4779 | 6.6667 | 0.3722 | 0.0381 | 0.0221 | 0.0000 |
| test | B2_GT | 12 | 1.0000 | 1.0000 | 0.9315 | 0.9515 | 6.2500 | 0.9385 | 0.6168 | 0.6437 | 0.0000 |
| val | B0 | 12 | 1.0000 | 1.0000 | 0.9590 | 0.9841 | 1.7500 | 0.9707 | 0.8583 | 0.8349 | 0.0000 |
| val | B1 | 12 | 0.5002 | 0.6382 | 0.5408 | 0.5173 | 4.6667 | 0.4187 | 0.0805 | 0.0069 | 0.0000 |
| val | B2 | 12 | 0.5002 | 0.6382 | 0.5193 | 0.4952 | 8.0833 | 0.3938 | 0.0406 | 0.0054 | 0.0000 |
| val | B2_GT | 12 | 1.0000 | 1.0000 | 0.9172 | 0.9433 | 6.5000 | 0.9271 | 0.6492 | 0.6663 | 0.0000 |

IDF1采用**器官观测级**而非像素级：评价器根据严格IoU>0.5单帧关联，再在整条时序做GT身份与输出track的一对一最大观测数分配；`2*IDTP/(GT observations + predicted leaf observations)`。
ID Switch是在同GT器官连续检测身份改变时计数，漏检间隔后的ID变化仍计；ID连续性为最长相同身份连续检测段/该GT身份实际观测数。`link_recall`将漏检计为失败，不能只在成功检测器官上报高准确率。

Test B1−B0 IDF1差：-0.4436，整株配对bootstrap95%CI [-0.4763, -0.4102]，n=12独立株。
所有额外区间、B1/B2及B0/B2_GT比较保存在 `paired_plant_intervals.csv`。

### B1 每个未见测试植株

| plant_uid | pq | idf1 | id_switches | new_event_f1 | length_change_tracked_loss | length_change_tracked_coverage | length_change_tracked_severe | reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| plant_002 | 0.5360 | 0.5350 | 7 | 0.0000 | 0.9172 | 0.5867 | 0.4467 | False |
| plant_007 | 0.4371 | 0.4674 | 5 | 0.0000 | 1.0091 | 0.5328 | 0.4836 | False |
| plant_010 | 0.4893 | 0.5724 | 2 | 0.0000 | 0.9890 | 0.5586 | 0.4828 | False |
| plant_012 | 0.5450 | 0.5130 | 5 | 0.1053 | 0.9800 | 0.5636 | 0.4818 | False |
| plant_016 | 0.5601 | 0.5684 | 4 | 0.0000 | 0.8484 | 0.6406 | 0.4141 | False |
| plant_023 | 0.4334 | 0.5017 | 2 | 0.0000 | 1.0583 | 0.4966 | 0.5101 | False |
| plant_026 | 0.4663 | 0.5205 | 2 | 0.0000 | 1.0074 | 0.5327 | 0.4860 | False |
| plant_029 | 0.5759 | 0.6099 | 3 | 0.0000 | 0.8770 | 0.6111 | 0.4352 | False |
| plant_040 | 0.4832 | 0.4919 | 6 | 0.0000 | 0.9280 | 0.5926 | 0.4537 | False |
| plant_041 | 0.4149 | 0.4590 | 6 | 0.0000 | 1.0504 | 0.5000 | 0.5000 | False |
| plant_045 | 0.4958 | 0.5238 | 6 | 0.0000 | 0.9932 | 0.5405 | 0.4932 | False |
| plant_046 | 0.5385 | 0.6203 | 3 | 0.0000 | 0.9037 | 0.5962 | 0.4295 | False |

全96个植株×流程结果在 `per_plant.csv`；所有1536个帧×流程结果在 `frame_metrics.csv`。没有删掉失败植株。

## 5. 测量、时间变化与误差传播

测量器是预设简单几何基线：PCA主轴1%–99%跨度(cm)，PCA平面凸包面积(cm²)，每个完整掩码最多4096点。
这与Sugar4D的“包含叶柄的总长度”和其自动叶面积**不是严格相同估计对象**，因此误差是基线相对于已发布自动参考的差异，不能宣称精确生物学长度/面积真值。
变化为相邻实际扫描差分，允许负增长；另报每DAS天数除后的变化率误差。缺失形态为NaN，不是0。

| pipeline | length_static_tracked_mae | length_static_tracked_loss | area_static_tracked_mae | area_static_tracked_loss | length_change_tracked_mae | area_change_tracked_mae | length_change_tracked_coverage | length_change_tracked_severe | area_change_tracked_severe |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 8.2529 | 0.4486 | 9.0084 | 0.1503 | 1.1217 | 6.9212 | 0.9944 | 0.0062 | 0.0407 |
| B1 | 8.4446 | 1.1069 | 29.3764 | 1.1241 | 3.0782 | 31.3821 | 0.5627 | 0.4680 | 0.6179 |
| B2 | 8.4446 | 1.1069 | 29.3764 | 1.1241 | 2.9214 | 29.5798 | 0.5277 | 0.4946 | 0.6329 |
| B2_GT | 8.2529 | 0.4486 | 9.0084 | 0.1503 | 1.1046 | 6.9953 | 0.9551 | 0.0474 | 0.0791 |

静态/变化MAE只在有测量时定义，必须同时读覆盖率。主要误差分解用所有有参考的合法GT观测/相邻变更，`min(relative absolute error,2)`；没有测量直接损失2，不让拒绝输出带来虚假低误差。
严重静态错误为相对误差>50%或缺失；严重动态错误为变更误差/上一时刻参考>50%或缺失。

`error_decomposition.csv`使用同一GT观测集合：
1. GT当前掩码 + **评价器真相邻链接**：测量器/参考差异；不运行Oracle追踪算法。
2. 预测当前掩码 + **评价器真相邻链接**：增加分割误差。
3. 同一预测当前掩码 + 实际输出链接：增加匹配错误/身份缺失影响。

三项是相同损失的加法差分，可能出现负差值，**不是独立、正交或因果方差分解**。GT链接永远只在误差诊断评价器里出现。
尤其若正确上一个器官没有可关联分割，Oracle-link测量为缺失损失2，而错误上一个器官却能提供一个有限数值，原始“tracking增量”会变负。这是**未经证实的数值替代/错连掩盖缺失**，不是可靠追踪收益。
因此额外报告**事后误差诊断**：只限定“相同预测掩码下前后真实GT器官均有可关联测量”的评价Oracle支持集，保留追踪丢失损失2，另外统计身份失败率。此诊断由负增量现象触发，不是重新注册的确认性主指标；不替换全体主要损失，也不删减主验收失败。

| split | pipeline | trait | mean | ci95_low | ci95_high | plants | common_support_fraction | tracking_identity_failure_rate | unverified_numerical_substitution_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | B0 | area | 0.0117 | 0.0042 | 0.0209 | 12 | 1.0000 | 0.0105 | 0.0000 |
| test | B0 | length | 0.0103 | 0.0028 | 0.0198 | 12 | 1.0000 | 0.0105 | 0.0000 |
| test | B1 | area | -0.0005 | -0.0108 | 0.0083 | 12 | 0.5033 | 0.0261 | 0.0619 |
| test | B1 | length | 0.0095 | 0.0034 | 0.0160 | 12 | 0.5033 | 0.0261 | 0.0619 |
| test | B2 | area | 0.0490 | 0.0291 | 0.0710 | 12 | 0.5033 | 0.0495 | 0.0465 |
| test | B2 | length | 0.0739 | 0.0451 | 0.1045 | 12 | 0.5033 | 0.0495 | 0.0465 |
| test | B2_GT | area | 0.0848 | 0.0654 | 0.1109 | 12 | 1.0000 | 0.0485 | 0.0000 |
| test | B2_GT | length | 0.0840 | 0.0653 | 0.1086 | 12 | 1.0000 | 0.0485 | 0.0000 |

### B1 追踪相对于评价Oracle链接的动态损失增量

| metric | mean | ci95_low | ci95_high | plants |
| --- | --- | --- | --- | --- |
| length_change_loss | -0.1044 | -0.1243 | -0.0882 | 12 |
| area_change_loss | -0.0728 | -0.0943 | -0.0533 | 12 |

GT-mask长度动态损失=0.0689；GT-mask面积动态损失=0.1301。
严重失败逐条见 `severe_failures.csv`，分类见 `transition_failure_types.csv`。最终人工长度仅做DAS93横断面对照：

| split | pipeline | stage | manual_length_mae | automatic_manual_mae | coverage | leaves |
| --- | --- | --- | --- | --- | --- | --- |
| test | B0 | GT_MASK_ORACLE_LINK | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B0 | PRED_MASK_ORACLE_LINK | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B0 | TRACKED | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B1 | GT_MASK_ORACLE_LINK | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B1 | PRED_MASK_ORACLE_LINK | 8.1012 | 1.7315 | 0.6301 | 9.7500 |
| test | B1 | TRACKED | 8.1012 | 1.7315 | 0.6301 | 9.7500 |
| test | B2 | GT_MASK_ORACLE_LINK | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B2 | PRED_MASK_ORACLE_LINK | 8.1012 | 1.7315 | 0.6301 | 9.7500 |
| test | B2 | TRACKED | 8.1012 | 1.7315 | 0.6301 | 9.7500 |
| test | B2_GT | GT_MASK_ORACLE_LINK | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B2_GT | PRED_MASK_ORACLE_LINK | 8.1652 | 1.7315 | 1.0000 | 9.7500 |
| test | B2_GT | TRACKED | 8.1652 | 1.7315 | 1.0000 | 9.7500 |

人工比较的遗漏也单独报覆盖率，不将成功输出子集的MAE当成全部器官的性能；没有人工纵向面积、死亡、出现真值。

## 6. 首次观察新器官与缺失语义

初始DAS所有已有器官不计新事件。GT器官首次在后续实际扫描出现，才是“首次观测”事件；不是净叶数增加，不是植物学出生。
追踪新ID只是预测事件；真阳性需要同一时刻通过IoU关联到新GT器官。

| pipeline | new_event_precision | new_event_recall | new_event_f1 | new_missed_instance | new_sampling_or_track_loss | new_attached_old_identity | false_rebirth | spurious_birth |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 | 0.9438 | 0.8618 | 0.9003 | 0.0000 | 0.0000 | 1.5000 | 0.5000 | 0.0000 |
| B1 | 0.0083 | 0.0093 | 0.0088 | 10.5833 | 0.0000 | 0.3333 | 3.1667 | 11.9167 |
| B2 | 0.0163 | 0.0353 | 0.0221 | 10.5833 | 0.0000 | 0.0833 | 8.2500 | 15.8333 |
| B2_GT | 0.5860 | 0.7205 | 0.6437 | 0.0000 | 0.0000 | 3.0000 | 5.7500 | 0.0000 |

`new_leaf_events.csv`记录逐新器官首观测状态与首次分割检测延迟。`observation_status_grid.csv`严格区分：
- `not_observed_before_first/gap/after_last`：GT记录没有器官观察，**不转换成零长度/零面积**。
- `instance_missed`：GT器官已观察但无足够IoU预测实例。
- `identity_lost`：实例可关联但追踪未覆盖或断链。
- `wrong_identity`/`wrong_match`：观察被连到错误时序身份/错误上一个器官。
- 真实器官消失/死亡：**数据未标注，始终unknown**。停止追踪、GT最后观察都不等于生物学死亡。

## 7. 严重错误、置信度和拒绝输出

置信度分两层：模型max-softmax语义置信度；语义置信度×匹配备选边际的组合启发式。两者都不是已校准的“结果正确概率”。
评估身份错误和严重动态错误AUROC，固定有价值阈值0.70，整株macro；单株只有一类时AUROC不可定义并保留NaN，不强制当成1。

| pipeline | target | score | auroc | error_rate | high_confidence_errors | high_confidence_count |
| --- | --- | --- | --- | --- | --- | --- |
| B0 | area_severe_change | confidence | 0.6459 | 0.0407 | 0.0000 | 0.0000 |
| B0 | identity_error | confidence | 0.7162 | 0.0244 | 0.2500 | 3.8333 |
| B0 | identity_error | semantic_confidence | 0.5000 | 0.0244 | 3.5000 | 145.9167 |
| B0 | length_severe_change | confidence | 0.9093 | 0.0062 | 0.0000 | 0.0000 |
| B1 | area_severe_change | confidence | 0.6726 | 0.3518 | 0.0000 | 0.0000 |
| B1 | identity_error | confidence | 0.7889 | 0.4314 | 2.7500 | 4.8333 |
| B1 | identity_error | semantic_confidence | 0.7537 | 0.4314 | 37.0000 | 101.4167 |
| B1 | length_severe_change | confidence | 0.7684 | 0.1030 | 0.0000 | 0.0000 |
| B2 | area_severe_change | confidence | 0.7544 | 0.3773 | 0.9167 | 9.0000 |
| B2 | identity_error | confidence | 0.7426 | 0.4564 | 3.5000 | 14.0833 |
| B2 | identity_error | semantic_confidence | 0.7289 | 0.4564 | 40.2500 | 101.4167 |
| B2 | length_severe_change | confidence | 0.8495 | 0.1475 | 0.0833 | 9.0000 |
| B2_GT | area_severe_change | confidence | 0.8499 | 0.0791 | 0.7500 | 41.1667 |
| B2_GT | identity_error | confidence | 0.6427 | 0.0685 | 1.8333 | 45.0000 |
| B2_GT | identity_error | semantic_confidence | 0.5000 | 0.0685 | 9.9167 | 145.9167 |
| B2_GT | length_severe_change | confidence | 0.9574 | 0.0474 | 0.0000 | 41.1667 |

`confidence_per_plant.csv`、`prediction_confidence.csv`与`confidence_selective_risk.csv`保留高置信错误和20/50/80/100%保留风险。
没有预测的遗漏器官没有模型置信度；不能把仅成功输出上的AUROC外推为“识别所有漏检”。测量缺失率和全队列严重失败率仍作为验收门槛。

## 8. 五个研究问题：根据证据作答

1. **当前实例分割是不是主要瓶颈？** 固定零样本PlantNet的PQ=0.4980，B0−B1 IDF1差=0.4436。GT掩码对照定位的是此基线分割的传播影响，不足以证明所有监督分割器都失败；按冻结优先级得到 INSTANCE-FIRST。
2. **追踪错误是否显著影响动态表型？** 结果：没有B1动态指标同时满足固定增量和CI门槛。B1共同支持长度损失增量=0.0095，95%CI [0.0034, 0.0160]，虽有正向区间但小于固定实际意义0.05门槛。简单匹配B2_GT的额外损失约0.084且区间大于0，说明错误匹配确实可传播，但官方追踪在GT掩码条件下不是目前最大的瓶颈。漏掉身份造成的测量不可用一并计入，不能只看成功链接MAE。
3. **新器官主要发生哪类错误？** B1主要是 **当前实例未得到IoU>0.5关联（实例漏检/合并/不完整）**，每株平均10.5833个；新实例漏检、采样/ID丢失、新器官附着旧ID、已有叶误报新ID和伪实例分别报告，不仅展示成功图。
4. **未见植株仍可靠吗？** B1可靠测试植株比例=0.0000，冻结要求≥0.80；整体验收=False。零样本跨目标物种迁移的局限与长期身份误差同时保留；不声称pepper结果。
5. **最值得下一阶段创新的问题？** 唯一建议 INSTANCE-FIRST，只按冻结判定优先级，不凭面积单项变好或更换主评价指标宣称成功。本任务结束后不直接设计或训练新方法。

此建议范围是“先解决当前器官检测/实例关联完整性”，不是已证明需要发明一种全新网络：尚无目标物种监督分割训练、独立跨物种下载或官方归一化完整追踪结果。官方零样本PQ诊断仍失败，因此分割方向证据成立，但方法创新的新颖性与可达上限尚未证明。
即便B0身份接近可靠，PCA静态长度相对误差仍约45%、DAS93人工长度差异约8.17cm，说明当前简单测量器也不具备可靠绝对形态精度。它不否定实例缺失导致动态覆盖约56%的主要传播瓶颈；不能把INSTANCE-FIRST解读为测量阶段已经解决。

## 9. 可视化、自动化测试与复现

`figures/`有逐株身份时间线、真实点云首观测事件、基线指标、分割→追踪→测量增量、事件错误和置信度风险。
案例按最差/最好B1测试IDF1选择并标注，非手选成功案例；点云是真实当前观测，图像由Python数据绘制，未调用生成式模型填补器官。
`figure_manifest.json`记录图与输入及哈希，`FIGURE_QA.md`记录直接查看输出的检查。
`final_verification.json`核验384帧、96条逐株流程结果、同掩码B1/B2、无重复关联、全缓存锁定哈希及加法闭合；真实早期和后期扫描重复推理的标签、置信度及采样均完全一致。

```bash
bash scripts/setup_m6_a_environment.sh
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/audit_m6_a_sources.py --download
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py freeze
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py smoke
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-m6-a/bin/python scripts/run_m6_a.py prepare --splits val test
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv-m6-a/bin/python scripts/run_m6_a.py evaluate
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py report
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv-m6-a/bin/python -m pytest tests/test_m6_a.py -q
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/run_m6_a.py verify
PYTHONDONTWRITEBYTECODE=1 .venv-m6-a/bin/python scripts/verify_m6_a.py --repeat
```

可分别 `prepare --splits val` / `prepare --splits test` 使用可见GPU0/1并发执行，不改变配置；完整缓存重用仍核验冻结哈希。
官方归一化诊断复现：`python scripts/diagnose_m6_a_normalization.py --splits val test`，完成后同命令加`--evaluate-only`；不是新主指标训练或择优。
原始数据与上游权重不重新发布；网络不通时使用已校验本地副本。完整依赖版本在 `environment.json` / `environment-lock.txt`。
M6-A专项测试18项通过。全库测试在M6隔离环境164通过/1失败，唯一旧M1编码器测试缺少torchvision；该测试在原`.venv-m1`环境单独复跑通过。没有安装无关依赖或修改旧测试来掩盖失败。
M0—M5保护验证在 `protection_verification.json`：历史文件及891个Sugar4D原始文件无修改。无合并、无commit、无push、无删除历史工件。

## 10. 信息来源及边界

- 当前本地实测：`baseline_metrics.csv`、`per_plant.csv`、`paired_plant_intervals.csv`、`measurement_errors.csv`、`observations.csv`、`runtime_*.csv`。
- 官方来源URL与检索失败记录：`SOURCE_AUDIT.md`、`sources/bocha_search.json`、`sources/http_audit.json`。
- Sugar4D：https://www.ipb.uni-bonn.de/data/sugar4d/index.html
- PlantNet/PSegNet：https://github.com/Huang2002200/PlantNet-and-PSegNet
- TrackPlant3D：https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- Pepper-4D：https://github.com/foysalahmed10/Pepper-4D

停止于baseline研究可行性判断，不把“可运行”写成“可靠”、不把自动参考当成生物学真相，也不把失败硬改成主指标成功。
