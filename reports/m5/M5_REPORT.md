# M5：扩展植物点云数据的 JEPA 表征恢复与验证

日期：2026-10-12。最终研究决策：**JEPA-STOP**。

工程状态：**PARTIAL**。真实下载、实验矩阵、评价与检查点已按下表记录；但
Crops3D 未完整公开物理植株身份/原生米制单位，不能冒称数据审计全部 PASS。
本阶段完成后停止，不自动启动下一阶段，不合并、提交或推送 Git。

## 1. 五个必须回答的问题

1. **更大的预训练数据是否改善表征？** 单独扩大数据的 P3 相对同 500 步 S500：official 50% MAE 1.5307→1.6712；pooled rank 12.863→2.971。P4 pooled rank 为 15.012，可缓解低秩风险，但未转化为优于三个控制的稳定主任务收益。P4 同时改变正则化，不能把 P4−S500 的差值全部归因于数据扩展；是否真正成功必须满足完整预设门槛。 下面同时报告方差、有效秩、
   中心化相似度、训练植株叶实例探针和全部下游结果，不能只凭 SSL loss 回答。
2. **更多训练步骤改善还是恶化退化？** M5-A pooled rank 从20.709→13.185→8.544→1.525，
   对应0/120/500/2000步。2000步方差上升却91.89%集中在第一特征方向，说明低维退化，
   不等于零方差完全坍塌。未再盲目加时；正式全部锁定500步。
3. **是否真正提高少标注结构任务性能？** 按50%当前可见叶数、三个同架构控制、
   最小实际改善、三种子、植株配对 CI、四折和分割风险共同判定为 **JEPA-STOP**。
   不从25%/100%、阶段、物种或辅助任务中挑最佳结果替换主任务。
4. **是否值得发展成未来生长预测模型？** 本次仅测试当前几何表征，未测试未来预测；
   当前证据不支持将此通用 Point-JEPA 分支推进为未来生长预测模型。即使当前主任务
   改善也不足以证明时序能力。M3.5的动态任务门槛仍未通过，不启动Temporal-JEPA。
5. **是否终止通用Point-JEPA路线？** 应停止当前通用 Point-JEPA 研发分支：数据扩展与固定正则化未通过完整主任务门槛，且器官分割负迁移持续。 这不代表植物表型预测研究停止。

## 2. 历史复核与成果保护

见 `HISTORY_AUDIT.md`、`history_audit.json`、`historical_checkpoints.csv`。
M4仍为LIMITED-BENEFIT/工程PARTIAL，P1官方权重仍未使用。
15份P2各24株/384扫描/120优化步，架构384×12×6、predictor192×6×6、64×32patch不变。
严格重载的30份P0/P2权重hash与原诊断一致；token384、pooled768、count795输入无维度错误；
EMA/stop-gradient/共享tokenizer及缓存归一化没有发现实现错误。
保护45614个既有文件：PASS，
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
外部实际采纳951份几何对象。来源计数：{"BonnBeetClouds3D": 128, "Crops3D": 823}。
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
0；无公开已知重叠采集，
但哈希无重复不能证明所有近重复或重复物理植株均不存在。

## 4. M5-A与预先锁定协议

协议SHA256：`6933ae2a7352da892577c5261da0b6bfd789d0827605f54208fd4bfbb90fab30`。`protocol_lock.json`冻结原始输入、代码与label budgets。
`preregistration_chronology.json`核验本地锁文件早于所有正式模型初始诊断。
这是本地探索性协议锁，不是公开注册或可信第三方时间戳。
M5-A只用official Train、seed17；梯度、EMA、特征与中间完整状态真实保存。

| step | ssl_loss | token_std_mean | token_effective_rank | pooled_std_mean | pooled_effective_rank | pooled_variance_mean | pooled_largest_eigenvalue_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | NA | 0.3128 | 3.5358 | 0.0859 | 20.7086 | 0.0108 | 0.1915 |
| 120 | 0.0112 | 0.0396 | 7.3989 | 0.0165 | 13.1852 | 0.0006 | 0.5042 |
| 500 | 0.0020 | 0.0535 | 7.2090 | 0.0288 | 8.5437 | 0.0013 | 0.5293 |
| 2000 | 0.0015 | 0.2806 | 1.5608 | 0.2393 | 1.5254 | 0.0858 | 0.9189 |

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

| partition | condition | mean | std |
| --- | --- | --- | --- |
| fold0 | P0 | 2.1732 | 0.0815 |
| fold0 | P2 | 1.7198 | 0.1481 |
| fold0 | P3 | 1.5922 | 0.0345 |
| fold0 | P4 | 1.7062 | 0.0907 |
| fold0 | S500 | 1.6245 | 0.0995 |
| fold1 | P0 | 1.8319 | 0.0409 |
| fold1 | P2 | 1.4038 | 0.0846 |
| fold1 | P3 | 1.3689 | 0.0888 |
| fold1 | P4 | 1.4903 | 0.0825 |
| fold1 | S500 | 1.3854 | 0.1077 |
| fold2 | P0 | 1.8890 | 0.0288 |
| fold2 | P2 | 1.5026 | 0.0073 |
| fold2 | P3 | 1.5722 | 0.0968 |
| fold2 | P4 | 1.5347 | 0.0457 |
| fold2 | S500 | 1.5097 | 0.0598 |
| fold3 | P0 | 2.0639 | 0.0806 |
| fold3 | P2 | 1.7215 | 0.0841 |
| fold3 | P3 | 1.7283 | 0.2060 |
| fold3 | P4 | 1.7406 | 0.0367 |
| fold3 | S500 | 1.8085 | 0.0494 |
| official | P0 | 1.7483 | 0.1090 |
| official | P2 | 1.5237 | 0.0765 |
| official | P3 | 1.6712 | 0.1456 |
| official | P4 | 1.5424 | 0.0878 |
| official | S500 | 1.5307 | 0.1065 |

官方植株配对结果（delta=left−right，MAE负值有利）：

| left | right | left_metric | right_metric | absolute_improvement | relative_improvement | improving_seeds | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P3 | P0 | 1.6712 | 1.7483 | 0.0771 | 0.0441 | 3 | -0.4147 | 0.2431 |
| P3 | P2 | 1.6712 | 1.5237 | -0.1475 | -0.0968 | 0 | 0.0623 | 0.2442 |
| P3 | S500 | 1.6712 | 1.5307 | -0.1405 | -0.0918 | 0 | -0.0544 | 0.3326 |
| P4 | P0 | 1.5424 | 1.7483 | 0.2060 | 0.1178 | 3 | -0.5099 | 0.0930 |
| P4 | P2 | 1.5424 | 1.5237 | -0.0187 | -0.0122 | 1 | -0.0413 | 0.0816 |
| P4 | S500 | 1.5424 | 1.5307 | -0.0116 | -0.0076 | 1 | -0.1651 | 0.1882 |

全部四折配对方向、种子delta与置信区间见`paired_comparisons.csv`，不把official和folds
拼成60株。四折Validation覆盖48株但重叠的模型训练和官方集不构成独立重复。
P0/P2未重新训练，原20epoch模型/预测原字节保留；S500/P3/P4复用相同下游实现。
head初值、可训练参数、epoch、step公平性：**PASS**。

## 6. Label efficiency与语义/实例结构辅助证据

25/50/100%即6/12/24株，原nested预算；下面5划分×3seed的均值/SD仅描述，不是15份独立生物学重复。

| condition | label_fraction | mean | std |
| --- | --- | --- | --- |
| P0 | 0.2500 | 1.9986 | 0.1729 |
| P0 | 0.5000 | 1.9413 | 0.1726 |
| P0 | 1.0000 | 1.3122 | 0.1243 |
| P2 | 0.2500 | 1.8593 | 0.1298 |
| P2 | 0.5000 | 1.5743 | 0.1519 |
| P2 | 1.0000 | 1.3376 | 0.1107 |
| P3 | 0.2500 | 1.8356 | 0.1429 |
| P3 | 0.5000 | 1.5866 | 0.1666 |
| P3 | 1.0000 | 1.3933 | 0.0805 |
| P4 | 0.2500 | 1.8892 | 0.1235 |
| P4 | 0.5000 | 1.6028 | 0.1208 |
| P4 | 1.0000 | 1.3581 | 0.1055 |
| S500 | 0.2500 | 1.8128 | 0.1291 |
| S500 | 0.5000 | 1.5718 | 0.1637 |
| S500 | 1.0000 | 1.3280 | 0.0857 |

语义分割：100%标签、同M4插值+XYZ/offset头，plant-macro三类IoU：

| condition | plant_macro_miou | plant_macro_iou_taproot | plant_macro_iou_core | plant_macro_iou_leaf |
| --- | --- | --- | --- | --- |
| P0 | 0.5539 | 0.3407 | 0.3446 | 0.9763 |
| P2 | 0.4714 | 0.2102 | 0.2367 | 0.9672 |
| P3 | 0.4360 | 0.1351 | 0.2082 | 0.9647 |
| P4 | 0.4533 | 0.1735 | 0.2205 | 0.9659 |
| S500 | 0.4567 | 0.1772 | 0.2232 | 0.9662 |

明显负迁移风险阈值：official mIoU下降至少0.03且植物配对CI上界<0；它只是否决风险，
不替代叶数主指标。2048点sampled IoU不是全分辨率，不是叶实例PQ/SBD。

Official paired semantic differences（left−P0，负值代表负迁移）：

| left | right | mean_delta | ci95_low | ci95_high | independent_plants |
| --- | --- | --- | --- | --- | --- |
| P3 | P0 | -0.1587 | -0.1846 | -0.1284 | 12 |
| P4 | P0 | -0.1187 | -0.1407 | -0.0954 | 12 |

训练植株只读叶结构探针：token的32邻域必须>=80%是同一当前leaf实例；同叶/异叶
cosine、per-scan中心化AUC、距离最近匹配异叶pair margin，先扫描再植物macro。
标签仅用于本探针评价，不反传到encoder、不从Val/Test实例标签训练提取器。

| condition | centred_pair_auc | centred_distance_matched_margin | centred_distance_matched_accuracy |
| --- | --- | --- | --- |
| P0 | 0.9839 | 0.2517 | 0.8883 |
| P2 | 0.9670 | 0.3150 | 0.8511 |
| P3 | 0.8437 | 0.2822 | 0.7094 |
| P4 | 0.9653 | 0.3290 | 0.8502 |
| S500 | 0.9665 | 0.3169 | 0.8502 |

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

| condition | pooled_effective_rank | token_effective_rank | pooled_std_mean | token_std_mean | pooled_largest_eigenvalue_fraction |
| --- | --- | --- | --- | --- | --- |
| P0 | 20.7703 | 3.5688 | 0.0889 | 0.3222 | 0.1875 |
| P2 | 13.8915 | 7.1354 | 0.0179 | 0.0435 | 0.5047 |
| P3 | 2.9713 | 3.0720 | 0.0600 | 0.0851 | 0.8143 |
| P4 | 15.0124 | 8.8722 | 0.0138 | 0.0309 | 0.4655 |
| S500 | 12.8629 | 7.2526 | 0.0160 | 0.0376 | 0.5183 |

这是下游微调 **之前** 的各源编码器；P0/P2 与新模型使用同一诊断实现。
`representation_summary.csv` 包含全部划分和种子，不能用局部 token 相似性冒充器官实例性能。
P4 的方差/协方差损失在 train-mode context token 上实施；dropout/局部 patch 多样性可能使
损失满足阈值而 eval-mode pooled/token 仍各向异性，因此不据正则项下降断言全面恢复。

补充 **训练完成后的只读外部 Train 几何诊断**：official 编码器三种子，按来源单独统计，
不使用外部 Test 标签，不改变模型、损失、权重、checkpoint 或成功判定。

| source | condition | objects | pooled_effective_rank | token_effective_rank | pooled_largest_eigenvalue_fraction |
| --- | --- | --- | --- | --- | --- |
| BonnBeetClouds3D | P0 | 128.0000 | 19.5351 | 3.4645 | 0.2818 |
| BonnBeetClouds3D | P2 | 128.0000 | 9.7200 | 6.8085 | 0.5232 |
| BonnBeetClouds3D | P3 | 128.0000 | 3.1027 | 2.3305 | 0.7565 |
| BonnBeetClouds3D | P4 | 128.0000 | 10.6958 | 8.4951 | 0.4605 |
| BonnBeetClouds3D | S500 | 128.0000 | 9.1669 | 6.7796 | 0.5387 |
| Crops3D | P0 | 823.0000 | 9.7142 | 3.5922 | 0.4160 |
| Crops3D | P2 | 823.0000 | 10.3424 | 7.2643 | 0.5390 |
| Crops3D | P3 | 823.0000 | 2.3513 | 2.1107 | 0.8355 |
| Crops3D | P4 | 823.0000 | 8.6603 | 8.2146 | 0.5645 |
| Crops3D | S500 | 823.0000 | 9.6454 | 7.3270 | 0.5565 |
| all | P0 | 951.0000 | 10.2436 | 3.6945 | 0.4091 |
| all | P2 | 951.0000 | 10.5220 | 7.4272 | 0.5362 |
| all | P3 | 951.0000 | 2.4447 | 2.1427 | 0.8261 |
| all | P4 | 951.0000 | 9.3833 | 8.5473 | 0.5422 |
| all | S500 | 951.0000 | 9.7645 | 7.4915 | 0.5537 |

外部 species 级完整结果见 `external_representation_summary.csv`，全维方差/协方差和
distinct-object 中心化/不中心化 cosine 见 `external_representation_diagnostics.json`。
混合来源的秩可能反映扫描系统/物种差异；distinct object 相似性不称为不同独立植株相似性，
也不能作为未见 Sugar4D 主任务成功的替代证据。

正式SSL完成45runs、22500steps，
实测累计8046.9秒；M5-A另2000steps。
新下游180runs；矩阵共
300/300（含原P0/P2）。
探针75/75。
显存、曝光量、wall time是实测共享负载run统计，不能作方法速度排名。
SSL/下游完整状态与best encoder保存于`outputs/m5/`；重载验证
180个新下游checkpoints：PASS。
完整 SSL 优化器/学习率调度/EMA/随机数/采样器及矩阵来源验证：**PASS**。
环境版本、CUDA/驱动与依赖见 `environment.json`、`environment_freeze.txt`。

## 8. 决策与复现

```json
[
  {
    "condition": "P3",
    "controls": [
      {
        "right": "P0",
        "passed": false,
        "absolute_improvement": 0.07708551785185058,
        "relative_improvement": 0.044091511236355276,
        "ci95": [
          -0.4146794154608563,
          0.24310227123634803
        ],
        "improving_seeds": 3,
        "improving_folds": 4
      },
      {
        "right": "P2",
        "passed": false,
        "absolute_improvement": -0.14751958527314923,
        "relative_improvement": -0.09681653701629951,
        "ci95": [
          0.06227609339555951,
          0.24417890008515364
        ],
        "improving_seeds": 0,
        "improving_folds": 2
      },
      {
        "right": "S500",
        "passed": false,
        "absolute_improvement": -0.14049661242299605,
        "relative_improvement": -0.09178433912269156,
        "ci95": [
          -0.05438608196046622,
          0.3326423292151755
        ],
        "improving_seeds": 0,
        "improving_folds": 3
      }
    ],
    "semantic_negative_transfer": true,
    "representation_gate": false,
    "primary_pass": false,
    "local_primary_benefit": true
  },
  {
    "condition": "P4",
    "controls": [
      {
        "right": "P0",
        "passed": false,
        "absolute_improvement": 0.20595397933490522,
        "relative_improvement": 0.1178019224242528,
        "ci95": [
          -0.5099282536207894,
          0.09301670664795922
        ],
        "improving_seeds": 3,
        "improving_folds": 4
      },
      {
        "right": "P2",
        "passed": false,
        "absolute_improvement": -0.018651123790094593,
        "relative_improvement": -0.012240660882253366,
        "ci95": [
          -0.04133405277022387,
          0.08156545683948778
        ],
        "improving_seeds": 1,
        "improving_folds": 1
      },
      {
        "right": "S500",
        "passed": false,
        "absolute_improvement": -0.011628150939941406,
        "relative_improvement": -0.007596497387624823,
        "ci95": [
          -0.16506133915649515,
          0.18823185542391405
        ],
        "improving_seeds": 1,
        "improving_folds": 1
      }
    ],
    "semantic_negative_transfer": true,
    "representation_gate": true,
    "primary_pass": false,
    "local_primary_benefit": true
  }
]
```

最终唯一研究选择：**JEPA-STOP**。不事后换任务，不自动训练Temporal-JEPA、
未来植物预测或Diffusion，不合并/提交/推送Git。

自动测试：{"tests_m5.xml": {"tests": 18, "failures": 0, "errors": 0, "skipped": 0}, "tests_contracts.xml": {"tests": 39, "failures": 0, "errors": 0, "skipped": 0}, "tests_all.xml": {"tests": 147, "failures": 0, "errors": 0, "skipped": 0}}。
评估输出曾遇到 NumPy boolean JSON 序列化错误，已仅在输出边界转换精确标量，
锁定评价实现、指标和判定不变；复核记录见 `OUTPUT_SERIALIZATION_NOTE.md`。
全套旧环境执行，保持隔离环境和历史依赖不被覆盖。
图在`figures/`，CSV可复算，editable PDF/SVG及PNG；直接查看图形质检不是人工签署。
图源/版面检查：PASS；当前模型直接查看 PNG 的视觉检查：PASS。

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
