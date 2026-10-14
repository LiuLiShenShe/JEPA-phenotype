# M7 工程执行审计

日期：2026-10-13。正式协议先冻结，独立植株split、随机种子、预算、IoU规则和成功门槛没有因结果更改。

## 有效正式运行

四种当前出生分类器、一个下一扫描整株RF和C1/C2匹配模型，每种17/29/43三种子，共21次有效正式拟合。NN从头40轮，RF200树。最终epoch固定，不挑Test最优；旧S2/TrackPlant3D结果不重评或改写。旧S2训练集预测仅为生成M7所需输入，记录为in-sample，不冒充out-of-fold。

## 保留的工程失败

1. 初次GRU输入时间数组float64与编码float32不兼容。真实异常保留于`GRU_dtype_failure.log`。修复仅为统一时间输入dtype；增加float64输入前向/反向自动测试。
2. 构建缓存与后来修正的单帧特征代码不同步，初次缓存56列仍含候选source/novelty。通过内容逐列审计确认后停止旧批次，将全部旧模型、日志和旧Validation选择隔离至`invalid_source_leak_runs`；旧NPZ也保留。重建54列当前特征，不含这两列，时间缓存float32。全部出生和匹配模型按同一锁定预算从头重训，forecast也重新拟合。作废批次不参与正式结果、方法选择或有效训练预算统计；不是删难样本或根据Test重训。
3. Validation导出阶段反复解压相同全分辨率点索引并重复计算IoU，运行较慢。停止该评估进程并保留`evaluate_val_redundant_io.log`，不停止或改写完成的21个模型。为所有方法复用同一评价专用IoU矩阵，规则仍为strict IoU>0.5及相同Hungarian匹配。`evaluation_cache_equivalence.json`逐株比较已完成的未缓存结果与缓存结果，数值容差1e-12；自动测试覆盖空集合、重复候选、子集和Oracle，结果完全一致。原Validation选择不更改，Test仍在系统锁定后运行。

`matrix_execution.log`中的进程终止来自第3项主动性能修复，不代表训练未完成。有效模型的`completed.json`、40轮`epochs.csv`、真实权重和checkpoint哈希由`verify_m7.py`逐一核验。使用缓存后分别完成Validation和Test评估，全部方法/种子保留，不省略失败植株。

## 保护和透明性

`.venv-m6-b`只读复用，不安装包或改变历史环境；历史模块运行设`PYTHONDONTWRITEBYTECODE=1`。只用GPU1，未干预GPU0其他任务。不修改上游代码，不新增segmentation主干或JEPA。旧成果核验以启动HEAD、tracked SHA256及7431个历史文件大小/mtime为准，完整结果见`protection_verification.json`。

一般博查实际HTTP403配额不足；官方Sugar4D网页和Dataverse API实际HTTP200；Context7当前官方文档已取回。没有把失败搜索描述为成功，也没有把点云提取的形态参照说成完整人工生长真值。
