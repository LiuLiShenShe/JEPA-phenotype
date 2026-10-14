# M7 复现

全部命令在 `/fj/JEPA-表型` 执行，旧 `.venv-m6-b` 只读复用。公开仓库不再分发无代码许可的PlantNet源码/权重；冻结S2模型与原始数据需要原本地授权资源，详见M6-B来源审计。

```bash
export PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
.venv-m6-b/bin/python scripts/audit_m7_sources.py
.venv-m6-b/bin/python scripts/build_birth_dataset.py --device cuda:1
bash scripts/train_m7_matrix.sh
.venv-m6-b/bin/python scripts/report_m7.py
.venv-m6-b/bin/python scripts/analyze_m7.py
.venv-m6-b/bin/python scripts/plot_m7.py
PYTHONPATH=src .venv-m6-b/bin/python -m pytest tests/test_m7.py tests/test_m6_a.py tests/test_m6_b.py -q
.venv-m6-b/bin/python scripts/verify_m7.py
```

`build_birth_dataset` 首次全768帧实际构建，已有 `dataset_manifest.json` 时先核验锁再跳过；不要删除历史输出。Train S2推理只新增M7预测，原Val/Test分割及TrackPlant3D直接读取。

单个实验：

```bash
.venv-m6-b/bin/python scripts/train_birth_detector.py --method PointNet_GRU --seed 17 --device cuda:1
.venv-m6-b/bin/python scripts/train_birth_detector.py --method C2 --seed 29 --device cuda:1
.venv-m6-b/bin/python scripts/evaluate_birth.py --split val
.venv-m6-b/bin/python scripts/evaluate_birth.py --split test
```

所有学习基线种子17/29/43；只在Train监督与标准化，统一最终40轮checkpoint；RF200棵树。Validation跨三种子选择阈值，再冻结system；Test新模型概率在冻结后才计算，不择优seed。Test为已使用过的探索性复核。

`outputs/m7/birth` 保存全候选概率、40轮日志、checkpoint；`matching` 保存pair概率及metric checkpoint；`tracking` 保存全器官轨迹/首观测事件/错接link；`frames` 保存原全分辨率候选索引、GT评价数组、原mask和测量。集成mask可由固定候选索引、阈值、确定性NMS和覆盖规则恢复，生成最终全分辨率预测另见delivery索引。

实际运行首次GRU遇到float64时间数组与float32网络不匹配，保留`GRU_dtype_failure.log`。随后内容审计发现缓存来自单帧控制修复前的构建进程，仍含候选source/novel两列。该批模型、日志和缓存完整隔离到`invalid_source_leak_runs`及`invalid_source_novel_leak_birth_dataset.npz`，不参与任何选择和正式结果。按冻结协议重建54维单帧控制，显式排除source/novel并统一时间float32，重新从头训练本轮所有出生/匹配模型；forecast输入不受影响。不是根据Test效果重训、未调整锁定预算和阈值。`training_budgets.csv`只统计21个有效正式拟合，隔离批次单独保留审计证据。

`candidate_oracle`仅评价时根据GT选中候选，得到该固定候选池的乐观召回上限；不进入模型输入、阈值选择或自动系统。它的F1不应作为自动模型结果宣传。

范围限制：不重跑/覆盖旧方法、不修改M6-B报告/指标、不新增segmentation骨干、不训练JEPA。当前模型输入中无GT时间ID，未来GT仅作训练target/evaluator，不作为当前输入。
