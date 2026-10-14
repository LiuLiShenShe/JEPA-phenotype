# M7 交付索引

本阶段只新增M7路径，不修改M0—M6-B；不提交、合并或推送仓库，不自动开展下一阶段。

## 阅读顺序

1. `M7_REPORT.md`：唯一研究结论、下一阶段建议及四个科学问题的回答。
2. `DATA_INTERFACE_AUDIT.md`：官方split、语义/实例、三类出生时间、删失、物理单位和测量参照来源。
3. `protocol_lock.json`、`../../configs/m7/protocol.json`：正式训练前冻结的规则及上游哈希。
4. `decision.json`、`system_selection.json`、`statistical_scope.json`：Validation选择、成功门槛和统计范围。
5. `REPRODUCE.md`、`EXECUTION_AUDIT.md`：复现命令和工程失败/恢复证据。

## Benchmark及逐样本证据

- `organ_birth_events.csv`：675个合法叶首次观测，含初始扫描左删失和全部要求字段。
- `birth_statistics.csv`、`birth_cohort_statistics.csv`、`label_stability_audit.csv`：逐株、DAS、基因型、点数/芯部代理及可见间断。
- `frame_audit.csv`、`dataset_manifest.json`：48株768帧数据接口。
- `candidate_recall_ceiling.csv`：固定候选池的评价用可达上限，不是自动方法。
- `event_level_results.csv`、`failure_cases.csv`、`event_failure_contributions.csv`：首观测漏检与已检出但错误身份分别记录。
- `track_level_results.csv`、`gt_track_level_results.csv`：预测轨迹和参照器官的漏失、混合身份、覆盖及间断。

## 指标、统计及测量

- `birth_metrics.csv`、`birth_summary.csv`：当前出生候选定位。
- `forecast_metrics.csv`、`forecast_summary.csv`：独立的下一扫描整株事件预测，含always/never-new控制；不是叶实例发现指标。
- `tracking_metrics.csv`、`tracking_summary.csv`：全部植株和三随机种子，旧Current只读导入。
- `plant_intervals.csv`、`tracking_paired_plants.csv`、`temporal_paired_plants.csv`、`temporal_comparisons.csv`：以植株为单位的区间和配对差。
- `growth_metrics.csv`、`growth_summary.csv`、`measurement_failure_cases.csv`、`measurement_failure_contributions.csv`：Oracle与自动系统、条件误差与完整覆盖分母。
- `training_budgets.csv`：21个有效正式拟合的样本数、参数量、40轮/200树预算及checkpoint哈希。

## 模型、预测、图与验证

- `../../outputs/m7/birth/*/seed_*`：四种出生模型的三种子权重及全部候选概率。
- `../../outputs/m7/matching/*/seed_*`：C1/C2监督matching及metric-learning权重、配对概率。
- `../../outputs/m7/forecast/seed_*`：下一扫描RF和全部预测。
- `../../outputs/m7/tracking/*/seed_*/{val,test}`：全分辨率预测NPZ、逐观察/逐track/逐link/出生及逐株CSV。
- `../../outputs/m7/frames`：候选原始点索引、物理坐标样本和评价专用标签；GT字段不进入运行模型。
- `figures/`、`FIGURE_CONTRACT.md`、`FIGURE_QA.md`：真实统计和非精选最佳植株的失败点云案例；PNG/SVG/PDF，以及直接查看后的检查记录。
- `train_*.log`及每个模型的`epochs.csv`：实际训练日志；作废工程批次独立隔离，不能参与选择。
- `tests_final.log`、`final_verification.json`、`protection_verification.json`、`delivery_manifest.json`：自动测试、完整性、旧成果保护和全交付哈希。
- `sources/`：实际取得的官方页面、元数据和Context7文档。一般博查HTTP403配额不足已披露，没有伪造搜索。
