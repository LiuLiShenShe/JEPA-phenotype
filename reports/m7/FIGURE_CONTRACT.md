# M7 实验图合同

已有工程为 Python 工作流，所有图、预览及导出只使用同一 Python/Matplotlib 后端。

目的不是挑最优植株，而是展示事件难度、时间信息的配对增量，以及出生/身份/测量三类错误的不同分母。

- `birth_benchmark`：非初始帧事件的 DAS 分布、小叶与 core 邻近比例、segmentable 延迟。来源 `organ_birth_events.csv`；被删失事件不伪造延迟。
- `temporal_evidence`：12株上的历史 RF/GRU 相对共享候选单帧对照的差值，和新系统事件 F1 / IDF1。三种子先按植株平均；95%植株 bootstrap CI 只是选择后探索性区间。
- `failure_contributions`：首帧事件错误与全部静态长度测量错误分别显示。不同分母不混在同一总百分比。
- `birth_failure_pointclouds`：按失败类型选择初始点数中位数附近的真实 Validation 事件；画全分辨率标签的确定性显示子采样，GT只作为图示参考，不进入推理。不能把图示子采样误称模型采样。

定量图白底，固定颜色族；SVG可编辑文字、PDF矢量、PNG预览。PNG实际查看后记录 QA；不宣称已达到任何期刊投稿规范。全部数据文件和案例选择表随图交付。
