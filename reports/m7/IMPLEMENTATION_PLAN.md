# M7 实施计划

1. **审查和冻结**：核对 M6-B 的真实 S2 预测、旧追踪结果、Sugar4D 48 株官方划分；记录当前 Git HEAD、旧成果哈希和数据源。冻结 `configs/m7/protocol.json` 后才训练。
2. **M7-A benchmark**：从全部官方 PLY 建立可见事件，首次扫描左删失；已有 S2/TrackPlant3D 结果定义模型相关 segmentable 与三帧稳定 trackable 代理。未达到者右删失，不能以生物出生时间解释。
3. **因果候选及 M7-B**：真实物理米坐标上 voxel occupancy/density 差分 + DBSCAN；同一候选集训练三种子历史 RF、PointNet-GRU，加入单帧 RF/PointNet 匹配对照。额外一帧前瞻 RF 与当前发现任务分开。
4. **M7-C**：相同生命周期的三种子 MLP matching 与 supervised metric learning organ embedding；GT 时间身份只作为 Train 的监督和评价标签。固定 S2 分割不再训练或优化 PQ。
5. **Validation 优先**：全植株事件/身份评估，三种子共同选择阈值及方法，按植株配对 bootstrap。冻结选择后才运行旧 Test 的探索性复核。
6. **M7-D 和交付**：成功后才开展恢复轨迹比较；若未稳定，只交覆盖/错误分解，不宣布绝对长度/面积可靠。保留所有候选、预测、事件、track、失败、训练日志、checkpoint、测试和保护检查。

写范围仅为新增 `src/birth_detection`、`src/organ_tracking`、`src/temporal_encoder`、`scripts/*m7*` 和三条要求的 CLI、`tests/test_m7.py`、`configs/m7`、`reports/m7`、`outputs/m7`。不改旧方法、不新增分割主干、不训练 JEPA、不提交/合并/推送。

运行资源：现有 `.venv-m6-b` 只读复用，不安装或修改历史环境；`PYTHONDONTWRITEBYTECODE=1` 防止写旧模块缓存。使用经核验空闲 GPU1，GPU0 外部任务不干预。小型模型预算固定40轮，而非无限调参。

主要风险：时间稀疏造成事件区间删失、差分误把已有叶增长当出生、候选无法独立分离芯部叶、Train S2 预测为 in-sample、48株同一来源的外推限制、测量器偏差。候选可达上限与每种失败必须报告。

测试：因果窗口不含未来、全部植株/合法叶保留、左/右删失、严格 IoU、输入字段排除 GT、独立帧局部 ID、身份一对一匹配、三种子预算和 Validation 选择、旧成果哈希不变。

检索状态：2026-10-13 博查请求实际 HTTP403（配额不足）；不可声称已获得一般搜索结果。官方 Sugar4D 网页实际 HTTP200；PyTorch/scikit-learn Context7 resolve→docs 实际获取，记录在 `sources/`。`/home/test/.codex/RTK.md` 不存在。没有 `.codegraph/`，不擅自索引。
