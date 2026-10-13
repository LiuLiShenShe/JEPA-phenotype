# M6-A 数据与代码来源审计

审计日期：2026-10-12。博查搜索实际返回 HTTP 403；直接访问官方 GitHub API、固定版本源码和 Bonn Dataverse 成功。下列项目情况来自这些实时官方材料与本地文件核验，不是假装搜索成功。

## Sugar4D V4：沿用，不升级历史数据
- 官方页面：https://www.ipb.uni-bonn.de/data/sugar4d/index.html
- DOI：https://doi.org/10.60507/FK2/IS8YBZ
- 实时元数据：https://bonndata.uni-bonn.de/api/datasets/:persistentId/?persistentId=doi:10.60507/FK2/IS8YBZ
- 2026-08-27 发布的 V4，CC BY 4.0。压缩包 6,270,780,517 bytes；SHA256 `69e25b87b470173d30a70fbac9d4caf2ae968082cd83378cd87d93601b370908`，MD5 `27e44210d95a85cb04b93513f0ef2beb`。
- 768 个真实 PLY，48 株×16次实际扫描；二进制 XYZ、法线、语义及实例标签。两张 CSV 包含植株与器官自动表型；最终时间点有部分人工长度验证，无人工面积或连续死亡事件真值。
- 保留原始 24/12/12 整株划分。新评估只处理 val/test 共24株384帧；没有训练，train 不需要再次推理。675个全数据器官身份是评价真值，不是追踪输入。

## Pepper-4D：官方信息可审计，数据不可访问
- 官方：https://github.com/foysalahmed10/Pepper-4D
- 官方论文：https://www.mdpi.com/2223-7747/15/4/599
- README 声明29株、916点云、约10GB、3个子集、不同生长条件；`frames/*.txt` 与 semantic/instance/tracking/health 标签文件、元数据并列。此结构为官方说明，**未通过下载实体核验**。
- 三个 Google Drive 官方下载地址均实际尝试，全部 `Network is unreachable`，见 `sources/pepper_download_attempts.json`。未取得任何完整数据子集、未训练或评价辣椒，不能声称完成跨物种验证。
- 仓库无显式许可证；GitHub license API 404。论文许可不自动等于数据许可。无模型实现或预训练权重；它是数据发布项目，不是可运行分割器。

## PlantNet / PSegNet：权重与实现成功获取
- 官方：https://github.com/Huang2002200/PlantNet-and-PSegNet
- 固定来源 `5efe9ced49473f0d59c618b7e140ffeeb692d9b1`。PyTorch模型/工具及两份epoch199权重实际下载；PlantNet权重53,599,453 bytes，PSegNet权重40,459,367 bytes，逐文件SHA256在 `downloaded_files.csv`。
- 官方原始546个单株点云为烟草/番茄/高粱，不解释为546个独立生物学植株；TXT六列 XYZ/instance/semantic/object。语义0/1、2/3、4/5分别对应三物种茎/叶；训练前转换H5，公开平均实例大小用于聚类过滤。
- 官方环境说明为Python3.8.18/PyTorch2.0.1，旧TensorFlow为Python3.7.13/TF1.13.1；本次不启用旧TensorFlow栈，使用隔离Python3.11.15/Torch2.8.0+cu128。
- 两网络都严格加载全部权重并实际4096点前向推理，重复运行标签/置信度完全一致。PSegNet仅可运行性检查，不做Sugar4D指标择优。
- 官方原始RAR与配套test.h5在仓库中是Git-LFS对象；本阶段未完整下载它们，也不把小型pointer文件记为数据下载成功。Sugar4D零样本运行不依赖它们。
- 代码和权重许可证未声明/未核实；公共可访问不表示可再分发。完整上游副本留在忽略的 `data/raw/m6_a`，只交付适配器和哈希/来源。

## TrackPlant3D：官方方法可运行，但需要接口适配
- 官方：https://github.com/entarot/TrackPlant3D-3D-organ-growth-tracking-framework-for-organ-level-dynamic-phenotyping
- 固定来源 `3c04919dc86d75acd1a3159a1e1f74893e0bc5e4`。下载3DEPS、CPD、器官匹配、评价源码及requirements；完整四物种配套数据不作为本次实验输入。
- 输入是每时刻**非时序一致的器官实例标签**。3DEPS256点→非刚性CPD→扩展代价矩阵MWPM；不需要预训练追踪模型。
- 官方依赖numpy2.2.5/scipy1.15.2/matplotlib3.10.1/open3d0.19.0。本次隔离环境锁定Open3D0.19.0并使用与历史数值环境相同的numpy2.2.6/scipy1.17.1/matplotlib3.10.6；兼容性修改和实测记录见 `COMPATIBILITY.md`、`official_runtime_smoke.json`。
- 原始脚本有硬编码物种/目录和稀疏标签大小问题。保留官方CPD与代价/dummy权重；只做数组接口、活动标签稠密化、固定随机种子和边界邻域等价性能修复。当前实例GT每帧独立随机置换，时序GT只在评估器出现。
- 无显式代码许可，GitHub license API404；不公开再分发上游代码。成功执行不等于证明与论文全部官方数值完全一致。

## 文档来源与兼容性证据
- Context7实际拉取SciPy、Open3D和Matplotlib当前官方文档；检索结果保存在 `sources/ctx7_*`。
- SciPy：https://github.com/scipy/scipy
- Open3D官方边界核：https://github.com/isl-org/Open3D/blob/v0.19.0/cpp/open3d/t/geometry/kernel/PointCloudImpl.h
- Matplotlib：https://github.com/matplotlib/matplotlib
- `http_audit.json`、官方仓库API响应、README、git tree及commit响应均本地保存；正文中所有项目数量声明明确区分官方陈述与实际本地核验。
