# M7 输入、事件标签与因果接口

## 数据和许可

保留 Sugar4D V4 2026-08-27 发布实体及官方植株划分，48株×16帧；Train24/Validation12/Test12。DOI `10.60507/FK2/IS8YBZ`，官方许可 CC BY4.0。实时元数据和页面取得HTTP200，哈希见`sources/retrieval_audit.json`；没有重新下载升级或覆盖旧数据。一般博查检索HTTP403配额不足，不冒称已搜索到结果。

来源： https://www.ipb.uni-bonn.de/data/sugar4d/index.html ; https://doi.org/10.60507/FK2/IS8YBZ 。PlantNet固定checkpoint和上游commit沿用M6-B，未公开再分发其源码/权重。

## 标签和时间

原PLY `label_semantic`: 0=taproot、1=core、2=leaf；taproot实例-2，core实例-1，合法叶ID>=0包括0。评价数组只把leaf原ID+1，以0表示非叶背景；没有把leaf0当背景。运行只消费XYZ、冻结模型预测语义/实例及过去XYZ，GT语义/身份仅Train监督或evaluator。

Visible birth=首次释放点云观测该ID；最初扫描198片叶为左删失，477片非初始事件保留。不是生物出生。Segmentable=固定S2严格IoU>0.5首匹配，是模型相关代理；Trackable=旧结果三连续scan同track首帧，是回顾性标注。Train没有旧trackable记录，以缺失及不可评价标记，不能把缺失当末帧发生。

## 面积及长度参照的真实来源

`measurements/leaf_related.csv`6778条来自官方**点云自动提取的形态测量**，不是各时刻人工器官真值。`initial_leaf_area`来自该CSV的cm²，仅描述/评价，不输入学习模型。

`measurements/reference/leaf_related_manual.csv`450条只含93DAS人工叶长/宽，不包含人工叶面积，也没有整条人工生长轨迹。因此M7的长度/面积/growth误差是相对官方提取参照的误差；不能称独立实测精度，也不据此认证表型系统可靠。即使Oracle全GT mask+GT identity，相同简易测量器也明显偏离官方提取参照。

## 物理坐标和采样

- XYZ保持米，不独立每帧单位球归一化，不通过GT器官对齐。初帧最低z10%点均值作为plant固定物理anchor，仅出生几何特征的质心减anchor；形态/面积/距离仍保持物理单位。
- 差分4mm voxel，7mm新颖距离，12mm密度/DBSCAN尺度。未增加非刚性配准；已有叶增长/扫描差异可能产生伪候选。
- 每帧均匀16384点空间缓存；候选在当前物理中心半径6cm提取过去最多3帧局部点，均匀128点，除0.12m，重建坐标为`patch*0.12+current_candidate_center`。历史点只来自<=t。
- 器官embedding使用各实例质心居中128点/0.12m，配对另有真实质心位移及方向/面积/长度变化。实例局部ID进入tracking前每帧独立随机重排，不以编号关联。
- 全分辨率候选原始点索引及offset保存在`outputs/m7/frames`，所有最终mask恢复到原PLY顶点顺序；显示图子采样与训练采样不同，单独注明。

## 模型字段白名单和控制

历史出生110维=当前29维+三个时刻各25维局部几何+六个时间/可用字段；单帧54维=当前25维形状+预测置信度/叶语义比例2维+当前局部25维+当前时间2维，**无候选source/novel两列**。

两种模型共享因果候选（冻结分割mask+差分region），所以“single-frame control”只是单帧**特征编码**控制，并非纯单帧端到端发现模型。早期缓存错误批次及修复证据留在`invalid_source_leak_runs`；这些不参与正式结果。

MLP/metric matching只输入75维物理/形态配对，C2附加两实例128点编码。GT时间ID只能用于Train“同叶/异叶”标签及事后评价，全候选/配对均保留，包括未匹配自动实例负例，不通过GT筛出好实例。

## 已知限制

24株训练的固定S2预测是in-sample，非out-of-fold。首观测区间3–7天，缺少微小/未分离叶的人工可分离性审核。GT ID缺扫描后重现可能来自遮挡或标注变化，不能只看表格判断标签错误。相对点比例“小叶”不等于绝对幼叶。Test曾多次使用，本轮只能探索性固定复核。
