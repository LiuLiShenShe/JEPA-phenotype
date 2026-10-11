# Plant Temporal-JEPA — M0

本项目只实施数据准备与质量验证，不训练 DINOv2、GRU、JEPA 或任何模型。
实际运行结论见 `reports/m0/M0_REPORT.md`；机器可读结果见
`reports/m0/validation_results.json`。项目结构直接位于当前根目录。

## GitHub 仓库与提交

项目仓库：<https://github.com/LiuLiShenShe/JEPA-phenotype>（私有）。
主仓库的 `origin` 和默认推送目标均为此仓库；提交作者使用
`LiuLiShenShe` 的 GitHub noreply 邮箱，不修改其他项目的全局 Git 配置。

第三方官方实现作为只读 Git 子模块固定在
`0d9a24ff146d12cc8fca6bdfa2538d196754f456`；其官方 URL 仅用于获取源码，
不用于提交或推送。本地来源记录见 `reports/m0/OFFICIAL_COMMIT.txt`。

新副本请递归克隆，并启用推送保护；Git 不会自动复制本地配置：

```bash
git clone --recurse-submodules https://github.com/LiuLiShenShe/JEPA-phenotype.git
cd JEPA-phenotype
git config core.hooksPath .githooks
git config remote.pushDefault origin
git config push.recurseSubmodules no
git -C third_party/PhenoLeaf-TS config remote.origin.pushurl no-push://PhenoLeaf-TS
```

已启用的 `.githooks/pre-push` 会拒绝推送到此项目仓库以外的地址。
正常提交使用 `git add .`、`git commit -m "说明本次修改"`、`git push`。
原始数据、派生数据表、虚拟环境和含原始像素的 QC 图片按 `.gitignore`
保留在本地，不随提交上传。

## 数据来源与许可

- 官方数据：<https://huggingface.co/datasets/rick77a/PhenoLeaf_TS>
- 官方代码：<https://github.com/pisyntor/PhenoLeaf-TS>
- 数据说明：`third_party/PhenoLeaf-TS/docs/DATASET.md`
- 官方预处理：`third_party/PhenoLeaf-TS/phenoleaf_ts/data/prepare.py`
- 数据许可：`third_party/PhenoLeaf-TS/DATASET_LICENSE.md`，仅非商业学术/教学，必须引用。
  未经书面许可，不得再分发、公开托管原图或 mask。代码 Apache-2.0 不改变数据许可。
- 下载替代方式、来源与核验证据：`reports/m0/SOURCE_AND_LICENSE.md`。

原始数据及包含原图的 QC 图像只保留本地，并在 `.gitignore` 中排除。
不要上传到公开仓库、外部视觉 API 或其他未经授权的平台。

## 全新环境安装

已实际验证：Linux、Python 3.13.12、PyTorch 2.9.1+cpu。
锁定文件针对此 Python/Linux 环境；至少预留 15 GB 磁盘空间。

```bash
cd /path/to/Plant-Temporal-JEPA
python3 --version
df -h .
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-lock.txt
python -m pip check
```

`requirements.txt` 是直接依赖约束，`requirements-lock.txt` 是实际安装的全部版本。
不要为 M0 安装 GPU 训练栈；Dataset 和验收均在 CPU 执行。

新副本中初始化固定官方实现；已有子模块会保留仓库记录的版本：

```bash
git submodule update --init --recursive
git -C third_party/PhenoLeaf-TS rev-parse HEAD
```

## 完整处理命令

从项目根目录执行。下载优先调用官方 `download_phenoleaf(..., extract=False)`。
此环境直连遇到网络问题；按照用户允许的替代方式，`auto` 会尝试明确记录的镜像。
所有来源必须通过固定 Hugging Face LFS SHA-256 和 ZIP CRC，不能仅凭名称/大小信任文件。

```bash
source .venv/bin/activate
python scripts/download_data.py --source auto --data-root data/raw
python scripts/prepare_metadata.py --data-root data/raw --output-dir data/metadata
python scripts/build_phenotypes.py --frames data/metadata/frames.csv --output-dir data/metadata
python scripts/create_splits.py --seed 42
python scripts/build_temporal_windows.py --history-length 3 --horizons 2 4 8
python scripts/run_m0_acceptance.py
```

`--source official` 只调用官方函数；`--source hf-mirror` 显式使用已验证的替代来源。
本次实际解压目录是 `data/raw/PhenoLeaf-TS_DS/`，不是说明中的 `leaf_dataset_colour/`。
扫描器根据真实 `01_raw_dataset/`、`02_leaf_labels/` 和工作簿结构发现根目录，
不重命名或覆盖原始文件。重复提取遇到原文件不一致即报错。

固定归档：5,486,793,580 bytes；SHA-256：
`608c7b07125281fb92517182c0873d1a9c87d696e2adedcb11c002173887de2f`。
固定数据版本：`ee4c8dac44593a69c8a72e1ca09d686133161d92`。
镜像不是官方域名；本次同时核对了该版本 LFS 元数据与官方 CDN 字节探针。
详情在 `reports/m0/download_report.json`、`official_cdn_probe.json`。

## 元数据与标签约定

- `plant_uid = genotype/replicate`；官方 `plant_id` 可能只表示基因型，不能直接使用。
  文件尾部的重复编号可能与整理后的目录不同；不能从尾号重新分组。本次 318 个
  采集前缀 ID 与目录 plant_uid 一一对应；1632 帧/65 株尾号别名已单独记录，
  没有跨目录复用采集 ID，见 `data/metadata/plant_identity_audit.csv`。
- 文件名 `YYYY_MM_DD_HH_MM_SS` 的日历、唯一性和排序均验证后才使用采集墙钟时间。
  保留原名、排序依据、异常标记；未知时区不假装 UTC。
  无法验证顺序的植株隔离，不能把文件序号解释为天数。
- `frame_idx`、`time_index` 是同株从零开始的原始观测索引；隔离帧后不压缩编号。
  历史必须是三次连续原始观测，缺口不能补拼。
- RGB/mask 逐路径、逐尺寸配对；缺失、损坏、孤立 mask 均记录，不静默丢弃。
- 官方 Excel 的 31 色必须与官方实现一致，精确 RGB 匹配；黑色背景不是叶片。
  `leaf_id` 是调色板位置 1–31，只在同株内解释，不是跨株全局叶 ID。
- `leaf_count` 是合法颜色实例数；`projected_area_px` 是非黑像素数。
  非法颜色不计新实例；另存合法像素面积、非法像素数量与位置。
  含非法颜色或零面积标签的帧隔离；诊断 CSV 保留，不修改 raw mask。
- `projected_area_ratio = projected_area_px / (H*W)`，不是物理叶面积或 LAI。
  裁剪尺寸有变化，未经尺度校准不能把全部面积变化解释为真实生长。
- 叶片消失/重现、面积跳变、描述范围外叶数只作预警，遮挡/衰老可能正常。
- 只有 `quality_status == valid` 可进入样本；连续性观察预警不等同坏帧。
- Train/val/test 采用 seed 42、按基因型分层的植株级近似 70/15/15。
  优先保持基因型代表性及整株独立；不用官方图像级 split，不复制三份图片。

## PyTorch Dataset

```python
import sys
sys.path.insert(0, "src")
from datasets.plant_temporal_dataset import PlantTemporalDataset

dataset = PlantTemporalDataset(
    metadata_path="data/metadata/frames.csv",
    split_path="splits/phenoleaf_plant_split_seed42.json",
    split="train",
    history_length=3,
    horizon=4,
    image_size=(224, 224),
)
sample = dataset[0]
```

`image_size` 为 `(height,width)`，只 resize 历史 RGB 的读取副本，不改 raw 或 GT 面积。
默认 `None` 保留原尺寸；窗口尺寸不一致时报明确错误；跨株批处理建议指定尺寸。
支持自定义 `transform`、`windows_path`、`project_root`。

返回：`images`/`input_images` 是 `[3,3,H,W]` float tensor，另含植株 ID、历史路径、
索引、时间戳、`time_deltas`、未来索引、跨度、叶数、原图投影面积及比例、
`target_growth_stage`、split 和 window ID。阶段为官方叶数阈值派生标签，
不是独立生理阶段 GT。验证可靠时，`time_deltas=[0, Δt12, Δt23]` 为秒，
`target_time_delta_seconds` 为最后历史帧至目标的秒数。

预测 +2/+4/+8 是**观测偏移，不是天数**。`__getitem__` 只打开历史 RGB，
不打开未来图像或 mask。未来标签离线来自 GT；索引被篡改/跨株时构造即失败。
窗口允许重叠，但同株永远在同一集合。RGB /255 是固定缩放，不拟合全数据统计量；
M1 需拟合的标签/特征标准化只能使用 train。

## 自动化验收

重新执行全部真实 M0 验收的一条命令：

```bash
.venv/bin/python scripts/run_m0_acceptance.py
```

依次执行 pytest/JUnit XML → 全管线字节一致性重跑 → 所有文件完全解码、ZIP CRC/SHA、
独立像素标签复算 → 本地 9 张图 → 几何/时间/标签分布 → 验收器。
验收器逐行检查所有窗口，运行九个 split/horizon Dataset smoke 和 CPU DataLoader。
重跑使用本地校验归档，不需要网络或 GPU；失败返回非零退出码。

单独执行：

```bash
.venv/bin/python -m pytest -q --junitxml=reports/m0/test_results.xml
.venv/bin/python scripts/check_reproducibility.py
.venv/bin/python scripts/audit_dataset.py --workers 8
.venv/bin/python scripts/visualize_m0.py
.venv/bin/python scripts/summarize_m0.py
.venv/bin/python scripts/validate_m0.py
```

不能仅凭合成 pytest 成功宣称真实 PASS；完整归档来源、全量审计、元数据 SHA 一致、
全窗口验证、测试结果和真实图表都必须成立。

## 主要交付路径

| 路径 | 内容 |
|---|---|
| `data/metadata/frames.csv`, `plants.csv` | 完整观测/植株、时间与质量 |
| `data/metadata/phenotypes.csv`, `leaf_instances.csv` | 整株及逐叶 ID/面积/bbox |
| `data/metadata/data_issues.csv`, `file_manifest.csv` | 异常定位、CRC/SHA |
| `data/metadata/dataset_statistics.json` | 全量统计与质量规则 |
| `splits/phenoleaf_plant_split_seed42.json` | 植株分配、图片索引及隔离列表 |
| `data/processed/temporal_windows.csv` | 3 帧历史/+2/+4/+8 未来标签 |
| `reports/m0/M0_REPORT.md`, `QA_CHECKLIST.md` | 最终结论与证据 |
| `reports/m0/*statistics.json`, `split_statistics.csv` | 表型、时间、窗口和集合分布 |
| `reports/m0/figures/` | 9 张图，12 组叠加，6 株曲线，6 组窗口，选样清单 |
| `reports/m0/test_results.xml`, `validation_results.json` | 自动化测试及真实验收 |
| `reports/m0/environment.json`, `requirements-lock.txt` | 环境及锁定依赖 |
| `tests/test_*.py` | 正常/异常、泄漏、复现、加载与归档安全测试 |

## 合成数据（仅开发）

```bash
.venv/bin/python scripts/create_synthetic_fixture.py --output-root data/synthetic_m0
```

可生成 12 株/3 基因型的小数据集。pytest 自行创建临时合成夹具，不需要此示例。
合成结果不能满足真实 M0；检查合成材料时必须指定 `validate_m0.py --synthetic`，
只能输出 PARTIAL/FAIL。最终人工检查由用户根据本地 figures/ 完成。

## 学术引用

以下为官方 README 的引用记录：

```bibtex
@InProceedings{saric2026phenoleafts,
  author = {Sari\'c, Rijad and Azam, Basim and Khan, Sarmad and \v{C}ustovi\'c, Edhem},
  title = {{PhenoLeaf-TS}: A Time-Series Benchmark for Leaf Instance Segmentation, Tracking, and Growth Stage Classification},
  booktitle = {Computer Vision -- ECCV 2026},
  year = {2026},
  publisher = {Springer Nature Switzerland}
}
```
