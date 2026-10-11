# M0 交付报告 — Plant Temporal-JEPA

**M0 状态：PASS**

## A. 执行结果
- 工作目录：`/fj/JEPA-表型`
- Python：`3.13.12`；PyTorch：`2.9.1+cpu`（CPU-only，无模型训练）。
- 官方仓库 commit：`0d9a24ff146d12cc8fca6bdfa2538d196754f456`
- Hugging Face dataset commit：`ee4c8dac44593a69c8a72e1ca09d686133161d92`
- 下载归档：5,486,793,580 bytes；SHA-256：`608c7b07125281fb92517182c0873d1a9c87d696e2adedcb11c002173887de2f`
- 全量配对观测：17,082；有效观测：17,056；隔离：26。
- 独立植株：318；基因型：21。

| 集合 | 植株 | 有效图像 | 基因型 | 时间窗口 |
|---|---:|---:|---:|---:|
| train | 219 | 11982 | 21 | 31564 |
| val | 53 | 2815 | 21 | 7385 |
| test | 46 | 2259 | 21 | 5857 |

- 时序窗口总数：44,806；按跨度：`{'2': 15782, '4': 15148, '8': 13876}`。
- 输入 3 个连续的原始观测；预测 +2/+4/+8 **观测索引偏移，不是天数**。

## B. 数据质量
- 原始图像与 mask 不作修改；所有文件完整解码、SHA-256、归档 CRC、RGB/mask 尺寸、精确调色板标签均由全量独立审计核对。
- 审计详情、重复文件/像素组与异常计数：`full_dataset_audit.json`；逐文件证据：`../../data/metadata/file_manifest.csv`。
- 318/318 株时间戳通过官方文件名格式、日历合法性、唯一性与严格排序验证。没有独立设备时钟校准或时区记录；保存无时区墙钟时间，不宣称 UTC 或真实天数预测。
- 黑色背景不计叶片；叶片 ID 是官方 31 色调色板位置（1–31），仅在同一植株内解释身份。
- 非法颜色不计作新叶片；诊断面积包含全部非黑像素，另存合法像素面积。所有含非法颜色的帧从 split 图像索引、历史输入和未来目标中隔离。
- 含隔离帧的植株（仅隔离帧，未删除整株）：`['Is-1/rep_24']`。
- 隔离不压缩观测编号；缺失/异常帧造成的历史窗口或目标缺口均跳过，不能把 +4 改成第 4 个剩余有效帧。
- 叶片身份消失/重现、面积突变和描述范围外叶数只作观察性预警，可能来自遮挡/衰老，不直接删数据。
- 异常事件分类：`{'invalid_palette_colors': 26, 'leaf_count_outside_dataset_card': 9, 'leaf_identity_reappears': 1726}`。同一帧可以对应多条事件，事件数不等于坏图像数。
- `projected_area_ratio` 是图像面积比例，不是真实物理叶面积，也不是 LAI；原图尺寸不统一，跨植株原始像素面积解释需谨慎。
- 高度重叠窗口合法，植株级划分避免跨集合泄漏；没有拟合全数据归一化统计量。仅 RGB /255 是固定缩放，未来需拟合的标准化只能使用 train。

## C. 测试情况
| 检查 | 状态 | 运行证据 |
|---|---|---|
| Nonempty metadata | PASS | {"frames":17082,"plants":318} |
| Plant UID and file paths | PASS | {"id_errors":0,"path_errors":0} |
| RGB/mask one-to-one row uniqueness | PASS | {"frames":17082,"unique_pairs":17082} |
| Plant inventory matches frame inventory | PASS | 318 |
| Strict verified within-plant time order | PASS | {"problem_plants":[],"reliable_plants":318} |
| Projected area ratios in [0,1] | PASS | "all usable frames" |
| Illegal colors recorded and quarantined | PASS | {"quarantined_illegal_color_frames":26} |
| Phenotypes and canonical frames agree | PASS | {"phenotypes":17082} |
| Nonempty plant-level split with seed 42 | PASS | {"train":219,"val":53,"test":46} |
| Plant split IDs disjoint and assignments consistent | PASS | "train∩val = train∩test = val∩test = empty" |
| All usable images in exactly one plant split | PASS | {"assigned_images":17056} |
| Seed 42 split reproducibility | PASS | "independently recomputed assignments" |
| Quarantined images excluded from split index | PASS | 26 |
| Nonempty unique windows for horizons 2/4/8 | PASS | {"windows":44806} |
| All windows: same plant, future GT, original offsets | PASS | {} |
| All eligible windows generated exactly once | PASS | {"expected":44806,"observed":44806} |
| Full-file independent audit | PASS | {"metadata_sha256":"44695a96c5f7fa0bfc46785f988f2b1493f8592dc3da2caeab2d4a8dca219b46","leaf_instances_sha256":"d962823b216cd1f3343307fe977adf2b8a288c5e95fba9d2b86f9a733b59c03f","status":"PASS","frames_audited":17082,"masks_independently_recomputed":17082,"archive_crc_comparison_enabled":true,"archive_sha256":"608c7b07125281fb92517182c0873d1a9c87d696e2adedcb11c002173887de2f","archive_member_files":34166,"extra_archive_file_errors":[],"frame_errors":[],"invalid_color_frames":26,"invalid_color_pixels":54418,"quarantined_frames":26,"duplicate_rgb_file_groups":0,"duplicate_rgb_pixel_groups":0,"duplicate_mask_file_groups":0,"cross_plant_duplicate_rgb_groups":0} |
| Audit evidence matches current metadata | PASS | "SHA-256 of canonical frames and leaf instances" |
| Every frame and mask audited | PASS | 17082 |
| Byte-identical full pipeline rerun | PASS | [] |
| Full archive identity, extraction CRCs and provenance | PASS | {"archive_bytes":5486793580,"sha256":"608c7b07125281fb92517182c0873d1a9c87d696e2adedcb11c002173887de2f","source_mode":"hf-mirror"} |
| Official repository commit pinned | PASS | "0d9a24ff146d12cc8fca6bdfa2538d196754f456" |
| Automated pytest suite | PASS | {"tests":21,"results":{"PASS":21}} |
| Local review figures and selection coverage | PASS | {"png_files":9,"overlays":12,"growth_plants":6,"temporal_windows":6} |
| Acquisition identity independently confirms directory plant UID | PASS | {"source_acquisition_ids":318,"source_ids_shared_across_plant_uids":{},"plants_with_multiple_source_ids":[],"filename_genotype_conflict_frames":0,"filename_replicate_differs_from_directory_frames":1632,"plants_with_filename_replicate_alias":65,"policy":"Use curated genotype/replicate directory identity. Filename suffix replicate numbers can be reused across acquisitions; acquisition-prefix IDs are one-to-one with directory plant_uids."} |
| Real Dataset samples and DataLoader on CPU | PASS | [{"split":"train","horizon":2,"samples":11104,"plant_uid":"Ba4-1/rep_02","shape":[3,3,224,224],"status":"PASS"},{"split":"train","horizon":4,"samples":10668,"plant_uid":"Ba4-1/rep_02","shape":[3,3,224,224],"status":"PASS"},{"split":"train","horizon":8,"samples":9792,"plant_uid":"Ba4-1/rep_02","shape":[3,3,224,224],"status":"PASS"},{"split":"val","horizon":2,"samples":2603,"plant_uid":"Ba4-1/rep_01","shape":[3,3,224,224],"status":"PASS"},{"split":"val","horizon":4,"samples":2497,"plant_uid":"Ba4-1/rep_01","shape":[3,3,224,224],"status":"PASS"},{"split":"val","horizon":8,"samples":2285,"plant_uid":"Ba4-1/rep_01","shape":[3,3,224,224],"status":"PASS"},{"split":"test","horizon":2,"samples":2075,"plant_uid":"Ba4-1/rep_09","shape":[3,3,224,224],"status":"PASS"},{"split":"test","horizon":4,"samples":1983,"plant_uid":"Ba4-1/rep_09","shape":[3,3,224,224],"status":"PASS"},{"split":"test","horizon":8,"samples":1799,"plant_uid":"Ba4-1/rep_09","shape":[3,3,224,224],"status":"PASS"},{"dataloader_batch_shape":[2,3,3,224,224],"device":"cpu","status":"PASS"}] |

### 自动化测试逐项结果
| 测试 | 结果 | 秒 |
|---|---|---:|
| `tests.test_audit::test_independent_full_audit_checks_every_mask_and_archive_member` | PASS | 1.157 |
| `tests.test_audit::test_independent_audit_rejects_forged_phenotype` | PASS | 0.774 |
| `tests.test_audit::test_independent_audit_detects_cross_plant_duplicate_pixels` | PASS | 0.772 |
| `tests.test_audit::test_archive_extraction_rejects_traversal_and_preserves_existing_raw` | PASS | 0.004 |
| `tests.test_audit::test_full_window_validator_rejects_cross_plant_history` | PASS | 0.588 |
| `tests.test_metadata::test_metadata_scans_all_plants_and_assigns_dense_order` | PASS | 0.549 |
| `tests.test_metadata::test_missing_mask_and_orphan_mask_are_not_silently_dropped` | PASS | 0.165 |
| `tests.test_metadata::test_corruption_and_dimension_mismatch_are_reported` | PASS | 0.187 |
| `tests.test_metadata::test_invalid_timestamp_quarantines_unverified_sequence` | PASS | 0.165 |
| `tests.test_metadata::test_duplicate_timestamps_are_an_error` | PASS | 0.160 |
| `tests.test_phenotypes::test_exact_palette_labels_and_area_ratio` | PASS | 0.571 |
| `tests.test_phenotypes::test_invalid_mask_color_is_recorded_without_being_counted_as_a_leaf` | PASS | 0.002 |
| `tests.test_phenotypes::test_workbook_string_rgb_layout_matches_official_palette` | PASS | 0.009 |
| `tests.test_phenotypes::test_phenotype_pipeline_is_byte_reproducible` | PASS | 0.692 |
| `tests.test_phenotypes::test_exact_instances_match_official_extractor` | PASS | 0.018 |
| `tests.test_splits::test_plant_level_split_is_disjoint_and_reproducible` | PASS | 0.825 |
| `tests.test_temporal_dataset::test_temporal_dataset_smoke_returns_history_tensor_and_future_target` | PASS | 0.583 |
| `tests.test_temporal_dataset::test_quarantined_frames_do_not_compress_observation_horizons` | PASS | 0.734 |
| `tests.test_temporal_dataset::test_getitem_never_opens_future_rgb_or_mask` | PASS | 0.555 |
| `tests.test_temporal_dataset::test_dataloader_batches_resized_history_images` | PASS | 0.558 |
| `tests.test_temporal_dataset::test_forged_cross_plant_window_is_rejected` | PASS | 0.563 |

### 额外数据风险统计
- 图像尺寸：84 种；同株尺寸变化植株：`['Col-0/rep_32', 'Cvi-0/rep_25', 'Kz-9/rep_08', 'Ler-1/rep_28', 'Uk-1/rep_13']`。使用 `image_size=(224,224)` 批处理；跨裁剪变化的面积曲线不是物理生长证据。
- 非法像素总数：54418；非法 RGB 色数：7。
- 隔离观测索引：`{'Is-1/rep_24': [105, 106, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]}`。
- 同株相邻观测时间间隔：`{'min_seconds': 10789.0, 'median_seconds': 21600.0, 'max_seconds': 431999.0}`；因此观测偏移不能统一换算为天数。
- 详情：`geometry_time_and_label_statistics.json`（尺寸变化逐事件、按集合叶数/阶段分布、预测跨度实际秒数）。
- 隔离对窗口的影响：`{'potential_windows_without_quality_exclusion': 44886, 'actual_windows': 44806, 'excluded_windows': 80, 'excluded_by_horizon': {'2': 28, '4': 26, '8': 26}}`。
- 文件身份编码交叉核验：采集前缀 ID 数 318，跨目录复用采集 ID `{}`；有 1632 帧/65 株的文件尾部重复编号与目录编号不同。原目录为权威 plant_uid，采集前缀与目录一一对应；不按文件尾号合并不同植株。逐株证据见 `../../data/metadata/plant_identity_audit.csv`。

一条命令重新执行全部验收（包含全量审计、测试和真实 Dataset smoke）：
```bash
.venv/bin/python scripts/run_m0_acceptance.py
```

## D. 可视化与文件
- CSV：`../../data/metadata/frames.csv`、`plants.csv`、`phenotypes.csv`、`leaf_instances.csv`、`data_issues.csv`、`file_manifest.csv`。
- 索引：`../../splits/phenoleaf_plant_split_seed42.json`、`../../data/processed/temporal_windows.csv`。
- 统计：`dataset_statistics.json`、`phenotype_statistics.json`、`split_statistics.csv`、`temporal_window_statistics.json`。
- 可视化：`figures/` 的 9 张图；`figures/selection_manifest.json` 记录实际选取的植株、阶段、异常及窗口。
- 完整机器证据：`download_report.json`、`full_dataset_audit.json`、`validation_results.json`、`dataset_smoke.json`、`test_results.xml/json`。
- 数据来源、许可与网络替代证据：README 与 `SOURCE_AND_LICENSE.md`。所有 RGB/mask 相关可视化仅留本地，未上传第三方视觉服务或公开代码仓库。

## E. 未解决问题与限制
- 隔离帧的非法调色板颜色未获官方修正；保留原标注，不猜测这些颜色对应哪个叶片。具体位置、像素数量和颜色保存在 data_issues.csv。
- 描述范围外叶数与身份间歇出现并不证明错误；保留预警，需要人工解释植物遮挡、衰老或标注一致性。
- 时间仅验证文件编码与顺序，无时区/设备时钟证据；不构建按天跨度任务。
- 二维投影像素面积依赖裁剪尺度，不当作物理面积或 LAI；RGB resize 仅在加载时进行，不改 GT 面积。
- 本交付提供完整人工验收材料；不代替用户最终人工验收，不声称全部叶片边界已人工审核。
- 官方站直连下载遇到网络问题，使用用户允许的镜像替代；归档 SHA-256 对照固定版本 Hugging Face LFS 对象，不能将镜像域名当作官方域名。

## F. 是否具备进入 M1 的条件
**YES** — 全部验收条件以 validation_results.json 为准；只在真实数据、全量审计、无泄漏索引、Dataset smoke 与测试通过时允许进入 M1。未启动 DINOv2、GRU、JEPA 或任何训练。

## 数据使用许可
- 数据集：https://huggingface.co/datasets/rick77a/PhenoLeaf_TS
- 官方定义与预处理：https://github.com/pisyntor/PhenoLeaf-TS
- La Trobe University Use Terms：仅非商业学术/教学，必须引用；不得未经许可再分发原图/标注或公开托管。工具代码 Apache-2.0 不赋予数据再分发权限。
