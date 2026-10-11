# Figure visual QA

图片内容由 Agnes API 识别。仅提交生成的量化图，不提交原始 RGB/Mask。



## 图表审查结果

### baseline_mae_leaf_count.png / baseline_mae_projected_area_ratio.png / error_vs_actual_time.png / genotype_error_distributions.png
未提供对应图片，无法审查。

---

### growth_curves_failure_cases.png
**无明显问题。** 布局为 3×2 网格，文字、轴标签、图例完整无裁切，oracle 标注（B0 Persistence / B1 Linear）清晰，时间单位"Days since first recorded observation (not sowing age)"明确。

---

### growth_curves_preselected.png
**无明显问题。** 布局为 6×2 网格，所有子图标题、轴标签完整，图例在底部共享，oracle 标注和时间单位与 failure_cases 图一致且清晰可读。

---

### horizon_comparison.png
**无明显问题。** 双图并排，文字与轴标签完整无裁切，图例在右上角清晰，oracle 标注明确。X 轴为"Original observation offset (not days)"，非时间单位但标注清晰。

---

### training_validation_losses.png
**无明显问题。** 2×3 子图网格，文字和轴标签完整。左上角子图内有共享图例（Train / Val），其余子图因样式一致无需重复图例，整体可读性良好。时间单位为 Epoch，标注清晰。

---

**总结：** 提供的 4 张图片均无裁切、重叠、空白或缺失问题，oracle 标注和时间/轮数单位均清晰可读。


## 补充审核：先前接口未处理的四张图单独重新提交



四张图逐一检查如下。

---

**1. `baseline_mae_leaf_count.png`**

- **布局**：三个子图水平并排，间距均匀，无重叠、无裁切。
- **坐标轴与文字**：Y 轴标题 `Plant-macro MAE: Visible leaf count` 完整；X 轴五组类别标签倾斜防重叠；每个子图上方均有对应说明（`+2 / +4 / +8 original observations`），位置清晰。
- **标题行**：三图共享顶部标题 `46 held-out plants | neural mean across 3 seeds | 95% plant-bootstrap CI`，完整无截断。
- **数量与比例目标**：Y 轴范围 0.0–3.5，B1\* 在 `+8` 子图中高度约 3.1，误差棒上沿接近 3.5 上限，未超出但未预留明显余量，可接受。
- **Oracle 标注**：B0\*、B1\* 带 `*` 并在底部统一注释 `*Oracle historical phenotypes`，信息完整。

**小结**：无裁切或重叠问题，标注清晰。

---

**2. `baseline_mae_projected_area_ratio.png`**

- **布局**：三子图并排，结构同图 1，无重叠裁切。
- **坐标轴与文字**：Y 轴标题 `Plant-macro MAE: Projected image-occupancy ratio`，刻度 0.00–0.07，完整。
- **数量与比例目标**：`+8` 子图中 B0\* 误差棒上沿接近 0.07 上边界（略超），存在轻微超出风险，但数据主体仍在范围内。
- **Oracle 标注**：与图 1 一致，底部 `*Oracle historical phenotypes` 注释完整。
- **数值可读性**：各柱体差异在 0.01–0.04 量级，Y 轴最小刻度 0.01，精度足够。

**小结**：整体清晰，`+8` 子图 B0\* 误差棒接近上界，轻微风险。

---

**3. `error_vs_actual_time.png`**

- **布局**：左右两个子图并排，线形图无重叠，线条颜色区分明确。
- **坐标轴与文字**：
  - X 轴：`Actual forecast wall-clock interval (days; 6 h bins)`，单位清晰，刻度 0–4.0 连续。
  - Y 轴左：`Within plant-macro MAE: Visible leaf count`，右：`Projected image-occupancy ratio`，各自独立，标题完整。
- **时间单位**：X 轴同时标注 `days` 和 `6 h bins`，语义明确。
- **Oracle 标注**：图例中 `B0 Persistence (oracle)` 和 `B1 Linear (oracle)` 均在图例内明确标注，无歧义。
- **关键问题**：**右图 Y 轴最大值为 0.10，但 B0 灰色线在 x=4.0 处峰值约 0.11，明显超出 Y 轴上限**，属于裁切溢出（clipping），数据点被截断，无法读取实际值。
- **图例**：右上角，五组线条颜色与图例一一对应，无重叠。

**小结**：X 轴单位和 Oracle 标注清晰；右图 Y 轴截断了 B0 线最高峰（~0.11 > 0.10），属于裁切问题，需调整 Y 轴范围。

---

**4. `genotype_error_distributions.png`**

- **布局**：上下两子图垂直排列，上为 `Visible leaf count`，下为 `Projected image-occupancy ratio`，结构整洁，无重叠裁切。
- **坐标轴与文字**：
  - Y 轴标题完整（`+4 plant MAE: Visible leaf count` / `+4 plant MAE: Projected image-occupancy ratio`）。
  - X 轴为基因型标签（如 `Ba4-1`、`Col-0` 等），角度旋转 45°，标签间无重叠，末尾标签 `Zdr-1` 完整显示。
- **数量与比例目标**：上下两图 Y 轴范围独立，上 0.5–3.0，下 0.005–0.035，各自匹配数据量级。
- **Oracle 标注**：此图为 genotype 级别分布，不涉及 oracle 方法，无需标注，正确。
- **图例**：右上角，`B2a DINO + MLP`（蓝）/ `B2b DINO + GRU`（橙）清晰，无遮挡。
- **箱线图**：箱体、须线、离群点（黑点）均完整显示，无超出绘图区域情况。

**小结**：无裁切、重叠或缺失问题，布局清晰，Y 轴比例适当。

---

**总结表**

| 图片 | 布局裁切/重叠 | Y 轴溢出 | Oracle 标注 | 时间单位 | 结论 |
|---|---|---|---|---|---|
| baseline_mae_leaf_count.png | 无 | 无 | ✓ | — | ✅ 通过 |
| baseline_mae_projected_area_ratio.png | 无 | 轻微（接近上界） | ✓ | — | ⚠️ 接近溢出，可接受 |
| error_vs_actual_time.png | 无 | **有（B0 右图）** | ✓（图例） | ✓ | ❌ 需修复右图 Y 轴上限 |
| genotype_error_distributions.png | 无 | 无 | 不适用 | 不适用 | ✅ 通过 |


## 上界余量更新后复审

程序化检查所有 line/errorbar 值均在轴内（figure_rendered_qa.json）。首轮 API 提示不等于程序证实原先裁切。更新为18%显式余量后：



**baseline_mae_projected_area_ratio.png**
- 完整性：文字、图例、数据点和误差棒均完整。
- 重叠/裁切：无明显重叠或裁切。
- 备注：Y轴显示到0.08，未见18%上界余量。

**error_vs_actual_time.png**
- 完整性：文字、图例、数据点和误差棒均完整。
- 重叠/裁切：无明显重叠或裁切。右侧子图中所有线条聚集在0附近，但清晰可见。
