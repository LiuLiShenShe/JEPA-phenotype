# M1.5 来源与检索记录

- 主要实验事实来自只读本地 M0/M1 元数据、缓存、checkpoint、预测和报告；所有数值由 M1.5 脚本计算，不来自网络。
- 项目地址（用户提供）：https://github.com/LiuLiShenShe/JEPA-phenotype
- 2026-10-11 博查检索调用返回 HTTP 403，搜索不可用；未将训练数据记忆当成已检索来源。
- 使用当前 Context7 CLI 先解析 `/pytorch/pytorch`，再读取精确版本 `/pytorch/pytorch/v2_8_0` 文档；不使用较新版本替代运行环境。
- PyTorch 2.8 GRU：https://docs.pytorch.org/docs/2.8/generated/torch.nn.GRU.html
- PyTorch 2.8 reproducibility：https://docs.pytorch.org/docs/2.8/notes/randomness.html
- PyTorch 2.8 AdamW：https://docs.pytorch.org/docs/2.8/generated/torch.optim.AdamW.html
- Matplotlib 官方文档经 Context7 `/websites/matplotlib_stable` 查询，返回原文保存在 matplotlib_docs.txt：https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html 和 https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html 。本地实际运行版本为 3.10.6。
- DINOv2 官方源码：https://github.com/facebookresearch/dinov2 （仅复用已验证 M1 冻结缓存，不更新编码器）。
- 重复测量的独立单位参考：Lazic, *The problem of pseudoreplication in neuroscientific studies*, https://pmc.ncbi.nlm.nih.gov/articles/PMC2817684/ 。本实验的重采样单位固定为植物而非窗口。
- 用户引用的 `/home/test/.codex/RTK.md` 在本机不存在；其内容未假定或编造。
- Git 当前 commit/分支由本地命令验证；`git ls-remote origin refs/heads/main` 因 GitHub:443 网络连接超时失败，因此不把本地 origin/main 引用当作远端实时验证。
