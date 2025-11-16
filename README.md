# sym_search

`sym_find.py` 构建了一个针对二维材料的“对称排布搜索”流程：
给定一份结构文件（示例 `GaSb_39.vasp`），程序会统计其中的中心/邻居元素键长，
自动识别其空间群，然后在其他空间群中利用 pyxtal 采样、裁剪、压厚、缩放，
筛选出满足几何约束的候选结构。

## 快速上手
1. 安装依赖：`pip install pymatgen pyxtal`。
2. 准备结构文件（本仓库提供了 `GaSb_39.vasp` 示例，也可以改为自己的 POSCAR/vasp5 文件）。
3. 根据需求编辑 `config.json`，然后运行 `python sym_find.py config.json`。
   输出会写入 `settings.outdir` 所指定的目录（默认 `out/`）。

`config.json` 中的大部分字段都与几何约束相关（配位数、厚度、真空层、键长缩放等）。
这里额外强调一下目标空间群的配置：

### target_sgs / exclude_sgs
- `target_sgs`：可选，指定要搜索的空间群。支持整数、列表或区间字符串
  （例如 `"25-60,191-194"`）。不填则默认遍历 1..230 的所有空间群。
- `exclude_sgs`：可选，用同样的语法表示要排除的空间群编号。

程序会先用 `SpacegroupAnalyzer` 自动识别输入结构的空间群，
若 `target_sgs` 未指定，则默认排除这个空间群，因此会尝试“其他 229 个空间群”。
如果仍需排除额外空间群，只需在 `exclude_sgs` 中补充即可。

示例片段：
```json
{
  "settings": {
    "structure_file": "GaSb_39.vasp",
    "outdir": "out_gaas_sg26",
    "center_species": "Ga",
    "neighbor_species": "Sb",
    "target_sgs": "25-40",
    "exclude_sgs": [39]
  }
}
```

将 `target_sgs` 留空即可恢复“遍历 1..230，排除输入结构所在 SG” 的默认行为。

### 几何过滤 / 调参要点
最新版脚本把候选筛选流程拆成“生成 → 预处理 → 几何过滤 → 评分”四层：

1. **预处理 (`CandidatePreprocessor`)**：套入真空层、压缩厚度，控制表面起伏，并按 `bond_target_min/max` 自动缩放 a/b。若 `enable_bond_scaling=false`，也会记录原始键长与最短原子间距。
2. **几何过滤 (`GeometryFilter`)**：
   - 依据元素对的最小距离矩阵/原子半径进行硬球检查，可选 `post_gen_relax` 做快速硬球松弛。
   - 计算层密度、motif 包络（基于等价原子 & 硬球半径），若 `reject_if_overlap=true` 则直接丢弃重叠或过密结构。
3. **评分 (`CandidateEvaluator`)**：只在通过几何过滤后计算 cost、配位数，并用 spglib 验证空间群，最后去重、写出结构。

可以在 `config.json` 中调节下列关键参数：

- `min_pair_dist_matrix`：元素对最小距离矩阵。既支持 `{"Ga-Sb": 2.3}` 这样的扁平键值，也支持嵌套写法：
  ```json
  "min_pair_dist_matrix": {
    "Ga": {"Ga": 2.6, "Sb": 2.3},
    "Sb": {"Sb": 2.6}
  }
  ```
- `min_bond_length_factor` 与 `hard_sphere_radius_scale`：基于共价半径估算默认阈值，通常保持 0.85–1.0 即可。
- `reject_if_overlap`：若设为 `false`，违规结构会进入 `post_gen_relax` 处理，只有修复失败才会被丢弃。
- `density_range`：限制候选的总体密度（单位 g/cm³）。
- `post_gen_relax`：可设置 `{"mode": "hard_sphere", "max_iter": 40, "step": 0.5}`，在几何过滤阶段对轻微重叠的原子进行硬球松弛。
- `debug_save_all_cands` / `debug_save_rejected`：现在会按照“raw/prepped/geom_fail/eval_fail/accepted”分阶段把结构保存在 `SG_xxx/debug/` 目录，方便定位问题。

借助这些参数，可以把 `min_pair_dist` 的“全局阈值”细分到指定元素对，并在过滤阶段加入 motif 重叠与密度控制，更贴近 RG2/CALYPSO 等结构搜索程序中的硬球 + 修复流程。
