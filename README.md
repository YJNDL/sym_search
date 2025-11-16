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

1. **预处理 (`CandidatePreprocessor`)**：套入真空层、压缩厚度，控制表面起伏，并按 `bond_target_min/max` 自动缩放 a/b。缩放后的结构会立刻套用 `min_pair_dist_matrix`/硬球半径，逐一检查所有元素对（Ga–Ga、Sb–Sb 等）的最小距离，若违反阈值直接在该阶段淘汰；若 `enable_bond_scaling=false`，仍会记录原始键长与最短原子间距。
   - 若在 `settings.motif_checker` 中开启了 `MotifReasonablenessChecker`，每个指定元素都会用 CrystalNN 找最近邻，再用 `LocalStructOrderParams` 计算诸如 `"tet"`/`"oct"` 等 motif 的局部有序参数。只要任意 motif 的分数超过 `threshold` 就视为“合理排布”，否则在预处理阶段直接拒收。
2. **几何过滤 (`GeometryFilter`)**：
   - 依据元素对的最小距离矩阵/原子半径进行硬球检查，可选 `post_gen_relax` 做快速硬球松弛。
   - 计算层密度、motif 包络（基于等价原子 & 硬球半径），若 `reject_if_overlap=true` 则直接丢弃重叠或过密结构。
3. **评分 (`CandidateEvaluator`)**：只在通过几何过滤后计算 cost、配位数，并用 spglib 验证空间群，最后去重、写出结构。

### 2D slab 几何控制
为了确保候选结构始终是“二维薄片 + 真空层”，脚本新增了专门的 slab 控制器：

- `vacuum_thickness` / `vacuum_buffer`：把 `layer_axis` 对应的晶格矢量直接拉伸到 `layer_thickness + vacuum_thickness (+ vacuum_buffer)` 的长度，`vacuum_buffer` 相当于额外的保护间隙。这样无论 pyxtal 生成怎样的 3D 晶格，都会被统一成“in-plane 周期 + 固定真空”的单轴晶胞。
- `layer_thickness`：可选的“目标层厚”，一旦设置，所有候选都会被压缩/拉伸到该绝对厚度，再在其外侧追加 `vacuum_thickness` 指定的真空。若留空，则默认保留原始厚度，只在超过 `layer_thickness_max` 时才压缩。
- `layer_thickness_max`：当 `layer_thickness` 未设置时，作为薄片厚度的上限。若希望只做重心对齐，可把该值设为 0 并单独调 `slab_center`。
- `slab_center`：薄片在真空方向上的中心位置（分数坐标，默认 0.5，对应居中真空）。
- `slab_center_tol`：控制薄片实际质心与目标 `slab_center` 的容差，默认 0.05（按轴长的 5% 计算）。也可以直接填写绝对 Å 值，一旦超出
  即视为“非 2D”结构并在预处理阶段拒收。
- `layer_axis`：指定哪条晶轴作为“法向 + 真空方向”（支持 `a/b/c` 或 `x/y/z`）。
- `reslab_after_relax`：若开启硬球松弛，松弛后的结构也会自动重新投影回薄片，以防 3D bulk 重新出现。

这些参数由 `SlabProjector` 统一处理：它会先把原始晶格在法向方向上正交化，再根据 `layer_thickness`/`layer_thickness_max` 重建薄片厚度，并在外侧追加指定的真空。`CandidatePreprocessor`、`GeometryFilter` 和 `CandidateEvaluator`（保存最终原胞前）都会调用该投影器，因此任何生成、松弛或转原胞操作之后，结构都会重新回到“二维薄片 + 真空”的几何形态，调试目录和最终输出都会反映这一点。

可以在 `config.json` 中调节下列关键参数：

- `min_pair_dist_matrix`：元素对最小距离矩阵，预处理和几何过滤都会使用。既支持 `{"Ga-Sb": 2.3}` 这样的扁平键值，也支持嵌套写法：
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
- `motif_checker`：基于 CrystalNN + `LocalStructOrderParams` 的 motif 合理性过滤。例如：
  ```json
  "motif_checker": {
    "enabled": true,
    "motifs": ["tet", "oct"],
    "threshold": 0.6,
    "species": ["Ga"],
    "crystalnn_kwargs": {"distance_cutoffs": null}
  }
  ```
  - `motifs` 列表支持 `LocalStructOrderParams` 的类型（如 `tet`, `oct`, `sq_plan` 等），脚本会对每个受检原子取分数最高的 motif，与 `threshold` 比较。
  - `species` 用于限定只检查部分元素；留空则表示全结构。
  - `threshold` 默认 0.55，可按体系松紧调节。
  - `motifs` 支持 `LocalStructOrderParams` 的所有类型（`tet`、`oct`、`tri_plan` 等），也可以写成更易读的别名（例如 `"tetrahedral"` 会自动映射为 `"tet"`，`"square planar"` 会映射为 `"sq_plan"`）。
  - 若启用该检查，debug 目录只会保存“已经通过 motif 检查”的结构（raw/prepped/eval_fail/accepted 等标签都会自动过滤）。
- `local_env_constraints`：面向元素的 CrystalNN + motif/CN 组合约束，比全局 `cn_range` 更精细。示例：
  ```json
  "local_env_constraints": {
    "enabled": true,
    "threshold": 0.7,
    "global_cn_range": [2, 6],
    "default_motifs": ["tetrahedral", "trigonal planar"],
    "cn_rules": {
      "Ga": {
        "allowed_range": [2, 6],
        "preferred_range": [3, 4],
        "preferred_motifs": ["tetrahedral", "trigonal planar"]
      },
      "Sb": {
        "allowed_range": [2, 6],
        "preferred_range": [3, 5],
        "preferred_motifs": ["trigonal pyramidal", "octahedral"]
      }
    }
  }
  ```
  - `global_cn_range` 用作未单独配置元素的兜底配位区间，也会在缺失 `cn_rules`/`elements` 时回退；如果仍沿用旧版 `cn_range`，会被自动当成该兜底窗口。
  - `default_motifs` 会在单个元素未声明 `preferred_motifs` 时提供公共 motif 列表。
  - `cn_rules` 中可以分别声明 `allowed_range`（硬性 CN 范围）与 `preferred_range`（更窄的优选窗口，如果缺失则回退到 `allowed_range`），并在需要时附加元素专属的 `preferred_motifs`。若继续使用旧版 `elements` + `cn_range` 的写法，脚本也会自动解析并与 `cn_rules` 合并，后一者具有更高优先级。
  - 若开启该模块，`CandidateEvaluator` 会在 SG 验证前对整结构运行 CrystalNN，任何元素 CN 越界或 motif 分数低于 `threshold` 的候选都会被立刻淘汰，调试信息会记录在 `meta.local_env` 中。
- `debug_save_all_cands` / `debug_save_rejected`：现在会按照“raw/prepped/geom_fail/eval_fail/accepted”分阶段把结构保存在 `SG_xxx/debug/` 目录，方便定位问题。

借助这些参数，可以把 `min_pair_dist` 的“全局阈值”细分到指定元素对，并在过滤阶段加入 motif 重叠与密度控制，更贴近 RG2/CALYPSO 等结构搜索程序中的硬球 + 修复流程。
