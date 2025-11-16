#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sym_search_2d_gaas.py  ——  POSCAR + settings.json 版本（通用元素版 + 自由晶格 + 厚度 / 表面起伏 + 键长缩放）

二维材料（起始结构来自 POSCAR）→ 其他空间群的对称排布搜索与验证。

核心逻辑：
1. 只用 POSCAR 来确定：
   - 元素种类与配比（默认中心–邻居元素对为 Ga–As，可在 config.json 中改）
   - 参考最近邻中心–邻居键长分布（用于后续 cost 打分）
2. “去掉”原始晶格约束：
   - 对每个目标空间群 SG，以及每个在 a,b 平面扩胞得到的倍数 det（HNF 的行列式）：
     - 计算需要的原子数：num_center = center_count_base * det，num_neighbor = neighbor_count_base * det
     - 调用 pyxtal.from_random(3, SG, [center, neighbor], [num_center, num_neighbor])
       让 pyxtal 自由生成一个 3D 晶格 + 原子排布
3. 对每个 pyxtal 生成的候选结构：
   - 只调 c 长度到指定真空区间（保持 a,b 与角度不变），模拟 2D slab
   - 根据 layer_thickness_max 将原子层本身在 z 方向的整体厚度压缩到给定上限
   - 用 surface_corrugation 控制表面原子的起伏（例如 MoS2 顶 / 底 S 原子在平面内的起伏）
   - 通过缩放 a,b，让中心–邻居最近邻键长落在用户指定区间（默认 2.2–2.8 Å），并检查：
       - 最短任意原子间距（避免重叠）
       - 最近邻中心–邻居键长分布是否接近参考（median 附近）
       - 中心原子的配位数 CN 是否在允许范围
   - 用 spglib 识别空间群，必须等于目标 SG
   - 转为原胞（primitive）并写出 CIF / POSCAR / meta.json
4. 每个 SG 保留 topk 个 cost 最低的候选，写 index.csv / index.json 汇总。

结构来源：
- 默认当前目录的 POSCAR（可通过 config.json 的 structure_file 字段修改路径）

参数来源：
- config.json 中的 "settings" 字段

使用方式：
1. 准备好 POSCAR（例如 SG=26 的二维 GaAs，4 Ga + 4 As）。
2. 写一个 config.json，只需要写 "settings"（见示例）。
3. 运行：python sym_search_2d_gaas.py config.json
   若不加参数，则默认读取当前目录的 config.json。
"""

import os
import sys
import json
import math
import argparse
import random
import traceback
from collections import defaultdict, Counter

import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# pyxtal（可选）
try:
    from pyxtal import pyxtal as _pyxtal
    _HAS_PYXTAL = True
except Exception:
    _HAS_PYXTAL = False


# ------------------------------ 工具：日志与IO ------------------------------

def makedirs(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def save_structure_pair(prim: Structure, dir_path: str, stem: str):
    makedirs(dir_path)
    cif_path = os.path.join(dir_path, f"{stem}.cif")
    poscar_path = os.path.join(dir_path, f"{stem}.poscar")
    CifWriter(prim).write_file(cif_path)
    Poscar(prim).write_file(poscar_path)
    return cif_path, poscar_path


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_log(path: str, line: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


# ------------------------------ 从 config.json 读取 settings ------------------------------

def load_settings(config_path: str) -> argparse.Namespace:
    """
    只读 settings 部分，结构一律从 structure_file（默认 POSCAR）读取。
    config.json 可以是：
    {
      "settings": { ... }
    }
    或直接：
    {
      "outdir": "...",
      "bond_tol": 0.4,
      ...
    }
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if "settings" in cfg and isinstance(cfg["settings"], dict):
        settings = cfg["settings"]
    else:
        settings = cfg

    # 默认值
    defaults = {
        "outdir": "out",

        "bond_tol": 0.4,
        "vacuum_c": 20.0,

        "det_max": 12,
        "max_atoms": 96,
        "enable_sqrt3": True,      # bool

        "symprec": [1e-3, 2e-3, 5e-3],
        "angle_tol": 0.5,

        "cn_target": 4,
        "cn_range": [3, 6],
        "min_pair_dist": 1.8,

        # 表面起伏上限（Å），控制顶 / 底表面原子的 z 起伏；不再直接表示整体厚度
        "z_flat_tol": 0.3,

        "ab_scale": 0.05,          # 现在不再用于 lattice，只保留接口
        "ab_scale_samples": 3,

        "samples_per_sg": 100,
        "topk": 3,
        "seed": 2025,

        # 目标原子层最大整体厚度（Å），>0 时会主动压扁 z 分布
        "layer_thickness_max": 3.0,

        # 用于最近邻统计和 CN 的元素对（可以改成任意中心/邻居元素）
        "center_species": "Ga",
        "neighbor_species": "As",

        # 结构文件路径，默认当前目录 POSCAR
        "structure_file": "POSCAR",

        # debug 相关开关
        "debug_save_all_cands": False,   # 保存所有 pyxtal 生成的候选
        "debug_save_rejected": False,    #（预留，目前未单独区分）
        "debug_max_per_sg": 50,

        # 键长目标区间（Å）
        "bond_target_min": 2.2,
        "bond_target_max": 2.8,
        "enable_bond_scaling": True,

        # 缩放模式：
        #   "strict": 所有 center–neighbor 键都必须能缩放进 [bond_target_min, bond_target_max]
        #   "soft":   大部分键进窗口即可，用 coverage 控制
        "bond_scaling_mode": "soft",
        "bond_coverage_min": 0.8,

        # cost 权重
        "w_cost_d": 1.0,
        "w_cost_cn": 0.5,

        # 用于 surface corrugation 的“表面厚度占比”（顶/底各取多少厚度参与起伏统计）
        "surface_frac": 0.25,
    }

    # 合并默认和用户设置
    params = {**defaults, **settings}

    # 规范类型
    # symprec: 列表
    if isinstance(params["symprec"], str):
        params["symprec"] = [float(x) for x in params["symprec"].split(",")]
    else:
        params["symprec"] = [float(x) for x in params["symprec"]]

    # cn_range: 两个整数
    if isinstance(params["cn_range"], str):
        lo, hi = params["cn_range"].split(",")
        params["cn_range"] = [int(lo), int(hi)]
    else:
        params["cn_range"] = [int(params["cn_range"][0]), int(params["cn_range"][1])]

    # enable_sqrt3 / enable_bond_scaling: 布尔
    for key in ("enable_sqrt3", "enable_bond_scaling"):
        if isinstance(params[key], str):
            params[key] = params[key].strip().lower() in ("1", "true", "yes", "y")

    # 数值
    params["bond_target_min"] = float(params["bond_target_min"])
    params["bond_target_max"] = float(params["bond_target_max"])
    params["min_pair_dist"] = float(params["min_pair_dist"])
    params["vacuum_c"] = float(params["vacuum_c"])
    params["z_flat_tol"] = float(params["z_flat_tol"])
    params["layer_thickness_max"] = float(params["layer_thickness_max"])
    params["w_cost_d"] = float(params["w_cost_d"])
    params["w_cost_cn"] = float(params["w_cost_cn"])
    params["surface_frac"] = float(params["surface_frac"])

    params["bond_scaling_mode"] = str(params.get("bond_scaling_mode", "soft")).lower()
    params["bond_coverage_min"] = float(params.get("bond_coverage_min", 0.8))

    return argparse.Namespace(**params)


# ------------------------------ 几何与邻接（通用元素版） ------------------------------

def ensure_vacuum_c(struct: Structure, c_target_low=18.0, c_target_high=25.0) -> Structure:
    """
    只调整 c 长度至给定范围，不改变 a,b 与角度，保持 z 分数坐标不变。
    """
    L = struct.lattice
    c = L.c
    if c_target_low <= c <= c_target_high:
        return struct.copy()
    c_new = max(min(c, c_target_high), c_target_low)
    L_new = Lattice.from_parameters(L.a, L.b, c_new, L.alpha, L.beta, L.gamma)
    return Structure(L_new, [s.species for s in struct.sites],
                     [s.frac_coords for s in struct.sites], coords_are_cartesian=False)


def min_image_vec(frac_delta):
    d = np.array(frac_delta, dtype=float)
    d -= np.round(d)
    return d


def get_species_indices(struct: Structure):
    """
    返回 {element_lower: [indices]}
    """
    idx_map = defaultdict(list)
    for i, s in enumerate(struct.sites):
        idx_map[s.species_string.lower()].append(i)
    return idx_map


def compute_nn_stats(struct: Structure,
                     center_species=None,
                     neighbor_species=None,
                     k_per_site: int = 4):
    """
    通用最近邻统计：

    - center_species: 作为中心原子的元素符号（如 "Ga"），None 表示所有原子。
    - neighbor_species: 作为邻居原子的元素符号（如 "As"），None 表示所有原子。
    - k_per_site: 对每个中心原子取最近的 k 个邻居。

    返回:
    {
      "median": 距离中位数,
      "mean":   距离均值,
      "std":    标准差,
      "per_site_knn": 展平的 k 近邻距离列表
    }
    """
    L = struct.lattice
    idx_map = get_species_indices(struct)
    n_sites = len(struct.sites)

    # 中心原子索引
    if center_species is None:
        center_idx = list(range(n_sites))
    else:
        key = center_species.lower()
        if key not in idx_map or not idx_map[key]:
            raise ValueError(f"结构中未找到元素 {center_species}")
        center_idx = idx_map[key]

    # 邻居原子索引
    if neighbor_species is None:
        neighbor_idx_all = list(range(n_sites))
    else:
        key = neighbor_species.lower()
        if key not in idx_map or not idx_map[key]:
            raise ValueError(f"结构中未找到元素 {neighbor_species}")
        neighbor_idx_all = idx_map[key]

    dists_all = []
    per_site_knn = []

    for i in center_idx:
        p = struct[i].frac_coords
        d_list = []
        for j in neighbor_idx_all:
            if j == i:
                continue
            q = struct[j].frac_coords
            df = min_image_vec(q - p)
            dc = df @ L.matrix
            d_list.append(np.linalg.norm(dc))
        if not d_list:
            continue
        d_list.sort()
        per_site_knn.extend(d_list[:k_per_site])
        dists_all.extend(d_list[:k_per_site])

    if not dists_all:
        raise ValueError("无法找到任何最近邻距离，请检查结构或 center/neighbor 元素设定。")

    dists_all = np.array(dists_all)
    stats = {
        "median": float(np.median(dists_all)),
        "mean":   float(np.mean(dists_all)),
        "std":    float(np.std(dists_all)),
        "per_site_knn": per_site_knn,
    }
    return stats


def compute_candidate_nn_distances(struct: Structure,
                                   center_species=None,
                                   neighbor_species=None,
                                   k_per_site: int = 4):
    """
    对候选结构统计「center_species–neighbor_species」最近邻距离列表。
    """
    L = struct.lattice
    idx_map = get_species_indices(struct)
    n_sites = len(struct.sites)

    if center_species is None:
        center_idx = list(range(n_sites))
    else:
        key = center_species.lower()
        if key not in idx_map or not idx_map[key]:
            raise ValueError(f"结构中未找到元素 {center_species}")
        center_idx = idx_map[key]

    if neighbor_species is None:
        neighbor_idx_all = list(range(n_sites))
    else:
        key = neighbor_species.lower()
        if key not in idx_map or not idx_map[key]:
            raise ValueError(f"结构中未找到元素 {neighbor_species}")
        neighbor_idx_all = idx_map[key]

    dists = []
    for i in center_idx:
        p = struct[i].frac_coords
        d_list = []
        for j in neighbor_idx_all:
            if j == i:
                continue
            q = struct[j].frac_coords
            df = min_image_vec(q - p)
            dc = df @ L.matrix
            d_list.append(np.linalg.norm(dc))
        if not d_list:
            continue
        d_list.sort()
        dists.extend(d_list[:k_per_site])

    return np.array(dists)


def count_cn_like_generic(struct: Structure,
                          d_ref: float,
                          band: float,
                          center_species=None,
                          neighbor_species=None):
    """
    通用配位数统计：
    以 [d_ref - band, d_ref + band] 为“成键窗口”，统计每个中心原子的邻居个数。
    """
    L = struct.lattice
    idx_map = get_species_indices(struct)
    n_sites = len(struct.sites)

    if center_species is None:
        center_idx = list(range(n_sites))
    else:
        key = center_species.lower()
        if key not in idx_map or not idx_map[key]:
            raise ValueError(f"结构中未找到元素 {center_species}")
        center_idx = idx_map[key]

    if neighbor_species is None:
        neighbor_idx_all = list(range(n_sites))
    else:
        key = neighbor_species.lower()
        if key not in idx_map or not idx_map[key]:
            raise ValueError(f"结构中未找到元素 {neighbor_species}")
        neighbor_idx_all = idx_map[key]

    lo, hi = max(0.0, d_ref - band), d_ref + band
    CNs = []
    for i in center_idx:
        p = struct[i].frac_coords
        c = 0
        for j in neighbor_idx_all:
            if j == i:
                continue
            q = struct[j].frac_coords
            df = min_image_vec(q - p)
            d = np.linalg.norm(df @ L.matrix)
            if lo <= d <= hi:
                c += 1
        CNs.append(c)
    return CNs


def min_any_pair_distance(struct: Structure) -> float:
    """
    任意两原子（所有元素、所有配对）的最近距离。
    """
    L = struct.lattice
    frac = np.array([s.frac_coords for s in struct.sites])
    n = len(struct.sites)
    dmin = 1e9
    for i in range(n):
        for j in range(i + 1, n):
            df = min_image_vec(frac[j] - frac[i])
            d = np.linalg.norm(df @ L.matrix)
            if d < dmin:
                dmin = d
    return float(dmin)


def layer_thickness_angstrom(struct: Structure) -> float:
    """
    整体层厚度：z 方向上 max(z) - min(z)（Å）。
    """
    c = struct.lattice.c
    zs = np.array([s.frac_coords[2] * c for s in struct.sites])
    return float(np.ptp(zs))


def surface_corrugation_angstrom(struct: Structure, surface_frac: float = 0.25):
    """
    估算“表面起伏”：
    - 先取所有原子的 z_cart，得到总厚度 thickness。
    - 顶部表面：z >= z_max - surface_frac * thickness
    - 底部表面：z <= z_min + surface_frac * thickness
    - 分别计算两个表面的 ptp，取最大值作为表面起伏。

    返回 (corrugation, thickness)，单位 Å。
    """
    L = struct.lattice
    c = L.c
    z_cart = np.array([s.frac_coords[2] * c for s in struct.sites])
    z_min = float(z_cart.min())
    z_max = float(z_cart.max())
    thickness = z_max - z_min
    if thickness < 1e-6:
        return 0.0, thickness

    window = max(1e-3, surface_frac * thickness)
    top_mask = z_cart >= (z_max - window)
    bot_mask = z_cart <= (z_min + window)

    corr_top = float(np.ptp(z_cart[top_mask])) if np.any(top_mask) else 0.0
    corr_bot = float(np.ptp(z_cart[bot_mask])) if np.any(bot_mask) else 0.0

    return max(corr_top, corr_bot), thickness


def squash_layer_thickness(struct: Structure, max_thickness: float) -> Structure:
    """
    将结构中原子在 z 方向的整体厚度压缩到不超过 max_thickness（Å）。
    只改变原子 z 分数坐标，不改变 a,b,c 和角度。
    如果 max_thickness<=0 或者原本就比它薄，则直接返回原结构。
    """
    if max_thickness is None or max_thickness <= 0:
        return struct

    L = struct.lattice
    c = L.c
    z_cart = np.array([s.frac_coords[2] * c for s in struct.sites])
    z_min = float(z_cart.min())
    z_max = float(z_cart.max())
    thickness = z_max - z_min

    if thickness < 1e-6 or thickness <= max_thickness:
        return struct

    z_mid = 0.5 * (z_max + z_min)
    scale = max_thickness / thickness

    new_frac = []
    for s in struct.sites:
        fc = np.array(s.frac_coords, dtype=float)
        zc = fc[2] * c
        zc_new = (zc - z_mid) * scale + z_mid
        fc[2] = zc_new / c
        new_frac.append(fc)

    return Structure(
        L,
        [s.species for s in struct.sites],
        new_frac,
        coords_are_cartesian=False
    )


def candidate_cost(struct: Structure,
                   ref_stats: dict,
                   bond_tol: float,
                   cn_target: int,
                   w_d: float,
                   w_cn: float,
                   center_species=None,
                   neighbor_species=None):
    """
    通用 cost：
    cost = w_d * MAD(候选最近邻 - d_ref) / bond_tol + w_cn * avg |CN - cn_target|
    center_species / neighbor_species 控制用哪一对元素来定义“键”。
    """
    d_ref = ref_stats["median"]
    cand_d = compute_candidate_nn_distances(
        struct,
        center_species=center_species,
        neighbor_species=neighbor_species,
        k_per_site=4,
    )
    if len(cand_d) == 0:
        return 1e9, {
            "mad": 1e9, "cn_avg_abs": 1e9, "cn_hist": {},
            "cand_d_min": None, "cand_d_max": None, "d_ref": d_ref,
            "CNs": [],
        }

    mad = float(np.mean(np.abs(cand_d - d_ref)))
    CNs = count_cn_like_generic(
        struct,
        d_ref=d_ref,
        band=bond_tol,
        center_species=center_species,
        neighbor_species=neighbor_species,
    )
    cn_avg_abs = float(np.mean([abs(cn - cn_target) for cn in CNs])) if CNs else 1e9
    cost = w_d * (mad / max(bond_tol, 1e-6)) + w_cn * cn_avg_abs
    cn_hist = Counter(CNs)
    stats = {
        "mad": mad,
        "cn_avg_abs": cn_avg_abs,
        "cn_hist": dict(cn_hist),
        "cand_d_min": float(np.min(cand_d)),
        "cand_d_max": float(np.max(cand_d)),
        "d_ref": d_ref,
        "CNs": CNs,
    }
    return cost, stats


# ------------------------------ 超胞（2D HNF + √3，用于 det 列表） ------------------------------

def divisors(n: int):
    out = []
    for k in range(1, int(math.sqrt(n)) + 1):
        if n % k == 0:
            out.append(k)
            if k * k != n:
                out.append(n // k)
    return sorted(out)


def enumerate_hnf_2d(det_max: int):
    """
    2D Hermite Normal Form（唯一枚举所有超胞）:
      H = [[a, 0],
           [b, c]]
    其中 a*c = det, 0<= b < c
    嵌入到 3x3，c 方向为 1。
    """
    mats = []
    for d in range(1, det_max + 1):
        for a in divisors(d):
            c = d // a
            for b in range(c):
                M = np.eye(3, dtype=int)
                M[0, 0] = a
                M[0, 1] = 0
                M[1, 0] = b
                M[1, 1] = c
                mats.append(M)
    return mats


def is_hex_like_lattice(latt: Lattice, len_tol=0.03, ang_tol=3.0) -> bool:
    return (abs(latt.a - latt.b) / max(latt.a, latt.b) <= len_tol) and (abs(latt.gamma - 120.0) <= ang_tol)


def sqrt3_r30_mats():
    """
    常见 √3×√3 R30° 的 2x2 整系数变换（det=3），适配六角/近六角格子。
    """
    cand = [
        np.array([[1, 1, 0],
                  [-1, 2, 0],
                  [0, 0, 1]], dtype=int),
        np.array([[2, -1, 0],
                  [1, 1, 0],
                  [0, 0, 1]], dtype=int),
        np.array([[1, 2, 0],
                  [-2, 1, 0],
                  [0, 0, 1]], dtype=int),
        np.array([[1, -2, 0],
                  [2, 1, 0],
                  [0, 0, 1]], dtype=int),
    ]
    return cand


def generate_allowed_supercell_dets(latt: Lattice, det_max=12, enable_sqrt3=True):
    """
    只返回允许的行列式 det 列表（不再使用具体 HNF 形状），
    代表在 a,b 平面扩胞时允许的“面积倍数”。
    """
    mats = enumerate_hnf_2d(det_max)
    if enable_sqrt3 and is_hex_like_lattice(latt):
        mats += sqrt3_r30_mats()
    dets = set()
    for M in mats:
        det2 = int(round(np.linalg.det(M[:2, :2])))
        if det2 > 0:
            dets.add(det2)
    return sorted(dets)


# ------------------------------ pyxtal 候选生成（自由晶格） ------------------------------

def pyxtal_generate_candidates(sg: int,
                               num_center: int,
                               num_neighbor: int,
                               center_species: str,
                               neighbor_species: str,
                               trials: int):
    """
    使用 pyxtal 在“指定空间群 + 指定元素配比”下自由生成晶格 + 原子排布。
    """
    if not _HAS_PYXTAL:
        return []
    cand = []
    for _ in range(trials):
        try:
            xtl = _pyxtal()
            xtl.from_random(
                3,
                sg,
                [center_species, neighbor_species],
                [num_center, num_neighbor],
            )
            st = xtl.to_pymatgen()
            cand.append(st)
        except Exception:
            continue
    return cand


# ------------------------------ SG 验证与原胞化 ------------------------------

def verify_spacegroup_exact(struct: Structure, target_sg: int, symprecs, angle_tolerance: float):
    """
    在多组 symprec 下尝试识别，若任何一组得到的 SG == 目标则通过。
    返回 (bool, best_sg, best_symprec)
    """
    best = None
    for sp in symprecs:
        try:
            sga = SpacegroupAnalyzer(struct, symprec=sp, angle_tolerance=angle_tolerance)
            sgnum = sga.get_space_group_number()
            if sgnum == target_sg:
                return True, sgnum, sp
            if best is None:
                best = (sgnum, sp)
        except Exception:
            continue
    return False, (best[0] if best else None), (best[1] if best else None)


def to_primitive(struct: Structure, symprec=1e-3, angle_tolerance=0.5) -> Structure:
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
        prim = sga.get_primitive_standard_structure()
        return prim
    except Exception:
        return struct.copy()


def wyckoff_map_by_species(struct: Structure, symprec=1e-3, angle_tolerance=0.5):
    """
    返回 {element: {wyckoff_letter: count}}，基于 spglib 的 dataset。
    """
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
        ds = sga.get_symmetry_dataset()
        letters = ds.get("wyckoffs", [])
        mp = defaultdict(lambda: defaultdict(int))
        for site, w in zip(struct.sites, letters):
            elem = site.species_string
            mp[elem][w] += 1
        return {e: dict(mp[e]) for e in mp}
    except Exception:
        return {}


# ------------------------------ 主流程：POSCAR + settings.json ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="二维材料对称排布搜索（结构从 POSCAR 读取，参数从 config.json 读取，元素对可配置，自由晶格 + 压厚 + 表面起伏 + 键长缩放）"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.json",
        help="settings JSON 文件路径（默认 config.json）"
    )
    args_cli = parser.parse_args()
    config_path = args_cli.config

    # 读取 settings
    args = load_settings(config_path)

    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = makedirs(args.outdir)
    log_dir = makedirs(os.path.join(outdir, "logs"))
    index_csv = os.path.join(outdir, "index.csv")
    index_json = os.path.join(outdir, "index.json")

    # 读入结构（默认 POSCAR）
    if not os.path.exists(args.structure_file):
        raise FileNotFoundError(f"结构文件不存在: {args.structure_file}")
    base = Structure.from_file(args.structure_file)
    base = ensure_vacuum_c(base, c_target_low=args.vacuum_c - 1.0, c_target_high=args.vacuum_c + 1.0)

    # 用用户指定的元素对做参考最近邻统计
    ref_stats = compute_nn_stats(
        base,
        center_species=args.center_species,
        neighbor_species=args.neighbor_species,
        k_per_site=4
    )
    print(
        f"[REF] {args.center_species}-{args.neighbor_species} 最近邻: "
        f"median={ref_stats['median']:.3f} Å  "
        f"mean={ref_stats['mean']:.3f} Å  std={ref_stats['std']:.3f} Å"
    )

    # 目标 SG 列表（示例：仅 26，可自行改为 1..230）
    target_sgs = [sg for sg in range(26, 27) if sg != 39]

    symprecs = tuple(float(x) for x in args.symprec)
    cn_lo, cn_hi = int(args.cn_range[0]), int(args.cn_range[1])

    # 基本信息：基胞里的元素计数（只算一次就行）
    idx_map_base = get_species_indices(base)
    center_key = args.center_species.lower()
    neighbor_key = args.neighbor_species.lower()
    if center_key not in idx_map_base or neighbor_key not in idx_map_base:
        raise ValueError(
            f"在基胞中找不到 center/neighbor 元素：{args.center_species}, {args.neighbor_species}；"
            f"现有元素为：{list(idx_map_base.keys())}"
        )
    center_count_base = len(idx_map_base[center_key])
    neighbor_count_base = len(idx_map_base[neighbor_key])
    base_n = len(base.sites)

    # 生成允许的“扩胞倍数 det”列表（代表在 a,b 平面扩胞的面积倍数）
    det_list = generate_allowed_supercell_dets(
        base.lattice,
        det_max=args.det_max,
        enable_sqrt3=bool(args.enable_sqrt3)
    )

    index_rows = []
    matchers = {}

    total_targets = len(target_sgs)
    print(
        f"[INFO] 将尝试空间群 {total_targets} 个，"
        f"扩胞倍数 det 列表：{det_list}；pyxtal 可用：{_HAS_PYXTAL}"
    )

    for idx, sg in enumerate(target_sgs, 1):
        sg_dir = makedirs(os.path.join(outdir, f"SG_{sg:03d}"))
        log_path = os.path.join(log_dir, f"SG_{sg:03d}.log")
        found = []

        debug_dir = makedirs(os.path.join(sg_dir, "debug"))
        debug_counter = 0

        def debug_save(struct: Structure, det2: int, scale_ab: float, tag: str):
            nonlocal debug_counter
            if not (args.debug_save_all_cands or args.debug_save_rejected):
                return
            if debug_counter >= args.debug_max_per_sg:
                return
            stem = f"cand_debug_det{det2}_scale{scale_ab:.4f}_{tag}_{debug_counter:03d}"
            save_structure_pair(struct, debug_dir, stem)
            debug_counter += 1

        if sg not in matchers:
            matchers[sg] = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5.0,
                                            primitive_cell=True, scale=True)

        for det2 in det_list:
            n_atoms = base_n * det2
            if n_atoms > args.max_atoms:
                continue

            num_center = center_count_base * det2
            num_neighbor = neighbor_count_base * det2

            if not _HAS_PYXTAL:
                append_log(log_path, f"[SKIP] det={det2}  未安装 pyxtal，无法枚举 Wyckoff")
                continue

            cands = pyxtal_generate_candidates(
                sg,
                num_center=num_center,
                num_neighbor=num_neighbor,
                center_species=args.center_species,
                neighbor_species=args.neighbor_species,
                trials=args.samples_per_sg
            )
            if not cands:
                append_log(log_path, f"[FAIL] det={det2}  pyxtal 采样均失败")
                continue

            for st in cands:
                # pyxtal 自由生成的 3D 晶格 + 原子排布
                # 1) 先“套上二维真空”：只改 c，不改 a,b 和分数坐标
                st = ensure_vacuum_c(
                    st,
                    c_target_low=args.vacuum_c - 1.0,
                    c_target_high=args.vacuum_c + 1.0
                )
                # 2) 再把整体厚度压缩到 layer_thickness_max 以内（若该值>0）
                st = squash_layer_thickness(st, args.layer_thickness_max)

                # 初始缩放因子（记录到 meta 里用）
                scale_ab = 1.0

                if args.debug_save_all_cands:
                    debug_save(st, det2, scale_ab, "raw")

                # --- 厚度与表面起伏约束 ---
                surf_corr, thickness = surface_corrugation_angstrom(st, surface_frac=args.surface_frac)
                if args.layer_thickness_max > 0 and thickness > args.layer_thickness_max + 1e-3:
                    append_log(
                        log_path,
                        f"[REJ] det={det2}  thickness={thickness:.3f}Å > layer_thickness_max={args.layer_thickness_max}Å"
                    )
                    continue
                if surf_corr > args.z_flat_tol:
                    append_log(
                        log_path,
                        f"[REJ] det={det2}  surface_corr={surf_corr:.3f}Å > z_flat_tol={args.z_flat_tol}Å"
                    )
                    continue

                # --- 计算中心–邻居最近邻距离 & 全局最短距离 ---
                try:
                    cand_d = compute_candidate_nn_distances(
                        st,
                        center_species=args.center_species,
                        neighbor_species=args.neighbor_species,
                        k_per_site=4,
                    )
                except ValueError as e:
                    append_log(log_path, f"[REJ] det={det2}  compute_candidate_nn_distances 失败: {e}")
                    continue

                if cand_d.size == 0:
                    append_log(log_path, f"[REJ] det={det2}  无法获得任何 {args.center_species}-{args.neighbor_species} 最近邻键长")
                    continue

                cand_min = float(np.min(cand_d))
                cand_max = float(np.max(cand_d))
                phys_min = min_any_pair_distance(st)

                bond_min = args.bond_target_min
                bond_max = args.bond_target_max

                # --- 键长缩放逻辑（只缩放 a,b，不动 c；开关由 enable_bond_scaling 控制） ---
                if args.enable_bond_scaling:
                    if cand_min <= 1e-6 or cand_max <= 1e-6:
                        append_log(
                            log_path,
                            f"[REJ] det={det2}  cand_min/cand_max 非法: {cand_min:.4f}/{cand_max:.4f}"
                        )
                        continue

                    mode = getattr(args, "bond_scaling_mode", "strict").lower()

                    if mode == "strict":
                        # === 原来的严格模式：必须存在统一 s，使所有键都落在 [bond_min, bond_max] ===
                        s_lo = bond_min / cand_min
                        s_hi = bond_max / cand_max

                        if s_lo > s_hi:
                            # 区间无交集，无法用同一个缩放因子同时满足上下限
                            append_log(
                                log_path,
                                f"[REJ] det={det2}  无法缩放到 [{bond_min:.2f}, {bond_max:.2f}]Å，"
                                f"原始键长范围 {cand_min:.2f}–{cand_max:.2f}Å (strict)"
                            )
                            continue

                        # 在可行区间 [s_lo, s_hi] 内选一个离 1 最近的缩放因子
                        if s_lo <= 1.0 <= s_hi:
                            s_target = 1.0
                        else:
                            s_target = s_lo if abs(s_lo - 1.0) < abs(s_hi - 1.0) else s_hi

                        # 限制缩放系数
                        s_target = max(0.5, min(s_target, 4.0))

                    else:
                        # === soft 模式：让中位数靠近区间中心，并保证最短键有机会被拉到 >= bond_min ===
                        d_med = float(np.median(cand_d))
                        if d_med <= 1e-6:
                            append_log(
                                log_path,
                                f"[REJ] det={det2}  soft 模式: d_med 非法 {d_med:.4f}"
                            )
                            continue

                        target_mid = 0.5 * (bond_min + bond_max)
                        s_target = target_mid / d_med

                        # 同时考虑把最短键拉到 >= bond_min
                        s_min_for_shortest = bond_min / cand_min
                        if s_target < s_min_for_shortest:
                            s_target = s_min_for_shortest

                        # 限制缩放系数
                        s_target = max(0.5, min(s_target, 4.0))

                    # === 应用 a,b 缩放（strict / soft 共用） ===
                    L_old = st.lattice
                    mat = L_old.matrix.copy()
                    mat[0, :] *= s_target
                    mat[1, :] *= s_target
                    st = Structure(
                        Lattice(mat),
                        [s.species for s in st.sites],
                        [s.frac_coords for s in st.sites],
                        coords_are_cartesian=False
                    )
                    scale_ab = s_target

                    # 缩放后重新计算键长
                    cand_d = compute_candidate_nn_distances(
                        st,
                        center_species=args.center_species,
                        neighbor_species=args.neighbor_species,
                        k_per_site=4,
                    )
                    cand_min = float(np.min(cand_d))
                    cand_max = float(np.max(cand_d))
                    phys_min = min_any_pair_distance(st)

                    if mode == "strict":
                        # 严格：所有 center–neighbor 最近邻都必须在 [bond_min, bond_max] 内
                        if cand_min < bond_min - 1e-3 or cand_max > bond_max + 1e-3:
                            append_log(
                                log_path,
                                f"[REJ] det={det2}  strict 模式: 缩放后键长范围 "
                                f"{cand_min:.2f}–{cand_max:.2f}Å 不在 [{bond_min:.2f}, {bond_max:.2f}]Å 内"
                            )
                            continue
                        # 最短任意原子间距也要 >= bond_min
                        if phys_min < bond_min - 1e-3:
                            append_log(
                                log_path,
                                f"[REJ] det={det2}  strict 模式: 缩放后最短任意原子间距 "
                                f"{phys_min:.3f}Å < {bond_min:.2f}Å"
                            )
                            continue
                    else:
                        # soft 模式：
                        # 1) 仍然要求最短任意原子间距 >= bond_min
                        if phys_min < bond_min - 1e-3:
                            append_log(
                                log_path,
                                f"[REJ] det={det2}  soft 模式: 缩放后最短任意原子间距 "
                                f"{phys_min:.3f}Å < {bond_min:.2f}Å"
                            )
                            continue

                        # 2) 要求“足够比例”的 center–neighbor 键落在目标区间内
                        mask_in = (cand_d >= bond_min - 1e-3) & (cand_d <= bond_max + 1e-3)
                        frac_in = float(np.mean(mask_in))
                        if frac_in < args.bond_coverage_min:
                            append_log(
                                log_path,
                                f"[REJ] det={det2}  soft 模式: 仅 {frac_in*100:.1f}% 键在 "
                                f"[{bond_min:.2f}, {bond_max:.2f}]Å 内 < bond_coverage_min="
                                f"{args.bond_coverage_min:.2f}"
                            )
                            continue
                else:
                    # 不做 a,b 缩放，只用 min_pair_dist 做硬约束避免重叠
                    if phys_min < args.min_pair_dist:
                        append_log(
                            log_path,
                            f"[REJ] det={det2}  min-pair={phys_min:.3f}Å < {args.min_pair_dist}Å (未启用键长缩放)"
                        )
                        continue

                # --- 计算 cost & CN 过滤 ---
                cost, cost_stats = candidate_cost(
                    st,
                    ref_stats=ref_stats,
                    bond_tol=args.bond_tol,
                    cn_target=args.cn_target,
                    w_d=args.w_cost_d,
                    w_cn=args.w_cost_cn,
                    center_species=args.center_species,
                    neighbor_species=args.neighbor_species,
                )

                CNs = cost_stats.get("CNs", [])
                if not CNs:
                    append_log(log_path, f"[REJ] det={det2}  CN 统计为空")
                    continue

                if not all(cn_lo <= cn <= cn_hi for cn in CNs):
                    append_log(
                        log_path,
                        f"[REJ] det={det2}  CN={CNs} 不在允许区间 [{cn_lo}, {cn_hi}] 内"
                    )
                    continue

                # --- SG 验证 ---
                ok_sg, sg_found, sym_used = verify_spacegroup_exact(
                    st, target_sg=sg, symprecs=symprecs, angle_tolerance=args.angle_tol
                )
                if not ok_sg:
                    append_log(
                        log_path,
                        f"[REJ] det={det2}  识别空间群={sg_found} != 目标 SG={sg}"
                    )
                    continue

                # --- 转原胞 ---
                prim = to_primitive(st, symprec=sym_used or symprecs[0], angle_tolerance=args.angle_tol)

                # --- 去重：避免同一个 SG 下结构重复 ---
                is_dup = False
                for item in found:
                    if matchers[sg].fit(prim, item["primitive"]):
                        is_dup = True
                        break
                if is_dup:
                    append_log(log_path, f"[SKIP] det={det2}  发现重复结构，跳过")
                    continue

                # --- 记录候选 ---
                meta = {
                    "sg_target": sg,
                    "sg_found": sg_found,
                    "det": det2,
                    "scale": scale_ab,
                    "symprec": sym_used,
                    "angle_tol": args.angle_tol,
                    "cost": cost,
                    "cost_stats": cost_stats,
                    "dmin_any": phys_min,
                    "bond_min": cand_min,
                    "bond_max": cand_max,
                    "bond_target_range": [bond_min, bond_max],
                    "z_thickness": thickness,
                    "z_surf_corr": surf_corr,
                    "z_ptp": thickness,   # 为了兼容旧的字段名，仍然记录一份
                    "n_atoms": len(prim),
                }
                found.append({"primitive": prim, "meta": meta})

                if args.debug_save_all_cands:
                    debug_save(prim, det2, scale_ab, "accepted")

        # 每个 SG 只保留 topk
        found = sorted(found, key=lambda x: (x["meta"]["cost"], -x["meta"]["dmin_any"]))[:args.topk]

        for rank, item in enumerate(found, 1):
            prim = item["primitive"]
            meta = item["meta"]

            wy_map = wyckoff_map_by_species(prim, symprec=meta["symprec"] or symprecs[0],
                                            angle_tolerance=args.angle_tol)
            meta["wyckoff_map"] = wy_map

            stem = f"cand_{rank}"
            cif_path, poscar_path = save_structure_pair(prim, sg_dir, stem)
            meta_path = os.path.join(sg_dir, f"{stem}.meta.json")
            write_json(meta_path, meta)

            index_rows.append({
                "sg": sg, "rank": rank, "cost": meta["cost"],
                "n_atoms": len(prim),
                "det": meta["det"], "scale": meta["scale"],
                "symprec": meta["symprec"], "angle_tol": meta["angle_tol"],
                "dmin": meta["dmin_any"], "z_ptp": meta["z_ptp"],
                "cif": os.path.relpath(cif_path, outdir),
                "poscar": os.path.relpath(poscar_path, outdir),
                "meta": os.path.relpath(meta_path, outdir)
            })

        if found:
            best_cost = found[0]["meta"]["cost"]
            append_log(log_path, f"[SUM] 命中 {len(found)} 条；最优 cost={best_cost:.3f}")
        else:
            append_log(log_path, "[SUM] 命中 0 条")

    # 写总索引 CSV / JSON
    with open(index_csv, "w", encoding="utf-8") as f:
        f.write("sg,rank,cost,n_atoms,det,scale,symprec,angle_tol,dmin,z_ptp,cif,poscar,meta\n")
        for r in sorted(index_rows, key=lambda x: (x["sg"], x["rank"])):
            sp = r["symprec"]
            symprec_str = f"{sp:.0e}" if sp else "0"
            f.write(
                f"{r['sg']},{r['rank']},{r['cost']:.5f},{r['n_atoms']},"
                f"{r['det']},{r['scale']:.6f},{symprec_str},"
                f"{r['angle_tol']:.3f},{r['dmin']:.3f},{r['z_ptp']:.3f},"
                f"{r['cif']},{r['poscar']},{r['meta']}\n"
            )

    write_json(index_json, {"rows": sorted(index_rows, key=lambda x: (x["sg"], x["rank"]))})

    print(f"[DONE] 输出目录：{os.path.abspath(outdir)}")
    if not _HAS_PYXTAL:
        print("[WARN] 未安装 pyxtal：未进行 Wyckoff 组合采样；建议 `pip install pyxtal` 后重试。")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
