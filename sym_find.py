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
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymatgen.core import Structure, Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.local_env import CrystalNN, LocalStructOrderParams
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


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def axis_to_index(value) -> int:
    mapping = {
        "a": 0, "x": 0, "0": 0, 0: 0,
        "b": 1, "y": 1, "1": 1, 1: 1,
        "c": 2, "z": 2, "2": 2, 2: 2,
    }
    if isinstance(value, str):
        key = value.strip().lower()
    else:
        key = value
    if key in mapping:
        return mapping[key]
    raise ValueError(f"layer_axis 不支持 {value!r}，请使用 a/b/c 或 x/y/z")


def axis_index_to_label(axis: int) -> str:
    return {0: "a", 1: "b", 2: "c"}.get(axis, "c")


def parse_pair_distance_matrix(raw) -> Dict[Tuple[str, str], float]:
    matrix: Dict[Tuple[str, str], float] = {}
    if not raw:
        return matrix

    def _store(a: str, b: str, dist):
        if dist is None:
            return
        key = tuple(sorted((a.lower(), b.lower())))
        matrix[key] = float(dist)

    if isinstance(raw, dict):
        for key, val in raw.items():
            if isinstance(val, dict):
                a = key
                for b, dist in val.items():
                    _store(a, b, dist)
            else:
                if "-" not in key:
                    raise ValueError(f"min_pair_dist_matrix 的键需形如 A-B，收到 {key!r}")
                a, b = key.split("-", 1)
                _store(a, b, val)
    else:
        raise TypeError("min_pair_dist_matrix 需为字典或嵌套字典")
    return matrix


@dataclass
class PairDistanceReport:
    min_distance: float
    min_pair: Tuple[str, str]
    violations: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class PreprocessResult:
    ok: bool
    struct: Optional[Structure] = None
    scale_ab: float = 1.0
    reason: Optional[str] = None
    metrics: Dict[str, object] = field(default_factory=dict)


@dataclass
class FilterResult:
    ok: bool
    struct: Optional[Structure] = None
    metrics: Dict[str, object] = field(default_factory=dict)
    reason: Optional[str] = None


@dataclass
class EvaluatorResult:
    ok: bool
    primitive: Optional[Structure] = None
    meta: Optional[dict] = None
    reason: Optional[str] = None


@dataclass
class SlabValidationResult:
    ok: bool
    reason: Optional[str] = None
    metrics: Dict[str, object] = field(default_factory=dict)


def record_slab_validation(metrics: Dict[str, object],
                           stage: str,
                           result: SlabValidationResult,
                           final: bool = False):
    """Record slab validation metadata under *metrics*."""

    if not stage:
        stage = "default"
    stage_map = metrics.setdefault("slab_validation", {})
    stage_map[stage] = {
        **result.metrics,
        "ok": result.ok,
        "reason": result.reason,
    }
    if final or "is_2d_valid" not in metrics:
        metrics["is_2d_valid"] = result.ok
        if result.ok:
            metrics.pop("is_2d_reason", None)
        else:
            metrics["is_2d_reason"] = result.reason


@dataclass
class LocalEnvRule:
    species: Optional[Tuple[str, ...]]
    cn_range: Tuple[float, float]
    threshold: float
    preferred_motifs: Tuple[str, ...]
    label: str = "default"
    species_lower: Tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if self.species:
            self.species_lower = tuple(s.lower() for s in self.species)
        else:
            self.species_lower = tuple()

    def matches(self, specie: str) -> bool:
        if not self.species_lower:
            return False
        return specie.lower() in self.species_lower

    def describe(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "species": self.species,
            "cn_range": self.cn_range,
            "threshold": self.threshold,
            "preferred_motifs": self.preferred_motifs,
        }


@dataclass
class LocalEnvReport:
    ok: bool
    reason: Optional[str]
    summary: Dict[str, object]

    def to_metrics(self) -> Dict[str, object]:
        return dict(self.summary)


class LocalEnvConstraintChecker:
    def __init__(self, config: Optional[Dict[str, object]], fallback_cn_range: Tuple[int, int]):
        fallback = (float(fallback_cn_range[0]), float(fallback_cn_range[1]))
        cfg_defaults = (config or {}).get("defaults") or {}
        default_rule = LocalEnvRule(
            species=None,
            cn_range=tuple(cfg_defaults.get("cn_range", fallback)),
            threshold=float(cfg_defaults.get("threshold", 0.0)),
            preferred_motifs=tuple(cfg_defaults.get("preferred_motifs", ())),
            label=str(cfg_defaults.get("label", "default")),
        )
        self.default_rule = default_rule

        rules_cfg = (config or {}).get("rules") or []
        self.rules: List[LocalEnvRule] = []
        for entry in rules_cfg:
            rule = LocalEnvRule(
                species=tuple(entry.get("species")) if entry.get("species") else None,
                cn_range=tuple(entry.get("cn_range", fallback)),
                threshold=float(entry.get("threshold", default_rule.threshold)),
                preferred_motifs=tuple(entry.get("preferred_motifs", default_rule.preferred_motifs)),
                label=str(entry.get("label", "rule")),
            )
            self.rules.append(rule)

        motif_types = sorted({m for rule in [self.default_rule] + self.rules for m in rule.preferred_motifs})
        self._motif_types = tuple(motif_types)
        self._cnn = CrystalNN()
        self._lsop = LocalStructOrderParams(list(self._motif_types)) if self._motif_types else None
        self._rule_summary = [rule.describe() for rule in ([self.default_rule] + self.rules)]

    def analyze(self, struct: Structure) -> LocalEnvReport:
        species_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "cn_fail": 0, "motif_fail": 0})
        cn_fail = 0
        motif_fail = 0
        failures: List[Dict[str, object]] = []

        for idx, site in enumerate(struct.sites):
            specie = site.species_string
            rule = self._rule_for(specie)
            cn_val, cn_ok = self._compute_cn(struct, idx, rule)
            motif_meta = self._compute_motif(struct, idx, rule)

            species_stats[specie]["count"] += 1
            if not cn_ok:
                species_stats[specie]["cn_fail"] += 1
                cn_fail += 1
            if motif_meta and not motif_meta["ok"]:
                species_stats[specie]["motif_fail"] += 1
                motif_fail += 1

            if (not cn_ok) or (motif_meta and not motif_meta["ok"]):
                fail_entry = {
                    "index": idx,
                    "species": specie,
                    "cn": None if cn_val is None else round(cn_val, 4),
                    "cn_range": rule.cn_range,
                    "cn_ok": cn_ok,
                    "rule": rule.label,
                }
                if motif_meta:
                    fail_entry.update({
                        "best_motif": motif_meta.get("best_motif"),
                        "best_score": motif_meta.get("best_score"),
                        "threshold": motif_meta.get("threshold"),
                        "motif_ok": motif_meta.get("ok"),
                    })
                failures.append(fail_entry)

        ok = (cn_fail == 0) and (motif_fail == 0)
        reason = None
        if failures:
            first = failures[0]
            if not first.get("cn_ok", True):
                reason = (
                    f"{first['species']} 位点 #{first['index']} 的 CN={first['cn']} 超出 [{first['cn_range'][0]}, {first['cn_range'][1]}]"
                )
            elif not first.get("motif_ok", True):
                reason = (
                    f"{first['species']} 位点 #{first['index']} motif 匹配失败："
                    f"{first.get('best_motif')} 得分={first.get('best_score')} < 阈值 {first.get('threshold')}"
                )

        species_summary = {k: dict(v) for k, v in species_stats.items()}
        summary: Dict[str, object] = {
            "ok": ok,
            "total_sites": len(struct.sites),
            "cn_failures": cn_fail,
            "motif_failures": motif_fail,
            "species_stats": species_summary,
            "rules": self._rule_summary,
        }
        if failures:
            summary["fail_sites"] = failures[:8]

        return LocalEnvReport(ok=ok, reason=reason, summary=summary)

    def _rule_for(self, specie: str) -> LocalEnvRule:
        for rule in self.rules:
            if rule.matches(specie):
                return rule
        return self.default_rule

    def _compute_cn(self, struct: Structure, index: int, rule: LocalEnvRule):
        try:
            cn_val = float(self._cnn.get_cn(struct, index))
        except Exception:
            cn_val = None
        if cn_val is None:
            return None, False
        lo, hi = rule.cn_range
        return cn_val, bool(lo - 1e-6 <= cn_val <= hi + 1e-6)

    def _compute_motif(self, struct: Structure, index: int, rule: LocalEnvRule) -> Optional[Dict[str, object]]:
        if not self._lsop or not rule.preferred_motifs:
            return None
        try:
            values = self._lsop.get_order_parameters(struct, index)
        except Exception:
            values = None
        if values is None:
            return {"ok": False, "best_motif": None, "best_score": None, "threshold": rule.threshold}
        motif_map = {motif: float(values[i]) for i, motif in enumerate(self._motif_types)}
        best_motif = None
        best_score = None
        for motif in rule.preferred_motifs:
            if motif not in motif_map:
                continue
            score = motif_map[motif]
            if best_score is None or score > best_score:
                best_score = score
                best_motif = motif
        ok = True
        if rule.preferred_motifs:
            ok = bool(best_score is not None and best_score >= rule.threshold)
        return {
            "ok": ok,
            "best_motif": best_motif,
            "best_score": None if best_score is None else round(best_score, 4),
            "threshold": rule.threshold,
        }


class GeometryConstraints:
    """Utility helpers for pair-distance and hard-sphere checks."""

    def __init__(self, args):
        self.min_pair_dist = float(args.min_pair_dist)
        self.min_pair_matrix = dict(getattr(args, "min_pair_dist_matrix", {}))
        self.min_bond_length_factor = float(getattr(args, "min_bond_length_factor", 0.9))
        self.hard_sphere_radius_scale = float(getattr(args, "hard_sphere_radius_scale", 0.95))
        self.reject_if_overlap = as_bool(getattr(args, "reject_if_overlap", True))
        self._radius_cache: Dict[str, float] = {}

    def _element_radius(self, symbol: str) -> Optional[float]:
        key = symbol.lower()
        if key in self._radius_cache:
            return self._radius_cache[key]
        try:
            elem = Element(symbol)
            rad = elem.covalent_radius or elem.atomic_radius
        except Exception:
            rad = None
        if rad is not None:
            rad = float(rad)
        self._radius_cache[key] = rad if rad is None else float(rad)
        return self._radius_cache[key]

    def pair_threshold(self, elem_a: str, elem_b: str) -> float:
        key = tuple(sorted((elem_a.lower(), elem_b.lower())))
        if key in self.min_pair_matrix:
            return float(self.min_pair_matrix[key])
        ra = self._element_radius(elem_a)
        rb = self._element_radius(elem_b)
        fallback = self.min_pair_dist
        if ra is not None and rb is not None:
            fallback = max(fallback, self.min_bond_length_factor * (ra + rb))
        return fallback

    def hard_sphere_radius(self, elem: str) -> float:
        rad = self._element_radius(elem)
        if rad is None:
            rad = 0.5 * self.min_pair_dist
        return self.hard_sphere_radius_scale * float(rad)

    def analyze_pairs(self, struct: Structure) -> PairDistanceReport:
        L = struct.lattice
        frac = np.array([s.frac_coords for s in struct.sites])
        species = [s.species_string for s in struct.sites]
        n = len(species)
        dmin = 1e9
        pair = ("", "")
        violations: List[Dict[str, object]] = []
        for i in range(n):
            for j in range(i + 1, n):
                df = min_image_vec(frac[j] - frac[i])
                d = np.linalg.norm(df @ L.matrix)
                if d < dmin:
                    dmin = d
                    pair = (species[i], species[j])
                thresh = self.pair_threshold(species[i], species[j])
                if d < thresh:
                    violations.append({
                        "i": i,
                        "j": j,
                        "species_i": species[i],
                        "species_j": species[j],
                        "distance": d,
                        "threshold": thresh,
                    })
        return PairDistanceReport(min_distance=float(dmin) if n else 0.0,
                                  min_pair=pair,
                                  violations=violations)


def hard_sphere_relax(struct: Structure,
                      constraints: GeometryConstraints,
                      max_iter: int = 30,
                      step: float = 0.4):
    """Simple hard-sphere relaxation that pushes overlapping atoms apart."""

    st = struct.copy()
    lattice = st.lattice
    frac = np.array([s.frac_coords for s in st.sites], dtype=float)
    species = [s.species_string for s in st.sites]

    iterations = 0
    for it in range(max_iter):
        moved = False
        for i in range(len(species)):
            for j in range(i + 1, len(species)):
                df = min_image_vec(frac[j] - frac[i])
                vec = df @ lattice.matrix
                dist = np.linalg.norm(vec)
                thresh = constraints.pair_threshold(species[i], species[j])
                if dist >= thresh or dist <= 1e-6:
                    continue
                direction = vec / (dist + 1e-8)
                delta = 0.5 * (thresh - dist) * min(1.0, step)
                move_cart = direction * delta
                move_frac = lattice.get_fractional_coords(move_cart)
                frac[i] = (frac[i] - move_frac) % 1.0
                frac[j] = (frac[j] + move_frac) % 1.0
                moved = True
        iterations = it + 1
        if not moved:
            break

    relaxed = Structure(lattice,
                        [s.species for s in st.sites],
                        frac,
                        coords_are_cartesian=False)
    return relaxed, iterations


def motif_overlap_report(struct: Structure,
                         constraints: GeometryConstraints,
                         symprec=1e-3,
                         angle_tolerance=0.5,
                         pad: float = 0.2):
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
        dataset = sga.get_symmetry_dataset()
    except Exception:
        return False, {}

    equiv = dataset.get("equivalent_atoms")
    if equiv is None:
        return False, {}

    groups = defaultdict(list)
    for idx, label in enumerate(equiv):
        groups[label].append(idx)

    lattice = struct.lattice
    centers = []
    for label, indices in groups.items():
        coords = np.array([struct[i].frac_coords for i in indices])
        cart = coords @ lattice.matrix
        center_cart = np.mean(cart, axis=0)
        center_frac = lattice.get_fractional_coords(center_cart)
        radii = [constraints.hard_sphere_radius(struct[i].species_string) for i in indices]
        local = np.max([np.linalg.norm(cart[k] - center_cart) + radii[k] for k in range(len(indices))])
        centers.append({
            "label": label,
            "center_frac": center_frac,
            "radius": float(local + pad),
        })

    overlaps = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            df = min_image_vec(centers[j]["center_frac"] - centers[i]["center_frac"])
            dist = np.linalg.norm(df @ lattice.matrix)
            limit = centers[i]["radius"] + centers[j]["radius"]
            if dist < limit:
                overlaps.append({
                    "motif_i": centers[i]["label"],
                    "motif_j": centers[j]["label"],
                    "distance": dist,
                    "limit": limit,
                })

    return bool(overlaps), {"overlaps": overlaps, "motif_count": len(centers)}


# ------------------------------ 从 config.json 读取 settings ------------------------------

def parse_sg_list(value):
    """Parse a user-provided space-group spec into a sorted list of ints.

    The spec can be:
    - ``None`` → returns ``None`` (caller decides the default behaviour).
    - an int → ``[int]``.
    - a list/tuple of ints or strings.
    - a string with comma/semicolon separated tokens. Each token can either be
      a single integer (``"33"``) or an inclusive range (``"20-45"``).
    """

    if value is None:
        return None

    if isinstance(value, int):
        tokens = [value]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    elif isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        tokens = [tok.strip() for tok in txt.replace(";", ",").split(",") if tok.strip()]
    else:
        raise TypeError(f"无法解析的空间群列表: {value!r}")

    out = []
    for tok in tokens:
        if isinstance(tok, int):
            nums = [tok]
        else:
            tok_str = str(tok).strip()
            if not tok_str:
                continue
            if "-" in tok_str:
                start_str, end_str = tok_str.split("-", 1)
                if not start_str.strip() or not end_str.strip():
                    raise ValueError(f"非法的空间群区间: {tok_str}")
                start, end = int(start_str), int(end_str)
                if start > end:
                    start, end = end, start
                nums = list(range(start, end + 1))
            else:
                nums = [int(tok_str)]
        for num in nums:
            if num < 1 or num > 230:
                raise ValueError(f"空间群编号应在 1..230 之间，收到 {num}")
            out.append(int(num))

    return sorted(set(out))


def _parse_cn_range(value, fallback: Tuple[float, float]) -> Tuple[float, float]:
    if value is None:
        return tuple(float(x) for x in fallback)
    if isinstance(value, str):
        tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
    elif isinstance(value, (list, tuple)):
        tokens = list(value)
    else:
        raise TypeError(f"cn_range 需为字符串或列表，收到 {value!r}")
    if len(tokens) != 2:
        raise ValueError(f"cn_range 需包含两个数值，收到 {tokens!r}")
    lo, hi = float(tokens[0]), float(tokens[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_motif_list(raw, fallback: Sequence[str] = ()) -> Tuple[str, ...]:
    if raw is None:
        return tuple(str(m) for m in fallback)
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        raise TypeError(f"preferred_motifs 需为字符串或数组，收到 {raw!r}")
    motifs: List[str] = []
    for item in items:
        txt = str(item).strip()
        if txt:
            motifs.append(txt)
    return tuple(motifs)


def _ensure_species_list(value, fallback: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    if value is None:
        if fallback is None:
            raise ValueError("local_env_constraints: 缺少 species 指定")
        value = fallback
    if isinstance(value, str):
        species = [value]
    elif isinstance(value, (list, tuple, set)):
        species = list(value)
    else:
        raise TypeError(f"species 需为字符串或字符串列表，收到 {value!r}")
    normalized = []
    for sp in species:
        txt = str(sp).strip()
        if txt:
            normalized.append(txt)
    if not normalized:
        raise ValueError("species 列表不能为空")
    return tuple(normalized)


def normalize_local_env_constraints(params: Dict[str, object], cn_range_fallback: Tuple[int, int]):
    raw = params.get("local_env_constraints")
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise TypeError("local_env_constraints 需为对象")

    default_cn_range = _parse_cn_range(raw.get("defaults", {}).get("cn_range"), tuple(float(x) for x in cn_range_fallback))
    default_threshold = float(raw.get("defaults", {}).get("threshold", 0.7))
    default_motifs = _parse_motif_list(raw.get("defaults", {}).get("preferred_motifs", ()))

    normalized = {
        "defaults": {
            "cn_range": default_cn_range,
            "threshold": default_threshold,
            "preferred_motifs": default_motifs,
            "label": raw.get("defaults", {}).get("label", "default"),
        },
        "rules": [],
    }

    entries = raw.get("elements")
    if entries is None:
        for key in ("rules", "per_species"):
            if key in raw:
                entries = raw[key]
                break
    if entries is None:
        inferred = {k: v for k, v in raw.items() if k not in {"defaults", "elements", "rules", "per_species"}}
        if inferred:
            entries = inferred

    if entries is None:
        return normalized

    def _iter_entries(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                yield key, val
        elif isinstance(obj, list):
            for val in obj:
                yield None, val
        else:
            raise TypeError("local_env_constraints.elements 需为对象或数组")

    for key, entry in _iter_entries(entries):
        if not isinstance(entry, dict):
            raise TypeError("local_env_constraints 元素需为对象")
        species = _ensure_species_list(entry.get("species"), fallback=[key] if key else None)
        cn_range = _parse_cn_range(entry.get("cn_range"), default_cn_range)
        threshold = float(entry.get("threshold", default_threshold))
        motifs = _parse_motif_list(entry.get("preferred_motifs"), default_motifs)
        label = entry.get("label") or "/".join(species)
        normalized["rules"].append({
            "species": species,
            "cn_range": cn_range,
            "threshold": threshold,
            "preferred_motifs": motifs,
            "label": label,
        })

    return normalized


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
        "vacuum_thickness": 20.0,
        "vacuum_buffer": 1.0,
        "min_vacuum_thickness": None,

        "det_max": 12,
        "max_atoms": 96,
        "enable_sqrt3": True,      # bool

        "symprec": [1e-3, 2e-3, 5e-3],
        "angle_tol": 0.5,

        "cn_target": 4,
        "cn_range": [3, 6],
        "min_pair_dist": 1.8,
        "min_pair_dist_matrix": {},
        "min_bond_length_factor": 0.9,
        "hard_sphere_radius_scale": 0.95,
        "reject_if_overlap": True,
        "motif_overlap_tol": 0.25,
        "density_range": None,
        "post_gen_relax": {"mode": "none", "max_iter": 30, "step": 0.4},

        # 表面起伏上限（Å），控制顶 / 底表面原子的 z 起伏；不再直接表示整体厚度
        "z_flat_tol": 0.3,

        "ab_scale": 0.05,          # 现在不再用于 lattice，只保留接口
        "ab_scale_samples": 3,

        "samples_per_sg": 100,
        "topk": 3,
        "seed": 2025,

        # 目标原子层最大整体厚度（Å），>0 时会主动压扁 z 分布
        "layer_thickness": None,
        "layer_thickness_max": 3.0,
        "layer_axis": "c",
        "slab_center": 0.5,
        "slab_center_tol": 0.02,
        "reslab_after_relax": True,

        # 用于最近邻统计和 CN 的元素对（可以改成任意中心/邻居元素）
        "center_species": "Ga",
        "neighbor_species": "As",

        "local_env_constraints": None,

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

        # target SG selection（None 表示自动）
        "target_sgs": None,
        "exclude_sgs": [],
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
    for key in ("enable_sqrt3", "enable_bond_scaling", "reject_if_overlap",
                "debug_save_all_cands", "debug_save_rejected"):
        params[key] = as_bool(params.get(key))

    # 数值
    params["bond_target_min"] = float(params["bond_target_min"])
    params["bond_target_max"] = float(params["bond_target_max"])
    params["min_pair_dist"] = float(params["min_pair_dist"])
    params["min_bond_length_factor"] = float(params.get("min_bond_length_factor", 0.9))
    params["hard_sphere_radius_scale"] = float(params.get("hard_sphere_radius_scale", 0.95))
    params["motif_overlap_tol"] = float(params.get("motif_overlap_tol", 0.25))
    if "vacuum_thickness" not in params and "vacuum_c" in params:
        params["vacuum_thickness"] = params["vacuum_c"]
    params["vacuum_thickness"] = float(params.get("vacuum_thickness", 20.0))
    if params["vacuum_thickness"] <= 0:
        raise ValueError("vacuum_thickness 必须为正数，才能形成二维层状结构")
    params["vacuum_buffer"] = float(params.get("vacuum_buffer", 1.0))
    min_vac = params.get("min_vacuum_thickness")
    if min_vac is None:
        params["min_vacuum_thickness"] = params["vacuum_thickness"]
    else:
        params["min_vacuum_thickness"] = float(min_vac)
    params["z_flat_tol"] = float(params["z_flat_tol"])
    layer_thickness = params.get("layer_thickness")
    if layer_thickness is None:
        params["layer_thickness"] = None
    else:
        layer_val = float(layer_thickness)
        params["layer_thickness"] = layer_val if layer_val > 0 else None

    params["layer_thickness_max"] = float(params["layer_thickness_max"])
    params["slab_center"] = float(params.get("slab_center", 0.5))
    params["slab_center_tol"] = float(params.get("slab_center_tol", 0.02))
    params["w_cost_d"] = float(params["w_cost_d"])
    params["w_cost_cn"] = float(params["w_cost_cn"])
    params["surface_frac"] = float(params["surface_frac"])
    params["layer_axis"] = str(params.get("layer_axis", "c"))
    params["layer_axis_index"] = axis_to_index(params["layer_axis"])
    params["reslab_after_relax"] = as_bool(params.get("reslab_after_relax", True))

    params["bond_scaling_mode"] = str(params.get("bond_scaling_mode", "soft")).lower()
    params["bond_coverage_min"] = float(params.get("bond_coverage_min", 0.8))

    params["min_pair_dist_matrix"] = parse_pair_distance_matrix(params.get("min_pair_dist_matrix"))

    density_range = params.get("density_range")
    if density_range is None:
        params["density_range"] = None
    else:
        if isinstance(density_range, str):
            density_range = [float(x) for x in density_range.split(",") if x.strip()]
        if len(density_range) != 2:
            raise ValueError("density_range 需包含两个数值 [min, max]")
        params["density_range"] = (float(density_range[0]), float(density_range[1]))

    post_relax = params.get("post_gen_relax") or {}
    if not isinstance(post_relax, dict):
        raise TypeError("post_gen_relax 需为对象，例如 {\"mode\": \"none\"}")
    base_relax = defaults["post_gen_relax"]
    merged_relax = {**base_relax, **post_relax}
    merged_relax["mode"] = str(merged_relax.get("mode", "none")).lower()
    merged_relax["max_iter"] = int(merged_relax.get("max_iter", base_relax["max_iter"]))
    merged_relax["step"] = float(merged_relax.get("step", base_relax["step"]))
    params["post_gen_relax"] = merged_relax

    params["target_sgs"] = parse_sg_list(params.get("target_sgs"))
    excl = parse_sg_list(params.get("exclude_sgs"))
    params["exclude_sgs"] = excl if excl is not None else []

    params["local_env_constraints"] = normalize_local_env_constraints(params, tuple(params["cn_range"]))

    return argparse.Namespace(**params)


# ------------------------------ 几何与邻接（通用元素版） ------------------------------

def ensure_vacuum_axis(struct: Structure,
                       axis_index: int = 2,
                       target: float = 20.0,
                       buffer: float = 1.0) -> Structure:
    """Adjust the lattice vector *axis_index* so it provides the desired vacuum."""

    if target <= 0:
        return struct.copy()

    lattice = struct.lattice
    matrix = lattice.matrix.copy()
    axis_vec = matrix[axis_index]
    current = np.linalg.norm(axis_vec)
    lo = target - max(buffer, 0.0)
    hi = target + max(buffer, 0.0)
    if lo <= current <= hi and current > 0:
        return struct.copy()

    if current <= 0:
        direction = np.zeros(3)
        direction[axis_index] = 1.0
    else:
        direction = axis_vec / current
    matrix[axis_index] = direction * target
    new_lattice = Lattice(matrix)
    return Structure(new_lattice,
                     [s.species for s in struct.sites],
                     [s.frac_coords for s in struct.sites],
                     coords_are_cartesian=False)


def ensure_vacuum_c(struct: Structure, c_target_low=18.0, c_target_high=25.0) -> Structure:
    target = 0.5 * (c_target_low + c_target_high)
    buffer = max(target - c_target_low, c_target_high - target)
    return ensure_vacuum_axis(struct, axis_index=2, target=target, buffer=buffer)


def orthogonalize_lattice_to_axis(struct: Structure, axis_index: int = 2) -> Structure:
    """Remove the component of in-plane lattice vectors along the slab normal."""

    lattice = struct.lattice
    mat = np.array(lattice.matrix, dtype=float)
    axis_vec = mat[axis_index]
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm <= 1e-8:
        axis_vec = np.zeros(3)
        axis_vec[axis_index] = 1.0
        axis_norm = 1.0
    axis_unit = axis_vec / axis_norm

    for idx in range(3):
        if idx == axis_index:
            continue
        vec = mat[idx]
        vec_proj = vec - np.dot(vec, axis_unit) * axis_unit
        if np.linalg.norm(vec_proj) <= 1e-8:
            basis = np.zeros(3)
            basis[(idx + 1) % 3] = 1.0
            vec_proj = basis - np.dot(basis, axis_unit) * axis_unit
        mat[idx] = vec_proj

    mat[axis_index] = axis_unit * axis_norm
    new_lattice = Lattice(mat)
    return Structure(new_lattice,
                     [s.species for s in struct.sites],
                     struct.cart_coords,
                     coords_are_cartesian=True)


def set_axis_length(struct: Structure, axis_index: int, target: float) -> Structure:
    """Rescale a lattice vector while keeping fractional coordinates fixed."""

    if target <= 0:
        return struct

    mat = np.array(struct.lattice.matrix, dtype=float)
    axis_vec = mat[axis_index]
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm <= 1e-8:
        axis_vec = np.zeros(3)
        axis_vec[axis_index] = 1.0
        axis_norm = 1.0
    mat[axis_index] = axis_vec / axis_norm * target
    new_lattice = Lattice(mat)
    return Structure(new_lattice,
                     [s.species for s in struct.sites],
                     [s.frac_coords for s in struct.sites],
                     coords_are_cartesian=False)


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


def layer_thickness_angstrom(struct: Structure, axis_index: int = 2) -> float:
    """Return the slab thickness (Å) along the selected axis."""

    axis_len = struct.lattice.lengths[axis_index]
    coords = np.array([s.frac_coords[axis_index] * axis_len for s in struct.sites])
    return float(np.ptp(coords))


def _normalize_layer_axis(layer_axis) -> Tuple[int, float]:
    center = 0.5
    axis_val = layer_axis
    if isinstance(layer_axis, (tuple, list)):
        if layer_axis:
            axis_val = layer_axis[0]
        if len(layer_axis) > 1 and layer_axis[1] is not None:
            center = float(layer_axis[1])
    axis_index = axis_to_index(axis_val)
    center = float(center)
    if not math.isfinite(center):
        center = 0.5
    center = center % 1.0
    return axis_index, center


def is_2d_structure(struct: Structure,
                    layer_axis,
                    max_layer_thickness: Optional[float],
                    min_vacuum_thickness: Optional[float],
                    slab_center_tol: float) -> SlabValidationResult:
    """Validate whether *struct* already represents a single 2D slab."""

    if struct is None:
        return SlabValidationResult(False, reason="结构为空")

    axis_index, slab_center = _normalize_layer_axis(layer_axis)
    axis_len = float(struct.lattice.lengths[axis_index])
    if axis_len <= 1e-8:
        return SlabValidationResult(
            False,
            reason=f"轴 {axis_index_to_label(axis_index)} 长度 {axis_len:.4f}Å 非法",
            metrics={"slab_axis_length": axis_len},
        )

    coords = np.array([s.frac_coords[axis_index] for s in struct.sites], dtype=float)
    if coords.size == 0:
        slab_min = slab_max = 0.0
    else:
        rel = ((coords - slab_center + 0.5) % 1.0) - 0.5
        rel_cart = rel * axis_len
        slab_min = float(rel_cart.min())
        slab_max = float(rel_cart.max())

    slab_thickness = max(0.0, slab_max - slab_min)
    vacuum_thickness = max(0.0, axis_len - slab_thickness)
    center_offset_cart = 0.5 * (slab_max + slab_min)
    center_offset_frac = center_offset_cart / axis_len
    actual_center = (slab_center + center_offset_frac) % 1.0

    metrics = {
        "slab_axis_index": axis_index,
        "slab_axis_label": axis_index_to_label(axis_index),
        "slab_axis_length": axis_len,
        "slab_thickness": slab_thickness,
        "slab_vacuum": vacuum_thickness,
        "slab_center_target": slab_center,
        "slab_center_actual": actual_center,
        "slab_center_offset": center_offset_frac,
    }

    tol = 1e-3
    if max_layer_thickness is not None and max_layer_thickness > 0:
        if slab_thickness > max_layer_thickness + tol:
            return SlabValidationResult(
                False,
                reason=(
                    f"厚度 {slab_thickness:.3f}Å 超出上限 {max_layer_thickness:.3f}Å"
                ),
                metrics=metrics,
            )

    if min_vacuum_thickness is not None and min_vacuum_thickness > 0:
        if vacuum_thickness + tol < min_vacuum_thickness:
            return SlabValidationResult(
                False,
                reason=(
                    f"真空厚度 {vacuum_thickness:.3f}Å < 要求 {min_vacuum_thickness:.3f}Å"
                ),
                metrics=metrics,
            )

    if slab_center_tol is not None and slab_center_tol > 0:
        if abs(center_offset_frac) > slab_center_tol + 1e-5:
            return SlabValidationResult(
                False,
                reason=(
                    f"层中心偏移 {center_offset_frac:.4f} 超出 tol={slab_center_tol:.4f}"
                ),
                metrics=metrics,
            )

    return SlabValidationResult(True, metrics=metrics)


def surface_corrugation_angstrom(struct: Structure,
                                 surface_frac: float = 0.25,
                                 axis_index: int = 2):
    """
    估算“表面起伏”：
    - 先取所有原子的 z_cart，得到总厚度 thickness。
    - 顶部表面：z >= z_max - surface_frac * thickness
    - 底部表面：z <= z_min + surface_frac * thickness
    - 分别计算两个表面的 ptp，取最大值作为表面起伏。

    返回 (corrugation, thickness)，单位 Å。
    """
    axis_len = struct.lattice.lengths[axis_index]
    z_cart = np.array([s.frac_coords[axis_index] * axis_len for s in struct.sites])
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


def squash_layer_thickness(struct: Structure,
                           max_thickness: float,
                           axis_index: int = 2,
                           target_center: Optional[float] = 0.5,
                           target_thickness: Optional[float] = None) -> Structure:
    """Compress atoms into a thin slab along the selected axis."""

    axis_len = struct.lattice.lengths[axis_index]
    if axis_len <= 0:
        return struct

    coords = np.array([s.frac_coords for s in struct.sites], dtype=float)
    axis_cart = coords[:, axis_index] * axis_len
    if axis_cart.size == 0:
        return struct

    z_min = float(axis_cart.min())
    z_max = float(axis_cart.max())
    thickness = z_max - z_min
    if thickness < 1e-9:
        thickness = 0.0

    scale = 1.0
    if target_thickness is not None and target_thickness > 0 and thickness > 0:
        scale = target_thickness / thickness
    elif max_thickness is not None and max_thickness > 0 and thickness > max_thickness:
        scale = max_thickness / thickness

    mid_current = 0.5 * (z_max + z_min)
    if target_center is None:
        target_mid = mid_current
    else:
        frac_center = float(target_center)
        target_mid = frac_center * axis_len

    new_frac = []
    for fc in coords:
        axis_val = fc[axis_index] * axis_len
        axis_new = (axis_val - mid_current) * scale + target_mid
        fc_new = fc.copy()
        fc_new[axis_index] = (axis_new / axis_len) % 1.0
        new_frac.append(fc_new)

    return Structure(struct.lattice,
                     [s.species for s in struct.sites],
                     new_frac,
                     coords_are_cartesian=False)


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


class CandidateGenerator:
    def __init__(self, args):
        self.args = args

    def generate(self, sg: int, num_center: int, num_neighbor: int):
        if not _HAS_PYXTAL:
            return []
        return pyxtal_generate_candidates(
            sg,
            num_center=num_center,
            num_neighbor=num_neighbor,
            center_species=self.args.center_species,
            neighbor_species=self.args.neighbor_species,
            trials=self.args.samples_per_sg,
        )


class SlabProjector:
    """Utility object that enforces the 2D slab geometry."""

    def __init__(self, args):
        self.axis_index = int(getattr(args, "layer_axis_index", 2))
        self.axis_label = axis_index_to_label(self.axis_index)
        self.vacuum_target = float(getattr(args, "vacuum_thickness", 20.0))
        self.vacuum_buffer = float(getattr(args, "vacuum_buffer", 1.0))
        self.layer_thickness = getattr(args, "layer_thickness", None)
        self.layer_thickness_max = float(getattr(args, "layer_thickness_max", 0.0))
        self.surface_frac = float(getattr(args, "surface_frac", 0.25))
        self.slab_center = float(getattr(args, "slab_center", 0.5))

    def project(self, struct: Structure, with_metrics: bool = False):
        st = orthogonalize_lattice_to_axis(struct, axis_index=self.axis_index)
        raw_thickness = max(layer_thickness_angstrom(st, axis_index=self.axis_index), 1e-6)
        target_layer = raw_thickness
        if self.layer_thickness and self.layer_thickness > 0:
            target_layer = self.layer_thickness
        elif self.layer_thickness_max > 0 and raw_thickness > self.layer_thickness_max:
            target_layer = self.layer_thickness_max
        target_layer = max(target_layer, 1e-3)

        st = squash_layer_thickness(
            st,
            max_thickness=target_layer,
            axis_index=self.axis_index,
            target_center=self.slab_center,
            target_thickness=target_layer,
        )
        total_axis = target_layer + self.vacuum_target
        buffer = max(self.vacuum_buffer, 0.0)
        if buffer > 0:
            total_axis += buffer
        st = set_axis_length(st, self.axis_index, total_axis)
        st = squash_layer_thickness(
            st,
            max_thickness=target_layer,
            axis_index=self.axis_index,
            target_center=self.slab_center,
            target_thickness=target_layer,
        )

        metrics: Dict[str, object] = {}
        corr, thickness = surface_corrugation_angstrom(
            st,
            surface_frac=self.surface_frac,
            axis_index=self.axis_index,
        )
        metrics.update({
            "z_surf_corr": corr,
            "z_thickness": thickness,
            "layer_axis": self.axis_label,
            "layer_target": target_layer,
            "vacuum_gap": self.vacuum_target,
            "axis_length_total": total_axis,
        })
        if not with_metrics:
            metrics = {}
        return st, metrics


class CandidatePreprocessor:
    def __init__(self,
                 args,
                 projector: SlabProjector,
                 constraints: GeometryConstraints,
                 local_env_checker: Optional[LocalEnvConstraintChecker] = None):
        self.args = args
        self.projector = projector
        self.constraints = constraints
        self.local_env_checker = local_env_checker
        self._layer_axis = (self.projector.axis_index if self.projector else axis_to_index(args.layer_axis),
                            float(getattr(args, "slab_center", 0.5)))
        self._min_vacuum = float(getattr(args, "min_vacuum_thickness", getattr(args, "vacuum_thickness", 0.0)))
        self._slab_center_tol = float(getattr(args, "slab_center_tol", 0.02))

    def preprocess(self, struct: Structure) -> PreprocessResult:
        st, slab_metrics = self.projector.project(struct, with_metrics=True)
        metrics: Dict[str, object] = dict(slab_metrics)

        slab_check = self._validate_slab(st, metrics, stage="preprocessor", final=True)
        if not slab_check.ok:
            reason = slab_check.reason or "未通过二维 slab 校验"
            return PreprocessResult(False, struct=st, reason=f"2D 几何无效: {reason}", metrics=metrics)

        if self.args.layer_thickness_max > 0 and metrics.get("z_thickness", 0.0) > self.args.layer_thickness_max + 1e-3:
            return PreprocessResult(False, struct=st, reason=(
                f"thickness={metrics['z_thickness']:.3f}Å > layer_thickness_max={self.args.layer_thickness_max:.3f}Å"
            ), metrics=metrics)

        if metrics.get("z_surf_corr", 0.0) > self.args.z_flat_tol:
            return PreprocessResult(False, struct=st, reason=(
                f"surface_corr={metrics['z_surf_corr']:.3f}Å > z_flat_tol={self.args.z_flat_tol:.3f}Å"
            ), metrics=metrics)

        try:
            cand_d = compute_candidate_nn_distances(
                st,
                center_species=self.args.center_species,
                neighbor_species=self.args.neighbor_species,
                k_per_site=4,
            )
        except ValueError as exc:
            return PreprocessResult(False, struct=st, reason=f"compute_candidate_nn_distances 失败: {exc}", metrics=metrics)

        if cand_d.size == 0:
            return PreprocessResult(False, struct=st, reason="无法获取中心-邻居最近邻列表", metrics=metrics)

        metrics["bond_target_range"] = [self.args.bond_target_min, self.args.bond_target_max]

        if not self.args.enable_bond_scaling:
            cand_min = float(np.min(cand_d))
            cand_max = float(np.max(cand_d))
            report, pair_metrics = self._pair_metrics(st, prefix="pre")
            metrics.update({
                "bond_min": cand_min,
                "bond_max": cand_max,
                "dmin_any": report.min_distance,
            })
            metrics.update(pair_metrics)
            if report.violations:
                return PreprocessResult(
                    False,
                    struct=st,
                    reason=self._pair_violation_reason(report.violations[0]),
                    metrics=metrics,
                )
            env_check = self._local_env_metrics(st)
            if env_check:
                metrics["local_env_pre"] = env_check.to_metrics()
                if not env_check.ok:
                    return PreprocessResult(False, struct=st, reason=f"local_env: {env_check.reason}", metrics=metrics)
            return PreprocessResult(True, struct=st, scale_ab=1.0, metrics=metrics)

        ok, scaled, scale, extra_metrics, reason = self._apply_scaling(st, cand_d)
        if not ok or scaled is None:
            return PreprocessResult(False, struct=st, reason=reason, metrics=metrics)

        metrics.update(extra_metrics)
        env_check = self._local_env_metrics(scaled)
        if env_check:
            metrics["local_env_pre"] = env_check.to_metrics()
            if not env_check.ok:
                return PreprocessResult(False, struct=scaled, reason=f"local_env: {env_check.reason}", metrics=metrics)
        return PreprocessResult(True, struct=scaled, scale_ab=scale, metrics=metrics)

    def _validate_slab(self, struct: Structure, metrics: Dict[str, object], stage: str, final: bool = False):
        if not self.projector:
            result = SlabValidationResult(True, metrics={})
            record_slab_validation(metrics, stage, result, final=final)
            return result
        result = is_2d_structure(
            struct,
            layer_axis=self._layer_axis,
            max_layer_thickness=self.args.layer_thickness_max,
            min_vacuum_thickness=self._min_vacuum,
            slab_center_tol=self._slab_center_tol,
        )
        record_slab_validation(metrics, stage, result, final=final)
        return result

    def _local_env_metrics(self, struct: Structure) -> Optional[LocalEnvReport]:
        if not self.local_env_checker:
            return None
        return self.local_env_checker.analyze(struct)

    def _apply_scaling(self, struct: Structure, cand_d: np.ndarray):
        bond_min = self.args.bond_target_min
        bond_max = self.args.bond_target_max
        cand_min = float(np.min(cand_d))
        cand_max = float(np.max(cand_d))
        if cand_min <= 1e-6 or cand_max <= 1e-6:
            return False, None, 1.0, {}, f"cand_min/cand_max 非法: {cand_min:.4f}/{cand_max:.4f}"

        mode = getattr(self.args, "bond_scaling_mode", "soft").lower()
        if mode == "strict":
            s_lo = bond_min / cand_min
            s_hi = bond_max / cand_max
            if s_lo > s_hi:
                return False, None, 1.0, {}, (
                    f"无法同时缩放至 [{bond_min:.2f}, {bond_max:.2f}]Å，原始范围 {cand_min:.2f}–{cand_max:.2f}Å"
                )
            if s_lo <= 1.0 <= s_hi:
                s_target = 1.0
            else:
                s_target = s_lo if abs(s_lo - 1.0) < abs(s_hi - 1.0) else s_hi
        else:
            d_med = float(np.median(cand_d))
            if d_med <= 1e-6:
                return False, None, 1.0, {}, f"soft 模式: d_med 非法 {d_med:.4f}"
            target_mid = 0.5 * (bond_min + bond_max)
            s_target = target_mid / d_med
            s_min_shortest = bond_min / cand_min
            if s_target < s_min_shortest:
                s_target = s_min_shortest

        s_target = max(0.5, min(s_target, 4.0))
        L_old = struct.lattice
        mat = L_old.matrix.copy()
        mat[0, :] *= s_target
        mat[1, :] *= s_target
        scaled = Structure(
            Lattice(mat),
            [s.species for s in struct.sites],
            [s.frac_coords for s in struct.sites],
            coords_are_cartesian=False,
        )

        cand_d_scaled = compute_candidate_nn_distances(
            scaled,
            center_species=self.args.center_species,
            neighbor_species=self.args.neighbor_species,
            k_per_site=4,
        )
        cand_min_scaled = float(np.min(cand_d_scaled))
        cand_max_scaled = float(np.max(cand_d_scaled))
        report, pair_metrics = self._pair_metrics(scaled, prefix="scaled")
        phys_min = report.min_distance

        if report.violations:
            return False, None, 1.0, pair_metrics, self._pair_violation_reason(report.violations[0])

        if mode == "strict":
            if cand_min_scaled < bond_min - 1e-3 or cand_max_scaled > bond_max + 1e-3:
                return False, None, 1.0, {}, (
                    f"strict 模式: 缩放后键长范围 {cand_min_scaled:.2f}–{cand_max_scaled:.2f}Å 不在目标区间"
                )
            if phys_min < bond_min - 1e-3:
                return False, None, 1.0, {}, (
                    f"strict 模式: 缩放后最短任意原子间距 {phys_min:.3f}Å < {bond_min:.2f}Å"
                )
            coverage = 1.0
        else:
            if phys_min < bond_min - 1e-3:
                return False, None, 1.0, {}, (
                    f"soft 模式: 缩放后最短任意原子间距 {phys_min:.3f}Å < {bond_min:.2f}Å"
                )
            mask_in = (cand_d_scaled >= bond_min - 1e-3) & (cand_d_scaled <= bond_max + 1e-3)
            coverage = float(np.mean(mask_in))
            if coverage < self.args.bond_coverage_min:
                return False, None, 1.0, {}, (
                    f"soft 模式: 仅 {coverage*100:.1f}% 键落在 [{bond_min:.2f}, {bond_max:.2f}]Å 内 < bond_coverage_min"
                )

        metrics = {
            "bond_min": cand_min_scaled,
            "bond_max": cand_max_scaled,
            "dmin_any": phys_min,
            "bond_coverage": coverage,
            "scale_applied": s_target,
            "bond_target_range": [bond_min, bond_max],
        }
        metrics.update(pair_metrics)
        return True, scaled, s_target, metrics, None

    def _pair_metrics(self, struct: Structure, prefix: str):
        report = self.constraints.analyze_pairs(struct)
        metrics = {
            f"{prefix}_pair_min": report.min_distance,
            f"{prefix}_pair_min_species": report.min_pair,
            f"{prefix}_pair_violation_count": len(report.violations),
            f"{prefix}_pair_violations": report.violations[:5],
        }
        return report, metrics

    @staticmethod
    def _pair_violation_reason(violation: Dict[str, object]) -> str:
        spec_i = violation.get("species_i")
        spec_j = violation.get("species_j")
        dist = violation.get("distance")
        thresh = violation.get("threshold")
        return (
            f"pair {spec_i}-{spec_j} 距离 {dist:.3f}Å < 阈值 {thresh:.3f}Å"
            if dist is not None and thresh is not None
            else "pair distance violation"
        )


class GeometryFilter:
    def __init__(self,
                 args,
                 constraints: GeometryConstraints,
                 symprecs: Sequence[float],
                 projector: Optional[SlabProjector] = None):
        self.args = args
        self.constraints = constraints
        self.symprecs = tuple(symprecs)
        self.projector = projector
        self._layer_axis = (self.projector.axis_index if self.projector else axis_to_index(args.layer_axis),
                            float(getattr(args, "slab_center", 0.5)))
        self._min_vacuum = float(getattr(args, "min_vacuum_thickness", getattr(args, "vacuum_thickness", 0.0)))
        self._slab_center_tol = float(getattr(args, "slab_center_tol", 0.02))

    def filter(self, struct: Structure) -> FilterResult:
        report = self.constraints.analyze_pairs(struct)
        metrics: Dict[str, object] = {
            "pair_min": report.min_distance,
            "pair_min_species": report.min_pair,
            "pair_violation_count": len(report.violations),
            "pair_violations": report.violations,
        }

        st = struct
        if self.projector and getattr(self.args, "reslab_after_relax", True):
            pre_val = self._validate_slab(st, metrics, stage="geom_pre_reslab", final=False)
            if not pre_val.ok:
                return FilterResult(False, struct=st, reason=f"2D 几何无效: {pre_val.reason}", metrics=metrics)
            st, _ = self.projector.project(st, with_metrics=False)
            post_val = self._validate_slab(st, metrics, stage="geom_post_reslab_init", final=False)
            if not post_val.ok:
                return FilterResult(False, struct=st, reason=f"2D 几何无效: {post_val.reason}", metrics=metrics)
        if report.violations:
            relax_cfg = getattr(self.args, "post_gen_relax", {}) or {}
            if relax_cfg.get("mode") == "hard_sphere":
                relaxed, iterations = hard_sphere_relax(
                    st,
                    self.constraints,
                    max_iter=relax_cfg.get("max_iter", 30),
                    step=relax_cfg.get("step", 0.4),
                )
                if self.projector and getattr(self.args, "reslab_after_relax", True):
                    post_relax_val = self._validate_slab(relaxed, metrics, stage="geom_post_relax", final=False)
                    if not post_relax_val.ok:
                        return FilterResult(False, struct=relaxed, reason=f"2D 几何无效: {post_relax_val.reason}", metrics=metrics)
                    relaxed, _ = self.projector.project(relaxed, with_metrics=False)
                    post_relax_reslab = self._validate_slab(relaxed, metrics, stage="geom_post_relax_reslab", final=False)
                    if not post_relax_reslab.ok:
                        return FilterResult(False, struct=relaxed, reason=f"2D 几何无效: {post_relax_reslab.reason}", metrics=metrics)
                st = relaxed
                report = self.constraints.analyze_pairs(st)
                metrics["relax"] = {
                    "mode": "hard_sphere",
                    "iterations": iterations,
                    "remaining_violations": len(report.violations),
                }
            if report.violations and self.constraints.reject_if_overlap:
                reason = (
                    f"最近原子距离 {report.min_distance:.3f}Å 违反硬球约束"
                    if report.min_distance
                    else "pair distance violation"
                )
                return FilterResult(False, struct=st, reason=reason, metrics=metrics)

        metrics["pair_min_after"] = report.min_distance

        if self.projector:
            corr, thickness = surface_corrugation_angstrom(
                st,
                surface_frac=self.projector.surface_frac,
                axis_index=self.projector.axis_index,
            )
            metrics["z_surf_corr_post"] = corr
            metrics["z_thickness_post"] = thickness
            if self.args.layer_thickness_max > 0 and thickness > self.args.layer_thickness_max + 1e-3:
                return FilterResult(
                    False,
                    struct=st,
                    reason=(
                        f"thickness={thickness:.3f}Å > layer_thickness_max={self.args.layer_thickness_max:.3f}Å"
                    ),
                    metrics=metrics,
                )
            if corr > self.args.z_flat_tol:
                return FilterResult(
                    False,
                    struct=st,
                    reason=f"surface_corr={corr:.3f}Å > z_flat_tol={self.args.z_flat_tol:.3f}Å",
                    metrics=metrics,
                )

        density = float(st.density)
        metrics["density"] = density
        if self.args.density_range is not None:
            lo, hi = self.args.density_range
            if not (lo <= density <= hi):
                return FilterResult(
                    False,
                    struct=st,
                    reason=f"density={density:.3f} g/cm^3 不在 [{lo}, {hi}] 区间",
                    metrics=metrics,
                )

        symprec = self.symprecs[0] if self.symprecs else 1e-3
        overlap_flag, overlap_meta = motif_overlap_report(
            st,
            self.constraints,
            symprec=symprec,
            angle_tolerance=self.args.angle_tol,
            pad=self.args.motif_overlap_tol,
        )
        overlap_meta.setdefault("overlaps", [])
        metrics["motif_overlap"] = overlap_meta
        metrics["motif_overlap_flag"] = overlap_flag
        if overlap_flag and self.constraints.reject_if_overlap:
            return FilterResult(False, struct=st, reason="motif 重叠", metrics=metrics)

        final_val = self._validate_slab(st, metrics, stage="geom_final", final=True)
        if not final_val.ok:
            return FilterResult(False, struct=st, reason=f"2D 几何无效: {final_val.reason}", metrics=metrics)

        return FilterResult(True, struct=st, metrics=metrics)

    def _validate_slab(self, struct: Structure, metrics: Dict[str, object], stage: str, final: bool = False):
        if not self.projector:
            result = SlabValidationResult(True, metrics={})
            record_slab_validation(metrics, stage, result, final=final)
            return result
        result = is_2d_structure(
            struct,
            layer_axis=self._layer_axis,
            max_layer_thickness=self.args.layer_thickness_max,
            min_vacuum_thickness=self._min_vacuum,
            slab_center_tol=self._slab_center_tol,
        )
        record_slab_validation(metrics, stage, result, final=final)
        return result


class CandidateEvaluator:
    def __init__(self,
                 args,
                 ref_stats: dict,
                 symprecs: Sequence[float],
                 local_env_checker: Optional[LocalEnvConstraintChecker] = None,
                 projector: Optional[SlabProjector] = None):
        self.args = args
        self.ref_stats = ref_stats
        self.symprecs = tuple(symprecs)
        self.projector = projector
        self.local_env_checker = local_env_checker

    def evaluate(self,
                 struct: Structure,
                 sg_target: int,
                 det2: int,
                 scale_ab: float,
                 base_meta: Dict[str, object]) -> EvaluatorResult:
        cost, cost_stats = candidate_cost(
            struct,
            ref_stats=self.ref_stats,
            bond_tol=self.args.bond_tol,
            cn_target=self.args.cn_target,
            w_d=self.args.w_cost_d,
            w_cn=self.args.w_cost_cn,
            center_species=self.args.center_species,
            neighbor_species=self.args.neighbor_species,
        )

        env_summary = None
        if self.local_env_checker:
            env_report = self.local_env_checker.analyze(struct)
            env_summary = env_report.to_metrics()
            if not env_report.ok:
                return EvaluatorResult(False, reason=f"local_env: {env_report.reason}")

        ok_sg, sg_found, sym_used = verify_spacegroup_exact(
            struct,
            target_sg=sg_target,
            symprecs=self.symprecs,
            angle_tolerance=self.args.angle_tol,
        )
        if not ok_sg:
            return EvaluatorResult(False, reason=f"识别空间群={sg_found} != 目标 SG={sg_target}")

        prim = to_primitive(struct, symprec=sym_used or self.symprecs[0], angle_tolerance=self.args.angle_tol)
        slab_metrics: Dict[str, object] = {}
        if self.projector is not None:
            prim, slab_metrics = self.projector.project(prim, with_metrics=True)

        meta = dict(base_meta)
        meta.update({
            "sg_target": sg_target,
            "sg_found": sg_found,
            "det": det2,
            "scale": scale_ab,
            "symprec": sym_used,
            "angle_tol": self.args.angle_tol,
            "cost": cost,
            "cost_stats": cost_stats,
            "n_atoms": len(prim),
        })
        if slab_metrics:
            meta.update(slab_metrics)
        if "z_thickness" in meta and "z_ptp" not in meta:
            meta["z_ptp"] = meta["z_thickness"]
        if env_summary:
            meta["local_env_final"] = env_summary

        return EvaluatorResult(True, primitive=prim, meta=meta)


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


def detect_structure_sg(struct: Structure, symprecs, angle_tolerance: float):
    """Return the first successfully detected space group for *struct*."""

    for sp in symprecs:
        try:
            sga = SpacegroupAnalyzer(struct, symprec=sp, angle_tolerance=angle_tolerance)
            return sga.get_space_group_number()
        except Exception:
            continue
    return None


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
    slab_projector = SlabProjector(args)
    base, _ = slab_projector.project(base, with_metrics=False)

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

    symprecs = tuple(float(x) for x in args.symprec)

    base_sg = detect_structure_sg(base, symprecs=symprecs, angle_tolerance=args.angle_tol)
    if base_sg is not None:
        print(f"[BASE] 输入结构识别空间群 SG={base_sg}")
    else:
        print("[BASE] 无法识别输入结构的空间群（将不会自动排除）")

    exclude_sgs = set(args.exclude_sgs or [])
    if args.target_sgs is not None:
        target_sgs = [sg for sg in args.target_sgs if sg not in exclude_sgs]
    else:
        if base_sg is not None:
            exclude_sgs.add(base_sg)
        target_sgs = [sg for sg in range(1, 231) if sg not in exclude_sgs]

    if not target_sgs:
        raise ValueError("目标空间群列表为空，请检查 target_sgs/exclude_sgs 设置。")

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

    constraints = GeometryConstraints(args)
    generator = CandidateGenerator(args)
    local_env_checker = LocalEnvConstraintChecker(args.local_env_constraints, tuple(args.cn_range))
    preprocessor = CandidatePreprocessor(args, slab_projector, constraints, local_env_checker)
    geom_filter = GeometryFilter(args, constraints, symprecs, slab_projector)
    evaluator = CandidateEvaluator(args, ref_stats, symprecs, local_env_checker, projector=slab_projector)

    for idx, sg in enumerate(target_sgs, 1):
        sg_dir = makedirs(os.path.join(outdir, f"SG_{sg:03d}"))
        log_path = os.path.join(log_dir, f"SG_{sg:03d}.log")
        found = []

        debug_dir = makedirs(os.path.join(sg_dir, "debug"))
        debug_counter = 0

        def debug_save(struct: Optional[Structure], det2: int, scale_ab: float, tag: str, accepted: bool = False):
            nonlocal debug_counter
            if struct is None:
                return
            if accepted:
                should_save = args.debug_save_all_cands
            else:
                should_save = args.debug_save_all_cands or args.debug_save_rejected
            if not should_save or debug_counter >= args.debug_max_per_sg:
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

            cands = generator.generate(sg, num_center=num_center, num_neighbor=num_neighbor)
            if not cands:
                append_log(log_path, f"[FAIL] det={det2}  pyxtal 采样均失败")
                continue

            for st_raw in cands:
                debug_save(st_raw, det2, 1.0, "raw")

                pre_res = preprocessor.preprocess(st_raw)
                if not pre_res.ok:
                    append_log(log_path, f"[REJ][prep] det={det2}  {pre_res.reason}")
                    debug_save(pre_res.struct or st_raw, det2, pre_res.scale_ab, "prep_fail")
                    continue

                debug_save(pre_res.struct, det2, pre_res.scale_ab, "prepped")

                filt_res = geom_filter.filter(pre_res.struct)
                if not filt_res.ok:
                    append_log(log_path, f"[REJ][geom] det={det2}  {filt_res.reason}")
                    debug_save(filt_res.struct or pre_res.struct, det2, pre_res.scale_ab, "geom_fail")
                    continue

                combined_meta: Dict[str, object] = {}
                combined_meta.update(pre_res.metrics)
                combined_meta.update(filt_res.metrics)
                if "pair_min_after" in combined_meta:
                    combined_meta["dmin_any"] = combined_meta["pair_min_after"]

                eval_res = evaluator.evaluate(filt_res.struct, sg, det2, pre_res.scale_ab, combined_meta)
                if not eval_res.ok:
                    append_log(log_path, f"[REJ][eval] det={det2}  {eval_res.reason}")
                    debug_save(filt_res.struct, det2, pre_res.scale_ab, "eval_fail")
                    continue

                prim = eval_res.primitive
                meta = eval_res.meta

                is_dup = any(matchers[sg].fit(prim, item["primitive"]) for item in found)
                if is_dup:
                    append_log(log_path, f"[SKIP] det={det2}  发现重复结构，跳过")
                    debug_save(prim, det2, pre_res.scale_ab, "dup")
                    continue

                found.append({"primitive": prim, "meta": meta})
                debug_save(prim, det2, pre_res.scale_ab, "accepted", accepted=True)

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
