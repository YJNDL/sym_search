#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
crys_modes.py

一个面向 2D 材料的“畸变模式 + 空间群搜索”小框架，
用于从高对称母相（比如 GaSb #39 Abm2）出发，
通过组合多种畸变模式，搜索可能的低对称子群（例如 #26 Pmc21 等）。

依赖:
    - numpy
    - spglib

使用方式:
    参考文件末尾的 main 示例。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Sequence

import numpy as np
import spglib  # 需要: pip install spglib


# ========== 工具函数 ==========

def compute_layer_thickness_and_vacuum(struct: Structure) -> Tuple[float, float]:
    """
    简单估算 2D 材料沿 c 方向的层厚和真空厚度。

    假设体系是标准 2D slab：c 轴大约垂直于层，原子主要集中在某个 Z 区间。
    做法：
      - 用分数坐标的 z 分量考虑周期边界，求出“最小包络区间”的厚度；
      - 厚度(Å) = thickness_frac * |c|
      - 真空(Å) = |c| - 厚度
    """
    lattice = struct.lattice
    coords = struct.coords % 1.0

    c_vec = lattice[2]
    c_len = float(np.linalg.norm(c_vec))

    # 分数坐标下 z∈[0,1)，考虑周期性，求最窄覆盖所有点的区间
    z = coords[:, 2] % 1.0
    z_sorted = np.sort(z)
    # 间隙：相邻点差值 + 尾首差值
    gaps = np.diff(np.concatenate([z_sorted, z_sorted[:1] + 1.0]))
    max_gap = float(np.max(gaps))
    thickness_frac = 1.0 - max_gap  # 最小覆盖区间长度（分数）
    thickness = thickness_frac * c_len
    vacuum = c_len - thickness
    return thickness, vacuum

def frac_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    分数坐标差 a - b，考虑周期性，返回 [-0.5, 0.5) 范围。
    """
    d = a - b
    return d - np.round(d)


def frac_add(a: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    分数坐标 a + d，结果 wrap 回 [0,1)。
    """
    x = a + d
    return x - np.floor(x)


def norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def unit_vector(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("零向量不能归一化")
    return v / n


# ========== 结构类 ==========

@dataclass
class Structure:
    lattice: np.ndarray       # (3,3)
    coords: np.ndarray        # (N,3) 分数坐标
    numbers: np.ndarray       # (N,) 原子序数
    species: List[str]        # 例如 ["Ga", "Sb"]
    counts: List[int]         # 例如 [4, 4]
    comment: str = "generated"
    scale: float = 1.0

    @classmethod
    def from_poscar(cls, filename: str) -> "Structure":
        path = Path(filename)
        lines = path.read_text().splitlines()
        if not lines:
            raise ValueError(f"空文件: {filename}")

        comment = lines[0].strip()
        scale = float(lines[1].split()[0])

        # 晶格
        lattice = np.array([[float(x) for x in lines[2 + i].split()] for i in range(3)], float) * scale

        species = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        natoms = sum(counts)

        idx = 7
        coord_mode_line = lines[idx].strip().lower()
        if coord_mode_line.startswith("s"):
            idx += 1
            coord_mode_line = lines[idx].strip().lower()

        coord_mode = coord_mode_line[0]  # 'd' / 'c' 等
        coord_start = idx + 1
        coord_lines = lines[coord_start:coord_start + natoms]
        if len(coord_lines) < natoms:
            raise ValueError("POSCAR 中原子坐标行数不足")

        coords = np.array([[float(x) for x in ln.split()[:3]] for ln in coord_lines], float)

        # 若是笛卡尔坐标，转成分数坐标
        if coord_mode == "c":
            inv_lat = np.linalg.inv(lattice.T)
            coords = coords @ inv_lat

        # 构造 numbers（可以按需补充元素表）
        element_Z: Dict[str, int] = {
            "H": 1, "He": 2,
            "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
            "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
            "K": 19, "Ca": 20,
            "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
            "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
            "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
            "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
        }
        numbers: List[int] = []
        for sp, n in zip(species, counts):
            if sp not in element_Z:
                raise KeyError(f"元素 {sp} 未在 element_Z 字典中定义，请手动补充原子序数。")
            numbers.extend([element_Z[sp]] * n)

        return cls(
            lattice=lattice,
            coords=coords,
            numbers=np.array(numbers, int),
            species=list(species),
            counts=list(counts),
            comment=comment,
            scale=scale,
        )

    def to_poscar(self, filename: str) -> None:
        coords = self.coords % 1.0
        lat = self.lattice / self.scale

        with open(filename, "w") as f:
            f.write(self.comment + "\n")
            f.write(f"  {self.scale:.16f}\n")
            for v in lat:
                f.write("  " + "  ".join(f"{x: .16f}" for x in v) + "\n")
            f.write("  " + "  ".join(self.species) + "\n")
            f.write("  " + "  ".join(str(n) for n in self.counts) + "\n")
            f.write("Direct\n")
            for r in coords:
                f.write("  " + "  ".join(f"{x: .16f}" for x in r) + "\n")

    # ---- spglib 接口 ----
    def spglib_cell(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.lattice, self.coords % 1.0, self.numbers)

    def get_symmetry_dataset(self, symprec: float = 1e-3):
        cell = self.spglib_cell()
        return spglib.get_symmetry_dataset(cell, symprec=symprec)

    def get_symmetry(self, symprec: float = 1e-3):
        cell = self.spglib_cell()
        return spglib.get_symmetry(cell, symprec=symprec)

    def copy(self) -> "Structure":
        return Structure(
            lattice=self.lattice.copy(),
            coords=self.coords.copy(),
            numbers=self.numbers.copy(),
            species=list(self.species),
            counts=list(self.counts),
            comment=str(self.comment),
            scale=float(self.scale),
        )


# ========== 找格心平移、配对 ==========

def find_centering_translations(symmetry, tol: float = 1e-5) -> List[np.ndarray]:
    """
    在 spglib.get_symmetry() 的结果里找所有“非平凡的纯平移”向量，
    也就是格心平移 (centering translations)。
    """
    rotations = symmetry["rotations"]
    translations = symmetry["translations"]

    cents: List[np.ndarray] = []
    for R, t in zip(rotations, translations):
        if np.all(R == np.eye(3, dtype=int)):
            t_red = t - np.round(t)
            if norm(t_red) < tol:
                continue
            # 规约到 [0,1)
            t_red = t_red % 1.0
            # 去重
            if not any(norm(frac_diff(t_red, c)) < tol for c in cents):
                cents.append(t_red)

    return cents


def pair_atoms_by_translation(
    positions: np.ndarray,
    numbers: np.ndarray,
    t: np.ndarray,
    tol: float = 1e-3,
) -> List[Tuple[int, int]]:
    """
    用一条平移 t，把原子按 (i, j ≈ i + t) 配成对。
    限制: 只配原子序数相同的原子。
    """
    n = len(positions)
    used = [False] * n
    pairs: List[Tuple[int, int]] = []

    for i in range(n):
        if used[i]:
            continue
        target = (positions[i] + t) % 1.0
        best_j = None
        for j in range(n):
            if i == j or used[j]:
                continue
            if numbers[j] != numbers[i]:
                continue
            diff = frac_diff(positions[j], target)
            if norm(diff) < tol:
                best_j = j
                break
        if best_j is not None:
            used[i] = used[best_j] = True
            pairs.append((i, best_j))

    return pairs


# ========== 畸变模式基类 ==========

@dataclass
class DistortionMode:
    """
    畸变模式基类: 给定母相 Structure 和幅度，返回所有原子的分数坐标增量。
    """
    name: str

    def delta(self, struct: Structure, amplitude: float, rng: np.random.RandomState) -> np.ndarray:
        """
        返回一个 (N,3) 的增量数组。
        """
        raise NotImplementedError


# ========== 具体模式 1：格心反相模式 ==========

@dataclass
class CenteringMode(DistortionMode):
    """
    针对每一条格心平移 t，找到配对 (i, j ≈ i + t)，
    对每一对施加 (+u, -u) 反相位移。
    """
    t: np.ndarray
    pairs: List[Tuple[int, int]]
    inplane_only: bool = True

    def delta(self, struct: Structure, amplitude: float, rng: np.random.RandomState) -> np.ndarray:
        n = len(struct.coords)
        d = np.zeros((n, 3), float)

        if amplitude == 0.0 or not self.pairs:
            return d

        for (i, j) in self.pairs:
            # 随机方向
            vec = rng.randn(3)
            if self.inplane_only:
                vec[2] = 0.0
            if norm(vec) < 1e-8:
                continue
            vec = unit_vector(vec)
            disp = amplitude * vec
            d[i] += disp
            d[j] -= disp

        return d


# ========== 具体模式 2：Wyckoff 轨道整体平移模式 ==========

@dataclass
class WyckoffShiftMode(DistortionMode):
    """
    对某一个 Wyckoff 轨道（equivalent_atoms == orbit_index 的那些原子），
    沿某个方向整体平移。
    """
    orbit_index: int
    direction: np.ndarray
    equivalent_atoms: np.ndarray  # 来自 dataset["equivalent_atoms"]
    inplane_only: bool = True

    def delta(self, struct: Structure, amplitude: float, rng: np.random.RandomState) -> np.ndarray:
        n = len(struct.coords)
        d = np.zeros((n, 3), float)
        if amplitude == 0.0:
            return d

        dir_vec = np.array(self.direction, float)
        if self.inplane_only:
            dir_vec[2] = 0.0
        if norm(dir_vec) < 1e-8:
            return d
        dir_vec = unit_vector(dir_vec)
        disp = amplitude * dir_vec

        for i in range(n):
            if self.equivalent_atoms[i] == self.orbit_index:
                d[i] += disp
        return d


# ========== 具体模式 3：全局随机扰动模式 ==========

@dataclass
class RandomMode(DistortionMode):
    """
    每个原子一个随机方向的位移（打破对称/模拟热扰动）。
    """
    inplane_only: bool = True

    def delta(self, struct: Structure, amplitude: float, rng: np.random.RandomState) -> np.ndarray:
        n = len(struct.coords)
        if amplitude == 0.0:
            return np.zeros((n, 3), float)
        d = rng.randn(n, 3)
        if self.inplane_only:
            d[:, 2] = 0.0
        # 归一化每个原子位移方向
        for i in range(n):
            v = d[i]
            l = np.linalg.norm(v)
            if l < 1e-8:
                # 防止全 0
                v = np.array([1.0, 0.0, 0.0])
                l = 1.0
            d[i] = v / l
        d *= amplitude
        return d


# ========== 具体模式 4：简单晶格应变模式（可选） ==========

@dataclass
class LatticeStrainMode(DistortionMode):
    """
    改变晶格常数（例如只拉伸 a, b，不动 c）。
    这里不直接返回 coords 的 delta，而是修改 lattice。
    因此它的 delta() 返回全 0，需要由 DistortionEngine 额外处理。
    """
    strain_matrix: np.ndarray  # 3x3, 单位矩阵附近
    scale_with_amplitude: bool = True

    def delta(self, struct: Structure, amplitude: float, rng: np.random.RandomState) -> np.ndarray:
        # 仅改变晶格，不改变分数坐标
        return np.zeros_like(struct.coords)

    def apply_lattice(self, lattice: np.ndarray, amplitude: float) -> np.ndarray:
        if amplitude == 0.0:
            return lattice
        if self.scale_with_amplitude:
            strain = np.eye(3) + amplitude * (self.strain_matrix - np.eye(3))
        else:
            strain = self.strain_matrix
        return strain @ lattice


# ========== DistortionEngine：组合模式并调用 spglib ==========

@dataclass
class DistortionEngine:
    base_struct: Structure
    modes: List[DistortionMode]
    symprec: float = 1e-3

    def apply_modes(
        self,
        amplitudes: Dict[str, float],
        seed: Optional[int] = None,
    ) -> Structure:
        """
        给定一个 {mode_name: amplitude} 字典，返回一个新的 Structure。
        这里所有模式都是相对于 base_struct 一次性叠加，而不是连续多步。
        """
        rng = np.random.RandomState(seed)
        new_struct = self.base_struct.copy()

        total_delta = np.zeros_like(new_struct.coords)
        # 先处理只作用在坐标上的模式
        for m in self.modes:
            amp = float(amplitudes.get(m.name, 0.0))
            if isinstance(m, LatticeStrainMode):
                continue
            d = m.delta(self.base_struct, amp, rng)
            total_delta += d

        new_struct.coords = frac_add(self.base_struct.coords, total_delta)

        # 然后单独处理 lattice strain 模式
        lat = self.base_struct.lattice.copy()
        for m in self.modes:
            amp = float(amplitudes.get(m.name, 0.0))
            if isinstance(m, LatticeStrainMode):
                lat = m.apply_lattice(lat, amp)
        new_struct.lattice = lat

        return new_struct

    def analyze_spacegroup(self, struct: Structure) -> Tuple[int, str]:
        ds = struct.get_symmetry_dataset(symprec=self.symprec)
        return int(ds["number"]), str(ds["international"])

    def random_search(
        self,
        target_sgs: Optional[Sequence[int]] = None,
        n_trials: int = 100,
        amp_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        stop_at_first_match: bool = False,
        verbose: bool = True,
        min_vacuum: Optional[float] = None,  # 新增：最小真空厚度（Å），None 表示不检查
    ) -> List[Tuple[Structure, Dict]]:
        """
        在给定模式组合 + 幅度范围下随机搜索，寻找指定空间群的候选结构。

        参数:
            target_sgs: 想要的空间群号列表；None 表示不筛选，只返回所有 trial。
            n_trials:   尝试次数。
            amp_ranges: dict: {mode_name: (amin, amax)}，未给出的模式默认 (0, 0.05)。
            stop_at_first_match: 若为 True 且设定了 target_sgs，则一找到匹配就停止。
            min_vacuum: 真空厚度下限（单位 Å）；若不为 None，则真空 < min_vacuum 的结构直接跳过。

        返回:
            [(structure, symmetry_dataset), ...]
        """
        rng = np.random.RandomState(seed)
        results: List[Tuple[Structure, Dict]] = []
        if amp_ranges is None:
            amp_ranges = {}

        for it in range(n_trials):
            amps: Dict[str, float] = {}
            for m in self.modes:
                amin, amax = amp_ranges.get(m.name, (0.0, 0.05))
                if amin == amax == 0.0:
                    amps[m.name] = 0.0
                else:
                    amps[m.name] = rng.uniform(amin, amax)

            trial_seed = rng.randint(0, 2**31 - 1)
            s_trial = self.apply_modes(amps, seed=int(trial_seed))

            # --- 真空检查：如果真空不够，直接跳过这个 trial ---
            if min_vacuum is not None:
                thickness, vacuum = compute_layer_thickness_and_vacuum(s_trial)
                if vacuum < min_vacuum:
                    if verbose:
                        print(
                            f"[trial {it:03d}] vacuum = {vacuum:6.3f} Å < {min_vacuum:6.3f} Å, skip"
                        )
                    continue

            ds = s_trial.get_symmetry_dataset(symprec=self.symprec)
            sg_num = int(ds["number"])
            sg_name = str(ds["international"])

            hit = (target_sgs is None) or (sg_num in target_sgs)
            if verbose:
                print(f"[trial {it:03d}] SG = {sg_num:3d} ({sg_name:>6s}), hit={hit}")

            if hit:
                results.append((s_trial, ds))
                if target_sgs is not None and stop_at_first_match:
                    break

        return results


# ========== 构造“默认模式集合”的辅助函数 ==========

def build_default_modes(
    struct: Structure,
    symprec: float = 1e-3,
    inplane_only: bool = True,
    include_lattice_strain: bool = False,
) -> List[DistortionMode]:
    """
    给定一个母相结构，自动构造一组“合理的”畸变模式：
      - 对每一条格心平移 -> 一个 CenteringMode
      - 对每个 Wyckoff 轨道沿 x/y(/z) -> WyckoffShiftMode
      - 一个全局 RandomMode
      - 可选：若 include_lattice_strain，则加几个简单的晶格拉伸模式
    """
    modes: List[DistortionMode] = []

    # 1) 格心反相模式
    symmetry = struct.get_symmetry(symprec=symprec)
    cents = find_centering_translations(symmetry)
    for idx, t in enumerate(cents):
        positions = struct.coords % 1.0
        pairs = pair_atoms_by_translation(positions, struct.numbers, t)
        if not pairs:
            continue
        m = CenteringMode(
            name=f"centering_{idx}",
            t=t,
            pairs=pairs,
            inplane_only=inplane_only,
        )
        modes.append(m)

    # 2) Wyckoff 轨道整体平移模式
    ds = struct.get_symmetry_dataset(symprec=symprec)
    eq = np.array(ds["equivalent_atoms"], int)
    orbit_ids = sorted(set(eq.tolist()))
    for orb in orbit_ids:
        # x/y 方向
        modes.append(
            WyckoffShiftMode(
                name=f"orb{orb}_shift_x",
                orbit_index=orb,
                direction=np.array([1.0, 0.0, 0.0]),
                equivalent_atoms=eq,
                inplane_only=inplane_only,
            )
        )
        modes.append(
            WyckoffShiftMode(
                name=f"orb{orb}_shift_y",
                orbit_index=orb,
                direction=np.array([0.0, 1.0, 0.0]),
                equivalent_atoms=eq,
                inplane_only=inplane_only,
            )
        )
        if not inplane_only:
            modes.append(
                WyckoffShiftMode(
                    name=f"orb{orb}_shift_z",
                    orbit_index=orb,
                    direction=np.array([0.0, 0.0, 1.0]),
                    equivalent_atoms=eq,
                    inplane_only=False,
                )
            )

    # 3) 全局随机模式
    modes.append(RandomMode(name="random", inplane_only=inplane_only))

    # 4) 简单晶格应变
    if include_lattice_strain:
        # 只拉伸 a / b / ab 同比
        I = np.eye(3)
        for label, mat in [
            ("strain_a", np.diag([1.01, 1.0, 1.0])),
            ("strain_b", np.diag([1.0, 1.01, 1.0])),
            ("strain_ab", np.diag([1.01, 1.01, 1.0])),
        ]:
            modes.append(
                LatticeStrainMode(
                    name=label,
                    strain_matrix=mat,
                    scale_with_amplitude=True,
                )
            )

    return modes


# ========== 示例：针对 GaSb_39 做 39 -> 26 搜索 ==========

def example_gasb_39_to_26():
    """
    一个最接近你需求的示例：
    - 从 GaSb_39.vasp 读入母相 (#39 Abm2)
    - 自动构造模式集合（包含格心反相 + Wyckoff shift + random）
    - 使用 DistortionEngine 随机搜索，目标空间群 = 26
    - 找到第一个命中的结构后写出 POSCAR
    """
    parent = Structure.from_poscar("GaSb_39.vasp")

    ds0 = parent.get_symmetry_dataset()
    print("母相空间群:", ds0["number"], ds0["international"])

    modes = build_default_modes(parent, symprec=1e-3, inplane_only=True, include_lattice_strain=True)
    print("构造的模式数:", len(modes))
    print("模式列表:")
    for m in modes:
        print("  -", m.name, "(", type(m).__name__, ")")

    engine = DistortionEngine(base_struct=parent, modes=modes, symprec=1e-1)

    # 为不同模式设置一个大致合理的幅度范围（分数坐标单位）
    amp_ranges: Dict[str, Tuple[float, float]] = {}
    for m in modes:
        if isinstance(m, CenteringMode):
            amp_ranges[m.name] = (0.01, 0.10)
        elif isinstance(m, WyckoffShiftMode):
            amp_ranges[m.name] = (0.0, 0.05)
        elif isinstance(m, RandomMode):
            amp_ranges[m.name] = (0.0, 0.02)
        elif isinstance(m, LatticeStrainMode):
            amp_ranges[m.name] = (0.0, 0.02)

    # 目标: 26 号空间群
    target_sgs = [26]

    results = engine.random_search(
        target_sgs=target_sgs,
        n_trials=30000,
        amp_ranges=amp_ranges,
        seed=12345,
        stop_at_first_match=True,
        verbose=True,
    )

    if not results:
        print("在当前搜索参数下没有找到空间群 26 的结构，可以加大 n_trials 或调整幅度范围再试。")
        return

    best_struct, ds_best = results[0]
    print("找到一个空间群 26 的候选结构:", ds_best["number"], ds_best["international"])

    best_struct.comment = "GaSb candidate (from #39 -> #26 via random modes)"
    best_struct.to_poscar("GaSb_26_candidate.vasp")
    print("已写出 GaSb_26_candidate.vasp")


if __name__ == "__main__":
    # 把这里当成一个示例入口，你可以按需要修改或拆分成模块使用。
    example_gasb_39_to_26()
