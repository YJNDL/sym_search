#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import csv

# ===== 用户配置 =====
STRUCTURE_PATH = "POSCAR"     # 或 "xxx.cif"
SITE_INDEX     = None         # None=全部位点；或给整数如 0
THRESHOLD      = 0.8
MOTIFS = [
    "tetrahedral", "square planar", "octahedral",
    "trigonal planar", "trigonal pyramidal",
    "see-saw-like", "rectangular see-saw-like",
    "square pyramidal", "trigonal bipyramidal",
]
ADD_OXIDATION_STATES = True
OUTPUT_CSV = "lostops_result.csv"
# ===================

def _norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("_", "").replace("-", "")

def _find_key(op_dict, motif):
    t = _norm(motif)
    for k, v in op_dict.items():
        kn = _norm(k)
        if kn == t:
            return k, v
        if t in ("squareplanar","squarecoplanar") and kn in ("squareplanar","squarecoplanar"):
            return k, v
    return None, None

def run(structure_path, site_index=None, threshold=0.8, motifs=None, outfile="lostops_result.csv"):
    motifs = motifs or ["tetrahedral", "square planar", "octahedral"]
    struct = Structure.from_file(structure_path).copy()
    if ADD_OXIDATION_STATES:
        try: struct.add_oxidation_state_by_guess()
        except Exception: pass

    cnn = CrystalNN()
    indices = range(len(struct)) if site_index is None else [site_index]

    with open(outfile, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["site","specie","cn","motif","op","threshold","match"])  # 表头

        for i in indices:
            cn_val = cnn.get_cn(struct, i)
            op_dict = cnn.get_local_order_parameters(struct, i) or {}
            for motif in motifs:
                key, val = _find_key(op_dict, motif)
                if key is None or val is None:
                    # 找不到该构型对应键名：可以选择跳过或写空值
                    continue
                match = bool(val >= threshold)
                w.writerow([i, str(struct[i].specie), f"{cn_val:.3f}", key, f"{val:.3f}", f"{threshold:.3f}", match])

    print(f"✅ 已写出：{outfile}")

if __name__ == "__main__":
    run(STRUCTURE_PATH, SITE_INDEX, THRESHOLD, MOTIFS, OUTPUT_CSV)
