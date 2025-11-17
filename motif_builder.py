import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Lattice, Specie, Structure


class MotifBuilder:
    """Generate lattice-free motifs validated by CrystalNN before embedding.

    The builder samples atom positions in a large virtual box (no real lattice),
    scores them with CrystalNN-based coordination heuristics, and returns the
    best motif that satisfies hard-distance constraints.
    """

    def __init__(
        self,
        args,
        ref_stats: dict,
        cn_range: Tuple[int, int],
        center_species: str,
        neighbor_species: str,
        crystalnn: Optional[CrystalNN] = None,
    ):
        self.args = args
        self.ref_stats = ref_stats
        self.cn_lo, self.cn_hi = cn_range
        self.center_species = center_species
        self.neighbor_species = neighbor_species
        self.cnn = crystalnn or CrystalNN()

        self.target_nn = float(ref_stats.get("median", ref_stats.get("mean", 2.5)))
        self.hard_dmin = float(getattr(args, "motif_hard_dmin", 1.5))
        self.box_scale = float(getattr(args, "motif_box_scale", 6.0))
        box_size_val = getattr(args, "motif_box_size", 0.0)
        self.box_size = float(box_size_val) if box_size_val not in (None, "None") else 0.0
        self.max_tries = int(getattr(args, "motif_max_tries", 50))
        self.max_cost = float(getattr(args, "motif_max_cost", 10.0))
        self.local_steps = int(getattr(args, "motif_local_steps", 10))
        self.perturb_sigma = float(getattr(args, "motif_perturb_sigma", 0.15))
        self.short_dist_factor = float(getattr(args, "motif_short_dist_factor", 0.7))
        self.w_cn = float(getattr(args, "motif_w_cn", 1.0))
        self.w_d = float(getattr(args, "motif_w_d", 1.0))

    def build_motif(
        self, num_center: int, num_neighbor: int
    ) -> Optional[Tuple[np.ndarray, List[Specie], Dict]]:
        """
        Build a single motif (cluster) as Cartesian coordinates without a real lattice.

        Returns:
            coords_cart: (N, 3) Cartesian coordinates
            species: list of pymatgen species
            meta: motif statistics such as cost and distance summary
        """

        if num_center <= 0 or num_neighbor <= 0:
            return None

        species: List[Specie] = [Specie(self.center_species)] * num_center
        species += [Specie(self.neighbor_species)] * num_neighbor

        best = None
        box_size = self.box_size if self.box_size > 0 else self.target_nn * self.box_scale
        box_size = max(box_size, self.hard_dmin * 4)

        for _ in range(self.max_tries):
            coords = self._random_layout(len(species), box_size)
            if coords is None:
                continue

            struct = self._cluster_structure(coords, species, box_size)
            cost, meta = self._score_motif(struct)
            if math.isfinite(cost):
                coords_best = coords
                cost_best = cost
                meta_best = meta
            else:
                continue

            for _ in range(max(0, self.local_steps)):
                perturbed = self._perturb(coords_best, box_size)
                if perturbed is None:
                    continue
                struct_pert = self._cluster_structure(perturbed, species, box_size)
                cost_new, meta_new = self._score_motif(struct_pert)
                if cost_new < cost_best:
                    coords_best = perturbed
                    cost_best = cost_new
                    meta_best = meta_new

            if cost_best <= self.max_cost:
                return coords_best, species, {**meta_best, "cost": cost_best}

            if best is None or cost_best < best[0]:
                best = (cost_best, coords_best, meta_best)

        if best is None:
            return None
        return best[1], species, {**best[2], "cost": best[0]}

    def _random_layout(self, n: int, box_size: float) -> Optional[np.ndarray]:
        coords: List[np.ndarray] = []
        span = max(box_size * 0.25, self.target_nn)
        max_place_tries = 30

        for _ in range(n):
            placed = False
            for _ in range(max_place_tries):
                x = random.uniform(-span, span)
                y = random.uniform(-span, span)
                z = random.uniform(-0.05 * self.target_nn, 0.05 * self.target_nn)
                cand = np.array([x, y, z], dtype=float)
                if all(np.linalg.norm(cand - p) >= self.hard_dmin for p in coords):
                    coords.append(cand)
                    placed = True
                    break
            if not placed:
                return None

        return np.array(coords, dtype=float)

    def _perturb(self, coords: np.ndarray, box_size: float) -> Optional[np.ndarray]:
        new_coords = coords.copy()
        idx = random.randrange(len(coords))
        sigma = self.perturb_sigma * self.target_nn
        delta = np.random.normal(scale=sigma, size=3)
        delta[2] *= 0.2  # keep near-plane
        new_coords[idx] += delta

        # keep inside a loose bounding box
        limit = 0.5 * box_size
        new_coords[idx] = np.clip(new_coords[idx], -limit, limit)

        for i in range(len(new_coords)):
            for j in range(i + 1, len(new_coords)):
                if np.linalg.norm(new_coords[i] - new_coords[j]) < self.hard_dmin:
                    return None
        return new_coords

    def _cluster_structure(self, coords: np.ndarray, species: List[Specie], box_size: float) -> Structure:
        # center coordinates and shift into middle of box to avoid periodic artifacts
        centered = coords - np.mean(coords, axis=0)
        shift = np.array([0.5 * box_size, 0.5 * box_size, 0.5 * box_size])
        coords_box = centered + shift
        lattice = Lattice.cubic(box_size)
        return Structure(lattice, species, coords_box, coords_are_cartesian=True)

    def _score_motif(self, struct: Structure) -> Tuple[float, Dict[str, float]]:
        center_indices = [i for i, s in enumerate(struct) if s.species_string.lower() == self.center_species.lower()]
        if not center_indices:
            return math.inf, {}

        cn_penalty = 0.0
        dist_penalty = 0.0
        short_penalty = 0.0
        d_means: List[float] = []

        for idx in center_indices:
            try:
                nn_info = self.cnn.get_nn_info(struct, idx)
                cn = self.cnn.get_cn(struct, idx)
            except Exception:
                return math.inf, {}

            if cn < self.cn_lo:
                cn_penalty += (self.cn_lo - cn) ** 2
            elif cn > self.cn_hi:
                cn_penalty += (cn - self.cn_hi) ** 2

            neigh_d = []
            for item in nn_info:
                site = item.get("site")
                if site is None:
                    site = struct[int(item.get("site_index", -1))]
                if site is None:
                    continue
                if site.species_string.lower() != self.neighbor_species.lower():
                    continue
                d = float(item.get("weight", 0.0))
                if d <= 0 and "site_index" in item:
                    j = int(item["site_index"])
                    d = struct.get_distance(idx, j)
                neigh_d.append(d)

            if neigh_d:
                d_mean = float(np.mean(neigh_d))
                d_means.append(d_mean)
                ref_std = max(self.ref_stats.get("std", 0.1), 1e-3)
                dist_penalty += ((d_mean - self.target_nn) ** 2) / (ref_std ** 2)
                too_close = [d for d in neigh_d if d < self.target_nn * self.short_dist_factor]
                if too_close:
                    short_penalty += 10.0 * len(too_close)
            else:
                dist_penalty += 10.0

        total_cost = self.w_cn * cn_penalty + self.w_d * dist_penalty + short_penalty
        meta = {
            "cn_penalty": cn_penalty,
            "dist_penalty": dist_penalty,
            "short_penalty": short_penalty,
            "d_mean": float(np.mean(d_means)) if d_means else None,
            "d_std": float(np.std(d_means)) if len(d_means) > 1 else 0.0,
        }
        return total_cost, meta
