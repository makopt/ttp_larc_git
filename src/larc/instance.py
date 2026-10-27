"""A TTP instance as NumPy arrays, read from its JSON file (instances with up to 33,810 cities and 338,090 items).
Distances are computed on demand, so no distance matrix is stored. Cities are numbered from 0 (depot = 0), and the item
arrays are sorted by item number.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LeanTTP:
    name: str
    n: int
    m: int
    capacity: float
    v_min: float
    v_max: float
    R: float
    edge_type: str
    xy: np.ndarray            # (n, 2) coordinates
    profit: np.ndarray        # (m,)
    weight: np.ndarray        # (m,)
    city: np.ndarray          # (m,) 0-based city of each item

    def dist(self, i, j):
        """Distance between city arrays/scalars i and j (vectorized). Only CEIL_2D/EUC_2D are supported."""
        d = np.hypot(self.xy[i, 0] - self.xy[j, 0], self.xy[i, 1] - self.xy[j, 1])
        if self.edge_type == "CEIL_2D":
            return np.ceil(d)
        if self.edge_type == "EUC_2D":
            return np.rint(d)
        raise NotImplementedError(self.edge_type)

    def tour_length(self, tour) -> float:
        t = np.asarray(tour)
        return float(self.dist(t, np.roll(t, -1)).sum())

    @property
    def nu(self) -> float:
        return (self.v_max - self.v_min) / self.capacity


def load_lean(path) -> LeanTTP:
    d = json.loads(Path(path).read_text())
    nodes = sorted(d["nodes"], key=lambda r: r["index"])
    items = sorted(d["items"], key=lambda r: r["index"])
    return LeanTTP(
        name=d["problem_name"], n=d["dimension"], m=d["number_of_items"],
        capacity=float(d["capacity_of_knapsack"]), v_min=float(d["min_speed"]), v_max=float(d["max_speed"]),
        R=float(d["renting_ratio"]), edge_type=d["edge_weight_type"],
        xy=np.array([[r["x"], r["y"]] for r in nodes], dtype=float),
        profit=np.array([r["profit"] for r in items], dtype=float),
        weight=np.array([r["weight"] for r in items], dtype=float),
        city=np.array([r["assigned_node"] - 1 for r in items], dtype=np.int64),
    )
