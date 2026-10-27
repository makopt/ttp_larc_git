"""TSP tours for LARC: chained Lin-Kernighan tours from the linkern program of Concorde (bin/linkern), the paths of the
light-heavy re-sequencing move (split_tours), and one cached tour per TSP base in data/tours/<base>.npy (used by the
experiments on the constructor alone).

Tours depend only on the coordinates, so the cache key is the TSP base (e.g. 'eil76'), shared by the three instances of
a base. Cities are 0-based, tour[0] is the depot, and distances follow CEIL_2D as in the benchmark.
"""
import time

import numpy as np

from . import DATA_DIR
from .instance import LeanTTP

TOUR_DIR = DATA_DIR / "tours"


def _tsplib(inst: LeanTTP) -> str:
    lines = [f"NAME: {inst.name}", "TYPE: TSP", f"DIMENSION: {inst.n}", f"EDGE_WEIGHT_TYPE: {inst.edge_type}",
             "NODE_COORD_SECTION"]
    lines += [f"{i + 1} {x} {y}" for i, (x, y) in enumerate(inst.xy)]
    return "\n".join(lines + ["EOF"])


def base_key(inst: LeanTTP) -> str:
    return inst.name.replace("-TTP", "").replace("-ttp", "")


def get_tour(name: str, inst: LeanTTP, compute: bool = True):
    """Cached tour for the instance's TSP base; computed (and cached) if missing and compute=True."""
    f = TOUR_DIR / f"{base_key(inst)}.npy"
    if f.exists():
        return np.load(f)
    if not compute:
        return None
    tour, _ = linkern_tour(inst, time_limit=60)   # chained LK, same family as CoCo's tour generator; kicks = n
    TOUR_DIR.mkdir(exist_ok=True)
    np.save(f, tour)
    return tour


# ----------------------------------------------------------------------------- chained Lin-Kernighan (Concorde linkern)
LINKERN = DATA_DIR.parent / "bin" / "linkern"


def linkern_tour(inst: LeanTTP, time_limit: float = 10.0, seed: int = 1, kicks: int = None, init: int = None):
    """Chained LK (Applegate et al. 2003, the tour generator used by CoCo). Fast on 30k+ cities.
    Binary built from Concorde sources (bin/linkern). Returns (tour, wall_time)."""
    import subprocess
    import tempfile
    from pathlib import Path
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as d:
        f, o = Path(d) / "a.tsp", Path(d) / "o.txt"
        f.write_text(_tsplib(inst))
        cmd = [str(LINKERN), "-Q", "-s", str(seed), "-t", str(time_limit), "-o", str(o)]
        if kicks is not None:
            cmd += ["-R", str(kicks)]
        if init is not None:                       # starting cycle: 0 random, 1 nearest neighbour, 2 greedy, 3 Boruvka, 4 quick Boruvka
            cmd += ["-I", str(init)]
        subprocess.run(cmd + [str(f)], check=True, capture_output=True)
        edges = np.loadtxt(o, skiprows=1, dtype=np.int64)[:, :2]
    nxt = {}
    for a, b in edges:
        nxt.setdefault(a, []).append(b)
        nxt.setdefault(b, []).append(a)
    tour, prev, cur = [0], -1, 0
    while len(tour) < inst.n:
        a, b = nxt[cur]
        cur, prev = (b if a == prev else a), cur
        tour.append(cur)
    tour = np.array(tour)
    assert sorted(tour.tolist()) == list(range(inst.n)), "invalid tour"
    return tour, time.monotonic() - t0


def subset_path(inst: LeanTTP, cities: np.ndarray, seed: int = 1, time_limit: float = 5.0) -> np.ndarray:
    """Chained-LK cycle over `cities` plus the depot; returned as the open sequence of `cities` after the depot."""
    from .instance import LeanTTP as _L
    cities = np.asarray(cities, dtype=np.int64)
    nodes = np.concatenate(([0], cities))
    if len(nodes) < 4:
        return cities
    sub = _L(inst.name, len(nodes), 0, inst.capacity, inst.v_min, inst.v_max, inst.R, inst.edge_type,
             inst.xy[nodes], np.zeros(0), np.zeros(0), np.zeros(0, np.int64))
    t, _ = linkern_tour(sub, time_limit=time_limit, seed=seed)
    return nodes[t[1:]]


def split_tours(inst: LeanTTP, city_weight: np.ndarray, q: float, seed: int = 1, time_limit: float = 5.0):
    """Light-heavy re-sequencing: cities whose picked weight exceeds the q-quantile (of non-zero weights) are 'heavy'.
    Returns the 4 tours depot -> light path -> heavy path (each path in both orientations), or [] if degenerate."""
    w = city_weight[1:]
    nz = w[w > 0]
    if len(nz) == 0:
        return []
    thr = 0.0 if q <= 0 else float(np.quantile(nz, q))
    heavy = 1 + np.flatnonzero(w > thr)
    light = 1 + np.flatnonzero(w <= thr)
    if len(heavy) < 3 or len(light) < 3:
        return []
    A, B = subset_path(inst, light, seed, time_limit), subset_path(inst, heavy, seed, time_limit)
    return [np.concatenate(([0], a, b)) for a in (A, A[::-1]) for b in (B, B[::-1])]


def strata_tours(inst: LeanTTP, city_weight: np.ndarray, qs, seed: int = 1, time_limit: float = 5.0):
    """Generalized LHR with len(qs)+1 strata (light -> ... -> heavy) cut at the given quantiles of non-zero city weights.
    Orientation of each stratum path is chosen greedily (shorter join with the previous one); the returned list holds
    that tour and the variant with the last stratum reversed. [] if a stratum has fewer than 3 cities."""
    w = city_weight[1:]
    nz = w[w > 0]
    if len(nz) == 0:
        return []
    cuts = [0.0 if q <= 0 else float(np.quantile(nz, q)) for q in sorted(qs)]
    groups, lo = [], -np.inf
    for c in cuts + [np.inf]:
        g = 1 + np.flatnonzero((w > lo) & (w <= c)) if lo > -np.inf else 1 + np.flatnonzero(w <= c)
        groups.append(g)
        lo = c
    if any(len(g) < 3 for g in groups):
        return []
    paths = [subset_path(inst, g, seed, time_limit) for g in groups]
    seq, prev = [], 0
    for p in paths[:-1]:
        a, b_ = p, p[::-1]
        p = a if inst.dist(prev, a[0]) <= inst.dist(prev, b_[0]) else b_
        seq.append(p)
        prev = p[-1]
    last = paths[-1]
    return [np.concatenate([[0]] + seq + [l]) for l in (last, last[::-1])]
