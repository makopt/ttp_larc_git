"""Item selection on a fixed tour: ICA+ISA (Zhang et al. 2025), PW-ICA, load-aware PW-ICA, flip search (C3).

Item keys (sorted descending by ICA):
  'ica'  p/w                                            tour-blind (Zhang 2025)
  'pw'   p/w - R*nu*D(k)/vmax^2                         first-order, light load
  'la'   p/w - R*(nu*G1(k) + w*nu^2*G2(k))              second-order at the *actual* load profile, G1,G2 suffix sums of
                                                        d/v(c)^2 and d/v(c)^3 (Taylor of 1/v(c+w) around the current load)
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .evaluator import city_weights, objective
from .instance import LeanTTP

CHI_GRID = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
import os as _os
KP_REPACK = _os.environ.get("TTP_KP_REPACK", "1") == "1"   # knapsack repack in the constructor; TTP_KP_REPACK=0 turns it off


# ----------------------------------------------------------------------------- tour helpers
def reverse_tour(tour: np.ndarray) -> np.ndarray:
    return np.concatenate(([tour[0]], tour[:0:-1]))


def edge_data(inst: LeanTTP, tour: np.ndarray):
    """d[s] = length of the edge leaving position s; pos[city] = position in the tour."""
    d = inst.dist(tour, np.roll(tour, -1))
    pos = np.empty(inst.n, dtype=np.int64)
    pos[tour] = np.arange(inst.n)
    return d, pos


def _suffix(a: np.ndarray) -> np.ndarray:
    return np.cumsum(a[::-1])[::-1]


def item_keys(inst: LeanTTP, tour: np.ndarray, mode: str, picked: Optional[np.ndarray] = None) -> np.ndarray:
    dens = inst.profit / inst.weight
    if mode == "ica":
        return dens
    d, pos = edge_data(inst, tour)
    k = pos[inst.city]
    nu, R = inst.nu, inst.R
    if mode == "pw":
        return dens - R * nu * _suffix(d)[k] / inst.v_max ** 2
    if mode == "la":
        c = np.zeros(inst.n) if picked is None else np.cumsum(city_weights(inst, picked)[tour])
        v = np.maximum(inst.v_max - nu * c, inst.v_min)
        g1, g2 = _suffix(d / v ** 2), _suffix(d / v ** 3)
        return dens - R * (nu * g1[k] + inst.weight * nu ** 2 * g2[k])
    raise ValueError(mode)


# ----------------------------------------------------------------------------- ICA + ISA
def _windows(xy: np.ndarray, theta_g: float, theta_h: float):
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    g = max(min(x1 - x0, y1 - y0) * theta_g, 1e-9)
    hx, hy = max((x1 - x0) * theta_h, 1e-9), max((y1 - y0) * theta_h, 1e-9)
    out, x = [], x0
    while x + g <= x1 + hx:
        y = y0
        while y + g <= y1 + hy:
            out.append((x, y, g))
            y += hy
        x += hx
    return out


def ica_isa(inst: LeanTTP, key: np.ndarray, chi: float, theta_g: float = 1 / 3, theta_h: float = 1 / 10,
            scoring: str = "profit") -> np.ndarray:
    """Return a bool mask of selected items. scoring: 'items' (Zhang 2025), 'profit', 'cities'."""
    order = np.argsort(-key, kind="stable")
    w = inst.weight[order]
    # Type-1: by decreasing key while cumulative weight < chi*C (items that would break capacity go to Type-2)
    thr, t1 = chi * inst.capacity, np.zeros(inst.m, bool)
    cs = np.cumsum(w)
    # vectorized form of the sequential rule: item i is Type-1 iff weight before it < thr
    before = cs - w
    t1_sorted = (before < thr) & (cs <= inst.capacity)
    t1[order[t1_sorted]] = True
    picked = t1.copy()
    remaining = inst.capacity - inst.weight[t1].sum()
    if remaining <= 0:
        return picked
    # ISA window scores from Type-1 items
    if scoring == "profit":
        per_city = np.bincount(inst.city[t1], weights=inst.profit[t1], minlength=inst.n)
    elif scoring == "items":
        per_city = np.bincount(inst.city[t1], minlength=inst.n).astype(float)
    elif scoring == "cities":
        per_city = (np.bincount(inst.city[t1], minlength=inst.n) > 0).astype(float)
    else:
        raise ValueError(scoring)
    wins = _windows(inst.xy, theta_g, theta_h)
    x, y = inst.xy[:, 0], inst.xy[:, 1]
    masks, scores = [], []
    for (wx, wy, g) in wins:
        mk = (x >= wx) & (x <= wx + g) & (y >= wy) & (y <= wy + g)
        masks.append(mk)
        scores.append(per_city[mk].sum())
    t2 = order[~t1_sorted]                       # Type-2 items in decreasing key order
    for wi in np.argsort(-np.asarray(scores), kind="stable"):
        if remaining <= 0:
            break
        cand = t2[masks[wi][inst.city[t2]] & ~picked[t2]]
        while len(cand) and remaining > 0:
            cand = cand[inst.weight[cand] <= remaining]       # greedy in order, skipping items that do not fit
            if not len(cand):
                break
            cw = np.cumsum(inst.weight[cand])
            kk = int(np.searchsorted(cw, remaining, side="right"))
            if kk == 0:
                break
            picked[cand[:kk]] = True
            remaining -= cw[kk - 1]
            cand = cand[kk + 1:] if kk < len(cand) else cand[:0]
    return picked


# ----------------------------------------------------------------------------- C3 flip search
def flip_search(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray, max_pass: int = 10, top_k: int = 5000):
    """Single-item flips with exact delta (segment recomputation); candidates ranked by a 2nd-order estimate.
    Never decreases the objective."""
    picked = picked.copy()
    d, pos = edge_data(inst, tour)
    ipos = pos[inst.city]
    nu, R, vmax, vmin = inst.nu, inst.R, inst.v_max, inst.v_min
    load = np.cumsum(city_weights(inst, picked)[tour])
    wtot = inst.weight[picked].sum()
    for _ in range(max_pass):
        v = np.maximum(vmax - nu * load, vmin)
        g1, g2 = _suffix(d / v ** 2), _suffix(d / v ** 3)
        k = ipos
        cost_add = R * (nu * inst.weight * g1[k] + (nu * inst.weight) ** 2 * g2[k])
        save_rem = R * (nu * inst.weight * g1[k] - (nu * inst.weight) ** 2 * g2[k])
        approx = np.where(picked, -inst.profit + save_rem, inst.profit - cost_add)
        cand = np.argsort(-approx)[:top_k]
        improved = False
        for j in cand:
            kj, wj, sign = ipos[j], inst.weight[j], (-1.0 if picked[j] else 1.0)
            if sign > 0 and wtot + wj > inst.capacity:
                continue
            seg_d = d[kj:]
            c_old = load[kj:]
            c_new = c_old + sign * wj
            t_old = (seg_d / np.maximum(vmax - nu * c_old, vmin)).sum()
            t_new = (seg_d / np.maximum(vmax - nu * c_new, vmin)).sum()
            delta = sign * inst.profit[j] - R * (t_new - t_old)
            if delta > 1e-9:
                picked[j] = not picked[j]
                load[kj:] = c_new
                wtot += sign * wj
                improved = True
        if not improved:
            break
    return picked


def kp_repack(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray, core: int = 150, rounds: int = 3,
              max_cells: float = 2e8):
    """Knapsack-exact repack on a fixed tour. Each item gets an adjusted profit: its profit minus the second-order
    travel-time cost of carrying it at the current load profile (the flip-search estimate). The resulting 0-1 knapsack is
    solved with a core: items sorted by adjusted profit per weight, those well above the break item fixed in, those well
    below fixed out, and the `core` items on each side of the break item decided exactly by dynamic programming
    (kernels.kp_dp; weights scaled only if the table would exceed max_cells). Costs depend on the load, so this is
    repeated `rounds` times from the new packing, then polished by flip_search. Every packing is judged by its exact
    objective; the result is never worse than the input. With R = 0 it solves the knapsack exactly up to the core."""
    from .kernels import kp_dp
    d, pos = edge_data(inst, tour)
    ipos = pos[inst.city]
    nu, R, vmax, vmin = inst.nu, inst.R, inst.v_max, inst.v_min
    W = int(inst.capacity)
    w = inst.weight
    best_pk, best_val = picked.copy(), objective(inst, tour, picked)
    cur = picked.copy()
    for _ in range(rounds):
        load = np.cumsum(city_weights(inst, cur)[tour])
        v = np.maximum(vmax - nu * load, vmin)
        g1, g2 = _suffix(d / v ** 2), _suffix(d / v ** 3)
        nw = nu * w
        cost = np.where(cur, R * (nw * g1[ipos] - nw ** 2 * g2[ipos]), R * (nw * g1[ipos] + nw ** 2 * g2[ipos]))
        a = inst.profit - cost
        cand = np.flatnonzero(a > 0)
        order = cand[np.argsort(-(a[cand] / w[cand]))]
        cumw = np.cumsum(w[order])
        b = int(np.searchsorted(cumw, W, side="right"))
        new = np.zeros(inst.m, bool)
        if b >= len(order):
            new[order] = True
        else:
            lo, hi = max(0, b - core), min(len(order), b + core)
            new[order[:lo]] = True
            ci = order[lo:hi]
            capr = W - int(w[order[:lo]].sum())
            s = max(1, int(np.ceil(len(ci) * (capr + 1) / max_cells)))
            wi = np.ceil(w[ci] / s).astype(np.int64)
            x = kp_dp(wi, a[ci].astype(np.float64), capr // s)
            new[ci[x]] = True
        val = objective(inst, tour, new)
        if val > best_val:
            best_pk, best_val = new, val
        if np.array_equal(new, cur):
            break
        cur = new
    pk = flip_search(inst, tour, best_pk)
    return pk, objective(inst, tour, pk)


# ----------------------------------------------------------------------------- strategies
@dataclass
class Built:
    objective: float
    picked: np.ndarray
    tour: np.ndarray
    chi: float
    orientation: str
    mode: str


def empty_solution(inst, tour):
    p = np.zeros(inst.m, bool)
    return Built(objective(inst, tour, p), p, tour, 0.0, "fwd", "empty")


def build(inst: LeanTTP, tour: np.ndarray, mode: str = "pw", chis=CHI_GRID, both_orientations: bool = True,
          scoring: str = "profit", la_rounds: int = 3, polish: bool = False,
          theta_g: float = 1 / 3, theta_h: float = 1 / 10) -> Built:
    """Best (orientation, chi) solution of the given key mode; empty knapsack is always the floor (Strategy E)."""
    best = empty_solution(inst, tour)
    tours = [("fwd", tour)] + ([("rev", reverse_tour(tour))] if both_orientations else [])
    for name, tr in tours:
        prev = None
        rounds = la_rounds if mode == "la" else 1
        for _ in range(rounds):
            key = item_keys(inst, tr, mode, prev)
            round_best = None
            for chi in chis:
                pk = ica_isa(inst, key, chi, theta_g, theta_h, scoring)
                val = objective(inst, tr, pk)
                if round_best is None or val > round_best.objective:
                    round_best = Built(val, pk, tr, chi, name, mode)
            if round_best is None:
                break
            prev = round_best.picked
            if round_best.objective > best.objective:
                best = round_best
    if polish:
        pk = flip_search(inst, best.tour, best.picked)
        best = Built(objective(inst, best.tour, pk), pk, best.tour, best.chi, best.orientation, best.mode + "+c3")
        if KP_REPACK:                                   # knapsack-exact repack (kp_repack), never worse
            pk, val = kp_repack(inst, best.tour, best.picked)
            if val > best.objective:
                best = Built(val, pk, best.tour, best.chi, best.orientation, best.mode + "+kp")
    return best


def pack_iterative(inst: LeanTTP, tour: np.ndarray, thetas=(0.5, 1.0, 2.0), n_prefix: int = 40,
                   both_orientations: bool = True) -> Built:
    """PackIterative-style packing (after Faulkner et al. 2015): score p^theta / (w^theta * D) with D the distance left
    to the depot, take the best-scored prefix that fits, best over theta, prefix length and both orientations.
    Deliberately crude (no flips): used as a weak start that leaves room for coordinated tour moves."""
    best = None
    for name, tr in (("fwd", tour), ("rev", reverse_tour(tour)))[:2 if both_orientations else 1]:
        d, pos = edge_data(inst, tr)
        D = np.maximum(_suffix(d)[pos[inst.city]], 1e-9)
        for th in thetas:
            order = np.argsort(-(inst.profit ** th / (inst.weight ** th * D)))
            kmax = int(np.searchsorted(np.cumsum(inst.weight[order]), inst.capacity, side="right"))
            for k in np.unique(np.linspace(0, kmax, n_prefix).astype(int)):
                pk = np.zeros(inst.m, bool)
                pk[order[:k]] = True
                val = objective(inst, tr, pk)
                if best is None or val > best.objective:
                    best = Built(val, pk, tr, float(th), name, "packiter")
    return best
