"""Exact best packing on a fixed tour, compared with the packings of LARC (Section 3.2.3).

For a fixed tour, the packing problem of the TTP (packing while traveling) is solved exactly by dynamic programming over
the carried weight: the tour is followed from the depot, f[w] is the best value (profit minus rent) of a packing of the
cities already visited that carries the weight w, the items of a city are added as 0-1 knapsack choices, and every edge
subtracts R d / (v_max - nu w) from f[w]. The optimum is max_w f[w]. This uses the exact nonlinear objective, not the
additive surrogate of the knapsack repack, and costs O((n + m) W) time and O(W) memory.

For each final solution of LARC (tour and packing), the script records the value of:
  - the final packing of LARC;
  - the knapsack repack of LARC started from the empty packing on the same tour (three rounds, then the flip search),
    which shows what the additive surrogate reaches alone;
  - the exact optimum on the same tour (same direction).
On the smaller instances ((n + m) W at most --check-ops), the optimal packing is also rebuilt and evaluated by the
evaluator of LARC, as a check of the dynamic program.

Usage: python3 scripts/experiments/run_exact_packing.py <folder of solutions> [--max-ops 1.3e11] [--workers 8]
       The folder holds one file <instance>_s<seed>.npz per solution, with the arrays tour (cities, depot first) and
       picked (bool per item), as written by the runs of LARC with saved solutions.
Output: runs/exact_packing.xlsx (one row per solution).
"""
import argparse
from pathlib import Path

import numpy as np
from numba import njit

from common import RUNS_DIR, run_jobs

ap = argparse.ArgumentParser()
ap.add_argument("solutions", type=Path)
ap.add_argument("--max-ops", type=float, default=1.3e11, help="only instances with (n + m) * W at most this value")
ap.add_argument("--check-ops", type=float, default=2e8, help="rebuild and check the optimal packing below this size")
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()


@njit(cache=True)
def _dp_value(tour, istart, iorder, weight, profit, dist, W, R, vmax, nu):
    f = np.full(W + 1, -1e300)
    f[0] = 0.0
    n = len(tour)
    for k in range(n):
        c = tour[k]
        for r in range(istart[c], istart[c + 1]):
            it = iorder[r]
            wt = weight[it]
            p = profit[it]
            for w in range(W, wt - 1, -1):
                v = f[w - wt] + p
                if v > f[w]:
                    f[w] = v
        d = dist[k]
        if d != 0.0:
            for w in range(W + 1):
                if f[w] > -1e299:
                    f[w] -= R * d / (vmax - nu * w)
    return f.max()


@njit(cache=True)
def _dp_packing(tour, istart, iorder, weight, profit, dist, W, R, vmax, nu, m):
    f = np.full(W + 1, -1e300)
    f[0] = 0.0
    take = np.zeros((m, W + 1), dtype=np.bool_)
    order = np.empty(m, dtype=np.int64)
    no = 0
    n = len(tour)
    for k in range(n):
        c = tour[k]
        for r in range(istart[c], istart[c + 1]):
            it = iorder[r]
            order[no] = it
            wt = weight[it]
            p = profit[it]
            for w in range(W, wt - 1, -1):
                v = f[w - wt] + p
                if v > f[w]:
                    f[w] = v
                    take[no, w] = True
            no += 1
        d = dist[k]
        if d != 0.0:
            for w in range(W + 1):
                if f[w] > -1e299:
                    f[w] -= R * d / (vmax - nu * w)
    w = int(np.argmax(f))
    best = f[w]
    picked = np.zeros(m, dtype=np.bool_)
    for e in range(no - 1, -1, -1):
        if take[e, w]:
            picked[order[e]] = True
            w -= weight[order[e]]
    return best, picked


def run(job):
    name, seed, path = job
    from larc import benchmark, construct
    from larc.evaluator import objective
    inst = benchmark.load_instance(name)
    z = np.load(path)
    tour, picked = z["tour"].astype(np.int64), z["picked"].astype(bool)
    order = np.argsort(inst.city, kind="stable")
    istart = np.searchsorted(inst.city[order], np.arange(inst.n + 1)).astype(np.int64)
    closed = np.r_[tour, tour[:1]]
    dist = inst.dist(closed[:-1], closed[1:]).astype(np.float64)
    W = int(inst.capacity)
    args = (tour, istart, order.astype(np.int64), inst.weight.astype(np.int64), inst.profit.astype(np.float64), dist,
            W, float(inst.R), float(inst.v_max), float(inst.nu))
    row = dict(instance=name, seed=int(seed), n_cities=inst.n, n_items=inst.m, capacity=W,
               larc_final=objective(inst, tour, picked))
    _, row["repack_from_empty"] = construct.kp_repack(inst, tour, np.zeros(inst.m, bool))
    if (inst.n + inst.m) * W <= A.check_ops:
        row["exact_optimum"], best = _dp_packing(*args, inst.m)
        row["check_rebuilt_value"] = objective(inst, tour, best)
        row["check_rebuilt_feasible"] = bool(inst.weight[best].sum() <= W)
    else:
        row["exact_optimum"] = _dp_value(*args)
    for k in ("larc_final", "repack_from_empty"):
        row["gap_" + k + "_pct"] = 100 * (row["exact_optimum"] - row[k]) / abs(row["exact_optimum"])
    return row


if __name__ == "__main__":
    from larc import benchmark
    size = {r["instance"]: (int(r["n_cities"]) + int(r["n_items"])) for r in benchmark.instance_list()}
    jobs = []
    for f in sorted(A.solutions.glob("*_s*.npz"), key=lambda p: -size.get(p.stem.rsplit("_s", 1)[0], 0)):
        name = f.stem.rsplit("_s", 1)[0]
        if name not in size:
            continue
        W = benchmark.load_instance(name).capacity
        if size[name] * W <= A.max_ops:
            jobs.append((name, int(f.stem.rsplit("_s", 1)[1]), f))
    run_jobs(run, jobs, RUNS_DIR / "exact_packing.xlsx", ("instance", "seed"), A.workers,
             lambda r: f"{r['instance']} s{r['seed']}: gap of LARC {r['gap_larc_final_pct']:.4f}%")
