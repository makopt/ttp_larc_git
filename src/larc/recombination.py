"""Recombination of two TTP solutions (Section 3.5.3 of the paper).

The union of two parent tours minus their common edges splits into connected components. The partition crossover of
Whitley et al. for the TSP can swap a component between parents when the component is entered and left by exactly two
common edges. The transplant used by LARC (ttp_transplant) works on every component: the edges of the base parent inside
a component are replaced by those of the donor, which keeps every city at degree two but may leave several cycles; the
cycles are merged by the cheapest exchange of two edges (in the spirit of Karp's patching). Components are tried largest
first, the items of the affected cities are re-decided by exact flips, both directions of the child are evaluated, and a
component is kept only if the exact TTP objective improves. recombine() applies the transplant between a solution and a
pool of other solutions.
"""
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .evaluator import objective
from .instance import LeanTTP


def _edge_keys(t: np.ndarray, n: int) -> np.ndarray:
    u, v = t, np.roll(t, -1)
    return np.minimum(u, v) * n + np.maximum(u, v)


def partition(A: np.ndarray, B: np.ndarray):
    """Return (common_keys, onlyA_keys, onlyB_keys, label, feasible_components)."""
    n = len(A)
    eA, eB = _edge_keys(A, n), _edge_keys(B, n)
    common = np.intersect1d(eA, eB)
    onlyA, onlyB = np.setdiff1d(eA, common), np.setdiff1d(eB, common)
    diff = np.concatenate([onlyA, onlyB])
    u, v = diff // n, diff % n
    g = csr_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    has = np.zeros(n, bool)
    has[u] = True
    has[v] = True
    # cut edges of A between different labels (they are common edges by construction)
    a, b = A, np.roll(A, -1)
    cut = lab[a] != lab[b]
    cnt = np.bincount(np.concatenate([lab[a[cut]], lab[b[cut]]]), minlength=lab.max() + 1)
    comps = np.unique(lab[has])
    feasible = [c for c in comps if cnt[c] == 2]
    return common, onlyA, onlyB, lab, feasible


def _cycle_from_keys(keys: np.ndarray, n: int):
    u, v = keys // n, keys % n
    adj = -np.ones((n, 2), np.int64)
    deg = np.zeros(n, np.int64)
    for x, y in zip(u, v):
        if deg[x] >= 2 or deg[y] >= 2:
            return None
        adj[x, deg[x]] = y
        deg[x] += 1
        adj[y, deg[y]] = x
        deg[y] += 1
    if (deg != 2).any():
        return None
    tour = np.empty(n, np.int64)
    tour[0], prev, cur = 0, -1, 0
    for k in range(1, n):
        a, b = adj[cur]
        nxt = b if a == prev else a
        prev, cur = cur, nxt
        if cur == 0:
            return None                              # closed early: not Hamiltonian
        tour[k] = cur
    return tour


def _best_orientation(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray):
    rev = np.concatenate(([tour[0]], tour[:0:-1]))
    a, b = objective(inst, tour, picked), objective(inst, rev, picked)
    return (tour, a) if a >= b else (rev, b)


def ttp_gpx(inst: LeanTTP, P: np.ndarray, Q: np.ndarray, picked: np.ndarray, rng=None):
    """Child of base parent P and donor Q: start from P's edges; for each feasible component (largest first) take Q's
    internal path if it improves the TTP objective with the items `picked` fixed. Returns (tour, value, n_feasible,
    n_taken) or None when no component is feasible."""
    n = len(P)
    common, onlyP, onlyQ, lab, feas = partition(P, Q)
    if not feas:
        return None
    labP, labQ = lab[onlyP // n], lab[onlyQ // n]
    sizes = np.bincount(lab)
    order = sorted(feas, key=lambda c: -sizes[c])
    use_q = set()
    cur_tour, cur_val = _best_orientation(inst, P, picked)
    taken = 0
    for c in order:
        trial = use_q | {c}
        keys = np.concatenate([common, onlyP[~np.isin(labP, list(trial))], onlyQ[np.isin(labQ, list(trial))]])
        t = _cycle_from_keys(keys, n)
        if t is None:
            continue
        t, val = _best_orientation(inst, t, picked)
        if val > cur_val + 1e-9:
            use_q, cur_tour, cur_val = trial, t, val
            taken += 1
    return cur_tour, cur_val, len(feas), taken


# ----------------------------------------------------------------------------- transplant with patching (all components)
def _cycles_from_keys(keys: np.ndarray, n: int):
    """Decompose a 2-regular edge set into cycles; returns (list of cycles as arrays, adjacency) or None if not 2-regular."""
    u, v = keys // n, keys % n
    deg = np.bincount(np.concatenate([u, v]), minlength=n)
    if (deg != 2).any():
        return None
    adj = -np.ones((n, 2), np.int64)
    fill = np.zeros(n, np.int64)
    for x, y in zip(u, v):
        adj[x, fill[x]] = y
        fill[x] += 1
        adj[y, fill[y]] = x
        fill[y] += 1
    seen = np.zeros(n, bool)
    cycles = []
    for s in range(n):
        if seen[s]:
            continue
        cyc, prev, cur = [s], -1, s
        seen[s] = True
        while True:
            a, b = adj[cur]
            nxt = b if a == prev else a
            if nxt == s:
                break
            prev, cur = cur, nxt
            seen[cur] = True
            cyc.append(cur)
        cycles.append(np.array(cyc, np.int64))
    return cycles


def _patch(inst: LeanTTP, cycles, nbr: np.ndarray):
    """Merge cycles into one tour with cheapest 2-exchanges searched through k-nearest lists (smallest cycle first)."""
    cycles = [c for c in cycles]
    n = inst.n
    while len(cycles) > 1:
        cycles.sort(key=len)
        S = cycles[0]
        cid = np.empty(n, np.int64)
        posin = np.empty(n, np.int64)
        for k, c in enumerate(cycles):
            cid[c] = k
            posin[c] = np.arange(len(c))
        best = None
        for ia, a in enumerate(S):
            b = S[(ia + 1) % len(S)]
            dab = inst.dist(a, b)
            for c in nbr[a]:
                k = cid[c]
                if k == 0:
                    continue
                T = cycles[k]
                ic = posin[c]
                for d in (T[(ic + 1) % len(T)], T[ic - 1]):
                    # remove (a,b),(c,d); add (a,c),(b,d)
                    delta = inst.dist(a, c) + inst.dist(b, d) - dab - inst.dist(c, d)
                    if best is None or delta < best[0]:
                        best = (delta, ia, k, ic, d == T[(ic + 1) % len(T)])
        if best is None:                                  # no neighbour in another cycle: join to the nearest cycle by brute force
            k = 1
            best = (0.0, 0, k, 0, True)
        _, ia, k, ic, fwd = best
        T = cycles[k]
        Sr = np.roll(S, -(ia + 1))                        # S starting at b ... ending at a
        Tr = np.roll(T, -ic)                              # T starting at c
        if fwd:                                           # d follows c in T: a -> c ... (T backwards from c) ... d -> b
            Tseq = np.concatenate(([Tr[0]], Tr[:0:-1]))
        else:                                             # d precedes c: a -> c -> ... (T forwards) ... -> d -> b
            Tseq = Tr
        merged = np.concatenate((Sr, Tseq))               # b..a then c..d, closing d->b
        cycles = [merged] + [c for j, c in enumerate(cycles) if j not in (0, k)]
    t = cycles[0]
    return np.roll(t, -int(np.flatnonzero(t == 0)[0]))


def _repacked_value(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray, cities: np.ndarray, csr):
    """Exact value after single flips (exact deltas) of the items of `cities` on `tour`; returns (value, picked)."""
    from . import kernels as K
    from .evaluator import city_weights
    pk = picked.copy()
    pos = np.empty(inst.n, np.int64)
    pos[tour] = np.arange(inst.n)
    cw = city_weights(inst, pk)
    load = np.cumsum(cw[tour])
    wtot = float(inst.weight[pk].sum())
    K.flip_local(np.ascontiguousarray(cities, np.int64), len(cities), np.ascontiguousarray(tour, np.int64), pos, pk, cw,
                 load, inst.xy, inst.profit, inst.weight, csr[0], csr[1], inst.capacity, wtot, inst.v_max, inst.v_min, inst.nu, inst.R)
    return objective(inst, tour, pk), pk


def ttp_transplant(inst: LeanTTP, P: np.ndarray, Q: np.ndarray, picked: np.ndarray, nbr: np.ndarray, max_comp: int = 50,
                   repack: bool = True):
    """Child of base parent P and donor Q: for each difference component (largest first, at most max_comp), replace P's
    edges inside it by Q's, patch the resulting cycles into one tour, and keep the change if the exact TTP objective
    improves. With repack=True the items of the component's cities are re-decided by exact single flips before the
    comparison (a moved component changes where its items are picked). Returns (tour, value, n_comp, n_taken, picked)."""
    from .search import _item_csr
    csr = _item_csr(inst)
    n = len(P)
    common, onlyP, onlyQ, lab, _ = partition(P, Q)
    labP, labQ = lab[onlyP // n], lab[onlyQ // n]
    comps = np.unique(labP)
    sizes = np.bincount(lab)
    comps = sorted(comps, key=lambda c: -sizes[c])[:max_comp]
    use_q = set()
    cur_tour, cur_val = _best_orientation(inst, P, picked)
    cur_pk = picked.copy()
    taken = 0
    for c in comps:
        trial = use_q | {c}
        keys = np.concatenate([common, onlyP[~np.isin(labP, list(trial))], onlyQ[np.isin(labQ, list(trial))]])
        cyc = _cycles_from_keys(keys, n)
        if cyc is None:
            continue
        t = _patch(inst, cyc, nbr)
        rev = np.concatenate(([t[0]], t[:0:-1]))
        cities = np.flatnonzero(lab == c)
        best = None
        for tt in (t, rev):
            if repack:
                val, pk = _repacked_value(inst, tt, cur_pk, cities, csr)
            else:
                val, pk = objective(inst, tt, cur_pk), cur_pk
            if best is None or val > best[0]:
                best = (val, tt, pk)
        if best[0] > cur_val + 1e-9:
            use_q, cur_val, cur_tour, cur_pk = trial, best[0], best[1], best[2]
            taken += 1
    return cur_tour, cur_val, len(comps), taken, cur_pk


def recombine(inst: LeanTTP, pool, nbr, deadline: float, rounds: int = 2):
    """Transplant crossover (ttp_transplant) between the first solution of the pool and the other pool tours,
    each child repaired by a short local search; keeps improvements only. Returns (State, n_improvements)."""
    from .search import State, local_search
    P = State(inst, pool[0].tour, pool[0].picked, nbr)
    local_search(P, time_limit=max(0.5, min(20.0, deadline - time.monotonic())))
    wins = 0
    for _ in range(rounds):
        improved = False
        for q in pool[1:]:
            if time.monotonic() > deadline:
                return P, wins
            tour, _, _, taken, pk = ttp_transplant(inst, P.tour, q.tour, P.picked, nbr)
            if taken == 0:
                continue
            ch = State(inst, tour, pk, nbr)
            local_search(ch, time_limit=max(0.2, min(10.0, deadline - time.monotonic())))
            if ch.value() > P.value() + 1e-9:
                P, wins, improved = ch, wins + 1, True
        if not improved:
            break
    return P, wins
