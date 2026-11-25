"""numba kernels for the tour-side search (CEIL_2D only).

State arrays (all indexed by tour position s = 0..n-1, depot at position 0 and never moved):
  tour[s]  city,  pos[city] = s,  cw[city] = weight picked at the city,
  load[s]  weight carried when leaving tour[s] (after pickup),
  et[s]    time of the edge tour[s] -> tour[s+1 mod n] = d / v(load[s]).
Objective = sum(profit of picked) - R * sum(et).
"""
import math

import numpy as np
from numba import njit

from .ablation import SECOND   # 1.0 in LARC; 0.0 drops the second-order term (ablation of the paper)


@njit(cache=True, inline="always")
def _d(xy, a, b):
    return math.ceil(math.hypot(xy[a, 0] - xy[b, 0], xy[a, 1] - xy[b, 1]))


@njit(cache=True, inline="always")
def _inv_v(L, vmax, vmin, nu):
    v = vmax - nu * L
    if v < vmin:
        v = vmin
    return 1.0 / v


@njit(cache=True)
def init_state(tour, cw, xy, vmax, vmin, nu):
    n = len(tour)
    load = np.empty(n)
    et = np.empty(n)
    L = 0.0
    for s in range(n):
        L += cw[tour[s]]
        load[s] = L
    for s in range(n):
        et[s] = _d(xy, tour[s], tour[(s + 1) % n]) * _inv_v(load[s], vmax, vmin, nu)
    return load, et


@njit(cache=True)
def _seg_new_old(tour, cw, load, et, xy, i, j, vmax, vmin, nu):
    """(new, old) travel time of edges i..j when tour[i+1..j] is reversed."""
    n = len(tour)
    L = load[i]
    prev = tour[i]
    new = 0.0
    old = 0.0
    for k in range(j, i, -1):
        c = tour[k]
        new += _d(xy, prev, c) * _inv_v(L, vmax, vmin, nu)
        old += et[k]
        L += cw[c]
        prev = c
    new += _d(xy, prev, tour[(j + 1) % n]) * _inv_v(L, vmax, vmin, nu)
    old += et[i]
    return new, old


@njit(cache=True)
def _apply_reverse(tour, pos, cw, load, et, xy, i, j, vmax, vmin, nu):
    n = len(tour)
    a, b = i + 1, j
    while a < b:
        ca, cb = tour[a], tour[b]
        tour[a], tour[b] = cb, ca
        pos[cb], pos[ca] = a, b
        a += 1
        b -= 1
    L = load[i]
    for s in range(i + 1, j + 1):
        L += cw[tour[s]]
        load[s] = L
    for s in range(i, j + 1):
        et[s] = _d(xy, tour[s], tour[(s + 1) % n]) * _inv_v(load[s], vmax, vmin, nu)


@njit(cache=True)
def two_opt_ls(tour, pos, cw, load, et, xy, nbr, vmax, vmin, nu, R, max_seg, slack, max_moves, dlb, max_work):
    """First-improvement 2-opt over neighbour lists with don't-look bits and exact segment delta.
    Returns (moves, total_gain, interrupted). Items are fixed; gains come from distance and from load re-ordering.
    `max_work` bounds the total length of evaluated segments, so a call returns (interrupted=True) even when no move
    is found for a long time; don't-look bits keep the progress for the next call."""
    n = len(tour)
    k = nbr.shape[1]
    moves = 0
    gain_total = 0.0
    work = 0
    interrupted = False
    changed = True
    while changed and moves < max_moves:
        changed = False
        for a in range(1, n):
            if work > max_work:
                interrupted = True
                break
            if dlb[a]:
                continue
            found = False
            pa = pos[a]
            for var in range(2):
                if found:
                    break
                for t in range(k):
                    c = nbr[a, t]
                    if c == 0 and var == 1:
                        pass
                    pc = pos[c]
                    pa = pos[a]
                    if var == 0:                       # new edge (a,c) replacing (a,succ a) and (c,succ c)
                        base = _d(xy, a, tour[(pa + 1) % n])
                        i, j = (pa, pc) if pa < pc else (pc, pa)
                    else:                              # new edge (a,c) replacing (pred a,a) and (pred c,c)
                        if pa == 0:
                            continue
                        base = _d(xy, tour[pa - 1], a)
                        if pc == 0:
                            continue
                        lo, hi = (pa, pc) if pa < pc else (pc, pa)
                        i, j = lo - 1, hi - 1
                    if _d(xy, a, c) >= slack * base:
                        continue
                    if i < 0 or j <= i or j - i > max_seg:
                        continue
                    work += j - i
                    new, old = _seg_new_old(tour, cw, load, et, xy, i, j, vmax, vmin, nu)
                    g = R * (old - new)
                    if g > 1e-9:
                        e0, e1, e2, e3 = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                        _apply_reverse(tour, pos, cw, load, et, xy, i, j, vmax, vmin, nu)
                        gain_total += g
                        moves += 1
                        dlb[e0] = False
                        dlb[e1] = False
                        dlb[e2] = False
                        dlb[e3] = False
                        found = True
                        changed = True
                        break
            if not found:
                dlb[a] = True
            if moves >= max_moves:
                break
        if interrupted:
            break
    return moves, gain_total, interrupted


@njit(cache=True)
def flip_pass(tour, pos, picked, cw, load, xy, profit, weight, city, capacity, wtot, vmax, vmin, nu, R, top_k):
    """One pass of single-item flips: candidates ranked by a 2nd-order estimate, accepted by exact delta.
    Updates picked, cw, load in place. Returns (n_flips, total_gain, new_wtot)."""
    n = len(tour)
    m = len(profit)
    d = np.empty(n)
    for s in range(n):
        d[s] = _d(xy, tour[s], tour[(s + 1) % n])
    g1 = np.zeros(n + 1)
    g2 = np.zeros(n + 1)
    for s in range(n - 1, -1, -1):
        v = vmax - nu * load[s]
        if v < vmin:
            v = vmin
        g1[s] = g1[s + 1] + d[s] / (v * v)
        g2[s] = g2[s + 1] + SECOND * d[s] / (v * v * v)
    approx = np.empty(m)
    for j in range(m):
        k = pos[city[j]]
        nw = nu * weight[j]
        if picked[j]:
            approx[j] = -profit[j] + R * (nw * g1[k] - nw * nw * g2[k])
        else:
            approx[j] = profit[j] - R * (nw * g1[k] + nw * nw * g2[k])
    kk = min(top_k, m)
    cand = np.argsort(-approx)[:kk]
    flips = 0
    gain = 0.0
    for q in range(kk):
        j = cand[q]
        kj = pos[city[j]]
        wj = weight[j]
        sign = -1.0 if picked[j] else 1.0
        if sign > 0 and wtot + wj > capacity:
            continue
        told = 0.0
        tnew = 0.0
        for s in range(kj, n):
            c0 = load[s]
            c1 = c0 + sign * wj
            v0 = vmax - nu * c0
            if v0 < vmin:
                v0 = vmin
            v1 = vmax - nu * c1
            if v1 < vmin:
                v1 = vmin
            told += d[s] / v0
            tnew += d[s] / v1
        delta = sign * profit[j] - R * (tnew - told)
        if delta > 1e-9:
            picked[j] = not picked[j]
            cw[city[j]] += sign * wj
            for s in range(kj, n):
                load[s] += sign * wj
            wtot += sign * wj
            flips += 1
            gain += delta
    return flips, gain, wtot


@njit(cache=True)
def _suffix_time(d, load, s0, delta_w, vmax, vmin, nu):
    t = 0.0
    for s in range(s0, len(d)):
        v = vmax - nu * (load[s] + delta_w)
        if v < vmin:
            v = vmin
        t += d[s] / v
    return t


@njit(cache=True)
def swap_pass(tour, pos, picked, cw, load, xy, profit, weight, city, capacity, wtot, vmax, vmin, nu, R, kp, kq, max_verify):
    """Swap moves (drop picked i, add unpicked j) for full knapsacks. Candidate pairs are ranked by an additive
    2nd-order estimate; the best `max_verify` feasible pairs are accepted by exact delta. Returns (swaps, gain, wtot)."""
    n = len(tour)
    m = len(profit)
    d = np.empty(n)
    for s in range(n):
        d[s] = _d(xy, tour[s], tour[(s + 1) % n])
    g1 = np.zeros(n + 1)
    g2 = np.zeros(n + 1)
    for s in range(n - 1, -1, -1):
        v = vmax - nu * load[s]
        if v < vmin:
            v = vmin
        g1[s] = g1[s + 1] + d[s] / (v * v)
        g2[s] = g2[s + 1] + SECOND * d[s] / (v * v * v)
    gain_add = np.full(m, -1e300)
    gain_rem = np.full(m, -1e300)
    for j in range(m):
        k = pos[city[j]]
        nw = nu * weight[j]
        if picked[j]:
            gain_rem[j] = -profit[j] + R * (nw * g1[k] - nw * nw * g2[k])
        else:
            gain_add[j] = profit[j] - R * (nw * g1[k] + nw * nw * g2[k])
    P = np.argsort(-gain_rem)[:kp]
    Q = np.argsort(-gain_add)[:kq]
    pairs_est = np.empty(len(P) * len(Q))
    pairs_i = np.empty(len(P) * len(Q), np.int64)
    pairs_j = np.empty(len(P) * len(Q), np.int64)
    cnt = 0
    for a in range(len(P)):
        i = P[a]
        if not picked[i]:
            continue
        for b in range(len(Q)):
            j = Q[b]
            if picked[j]:
                continue
            pairs_est[cnt] = gain_rem[i] + gain_add[j]
            pairs_i[cnt] = i
            pairs_j[cnt] = j
            cnt += 1
    order = np.argsort(-pairs_est[:cnt])
    swaps = 0
    gain = 0.0
    tried = 0
    for q in range(cnt):
        if tried >= max_verify:
            break
        if pairs_est[order[q]] < -1e-9 and tried > 0:
            break
        i = pairs_i[order[q]]
        j = pairs_j[order[q]]
        if not picked[i] or picked[j]:
            continue
        if wtot - weight[i] + weight[j] > capacity:
            continue
        tried += 1
        ki, kj = pos[city[i]], pos[city[j]]
        # exact: remove i
        t0 = _suffix_time(d, load, ki, 0.0, vmax, vmin, nu)
        t1 = _suffix_time(d, load, ki, -weight[i], vmax, vmin, nu)
        for s in range(ki, n):
            load[s] -= weight[i]
        # then add j
        t2 = _suffix_time(d, load, kj, 0.0, vmax, vmin, nu)
        t3 = _suffix_time(d, load, kj, weight[j], vmax, vmin, nu)
        delta = -profit[i] + profit[j] - R * ((t1 - t0) + (t3 - t2))
        if delta > 1e-9:
            for s in range(kj, n):
                load[s] += weight[j]
            picked[i] = False
            picked[j] = True
            cw[city[i]] -= weight[i]
            cw[city[j]] += weight[j]
            wtot += weight[j] - weight[i]
            swaps += 1
            gain += delta
        else:
            for s in range(ki, n):
                load[s] += weight[i]
    return swaps, gain, wtot


@njit(cache=True)
def _build_seq(tour, p, L, q, rev, seq):
    """New order of the cities of positions lo+1..hi after moving segment tour[p..p+L-1] after position q
    (insertion between tour[q] and tour[q+1]); rev reverses the segment. Returns (lo, hi, count)."""
    k = 0
    if q >= p + L:                     # forward: lo = p-1, hi = q
        lo, hi = p - 1, q
        for s in range(p + L, q + 1):
            seq[k] = tour[s]
            k += 1
        for t in range(L):
            seq[k] = tour[p + L - 1 - t] if rev else tour[p + t]
            k += 1
    else:                              # backward: q < p-1 or q == p-1 handled by caller (no-op), lo = q, hi = p+L-1
        lo, hi = q, p + L - 1
        for t in range(L):
            seq[k] = tour[p + L - 1 - t] if rev else tour[p + t]
            k += 1
        for s in range(q + 1, p):
            seq[k] = tour[s]
            k += 1
    return lo, hi, k


@njit(cache=True)
def or_opt_ls(tour, pos, cw, load, et, xy, nbr, vmax, vmin, nu, R, max_seg, max_len, dist_slack, max_moves, dlb):
    """Or-opt: relocate a segment of 1..max_len cities (either orientation) next to a neighbour of one of its ends.
    Exact delta over the affected position range; first improvement. Returns (moves, gain)."""
    n = len(tour)
    kn = nbr.shape[1]
    seq = np.empty(n, np.int64)
    moves = 0
    gain_total = 0.0
    changed = True
    while changed and moves < max_moves:
        changed = False
        for f in range(1, n):
            if dlb[f]:
                continue
            found = False
            for L in range(1, max_len + 1):
                if found:
                    break
                for end in range(2):
                    # segment starts at pos[f] (end=0: f is its first city) or ends at pos[f] (end=1)
                    p = pos[f] if end == 0 else pos[f] - L + 1
                    if p < 1 or p + L > n or (end == 1 and L == 1):
                        continue
                    first, last = tour[p], tour[p + L - 1]
                    prev_c = tour[p - 1]
                    next_c = tour[(p + L) % n]
                    removed = _d(xy, prev_c, first) + _d(xy, last, next_c) - _d(xy, prev_c, next_c)
                    for t in range(kn):
                        c = nbr[f, t]
                        pc = pos[c]
                        if p <= pc < p + L:
                            continue
                        for side in range(2):               # insert after c (side 0) or before c (side 1)
                            q = pc if side == 0 else pc - 1
                            if q < 0 or q == p - 1:
                                continue
                            if q >= p and q < p + L:
                                continue
                            if abs(q - p) > max_seg:
                                continue
                            a_c = tour[q]
                            b_c = tour[(q + 1) % n]
                            for rev in range(2):
                                x, y = (last, first) if rev == 1 else (first, last)
                                added = _d(xy, a_c, x) + _d(xy, y, b_c) - _d(xy, a_c, b_c)
                                if added - removed > dist_slack:
                                    continue
                                lo, hi, cnt = _build_seq(tour, p, L, q, rev == 1, seq)
                                Lc = load[lo]
                                prev = tour[lo]
                                new = 0.0
                                old = 0.0
                                for u in range(cnt):
                                    cc = seq[u]
                                    new += _d(xy, prev, cc) * _inv_v(Lc, vmax, vmin, nu)
                                    Lc += cw[cc]
                                    prev = cc
                                new += _d(xy, prev, tour[(hi + 1) % n]) * _inv_v(Lc, vmax, vmin, nu)
                                for s in range(lo, hi + 1):
                                    old += et[s]
                                g = R * (old - new)
                                if g > 1e-9:
                                    e1, e2, e3, e4 = prev_c, next_c, a_c, b_c
                                    for u in range(cnt):
                                        tour[lo + 1 + u] = seq[u]
                                        pos[seq[u]] = lo + 1 + u
                                    Lc = load[lo]
                                    for s in range(lo + 1, hi + 1):
                                        Lc += cw[tour[s]]
                                        load[s] = Lc
                                    for s in range(lo, hi + 1):
                                        et[s] = _d(xy, tour[s], tour[(s + 1) % n]) * _inv_v(load[s], vmax, vmin, nu)
                                    gain_total += g
                                    moves += 1
                                    dlb[e1] = False
                                    dlb[e2] = False
                                    dlb[e3] = False
                                    dlb[e4] = False
                                    dlb[first] = False
                                    dlb[last] = False
                                    found = True
                                    changed = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
            if not found:
                dlb[f] = True
            if moves >= max_moves:
                break
    return moves, gain_total


# ----------------------------------------------------------------------------- iterated local search / annealing
@njit(cache=True)
def flip_local(cities, ncities, tour, pos, picked, cw, load, xy, profit, weight, istart, iorder, capacity, wtot, vmax, vmin, nu, R):
    """Single flips of the items of the given cities. Each item is screened with the O(1) second-order estimate
    (suffix sums of d/v^2 and d/v^3 computed once, O(n)); only items whose estimate is close to
    being improving (within 0.1% of its profit) get the exact O(n - k) check. Accepts exact improvements only. Returns (profit_delta, wtot)."""
    n = len(tour)
    d = np.empty(n)
    for s in range(n):
        d[s] = _d(xy, tour[s], tour[(s + 1) % n])
    g1 = np.zeros(n + 1)
    g2 = np.zeros(n + 1)
    for s in range(n - 1, -1, -1):
        v = vmax - nu * load[s]
        if v < vmin:
            v = vmin
        g1[s] = g1[s + 1] + d[s] / (v * v)
        g2[s] = g2[s + 1] + SECOND * d[s] / (v * v * v)
    dp = 0.0
    for u in range(ncities):
        c = cities[u]
        kj = pos[c]
        for q in range(istart[c], istart[c + 1]):
            j = iorder[q]
            sign = -1.0 if picked[j] else 1.0
            if sign > 0 and wtot + weight[j] > capacity:
                continue
            nw = nu * weight[j]
            if sign > 0:
                est = profit[j] - R * (nw * g1[kj] + nw * nw * g2[kj])
            else:
                est = -profit[j] + R * (nw * g1[kj] - nw * nw * g2[kj])
            if est < -1e-3 * profit[j]:          # 2nd-order estimate: error is third order in the weight
                continue
            told = 0.0
            tnew = 0.0
            for s in range(kj, n):
                v0 = vmax - nu * load[s]
                if v0 < vmin:
                    v0 = vmin
                v1 = vmax - nu * (load[s] + sign * weight[j])
                if v1 < vmin:
                    v1 = vmin
                told += d[s] / v0
                tnew += d[s] / v1
            delta = sign * profit[j] - R * (tnew - told)
            if delta > 1e-9:
                for s in range(kj, n):
                    load[s] += sign * weight[j]
                cw[c] += sign * weight[j]
                picked[j] = not picked[j]
                wtot += sign * weight[j]
                dp += sign * profit[j]
    return dp, wtot


@njit(cache=True)
def _recompute_et(tour, load, et, xy, vmax, vmin, nu):
    n = len(tour)
    tot = 0.0
    for s in range(n):
        et[s] = _d(xy, tour[s], tour[(s + 1) % n]) * _inv_v(load[s], vmax, vmin, nu)
        tot += et[s]
    return tot


@njit(cache=True)
def _flip_item(j, picked, cw, load, pos, city, weight, n):
    sign = -1.0 if picked[j] else 1.0
    for s in range(pos[city[j]], n):
        load[s] += sign * weight[j]
    cw[city[j]] += sign * weight[j]
    picked[j] = not picked[j]
    return sign * weight[j]


@njit(cache=True)
def ils_chunk(tour, pos, picked, cw, load, et, xy, nbr, profit, weight, city, istart, iorder, capacity,
              vmax, vmin, nu, R, st, iters, T, win, p_item, max_seg, max_len, dlb,
              b_tour, b_picked, tmp_tour, tmp_pos, tmp_picked, tmp_cw, tmp_load, tmp_et, touched):
    """`iters` iterations of: kick (item-aware or tour segment move) -> local repair (2-opt, Or-opt, local flips,
    all exact) -> Metropolis acceptance at temperature T. st = [profit_sum, weight_sum, best_value].
    The best solution is copied to b_tour / b_picked. Returns (accepted, new_bests)."""
    n = len(tour)
    m = len(profit)
    accepted = 0
    new_best = 0
    seq = np.empty(n, np.int64)
    for it in range(iters):
        tmp_tour[:] = tour
        tmp_pos[:] = pos
        tmp_picked[:] = picked
        tmp_cw[:] = cw
        tmp_load[:] = load
        tmp_et[:] = et
        old_profit = st[0]
        old_w = st[1]
        old_val = old_profit - R * et.sum()
        nt = 0
        feasible = True
        if np.random.random() < p_item:                                  # ---- item-aware kick
            r = 1 + int(np.random.random() * 3)
            for _ in range(r):
                j = int(np.random.random() * m)
                if picked[j]:
                    st[1] += _flip_item(j, picked, cw, load, pos, city, weight, n)
                    st[0] -= profit[j]
                    touched[nt] = city[j]
                    nt += 1
                    for _try in range(20):                               # swap-type kick: add another item that fits
                        j2 = int(np.random.random() * m)
                        if (not picked[j2]) and st[1] + weight[j2] <= capacity:
                            st[1] += _flip_item(j2, picked, cw, load, pos, city, weight, n)
                            st[0] += profit[j2]
                            touched[nt] = city[j2]
                            nt += 1
                            break
                else:
                    st[1] += _flip_item(j, picked, cw, load, pos, city, weight, n)
                    st[0] += profit[j]
                    touched[nt] = city[j]
                    nt += 1
                    tries = 0
                    while st[1] > capacity and tries < 300:               # repair capacity with random drops
                        j2 = int(np.random.random() * m)
                        tries += 1
                        if picked[j2] and j2 != j and nt < len(touched) - 2:
                            st[1] += _flip_item(j2, picked, cw, load, pos, city, weight, n)
                            st[0] -= profit[j2]
                            touched[nt] = city[j2]
                            nt += 1
                    if st[1] > capacity:
                        feasible = False
                if nt >= len(touched) - 6:
                    break
        else:                                                            # ---- tour kick: relocate a random segment
            if n > 8:
                Lmax = min(win, n // 4)
                L = 1 + int(np.random.random() * Lmax)
                p = 1 + int(np.random.random() * (n - L - 2))
                room = n - 1 - (p + L)
                if room >= 1:
                    q = p + L + int(np.random.random() * min(win, room))
                    lo, hi, cnt = _build_seq(tour, p, L, q, np.random.random() < 0.5, seq)
                    for u in range(cnt):
                        tour[lo + 1 + u] = seq[u]
                        pos[seq[u]] = lo + 1 + u
                    Lc = load[lo]
                    for s in range(lo + 1, hi + 1):
                        Lc += cw[tour[s]]
                        load[s] = Lc
                    touched[nt] = tour[lo]
                    nt += 1
                    for u in range(min(cnt, len(touched) - 4)):
                        touched[nt] = seq[u]
                        nt += 1
                    touched[nt] = tour[(hi + 1) % n]
                    nt += 1
        if feasible:
            tot = _recompute_et(tour, load, et, xy, vmax, vmin, nu)
            for u in range(nt):
                dlb[touched[u]] = False
            for _round in range(3):
                m1, g1_, _ = two_opt_ls(tour, pos, cw, load, et, xy, nbr, vmax, vmin, nu, R, max_seg, 1.0, 1000000, dlb, 1 << 62)
                m2, g2_ = or_opt_ls(tour, pos, cw, load, et, xy, nbr, vmax, vmin, nu, R, max_seg, max_len, 1e18, 1000000, dlb)
                dp, w2 = flip_local(touched, nt, tour, pos, picked, cw, load, xy, profit, weight, istart, iorder,
                                    capacity, st[1], vmax, vmin, nu, R)
                st[0] += dp
                st[1] = w2
                tot = _recompute_et(tour, load, et, xy, vmax, vmin, nu)
                if m1 == 0 and m2 == 0 and dp == 0.0:
                    break
            new_val = st[0] - R * tot
            delta = new_val - old_val
            ok = delta >= 0.0 or (T > 0.0 and np.random.random() < math.exp(delta / T))
        else:
            ok = False
            new_val = old_val
        if ok:
            accepted += 1
            if new_val > st[2] + 1e-9:
                st[2] = new_val
                b_tour[:] = tour
                b_picked[:] = picked
                new_best += 1
        else:
            tour[:] = tmp_tour
            pos[:] = tmp_pos
            picked[:] = tmp_picked
            cw[:] = tmp_cw
            load[:] = tmp_load
            et[:] = tmp_et
            st[0] = old_profit
            st[1] = old_w
            dlb[:] = True
    return accepted, new_best


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def two_opt_repack(tour, pos, picked, cw, load, et, xy, nbr, profit, weight, city, istart, iorder, capacity, st,
                   vmax, vmin, nu, R, max_seg, tau, max_attempts, dlb,
                   tmp_tour, tmp_pos, tmp_picked, tmp_cw, tmp_load, tmp_et, touched):
    """Compound 2-opt: a reversal that is not improving by itself (exact gain > -tau) is applied tentatively, the items
    of the cities in the reversed segment are re-decided (exact single flips), and the pair is kept only if the total
    objective improves. Plain improving reversals are applied directly. st = [profit_sum, weight_sum].
    Returns (plain_moves, compound_moves, attempts)."""
    n = len(tour)
    k = nbr.shape[1]
    plain = 0
    comp = 0
    attempts = 0
    changed = True
    while changed and attempts < max_attempts:
        changed = False
        for a in range(1, n):
            if dlb[a]:
                continue
            found = False
            for var in range(2):
                if found:
                    break
                for t in range(k):
                    c = nbr[a, t]
                    pa = pos[a]
                    pc = pos[c]
                    if var == 0:
                        i, j = (pa, pc) if pa < pc else (pc, pa)
                    else:
                        if pa == 0 or pc == 0:
                            continue
                        lo, hi = (pa, pc) if pa < pc else (pc, pa)
                        i, j = lo - 1, hi - 1
                    if i < 0 or j <= i or j - i > max_seg:
                        continue
                    new, old = _seg_new_old(tour, cw, load, et, xy, i, j, vmax, vmin, nu)
                    g = R * (old - new)
                    e0, e1, e2, e3 = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                    if g > 1e-9:
                        _apply_reverse(tour, pos, cw, load, et, xy, i, j, vmax, vmin, nu)
                        plain += 1
                        ok = True
                    elif g > -tau and attempts < max_attempts:
                        attempts += 1
                        tmp_tour[:] = tour
                        tmp_pos[:] = pos
                        tmp_picked[:] = picked
                        tmp_cw[:] = cw
                        tmp_load[:] = load
                        tmp_et[:] = et
                        old_p, old_w = st[0], st[1]
                        old_val = old_p - R * et.sum()
                        _apply_reverse(tour, pos, cw, load, et, xy, i, j, vmax, vmin, nu)
                        cnt = 0
                        for s in range(i, min(j + 2, n)):
                            touched[cnt] = tour[s]
                            cnt += 1
                        dp, w2 = flip_local(touched, cnt, tour, pos, picked, cw, load, xy, profit, weight, istart, iorder,
                                            capacity, st[1], vmax, vmin, nu, R)
                        st[0] += dp
                        st[1] = w2
                        tot = _recompute_et(tour, load, et, xy, vmax, vmin, nu)
                        if st[0] - R * tot > old_val + 1e-9:
                            comp += 1
                            ok = True
                        else:
                            tour[:] = tmp_tour
                            pos[:] = tmp_pos
                            picked[:] = tmp_picked
                            cw[:] = tmp_cw
                            load[:] = tmp_load
                            et[:] = tmp_et
                            st[0], st[1] = old_p, old_w
                            ok = False
                    else:
                        ok = False
                    if ok:
                        dlb[e0] = False
                        dlb[e1] = False
                        dlb[e2] = False
                        dlb[e3] = False
                        found = True
                        changed = True
                        break
            if not found:
                dlb[a] = True
            if attempts >= max_attempts:
                break
    return plain, comp, attempts


@njit(cache=True)
def reversal_ls(tour, pos, cw, load, et, xy, nbr, vmax, vmin, nu, R, max_seg, max_moves, dlb, max_work):
    """Segment reversal between geometric neighbours (as in CoCo's 2-opt): for a city a and a neighbour c, reverse
    the inclusive segment tour[min(pa,pc) .. max(pa,pc)]. The new edges join the segment's outer neighbours to its ends
    and can be long, unlike standard 2-opt whose new edge (a, c) is short. These moves re-route which part of the map is
    travelled heavy. Exact evaluation in O(segment); first improvement. Returns (moves, gain, interrupted)."""
    n = len(tour)
    k = nbr.shape[1]
    moves = 0
    gain_total = 0.0
    work = 0
    interrupted = False
    changed = True
    while changed and moves < max_moves:
        changed = False
        for a in range(1, n):
            if work > max_work:
                interrupted = True
                break
            if dlb[a]:
                continue
            found = False
            for t in range(k):
                c = nbr[a, t]
                if c == 0:
                    continue
                pa = pos[a]
                pc = pos[c]
                lo, hi = (pa, pc) if pa < pc else (pc, pa)
                i, j = lo - 1, hi                      # reverse tour[i+1 .. j] = tour[lo .. hi]
                if i < 0 or j - i > max_seg or j - i < 2:
                    continue
                work += j - i
                new, old = _seg_new_old(tour, cw, load, et, xy, i, j, vmax, vmin, nu)
                g = R * (old - new)
                if g > 1e-9:
                    e0, e1, e2, e3 = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                    _apply_reverse(tour, pos, cw, load, et, xy, i, j, vmax, vmin, nu)
                    gain_total += g
                    moves += 1
                    dlb[e0] = False
                    dlb[e1] = False
                    dlb[e2] = False
                    dlb[e3] = False
                    found = True
                    changed = True
                    break
            if not found:
                dlb[a] = True
            if moves >= max_moves:
                break
        if interrupted:
            break
    return moves, gain_total, interrupted


# ----------------------------------------------------------------------------- fast coordinated 2-opt
@njit(cache=True)
def _suffix_sums(tour, load, xy, vmax, vmin, nu, g1s, g2s, esuf):
    """g1s[s] = sum_{q>=s} d_q/v_q^2, g2s[s] = sum d_q/v_q^3, esuf[s] = sum d_q/v_q (current state), with [n] = 0."""
    n = len(tour)
    g1s[n] = 0.0
    g2s[n] = 0.0
    esuf[n] = 0.0
    for s in range(n - 1, -1, -1):
        d = _d(xy, tour[s], tour[(s + 1) % n])
        iv = _inv_v(load[s], vmax, vmin, nu)
        g1s[s] = g1s[s + 1] + d * iv * iv
        g2s[s] = g2s[s + 1] + SECOND * d * iv * iv * iv
        esuf[s] = esuf[s + 1] + d * iv


@njit(cache=True)
def _g_shift(tour, load, xy, vmax, vmin, nu, s0, dW, power):
    """sum_{s>=s0} d_s / v(load_s + dW)^power (used when a repack changes the weight carried beyond the segment)."""
    n = len(tour)
    acc = 0.0
    for s in range(s0, n):
        iv = _inv_v(load[s] + dW, vmax, vmin, nu)
        acc += _d(xy, tour[s], tour[(s + 1) % n]) * iv ** (power - 1) * iv
    return acc


@njit(cache=True)
def coord_two_opt_ls(tour, pos, picked, cw, load, et, xy, nbr, profit, weight, city, istart, iorder, capacity, st,
                     vmax, vmin, nu, R, max_seg, max_moves, dlb, max_work, passes, frac, screen=False):
    """Coordinated 2-opt: each candidate reversal tour[i+1..j] is evaluated together with a repack of the items of the
    segment's cities. Items are scored by the load-aware second-order margin at their new position (loads of the
    reversed segment, suffix sums beyond it) and flipped in damped best-first passes (the best `frac` of the positive
    margins per pass, margins recomputed between passes; deciding all flips at once oscillates). The compound move and
    the plain reversal are then evaluated exactly in O(n - i); the better one is applied if it improves. No state copy. st = [profit_sum, weight_sum]. Returns (moves, gain, interrupted).
    screen=True: after the first repack pass, if the exact plain-reversal gain plus the sum of the positive first-pass
    item margins is not positive, the remaining passes and the compound evaluation are skipped (the candidate can still
    be applied as a plain reversal). Moves are only applied on their exact gain, so the screen can miss moves, never
    apply a worsening one."""
    n = len(tour)
    k = nbr.shape[1]
    g1s = np.zeros(n + 1)
    g2s = np.zeros(n + 1)
    esuf = np.zeros(n + 1)
    _suffix_sums(tour, load, xy, vmax, vmin, nu, g1s, g2s, esuf)
    nl = np.empty(n)                     # new loads in the segment (by segment index q)
    dn = np.empty(n)                     # new edge lengths leaving segment position q
    G1 = np.empty(n + 1)
    G2 = np.empty(n + 1)
    dwq = np.zeros(n)                    # weight change per segment index
    nlx = np.empty(n)
    m_ = len(profit)
    sit = np.empty(m_, np.int64)         # items of the segment's cities, their segment index, tentative state
    sq = np.empty(m_, np.int64)
    sp = np.zeros(m_, np.bool_)
    sg = np.empty(m_)
    se = np.empty(m_, np.int64)
    flips = np.empty(m_, np.int64)
    moves = 0
    gain_total = 0.0
    work = 0
    interrupted = False
    changed = True
    while changed and moves < max_moves:
        changed = False
        for a in range(1, n):
            if work > max_work:
                interrupted = True
                break
            if dlb[a]:
                continue
            found = False
            for var in range(2):
                if found:
                    break
                for t in range(k):
                    c = nbr[a, t]
                    pa = pos[a]
                    pc = pos[c]
                    if var == 0:
                        i, j = (pa, pc) if pa < pc else (pc, pa)
                    else:
                        if pa == 0 or pc == 0:
                            continue
                        lo, hi = (pa, pc) if pa < pc else (pc, pa)
                        i, j = lo - 1, hi - 1
                    if i < 0 or j - i < 2 or j - i > max_seg:
                        continue
                    L = j - i
                    work += n - i
                    # --- new order, loads and edge lengths of the reversed segment (items unchanged)
                    Lc = load[i]
                    for q in range(L):
                        cq = tour[j - q]
                        Lc += cw[cq]
                        nl[q] = Lc
                        nxt = tour[j - q - 1] if q < L - 1 else tour[(j + 1) % n]
                        dn[q] = _d(xy, cq, nxt)
                    d_first = _d(xy, tour[i], tour[j])
                    t_plain = d_first * _inv_v(load[i], vmax, vmin, nu)
                    for q in range(L):
                        t_plain += dn[q] * _inv_v(nl[q], vmax, vmin, nu)
                    t_plain += esuf[j + 1]
                    old = esuf[i]
                    g_plain = -R * (t_plain - old)
                    # --- damped best-first repack of the segment's items: per pass, flip the best `frac` of the items
                    #     with a positive load-aware margin (loads/sensitivities recomputed with the tentative flips)
                    ns = 0
                    for q in range(L):
                        cq = tour[j - q]
                        for r in range(istart[cq], istart[cq + 1]):
                            sit[ns] = iorder[r]
                            sq[ns] = q
                            sp[ns] = picked[iorder[r]]
                            ns += 1
                    nf = 0
                    dp = 0.0
                    dW = 0.0
                    for q in range(L):
                        dwq[q] = 0.0
                    for ps in range(passes):
                        acc = 0.0
                        for q in range(L):
                            acc += dwq[q]
                            nlx[q] = nl[q] + acc
                        G1[L] = g1s[j + 1] if dW == 0.0 else _g_shift(tour, load, xy, vmax, vmin, nu, j + 1, dW, 2)
                        G2[L] = g2s[j + 1] if dW == 0.0 else _g_shift(tour, load, xy, vmax, vmin, nu, j + 1, dW, 3)
                        work += L + ns + (2 * (n - j) if dW != 0.0 else 0)          # real cost of this pass
                        for q in range(L - 1, -1, -1):
                            iv = _inv_v(nlx[q], vmax, vmin, nu)
                            G1[q] = G1[q + 1] + dn[q] * iv * iv
                            G2[q] = G2[q + 1] + SECOND * dn[q] * iv * iv * iv
                        npos = 0
                        sum_pos = 0.0
                        for e in range(ns):
                            it = sit[e]
                            q = sq[e]
                            nw = nu * weight[it]
                            if sp[e]:
                                gain = -profit[it] + R * (nw * G1[q] - nw * nw * G2[q])
                            else:
                                if st[1] + dW + weight[it] > capacity:
                                    continue
                                gain = profit[it] - R * (nw * G1[q] + nw * nw * G2[q])
                            if gain > 1e-9:
                                sum_pos += gain
                                sg[npos] = -gain
                                se[npos] = e
                                npos += 1
                        if npos == 0:
                            break
                        if screen and ps == 0 and g_plain + sum_pos <= 1e-9:
                            break
                        take = max(1, int(frac * npos))
                        order = np.argsort(sg[:npos])[:take]
                        for o in order:
                            e = se[o]
                            it = sit[e]
                            q = sq[e]
                            if sp[e]:
                                sp[e] = False
                                dp -= profit[it]
                                dW -= weight[it]
                                dwq[q] -= weight[it]
                            elif st[1] + dW + weight[it] <= capacity:
                                sp[e] = True
                                dp += profit[it]
                                dW += weight[it]
                                dwq[q] += weight[it]
                    for e in range(ns):
                        if sp[e] != picked[sit[e]]:
                            flips[nf] = sit[e]
                            nf += 1
                    # --- exact time of the compound move over positions i..n-1 (plain reversal computed above)
                    g_comp = -1e300
                    if nf > 0:
                        t_c = d_first * _inv_v(load[i], vmax, vmin, nu)
                        acc = 0.0
                        for q in range(L):
                            acc += dwq[q]
                            t_c += dn[q] * _inv_v(nl[q] + acc, vmax, vmin, nu)
                        if dW == 0.0:
                            t_c += esuf[j + 1]
                        else:
                            for s in range(j + 1, n):
                                t_c += _d(xy, tour[s], tour[(s + 1) % n]) * _inv_v(load[s] + dW, vmax, vmin, nu)
                        g_comp = dp - R * (t_c - old)
                    use_comp = g_comp > g_plain
                    g = g_comp if use_comp else g_plain
                    if g > 1e-9:
                        e0, e1, e2, e3 = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                        # apply reversal
                        aa, bb = i + 1, j
                        while aa < bb:
                            ca, cb = tour[aa], tour[bb]
                            tour[aa], tour[bb] = cb, ca
                            pos[cb], pos[ca] = aa, bb
                            aa += 1
                            bb -= 1
                        if use_comp:
                            for f in range(nf):
                                it = flips[f]
                                if picked[it]:
                                    picked[it] = False
                                    cw[city[it]] -= weight[it]
                                    st[0] -= profit[it]
                                    st[1] -= weight[it]
                                else:
                                    picked[it] = True
                                    cw[city[it]] += weight[it]
                                    st[0] += profit[it]
                                    st[1] += weight[it]
                        Lc = load[i]
                        for s in range(i + 1, n):
                            Lc += cw[tour[s]]
                            load[s] = Lc
                        for s in range(i, n):
                            et[s] = _d(xy, tour[s], tour[(s + 1) % n]) * _inv_v(load[s], vmax, vmin, nu)
                        _suffix_sums(tour, load, xy, vmax, vmin, nu, g1s, g2s, esuf)
                        gain_total += g
                        moves += 1
                        dlb[e0] = False
                        dlb[e1] = False
                        dlb[e2] = False
                        dlb[e3] = False
                        found = True
                        changed = True
                        break
            if not found:
                dlb[a] = True
            if moves >= max_moves:
                break
        if interrupted:
            break
    return moves, gain_total, interrupted


@njit(cache=True)
def kp_dp(w, a, cap):
    """Exact 0-1 knapsack by dynamic programming: maximise sum a[j] x[j] s.t. sum w[j] x[j] <= cap (w integer >= 0).
    O(len(w) * cap) time, len(w) * (cap + 1) bits of choice table. Returns the boolean choice."""
    m = len(w)
    best = np.zeros(cap + 1)
    take = np.zeros((m, cap + 1), np.bool_)
    for j in range(m):
        wj = w[j]
        aj = a[j]
        if wj > cap or aj <= 0.0:
            continue
        for c in range(cap, wj - 1, -1):
            v = best[c - wj] + aj
            if v > best[c]:
                best[c] = v
                take[j, c] = True
    x = np.zeros(m, np.bool_)
    c = cap
    for j in range(m - 1, -1, -1):
        if take[j, c]:
            x[j] = True
            c -= w[j]
    return x
