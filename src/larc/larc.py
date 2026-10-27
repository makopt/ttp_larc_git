"""LARC (Load-Aware Racing with Coordinated moves) for the single-objective Traveling Thief Problem.

solve_larc() is the method of the paper. Every step, the computation of tours included, runs inside one wall-clock
budget, and every move is accepted on the exact objective. The constants are those of Table A1 of the paper."""
import dataclasses
import time

import numpy as np

from . import construct, trace
from .instance import LeanTTP
from .recombination import recombine
from .search import State, knn_lists, local_search, macro_search
from .tours import linkern_tour

# search settings shared by every call of macro_search (Algorithm 5): annealing period a = max(5 s, 0.04 T_s), and
# the items of every re-sequenced tour also rebuilt by the constructor
DEFAULTS = dict(anneal_frac=0.04, rebuild_items=True)


def solve_larc(inst: LeanTTP, time_limit: float, seed: int = 1, k_race: int = 4, share_min: float = 0.25,
                 share_max: float = 0.8, k_target: int = 1000, k_min: int = 8, n_probe: int = 2, share_tours: float = 0.45, final_kp: float = 0.03, race_end: float = 0.75,
               xover_share: float = 0.1, sample_stall: bool = False, tour_noise=(0.0, 0.1, 0.25, 0.5), noise_max_n: int = 1500, probe_per_city: float = 2.5e-4,
               probe_min: float = 0.5, probe_max: float = 10.0, log=None, **kw):
    """Run LARC on one instance within a wall-clock budget of time_limit seconds (Algorithm 1 of the paper).

    Phases (Section 3.1):
      1. Sampling (Algorithm 6). Each candidate is a chained Lin-Kernighan tour with the load-aware constructor packing.
         Instances with at most noise_max_n cities: tours on coordinates perturbed by Gaussian noise of standard
         deviation tour_noise[i % 4] times the mean nearest-neighbour distance, a coordinated local search capped at
         share_min * T / k_min, and a sampling end measured from the share rho of searches that converge
         (share_min + (share_max - share_min) * rho, or share_tours if rho < 0.5). Larger instances: plain tours, a probe
         search of min(probe_max, max(probe_min, probe_per_city * n)) seconds per tour, and a sampling end at
         share_tours * T. Sampling also stops at k_target candidates.
      2. Race (Algorithm 7) of the k_race best candidates by successive halving until race_end * T; candidates of rank
         1 and 3 use the coordinated 2-opt, those of rank 2 and 4 do not.
      3. Recombination of the winner with the seven best other candidates for at most xover_share * T.
      4. Final search (Algorithm 8) in chunks of max(30 s, 0.1 T) until (1 - final_kp) * T, with an elite pool of eight
         solutions and a recombination of at most 0.05 T after a chunk without improvement.
      5. Final knapsack repack (Algorithm 3) in the last final_kp * T, kept only if better, then local search.
    Returns (state, info): the best solution found and a dictionary of run statistics."""
    kw = {**DEFAULTS, **kw}
    t0 = time.monotonic()
    el = lambda: time.monotonic() - t0
    nbr = knn_lists(inst, 10)
    kicks = max(10, int((1.0 if inst.n <= 2000 else 0.3) * inst.n))
    ls_cap = share_min * time_limit / k_min
    from scipy.spatial import cKDTree
    d_nn = float(cKDTree(inst.xy).query(inst.xy, 2)[0][:, 1].mean())
    rng_xy = np.random.default_rng(seed * 7919)
    cands, per, per_build, n_conv, n_ls = [], None, 0.0, 0, 0
    s_best, s_last = -np.inf, 0
    large = inst.n > noise_max_n
    probe = min(probe_max, max(probe_min, probe_per_city * inst.n))
    while len(cands) < k_target:
        if sample_stall and len(cands) >= 2 * k_min and len(cands) - s_last >= max(k_min, len(cands) // 2):
            break
        rho = n_conv / n_ls if n_ls else 0.0
        if large:
            stop = share_tours * time_limit
        elif n_ls >= n_probe and rho < 0.5:                    # searches hit their cap: many constructor-scored tours
            stop = share_tours * time_limit
        else:
            stop = (share_min + (share_max - share_min) * rho) * time_limit
        if cands and el() + 1.3 * per > stop:
            break
        ts = time.monotonic()
        i = len(cands)
        lev = tour_noise[i % len(tour_noise)] if tour_noise and inst.n <= noise_max_n else 0.0
        J = dataclasses.replace(inst, xy=inst.xy + rng_xy.normal(0.0, lev * d_nn, inst.xy.shape)) if lev > 0 else inst
        tour, _ = linkern_tour(J, time_limit=max(1.0, share_min * time_limit), seed=seed * 1000 + i, kicks=kicks,
                               init=i % 5)
        tb = time.monotonic()
        b0 = construct.build(inst, tour, mode="la", polish=True)
        trace.observe(b0.objective)
        per_build = max(per_build, time.monotonic() - tb)
        st = State(inst, b0.tour, b0.picked, nbr)
        if large:                                           # probe search of every tour
            local_search(st, time_limit=probe, max_seg=inst.n, slice_moves=50, coord=True)
            n_ls += 1
        elif n_ls < n_probe or n_conv >= 0.5 * n_ls:        # converging regime (or still probing): search every tour
            tl = max(0.5, min(ls_cap, share_max * time_limit - el()))
            tl0 = time.monotonic()
            local_search(st, time_limit=tl, max_seg=inst.n, slice_moves=50, coord=True)
            n_conv += (time.monotonic() - tl0) < 0.9 * tl
            n_ls += 1
        trace.observe_state(st, force=True)
        cands.append(dict(st=st, k=i, built=b0.objective))
        if st.value() > s_best + 1e-9:
            s_best, s_last = st.value(), len(cands)
        dt = time.monotonic() - ts
        per = dt if per is None else 0.7 * per + 0.3 * dt
    t_sample, k0 = el(), len(cands)
    trace.phase("sampling_end")
    if n_ls < len(cands):                                   # mixed searched / unsearched: compare constructor values
        cands.sort(key=lambda c: -c["built"])
    else:
        cands.sort(key=lambda c: -c["st"].value())
    for r, c in enumerate(cands):
        c["coord"] = (r % 2 == 0)
    kw.update(build_est=per_build, ls_max_seg=inst.n, kick=True)
    alive, rnd, history = list(cands[:k_race]), 0, []
    n_rounds = max(1, int(np.ceil(np.log2(max(2, len(alive))))))
    while len(alive) > 1:
        rounds_left = n_rounds - rnd
        r_budget = (race_end * time_limit - el()) / max(1, rounds_left)
        for c in alive:
            share = r_budget / len(alive)
            if share <= 0.2:
                break
            st, _ = macro_search(inst, c["st"].tour, c["st"].picked, nbr, time_limit=share,
                                 seed=seed * 100 + 16 * rnd + c["k"], ls_coord=c["coord"], **kw)
            if st.value() >= c["st"].value():
                c["st"] = st
        for c in alive:
            trace.observe_state(c["st"], force=True)
        trace.phase(f"race_round_{rnd}_end")
        alive.sort(key=lambda c: -c["st"].value())
        history.append([(c["k"], round(c["st"].value(), 1)) for c in alive])
        if log:
            log(f"t={el():.0f}s round {rnd}: {history[-1][:4]}")
        alive = alive[:max(1, (len(alive) + 1) // 2)]
        rnd += 1
        if el() >= race_end * time_limit:
            break
    cands.sort(key=lambda c: -c["st"].value())
    win = alive[0] if alive else cands[0]
    if cands[0]["st"].value() > win["st"].value():
        win = cands[0]
    info = dict(candidates=k0, sample_s=round(t_sample, 1), converged=round(n_conv / max(1, n_ls), 2), searched=n_ls, race_rounds=rnd, race_winner=win["k"],
                winner_coord=win["coord"], race_best=win["st"].value())
    as_built = lambda c: construct.Built(c["st"].value(), c["st"].picked, c["st"].tour, 0.0, "race", "race")
    pool = [as_built(win)] + [as_built(c) for c in cands if c is not win][:7]
    st_best = win["st"]
    if len(pool) > 1 and xover_share > 0:
        P, wins = recombine(inst, pool, nbr, deadline=time.monotonic() + xover_share * time_limit)
        trace.observe_state(P, force=True)
        trace.phase("recombination_end")
        info["xover_wins"] = wins
        if P.value() > st_best.value():
            st_best = P
    final_reserve = final_kp * time_limit if construct.KP_REPACK else 0.0
    # final search in chunks; a chunk without improvement triggers a recombination with the elite pool
    end = time_limit - final_reserve
    chunk = max(30.0, 0.1 * time_limit)
    elite = [c["st"] for c in cands[:8]]
    st, stats, j = st_best, dict(macro=0, macro_win=0, kick=0, kick_win=0, xover_stall=0), 0
    while end - el() > 1.0:
        j += 1
        y, s_ = macro_search(inst, st.tour, st.picked, nbr, time_limit=min(chunk, end - el()),
                             seed=seed * 100 + 99 + j, ls_coord=win["coord"], **kw)
        for k_ in ("macro", "macro_win", "kick", "kick_win"):
            stats[k_] += s_[k_]
        trace.observe_state(y, force=True)
        if y.value() > st.value() + 1e-9:
            st = y
            elite = sorted(elite + [y], key=lambda e: -e.value())[:8]
        elif end - el() > 2.0:
            base = construct.Built(st.value(), st.picked, st.tour, 0.0, "elite", "elite")
            others = [construct.Built(e.value(), e.picked, e.tour, 0.0, "elite", "elite") for e in elite
                      if e.value() < st.value() - 1e-9][:7]
            if others:
                P, _ = recombine(inst, [base] + others, nbr, deadline=min(time.monotonic() + 0.05 * time_limit,
                                                                          time.monotonic() + end - el()))
                stats["xover_stall"] += 1
                trace.observe_state(P, force=True)
                if P.value() > st.value() + 1e-9:
                    st = P
    trace.observe_state(st, force=True)
    trace.phase("final_search_end")
    # knapsack-exact repack of the final solution, then local search. The repack runs only if the time left exceeds
    # the measured constructor time per_build, which includes one repack.
    if final_reserve > 0 and time_limit - el() > max(0.5, per_build):
        pk, val = construct.kp_repack(inst, st.tour, st.picked)
        trace.observe(val)
        if val > st.value() + 1e-9:
            cand = State(inst, st.tour, pk, nbr)
            local_search(cand, time_limit=max(0.2, time_limit - el()), max_seg=inst.n, slice_moves=10, coord=True)
            if cand.value() > st.value():
                st = cand
                info["final_kp"] = True
    if final_reserve > 0 and time_limit - el() > 0.5:   # unused reserve: keep improving the final solution
        st = State(inst, st.tour, st.picked, nbr)
        local_search(st, time_limit=max(0.2, time_limit - el()), max_seg=inst.n, slice_moves=10, coord=True)
    trace.observe_state(st, force=True)
    trace.phase("end")
    info.update(time_s=el(), constructor=max(c["built"] for c in cands), **stats)
    return st, info
