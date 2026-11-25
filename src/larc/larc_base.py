"""The configuration of LARC used in the grouped ablation of Section 5.6 of the paper."""
import time

import numpy as np

from . import construct, trace
from .instance import LeanTTP
from .larc import DEFAULTS
from .recombination import recombine
from .search import State, knn_lists, local_search, macro_search
from .tours import linkern_tour


def solve_larc_base(inst: LeanTTP, time_limit: float, seed: int = 1, k_race: int = 4, share_min: float = 0.25,
               share_max: float = 0.8, k_target: int = 1000, k_min: int = 8, n_probe: int = 2, share_tours: float = 0.45, final_kp: float = 0.03, race_end: float = 0.75,
               xover_share: float = 0.1, log=None, **kw):
    """LARC without perturbed tours, probe screening, double-bridge kicks and chunked final search (Section 5.6).

    Every candidate tour is a plain chained Lin-Kernighan tour, the measured sampling rule is used on every instance
    (so that on large instances most candidates are ranked by their constructor value), and the search after the
    recombination is one call of macro_search without kicks. The race, the recombination and the final repack are those
    of solve_larc. Returns (state, info)."""
    kw = {**DEFAULTS, **kw}
    t0 = time.monotonic()
    el = lambda: time.monotonic() - t0
    nbr = knn_lists(inst, 10)
    kicks = max(10, int((1.0 if inst.n <= 2000 else 0.3) * inst.n))
    ls_cap = share_min * time_limit / k_min
    cands, per, per_build, n_conv, n_ls = [], None, 0.0, 0, 0
    while len(cands) < k_target:
        phi = n_conv / n_ls if n_ls else 0.0
        if n_ls >= n_probe and phi < 0.5:                    # searches hit their cap: many constructor-scored tours
            stop = share_tours * time_limit
        else:
            stop = (share_min + (share_max - share_min) * phi) * time_limit
        if cands and el() + 1.3 * per > stop:
            break
        ts = time.monotonic()
        i = len(cands)
        tour, _ = linkern_tour(inst, time_limit=max(1.0, share_min * time_limit), seed=seed * 1000 + i, kicks=kicks,
                               init=i % 5)
        tb = time.monotonic()
        b0 = construct.build(inst, tour, mode="la", polish=True)
        trace.observe(b0.objective)
        per_build = max(per_build, time.monotonic() - tb)
        st = State(inst, b0.tour, b0.picked, nbr)
        if n_ls < n_probe or n_conv >= 0.5 * n_ls:          # converging regime (or still probing): search every tour
            tl = max(0.5, min(ls_cap, share_max * time_limit - el()))
            tl0 = time.monotonic()
            local_search(st, time_limit=tl, max_seg=inst.n, slice_moves=50, coord=True)
            n_conv += (time.monotonic() - tl0) < 0.9 * tl
            n_ls += 1
        trace.observe_state(st, force=True)
        cands.append(dict(st=st, k=i, built=b0.objective))
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
    kw.update(build_est=per_build, ls_max_seg=inst.n)
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
    st, stats = macro_search(inst, st_best.tour, st_best.picked, nbr, time_limit=max(0.0, time_limit - final_reserve - el()),
                             seed=seed * 100 + 99, ls_coord=win["coord"], **kw)
    if st_best.value() > st.value():
        st = st_best
    trace.observe_state(st, force=True)
    trace.phase("final_search_end")
    # knapsack-exact repack of the final solution, then local search. The repack runs only if the time left exceeds
    # the measured constructor time per_build, which includes one repack.
    if final_reserve > 0 and time_limit - el() > max(0.5, per_build):
        pk, val = construct.kp_repack(inst, st.tour, st.picked)
        trace.observe(val)
        if val > st.value() + 1e-9:
            cand = State(inst, st.tour, pk, nbr)
            local_search(cand, time_limit=max(0.2, time_limit - el()), max_seg=inst.n, slice_moves=50, coord=True)
            if cand.value() > st.value():
                st = cand
                info["final_kp"] = True
    if final_reserve > 0 and time_limit - el() > 0.5:   # unused reserve: keep improving the final solution
        st = State(inst, st.tour, st.picked, nbr)
        local_search(st, time_limit=max(0.2, time_limit - el()), max_seg=inst.n, slice_moves=50, coord=True)
    trace.observe_state(st, force=True)
    trace.phase("end")
    info.update(time_s=el(), constructor=max(c["built"] for c in cands), **stats)
    return st, info
