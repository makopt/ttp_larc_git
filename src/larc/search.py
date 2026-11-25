"""Alternating tour / item local search with exact deltas (Phase 2a baseline: 2-opt on neighbour lists + flip search)."""
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from . import ablation as AB
from . import kernels as K
from . import trace
from .evaluator import city_weights, objective
from .instance import LeanTTP


def knn_lists(inst: LeanTTP, k: int = 10) -> np.ndarray:
    """k nearest neighbours of every city (self excluded), int64 (n, k)."""
    _, idx = cKDTree(inst.xy).query(inst.xy, k=min(k + 1, inst.n))
    return np.ascontiguousarray(idx[:, 1:], dtype=np.int64)


@dataclass
class State:
    inst: LeanTTP
    tour: np.ndarray
    picked: np.ndarray
    nbr: np.ndarray
    history: list = field(default_factory=list)

    def __post_init__(self):
        assert self.inst.edge_type == "CEIL_2D"
        self.tour = np.array(self.tour, dtype=np.int64, copy=True)   # always copy: kernels modify it in place
        self.picked = self.picked.copy()
        self.refresh()

    def refresh(self):
        i = self.inst
        self.pos = np.empty(i.n, np.int64)
        self.pos[self.tour] = np.arange(i.n)
        self.cw = city_weights(i, self.picked)
        self.load, self.et = K.init_state(self.tour, self.cw, i.xy, i.v_max, i.v_min, i.nu)
        self.wtot = float(i.weight[self.picked].sum())

    def value(self) -> float:
        return float(self.inst.profit[self.picked].sum()) - self.inst.R * float(self.et.sum())


def local_search(st: State, time_limit: float = 60.0, max_seg: int = 1000, slack: float = 1.0,
                 top_k: int = 5000, max_rounds: int = 1000, log=None, swaps: bool = True,
                 kp: int = 100, kq: int = 100, max_verify: int = 300, or_opt: bool = True, max_len: int = 3,
                 dist_slack: float = 1e18, compound: bool = False, tau_frac: float = 3.0, max_attempts: int = 2000,
                 slice_moves: int = 500, reversal: bool = False, coord: bool = False, coord_passes: int = 40, coord_frac: float = None,
                 coord_screen: bool = True):        # first-pass screen of the coordinated 2-opt
    """Alternate tour moves (2-opt, Or-opt; items fixed) and item moves (flips, swaps; tour fixed) until no improvement
    or the time limit. Tour kernels run in slices of `slice_moves` moves with persistent don't-look bits, so the clock
    is checked often even on the largest instances. `compound` (2-opt + segment repack) is off by default: it copies
    the full state per attempt and gave little gain. The damping coord_frac is 0.25 unless the variable LARC_COORD_FRAC
    is set, and the switches of ablation.py can turn off the coordinated 2-opt and its first-pass screen."""
    coord = coord and AB.COORD                      # switches of the ablation (ablation.py); no effect when unset
    coord_screen = coord_screen and AB.SCREEN
    if coord_frac is None:
        coord_frac = AB.COORD_FRAC
    i = st.inst
    t0 = time.monotonic()
    deadline = t0 + time_limit
    csr = tmp = None
    dlb2 = np.zeros(i.n, np.bool_)
    dlbo = np.zeros(i.n, np.bool_)
    dlbr = np.zeros(i.n, np.bool_)
    dlbc = np.zeros(i.n, np.bool_)
    st.history.append((0.0, st.value()))
    for rnd in range(max_rounds):
        mv = mo = n_pl = n_cp = fl = sw = 0
        g = gf = 0.0
        while time.monotonic() < deadline:
            m_, g_, intr = K.two_opt_ls(st.tour, st.pos, st.cw, st.load, st.et, i.xy, st.nbr, i.v_max, i.v_min, i.nu,
                                        i.R, max_seg, slack, slice_moves, dlb2, 20_000_000)
            mv += m_
            g += g_
            trace.observe_state(st)
            if m_ < slice_moves and not intr:
                break
        n_co = 0
        if coord:                                      # fast coordinated 2-opt (reversal + load-aware item repack)
            if csr is None:
                csr = _item_csr(i)
            stv = np.array([float(i.profit[st.picked].sum()), st.wtot])
            while time.monotonic() < deadline:
                m_, _, intr = K.coord_two_opt_ls(st.tour, st.pos, st.picked, st.cw, st.load, st.et, i.xy, st.nbr,
                                                 i.profit, i.weight, i.city, csr[0], csr[1], i.capacity, stv, i.v_max,
                                                 i.v_min, i.nu, i.R, max(max_seg, i.n), slice_moves, dlbc, 20_000_000, coord_passes, coord_frac,
                                                 coord_screen)
                n_co += m_
                trace.observe_state(st)
                if m_ < slice_moves and not intr:
                    break
            st.wtot = float(stv[1])
        n_rv = 0
        if reversal:                                   # CoCo-style reversals between neighbouring cities (long new edges)
            while time.monotonic() < deadline:
                m_, _, intr = K.reversal_ls(st.tour, st.pos, st.cw, st.load, st.et, i.xy, st.nbr, i.v_max, i.v_min,
                                            i.nu, i.R, max(max_seg, i.n), slice_moves, dlbr, 20_000_000)
                n_rv += m_
                if m_ < slice_moves and not intr:
                    break
        if compound and time.monotonic() < deadline:
            if csr is None:
                csr = _item_csr(i)
                tmp = (np.empty(i.n, np.int64), np.empty(i.n, np.int64), np.empty(i.m, np.bool_), np.empty(i.n),
                       np.empty(i.n), np.empty(i.n), np.empty(i.n + 4, np.int64))
            stv = np.array([float(i.profit[st.picked].sum()), st.wtot])
            n_pl, n_cp, _ = K.two_opt_repack(st.tour, st.pos, st.picked, st.cw, st.load, st.et, i.xy, st.nbr, i.profit,
                                             i.weight, i.city, csr[0], csr[1], i.capacity, stv, i.v_max, i.v_min, i.nu,
                                             i.R, max_seg, tau_frac * float(i.profit.mean()), max_attempts,
                                             np.zeros(i.n, np.bool_), *tmp)
            st.wtot = float(stv[1])
        if or_opt:
            while time.monotonic() < deadline:
                m_, _ = K.or_opt_ls(st.tour, st.pos, st.cw, st.load, st.et, i.xy, st.nbr, i.v_max, i.v_min, i.nu, i.R,
                                    max_seg, max_len, dist_slack, slice_moves, dlbo)
                mo += m_
                if m_ < slice_moves:
                    break
        if time.monotonic() < deadline:
            fl, gf, st.wtot = K.flip_pass(st.tour, st.pos, st.picked, st.cw, st.load, i.xy, i.profit, i.weight, i.city,
                                          i.capacity, st.wtot, i.v_max, i.v_min, i.nu, i.R, top_k)
        if swaps:
            for _ in range(20):                                   # repeat while swaps keep improving
                if time.monotonic() >= deadline:
                    break
                n_sw, gs, st.wtot = K.swap_pass(st.tour, st.pos, st.picked, st.cw, st.load, i.xy, i.profit, i.weight,
                                                i.city, i.capacity, st.wtot, i.v_max, i.v_min, i.nu, i.R, kp, kq, max_verify)
                sw += n_sw
                if n_sw == 0:
                    break
        st.load, st.et = K.init_state(st.tour, st.cw, i.xy, i.v_max, i.v_min, i.nu)   # et is not kept by flips/swaps
        if fl or sw:                                              # loads changed: tour moves may pay anywhere again
            dlb2[:] = False
            dlbo[:] = False
            dlbr[:] = False
            dlbc[:] = False
        st.history.append((time.monotonic() - t0, st.value()))
        trace.observe(st.history[-1][1])
        if log:
            log(f"round {rnd}: 2opt moves={mv} coord={n_co} rev={n_rv} oropt={mo} compound={n_cp}/{n_pl} gain={g:.1f} | flips={fl} gain={gf:.1f} swaps={sw} | obj={st.value():.1f}")
        if mv == 0 and mo == 0 and fl == 0 and sw == 0 and n_pl == 0 and n_cp == 0 and n_rv == 0 and n_co == 0:
            break
        if time.monotonic() >= deadline:
            break
    return st


def check(st: State) -> float:
    """Independent re-evaluation with the validated evaluator; returns |incremental - exact|."""
    return abs(st.value() - objective(st.inst, st.tour, st.picked))


def _item_csr(inst: LeanTTP):
    order = np.argsort(inst.city, kind="stable").astype(np.int64)
    start = np.zeros(inst.n + 1, np.int64)
    np.cumsum(np.bincount(inst.city, minlength=inst.n), out=start[1:])
    return start, order


def anneal(st: State, time_limit: float = 30.0, t0_frac: float = 0.3, t1_frac: float = 0.001, win: int = 30,
           p_item: float = 0.5, max_seg: int = 100, max_len: int = 3, chunk: int = 50, seed: int = 1, log=None,
           target_chunk_s: float = 0.25):
    """ILS with Metropolis acceptance; temperature decays geometrically from t0_frac to t1_frac times the mean item
    profit over the time budget. Returns the best State found (also stored back into `st`)."""
    i = st.inst
    n = i.n
    istart, iorder = _item_csr(i)
    K.seed_rng(seed)
    st.refresh()
    stv = np.array([float(i.profit[st.picked].sum()), st.wtot, st.value()])
    b_tour, b_picked = st.tour.copy(), st.picked.copy()
    tmp = (np.empty(n, np.int64), np.empty(n, np.int64), np.empty(i.m, np.bool_), np.empty(n), np.empty(n), np.empty(n))
    touched = np.empty(2 * win + 40, np.int64)
    dlb = np.ones(n, np.bool_)
    scale = float(i.profit.mean())
    t_start = time.monotonic()
    tot_it = acc = 0
    while True:
        now = time.monotonic()
        frac = (now - t_start) / time_limit if time_limit > 0 else 1.0
        if frac >= 1.0:
            break
        T = scale * t0_frac * (t1_frac / t0_frac) ** frac
        a, nb = K.ils_chunk(st.tour, st.pos, st.picked, st.cw, st.load, st.et, i.xy, st.nbr, i.profit, i.weight,
                            i.city, istart, iorder, i.capacity, i.v_max, i.v_min, i.nu, i.R, stv, chunk, T, win,
                            p_item, max_seg, max_len, dlb, b_tour, b_picked, *tmp, touched)
        dt = max(time.monotonic() - now, 1e-6)
        tot_it += chunk
        acc += a
        left = time_limit - (time.monotonic() - t_start)                # adapt the chunk so one call takes ~target_chunk_s
        chunk = int(max(1, min(50000, chunk * min(target_chunk_s, max(left, 0.01)) / dt)))
        st.history.append((time.monotonic() - t_start, stv[2]))
        trace.observe(float(stv[2]))
        if log and len(st.history) % 20 == 0:
            log(f"t={time.monotonic()-t_start:.0f}s it={tot_it} acc={acc} cur={stv[0]-i.R*st.et.sum():.1f} best={stv[2]:.1f} T={T:.2f}")
    st.tour, st.picked = b_tour, b_picked
    st.refresh()
    return st


def double_bridge(tour: np.ndarray, rng, window: int) -> np.ndarray:
    """Local double-bridge kick: inside a random window of the tour (depot excluded), segments A B C D -> A C B D."""
    n = len(tour)
    L = min(n - 1, window)
    if L < 4:
        return tour.copy()
    s = int(rng.integers(1, n - L + 1))
    a, b, c = sorted(rng.choice(np.arange(1, L), 3, replace=False))
    seg = tour[s:s + L]
    t = tour.copy()
    t[s:s + L] = np.concatenate([seg[:a], seg[b:c], seg[a:b], seg[c:]])
    return t


def macro_search(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray, nbr: np.ndarray, time_limit: float,
                 anneal_s: float = None, qs=(0.0, 0.25, 0.5, 0.75), seed: int = 1, log=None,
                 anneal_frac: float = 0.1, q_mode: str = "uniform", rebuild_items: bool = False,
                 strata3_prob: float = 0.0, anneal_min: float = 5.0, ls_max_seg: int = 1000,
                 build_est: float = 0.0, ls_compound: bool = False, ls_coord: bool = False, kick: bool = False):
    """Annealing interleaved with light-heavy re-sequencing (LHR) macro-moves.
    Each macro step rebuilds the tour of the best solution as depot -> light cities -> heavy cities (random threshold q,
    best of 4 orientations after local search), anneals it briefly and keeps it if it beats the best.
    kick=True: when the best value has not improved for max(10 s, 2 * annealing period), the iterations
    alternate LHR moves with double-bridge kicks of the best solution (local search and annealing, kept if better).
    Returns (best_state, stats)."""
    from .evaluator import city_weights
    from .tours import split_tours, strata_tours
    rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    left = lambda: time_limit - (time.monotonic() - t0)
    if anneal_s is None:
        anneal_s = max(anneal_min, anneal_frac * time_limit)
    best = State(inst, tour, picked, nbr)
    local_search(best, time_limit=0.3 * time_limit if ls_max_seg <= 1000 else 0.9 * time_limit, max_seg=ls_max_seg,
                 slice_moves=500 if ls_max_seg <= 1000 else 20,      # long reversals: check the clock more often
                 compound=ls_compound, max_attempts=10 ** 6, coord=ls_coord)
    anneal(best, time_limit=min(anneal_s, max(0.0, left())), seed=seed)
    stats = dict(macro=0, macro_win=0, kick=0, kick_win=0)
    wins = np.zeros(len(qs))
    tries = np.zeros(len(qs))
    k = 0
    stall_after = max(10.0, 2.0 * anneal_s)
    t_best, v_best, do_kick = time.monotonic(), best.value(), False
    while left() > 1.0:
        k += 1
        if best.value() > v_best + 1e-9:
            t_best, v_best = time.monotonic(), best.value()
        if kick and time.monotonic() - t_best > stall_after:
            do_kick = not do_kick                           # stagnation: alternate kicks and LHR moves
        else:
            do_kick = False
        if do_kick:
            t_k, v_k = time.monotonic(), best.value()
            AB.STATS["kick_tries"] += 1
            s = State(inst, double_bridge(best.tour, rng, int(rng.integers(8, 61))), best.picked, nbr)
            local_search(s, time_limit=max(0.2, min(0.05 * time_limit, 0.2 * left())), coord=ls_coord)
            anneal(s, time_limit=max(0.0, min(anneal_s, left())), seed=seed + 2000 + k)
            stats["kick"] += 1
            if s.value() > best.value() + 1e-9:
                best = s
                stats["kick_win"] += 1
                AB.STATS["kick_wins"] += 1
                AB.STATS["kick_gain"] += best.value() - v_k
            AB.STATS["kick_time"] += time.monotonic() - t_k
            continue
        if not AB.LHR:                                      # ablation: annealing of the incumbent instead of an LHR step
            anneal(best, time_limit=min(anneal_s, left()), seed=seed + 1000 + k)
            continue
        t_l, v_l = time.monotonic(), best.value()
        AB.STATS["lhr_tries"] += 1
        if q_mode == "bandit":                              # Thompson sampling on the success rate of each q
            qi = int(np.argmax(rng.beta(wins + 1.0, tries - wins + 1.0)))
        else:
            qi = int(rng.integers(len(qs)))
        q = float(qs[qi])
        tl_sub = max(0.5, min(5.0, 0.1 * left()))
        if strata3_prob > 0 and rng.random() < strata3_prob:
            q1, q2 = sorted(rng.choice([0.0, 0.25, 0.5, 0.75], 2, replace=False))
            cands = strata_tours(inst, city_weights(inst, best.picked), (q1, q2), seed=seed + k, time_limit=tl_sub)
        else:
            cands = split_tours(inst, city_weights(inst, best.picked), q, seed=seed + k, time_limit=tl_sub)
        if not cands:
            AB.STATS["lhr_time"] += time.monotonic() - t_l
            if k > 50:
                break
            continue
        cand = None
        starts = [(tr, best.picked) for tr in cands]
        if rebuild_items:                                    # also re-decide the items from scratch on each new tour
            from . import construct
            for tr in cands:
                if left() < max(0.3 * anneal_s, 1.2 * build_est):     # build_est: measured constructor time
                    break
                b_ = construct.build(inst, tr, mode="la", polish=True, both_orientations=False)
                starts.append((tr, b_.picked))
        for tr, pk in starts:
            if left() <= 0.5:
                break
            s = State(inst, tr, pk, nbr)
            local_search(s, time_limit=max(0.2, min(0.05 * time_limit, 0.2 * left())), coord=ls_coord)
            if cand is None or s.value() > cand.value():
                cand = s
        if cand is None:
            AB.STATS["lhr_time"] += time.monotonic() - t_l
            break
        anneal(cand, time_limit=max(0.0, min(anneal_s, left())), seed=seed + k)
        stats["macro"] += 1
        tries[qi] += 1
        if cand.value() > best.value() + 1e-9:
            wins[qi] += 1
            best = cand
            stats["macro_win"] += 1
            AB.STATS["lhr_wins"] += 1
            AB.STATS["lhr_gain"] += best.value() - v_l
            if log:
                log(f"t={time.monotonic()-t0:.0f}s macro {k} q={q}: new best {best.value():.1f}")
        elif left() > 1.0:                                   # intensify the incumbent with the same time
            anneal(best, time_limit=min(anneal_s, left()), seed=seed + 1000 + k)
        AB.STATS["lhr_time"] += time.monotonic() - t_l
    return best, stats
