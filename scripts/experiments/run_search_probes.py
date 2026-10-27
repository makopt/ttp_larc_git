"""Experiments on the coordinated local search (Algorithms 4 and 5 of the paper), each from chained Lin-Kernighan tours
and the constructor of LARC:

  coordinated_2opt   Section 5.6, Table 9: damping f and maximum number of passes P_max of the coordinated 2-opt, one
                     changed at a time from f = 0.25, P_max = 40; 2 tours on each of 8 instances; value after 30, 60
                     and 120 s.
  first_pass_screen  Section 3.3.3: the coordinated 2-opt with and without the first-pass screen; 2 tours on each of
                     6 instances; value after 30, 60 and 120 s.
  race_probe         Section 3.5.2: 16 tours per instance on 6 instances; value after 10, 20, 40, 80 and 160 s, to see
                     whether a short search predicts the result of a long one.
  probe_screening    Section 3.5.1 and Figure 9: constructor value and values after a probe of 0.5 s and 2 s more, for
                     many tours of one instance (d2103 with 320 tours: --tours 40 --seeds 1-8; rl11849 with 40 tours:
                     --tours 5 --seeds 1-8).
  start_tours_noise  Table 4: 30 tours per noise level (0, 0.1, 0.25, 0.5 and 1 times the mean nearest-neighbour
                     distance) on 6 instances, constructor value and value after a short search (2 s on instances with
                     at most 159 cities, 5 s on the others).

Usage: python3 scripts/experiments/run_search_probes.py <experiment> [--workers 8] [--names a,b]
           [--instance NAME --tours K --seeds 1-8]
Output: runs/<experiment>.xlsx, with the columns of the sheets of the same name in results/method_experiments.xlsx
(coordinated_2opt: results/sensitivity_other_constants.xlsx)."""
import argparse
import dataclasses
import time

import numpy as np

from common import RUNS_DIR, run_jobs, seeds, select_instances

ap = argparse.ArgumentParser()
ap.add_argument("experiment", choices=["coordinated_2opt", "first_pass_screen", "race_probe", "probe_screening",
                                       "start_tours_noise"])
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--names", default="", help="only these instances of the list of the experiment (names separated by commas)")
ap.add_argument("--instance", default="d2103_n10510_uncorr-similar-weights_05", help="probe_screening only")
ap.add_argument("--tours", type=int, default=40, help="probe_screening only: tours per seed")
ap.add_argument("--seeds", default="1-8", help="probe_screening only")
A = ap.parse_args()

COORD = [(0.25, 40), (0.1, 40), (0.5, 40), (1.0, 40), (0.25, 10), (0.25, 20), (0.25, 80)]   # (damping f, P_max)
NOISE = (0.0, 0.1, 0.25, 0.5, 1.0)


def start(name, i, seed0, kicks_all_small=False):
    """Chained Lin-Kernighan tour (120 s at most) and the constructor of LARC on it."""
    from larc import benchmark, construct, tours
    inst = benchmark.load_instance(name)
    kicks = inst.n if (kicks_all_small and inst.n <= 2000) else int(0.3 * inst.n)
    tour, _ = tours.linkern_tour(inst, time_limit=120, seed=seed0 + i, kicks=kicks, init=i % 5)
    return inst, construct.build(inst, tour, mode="la", polish=True)


def checkpoints(st, row, marks, prefix, stop_when_converged=False, **kw):
    """Continues the local search up to each cumulative time of marks and records the value."""
    from larc.search import local_search
    spent, prev = 0.0, None
    for k, cp in enumerate(marks):
        t0 = time.monotonic()
        local_search(st, time_limit=cp - spent, max_seg=st.inst.n, slice_moves=50, coord=True, **kw)
        spent += time.monotonic() - t0
        row[f"{prefix}{cp}s"] = st.value()
        if stop_when_converged and prev is not None and st.value() == prev and spent < cp - 0.5:
            for later in marks[k + 1:]:                            # converged before this mark
                row[f"{prefix}{later}s"] = st.value()
            break
        prev = st.value()
    row["time_spent_s"] = spent


def run_coord(job):
    name, i, f, passes = job
    from larc.search import State, check, knn_lists, local_search
    inst, s = start(name, i, 9000)
    kw = dict(coord_frac=f, coord_passes=passes)
    local_search(State(inst, s.tour, s.picked, knn_lists(inst, 10)), time_limit=1, max_seg=inst.n, slice_moves=50,
                 coord=True, **kw)                                 # compilation of the kernels
    st = State(inst, s.tour, s.picked, knn_lists(inst, 10))
    row = dict(instance=name, tour=i, damping_f=f, max_passes=passes, start_value=s.objective)
    checkpoints(st, row, (30, 60, 120), "value_", **kw)
    row["evaluation_error"] = check(st)
    return row


def run_screen(job):
    name, i, screen = job
    from larc.search import State, check, knn_lists, local_search
    inst, s = start(name, i, 9000)
    local_search(State(inst, s.tour, s.picked, knn_lists(inst, 10)), time_limit=1, max_seg=inst.n, slice_moves=50,
                 coord=True, coord_screen=screen)
    st = State(inst, s.tour, s.picked, knn_lists(inst, 10))
    row = dict(instance=name, tour=i, first_pass_screen=screen, start_value=s.objective)
    checkpoints(st, row, (30, 60, 120), "value_", coord_screen=screen)
    row["evaluation_error"] = check(st)
    return row


def run_race(job):
    name, i = job
    from larc.search import State, knn_lists
    inst, s = start(name, i, 7000, kicks_all_small=True)
    st = State(inst, s.tour, s.picked, knn_lists(inst, 10))
    row = dict(instance=name, tour=i, constructor_value=s.objective)
    checkpoints(st, row, (10, 20, 40, 80, 160), "value_", stop_when_converged=True)
    return row


def run_probe(job):
    name, seed = job
    from larc import benchmark, construct, tours
    from larc.search import State, knn_lists, local_search
    inst = benchmark.load_instance(name)
    nbr = knn_lists(inst, 10)
    kicks = max(10, int((1.0 if inst.n <= 2000 else 0.3) * inst.n))
    rows = []
    for i in range(A.tours):
        t0 = time.monotonic()
        tour, _ = tours.linkern_tour(inst, time_limit=150, seed=seed * 1000 + i, kicks=kicks, init=i % 5)
        t1 = time.monotonic()
        b = construct.build(inst, tour, mode="la", polish=True)
        t2 = time.monotonic()
        st = State(inst, b.tour, b.picked, nbr)
        local_search(st, time_limit=0.5, max_seg=inst.n, slice_moves=50, coord=True)
        v05 = st.value()
        local_search(st, time_limit=1.5, max_seg=inst.n, slice_moves=50, coord=True)
        rows.append(dict(instance=name, worker=seed, candidate=i, starting_cycle=i % 5, constructor_value=b.objective,
                         value_after_first_probe=v05, value_after_second_probe=st.value(), tour_time_s=t1 - t0,
                         constructor_time_s=t2 - t1))
    return dict(instance=name, worker=seed, tours=A.tours, _trace=rows)


def run_noise(job):
    name, = job
    from scipy.spatial import cKDTree
    from larc import benchmark, construct
    from larc.search import State, knn_lists, local_search
    from larc.tours import linkern_tour
    inst = benchmark.load_instance(name)
    nbr = knn_lists(inst, 10)
    ls = 2.0 if inst.n <= 159 else 5.0
    dnn = cKDTree(inst.xy).query(inst.xy, 2)[0][:, 1].mean()
    kicks = max(10, int((1.0 if inst.n <= 2000 else 0.3) * inst.n))
    rows = []
    for lev in NOISE:
        rng = np.random.default_rng(7)
        for i in range(30):
            J = dataclasses.replace(inst, xy=inst.xy + rng.normal(0, lev * dnn, inst.xy.shape)) if lev > 0 else inst
            t, _ = linkern_tour(J, time_limit=10, seed=1000 + i, kicks=kicks, init=i % 5)
            edges = sorted(tuple(sorted((int(t[j]), int(t[(j + 1) % inst.n])))) for j in range(inst.n))
            b = construct.build(inst, t, mode="la", polish=True)
            st = State(inst, b.tour, b.picked, nbr)
            local_search(st, time_limit=ls, max_seg=inst.n, slice_moves=50, coord=True)
            closed = np.r_[t, t[:1]]
            rows.append(dict(instance=name, noise_level_sigma=lev, tour_index=i, tour_hash=hash(tuple(edges)),
                             tour_length=float(inst.dist(closed[:-1], closed[1:]).sum()), constructor_value=b.objective,
                             value_after_short_search=st.value()))
    return dict(instance=name, short_search_s=ls, _trace=rows)


if __name__ == "__main__":
    out = RUNS_DIR / f"{A.experiment}.xlsx"
    if A.experiment == "coordinated_2opt":
        jobs = [(n, i, f, p) for n in select_instances("coordinated_2opt", A.names) for i in range(2) for f, p in COORD]
        run_jobs(run_coord, jobs, out, ("instance", "tour", "damping_f", "max_passes"), A.workers,
                 lambda r: f"{r['instance']} tour {r['tour']} f={r['damping_f']} P={r['max_passes']}")
    elif A.experiment == "first_pass_screen":
        jobs = [(n, i, s) for n in select_instances("first_pass_screen", A.names) for i in range(2) for s in (False, True)]
        run_jobs(run_screen, jobs, out, ("instance", "tour", "first_pass_screen"), A.workers,
                 lambda r: f"{r['instance']} tour {r['tour']} screen={r['first_pass_screen']}")
    elif A.experiment == "race_probe":
        jobs = [(n, i) for n in select_instances("race_probe", A.names) for i in range(16)]
        run_jobs(run_race, jobs, out, ("instance", "tour"), A.workers, lambda r: f"{r['instance']} tour {r['tour']}")
    elif A.experiment == "probe_screening":
        jobs = [(A.instance, s) for s in seeds(A.seeds)]
        run_jobs(run_probe, jobs, RUNS_DIR / f"probe_screening_{A.instance}.xlsx", ("instance", "worker"), A.workers,
                 lambda r: f"{r['instance']} worker {r['worker']}")
    else:
        jobs = [(n,) for n in select_instances("start_tours_noise", A.names)]
        run_jobs(run_noise, jobs, out, ("instance",), A.workers, lambda r: r["instance"])
