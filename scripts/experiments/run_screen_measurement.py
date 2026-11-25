"""What the first-pass screen of the coordinated 2-opt rejects (Section 3.3.3).

The local search of LARC is run with the coordinated 2-opt and WITHOUT the screen, so that every candidate receives all
its repacking passes. An instrumented copy of the kernel, made at run time from the source of
larc.kernels.coord_two_opt_ls with counters added and no other change, records for every candidate:
  - whether its first repacking pass finds items with a positive estimated gain;
  - whether the screen would have rejected it (exact gain of the plain reversal plus the estimated gains of the first
    pass not positive);
  - for these rejected candidates, whether the complete repacking would have given an improving coordinated move
    (exact gain of the compound move positive and better than the plain reversal), and the work (positions and items
    visited) that the repacking passes after the first one took.
Setting: the instances, tours and time of the experiment first_pass_screen of run_search_probes.py (6 instances with
five items per city, 2 chained Lin-Kernighan tours each, constructor of LARC, 120 s of local search).

Usage: python3 scripts/experiments/run_screen_measurement.py [--workers 8] [--seconds 120]
Output: runs/screen_measurement.xlsx (one row per instance and tour).
"""
import argparse
import inspect
import re
import time

import numpy as np

from common import RUNS_DIR, run_jobs, select_instances

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--seconds", type=float, default=120.0)
ap.add_argument("--names", default="")
A = ap.parse_args()

COUNTERS = ["candidates_with_positive_first_pass", "rejected_by_screen", "rejected_but_improving_compound",
            "rejected_but_improving_plain", "work_total", "work_after_first_pass_on_rejected"]


def instrumented_kernel(show=False):
    """The coordinated 2-opt kernel of LARC with counters (argument cnt, int64 array of len(COUNTERS))."""
    from numba import njit
    from larc import kernels as K
    original = getattr(K, "_coord_two_opt_ls_original", K.coord_two_opt_ls)
    K._coord_two_opt_ls_original = original
    src = inspect.getsource(original.py_func)
    src = re.sub(r"^@njit.*\n", "", src, flags=re.M)
    src = src.replace("passes, frac, screen=False):", "passes, frac, screen, cnt):", 1)

    def insert(anchor, lines, after=False):
        nonlocal src
        hits = [l for l in src.splitlines(keepends=True) if l.strip() == anchor or l.strip().startswith(anchor)]
        assert len(hits) == 1, f"kernel source changed: {anchor!r} found {len(hits)} times"
        line = hits[0]
        ind = line[:len(line) - len(line.lstrip())]
        block = "".join(ind + l + "\n" for l in lines)
        src = src.replace(line, line + block if after else block + line, 1)

    insert("for ps in range(passes):", ["scr = False", "W1 = work"])
    insert("if screen and ps == 0 and g_plain + sum_pos <= 1e-9:",
           ["if ps == 0:", "    cnt[0] += 1", "    if g_plain + sum_pos <= 1e-9:", "        cnt[1] += 1",
            "        scr = True", "        W1 = work"])
    insert("use_comp = g_comp > g_plain", ["if scr:", "    cnt[5] += work - W1", "    if use_comp and g_comp > 1e-9:",
                                           "        cnt[2] += 1", "    elif g_plain > 1e-9:", "        cnt[3] += 1"],
           after=True)
    insert("return moves, gain_total, interrupted", ["cnt[4] += work"])
    if show:
        print(src)
    ns = dict(vars(K))
    exec(compile(src, "<instrumented coord_two_opt_ls>", "exec"), ns)
    return njit(cache=False)(ns["coord_two_opt_ls"])


KERNEL = None


def run(job):
    name, i = job
    from larc import benchmark, construct, tours
    from larc import search as S
    global KERNEL
    if KERNEL is None:
        KERNEL = instrumented_kernel()
    kern = KERNEL
    cnt = np.zeros(len(COUNTERS), dtype=np.int64)

    def patched(*args):
        return kern(*args[:-1], False, cnt)                   # never screen: every candidate gets all its passes

    S.K.coord_two_opt_ls = patched
    inst = benchmark.load_instance(name)
    tour, _ = tours.linkern_tour(inst, time_limit=120, seed=9000 + i, kicks=int(0.3 * inst.n), init=i % 5)
    s = construct.build(inst, tour, mode="la", polish=True)
    st = S.State(inst, s.tour, s.picked, S.knn_lists(inst, 10))
    S.local_search(S.State(inst, s.tour, s.picked, S.knn_lists(inst, 10)), time_limit=1, max_seg=inst.n,
                   slice_moves=50, coord=True)                # compiles the kernel before the measurement
    cnt[:] = 0
    t0 = time.monotonic()
    S.local_search(st, time_limit=A.seconds, max_seg=inst.n, slice_moves=50, coord=True)
    row = dict(instance=name, tour=i, seconds=time.monotonic() - t0, start_value=s.objective, final_value=st.value())
    row.update({k: int(v) for k, v in zip(COUNTERS, cnt)})
    return row


if __name__ == "__main__":
    jobs = [(n, i) for n in select_instances("first_pass_screen", A.names) for i in range(2)]
    run_jobs(run, jobs, RUNS_DIR / "screen_measurement.xlsx", ("instance", "tour"), A.workers,
             lambda r: f"{r['instance']} tour {r['tour']}: rejected {r['rejected_by_screen']}, "
                       f"improving after full repacking {r['rejected_but_improving_compound']}")
