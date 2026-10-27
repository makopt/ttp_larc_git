"""Section 5.2: runs LARC again (600 s, as in the main runs) with the best seed of each instance where the best of its
10 main runs reaches or exceeds every published value of the six methods that use the benchmark distances (CoCo+, CoCo,
RWS, SAVI, S5 and MATLS), and saves the final solution. A run stops on wall-clock time, so it does not repeat the main
run exactly; the value of the new run is the one reported.

Each solution is checked by a second evaluator, written here from the definition of the benchmark and independent of
the code of LARC: it reads the original .ttp file (data/<base>-ttp/, created by scripts/setup/prepare_instances.py) and
rounds every distance up, as TTPInstance.evaluate of the benchmark code.

Usage: python3 scripts/experiments/run_best_solutions.py [--workers 8]
Output: runs/best_solutions.xlsx and runs/best_solutions/<instance>.txt (first line the tour, city numbers from 1,
starting at city 1; second line the picked items, item numbers from 1, as in the .ttp file). Results of the paper:
results/best_solutions.xlsx and results/best_solutions/."""
import argparse
import math
import time

import numpy as np
import pandas as pd

from common import REPO, RUNS_DIR, benchmark, run_jobs

T = 600.0
ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()


def read_ttp(path):
    head, xy, items, sec = {}, [], [], None
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("NODE_COORD_SECTION"):
            sec = "n"
            continue
        if s.startswith("ITEMS SECTION"):
            sec = "i"
            continue
        if sec is None:
            k, v = s.split(":", 1)
            head[k.strip()] = v.strip()
        elif sec == "n":
            _, x, y = s.split()[:3]
            xy.append((float(x), float(y)))
        else:
            _, p, w, node = s.split()[:4]
            items.append((int(p), int(w), int(node)))
    return head, np.array(xy), np.array(items, dtype=np.int64)


def independent_value(name, tour1, items1):
    """Objective of a solution in the format of the competitions, computed from the .ttp file only."""
    head, xy, it = read_ttp(REPO / "data" / f"{name.split('_')[0]}-ttp" / f"{name}.ttp")
    W, vmin, vmax = int(head["CAPACITY OF KNAPSACK"]), float(head["MIN SPEED"]), float(head["MAX SPEED"])
    R = float(head["RENTING RATIO"])
    assert sorted(tour1) == list(range(1, len(xy) + 1)) and tour1[0] == 1
    picked = np.zeros(len(it), bool)
    picked[np.array(items1, dtype=int) - 1] = True
    assert int(it[picked, 1].sum()) <= W
    cw = np.bincount(it[picked, 2], weights=it[picked, 1], minlength=len(xy) + 1)
    t, wc, n = 0.0, 0.0, len(tour1)
    for i in range(n):
        a, b = tour1[i], tour1[(i + 1) % n]
        wc += cw[a]
        t += math.ceil(math.hypot(*(xy[a - 1] - xy[b - 1]))) / (vmax - wc * (vmax - vmin) / W)
    return float(it[picked, 0].sum() - R * t)


def run(job):
    name, seed, main_best, published_best = job
    t_start = time.monotonic()
    from larc.evaluator import is_feasible, objective
    from larc.larc import solve_larc
    inst = benchmark.load_instance(name)
    st, _ = solve_larc(inst, T - (time.monotonic() - t_start), seed=seed)
    wall = time.monotonic() - t_start
    tour1 = [int(c) + 1 for c in st.tour]
    items1 = [int(k) + 1 for k in np.flatnonzero(st.picked)]
    out = RUNS_DIR / "best_solutions"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.txt").write_text("[" + ",".join(map(str, tour1)) + "]\n[" + ",".join(map(str, items1)) + "]\n")
    return dict(instance=name, seed=seed, objective=objective(inst, st.tour, st.picked),
                objective_independent_evaluator=independent_value(name, tour1, items1),
                feasible=bool(is_feasible(inst, st.tour, st.picked)), time_s=wall,
                best_of_the_10_main_runs=main_best, best_published_value_6_methods=published_best)


if __name__ == "__main__":
    runs = pd.read_excel(REPO / "results/comparison_LARC_vs_CoCoPlus.xlsx", sheet_name="runs")
    runs = runs[runs.method == "LARC"]
    best = runs.loc[runs.groupby("instance").objective.idxmax()].set_index("instance")
    pub = benchmark.published_values().loc[best.index].drop(columns=["CS2SA", "MA2B", "CS2SA*"]).max(axis=1)
    sel = best[best.objective > pub - 0.5]
    jobs = [(n, int(r.seed), float(r.objective), float(pub[n])) for n, r in sel.iterrows()]
    run_jobs(run, jobs, RUNS_DIR / "best_solutions.xlsx", keys=("instance",), workers=A.workers,
             label=lambda r: f"{r['instance']}: {r['objective']:.0f} (independent {r['objective_independent_evaluator']:.0f})")
