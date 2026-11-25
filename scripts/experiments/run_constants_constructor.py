"""Constants of the constructor (Algorithm 2 of the paper) on the 60 instances, one cached chained Lin-Kernighan tour per
TSP base, as in run_constructor_versions.py (Section 5.8, Table 13). One constant changed at a time from the values of
LARC:
  rounds of the load-aware key                3 -> 1, 2, 5
  core half-width h of the knapsack repack    150 -> 50, 100, 300, 600
  rounds of the knapsack repack               3 -> 1, 2, 5
  flip passes / flip candidates               10 / 5000 -> 5, 20 passes; 1000, 20000 candidates
Value and time of the constructor for each value. The values of LARC are run first and again last, so that the time
reference does not depend on the order. Usage: python3 scripts/experiments/run_constants_constructor.py [--workers 8]
Output: runs/constants_constructor.xlsx (as the sheet constructor of results/sensitivity_other_constants.xlsx)."""
import argparse
import functools
import time

from common import RUNS_DIR, run_jobs, select_instances

LARC = dict(key_rounds=3, core_half_width_h=150, repack_rounds=3, flip_passes=10, flip_candidates=5000)
VALUES = [("LARC_values", {})]
VALUES += [(f"key_rounds={v}", dict(key_rounds=v)) for v in (1, 2, 5)]
VALUES += [(f"core_half_width_h={v}", dict(core_half_width_h=v)) for v in (50, 100, 300, 600)]
VALUES += [(f"repack_rounds={v}", dict(repack_rounds=v)) for v in (1, 2, 5)]
VALUES += [(f"flip_passes={v}", dict(flip_passes=v)) for v in (5, 20)]
VALUES += [(f"flip_candidates={v}", dict(flip_candidates=v)) for v in (1000, 20000)]
VALUES += [("LARC_values_run2", {})]

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()
_FLIP = _REPACK = None


def build(c, inst, tour, p):
    """construct.build with the given constants; flip_search and kp_repack are module functions called by build."""
    c.flip_search = functools.partial(_FLIP, max_pass=p["flip_passes"], top_k=p["flip_candidates"])
    c.kp_repack = functools.partial(_REPACK, core=p["core_half_width_h"], rounds=p["repack_rounds"])
    t0 = time.perf_counter()
    v = c.build(inst, tour, mode="la", polish=True, la_rounds=p["key_rounds"]).objective
    return v, time.perf_counter() - t0


def run(job):
    global _FLIP, _REPACK
    name, = job
    from larc import benchmark, construct as c, tours
    if _FLIP is None:
        _FLIP, _REPACK = c.flip_search, c.kp_repack
        warm = "eil76_n75_bounded-strongly-corr_01"            # compilation of the Numba kernels, not timed
        wi = benchmark.load_instance(warm)
        build(c, wi, tours.get_tour(warm, wi, compute=False), LARC)
    inst = benchmark.load_instance(name)
    tour = tours.get_tour(name, inst, compute=False)
    row = dict(instance=name)
    for lab, change in VALUES:
        row[lab], row[lab + "_s"] = build(c, inst, tour, {**LARC, **change})
    return row


if __name__ == "__main__":
    jobs = [(n,) for n in select_instances()]
    run_jobs(run, jobs, RUNS_DIR / "constants_constructor.xlsx", keys=("instance",), workers=A.workers, maxtasks=100,
             label=lambda r: f"{r['instance']}: {r['LARC_values_s']:.1f} s")
