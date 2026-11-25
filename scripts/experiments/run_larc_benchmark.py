"""Runs LARC (or the configuration without the additions of Section 5.6) on many instances and seeds, one single-threaded
process per run, with a convergence trace: the best value found so far at every improvement and the end time of each
phase. Times are measured from the start of the process, so the reading of the instance counts in the budget.

Usage: python3 scripts/experiments/run_larc_benchmark.py <seconds> <name> [--seeds 1-10] [--workers 8]
           [--list LIST | --list heldout | --names a,b] [--config larc|larc_base] [--param key=value ...]
Output: runs/<name>.xlsx, sheet runs (one row per run) and sheet traces. Started again, it skips the runs already done.

The switches of src/larc/ablation.py (environment variables LARC_SECOND, LARC_COORD, LARC_SCREEN, LARC_LHR, LARC_RACE
and LARC_COORD_FRAC) apply to the configuration larc. Each run records the name of the configuration and the counters of
the light-heavy steps and of the double-bridge kicks.

The runs of the paper:
  Sections 5.1 and 5.4 (LARC):          run_larc_benchmark.py 600 larc_main
  Section 5.6 (without the additions):  run_larc_benchmark.py 600 larc_without_additions --config larc_base
  Sections 5.7 and 5.8 (annealing):     scripts/experiments/run_sensitivity.sh
  Section 5.5 (ablation):               scripts/experiments/run_ablation.sh
  Section 5.8 (two-factor grid):        scripts/experiments/run_two_factor_grid.sh
  Section 5.2 (held-out instances):     run_larc_benchmark.py 600 larc_heldout --list heldout
                                        (after scripts/setup/prepare_heldout_instances.py)
Their results are in results/comparison_LARC_vs_CoCoPlus.xlsx (sheet LARC_run_details), results/convergence_LARC.xlsx,
results/ablation_LARC_without_additions.xlsx, results/sensitivity_*.xlsx, results/ablation_components.xlsx,
results/two_factor_grid.xlsx and results/heldout_LARC_vs_CoCoPlus.xlsx."""
import argparse
import time

from common import RUNS_DIR, run_jobs, seeds, select_instances

ap = argparse.ArgumentParser()
ap.add_argument("seconds", type=float)
ap.add_argument("name")
ap.add_argument("--seeds", default="1-10")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--list", default="", help="a list of the sheet experiment_lists of data/benchmark_instances.xlsx, or heldout")
ap.add_argument("--names", default="", help="instance names separated by commas")
ap.add_argument("--config", default="larc", choices=["larc", "larc_base"])
ap.add_argument("--param", action="append", default=[], help="parameter of the solver, key=value (number), repeatable")
A = ap.parse_args()


def run(job):
    name, seed = job
    t_start = time.monotonic()
    from larc import ablation, benchmark, trace
    from larc.evaluator import is_feasible, objective
    if A.config == "larc":
        from larc.larc import solve_larc as solve
    else:
        from larc.larc_base import solve_larc_base as solve
    inst = benchmark.load_instance(name)
    offset = time.monotonic() - t_start
    ablation.reset_stats()
    trace.start()
    params = {k: float(v) for k, v in (x.split("=") for x in A.param)}
    st, info = solve(inst, A.seconds - offset, seed=seed, **params)
    rows, phases = trace.stop()
    wall = time.monotonic() - t_start
    rec = dict(configuration=ablation.VARIANT if A.config == "larc" else "larc_base", instance=name, seed=seed, objective=objective(inst, st.tour, st.picked), time_s=wall,
               feasible=bool(is_feasible(inst, st.tour, st.picked)), improvements_recorded=len(rows),
               best_value_in_trace=max((v for _, v in rows), default=float("nan")), instance_loading_s=offset,
               candidates=info.get("candidates"), candidates_searched=info.get("searched"),
               share_of_searches_converged=info.get("converged"), sampling_s=info.get("sample_s"))
    for ph, t in phases:
        ph = {"race_round_0_end": "race_round_1_end", "race_round_1_end": "race_round_2_end"}.get(ph, ph)
        rec[f"t_{ph}_s"] = t + offset
    rec.update(ablation.STATS)
    rec["_trace"] = [dict(instance=name, seed=seed, time_s=t + offset, best_value=v) for t, v in rows]
    return rec


if __name__ == "__main__":
    names = select_instances(A.list, A.names)
    jobs = [(n, s) for n in names for s in seeds(A.seeds)]
    run_jobs(run, jobs, RUNS_DIR / f"{A.name}.xlsx", keys=("instance", "seed"), workers=A.workers,
             label=lambda r: f"{r['instance']} seed {r['seed']}: {r['objective']:.1f} in {r['time_s']:.1f} s")
