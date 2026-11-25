"""Runs CoCo+ on many instances and seeds: the public code of Namazi et al. (algorithm 4, CoCoP) with the seed patch of
patches/coco_larc.patch, built by scripts/setup/setup_external.sh. One single-threaded process per run. Every solution
is evaluated again with the evaluator of LARC.

Usage: python3 scripts/experiments/run_cocoplus_benchmark.py <seconds> <name> [--seeds 1-10] [--workers 8]
           [--list LIST | --list heldout | --names a,b]
Output: runs/<name>.xlsx (sheet runs) and the outputs of CoCo+, compressed, in runs/<name>_cocoplus_output/.
Started again, it skips the runs already done.
The runs of the paper:
  Section 5.1:            run_cocoplus_benchmark.py 600 cocoplus_main
  Section 5.2 (held-out): run_cocoplus_benchmark.py 600 cocoplus_heldout --list heldout
                          (after scripts/setup/prepare_heldout_instances.py)
Their results are in the sheet runs of results/comparison_LARC_vs_CoCoPlus.xlsx and of
results/heldout_LARC_vs_CoCoPlus.xlsx (rows with method CoCo+)."""
import argparse
import gzip
import subprocess
import time

from common import REPO, RUNS_DIR, run_jobs, seeds, select_instances

COCO_BIN = REPO / "external" / "CoCo-src" / "coco_seeded"   # built by scripts/setup/setup_external.sh
COCO_CWD = REPO                                               # CoCo+ reads data/<base>-ttp/<instance>.ttp from here
HELDOUT_CWD = RUNS_DIR / "cocoplus_heldout"                   # for the held-out instances: data -> <repo>/data/heldout

ap = argparse.ArgumentParser()
ap.add_argument("seconds", type=int)
ap.add_argument("name")
ap.add_argument("--seeds", default="1-10")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--list", default="")
ap.add_argument("--names", default="")
A = ap.parse_args()


def run(job):
    name, seed = job
    from larc import benchmark
    from larc.evaluator import is_feasible, objective
    from cocoplus_output import parse
    inst = benchmark.load_instance(name)
    out_dir = RUNS_DIR / f"{A.name}_cocoplus_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    cwd = HELDOUT_CWD if A.list == "heldout" else COCO_CWD
    proc = subprocess.run([str(COCO_BIN), name, str(A.seconds), "4", str(seed)], cwd=cwd,
                          capture_output=True, text=True, timeout=A.seconds * 3)
    wall = time.monotonic() - t0
    raw = out_dir / f"{name}_seed{seed}.out.gz"
    with gzip.open(raw, "wt") as f:
        f.write(proc.stdout)
    try:
        tour, picked, reported, restarts = parse(raw)
        obj, feas = objective(inst, tour, picked), bool(is_feasible(inst, tour, picked))
    except Exception as e:                                   # no readable solution: recorded, never dropped
        obj, feas, reported, restarts = float("nan"), False, float("nan"), 0
        print(f"no solution read for {name} seed {seed}: {e}", flush=True)
    return {"instance": name, "seed": seed, "method": "CoCo+", "objective": obj, "time_s": wall, "feasible": feas,
            "CoCo+_reported_value": reported, "reevaluation_difference": obj - reported, "CoCo+_restarts": restarts}


if __name__ == "__main__":
    if not COCO_BIN.exists():
        raise SystemExit(f"{COCO_BIN} not found: run scripts/setup/setup_external.sh first")
    if A.list == "heldout":
        HELDOUT_CWD.mkdir(parents=True, exist_ok=True)
        if not (HELDOUT_CWD / "data").exists():
            (HELDOUT_CWD / "data").symlink_to(REPO / "data" / "heldout", target_is_directory=True)
    names = select_instances(A.list, A.names)
    jobs = [(n, s) for n in names for s in seeds(A.seeds)]
    run_jobs(run, jobs, RUNS_DIR / f"{A.name}.xlsx", keys=("instance", "seed"), workers=A.workers, maxtasks=20,
             label=lambda r: f"CoCo+ {r['instance']} seed {r['seed']}: {r['objective']:.1f} in {r['time_s']:.1f} s")
