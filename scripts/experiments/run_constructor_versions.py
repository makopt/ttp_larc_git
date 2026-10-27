"""The seven versions of the constructor of Table 3, alone, on the 60 instances: one cached chained Lin-Kernighan tour per
TSP base (data/tours) and no tour search. Version 7 is the constructor of LARC (Algorithm 2 of the paper).
Usage: python3 scripts/experiments/run_constructor_versions.py [--workers 8]
Output: runs/constructor_versions.xlsx (sheet runs: one row per instance, one column per version, as in
results/constructor_versions.xlsx). The mean ranks and gaps of Table 3 are computed by
scripts/analysis/reproduce_paper_numbers.py."""
import argparse

from common import RUNS_DIR, run_jobs, select_instances

# version: (options of construct.build, knapsack repack)
VERSIONS = {
    "version_1_ICA_ISA_chi0.4": (dict(mode="ica", chis=(0.4,), both_orientations=False, scoring="items"), False),
    "version_2_chi_grid_empty": (dict(mode="ica", both_orientations=False, scoring="profit"), False),
    "version_3_position_weighted": (dict(mode="pw"), False),
    "version_4_position_weighted_flip": (dict(mode="pw", polish=True), False),
    "version_5_load_aware": (dict(mode="la"), False),
    "version_6_load_aware_flip": (dict(mode="la", polish=True), False),
    "version_7_load_aware_flip_repack": (dict(mode="la", polish=True), True),
}

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=8)
A = ap.parse_args()


def run(job):
    name, = job
    from larc import benchmark, construct, tours
    from larc.evaluator import is_feasible
    inst = benchmark.load_instance(name)
    tour = tours.get_tour(name, inst, compute=False)
    row = dict(instance=name)
    for v, (kw, repack) in VERSIONS.items():
        construct.KP_REPACK = repack                  # the repack is called inside construct.build when this is True
        s = construct.build(inst, tour, **kw)
        assert is_feasible(inst, s.tour, s.picked)
        row[v] = s.objective
    return row


if __name__ == "__main__":
    jobs = [(n,) for n in select_instances()]
    run_jobs(run, jobs, RUNS_DIR / "constructor_versions.xlsx", keys=("instance",), workers=A.workers,
             label=lambda r: r["instance"])
