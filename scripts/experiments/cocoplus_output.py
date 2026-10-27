"""Reads the output of one CoCo+ run (sections Tour, Plan and Total gain) so that the solution can be evaluated again by
the evaluator of LARC.
Usage as a module: parse(path) -> (tour, picked, reported_value, restarts)
Usage as a script: python3 scripts/experiments/cocoplus_output.py <instance> <output file>"""
import gzip
import re
import sys
from pathlib import Path

import numpy as np


def _text(path) -> str:
    path = Path(path)
    return gzip.open(path, "rt").read() if path.suffix == ".gz" else path.read_text()


def parse(path):
    """Tour (cities from 0, starting at the depot), picked items (boolean mask), value printed by CoCo+ and number of
    restarts of the run."""
    txt = _text(path)
    tour = np.array(re.search(r"Tour:\s*\n(.*?)\n\s*\n", txt, re.S).group(1).split(), dtype=np.int64)
    plan = np.array(re.search(r"Plan:\s*\n(.*?)\n\s*\n", txt, re.S).group(1).split(), dtype=np.int64)
    m = re.search(r"Total gain:\s*\n\s*([-\d.eE+]+)", txt)
    # the final value is often missing from the output (operator<< of CoCo has no return); the best value of the
    # restart log is used instead
    best_log = re.findall(r"^\s+[-\d.e+]+\s+([-\d.e+]+)\s*$", txt, re.M)
    reported = float(m.group(1)) if m else (max(map(float, best_log)) if best_log else float("nan"))
    if tour.min() == 1:
        tour = tour - 1
    tour = np.roll(tour, -int(np.flatnonzero(tour == 0)[0]))
    ks = [int(k) for k in re.findall(r"(?m)^(\d+) : ", txt)]
    return tour, plan.astype(bool), reported, (max(ks) if ks else 0)


if __name__ == "__main__":
    from common import benchmark
    from larc.evaluator import objective
    name, path = sys.argv[1], sys.argv[2]
    inst = benchmark.load_instance(name)
    tour, picked, reported, restarts = parse(path)
    print(f"{name}: {restarts} restarts | value printed by CoCo+ {reported:.2f} | evaluator of LARC "
          f"{objective(inst, tour, picked):.2f} | {picked.sum()} of {len(picked)} items picked")
