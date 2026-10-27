"""Runs LARC once on one instance of the benchmark and saves the solution.
Usage: python3 scripts/experiments/run_larc.py <instance> [--seconds 600] [--seed 1]
Example: python3 scripts/experiments/run_larc.py a280_n279_bounded-strongly-corr_01 --seconds 60
The solution is written to runs/solutions/<instance>_seed<seed>.txt in the format of the TTP competitions: first line
the tour (city numbers from 1, starting at city 1), second line the picked items (item numbers from 1, as in the .ttp
file). The objective value is computed again from the solution before it is printed."""
import argparse
import time

import numpy as np

from common import RUNS_DIR, benchmark

ap = argparse.ArgumentParser()
ap.add_argument("instance")
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--seed", type=int, default=1)
A = ap.parse_args()

from larc.evaluator import is_feasible, objective  # noqa: E402
from larc.larc import solve_larc  # noqa: E402

t0 = time.monotonic()
inst = benchmark.load_instance(A.instance)
st, info = solve_larc(inst, A.seconds - (time.monotonic() - t0), seed=A.seed)
wall = time.monotonic() - t0
z = objective(inst, st.tour, st.picked)
assert is_feasible(inst, st.tour, st.picked)
out = RUNS_DIR / "solutions" / f"{A.instance}_seed{A.seed}.txt"
out.parent.mkdir(parents=True, exist_ok=True)
tour = [int(c) + 1 for c in st.tour]
items = [int(k) + 1 for k in np.flatnonzero(st.picked)]
out.write_text("[" + ",".join(map(str, tour)) + "]\n[" + ",".join(map(str, items)) + "]\n")
print(f"{A.instance} seed {A.seed}: objective {z:.2f} in {wall:.1f} s, {info.get('candidates')} candidates -> {out}")
