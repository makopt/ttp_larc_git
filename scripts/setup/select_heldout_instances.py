"""Repeats the selection of the 60 held-out instances (Section 4.1 of the paper) from a local copy of the full
benchmark of 9,720 instances, and checks that it gives the list used in the paper.

Rule, fixed before any run: the 60 categorized instances are taken in the order of the sheet "instances" of
data/benchmark_instances.xlsx. For each of them, the candidates are the other files of the same TSP base with the same
number of items (another knapsack type and/or capacity index), excluding the 60 categorized instances, the instances
already drawn and the files whose renting ratio is 0.00. One candidate is drawn uniformly at random, the candidates being
sorted by file name and a single numpy generator (default_rng, seed 20261120) being used for the whole list.

Usage: python3 scripts/setup/select_heldout_instances.py <folder of the full benchmark, with one <base>-ttp folder per TSP>
The full benchmark is distributed at https://cs.adelaide.edu.au/~optlog/CEC2014COMP_InstancesNew/ (Polyakovskiy et al., 2014).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(bench: Path):
    cat = pd.read_excel(ROOT / "data/benchmark_instances.xlsx", sheet_name="instances")
    rng = np.random.default_rng(20261120)
    used, rows = set(cat.instance), []
    for _, r in cat.iterrows():
        prefix = f"{r.tsp_base}_n{r.n_items}_"
        cands = []
        for f in sorted((bench / r.ttp_folder).glob(prefix + "*.ttp")):
            if f.stem in used:
                continue
            rent = float(re.search(r"RENTING RATIO:\s*([0-9.]+)", f.read_text()[:2000]).group(1))
            if rent == 0.0:
                continue
            cands.append(f.stem)
        pick = cands[rng.integers(len(cands))]
        used.add(pick)
        rows.append((pick, r.instance))
    drawn = pd.DataFrame(rows, columns=["instance", "paired_benchmark_instance"])
    ref = pd.read_excel(ROOT / "results/heldout_LARC_vs_CoCoPlus.xlsx", sheet_name="instances")
    same = drawn.set_index("paired_benchmark_instance").instance.equals(
        ref.set_index("paired_benchmark_instance").instance.loc[drawn.paired_benchmark_instance])
    print(drawn.to_string(index=False))
    print("\nidentical to the list used in the paper:", same)
    return 0 if same else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1])))
