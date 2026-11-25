"""Uncertainty of the same-machine comparison of LARC and CoCo+ (Section 5.1 and Table 7).

From results/comparison_LARC_vs_CoCoPlus.xlsx: the relative gap of the mean values on each instance,
100 (LARC - CoCo+) / |CoCo+|, then for each category and over the 60 instances the mean gap with a percentile bootstrap
95% confidence interval (10,000 resamples of the instances, a new numpy generator with seed 20261120 for every interval),
and the numbers of instances with a gap above 0.1% and above 1% in favour of each method.
Usage: python3 scripts/analysis/comparison_uncertainty.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from paper_check import check, summary

REPO = Path(__file__).resolve().parents[2]
runs = pd.read_excel(REPO / "results/comparison_LARC_vs_CoCoPlus.xlsx", sheet_name="runs")
m = runs.groupby(["instance", "category", "method"]).objective.mean().unstack()
gap = (100 * (m["LARC"] - m["CoCo+"]) / m["CoCo+"].abs()).reset_index().rename(columns={0: "gap"})


def boot_ci(v):
    b = np.random.default_rng(20261120).choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    return round(np.percentile(b, 2.5), 2), round(np.percentile(b, 97.5), 2)


rows = []
for label, g in list(gap.groupby("category")) + [("all", gap)]:
    v = g.gap.values
    rows.append(dict(group=label, n=len(v), mean_gap=round(v.mean(), 2), ci95=boot_ci(v),
                     median_gap=round(float(np.median(v)), 2),
                     larc_ahead_over_0_1=int((v > 0.1).sum()), coco_ahead_over_0_1=int((v < -0.1).sum()),
                     larc_ahead_over_1=int((v > 1).sum()), coco_ahead_over_1=int((v < -1).sum())))
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).to_string(index=False))

print("\n== Numbers printed in the paper (Section 5.1 and Table 7)")
R = pd.DataFrame(rows).set_index("group")
for g, ci in (("A", (-0.02, "1.30")), ("B", (-0.83, 0.25)), ("C", (-0.16, 0.21)), ("all", (-0.18, 0.42))):
    check(f"Table 7, {g}: 95% confidence interval of the mean gap (%)", R.loc[g, "ci95"], ci)
check("all 60: LARC ahead by more than 0.1% / CoCo+ ahead by more than 0.1%",
      (R.loc["all", "larc_ahead_over_0_1"], R.loc["all", "coco_ahead_over_0_1"]), (25, 25))
check("all 60: LARC ahead by more than 1% / CoCo+ ahead by more than 1%",
      (R.loc["all", "larc_ahead_over_1"], R.loc["all", "coco_ahead_over_1"]), (9, 7))
summary()
