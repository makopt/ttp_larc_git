"""Held-out test (Section 5.2 and Table 8): LARC against CoCo+ on 60 instances that were never used to set the constants of
LARC (results/heldout_LARC_vs_CoCoPlus.xlsx), with the same statistics as the main comparison (Section 5.1), and the
comparison with the 60 categorized instances (results/comparison_LARC_vs_CoCoPlus.xlsx).

Per instance: relative gap of the means 100 (LARC - CoCo+) / |CoCo+|, two-sided Mann-Whitney U test, Holm correction
over the 60 instances, Vargha-Delaney A12. Over the instances: Wilcoxon signed-rank test on the 60 gaps and a percentile
bootstrap 95% CI of the mean gap (10,000 resamples, numpy seed 20261120). Groups: items per city, knapsack type, size.
Usage: python3 scripts/analysis/heldout_comparison.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

from paper_check import check, summary as check_summary

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results"


def holm(p):
    p = np.asarray(p, float); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] > y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))


def mwu(x, y):
    z = np.concatenate([x, y])
    return 1.0 if np.all(z == z[0]) else mannwhitneyu(x, y, alternative="two-sided").pvalue


def boot_ci(v):
    m = np.random.default_rng(20261120).choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    return round(np.percentile(m, 2.5), 2), round(np.percentile(m, 97.5), 2)


def compare(runs, labels):
    rows = []
    for inst, g in runs.groupby("instance"):
        x = g[g.method == labels[0]].sort_values("seed").objective.values
        y = g[g.method == labels[1]].sort_values("seed").objective.values
        rows.append(dict(instance=inst, mean_larc=x.mean(), mean_coco=y.mean(), best_larc=x.max(), best_coco=y.max(),
                         cv_larc=100 * x.std(ddof=1) / x.mean(), cv_coco=100 * y.std(ddof=1) / y.mean(),
                         gap=100 * (x.mean() - y.mean()) / abs(y.mean()),
                         gap_best=100 * (x.max() - y.max()) / abs(y.max()), A12=a12(x, y), p=mwu(x, y)))
    d = pd.DataFrame(rows).set_index("instance")
    d["p_holm"] = holm(d.p)
    d["outcome"] = np.where(d.p_holm >= 0.05, "n.s.", np.where(d.gap > 0, "win", "loss"))
    return d


def summary(d, label):
    lo, hi = boot_ci(d.gap.values)
    s = dict(group=label, n=len(d), gap_mean=round(d.gap.mean(), 2), ci=(lo, hi), gap_median=round(d.gap.median(), 2),
             higher=int((d.gap > 0).sum()), wins=int((d.outcome == "win").sum()), losses=int((d.outcome == "loss").sum()),
             ns=int((d.outcome == "n.s.").sum()), A12=round(d.A12.mean(), 2),
             best_higher=int((d.gap_best > 0).sum()), gap_best=round(d.gap_best.mean(), 2))
    if len(d) >= 6 and np.any(d.gap != 0):
        s["wilcoxon_p"] = round(wilcoxon(d.gap).pvalue, 3)
    return s


inst = pd.read_excel(RES / "heldout_LARC_vs_CoCoPlus.xlsx", sheet_name="instances").set_index("instance")
runs = pd.read_excel(RES / "heldout_LARC_vs_CoCoPlus.xlsx", sheet_name="runs")
d = compare(runs, ("LARC", "CoCo+")).join(inst)
d["size"] = pd.cut(d.n_cities, [0, 300, 5000, 1e6], labels=["n<=300", "300<n<=5000", "n>5000"]).astype(str)

print("== Held-out instances: LARC against CoCo+ (1,200 runs)")
print("all runs feasible:", bool(runs.feasible.all()), " max time LARC / CoCo+ (s):",
      round(runs[runs.method == "LARC"].time_s.max(), 1), round(runs[runs.method == "CoCo+"].time_s.max(), 1))
print("max |re-evaluation difference| of CoCo+ solutions:", round(runs.reevaluation_difference.abs().max(), 2))
print(summary(d, "all 60"))
print("A12 effect: large / medium / small / negligible",
      [int(((d.A12 - 0.5).abs() >= t1).sum() - ((d.A12 - 0.5).abs() >= t2).sum()) for t1, t2 in
       [(0.21, 9), (0.14, 0.21), (0.06, 0.14), (0, 0.06)]])
print("mean CV of the ten runs (%): LARC, CoCo+", round(d.cv_larc.mean(), 2), round(d.cv_coco.mean(), 2))
for col in ["item_multiplier", "kp_type", "size"]:
    for g, dg in d.groupby(col):
        print(" ", col, "=", g, summary(dg, g))

print("\n== The same 60 pairs on the categorized benchmark (Section 5.1)")
main = pd.read_excel(RES / "comparison_LARC_vs_CoCoPlus.xlsx", sheet_name="runs")
dm = compare(main, ("LARC", "CoCo+"))
print(summary(dm, "categorized 60"))
pair = d.join(dm[["gap", "outcome"]], on="paired_benchmark_instance", rsuffix="_bench")
r = spearmanr(pair.gap, pair.gap_bench)
print("Spearman correlation of the gaps of paired instances:", round(r.statistic, 2), "p =", round(r.pvalue, 4))
print("same sign of the gap on", int((np.sign(pair.gap) == np.sign(pair.gap_bench)).sum()), "of 60 pairs")
print("\nper instance (gap %, A12, Holm outcome):")
print(d[["paired_benchmark_instance", "kp_type", "capacity_index", "gap", "A12", "outcome"]].round(3)
      .sort_values("gap").to_string())

print("\n== Numbers printed in the paper (Section 5.2 and Table 8)")
chk = check


def row(dd):
    s = summary(dd, "")
    return (s["gap_mean"], s["ci"], s["higher"], f'{s["wins"]}/{s["ns"]}/{s["losses"]}', s["A12"], s.get("wilcoxon_p"))


chk("Table 8, all: gap, CI, higher, W/ns/L, A12, p", row(d), (-0.21, (-0.59, 0.17), 24, "17/25/18", 0.47, 0.22))
chk("Table 8, at most 300 cities", row(d[d["size"] == "n<=300"]), (0.89, ("0.40", 1.42), 13, "11/3/1", 0.87, 0.005))
chk("Table 8, 301 to 5,000 cities", row(d[d["size"] == "300<n<=5000"]), ("-0.50", (-1.08, 0.03), 8, "5/10/9", 0.38, 0.12))
chk("Table 8, more than 5,000 cities", row(d[d["size"] == "n>5000"]), (-0.66, (-1.25, 0.02), 3, "1/12/8", 0.29, 0.002))
for kp, paper in [("bounded-strongly-corr", (-1.14, (-1.81, -0.56), 3, "0/9/9", 0.23, 0.001)),
                  ("uncorr", (-0.04, ("-0.40", "0.30"), 11, "9/11/5", 0.56, 0.94)),
                  ("uncorr-similar-weights", (0.53, ("-0.30", 1.39), 10, "8/5/4", "0.60", 0.31))]:
    chk("Table 8, " + kp, row(d[d.kp_type == kp]), paper)
chk("Table 8, categorized instances (main comparison)", row(dm), (0.11, (-0.18, 0.42), 31, "15/33/12", 0.52, 0.81))
chk("held-out median gap (%)", round(d.gap.median(), 2), -0.13)
bsc = d[(d.kp_type == "bounded-strongly-corr") & (d.item_multiplier != "1x")]
chk("bounded strongly correlated, 5 or 10 items per city: count, mean gap (%)", (len(bsc), bsc.gap.mean()), (14, "-1.30"))
chk("Spearman correlation of paired gaps, p; same sign", (r.statistic, r.pvalue, int((np.sign(pair.gap) == np.sign(pair.gap_bench)).sum())),
    ("0.60", "<0.001", 49))
chk("held-out gap minus categorized gap (points)", d.gap.mean() - dm.gap.mean(), -0.31)
chk("best run of LARC higher: count, mean gap of the best runs (%)", (int((d.gap_best > 0).sum()), d.gap_best.mean()), (34, 0.21))
chk("mean CV (%): LARC, CoCo+", (d.cv_larc.mean(), d.cv_coco.mean()), (0.54, 0.32))
chk("all runs feasible; longest run of LARC, of CoCo+ (s)", (bool(runs.feasible.all()), runs[runs.method == "LARC"].time_s.max(),
    runs[runs.method == "CoCo+"].time_s.max()), (True, 608.3, 602.8))
check_summary()
