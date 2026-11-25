"""Recomputes the ablation of the main components of LARC (Section 5.5 and Table 11) from
results/ablation_components.xlsx, and prints each number next to the value printed in the paper (status of
scripts/analysis/paper_check.py).

Each variant is compared with LARC run in the same batch, on the same instances and seeds. Per instance: relative gap of
the mean values, 100 (variant - LARC) / LARC, two-sided Mann-Whitney U test, Holm correction over the 60 instances.
Over the instances: mean gap with a percentile bootstrap 95% confidence interval (10,000 resamples of the instances,
a new numpy generator with seed 20261120 for every interval, so that each interval does not depend on the others), Wilcoxon signed-rank test on the 60 gaps, and instances significantly worse or better.
Usage: python3 scripts/analysis/ablation_components.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import benchmark  # noqa: E402
from paper_check import check, summary  # noqa: E402

runs = pd.read_excel(REPO / "results/ablation_components.xlsx", sheet_name="runs")
meta = pd.read_excel(benchmark.INSTANCE_LIST, sheet_name="instances").set_index("instance")
VARIANTS = {"no_second_order": "Second-order term", "no_coordinated_2opt": "Coordinated 2-opt",
            "no_first_pass_screen": "First-pass screen", "no_light_heavy": "Light-heavy re-sequencing",
            "no_racing": "Racing"}
PAPER = {  # Table 11: gap, CI low, CI high, p, worse, better, A, B, C
    "no_second_order": (+0.04, -0.02, +0.13, 0.93, 0, 1, +0.17, -0.05, "0.00"),
    "no_coordinated_2opt": (-0.64, -0.94, -0.37, "<0.001", 11, 0, -0.02, -1.68, -0.23),
    "no_first_pass_screen": (-0.12, -0.19, -0.04, "<0.001", 0, 0, +0.03, -0.26, -0.13),
    "no_light_heavy": (-0.03, -0.08, +0.02, "0.40", 0, 0, -0.03, "0.00", -0.04),
    "no_racing": (-0.11, -0.17, -0.06, "<0.001", 0, 0, -0.08, -0.14, -0.11)}


def holm(p):
    p = np.asarray(p, float); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def mwu(x, y):
    z = np.concatenate([x, y])
    return 1.0 if np.all(z == z[0]) else mannwhitneyu(x, y, alternative="two-sided").pvalue


def per_instance(var, base):
    rows = []
    for inst in base.instance.unique():
        a = var[var.instance == inst].set_index("seed").objective
        b = base[base.instance == inst].set_index("seed").objective
        s = sorted(set(a.index) & set(b.index))
        a, b = a.loc[s].values, b.loc[s].values
        rows.append(dict(instance=inst, category=meta.loc[inst, "category"], n_cities=meta.loc[inst, "n_cities"],
                         gap=100 * (a.mean() - b.mean()) / abs(b.mean()), p=mwu(a, b)))
    d = pd.DataFrame(rows)
    d["p_holm"] = holm(d.p)
    return d


def boot_ci(v):
    m = np.random.default_rng(20261120).choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    return round(np.percentile(m, 2.5), 2), round(np.percentile(m, 97.5), 2)


def fmt_p(p):
    return "<0.001" if p < 0.001 else p


base = runs[runs.configuration == "LARC"]
main = pd.read_excel(REPO / "results/comparison_LARC_vs_CoCoPlus.xlsx", sheet_name="runs")
main = main[main.method == "LARC"]
g0 = []
for inst in base.instance.unique():
    a0, b0 = base[base.instance == inst].objective.mean(), main[main.instance == inst].objective.mean()
    g0.append(100 * (a0 - b0) / abs(b0))
check("LARC of this batch against the runs of the main comparison: mean gap (%), Wilcoxon p",
      (float(np.mean(g0)), wilcoxon(g0).pvalue), (0.02, 0.06))
print("\n== Table 11: ablation of the main components (gap %, 95% CI, Wilcoxon p, worse, better, gap in A, B, C)")
per = {}
for v, label in VARIANTS.items():
    d = per_instance(runs[runs.configuration == v], base)
    lo, hi = boot_ci(d.gap.values)
    sig = d.p_holm < 0.05
    cat = d.groupby("category").gap.mean()
    got = (d.gap.mean(), lo, hi, fmt_p(wilcoxon(d.gap).pvalue), int((sig & (d.gap < 0)).sum()),
           int((sig & (d.gap > 0)).sum()), cat["A"], cat["B"], cat["C"])
    check(f"Table 11, {label}", got, PAPER[v])
    per[v] = d

    if v == "no_light_heavy":
        ds = d[d.n_cities <= 300]
        lhr_small = (ds.gap.mean(), *boot_ci(ds.gap.values), wilcoxon(ds.gap).pvalue)
print("\n== Numbers of the text of Section 5.5")
check("no light-heavy, instances with n<=300: gap, CI, Wilcoxon p", lhr_small, (-0.17, -0.33, -0.02, 0.06))
dc = per["no_coordinated_2opt"].set_index("instance")
check("no coordinated 2-opt: significantly worse instances in category B; gap on a280 with five items per city (%)",
      (int(((dc.p_holm < 0.05) & (dc.gap < 0) & (dc.category == "B")).sum()), dc.loc["a280_n1395_uncorr-similar-weights_05", "gap"]),
      (9, -5.4))
dsc = per["no_first_pass_screen"]
check("no first-pass screen: mean gap on the instances with more than 5,000 cities (%)",
      dsc[dsc.n_cities > 5000].gap.mean(), -0.25)
d2 = per["no_second_order"]
check("no second-order term: the instance significantly better",
      d2[(d2.p_holm < 0.05) & (d2.gap > 0)].instance.tolist(), ["kroA100_n99_bounded-strongly-corr_01"])

print("\n== Light-heavy re-sequencing and time shares in the runs of LARC (Section 5.5)")
small = base.instance.map(meta.n_cities) <= 300
check("attempts, successes, success rate (%), share of run time (%)",
      (int(base.lhr_tries.sum()), int(base.lhr_wins.sum()), 100 * base.lhr_wins.sum() / base.lhr_tries.sum(),
       100 * base.lhr_time.sum() / base.time_s.sum()), (8546, 237, 2.8, 28))
check("n<=300: share of runs with a success (%), gain / final value (%)",
      (100 * (base[small].lhr_wins > 0).mean(), (100 * base[small].lhr_gain / base[small].objective).mean()), (61, 1.1))
check("n>300: attempts, successes", (int(base[~small].lhr_tries.sum()), int(base[~small].lhr_wins.sum())), (3518, 1))
check("sampling share of the run time (%)", 100 * (base.t_sampling_end / base.time_s).mean(), 46)
race = (base[["t_race_round_0_end", "t_race_round_1_end"]].max(axis=1) - base.t_sampling_end).clip(lower=0)
check("race share of the run time (%)", 100 * (race / base.time_s).mean(), 31)
check("double-bridge kicks share of the run time (%)", 100 * (base.kick_time / base.time_s).mean(), 7)
summary()
