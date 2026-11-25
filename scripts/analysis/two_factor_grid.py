"""Two-factor grid (Section 5.8): sampling end of the large instances x damping of the coordinated 2-opt.

From results/two_factor_grid.xlsx (540 runs, 12 instances with more than 1,500 cities, seeds 1 to 5). For each setting,
the relative gap of the mean value on each instance to the setting of the paper (0.45, 0.25), in percent, its mean over
the 12 instances with a percentile bootstrap 95% confidence interval (10,000 resamples, seed 20261120), and the Wilcoxon
signed-rank test on the 12 gaps. Interaction: two-way analysis of variance of these gaps with the instance as a block
(factors sampling end and damping, 3 levels each), F test of the interaction against the instance x setting residual.
Usage: python3 scripts/analysis/two_factor_grid.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as fdist, wilcoxon

from paper_check import check, summary

RES = Path(__file__).resolve().parents[2] / "results"
runs = pd.read_excel(RES / "two_factor_grid.xlsx", sheet_name="runs")
m = runs.groupby(["sampling_end_share", "damping_f", "instance"]).objective.mean()
base = m.loc[(0.45, 0.25)]
gap = (100 * (m - base.reindex(m.index, level="instance")) / base.abs().reindex(m.index, level="instance")).rename("gap")
gap = gap.reset_index()


def boot_ci(v):
    b = np.random.default_rng(20261120).choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
    return round(np.percentile(b, 2.5), 2), round(np.percentile(b, 97.5), 2)


rows = []
for (s, f), g in gap.groupby(["sampling_end_share", "damping_f"]):
    v = g.gap.values
    p = wilcoxon(v).pvalue if (s, f) != (0.45, 0.25) and np.any(v != 0) else np.nan
    rows.append(dict(sampling_end=s, damping=f, mean_gap=round(v.mean(), 3), ci95=boot_ci(v) if (s, f) != (0.45, 0.25)
                     else (0.0, 0.0), higher=int((v > 0).sum()), lower=int((v < 0).sum()), wilcoxon_p=p))
pd.set_option("display.width", 200)
print("== Gap of each setting to the setting of the paper (0.45, 0.25), percent, 12 instances")
print(pd.DataFrame(rows).to_string(index=False))
print("\nmean gap by sampling end:", gap.groupby("sampling_end_share").gap.mean().round(3).to_dict())
print("mean gap by damping:     ", gap.groupby("damping_f").gap.mean().round(3).to_dict())

# two-way ANOVA with instance as a block, on the per-instance gaps (12 instances x 9 settings)
y = gap.pivot_table(index="instance", columns=["sampling_end_share", "damping_f"], values="gap")
Y = y.values                                    # 12 x 9, columns ordered (s, f)
ni, ns, nf = Y.shape[0], 3, 3
Z = Y.reshape(ni, ns, nf)
grand = Z.mean()
inst = Z.mean(axis=(1, 2))
sm = Z.mean(axis=(0, 2))
fm = Z.mean(axis=(0, 1))
cell = Z.mean(axis=0)
ss_s = ni * nf * ((sm - grand) ** 2).sum()
ss_f = ni * ns * ((fm - grand) ** 2).sum()
ss_sf = ni * ((cell - sm[:, None] - fm[None, :] + grand) ** 2).sum()
ss_tot = ((Z - grand) ** 2).sum()
ss_inst = ns * nf * ((inst - grand) ** 2).sum()
ss_res = ss_tot - ss_inst - ss_s - ss_f - ss_sf
df_res = (ni - 1) * (ns * nf - 1)
ms_res = ss_res / df_res
print("\n== Two-way analysis of variance with the instance as a block (residual df = %d)" % df_res)
for name, ss, df in [("sampling end", ss_s, 2), ("damping", ss_f, 2), ("interaction", ss_sf, 4)]:
    F = (ss / df) / ms_res
    print(f"{name:13s} F({df},{df_res}) = {F:.2f}   p = {fdist.sf(F, df, df_res):.3f}")

print("\n== Numbers printed in the paper (Section 5.8) and in the response to Reviewer 1 (Comment 6)")
pv = {name: fdist.sf((ss / df) / ms_res, df, df_res) for name, ss, df in
      [("sampling end", ss_s, 2), ("damping", ss_f, 2), ("interaction", ss_sf, 4)]}
check("interaction: F(4, 88), p", ((ss_sf / 4) / ms_res, pv["interaction"]), (0.16, 0.96))
check("damping: p; sampling end: p", (pv["damping"], pv["sampling end"]), (0.003, 0.61))
R = pd.DataFrame(rows).set_index(["sampling_end", "damping"])
check("no setting significantly better than that of LARC: smallest Wilcoxon p", R.wilcoxon_p.min(), ">=0.09")
check("best damping: mean gap by damping highest at 0.25", bool(gap.groupby("damping_f").gap.mean().idxmax() == 0.25), True)
LETTER = {(0.30, 0.10): -0.18, (0.30, 0.25): -0.03, (0.30, 0.50): -0.42, (0.45, 0.10): -0.03, (0.45, 0.50): -0.38,
          (0.60, 0.10): -0.11, (0.60, 0.25): 0.04, (0.60, 0.50): -0.27}
check("response to Reviewer 1: mean gaps of the eight other settings",
      tuple(R.loc[k, "mean_gap"] for k in LETTER), tuple(LETTER.values()))
summary()
