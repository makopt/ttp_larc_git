"""Probe screening on every instance with more than 1,500 cities (Section 3.5.1 and Figure 9c).

From results/probe_screening_large.xlsx (40 candidates per instance, from run_search_probes.py probe_screening_large):
for each instance, the Spearman rank correlation between the value after the longer search (probe + 3 probes) and
  - the constructor value, and
  - the value after the probe;
and whether the best candidate after the longer search is among the 10 best candidates by constructor value and among
the 10 best by probe value. The overhead of probing in the runs of LARC is the time of the probes of a run,
candidates x probe length, divided by the budget, from the runs of LARC in results/ablation_components.xlsx.
Usage: python3 scripts/analysis/probe_screening_large.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from paper_check import check, summary

RES = Path(__file__).resolve().parents[2] / "results"
t = pd.read_excel(RES / "probe_screening_large.xlsx", sheet_name="traces")
rows = []
for inst, g in t.groupby("instance"):
    final = g.value_after_longer_search
    best = final.idxmax()
    top_c = g.constructor_value.nlargest(10).index
    top_p = g.value_after_probe.nlargest(10).index
    rows.append(dict(instance=inst, candidates=len(g), probe_s=g.probe_s.iloc[0],
                     rho_constructor=spearmanr(g.constructor_value, final).statistic,
                     rho_probe=spearmanr(g.value_after_probe, final).statistic,
                     best_in_top10_constructor=best in top_c, best_in_top10_probe=best in top_p))
d = pd.DataFrame(rows).sort_values("instance")
pd.set_option("display.width", 200)
print(d.round(2).to_string(index=False))
print("\ninstances:", len(d), " candidates:", int(d.candidates.sum()))
print("median Spearman: constructor %.2f, probe %.2f" % (d.rho_constructor.median(), d.rho_probe.median()))
print("probe correlation higher on %d of %d instances (Wilcoxon p = %.2g)"
      % ((d.rho_probe > d.rho_constructor).sum(), len(d), wilcoxon(d.rho_probe, d.rho_constructor).pvalue))
print("probe correlation at least 0.5 on %d instances; constructor on %d"
      % ((d.rho_probe >= 0.5).sum(), (d.rho_constructor >= 0.5).sum()))
print("best candidate among the 10 best: by probe %d, by constructor %d (of %d)"
      % (d.best_in_top10_probe.sum(), d.best_in_top10_constructor.sum(), len(d)))

runs = pd.read_excel(RES / "ablation_components.xlsx", sheet_name="runs")
runs = runs[runs.configuration == "LARC"].copy()
n = runs.instance.str.extract(r"^[A-Za-z]+(\d+)_")[0].astype(int)
large = runs[n > 1500].copy()
probe = np.minimum(10, np.maximum(0.5, 2.5e-4 * n[n > 1500]))
large["probe_share"] = large.candidates * probe / 600
print("\nprobe time in the runs of LARC on these instances: median %.1f%% of the budget, range %.1f%% to %.1f%%"
      % (100 * large.probe_share.median(), 100 * large.probe_share.min(), 100 * large.probe_share.max()))

print("\n== Numbers printed in the paper (Section 3.5.1 and Figure 9c)")
check("instances, candidates per instance", (len(d), int(d.candidates.max()), int(d.candidates.min())), (33, 40, 40))
check("median Spearman correlation: constructor, probe", (d.rho_constructor.median(), d.rho_probe.median()), ("0.70", "0.90"))
check("probe the better guide: instances, Wilcoxon p",
      (int((d.rho_probe > d.rho_constructor).sum()), wilcoxon(d.rho_probe, d.rho_constructor).pvalue), (32, "<0.001"))
check("best candidate among the 10 best: by probe, by constructor",
      (int(d.best_in_top10_probe.sum()), int(d.best_in_top10_constructor.sum())), (27, 22))
parts = d.instance.str.extract(r"^[A-Za-z]+(\d+)_n(\d+)_")
d["items_per_city"] = np.round(parts[1].astype(int) / (parts[0].astype(int) - 1)).astype(int)
med = d.groupby("items_per_city")[["rho_constructor", "rho_probe"]].median()
check("median correlations with 5 items per city: constructor, probe", tuple(med.loc[5]), (0.55, 0.84))
check("median correlations with 10 items per city: constructor, probe", tuple(med.loc[10]), (0.53, 0.88))
check("median correlations with 1 item per city: constructor, probe", tuple(med.loc[1]), (0.91, 0.96))
check("probe time: median, smallest and largest share of the budget (%)",
      (100 * large.probe_share.median(), 100 * large.probe_share.min(), 100 * large.probe_share.max()), (17, 10, 28))
summary()
