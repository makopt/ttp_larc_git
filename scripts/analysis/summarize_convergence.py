"""Convergence statistics of the runs of LARC (Section 5.3, Table 7 and Figure 16), computed from the traces: per run,
the time of the last improvement and the first times within 1% and 0.1% of the final value; per size group, the medians
and the quartiles of the gap to the final value at each second; and, at each second, the share of instances on which
the mean best value of LARC so far exceeds the mean final value of CoCo+.

Usage: python3 scripts/analysis/summarize_convergence.py [runs.xlsx]
Without an argument it reads the runs of the paper (results/convergence_LARC.xlsx and
results/comparison_LARC_vs_CoCoPlus.xlsx); with the workbook of new runs of scripts/experiments/run_larc_benchmark.py
it reads the traces and phase times from that workbook. Output: runs/convergence_summary.xlsx, with the sheets
run_phases, by_size_group, gap_to_final and share_above_CoCo+ of results/convergence_LARC.xlsx."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import RUNS_DIR, benchmark  # noqa: E402

RES = REPO / "results"
if len(sys.argv) > 1:
    runs = pd.read_excel(sys.argv[1], sheet_name="runs")
    traces = pd.read_excel(sys.argv[1], sheet_name="traces")
else:
    runs = pd.read_excel(RES / "comparison_LARC_vs_CoCoPlus.xlsx", sheet_name="LARC_run_details")
    traces = pd.read_excel(RES / "convergence_LARC.xlsx", sheet_name="traces")
comp = pd.read_excel(RES / "comparison_LARC_vs_CoCoPlus.xlsx", sheet_name="runs")
coco = comp[comp.method == "CoCo+"].groupby("instance").objective.mean()
man = pd.DataFrame(benchmark.instance_list()).set_index("instance")
GROUPS = ["n<=300", "300<n<=5000", "n>5000"]
size = lambda n: GROUPS[0] if n <= 300 else (GROUPS[1] if n <= 5000 else GROUPS[2])
grid = np.arange(1, 601)
traces = {k: g.sort_values("time_s") for k, g in traces.groupby(["instance", "seed"])}

rows, G = [], []
for _, r in runs.iterrows():
    t = traces[(r.instance, r.seed)]
    fin, tt, vv = r.objective, t.time_s.values, t.best_value.values
    first = lambda thr: float(tt[np.argmax(vv >= thr)]) if (vv >= thr).any() else np.nan
    rows.append(dict(instance=r.instance, seed=r.seed, size_group=size(man.loc[r.instance, "n_cities"]),
                     category=man.loc[r.instance, "category"], final_value=fin, t_last_improvement_s=float(tt[-1]),
                     t99_s=first(fin - 0.01 * abs(fin)), **{"t99.9_s": first(fin - 0.001 * abs(fin))},
                     t_sampling_end_s=r.t_sampling_end_s, t_race_end_s=r.get("t_race_round_2_end_s", np.nan),
                     t_recombination_end_s=r.t_recombination_end_s, t_final_search_end_s=r.t_final_search_end_s,
                     time_s=r.time_s))
    k = np.searchsorted(tt, grid, side="right") - 1
    best = np.where(k >= 0, vv[np.clip(k, 0, None)], np.nan)
    G.append(pd.DataFrame(dict(instance=r.instance, seed=r.seed, t=grid, best=best, gap=100 * (fin - best) / abs(fin))))
ph = pd.DataFrame(rows)
G = pd.concat(G)
M = G.groupby(["instance", "t"]).best.mean().unstack()               # mean best value so far over the seeds
above = M.sub(coco.loc[M.index], axis=0) > 0
grp = pd.Series([size(man.loc[i, "n_cities"]) for i in above.index], above.index)
share = pd.DataFrame({f"share_pct_{g}": above[grp == g].mean() * 100 for g in GROUPS})
share["share_pct_all"] = above.mean() * 100
share = share.rename_axis("time_s").reset_index()
q = G.groupby([G.instance.map(lambda i: size(man.loc[i, "n_cities"])), "t"]).gap.quantile([0.25, 0.5, 0.75]).unstack()
q = q.reset_index()
q.columns = ["size_group", "time_s", "gap_to_final_pct_q25", "gap_to_final_pct_median", "gap_to_final_pct_q75"]
cols = ["t_last_improvement_s", "t99_s", "t99.9_s", "t_sampling_end_s", "t_race_end_s", "t_recombination_end_s",
        "t_final_search_end_s"]
tab = ph.groupby("size_group")[cols].median()
tab["runs"] = ph.groupby("size_group").size()
tab["late_runs_pct"] = ph.groupby("size_group").t_last_improvement_s.apply(lambda x: (x > 550).mean() * 100)
out = RUNS_DIR / "convergence_summary.xlsx"
out.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(out, engine="openpyxl") as w:
    ph.to_excel(w, sheet_name="run_phases", index=False)
    tab.reset_index().to_excel(w, sheet_name="by_size_group", index=False)
    q.to_excel(w, sheet_name="gap_to_final", index=False)
    share.to_excel(w, sheet_name="share_above_CoCo+", index=False)
print(tab.round(1).to_string())
print("all runs: medians of the last improvement, t99, t99.9:", ph.t_last_improvement_s.median().round(1),
      ph.t99_s.median().round(1), ph["t99.9_s"].median().round(1))
print("->", out)
