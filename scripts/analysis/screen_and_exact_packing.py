"""Numbers of the paper on the first-pass screen (Section 3.3.3) and on the exact packing of a fixed tour (Section 3.2.3),
from results/screen_measurement.xlsx (run_screen_measurement.py) and results/exact_packing.xlsx
(run_exact_packing.py on results/final_solutions_LARC_ablation_baseline.tar.xz).
Usage: python3 scripts/analysis/screen_and_exact_packing.py
"""
from pathlib import Path

import pandas as pd

from paper_check import check, summary

ROOT = Path(__file__).resolve().parents[2]
s = pd.read_excel(ROOT / "results/screen_measurement.xlsx", sheet_name="runs")
t = s.sum(numeric_only=True)
print("== First-pass screen (%d local searches of 120 s)" % len(s))
print("candidates with a positive first pass: %d, rejected by the screen: %d (%.1f%%)"
      % (t.candidates_with_positive_first_pass, t.rejected_by_screen,
         100 * t.rejected_by_screen / t.candidates_with_positive_first_pass))
print("rejected candidates that would give an improving coordinated move: %d" % t.rejected_but_improving_compound)
print("share of the work of the coordinated 2-opt spent after the first pass on rejected candidates: %.0f%%"
      % (100 * t.work_after_first_pass_on_rejected / t.work_total))

e = pd.read_excel(ROOT / "results/exact_packing.xlsx", sheet_name="runs")
inst = pd.read_excel(ROOT / "data/benchmark_instances.xlsx", sheet_name="instances").set_index("instance")
e["category"] = e.instance.map(inst.category)
chk = e.dropna(subset=["check_rebuilt_value"])
print("\n== Exact best packing on the final tours of LARC (%d runs on %d instances)" % (len(e), e.instance.nunique()))
print("dynamic program checked by rebuilding the optimal packing: %d of %d identical"
      % (((chk.check_rebuilt_value - chk.exact_optimum).abs() <= 1e-6 * chk.exact_optimum.abs()).sum(), len(chk)))
print("final packing of LARC optimal for its tour: %d runs; gap mean %.2f%%, max %.2f%%"
      % ((e.gap_larc_final_pct < 1e-6).sum(), e.gap_larc_final_pct.mean(), e.gap_larc_final_pct.max()))
print(e.groupby("category").agg(instances=("instance", "nunique"), runs=("seed", "size"),
                                larc_gap_mean=("gap_larc_final_pct", "mean"), larc_gap_max=("gap_larc_final_pct", "max"),
                                repack_gap_mean=("gap_repack_from_empty_pct", "mean"),
                                repack_gap_max=("gap_repack_from_empty_pct", "max")).round(3).to_string())

print("\n== Numbers printed in the paper (Sections 3.2.3 and 3.3.3)")
check("screen: rejected share of the candidates with a positive first pass (%)",
      100 * t.rejected_by_screen / t.candidates_with_positive_first_pass, 94.6)
check("screen: rejected candidates that would give an improving coordinated move", int(t.rejected_but_improving_compound), 0)
check("screen: share of the work after the first pass on rejected candidates (%)",
      100 * t.work_after_first_pass_on_rejected / t.work_total, 62)
check("exact packing: instances, runs", (e.instance.nunique(), len(e)), (41, 410))
check("final packing of LARC optimal for its tour: runs; gap mean, max (%)",
      (int((e.gap_larc_final_pct < 1e-6).sum()), e.gap_larc_final_pct.mean(), e.gap_larc_final_pct.max()), (261, 0.03, 0.79))
bc = e[e.category.isin(["B", "C"])]
check("knapsack repack alone, categories B and C: largest gap to the optimum (%)", bc.gap_repack_from_empty_pct.max(), "<=0.04")
a = e[e.category == "A"]
check("knapsack repack alone, category A: gap mean, max (%)",
      (a.gap_repack_from_empty_pct.mean(), a.gap_repack_from_empty_pct.max()), (1.65, 16.8))
summary()
