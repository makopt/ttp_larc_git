"""Which instance characteristics explain the gap between LARC and CoCo+ (Section 5.9; response to Reviewer 1).

On the 60 categorized instances, the knapsack type, the number of items per city and the capacity level move together
(one combination per category). The 60 held-out instances break this link, so the analysis uses the 120 instances of
both tests: the relative gap of the mean values, 100 (LARC - CoCo+) / |CoCo+|, by knapsack type, items per city and
capacity level, Spearman correlations, and a linear regression of the gap on log10(cities), items per city, knapsack
type and capacity index, with heteroskedasticity-robust (HC3) standard errors.
Usage: python3 scripts/analysis/gap_by_instance_features.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from paper_check import check, summary

RES = Path(__file__).resolve().parents[2] / "results"


def gaps(book):
    r = pd.read_excel(RES / book, sheet_name="runs")
    m = r.groupby(["instance", "method"]).objective.mean().unstack()
    return (100 * (m["LARC"] - m["CoCo+"]) / m["CoCo+"].abs()).rename("gap")


d = pd.concat([gaps("comparison_LARC_vs_CoCoPlus.xlsx"), gaps("heldout_LARC_vs_CoCoPlus.xlsx")]).reset_index()
parts = d.instance.str.extract(r"^[A-Za-z]+(\d+)_n(\d+)_(.+)_(\d\d)$")
d["cities"], d["items"], d["kp_type"], d["capacity_index"] = (parts[0].astype(int), parts[1].astype(int), parts[2],
                                                              parts[3].astype(int))
d["items_per_city"] = np.round(d["items"] / (d.cities - 1)).astype(int)
d["log10_cities"] = np.log10(d.cities)
d["capacity_group"] = pd.cut(d.capacity_index, [0, 3, 7, 10], labels=["1-3", "4-7", "8-10"])

pd.set_option("display.width", 200)
print("== 120 instances (60 categorized + 60 held-out): mean and median gap (%)")
for col in ["kp_type", "items_per_city", "capacity_group"]:
    print(d.groupby(col, observed=True).gap.agg(["count", "mean", "median"]).round(2).to_string(), "\n")
for col in ["log10_cities", "items_per_city", "capacity_index"]:
    r = spearmanr(d[col], d.gap)
    print(f"Spearman correlation of the gap with {col}: {r.statistic:.2f} (p = {r.pvalue:.4f})")

X = pd.DataFrame({"intercept": 1.0, "log10_cities": d.log10_cities,
                  "five_items_per_city": (d.items_per_city == 5) * 1.0, "ten_items_per_city": (d.items_per_city == 10) * 1.0,
                  "uncorrelated": (d.kp_type == "uncorr") * 1.0,
                  "uncorrelated_similar_weights": (d.kp_type == "uncorr-similar-weights") * 1.0,
                  "capacity_index": d.capacity_index * 1.0})
A, y = X.values, d.gap.values
beta = np.linalg.lstsq(A, y, rcond=None)[0]
e = y - A @ beta
inv = np.linalg.inv(A.T @ A)
h = np.einsum("ij,jk,ik->i", A, inv, A)
cov = inv @ (A.T * (e ** 2 / (1 - h) ** 2)) @ A @ inv
se = np.sqrt(np.diag(cov))
print("\n== Linear regression of the gap (%), reference: one item per city, bounded strongly correlated items")
print(pd.DataFrame({"coefficient": beta, "robust_se": se, "p": 2 * norm.sf(np.abs(beta / se))},
                   index=X.columns).round(3).to_string())
print("R2 =", round(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), 2), " n =", len(y))

print("\n== Numbers printed in the paper (Section 5.9) and in the response to Reviewer 1 (Comment 5)")
c = pd.Series(beta, X.columns)
pv = pd.Series(2 * norm.sf(np.abs(beta / se)), X.columns)
check("per tenfold increase of the cities: effect, p", (c.log10_cities, pv.log10_cities), (-0.71, "<0.001"))
check("five items per city: effect, p", (c.five_items_per_city, pv.five_items_per_city), (-1.19, 0.001))
check("ten items per city: effect, p", (c.ten_items_per_city, pv.ten_items_per_city), (-1.16, "<0.001"))
check("cities, five and ten items per city: largest p", float(max(pv.log10_cities, pv.five_items_per_city,
      pv.ten_items_per_city)), "<0.002")
check("uncorrelated items: effect, p", (c.uncorrelated, pv.uncorrelated), (0.68, 0.04))
check("uncorrelated items with similar weights: effect, p", (c.uncorrelated_similar_weights, pv.uncorrelated_similar_weights), (0.69, 0.11))
check("capacity level: effect, p", (c.capacity_index, pv.capacity_index), (0.01, 0.76))
check("R2", 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), 0.32)
summary()
