"""Recomputes the numbers of the paper (Table 3, Table 4, Section 3.5.1 and Section 5) from the workbooks of results/
and the published values of data/published_values.xlsx. Each line gives the recomputed value and the value printed in
the paper. Tables 6 to 9 are printed in full.
Usage: python3 scripts/analysis/reproduce_paper_numbers.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, mannwhitneyu, spearmanr, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import benchmark, stats  # noqa: E402

RES = REPO / "results"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
lines = []


def plain(v):
    """Python numbers instead of NumPy scalars, for printing."""
    if isinstance(v, tuple):
        return tuple(plain(x) for x in v)
    if isinstance(v, list):
        return [plain(x) for x in v]
    return v.item() if isinstance(v, np.generic) else v


def chk(label, value, paper):
    lines.append((label, plain(value), paper))


def section(title):
    lines.append((f"\n== {title}", "", None))


def sheet(book, name, **kw):
    return pd.read_excel(RES / book, sheet_name=name, **kw)


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for r, k in enumerate(np.argsort(p)):
        run = max(run, (len(p) - r) * p[k])
        adj[k] = min(1.0, run)
    return adj


def wil(g):
    g = g[np.abs(g) > 1e-12]
    return wilcoxon(g).pvalue if len(g) >= 2 else np.nan


def ranks(P):
    return P.rank(axis=1, ascending=False, method="average").mean()


def rdi(P):
    return pd.DataFrame(stats.rdi(P), index=P.index, columns=P.columns)


def fried(P):
    return friedmanchisquare(*[P[c].values for c in P.columns]).statistic


man = pd.DataFrame(benchmark.instance_list()).set_index("instance")
runs = sheet("comparison_LARC_vs_CoCoPlus.xlsx", "runs")
L = runs[runs.method == "LARC"].pivot(index="instance", columns="seed", values="objective")
C = runs[runs.method == "CoCo+"].pivot(index="instance", columns="seed", values="objective")
idx = L.index
cat, n = man.loc[idx, "category"], man.loc[idx, "n_cities"]
base = pd.Series(idx.str.split("_").str[0], idx)
pub = benchmark.published_values().loc[idx]
cocop = benchmark.published_cocop_means().loc[idx]
lm, cm, lx, cx, ln, cn = L.mean(1), C.mean(1), L.max(1), C.max(1), L.min(1), C.min(1)

# ------------------------------------------------------------------------------------------------ Table 3
section("Table 3: the constructor alone (mean rank among the nine published methods and the version, mean gap to the "
        "published CoCo+ value)")
cv = sheet("constructor_versions.xlsx", "values").set_index("instance").loc[idx]
paper3 = {1: (9.98, -153), 2: (9.95, -110), 3: (9.40, -57), 4: (8.62, -7.8), 5: (8.70, -47), 6: (7.98, -5.9),
          7: (7.84, -5.8)}
for col in [c for c in cv.columns if c.startswith("version_")]:
    k = int(col.split("_")[1])
    P = pub.assign(version=cv[col])
    gap = 100 * (cv[col] - pub["CoCo+"]) / pub["CoCo+"].abs()
    chk(f"version {k}: mean rank, mean gap (%)", (round(stats.mean_ranks(P)["version"], 2), round(gap.mean(), 1)), paper3[k])

# ------------------------------------------------------------------------------------------------ Table 4
section("Table 4: start tours on perturbed coordinates (distinct tours at sigma 0 and 0.1, best value per level)")
me = "method_experiments.xlsx"
nz = sheet(me, "start_tours_noise")
for inst, g in nz.groupby("instance", sort=False):
    d = g.groupby("noise_level_sigma")
    distinct = (int(d.tour_hash.nunique()[0.0]), int(d.tour_hash.nunique()[0.1]))
    chk(f"{inst.split('_')[0]}", (distinct, [round(v) for v in d.value_after_short_search.max()]), "see Table 4")

# ------------------------------------------------------------------------------------------------ Section 3.5.1
section("Section 3.5.1: probe screening on d2103 with five items per city (320 candidates)")
pr = sheet(me, "probe_screening")
d = pr[pr.instance.str.startswith("d2103")].reset_index(drop=True)
top = set(d.nlargest(10, "value_after_second_probe").index)
chk("candidates", len(d), 320)
chk("Spearman: constructor / 0.5 s probe against value after 2 s",
    (round(spearmanr(d.constructor_value, d.value_after_second_probe).statistic, 2),
     round(spearmanr(d.value_after_first_probe, d.value_after_second_probe).statistic, 2)), (0.28, 0.90))
chk("of the 10 best after 2 s, kept by the 32 best constructor values / 32 best probes",
    (len(top & set(d.nlargest(32, "constructor_value").index)), len(top & set(d.nlargest(32, "value_after_first_probe").index))),
    (0, 9))

# ------------------------------------------------------------------------------------------------ Section 5.1
section("Section 5.1: LARC against CoCo+ on the same machine")
gap = 100 * (lm - cm) / cm.abs()
gapb = 100 * (lx - cx) / cx.abs()
p = pd.Series([mannwhitneyu(L.loc[i], C.loc[i], alternative="two-sided").pvalue for i in idx], idx)
adj = pd.Series(holm(p), idx)
outc = pd.Series(np.where(adj < 0.05, np.where(lm > cm, "win", "loss"), "tie"), idx)
a12 = pd.Series([np.mean([(x > y) + 0.5 * (x == y) for x in L.loc[i] for y in C.loc[i]]) for i in idx], idx)
pubgap = 100 * (lx - pub["CoCo+"]) / pub["CoCo+"].abs()
wtl = lambda m: tuple(int(outc[m].eq(o).sum()) for o in ("win", "tie", "loss"))
chk("gap of the means: average, median (%)", (round(gap.mean(), 2), round(gap.median(), 2)), (0.11, 0.03))
chk("instances with a higher mean for LARC", int((lm > cm).sum()), 31)
chk("Wilcoxon p on the 60 gaps", round(wilcoxon(gap).pvalue, 2), 0.81)
chk("better / not significant / worse (Holm)", wtl(slice(None)), (15, 33, 12))
chk("raw p < 0.05 with LARC higher / lower", (int(((p < 0.05) & (lm > cm)).sum()), int(((p < 0.05) & (lm < cm)).sum())), (19, 19))
t6 = []
for lab, m in [("A (1 item/city)", cat == "A"), ("B (5 items/city)", cat == "B"), ("C (10 items/city)", cat == "C"),
               ("All 60", cat.notna())]:
    t6.append([lab, f"{gap[m].mean():+.2f}", f"{gapb[m].mean():+.2f}", f"{int((lm > cm)[m].sum())}/{int(m.sum())}",
               f"{int((lx > cx)[m].sum())}/{int(m.sum())}", " / ".join(map(str, wtl(m))), f"{a12[m].mean():.2f}",
               f"{pubgap[m].mean():+.2f}", f"{int((lx > pub['CoCo+'])[m].sum())}/{int(m.sum())}"])
t6 = pd.DataFrame(t6, columns=["category", "gap means", "gap best runs", "LARC higher mean", "LARC higher best run",
                               "better/n.s./worse", "A12", "gap to published", "LARC higher than published"])
chk("Table 6", "\n" + t6.to_string(index=False), "see Table 6")
chk("A12 > 0.5 on category A", int((a12[cat == "A"] > 0.5).sum()), 13)
e = (a12 - 0.5).abs()
chk("effect large / medium / small / negligible",
    (int((e >= 0.21).sum()), int(((e >= 0.14) & (e < 0.21)).sum()), int(((e >= 0.06) & (e < 0.14)).sum()), int((e < 0.06).sum())),
    (41, 8, 7, 4))
for lab, m, paper in (("n<=300", n <= 300, (10, 4, 1, 13, 1.03)), ("300<n<=5000", (n > 300) & (n <= 5000), (4, 13, 7, "not given", -0.08)),
                      ("n>5000", n > 5000, (1, 16, 4, "not given", -0.34))):
    chk(f"{lab}: better, n.s., worse, LARC higher, mean gap", (*wtl(m), int((lm > cm)[m].sum()), round(gap[m].mean(), 2)), paper)
cvl, cvc = 100 * L.std(1, ddof=1) / lm, 100 * C.std(1, ddof=1) / cm
chk("coefficient of variation LARC, CoCo+; LARC smaller", (round(cvl.mean(), 2), round(cvc.mean(), 2), int((L.std(1) < C.std(1)).sum())),
    (0.39, 0.31, 23))
green, orange = ln > cx, cn > lx
chk("ranges overlap / worst LARC above best CoCo+ / the reverse", (int((~green & ~orange).sum()), int(green.sum()), int(orange.sum())),
    (42, 12, 6))
tc, tl = runs[runs.method == "CoCo+"].time_s, runs[runs.method == "LARC"].time_s
chk("CoCo+ time: mean, max, runs above 600.5 s", (round(tc.mean(), 1), round(tc.max(), 1), int((tc > 600.5).sum())), (600.2, 604.4, 83))
chk("LARC time: mean, runs above 600.5 s, max", (round(tl.mean(), 1), int((tl > 600.5).sum()), round(tl.max(), 1)), (588.2, 23, 604.5))
chk("all runs of LARC feasible", bool(runs[runs.method == "LARC"].feasible.all()), True)

# ------------------------------------------------------------------------------------------------ Section 5.2
section("Section 5.2: LARC among the published methods")
m9, b9 = pub.assign(**{"CoCo+": cocop}), pub
chk("before LARC, Friedman chi2: means / best values", (round(fried(m9), 1), round(fried(b9), 1)), (182.9, 211.2))
chk("before LARC, mean ranks with means: CoCo, CoCo+", (round(ranks(m9)["CoCo"], 2), round(ranks(m9)["CoCo+"], 2)), (2.19, 2.56))
chk("before LARC, mean ranks with best values: CoCo+, CoCo", (round(ranks(b9)["CoCo+"], 2), round(ranks(b9)["CoCo"], 2)), (1.53, 2.80))
chk("before LARC, mean RDI with means: CoCo, CoCo+", (round(rdi(m9).mean()["CoCo"], 1), round(rdi(m9).mean()["CoCo+"], 1)), (15.4, 16.7))
chk("before LARC, mean RDI with best values: CoCo+, CoCo", (round(rdi(b9).mean()["CoCo+"], 1), round(rdi(b9).mean()["CoCo"], 1)), (11.9, 20.6))
m10, b10 = m9.assign(LARC=lm), b9.assign(LARC=lx)
rm, rb = ranks(m10), ranks(b10)
chk("with LARC, Friedman chi2: means / best values", (round(fried(m10), 1), round(fried(b10), 1)), (244.1, 288.8))
chk("with LARC, mean ranks with means: LARC, CoCo, CoCo+", (round(rm["LARC"], 2), round(rm["CoCo"], 2), round(rm["CoCo+"], 2)), (2.30, 2.86, 3.28))
chk("with LARC, mean ranks with best values: LARC, CoCo+, CoCo", (round(rb["LARC"], 2), round(rb["CoCo+"], 2), round(rb["CoCo"], 2)), (1.84, 2.24, 3.65))
lo_m = [ranks(m10.drop(columns=c))["LARC"] for c in m9.columns]
lo_b = [ranks(b10.drop(columns=c))["LARC"] for c in b9.columns]
chk("leaving out one method: range of the rank of LARC (means, best values)",
    (round(min(lo_m), 2), round(max(lo_m), 2), round(min(lo_b), 2), round(max(lo_b), 2)), (1.97, 2.30, 1.55, 1.84))
cd10 = stats.nemenyi_cd(10, 60)
chk("critical difference, 10 methods", round(cd10, 2), 1.75)
gp = 100 * (lm - cocop) / cocop.abs()
chk("mean of LARC against the published CoCoP mean: higher, average gap, Wilcoxon p",
    (int((lm > cocop).sum()), round(gp.mean(), 2), float(f"{wilcoxon(gp).pvalue:.1g}")), (43, 0.62, 0.0001))
Rm, Rb = rdi(m10), rdi(b10)
chk("with LARC, mean RDI with means: LARC, CoCo, CoCo+", (round(Rm.mean()["LARC"], 1), round(Rm.mean()["CoCo"], 1), round(Rm.mean()["CoCo+"], 1)), (16.1, 20.7, 21.9))
chk("with LARC, mean RDI with best values: LARC, CoCo+, CoCo", (round(Rb.mean()["LARC"], 1), round(Rb.mean()["CoCo+"], 1), round(Rb.mean()["CoCo"], 1)), (13.3, 18.0, 26.0))
gc = 100 * (cm - cocop) / cocop.abs()
chk("our CoCo+ runs against the published CoCoP means: average gap, higher", (round(gc.mean(), 2), int((cm > cocop).sum())), (0.51, 54))
chk("best run of LARC above the published CoCo+ value: count, average gap", (int((lx > pub["CoCo+"]).sum()), round(pubgap.mean(), 2)), (42, 0.50))
chk("best run of our CoCo+ at least the published CoCo+ value", int((cx >= pub["CoCo+"]).sum()), 44)
six = pub.drop(columns=["CS2SA", "MA2B", "CS2SA*"]).max(1)
gf = 100 * (lx - six) / six.abs()
chk("best run of LARC above all nine published values", int((lx > pub.max(1)).sum()), 34)
chk("above the six methods with the benchmark distances; equal; median and largest gain (%)",
    (int((lx > six + 1e-6).sum()), int(np.isclose(lx, six, atol=0.5).sum()), round(gf[lx > six + 1e-6].median(), 2),
     round(gf.max(), 2), base[gf.idxmax()]), (41, 2, 0.51, 4.24, "a280"))
cand = pd.concat([pub, lx.rename("LARC"), cx.rename("CoCo+ (our runs)")], axis=1)
bk = cand.max(1)
reach = np.isclose(lx, bk, atol=0.5)
alone = reach & ~np.isclose(cand.drop(columns=["LARC"]).max(1), bk, atol=0.5)
chk("best known value of the comparison reached by LARC / by LARC alone", (int(reach.sum()), int(alone.sum())), (29, 28))
bs = sheet("best_solutions.xlsx", "runs").set_index("instance")
pb = bs.best_published_value_6_methods
chk("saved solutions: count, above the six methods, above all nine published values",
    (len(bs), int((bs.objective > pb + 1e-6).sum()), int((bs.objective > pub.loc[bs.index].max(1)).sum())), (43, 36, 30))
chk("saved solutions: largest difference between the two evaluators, all feasible",
    (float(f"{(bs.objective - bs.objective_independent_evaluator).abs().max():.1g}"), bool(bs.feasible.all())), "agree")

# ------------------------------------------------------------------------------------------------ Section 5.3
section("Section 5.3 and Table 7: convergence")
cr = sheet("convergence_LARC.xlsx", "run_phases")
t7 = []
for g in ("n<=300", "300<n<=5000", "n>5000"):
    r = cr[cr.size_group == g]
    t7.append([g, len(r), round(r.t99_s.median()), round(r["t99.9_s"].median()), round(r.t_last_improvement_s.median()),
               round(r.t_sampling_end_s.median()), round(100 * (r.t_last_improvement_s > 550).mean())])
t7.append(["All", len(cr), round(cr.t99_s.median()), round(cr["t99.9_s"].median()), round(cr.t_last_improvement_s.median()), "", ""])
chk("Table 7", "\n" + pd.DataFrame(t7, columns=["instances", "runs", "t99", "t99.9", "last improvement", "sampling end",
                                                "late (%)"]).to_string(index=False), "see Table 7")
sh = sheet("convergence_LARC.xlsx", "share_above_CoCo+").set_index("time_s")
chk("share of instances where LARC leads, n<=300, at 30 / 450 / 600 s",
    tuple(round(sh.loc[t, "share_pct_n<=300"]) for t in (30, 450, 600)), (47, 80, 87))
chk("n>5000: first second with a share above 0; share at 600 s",
    (int(sh.index[sh["share_pct_n>5000"] > 0].min()), round(sh.loc[600, "share_pct_n>5000"], 1)), ("after 270", "a third"))
chk("all instances at 30 / 450 / 600 s", tuple(round(sh.loc[t, "share_pct_all"]) for t in (30, 450, 600)), (15, 33, 52))

# ------------------------------------------------------------------------------------------------ Section 5.4
section("Section 5.4: LARC without perturbed tours, probe screening, kicks and chunked final search")
ab = sheet("ablation_LARC_without_additions.xlsx", "runs").groupby("instance").objective.mean().loc[idx]
ga = 100 * (lm - ab) / ab.abs()
chk("gap of LARC to the configuration: mean, higher, lower, equal",
    (round(ga.mean(), 2), int((ga > 1e-9).sum()), int((ga < -1e-9).sum()), int((ga.abs() <= 1e-9).sum())), (0.16, 27, 24, 9))
chk("Wilcoxon p", round(wil(ga), 2), 0.10)
chk("configuration against CoCo+: mean gap", round((100 * (ab - cm) / cm.abs()).mean(), 2), -0.05)

# ------------------------------------------------------------------------------------------------ Section 5.5
section("Section 5.5 and Table 8: sampling shares (recomputed from the runs)")
det = sheet("comparison_LARC_vs_CoCoPlus.xlsx", "LARC_run_details")
sr = sheet("sensitivity_sampling_shares.xlsx", "runs")
t8, ps = [], []
for (share, val), g in sr.groupby(["share_changed", "tested_value"], sort=False):
    rows = []
    for inst, gi in g.groupby("instance"):
        b = det[det.instance == inst]
        rows.append(dict(gap=100 * (gi.objective.mean() - b.objective.mean()) / abs(b.objective.mean()),
                         ts=gi.t_sampling_end_s.median(), tsb=b.t_sampling_end_s.median(), nc=gi.candidates.median(),
                         ncb=b.candidates.median(), cg=100 * (gi.objective.mean() - cm[inst]) / abs(cm[inst]), cgb=gap[inst],
                         p=mannwhitneyu(gi.objective, b.objective, alternative="two-sided").pvalue))
    d = pd.DataFrame(rows)
    ps.append(wilcoxon(d.gap).pvalue)
    t8.append([share, val, len(d), f"{d.tsb.median():.0f}->{d.ts.median():.0f}", f"{d.ncb.median():g}->{d.nc.median():g}",
               f"{d.gap.mean():+.3f} [{d.gap.min():+.2f}, {d.gap.max():+.2f}]",
               f"{(d.gap > 0).sum()}/{(d.gap < 0).sum()}/{(d.gap == 0).sum()}", f"{ps[-1]:.3f}",
               f"{d.cgb.mean():+.2f}->{d.cg.mean():+.2f}", int((holm(d.p) < 0.05).sum())])
chk("Table 8 (last column: instances that differ after Holm)", "\n" + pd.DataFrame(
    t8, columns=["share", "value", "inst.", "sampling end", "candidates", "gap to LARC", "better/worse/equal", "p",
                 "gap to CoCo+", "Holm"]).to_string(index=False), "see Table 8")
chk("runs", len(sr), 1030)
chk("Holm over the five variants: adjusted p", [round(x, 3) for x in holm(ps)], "only 0.60T below 0.05 (0.011)")
l60 = sr[sr.tested_value == "0.60T"].pivot(index="instance", columns="seed", values="objective")
L2 = L.copy()
L2.loc[l60.index] = l60.loc[:, L.columns]
g2 = 100 * (L2.mean(1) - cm) / cm.abs()
a2 = holm([mannwhitneyu(L2.loc[i], C.loc[i], alternative="two-sided").pvalue for i in idx])
chk("LARC with the sampling end 0.60T on n>1500, against CoCo+: mean gap, better, worse, Wilcoxon p",
    (round(g2.mean(), 2), int(((a2 < 0.05) & (L2.mean(1) > cm)).sum()), int(((a2 < 0.05) & (L2.mean(1) < cm)).sum()),
     round(wilcoxon(g2).pvalue, 2)), (0.14, 15, 12, 0.55))

# ------------------------------------------------------------------------------------------------ Section 5.6
section("Section 5.6 and Table 9: other constants (recomputed from the runs)")
oc = "sensitivity_other_constants.xlsx"
A = sheet(oc, "constructor").set_index("instance")
tb = A[["LARC_values_s", "LARC_values_run2_s"]].mean(axis=1)
t9 = []
for col in [c for c in A.columns if "=" in c and not c.endswith("_s")]:
    g = 100 * (A[col] - A.LARC_values) / A.LARC_values.abs()
    t9.append(["constructor", col, len(g), f"{g.mean():+.3f} [{g.min():+.2f}, {g.max():+.2f}]",
               f"{(g > 1e-9).sum()}/{(g < -1e-9).sum()}/{(g.abs() <= 1e-9).sum()}", f"{wil(g):.3f}",
               f"{A[col + '_s'].sum() / tb.sum():.2f}"])
B = sheet(oc, "coordinated_2opt")
key = ["instance", "tour"]
b0 = B[(B.damping_f == 0.25) & (B.max_passes == 40)].set_index(key)
for (f, pm), g in B.groupby(["damping_f", "max_passes"]):
    if (f, pm) == (0.25, 40):
        continue
    g = g.set_index(key).loc[b0.index]
    parts = []
    for cp in ("value_30s", "value_60s", "value_120s"):
        gg = 100 * (g[cp] - b0[cp]) / b0[cp].abs()
        parts.append(f"{gg.mean():+.2f} ({(gg > 1e-9).sum()}/{(gg < -1e-9).sum()}, p {wil(gg):.3f})")
    gg = 100 * (g.value_120s - b0.value_120s) / b0.value_120s.abs()
    t9.append(["coordinated 2-opt", f"f={f}" if pm == 40 else f"P_max={pm}", len(g),
               f"{gg.mean():+.3f} [{gg.min():+.2f}, {gg.max():+.2f}]", f"{(gg > 1e-9).sum()}/{(gg < -1e-9).sum()}/{(gg.abs() <= 1e-9).sum()}",
               f"{wil(gg):.3f}", "30/60/120 s: " + "; ".join(parts)])
an = sheet(oc, "annealing_runs")
ref = an[an.variant.str.startswith("LARC repeated")]
for v, g in an.groupby("variant", sort=False):
    base_runs = det if v.startswith("LARC repeated") else ref
    gg = pd.Series([100 * (gi.objective.mean() - base_runs[(base_runs.instance == i) & base_runs.seed.isin(gi.seed)].objective.mean())
                    / abs(base_runs[(base_runs.instance == i) & base_runs.seed.isin(gi.seed)].objective.mean())
                    for i, gi in g.groupby("instance")])
    t9.append(["annealing", v, len(gg), f"{gg.mean():+.3f} [{gg.min():+.2f}, {gg.max():+.2f}]",
               f"{(gg > 1e-9).sum()}/{(gg < -1e-9).sum()}/{(gg.abs() <= 1e-9).sum()}", f"{wil(gg):.3f}",
               "against the main runs" if v.startswith("LARC repeated") else "against LARC repeated"])
chk("Table 9", "\n" + pd.DataFrame(t9, columns=["set", "value", "cases", "gap to LARC mean [min, max]", "better/worse/equal",
                                                "p", "time ratio or note"]).to_string(index=False), "see Table 9")
chk("constructor: total time of the two runs with the values of LARC (s)", (round(A.LARC_values_s.sum()), round(A.LARC_values_run2_s.sum())), (89, 78))

# ------------------------------------------------------------------------------------------------ Section 5.7
section("Section 5.7: candidates of LARC and restarts of CoCo+ on three instances")
for i, paper in (("d2103_n10510_uncorr-similar-weights_05", (182, "about 1,700")),
                 ("rl11849_n59240_uncorr-similar-weights_05", (34, "about 70")),
                 ("pla33810_n169045_uncorr-similar-weights_05", ("eleven", "six to nine"))):
    r = runs[(runs.instance == i) & (runs.method == "CoCo+")]["CoCo+_restarts"]
    chk(f"{i.split('_')[0]}: median candidates of LARC, median restarts of CoCo+",
        (float(det[det.instance == i].candidates.median()), f"median {r.median():g}, range {r.min():g} to {r.max():g}"), paper)

w = max(len(l) for l, _, pp in lines if pp is not None)
for l, v, pp in lines:
    print(l if pp is None else f"{l:<{w}}  {v}   | paper: {pp}")
