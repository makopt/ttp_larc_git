"""Statistical figures of the paper (Figures 7 to 10 and 12 to 16), and the ranking of the nine published methods
without LARC, which is given in the repository only, drawn from the workbooks of results/ and the published values of
data/published_values.xlsx only. The schematic figures are drawn in TikZ in the LaTeX source.
Usage: python3 scripts/analysis/make_figures.py [out_dir]   (default: figures/)
Each figure is written as PDF and PNG. Light pastel colours, one meaning per colour in every figure, and
Friedman-Nemenyi critical-difference diagrams."""
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.stats import friedmanchisquare, spearmanr  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import benchmark, stats as S  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)
RES = REPO / "results"


def sheet(book, name):
    return pd.read_excel(RES / f"{book}.xlsx", sheet_name=name)

# ---------------------------------------------------------------------------- colours and style
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.6,
    "axes.titlesize": 9.4, "axes.titleweight": "normal", "axes.titlepad": 7,
    "axes.labelsize": 8.6, "axes.edgecolor": "#8A8A8A", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#E4E4E4", "grid.linewidth": 0.5,
    "xtick.color": "#333333", "ytick.color": "#333333", "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "legend.fontsize": 7.8, "legend.frameon": False,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
# lime green, lavender and light orange, with darker shades for emphasis
TAN, TAN_L = "#B8E07A", "#EAF6DA"                      # lime green
LAV, LAV_D, LAV_L = "#B399D4", "#7B68AE", "#E6DDF3"
CORAL, CORAL_D, CORAL_L = "#FDB462", "#DD8A3A", "#FEE3C4"   # light orange, as light as the green and the lavender
YEL_L = "#FFE98A"
# LARC and CoCo+ have their own colours, distinct from the three category colours: deep emerald green and brick red, each with
# light tints for the head-to-head polarity ("LARC better" / "CoCo+ better") and the A12 scale
LARC_C, LARC_L, LARC_LL = "#1E7B5A", "#9FD3BD", "#E3F3EC"
COCO_C, COCO_L, COCO_LL = "#B5443A", "#E9AAA3", "#F8E4E1"
INK, MUTED = LARC_C, "#9096A0"                          # emphasis (LARC) and secondary text
GREEN_P, LAV_P, ORANGE_P = "#CDEBA4", "#CBB9E6", "#FECB91"  # lighter versions for large filled areas
CAT_COLOR = {"A": GREEN_P, "B": LAV_P, "C": ORANGE_P}
CAT_MARK = {"A": "o", "B": "s", "C": "^"}
GOOD, BAD, NEUTRAL = "#CDEBA4", "#FECB91", "#DCDCDC"    # LARC better / CoCo+ better / not significant (green and orange of the ranking figures)
GOOD_D, BAD_D = "#6E9A2E", "#DD8A3A"                    # frames of significant differences
SIZE_COLOR = {"n<=300": "#CDBDE6", "300<n<=5000": "#9C85C9", "n>5000": "#5E4C94"}   # light -> dark: small -> large
# convergence figure: the green, lavender and orange of the ranking figures, one shade darker so that thin lines stay visible
CONV_COLOR = {"n<=300": "#86B843", "300<n<=5000": "#9A82CC", "n>5000": "#EE9A45"}
SIZE_LABEL = {"n<=300": "$n\\leq 300$", "300<n<=5000": "$300<n\\leq 5000$", "n>5000": "$n>5000$"}
PHASE_COLOR = {"sampling": GREEN_P, "race": LAV_P, "end": ORANGE_P}   # same three colours as the ranking figures
CMAP_A12 = LinearSegmentedColormap.from_list("a12", ["#FECB91", "#F8F8F8", "#CDEBA4"])
CD_BAR = "#7A7A7A"                                      # bars joining methods that do not differ significantly
W = 6.6                                                 # text width of the paper, inches
METHOD_COLOR = {"LARC": LARC_C, "CoCo+": COCO_C}


FILE_NAMES = {"fig_repack": "figure07_repack_passes", "fig_lhr": "figure08_light_heavy_move",
              "fig_probe": "figure09_probe_screening", "fig_timeline": "figure10_budget_split",
              "fig_a12": "figure12_A12_and_outcomes", "fig_gap": "figure13_gap_to_CoCoPlus",
              "fig_ranks9": "ranks_published_methods_without_LARC", "fig_ranks": "figure14_ranks_with_LARC",
              "fig_conv": "figure15_convergence", "fig_sens": "figure16_sensitivity_sampling_shares"}


def save(fig, name):
    """Writes the figure as PDF and PNG, under the number it has in the paper (the ranking without LARC has none)."""
    name = FILE_NAMES[name]
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("  wrote", OUT / f"{name}.pdf")


def panel(ax, text):
    ax.set_title(text, loc="left")


per = sheet("comparison_LARC_vs_CoCoPlus", "per_instance").rename(columns={
    "LARC_mean": "larc_mean", "LARC_min": "larc_min", "LARC_max": "larc_max", "CoCo+_mean": "coco_mean",
    "CoCo+_min": "coco_min", "CoCo+_max": "coco_max", "gap_of_means_pct": "gap_mean_pct"})
per["base"] = per.instance.str.split("_").str[0]
per["size"] = pd.cut(per.n_cities, [0, 300, 5000, 1e6], labels=list(SIZE_COLOR))
pub = benchmark.published_values()
cocop = benchmark.published_cocop_means()
CAT = per.set_index("instance")["category"]


# ---------------------------------------------------------------------------- 1. damped repack on one candidate
def fig_repack():
    """One real candidate move on dsj1000 with five items per city (data: scripts/experiments/run_figure_examples.py).
    (a) exact gain of the move after each repack pass, (b) number of items flipped in each pass."""
    r = sheet("method_experiments", "repack_passes").rename(columns={"pass": "p", "gain_thousands": "v",
                                                                     "items_dropped": "dropped", "items_added": "added"})
    a, d = r[r.repack_rule.str.startswith("all")], r[r.repack_rule.str.startswith("best")]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W * 0.8, 4.3), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.35, 1.0], hspace=0.22))
    ax.axhspan(0, 150, color=LAV_L, alpha=0.6, lw=0.0)
    ax.axhspan(-780, 0, color=CORAL_L, alpha=0.35, lw=0.0)
    ax.axhline(0, color=MUTED, lw=0.6, ls="--")
    ax.text(20.2, 112, "above zero, the move improves the solution", ha="right", va="center", fontsize=6.6, color="#555555")
    ax.text(20.2, -400, "below zero, it makes the solution worse", ha="right", va="center", fontsize=6.6, color="#555555")
    ax.plot(a.p, a.v, marker="o", ms=3.4, lw=0.8, color=CORAL, label="all items with a positive gain at once")
    ax.plot(d.p, d.v, marker="s", ms=3.2, lw=1.0, color=INK, label="best quarter per pass ($f=0.25$)")
    v0 = a.v.iloc[0]
    ax.annotate(f"pass 0, reversal\nalone: {v0:.0f}k".replace("-", "\u2212"), (0, v0), xytext=(0.15, -560), textcoords="data", fontsize=6.8,
                color="#333333", ha="left", va="center", arrowprops=dict(arrowstyle="-", lw=0.5, color="#777777"))
    ax.annotate("even passes: the dropped\nitems are added back", (12, a.v.iloc[12]), xytext=(15.6, -175),
                textcoords="data", fontsize=6.8, color=CORAL_D, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=CORAL_D))
    ax.annotate("odd passes: about 1,770\nitems are dropped", (11, a.v.iloc[11]), xytext=(15.6, -620),
                textcoords="data", fontsize=6.8, color=CORAL_D, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=CORAL_D))
    ax.annotate(f"+{d.v.iloc[-1]:.1f}k after {int(d.p.iloc[-1])} passes", (d.p.iloc[-1], d.v.iloc[-1]),
                xytext=(-2, -13), textcoords="offset points", ha="right", fontsize=6.8, color=INK)
    ax.set_ylabel("exact gain of the move ($\\times 10^3$)")
    ax.set_ylim(-790, 150)
    ax.set_title("(a)", loc="left", fontsize=8.4)
    w = 0.38
    na = (a.dropped + a.added).to_numpy(); nd = (d.dropped + d.added).to_numpy()
    bx.bar(a.p[1:] - w / 2, na[1:], width=w, color=CORAL, lw=0)
    bx.bar(d.p[1:] + w / 2, nd[1:], width=w, color=INK, lw=0)
    bx.set_yscale("log"); bx.set_ylim(0.7, 4000)
    bx.set_yticks([1, 10, 100, 1000]); bx.set_yticklabels(["1", "10", "100", "1,000"])
    bx.set_ylabel("items flipped in the pass")
    bx.set_xlabel("repack pass")
    bx.set_xlim(-0.5, 20.5); bx.set_xticks(range(0, 21, 2))
    bx.set_title("(b)", loc="left", fontsize=8.4)
    fig.legend(handles=ax.get_lines()[1:3], loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    save(fig, "fig_repack")


# ---------------------------------------------------------------------------- 2. one LHR move on ch130 (real data)
def fig_lhr():
    tr, pts = sheet("method_experiments", "light_heavy_tours"), sheet("method_experiments", "light_heavy_cities")
    ld = sheet("method_experiments", "light_heavy_load")
    t0, t1 = tr[tr.solution == "starting"], tr[tr.solution != "starting"]
    hv, lt, dp = pts[pts.city == "heavy"], pts[pts.city == "light"], pts[pts.city == "depot"]
    p0 = ld[ld.solution == "starting"].reset_index(drop=True)
    p1 = ld[ld.solution != "starting"].reset_index(drop=True)
    fig = plt.figure(figsize=(W, 4.6))
    g = fig.add_gridspec(2, 2, height_ratios=[1.55, 1], hspace=0.38, wspace=0.08)
    for k, (t, ttl) in enumerate([(t0, "(a)"), (t1, "(b)")]):
        ax = fig.add_subplot(g[0, k])
        ax.plot(t.x, t.y, color="#B8B8B8", lw=0.7, zorder=1)
        ax.scatter(lt.x, lt.y, s=10, color="#86B843", zorder=2, label="light cities")
        ax.scatter(hv.x, hv.y, s=16, color="#EE9A45", zorder=3, label="heavy cities ($q=0.75$)")
        ax.scatter(dp.x, dp.y, s=30, marker="s", color="black", zorder=4, label="depot")
        ax.set_title(ttl, loc="left", fontsize=8.4)
        ax.set_aspect("equal"); ax.axis("off")
        if k == 0:
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, -0.13), ncol=3, fontsize=7.4, handletextpad=0.3)
    ax = fig.add_subplot(g[1, :])
    ax.fill_between(p0["frac"], p0["load"], p1.set_index("frac")["load"].reindex(p0["frac"], method="nearest").values,
                    step="post", color=TAN, alpha=0.35, lw=0.0, label="weight saved")
    ax.step(p0["frac"], p0["load"], where="post", color=CORAL, lw=0.96, label="start")
    ax.step(p1["frac"], p1["load"], where="post", color=INK, lw=0.96, label="after one LHR move")
    ax.set_xlabel("fraction of the tour length travelled")
    ax.set_ylabel("carried weight / $W$")
    ax.set_title("(c)", loc="left", fontsize=8.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    save(fig, "fig_lhr")


# ---------------------------------------------------------------------------- 3. how the budget was split (seed 1)
def fig_timeline():
    s = sheet("comparison_LARC_vs_CoCoPlus", "LARC_run_details").rename(
        columns={"sampling_s": "sample_s", "share_of_searches_converged": "converged"})
    s = s[s.seed == 1].set_index("instance")
    rows = [("eil76_n75_bounded-strongly-corr_01", "$\\mathtt{eil76}$, 1 item/city", 76),
            ("rl1304_n1303_bounded-strongly-corr_01", "$\\mathtt{rl1304}$, 1 item/city", 1304),
            ("pla7397_n73960_uncorr_10", "$\\mathtt{pla7397}$, 10 items/city", 7397),
            ("d15112_n75555_uncorr-similar-weights_05", "$\\mathtt{d15112}$, 5 items/city", 15112)]
    T, race_end = 600, 450
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for y, (inst, lab, n) in enumerate(rows[::-1]):
        r = s.loc[inst]
        e = max(r.sample_s, race_end)
        ax.barh(y, r.sample_s, color=PHASE_COLOR["sampling"], height=0.56, edgecolor="white")
        if r.sample_s < race_end:
            ax.barh(y, race_end - r.sample_s, left=r.sample_s, color=PHASE_COLOR["race"], height=0.56, edgecolor="white")
        ax.barh(y, T - e, left=e, color=PHASE_COLOR["end"], height=0.56, edgecolor="white")
        if n > 1500:
            txt = f"{int(r.candidates)} candidates, probe of {min(10, max(0.5, 2.5e-4 * n)):.1f} s each"
        else:
            txt = f"{int(r.candidates):,} candidates, perturbed tours, $\\phi={r.converged:.2f}$"
        ax.text(5, y, txt, fontsize=7.2, va="center", color="#333333")
        ax.text(r.sample_s, y + 0.36, f"{r.sample_s:.0f} s", fontsize=6.8, ha="center", va="bottom", color=INK)
    for x, l, ha in [(270, "$0.45T$", "center"), (450, "$0.75T$", "right"), (480, "$0.8T$", "left")]:
        ax.axvline(x, color=MUTED, lw=0.6, ls="--")
        ax.text(x + (-2 if ha == "right" else 2 if ha == "left" else 0), 1.01, l, fontsize=7.2, color="#555555", ha=ha,
                va="bottom", transform=ax.get_xaxis_transform())
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l for _, l, _ in rows[::-1]])
    ax.set_xlim(0, T); ax.set_ylim(-0.5, len(rows) - 0.1); ax.set_xlabel("time (s), budget $T=600$ s")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=PHASE_COLOR["sampling"], label="sampling"),
                       Patch(color=PHASE_COLOR["race"], label="race"),
                       Patch(color=PHASE_COLOR["end"], label="recombination, final search and repack")],
              loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
    save(fig, "fig_timeline")


# ---------------------------------------------------------------------------- 4. head to head: gap of the means
def fig_gap():
    """(a) gap of the means against n, one marker per instance; (b) distribution of the gaps by category, by size
    group and over all instances, with the Holm-corrected outcomes."""
    fig = plt.figure(figsize=(W, 6.9))
    g = gridspec.GridSpec(2, 1, height_ratios=[1.1, 1.0], hspace=0.5)
    ax, bx = fig.add_subplot(g[0]), fig.add_subplot(g[1])

    # (a) range of the ten runs of each method on every instance, relative to the mean of CoCo+
    G_BAR, O_BAR = "#86B843", "#EE9A45"                 # green and orange of the ranking figures, one shade darker
    G_BG, O_BG = "#EEF8E3", "#FFF0DF"
    ref = per.coco_mean.abs()
    rel = lambda c: 100 * (per[c] - per.coco_mean) / ref
    lo_l, me_l, hi_l = rel("larc_min"), rel("larc_mean"), rel("larc_max")
    lo_c, hi_c = rel("coco_min"), rel("coco_max")
    order = per.sort_values(["category", "n_cities"]).index
    xs, x, prev = {}, 0.0, None
    for k in order:
        if prev is not None and per.category[k] != prev:
            x += 1.2
        xs[k] = x; x += 1.0; prev = per.category[k]
    ax.axhline(0, color=MUTED, lw=0.6, ls="--", zorder=1)
    for k in order:
        x0 = xs[k]
        if per.larc_min[k] > per.coco_max[k]:
            ax.axvspan(x0 - 0.5, x0 + 0.5, color=G_BG, lw=0.0, zorder=0)
        elif per.coco_min[k] > per.larc_max[k]:
            ax.axvspan(x0 - 0.5, x0 + 0.5, color=O_BG, lw=0.0, zorder=0)
        ax.plot([x0 - 0.17] * 2, [lo_l[k], hi_l[k]], color=G_BAR, lw=1.5, solid_capstyle="butt", zorder=3)
        ax.plot([x0 + 0.17] * 2, [lo_c[k], hi_c[k]], color=O_BAR, lw=1.5, solid_capstyle="butt", zorder=3)
        ax.plot([x0 - 0.17], [me_l[k]], marker="o", ms=2.6, mfc="white", mec=G_BAR, mew=0.7, zorder=4)
        ax.plot([x0 + 0.17], [0], marker="o", ms=2.6, mfc="white", mec=O_BAR, mew=0.7, zorder=4)
    ax.set_xticks([xs[k] for k in order]); ax.set_xticklabels([per.base[k] for k in order], rotation=90, fontsize=5.0, fontfamily="DejaVu Sans Mono")  # instance names in monospace, as \texttt in the text
    ax.tick_params(axis="x", length=0, pad=2); ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.8, x - 0.2); ax.set_ylim(-4.2, 6.6)
    for c in "ABC":
        ks = [k for k in order if per.category[k] == c]
        ax.text((xs[ks[0]] + xs[ks[-1]]) / 2, 6.25, f"category {c}", ha="center", va="center", fontsize=7.0,
                color="#555555")
    ax.set_ylabel("gap to the CoCo$^+$ mean (%)")
    n_l = int((per.larc_min > per.coco_max).sum()); n_c = int((per.coco_min > per.larc_max).sum())
    h = [Line2D([], [], color=G_BAR, lw=1.5, marker="o", ms=2.6, mfc="white", mec=G_BAR, label="LARC, min to max (mean)"),
         Line2D([], [], color=O_BAR, lw=1.5, marker="o", ms=2.6, mfc="white", mec=O_BAR,
                label="CoCo$^+$, min to max (mean)"),
         Patch(color=G_BG, label=f"worst LARC run above best CoCo$^+$ run ({n_l})"),
         Patch(color=O_BG, label=f"worst CoCo$^+$ run above best LARC run ({n_c})")]
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=6.4, columnspacing=1.0,
              handletextpad=0.4, handlelength=1.6, borderaxespad=0.3)
    ax.text(-0.085, 1.12, "(a)", transform=ax.transAxes, fontsize=9.4, va="center")

    # (b) box plots by group
    size = pd.cut(per.n_cities, [0, 300, 5000, 1e6], labels=list(SIZE_COLOR))
    groups = [(f"category {c}", per.category == c, CAT_COLOR[c]) for c in "ABC"]
    groups += [(SIZE_LABEL[k], size == k, "#E4E4E4") for k in SIZE_COLOR]
    groups += [("all 60", per.category.notna(), "#C8C8C8")]
    pos = [0, 1.15, 2.3, 3.95, 5.25, 6.55, 8.2]
    OUTC = {"win": GOOD_D, "loss": BAD_D, "tie": "#B0B0B0"}
    bx.axhspan(0, 6, color=G_BG, lw=0.0); bx.axhspan(-6, 0, color=O_BG, lw=0.0)
    bx.axhline(0, color=MUTED, lw=0.6, ls="--")
    rng = np.random.default_rng(7)
    ticks = []
    for p_, (lab, m, colr) in zip(pos, groups):
        d = per[m]
        v = d.gap_mean_pct.values
        bp = bx.boxplot(v, positions=[p_], widths=0.56, patch_artist=True, showfliers=False, whis=1.5,
                        medianprops=dict(color="#222222", lw=0.9), whiskerprops=dict(color="#777777", lw=0.7),
                        capprops=dict(color="#777777", lw=0.7), boxprops=dict(edgecolor="#777777", lw=0.7))
        bp["boxes"][0].set_facecolor(colr)
        bx.scatter(p_ + rng.uniform(-0.2, 0.2, len(v)), v, s=8, c=[OUTC[o] for o in d.outcome], zorder=3, lw=0.0)
        bx.scatter([p_], [v.mean()], marker="D", s=20, color="white", edgecolor="#222222", lw=0.8, zorder=4)
        w, t, l = (d.outcome == "win").sum(), (d.outcome == "tie").sum(), (d.outcome == "loss").sum()
        bx.text(p_, 5.5, f"{v.mean():+.2f}", ha="center", va="center", fontsize=6.6, color="#222222")
        ticks.append(f"{lab}\n{w} / {t} / {l}")
    bx.text(-0.8, 5.5, "mean", ha="right", va="center", fontsize=6.6, color="#222222")
    bx.annotate("LARC better / n.s. / worse", (0, 0), xycoords="axes fraction", xytext=(-6, -16.5),
                textcoords="offset points", ha="right", va="center", fontsize=6.4, color="#333333")
    for x in (3.12, 7.37):
        bx.axvline(x, color="#CCCCCC", lw=0.6)
    bx.set_xticks(pos); bx.set_xticklabels(ticks, fontsize=6.6, linespacing=1.5)
    bx.set_xlim(-0.65, 8.85); bx.set_ylim(-3.2, 5.9); bx.grid(axis="x", visible=False)
    bx.set_ylabel("gap of the means (%)")
    hb = [Patch(facecolor="#E4E4E4", edgecolor="#777777", lw=0.7, label="quartiles (box)"),
          Line2D([], [], color="#222222", lw=0.9, label="median"),
          Line2D([], [], marker="D", ls="", ms=4.5, markerfacecolor="white", markeredgecolor="#222222", mew=0.8,
                 label="mean"),
          Line2D([], [], color="#777777", lw=0.7, label="whiskers (1.5 IQR)"),
          Line2D([], [], marker="o", ls="", ms=3.5, color=GOOD_D, label="LARC better"),
          Line2D([], [], marker="o", ls="", ms=3.5, color="#B0B0B0", label="n.s."),
          Line2D([], [], marker="o", ls="", ms=3.5, color=BAD_D, label="CoCo$^+$ better")]
    bx.legend(handles=hb, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=7, fontsize=6.2, columnspacing=0.8,
              handletextpad=0.3, handlelength=1.3, borderaxespad=0.3)
    bx.text(-0.085, 1.07, "(b)", transform=bx.transAxes, fontsize=9.4, va="center")
    save(fig, "fig_gap")


# ---------------------------------------------------------------------------- 5. head to head: A12 grid and outcomes
def fig_a12():
    fig = plt.figure(figsize=(W, 4.4))
    g = gridspec.GridSpec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1, 1], wspace=0.55, hspace=0.75)
    ax = fig.add_subplot(g[:, 0])
    bases = per.drop_duplicates("base").sort_values("n_cities")[["base", "n_cities"]].values
    for j, c in enumerate("ABC"):
        for i, (b, n) in enumerate(bases):
            r = per[(per.base == b) & (per.category == c)].iloc[0]
            ax.add_patch(Rectangle((j, i), 1, 1, color=CMAP_A12(r.A12), ec="white", lw=0.72))
            if r.outcome != "tie":
                ax.add_patch(Rectangle((j + 0.05, i + 0.08), 0.9, 0.84, fill=False,
                                       ec=GOOD_D if r.outcome == "win" else BAD_D, lw=0.9))
            ax.text(j + 0.5, i + 0.5, f"{r.gap_mean_pct:+.2f}", ha="center", va="center", fontsize=6.4,
                    color="#1A1A1A")
    ax.set_xlim(0, 3); ax.set_ylim(len(bases), 0)
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(["A\n1 item/city", "B\n5 items/city", "C\n10 items/city"],
                                                       fontsize=7.2)
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(bases)) + 0.5); ax.set_yticklabels([b for b, _ in bases], fontsize=7, fontfamily="DejaVu Sans Mono")  # instance names in monospace
    ax.grid(False); [s.set_visible(False) for s in ax.spines.values()]; ax.tick_params(length=0)
    ax.set_title("(a)", loc="left", pad=26)
    sm = matplotlib.cm.ScalarMappable(cmap=CMAP_A12, norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.03, aspect=28)
    cb.set_label("$\\hat A_{12}$ = P(a LARC run beats a CoCo$^+$ run)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.8); cb.outline.set_visible(False)
    # (b), (c): outcome counts by size and by category
    for k, (col, labels, ttl) in enumerate([("size", [SIZE_LABEL[s] for s in SIZE_COLOR], "(b)"),
                                            ("category", [f"category {c}" for c in "ABC"], "(c)")]):
        bx = fig.add_subplot(g[k, 1])
        tab = pd.crosstab(per[col], per.outcome).reindex(columns=["win", "tie", "loss"], fill_value=0)
        left = np.zeros(len(tab))
        for o, colr, lab in [("win", GOOD, "LARC better"), ("tie", NEUTRAL, "not significant"),
                             ("loss", BAD, "CoCo$^+$ better")]:
            bx.barh(range(len(tab)), tab[o], left=left, color=colr, label=lab, height=0.62)
            for i, v in enumerate(tab[o]):
                if v:
                    bx.text(left[i] + v / 2, i, str(v), ha="center", va="center", fontsize=7.4)
            left += tab[o].values
        bx.set_yticks(range(len(tab))); bx.set_yticklabels(labels, fontsize=7.4); bx.invert_yaxis()
        bx.set_xlim(0, 25); bx.grid(axis="y", visible=False)
        bx.set_title(ttl, loc="left", fontsize=8.6)
        if k == 1:
            bx.set_xlabel("instances")
            bx.legend(loc="upper center", bbox_to_anchor=(0.3, -0.42), ncol=3, fontsize=6.8, handlelength=1.0,
                      handletextpad=0.4, columnspacing=0.9)
    save(fig, "fig_a12")


# ---------------------------------------------------------------------------- 6./7. state of the art before / after LARC
Q05 = {9: 3.102, 10: 3.164}


def r2(v):
    """Two decimals, rounding half to even on the exact value (mean ranks are multiples of 1/120)."""
    from decimal import Decimal, ROUND_HALF_EVEN
    return str(Decimal(repr(round(float(v), 9))).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def _cd(ax, P, title, hl=METHOD_COLOR):
    """Critical-difference diagram (Demsar 2006): methods joined by a grey bar do
    not differ significantly (Nemenyi, alpha 0.05)."""
    rk = S.mean_ranks(P).sort_values()
    k, N = P.shape[1], P.shape[0]
    cd = Q05[k] * math.sqrt(k * (k + 1) / (6 * N))
    chi = friedmanchisquare(*[P[c] for c in P]).statistic
    names, ranks = list(rk.index), rk.values
    lo, hi = 1, math.ceil(ranks.max()) + 1
    ax.set_xlim(lo - 2.9, hi + 2.9); ax.set_ylim(-2.3, 1.3); ax.axis("off")
    ax.text(lo - 2.9, 1.2, title, ha="left", fontsize=7.6)
    ax.plot([lo, hi], [0, 0], color="#333", lw=0.72)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.08], color="#333", lw=0.6)
        ax.text(t, 0.2, str(t), ha="center", fontsize=6.6)
    ax.plot([lo, lo + cd], [0.7, 0.7], color="black", lw=0.9, ls="--")
    ax.text(lo + cd + 0.12, 0.7, f"CD = {cd:.2f}", va="center", fontsize=6.4, color="black")
    half = (k + 1) // 2
    for i, (nm, r) in enumerate(zip(names, ranks)):
        left = i < half
        lev = -0.85 - (i if left else k - 1 - i) * 0.3
        xend = lo - 0.2 if left else hi + 0.2
        colr = hl.get(nm, "#999999")
        ax.plot([r, r], [0, lev], color=colr, lw=0.9 if nm in hl else 0.6)
        ax.plot([r, xend], [lev, lev], color=colr, lw=0.9 if nm in hl else 0.6)
        ax.text(xend + (-0.12 if left else 0.12), lev, f"{nm} ({r2(r)})", ha="right" if left else "left", va="center",
                fontsize=6.6, color=hl.get(nm, "#333333"))
    bn, prev_j = 0, -1
    for i in range(k):
        j = i
        while j + 1 < k and ranks[j + 1] - ranks[i] < cd:
            j += 1
        if j > i and j > prev_j:
            yy = -0.26 - bn * 0.13
            ax.plot([ranks[i] - 0.03, ranks[j] + 0.03], [yy, yy], color=CD_BAR, lw=1.4, solid_capstyle="round")
            prev_j, bn = j, bn + 1
    print(f"  {title}: chi2 {chi:.1f}, CD {cd:.2f}, " + ", ".join(f"{m} {v:.2f}" for m, v in rk.items()))


def _rdi_bars(ax, P, title, hl=METHOD_COLOR, legend=True):
    R = S.rdi(P)
    m = R.groupby(CAT.loc[R.index]).mean().T
    order = R.mean().sort_values().index
    m = m.loc[order]
    x, w = np.arange(len(m)), 0.26
    for off, c in zip((-w, 0, w), "ABC"):
        ax.bar(x + off, m[c], w, color=CAT_COLOR[c], label=f"category {c}")
        for xi, v in zip(x, m[c]):
            ax.text(xi + off, v + 1.5, f"{v:.0f}", ha="center", fontsize=5.2, color="#333333")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=15, ha="right", fontsize=6.6)
    for t in ax.get_xticklabels():
        t.set_color(hl.get(t.get_text(), "#333333"))
    ax.set_ylabel("mean RDI (%), lower is better")
    ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    if legend:
        ax.legend(ncol=3, loc="upper left", fontsize=6.4)
    panel(ax, title)


def _sota(means, best, means_ttl, best_ttl, name, hl=METHOD_COLOR):
    """(a) mean RDI by category and (b) CD diagram with mean values; (c) mean RDI and (d) CD diagram with best values."""
    rc = {"axes.titlesize": 7.6, "axes.labelsize": 6.8, "xtick.labelsize": 6.4, "ytick.labelsize": 6.4}
    with plt.rc_context(rc):
        _sota_draw(means, best, means_ttl, best_ttl, name, hl)


def _sota_draw(means, best, means_ttl, best_ttl, name, hl):
    fig = plt.figure(figsize=(W, 6.6))
    g = gridspec.GridSpec(4, 1, height_ratios=[1.0, 1.05, 1.0, 1.05], hspace=0.38)
    _rdi_bars(fig.add_subplot(g[0]), means, f"(a) {means_ttl}", hl, legend=True)
    _cd(fig.add_subplot(g[1]), means, f"(b) {means_ttl}", hl)
    _rdi_bars(fig.add_subplot(g[2]), best, f"(c) {best_ttl}", hl, legend=False)
    _cd(fig.add_subplot(g[3]), best, f"(d) {best_ttl}", hl)
    save(fig, name)


def fig_ranks9():
    """Before LARC: the nine published methods; CoCo+ by the published CoCoP means or by its published best values."""
    idx = per.instance
    means, best = pub.loc[idx].assign(**{"CoCo+": cocop.loc[idx]}), pub.loc[idx]
    _sota(means, best, "mean values", "best values for CoCo$^+$", "fig_ranks9",
          hl={})                                        # before LARC: no method is singled out


def fig_ranks():
    """After LARC: LARC and the nine published methods, by means, and with LARC and CoCo+ by best values."""
    idx = per.instance
    P = per.set_index("instance")
    means = pub.loc[idx].assign(**{"CoCo+": cocop.loc[idx], "LARC": P["larc_mean"].loc[idx]})
    best = pub.loc[idx].assign(LARC=P["larc_max"].loc[idx])
    _sota(means, best, "mean values", "best values for LARC and CoCo$^+$", "fig_ranks")


# ---------------------------------------------------------------------------- 8. probe screening (d2103, 320 tours)
def fig_probe():
    """(a, b) the 320 candidates of the probe on d2103 with five items per city; (c) the rank correlations of the
    constructor value and of the probe value with the value after the longer search, on each of the 33 instances with
    more than 1,500 cities (results/probe_screening_large.xlsx, as in scripts/analysis/probe_screening_large.py)."""
    d = sheet("method_experiments", "probe_screening").rename(columns={
        "constructor_value": "built", "value_after_first_probe": "ls05", "value_after_second_probe": "ls2"})
    d = d[d.instance == "d2103_n10510_uncorr-similar-weights_05"].reset_index(drop=True)
    top = d.nlargest(10, "ls2").index
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.55), gridspec_kw={"wspace": 0.42})
    for ax, (col, lab) in zip(axs[:2], [("built", "(a)"), ("ls05", "(b)")]):
        rho = spearmanr(d[col], d.ls2).statistic
        kept = d.nlargest(32, col).index
        ax.axvspan(d.loc[kept, col].min() / 1e3, d[col].max() / 1e3 + 3, color=TAN_L, alpha=0.6, lw=0.0)
        ax.scatter(d[col] / 1e3, d.ls2 / 1e3, s=7, color=LAV_L, edgecolor=LAV, lw=0.5, zorder=2)
        ax.scatter(d.loc[top, col] / 1e3, d.loc[top, "ls2"] / 1e3, s=18, color=CORAL, edgecolor=CORAL_D, lw=0.6,
                   zorder=3, label="10 best after 2 s")
        n_in = len(set(kept) & set(top))
        ax.text(0.97, 0.04, f"$\\rho_s$ = {rho:.2f}\n32 best kept\n{n_in} of the 10", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=6.8, color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
        panel(ax, lab)
        ax.set_xlabel(f"{'constructor value' if col == 'built' else 'value after 0.5 s'} ($\\times 10^3$)")
        ax.set_ylabel("value after 2 s ($\\times 10^3$)")
    t = pd.read_excel(RES / "probe_screening_large.xlsx", sheet_name="traces")
    rows = []
    for inst, g in t.groupby("instance"):
        n_c, n_i = (int(x) for x in re.match(r"^[A-Za-z]+(\d+)_n(\d+)_", inst).groups())
        rows.append(dict(cat={1: "A", 5: "B", 10: "C"}[round(n_i / (n_c - 1))],
                         rc=spearmanr(g.constructor_value, g.value_after_longer_search).statistic,
                         rp=spearmanr(g.value_after_probe, g.value_after_longer_search).statistic))
    r = pd.DataFrame(rows)
    cx = axs[2]
    lo = min(r.rc.min(), r.rp.min()) - 0.05
    cx.plot([lo, 1.02], [lo, 1.02], color=MUTED, lw=0.7, ls="--", zorder=1)
    for c in "ABC":
        g = r[r.cat == c]
        cx.scatter(g.rc, g.rp, s=16, marker=CAT_MARK[c], color=CAT_COLOR[c], edgecolor="#555555", lw=0.5, zorder=2,
                   label=f"category {c} ({len(g)})")
    cx.set_xlim(lo, 1.02); cx.set_ylim(lo, 1.02)
    cx.text(0.97, 0.04, f"probe better on\n{int((r.rp > r.rc).sum())} of {len(r)} instances", transform=cx.transAxes,
            va="bottom", ha="right", fontsize=6.8, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", lw=0.6))
    panel(cx, "(c)")
    cx.set_xlabel("$\\rho_s$ of the constructor value")
    cx.set_ylabel("$\\rho_s$ of the probe value")
    h = [Line2D([], [], marker="o", ls="", color=CORAL, markeredgecolor=CORAL_D, label="10 best after 2 s"),
         Patch(color=TAN_L, label="32 best by the horizontal value")]
    h += [Line2D([], [], marker=CAT_MARK[c], ls="", color=CAT_COLOR[c], markeredgecolor="#555555",
                 label=f"category {c}") for c in "ABC"]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, fontsize=6.8, columnspacing=1.0,
               handletextpad=0.3)
    save(fig, "fig_probe")


# ---------------------------------------------------------------------------- 9. convergence of LARC (600 runs)
def fig_conv():
    q = sheet("convergence_LARC", "gap_to_final")
    q.columns = ["size", "t", "q25", "q50", "q75"]
    sh = sheet("convergence_LARC", "share_above_CoCo+").set_index("time_s")
    sh.columns = [c.replace("share_pct_", "") for c in sh.columns]
    cs = sheet("convergence_LARC", "by_size_group").set_index("size_group").rename(columns={"t_sampling_end_s": "t_sampling_end"})
    labs = {"n<=300": "$n\\leq 300$ (15)", "300<n<=5000": "$300<n\\leq 5000$ (24)", "n>5000": "$n>5000$ (21)"}
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(W, 5.4), gridspec_kw={"hspace": 0.38})
    for a_ in (ax, bx):
        a_.axvspan(450, 600, color="#F2F2F2", lw=0.0)
        a_.axvline(450, color=MUTED, lw=0.6, ls=":")
        a_.text(455, 0.98, "after the race", transform=a_.get_xaxis_transform(), fontsize=6.8, color="#555555", va="top")
    for sc, colr in CONV_COLOR.items():
        g = q[q["size"] == sc]
        ax.plot(g.t, g.q50, color=colr, lw=1.08, label=labs[sc])
        ax.fill_between(g.t, g.q25, g.q75, color=colr, alpha=0.16, lw=0.0)
        ts = cs.loc[sc, "t_sampling_end"]
        ax.plot([ts], [np.interp(ts, g.t, g.q50)], marker="o", ms=4.5, mfc="white", mec=colr, mew=1.2, ls="", zorder=4)
        bx.plot(sh.index, sh[sc], color=colr, lw=1.08, drawstyle="steps-post")
    bx.plot(sh.index, sh["all"], color="#333333", lw=0.72, ls="--", drawstyle="steps-post", label="all 60 instances")
    ax.set_yscale("symlog", linthresh=0.01); ax.set_ylim(0, 30)
    ax.set_yticks([0, 0.01, 0.1, 1, 10]); ax.set_yticklabels(["0", "0.01", "0.1", "1", "10"])
    ax.set_xlim(0, 600); bx.set_xlim(0, 600); ax.set_xlabel("time (s)"); bx.set_xlabel("time (s)")
    ax.set_ylabel("gap to the final value of the run (%)")
    panel(ax, "(a)")
    bx.set_ylabel("instances where LARC leads (%)"); bx.set_ylim(0, 100)
    panel(bx, "(b)")
    h, _ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], marker="o", ms=4.5, ls="", mfc="white", mec="#555555", mew=1.2, label="median end of sampling"))
    ax.legend(handles=h, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=6.8, frameon=False,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0.2)
    bx.legend(loc="upper left", fontsize=7.2)
    for sc, colr in CONV_COLOR.items():
        bx.text(603, sh[sc].iloc[-1], f"{sh[sc].iloc[-1]:.0f}%", fontsize=6.8, color=colr, va="center")
    save(fig, "fig_conv")


# ---------------------------------------------------------------------------- 10. sensitivity to the sampling shares
def fig_sens():
    """Gap of each variant to LARC (same instances, same seeds): (a) sampling end on n > 1500, (b) latest sampling end
    on n <= 1500. One dot per instance; LARC is the zero line."""
    VAR = {"L030": "0.30T", "L060": "0.60T", "U045": "0.45T", "S060": "0.6T", "S090": "0.9T"}
    summ = sheet("sensitivity_sampling_shares", "summary").rename(columns={
        "gap_to_LARC_mean_pct": "gap_mean", "p_Wilcoxon": "wilcoxon_p"})
    summ = summ.set_index(summ.tested_value.map({v: k for k, v in VAR.items()}))
    pi = sheet("sensitivity_sampling_shares", "per_instance").rename(columns={"instance": "inst", "gap_to_LARC_pct": "gap"})
    sens = {k: pi[pi.tested_value == v].reset_index(drop=True) for k, v in VAR.items()}
    fig = plt.figure(figsize=(W, 3.6))
    g = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.25], wspace=0.28)
    OUTC = lambda v: GOOD_D if v > 0 else (BAD_D if v < 0 else "#B0B0B0")
    rng = np.random.default_rng(3)

    def boxes(ax, groups, larc_x, xlab):
        ax.axhspan(0, 2, color=LARC_LL, lw=0.0, zorder=0); ax.axhspan(-2, 0, color=COCO_LL, lw=0.0, zorder=0)
        ax.axhline(0, color="#333333", lw=0.8, ls="--", zorder=2)
        for x, v in groups:
            d = sens[v]
            bp = ax.boxplot(d.gap, positions=[x], widths=0.075, patch_artist=True, showfliers=False, whis=1.5, manage_ticks=False,
                            medianprops=dict(color="#222222", lw=0.9), whiskerprops=dict(color="#777777", lw=0.7),
                            capprops=dict(color="#777777", lw=0.7), boxprops=dict(edgecolor="#777777", lw=0.7))
            bp["boxes"][0].set_facecolor("#E4E4E4")
            ax.scatter(x + rng.uniform(-0.028, 0.028, len(d)), d.gap, s=9, c=[OUTC(v_) for v_ in d.gap], zorder=3,
                       lw=0.0)
            ax.scatter([x], [d.gap.mean()], marker="D", s=20, color="white", edgecolor="#222222", lw=0.8, zorder=4)
            r = summ.loc[v]
            ax.text(x, 0.93, f"{r.gap_mean:+.3f}", ha="center", fontsize=6.6, color="#222222")
            ax.text(x, -1.08, f"{r.better} better\n{r.worse} worse\n$p$ = {r.wilcoxon_p:.3f}", ha="center", va="top",
                    fontsize=6.2, color="#333333", linespacing=1.25)
        ax.annotate("LARC", (larc_x, 0), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.8,
                    color="#222222")
        ax.plot([larc_x], [0], marker="o", ms=4.5, mfc="white", mec="#222222", mew=1.0, zorder=5)
        ax.set_ylim(-1.55, 1.08); ax.set_xlabel(xlab); ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(g[0])
    for inst in sens["L030"].inst:                                      # one line per instance, 0.30 -> 0.45 -> 0.60
        a = sens["L030"].set_index("inst").gap[inst]
        b = sens["L060"].set_index("inst").gap[inst]
        ax.plot([0.30, 0.45, 0.60], [a, 0, b], color="#AAAAAA", lw=0.45, zorder=1.5)
    boxes(ax, [(0.30, "L030"), (0.60, "L060")], 0.45, "sampling end on $n>1500$ (share of $T$)")
    ax.set_xticks([0.30, 0.45, 0.60]); ax.set_xlim(0.22, 0.68)
    ax.set_ylabel("gap of the means to LARC (%)")
    ax.text(-0.17, 1.06, "(a)", transform=ax.transAxes, fontsize=9.4, va="center")
    ax.text(0.5, 1.06, "33 instances with $n>1500$", transform=ax.transAxes, ha="center", va="center", fontsize=7.2)

    bx = fig.add_subplot(g[1])
    boxes(bx, [(0.45, "U045"), (0.60, "S060"), (0.90, "S090")], 0.80, "latest sampling end on $n\\leq 1500$ (share of $T$)")
    bx.set_xticks([0.45, 0.60, 0.80, 0.90]); bx.set_xlim(0.37, 0.98)
    for x, v in [(0.45, "U045"), (0.60, "S060"), (0.90, "S090")]:
        bx.text(x, 1.0, f"{int(summ.loc[v].instances)} inst.", ha="center", va="bottom", fontsize=6.2, color="#555555",
                transform=bx.get_xaxis_transform())
    bx.text(-0.12, 1.06, "(b)", transform=bx.transAxes, fontsize=9.4, va="center")
    h = [Line2D([], [], marker="o", ls="", ms=3.5, color=GOOD_D, label="better than LARC"),
         Line2D([], [], marker="o", ls="", ms=3.5, color=BAD_D, label="worse than LARC"),
         Line2D([], [], marker="o", ls="", ms=3.5, color="#B0B0B0", label="equal"),
         Line2D([], [], marker="D", ls="", ms=4.5, markerfacecolor="white", markeredgecolor="#222222", mew=0.8,
                label="mean (value above)"),
         Patch(facecolor="#E4E4E4", edgecolor="#777777", lw=0.7, label="quartiles"),
         Line2D([], [], color="#BBBBBB", lw=0.6, label="one instance")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=6.6, columnspacing=0.9,
               handletextpad=0.3, handlelength=1.3)
    save(fig, "fig_sens")


if __name__ == "__main__":
    fig_repack(); fig_lhr(); fig_timeline(); fig_gap(); fig_a12(); fig_ranks9(); fig_ranks(); fig_probe(); fig_conv(); fig_sens()
