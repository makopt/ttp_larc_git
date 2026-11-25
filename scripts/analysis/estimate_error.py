"""Error of the load-aware estimates on the final solutions of LARC (Proposition 1 and the text after it).

Input: results/final_solutions_LARC_ablation_baseline.tar.xz, the final tour and packing of the 600 runs of LARC made with
the ablation (10 seeds on each of the 60 instances). For each solution, up to 2,000 items are drawn at random (all items
when there are fewer; numpy seed 20261120), and for each drawn item at tour position k, with x_s = nu w / v(c_s):
  - a picked item: the exact travel time saved by removing it, its second-order estimate nu w G1 - nu^2 w^2 G2, the
    relative error E_rem / Delta T_rem (Proposition 1(b)), and the relative error of the first-order estimate nu w G1
    (negative: the first-order estimate overestimates the time saved);
  - an item that is not picked and still fits: the exact travel time added by picking it, its second-order estimate
    nu w G1 + nu^2 w^2 G2, the relative error E_add / Delta T_add (Proposition 1(a)), and the relative error of the
    first-order estimate nu w G1 alone;
  - rho = max over s >= k of x_s.
Relative errors are nonnegative by Proposition 1. Results by size class and category: median, 95th percentile and
maximum of the relative errors, and the share of items with rho > 0.5.
Usage: python3 scripts/analysis/estimate_error.py
"""
import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import benchmark  # noqa: E402
from paper_check import check, summary  # noqa: E402

SAMPLE = 2000
meta = pd.read_excel(benchmark.INSTANCE_LIST, sheet_name="instances").set_index("instance")
rows = []
with tarfile.open(REPO / "results/final_solutions_LARC_ablation_baseline.tar.xz") as tf:
    members = sorted((m for m in tf.getmembers() if m.name.endswith(".npz") and "/._" not in m.name),
                     key=lambda m: m.name)
    cache = {}
    for mem in members:
        name, seed = Path(mem.name).stem.rsplit("_s", 1)
        if name not in cache:
            cache = {name: benchmark.load_instance(name)}
        inst = cache[name]
        z = np.load(io.BytesIO(tf.extractfile(mem).read()))
        tour, picked = z["tour"].astype(np.int64), z["picked"].astype(bool)
        n = inst.n
        pos = np.empty(n, np.int64)
        pos[tour] = np.arange(n)
        closed = np.r_[tour, tour[:1]]
        d = inst.dist(closed[:-1], closed[1:]).astype(float)
        cw = np.bincount(inst.city[picked], weights=inst.weight[picked], minlength=n)
        load = np.cumsum(cw[tour])
        v = inst.v_max - inst.nu * load
        g1 = np.cumsum((d / v ** 2)[::-1])[::-1]
        g2 = np.cumsum((d / v ** 3)[::-1])[::-1]
        free = inst.capacity - inst.weight[picked].sum()
        rng = np.random.default_rng(20261120 + int(seed))
        items = np.arange(inst.m) if inst.m <= SAMPLE else rng.choice(inst.m, SAMPLE, replace=False)
        for j in items:
            k, w = pos[inst.city[j]], float(inst.weight[j])
            nw = inst.nu * w
            vs, ds = v[k:], d[k:]
            rho = float((nw / vs).max())
            if picked[j]:
                exact = float((ds / vs - ds / (vs + nw)).sum())
                est = nw * g1[k] - nw * nw * g2[k]
                rows.append((name, "removal", rho, (exact - est) / exact, (exact - nw * g1[k]) / exact))
            elif w <= free:
                exact = float((ds / (vs - nw) - ds / vs).sum())
                est = nw * g1[k] + nw * nw * g2[k]
                rows.append((name, "addition", rho, (exact - est) / exact, (exact - nw * g1[k]) / exact))

d = pd.DataFrame(rows, columns=["instance", "move", "rho", "rel_error", "rel_error_first_order"])
d["category"] = d.instance.map(meta.category)
d["size"] = pd.cut(d.instance.map(meta.n_cities), [0, 300, 5000, 1e6],
                   labels=["n<=300", "300<n<=5000", "n>5000"]).astype(str)


def summ(g):
    return pd.Series(dict(items=len(g), median_pct=100 * g.rel_error.median(), p95_pct=100 * g.rel_error.quantile(0.95),
                          max_pct=100 * g.rel_error.max(), first_order_median_pct=100 * g.rel_error_first_order.abs().median(),
                          first_order_p95_pct=100 * g.rel_error_first_order.abs().quantile(0.95),
                          share_rho_above_half_pct=100 * (g.rho > 0.5).mean(),
                          share_rho_above_one_pct=100 * (g.rho > 1).mean(),
                          share_error_above_1pct=100 * (g.rel_error > 0.01).mean(),
                          share_error_above_10pct=100 * (g.rel_error > 0.1).mean(),
                          share_negative_estimate_pct=100 * (g.rel_error > 1).mean()))


pd.set_option("display.width", 200)
print("relative error of the second-order estimate (and of the first-order estimate for additions), in percent")
print("min relative error (must be >= 0 by Proposition 1):", d.rel_error.min())
print(d.groupby("move").apply(summ, include_groups=False).round(4).to_string())
for col in ["size", "category"]:
    print(d.groupby(["move", col]).apply(summ, include_groups=False).round(4).to_string())

print("\n== Numbers printed in the paper (after Proposition 1)")
add, rem = d[d.move == "addition"], d[d.move == "removal"]
check("possible additions, removals", (len(add), len(rem)), (300504, 467365))
check("relative error, 95th percentile (%): additions, removals",
      (100 * add.rel_error.quantile(0.95), 100 * rem.rel_error.quantile(0.95)), (0.005, 0.006))
check("first-order estimate alone, 95th percentile (%): additions, removals",
      (100 * add.rel_error_first_order.abs().quantile(0.95), 100 * rem.rel_error_first_order.abs().quantile(0.95)),
      (0.63, "0.70"))
check("errors above 1%, share of the items (%): additions, removals",
      (100 * (add.rel_error > 0.01).mean(), 100 * (rem.rel_error > 0.01).mean()), (0.005, 0.31))
check("errors above 1% all on instances of category A", bool((d[d.rel_error > 0.01].category == "A").all()), True)
big = rem[rem.rho > 1]
check("removals with rho > 1: count, share (%), all on instances with at most 300 cities",
      (len(big), 100 * len(big) / len(rem), bool((big["size"] == "n<=300").all())), (52, 0.011, True))
check("removals with a negative estimate of the time saved", int((rem.rel_error > 1).sum()), 42)
check("estimate never above the exact value (Proposition 1): smallest relative error", float(d.rel_error.min()), ">=-1e-8")
summary()
