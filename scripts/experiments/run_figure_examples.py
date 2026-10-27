"""Data of the two figures of the paper that show one real example each:

  repack_passes   Figure 7: one candidate 2-opt move on dsj1000 with five items per city (reversal of positions 196 to
                  953 of a constructor solution); exact gain of the move after each repack pass, for the rule that
                  flips all items with a positive gain at once and for the damped rule of LARC (best quarter per pass,
                  f = 0.25), with the number of items dropped and added in each pass.
  light_heavy     Figure 8: one light-heavy re-sequencing move on ch130 with five items per city: the starting solution
                  (constructor and local search until convergence) and the best result of the move over the four
                  thresholds q in {0, 0.25, 0.5, 0.75}; tours, load profiles and the cities of each class.

Usage: python3 scripts/experiments/run_figure_examples.py [repack_passes|light_heavy|all]
Output: runs/figure_examples.xlsx (sheets as in results/method_experiments.xlsx)."""
import sys

import numpy as np
import pandas as pd

from common import RUNS_DIR, benchmark, write_xlsx
from larc import construct as c, tours  # noqa: E402
from larc.evaluator import city_weights, objective  # noqa: E402
from larc.search import State, knn_lists, local_search  # noqa: E402


def repack_passes():
    inst = benchmark.load_instance("dsj1000_n4995_uncorr-similar-weights_05")
    t, _ = tours.linkern_tour(inst, time_limit=60, seed=1103, kicks=inst.n, init=3)
    s = c.build(inst, c.reverse_tour(t), mode="la", polish=True, both_orientations=False)
    base_v = objective(inst, s.tour, s.picked)
    i, j = 195, 953
    tour = s.tour.copy()
    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
    pos = np.empty(inst.n, int)
    pos[tour] = np.arange(inst.n)
    seg = set(tour[i + 1:j + 1].tolist())
    items = np.array([k for k in range(inst.m) if inst.city[k] in seg])

    def margins(pk):
        """Second-order estimate of the gain of flipping each item of the segment (Section 3.3.2)."""
        d = inst.dist(tour, np.roll(tour, -1))
        load = np.cumsum(city_weights(inst, pk)[tour])
        v = np.maximum(inst.v_max - inst.nu * load, inst.v_min)
        G1 = np.cumsum((d / v ** 2)[::-1])[::-1]
        G2 = np.cumsum((d / v ** 3)[::-1])[::-1]
        k = pos[inst.city[items]]
        nw = inst.nu * inst.weight[items]
        return np.where(pk[items], -inst.profit[items] + inst.R * (nw * G1[k] - nw * nw * G2[k]),
                        inst.profit[items] - inst.R * (nw * G1[k] + nw * nw * G2[k]))

    ALL, DAMPED = "all items with a positive gain at once", "best quarter per pass (f=0.25)"
    plain = (objective(inst, tour, s.picked) - base_v) / 1000
    rows = [dict(repack_rule=ALL, **{"pass": 0}, gain_thousands=plain, items_dropped=0, items_added=0)]
    pk = s.picked.copy()
    for ps in range(12):                                # drops on even passes, adds on odd passes
        gn = margins(pk)
        w = inst.weight[pk].sum()
        before = pk.copy()
        if ps % 2 == 0:
            pk[items[(gn > 0) & pk[items]]] = False
        else:
            for e in np.argsort(-gn):
                if gn[e] > 0 and not pk[items[e]] and w + inst.weight[items[e]] <= inst.capacity:
                    pk[items[e]] = True
                    w += inst.weight[items[e]]
        rows.append(dict(repack_rule=ALL, **{"pass": ps + 1}, gain_thousands=(objective(inst, tour, pk) - base_v) / 1000,
                         items_dropped=int((before & ~pk).sum()), items_added=int((~before & pk).sum())))
    pk = s.picked.copy()
    rows.append(dict(repack_rule=DAMPED, **{"pass": 0}, gain_thousands=plain, items_dropped=0, items_added=0))
    for ps in range(60):
        gn = margins(pk)
        w = inst.weight[pk].sum()
        feas = pk[items] | (w + inst.weight[items] <= inst.capacity)
        idx = np.flatnonzero((gn > 1e-9) & feas)
        if len(idx) == 0:
            break
        before = pk.copy()
        for e in idx[np.argsort(-gn[idx])][:max(1, int(0.25 * len(idx)))]:
            it = items[e]
            if pk[it]:
                pk[it] = False
            elif inst.weight[pk].sum() + inst.weight[it] <= inst.capacity:
                pk[it] = True
        rows.append(dict(repack_rule=DAMPED, **{"pass": ps + 1}, gain_thousands=(objective(inst, tour, pk) - base_v) / 1000,
                         items_dropped=int((before & ~pk).sum()), items_added=int((~before & pk).sum())))
    return {"repack_passes": pd.DataFrame(rows)}


def light_heavy():
    name = "ch130_n645_uncorr-similar-weights_05"
    inst = benchmark.load_instance(name)
    nbr = knn_lists(inst, 10)
    s0 = c.build(inst, tours.get_tour(name, inst), mode="la", polish=True)
    st = State(inst, s0.tour, s0.picked, nbr)
    local_search(st, time_limit=60)                     # starting solution: constructor and local search (converges)
    cw = city_weights(inst, st.picked)
    nz = cw[1:][cw[1:] > 0]
    best, q, thr = None, None, None
    for qq in (0.0, 0.25, 0.5, 0.75):                   # the four thresholds of LARC; the best result is kept
        for tr in tours.split_tours(inst, cw, qq, seed=1):
            for pk in (st.picked, c.build(inst, tr, mode="la", polish=True, both_orientations=False).picked):
                s = State(inst, tr, pk, nbr)
                local_search(s, time_limit=60)
                if best is None or s.value() > best.value():
                    best, q, thr = s, qq, float(np.quantile(nz, qq)) if qq > 0 else 0.0
    heavy = cw > thr
    idx = np.arange(inst.n)
    cities = pd.concat([pd.DataFrame(dict(x=inst.xy[m, 0], y=inst.xy[m, 1], city=lab))
                        for lab, m in (("heavy", heavy & (idx > 0)), ("light", ~heavy & (idx > 0)), ("depot", idx == 0))])
    tours_, prof = [], []
    for lab, s in (("starting", st), ("after light-heavy move", best)):
        t = np.concatenate((s.tour, [s.tour[0]]))
        tours_.append(pd.DataFrame(dict(x=inst.xy[t, 0], y=inst.xy[t, 1], solution=lab)))
        d = inst.dist(s.tour, np.roll(s.tour, -1))
        load = np.cumsum(city_weights(inst, s.picked)[s.tour])
        prof.append(pd.DataFrame(dict(frac=np.concatenate(([0.0], np.cumsum(d)))[:-1] / d.sum(), load=load / inst.capacity,
                                      factor=1.0 / np.maximum(inst.v_max - inst.nu * load, inst.v_min), solution=lab)))
    cocop = benchmark.published_values().loc[name, "CoCo+"]
    gap = lambda v: 100 * (v - cocop) / abs(cocop)
    info = pd.DataFrame([("instance", name),
                         ("incumbent", f"{st.value():.1f} gap {gap(st.value()):.2f} len {inst.tour_length(st.tour):.0f} "
                                       f"items {st.picked.sum()}"),
                         ("lhr", f"{best.value():.1f} gap {gap(best.value()):.2f} len {inst.tour_length(best.tour):.0f} "
                                 f"items {best.picked.sum()} q {q} thr {thr:.0f} heavy {heavy.sum()} light {(cw[1:] <= thr).sum()}")],
                        columns=["item", "value"])
    return {"light_heavy_tours": pd.concat(tours_), "light_heavy_load": pd.concat(prof), "light_heavy_cities": cities,
            "light_heavy_info": info}


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    sheets = {}
    if what in ("repack_passes", "all"):
        sheets.update(repack_passes())
    if what in ("light_heavy", "all"):
        sheets.update(light_heavy())
    write_xlsx(RUNS_DIR / "figure_examples.xlsx", sheets)
    for k, v in sheets.items():
        print(k, len(v), "rows")
    print("->", RUNS_DIR / "figure_examples.xlsx")
