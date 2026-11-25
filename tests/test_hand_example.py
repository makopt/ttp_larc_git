"""The example of docs/hand_example.md, checked with the code of LARC.

Every number of the document is recomputed twice: by plain arithmetic from the formulas of the paper (Equations 1 to 3
and Proposition 1), and by the code of LARC (evaluator, load-aware item keys, local search). The script prints the
numbers of the document and stops at the first one that does not agree.
Usage: python3 tests/test_hand_example.py   (or python3 -m pytest -q tests)"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from larc import construct  # noqa: E402
from larc.evaluator import objective  # noqa: E402
from larc.instance import LeanTTP  # noqa: E402
from larc.search import State, knn_lists, local_search  # noqa: E402

A, B, C, D = 0, 1, 2, 3
NAME = "ABCD"
VMAX, VMIN, W, R = 1.0, 0.1, 9.0, 1.0
NU = (VMAX - VMIN) / W                                   # 0.1
XY = {A: (0, 0), B: (16, 12), C: (40, 30), D: (20, 15)}   # all on the line y = 0.75 x: integer distances
ITEMS = {B: (1.0, 5.0), C: (2.0, 16.0), D: (4.0, 60.0)}   # city: (weight, profit), one item per city


def instance() -> LeanTTP:
    return LeanTTP(name="hand_example", n=4, m=3, capacity=W, v_min=VMIN, v_max=VMAX, R=R, edge_type="CEIL_2D",
                   xy=np.array([XY[c] for c in range(4)], float),
                   profit=np.array([ITEMS[c][1] for c in (B, C, D)]), weight=np.array([ITEMS[c][0] for c in (B, C, D)]),
                   city=np.array([B, C, D]))


def dist(a, b) -> float:
    return float(np.ceil(np.hypot(XY[a][0] - XY[b][0], XY[a][1] - XY[b][1])))


def edges(tour, picked):
    """Per edge s (leaving position s): length d_s, load c_s, speed v(c_s) = vmax - nu c_s (Equations 1 and 2)."""
    rows, load = [], 0.0
    for s, a in enumerate(tour):
        load += ITEMS[a][0] if a in picked else 0.0
        b = tour[(s + 1) % len(tour)]
        rows.append((dist(a, b), load, VMAX - NU * load))
    return rows


def travel_time(tour, picked) -> float:
    return sum(d / v for d, _, v in edges(tour, picked))


def value(tour, picked) -> float:                         # Equation 3
    return sum(ITEMS[c][1] for c in picked) - R * travel_time(tour, picked)


def suffix(tour, picked, power):                          # D (power 0), G1 (2), G2 (3), Equation 6
    terms = [d / v ** power for d, _, v in edges(tour, picked)]
    return [sum(terms[k:]) for k in range(len(terms))] + [0.0]


def approx(a, b, tol=5e-4):
    return abs(a - b) <= tol


def show(label, got, expected, larc=None):
    ok = approx(got, expected) and (larc is None or approx(larc, expected, 1e-6 * max(1, abs(expected)) + 5e-4))
    extra = "" if larc is None else f"   code of LARC: {larc:.3f}"
    print(f"{'ok  ' if ok else 'FAIL'} {label:<62s} {got:10.3f}   document: {expected:.3f}{extra}")
    assert ok, label


def main():
    inst = instance()
    mask = lambda picked: np.array([c in picked for c in (B, C, D)])

    print("1. Objective of the solution of Figure 1: tour (A, D, C, B, A), items of D and C picked")
    t0, p0 = (A, D, C, B), {C, D}
    for (d, c, v), want in zip(edges(t0, p0), [(25, 0, 1.0), (25, 4, 0.6), (30, 6, 0.4), (20, 6, 0.4)]):
        assert (d, c) == want[:2] and approx(v, want[2])
    show("travel time T", travel_time(t0, p0), 575 / 3)
    show("objective Z = 76 - T", value(t0, p0), 76 - 575 / 3, objective(inst, np.array(t0), mask(p0)))

    print("\n2. Suffix sums of Figure 1")
    g1, g2 = suffix(t0, p0, 2), suffix(t0, p0, 3)
    for k, (e1, e2) in enumerate([(406.944, 921.991), (381.944, 896.991), (312.5, 781.25), (125.0, 312.5)]):
        show(f"G1({k})", g1[k], e1)
        show(f"G2({k})", g2[k], e2)

    print("\n3. Proposition 1(a): adding the item of B (k = 3, w = 1)")
    w = 1.0
    exact = travel_time(t0, p0 | {B}) - travel_time(t0, p0)
    est = NU * w * g1[3] + NU ** 2 * w ** 2 * g2[3]
    rho = NU * w / 0.4
    show("exact time added", exact, 50 / 3)
    show("estimate nu w G1(3) + nu^2 w^2 G2(3)", est, 15.625)
    show("remainder (exact - estimate), nonnegative", exact - est, 50 / 3 - 15.625)
    show("bound rho/(1-rho) nu^2 w^2 G2(3), rho = 0.25", rho / (1 - rho) * NU ** 2 * w ** 2 * g2[3], 3.125 / 3)
    key = construct.item_keys(inst, np.array(t0), "la", mask(p0))[0]
    show("load-aware density of the item: 5 - R x estimate", 5 - R * est, -10.625, key)
    show("exact value of picking it: 5 - R x exact", 5 - R * exact, 5 - 50 / 3)

    print("\n3. Proposition 1(b): removing the item of C (k = 2, w = 2)")
    w = 2.0
    exact = travel_time(t0, p0) - travel_time(t0, p0 - {C})
    est = NU * w * g1[2] - NU ** 2 * w ** 2 * g2[2]
    rho = NU * w / 0.4
    show("exact time saved", exact, 125 / 3)
    show("estimate nu w G1(2) - nu^2 w^2 G2(2)", est, 31.25)
    show("remainder (exact - estimate), nonnegative", exact - est, 125 / 3 - 31.25)
    show("bound rho nu^2 w^2 G2(2), rho = 0.5", rho * NU ** 2 * w ** 2 * g2[2], 15.625)

    print("\n4. One coordinated 2-opt move: reversal of (D, C), then repacking of the items of D and C")
    t1 = (A, C, D, B)
    show("tour length before the move", sum(d for d, _, _ in edges(t0, p0)), 100)
    show("tour length after the move", sum(d for d, _, _ in edges(t1, p0)), 100)
    show("plain reversal, items fixed: objective", value(t1, p0), 76 - 143.75, objective(inst, np.array(t1), mask(p0)))
    show("plain reversal: gain", value(t1, p0) - value(t0, p0), 575 / 3 - 143.75)
    h1, h2 = suffix(t1, p0, 2), suffix(t1, p0, 3)
    show("new tour: G1(1)", h1[1], 195.3125)
    show("new tour: G2(1)", h2[1], 439.453)
    show("new tour: G1(2)", h1[2], 156.25)
    show("new tour: G2(2)", h2[2], 390.625)
    est_c = NU * 2 * h1[1] - NU ** 2 * 4 * h2[1] - 16
    est_d = NU * 4 * h1[2] - NU ** 2 * 16 * h2[2] - 60
    show("estimated gain of dropping the item of C (flipped)", est_c, 5.484)
    show("estimated gain of dropping the item of D (kept)", est_d, -60)
    show("exact gain of dropping the item of C", value(t1, {D}) - value(t1, p0), 11.083)
    show("exact gain of dropping the item of D", value(t1, {C}) - value(t1, p0), -28.75)
    show("coordinated move: objective", value(t1, {D}), 60 - 350 / 3, objective(inst, np.array(t1), mask({D})))
    show("coordinated move: gain", value(t1, {D}) - value(t0, p0), 59.0)

    print("\n5. The optimum of the instance, by enumeration of the 6 tours x 8 packings")
    best = max((value((A,) + q, set(s)), q, s) for q in itertools.permutations((B, C, D))
               for k in range(4) for s in itertools.combinations((B, C, D), k))
    show("optimum", best[0], 60 - 350 / 3)
    print(f"     reached by the tour ({', '.join(NAME[c] for c in (A,) + best[1])}, A) "
          f"with the items of {', '.join(NAME[c] for c in best[2])}")
    assert best[1] == (C, D, B) and best[2] == (D,)
    st = State(inst, np.array(t0), mask(p0), knn_lists(inst, 3))
    local_search(st, time_limit=5, max_seg=4, coord=True)
    show("local search of LARC with the coordinated 2-opt, from section 1", st.value(), 60 - 350 / 3)
    print("\nok: every number of docs/hand_example.md agrees")


def test_hand_example():
    main()


if __name__ == "__main__":
    main()
