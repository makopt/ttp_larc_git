# A small example that can be checked by hand

This example uses the four-city instance of Figure 1 of the paper. It computes the objective of a solution, the item
costs of Proposition 1, and one coordinated 2-opt move. Every number below can be checked with a calculator.
`tests/test_hand_example.py` recomputes each number from the formulas of the paper and with the code of LARC:

```bash
python3 tests/test_hand_example.py
```

## The instance

| | |
|---|---|
| Cities | A (the depot), B, C, D at (0, 0), (16, 12), (40, 30), (20, 15) |
| Distances (`CEIL_2D`) | AB = 20, AC = 50, AD = 25, BC = 30, BD = 5, CD = 25 |
| Items | one per city: B (w = 1, p = 5), C (w = 2, p = 16), D (w = 4, p = 60) |
| Knapsack and speeds | W = 9, v<sub>max</sub> = 1, v<sub>min</sub> = 0.1, so ν = (v<sub>max</sub> − v<sub>min</sub>)/W = 0.1 |
| Renting ratio | R = 1 |

The four cities lie on the line y = 0.75x, so that all distances are integers. The paper gives the edges of the tour
(25, 25, 30 and 20) and the items; the coordinates only add the two distances AC and BD, which the 2-opt move of
Section 4 uses.

Notation: d<sub>s</sub> is the length of the edge that leaves position s of the tour, c<sub>s</sub> the weight carried
on it, and v(c<sub>s</sub>) = v<sub>max</sub> − ν c<sub>s</sub> its speed.

## 1. Objective of a solution

The solution of Figure 1 is the tour (A, D, C, B, A) with the items of D and C picked.

| Edge | d<sub>s</sub> | Load c<sub>s</sub> | Speed v(c<sub>s</sub>) | Time d<sub>s</sub>/v(c<sub>s</sub>) |
|---|---|---|---|---|
| A → D | 25 | 0 | 1 | 25 |
| D → C | 25 | 4 | 0.6 | 41.667 |
| C → B | 30 | 6 | 0.4 | 75 |
| B → A | 20 | 6 | 0.4 | 50 |

The travel time is T = 191.667, the profit is 60 + 16 = 76, and the objective is Z = 76 − 1 × 191.667 = **−115.667**.
The value is negative because, with R = 1, the rent of the knapsack exceeds the profit of these items.

## 2. Suffix sums

G<sub>1</sub>(k) and G<sub>2</sub>(k) are the sums of d<sub>s</sub>/v(c<sub>s</sub>)<sup>2</sup> and
d<sub>s</sub>/v(c<sub>s</sub>)<sup>3</sup> over the edges s ≥ k:

| Position k | 0 (A) | 1 (D) | 2 (C) | 3 (B) |
|---|---|---|---|---|
| d<sub>k</sub>/v<sup>2</sup> | 25 | 69.444 | 187.5 | 125 |
| d<sub>k</sub>/v<sup>3</sup> | 25 | 115.741 | 468.75 | 312.5 |
| G<sub>1</sub>(k) | 406.944 | 381.944 | 312.5 | 125 |
| G<sub>2</sub>(k) | 921.991 | 896.991 | 781.25 | 312.5 |

Figure 1 rounds these to one decimal.

## 3. Proposition 1: cost of adding and removing an item

**Adding the item of B** (position k = 3, w = 1). It would be carried on the last edge only, whose speed would drop from
0.4 to 0.3:

- exact time added: 20/0.3 − 20/0.4 = **16.667**;
- second-order estimate: ν w G<sub>1</sub>(3) + ν<sup>2</sup> w<sup>2</sup> G<sub>2</sub>(3) = 0.1 × 125 + 0.01 × 312.5 = 12.5 + 3.125 = **15.625**;
- remainder: 16.667 − 15.625 = 1.042 ≥ 0, so the estimate does not overestimate the time added;
- bound: with x<sub>3</sub> = ν w/v(c<sub>3</sub>) = 0.1/0.4 = 0.25 = ρ, the bound ρ/(1 − ρ) × 3.125 = 1.042 equals the
  remainder, since the item is carried on a single edge.

The load-aware density of the item is 5 − 1 × 15.625 = **−10.625** (the value of `construct.item_keys` in LARC), and the
exact value of picking it is 5 − 16.667 = −11.667. Both say that the item should not be picked, while the
position-weighted density, which ignores the load, gives 5 − 0.1 × 20 = 3 (Section 3.2.1 of the paper).

**Removing the item of C** (position k = 2, w = 2). The last two edges would be travelled at speed 0.6 instead of 0.4:

- exact time saved: (30/0.4 − 30/0.6) + (20/0.4 − 20/0.6) = 25 + 16.667 = **41.667**;
- second-order estimate: ν w G<sub>1</sub>(2) − ν<sup>2</sup> w<sup>2</sup> G<sub>2</sub>(2) = 0.2 × 312.5 − 0.04 × 781.25 = 62.5 − 31.25 = **31.25**;
- remainder: 41.667 − 31.25 = 10.417 ≥ 0, so the estimate does not overestimate the time saved;
- bound: x<sub>s</sub> = 0.2/0.4 = 0.5 = ρ on both edges, and ρ × 31.25 = 15.625 ≥ 10.417.

## 4. One coordinated 2-opt move

Start from the solution of Section 1. The 2-opt move reverses the segment (D, C): the edges A → D and C → B
(25 + 30) are replaced by A → C and D → B (50 + 5), so the tour (A, C, D, B, A) has the same length, 100. Only the
order in which the items are collected changes.

**Plain reversal, items fixed.** The loads become 0, 2, 6, 6 and the times 50, 31.25, 12.5 and 50, so T = 143.75 and
Z = 76 − 143.75 = −67.75. The gain of the plain reversal is −67.75 − (−115.667) = **47.917**.

**Repacking the items of the reversed cities.** The coordinated 2-opt then re-decides the items of D and C on the new
tour, ranked by the second-order estimate. On the new tour, the terms d<sub>s</sub>/v<sup>2</sup> are 50, 39.0625,
31.25, 125 and the terms d<sub>s</sub>/v<sup>3</sup> are 50, 48.828, 78.125, 312.5, so that G<sub>1</sub>(1) = 195.313,
G<sub>2</sub>(1) = 439.453, G<sub>1</sub>(2) = 156.25 and G<sub>2</sub>(2) = 390.625.

- Dropping the item of C (k = 1, w = 2): estimated time saved 0.2 × 195.313 − 0.04 × 439.453 = 21.484, estimated gain
  21.484 − 16 = **5.484** > 0, so the item is flipped. Its exact gain is 11.083.
- Dropping the item of D (k = 2, w = 4): estimated time saved 0.4 × 156.25 − 0.16 × 390.625 = 0, estimated gain
  0 − 60 = **−60**, so the item is kept. Its exact gain is −28.75. Here x<sub>s</sub> = 0.4/0.4 = 1 on the last two
  edges, the case ρ ≥ 1 of Proposition 1 in which the estimate of the time saved is poor (0 against 31.25), but the
  decision is the same as with the exact value.

**The coordinated move** is the reversal with the item of C dropped. The loads become 0, 0, 4, 4 and the times 50, 25,
8.333 and 33.333, so T = 116.667 and Z = 60 − 116.667 = **−56.667**. Its gain, 59.0, exceeds the gain of the plain
reversal, 47.917, so the coordinated 2-opt accepts the compound move. Like every move of LARC, it is accepted on its
exact value.

## 5. Check against the optimum

With three items and three cities to order, the instance has 6 tours and 8 packings, all feasible. Their enumeration
gives the optimum −56.667, reached by the tour (A, C, D, B, A) with the item of D only: the result of the coordinated
move above. The local search of LARC with the coordinated 2-opt, started from the solution of Section 1, ends at the
same value.
