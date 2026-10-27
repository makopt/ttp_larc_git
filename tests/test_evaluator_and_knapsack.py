"""Checks of the evaluator and of the knapsack repack (run after scripts/setup/prepare_instances.py):
python3 -m pytest -q tests, or python3 tests/test_evaluator_and_knapsack.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from larc import benchmark, construct  # noqa: E402
from larc.evaluator import objective  # noqa: E402
from larc.kernels import kp_dp  # noqa: E402


def test_empty_packing_costs_the_tour():
    inst = benchmark.load_instance("eil76_n75_bounded-strongly-corr_01")
    tour = np.arange(inst.n)
    z = objective(inst, tour, np.zeros(inst.m, bool))
    assert abs(z + inst.R * inst.tour_length(tour) / inst.v_max) < 1e-6


def test_knapsack_repack_is_optimal_with_zero_rent():
    """dsj1000 with one item per city has R = 0: a pure 0-1 knapsack whose optimum is 144,426."""
    inst = benchmark.load_instance("dsj1000_n999_bounded-strongly-corr_01")
    assert inst.R == 0
    x = kp_dp(inst.weight.astype(np.int64), inst.profit.astype(np.float64), int(inst.capacity))
    assert inst.profit[x].sum() == 144426
    pk, val = construct.kp_repack(inst, np.arange(inst.n), np.zeros(inst.m, bool))
    assert abs(val - 144426) < 1e-6


if __name__ == "__main__":
    test_empty_packing_costs_the_tour()
    test_knapsack_repack_is_optimal_with_zero_rent()
    print("ok")
