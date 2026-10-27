"""Checks of the coordinated local search and of LARC (run after scripts/setup/prepare_instances.py and
scripts/setup/setup_external.sh): python3 -m pytest -q tests, or python3 tests/test_local_search.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from larc import benchmark, construct, tours  # noqa: E402
from larc.evaluator import is_feasible, objective  # noqa: E402
from larc.search import State, check, knn_lists, local_search  # noqa: E402


def test_local_search_is_exact_and_monotone():
    for name in ("eil76_n375_uncorr-similar-weights_05", "kroA100_n99_bounded-strongly-corr_01"):
        inst = benchmark.load_instance(name)
        s0 = construct.build(inst, tours.get_tour(name, inst), mode="la", polish=True)
        st = State(inst, s0.tour, s0.picked, knn_lists(inst, 10))
        v0 = st.value()
        local_search(st, time_limit=10)
        assert st.value() >= v0 - 1e-9
        assert check(st) < 1e-6                      # the incremental values equal the evaluator
        assert st.wtot <= inst.capacity + 1e-9


def test_state_does_not_alias_inputs():
    name = "a280_n1395_uncorr-similar-weights_05"
    inst = benchmark.load_instance(name)
    s0 = construct.build(inst, tours.get_tour(name, inst), mode="la", polish=True)
    tour0, pk0 = s0.tour.copy(), s0.picked.copy()
    st = State(inst, s0.tour, s0.picked, knn_lists(inst, 10))
    local_search(st, time_limit=5)
    assert np.array_equal(s0.tour, tour0) and np.array_equal(s0.picked, pk0)


def test_short_run_of_larc_is_feasible():
    from larc.larc import solve_larc
    inst = benchmark.load_instance("eil76_n75_bounded-strongly-corr_01")
    st, info = solve_larc(inst, 20.0, seed=1)
    assert is_feasible(inst, st.tour, st.picked)
    assert abs(objective(inst, st.tour, st.picked) - st.value()) < 1e-6
    assert st.tour[0] == 0 and sorted(st.tour.tolist()) == list(range(inst.n))


if __name__ == "__main__":
    test_local_search_is_exact_and_monotone()
    test_state_does_not_alias_inputs()
    test_short_run_of_larc_is_feasible()
    print("ok")
