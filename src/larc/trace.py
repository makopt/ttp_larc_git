"""Optional convergence trace of a LARC run: (elapsed time, best objective value found so far) at every improvement,
plus phase markers. Off by default; it only records values and never influences a decision of the search.
States are evaluated at most once every PERIOD seconds, so the overhead is negligible."""
import time

PERIOD = 1.0
_on = False
_t0 = 0.0
_best = float("-inf")
_last = 0.0
_rows = []
_phases = []


def start():
    global _on, _t0, _best, _last, _rows, _phases
    _on, _t0, _best, _last, _rows, _phases = True, time.monotonic(), float("-inf"), 0.0, [], []


def observe(value: float):
    """Record a value known to be the objective of a feasible solution."""
    global _best
    if _on and value > _best + 1e-9:
        _best = value
        _rows.append((time.monotonic() - _t0, float(value)))


def observe_state(st, force: bool = False):
    """Record the value of a search state, at most once every PERIOD seconds unless force."""
    global _last
    if not _on:
        return
    now = time.monotonic()
    if force or now - _last >= PERIOD:
        _last = now
        observe(st.value())


def phase(name: str):
    if _on:
        _phases.append((name, time.monotonic() - _t0))


def stop():
    global _on
    _on = False
    return list(_rows), list(_phases)
