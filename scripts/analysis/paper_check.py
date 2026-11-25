"""Comparison of a recomputed number with the value printed in the paper, used by the analysis scripts.

check(label, value, paper) prints one line:  <status>  <label>  <recomputed value>  | paper: <printed value>
with the status
  ok    every number agrees with the printed one at the precision of the paper (the printed number with d decimals
        agrees when the recomputed one rounds to it, e.g. 0.6213 agrees with 0.62 and 9e-05 with 0.0001)
  DIFF  at least one number differs
  --    not compared: the paper gives a description (e.g. "see Table 7") or a whole table, printed for reading;
        inside a tuple, None marks a number that the paper does not give
A value and a printed value may be numbers, strings, booleans, or tuples and lists of them, compared element by element.
A printed number may also be given as a string, to keep its trailing zeros ("0.90" has two decimals), or as a bound
("<0.001", ">=0.31").
summary() prints the counts; scripts/analysis/check_paper_numbers.py collects the lines of all scripts."""
from decimal import Decimal

import numpy as np

COUNTS = {"ok": 0, "DIFF": 0, "--": 0}


def plain(v):
    """Python numbers instead of NumPy scalars, for printing and comparing."""
    if isinstance(v, (tuple, list)):
        return type(v)(plain(x) for x in v)
    return v.item() if isinstance(v, np.generic) else v


def _decimals(x) -> int:
    if isinstance(x, str):
        return max(0, -Decimal(x).as_tuple().exponent)
    e = Decimal(repr(float(x))).normalize().as_tuple().exponent if isinstance(x, float) else 0
    return max(0, -e)


def _number(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _agree(v, p) -> bool:
    if p is None:                                   # inside a tuple: a number the paper does not give
        return True
    if isinstance(p, (tuple, list)):
        return isinstance(v, (tuple, list)) and len(v) == len(p) and all(_agree(a, b) for a, b in zip(v, p))
    if isinstance(p, bool) or isinstance(v, bool):
        return v == p
    if isinstance(p, str) and _numeric(v):
        for op, f in (("<=", float.__le__), (">=", float.__ge__), ("<", float.__lt__), (">", float.__gt__)):
            if p.startswith(op) and _number(p[len(op):]) is not None:
                return f(float(v), _number(p[len(op):]))
        if _number(p) is None:
            return False
        return abs(v - _number(p)) <= 0.5 * 10 ** -_decimals(p) + 1e-9
    if isinstance(p, str):
        return isinstance(v, str) and v == p
    if _numeric(v) and _numeric(p):
        return not np.isnan(v) and abs(v - p) <= 0.5 * 10 ** -_decimals(p) + 1e-9
    return False


def _comparable(p: str) -> bool:
    return _number(p.lstrip("<>=")) is not None


def status(value, paper) -> str:
    value, paper = plain(value), plain(paper)
    if paper is None:
        return "--"
    if isinstance(paper, str) and not _comparable(paper):     # a description, or a single string to match exactly
        return "ok" if value == paper else "--"
    return "ok" if _agree(value, paper) else "DIFF"


def _short(v):
    """Rounded copy of a value, for printing only (the comparison uses the full value)."""
    if isinstance(v, (tuple, list)):
        return type(v)(_short(x) for x in v)
    if isinstance(v, float) and np.isfinite(v):
        return float(f"{v:.4g}") if abs(v) < 1 else round(v, 3)
    return v


def check(label, value, paper, width: int = 78):
    s = status(value, paper)
    COUNTS[s] += 1
    print(f"{s:<5}{label:<{width}} {_short(plain(value))}   | paper: {paper}")


def summary():
    print(f"\n{COUNTS['ok']} numbers agree with the paper, {COUNTS['DIFF']} differ, {COUNTS['--']} printed for reading")
