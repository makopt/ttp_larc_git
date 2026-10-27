"""Access to the 60 instances of the categorized benchmark and to the published values of the nine methods."""
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from . import INSTANCE_LIST, PUBLISHED_VALUES, REPO_ROOT

METHODS = ["CoCo+", "SAVI", "RWS", "CoCo", "CS2SA*", "CS2SA", "MA2B", "S5", "MATLS"]


@lru_cache(maxsize=1)
def _instances() -> pd.DataFrame:
    return pd.read_excel(INSTANCE_LIST, sheet_name="instances")


def instance_list() -> List[Dict]:
    """One dictionary per instance: name, category, TSP base, numbers of cities and items, path of the JSON file."""
    return _instances().to_dict("records")


def instance_names() -> List[str]:
    return list(_instances().instance)


def load_instance(name: str):
    """The instance as arrays (class LeanTTP of larc.instance), from its JSON file."""
    from .instance import load_lean
    row = _instances().set_index("instance").loc[name]
    return load_lean(REPO_ROOT / row["path"])


def published_values() -> pd.DataFrame:
    """Published objective values of the nine methods: index = instance, columns = methods (sources in the workbook)."""
    return pd.read_excel(PUBLISHED_VALUES, sheet_name="published_values", index_col="instance")[METHODS]


def published_cocop_means() -> pd.Series:
    """Mean of CoCoP over ten runs on each instance, Tables 6 to 8 of Namazi et al. (2023)."""
    return pd.read_excel(PUBLISHED_VALUES, sheet_name="CoCoP_means", index_col="instance")["CoCoP_mean"]
