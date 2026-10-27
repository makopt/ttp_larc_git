"""Exact TTP1 objective (Polyakovskiy et al. 2014) on LeanTTP.

Z = sum(profit of picked items) - R * total travel time, where the speed on the edge leaving a city is
v(c) = v_max - nu * c, c = weight carried after picking at that city (depot weight 0), return leg to the depot included.
The tour is an array of n distinct 0-based cities starting at the depot 0 (closing edge implicit).
"""
import numpy as np

from .instance import LeanTTP


def city_weights(inst: LeanTTP, picked: np.ndarray) -> np.ndarray:
    """Weight picked at each city; `picked` is a bool mask over items."""
    return np.bincount(inst.city[picked], weights=inst.weight[picked], minlength=inst.n)


def travel_time(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray) -> float:
    tour = np.asarray(tour)
    load = np.cumsum(city_weights(inst, picked)[tour])          # weight on the edge leaving tour[k]
    speed = np.maximum(inst.v_max - inst.nu * load, inst.v_min)
    return float((inst.dist(tour, np.roll(tour, -1)) / speed).sum())


def objective(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray) -> float:
    return float(inst.profit[picked].sum()) - inst.R * travel_time(inst, tour, picked)


def is_feasible(inst: LeanTTP, tour: np.ndarray, picked: np.ndarray) -> bool:
    tour = np.asarray(tour)
    return (tour[0] == 0 and len(tour) == inst.n and len(np.unique(tour)) == inst.n
            and inst.weight[picked].sum() <= inst.capacity)
