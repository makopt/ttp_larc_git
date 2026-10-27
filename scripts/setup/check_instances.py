"""Checks that the JSON instances read by LARC and the original .ttp files read by CoCo+ hold the same data: header
values, city coordinates, and profit, weight and city of every item.
Usage: python3 scripts/setup/check_instances.py   (after scripts/setup/prepare_instances.py)"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import benchmark  # noqa: E402


def read_ttp(path):
    head, xy, items, sec = {}, [], [], None
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("NODE_COORD_SECTION"):
            sec = "n"
            continue
        if s.startswith("ITEMS SECTION"):
            sec = "i"
            continue
        if sec is None:
            k, v = s.split(":", 1)
            head[k.strip()] = v.strip()
        elif sec == "n":
            xy.append([float(t) for t in s.split()[1:3]])
        else:
            items.append([float(t) for t in s.split()[:4]])
    return head, np.array(xy), np.array(items)


bad = 0
rows = benchmark.instance_list()
for r in rows:
    inst = benchmark.load_instance(r["instance"])
    head, xy, it = read_ttp(REPO / "data" / r["ttp_folder"] / (r["instance"] + ".ttp"))
    checks = {
        "n": int(head["DIMENSION"]) == inst.n,
        "m": int(head["NUMBER OF ITEMS"]) == inst.m,
        "capacity": float(head["CAPACITY OF KNAPSACK"]) == inst.capacity,
        "v_min": float(head["MIN SPEED"]) == inst.v_min,
        "v_max": float(head["MAX SPEED"]) == inst.v_max,
        "R": float(head["RENTING RATIO"]) == inst.R,
        "edge": head["EDGE_WEIGHT_TYPE"] == inst.edge_type,
        "xy": xy.shape == inst.xy.shape and np.array_equal(xy, inst.xy),
        "item_index": np.array_equal(it[:, 0], np.arange(1, inst.m + 1)),
        "profit": np.array_equal(it[:, 1], inst.profit),
        "weight": np.array_equal(it[:, 2], inst.weight),
        "city": np.array_equal(it[:, 3] - 1, inst.city),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        bad += 1
        print("DIFFERENT", r["instance"], failed)
print(f"{len(rows) - bad} of {len(rows)} instances identical in both formats")
