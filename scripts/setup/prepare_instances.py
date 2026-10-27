"""Unpacks the 60 original benchmark files (data/ttp_instances_60.tar.xz -> data/<base>-ttp/*.ttp, the layout CoCo+
expects) and converts each to the JSON format read by LARC (data/instances/<base>-ttp/*.json).
The .ttp files are those of the benchmark of Polyakovskiy et al. (2014), as distributed at
https://cs.adelaide.edu.au/~optlog/CEC2014COMP_InstancesNew/ ; data/ttp_sha256.txt lists their checksums.
Usage: python3 scripts/setup/prepare_instances.py"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def ttp_to_dict(path: Path) -> dict:
    head, nodes, items, sec = {}, [], [], None
    for line in path.read_text().splitlines():
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
            t = s.split()
            nodes.append({"index": int(t[0]), "x": float(t[1]), "y": float(t[2])})
        else:
            t = s.split()
            items.append({"index": int(t[0]), "profit": float(t[1]), "weight": int(t[2]), "assigned_node": int(t[3])})
    return {
        "problem_name": head["PROBLEM NAME"], "knapsack_data_type": head["KNAPSACK DATA TYPE"],
        "dimension": int(head["DIMENSION"]), "number_of_items": int(head["NUMBER OF ITEMS"]),
        "capacity_of_knapsack": int(head["CAPACITY OF KNAPSACK"]), "min_speed": float(head["MIN SPEED"]),
        "max_speed": float(head["MAX SPEED"]), "renting_ratio": float(head["RENTING RATIO"]),
        "edge_weight_type": head["EDGE_WEIGHT_TYPE"], "nodes": nodes, "items": items,
    }


def main():
    with tarfile.open(DATA / "ttp_instances_60.tar.xz") as tf:
        tf.extractall(ROOT, filter="data")          # creates data/<base>-ttp/*.ttp
    sums = dict(l.split()[::-1] for l in (DATA / "ttp_sha256.txt").read_text().splitlines() if l.strip())
    n = 0
    for ttp in sorted(DATA.glob("*-ttp/*.ttp")):
        if ttp.name.startswith("._"):                 # macOS metadata, not an instance
            continue
        rel = str(ttp.relative_to(DATA))
        digest = hashlib.sha256(ttp.read_bytes()).hexdigest()
        if sums.get(rel) != digest:
            sys.exit(f"checksum mismatch: {rel}")
        out = DATA / "instances" / ttp.parent.name / (ttp.stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ttp_to_dict(ttp)))
        n += 1
    print(f"{n} instances checked (sha256) and converted to {DATA / 'instances'}")


if __name__ == "__main__":
    main()
