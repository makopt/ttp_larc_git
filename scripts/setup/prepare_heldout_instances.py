"""Unpacks the 60 held-out benchmark files (data/ttp_instances_heldout_60.tar.xz -> data/heldout/<base>-ttp/*.ttp),
checks their sha256 against data/ttp_sha256_heldout.txt, and converts each to the JSON format read by LARC
(data/instances_heldout/<base>-ttp/*.json). The list of these instances, and the categorized instance each one was drawn
for, is in the sheet "instances" of results/heldout_LARC_vs_CoCoPlus.xlsx; scripts/setup/select_heldout_instances.py
repeats the draw from a copy of the full benchmark.
Usage: python3 scripts/setup/prepare_heldout_instances.py"""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_instances import ttp_to_dict  # noqa: E402


def main():
    with tarfile.open(DATA / "ttp_instances_heldout_60.tar.xz") as tf:
        tf.extractall(ROOT, filter="data")          # creates data/heldout/<base>-ttp/*.ttp
    folder = DATA / "heldout"
    sums = dict(l.split()[::-1] for l in (DATA / "ttp_sha256_heldout.txt").read_text().splitlines() if l.strip())
    n = 0
    for ttp in sorted(folder.glob("*-ttp/*.ttp")):
        if ttp.name.startswith("._"):                 # macOS metadata, not an instance
            continue
        rel = str(ttp.relative_to(folder))
        if sums.get(rel) != hashlib.sha256(ttp.read_bytes()).hexdigest():
            sys.exit(f"checksum mismatch: {rel}")
        out = DATA / "instances_heldout" / ttp.parent.name / (ttp.stem + ".json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ttp_to_dict(ttp)))
        n += 1
    if n != 60:
        sys.exit(f"expected 60 held-out instances, found {n}")
    print(f"{n} held-out instances checked (sha256) and converted to {DATA / 'instances_heldout'}")


if __name__ == "__main__":
    main()
