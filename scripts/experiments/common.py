"""Helpers shared by the experiment scripts: import path of the larc package, instance selection, seeds, and the Excel
output of new runs (written to runs/, see larc.RUNS_DIR). Each output workbook is rewritten after every finished job,
so an interrupted experiment can be started again with the same command: the jobs already in the workbook are skipped."""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from larc import RUNS_DIR, benchmark  # noqa: E402,F401


def select_instances(list_name: str = "", names: str = "", largest_first: bool = True):
    """Names of the instances to run: a list of the sheet experiment_lists of data/benchmark_instances.xlsx (list_name),
    the 60 held-out instances (list_name "heldout"), or names separated by commas, or all 60 categorized instances.
    Largest first by default, so that long jobs start early."""
    if list_name == "heldout":
        inst = pd.DataFrame(benchmark.heldout_list())
    else:
        inst = pd.DataFrame(benchmark.instance_list())
    if list_name and list_name != "heldout":
        lists = pd.read_excel(benchmark.INSTANCE_LIST, sheet_name="experiment_lists")
        keep = lists[lists["list"] == list_name].instance
        if keep.empty:
            sys.exit(f"unknown list {list_name}: {sorted(lists['list'].unique())}")
        inst = inst[inst.instance.isin(keep)]
    if names:
        inst = inst[inst.instance.isin(names.split(","))]
    if largest_first:
        inst = inst.sort_values("n_items", ascending=False)
    return list(inst.instance)


def seeds(text: str):
    """'1-10' -> 1, ..., 10; '3' -> 3."""
    lo, hi = (int(x) for x in text.split("-")) if "-" in text else (int(text), int(text))
    return list(range(lo, hi + 1))


def write_xlsx(path: Path, sheets: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name, index=False)


def run_jobs(fn, jobs, out: Path, keys, workers: int = 8, label=lambda r: "", maxtasks: int = 10):
    """Runs fn(job) for every job not yet in the sheet 'runs' of out (keys: the columns that identify a job, in the
    order of the job tuple). fn returns one record (a dict); a record may carry a list of rows under '_trace', which
    are written to a second sheet 'traces'."""
    runs, traces = [], []
    if out.exists():
        old = pd.read_excel(out, sheet_name=None)
        runs, traces = old["runs"].to_dict("records"), old.get("traces", pd.DataFrame()).to_dict("records")
    seen = {tuple(r[k] for k in keys) for r in runs}
    todo = [j for j in jobs if tuple(j[:len(keys)]) not in seen]
    print(f"{len(todo)} jobs to run ({len(jobs) - len(todo)} already in {out.name})", flush=True)
    t0 = time.monotonic()
    with Pool(workers, maxtasksperchild=maxtasks) as p:
        for k, r in enumerate(p.imap_unordered(fn, todo), 1):
            traces += r.pop("_trace", [])
            runs.append(r)
            sheets = {"runs": pd.DataFrame(runs)}
            if traces:
                sheets["traces"] = pd.DataFrame(traces)
            write_xlsx(out, sheets)
            print(f"[{k}/{len(todo)} {time.monotonic() - t0:.0f} s] {label(r)}", flush=True)
    return out
