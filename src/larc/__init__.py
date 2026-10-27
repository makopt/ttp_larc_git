"""LARC (Load-Aware Racing with Coordinated moves) for the single-objective Traveling Thief Problem (TTP1).

All paths are relative to the repository root. New runs write to <repo>/runs (or to the folder given by the environment
variable LARC_RUNS_DIR); the Numba compilation cache is kept there as well. The results of the paper are in
<repo>/results."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = Path(os.environ.get("LARC_RUNS_DIR", REPO_ROOT / "runs"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(RUNS_DIR / "numba_cache"))   # must be set before numba is imported

DATA_DIR = REPO_ROOT / "data"
INSTANCES_DIR = DATA_DIR / "instances"                 # JSON instances, created by scripts/setup/prepare_instances.py
INSTANCE_LIST = DATA_DIR / "benchmark_instances.xlsx"  # the 60 instances of the categorized benchmark
PUBLISHED_VALUES = DATA_DIR / "published_values.xlsx"  # values of the nine published methods
RESULTS_DIR = REPO_ROOT / "results"                    # results of the paper (Excel workbooks)
