#!/usr/bin/env bash
# Sensitivity runs of the paper, one constant changed at a time, 600 s per run, only on the instances that each value
# can affect (lists in the sheet experiment_lists of data/benchmark_instances.xlsx).
#   Section 5.5, sampling shares (seeds 1 to 10): share_tours = 0.45 in LARC (sampling end on instances with more than
#     1,500 cities), share_max = 0.8 in LARC (latest sampling end on instances with at most 1,500 cities).
#   Section 5.6, annealing share (seeds 1 to 5): anneal_frac = 0.04 in LARC; the runs with 0.04 are repeated in the same
#     batch as a reference.
# Output: runs/sensitivity_*.xlsx. Results of the paper: results/sensitivity_sampling_shares.xlsx and the sheet
# annealing_runs of results/sensitivity_other_constants.xlsx.
set -euo pipefail
cd "$(dirname "$0")"
run() { python3 run_larc_benchmark.py 600 "$1" --workers 8 --seeds "$2" --list "$3" ${4:+--param "$4"}; }
run sensitivity_sampling_latest_0.45T 1-10 sampling_latest_0.45T share_max=0.45
run sensitivity_sampling_latest_0.6T  1-10 sampling_latest_0.6T  share_max=0.6
run sensitivity_sampling_latest_0.9T  1-10 sampling_latest_0.9T  share_max=0.9
run sensitivity_sampling_end_0.30T    1-10 sampling_end_0.30T    share_tours=0.30
run sensitivity_sampling_end_0.60T    1-10 sampling_end_0.60T    share_tours=0.60
run sensitivity_annealing_0.02        1-5  annealing_share       anneal_frac=0.02
run sensitivity_annealing_0.08        1-5  annealing_share       anneal_frac=0.08
run sensitivity_annealing_0.04_repeat 1-5  annealing_share       ""
