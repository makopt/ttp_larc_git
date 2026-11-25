#!/usr/bin/env bash
# Ablation of the main components (Section 5.5 and Table 11): LARC and five variants, each without one component, ten runs of
# 600 s per instance on the 60 categorized instances (3,600 runs). The switches are those of src/larc/ablation.py.
# LARC is run in the same batch as the variants, one block of seeds after the other, so that every variant is compared
# with runs of LARC made under the same machine conditions.
# Output: runs/ablation_<configuration>.xlsx. Results of the paper: results/ablation_components.xlsx (sheet runs).
set -euo pipefail
cd "$(dirname "$0")"
base() { python3 run_larc_benchmark.py 600 ablation_LARC --workers 8 --seeds "$1"; }
variant() { env "$1=0" python3 run_larc_benchmark.py 600 "ablation_$2" --workers 8 --seeds "$3"; }
for B in 1-3 4-5 6-10; do
  base $B
  variant LARC_SECOND no_second_order      $B
  variant LARC_COORD  no_coordinated_2opt  $B
  variant LARC_SCREEN no_first_pass_screen $B
  variant LARC_LHR    no_light_heavy       $B
  variant LARC_RACE   no_racing            $B
done
