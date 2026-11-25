#!/usr/bin/env bash
# Two-factor grid (Section 5.8): the sampling end of the large instances (share_tours = 0.30, 0.45, 0.60) crossed
# with the damping of the coordinated 2-opt (LARC_COORD_FRAC = 0.1, 0.25, 0.5), on 12 instances with more than 1,500
# cities, four per category, five runs of 600 s each (540 runs). The setting of LARC (0.45, 0.25) is in the same batch.
# Output: runs/grid_s<share_tours>_f<damping>.xlsx. Results of the paper: results/two_factor_grid.xlsx (sheet runs).
set -euo pipefail
cd "$(dirname "$0")"
GRID="d2103_n2102_bounded-strongly-corr_01,fnl4461_n4460_bounded-strongly-corr_01,rl11849_n11848_bounded-strongly-corr_01,\
d18512_n18511_bounded-strongly-corr_01,d2103_n10510_uncorr-similar-weights_05,pcb3038_n15185_uncorr-similar-weights_05,\
usa13509_n67540_uncorr-similar-weights_05,pla33810_n169045_uncorr-similar-weights_05,fl1577_n15760_uncorr_10,\
fnl4461_n44600_uncorr_10,brd14051_n140500_uncorr_10,d15112_n151110_uncorr_10"
for S in 0.45 0.30 0.60; do
  for F in 0.25 0.1 0.5; do
    LARC_COORD_FRAC=$F python3 run_larc_benchmark.py 600 "grid_s${S}_f${F}" --workers 8 --seeds 1-5 --names "$GRID" \
      --param share_tours=$S
  done
done
