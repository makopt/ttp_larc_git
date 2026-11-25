# LARC for the Traveling Thief Problem

Code, data and results of the paper

> Khalil Chebil, Mahdi Khemakhem and Rahma Lahyani. **Load-Aware Packing, Coordinated Moves and Racing for the Traveling
> Thief Problem.** Submitted to *Mathematics* (MDPI).

LARC (Load-Aware Racing with Coordinated moves) is a heuristic for the single-objective Traveling Thief Problem (TTP1).
It samples many candidate solutions (a chained Lin-Kernighan tour and a load-aware packing), races them, recombines the
best ones, and spends the end of the time budget on a final search and a final knapsack repack. Every move is accepted
on the exact objective.

This repository holds everything needed to run LARC, to run the experiments of the paper again, and to recompute every
number and statistical figure of the paper from the saved results.

## Contents

```
src/larc/                  the code of LARC (Python 3, kernels compiled with Numba)
scripts/setup/             builds CoCo+ and linkern, unpacks and checks the instances
scripts/experiments/       runs LARC, CoCo+ and every experiment of the paper
scripts/analysis/          recomputes the numbers of the paper and draws its statistical figures
data/                      the 60 benchmark instances, the 60 held-out instances, the published values, cached tours
results/                   the results of the paper, one Excel workbook per experiment
figures/                   the statistical figures of the paper (PDF and PNG), named by their number in the paper
docs/                      a small example checkable by hand, and the corrections made to the published values
patches/coco_larc.patch    the patch applied to the public code of CoCo+ (seed argument only)
tests/                     quick checks of the evaluator, the knapsack repack, the local search and the example
```

### The code (`src/larc/`)

| Module | Content |
|---|---|
| `larc.py` | `solve_larc`, the method of the paper (Algorithm 1): sampling, race, recombination, final search, final repack |
| `larc_base.py` | `solve_larc_base`, LARC without perturbed tours, probe screening, kicks and chunked final search (Section 5.6) |
| `construct.py` | the constructor (Algorithm 2): load-aware item key, item windows, flip search, knapsack repack |
| `search.py` | the coordinated local search (Algorithms 4 and 5), the light-heavy re-sequencing move, annealing, kicks |
| `kernels.py` | the compiled kernels (Numba): 2-opt, Or-opt, item flips, coordinated 2-opt, knapsack dynamic program |
| `recombination.py` | the recombination of two solutions (partition crossover with item transplant) |
| `tours.py` | chained Lin-Kernighan tours from Concorde `linkern`, the paths of the light-heavy move, the cached tours |
| `evaluator.py` | the objective function and the feasibility check |
| `instance.py` | the instance as arrays, read from its JSON file |
| `benchmark.py` | the lists of the 60 categorized and the 60 held-out instances, and the published values |
| `trace.py`, `stats.py` | convergence trace of a run; mean ranks, RDI and critical difference |
| `ablation.py` | the switches of the ablation and of the two-factor grid (environment variables, see below) |

## Installation

Tested on macOS 26 (Apple M1 Pro, Python 3.13, Apple clang 17). Linux should work the same way; the build script only
passes an explicit host type to the configure script of Concorde on macOS.

```bash
pip install -r requirements.txt
./scripts/setup/setup_external.sh             # CoCo+ -> external/CoCo-src/coco_seeded, linkern -> bin/linkern (about 30 s)
python3 scripts/setup/prepare_instances.py    # unpacks the 60 .ttp files, checks their sha256, converts them to JSON
python3 scripts/setup/check_instances.py      # checks that the JSON and .ttp files hold the same data
python3 scripts/setup/prepare_heldout_instances.py   # the 60 held-out instances (optional): sha256 check, JSON
python3 -m pytest -q tests                    # optional; each test also runs alone, e.g. python3 tests/test_hand_example.py
```

The JSON files of all 120 instances (the 60 categorized and the 60 held-out ones) are identical to the original
versions in the dataset of the Python library [ttplib](https://pypi.org/project/ttplib/) (Alanazi and Khemakhem,
arXiv:2611.12597), which loads any instance of the benchmark by its name. For `dsj1000_n999_bounded-strongly-corr_01`,
whose renting ratio is 0.00, the library returns a corrected version by default; the results of the paper use the
original file, which `ttplib.load(name, corrected=False)` returns.

Run LARC on one instance:

```bash
python3 scripts/experiments/run_larc.py a280_n279_bounded-strongly-corr_01 --seconds 60 --seed 1
```

It prints the objective value and writes the solution to `runs/solutions/` in the format of the TTP competitions (first
line the tour, second line the picked items, both numbered from 1). The first run compiles the Numba kernels (about a
minute); later runs use the cache in `runs/numba_cache`. New runs are written to `runs/`, or to the folder given by the
environment variable `LARC_RUNS_DIR`.

## Reproducing the paper

### From the saved results (minutes)

```bash
python3 scripts/analysis/check_paper_numbers.py      # all the scripts below, with a summary (about 3 minutes)
python3 scripts/analysis/make_figures.py             # Figures 7 to 10 and 12 to 16 -> figures/
```

`check_paper_numbers.py` runs the analysis scripts of the table below. Each script recomputes numbers of the paper from
the saved results and prints each one next to the value printed in the paper, with the status `ok` when they agree at
the precision of the paper, `DIFF` when they do not, and `--` for a table printed for reading. The full outputs are
written to `runs/paper_numbers/`, and the script ends with the list of the numbers that differ. The scripts read only
`results/` and `data/`; each can also be run alone.

| Script (`scripts/analysis/`) | Paper |
|---|---|
| `reproduce_paper_numbers.py` | Tables 3, 4, 7, 10, 12 and 13; Sections 3.3.3, 3.5.1, 5.1, 5.3, 5.4 and 5.6 to 5.9 |
| `comparison_uncertainty.py` | confidence intervals of Table 7 (Section 5.1) |
| `heldout_comparison.py` | held-out instances, Section 5.2 and Table 8 |
| `ablation_components.py` | ablation of the main components, Section 5.5 and Table 11 |
| `estimate_error.py` | error of the load-aware estimates, after Proposition 1 (about 3 minutes) |
| `screen_and_exact_packing.py` | exact packing on a fixed tour (Section 3.2.3) and first-pass screen (Section 3.3.3) |
| `probe_screening_large.py` | probe screening on the 33 large instances, Section 3.5.1 and Figure 9c |
| `two_factor_grid.py` | sampling end and damping of the coordinated 2-opt, Section 5.8 |
| `gap_by_instance_features.py` | gap to CoCo+ and characteristics of the instances, Section 5.9 |
| `summarize_convergence.py` | Table 10 and the data of Figure 15, from the traces; also accepts the workbook of new runs |

`docs/hand_example.md` works through the four-city example of Figure 1 by hand: the objective of a solution, the item
costs of Proposition 1 and one coordinated 2-opt move. `tests/test_hand_example.py` checks every number of it with the
code of LARC.

The 60 held-out instances were drawn from the full benchmark of 9,720 instances by a rule fixed before any run (one
other instance with the same TSP base and number of items for each categorized instance, renting ratio 0.00 excluded,
numpy seed 20261120). With a local copy of the full benchmark, the draw can be repeated and checked against the list
used in the paper:

```bash
python3 scripts/setup/select_heldout_instances.py <folder of the full benchmark>
```

### Running the experiments again

Each run has 600 s of wall-clock time from the start of its process, and eight runs are made at a time. The results
depend on the machine, so a new run does not repeat the values of the paper exactly; the saved results are those of the
paper. Every experiment script writes an Excel workbook to `runs/` with the same columns as the matching sheet of
`results/`, writes it again after every finished run, and skips the runs already done when it is started again.

| Paper | Command | Saved result | Time (8 cores) |
|---|---|---|---|
| Sections 5.1, 5.4 (LARC, 600 runs) | `scripts/experiments/run_larc_benchmark.py 600 larc_main` | `comparison_LARC_vs_CoCoPlus.xlsx`, `convergence_LARC.xlsx` | about 13 h |
| Section 5.1 (CoCo+, 600 runs) | `scripts/experiments/run_cocoplus_benchmark.py 600 cocoplus_main` | `comparison_LARC_vs_CoCoPlus.xlsx` | about 13 h |
| Section 5.2 (LARC, 600 runs) | `scripts/experiments/run_larc_benchmark.py 600 larc_heldout --list heldout` | `heldout_LARC_vs_CoCoPlus.xlsx` | about 13 h |
| Section 5.2 (CoCo+, 600 runs) | `scripts/experiments/run_cocoplus_benchmark.py 600 cocoplus_heldout --list heldout` | `heldout_LARC_vs_CoCoPlus.xlsx` | about 13 h |
| Section 5.3 (43 saved solutions) | `scripts/experiments/run_best_solutions.py` | `best_solutions.xlsx`, `best_solutions/` | about 1 h |
| Section 5.5 (3,600 runs) | `scripts/experiments/run_ablation.sh` | `ablation_components.xlsx` | about 75 h |
| Section 5.6 (600 runs) | `scripts/experiments/run_larc_benchmark.py 600 larc_without_additions --config larc_base` | `ablation_LARC_without_additions.xlsx` | about 13 h |
| Sections 5.7, 5.8 (1,270 runs) | `scripts/experiments/run_sensitivity.sh` | `sensitivity_sampling_shares.xlsx`, `sensitivity_other_constants.xlsx` | about 27 h |
| Section 5.8, constructor | `scripts/experiments/run_constants_constructor.py` | `sensitivity_other_constants.xlsx` | minutes |
| Section 5.8, coordinated 2-opt | `scripts/experiments/run_search_probes.py coordinated_2opt` | `sensitivity_other_constants.xlsx` | about 1 h |
| Section 5.8, two-factor grid (540 runs) | `scripts/experiments/run_two_factor_grid.sh` | `two_factor_grid.xlsx` | about 11 h |
| Table 3 | `scripts/experiments/run_constructor_versions.py` | `constructor_versions.xlsx` | minutes |
| Table 4 | `scripts/experiments/run_search_probes.py start_tours_noise` | `method_experiments.xlsx` | under 1 h |
| Section 3.2.3, exact packing | `scripts/experiments/run_exact_packing.py runs/solutions` (after the line below) | `exact_packing.xlsx` | about 15 min |
| Section 3.3.3 | `scripts/experiments/run_search_probes.py first_pass_screen` | `method_experiments.xlsx` | about 10 min |
| Section 3.3.3, rejected candidates | `scripts/experiments/run_screen_measurement.py` | `screen_measurement.xlsx` | about 10 min |
| Section 3.5.1, Figure 9 | `scripts/experiments/run_search_probes.py probe_screening --instance d2103_n10510_uncorr-similar-weights_05 --tours 40` | `method_experiments.xlsx` | minutes |
| Section 3.5.1, Figure 9c | `scripts/experiments/run_search_probes.py probe_screening_large --tours 5 --seeds 1-8` | `probe_screening_large.xlsx` | under 1 h |
| Section 3.5.2 | `scripts/experiments/run_search_probes.py race_probe` | `method_experiments.xlsx` | about 1 h |
| Figures 7 and 8 | `scripts/experiments/run_figure_examples.py` | `method_experiments.xlsx` | seconds |

The runs on the held-out instances need `scripts/setup/prepare_heldout_instances.py` first. The exact packing is
computed on the final solutions of the runs of LARC of Section 5.5, saved in
`results/final_solutions_LARC_ablation_baseline.tar.xz`: `tar xJf results/final_solutions_LARC_ablation_baseline.tar.xz
-C runs` unpacks them to `runs/solutions/`.

All commands are run with `python3` from the root of the repository, except the `.sh` scripts. Instances are chosen
with `--list` (a list of the sheet `experiment_lists` of `data/benchmark_instances.xlsx`, or `heldout` for the 60
held-out instances) or `--names a,b`; seeds with
`--seeds 1-10`; the number of parallel runs with `--workers 8`. The experiments that run on a subset of the instances
take their list from that sheet.

The ablation (Section 5.5) and the two-factor grid (Section 5.8) use switches of `src/larc/ablation.py`, read from the environment when `larc` is
imported. When none is set, LARC is the method of the paper.

| Variable | Effect |
|---|---|
| `LARC_SECOND=0` | no second-order term: every load-aware estimate keeps only its first-order term |
| `LARC_COORD=0` | no coordinated 2-opt: every local search uses the plain 2-opt with the items fixed |
| `LARC_SCREEN=0` | coordinated 2-opt without the first-pass screen |
| `LARC_LHR=0` | each light-heavy re-sequencing step is replaced by annealing of the incumbent for the same time |
| `LARC_RACE=0` | no race: the best sampled candidate receives the time of the race alone |
| `LARC_COORD_FRAC=f` | damping of the coordinated 2-opt (0.25 in LARC) |

For example, `LARC_LHR=0 python3 scripts/experiments/run_larc_benchmark.py 600 no_lhr --seeds 1-10`. Each run records
the name of its configuration and the counters of the light-heavy steps and of the double-bridge kicks. The kernels
compiled with `LARC_SECOND=0` are cached separately, in `runs/numba_cache_no_second_order`.

## Results (`results/`)

One workbook per experiment. Every workbook starts with a sheet `README` that describes each sheet and column.

| Workbook | Content |
|---|---|
| `comparison_LARC_vs_CoCoPlus.xlsx` | the 1,200 runs of the same-machine comparison (60 instances, 10 seeds, 2 methods), the per-instance statistics (means, A12, Mann-Whitney and Holm p-values, outcome), the best known values, and the phase statistics of the 600 runs of LARC (Section 5.1, Tables 7 and A2) |
| `heldout_LARC_vs_CoCoPlus.xlsx` | the 60 held-out instances, the instance each one was drawn for, and the 1,200 runs of LARC and CoCo+ on them (Section 5.2, Table 8) |
| `best_solutions.xlsx` | the 43 runs of Section 5.3, each checked by two evaluators; the solutions are in `best_solutions/<instance>.txt` |
| `convergence_LARC.xlsx` | the convergence traces of the 600 runs of LARC, the time of each phase, and the data of Table 10 and Figure 15 |
| `ablation_components.xlsx` | the 3,600 runs of the ablation of the main components, with the counters of the light-heavy steps and of the kicks (Section 5.5, Table 11) |
| `final_solutions_LARC_ablation_baseline.tar.xz` | the final tour and packing of the 600 runs of LARC of the ablation (one `.npz` file per run), used for the error of the estimates and the exact packing |
| `ablation_LARC_without_additions.xlsx` | the 600 runs of Section 5.6 and the means per instance |
| `sensitivity_sampling_shares.xlsx` | the 1,030 runs of Section 5.7, the gaps per instance, and Table 12 |
| `sensitivity_other_constants.xlsx` | the constructor with other constants, the coordinated 2-opt with other damping and passes, the 240 runs with other annealing shares, and Table 13 |
| `two_factor_grid.xlsx` | the 540 runs of the grid of the sampling end and the damping of the coordinated 2-opt (Section 5.8) |
| `constructor_versions.xlsx` | the seven versions of the constructor of Table 3 on the 60 instances |
| `exact_packing.xlsx` | the exact best packing on the final tour of 410 runs of LARC, with the gaps of LARC and of the knapsack repack (Section 3.2.3) |
| `screen_measurement.xlsx` | the candidates rejected by the first-pass screen in 12 local searches, and what their complete repacking would give (Section 3.3.3) |
| `probe_screening_large.xlsx` | 40 candidates on each of the 33 instances with more than 1,500 cities: constructor value, value after the probe and after a longer search (Section 3.5.1, Figure 9c) |
| `method_experiments.xlsx` | the data of Section 3: starting tours on perturbed coordinates (Table 4), probe screening (Figure 9), first-pass screen, race probe, repack passes (Figure 7), light-heavy move (Figure 8), budget split (Figure 10) |

The objective values of CoCo+ were computed by the evaluator of LARC from the solutions printed by CoCo+. The value
printed by CoCo+ is in the sheet `runs` of `comparison_LARC_vs_CoCoPlus.xlsx`; CoCo+ prints six significant digits, and
on all 600 runs our value rounded to six significant digits equals the printed one.

## Data (`data/`)

- `ttp_instances_60.tar.xz`: the 60 instances of the categorized benchmark (20 TSP bases from eil76 to pla33810, with 1,
  5 and 10 items per city), the original `.ttp` files of the benchmark of Polyakovskiy et al. (2014), available at
  https://cs.adelaide.edu.au/~optlog/CEC2014COMP_InstancesNew/. `ttp_sha256.txt` lists their checksums.
- `ttp_instances_heldout_60.tar.xz`: the original `.ttp` files of the 60 held-out instances (Section 4.1), drawn from
  the same benchmark; `ttp_sha256_heldout.txt` lists their checksums. Their list is in the sheet `instances` of
  `results/heldout_LARC_vs_CoCoPlus.xlsx`.
- `benchmark_instances.xlsx`: the 60 instances with their category, size and file paths, and the instance lists of the
  experiments that use a subset of them.
- `published_values.xlsx`: the values of the nine published methods compared in the paper, with their sources, and the
  mean of CoCoP over ten runs from Namazi et al. (2023).
- `tours/`: one cached chained Lin-Kernighan tour per TSP base, used by the experiments on the constructor alone.

The instance `dsj1000` with one item per city has a renting rate of zero in the benchmark files, so it is a pure knapsack
problem (Section 2.2.1 of the paper). It is kept as distributed.

## The CoCo+ patch

CoCo+ seeds its random generator with `time(0)` and its embedded Lin-Kernighan with the clock in seconds, so runs started
in the same second share their random numbers. `patches/coco_larc.patch` adds an optional fourth argument, the run seed,
that is added as an offset to both seeds; the time-based seeding is kept. It also replaces `malloc.h` by `stdlib.h` so
that the code builds on macOS. The algorithm is not changed. The experiments use algorithm 4 of the code (CoCoP), as in
the code of its authors: https://github.com/majid75/CoCo.

## Licence

The licence of this repository is to be chosen by the authors. CoCo+ and Concorde are downloaded from their sources by
`scripts/setup/setup_external.sh` and keep their own licences.
