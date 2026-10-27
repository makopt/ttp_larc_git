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
data/                      the 60 benchmark instances, the published values, cached tours
results/                   the results of the paper, one Excel workbook per experiment
figures/                   the statistical figures of the paper (PDF and PNG)
patches/coco_larc.patch    the patch applied to the public code of CoCo+ (seed argument only)
tests/                     quick checks of the evaluator, the knapsack repack and the local search
```

### The code (`src/larc/`)

| Module | Content |
|---|---|
| `larc.py` | `solve_larc`, the method of the paper (Algorithm 1): sampling, race, recombination, final search, final repack |
| `larc_base.py` | `solve_larc_base`, LARC without perturbed tours, probe screening, kicks and chunked final search (Section 5.4) |
| `construct.py` | the constructor (Algorithm 2): load-aware item key, item windows, flip search, knapsack repack |
| `search.py` | the coordinated local search (Algorithms 4 and 5), the light-heavy re-sequencing move, annealing, kicks |
| `kernels.py` | the compiled kernels (Numba): 2-opt, Or-opt, item flips, coordinated 2-opt, knapsack dynamic program |
| `recombination.py` | the recombination of two solutions (partition crossover with item transplant) |
| `tours.py` | chained Lin-Kernighan tours from Concorde `linkern`, the paths of the light-heavy move, the cached tours |
| `evaluator.py` | the objective function and the feasibility check |
| `instance.py` | the instance as arrays, read from its JSON file |
| `benchmark.py` | the list of the 60 instances and the published values |
| `trace.py`, `stats.py` | convergence trace of a run; mean ranks, RDI and critical difference |

## Installation

Tested on macOS 26 (Apple M1 Pro, Python 3.13, Apple clang 17). Linux should work the same way; the build script only
passes an explicit host type to the configure script of Concorde on macOS.

```bash
pip install -r requirements.txt
./scripts/setup/setup_external.sh             # CoCo+ -> external/CoCo-src/coco_seeded, linkern -> bin/linkern (about 30 s)
python3 scripts/setup/prepare_instances.py    # unpacks the 60 .ttp files, checks their sha256, converts them to JSON
python3 scripts/setup/check_instances.py      # checks that the JSON and .ttp files hold the same data
python3 -m pytest -q tests                    # optional
```

Run LARC on one instance:

```bash
python3 scripts/experiments/run_larc.py a280_n279_bounded-strongly-corr_01 --seconds 60 --seed 1
```

It prints the objective value and writes the solution to `runs/solutions/` in the format of the TTP competitions (first
line the tour, second line the picked items, both numbered from 1). The first run compiles the Numba kernels (about a
minute); later runs use the cache in `runs/numba_cache`. New runs are written to `runs/`, or to the folder given by the
environment variable `LARC_RUNS_DIR`.

## Reproducing the paper

### From the saved results (seconds)

```bash
python3 scripts/analysis/reproduce_paper_numbers.py   # Tables 3, 4, 6 to 9 and the numbers of Sections 3.5 and 5
python3 scripts/analysis/make_figures.py              # Figures 7 to 10 and 12 to 17 -> figures/
python3 scripts/analysis/summarize_convergence.py     # Table 7 and the data of Figure 16, from the traces of the runs
```

`reproduce_paper_numbers.py` prints each number next to the value printed in the paper. These scripts read only
`results/` and `data/`. `summarize_convergence.py` also accepts the workbook of new runs of LARC.

### Running the experiments again

Each run has 600 s of wall-clock time from the start of its process, and eight runs are made at a time. The results
depend on the machine, so a new run does not repeat the values of the paper exactly; the saved results are those of the
paper. Every experiment script writes an Excel workbook to `runs/` with the same columns as the matching sheet of
`results/`, writes it again after every finished run, and skips the runs already done when it is started again.

| Paper | Command | Saved result | Time (8 cores) |
|---|---|---|---|
| Section 5.1, 5.3 (LARC, 600 runs) | `scripts/experiments/run_larc_benchmark.py 600 larc_main` | `comparison_LARC_vs_CoCoPlus.xlsx`, `convergence_LARC.xlsx` | about 13 h |
| Section 5.1 (CoCo+, 600 runs) | `scripts/experiments/run_cocoplus_benchmark.py 600 cocoplus_main` | `comparison_LARC_vs_CoCoPlus.xlsx` | about 13 h |
| Section 5.2 (43 saved solutions) | `scripts/experiments/run_best_solutions.py` | `best_solutions.xlsx`, `best_solutions/` | about 1 h |
| Section 5.4 (600 runs) | `scripts/experiments/run_larc_benchmark.py 600 larc_without_additions --config larc_base` | `ablation_LARC_without_additions.xlsx` | about 13 h |
| Sections 5.5, 5.6 (1,270 runs) | `scripts/experiments/run_sensitivity.sh` | `sensitivity_sampling_shares.xlsx`, `sensitivity_other_constants.xlsx` | about 27 h |
| Section 5.6, constructor | `scripts/experiments/run_constants_constructor.py` | `sensitivity_other_constants.xlsx` | minutes |
| Section 5.6, coordinated 2-opt | `scripts/experiments/run_search_probes.py coordinated_2opt` | `sensitivity_other_constants.xlsx` | about 1 h |
| Table 3 | `scripts/experiments/run_constructor_versions.py` | `constructor_versions.xlsx` | minutes |
| Table 4 | `scripts/experiments/run_search_probes.py start_tours_noise` | `method_experiments.xlsx` | under 1 h |
| Section 3.3.3 | `scripts/experiments/run_search_probes.py first_pass_screen` | `method_experiments.xlsx` | about 10 min |
| Section 3.5.1, Figure 9 | `scripts/experiments/run_search_probes.py probe_screening --instance d2103_n10510_uncorr-similar-weights_05 --tours 40` | `method_experiments.xlsx` | minutes |
| Section 3.5.2 | `scripts/experiments/run_search_probes.py race_probe` | `method_experiments.xlsx` | about 1 h |
| Figures 7 and 8 | `scripts/experiments/run_figure_examples.py` | `method_experiments.xlsx` | seconds |

All commands are run with `python3` from the root of the repository, except `run_sensitivity.sh`. Instances are chosen
with `--list` (a list of the sheet `experiment_lists` of `data/benchmark_instances.xlsx`) or `--names a,b`; seeds with
`--seeds 1-10`; the number of parallel runs with `--workers 8`. The experiments that run on a subset of the instances
take their list from that sheet.

## Results (`results/`)

One workbook per experiment. Every workbook starts with a sheet `README` that describes each sheet and column.

| Workbook | Content |
|---|---|
| `comparison_LARC_vs_CoCoPlus.xlsx` | the 1,200 runs of the same-machine comparison (60 instances, 10 seeds, 2 methods), the per-instance statistics (means, A12, Mann-Whitney and Holm p-values, outcome), the best known values, and the phase statistics of the 600 runs of LARC |
| `convergence_LARC.xlsx` | the convergence traces of the 600 runs of LARC, the time of each phase, and the data of Table 7 and Figure 16 |
| `ablation_LARC_without_additions.xlsx` | the 600 runs of Section 5.4 and the means per instance |
| `sensitivity_sampling_shares.xlsx` | the 1,030 runs of Section 5.5, the gaps per instance, and Table 8 |
| `sensitivity_other_constants.xlsx` | the constructor with other constants, the coordinated 2-opt with other damping and passes, the 240 runs with other annealing shares, and Table 9 |
| `constructor_versions.xlsx` | the seven versions of the constructor of Table 3 on the 60 instances |
| `best_solutions.xlsx` | the 43 runs of Section 5.2, each checked by two evaluators; the solutions are in `best_solutions/<instance>.txt` |
| `method_experiments.xlsx` | the data of Section 3: start tours on perturbed coordinates (Table 4), probe screening (Figure 9), first-pass screen, race probe, repack passes (Figure 7), light-heavy move (Figure 8), budget split (Figure 10) |

The objective values of CoCo+ were computed by the evaluator of LARC from the solutions printed by CoCo+. The value
printed by CoCo+ is in the sheet `runs` of `comparison_LARC_vs_CoCoPlus.xlsx`; CoCo+ prints six significant digits, and
on all 600 runs our value rounded to six significant digits equals the printed one.

## Data (`data/`)

- `ttp_instances_60.tar.xz`: the 60 instances of the categorized benchmark (20 TSP bases from eil76 to pla33810, with 1,
  5 and 10 items per city), the original `.ttp` files of the benchmark of Polyakovskiy et al. (2014), available at
  https://cs.adelaide.edu.au/~optlog/CEC2014COMP_InstancesNew/. `ttp_sha256.txt` lists their checksums.
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
