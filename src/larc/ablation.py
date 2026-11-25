"""Switches of the ablation and of the two-factor grid of the paper (Sections 5.5 and 5.8), read once from the environment
when the package is imported. When none of them is set, LARC is the method of the paper.

  LARC_SECOND=0      no second-order term: every load-aware estimate keeps only its first-order term (G2 = 0)
  LARC_COORD=0       no coordinated 2-opt: every local search uses the plain 2-opt with the items fixed
  LARC_SCREEN=0      coordinated 2-opt without the first-pass screen: every candidate gets all its repacking passes
  LARC_LHR=0         no light-heavy re-sequencing: each light-heavy step of macro_search is replaced by annealing of the
                     incumbent for the same time
  LARC_RACE=0        no racing: the best sampled candidate receives the time of the race alone
  LARC_COORD_FRAC=f  damping of the coordinated 2-opt, the share of the promising items flipped per pass (0.25 in LARC),
                     used by every local search that does not set coord_frac itself

The variables must be set before larc is imported. Numba stores the value of LARC_SECOND in its compiled kernels, so
larc/__init__.py gives each value its own compilation cache.

STATS holds counters of the light-heavy steps and of the double-bridge kicks of macro_search (attempts, improvements, time
and gain), reset by reset_stats() at the start of a run. They do not change the search."""
import os


def _on(name: str) -> bool:
    return os.environ.get(name, "1") != "0"


SECOND = 1.0 if _on("LARC_SECOND") else 0.0
COORD = _on("LARC_COORD")
SCREEN = _on("LARC_SCREEN")
LHR = _on("LARC_LHR")
RACE = _on("LARC_RACE")
COORD_FRAC = float(os.environ.get("LARC_COORD_FRAC", "0.25"))

# name of the configuration, recorded with each run: "LARC", or the switches that differ from LARC
VARIANT = "+".join(k for k, off in (("no_second_order", SECOND == 0.0), ("no_coordinated_2opt", not COORD),
                                    ("no_first_pass_screen", not SCREEN), ("no_light_heavy", not LHR),
                                    ("no_racing", not RACE), (f"coord_frac_{COORD_FRAC:g}", COORD_FRAC != 0.25))
                   if off) or "LARC"

STATS = {}


def reset_stats():
    STATS.clear()
    STATS.update(lhr_tries=0, lhr_wins=0, lhr_time=0.0, lhr_gain=0.0,
                 kick_tries=0, kick_wins=0, kick_time=0.0, kick_gain=0.0)


reset_stats()
