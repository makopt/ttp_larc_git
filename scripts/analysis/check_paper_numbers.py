"""Recomputes the numbers of the paper from the saved results and compares each one with the value printed in the paper.

Runs every analysis script below, writes its full output to runs/paper_numbers/<script>.txt, and prints one summary line
per script and every number that differs from the paper (status DIFF of scripts/analysis/paper_check.py). The scripts
read only results/ and data/; estimate_error.py takes about three minutes, the others a few seconds each.
Exit status: 0 when no number differs, 1 otherwise.

Usage: python3 scripts/analysis/check_paper_numbers.py [--skip-slow]"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from larc import RUNS_DIR  # noqa: E402

SCRIPTS = [  # script, what it checks
    ("reproduce_paper_numbers.py", "Tables 3, 4, 7, 10, 12, 13; Sections 3.3.3, 3.5.1, 5.1, 5.3, 5.4, 5.6 to 5.9"),
    ("comparison_uncertainty.py", "confidence intervals of Table 7 and Section 5.1"),
    ("heldout_comparison.py", "held-out instances: Section 5.2 and Table 8"),
    ("ablation_components.py", "ablation of the main components: Section 5.5 and Table 11"),
    ("estimate_error.py", "error of the load-aware estimates: text after Proposition 1"),
    ("screen_and_exact_packing.py", "exact packing on a fixed tour and first-pass screen: Sections 3.2.3 and 3.3.3"),
    ("probe_screening_large.py", "probe screening on the large instances: Section 3.5.1 and Figure 9c"),
    ("two_factor_grid.py", "two-factor grid: Section 5.8"),
    ("gap_by_instance_features.py", "gap and instance characteristics: Section 5.9"),
]
SLOW = {"estimate_error.py"}

ap = argparse.ArgumentParser()
ap.add_argument("--skip-slow", action="store_true", help="skip estimate_error.py (about three minutes)")
A = ap.parse_args()

out_dir = RUNS_DIR / "paper_numbers"
out_dir.mkdir(parents=True, exist_ok=True)
total = {"ok": 0, "DIFF": 0, "--": 0}
diffs, failed = [], []
for script, what in SCRIPTS:
    if A.skip_slow and script in SLOW:
        print(f"{script:30s} skipped")
        continue
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, str(HERE / script)], cwd=REPO, capture_output=True, text=True)
    (out_dir / script.replace(".py", ".txt")).write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        failed.append(script)
        print(f"{script:30s} FAILED (see {out_dir / script.replace('.py', '.txt')})")
        continue
    n = {"ok": 0, "DIFF": 0, "--": 0}
    for line in p.stdout.splitlines():
        s = line[:5].strip()
        if s in n and "| paper:" in line:
            n[s] += 1
            if s == "DIFF":
                diffs.append(f"  {script}: {line[5:].strip()}")
    for k in n:
        total[k] += n[k]
    print(f"{script:30s} {n['ok']:4d} agree {n['DIFF']:3d} differ {n['--']:3d} printed  "
          f"({time.monotonic() - t0:.0f} s)  {what}")

print(f"\nall scripts: {total['ok']} numbers agree with the paper, {total['DIFF']} differ, "
      f"{total['--']} printed for reading; full outputs in {out_dir}")
if diffs:
    print("numbers that differ:\n" + "\n".join(diffs))
if failed:
    print("scripts that failed: " + ", ".join(failed))
sys.exit(1 if diffs or failed else 0)
