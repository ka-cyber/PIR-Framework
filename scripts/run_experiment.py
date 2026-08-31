#!/usr/bin/env python3
"""
Convenience CLI wrapper around `pir_framework.evaluate`.

Examples
--------
Quick single-seed smoke run (fast, ~1200 windows):
    python scripts/run_experiment.py --n-per-regime 200

Full statistical-significance sweep (multi-seed, 95% CIs, CI-shaded
convergence plot, Pareto frontier):
    python scripts/run_experiment.py --n-per-regime 200 --n-seeds 5 --outdir results

Thompson Sampling variant instead of the LinUCB default:
    python scripts/run_experiment.py --n-per-regime 200 --n-seeds 5 --bandit thompson

This is a thin wrapper (`python -m pir_framework.evaluate` does the same
thing) provided for discoverability alongside `configs/default.yaml`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pir_framework.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
