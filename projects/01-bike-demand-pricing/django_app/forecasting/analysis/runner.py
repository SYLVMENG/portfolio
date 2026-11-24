"""Runs bike_demand_pricing.py (the Anaconda/Jupyter analysis) and returns its result tables."""
import os
import runpy
import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ANALYSIS_DIR = Path(__file__).resolve().parent
_lock = threading.Lock()
_cache = {}


def run_analysis():
    with _lock:
        if not _cache:
            cwd = os.getcwd()
            os.chdir(ANALYSIS_DIR)  # the script reads/writes files next to itself
            try:
                import matplotlib.pyplot as plt
                plt.show = lambda *a, **k: None
                g = runpy.run_path(str(ANALYSIS_DIR / "bike_demand_pricing.py"))
            finally:
                os.chdir(cwd)
            _cache.update({k: g[k] for k in ("validation", "q1_summary", "q2_table", "units", "plan", "comparison", "holdout", "grid_check", "dense_check", "forecast_eval", "q2_oos", "estimation_check", "sensitivity")})
        return _cache
