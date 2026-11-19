"""
Project-wide settings.

Every constant that is shared by more than one analysis lives here, so that a
change (for example a different data file or a different minimum separation
between events) is made in exactly one place.

Settings that only matter for one analysis (number of bootstrap repeats,
forecast horizon, ...) are defined at the top of the corresponding notebook.
"""

from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "srb_notST4_2010_df.txt"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
TABLES_DIR = REPO_ROOT / "results" / "tables"

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
SECONDS_PER_DAY = 24 * 3600

# --------------------------------------------------------------------------
# Event selection
# --------------------------------------------------------------------------
# Two consecutive events closer than this (in seconds) are not counted as an
# inter-event time: the gap is dropped (for example near-simultaneous reports
# of the same burst).
MIN_INTER_EVENT_SECONDS = 119

# A threshold is only analysed if it leaves at least this many inter-event
# times; with fewer points the fits are not reliable.
MIN_INTER_EVENT_TIMES_PER_THRESHOLD = 10

# Default grid of peak-flux thresholds (in SFU): 20 values, log-spaced
# between 10^3 and 10^5.
THRESHOLD_GRID = np.logspace(3, 5, num=20)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_SEED = 12345
