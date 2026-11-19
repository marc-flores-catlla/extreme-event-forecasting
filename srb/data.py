"""
Loading the burst catalogue and turning it into inter-event times.

Pipeline
--------
1. ``load_catalogue``                  -> one row per burst (time, station, peak flux, type).
2. ``inter_event_times``               -> waiting times between bursts above ONE threshold.
3. ``inter_event_times_by_threshold``  -> the same for MANY thresholds, as a dict.
4. ``to_long_table``                   -> that dict as a long DataFrame
                                          (one row per waiting time), used by the
                                          threshold-regression models.
"""

import numpy as np
import pandas as pd

from .config import (
    DATA_FILE,
    MIN_INTER_EVENT_SECONDS,
    MIN_INTER_EVENT_TIMES_PER_THRESHOLD,
    SECONDS_PER_DAY,
    THRESHOLD_GRID,
)


def load_catalogue(path=DATA_FILE):
    """
    Read the burst catalogue.

    The text file has a header line and five whitespace-separated columns:
    date, time, station, peak flux (SFU) and spectral type (ST).

    Returns
    -------
    pandas.DataFrame with columns ``datetime``, ``station``, ``peak_flux``,
    ``st``, sorted by time.
    """
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=1,
        names=["date", "time", "station", "peak_flux", "st"],
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
    df = df[["datetime", "station", "peak_flux", "st"]]
    return df.sort_values("datetime").reset_index(drop=True)


def select_thresholds(start, end, grid=THRESHOLD_GRID):
    """Keep the thresholds of ``grid`` that lie in [start, end]."""
    grid = np.asarray(grid, dtype=float)
    return grid[(grid >= start) & (grid <= end)]


def inter_event_times(df, threshold, min_separation_seconds=MIN_INTER_EVENT_SECONDS):
    """
    Waiting times (in days) between consecutive bursts with peak flux >= threshold.

    Steps
    -----
    1. Keep the bursts whose peak flux reaches the threshold.
    2. Sort them in time and take the difference between consecutive times.
    3. Drop gaps of ``min_separation_seconds`` or less.
    4. Convert seconds to days.
    """
    events = df[df["peak_flux"] >= threshold].sort_values("datetime")
    gaps_seconds = events["datetime"].diff().dt.total_seconds().dropna()
    gaps_seconds = gaps_seconds[gaps_seconds > min_separation_seconds]
    return gaps_seconds.to_numpy() / SECONDS_PER_DAY


def inter_event_times_by_threshold(
    df,
    thresholds,
    min_count=MIN_INTER_EVENT_TIMES_PER_THRESHOLD,
    verbose=True,
):
    """
    Compute ``inter_event_times`` for every threshold.

    Thresholds that leave fewer than ``min_count`` waiting times are skipped.

    Returns
    -------
    dict {threshold: array of waiting times in days}, in increasing threshold order.
    """
    groups = {}
    for threshold in thresholds:
        times = inter_event_times(df, threshold)
        if len(times) < min_count:
            if verbose:
                print(f"Skipping T = {threshold:.2f}: only {len(times)} inter-event times")
            continue
        groups[float(threshold)] = times
    return groups


def to_long_table(groups):
    """
    Flatten ``{threshold: times}`` into a DataFrame with one row per waiting time.

    Columns: ``threshold``, ``inter_event_days``.
    """
    frames = [
        pd.DataFrame({"threshold": threshold, "inter_event_days": times})
        for threshold, times in groups.items()
    ]
    return pd.concat(frames, ignore_index=True)
