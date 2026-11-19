"""
Plotting helpers shared by the notebooks.

Every function draws one kind of figure and returns the matplotlib Figure (or
draws on a given Axes), so the notebook decides whether to show or save it.
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SECONDS_PER_DAY


# ==========================================================================
# Autocorrelation
# ==========================================================================
def plot_acf(ax, lags, acf, bound, title="", ylim=None, show_legend=False):
    """Stem plot of the ACF with the +/- ``bound`` band of an independent sequence."""
    ax.vlines(lags, 0, acf, color="C0", linewidth=1.1, alpha=0.8)
    ax.plot(lags, acf, "o", color="C0", markersize=4)
    ax.axhline(0, color="black", linewidth=1)
    ax.axhline(bound, color="red", linestyle="--", linewidth=1.2, label="95% band")
    ax.axhline(-bound, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlim(1, max(1, lags[-1]))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.grid(True, alpha=0.35)
    if show_legend:
        ax.legend(loc="upper right")


def plot_acf_grid(acf_results, ncols=3, ylim=(-0.6, 0.6)):
    """
    One ACF panel per threshold.

    ``acf_results`` is a list of dicts with keys ``threshold``, ``n``,
    ``lags``, ``acf``, ``bound``.
    """
    nrows = math.ceil(len(acf_results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, r in enumerate(acf_results):
        plot_acf(
            axes[i],
            r["lags"],
            r["acf"],
            r["bound"],
            title=f"Threshold = {r['threshold']:g} SFU (N={r['n']})",
            ylim=ylim,
            show_legend=(i == 0),
        )
    for ax in axes[len(acf_results):]:
        ax.axis("off")

    fig.tight_layout()
    return fig


# ==========================================================================
# Raw catalogue
# ==========================================================================
def plot_peak_flux_timeline(df, start, end):
    """Each burst as a vertical line whose height is its peak flux (log scale)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.vlines(df["datetime"], ymin=1, ymax=df["peak_flux"], color="red", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_ylim(1, df["peak_flux"].max() * 1.5)
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.set_xlabel("Time")
    ax.set_ylabel("Peak Flux (SFU)")
    fig.tight_layout()
    return fig


# ==========================================================================
# Parameters vs threshold
# ==========================================================================
TREND_LABELS = {
    "constant": "constant-value line",
    "semilog": "semi-log regression line",
    "loglog": "log-log regression line",
}


def trend_line(x, y, kind):
    """
    Reference line to compare the estimates with:

    * ``constant``: y = mean(y)                  (no dependence on T)
    * ``semilog`` : y = a + b log(T)
    * ``loglog``  : log y = a + b log(T)          (power law)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)

    if kind == "constant":
        return np.repeat(np.mean(y[good]), len(x))
    if kind == "semilog":
        slope, intercept = np.polyfit(np.log(x[good]), y[good], 1)
        return intercept + slope * np.log(x)
    if kind == "loglog":
        good = good & (y > 0)
        slope, intercept = np.polyfit(np.log(x[good]), np.log(y[good]), 1)
        return np.exp(intercept + slope * np.log(x))
    raise ValueError(f"kind must be one of {list(TREND_LABELS)}")


def plot_parameter_vs_threshold(table, column, title, y_label, kind, level=0.95, log_x=True):
    """
    Estimate of ``column`` vs threshold, its confidence band
    (``column_ci_lower`` / ``column_ci_upper``) and a reference trend line.
    """
    table = table.sort_values("threshold")
    x = table["threshold"].to_numpy()
    y = table[column].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        x,
        table[f"{column}_ci_lower"],
        table[f"{column}_ci_upper"],
        alpha=0.35,
        label=f"{100 * level:.0f}% CI",
    )
    ax.plot(x, y, "o-", label="estimate")
    ax.plot(x, trend_line(x, y, kind), "--", label=TREND_LABELS[kind])

    if log_x:
        ax.set_xscale("log")
    if kind == "loglog":
        ax.set_yscale("log")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


# ==========================================================================
# Histogram of waiting times with fitted densities
# ==========================================================================
def plot_interevent_histogram_with_fits(times_days, fits, n_bins=12):
    """
    Log-log histogram (logarithmic bins) of the waiting times, in seconds,
    with the fitted probability densities on top.

    ``fits`` are ``FitResult`` objects fitted in days. Changing the time unit
    only rescales the scale parameter (the shape is unchanged), so they are
    converted to seconds here instead of being refitted.
    """
    times = np.asarray(times_days) * SECONDS_PER_DAY
    times = times[times > 0]

    edges = np.logspace(np.log10(times.min()), np.log10(times.max()), n_bins)
    counts, edges = np.histogram(times, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    density = counts / (len(times) * widths)
    density_error = np.sqrt(counts) / (len(times) * widths)
    has_counts = counts > 0

    x = np.logspace(np.log10(times.min()), np.log10(times.max()), 300)
    styles = ["r--", "g--", "m--", "c--", "y--"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for fit, style in zip(fits, styles):
        dist = fit.distribution
        params_seconds = (*fit.params[:-1], fit.params[-1] * SECONDS_PER_DAY)
        tag = dist.name[0]
        scale_s = params_seconds[-1]
        label = f"{dist.name} Fit ("
        if dist.n_params == 2:
            label += f"$k^{{({tag})}}$={params_seconds[0]:.2f}, "
        label += f"$\\theta^{{({tag})}}$={scale_s:.2f} s = {scale_s / SECONDS_PER_DAY:.2f} days)"
        ax.plot(x, dist.pdf(x, *params_seconds), style, linewidth=2, label=label)

    ax.errorbar(
        centers[has_counts],
        density[has_counts],
        yerr=density_error[has_counts],
        fmt="o",
        color="blue",
        ecolor="blue",
        capsize=3,
        label="Data (log bins)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-14, 1)
    ax.set_xlabel("Inter-Event Time (seconds)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Log-Log Histogram of Inter-Event Times with Fits")
    ax.grid(True, alpha=0.7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig
