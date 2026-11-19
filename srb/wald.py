"""
Wald test for a constant shape parameter across thresholds.

Question
--------
If bursts above different thresholds follow the same kind of process, only
the time scale should change with the threshold, not the shape. We test

    H0: k_1 = k_2 = ... = k_m      (same shape at every threshold)
    H1: at least one k_i differs.

Why a bootstrap is needed
-------------------------
The thresholds are nested (every burst above 10^4 SFU is also above 5*10^3),
so the shape estimates are strongly correlated. Their full covariance matrix
is estimated by resampling the catalogue and refitting every threshold.
"""

import numpy as np
from scipy.stats import chi2

from .data import inter_event_times_by_threshold
from .fitting import fit_mle


def bootstrap_catalogue(df, rng):
    """Resample the rows of the burst catalogue with replacement."""
    rows = rng.integers(0, len(df), size=len(df))
    return df.iloc[rows].sort_values("datetime").reset_index(drop=True)


def fit_shapes(groups, dist):
    """Fit every threshold separately; return the shapes, or None if any fit fails."""
    shapes = []
    for times in groups.values():
        fit = fit_mle(times, dist)
        if not fit.success:
            return None
        shapes.append(fit.params[0])
    return np.array(shapes)


def bootstrap_shape_covariance(df, thresholds, dist, n_bootstrap, rng):
    """
    Covariance matrix of the shape estimates across thresholds.

    A bootstrap sample is discarded if it loses any threshold (fewer than the
    minimum number of waiting times) or if any fit fails.

    Returns
    -------
    (covariance, shape_samples) where ``shape_samples`` has one row per
    accepted bootstrap sample.
    """
    samples = []
    for _ in range(n_bootstrap):
        groups = inter_event_times_by_threshold(bootstrap_catalogue(df, rng), thresholds, verbose=False)
        if len(groups) != len(thresholds):
            continue
        shapes = fit_shapes(groups, dist)
        if shapes is not None:
            samples.append(shapes)

    samples = np.asarray(samples)
    if len(samples) < 2:
        raise RuntimeError("Fewer than two successful bootstrap fits; cannot estimate covariance.")
    return np.cov(samples, rowvar=False), samples


def wald_test_equal_shapes(shapes, covariance):
    """
    Wald statistic for H0: all shapes equal.

    The m shapes are turned into m-1 differences with respect to the first
    one, d = C k. Under H0, d = 0 and

        W = d^T (C Sigma C^T)^-1 d  ~  chi^2 with m-1 degrees of freedom.

    Returns
    -------
    (W, degrees_of_freedom, p_value)
    """
    k = np.asarray(shapes)
    m = len(k)

    contrasts = np.zeros((m - 1, m))
    contrasts[:, 0] = -1
    contrasts[np.arange(m - 1), np.arange(1, m)] = 1

    differences = contrasts @ k
    covariance_of_differences = contrasts @ covariance @ contrasts.T

    W = differences @ np.linalg.pinv(covariance_of_differences) @ differences
    dof = m - 1
    return W, dof, chi2.sf(W, dof)
