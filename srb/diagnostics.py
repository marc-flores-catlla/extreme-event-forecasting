"""
Model diagnostics: autocorrelation, goodness of fit and bootstrap summaries.
"""

import numpy as np
from scipy.stats import norm

from .fitting import fit_mle


# ==========================================================================
# Autocorrelation of the sequence of waiting times
# ==========================================================================
def autocorrelation(x, max_lag=None):
    """
    Sample autocorrelation function (ACF) for lags 1..max_lag.

        r_k = sum_i (x_i - mean)(x_{i+k} - mean) / sum_i (x_i - mean)^2

    If the waiting times are independent (as in a Poisson process) every r_k
    should be close to zero.

    Returns
    -------
    (lags, acf), or (None, None) if the ACF is undefined (fewer than two
    values or zero variance).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 or np.var(x) == 0:
        return None, None

    max_lag = n - 1 if max_lag is None else min(max_lag, n - 1)
    centered = x - np.mean(x)
    denominator = np.sum(centered**2)

    lags = np.arange(1, max_lag + 1)
    acf = np.array([np.sum(centered[:-k] * centered[k:]) / denominator for k in lags])
    return lags, acf


def acf_confidence_bound(n, level=0.95):
    """
    Half-width of the band in which ~``level`` of the ACF values of an
    independent sequence of length ``n`` should fall: z / sqrt(n).
    """
    return norm.ppf(0.5 + level / 2) / np.sqrt(n)


# ==========================================================================
# Goodness of fit: Kolmogorov-Smirnov with estimated parameters (Lilliefors)
# ==========================================================================
def ks_statistic(times, dist, params):
    """Largest vertical distance between the empirical CDF and the model CDF."""
    times = np.sort(times)
    n = len(times)
    cdf = np.clip(dist.cdf(times, *params), 0, 1)
    d_plus = np.max(np.arange(1, n + 1) / n - cdf)
    d_minus = np.max(cdf - np.arange(0, n) / n)
    return max(d_plus, d_minus)


def lilliefors_test(times, dist, n_repeats, rng):
    """
    Monte-Carlo p-value of the KS test when the parameters are estimated
    from the same data (Lilliefors correction).

    The usual KS tables assume the parameters are known in advance. Here they
    are fitted, which makes the observed KS distance smaller than it would be
    otherwise. To get a fair reference distribution:

    1. Fit the model to the data and compute the observed KS distance.
    2. Repeat ``n_repeats`` times: simulate a sample of the same size from the
       fitted model, refit it, and compute its KS distance.
    3. p-value = fraction of simulated distances >= the observed one
       (with the +1 correction so it is never exactly 0).

    Returns
    -------
    (p_value, fit) where ``fit`` is the ``FitResult`` on the original data.
    """
    fit = fit_mle(times, dist)
    observed = ks_statistic(times, dist, fit.params)

    count = 0
    for _ in range(n_repeats):
        simulated = dist.sample(rng, len(times), *fit.params)
        simulated_fit = fit_mle(simulated, dist)
        if ks_statistic(simulated, dist, simulated_fit.params) >= observed:
            count += 1

    p_value = (count + 1) / (n_repeats + 1)
    return p_value, fit


def bootstrap_aic(times, dist, n_repeats, rng):
    """AIC of the model refitted on ``n_repeats`` bootstrap resamples of ``times``."""
    values = []
    for _ in range(n_repeats):
        resample = rng.choice(times, size=len(times), replace=True)
        values.append(fit_mle(resample, dist).aic)
    return np.array(values)


# ==========================================================================
# Uncertainty summaries
# ==========================================================================
def bootstrap_summary(values):
    """Standard deviation and 95% percentile interval of bootstrap values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    return np.std(values, ddof=1), np.percentile(values, 2.5), np.percentile(values, 97.5)


def monte_carlo_p_value_error(p_value, n_repeats):
    """
    Uncertainty of a Monte-Carlo p-value coming from the finite number of
    simulations (binomial standard error) and its 95% interval.
    """
    if not np.isfinite(p_value):
        return np.nan, np.nan, np.nan
    error = np.sqrt(p_value * (1 - p_value) / n_repeats)
    return error, max(0, p_value - 1.96 * error), min(1, p_value + 1.96 * error)
