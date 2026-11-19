"""
Probability of an extreme burst within a time horizon.

Using a threshold-regression fit (``fitting.RegressionFit``) we extrapolate
the waiting-time distribution to a very high threshold and compute:

* **Unconditional probability**: P(waiting time <= horizon) = 1 - S(h),
  i.e. the chance that the next extreme burst arrives within the horizon,
  counting from the previous one.

* **Conditional probability**: we already know that ``a`` days have passed
  since the last extreme burst without a new one, so

      P(event in (a, a+h] | no event in (0, a]) = 1 - S(a + h) / S(a).

  For the exponential distribution both numbers are identical (no memory).
"""

import numpy as np

from .config import SECONDS_PER_DAY
from .fitting import fit_scale_regression


def forecast_probabilities(df, fit, forecast_threshold, horizon_days, reference_time):
    """
    Returns
    -------
    dict with ``p_unconditional``, ``p_conditional``, ``last_event_time``,
    ``elapsed_days`` and ``scale`` (the extrapolated scale, in days).
    """
    dist = fit.distribution
    params = fit.params_at(forecast_threshold)
    p_unconditional = 1 - dist.sf(horizon_days, *params)

    extreme = df[df["peak_flux"] >= forecast_threshold]
    if len(extreme) == 0:
        last_event_time, elapsed_days, p_conditional = None, np.nan, np.nan
    else:
        last_event_time = extreme["datetime"].max()
        elapsed_days = (reference_time - last_event_time).total_seconds() / SECONDS_PER_DAY
        p_conditional = 1 - dist.sf(elapsed_days + horizon_days, *params) / dist.sf(elapsed_days, *params)

    return {
        "p_unconditional": p_unconditional,
        "p_conditional": p_conditional,
        "last_event_time": last_event_time,
        "elapsed_days": elapsed_days,
        "scale": fit.scale(forecast_threshold),
    }


def bootstrap_forecast(long_table, df, dist, forecast_threshold, horizon_days, reference_time, n_repeats, rng):
    """
    Bootstrap distribution of both probabilities.

    Each repeat resamples the rows of ``long_table`` (waiting times of all
    thresholds) with replacement, refits the regression and recomputes the
    forecast.

    Returns
    -------
    (p_unconditional_values, p_conditional_values) as arrays.
    """
    unconditional, conditional = [], []
    for _ in range(n_repeats):
        resample = long_table.sample(n=len(long_table), replace=True, random_state=rng)
        fit = fit_scale_regression(resample, dist)
        result = forecast_probabilities(df, fit, forecast_threshold, horizon_days, reference_time)
        unconditional.append(result["p_unconditional"])
        conditional.append(result["p_conditional"])
    return np.array(unconditional), np.array(conditional)
