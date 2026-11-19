"""
Maximum-likelihood fitting.

Two kinds of model are fitted:

1. **One threshold at a time** (``fit_mle``): the waiting times above a single
   threshold T are fitted with their own (shape, scale).

2. **All thresholds jointly** (``fit_scale_regression``): one shared shape and
   a scale that grows with the threshold as a power law,

       scale(T) = b0 * T**b1.

   This is what allows extrapolating to thresholds that were never (or very
   rarely) observed, which is needed for forecasting.

Positive parameters (shape, scale, b0) are optimised on the log scale, so the
optimiser can move freely without ever proposing a negative value.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

OPTIMIZER_OPTIONS = {"maxiter": 10000}


def negative_log_likelihood(dist, params, times):
    """-sum(log f(t_i | params)); returns +inf where the density is not defined."""
    with np.errstate(all="ignore"):
        value = -np.sum(dist.logpdf(times, *params))
    return value if np.isfinite(value) else np.inf


def aic(nll, n_params):
    """Akaike Information Criterion: lower is better."""
    return 2 * n_params + 2 * nll


# ==========================================================================
# 1. One threshold at a time
# ==========================================================================
@dataclass
class FitResult:
    distribution: object
    params: np.ndarray  # same order as distribution.param_names
    nll: float
    success: bool

    @property
    def aic(self):
        return aic(self.nll, self.distribution.n_params)

    def as_dict(self):
        return dict(zip(self.distribution.param_names, self.params))


def fit_mle(times, dist):
    """
    Maximum-likelihood fit of ``dist`` to ``times``.

    The optimiser starts from shape = 1 (i.e. the exponential case) and
    scale = mean(times).
    """
    times = np.asarray(times, dtype=float)

    if dist.closed_form_mle is not None:
        params = dist.closed_form_mle(times)
        return FitResult(dist, params, negative_log_likelihood(dist, params, times), True)

    start = np.log(np.r_[np.ones(dist.n_params - 1), np.mean(times)])

    def objective(log_params):
        return negative_log_likelihood(dist, np.exp(log_params), times)

    result = minimize(objective, start, method="Nelder-Mead", options=OPTIMIZER_OPTIONS)
    success = bool(result.success and np.isfinite(result.fun))
    return FitResult(dist, np.exp(result.x), float(result.fun), success)


def numerical_hessian(function, x, step=1e-4):
    """Central-difference Hessian of a scalar function at point ``x``."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    steps = step * np.maximum(1.0, np.abs(x))
    hessian = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = steps[i]
            ej[j] = steps[j]
            hessian[i, j] = (
                function(x + ei + ej)
                - function(x + ei - ej)
                - function(x - ei + ej)
                + function(x - ei - ej)
            ) / (4 * steps[i] * steps[j])

    return 0.5 * (hessian + hessian.T)


def fit_with_confidence_intervals(times, dist, level=0.95):
    """
    MLE fit plus Wald confidence intervals for every parameter.

    How the intervals are built
    ---------------------------
    1. Fit the parameters on the log scale.
    2. The Hessian of the negative log-likelihood at the optimum (the
       *observed information*) is inverted to get the covariance matrix of
       the log-parameters.
    3. log(param) +/- z * se gives an interval on the log scale, which is
       exponentiated back. The interval is therefore asymmetric and always
       positive.

    Returns
    -------
    dict with, for every parameter ``p``: ``p``, ``p_ci_lower``, ``p_ci_upper``,
    ``p_se_log``; plus ``cov_log_shape_scale`` for two-parameter models.
    """
    times = np.asarray(times, dtype=float)
    fit = fit_mle(times, dist)
    log_params = np.log(fit.params)

    def objective(log_p):
        return negative_log_likelihood(dist, np.exp(log_p), times)

    hessian = numerical_hessian(objective, log_params)
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian)
    covariance = 0.5 * (covariance + covariance.T)

    variances = np.diag(covariance)
    variances = np.where(variances < -1e-10, np.nan, np.maximum(variances, 0.0))
    se_log = np.sqrt(variances)
    z = norm.ppf(0.5 + level / 2)

    row = {}
    for i, name in enumerate(dist.param_names):
        row[name] = np.exp(log_params[i])
        row[f"{name}_ci_lower"] = np.exp(log_params[i] - z * se_log[i])
        row[f"{name}_ci_upper"] = np.exp(log_params[i] + z * se_log[i])
        row[f"{name}_se_log"] = se_log[i]
    if dist.n_params == 2:
        row["cov_log_shape_scale"] = covariance[0, 1]
    return row


# ==========================================================================
# 2. All thresholds jointly: scale(T) = b0 * T**b1
# ==========================================================================
@dataclass
class RegressionFit:
    distribution: object
    shape_params: np.ndarray  # empty for the exponential
    b0: float
    b1: float
    nll: float

    @property
    def n_params(self):
        return len(self.shape_params) + 2

    @property
    def aic(self):
        return aic(self.nll, self.n_params)

    def scale(self, threshold):
        return self.b0 * np.asarray(threshold, dtype=float) ** self.b1

    def params_at(self, threshold):
        """Distribution parameters for a given threshold: (*shape_params, scale(T))."""
        return (*self.shape_params, self.scale(threshold))


def fit_scale_regression(long_table, dist, start_shape=1.0, start_b0=1.5, start_b1=0.0):
    """
    Fit one shared shape and scale(T) = b0 * T**b1 to all thresholds at once.

    ``long_table`` has one row per waiting time with columns ``threshold`` and
    ``inter_event_days`` (see ``data.to_long_table``).

    Free parameters: log(shape) (if any), log(b0), b1.
    """
    thresholds = long_table["threshold"].to_numpy(dtype=float)
    times = long_table["inter_event_days"].to_numpy(dtype=float)
    n_shape = dist.n_params - 1

    def unpack(theta):
        shape_params = np.exp(theta[:n_shape])
        b0 = np.exp(theta[n_shape])
        b1 = theta[n_shape + 1]
        return shape_params, b0, b1

    def objective(theta):
        shape_params, b0, b1 = unpack(theta)
        scale = b0 * thresholds**b1
        return negative_log_likelihood(dist, (*shape_params, scale), times)

    start = np.r_[np.log(np.full(n_shape, start_shape)), np.log(start_b0), start_b1]
    result = minimize(objective, start, method="Nelder-Mead", options=OPTIMIZER_OPTIONS)

    shape_params, b0, b1 = unpack(result.x)
    return RegressionFit(dist, shape_params, float(b0), float(b1), float(result.fun))
