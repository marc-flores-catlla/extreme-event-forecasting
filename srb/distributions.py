"""
Candidate distributions for the inter-event times.

Each distribution is described by a ``Distribution`` object that bundles the
formulas the rest of the code needs (log-density, CDF, survival function and a
random sampler). The fitting, testing and forecasting code only talks to this
interface, so it never needs to know which distribution it is working with.

Parameter convention
--------------------
Parameters are passed positionally and the **scale is always the last one**:

    Exponential : (scale,)
    Weibull     : (shape, scale)
    Gamma       : (shape, scale)

The threshold-regression model (see ``fitting.fit_scale_regression``) relies on
this convention: it keeps the shape parameters and replaces the scale with
``b0 * T**b1``.

Adding a new distribution
-------------------------
Write its ``logpdf``, ``cdf``, ``sf`` and ``sample`` functions following the
convention above, build a ``Distribution`` and add it to ``DISTRIBUTIONS``.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.special import gammainc, gammaincc, gammaln


@dataclass(frozen=True)
class Distribution:
    name: str
    param_names: tuple
    logpdf: Callable  # logpdf(t, *params)
    cdf: Callable  # cdf(t, *params)
    sf: Callable  # survival function, sf(t, *params) = P(X > t)
    sample: Callable  # sample(rng, size, *params)
    closed_form_mle: Optional[Callable] = None  # times -> params, if an exact MLE exists

    @property
    def n_params(self):
        return len(self.param_names)

    def pdf(self, t, *params):
        return np.exp(self.logpdf(t, *params))


# --------------------------------------------------------------------------
# Exponential: f(t) = (1/s) exp(-t/s)
# Memoryless; corresponds to a homogeneous Poisson process.
# --------------------------------------------------------------------------
EXPONENTIAL = Distribution(
    name="Exponential",
    param_names=("scale",),
    logpdf=lambda t, scale: -np.log(scale) - t / scale,
    cdf=lambda t, scale: 1 - np.exp(-t / scale),
    sf=lambda t, scale: np.exp(-t / scale),
    sample=lambda rng, size, scale: rng.exponential(scale, size=size),
    closed_form_mle=lambda times: np.array([np.mean(times)]),
)


# --------------------------------------------------------------------------
# Weibull: f(t) = (k/s) z^(k-1) exp(-z^k),  z = t/s
# k < 1 means clustering (many short and a few very long waiting times).
# --------------------------------------------------------------------------
def _weibull_logpdf(t, shape, scale):
    z = t / scale
    return np.log(shape) - np.log(scale) + (shape - 1) * np.log(z) - z**shape


WEIBULL = Distribution(
    name="Weibull",
    param_names=("shape", "scale"),
    logpdf=_weibull_logpdf,
    cdf=lambda t, shape, scale: 1 - np.exp(-((t / scale) ** shape)),
    sf=lambda t, shape, scale: np.exp(-((t / scale) ** shape)),
    sample=lambda rng, size, shape, scale: scale * rng.weibull(shape, size=size),
)


# --------------------------------------------------------------------------
# Gamma: f(t) = z^(k-1) exp(-z) / (Gamma(k) s),  z = t/s
# k = 1 recovers the exponential; k < 1 again means clustering.
# --------------------------------------------------------------------------
def _gamma_logpdf(t, shape, scale):
    z = t / scale
    return (shape - 1) * np.log(z) - z - gammaln(shape) - np.log(scale)


GAMMA = Distribution(
    name="Gamma",
    param_names=("shape", "scale"),
    logpdf=_gamma_logpdf,
    cdf=lambda t, shape, scale: gammainc(shape, t / scale),
    sf=lambda t, shape, scale: gammaincc(shape, t / scale),
    sample=lambda rng, size, shape, scale: rng.gamma(shape, scale, size=size),
)


DISTRIBUTIONS = {
    "weibull": WEIBULL,
    "gamma": GAMMA,
    "exponential": EXPONENTIAL,
}
