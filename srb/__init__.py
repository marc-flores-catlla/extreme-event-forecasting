"""
srb: statistical analysis of waiting times between solar radio bursts.

Modules
-------
config         shared settings (paths, minimum separation, threshold grid, seed)
data           load the catalogue and build inter-event times per threshold
distributions  candidate distributions (exponential, Weibull, gamma)
fitting        maximum-likelihood fits, per threshold and across thresholds
diagnostics    autocorrelation, Lilliefors/KS test, bootstrap summaries
wald           Wald test for a constant shape across thresholds
forecasting    probability of an extreme burst within a horizon
plotting       figures used by the notebooks
"""
